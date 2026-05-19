/*------------------------------------------|
| ::   BaBa Temporal Anti-Aliasing (TAA) :: |
|------------------------------------------*/

#pragma once

#include "bb_colorspace.fxh"
#include "bb_common.fxh"
#include "bb_normal.fxh"

// TAA Compress / Resolve 
float3 TAA_Compress(float3 color)
{
    return color / (1.0 + color);
}

float3 TAA_Resolve(float3 color)
{
    return color / max(1e-6, 1.0 - color);
}

float4 TAA_Compress4(float4 color)
{
    float luma = max(color.r, max(color.g, color.b));
    return float4(color.rgb / (1.0 + luma), color.a);
}

float4 TAA_Resolve4(float4 color)
{
    float luma = max(color.r, max(color.g, color.b));
    return float4(color.rgb / max(1e-6, 1.0 - luma), color.a);
}

// Clip to AABB 
float3 TAA_ClipToAABB(float3 aabb_min, float3 aabb_max, float3 history_sample)
{
    float3 p_clip = 0.5 * (aabb_max + aabb_min);
    float3 e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
    float3 v_clip = history_sample - p_clip;
    float3 v_unit = v_clip / e_clip;
    float3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    
    return (ma_unit > 1.0) ? (p_clip + v_clip / ma_unit) : history_sample;
}

// Neighborhood Variance
void TAA_ComputeNeighborhoodVariance(sampler sInput, float2 texcoord, float4 current_raw, float2 pSize, out float4 color_min, out float4 color_max)
{
    float current_luma = max(current_raw.r, max(current_raw.g, current_raw.b));
    float current_w = 1.0 / (1.0 + current_luma);
    
    float4 current_c;
    current_c.rgb = RGBToYCoCg(TAA_Compress4(current_raw).rgb);
    current_c.a = current_raw.a;

    float4 m1 = current_c * current_w;
    float4 m2 = (current_c * current_c) * current_w;
    float weightSum = current_w;

    [unroll]
    for (int x = -1; x <= 1; x++)
    {
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            if (x == 0 && y == 0) continue;
            
            float4 c_raw = GetLod(sInput, texcoord + float2(x, y) * pSize);
            
            float luma = max(c_raw.r, max(c_raw.g, c_raw.b));
            float w = 1.0 / (1.0 + luma);
            
            float4 c;
            c.rgb = RGBToYCoCg(TAA_Compress4(c_raw).rgb);
            c.a = c_raw.a;
            
            m1 += c * w;
            m2 += c * c * w;
            weightSum += w;
        }
    }
    
    m1 /= weightSum;
    m2 /= weightSum;
    
    float4 sigma = sqrt(abs(m2 - m1 * m1));
    float gamma = 1.25;
    
    color_min = m1 - gamma * sigma;
    color_max = m1 + gamma * sigma;
}

// Catmull-Rom History Sampling
float4 TAA_SampleHistoryCatmullRom(sampler sInput, float2 uv, float2 texSize)
{
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - texPos1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    float2 texPos0 = texPos1 - 1.0;
    float2 texPos3 = texPos1 + 2.0;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float4 result = 0.0;
    result += GetLod(sInput, float2(texPos0.x, texPos0.y)) * w0.x * w0.y;
    result += GetLod(sInput, float2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += GetLod(sInput, float2(texPos3.x, texPos0.y)) * w3.x * w0.y;

    result += GetLod(sInput, float2(texPos0.x, texPos12.y)) * w0.x * w12.y;
    result += GetLod(sInput, float2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += GetLod(sInput, float2(texPos3.x, texPos12.y)) * w3.x * w12.y;

    result += GetLod(sInput, float2(texPos0.x, texPos3.y)) * w0.x * w3.y;
    result += GetLod(sInput, float2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += GetLod(sInput, float2(texPos3.x, texPos3.y)) * w3.x * w3.y;

    return max(result, 0.0);
}

// joint Bilateral Upsample
float4 TAA_JointBilateralUpsample(float2 uv, float highDepth, float2 pScale, float renderScale, sampler sLowRes, sampler sNormal)
{
    float2 lowResUV = uv * renderScale;
    float3 highNormal = NM_CalculateNormal(uv, pScale);

    float4 result = GetLod(sLowRes, lowResUV);
    float4 sum = 0.0;
    float sumWeight = 0.0;

    float2 texelSize = bb::PixelSize;
    float2 baseUV = (floor(lowResUV / texelSize) + 0.5) * texelSize;

    float depth_weight_factor = 1.0 / (max(0.1 * highDepth, 1e-6));

    [loop]
    for (int x = -1; x <= 1; x++)
    {
        [loop]
        for (int y = -1; y <= 1; y++)
        {
            float2 sampleUV = baseUV + float2(x, y) * texelSize;
            float4 lowData = GetLod(sLowRes, sampleUV);
            float4 gbuffer = GetLod(sNormal, sampleUV);

            float3 lowNormal = gbuffer.rgb;
            float lowDepth = gbuffer.a;

            float wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
            float dotN = max(0.0, dot(highNormal, lowNormal));
            float dotN2 = dotN * dotN;
            float dotN4 = dotN2 * dotN2;
            float dotN8 = dotN4 * dotN4;
            float wNormal = dotN8 * dotN8;
            float wSpatial = exp(-0.5 * float(x * x + y * y));

            float weight = wDepth * wNormal * wSpatial;
            sum += lowData * weight;
            sumWeight += weight;
        }
    }

    if (sumWeight >= 1e-6)
        result = sum / sumWeight;
    return result;
}

// Joint Bilateral Upsample (RG-encoded normals)
float4 TAA_JointBilateralUpsampleRG(float2 uv, float highDepth, float2 pScale, float renderScale, sampler sLowRes, sampler sNormal)
{
    float2 lowResUV = uv * renderScale;
    float3 highNormal = NM_CalculateNormal(uv, pScale);

    float4 result = GetLod(sLowRes, lowResUV);
    float4 sum = 0.0;
    float sumWeight = 0.0;

    float2 texelSize = bb::PixelSize;
    float2 baseUV = (floor(lowResUV / texelSize) + 0.5) * texelSize;

    float depth_weight_factor = 1.0 / (max(0.1 * highDepth, 1e-6));

    [loop]
    for (int x = -1; x <= 1; x++)
    {
        [loop]
        for (int y = -1; y <= 1; y++)
        {
            float2 sampleUV = baseUV + float2(x, y) * texelSize;
            float4 lowData = GetLod(sLowRes, sampleUV);
            float4 gbuffer = GetLod(sNormal, sampleUV);
            
            float3 lowNormal = NM_DecodeNormal(gbuffer.rg);
            float lowDepth = gbuffer.a;

            float wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
            float dotN = max(0.0, dot(highNormal, lowNormal));
            float dotN2 = dotN * dotN;
            float dotN4 = dotN2 * dotN2;
            float dotN8 = dotN4 * dotN4;
            float wNormal = dotN8 * dotN8;
            float wSpatial = exp(-0.5 * float(x * x + y * y));

            float weight = wDepth * wNormal * wSpatial;
            sum += lowData * weight;
            sumWeight += weight;
        }
    }

    if (sumWeight >= 1e-6)
        result = sum / sumWeight;
    return result;
}
