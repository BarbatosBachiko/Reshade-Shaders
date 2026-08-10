/*----------------------------------------------|
| ::        bb Normal Functions                :: |
|----------------------------------------------*/

#pragma once

#include "bb_depth.fxh"
#include "bb_noise.fxh"

// Never let a zero, NaN or Inf reach normalize/rsqrt.
float3 NM_SafeNormalize(float3 v)
{
    float lenSq = dot(v, v);
    if (any(v != v) || lenSq != lenSq || lenSq <= 1e-12 || lenSq >= 1e12)
        return float3(0.0, 0.0, -1.0);
    return v * rsqrt(lenSq);
}

float NM_SafeDepth(float depth)
{
    if (depth != depth)
        return 1.0;
    return saturate(depth);
}

float4 NM_SanitizeNormalData(float4 data)
{
    return float4(NM_SafeNormalize(data.rgb), NM_SafeDepth(data.a));
}

// Encode normal into 2 channels (RG)
float2 NM_EncodeNormal(float3 n)
{
    n = NM_SafeNormalize(n);
    return n.xy * 0.5 + 0.5;
}

// Decode normal from 2 channels (RG) back to float3
float3 NM_DecodeNormal(float2 enc)
{
    if (any(enc != enc))
        enc = float2(0.5, 0.5);

    float3 n;
    n.xy = saturate(enc) * 2.0 - 1.0;
    n.z = -sqrt(saturate(1.0 - dot(n.xy, n.xy)));
    return NM_SafeNormalize(n);
}

float3 NM_CalculateNormal(float2 uv, float2 pScale)
{
    float2 offset = bb::PixelSize;
    float3 pos_c  = GetViewPosForNormal(uv,                              pScale);
    float3 pos_l  = GetViewPosForNormal(uv + float2(-offset.x,  0.0),    pScale);
    float3 pos_r  = GetViewPosForNormal(uv + float2( offset.x,  0.0),    pScale);
    float3 pos_u  = GetViewPosForNormal(uv + float2( 0.0, -offset.y),    pScale);
    float3 pos_d  = GetViewPosForNormal(uv + float2( 0.0,  offset.y),    pScale);
    // Select the shorter one-sided derivative on each axis.
    float3 dx_left  = pos_c - pos_l;
    float3 dx_right = pos_r - pos_c;
    float3 dy_up    = pos_c - pos_u;
    float3 dy_down  = pos_d - pos_c;

    float3 hor = dot(dx_right, dx_right) < dot(dx_left, dx_left) ? dx_right : dx_left;
    float3 ver = dot(dy_down, dy_down) < dot(dy_up, dy_up) ? dy_down : dy_up;

    float3 n = cross(hor, ver);
    float lenSq = dot(n, n);
    if (any(n != n) || lenSq != lenSq || lenSq <= 1e-25 || lenSq >= 1e12)
        return float3(0, 0, -1);
    n *= rsqrt(lenSq);

    // Correct the hemisphere once here: every consumer expects a camera-facing normal.
    if (dot(n, -pos_c) < 0.0)
        n = -n;
    return n;
}

// Simple Normal (2 samples)
float3 NM_CalculateNormalSimple(float2 texcoord, float2 pScale)
{
    float3 offset_x = UVToViewPos(texcoord + float2(bb::PixelSize.x, 0.0), GetDepth(texcoord + float2(bb::PixelSize.x, 0.0)), pScale);
    float3 offset_y = UVToViewPos(texcoord + float2(0.0, bb::PixelSize.y), GetDepth(texcoord + float2(0.0, bb::PixelSize.y)), pScale);
    float3 center = UVToViewPos(texcoord, GetDepth(texcoord), pScale);
    return NM_SafeNormalize(cross(center - offset_x, center - offset_y));
}

// Slope-adjusted edge weights from 4 neighboring view-space depths.
float4 NM_CalculateDepthEdges(const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ)
{
    float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
    float4 edgesLRTBSlopeAdjusted = edgesLRTB + edgesLRTB.yxwz;
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    return saturate(1.3 - (edgesLRTB * 25.0) / centerZ);
}

// Normal from 4 neighbor view-space positions, weighted by the edge mask above.
float3 NM_CalculateEdgeWeightedNormal(const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
{
    float4 acceptedNormals = float4(edgesLRTB.x * edgesLRTB.z, edgesLRTB.z * edgesLRTB.y, edgesLRTB.y * edgesLRTB.w, edgesLRTB.w * edgesLRTB.x);
    pixLPos = normalize(pixLPos - pixCenterPos);
    pixRPos = normalize(pixRPos - pixCenterPos);
    pixTPos = normalize(pixTPos - pixCenterPos);
    pixBPos = normalize(pixBPos - pixCenterPos);
    float3 pixelNormal = float3(0, 0, -0.0005);
    pixelNormal += (acceptedNormals.x) * cross(pixLPos, pixTPos);
    pixelNormal += (acceptedNormals.y) * cross(pixTPos, pixRPos);
    pixelNormal += (acceptedNormals.z) * cross(pixRPos, pixBPos);
    pixelNormal += (acceptedNormals.w) * cross(pixBPos, pixLPos);
    return normalize(pixelNormal);
}

// Bump Blending
float3 NM_BlendBump(float3 n1, float3 n2)
{
    n1.z++;
    return n1 * dot(n1, n2) / n1.z - n2;
}

float3 NM_BlendNormals(float3 n1, float3 n2)
{
    n1 += float3(0, 0, 1);
    n2 *= float3(-1, -1, 1);
    return n1 * dot(n1, n2) / n1.z - n2;
}

// Textured/bump normal detail from a Scharr edge filter on scene luminance.
// BaBa_SSR keeps its own independent copy as "Micro-Surface Details".
float3 NM_ApplyTextureBump(float3 normal, float2 uv, sampler colorSampler, float2 pixelSize, float intensity, float edgeThreshold)
{
    if (intensity <= 0.0)
        return normal;

    float4 pTL = tex2DgatherG(colorSampler, uv - pixelSize * 0.5);
    float4 pTR = tex2DgatherG(colorSampler, uv + float2(pixelSize.x, -pixelSize.y) * 0.5);
    float4 pBL = tex2DgatherG(colorSampler, uv + float2(-pixelSize.x, pixelSize.y) * 0.5);
    float4 pBR = tex2DgatherG(colorSampler, uv + pixelSize * 0.5);

    float Gx = (3.0 * pTR.z + 10.0 * pBR.z + 3.0 * pBR.y) - (3.0 * pTL.w + 10.0 * pTL.x + 3.0 * pBL.x);
    float Gy = (3.0 * pBL.x + 10.0 * pBL.y + 3.0 * pBR.y) - (3.0 * pTL.w + 10.0 * pTL.z + 3.0 * pTR.z);
    Gx *= 0.25;
    Gy *= 0.25;

    if ((Gx * Gx + Gy * Gy) < (edgeThreshold * edgeThreshold))
        return normal;

    float2 slope = float2(Gx, Gy) * intensity;
    float3 up = abs(normal.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, normal));
    float3 B = cross(normal, T);
    return NM_SafeNormalize(normal + T * slope.x - B * slope.y);
}

// Smoothed Normal (SmartSurface)
float4 NM_ComputeSmoothedNormal(float2 uv, float2 direction, sampler sInput, int smartSurfaceMode, float smoothThreshold, float farPlane)
{
    float4 color = NM_SanitizeNormalData(GetLod(sInput, uv));
    float SNWidth = (smartSurfaceMode == 1) ? 5.5 : ((smartSurfaceMode == 2) ? 2.5 : 1.0);
    int SNSamples = (smartSurfaceMode == 1) ? 1 : ((smartSurfaceMode == 2) ? 3 : 30);
    float2 p = bb::PixelSize * SNWidth * direction;
    float T = rcp(max(smoothThreshold * saturate(2 * (1 - color.a)), 0.0001));
    float4 s1 = 0.0;
    float sc = 0.0;
    float4 flatSum = 0.0;
    float sampleCount = 0.0;

    // Jitter the tap comb by up to half a spacing every frame; the Launcher's
    // temporal normal history resolves the result.
    float jitter = N_GetAnimatedSpatialNoise(uv / bb::PixelSize, FRAME_COUNT) - 0.5;

    [loop]
    for (int x = -SNSamples; x <= SNSamples; x++)
    {
        float4 s = NM_SanitizeNormalData(GetLod(sInput, uv + p * (x + jitter)));
        float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (farPlane * smoothThreshold);
        diff = 1 - saturate(diff * T);
        s1 += s * diff;
        sc += diff;
        flatSum += s;
        sampleCount += 1.0;
    }

    float4 bilateral = (sc > 0.0001) ? (s1 / sc) : color;

    // On dense thin geometry (foliage, wires, grating) nearly every neighbor is
    // rejected, so blend toward a plain average instead of leaving noise untouched.
    float coherence = saturate(sc / max(sampleCount, 1.0));
    float chaos = 1.0 - smoothstep(0.12, 0.45, coherence);
    float4 flatAverage = flatSum / max(sampleCount, 1.0);
    float4 result = lerp(bilateral, flatAverage, chaos);
    result.rgb = NM_SafeNormalize(result.rgb);
    return result;
}

// Edge-avoiding A-Trous wavelet pass (Dammertz et al., HPG 2010). B3-spline
// stencil {1/16, 1/4, 3/8, 1/4, 1/16}; stepWidth is the hole spacing, so
// callers can chain passes with growing dilation (1, 2, 4, ...).
float4 NM_ATrousSmoothNormal(float2 uv, float2 direction, sampler sInput, float stepWidth, float smoothThreshold, float farPlane)
{
    float4 color = NM_SanitizeNormalData(GetLod(sInput, uv));
    float T = rcp(max(smoothThreshold * saturate(2 * (1 - color.a)), 0.0001));

    static const float KERNEL[3] = { 0.375, 0.25, 0.0625 };

    // Jitter the hole spacing by up to half a tap every frame.
    float jitter = N_GetAnimatedSpatialNoise(uv / bb::PixelSize, FRAME_COUNT) - 0.5;
    float2 texel = bb::PixelSize * direction * stepWidth;

    float4 s1 = color * KERNEL[0];
    float sc = KERNEL[0];
    float4 flatSum = color * KERNEL[0];

    [unroll]
    for (int i = 1; i <= 2; i++)
    {
        [unroll]
        for (int side = -1; side <= 1; side += 2)
        {
            float2 offset = texel * (i + jitter * 0.5) * side;
            float4 s = NM_SanitizeNormalData(GetLod(sInput, uv + offset));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (farPlane * smoothThreshold);
            float weight = saturate(1.0 - saturate(diff * T)) * KERNEL[i];
            s1 += s * weight;
            sc += weight;
            flatSum += s * KERNEL[i];
        }
    }

    float4 bilateral = (sc > 0.0001) ? (s1 / sc) : color;

    // Same fallback as above for dense thin geometry: blend toward the B3 average.
    float chaos = 1.0 - smoothstep(0.12, 0.45, saturate(sc));
    float4 result = lerp(bilateral, flatSum, chaos);
    result.rgb = NM_SafeNormalize(result.rgb);
    return result;
}

float4 NM_GenerateNormalWithBump(float2 viewUV, float2 pScale, float3 bumpNormal, float bumpAmount)
{
    float depth = NM_SafeDepth(GetDepth(viewUV));
    if (depth >= 0.999)
        return float4(0.5, 0.5, 0.0, 1.0);

    float3 geomNormal = NM_CalculateNormal(viewUV, pScale);
    
    if (bumpAmount > 0.001)
    {
        float3 blended = NM_BlendBump(geomNormal, NM_SafeNormalize(bumpNormal));
        geomNormal = NM_SafeNormalize(lerp(geomNormal, blended, bumpAmount));
    }
    
    return float4(NM_EncodeNormal(geomNormal), 0.0, depth);
}

float4 NM_GenerateNormal(float2 viewUV, float2 pScale)
{
    float depth = NM_SafeDepth(GetDepth(viewUV));
    if (depth >= 0.999)
        return float4(NM_EncodeNormal(float3(0, 0, -1)), 0.0, depth);
    
    float3 normal = NM_CalculateNormal(viewUV, pScale);
    return float4(NM_EncodeNormal(normal), 0.0, depth);
}
