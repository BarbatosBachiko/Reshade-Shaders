/*----------------------------------------------|
| ::        bb Normal Functions                :: |
|----------------------------------------------*/

#pragma once

#include "bb_depth.fxh"

// Encode normal into 2 channels (RG)
float2 NM_EncodeNormal(float3 n)
{
    return n.xy * 0.5 + 0.5;
}

// Decode normal from 2 channels (RG) back to float3
float3 NM_DecodeNormal(float2 enc)
{
    float3 n;
    n.xy = enc * 2.0 - 1.0;
    n.z = -sqrt(saturate(1.0 - dot(n.xy, n.xy)));
    return n;
}

float3 NM_CalculateNormal(float2 uv, float2 pScale)
{
    float2 offset = bb::PixelSize;
    float3 pos_c  = GetViewPosForNormal(uv,                              pScale);
    float3 pos_l  = GetViewPosForNormal(uv + float2(-offset.x,  0.0),    pScale);
    float3 pos_r  = GetViewPosForNormal(uv + float2( offset.x,  0.0),    pScale);
    float3 pos_u  = GetViewPosForNormal(uv + float2( 0.0, -offset.y),    pScale);
    float3 pos_d  = GetViewPosForNormal(uv + float2( 0.0,  offset.y),    pScale);
    float3 pos_l2 = GetViewPosForNormal(uv + float2(-2.0*offset.x, 0.0), pScale);
    float3 pos_r2 = GetViewPosForNormal(uv + float2( 2.0*offset.x, 0.0), pScale);
    float3 pos_u2 = GetViewPosForNormal(uv + float2( 0.0, -2.0*offset.y),pScale);
    float3 pos_d2 = GetViewPosForNormal(uv + float2( 0.0,  2.0*offset.y),pScale);
    float3 dl  = pos_c - pos_l;
    float3 dr  = pos_r - pos_c;
    float3 du  = pos_c - pos_u;
    float3 dd  = pos_d - pos_c;
    float3 dl2 = pos_c  - pos_l2;
    float3 dr2 = pos_r2 - pos_c;
    float3 du2 = pos_c  - pos_u2;
    float3 dd2 = pos_d2 - pos_c;

    float dxl = abs(dl.z + (dl.z - dl2.z));
    float dxr = abs(dr.z + (dr.z - dr2.z));
    float dyu = abs(du.z + (du.z - du2.z));
    float dyd = abs(dd.z + (dd.z - dd2.z));

    float3 hor = dxl < dxr ? dl : dr;
    float3 ver = dyu < dyd ? du : dd;

    float3 n = cross(hor, ver);
    n.x = -n.x;
    float lenSq = dot(n, n);
    return (lenSq > 1e-25) ? n * rsqrt(lenSq) : float3(0, 0, -1);
}

// Simple Normal (2 samples)
float3 NM_CalculateNormalSimple(float2 texcoord, float2 pScale)
{
    float3 offset_x = UVToViewPos(texcoord + float2(bb::PixelSize.x, 0.0), GetDepth(texcoord + float2(bb::PixelSize.x, 0.0)), pScale);
    float3 offset_y = UVToViewPos(texcoord + float2(0.0, bb::PixelSize.y), GetDepth(texcoord + float2(0.0, bb::PixelSize.y)), pScale);
    float3 center = UVToViewPos(texcoord, GetDepth(texcoord), pScale);
    return normalize(cross(center - offset_x, center - offset_y));
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

// Smoothed Normal (SmartSurface)
float4 NM_ComputeSmoothedNormal(float2 uv, float2 direction, sampler sInput, int smartSurfaceMode, float smoothThreshold, float farPlane)
{
    float4 color = GetLod(sInput, uv);
    float SNWidth = (smartSurfaceMode == 1) ? 5.5 : ((smartSurfaceMode == 2) ? 2.5 : 1.0);
    int SNSamples = (smartSurfaceMode == 1) ? 1 : ((smartSurfaceMode == 2) ? 3 : 30);
    float2 p = bb::PixelSize * SNWidth * direction;
    float T = rcp(max(smoothThreshold * saturate(2 * (1 - color.a)), 0.0001));
    float4 s1 = 0.0;
    float sc = 0.0;
    
    [loop]
    for (int x = -SNSamples; x <= SNSamples; x++)
    {
        float4 s = GetLod(sInput, uv + (p * x));
        float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (farPlane * smoothThreshold);
        diff = 1 - saturate(diff * T);
        s1 += s * diff;
        sc += diff;
    }
    return (sc > 0.0001) ? (s1 / sc) : color;
}

float4 NM_GenerateNormalWithBump(float2 viewUV, float2 pScale, float3 bumpNormal, float bumpAmount)
{
    float depth = GetDepth(viewUV);
    if (depth >= 0.999)
        return float4(0.5, 0.5, 0.0, 1.0);

    float3 geomNormal = NM_CalculateNormal(viewUV, pScale);
    
    if (bumpAmount > 0.001)
    {
        float3 blended = NM_BlendBump(geomNormal, bumpNormal);
        geomNormal = normalize(lerp(geomNormal, blended, bumpAmount));
    }
    
    return float4(NM_EncodeNormal(geomNormal), 0.0, depth);
}

float4 NM_GenerateNormal(float2 viewUV, float2 pScale)
{
    float depth = GetDepth(viewUV);
    if (depth >= 0.999)
        return float4(NM_EncodeNormal(float3(0, 0, -1)), 0.0, depth);
    
    float3 normal = NM_CalculateNormal(viewUV, pScale);
    return float4(NM_EncodeNormal(normal), 0.0, depth);
}
