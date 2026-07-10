/*----------------------------------------------|
| ::        BaBa Noise Functions             :: |
|----------------------------------------------*/

#pragma once

#include "bb_common.fxh"

// Blue Noise Sampling — single channel
float N_GetSpatialTemporalNoise(float2 pos, sampler sTexBlueNoise, int frameCount)
{
    float2 bn_uv = pos / 1024.0;
    float frame = fmod((float)frameCount, 64.0);
    bn_uv += float2(0.61803398875, 0.73205080757) * frame;
    return tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0)).r;
}

// Blue Noise Sampling — 4 channels
float4 N_GetSpatialTemporalNoise4(float2 pos, sampler sTexBlueNoise, int frameCount)
{
    float2 bn_uv = pos / 1024.0;
    float frame = fmod((float)frameCount, 64.0);
    bn_uv += float2(0.61803398875, 0.73205080757) * frame;
    return tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0));
}

// Concentric Square Mapping
float2 N_ConcentricSquareMapping(float2 u)
{
    float2 ab = 2.0 * u - 1.0;
    float2 ab2 = ab * ab;
    float r, phi;
    if (ab2.x > ab2.y)
    {
        r = ab.x;
        phi = (PI / 4.0) * (ab.y / ab.x);
    }
    else
    {
        r = ab.y;
        phi = (ab.y != 0.0) ? (PI / 2.0) - (PI / 4.0) * (ab.x / ab.y) : 0.0;
    }
    
    float2 sincosPhi;
    sincos(phi, sincosPhi.y, sincosPhi.x);
    return r * sincosPhi;
}

// Noise Sequences
float N_GoldenSequence(uint i)
{
    return float(2654435769u * i) / 4294967296.0;
}

float2 N_PlasticSequence(uint i)
{
    return float2(3242174889u * i, 2447445414u * i) / 4294967296.0;
}

float3 N_Sequence3D(uint i)
{
    return float3(N_PlasticSequence(i), N_GoldenSequence(i));
}

float3 N_ToroidalJitter(float3 x, float3 jitter)
{
    return 2.0 * abs(frac(x + jitter) - 0.5);
}

float N_GetSpatialNoise(float2 pos)
{
    return frac(52.9829189 * frac(0.06711056 * pos.x + 0.00583715 * pos.y));
}
