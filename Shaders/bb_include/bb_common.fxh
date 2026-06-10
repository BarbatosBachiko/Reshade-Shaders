/*----------------------------------------------|
| ::            BaBa Common Utilities        :: |
|----------------------------------------------*/

#pragma once

// Universal Macros
#define GetColor(c) tex2Dlod(bb::BackBuffer, float4((c).xy, 0.0, 0.0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define fmod(x, y) (frac((x)*rcp(y)) * (y))



// Constants
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
static const float DEG2RAD = 0.017453292;

// Frame Count
uniform int FRAME_COUNT < source = "framecount"; >;

float ComputeDepthWeight(float depth, float sigma)
{
    return 1.0 / max(sigma * depth, 1e-6);
}

float ComputeEdgeMask(float depth, float threshold)
{
    float depthDerivative = fwidth(depth);
    return 1.0 - smoothstep(0.0, threshold, depthDerivative);
}

#ifndef USE_HALF
    #define USE_HALF 0
#endif

#if USE_HALF
    #define hfloat  min16float
    #define hfloat2 min16float2
    #define hfloat3 min16float3
    #define hfloat4 min16float4
#else
    #define hfloat  float
    #define hfloat2 float2
    #define hfloat3 float3
    #define hfloat4 float4
#endif
