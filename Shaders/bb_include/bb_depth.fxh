/*----------------------------------------------|
| ::        bb Depth & View Space            :: |
|----------------------------------------------*/

#pragma once

#include "bb_common.fxh"

#ifndef RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN
    #define RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN 0
#endif
#ifndef RESHADE_DEPTH_INPUT_IS_REVERSED
    #define RESHADE_DEPTH_INPUT_IS_REVERSED 1
#endif
#ifndef RESHADE_DEPTH_INPUT_IS_MIRRORED
    #define RESHADE_DEPTH_INPUT_IS_MIRRORED 0
#endif
#ifndef RESHADE_DEPTH_INPUT_IS_LOGARITHMIC
    #define RESHADE_DEPTH_INPUT_IS_LOGARITHMIC 0
#endif
#ifndef RESHADE_DEPTH_MULTIPLIER
    #define RESHADE_DEPTH_MULTIPLIER 1
#endif
#ifndef RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
    #define RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 1000.0
#endif
#ifndef RESHADE_DEPTH_INPUT_Y_SCALE
    #define RESHADE_DEPTH_INPUT_Y_SCALE 1
#endif
#ifndef RESHADE_DEPTH_INPUT_X_SCALE
    #define RESHADE_DEPTH_INPUT_X_SCALE 1
#endif
#ifndef RESHADE_DEPTH_INPUT_Y_OFFSET
    #define RESHADE_DEPTH_INPUT_Y_OFFSET 0
#endif
#ifndef RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET
    #define RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET 0
#endif
#ifndef RESHADE_DEPTH_INPUT_X_OFFSET
    #define RESHADE_DEPTH_INPUT_X_OFFSET 0
#endif
#ifndef RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET
    #define RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET 0
#endif

// Depth Functions
float GetDepth(float2 xy)
{
    // Clamp to prevent reading outside screen bounds which can wrap around
    xy = clamp(xy, 0.0, 1.0);
    return bb::GetLinearizedDepth(xy);
}

float GetRawDepth(float2 xy)
{
    xy = clamp(xy, 0.0, 1.0);
    return bb::GetRawDepth(xy);
}

float LinearizeDepth(float depth)
{
    return bb::LinearizeDepth(depth);
}

float GetDepthDerivative(float2 xy)
{
    return fwidth(GetDepth(xy));
}

bool IsSky(float depth)
{
    return depth >= 0.9999;
}

// View Space
float3 UVToViewPos(float2 uv, float view_z, float2 pScale)
{
    float2 ndc = uv * 2.0 - 1.0;
    return float3(ndc.x * pScale.x * view_z, -ndc.y * pScale.y * view_z, view_z);
}

float2 ViewPosToUV(float3 view_pos, float2 pScale)
{
    float z_safe = max(1e-6, view_pos.z);
    float2 ndc = view_pos.xy / (z_safe * pScale);
    return float2(ndc.x, -ndc.y) * 0.5 + 0.5;
}

float3 GetViewPosForNormal(float2 uv, float2 pScale)
{
    return UVToViewPos(uv, GetDepth(uv), pScale);
}

float2 GetProjectionScale(float fovDegrees)
{
    float y = tan(fovDegrees * DEG2RAD * 0.5);
    return float2(y * bb::AspectRatio, y);
}
