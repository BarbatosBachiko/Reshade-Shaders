/*----------------------------------------------|
| ::       BaBa Launcher Sampling             :: |
|----------------------------------------------*/

#pragma once

#include "bb_launcher_resources.fxh"

#if _BABA_USE_LAUNCHER
float BaBa_Launcher_SampleDepth(float2 uv)
{
    return tex2Dlod(BaBa_Launcher::sLinearDepth, float4(clamp(uv, 0.0, 1.0), 0.0, 0.0)).r;
}

float BaBa_Launcher_SampleViewDepth(float2 uv)
{
    return tex2Dlod(BaBa_Launcher::sViewDepth, float4(clamp(uv, 0.0, 1.0), 0.0, 0.0)).r;
}

float3 BaBa_Launcher_SampleNormal(float2 uv)
{
    float3 normal = BaBa_Launcher_SampleNormalData(uv).rgb;
    return NM_SafeNormalize(normal);
}

float2 BaBa_Launcher_SampleMotion(float2 uv)
{
    return tex2Dlod(BaBa_Launcher::sMotionVectors, float4(clamp(uv, 0.0, 1.0), 0.0, 0.0)).rg;
}

float BaBa_Launcher_SampleMotionConfidence(float2 uv)
{
    return tex2Dlod(BaBa_Launcher::sMotionConfidence, float4(clamp(uv, 0.0, 1.0), 0.0, 0.0)).r;
}
#endif
