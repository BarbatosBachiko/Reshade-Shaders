///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion",
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
//
// Implementation: Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>                 (\_/)
// Version:        (see XeGTAO.h)                                                                                    (='.'=)
// Details:        https://github.com/GameTechDev/XeGTAO                                                             (")_(")
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*------------------.
| :: Description :: |
'-------------------/

    XeoGTAO Lite
    Version 0.9
    Author: Barbatos

    About: A performance-focused version of XeGTAO.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 0.9
    !Need minimize noise
*/

#include "ReShade.fxh"

// Motion Vector source selection
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Type definitions for performance
#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4
#define half3x3 float3x3

// Sampler states
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

// Helper macros
#define getDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define PI 3.1415926535
#define PI_HALF 1.57079632679
#define MAX_Frames 64

/*------------------.
| :: UI Controls :: |
'------------------*/

uniform float Intensity <
    ui_category = "General";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float EffectRadius <
    ui_category = "GTAO";
    ui_label = "Effect Radius";
    ui_tooltip = "Main radius of the AO effect in view-space units.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.01;
> = 1.0;

uniform float RadiusMultiplier <
    ui_category = "GTAO";
    ui_label = "Radius Multiplier";
    ui_tooltip = "Additional radius multiplier, useful for scaling the effect without changing the base radius.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float FinalValuePower <
    ui_category = "GTAO";
    ui_label = "Occlusion Power";
    ui_tooltip = "Final occlusion modifier. Higher values make the occlusion stronger.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.1;
> = 1.5;

uniform float FalloffRange <
    ui_category = "GTAO";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge, as a percentage of the radius.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.6;

uniform bool EnableTemporal <
    ui_category = "Temporal";
    ui_type = "checkbox";
    ui_label = "Enable Temporal Filter";
> = true;

uniform float AccumFrames <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "Temporal Accumulation";
    ui_tooltip = "Number of frames to accumulate. Higher values are smoother but can cause more ghosting.";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 4.0;

uniform float DepthMultiplier <
    ui_category = "Depth/Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

uniform float DepthThreshold <
    ui_category = "Depth/Normals";
    ui_label = "Sky Threshold";
    ui_tooltip = "Sets the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.999;

uniform float4 OcclusionColor <
    ui_category = "Extra";
    ui_type = "color";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform int ViewMode <
    ui_category = "Debug";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "None\0AO Only\0Normals\0Depth (View-Space)\0";
> = 0;

uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 90.0;

uniform int FRAME_COUNT < source = "framecount"; >;
uniform float frametime < source = "frametime"; >;

/*----------------.
| :: Textures :: |
'----------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).rg;
    }
#else
texture texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler sTexMotionVectorsSampler
{
    Texture = texMotionVectors;S_PC
};
float2 SampleMotionVectors(float2 texcoord)
{
    return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).rg;
}
#endif

namespace XeGTAO_LITE
{
    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = normalTex;S_LC
    };

    texture ViewDepthTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
    };
    sampler sViewDepthTex
    {
        Texture = ViewDepthTex;S_LC
    };

    texture AOTermTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sAOTermTex
    {
        Texture = AOTermTex;S_LC
    };

    texture TempTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sTempTex
    {
        Texture = TempTex;S_LC
    };

    texture HistoryTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sHistoryTex
    {
        Texture = HistoryTex;S_LC
    };

/*----------------.
| :: Functions :: |
'----------------*/

    float3 RGBToYCoCg(float3 rgb)
    {
        float Y = dot(rgb, float3(0.25, 0.5, 0.25));
        float Co = dot(rgb, float3(0.5, 0, -0.5));
        float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
        return float3(Y, Co, Cg);
    }

    float3 YCoCgToRGB(float3 ycocg)
    {
        float Y = ycocg.x;
        float Co = ycocg.y;
        float Cg = ycocg.z;
        float r = Y + Co - Cg;
        float g = Y + Cg;
        float b = Y - Co - Cg;
        return float3(r, g, b);
    }

    float3 rand3d(float3 p)
    {
        return frac(sin(dot(p, float3(12.9898, 78.233, 151.7182))) * float3(43758.5453, 21783.13224, 9821.42631));
    }

    // http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
    half XeGTAO_FastSqrt(float x)
    {
        return (half) (asfloat(0x1fbd1df5 + (asint(x) >> 1)));
    }

    // input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
    half XeGTAO_FastACos(half inX)
    {
        half x = abs(inX);
        half res = -0.156583 * x + PI_HALF;
        res *= XeGTAO_FastSqrt(1.0 - x);
        return (inX >= 0) ? res : PI - res;
    }

    float3 GetViewPos(float2 uv, float linear_depth)
    {
        float view_z = linear_depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
        float fov_rad = FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / BUFFER_ASPECT_RATIO;
        float2 clip_pos = uv * 2.0 - 1.0;
        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z;
        view_pos.z = view_z;
        return view_pos * DepthMultiplier;
    }

    float3 computeNormal(float2 texcoord)
    {
        const float2 p = ReShade::PixelSize;
        float3 center_pos = GetViewPos(texcoord, getDepth(texcoord));
        float3 ddx = GetViewPos(texcoord + float2(p.x, 0), getDepth(texcoord + float2(p.x, 0))) - center_pos;
        float3 ddy = GetViewPos(texcoord + float2(0, p.y), getDepth(texcoord + float2(0, p.y))) - center_pos;
        return normalize(cross(ddx, ddy));
    }

/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/

    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = computeNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float PS_ViewDepth(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return getDepth(uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }

    float4 PS_GTAO_Main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        const int sliceCount = 2;
        const int stepsPerSlice = 3;

        float2 screen_uv = uv;
        
        float viewspaceZ_raw = tex2Dlod(sViewDepthTex, float4(uv, 0, 0)).r;
        float3 pixCenterPos = GetViewPos(screen_uv, viewspaceZ_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
        float viewspaceZ = pixCenterPos.z;

        if (getDepth(screen_uv) >= DepthThreshold)
        {
            return float4(1.0, 1.0, 1.0, 1.0);
        }

        half3 viewspaceNormal = tex2Dlod(sNormal, float4(uv, 0, 0)).xyz * 2.0 - 1.0;
        const half3 viewVec = (half3) normalize(-pixCenterPos);
        
        const half effectRadius = (half) EffectRadius * (half) RadiusMultiplier;
        const half falloffRange = (half) FalloffRange * effectRadius;
        const half falloffFrom = effectRadius * (1.0 - (half) FalloffRange);
        const half falloffMul = -1.0 / falloffRange;
        const half falloffAdd = falloffFrom / falloffRange + 1.0;

        half visibility = 0;

        const half2 localNoise = rand3d(float3(screen_uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT) / 64.0, frametime * 0.001)).xy;
        const half noiseSlice = localNoise.x;
        const half noiseSample = localNoise.y;

        const float fov_rad = FOV * (PI / 180.0);
        const float pixelViewspaceSize = 2.0 * viewspaceZ * tan(fov_rad * 0.5) / BUFFER_HEIGHT;
        half screenspaceRadius = effectRadius / (half) pixelViewspaceSize;
        const half minS = 1.3 / screenspaceRadius;

        [loop]
        for (half slice = 0; slice < sliceCount; slice++)
        {
            half sliceK = (slice + noiseSlice) / sliceCount;
            half phi = sliceK * PI;
            half cosPhi, sinPhi;
            sincos(phi, sinPhi, cosPhi);
            half2 omega = half2(cosPhi, -sinPhi) * screenspaceRadius;

            const half3 directionVec = half3(cosPhi, sinPhi, 0);
            const half3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            const half3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            half3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

            half signNorm = (half) sign(dot(orthoDirectionVec, projectedNormalVec));
            half projectedNormalVecLength = length(projectedNormalVec);
            half cosNorm = saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);
            half n = signNorm * XeGTAO_FastACos(cosNorm);

            const half lowHorizonCos0 = cos(n + PI_HALF);
            const half lowHorizonCos1 = cos(n - PI_HALF);

            half horizonCos0 = lowHorizonCos0;
            half horizonCos1 = lowHorizonCos1;

            [unroll]
            for (half step = 0; step < stepsPerSlice; step++)
            {
                const half stepBaseNoise = half(slice + step * stepsPerSlice) * 0.61803398875;
                half stepNoise = frac(noiseSample + stepBaseNoise);
                half s = (step + stepNoise) / stepsPerSlice;
                s = pow(s, 2.0);
                s += minS;
                
                half2 sampleOffset = s * omega * ReShade::PixelSize;

                float2 sampleCoord0 = uv + sampleOffset;
                float SZ0_raw = tex2Dlod(sViewDepthTex, float4(sampleCoord0, 0, 0)).r;
                float3 samplePos0 = GetViewPos(sampleCoord0, SZ0_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float2 sampleCoord1 = uv - sampleOffset;
                float SZ1_raw = tex2Dlod(sViewDepthTex, float4(sampleCoord1, 0, 0)).r;
                float3 samplePos1 = GetViewPos(sampleCoord1, SZ1_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float3 sampleDelta0 = (samplePos0 - pixCenterPos);
                float3 sampleDelta1 = (samplePos1 - pixCenterPos);
                half sampleDist0 = (half) length(sampleDelta0);
                half sampleDist1 = (half) length(sampleDelta1);

                half3 sampleHorizonVec0 = (half3) (sampleDelta0 / max(sampleDist0, 1e-6));
                half3 sampleHorizonVec1 = (half3) (sampleDelta1 / max(sampleDist1, 1e-6));
                
                half weight0 = saturate((half) length(sampleDelta0) * falloffMul + falloffAdd);
                half weight1 = saturate((half) length(sampleDelta1) * falloffMul + falloffAdd);

                half shc0 = dot(sampleHorizonVec0, viewVec);
                half shc1 = dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            projectedNormalVecLength = lerp(projectedNormalVecLength, 1, 0.05);

            half h0 = -XeGTAO_FastACos(horizonCos1);
            half h1 = XeGTAO_FastACos(horizonCos0);
            half iarc0 = (cosNorm + 2 * h0 * sin(n) - cos(2 * h0 - n)) / 4;
            half iarc1 = (cosNorm + 2 * h1 * sin(n) - cos(2 * h1 - n)) / 4;
            half localVisibility = projectedNormalVecLength * (iarc0 + iarc1);
            visibility += localVisibility;
        }

        visibility /= sliceCount;
        visibility = pow(saturate(visibility), (half) FinalValuePower);
        visibility = max(0.03, visibility);

        return float4(visibility.xxx, 1.0);
    }

    float4 PS_ApplyTemporalFilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 currentSpec = tex2D(sAOTermTex, uv).rgb;
        float currentDepth = getDepth(uv);

        float2 motion = SampleMotionVectors(uv);
        float2 reprojectedUV = uv + motion;

        float historyDepth = getDepth(reprojectedUV);
        bool validHistory = all(saturate(reprojectedUV) == reprojectedUV) &&
                              FRAME_COUNT > 1 &&
                              abs(historyDepth - currentDepth) < 0.01;

        float3 blendedSpec = currentSpec;

        if (EnableTemporal && validHistory)
        {
            float3 minBox = RGBToYCoCg(currentSpec);
            float3 maxBox = minBox;

            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    float3 neighborSpec = RGBToYCoCg(tex2Dlod(sAOTermTex, float4(uv + float2(x, y) * ReShade::PixelSize, 0, 0)).rgb);
                    minBox = min(minBox, neighborSpec);
                    maxBox = max(maxBox, neighborSpec);
                }
            }
        
            float3 historySpec = RGBToYCoCg(tex2Dlod(sHistoryTex, float4(reprojectedUV, 0, 0)).rgb);
            float3 clampedHistorySpec = clamp(historySpec, minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, AccumFrames);
            blendedSpec = YCoCgToRGB(lerp(clampedHistorySpec, RGBToYCoCg(currentSpec), alpha));
        }

        float historyLengthPacked = validHistory ? min(FRAME_COUNT, MAX_Frames) : 0;
        return float4(blendedSpec, historyLengthPacked);
    }

    float4 PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float occlusion = EnableTemporal
            ? tex2Dlod(sTempTex, float4(uv, 0, 0)).r 
            : tex2Dlod(sAOTermTex, float4(uv, 0, 0)).r;
        return float4(occlusion.xxx, 1.0);
    }

    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);

        if (ViewMode == 0) // Normal
        {
            if (getDepth(uv) >= DepthThreshold)
                return originalColor;

            float visibility = EnableTemporal
                ? tex2D(sTempTex, uv).r
                : tex2D(sAOTermTex, uv).r;
            
            float occlusion = 1.0 - visibility;
            occlusion = saturate(occlusion * Intensity);

            float fade = saturate(1.0 - smoothstep(0.95, 1.0, getDepth(uv)));
            occlusion *= fade;

            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);
            return originalColor;
        }
        else if (ViewMode == 1) // AO Only
        {
            float visibility = EnableTemporal
                ? tex2D(sTempTex, uv).r
                : tex2D(sAOTermTex, uv).r;

            return float4(visibility.xxx, 1.0);
        }
        else if (ViewMode == 2) // Normals
        {
            return float4(tex2D(sNormal, uv).rgb, 1.0);
        }
        else if (ViewMode == 3) // View-Space Depth
        {
            float depth = tex2D(sViewDepthTex, uv).r / (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * DepthMultiplier);
            return float4(saturate(depth.rrr), 1.0);
        }

        return originalColor;
    }

    technique XeGTAO_Lite < ui_tooltip = "A performance-focused version of XeGTAO"; >
    {
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
        }
        pass ViewDepthPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ViewDepth;
            RenderTarget = ViewDepthTex;
        }
        pass GTAOMainPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GTAO_Main;
            RenderTarget = AOTermTex;
        }
        pass TemporalFilterPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ApplyTemporalFilter;
            RenderTarget = TempTex;
        }
        pass HistoryUpdatePass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HistoryTex;
        }
        pass OutputPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
