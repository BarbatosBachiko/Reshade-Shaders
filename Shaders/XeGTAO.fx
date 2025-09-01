/*
    FSR code is derived from the FidelityFX SDK, provided under the MIT license.
    Copyright (C) 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion",
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
//
// Implementation: Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>               (\_/)
// Version:        (see XeGTAO.h)                                                                                  (='.'=)
// Details:        https://github.com/GameTechDev/XeGTAO                                                           (")_(")
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmoothNormals from https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/smoothNormals.fx
// Atrous from https://github.com/yplebedev/BFBSOUPCAN/blob/main/reshade-shaders/Shaders/SoupCanSmooth.fx
/*------------------.
| :: Description :: |
'-------------------/

    XeGTAO
    Version 1.2
    Author: Barbatos

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
*/

#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4
#define half3x3 float3x3

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

#define getDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define PI 3.1415926535
#define PI_HALF 1.57079632679
#define fmod(x, y) (frac((x)*rcp(y)) * (y))
/*---------.
| :: UI :: |
'---------*/

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if UI_DIFFICULTY == 0
#define EnableDenoise 1
#define p_phi 0.1
#define c_phi 1.0
#define n_phi 1.0
#define RadiusMultiplier         1.0
#define FinalValuePower      0.8
#define FalloffRange             0.6 
#define SampleDistributionPower  8.0
#define ThinOccluderCompensation -0.60
#define ComputeBentNormals       false
#define EnableTemporal           true
#define DepthMultiplier          1.0
#define DepthThreshold           0.999
#define bSmoothNormals           false
#define FOV                      35.0
#endif


uniform float Intensity <
    ui_category = "General";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float RenderScale <
    ui_category = "General";
    ui_label = "Render Scale";
    ui_tooltip = "Renders the effect at a lower resolution for performance, then upscales using FSR 1.0.";
    ui_type = "drag";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform int QualityLevel <
    ui_category = "General";
    ui_label = "Quality Level";
    ui_tooltip = "Defines the number of directions and samples per direction.";
    ui_type = "combo";
    ui_items = "Low (2 directions, 2 samples)\0Medium (3 directions, 4 samples)\0High (4 directions, 8 samples)\0Ultra (8 directions, 8 samples)\0";
> = 2;

uniform float EffectRadius <
    ui_category = "General";
    ui_label = "Effect Radius";
    ui_tooltip = "Main radius of the AO effect in view-space units.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 50.0; ui_step = 0.01;
> = 20.0;

uniform float4 OcclusionColor <
    ui_category = "General";
    ui_type = "color";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform float TemporalAccumulationFrames <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 32.0;
    ui_step = 1.0;
    ui_category = "Temporal Filtering";
    ui_label = "Temporal Accumulation Frames";
    ui_tooltip = "Number of frames to accumulate. Higher values are smoother but may cause more ghosting on moving objects.";
> = 4.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "Normal\0Normals\0View-Space Depth\0Raw AO\0Denoised AO\0Upscaled AO\0";
    ui_tooltip = "Selects the debug view mode.";
> = 0;

#if UI_DIFFICULTY == 1
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
> = 0.8;

uniform float FalloffRange <
    ui_category = "GTAO";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge, as a percentage of the radius.";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
> = 0.6;

uniform float SampleDistributionPower <
    ui_category = "GTAO Quality";
    ui_label = "Sample Distribution Power";
    ui_tooltip = "Controls how samples are distributed along the view ray. >1 pushes samples further away.";
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
> = 8.0;

uniform float ThinOccluderCompensation <
    ui_category = "GTAO Quality";
    ui_label = "Thin Occluder Compensation";
    ui_tooltip = "Reduces self-occlusion on thin objects.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = -0.6;

uniform bool ComputeBentNormals <
    ui_category = "GTAO Quality";
    ui_label = "Compute Bent Normals";
> = false;

uniform bool EnableTemporal <
    ui_category = "Temporal Filtering";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Blends the current frame's AO with previous frames to reduce noise and flickering.";
> = true;

uniform bool EnableDenoise <
    ui_category = "Denoiser";
    ui_type = "checkbox";
    ui_label = "Enable A-Trous Denoiser";
    ui_tooltip = "Apply a spatial denoiser after the temporal filter to further smooth the result.";
> = true;

uniform float c_phi <
    ui_category = "Denoiser";
    ui_type = "slider";
    ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
    ui_label = "AO Sigma";
    ui_tooltip = "Controls the sensitivity to AO value differences.";
> = 0.1;

uniform float n_phi <
    ui_category = "Denoiser";
    ui_type = "slider";
    ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
    ui_label = "Normals Sigma";
    ui_tooltip = "Controls the sensitivity to surface normal differences.";
> = 1.0;

uniform float p_phi <
    ui_category = "Denoiser";
    ui_type = "slider";
    ui_min = 0.01; ui_max = 10.0; ui_step = 0.01;
    ui_label = "Position (Depth) Sigma";
    ui_tooltip = "Controls the sensitivity to view space position differences.";
> = 1.0;

uniform float DepthMultiplier <
    ui_category = "Advanced";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
    ui_type = "drag";
> = 1.0;

uniform float DepthThreshold <
    ui_category = "Advanced";
    ui_label = "Sky Threshold";
    ui_tooltip = "Sets the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_type = "drag";
> = 0.999;

uniform bool bSmoothNormals <
    ui_category = "Advanced";
    ui_label = "Smooth Normals";
> = false;

uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 35.0;

#endif // UI_DIFFICULTY == 1

uniform float frametime < source = "frametime"; >;
uniform int FRAME_COUNT < source = "framecount"; >;

/*----------------.
| :: Textures :: |
'----------------*/

// Motion Vectors Texture
#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2D(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2D(sMotVectTexVort, texcoord).rg;
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
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif


namespace XeGTAO
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

    texture AccumulatedAOTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sAccumulatedAOTex
    {
        Texture = AccumulatedAOTex;S_LC
    };

    texture HistoryTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
        MipLevels = 1;
    };
    sampler sHistoryTex
    {
        Texture = HistoryTex;S_LC
    };

    // Denoiser Textures
    texture DenoiseInputTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sDenoiseInputTex
    {
        Texture = DenoiseInputTex;S_LC
    };

    texture DenoiseTex0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sDenoiseTex0
    {
        Texture = DenoiseTex0;S_LC
    };

    texture DenoiseTex1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sDenoiseTex1
    {
        Texture = DenoiseTex1;S_LC
    };

    // Upscaling Texture
    texture UpscaledAOTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sUpscaledAOTex
    {
        Texture = UpscaledAOTex;S_LC
    };


/*----------------.
| :: Functions :: |
'----------------*/

    // FSR
    struct RectificationBox
    {
        float boxCenter;
        float boxVec;
        float aabbMin;
        float aabbMax;
        float fBoxCenterWeight;
    };

    void RectificationBoxAddSample(inout RectificationBox box, bool bInitialSample, float fSample, float fWeight)
    {
        if (bInitialSample)
        {
            box.boxCenter = fSample * fWeight;
            box.boxVec = fSample * fSample * fWeight;
            box.aabbMin = fSample;
            box.aabbMax = fSample;
            box.fBoxCenterWeight = fWeight;
        }
        else
        {
            box.boxCenter += fSample * fWeight;
            box.boxVec += fSample * fSample * fWeight;
            box.aabbMin = min(box.aabbMin, fSample);
            box.aabbMax = max(box.aabbMax, fSample);
            box.fBoxCenterWeight += fWeight;
        }
    }

    void RectificationBoxComputeVarianceBoxData(inout RectificationBox box)
    {
        const float fBoxCenterWeightRcp = rcp(max(1e-6, box.fBoxCenterWeight));
        box.boxCenter *= fBoxCenterWeightRcp;
        box.boxVec = max(0.0, box.boxVec * fBoxCenterWeightRcp - (box.boxCenter * box.boxCenter));
    }


    float xy2hilbert6(float x, float y)
    {
        return x + y * 64.0;
    }

    float2 R2(float n)
    {
        const float a1 = 0.7548776662466927; // ~ 1 / G
        const float a2 = 0.5698402909980532; // ~ 1 / (G^2)
        float2 v = float2(0.5, 0.5) + float2(a1, a2) * (n + 1.0);
        return frac(v);
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
    
    float3 XeGTAO_CalculateNormal(half4 edgesLRTB, float3 CENTER, float3 LEFT, float3 RIGHT, float3 TOP, float3 BOTTOM)
    {
        float3 ddx = RIGHT - LEFT;
        float3 ddy = BOTTOM - TOP;
        return normalize(cross(ddy, ddx));
    }

    float3 XeGTAO_ComputeViewspaceNormal(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
    
        float2 uv_down1 = min(texcoord + float2(0, p.y), 1.0 - p);
        float2 uv_down2 = min(texcoord + float2(0, p.y * 2), 1.0 - p);
        float2 uv_up1 = max(texcoord - float2(0, p.y), p);
        float2 uv_up2 = max(texcoord - float2(0, p.y * 2), p);
        float2 uv_right1 = min(texcoord + float2(p.x, 0), 1.0 - p);
        float2 uv_right2 = min(texcoord + float2(p.x * 2, 0), 1.0 - p);
        float2 uv_left1 = max(texcoord - float2(p.x, 0), p);
        float2 uv_left2 = max(texcoord - float2(p.x * 2, 0), p);
    
        float3 center_pos = GetViewPos(texcoord, getDepth(texcoord));
        float3 pos_down1 = GetViewPos(uv_down1, getDepth(uv_down1));
        float3 pos_down2 = GetViewPos(uv_down2, getDepth(uv_down2));
        float3 extrapolated_down = pos_down1 + (pos_down1 - pos_down2);
        float3 pos_up1 = GetViewPos(uv_up1, getDepth(uv_up1));
        float3 pos_up2 = GetViewPos(uv_up2, getDepth(uv_up2));
        float3 extrapolated_up = pos_up1 + (pos_up1 - pos_up2);
        float3 ddy = pos_down1 - center_pos;
        if (abs(extrapolated_up.z - center_pos.z) < abs(extrapolated_down.z - center_pos.z))
        {
            ddy = center_pos - pos_up1;
        }
        float3 pos_right1 = GetViewPos(uv_right1, getDepth(uv_right1));
        float3 pos_right2 = GetViewPos(uv_right2, getDepth(uv_right2));
        float3 extrapolated_right = pos_right1 + (pos_right1 - pos_right2);
        float3 pos_left1 = GetViewPos(uv_left1, getDepth(uv_left1));
        float3 pos_left2 = GetViewPos(uv_left2, getDepth(uv_left2));
        float3 extrapolated_left = pos_left1 + (pos_left1 - pos_left2);
        float3 ddx = pos_right1 - center_pos;
        if (abs(extrapolated_left.z - center_pos.z) < abs(extrapolated_right.z - center_pos.z))
        {
            ddx = center_pos - pos_left1;
        }
        return normalize(cross(ddy, ddx));
    }
    
    float XeGTAO_EncodeVisibilityBentNormal(half visibility, half3 bentNormal)
    {
        float4 unpackedInput = float4(bentNormal * 0.5 + 0.5, visibility);
        unpackedInput = saturate(unpackedInput) * 255.0;
        return floor(unpackedInput.x) + floor(unpackedInput.y) * 256.0 + floor(unpackedInput.z) * 65536.0 + floor(unpackedInput.w) * 16777216.0;
    }

    void XeGTAO_DecodeVisibilityBentNormal(const float packedValue, out half visibility, out half3 bentNormal)
    {
        float val = packedValue;
        float r = fmod(val, 256.0);
        val = floor(val / 256.0);
        float g = fmod(val, 256.0);
        val = floor(val / 256.0);
        float b = fmod(val, 256.0);
        val = floor(val / 256.0);
        float a = fmod(val, 256.0);
        
        half4 decoded = half4(r, g, b, a) / 255.0;
        bentNormal = decoded.xyz * 2.0 - 1.0;
        visibility = decoded.w;
    }

    float ReconstructUint(float4 packedFloat)
    {
        packedFloat = round(packedFloat * 255.0);
        return packedFloat.x + packedFloat.y * 256.0 + packedFloat.z * 65536.0 + packedFloat.w * 16777216.0;
    }
    
    float4 ReconstructFloat4(float val)
    {
        float r = fmod(val, 256.0);
        val = floor(val / 256.0);
        float g = fmod(val, 256.0);
        val = floor(val / 256.0);
        float b = fmod(val, 256.0);
        val = floor(val / 256.0);
        float a = fmod(val, 256.0);
        return float4(r, g, b, a) / 255.0;
    }

    half3x3 XeGTAO_RotFromToMatrix(half3 from, half3 to)
    {
        const half e = dot(from, to);
        const half f = abs(e);

        if (f > half(1.0 - 0.0003))
            return half3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);

        const half3 v = cross(from, to);
        const half h = (1.0) / (1.0 + e);
        const half hvx = h * v.x;
        const half hvz = h * v.z;
        const half hvxy = hvx * v.y;
        const half hvxz = hvx * v.z;
        const half hvyz = hvz * v.y;

        half3x3 mtx;
        mtx[0][0] = e + hvx * v.x;
        mtx[0][1] = hvxy - v.z;
        mtx[0][2] = hvxz + v.y;
        mtx[1][0] = hvxy + v.z;
        mtx[1][1] = e + h * v.y * v.y;
        mtx[1][2] = hvyz - v.x;
        mtx[2][0] = hvxz - v.y;
        mtx[2][1] = hvyz + v.x;
        mtx[2][2] = e + hvz * v.z;
        return mtx;
    }
    
    static const float2 atrous_offsets[9] =
    {
        float2(-1, -1), float2(0, -1), float2(1, -1),
        float2(-1, 0), float2(0, 0), float2(1, 0),
        float2(-1, 1), float2(0, 1), float2(1, 1)
    };

    float atrous_refactored(sampler input_sampler, float2 texcoord, float level)
    {
        float sum = 0.0;
        float cum_w = 0.0;
        const float2 step_size = ReShade::PixelSize * exp2(level);

        float center_ao = tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0)).r;
        
        float2 center_full_res_uv = texcoord / RenderScale;
        float center_depth = tex2Dlod(sViewDepthTex, float4(center_full_res_uv, 0.0, 0.0)).r;
        
        if (center_depth / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE >= DepthThreshold)
            return center_ao;

        float3 center_normal = tex2Dlod(sNormal, float4(center_full_res_uv, 0.0, 0.0)).xyz * 2.0 - 1.0;
        float3 center_pos = GetViewPos(center_full_res_uv, center_depth / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

        [loop]
        for (int i = 0; i < 9; i++)
        {
            const float2 uv_low = texcoord + atrous_offsets[i] * step_size;

            if (any(uv_low < 0.0) || any(uv_low > RenderScale))
                continue;

            const float sample_ao = tex2Dlod(input_sampler, float4(uv_low, 0.0, 0.0)).r;
            
            const float2 sample_full_res_uv = uv_low / RenderScale;
            const float sample_depth = tex2Dlod(sViewDepthTex, float4(sample_full_res_uv, 0.0, 0.0)).r;
            
            if (sample_depth / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE >= DepthThreshold)
                continue;

            const float3 sample_normal = tex2Dlod(sNormal, float4(sample_full_res_uv, 0.0, 0.0)).xyz * 2.0 - 1.0;
            const float3 sample_pos = GetViewPos(sample_full_res_uv, sample_depth / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
            
            // AO Weight
            float diff_c = center_ao - sample_ao;
            float w_c = exp(-(diff_c * diff_c) / c_phi);

            // Normal Weight
            float diff_n = dot(center_normal, sample_normal);
            float w_n = pow(saturate(diff_n), n_phi);

            // Position Weight
            float diff_p = distance(center_pos, sample_pos);
            float w_p = exp(-(diff_p * diff_p) / p_phi);

            const float weight = w_c * w_n * w_p;

            sum += sample_ao * weight;
            cum_w += weight;
        }

        return cum_w > 1e-6 ? (sum / cum_w) : center_ao;
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }
    float2 ComputeGradient(float2 texcoord)
    {
        const float2 offset = ReShade::PixelSize.xy;

        float3 colorTL = tex2D(ReShade::BackBuffer, texcoord + float2(-offset.x, -offset.y)).rgb;
        float3 colorT = tex2D(ReShade::BackBuffer, texcoord + float2(0.0, -offset.y)).rgb;
        float3 colorTR = tex2D(ReShade::BackBuffer, texcoord + float2(offset.x, -offset.y)).rgb;
        float3 colorL = tex2D(ReShade::BackBuffer, texcoord + float2(-offset.x, 0.0)).rgb;
        float3 colorR = tex2D(ReShade::BackBuffer, texcoord + float2(offset.x, 0.0)).rgb;
        float3 colorBL = tex2D(ReShade::BackBuffer, texcoord + float2(-offset.x, offset.y)).rgb;
        float3 colorB = tex2D(ReShade::BackBuffer, texcoord + float2(0.0, offset.y)).rgb;
        float3 colorBR = tex2D(ReShade::BackBuffer, texcoord + float2(offset.x, offset.y)).rgb;

        float lumTL = GetLuminance(colorTL);
        float lumT = GetLuminance(colorT);
        float lumTR = GetLuminance(colorTR);
        float lumL = GetLuminance(colorL);
        float lumR = GetLuminance(colorR);
        float lumBL = GetLuminance(colorBL);
        float lumB = GetLuminance(colorB);
        float lumBR = GetLuminance(colorBR);

        float gx = (-3.0 * lumTL - 10.0 * lumL - 3.0 * lumBL) + (3.0 * lumTR + 10.0 * lumR + 3.0 * lumBR);
        float gy = (-3.0 * lumTL - 10.0 * lumT - 3.0 * lumTR) + (3.0 * lumBL + 10.0 * lumB + 3.0 * lumBR);

        float2 gradient = float2(gx, gy);

        if (0 > 0.0 && length(gradient) < 0)
        {
            gradient = 0.0;
        }

        return gradient;
    }
    float3 Bump(float2 texcoord, float intensity)
    {
        float2 gradient = ComputeGradient(texcoord);
        float3 bumpNormal = float3(gradient.x * -intensity, gradient.y * -intensity, 1.0);
        return normalize(bumpNormal);
    }
    
/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/
    
    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 geomNormal = -XeGTAO_ComputeViewspaceNormal(uv);
        float3 bumpNormal = Bump(uv, 1.0);
        float blend = 0.5; 
        float3 finalNormal = normalize(lerp(geomNormal, bumpNormal, blend));
        return float4(saturate(finalNormal * 0.5 + 0.5), 1.0);
    }

    float PS_ViewDepth(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return getDepth(uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }

    float4 PS_GTAO_Main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 0;
        }
        float2 scaled_uv = uv / RenderScale;

        float viewspaceZ_raw = tex2Dlod(sViewDepthTex, float4(scaled_uv, 0, 0)).r;
        float2 p = ReShade::PixelSize / RenderScale;

        int sliceCount, stepsPerSlice;
        if (QualityLevel == 0)
        {
            sliceCount = 2;
            stepsPerSlice = 2;
        }
        else if (QualityLevel == 1)
        {
            sliceCount = 3;
            stepsPerSlice = 4;
        }
        else if (QualityLevel == 2)
        {
            sliceCount = 4;
            stepsPerSlice = 8;
        }
        else
        {
            sliceCount = 8;
            stepsPerSlice = 8;
        }

        float3 pixCenterPos = GetViewPos(scaled_uv, viewspaceZ_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
        float viewspaceZ = pixCenterPos.z;

        if (getDepth(scaled_uv) >= DepthThreshold)
        {
            float encodedValue = XeGTAO_EncodeVisibilityBentNormal(1.0, float3(0, 0, 1));
            return ReconstructFloat4(encodedValue);
        }

        half3 viewspaceNormal = tex2Dlod(sNormal, float4(scaled_uv, 0, 0)).xyz * 2.0 - 1.0;
        const half3 viewVec = (half3) normalize(-pixCenterPos);

        const half effectRadius = (half) EffectRadius * (half) RadiusMultiplier;
        const half falloffRange = (half) FalloffRange * effectRadius;
        const half falloffFrom = effectRadius * (1.0 - (half) FalloffRange);
        const half falloffMul = -1.0 / falloffRange;
        const half falloffAdd = falloffFrom / falloffRange + 1.0;
        const half sampleDistributionPower = (half) SampleDistributionPower;
        const half thinOccluderCompensation = (half) ThinOccluderCompensation;

        half visibility = 0;
        half3 bentNormal = 0;

        float2 tileCoord = float2(fmod(uv.x * BUFFER_WIDTH, 64.0), fmod(uv.y * BUFFER_HEIGHT, 64.0));
        float hilbertIndex = xy2hilbert6(tileCoord.x, tileCoord.y);
        float temporalOffset = 288.0 * fmod(FRAME_COUNT, 64.0);
        float seqIndex = hilbertIndex + temporalOffset;
        float2 localNoise = R2(seqIndex);

        const half noiseSlice = (half) localNoise.x;
        const half noiseSample = (half) localNoise.y;

        const float fov_rad = FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / BUFFER_ASPECT_RATIO;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * (float2(2.0 / proj_scale_x, 2.0 / proj_scale_y) / (BUFFER_SCREEN_SIZE * RenderScale));

        half screenspaceRadius = effectRadius / (half) pixelDirRBViewspaceSizeAtCenterZ.x;
        
        visibility += saturate((10.0 - screenspaceRadius) / 100.0) * 0.5;

        const half pixelTooCloseThreshold = 1.3;
        const half minS = screenspaceRadius > 0 ? (pixelTooCloseThreshold / screenspaceRadius) : 0.0;

        for (half slice = 0; slice < sliceCount; slice++)
        {
            half sliceK = (slice + noiseSlice) / sliceCount;
            half phi = sliceK * PI;
            half cosPhi, sinPhi;
            sincos(phi, sinPhi, cosPhi);
            half2 omega = half2(cosPhi, -sinPhi);

            omega *= screenspaceRadius;

            const half3 directionVec = half3(cosPhi, sinPhi, 0);
            const half3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            const half3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            half3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

            half signNorm = (half) sign(dot(orthoDirectionVec, projectedNormalVec));
            half projectedNormalVecLength = length(projectedNormalVec);
            half cosNorm = saturate(dot(projectedNormalVec, viewVec) / max(projectedNormalVecLength, 1e-4));

            half n = signNorm * XeGTAO_FastACos(cosNorm);

            const half lowHorizonCos0 = cos(n + PI_HALF);
            const half lowHorizonCos1 = cos(n - PI_HALF);

            half horizonCos0 = lowHorizonCos0;
            half horizonCos1 = lowHorizonCos1;

            [unroll]
            for (half step = 0; step < stepsPerSlice; step++)
            {
                const half stepBaseNoise = (slice + step * stepsPerSlice) * 0.61803398875;
                half stepNoise = frac(noiseSample + stepBaseNoise);
                half s = (step + stepNoise) / stepsPerSlice;
                s = pow(s, sampleDistributionPower);
                s += minS;

                float2 sampleOffset = round(s * omega) * (ReShade::PixelSize / RenderScale);

                float2 sampleScreenPos0 = scaled_uv + sampleOffset;
                float SZ0 = tex2Dlod(sViewDepthTex, float4(sampleScreenPos0, 0, 0)).r;
                float3 samplePos0 = GetViewPos(sampleScreenPos0, SZ0 / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float2 sampleScreenPos1 = scaled_uv - sampleOffset;
                float SZ1 = tex2Dlod(sViewDepthTex, float4(sampleScreenPos1, 0, 0)).r;
                float3 samplePos1 = GetViewPos(sampleScreenPos1, SZ1 / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
                
                float3 sampleDelta0 = (samplePos0 - pixCenterPos);
                float3 sampleDelta1 = (samplePos1 - pixCenterPos);
                half sampleDist0 = (half) length(sampleDelta0);
                half sampleDist1 = (half) length(sampleDelta1);

                half3 sampleHorizonVec0 = (half3) (sampleDelta0 / max(sampleDist0, 1e-6));
                half3 sampleHorizonVec1 = (half3) (sampleDelta1 / max(sampleDist1, 1e-6));

                half falloffBase0 = length(half3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1.0 + thinOccluderCompensation)));
                half falloffBase1 = length(half3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1.0 + thinOccluderCompensation)));
                half weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                half weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);

                half shc0 = dot(sampleHorizonVec0, viewVec);
                half shc1 = dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            projectedNormalVecLength = lerp(projectedNormalVecLength, 1.0, 0.05);

            half h0 = -XeGTAO_FastACos(clamp(horizonCos1, -1.0, 1.0));
            half h1 = XeGTAO_FastACos(clamp(horizonCos0, -1.0, 1.0));

            half iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
            half iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
            half localVisibility = projectedNormalVecLength * (iarc0 + iarc1);
            visibility += localVisibility;

            if (ComputeBentNormals)
            {
                half t0 = (6.0 * sin(h0 - n) - sin(3.0 * h0 - n) + 6.0 * sin(h1 - n) - sin(3.0 * h1 - n) + 16.0 * sin(n) - 3.0 * (sin(h0 + n) + sin(h1 + n))) / 12.0;
                half t1 = (-cos(3.0 * h0 - n) - cos(3.0 * h1 - n) + 8.0 * cos(n) - 3.0 * (cos(h0 + n) + cos(h1 + n))) / 12.0;
                half3 localBentNormal = half3(directionVec.x * t0, directionVec.y * t0, -t1);
                localBentNormal = mul(XeGTAO_RotFromToMatrix(half3(0, 0, -1), viewVec), localBentNormal) * projectedNormalVecLength;
                bentNormal += localBentNormal;
            }
        }
        
        visibility /= (half) sliceCount;
        visibility = pow(saturate(visibility), (half) FinalValuePower);
        visibility = max(0.03, visibility);

        if (ComputeBentNormals)
        {
            bentNormal = normalize(bentNormal + half3(0, 0, 1e-6));
        }
        else
        {
            bentNormal = half3(0, 0, 1);
        }

        float encodedValue = XeGTAO_EncodeVisibilityBentNormal(visibility, bentNormal);
        return ReconstructFloat4(encodedValue);
    }

    float4 PS_TemporalAccumulate(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 0;
        }

        float4 currentPackedAO = tex2D(sAOTermTex, uv);

        if (!EnableTemporal)
        {
            return currentPackedAO;
        }

        float2 full_res_uv = uv / RenderScale;
        
        float2 motion = SampleMotionVectors(full_res_uv);
        float2 reprojected_uv_full = full_res_uv + motion;

        float currentDepth = tex2D(sViewDepthTex, full_res_uv).r;
        float historyDepth = tex2D(sViewDepthTex, reprojected_uv_full).r;

        float2 reprojected_uv_low = reprojected_uv_full * RenderScale;

        bool validHistory = all(saturate(reprojected_uv_low) == reprojected_uv_low) &&
                                  FRAME_COUNT > 1 &&
                                  abs(historyDepth - currentDepth) < (currentDepth * 0.02);

        float4 blendedAO = currentPackedAO;
        if (validHistory)
        {
            float4 historyPackedAO = tex2D(sHistoryTex, reprojected_uv_low);
            float4 minBox = currentPackedAO;
            float4 maxBox = currentPackedAO;

            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    
                    float4 neighborPackedAO = tex2Doffset(sAOTermTex, uv, int2(x, y));
                    minBox = min(minBox, neighborPackedAO);
                    maxBox = max(maxBox, neighborPackedAO);
                }
            }
            
            float4 clampedHistoryAO = clamp(historyPackedAO, minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, TemporalAccumulationFrames);
            blendedAO = lerp(clampedHistoryAO, currentPackedAO, alpha);
        }
        
        return blendedAO;
    }

    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outHistory : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outHistory = 0;
            return;
        }
        outHistory = tex2D(sAccumulatedAOTex, uv);
    }

    float PS_PrepareDenoise(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 1.0;
        }

        float4 packedAO = tex2D(sAccumulatedAOTex, uv);
        float encodedValue = ReconstructUint(packedAO);
        half visibility, bentNormal;
        XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
        return visibility;
    }

    float PS_DenoisePass(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, int level, sampler input_sampler) : SV_Target
    {
        if (any(texcoord > RenderScale))
        {
            return tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0)).r;
        }
        return atrous_refactored(input_sampler, texcoord, level);
    }
    
    float PS_DenoisePass0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return PS_DenoisePass(vpos, texcoord, 0, sDenoiseInputTex);
    }

    float PS_DenoisePass1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return PS_DenoisePass(vpos, texcoord, 1, sDenoiseTex0);
    }

    float PS_DenoisePass2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return PS_DenoisePass(vpos, texcoord, 2, sDenoiseTex1);
    }

    float PS_Upscale(float4 vpos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (RenderScale >= 1.0)
        {
            if (EnableDenoise)
                return tex2D(sDenoiseTex0, uv).r;
            else
                return tex2D(sDenoiseInputTex, uv).r;
        }

        // FSR
        const float2 fDownscaleFactor = float2(RenderScale, RenderScale);
        const float2 fRenderSize = BUFFER_SCREEN_SIZE * fDownscaleFactor;

        const float2 fDstOutputPos = uv * BUFFER_SCREEN_SIZE + 0.5f;
        const float2 fSrcOutputPos = fDstOutputPos * fDownscaleFactor;
        const int2 iSrcInputPos = int2(floor(fSrcOutputPos));
        
        const float2 fSrcUnjitteredPos = (float2(iSrcInputPos) + 0.5f);
        const float2 fBaseSampleOffset = fSrcUnjitteredPos - fSrcOutputPos;

        int2 offsetTL;
        offsetTL.x = (fSrcUnjitteredPos.x > fSrcOutputPos.x) ? -2 : -1;
        offsetTL.y = (fSrcUnjitteredPos.y > fSrcOutputPos.y) ? -2 : -1;

        const bool bFlipCol = fSrcUnjitteredPos.x > fSrcOutputPos.x;
        const bool bFlipRow = fSrcUnjitteredPos.y > fSrcOutputPos.y;

        RectificationBox clippingBox;
        float fSamples[9];
        int iSampleIndex = 0;

        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                const int2 sampleColRow = int2(bFlipCol ? (2 - col) : col, bFlipRow ? (2 - row) : row);
                const int2 iSrcSamplePos = iSrcInputPos + offsetTL + sampleColRow;
                
                const float2 sample_uv = ((iSrcSamplePos + 0.5) * rcp(fRenderSize)) * RenderScale;
                
                if (EnableDenoise)
                    fSamples[iSampleIndex] = tex2Dlod(sDenoiseTex0, float4(sample_uv, 0, 0)).r;
                else
                    fSamples[iSampleIndex] = tex2Dlod(sDenoiseInputTex, float4(sample_uv, 0, 0)).r;

                iSampleIndex++;
            }
        }

        iSampleIndex = 0;
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                const int2 sampleColRow = int2(bFlipCol ? (2 - col) : col, bFlipRow ? (2 - row) : row);
                const float2 fOffset = (float2) offsetTL + (float2) sampleColRow;
                const float2 fSrcSampleOffset = fBaseSampleOffset + fOffset;

                const float fRectificationCurveBias = -2.3f;
                const float fSrcSampleOffsetSq = dot(fSrcSampleOffset, fSrcSampleOffset);
                const float fBoxSampleWeight = exp(fRectificationCurveBias * fSrcSampleOffsetSq);

                const bool bInitialSample = (row == 0) && (col == 0);
                RectificationBoxAddSample(clippingBox, bInitialSample, fSamples[iSampleIndex], fBoxSampleWeight);
                iSampleIndex++;
            }
        }
        
        RectificationBoxComputeVarianceBoxData(clippingBox);

        return clippingBox.boxCenter;
    }

    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);

        if (ViewMode == 0) // Normal
        {
            if (getDepth(uv) >= DepthThreshold)
                return originalColor;

            half visibility = tex2D(sUpscaledAOTex, uv).r;
            float occlusion = 1.0 - visibility;
            occlusion = saturate(occlusion * Intensity);

            float fade = saturate(1.0 - smoothstep(0.95, 1.0, getDepth(uv)));
            occlusion *= fade;

            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);
            return originalColor;
        }
        // Debug
        else if (ViewMode == 1) // Normals
        {
            return float4(tex2D(sNormal, uv).rgb, 1.0);
        }
        else if (ViewMode == 2) // View-Space Depth
        {
            float depth = tex2D(sViewDepthTex, uv).r / (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * DepthMultiplier);
            return float4(saturate(depth.rrr), 1.0);
        }
        else if (ViewMode == 3) // Raw AO
        {
            float4 packedAO = tex2D(sAOTermTex, uv);
            float encodedValue = ReconstructUint(packedAO);
            half visibility, bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            return float4(visibility.xxx, 1.0);
        }
        else if (ViewMode == 4) // Denoised AO
        {
            if (EnableDenoise)
                return float4(tex2D(sDenoiseTex0, uv).rrr, 1.0);
            else
                return float4(0.0, 1.0, 0.0, 1.0); // Green screen to indicate denoiser is off
        }
        else if (ViewMode == 5) // Upscaled AO
        {
            return float4(tex2D(sUpscaledAOTex, uv).rrr, 1.0);
        }

        return originalColor;
    }

    technique XeGTAO
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
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TemporalAccumulate;
            RenderTarget = AccumulatedAOTex;
            ClearRenderTargets = true;
        }
        pass UpdateHistoryPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HistoryTex;
            ClearRenderTargets = true;
        }
        
        pass PrepareDenoisePass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_PrepareDenoise;
            RenderTarget = DenoiseInputTex;
            ClearRenderTargets = true;
        }
        pass DenoisePass0
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass0;
            RenderTarget = DenoiseTex0;
            ClearRenderTargets = true;
        }
        pass DenoisePass1
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass1;
            RenderTarget = DenoiseTex1;
            ClearRenderTargets = true;
        }
        pass DenoisePass2
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass2;
            RenderTarget = DenoiseTex0;
            ClearRenderTargets = true;
        }
        
        pass UpscalePass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale;
            RenderTarget = UpscaledAOTex;
        }
        
        pass OutputPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
