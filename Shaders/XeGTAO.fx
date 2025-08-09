///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation 
// 
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion", 
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// 
// Implementation:  Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>             (\_/)
// Version:         (see XeGTAO.h)                                                                                (='.'=)
// Details:         https://github.com/GameTechDev/XeGTAO                                                         (")_(")
//
// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*------------------.
| :: Description :: |
'-------------------/

    XeGTAO
    Versão 0.9
    Author: Barbatos

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 0.9
    * Initial Version
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

/*---------.
| :: UI :: |
'---------*/

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if UI_DIFFICULTY == 0
#define RadiusMultiplier              1.0
#define FinalValuePower                0.8
#define FalloffRange                   0.01
#define RenderScale                     1.0
#define SampleDistributionPower         1.0
#define ThinOccluderCompensation        0.0
#define ComputeBentNormals              false
#define EnableTemporal                  true
#define DepthMultiplier                 1.0
#define DepthThreshold                  0.999
#define bSmoothNormals                  false
#define FOV                             90.0
#endif

uniform float Intensity <
    ui_category = "General";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
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
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.01;
> = 1.0;

uniform float4 OcclusionColor <
    ui_category = "Extra";
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
    ui_category = "Debug";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "None\0AO Only\0Normals\0Depth (View-Space)\0Raw AO\0";
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
> = 0.01;

uniform float RenderScale <
    ui_category = "GTAO Quality";
    ui_label = "Render Scale";
    ui_tooltip = "Renders AO at a lower resolution to improve performance. 1.0 is full resolution.";
    ui_type = "drag";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform float SampleDistributionPower <
    ui_category = "GTAO Quality";
    ui_label = "Sample Distribution Power";
    ui_tooltip = "Controls how samples are distributed along the view ray. >1 pushes samples further away.";
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
> = 1.0;

uniform float ThinOccluderCompensation <
    ui_category = "GTAO Quality";
    ui_label = "Thin Occluder Compensation";
    ui_tooltip = "Reduces self-occlusion on thin objects.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool ComputeBentNormals <
    ui_category = "GTAO Quality";
    ui_label = "Compute Bent Normals";
> = false;

uniform bool EnableTemporal <
    ui_category = "Temporal Filtering";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Blends the current frame's AO with previous frames to reduce noise and flickering.";
> = true;

uniform float DepthMultiplier <
    ui_category = "Depth/Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
    ui_type = "drag";
> = 1.0;

uniform float DepthThreshold <
    ui_category = "Depth/Normals";
    ui_label = "Sky Threshold";
    ui_tooltip = "Sets the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_type = "drag";
> = 0.999;

uniform bool bSmoothNormals <
    ui_category = "Depth/Normals";
    ui_label = "Smooth Normals";
> = false;

uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 90.0;

#endif // UI_DIFFICULTY == 1


uniform float frametime < source = "frametime"; >;
uniform int FRAME_COUNT < source = "framecount"; >;

/*----------------.
| :: Textures :: |
'----------------*/

// Motion Vectors Texture (copied from SSRT)
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
    Texture = texMotionVectors;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};
float2 SampleMotionVectors(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif


namespace NEOGTAO
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


/*----------------.
| :: Functions :: |
'----------------*/

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
        float2 p = ReShade::PixelSize;
        float3 center_pos = GetViewPos(texcoord, getDepth(texcoord));
        float3 pos_down1 = GetViewPos(texcoord + float2(0, p.y), getDepth(texcoord + float2(0, p.y)));
        float3 pos_down2 = GetViewPos(texcoord + float2(0, p.y * 2), getDepth(texcoord + float2(0, p.y * 2)));
        float3 extrapolated_down = pos_down1 + (pos_down1 - pos_down2);
        float3 pos_up1 = GetViewPos(texcoord - float2(0, p.y), getDepth(texcoord - float2(0, p.y)));
        float3 pos_up2 = GetViewPos(texcoord - float2(0, p.y * 2), getDepth(texcoord - float2(0, p.y * 2)));
        float3 extrapolated_up = pos_up1 + (pos_up1 - pos_up2);
        float3 ddy = pos_down1 - center_pos;
        if (abs(extrapolated_up.z - center_pos.z) < abs(extrapolated_down.z - center_pos.z))
        {
            ddy = center_pos - pos_up1;
        }
        float3 pos_right1 = GetViewPos(texcoord + float2(p.x, 0), getDepth(texcoord + float2(p.x, 0)));
        float3 pos_right2 = GetViewPos(texcoord + float2(p.x * 2, 0), getDepth(texcoord + float2(p.x * 2, 0)));
        float3 extrapolated_right = pos_right1 + (pos_right1 - pos_right2);
        float3 pos_left1 = GetViewPos(texcoord - float2(p.x, 0), getDepth(texcoord - float2(p.x, 0)));
        float3 pos_left2 = GetViewPos(texcoord - float2(p.x * 2, 0), getDepth(texcoord - float2(p.x * 2, 0)));
        float3 extrapolated_left = pos_left1 + (pos_left1 - pos_left2);
        float3 ddx = pos_right1 - center_pos;
        if (abs(extrapolated_left.z - center_pos.z) < abs(extrapolated_right.z - center_pos.z))
        {
            ddx = center_pos - pos_left1;
        }
        return normalize(cross(ddy, ddx));
    }
    
    float3 GetSmoothedNormal(float2 texcoord)
    {
        float3 normal = computeNormal(texcoord);
        if (bSmoothNormals)
        {
            float center_depth = getDepth(texcoord);
            float2 offset = ReShade::PixelSize * 7.5 * (1.0 - center_depth);

            float3 normalT = computeNormal(texcoord - float2(0, offset.y));
            float3 normalB = computeNormal(texcoord + float2(0, offset.y));
            float3 normalL = computeNormal(texcoord - float2(offset.x, 0));
            float3 normalR = computeNormal(texcoord + float2(offset.x, 0));

            float wT = smoothstep(1, 0, distance(normal, normalT) * 1.5) * 2;
            float wB = smoothstep(1, 0, distance(normal, normalB) * 1.5) * 2;
            float wL = smoothstep(1, 0, distance(normal, normalL) * 1.5) * 2;
            float wR = smoothstep(1, 0, distance(normal, normalR) * 1.5) * 2;

            float4 weightedNormal = float4(normal, 1.0)
                + float4(normalT * wT, wT)
                + float4(normalB * wB, wB)
                + float4(normalL * wL, wL)
                + float4(normalR * wR, wR);

            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }
        return normal;
    }

    uint XeGTAO_EncodeVisibilityBentNormal(half visibility, half3 bentNormal)
    {
        float4 unpackedInput = float4(bentNormal * 0.5 + 0.5, visibility);
        return ((uint(saturate(unpackedInput.x) * 255.0 + 0.5)) |
                (uint(saturate(unpackedInput.y) * 255.0 + 0.5) << 8) |
                (uint(saturate(unpackedInput.z) * 255.0 + 0.5) << 16) |
                (uint(saturate(unpackedInput.w) * 255.0 + 0.5) << 24));
    }

    void XeGTAO_DecodeVisibilityBentNormal(const uint packedValue, out half visibility, out half3 bentNormal)
    {
        half4 decoded;
        decoded.x = half(packedValue & 0x000000ff) / 255.0;
        decoded.y = half(((packedValue >> 8) & 0x000000ff)) / 255.0;
        decoded.z = half(((packedValue >> 16) & 0x000000ff)) / 255.0;
        decoded.w = half(packedValue >> 24) / 255.0;
        bentNormal = decoded.xyz * 2.0 - 1.0;
        visibility = decoded.w;
    }

    uint ReconstructUint(float4 packedFloat)
    {
        return (uint(round(packedFloat.x * 255.0))) |
               (uint(round(packedFloat.y * 255.0)) << 8) |
               (uint(round(packedFloat.z * 255.0)) << 16) |
               (uint(round(packedFloat.w * 255.0)) << 24);
    }
    
    float4 ReconstructFloat4(uint val)
    {
        return float4(
            (val & 0xFF) / 255.0,
            ((val >> 8) & 0xFF) / 255.0,
            ((val >> 16) & 0xFF) / 255.0,
            ((val >> 24) & 0xFF) / 255.0
        );
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

/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/

    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            discard;
            
        float3 normal = -GetSmoothedNormal(uv / RenderScale);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float PS_ViewDepth(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            discard;

        return getDepth(uv / RenderScale) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }

    float4 PS_GTAO_Main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            discard;

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

        float2 screen_uv = uv / RenderScale;
    
        float viewspaceZ_raw = tex2Dlod(sViewDepthTex, float4(uv, 0, 0)).r;
        float3 pixCenterPos = GetViewPos(screen_uv, viewspaceZ_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
        float viewspaceZ = pixCenterPos.z;

        if (getDepth(screen_uv) >= DepthThreshold)
        {
            uint encodedValue = XeGTAO_EncodeVisibilityBentNormal(1.0, float3(0, 0, 1));
            return ReconstructFloat4(encodedValue);
        }

        half3 viewspaceNormal = tex2Dlod(sNormal, float4(uv, 0, 0)).xyz * 2.0 - 1.0;
        const half3 viewVec = (half3) normalize(-pixCenterPos);
    
        const half effectRadius = (half) EffectRadius * (half) RadiusMultiplier;
        const half falloffRange = (half) FalloffRange * effectRadius;
        const half falloffFrom = effectRadius * (1.0 - (half) FalloffRange);
        const half falloffMul = -1.0 / falloffRange;
        const half falloffAdd = falloffFrom / falloffRange + 1.0;

        half visibility = 0;
        half3 bentNormal = 0;

        const half2 localNoise = rand3d(float3(screen_uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT) / 64.0, frametime * 0.001)).xy;

        const half noiseSlice = localNoise.x;
        const half noiseSample = localNoise.y;

        const float fov_rad = FOV * (PI / 180.0);
        const float pixelViewspaceSize = 2.0 * viewspaceZ * tan(fov_rad * 0.5) / (BUFFER_HEIGHT * RenderScale);
        half screenspaceRadius = effectRadius / (half) pixelViewspaceSize;
    
        screenspaceRadius = max(screenspaceRadius, 4.0);
        const half minS = 1.3 / screenspaceRadius;

        for (int sliceIndex = 0; sliceIndex < sliceCount; sliceIndex++)
        {
            half slice = (half) sliceIndex;
            half sliceK = (slice + noiseSlice) / (half) sliceCount;
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
            half cosNorm = saturate(dot(projectedNormalVec, viewVec) / max(projectedNormalVecLength, 1e-6));
            half n = signNorm * XeGTAO_FastACos(cosNorm);

            const half lowHorizonCos0 = cos(n + PI_HALF);
            const half lowHorizonCos1 = cos(n - PI_HALF);

            half horizonCos0 = lowHorizonCos0;
            half horizonCos1 = lowHorizonCos1;

            for (int stepIndex = 0; stepIndex < stepsPerSlice; stepIndex++)
            {
                half step = (half) stepIndex;
                const half stepBaseNoise = (step + slice * (half) stepsPerSlice) * 0.61803398875;
                half stepNoise = frac(noiseSample + stepBaseNoise);
                half s = (step + stepNoise) / (half) stepsPerSlice;
                s = pow(s, (half) SampleDistributionPower);
                s = max(s, minS); 
            
                half2 sampleOffset = s * omega * ReShade::PixelSize * RenderScale;

                float2 sampleCoord0 = uv + sampleOffset;
                float2 sampleCoord1 = uv - sampleOffset;
            
                sampleCoord0 = clamp(sampleCoord0, 0.0, RenderScale);
                sampleCoord1 = clamp(sampleCoord1, 0.0, RenderScale);

                float SZ0_raw = tex2Dlod(sViewDepthTex, float4(sampleCoord0, 0, 0)).r;
                float3 samplePos0 = GetViewPos(sampleCoord0 / RenderScale, SZ0_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float SZ1_raw = tex2Dlod(sViewDepthTex, float4(sampleCoord1, 0, 0)).r;
                float3 samplePos1 = GetViewPos(sampleCoord1 / RenderScale, SZ1_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float3 sampleDelta0 = (samplePos0 - pixCenterPos);
                float3 sampleDelta1 = (samplePos1 - pixCenterPos);
                half sampleDist0 = (half) length(sampleDelta0);
                half sampleDist1 = (half) length(sampleDelta1);

                half3 sampleHorizonVec0 = (half3) (sampleDelta0 / max(sampleDist0, 1e-6));
                half3 sampleHorizonVec1 = (half3) (sampleDelta1 / max(sampleDist1, 1e-6));

                half falloffBase0 = length(half3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1.0 + (half) ThinOccluderCompensation)));
                half falloffBase1 = length(half3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1.0 + (half) ThinOccluderCompensation)));
                half weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                half weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);

                half shc0 = dot(sampleHorizonVec0, viewVec);
                half shc1 = dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            projectedNormalVecLength = max(projectedNormalVecLength, 1.0);
            projectedNormalVecLength = lerp(projectedNormalVecLength, 1.0, 0.05);

            half h0 = -XeGTAO_FastACos(clamp(horizonCos1, -1.0, 1.0));
            half h1 = XeGTAO_FastACos(clamp(horizonCos0, -1.0, 1.0));
        
            h0 = clamp(h0, -PI_HALF, PI_HALF);
            h1 = clamp(h1, -PI_HALF, PI_HALF);
            n = clamp(n, -PI_HALF, PI_HALF);
        
            half iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
            half iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
            half localVisibility = projectedNormalVecLength * (iarc0 + iarc1);
        
            localVisibility = clamp(localVisibility, 0.0, 1.0);
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

        uint encodedValue = XeGTAO_EncodeVisibilityBentNormal(visibility, bentNormal);
        return ReconstructFloat4(encodedValue);
    }
    
    float4 PS_TemporalAccumulate(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 currentPackedAO = tex2D(sAOTermTex, uv);

        if (!EnableTemporal)
        {
            return currentPackedAO;
        }

        float2 screen_uv = uv / RenderScale;
        float2 motion = SampleMotionVectors(screen_uv);
        float2 reprojected_uv = screen_uv - motion;

        float currentDepth = tex2D(sViewDepthTex, uv).r;
        float historyDepth = tex2D(sViewDepthTex, reprojected_uv * RenderScale).r;

        bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) &&
                            FRAME_COUNT > 1 &&
                            abs(historyDepth - currentDepth) < (currentDepth * 0.02); 

        float4 blendedAO = currentPackedAO;
        if (validHistory)
        {
            float4 historyPackedAO = tex2D(sHistoryTex, reprojected_uv * RenderScale);
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
        outHistory = tex2D(sAccumulatedAOTex, uv);
    }

    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);

        if (ViewMode == 0) // Normal
        {
            if (getDepth(uv) >= DepthThreshold)
                return originalColor;

            float4 packedAO = tex2D(sAccumulatedAOTex, uv * RenderScale);
            uint encodedValue = ReconstructUint(packedAO);
            
            half visibility;
            half3 bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            
            float occlusion = 1.0 - visibility;
            occlusion = saturate(occlusion * Intensity);

            float fade = saturate(1.0 - smoothstep(0.95, 1.0, getDepth(uv)));
            occlusion *= fade;

            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);
            return originalColor;
        }
        else if (ViewMode == 1) // AO Only
        {
            float4 packedAO = tex2D(sAccumulatedAOTex, uv * RenderScale);
            uint encodedValue = ReconstructUint(packedAO);
            half visibility;
            half3 bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            return float4(visibility.xxx, 1.0);
        }
        else if (ViewMode == 2) // Normals
        {
            return float4(tex2D(sNormal, uv * RenderScale).rgb, 1.0);
        }
        else if (ViewMode == 3) // View-Space Depth
        {
            float depth = tex2D(sViewDepthTex, uv * RenderScale).r / (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * DepthMultiplier);
            return float4(saturate(depth.rrr), 1.0);
        }
        else if (ViewMode == 4) // Raw AO
        {
            float4 packedAO = tex2D(sAOTermTex, uv * RenderScale);
            uint encodedValue = ReconstructUint(packedAO);
            half visibility;
            half3 bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            return float4(visibility.xxx, 1.0);
        }

        return originalColor;
    }

technique XeGTAO< ui_tooltip = "XeGTAO BETA"; >
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
    pass Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_TemporalAccumulate;
        RenderTarget = AccumulatedAOTex;
    }
    pass UpdateHistoryPass
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
