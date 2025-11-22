/*
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

/*----------------------------------------------|
| ::           Barbatos XeGTAO               :: |
'-----------------------------------------------|
| Version: 0.9.0                                |
| Author: Barbatos                              |
'----------------------------------------------*/

#include "ReShade.fxh"

#define XE_GTAO_PI (3.1415926535897932384626433832795)
#define XE_GTAO_PI_HALF (1.5707963267948966192313216916398)

#define lpfloat float
#define lpfloat2 float2
#define lpfloat3 float3
#define lpfloat4 float4
#define lpfloat3x3 float3x3
#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4
#define half3x3 float3x3

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define getDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define PI 3.1415926535
#define PI_HALF 1.57079632679
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

#define bEnableDenoise 1
#define c_phi 1
#define n_phi 5
#define p_phi 1

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "AO Intensity";
    ui_tooltip = "Strength of the Ambient Occlusion effect.";
> = 1.0;

uniform float EffectRadius <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 50.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Effect Radius";
    ui_tooltip = "Main radius of the AO effect in view-space units.";
> = 2.0;

uniform float4 OcclusionColor <
    ui_type = "color";
    ui_category = "Basic Settings";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform bool EnableDiffuse <
    ui_category = "Diffuse / Indirect Lighting";
    ui_label = "Enable Diffuse Mode";
    ui_tooltip = "Enables Indirect Lighting (Bounces) calculation.";
> = false;

uniform float DiffuseIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Diffuse / Indirect Lighting";
    ui_label = "Diffuse Intensity";
> = 1.0;

uniform int QualityLevel <
    ui_type = "combo";
    ui_items = "Low (2 directions, 2 samples)\0Medium (3 directions, 4 samples)\0High (4 directions, 8 samples)\0Ultra (8 directions, 8 samples)\0";
    ui_category = "Performance";
    ui_label = "Quality Level";
    ui_tooltip = "Defines the number of directions and samples per direction. Higher = slower.";
> = 0;

uniform float RadiusMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Options";
    ui_label = "Radius Multiplier";
    ui_tooltip = "Additional radius multiplier.";
> = 1.0;

uniform float FinalValuePower <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Advanced Options";
    ui_label = "Occlusion Power";
    ui_tooltip = "Final occlusion modifier. Higher values make the occlusion contrast stronger.";
> = 0.8;

uniform float FalloffRange <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Options";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge.";
> = 0.6;

uniform float SampleDistributionPower <
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Advanced Options";
    ui_label = "Sample Distribution";
    ui_tooltip = "Controls how samples are distributed along the view ray.";
> = 2.0;

uniform float ThinOccluderCompensation <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Options";
    ui_label = "Thin Occluder Fix";
    ui_tooltip = "Reduces self-occlusion on thin objects.";
> = 0.0;

uniform bool EnableTemporal <
    ui_category = "Advanced Options";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Blends the current frame's AO with previous frames to reduce noise.";
> = true;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1000.0; ui_step = 10.0;
    ui_category = "Advanced Options";
    ui_label = "Depth Multiplier";
> = 100.0;

uniform float DepthThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Advanced Options";
    ui_label = "Sky Threshold";
> = 0.999;

uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
    ui_category = "Advanced Options";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View.";
> = 60.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Normals\0View-Space Depth\0Raw AO\0Denoised AO\0Temporal AO\0Edges\0Diffuse Lighting\0";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_tooltip = "Selects the debug view mode.";
> = 0;

uniform float frametime < source = "frametime"; >;
uniform int FRAME_COUNT < source = "framecount"; >;
uniform bool ComputeBentNormals = false;

//----------------|
// :: Textures :: |
//----------------|

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, texcoord).rg;
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
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace Barbatos_XeGTAO
{
    texture texNormalEdges
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormalEdges
    {
        Texture = texNormalEdges;S_PC
    };
    sampler sNormalLinear
    {
        Texture = texNormalEdges;S_LC
    };

    texture B_PrevLuma
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R8;
    };
    sampler sB_PrevLuma
    {
        Texture = B_PrevLuma;
    };

    texture DepthT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
    };
    sampler sDepth
    {
        Texture = DepthT;S_LC
    };
    sampler sViewDepthLinear
    {
        Texture = DepthT;S_LC
    };

    texture AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sAO
    {
        Texture = AO;S_LC
    };

    texture DenoiseT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDenoise
    {
        Texture = DenoiseT;S_LC
    };

    texture DenoiseTex0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDenoiseTex0
    {
        Texture = DenoiseTex0;S_LC
    };

    texture DenoiseTex1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDenoiseTex1
    {
        Texture = DenoiseTex1;S_LC
    };

    texture TempT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sTemp
    {
        Texture = TempT;S_LC
    };

    texture HistoryT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sHistory
    {
        Texture = HistoryT;S_LC
    };

//-------------|
// :: Utility::|
//-------------|

#if __SHADERMODEL__ >= 40
    int hilbert(int2 p, int level)
    {
        int d = 0;
        for (int k = 0; k < level; k++)
        {
            int n = level - k - 1;
            int2 r = (p >> n) & 1;
            d += ((3 * r.x) ^ r.y) << (2 * n);
            if (r.y == 0) {
                if (r.x == 1) p = (1 << n) - 1 - p;
                p = p.yx;
            }
        }
        return d;
    }
    uint HilbertIndex(uint x, uint y) { return hilbert(int2(x % 64, y % 64), 6); }
#else
    float hilbert(float2 p, int level)
    {
        float d = 0;
        for (int k = 0; k < level; k++)
        {
            int n = level - k - 1;
            float n_pow2 = exp2(n);
            float2 r = fmod(floor(p / n_pow2), 2.0);
            float term = r.y + r.x * (3.0 - 2.0 * r.y);
            d += term * exp2(2 * n);
            if (r.y < 0.5)
            {
                if (r.x > 0.5)
                    p = n_pow2 - 1.0 - p;
                p = p.yx;
            }
        }
        return d;
    }
    uint HilbertIndex(uint x, uint y)
    {
        return hilbert(float2(x % 64, y % 64), 6);
    }
#endif

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        uint index = HilbertIndex(pixCoord.x, pixCoord.y);
        index += 288 * (temporalIndex % 64);
        return frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114));
    }
    
    half XeGTAO_FastSqrt(float x)
    {
        return (half) (asfloat(0x1fbd1df5 + (asint(x) >> 1)));
    }
    
    half XeGTAO_FastACos(half inX)
    {
        half x = abs(inX);
        half res = -0.156583 * x + PI_HALF;
        res *= XeGTAO_FastSqrt(1.0 - x);
        return (inX >= 0) ? res : PI - res;
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
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

    half4 XeGTAO_CalculateEdges(const half centerZ, const half leftZ, const half rightZ, const half topZ, const half bottomZ)
    {
        half4 edgesLRTB = half4(leftZ, rightZ, topZ, bottomZ) - centerZ;
        half slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
        half slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
        half4 edgesLRTBSlopeAdjusted = edgesLRTB + half4(slopeLR, -slopeLR, slopeTB, -slopeTB);
        edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
        return saturate((1.25 - edgesLRTB / (centerZ * 0.011)));
    }

    half XeGTAO_PackEdges(half4 edgesLRTB)
    {
        edgesLRTB = round(saturate(edgesLRTB) * 2.9);
        return dot(edgesLRTB, half4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
    }

    float3 XeGTAO_CalculateNormalBGI(const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
    {
        float4 acceptedNormals = saturate(float4(edgesLRTB.x * edgesLRTB.z, edgesLRTB.z * edgesLRTB.y, edgesLRTB.y * edgesLRTB.w, edgesLRTB.w * edgesLRTB.x) + 0.01);
        pixLPos = normalize(pixLPos - pixCenterPos);
        pixRPos = normalize(pixRPos - pixCenterPos);
        pixTPos = normalize(pixTPos - pixCenterPos);
        pixBPos = normalize(pixBPos - pixCenterPos);
        float3 pixelNormal = acceptedNormals.x * cross(pixLPos, pixTPos) +
                             acceptedNormals.y * cross(pixTPos, pixRPos) +
                             acceptedNormals.z * cross(pixRPos, pixBPos) +
                             acceptedNormals.w * cross(pixBPos, pixLPos);
        return normalize(pixelNormal);
    }

    float3 GetViewPos(float2 uv, float view_z)
    {
        float fov_rad = FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / BUFFER_ASPECT_RATIO;
        float2 clip_pos = uv * 2.0 - 1.0;
        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z;
        view_pos.z = view_z;
        return view_pos;
    }

    float4 atrous(sampler input_sampler, float2 texcoord, int level, float center_depth, float3 center_normal)
    {
        float4 sum = 0.0;
        float cum_w = 0.0;
        const float2 step_size = ReShade::PixelSize * exp2(level);
        float4 center_color = tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0));
        float3 center_pos = GetViewPos(texcoord, center_depth);

        static const float2 atrous_offsets[9] =
        {
            float2(-1, -1), float2(0, -1), float2(1, -1),
            float2(-1, 0), float2(0, 0), float2(1, 0),
            float2(-1, 1), float2(0, 1), float2(1, 1)
        };

        [loop]
        for (int i = 0; i < 9; i++)
        {
            const float2 uv = texcoord + atrous_offsets[i] * step_size;
            const float4 sample_color = tex2Dlod(input_sampler, float4(uv, 0.0, 0.0));
            const float sample_depth = tex2Dlod(sViewDepthLinear, float4(uv, 0.0, 0.0)).r;

            if (sample_depth / (DepthMultiplier * 10.0) >= DepthThreshold)
                continue;

            const float3 sample_normal = tex2Dlod(sNormalLinear, float4(uv, 0.0, 0.0)).xyz * 2.0 - 1.0;
            const float3 sample_pos = GetViewPos(uv, sample_depth);
            
            float diff_c = distance(center_color.rgb, sample_color.rgb); 
            float w_c = exp(-(diff_c * diff_c) / c_phi);
            float diff_n = dot(center_normal, sample_normal);
            float w_n = pow(saturate(diff_n), n_phi);
            float diff_p = distance(center_pos, sample_pos);
            float w_p = exp(-(diff_p * diff_p) / p_phi);

            const float weight = w_c * w_n * w_p;
            sum += sample_color * weight;
            cum_w += weight;
        }
        return cum_w > 1e-6 ? (sum / cum_w) : center_color;
    }

    struct GTAOConstants
    {
        float2 ViewportSize;
        float2 ViewportPixelSize;
        float2 NDCToViewMul;
        float2 NDCToViewAdd;
        float2 NDCToViewMul_x_PixelSize;
        float EffectRadius;
        float EffectFalloffRange;
        float RadiusMultiplier;
        float SampleDistributionPower;
        float ThinOccluderCompensation;
        float FinalValuePower;
    };

    float XeGTAO_ScreenSpaceToViewSpaceDepth(const float screenDepth, const GTAOConstants consts)
    {
        return screenDepth * DepthMultiplier;
    }

    float3 XeGTAO_ComputeViewspacePosition(const float2 screenPos, const float viewspaceDepth, const GTAOConstants consts)
    {
        float3 ret;
        ret.xy = (consts.NDCToViewMul * screenPos.xy + consts.NDCToViewAdd) * viewspaceDepth;
        ret.z = viewspaceDepth;
        return ret;
    }

//--------------------|
// :: Pixel Shaders ::|
//--------------------|

    float PS_ViewDepth(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        GTAOConstants consts;
        return XeGTAO_ScreenSpaceToViewSpaceDepth(ReShade::GetLinearizedDepth(uv), consts);
    }

    void PS_NormalsEdges(float4 pos : SV_Position, float2 texcoord : TEXCOORD, out float4 outNormalEdges : SV_Target)
    {
        GTAOConstants consts;
        consts.ViewportSize = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        consts.ViewportPixelSize = 1.0 / consts.ViewportSize;
        float tanHalfFOV = tan(radians(FOV * 0.5));
        float aspect = BUFFER_WIDTH / BUFFER_HEIGHT;
        consts.NDCToViewMul = float2(aspect * tanHalfFOV * 2.0, -tanHalfFOV * 2.0);
        consts.NDCToViewAdd = float2(aspect * tanHalfFOV * -1.0, tanHalfFOV);
        
        float dC = tex2D(sDepth, texcoord).r;
        float2 pixelSize = ReShade::PixelSize;
        float dL = tex2D(sDepth, texcoord + float2(-pixelSize.x, 0)).r;
        float dR = tex2D(sDepth, texcoord + float2(pixelSize.x, 0)).r;
        float dT = tex2D(sDepth, texcoord + float2(0, -pixelSize.y)).r;
        float dB = tex2D(sDepth, texcoord + float2(0, pixelSize.y)).r;

        lpfloat4 edgesLRTB = XeGTAO_CalculateEdges((lpfloat) dC, (lpfloat) dL, (lpfloat) dR, (lpfloat) dT, (lpfloat) dB);
        lpfloat packedEdges = XeGTAO_PackEdges(edgesLRTB);

        float3 CENTER = XeGTAO_ComputeViewspacePosition(texcoord, dC, consts);
        float3 LEFT = XeGTAO_ComputeViewspacePosition(texcoord + float2(-1, 0) * consts.ViewportPixelSize, dL, consts);
        float3 RIGHT = XeGTAO_ComputeViewspacePosition(texcoord + float2(1, 0) * consts.ViewportPixelSize, dR, consts);
        float3 TOP = XeGTAO_ComputeViewspacePosition(texcoord + float2(0, -1) * consts.ViewportPixelSize, dT, consts);
        float3 BOTTOM = XeGTAO_ComputeViewspacePosition(texcoord + float2(0, 1) * consts.ViewportPixelSize, dB, consts);
        
        float3 viewspaceNormal = XeGTAO_CalculateNormalBGI(edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM);
        outNormalEdges = float4(viewspaceNormal * 0.5 + 0.5, packedEdges);
    }

    float4 PS_GTAO_Main(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        GTAOConstants consts;
        consts.ViewportSize = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        consts.ViewportPixelSize = 1.0 / consts.ViewportSize;
        float tanHalfFOV = tan(radians(FOV * 0.5));
        float aspect = BUFFER_WIDTH / BUFFER_HEIGHT;
        consts.NDCToViewMul = float2(aspect * tanHalfFOV * 2.0, -tanHalfFOV * 2.0);
        consts.NDCToViewAdd = float2(aspect * tanHalfFOV * -1.0, tanHalfFOV);
        consts.NDCToViewMul_x_PixelSize = consts.NDCToViewMul * consts.ViewportPixelSize;
        
        consts.EffectRadius = EffectRadius;
        consts.EffectFalloffRange = FalloffRange;
        consts.RadiusMultiplier = RadiusMultiplier;
        consts.SampleDistributionPower = SampleDistributionPower;
        consts.ThinOccluderCompensation = ThinOccluderCompensation;
        consts.FinalValuePower = FinalValuePower;

        lpfloat viewspaceZ = (lpfloat) tex2D(sDepth, texcoord).r;
        float4 normalEdges = tex2D(sNormalEdges, texcoord);
        lpfloat3 viewspaceNormal = (lpfloat3) (normalEdges.xyz * 2.0 - 1.0);
        viewspaceZ *= 0.99920;

        const float3 pixCenterPos = XeGTAO_ComputeViewspacePosition(texcoord, viewspaceZ, consts);
        const lpfloat3 viewVec = (lpfloat3) normalize(-pixCenterPos);

        lpfloat sliceCount;
        lpfloat stepsPerSlice;
        if (QualityLevel == 0)
        {
            sliceCount = 2;
            stepsPerSlice = 2;
        }
        else if (QualityLevel == 1)
        {
            sliceCount = 3;
            stepsPerSlice = 3;
        }
        else if (QualityLevel == 2)
        {
            sliceCount = 9;
            stepsPerSlice = 3;
        }
        else
        {
            sliceCount = 9;
            stepsPerSlice = 9;
        }

        uint2 pixCoord = uint2(pos.xy);
        lpfloat2 localNoise = (lpfloat2) SpatioTemporalNoise(pixCoord, FRAME_COUNT);
        
        const lpfloat effectRadius = (lpfloat) consts.EffectRadius * (lpfloat) consts.RadiusMultiplier;
        const lpfloat falloffRange = (lpfloat) consts.EffectFalloffRange * effectRadius;
        const lpfloat falloffFrom = effectRadius * ((lpfloat) 1 - (lpfloat) consts.EffectFalloffRange);
        const lpfloat falloffMul = (lpfloat) -1.0 / (falloffRange);
        const lpfloat falloffAdd = falloffFrom / (falloffRange) + (lpfloat) 1.0;

        lpfloat visibility = 0;
        lpfloat3 bentNormal = 0;
        lpfloat3 accumDiffuse = 0;
        lpfloat totalDiffuseWeight = 0;

        const lpfloat pixelTooCloseThreshold = 1.3;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * consts.NDCToViewMul_x_PixelSize;
        lpfloat screenspaceRadius = effectRadius / (lpfloat) pixelDirRBViewspaceSizeAtCenterZ.x;

        visibility += saturate((10 - screenspaceRadius) / 100) * 0.5;
        const lpfloat minS = (lpfloat) pixelTooCloseThreshold / screenspaceRadius;

        for (lpfloat slice = 0; slice < sliceCount; slice++)
        {
            lpfloat sliceK = (slice + localNoise.x) / sliceCount;
            lpfloat phi = sliceK * XE_GTAO_PI;
            lpfloat cosPhi = cos(phi);
            lpfloat sinPhi = sin(phi);
            lpfloat2 omega = lpfloat2(cosPhi, -sinPhi) * screenspaceRadius;
            const lpfloat3 directionVec = lpfloat3(cosPhi, sinPhi, 0);
            const lpfloat3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            const lpfloat3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            lpfloat3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);
            lpfloat signNorm = (lpfloat) sign(dot(orthoDirectionVec, projectedNormalVec));
            lpfloat projectedNormalVecLength = length(projectedNormalVec);
            lpfloat cosNorm = (lpfloat) saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);
            lpfloat n = signNorm * XeGTAO_FastACos(cosNorm);

            const lpfloat lowHorizonCos0 = cos(n + XE_GTAO_PI_HALF);
            const lpfloat lowHorizonCos1 = cos(n - XE_GTAO_PI_HALF);
            lpfloat horizonCos0 = lowHorizonCos0;
            lpfloat horizonCos1 = lowHorizonCos1;

            [unroll]
            for (lpfloat step = 0; step < stepsPerSlice; step++)
            {
                const lpfloat stepBaseNoise = lpfloat(slice + step * stepsPerSlice) * 0.6180339887498948482;
                lpfloat stepNoise = frac(localNoise.y + stepBaseNoise);
                lpfloat s = (step + stepNoise) / (stepsPerSlice);
                s = (lpfloat) pow(s, (lpfloat) consts.SampleDistributionPower);
                s += minS;

                lpfloat2 sampleOffset = s * omega;
                sampleOffset = round(sampleOffset) * (lpfloat2) consts.ViewportPixelSize;

                float2 sampleScreenPos0 = texcoord + sampleOffset;
                float SZ0 = tex2D(sDepth, sampleScreenPos0).r;
                float3 samplePos0 = XeGTAO_ComputeViewspacePosition(sampleScreenPos0, SZ0, consts);

                float2 sampleScreenPos1 = texcoord - sampleOffset;
                float SZ1 = tex2D(sDepth, sampleScreenPos1).r;
                float3 samplePos1 = XeGTAO_ComputeViewspacePosition(sampleScreenPos1, SZ1, consts);

                float3 sampleDelta0 = (samplePos0 - float3(pixCenterPos));
                float3 sampleDelta1 = (samplePos1 - float3(pixCenterPos));
                lpfloat sampleDist0 = (lpfloat) length(sampleDelta0);
                lpfloat sampleDist1 = (lpfloat) length(sampleDelta1);

                lpfloat3 sampleHorizonVec0 = (lpfloat3) (sampleDelta0 / sampleDist0);
                lpfloat3 sampleHorizonVec1 = (lpfloat3) (sampleDelta1 / sampleDist1);

                lpfloat falloffBase0 = length(lpfloat3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1 + consts.ThinOccluderCompensation)));
                lpfloat falloffBase1 = length(lpfloat3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1 + consts.ThinOccluderCompensation)));
                lpfloat weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                lpfloat weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);

                lpfloat shc0 = (lpfloat) dot(sampleHorizonVec0, viewVec);
                lpfloat shc1 = (lpfloat) dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);

                if (EnableDiffuse)
                {
                    float3 col0 = GetColor(sampleScreenPos0).rgb;
                    float3 col1 = GetColor(sampleScreenPos1).rgb;
                    
                    // Cosine Theta term: N dot L
                    float3 L0 = normalize(sampleDelta0);
                    float3 L1 = normalize(sampleDelta1);
                    
                    float NdotL0 = saturate(dot(viewspaceNormal, L0));
                    float NdotL1 = saturate(dot(viewspaceNormal, L1));
                    
                    accumDiffuse += col0 * NdotL0 * weight0;
                    accumDiffuse += col1 * NdotL1 * weight1;
                    totalDiffuseWeight += (weight0 + weight1);
                }
            }

            projectedNormalVecLength = lerp(projectedNormalVecLength, 1, 0.05);
            lpfloat h0 = -XeGTAO_FastACos((lpfloat) horizonCos1);
            lpfloat h1 = XeGTAO_FastACos((lpfloat) horizonCos0);
            lpfloat iarc0 = ((lpfloat) cosNorm + (lpfloat) 2 * (lpfloat) h0 * (lpfloat) sin(n) - (lpfloat) cos((lpfloat) 2 * (lpfloat) h0 - n)) / (lpfloat) 4;
            lpfloat iarc1 = ((lpfloat) cosNorm + (lpfloat) 2 * (lpfloat) h1 * (lpfloat) sin(n) - (lpfloat) cos((lpfloat) 2 * (lpfloat) h1 - n)) / (lpfloat) 4;
            lpfloat localVisibility = (lpfloat) projectedNormalVecLength * (lpfloat) (iarc0 + iarc1);
            visibility += localVisibility;

            if (ComputeBentNormals && !EnableDiffuse)
            {
                lpfloat t0 = (6 * sin(h0 - n) - sin(3 * h0 - n) + 6 * sin(h1 - n) - sin(3 * h1 - n) + 16 * sin(n) - 3 * (sin(h0 + n) + sin(h1 + n))) / 12;
                lpfloat t1 = (-cos(3 * h0 - n) - cos(3 * h1 - n) + 8 * cos(n) - 3 * (cos(h0 + n) + cos(h1 + n))) / 12;
                lpfloat3 localBentNormal = lpfloat3(directionVec.x * (lpfloat) t0, directionVec.y * (lpfloat) t0, -lpfloat(t1));
                localBentNormal = (lpfloat3) mul(XeGTAO_RotFromToMatrix(lpfloat3(0, 0, -1), viewVec), localBentNormal) * projectedNormalVecLength;
                bentNormal += localBentNormal;
            }
        }

        visibility /= (lpfloat) sliceCount;
        visibility = pow(visibility, (lpfloat) consts.FinalValuePower);
        visibility = max((lpfloat) 0.03, visibility);
        
        float3 finalDiffuse = 0;
        if (EnableDiffuse && totalDiffuseWeight > 0.001)
        {
            finalDiffuse = accumDiffuse / totalDiffuseWeight;
        }

        visibility = saturate(visibility);
        
        if (EnableDiffuse)
        {
            return float4(finalDiffuse, visibility);
        }
        else
        {
            if (ComputeBentNormals)
                bentNormal = normalize(bentNormal + half3(0, 0, 1e-6));
            else
                bentNormal = half3(0, 0, 1);
            
            float encoded = XeGTAO_EncodeVisibilityBentNormal(visibility, bentNormal);
            return ReconstructFloat4(encoded);
        }
    }

    float4 PS_PrepareDenoise(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 rawData = GetLod(sAO, uv);
        
        if (EnableDiffuse)
        {
            return rawData;
        }
        else
        {
            float encodedValue = ReconstructUint(rawData);
            half visibility, bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            return float4(visibility.xxx, visibility);
        }
    }
    
    float4 PS_DenoisePass(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, int level, sampler input_sampler) : SV_Target
    {
        if (!bEnableDenoise)
            return tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0));

        float depth = tex2Dlod(sViewDepthLinear, float4(texcoord, 0.0, 0.0)).r;
        if (depth / (DepthMultiplier * 10.0) >= DepthThreshold)
            return tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0));

        float3 normal = tex2Dlod(sNormalLinear, float4(texcoord, 0.0, 0.0)).xyz * 2.0 - 1.0;
        return atrous(input_sampler, texcoord, level, depth, normal);
    }

    float4 PS_DenoisePass0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return PS_DenoisePass(vpos, texcoord, 0, sDenoise);
    }
    float4 PS_DenoisePass1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return PS_DenoisePass(vpos, texcoord, 1, sDenoiseTex0);
    }
    
//------------|
// :: TAA  :: |
//------------|

    float2 GetVelocityFromClosestFragment(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0;
        const int2 offsets[9] = { int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(0, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1) };
        [unroll]
        for (int i = 0; i < 9; i++)
        {
            float2 s_coord = texcoord + offsets[i] * pixel_size;
            float s_depth = getDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                closest_velocity = SampleMotionVectors(s_coord);
            }
        }
        return closest_velocity;
    }
    
    void ComputeNeighborhoodMinMax(sampler2D color_tex, float2 texcoord, out float4 color_min, out float4 color_max)
    {
        float2 pixel_size = ReShade::PixelSize;
        float4 center_val = GetLod(color_tex, float4(texcoord, 0, 0));
        color_min = center_val;
        color_max = center_val;
        const int2 offsets_3x3[8] = { int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1) };
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            float4 n_val = GetLod(color_tex, float4(texcoord + offsets_3x3[i] * pixel_size, 0, 0));
            color_min = min(color_min, n_val);
            color_max = max(color_max, n_val);
        }
        const int2 offsets_cross[4] = { int2(0, -2), int2(-2, 0), int2(2, 0), int2(0, 2) };
        float4 cross_min = center_val;
        float4 cross_max = center_val;
        [unroll]
        for (int j = 0; j < 4; j++)
        {
            float4 n_val = GetLod(color_tex, float4(texcoord + offsets_cross[j] * pixel_size, 0, 0));
            cross_min = min(cross_min, n_val);
            cross_max = max(cross_max, n_val);
        }
        color_min = lerp(cross_min, color_min, 0.5);
        color_max = lerp(cross_max, color_max, 0.5);
    }
    
    float ComputeTrustFactor(float2 velocity_pixels, float low_threshold = 2.0, float high_threshold = 15.0)
    {
        float vel_mag = length(velocity_pixels);
        return saturate((high_threshold - vel_mag) / (high_threshold - low_threshold));
    }
    
    //Based on LumaFlow.fx from LumeniteFX CC-BY-NC-4.0
    float Confidence(float2 uv, float2 velocity)
    {
        float2 prev_uv = uv + velocity;
        if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
            return 0.0;

        float curr_luma = GetLuminance(GetColor(uv).rgb);
        float prev_luma = tex2D(sB_PrevLuma, prev_uv).r;
        float luma_error = abs(curr_luma - prev_luma);
        float flow_magnitude = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        
        if (flow_magnitude <= 1.0)
            return 1.0;

        float2 destination_velocity = SampleMotionVectors(prev_uv);
        float2 diff = velocity - destination_velocity;
        float error = length(diff);
        float normalized_error = error / length(velocity);

        float motion_penalty = flow_magnitude;
        float length_conf = rcp(motion_penalty * 0.05 + 1.0);
        float consistency_conf = rcp(normalized_error + 1.0);
        float photometric_conf = exp(-luma_error * 5.0);

        return (consistency_conf * length_conf * photometric_conf);
    }

    float4 PS_DenoiseAndAccumulate(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {  
        float center_depth = tex2Dlod(sViewDepthLinear, float4(uv, 0.0, 0.0)).r;
        float3 center_normal = tex2Dlod(sNormalLinear, float4(uv, 0.0, 0.0)).xyz * 2.0 - 1.0;

        if (center_depth / (DepthMultiplier * 10.0) >= DepthThreshold)
            return EnableDiffuse ? float4(0, 0, 0, 1) : float4(1, 1, 1, 1); 

        float4 spatial_result = atrous(sDenoiseTex1, uv, 2, center_depth, center_normal);
        float4 current_signal = spatial_result;

        if (!EnableTemporal)
            return current_signal;
        
        float2 velocity = GetVelocityFromClosestFragment(uv);
        float confidence = Confidence(uv, velocity);

        float2 reprojected_uv = uv + velocity;
        
        float4 history_signal = GetLod(sHistory, float4(reprojected_uv, 0, 0));
        
        float current_depth = getDepth(uv);
        float history_depth = getDepth(reprojected_uv);
        
        bool valid_history = all(saturate(reprojected_uv) == reprojected_uv) && FRAME_COUNT > 1 && abs(history_depth - current_depth) < 0.01;

        if (!valid_history)
            return current_signal;

        float4 sig_min, sig_max;
        ComputeNeighborhoodMinMax(sDenoiseTex1, uv, sig_min, sig_max);
        float4 clipped_history = clamp(history_signal, sig_min, sig_max);
        float4 temporal_signal = lerp(current_signal, clipped_history, 0.9 * confidence);

        float trust_factor = ComputeTrustFactor(velocity * BUFFER_SCREEN_SIZE);
        if (trust_factor < 1.0)
        {
            float4 blurred_signal = current_signal;
            const int blur_samples = 5;
            [unroll]
            for (int i = 1; i < blur_samples; i++)
            {
                float t = (float) i / (float) (blur_samples - 1);
                float2 blur_coord = uv - velocity * 0.5 * t;
                if (all(saturate(blur_coord) == blur_coord))
                    blurred_signal += GetLod(sDenoiseTex1, float4(blur_coord,0,0));
            }
            blurred_signal /= (float) blur_samples;
            temporal_signal = lerp(blurred_signal, temporal_signal, trust_factor);
        }
        return temporal_signal;
    }
    
    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outHistory : SV_Target)
    {
        outHistory = GetLod(sTemp, uv);
    }

    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);
        float depth = getDepth(uv);
        float4 aoData = GetLod(sTemp, uv);

        if (ViewMode == 0) // Normal
        {
            if (depth >= DepthThreshold || depth < 0.0001)
                return originalColor;
            
            half visibility = aoData.a;
            float3 diffuseLight = aoData.rgb;

            float occlusion = 1.0 - visibility;
            occlusion = saturate(occlusion * Intensity);
            float fade = saturate(1.0 - smoothstep(0.95, 1.0, depth));
            occlusion *= fade;
            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);

            if (EnableDiffuse)
            {
                originalColor.rgb += diffuseLight * DiffuseIntensity * fade;
            }

            return originalColor;
        }
        else if (ViewMode == 1)
            return float4(GetLod(sNormalEdges, uv).rgb, 1.0);
        else if (ViewMode == 2)
        {
            float view_depth = GetLod(sDepth, uv).r / (DepthMultiplier * 10.0);
            return float4(saturate(view_depth.rrr), 1.0);
        }
        else if (ViewMode == 3)
        {
            return float4(aoData.aaa, 1.0);
        }
        else if (ViewMode == 4)
            return float4(aoData.aaa, 1.0); 
        else if (ViewMode == 5)
            return EnableTemporal ? float4(aoData.aaa, 1.0) : float4(0, 1, 0, 1);
        else if (ViewMode == 6)
            return float4(GetLod(sNormalEdges, uv).aaa, 1.0);
        else if (ViewMode == 7) 
            return float4(aoData.rgb * DiffuseIntensity, 1.0);

        return originalColor;
    }

    technique Barbatos_XeGTAO
    {
        pass ViewDepth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ViewDepth;
            RenderTarget = DepthT;
        }
        pass NormalsEdges
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_NormalsEdges;
            RenderTarget = texNormalEdges;
            ClearRenderTargets = true;
        }
        pass GTAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GTAO_Main;
            RenderTarget = AO;
            ClearRenderTargets = true;
        }
        pass Prepare_Denoise
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_PrepareDenoise;
            RenderTarget = DenoiseT;
            ClearRenderTargets = true;
        }
        pass Denoise_0
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass0;
            RenderTarget = DenoiseTex0;
            ClearRenderTargets = true;
        }
        pass Denoise_1
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass1;
            RenderTarget = DenoiseTex1;
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoiseAndAccumulate;
            RenderTarget = TempT;
            ClearRenderTargets = true;
        }
        pass History
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HistoryT;
            ClearRenderTargets = true;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
