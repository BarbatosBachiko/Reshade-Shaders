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
| Version: 0.9.9                                |
| Author: Barbatos                              |
'----------------------------------------------*/
#include "ReShade.fxh"

// Constants
static const float PI = 3.1415926535897932384626433832795;
static const float PI_HALF = 1.5707963267948966192313216916398;
static const float PI_OVER_360 = 0.00872664625997164788461845384244;
static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);

// Definitions
#define lpfloat float
#define lpfloat2 float2
#define lpfloat3 float3
#define lpfloat4 float4
#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4

// Sampler
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

// Helper Macros
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define getDepth(coords) ReShade::GetLinearizedDepth(coords)
#define GetColor(c) tex2D(ReShade::BackBuffer, (c).xy)
#define fmod(x, y) (frac((x) * rcp(y)) * (y))

// Config
#define bEnableDenoise 1
#define c_phi 1
#define n_phi 5
#define p_phi 1

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
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

uniform int QualityLevel <
    ui_type = "combo";
    ui_items = "Low (2 directions, 2 samples)\0Medium (3 directions, 4 samples)\0High (4 directions, 8 samples)\0Ultra (8 directions, 8 samples)\0";
    ui_category = "Performance";
    ui_label = "Quality Level";
    ui_tooltip = "Defines the number of directions and samples per direction. Higher = slower.";
> = 2;

uniform float RadiusMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 2.0;
    ui_step = 0.01;
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
    ui_min = 0.01; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Advanced Options";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge.";
> = 0.6;

uniform float SampleDistributionPower <
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0;
    ui_step = 0.1;
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

uniform bool UseStaticNoise <
    ui_category = "Advanced Options";
    ui_label = "Use Static Noise";
    ui_tooltip = "Disables temporal noise dithering.";
> = false;

uniform bool EnableTemporal <
    ui_category = "Temporal Settings";
    ui_label = "Enable Temporal AA";
    ui_tooltip = "Enables temporal accumulation to reduce noise.";
> = true;

uniform float TemporalStability <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 0.99; ui_step = 0.01;
    ui_category = "Temporal Settings";
    ui_label = "Temporal Stability";
    ui_tooltip = "Controls the accumulation strength.\nHigher values = Less noise, potential ghosting.\nLower values = More noise, less ghosting.\nDefault: 0.90";
> = 0.70;

uniform float ConfidenceSensitivity <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.1;
    ui_category = "Temporal Settings";
    ui_label = "Confidence Sensitivity";
    ui_tooltip = "Controls how aggressive the confidence rejection is.\nHigher = Stricter (less ghosting).\nLower = More relaxed.";
> = 1.0;

uniform float HeightmapIntensity <
    ui_category = "Heightmap Normals";
    ui_label = "Heightmap Intensity";
    ui_tooltip = "Controls the strength of heightmap-based normal perturbation.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 200.0;
    ui_step = 0.1;
> = 100.0;

uniform float HeightmapBlendAmount <
    ui_category = "Heightmap Normals";
    ui_label = "Heightmap Blend Amount";
    ui_tooltip = "How much to blend heightmap normals with geometric normals.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float HeightmapDepthThreshold <
    ui_category = "Heightmap Normals";
    ui_label = "Heightmap Depth Threshold";
    ui_tooltip = "Limits the heightmap normals to objects closer than this value. Useful to prevent the sky from looking bumpy.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.900;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0AO\0Normals\0Depth\0Confidence Check\0";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_tooltip = "Selects the debug view mode.";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;

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
        texture MotionVectorsTex { Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) { return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
}
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) { return GetLod(sMotVectTexVort, texcoord).rg;
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

namespace Barbatos_XeGTAO1
{
    texture texNormalEdges
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
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
        Format = R16F;
    };
    sampler sAO
    {
        Texture = AO;S_LC
    };
    texture History0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory0
    {
        Texture = History0;S_LC
    };
    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory1
    {
        Texture = History1;S_LC
    };

    //-------------|
    // :: Utility::|
    //-------------|
    
    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float4 NDCToView : TEXCOORD1;
    };

    void VS_GTAO(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ? 2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
        
        float tanHalfFOV = tan(FOV * PI_OVER_360);
        float aspect = BUFFER_ASPECT_RATIO;
        outStruct.NDCToView.xy = float2(aspect * tanHalfFOV * 2.0, -tanHalfFOV * 2.0);
        outStruct.NDCToView.zw = float2(aspect * tanHalfFOV * -1.0, tanHalfFOV);
    }

    float GetActiveHistory(float2 uv)
    {
#if __RENDERER__ >= 0xa000 // DX10+
            bool isEven = (FRAME_COUNT & 1) == 0;
#else // DX9
        bool isEven = (FRAME_COUNT % 2) == 0;
#endif

        return isEven ?
            tex2Dlod(sHistory0, float4(uv, 0, 0)).r :
            tex2Dlod(sHistory1, float4(uv, 0, 0)).r;
    }
    
    float hilbert(float2 p, int level)
    {
        float d = 0;
        [unroll]
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
#if __RENDERER__ >= 0xa000 // DX10+
            return hilbert(float2(x & 63, y & 63), 6);
#else // DX9
        return hilbert(float2(x % 64, y % 64), 6);
#endif
    }

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        uint index = HilbertIndex(pixCoord.x, pixCoord.y);
        
#if __RENDERER__ >= 0xa000 // DX10+
            index += 288 * (temporalIndex & 63);
#else // DX9
        index += 288 * (temporalIndex % 64);
#endif

        return frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114));
    }
    
    half XeGTAO_FastSqrt(float x)
    {
#if __RENDERER__ >= 0xa000 // DX10+
            return (half) (asfloat(0x1fbd1df5 + (asint(x) >> 1)));
#else // DX9
        return (half) sqrt(x);
#endif
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

    float3 XeGTAO_CalculateNormalBGI(float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
    {
        pixLPos = normalize(pixLPos - pixCenterPos);
        pixRPos = normalize(pixRPos - pixCenterPos);
        pixTPos = normalize(pixTPos - pixCenterPos);
        pixBPos = normalize(pixBPos - pixCenterPos);
        
        float3 pixelNormal = cross(pixLPos, pixTPos) +
                             cross(pixTPos, pixRPos) +
                             cross(pixRPos, pixBPos) +
                             cross(pixBPos, pixLPos);
        return normalize(pixelNormal);
    }
    
    float3 DecodeNormal(float2 enc)
    {
        float3 n;
        n.xy = enc * 2.0 - 1.0;
        n.z = -sqrt(saturate(1.0 - dot(n.xy, n.xy)));
        return n;
    }

    float3 blend_normals(float3 n1, float3 n2)
    {
        n1 += float3(0, 0, 1);
        n2 *= float3(-1, -1, 1);
        return n1 * dot(n1, n2) / n1.z - n2;
    }

    float3 ComputeHeightmapNormal(float h00, float h10, float h20, float h01, float h11, float h21, float h02, float h12, float h22, const float3 pixelWorldSize)
    {
        h00 -= h11;
        h10 -= h11;
        h20 -= h11;
        h01 -= h11;
        h21 -= h11;
        h02 -= h11;
        h12 -= h11;
        h22 -= h11;
        
        float Gx = h00 - h20 + 2.0 * h01 - 2.0 * h21 + h02 - h22;
        float Gy = h00 + 2.0 * h10 + h20 - h02 - 2.0 * h12 - h22;
        return normalize(float3(Gx * pixelWorldSize.y * pixelWorldSize.z,
                                Gy * pixelWorldSize.x * pixelWorldSize.z,
                                pixelWorldSize.x * pixelWorldSize.y * 8));
    }

    float3 GetHeightmapNormal(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float h00 = GetColor(texcoord + float2(-p.x, -p.y)).r;
        float h10 = GetColor(texcoord + float2(0, -p.y)).r;
        float h20 = GetColor(texcoord + float2(p.x, -p.y)).r;
        float h01 = GetColor(texcoord + float2(-p.x, 0)).r;
        float h11 = GetColor(texcoord).r;
        float h21 = GetColor(texcoord + float2(p.x, 0)).r;
        float h02 = GetColor(texcoord + float2(-p.x, p.y)).r;
        float h12 = GetColor(texcoord + float2(0, p.y)).r;
        float h22 = GetColor(texcoord + float2(p.x, p.y)).r;
        float3 pixelWorldSize = float3(p.x, p.y, HeightmapIntensity * 0.001);
        return ComputeHeightmapNormal(h00, h10, h20, h01, h11, h21, h02, h12, h22, pixelWorldSize);
    }

    float atrous_scalar(sampler input_sampler, float2 texcoord, int level, float center_depth, float3 center_normal, float4 NDCToView)
    {
        float2 inv_proj_scale = float2(-NDCToView.z, NDCToView.w);
        const float2 step_size = ReShade::PixelSize * exp2(level);
        
        float center_val = tex2Dlod(input_sampler, float4(texcoord, 0.0, 0.0)).r;
        float2 center_clip_pos = texcoord * 2.0 - 1.0;
        float3 center_pos;
        center_pos.xy = center_clip_pos * inv_proj_scale * center_depth;
        center_pos.y = -center_pos.y;
        center_pos.z = center_depth;

        float sum = center_val;
        float cum_w = 1.0;
        static const float2 atrous_offsets[4] = { float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1) };
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            float2 uv = texcoord + atrous_offsets[i] * step_size;
            float sample_val = tex2Dlod(input_sampler, float4(uv, 0.0, 0.0)).r;
            float sample_depth = tex2Dlod(sViewDepthLinear, float4(uv, 0.0, 0.0)).r;
            
            float3 sample_normal = DecodeNormal(tex2Dlod(sNormalLinear, float4(uv, 0.0, 0.0)).xy);
            
            float is_valid_depth = step(sample_depth / (DepthMultiplier * 10.0), DepthThreshold);
            float2 sample_clip_pos = uv * 2.0 - 1.0;
            
            float3 sample_pos;
            sample_pos.xy = sample_clip_pos * inv_proj_scale * sample_depth;
            sample_pos.y = -sample_pos.y;
            sample_pos.z = sample_depth;

            float diff_c = center_val - sample_val;
            float3 diff_p_vec = center_pos - sample_pos;
            float diff_n = dot(center_normal, sample_normal);
            float w_c = exp(-diff_c * diff_c / c_phi);
            float w_p = exp(-dot(diff_p_vec, diff_p_vec) / p_phi);
            float w_n = pow(saturate(diff_n), n_phi);
            float weight = w_c * w_n * w_p * is_valid_depth;

            sum += sample_val * weight;
            cum_w += weight;
        }
        return sum / cum_w;
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

    float PS_ViewDepth(VS_OUTPUT input) : SV_Target
    {
        return getDepth(input.uv) * DepthMultiplier;
    }

    void PS_NormalsEdges(VS_OUTPUT input, out float4 outNormalEdges : SV_Target)
    {
        float depth01 = getDepth(input.uv);
        if (depth01 >= DepthThreshold)
        {
            outNormalEdges = float4(0.5, 0.5, 0.0, 1.0);
            return;
        }

        GTAOConstants consts;
        consts.ViewportSize = BUFFER_SCREEN_SIZE;
        consts.ViewportPixelSize = ReShade::PixelSize;
        consts.NDCToViewMul = input.NDCToView.xy;
        consts.NDCToViewAdd = input.NDCToView.zw;

        float2 pixelSize = consts.ViewportPixelSize;
        float dC = tex2D(sDepth, input.uv).r;
        float dL = tex2D(sDepth, input.uv + float2(-pixelSize.x, 0)).r;
        float dR = tex2D(sDepth, input.uv + float2(pixelSize.x, 0)).r;
        float dT = tex2D(sDepth, input.uv + float2(0, -pixelSize.y)).r;
        float dB = tex2D(sDepth, input.uv + float2(0, pixelSize.y)).r;

        float3 CENTER = XeGTAO_ComputeViewspacePosition(input.uv, dC, consts);
        float3 LEFT = XeGTAO_ComputeViewspacePosition(input.uv + float2(-1, 0) * pixelSize, dL, consts);
        float3 RIGHT = XeGTAO_ComputeViewspacePosition(input.uv + float2(1, 0) * pixelSize, dR, consts);
        float3 TOP = XeGTAO_ComputeViewspacePosition(input.uv + float2(0, -1) * pixelSize, dT, consts);
        float3 BOTTOM = XeGTAO_ComputeViewspacePosition(input.uv + float2(0, 1) * pixelSize, dB, consts);
        
        float3 viewspaceNormal = XeGTAO_CalculateNormalBGI(CENTER, LEFT, RIGHT, TOP, BOTTOM);

        if (HeightmapBlendAmount > 0.001 && depth01 < HeightmapDepthThreshold)
        {
            float3 heightmapNormal = GetHeightmapNormal(input.uv);
            float3 blended = blend_normals(heightmapNormal, viewspaceNormal);
            float fade = 1.0 - smoothstep(HeightmapDepthThreshold - 0.05, HeightmapDepthThreshold, depth01);
            viewspaceNormal = normalize(lerp(viewspaceNormal, blended, HeightmapBlendAmount * fade));
        }
        
        outNormalEdges = float4(viewspaceNormal.xy * 0.5 + 0.5, 0.0, 1.0);
    }

    float4 PS_GTAO_Main(VS_OUTPUT input) : SV_Target
    {
        float depth = getDepth(input.uv);
        if (depth >= DepthThreshold)
            return float4(0, 0, 0, 1.0);
        
        GTAOConstants consts;
        consts.ViewportSize = BUFFER_SCREEN_SIZE;
        consts.ViewportPixelSize = ReShade::PixelSize;
        consts.NDCToViewMul = input.NDCToView.xy;
        consts.NDCToViewAdd = input.NDCToView.zw;
        consts.NDCToViewMul_x_PixelSize = consts.NDCToViewMul * consts.ViewportPixelSize;
        consts.EffectRadius = EffectRadius;
        consts.EffectFalloffRange = FalloffRange;
        consts.RadiusMultiplier = RadiusMultiplier;
        consts.SampleDistributionPower = SampleDistributionPower;
        consts.ThinOccluderCompensation = ThinOccluderCompensation;
        consts.FinalValuePower = FinalValuePower;

        lpfloat viewspaceZ = (lpfloat) tex2D(sDepth, input.uv).r * 0.99920;
        
        float2 encodedNormal = tex2D(sNormalEdges, input.uv).xy;
        lpfloat3 viewspaceNormal = (lpfloat3) DecodeNormal(encodedNormal);
        
        const float3 pixCenterPos = XeGTAO_ComputeViewspacePosition(input.uv, viewspaceZ, consts);
        const lpfloat3 viewVec = (lpfloat3) normalize(-pixCenterPos);

        lpfloat sliceCount, stepsPerSlice;
        [branch]
        switch (QualityLevel)
        {
            case 0:
                sliceCount = 2;
                stepsPerSlice = 2;
                break;
            case 1:
                sliceCount = 3;
                stepsPerSlice = 3;
                break;
            case 2:
                sliceCount = 9;
                stepsPerSlice = 3;
                break;
            default:
                sliceCount = 9;
                stepsPerSlice = 9;
                break;
        }

        uint2 pixCoord = uint2(input.vpos.xy);
        uint noiseIndex = UseStaticNoise ? 0 : FRAME_COUNT;
        lpfloat2 localNoise = (lpfloat2) SpatioTemporalNoise(pixCoord, noiseIndex);
        const lpfloat effectRadius = (lpfloat) consts.EffectRadius * (lpfloat) consts.RadiusMultiplier;
        const lpfloat falloffRange = (lpfloat) consts.EffectFalloffRange * effectRadius;
        const lpfloat falloffFrom = effectRadius * ((lpfloat) 1 - (lpfloat) consts.EffectFalloffRange);
        
        const lpfloat falloffMul = (lpfloat) -1.0 / falloffRange;
        const lpfloat falloffAdd = falloffFrom / falloffRange + (lpfloat) 1.0;
        lpfloat visibility = 0;
        
        const lpfloat pixelTooCloseThreshold = 1.3;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * consts.NDCToViewMul_x_PixelSize;
        lpfloat screenspaceRadius = effectRadius / (lpfloat) pixelDirRBViewspaceSizeAtCenterZ.x;
        visibility += saturate((10 - screenspaceRadius) * 0.01) * 0.5;
        const lpfloat minS = (lpfloat) pixelTooCloseThreshold / screenspaceRadius;
        
        for (lpfloat slice = 0; slice < sliceCount; slice++)
        {
            lpfloat sliceK = (slice + localNoise.x) / sliceCount;
            lpfloat phi = sliceK * PI;
            
            lpfloat sinPhi, cosPhi;
            sincos(phi, sinPhi, cosPhi);

            lpfloat2 omega = lpfloat2(cosPhi, -sinPhi) * screenspaceRadius;
            const lpfloat3 directionVec = lpfloat3(cosPhi, sinPhi, 0);
            const lpfloat3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            const lpfloat3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            lpfloat3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);
            lpfloat signNorm = (lpfloat) sign(dot(orthoDirectionVec, projectedNormalVec));
            lpfloat projectedNormalVecLength = length(projectedNormalVec);
            lpfloat cosNorm = (lpfloat) saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);
            lpfloat n = signNorm * XeGTAO_FastACos(cosNorm);
            const lpfloat lowHorizonCos0 = cos(n + PI_HALF);
            const lpfloat lowHorizonCos1 = cos(n - PI_HALF);
            lpfloat horizonCos0 = lowHorizonCos0;
            lpfloat horizonCos1 = lowHorizonCos1;

            for (lpfloat step = 0; step < stepsPerSlice; step++)
            {
                const lpfloat stepBaseNoise = lpfloat(slice + step * stepsPerSlice) * 0.6180339887498948482;
                lpfloat stepNoise = frac(localNoise.y + stepBaseNoise);
                lpfloat s = (step + stepNoise) / stepsPerSlice;
                s = (lpfloat) pow(abs(s), (lpfloat) consts.SampleDistributionPower);
                s += minS;

                lpfloat2 sampleOffset = round(s * omega) * (lpfloat2) consts.ViewportPixelSize;
                
                float2 sampleScreenPos0 = input.uv + sampleOffset;
                float SZ0 = tex2Dlod(sDepth, float4(sampleScreenPos0, 0, 0)).r;
                float3 samplePos0 = XeGTAO_ComputeViewspacePosition(sampleScreenPos0, SZ0, consts);

                float2 sampleScreenPos1 = input.uv - sampleOffset;
                float SZ1 = tex2Dlod(sDepth, float4(sampleScreenPos1, 0, 0)).r;
                float3 samplePos1 = XeGTAO_ComputeViewspacePosition(sampleScreenPos1, SZ1, consts);
                
                float3 sampleDelta0 = samplePos0 - pixCenterPos;
                float3 sampleDelta1 = samplePos1 - pixCenterPos;
                lpfloat sampleDist0 = (lpfloat) length(sampleDelta0);
                lpfloat sampleDist1 = (lpfloat) length(sampleDelta1);
                lpfloat3 sampleHorizonVec0 = (lpfloat3) (sampleDelta0 / sampleDist0);
                lpfloat3 sampleHorizonVec1 = (lpfloat3) (sampleDelta1 / sampleDist1);
                lpfloat falloffBase0 = length(lpfloat3(sampleDelta0.xy, sampleDelta0.z * (1 + consts.ThinOccluderCompensation)));
                lpfloat falloffBase1 = length(lpfloat3(sampleDelta1.xy, sampleDelta1.z * (1 + consts.ThinOccluderCompensation)));
                lpfloat weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                lpfloat weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);
                lpfloat shc0 = (lpfloat) dot(sampleHorizonVec0, viewVec);
                lpfloat shc1 = (lpfloat) dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            projectedNormalVecLength = lerp(projectedNormalVecLength, 1, 0.05);
            lpfloat h0 = -XeGTAO_FastACos((lpfloat) horizonCos1);
            lpfloat h1 = XeGTAO_FastACos((lpfloat) horizonCos0);
            lpfloat iarc0 = ((lpfloat) cosNorm + (lpfloat) 2 * (lpfloat) h0 * (lpfloat) sin(n) - (lpfloat) cos((lpfloat) 2 * (lpfloat) h0 - n)) * 0.25;
            lpfloat iarc1 = ((lpfloat) cosNorm + (lpfloat) 2 * (lpfloat) h1 * (lpfloat) sin(n) - (lpfloat) cos((lpfloat) 2 * (lpfloat) h1 - n)) * 0.25;
            lpfloat localVisibility = (lpfloat) projectedNormalVecLength * (lpfloat) (iarc0 + iarc1);
            visibility += localVisibility;
        }

        visibility *= rcp((lpfloat) sliceCount);
        visibility = pow(max(1e-5, visibility), (lpfloat) consts.FinalValuePower);
        visibility = max((lpfloat) 0.03, visibility);
        
        return float4(saturate(visibility).xxx, 1.0);
    }
    
    //---------|
    // :: TAA::|
    //---------|
    
    float ClipToAABB(float aabb_min, float aabb_max, float history_sample)
    {
        float p_clip = 0.5 * (aabb_max + aabb_min);
        float e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float v_clip = history_sample - p_clip;
        float v_unit = v_clip / e_clip;
        float a_unit = abs(v_unit);
        return (a_unit > 1.0) ? (p_clip + v_clip / a_unit) : history_sample;
    }
    
    float2 GetVelocity(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0.0;
        static const float2 offsets[5] = { float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1) };
        [unroll] 
        for (int i = 0; i < 5; i++)
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
    
    void ComputeNeighborhoodMinMax_Scalar(sampler2D color_tex, float2 texcoord, float center_val, out float val_min, out float val_max)
    {
        float4 r_quad = tex2DgatherR(color_tex, texcoord);
        float q_min = min(min(r_quad.x, r_quad.y), min(r_quad.z, r_quad.w));
        float q_max = max(max(r_quad.x, r_quad.y), max(r_quad.z, r_quad.w));
        val_min = min(center_val, q_min);
        val_max = max(center_val, q_max);
    }
    
    float ComputeTrustFactor(float2 velocity_pixels, float low_threshold = 2.0, float high_threshold = 15.0)
    {
        float vel_mag = length(velocity_pixels);
        return saturate((high_threshold - vel_mag) / (high_threshold - low_threshold));
    }
    
    float Confidence(float2 uv, float2 velocity, float flow_magnitude)
    {
        float2 prev_uv = uv + velocity;
        if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
            return 0.0;
        float curr_luma = GetLuminance(GetColor(uv).rgb);
        float prev_luma = tex2D(sB_PrevLuma, prev_uv).r;
        float luma_error = abs(curr_luma - prev_luma);
        if (flow_magnitude <= 0.5)
            return 1.0;
        float2 destination_velocity = SampleMotionVectors(prev_uv);
        float error = length(velocity - destination_velocity);
        float normalized_error = (error * ConfidenceSensitivity) / (length(velocity) + 1e-6);
        float length_conf = rcp(flow_magnitude * 0.02 + 1.0);
        float consistency_conf = rcp(normalized_error + 1.0);
        float photometric_conf = exp(-luma_error * 1.5);
        return consistency_conf * length_conf * photometric_conf;
    }

    float4 SpatioTemporalDenoise(VS_OUTPUT input, sampler sHistoryParams)
    {
        float rawDepth = getDepth(input.uv);
        if (rawDepth >= DepthThreshold)
            return float4(1, 1, 1, 1);
        float center_depth = tex2Dlod(sViewDepthLinear, float4(input.uv, 0.0, 0.0)).r;
        if (center_depth / (DepthMultiplier * 10.0) >= DepthThreshold)
            return float4(1, 1, 1, 1);
        
        float3 normal = DecodeNormal(tex2Dlod(sNormalLinear, float4(input.uv, 0.0, 0.0)).xy);
        
        float current_signal;
        if (bEnableDenoise)
            current_signal = atrous_scalar(sAO, input.uv, 1, center_depth, normal, input.NDCToView);
        else
            current_signal = tex2Dlod(sAO, float4(input.uv, 0.0, 0.0)).r;
        if (!EnableTemporal)
            return float4(current_signal.xxxx);
        // Temporal
        float2 velocity = GetVelocity(input.uv);
        float2 reprojected_uv = input.uv + velocity;
        float history_signal = GetLod(sHistoryParams, float4(reprojected_uv, 0, 0)).r;
        
        float current_depth = getDepth(input.uv);
        float history_depth = getDepth(reprojected_uv);
        bool valid_depth = abs(history_depth - current_depth) < 0.02;
        bool inside_screen = all(saturate(reprojected_uv) == reprojected_uv);
        if (!inside_screen || FRAME_COUNT <= 1 || !valid_depth)
            return float4(current_signal.xxxx);
        float2 velocity_pixels = velocity * BUFFER_SCREEN_SIZE;
        float flow_magnitude = length(velocity_pixels);

        float confidence = Confidence(input.uv, velocity, flow_magnitude);
        
        float val_min, val_max;
        ComputeNeighborhoodMinMax_Scalar(sAO, input.uv, current_signal, val_min, val_max);
        
        float box_size = val_max - val_min;
        val_min -= box_size * 0.5;
        val_max += box_size * 0.5;

        float clipped_history = ClipToAABB(val_min, val_max, history_signal);
        
        float velocity_factor = saturate(flow_magnitude * 0.25);
        float final_history = lerp(history_signal, clipped_history, velocity_factor);

        float compressed_confidence = saturate(confidence + log2(2.0 - confidence) * 0.5);
        float blendFactor = max(0.1, TemporalStability * compressed_confidence);
        float temporal_val = lerp(current_signal, final_history, blendFactor);
        float trust_factor = ComputeTrustFactor(velocity_pixels, 4.0, 20.0);
        [branch] 
        if (trust_factor < 1.0)
        {
            temporal_val = lerp(current_signal, temporal_val, lerp(0.5, 1.0, trust_factor));
        }

        return float4(temporal_val.xxx, 1.0);
    }
    
    void PS_SpatioTemporal0(VS_OUTPUT input, out float4 outHistory : SV_Target)
    {
#if __RENDERER__ >= 0xa000 // DX10+
            if ((FRAME_COUNT & 1) != 0)
#else // DX9
        if ((FRAME_COUNT % 2) != 0)
#endif
            discard;

        outHistory = SpatioTemporalDenoise(input, sHistory1);
    }

    void PS_SpatioTemporal1(VS_OUTPUT input, out float4 outHistory : SV_Target)
    {
#if __RENDERER__ >= 0xa000 // DX10+
            if ((FRAME_COUNT & 1) == 0)
#else // DX9
        if ((FRAME_COUNT % 2) == 0)
#endif
            discard;

        outHistory = SpatioTemporalDenoise(input, sHistory0);
    }
    
    void PS_UpdateLuma(VS_OUTPUT input, out float4 outLuma : SV_Target)
    {
        float luma = GetLuminance(GetColor(input.uv).rgb);
        outLuma = float4(luma.xxx, 1.0);
    }

    float4 PS_Output(VS_OUTPUT input) : SV_Target
    {
        float4 originalColor = GetColor(input.uv);
        float depth = getDepth(input.uv);
        
        float visibility = GetActiveHistory(input.uv);

        if (ViewMode == 0) // Normal
        {
            if (depth >= DepthThreshold || depth < 0.0001)
                return originalColor;
            float occlusion = saturate((1.0 - visibility) * Intensity);
            float fade = saturate(1.0 - smoothstep(0.95, 1.0, depth));
            occlusion *= fade;
            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);
            return originalColor;
        }
        else if (ViewMode == 1) // AO Only
        {
            float occlusion = saturate((1.0 - visibility) * Intensity);
            return float4(lerp(float3(1, 1, 1), OcclusionColor.rgb, occlusion), 1.0);
        }
        else if (ViewMode == 2) // Normals
        {
            float3 n = DecodeNormal(GetLod(sNormalEdges, input.uv).xy);
            return float4(n.rgb * 0.5 + 0.5, 1.0);
        }
        else if (ViewMode == 3) // Depth
        {
            float view_depth = GetLod(sDepth, input.uv).r / (DepthMultiplier * 10.0);
            return float4(saturate(view_depth).xxx, 1.0);
        }
        else if (ViewMode == 4) // Confidence Check
        {
            float2 velocity = GetVelocity(input.uv);
            float conf = Confidence(input.uv, velocity, length(velocity * BUFFER_SCREEN_SIZE));
            return float4(conf.xxx, 1.0);
        }

        return originalColor;
    }
    
    technique Barbatos_XeGTAO
    {
        pass ViewDepth
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_ViewDepth;
            RenderTarget = DepthT;
        }
        pass NormalsEdges
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_NormalsEdges;
            RenderTarget = texNormalEdges;
            ClearRenderTargets = true;
        }
        pass GTAO
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_GTAO_Main;
            RenderTarget = AO;
            ClearRenderTargets = true;
        }
        pass SpatioTemporal0
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_SpatioTemporal0;
            RenderTarget = History0;
        }
        pass SpatioTemporal1
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_SpatioTemporal1;
            RenderTarget = History1;
        }
        pass UpdateLuma
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_UpdateLuma;
            RenderTarget = B_PrevLuma;
        }
        pass Output
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_Output;
        }
    }
}
