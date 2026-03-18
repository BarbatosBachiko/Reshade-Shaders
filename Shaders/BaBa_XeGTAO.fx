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
// Version:        (see XeGTAO.h)                                                                                   (='.'=)
// Details:        https://github.com/GameTechDev/XeGTAO                                                            (")_(")
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------|
| ::           Barbatos XeGTAO               :: |
|-----------------------------------------------|
| Version: 1.5                                  |
| Author: Barbatos                              |
'----------------------------------------------*/

#include "ReShade.fxh"
#include "BaBa_MV.fxh"
#include "BaBa_ColorSpace.fxh"

#ifndef USE_HILBERT_LUT
    #define USE_HILBERT_LUT 1 
#endif

static const float PI = 3.1415926535897932384626433832795;
static const float PI_HALF = 1.5707963267948966192313216916398;
static const float PI_OVER_360 = 0.00872664625997164788461845384244;
static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);

#define lpfloat float
#define lpfloat2 float2
#define lpfloat3 float3
#define lpfloat4 float4
#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0.0, 0.0))
#define getDepth(coords) ReShade::GetLinearizedDepth(coords)
#define GetColor(c) tex2D(ReShade::BackBuffer, (c).xy)
#define fmod(x, y) (frac((x) * rcp(y)) * (y))

#define bEnableDenoise 1
#define c_phi 1
#define n_phi 5
#define p_phi 1

//---------|
// :: UI ::|
//---------|

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

uniform float DepthThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Basic Settings";
    ui_label = "Sky Threshold";
> = 0.999;
uniform float4 OcclusionColor <
    ui_type = "color";
    ui_category = "Basic Settings";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform int QualityLevel <
    ui_type = "combo";
    ui_items = "Low (2 directions, 2 samples)\0Medium (3 directions, 4 samples)\0High (4 directions, 8 samples)\0Ultra (8 directions, 8 samples)\0";
    ui_category = "Performance & Quality";
    ui_label = "Quality Level";
    ui_tooltip = "Defines the number of directions and samples per direction. Higher = slower.";
> = 2;

uniform float RenderScale <
    ui_type = "drag";
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_min = 0.3; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Renders AO at a lower resolution for better performance, then upscales it.";
> = 0.5;
uniform float TemporalStability <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 0.99; ui_step = 0.01;
    ui_category = "Performance & Quality";
    ui_label = "Temporal Stability";
    ui_tooltip = "Controls the accumulation strength.\nHigher values = Less noise, potential ghosting.\nLower values = More noise, less ghosting.\nDefault: 0.80";
> = 0.90;

uniform bool EnableTemporal <
    ui_category = "Performance & Quality";
    ui_label = "Enable Temporal AA";
    ui_tooltip = "Enables temporal accumulation to reduce noise.";
> = true;
uniform bool UseStaticNoise <
    ui_category = "Performance & Quality";
    ui_label = "Use Static Noise";
    ui_tooltip = "Disables temporal noise dithering.";
> = false;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1000.0; ui_step = 10.0;
    ui_category = "Engine";
    ui_category_closed = true;
    ui_label = "Depth Multiplier";
> = 100.0;
uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
    ui_category = "Engine";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View.";
> = 60.0;

uniform bool EnableDistantRadius <
    ui_category = "Fine Tuning";
    ui_category_closed = true;
    ui_label = "Enable Distant Radius";
    ui_tooltip = "Scales the AO radius with distance to maintain visual presence on far objects.";
> = false;
uniform float FinalValuePower <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Fine Tuning";
    ui_label = "Occlusion Power";
    ui_tooltip = "Final occlusion modifier. Higher values make the occlusion contrast stronger.";
> = 0.8;

uniform float FalloffRange <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Fine Tuning";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge.";
> = 0.6;

uniform float SampleDistributionPower <
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Fine Tuning";
    ui_label = "Sample Distribution";
    ui_tooltip = "Controls how samples are distributed along the view ray.";
> = 2.0;
uniform float ThinOccluderCompensation <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Fine Tuning";
    ui_label = "Thin Occluder Fix";
    ui_tooltip = "Reduces self-occlusion on thin objects.";
> = 0.0;
uniform float RadiusMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Fine Tuning";
    ui_label = "Radius Multiplier";
    ui_tooltip = "Additional radius multiplier.";
> = 1.0;

uniform float HeightmapIntensity <
    ui_category = "Heightmap Normals";
    ui_category_closed = true;
    ui_label = "Heightmap Intensity";
    ui_tooltip = "Controls the strength of heightmap-based normal perturbation.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 200.0; ui_step = 0.1;
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

uniform int SmartSurfaceMode <
    ui_category = "Advanced";
    ui_category_closed = true;
    ui_label = "Normal Smoothing Quality";
    ui_tooltip = "Blurs the normals to prevent blocky artifacts on low-poly geometry.\n"
                 "Higher quality uses more samples but is slightly heavier.";
    ui_type = "combo";
    ui_items = "Off\0Performance\0Balanced\0Quality\0";
> = 1;

uniform float Smooth_Threshold <
    ui_category = "Advanced";
    ui_label = "Normal Smooth Threshold";
    ui_tooltip = "Limits how far the normal blur can spread. Lower values preserve hard geometry edges but reduce overall smoothing.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0AO\0Normals\0Depth\0Confidence Check\0";
    ui_category = "Debug";
    ui_category_closed = true;
    ui_label = "View Mode";
    ui_tooltip = "Selects the debug view mode.";
> = 0;
uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures  ::|
//----------------|
namespace Barbatos_XeGTAO150
{
    texture texNormalEdges
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler sNormalEdges
    {
        Texture = texNormalEdges;
        S_PC
    };
    sampler sNormalLinear
    {
        Texture = texNormalEdges;
        S_LC
    };

    texture texNormalEdges1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler sNormalEdges1
    {
        Texture = texNormalEdges1;
        S_PC
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

    texture AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sAO
    {
        Texture = AO;
        S_LC
    };

    texture History0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory0
    {
        Texture = History0;
        S_LC
    };

    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory1
    {
        Texture = History1;
        S_LC
    };

    texture RS_Prev
    {
        Width = 1;
        Height = 1;
        Format = R16F;
    };
    sampler sRS_Prev
    {
        Texture = RS_Prev;
        MagFilter = POINT;
        MinFilter = POINT;
    };

#if USE_HILBERT_LUT
    texture texHilbertLUT < source = "Barbatos_Hilbert_RGB.png"; >
    {
        Width = 64;
        Height = 64;
        Format = RGBA8;
    };
    sampler sHilbertLUT
    {
        Texture = texHilbertLUT;
        AddressU = Wrap;
        AddressV = Wrap;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };
#endif
    
    //----------------|
    // :: Functions ::|
    //----------------|
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

    void VS_SpatioTemporal_Even(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_GTAO(id, outStruct);
        bool isOdd = false;
#if __RENDERER__ >= 0xa000
        isOdd = (FRAME_COUNT & 1) != 0;
#else
        isOdd = (FRAME_COUNT % 2) != 0;
#endif
        if (isOdd)
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 1.0);
    }

    void VS_SpatioTemporal_Odd(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_GTAO(id, outStruct);
        bool isEven = false;
#if __RENDERER__ >= 0xa000
        isEven = (FRAME_COUNT & 1) == 0;
#else
        isEven = (FRAME_COUNT % 2) == 0;
#endif
        if (isEven)
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 1.0);
    }
    
#if !USE_HILBERT_LUT
    float hilbert_procedural(float2 p, int level)
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

    uint HilbertIndex_Procedural(uint x, uint y)
    {
        return (uint)hilbert_procedural(float2(x % 64, y % 64), 6);
    }
#endif
    
    float GetActiveHistory(float2 uv)
    {
#if __RENDERER__ >= 0xa000 
        bool isEven = (FRAME_COUNT & 1) == 0;
#else 
        bool isEven = (FRAME_COUNT % 2) == 0;
#endif
        return isEven ? tex2Dlod(sHistory0, float4(uv, 0, 0)).r : tex2Dlod(sHistory1, float4(uv, 0, 0)).r;
    }

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        uint index;
#if USE_HILBERT_LUT
        float4 encodedVal = tex2Dfetch(sHilbertLUT, int2(pixCoord.x % 64, pixCoord.y % 64));
        uint high_byte = (uint) (encodedVal.r * 255.0 + 0.1);
        uint low_byte = (uint) (encodedVal.g * 255.0 + 0.1);
        index = (high_byte * 256) + low_byte;
#else
        index = HilbertIndex_Procedural(pixCoord.x, pixCoord.y);
#endif

#if __RENDERER__ >= 0xa000 
        index += 288 * (temporalIndex & 63);
#else 
        index += 288 * (temporalIndex % 64);
#endif

        return frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114));
    }
    
    half XeGTAO_FastSqrt(float x)
    {
#if __RENDERER__ >= 0xa000 
        return (half) (asfloat(0x1fbd1df5 + (asint(x) >> 1)));
#else 
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

    float3 XeGTAO_CalculateNormalBGI(float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
    {
        pixLPos = normalize(pixLPos - pixCenterPos);
        pixRPos = normalize(pixRPos - pixCenterPos);
        pixTPos = normalize(pixTPos - pixCenterPos);
        pixBPos = normalize(pixBPos - pixCenterPos);
        float3 pixelNormal = cross(pixLPos, pixTPos) + cross(pixTPos, pixRPos) + cross(pixRPos, pixBPos) + cross(pixBPos, pixLPos);
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
        h00 -= h11; h10 -= h11; h20 -= h11; h01 -= h11; h21 -= h11; h02 -= h11; h12 -= h11; h22 -= h11;
        float Gx = h00 - h20 + 2.0 * h01 - 2.0 * h21 + h02 - h22;
        float Gy = h00 + 2.0 * h10 + h20 - h02 - 2.0 * h12 - h22;
        return normalize(float3(Gx * pixelWorldSize.y * pixelWorldSize.z, Gy * pixelWorldSize.x * pixelWorldSize.z, pixelWorldSize.x * pixelWorldSize.y * 16));
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

    float atrous_scalar(sampler input_sampler, float2 bufferUV, float2 viewUV, int level, float center_depth, float3 center_normal, float4 NDCToView)
    {
        float2 inv_proj_scale = float2(-NDCToView.z, NDCToView.w);
        const float2 view_step = ReShade::PixelSize * exp2(level);
        const float2 buffer_step = view_step * RenderScale;
        
        float center_val = tex2Dlod(input_sampler, float4(bufferUV, 0.0, 0.0)).r;
        float2 center_clip_pos = viewUV * 2.0 - 1.0;
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
            float2 v_uv = viewUV + atrous_offsets[i] * view_step;
            float2 b_uv = bufferUV + atrous_offsets[i] * buffer_step;
            
            float sample_val = tex2Dlod(input_sampler, float4(b_uv, 0.0, 0.0)).r;
            float sample_depth = getDepth(v_uv) * DepthMultiplier;
            float3 sample_normal = DecodeNormal(tex2Dlod(sNormalLinear, float4(b_uv, 0.0, 0.0)).xy);
            float is_valid_depth = step(sample_depth / (DepthMultiplier * 10.0), DepthThreshold);
            float2 sample_clip_pos = v_uv * 2.0 - 1.0;
            
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

    float3 UVToViewPos(float2 uv, float view_z)
    {
        float2 ndc = uv * 2.0 - 1.0;
        float tanHalfFOV = tan(FOV * PI_OVER_360);
        float aspect = BUFFER_ASPECT_RATIO;
        float2 pScale = float2(aspect * tanHalfFOV * 2.0, tanHalfFOV * 2.0);
        return float3(ndc.x * pScale.x * view_z, -ndc.y * pScale.y * view_z, view_z);
    }

    float3 CalculateNormal(float2 uv)
    {
        float3 center = UVToViewPos(uv, getDepth(uv));
        float3 offset_x = UVToViewPos(uv + float2(ReShade::PixelSize.x, 0), getDepth(uv + float2(ReShade::PixelSize.x, 0)));
        float3 offset_y = UVToViewPos(uv + float2(0, ReShade::PixelSize.y), getDepth(uv + float2(0, ReShade::PixelSize.y)));
        float3 n = cross(center - offset_x, center - offset_y);
        float lenSq = dot(n, n);
        return (lenSq > 1e-25) ? n * rsqrt(lenSq) : float3(0, 0, -1);
    }

    float ClipToAABB(float aabb_min, float aabb_max, float history_sample)
    {
        float p_clip = 0.5 * (aabb_max + aabb_min);
        float e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float v_clip = history_sample - p_clip;
        float v_unit = v_clip / e_clip;
        float a_unit = abs(v_unit);
        return (a_unit > 1.0) ? (p_clip + v_clip / a_unit) : history_sample;
    }
    
    void ComputeNeighborhoodMinMax_Scalar(sampler2D color_tex, float2 buffer_uv, float center_val, out float val_min, out float val_max)
    {
        float4 r_quad = tex2DgatherR(color_tex, buffer_uv);
        float q_min = min(min(r_quad.x, r_quad.y), min(r_quad.z, r_quad.w));
        float q_max = max(max(r_quad.x, r_quad.y), max(r_quad.z, r_quad.w));
        val_min = min(center_val, q_min);
        val_max = max(center_val, q_max);
    }
    
    float ComputeTrustFactor(float2 velocity_pixels)
    {
        const float low_threshold = 4.0;
        const float high_threshold = 20.0;
        float vel_mag = length(velocity_pixels);
        return saturate((high_threshold - vel_mag) / (high_threshold - low_threshold));
    }

    float JointBilateralUpsample(float2 uv, float highDepth)
    {
        float2 lowResUV = uv * RenderScale;
        float3 highNormal = CalculateNormal(uv);
        
        float sumAO = 0.0;
        float sumWeight = 0.0;
        
        float2 texelSize = ReShade::PixelSize;
        float2 baseUV = (floor(lowResUV / texelSize) + 0.5) * texelSize;
        
        float depth_weight_factor = 1.0 / (0.1 * highDepth + 1e-6);
        
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                float2 sampleUV = baseUV + float2(x, y) * texelSize;
                
                float sampleAO;
                #if __RENDERER__ >= 0xa000 
                if ((FRAME_COUNT & 1) == 0) sampleAO = GetLod(sHistory0, sampleUV).r;
                else sampleAO = GetLod(sHistory1, sampleUV).r;
                #else 
                if ((FRAME_COUNT % 2) == 0) sampleAO = GetLod(sHistory0, sampleUV).r;
                else sampleAO = GetLod(sHistory1, sampleUV).r;
                #endif
                
                float3 lowNormal = DecodeNormal(GetLod(sNormalEdges, sampleUV).xy);
                float lowDepth = getDepth(sampleUV / RenderScale);
                
                float wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
                float dotN = max(0.0, dot(highNormal, lowNormal));
                float wNormal = pow(dotN, 16.0);
                float wSpatial = exp(-0.5 * float(x * x + y * y));
                
                float weight = wDepth * wNormal * wSpatial;
                sumAO += sampleAO * weight;
                sumWeight += weight;
            }
        }
        if (sumWeight < 1e-6)
        {
            #if __RENDERER__ >= 0xa000 
            return ((FRAME_COUNT & 1) == 0) ? GetLod(sHistory0, lowResUV).r : GetLod(sHistory1, lowResUV).r;
            #else 
            return ((FRAME_COUNT % 2) == 0) ? GetLod(sHistory0, lowResUV).r : GetLod(sHistory1, lowResUV).r;
            #endif
        }
        return sumAO / sumWeight;
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_NormalsEdges(VS_OUTPUT input, out float4 outNormalEdges : SV_Target)
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        
        float depth01 = getDepth(viewUV);
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
        
        float dC = depth01 * DepthMultiplier;
        float dL = getDepth(viewUV + float2(-pixelSize.x, 0)) * DepthMultiplier;
        float dR = getDepth(viewUV + float2(pixelSize.x, 0)) * DepthMultiplier;
        float dT = getDepth(viewUV + float2(0, -pixelSize.y)) * DepthMultiplier;
        float dB = getDepth(viewUV + float2(0, pixelSize.y)) * DepthMultiplier;

        float3 CENTER = XeGTAO_ComputeViewspacePosition(viewUV, dC, consts);
        float3 LEFT = XeGTAO_ComputeViewspacePosition(viewUV + float2(-1, 0) * pixelSize, dL, consts);
        float3 RIGHT = XeGTAO_ComputeViewspacePosition(viewUV + float2(1, 0) * pixelSize, dR, consts);
        float3 TOP = XeGTAO_ComputeViewspacePosition(viewUV + float2(0, -1) * pixelSize, dT, consts);
        float3 BOTTOM = XeGTAO_ComputeViewspacePosition(viewUV + float2(0, 1) * pixelSize, dB, consts);
        
        float3 viewspaceNormal = XeGTAO_CalculateNormalBGI(CENTER, LEFT, RIGHT, TOP, BOTTOM);

        if (HeightmapBlendAmount > 0.001 && depth01 < HeightmapDepthThreshold)
        {
            float3 heightmapNormal = GetHeightmapNormal(viewUV);
            float3 blended = blend_normals(heightmapNormal, viewspaceNormal);
            float fade = 1.0 - smoothstep(HeightmapDepthThreshold - 0.05, HeightmapDepthThreshold, depth01);
            viewspaceNormal = normalize(lerp(viewspaceNormal, blended, HeightmapBlendAmount * fade));
        }
        
        outNormalEdges = float4(viewspaceNormal.xy * 0.5 + 0.5, 0.0, 1.0);
    }

    float4 PS_GTAO_Main(VS_OUTPUT input) : SV_Target
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        float depth = getDepth(viewUV);
        
        if (depth >= DepthThreshold)
            return float4(1.0, 1.0, 1.0, 1.0);

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

        lpfloat viewspaceZ = (lpfloat) depth * DepthMultiplier * 0.99920;
        
        float2 encodedNormal = tex2D(sNormalEdges, input.uv).xy;
        lpfloat3 viewspaceNormal = (lpfloat3) DecodeNormal(encodedNormal);

        const float3 pixCenterPos = XeGTAO_ComputeViewspacePosition(viewUV, viewspaceZ, consts);
        const lpfloat3 viewVec = (lpfloat3) normalize(-pixCenterPos);

        lpfloat sliceCount, stepsPerSlice;
        [branch]
        switch (QualityLevel)
        {
            case 0: sliceCount = 2; stepsPerSlice = 2; break;
            case 1: sliceCount = 3; stepsPerSlice = 3; break;
            case 2: sliceCount = 9; stepsPerSlice = 3; break;
            default: sliceCount = 9; stepsPerSlice = 9; break;
        }

        uint2 pixCoord = uint2(input.vpos.xy);
        uint noiseIndex = UseStaticNoise ? 0 : FRAME_COUNT;
        lpfloat2 localNoise = (lpfloat2) SpatioTemporalNoise(pixCoord, noiseIndex);
        
        float activeRadiusDistanceScale = EnableDistantRadius ? 1.0 : 0.0;
        const lpfloat baseRadius = (lpfloat) consts.EffectRadius + (viewspaceZ * activeRadiusDistanceScale);
        
        const lpfloat effectRadius = baseRadius * (lpfloat) consts.RadiusMultiplier;
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
                
                float2 sampleScreenPos0 = viewUV + sampleOffset;
                float SZ0 = getDepth(sampleScreenPos0) * DepthMultiplier;
                float3 samplePos0 = XeGTAO_ComputeViewspacePosition(sampleScreenPos0, SZ0, consts);

                float2 sampleScreenPos1 = viewUV - sampleOffset;
                float SZ1 = getDepth(sampleScreenPos1) * DepthMultiplier;
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

    float4 SpatioTemporalDenoise(VS_OUTPUT input, sampler sHistoryParams)
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        
        float rawDepth = getDepth(viewUV);
        if (rawDepth >= DepthThreshold)
            return float4(1, 1, 1, 1);

        float center_depth = rawDepth * DepthMultiplier;
        if (center_depth / (DepthMultiplier * 10.0) >= DepthThreshold)
            return float4(1, 1, 1, 1);

        float3 normal = DecodeNormal(tex2Dlod(sNormalLinear, float4(input.uv, 0.0, 0.0)).xy);
        float current_signal;
        
        if (bEnableDenoise)
            current_signal = atrous_scalar(sAO, input.uv, viewUV, 1, center_depth, normal, input.NDCToView);
        else
            current_signal = tex2Dlod(sAO, float4(input.uv, 0.0, 0.0)).r;

        float prevRenderScale = tex2Dlod(sRS_Prev, float4(0, 0, 0, 0)).x;
        if (abs(RenderScale - prevRenderScale) > 0.001 || !EnableTemporal)
            return float4(current_signal.xxxx);

        // Temporal
        float2 velocity = MV_GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        float2 reprojected_buffer_uv = reprojected_view_uv * RenderScale;
        
        float history_signal = GetLod(sHistoryParams, float4(reprojected_buffer_uv, 0, 0)).r;
        
        float history_depth = getDepth(reprojected_view_uv);
        bool valid_depth = abs(history_depth - rawDepth) < 0.02;
        bool inside_screen = all(saturate(reprojected_view_uv) == reprojected_view_uv);

        if (!inside_screen || FRAME_COUNT <= 1 || !valid_depth)
            return float4(current_signal.xxxx);

        float2 velocity_pixels = velocity * BUFFER_SCREEN_SIZE;
        float flow_magnitude = length(velocity_pixels);
        float curr_luma_ao = GetLuminance(Input2Linear(GetColor(viewUV).rgb));
        
        float confidence = MV_GetConfidenceAO(viewUV, velocity, flow_magnitude, curr_luma_ao, sB_PrevLuma);
        
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
        float trust_factor = ComputeTrustFactor(velocity_pixels);

        [branch] 
        if (trust_factor < 1.0)
        {
            temporal_val = lerp(current_signal, temporal_val, lerp(0.5, 1.0, trust_factor));
        }

        return float4(temporal_val.xxx, 1.0);
    }
    
    float2 ComputeSmoothedNormal(float2 bufferUV, float2 viewUV, float2 direction, sampler sInput)
    {
        float2 color = tex2Dlod(sInput, float4(bufferUV, 0.0, 0.0)).xy;
        float center_depth = getDepth(viewUV);

        float SNWidth = (SmartSurfaceMode == 1) ? 5.5 : ((SmartSurfaceMode == 2) ? 2.5 : 1.0);
        int SNSamples = (SmartSurfaceMode == 1) ? 1 : ((SmartSurfaceMode == 2) ? 3 : 30);

        float2 pBuffer = ReShade::PixelSize * SNWidth * direction;
        float2 pView = pBuffer / RenderScale;
        
        float T = rcp(max(Smooth_Threshold * saturate(2.0 * (1.0 - center_depth)), 0.0001));

        float2 s1 = 0.0;
        float sc = 0.0;
        
        [loop]
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            float2 sample_bufferUV = bufferUV + (pBuffer * x);
            float2 sample_viewUV = viewUV + (pView * x);

            float2 s = tex2Dlod(sInput, float4(sample_bufferUV, 0.0, 0.0)).xy;
            float s_depth = getDepth(sample_viewUV);

            float diff = length(s - color) + abs(s_depth - center_depth) * (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * Smooth_Threshold);
            diff = 1.0 - saturate(diff * T);
            
            s1 += s * diff;
            sc += diff;
        }
        return (sc > 0.0001) ? (s1 / sc) : color;
    }

    void PS_SmoothNormals_H(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        float depth = getDepth(viewUV);
        
        if (SmartSurfaceMode == 0 || depth >= DepthThreshold)
        {
            outNormal = float4(tex2Dlod(sNormalEdges, float4(input.uv, 0.0, 0.0)).xy, 0.0, 1.0);
            return;
        }
        
        float2 smoothed = ComputeSmoothedNormal(input.uv, viewUV, float2(1.0, 0.0), sNormalEdges);
        outNormal = float4(smoothed, 0.0, 1.0);
    }

    void PS_SmoothNormals_V(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        float depth = getDepth(viewUV);
        
        if (SmartSurfaceMode == 0 || depth >= DepthThreshold)
        {
            outNormal = float4(tex2Dlod(sNormalEdges1, float4(input.uv, 0.0, 0.0)).xy, 0.0, 1.0);
            return;
        }
        
        float2 smoothed = ComputeSmoothedNormal(input.uv, viewUV, float2(0.0, 1.0), sNormalEdges1);
        outNormal = float4(smoothed, 0.0, 1.0);
    }

    void PS_SpatioTemporal0(VS_OUTPUT input, out float4 outHistory : SV_Target)
    {
#if __RENDERER__ >= 0xa000 
        if ((FRAME_COUNT & 1) != 0)
#else 
        if ((FRAME_COUNT % 2) != 0)
#endif
            discard;
        outHistory = SpatioTemporalDenoise(input, sHistory1);
    }

    void PS_SpatioTemporal1(VS_OUTPUT input, out float4 outHistory : SV_Target)
    {
#if __RENDERER__ >= 0xa000 
        if ((FRAME_COUNT & 1) == 0)
#else 
        if ((FRAME_COUNT % 2) == 0)
#endif
            discard;
        outHistory = SpatioTemporalDenoise(input, sHistory0);
    }
    
    void PS_UpdateLuma(VS_OUTPUT input, out float4 outLuma : SV_Target)
    {
        float luma = GetLuminance(Input2Linear(GetColor(input.uv).rgb));
        outLuma = float4(luma.xxx, 1.0);
    }
    
    float4 PS_Output(VS_OUTPUT input) : SV_Target
    {
        float4 originalColor = GetColor(input.uv);
        float depth = getDepth(input.uv);
        float visibility;

        if (depth >= DepthThreshold || depth < 0.0001)
            return originalColor;

        if (RenderScale >= 0.999)
            visibility = GetActiveHistory(input.uv);
        else
            visibility = JointBilateralUpsample(input.uv, depth);

        float3 linearColor = Input2Linear(originalColor.rgb);
        float occlusion = saturate((1.0 - visibility) * Intensity);
        float fade = saturate(1.0 - smoothstep(0.95, 1.0, depth));
        occlusion *= fade;

        float3 occludedLinear = lerp(linearColor, Input2Linear(OcclusionColor.rgb), occlusion);

        if (ViewMode == 0) // Normal
        {
            return float4(Linear2Output(occludedLinear), originalColor.a);
        }
        else if (ViewMode == 1) // AO Only
        {
            float3 aoOnly = lerp(float3(1, 1, 1), Input2Linear(OcclusionColor.rgb), occlusion);
            return float4(Linear2Output(aoOnly), 1.0);
        }
        else if (ViewMode == 2) // Normals
        {
            float3 debugNormals = DecodeNormal(GetLod(sNormalEdges, input.uv * RenderScale).xy);
            debugNormals.x = -debugNormals.x;
            debugNormals.z = -debugNormals.z;
            return float4(debugNormals * 0.5 + 0.5, 1.0);
        }
        else if (ViewMode == 3) // Depth
        {
            float view_depth = (depth * DepthMultiplier) / (DepthMultiplier * 10.0);
            return float4(saturate(view_depth).xxx, 1.0);
        }
        else if (ViewMode == 4) // Confidence Check
        {
            float2 velocity = MV_GetVelocity(input.uv);
            float  flow_mag = length(velocity * BUFFER_SCREEN_SIZE);
            float  curr_luma_ao = GetLuminance(Input2Linear(GetColor(input.uv).rgb));
            float  conf = MV_GetConfidenceAO(input.uv, velocity, flow_mag, curr_luma_ao, sB_PrevLuma);
            return float4(conf.xxx, 1.0);
        }

        return originalColor;
    }

    void PS_SaveScale(VS_OUTPUT input, out float4 outScale : SV_Target)
    {
        outScale = float4(RenderScale, 0.0, 0.0, 1.0);
    }

    technique BaBa_XeGTAO
    <
    ui_label = "BaBa: XeGTAO";
    >
    {
        pass NormalsEdges
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_NormalsEdges;
            RenderTarget = texNormalEdges;
            ClearRenderTargets = true;
        }
        pass SmoothNormals_H
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_SmoothNormals_H;
            RenderTarget = texNormalEdges1;
        }
        pass SmoothNormals_V
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_SmoothNormals_V;
            RenderTarget = texNormalEdges;
        }
        pass GTAO_Main
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_GTAO_Main;
            RenderTarget = AO;
        }
        pass SpatioTemporal0
        {
            VertexShader = VS_SpatioTemporal_Even;
            PixelShader = PS_SpatioTemporal0;
            RenderTarget = History0;
        }
        pass SpatioTemporal1
        {
            VertexShader = VS_SpatioTemporal_Odd;
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
        pass SaveScale
        {
            VertexShader = VS_GTAO;
            PixelShader = PS_SaveScale;
            RenderTarget = RS_Prev;
        }
    }
}