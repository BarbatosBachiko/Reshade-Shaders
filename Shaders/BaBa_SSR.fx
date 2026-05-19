/*----------------------------------------------|
| | :: Barbatos SSR (Screen-Space Reflections) :: |
| |-----------------------------------------------|
| | Version: 1.5.3                                |
| | Author: Barbatos                              |
| | License: MIT                                  |
| |----------------------------------------------*/

#include ".\bb_include\bb_reshade.fxh"
#include ".\bb_include\bb_ui.fxh"
#include ".\bb_include\bb_common.fxh"
#include ".\bb_include\bb_vertex.fxh"
#include ".\bb_include\bb_depth.fxh"
#include ".\bb_include\bb_normal.fxh"
#include ".\bb_include\bb_noise.fxh"
#include ".\bb_include\bb_raytracing.fxh"
#include ".\bb_include\bb_taa.fxh"
#include ".\bb_include\bb_mv.fxh"
#include ".\bb_include\bb_colorspace.fxh"

//----------|
// :: UI :: |
//----------|
//Reflections
uniform float Intensity <
    ui_category = "Reflections";
    ui_label = "Intensity";
    ui_tooltip = "Controls the overall global strength and visibility of the screen-space reflections.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float THICKNESS_THRESHOLD <
    ui_category = "Reflections";
    ui_label = "Thickness";
    ui_tooltip = "Estimates the depth thickness of objects on screen.\n"
                 "Higher values prevent rays from passing through objects (fixing missing reflections), but values too high may cause rays to incorrectly hit objects hidden behind others.";
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.6; ui_step = 0.001;
> = 0.003;

uniform float FadeDistance <
    ui_category = "Reflections";
    ui_label = "Fade Distance";
    ui_tooltip = "Controls how far into the background the reflections will be rendered before smoothly fading out. Useful for hiding distant artifacts.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 4.999;

uniform int ReflectionMode <
    ui_category = "Reflections";
    ui_label = "Surface Target Mode";
    ui_tooltip = "Restricts ray tracing to specific geometric planes using surface normals. Useful if you only want wet floors or shiny walls.";
    ui_type = "combo";
    ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
> = 4;

uniform float OrientationThreshold <
    ui_category = "Reflections";
    ui_label = "Surface Angle Threshold";
    ui_tooltip = "Determines the angle separation used by the 'Surface Target Mode' to distinguish between walls, floors, and ceilings. Lower values are stricter.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform int RayTraceQuality <
    ui_category = "Reflections";
    ui_label = "Ray Tracing Quality";
    ui_tooltip = "Defines the amount of steps rays take to find intersections.\n"
                 "Normal: 12 steps (Good performance)\n"
                 "High: 32 steps (Better accuracy)\n"
                 "Extreme: 128 steps (Perfect accuracy, heavy performance impact)";
    ui_type = "combo";
    ui_items = "Normal (12 steps)\0High (32 steps)\0Extreme (128 steps)\0";
> = 0;

uniform bool EnableRayJitter <
    ui_category = "Reflections";
    ui_label = "Enable Ray Jitter";
    ui_tooltip = "Applies noise to the ray marching steps to trade banding artifacts for noise (which is then smoothed by the denoiser).";
> = true;

uniform float RenderResolution <
    ui_category = "Reflections";
    ui_label = "Resolution";
    ui_tooltip = "Scales down the internal resolution of the reflections to massively improve performance. 0.8 is recommended for a great balance of quality and speed.";
    ui_type = "drag";
    ui_min = 0.3; ui_max = 1.0; ui_step = 0.05;
> = 0.8;

uniform float VERTICAL_FOV <
    ui_category = "Reflections";
    ui_label = "Game FOV";
    ui_tooltip = "CRITICAL: This must match the actual vertical Field of View used in the game for reflections to align perfectly with the geometry.";
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
> = 60.0;

//Material
uniform float SurfaceGlossiness <
    ui_category = "Material";
    ui_label = "Glossiness";
    ui_tooltip = "Sets the baseline smoothness of all reflective surfaces.\n"
                 "0.0 = Mirror-like sharp reflections\n"
                 "1.0 = Completely rough, diffused reflections";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.30;

#if !ENABLE_VNDF
uniform float Anisotropy <
    ui_category = "Material";
    ui_label = "Anisotropic Stretching";
    ui_tooltip = "Stretches the specular highlight to simulate brushed metals or anisotropic materials.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 0.4; ui_step = 0.01;
> = 0.0;
#endif

uniform float Metallic <
    ui_category = "Material";
    ui_label = "Metallic";
    ui_tooltip = "Determines how much the reflection absorbs the base color of the scene underneath it (0 = Dielectric/Plastic, 1 = Pure Metal).";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float DIELECTRIC_REFLECTANCE <
    ui_category = "Material";
    ui_label = "Dielectric Reflectance (F0)";
    ui_tooltip = "Base front-facing reflectivity for non-metallic surfaces (Fresnel 0). 0.04 is the physically accurate standard for most non-metals.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform float RoughnessDetection <
    ui_category = "Material";
    ui_label = "Micro-Contrast Roughness";
    ui_tooltip = "Analyzes local screen texture contrast to automatically guess where surfaces should be rough. Higher values make detailed/noisy textures appear rougher.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.0;

uniform float SurfaceDetails <
    ui_category = "Material";
    ui_label = "Micro-Surface Details";
    ui_tooltip = "Generates a pseudo-normal map from the screen to add bumps and micro-details to the reflections.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 0.3;

uniform float SurfaceDetailsRadius <
    ui_category = "Material";
    ui_label = "Micro-Surface Details Radius";
    ui_tooltip = "Controls the radius of the micro-surface details.";
    ui_type = "drag";
    ui_min = 1.0; ui_max = 100.0;
    ui_step = 1.0;
> = 10.0;

uniform float SobelEdgeThreshold <
    ui_category = "Material";
    ui_label = "Detail Edge Threshold";
    ui_tooltip = "Ignores soft color changes when applying Micro-Surface Details. Higher values mean details are only added on harsh texture edges.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.0;

//Denoiser
uniform bool EnableSmoothing <
    ui_category = "Denoiser";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Uses data from previous frames to heavily denoise the reflections. Highly recommended to keep ON.";
> = true;

uniform bool EnableAntiSmear <
    ui_category = "Denoiser";
    ui_label = "Reduce Motion Smearing";
    ui_tooltip = "Aggressively discards old frame data when movement is detected to prevent ghosting, at the cost of slightly more noise during motion.";
> = false;

uniform int MaxFrames <
    ui_category = "Denoiser";
    ui_label = "Max Accumulation Frames";
    ui_tooltip = "How many previous frames are blended together. Higher = less noise but more ghosting. Lower = more noise but faster response.";
    ui_type = "slider";
    ui_min = 4; ui_max = 128; ui_step = 1;
> = 24;

//Color Grading
BLENDING_COMBO(BlendMode, "Blend Mode", "Select how reflections are composited over the original scene. Mode 0 is physically based (PBR).", "Color Grading", false, 0, 0)

uniform float Preserve_Scene_Highlights <
    ui_category = "Color Grading";
    ui_label = "Preserve Scene Highlights";
    ui_tooltip = "Protects bright light sources in the original scene from being incorrectly darkened or overwritten by reflections.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool Use_Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Use Kelvin Temperature Tint";
    ui_tooltip = "Overrides the standard Tint color with a physically based Kelvin temperature scale.";
> = false;

uniform float Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Temperature (Kelvin)";
    ui_tooltip = "Lower values = Warm/Orange. Higher values = Cool/Blue. 6500 is neutral daylight.";
    ui_type = "drag";
    ui_min = 1500.0;
    ui_max = 15000.0; ui_step = 10.0;
> = 6500.0;

uniform float SSR_Vibrance <
    ui_category = "Color Grading";
    ui_label = "Vibrance";
    ui_tooltip = "Boosts the color saturation of the reflections.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 10.0; ui_step = 0.1;
> = 1.0;

uniform float SSR_Contrast <
    ui_category = "Color Grading";
    ui_label = "Contrast";
    ui_tooltip = "Adjusts the contrast curve of the reflections relative to a mid-gray point.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float3 SSR_Tint <
    ui_category = "Color Grading";
    ui_label = "Global Tint";
    ui_tooltip = "Multiplies a specific color over the entire reflection.";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Shadow_Tint <
    ui_category = "Color Grading";
    ui_label = "Shadow Tint";
    ui_tooltip = "Tints the darker areas of the reflection.";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Highlight_Tint <
    ui_category = "Color Grading";
    ui_label = "Highlight Tint";
    ui_tooltip = "Tints the brighter areas of the reflection.";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float SSR_Split_Balance <
    ui_category = "Color Grading";
    ui_label = "Split Tint Balance";
    ui_tooltip = "Adjusts the threshold point between Shadow Tint and Highlight Tint.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0; ui_step = 0.01;
> = 0.5;

//Advanced
uniform int SmartSurfaceMode <
    ui_category = "Advanced";
    ui_label = "Normal Smoothing Quality";
    ui_tooltip = "Blurs the generated depth normals to prevent blocky artifacts on low-poly geometry.\n"
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

uniform float Vegetation_Protection <
    ui_category = "Advanced";
    ui_label = "Vegetation Masking";
    ui_tooltip = "Detects highly erratic depth changes (like leaves or grass) and forcefully disables reflections there to prevent noise/flickering.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float EDGE_MASK_THRESHOLD <
    ui_category = "Advanced";
    ui_label = "Geometry Edge Masking";
    ui_tooltip = "Fades out reflections tightly around the edges of objects to hide screen-space tracing errors where depth discontinuities occur.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform float GeoCorrectionIntensity <
    ui_category = "Advanced";
    ui_label = "Geometric Correction Intensity";
    ui_tooltip = "Applies a normal bump correction based on gathered texture edges.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 0.0;

uniform int ViewMode <
    ui_category = "Advanced";
    ui_label = "Debug View";
    ui_tooltip = "Visualize specific render passes to aid in tweaking and debugging.";
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0Motion Vectors\0Motion Confidence\0";
> = 0;

#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 0
#endif

#ifndef ENABLE_VNDF
#define ENABLE_VNDF 1
#endif

#ifndef ENABLE_RAY_FALLBACK
#define ENABLE_RAY_FALLBACK 0
#endif

// SSR-specific defines
#define EnableTAAUpscaling 1

static const float2 TAA_Offsets[5] =
{
    float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1)
};

//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_SSR152
{
    texture Normal
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
        MipLevels = 6;
    };

    sampler sNormal
    {
        Texture = Normal;
        MipFilter = LINEAR;
    };

    texture Normal1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal1
    {
        Texture = Normal1;
    };
    
    texture RayData
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sRayData
    {
        Texture = RayData;
        AddressU = Clamp;
        AddressV = Clamp;
        MipFilter = POINT;
        MagFilter = POINT;
        MinFilter = POINT;
    };

    texture Reflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
        MipLevels = 6;
    };
    sampler sReflection
    {
        Texture = Reflection;
        AddressU = Clamp;
        AddressV = Clamp;
        MipFilter = LINEAR;
    };

    texture History0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sHistory0
    {
        Texture = History0;
    };

    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sHistory1
    {
        Texture = History1;
    };

    texture TexColorCopy
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
        MipLevels = 8;
    };
    sampler sTexColorCopy
    {
        Texture = TexColorCopy;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
    };

    texture TexBlueNoise < source = "SS_BN3.png"; >
    {
        Width = 1024;
        Height = 1024;
        Format = RGBA8;
    };
    sampler sTexBlueNoise
    {
        Texture = TexBlueNoise;
        AddressU = Repeat;
        AddressV = Repeat;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
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
	
    //-------------|
    // :: Utility::|
    //-------------|

    void VS_Barbatos_SSR(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, VERTICAL_FOV);
    }

    void VS_Accumulate0(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, VERTICAL_FOV);
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
    }
    
    void VS_Accumulate1(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, VERTICAL_FOV);
        if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
    }

    //------------|
    // :: Noise ::|
    //------------|
    float GetSpatialTemporalNoise(float2 pos)
    {
        return N_GetSpatialTemporalNoise(pos, sTexBlueNoise, FRAME_COUNT);
    }
    
    float2 ConcentricSquareMapping(float2 u)
    {
        return N_ConcentricSquareMapping(u);
    }

    //------------------------------------|
    // :: View Space & Normal Functions ::|
    //------------------------------------|

    float4 SampleGBuffer(float2 uv)
    {
        return GetLod(sNormal, uv);
    }

    float4 ComputeSmoothedNormal(float2 uv, float2 direction, sampler sInput)
    {
        return NM_ComputeSmoothedNormal(uv, direction, sInput, SmartSurfaceMode, Smooth_Threshold, FAR_PLANE);
    }

    //-------------------|
    // :: Ray Tracing  ::|
    //-------------------|
    float GetThickness(float2 uv, float3 normal, float3 viewDir, float depth)
    {
        return RT_GetThickness(uv, normal, viewDir, depth, THICKNESS_THRESHOLD);
    }

    HitResult TraceRay2D(Ray r, int num_steps, float max_dist, float2 pScale, float jitter, float geoThickness)
    {
        return RT_TraceRay2D(r, num_steps, max_dist, pScale, jitter, geoThickness);
    }

    //---------------|
    // :: Glossy  :: |
    //---------------|

    float3 GetGlossySample(float2 sample_uv, float2 pixel_uv, float local_roughness, float3 n, float2 pScale)
    {
        float netRoughness = saturate(SurfaceGlossiness + (local_roughness * RoughnessDetection));
        if (netRoughness <= 0.001)
            return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, 0)).rgb;
#if ENABLE_VNDF
        float mipLevel = netRoughness * 4.0;
        return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, mipLevel)).rgb;
#else
        float specularPower = exp2(10.0 * (1.0 - netRoughness) + 1.0);
        float coneTheta = RT_SpecularPowerToConeAngle(specularPower) * 0.5;
    
        float2 screen_size = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 deltaP = (sample_uv - pixel_uv) * screen_size;
        float adjacentLength = length(deltaP);
        float oppositeLength = RT_IsoscelesTriangleOpposite(adjacentLength, coneTheta);
        float incircleSize = RT_IsoscelesTriangleInRadius(oppositeLength, adjacentLength);

        float rawMip = log2(max(1.0, incircleSize));
        float mipLevel = clamp(rawMip - 1.5, 0.0, 4.0);
    
        float2 blurRadiusUV = incircleSize * bb::PixelSize;

        float2 virtual_vpos = pixel_uv * screen_size;
        uint pixelIndex = uint(virtual_vpos.y * BUFFER_WIDTH + virtual_vpos.x);
        uint perFrameSeedBase = uint(FRAME_COUNT);
        float3 blueNoiseSeed = float3(
            frac(pixelIndex * 0.1031),
            frac(pixelIndex * 0.11369),
            frac(pixelIndex * 0.13787)
        );
        float3 seq = float3(
            float2(3242174889u * perFrameSeedBase, 2447445414u * perFrameSeedBase) / 4294967296.0,
            float(2654435769u * perFrameSeedBase) / 4294967296.0
        );
        float3 rand = 2.0 * abs(frac(seq + blueNoiseSeed) - 0.5);
        float2 rand_noise = rand.xy;

        float anisoScaleX = 1.0;
        float anisoScaleY = 1.0;
        float sinRot = 0.0;
        float cosRot = 1.0;
        bool useAnisotropy = (Anisotropy > 0.01);
        if (useAnisotropy)
        {
            anisoScaleX = 1.0 + (Anisotropy * 15.0);
            anisoScaleY = 1.0 / (1.0 + Anisotropy * 2.0);
            float NdotUp = dot(n, float3(0, 0, 0));
            float3 refAxis = abs(NdotUp) < 0.9 ? float3(0, 0, 1) : float3(0, 0, 1);
            float3 tangentUp = normalize(refAxis - n * dot(n, refAxis));
            float2 screenTangent = float2(tangentUp.x / max(pScale.x, 1e-6),
                                         -tangentUp.y / max(pScale.y, 1e-6));
            float angle = atan2(screenTangent.y, screenTangent.x);
            sincos(angle, sinRot, cosRot);
        }

        float2 offset = ConcentricSquareMapping(rand_noise);
        float2 bn_uv = virtual_vpos / 1024.0;
        float2 golden_offset = float2(0.61803398875, 0.73205080757) * fmod((float)FRAME_COUNT, 64.0);
        bn_uv += golden_offset;
        float blue_noise_val = tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0)).r;
        float scrambleAngle = blue_noise_val * 6.283185307;
        float s_scram = 0.0, c_scram = 1.0;
        sincos(scrambleAngle, s_scram, c_scram);
        float2 scrambledOffset = float2(0.0, 0.0);
        scrambledOffset.x = offset.x * c_scram - offset.y * s_scram;
        scrambledOffset.y = offset.x * s_scram + offset.y * c_scram;
        offset = scrambledOffset;

        if (useAnisotropy)
        {
            offset.x *= anisoScaleX;
            offset.y *= anisoScaleY;
            float2 rotatedOffset = float2(0.0, 0.0);
            rotatedOffset.x = offset.x * cosRot - offset.y * sinRot;
            rotatedOffset.y = offset.x * sinRot + offset.y * cosRot;
            offset = rotatedOffset;
        }

        return tex2Dlod(sTexColorCopy, float4(sample_uv + offset * blurRadiusUV, 0, mipLevel)).rgb;
#endif
    }
    
    //------------|
    // :: TAA  :: |
    //------------|

    float4 GetActiveHistory(float2 uv)
    {
        return (fmod((float) FRAME_COUNT, 2.0) < 0.5) ?
            GetLod(sHistory0, uv) : GetLod(sHistory1, uv);
    }

    float4 JointBilateralUpsample(float2 uv, float highDepth, float2 pScale)
    {
        return TAA_JointBilateralUpsample(uv, highDepth, pScale, RenderResolution, sHistory0, sNormal);
    }

    float4 ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        if (any(input.uv > RenderResolution))
            discard;

        float2 viewUV = input.uv / RenderResolution;
        float depth = GetDepth(viewUV);
        if (depth >= 0.999)
            return 0.0;

        float4 current_reflection = GetLod(sReflection, input.uv);

        if (!EnableSmoothing)
            return current_reflection;

        float2 velocity = MV_GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;

        if (any(saturate(reprojected_view_uv) != reprojected_view_uv) || FRAME_COUNT <= 1)
            return current_reflection;

        float2 reprojected_buffer_uv = reprojected_view_uv * RenderResolution;
        
        float4 history_reflection = TAA_SampleHistoryCatmullRom(sHistoryParams, reprojected_buffer_uv, float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        
        float3 current_compressed = TAA_Compress4(current_reflection).rgb;
        float3 current_ycocg = RGBToYCoCg(current_compressed);
        float4 current_c = float4(current_ycocg, current_reflection.a);
        
        float3 history_compressed = TAA_Compress4(history_reflection).rgb;
        float3 history_ycocg = RGBToYCoCg(history_compressed);
        
        float2 lowres_px = bb::PixelSize;
        float4 color_min, color_max;
        TAA_ComputeNeighborhoodVariance(sReflection, input.uv, current_reflection, lowres_px, color_min, color_max);

        float raw_confidence = saturate(MV_GetConfidence(viewUV));
        float relax_amount = 0.15 * raw_confidence;
        color_min.rgb -= relax_amount;
        color_max.rgb += relax_amount;

        float3 clipped_history_ycocg = TAA_ClipToAABB(color_min.rgb, color_max.rgb, history_ycocg);
        
        float clamp_distance = length(clipped_history_ycocg - history_ycocg);
        float blend_adapt = saturate(1.0 - clamp_distance * 2.0); 
        
        float frames = max(1.0, (float)MaxFrames);
        float blendVal = raw_confidence * (frames / (frames + 1.0));
        float final_feedback = blendVal * lerp(0.8, 1.0, blend_adapt);

        if (EnableAntiSmear)
        {
            float motion_factor = saturate(length(velocity) * 100.0);
            final_feedback = lerp(final_feedback, 0.0, motion_factor * 0.5);
        }

        float prevRenderResolution = tex2Dlod(sRS_Prev, float4(0, 0, 0, 0)).x;
        if (abs(RenderResolution - prevRenderResolution) > 0.001)
            final_feedback = 0.0;

        float3 result_ycocg = lerp(current_ycocg, clipped_history_ycocg, final_feedback);
        float3 result_compressed = YCoCgToRGB(result_ycocg);
        
        float result_alpha = lerp(current_reflection.a, history_reflection.a, blendVal);

        return float4(TAA_Resolve4(float4(result_compressed, 0.0)).rgb, result_alpha);
    }
    
    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
        {
            outNormal = float4(0.0, 0.0, 1.0, 1.0);
            return;
        }
        
        float3 normal = NM_CalculateNormal(input.uv, input.pScale);
        if (Vegetation_Protection > 0.0)
        {
            float2 p = bb::PixelSize * 1.5;
            float dX = abs(GetDepth(input.uv + float2(p.x, 0)) - depth);
            float dY = abs(GetDepth(input.uv + float2(0, p.y)) - depth);
            float dX_inv = abs(GetDepth(input.uv - float2(p.x, 0)) - depth);
            float dY_inv = abs(GetDepth(input.uv - float2(0, p.y)) - depth);
            float depthNoise = dX + dY + dX_inv + dY_inv;
            float threshold = (1.0 - Vegetation_Protection) * 0.05;
            if (depthNoise > threshold)
            {
                normal = float3(0.0, 0.0, -1.0);
            }
        }

        outNormal = float4(normal, depth);
    }
    
    void PS_SmoothNormals_H(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 centerNormal = GetLod(sNormal, input.uv);
        if (centerNormal.a >= 0.999 || SmartSurfaceMode == 0)
        {
            outNormal = centerNormal;
            return;
        }
        
        outNormal = ComputeSmoothedNormal(input.uv, float2(1, 0), sNormal);
    }

    void PS_MaterialResolve(VS_OUTPUT input, out float4 outNormal : SV_Target0, out float4 outColor : SV_Target1)
    {
        float3 baseColor = GetColor(input.uv).rgb;
        float4 centerNormal = GetLod(sNormal1, input.uv);
        float4 smoothed = centerNormal;
        
        if (SmartSurfaceMode != 0 && centerNormal.a < 0.999)
        {
            smoothed = ComputeSmoothedNormal(input.uv, float2(0, 1), sNormal1);
        }
        
        float depth = smoothed.a;
        float3 normal = smoothed.rgb;
        float roughness = 0.0;

        // Scharr Filter
        if (SurfaceDetails > 0.0 || GeoCorrectionIntensity > 0.0 || RoughnessDetection > 0.0)
        {
            float4 pTL = tex2Dgather(bb::BackBuffer, input.uv - bb::PixelSize * 0.5, 1);
            float4 pTR = tex2Dgather(bb::BackBuffer, input.uv + float2(bb::PixelSize.x, -bb::PixelSize.y) * 0.5, 1);
            float4 pBL = tex2Dgather(bb::BackBuffer, input.uv + float2(-bb::PixelSize.x, bb::PixelSize.y) * 0.5, 1);
            float4 pBR = tex2Dgather(bb::BackBuffer, input.uv + bb::PixelSize * 0.5, 1);

            float Gx = (3.0 * pTR.z + 10.0 * pBR.z + 3.0 * pBR.y) - (3.0 * pTL.w + 10.0 * pTL.x + 3.0 * pBL.x);
            float Gy = (3.0 * pBL.x + 10.0 * pBL.y + 3.0 * pBR.y) - (3.0 * pTL.w + 10.0 * pTL.z + 3.0 * pTR.z);
            
            Gx *= 0.25; 
            Gy *= 0.25;

            if (RoughnessDetection > 0.0)
            {
                float gradientMagnitude = sqrt(Gx * Gx + Gy * Gy);
                roughness = saturate(gradientMagnitude * max(SurfaceDetails, 0.5) * 5.0);
            }

            // Apply Bump Mapping
            if (SurfaceDetails > 0.0 && (Gx * Gx + Gy * Gy) >= (SobelEdgeThreshold * SobelEdgeThreshold))
            {
                float2 slope = float2(Gx, Gy) * SurfaceDetails;
                float3 up = abs(normal.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
                float3 T = normalize(cross(up, normal));
                float3 B = cross(normal, T);
                normal = normalize(normal + T * slope.x - B * slope.y);
            }

            // Apply Geometric Correction
            if (GeoCorrectionIntensity != 0.0)
            {
                float3 bumpNormal = normalize(float3(Gx, Gy, 1.0));
                normal = normalize(normal + bumpNormal * GeoCorrectionIntensity);
            }
        }

        normal.x = -normal.x;
        outNormal = float4(normal, depth);
        outColor  = float4(baseColor, roughness);
    }

    void PS_TraceRays(VS_OUTPUT input, out float4 outRayData : SV_Target)
    {
        float2 scaled_uv = input.uv / RenderResolution;
        float depth = GetDepth(scaled_uv);

        if (EnableTAAUpscaling)
        {
            const float2 jitterOffsets[8] = {
                float2(0.125, -0.375), float2(-0.125, 0.375), float2(-0.375, -0.125), float2(0.375, 0.125),
                float2(0.375, -0.375), float2(-0.375, 0.375), float2(0.125, 0.125), float2(-0.125, -0.125)
            };
            uint jitter_idx = uint(fmod((float)FRAME_COUNT, 8.0));
            scaled_uv += jitterOffsets[jitter_idx] * (bb::PixelSize / RenderResolution);
        }

        if (any(scaled_uv < 0.001) || any(scaled_uv > 0.999) || depth >= 1.0)
        {
            outRayData = 0.0;
            return;
        }

        float4 gbuffer = SampleGBuffer(scaled_uv);
        float2 pScale = input.pScale;
        float3 normal = normalize(gbuffer.rgb);
        float3 viewPos = UVToViewPos(scaled_uv, depth, pScale);
        float3 viewDir = -normalize(viewPos);
        float estimatedRoughness = GetLod(sTexColorCopy, scaled_uv).a;

        bool showFloor = (ReflectionMode == 0 || ReflectionMode == 3 || ReflectionMode == 4);
        bool showWall = (ReflectionMode == 1 || ReflectionMode == 4);
        bool showCeil = (ReflectionMode == 2 || ReflectionMode == 3 || ReflectionMode == 4);

        float orientationIntensity = 0.0;
        if (normal.y > OrientationThreshold && showFloor) orientationIntensity = 1.0;
        else if (normal.y < -OrientationThreshold && showCeil) orientationIntensity = 1.0;
        else if (abs(normal.y) <= OrientationThreshold && showWall) orientationIntensity = 1.0;

        if (orientationIntensity <= 0.0)
        {
            outRayData = 0.0;
            return;
        }

        float2 bn_uv = input.vpos.xy / (RenderResolution * 1024.0);
        float frame = fmod((float)FRAME_COUNT, 64.0);
        float4 bn = tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0));
        float ray_jitter = EnableRayJitter ? frac(bn.b + 0.51248584407 * frame) : 0.0;

        Ray r;
        r.origin = viewPos;

#if ENABLE_VNDF
        float netRoughness = saturate(SurfaceGlossiness + (estimatedRoughness * RoughnessDetection)) * 0.5;
        float2 golden_offset = float2(0.61803398875, 0.73205080757) * frame;
        float3 stbn_noise = tex2Dlod(sTexBlueNoise, float4(bn_uv + golden_offset, 0, 0)).rgb;
        float2 Xi = frac(stbn_noise.gb + frac(sin(dot(scaled_uv ,float2(12.9898,78.233))) * 43758.5453));
        float3 H = RT_ImportanceSampleGGX_VNDF(Xi, normal, viewDir, netRoughness);
        float3 reflectDir = reflect(-viewDir, H);
        if (dot(reflectDir, normal) < 0.0) reflectDir = reflectDir - 2.0 * dot(reflectDir, normal) * normal;
        r.direction = normalize(reflectDir);
#else
        r.direction = normalize(reflect(-viewDir, normal));
#endif

        r.origin += r.direction * (0.0005 + (depth * 0.02));

        if (dot(viewDir, normal) > 0.9 || r.direction.z < 0.0)
        {
            outRayData = 0.0;
            return;
        }

        int ray_steps = (RayTraceQuality == 2) ? 128 : ((RayTraceQuality == 1) ? 32 : 12);
        float max_dist = (RayTraceQuality == 2) ? 100.0 : ((RayTraceQuality == 1) ? 12.0 : 4.0);

        float geoThickness = GetThickness(scaled_uv, normal, normalize(viewPos), depth);
        HitResult hit = TraceRay2D(r, ray_steps, max_dist, pScale, ray_jitter, geoThickness);

        float reflectionAlpha = 0.0;
        float2 finalUV = 0.0;

        if (hit.found)
        {
            finalUV = hit.uv;
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / 10.0);
            float depthFade = saturate((FadeDistance - depth) / max(FadeDistance, 0.001));
            depthFade *= depthFade;
            float2 edgeDist = min(hit.uv, 1.0 - hit.uv);
            float screenFade = smoothstep(0.0, 0.001, min(edgeDist.x, edgeDist.y));
            reflectionAlpha = distFactor * depthFade * screenFade;

            float3 nR = SampleGBuffer(scaled_uv + float2(bb::PixelSize.x, 0)).rgb;
            float3 nD = SampleGBuffer(scaled_uv + float2(0, bb::PixelSize.y)).rgb;
            float geoMask = 1.0 - smoothstep(0.05, EDGE_MASK_THRESHOLD, length(normal - nR) + length(normal - nD));
            reflectionAlpha *= geoMask;
        }
#if ENABLE_RAY_FALLBACK
        else
        {
            float adaptiveDist = min(depth * 1.2 + 0.012, 10.0);
            finalUV = saturate(ViewPosToUV(viewPos + r.direction * adaptiveDist, pScale).xy);
            reflectionAlpha = smoothstep(0.0, 0.2, 1.0 - scaled_uv.y);
        }
#endif

        float fresnelFadeNV = max(0.0, dot(-viewDir, r.direction));
        float fresnelFade = fresnelFadeNV * fresnelFadeNV;
        float finalMask = reflectionAlpha * fresnelFade * orientationIntensity;

        // Output Geometry Hit Data (R, G) and Hit Mask (A)
        outRayData = float4(finalUV, 0.0, finalMask);
    }

    void PS_ResolveColor(VS_OUTPUT input, out float4 outReflection : SV_Target)
    {
        if (any(input.uv > RenderResolution))
        {
            outReflection = 0.0;
            return;
        }

        float2 scaled_uv = input.uv / RenderResolution;
        float4 rayData = GetLod(sRayData, input.uv);
        
        float mask = rayData.w;
        if (mask <= 0.001)
        {
            outReflection = 0.0;
            return;
        }

        float estimatedRoughness = GetLod(sTexColorCopy, scaled_uv).a;
        float netRoughness = saturate(SurfaceGlossiness + (estimatedRoughness * RoughnessDetection));
        
        float2 hitUV = rayData.xy;

        // Deferred Ray Spatial Filter
        if (netRoughness > 0.1 && EnableRayJitter)
        {
            float2 px = bb::PixelSize * max(netRoughness * 2.0, 1.0);
            
            float4 rT = GetLod(sRayData, input.uv + float2(0, -px.y));
            float4 rB = GetLod(sRayData, input.uv + float2(0, px.y));
            float4 rL = GetLod(sRayData, input.uv + float2(-px.x, 0));
            float4 rR = GetLod(sRayData, input.uv + float2(px.x, 0));
            
            float weightSum = 1.0 + rT.w + rB.w + rL.w + rR.w;
            hitUV = (hitUV + rT.xy*rT.w + rB.xy*rB.w + rL.xy*rL.w + rR.xy*rR.w) / max(weightSum, 0.001);
        }

        float3 normal = normalize(SampleGBuffer(scaled_uv).rgb);
        float3 reflectionColor = GetGlossySample(hitUV, scaled_uv, estimatedRoughness, normal, input.pScale);
        outReflection = float4(Input2Linear(reflectionColor), mask);
    }
    
    void PS_Accumulate0(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
        outBlended = 0.0;
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
            discard;
        outBlended = ComputeTAA(input, sHistory1);
    }

    void PS_Accumulate1(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
        outBlended = 0.0;
        if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
            discard;
        outBlended = ComputeTAA(input, sHistory0);
    }

    void PS_Output(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        // Debug Views
        if (ViewMode != 0)
        {
            float3 debugColor = 0.0;
            if (ViewMode == 1)
            {
                debugColor = Linear2Output(GetActiveHistory(input.uv * RenderResolution).rgb);
            }
            else if (ViewMode == 2)
            {
                float3 debugNormals = SampleGBuffer(input.uv).rgb;
                if (GetDepth(input.uv) < 0.999)
                {
                    debugNormals.x = -debugNormals.x;
                    debugNormals.z = -debugNormals.z;
                }
                debugColor = debugNormals * 0.5 + 0.5;
            }
            else if (ViewMode == 3)
            {
                debugColor = SampleGBuffer(input.uv).aaa;
            }
            else if (ViewMode == 4)
            {
                float2 m = SampleMotionVectors(input.uv);
                float v_mag = length(m) * 100.0;
                float a = atan2(m.y, m.x);
                float3 hsv_color = HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, 1.0));
                debugColor = lerp(float3(0.5, 0.5, 0.5), hsv_color, saturate(v_mag));
            }
            else if (ViewMode == 5)
            {
                float conf = MV_GetConfidence(input.uv);
                debugColor = conf.xxx;
            }

            outColor = float4(debugColor, 1.0);
            return;
        }

        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
        {
            outColor = GetColor(input.uv);
            return;
        }

        float3 rawScene = GetColor(input.uv).rgb;
        float3 scene = Input2Linear(rawScene);
        float4 gbuffer = SampleGBuffer(input.uv);
        float3 normal = gbuffer.rgb;
        
        float4 reflectionSample = JointBilateralUpsample(input.uv, depth, input.pScale);
        float3 reflectionColor = reflectionSample.rgb;
        float reflectionMask = reflectionSample.a;
        float estimatedRoughness_out = GetLod(sTexColorCopy, input.uv).a;
        float netRoughness_out = saturate(SurfaceGlossiness + (estimatedRoughness_out * RoughnessDetection));
        
        // Color Grading
        float3 tint = Use_Color_Temperature ?
            KelvinToRGB(Color_Temperature) : SSR_Tint;
        reflectionColor *= tint;
        float paper_white_norm = 80.0 / HDR_Peak_Nits;
        float mid_gray = paper_white_norm * 0.18;

        // Contrast
        reflectionColor = (reflectionColor - mid_gray) * SSR_Contrast + mid_gray;
        reflectionColor = max(0.0, reflectionColor);
        
        // Vibrance
        float reflLum = GetLuminance(reflectionColor);
        float3 chroma = reflectionColor - reflLum;
        reflectionColor = reflLum + chroma * (SSR_Vibrance);
        float luma_normalized = saturate(reflLum / (paper_white_norm * 3.0));
        float shadowCurve = 1.0 - smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);
        float highlightCurve = smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);

        // Split Tint
        float3 splitTint = shadowCurve * SSR_Shadow_Tint + highlightCurve * SSR_Highlight_Tint;
        reflectionColor = reflectionColor * splitTint;
        float splitTintAlpha = max(splitTint.r, max(splitTint.g, splitTint.b));
        reflectionMask *= saturate(splitTintAlpha);

        //Albedo
        float sceneLuma = max(GetLuminance(scene), 1e-6);
        float3 sceneTint = saturate(scene / sceneLuma);
        float highlightProtectionMask = smoothstep(paper_white_norm, paper_white_norm * 4.0, sceneLuma);
        reflectionMask *= saturate(1.0 - (highlightProtectionMask * Preserve_Scene_Highlights));

        // PBR & Blending
        float3 viewDir = -normalize(UVToViewPos(input.uv, depth, input.pScale));
        float VdotN = saturate(dot(viewDir, normal));
        
        // Metallic
        float3 metalF0 = sceneTint * max(sceneLuma, DIELECTRIC_REFLECTANCE);
        float3 f0 = lerp(DIELECTRIC_REFLECTANCE.xxx, metalF0, Metallic);
        float3 F = saturate(RT_F_Schlick(VdotN, f0));

        float3 finalColor;
        if (BlendMode == 0) // Default PBR
        {
            float validReflection = reflectionMask * saturate(Intensity);
            float3 kD = saturate(1.0 - (F * validReflection));
            float3 specularLight = reflectionColor * F * Intensity * reflectionMask;
            finalColor = (scene * kD) + specularLight;
        }
        else // Legacy Blending modes
        {
            float blendAmount = saturate(dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask * Intensity);
            finalColor = bb::Blending::Blend(BlendMode, scene, reflectionColor, blendAmount);
        }
    
        outColor = float4(Linear2Output(finalColor), 1.0);
    }
    
    void PS_SaveResolution(VS_OUTPUT input, out float4 outRes : SV_Target)
    {
        outRes = float4(RenderResolution, 0.0, 0.0, 1.0);
    }

    technique BaBa_SSR
    <
    ui_label = "BaBa: SSR";
    >
    {
        pass GenNormals
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_GenNormals;
            RenderTarget = Normal;
        }
        pass SmoothNormals_H
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_SmoothNormals_H;
            RenderTarget = Normal1;
        }
        pass MaterialResolve
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_MaterialResolve;
            RenderTarget0 = Normal;
            RenderTarget1 = TexColorCopy;
        }
        pass TraceRays
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_TraceRays;
            RenderTarget = RayData;
        }
        pass ResolveColor
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_ResolveColor;
            RenderTarget = Reflection;
        }
        pass Accumulate0
        {
            VertexShader = VS_Accumulate0;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Accumulate1;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass Output
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_Output;
        }
        pass SaveResolution
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_SaveResolution;
            RenderTarget = RS_Prev;
        }
    }
}
