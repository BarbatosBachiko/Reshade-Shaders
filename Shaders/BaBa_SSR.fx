/*----------------------------------------------|
| :: Barbatos SSR (Screen-Space Reflections) :: |
|-----------------------------------------------|
| Version: 1.5.0                                |
| Author: Barbatos                              |
| License: MIT                                  |
|----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "Blending.fxh"
#include "BaBa_MV.fxh"
#include "BaBa_ColorSpace.fxh"

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
> = 0.2;

uniform float DIELECTRIC_REFLECTANCE <
    ui_category = "Material";
    ui_label = "Dielectric Reflectance (F0)";
    ui_tooltip = "Base front-facing reflectivity for non-metallic surfaces (Fresnel 0). 0.04 is the physically accurate standard for most non-metals.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.04;

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
#define ENABLE_VNDF 0
#endif

#ifndef ENABLE_RAY_FALLBACK
#define ENABLE_RAY_FALLBACK 0
#endif

// Defines & Constants
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

static const float DEG2RAD = 0.017453292;

static const float2 TAA_Offsets[5] =
{
    float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1)
};

uniform int FRAME_COUNT < source = "framecount"; >;
#define EnableTAAUpscaling 1

//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_SSR150
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
    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };

    struct Ray
    {
        float3 origin;
        float3 direction;
    };

    struct HitResult
    {
        bool found;
        float3 viewPos;
        float2 uv;
    };
    
    void VS_Barbatos_SSR(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ?
            2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        float y = tan(VERTICAL_FOV * DEG2RAD * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }

    void VS_Accumulate0(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_SSR(id, outStruct);
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
    }
    
    void VS_Accumulate1(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_SSR(id, outStruct);
        if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
    }
    
    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }

    float3 F_Schlick(float VdotH, float3 f0)
    {
        float t = 1.0 - VdotH;
        float t2 = t * t;
        return f0 + (1.0 - f0) * (t2 * t2 * t);
    }

    //------------|
    // :: Noise ::|
    //------------|
    float GetSpatialTemporalNoise(float2 pos)
    {
        float2 bn_uv = pos / 1024.0;
        float frame = fmod((float)FRAME_COUNT, 64.0);
        bn_uv += float2(0.61803398875, 0.73205080757) * frame;
        return tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0)).r;
    }
    
    float2 ConcentricSquareMapping(float2 u)
    {
        float2 ab = 2.0 * u - 1.0;
        float2 ab2 = ab * ab;
        float r, phi;
        if (ab2.x > ab2.y)
        {
            r = ab.x;
            phi = (PI / 4.0) * (ab.y / ab.x);
        }
        else
        {
            r = ab.y;
            phi = (ab.y != 0.0) ? (PI / 2.0) - (PI / 4.0) * (ab.x / ab.y) : 0.0;
        }
        
        float2 sincosPhi;
        sincos(phi, sincosPhi.y, sincosPhi.x);
        return r * sincosPhi;
    }

    //------------------------------------|
    // :: View Space & Normal Functions ::|
    //------------------------------------|
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
        float z     = GetDepth(uv) * FAR_PLANE + 1.0;
        float2 ndc  = uv * 2.0 - 1.0;
        return float3(ndc.x * pScale.x * z, -ndc.y * pScale.y * z, z);
    }

    float3 CalculateNormal(float2 uv, float2 pScale)
    {
        float2 offset = ReShade::PixelSize;
        float3 pos_c  = GetViewPosForNormal(uv,                              pScale);
        float3 pos_l  = GetViewPosForNormal(uv + float2(-offset.x,  0.0),    pScale);
        float3 pos_r  = GetViewPosForNormal(uv + float2( offset.x,  0.0),    pScale);
        float3 pos_u  = GetViewPosForNormal(uv + float2( 0.0, -offset.y),    pScale);
        float3 pos_d  = GetViewPosForNormal(uv + float2( 0.0,  offset.y),    pScale);
        float3 pos_l2 = GetViewPosForNormal(uv + float2(-2.0*offset.x, 0.0), pScale);
        float3 pos_r2 = GetViewPosForNormal(uv + float2( 2.0*offset.x, 0.0), pScale);
        float3 pos_u2 = GetViewPosForNormal(uv + float2( 0.0, -2.0*offset.y),pScale);
        float3 pos_d2 = GetViewPosForNormal(uv + float2( 0.0,  2.0*offset.y),pScale);
        float3 dl  = pos_c - pos_l;
        float3 dr  = pos_r - pos_c;
        float3 du  = pos_c - pos_u;
        float3 dd  = pos_d - pos_c;
        float3 dl2 = pos_c  - pos_l2;
        float3 dr2 = pos_r2 - pos_c;
        float3 du2 = pos_c  - pos_u2;
        float3 dd2 = pos_d2 - pos_c;

        float dxl = abs(dl.z + (dl.z - dl2.z));
        float dxr = abs(dr.z + (dr.z - dr2.z));
        float dyu = abs(du.z + (du.z - du2.z));
        float dyd = abs(dd.z + (dd.z - dd2.z));

        float3 hor = dxl < dxr ? dl : dr;
        float3 ver = dyu < dyd ? du : dd;

        float3 n = cross(hor, ver);
        n.x = -n.x;
        float lenSq = dot(n, n);
        return (lenSq > 1e-25) ? n * rsqrt(lenSq) : float3(0, 0, -1);
    }
    
    float3 BlendBump(float3 n1, float3 n2)
    {
        n1.z++;
        return n1 * dot(n1, n2) / n1.z - n2;
    }

    float3 ApplySurfaceDetails(float2 texcoord, float3 normal, float depth)
    {
        if (SurfaceDetails <= 0.0)
        {
            normal.x = -normal.x;
            return normal;
        }

        float viewDistance = max(depth * FAR_PLANE, 1.0);
        float radius = clamp(SurfaceDetailsRadius / viewDistance, 0.0, 8192.0);
        
        float2 p = ReShade::PixelSize * radius;
        float2 kernelWeight[9] =
        {
            float2(-1.0, -1.0), float2(0.0, -2.0), float2(1.0, -1.0),
            float2(-2.0, 0.0), float2(0.0, 0.0), float2(2.0, 0.0),
            float2(-1.0, 1.0), float2(0.0, 2.0), float2(1.0, 1.0)
        };
        float2 sampleOffset[9] =
        {
            float2(-1.0, -1.0), float2(0.0, -1.0), float2(1.0, -1.0),
            float2(-1.0, 0.0), float2(0.0, 0.0), float2(1.0, 0.0),
            float2(-1.0, 1.0), float2(0.0, 1.0), float2(1.0, 1.0)
        };
        float2 sobel = 0.0;
        
        for (int idx = 0; idx < 9; idx++)
        {
            float3 col = tex2Dlod(ReShade::BackBuffer, float4(sampleOffset[idx] * p + texcoord, 0, 0)).rgb;
            float luma = dot(col, float3(0.299, 0.587, 0.114));
            sobel += float2(luma, luma) * kernelWeight[idx];
        }

        float3 finalNormal = normal;
        if (SurfaceDetails > 0.0 && dot(sobel, sobel) >= (SobelEdgeThreshold * SobelEdgeThreshold))
        {
            float height = SurfaceDetails * 2.0;
            float3 bump = float3(sobel.x * height, sobel.y * height, 1.0);
            bump = normalize(bump);

            finalNormal = normalize(BlendBump(normal, bump));
        }

        finalNormal.x = -finalNormal.x;

        return finalNormal;
    }
    
    float4 ComputeSmoothedNormal(float2 uv, float2 direction, sampler sInput)
    {
        float4 color = GetLod(sInput, uv);
        float SNWidth = (SmartSurfaceMode == 1) ? 5.5 : ((SmartSurfaceMode == 2) ? 2.5 : 1.0);
        int SNSamples = (SmartSurfaceMode == 1) ? 1 : ((SmartSurfaceMode == 2) ? 3 : 30);
        float2 p = ReShade::PixelSize * SNWidth * direction;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        float4 s1 = 0.0;
        float sc = 0.0;
        
        [loop]
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            float4 s = GetLod(sInput, uv + (p * x));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T);
            s1 += s * diff;
            sc += diff;
        }
        return (sc > 0.0001) ? (s1 / sc) : color;
    }
    
    float4 SampleGBuffer(float2 uv)
    {
        return GetLod(sNormal, uv);
    }

    //-------------------|
    // :: Ray Tracing  ::|
    //-------------------|
    float GetThickness(float2 uv, float3 normal, float3 viewDir, float depth)
    {
        // Calculate angle between view and surface
        float NdotV = abs(dot(normal, -viewDir));
        //Volume Expansion Factor
        float geometricScale = 1.0 / max(NdotV, 0.2);
        //Edge Guard 
        float depthDerivative = fwidth(depth);
        float edgeMask = 1.0 - smoothstep(0.0, 0.002, depthDerivative);

        return THICKNESS_THRESHOLD * geometricScale * edgeMask;
    }

    HitResult TraceRay2D(Ray r, int num_steps, float max_dist, float2 pScale, float jitter, float geoThickness)
    {
        HitResult result;
        result.found = false;
        result.viewPos = 0.0;
        result.uv = 0.0;
        
        float3 endPos = r.origin + r.direction * max_dist;
        float2 startUV = ViewPosToUV(r.origin, pScale);
        float2 endUV = ViewPosToUV(endPos, pScale);
        float2 deltaUV = endUV - startUV;
        if (dot(deltaUV, deltaUV) < 0.0001)
            return result;
        float startK = 1.0 / r.origin.z;
        float endK = 1.0 / endPos.z;
        float deltaK = endK - startK;
        float stepSize = 1.0 / (float) num_steps;
        
        float t = stepSize * jitter;
        float2 currUV = startUV + deltaUV * t;
        float currK = startK + deltaK * t;
        float2 stepUV = deltaUV * stepSize;
        float stepK = deltaK * stepSize;
        float prevRayDepth = 1.0 / max(abs(currK - stepK), 1e-10);

        [loop]
        for (int i = 0; i < num_steps; ++i)
        {
            if (any(currUV < 0.0) || any(currUV > 1.0))
                break;
            float rayDepth = 1.0 / currK;
            float sceneDepth = GetDepth(currUV);
            float depthDiff = rayDepth - sceneDepth;
            // Adaptive thickness 
            float rayStepSizeZ = abs(rayDepth - prevRayDepth);
            float adaptiveThickness = max(geoThickness, rayStepSizeZ * 1.5);
            adaptiveThickness *= (1.0 + rayDepth * 0.2);
            if (depthDiff > 0.0 && depthDiff < adaptiveThickness)
            {
                float2 loUV = currUV - stepUV;
                float2 hiUV = currUV;
                float2 midUV;
                
                [unroll]
                for (int j = 0; j < 2; j++)
                {
                    midUV = (loUV + hiUV) * 0.5;
                    float midRayDepth = 1.0 / (currK - stepK * 0.5);
                    if (midRayDepth > GetDepth(midUV))
                        hiUV = midUV;
                    else
                        loUV = midUV;
                }

                result.found = true;
                result.uv = hiUV;
                result.viewPos = UVToViewPos(hiUV, GetDepth(hiUV), pScale);
                return result;
            }
            
            prevRayDepth = rayDepth;
            currUV += stepUV;
            currK += stepK;
        }

        return result;
    }

    //---------------|
    // :: Glossy  :: |
    //---------------|
    float GetLocalRoughness(float2 uv)
    {
        float3 center = GetColor(uv).rgb;
        float lumaC = GetLuminance(center);
        float2 p = ReShade::PixelSize;
        
        float lumaN = GetLuminance(GetColor(uv + float2(p.x, 0)).rgb);
        float lumaS = GetLuminance(GetColor(uv - float2(p.x, 0)).rgb);
        float lumaE = GetLuminance(GetColor(uv + float2(0, p.y)).rgb);
        float lumaW = GetLuminance(GetColor(uv - float2(0, p.y)).rgb);
        
        return saturate((abs(lumaN - lumaC) + abs(lumaS - lumaC) + abs(lumaE - lumaC) + abs(lumaW - lumaC)) * 10.0);
    }
    
    float specularPowerToConeAngle(float specularPower)
    {
        if (specularPower >= 4096.0)
            return 0.0;
        float exponent = rcp(specularPower + 1.0);
        return acos(clamp(pow(0.244, exponent), -1.0, 1.0));
    }
    
    float isoscelesTriangleOpposite(float adjacentLength, float coneTheta)
    {
        return 2.0 * tan(coneTheta) * adjacentLength;
    }
    
    float isoscelesTriangleInRadius(float a, float h)
    {
        float a2 = a * a;
        float fh2 = 4.0 * h * h;
        return (a * (sqrt(a2 + fh2) - a)) / max(4.0 * h, 1e-6);
    }

// VNDF Sampling
#if ENABLE_VNDF
float3 ImportanceSampleGGX_VNDF(float2 Xi, float3 N, float3 V, float roughness)
{
    float alpha = roughness * roughness;
    Xi = clamp(Xi, 0.001, 0.999);
    
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    
    float3 V_local = float3(dot(V, T), dot(V, B), dot(V, N));
    float3 V_stretch = normalize(float3(alpha * V_local.x, alpha * V_local.y, V_local.z));
    
    float3 T1 = (V_stretch.z < 0.9999) ?
        normalize(cross(V_stretch, float3(0.0, 0.0, 1.0))) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(T1, V_stretch);
    float a = 1.0 / (1.0 + V_stretch.z);
    float r = sqrt(Xi.x);
    float phi = (Xi.y < a) ?
        (Xi.y / a) * 3.14159265359 : 3.14159265359 + (Xi.y - a) / (1.0 - a) * 3.14159265359;
    float P1 = r * cos(phi);
    float P2 = r * sin(phi) * ((Xi.y < a) ? 1.0 : V_stretch.z);
    float3 N_local = P1 * T1 + P2 * T2 + sqrt(max(0.0, 1.0 - P1 * P1 - P2 * P2)) * V_stretch;
    N_local = normalize(float3(alpha * N_local.x, alpha * N_local.y, max(0.0, N_local.z)));
    return normalize(T * N_local.x + B * N_local.y + N * N_local.z);
}
#endif

    float3 GetGlossySample(float2 sample_uv, float2 pixel_uv, float local_roughness, float3 n, float2 pScale)
    {
        float netRoughness = saturate(SurfaceGlossiness + (local_roughness * RoughnessDetection));
        if (netRoughness <= 0.001)
            return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, 0)).rgb;
#if ENABLE_VNDF
        float mipLevel = netRoughness * 4.0; 
        return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, mipLevel)).rgb;
#else
        float specularPower = pow(2.0, 10.0 * (1.0 - netRoughness) + 1.0);
        float coneTheta = specularPowerToConeAngle(specularPower) * 0.5;
    
        float2 screen_size = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 deltaP = (sample_uv - pixel_uv) * screen_size;
        float adjacentLength = length(deltaP);
        float oppositeLength = isoscelesTriangleOpposite(adjacentLength, coneTheta);
        float incircleSize = isoscelesTriangleInRadius(oppositeLength, adjacentLength);

        float rawMip = log2(max(1.0, incircleSize));
        float mipLevel = clamp(rawMip - 1.5, 0.0, 4.0);
    
        float2 blurRadiusUV = incircleSize * ReShade::PixelSize;

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
        float s_scram, c_scram;
        sincos(scrambleAngle, s_scram, c_scram);
        float2 scrambledOffset;
        scrambledOffset.x = offset.x * c_scram - offset.y * s_scram;
        scrambledOffset.y = offset.x * s_scram + offset.y * c_scram;
        offset = scrambledOffset;

        if (useAnisotropy)
        {
            offset.x *= anisoScaleX;
            offset.y *= anisoScaleY;
            float2 rotatedOffset;
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

    float4 TAA_Compress(float4 color)
    {
        return float4(color.rgb / (10.0 + color.rgb), color.a);
    }

    float4 TAA_Resolve(float4 color)
    {
        return float4((color.rgb * 10.0) / max(1e-6, 1.0 - color.rgb), color.a);
    }

    float4 GetActiveHistory(float2 uv)
    {
        return (fmod((float) FRAME_COUNT, 2.0) < 0.5) ?
            GetLod(sHistory0, uv) : GetLod(sHistory1, uv);
    }

   float4 JointBilateralUpsample(float2 uv, float highDepth, float2 pScale)
    {
        float2 lowResUV = uv * RenderResolution;
        float3 highNormal = CalculateNormal(uv, pScale);

        float4 sumRefl = 0.0;
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
                float4 refl = GetActiveHistory(sampleUV);
                
                float2 fullScreenUV = sampleUV / RenderResolution;
                float4 gbuffer = SampleGBuffer(fullScreenUV);

                float3 lowNormal = gbuffer.rgb;
                float lowDepth = gbuffer.a;

                float wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
                float dotN = max(0.0, dot(highNormal, lowNormal));
                float wNormal = pow(dotN, 16.0);
                float wSpatial = exp(-0.5 * float(x * x + y * y));

                float weight = wDepth * wNormal * wSpatial;

                sumRefl += refl * weight;
                sumWeight += weight;
            }
        }

        if (sumWeight < 1e-6)
            return GetActiveHistory(lowResUV);
        
        return sumRefl / sumWeight;
    }

    float3 ClipToAABB(float3 aabb_min, float3 aabb_max, float3 history_sample)
    {
        float3 p_clip = 0.5 * (aabb_max + aabb_min);
        float3 e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float3 v_clip = history_sample - p_clip;
        float3 v_unit = v_clip / e_clip;
        float3 a_unit = abs(v_unit);
        float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
        return (ma_unit > 1.0) ? (p_clip + v_clip / ma_unit) : history_sample;
    }

    void ComputeNeighborhoodVariance(sampler sInput, float2 texcoord, float4 current_c, float2 pSize, out float4 color_min, out float4 color_max)
    {
        float4 m1 = current_c;
        float4 m2 = current_c * current_c;
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                if (x == 0 && y == 0) continue;
                float4 c = GetLod(sInput, texcoord + float2(x, y) * pSize);
                c.rgb = TAA_Compress(c).rgb;
                c.rgb = RGBToYCoCg(c.rgb); 
                
                m1 += c;
                m2 += c * c;
            }
        }
        
        m1 /= 9.0;
        m2 /= 9.0;
        
        float4 sigma = sqrt(abs(m2 - m1 * m1));
        float gamma = 1.25;
        color_min = m1 - gamma * sigma;
        color_max = m1 + gamma * sigma;
    }

    float4 SampleHistoryCatmullRom(sampler sInput, float2 uv, float2 texSize)
    {
        float2 samplePos = uv * texSize;
        float2 texPos1 = floor(samplePos - 0.5) + 0.5;
        float2 f = samplePos - texPos1;
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);

        float2 w12 = w1 + w2;
        float2 offset12 = w2 / (w1 + w2);

        float2 texPos0 = texPos1 - 1.0;
        float2 texPos3 = texPos1 + 2.0;
        float2 texPos12 = texPos1 + offset12;

        texPos0 /= texSize;
        texPos3 /= texSize;
        texPos12 /= texSize;

        float4 result = 0.0;
        result += GetLod(sInput, float2(texPos0.x, texPos0.y)) * w0.x * w0.y;
        result += GetLod(sInput, float2(texPos12.x, texPos0.y)) * w12.x * w0.y;
        result += GetLod(sInput, float2(texPos3.x, texPos0.y)) * w3.x * w0.y;

        result += GetLod(sInput, float2(texPos0.x, texPos12.y)) * w0.x * w12.y;
        result += GetLod(sInput, float2(texPos12.x, texPos12.y)) * w12.x * w12.y;
        result += GetLod(sInput, float2(texPos3.x, texPos12.y)) * w3.x * w12.y;

        result += GetLod(sInput, float2(texPos0.x, texPos3.y)) * w0.x * w3.y;
        result += GetLod(sInput, float2(texPos12.x, texPos3.y)) * w12.x * w3.y;
        result += GetLod(sInput, float2(texPos3.x, texPos3.y)) * w3.x * w3.y;

        return max(result, 0.0);
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
        
        float4 history_reflection = SampleHistoryCatmullRom(sHistoryParams, reprojected_buffer_uv, float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        
        float3 current_compressed = TAA_Compress(current_reflection).rgb;
        float3 current_ycocg = RGBToYCoCg(current_compressed);
        float4 current_c = float4(current_ycocg, current_reflection.a);
        
        float3 history_compressed = TAA_Compress(history_reflection).rgb;
        float3 history_ycocg = RGBToYCoCg(history_compressed);
        
        float2 lowres_px = ReShade::PixelSize;
        float4 color_min, color_max;
        ComputeNeighborhoodVariance(sReflection, input.uv, current_c, lowres_px, color_min, color_max);

        float raw_confidence = saturate(MV_GetConfidence(viewUV));
        float relax_amount = 0.15 * raw_confidence;
        color_min.rgb -= relax_amount;
        color_max.rgb += relax_amount;

        float3 clipped_history_ycocg = ClipToAABB(color_min.rgb, color_max.rgb, history_ycocg);
        
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

        return float4(TAA_Resolve(float4(result_compressed, 0.0)).rgb, result_alpha);
    }
    
    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_CopyColor(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        float localRoughness = (RoughnessDetection > 0.0) ?
            GetLocalRoughness(input.uv) : 0.0;
        outColor = float4(GetColor(input.uv).rgb, localRoughness);
    }
    
    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
        {
            outNormal = float4(0.0, 0.0, 1.0, 1.0);
            return;
        }
        
        float3 normal = CalculateNormal(input.uv, input.pScale);
        if (Vegetation_Protection > 0.0)
        {
            float2 p = ReShade::PixelSize * 1.5;
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

    void PS_SmoothNormals_V(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 centerNormal = GetLod(sNormal1, input.uv);
        if (centerNormal.a >= 0.999)
        {
            outNormal = centerNormal;
            return;
        }
    
        float4 smoothed;
        if (SmartSurfaceMode == 0)
        {
            smoothed = centerNormal;
        }
        else
        {
            smoothed = ComputeSmoothedNormal(input.uv, float2(0, 1), sNormal1);
        }
        
        float depth = smoothed.a;
        float3 finalNormal = ApplySurfaceDetails(input.uv, smoothed.rgb, depth);
        outNormal = float4(finalNormal, depth);
    }

   void PS_TraceReflections(VS_OUTPUT input, out float4 outReflection : SV_Target)
    {
        // Virtual Resolution
        float2 scaled_uv = input.uv / RenderResolution;
        float depth = GetDepth(scaled_uv);

        // TAA Upscaling Jitter
        if (EnableTAAUpscaling)
        {
            const float2 jitterOffsets[8] =
            {
                float2(0.125, -0.375), float2(-0.125, 0.375),
                float2(-0.375, -0.125), float2(0.375, 0.125),
                float2(0.375, -0.375), float2(-0.375, 0.375),
                float2(0.125, 0.125), float2(-0.125, -0.125)
            };
            uint jitter_idx = uint(fmod((float)FRAME_COUNT, 8.0));
            float2 jitter = jitterOffsets[jitter_idx] * (ReShade::PixelSize / RenderResolution);
            scaled_uv += jitter;
        }

        if (any(scaled_uv < 0.001) || any(scaled_uv > 0.999))
        {
            outReflection = 0.0;
            return;
        }

        if (depth >= 1.0)
        {
            outReflection = 0.0;
            return;
        }
        
        float4 gbuffer = SampleGBuffer(scaled_uv);
        float2 pScale = input.pScale;
        float3 normal = normalize(gbuffer.rgb);
        float3 viewPos = UVToViewPos(scaled_uv, depth, pScale);
        // Sub-Pixel TAAU Jitter: Perturb the view position infinitesimally
        float3 jittered_viewPos = viewPos;
        float3 viewDir = -normalize(viewPos);
        float estimatedRoughness = GetLod(sTexColorCopy, scaled_uv).a;

        // Surface Orientation
        bool showFloor = (ReflectionMode == 0 || ReflectionMode == 3 || ReflectionMode == 4);
        bool showWall = (ReflectionMode == 1 || ReflectionMode == 4);
        bool showCeil = (ReflectionMode == 2 || ReflectionMode == 3 || ReflectionMode == 4);

        float orientationIntensity = 0.0;
        if (normal.y > OrientationThreshold && showFloor)
            orientationIntensity = 1.0;
        else if (normal.y < -OrientationThreshold && showCeil)
            orientationIntensity = 1.0;
        else if (abs(normal.y) <= OrientationThreshold && showWall)
            orientationIntensity = 1.0;
        if (orientationIntensity <= 0.0)
        {
            outReflection = 0.0;
            return;
        }

        float2 bn_uv = input.vpos.xy / (RenderResolution * 1024.0);
        float frame = fmod((float)FRAME_COUNT, 64.0);
        float4 bn = tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0));
        // Ray Jitter 
        float ray_jitter = 0.0;
        if (EnableRayJitter)
        {
            ray_jitter = frac(bn.b + 0.51248584407 * frame);
        }

        Ray r;
        r.origin = viewPos;
#if ENABLE_VNDF
        float netRoughness = saturate(SurfaceGlossiness + (estimatedRoughness * RoughnessDetection)) * 0.5;
        float2 screen_size = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        
        float frame_anim = fmod((float)FRAME_COUNT, 64.0);
        float2 golden_offset = float2(0.61803398875, 0.73205080757) * frame_anim;
        float3 stbn_noise = tex2Dlod(sTexBlueNoise, float4(bn_uv + golden_offset, 0, 0)).rgb;
        
        float2 Xi = frac(stbn_noise.gb + frac(sin(dot(scaled_uv ,float2(12.9898,78.233))) * 43758.5453));
        float3 H = ImportanceSampleGGX_VNDF(Xi, normal, viewDir, netRoughness);
        float3 reflectDir = reflect(-viewDir, H);
        if (dot(reflectDir, normal) < 0.0) 
            reflectDir = reflectDir - 2.0 * dot(reflectDir, normal) * normal;
        r.direction = normalize(reflectDir);
#else
        r.direction = normalize(reflect(-viewDir, normal));
#endif

        float bias = 0.0005 + (depth * 0.02);
        r.origin += r.direction * bias;

        float VdotN = dot(viewDir, normal);
        if (VdotN > 0.9 || r.direction.z < 0.0)
        {
            outReflection = 0.0;
            return;
        }

        // Quality 
        int ray_steps = 12;
        float max_dist = 4.0;
        
        if (RayTraceQuality == 1)
        {
            ray_steps = 32;
            max_dist = 12.0;
        }
        else if (RayTraceQuality == 2)
        {
            ray_steps = 128;
            max_dist = 100.0;
        }

        // Ray Tracing
        float geoThickness = GetThickness(scaled_uv, normal, normalize(viewPos), depth);
        HitResult hit = TraceRay2D(r, ray_steps, max_dist, pScale, ray_jitter, geoThickness);

        float3 reflectionColor = 0.0;
        float reflectionAlpha = 0.0;
        if (hit.found)
        {
            reflectionColor = GetGlossySample(hit.uv, scaled_uv, estimatedRoughness, normal, pScale);
            // Calculate Fading
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / 10.0);
            float fadeRange = max(FadeDistance, 0.001);
            float depthFade = saturate((FadeDistance - depth) / fadeRange);
            depthFade *= depthFade;
            float2 edgeDist = min(hit.uv, 1.0 - hit.uv);
            float screenFade = smoothstep(0.0, 0.001, min(edgeDist.x, edgeDist.y));
            reflectionAlpha = distFactor * depthFade * screenFade;

            // Masking
            float3 nR = SampleGBuffer(scaled_uv + float2(ReShade::PixelSize.x, 0)).rgb;
            float3 nD = SampleGBuffer(scaled_uv + float2(0, ReShade::PixelSize.y)).rgb;
            float edgeDelta = length(normal - nR) + length(normal - nD);
            float geoMask = 1.0 - smoothstep(0.05, EDGE_MASK_THRESHOLD, edgeDelta);
            
            reflectionAlpha *= geoMask;
        }
#if ENABLE_RAY_FALLBACK
        else
        {
            // Fallback 
            float adaptiveDist = min(depth * 1.2 + 0.012, 10.0);
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos, pScale).xy);

            reflectionColor = GetGlossySample(uvFb, scaled_uv, estimatedRoughness, normal, pScale);
            float baseAlpha = smoothstep(0.0, 0.2, 1.0 - scaled_uv.y);
            reflectionAlpha = baseAlpha ;
        }
#endif
        
        float fresnelFade = pow(saturate(dot(-viewDir, r.direction)), 2.0);
        outReflection = float4(Input2Linear(reflectionColor), reflectionAlpha * fresnelFade * orientationIntensity);
    }
    
    void PS_Accumulate0(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
            discard;
        outBlended = ComputeTAA(input, sHistory1);
    }

    void PS_Accumulate1(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
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
        //Scene Highlight Preservation
        float sceneLuma = GetLuminance(scene);
        float highlightProtectionMask = smoothstep(paper_white_norm, paper_white_norm * 4.0, sceneLuma);
        reflectionMask *= saturate(1.0 - (highlightProtectionMask * Preserve_Scene_Highlights));
        // PBR & Blending
        float3 viewDir = -normalize(UVToViewPos(input.uv, depth, input.pScale));
        float VdotN = saturate(dot(viewDir, normal));
        float3 f0 = lerp(DIELECTRIC_REFLECTANCE.xxx, saturate(scene), Metallic);
        float3 F = saturate(F_Schlick(VdotN, f0));

        float3 finalColor;
        if (BlendMode == 0) // Default PBR 
    {
    float validReflection = reflectionMask * saturate(Intensity);
    float3 kD = saturate(1.0 - (F * validReflection));
    kD *= lerp(1.0, 1.0 - Metallic, validReflection);
    
    float3 specularLight = reflectionColor * F * Intensity * reflectionMask;
    finalColor = (scene * kD) + specularLight;
    }
    else // Legacy Blending modes
    {
        float blendAmount = dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask;
        finalColor = ComHeaders::Blending::Blend(BlendMode, scene, reflectionColor, blendAmount * Intensity);
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
        pass SmoothNormals_V
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_SmoothNormals_V;
            RenderTarget = Normal;
        }
        pass CopyColorGenMips
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_CopyColor;
            RenderTarget = TexColorCopy;
        }
        pass TraceReflections
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_TraceReflections;
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