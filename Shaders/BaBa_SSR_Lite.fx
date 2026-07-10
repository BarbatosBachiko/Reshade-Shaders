/*----------------------------------------------|
| ::          Barbatos SSR  LITE             :: |
'-----------------------------------------------|
| Version: 2.0.2                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include ".\Includes\bb_reshade.fxh"
#include ".\Includes\bb_ui.fxh"
#define USE_HALF 1
#include ".\Includes\bb_common.fxh"
#include ".\Includes\bb_colorspace.fxh"
#include ".\Includes\bb_depth.fxh"
#include ".\Includes\bb_normal.fxh"
#include ".\Includes\bb_noise.fxh"
#include ".\Includes\bb_raytracing.fxh"
#include ".\Includes\bb_mv.fxh"
#include ".\Includes\bb_taa.fxh"
#include ".\Includes\bb_vertex.fxh"

//----------|
// :: UI :: |
//----------|

// Reflections
uniform float Intensity <
    ui_category = "Reflections";
    ui_label = "Intensity";
    ui_tooltip = "Overall strength of screen-space reflections.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float THICKNESS_THRESHOLD <
    ui_category = "Reflections";
    ui_label = "Thickness";
    ui_tooltip = "Estimates depth thickness for ray hits. Higher values reduce missed reflections.";
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.6; ui_step = 0.001;
> = 0.003;

uniform float FadeDistance <
    ui_category = "Reflections";
    ui_label = "Fade Distance";
    ui_tooltip = "How far into the background reflections fade out.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 4.999;

uniform int ReflectionMode <
    ui_category = "Reflections";
    ui_label = "Surface Target Mode";
    ui_tooltip = "Restrict reflections to floors, walls, ceilings, or all surfaces.";
    ui_type = "combo";
    ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
> = 0;

uniform float OrientationThreshold <
    ui_category = "Reflections";
    ui_label = "Surface Angle Threshold";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform int RayCount <
    ui_category = "Reflections";
    ui_label = "Rays per Pixel";
    ui_type = "drag";
    ui_min = 1; ui_max = 4; ui_step = 1;
> = 1;

uniform int RaySteps <
    ui_category = "Reflections";
    ui_label = "Ray Steps";
    ui_tooltip = "More steps = finer march along the SSR range (not farther).";
    ui_type = "drag";
    ui_min = 8; ui_max = 64; ui_step = 1;
> = 16;

uniform bool EnableRayJitter <
    ui_category = "Reflections";
    ui_label = "Enable Ray Jitter";
    ui_tooltip = "Trades banding for noise (smoothed by the denoiser).";
> = true;

uniform float RenderScale <
    ui_category = "Reflections";
    ui_label = "Resolution Scale";
    ui_tooltip = "Internal reflection resolution. Lower = faster.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.001;
> = 0.5;

uniform float VERTICAL_FOV <
    ui_category = "Reflections";
    ui_label = "Game FOV";
    ui_tooltip = "Must match the game vertical FOV for correct alignment.";
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
> = 60.0;

// Material
uniform float SurfaceGlossiness <
    ui_category = "Material";
    ui_label = "Glossiness";
    ui_tooltip = "0 = mirror-sharp reflections. Soft cone sampling is intentionally limited in Lite.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float Metallic <
    ui_category = "Material";
    ui_label = "Metallic";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;

uniform float DIELECTRIC_REFLECTANCE <
    ui_category = "Material";
    ui_label = "Dielectric Reflectance (F0)";
    ui_tooltip = "Base front-facing reflectivity for non-metals. 0.04 is physically typical.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.04;

// Denoiser
uniform bool EnableSmoothing <
    ui_category = "Denoiser";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Uses previous frames to denoise reflections.";
> = true;

uniform bool EnableAntiSmear <
    ui_category = "Denoiser";
    ui_label = "Reduce Motion Smearing";
> = false;

uniform int MaxFrames <
    ui_category = "Denoiser";
    ui_label = "Max Accumulation Frames";
    ui_type = "slider";
    ui_min = 4; ui_max = 128; ui_step = 1;
> = 24;

// Color Grading
BLENDING_COMBO(BlendMode, "Blend Mode", "Select how reflections are composited. Mode 0 is physically based (PBR).", "Color Grading", false, 0, 0)

uniform float Preserve_Scene_Highlights <
    ui_category = "Color Grading";
    ui_label = "Preserve Scene Highlights";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool Use_Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Use Kelvin Temperature Tint";
> = false;

uniform float Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Temperature (Kelvin)";
    ui_type = "drag";
    ui_min = 1500.0; ui_max = 15000.0; ui_step = 10.0;
> = 6500.0;

uniform float SSR_Vibrance <
    ui_category = "Color Grading";
    ui_label = "Vibrance";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
> = 1.0;

uniform float SSR_Contrast <
    ui_category = "Color Grading";
    ui_label = "Contrast";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float3 SSR_Tint <
    ui_category = "Color Grading";
    ui_label = "Global Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Shadow_Tint <
    ui_category = "Color Grading";
    ui_label = "Shadow Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Highlight_Tint <
    ui_category = "Color Grading";
    ui_label = "Highlight Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float SSR_Split_Balance <
    ui_category = "Color Grading";
    ui_label = "Split Tint Balance";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

// Advanced
uniform float Vegetation_Protection <
    ui_category = "Advanced";
    ui_category_closed = true;
    ui_label = "Vegetation Masking";
    ui_tooltip = "Disables reflections on erratic depth (leaves/grass).";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float EDGE_MASK_THRESHOLD <
    ui_category = "Advanced";
    ui_label = "Geometry Edge Masking";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform int ViewMode <
    ui_category = "Advanced";
    ui_label = "Debug View";
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0Motion Vectors\0Raw LowRes\0";
> = 0;

#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 0
#endif

#define SSR_MAX_DIST 4.0
static const float SSR_DIST_FADE = 10.0;

//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_SSR_Lite_200
{
    texture Normal
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = Normal;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture Accum
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sAccum
    {
        Texture = Accum;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
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
        MagFilter = LINEAR;
        MinFilter = LINEAR;
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
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture DNA
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDNA
    {
        Texture = DNA;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture DNB
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDNB
    {
        Texture = DNB;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture TexColorCopy
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
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

#include ".\Includes\bb_bluenoise.fxh"

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

    //---------------------|
    // :: Vertex Shaders ::|
    //---------------------|

    void VS_Barbatos_SSR_Lite(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, VERTICAL_FOV);
    }

    //----------------|
    // :: Functions ::|
    //----------------|

    float GetDepth(float2 xy)
    {
        return bb::GetLinearizedDepth(xy);
    }

    float3 CalculateNormal(float2 uv, float2 pScale)
    {
        // Match BaBa_SSR: NM_CalculateNormal + flip X before tracing
        float3 n = NM_CalculateNormal(uv, pScale);
        n.x = -n.x;
        return n;
    }

    float GetOrientationIntensity(float3 normal)
    {
        bool showFloor = false;
        bool showWall = false;
        bool showCeil = false;
        switch (ReflectionMode)
        {
            case 0: showFloor = true; break;
            case 1: showWall = true; break;
            case 2: showCeil = true; break;
            case 3: showFloor = true; showCeil = true; break;
            default: showFloor = true; showWall = true; showCeil = true; break;
        }

        float intensity = 0.0;
        if (normal.y > OrientationThreshold && showFloor)
            intensity = 1.0;
        else if (normal.y < -OrientationThreshold && showCeil)
            intensity = 1.0;
        else if (abs(normal.y) <= OrientationThreshold && showWall)
            intensity = 1.0;
        return intensity;
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

    void PS_CopyColor(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        outColor = GetColor(input.uv);
    }

    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        outNormal = 0.0;
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float d = GetDepth(viewUV);
        if (d >= 0.999)
        {
            outNormal = float4(0, 0, 1, d);
            return;
        }

        float3 normal = CalculateNormal(viewUV, input.pScale);
        if (Vegetation_Protection > 0.0)
        {
            float2 p = bb::PixelSize * 1.5;
            float dX = abs(GetDepth(viewUV + float2(p.x, 0)) - d);
            float dY = abs(GetDepth(viewUV + float2(0, p.y)) - d);
            float dX_inv = abs(GetDepth(viewUV - float2(p.x, 0)) - d);
            float dY_inv = abs(GetDepth(viewUV - float2(0, p.y)) - d);
            float depthNoise = dX + dY + dX_inv + dY_inv;
            float threshold = (1.0 - Vegetation_Protection) * 0.05;
            if (depthNoise > threshold)
                normal = float3(0.0, 0.0, -1.0);
        }

        outNormal = float4(normal, d);
    }

    void PS_Trace(VS_OUTPUT input, out float4 outSSR : SV_Target)
    {
        outSSR = 0.0;
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float4 gbuffer = GetLod(sNormal, input.uv);
        float depth = gbuffer.a;
        if (depth >= 0.999)
            return;

        float3 normal = normalize(gbuffer.rgb);
        float orientationIntensity = GetOrientationIntensity(normal);
        if (orientationIntensity <= 0.0)
            return;

        // Same convention as BaBa_SSR: viewDir points toward the camera
        float3 viewPos = UVToViewPos(viewUV, depth, input.pScale);
        float3 viewDir = -normalize(viewPos);

        float2 bn_uv = input.vpos.xy / (RenderScale * 1024.0);
        float frame = fmod((float)FRAME_COUNT, 64.0);
        float4 bn = tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0));
        float ray_jitter = EnableRayJitter ? frac(bn.b + 0.51248584407 * frame) : 0.0;

        // Avoid RT_GetThickness fwidth kill at low RenderScale — use stable geometric scale
        float NdotV = abs(dot(normal, viewDir));
        float geoThickness = THICKNESS_THRESHOLD / max(NdotV, 0.2);

        float3 totalRadiance = 0.0;
        float totalMask = 0.0;
        float invRays = 1.0 / float(max(1, RayCount));
        float roughness = saturate(SurfaceGlossiness);

        [loop]
        for (int s = 0; s < RayCount; s++)
        {
            Ray r;
            r.origin = viewPos;

            // Default path matches BaBa_SSR with ENABLE_VNDF=0 (mirror).
            // Mild VNDF only when glossiness > 0.
            if (roughness > 0.001)
            {
                float2 Xi = frac(bn.gb + float2(0.1732, 0.419) * float(s) +
                    frac(sin(dot(viewUV + float(s) * 0.11, float2(12.9898, 78.233))) * 43758.5453));
                float3 H = RT_ImportanceSampleGGX_VNDF(Xi, normal, viewDir, roughness);
                float3 reflectDir = reflect(-viewDir, H);
                if (dot(reflectDir, normal) < 0.0)
                    reflectDir = reflectDir - 2.0 * dot(reflectDir, normal) * normal;
                r.direction = normalize(reflectDir);
            }
            else
            {
                r.direction = normalize(reflect(-viewDir, normal));
            }

            r.origin += r.direction * (0.0005 + (depth * 0.02));

            if (dot(viewDir, normal) > 0.9 || r.direction.z < 0.0)
                continue;

            float stepJitter = EnableRayJitter ? frac(ray_jitter + float(s) * 0.61803398875) : 0.0;
            HitResult hit = RT_TraceRay2D(r, RaySteps, SSR_MAX_DIST, input.pScale, stepJitter, geoThickness);
            if (!hit.found)
                continue;

            float3 rawColor = tex2Dlod(sTexColorCopy, float4(hit.uv, 0, 0)).rgb;
            float3 linearColor = Input2Linear(rawColor);

            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / SSR_DIST_FADE);
            float depthFade = saturate((FadeDistance - depth) / max(FadeDistance, 0.001));
            depthFade *= depthFade;

            float2 edgeDist = min(hit.uv, 1.0 - hit.uv);
            float screenFade = smoothstep(0.0, 0.10, min(edgeDist.x, edgeDist.y));

            float3 nR = GetLod(sNormal, input.uv + float2(bb::PixelSize.x, 0) * RenderScale).rgb;
            float3 nD = GetLod(sNormal, input.uv + float2(0, bb::PixelSize.y) * RenderScale).rgb;
            float geoMask = 1.0 - smoothstep(0.05, max(EDGE_MASK_THRESHOLD, 0.001), length(normal - nR) + length(normal - nD));

            // Must match BaBa_SSR: dot(-viewDir, reflectDir)
            float fresnelFadeNV = max(0.0, dot(-viewDir, r.direction));
            float fresnelFade = fresnelFadeNV * fresnelFadeNV;

            float mask = distFactor * depthFade * screenFade * geoMask * fresnelFade * orientationIntensity;

            totalRadiance += linearColor;
            totalMask += mask;
        }

        outSSR = float4(totalRadiance * invRays, saturate(totalMask * invRays));
    }

    float4 AtrousFilter(VS_OUTPUT input, sampler sInputTex, float stepWidth)
    {
        if (any(input.uv > RenderScale))
            discard;
        float4 c_data = GetLod(sInputTex, input.uv);
        float3 c_val = c_data.rgb;
        float c_a = c_data.a;

        float4 c_gbuffer = GetLod(sNormal, input.uv);
        float3 c_norm = c_gbuffer.rgb;
        float c_depth = c_gbuffer.a;

        static const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };
        hfloat4 sum = hfloat4(c_val, c_a);
        hfloat cum_w = 1.0;

        float2 px = bb::PixelSize * stepWidth;
        float depth_weight_factor = ComputeDepthWeight(c_depth, 0.1);

        [loop]
        for (int x = -2; x <= 2; x++)
        {
            [loop]
            for (int y = -2; y <= 2; y++)
            {
                if (x == 0 && y == 0)
                    continue;
                float2 uv_offset = input.uv + float2(x, y) * px;
                float4 s_data = GetLod(sInputTex, uv_offset);
                float4 s_gbuffer = GetLod(sNormal, uv_offset);
                float3 s_norm = s_gbuffer.rgb;
                float s_depth = s_gbuffer.a;

                hfloat w_z = exp(-abs(c_depth - s_depth) * depth_weight_factor);
                hfloat dotN = max(0.0, dot(c_norm, s_norm));
                hfloat dotN2 = dotN * dotN;
                hfloat w_n = dotN2 * dotN2;

                hfloat k_w = kernel[abs(x)] * kernel[abs(y)];
                hfloat weight = w_z * w_n * k_w;

                sum += s_data * weight;
                cum_w += weight;
            }
        }
        return sum / max(cum_w, 0.0001);
    }

    //------------|
    // :: TAA  :: |
    //------------|

    float4 ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;
        float depth = GetDepth(viewUV);
        if (depth >= 0.999)
            return float4(0.0, 0.0, 0.0, 0.0);

        float4 current_ssr = GetLod(sAccum, input.uv);
        if (!EnableSmoothing)
            return current_ssr;

        float2 velocity = MV_GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        if (any(saturate(reprojected_view_uv) != reprojected_view_uv) || FRAME_COUNT <= 1)
            return current_ssr;

        float2 reprojected_buffer_uv = reprojected_view_uv * RenderScale;

        float4 history_ssr = TAA_SampleHistoryCatmullRom(sHistoryParams, reprojected_buffer_uv, float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float3 current_compressed = TAA_Compress(current_ssr.rgb);
        float3 current_ycocg = RGBToYCoCg(current_compressed);

        float3 history_compressed = TAA_Compress(history_ssr.rgb);
        float3 history_ycocg = RGBToYCoCg(history_compressed);

        float raw_confidence = saturate(MV_GetConfidence(viewUV));
        float4 color_min, color_max;
        TAA_ComputeNeighborhoodVariance(sAccum, input.uv, current_ssr, bb::PixelSize, color_min, color_max);

        float relax_amount = 0.15 * raw_confidence;
        color_min -= relax_amount;
        color_max += relax_amount;
        float3 clipped_history_ycocg = TAA_ClipToAABB(color_min.rgb, color_max.rgb, history_ycocg);
        float clipped_history_a = clamp(history_ssr.a, color_min.a, color_max.a);

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

        float prevRenderScale = tex2Dlod(sRS_Prev, float4(0, 0, 0, 0)).x;
        if (abs(RenderScale - prevRenderScale) > 0.001)
            final_feedback = 0.0;

        float3 result_ycocg = lerp(current_ycocg, clipped_history_ycocg, final_feedback);
        float3 result_compressed = YCoCgToRGB(result_ycocg);
        float result_alpha = lerp(current_ssr.a, clipped_history_a, final_feedback);

        return float4(TAA_Resolve(result_compressed), result_alpha);
    }

    void PS_Accumulate0(VS_OUTPUT input, out float4 outAccum : SV_Target)
    {
        outAccum = 0.0;
        if (uint(FRAME_COUNT) % 2 != 0)
            discard;
        outAccum = ComputeTAA(input, sHistory1);
    }

    void PS_Accumulate1(VS_OUTPUT input, out float4 outAccum : SV_Target)
    {
        outAccum = 0.0;
        if (uint(FRAME_COUNT) % 2 == 0)
            discard;
        outAccum = ComputeTAA(input, sHistory0);
    }

    void PS_Atrous1(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        if (uint(FRAME_COUNT) % 2 == 0)
            outColor = AtrousFilter(input, sHistory0, 1.0);
        else
            outColor = AtrousFilter(input, sHistory1, 1.0);
    }

    void PS_AtrousFinal(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        outColor = AtrousFilter(input, sDNA, 3.0);
    }

    float4 JointBilateralUpsample(float2 uv, float highDepth, float2 pScale)
    {
        float2 lowResUV = uv * RenderScale;
        float3 highNormal = CalculateNormal(uv, pScale);

        float4 result = GetLod(sDNB, lowResUV);
        hfloat4 sumSSR = 0.0;
        hfloat sumWeight = 0.0;

        float2 texelSize = bb::PixelSize;
        float2 baseUV = (floor(lowResUV / texelSize) + 0.5) * texelSize;
        float depth_weight_factor = ComputeDepthWeight(highDepth, 0.1);

        [loop]
        for (int x = -1; x <= 1; x++)
        {
            [loop]
            for (int y = -1; y <= 1; y++)
            {
                float2 sampleUV = baseUV + float2(x, y) * texelSize;
                float4 ssr = GetLod(sDNB, sampleUV);
                float4 gbuffer = GetLod(sNormal, sampleUV);

                float3 lowNormal = gbuffer.rgb;
                float lowDepth = gbuffer.a;

                hfloat wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
                hfloat dotN = max(0.0, dot(highNormal, lowNormal));
                hfloat dotN2 = dotN * dotN;
                hfloat dotN4 = dotN2 * dotN2;
                hfloat dotN8 = dotN4 * dotN4;
                hfloat wNormal = dotN8 * dotN8;
                hfloat wSpatial = exp(-0.5 * float(x * x + y * y));

                hfloat weight = wDepth * wNormal * wSpatial;
                sumSSR += ssr * weight;
                sumWeight += weight;
            }
        }

        if (sumWeight >= 1e-6)
            result = sumSSR / sumWeight;

        return result;
    }

    void PS_Output(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        float depth = GetDepth(input.uv);
        float3 rawScene = GetColor(input.uv).rgb;
        float3 scene = Input2Linear(rawScene);

        if (ViewMode != 0)
        {
            float3 debugColor = 0.0;
            if (ViewMode == 1)
            {
                float4 ssr = JointBilateralUpsample(input.uv, depth, input.pScale);
                debugColor = Linear2Output(ssr.rgb * ssr.a);
            }
            else if (ViewMode == 2)
            {
                float3 debugNormals = GetLod(sNormal, input.uv * RenderScale).rgb;
                if (depth < 0.999)
                {
                    debugNormals.x = -debugNormals.x;
                    debugNormals.z = -debugNormals.z;
                }
                debugColor = debugNormals * 0.5 + 0.5;
            }
            else if (ViewMode == 3)
            {
                debugColor = depth.xxx;
            }
            else if (ViewMode == 4)
            {
                float2 mv = SampleMotionVectors(input.uv);
                debugColor = saturate(float3(mv.x, mv.y, 0.0) * 50.0 + 0.5);
            }
            else if (ViewMode == 5)
            {
                debugColor = (uint(FRAME_COUNT) % 2 == 0 ?
                    GetLod(sHistory0, input.uv * RenderScale).rgb :
                    GetLod(sHistory1, input.uv * RenderScale).rgb);
            }
            outColor = float4(debugColor, 1.0);
            return;
        }

        if (depth >= 1.0)
        {
            outColor = float4(rawScene, 1.0);
            return;
        }

        float4 reflectionSample = JointBilateralUpsample(input.uv, depth, input.pScale);
        float3 reflectionColor = reflectionSample.rgb;
        float reflectionMask = reflectionSample.a;

        float3 normal = GetLod(sNormal, input.uv * RenderScale).rgb;

        // Color Grading (from BaBa_SSR)
        float3 tint = Use_Color_Temperature ?
            KelvinToRGB(Color_Temperature) : SSR_Tint;
        reflectionColor *= tint;
        float paper_white_norm = 80.0 / HDR_Peak_Nits;
        float mid_gray = paper_white_norm * 0.18;

        reflectionColor = (reflectionColor - mid_gray) * SSR_Contrast + mid_gray;
        reflectionColor = max(0.0, reflectionColor);

        float reflLum = GetLuminance(reflectionColor);
        float3 chroma = reflectionColor - reflLum;
        reflectionColor = reflLum + chroma * SSR_Vibrance;

        float luma_normalized = saturate(reflLum / (paper_white_norm * 3.0));
        float shadowCurve = 1.0 - smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);
        float highlightCurve = smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);

        float3 splitTint = shadowCurve * SSR_Shadow_Tint + highlightCurve * SSR_Highlight_Tint;
        reflectionColor = reflectionColor * splitTint;
        float splitTintAlpha = max(splitTint.r, max(splitTint.g, splitTint.b));
        reflectionMask *= saturate(splitTintAlpha);

        float sceneLuma = max(GetLuminance(scene), 1e-6);
        float3 sceneTint = saturate(scene / sceneLuma);
        float highlightProtectionMask = smoothstep(paper_white_norm, paper_white_norm * 4.0, sceneLuma);
        reflectionMask *= saturate(1.0 - (highlightProtectionMask * Preserve_Scene_Highlights));

        float3 viewDir = -normalize(UVToViewPos(input.uv, depth, input.pScale));
        float VdotN = saturate(dot(viewDir, normal));

        float3 metalF0 = sceneTint * max(sceneLuma, DIELECTRIC_REFLECTANCE);
        float3 f0 = lerp(DIELECTRIC_REFLECTANCE.xxx, metalF0, Metallic);
        float3 F = saturate(RT_F_Schlick(VdotN, f0));

        float3 finalColor;
        if (BlendMode == 0)
        {
            float validReflection = reflectionMask * saturate(Intensity);
            float3 kD = saturate(1.0 - (F * validReflection));
            float3 specularLight = reflectionColor * F * Intensity * reflectionMask;
            finalColor = (scene * kD) + specularLight;
        }
        else
        {
            float blendAmount = saturate(dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask * Intensity);
            finalColor = bb::Blending::Blend(BlendMode, scene, reflectionColor, blendAmount);
        }

        outColor = float4(Linear2Output(finalColor), 1.0);
    }

    void PS_SaveScale(VS_OUTPUT input, out float4 outScale : SV_Target)
    {
        outScale = float4(RenderScale, 0.0, 0.0, 1.0);
    }

    technique BaBa_SSR_Lite
    <
        ui_label = "BaBa: SSR Lite";
        ui_tooltip = "Lightweight screen-space reflections (GI-style pipeline)";
    >
    {
        pass CopyColorGenMips
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_CopyColor;
            RenderTarget = TexColorCopy;
        }
        pass Normals
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_GenNormals;
            RenderTarget = Normal;
        }
        pass Trace
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_Trace;
            RenderTarget = Accum;
        }
        pass Accumulate0
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass DenoiseStep1
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_Atrous1;
            RenderTarget = DNA;
        }
        pass DenoiseStep2
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_AtrousFinal;
            RenderTarget = DNB;
        }
        pass Output
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_Output;
        }
        pass SaveScale
        {
            VertexShader = VS_Barbatos_SSR_Lite;
            PixelShader = PS_SaveScale;
            RenderTarget = RS_Prev;
        }
    }
}
