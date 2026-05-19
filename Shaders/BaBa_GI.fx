/*----------------------------------------------|
| | ::             Barbatos GI               :: |
| |---------------------------------------------|
| Version: 1.6.0                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include ".\bb_include\bb_reshade.fxh"
#define USE_HALF 1
#include ".\bb_include\bb_common.fxh"
#include ".\bb_include\bb_depth.fxh"
#include ".\bb_include\bb_normal.fxh"
#include ".\bb_include\bb_noise.fxh"
#include ".\bb_include\bb_raytracing.fxh"
#include ".\bb_include\bb_colorspace.fxh"
#include ".\bb_include\bb_mv.fxh"
#include ".\bb_include\bb_taa.fxh"
#include ".\bb_include\bb_vertex.fxh"

//----------|
// :: UI :: |
//----------|

// Global Lighting
uniform float Intensity <
    ui_category = "Global Lighting";
    ui_label = "GI Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 1.0;

uniform float GI_RenderDistance <
    ui_category = "Global Lighting";
    ui_label = "Render Distance";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 1.0;

// Raytracing (Advanced)
uniform int RayCount <
    ui_category = "Raytracing (Advanced)";
    ui_category_closed = true;
    ui_label = "Rays per Pixel";
    ui_type = "drag";
    ui_min = 1; ui_max = 16; ui_step = 1.0;
> = 2;

uniform int RaySteps <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Ray Steps";
    ui_type = "drag";
    ui_min = 2; ui_max = 16;
> = 6;

uniform float ZThickness <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Z-Thickness Tolerance";
    ui_tooltip = "Defines the virtual thickness of surfaces to prevent light leaking behind geometry. Lower values are more accurate but may cause light leaking on thin geometry.";
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.5; ui_step = 0.001;
> = 0.004;

uniform float MaxRayDistance <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Max Ray Distance";
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.5; ui_step = 0.001;
> = 0.020;

uniform float Near_Intensity <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Near Field Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.8;

uniform float GI_Bounce_Saturation <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Bounce Color Saturation";
    ui_tooltip = "Increases or decreases the color intensity of the bounced light.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 5.0; ui_step = 0.01;
> = 1.5;

uniform float GI_Bounce_Energy <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Bounce Energy Multiplier";
    ui_tooltip = "Multiplier for the raw bounce light intensity.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 5.0; ui_step = 0.01;
> = 1.5;

uniform float MultiBounce_Weight <
    ui_category = "Raytracing (Advanced)";
    ui_label = "Infinite Bounces Weight";
    ui_tooltip = "Simulates multiple light bounces by accumulating previous frame GI. \nHigh values may cause glowing feedback loops.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 1.0;
> = 0.0;

// Ambient Occlusion
uniform float AO_Intensity <
    ui_category = "Ambient Occlusion";
    ui_category_closed = true;
    ui_label = "AO Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float AO_Radius <
    ui_category = "Ambient Occlusion";
    ui_label = "AO Radius";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 5.0;
> = 0.1;

uniform int AO_BlendMode <
    ui_category = "Ambient Occlusion";
    ui_label = "AO Blend Mode";
    ui_type = "combo";
    ui_items = "Multiplicative\0Luminance Masked\0";
> = 0;

// Color Grading
uniform bool Use_Color_Temperature <
    ui_category = "Color Grading";
    ui_category_closed = true;
    ui_label = "Use Color Temperature";
> = false;

uniform float Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Temperature (Kelvin)";
    ui_type = "drag";
    ui_min = 1500.0; ui_max = 15000.0; ui_step = 10.0;
> = 6500.0;

uniform float GI_Color_Bleed <
    ui_category = "Color Grading";
    ui_label = "Material Color Bleed";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.8;

uniform float3 GI_Color <
    ui_category = "Color Grading";
    ui_label = "Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float GI_Vibrance <
    ui_category = "Color Grading";
    ui_label = "Saturation";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
> = 2.0;

uniform float GI_Contrast <
    ui_category = "Color Grading";
    ui_label = "Contrast";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
> = 1.0;

uniform float3 GI_Shadow_Tint <
    ui_category = "Color Grading";
    ui_label = "Shadow Color";
    ui_tooltip = "Color of GI in dark areas. Set to Black to disable GI in shadows.";
    ui_type = "color";
> = float3(0.5, 0.5, 0.5);

uniform float3 GI_Highlight_Tint <
    ui_category = "Color Grading";
    ui_label = "Highlight Color";
    ui_tooltip = "Color of GI in bright areas. Set to Black to disable GI in highlights.";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float GI_Split_Balance <
    ui_category = "Color Grading";
    ui_label = "Split Balance";
    ui_tooltip = "Determines the separation point between Shadows and Highlights.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
> = 0.5;

// Manual Light
uniform bool SSS_Enabled <
    ui_category = "Manual Light";
    ui_category_closed = true;
    ui_label = "Enable Screen Space Shadows";
    ui_tooltip = "Adds directional shadows";
> = false;

uniform float SSS_Intensity <
    ui_category = "Manual Light";
    ui_label = "Shadow Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
> = 1.0;

uniform bool Manual_Sun_Enabled <
    ui_category = "Manual Light";
    ui_label = "Enable Manual Sun";
> = false;

uniform float Sun_Azimuth <
    ui_category = "Manual Light";
    ui_label = "Sun Rotation (Azimuth)";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 360.0; ui_step = 1.0;
> = 175.0;

uniform float Sun_Elevation <
    ui_category = "Manual Light";
    ui_label = "Sun Elevation (Altitude)";
    ui_type = "drag";
    ui_min = -15.0; ui_max = 90.0; ui_step = 1.0;
> = 22.0;

uniform float Shadow_Softness <
    ui_category = "Manual Light";
    ui_label = "Sun Shadow Softness";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
> = 0.1;

uniform float Sun_Shadow_Fill <
    ui_category = "Manual Light";
    ui_label = "Ambient Fill";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool Show_Sun_Widget <
    ui_category = "Manual Light";
    ui_label = "Show Sun Widget";
> = true;

uniform float2 Sun_Widget_Pos <
    ui_category = "Manual Light";
    ui_label = "Widget Position";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = float2(0.88, 0.15);

uniform float Sun_Widget_Scale <
    ui_category = "Manual Light";
    ui_label = "Widget Scale";
    ui_type = "drag";
    ui_min = 0.05; ui_max = 0.5; ui_step = 0.001;
> = 0.100;

// System / Advanced
uniform float RenderScale <
    ui_category = "System / Debug";
    ui_category_closed = true;
    ui_label = "Resolution Scale";
    ui_tooltip = "Scales the rendering resolution of GI.";
    ui_type = "drag";
    ui_min = 0.1;
    ui_max = 1.0; ui_step = 0.001;
> = 0.333;

uniform float Roughness <
    ui_category_closed = true;
    ui_category = "System / Debug";
    ui_label = "Roughness";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform bool EnableDepthMultiplier <
    ui_category = "System / Debug";
    ui_label = "Enable Depth Multiplier";
> = false;

uniform float DepthMultiplier <
    ui_category = "System / Debug";
    ui_label = "Depth Multiplier";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.1;
> = 1.0;

uniform float VERTICAL_FOV <
    ui_category = "System / Debug";
    ui_label = "FOV";
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
> = 60.0;

uniform int ViewMode <
    ui_category = "System / Debug";
    ui_label = "Debug View";
    ui_type = "combo";
    ui_items = "Off\0GI Only\0AO Only\0Surface Normals\0Motion Vectors\0Raw LowRes GI\0White World\0Luminance\0";
> = 0;

namespace Barbatos_RTGI_150
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

    //---------------------|
    // :: Vertex Shaders ::|
    //---------------------|

    void VS_Barbatos_PTGI(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, VERTICAL_FOV);
    }
    
    //-----------------|
    // :: Functions :: |
    //-----------------|

    float3 GetFalseColor(float luminance)
    {
        float3 color = float3(0.0, 0.0, 0.0);
        if (luminance < 0.25)
            color = lerp(float3(0, 0, 1), float3(0, 1, 1), luminance * 4.0);
        else if (luminance < 0.5)
            color = lerp(float3(0, 1, 1), float3(0, 1, 0), (luminance - 0.25) * 4.0);
        else if (luminance < 0.75)
            color = lerp(float3(0, 1, 0), float3(1, 1, 0), (luminance - 0.5) * 4.0);
        else
            color = lerp(float3(1, 1, 0), float3(1, 0, 0), (luminance - 0.75) * 4.0);
        return color;
    }

    float3 GetSunVector()
    {
        float az = radians(Sun_Azimuth);
        float el = radians(Sun_Elevation);
        float x = sin(az) * cos(el);
        float y = sin(el);
        float z = cos(az) * cos(el);
        return normalize(float3(x, y, z));
    }

    float GetDepth(float2 xy)
    {
        float depth = bb::GetLinearizedDepth(xy);
        if (EnableDepthMultiplier)
        {
            depth = saturate(depth * DepthMultiplier);
        }

        return depth;
    }

    //---------------------------|
    // :: View Space & Normal :: |
    //---------------------------|
    
    float3 CalculateNormal(float2 uv, float2 pScale)
    {
        float3 center = UVToViewPos(uv, GetDepth(uv), pScale);
        float3 offset_x = UVToViewPos(uv + float2(bb::PixelSize.x, 0), GetDepth(uv + float2(bb::PixelSize.x, 0)), pScale);
        float3 offset_y = UVToViewPos(uv + float2(0, bb::PixelSize.y), GetDepth(uv + float2(0, bb::PixelSize.y)), pScale);
        float3 n = cross(center - offset_x, center - offset_y);
        float lenSq = dot(n, n);
        return (lenSq > 1e-25) ?
            n * rsqrt(lenSq) : float3(0, 0, -1);
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
        outNormal = float4(normal, d);
    }
    
    void PS_Trace(VS_OUTPUT input, out float4 outGI : SV_Target)
    {
        outGI = 0.0;
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float4 gbuffer = GetLod(sNormal, input.uv);
        float depth = gbuffer.a;
        if (depth >= 0.999 || depth > GI_RenderDistance)
        {
            outGI = float4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        float3 normal = gbuffer.rgb;
        float3 viewPos = UVToViewPos(viewUV, depth, input.pScale);
        float currentMaxRayDistance = MaxRayDistance;
        float currentAORadius = AO_Radius;
        
        // Use ZThickness 
        float currentThickness = ZThickness;
        
        float3 totalRadiance = 0.0;
        float totalVisibility = 0.0;

        uint pixelIndex = uint((input.vpos.y / RenderScale) * BUFFER_WIDTH + (input.vpos.x / RenderScale));
        uint perFrameSeedBase = uint(FRAME_COUNT) * RayCount;

        float2 bn_uv = float2(input.vpos.xy / (RenderScale * 1024.0));
        float frame = fmod((float)FRAME_COUNT, 64.0);
        float4 bn = tex2Dlod(sTexBlueNoise, float4(bn_uv + float2(0.61803398875 * frame, 0.73205080757 * frame), 0, 0));
        float3 blueNoiseSeed = float3(bn.r, bn.g, bn.b);
        float bias = (depth * 0.002) + 0.0005;
        float3 rayOrigin = viewPos + normal * bias;
        float3 sunDir = float3(0, 1, 0);
        
        // View Vector for Reflection
        float3 V = normalize(-viewPos);
        float NdotV = dot(normal, V);  // Backface rejection
        if (Manual_Sun_Enabled || SSS_Enabled)
            sunDir = GetSunVector();
            
        [loop]
        for (int s = 0; s < RayCount; s++)
        {
            uint currentSeed = perFrameSeedBase + s;
            float3 rand = N_ToroidalJitter(N_Sequence3D(currentSeed), blueNoiseSeed);
            float3 rayDir;

            if (Manual_Sun_Enabled)
            {
                float3 jitter = (rand - 0.5) * Shadow_Softness;
                rayDir = normalize(sunDir + jitter);
                if (dot(normal, rayDir) <= 0.0)
                {
                    totalVisibility += Sun_Shadow_Fill;
                    continue;
                }
            }
            else
            {
                // VNDF GGX Sampling
                float3 H = RT_ImportanceSampleGGX_VNDF(rand.xy, normal, V, Roughness);
                rayDir = reflect(-V, H);
                
                // Fallback
                if (dot(normal, rayDir) <= 0.0)
                    rayDir = RT_CosineSample(normal, rand.xy);
            }
    
            float2 hitUV;
            float3 hitPos;
            float hitDist;

            if (RT_TraceRayGI(rayOrigin, rayDir, input.pScale, RaySteps, rand.z, currentMaxRayDistance, currentThickness, hitUV, hitPos, hitDist))
            {
                float4 hitGbuffer = tex2Dlod(sNormal, float4(hitUV * RenderScale, 0, 0));
                float3 hitNormal = hitGbuffer.rgb;
                float hitBufferDepth = hitGbuffer.a;
                
                // Z-Thickness Validation: check if the ray fell behind the geometry's backface
                float3 hitBufferViewPos = UVToViewPos(hitUV, hitBufferDepth, input.pScale);
                float zDistance = abs(hitPos.z - hitBufferViewPos.z);
                bool validZThickness = (zDistance <= currentThickness);

                bool validHit = Manual_Sun_Enabled ? validZThickness : ((dot(rayDir, hitNormal) < 0.1) && validZThickness);
                
                if (validHit)
                {
                    float3 rawAlbedo = tex2Dlod(sTexColorCopy, float4(hitUV, 0, 3.0)).rgb;
                    float3 linearAlbedo = GetStrictLinearAlbedo(rawAlbedo);
                    
                    float albedoLuma = GetLuminance(linearAlbedo);
                    float3 chroma = linearAlbedo - albedoLuma;
                    linearAlbedo = saturate(albedoLuma + chroma * GI_Bounce_Saturation); 
                    linearAlbedo *= GI_Bounce_Energy;
                    
                    // Infinite Bounces
                    if (MultiBounce_Weight > 0.0)
                    {
                        float3 prevGI = (uint(FRAME_COUNT) % 2 == 0) ?
                            GetLod(sHistory1, hitUV * RenderScale).rgb : 
                            GetLod(sHistory0, hitUV * RenderScale).rgb;
                        linearAlbedo *= (1.0 + prevGI * MultiBounce_Weight);
                    }
                    
                    if (!Manual_Sun_Enabled)
                        totalRadiance += linearAlbedo;
                    else
                        totalRadiance += linearAlbedo * Sun_Shadow_Fill;
                }
        
                float distFactor = saturate(hitDist / max(0.001, currentAORadius));
                float weight_falloff = saturate(1.0 - distFactor * distFactor); // Quadratic
                float weight = Manual_Sun_Enabled ?
                    Sun_Shadow_Fill : weight_falloff;

                totalVisibility += weight;
            }
            else
            {
                if (Manual_Sun_Enabled)
                {
                    totalVisibility += 1.0;
                    totalRadiance += 0.1;
                }
            }
        }

        float invRays = 1.0 / float(max(1, RayCount));
        float finalVisibility;

        if (Manual_Sun_Enabled)
        {
            finalVisibility = totalVisibility * invRays;
            finalVisibility = lerp(1.0, finalVisibility, AO_Intensity);
        }
        else
        {
            finalVisibility = 1.0 - saturate(totalVisibility * invRays * AO_Intensity);
            if (SSS_Enabled)
            {
                float3 rand = N_ToroidalJitter(N_Sequence3D(perFrameSeedBase), blueNoiseSeed);
                float3 jitter = (rand - 0.5) * Shadow_Softness;
                float3 shadowRayDir = normalize(sunDir + jitter);
                if (dot(normal, shadowRayDir) > 0.0)
                {
                    float2 sUV;
                    float3 sPos;
                    float sDist;

                    if (RT_TraceRayGI(rayOrigin, shadowRayDir, input.pScale, RaySteps, rand.z, currentMaxRayDistance, currentThickness, sUV, sPos, sDist))
                    {
                        float2 edgeFade = smoothstep(0.0, 0.05, sUV) * (1.0 - smoothstep(0.95, 1.0, sUV));
                        float screenFade = edgeFade.x * edgeFade.y;
                        finalVisibility *= lerp(1.0, (1.0 - SSS_Intensity), screenFade);
                    }
                }
            }
        }

        outGI = float4(totalRadiance * invRays * saturate(NdotV * 8.0), finalVisibility);
    }

    float4 AtrousFilter(VS_OUTPUT input, sampler sInputTex, float stepWidth)
    {
        if (any(input.uv > RenderScale))
            discard;
        float4 c_data = GetLod(sInputTex, input.uv);
        float3 c_val = c_data.rgb;
        float c_ao = c_data.a;
        
        float4 c_gbuffer = GetLod(sNormal, input.uv);
        float3 c_norm = c_gbuffer.rgb;
        float c_depth = c_gbuffer.a;
        
        static const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };
        hfloat4 sum = hfloat4(c_val, c_ao);
        hfloat cum_w = 1.0;
        
        float2 px = bb::PixelSize * stepWidth;
        float depth_weight_factor = ComputeDepthWeight(c_depth, 0.1);
        
        [loop]
        for (int x = -2; x <= 2; x++)
        {
            [loop]
            for (int y = -2; y <= 2; y++)
            {
                if (x == 0 && y == 0) continue;
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
            return float4(0.0, 0.0, 0.0, 1.0);
            
        float4 current_gi = GetLod(sAccum, input.uv);
        
        float2 velocity = MV_GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        float2 reprojected_buffer_uv = reprojected_view_uv * RenderScale;
        
        float4 history_gi = TAA_SampleHistoryCatmullRom(sHistoryParams, reprojected_buffer_uv, float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float3 current_compressed = TAA_Compress(current_gi.rgb);
        float3 current_ycocg = RGBToYCoCg(current_compressed);
        
        float3 history_compressed = TAA_Compress(history_gi.rgb);
        float3 history_ycocg = RGBToYCoCg(history_compressed);
        
        float raw_confidence = saturate(MV_GetConfidence(viewUV));
        float4 color_min, color_max;
        TAA_ComputeNeighborhoodVariance(sAccum, input.uv, current_gi, bb::PixelSize, color_min, color_max);

        float relax_amount = 0.15 * raw_confidence;
        color_min -= relax_amount;
        color_max += relax_amount;
        float3 clipped_history_ycocg = TAA_ClipToAABB(color_min.rgb, color_max.rgb, history_ycocg);
        float clipped_history_a = clamp(history_gi.a, color_min.a, color_max.a);
        
        float clamp_distance = length(clipped_history_ycocg - history_ycocg);
        float blend_adapt = saturate(1.0 - clamp_distance * 2.0); 
        
        float max_feedback = 0.98;
        float min_feedback = 0.85;
        float final_feedback = lerp(min_feedback, max_feedback, raw_confidence) * lerp(0.8, 1.0, blend_adapt);
        
        float prevRenderScale = tex2Dlod(sRS_Prev, float4(0, 0, 0, 0)).x;
        if (abs(RenderScale - prevRenderScale) > 0.001)
            final_feedback = 0.0;
            
        float3 result_ycocg = lerp(current_ycocg, clipped_history_ycocg, final_feedback);
        float3 result_compressed = YCoCgToRGB(result_ycocg);
        float result_alpha = lerp(current_gi.a, clipped_history_a, final_feedback);
        
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

    float sdTorus(float3 p, float2 t)
    {
        float2 q = float2(length(p.xz) - t.x, p.y);
        return length(q) - t.y;
    }
    
    float GetBit(int n, int b)
    {
        return fmod(floor(float(n) / exp2(float(b))), 2.0);
    }
    
    float GetDigit(float2 uv, int d)
    {
        int font[10] =
        {
            31599, // 0
            9362,  // 1
            29671, // 2
            29391, // 3
            23497, // 4
            31183, // 5
            31215, // 6
            29257, // 7
            31727, // 8
            31695  // 9
        };
        int2 ip = int2(floor(uv * float2(3.0, 5.0)));
        if (ip.x < 0 || ip.x > 2 || ip.y < 0 || ip.y > 4)
            return 0.0;
        int bit = (4 - ip.y) * 3 + (2 - ip.x);
        return GetBit(font[d], bit);
    }
    
    float PrintNumber(float2 uv, float2 pos, float size, int number)
    {
        float2 localUV = (uv - pos) / size;
        if (localUV.y < 0.0 || localUV.y > 1.0)
            return 0.0;
            
        float res = 0.0;
        uint unumber = uint(abs(number));
        int d1 = int(unumber / 100u);
        int d2 = int((unumber / 10u) % 10u);
        int d3 = int(unumber % 10u);
       
        float spacing = 0.4;
        
        // Digit 1
        float2 digitUV = localUV;
        digitUV.x = (localUV.x) * 2.0;
        if (number >= 100)
            if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
                res += GetDigit(digitUV, d1);
                
        // Digit 2
        digitUV.x = (localUV.x - 0.6) * 2.0;
        if (number >= 10)
            if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
                res += GetDigit(digitUV, d2);
                
        // Digit 3
        digitUV.x = (localUV.x - 1.2) * 2.0;
        if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
            res += GetDigit(digitUV, d3);
            
        return res;
    }

    float4 DrawSunWidget(float2 texcoord, float3 sunDir, float4 sceneColorLinear)
    {
        float2 uv = texcoord - Sun_Widget_Pos;
        uv.x *= bb::AspectRatio;
        uv /= Sun_Widget_Scale;

        float distCenter = length(uv);
        if (distCenter > 2.0)
            return sceneColorLinear;
            
        float3 ro = float3(0, 0, -3.5);
        float3 rd = normalize(float3(uv, 2.0));
        
        // Tilt camera to see the floor
        float thV = radians(30.0);
        float cV = cos(thV);
        float sV = sin(thV);
        float3x3 mTilt = float3x3(1, 0, 0, 0, cV, -sV, 0, sV, cV);
        ro = mul(mTilt, ro);
        rd = mul(mTilt, rd);

        float3 p = ro;
        float t = 0.0;
        bool hit = false;
        int objID = 0;
        float glowAcc = 0.0;
        
        float radAz = radians(-Sun_Azimuth);
        float radEl = radians(Sun_Elevation);
        
        // Calculate Sun Position 
        float3 sunPos = float3(
            sin(radAz) * cos(radEl),
            sin(radEl),
            cos(radAz) * cos(radEl)
        ) * 1.2;
        
        [loop]
        for (int i = 0; i < 60; i++)
        {
            p = ro + rd * t;
            
            // Center Cross 
            float dCross = min(length(p.xy), min(length(p.xz), length(p.yz))) - 0.01;
            float dAnchor = max(length(p) - 0.2, dCross);
            
            //Compass Ring
            float dCompass = abs(length(p.xz) - 1.2) - 0.02;
            dCompass = max(dCompass, abs(p.y) - 0.01);
            
            // Elevation Ring 
            float3 pElv = p;
            float cA = cos(radAz);
            float sA = sin(radAz);
            pElv.xz = float2(pElv.x * cA - pElv.z * sA, pElv.x * sA + pElv.z * cA);
            float dElvRing = length(float2(length(pElv.zy) - 1.2, pElv.x)) - 0.015;
            
            // Sun Sphere
            float dSun = length(p - sunPos) - 0.15;
            float dScene = min(dAnchor, min(dCompass, min(dElvRing, dSun)));
            
            glowAcc += 1.0 / (1.0 + dSun * dSun * 100.0);
            
            if (dScene < 0.002)
            {
                hit = true;
                if (dScene == dSun)
                    objID = 1; // Sun
                else if (dScene == dCompass)
                    objID = 2; // Horizontal Ring
                else if (dScene == dElvRing)
                    objID = 3; // Vertical Ring
                else
                    objID = 4; // Center
                break;
            }
            t += dScene * 0.8;
            if (t > 8.0)
                break;
        }

        float shadowMask = smoothstep(1.8, 0.4, distCenter) * 0.6;
        float3 finalColor = sceneColorLinear.rgb * (1.0 - shadowMask);

        if (hit)
        {
            float3 N = normalize(p);
            
            // Fake simple lighting
            float3 L = normalize(float3(0.5, 1.0, -0.5));
            float NdotL = max(0.2, dot(N, L));
            
            float3 objColor = float3(0, 0, 0);
            if (objID == 1) // Sun
                objColor = float3(1.0, 0.9, 0.5) * 4.0; // Emission
            else if (objID == 2) // Compass Ring 
                objColor = float3(1.0, 0.5, 0.0);
            else if (objID == 3) // Vertical Ring 
                objColor = float3(0.0, 0.8, 1.0);
            else if (objID == 4) // Center 
                objColor = float3(0.5, 0.5, 0.5);
                
            finalColor = objColor;
        }

        float3 glowColor = float3(1.0, 0.6, 0.2) * glowAcc * 0.05;
        finalColor += glowColor;
        
        // Text: Rotation 
        float numMask = PrintNumber(uv, float2(0.0, -1.2), 0.15, int(Sun_Azimuth));
        // Text: Elevation
        numMask += PrintNumber(uv, float2(0.0, 1.2), 0.15, int(Sun_Elevation));
        
        finalColor = lerp(finalColor, float3(1.0, 1.0, 1.0), numMask);
        
        float edgeFade = 1.0 - smoothstep(1.4, 1.6, distCenter);
        return lerp(sceneColorLinear, float4(finalColor, 1.0), edgeFade);
    }

    float4 JointBilateralUpsample(float2 uv, float highDepth, float2 pScale)
    {
        float2 lowResUV = uv * RenderScale;
        float3 highNormal = CalculateNormal(uv, pScale);

        float4 result = GetLod(sDNB, lowResUV);
        hfloat4 sumGI = 0.0;
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
                float4 gi = GetLod(sDNB, sampleUV);
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
                sumGI += gi * weight;
                sumWeight += weight;
            }
        }

        if (sumWeight >= 1e-6)
            result = sumGI / sumWeight;
            
        return result;
    }

    void PS_Output(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        float depth = GetDepth(input.uv);
        float3 rawScene = GetColor(input.uv).rgb;
        float3 scene = Input2Linear(rawScene);
        float3 finalColor = scene;
        
        if (depth < 0.99)
        {
            float4 giData = JointBilateralUpsample(input.uv, depth, input.pScale);
            
            float3 n = GetLod(sNormal, input.uv * RenderScale).rgb;
            float3 vp = UVToViewPos(input.uv, depth, input.pScale);
            float backNdotV = dot(n, normalize(-vp));
            giData.rgb *= saturate(backNdotV * 8.0);

            // Tint
            float3 tint = Use_Color_Temperature ?
                KelvinToRGB(Color_Temperature) : GI_Color;
            float3 processedGI = giData.rgb * tint;
            
            //HDR Contrast
            float paper_white_norm = 80.0 / HDR_Peak_Nits;
            float mid_gray = paper_white_norm * 0.18;
            processedGI = (processedGI - mid_gray) * GI_Contrast + mid_gray;
            processedGI = max(0.0, processedGI);
            
            // Vibrance
            float lum = GetLuminance(processedGI);
            float3 chroma = processedGI - lum;
            processedGI = lum + chroma * (1.0 + GI_Vibrance);
            
            float fadeStart = GI_RenderDistance * 0.9;
            float fade = 1.0 - smoothstep(fadeStart, GI_RenderDistance, depth);
            float depthWeight = lerp(Near_Intensity, 1.0, saturate(depth * 10.0));

            float rawAO = saturate(giData.a);
            float finalAO = 1.0;
            
            if (AO_BlendMode == 0)
            {
                finalAO = lerp(1.0, rawAO, AO_Intensity);
            }
            else
            {
                float sceneLum = GetLuminance(scene);
                float brightMask = saturate(sceneLum / paper_white_norm);
                finalAO = lerp(1.0, lerp(rawAO, 1.0, brightMask), AO_Intensity);
            }

            finalAO = lerp(1.0, finalAO, depthWeight);
            finalAO = lerp(1.0, finalAO, fade);
            
            float3 occludedScene = scene * finalAO;
            float3 bouncedLight = processedGI * Intensity * depthWeight * fade;
            
            // Debug Views
            if (ViewMode != 0)
            {
                switch (ViewMode)
                {
                    case 1: // GI Only
                        outColor = float4(Linear2Output(processedGI), 1.0);
                        break;
                    case 2: // AO Only
                        outColor = float4(finalAO, finalAO, finalAO, 1.0);
                        break;
                        
                    case 3: // Surface Normals
                    {
                        float3 debugNormals = GetLod(sNormal, input.uv * RenderScale).rgb;
                        if (depth < 0.999)
                        {
                            debugNormals.x = -debugNormals.x;
                            debugNormals.z = -debugNormals.z;
                        }
                        outColor = float4(debugNormals * 0.5 + 0.5, 1.0);
                        break;
                    }
                    
                    case 4: // Motion Vectors
                    {
                        float2 mv = SampleMotionVectors(input.uv);
                        outColor = float4(saturate(float3(mv.x, mv.y, 0.0) * 50.0 + 0.5), 1.0);
                        break;
                    }
                    
                    case 5: // Raw LowRes GI
                        outColor = float4((uint(FRAME_COUNT) % 2 == 0 ? GetLod(sHistory0, input.uv * RenderScale).rgb : GetLod(sHistory1, input.uv * RenderScale).rgb), 1.0);
                        break;
                        
                    case 6: // White World
                    {
                        float3 clayColor = float3(0.5, 0.5, 0.5);
                        float3 clayComposite = (clayColor * finalAO) + bouncedLight;
                        outColor = float4(Linear2Output(clayComposite), 1.0);
                        break;
                    }
                        
                    case 7: // Luminance
                        {
                            float debugLum = GetLuminance(processedGI);
                            outColor = float4(GetFalseColor(saturate(debugLum)), 1.0);
                            break;
                        }
                    default:
                        outColor = 0.0;
                        break;
                    }

                if (Manual_Sun_Enabled && Show_Sun_Widget)
                {
                    float3 debugLinear = Input2Linear(outColor.rgb);
                    float4 widgetRes = DrawSunWidget(input.uv, GetSunVector(), float4(debugLinear, 1.0));
                    outColor = float4(Linear2Output(widgetRes.rgb), 1.0);
                }
                
                return;
            }

            // Split Toning
            float sceneLuma = GetLuminance(scene);
            float3 approxAlbedo = scene / max(sceneLuma, 0.05); 
            approxAlbedo = saturate(approxAlbedo);

            float luma_normalized = saturate(sceneLuma / (paper_white_norm * 3.0));
            float transition_width = 0.45;
            float lower_bound = saturate(GI_Split_Balance - transition_width);
            float upper_bound = saturate(GI_Split_Balance + transition_width);

            float shadowCurve = 1.0 - smoothstep(lower_bound, upper_bound, luma_normalized);
            float highlightCurve = smoothstep(lower_bound, upper_bound, luma_normalized);
            
            float3 surfaceIntegration = lerp(float3(1.0, 1.0, 1.0), approxAlbedo, GI_Color_Bleed);
            
            float3 shadowLight = bouncedLight * surfaceIntegration * shadowCurve * GI_Shadow_Tint;
            float3 litLight = bouncedLight * surfaceIntegration * highlightCurve * GI_Highlight_Tint;
            
            finalColor = occludedScene + shadowLight + litLight;
        }

        // Widget Overlay
        if (Manual_Sun_Enabled && Show_Sun_Widget && ViewMode == 0)
        {
            float4 widgetRes = DrawSunWidget(input.uv, GetSunVector(), float4(finalColor, 1.0));
            finalColor = widgetRes.rgb;
        }

        outColor = float4(Linear2Output(finalColor), 1.0);
    }
    
    void PS_SaveScale(VS_OUTPUT input, out float4 outScale : SV_Target)
    {
        outScale = float4(RenderScale, 0.0, 0.0, 1.0);
    }

    technique BaBa_GI
    <
    ui_label = "BaBa: GI";
    ui_tooltip = "GI, AO and Shadows";
    >
    {
        pass CopyColorGenMips
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_CopyColor;
            RenderTarget = TexColorCopy;
        }
        pass Normals
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_GenNormals;
            RenderTarget = Normal;
        }
        pass Trace
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Trace;
            RenderTarget = Accum;
        }
        pass Accumulate0
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass DenoiseStep1
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Atrous1;
            RenderTarget = DNA;
        }
        pass DenoiseStep2
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_AtrousFinal;
            RenderTarget = DNB;
        }
        pass Output
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Output;
        }
        pass SaveScale
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_SaveScale;
            RenderTarget = RS_Prev;
        }
    }
}