/*----------------------------------------------.
| ::             NeoSSAO 2                   :: |
|-----------------------------------------------|
| Version: 0.9.1                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/
#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#ifndef RES_SCALE
#define RES_SCALE 1.0 
#endif

#define RES_WIDTH (BUFFER_WIDTH * RES_SCALE)
#define RES_HEIGHT (BUFFER_HEIGHT * RES_SCALE)
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define PI 3.1415926535

#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))

uniform float Intensity <
    ui_category = "Occlusion Settings";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform float Power <
    ui_category = "Occlusion Settings";
    ui_type = "drag";
    ui_label = "AO Power";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;

uniform float AORadius <
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "AO Radius";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.18;

uniform int RaySteps <
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "Ray Steps";
    ui_min = 3; ui_max = 32;
> = 3;

uniform int SampleCount <
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "SampleCount";
    ui_min = 1; ui_max = 16;
> = 8;

uniform float FadeStart <
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade Start";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float FadeEnd <
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade End";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;
    
uniform bool EnableTemporal <
    ui_category = "Temporal Filter";
    ui_type = "checkbox";
    ui_label = "Enable TAA";
> = false;

uniform bool EnableDepthMultiplier <
    ui_category = "Depth & Normals";
    ui_type = "checkbox";
    ui_label = "DepthMultiplier";
> = false;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_category = "Depth & Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;
    
uniform float DepthThreshold <
    ui_type = "slider";
    ui_category = "Depth & Normals";
    ui_label = "Sky Threshold";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.999; 

uniform float FOV <
    ui_category = "Depth & Normals";
    ui_label = "Field of View (Vertical)";
    ui_type = "slider";
    ui_min = 1.0; ui_max = 120.0;
> = 75.0;

uniform float4 OcclusionColor <
    ui_category = "Debug & Style";
    ui_type = "color";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);
    
uniform int ViewMode < 
    ui_category = "Debug & Style";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "None\0AO Only\0Depth\0Normals\0Motion Vectors\0";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred 
    {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 GetMV(float2 texcoord) { return tex2D(Deferred::sMotionVectorsTex, texcoord).rg; }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 GetMV(float2 texcoord) { return tex2D(sMotVectTexVort, texcoord).rg; }
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
float2 GetMV(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace NEOSPACEAO
{
    texture2D normalTex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = normalTex;
    };

    texture2D AO
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = R8;
    };
    sampler2D sAO
    {
        Texture = AO;
    };

    texture History0
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = R8;
    };
    sampler sHistory0
    {
        Texture = History0;
    };

    texture History1
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = R8;
    };
    sampler sHistory1
    {
        Texture = History1;
    };

    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };

    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }
    
    void VS_NeoSSAO(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ? 2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        float fov_rad = FOV * (PI / 180.0);
        float y = tan(fov_rad * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }

    float3 UVToViewPos(float2 uv, float view_z, float2 pScale, float depthMult)
    {
        float2 ndc = uv * 2.0 - 1.0;
        return float3(ndc.x * pScale.x * view_z, -ndc.y * pScale.y * view_z, view_z) * depthMult;
    }
    
    float3 GVPFUV(float2 uv, float2 pScale, float depthMult)
    {
        float depth = GetDepth(uv);
        return UVToViewPos(uv, depth * FAR_PLANE, pScale, depthMult);
    }

    float3 CalculateNormal(float2 texcoord, float2 pScale, float depthMult)
    {
        float3 offset_x = GVPFUV(texcoord + float2(ReShade::PixelSize.x, 0.0), pScale, depthMult);
        float3 offset_y = GVPFUV(texcoord + float2(0.0, ReShade::PixelSize.y), pScale, depthMult);
        float3 center = GVPFUV(texcoord, pScale, depthMult);
        
        return normalize(cross(center - offset_x, center - offset_y));
    }

    float3 getNormalFromTex(float2 coords)
    {
        return tex2Dlod(sNormal, float4(coords, 0, 0)).xyz;
    }

    float GetBayer8x8(float2 uv)
    {
        int2 pixelPos = int2(uv * float2(RES_WIDTH, RES_HEIGHT));
        const int bayer[64] =
        {
            0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
            12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
            15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21
        };
        return float(bayer[(pixelPos.x % 8) + (pixelPos.y % 8) * 8]) * (1.0 / 64.0);
    }

    float3 getHemisphereSample(float3 normal, float2 random_uv)
    {
        float cos_theta = sqrt(1.0 - random_uv.x);
        float sin_theta = sqrt(random_uv.x);
        float phi = 2.0 * PI * random_uv.y;
        float3 sample_dir = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        float3 up = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
        float3 tangent = normalize(cross(up, normal));
        float3 bitangent = cross(normal, tangent);
        return sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * normal;
    }

    float GetActiveHistory(float2 uv)
    {
        return (FRAME_COUNT % 2 == 0) ? tex2Dlod(sHistory0, float4(uv, 0, 0)).r : tex2Dlod(sHistory1, float4(uv, 0, 0)).r;
    }

    float ClipToAABB(float aabb_min, float aabb_max, float history_sample)
    {
        float p_clip = 0.5 * (aabb_max + aabb_min);
        float e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float v_clip = history_sample - p_clip;
        float v_unit = v_clip / e_clip;
        float a_unit = abs(v_unit);
        float ma_unit = a_unit;
        return (ma_unit > 1.0) ? (p_clip + v_clip / ma_unit) : history_sample;
    }

    float2 GetVelocityFromClosestFragment(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0.0;
        const float2 offsets[5] = { float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1) };
        [unroll]
        for (int i = 0; i < 5; i++)
        {
            float2 s_coord = texcoord + offsets[i] * pixel_size;
            float s_depth = GetDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                closest_velocity = GetMV(s_coord);
            }
        }
        return closest_velocity;
    }

    void ComputeNeighborhoodMinMax(sampler2D color_tex, float2 texcoord, float center_val, out float val_min, out float val_max)
    {
        float min_v = center_val;
        float max_v = center_val;
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                if (x == 0 && y == 0)
                    continue;
                float s = tex2Dlod(color_tex, float4(texcoord + float2(x, y) * ReShade::PixelSize, 0, 0)).r;
                min_v = min(min_v, s);
                max_v = max(max_v, s);
            }
        }
        val_min = min_v;
        val_max = max_v;
    }

    float ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        float current_ao = tex2Dlod(sAO, float4(input.uv, 0, 0)).r;
        if (!EnableTemporal)
            return current_ao;

        float2 velocity = GetVelocityFromClosestFragment(input.uv);
        float2 reprojected_uv = input.uv + velocity;
        
        if (any(saturate(reprojected_uv) != reprojected_uv) || FRAME_COUNT <= 1)
            return current_ao;
            
        float history_ao = tex2Dlod(sHistoryParams, float4(reprojected_uv, 0, 0)).r;
        float current_depth = GetDepth(input.uv);
        float history_depth = GetDepth(reprojected_uv);

        if (abs(history_depth - current_depth) > 0.01)
            return current_ao;

        float ao_min, ao_max;
        ComputeNeighborhoodMinMax(sAO, input.uv, current_ao, ao_min, ao_max);
        float clipped_history = ClipToAABB(ao_min, ao_max, history_ao);

        float diff = abs(current_ao - clipped_history);
        float velocity_weight = saturate(1.0 - length(velocity * float2(RES_WIDTH, RES_HEIGHT) * 100.0));
        float raw_confidence = (1.0 - saturate(diff * 8.0)) * velocity_weight;
        float boost_factor = 0.5;
        float compressed_confidence = saturate(raw_confidence + log2(2.0 - raw_confidence) * boost_factor);
        float feedback = compressed_confidence * 0.95;
        
        return lerp(current_ao, clipped_history, feedback);
    }

    float4 PS_GenNormals(VS_OUTPUT input) : SV_Target
    {
        float depth = GetDepth(input.uv);
        if (depth >= DepthThreshold)
            return float4(0, 0, 1, 1);

        float realDepthMult = 1.0;
        if (EnableDepthMultiplier)
            realDepthMult = lerp(0.1, 5.0, DepthMultiplier);

        float3 normal = CalculateNormal(input.uv, input.pScale, realDepthMult);
        return float4(normal, 1.0);
    }

    float4 PS_SSAO(VS_OUTPUT input) : SV_Target
    {
        float center_depth = GetDepth(input.uv);
        if (center_depth >= DepthThreshold)
            return 1.0;

        float realDepthMult = 1.0;
        if (EnableDepthMultiplier)
            realDepthMult = lerp(0.1, 5.0, DepthMultiplier);
        
        float2 invPScale = 1.0 / input.pScale;

        float3 center_pos = UVToViewPos(input.uv, center_depth * FAR_PLANE, input.pScale, realDepthMult);
        float3 normal = getNormalFromTex(input.uv);
        float random_angle = GetBayer8x8(input.uv) * 2.0 * PI;
        
        float occlusion = 0.0;
        float realRadius = lerp(0.1, 5.0, AORadius);

   
        for (int i = 0; i < SampleCount; i++)
        {
            float2 sequence = float2((i + 0.5) / SampleCount, frac((i * 0.61803398875) + random_angle / (2.0 * PI)));
            float3 ray_dir = getHemisphereSample(normal, sequence);
            float ray_occlusion = 0.0;

            [loop]
            for (int j = 1; j <= RaySteps; j++)
            {
                float ray_dist = (float(j) / RaySteps) * realRadius;
                float3 sample_pos = center_pos + ray_dir * ray_dist;

                if (sample_pos.z > 1e-6)
                {
                    float invZ = 1.0 / sample_pos.z;
                    float2 projected = sample_pos.xy * invZ * invPScale;
                    
                    float2 sample_uv;
                    sample_uv.x = projected.x * 0.5 + 0.5;
                    sample_uv.y = -projected.y * 0.5 + 0.5;

                    if (all(saturate(sample_uv) == sample_uv))
                    {
                        float scene_depth = GetDepth(sample_uv);
                        float scene_z = scene_depth * FAR_PLANE * realDepthMult;
                        
                        if (scene_z < sample_pos.z - 0.005)
                        {
                            float falloff = 1.0 - smoothstep(0.0, 1.0, ray_dist / realRadius);
                            ray_occlusion = max(ray_occlusion, falloff);
                        }
                    }
                }
            }
            occlusion += ray_occlusion;
        }

        occlusion /= SampleCount;
        
        float realIntensity = Intensity * 2.0;
        float realPower = lerp(0.5, 8.0, Power);
        occlusion = pow(saturate(occlusion * realIntensity), realPower);

        float realFadeEnd = FadeEnd * 2.0;
        float fade = smoothstep(realFadeEnd, FadeStart, center_depth);
        occlusion *= fade;

        return 1.0 - saturate(occlusion);
    }

    float4 PS_Accumulate0(VS_OUTPUT input) : SV_Target
    {
        if (FRAME_COUNT % 2 != 0)
            discard;
        return ComputeTAA(input, sHistory1);
    }

    float4 PS_Accumulate1(VS_OUTPUT input) : SV_Target
    {
        if (FRAME_COUNT % 2 == 0)
            discard;
        return ComputeTAA(input, sHistory0);
    }

    float4 PS_Output(VS_OUTPUT input) : SV_Target
    {
        float4 originalColor = GetColor(input.uv);
        float occlusion = GetActiveHistory(input.uv);

        switch (ViewMode)
        {
            case 0:
                originalColor.rgb *= occlusion;
                originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, 1.0 - occlusion);
                return originalColor;
            case 1:
                return 1.0 - (1.0 - occlusion);
            case 2:
                return GetDepth(input.uv);
            case 3:
                return float4(getNormalFromTex(input.uv) * 0.5 + 0.5, 1.0);
            case 4:
                return float4(GetMV(input.uv) * 100.0 + 0.5, 0.0, 1.0);
        }
        return originalColor;
    }

    technique NeoSSAO
    {
        pass GenNormals
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_GenNormals;
            RenderTarget = normalTex;
        }
        pass SSAOPass
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_SSAO;
            RenderTarget = AO;
        }
        pass Accumulate0
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass OutputPass
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_Output;
        }
    }
}
