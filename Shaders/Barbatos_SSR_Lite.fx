/*----------------------------------------------|
| :: Barbatos SSR  LITE                      :: |
'-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/
//Contains AI-assisted content.

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "Blending.fxh"

#define Quality 1     
#define RenderScale 0.5

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

//----------|
// :: UI :: |
//----------|

uniform float ReflectionIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Reflection Strength";
    ui_tooltip = "Overall intensity of reflections";
> = 1.1;

uniform int ReflectionMode <
    ui_type = "combo";
    ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
    ui_category = "Basic Settings";
    ui_label = "Reflection Surfaces";
    ui_tooltip = "Choose which surfaces show reflections";
> = 0;

uniform float FadeDistance <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Fade Distance";
    ui_tooltip = "How far away reflections start to fade out";
> = 4.999;

uniform float MetallicLook <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Surface Quality";
    ui_label = "Metallic Look";
    ui_tooltip = "Make surfaces look more metallic (0=non-metal, 1=metal)";
> = 0.2;

BLENDING_COMBO(g_BlendMode, "Blending Mode", "Select how reflections are blended with the scene.", "Color Adjustments", false, 0, 0)

uniform float g_Contrast <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Color Adjustments";
    ui_label = "Contrast";
> = 1.0;

uniform float g_Saturation <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Color Adjustments";
    ui_label = "Saturation";
> = 1.0;

uniform float VERTICAL_FOV <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
> = 37.0;

uniform float THICKNESS_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Reflection Thickness Threshold";
    ui_tooltip = "Controls how 'thick' surfaces are before a ray passes through them.";
> = 0.5;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0";
    ui_category = "Debug";
    ui_label = "Debug View";
    ui_tooltip = "Special views";
> = 0;

#define SPIntensity ReflectionIntensity
#define FadeEnd FadeDistance
#define Metallic MetallicLook
#define ViewMode (DebugView == 0 ? 0 : (DebugView == 1 ? 1 : (DebugView == 2 ? 2 : (DebugView == 3 ? 3 : 0)))) 
static const float OrientationThreshold = 0.5;

#if __RENDERER__== 0x9000
static const int STEPS_PER_RAY_WALLS_DX9 = 20;
static const int STEPS_PER_RAY_FLOOR_CEILING_QUALITY_DX9 = 64;
static const int STEPS_PER_RAY_FLOOR_CEILING_BALANCED_DX9 = 48;
static const int STEPS_PER_RAY_FLOOR_CEILING_PERF_DX9 = 32;
#else
static const int STEPS_PER_RAY_WALLS = 32;
#endif

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_SSR_Lite
{

    texture Reflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sReflection
    {
        Texture = Reflection;
    };

    texture Reconstructed
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sReconstructed
    {
        Texture = Reconstructed;
    };

    texture Upscaled
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sUpscaled
    {
        Texture = Upscaled;
    };

//-------------|
// :: Utility::|
//-------------|

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
    
    static const float DIELECTRIC_REFLECTANCE = 0.04;

    float3 F_Schlick(float VdotH, float3 f0)
    {
        return f0 + (1.0.xxx - f0) * pow(1.0 - VdotH, 5.0);
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }

    float3 AdjustContrast(float3 color, float contrast)
    {
        return (color - 0.5) * contrast + 0.5;
    }

    float3 AdjustSaturation(float3 color, float saturation)
    {
        float lum = GetLuminance(color);
        return lerp(lum.xxx, color, saturation);
    }

    float3 UVToViewPos(float2 uv, float view_z)
    {
        float fov_rad = VERTICAL_FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;
        float2 clip_pos = uv * 2.0 - 1.0;
        float3 view_pos = float3(clip_pos.x / proj_scale_x * view_z, -clip_pos.y / proj_scale_y * view_z, view_z);
        return view_pos;
    }

    float2 ViewPosToUV(float3 view_pos)
    {
        float fov_rad = VERTICAL_FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;
        float2 clip_pos = float2(view_pos.x * proj_scale_x / view_pos.z, -view_pos.y * proj_scale_y / view_pos.z);
        return clip_pos * 0.5 + 0.5;
    }

    float3 GVPFUV(float2 uv)
    {
        float depth = GetDepth(uv);
        return UVToViewPos(uv, depth);
    }

    float3 CalculateNormal(float2 texcoord)
    {
        float3 offset_x = GVPFUV(texcoord + float2(ReShade::PixelSize.x, 0.0));
        float3 offset_y = GVPFUV(texcoord + float2(0.0, ReShade::PixelSize.y));
        float3 center = GVPFUV(texcoord);
        return normalize(cross(center - offset_x, center - offset_y));
    }
    
//-------------------|
// :: Ray Tracing  ::|
//-------------------|

    HitResult TraceRay(Ray r, int num_steps)
    {
        HitResult result;
        result.found = false;

        float step_scale, min_step_size, max_step_size;
        int refinement_steps;

        refinement_steps = 0;
        step_scale = 0.7;
        min_step_size = 0.001;
        max_step_size = 1.0;

        float stepSize = min_step_size;
        float totalDist = 0.0;
        float3 prevPos = r.origin;

        [loop]
        for (int i = 0; i < num_steps; ++i)
        {
            float3 currPos = prevPos + r.direction * stepSize;
            totalDist += stepSize;

            float2 uvCurr = ViewPosToUV(currPos);
            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) || totalDist > 10.0)
                break;

            float sceneDepth = GetDepth(uvCurr);
            
            float thickness = abs(currPos.z - sceneDepth);

            if (currPos.z < sceneDepth || thickness > THICKNESS_THRESHOLD)
            {
                prevPos = currPos;
                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * step_scale, min_step_size, max_step_size);
                continue;
            }
            
            if (currPos.z < sceneDepth) 
            {
                prevPos = currPos;
                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * step_scale, min_step_size, max_step_size);
                continue;
            }

            float3 lo = prevPos, hi = currPos;
            [unroll]
            for (int ref_step = 0; ref_step < refinement_steps; ++ref_step)
            {
                float3 mid = 0.5 * (lo + hi);
                float midDepth = GetDepth(ViewPosToUV(mid));
                if (mid.z >= midDepth)
                    hi = mid;
                else
                    lo = mid;
            }
            result.viewPos = hi;
            result.uv = ViewPosToUV(result.viewPos).xy;
            result.found = true;
            return result;
        }
        return result;
    }
    
    void PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outReflection : SV_Target)
    {
        float2 scaled_uv = uv / RenderScale;
        if (any(scaled_uv > 1.0))
        {
            outReflection = float4(0, 0, 0, -1.0); 
            return;
        }

        // Checkerboard
        float2 low_res_dims = BUFFER_SCREEN_SIZE * RenderScale;
        float2 low_res_coord = floor(scaled_uv * low_res_dims);
        float checker = fmod(low_res_coord.x + low_res_coord.y, 2.0);
        if (checker != 0.0)
        {
            outReflection = float4(0, 0, 0, -1.0); 
            return;
        }

        float depth = GetDepth(scaled_uv);
        if (depth >= 1.0)
        {
            outReflection = 0; 
            return;
        }

        float3 viewPos = UVToViewPos(scaled_uv, depth);
        float3 viewDir = -normalize(viewPos);
        float3 normal = CalculateNormal(scaled_uv); 

        float fReflectFloors = 0.0, fReflectWalls = 0.0, fReflectCeilings = 0.0;
        switch (ReflectionMode)
        {
            case 0:
                fReflectFloors = 1;
                break;
            case 1:
                fReflectWalls = 1;
                break;
            case 2:
                fReflectCeilings = 1;
                break;
            case 3:
                fReflectFloors = 1;
                fReflectCeilings = 1;
                break;
            case 4:
                fReflectFloors = 1;
                fReflectWalls = 1;
                fReflectCeilings = 1;
                break;
        }

        bool isFloor = normal.y > OrientationThreshold;
        bool isCeiling = normal.y < -OrientationThreshold;
        bool isWall = abs(normal.y) <= OrientationThreshold;
        float orientationIntensity = (isFloor * fReflectFloors) + (isWall * fReflectWalls) + (isCeiling * fReflectCeilings);

        if (orientationIntensity <= 0.0)
        {
            outReflection = 0;
            return;
        }

        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(-viewDir, normal));
        r.origin += r.direction * 0.0001;
        HitResult hit;

        float3 reflectionColor = 0;
        float reflectionAlpha = 0.0;
        if (hit.found)
        {
            reflectionColor = GetColor(hit.uv).rgb;
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / 10.0);
            float fadeRange = max(FadeEnd, 0.001);
            float depthFade = saturate((FadeEnd - depth) / fadeRange);
            depthFade *= depthFade;
            reflectionAlpha = distFactor * depthFade;
        }
        else
        {
            float adaptiveDist = min(depth * 1.2 + 0.012, 10.0);
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos).xy);
            reflectionColor = GetColor(uvFb).rgb;
            float vertical_fade = pow(saturate(1.0 - scaled_uv.y), 3.0);
            reflectionAlpha = vertical_fade;
        }
        reflectionAlpha *= pow(saturate(dot(-viewDir, r.direction)), 2.0);
        reflectionAlpha *= orientationIntensity;
        outReflection = float4(reflectionColor, reflectionAlpha);
    }
    
    void PS_Reconstruct(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outRecon : SV_Target)
    {
        float4 current = GetLod(sReflection, uv);
        if (current.a >= 0.0) 
        {
            outRecon = current;
            return;
        }

        float2 p = ReShade::PixelSize;
        float4 tl = GetLod(sReflection, uv + float2(-p.x, -p.y));
        float4 tr = GetLod(sReflection, uv + float2( p.x, -p.y));
        float4 bl = GetLod(sReflection, uv + float2(-p.x,  p.y));
        float4 br = GetLod(sReflection, uv + float2( p.x,  p.y));

        float3 color_sum = 0;
        float alpha_sum = 0;
        float count = 0;

        if (tl.a >= 0.0)
        {
            color_sum += tl.rgb;
            alpha_sum += tl.a;
            count += 1.0;
        }
        if (tr.a >= 0.0)
        {
            color_sum += tr.rgb;
            alpha_sum += tr.a;
            count += 1.0;
        }
        if (bl.a >= 0.0)
        {
            color_sum += bl.rgb;
            alpha_sum += bl.a;
            count += 1.0;
        }
        if (br.a >= 0.0)
        {
            color_sum += br.rgb;
            alpha_sum += br.a;
            count += 1.0;
        }

        if (count > 0.0)
            outRecon = float4(color_sum / count, alpha_sum / count);
        else
            outRecon = 0;
    }
    
    void PS_Upscale(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outUpscaled : SV_Target)
    {
        outUpscaled = GetLod(sReconstructed, uv * RenderScale);
    }

    void PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1:
                    outColor = float4(GetLod(sUpscaled, uv).rgb, 1.0);
                    return;
                case 2:
                    outColor = float4(CalculateNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 3:
                    outColor = GetDepth(uv).xxxx;
                    return;
            }
        }

        float3 originalColor = GetColor(uv).rgb;
        if (GetDepth(uv) >= 1.0)
        {
            outColor = float4(originalColor, 1.0);
            return;
        }

        float4 reflectionSample = GetLod(sUpscaled, uv);
        float3 reflectionColor = reflectionSample.rgb;
        float reflectionMask = reflectionSample.a;

        reflectionColor = AdjustContrast(reflectionColor, g_Contrast);
        reflectionColor = AdjustSaturation(reflectionColor, g_Saturation);

        float3 viewDir = -normalize(UVToViewPos(uv, GetDepth(uv)));
        float3 normal = CalculateNormal(uv);
        float VdotN = saturate(dot(viewDir, normal));
        float3 f0 = lerp(float3(DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE), originalColor, Metallic);
        float3 F = F_Schlick(VdotN, f0);

        float blendAmount = dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask;
        float3 finalColor = ComHeaders::Blending::Blend(g_BlendMode, originalColor, reflectionColor * SPIntensity, blendAmount);
        
        outColor = float4(finalColor, 1.0);
    }

    technique Barbatos_SSR_Lite
    {
        pass TraceReflections
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TraceReflections;
            RenderTarget = Reflection;
            ClearRenderTargets = true;
        }
        pass Reconstruct
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Reconstruct;
            RenderTarget = Reconstructed;
        }
        pass Upscale
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale;
            RenderTarget = Upscaled;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }

}
