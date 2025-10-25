/*----------------------------------------------|
| :: Barbatos SSR (Screen-Space Reflections) :: |
'-----------------------------------------------|
| Version: 0.3.1                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))

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
> = 4;

uniform float FadeDistance <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Fade Distance";
    ui_tooltip = "How far away reflections start to fade out";
> = 4.999;

uniform float SurfaceSharpness <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Surface Quality";
    ui_label = "Surface Sharpness";
    ui_tooltip = "How clear or blurry reflections appear (0=blurry, 1=sharp)";
> = 0.75;

uniform float MetallicLook <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Surface Quality";
    ui_label = "Metallic Look";
    ui_tooltip = "Make surfaces look more metallic (0=non-metal, 1=metal)";
> = 0.2;

uniform float SurfaceDetails <
    ui_label = "Surface Details";
    ui_type = "drag";
    ui_category = "Surface Quality";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_tooltip = "Adds small surface details to reflections";
> = 0.1;

uniform int QualityPreset <
    ui_type = "combo";
    ui_items = "Balanced\0Performance\0";
    ui_category = "Performance";
    ui_label = "RT Quality";
    ui_tooltip = "Higher quality = better reflections but slower performance";
> = 1;

uniform float RenderResolution <
    ui_type = "drag";
    ui_min = 0.3; ui_max = 1.0; ui_step = 0.05;
    ui_category = "Performance";
    ui_label = "Render Resolution";
    ui_tooltip = "Lower values = better performance but less details";
> = 0.8;

uniform bool EnableSmoothing <
    ui_category = "Advanced Options";
    ui_label = "Reduce Reflection Noise (TAA)";
    ui_tooltip = "Reduces flickering and noise in reflections using temporal anti-aliasing";
> = true;

uniform int SmartSurfaceMode <
    ui_type = "combo";
    ui_items = "Off\0Performance\0Balanced\0Quality\0";
    ui_category = "Advanced Options";
    ui_label = "Smart Surface Detection";
    ui_tooltip = "Quality of the normal smoothing filter.";
> = 1;

uniform float THICKNESS_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.5; ui_step = 0.001;
    ui_category = "Advanced Options";
    ui_label = "Reflection Thickness Threshold";
    ui_tooltip = "Controls how 'thick' surfaces are before a ray passes through them.";
> = 0.009;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0Motion\0";
    ui_category = "Debug";
    ui_label = "Debug View";
    ui_tooltip = "Special views";
> = 0;

#define SPIntensity ReflectionIntensity
#define FadeEnd FadeDistance
#define Roughness (1.0 - SurfaceSharpness)
#define Metallic MetallicLook
#define BumpIntensity SurfaceDetails
#define EnableTAA EnableSmoothing
#define Quality (QualityPreset)
#define RenderScale RenderResolution
#define ViewMode (DebugView == 0 ? 0 : (DebugView == 1 ? 1 : (DebugView == 2 ? 2 : (DebugView == 3 ? 3 : 5))))

static const float SobelEdgeThreshold = 0.03;
static const float Smooth_Threshold = 0.5;
static const int GlossySamples = 10;
static const float FeedbackFactor = 0.99;
static const float OrientationThreshold = 0.5;
static const float GeoCorrectionIntensity = -0.01;
static const float VERTICAL_FOV = 37.0;
static const bool EnableGlossy = true;
#define SmoothMode (SmartSurfaceMode) 

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
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
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
#endif

#if !USE_MARTY_LAUNCHPAD_MOTION && !USE_VORT_MOTION
float2 SampleMotionVectors(float2 texcoord)
{
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
#elif USE_MARTY_LAUNCHPAD_MOTION
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, texcoord).rg;
    }
#endif

namespace Barbatos_SSR202
{
    texture TNormal
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = TNormal;
    };

    texture NormTex_Pass1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormTex_Pass1
    {
        Texture = NormTex_Pass1;
    };

    texture NormTex_Pass2
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormTex_Pass2
    {
        Texture = NormTex_Pass2;
    };

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

    texture Temp
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sTemp
    {
        Texture = Temp;
    };

    texture History
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sHistory
    {
        Texture = History;
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
    
    float3 HSVToRGB(float3 c)
    {
        const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }

    float2 GetMotionVectors(float2 texcoord)
    {
        return SampleMotionVectors(texcoord);
    }
    
    float2 ConcentricSquareMapping(float2 u)
    {
        float a = 2.0 * u.x - 1.0;
        float b = 2.0 * u.y - 1.0;

        float r, phi;

        if ((a * a) > (b * b))
        {
            r = a;
            phi = (PI / 4.0) * (b / a);
        }
        else
        {
            r = b;
            if (b != 0.0)
            {
                phi = (PI / 2.0) - (PI / 4.0) * (a / b);
            }
            else
            {
                phi = 0.0;
            }
        }
        
        float X = r * cos(phi);
        float Y = r * sin(phi);

        return float2(X, Y);
    }
    
//------------------------------------|
// :: View Space & Normal Functions ::|
//------------------------------------|

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
    
    float3 ApplyBumpMapping(float2 texcoord, float3 normal)
    {
        if (BumpIntensity == 0.0)
            return normal;

        float l00 = GetColor(texcoord + ReShade::PixelSize * float2(-1, -1)).g, l10 = GetColor(texcoord + ReShade::PixelSize * float2( 0, -1)).g, l20 = GetColor(texcoord + ReShade::PixelSize * float2( 1, -1)).g;
        float l01 = GetColor(texcoord + ReShade::PixelSize * float2(-1,  0)).g, l21 = GetColor(texcoord + ReShade::PixelSize * float2( 1,  0)).g;
        float l02 = GetColor(texcoord + ReShade::PixelSize * float2(-1,  1)).g, l12 = GetColor(texcoord + ReShade::PixelSize * float2( 0,  1)).g, l22 = GetColor(texcoord + ReShade::PixelSize * float2( 1,  1)).g;

        float Gx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
        float Gy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);

        if (length(float2(Gx, Gy)) < SobelEdgeThreshold)
            return normal;

        float2 slope = float2(Gx, Gy) * BumpIntensity;
        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T = normalize(cross(up, normal));
        float3 B = cross(normal, T);
        return normalize(normal + T * slope.x - B * slope.y);
    }

    float3 GeometryCorrection(float2 texcoord, float3 normal)
    {
        if (GeoCorrectionIntensity == 0.0)
            return normal;
        float lumCenter = GetLuminance(GetColor(texcoord).rgb);
        float lumRight = GetLuminance(GetColor(texcoord + ReShade::PixelSize * int2(1, 0)).rgb);
        float lumDown = GetLuminance(GetColor(texcoord + ReShade::PixelSize * int2(0, 1)).rgb);
        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));
        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }
    
    float3 SampleNormal(float2 uv)
    {
        return (GetLod(sNormal, uv).xyz - 0.5) * 2.0;
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

        if (Quality == 1) // Performance
        {
            refinement_steps = 5;
            step_scale = 0.7;
            min_step_size = 0.001;
            max_step_size = 1.0;
        }
        else if (Quality == 0) // Balanced
        {
            refinement_steps = 5;
            step_scale = 0.4;
            min_step_size = 0.0005;
            max_step_size = 1.0;
        }

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

//---------------|
// :: Glossy  :: |
//---------------|

    float specularPowerToConeAngle(float specularPower)
    {
        if (specularPower >= exp2(12.0))
            return 0.0f;
        const float xi = 0.244f;
        float exponent = 1.0f / (specularPower + 1.0f);
        return acos(pow(xi, exponent));
    }
    float isoscelesTriangleOpposite(float adjacentLength, float coneTheta)
    {
        return 2.0f * tan(coneTheta) * adjacentLength;
    }
    float isoscelesTriangleInRadius(float a, float h)
    {
        float a2 = a * a;
        float fh2 = 4.0f * h * h;
        return (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);
    }
    
    float3 GetGlossySample(float2 sample_uv, float2 pixel_uv)
    {
        if (!EnableGlossy || Roughness <= 0.0)
            return GetColor(float4(sample_uv, 0, 0)).rgb;

        float gloss = 1.0 - Roughness;
        float specularPower = pow(2.0, 10.0 * gloss + 1.0);
        float coneTheta = specularPowerToConeAngle(specularPower) * 0.5;

        if (coneTheta <= 0.001)
            return GetColor(float4(sample_uv, 0, 0)).rgb;

        float2 deltaP = (sample_uv - pixel_uv) * BUFFER_SCREEN_SIZE;
        float adjacentLength = length(deltaP);

        if (adjacentLength <= 1.0)
            return GetColor(float4(sample_uv, 0, 0)).rgb;

        float oppositeLength = isoscelesTriangleOpposite(adjacentLength, coneTheta);
        float incircleSize = isoscelesTriangleInRadius(oppositeLength, adjacentLength);
        float blurRadiusUV = incircleSize * ReShade::PixelSize.x;

        float3 reflectionColor = 0.0.xxx;
        [loop]
        for (int i = 0; i < GlossySamples; ++i)
        {
            float2 random_seed = pixel_uv * BUFFER_SCREEN_SIZE + float2(i, i * 2.0) + float2(float(FRAME_COUNT % 100), float(FRAME_COUNT % 50));
            float2 u = frac(sin(float2(dot(random_seed, float2(12.9898, 78.233)), dot(random_seed, float2(39.345, 41.123)))) * 43758.5453);
            float2 offset = ConcentricSquareMapping(u) * blurRadiusUV;
            reflectionColor += GetColor(float4(sample_uv + offset, 0, 0)).rgb;
        }
        return reflectionColor / float(GlossySamples);
    }
    
//------------|
// :: TAA  :: |
//------------|

    float3 ClipToAABB(float3 aabb_min, float3 aabb_max, float3 history_sample)
    {
        float3 p_clip = 0.5 * (aabb_max + aabb_min);
        float3 e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float3 v_clip = history_sample - p_clip;
        float3 v_unit = v_clip / e_clip;
        float3 a_unit = abs(v_unit);
        float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
        if (ma_unit > 1.0)
            return p_clip + v_clip / ma_unit;
        else
            return history_sample;
    }
    float2 GetVelocityFromClosestFragment(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0;
        const int2 offsets[9] = { int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(0, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1) };[unroll]
        for (int i = 0; i < 9; i++)
        {
            float2 s_coord = texcoord + offsets[i] * pixel_size;
            float s_depth = GetDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                closest_velocity = SampleMotionVectors(s_coord);
            }
        }
        return closest_velocity;
    }
    void ComputeNeighborhoodMinMax(sampler2D color_tex, float2 texcoord, out float3 color_min, out float3 color_max)
    {
        float2 pixel_size = ReShade::PixelSize / RenderScale;
        float3 center_color = GetLod(color_tex, float4(texcoord, 0, 0)).rgb;
        color_min = center_color;
        color_max = center_color;
        const int2 offsets_3x3[8] = { int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1) };[unroll]
        for (int i = 0; i < 8; i++)
        {
            float3 n_color = GetLod(color_tex, float4(texcoord+offsets_3x3[i]*pixel_size,0,0)).rgb;
            color_min = min(color_min, n_color);
            color_max = max(color_max, n_color);
        }
        const int2 offsets_cross[4] = { int2(0, -2), int2(-2, 0), int2(2, 0), int2(0, 2) };
        float3 cross_min = center_color;
        float3 cross_max = center_color;[unroll]
        for (int j = 0; j < 4; j++)
        {
            float3 n_color = GetLod(color_tex, float4(texcoord+offsets_cross[j]*pixel_size,0,0)).rgb;
            cross_min = min(cross_min, n_color);
            cross_max = max(cross_max, n_color);
        }
        color_min = lerp(cross_min, color_min, 0.5);
        color_max = lerp(cross_max, color_max, 0.5);
    }
    float ComputeTrustFactor(float2 velocity_pixels, float low_threshold = 2.0, float high_threshold = 15.0)
    {
        float vel_mag = length(velocity_pixels);
        return saturate((high_threshold - vel_mag) / (high_threshold - low_threshold));
    }

//--------------------|
// :: Pixel Shaders ::|
//--------------------|

    void PS_GBuffer_NoSmooth(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode != 0)
            discard;

        float3 normal = CalculateNormal(uv);
        normal = ApplyBumpMapping(uv, normal);
        normal = GeometryCorrection(uv, normal);
        outNormal = float4(normal * 0.5 + 0.5, GetDepth(uv));
    }

    void PS_GBuffer_WithSmooth(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
            
        outNormal.rgb = CalculateNormal(uv.xy);
        outNormal.a = GetDepth(uv.xy);
    }

    void PS_SmoothNormals_H(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
            
        float4 color = GetLod(sNormTex_Pass1, uv);
        float4 s, s1;
        float sc;
        
        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = GetLod(sNormTex_Pass1, uv.xy + float2(x * p.x, 0));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T);
            s1 += s * diff;
            sc += diff;
        }
        outNormal = s1.rgba / sc;
    }

    void PS_SmoothNormals_V(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
            
        float4 color = GetLod(sNormTex_Pass2, uv);
        float4 s, s1;
        float sc;

        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = GetLod(sNormTex_Pass2, uv + float2(0, x * p.y));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T * 2);
            s1 += s * diff;
            sc += diff;
        }
        
        s1.rgba = s1.rgba / sc;
        float3 finalNormal = s1.rgb;
        finalNormal = ApplyBumpMapping(uv, finalNormal);
        finalNormal = GeometryCorrection(uv, finalNormal);
        
        outNormal = float4(finalNormal * 0.5 + 0.5, s1.a);
    }
    
    void PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outReflection : SV_Target)
    {
        float2 scaled_uv = uv / RenderScale;
        float depth = GetDepth(scaled_uv);
        if (depth >= 1.0)
        {
            outReflection = 0;
            return;
        }

        float3 viewPos = UVToViewPos(scaled_uv, depth);
        float3 viewDir = -normalize(viewPos);
        float3 normal = normalize(SampleNormal(scaled_uv));

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
#if __RENDERER__ == 0xd0900
        // MODIFIED: Use 3-tier quality for DX9
        if (isWall) 
            hit = TraceRay(r, STEPS_PER_RAY_WALLS_DX9); 
        else 
            hit = TraceRay(r, (Quality == 2) ? STEPS_PER_RAY_FLOOR_CEILING_PERF_DX9 : ((Quality == 1) ? STEPS_PER_RAY_FLOOR_CEILING_BALANCED_DX9 : STEPS_PER_RAY_FLOOR_CEILING_QUALITY_DX9));
#else
        if (isWall)
            hit = TraceRay(r, STEPS_PER_RAY_WALLS);
        else
            hit = TraceRay(r, (Quality == 2) ? 128 : ((Quality == 1) ? 192 : 256));
#endif
        float3 reflectionColor = 0;
        float reflectionAlpha = 0.0;
        if (hit.found)
        {
            reflectionColor = GetGlossySample(hit.uv, scaled_uv);
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
            reflectionColor = GetGlossySample(uvFb, scaled_uv);
            float vertical_fade = pow(saturate(1.0 - scaled_uv.y), 3.0);
            reflectionAlpha = vertical_fade;
        }
        reflectionAlpha *= pow(saturate(dot(-viewDir, r.direction)), 2.0);
        reflectionAlpha *= orientationIntensity;
        outReflection = float4(reflectionColor, reflectionAlpha);
    }
    
    void PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outBlended : SV_Target)
    {
        float4 current_reflection = GetLod(sReflection, float4(uv, 0, 0));
        if (!EnableTAA)
        {
            outBlended = current_reflection;
            return;
        }
        float2 full_res_uv = uv / RenderScale;
        float2 velocity = GetVelocityFromClosestFragment(full_res_uv);
        float current_depth = GetDepth(full_res_uv);
        float2 reprojected_uv_full = full_res_uv + velocity;
        float history_depth = GetDepth(reprojected_uv_full);
        float2 reprojected_uv_low = reprojected_uv_full * RenderScale;
        bool valid_history = all(saturate(reprojected_uv_low) == reprojected_uv_low) && FRAME_COUNT > 1 && abs(history_depth - current_depth) < 0.01;
        if (!valid_history)
        {
            outBlended = current_reflection;
            return;
        }
        float4 history_reflection = GetLod(sHistory, float4(reprojected_uv_low, 0, 0));
        float3 color_min, color_max;
        ComputeNeighborhoodMinMax(sReflection, uv, color_min, color_max);
        float3 clipped_history_rgb = ClipToAABB(color_min, color_max, history_reflection.rgb);
        float rejection_factor = saturate(length(history_reflection.rgb - clipped_history_rgb) * 900.0);
        float final_feedback = lerp(FeedbackFactor, 0.1, rejection_factor);
        float3 temporal_rgb = lerp(current_reflection.rgb, clipped_history_rgb, final_feedback);
        float temporal_a = lerp(current_reflection.a, history_reflection.a, final_feedback);
        float trust_factor = ComputeTrustFactor(velocity * BUFFER_SCREEN_SIZE);
        if (trust_factor < 1.0)
        {
            float3 blurred_color = current_reflection.rgb;
            const int blur_samples = 5;
            [unroll]
            for (int i = 1; i < blur_samples; i++)
            {
                float t = (float) i / (float) (blur_samples - 1);
                float2 blur_coord = uv - (velocity * RenderScale) * 0.5 * t;
                if (all(saturate(blur_coord) == blur_coord))
                    blurred_color += GetLod(sReflection, float4(blur_coord,0,0)).rgb;
            }
            blurred_color /= (float) blur_samples;
            temporal_rgb = lerp(blurred_color, temporal_rgb, trust_factor);
        }
        outBlended = float4(temporal_rgb, temporal_a);
    }

    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outHistory : SV_Target)
    {
        outHistory = GetLod(sTemp, float4(uv, 0, 0));
    }
    
    void PS_Upscale(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outUpscaled : SV_Target)
    {
        if (RenderScale >= 1.0)
        {
            outUpscaled = GetLod(sTemp, uv);
            return;
        }
        outUpscaled = GetLod(sTemp, uv * RenderScale);
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
                    outColor = float4(SampleNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 3:
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5:{    
                        float2 m = GetMotionVectors(uv);
                        float v_mag = length(m) * 100.0;
                        float a = atan2(m.y, m.x);
                        float3 hsv_color = HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, 1.0));
                        float3 grey_bg = float3(0.5, 0.5, 0.5);
                        float3 final_color = lerp(grey_bg, hsv_color, saturate(v_mag));
                        outColor = float4(final_color, 1.0);
                        return;
                    }
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
        float3 viewDir = -normalize(UVToViewPos(uv, GetDepth(uv)));
        float3 normal = SampleNormal(uv);
        float VdotN = saturate(dot(viewDir, normal));
        float3 f0 = lerp(float3(DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE), originalColor, Metallic);
        float3 F = F_Schlick(VdotN, f0);
        float3 finalColor = lerp(originalColor, reflectionColor * SPIntensity, F * reflectionMask);
        outColor = float4(finalColor, 1.0);
    }

    technique Barbatos_SSR
    {
        pass GBuffer_NoSmooth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GBuffer_NoSmooth;
            RenderTarget = TNormal;
        }
        pass GBuffer_WithSmooth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GBuffer_WithSmooth;
            RenderTarget = NormTex_Pass1;
        }
        pass SmoothNormals_H
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SmoothNormals_H;
            RenderTarget = NormTex_Pass2;
        }
        pass SmoothNormals_V
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SmoothNormals_V;
            RenderTarget = TNormal;
        }
        pass TraceReflections
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TraceReflections;
            RenderTarget = Reflection;
            ClearRenderTargets = true;
        }
        pass Accumulate
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Accumulate;
            RenderTarget = Temp;
        }
        pass UpdateHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = History;
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
