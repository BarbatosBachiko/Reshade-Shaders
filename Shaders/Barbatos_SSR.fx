/*----------------------------------------------|
| :: Barbatos SSR (Screen-Space Reflections) :: |
'-----------------------------------------------|
| Version: 0.3.0                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//--------------------|
// :: Preprocessor :: |
//--------------------|

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if __RESHADE__ < 40000
static const int STEPS_PER_RAY_WALLS_DX9 = 20;
static const int STEPS_PER_RAY_FLOOR_CEILING_QUALITY_DX9 = 64;
static const int STEPS_PER_RAY_FLOOR_CEILING_PERF_DX9 = 32;
#else
    static const int STEPS_PER_RAY_WALLS = 32;
#endif

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

#if UI_DIFFICULTY == 0 // Simple Mode
#define Metallic 0.2
#define Roughness 0.25
#define BumpIntensity 0.1
#define SobelEdgeThreshold 0.03
#define EnableGlossy true
#define GlossySamples 10
#define EnableTAA true
#define FeedbackFactor 0.99
#define OrientationThreshold 0.5
#define GeoCorrectionIntensity -0.01
#define VERTICAL_FOV 37.0
#define THICKNESS_THRESHOLD 0.1
// Denoiser settings
#define c_phi 0.3
#define n_phi 5.0
#define p_phi 1.0
// Smooth Normals settings
#define SmoothMode 0
#define Smooth_Threshold 0.5


uniform float SPIntensity <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
        ui_category = "Reflection Settings";
        ui_label = "Intensity";
    > = 1.1;

uniform float FadeEnd <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 5.0; ui_step = 0.010;
        ui_category = "Reflection Settings";
        ui_label = "Fade Distance";
    > = 4.999;

uniform int Quality <
        ui_type = "combo";
        ui_items = "Quality\0Performance\0";
        ui_category = "Performance & Quality";
        ui_label = "Quality Preset";
        ui_tooltip = "Choose between higher quality ray tracing or a faster preset.";
    > = 1;

uniform float RenderScale <
        ui_type = "drag";
        ui_min = 0.1; ui_max = 0.99; ui_step = 0.01;
        ui_category = "Performance & Quality";
        ui_label = "Render Scale";
        ui_tooltip = "Renders reflections at a lower resolution for performance.";
    > = 0.8;

uniform bool bEnableDenoise <
        ui_category = "Denoiser";
        ui_type = "checkbox";
        ui_label = "Enable A-Trous Denoiser";
    > = false;

uniform int ReflectionMode <
        ui_type = "combo";
        ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
        ui_category = "Reflection Settings";
        ui_label = "Reflection Mode";
    > = 4;

uniform int ViewMode <
        ui_type = "combo";
        ui_items = "None\0Motion Vectors\0Final Reflection\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0Reflection Mask\0";
        ui_category = "Debug";
        ui_label = "Debug View Mode";
    > = 0;

#elif UI_DIFFICULTY == 1 // Advanced Mode

    // -- Main Settings --
    uniform float SPIntensity <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
        ui_category = "Main Settings";
        ui_label = "Reflection Intensity";
    > = 1.1;

    uniform int ReflectionMode <
        ui_type = "combo";
        ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
        ui_category = "Main Settings";
        ui_label = "Reflection Mode";
        ui_tooltip = "Controls which surfaces will cast reflections.";
    > = 4;

    uniform float FadeEnd <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 5.0; ui_step = 0.010;
        ui_category = "Main Settings";
        ui_label = "Fade Distance";
        ui_tooltip = "Distance at which reflections begin to fade out.";
    > = 4.999;

    uniform float THICKNESS_THRESHOLD <
        ui_type = "drag";
        ui_min = 0.001; ui_max = 0.2; ui_step = 0.001;
        ui_category = "Main Settings";
        ui_label = "Thickness Threshold";
        ui_tooltip = "Determines how thick a surface is before a ray passes through it. Prevents self-reflection artifacts.";
    > = 0.1;

    // -- Surface & Material --
    uniform float Metallic <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Surface & Material";
        ui_label = "Metallic";
        ui_tooltip = "Controls how metallic a surface is. 0.0 for dielectrics (plastic, wood), 1.0 for metals.";
    > = 0.2;

    uniform float Roughness <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Surface & Material";
        ui_label = "Roughness";
        ui_tooltip = "Controls the surface roughness. 0.0 is a perfect mirror, 1.0 is very rough (diffuse).";
    > = 0.25;

    uniform float BumpIntensity <
        ui_label = "Bump Mapping Intensity";
        ui_type = "drag";
        ui_category = "Surface & Material";
        ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    > = 0.1;

    uniform float SobelEdgeThreshold <
        ui_label = "Sobel Edge Threshold";
        ui_tooltip = "Sets a minimum edge strength for bump mapping to occur. Helps reduce noise on flat surfaces.";
        ui_type = "drag";
        ui_category = "Surface & Material";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    > = 0.03;

    uniform int SmoothMode <
        ui_label = "Smooth Normals Mode";
        ui_type = "combo";
        ui_items = "Off\0Fast (13x13)\0Medium (16x16)\0High Quality (31x31)\0";
        ui_category = "Surface & Material";
    > = 0;

    uniform float Smooth_Threshold <
        ui_label = "Smooth Normals Threshold";
        ui_type = "drag";
        ui_min = 0.0; ui_max = 1.0;
        ui_category = "Surface & Material";
    > = 0.5;

    uniform int Quality <
        ui_type = "combo";
        ui_items = "Quality\0Performance\0";
        ui_category = "Performance & Quality";
        ui_label = "Quality Preset";
        ui_tooltip = "Choose between higher quality ray tracing or a faster preset.";
    > = 1;

    uniform float RenderScale <
        ui_type = "drag";
        ui_min = 0.1; ui_max = 0.99; ui_step = 0.01;
        ui_category = "Performance & Quality";
        ui_label = "Render Scale";
        ui_tooltip = "Renders reflections at a lower resolution for performance";
    > = 0.8;

    // -- Glossy Reflections --
    uniform bool EnableGlossy <
        ui_category = "Glossy Reflections";
        ui_label = "Enable Glossy Reflections";
        ui_tooltip = "Simulates reflections on non-perfectly smooth surfaces, creating a blurry/glossy effect.";
    > = true;

    uniform int GlossySamples <
        ui_type = "slider";
        ui_min = 1; ui_max = 20;
        ui_category = "Glossy Reflections";
        ui_label = "Glossy Samples";
        ui_tooltip = "Number of samples for the blur effect. Higher values are better quality but slower.";
    > = 10;
    
    // -- Denoiser --
    uniform bool bEnableDenoise <
        ui_category = "Denoiser";
        ui_type = "checkbox";
        ui_label = "Enable A-Trous Denoiser";
    > = false;

    uniform float c_phi <
        ui_category = "Denoiser";
        ui_type = "drag";
        ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
        ui_label = "Color Sigma";
        ui_tooltip = "Controls the influence of color similarity in the denoiser. Lower values consider only very similar colors.";
    > = 0.3;

    uniform float n_phi <
        ui_category = "Denoiser";
        ui_type = "drag";
        ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
        ui_label = "Normals Sigma";
        ui_tooltip = "Controls the influence of normal similarity. Lower values restrict filtering to surfaces with similar orientation.";
    > = 5.0;

    uniform float p_phi <
        ui_category = "Denoiser";
        ui_type = "drag";
        ui_min = 0.01; ui_max = 10.0; ui_step = 0.01;
        ui_label = "Position (Depth) Sigma";
        ui_tooltip = "Controls the influence of world-space position similarity. Lower values restrict filtering to nearby pixels.";
    > = 1.0;

    // -- Temporal Filtering --
    uniform bool EnableTAA <
        ui_category = "Temporal Filtering";
        ui_label = "Enable Temporal Reprojection";
        ui_tooltip = "Blends the current frame's reflection with previous frames to reduce noise and flickering using temporal reprojection.";
    > = true;

    uniform float FeedbackFactor <
        ui_type = "drag";
        ui_min = 0.0; ui_max = 0.99; ui_step = 0.01;
        ui_category = "Temporal Filtering";
        ui_label = "Temporal Feedback";
        ui_tooltip = "Controls how much of the previous frame is blended in. Higher values are smoother but can cause more ghosting.";
    > = 0.99;

    // -- Advanced --
    uniform float OrientationThreshold <
        ui_type = "drag";
        ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Advanced";
        ui_label = "Orientation Threshold";
        ui_tooltip = "Controls sensitivity for detecting floors/walls/ceilings based on their normal vector. Lower is stricter.";
    > = 0.50;

    uniform float GeoCorrectionIntensity <
        ui_type = "drag";
        ui_min = -0.1; ui_max = 0.01;
        ui_step = 0.01;
        ui_category = "Advanced";
        ui_label = "Geometry Correction Intensity";
        ui_tooltip = "Subtly adjusts surface normals based on color data to correct minor geometry inaccuracies.";
    > = -0.01;

    uniform float VERTICAL_FOV <
        ui_type = "drag";
        ui_min = 15.0; ui_max = 120.0;
        ui_step = 0.1;
        ui_category = "Advanced";
        ui_label = "Vertical FOV";
    > = 37.0;

    // -- Debug --
    uniform int ViewMode <
        ui_type = "combo";
        ui_items = "None\0Motion Vectors\0Final Reflection\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0Reflection Mask\0";
        ui_category = "Debug";
        ui_label = "Debug View Mode";
    > = 0;

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

namespace Barbatos_SSR201
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

    texture DenoiseTex0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sDenoiseTex0
    {
        Texture = DenoiseTex0;
    };

    texture DenoiseTex1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sDenoiseTex1
    {
        Texture = DenoiseTex1;
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

//---------------|
// :: Structs :: |
//---------------|

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

//-------------|
// :: Utility::|
//-------------|

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
        else // Quality
        {
            refinement_steps = 5;
            step_scale = 0.1;
            min_step_size = 0.0001;
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

//--------------------------|
// :: A-Trous Denoising  :: |
//--------------------------|
    float4 PS_DenoisePass(float4 vpos : SV_Position, float2 uv : TEXCOORD, int level, sampler input_sampler)
    {
        if (any(uv > RenderScale))
            return GetLod(input_sampler, uv);

        float4 center_color = GetLod(input_sampler, uv);
        float2 full_res_uv = uv / RenderScale;
        float center_depth = GetDepth(full_res_uv);
        
        if (center_depth >= 0.999)
            return center_color;

        float3 center_normal = SampleNormal(full_res_uv);
        float center_normal_len_sq = dot(center_normal, center_normal);
        float3 center_pos = UVToViewPos(full_res_uv, center_depth);

        float4 sum = 0.0;
        float cum_w = 0.0;
        
        const float2 step_size = ReShade::PixelSize * exp2(level);
        static const float2 atrous_offsets[9] = { float2(-1, -1), float2(0, -1), float2(1, -1), float2(-1, 0), float2(0, 0), float2(1, 0), float2(-1, 1), float2(0, 1), float2(1, 1) };

        [loop]
        for (int i = 0; i < 9; i++)
        {
            const float2 neighbor_uv_low = uv + atrous_offsets[i] * step_size;
            if (any(neighbor_uv_low < 0.0) || any(neighbor_uv_low > RenderScale))
                continue;

            const float4 sample_color = GetLod(input_sampler, neighbor_uv_low);
            const float2 neighbor_uv_full = neighbor_uv_low / RenderScale;
            const float sample_depth = GetDepth(neighbor_uv_full);

            if (sample_depth >= 0.999)
                continue;

            const float3 sample_normal = SampleNormal(neighbor_uv_full);
            const float3 sample_pos = UVToViewPos(neighbor_uv_full, sample_depth);
            
            float diff_c = distance(center_color.rgb, sample_color.rgb);
            float w_c = exp(-(diff_c * diff_c) / c_phi);
            const float sample_normal_len_sq = dot(sample_normal, sample_normal);
            float diff_n = dot(center_normal, sample_normal) * rsqrt(max(1e-6, center_normal_len_sq * sample_normal_len_sq));
            float w_n = exp(n_phi * (saturate(diff_n) - 1.0));
            float diff_p = distance(center_pos, sample_pos);
            float w_p = exp(-(diff_p * diff_p) / p_phi);

            const float weight = w_c * w_n * w_p;
            sum += sample_color * weight;
            cum_w += weight;
        }
        return cum_w > 1e-6 ? (sum / cum_w) : center_color;
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
        if (any(uv > RenderScale))
        {
            outReflection = 0;
            return;
        }

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
        if (isWall) hit = TraceRay(r, STEPS_PER_RAY_WALLS_DX9); else hit = TraceRay(r, (Quality == 1) ? STEPS_PER_RAY_FLOOR_CEILING_PERF_DX9 : STEPS_PER_RAY_FLOOR_CEILING_QUALITY_DX9);
#else
        if (isWall)
            hit = TraceRay(r, STEPS_PER_RAY_WALLS);
        else
            hit = TraceRay(r, (Quality == 1) ? 128 : 256);
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
        if (any(uv > RenderScale))
        {
            outBlended = 0;
            return;
        }
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
        float rejection_factor = saturate(length(history_reflection.rgb - clipped_history_rgb) * 9000.0);
        float final_feedback = lerp(FeedbackFactor, 0.1, rejection_factor);
        float3 temporal_rgb = lerp(current_reflection.rgb, clipped_history_rgb, final_feedback);
        float temporal_a = lerp(current_reflection.a, history_reflection.a, final_feedback);
        float trust_factor = ComputeTrustFactor(velocity * BUFFER_SCREEN_SIZE);
        if (trust_factor < 1.0)
        {
            float3 blurred_color = current_reflection.rgb;
            const int blur_samples = 5;[unroll]
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
    float4 PS_DenoisePass0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return GetLod(sTemp, texcoord);
        return PS_DenoisePass(vpos, texcoord, 0, sTemp);
    }
    float4 PS_DenoisePass1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return GetLod(sDenoiseTex0, texcoord);
        return PS_DenoisePass(vpos, texcoord, 1, sDenoiseTex0);
    }
    float4 PS_DenoisePass2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return GetLod(sDenoiseTex1, texcoord);
        return PS_DenoisePass(vpos, texcoord, 2, sDenoiseTex1);
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
                case 1:{
                        float2 m = GetMotionVectors(uv);
                        float v = length(m) * 100.0;
                        float a = atan2(m.y, m.x);
                        outColor = float4(HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, saturate(v))), 1.0);
                        return;
                    }
                case 2:
                    outColor = float4(GetLod(sUpscaled, uv).rgb, 1.0);
                    return;
                case 3:
                    outColor = float4(SampleNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 4:
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5:
                    outColor = float4(GetLod(sReflection, uv).rgb, 1.0);
                    return;
                case 6:
                    outColor = float4(GetLod(sTemp, uv).rgb, 1.0);
                    return;
                case 7:
                    outColor = GetLod(sUpscaled, uv).aaaa;
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
        float3 viewDir = -normalize(UVToViewPos(uv, GetDepth(uv)));
        float3 normal = SampleNormal(uv);
        float VdotN = saturate(dot(viewDir, normal));
        float3 f0 = lerp(float3(DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE, DIELECTRIC_REFLECTANCE), originalColor, Metallic);
        float3 F = F_Schlick(VdotN, f0);
        float3 finalColor = lerp(originalColor, reflectionColor * SPIntensity, F * reflectionMask);
        outColor = float4(finalColor, 1.0);
    }

//-----------------|
// :: Technique :: |
//-----------------|

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
        pass DenoisePass0
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass0;
            RenderTarget = DenoiseTex0;
        }
        pass DenoisePass1
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass1;
            RenderTarget = DenoiseTex1;
        }
        pass DenoisePass2
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass2;
            RenderTarget = Temp;
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
