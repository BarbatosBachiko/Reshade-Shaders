/*------------------.
| :: Description :: |
'-------------------/

    SSRT

    Version 1.3.3
    Author: Barbatos Bachiko
    Original SSRT by jebbyk : https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx

    License: GNU Affero General Public License v3.0 : https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE
    Aces Tonemapping use Pentalimbed Unlicense: https://github.com/Pentalimbed/YASSGI/blob/main/UNLICENSE

    About: Screen Space Ray Tracing

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 1.3.3
    + Schlick Aproximation to Trace
    * Add Aces Tone Mapping
    * Add Linear RGB
    - Removed temporal mode "Quality". was causing strange artifacts
*/

/*-------------------.
| :: Definitions ::  |
'-------------------*/

#include "ReShade.fxh"

// Motion vector configuration+
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Resolution scaling
#ifndef RES_SCALE
#define RES_SCALE 0.8
#endif
#define RES_WIDTH (ReShade::ScreenSize.x * RES_SCALE)
#define RES_HEIGHT (ReShade::ScreenSize.y * RES_SCALE)

// Utility macros
#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords) * DEPTH_MUL)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define DEPTH_MUL   ((Preset == 1) ? 0.1 : DepthMultiplier)
#define PERSPECTIVE ((Preset == 1) ? 1.0 : PERSPECTIVE_COEFFICIENT)

/*-------------------.
| :: Parameters ::   |
'-------------------*/

uniform int Guide <
    ui_text = 
    
        "SSRT SCREEN SPACE RAY TRACING GUIDE:\n\n"
        
        "CORE SETTINGS:\n"
        "1. Base Ray Length: Controls how far rays travel (0.1-50)\n"
        "2. Rays Amount: Number of rays per pixel (1-4). Note: this option is useless for now.\n"
        "3. Steps Per Ray: Detail of each ray (1-256). Higher = more accurate but slower\n"
        
        "QUALITY TUNING:\n"
        "1. Depth Threshold: Surface detection precision (0.001-0.01)\n"
        "2. Normal Threshold: Angle sensitivity (-1.0-1.0)\n"
        "3. Perspective: Adjust for correct depth scaling (0-10)\n\n"
        
        "OPTIMIZATION TIPS:\n"
        "1. Start with Rays Amount=1 and Steps=60 for performance\n"
        "2. Use Temporal Filtering to smooth results\n"
        "3. Lower RES_SCALE in definitions for faster rendering\n\n"
        
        "VISUAL TUNING:\n"
        "1. Use View Modes to debug GI/Normals/Depth\n"
        "2. Adjust FadeStart/FadeEnd to control effect distance\n"
        "3. Set SkyColor to match your game's environment. Note: if necessary\n"
        "4. Enable Bump Mapping for surface detail (adjust BumpIntensity)\n\n"
        
        "PRESET ADVICE:\n"
        "1. Game B: Alternative depth scaling and Pespective (0.1 multiplier), (1.0 Pespective)\n\n"
        
        "BLEND MODES:\n"
        "1. Additive: Brightens surfaces (good for dark games)\n"
        "2. Multiplicative: Natural shadowing (default)\n"
        "3. Alpha Blend: For subtle effects";
        
    ui_category = "Guide";
    ui_category_closed = true;
    ui_label = " ";
    ui_type = "radio";
> = 0;

// General Settings
uniform int ViewMode <
    ui_type = "combo";
    ui_category = "General";
    ui_label = "View Mode";
    ui_items = "None\0GI Debug\0Normal Debug\0Depth Debug\0";
> = 0;

uniform int Preset <
    ui_type     = "combo";
    ui_category = "General";
    ui_label    = "Preset for View";
    ui_tooltip = "GAME B preset may be better in some games";
    ui_items    = "Custom\0GAME B\0";
> = 0;

uniform int BlendMode <
    ui_type = "combo";
    ui_category = "General";
    ui_label = "Blend Mode";
    ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
> = 1;

// Ray Tracing Settings
uniform float BASE_RAYS_LENGTH <
    ui_type = "slider";
    ui_min = 0.1; ui_max = 2.0;
    ui_step = 0.001;
    ui_category = "Ray Tracing";
    ui_label = "Base ray length";
> = 0.1;

uniform int RAYS_AMOUNT <
    ui_type = "slider";
    ui_min = 1; ui_max = 4;
    ui_step = 1;
    ui_category = "Ray Tracing";
    ui_label = "Rays amount";
> = 1;

uniform int STEPS_PER_RAY <
    ui_type = "slider";
    ui_min = 1; ui_max = 256;
    ui_step = 1;
    ui_category = "Ray Tracing";
    ui_label = "Steps per ray";
> = 64.0;

uniform float Intensity <
    ui_type = "slider";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
    ui_category = "Ray Tracing";
    ui_label = "Intensity";
> = 2.5;

uniform float RandomIntensity <
    ui_label = "Random Intensity";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
> = 0.0;

uniform float DEPTH_THRESHOLD <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Ray Tracing";
    ui_label = "Depth Threshold";
> = 0.002;

uniform float NORMAL_THRESHOLD <
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_category = "Ray Tracing";
    ui_label = "Normal Threshold";
> = -1.0;

uniform float PERSPECTIVE_COEFFICIENT <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
    ui_category = "Ray Tracing";
    ui_label = "Pespective";
> = 1.0;

uniform float FadeStart
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade Start";
        ui_tooltip = "Distance starts to fade out";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.0;

uniform float FadeEnd
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade End";
        ui_tooltip = "Distance completely fades out";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 1.0;

// Depth Settings
uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

// Normal Settings
uniform bool bSmoothNormals <
    ui_category = "Normal";
    ui_label = "Smooth Normals";
> = true;

// Bump Mapping Settings
uniform bool EnableBumpMapping <
    ui_category = "Bump Mapping";
    ui_label = "Enable Bump Mapping";
> = true;

uniform float BumpIntensity <
    ui_type = "slider";
    ui_category = "Bump Mapping";
    ui_label = "Bump Intensity";
    ui_min = 0.0; ui_max = 20.0; ui_step = 0.01;
> = 0.5;

uniform float3 BumpDirection < 
    ui_type = "slider";
    ui_category = "Bump Mapping";
    ui_label = "Bump Direction XYZ";
    ui_min = -2.0; ui_max = 2.0;
> = float3(-2.0, 1.0, -0.5); 

uniform float BumpDepth <
    ui_type = "slider";
    ui_category = "Bump Mapping";
    ui_label = "Bump Depth";
    ui_min = 0.0; ui_max = 3.0;
> = 0.7;

// Temporal Settings
uniform bool EnableTemporal <
    ui_type = "checkbox";
    ui_category = "Temporal";
    ui_label = "Temporal Filtering";
> = true;

uniform int TemporalMode <
    ui_category = "Temporal";
    ui_type = "combo";
    ui_items = "Blurry\0Standard\0";
    ui_label = "Temporal Mode";
> = 1;

uniform float TemporalFilterStrength <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "Temporal Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.3;

// Extra Settings
uniform bool AssumeSRGB < 
    ui_category = "Tone Mapping";
    ui_label = "Assume sRGB Input";
> = true;

uniform bool EnableACES <
    ui_category = "Tone Mapping";
    ui_label = "Enable ACES Tone Mapping";
> = true;

uniform float3 SkyColor < 
    ui_type = "color";
    ui_category = "Extra";
    ui_label = "Sky Color";
> = float3(0.0, 0.0, 0.0);

uniform float Saturation <
    ui_type = "slider";
    ui_category = "Extra";
    ui_label = "Saturation";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 1.0;

uniform int FRAME_COUNT < source = "framecount"; >;

/*---------------.
| :: Textures :: |
'---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC };
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
#endif

namespace SSRT
{
    texture GiTex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    sampler sGiTex
    {
        Texture = GiTex;
    };

    texture GiTemporal
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    sampler sGITemporal
    {
        Texture = GiTemporal;
    };

    texture2D giHistory
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sGIHistory
    {
        Texture = giHistory;
        SRGBTexture = false;
    };

    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler normalSampler
    {
        Texture = normalTex;S_PC
    };

    texture depthTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
        MipLevels = 6;
    };
    sampler depthSampler
    {
        Texture = depthTex;
        MinLOD = 0.0f;
        MaxLOD = 5.0f;
    };
    
/*----------------.
| :: Functions :: |
'----------------*/
    
    // Schlick
    static const float ior = 1.5;
    static const float R0 = (1.0 - 1.0 / ior) * (1.0 - 1.0 / ior);
    
    //Unlicence Start
    static const float3x3 g_sRGBToACEScg = float3x3(
    0.613117812906440, 0.341181995855625, 0.045787344282337,
    0.069934082307513, 0.918103037508582, 0.011932775530201,
    0.020462992637737, 0.106768663382511, 0.872715910619442
);

    static const float3x3 g_ACEScgToSRGB = float3x3(
    1.704887331049502, -0.624157274479025, -0.080886773895704,
    -0.129520935348888, 1.138399326040076, -0.008779241755018,
    -0.024127059936902, -0.124620612286390, 1.148822109913262
);
    //End
    
    // MIT License functions
    float3 getWorldPositionForNormal(float2 coords)
    {
        float depth = getDepth(coords).x;
        return float3((coords - 0.5) * depth, depth);
    }

    float4 mulByA(float4 v)
    {
        return float4(v.rgb * v.a, v.a);
    }

    float4 computeNormal(float3 wpCenter, float3 wpNorth, float3 wpEast)
    {
        return float4(normalize(cross(wpCenter - wpNorth, wpCenter - wpEast)), 1.0);
    }

    float GetLuminance(float3 color)
    {
        return dot(color, float3(0.299, 0.587, 0.114));
    }

    float4 computeNormal(float2 coords, float3 offset, bool reverse)
    {
        float3 posCenter = getWorldPositionForNormal(coords);
        float3 posNorth = getWorldPositionForNormal(coords - (reverse ? -1 : 1) * offset.zy);
        float3 posEast = getWorldPositionForNormal(coords + (reverse ? -1 : 1) * offset.xz);
    
        float4 r = computeNormal(posCenter, posNorth, posEast);
        float mD = max(abs(posCenter.z - posNorth.z), abs(posCenter.z - posEast.z));
        if (mD > 16)
            r.a = 0;
        return r;
    }

    float3 GetNormal(float2 coords)
    {
        float3 offset = float3(ReShade::PixelSize, 0.0);
        float4 normal = computeNormal(coords, offset, false);
    
        if (normal.a == 0)
        {
            normal = computeNormal(coords, offset, true);
        }
    
        if (bSmoothNormals)
        {
            float3 offset2 = offset * 7.5 * (1.0 - getDepth(coords).x);
            float4 normalTop = computeNormal(coords - offset2.zy, offset, false);
            float4 normalBottom = computeNormal(coords + offset2.zy, offset, false);
            float4 normalLeft = computeNormal(coords - offset2.xz, offset, false);
            float4 normalRight = computeNormal(coords + offset2.xz, offset, false);
        
            normalTop.a *= smoothstep(1, 0, distance(normal.xyz, normalTop.xyz) * 1.5) * 2;
            normalBottom.a *= smoothstep(1, 0, distance(normal.xyz, normalBottom.xyz) * 1.5) * 2;
            normalLeft.a *= smoothstep(1, 0, distance(normal.xyz, normalLeft.xyz) * 1.5) * 2;
            normalRight.a *= smoothstep(1, 0, distance(normal.xyz, normalRight.xyz) * 1.5) * 2;
        
            float4 normal2 = mulByA(normal) + mulByA(normalTop) + mulByA(normalBottom) +
                         mulByA(normalLeft) + mulByA(normalRight);
        
            if (normal2.a > 0)
            {
                normal2.xyz /= normal2.a;
                normal.xyz = normalize(normal2.xyz);
            }
        }
    
        if (EnableBumpMapping)
        {
            float3 color = GetColor(coords).rgb;
            float height = GetLuminance(color);
            float heightRight = GetLuminance(GetColor(coords + offset.xz).rgb);
            float heightUp = GetLuminance(GetColor(coords - offset.zy).rgb);
        
            float2 slope = float2(
            (height - heightRight) * BumpIntensity * BumpDirection.x,
            (height - heightUp) * BumpIntensity * BumpDirection.y
        );
            float holeDepth = (1.0 - height) * BumpDepth * BumpDirection.z;
        
            float3 N = normal;
            float3 T = normalize(cross(N, float3(0, 1, 0)));
            float3 B = cross(N, T);
        
            float3 bumpedNormal = N + (T * slope.x + B * slope.y + N * holeDepth);
            bumpedNormal = normalize(bumpedNormal);
            normal = bumpedNormal;
        }
    
        return normal.xyz;
    }

    float3 getNormal(float2 coords)
    {
        float3 normal = -(tex2Dlod(normalSampler, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
    }

    float3 LinearizeSRGB(float3 color)
    {
        return pow(color, 2.2);
    }

    float3 sRGB_to_ACEScg(float3 srgb)
    {
        return mul(g_sRGBToACEScg, srgb);
    }

    float3 ACEScg_to_sRGB(float3 acescg)
    {
        return mul(g_ACEScgToSRGB, acescg);
    }

    // ACES tone mapping approximation (RRT + ODT)
    float3 ApplyACES(float3 color)
    {
        if (!EnableACES)
            return color;
    
        float3 acescg = sRGB_to_ACEScg(color);

        const float A = 2.51;
        const float B = 0.03;
        const float C = 2.43;
        const float D = 0.59;
        const float E = 0.14;

        float3 toneMapped = (acescg * (A * acescg + B)) / (acescg * (C * acescg + D) + E);

        return ACEScg_to_sRGB(toneMapped);
    }
    
    // GNU 3 License functions
    float3 rand3d(float2 uv)
    {
        uv = frac(uv * float2(123.34, 456.21));
        float x = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
        float y = frac(sin(dot(uv, float2(93.9898, 67.345))) * 24634.6345);
        float z = frac(sin(dot(uv, float2(45.332, 12.345))) * 12445.1234);
        return float3(x, y, z);
    }

    float3 uvz_to_xyz(float2 uv, float z)
    {
        uv -= float2(0.5, 0.5);
        float perspective = (Preset == 1) ? 1.0 : PERSPECTIVE_COEFFICIENT;
        return float3(uv.x * z * perspective, uv.y * z * perspective, z);
    }

    float2 xyz_to_uv(float3 pos)
    {
        float perspective = (Preset == 1) ? 1.0 : PERSPECTIVE_COEFFICIENT;
        float2 uv = float2(pos.x / (pos.z * perspective), pos.y / (pos.z * perspective));
        return uv + float2(0.5, 0.5);
    }

    // Main ray tracing function
    float4 Trace(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
    {
        float depth = getDepth(texcoord).x;
        float3 normal = getNormal(texcoord);
        float3 selfPos = uvz_to_xyz(texcoord, depth);
        float3 viewDir = normalize(selfPos);

        float4 accumulatedColor = float4(0, 0, 0, 0);
        float invRays = 1.0 / RAYS_AMOUNT;

        // For each ray
        for (int j = 0; j < RAYS_AMOUNT; ++j)
        {
            float3 jitter = normalize(rand3d(texcoord * (32.0 * (j + 1.0)) + frac(FRAME_COUNT / 4.0)) * 2.0 - 1.0);
            float3 pertN = normalize(lerp(normal, normal + jitter * 0.2, RandomIntensity));
            float3 rayDir = normalize(reflect(viewDir, pertN));
            float3 rayStep = rayDir * (BASE_RAYS_LENGTH / STEPS_PER_RAY);
            float3 currPos = selfPos;

            bool hit = false;

             // March along the ray
            for (int i = 0; i < STEPS_PER_RAY; ++i)
            {
                currPos += rayStep;

                if (currPos.z <= 0.0)
                    continue;

                float2 uvNew = xyz_to_uv(currPos);
                if (any(uvNew < 0.0) || any(uvNew > 1.0))
                    continue;

                float sceneDepth = getDepth(uvNew).x;

                if (currPos.z <= sceneDepth || currPos.z >= sceneDepth + DEPTH_THRESHOLD)
                    continue;

                float3 hitNormal = getNormal(uvNew);
                float normalDot = dot(hitNormal, pertN);
                if (normalDot < NORMAL_THRESHOLD)
                    continue;

                // Schlick APP
                float cosTheta = saturate(dot(hitNormal, rayDir));
                float oneMinusC = 1.0 - cosTheta;
                float fresnel = R0 + (1.0 - R0) * pow(oneMinusC, 5.0);
                float factor = max(0.2, fresnel);

                float angleWeight = pow(saturate(dot(viewDir, rayDir)), 2.0);

                float3 sampleColor = tex2Dlod(ReShade::BackBuffer, float4(uvNew, 0, 0)).rgb;
                accumulatedColor.rgb += sampleColor * invRays * factor * angleWeight;

                hit = true;
                break;
            }

            if (!hit)
            {
                accumulatedColor.rgb += SkyColor * 0.1 * invRays;
            }
        }
        
        //Fade, Alpha, Remap
        float fadeF = max(FadeEnd - FadeStart, 0.001);
        float fade = saturate((FadeEnd - depth) / fadeF);
        accumulatedColor.rgb *= fade;
        accumulatedColor.a = depth;
        accumulatedColor.rgb = accumulatedColor.rgb / (accumulatedColor.rgb + 1.0);

        return accumulatedColor;
    }
    //END GNU3
    
    // Motion vector function
    float2 GetMotionVector(float2 texcoord)
    {
#if USE_MARTY_LAUNCHPAD_MOTION
        return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
        return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#else
        return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).xy;
#endif
    }

    // Shader passes
    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float3 RGBToYCoCg(float3 rgb)
    {
        float Y = dot(rgb, float3(0.25, 0.5, 0.25));
        float Co = dot(rgb, float3(0.5, 0.0, -0.5));
        float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
        return float3(Y, Co, Cg);
    }

    float3 YCoCgToRGB(float3 ycocg)
    {
        float Y = ycocg.x;
        float Co = ycocg.y;
        float Cg = ycocg.z;
        return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
    }
    
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 motion = GetMotionVector(uv);
        float3 currentGI = tex2Dlod(sGiTex, float4(uv, 0, 0)).rgb;
        float2 reprojectedUV = uv + motion;
        float2 offset = ReShade::PixelSize.xy;

        float3 history = tex2Dlod(sGIHistory, float4(reprojectedUV, 0, 0)).rgb;

        if (TemporalMode == 0) // Method A - Blurry (Basic neighborhood averaging)
        {
            float3 historyUp = tex2Dlod(sGIHistory, float4(reprojectedUV + float2(0, -offset.y), 0, 0)).rgb;
            float3 historyDown = tex2Dlod(sGIHistory, float4(reprojectedUV + float2(0, offset.y), 0, 0)).rgb;
            float3 historyLeft = tex2Dlod(sGIHistory, float4(reprojectedUV + float2(-offset.x, 0), 0, 0)).rgb;
            float3 historyRight = tex2Dlod(sGIHistory, float4(reprojectedUV + float2(offset.x, 0), 0, 0)).rgb;

            float3 historyAvg = (history + historyUp + historyDown + historyLeft + historyRight) / 5.0;
            return float4(lerp(currentGI, historyAvg, TemporalFilterStrength), 1.0);
        }
        else // Method B - Standard (YCoCg blending)
        {
            float3 currentYCoCg = RGBToYCoCg(currentGI);
            float3 historyYCoCg = RGBToYCoCg(history);
            float3 blendedYCoCg = lerp(currentYCoCg, historyYCoCg, TemporalFilterStrength);
            return float4(YCoCgToRGB(blendedYCoCg), 1.0);
        }
    }

    float4 PS_SaveHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 gi = EnableTemporal ? tex2Dlod(sGITemporal, float4(uv, 0, 0)).rgb : tex2Dlod(sGiTex, float4(uv, 0, 0)).rgb;
        return float4(gi, 1.0);
    }

    float4 Combine(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, texcoord);
        float3 giColor = EnableTemporal ? tex2D(sGITemporal, texcoord).rgb : tex2D(sGiTex, texcoord).rgb;

        float depth = getDepth(texcoord);
        float3 normal = getNormal(texcoord);

        giColor *= Intensity;

        if (AssumeSRGB)
        {
            giColor = LinearizeSRGB(giColor);
        }
        if (EnableACES)
        {
            giColor = ApplyACES(giColor);
        }

        // Saturation
        float greyValue = dot(giColor, float3(0.299, 0.587, 0.114)); 
        float3 grey = float3(greyValue, greyValue, greyValue); 
        giColor = lerp(grey, giColor, Saturation);

        if (ViewMode == 0)
        {
            if (BlendMode == 0) // Additive
                return float4(originalColor.rgb + giColor, originalColor.a);
            else if (BlendMode == 1) // Multiplicative
                return float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - giColor), originalColor.a);
            else if (BlendMode == 2) // Alpha Blend
                return float4(lerp(originalColor.rgb, giColor, saturate(giColor.r)), originalColor.a);
        }
        else if (ViewMode == 1) // GI Debug
            return float4(giColor, 1.0);
        else if (ViewMode == 2) // Normal Debug
            return float4(normal * 0.5 + 0.5, 1.0);
        else if (ViewMode == 3) // Depth Debug
            return float4(depth, depth, depth, 1.0);

        return originalColor;
    }

/*------------------.
| :: Techniques :: |
'------------------*/

    technique SSRT
    {
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
            RenderTarget1 = depthTex;
        }
    
        pass Trace
        {
            VertexShader = PostProcessVS;
            PixelShader = Trace;
            RenderTarget = GiTex;
        }
    
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget = GiTemporal;
        }
    
        pass SaveHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistory;
            RenderTarget = giHistory;
        }
    
        pass Combine
        {
            VertexShader = PostProcessVS;
            PixelShader = Combine;
        }
    }
}
