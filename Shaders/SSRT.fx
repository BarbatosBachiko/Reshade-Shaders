 /*------------------.
| :: Description :: |
'-------------------/

    SSRT

    Version 1.5.3
    Author: Barbatos Bachiko
    Original SSRT by jebbyk : https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx

    License: GNU Affero General Public License v3.0 : https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE
    Aces Tonemapping use Pentalimbed Unlicense: https://github.com/Pentalimbed/YASSGI/blob/main/UNLICENSE

    About: Screen Space Ray Tracing

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.5.3
    + Performance
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
#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

/*-------------------.
| :: Parameters ::   |
'-------------------*/

uniform float Intensity <
    ui_type = "slider";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.001;
    ui_category = "General";
    ui_label = "Specular Intensity";
> = 2.0;

uniform float IndirectIntensity <
    ui_type = "slider";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.001;
    ui_category = "General";
    ui_label = "GI Intensity";
> = 2.0;

uniform bool EnableSpecularGI <
    ui_type = "checkbox";
    ui_category = "General";
    ui_label = "Enable Specular GI";
> = true;

uniform bool EnableDiffuseGI <
    ui_type = "checkbox";
    ui_category = "General";
    ui_label = "Enable Diffuse GI (BETA)";
> = false;

// Bump Mapping Settings
uniform float BumpIntensity <
    ui_type = "slider";
    ui_category = "Bump Mapping";
    ui_label = "Bump Intensity";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;

uniform float ThicknessThreshold <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Thickness Threshold";
> = 0.010;

// Temporal Settings
uniform float AccumFramesDF <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "GI Temporal";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 12.0;

uniform float AccumFramesSG <
    ui_type = "slider";
    ui_category = "Temporal";
     ui_label = "SSR Temporal";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
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
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    >
    = 1.0;

// Depth Settings
uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normal";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

// Normal Settings
uniform bool bSmoothNormals <
 ui_category = "Depth/Normal";
    ui_label = "Smooth Normals";
> = true;

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_items = "None\0Motion\0GI\0Diffuse Light\0Specular\0Normal\0Depth\0";
> = 0;

// Extra Settings
uniform bool AssumeSRGB < 
    ui_category = "Tone Mapping";
    ui_label = "Assume sRGB Input";
> = true;

uniform bool EnableACES <
    ui_category = "Tone Mapping";
    ui_label = "Enable ACES Tone Mapping";
> = false;

uniform float Saturation <
    ui_type = "slider";
    ui_category = "Extra";
    ui_label = "Saturation";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 1.0;

uniform int BlendMode <
    ui_type = "combo";
    ui_category = "Extra";
    ui_label = "Blend Mode";
    ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
> = 1;

uniform int FRAME_COUNT < source = "framecount"; >;
uniform int random < source = "random";min = 0; max = 512; >;

//Random
static const float RandomIntensity = 0.0;

//Bump Mapping
static const float3 BumpDirection = float3(-2.0, 1.0, -0.5);
static const float BumpDepth = 0.7;
static const float PERSPECTIVE_COEFFITIENT = 1.0;

//Ray Marching
static const float MaxTraceDistance = 1.0;
static const float BASE_RAYS_LENGTH = 1.0;
static const float RAYS_AMOUNT = 1.0;
static const int STEPS_PER_RAY = 96;
static const float EnableTemporal = true;

// Adaptive step 
static const float MIN_STEP_SIZE = 0.001;
static const float STEP_GROWTH = 1.2;
static const int REFINEMENT_STEPS = 6;

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

namespace BSSRT
{
    texture DiffuseGI
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16f;
    };
    sampler sDFGI
    {
        Texture = DiffuseGI;
    };
    
    texture DiffuseTemp
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16f;
    };
    sampler sDiffuseTemp
    {
        Texture = DiffuseTemp;
    };

    texture2D DiffuseHistory
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16f;
    };
    sampler2D sDFGIHistory
    {
        Texture = DiffuseHistory;
        SRGBTexture = false;
    };

    texture Specular
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSpecular
    {
        Texture = Specular;
    };
    
    texture SpecularTemp
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSpecularTemp
    {
        Texture = SpecularTemp;
    };
    
    texture SpecularHistory
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSpecularHistory
    {
        Texture = SpecularHistory;
    };
    
    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
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
    sampler sDepth
    {
        Texture = depthTex;
        MinLOD = 0.0f;
        MaxLOD = 5.0f;
    };
    
/*----------------.
| :: Functions :: |
'----------------*/
    
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

    float lum(float3 color)
    {
        return (color.r + color.g + color.b) * 0.3333333;
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
    
        {
            float3 color = GetColor(coords).rgb;
            float height = lum(color);
            float heightRight = lum(GetColor(coords + offset.xz).rgb);
            float heightUp = lum(GetColor(coords - offset.zy).rgb);
        
            float2 slope = float2(
            (height - heightRight) * BumpIntensity * BumpDirection.x,
            (height - heightUp) * BumpIntensity * BumpDirection.y
        );
            float holeDepth = (1.0 - height) * BumpDepth * BumpDirection.z;
        
            float3 N = normal.xyz;
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
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
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

    float3 rand3d(float2 uv)
    {
        uv += random;
        float3 r;
        r.x = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453) * 2.0 - 1.0;
        r.y = frac(sin(dot(uv, float2(93.9898, 67.345))) * 12741.3427) * 2.0 - 1.0;
        r.z = frac(sin(dot(uv, float2(29.533, 94.729))) * 31415.9265) * 2.0 - 1.0;
        return r;
    }

    float2 xyz_to_uv(float3 pos)
    {
        return pos.xy / pos.z + 0.5;
    }

    float3 uvz_to_xyz(float2 uv, float z)
    {
        return float3((uv - 0.5) * z, z);
    }
    
   // GNU 3 License functions 
    struct PixelData
    {
        float PerspectiveFactor;
        float InvSteps;
        float InvRays;
        float Depth;
        float3 SelfPos;
        float3 ViewDir;
        float3 Normal;
    };

    PixelData PreparePixelData(float2 uv)
    {
        PixelData pd;
        pd.PerspectiveFactor = PERSPECTIVE_COEFFITIENT;
        pd.InvSteps = rcp((float) STEPS_PER_RAY);
        pd.InvRays = rcp(RAYS_AMOUNT);
        pd.Depth = getDepth(uv);
        pd.SelfPos = float3((uv - 0.5) * pd.Depth * pd.PerspectiveFactor, pd.Depth);
        pd.ViewDir = normalize(pd.SelfPos);
        pd.Normal = normalize(getNormal(uv));
        return pd;
    }

    bool RayGen(
    in float3 rayOrigin,
    in float3 rayDir,
    out float3 hitPos,
    out float2 uvHit,
    bool enableRefinement
)
    {
        float stepSize = MIN_STEP_SIZE;
        float totalDist = 0.0;
        float3 prevPos = rayOrigin;
        float2 uvPrev = xyz_to_uv(rayOrigin); 
        float minStep = BASE_RAYS_LENGTH / (float) STEPS_PER_RAY; 

[loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + rayDir * stepSize;
            totalDist += stepSize;
            stepSize = min(stepSize * STEP_GROWTH, minStep);

            float2 uvCurr = xyz_to_uv(currPos);

            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) ||
            all(abs(uvCurr - uvPrev) < 0.0005) ||
            totalDist > MaxTraceDistance)
                break;

            // 
            float sceneDepth = getDepth(uvCurr);
            float thickness = abs(currPos.z - sceneDepth); 

            // Skip if behind geometry or too thick
            if (currPos.z < sceneDepth || thickness > ThicknessThreshold)
            {
                prevPos = currPos;
                uvPrev = uvCurr; 
                continue;
            }

            // Binary refinement to find more precise hit point (Specular)
            if (enableRefinement)
            {
                float3 lo = prevPos, hi = currPos;
        [unroll]
                for (int r = 0; r < REFINEMENT_STEPS; ++r)
                {
                    float3 mid = 0.5 * (lo + hi);
                    float2 uvm = xyz_to_uv(mid);
                    if (mid.z >= getDepth(uvm))
                        hi = mid;
                    else
                        lo = mid;
                }
                hitPos = hi;
            }
            else
            {
                hitPos = currPos;
            }

            uvHit = xyz_to_uv(hitPos);
            return true;
        }

        return false; 
    }

    // Implements screen-space reflections using stochastic ray marching
    float4 TraceSpecularGI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        PixelData pd = PreparePixelData(uv);
        float4 accum = 0;

        if (EnableSpecularGI)
        {
            float3 rayDir = reflect(pd.ViewDir, pd.Normal);
            float3 hitPos;
            float2 uvHit;

            bool hit = RayGen(pd.SelfPos, rayDir, hitPos, uvHit, true);

            if (hit)
            {
                // Compute Fresnel term using Schlick’s approximation
                const float ior = 1.5;
                const float R0 = pow((1 - ior) / (1 + ior), 2);
                float3 fn = normalize(getNormal(uvHit));
                float cosT = saturate(dot(fn, rayDir));
                float fresnel = R0 + (1 - R0) * pow(1 - cosT, 5);
                
                // Weight by view angle squared
                float angleW = pow(saturate(dot(pd.ViewDir, rayDir)), 2);
                float distFactor = saturate(1.0 - length(hitPos - pd.SelfPos) / MaxTraceDistance); 
                
                accum.rgb += GetColor(uvHit).rgb * pd.InvRays * fresnel * angleW * distFactor;
            }
            else
            {
                // Fallback in case the ray never hit any geometry
                float3 fbPos = pd.SelfPos + rayDir * (pd.Depth + 0.01) * BASE_RAYS_LENGTH;
                float2 uvFb = saturate(xyz_to_uv(fbPos));
                float align = dot(rayDir, pd.ViewDir);
                if (align > 0.3 && fbPos.z > 0 && fbPos.z < MaxTraceDistance)
                    
                accum.rgb += GetColor(float4(uvFb, 0, 0)).rgb * 0.4 * pd.InvRays;
            }
        }
        //Fade, Tone
        float fade = saturate((FadeEnd - pd.Depth) / max(FadeEnd - FadeStart, 0.001));
        accum.rgb = saturate(accum.rgb * fade / (accum.rgb + 1));
        accum.a = pd.Depth;
        return accum;
    }

// Diffuse GI 
    float4 TraceDiffuseGI(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        PixelData pd = PreparePixelData(texcoord);
        float4 accum = 0;
        
        if (EnableDiffuseGI)
        {
        float3 baseNormal = pd.Normal;
            
        const float JITTER_INTENSITY = 0.09;
        const float DIR_PERTURB = 0.09;
            
            [unroll]
            for (int r = 0; r < RAYS_AMOUNT; ++r)
            {
                // Jitter
                float seed = 32.0 * (r + 1.0) + frac(FRAME_COUNT / 48.0);
                float3 jitter = rand3d(texcoord + seed) * 2.0 - 1.0; 
                float3 pertN = normalize(baseNormal + jitter * JITTER_INTENSITY);
                float3 rayDir = normalize(pertN + jitter * DIR_PERTURB);
                float3 hitPos;
                float2 uvHit;

                if (RayGen(pd.SelfPos, rayDir, hitPos, uvHit, false))
                {
                    accum.rgb += GetColor(uvHit).rgb * pd.InvRays * IndirectIntensity;
                }
            }
        }

       //Fade, Tone
        float fade = saturate((FadeEnd - pd.Depth) / max(FadeEnd - FadeStart, 0.001));
        accum.rgb = saturate(accum.rgb * fade / (accum.rgb + 1.0));
        accum.a = pd.Depth;
        return accum;
    }
    //End GNU3
    
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

    // For Temporal
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
   
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outSpec : SV_Target1) : SV_Target
    {
        float2 motion = GetMotionVector(uv);

        // Diffuse
        float3 currentGI = tex2Dlod(sDFGI, float4(uv, 0, 0)).rgb;
        float3 historyGI = tex2Dlod(sDFGIHistory, float4(uv + motion, 0, 0)).rgb;
        float3 blendedGI = currentGI;

        if (EnableTemporal && AccumFramesDF > 0 && FRAME_COUNT > 1)
        {
            uint N = min(FRAME_COUNT, (uint) AccumFramesDF);
            blendedGI = (historyGI * (N - 1) + currentGI) / N;
        }

        // Specular
        float3 currentSpec = tex2Dlod(sSpecular, float4(uv, 0, 0)).rgb;
        float3 historySpec = tex2Dlod(sSpecularHistory, float4(uv + motion, 0, 0)).rgb;
        float3 blendedSpec = currentSpec;

        if (EnableTemporal && AccumFramesSG > 0 && FRAME_COUNT > 1)
        {
            uint N = min(FRAME_COUNT, (uint) AccumFramesSG);
            blendedSpec = (historySpec * (N - 1) + currentSpec) / N;
        }

        outSpec = float4(blendedSpec, currentSpec.r);
        return float4(blendedGI, currentGI.r);
    }

    float4 PS_SaveHistoryDFGI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 gi = EnableTemporal ? tex2Dlod(sDiffuseTemp, float4(uv, 0, 0)).rgb : tex2Dlod(sDFGI, float4(uv, 0, 0)).rgb;
        return float4(gi, 1.0);
    }
    
    float4 PS_SaveHistorySpecular(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 gi = EnableTemporal ? tex2Dlod(sSpecularTemp, float4(uv, 0, 0)).rgb : tex2Dlod(sSpecular, float4(uv, 0, 0)).rgb;
        return float4(gi, 1.0);
    }

    float3 HSVtoRGB(float3 c)
    {
        float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }
    
    float4 Combine(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        // Common data preparation
        float4 originalColor = GetColor(texcoord);
        float depth = getDepth(texcoord);
        float3 normal = getNormal(texcoord);
        float2 motion = GetMotionVector(texcoord);
        bool isValidScene = depth < 1.0;
    
        // Tex
        float3 diffuseGI = EnableTemporal
        ? tex2Dlod(sDiffuseTemp, float4(texcoord, 0, 0)).rgb 
        : tex2Dlod(sDFGI, float4(texcoord, 0, 0)).rgb;
    
        float3 specularGI = EnableTemporal
        ? tex2Dlod(sSpecularTemp, float4(texcoord, 0, 0)).rgb 
        : tex2Dlod(sSpecular, float4(texcoord, 0, 0)).rgb;
    
        diffuseGI *= IndirectIntensity;
        specularGI *= Intensity;
        float3 giColor = diffuseGI + specularGI;

        // post-processing
        if (AssumeSRGB)
            giColor = LinearizeSRGB(giColor);
        if (EnableACES)
            giColor = ApplyACES(giColor);
    
        float luminance = lum(giColor);
        giColor = lerp(luminance.xxx, giColor, Saturation);

        // Debug visualization
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / 6.283185) + 0.5, 1.0, saturate(velocity));
                    return float4(HSVtoRGB(hsv), 1.0);
            
                case 2: // Combined GI
                    return float4(giColor, 1.0);
            
                case 3: // Diffuse GI
                    return float4(diffuseGI, 1.0);
            
                case 4: // Specular GI
                    return float4(specularGI, 1.0);
            
                case 5: // Normals
                    return float4(normal * 0.5 + 0.5, 1.0);
            
                case 6: // Depth
                    return float4(depth.xxx, 1.0);
            }
            return originalColor;
        }

        switch (BlendMode)
        {
            case 0: // Additive
                return float4(originalColor.rgb + giColor, originalColor.a);
        
            case 1: // Multiplicative
                return float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - giColor), originalColor.a);
        
            case 2: // Alpha Blend
                return float4(lerp(originalColor.rgb, giColor, saturate(giColor.r)), originalColor.a);
        }

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
        pass Specular
        {
            VertexShader = PostProcessVS;
            PixelShader = TraceSpecularGI;
            RenderTarget = Specular;
        }
        pass Diffuse
        {
            VertexShader = PostProcessVS;
            PixelShader = TraceDiffuseGI;
            RenderTarget = DiffuseGI;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget0 = DiffuseTemp;
            RenderTarget1 = SpecularTemp;
        }
        pass Save_History_Specular
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistorySpecular;
            RenderTarget = SpecularHistory;
        }
        pass Save_History_Diffuse
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistoryDFGI;
            RenderTarget = DiffuseHistory;
        }
        pass Combine
        {
            VertexShader = PostProcessVS;
            PixelShader = Combine;
        }
    }
}
