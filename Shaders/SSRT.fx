 /*------------------.
| :: Description :: |
'-------------------/

    SSRT

    Version 1.6
    Author: Barbatos Bachiko
    Original SSRT by jebbyk : https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx

    License: GNU Affero General Public License v3.0 : https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE
    Aces Tonemapping use Pentalimbed Unlicense: https://github.com/Pentalimbed/YASSGI/blob/main/UNLICENSE

    About: Screen Space Ray Tracing focused on reflections.
 
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.6
    - Remove DF
    x Marty Mods Launchpad working 
*/

/*-------------------.
| :: Definitions ::  |
'-------------------*/

#include "ReShade.fxh"

// Motion vector configuration
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
#define fov 28.6
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 
#define AspectRatio BUFFER_WIDTH/BUFFER_HEIGHT
#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

#define MVErrorTolerance 0.96
#define SkyDepth 0.99
#define MAX_Frames 64
/*-------------------.
| :: Parameters ::   |
'-------------------*/

uniform float SPIntensity <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 3.0;
    ui_step = 0.001;
    ui_category = "General";
    ui_label = "Specular Intensity";
> = 1.1;

// Bump Mapping Settings
uniform float BumpIntensity <
    ui_type = "slider";
    ui_category = "Bump Mapping";
    ui_label = "Bump Intensity";
    ui_min = 0.030; ui_max = 1.0; ui_step = 0.001;
> = 0.030;

uniform float FadeStart
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade Start";
        ui_tooltip = "Distance starts to fade out";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.0;

uniform float FadeEnd
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade End";
        ui_tooltip = "Distance completely fades out";
        ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    >
    = 4.999;

// Temporal Settings
uniform float AccumFramesSG <
    ui_type = "slider";
    ui_category = "Temporal";
     ui_label = "SSR Temporal";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 1.0;

uniform float ThicknessThreshold <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Thickness Threshold";
> = 0.010;

// Depth Settings
uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normal";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 0.3;

// Normal Settings
uniform bool bSmoothNormals <
 ui_category = "Depth/Normal";
    ui_label = "Smooth Normals";
> = true;

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_items = "None\0Motion\0GI\0Normal\0Depth\0";
> = 0;

// Extra Settings
uniform bool AssumeSRGB < 
    ui_category = "Tone Mapping";
    ui_label = "Assume sRGB Input";
> = false;

uniform bool EnableACES <
    ui_category = "Tone Mapping";
    ui_label = "Enable ACES Tone Mapping";
> = false;

uniform float Saturation <
    ui_type = "slider";
    ui_category = "Extra";
    ui_label = "Saturation";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 1.05;

uniform int BlendMode <
    ui_type = "combo";
    ui_category = "Extra";
    ui_label = "Blend Mode";
    ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
> = 1;

uniform int FRAME_COUNT < source = "framecount"; >;

static const float PI2div360 = 0.01745329;
#define rad(x) (x * PI2div360)

//Bump Mapping
static const float3 BumpDirection = float3(-2.0, 1.0, -0.5);
static const float BumpDepth = 0.7;

//Ray Marching
static const float MaxTraceDistance = 1.0;
static const float BASE_RAYS_LENGTH = 1.0;
static const int STEPS_PER_RAY = 128;

// Adaptive step 
static const float MIN_STEP_SIZE = 0.001;
static const float STEP_GROWTH = 1.0803;
static const int REFINEMENT_STEPS = 6;

static const float EnableTemporal = true;
/*---------------.
| :: Textures :: |
'---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
}
float2 sampleMotion(float2 texcoord) {
    return tex2D(Deferred::sMotionVectorsTex, texcoord).rg;
}

#elif USE_VORT_MOTION
texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC };
float2 sampleMotion(float2 texcoord) {
    return tex2D(sMotVectTexVort, texcoord).rg;
}
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
float2 sampleMotion(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace SSRT24
{
    texture SP
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSP
    {
        Texture = SP;
    };
    
    texture SP_TEMP
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSP_TEMP
    {
        Texture = SP_TEMP;
    };
    
    texture SP_HISTORY
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSP_HISTORY
    {
        Texture = SP_HISTORY;
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

    float lum(float3 color)
    {
        return (color.r + color.g + color.b) * 0.3333333;
    }

    // Color
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

    float3 UVtoPos(float2 texcoord)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, getDepth(texcoord) * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord.xy *= rad(fov);
        return scrncoord.xyz;
    }

    float3 UVtoPos(float2 texcoord, float depth)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, depth * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord *= rad(fov);
        return scrncoord.xyz;
    }

    float2 PostoUV(float3 position)
    {
        float2 scrnpos = position.xy;
        scrnpos /= rad(fov);
        scrnpos.x /= AspectRatio;
        scrnpos /= position.z;
        return scrnpos / 2 + 0.5;
    }

    float3 computeNormal(float2 texcoord)
    {
        float2 p = pix;
        float3 u, d, l, r, u2, d2, l2, r2;
	
        u = UVtoPos(texcoord + float2(0, p.y));
        d = UVtoPos(texcoord - float2(0, p.y));
        l = UVtoPos(texcoord + float2(p.x, 0));
        r = UVtoPos(texcoord - float2(p.x, 0));
	
        p *= 2;
	
        u2 = UVtoPos(texcoord + float2(0, p.y));
        d2 = UVtoPos(texcoord - float2(0, p.y));
        l2 = UVtoPos(texcoord + float2(p.x, 0));
        r2 = UVtoPos(texcoord - float2(p.x, 0));
	
        u2 = u + (u - u2);
        d2 = d + (d - d2);
        l2 = l + (l - l2);
        r2 = r + (r - r2);
	
        float3 c = UVtoPos(texcoord);
	
        float3 v = u - c;
        float3 h = r - c;
	
        if (abs(d2.z - c.z) < abs(u2.z - c.z))
            v = c - d;
        if (abs(l2.z - c.z) < abs(r2.z - c.z))
            h = c - l;
	
        return normalize(cross(v, h));
    }

    // SmoothNormal by AlucardDH MIT Licence
    float3 GetNormal(float2 texcoord)
    {
        float2 px = ReShade::PixelSize;
        float3 normal = computeNormal(texcoord);

        if (bSmoothNormals)
        {
            float2 offset2 = px * 7.5 * (1.0 - getDepth(texcoord));

            float3 normalTop = computeNormal(texcoord - float2(0, offset2.y));
            float3 normalBottom = computeNormal(texcoord + float2(0, offset2.y));
            float3 normalLeft = computeNormal(texcoord - float2(offset2.x, 0));
            float3 normalRight = computeNormal(texcoord + float2(offset2.x, 0));

            float weightTop = smoothstep(1, 0, distance(normal, normalTop) * 1.5) * 2;
            float weightBottom = smoothstep(1, 0, distance(normal, normalBottom) * 1.5) * 2;
            float weightLeft = smoothstep(1, 0, distance(normal, normalLeft) * 1.5) * 2;
            float weightRight = smoothstep(1, 0, distance(normal, normalRight) * 1.5) * 2;

            float4 weightedNormal =
              float4(normal, 1.0)
            + float4(normalTop * weightTop, weightTop)
            + float4(normalBottom * weightBottom, weightBottom)
            + float4(normalLeft * weightLeft, weightLeft)
            + float4(normalRight * weightRight, weightRight);

            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }

    // Bump mapping 
    {
            float h00 = lum(GetColor(texcoord + px * float2(-1, -1)).rgb);
            float h10 = lum(GetColor(texcoord + px * float2( 0, -1)).rgb);
            float h20 = lum(GetColor(texcoord + px * float2( 1, -1)).rgb);
            float h01 = lum(GetColor(texcoord + px * float2(-1,  0)).rgb);
            float h21 = lum(GetColor(texcoord + px * float2( 1,  0)).rgb);
            float h02 = lum(GetColor(texcoord + px * float2(-1,  1)).rgb);
            float h12 = lum(GetColor(texcoord + px * float2( 0,  1)).rgb);
            float h22 = lum(GetColor(texcoord + px * float2( 1,  1)).rgb);

            float dx = (h00 + 2 * h01 + h02) - (h20 + 2 * h21 + h22);
            float dy = (h00 + 2 * h10 + h20) - (h02 + 2 * h12 + h22);

            float2 slope = float2(dx, dy) * BumpIntensity;

            float height = lum(GetColor(texcoord).rgb);
            float holeDepth = (1.0 - height) * BumpDepth * BumpDirection.z;

            float3 N = normal;
            float3 T = normalize(cross(N, float3(0, 1, 0)));
            float3 B = cross(N, T);

            float3 bumpedNormal = N + (T * slope.x * BumpDirection.x + B * slope.y * BumpDirection.y + N * holeDepth);
            normal = normalize(bumpedNormal);
        }

        return normal;
    }
    
    float3 getNormal(float2 coords)
    {
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
    }
 
    float2 GetMotionVector(float2 texcoord)
    {
        float2 p = pix;
        float2 MV = sampleMotion(texcoord);

        if (MVErrorTolerance < 1)
        {
            if (abs(MV.x) < p.x && abs(MV.y) < p.y)
                MV = 0;
        }

#if USE_MARTY_LAUNCHPAD_MOTION
    MV = tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
    MV = tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#endif

        return MV;
    }
    
    // GNU 3 License functions
    struct PixelData
    {
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
        pd.InvSteps = rcp((float) STEPS_PER_RAY);
        pd.Depth = getDepth(uv).x;
        pd.SelfPos = float3((uv - 0.5) * pd.Depth, pd.Depth);
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
        float2 uvPrev = PostoUV(rayOrigin);
        float minStep = BASE_RAYS_LENGTH / (float) STEPS_PER_RAY;

    [loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + rayDir * stepSize;
            totalDist += stepSize;
            stepSize = min(stepSize * STEP_GROWTH, minStep);

            float2 uvCurr = PostoUV(currPos);

            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) ||
                all(abs(uvCurr - uvPrev) < 0.0005) ||
                totalDist > MaxTraceDistance)
                break;

            float sceneDepth = getDepth(uvCurr).x;
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

                for (int r = 0; r < REFINEMENT_STEPS; ++r)
                {
                    float3 mid = 0.5 * (lo + hi);
                    float2 uvm = PostoUV(mid);
                    if (mid.z >= getDepth(uvm).x)
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

            uvHit = PostoUV(hitPos);
            return true;
        }

        return false;
    }
    
    struct PS_OUTPUT
    {
        float4 specular : SV_Target0;
    };

    PS_OUTPUT SPDF(float4 pos : SV_Position, float2 uv : TEXCOORD)
    {
        PixelData pd = PreparePixelData(uv);
        PS_OUTPUT output;
        output.specular = 0;
        const float specularRayWeight = 1.0;

        float3 position = UVtoPos(uv);
        float3 eyedir = normalize(position);
        float3 rayDir = normalize(reflect(eyedir, pd.Normal));
    
        float3 hitPos;
        float2 uvHit;
        bool hit = RayGen(pd.SelfPos, rayDir, hitPos, uvHit, true);
        if (hit)
        {
            float angleW = saturate(dot(pd.ViewDir, rayDir));
            angleW *= angleW; // Quadratic weighting

            float distFactor = saturate(1.0 - length(hitPos - pd.SelfPos) / MaxTraceDistance);
            float3 specColor = GetColor(uvHit).rgb;
            output.specular.rgb = specColor * specularRayWeight * angleW * distFactor;
        }
        else
    {
        // Fallback para raios que não bateram
        float3 fbWorld = pd.SelfPos + rayDir * (pd.Depth + 0.01) * BASE_RAYS_LENGTH;
        float2 uvFb = saturate(PostoUV(fbWorld));
        float align = saturate(dot(rayDir, pd.ViewDir));
        float3 fbColor = GetColor(uvFb).rgb;
        output.specular.rgb = fbColor * 0.4 * specularRayWeight *
                          step(0.3, align) *
                          step(0.0, fbWorld.z) *
                          step(fbWorld.z, MaxTraceDistance);
    }


        float fadeRange = max(FadeEnd - FadeStart, 0.001);
        float fade = saturate((FadeEnd - pd.Depth) / fadeRange);
        fade *= fade;

        output.specular.rgb *= fade;
        output.specular.a = pd.Depth;
        return output;
    }
    //End GNU3
    
    // Shader passes
    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }
    
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD,
                                     out float outHistoryLength : SV_Target1) : SV_Target
    {
        float2 motion = GetMotionVector(uv);
        float2 reprojectedUV = uv + motion;
    
        float3 currentSpec = tex2Dlod(sSP, float4(uv, 0, 0)).rgb;
        float3 historySpec = tex2Dlod(sSP_HISTORY, float4(reprojectedUV, 0, 0)).rgb;
    
        // Disocclusion
        float depth = getDepth(uv);
        bool validHistory = (depth <= SkyDepth) &&
                          all(saturate(reprojectedUV) == reprojectedUV) &&
                          FRAME_COUNT > 1;
    
        float3 blendedSpec = currentSpec;
    
        if (EnableTemporal && validHistory)
        {
            if (AccumFramesSG > 0)
            {
                float alphaSpec = 1.0 / min(FRAME_COUNT, (float) AccumFramesSG);
                blendedSpec = lerp(historySpec, currentSpec, alphaSpec);
            }
        }
        
        outHistoryLength = validHistory ? min(FRAME_COUNT, MAX_Frames) : 0;
        return float4(blendedSpec, currentSpec.r);
    }
    
    float4 PS_SaveHistorySP(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 spec = EnableTemporal ? tex2Dlod(sSP_TEMP, float4(uv, 0, 0)).rgb : tex2Dlod(sSP, float4(uv, 0, 0)).rgb;
        return float4(spec, 1.0);
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
        float3 specularGI = EnableTemporal
            ? tex2Dlod(sSP_TEMP, float4(texcoord, 0, 0)).rgb 
            : tex2Dlod(sSP, float4(texcoord, 0, 0)).rgb;
    
        specularGI *= SPIntensity;
        float3 giColor = specularGI;

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
                
                case 2: // GI (agora apenas especular)
                    return float4(giColor, 1.0);
                
                case 3: // Normals
                    return float4(normal * 0.5 + 0.5, 1.0);
                
                case 4: // Depth
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
        }
        pass SPDF
        {
            VertexShader = PostProcessVS;
            PixelShader = SPDF;
            RenderTarget = SP;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget = SP_TEMP;
        }
        pass Save_History
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistorySP;
            RenderTarget = SP_HISTORY;
        }
        pass Combine
        {
            VertexShader = PostProcessVS;
            PixelShader = Combine;
        }
    }
}
