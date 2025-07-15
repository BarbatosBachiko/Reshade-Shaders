/*--------------------.
| :: Description ::  |
'--------------------/

    SSRT

    Version 1.6.5
    Author: Barbatos Bachiko
    Original SSRT by jebbyk : https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx

    License: GNU Affero General Public License v3.0 : https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE
    Aces Tonemapping use Pentalimbed Unlicense: https://github.com/Pentalimbed/YASSGI/blob/main/UNLICENSE

    About: Screen Space Ray Tracing focused on reflections.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.6.5
    + Revised
*/

#include "ReShade.fxh"

//------------------------------------------------------------------------------------------------|
// :: Preprocessor Definitions & Constants ::
//------------------------------------------------------------------------------------------------|

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#ifndef RES_SCALE
#define RES_SCALE 0.8
#endif

#define RES_WIDTH (ReShade::ScreenSize.x * RES_SCALE)
#define RES_HEIGHT (ReShade::ScreenSize.y * RES_SCALE)

#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define fov 28.6
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 
#define AspectRatio BUFFER_WIDTH/BUFFER_HEIGHT

#define MVErrorTolerance 0.96
#define SkyDepth 0.99
#define MAX_Frames 64

static const float PI2div360 = 0.01745329;
#define rad(x) (x * PI2div360)

//Bump Mapping
static const float3 BumpDirection = float3(-2.0, 1.0, -0.5);
static const float BumpDepth = 0.7;

//Ray Marching
#define  MaxTraceDistance 1
static const int STEPS_PER_RAY = 128;

// Adaptive step 
#define STEP_SCALE  0.7
#define MIN_STEP_SIZE 0.001 
#define MAX_STEP_SIZE 1.0
static const int REFINEMENT_STEPS = 5;

static const float EnableTemporal = true;

//------------------------------------------------------------------------------------------------|
// :: UI Parameters ::
//------------------------------------------------------------------------------------------------|

uniform float SPIntensity <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 3.0;
    ui_step = 0.001;
    ui_category = "General";
    ui_label = "Specular Intensity";
> = 1.1;

uniform float BumpIntensity <
    ui_type = "drag";
    ui_category = "Bump Mapping";
    ui_label = "Bump Intensity";
    ui_min = 0.000; ui_max = 1.0; ui_step = 0.001;
> = 0.030;

uniform float FadeStart <
    ui_category = "Fade Settings";
    ui_type = "slider";
    ui_label = "Fade Start";
    ui_tooltip = "Distance starts to fade out";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.0;

uniform float FadeEnd <
    ui_category = "Fade Settings";
    ui_type = "slider";
    ui_label = "Fade End";
    ui_tooltip = "Distance completely fades out";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
> = 4.999;

uniform float ThicknessThreshold <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Thickness Threshold";
> = 0.010;

uniform float AccumFramesSG <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "SSR Temporal Frames";
    ui_tooltip = "Higher values are smoother but can cause more ghosting.";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 4.0;

uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normal";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 0.3;

uniform bool bSmoothNormals <
    ui_category = "Depth/Normal";
    ui_label = "Smooth Normals";
> = true;

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_items = "None\0Motion\0Reflection\0Normal\0Depth\0";
> = 0;

uniform bool AssumeSRGB < 
    ui_category = "Tone Mapping";
    ui_label = "Assume sRGB Input";
> = false;

uniform bool EnableACES <
    ui_category = "Tone Mapping";
    ui_label = "Enable ACES Tone Mapping";
> = false;

uniform float3 Adjustments <
    ui_category = "Tone Mapping";
    ui_label = "Saturation / Exposure / Contrast";
> = float3(1.5, 0.8, 1.1);

uniform int BlendMode <
    ui_type = "combo";
    ui_category = "Extra";
    ui_label = "Blend Mode";
    ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
> = 1;

uniform int FRAME_COUNT < source = "framecount"; >;

//------------------------------------------------------------------------------------------------|
// :: Textures & Samplers ::
//------------------------------------------------------------------------------------------------|

#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
}
float2 SampleMotionVectors(float2 texcoord) {
    return tex2D(Deferred::sMotionVectorsTex, texcoord).rg;
}

#elif USE_VORT_MOTION
texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC };
float2 SampleMotionVectors(float2 texcoord) {
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
float2 SampleMotionVectors(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace SSRT165
{
    texture SSR
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sSSR
    {
        Texture = SSR;
    };
    
    texture TEMP
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sTEMP
    {
        Texture = TEMP;
    };
    
    texture HISTORY
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };
    sampler sHISTORY
    {
        Texture = HISTORY;
    };

    texture normalTex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormal
    {
        Texture = normalTex;S_PC
    };

//----------------|
// :: Functions ::|
//----------------|

    float3 RGBToYCoCg(float3 rgb)
    {
        float Y = dot(rgb, float3(0.25, 0.5, 0.25));
        float Co = dot(rgb, float3(0.5, 0, -0.5));
        float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
        return float3(Y, Co, Cg);
    }

    float3 YCoCgToRGB(float3 ycocg)
    {
        float Y = ycocg.x;
        float Co = ycocg.y;
        float Cg = ycocg.z;
        float r = Y + Co - Cg;
        float g = Y + Cg;
        float b = Y - Co - Cg;
        return float3(r, g, b);
    }
    
    float3 HSVToRGB(float3 c)
    {
        float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }
    
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

    float GetLuminance(float3 color)
    {
        return (color.r + color.g + color.b) * 0.3333333;
    }

    float3 LinearizeSRGB(float3 color)
    {
        return pow(color, 2.2);
    }

    float3 sRGBToACEScg(float3 srgb)
    {
        return mul(g_sRGBToACEScg, srgb);
    }

    float3 ACEScgToSRGB(float3 acescg)
    {
        return mul(g_ACEScgToSRGB, acescg);
    }

    float3 Apply_ACES(float3 color)
    {
        if (!EnableACES)
            return color;
    
        float3 acescg = sRGBToACEScg(color);

        const float A = 2.51;
        const float B = 0.03;
        const float C = 2.43;
        const float D = 0.59;
        const float E = 0.14;

        float3 toneMapped = (acescg * (A * acescg + B)) / (acescg * (C * acescg + D) + E);

        return ACEScgToSRGB(toneMapped);
    }

    float3 ApplyBumpMapping(float2 texcoord, float3 normal)
    {
        float2 px = ReShade::PixelSize;

        float3 col00 = GetColor(texcoord + px * float2(-1, -1)).rgb;
        float3 col10 = GetColor(texcoord + px * float2( 0, -1)).rgb;
        float3 col20 = GetColor(texcoord + px * float2( 1, -1)).rgb;
        float3 col01 = GetColor(texcoord + px * float2(-1,  0)).rgb;
        float3 col21 = GetColor(texcoord + px * float2( 1,  0)).rgb;
        float3 col02 = GetColor(texcoord + px * float2(-1,  1)).rgb;
        float3 col12 = GetColor(texcoord + px * float2( 0,  1)).rgb;
        float3 col22 = GetColor(texcoord + px * float2( 1,  1)).rgb;
        float3 colCenter = GetColor(texcoord).rgb;

        float h00 = GetLuminance(col00);
        float h10 = GetLuminance(col10);
        float h20 = GetLuminance(col20);
        float h01 = GetLuminance(col01);
        float h21 = GetLuminance(col21);
        float h02 = GetLuminance(col02);
        float h12 = GetLuminance(col12);
        float h22 = GetLuminance(col22);
        float height = GetLuminance(colCenter);

        // Sobel
        float dx = (h00 + 2 * h01 + h02) - (h20 + 2 * h21 + h22);
        float dy = (h00 + 2 * h10 + h20) - (h02 + 2 * h12 + h22);
        float2 slope = float2(dx, dy) * BumpIntensity;

        float holeDepth = (1.0 - height) * BumpDepth * BumpDirection.z;

        //TBN
        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T = normalize(cross(up, normal));
        float3 B = cross(normal, T);

        float3 bumpedNormal = normal + (T * slope.x * BumpDirection.x + B * slope.y * BumpDirection.y + normal * holeDepth);

        return normalize(bumpedNormal);
    }
    
    float3 UVToWorld(float2 texcoord)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, getDepth(texcoord) * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord.xy *= rad(fov);
        return scrncoord.xyz;
    }

    float3 UVToWorld(float2 texcoord, float depth)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, depth * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord *= rad(fov);
        return scrncoord.xyz;
    }

    float2 WorldToUV(float3 position)
    {
        float2 scrnpos = position.xy;
        scrnpos /= rad(fov);
        scrnpos.x /= AspectRatio;
        scrnpos /= position.z;
        return scrnpos / 2 + 0.5;
    }

    float3 CalculateNormalFromDepth(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;

        float3 center = UVToWorld(texcoord);
        float3 up = UVToWorld(texcoord + float2(0, p.y));
        float3 down = UVToWorld(texcoord - float2(0, p.y));
        float3 left = UVToWorld(texcoord - float2(p.x, 0));
        float3 right = UVToWorld(texcoord + float2(p.x, 0));
        float3 dx = right - left;
        float3 dy = up - down;
        float3 normal = normalize(cross(dy, dx));

        return normal;
    }

    float3 GetSurfaceNormal(float2 texcoord)
    {
        float2 px = ReShade::PixelSize;
        float3 normal = CalculateNormalFromDepth(texcoord);

        if (bSmoothNormals)
        {
            float2 offset2 = px * 7.5 * (1.0 - getDepth(texcoord));

            float3 normalTop = CalculateNormalFromDepth(texcoord - float2(0, offset2.y));
            float3 normalBottom = CalculateNormalFromDepth(texcoord + float2(0, offset2.y));
            float3 normalLeft = CalculateNormalFromDepth(texcoord - float2(offset2.x, 0));
            float3 normalRight = CalculateNormalFromDepth(texcoord + float2(offset2.x, 0));

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

        normal = ApplyBumpMapping(texcoord, normal);

        return normal;
    }
    
    float3 SampleNormalFromTexture(float2 coords)
    {
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
    }
 
    float2 GetFilteredMotionVector(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float2 MV = SampleMotionVectors(texcoord);

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
    
    bool TraceRay(in float3 rayOrigin, in float3 rayDir, out float3 hitPos, out float2 uvHit, bool enableRefinement)
    {
        float stepSize = MIN_STEP_SIZE;
        float totalDist = 0.0;
        float3 prevPos = rayOrigin;
        float2 uvPrev = WorldToUV(rayOrigin);

        [loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + rayDir * stepSize;
            totalDist += stepSize;

            float2 uvCurr = WorldToUV(currPos);
            if ((any(uvCurr < 0.0) || any(uvCurr > 1.0)) || totalDist > MaxTraceDistance)
                break;

            float sceneDepth = getDepth(uvCurr).x;
            float thickness = abs(currPos.z - sceneDepth);

            if (currPos.z < sceneDepth || thickness > ThicknessThreshold)
            {
                prevPos = currPos;
                uvPrev = uvCurr;

                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * STEP_SCALE, MIN_STEP_SIZE, MAX_STEP_SIZE);
                continue;
            }

            if (enableRefinement)
            {
                float3 lo = prevPos, hi = currPos;

                [unroll]
                for (int r = 0; r < REFINEMENT_STEPS; ++r)
                {
                    float3 mid = 0.5 * (lo + hi);
                    float2 uvm = WorldToUV(mid);
                    float midDepth = getDepth(uvm).x;

                    if (mid.z >= midDepth)
                        hi = mid;
                    else
                        lo = mid;
                }

                hitPos = hi;
                uvHit = WorldToUV(hi);
            }
            else
            {
                hitPos = currPos;
                uvHit = uvCurr;
            }

            return true;
        }

        return false;
    }

    struct PixelData
    {
        float InvSteps;
        float Depth;
        float3 SelfPos;
        float3 ViewDir;
        float3 Normal;
    };

    PixelData SetupPixelData(float2 uv)
    {
        PixelData pd;
        pd.InvSteps = rcp((float) STEPS_PER_RAY);
        pd.Depth = getDepth(uv).x;
        pd.SelfPos = float3((uv - 0.5) * pd.Depth, pd.Depth);
        pd.ViewDir = normalize(pd.SelfPos);
        pd.Normal = normalize(SampleNormalFromTexture(uv));

        return pd;
    }

//--------------------|
// :: Pixel Shaders ::|
//--------------------|

    struct PS_OUTPUT
    {
        float4 specular : SV_Target0;
    };

    PS_OUTPUT PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD)
    {
        float Saturation = Adjustments.r;
        float Exposure = Adjustments.g;
        float Contrast = Adjustments.b;

        PixelData pd = SetupPixelData(uv);
        PS_OUTPUT output;
        output.specular = 0;
        const float specularRayWeight = 1.0;
        float3 position = UVToWorld(uv);
        float3 eyedir = normalize(position);
        float3 rayDir = normalize(reflect(eyedir, pd.Normal));

        float3 hitPos;
        float2 uvHit;
        bool hit = TraceRay(pd.SelfPos, rayDir, hitPos, uvHit, true);

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
            //fallback
            float adaptiveDist = pd.Depth * 1.2 + 0.01;
            float3 fbWorld = pd.SelfPos + rayDir * adaptiveDist;
            float2 uvFb = saturate(WorldToUV(fbWorld));
            float3 fbColor = GetColor(uvFb).rgb;
            float angleWeight = pow(saturate(dot(pd.ViewDir, rayDir)), 2.0);
            float depthFactor = saturate(1.0 - pd.Depth / MaxTraceDistance);
            float weight = specularRayWeight * angleWeight * depthFactor;
            output.specular.rgb = fbColor * weight;
        }
        output.specular.rgb *= Exposure;

        if (AssumeSRGB)
            output.specular.rgb = LinearizeSRGB(output.specular.rgb);

        if (EnableACES)
            output.specular.rgb = Apply_ACES(output.specular.rgb);
        
        float luminance = GetLuminance(output.specular.rgb);
        output.specular.rgb = lerp(luminance.xxx, output.specular.rgb, Saturation);
        output.specular.rgb = (output.specular.rgb - 0.5) * Contrast + 0.5;

        float fadeRange = max(FadeEnd - FadeStart, 0.001);
        float fade = saturate((FadeEnd - pd.Depth) / fadeRange);
        fade *= fade;
        output.specular.rgb *= fade;
        output.specular.a = pd.Depth;

        return output;
    }
    
    float4 PS_GenerateNormals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetSurfaceNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float4 PS_ApplyTemporalFilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 currentSpec = tex2D(sSSR, uv).rgb;
        float currentDepth = getDepth(uv);

        float2 motion = GetFilteredMotionVector(uv);
        float2 reprojectedUV = uv + motion;

        float historyDepth = getDepth(reprojectedUV);
        bool validHistory = all(saturate(reprojectedUV) == reprojectedUV) &&
                                  FRAME_COUNT > 1 &&
                                  abs(historyDepth - currentDepth) < 0.01;

        float3 blendedSpec = currentSpec;

        if (EnableTemporal && validHistory)
        {
            float3 minBox = RGBToYCoCg(currentSpec);
            float3 maxBox = minBox;

            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    float3 neighborSpec = RGBToYCoCg(tex2Dlod(sSSR, float4(uv + float2(x, y) * (ReShade::PixelSize / RES_SCALE), 0, 0)).rgb);
                    minBox = min(minBox, neighborSpec);
                    maxBox = max(maxBox, neighborSpec);
                }
            }
        
            float3 historySpec = RGBToYCoCg(tex2Dlod(sHISTORY, float4(reprojectedUV, 0, 0)).rgb);
            float3 clampedHistorySpec = clamp(historySpec, minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, AccumFramesSG);
            blendedSpec = YCoCgToRGB(lerp(clampedHistorySpec, RGBToYCoCg(currentSpec), alpha));
        }

        float historyLengthPacked = validHistory ? min(FRAME_COUNT, MAX_Frames) : 0;
        return float4(blendedSpec, historyLengthPacked);
    }

    float4 PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 spec = EnableTemporal ? tex2Dlod(sTEMP, float4(uv, 0, 0)).rgb : tex2Dlod(sSSR, float4(uv, 0, 0)).rgb;
        return float4(spec, 1.0);
    }
    
    float4 PS_Output(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(texcoord);
        float depth = getDepth(texcoord);
        float3 normal = GetSurfaceNormal(texcoord);
        float2 motion = GetFilteredMotionVector(texcoord);
        bool isValidScene = depth < 1.0;

        float3 specularGI = EnableTemporal
            ? tex2Dlod(sTEMP, float4(texcoord, 0, 0)).rgb
            : tex2Dlod(sSSR, float4(texcoord, 0, 0)).rgb;

        specularGI *= SPIntensity;
        float3 giColor = specularGI;

        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / 6.283185) + 0.5, 1.0, saturate(velocity));
                    return float4(HSVToRGB(hsv), 1.0);
            
                case 2: // GI
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
            {
                    float giLuminance = GetLuminance(giColor);
                    return float4(lerp(originalColor.rgb, giColor, saturate(giLuminance)), originalColor.a);
                }
        }
        return originalColor;
    }
}

technique SSRT
{
    pass GenerateNormals
    {
        VertexShader = PostProcessVS;
        PixelShader = SSRT165::PS_GenerateNormals;
        RenderTarget = SSRT165::normalTex;
    }
    pass TraceReflections
    {
        VertexShader = PostProcessVS;
        PixelShader = SSRT165::PS_TraceReflections;
        RenderTarget = SSRT165::SSR;
    }
    pass TemporalFilter
    {
        VertexShader = PostProcessVS;
        PixelShader = SSRT165::PS_ApplyTemporalFilter;
        RenderTarget = SSRT165::TEMP;
    }
    pass UpdateHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = SSRT165::PS_UpdateHistory;
        RenderTarget = SSRT165::HISTORY;
    }
    pass Output
    {
        VertexShader = PostProcessVS;
        PixelShader = SSRT165::PS_Output;
    }
}
