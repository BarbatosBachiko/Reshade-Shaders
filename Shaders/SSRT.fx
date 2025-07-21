/*--------------------.
| :: Description ::  |
'--------------------/

    SSRT

    Version 1.6.91
    Author: Barbatos Bachiko
    Original SSRT by jebbyk : https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx
    Modification: Sky reflection fallback logic.

    License: GNU Affero General Public License v3.0 : https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE
    Aces Tonemapping use Pentalimbed Unlicense: https://github.com/Pentalimbed/YASSGI/blob/main/UNLICENSE

    About: Screen Space Ray Tracing focused on reflections.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.6.91
    + Fallback SSR
*/

#include "ReShade.fxh"

//-------------------------------------------|
// :: Preprocessor Definitions & Constants ::|
//-------------------------------------------|

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define getDepth(coords) (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;
#define AspectRatio (BUFFER_WIDTH / (float)BUFFER_HEIGHT)

#define MVErrorTolerance 0.96

//Ray Marching
#define MaxTraceDistance 2
static const int STEPS_PER_RAY = 128;
static const int REFINEMENT_STEPS = 5;

// Adaptive step 
#define STEP_SCALE 0.7
#define MIN_STEP_SIZE 0.001
#define MAX_STEP_SIZE 1.0

// Original Bump Mapping Constants
static const float3 BumpDirection = float3(-2.0, 1.0, -0.5);
static const float BumpDepth = 0.7;

//----------|
// :: UI :: |
//----------|

uniform float SPIntensity <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 3.0;
    ui_step = 0.001;
    ui_category = "General";
    ui_label = "Specular Intensity";
> = 1.1;

uniform float BumpIntensity <
    ui_type = "drag";
    ui_category = "BumpMapping";
    ui_label = "Bump Mapping Intensity";
    ui_min = 0.000; ui_max = 0.010; ui_step = 0.001;
> = 0.002;

uniform float FadeStart <
    ui_category = "Fade Settings";
    ui_type = "slider";
    ui_label = "Fade Start";
    ui_tooltip = "Distance at which the reflection begins to fade out.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.0;

uniform float FadeEnd <
    ui_category = "Fade Settings";
    ui_type = "slider";
    ui_label = "Fade End";
    ui_tooltip = "Distance at which the reflection completely disappears.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
> = 4.999;

uniform float VerticalFOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
> = 37.0;

uniform float RenderScale <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Advanced";
    ui_label = "Render Scale";
    ui_tooltip = "Controls the resolution at which reflections are rendered. Lower values improve performance.";
> = 0.8;

uniform float GeoCorrectionIntensity <
    ui_type = "drag";
    ui_category = "Advanced";
    ui_label = "Geometry Correction";
    ui_min = -0.070; ui_max = 0.150; ui_step = 0.001;
> = 0.009;

uniform float ThicknessThreshold <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Thickness Threshold";
> = 0.010;

uniform float NormalBias <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Advanced";
    ui_label = "Reflection Normal Bias";
    ui_tooltip = "Prevents self-reflection artifacts by comparing surface normals.";
> = 0.2;

uniform float AccumFramesSG <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "SSR Temporal Frames";
    ui_tooltip = "Higher values are smoother but may cause more ghosting.";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 4.0;

uniform bool EnableTemporal <
    ui_category = "Temporal";
    ui_label = "Enable Temporal Accumulation";
> = true;

uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normal";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

uniform bool bSmoothNormals <
    ui_category = "Depth/Normal";
    ui_label = "Smooth Normals";
> = true;

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "View Mode";
    ui_items = "None\0Motion\0Reflection\0Normal\0Depth\0Raw Reflection\0";
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
> = float3(1.5, 1.0, 1.1);

uniform int BlendMode <
    ui_type = "combo";
    ui_category = "Extra";
    ui_label = "Blend Mode";
    ui_items = "Additive\0Screen\0Alpha Blend\0";
> = 1;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

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

namespace SSRT1661
{
    texture SSR
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sSSR
    {
        Texture = SSR;
        AddressU = Clamp;
        AddressV = Clamp;
    };
    
    texture TEMP
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sTEMP
    {
        Texture = TEMP;
        AddressU = Clamp;
        AddressV = Clamp;
    };
    
    texture HISTORY
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sHISTORY
    {
        Texture = HISTORY;
        AddressU = Clamp;
        AddressV = Clamp;
    };

    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormal
    {
        Texture = normalTex;S_PC
    };

//-------------------|
// :: Functions    ::|
//-------------------|
    
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

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
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

        const float A = 2.51, B = 0.03, C = 2.43, D = 0.59, E = 0.14;
        float3 toneMapped = (acescg * (A * acescg + B)) / (acescg * (C * acescg + D) + E);

        return ACEScgToSRGB(toneMapped);
    }

    float3 UVToViewPos(float2 uv, float view_z)
    {
        float2 clip_pos = uv * 2.0 - 1.0;

        float proj_scale_y = 1.0 / tan(radians(VerticalFOV * 0.5));
        float proj_scale_x = proj_scale_y / AspectRatio;
        
        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z;
        view_pos.z = view_z;

        return view_pos;
    }

    float2 ViewPosToUV(float3 view_pos)
    {
        float proj_scale_y = 1.0 / tan(radians(VerticalFOV * 0.5));
        float proj_scale_x = proj_scale_y / AspectRatio;
        float2 clip_pos;
        clip_pos.x = view_pos.x * proj_scale_x / view_pos.z;
        clip_pos.y = -view_pos.y * proj_scale_y / view_pos.z;
        return clip_pos * 0.5 + 0.5;
    }

    float3 CalculateNormalFromDepth(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float center_depth = getDepth(texcoord);

        float3 center_pos = UVToViewPos(texcoord, center_depth);
        float3 right_pos = UVToViewPos(texcoord + float2(p.x, 0), getDepth(texcoord + float2(p.x, 0)));
        float3 up_pos = UVToViewPos(texcoord + float2(0, p.y), getDepth(texcoord + float2(0, p.y)));

        return normalize(cross(up_pos - center_pos, right_pos - center_pos));
    }

    float3 ApplyBumpMapping(float2 texcoord, float3 normal)
    {
        if (BumpIntensity == 0.0)
            return normal;

        float2 px = ReShade::PixelSize;

        float h_c = GetLuminance(GetColor(texcoord).rgb);
        float h_x = GetLuminance(GetColor(texcoord + float2(px.x, 0)).rgb);
        float h_y = GetLuminance(GetColor(texcoord + float2(0, px.y)).rgb);

        float2 slope = float2(h_x - h_c, h_y - h_c) * BumpIntensity * 100.0;
        
        float holeDepth = (1.0 - h_c) * BumpDepth * BumpDirection.z;

        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T = normalize(cross(up, normal));
        float3 B = cross(normal, T);
        float3 bumpedNormal = normal + (T * slope.x * BumpDirection.x + B * slope.y * BumpDirection.y + normal * holeDepth);

        return normalize(bumpedNormal);
    }

    float3 ApplyGeometryCorrection(float2 texcoord, float3 normal)
    {
        if (GeoCorrectionIntensity == 0.0)
            return normal;

        float2 px = ReShade::PixelSize;
        
        float lumCenter = GetLuminance(GetColor(texcoord).rgb);
        float lumRight = GetLuminance(GetColor(texcoord + float2(px.x, 0)).rgb);
        float lumDown = GetLuminance(GetColor(texcoord + float2(0, px.y)).rgb);

        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));

        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }

    float3 GetSurfaceNormal(float2 texcoord)
    {
        float3 normal = CalculateNormalFromDepth(texcoord);

        if (bSmoothNormals)
        {
            float2 px = ReShade::PixelSize;
            float2 offset = px * 7.5 * (1.0 - getDepth(texcoord));

            float3 n_t = CalculateNormalFromDepth(texcoord - float2(0, offset.y));
            float3 n_b = CalculateNormalFromDepth(texcoord + float2(0, offset.y));
            float3 n_l = CalculateNormalFromDepth(texcoord - float2(offset.x, 0));
            float3 n_r = CalculateNormalFromDepth(texcoord + float2(offset.x, 0));

            float w_t = smoothstep(1, 0, distance(normal, n_t) * 1.5) * 2;
            float w_b = smoothstep(1, 0, distance(normal, n_b) * 1.5) * 2;
            float w_l = smoothstep(1, 0, distance(normal, n_l) * 1.5) * 2;
            float w_r = smoothstep(1, 0, distance(normal, n_r) * 1.5) * 2;

            float4 weightedNormal = float4(normal, 1.0)
                                      + float4(n_t * w_t, w_t)
                                      + float4(n_b * w_b, w_b)
                                      + float4(n_l * w_l, w_l)
                                      + float4(n_r * w_r, w_r);

            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }

        normal = ApplyBumpMapping(texcoord, normal);
        normal = ApplyGeometryCorrection(texcoord, normal);

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

    bool TraceRay(in float3 rayOrigin, in float3 rayDir, out float3 hitViewPos, out float2 uvHit, bool enableRefinement)
    {
        float stepSize = MIN_STEP_SIZE;
        float totalDist = 0.0;
        float3 prevPos = rayOrigin;

        [loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + rayDir * stepSize;
            totalDist += stepSize;

            float2 uvCurr = ViewPosToUV(currPos);
            if ((any(uvCurr < 0.0) || any(uvCurr > 1.0)) || totalDist > MaxTraceDistance)
                break;

            float sceneDepth = getDepth(uvCurr).x;
            float thickness = abs(currPos.z - sceneDepth);

            if (currPos.z < sceneDepth || thickness > ThicknessThreshold)
            {
                prevPos = currPos;
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
                    float midDepth = getDepth(ViewPosToUV(mid)).x;

                    if (mid.z >= midDepth)
                        hi = mid;
                    else
                        lo = mid;
                }
                hitViewPos = hi;
            }
            else
            {
                hitViewPos = currPos;
            }
            
            uvHit = ViewPosToUV(hitViewPos);
            return true;
        }

        return false;
    }

    struct PixelData
    {
        float Depth;
        float3 ViewPos;
        float3 ViewDir;
        float3 Normal;
    };

    PixelData SetupPixelData(float2 uv_screen)
    {
        PixelData pd;
        pd.Depth = getDepth(uv_screen).x;
        pd.ViewPos = UVToViewPos(uv_screen, pd.Depth);
        pd.ViewDir = -normalize(pd.ViewPos);
        pd.Normal = float3(0, 0, 1);
        return pd;
    }

//--------------------|
// :: Pixel Shaders ::|
//--------------------|
    float4 PS_GenerateNormals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            discard;
        }
        float2 screen_uv = uv / RenderScale;
        float3 normal = GetSurfaceNormal(screen_uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float4 PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 0;
        }
        
        float2 screen_uv = uv / RenderScale;
        PixelData pd = SetupPixelData(screen_uv);
        pd.Normal = normalize(SampleNormalFromTexture(uv));

        float4 reflection = 0;
        
        float3 eyeDir = -pd.ViewDir;
        float3 rayDir = normalize(reflect(eyeDir, pd.Normal));

        float3 hitViewPos;
        float2 uvHit;
        bool hit = TraceRay(pd.ViewPos, rayDir, hitViewPos, uvHit, true);

        if (hit)
        {
            float3 hitNormal = SampleNormalFromTexture(uvHit * RenderScale);
            if (distance(hitNormal, pd.Normal) >= NormalBias)
            {
                float distFactor = saturate(1.0 - length(hitViewPos - pd.ViewPos) / MaxTraceDistance);
                reflection.rgb = GetColor(uvHit).rgb * distFactor;
            }
        }
        else // Fallback
        {
            float adaptiveDist = pd.Depth * 1.2 + 0.003;
            float3 fbViewPos = pd.ViewPos + rayDir * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos));
            bool isSky = getDepth(uvFb) >= 1.0;
            float3 fbColor = GetColor(uvFb).rgb;

            if (isSky)
            {
                reflection.rgb = fbColor;
            }
            else
            {
                float depthFactor = saturate(1.0 - pd.Depth / MaxTraceDistance);
                float vertical_fade = 1.0 - screen_uv.y;
                reflection.rgb = fbColor * depthFactor * vertical_fade;
            }
        }
        
        float fresnel = pow(1.0 - saturate(dot(eyeDir, pd.Normal)), 3.0);
        float angleWeight = pow(saturate(dot(-pd.ViewDir, rayDir)), 2.0);
        float fadeRange = max(FadeEnd - FadeStart, 0.001);
        float depthFade = saturate((FadeEnd - pd.Depth) / fadeRange);
        depthFade *= depthFade;

        reflection.rgb *= fresnel * angleWeight * depthFade;
        reflection.a = pd.Depth;

        return reflection;
    }
    
    float4 PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 sample_uv = uv * RenderScale;
        float3 currentSpec = tex2D(sSSR, sample_uv).rgb;

        float2 motion = GetFilteredMotionVector(uv);
        float2 reprojected_uv = uv + motion;

        float currentDepth = getDepth(uv);
        float historyDepth = getDepth(reprojected_uv);
        
        bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) &&
                                  FRAME_COUNT > 1 &&
                                  abs(historyDepth - currentDepth) < 0.01;

        float3 blendedSpec = currentSpec;

        if (EnableTemporal && validHistory)
        {
            float3 historySpec = tex2D(sHISTORY, reprojected_uv).rgb;
            
            float3 minBox = RGBToYCoCg(currentSpec);
            float3 maxBox = minBox;

            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    float2 neighbor_uv = uv + float2(x, y) * ReShade::PixelSize;
                    float3 neighborSpec = RGBToYCoCg(tex2D(sSSR, neighbor_uv * RenderScale).rgb);
                    minBox = min(minBox, neighborSpec);
                    maxBox = max(maxBox, neighborSpec);
                }
            }
            
            float3 clampedHistorySpec = clamp(RGBToYCoCg(historySpec), minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, AccumFramesSG);

            blendedSpec = YCoCgToRGB(lerp(clampedHistorySpec, RGBToYCoCg(currentSpec), alpha));
        }
        
        return float4(blendedSpec, 1.0);
    }

    float4 PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sTEMP, uv);
    }
    
    float4 PS_Output(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (ViewMode != 0)
        {
            float depth = getDepth(texcoord);
            float3 normal = GetSurfaceNormal(texcoord);
            float2 motion = GetFilteredMotionVector(texcoord);

            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / 6.283185) + 0.5, 1.0, saturate(velocity));
                    return float4(HSVToRGB(hsv), 1.0);
                
                case 2: // Final Reflection
                    return float4(tex2D(sTEMP, texcoord).rgb, 1.0);
                
                case 3: // Normals
                    return float4(normal * 0.5 + 0.5, 1.0);
                
                case 4: // Depth
                    return float4(depth.xxx, 1.0);

                case 5: // Raw Reflection
                    return float4(tex2D(sSSR, texcoord * RenderScale).rgb, 1.0);
            }
        }

        float4 originalColor = GetColor(texcoord);
        float3 specularGI = tex2D(sTEMP, texcoord).rgb;
        
        // Colors and ToneMapping
        float Saturation = Adjustments.r;
        float Exposure = Adjustments.g;
        float Contrast = Adjustments.b;

        specularGI *= Exposure;

        if (AssumeSRGB)
            specularGI = LinearizeSRGB(specularGI);
        if (EnableACES)
            specularGI = Apply_ACES(specularGI);
    
        float luminance = GetLuminance(specularGI);
        specularGI = lerp(luminance.xxx, specularGI, Saturation);
        specularGI = (specularGI - 0.5) * Contrast + 0.5;

        specularGI *= SPIntensity;

        switch (BlendMode)
        {
            case 0: // Additive
                return float4(originalColor.rgb + specularGI, originalColor.a);
            
            case 1: // Screen 
                return float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - specularGI), originalColor.a);
            
            case 2: // Alpha Blend
                float giLuminance = GetLuminance(saturate(specularGI));
                return float4(lerp(originalColor.rgb, specularGI, giLuminance), originalColor.a);
        }

        return originalColor;
    }

    technique SSRT < ui_tooltip = "Screen Space Ray Traced Reflections."; >
    {
        pass GenerateNormals
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GenerateNormals;
            RenderTarget = normalTex;
        }
        pass TraceReflections
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TraceReflections;
            RenderTarget = SSR;
        }
        pass Accumulate
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Accumulate;
            RenderTarget = TEMP;
        }
        pass UpdateHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HISTORY;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
