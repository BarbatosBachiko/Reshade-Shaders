/*------------------.
| :: Description :: |
'-------------------/

    NeoGI 
    Version 1.4
    Author: Barbatos Bachiko
    License: MIT
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE

    About: Simple Indirect lighting.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.4
    * New Algoritm
*/

#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define _NORMAL_EDGE_THRESHOLD 0.1f 
#define _SKY_FADE_RANGE 0.015f  
#define MVErrorTolerance 0.96f
#define SkyDepth 0.99f
#define MAX_Frames 128
#define PI 3.14159265359
#define HALF_PI 1.57079632679
#define TAU 6.28318530718
#define GOLDEN_ANGLE 2.39996323

#ifndef RES_SCALE
#define RES_SCALE 0.8
#endif
#define RES_WIDTH (ReShade::ScreenSize.x * RES_SCALE)
#define RES_HEIGHT (ReShade::ScreenSize.y * RES_SCALE)

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LINEAR MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define getDepth(coords) (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 
#define AspectRatio (BUFFER_WIDTH / (float)BUFFER_HEIGHT)
#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)

uniform float IndirectIntensity <
    ui_category = "Indirect Lighting";
    ui_type = "drag";
    ui_label = "Intensity";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 2.0;

uniform float IndirectSaturation <
    ui_category = "Indirect Lighting";
    ui_type = "drag";
    ui_label = "Saturation";
    ui_tooltip = "Controls the saturation of the bounced light.";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float SamplingRadius <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sampling Radius";
    ui_min = 0.1; ui_max = 15.0; ui_step = 0.1;
> = 4.0;

uniform float SamplingRadius2 <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sampling Radius 2";
    ui_min = 0.1; ui_max = 30.0; ui_step = 0.1;
> = 8.0;

static const float MultiScaleWeight = 0.3;
    
uniform int SamplingDirections <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sampling Directions";
    ui_min = 2; ui_max = 32; ui_step = 1;
> = 12;
    
uniform int SamplingSteps <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sampling Steps";
    ui_min = 2; ui_max = 24; ui_step = 1;
> = 6;

uniform float FalloffPower <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Falloff Power";
    ui_min = 0.5; ui_max = 4.0; ui_step = 0.1;
> = 2.0;
    
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
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float FadePower <
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade Power";
    ui_min = 0.5; ui_max = 4.0; ui_step = 0.1;
> = 1.0;
    
uniform bool EnableTemporal <
    ui_category = "Temporal";
    ui_type = "checkbox";
    ui_label = "Temporal Filtering";
> = true;

uniform float AccumFramesGI <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "Temporal";
    ui_min = 1.0; ui_max = 32.0; ui_step = 1.0;
> = 1.0;

uniform bool AdaptiveQuality <
    ui_category = "Quality";
    ui_type = "checkbox";
    ui_label = "Adaptive Quality";
> = true;

uniform float Fov <
    ui_category = "Depth/Normals";
    ui_type = "drag";
    ui_label = "Field of View (Vertical)";
    ui_min = 10.0; ui_max = 120.0; ui_step = 1.0;
> = 56.0;

uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;
    
uniform float DepthThreshold <
    ui_type = "slider";
    ui_category = "Depth/Normals";
    ui_label = "Sky Threshold";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.985;

uniform bool bSmoothNormals <
    ui_category = "Depth/Normals";
    ui_label = "Smooth Normals";
> = false;

uniform bool bEnhancedNormals <
    ui_category = "Depth/Normals";
    ui_label = "NormalMap 2";
    ui_tooltip = "Use normal computation with edge detection";
> = false;

uniform float BrightnessThreshold <
    ui_category = "Masking";
    ui_type = "slider";
    ui_label = "Brightness Threshold";
    ui_tooltip = "Reduce GI on bright surfaces";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 2.0;

uniform float SaturationMask <
    ui_category = "Masking";
    ui_type = "slider";
    ui_label = "Saturation Mask";
    ui_tooltip = "Reduce GI on highly saturated surfaces";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 2.0;
    
uniform int ViewMode <
    ui_category = "General";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for debugging";
    ui_items = "Normal\0GI Debug\0Depth\0Sky Debug\0Normal Debug\0";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;
uniform int RANDOM_INT < source = "random";min = 0; max = 32767; >;
uniform float TIMER < source = "timer"; >;

// Motion vector 
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

namespace CNGI
{
    texture2D GI
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    texture2D GI_TEMP
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    texture2D GI_HISTORY
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };

    sampler2D sGI
    {
        Texture = GI;
        SRGBTexture = false;S_LINEAR
    };
    sampler2D sGI_TEMP
    {
        Texture = GI_TEMP;
        SRGBTexture = false;S_LINEAR
    };
    sampler2D sGI_HISTORY
    {
        Texture = GI_HISTORY;
        SRGBTexture = false;S_LINEAR
    };
    
    texture normalTex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = normalTex;S_LINEAR
    };
    
    /*----------------.
    | :: Functions :: |
    '----------------*/
    
    float lum(float3 color)
    {
        return dot(color, float3(0.299, 0.587, 0.114));
    }
    float saturation(float3 color)
    {
        float maxC = max(max(color.r, color.g), color.b);
        float minC = min(min(color.r, color.g), color.b);
        return (maxC - minC) / (maxC + 1e-6);
    }
    float interleavedGradientNoise(float2 coord)
    {
        float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
        return frac(magic.z * frac(dot(coord, magic.xy)));
    }
    float3 UVtoPos(float2 texcoord, float depth)
    {
        float3 scrncoord = float3(texcoord.xy * 2.0 - 1.0, depth * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord.xy *= tan(radians(Fov * 0.5));
        return scrncoord.xyz;
    }
    float3 UVtoPos(float2 texcoord)
    {
        return UVtoPos(texcoord, getDepth(texcoord));
    }
    float3 computeNormalEnhanced(float2 texcoord)
    {
        float2 p = pix;
        float centerDepth = getDepth(texcoord);
        
        float depthU = getDepth(texcoord + float2(0, p.y));
        float depthD = getDepth(texcoord - float2(0, p.y));
        float depthL = getDepth(texcoord - float2(p.x, 0));
        float depthR = getDepth(texcoord + float2(p.x, 0));
        
        float4 depthDiffs = abs(float4(depthU, depthD, depthL, depthR) - centerDepth);
        float edgeThreshold = _NORMAL_EDGE_THRESHOLD * centerDepth;
        bool4 isEdge = depthDiffs > edgeThreshold;
        
        float2 adaptiveP = p * (any(isEdge) ? 2.0 : 1.0);
        
        float3 positions[4] =
        {
            UVtoPos(texcoord + float2(0, adaptiveP.y)), // U
            UVtoPos(texcoord - float2(0, adaptiveP.y)), // D
            UVtoPos(texcoord - float2(adaptiveP.x, 0)), // L
            UVtoPos(texcoord + float2(adaptiveP.x, 0)) // R
        };
        
        float3 center = UVtoPos(texcoord);
        
        float3 gradV = isEdge.x ? (center - positions[1]) : (positions[0] - center);
        float3 gradH = isEdge.z ? (center - positions[3]) : (positions[2] - center);
        
        if (isEdge.y && !isEdge.x)
            gradV = center - positions[1];
        if (isEdge.w && !isEdge.z)
            gradH = center - positions[3];
        
        return normalize(cross(gradV, gradH));
    }
    float3 computeNormal(float2 texcoord)
    {
        if (bEnhancedNormals)
            return computeNormalEnhanced(texcoord);
        
        float2 p = pix;
        float3 u = UVtoPos(texcoord + float2(0, p.y));
        float3 d = UVtoPos(texcoord - float2(0, p.y));
        float3 l = UVtoPos(texcoord - float2(p.x, 0));
        float3 r = UVtoPos(texcoord + float2(p.x, 0));
        
        float3 c = UVtoPos(texcoord);
        float3 v = u - c;
        float3 h = r - c;
        
        float2 p2 = p * 2.0;
        if (abs(getDepth(texcoord - float2(0, p2.y)) - c.z) < abs(getDepth(texcoord + float2(0, p2.y)) - c.z))
            v = c - d;
        if (abs(getDepth(texcoord - float2(p2.x, 0)) - c.z) < abs(getDepth(texcoord + float2(p2.x, 0)) - c.z))
            h = c - l;
        
        return normalize(cross(v, h));
    }
    float3 GetNormalFromDepth(float2 texcoord)
    {
        float3 normal = computeNormal(texcoord);
        
        if (bSmoothNormals)
        {
            float depth = getDepth(texcoord);
            float2 offset = pix * lerp(3.0, 12.0, 1.0 - depth);
            
            float3 normals[4] =
            {
                computeNormal(texcoord + float2(0, offset.y)),
                computeNormal(texcoord - float2(0, offset.y)),
                computeNormal(texcoord - float2(offset.x, 0)),
                computeNormal(texcoord + float2(offset.x, 0))
            };
            
            float4 weights = 1.0;
            
            [unroll]
            for (int i = 0; i < 4; i++)
            {
                float similarity = dot(normal, normals[i]);
                weights[i] = pow(saturate(similarity), 4.0);
            }
            
            float totalWeight = dot(weights, 1.0) + 1.0;
            float3 smoothNormal = normal + normals[0] * weights[0] + normals[1] * weights[1] + normals[2] * weights[2] + normals[3] * weights[3];
            
            normal = normalize(smoothNormal / totalWeight);
        }
        
        return normal;
    }
    float3 getNormal(float2 coords)
    {
        return normalize(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz * 2.0 - 1.0);
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

    float3 ComputeGI(float2 texcoord, float radius, int directions, int steps)
    {
        if (directions == 0 || steps == 0)
            return 0.0;

        float centerDepth = getDepth(texcoord);
        float3 centerPos = UVtoPos(texcoord, centerDepth);
        float3 centerNormal = getNormal(texcoord);
        
        float3 totalIndirectLight = 0.0;
        float stepSize = radius * pix.x;
        
        float jitter = interleavedGradientNoise(texcoord * BUFFER_WIDTH + FRAME_COUNT);
        
        [loop]
        for (int dir = 0; dir < directions; dir++)
        {
            float angle = (float(dir) + jitter) / float(directions) * TAU;
            float2 direction = float2(cos(angle), sin(angle));
            
            float3 directionLight = 0.0;
            float bestHorizon = 0.0;
            
            [loop]
            for (int step = 1; step <= steps; step++)
            {
                float stepDistance = stepSize * step;
                float2 sampleUV = texcoord + direction * stepDistance;
                sampleUV = saturate(sampleUV);
                
                float sampleDepth = getDepth(sampleUV);
                float3 samplePos = UVtoPos(sampleUV, sampleDepth);
                float3 sampleVector = samplePos - centerPos;
                float sampleDistance = length(sampleVector);
                
                if (sampleDistance > radius)
                    break;
                
                sampleVector = normalize(sampleVector);
                float horizon = dot(sampleVector, centerNormal);
                
                if (horizon > bestHorizon)
                {
                    bestHorizon = horizon;
                    float3 sampleColor = GetColor(sampleUV).rgb;
                    float weight = 1.0 - pow(sampleDistance / radius, FalloffPower);
                    directionLight = sampleColor * horizon * weight;
                }
            }
            
            totalIndirectLight += directionLight;
        }
        
        float3 indirectLight = totalIndirectLight / float(directions);
        
        return indirectLight;
    }

    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetNormalFromDepth(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }
    
    float4 PS_GI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = getDepth(uv);
        float3 originalColor = GetColor(uv).rgb;
        
        float brightness = lum(originalColor);
        float sat = saturation(originalColor);
        
        float brightnessFactor = saturate(1.0 - smoothstep(BrightnessThreshold - 0.2, BrightnessThreshold + 0.2, brightness));
        float saturationFactor = saturate(1.0 - smoothstep(SaturationMask - 0.2, SaturationMask + 0.2, sat));
        float maskFactor = min(brightnessFactor, saturationFactor);
        
        float2 motion = GetMotionVector(uv);
        float motionAmount = length(motion);
        
        int adaptiveDirections = SamplingDirections;
        int adaptiveSteps = SamplingSteps;
        
        if (AdaptiveQuality && motionAmount > 0.01)
        {
            adaptiveDirections = max(4, SamplingDirections / 2);
            adaptiveSteps = max(2, SamplingSteps / 2);
        }
        
        float3 result1 = ComputeGI(uv, SamplingRadius, adaptiveDirections, adaptiveSteps);
        float3 result2 = ComputeGI(uv, SamplingRadius2, max(1, adaptiveDirections / 2), max(1, adaptiveSteps / 2));
        
        float3 indirectLight = lerp(result1, result2, MultiScaleWeight);
        
        indirectLight *= maskFactor;
        
        float fade = 1.0;
        if (depthValue > FadeStart)
        {
            fade = saturate((FadeEnd - depthValue) / (FadeEnd - FadeStart));
            fade = pow(fade, FadePower);
        }
        indirectLight *= fade;
        
        float skyMask = 1.0;
        if (depthValue > DepthThreshold - _SKY_FADE_RANGE)
        {
            skyMask = saturate((DepthThreshold - depthValue) / _SKY_FADE_RANGE);
        }
        indirectLight *= skyMask;
        
        return float4(indirectLight, 1.0);
    }
    
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 motion = GetMotionVector(uv);
        float2 reprojectedUV = uv + motion;
    
        float4 currentData = tex2Dlod(sGI, float4(uv, 0, 0));
        float4 historyData = tex2Dlod(sGI_HISTORY, float4(reprojectedUV, 0, 0));
    
        float depth = getDepth(uv);
        bool validHistory = (depth < SkyDepth) && all(saturate(reprojectedUV) == reprojectedUV) && FRAME_COUNT > 1;
    
        float4 blendedData = currentData;
    
        if (EnableTemporal && validHistory)
        {
            float alpha = (AccumFramesGI > 0) ? (1.0 / AccumFramesGI) : 0.1;
            blendedData = lerp(historyData, currentData, alpha);
        }
        
        return blendedData;
    }

    float4 PS_SaveHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return EnableTemporal
            ? tex2Dlod(sGI_TEMP, float4(uv, 0, 0))
            : tex2Dlod(sGI, float4(uv, 0, 0));
    }
    
    float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);
        float depthValue = getDepth(uv);
        
        float3 indirectLight;

        if (EnableTemporal)
        {
            indirectLight = tex2Dlod(sGI_TEMP, float4(uv, 0, 0)).rgb;
        }
        else
        {
            indirectLight = tex2Dlod(sGI, float4(uv, 0, 0)).rgb;
        }
            
        switch (ViewMode)
        {
            case 0: // Normal
            {
                    if (depthValue >= DepthThreshold)
                        return originalColor;
                
                    float gi_lum = lum(indirectLight);
                    float3 il_term = lerp(gi_lum.xxx, indirectLight, IndirectSaturation) * IndirectIntensity;

                    float3 finalColor = originalColor.rgb + il_term;
                
                    return float4(finalColor, originalColor.a);
                }
            case 1: // GI Debug
                return float4(indirectLight * 2.0, 1.0);

            case 2: // Depth
                return float4(depthValue, depthValue, depthValue, 1.0);

            case 3: // Sky Debug
                return (depthValue >= DepthThreshold)
                    ? float4(1.0, 0.0, 0.0, 1.0)
                    : float4(depthValue, depthValue, depthValue, 1.0);

            case 4: // Normal Debug
            {
                    float3 normal = getNormal(uv);
                    return float4(normal * 0.5 + 0.5, 1.0);
                }
        }

        return originalColor;
    }

    /*-------------------.
    | :: Techniques ::   |
    '-------------------*/
    
    technique NeoGI
    {
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
        }
        pass GI_Pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GI;
            RenderTarget = GI;
            ClearRenderTargets = true;
        }
        pass TemporalGIPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget = GI_TEMP;
            ClearRenderTargets = true;
        }
        pass SaveHistoryGIPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistory;
            RenderTarget = GI_HISTORY;
        }
        pass Composite_Pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Composite;
        }
    }
}
