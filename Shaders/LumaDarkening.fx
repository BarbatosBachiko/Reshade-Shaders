/*------------------------------------------------------------------------------------------------|
| ::                                   Luma Darkening                                          :: |
'-------------------------------------------------------------------------------------------------|
| Version 1.1                                                                                     |
| Author: Barbatos, Based on the paper by Thomas Luft, Carsten Colditz, and Oliver Deussen (2006).|
| License: MIT                                                                                    |
| About: Enhances perceptual depth by applying unsharp masking to the Image Luminance.            |
'------------------------------------------------------------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))

/*---------.
| :: UI::  |
'---------*/

uniform int EffectMode <
    ui_type = "combo";
    ui_items = "Luma Darkening\0Foregound Halos\0Omni-directional\0Artistic\0";
    ui_category = "Basic Settings";
    ui_label = "Effect Mode";
    ui_tooltip = "Select the type of spatial enhancement.\n\nLuma Darkening: Adds contact shadows to contrast edges.";
> = 0;

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_category = "Basic Settings";
    ui_label = "Intensity";
> = 0.050; 

uniform float ClampLimit <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Clamp Limit";
    ui_tooltip = "Limits the maximum intensity of the effect to prevent black crushing or white clipping.";
> = 0.7; 

uniform float Radius <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "Basic Settings";
    ui_label = "Spread Radius";
    ui_tooltip = "How far the effect spreads from object edges.";
> = 2.0; 

uniform bool AnimeMode <
    ui_category = "Basic Settings";
    ui_label = "Anime Mode";
    ui_tooltip = "Optimizes for anime/cel-shaded content by ignoring thin line details and facial features.";
> = true;

uniform float3 LumaShadowTint <
    ui_type = "color";
    ui_category = "Color Settings";
    ui_label = "Luma Shadow Tint";
    ui_tooltip = "Tints the shadows in 'Luma Darkening' mode.\nDefault is White (1.0, 1.0, 1.0) for neutral darkening.\nWorks subtractively (Red tint removes Red from shadows).";
> = float3(1.0, 1.0, 1.0);

uniform float3 HaloColor <
    ui_type = "color";
    ui_category = "Color Settings";
    ui_label = "Foreground Halo Color";
    ui_tooltip = "Sets the color of the glow in 'Foreground Halos' mode.\nDefault is White (1.0, 1.0, 1.0).";
> = float3(1.0, 1.0, 1.0);

uniform float3 NearColor <
    ui_type = "color";
    ui_category = "Artistic Settings";
    ui_label = "Darkening Color";
> = float3(1.0, 0.8, 0.2); 

uniform float3 FarColor <
    ui_type = "color";
    ui_category = "Artistic Settings";
    ui_label = "Brightening Color";
> = float3(0.0, 0.0, 0.5); 

uniform float TemporalStabilization <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 0.99; ui_step = 0.01;
    ui_category = "Advanced";
    ui_label = "Stabilization";
    ui_tooltip = "Reduces flickering on edges by blending with previous frames.";
> = 0.0;

uniform float NoiseThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001; 
    ui_category = "Advanced";
    ui_label = "Noise Threshold";
    ui_tooltip = "Ignores micro-luma changes to prevent texture flickering on flat surfaces.";
> = 0.025; 

uniform float EdgeThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Advanced";
    ui_label = "Edge Strength Threshold";
    ui_tooltip = "Only apply effect to edges stronger than this value. Higher = fewer thin lines affected.\nUseful for anime to avoid affecting facial details like mouths and eyes.";
> = 0.08;

uniform float AnimeDetailProtection <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced";
    ui_label = "Anime Detail Protection";
    ui_tooltip = "Protects fine details in Anime Mode. 0 = no protection, 1 = maximum protection of thin lines.";
> = 0.5;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "None\0Spatial Importance (Delta L)\0Motion Vectors\0Edge Gradient Mask\0Confidence Map\0";
    ui_category = "Debug";
    ui_label = "Debug View";
> = 0;

/*---------------.
| :: Textures :: |
'---------------*/

texture TexBlurH
{
    Width = BUFFER_WIDTH / 2;
    Height = BUFFER_HEIGHT / 2;
    Format = R16F;
};
sampler SamplerBlurH
{
    Texture = TexBlurH;
};

texture TexBlurV
{
    Width = BUFFER_WIDTH / 2;
    Height = BUFFER_HEIGHT / 2;
    Format = R16F;
};
sampler SamplerBlurV
{
    Texture = TexBlurV;
};

texture TexDeltaD
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler SamplerDeltaD
{
    Texture = TexDeltaD;
};

texture TexFilteredDeltaD
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler SamplerFilteredDeltaD
{
    Texture = TexFilteredDeltaD;
};

texture TexHistory
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler SamplerHistory
{
    Texture = TexHistory;
};

texture TexPrevLuma
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R8;
};
sampler SamplerPrevLuma
{
    Texture = TexPrevLuma;
};

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT; MinFilter=POINT; MipFilter=POINT; AddressU=Clamp; AddressV=Clamp; };
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

/*----------------.
| :: Functions :: |
'----------------*/

float GetGaussianWeight(float x, float sigma)
{
    return 0.39894228 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

float2 GetMotion(float2 texcoord)
{
#if USE_MARTY_LAUNCHPAD_MOTION
    return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
#elif USE_VORT_MOTION
    return GetLod(sMotVectTexVort, texcoord).rg;
#else
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
#endif
}

float GetInput(float2 uv)
{
    return dot(tex2D(ReShade::BackBuffer, uv).rgb, float3(0.2126, 0.7152, 0.0722));
}

float GetLocalGradient(float2 uv)
{
    float2 offsetX = float2(BUFFER_PIXEL_SIZE.x, 0.0);
    float2 offsetY = float2(0.0, BUFFER_PIXEL_SIZE.y);
    
    float gx = GetInput(uv + offsetX) - GetInput(uv - offsetX);
    float gy = GetInput(uv + offsetY) - GetInput(uv - offsetY);
    
    return length(float2(gx, gy));
}

// Based on LumaFlow.fx from LumeniteFX CC-BY-NC-4.0
float Confidence(float2 uv, float2 velocity)
{
    float2 prev_uv = uv + velocity;
    
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
        return 0.0;

    float curr_luma = GetInput(uv);
    float prev_luma = tex2D(SamplerPrevLuma, prev_uv).r;
    float luma_error = abs(curr_luma - prev_luma);

    float flow_magnitude = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float subpixel_threshold = 1.0;
    
    if (flow_magnitude <= subpixel_threshold)
        return 1.0;

    float2 destination_velocity = GetMotion(prev_uv);
    float2 diff = velocity - destination_velocity;
    float error = length(diff);
    float normalized_error = error / (length(velocity) + 1e-6);

    float motion_penalty = flow_magnitude;
    float length_conf = rcp(motion_penalty * 0.05 + 1.0);
    float consistency_conf = rcp(normalized_error + 1.0);
    float photometric_conf = exp(-luma_error * 5.0);

    return (consistency_conf * length_conf * photometric_conf);
}

void PS_BlurH(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outLuma : SV_Target)
{
    float totalWeight = 0.0;
    float result = 0.0;
    float2 pixelSize = BUFFER_PIXEL_SIZE * 2.0;
    float sigma = max(0.1, Radius);
    int radius = clamp(int(ceil(sigma * 3.0)), 1, 20);

    for (int x = -radius; x <= radius; x++)
    {
        float weight = GetGaussianWeight(float(x), sigma);
        result += GetInput(uv + float2(float(x) * pixelSize.x, 0.0)) * weight;
        totalWeight += weight;
    }
    outLuma = result / totalWeight;
}

void PS_CalcDeltaD(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outDeltaL : SV_Target)
{
    float luma = tex2D(SamplerBlurH, uv).r;
    
    float totalWeight = 0.0;
    float result = 0.0;
    float2 pixelSize = BUFFER_PIXEL_SIZE * 2.0;
    float sigma = max(0.1, Radius);
    int radius = clamp(int(ceil(sigma * 3.0)), 1, 20);

    for (int y = -radius; y <= radius; y++)
    {
        float weight = GetGaussianWeight(float(y), sigma);
        result += tex2D(SamplerBlurH, uv + float2(0.0, float(y) * pixelSize.y)).r * weight;
        totalWeight += weight;
    }
    luma = result / totalWeight;
    
    float originalLuma = GetInput(uv);
    float deltaL = luma - originalLuma;
    if (abs(deltaL) < NoiseThreshold)
        deltaL = 0.0;

    outDeltaL = deltaL;
}

void PS_TemporalFilter(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outFiltered : SV_Target)
{
    float currentDeltaL = tex2D(SamplerDeltaD, uv).r;
    
    float2 motion = GetMotion(uv);
    float2 prevUV = uv + motion;

    float conf = Confidence(uv, motion);
    
    // Reprojection
    float historyDeltaL = tex2D(SamplerHistory, prevUV).r;

    float blend = TemporalStabilization * conf;
    
    if (any(prevUV < 0.0) || any(prevUV > 1.0))
        blend = 0.0;

    outFiltered = lerp(currentDeltaL, historyDeltaL, blend);
}

void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outHistory : SV_Target)
{
    outHistory = tex2D(SamplerFilteredDeltaD, uv).r;
}

void PS_SaveLuma(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outLuma : SV_Target)
{
    outLuma = GetInput(uv);
}

void PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD, out float3 outColor : SV_Target)
{
    float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
    float deltaL = tex2D(SamplerFilteredDeltaD, uv).r;
    float gradient = GetLocalGradient(uv);
    
    // Debug Views
    if (DebugView == 1) // View Spatial 
    {
        float val = deltaL * Intensity * 100.0;
        float3 debugColor = float3(0.5, 0.5, 0.5);
        if (deltaL > 0.0)
            debugColor = lerp(debugColor, float3(1.0, 0.0, 0.0), saturate(val));
        else
            debugColor = lerp(debugColor, float3(0.0, 0.0, 1.0), saturate(-val));
        outColor = debugColor;
        return;
    }
    else if (DebugView == 2) // View Motion
    {
        float2 m = GetMotion(uv) * 100.0;
        outColor = float3(abs(m.x), abs(m.y), 0.0);
        return;
    }
    else if (DebugView == 3) // View Edge Gradient Mask
    {
        float normalizedGradient = saturate(gradient * 10.0);
        float mask = 1.0;
        if (AnimeMode)
        {
            float protection = smoothstep(0.0, EdgeThreshold, gradient);
            mask = lerp(1.0, 1.0 - protection, AnimeDetailProtection);
            outColor = float3(1.0 - mask, mask, 0.0);
        }
        else
        {
            outColor = float3(normalizedGradient, normalizedGradient, normalizedGradient);
        }
        return;
    }
    else if (DebugView == 4) // View Confidence Map
    {
        float2 motion = GetMotion(uv);
        float conf = Confidence(uv, motion);
        outColor = float3(conf, conf, conf);
        return;
    }

    float val = deltaL * Intensity * 100.0;
    
    if (AnimeMode && abs(val) > 0.0001)
    {
        float edgeStrength = gradient / max(EdgeThreshold, 0.001);
        float protectionAmount = 1.0 - smoothstep(0.0, 1.0, edgeStrength);
        float mask = lerp(1.0, 1.0 - protectionAmount, AnimeDetailProtection);
        
        val *= mask;
    }
    
    val = clamp(val, -ClampLimit, ClampLimit);

    if (EffectMode == 0) // Luma Darkening
    {
        float importance = min(0.0, val);
        outColor = saturate(color + importance * LumaShadowTint);
    }
    else if (EffectMode == 1) // Foreground Halos
    {
        float importance = max(0.0, val);
        outColor = saturate(color + importance * HaloColor);
    }
    else if (EffectMode == 2) // Omni-directional
    {
        outColor = saturate(color + val);
    }
    else if (EffectMode == 3) // Artistic
    {
        float3 targetColor = (val > 0.0) ? NearColor : FarColor;
        float factor = saturate(abs(val));
        outColor = lerp(color, targetColor, factor);
    }
    else
    {
        outColor = color;
    }
}

technique LumaDarkening<
    ui_tooltip = "Enhances local contrast using Unsharp Masking on the Luma Channel.\n";
>
{
    pass P_BlurH
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BlurH;
        RenderTarget = TexBlurH;
    }
    pass P_CalcDeltaD
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcDeltaD;
        RenderTarget = TexDeltaD;
    }
    pass P_Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_TemporalFilter;
        RenderTarget = TexFilteredDeltaD;
    }
    pass P_History
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_UpdateHistory;
        RenderTarget = TexHistory;
    }
    pass P_SaveLuma
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveLuma;
        RenderTarget = TexPrevLuma;
    }
    pass P_Composite
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Composite;
    }
}
