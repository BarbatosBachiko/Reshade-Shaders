/*------------------------------------------------------------------------------------------------|
| ::                                   Depth Darkening                                         :: |
'-------------------------------------------------------------------------------------------------|
| Version 1.0.5                                                                                   |
| Author: Barbatos, Based on the paper by Thomas Luft, Carsten Colditz, and Oliver Deussen (2006).|
| License: MIT                                                                                    |
| About: Enhances perceptual depth by applying unsharp masking to the depth buffer.               |
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
    ui_items = "Depth Darkening\0Foregound Halos\0Omni-directional\0Artistic\0";
    ui_category = "Basic Settings";
    ui_label = "Effect Mode";
    ui_tooltip = "Select the type of spatial enhancement.\n\nDepth Darkening: Best for general usage, adds contact shadows to depth edges.";
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
> = 1.0;

uniform float Radius <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "Basic Settings";
    ui_label = "Spread Radius";
    ui_tooltip = "How far the effect spreads from object edges.";
> = 4.0;

uniform float3 NearColor <
    ui_type = "color";
    ui_category = "Artistic Settings";
    ui_label = "Near Color";
> = float3(1.0, 0.8, 0.2); 

uniform float3 FarColor <
    ui_type = "color";
    ui_category = "Artistic Settings";
    ui_label = "Far Color";
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
    ui_min = 0.0; ui_max = 0.01; ui_step = 0.0001;
    ui_category = "Advanced";
    ui_label = "Noise Threshold";
    ui_tooltip = "Ignores micro-depth changes to prevent texture flickering on flat surfaces.";
> = 0.0002;

uniform float DepthCutoff <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_category = "Advanced";
    ui_label = "Depth Cutoff";
    ui_tooltip = "Distance at which the effect is disabled (e.g., Skybox). Ignored in Color Only mode.";
> = 0.99;

uniform bool ColorOnly <
    ui_label = "Color Only Mode";
    ui_tooltip = "Uses image luminance instead of depth buffer. Useful if Depth Buffer is unavailable, not recomended";
    ui_category = "Advanced";
> = false;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "None\0Spatial Importance (Delta D)\0Motion Vectors\0Confidence Map\0";
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

// Added for Temporal Confidence Check (R16F for Depth Precision)
texture TexPrevInput
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler SamplerPrevInput
{
    Texture = TexPrevInput;
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
    if (ColorOnly)
        return dot(tex2D(ReShade::BackBuffer, uv).rgb, float3(0.2126, 0.7152, 0.0722));
    else
        return ReShade::GetLinearizedDepth(uv);
}

float Confidence(float2 uv, float2 velocity)
{
    float2 prev_uv = uv + velocity;
    
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
        return 0.0;

    float curr_val = GetInput(uv);
    float prev_val = tex2D(SamplerPrevInput, prev_uv).r;
    float val_error = abs(curr_val - prev_val);

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
    float value_conf = exp(-val_error * 5.0);

    return (consistency_conf * length_conf * value_conf);
}

void PS_BlurH(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outDepth : SV_Target)
{
    float depth = GetInput(uv);
    if (!ColorOnly && depth >= DepthCutoff)
    {
        outDepth = depth;
        return;
    }

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
    outDepth = result / totalWeight;
}

void PS_CalcDeltaD(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outDeltaD : SV_Target)
{
    float depth = tex2D(SamplerBlurH, uv).r;
    
    if (ColorOnly || depth < DepthCutoff)
    {
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
        depth = result / totalWeight; // This is G * D
    }

    float originalDepth = GetInput(uv);
    float deltaD = depth - originalDepth;

    if (!ColorOnly && originalDepth >= DepthCutoff)
        deltaD = 0.0;
    if (abs(deltaD) < NoiseThreshold)
        deltaD = 0.0;

    outDeltaD = deltaD;
}

void PS_TemporalFilter(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outFiltered : SV_Target)
{
    float currentDeltaD = tex2D(SamplerDeltaD, uv).r;
    
    float2 motion = GetMotion(uv);
    float2 prevUV = uv + motion;

    float conf = Confidence(uv, motion);

    float historyDeltaD = tex2D(SamplerHistory, prevUV).r;

    float blend = TemporalStabilization * conf;
    
    if (any(prevUV < 0.0) || any(prevUV > 1.0))
        blend = 0.0;

    outFiltered = lerp(currentDeltaD, historyDeltaD, blend);
}

void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outHistory : SV_Target)
{
    outHistory = tex2D(SamplerFilteredDeltaD, uv).r;
}

void PS_SaveInput(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outInput : SV_Target)
{
    outInput = GetInput(uv);
}

void PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD, out float3 outColor : SV_Target)
{
    float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
    float deltaD = tex2D(SamplerFilteredDeltaD, uv).r;
    
    if (DebugView == 1) // View Spatial 
    {
        float val = deltaD * Intensity * 100.0;
        float3 debugColor = float3(0.5, 0.5, 0.5);
        if (deltaD > 0.0)
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
    else if (DebugView == 3) // View Confidence Map
    {
        float2 motion = GetMotion(uv);
        float conf = Confidence(uv, motion);
        outColor = float3(conf, conf, conf);
        return;
    }

    float val = deltaD * Intensity * 100.0;
    val = clamp(val, -ClampLimit, ClampLimit);

    if (EffectMode == 0) // Depth Darkening
    {
        float importance = min(0.0, val);
        outColor = saturate(color + importance);
    }
    else if (EffectMode == 1) // Foreground Halos
    {
        float importance = max(0.0, val);
        outColor = saturate(color + importance);
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

technique DepthDarkening <
    ui_tooltip = "Enhances scene depth perception using Unsharp Masking on the Depth Buffer.\n";
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
    pass P_SaveInput
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveInput;
        RenderTarget = TexPrevInput;
    }
    pass P_Composite
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Composite;
    }
}
