/*-------------------------------------------------|
| ::                SoftMotion                   ::|
'--------------------------------------------------|
| Version: 1.1                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: >EMULATES< frame interpolation or   |
| extrapolation by blending or projecting frames   |
| using motion vectors.                            |
'--------------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//----------|
// :: UI  ::|
//----------|

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define PI 3.1415927

uniform int Mode <
    __UNIFORM_COMBO_INT1
    ui_items = "Interpolation\0Extrapolation\0";
    ui_label = "Mode";
    ui_tooltip = "Interpolation: Blends current frame with motion-compensated previous frame.\nExtrapolation: Projects current frame forward to predict the next frame.";
> = 0;

uniform float BlendAmount <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 0.99; ui_step = 0.01;
    ui_label = "Blend Amount";
    ui_tooltip = "Controls the mix between the current frame and the alternate (previous in interpolation, projected in extrapolation).\n0.0 = No blending (current frame only)\n0.5 = 50/50 mix\n0.99 = Alternate frame dominant";
> = 0.5;

uniform float MotionScale <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "Motion Vector Scale";
    ui_tooltip = "Adjusts the influence of motion vectors. Can help fine-tune the effect if motion seems exaggerated or too subtle.";
> = 1.0;

uniform bool UseLumaFlowConfidence <
    ui_label = "Use LumaFlow Confidence";
    ui_tooltip = "Need LumaFlow";
> = false;

uniform float ConfidenceScale <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "Confidence Scale";
    ui_tooltip = "Scales the LumaFlow confidence value. Values >1.0 make it more aggressive, <1.0 more conservative.";
> = 1.0;

uniform float OcclusionSensitivity <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_label = "Occlusion Sensitivity";
    ui_tooltip = "Reduces ghosting artifacts. Higher values make the blending more sensitive to color differences between frames.\nOnly used when LumaFlow Confidence is disabled.";
> = 0.4;

uniform int DebugView <
    __UNIFORM_COMBO_INT1
    ui_items = "None\0Motion Vectors\0Occlusion/Confidence Mask\0";
    ui_label = "Debug View";
    ui_tooltip = "Visualize internal data for debugging.";
> = 0;

//----------------|
// :: Textures  ::|
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
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT; MinFilter=POINT; MipFilter=POINT; AddressU=Clamp; AddressV=Clamp; };
#else
    texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sTexMotionVectorsSampler { Texture = texMotionVectors; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; AddressU = Clamp; AddressV = Clamp; };
#endif

texture tMotionConfidence { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; };
sampler sMotionConfidence { Texture = tMotionConfidence; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

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

float GetConfidence(float2 texcoord)
{
    return GetLod(sMotionConfidence, texcoord).r;
}

texture HistoryTex
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler sHistoryTex
{
    Texture = HistoryTex;
};

//----------------|
// :: Functions ::|
//----------------|

float3 HSVToRGB(float3 c)
{
    const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

//--------------------|
// :: Pixel Shaders ::|
//--------------------|

void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    outColor = GetLod(ReShade::BackBuffer, uv);
}

void PS_FrameBlend(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    float3 currentColor = GetLod(ReShade::BackBuffer, uv).rgb;
    float2 motion = GetMotion(uv) * MotionScale;
    float2 offsetUV = uv + motion;
    float3 altColor;

    if (Mode == 0) // Interpolation
    {
        altColor = tex2D(sHistoryTex, saturate(offsetUV)).rgb;
    }
    else // Extrapolation
    {
        altColor = GetLod(ReShade::BackBuffer, offsetUV).rgb;
    }

    float confidence;
    
    if (UseLumaFlowConfidence)
    {
        confidence = GetConfidence(uv) * ConfidenceScale;
        confidence = saturate(confidence);
    }
    else
    {
        float colorDiff = distance(currentColor, altColor);
        confidence = 1.0 - saturate(colorDiff * OcclusionSensitivity);
    }

    float effectiveBlend = BlendAmount * confidence;

    // Debug Views
    if (DebugView == 1) // Motion Vectors
    {
        float2 m = GetMotion(uv);
        float v_mag = length(m) * 100.0;
        float a = atan2(m.y, m.x);
        float3 hsv_color = HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, 1.0));
        float3 grey_bg = float3(0.5, 0.5, 0.5);
        float3 final_color = lerp(grey_bg, hsv_color, saturate(v_mag));
        outColor = float4(final_color, 1.0);
        return;
    }
    else if (DebugView == 2) // Occlusion/Confidence Mask
    {
        float3 colorA = float3(0.5, 0.0, 0.0);
        float3 colorB = float3(1.0, 1.0, 0.0); 
        float3 colorC = float3(0.0, 0.0, 0.1); 

        float3 finalColor;
        if (confidence < 0.5)
        {
            finalColor = lerp(colorA, colorB, confidence * 2.0);
        }
        else
        {
            finalColor = lerp(colorB, colorC, (confidence - 0.5) * 2.0);
        }
        
        outColor = float4(finalColor, 1.0);
        return;
    }
    
    float3 blendedColor = lerp(currentColor, altColor, effectiveBlend);
    outColor = float4(blendedColor, 1.0);
}

technique SoftMotion <
    ui_label = "SoftMotion";
    ui_tooltip = "Blending or projecting frames using motion vectors, It doesn't generate new frames or predict the future. It's more of a motion blur based on motion vectors with temporal blending.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_FrameBlend;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_UpdateHistory;
        RenderTarget = HistoryTex;
    }
}
