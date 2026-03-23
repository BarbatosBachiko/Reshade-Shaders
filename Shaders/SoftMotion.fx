/*-------------------------------------------------|
| ::                SoftMotion                   ::|
'--------------------------------------------------|
| Version: 1.3                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: >EMULATES< frame interpolation or   |
| extrapolation by blending or projecting frames   |
| using motion vectors.                            |
'--------------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include ".\BaBa_Includes\BaBa_MV.fxh"

uniform int Mode <
    __UNIFORM_COMBO_INT1
    ui_items = "Interpolation\0Extrapolation\0";
    ui_label = "Mode";
    ui_tooltip = "Interpolation: Blends current frame with motion-compensated previous frame.\nExtrapolation: Projects current frame forward to predict the next frame.";
> = 0;

uniform float BlendAmount <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0;
    ui_max = 0.99;
    ui_step = 0.01;
    ui_label = "Blend Amount";
    ui_tooltip = "Controls the mix between the current frame and the alternate.\n0.0 = No blending\n0.5 = 50/50 mix\n0.99 = Alternate frame dominant";
> = 0.5;

uniform float MotionScale <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.01;
    ui_label = "Motion Vector Scale";
    ui_tooltip = "Adjusts the influence of motion vectors.";
> = 1.0;

uniform int DebugView <
    __UNIFORM_COMBO_INT1
    ui_items = "None\0MotionVectors\0Confidence\0";
    ui_label = "Debug View";
> = 0;

//----------------|
// :: Textures  ::|
//----------------|

texture HistoryTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler sHistoryTex { Texture = HistoryTex; };

//----------------|
// :: Functions ::|
//----------------|

void PS_FrameBlend(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    float2 motion = MV_GetVelocity(uv);

    // Motion Vectors
    if (DebugView == 1) 
    {
        outColor = float4(motion * 50.0 + 0.5, 0.0, 1.0);
        return;
    }

    float3 currColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb;
    float conf = MV_GetConfidence(uv);

    if (DebugView == 2)
    {
        outColor = float4(lerp(float3(1,0,0), float3(0,1,0), conf), 1.0);
        return;
    }

    float2 offsetUV = uv + (motion * MotionScale);
    float3 altColor;
    
    if (Mode == 0) // Interpolation
        altColor = tex2D(sHistoryTex, offsetUV).rgb;
    else // Extrapolation
        altColor = tex2Dlod(ReShade::BackBuffer, float4(offsetUV, 0, 0)).rgb;

    outColor = float4(lerp(currColor, altColor, BlendAmount * conf), 1.0);
}

void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    outColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0));
}

technique BaBa_SoftMotion <
    ui_label = "BaBa: Soft Motion";
    ui_tooltip = "Blending or projecting frames using motion vectors. It doesn't generate new frames or predict the future. It's more of a motion blur based on motion vectors with temporal blending.";
>
{
    pass { VertexShader = PostProcessVS; PixelShader = PS_FrameBlend; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_UpdateHistory; RenderTarget = HistoryTex; }
}