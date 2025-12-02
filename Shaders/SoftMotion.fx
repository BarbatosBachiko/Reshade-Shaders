/*-------------------------------------------------|
| ::                SoftMotion                   ::|
'--------------------------------------------------|
| Version: 1.2                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: >EMULATES< frame interpolation or   |
| extrapolation by blending or projecting frames   |
| using motion vectors.                            |
'--------------------------------------------------*/
//Contains AI-assisted content.

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

uniform int Mode <
    __UNIFORM_COMBO_INT1
    ui_items = "Interpolation\0Extrapolation\0";
    ui_label = "Mode";
    ui_tooltip = "Interpolation: Blends current frame with motion-compensated previous frame.\nExtrapolation: Projects current frame forward to predict the next frame.";
> = 0;

uniform float BlendAmount <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 0.99;ui_step = 0.01;
    ui_label = "Blend Amount";
    ui_tooltip = "Controls the mix between the current frame and the alternate (previous in interpolation, projected in extrapolation).\n0.0 = No blending (current frame only)\n0.5 = 50/50 mix\n0.99 = Alternate frame dominant";
> = 0.5;

uniform float MotionScale <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 2.0;ui_step = 0.01;
    ui_label = "Motion Vector Scale";
    ui_tooltip = "Adjusts the influence of motion vectors. Can help fine-tune the effect if motion seems exaggerated or too subtle.";
> = 1.0;

uniform int DebugView <
    __UNIFORM_COMBO_INT1
    ui_items = "None\0MotionVectors\0Confidence\0";
    ui_label = "Debug View";
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
    namespace Deferred { texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; }; sampler sMotionVectorsTex { Texture = MotionVectorsTex; }; }
    #define GET_MOTION(uv) tex2Dlod(Deferred::sMotionVectorsTex, float4(uv, 0, 0)).xy
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT; MinFilter=POINT; MipFilter=POINT; AddressU=Clamp; AddressV=Clamp; };
    #define GET_MOTION(uv) tex2Dlod(sMotVectTexVort, float4(uv, 0, 0)).xy
#else
    texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sTexMotionVectorsSampler { Texture = texMotionVectors; MagFilter=POINT; MinFilter=POINT; MipFilter=POINT; AddressU=Clamp; AddressV=Clamp; };
    #define GET_MOTION(uv) tex2Dlod(sTexMotionVectorsSampler, float4(uv, 0, 0)).xy
#endif

texture HistoryTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler sHistoryTex { Texture = HistoryTex; };

//----------------|
// :: Functions ::|
//----------------|

float GetConfidence(float2 uv, float2 velocity, float curr_luma)
{
    float2 prev_uv = uv + velocity;
    
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0)) return 0.0;

    float prev_luma = dot(tex2D(sHistoryTex, prev_uv).rgb, float3(0.2126, 0.7152, 0.0722));
    float flow_mag = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));

    if (flow_mag <= 1.0) return 1.0;

    float2 diff = velocity - GET_MOTION(prev_uv); 

    float conf = rcp(flow_mag * 0.05 + 1.0);
    conf *= rcp((length(diff) / length(velocity)) + 1.0);
    conf *= exp(-abs(curr_luma - prev_luma) * 5.0);    

    return conf;
}

void PS_FrameBlend(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    float3 currColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb;
    float2 motion = GET_MOTION(uv);
    
    float conf = GetConfidence(uv, motion, dot(currColor, float3(0.2126, 0.7152, 0.0722)));
    
    float2 offsetUV = uv + (motion * MotionScale);
    float3 altColor;

    if (Mode == 0) // Interpolation
        altColor = tex2D(sHistoryTex, offsetUV).rgb;
    else // Extrapolation
        altColor = tex2Dlod(ReShade::BackBuffer, float4(offsetUV, 0, 0)).rgb;

    if (DebugView > 0)
    {
        if (DebugView == 1) // Motion Vectors
            outColor = float4(motion * 50.0 + 0.5, 0.0, 1.0); 
        else // Confidence
            outColor = float4(lerp(float3(1,0,0), float3(0,1,0), conf), 1.0);
        return;
    }

    outColor = float4(lerp(currColor, altColor, BlendAmount * conf), 1.0);
}

void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
{
    outColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0));
}

technique SoftMotion <
    ui_label = "SoftMotion";
    ui_tooltip = "Blending or projecting frames using motion vectors, It doesn't generate new frames or predict the future. It's more of a motion blur based on motion vectors with temporal blending.";
>
{
    pass { VertexShader = PostProcessVS; PixelShader = PS_FrameBlend; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_UpdateHistory; RenderTarget = HistoryTex; }
}
