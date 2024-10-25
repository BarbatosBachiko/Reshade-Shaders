/*------------------.
| :: Description :: |
'-------------------/

    DTAA Shader (version 1.1)

    Author: BarbatosBachiko
    License: MIT
    Original DLAA method by Dmitry Andreev
    TAA method by Mortalitas from GShade

    About:
    This shader integrates DLAA (Directional Localized Anti-Aliasing) and TAA (Temporal Anti-Aliasing) techniques. It applies both spatial and temporal anti-aliasing to reduce jagged edges and improve visual quality.

    Ideas for future improvement:
    * Integrate an adjustable sharpness pass after DLAA/TAA blending.
    * Update the DLAA

    Version 1.0
    * Initial implementation combining DLAA and TAA with customizable blending and mixing rates.
    Version 1.1
    * Code optimizations and adjustment/addition of mixRate and BlendRate
*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

// DLAA Settings
uniform int View_Mode < 
     ui_type = "combo";
     ui_items = "DTAA Out\0";
     ui_label = "View Mode";
     ui_tooltip = "Select the normal view output.";
> = 0;

// Custom Settings
uniform float mixRate < 
    ui_type = "slider";
    ui_label = "Mix Rate"; 
    ui_tooltip = "Adjust the mix rate between current and past frames.";
    ui_min = 0.1; 
    ui_max = 1.0; 
    ui_step = 0.05;
> = 0.50;

uniform float blendRate < 
    ui_type = "slider";
    ui_label = "Blend Rate"; 
    ui_tooltip = "Adjust the blend between DLAA and TAA.";
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_step = 0.05;
> = 0.50;

/*---------------.
| :: Textures :: |
'---------------*/

// BackBuffer Texture
texture BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

// Past Buffers for TAA
texture PastBackBuffer
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler PBackBuffer
{
    Texture = PastBackBuffer;
};

/*----------------.
| :: Constants :: |
'----------------*/

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define iResolution float2(BUFFER_WIDTH, BUFFER_HEIGHT)

// Halton Sequence for jittering
float Halton(float i, float base)
{
    float x = 1.0f / base;
    float v = 0.0f;
    while (i > 0)
    {
        v += x * (i % base);
        i = floor(i / base);
        x /= base;
    }
    return v;
}

// Pseudo-random number generator
float random(float2 coords)
{
    float seed = dot(coords, float2(12.9898, 78.233));
    return frac(sin(seed) * 43758.5453);
}

/*----------------.
| :: Functions :: |
'----------------*/

// Apply DLAA
float4 ApplyDLAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = (centerDiffH.r + centerDiffH.g + centerDiffH.b) / 3.0;
    const float satAmountH = saturate((3.0 * LumH - 0.1) / LumH);
    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

// TAA Function
float4 TAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 XY = float2(Halton(random(texcoord * iResolution), 2), Halton(random(texcoord * iResolution), 3));
    XY *= pix;
    const float4 PastColor = tex2D(PBackBuffer, texcoord + XY);
    float3 antialiased = PastColor.xyz;
    antialiased = lerp(antialiased * antialiased, tex2D(BackBuffer, texcoord).xyz * tex2D(BackBuffer, texcoord).xyz, mixRate);
    antialiased = sqrt(antialiased);

    float3 in0 = tex2D(BackBuffer, texcoord).xyz;
    float3 in1 = tex2D(BackBuffer, texcoord + float2(+pix.x, 0.0)).xyz;
    float3 in2 = tex2D(BackBuffer, texcoord + float2(-pix.x, 0.0)).xyz;
    float3 in3 = tex2D(BackBuffer, texcoord + float2(0.0, +pix.y)).xyz;
    float3 in4 = tex2D(BackBuffer, texcoord + float2(0.0, -pix.y)).xyz;
    antialiased = clamp(antialiased, min(min(in0, in1), min(in2, in3)), max(max(in0, in1), max(in2, in3)));

    return float4(antialiased, 0);
}

// Output function
void Out(float4 position : SV_Position, float2 texcoord : TEXCOORD, out float4 color : SV_Target)
{
    const float4 center = tex2D(BackBuffer, texcoord);
    const float4 left = tex2D(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    const float4 right = tex2D(BackBuffer, texcoord + float2(1.0 * pix.x, 0));
    float4 DLAA = ApplyDLAA(center, left, right);
    float4 TAA_result = TAA(position, texcoord);

    if (View_Mode == 0)
    {
        color = lerp(DLAA, TAA_result, blendRate);
    }
}

// Vertex Shader
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DTAA
{
    pass TAA
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = Out;
    }
}
