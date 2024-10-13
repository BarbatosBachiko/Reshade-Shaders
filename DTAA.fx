/*------------------.
| :: Description :: |
'-------------------/

    DTAA Anti-Aliasing Shader (TAA + DLAA)

    Author: BarbatosBachiko

    About:
    This shader implements Temporal Anti-Aliasing (TAA) and Directionally Localized Anti-Aliasing (DLAA).

    Ideas for future improvement:
    * Further optimization for performance.
    * Additional customization options for users.
    
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.0
    * Initial implementation of TAA and DLAA.

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

/*---------------.
| :: Textures :: |
'---------------*/

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

// Constants
#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define iResolution float2(BUFFER_WIDTH, BUFFER_HEIGHT)

// Randomized Halton
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

// Pseudo-random number generator based on pixel coordinates
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
    XY *= (pix * 1.5); // Increased jitter multiplier

    const float4 PastColor = tex2D(PBackBuffer, texcoord + XY);
    float3 antialiased = PastColor.xyz;
    const float mixRate = min(PastColor.w, 0.75); // Increased mix rate for stronger effect

    float3 in0 = tex2D(BackBuffer, texcoord).xyz;
    antialiased = lerp(antialiased * antialiased, in0 * in0, mixRate);
    antialiased = sqrt(antialiased);

    // Neighboring samples for increased effectiveness
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

    // Final TAA Pass
    float4 TAA_result = TAA(position, texcoord);

    // Determine output based on View Mode
    if (View_Mode == 0)
    {
        // DTAA Output
        color = lerp(DLAA, TAA_result, 0.5); 
    }
}

/*-----------------.
| :: Vertex Shader :: |
'-----------------*/

// Generate a triangle covering the entire screen
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
