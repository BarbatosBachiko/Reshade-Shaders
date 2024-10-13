/*------------------.
| :: Description :: |
'-------------------/

    DLAA Lite V4.1

    Author: BarbatosBachiko

    About:
    Directionally Localized Anti-Aliasing (DLAA) Lite Version 4.1 with variable rate shading (VRS) and sharpness control.


    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 4.1
    * Added VRS intensity control.
    * Introduced sharpness level adjustment.

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int View_Mode
< 
    ui_type = "combo";
    ui_items = "DLAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "This is used to select the normal view output or debug view."; 
> = 0;

uniform float vrsIntensity
< 
    ui_label = "VRS Intensity";
    ui_tooltip = "Control the intensity of Variable Rate Shading (VRS)."; 
> = 0.5;

uniform float sharpness
< 
    ui_label = "Sharpness";
    ui_tooltip = "Control the sharpness level."; 
> = 1.0;

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define lambda 3.0f
#define epsilon 0.1f

/*---------------.
| :: Textures :: |
'---------------*/

texture BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

// Load pixel color from the BackBuffer
float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

// Apply DLAA
float4 ApplyDLAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);

    const float LumH = (centerDiffH.r + centerDiffH.g + centerDiffH.b) / 3.0;
    const float satAmountH = saturate((lambda * LumH - epsilon) / LumH);

    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

// Apply sharpness
float4 ApplySharpness(float4 color, float2 texcoord)
{
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -1.0 * pix.y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, 1.0 * pix.y));

    float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);
    return clamp(sharpened, 0.0, 1.0);
}

// Apply Variable Rate Shading (VRS)
float4 ApplyVRS(float4 color, float2 texcoord)
{
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -1.0 * pix.y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, 1.0 * pix.y));

    float4 diffLeft = abs(left - color);
    float4 diffRight = abs(right - color);
    float4 diffTop = abs(top - color);
    float4 diffBottom = abs(bottom - color);

    float complexity = (dot(diffLeft.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffRight.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffTop.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffBottom.rgb, float3(1.0, 1.0, 1.0))) / 4.0;

    float vrsFactor = 1.0 - (complexity * vrsIntensity);
    return color * vrsFactor;
}

/*-----------------.
| :: Pixel Shader :: |
'-----------------*/

float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float4 center = LoadPixel(BackBuffer, texcoord);
    const float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    const float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));

    float4 DLAA = ApplyDLAA(center, left, right);

    DLAA = ApplyVRS(DLAA, texcoord);
    DLAA = ApplySharpness(DLAA, texcoord);

    return DLAA;
}

/*-----------------.
| :: Vertex Shader :: |
'-----------------*/

// Vertex shader generating a full-screen triangle
void DLAA_VertexShader(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DLAA_Lite_V4
{
    pass DLAA_Light
    {
        VertexShader = DLAA_VertexShader; 
        PixelShader = Out;
    }
}

/*-------------.
| :: Footer :: |
'--------------/

This shader implements Directionally Localized Anti-Aliasing with optional VRS and sharpness adjustment.
