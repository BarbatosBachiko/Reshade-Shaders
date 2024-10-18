/*------------------.
| :: Description :: |
'-------------------/

    DLAA Lite V4.3 BETA - with BETA Edge Enhance

    Author: BarbatosBachiko

    About:
    Directionally Localized Anti-Aliasing (DLAA) Lite Version 4.3 

    Ideas for future improvement:
    * Further optimization for edge enhancement.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 4.3
    * Introduced edge enhancement.
    * Included DLAA debug view mode.

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
    ui_items = "DLAA Out\0DLAA Debug\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "This is used to select the normal view output or debug view."; 
> = 0;

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define lambda 3.0f
#define epsilon 0.1f
#define EdgeThreshold 0.2f

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

// Apply edge enhancement
float4 ApplyEdgeEnhance(float4 color, float2 texcoord)
{
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -1.0 * pix.y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, 1.0 * pix.y));

    float4 diffLeft = abs(left - color);
    float4 diffRight = abs(right - color);
    float4 diffTop = abs(top - color);
    float4 diffBottom = abs(bottom - color);

    float complexity = (dot(diffLeft.rgb, float3(0.2126, 0.7152, 0.0722)) +
                        dot(diffRight.rgb, float3(0.2126, 0.7152, 0.0722)) +
                        dot(diffTop.rgb, float3(0.2126, 0.7152, 0.0722)) +
                        dot(diffBottom.rgb, float3(0.2126, 0.7152, 0.0722))) / 4.0;

    float enhanceFactor = 1.0 + (complexity * 0.2);
    return color * enhanceFactor;
}

// Apply DLAA Debug
float4 BlurMask(float4 mask, float2 texcoord)
{
    return mask;
}

float4 ShowDLAADebug(float4 center, float4 left, float4 right, float2 texcoord, float4 original)
{
    const float4 maskColorA = float4(1.0, 1.0, 0.0, 1.0);

    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));

    const float maskA = LumH > EdgeThreshold ? 1.0 : 0.0;

    return lerp(original, BlurMask(maskA * maskColorA, texcoord), maskA);
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
    float4 DLAA_noEnhance = DLAA;

    DLAA = ApplyEdgeEnhance(DLAA, texcoord);

    if (View_Mode == 1) // DLAA Debug
    {
        return ShowDLAADebug(center, left, right, texcoord, center);
    }
    
    return DLAA;
}

/*-----------------.
| :: Vertex Shader :: |
'-----------------*/

// Renamed to CustomPostProcessVS to avoid ambiguity
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DLAA_Lite_V4_3
{
    pass DLAA_Light
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = Out;
    }
}

/*-------------.
| :: Footer :: |
'--------------/

This shader implements Directionally Localized Anti-Aliasing

