/*------------------.
| :: Description :: |
'-------------------/

    DLAA Lite V1.3

    Author: BarbatosBachiko

    License: MIT	
    Original method by Dmitry Andreev

    About:
    Directionally Localized Anti-Aliasing (DLAA) Lite Version 1.3.
 
    Ideas for future improvement:
    * Add edge enhancement.
    
    Version 1.0
    *Improved for performance
    Version 1.1
    * Added Shading intensity control.
    * Introduced sharpness level adjustment.
    Version 1.2
    * Introduced edge enhancement.
    * Included DLAA debug view mode.
    Version 1.3
    * Adjustable parameters, 
    * Added Shading intensity control
    * Added initial sharpness level adjustment
    * other corrections

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
    ui_items = "DLAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "Select normal output or debug view."; 
> = 0; // Default view mode as DLAA Out

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Adjust the edge threshold for mask creation."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.1; 
> = 0.1; // Adjustable edge threshold

uniform float Lambda < 
    ui_type = "slider";
    ui_label = "Lambda"; 
    ui_tooltip = "Adjust the lambda for saturation amount."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 3.0; 
> = 3.0; // Adjustable lambda

uniform bool EnableShading < 
    ui_type = "checkbox";
    ui_label = "Enable Shading"; 
    ui_tooltip = "Enable or disable the shading effect."; 
> = true; // Habilitado por padrão

uniform float ShadingIntensity < 
    ui_type = "slider";
    ui_label = "Shading Intensity"; 
    ui_tooltip = "Control the Shading."; 
> = 0.2; // Default Shading intensity

uniform float sharpness < 
    ui_type = "slider";
    ui_label = "Sharpness";
    ui_tooltip = "Control the sharpness level."; 
> = 0.2; // Default sharpness

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

// Load pixel from texture
float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

// Apply DLAA
float4 ApplyDLAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);

    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    const float satAmountH = saturate((Lambda * LumH - 0.1f) / LumH);

    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

// Apply sharpness
float4 ApplySharpness(float4 color, float2 texcoord)
{
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, 1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));

    float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);
    return clamp(sharpened, 0.0, 1.0);
}

// Apply Shading
float4 ApplyShading(float4 color, float2 texcoord)
{
    if (!EnableShading)
    {
        return color;
    }
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, 1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
    float4 diffLeft = abs(left - color);
    float4 diffRight = abs(right - color);
    float4 diffTop = abs(top - color);
    float4 diffBottom = abs(bottom - color);
    float complexity = (dot(diffLeft.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffRight.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffTop.rgb, float3(1.0, 1.0, 1.0)) +
                        dot(diffBottom.rgb, float3(1.0, 1.0, 1.0))) / 4.0;
    float vrsFactor = 1.0 - (complexity * ShadingIntensity);
    return color * vrsFactor;
}

// Apply blur to mask
float4 BlurMask(float4 mask, float2 tc, float2 pix)
{
    float4 blur = mask * 0.25; // Start with current mask weight
    blur += LoadPixel(BackBuffer, tc + float2(-pix.x, 0)) * 0.125; // Left
    blur += LoadPixel(BackBuffer, tc + float2(pix.x, 0)) * 0.125; // Right
    blur += LoadPixel(BackBuffer, tc + float2(0, -pix.y)) * 0.125; // Up
    blur += LoadPixel(BackBuffer, tc + float2(0, pix.y)) * 0.125; // Down
    return saturate(blur); // Prevent overflow
}

// Main DLAA pass
float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    // Define pix based on texture size
    float2 pix = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    const float4 center = LoadPixel(BackBuffer, texcoord);
    const float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    const float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));

    float4 DLAA = ApplyDLAA(center, left, right);
    
    // Apply Shadding to the DLAA result
    DLAA = ApplyShading(DLAA, texcoord);
    
    // Apply sharpness to the DLAA result
    DLAA = ApplySharpness(DLAA, texcoord);

    // Mask calculation
    const float4 maskColorA = float4(1.0, 1.0, 0.0, 1.0); // Yellow for Mask A
    const float4 maskColorB = float4(1.0, 0.0, 0.0, 1.0); // Red for Mask B

    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    
    const float maskA = LumH > EdgeThreshold ? 1.0 : 0.0;

    if (View_Mode == 1)
    {
        return BlurMask(maskA * maskColorA, texcoord, pix); // Returns blurred mask A in yellow
    }

    // Mask View B
    const float4 diff = abs(DLAA - center);
    const float maxDiff = max(max(diff.r, diff.g), diff.b);

    // Adjust output of mask with gradient color
    if (View_Mode == 2)
    {
        float intensity = saturate(maxDiff * 5.0); // Increase intensity for better visualization
        return float4(intensity, 0.0, intensity * 0.5, 1.0); // Returns color based on intensity
    }

    return DLAA; // Final DLAA output
}

// Vertex shader generating a triangle covering the entire screen
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

// Unique technique for optimized DLAA
technique DLAA_Lite
{
    pass DLAA_Light
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = Out;
    }
}
