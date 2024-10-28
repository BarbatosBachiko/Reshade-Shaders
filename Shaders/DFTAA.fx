/*------------------.
| :: Description :: |
'-------------------/

	DFTAA (version 1.1)

	Author: Barbatos Bachiko

	About:
	Implementation of Directionally Fast Temporal Anti-Aliasing (DFTAA) that combines DLAA, FSMAA and TAA.

	Ideas for future improvement:
	* Integrate an adjustable sharpness
	* Swap TAA to other version

        History:
        (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

	Version 1.0
	* Initial version with integrated DFTAA
        Version 1.1
	x Bugfix

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int View_Mode < 
    ui_type = "combo";
    ui_items = "DFTAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "Select normal output or debug view."; 
> = 0; // Default view mode as DFTAA Out

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Adjust the edge threshold for mask creation."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.020; 
> = 0.020; // Default edge threshold

uniform float Lambda < 
    ui_type = "slider";
    ui_label = "Lambda"; 
    ui_tooltip = "Adjust the lambda for saturation amount."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 3.0; 
> = 3.0; // Adjustable lambda

/*---------------.
| :: Textures :: |
'---------------*/

texture BackBufferTex : COLOR;
texture PreviousFrameTex : COLOR;

/*---------------.
| :: Samplers :: |
'---------------*/

sampler BackBuffer
{
    Texture = BackBufferTex;
};

sampler texPrevious
{
    Texture = PreviousFrameTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

// Function to load pixels
float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

// Function to apply DFTAA 
float4 ApplyDFTAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);

    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    const float satAmountH = saturate((Lambda * LumH - 0.1f) / LumH);

    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

// Function to apply a blur to the mask
float4 BlurMask(float4 mask, float2 tc)
{
    float4 blur = mask * 0.25; // Start with current mask weight
    blur += LoadPixel(BackBuffer, tc + float2(-1.0 * BUFFER_RCP_WIDTH, 0)) * 0.125; // Left
    blur += LoadPixel(BackBuffer, tc + float2(1.0 * BUFFER_RCP_WIDTH, 0)) * 0.125; // Right
    blur += LoadPixel(BackBuffer, tc + float2(0, -BUFFER_RCP_HEIGHT)) * 0.125; // Up
    blur += LoadPixel(BackBuffer, tc + float2(0, BUFFER_RCP_HEIGHT)) * 0.125; // Down
    return saturate(blur); // Prevent overflow
}

// Main DFTAA pass
float4 DFTAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float4 center = LoadPixel(BackBuffer, texcoord);
    const float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * BUFFER_RCP_WIDTH, 0));
    const float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * BUFFER_RCP_WIDTH, 0));

    float4 DFTAA = ApplyDFTAA(center, left, right);

    // Mask calculation for DFTAA
    const float4 maskColorA = float4(1.0, 1.0, 0.0, 1.0);

    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    
    const float maskA = LumH > EdgeThreshold ? 1.0 : 0.0;

    // Mask View A
    if (View_Mode == 1)
    {
        return BlurMask(maskA * maskColorA, texcoord);
    }

    // Mask View B
    const float4 diff = abs(DFTAA - center);
    const float maxDiff = max(max(diff.r, diff.g), diff.b);
    const float maskB = maxDiff > EdgeThreshold ? 1.0 : 0.0;

    if (View_Mode == 2)
    {
        return BlurMask(maskB * float4(1.0, 0.0, 0.0, 1.0), texcoord); // Returns blurred mask B in red
    }

    return DFTAA; // Final DFTAA output
}

// FSMAA implementation
float4 SMAALumaEdgeDetectionPS(float2 texcoord)
{
    float4 pixel = LoadPixel(BackBuffer, texcoord);
    float luminance = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
    return float4(luminance, luminance, luminance, 1.0);
}

// FSMAA Edge Detection
float4 FSMAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 edgeDetection = SMAALumaEdgeDetectionPS(texcoord);
    
    // Just return DFTAA output by default
    return DFTAAPass(position, texcoord); // Final DFTAA output
}

// TAA implementation
float4 TAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    static float2 JitterPattern[4] =
    {
        float2(0.0, 0.0),
        float2(0.25, 0.25),
        float2(0.75, 0.25),
        float2(0.25, 0.75)
    };
    
    static int frameIndex = 0;
    int jitterIndex = frameIndex % 4;
    float2 jitter = JitterPattern[jitterIndex] / float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    float4 previousColor = LoadPixel(texPrevious, texcoord);
    float4 currentColor = LoadPixel(BackBuffer, texcoord + jitter);
    
    frameIndex++;

    return lerp(previousColor, currentColor, 0.5);
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

technique DFTAA
{
    pass DFTAA_Light
    {
        VertexShader = CustomPostProcessVS; 
        PixelShader = DFTAAPass; 
    }

    pass FSMAA_Pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = FSMAAPass; 
    }

    pass TAA_Pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = TAAPass; 
    }
}
