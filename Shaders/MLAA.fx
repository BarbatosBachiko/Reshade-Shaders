/*------------------.
| :: Description :: |
'-------------------/

	Author: Barbatos Bachiko
	License: MIT

	About:
	This shader implements Morphological Anti-Aliasing (MLAA)

	Ideas for future improvement:
	* Improve Blending
	
	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 1.0
	* Initial implementation of MLAA shader.

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform bool Boolean
<
	ui_label = "Enable MLAA";
	ui_tooltip = "Turn on/off MLAA effect.";
>
= true;

uniform float Slider
<
	ui_type = "slider";
	ui_label = "Edge Threshold";
	ui_tooltip = "Adjust the edge threshold for detection.";
	ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
>
= 0.06;

uniform bool DebugMode
<
    ui_label = "Enable Debug Mode";
    ui_tooltip = "Turn on/off edge highlighting.";
>
= false;

/*---------------.
| :: Textures :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

/*---------------.
| :: Samplers :: |
'---------------*/

float4 GetColor(float2 uv)
{
    return tex2D(BackBuffer, uv);
}

#define MAX_EDGE_LENGTH 32 // Define the maximum edge length to check

/*----------------.
| :: Functions :: |
'----------------*/

// Calculate luminance
float CalculateLuminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114)); // luminance calculation
}

// Edge detection
void EdgeDetectionPass(float2 texcoord, out bool isEdge, out int mask)
{
    float4 colorCurrent = GetColor(texcoord);
    float4 colorUp = GetColor(texcoord + float2(0.0, 1.0) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT));
    float4 colorRight = GetColor(texcoord + float2(1.0, 0.0) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT));
    
    float alphaCurrent = CalculateLuminance(colorCurrent.rgb);
    float alphaUp = CalculateLuminance(colorUp.rgb);
    float alphaRight = CalculateLuminance(colorRight.rgb);
    
    // Calculate luminance differences
    float luminanceDiffUp = abs(alphaCurrent - alphaUp);
    float luminanceDiffRight = abs(alphaCurrent - alphaRight);
    
    // Threshold check
    isEdge = (luminanceDiffUp > Slider || luminanceDiffRight > Slider);
    
    // Set masks for edge direction
    mask = 0;
    if (luminanceDiffUp > Slider)
        mask |= 0x1; // kUpperMask
    if (luminanceDiffRight > Slider)
        mask |= 0x2; // kRightMask
}

// Edge length
uint2 ComputeEdgeLength(float2 texcoord, int mask)
{
    uint2 edgeLength = uint2(0, 0); // Store the lengths: up/down and left/right

    // For the up direction
    if ((mask & 0x1) != 0)
    {
        float luminanceUp0 = CalculateLuminance(GetColor(texcoord + float2(0.0, -1) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp1 = CalculateLuminance(GetColor(texcoord + float2(0.0, -2) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp2 = CalculateLuminance(GetColor(texcoord + float2(0.0, -3) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp3 = CalculateLuminance(GetColor(texcoord + float2(0.0, -4) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp4 = CalculateLuminance(GetColor(texcoord + float2(0.0, -5) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp5 = CalculateLuminance(GetColor(texcoord + float2(0.0, -6) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp6 = CalculateLuminance(GetColor(texcoord + float2(0.0, -7) * BUFFER_RCP_HEIGHT).rgb);
        float luminanceUp7 = CalculateLuminance(GetColor(texcoord + float2(0.0, -8) * BUFFER_RCP_HEIGHT).rgb);

        edgeLength.x = (luminanceUp0 > Slider ? 1 : 0) +
                        (luminanceUp1 > Slider ? 1 : 0) +
                        (luminanceUp2 > Slider ? 1 : 0) +
                        (luminanceUp3 > Slider ? 1 : 0) +
                        (luminanceUp4 > Slider ? 1 : 0) +
                        (luminanceUp5 > Slider ? 1 : 0) +
                        (luminanceUp6 > Slider ? 1 : 0) +
                        (luminanceUp7 > Slider ? 1 : 0);
    }

    // For the right direction
    if ((mask & 0x2) != 0)
    {
        float luminanceRight0 = CalculateLuminance(GetColor(texcoord + float2(1, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight1 = CalculateLuminance(GetColor(texcoord + float2(2, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight2 = CalculateLuminance(GetColor(texcoord + float2(3, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight3 = CalculateLuminance(GetColor(texcoord + float2(4, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight4 = CalculateLuminance(GetColor(texcoord + float2(5, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight5 = CalculateLuminance(GetColor(texcoord + float2(6, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight6 = CalculateLuminance(GetColor(texcoord + float2(7, 0.0) * BUFFER_RCP_WIDTH).rgb);
        float luminanceRight7 = CalculateLuminance(GetColor(texcoord + float2(8, 0.0) * BUFFER_RCP_WIDTH).rgb);

        edgeLength.y = (luminanceRight0 > Slider ? 1 : 0) +
                       (luminanceRight1 > Slider ? 1 : 0) +
                       (luminanceRight2 > Slider ? 1 : 0) +
                       (luminanceRight3 > Slider ? 1 : 0) +
                       (luminanceRight4 > Slider ? 1 : 0) +
                       (luminanceRight5 > Slider ? 1 : 0) +
                       (luminanceRight6 > Slider ? 1 : 0) +
                       (luminanceRight7 > Slider ? 1 : 0);
    }

    return edgeLength;
}

// Main MLAA
float4 MLAA_PixelShader(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!Boolean)
        return GetColor(texcoord);

    bool isEdge;
    int mask;

    // Run edge detection
    EdgeDetectionPass(texcoord, isEdge, mask);

    // If an edge is detected, compute the edge length
    if (isEdge)
    {
        uint2 edgeLength = ComputeEdgeLength(texcoord, mask);

        // Smooth the edges by averaging colors
        float4 colorCurrent = GetColor(texcoord);
        float4 colorUp = GetColor(texcoord + float2(0.0, 1.0) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT));
        float4 colorRight = GetColor(texcoord + float2(1.0, 0.0) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT));

        // Blending color
        float4 blendedColor = (colorCurrent + colorUp + colorRight) / 3.0;

        // Debug mode: highlight edges in red
        if (DebugMode)
        {
            return float4(1.0, 0.0, 0.0, 1.0); // Return red for edges
        }

        return blendedColor; // Return the blended color
    }

    // return the original color
    return GetColor(texcoord);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique MLAA
{
    pass
    {
        VertexShader = PostProcessVS; // in ReShade.fxh
        PixelShader = MLAA_PixelShader;
    }
}

/*-------------.
| :: Footer :: |
'--------------/
https://github.com/GPUOpen-LibrariesAndSDKs/MLAA11/tree/master