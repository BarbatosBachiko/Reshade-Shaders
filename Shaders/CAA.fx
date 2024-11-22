/*------------------.
| :: Description :: |
'-------------------/

   Convolution Anti-Aliasing (Version 1.1)

    Author: Barbatos Bachiko
    License: MIT
    About: Implements an anti-aliasing technique using a local convolution filter and pixel warping if necessary for smoothing.

    History:
	 (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 1.1
	+ Improve Distortion
   + Add Kernel Size
*/

#include "ReShade.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Controls the sensitivity for detecting edges."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.15; 
> = 0.15;

uniform float DistortionStrength < 
    ui_type = "slider";
    ui_label = "Distortion Strength"; 
    ui_tooltip = "Strength of pixel distortion for anti-aliasing.";
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.0; 
> = 0.0;

uniform int KernelSize < 
    ui_type = "slider";
    ui_label = "Kernel Size"; 
    ui_tooltip = "Adjust the size of the convolution kernel."; 
    ui_min = 3; 
    ui_max = 11; 
    ui_default = 3; 
> = 3;

/*---------------.
| :: Textures :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler2D BackBuffer
{
    Texture = BackBufferTex;
};

#define PI 3.14159265358979323846

/*----------------.
| :: Functions :: |
'----------------*/

// Loads a pixel from the texture
float4 LoadPixel(float2 texcoord)
{
    return tex2D(BackBuffer, texcoord);
}

// Calculates the variance between the pixel and its neighbors based on luminance
float GetPixelVariance(float2 texcoord)
{
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
    float4 center = LoadPixel(texcoord);
    float centerLuminance = 0.299 * center.r + 0.587 * center.g + 0.114 * center.b;

    float variance = 0.0;
    int halfSize = (KernelSize - 1) / 2;
    
    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            float2 sampleOffset = float2(x * offset.x, y * offset.y);
            float4 neighbor = LoadPixel(texcoord + sampleOffset);
            float neighborLuminance = 0.299 * neighbor.r + 0.587 * neighbor.g + 0.114 * neighbor.b;
            variance += abs(centerLuminance - neighborLuminance);
        }
    }

    return variance / (KernelSize * KernelSize);
}

// Applies adaptive convolution 
float4 AdaptiveConvolutionAA(float2 texcoord)
{
    float variance = GetPixelVariance(texcoord);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + 0.1, variance);
    
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
    int halfSize = (KernelSize - 1) / 2;

    float4 smoothedPixel = 0.0;
    float totalWeight = 0.0;

    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            float2 sampleOffset = float2(x * offset.x, y * offset.y);
            float2 diagonalOffset = float2(x + y, x - y) * offset;
            float4 samplePixel1 = LoadPixel(texcoord + sampleOffset);
            float4 samplePixel2 = LoadPixel(texcoord + diagonalOffset);

            float kernelWeight = 1.0;
            smoothedPixel += samplePixel1 * kernelWeight;
            smoothedPixel += samplePixel2 * kernelWeight;
            totalWeight += 2.0 * kernelWeight;
        }
    }

    smoothedPixel /= totalWeight;

    float2 distortionOffset = float2(sin(texcoord.x * PI * 2.0), cos(texcoord.y * PI * 2.0)) * DistortionStrength;
    texcoord += distortionOffset * weight;
    
    return lerp(LoadPixel(texcoord), smoothedPixel, weight);
}

// Main function
float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return AdaptiveConvolutionAA(texcoord);
}

// Vertex shader to cover the entire screen
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique CAA
{
    pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = Out;
    }
}
