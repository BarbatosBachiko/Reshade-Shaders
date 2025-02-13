/*------------------.
| :: Description :: |
'-------------------/

   Convolution Anti-Aliasing

    Version 1.2
    Author: Barbatos Bachiko
    License: MIT

    About: Implements an anti-aliasing technique using a local convolution filter.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
    Version 1.2
    + Added Debug Mode
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

uniform float ArtifactStrength < 
    ui_type = "slider";
    ui_label = "Artifact Strength"; 
    ui_tooltip = "Strength of pixel distortion."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.15; 
> = 0.15;

uniform int KernelSize < 
    ui_type = "slider";
    ui_label = "Kernel Size"; 
    ui_tooltip = "Adjust the size of the convolution kernel."; 
    ui_min = 3; 
    ui_max = 11; 
    ui_default = 3; 
> = 7;

uniform bool DebugMode <
    ui_type = "checkbox";
    ui_label = "Debug Mode";
    ui_tooltip = "Enable debug visualization of edges or variance.";
    ui_default = false;
> = false;

#define PI 3.14159265358979323846

/*----------------.
| :: Functions :: |
'----------------*/

// Loads a pixel from the texture
float4 LoadPixel(float2 texcoord)
{
    return tex2D(ReShade::BackBuffer, texcoord);
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

float4 AdaptiveConvolutionAA(float2 texcoord)
{
    float variance = GetPixelVariance(texcoord);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + 0.1, variance);
    
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
    int halfSize = (KernelSize - 1) / 2;

    float kernelWeights[121]; 
    float sigma = float(halfSize) / 2.0; 
    float weightSum = 0.0;

    // Precompute Gaussian weights
    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            float dist = sqrt(x * x + y * y);
            float gaussianWeight = exp(-(dist * dist) / (2.0 * sigma * sigma)) / (2.0 * PI * sigma * sigma);
            int index = (y + halfSize) * KernelSize + (x + halfSize);
            kernelWeights[index] = gaussianWeight;
            weightSum += gaussianWeight;
        }
    }

    for (int i = 0; i < KernelSize * KernelSize; i++)
    {
        kernelWeights[i] /= weightSum;
    }
    
    float4 smoothedPixel = 0.0;
    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++) 
        {
            float2 sampleOffset = float2(x * offset.x, y * offset.y);
            float4 neighbor = LoadPixel(texcoord + sampleOffset);
            int index = (y + halfSize) * KernelSize + (x + halfSize); 
            float gaussianWeight = kernelWeights[index];
            smoothedPixel += neighbor * gaussianWeight;
        }
    }

    float2 distortionOffset = float2(sin(texcoord.x * PI * 2.0), cos(texcoord.y * PI * 2.0)) * ArtifactStrength;
    texcoord += distortionOffset * weight;

    return lerp(LoadPixel(texcoord), smoothedPixel, weight);
}

float4 Debug(float2 texcoord)
{
    float variance = GetPixelVariance(texcoord);
    float edgeMask = smoothstep(EdgeThreshold, EdgeThreshold + 0.1, variance);

    // Visualize edges in red
    if (edgeMask > 0.5)
    {
        return float4(1.0, 0.0, 0.0, 1.0); 
    }
    else
    {
        return float4(variance, variance, variance, 1.0);
    }
}

float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    if (DebugMode)
    {
        return Debug(texcoord);
    }
    else
    {
        return AdaptiveConvolutionAA(texcoord);
    }
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique CAA
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Out;
    }
}
