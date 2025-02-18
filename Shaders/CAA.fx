/*------------------.
| :: Description :: |
'-------------------/

   Convolution Anti-Aliasing

    Version 1.3
    Author: Barbatos Bachiko
    License: MIT

    About: Implements an anti-aliasing technique using a local convolution filter.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
    Version 1.3
    + Distortion using noise-based jitter
    + Adjustable smoothstep range for edge detection
    * Pixel Width
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

uniform float EdgeRange <
    ui_type = "slider";
    ui_label = "Edge Range";
    ui_tooltip = "Controls the transition smoothness of edge detection.";
    ui_min = 0.01;
    ui_max = 0.5;
    ui_default = 0.1;
> = 0.1;

uniform float ArtifactStrength < 
    ui_type = "slider";
    ui_label = "Artifact Strength"; 
    ui_tooltip = "Strength of pixel distortion."; 
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
> = 5;

uniform float PixelWidth <
    ui_type = "slider";
    ui_label = "Pixel Width";
    ui_tooltip = "Adjusts the pixel sampling width.";
    ui_min = 0.5;
    ui_max = 2.0;
    ui_default = 1.0;
> = 2.0;

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

// Generates a pseudo-random value
float2 hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy) * 2.0 - 1.0;
}

float4 LoadPixel(float2 texcoord)
{
    return tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0));
}

// Computes luminance variation around a pixel to detect edges
float GetPixelVariance(float2 texcoord)
{
    const float2 offset = float2(ReShade::PixelSize.xy) * PixelWidth;
    const float4 center = LoadPixel(texcoord);
    const float centerLuminance = dot(center.rgb, float3(0.299, 0.587, 0.114));

    float variance = 0.0;
    const int halfSize = (KernelSize - 1) / 2;
    
    // Iterates through the pixel neighborhood 
    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            const float2 sampleOffset = float2(x, y) * offset;
            const float4 neighbor = LoadPixel(texcoord + sampleOffset);
            const float neighborLuminance = dot(neighbor.rgb, float3(0.299, 0.587, 0.114));
            
            variance += abs(centerLuminance - neighborLuminance);
        }
    }
    
    return variance / (KernelSize * KernelSize);
}

// Applies an adaptive filter to smooth high-variation areas
float4 AdaptiveCAA(float2 texcoord, float variance)
{
    // Computes the smoothing weight based on edge detection
    const float weight = smoothstep(EdgeThreshold, EdgeThreshold + EdgeRange, variance);
    const float2 offset = float2(ReShade::PixelSize.xy) * PixelWidth;
    const int halfSize = (KernelSize - 1) / 2;
    const float sigma = max(float(halfSize) / 3.0, 0.5);

    float weightSum = 0.0;
    float4 smoothedPixel = 0.0;

    // Applies a Gaussian filter to smooth the region around the pixel
    for (int y = -halfSize; y <= halfSize; y++)
    {
        for (int x = -halfSize; x <= halfSize; x++)
        {
            const float dist = length(float2(x, y));
            const float gaussianWeight = exp(-(dist * dist) / (2.0 * sigma * sigma));
            
            const float2 sampleOffset = float2(x, y) * offset;
            const float4 neighbor = LoadPixel(texcoord + sampleOffset);
            
            smoothedPixel += neighbor * gaussianWeight;
            weightSum += gaussianWeight;
        }
    }
    smoothedPixel /= weightSum;

    // Adds a slight random offset to the pixel
    const float2 distortionOffset = hash22(texcoord) * ArtifactStrength * weight * 0.01;
    const float4 distortedPixel = LoadPixel(texcoord + distortionOffset);

    return lerp(distortedPixel, smoothedPixel, weight);
}

float4 DebugViz(float2 texcoord, float variance)
{
    const float edgeMask = smoothstep(EdgeThreshold, EdgeThreshold + EdgeRange, variance);
    return edgeMask > 0.5 ? float4(1.0, 0.0, 0.0, 1.0) : float4(variance.xxx, 1.0);
}

float4 PS_CAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float variance = GetPixelVariance(texcoord);
    
    if (DebugMode)
        return DebugViz(texcoord, variance);
    
    return AdaptiveCAA(texcoord, variance);
}

technique CAA
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CAA;
    }
}
