/*------------------.
| :: Description :: |
'-------------------/

   Convolution Anti-Aliasing (Version 1.0)

    Author: Barbatos Bachiko
    License: MIT
    About: Implements an anti-aliasing technique using a local convolution filter and pixel warping if necessary for smoothing.

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

// Calculates the difference between neighboring pixels (local variance)
float GetPixelVariance(float2 texcoord)
{
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
    float4 center = LoadPixel(texcoord);
    
    float4 neighbors[8] =
    {
        LoadPixel(texcoord + float2(-offset.x, -offset.y)),
        LoadPixel(texcoord + float2(offset.x, -offset.y)),
        LoadPixel(texcoord + float2(-offset.x, offset.y)),
        LoadPixel(texcoord + float2(offset.x, offset.y)),
        LoadPixel(texcoord + float2(-offset.x, 0)),
        LoadPixel(texcoord + float2(offset.x, 0)),
        LoadPixel(texcoord + float2(0, -offset.y)),
        LoadPixel(texcoord + float2(0, offset.y))
    };
    
    float variance = 0.0;
    for (int i = 0; i < 8; i++)
    {
        variance += length(center.rgb - neighbors[i].rgb);
    }
    
    return variance / 8.0;
}

// Adaptive convolution for anti-aliasing
float4 AdaptiveConvolutionAA(float2 texcoord)
{
    float variance = GetPixelVariance(texcoord);
    
    // If variance is above threshold, we consider it an edge and apply strong anti-aliasing
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + 0.1, variance);
    
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
    
    // Apply adaptive smoothing: stronger filter where variance is low (smooth areas)
    float4 smoothedPixel = 0.0;
    smoothedPixel += LoadPixel(texcoord + float2(-offset.x, -offset.y));
    smoothedPixel += LoadPixel(texcoord + float2(offset.x, -offset.y));
    smoothedPixel += LoadPixel(texcoord + float2(-offset.x, offset.y));
    smoothedPixel += LoadPixel(texcoord + float2(offset.x, offset.y));
    
    smoothedPixel += LoadPixel(texcoord + float2(-offset.x, 0));
    smoothedPixel += LoadPixel(texcoord + float2(offset.x, 0));
    smoothedPixel += LoadPixel(texcoord + float2(0, -offset.y));
    smoothedPixel += LoadPixel(texcoord + float2(0, offset.y));
    
    smoothedPixel /= 8.0;
    
    // If the area is not an edge, apply a distortion to smooth the pixels
    float2 distortionOffset = float2(sin(texcoord.x * PI * 2.0), cos(texcoord.y * PI * 2.0)) * DistortionStrength;
    texcoord += distortionOffset * weight;
    
    return lerp(LoadPixel(texcoord), smoothedPixel, weight);
}

// Main 
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