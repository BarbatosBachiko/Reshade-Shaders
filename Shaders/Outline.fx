/*------------------.
| :: Description :: |
'-------------------/

   Outline (Version 1.0)

    Author: Barbatos Bachiko
    License: MIT

    About: This shader creates an edge contour effect to apply a colored border to the edges of objects in the scene.

*/

#include "ReShade.fxh"

namespace Outline
{
/*---------------.
| :: Settings :: |
'---------------*/

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Controls the sensitivity for detecting edges."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.2; 
> = 0.2;

uniform float3 EdgeColor < 
    ui_type = "color";
    ui_label = "Edge Color"; 
    ui_tooltip = "Color of the edge outline."; 
    ui_default = float3(0.0, 0.0, 0.0); 
> = float3(0.0, 0.0, 0.0);

/*---------------.
| :: Textures :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler2D BackBuffer
{
    Texture = BackBufferTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

float4 LoadPixel(float2 texcoord)
{
    return tex2D(BackBuffer, texcoord);
}

float GetLuminance(float4 color)
{
    return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Sobel edge detection
float ComputeSobelEdge(float2 texcoord)
{
    float2 offset = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);

    // Sobel kernels
    float3x3 sobelX = float3x3(
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );

    float3x3 sobelY = float3x3(
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    );

    float gx = 0.0, gy = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float2 offsetCoord = texcoord + float2(x, y) * offset;
            float lum = GetLuminance(LoadPixel(offsetCoord));
            gx += sobelX[y + 1][x + 1] * lum;
            gy += sobelY[y + 1][x + 1] * lum;
        }
    }

    return sqrt(gx * gx + gy * gy);
}

// Applies an edge outline based on edge detection
float4 ApplyOutline(float2 texcoord)
{
    float edgeDistance = ComputeSobelEdge(texcoord);
    float edgeVisibility = smoothstep(EdgeThreshold - 0.1, EdgeThreshold + 0.1, edgeDistance);
    float4 outlineColor = float4(EdgeColor, 1.0) * edgeVisibility;
    float4 originalColor = LoadPixel(texcoord);

    return lerp(originalColor, outlineColor, edgeVisibility);
}

// Main Outline
float4 Outline(float2 texcoord)
{
    return ApplyOutline(texcoord);
}

// Main Pixel Shader
float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return Outline(texcoord);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique Outline
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Out;
    }
  }
}