/*------------------.
| :: Description :: |
'-------------------/

   Outline
    
    Version: 1.2
    Author: Barbatos Bachiko
    License: MIT

    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    About: Outline Shader
    
    Version 1.2
    + Revised
*/

#include "ReShade.fxh"
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))

    /*---------------.
    | :: Settings :: |
    '---------------*/

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.2; 
> = 0.2;

uniform float3 EdgeColor < 
    ui_type = "color";
    ui_label = "Edge Color"; 
    ui_default = float3(0.0, 0.0, 0.0); 
> = float3(0.0, 0.0, 0.0);

uniform float Intensity <
    ui_type = "slider";
    ui_label = "Intensity";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_default = 1.0;
> = 0.3;

    /*----------------.
    | :: Functions :: |
    '----------------*/

float4 LoadPixel(float2 uv)
{
    return GetColor(uv);
}

float GetLuminance(float4 c)
{
    return dot(c.rgb, float3(0.299, 0.587, 0.114));
}

float ComputeSobel(float2 uv)
{
    float2 invRes = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);

    // Sobel kernels
    float3x3 Kx = float3x3(
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0
        );
    float3x3 Ky = float3x3(
            -1.0, -2.0, -1.0,
             0.0, 0.0, 0.0,
             1.0, 2.0, 1.0
        );

    float gx = 0.0;
    float gy = 0.0;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            float2 sampleUV = uv + float2(i, j) * invRes;
            float lum = GetLuminance(LoadPixel(sampleUV));
            gx += Kx[j + 1][i + 1] * lum;
            gy += Ky[j + 1][i + 1] * lum;
        }
    }

    return sqrt(gx * gx + gy * gy);
}

float4 ApplyOutline(float2 texcoord)
{
    float edgeDistance = ComputeSobel(texcoord);
    float edgeVisibility = smoothstep(EdgeThreshold - 0.1, EdgeThreshold + 0.1, edgeDistance);
    float4 outlineColor = float4(EdgeColor, 1.0) * edgeVisibility * Intensity;
    float4 originalColor = LoadPixel(texcoord);

    return lerp(originalColor, outlineColor, edgeVisibility * Intensity);
}

float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return ApplyOutline(texcoord);
}

    /*-----------------.
    | :: Techniques :: |
    '-----------------*/

technique SOutline
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Out;
    }
}
