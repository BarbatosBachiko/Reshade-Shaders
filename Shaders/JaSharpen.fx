/*-----------------------------------------------------------------------------------------------------------------------------|
| ::                                                JaSharpen                                                               :: |
'------------------------------------------------------------------------------------------------------------------------------|
| Version 1.0                                                                                                                  |
| Author: Barbatos, based on "Image Sharpening Convolution Kernels" by Alan Wolfe (https://github.com/Atrix256/ImageSharpening)|
| License: MIT                                                                                                                 |
| About: Implements image sharpening using convolution kernels and unsharp masking.                                            |
'-----------------------------------------------------------------------------------------------------------------------------*/

#include "ReShade.fxh"

/*---------.
| :: UI::  |
'---------*/

uniform float INTENSITY <
    ui_type = "drag";
    ui_label = "Sharpening Intensity";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.01;
> = 1.0;

/*----------------.
| :: Functions :: |
'----------------*/

// Box blur kernel 3x3
static const float BoxLPF[9] =
{
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0
};

float3 ConvolvePixel3x3(float2 texcoord, float kernel[9])
{
    float2 pixelSize = ReShade::PixelSize;
    float3 result = 0.0;
    
    int index = 0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float2 offset = float2(x, y) * pixelSize;
            float3 sampleA = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;
            result += sampleA * kernel[index];
            index++;
        }
    }
    
    return result;
}

float4 PS_Sharpen(float4 pos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 image = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 blurred = ConvolvePixel3x3(texcoord, BoxLPF);
    float3 highPass = image - blurred;
    float3 sharpened = saturate(image + INTENSITY * highPass);
    return float4(sharpened, 1.0);
}

technique JaSharpen
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Sharpen;
    }
}
