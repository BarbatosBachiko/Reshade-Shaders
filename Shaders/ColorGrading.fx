/*------------------.
| :: Description :: |
'-------------------/

	Color Grading (version 1.0)

	Author: BarbatosBachiko
        License: MIT

	About:
	This shader adjusts brightness, saturation, and contrast and color grading interpolation.

	Ideas for future improvement:

        History:
        (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
	
	Version 1.0
	* Release

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int StartColorGradingMethod < 
    ui_type = "combo";
    ui_label = "Start Color Grading Method"; 
    ui_items = "Neutral\0Warm\0Cool\0Sepia\0Black & White\0Vintage\0Vibrant\0Horror\0Cine Style\0Teal and Orange\0"; 
> = 1;

uniform int EndColorGradingMethod < 
    ui_type = "combo";
    ui_label = "End Color Grading Method"; 
    ui_items = "Neutral\0Warm\0Cool\0Sepia\0Black & White\0Vintage\0Vibrant\0Horror\0Cine Style\0Teal and Orange\0"; 
> = 2;

uniform float GradingInterpolationFactor < 
    ui_type = "slider";
    ui_label = "Color Grading Interpolation"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
> = 0.5;

uniform float SaturationFactor < 
    ui_type = "slider";
    ui_label = "Saturation Adjustment"; 
    ui_min = 0.0; 
    ui_max = 2.0; 
> = 1.0;

uniform float Brightness < 
    ui_type = "slider";
    ui_label = "Brightness Adjustment"; 
    ui_min = -1.0; 
    ui_max = 1.0; 
> = 0.0;

uniform float Contrast < 
    ui_type = "slider";
    ui_label = "Contrast Adjustment"; 
    ui_min = 0.0; 
    ui_max = 2.0; 
> = 1.0;

/*---------------.
| :: Textures :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

// Function to apply a color gradient style
float3 ApplyColorGradingStyle(float3 color, int method)
{
    float3 gradedColor = color;

    if (method == 0) // Neutral
        return color;
    
    if (method == 1) // Warm
        return color * float3(1.2, 1.1, 0.9);
    
    if (method == 2) // Cool
        return color * float3(0.9, 1.1, 1.2);
    
    if (method == 3) // Sepia
        return float3(dot(color, float3(0.393, 0.769, 0.189)),
                      dot(color, float3(0.349, 0.686, 0.168)),
                      dot(color, float3(0.272, 0.534, 0.131)));

    if (method == 4) // Black and White
    {
        float gray = dot(color, float3(0.2989, 0.5870, 0.1140));
        return float3(gray, gray, gray);
    }
    
    if (method == 5) // Vintage
        return color * float3(1.0, 0.9, 0.8) * 0.8 + float3(0.1, 0.05, 0.05);
    
    if (method == 6) // Vibrant
        return pow(color, float3(0.8, 1.0, 0.9));
    
    if (method == 7) // Horror
        return float3(color.r * 0.5, color.g * 0.2, color.b * 0.2);

    if (method == 8) // Cine Style
        return float3(color.r * 1.1, color.g * 0.95, color.b * 0.85);

    if (method == 9) // Teal and Orange
        return float3(color.r * 0.3 + color.g * 0.5 + color.b * 0.2, color.g * 0.6 + color.b * 0.4, color.b * 0.8);

    return gradedColor;
}

// // Function to adjust brightness and contrast
float3 AdjustBrightnessContrast(float3 color, float brightness, float contrast)
{
    color += brightness; 
    return ((color - 0.5) * contrast) + 0.5;
}

// Color grading function with interpolation
float3 ApplyInterpolatedColorGrading(float3 color)
{
    float3 startGrading = ApplyColorGradingStyle(color, StartColorGradingMethod);
    float3 endGrading = ApplyColorGradingStyle(color, EndColorGradingMethod);
    float3 interpolatedColor = lerp(startGrading, endGrading, GradingInterpolationFactor);
    float gray = dot(interpolatedColor, float3(0.2989, 0.5870, 0.1140));
    interpolatedColor = lerp(float3(gray, gray, gray), interpolatedColor, SaturationFactor);
    return AdjustBrightnessContrast(interpolatedColor, Brightness, Contrast);
}

// Main Color Grading Pass
float4 ColorGradingPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(BackBuffer, texcoord).rgb;
    
    color = ApplyInterpolatedColorGrading(color);
    
    return float4(saturate(color), 1.0); 
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique ColorGrading
{
    pass 
    {
        VertexShader = PostProcessVS;
        PixelShader = ColorGradingPass;
    }
}
