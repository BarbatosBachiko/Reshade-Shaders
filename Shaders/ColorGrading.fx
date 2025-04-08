/*------------------.
| :: Description :: |
'-------------------/

    Color Grading 

    Version 1.2
    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adjusts brightness, saturation, contrast, and performs color grading interpolation.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

*/

#include "ReShade.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int StartColorGradingMethod < 
    ui_type = "combo";
    ui_label = "Start Color Grading Method"; 
    ui_items = "Neutral\0Warm\0Cool\0Sepia\0Black & White\0Vintage\0Vibrant\0Horror\0Cine Style\0Teal and Orange\0"; 
> = 6;

uniform int EndColorGradingMethod < 
    ui_type = "combo";
    ui_label = "End Color Grading Method"; 
    ui_items = "Neutral\0Warm\0Cool\0Sepia\0Black & White\0Vintage\0Vibrant\0Horror\0Cine Style\0Teal and Orange\0"; 
> = 8;

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

/*----------------.
| :: Functions :: |
'----------------*/

// Function to convert from RGB to HSL
float3 RGBtoHSL(float3 color)
{
    float3 hsl;
    float minVal = min(min(color.r, color.g), color.b);
    float maxVal = max(max(color.r, color.g), color.b);
    float delta = maxVal - minVal;

    // Luminosity (L)
    hsl.z = (maxVal + minVal) / 2.0;

    if (delta < 0.00001)
    {
        hsl.x = 0.0;
        hsl.y = 0.0;
    }
    else
    {
        // Saturation (S)
        if (hsl.z < 0.5)
            hsl.y = delta / (maxVal + minVal);
        else
            hsl.y = delta / (2.0 - maxVal - minVal);

        // Hue (H)
        float3 diff = (((maxVal - color) / 6.0) + (delta / 2.0)) / delta;
        
        if (maxVal == color.r)
            hsl.x = diff.b - diff.g;
        else if (maxVal == color.g)
            hsl.x = (1.0 / 3.0) + diff.r - diff.b;
        else
            hsl.x = (2.0 / 3.0) + diff.g - diff.r;

        if (hsl.x < 0.0)
            hsl.x += 1.0;
        if (hsl.x > 1.0)
            hsl.x -= 1.0;
    }

    return hsl;
}

// Helper
float HueToRGB(float temp1, float temp2, float temp3)
{
    if (temp3 < 0.0)
        temp3 += 1.0;
    if (temp3 > 1.0)
        temp3 -= 1.0;
    
    if ((temp3 * 6.0) < 1.0)
        return temp1 + (temp2 - temp1) * 6.0 * temp3;
    else if ((temp3 * 2.0) < 1.0)
        return temp2;
    else if ((temp3 * 3.0) < 2.0)
        return temp1 + (temp2 - temp1) * ((2.0 / 3.0) - temp3) * 6.0;
    
    return temp1;
}

// HSL to RGB 
float3 HSLtoRGB(float3 hsl)
{
    float3 rgb;
    if (hsl.y == 0.0)
    {
        rgb = float3(hsl.z, hsl.z, hsl.z);
    }
    else
    {
        float temp2;
        if (hsl.z < 0.5)
            temp2 = hsl.z * (1.0 + hsl.y);
        else
            temp2 = hsl.z + hsl.y - (hsl.y * hsl.z);

        float temp1 = 2.0 * hsl.z - temp2;

        rgb.r = HueToRGB(temp1, temp2, hsl.x + (1.0 / 3.0));
        rgb.g = HueToRGB(temp1, temp2, hsl.x);
        rgb.b = HueToRGB(temp1, temp2, hsl.x - (1.0 / 3.0));
    }

    return rgb;
}

float3 ApplyColorGradingStyle(float3 color, int method)
{
    float3 gradedColor = color;
    float3 hsl;
    
    if (method == 0) // Neutral
        return color;
    
    if (method == 1) // Warm
    {
        const float3x3 warmMatrix = float3x3(
            1.2, 0.1, 0.0,
            0.1, 1.1, 0.0,
            0.0, 0.0, 0.9
        );
        return mul(warmMatrix, color);
    }
    
    if (method == 2) // Cool
    {
        const float3x3 coolMatrix = float3x3(
                    0.85, 0.0, 0.05,
                    0.0, 1.0, 0.1,
                    0.15, 0.1, 1.2
                );
        return saturate(mul(coolMatrix, color));
    }
    
    if (method == 3) // Sepia
        return float3(dot(color, float3(0.393, 0.769, 0.189)),
                      dot(color, float3(0.349, 0.686, 0.168)),
                      dot(color, float3(0.272, 0.534, 0.131)));

    if (method == 4) // Black and White
    {
        float gray = dot(color, float3(0.2126, 0.7152, 0.0722));
        return float3(gray, gray, gray);
    }
    
    if (method == 5) // Vintage
    {
        gradedColor = color * float3(1.05, 0.9, 0.75) * 0.85 + float3(0.08, 0.04, 0.04);
        hsl = RGBtoHSL(gradedColor);
        hsl.y *= 0.8; 
        return saturate(HSLtoRGB(hsl));
    }
    
    if (method == 6) // Vibrant
    {
        float3 hsl = RGBtoHSL(color);
        hsl.y = clamp(hsl.y * 1.5, 0.0, 1.0);
        color = HSLtoRGB(hsl);
        color = (color - 0.5) * 1.1 + 0.5;

        return color;
    }
    
    if (method == 7) // Horror
    {
        gradedColor = color * float3(1.0, 0.8, 0.8);
        gradedColor = (gradedColor - 0.5) * 1.2 + 0.45; 
                
        float3x3 horrorMatrix = float3x3(
                    1.1, -0.1, 0.1,
                    -0.1, 0.8, -0.1,
                    -0.1, -0.2, 0.7
                );
                
        return saturate(mul(horrorMatrix, gradedColor));
    }

    if (method == 8) // Cine Style
    {
        const float3x3 cineMatrix = float3x3(
                    1.05, 0.0, -0.05,
                    0.0, 0.95, 0.05,
                    -0.05, 0.05, 0.85
                );
                
        gradedColor = mul(cineMatrix, color);
        gradedColor = (gradedColor - 0.5) * 1.05 + 0.5;
                
        return saturate(gradedColor);
    }

    if (method == 9) // Teal and Orange
    {
        float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
                
        float3 shadows = float3(0.05, 0.27, 0.35); 
        float3 midtones = float3(0.9, 0.8, 0.6); 
        float3 highlights = float3(0.9, 0.45, 0.2); 
                
        float shadowsWeight = 1.0 - smoothstep(0.0, 0.4, luminance);
        float highlightsWeight = smoothstep(0.6, 1.0, luminance);
        float midtonesWeight = 1.0 - shadowsWeight - highlightsWeight;
                
        float3 result = color * (
                    (shadows * shadowsWeight) +
                    (midtones * midtonesWeight) +
                    (highlights * highlightsWeight)
                );
                
        return saturate(result);
    }

    return gradedColor;
}

// Brightness and Contrast
float3 BContrast(float3 color, float brightness, float contrast)
{
    color += brightness;
    return ((color - 0.5) * contrast) + 0.5;
}

// Color grading with interpolation
float3 IntColorGrading(float3 color)
{
    float3 startGrading = ApplyColorGradingStyle(color, StartColorGradingMethod);
    float3 endGrading = ApplyColorGradingStyle(color, EndColorGradingMethod);
    float3 interpolatedColor = lerp(startGrading, endGrading, GradingInterpolationFactor);
    float gray = dot(interpolatedColor, float3(0.2989, 0.5870, 0.1140));
    interpolatedColor = lerp(float3(gray, gray, gray), interpolatedColor, SaturationFactor);
    return BContrast(interpolatedColor, Brightness, Contrast);
}

float4 ColorGradingPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    
    color = IntColorGrading(color);
    
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
