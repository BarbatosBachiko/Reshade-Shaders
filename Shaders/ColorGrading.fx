/*------------------.
| :: Description :: |
'-------------------/

    Color Grading 

    Version 1.2.1
    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adjusts brightness, saturation, contrast, and performs color grading interpolation.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 1.2.1

*/

#include "ReShade.fxh"
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))

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

/*----------------.
| :: Functions :: |
'----------------*/

float lum(float3 color)
{
    return (color.r + color.g + color.b) * 0.3333333;
}

float3 RGBtoHSL(float3 color)
{
    float3 hsl;
    float minVal = min(color.r, min(color.g, color.b));
    float maxVal = max(color.r, max(color.g, color.b));
    float delta = maxVal - minVal;

    // Luminosity (L)
    hsl.z = (maxVal + minVal) / 2.0;

    if (delta == 0.0) 
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
        if (maxVal == color.r)
            hsl.x = (color.g - color.b) / delta;
        else if (maxVal == color.g)
            hsl.x = (color.b - color.r) / delta + 2.0;
        else
            hsl.x = (color.r - color.g) / delta + 4.0;

        hsl.x /= 6.0;
        if (hsl.x < 0.0)
            hsl.x += 1.0;
    }

    return hsl;
}

float HueToRGB(float temp1, float temp2, float h)
{
    if (h < 0.0)
        h += 1.0;
    if (h > 1.0)
        h -= 1.0;
    if (h < 1.0 / 6.0)
        return temp1 + (temp2 - temp1) * 6.0 * h;
    if (h < 0.5)
        return temp2;
    if (h < 2.0 / 3.0)
        return temp1 + (temp2 - temp1) * (2.0 / 3.0 - h) * 6.0;
    return temp1;
}

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
        float temp1;
        if (hsl.z < 0.5)
            temp2 = hsl.z * (1.0 + hsl.y);
        else
            temp2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);

        temp1 = 2.0 * hsl.z - temp2;

        rgb.r = HueToRGB(temp1, temp2, hsl.x + 1.0 / 3.0);
        rgb.g = HueToRGB(temp1, temp2, hsl.x);
        rgb.b = HueToRGB(temp1, temp2, hsl.x - 1.0 / 3.0);
    }

    return rgb;
}

float3 ApplyColorGradingStyle(float3 color, int method)
{
    float3 gradedColor = color;

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
        return color * float3(0.9, 1.1, 1.2);
    
    if (method == 3) // Sepia
        return float3(dot(color, float3(0.393, 0.769, 0.189)),
                      dot(color, float3(0.349, 0.686, 0.168)),
                      dot(color, float3(0.272, 0.534, 0.131)));

    if (method == 4) // Black and White
    {
        float gray = lum(color);
        return float3(gray, gray, gray);
    }
    
    if (method == 5) // Vintage
        return color * float3(1.0, 0.9, 0.8) * 0.8 + float3(0.1, 0.05, 0.05);
    
    if (method == 6) // Vibrant
    {
        float3 hsl = RGBtoHSL(color);
        hsl.y = clamp(hsl.y * 1.5, 0.0, 1.0);
        color = HSLtoRGB(hsl);

        // Contrast adjustment
        color = (color - 0.5) * 1.1 + 0.5;

        return color;
    }
    
    if (method == 7) // Horror
        return float3(color.r * 0.5, color.g * 0.2, color.b * 0.2);

    if (method == 8) // Cine Style
        return float3(color.r * 1.1, color.g * 0.95, color.b * 0.85);

    if (method == 9) // Teal and Orange (Split-Toning)
    {
        float luminance = lum(color);
        float3 shadows = float3(0.0, 0.3, 0.3); // Teal
        float3 highlights = float3(0.8, 0.4, 0.0); // Orange
        float t = smoothstep(0.2, 0.8, luminance);
        return lerp(shadows * color, highlights * color, t);
    }

    return gradedColor;
}

float3 AdjustBrightnessContrast(float3 color, float brightness, float contrast)
{
    color += brightness;
    return ((color - 0.5) * contrast) + 0.5;
}

float3 ApplyInterpolatedColorGrading(float3 color)
{
    float3 startGrading = ApplyColorGradingStyle(color, StartColorGradingMethod);
    float3 endGrading = ApplyColorGradingStyle(color, EndColorGradingMethod);
    float3 interpolatedColor = lerp(startGrading, endGrading, GradingInterpolationFactor);
    float gray = lum(interpolatedColor);
    interpolatedColor = lerp(float3(gray, gray, gray), interpolatedColor, SaturationFactor);
    return AdjustBrightnessContrast(interpolatedColor, Brightness, Contrast);
}

float4 ColorGradingPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = GetColor(texcoord).rgb;
    
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
