/*------------------------------------------|
| ::      Barbatos Color Space           :: |
|-----------------------------------------*/
#pragma once

#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 0
#endif

//----------|
// :: UI :: |
//----------|

uniform int HDR_Input_Format <
    ui_category_closed = true;
    ui_category = "HDR";
    ui_label = "Input Format";
    ui_tooltip = "Select the color space of the game.\n"
                 "Auto = Detect automatically (Recommended)\n"
                 "SDR/HDR formats = Force specific color space\n"
                 "Raw = No conversion applied";
    ui_type = "combo";
    ui_items = "Auto\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0Raw (No Conversion)\0";
> = 0;

uniform float HDR_Peak_Nits <
    ui_category_closed = true;
    ui_category = "HDR";
    ui_label = "HDR Peak Brightness (Nits)";
    ui_tooltip = "Set this to match your monitor's maximum HDR brightness capabilities (e.g., 400 for DisplayHDR 400, 1000 for high-end HDR). Only affects HDR formats.";
    ui_type = "drag";
    ui_min = 400.0; ui_max = 10000.0; ui_step = 10.0;
> = 1000.0;

uniform bool SDR_Enable_ITM <
    ui_category_closed = true;
    ui_category = "HDR";
    ui_label = "Enable SDR Inverse Tonemapping";
    ui_tooltip = "Expands SDR brightness to HDR range using Inverse Reinhard.";
> = true;

uniform bool SDR_ITM_Hue_Preserving <
    ui_category_closed = true;
    ui_category = "HDR";
    ui_label = "SDR ITM Hue Preserving";
    ui_tooltip = "Enable to preserve original hues during brightness expansion.\n"
                 "Disable for per-channel expansion.";
> = false;

//------------------------|
// :: Color Management :: |
//------------------------|

static const float3 LUMA_709 = float3(0.2126, 0.7152, 0.0722);
static const float3 LUMA_2020 = float3(0.2627, 0.6780, 0.0593);

// PQ Constants 
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

int GetHDRMode()
{
    if (HDR_Input_Format != 0)
        return HDR_Input_Format;

#if BUFFER_COLOR_SPACE == 1
    return 1;
#elif BUFFER_COLOR_SPACE == 2
    return 2;
#elif BUFFER_COLOR_SPACE == 3
    return 3;
#else
    return 1;
#endif
}

float3 PQ2Linear(float3 color)
{
    float3 val = max(pow(abs(color), 1.0 / PQ_M2) - PQ_C1, 0.0);
    float3 den = PQ_C2 - PQ_C3 * pow(abs(color), 1.0 / PQ_M2);
    float3 linearHdr = pow(abs(val / den), 1.0 / PQ_M1);

    return linearHdr * (10000.0 / HDR_Peak_Nits);
}

float3 Linear2PQ(float3 linearColor)
{
    float3 Y = max(0.0, linearColor * (HDR_Peak_Nits / 10000.0));
    float3 num = PQ_C1 + PQ_C2 * pow(Y, PQ_M1);
    float3 den = 1.0 + PQ_C3 * pow(Y, PQ_M1);
    
    return pow(num / den, PQ_M2);
}

float3 sRGB2Linear(float3 x)
{
    float3 linear_srgb = (x < 0.04045) ? (x / 12.92) : pow(abs((x + 0.055) / 1.055), 2.4);

    if (!SDR_Enable_ITM)
        return linear_srgb;

    float3 expanded_rgb;
    if (SDR_ITM_Hue_Preserving)
    {
        // Hue-Preserving Inverse Reinhard
        float luma = dot(linear_srgb, LUMA_709);
        float safe_luma = min(luma, 0.99); 
        float expanded_luma = safe_luma / max(1.0 - safe_luma, 0.001);
        
        expanded_rgb = linear_srgb * (expanded_luma / max(luma, 1e-5));
    }
    else
    {
        // Per-channel Inverse Reinhard
        float3 safe_rgb = min(linear_srgb, 0.99);
        expanded_rgb = (safe_rgb / max(1.0 - safe_rgb, 0.001));
    }

    return expanded_rgb;
}

float3 Linear2sRGB(float3 x)
{
    x = max(x, 0.0);

    if (SDR_Enable_ITM)
    {
        if (SDR_ITM_Hue_Preserving)
        {
            // Hue-Preserving Reinhard
            float luma = dot(x, LUMA_709);
            float compressed_luma = luma / (1.0 + luma);
            
            x = x * (compressed_luma / max(luma, 1e-5));
        }
        else
        {
            // Per-channel Reinhard
            x = x / (1.0 + x);
        }
    }
    
    return (x < 0.0031308) ? (12.92 * x) : (1.055 * pow(abs(x), 1.0 / 2.4) - 0.055);
}

float3 Input2Linear(float3 color)
{
    int mode = GetHDRMode();

    if (mode == 4)
        return color;
    else if (mode == 2)
        return color * (80.0 / HDR_Peak_Nits);
    else if (mode == 3)
        return PQ2Linear(color);
    else
        return sRGB2Linear(color);
}

float3 Linear2Output(float3 color)
{
    int mode = GetHDRMode();

    if (mode == 4)
        return color;
    else if (mode == 2)
        return color * (HDR_Peak_Nits / 80.0);
    else if (mode == 3)
        return Linear2PQ(color);
    else
        return Linear2sRGB(color);
}

float GetLuminance(float3 color)
{
    int mode = GetHDRMode();
    float3 lumaCoeff = (mode == 2 || mode == 3) ? LUMA_2020 : LUMA_709;
    return dot(color, lumaCoeff);
}

float3 KelvinToRGB(float k)
{
    float3 color;

    k = clamp(k, 1000.0, 40000.0) / 100.0;

    if (k <= 66.0)
    {
        color.r = 255.0;
        color.g = 99.4708025861 * log(k) - 161.1195681661;
        
        if (k <= 19.0)
            color.b = 0.0;
        else
            color.b = 138.5177312231 * log(k - 10.0) - 305.0447927307;
    }
    else
    {
        color.r = 329.698727446 * pow(k - 60.0, -0.1332047592);
        color.g = 288.1221695283 * pow(k - 60.0, -0.0755148492);
        color.b = 255.0;
    }
    
    return saturate(color / 255.0);
}
    
float3 HSVToRGB(float3 c)
{
    const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

float3 RGBToYCoCg(float3 c)
{
    return float3(
    0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
    0.5 * c.r - 0.5 * c.b,
    -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

float3 YCoCgToRGB(float3 c)
{
    return float3(
    c.x + c.y - c.z,
    c.x + c.z,
    c.x - c.y - c.z
    );
}
