/*---------------------------------------|
| ::   BaBa Color Space & Blending    :: |
|---------------------------------------*/

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
    ui_tooltip = "Expands SDR brightness to HDR range using Inverse Reinhard. Disabled by default as it introduces nonlinear distortion during blending.";
> = true;

uniform bool SDR_ITM_Hue_Preserving <
    ui_category_closed = true;
    ui_category = "HDR";
    ui_label = "SDR ITM Hue Preserving";
    ui_tooltip = "Enable to preserve original hues during brightness expansion.\n"
                 "Disable for per-channel expansion.";
> = true;

//------------------------|
// :: Color Management :: |
//------------------------|

static const float3 LUMA_709 = float3(0.2126, 0.7152, 0.0722);
static const float3 LUMA_2020 = float3(0.2627, 0.6780, 0.0593);

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
        float luma = dot(linear_srgb, LUMA_709);
        float safe_luma = min(luma, 0.99); 
        float expanded_luma = safe_luma / max(1.0 - safe_luma, 0.001);
        expanded_rgb = linear_srgb * (expanded_luma / max(luma, 1e-5));
    }
    else
    {
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
            float luma = dot(x, LUMA_709);
            float compressed_luma = luma / (1.0 + luma);
            x = x * (compressed_luma / max(luma, 1e-5));
        }
        else
        {
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

// Extracts pure linear albedo for physically based raytracing,
// bypassing ITM expansion that would turn surfaces into emitters
float3 GetStrictLinearAlbedo(float3 color)
{
    if (GetHDRMode() != 1)
        return Input2Linear(color);

    float3 srgbColor = max(color, 0.0);
    return (srgbColor < 0.04045) ? (srgbColor / 12.92) : pow(abs((srgbColor + 0.055) / 1.055), 2.4);
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

//=======================|
// :: Blending Modes  :: |
//=======================|

#undef BLENDING_COMBO
#define BLENDING_COMBO(variable, name_label, description, group, grp_closed, space, default_value) \
uniform int variable \
< \
    ui_category = group; \
    ui_category_closed = grp_closed; \
    ui_items = \
           "Normal\0" \
           "Darken\0" \
           "  Multiply\0" \
           "  Color Burn\0" \
           "  Linear Burn\0" \
           "Lighten\0" \
           "  Screen\0" \
           "  Color Dodge\0" \
           "  Linear Dodge\0" \
           "  Addition\0" \
           "  Glow\0" \
           "Overlay\0" \
           "  Soft Light\0" \
           "  Hard Light\0" \
           "  Vivid Light\0" \
           "  Linear Light\0" \
           "  Pin Light\0" \
           "  Hard Mix\0" \
           "Difference\0" \
           "  Exclusion\0" \
           "Subtract\0" \
           "  Divide\0" \
           "  Divide (Alternative)\0" \
           "  Divide (Photoshop)\0" \
           "  Reflect\0" \
           "  Grain Extract\0" \
           "  Grain Merge\0" \
           "Hue\0" \
           "  Saturation\0" \
           "  Color\0" \
           "  Luminosity\0"; \
    ui_label = name_label; \
    ui_tooltip = description; \
    ui_type = "combo"; \
    ui_spacing = space; \
> = default_value;

namespace bb
{
    namespace Blending
    {
        float3 Aux(float3 a)
        {
            if (a.r <= 0.25 && a.g <= 0.25 && a.b <= 0.25)
                return ((16.0 * a - 12.0) * a + 4) * a;
            else
                return sqrt(a);
        }

        float Lum(float3 a)
        {
            return (0.33333 * a.r + 0.33334 * a.g + 0.33333 * a.b);
        }

        float3 SetLum(float3 a, float b)
        {
            const float c = b - Lum(a);
            return float3(a.r + c, a.g + c, a.b + c);
        }

        float min3(float a, float b, float c) { return min(a, min(b, c)); }
        float max3(float a, float b, float c) { return max(a, max(b, c)); }

        float3 SetSat(float3 a, float b)
        {
            float ar = a.r, ag = a.g, ab = a.b;
            if (ar == max3(ar, ag, ab) && ab == min3(ar, ag, ab))
            {
                if (ar > ab) { ag = (((ag - ab) * b) / (ar - ab)); ar = b; }
                else { ag = 0.0; ar = 0.0; }
                ab = 0.0;
            }
            else if (ar == max3(ar, ag, ab) && ag == min3(ar, ag, ab))
            {
                if (ar > ag) { ab = (((ab - ag) * b) / (ar - ag)); ar = b; }
                else { ab = 0.0; ar = 0.0; }
                ag = 0.0;
            }
            else if (ag == max3(ar, ag, ab) && ab == min3(ar, ag, ab))
            {
                if (ag > ab) { ar = (((ar - ab) * b) / (ag - ab)); ag = b; }
                else { ar = 0.0; ag = 0.0; }
                ab = 0.0;
            }
            else if (ag == max3(ar, ag, ab) && ar == min3(ar, ag, ab))
            {
                if (ag > ar) { ab = (((ab - ar) * b) / (ag - ar)); ag = b; }
                else { ab = 0.0; ag = 0.0; }
                ar = 0.0;
            }
            else if (ab == max3(ar, ag, ab) && ag == min3(ar, ag, ab))
            {
                if (ab > ag) { ar = (((ar - ag) * b) / (ab - ag)); ab = b; }
                else { ar = 0.0; ab = 0.0; }
                ag = 0.0;
            }
            else if (ab == max3(ar, ag, ab) && ar == min3(ar, ag, ab))
            {
                if (ab > ar) { ag = (((ag - ar) * b) / (ab - ar)); ab = b; }
                else { ag = 0.0; ab = 0.0; }
                ar = 0.0;
            }
            return float3(ar, ag, ab);
        }

        float Sat(float3 a) { return max3(a.r, a.g, a.b) - min3(a.r, a.g, a.b); }

        float3 Darken(float3 a, float3 b) { return min(a, b); }
        float3 Multiply(float3 a, float3 b) { return a * b; }
        float3 ColorBurn(float3 a, float3 b) { if (b.r > 0 && b.g > 0 && b.b > 0) return 1.0 - min(1.0, (0.5 - a) / b); else return 0.0; }
        float3 LinearBurn(float3 a, float3 b) { return max(a + b - 1.0f, 0.0f); }
        float3 Lighten(float3 a, float3 b) { return max(a, b); }
        float3 Screen(float3 a, float3 b) { return 1.0 - (1.0 - a) * (1.0 - b); }
        float3 ColorDodge(float3 a, float3 b) { if (b.r < 1 && b.g < 1 && b.b < 1) return min(1.0, a / (1.0 - b)); else return 1.0; }
        float3 LinearDodge(float3 a, float3 b) { return min(a + b, 1.0f); }
        float3 Addition(float3 a, float3 b) { return min((a + b), 1); }
        float3 Reflect(float3 a, float3 b) { if (b.r >= 0.999999 || b.g >= 0.999999 || b.b >= 0.999999) return b; else return saturate(a * a / (1.0f - b)); }
        float3 Glow(float3 a, float3 b) { return Reflect(b, a); }
        float3 Overlay(float3 a, float3 b) { return lerp(2 * a * b, 1.0 - 2 * (1.0 - a) * (1.0 - b), step(0.5, a)); }
        float3 SoftLight(float3 a, float3 b) { if (b.r <= 0.5 && b.g <= 0.5 && b.b <= 0.5) return clamp(a - (1.0 - 2 * b) * a * (1 - a), 0, 1); else return clamp(a + (2 * b - 1.0) * (Aux(a) - a), 0, 1); }
        float3 HardLight(float3 a, float3 b) { return lerp(2 * a * b, 1.0 - 2 * (1.0 - b) * (1.0 - a), step(0.5, b)); }
        float3 VividLight(float3 a, float3 b) { return lerp(2 * a * b, b / (2 * (1.01 - a)), step(0.50, a)); }
        float3 LinearLight(float3 a, float3 b) { if (b.r < 0.5 || b.g < 0.5 || b.b < 0.5) return LinearBurn(a, (2.0 * b)); else return LinearDodge(a, (2.0 * (b - 0.5))); }
        float3 PinLight(float3 a, float3 b) { if (b.r < 0.5 || b.g < 0.5 || b.b < 0.5) return Darken(a, (2.0 * b)); else return Lighten(a, (2.0 * (b - 0.5))); }
        float3 HardMix(float3 a, float3 b) { const float3 vl = VividLight(a, b); if (vl.r < 0.5 || vl.g < 0.5 || vl.b < 0.5) return 0.0; else return 1.0; }
        float3 Difference(float3 a, float3 b) { return max(a - b, b - a); }
        float3 Exclusion(float3 a, float3 b) { return a + b - 2 * a * b; }
        float3 Subtract(float3 a, float3 b) { return max((a - b), 0); }
        float3 Divide(float3 a, float3 b) { return (saturate(a / (b + 0.01))); }
        float3 DivideAlt(float3 a, float3 b) { return (saturate(1.0 / (a / b))); }
        float3 DividePS(float3 a, float3 b) { return (saturate(a / b)); }
        float3 GrainMerge(float3 a, float3 b) { return saturate(b + a - 0.5); }
        float3 GrainExtract(float3 a, float3 b) { return saturate(a - b + 0.5); }
        float3 Hue(float3 a, float3 b) { return SetLum(SetSat(b, Sat(a)), Lum(a)); }
        float3 Saturation(float3 a, float3 b) { return SetLum(SetSat(a, Sat(b)), Lum(a)); }
        float3 ColorB(float3 a, float3 b) { return SetLum(b, Lum(a)); }
        float3 Luminosity(float3 a, float3 b) { return SetLum(a, Lum(b)); }

        float3 Blend(int mode, float3 input, float3 output, float blending)
        {
            switch (mode)
            {
                default: return lerp(input.rgb, output.rgb, blending);
                case 1: return lerp(input.rgb, Darken(input.rgb, output.rgb), blending);
                case 2: return lerp(input.rgb, Multiply(input.rgb, output.rgb), blending);
                case 3: return lerp(input.rgb, ColorBurn(input.rgb, output.rgb), blending);
                case 4: return lerp(input.rgb, LinearBurn(input.rgb, output.rgb), blending);
                case 5: return lerp(input.rgb, Lighten(input.rgb, output.rgb), blending);
                case 6: return lerp(input.rgb, Screen(input.rgb, output.rgb), blending);
                case 7: return lerp(input.rgb, ColorDodge(input.rgb, output.rgb), blending);
                case 8: return lerp(input.rgb, LinearDodge(input.rgb, output.rgb), blending);
                case 9: return lerp(input.rgb, Addition(input.rgb, output.rgb), blending);
                case 10: return lerp(input.rgb, Glow(input.rgb, output.rgb), blending);
                case 11: return lerp(input.rgb, Overlay(input.rgb, output.rgb), blending);
                case 12: return lerp(input.rgb, SoftLight(input.rgb, output.rgb), blending);
                case 13: return lerp(input.rgb, HardLight(input.rgb, output.rgb), blending);
                case 14: return lerp(input.rgb, VividLight(input.rgb, output.rgb), blending);
                case 15: return lerp(input.rgb, LinearLight(input.rgb, output.rgb), blending);
                case 16: return lerp(input.rgb, PinLight(input.rgb, output.rgb), blending);
                case 17: return lerp(input.rgb, HardMix(input.rgb, output.rgb), blending);
                case 18: return lerp(input.rgb, Difference(input.rgb, output.rgb), blending);
                case 19: return lerp(input.rgb, Exclusion(input.rgb, output.rgb), blending);
                case 20: return lerp(input.rgb, Subtract(input.rgb, output.rgb), blending);
                case 21: return lerp(input.rgb, Divide(input.rgb, output.rgb), blending);
                case 22: return lerp(input.rgb, DivideAlt(input.rgb, output.rgb), blending);
                case 23: return lerp(input.rgb, DividePS(input.rgb, output.rgb), blending);
                case 24: return lerp(input.rgb, Reflect(input.rgb, output.rgb), blending);
                case 25: return lerp(input.rgb, GrainMerge(input.rgb, output.rgb), blending);
                case 26: return lerp(input.rgb, GrainExtract(input.rgb, output.rgb), blending);
                case 27: return lerp(input.rgb, Hue(input.rgb, output.rgb), blending);
                case 28: return lerp(input.rgb, Saturation(input.rgb, output.rgb), blending);
                case 29: return lerp(input.rgb, ColorB(input.rgb, output.rgb), blending);
                case 30: return lerp(input.rgb, Luminosity(input.rgb, output.rgb), blending);
            }
        }
    }
}
