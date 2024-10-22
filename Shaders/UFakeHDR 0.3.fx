/*------------------.
| :: Description :: |
'-------------------/
UFakeHDR (version 0.3)

Author: BarbatosBachiko
License: MIT

About: This shader simulates HDR effects for SDR.
  
Version 0.1
Initial implementation of FakeHDR

Version 0.2
Added tone mapping and debanding (beta)

Version 0.3
Add Color Grading, Add Bloom Effect, Add Dithering, Remove Debanding.

*/

/*---------------.
| :: Includes :: |
'---------------*/
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/
// HDR Power configuration
uniform float HDRPower < 
    ui_type = "slider";
    ui_label = "HDR Power"; 
    ui_min = 1.0; 
    ui_max = 4.0; 
> = 1.150; // Default adjusted to 1.150

// Tone mapping method selection
uniform int ToneMappingMethod < 
    ui_type = "combo";
    ui_label = "Tone Mapping Method"; 
    ui_items = "Reinhard (bad)\0Filmic\0ACES\0BT.709\0Logarithmic\\0";
> = 1;

// Color grading method selection
uniform int ColorGradingMethod < 
    ui_type = "combo";
    ui_label = "Color Grading Method"; 
    ui_items = "Neutral\0Warm\0Cool\0Sepia\0Black & White\0Vintage\0Vibrant\0Horror\0"; 
> = 0;

// Adaptive tone mapping toggle
uniform bool UseAdaptiveToneMapping < 
    ui_type = "checkbox";
    ui_label = "Use Adaptive Tone Mapping"; 
> = true; 

// Dithering toggle
uniform bool EnableDithering < 
    ui_type = "checkbox";
    ui_label = "Enable Dithering"; 
> = false; 

// Dithering strength configuration
uniform float DitherStrength < 
    ui_type = "slider";
    ui_label = "Dither Strength"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
> = 0.05; 

// Bloom Effect
uniform bool EnableBloom < 
    ui_type = "checkbox";
    ui_label = "Enable Bloom"; 
> = false;

// Bloom Strength
uniform float BloomStrength < 
    ui_type = "slider";
    ui_label = "Bloom Strength"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
> = 0.200; // Força padrão do Bloom

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

// Reinhard Tone Mapping
float3 ReinhardToneMapping(float3 color)
{
// Defining constants
    const float a = 0.18; // Desired average brightness
    const float burn = 2.0; // Sets the brightness that will be mapped to white
    const float maxLum = 1.0; // Maximum luminance

// Calculate the average luminance of the color
    float luminance = dot(color, float3(0.2126, 0.7152, 0.0722)); // Luminance
    luminance = max(luminance, 0.0001);

// Calculate the normalized luminance
   float normalizedLuminance = luminance / (a * maxLum); // Normalize to the desired average brightness

    float3 mapped = color * (normalizedLuminance / (normalizedLuminance + 1.0));

    return saturate(mapped);
}

// Filmic Tone Mapping
float3 FilmicToneMapping(float3 color)
{
    float3 mapped = (color * (color * 0.6 + 0.4)) / (color + 0.6);
    return saturate(mapped * 1.5);
}

// ACES Tone Mapping
float3 ACESToneMapping(float3 color)
{
    float3 mapped = (color * (color + 0.0245786) - (color * color * 0.000093607)) / (color + 0.000009);
    return saturate(mapped);
}

// BT 709 Tone Mapping
float3 BTToneMapping(float3 color)
{
    float3 mapped = color * (color * 0.7 + 0.3);
    return saturate(mapped);
}

float3 LogarithmicToneMapping(float3 color)
{
    return log(color + 1.0) / log(2.0);
}

//Apply ToneMaaping
float3 ApplyToneMapping(float3 color)
{
    if (UseAdaptiveToneMapping)
    {
        switch (ToneMappingMethod)
        {
            case 0: // Reinhard
                return ReinhardToneMapping(color);
            case 1: // Filmic
                return FilmicToneMapping(color);
            case 2: // ACES
                return ACESToneMapping(color);
            case 3: // BT.709
                return BTToneMapping(color);
            case 4: // Logarithmic
                return LogarithmicToneMapping(color);
            case 5: // Neutral
            default:
                return color;
        }
    }
    return color; // No tone mapping
}

// Function to generate noise
float make_noise(float2 uv)
{
    return frac(sin(dot(uv.xy, float2(12.9898, 78.233))) * 43758.5453); // Perlin-like noise
}

// Dithering
float3 ApplyDithering(float3 color, float2 texcoord)
{
    // Generate noise based on texture coordinates
    float noise = make_noise(texcoord * 10.0); // Scale noise by texcoord
    // Apply dithering effect
    color += (noise - 0.5) * DitherStrength; // DitherStrength
    return saturate(color);
}

// Color Grading
float3 ApplyColorGrading(float3 color)
{
    switch (ColorGradingMethod)
    {
        case 0: // Neutral
            return color; // No change
        case 1: // Warm
            return color * float3(1.2, 1.1, 0.9); // Increases red and green
        case 2: // Cool
            return color * float3(0.9, 1.1, 1.2); // Increases blue and green
        case 3: // Sepia
            return float3(dot(color, float3(0.393, 0.769, 0.189)),
                           dot(color, float3(0.349, 0.686, 0.168)),
                           dot(color, float3(0.272, 0.534, 0.131))); // Sepia effect
        case 4: // Black & White
            float gray = dot(color, float3(0.2989, 0.5870, 0.1140)); // Grayscale conversion
            return float3(gray, gray, gray);
        case 5: // Vintage
            return color * float3(1.0, 0.9, 0.8) * 0.8 + float3(0.1, 0.05, 0.05); // Faded colors
        case 6: // Vibrant
            return pow(color, float3(0.8, 1.0, 0.9)); // Boosts saturation
        case 7: // Horror
            return float3(color.r * 0.5, color.g * 0.2, color.b * 0.2); // Desaturated red tones
        default:
            return color; // No change
    }
}

// Function to apply the Bloom effect
float3 ApplyBloom(float3 color, float2 texcoord)
{
// Diffusion
    float bloomRadius = 2.0; // Diffusion radius

// Initialize the bloom variable
    float3 bloomColor = float3(0.0, 0.0, 0.0);

// Sample the texture around the current coordinate
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            float2 offset = float2(x, y) * (bloomRadius / 512.0);

// Get the color of the sample
            float3 sampleColor = tex2D(BackBuffer, texcoord + offset).rgb;

// Accumulate the luminance of the samples
            bloomColor += sampleColor;
        }
    }

// Average color of the swatches
    bloomColor /= 9.0; // Number of swatches

// Apply Bloom strength
    bloomColor *= BloomStrength;

// Return the original color mixed with Bloom
    return saturate(color + bloomColor);
}

// Main
float3 FakeHDRPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
// Sample color from framebuffer
    float3 color = tex2D(BackBuffer, texcoord).rgb;

// Simulate HDR effect by adjusting luminance
    float3 hdrColor = pow(color, HDRPower);

// Apply selected tone mapping
    float3 toneMappedColor = ApplyToneMapping(hdrColor);

// Apply Color Grading if enabled
    toneMappedColor = ApplyColorGrading(toneMappedColor);

// Apply Bloom effect if enabled
    if (EnableBloom)
    {
        toneMappedColor = ApplyBloom(toneMappedColor, texcoord);
    }

// Returns the adjusted color for SDR
    return saturate(toneMappedColor);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/
technique UFakeHDR
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = FakeHDRPass;
    }
}
