/*------------------.
| :: Description :: |
'-------------------/
UFakeHDR (version 0.2) 

Author: BarbatosBachiko
License: MIT
Original FastDeband method by CeeJay.dk

About: This shader simulates HDR effects for SDR.
  
Version 0.1
Initial implementation of FakeHDR

Version 0.2
Added tone mapping and debanding (beta)
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
    ui_items = "Reinhard\0Filmic\0ACES\0BT.709\0Logarithmic\\0"; // New Neutral option added
> = 2; // Default adjusted to ACES (index 2)

// Adaptive tone mapping toggle
uniform bool UseAdaptiveToneMapping < 
    ui_type = "checkbox";
    ui_label = "Use Adaptive Tone Mapping"; 
> = true; // Default adjusted to true

// Beta Debanding toggle
uniform bool EnableBetaDeband < 
    ui_type = "checkbox";
    ui_label = "Enable Beta Deband"; 
> = true; // Default disabled

// Debanding parameters
uniform float Diameter < 
    ui_type = "slider";
    ui_label = "Deband Diameter"; 
    ui_min = 1.0; 
    ui_max = 10.0; 
> = 2.0; // Default adjusted to 2.0

uniform float Strength < 
    ui_type = "slider";
    ui_label = "Deband Strength"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
> = 0.5; // Default adjusted to 0.5

uniform float Threshold < 
    ui_type = "slider";
    ui_label = "Deband Threshold"; 
    ui_min = 0.0; 
    ui_max = 255.0; 
> = 5.0; // Default adjusted to 5.0

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
float3 ReinhardToneMapping(float3 color)
{
    return color / (color + 1.0);
}

float3 FilmicToneMapping(float3 color)
{
    float3 mapped = (color * (color * 0.6 + 0.4)) / (color + 1.0);
    return saturate(mapped);
}

float3 ACESToneMapping(float3 color)
{
    float3 mapped = (color * (color + 0.0245786) - (color * color * 0.000093607)) / (color + 0.000009);
    return saturate(mapped);
}

float3 BTToneMapping(float3 color)
{
    float3 mapped = color * (color * 0.7 + 0.3);
    return saturate(mapped);
}

float3 LogarithmicToneMapping(float3 color)
{
    return log(color + 1.0) / log(2.0); 
}

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
float2 make_some_noise(float2 uv)
{
    return float2(sin(uv.x * 100.0), cos(uv.y * 100.0)) * 0.5; 
}

// Debanding function
float3 ImprovedDeband(float3 color, float2 texcoord, float diameter, float strength, float threshold)
{
    // Add noise based on position
    float2 noise = make_some_noise(texcoord * 100.0);
    noise = (ReShade::PixelSize * diameter) * noise - (ReShade::PixelSize * diameter * 0.5);

    // Sample color with noise
    float3 noise_color = tex2D(BackBuffer, texcoord + noise).rgb;

    // Calculate difference between sampled color and original color
    float3 diff = abs(noise_color - color);

    // Apply debanding if difference is below threshold
    if (max(max(diff.x, diff.y), diff.z) <= float(threshold * (1.0 / 255.0)))
    {
        // Blend original color with noise color
        color = lerp(color, noise_color, strength);
    }

    // Return adjusted color
    return color;
}

float3 FakeHDRPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // Sample the color from the framebuffer
    float3 color = tex2D(BackBuffer, texcoord).rgb; 

    // Simulate HDR effect by adjusting luminance
    float3 hdrColor = pow(color, HDRPower);

    // Apply selected tone mapping
    float3 toneMappedColor = ApplyToneMapping(hdrColor);

    // If debanding is enabled
    if (EnableBetaDeband)
    {
        toneMappedColor = ImprovedDeband(toneMappedColor, texcoord, Diameter, Strength, Threshold); 
    }

    // Return adjusted color for SDR
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
