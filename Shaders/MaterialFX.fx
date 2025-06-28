/*------------------.
| :: Description :: |
'-------------------/
    MaterialFX

    Version 1.4
    Author: Barbatos Bachiko
    License: MIT

    About: Bits, Chromatic Aberration, Fog, Film Grain, Vignette, and Color Grading.

     History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    Version 1.4
    * Added Film Grain effect 
    * Added Vignette effect
*/

#include "ReShade.fxh"
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))

/*---------------.
| :: Settings :: |
'---------------*/

// Main Effect Selection
uniform int combo
<
    ui_category = "Effect Selection";
    ui_type = "combo";
    ui_label = "Material Effect";
    ui_tooltip = "Choose a primary visual effect";
    ui_items =
    "Bits\0"
    "Chromatic Aberration\0"
    "Film Grain\0"
    "Vignette\0";
>
= 0;

uniform float effect_intensity
<
    ui_category = "Effect Selection";
    ui_type = "slider";
    ui_label = "Effect Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
>
= 1.0;

// Bits Settings
uniform int bits_levels
<
    ui_category = "Bits Settings";
    ui_type = "slider";
    ui_label = "Bit Levels";
    ui_tooltip = "Number of color levels per channel (lower = more posterized)";
    ui_min = 2; ui_max = 32; ui_step = 1;
>
= 8;

// Chromatic Aberration Settings
uniform float CAStrength
<
    ui_category = "Chromatic Aberration";
    ui_type = "slider";
    ui_label = "Horizontal Strength";
    ui_min = 0.0; ui_max = 0.050; ui_step = 0.001;
>
= 0.001;

uniform float CAVertical
<
    ui_category = "Chromatic Aberration";
    ui_type = "slider";
    ui_label = "Vertical Strength";
    ui_min = 0.0; ui_max = 0.050; ui_step = 0.001;
>
= 0.0;

uniform bool CA_radial
<
    ui_category = "Chromatic Aberration";
    ui_label = "Radial Mode";
    ui_tooltip = "Apply chromatic aberration radially from center";
>
= false;

// Film Grain Settings
uniform float grain_intensity
<
    ui_category = "Film Grain";
    ui_type = "slider";
    ui_label = "Grain Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
>
= 1.0;

uniform float grain_size
<
    ui_category = "Film Grain";
    ui_type = "slider";
    ui_label = "Grain Size";
    ui_tooltip = "Size of individual grain particles";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.1;
>
= 1.0;

uniform bool grain_colored
<
    ui_category = "Film Grain";
    ui_label = "Colored Grain";
>
= false;

uniform int grain_noise_type
<
    ui_category = "Film Grain";
    ui_type = "combo";
    ui_label = "Noise Type";
    ui_tooltip = "Type of noise algorithm to use";
    ui_items = "Simple Random\0IGN (Interleaved Gradient)\0Procedural Blue Noise\0";
>
= 1;

uniform float grain_hold_length
<
    ui_category = "Film Grain";
    ui_type = "slider";
    ui_label = "Hold Length";
    ui_tooltip = "How many frames to hold the same noise pattern";
    ui_min = 1.0; ui_max = 120.0; ui_step = 1.0;
>
= 48.0;

// Vignette Settings
uniform float vignette_radius
<
    ui_category = "Vignette";
    ui_type = "slider";
    ui_label = "Vignette Radius";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.01;
>
= 0.7;

uniform float vignette_strength
<
    ui_category = "Vignette";
    ui_type = "slider";
    ui_label = "Vignette Strength";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
>
= 0.5;

uniform float3 vignette_color
<
    ui_category = "Vignette";
    ui_type = "color";
    ui_label = "Vignette Color";
>
= float3(0.0, 0.0, 0.0);

uniform int FRAME_COUNT < source = "framecount"; >;

/*----------------.
| :: Functions :: |
'----------------*/

float lum(float3 color)
{
    return (color.r + color.g + color.b) * 0.3333333;
}

float IGN(float2 n)
{
    float f = 0.06711056 * n.x + 0.00583715 * n.y;
    return frac(52.9829189 * frac(f));
}

float3 IGN3dts(float2 texcoord, float HL)
{
    float3 OutColor;
    float2 seed = texcoord * BUFFER_SCREEN_SIZE + (FRAME_COUNT % HL) * 5.588238;
    OutColor.r = IGN(seed);
    OutColor.g = IGN(seed + 91.534651 + 189.6854);
    OutColor.b = IGN(seed + 167.28222 + 281.9874);
    return OutColor;
}

float3 ProceduralBN3dts(float2 texcoord, float HL)
{
    float2 uv = texcoord * BUFFER_SCREEN_SIZE;
    float frame = FRAME_COUNT % HL;
    float3 noise = float3(0, 0, 0);
    float2 p = uv * 0.1 + frame * 0.01;

    for (int i = 0; i < 4; i++)
    {
        float scale = pow(2.0, i);
        float2 coord = p * scale + frame * (i + 1) * 0.1;
        noise.r += IGN(coord) / scale;
        noise.g += IGN(coord + 127.1) / scale;
        noise.b += IGN(coord + 311.7) / scale;
    }

    return frac(noise);
}

float random(float2 coords, float time)
{
    return frac(sin(dot(coords + time, float2(12.9898, 78.233))) * 43758.5453);
}

float random(float2 coords)
{
    return random(coords, 0.0);
}

float4 MaterialFXPass(float4 pos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 color = GetColor(texcoord);
    float4 original = color;
    
    if (combo == 0) // Bits
    {
        float levels = float(bits_levels);
        color.rgb = floor(color.rgb * levels) / levels;
    }
    else if (combo == 1) // Chromatic Aberration
    {
        if (CA_radial)
        {
            float2 center = float2(0.5, 0.5);
            float2 offset = texcoord - center;
            float distance = length(offset);
            float2 direction = normalize(offset);
            
            float2 redOffset = direction * CAStrength * distance;
            float2 blueOffset = direction * -CAStrength * distance;
            
            float4 redChannel = GetColor(texcoord + redOffset);
            float4 blueChannel = GetColor(texcoord + blueOffset);
            color.rgb = float3(redChannel.r, color.g, blueChannel.b);
        }
        else
        {
            float2 offsetAmount = float2(CAStrength, CAVertical);
            float4 redChannel = GetColor(texcoord + offsetAmount);
            float4 blueChannel = GetColor(texcoord - offsetAmount);
            color.rgb = float3(redChannel.r, color.g, blueChannel.b);
        }
    }
    else if (combo == 2) // Film Grain
    {
        float2 grainCoord = texcoord * BUFFER_SCREEN_SIZE / (grain_size * 100.0);
        float3 grainColor = float3(0, 0, 0);
        
        if (grain_noise_type == 0) // Simple Random
        {
            float time = FRAME_COUNT % grain_hold_length;
            float grain = random(grainCoord, time) - 0.5;
            
            if (grain_colored)
            {
                grainColor = float3(
                    random(grainCoord + float2(1.0, 0.0), time),
                    random(grainCoord + float2(0.0, 1.0), time),
                    random(grainCoord + float2(1.0, 1.0), time)
                ) - 0.5;
            }
            else
            {
                grainColor = float3(grain, grain, grain);
            }
        }
        else if (grain_noise_type == 1) // IGN (Interleaved Gradient)
        {
            float3 noise = IGN3dts(texcoord, grain_hold_length) - 0.5;
            
            if (grain_colored)
            {
                grainColor = noise;
            }
            else
            {
                float luminance = lum(noise);
                grainColor = float3(luminance, luminance, luminance);
            }
        }
        else if (grain_noise_type == 2) // Procedural Blue Noise
        {
            float3 noise = ProceduralBN3dts(texcoord, grain_hold_length) - 0.5;
            
            if (grain_colored)
            {
                grainColor = noise;
            }
            else
            {
                float luminance = lum(noise);
                grainColor = float3(luminance, luminance, luminance);
            }
        }
        
        color.rgb += grainColor * grain_intensity * 0.1;
    }
    else if (combo == 3) // Vignette
    {
        float2 center = float2(0.5, 0.5);
        float distance = length(texcoord - center);
        float vignette = smoothstep(vignette_radius, vignette_radius - 0.3, distance);
        vignette = lerp(1.0, vignette, vignette_strength);
        color.rgb = lerp(vignette_color, color.rgb, vignette);
    }

    color.rgb = lerp(original.rgb, color.rgb, effect_intensity);
    return float4(saturate(color.rgb), 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/
technique MaterialFX
<
    ui_tooltip = "Bits, Chromatic Aberration, Film Grain and Vignette";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = MaterialFXPass;
    }
}
