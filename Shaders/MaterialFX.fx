/*------------------.
| :: Description :: |
'-------------------/

	MaterialFX (version 1.0)

	Author: Barbatos Bachiko
	License: MIT

	About:
	A post-processing shader with options for Bits, Initial Blur, Chromatic Aberration, Convolution, Fog, Outline, and Pixelate.
	
	Ideas for future improvement:
	* Add more material options.
	* Improve the quality of effects.
	
	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 1.0
	* Initial release

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int combo
<
    ui_type = "combo";
    ui_label = "Material";
    ui_tooltip = "Choose an effect";
    ui_items = 
    "Bits\0"
    "Blur\0"
    "Chromatic Aberration\0"
    "Convolution\0"
    "Fog\0"
    "Outline\0"
    "Pixelate\0";
>
= 0; // Default value for selected effect

uniform float pixelate_amount
<
    ui_type = "slider";
    ui_label = "Pixelate Intensity";
    ui_tooltip = "Adjust the intensity of the pixelate effect";
    ui_min = 10.0; ui_max = 500.0; ui_step = 1.0;
>
= 500.0; // Default value for pixelation intensity

uniform float chromatic_aberration_strength
<
    ui_type = "slider";
    ui_label = "Chromatic Aberration Strength";
    ui_tooltip = "Adjust the strength of chromatic aberration effect";
    ui_min = 0.0; ui_max = 0.050; ui_step = 0.001;
>
= 0.001; // Default value for chromatic aberration strength

uniform float fog_density
<
    ui_type = "slider";
    ui_label = "Fog Density";
    ui_tooltip = "Adjust the density of the fog effect";
    ui_min = 1.0; ui_max = 5.0; ui_step = 1.0;
>
= 5.0;

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

float4 MaterialFXPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 color = tex2D(BackBuffer, texcoord);

    switch (combo)
    {
        case 0: // Bits
            color.rgb = floor(color.rgb * 8.0) / 8.0; // Reduces color depth
            break;

        case 1: // Blur
            {
                float blurAmount = 0.005; // Adjust for intensity
                color = (
                    tex2D(BackBuffer, texcoord + float2(-blurAmount, -blurAmount)) +
                    tex2D(BackBuffer, texcoord + float2(blurAmount, -blurAmount)) +
                    tex2D(BackBuffer, texcoord + float2(-blurAmount, blurAmount)) +
                    tex2D(BackBuffer, texcoord + float2(blurAmount, blurAmount)) +
                    tex2D(BackBuffer, texcoord)
                ) / 5.0; // Average the colors
            }
            break;

        case 2: // Chromatic Aberration
            {
                float2 offsetAmount = float2(chromatic_aberration_strength, 0.0);
                float4 redChannel = tex2D(BackBuffer, texcoord + offsetAmount); // Red channel
                float4 greenChannel = tex2D(BackBuffer, texcoord); // Green channel
                float4 blueChannel = tex2D(BackBuffer, texcoord - offsetAmount); // Blue channel
                color = float4(redChannel.r, greenChannel.g, blueChannel.b, 1.0); // Combine channels
            }
            break;

        case 3: // Convolution
            {
                float3 kernel[9] =
                {
                    float3(-1, -1, -1), float3(-1, 9, -1), float3(-1, -1, -1),
                    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0),
                    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0)
                };
                float3 result = float3(0, 0, 0);
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        result += tex2D(BackBuffer, texcoord + float2(i, j) * 0.001).rgb * kernel[(i + 1) * 3 + (j + 1)];
                    }
                }
                color.rgb = result; // Assign the result to color
            }
            break;

        case 4: // Fog
            {
                float fogFactor = smoothstep(0.0, fog_density, texcoord.y); // Density of the fog effect
                color.rgb = lerp(color.rgb, float3(1.0, 1.0, 1.0), fogFactor); // Mix with white
            }
            break;

        case 5: // Outline
            {
                float edgeThickness = 1.0; // Outline thickness
                float2 offset = edgeThickness / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
                float4 left = tex2D(BackBuffer, texcoord - float2(offset.x, 0));
                float4 right = tex2D(BackBuffer, texcoord + float2(offset.x, 0));
                float4 top = tex2D(BackBuffer, texcoord - float2(0, offset.y));
                float4 bottom = tex2D(BackBuffer, texcoord + float2(0, offset.y));
                if (length(color.rgb - left.rgb) > 0.1 || length(color.rgb - right.rgb) > 0.1 ||
                    length(color.rgb - top.rgb) > 0.1 || length(color.rgb - bottom.rgb) > 0.1)
                {
                    color.rgb = float3(0.0, 0.0, 0.0); // Black color for outline
                }
            }
            break;

        case 6: // Pixelate
            {
                float2 pixelSize = 1.0 / pixelate_amount; // Pixel size based on intensity
                color = tex2D(BackBuffer, floor(texcoord / pixelSize) * pixelSize); // Pixelate effect
            }
            break;
    }

    // Return the result
    return float4(saturate(color.rgb), 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique MaterialFX
{
    pass
    {
        VertexShader = PostProcessVS; // in ReShade.fxh
        PixelShader = MaterialFXPS;
    }
}