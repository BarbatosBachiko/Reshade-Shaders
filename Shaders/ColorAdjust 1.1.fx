/*------------------.
| :: Description :: |
'-------------------/

	ColorAdjust (version 1.1)

	Author: BarbatosBachiko

	About:
	This shader adjusts brightness, saturation, and contrast.

	Ideas for future improvement:
	* Include a color filter mode (e.g., sepia, black and white)
	
	Version 1.0
	* Simple Color shader with controls for brightness, saturation, and contrast.
  Version 1.1
  * Bug fixes

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

// Brightness slider configuration
uniform float Brightness
<
	ui_type = "slider";
	ui_label = "Brightness";
	ui_min = -1.0;
	ui_max = 1.0;
>
= 0.0;

// Saturation slider configuration
uniform float Saturation
<
	ui_type = "slider";
	ui_label = "Saturation";
	ui_min = -1.0;
	ui_max = 1.0;
>
= 0.0;

// Contrast slider configuration
uniform float Contrast
<
	ui_type = "slider";
	ui_label = "Contrast";
	ui_min = -1.0;
	ui_max = 1.0;
>
= 0.0;

/*----------------.
| :: Functions :: |
'----------------*/

float3 ColorAdjustPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // Sample color from the framebuffer
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // Apply brightness adjustment
    color += Brightness;

    // Saturation adjustment
    float3 gray = dot(color, float3(0.299, 0.587, 0.114));
    color = lerp(gray.xxx, color, 1.0 + Saturation);

    // Contrast adjustment
    float3 mid = 0.5; // Middle point for contrast adjustment
    color = (color - mid) * (1.0 + Contrast) + mid; // Adjust contrast

    // Return the adjusted color
    return saturate(color);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique ColorAdjust
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = ColorAdjustPass;
    }
}
