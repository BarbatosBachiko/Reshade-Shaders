/**
  Simple Color
  by Barbatos Bachiko

  This shader adjust brightness, saturation, and contrast.
 */

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

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

// Main shader function
float3 SimpleColorPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // Sample color from the framebuffer
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // Apply brightness adjustment
    color += Brightness;

    // Saturation adjustment
    float3 gray = dot(color, float3(0.299, 0.587, 0.114));
    color = lerp(gray.xxx, color, 1.0 + Saturation);

    // Contrast adjustment
    float3 mid = 0.5;
    color = lerp(mid, color, 1.0 + Contrast);

    // Return the adjusted color
    return saturate(color);
}

technique SimpleColor
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = SimpleColorPass;
    }
}
