/*------------------.
| :: Description :: |
'-------------------/
UFakeHDR (version 0.1) 

Author: BarbatosBachiko
License: MIT

About: This shader simulates HDR effects to enhance image quality by increasing dynamic range.

Ideas for future improvement:
Incorporate adaptive tone mapping. 
Add bloom effects for brighter areas.
Add Deband
  
 Version 0.1
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
> = 1.050;

/*----------------.
| :: Functions :: |
'----------------*/
float3 FakeHDRPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // Sample the color from the framebuffer
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // Apply HDR effect
    color = pow(color, HDRPower);

    // Return the adjusted color
    return saturate(color);
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
