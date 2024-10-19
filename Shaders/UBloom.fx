/*------------------.
| :: Description :: |
'-------------------/

	UBloom (version 0.1)

	Author: BarbatosBachiko

	About: This shader creates a simple bloom effect around high brightness objects.

	Ideas for future improvement:
    * Add more configuration options.
    * Implement different bloom modes.
	
	Version 0.1
    * Create simple bloom

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float Threshold < ui_type = "slider";ui_label = "Threshold"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; > = 0.7;
uniform float Intensity < ui_type = "slider";ui_label = "Intensity"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; > = 1.0;
uniform float BloomWidth < ui_type = "slider";ui_label = "Bloom Width"; ui_min = 0.0; ui_max = 10.0; ui_step = 0.1; > = 3.0;

/*----------------.
| :: Functions :: |
'----------------*/

float3 UBloomPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float brightness = dot(color, float3(0.2126, 0.7152, 0.0722));

    float3 bloom = float3(0, 0, 0);
    if (brightness > Threshold)
    {
        bloom += tex2D(ReShade::BackBuffer, texcoord + float2(BloomWidth, 0) * 0.01).rgb;
        bloom += tex2D(ReShade::BackBuffer, texcoord + float2(-BloomWidth, 0) * 0.01).rgb;
        bloom += tex2D(ReShade::BackBuffer, texcoord + float2(0, BloomWidth) * 0.01).rgb;
        bloom += tex2D(ReShade::BackBuffer, texcoord + float2(0, -BloomWidth) * 0.01).rgb;
        bloom = bloom / 4.0;
        bloom *= Intensity;
    }
    return saturate(color + bloom);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique UBloom
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = UBloomPS;
    }
}
