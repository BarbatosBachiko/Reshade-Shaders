/*-------------------------------------------------|
| ::                 uFakeHDR                   :: |
| Version: 2.0                                     |
| Author: Barbatos                                 |
| License: CC0                                     |
'-------------------------------------------------*/

#include "ReShade.fxh"

static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

uniform float INTENSITY <
    ui_type = "slider";
    ui_label = "Intensity";
    ui_tooltip = "Adjusts the strength of the HDR effect.";
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;

float4 FHDR(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
    float lum = dot(color, LUMA);
    float adjust = 0.5 * (1.0 + min(rcp(lum + 0.002), 2.0));
    float3 hdrColor = pow(abs(color), 1.4) * adjust; 
    color = lerp(color, hdrColor, INTENSITY);

    return float4(saturate(color), 1.0);
}

technique UFakeHDR
<
    ui_label = "uFakeHDR";
    ui_tooltip = "Make the game less gray ;)";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = FHDR;
    }
}
