/*-------------------------------------------------|
| ::                 uFakeHDR                   :: |
'--------------------------------------------------|
| Version: 1.9                                     |
| Author: Barbatos                                 |
| License: CC0                                     |
'---------------------------------------------------*/

#include "ReShade.fxh"

uniform float HDRPower <
    ui_type = "slider";
    ui_label = "HDR Power";
    ui_min = 0.1; ui_max = 4.0;
> = 1.4;

uniform int HDRExtraMode <
    ui_type = "combo";
    ui_label = "Extra Mode";
    ui_items = "None\0Multiple Exposures\0";
> = 0;

float4 FHDR(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
    float lum = dot(color, float3(0.2126, 0.7152, 0.0722));
    float sceneLum = lum * 0.5;

    float3 c = pow(color, HDRPower);

    float adjust = lerp(1.0, clamp(0.5 / (sceneLum + 0.001), 0.5, 2.0), 0.5);
    c = saturate(c * adjust);

    if (HDRExtraMode == 1)
    {
        c = sqrt(c);
    }
        
    return float4(saturate(c), 1.0);
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
