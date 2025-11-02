/*-------------------------------------------------|
| ::                 uFakeHDR                   :: |
'--------------------------------------------------|
| Version: 2.0                                     |
| Author: Barbatos                                 |
| License: CC0                                     |
'---------------------------------------------------*/
#include "ReShade.fxh"

#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define Trans 0.500

uniform float HDRPower < ui_type="slider";ui_label="HDR Power"; ui_min=0.1; ui_max=4.0; > = 1.4;
uniform int HDRExtraMode < ui_type="combo";ui_label="Extra Mode"; ui_items="None\0Multiple Exposures\0"; > = 0;

static float LUM = 0.0;

float CalculateSceneLuminance(float2 uv)
{
    float lum = dot(GetColor(float4(uv, 0, 0)).rgb,
                    float3(0.2126, 0.7152, 0.0722));
                    
    LUM = lerp(LUM, lum, Trans);
    return LUM;
}

float3 ToneMapping(float3 c, float sceneLum)
{
    float adjust = lerp(1.0, clamp(0.5 / (sceneLum + 0.001), 0.5, 2.0), Trans);
    return saturate(c * adjust);
}

float3 MultipleExposuresHDR(float3 c)
{
    return saturate(max(max(pow(c, 2.0), c), pow(c, 0.5)));
}

float4 FHDR(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float sceneLum = CalculateSceneLuminance(uv);
    float3 c = pow(tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb, HDRPower);
    c = ToneMapping(c, sceneLum);
    
    if (HDRExtraMode == 1)
        c = MultipleExposuresHDR(c);
        
    return float4(saturate(c), 1.0);
}

technique UFakeHDR
<
    ui_label = "uFakeHDR 2.0";
    ui_tooltip = "Make the game less gray ;)";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = FHDR;
    }
}
