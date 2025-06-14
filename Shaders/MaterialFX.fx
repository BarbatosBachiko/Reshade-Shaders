/*------------------.
| :: Description :: |
'-------------------/

    MaterialFX

    Version 1.3 
    Author: Barbatos Bachiko
    License: MIT

    About: Bits, Chromatic Aberration and Fog.

  History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.3
    + Revised
*/

#include "ReShade.fxh"
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))

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
    "Chromatic Aberration\0"
    "Fog\0";
>
= 0;

uniform float CAStrength
<
    ui_type = "slider";
    ui_label = "Chromatic Aberration Strength";
    ui_min = 0.0; ui_max = 0.050; ui_step = 0.001;
>
= 0.001;

uniform float fog_density
<
    ui_type = "slider";
    ui_label = "Fog Density";
    ui_min = 1.0; ui_max = 5.0; ui_step = 1.0;
>
= 3.0;

/*----------------.
| :: Functions :: |
'----------------*/

float4 MaterialFXPass(float4 pos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 color = GetColor(texcoord);

    if (combo == 0) // Bits
    {
        color.rgb = floor(color.rgb * 8.0) / 8.0;
    }
    else if (combo == 1) // Chromatic Aberration
    {
        float2 offsetAmount = float2(CAStrength, 0.0);
        float4 redChannel = GetColor(texcoord + offsetAmount);
        float4 blueChannel = GetColor(texcoord - offsetAmount);
        color.rgb = float3(redChannel.r, color.g, blueChannel.b);
    }
    else if (combo == 2) // Fog
    {
        float fogFactor = smoothstep(0.0, fog_density, texcoord.y);
        color.rgb = lerp(color.rgb, float3(1.0, 1.0, 1.0), fogFactor);
    }

    return float4(saturate(color.rgb), 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/
technique MaterialFX
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = MaterialFXPass;
    }
}
