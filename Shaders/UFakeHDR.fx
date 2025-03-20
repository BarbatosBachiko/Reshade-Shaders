/*

uFakeHDR

Version 1.4
Author: Barbatos Bachiko
License: MIT
 
About : This shader simulates HDR effects(expected byme) for SDR. 

History:
(*) Feature (+) Improvement	(x) Bugfix (-) Information (!)Compatibility

Version 1.4:
* Multiple Exposures
x Fix Bug for Directx 9
+ code reduction

*/

 //: Includes
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//: Settings
uniform float HDRPower < ui_type="slider";ui_label="HDR Power"; ui_min=1.0; ui_max=4.0; > = 1.150;
uniform int ToneMappingMethod < ui_type="combo";ui_label="Tone Mapping Method"; ui_items="Reinhard\0Filmic\0ACES\0BT.709\0Logarithmic\0Adaptive\0"; > = 5;
uniform int Extra < ui_type="combo";ui_label="Extra"; ui_items="None\0Multiple Exposures\0"; > = 0;
uniform bool EnableDithering < ui_category="Dithering";ui_type="checkbox"; ui_label="Enable Dithering"; > = false;
uniform float DitherStrength < ui_category="Dithering";ui_type="slider"; ui_label="Dither Strength"; ui_min=0.0; ui_max=1.0; > = 0.05;
uniform float NoiseScale < ui_category="Dithering";ui_type="slider"; ui_label="Noise Scale"; ui_min=0.1; ui_max=10.0; > = 1.0;
uniform float NoiseSeed < ui_category="Dithering";ui_type="slider"; ui_label="Noise Seed"; ui_min=1.0; ui_max=100000.0; > = 43758.5453;
uniform float Luminance < ui_category="Luminance (only for Adaptive)";ui_type="slider"; ui_label="Luminance"; ui_min=0.01; ui_max=1.0; > = 0.1;

//: Textures
texture FakeHDRTex
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler sFakeHDR
{
    Texture = FakeHDRTex;
};

//: Functions
static float lastSceneLuminance = 0.0;

float CalculateSceneLuminance(float2 uv)
{
    float lum = dot(tex2Dlod(sFakeHDR, float4(uv, 0, 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    lastSceneLuminance = lerp(lastSceneLuminance, lum, Luminance);
    return lastSceneLuminance;
}

float3 ReinhardToneMapping(float3 c)
{
    float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)) * 1.2, 0.0001);
    float nLum = clamp(lum / (0.25 * 1.0), 0.0, 1.0);
    return saturate(c * (nLum / (nLum + 1.0)) * 1.2);
}

float3 FilmicToneMapping(float3 c)
{
    return saturate((c * (c * 0.6 + 0.4)) / (c + 0.6) * 1.5);
}
float3 ACESToneMapping(float3 c)
{
    return saturate((c * (c + 0.0245786) - (c * c * 0.000093607)) / (c + 0.000009));
}
float3 BTToneMapping(float3 c)
{
    return saturate(c * (c * 0.7 + 0.3));
}
float3 LogarithmicToneMapping(float3 c)
{
    return saturate(log(c + 1.0) / log(2.0));
}
float3 AdaptiveToneMapping(float3 c, float sceneLum)
{
    float adjust = lerp(1.0, clamp(0.5 / (sceneLum + 0.001), 0.5, 2.0), Luminance);
    return saturate(c * adjust);
}
float3 MultipleExposuresHDR(float3 c)
{
    return saturate(max(max(pow(c, 2.0), c), pow(c, 0.5)));
}
float make_noise(float2 uv)
{
    return frac(sin(dot(uv * NoiseScale, float2(12.9898, 78.233))) * NoiseSeed);
}
float3 ApplyDithering(float3 c, float2 uv)
{
    return saturate(c + (make_noise(uv) - 0.5) * DitherStrength);
}

float3 ApplyToneMapping(float3 c, float2 uv)
{
    if (ToneMappingMethod == 5)
        return AdaptiveToneMapping(c, CalculateSceneLuminance(uv));
    else if (ToneMappingMethod == 0)
        return ReinhardToneMapping(c);
    else if (ToneMappingMethod == 1)
        return FilmicToneMapping(c);
    else if (ToneMappingMethod == 2)
        return ACESToneMapping(c);
    else if (ToneMappingMethod == 3)
        return BTToneMapping(c);
    else if (ToneMappingMethod == 4)
        return LogarithmicToneMapping(c);
    return c;
}

float4 uFakeHDRPass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float3 c = pow(tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb, HDRPower);
    c = ApplyToneMapping(c, uv);
    if (Extra == 1)
        c = MultipleExposuresHDR(c);
    if (EnableDithering)
        c = ApplyDithering(c, uv);
    return float4(saturate(c), 1.0);
}

float4 Composite_PS(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return float4(saturate(tex2Dlod(sFakeHDR, float4(uv, 0, 0)).rgb), 1.0);
}

//: Techniques
technique uFakeHDR
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = uFakeHDRPass;
        RenderTarget = FakeHDRTex;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Composite_PS;
    }
}
