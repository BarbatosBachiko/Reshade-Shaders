/*------------------.
| :: Description :: |
'-------------------/

uFakeHDR 

Version 1.3.1
Author: BarbatosBachiko
License: MIT

About: This shader simulates HDR effects (expected by me) for SDR. 
History:
(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

Version 1.3.1:
+ Code clean

*/

/*---------------.
| :: Includes :: |
'---------------*/
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/
uniform float HDRPower < 
    ui_type = "slider";
    ui_label = "HDR Power"; 
    ui_min = 1.0; 
    ui_max = 4.0; 
> = 1.150;

uniform int ToneMappingMethod < 
    ui_type = "combo";
    ui_label = "Tone Mapping Method"; 
    ui_items = "Reinhard\0Filmic\0ACES\0BT.709\0Logarithmic\0Adaptive\0";
> = 5;

uniform bool EnableDithering < 
    ui_category = "Dithering";
    ui_type = "checkbox";
    ui_label = "Enable Dithering"; 
> = false; 

uniform float DitherStrength < 
    ui_category = "Dithering";
    ui_type = "slider";
    ui_label = "Dither Strength"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
> = 0.05; 

uniform float NoiseScale < 
    ui_category = "Dithering";
    ui_type = "slider";
    ui_label = "Noise Scale"; 
    ui_min = 0.1; 
    ui_max = 10.0; 
> = 1.0; 

uniform float NoiseSeed < 
    ui_category = "Dithering";
    ui_type = "slider";
    ui_label = "Noise Seed"; 
    ui_min = 1.0; 
    ui_max = 100000.0; 
> = 43758.5453; 

uniform float Luminance < 
    ui_category = "Luminace (only for Adaptative)";
    ui_type = "slider";
    ui_label = "Luminance"; 
    ui_min = 0.01; 
    ui_max = 1.0; 
> = 0.1;

/*---------------.
| :: Textures :: |
'---------------*/

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

#define textureSize float2(BUFFER_WIDTH, BUFFER_HEIGHT)
static float lastSceneLuminance = 0.0;

/*----------------.
| :: Functions :: |
'----------------*/

float CalculateSceneLuminance(float2 texcoord)
{
    float sampleRadius = 2.0;
    float luminanceSum = 0.0;
    int sampleCount = 0;

    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            float2 offset = float2(x, y) * (sampleRadius / textureSize.x);
            float2 sampleCoord = texcoord + offset;

            if (all(saturate(sampleCoord) == sampleCoord))
            {
                float3 sampleColor = tex2D(sFakeHDR, sampleCoord).rgb;
                float luminance = dot(sampleColor, float3(0.2126, 0.7152, 0.0722));

                luminanceSum += luminance;
                sampleCount++;
            }
        }
    }

    float currentLuminance = (sampleCount > 0) ? luminanceSum / sampleCount : 0.0;
    lastSceneLuminance = lerp(lastSceneLuminance, currentLuminance, Luminance);

    return lastSceneLuminance;
}

float3 ReinhardToneMapping(float3 color)
{
    const float a = 0.25;
    const float maxLum = 1.0;

    float luminance = dot(color, float3(0.2126, 0.7152, 0.0722)) * 1.2;
    luminance = max(luminance, 0.0001);
    float normalizedLuminance = luminance / (a * maxLum);
    normalizedLuminance = clamp(normalizedLuminance, 0.0, 1.0);
    
    float3 mapped = color * (normalizedLuminance / (normalizedLuminance + 1.0));
    mapped *= 1.2;
    
    return saturate(mapped);
}

float3 FilmicToneMapping(float3 color)
{
    float3 mapped = (color * (color * 0.6 + 0.4)) / (color + 0.6);
    return saturate(mapped * 1.5);
}

float3 ACESToneMapping(float3 color)
{
    float3 mapped = (color * (color + 0.0245786) - (color * color * 0.000093607)) / (color + 0.000009);
    return saturate(mapped);
}

float3 BTToneMapping(float3 color)
{
    float3 mapped = color * (color * 0.7 + 0.3);
    return saturate(mapped);
}

float3 LogarithmicToneMapping(float3 color)
{
    return log(color + 1.0) / log(2.0);
}

float3 AdaptiveToneMapping(float3 color, float sceneLuminance)
{
    float targetLuminance = 0.5;
    float minAdjustment = 0.5;
    float maxAdjustment = 2.0;
    float luminanceAdjustment = targetLuminance / (sceneLuminance + 0.001);
    luminanceAdjustment = clamp(luminanceAdjustment, minAdjustment, maxAdjustment);
    float adjustmentFactor = lerp(1.0, luminanceAdjustment, Luminance);
    color *= adjustmentFactor;

    return saturate(color);
}

float3 ApplyToneMapping(float3 color, float2 texcoord)
{
    float sceneLuminance = CalculateSceneLuminance(texcoord);

    switch (ToneMappingMethod)
    {
        case 0:
            return ReinhardToneMapping(color);
        case 1:
            return FilmicToneMapping(color);
        case 2:
            return ACESToneMapping(color);
        case 3:
            return BTToneMapping(color);
        case 4:
            return LogarithmicToneMapping(color);
        case 5:
            return AdaptiveToneMapping(color, sceneLuminance);
        default:
            return color;
    }
}

float make_noise(float2 uv)
{
    return frac(sin(dot(uv.xy * NoiseScale, float2(12.9898, 78.233))) * NoiseSeed);
}

float3 ApplyDithering(float3 color, float2 texcoord, float DitherStrength)
{
    float noise = make_noise(texcoord);
    color += (noise - 0.5) * DitherStrength;

    return saturate(color);
}

float4 uFakeHDRPass(float4 position : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 hdrColor = pow(color, HDRPower);
    
    hdrColor = ApplyToneMapping(hdrColor, texcoord);

    if (EnableDithering)
    {
        hdrColor = ApplyDithering(hdrColor, texcoord, DitherStrength);
    }

    return float4(saturate(hdrColor), 1.0);
}

float4 Composite_PS(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 hdrColor = tex2D(sFakeHDR, uv).rgb;
    return float4(saturate(hdrColor), 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/
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
