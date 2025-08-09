/*

VividTone

Version 1.0
Author: Barbados Bachiko
License: MIT

About: Transforms ordinary game visuals into vibrant, high-contrast scenes
*/

#include "ReShade.fxh"

#define GetColor(coord) tex2Dlod(ReShade::BackBuffer, float4(coord, 0, 0))

/*-------------------.
| :: Parameters ::   |
'-------------------*/

// Basic HDR Controls
uniform float HDRPower <
    ui_type = "slider";
    ui_label = "Intensity";
    ui_min = 0.1;
    ui_max = 4.0;
    ui_step = 0.01;
> = 1.4;

uniform float Exposure <
    ui_type = "slider";
    ui_label = "Exposure";
    ui_tooltip = "Adjust overall brightness";
    ui_min = -3.0;
    ui_max = 3.0;
    ui_step = 0.01;
> = 0.3;

// Scene Adaptation Controls
uniform float DarkSceneLift <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Shadow Brightness";
    ui_tooltip = "Lifts shadows in dark areas";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.2;

uniform float Contrast <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Contrast";
    ui_tooltip = "Adjusts image contrast";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.01;
> = 1.2;

// Adaptive Tone Mapping Controls
uniform float AdaptationSpeed <
    ui_category = "Advanced";
    ui_type = "slider";
    ui_label = "Adaptation Speed";
    ui_tooltip = "How quickly the effect adjusts to scene changes";
    ui_min = 0.01;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.05;

uniform float TargetBrightness <
    ui_category = "Advanced";
    ui_type = "slider";
    ui_label = "Target Brightness";
    ui_tooltip = "Target mid-range brightness level";
    ui_min = 0.1;
    ui_max = 0.5;
    ui_step = 0.01;
> = 0.5;

uniform float timer < source = "timer"; >;

/*---------------.
| :: Textures :: |
'---------------*/
namespace VividTone
{
    texture LUM
{
    Width = 1;
    Height = 1;
    Format = R32F;
};
sampler sLUM
{
    Texture = LUM;
};

texture PREV_LUM
{
    Width = 1;
    Height = 1;
    Format = R32F;
};
sampler s_PREV_LUM
{
    Texture = PREV_LUM;
};

// Constants
static const float3 LumaCoeff = float3(0.2126, 0.7152, 0.0722);
static const float SceneBrightThreshold = 0.5;
static const float SceneDarkThreshold = 0.15;
static const float MinLuminance = 0.01;
static const float MaxLuminance = 2.0;
static const float AdaptiveSaturation = 1.0;
static const float AdaptiveStrength = 0.5;
static const float BrightSceneVibrancy = 0.0;
static const float MultiExposureStrength = 0.1;

/*----------------.
| :: Functions :: |
'----------------*/

float GetLuminance(float3 color)
{
    return dot(color, LumaCoeff);
}

float CalculateSceneLuminance()
{
    float totalLum = 0.0;
    int samples = 0;
    
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            float2 uv = float2((x + 0.5) / 8.0, (y + 0.5) / 8.0);
            float3 color = GetColor(uv).rgb;
            float lum = GetLuminance(color);
            
            totalLum += log(max(lum, 0.0001));
            samples++;
        }
    }
    
    return exp(totalLum / samples);
}

float3 EnhanceVibrancy(float3 color, float vibrancy)
{
    float lum = GetLuminance(color);
    float3 gray = float3(lum, lum, lum);
    
    float vibrancyFactor = vibrancy * (1.0 - lum * 0.7);
    return lerp(gray, color, 1.0 + vibrancyFactor);
}

float3 LiftShadows(float3 color, float lift)
{
    float lum = GetLuminance(color);
    float shadowMask = pow(1.0 - lum, 2.0);
    
    float3 lifted = color + (lift * shadowMask * 0.1);
    return lifted;
}

float3 AdaptiveToneMapping(float3 color, float sceneLum, float exposure)
{
    float3 originalColor = color;
    
    float brightSceneFactor = smoothstep(SceneDarkThreshold, SceneBrightThreshold, sceneLum);
    float darkSceneFactor = 1.0 - brightSceneFactor;
    
    color *= exposure;
    
    if (brightSceneFactor > 0.5)
    {
        color = EnhanceVibrancy(color, BrightSceneVibrancy * brightSceneFactor);
        float midGray = 0.5;
        color = lerp(color, pow(color / midGray, 1.0 + Contrast * brightSceneFactor) * midGray, 0.5);
    }
    else
    {
        color = LiftShadows(color, DarkSceneLift * darkSceneFactor);
    }
    
    float adaptedLum = clamp(sceneLum, MinLuminance, MaxLuminance);
    float key = TargetBrightness / max(adaptedLum, 0.001);
    key = lerp(key * 1.5, key * 0.8, brightSceneFactor);
    key = clamp(key, 0.1, 5.0);
    
    float lum = GetLuminance(color);
    float scaledLum = lum * key;
    float whitePoint = lerp(MaxLuminance * 0.8, MaxLuminance * 1.2, brightSceneFactor);
    float toneMappedLum = scaledLum / (1.0 + scaledLum / whitePoint);
    
    float3 toneMappedColor = color * (toneMappedLum / max(lum, 0.0001));
    float adaptiveSat = lerp(AdaptiveSaturation * 1.2, AdaptiveSaturation * 0.9, brightSceneFactor);
    float3 lumColor = float3(toneMappedLum, toneMappedLum, toneMappedLum);
    toneMappedColor = lerp(lumColor, toneMappedColor, adaptiveSat);
    
    float3 result = lerp(originalColor, toneMappedColor, AdaptiveStrength);
    return saturate(result);
}

float3 MultipleExposuresHDR(float3 color, float sceneLum)
{
    float brightSceneFactor = smoothstep(SceneDarkThreshold, SceneBrightThreshold, sceneLum);
    
    float underExp = lerp(2.5, 1.5, brightSceneFactor);
    float overExp = lerp(0.4, 0.6, brightSceneFactor);
    
    float3 underexposed = pow(color, underExp);
    float3 normal = color;
    float3 overexposed = pow(color, overExp);
    
    float lum = GetLuminance(color);
    float underWeight = saturate(1.0 - lum * 2.0) * (1.0 - brightSceneFactor * 0.5);
    float overWeight = saturate(lum * 2.0 - 1.0) * (1.0 + brightSceneFactor * 0.5);
    float normalWeight = 1.0 - underWeight - overWeight;
    
    return underexposed * underWeight + normal * normalWeight + overexposed * overWeight;
}

float4 LuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float currentLum = CalculateSceneLuminance();
    float prevLum = tex2Dfetch(s_PREV_LUM, int2(0, 0)).r;
    
    float adaptSpeed = AdaptationSpeed;
    if (abs(currentLum - prevLum) > 0.1)
        adaptSpeed *= 2.0;
    
    float adaptedLum = lerp(prevLum, currentLum, adaptSpeed);
    
    return float4(adaptedLum, 0, 0, 1);
}

float4 StoreLuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    return tex2Dfetch(sLUM, int2(0, 0));
}

float4 Final(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float3 color = GetColor(uv).rgb;
    
    color = pow(abs(color), HDRPower);
    
    float sceneLum = tex2Dfetch(sLUM, int2(0, 0)).r;
    float exposure = pow(2.0, Exposure);
    
    color = AdaptiveToneMapping(color, sceneLum, exposure);
    color = MultipleExposuresHDR(color, sceneLum);
    
    return float4(saturate(color), 1.0);
}

technique VividTone
{
    pass LuminanceCalculation
    {
        VertexShader = PostProcessVS;
        PixelShader = LuminancePass;
        RenderTarget = LUM;
    }
    
    pass StorePreviousLuminance
    {
        VertexShader = PostProcessVS;
        PixelShader = StoreLuminancePass;
        RenderTarget = PREV_LUM;
    }
    
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Final;
    }
  }

}
