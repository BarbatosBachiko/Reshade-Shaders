/*-------------------------------------------------|
| ::                 VividTone                  :: |
'--------------------------------------------------|
| Version 1.2                                      |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: Transforms ordinary game visuals    | 
| into vibrant, high-contrast scenes               |
'-------------------------------------------------*/

#include "ReShade.fxh"

#define GetColor(coord) tex2Dlod(ReShade::BackBuffer, float4(coord, 0, 0))

/*-------------------.
| :: Parameters ::   |
'-------------------*/

uniform float Intensity <
    ui_label = "Global Intensity";
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.001;
> = 0.9;

uniform float HDRStrength <
    ui_category = "Tone Controls";
    ui_type = "slider";
    ui_label = "FakeHDR Power";
    ui_tooltip = "Gamma correction intensity";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.15;

uniform float ExposureStrength <
    ui_category = "Tone Controls";
    ui_type = "slider";
    ui_label = "Exposure";
    ui_tooltip = "Added exposure";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.1; 

uniform float ShadowLiftStrength <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Shadow Lift";
    ui_tooltip = "Strength of shadow brightening in dark scenes";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.2;

uniform float ContrastStrength <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Adaptive Contrast";
    ui_tooltip = "Strength of contrast in bright scenes.";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.6; 

uniform bool StaticExposure <
    ui_category = "Settings";
    ui_label = "Static Exposure";
    ui_tooltip = "Disables brightness adaptation to provide a more consistent image.";
> = false;

uniform float AdaptationRate <
    ui_category = "Settings";
    ui_type = "slider";
    ui_label = "Adaptation Rate";
    ui_tooltip = "Speed of eye adaptation.";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.05;

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

    static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
    
    // Limits
    static const float MinHDR = 1.0; // Neutral Gamma
    static const float MaxHDR = 4.0; // Strong Gamma
    static const float MinExp = 0.0; // Neutral Exposure
    static const float MaxExp = 3.0; // Strong Exposure
    static const float MaxContrast = 2.0;
    
    // Internal Settings
    static const float SceneBrightThreshold = 0.5;
    static const float SceneDarkThreshold = 0.15;
    static const float MinLuminance = 0.01;
    static const float MaxLuminance = 2.0;
    static const float TargetBrightness = 0.5; 
    static const float AdaptiveSaturation = 1.0;
    static const float AdaptiveStrength = 0.5;
    static const float BrightSceneVibrancy = 0.0;

/*----------------.
| :: Functions :: |
'----------------*/

    float GetLuminance(float3 color)
    {
        return dot(color, LUMA);
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
    
        return color + (lift * shadowMask * 0.1);
    }

    float3 AdaptiveToneMapping(float3 color, float sceneLum, float exposure, float contrastNorm, float shadowLiftNorm, float brightSceneFactor)
    {
        float3 originalColor = color;
        float darkSceneFactor = 1.0 - brightSceneFactor;
        color *= exposure;
        
        if (brightSceneFactor > 0.5)
        {
            color = EnhanceVibrancy(color, BrightSceneVibrancy * brightSceneFactor);
            float midGray = 0.5;
            float realContrast = lerp(0.0, MaxContrast, contrastNorm);
            color = lerp(color, pow(color / midGray, 1.0 + realContrast * brightSceneFactor) * midGray, 0.5);
        }
        else
        {
            color = LiftShadows(color, shadowLiftNorm * darkSceneFactor);
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
    
        return lerp(originalColor, toneMappedColor, AdaptiveStrength);
    }

    float3 MultipleExposuresHDR(float3 color, float sceneLum, float brightSceneFactor)
    {
        float underExp = lerp(2.5, 1.5, brightSceneFactor);
        float overExp = lerp(0.6, 0.8, brightSceneFactor); 
    
        float3 underexposed = pow(color, underExp);
        float3 overexposed = pow(color, overExp);
    
        float lum = GetLuminance(color);
        float underWeight = saturate(1.0 - lum * 2.0) * (1.0 - brightSceneFactor * 0.5);
        float overWeight = saturate(lum * 2.0 - 1.0) * (1.0 + brightSceneFactor * 0.5);
        float normalWeight = 1.0 - underWeight - overWeight;
    
        return underexposed * underWeight + color * normalWeight + overexposed * overWeight;
    }

/*----------------.
| :: Shaders ::   |
'----------------*/

    float4 LuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        float prevLum = tex2Dfetch(s_PREV_LUM, int2(0, 0)).r;
        float adaptedLum;

        if (StaticExposure)
        {
            adaptedLum = prevLum;
        }
        else
        {
            float currentLum = CalculateSceneLuminance();
            
            float adaptSpeed = lerp(0.01, 1.0, AdaptationRate);
            if (abs(currentLum - prevLum) > 0.1)
                adaptSpeed *= 2.0;
            
            adaptedLum = lerp(prevLum, currentLum, adaptSpeed);
        }
    
        return float4(adaptedLum, 0, 0, 1);
    }

    float4 StoreLuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        return tex2Dfetch(sLUM, int2(0, 0));
    }

    float4 Final(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        float3 originalColor = GetColor(uv).rgb;
        
        //Map HDR Power
        float realHDR = lerp(MinHDR, MaxHDR, HDRStrength);
        float3 color = pow(abs(originalColor), realHDR);
        float sceneLum = tex2Dfetch(sLUM, int2(0, 0)).r;
        float brightSceneFactor = smoothstep(SceneDarkThreshold, SceneBrightThreshold, sceneLum);

        //Map Exposure 
        float realExposureVal = lerp(MinExp, MaxExp, ExposureStrength);
        float exposure = pow(2.0, realExposureVal);
        
        //ToneMapping
        color = AdaptiveToneMapping(color, sceneLum, exposure, ContrastStrength, ShadowLiftStrength, brightSceneFactor);
        color = MultipleExposuresHDR(color, sceneLum, brightSceneFactor);
        
        color = lerp(originalColor, saturate(color), Intensity);
        return float4(color, 1.0);
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
