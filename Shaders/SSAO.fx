/*------------------.
| :: Description :: |
'-------------------/

  ___ ___   _   ___  
 / __/ __| /_\ / _ \ 
 \__ \__ \/ _ \ (_) |
 |___/___/_/ \_\___/  
                                                                                       
    Version 0.7.2
	Author: Barbatos Bachiko
	License: MIT

	About: This shader implements a screen space ambient occlusion (SSAO) effect for ReShade.
    The shader supports two types of ambient occlusion sampling:
    -Random Direction Sampling: Randomized sampling direction for more natural, irregular occlusion patterns.
    -Hemisphere Sampling: Uniform hemispherical sampling for smoother, more controlled occlusion.

	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 0.7.2
    * add MXAO (Random + Hemisphere)

*/ 

    /*---------------.
    | :: Includes :: |
    '---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

    /*---------------.
    | :: Settings :: |
    '---------------*/

uniform int viewMode
    <
        ui_type = "combo";
        ui_label = "View Mode";
        ui_tooltip = "Select the view mode for SSAO";
        ui_items = 
"Normal\0" 
"AO Debug\0"
"Depth\0"
"Sky Debug\0";

    >
    = 0;

uniform int qualityLevel
    <
        ui_type = "combo";
        ui_label = "Quality Level";
        ui_tooltip = "Select quality level for ambient occlusion";
        ui_items = 
        "Low\0"
        "Medium\0"
        "High\0";
    >
    = 2;

uniform float intensity
    <
        ui_type = "slider";
        ui_label = "Occlusion Intensity";
        ui_tooltip = "Adjust the intensity of ambient occlusion";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
    >
    = 0.3;

uniform int aoType
<
    ui_type = "combo";
    ui_label = "AO Type";
    ui_tooltip = "Select ambient occlusion type";
    ui_items = 
    "Random Direction\0"
    "Hemisphere\0"
    "MXAO\0"; 
>
= 2;

uniform float sampleRadius
    <
        ui_type = "slider";
        ui_label = "Sample Radius";
        ui_tooltip = "Adjust the radius of the samples for SSAO";
        ui_min = 0.001; ui_max = 0.01; ui_step = 0.001;
    >
    = 0.005;

uniform float noiseScale
    <
        ui_type = "slider";
        ui_category = "Random Directios Settings";
        ui_label = "Noise Scale";
        ui_tooltip = "Adjust the scale of noise for random direction sampling";
        ui_min = 0.1; ui_max = 1.0; ui_step = 0.05;
    >
    = 1.0;

uniform float fDepthMultiplier <
        ui_type = "slider";
        ui_category = "Depth";
        ui_label = "Depth multiplier";
        ui_min = 0.001; ui_max = 20.00;
        ui_step = 0.001;
    > = 1.0;

uniform float depthThreshold
<
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Threshold (Sky)";
    ui_tooltip = "Set the depth threshold to ignore the sky during occlusion.";
    ui_min = 0.9; ui_max = 1.0; ui_step = 0.01;
>
= 0.95;

    /*---------------.
    | :: Textures :: |
    '---------------*/

namespace SSAO
{
    texture ColorTex : COLOR;
    texture DepthTex : DEPTH;
    texture NormalTex : NORMAL;

    sampler ColorSampler
    {
        Texture = ColorTex;
    };

    sampler DepthSampler
    {
        Texture = DepthTex;
    };

    sampler NormalSampler
    {
        Texture = NormalTex;
    };

    /*----------------.
    | :: Functions :: |
    '----------------*/

    float GetLinearDepth(float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords) * fDepthMultiplier;
    }

    float3 GetNormal(float2 coords)
    {
        float4 normalTex = tex2D(NormalSampler, coords);
        float3 normal = normalize(normalTex.xyz * 2.0 - 1.0);
        return normal;
    }

    // Random Direction
    float3 RandomDirection(float2 texcoord, int sampleIndex, float noiseScale)
    {
        float randomValue = frac(sin(dot(texcoord * 100.0 + float2(sampleIndex, 0.0), float2(12.9898, 78.233))) * 43758.5453);
        float phi = randomValue * 2.0 * 3.14159265359;
        float theta = acos(1.0 - 2.0 * (frac(randomValue * 0.12345)));

        float3 dir;
        dir.x = sin(theta) * cos(phi);
        dir.y = sin(theta) * sin(phi);
        dir.z = cos(theta);

        float3 normal = GetNormal(texcoord);
        dir = normalize(reflect(dir, normal));

        return normalize(dir * noiseScale + float3(0.5, 0.5, 0.5));
    }

    // Hemisphere Sampling
    float3 HemisphereSampling(int sampleIndex, float3 normal)
    {
        float phi = (sampleIndex + 0.5) * 3.14159265359 * 2.0 / 26.0; // amostras
        float theta = acos(2.0 * frac(sin(phi) * 0.5 + 0.5) - 1.0);

        float3 dir;
        dir.x = sin(theta) * cos(phi);
        dir.y = sin(theta) * sin(phi);
        dir.z = cos(theta);
        dir = normalize(reflect(dir, normal));
        return dir;
    }

    // Main SSAO
    float4 SSAO(float2 texcoord)
    {
        float4 originalColor = tex2D(ColorSampler, texcoord);
        float depthValue = GetLinearDepth(texcoord);
        float3 normal = GetNormal(texcoord);
        float occlusion = 0.0;

        int sampleCount;
        if (qualityLevel == 0)
            sampleCount = 8;
        else if (qualityLevel == 1)
            sampleCount = 16;
        else
            sampleCount = 32;

        float radius = sampleRadius;
        float falloff = 0.01;
        for (int i = 0; i < sampleCount; i++)
        {
            float3 sampleDir = float3(0.0, 0.0, 0.0);

            if (aoType == 0) 
                sampleDir = RandomDirection(texcoord, i, noiseScale);
            else if (aoType == 1) 
                sampleDir = HemisphereSampling(i, normal);
            else if (aoType == 2) 
            {
                float3 randomDir = RandomDirection(texcoord, i, noiseScale);
                float3 hemisphereDir = HemisphereSampling(i, normal);
                sampleDir = normalize(randomDir + hemisphereDir); 
            }

            float2 sampleCoord = clamp(texcoord + sampleDir.xy * radius, 0.0, 1.0);
            float sampleDepth = GetLinearDepth(sampleCoord);

            if (sampleDepth < depthThreshold)
            {
                float rangeCheck = exp(-abs(depthValue - sampleDepth) / falloff);
                occlusion += (sampleDepth < depthValue) ? rangeCheck : 0.0;
            }
        }

        occlusion = (occlusion / sampleCount) * intensity;

        if (viewMode == 0)
        {
            return originalColor * (1.0 - saturate(occlusion));
        }

        if (viewMode == 1)
        {
            return float4(saturate(occlusion), saturate(occlusion), saturate(occlusion), 1.0);
        }
        else if (viewMode == 2)
        {
            return float4(depthValue, depthValue, depthValue, 1.0);
        }
        else if (viewMode == 3)
        {
            if (depthValue >= depthThreshold)
                return float4(1.0, 0.0, 0.0, 1.0);
            return float4(depthValue, depthValue, depthValue, 1.0);
        }

        return originalColor;
    }

    // PixelShader
    float4 SSAOPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
    {
        float4 ssaoColor = SSAO(texcoord);
        return ssaoColor;
    }

    /*-----------------.
    | :: Techniques :: |
    '-----------------*/

    technique SSAO
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = SSAOPS;
        }
    }
}
