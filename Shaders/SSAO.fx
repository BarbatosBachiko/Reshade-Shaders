/*
        SSAO (version 0.6)

	Author: Barbatos Bachiko
	License: MIT

	About:
        Screen-Space Ambient Occlusion (SSAO) using a mix of fixed and random direction sampling for depth-based occlusion calculations.
        The different ambient occlusion methods:
        - AO-F: Uses fixed direction samples (a predefined set of directions for occlusion).
          Performance: Very fast.

        - AO-R: Uses random direction samples (randomly generated directions for occlusion).
          Performance: Slightly slower than AO-F.

        - AO-R2: Uses a more randomized approach with golden angle-based distribution for sampling directions.
          Performance: Slower than AO-R.

        - AO-RH: Uses random hemisphere sampling (generates occlusion based on random samples within a hemisphere around the surface.
          Performance: Slower than AO-R2.

        Best for performance: AO-F (Fixed Direction), if you prioritize speed over visual quality.
        Best balance: AO-R (Random Direction) or AO-R2 (Golden Angle), providing better quality with relatively modest performance overhead.
        Best quality: O-RH (Random Hemisphere Sampling), which will give the best-looking results but requires the most computational resources.

	Ideas for future improvement:

	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 0.6
	x Namespace implemented to avoid conflicts with other shaders.

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
"AO Only\0"
"AO Debug\0";

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
    = 0;

uniform float intensity
    <
        ui_type = "slider";
        ui_label = "Occlusion Intensity";
        ui_tooltip = "Adjust the intensity of ambient occlusion";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
    >
    = 0.1;

uniform int aoType
<
    ui_type = "combo";
    ui_label = "AO Type";
    ui_tooltip = "Select ambient occlusion type";
    ui_items = 
    "AO-F\0" 
    "AO-R\0"
    "AO-R2\0"
    "AO-RH\0"; // Renamed Hemisphere AO Type to O-RH
>
= 1;

    /*---------------.
    | :: Textures :: |
    '---------------*/
   
namespace SSAO
{
    texture ColorTex : COLOR;
    texture DepthTex : DEPTH;

    sampler ColorSampler
    {
        Texture = ColorTex;
    };

    sampler DepthSampler
    {
        Texture = DepthTex;
    };

    /*----------------.
    | :: Functions :: |
    '----------------*/

    // Random Direction
    float3 RandomDirection(float2 texcoord, int sampleIndex)
    {
        float randomValue = frac(sin(dot(texcoord * 100.0 + float2(sampleIndex, 0.0), float2(12.9898, 78.233))) * 43758.5453);
        float phi = randomValue * 2.0 * 3.14159265359;
        float theta = acos(1.0 - 2.0 * (frac(randomValue * 0.12345)));

        float3 dir;
        dir.x = sin(theta) * cos(phi);
        dir.y = sin(theta) * sin(phi);
        dir.z = cos(theta);

        return normalize(dir * 0.5 + float3(0.5, 0.5, 0.5));
    }

    //Fixed Direction
    float3 FixedDirection(int sampleIndex)
    {
        float3 directions[8] =
        {
            float3(1.0, 0.0, 0.0),
            float3(-1.0, 0.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float3(0.0, -1.0, 0.0),
            float3(0.707, 0.707, 0.0),
            float3(-0.707, 0.707, 0.0),
            float3(0.707, -0.707, 0.0),
            float3(-0.707, -0.707, 0.0)
        };
        return directions[sampleIndex % 8];
    }

    // Random Direction 2
    float3 RandomDirection2(float2 texcoord, int sampleIndex)
    {
        const float goldenAngle = 2.399963229728653;
        float theta = goldenAngle * sampleIndex;
        float r = sqrt(float(sampleIndex) / 16.0);
        float2 xy = float2(r * cos(theta), r * sin(theta));
        float z = sqrt(1.0 - r * r);
        return normalize(float3(xy, z));
    }

    // Hemisphere Sampling 
    float3 HemisphereSampling(int sampleIndex)
    {
        // Generate a random hemisphere direction for each sample
        float phi = (sampleIndex + 0.5) * 3.14159265359 * 2.0 / 26.0; // 16 samples in the hemisphere
        float theta = acos(2.0 * frac(sin(phi) * 0.5 + 0.5) - 1.0);
        
        float3 dir;
        dir.x = sin(theta) * cos(phi);
        dir.y = sin(theta) * sin(phi);
        dir.z = cos(theta);

        return normalize(dir);
    }

    // Main SSAO
    float4 SSAO(float2 texcoord)
    {
        float4 originalColor = tex2D(ColorSampler, texcoord);
        float depthValue = tex2D(DepthSampler, texcoord).r;
        float occlusion = 0.0;

        int sampleCount;
        if (qualityLevel == 0)
            sampleCount = 8;
        else if (qualityLevel == 1)
            sampleCount = 16;
        else
            sampleCount = 32;

        float radius = 0.005;
        float falloff = 0.01;

        for (int i = 0; i < sampleCount; i++)
        {
            float3 sampleDir;
            if (aoType == 0)
                sampleDir = FixedDirection(i);
            else if (aoType == 1)
                sampleDir = RandomDirection(texcoord, i);
            else if (aoType == 2)
                sampleDir = RandomDirection2(texcoord, i);
            else if (aoType == 3) 
                sampleDir = HemisphereSampling(i);

            float2 sampleCoord = clamp(texcoord + sampleDir.xy * radius, 0.0, 1.0);
            float sampleDepth = tex2D(DepthSampler, sampleCoord).r;
            float rangeCheck = exp(-abs(depthValue - sampleDepth) / falloff);

            occlusion += (sampleDepth < depthValue) ? rangeCheck : 0.0;
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
