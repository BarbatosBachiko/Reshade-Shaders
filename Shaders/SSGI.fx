/*------------------.
| :: Description :: |
'-------------------/

  ___ ___  ___ ___ 
 / __/ __|/ __|_ _|
 \__ \__ \ (_ || | 
 |___/___/\___|___|
                   
    Author: Barbatos Bachiko
    License: MIT

    About: This shader simulates screen-space global illumination (SSGI) to enhance scene lighting by calculating indirect light bounces.

    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    
    Version 1.2
    + Add NormalMap
*/
#include "ReShade.fxh"

namespace SSGI
{
/*---------------.
| :: Settings :: |
'---------------*/

    uniform int viewMode
    < 
        ui_type = "combo";
        ui_label = "View Mode";
        ui_tooltip = "Select the view mode";
        ui_items = 
            "Normal\0" 
            "GI Debug\0";
    >
    = 0;

    uniform float giIntensity
    < 
        ui_type = "slider";
        ui_label = "GI Intensity";
        ui_tooltip = "Adjust the intensity.";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
    >
    = 0.5;

    uniform float sampleRadius
    < 
        ui_type = "slider";
        ui_label = "Sample Radius";
        ui_tooltip = "Adjust the radius of the samples.";
        ui_min = 0.001; ui_max = 10.0; ui_step = 0.01;
    >
    = 1.0;

    uniform int numSamples
    < 
        ui_type = "slider";
        ui_label = "Sample Count";
        ui_tooltip = "Number of samples (higher = better quality).";
        ui_min = 4; ui_max = 64; ui_step = 1;
    >
    = 32;

    uniform int numBounces
    < 
        ui_type = "slider";
        ui_label = "Number of Bounces";
        ui_tooltip = "Number of GI bounces.";
        ui_min = 1; ui_max = 4; ui_step = 1;
    >
    = 2;

/*---------------.
| :: Textures :: |
'---------------*/

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
        return ReShade::GetLinearizedDepth(coords);
    }

    float2 RandomOffset(float2 coord)
    {
        return frac(sin(dot(coord, float2(12.9898, 78.233))) * 43758.5453);
    }
    
    float3 GetPosition(float2 coords)
    {
        float EyeDepth = GetLinearDepth(coords.xy) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
        return float3((coords.xy * 2.0 - 1.0) * EyeDepth, EyeDepth);
    }

    float3 SampleDiffuse(float2 coord)
    {
        return tex2D(ColorSampler, clamp(coord, 0.0, 1.0)).rgb;
    }

    float3 GetNormal(float2 coords)
    {
        return tex2D(NormalSampler, coords).xyz * 2.0 - 1.0;
    }

    float3 GatherDiffuseGI(float2 texcoord, float radius)
    {
        float3 indirectLight = 0.0;
        float3 viewPos = GetPosition(texcoord);
        float3 normal = GetNormal(texcoord);
        float radiusOverSamples = radius / float(numSamples);

        for (int i = 0; i < numSamples; ++i)
        {
 
            float angle = (float(i) / numSamples) * 6.28318530718;
            float2 randomDir = float2(cos(angle), sin(angle)) * RandomOffset(texcoord + float2(i, i));
            float2 sampleCoord = texcoord + randomDir * radiusOverSamples;
            float3 sampleColor = SampleDiffuse(sampleCoord);
            if (GetLinearDepth(sampleCoord) < viewPos.z)
            {
                float3 sampleNormal = GetNormal(sampleCoord);
                float normalDot = dot(normal, sampleNormal);
                if (normalDot > 0.0)
                {
                    indirectLight += sampleColor;
                }
            }
        }
        
        return indirectLight / numSamples;
    }

    float3 ComputeMultiBounceGI(float2 texcoord)
    {
        float3 totalIndirectLight = 0.0;
        float3 indirectLight = GatherDiffuseGI(texcoord, sampleRadius);

        totalIndirectLight += giIntensity * indirectLight;

        for (int bounce = 1; bounce < numBounces; ++bounce)
        {
            indirectLight = GatherDiffuseGI(texcoord, sampleRadius * 0.5);
            totalIndirectLight += giIntensity * indirectLight * (1.0 / (bounce + 1));
        }

        return totalIndirectLight;
    }

    float4 GlobalIlluminationPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
    {
        float3 originalColor = SampleDiffuse(texcoord);
        float3 indirectLight = ComputeMultiBounceGI(texcoord);
        float3 finalColor = originalColor + indirectLight;

        if (viewMode == 0)
        {
            return float4(finalColor, 1.0);
        }
        else if (viewMode == 1)
        {
            float3 giContribution = clamp(indirectLight - originalColor, 0.0, 5.0);
            return float4(giContribution, 1.0);
        }
    
        return float4(originalColor, 1.0);
    }

    void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
    {
        texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
        position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    }

/*-----------------.
| :: Techniques :: |
'-----------------*/

    technique SSGI
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = GlobalIlluminationPS;
        }
    }
}
