/*------------------.
| :: Description :: |
'-------------------/

  ___ ___   _   ___  
 / __/ __| /_\ / _ \ 
 \__ \__ \/ _ \ (_) |
 |___/___/_/ \_\___/  
                                                                                       
    Version 0.9
	Author: Barbatos Bachiko
	License: MIT

	About: This shader implements a screen space ambient occlusion (SSAO) effect for ReShade.
    The shader supports two types of ambient occlusion sampling:
    -Random Direction Sampling: Randomized sampling direction for more natural, irregular occlusion patterns.
    -Hemisphere Sampling: Uniform hemispherical sampling for smoother, more controlled occlusion.

	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 0.9
    + Render Scale, new texture, code optimization.
*/ 
namespace SSAOmaisoumenos
{
#ifndef RENDER_SCALE
#define RENDER_SCALE 0.888
#endif
#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 
#define RENDER_WIDTH (INPUT_WIDTH * RENDER_SCALE)
#define RENDER_HEIGHT (INPUT_HEIGHT * RENDER_SCALE) 
    /*---------------.
    | :: Includes :: |
    '---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

    /*---------------.
    | :: Settings :: |
    '---------------*/

uniform int ViewMode
<
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for SSAO";
    ui_items = "Normal\0AO Debug\0Depth\0Sky Debug\0Normal Debug\0";
>
= 0;

uniform int sampleCount
<
    ui_type = "slider";
    ui_label = "Sample Count";
    ui_tooltip = "Number of samples for SSAO calculation";
    ui_min = 1; ui_max = 64; ui_step = 1;
>
= 16;

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
    "Mixed (Random + Hemisphere)\0"; 
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
    ui_category = "Random Directions Settings";
    ui_label = "Noise Scale";
    ui_tooltip = "Adjust the scale of noise for random direction sampling";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.05;
>
= 1.0;

uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth multiplier";
    ui_min = 0.001; ui_max = 20.00;
    ui_step = 0.001;
> = 1.0;

uniform float DepthThreshold
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
    
    texture2D SSAOTex
    {
        Width = RENDER_WIDTH;
        Height = RENDER_HEIGHT;
        Format = RGBA16F;
    };

    sampler2D sSSAO
    {
        Texture = SSAOTex;
    };

    /*----------------.
    | :: Functions :: |
    '----------------*/

    float GetLinearDepth(float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords) * DepthMultiplier;
    }

    float3 GetNormalFromDepth(float2 coords)
    {
        float2 texelSize = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float depthCenter = GetLinearDepth(coords);
        float depthX = GetLinearDepth(coords + float2(texelSize.x, 0.0));
        float depthY = GetLinearDepth(coords + float2(0.0, texelSize.y));
        float3 deltaX = float3(texelSize.x, 0.0, depthX - depthCenter);
        float3 deltaY = float3(0.0, texelSize.y, depthY - depthCenter);
        float3 normal = normalize(cross(deltaX, deltaY));

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

        float3 normal = GetNormalFromDepth(texcoord);
        dir = normalize(reflect(dir, normal));

        return normalize(dir * noiseScale + float3(0.5, 0.5, 0.5));
    }

    // Hemisphere Sampling
    float3 HemisphereSampling(int sampleIndex, float3 normal)
    {
        float phi = (sampleIndex + 0.5) * 3.14159265359 * 2.0 / 16.0; // amostras
        float theta = acos(2.0 * frac(sin(phi) * 0.5 + 0.5) - 1.0);

        float3 dir;
        dir.x = sin(theta) * cos(phi);
        dir.y = sin(theta) * sin(phi);
        dir.z = cos(theta);
        dir = normalize(reflect(dir, normal));
        return dir;
    }

    // Main SSAO
    float4 SSAO_PS(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, texcoord);
        float depthValue = GetLinearDepth(texcoord);
        float3 normal = GetNormalFromDepth(texcoord);
        float occlusion = 0.0;

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

            if (sampleDepth < DepthThreshold)
            {
                float rangeCheck = exp(-abs(depthValue - sampleDepth) / falloff);
                occlusion += (sampleDepth < depthValue) ? rangeCheck : 0.0;
            }
        }

        occlusion = (occlusion / sampleCount) * intensity;

        return float4(occlusion, occlusion, occlusion, 1.0);
    }
    
    float4 Composite_PS(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, uv);
        float occlusion = tex2D(sSSAO, uv).r;
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetNormalFromDepth(uv);

         // View modes: Normal, AO Debug, Depth, Sky Debug
        if (ViewMode == 0)
        {
            if (depthValue >= DepthThreshold)
            {
                return originalColor * (1.0 - saturate(occlusion));
            }
            return originalColor * (1.0 - saturate(occlusion));
        }
        else if (ViewMode == 1)
        {
            return float4(saturate(occlusion), saturate(occlusion), saturate(occlusion), 1.0);
        }
        else if (ViewMode == 2)
        {
            return float4(depthValue, depthValue, depthValue, 1.0);
        }
        else if (ViewMode == 3)
        {
            return (depthValue >= DepthThreshold)
                ? float4(1.0, 0.0, 0.0, 1.0)
                : float4(depthValue, depthValue, depthValue, 1.0);
        }
        else if (ViewMode == 4)
        {
            return float4(normal * 0.5 + 0.5, 1.0);
        }
        return originalColor;
    }

    /*-----------------.
    | :: Techniques :: |
    '-----------------*/

    technique SSAO
    {
        pass 
        {
            VertexShader = PostProcessVS;
            PixelShader = SSAO_PS;
            RenderTarget = SSAOTex;
        }
        pass 
        {
            VertexShader = PostProcessVS;
            PixelShader = Composite_PS;
        }
    }
}
