/*------------------.
| :: Description :: |
'-------------------/

NeoSSAO
                                                                                       
    Version 1.1
	Author: Barbatos Bachiko
	License: MIT

	About: Screen-Space Ambient Occlusion using ray marching.
	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 1.1
    + New textures
    - Render Scale to 0.333
*/ 

#ifndef RENDER_SCALE
#define RENDER_SCALE 0.333
#endif
#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 
#define RENDER_WIDTH (INPUT_WIDTH * RENDER_SCALE)
#define RENDER_HEIGHT (INPUT_HEIGHT * RENDER_SCALE) 

/*-------------------.
| :: Includes ::    |
'-------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*-------------------.
| :: Settings ::    |
'-------------------*/

uniform int ViewMode
<
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for SSAO";
    ui_items = "Normal\0AO Debug\0Depth\0Sky Debug\0Normal Debug\0";
>
= 0;

uniform int QualityLevel
<
    ui_type = "combo";
    ui_label = "Quality Level";
    ui_tooltip = "Select quality level for ambient occlusion";
    ui_items = "Low\0Medium\0High\0";
>
= 2; 

uniform float Intensity
<
    ui_type = "slider";
    ui_label = "Occlusion Intensity";
    ui_tooltip = "Adjust the intensity of ambient occlusion";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
>
= 0.2; 

uniform float SampleRadius
<
    ui_type = "slider";
    ui_label = "Sample Radius";
    ui_tooltip = "Adjust the radius of the samples for SSAO";
    ui_min = 0.001; ui_max = 5.0; ui_step = 0.001;
>
= 1.0; 

uniform float MaxRayDistance
<
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "Max Ray Distance";
    ui_tooltip = "Maximum distance for ray marching";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
>
= 0.010;

uniform float DepthMultiplier
<
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Multiplier";
    ui_tooltip = "Adjust the depth multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
>
= 1.0;

uniform float DepthThreshold
<
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Threshold (Sky)";
    ui_tooltip = "Set the depth threshold to ignore the sky during occlusion.";
    ui_min = 0.9; ui_max = 1.0; ui_step = 0.01;
>
= 0.95; 

uniform float4 OcclusionColor
<
    ui_category = "Extra";
    ui_type = "color";
    ui_label = "Occlusion Color";
    ui_tooltip = "Select the color for ambient occlusion.";
>
= float4(0.0, 0.0, 0.0, 1.0);

/*---------------.
| :: Textures :: |
'---------------*/

namespace NEOSSAOMEGAETC
{
    texture2D SSAOTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };

    sampler2D sSSAO
    {
        Texture = SSAOTex;
    };

    texture DepthTex : DEPTH;
    
    sampler2D DepthSampler
    {
        Texture = DepthTex;
    };
    
    /*----------------.
    | :: Functions :: |
    '----------------*/

    float GetLinearDepth(float2 coords, int mipLevel = 0)
    {
        float depth = (mipLevel > 0)
            ? tex2Dlod(DepthSampler, float4(coords, 0, mipLevel)).r
            : tex2D(DepthSampler, coords).r;
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

    // Ray marching to calculate occlusion
    float RayMarching(float2 texcoord, float3 rayDir, float3 normal)
    {
        float occlusion = 0.0;
        float depthValue = GetLinearDepth(texcoord);
        float currentPos = 0.0;
        float stepSize = ReShade::PixelSize / RENDER_SCALE;
        float maxDistance = MaxRayDistance;
        int mipLevel = clamp(int(log2(SampleRadius)), 0, 5);

        while (currentPos < maxDistance)
        {
            float2 sampleCoord = clamp(texcoord + rayDir.xy * currentPos, 0.0, 1.0);
            float sampleDepth = GetLinearDepth(sampleCoord, mipLevel);
        
            if (sampleDepth < depthValue)
            {
                occlusion += (1.0 - (currentPos / maxDistance));
            }

            currentPos += stepSize;
        }

        return occlusion;
    }

    float4 SSAO_PS(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetNormalFromDepth(uv);
        float occlusion = 0.0;

        int sampleCount = (QualityLevel == 0) ? 8 : (QualityLevel == 1) ? 16 : 32;
        for (int i = 0; i < sampleCount; i++)
        {
            float phi = (i + 0.5) * 6.28318530718 / sampleCount;
            float theta = acos(2.0 * frac(sin(phi) * 0.5 + 0.5) - 1.0);
            float3 sampleDir = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            occlusion += RayMarching(uv, sampleDir * SampleRadius, normal);
        }
        occlusion = (occlusion / sampleCount) * Intensity;
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
                return originalColor * (1.0 - saturate(occlusion)) + OcclusionColor * saturate(occlusion);
            }
            return originalColor * (1.0 - saturate(occlusion)) + OcclusionColor * saturate(occlusion);
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

    /*-------------------.
    | :: Techniques ::   |
    '-------------------*/

    technique NeoSSAO
    {
        pass SSAOPass
        {
            VertexShader = PostProcessVS;
            PixelShader = SSAO_PS;
            RenderTarget = SSAOTex;
        }
        pass Composite
        {
            VertexShader = PostProcessVS;
            PixelShader = Composite_PS;
        }
    }
} // https://www.comp.nus.edu.sg/~lowkl/publications/mssao_visual_computer_2012.pdf (this is just my study material, it doesn't mean there are implementations from here)
