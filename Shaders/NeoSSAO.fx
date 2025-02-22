/*------------------.
| :: Description :: |
'-------------------/

_  _ ____ ____ ____ ____ ____ ____
|\ | |___ |  | [__  [__  |__| |  | 
| \| |___ |__| ___] ___] |  | |__| 
                                                                       
    Version 1.4
    Author: Barbatos Bachiko
    License: MIT

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.4
    + Performance
    + Ray Marching
    x Blocks Artifacts
*/ 
namespace NEOSSAO
{
#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 

#ifndef RES_SCALE
#define RES_SCALE 0.888
#endif
#define RES_WIDTH (INPUT_WIDTH * RES_SCALE)
#define RES_HEIGHT (INPUT_HEIGHT * RES_SCALE) 
/*-------------------.
| :: Includes ::    |
'-------------------*/

#include "ReShade.fxh"

/*-------------------.
| :: Settings ::    |
'-------------------*/

    uniform int ViewMode
< 
    ui_category = "Geral";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for SSAO";
    ui_items = "Normal\0AO Debug\0Depth\0Sky Debug\0Normal Debug\0";
>
= 0;

    uniform int QualityLevel
<
    ui_category = "Geral";
    ui_type = "combo";
    ui_label = "Quality Level";
    ui_tooltip = "Select quality level for ambient occlusion";
    ui_items = "Low\0Medium\0High\0"; 
>
= 1;

    uniform float Intensity
<
    ui_category = "Geral";
    ui_type = "slider";
    ui_label = "Occlusion Intensity";
    ui_tooltip = "Adjust the intensity of ambient occlusion";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
>
= 0.2; 

    uniform float SampleRadius
<
    ui_category = "Geral";
    ui_type = "slider";
    ui_label = "Sample Radius";
    ui_tooltip = "Adjust the radius of the samples for SSAO";
    ui_min = 0.001; ui_max = 5.0; ui_step = 0.001;
>
= 0.5; 

    uniform float MaxRayDistance
<
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "Max Ray Distance";
    ui_tooltip = "Maximum distance for ray marching";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
>
= 0.011;

    uniform float RayScale
    <
        ui_category = "Ray Marching";
        ui_type = "slider";
        ui_label = "Ray Scale";
        ui_tooltip = "Adjust the ray scale";
        ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.222;
    
    uniform int AngleMode
<
    ui_category = "Ray Marching";
    ui_type = "combo";
    ui_label = "Angle Mode";
    ui_tooltip = "Horizon Only, Vertical Only, Unilateral or Bidirectional";
    ui_items = "Horizon Only\0Vertical Only\0Unilateral\0Bidirectional\0";
>
= 3;
    
    uniform float FadeStart
<
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade Start";
    ui_tooltip = "Distance at which SSAO starts to fade out.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
>
= 0.0;

    uniform float FadeEnd
<
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade End";
    ui_tooltip = "Distance at which SSAO completely fades out.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
>
= 0.5;
    
    uniform float DepthMultiplier
<
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Multiplier";
    ui_tooltip = "Adjust the depth multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
>
= 0.5;

    uniform float DepthSmoothEpsilon
    <
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Smooth Epsilon";
    ui_tooltip = "Controls the smoothing of depth comparison";
    ui_min = 0.0001; ui_max = 0.01; ui_step = 0.0001;
    > = 0.0001;
    
    uniform float DepthThreshold
<
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Threshold (Sky)";
    ui_tooltip = "Set the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
>
= 0.50; 

    uniform bool EnableBrightnessThreshold
< 
    ui_category = "Extra";
    ui_type = "checkbox";
    ui_label = "Enable Brightness Threshold"; 
    ui_tooltip = "Enable or disable the brightness threshold.";
> 
= false;

    uniform float BrightnessThreshold
<
    ui_category = "Extra";
    ui_type = "slider";
    ui_label = "Brightness Threshold";
    ui_tooltip = "Pixels with brightness above this threshold will have reduced occlusion.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
>
= 0.8; 
    
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

    texture2D AOTex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    sampler2D sAO
    {
        Texture = AOTex;
    };

/*----------------.
| :: Functions :: |
'----------------*/

    float GetLinearDepth(float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords) * DepthMultiplier;
    }

    //From DisplayDepth.fx
    float3 GetScreenSpaceNormal(float2 texcoord)
    {
        float3 offset = float3(BUFFER_PIXEL_SIZE, 0.0);
        float2 posCenter = texcoord.xy;
        float2 posNorth = posCenter - offset.zy;
        float2 posEast = posCenter + offset.xz;

        float3 vertCenter = float3(posCenter - 0.5, 1) * GetLinearDepth(posCenter);
        float3 vertNorth = float3(posNorth - 0.5, 1) * GetLinearDepth(posNorth);
        float3 vertEast = float3(posEast - 0.5, 1) * GetLinearDepth(posEast);

        return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)) * 0.5 + 0.5;
    }
    
    // Ray Marching
    float RayMarching(float2 texcoord, float3 rayDir, float3 normal)
    {
        float occlusion = 0.0;
        float depthValue = GetLinearDepth(texcoord);
        float stepSize = ReShade::PixelSize.x / RayScale;
        int numSteps = max(int(MaxRayDistance / stepSize), 2);
        float invNumSteps = rcp(float(numSteps - 1));

        bool hitDetected = false;
    
    [loop]
        for (int i = 0; i < numSteps; i++)
        {
            float t = float(i) * invNumSteps;
            float sampleDistance = mad(t, t * MaxRayDistance, 0.0);
            float2 sampleCoord = mad(rayDir.xy, sampleDistance, texcoord);
    
            if (any(sampleCoord < 0.0) || any(sampleCoord > 1.0))
                break;

            float sampleDepth = GetLinearDepth(sampleCoord);
            float depthDiff = depthValue - sampleDepth;
            float hitFactor = saturate(depthDiff * rcp(DepthSmoothEpsilon + 1e-6));
            if (hitFactor > 0.01)
            {
                occlusion += (1.0 - (sampleDistance / MaxRayDistance)) * AngleMode * hitFactor;
            
                if (hitFactor < 0.001)
                    break;
            }
        }

        return occlusion;
    }
    
    float CalculateBrightness(float3 color)
    {
        return dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    }
    
    float4 PS_SSAO(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetScreenSpaceNormal(uv);
        float occlusion = 0.0;

        float3 originalColor = tex2D(ReShade::BackBuffer, uv).rgb;
        float brightness = CalculateBrightness(originalColor);
        float brightnessFactor = EnableBrightnessThreshold ? saturate(1.0 - smoothstep(BrightnessThreshold - 0.1, BrightnessThreshold + 0.1, brightness)) : 1.0;
        
        int sampleCount = (QualityLevel == 0) ? 4 : (QualityLevel == 1) ? 8 : 16;
        float invSampleCount = 1.0 / sampleCount;
        float stepPhi = 6.28318530718 / sampleCount;

        if (AngleMode == 3) // Bidirectional
        {
            float3 tangent = normalize(cross(normal, float3(0.0, 0.0, 1.0)));
            float3 bitangent = cross(normal, tangent);
        
            for (int i = 0; i < sampleCount; i++)
            {
                float phi = (i + 0.5) * stepPhi;
                float3 sampleDir = tangent * cos(phi) + bitangent * sin(phi);
                occlusion += RayMarching(uv, sampleDir * SampleRadius, normal);
            }
        }
        else
        {
            for (int i = 0; i < sampleCount; i++)
            {
                float phi = (i + 0.5) * stepPhi;
                float3 sampleDir;
                if (AngleMode == 0) // Horizon Only
                {
                    sampleDir = float3(cos(phi), sin(phi), 0.0);
                }
                else if (AngleMode == 1) // Vertical Only
                {
                    sampleDir = (i % 2 == 0) ? float3(0.0, 1.0, 0.0) : float3(0.0, -1.0, 0.0);
                }
                else if (AngleMode == 2) // Unilateral
                {
                    float phi = (i + 0.5) * 3.14159265359 / sampleCount;
                    sampleDir = float3(cos(phi), sin(phi), 0.0);
                }
                occlusion += RayMarching(uv, sampleDir * SampleRadius, normal);
            }
        }

        occlusion = (occlusion / sampleCount) * Intensity;
        occlusion *= brightnessFactor;

        float fade = saturate((FadeEnd - depthValue) / (FadeEnd - FadeStart));
        occlusion *= fade;

        return float4(occlusion, occlusion, occlusion, 1.0);
    }

    float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, uv);
        float occlusion = tex2D(sAO, uv).r;
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetScreenSpaceNormal(uv);

        switch (ViewMode)
        {
            case 0: // Normal
                return (depthValue >= DepthThreshold)
                ? originalColor
                : originalColor * (1.0 - saturate(occlusion)) + OcclusionColor * saturate(occlusion);
        
            case 1: // AO Debug
                return float4(1.0 - occlusion, 1.0 - occlusion, 1.0 - occlusion, 1.0);

            case 2: // Depth
                return float4(depthValue, depthValue, depthValue, 1.0);

            case 3: // Sky Debug
                return (depthValue >= DepthThreshold)
                ? float4(1.0, 0.0, 0.0, 1.0)
                : float4(depthValue, depthValue, depthValue, 1.0);

            case 4: // Normal Debug
                return float4(normal * 0.5 + 0.5, 1.0);
        }

        return originalColor;
    }

/*-------------------.
| :: Techniques ::   |
'-------------------*/

    technique NeoSSAO
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SSAO;
            RenderTarget = AOTex;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Composite;
        }
    }
} // https://www.comp.nus.edu.sg/~lowkl/publications/mssao_visual_computer_2012.pdf (this is just my study material, it doesn't mean there are implementations from here)
