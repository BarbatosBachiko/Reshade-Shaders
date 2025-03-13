/*------------------.
| :: Description :: |
'-------------------/

    NeoGI 
    Version 1.2
    Author: Barbatos Bachiko
    License: MIT

    About: Simple Indirect/Direct lighting using ray marching.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.2
    * Temporal Filter
    - Removed Blur
    + Manual SampleCount
*/

#include "ReShade.fxh"

namespace NEOSPACEG
{
#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 

#ifndef RES_SCALE
#define RES_SCALE 0.5
#endif
#define RES_WIDTH (INPUT_WIDTH * RES_SCALE)
#define RES_HEIGHT (INPUT_HEIGHT * RES_SCALE) 

#define TWO_PI 6.28318530718
#define PI 3.14159265359

    /*-------------------.
    | :: Settings ::    |
    '-------------------*/
    
    uniform int ViewMode
    < 
        ui_type = "combo";
        ui_category = "Geral";
        ui_label = "View Mode";
        ui_tooltip = "Select the view mode";
        ui_items = "Composite\0GI Debug\0Normal Debug\0Depth Debug\0";
    >
    = 0;

    uniform int SampleCount
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "SampleCount";
        ui_min = 0.0; ui_max = 32.0; ui_step = 1.0;
    >
    = 10.0;

    uniform float Intensity
    <
        ui_type = "slider";
        ui_category = "Geral";
        ui_label = "Intensity";
        ui_tooltip = "Adjust the intensity";
        ui_min = 0.5; ui_max = 2.0; ui_step = 0.01;
    >
    = 1.0; 

    uniform float Saturation
    <
        ui_type = "slider";
        ui_category = "Geral";
        ui_label = "Saturation";
        ui_tooltip = "Adjust GI saturation";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
    >
    = 1.5;

    uniform float SampleRadius
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "Sample Radius";
        ui_tooltip = "Adjust the radius of the samples";
        ui_min = 0.001; ui_max = 5.0; ui_step = 0.001;
    >
    = 0.180; 

    uniform float MaxRayDistance
    <
        ui_category = "Ray Marching";
        ui_type = "slider";
        ui_label = "Max Ray Distance";
        ui_tooltip = "Maximum distance for ray marching";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.035;
    
    uniform float RayScale
    <
        ui_category = "Ray Marching";
        ui_type = "slider";
        ui_label = "Ray Scale";
        ui_tooltip = "Adjust the ray scale";
        ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.05;

    uniform float FadeStart
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade Start";
        ui_tooltip = "Distance at which GI starts to fade out";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.0;

    uniform float FadeEnd
    <
        ui_category = "Fade Settings";
        ui_type = "slider";
        ui_label = "Fade End";
        ui_tooltip = "Distance at which GI completely fades out";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 1.0;

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
    >
    = 0.0005;
    
    uniform bool EnableTemporal
    <
        ui_category = "Temporal";
        ui_type = "checkbox";
        ui_label = "Temporal Filtering";
    >
    = true;

    uniform float TemporalFilterStrength
    <
        ui_category = "Temporal";
        ui_type = "slider";
        ui_label = "Temporal Filter Strength";
        ui_tooltip = "Blend factor between current GI and history";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.3;

    uniform int BlendMode
    <
        ui_type = "combo";
        ui_category = "Advanced";
        ui_label = "Blend Mode";
        ui_tooltip = "Select the blend mode for GI";
        ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
    >
    = 2;
    
    uniform int colorSpace
    <
        ui_category = "Advanced";
        ui_type = "combo";
        ui_label = "Color Space";
        ui_items = "Linear\0BColor\0";
    >
    = 0;
    
    uniform int AngleMode
    <
        ui_category = "Advanced";
        ui_type = "combo";
        ui_label = "Angle Mode";
        ui_items = "Horizon Only\0Vertical Only\0Unilateral\0Bidirectional\0";
    >
    = 3;
    
    uniform float3 LightDirection
    < 
        ui_category = "Advanced";
        ui_label = "Light Direction";
        ui_type = "slider"; 
        ui_min = -1.0; 
        ui_max = 1.0; 
    >
    = float3(0.3, 0.6, 1.0);

    uniform float3 LightColor
    < 
        ui_category = "Advanced";
        ui_label = "Light Color";
        ui_type = "color"; 
    >
    = float3(1.0, 1.0, 1.0);

    uniform float SampleJitter
    <
        ui_category = "Advanced";
        ui_type = "slider";
        ui_label = "Sample Jitter";
        ui_tooltip = "Controls the amount of jitter in the Sample Radius";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.0;
    
    /*---------------.
    | :: Textures :: |
    '---------------*/

    texture2D GITex
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    texture2D NormalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };

    texture2D giTemporal
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    texture2D giHistory
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    sampler2D sGI
    {
        Texture = GITex;
    };

    sampler2D sNormal
    {
        Texture = NormalTex;
    };

    sampler2D sGITemporal
    {
        Texture = giTemporal;
        SRGBTexture = false;
    };

    sampler2D sGIHistory
    {
        Texture = giHistory;
        SRGBTexture = false;
    };

    /*----------------.
    | :: Functions :: |
    '----------------*/

    float GetLinearDepth(float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords) * DepthMultiplier;
    }

    // From DisplayDepth.fx
    float3 GetScreenSpaceNormal(float2 texcoord)
    {
        float3 offset = float3(BUFFER_PIXEL_SIZE, 0.0);
        float2 posCenter = texcoord.xy;
        float2 posNorth = posCenter - offset.zy;
        float2 posEast = posCenter + offset.xz;

        float3 vertCenter = float3(posCenter - 0.5, 1.0) * GetLinearDepth(posCenter);
        float3 vertNorth = float3(posNorth - 0.5, 1.0) * GetLinearDepth(posNorth);
        float3 vertEast = float3(posEast - 0.5, 1.0) * GetLinearDepth(posEast);

        return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)) * 0.5 + 0.5;
    }

    float3 ApplyGammaCorrection(float3 color, int colorSpace)
    {
        if (colorSpace == 0)
        {
            return color; // Linear
        }
        else if (colorSpace == 1)
        {
            return (color < 0.5) ? (color * 2.0) : (exp(color * 1.0 / 2.4));
        }
        else
        {
            return color;
        }
    }

    // Ray Marching
    float3 RayMarching(float2 texcoord, float3 rayDir, float3 normal)
    {
        float3 giAccum = 0.0;
        float depthValue = GetLinearDepth(texcoord);
        float stepSize = ReShade::PixelSize.x / RayScale;
        int numSteps = max(int(MaxRayDistance / stepSize), 2);
        float invNumSteps = rcp(float(numSteps - 1));
        float depthEpsilon = rcp(DepthSmoothEpsilon + 1e-6);

    [loop]
        for (int i = 0; i < numSteps; i++)
        {
            float t = float(i) * invNumSteps;
            float sampleDistance = mad(t, t * MaxRayDistance, 0.0);
            float2 sampleCoord = mad(rayDir.xy, sampleDistance, texcoord);
            sampleCoord = clamp(sampleCoord, 0.0, 1.0);

            float sampleDepth = GetLinearDepth(sampleCoord);
            float depthDiff = depthValue - sampleDepth;
            float hitFactor = saturate(depthDiff * depthEpsilon);

            if (hitFactor > 0.01)
            {
                float4 sampleData = tex2Dlod(ReShade::BackBuffer, float4(sampleCoord, 0, 0));
                float3 sampleColor = ApplyGammaCorrection(sampleData.rgb, colorSpace);
            
                float3 lightDir = normalize(LightDirection);
                float diffuse = max(dot(normal, lightDir), 0.0);

                giAccum += sampleColor * hitFactor * diffuse * LightColor;

                if (hitFactor < 0.001)
                    break;
            }
        }
        return giAccum;
    }

    float4 PS_GI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetScreenSpaceNormal(uv);
        float3 giColor = float3(0.0, 0.0, 0.0);
        int sampleCount = clamp(SampleCount, 1, 32);
        float invSampleCount = 1.0 / sampleCount;
        float stepPhiLocal = (AngleMode == 2) ? (PI / sampleCount) : (TWO_PI / sampleCount);

        float randomValue = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
        float jitterFactor = 1.0 + SampleJitter * (randomValue * 2.0 - 1.0);
        float dynamicSampleRadius = SampleRadius * jitterFactor;

        if (AngleMode == 3) // Bidirectional
        {
            float3 tangent;
            if (abs(normal.z) > 0.999)
            {
                tangent = float3(1.0, 0.0, 0.0);
            }
            else
            {
                tangent = normalize(cross(normal, float3(0.0, 0.0, 1.0)));
            }
            float3 bitangent = cross(normal, tangent);

            for (int i = 0; i < sampleCount; i++)
            {
                float phi = (i + 0.5) * stepPhiLocal;
                float3 sampleDir = tangent * cos(phi) + bitangent * sin(phi);
                giColor += RayMarching(uv, sampleDir * dynamicSampleRadius, normal);
            }
        }
        else
        {
            for (int i = 0; i < sampleCount; i++)
            {
                float phi = (i + 0.5) * stepPhiLocal;
                float3 sampleDir = float3(0.0, 0.0, 0.0);

                switch (AngleMode)
                {
                    case 0: // Horizon Only
                        sampleDir = float3(cos(phi), sin(phi), 0.0);
                        break;
                    case 1: // Vertical Only
                        sampleDir = (i % 2 == 0) ? float3(0.0, 1.0, 0.0) : float3(0.0, -1.0, 0.0);
                        break;
                    case 2: // Unilateral
                        sampleDir = float3(cos(phi), sin(phi), 0.0);
                        break;
                    default:
                        break;
                }

                giColor += RayMarching(uv, sampleDir * dynamicSampleRadius, normal);
            }
        }

        giColor *= invSampleCount * Intensity;
        float fadeFactor = max(FadeEnd - FadeStart, 0.001);
        float fade = saturate((FadeEnd - depthValue) / fadeFactor);
        giColor *= fade;
    
        giColor = ApplyGammaCorrection(giColor, colorSpace);

        return float4(giColor, 1.0);
    }

    // Normals Debug
    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetScreenSpaceNormal(uv);
        return float4(normal, 1.0);
    }

    // Temporal
    float4 PS_Temporal_GI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 currentGI = tex2D(sGI, uv).rgb;
        float3 historyGI = tex2D(sGIHistory, uv).rgb;
        float3 gi = EnableTemporal ? lerp(currentGI, historyGI, TemporalFilterStrength) : currentGI;
        return float4(gi, 1.0);
    }

    // History
    float4 PS_SaveHistory_GI(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 gi = EnableTemporal ? tex2D(sGITemporal, uv).rgb : tex2D(sGI, uv).rgb;
        return float4(gi, 1.0);
    }

    // Final Image
    float4 PS_GI_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, uv);
        float3 giColor = EnableTemporal ? tex2D(sGITemporal, uv).rgb : tex2D(sGI, uv).rgb;
        float greyValue = dot(giColor, float3(0.299, 0.587, 0.114));
        float3 grey = float3(greyValue, greyValue, greyValue);
        giColor = lerp(grey, giColor, Saturation);

        if (ViewMode == 0)
        {
            if (BlendMode == 0) // Additive
                return float4(originalColor.rgb + giColor, originalColor.a);
            else if (BlendMode == 1) // Multiplicative
                return float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - giColor), originalColor.a);
            else if (BlendMode == 2) // Alpha Blend
            {
                float blendFactor = saturate(giColor.r);
                return float4(lerp(originalColor.rgb, giColor, blendFactor), originalColor.a);
            }
        }
        else if (ViewMode == 1) // GI Debug
            return float4(giColor, 1.0);
        else if (ViewMode == 2) // Normal Debug
        {
            float3 normal = GetScreenSpaceNormal(uv);
            return float4(normal * 0.5 + 0.5, 1.0);
        }
        else if (ViewMode == 3) // Depth Debug
        {
            float depth = GetLinearDepth(uv);
            return float4(depth, depth, depth, 1.0);
        }
        return originalColor;
    }

    /*----------------.
    | :: Techniques :: |
    '----------------*/
    technique NeoGI
    {
        pass Normal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = NormalTex;
        }
        pass GI
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GI;
            RenderTarget = GITex;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal_GI;
            RenderTarget = giTemporal;
            ClearRenderTargets = true;
        }
        pass SaveHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistory_GI;
            RenderTarget = giHistory;
        }
        pass Composite
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GI_Composite;
            SRGBWriteEnable = false;
            BlendEnable = false;
        }
    }
}
