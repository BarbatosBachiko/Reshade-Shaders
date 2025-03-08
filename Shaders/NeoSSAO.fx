/*------------------.
| :: Description :: |
'-------------------/

_  _ ____ ____ ____ ____ ____ ____
|\ | |___ |  | [__  [__  |__| |  | 
| \| |___ |__| ___] ___] |  | |__| 
                                                                       
    Version 1.7
    Author: Barbatos Bachiko
    License: MIT

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.7
    * Temporal Filter
    - delete unused textures
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

    uniform int SampleCount
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "SampleCount";
        ui_min = 0.0; ui_max = 16.0; ui_step = 1.0;
    >
    = 10.0;

    uniform float Intensity
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "Occlusion Intensity";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    >
    = 0.5; 

    uniform float SampleRadius
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "Sample Radius";
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
        ui_items = "Horizon Only\0Vertical Only\0Unilateral\0Bidirectional\0";
    >
    = 3;
    
    uniform float FadeStart
    <
        ui_category = "Fade";
        ui_type = "slider";
        ui_label = "Fade Start";
        ui_tooltip = "Distance at which AO starts to fade out.";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.0;

    uniform float FadeEnd
    <
        ui_category = "Fade";
        ui_type = "slider";
        ui_label = "Fade End";
        ui_tooltip = "Distance at which AO completely fades out.";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 1.0;
    
    uniform float DepthMultiplier
    <
        ui_type = "slider";
        ui_category = "Depth";
        ui_label = "Depth Multiplier";
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
    > = 0.0005;
    
    uniform float DepthThreshold
    <
        ui_type = "slider";
        ui_category = "Depth";
        ui_label = "Depth Threshold (Sky)";
        ui_tooltip = "Set the depth threshold to ignore the sky.";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.50; 
    
    uniform bool EnableTemporal
    <
        ui_category = "Temporal";
        ui_type = "checkbox";
        ui_label = "Temporal Filtering";
    >
    = false;

    uniform float TemporalFilterStrength
    <
        ui_category = "Temporal";
        ui_type = "slider";
        ui_label = "Temporal Filter";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_tooltip = "Blend factor between current SSAO and history.";
    >
    = 0.5;

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

    texture2D ssaoTemporal
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    texture2D ssaoHistory
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    sampler2D sAO
    {
        Texture = AOTex;
        SRGBTexture = false;
    };

    sampler2D sTemporal
    {
        Texture = ssaoTemporal;
        SRGBTexture = false;
    };

    sampler2D sSSAOHistory
    {
        Texture = ssaoHistory;
        SRGBTexture = false;
    };
    
    /*----------------.
    | :: Functions :: |
    '----------------*/
    
    float GetLinearDepth(in float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords) * DepthMultiplier;
    }

    // From DisplayDepth.fx
    float3 GetScreenSpaceNormal(in float2 texcoord)
    {
        float3 offset = float3(BUFFER_PIXEL_SIZE, 0.0);
        float2 posCenter = texcoord.xy;
        float2 posNorth = posCenter - offset.zy;
        float2 posEast = posCenter + offset.xz;

        float depthCenter = GetLinearDepth(posCenter);
        float depthNorth = GetLinearDepth(posNorth);
        float depthEast = GetLinearDepth(posEast);

        float3 vertCenter = float3(posCenter - 0.5, 1) * depthCenter;
        float3 vertNorth = float3(posNorth - 0.5, 1) * depthNorth;
        float3 vertEast = float3(posEast - 0.5, 1) * depthEast;

        return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)) * 0.5 + 0.5;
    }
    
    float CalculateBrightness(float3 color)
    {
        return dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    }
    
    // Ray Marching 
    struct RayMarchData
    {
        float occlusion;
        float depthValue;
        float stepSize;
        int numSteps;
        float2 texcoord;
        float3 rayDir;
        float3 normal;
    };

    float RayMarching(in float2 texcoord, in float3 rayDir, in float3 normal)
    {
        RayMarchData data;
        data.occlusion = 0.0;
        data.depthValue = GetLinearDepth(texcoord);
        data.stepSize = ReShade::PixelSize.x / RayScale;
        data.numSteps = max(int(MaxRayDistance / data.stepSize), 2);
        data.texcoord = texcoord;
        data.rayDir = rayDir;
        data.normal = normal;

    [loop]
        for (int i = 0; i < data.numSteps; i++)
        {
            float t = float(i) * rcp(float(data.numSteps - 1));
            float sampleDistance = mad(t, t * MaxRayDistance, 0.0);
            float2 sampleCoord = mad(data.rayDir.xy, sampleDistance, data.texcoord);

            sampleCoord = clamp(sampleCoord, 0.0, 1.0);

            float sampleDepth = GetLinearDepth(sampleCoord);
            float depthDiff = data.depthValue - sampleDepth;
            float hitFactor = saturate(depthDiff * rcp(DepthSmoothEpsilon + 1e-6));

            if (hitFactor > 0.01)
            {
                data.occlusion += (1.0 - (sampleDistance / MaxRayDistance)) * hitFactor;
                if (hitFactor < 0.001)
                    break;
            }
        }

        return data.occlusion;
    }
    
    float4 PS_SSAO(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetScreenSpaceNormal(uv);
        float occlusion = 0.0;

        float3 originalColor = tex2D(ReShade::BackBuffer, uv).rgb;
        float brightness = CalculateBrightness(originalColor);
        float brightnessFactor = EnableBrightnessThreshold ? saturate(1.0 - smoothstep(BrightnessThreshold - 0.1, BrightnessThreshold + 0.1, brightness)) : 1.0;

        int sampleCount = clamp(SampleCount, 1, 16);
        float invSampleCount = 1.0 / sampleCount;
        float stepPhi = 6.28318530718 / sampleCount;

        float3 tangent = normalize(cross(normal, float3(0.0, 0.0, 1.0)));
        float3 bitangent = cross(normal, tangent);

        for (int i = 0; i < sampleCount; i++)
        {
            float phi = (i + 0.5) * stepPhi;

            float3 sampleDir;
            if (AngleMode == 3) // Bidirectional
            {
                sampleDir = tangent * cos(phi) + bitangent * sin(phi);
            }
            else if (AngleMode == 0) // Horizon Only
            {
                sampleDir = float3(cos(phi), sin(phi), 0.0);
            }
            else if (AngleMode == 1) // Vertical Only
            {
                sampleDir = (i % 2 == 0) ? float3(0.0, 1.0, 0.0) : float3(0.0, -1.0, 0.0);
            }
            else // Unilateral
            {
                float phi = (i + 0.5) * 3.14159265359 / sampleCount;
                sampleDir = float3(cos(phi), sin(phi), 0.0);
            }

            occlusion += RayMarching(uv, sampleDir * SampleRadius, normal);
        }

        occlusion = (occlusion / sampleCount) * Intensity;
        occlusion *= brightnessFactor;

        float fade = (depthValue < FadeStart) ? 1.0 : saturate((FadeEnd - depthValue) / (FadeEnd - FadeStart));
        occlusion *= fade;

        return float4(occlusion, occlusion, occlusion, 1.0);
    }

    // Temporal
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float currentOcclusion = tex2D(sAO, uv).r;
        float historyOcclusion = tex2D(sSSAOHistory, uv).r;
        float occlusion = EnableTemporal ? lerp(currentOcclusion, historyOcclusion, TemporalFilterStrength) : currentOcclusion;
        return float4(occlusion, occlusion, occlusion, 1.0);
    }

    // History
    float4 PS_SaveHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float occlusion = EnableTemporal ? tex2D(sTemporal, uv).r : tex2D(sAO, uv).r;
        return float4(occlusion, occlusion, occlusion, 1.0);
    }
    
    // Final Image
    float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, uv);
        float occlusion = EnableTemporal ? tex2D(sTemporal, uv).r : tex2D(sAO, uv).r;
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
    <
        ui_tooltip = "Screen Space Ambient Occlusion using ray marching";
    >
    {
        pass SSAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SSAO;
            RenderTarget = AOTex;
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget = ssaoTemporal;
            ClearRenderTargets = true;
        }
        pass SaveHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistory;
            RenderTarget = ssaoHistory;
        }
        pass Composite
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Composite;
            SRGBWriteEnable = false;
            BlendEnable = false;
        }
    }
} 
