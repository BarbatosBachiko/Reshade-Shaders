/*------------------.
| :: Description :: |
'-------------------/

_  _ ____ ____ ____ ____ ____ ____
|\ | |___ |  | [__  [__  |__| |  | 
| \| |___ |__| ___] ___] |  | |__| 
                                                                       
    Version 1.7.5
    Author: Barbatos Bachiko
    License: MIT

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.7.5
    + Maintenance in SSAO and now use hemisphere samples
    * Replaced another shader's GetPosition with my GetPosition
    
*/ 

namespace NEOSPACE
{
    #include "ReShade.fxh"
    
#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 

#ifndef RES_SCALE
#define RES_SCALE 0.888
#endif
#define RES_WIDTH (INPUT_WIDTH * RES_SCALE)
#define RES_HEIGHT (INPUT_HEIGHT * RES_SCALE) 

    // version-number.fxh
#ifndef _VERSION_NUMBER_H
#define _VERSION_NUMBER_H

#define MAJOR_VERSION 1
#define MINOR_VERSION 7
#define PATCH_VERSION 5

#define BUILD_DOT_VERSION_(mav, miv, pav) #mav "." #miv "." #pav
#define BUILD_DOT_VERSION(mav, miv, pav) BUILD_DOT_VERSION_(mav, miv, pav)
#define DOT_VERSION_STR BUILD_DOT_VERSION(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

#define BUILD_UNDERSCORE_VERSION_(prefix, mav, miv, pav) prefix ## _ ## mav ## _ ## miv ## _ ## pav
#define BUILD_UNDERSCORE_VERSION(p, mav, miv, pav) BUILD_UNDERSCORE_VERSION_(p, mav, miv, pav)
#define APPEND_VERSION_SUFFIX(prefix) BUILD_UNDERSCORE_VERSION(prefix, MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

#endif  //  _VERSION_NUMBER_H

    uniform int APPEND_VERSION_SUFFIX(version) <
    ui_text = "Version: "
DOT_VERSION_STR;
    ui_label = " ";
    ui_type = "radio";
>;
    
    /*-------------------.
    | :: Settings ::    |
    '-------------------*/

    uniform int Guide <
    ui_text = 

        "OPTIMIZATION:\n"
        "1. SampleCount: Number of samples (0-12). Higher values ​​= more precision\n"
        "2. RayScale: Controls the precision of ray marching\n"
        "3. MaxRayDistance: Maximum distance of ray marching. Controls the range of AO\n\n"
        
        "RECOMMENDATIONS:\n"
        "1. Use BrightnessThreshold to preserve bright areas\n"
        "2. Use DAA.fx in Temporal mode to reduce flickering\n"
        "3. Adjust FadeStart/FadeEnd to control the distance of the effect\n\n"
        "4. Adjust the DepthSmoothEpsilon if the AO is showing some artifact\n"
        "5 Adjust DepthThreshold, Ignore the sky (adjust for your game)\n\n";
        
	ui_category = "Guide";
	ui_category_closed = true;
	ui_label = " ";
	ui_type = "radio";
>;
    
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
        ui_min = 0.0; ui_max = 12.0; ui_step = 1.0;
    >
    = 8.0;

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
    = 1.0;

    uniform float FOV
<
    ui_category = "Depth";
    ui_type = "slider";
    ui_label = "Field of View (FOV)";
    ui_tooltip = "Adjust the field of view for position reconstruction";
    ui_min = 1.0; ui_max = 270.0; ui_step = 1.0;
> = 60.0;
    
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
        ui_category = "Visibility";
        ui_type = "checkbox";
        ui_label = "Enable Brightness Threshold"; 
        ui_tooltip = "Enable or disable the brightness threshold.";
    > 
    = false;

    uniform float BrightnessThreshold
    <
        ui_category = "Visibility";
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

    texture2D AOTex1
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

    sampler2D sAO1
    {
        Texture = AOTex1;
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

    float3 GetPosition(float2 uv, float depth, float4 projParams)
    {
        float2 clip = uv * 2.0 - 1.0;
        clip.y *= -1.0;
        float3 ray = float3(clip.x / projParams.x, clip.y / projParams.y, 1.0);
        return ray * depth;
    }

    float4 GetProjParams(float fov, float aspectRatio)
    {
        float cot = 1.0 / tan(radians(fov) * 0.5);
        return float4(cot * aspectRatio, cot, 0, 0);
    }

    float3 GetNormal(in float2 uv)
    {
        const float2 offset1 = float2(0.0, BUFFER_PIXEL_SIZE.y);
        const float2 offset2 = float2(BUFFER_PIXEL_SIZE.x, 0.0);
    
        float depth = GetLinearDepth(uv);
        float depth1 = GetLinearDepth(uv + offset1);
        float depth2 = GetLinearDepth(uv + offset2);
    
        float4 proj = GetProjParams(FOV, BUFFER_ASPECT_RATIO);
    
        float3 pos = GetPosition(uv, depth, proj);
        float3 pos1 = GetPosition(uv + offset1, depth1, proj);
        float3 pos2 = GetPosition(uv + offset2, depth2, proj);
    
        float3 v1 = pos1 - pos;
        float3 v2 = pos2 - pos;
    
        float3 normal = normalize(cross(v1, v2));
        normal.z = abs(normal.z);
    
        return normal * 0.5 + 0.5;
    }
    
    float GetLum(float3 color)
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
    
    static const float3 hemisphereSamples[12] =
    {
    //45° 
        float3(0.707107, 0.0, 0.707107),
    float3(0.5, 0.5, 0.707107),
    float3(0.0, 0.707107, 0.707107),
    float3(-0.5, 0.5, 0.707107),
    float3(-0.707107, 0.0, 0.707107),
    float3(-0.5, -0.5, 0.707107),
    float3(0.0, -0.707107, 0.707107),
    float3(0.5, -0.5, 0.707107),

    //15°
    float3(0.258819, 0.0, 0.965926),
    float3(0.0, 0.258819, 0.965926),
    float3(-0.258819, 0.0, 0.965926),
    float3(0.0, -0.258819, 0.965926)
    };


    float4 PS_SSAO(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetNormal(uv);
        float3 originalColor = tex2D(ReShade::BackBuffer, uv).rgb;
    
        float brightness = GetLum(originalColor);
        float brightnessFactor = 1.0;
        if (EnableBrightnessThreshold)
        {
            brightnessFactor = saturate(1.0 - smoothstep(BrightnessThreshold - 0.1, BrightnessThreshold + 0.1, brightness));
        }
    
        float3 tangent = normalize(abs(normal.z) < 0.999 ? cross(normal, float3(0.0, 0.0, 1.0)) : float3(1.0, 0.0, 0.0));
        float3 bitangent = cross(normal, tangent);
        float3x3 TBN = float3x3(tangent, bitangent, normal);
    
        int sampleCount = clamp(SampleCount, 1, 12);
        float occlusion = 0.0;
    
        for (int i = 0; i < sampleCount; i++)
        {
            float3 sampleDir = mul(TBN, hemisphereSamples[i]);
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
        float currentOcclusion = tex2D(sAO1, uv).r;
        float historyOcclusion = tex2D(sSSAOHistory, uv).r;
        float occlusion = EnableTemporal ? lerp(currentOcclusion, historyOcclusion, TemporalFilterStrength) : currentOcclusion;
        return float4(occlusion, occlusion, occlusion, 1.0);
    }

    // History
    float4 PS_SaveHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float occlusion = EnableTemporal ? tex2D(sTemporal, uv).r : tex2D(sAO1, uv).r;
        return float4(occlusion, occlusion, occlusion, 1.0);
    }
    
    // Final Image
    float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = tex2D(ReShade::BackBuffer, uv);
        float occlusion = EnableTemporal ? tex2D(sTemporal, uv).r : tex2D(sAO1, uv).r;
        float depthValue = GetLinearDepth(uv);
        float3 normal = GetNormal(uv);

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
            RenderTarget = AOTex1;
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
