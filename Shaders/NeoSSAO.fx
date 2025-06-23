/*------------------.
| :: Description :: |
'-------------------/

_  _ ____ ____ ____ ____ ____ ____
|\ | |___ |  | [__  [__  |__| |  | 
| \| |___ |__| ___] ___] |  | |__| 
                                                                       
    Version 1.7.71
    Author: Barbatos Bachiko
    License: MIT
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.7.71
    * Revised
    x tiny fix
    
*/ 

#include "ReShade.fxh"
    
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define INPUT_WIDTH BUFFER_WIDTH 
#define INPUT_HEIGHT BUFFER_HEIGHT 

#ifndef RES_SCALE
#define RES_SCALE 0.8
#endif
#define RES_WIDTH (INPUT_WIDTH * RES_SCALE)
#define RES_HEIGHT (INPUT_HEIGHT * RES_SCALE) 

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
    
    uniform int ViewMode
    < 
        ui_category = "Geral";
        ui_type = "combo";
        ui_label = "View Mode";
        ui_tooltip = "Select the view mode for SSAO";
        ui_items = "None\0AO Debug\0Depth\0Sky Debug\0Normal Debug\0";
    >
    = 0;

    uniform float Intensity
    <
        ui_category = "Geral";
        ui_type = "slider";
        ui_label = "Occlusion Intensity";
        ui_min = 0.0; ui_max = 20.0; ui_step = 0.01;
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
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    >
    = 1.0;
    
    uniform float DepthMultiplier
    <
        ui_type = "slider";
        ui_category = "Depth/Normals";
        ui_label = "Depth Multiplier";
        ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
    >
    = 1.0;
    
    uniform float DepthSmoothEpsilon
    <
        ui_type = "slider";
        ui_category = "Depth/Normals";
        ui_label = "Depth Smooth Epsilon";
        ui_tooltip = "Controls the smoothing of depth comparison";
        ui_min = 0.0001; ui_max = 0.01; ui_step = 0.0001;
    > = 0.0005;
    
    uniform float DepthThreshold
    <
        ui_type = "slider";
        ui_category = "Depth/Normals";
        ui_label = "Depth Threshold (Sky)";
        ui_tooltip = "Set the depth threshold to ignore the sky.";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.50; 
    
uniform bool bSmoothNormals <
 ui_category = "Depth/Normals";
    ui_label = "Smooth Normals";
> = false;

uniform bool EnableTemporal
    <
        ui_category = "Temporal";
        ui_type = "checkbox";
        ui_label = "Temporal Filtering";
    >
    = false;

uniform float AccumFramesAO <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "AO Temporal";
    ui_min = 0.0; ui_max = 16.0; ui_step = 1.0;
> = 1.0;

    uniform float BrightnessThreshold
    <
        ui_category = "Visibility";
        ui_type = "slider";
        ui_label = "Brightness Threshold";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 0.92; 
    
    uniform float4 OcclusionColor
    <
        ui_category = "Extra";
        ui_type = "color";
        ui_label = "Occlusion Color";
        ui_tooltip = "Select the color for ambient occlusion.";
    >
    = float4(0.0, 0.0, 0.0, 1.0);
    
uniform int FRAME_COUNT < source = "framecount"; >;
static const float SampleRadius = 1.0;
static const int SampleCount = 8;
static const float RayScale = 0.222;
static const float EnableBrightnessThreshold = true;

    /*---------------.
    | :: Textures :: |
    '---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex;  };
}
#elif USE_VORT_MOTION
    texture2D MotVectTexVort {  Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC  };
#else
    texture texMotionVectors
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler sTexMotionVectorsSampler
    {
        Texture = texMotionVectors;
    S_PC
    };
#endif

namespace NEOSPACE
{
    texture2D AO
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    texture2D AO_TEMP
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    texture2D AO_HISTORY
    {
        Width = RES_WIDTH;
        Height = RES_HEIGHT;
        Format = RGBA8;
    };

    sampler2D sAO
    {
        Texture = AO;
        SRGBTexture = false;
    };

    sampler2D sAO_TEMP
    {
        Texture = AO_TEMP;
        SRGBTexture = false;
    };

    sampler2D sAO_HISTORY
    {
        Texture = AO_HISTORY;
        SRGBTexture = false;
    };
    
    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = normalTex;S_PC
    };

    texture depthTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
        MipLevels = 6;
    };
    sampler sDepth
    {
        Texture = depthTex;
        MinLOD = 0.0f;
        MaxLOD = 5.0f;
    };
    
    
    /*----------------.
    | :: Functions :: |
    '----------------*/
    
    // MIT License functions
    float3 getWorldPositionForNormal(float2 coords)
    {
        float depth = getDepth(coords).x;
        return float3((coords - 0.5) * depth, depth);
    }

    float4 mulByA(float4 v)
    {
        return float4(v.rgb * v.a, v.a);
    }

    float4 computeNormal(float3 wpCenter, float3 wpNorth, float3 wpEast)
    {
        return float4(normalize(cross(wpCenter - wpNorth, wpCenter - wpEast)), 1.0);
    }

    float GetLuminance(float3 color)
    {
        return dot(color, float3(0.299, 0.587, 0.114));
    }

    float4 computeNormal(float2 coords, float3 offset, bool reverse)
    {
        float3 posCenter = getWorldPositionForNormal(coords);
        float3 posNorth = getWorldPositionForNormal(coords - (reverse ? -1 : 1) * offset.zy);
        float3 posEast = getWorldPositionForNormal(coords + (reverse ? -1 : 1) * offset.xz);
    
        float4 r = computeNormal(posCenter, posNorth, posEast);
        float mD = max(abs(posCenter.z - posNorth.z), abs(posCenter.z - posEast.z));
        if (mD > 16)
            r.a = 0;
        return r;
    }

    float3 GetNormal(float2 coords)
    {
        float3 offset = float3(ReShade::PixelSize, 0.0);
        float4 normal = computeNormal(coords, offset, false);
    
        if (normal.a == 0)
        {
            normal = computeNormal(coords, offset, true);
        }
    
        if (bSmoothNormals)
        {
            float3 offset2 = offset * 7.5 * (1.0 - getDepth(coords).x);
            float4 normalTop = computeNormal(coords - offset2.zy, offset, false);
            float4 normalBottom = computeNormal(coords + offset2.zy, offset, false);
            float4 normalLeft = computeNormal(coords - offset2.xz, offset, false);
            float4 normalRight = computeNormal(coords + offset2.xz, offset, false);
        
            normalTop.a *= smoothstep(1, 0, distance(normal.xyz, normalTop.xyz) * 1.5) * 2;
            normalBottom.a *= smoothstep(1, 0, distance(normal.xyz, normalBottom.xyz) * 1.5) * 2;
            normalLeft.a *= smoothstep(1, 0, distance(normal.xyz, normalLeft.xyz) * 1.5) * 2;
            normalRight.a *= smoothstep(1, 0, distance(normal.xyz, normalRight.xyz) * 1.5) * 2;
        
            float4 normal2 = mulByA(normal) + mulByA(normalTop) + mulByA(normalBottom) +
                         mulByA(normalLeft) + mulByA(normalRight);
        
            if (normal2.a > 0)
            {
                normal2.xyz /= normal2.a;
                normal.xyz = normalize(normal2.xyz);
            }
        }
    
        return normal.xyz;
    }

    float3 getNormal(float2 coords)
    {
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
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
        data.depthValue = getDepth(texcoord);
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

            float sampleDepth = getDepth(sampleCoord);
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
        float depthValue = getDepth(uv);
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
    
    float2 GetMotionVector(float2 texcoord)
    {
#if USE_MARTY_LAUNCHPAD_MOTION
    return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
    return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#else
        return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).xy;
#endif
    }
    
    // For Temporal
    float3 RGBToYCoCg(float3 rgb)
    {
        float Y = dot(rgb, float3(0.25, 0.5, 0.25));
        float Co = dot(rgb, float3(0.5, 0.0, -0.5));
        float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
        return float3(Y, Co, Cg);
    }

    float3 YCoCgToRGB(float3 ycocg)
    {
        float Y = ycocg.x;
        float Co = ycocg.y;
        float Cg = ycocg.z;
        return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
    }
   
   float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
 {
        float2 motion = GetMotionVector(uv);
        
        float3 currentAO = tex2Dlod(sAO, float4(uv, 0, 0)).rgb;
        float3 currentAO_YC = RGBToYCoCg(currentAO);
        float3 historyAO_YC = RGBToYCoCg(tex2Dlod(sAO_HISTORY, float4(uv + motion, 0, 0)).rgb);

        float accumulationWeight = 1.0f;
        float3 accumulation = float3(1.0, 1.0, 1.0);
        float3 safeAccumulation = max(float3(1e-10, 1e-10, 1e-10), accumulation + accumulationWeight);
        float3 alpha = accumulationWeight / safeAccumulation;

        float3 blendedAO_YC = lerp(historyAO_YC, currentAO_YC, alpha);
        float3 resultAO = YCoCgToRGB(blendedAO_YC);

        if (EnableTemporal && AccumFramesAO > 0 && FRAME_COUNT > 1)
        {
            uint N = min(FRAME_COUNT, (uint) AccumFramesAO);
            float4 prev = tex2Dlod(sAO_HISTORY, float4(uv + motion, 0, 0));
            resultAO = (prev.rgb * (N - 1) + resultAO) / N;
        }
        return float4(resultAO, currentAO.r);
    }

    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }
    
    // History
    float4 PS_SaveHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float occlusion = EnableTemporal
        ? tex2Dlod(sAO_TEMP, float4(uv, 0, 0)).r 
        : tex2Dlod(sAO, float4(uv, 0, 0)).r;
        return float4(occlusion, occlusion, occlusion, 1.0);
    }
    
    // Final Image
    float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);
        
        float occlusion = EnableTemporal 
        ? tex2Dlod(sAO_TEMP, float4(uv, 0, 0)).r 
        : tex2Dlod(sAO, float4(uv, 0, 0)).r;

        float depthValue = getDepth(uv);
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
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
            RenderTarget1 = depthTex;
        }
        pass SSAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SSAO;
            RenderTarget = AO;
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Temporal;
            RenderTarget = AO_TEMP;
            ClearRenderTargets = true;
        }
        pass SaveHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistory;
            RenderTarget = AO_HISTORY;
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
