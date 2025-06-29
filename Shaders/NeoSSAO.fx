/*------------------.
| :: Description :: |
'-------------------/

_  _ ____ ____ ____ ____ ____ ____
|\ | |___ |  | [__  [__  |__| |  | 
| \| |___ |__| ___] ___] |  | |__| 
                                                                       
    Version 1.8.1
    Author: Barbatos Bachiko
    License: MIT
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.8
    x IMMERSE Launchpad working
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
#define fov 28.6
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 
#define AspectRatio BUFFER_WIDTH/BUFFER_HEIGHT
#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    
#define MVErrorTolerance 0.96
#define SkyDepth 0.99
#define MAX_Frames 64
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
        ui_type = "drag";
        ui_label = "AO Intensity";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
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
    = 0.015;
    
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

    uniform float DepthMultiplier
    <
        ui_type = "slider";
        ui_category = "Depth/Normals";
        ui_label = "Depth Multiplier";
        ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
    >
    = 1.0;
    
    uniform float DepthThreshold
    <
        ui_type = "slider";
        ui_category = "Depth/Normals";
        ui_label = "Sky Threshold";
        ui_tooltip = "Set the depth threshold to ignore the sky.";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    >
    = 0.99; 
    
    uniform bool bSmoothNormals <
     ui_category = "Depth/Normals";
    ui_label = "Smooth Normals";
    > = false;

    uniform float BrightnessThreshold
    <
        ui_category = "Visibility";
        ui_type = "slider";
        ui_label = "Brightness Threshold";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    >
    = 1.0; 
    
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
static const float DepthSmoothEpsilon = 0.0003;

static const float PI2div360 = 0.01745329;
#define rad(x) (x * PI2div360)

    /*---------------.
    | :: Textures :: |
    '---------------*/
#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
}
float2 sampleMotion(float2 texcoord) {
    return tex2D(Deferred::sMotionVectorsTex, texcoord).rg;
}

#elif USE_VORT_MOTION
texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC };
float2 sampleMotion(float2 texcoord) {
    return tex2D(sMotVectTexVort, texcoord).rg;
}
#else

texture texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler sTexMotionVectorsSampler
{
    Texture = texMotionVectors;S_PC
};
float2 sampleMotion(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace NEOSPACEAO
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
    
    /*----------------.
    | :: Functions :: |
    '----------------*/
    
    float lum(float3 color)
    {
        return (color.r + color.g + color.b) * 0.3333333;
    }
   
    float3 UVtoPos(float2 texcoord)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, getDepth(texcoord) * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord.xy *= rad(fov);
        return scrncoord.xyz;
    }

    float3 UVtoPos(float2 texcoord, float depth)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, depth * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= AspectRatio;
        scrncoord *= rad(fov);
        return scrncoord.xyz;
    }

    float2 PostoUV(float3 position)
    {
        float2 scrnpos = position.xy;
        scrnpos /= rad(fov);
        scrnpos.x /= AspectRatio;
        scrnpos /= position.z;
        return scrnpos / 2 + 0.5;
    }

    float3 computeNormal(float2 texcoord)
    {
        float2 p = pix;
        float3 u, d, l, r, u2, d2, l2, r2;
	
        u = UVtoPos(texcoord + float2(0, p.y));
        d = UVtoPos(texcoord - float2(0, p.y));
        l = UVtoPos(texcoord + float2(p.x, 0));
        r = UVtoPos(texcoord - float2(p.x, 0));
	
        p *= 2;
	
        u2 = UVtoPos(texcoord + float2(0, p.y));
        d2 = UVtoPos(texcoord - float2(0, p.y));
        l2 = UVtoPos(texcoord + float2(p.x, 0));
        r2 = UVtoPos(texcoord - float2(p.x, 0));
	
        u2 = u + (u - u2);
        d2 = d + (d - d2);
        l2 = l + (l - l2);
        r2 = r + (r - r2);
	
        float3 c = UVtoPos(texcoord);
	
        float3 v = u - c;
        float3 h = r - c;
	
        if (abs(d2.z - c.z) < abs(u2.z - c.z))
            v = c - d;
        if (abs(l2.z - c.z) < abs(r2.z - c.z))
            h = c - l;
	
        return normalize(cross(v, h));
    }
    
    // SmoothNormal by AlucardDH MIT Licence
    float3 GetNormal(float2 texcoord)
    {
        float3 offset = float3(ReShade::PixelSize, 0.0);
        float3 normal = computeNormal(texcoord);

        if (bSmoothNormals)
        {
            float2 offset2 = ReShade::PixelSize * 7.5 * (1.0 - getDepth(texcoord));
    
            float3 normalTop = computeNormal(texcoord - float2(0, offset2.y));
            float3 normalBottom = computeNormal(texcoord + float2(0, offset2.y));
            float3 normalLeft = computeNormal(texcoord - float2(offset2.x, 0));
            float3 normalRight = computeNormal(texcoord + float2(offset2.x, 0));
    
            float weightTop = smoothstep(1, 0, distance(normal, normalTop) * 1.5) * 2;
            float weightBottom = smoothstep(1, 0, distance(normal, normalBottom) * 1.5) * 2;
            float weightLeft = smoothstep(1, 0, distance(normal, normalLeft) * 1.5) * 2;
            float weightRight = smoothstep(1, 0, distance(normal, normalRight) * 1.5) * 2;
    
            float4 weightedNormal =
        float4(normal, 1.0) +
        float4(normalTop * weightTop, weightTop) +
        float4(normalBottom * weightBottom, weightBottom) +
        float4(normalLeft * weightLeft, weightLeft) +
        float4(normalRight * weightRight, weightRight);
    
            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }

        return normal;
    }
    
    float3 getNormal(float2 coords)
    {
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
    }

    float RayMarching(in float2 texcoord, in float3 rayDir)
    {
        float occlusion = 0.0;
        float depthValue = getDepth(texcoord);
        float3 normal = getNormal(texcoord);

        float stepSize = ReShade::PixelSize.x / RayScale;
        int numSteps = max(int(MaxRayDistance / stepSize), 2);

    [loop]
        for (int i = 0; i < numSteps; i++)
        {
            float t = float(i) * rcp(float(numSteps - 1));
            float sampleDistance = mad(t, t * MaxRayDistance, 0.0);
            float2 sampleCoord = mad(rayDir.xy, sampleDistance, texcoord);

            sampleCoord = clamp(sampleCoord, 0.0, 1.0);

            float sampleDepth = getDepth(sampleCoord);
            float depthDiff = depthValue - sampleDepth;
            float hitFactor = saturate(depthDiff * rcp(DepthSmoothEpsilon + 1e-6));

            if (hitFactor > 0.01)
            {
                float angleFactor = saturate(dot(normal, +rayDir));
                float weight = (1.0 - (sampleDistance / MaxRayDistance)) * hitFactor * angleFactor;

                occlusion += weight;

                if (hitFactor < 0.001)
                    break;
            }
        }

        return occlusion;
    }

    static const float3 hemisphereSamples[12] =
    {
        float3(0.5381, 0.1856, 0.4319),
    float3(0.1379, 0.2486, 0.6581),
    float3(-0.3371, 0.5679, 0.6981),
    float3(-0.7250, 0.4233, 0.5429),
    float3(-0.4571, -0.5329, 0.7116),
    float3(0.0649, -0.9270, 0.3706),
    float3(0.3557, -0.6380, 0.6827),
    float3(0.6494, -0.2861, 0.7065),
    float3(0.7969, 0.5845, 0.1530),
    float3(-0.0195, 0.8512, 0.5234),
    float3(-0.5890, -0.7287, 0.3487),
    float3(-0.6729, 0.2057, 0.7117)
    };

    float4 PS_SSAO(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float depthValue = getDepth(uv);
        float3 normal = getNormal(uv);
        float3 originalColor = GetColor(uv).rgb;
    
        float brightness = lum(originalColor);
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
            occlusion += RayMarching(uv, sampleDir * SampleRadius);
        }
    
        occlusion = (occlusion / sampleCount) * Intensity;
        occlusion *= brightnessFactor;
        float fade = (depthValue < FadeStart) ? 1.0 : saturate((FadeEnd - depthValue) / (FadeEnd - FadeStart));
        occlusion *= fade;

        return float4(occlusion, occlusion, occlusion, 1.0);
    }

    float2 GetMotionVector(float2 texcoord)
    {
        float2 p = pix;
        float2 MV = sampleMotion(texcoord);

        if (MVErrorTolerance < 1)
        {
            if (abs(MV.x) < p.x && abs(MV.y) < p.y)
                MV = 0;
        }

#if USE_MARTY_LAUNCHPAD_MOTION
    MV = tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
    MV = tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#endif

        return MV;
    }
   
    float4 PS_Temporal(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outHistoryLength : SV_Target1) : SV_Target
    {
        float2 motion = GetMotionVector(uv);
        float2 reprojectedUV = uv + motion;
    
        float3 currentAO = tex2Dlod(sAO, float4(uv, 0, 0)).rgb;
        float3 historyAO= tex2Dlod(sAO_HISTORY, float4(reprojectedUV, 0, 0)).rgb;
    
        float depth = getDepth(uv);
        bool validHistory = (depth <= SkyDepth) &&
                       all(saturate(reprojectedUV) == reprojectedUV) &&
                       FRAME_COUNT > 1;
    
        float3 blendedAO = currentAO;
    
        if (EnableTemporal && validHistory)
        {
            if (AccumFramesAO > 0)
            {
                float alphaGI = 1.0 / min(FRAME_COUNT, (float) AccumFramesAO);
                blendedAO = lerp(historyAO, currentAO, alphaGI);
            }
       
        }
    
        outHistoryLength = validHistory ? min(FRAME_COUNT, MAX_Frames) : 0;
        return float4(blendedAO, currentAO.r);
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
        ui_tooltip = "RT Ambient Occlusion";
    >
    {
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
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
