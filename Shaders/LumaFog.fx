/*

    LumaFog
    Version 1.0

    Author: Barbados Bachiko
    License: CC0
*/

#include "ReShade.fxh"
#define GetColor(coord) tex2Dlod(ReShade::BackBuffer, float4(coord, 0, 0))

//----------|
// :: UI :: |
//----------|

uniform float MistDensity <
    ui_category = "Fog Settings";
    ui_type = "slider";
    ui_label = "Fog Density";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.01;
> = 0.8;

uniform float3 MistColor <
    ui_category = "Fog Settings";
    ui_type = "color";
    ui_label = "Fog Color";
> = float3(0.8, 0.85, 0.9);

uniform float DepthInfluence <
    ui_category = "Fog Settings";
    ui_type = "slider";
    ui_label = "Depth Influence (Luminance)";
    ui_min = 0.0;
    ui_max = 10.0;
    ui_step = 0.1;
> = 10.0;

uniform int NoiseType <
    ui_category = "Noise";
    ui_type = "combo";
    ui_label = "Noise Type";
    ui_items = "None\0White Noise\0Interleaved Gradient Noise\0Blue Noise\0";
> = 0;

uniform float NoiseIntensity <
    ui_category = "Noise";
    ui_type = "slider";
    ui_label = "Noise Intensity";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.1;

uniform float AdaptationSpeed <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Adaptation Speed";
    ui_min = 0.01;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.05;

uniform float MistStartThreshold <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Fog Start Threshold (Bright Scene)";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.40;

uniform float MistEndThreshold <
    ui_category = "Scene Adaptation";
    ui_type = "slider";
    ui_label = "Fog End Threshold (Dark Scene)";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.15;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

namespace LumaFog
{
    texture LUM
    {
        Width = 1;
        Height = 1;
        Format = R32F;
    };
    sampler sLUM
    {
        Texture = LUM;
    };

    texture PREV_LUM
    {
        Width = 1;
        Height = 1;
        Format = R32F;
    };
    sampler s_PREV_LUM
    {
        Texture = PREV_LUM;
    };

    texture SSSR_BlueNoise <source="SS_BN.png";>
    {
        Width = 1024;
        Height = 1024;
        Format = RGBA8;
    };
    sampler sSSSR_BlueNoise
    {
        Texture = SSSR_BlueNoise;
        AddressU = REPEAT;
        AddressV = REPEAT;
        MipFilter = POINT;
        MinFilter = POINT;
        MagFilter = POINT;
    };

    //-------------------|
    // :: Functions  :: |
    //-------------------|

    static const float3 LumaCoeff = float3(0.2126, 0.7152, 0.0722);

    float GetLuminance(float3 color)
    {
        return dot(color, LumaCoeff);
    }

    float WN(float2 co)
    {
        return frac(sin(dot(co.xy, float2(1.0, 73))) * 437580.5453);
    }

    float IGN(float2 n)
    {
        float f = 0.06711056 * n.x + 0.00583715 * n.y;
        return frac(52.9829189 * frac(f));
    }

    float BN(float2 texcoord)
    {
        texcoord *= BUFFER_SCREEN_SIZE;
        texcoord = texcoord % 128;
        
        float frame = FRAME_COUNT % 64; 
        int2 F;
        F.x = frame % 8;
        F.y = floor(frame / 8) % 8;
        F *= 128;
        texcoord += F;
        
        texcoord /= 1024;
        return tex2D(sSSSR_BlueNoise, texcoord).r;
    }

    float CalculateSceneLuminance()
    {
        float totalLum = 0.0;
        const int samples = 64;
    
    [unroll]
        for (int x = 0; x < 8; x++)
        {
        [unroll]
            for (int y = 0; y < 8; y++)
            {
                float2 uv = float2((x + 0.5) / 8.0, (y + 0.5) / 8.0);
                float3 color = GetColor(uv);
                float lum = GetLuminance(color);
            
                totalLum += log(max(lum, 0.0001));
            }
        }
    
        return exp(totalLum / samples);
    }

    float4 LuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        float currentLum = CalculateSceneLuminance();
        float prevLum = tex2Dfetch(s_PREV_LUM, int2(0, 0)).r;
        float adaptedLum = lerp(prevLum, currentLum, AdaptationSpeed);
    
        return float4(adaptedLum, 0, 0, 1);
    }

    float4 StoreLuminancePass(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        return tex2Dfetch(sLUM, int2(0, 0));
    }

    float4 Output(float4 pos : SV_Position, float2 uv : TexCoord) : SV_Target
    {
        float3 color = GetColor(uv);
        float sceneLum = tex2Dfetch(sLUM, int2(0, 0)).r;
        float sceneMistFactor = smoothstep(MistStartThreshold, MistEndThreshold, sceneLum);
        float pixelLum = GetLuminance(color);
        float depthFactor = pow(1.0 - pixelLum, DepthInfluence);
        float totalMist = saturate(sceneMistFactor * depthFactor * MistDensity);
        
        float noise = 0.0;
        if (NoiseType > 0)
        {
            float2 noise_uv = uv * BUFFER_SCREEN_SIZE;
            if (NoiseType == 1)
            {
                noise = WN(noise_uv);
            }
            else if (NoiseType == 2)
            {
                float2 seed = uv * BUFFER_SCREEN_SIZE + (FRAME_COUNT % 256) * 5.588238;
                noise = IGN(seed);
            }
            else if (NoiseType == 3) 
            {
                noise = BN(uv);
            }
            noise = (noise - 0.5) * 2.0 * NoiseIntensity;
        }

        totalMist = saturate(totalMist + noise * totalMist);
        color = lerp(color, MistColor, totalMist);
    
        return float4(saturate(color), 1.0);
    }

    technique LumaFog
    {
        pass Luminance
        {
            VertexShader = PostProcessVS;
            PixelShader = LuminancePass;
            RenderTarget = LUM;
        }
        pass Previous
        {
            VertexShader = PostProcessVS;
            PixelShader = StoreLuminancePass;
            RenderTarget = PREV_LUM;
        }
        pass Apply
        {
            VertexShader = PostProcessVS;
            PixelShader = Output;
        }
    }
}
