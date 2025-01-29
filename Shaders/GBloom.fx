//+++++++++++++++++++++++++++++++++++++++++++++++++
// GBloom
//+++++++++++++++++++++++++++++++++++++++++++++++++
// Author: Barbatos Bachiko
// Version: 1.0
// License: MIT
//+++++++++++++++++++++++++++++++++++++++++++++++++

namespace oaugedadepressaoeansidade
{
    #include "ReShade.fxh"

    // Settings
    uniform float BloomIntensity <
        ui_type = "slider";
        ui_label = "Bloom Intensity";
        ui_tooltip = "Strength of the bloom effect";
        ui_min = 0.0;
        ui_max = 5.0;
        ui_step = 0.01;
        ui_default = 1.5;
    > = 1.5;

    uniform float LuminanceThreshold <
        ui_type = "slider";
        ui_label = "Luminance Threshold";
        ui_tooltip = "Minimum brightness to generate bloom";
        ui_min = 0.0;
        ui_max = 2.0;
        ui_step = 0.01;
        ui_default = 0.75;
    > = 0.75;

    uniform float BlurRadius <
        ui_type = "slider";
        ui_label = "Blur Radius";
        ui_tooltip = "Size of the blur effect";
        ui_min = 0.5;
        ui_max = 10.0;
        ui_step = 0.1;
        ui_default = 3.0;
    > = 3.0;

    // Textures
    texture2D BloomTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    texture2D TempTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };

    sampler2D sBloom
    {
        Texture = BloomTex;
    };
    sampler2D sTemp
    {
        Texture = TempTex;
    };

    // Convolution Kernel
    static const float weight[5] =
    {
        0.2270270270,
        0.1945945946,
        0.1216216216,
        0.0540540541,
        0.0162162162
    };

    // Bright Pass Extraction
    float4 BrightPass(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
        float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
        float bright = smoothstep(LuminanceThreshold, LuminanceThreshold + 0.2, luminance);
        return float4(color * bright, 1.0);
    }

    // Horizontal Gaussian Blur
    float4 BlurHorizontal(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 texelSize = BlurRadius / resolution;

        float3 result = tex2D(sBloom, uv).rgb * weight[0];
        
        [unroll]
        for (int i = 1; i < 5; ++i)
        {
            result += tex2D(sBloom, uv + float2(texelSize.x * i, 0.0)).rgb * weight[i];
            result += tex2D(sBloom, uv - float2(texelSize.x * i, 0.0)).rgb * weight[i];
        }
        
        return float4(result, 1.0);
    }

    // Vertical Gaussian Blur
    float4 BlurVertical(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 texelSize = BlurRadius / resolution;

        float3 result = tex2D(sTemp, uv).rgb * weight[0];
        
        [unroll]
        for (int i = 1; i < 5; ++i)
        {
            result += tex2D(sTemp, uv + float2(0.0, texelSize.y * i)).rgb * weight[i];
            result += tex2D(sTemp, uv - float2(0.0, texelSize.y * i)).rgb * weight[i];
        }
        
        return float4(result, 1.0);
    }

    // Final Composition
    float4 Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 scene = tex2D(ReShade::BackBuffer, uv).rgb;
        float3 bloom = tex2D(sBloom, uv).rgb;
        float3 result = scene + bloom * BloomIntensity;
        return float4(result, 1.0);
    }

    // Fullscreen Vertex Shader
    void VS(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD)
    {
        uv = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
        pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    }

    // Techniques
    technique GBloom
    {
        pass BrightPass
        {
            VertexShader = VS;
            PixelShader = BrightPass;
            RenderTarget = BloomTex;
        }
        
        pass HorizontalBlur
        {
            VertexShader = VS;
            PixelShader = BlurHorizontal;
            RenderTarget = TempTex;
        }
        
        pass VerticalBlur
        {
            VertexShader = VS;
            PixelShader = BlurVertical;
            RenderTarget = BloomTex;
        }
        
        pass Composite
        {
            VertexShader = VS;
            PixelShader = Composite;
        }
    }
}