/*-------------------------------------------------------------------------------------------------------|
| ::                                     Anime4K Thin HQ                                              :: |
|--------------------------------------------------------------------------------------------------------|
| Version: 1.0                                                                                           |
| Author: Ported to ReShade by Barbatos, Original by Bloc97 (Magpie port)                                |
| (https://github.com/bloc97/Anime4K)                                                                    |
| License: MIT                                                                                           |
|-------------------------------------------------------------------------------------------------------*/

#include "ReShade.fxh"

//----------------|
// :: Settings :: |
//----------------|

uniform float Strength <
    ui_label = "Strength";
    ui_type = "slider";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
    ui_tooltip = "Strength of warping for each iteration.";
> = 0.6;

uniform int Iterations <
    ui_label = "Iterations";
    ui_type = "slider";
    ui_min = 1; ui_max = 5; 
    ui_step = 1;
    ui_tooltip = "Number of iterations for the forwards solver. Higher values improve quality but cost performance.";
> = 1;

//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_NS
{
    texture Tex_Inter1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    texture Tex_Inter2
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };

    sampler sInter1
    {
        Texture = Tex_Inter1;
    };
    sampler sInter2
    {
        Texture = Tex_Inter2;
    };
    
    sampler sInter1_Linear
    {
        Texture = Tex_Inter1;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };
    sampler sBackBuffer_Linear
    {
        Texture = ReShade::BackBufferTex;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

//----------------|
// :: Functions ::|
//----------------|

    float GetLuma(float3 rgb)
    {
        return dot(float3(0.299, 0.587, 0.114), rgb);
    }

    float Gaussian(float x, float s)
    {
        float scaled = x / s;
        return exp(-0.5 * scaled * scaled);
    }

    void PS_Sobel(float4 vpos : SV_Position, float2 texcoord : TexCoord, out float2 result : SV_Target)
    {
        float3 c = tex2D(ReShade::BackBuffer, texcoord).rgb;
        float2 pt = ReShade::PixelSize;

        float nw = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(-pt.x, -pt.y)).rgb);
        float n = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(0.0, -pt.y)).rgb);
        float ne = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(pt.x, -pt.y)).rgb);
        float w = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(-pt.x, 0.0)).rgb);
        float e = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(pt.x, 0.0)).rgb);
        float sw = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(-pt.x, pt.y)).rgb);
        float s = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(0.0, pt.y)).rgb);
        float se = GetLuma(tex2D(ReShade::BackBuffer, texcoord + float2(pt.x, pt.y)).rgb);

        float xgrad = (ne + 2.0 * e + se) - (nw + 2.0 * w + sw);
        float ygrad = (sw + 2.0 * s + se) - (nw + 2.0 * n + ne);

        xgrad /= 8.0;
        ygrad /= 8.0;

        float norm = sqrt(xgrad * xgrad + ygrad * ygrad);
        
        result = float2(pow(abs(norm), 0.7), 0.0);
    }

    void PS_GaussianX(float4 vpos : SV_Position, float2 texcoord : TexCoord, out float2 result : SV_Target)
    {
        float2 inputPt = ReShade::PixelSize;
        float spatial_sigma = 2.0 * BUFFER_HEIGHT / 1080.0;
        
        int kernel_size = max(int(ceil(spatial_sigma * 2.0)), 1) * 2 + 1;
        int kernel_half = kernel_size >> 1;

        float g = 0.0;
        float gn = 0.0;

        for (int k = -kernel_half; k <= kernel_half; ++k)
        {
            float gf = Gaussian(float(k), spatial_sigma);
            float2 samplePos = texcoord + float2(k * inputPt.x, 0.0);

            g += tex2D(sInter2, samplePos).x * gf;
            gn += gf;
        }

        result = float2(g / gn, 0.0);
    }

    void PS_GaussianY(float4 vpos : SV_Position, float2 texcoord : TexCoord, out float2 result : SV_Target)
    {
        float2 inputPt = ReShade::PixelSize;
        float spatial_sigma = 2.0 * BUFFER_HEIGHT / 1080.0;
        
        int kernel_size = max(int(ceil(spatial_sigma * 2.0)), 1) * 2 + 1;
        int kernel_half = kernel_size >> 1;

        float g = 0.0;
        float gn = 0.0;

        for (int k = -kernel_half; k <= kernel_half; ++k)
        {
            float gf = Gaussian(float(k), spatial_sigma);
            float2 samplePos = texcoord + float2(0.0, k * inputPt.y);
            
            g += tex2D(sInter1, samplePos).x * gf;
            gn += gf;
        }

        result = float2(g / gn, 0.0);
    }

    void PS_GradientGen(float4 vpos : SV_Position, float2 texcoord : TexCoord, out float2 result : SV_Target)
    {
        float2 pt = ReShade::PixelSize;
        
        float nw = tex2D(sInter2, texcoord + float2(-pt.x, -pt.y)).x;
        float n = tex2D(sInter2, texcoord + float2(0.0, -pt.y)).x;
        float ne = tex2D(sInter2, texcoord + float2(pt.x, -pt.y)).x;
        float w = tex2D(sInter2, texcoord + float2(-pt.x, 0.0)).x;
        float e = tex2D(sInter2, texcoord + float2(pt.x, 0.0)).x;
        float sw = tex2D(sInter2, texcoord + float2(-pt.x, pt.y)).x;
        float s = tex2D(sInter2, texcoord + float2(0.0, pt.y)).x;
        float se = tex2D(sInter2, texcoord + float2(pt.x, pt.y)).x;
        
        float xgrad = -nw + ne - w + e - w + e - sw + se;
        float ygrad = -nw - n - n - ne + sw + s + s + se;

        result = float2(xgrad, ygrad) / 8.0;
    }

    void PS_Warp(float4 vpos : SV_Position, float2 texcoord : TexCoord, out float4 result : SV_Target)
    {
        float2 inputPt = ReShade::PixelSize;
        float relstr = (BUFFER_HEIGHT / 1080.0) * Strength;
        
        float2 pos = texcoord;

        for (int i = 0; i < Iterations; ++i)
        {
            float2 dn = tex2D(sInter1_Linear, pos).xy;
            float2 dd = (dn / (length(dn) + 0.01)) * inputPt * relstr;
            pos -= dd;
        }

        result = tex2D(sBackBuffer_Linear, pos);
    }

//----------------|
// :: Technique ::|
//----------------|

    technique Anime4K_Thin_HQ
    {
        pass Sobel
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Sobel;
            RenderTarget = Tex_Inter2;
        }
        pass GaussianX
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GaussianX;
            RenderTarget = Tex_Inter1; 
        }
        pass GaussianY
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GaussianY;
            RenderTarget = Tex_Inter2;
        }
        pass Gradient
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GradientGen;
            RenderTarget = Tex_Inter1;
        }
        pass Warp
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Warp;
        }
    }
}