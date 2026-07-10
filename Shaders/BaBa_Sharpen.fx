/*----------------------------------------------|
| | ::           Barbatos Sharpen            :: |
| |---------------------------------------------|
| Version: 1.0.0                                |
| Author: Barbatos                              |
| License: MIT                                  |
|                                               |
| Residual multi-scale unsharp sharpening       |
'----------------------------------------------*/

#include ".\bb_include\bb_reshade.fxh"
#define USE_HALF 1
#include ".\bb_include\bb_common.fxh"
#include ".\bb_include\bb_colorspace.fxh"
#include ".\bb_include\bb_vertex.fxh"

//----------|
// :: UI :: |
//----------|

// Sharpening
uniform float Intensity <
    ui_category = "Sharpening";
    ui_label = "Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 1.0;

uniform float FineStrength <
    ui_category = "Sharpening";
    ui_label = "Fine Strength";
    ui_tooltip = "Weight of the fine 3x3 Gaussian high-pass detail layer.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 1.0;

uniform float CoarseStrength <
    ui_category = "Sharpening";
    ui_label = "Coarse Strength";
    ui_tooltip = "Weight of the coarse 5x5 Gaussian high-pass structural detail layer.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
> = 0.5;

uniform float EdgeResponse <
    ui_category = "Sharpening";
    ui_label = "Edge Response";
    ui_tooltip = "Negative = smooth edges, Positive = enhance edges, 0 = uniform.";
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0; ui_step = 0.05;
> = 0.0;

// System / Debug
uniform float ResidualScale <
    ui_category = "System / Debug";
    ui_category_closed = true;
    ui_label = "Residual Scale";
    ui_tooltip = "Matches Neural Sharpen SHADER_RESIDUAL_SCALE (default 0.44).";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.44;

uniform bool ShadowRecovery <
    ui_category = "System / Debug";
    ui_label = "Shadow Recovery";
    ui_tooltip = "Boost residual in darker regions (matches Neural Sharpen shadow recovery).";
> = true;

uniform int ViewMode <
    ui_category = "System / Debug";
    ui_label = "Debug View";
    ui_type = "combo";
    ui_items = "Normal\0Map Only\0Residual Map\0";
> = 0;

namespace Barbatos_Sharpen
{
    texture TexLuma
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sTexLuma
    {
        Texture = TexLuma;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    // R = fine detail (luma - 3x3 Gaussian), G = coarse detail (luma - 5x5 Gaussian)
    texture TexDetail
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler sTexDetail
    {
        Texture = TexDetail;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    //---------------------|
    // :: Vertex Shaders ::|
    //---------------------|

    void VS_Barbatos_Sharpen(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_FullScreen(id, outStruct, 60.0);
    }

    //-----------------|
    // :: Functions :: |
    //-----------------|

    float GetLuma(float3 rgb)
    {
        return dot(rgb, float3(0.299, 0.587, 0.114));
    }

    float3 RGBToYCbCr(float3 rgb)
    {
        float y = dot(rgb, float3(0.299, 0.587, 0.114));
        float cb = (rgb.b - y) * 0.564 + 0.5;
        float cr = (rgb.r - y) * 0.713 + 0.5;
        return float3(y, cb, cr);
    }

    float3 YCbCrToRGB(float3 ycbcr)
    {
        float y = ycbcr.x;
        float cb = ycbcr.y - 0.5;
        float cr = ycbcr.z - 0.5;
        float r = y + 1.403 * cr;
        float g = y - 0.344 * cb - 0.714 * cr;
        float b = y + 1.770 * cb;
        return float3(r, g, b);
    }

    float ComputeShadowRecoveryGain(float luma)
    {
        float gain = 0.10 / (luma + 0.035);
        return clamp(gain, 1.0, 2.75);
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

    void PS_GetLuma(VS_OUTPUT input, out min16float outLuma : SV_Target)
    {
        outLuma = (min16float)GetLuma(GetColor(input.uv).rgb);
    }

    // Fine:  luma - separable 3x3 Gaussian {1,2,1}/4
    // Coarse: luma - separable 5x5 Gaussian {1,4,6,4,1}/16
    void PS_GetDetail(VS_OUTPUT input, out float2 outDetail : SV_Target)
    {
        const float2 pixel = bb::PixelSize;
        const float lumaC = tex2D(sTexLuma, input.uv).r;

        const float3 w3 = float3(0.25, 0.5, 0.25);
        float baseFine = 0.0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                float s = tex2D(sTexLuma, input.uv + float2(x, y) * pixel).r;
                baseFine += s * w3[x + 1] * w3[y + 1];
            }
        }

        // Separable binomial weights {1,4,6,4,1}/16
        static const float w5[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
        float baseCoarse = 0.0;
        [unroll]
        for (int y5 = -2; y5 <= 2; y5++)
        {
            [unroll]
            for (int x5 = -2; x5 <= 2; x5++)
            {
                float s = tex2D(sTexLuma, input.uv + float2(x5, y5) * pixel).r;
                baseCoarse += s * w5[x5 + 2] * w5[y5 + 2];
            }
        }

        outDetail = float2(lumaC - baseFine, lumaC - baseCoarse);
    }

    void PS_Apply(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        const float3 c = GetColor(input.uv).rgb;
        const float lumaC = tex2D(sTexLuma, input.uv).r;
        const float2 detail = tex2D(sTexDetail, input.uv).rg;

        // Analytic multi-scale residual
        float residual = FineStrength * detail.r + CoarseStrength * detail.g;
        float rawResidual = residual;

        float residualMagnitude = abs(rawResidual);
        float normalizedResidual = pow(saturate(residualMagnitude * 2.5), 0.65);

        // Debug Mode: Residual Map
        if (ViewMode == 2)
        {
            outColor = float4(normalizedResidual, normalizedResidual, normalizedResidual, 1.0);
            return;
        }

        residual = clamp(rawResidual, -0.5, 0.5);
        residual *= ResidualScale;

        float shadowGain = ShadowRecovery ? ComputeShadowRecoveryGain(lumaC) : 1.0;
        residual *= shadowGain;

        // Edge-responsive adaptation
        float adaptationMask;
        if (EdgeResponse < 0.0)
            adaptationMask = lerp(1.0, 1.0 - normalizedResidual, abs(EdgeResponse));
        else if (EdgeResponse > 0.0)
            adaptationMask = lerp(1.0, 1.0 + normalizedResidual, EdgeResponse);
        else
            adaptationMask = 1.0;

        if (abs(EdgeResponse) > 0.001)
            residual *= adaptationMask;

        // Debug Mode: Map Only
        if (ViewMode == 1)
        {
            float debugRes = residual * Intensity;
            outColor = float4(0.5 + debugRes, 0.5 + debugRes, 0.5 + debugRes, 1.0);
            return;
        }

        float sharpenedLuma = max(0.0, lumaC + (residual * Intensity));

        float3 ycbcr = RGBToYCbCr(c);
        ycbcr.x = sharpenedLuma;
        float3 finalColor = max(0.0, YCbCrToRGB(ycbcr));

        outColor = float4(finalColor, 1.0);
    }

    technique BaBa_Sharpen
    <
        ui_label = "BaBa: Sharpen";
        ui_tooltip = "Multi-scale residual sharpening (fine 3x3 + coarse 5x5).";
    >
    {
        pass GetLuma
        {
            VertexShader = VS_Barbatos_Sharpen;
            PixelShader = PS_GetLuma;
            RenderTarget = TexLuma;
        }
        pass GetDetail
        {
            VertexShader = VS_Barbatos_Sharpen;
            PixelShader = PS_GetDetail;
            RenderTarget = TexDetail;
        }
        pass Apply
        {
            VertexShader = VS_Barbatos_Sharpen;
            PixelShader = PS_Apply;
        }
    }
}
