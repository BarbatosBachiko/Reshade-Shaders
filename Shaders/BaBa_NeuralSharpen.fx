/*----------------------------------------------|
| ::        Barbatos Neural Sharpening       :: |
|-----------------------------------------------|
| Version: 1.2                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| 12-Channel Neural Network                     |
|----------------------------------------------*/

#include "ReShade.fxh"
#include "Barbatos_SWeights.fxh" 

//----------|
// :: UI :: |
//----------|

uniform int ModelType <
    ui_type = "combo";
    ui_items = "Model A\0Model B\0";
    ui_label = "Model Type";
    ui_tooltip = "Model A: Model trained to deliver controlled sharpness. \nModel B: Model trained to deliver raw sharpness.";
> = 0;

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; 
    ui_max = 4.0;
    ui_step = 0.1;
    ui_label = "Intensity";
> = 2.0;

uniform float AntiHalo <
    ui_type = "drag";
    ui_min = 0.0; 
    ui_max = 1.0;
    ui_step = 0.05;
    ui_label = "Anti-Halo";
    ui_tooltip = "Reduces ringing artifacts.";
> = 0.0;

/* // Debug Mode 
uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Map Only\0";
    ui_label = "Debug";
> = 0;
*/

//----------------|
// :: Textures :: |
//----------------|
namespace Barbatos_NS
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
    };

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
    
    float RunNet_A(float2 uv)
    {
        const float2 pixel = ReShade::PixelSize;
        
        // LAYER 1: 3x3 Conv
        float4 L1_0 = BarbatosSWeights::ModelA::Local_B[0];
        float4 L1_1 = BarbatosSWeights::ModelA::Local_B[1];
        float4 L1_2 = BarbatosSWeights::ModelA::Local_B[2];

        int w_idx = 0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                const float val = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                L1_0 += val * BarbatosSWeights::ModelA::Local_W[w_idx];
                L1_1 += val * BarbatosSWeights::ModelA::Local_W[w_idx + 1];
                L1_2 += val * BarbatosSWeights::ModelA::Local_W[w_idx + 2];
                w_idx += 3;
            }
        }
        
        // PReLU 1
        L1_0 = max(0, L1_0) + min(0, L1_0) * BarbatosSWeights::ModelA::Local_A[0];
        L1_1 = max(0, L1_1) + min(0, L1_1) * BarbatosSWeights::ModelA::Local_A[1];
        L1_2 = max(0, L1_2) + min(0, L1_2) * BarbatosSWeights::ModelA::Local_A[2];
        
        // LAYER 2: 1x1 Dense
        float4 L2_0 = BarbatosSWeights::ModelA::Mix_B[0];
        float4 L2_1 = BarbatosSWeights::ModelA::Mix_B[1];
        float4 L2_2 = BarbatosSWeights::ModelA::Mix_B[2];
        
        int m_idx = 0;
        // Group 0
        L2_0.x += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.x += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.x += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        
        // Group 1
        L2_1.x += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.x += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.x += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);

        // Group 2
        L2_2.x += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_0, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.x += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_1, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.x += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_2, BarbatosSWeights::ModelA::Mix_W[m_idx++]);

        // PReLU 2
        L2_0 = max(0, L2_0) + min(0, L2_0) * BarbatosSWeights::ModelA::Mix_A[0];
        L2_1 = max(0, L2_1) + min(0, L2_1) * BarbatosSWeights::ModelA::Mix_A[1];
        L2_2 = max(0, L2_2) + min(0, L2_2) * BarbatosSWeights::ModelA::Mix_A[2];
        
        // LAYER 3: Output
        float res = BarbatosSWeights::ModelA::Out_Bias;
        res += dot(L2_0, BarbatosSWeights::ModelA::Out_W[0]);
        res += dot(L2_1, BarbatosSWeights::ModelA::Out_W[1]);
        res += dot(L2_2, BarbatosSWeights::ModelA::Out_W[2]);

        return tanh(res);
    }

    float RunNet_B(float2 uv)
    {
        const float2 pixel = ReShade::PixelSize;
        
        // LAYER 1
        float4 L1_0 = BarbatosSWeights::ModelB::Local_B[0];
        float4 L1_1 = BarbatosSWeights::ModelB::Local_B[1];
        float4 L1_2 = BarbatosSWeights::ModelB::Local_B[2];

        int w_idx = 0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                const float val = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                L1_0 += val * BarbatosSWeights::ModelB::Local_W[w_idx];
                L1_1 += val * BarbatosSWeights::ModelB::Local_W[w_idx + 1];
                L1_2 += val * BarbatosSWeights::ModelB::Local_W[w_idx + 2];
                w_idx += 3;
            }
        }
        
        // PReLU 1
        L1_0 = max(0, L1_0) + min(0, L1_0) * BarbatosSWeights::ModelB::Local_A[0];
        L1_1 = max(0, L1_1) + min(0, L1_1) * BarbatosSWeights::ModelB::Local_A[1];
        L1_2 = max(0, L1_2) + min(0, L1_2) * BarbatosSWeights::ModelB::Local_A[2];
        
        // LAYER 2
        float4 L2_0 = BarbatosSWeights::ModelB::Mix_B[0];
        float4 L2_1 = BarbatosSWeights::ModelB::Mix_B[1];
        float4 L2_2 = BarbatosSWeights::ModelB::Mix_B[2];
        
        int m_idx = 0;
        // Group 0
        L2_0.x += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.x += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.x += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        
        // Group 1
        L2_1.x += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.x += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.x += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);

        // Group 2
        L2_2.x += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_0, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.x += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_1, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.x += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_2, BarbatosSWeights::ModelB::Mix_W[m_idx++]);

        // PReLU 2
        L2_0 = max(0, L2_0) + min(0, L2_0) * BarbatosSWeights::ModelB::Mix_A[0];
        L2_1 = max(0, L2_1) + min(0, L2_1) * BarbatosSWeights::ModelB::Mix_A[1];
        L2_2 = max(0, L2_2) + min(0, L2_2) * BarbatosSWeights::ModelB::Mix_A[2];
        
        // LAYER 3
        float res = BarbatosSWeights::ModelB::Out_Bias;
        res += dot(L2_0, BarbatosSWeights::ModelB::Out_W[0]);
        res += dot(L2_1, BarbatosSWeights::ModelB::Out_W[1]);
        res += dot(L2_2, BarbatosSWeights::ModelB::Out_W[2]);

        return tanh(res);
    }

    void PS_GetLuma(float4 vpos : SV_Position, float2 uv : TexCoord, out float outLuma : SV_Target)
    {
        outLuma = GetLuma(tex2D(ReShade::BackBuffer, uv).rgb);
    }

    void PS_Apply(float4 vpos : SV_Position, float2 uv : TexCoord, out float4 outColor : SV_Target)
    {
        const float3 c = tex2D(ReShade::BackBuffer, uv).rgb;
        float residual = 0.0;
        float normFactor = 5.0;

        if (ModelType == 0) // Model A
        {
            residual = RunNet_A(uv);
            normFactor = 5.0;
        }
        else // Model B
        {
            residual = RunNet_B(uv);
            normFactor = 20.0;
        }

        /* Debug Mode
        if (ViewMode == 1)
        {
            float debugRes = residual * Intensity;
            outColor = float4(0.5 + debugRes, 0.5 + debugRes, 0.5 + debugRes, 1.0);
            return;
        }
        */

        residual = residual / normFactor;
        residual = clamp(residual, -0.15, 0.15);

        float3 ycbcr = RGBToYCbCr(c);
        float sharpenedLuma = max(0.0, ycbcr.x + (residual * Intensity));
        
        // Anti-Halo Local Min/Max Clamping
        if (AntiHalo > 0.0)
        {
            const float2 pixel = ReShade::PixelSize;
            float minLuma = 1.0;
            float maxLuma = 0.0;
            
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                [unroll]
                for (int x = -1; x <= 1; x++)
                {
                    float lumaTap = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                    minLuma = min(minLuma, lumaTap);
                    maxLuma = max(maxLuma, lumaTap);
                }
            }
            
            float clampedLuma = clamp(sharpenedLuma, minLuma, maxLuma);
            sharpenedLuma = lerp(sharpenedLuma, clampedLuma, AntiHalo);
        }
        
        ycbcr.x = sharpenedLuma;
        float3 finalColor = max(0.0, YCbCrToRGB(ycbcr));

        outColor = float4(finalColor, 1.0);
    }

    technique Barbatos_NS
    <
    ui_label = "Barbatos: Neural Sharpen";
    >
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GetLuma;
            RenderTarget = TexLuma;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Apply;
        }
    }
}
