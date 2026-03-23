/*----------------------------------------------|
| ::        Barbatos Neural Sharpening       :: |
|-----------------------------------------------|
| Version: 1.3                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| 12-Channel Neural Network                     |
|----------------------------------------------*/

#include "ReShade.fxh"
#include ".\BaBa_Includes\BaBa_Model_A.fxh" 
#include ".\BaBa_Includes\BaBa_Model_B.fxh" 

//----------|
// :: UI :: |
//----------|
uniform int ModelType <
    ui_type = "combo";
    ui_items = "Model A2\0Model B2\0";
    ui_label = "Model Type";
    ui_tooltip = "Model A: Model trained to deliver controlled sharpness. \nModel B: Model trained to deliver raw sharpness.";
> = 0;

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; 
    ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Intensity";
> = 1.0;

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
namespace Barbatos_NS120
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
        float4 L1_0 = ModelA::Local_B[0];
        float4 L1_1 = ModelA::Local_B[1];
        float4 L1_2 = ModelA::Local_B[2];

        int w_idx = 0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                const float val = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                L1_0 += val * ModelA::Local_W[w_idx];
                L1_1 += val * ModelA::Local_W[w_idx + 1];
                L1_2 += val * ModelA::Local_W[w_idx + 2];
                w_idx += 3;
            }
        }
        
        // PReLU 1
        L1_0 = max(0, L1_0) + min(0, L1_0) * ModelA::Local_A[0];
        L1_1 = max(0, L1_1) + min(0, L1_1) * ModelA::Local_A[1];
        L1_2 = max(0, L1_2) + min(0, L1_2) * ModelA::Local_A[2];

        // LAYER 2: 1x1 Dense
        float4 L2_0 = ModelA::Mix_B[0];
        float4 L2_1 = ModelA::Mix_B[1];
        float4 L2_2 = ModelA::Mix_B[2];
        
        int m_idx = 0;
        // Group 0
        L2_0.x += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_0.x += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_0.x += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_0.y += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_0.z += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_0.w += dot(L1_2, ModelA::Mix_W[m_idx++]);
        
        // Group 1
        L2_1.x += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_1.x += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_1.x += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_1.y += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_1.z += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_1.w += dot(L1_2, ModelA::Mix_W[m_idx++]);

        // Group 2
        L2_2.x += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_0, ModelA::Mix_W[m_idx++]);
        L2_2.x += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_1, ModelA::Mix_W[m_idx++]);
        L2_2.x += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_2.y += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_2.z += dot(L1_2, ModelA::Mix_W[m_idx++]);
        L2_2.w += dot(L1_2, ModelA::Mix_W[m_idx++]);

        // PReLU 2
        L2_0 = max(0, L2_0) + min(0, L2_0) * ModelA::Mix_A[0];
        L2_1 = max(0, L2_1) + min(0, L2_1) * ModelA::Mix_A[1];
        L2_2 = max(0, L2_2) + min(0, L2_2) * ModelA::Mix_A[2];

        // LAYER 3: Output
        float res = ModelA::Out_Bias;
        res += dot(L2_0, ModelA::Out_W[0]);
        res += dot(L2_1, ModelA::Out_W[1]);
        res += dot(L2_2, ModelA::Out_W[2]);

        return tanh(res);
    }

    float RunNet_B(float2 uv)
    {
        const float2 pixel = ReShade::PixelSize;
        // LAYER 1
        float4 L1_0 = ModelB::Local_B[0];
        float4 L1_1 = ModelB::Local_B[1];
        float4 L1_2 = ModelB::Local_B[2];

        int w_idx = 0;
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                const float val = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                L1_0 += val * ModelB::Local_W[w_idx];
                L1_1 += val * ModelB::Local_W[w_idx + 1];
                L1_2 += val * ModelB::Local_W[w_idx + 2];
                w_idx += 3;
            }
        }
        
        // PReLU 1
        L1_0 = max(0, L1_0) + min(0, L1_0) * ModelB::Local_A[0];
        L1_1 = max(0, L1_1) + min(0, L1_1) * ModelB::Local_A[1];
        L1_2 = max(0, L1_2) + min(0, L1_2) * ModelB::Local_A[2];

        // LAYER 2
        float4 L2_0 = ModelB::Mix_B[0];
        float4 L2_1 = ModelB::Mix_B[1];
        float4 L2_2 = ModelB::Mix_B[2];
        
        int m_idx = 0;
        // Group 0
        L2_0.x += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_0.x += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_0.x += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_0.y += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_0.z += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_0.w += dot(L1_2, ModelB::Mix_W[m_idx++]);
        
        // Group 1
        L2_1.x += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_1.x += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_1.x += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_1.y += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_1.z += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_1.w += dot(L1_2, ModelB::Mix_W[m_idx++]);

        // Group 2
        L2_2.x += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_0, ModelB::Mix_W[m_idx++]);
        L2_2.x += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_1, ModelB::Mix_W[m_idx++]);
        L2_2.x += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_2.y += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_2.z += dot(L1_2, ModelB::Mix_W[m_idx++]);
        L2_2.w += dot(L1_2, ModelB::Mix_W[m_idx++]);

        // PReLU 2
        L2_0 = max(0, L2_0) + min(0, L2_0) * ModelB::Mix_A[0];
        L2_1 = max(0, L2_1) + min(0, L2_1) * ModelB::Mix_A[1];
        L2_2 = max(0, L2_2) + min(0, L2_2) * ModelB::Mix_A[2];

        // LAYER 3
        float res = ModelB::Out_Bias;
        res += dot(L2_0, ModelB::Out_W[0]);
        res += dot(L2_1, ModelB::Out_W[1]);
        res += dot(L2_2, ModelB::Out_W[2]);

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

        if (ModelType == 0) // Model A
        {
            residual = RunNet_A(uv);
        }
        else // Model B
        {
            residual = RunNet_B(uv);
        }

        /* Debug Mode
        if (ViewMode == 1)
        {
            float debugRes = residual * Intensity;
            outColor = float4(0.5 + debugRes, 0.5 + debugRes, 0.5 + debugRes, 1.0);
            return;
        }
        */

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

    technique BaBa_NeuralSharpen
    <
        ui_label = "BaBa: Neural Sharpen";
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