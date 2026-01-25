/*----------------------------------------------|
| ::        Barbatos Neural Sharpening       :: |
|-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| 12-Channel Neural Network                     |
'----------------------------------------------*/
#include "ReShade.fxh"
#include "Barbatos_SWeights.fxh" 

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; 
    ui_max = 4.0;
    ui_step = 0.5;
    ui_label = "Intensity";
> = 2.0;

/*uniform int ViewMode <
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
    
    float RunNet(float2 uv)
    {
        const float2 pixel = ReShade::PixelSize;

        // LAYER 1: 3x3 Conv (1 -> 12)
        float4 L1_0 = BarbatosSWeights::Local_B[0];
        float4 L1_1 = BarbatosSWeights::Local_B[1];
        float4 L1_2 = BarbatosSWeights::Local_B[2];

        int w_idx = 0;
        
        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                const float val = tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
                
                L1_0 += val * BarbatosSWeights::Local_W[w_idx];
                L1_1 += val * BarbatosSWeights::Local_W[w_idx + 1];
                L1_2 += val * BarbatosSWeights::Local_W[w_idx + 2];
                
                w_idx += 3;
            }
        }
        
        // PReLU Layer 1
        const float4 prelu_a0 = BarbatosSWeights::Local_A[0];
        const float4 prelu_a1 = BarbatosSWeights::Local_A[1];
        const float4 prelu_a2 = BarbatosSWeights::Local_A[2];
        
        L1_0 = max(0, L1_0) + min(0, L1_0) * prelu_a0;
        L1_1 = max(0, L1_1) + min(0, L1_1) * prelu_a1;
        L1_2 = max(0, L1_2) + min(0, L1_2) * prelu_a2;

        // LAYER 2: 1x1 Dense (12 -> 12) 
        float4 L2_0 = BarbatosSWeights::Mix_B[0];
        float4 L2_1 = BarbatosSWeights::Mix_B[1];
        float4 L2_2 = BarbatosSWeights::Mix_B[2];
        
        int m_idx = 0;
        
        // Output Group 0
        L2_0.x += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.y += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.z += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.w += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        
        L2_0.x += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.y += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.z += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.w += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);

        L2_0.x += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.y += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.z += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_0.w += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);

        // Output Group 1
        L2_1.x += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.y += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.z += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.w += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        
        L2_1.x += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.y += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.z += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.w += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);

        L2_1.x += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.y += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.z += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_1.w += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);

        // Output Group 2
        L2_2.x += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.y += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.z += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.w += dot(L1_0, BarbatosSWeights::Mix_W[m_idx++]);
        
        L2_2.x += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.y += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.z += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.w += dot(L1_1, BarbatosSWeights::Mix_W[m_idx++]);

        L2_2.x += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.y += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.z += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);
        L2_2.w += dot(L1_2, BarbatosSWeights::Mix_W[m_idx++]);

        // PReLU Layer 2
        const float4 prelu_ma0 = BarbatosSWeights::Mix_A[0];
        const float4 prelu_ma1 = BarbatosSWeights::Mix_A[1];
        const float4 prelu_ma2 = BarbatosSWeights::Mix_A[2];
        
        L2_0 = max(0, L2_0) + min(0, L2_0) * prelu_ma0;
        L2_1 = max(0, L2_1) + min(0, L2_1) * prelu_ma1;
        L2_2 = max(0, L2_2) + min(0, L2_2) * prelu_ma2;

        // LAYER 3: Output (12 -> 1)
        float res = BarbatosSWeights::Out_Bias;
        res += dot(L2_0, BarbatosSWeights::Out_W[0]);
        res += dot(L2_1, BarbatosSWeights::Out_W[1]);
        res += dot(L2_2, BarbatosSWeights::Out_W[2]);

        return tanh(res);
    }

    void PS_GetLuma(float4 vpos : SV_Position, float2 uv : TexCoord, out float outLuma : SV_Target)
    {
        outLuma = GetLuma(tex2D(ReShade::BackBuffer, uv).rgb);
    }

    void PS_Apply(float4 vpos : SV_Position, float2 uv : TexCoord, out float4 outColor : SV_Target)
    {
        const float3 c = tex2D(ReShade::BackBuffer, uv).rgb;
        
        /* Debug Mode
        if (ViewMode == 1)
        {
            const float residual = RunNet(uv) / 5.0 * Intensity;
            outColor = float4(residual + 0.5, residual + 0.5, residual + 0.5, 1.0);
            return;
        }
        */
        const float l = GetLuma(c);
        float residual = RunNet(uv) / 5.0;
        residual = clamp(residual, -0.15, 0.15);
        const float sharp_luma = max(0, l + residual * Intensity);
        outColor = float4(c * (sharp_luma / max(l, 0.0001)), 1.0);
    }

    technique Barbatos_NS
    <
    ui_label = "Barbatos: Neural Sharpening";
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
