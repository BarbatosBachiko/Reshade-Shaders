/*----------------------------------------------|
| ::        Barbatos Neural Sharpening       :: |
|-----------------------------------------------|
| Version: 2.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| 12-Channel Neural Network Ver 2.8             |
|----------------------------------------------*/

#include ".\Includes\bb_reshade.fxh"

namespace Barbatos_ModelAB
{

#include ".\Includes\Sharpen_Neural\ModelAB_weights.fxh"

//----------|
// :: UI :: |
//----------|

uniform int ModelSelector <
    ui_type = "combo";
    ui_items = "Model A\0Model B\0";
    ui_label = "Neural Model";
    ui_tooltip = "Model A: Standard sharpening. Model B: Edge-enhanced.";
> = 0;

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Intensity";
> = 1.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Map Only\0Residual Map\0";
    ui_label = "Debug";
> = 0;

uniform float EdgeResponse <
    ui_type = "drag";
    ui_min = -1.0;
    ui_max = 1.0;
    ui_step = 0.05;
    ui_label = "Edge Response";
    ui_tooltip = "Controls how sharpening adapts to detected edges. Negative = smooth edges, Positive = enhance edges, 0 = uniform";
> = 0.0;

//----------------|
// :: Textures :: |
//----------------|

texture TexLuma { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; };
sampler sTexLuma { Texture = TexLuma; };

//----------------|
// :: Functions ::|
//----------------|

float GetLuma(float3 rgb) { return dot(rgb, float3(0.299, 0.587, 0.114)); }

float3 RGBToYCbCr(float3 rgb)
{
    float y  = dot(rgb, float3(0.299, 0.587, 0.114));
    float cb = (rgb.b - y) * 0.564 + 0.5;
    float cr = (rgb.r - y) * 0.713 + 0.5;
    return float3(y, cb, cr);
}

float3 YCbCrToRGB(float3 ycbcr)
{
    float y  = ycbcr.x;
    float cb = ycbcr.y - 0.5;
    float cr = ycbcr.z - 0.5;
    float r  = y + 1.403 * cr;
    float g  = y - 0.344 * cb - 0.714 * cr;
    float b  = y + 1.770 * cb;
    return float3(r, g, b);
}

min16float RunNet_A(float2 uv)
{
    const float2 pixel = bb::PixelSize;

    // LAYER 1: 3x3 Conv
    min16float4 L1_0 = (min16float4)ModelA::Local_B[0];
    min16float4 L1_1 = (min16float4)ModelA::Local_B[1];
    min16float4 L1_2 = (min16float4)ModelA::Local_B[2];

    int w_idx = 0;
    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            const min16float val = (min16float)tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
            L1_0 += val * (min16float4)ModelA::Local_W[w_idx + 0];
            L1_1 += val * (min16float4)ModelA::Local_W[w_idx + 1];
            L1_2 += val * (min16float4)ModelA::Local_W[w_idx + 2];
            w_idx += 3;
        }
    }

    // PReLU 1
    L1_0 = max(0.0, L1_0) + min(0.0, L1_0) * (min16float4)ModelA::Local_A[0];
    L1_1 = max(0.0, L1_1) + min(0.0, L1_1) * (min16float4)ModelA::Local_A[1];
    L1_2 = max(0.0, L1_2) + min(0.0, L1_2) * (min16float4)ModelA::Local_A[2];

    // LAYER 2: 1x1 Dense
    min16float4 L2_0 = (min16float4)ModelA::Mix_B[0];
    min16float4 L2_1 = (min16float4)ModelA::Mix_B[1];
    min16float4 L2_2 = (min16float4)ModelA::Mix_B[2];

    int m_idx = 0;
    // Group 0
    L2_0.x += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.y += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.z += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.w += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.x += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.y += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.z += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.w += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.x += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.y += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.z += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_0.w += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    // Group 1
    L2_1.x += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.y += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.z += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.w += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.x += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.y += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.z += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.w += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.x += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.y += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.z += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_1.w += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    // Group 2
    L2_2.x += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.y += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.z += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.w += dot(L1_0, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.x += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.y += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.z += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.w += dot(L1_1, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.x += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.y += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.z += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);
    L2_2.w += dot(L1_2, (min16float4)ModelA::Mix_W[m_idx++]);

    // PReLU 2
    L2_0 = max(0.0, L2_0) + min(0.0, L2_0) * (min16float4)ModelA::Mix_A[0];
    L2_1 = max(0.0, L2_1) + min(0.0, L2_1) * (min16float4)ModelA::Mix_A[1];
    L2_2 = max(0.0, L2_2) + min(0.0, L2_2) * (min16float4)ModelA::Mix_A[2];

    // LAYER 3: Output
    min16float res = (min16float)ModelA::Out_Bias;
    res += dot(L2_0, (min16float4)ModelA::Out_W[0]);
    res += dot(L2_1, (min16float4)ModelA::Out_W[1]);
    res += dot(L2_2, (min16float4)ModelA::Out_W[2]);

    return res;
}

min16float RunNet_B(float2 uv)
{
    const float2 pixel = bb::PixelSize;

    // LAYER 1: 3x3 Conv
    min16float4 L1_0 = (min16float4)ModelB::Local_B[0];
    min16float4 L1_1 = (min16float4)ModelB::Local_B[1];
    min16float4 L1_2 = (min16float4)ModelB::Local_B[2];

    int w_idx = 0;
    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            const min16float val = (min16float)tex2D(sTexLuma, uv + float2(x, y) * pixel).r;
            L1_0 += val * (min16float4)ModelB::Local_W[w_idx + 0];
            L1_1 += val * (min16float4)ModelB::Local_W[w_idx + 1];
            L1_2 += val * (min16float4)ModelB::Local_W[w_idx + 2];
            w_idx += 3;
        }
    }

    // PReLU 1
    L1_0 = max(0.0, L1_0) + min(0.0, L1_0) * (min16float4)ModelB::Local_A[0];
    L1_1 = max(0.0, L1_1) + min(0.0, L1_1) * (min16float4)ModelB::Local_A[1];
    L1_2 = max(0.0, L1_2) + min(0.0, L1_2) * (min16float4)ModelB::Local_A[2];

    // LAYER 2: 1x1 Dense
    min16float4 L2_0 = (min16float4)ModelB::Mix_B[0];
    min16float4 L2_1 = (min16float4)ModelB::Mix_B[1];
    min16float4 L2_2 = (min16float4)ModelB::Mix_B[2];

    int m_idx = 0;
    // Group 0
    L2_0.x += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.y += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.z += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.w += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.x += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.y += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.z += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.w += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.x += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.y += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.z += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_0.w += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    // Group 1
    L2_1.x += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.y += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.z += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.w += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.x += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.y += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.z += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.w += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.x += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.y += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.z += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_1.w += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    // Group 2
    L2_2.x += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.y += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.z += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.w += dot(L1_0, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.x += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.y += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.z += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.w += dot(L1_1, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.x += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.y += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.z += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);
    L2_2.w += dot(L1_2, (min16float4)ModelB::Mix_W[m_idx++]);

    // PReLU 2
    L2_0 = max(0.0, L2_0) + min(0.0, L2_0) * (min16float4)ModelB::Mix_A[0];
    L2_1 = max(0.0, L2_1) + min(0.0, L2_1) * (min16float4)ModelB::Mix_A[1];
    L2_2 = max(0.0, L2_2) + min(0.0, L2_2) * (min16float4)ModelB::Mix_A[2];

    // LAYER 3: Output
    min16float res = (min16float)ModelB::Out_Bias;
    res += dot(L2_0, (min16float4)ModelB::Out_W[0]);
    res += dot(L2_1, (min16float4)ModelB::Out_W[1]);
    res += dot(L2_2, (min16float4)ModelB::Out_W[2]);

    return res;
}


min16float RunNet(float2 uv)
{
    if (ModelSelector == 0)
    {
        return RunNet_A(uv);
    }
    else
    {
        return RunNet_B(uv);
    }
}

void PS_GetLuma(float4 vpos : SV_Position, float2 uv : TexCoord, out min16float outLuma : SV_Target)
{
    outLuma = (min16float)GetLuma(tex2D(bb::BackBuffer, uv).rgb);
}

void PS_Apply(float4 vpos : SV_Position, float2 uv : TexCoord, out float4 outColor : SV_Target)
{
    const float3 c = tex2D(bb::BackBuffer, uv).rgb;

    min16float residual = RunNet(uv);
    min16float rawResidual = residual;
    float residualMagnitude = abs((float)rawResidual);
    float normalizedResidual = saturate(residualMagnitude * 2.63); // 2.63 ~ 1/0.38

    // Debug Mode
    if (ViewMode == 2)
    {
        outColor = float4(normalizedResidual, normalizedResidual, normalizedResidual, 1.0);
        return;
    }

    residual = clamp(rawResidual, -0.5, 0.5);
    residual *= 0.22;

    float3 ycbcr = RGBToYCbCr(c);
    float adaptationMask;
    if (EdgeResponse < 0.0) {
        adaptationMask = lerp(1.0, 1.0 - normalizedResidual, abs(EdgeResponse));
    } else if (EdgeResponse > 0.0) {
        adaptationMask = lerp(1.0, 1.0 + normalizedResidual, EdgeResponse);
    } else {
        adaptationMask = 1.0;
    }

    if (abs(EdgeResponse) > 0.001)
    {
        residual *= (min16float)adaptationMask;
    }

    // Debug Mode
    if (ViewMode == 1)
    {
        float debugRes = residual * Intensity;
        outColor = float4(0.5 + debugRes, 0.5 + debugRes, 0.5 + debugRes, 1.0);
        return;
    }

    {
        const float2 pixel = bb::PixelSize;
        float lumaC = ycbcr.x;

        // Cross-neighborhood sampling
        float lumaN = tex2D(sTexLuma, uv + float2( 0.0,      -pixel.y)).r;
        float lumaS = tex2D(sTexLuma, uv + float2( 0.0,       pixel.y)).r;
        float lumaE = tex2D(sTexLuma, uv + float2( pixel.x,   0.0    )).r;
        float lumaW = tex2D(sTexLuma, uv + float2(-pixel.x,   0.0    )).r;

        float maxNeighbors = max(max(lumaN, lumaS), max(lumaE, lumaW));
        float minNeighbors = min(min(lumaN, lumaS), min(lumaE, lumaW));

        float localContrast  = maxNeighbors - minNeighbors;
        float aliasingSignal = saturate((localContrast - 0.25) * 4.0);

        float brightSpike     = max(0.0, lumaC - maxNeighbors);
        float darkSpike       = max(0.0, minNeighbors - lumaC);
        float spikeIntensity  = max(brightSpike, darkSpike);
        float fireflySignal   = saturate((spikeIntensity - 0.1) * 6.0);

        float artifactSignal  = saturate(aliasingSignal + fireflySignal);
        float suppressionMask = 1.0 - artifactSignal;
        residual *= (min16float)suppressionMask;
    }

    float sharpenedLuma = max(0.0, ycbcr.x + (residual * Intensity));
    ycbcr.x = sharpenedLuma;
    float3 finalColor = max(0.0, YCbCrToRGB(ycbcr));

    outColor = float4(finalColor, 1.0);
}

//-----------------|
// :: Techniques ::|
//-----------------|

technique BaBa_NeuralSharpen_28
<
    ui_label = "BaBa: Neural Sharpen 2.8 Preview";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_GetLuma;
        RenderTarget = TexLuma;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_Apply;
    }
}

}
