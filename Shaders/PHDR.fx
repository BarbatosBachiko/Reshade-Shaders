/*-----------------------------------------------------|
| ::                 Perceptual HDR                 :: |
'------------------------------------------------------|
|  Perceptual HDR                                      |
|  Version: 1.1                                        |
|  Based on: https://github.com/ray075hl/singleLDR2HDR |
|  using (WLS + SRS + VIG + ToneMap)                   |
|  License: MIT                                        |
|  About: An attempt to restore some brightness and    |
|  shadows to LDR monitors. It's not true HDR.         | 
'-----------------------------------------------------*/

#include "ReShade.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Strength <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "INTENSITY";
> = 1.0;

uniform float Radius <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 30.0;
    ui_label = "Smoothing Radius";
    ui_tooltip = "Simulates the Lambda/Alpha of WLS.";
> = 12.5;

uniform float Epsilon <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.005;
    ui_label = "Edge Sensitivity";
> = 0.001;

uniform bool EnableAdaptation <
    ui_label = "Enable Eye Adaptation";
    ui_tooltip = "Enables dynamic brightness adaptation. Disable to use a fixed exposure value (prevents washing out dark scenes).";
> = true;

uniform float AdaptationTime <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "Eye Adaptation Speed";
    ui_tooltip = "Time in seconds for the eye to adjust to brightness changes. Higher = Smoother/Slower.";
> = 1.0;

uniform float ManualExposure <
    ui_type = "slider";
    ui_min = 0.001; ui_max = 1.0;
    ui_label = "Manual Exposure";
    ui_tooltip = "Fixed exposure value used when Eye Adaptation is disabled. Lower values preserve darkness.";
> = 0.1; 

uniform float FrameTime < source = "frametime"; >;

//----------------|
// :: Textures :: |
//----------------|

#define SCALE 2
#define GW (BUFFER_WIDTH / SCALE)
#define GH (BUFFER_HEIGHT / SCALE)
namespace BaBaPHDR
{
texture TexColor : COLOR;
sampler sTexColor
{
    Texture = TexColor;
};

texture TexLuma
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
    MipLevels = 8;
};
sampler sTexLuma
{
    Texture = TexLuma;
};

texture TexTempMeans
{
    Width = GW;
    Height = GH;
    Format = RG16F;
};
sampler sTexTempMeans
{
    Texture = TexTempMeans;
};

texture TexStats
{
    Width = GW;
    Height = GH;
    Format = RG16F;
};
sampler sTexStats
{
    Texture = TexStats;
};

texture TexVarI
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler sTexVarI
{
    Texture = TexVarI;
};

texture TexAdapt
{
    Format = R32F;
    Width = 1;
    Height = 1;
};
sampler sTexAdapt
{
    Texture = TexAdapt;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

texture TexLastAdapt
{
    Format = R32F;
    Width = 1;
    Height = 1;
};
sampler sTexLastAdapt
{
    Texture = TexLastAdapt;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

//---------------|
// :: Functions::|
//---------------|

float GetLuma(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Helper for VIG
float ScaleFun(float v, float mean_i)
{
    float r = 1.0 - (mean_i * 0.999999);
    return r * (1.0 / (1.0 + exp(-1.0 * (v - mean_i))) - 0.5);
}

//--------------------|
// :: Pixel Shaders ::|
//--------------------|

void PS_Luma(VS_OUTPUT input, out float luma : SV_Target)
{
    luma = GetLuma(tex2D(sTexColor, input.uv).rgb);
}

void PS_CalcMeansH(VS_OUTPUT input, out float2 mean_horiz : SV_Target)
{
    float2 ps = ReShade::PixelSize;
    float step = max(1.0, Radius / 3.0);
    
    float2 sum = 0.0;
    float count = 0.0;

    for (float x = -Radius; x <= Radius; x += step)
    {
        float val = tex2Dlod(sTexLuma, float4(input.uv + float2(x * ps.x, 0), 0, 0)).r;
        sum += float2(val, val * val);
        count += 1.0;
    }
    
    mean_horiz = sum / count;
}

void PS_CalcMeansV(VS_OUTPUT input, out float2 mean_corr : SV_Target)
{
    float2 ps = ReShade::PixelSize;
    float step = max(1.0, Radius / 3.0);
    
    float2 sum = 0.0;
    // r = Mean, g = Corr
    float count = 0.0;

    for (float y = -Radius; y <= Radius; y += step)
    {
        float2 val = tex2Dlod(sTexTempMeans, float4(input.uv + float2(0, y * ps.y), 0, 0)).rg;
        sum += val;
        count += 1.0;
    }
    
    mean_corr = sum / count;
}

void PS_GuidedFilterResult(VS_OUTPUT input, out float base_layer : SV_Target)
{
    float2 stats = tex2D(sTexStats, input.uv).rg;
    float mean_I = stats.r;
    float corr_I = stats.g;
    
    // Variance = E(I^2) - (E(I))^2
    float var_I = corr_I - mean_I * mean_I;
    float a = var_I / (var_I + Epsilon);

    float I = tex2D(sTexLuma, input.uv).r;
    base_layer = lerp(mean_I, I, a);
}


void PS_CalcAdapt(VS_OUTPUT input, out float adapt : SV_Target)
{
    float current = tex2Dlod(sTexLuma, float4(0.5, 0.5, 0, 8)).r;
    float last = tex2Dfetch(sTexLastAdapt, 0).r;
    
    if (AdaptationTime > 0.0)
    {
        float smoothFactor = saturate((FrameTime * 0.001) / AdaptationTime);
        adapt = lerp(last, current, smoothFactor);
    }
    else
    {
        adapt = current;
    }
    
    adapt = max(adapt, 1e-5);
}

void PS_SaveAdapt(VS_OUTPUT input, out float save : SV_Target)
{
    save = tex2Dfetch(sTexAdapt, 0).r;
}

float3 PS_FinalCombine(VS_OUTPUT input) : SV_Target
{
    float3 original = tex2D(sTexColor, input.uv).rgb;
    float L = tex2D(sTexLuma, input.uv).r;
    float Base = tex2D(sTexVarI, input.uv).r;
    
    L = max(L, 1e-5);
    Base = max(Base, 1e-5);
    float R_val = log(L) - log(Base);

    // SRS: Selective Reflectance Scaling
    float scene_mean = EnableAdaptation ?
        tex2Dfetch(sTexAdapt, 0).r : ManualExposure;

    float R_new = R_val;
    if (L > scene_mean)
    {
        float factor = pow(abs(L / scene_mean), 0.5);
        R_new = R_val * factor;
    }
    
    // VIG: Virtual Illumination
    float inv_L = 1.0 - L;
    float v1 = 0.2;
    float v3 = scene_mean;
    float v2 = 0.5 * (v1 + v3);
    float v5 = 0.8;
    float v4 = 0.5 * (v3 + v5);
    
    // Tone Mapping / Fusion Accumulators
    float A = 0.0;
    float B = 0.0;
    float exp_R_new = exp(R_new);
    float v_scales[5] = { v1, v2, v3, v4, v5 };

    [unroll]
    for (int i = 0; i < 5; i++)
    {
        float fvk = ScaleFun(v_scales[i], scene_mean);
        float I_k = (1.0 + fvk) * (L + fvk * inv_L);
        float Lk = exp_R_new * I_k;

        // if (i < 3) wk = I_k; else wk = 0.5 * (1.0 - I_k);
        float wk = (i < 3) ? I_k : 0.5 * (1.0 - I_k);
        wk = clamp(wk, 0.001, 1.0);

        A += Lk * wk;
        B += wk;
    }
    
    float L_final = A / (B + 1e-6);
    float ratio = clamp(L_final / L, 0.0, 3.0);
    
    return lerp(original, original * ratio, Strength);
}

technique PerceptualHDR
{
    pass Luma
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Luma;
        RenderTarget = TexLuma;
    }
    pass CalcAdapt
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcAdapt;
        RenderTarget = TexAdapt;
    }
    pass SaveAdapt
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveAdapt;
        RenderTarget = TexLastAdapt;
    }
    pass CalcMeansH
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcMeansH;
        RenderTarget = TexTempMeans;
    }
    pass CalcMeansV
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcMeansV;
        RenderTarget = TexStats;
    }
    pass GuidedFilter
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_GuidedFilterResult;
        RenderTarget = TexVarI;
    }
    pass Combine
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_FinalCombine;
    }
  }
}