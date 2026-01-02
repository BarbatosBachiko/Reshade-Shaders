/*-------------------------------------------------|
| ::                 VividTone                  :: |
'--------------------------------------------------|
| Version 2.0                                      |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: Transforms ordinary game visuals    | 
| into vibrant, high-contrast scenes               |
'-------------------------------------------------*/

#include "ReShade.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Strength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "Tone Mapping Strength";
    ui_tooltip = "Overall intensity of the tone mapping effect";
    ui_category = "Basic Settings";
> = 1.0;

uniform float Adaptation <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "Adaptation Speed";
    ui_tooltip = "How quickly the algorithm adapts to scene changes";
    ui_category = "Basic Settings";
> = 0.5;

uniform float HighlightProtection <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "Highlight Protection";
    ui_tooltip = "Preserves detail in bright areas and prevents clipping";
    ui_category = "Dynamic Range";
> = 0.8;

uniform float ShadowRecovery <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "Shadow Recovery";
    ui_tooltip = "Lifts shadows while preserving black levels";
    ui_category = "Dynamic Range";
> = 1.0;

uniform float LocalContrast <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "Local Contrast";
    ui_tooltip = "Enhances micro-contrast for better detail perception";
    ui_category = "Dynamic Range";
> = 1.2;

uniform float Vibrance <
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
    ui_label = "Vibrance";
    ui_tooltip = "Smart saturation";
    ui_category = "Color";
> = 0.2;

uniform float ColorGrading <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_label = "Smart Color Grading";
    ui_tooltip = "Intelligent color adjustment based on luminance";
    ui_category = "Color";
> = 0.5;

uniform float WhitePoint <
    ui_type = "slider";
    ui_min = 0.5; ui_max = 2.0;
    ui_label = "White Point";
    ui_tooltip = "Target white point for tone mapping";
    ui_category = "Advanced";
> = 1.0;

uniform float BlackPoint <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.5;
    ui_label = "Black Point";
    ui_tooltip = "Target black point for tone mapping";
    ui_category = "Advanced";
> = 0.0;

uniform bool EnableDebug <
    ui_label = "Show Histogram Debug";
    ui_tooltip = "Visualizes the luminance histogram";
    ui_category = "Advanced";
> = false;

//----------------|
// :: Textures :: |
//----------------|

texture texHistogram
{
    Width = 128;
    Height = 1;
    Format = R16F;
};

sampler sHistogram
{
    Texture = texHistogram;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = POINT;
    MagFilter = POINT;
};

texture texLuminance
{
    Width = BUFFER_WIDTH / 6;
    Height = BUFFER_HEIGHT / 6;
    Format = R16F;
    MipLevels = 1;
};

sampler sLuminance
{
    Texture = texLuminance;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
};

texture texAdaptation
{
    Width = 1;
    Height = 1;
    Format = R16F;
};

sampler sAdaptation
{
    Texture = texAdaptation;
    MinFilter = POINT;
    MagFilter = POINT;
};

texture texPrevAdaptation
{
    Width = 1;
    Height = 1;
    Format = R16F;
};

sampler sPrevAdaptation
{
    Texture = texPrevAdaptation;
    MinFilter = POINT;
    MagFilter = POINT;
};

//-------------|
// :: Utility::|
//-------------|

static const float EPSILON = 1e-6;
static const float LOG_MIN = -10.0;
static const float LOG_MAX = 10.0;

#if (BUFFER_COLOR_SPACE == 3 || BUFFER_COLOR_SPACE == 4)
#define IS_HDR10 1
#define HDR_PEAK_LUMINANCE 10000.0
#elif (BUFFER_COLOR_SPACE == 2)
#define IS_HDR10 0
#define HDR_PEAK_LUMINANCE 80.0
#else
#define IS_HDR10 0
#define HDR_PEAK_LUMINANCE 1.0
#endif

#define IS_HDR (BUFFER_COLOR_SPACE > 1)

float GetLuminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 ToLinear(float3 color)
{
#if (BUFFER_COLOR_SPACE == 1) // sRGB
    return pow(abs(color), 2.2);
#elif (BUFFER_COLOR_SPACE == 3) // HDR10 ST2084 (PQ)
    float3 p = pow(abs(color), 1.0 / 78.84375);
    return pow(max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125) * HDR_PEAK_LUMINANCE;
#else
    return color;
#endif
}

float3 FromLinear(float3 color)
{
#if (BUFFER_COLOR_SPACE == 1) // sRGB
    return pow(abs(color), 1.0 / 2.2);
#elif (BUFFER_COLOR_SPACE == 3) // HDR10 ST2084 (PQ)
    float3 y = pow(abs(color / HDR_PEAK_LUMINANCE), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
#else
    return color;
#endif
}

// RGB to HSV
float3 RGBtoHSV(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + EPSILON)), d / (q.x + EPSILON), q.x);
}

// HSV to RGB
float3 HSVtoRGB(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

// ACES Filmic
float3 ACESFilm(float3 x, float adapted_lum)
{
    float a = 2.51 * adapted_lum;
    static const float b = 0.03;
    float c = 2.43 * adapted_lum;
    static const float d = 0.59;
    static const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Uncharted 2 
float3 Uncharted2Tonemap(float3 x)
{
    static const float A = 0.15;
    static const float B = 0.50;
    static const float C = 0.10;
    static const float D = 0.20;
    static const float E = 0.02;
    static const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Reinhard
float Reinhard(float lum, float max_white)
{
    float numerator = lum * (1.0 + (lum / (max_white * max_white)));
    return numerator / (1.0 + lum);
}

//---------------------|
// :: Pixel Shaders :: |
//---------------------|

float PS_Luminance(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    color = ToLinear(color);
    float lum = GetLuminance(color);
#if IS_HDR
    lum = clamp(lum, EPSILON, HDR_PEAK_LUMINANCE);
#endif
    
    return log2(max(lum, EPSILON));
}

float PS_Histogram(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float bin = floor(texcoord.x * 128.0);
    float count = 0.0;
    float2 pixel_size = ReShade::PixelSize * 6.0;
    
    [loop]
    for (float y = 0; y < 1.0; y += pixel_size.y)
    {
        [loop]
        for (float x = 0; x < 1.0; x += pixel_size.x)
        {
            float log_lum = tex2Dlod(sLuminance, float4(x, y, 0, 0)).r;
            float normalized_lum = saturate((log_lum - LOG_MIN) / (LOG_MAX - LOG_MIN));
            float pixel_bin = floor(normalized_lum * 127.0);
            
            if (abs(pixel_bin - bin) < 0.5)
                count += 1.0;
        }
    }
    
    return count;
}

float PS_Adaptation(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float total_weight = 0.0;
    float weighted_sum = 0.0;
    
    [loop]
    for (int i = 0; i < 128; i++)
    {
        float bin_value = tex2Dfetch(sHistogram, int2(i, 0)).r;
        float normalized_lum = float(i) / 127.0;
        
        float weight = 1.0 - abs(normalized_lum - 0.5) * 2.0;
        weight = weight * weight;
        
        weighted_sum += normalized_lum * bin_value * weight;
        total_weight += bin_value * weight;
    }
    
    float avg_lum = weighted_sum / max(total_weight, EPSILON);
    float log_avg_lum = lerp(LOG_MIN, LOG_MAX, avg_lum);
    float scene_lum = exp2(log_avg_lum);
    float prev_adaptation = tex2Dfetch(sPrevAdaptation, int2(0, 0)).r;
    float adaptation_rate = lerp(0.05, 0.5, Adaptation);
    
    float new_adaptation = lerp(prev_adaptation, scene_lum, adaptation_rate);
    
    return max(new_adaptation, EPSILON);
}

float PS_StoreAdaptation(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    return tex2Dfetch(sAdaptation, int2(0, 0)).r;
}

float3 PS_ToneMap(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    color = ToLinear(color);
    
    float original_lum = GetLuminance(color);
    float adapted_lum = tex2Dfetch(sAdaptation, int2(0, 0)).r;

#if IS_HDR
    static const float max_adapted = HDR_PEAK_LUMINANCE;
    static const float target_lum = 100.0;
    static const float max_exposure = 2.0;
#else
    static const float max_adapted = 10.0;
    static const float target_lum = 0.18;
    static const float max_exposure = 10.0;
#endif
    
    adapted_lum = clamp(adapted_lum, 0.001, max_adapted);
    float exposure = target_lum / max(adapted_lum, EPSILON);
    exposure = clamp(exposure, 0.1, max_exposure);
    
    float3 exposed = color;
    
#if IS_HDR
    float exposure_factor = lerp(1.0, exposure, saturate(1.0 - original_lum / 1000.0) * HighlightProtection);
#else
    float exposure_factor = lerp(1.0, exposure, saturate(1.0 - original_lum) * HighlightProtection);
#endif

    exposed *= exposure_factor;
    float pixel_lum = GetLuminance(exposed);
    float local_lum = pixel_lum;
    
    if (LocalContrast != 1.0)
    {
        float3 local_mean = 0.0;
        float2 step_size = ReShade::PixelSize * 3.0;
        
        local_mean += ToLinear(tex2D(ReShade::BackBuffer, texcoord).rgb);
        local_mean += ToLinear(tex2D(ReShade::BackBuffer, texcoord + float2(step_size.x, 0)).rgb);
        local_mean += ToLinear(tex2D(ReShade::BackBuffer, texcoord - float2(step_size.x, 0)).rgb);
        local_mean += ToLinear(tex2D(ReShade::BackBuffer, texcoord + float2(0, step_size.y)).rgb);
        local_mean += ToLinear(tex2D(ReShade::BackBuffer, texcoord - float2(0, step_size.y)).rgb);
        
        local_mean /= 5.0;
        local_lum = GetLuminance(local_mean) * exposure_factor;
        
        float contrast_ratio = pixel_lum / max(local_lum, EPSILON);
        float contrast_factor = lerp(1.0, contrast_ratio, (LocalContrast - 1.0) * 0.3);
        contrast_factor = clamp(contrast_factor, 0.5, 2.0);
        
        exposed *= contrast_factor;
        pixel_lum *= contrast_factor;
    }

#if IS_HDR
    float max_white = lerp(1000.0, 10000.0, 1.0 - HighlightProtection);
#else
    float max_white = lerp(0.8, 2.0, 1.0 - HighlightProtection);
#endif

    float tone_mapped_lum = Reinhard(pixel_lum, max_white);
    
    if (ShadowRecovery > 0.0)
    {
#if IS_HDR
        float shadow_mask = pow(saturate(1.0 - tone_mapped_lum / 100.0), 3.0);
        float shadow_lift = shadow_mask * ShadowRecovery * 10.0;
#else
        float shadow_mask = pow(saturate(1.0 - tone_mapped_lum), 3.0);
        float shadow_lift = shadow_mask * ShadowRecovery * 0.2;
#endif
        tone_mapped_lum += shadow_lift;
    }
    
    float lum_ratio = tone_mapped_lum / max(pixel_lum, EPSILON);
    
#if IS_HDR
    lum_ratio = clamp(lum_ratio, 0.0, 10.0);
#else
    lum_ratio = clamp(lum_ratio, 0.0, 2.0);
#endif

    float3 result = exposed * lum_ratio;
    result = (result - BlackPoint) / max(WhitePoint - BlackPoint, 0.1);

#if IS_HDR
    result = max(result, 0.0);
#else
    result = saturate(result);
#endif
    
    // Vibrance 
    if (Vibrance != 0.0)
    {
        float result_lum = GetLuminance(result);
        float avg_sat = (max(max(result.r, result.g), result.b) - min(min(result.r, result.g), result.b)) / max(result_lum, EPSILON);
        float sat_mask = 1.0 - pow(saturate(avg_sat), 2.0);
        float vibrance_amount = Vibrance * sat_mask;
        
        result = lerp(result_lum.xxx, result, 1.0 + vibrance_amount);
    }
    
    if (ColorGrading > 0.0)
    {
#if IS_HDR
        float3 hsv = RGBtoHSV(saturate(result / 100.0));
#else
        float3 hsv = RGBtoHSV(saturate(result));
#endif
        float mid_tone_factor = 1.0 - abs(hsv.z - 0.5) * 2.0;
        hsv.y = lerp(hsv.y, hsv.y * (1.0 + mid_tone_factor * 0.15), ColorGrading);
        
#if IS_HDR
        result = HSVtoRGB(hsv) * 100.0;
#else
        result = HSVtoRGB(hsv);
#endif
    }
    
    result = lerp(color, result, Strength);
    result = FromLinear(result);
    
    if (EnableDebug && texcoord.y > 0.9)
    {
        float bin = floor(texcoord.x * 128.0);
        float hist_value = tex2Dfetch(sHistogram, int2(bin, 0)).r;
        float normalized = hist_value / 500.0;
        
        if ((1.0 - texcoord.y) * 10.0 < normalized)
            result = float3(1.0, 1.0, 0.0);
    }
    
    return result;
}

//---------------|
// :: Technique ::|
//---------------|

technique VividTone
{
    pass CalculateLuminance
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Luminance;
        RenderTarget = texLuminance;
    }
    pass BuildHistogram
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Histogram;
        RenderTarget = texHistogram;
    }
    pass CalculateAdaptation
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Adaptation;
        RenderTarget = texAdaptation;
    }
    pass StoreAdaptation
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_StoreAdaptation;
        RenderTarget = texPrevAdaptation;
    }
    pass ApplyToneMapping
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_ToneMap;
    }
}
