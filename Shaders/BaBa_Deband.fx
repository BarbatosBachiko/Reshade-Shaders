/*----------------------------------------------|
| ::                B-Deband                 :: |
|-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include ".\bb_include\bb_reshade.fxh"
#include ".\bb_include\bb_common.fxh"
#include ".\bb_include\bb_colorspace.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Strength <
    ui_type = "slider";
    ui_label = "Strength";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 1.0;

uniform float Radius <
    ui_type = "slider";
    ui_label = "Radius";
    ui_min = 0.0;
    ui_max = 128.0;
    ui_step = 1.0;
> = 40.0;

uniform float Threshold <
    ui_type = "slider";
    ui_label = "Threshold";
    ui_min = 0.001;
    ui_max = 0.10;
    ui_step = 0.0001;
> = 0.016;

uniform int Iterations <
    ui_type = "slider";
    ui_label = "Quality Iterations";
    ui_min = 1;
    ui_max = 3;
> = 1;

static const float ANGLE_OFFSETS[3] = { 0.0, 2.09439510239, 4.18879020478 };

/*------------------.
| :: Functions ::   |
'------------------*/

float2 GetDirection(float base_angle, int iteration)
{
    float angle = base_angle + ANGLE_OFFSETS[iteration];
    float2 dir;
    sincos(angle, dir.y, dir.x);
    return dir;
}

float GetDither(float2 pos)
{
    return frac(52.9829189 * frac(dot(pos, float2(0.06711056, 0.00583715)))) - 0.5;
}

/*------------------.
| :: Pixel Shader :: |
'------------------*/

float4 MainPS(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    const float3 color = tex2D(bb::BackBuffer, uv).rgb;
    float3 result = color;
    
    float base_angle = frac(dot(pos.xy, float2(0.754877666, 0.569840296))) * (2.0 * PI);

    [loop]
    for (int i = 0; i < Iterations; i++)
    {
        float2 dir = GetDirection(base_angle, i);
        float radius_scale = 1.0 + (float(i) * 0.618034);
        float2 offset = dir * (Radius * radius_scale * bb::PixelSize);

        float3 s1 = tex2Dlod(bb::BackBuffer, float4(uv + offset, 0.0, 0.0)).rgb;
        float3 s2 = tex2Dlod(bb::BackBuffer, float4(uv - offset, 0.0, 0.0)).rgb;

        float3 avg = (s1 + s2) * 0.5;
        float3 delta = result - avg;
        
        float luma_diff = GetLuminance(abs(delta));
        float chroma_diff = length(delta.gb) * 0.5;
        float total_diff = luma_diff + chroma_diff;

        float adaptive_threshold = Threshold * (1.0 + float(i) * 0.25);
        
        float blend = smoothstep(adaptive_threshold * 0.3, adaptive_threshold * 1.8, total_diff);
        result = lerp(avg, result, blend);
    }
    
    result = lerp(color, result, Strength);

    float dither = GetDither(pos.xy);
    result += dither * Threshold * 0.3;
        
    return float4(saturate(result), 1.0);
}

technique BaBa_Deband
 <
    ui_label = "BaBa: Deband";
 >
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = MainPS;
    }
}