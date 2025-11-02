/*----------------------------------------------|
| ::               S_Outline                 :: |
'-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| successor to Outline.fx and Shading.fx        |
| from BBFX                                     |   
'----------------------------------------------*/

#include "ReShade.fxh"

//-----------------|
// :: UI (New)  ::  |
//-----------------|

uniform float OutlineIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Outline Settings";
    ui_label = "Outline Strength";
    ui_tooltip = "Overall intensity of the outlines.\n0.0 = Off\n1.0 = Full Strength";
> = 1.0;

uniform float3 OutlineColor <
    ui_type = "color";
    ui_category = "Outline Settings";
    ui_label = "Outline Color";
    ui_tooltip = "Sets the color of the outlines.";
> = float3(0.0, 0.0, 0.0);

uniform float OutlineThickness <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "Outline Settings";
    ui_label = "Outline Thickness";
    ui_tooltip = "Sets the width of the outlines in pixels.";
> = 1.5;

uniform int OutlineMethod <
    ui_type = "combo";
    ui_items = "Depth Only\0Color Only\0Depth + Color (AND)\0Depth OR Color\0";
    ui_category = "Outline Settings";
    ui_label = "Outline Detection Method";
    ui_tooltip = "Choose how edges are detected:\n"
                 "- Depth Only: Uses 3D depth differences.\n"
                 "- Color Only: Uses color/brightness differences.\n"
                 "- Depth + Color (AND): Both must trigger (strictest).\n"
                 "- Depth OR Color: Either can trigger (loosest).";
> = 3;

uniform float DepthSensitivity <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.1; ui_step = 0.001;
    ui_category = "Detection Details";
    ui_label = "Depth Sensitivity";
    ui_tooltip = "Controls sensitivity to depth changes.\n"
                 "Higher = more sensitive, detects smaller gaps.\n"
                 "Lower = less sensitive, only detects large gaps.";
> = 0.01;

uniform float ColorSensitivity <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Detection Details";
    ui_label = "Color Sensitivity";
    ui_tooltip = "Controls sensitivity to color changes.\n"
                 "Higher = more sensitive, detects subtle color shifts.\n"
                 "Lower = less sensitive, only detects strong color changes.";
> = 0.3;

uniform int ColorDetectionMode <
    ui_type = "combo";
    ui_items = "Luminance (Brightness)\0Full RGB Color\0";
    ui_category = "Detection Details";
    ui_label = "Color Detection Mode";
    ui_tooltip = "Luminance: Detects edges based on brightness (faster).\n"
                 "Full RGB Color: Detects edges based on all color channels (more accurate).";
> = 0;

uniform bool bEnableWobble <
    ui_category = "Wobble Effect";
    ui_label = "Enable Wobble Effect";
    ui_tooltip = "Adds an animated 'wobble' to the outlines for a hand-drawn look.";
> = true;

uniform float WobbleAmount <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "Wobble Effect";
    ui_label = "Wobble Amount";
    ui_tooltip = "How far the lines wobble from their original position.";
> = 2.0;

uniform float WobbleSpeed <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.1;
    ui_category = "Wobble Effect";
    ui_label = "Wobble Speed";
    ui_tooltip = "The animation speed of the wobble.";
> = 1.0;

uniform float WobbleFrequency <
    ui_type = "drag";
    ui_min = 1.0; ui_max = 50.0; ui_step = 1.0;
    ui_category = "Wobble Effect";
    ui_label = "Wobble Frequency";
    ui_tooltip = "The spatial frequency (waviness) of the wobble pattern.";
> = 10.0;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Off\0Outline Mask Only\0";
    ui_category = "Debug";
    ui_label = "Debug View";
    ui_tooltip = "Show a debug view instead of the final image.\n"
                 "Off: Show final image.\n"
                 "Outline Mask Only: Show the black/white outline mask.";
> = 0;

uniform float timer < source = "timer"; >;

//-------------------|
// :: Functions ::   |
//-------------------|

float GetDepth(float2 uv)
{
    return ReShade::GetLinearizedDepth(uv);
}

float GetLuminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float GetColorDifference(float3 color1, float3 color2)
{
    if (ColorDetectionMode == 0) // 0 = Luminance
    {
        return abs(GetLuminance(color1) - GetLuminance(color2));
    }
    else // 1 = Full RGB
    {
        return length(color1 - color2);
    }
}

//--------------------|
// :: Pixel Shader :: |
//--------------------|

void PS_OutlineOnly(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target0)
{
    float3 original_color = tex2D(ReShade::BackBuffer, uv).rgb;
    
    float2 wobble_uv = 0;
    if (bEnableWobble)
    {
        float time = (timer / 1000.0f) * WobbleSpeed;
        float wobble_offset_x = sin(time + uv.y * WobbleFrequency) * WobbleAmount * ReShade::PixelSize.x;
        float wobble_offset_y = cos(time + uv.x * WobbleFrequency) * WobbleAmount * ReShade::PixelSize.y;
        wobble_uv = float2(wobble_offset_x, wobble_offset_y);
    }
    
    float2 offsets[8] =
    {
        float2(-1, 0), float2(1, 0),
        float2(0, -1), float2(0, 1),
        float2(-1, -1), float2(1, -1),
        float2(-1, 1), float2(1, 1)
    };
    
    float depth_diff = 0.0;
    if (OutlineMethod == 0 || OutlineMethod == 2 || OutlineMethod == 3)
    {
        float center_depth = GetDepth(uv);
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            float sample_depth = GetDepth(uv + offsets[i] * ReShade::PixelSize * OutlineThickness + wobble_uv);
            depth_diff += abs(center_depth - sample_depth);
        }
        depth_diff /= 8.0;
    }
    
    float color_diff = 0.0;
    if (OutlineMethod == 1 || OutlineMethod == 2 || OutlineMethod == 3)
    {
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            float3 sample_color = tex2D(ReShade::BackBuffer, uv + offsets[i] * ReShade::PixelSize * OutlineThickness + wobble_uv).rgb;
            color_diff += GetColorDifference(original_color, sample_color);
        }
        color_diff /= 8.0;
    }
    
    float outline_mask = 0.0;
    
    if (OutlineMethod == 0)
    {
        outline_mask = smoothstep(0.0, DepthSensitivity * OutlineThickness, depth_diff);
    }
    else if (OutlineMethod == 1)
    {
        outline_mask = smoothstep(0.0, ColorSensitivity, color_diff);
    }
    else if (OutlineMethod == 2)
    {
        float depth_mask = smoothstep(0.0, DepthSensitivity * OutlineThickness, depth_diff);
        float color_mask = smoothstep(0.0, ColorSensitivity, color_diff);
        outline_mask = min(depth_mask, color_mask);
    }
    else if (OutlineMethod == 3)
    {
        float depth_mask = smoothstep(0.0, DepthSensitivity * OutlineThickness, depth_diff);
        float color_mask = smoothstep(0.0, ColorSensitivity, color_diff);
        outline_mask = max(depth_mask, color_mask);
    }
    
    outline_mask *= OutlineIntensity;
    
    // Debug view
    if (DebugView == 1)
    {
        outColor = float4(outline_mask.xxx, 1.0);
        return;
    }
    
    float3 final_color = lerp(original_color, OutlineColor, outline_mask);
    
    outColor = float4(final_color, 1.0);
}

technique S_Outline
{
    pass Outline
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_OutlineOnly;
    }
}