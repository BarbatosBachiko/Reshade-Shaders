/*------------------.
| :: Description :: |
'-------------------/

    Emboss

    Version 1.1
    Author: Barbatos Bachiko 
    License: MIT

    About: Emboss effect.
    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
     
    Version 1.1
    + Code Clean
*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float emboss_intensity
<
    ui_type = "slider";
    ui_label = "Emboss Intensity";
    ui_tooltip = "Adjust the intensity of the emboss effect";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.1;
>
= 0.2;

uniform int kernel_type
<
    ui_type = "combo";
    ui_label = "Kernel Type"; 
    ui_tooltip = "Choose the emboss kernel type";
    ui_items = "Normal\0Inverted\0Soft\0";
>
= 2;

uniform float kernel_scale
<
    ui_type = "slider";
    ui_label = "Kernel Scale";
    ui_tooltip = "Adjust the scaling factor for the kernel effect";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.1;
>
= 1.0;

uniform float edge_highlight_intensity
<
    ui_type = "slider";
    ui_label = "Edge Highlight Intensity";
    ui_tooltip = "Adjust the intensity of edge highlighting";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.1;
>
= 0.2;

uniform int direction_mode
<
    ui_type = "combo";
    ui_label = "Emboss Direction";
    ui_tooltip = "Select the direction of the emboss effect";
    ui_items = "Vertical\0Horizontal\0Diagonal\0All\0";
>
= 3;

/*----------------.
| :: Functions :: |
'----------------*/

void get_kernel(int type, out float kernel[9])
{
    if (type == 0) // Normal
    {
        kernel[0] = -2;
        kernel[1] = -1;
        kernel[2] = 0;
        kernel[3] = -1;
        kernel[4] = 1;
        kernel[5] = 1;
        kernel[6] = 0;
        kernel[7] = 1;
        kernel[8] = 2;
    }
    else if (type == 1) // Inverted
    {
        kernel[0] = 2;
        kernel[1] = 1;
        kernel[2] = 0;
        kernel[3] = 1;
        kernel[4] = -1;
        kernel[5] = -1;
        kernel[6] = 0;
        kernel[7] = -1;
        kernel[8] = -2;
    }
    else if (type == 2) // Soft
    {
        kernel[0] = -1;
        kernel[1] = -1;
        kernel[2] = 0;
        kernel[3] = -1;
        kernel[4] = 0;
        kernel[5] = 1;
        kernel[6] = 0;
        kernel[7] = 1;
        kernel[8] = 1;
    }
    else // Default to Normal
    {
        kernel[0] = -2;
        kernel[1] = -1;
        kernel[2] = 0;
        kernel[3] = -1;
        kernel[4] = 1;
        kernel[5] = 1;
        kernel[6] = 0;
        kernel[7] = 1;
        kernel[8] = 2;
    }

    for (int i = 0; i < 9; i++)
    {
        kernel[i] *= kernel_scale;
    }
}

// 3x3 kernel
static const float2 offsets[9] =
{
    float2(-1, -1), float2(0, -1), float2(1, -1),
    float2(-1, 0), float2(0, 0), float2(1, 0),
    float2(-1, 1), float2(0, 1), float2(1, 1)
};

bool is_sample_active(int index, int mode)
{
    if (mode == 0) // Vertical
        return (index == 1 || index == 4 || index == 7);
    else if (mode == 1) // Horizontal
        return (index == 3 || index == 4 || index == 5);
    else if (mode == 2) // Diagonal
        return (index == 0 || index == 2 || index == 6 || index == 8);
    else
        return true;
}

float4 EmbossPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 color = tex2D(ReShade::BackBuffer, texcoord);
    float kernel[9];
    get_kernel(kernel_type, kernel);
    
    float3 result = 0.0;

    for (int i = 0; i < 9; i++)
    {
        if (is_sample_active(i, direction_mode))
        {
            float2 sampleCoord = texcoord + offsets[i] * 0.001;
            sampleCoord = clamp(sampleCoord, 0.0, 1.0);
            result += tex2D(ReShade::BackBuffer, sampleCoord).rgb * kernel[i];
        }
    }

    float gray = dot(result, float3(0.2989, 0.5870, 0.1140));
    result = float3(gray, gray, gray);
    float edge_highlight = saturate(gray * edge_highlight_intensity);
    color.rgb = lerp(color.rgb, result + edge_highlight, emboss_intensity);
    return float4(saturate(color.rgb), 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique EmbossFX
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = EmbossPS;
    }
}
