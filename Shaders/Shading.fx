/*------------------.
| :: Description :: |
'-------------------/

    Shading (only for DX11 and above) 
    
    Version 1.2
    Author: Barbatos Bachiko
    License: MIT

    About: This shader adds shading based on visual complexity.

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float ShadingIntensity < 
    ui_type = "slider";
    ui_label = "Shading"; 
    ui_tooltip = "Control the Shading."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 0.5; 
> = 0.5;

uniform float sharpness < 
    ui_type = "slider";
    ui_label = "Sharpness"; 
    ui_tooltip = "Control the sharpness level."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 0.2; 
> = 0.2;

uniform int SamplingArea < 
    ui_type = "combo";
    ui_label = "Sampling Area"; 
    ui_tooltip = "Select the sampling area size."; 
    ui_items = "3x3\0 4x4\0 6x6\0 8x8\0 10x10\0"; 
    ui_default = 0; 
> = 0;

uniform bool EnableShading < 
    ui_label = "Enable Shading";
    ui_tooltip = "Enable or disable"; 
> = true;

uniform bool EnableMixWithSceneColor < 
    ui_label = "Mix with Scene Colors";
    ui_tooltip = "Mixes the shaded color with the original scene colors"; 
> = false;

/*---------------.
| :: Textures :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

texture2D NormalMapTex : NORMAL;
sampler NormalMap
{
    Texture = NormalMapTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

float4 LoadPixel(sampler2D tex, float2 uv)
{
    return tex2D(tex, uv);
}

float3 LoadNormal(float2 texcoord)
{
    return normalize(tex2D(NormalMap, texcoord).rgb * 2.0 - 1.0);
}

float4 ApplySharpness(float4 color, float2 texcoord)
{
    float4 left = tex2D(BackBuffer, texcoord + float2(-1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 right = tex2D(BackBuffer, texcoord + float2(1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 top = tex2D(BackBuffer, texcoord + float2(0, -1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
    float4 bottom = tex2D(BackBuffer, texcoord + float2(0, 1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));

    float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);
    return clamp(sharpened, 0.0, 1.0);
}

float4 ApplyShading(float4 color, float2 texcoord)
{
    if (!EnableShading)
    {
        return color;
    }

    float2 bufferSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float4 totalDiff = float4(0.0, 0.0, 0.0, 0.0);
    int count = 0;

    float3 currentNormal = LoadNormal(texcoord);

    int sampleOffset = (SamplingArea == 0) ? 1 :
                       (SamplingArea == 1) ? 2 :
                       (SamplingArea == 2) ? 3 :
                       (SamplingArea == 3) ? 4 : 5;

    for (int x = -sampleOffset; x <= sampleOffset; ++x)
    {
        for (int y = -sampleOffset; y <= sampleOffset; ++y)
        {
            if (x == 0 && y == 0)
                continue;

            float4 neighborColor = LoadPixel(BackBuffer, texcoord + float2(x * bufferSize.x, y * bufferSize.y));
            float4 diff = abs(neighborColor - color);
            float3 neighborNormal = LoadNormal(texcoord + float2(x * bufferSize.x, y * bufferSize.y));
            float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0);

            totalDiff += diff * normalInfluence;
            count++;
        }
    }

    float complexity = dot(totalDiff.rgb, float3(1.0, 1.0, 1.0)) / count;
    float vrsFactor = 1.0 - (complexity * ShadingIntensity);

    if (EnableMixWithSceneColor)
    {
        return lerp(color, color * vrsFactor, 0.5);
    }

    return color * vrsFactor;
}

float4 ShadingPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 color = tex2D(ReShade::BackBuffer, texcoord);
    color = ApplyShading(color, texcoord);
    color = ApplySharpness(color, texcoord);
    return color;
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique Shading
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = ShadingPS;
    }
}
