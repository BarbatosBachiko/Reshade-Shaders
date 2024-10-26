/*------------------.
| :: Description :: |
'-------------------/

    Shading (version 1.0)

    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adds shading based on visual complexity.

    Ideas for future improvements:
    * Yes
    
    Version 1.0:
    * Added sampling of a 3x3 area to improve shading precision
    * Implemented normal map for shading adjustment
    * Added option to mix with scene colors

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float ShadingIntensity
<
    ui_type = "slider";
    ui_label = "Shading Intensity";
    ui_tooltip = "Control the shading intensity based on visual complexity";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
>
= 0.5;

uniform bool EnableShading
<
    ui_label = "Enable Shading";
    ui_tooltip = "Enables or disables visual complexity shading";
>
= true;

uniform bool EnableMixWithSceneColor
<
    ui_label = "Mix with Scene Colors";
    ui_tooltip = "Mixes the shaded color with the original scene colors";
>
= false;

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

// Loads the color from the neighboring pixel
float4 LoadPixel(sampler2D tex, float2 uv)
{
    return tex2D(tex, uv);
}

// Loads the normal from the normal map
float3 LoadNormal(float2 texcoord)
{
    // Assumes the normal map is in [0, 1] format
    float3 normal = tex2D(NormalMap, texcoord).rgb * 2.0 - 1.0;
    return normalize(normal);
}

// Applies shading based on visual complexity
float4 ApplyShading(float4 color, float2 texcoord)
{
    if (!EnableShading)
    {
        return color;
    }

    float2 bufferSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float4 totalDiff = float4(0.0, 0.0, 0.0, 0.0); 
    int count = 0;

    // Loads the current normal
    float3 currentNormal = LoadNormal(texcoord);

    // Sampling a 3x3 area of neighboring pixels
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            // Skip the central pixel
            if (x == 0 && y == 0)
                continue;

            // Loads the neighboring pixel
            float4 neighborColor = LoadPixel(BackBuffer, texcoord + float2(x * bufferSize.x, y * bufferSize.y));
            float4 diff = abs(neighborColor - color);
            totalDiff += diff;
            count++;

            // Loads the normal of the neighbor and adjusts the shading intensity
            float3 neighborNormal = LoadNormal(texcoord + float2(x * bufferSize.x, y * bufferSize.y));
            float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0); // Calculate influence factor
            totalDiff *= normalInfluence; // Adjust the difference based on normal influence
        }
    }

    // Calculates visual complexity based on color differences
    float complexity = (dot(totalDiff.rgb, float3(1.0, 1.0, 1.0)) / count);

    // Adjusts shading intensity based on complexity
    float vrsFactor = 1.0 - (complexity * ShadingIntensity);

    // Applies mixing with scene colors if enabled
    if (EnableMixWithSceneColor)
    {
        return lerp(color, color * vrsFactor, 0.5); // Mix 50% of the original color with the shaded color
    }

    return color * vrsFactor;
}

float4 ShadingPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // Applies the calculated shading
    float4 color = tex2D(ReShade::BackBuffer, texcoord);
    return ApplyShading(color, texcoord);
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