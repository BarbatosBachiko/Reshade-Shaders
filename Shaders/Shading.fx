/*------------------.
| :: Description :: |
'-------------------/

    Shading (version 1.1)

    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adds shading based on visual complexity.

    Ideas for future improvements:
    * Yes

    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    
    Version 1.1:
    + Update ui_max for ShadingIntensity
    + Use of 2x2 sampling instead of 3x3
    * Added Sharpness

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

// Sharpnes
float4 ApplySharpness(float4 color, float2 texcoord)
{
    // Sample neighboring pixels to apply the sharpness filter
    float4 left = tex2D(BackBuffer, texcoord + float2(-1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 right = tex2D(BackBuffer, texcoord + float2(1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
    float4 top = tex2D(BackBuffer, texcoord + float2(0, -1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
    float4 bottom = tex2D(BackBuffer, texcoord + float2(0, 1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));

    // Apply the sharpening filter: enhance the current pixel and subtract neighboring pixels
    float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);

    // Clamp the result to ensure valid color values
    return clamp(sharpened, 0.0, 1.0);
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

    // Sampling a 2x2 area of neighboring pixels
    for (int x = -1; x <= 1; x += 2)
    {
        for (int y = -1; y <= 1; y += 2)
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
            float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0); 
            totalDiff += diff * normalInfluence;
        }
    }

    // Calculates visual complexity based on color differences
    float complexity = (dot(totalDiff.rgb, float3(1.0, 1.0, 1.0)) / count);

    // Adjusts shading intensity based on complexity
    float vrsFactor = 1.0 - (complexity * ShadingIntensity);

    // Applies mixing with scene colors if enabled
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
