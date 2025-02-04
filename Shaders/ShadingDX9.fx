/*------------------.
| :: Description :: |
'-------------------/

    Shading (version 1.2)

    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adds shading based on visual complexity.

    Ideas for future improvements:
    * Yes

    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    
    Version 1.2:
    + Add Render Scale, get normal from depth

*/
namespace ShadingDX9
{
#ifndef RENDER_SCALE
#define RENDER_SCALE 1.0
#endif
#define INPUT_WIDTH BUFFER_RCP_WIDTH 
#define INPUT_HEIGHT BUFFER_RCP_HEIGHT
#define RENDER_WIDTH (INPUT_WIDTH * RENDER_SCALE)
#define RENDER_HEIGHT (INPUT_HEIGHT * RENDER_SCALE) 
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
    ui_items = "3x3\0 4x4\0 6x6\0";
    ui_default = 0;
> = 0;

uniform bool EnableMixWithSceneColor
<
    ui_label = "Mix with Scene Colors";
    ui_tooltip = "Mixes the shaded color with the original scene colors";
>
= false;

/*----------------.
| :: Functions :: |
'----------------*/

float4 LoadPixel(sampler2D tex, float2 uv)
{
    return tex2D(tex, uv);
}
    
float GetLinearDepth(float2 coords)
{
  return ReShade::GetLinearizedDepth(coords);
}
    
float3 GetNormalFromDepth(float2 coords)
{
 float2 texelSize = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
 float depthCenter = GetLinearDepth(coords);
 float depthX = GetLinearDepth(coords + float2(texelSize.x, 0.0));
 float depthY = GetLinearDepth(coords + float2(0.0, texelSize.y));
 float3 deltaX = float3(texelSize.x, 0.0, depthX - depthCenter);
 float3 deltaY = float3(0.0, texelSize.y, depthY - depthCenter);
 float3 normal = normalize(cross(deltaX, deltaY));
 return normal;
}
    
float4 ApplySharpness(float4 color, float2 texcoord)
{
// Sample neighboring pixels to apply the sharpness filter
float4 left = tex2D(ReShade::BackBuffer, texcoord + float2(-1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
float4 right = tex2D(ReShade::BackBuffer, texcoord + float2(1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).x, 0));
float4 top = tex2D(ReShade::BackBuffer, texcoord + float2(0, -1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
float4 bottom = tex2D(ReShade::BackBuffer, texcoord + float2(0, 1.0 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT).y));
float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);
 return clamp(sharpened, 0.0, 1.0);
}

float4 ApplyShading(float4 color, float2 texcoord)
{
    float2 bufferSize = float2(RENDER_WIDTH, RENDER_HEIGHT);
    float4 totalDiff = float4(0.0, 0.0, 0.0, 0.0);
    int count = 0;
    float3 currentNormal = GetNormalFromDepth(texcoord);

    if (SamplingArea == 0) // 3x3
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            [unroll]
            for (int y = -1; y <= 1; ++y)
            {
                if (x == 0 && y == 0)
                    continue;

                float4 neighborColor = LoadPixel(ReShade::BackBuffer, texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float4 diff = abs(neighborColor - color);
                float3 neighborNormal = GetNormalFromDepth(texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0);

                totalDiff += diff * normalInfluence;
                count++;
            }
        }
    }
    else if (SamplingArea == 1) // 4x4
    {
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            [unroll]
            for (int y = -2; y <= 2; ++y)
            {
                if (x == 0 && y == 0)
                    continue;

                float4 neighborColor = LoadPixel(ReShade::BackBuffer, texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float4 diff = abs(neighborColor - color);
                float3 neighborNormal = GetNormalFromDepth(texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0);

                totalDiff += diff * normalInfluence;
                count++;
            }
        }
    }
    else if (SamplingArea == 2) // 6x6
    {
        [unroll]
        for (int x = -3; x <= 3; ++x)
        {
            [unroll]
            for (int y = -3; y <= 3; ++y)
            {
                if (x == 0 && y == 0)
                    continue;

                float4 neighborColor = LoadPixel(ReShade::BackBuffer, texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float4 diff = abs(neighborColor - color);
                float3 neighborNormal = GetNormalFromDepth(texcoord + float2(x * bufferSize.x, y * bufferSize.y));
                float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0);

                totalDiff += diff * normalInfluence;
                count++;
            }
        }
    }

    // Compute shading based on accumulated differences
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

technique Shading_DX
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = ShadingPS;
    }
  }
}
