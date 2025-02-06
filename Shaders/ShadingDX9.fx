/*------------------.
| :: Description :: |
'-------------------/

    Shading
    
    Version 1.3
    Author: Barbatos Bachiko
    License: MIT

    About:
    This shader adds shading based on visual complexity.
    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    
    Version 1.3:
    x Maximum temp register index exceeded - Fixed
    + Restructuring main shading
*/
namespace ShadingDX9
{

/*---------------.
| :: Defines :: |
'---------------*/

#ifndef RENDER_SCALE
#define RENDER_SCALE 2.0
#endif

#define INPUT_WIDTH BUFFER_RCP_WIDTH 
#define INPUT_HEIGHT BUFFER_RCP_HEIGHT
#define RENDER_WIDTH (INPUT_WIDTH * RENDER_SCALE)
#define RENDER_HEIGHT (INPUT_HEIGHT * RENDER_SCALE)

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"

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
> = 1.5;

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

    uniform bool EnableMixWithSceneColor <
    ui_label = "Mix with Scene Colors";
    ui_tooltip = "Mixes the shaded color with the original scene colors";
> = false;

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
 return normalize(cross(deltaX, deltaY));
}
    
float4 ApplySharpness(float4 color, float2 texcoord)
{
 float2 texelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
 float4 left = tex2D(ReShade::BackBuffer, texcoord + float2(-texelSize.x, 0));
 float4 right = tex2D(ReShade::BackBuffer, texcoord + float2(texelSize.x, 0));
 float4 top = tex2D(ReShade::BackBuffer, texcoord + float2(0, -texelSize.y));
 float4 bottom = tex2D(ReShade::BackBuffer, texcoord + float2(0, texelSize.y));
 float4 sharpened = color * (1.0 + sharpness) - (left + right + top + bottom) * (sharpness * 0.25);
 return clamp(sharpened, 0.0, 1.0);
}
  
float4 ApplyShading(float4 color, float2 texcoord)
{
 float2 bufferSize = float2(RENDER_WIDTH, RENDER_HEIGHT);
 float totalDiff = 0.0;
 int count = 0;

 float3 currentNormal = GetNormalFromDepth(texcoord);
int range = (SamplingArea == 0) ? 1 : (SamplingArea == 1) ? 2 : 3;

 [loop]
 for (int x = -range; x <= range; ++x)
 {
 [loop]
 for (int y = -range; y <= range; ++y)
 {
 if (x == 0 && y == 0)
 continue;

  float2 offset = float2(x, y) * bufferSize;
  float4 neighborColor = tex2Dlod(ReShade::BackBuffer, float4(texcoord + offset, 0, 0));
  float4 diff = abs(neighborColor - color);
  float diffValue = (diff.r + diff.g + diff.b) * (1.0 / 3.0);

  float3 neighborNormal = GetNormalFromDepth(texcoord + offset);
  float normalInfluence = max(dot(currentNormal, neighborNormal), 0.0);

   totalDiff += diffValue * normalInfluence;
   count++;
 }
}

 float complexity = totalDiff / count;
 float vrsFactor = 1.0 - (complexity * ShadingIntensity);
 return EnableMixWithSceneColor ? lerp(color, color * vrsFactor, 0.5) : color * vrsFactor;
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
