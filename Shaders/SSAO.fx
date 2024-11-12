/*------------------.
| :: Description :: |
'-------------------/

SSAO (version 0.5)

	Author: Barbatos Bachiko
	License: MIT

	About:
    Screen-Space Ambient Occlusion (SSAO) using a mix of fixed and random direction sampling for depth-based occlusion calculations.
    
	Ideas for future improvement:
	Add a debug mode or more

	History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
	Version 0.5
	* Simple functions added

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int viewMode <
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for SSAO";
    ui_items = 
    "Normal\0"; 
> = 0;

uniform int qualityLevel
< 
	ui_type = "combo";
	ui_label = "Quality Level";
	ui_tooltip = "Select quality level for ambient occlusion";
	ui_items = 
	"Low\0"
	"Medium\0"
	"High\0";
>
= 0;

uniform float intensity
<
	ui_type = "slider";
	ui_label = "Occlusion Intensity";
	ui_tooltip = "Adjust the intensity of ambient occlusion";
	ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
>
= 0.1;

uniform int aoType
<
    ui_type = "combo";
    ui_label = "AO Type";
    ui_tooltip = "Select ambient occlusion type";
    ui_items = 
    "AO-F\0" 
    "AO-R\0"; 
>
= 1;

/*---------------.
| :: Textures :: |
'---------------*/

texture2D ColorTex : COLOR;
texture2D DepthTex : DEPTH;
sampler ColorSampler
{
    Texture = ColorTex;
};
sampler DepthSampler
{
    Texture = DepthTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

float3 RandomDirection(float2 texcoord, int sampleIndex)
{
    float randomValue = frac(sin(dot(texcoord * 100.0 + float2(sampleIndex, 0.0), float2(12.9898, 78.233))) * 43758.5453);
    float phi = randomValue * 2.0 * 3.14159265359;
    float theta = acos(1.0 - 2.0 * (frac(randomValue * 0.12345)));

    float3 dir;
    dir.x = sin(theta) * cos(phi);
    dir.y = sin(theta) * sin(phi);
    dir.z = cos(theta);

    return normalize(dir * 0.5 + float3(0.5, 0.5, 0.5));
}

float3 FixedDirection(int sampleIndex)
{
    float3 directions[8] =
    {
        float3(1.0, 0.0, 0.0),
        float3(-1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, -1.0, 0.0),
        float3(0.707, 0.707, 0.0),
        float3(-0.707, 0.707, 0.0),
        float3(0.707, -0.707, 0.0),
        float3(-0.707, -0.707, 0.0)
    };
    return directions[sampleIndex % 8];
}

float4 SSAO(float2 texcoord)
{
    float4 originalColor = tex2D(ColorSampler, texcoord);
    float depthValue = tex2D(DepthSampler, texcoord).r;
    float occlusion = 0.0;

    int sampleCount;
    if (qualityLevel == 0)
        sampleCount = 8;
    else if (qualityLevel == 1)
        sampleCount = 16;
    else
        sampleCount = 32;

    float radius = 0.005;
    float falloff = 0.01;

    // Loop over the samples to compute occlusion
    for (int i = 0; i < sampleCount; i++)
    {
        float3 sampleDir;

        if (aoType == 0)
            sampleDir = FixedDirection(i);
        else if (aoType == 1)
            sampleDir = RandomDirection(texcoord, i);

        float2 sampleCoord = texcoord + sampleDir.xy * radius;
        sampleCoord = clamp(sampleCoord, 0.0, 1.0); 

        float sampleDepth = tex2D(DepthSampler, sampleCoord).r;
        float rangeCheck = exp(-abs(depthValue - sampleDepth) / falloff); 

        occlusion += (sampleDepth < depthValue) ? rangeCheck : 0.0;
    }

    // Normalize occlusion
    occlusion = (occlusion / sampleCount) * intensity;

    // Selected view mode
    if (viewMode == 0)
    {
        float4 finalColor = originalColor * (1.0 - saturate(occlusion)); 
        return finalColor;
    }

    return originalColor;
}

float4 SSAOPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 ssaoColor = SSAO(texcoord);
    return ssaoColor;
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique SSAO
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = SSAOPS;
    }
}

/*-------------.
| :: Footer :: |
'--------------/

https://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
https://github.com/d3dcoder/d3d12book/blob/master/Chapter%2021%20Ambient%20Occlusion/Ssao/Shaders/Ssao.hlsl
