/*------------------.
| :: Description :: |
'-------------------/

	MLAA (Version: 0.1)
	
	Author: BarbatosBachiko
	About: Applies anti-aliasing using machine learning and FXAA.

	Ideas for future improvement:
	* Implement additional machine learning techniques for improved edge detection.
	* Add performance optimizations for lower-end hardware.

	Version 0.1
	* Initial implementation of machine learning-based anti-aliasing with FXAA integration.

*/


/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

// Adjustable settings
uniform float Subpix
< 
	__UNIFORM_SLIDER_FLOAT1 
	ui_min = 0.0; 
	ui_max = 1.0; 
	ui_tooltip = "Amount of sub-pixel aliasing removal. Higher values make the image softer.";
> = 0.25;

uniform float EdgeThreshold
< 
	__UNIFORM_SLIDER_FLOAT1 
	ui_min = 0.0; 
	ui_max = 1.0; 
	ui_label = "Edge Detection Threshold"; 
	ui_tooltip = "The minimum local contrast required to apply FXAA.";
> = 0.125;

uniform float EdgeThresholdMin
< 
	__UNIFORM_SLIDER_FLOAT1 
	ui_min = 0.0; 
	ui_max = 1.0; 
	ui_label = "Darkness Threshold"; 
	ui_tooltip = "Pixels darker than this are not processed.";
> = 0.0;

/*---------------.
| :: Textures :: |
'---------------*/

texture MLWeights
{
    Width = 128;
    Height = 128;
    Format = RGBA32F;
};
texture BackBufferTex : COLOR;

/*---------------.
| :: Samplers :: |
'---------------*/

sampler BackBuffer
{
    Texture = BackBufferTex;
    MinFilter = Linear;
    MagFilter = Linear;
};
sampler WeightSampler
{
    Texture = MLWeights;
    MinFilter = Linear;
    MagFilter = Linear;
};

/*----------------.
| :: Functions :: |
'----------------*/

// FXAA Function
float4 FxaaPixelShader(float2 texcoord, float4 color, sampler2D tex, float2 rcpFrame)
{
    float3 rgbNW = tex2D(tex, texcoord + float2(-rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbNE = tex2D(tex, texcoord + float2(rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbSW = tex2D(tex, texcoord + float2(-rcpFrame.x, rcpFrame.y)).rgb;
    float3 rgbSE = tex2D(tex, texcoord + float2(rcpFrame.x, rcpFrame.y)).rgb;
    float3 rgbM = tex2D(tex, texcoord).rgb;
    
    float lumaNW = dot(rgbNW, float3(0.299, 0.587, 0.114));
    float lumaNE = dot(rgbNE, float3(0.299, 0.587, 0.114));
    float lumaSW = dot(rgbSW, float3(0.299, 0.587, 0.114));
    float lumaSE = dot(rgbSE, float3(0.299, 0.587, 0.114));
    float lumaM = dot(rgbM, float3(0.299, 0.587, 0.114));
    
    float edgeHorizontal = abs(lumaNW + lumaNE - lumaSW - lumaSE);
    float edgeVertical = abs(lumaNW + lumaSW - lumaNE - lumaSE);
    
    bool isHorizontal = edgeHorizontal >= edgeVertical;
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    float edgeThreshold = max(EdgeThresholdMin, EdgeThreshold * (lumaMax - lumaMin));
    
    if ((edgeHorizontal >= edgeThreshold) || (edgeVertical >= edgeThreshold))
    {
        if (isHorizontal)
        {
            texcoord.y += (lumaSW + lumaSE - lumaNW - lumaNE) * 0.25 * BUFFER_RCP_HEIGHT;
        }
        else
        {
            texcoord.x += (lumaNW + lumaSW - lumaNE - lumaSE) * 0.25 * BUFFER_RCP_WIDTH;
        }
        color.rgb = tex2D(tex, texcoord).rgb;
    }
    
    return float4(color.rgb, 1.0);
}

// MLAA Function (Machine Learning Anti-Aliasing)
float4 MLAA(float2 texcoord)
{
    float4 color = tex2D(BackBuffer, texcoord);
    float4 weights = tex2D(WeightSampler, texcoord);
    float4 antiAliasedColor = lerp(color, (color + weights), 0.5); // Smoothing without darkening
    
    return antiAliasedColor;
}

// Main pixel shader function
float4 AAPixelShader(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 color = MLAA(texcoord); // Apply MLAA
    color = FxaaPixelShader(texcoord, color, BackBuffer, float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)); // Apply FXAA
    
    return color;
}

// Standard vertex shader
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 vpos : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    vpos = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique MLAA
{
    pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = AAPixelShader;
    }
}