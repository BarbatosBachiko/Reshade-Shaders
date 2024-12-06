/*------------------.
| :: Description :: |
'-------------------/

Directionally Localized Anti Aliasing Lite (version 1.6)

	Author: Barbatos Bachiko
	License: Creative Commons Attribution 3.0
        Original by: BlueSkyDefender. Thanks!

	About:
        This shader applies Directionally Localized Anti-Aliasing (DLAA) with
        Shading effect and Sharpness.
    
	Ideas for future improvement:
	Implement Additional Anti-Aliasing

	History:
	(*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
	
	Version 1.6
	+ Improved Shading
*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int View_Mode <
	ui_type = "combo";
	ui_items = "DLAA Out\0Mask View A\0Mask View B\0";
	ui_label = "View Mode";
	ui_tooltip = "This is used to select the normal view output or debug view.";
> = 0;

uniform float LongEdgeSampleSize < 
    ui_type = "slider";
    ui_label = "Long Edge Sample Size";
    ui_tooltip = "Adjust the sample size for long edge detection."; 
    ui_min = 2.0; 
    ui_max = 12.0; 
    ui_default = 8.0; 
> = 8.0;

uniform float sharpness < 
    ui_type = "slider";
    ui_label = "Sharpness";
    ui_tooltip = "Control the sharpness level."; 
> = 0.2;

uniform float ShadingIntensity < 
    ui_type = "slider";
    ui_label = "Shading"; 
    ui_tooltip = "Control the Shading."; 
    ui_min = 0.0; 
    ui_max = 4.0; 
    ui_default = 2.0; 
> = 2.0;

uniform bool EnableShading < 
    ui_type = "checkbox";
    ui_label = "Enable Shading"; 
    ui_tooltip = "Enable or disable the shading effect."; 
> = true; 

uniform int PixelWidth < 
    ui_type = "combo";
    ui_items = "1\0 16\0 32\0 64\0 128\0"; 
    ui_label = "Pixel Width"; 
    ui_tooltip = "Select the pixel width."; 
> = 1; 

/*---------------.
| :: Textures :: |
'---------------*/

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define lambda 3.0f
#define epsilon 0.1f

texture BackBufferTex : COLOR;

sampler BackBuffer
{
    Texture = BackBufferTex;
};
texture SLPtex
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
																				
sampler SamplerLoadedPixel
{
    Texture = SLPtex;
};

/*----------------.
| :: Functions :: |
'----------------*/

float LI(in float3 value)
{
    return dot(value.ggg, float3(0.333, 0.333, 0.333));
}

float4 LP(float2 tc, float dx, float dy) 
{
    return tex2D(BackBuffer, tc + float2(dx, dy) * pix.xy);
}

float4 PreFilter(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target 
{

    const float4 center = LP(texcoord, 0, 0);
    const float4 left = LP(texcoord, -1, 0);
    const float4 right = LP(texcoord, 1, 0);
    const float4 top = LP(texcoord, 0, -1);
    const float4 bottom = LP(texcoord, 0, 1);

    const float4 edges = 4.0 * abs((left + right + top + bottom) - 4.0 * center);
    const float edgesLum = LI(edges.rgb);

    return float4(center.rgb, edgesLum);
}

float4 SLP(float2 tc, float dx, float dy) 
{
    return tex2D(SamplerLoadedPixel, tc + float2(dx, dy) * pix.xy);
}

// Load pixel from texture
float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

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

float4 ApplyShading(float4 color, float2 texcoord)
{
    if (!EnableShading)
    {
        return color;
    }

    // Load neighboring pixels
    float4 left = LoadPixel(BackBuffer, texcoord + float2(-BUFFER_RCP_WIDTH, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(BUFFER_RCP_WIDTH, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -BUFFER_RCP_HEIGHT));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, BUFFER_RCP_HEIGHT));

    // Calculate differences
    float3 diffLeft = abs(left.rgb - color.rgb);
    float3 diffRight = abs(right.rgb - color.rgb);
    float3 diffTop = abs(top.rgb - color.rgb);
    float3 diffBottom = abs(bottom.rgb - color.rgb);

    // Aggregate edge intensity
    float3 weightVector = float3(0.333, 0.333, 0.333); 
    float edgeIntensity = (dot(diffLeft, weightVector) +
                           dot(diffRight, weightVector) +
                           dot(diffTop, weightVector) +
                           dot(diffBottom, weightVector)) / 4.0;

    float weight = smoothstep(0.0, 1.0, edgeIntensity * ShadingIntensity);

    float4 dimmedColor = color * (1.0 - ShadingIntensity);
    float4 shadedColor = lerp(color, dimmedColor, weight);

    return shadedColor;
}

// Pixel width
float4 ApplyPixelWidth(float4 color, float2 texcoord)
{
    if (!PixelWidth)
    {
        return color;
    }

    float2 pixelOffset = float2(float(PixelWidth), float(PixelWidth)) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    float4 left = LoadPixel(BackBuffer, texcoord + float2(-pixelOffset.x, 0));
    float4 right = LoadPixel(BackBuffer, texcoord + float2(pixelOffset.x, 0));
    float4 top = LoadPixel(BackBuffer, texcoord + float2(0, -pixelOffset.y));
    float4 bottom = LoadPixel(BackBuffer, texcoord + float2(0, pixelOffset.y));

    float4 laplacian = (left + right + top + bottom - 4.0 * color) * 0.2;

    float edgeIntensity = length(laplacian.rgb);

    // Dynamic complexity factor based on scene metrics
    float complexityFactor = 1.0 + edgeIntensity * 0.5;
    float enhancementFactor = saturate(edgeIntensity * 3.0 * complexityFactor);

    float4 enhancedColor = color + laplacian * enhancementFactor;
    enhancedColor = clamp(enhancedColor, 0.0, 1.0);

    return enhancedColor;
}

//Information on Slide 44 says to run the edge processing jointly short and Large.
float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
	//Short Edge Filter 
    float4 DLAA, DLAA_S, DLAA_L; //DLAA is the completed AA Result.
	
	//5 bi-linear samples cross
    const float4 Center = SLP(texcoord, 0, 0);
    const float4 Left = SLP(texcoord, -1.0, 0);
    const float4 Right = SLP(texcoord, 1.0, 0);
    const float4 Up = SLP(texcoord, 0, -1.0);
    const float4 Down = SLP(texcoord, 0, 1.);

	
	//Combine horizontal and vertical blurs together
    const float4 combH = 2.0 * (Left + Right);
    const float4 combV = 2.0 * (Up + Down);
	
	//Bi-directional anti-aliasing using HORIZONTAL & VERTICAL blur and horizontal edge detection
	//Slide information triped me up here. Read slide 43.
	//Edge detection
    const float4 CenterDiffH = abs(combH - 4.0 * Center) / 4.0;
    const float4 CenterDiffV = abs(combV - 4.0 * Center) / 4.0;

	//Blur
    const float4 blurredH = (combH + 2.0 * Center) / 6.0;
    const float4 blurredV = (combV + 2.0 * Center) / 6.0;
	
	//Edge detection
    const float LumH = LI(CenterDiffH.rgb);
    const float LumV = LI(CenterDiffV.rgb);
	
    const float LumHB = LI(blurredH.xyz);
    const float LumVB = LI(blurredV.xyz);
    
	//t
    const float satAmountH = saturate((lambda * LumH - epsilon) / LumVB);
    const float satAmountV = saturate((lambda * LumV - epsilon) / LumHB);
	
	//color = lerp(color,blur,sat(Edge/blur)
	//Re-blend Short Edge Done
    DLAA = lerp(Center, blurredH, satAmountV);
    DLAA = lerp(DLAA, blurredV, satAmountH * 0.5f);
   	
    float4 HNeg, HNegA, HNegB, HNegC, HNegD, HNegE,
			HPos, HPosA, HPosB, HPosC, HPosD, HPosE,
			VNeg, VNegA, VNegB, VNegC,
			VPos, VPosA, VPosB, VPosC;
			
	// Long Edges 
    //16 bi-linear samples cross, added extra bi-linear samples in each direction.
    HNeg = Left;
    HNegA = SLP(texcoord, -3.5, 0.0);
    HNegB = SLP(texcoord, -5.5, 0.0);
    HNegC = SLP(texcoord, -7.5, 0.0);
	
    HPos = Right;
    HPosA = SLP(texcoord, 3.5, 0.0);
    HPosB = SLP(texcoord, 5.5, 0.0);
    HPosC = SLP(texcoord, 7.5, 0.0);
	
    VNeg = Up;
    VNegA = SLP(texcoord, 0.0, -3.5);
    VNegB = SLP(texcoord, 0.0, -5.5);
    VNegC = SLP(texcoord, 0.0, -7.5);
	
    VPos = Down;
    VPosA = SLP(texcoord, 0.0, 3.5);
    VPosB = SLP(texcoord, 0.0, 5.5);
    VPosC = SLP(texcoord, 0.0, 7.5);
	
    //Long Edge detection H & V
    const float4 AvgBlurH = (HNeg + HNegA + HNegB + HNegC + HPos + HPosA + HPosB + HPosC) / LongEdgeSampleSize;
    const float4 AvgBlurV = (VNeg + VNegA + VNegB + VNegC + VPos + VPosA + VPosB + VPosC) / LongEdgeSampleSize;
    const float EAH = saturate(AvgBlurH.a * 2.0 - 1.0);
    const float EAV = saturate(AvgBlurV.a * 2.0 - 1.0);
        
    const float longEdge = abs(EAH - EAV) + abs(LumH + LumV);
    const float Mask = longEdge > 0.2;
	//Used to Protect Text
    if (Mask)
    {
        const float4 left = LP(texcoord, -1, 0);
        const float4 right = LP(texcoord, 1, 0);
        const float4 up = LP(texcoord, 0, -1);
        const float4 down = LP(texcoord, 0, 1);
            
	//Merge for BlurSamples.
	//Long Blur H
        const float LongBlurLumH = LI(AvgBlurH.rgb);
    //Long Blur V
        const float LongBlurLumV = LI(AvgBlurV.rgb);
	
        const float centerLI = LI(Center.rgb);
        const float leftLI = LI(left.rgb);
        const float rightLI = LI(right.rgb);
        const float upLI = LI(up.rgb);
        const float downLI = LI(down.rgb);
  
        const float blurUp = saturate(0.0 + (LongBlurLumH - upLI) / (centerLI - upLI));
        const float blurLeft = saturate(0.0 + (LongBlurLumV - leftLI) / (centerLI - leftLI));
        const float blurDown = saturate(1.0 + (LongBlurLumH - centerLI) / (centerLI - downLI));
        const float blurRight = saturate(1.0 + (LongBlurLumV - centerLI) / (centerLI - rightLI));

        float4 UDLR = float4(blurLeft, blurRight, blurUp, blurDown);

        if (UDLR.r == 0.0 && UDLR.g == 0.0 && UDLR.b == 0.0 && UDLR.a == 0.0)
            UDLR = float4(1.0, 1.0, 1.0, 1.0);
	
        float4 V = lerp(left, Center, UDLR.x);
        V = lerp(right, V, UDLR.y);
		       
        float4 H = lerp(up, Center, UDLR.z);
        H = lerp(down, H, UDLR.w);
	
	//Reuse short samples and DLAA Long Edge Out.
        DLAA = lerp(DLAA, V, EAV);
        DLAA = lerp(DLAA, H, EAH);
    }

    if (View_Mode == 1)
    {
        DLAA = Mask * 2;
    }
    else if (View_Mode == 2)
    {
        DLAA = lerp(DLAA, float4(1, 1, 0, 1), Mask * 2);
    }
    
    if (PixelWidth > 0.0)
    {
        DLAA = ApplyPixelWidth(DLAA, texcoord);
    }
    
    if (sharpness > 0.0)
    {
        DLAA = ApplySharpness(DLAA, texcoord);
    }

    if (EnableShading > 0.0)
    {
        DLAA = ApplyShading(DLAA, texcoord);
    }
    
    return DLAA;
}

/*-----------------.
| :: Reshade.fxh:: |
'-----------------*/

// Vertex shader generating a triangle covering the entire screen
void DPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    if (id == 2)
        texcoord.x = 2.0;
    else
        texcoord.x = 0.0;
    if (id == 1)
        texcoord.y = 2.0;
    else
        texcoord.y = 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DLAA_Lite
{
    pass Pre_Filter
    {
        VertexShader = DPostProcessVS;
        PixelShader = PreFilter;
        RenderTarget = SLPtex;
    }
    pass DLAA_Light
    {
        VertexShader = DPostProcessVS;
        PixelShader = Out;
    }
}

/*-------------.
| :: Footer :: |
'--------------/

GShade DLAA https://github.com/Mortalitas/GShade/blob/master/Shaders/DLAA.fx
Short Edge Filter http://and.intercon.ru/releases/talks/dlaagdc2011/slides/#slide43
License https://creativecommons.org/licenses/by/3.0/us/
