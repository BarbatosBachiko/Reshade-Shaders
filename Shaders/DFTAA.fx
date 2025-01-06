/*------------------.
| :: Description :: |
'-------------------/

	DFTAA (version 1.2.1)

	Author: Barbatos Bachiko
        License: MIT

	About:
	Implementation of Directionally Fast Temporal Anti-Aliasing (DFTAA) that combines DLAA, FSMAA and Simple TAA.

	Ideas for future improvement:
	* Integrate an adjustable sharpness
	* Improve TAA

        History:
        (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

	Version 1.2.1
	+ Code org

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

// SMAA 
#if !defined(SMAA_PRESET_LOW) && !defined(SMAA_PRESET_MEDIUM) && !defined(SMAA_PRESET_HIGH) && !defined(SMAA_PRESET_ULTRA)
#define SMAA_PRESET_CUSTOM 
#define SMAA_DISABLE_DIAG_DETECTION
#define SMAA_DISABLE_DEPTH_BUFFER
#endif

/*---------------.
| :: Uniforms :: |
'---------------*/

uniform int View_Mode < 
    ui_type = "combo";
    ui_items = "DFTAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "Select normal output or debug view."; 
> = 0; 

uniform float EdgeThreshold < 
    ui_category = "DLAA";
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Adjust the edge threshold for mask creation."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.020; 
> = 0.020; 

uniform float Lambda < 
    ui_category = "DLAA";
    ui_type = "slider";
    ui_label = "Lambda"; 
    ui_tooltip = "Adjust the lambda for saturation amount."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 3.0; 
> = 3.0;

#ifndef SMAA_DISABLE_DEPTH_BUFFER
uniform int EdgeDetectionType < __UNIFORM_COMBO_INT1
	ui_items = "Luminance edge detection\0Color edge detection\0Depth edge detection\0";
	ui_label = "Edge Detection Type";
> = 1;
#else
uniform int EdgeDetectionType < __UNIFORM_COMBO_INT1
    ui_category = "SMAA";
	ui_items = "Luminance edge detection\0Color edge detection\0";
	ui_label = "Edge Detection Type";
> = 1;
#endif

#ifdef SMAA_PRESET_CUSTOM
uniform float EdgeDetectionThreshold < __UNIFORM_DRAG_FLOAT1
    ui_category = "SMAA";
	ui_min = 0.05; ui_max = 0.20; ui_step = 0.001;
	ui_tooltip = "Edge detection threshold. If SMAA misses some edges try lowering this slightly.";
	ui_label = "Edge Detection Threshold";
> = 0.10;

#ifndef SMAA_DISABLE_DEPTH_BUFFER
uniform float DepthEdgeDetectionThreshold < __UNIFORM_DRAG_FLOAT1
	ui_min = 0.001; ui_max = 0.10; ui_step = 0.001;
	ui_tooltip = "Depth Edge detection threshold. If SMAA misses some edges try lowering this slightly.";
	ui_label = "Depth Edge Detection Threshold";
> = 0.01;
#endif

uniform int MaxSearchSteps < __UNIFORM_SLIDER_INT1
    ui_category = "SMAA";
	ui_min = 0; ui_max = 112;
	ui_label = "Max Search Steps";
	ui_tooltip = "Determines the radius SMAA will search for aliased edges.";
> = 32;

#ifndef SMAA_DISABLE_DIAG_DETECTION
uniform int MaxSearchStepsDiagonal < __UNIFORM_SLIDER_INT1
	ui_min = 0; ui_max = 20;
	ui_label = "Max Search Steps Diagonal";
	ui_tooltip = "Determines the radius SMAA will search for diagonal aliased edges";
> = 16;
#endif

#ifndef SMAA_DISABLE_CORNER_DETECTION
uniform int CornerRounding < __UNIFORM_SLIDER_INT1
    ui_category = "SMAA";
	ui_min = 0; ui_max = 100;
	ui_label = "Corner Rounding";
	ui_tooltip = "Determines the percent of anti-aliasing to apply to corners.";
> = 25;
#endif

#ifndef SMAA_DISABLE_DEPTH_BUFFER
uniform bool PredicationEnabled < __UNIFORM_INPUT_BOOL1
	ui_label = "Enable Predicated Thresholding";
> = false;

uniform float PredicationThreshold < __UNIFORM_DRAG_FLOAT1
	ui_min = 0.005; ui_max = 1.00; ui_step = 0.01;
	ui_tooltip = "Threshold to be used in the additional predication buffer.";
	ui_label = "Predication Threshold";
> = 0.01;

uniform float PredicationScale < __UNIFORM_SLIDER_FLOAT1
	ui_min = 1; ui_max = 8;
	ui_tooltip = "How much to scale the global threshold used for luma or color edge.";
	ui_label = "Predication Scale";
> = 2.0;

uniform float PredicationStrength < __UNIFORM_SLIDER_FLOAT1
	ui_min = 0; ui_max = 10;
	ui_tooltip = "How much to locally decrease the threshold.";
	ui_label = "Predication Strength";
> = 0.4;
#endif
#endif

uniform int DebugOutput < __UNIFORM_COMBO_INT1
    ui_category = "SMAA";
	ui_items = "None\0View edges\0View weights\0";
	ui_label = "Debug Output";
> = false;

#ifdef SMAA_PRESET_CUSTOM
#define SMAA_THRESHOLD EdgeDetectionThreshold
#define SMAA_MAX_SEARCH_STEPS MaxSearchSteps
#ifndef SMAA_DISABLE_CORNER_DETECTION
#define SMAA_CORNER_ROUNDING CornerRounding
#endif
#ifndef SMAA_DISABLE_DIAG_DETECTION
#define SMAA_MAX_SEARCH_STEPS_DIAG MaxSearchStepsDiagonal
#endif
#ifndef SMAA_DISABLE_DEPTH_BUFFER
#define SMAA_DEPTH_THRESHOLD DepthEdgeDetectionThreshold
#define SMAA_PREDICATION PredicationEnabled
#define SMAA_PREDICATION_THRESHOLD PredicationThreshold
#define SMAA_PREDICATION_SCALE PredicationScale
#define SMAA_PREDICATION_STRENGTH PredicationStrength
#else
#define SMAA_PREDICATION false
#endif
#endif

#define SMAA_RT_METRICS float4(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, BUFFER_WIDTH, BUFFER_HEIGHT)
#define SMAA_CUSTOM_SL 1

#define SMAATexture2D(tex) sampler tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex2Dlod(tex, float4(coord, coord))
#define SMAASampleLevelZeroPoint(tex, coord) SMAASampleLevelZero(tex, coord)
#define SMAASampleLevelZeroOffset(tex, coord, offset) tex2Dlodoffset(tex, float4(coord, coord), offset)
#define SMAASample(tex, coord) tex2D(tex, coord)
#define SMAASamplePoint(tex, coord) SMAASample(tex, coord)
#define SMAASampleOffset(tex, coord, offset) tex2Doffset(tex, coord, offset)
#define SMAA_BRANCH [branch]
#define SMAA_FLATTEN [flatten]

#if (__RENDERER__ == 0xb000 || __RENDERER__ == 0xb100)
#define SMAAGather(tex, coord) tex2Dgather(tex, coord, 0)
#endif

#include "SMAA.fxh"

/*---------------.
| :: Textures :: |
'---------------*/

#ifndef SMAA_DISABLE_DEPTH_BUFFER
texture depthTex < pooled = true; >
{ 
	Width = BUFFER_WIDTH;   
	Height = BUFFER_HEIGHT;   
	Format = R16F;  
};
#endif

texture BackBufferTex : COLOR;
texture PreviousFrameTex : COLOR;

texture edgesTex < pooled = true; >
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG8;
};
texture blendTex < pooled = true; >
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};

texture areaTex < source = "AreaTex.png"; >
{
    Width = 160;
    Height = 560;
    Format = RG8;
};
texture searchTex < source = "SearchTex.png"; >
{
    Width = 64;
    Height = 16;
    Format = R8;
};

/*---------------.
| :: Samplers :: |
'---------------*/

#ifndef SMAA_DISABLE_DEPTH_BUFFER
sampler depthLinearSampler
{
	Texture = depthTex;
};
#endif

sampler BackBuffer
{
    Texture = BackBufferTex;
};

sampler texPrevious
{
    Texture = PreviousFrameTex;
};

sampler colorGammaSampler
{
    Texture = ReShade::BackBufferTex;
    AddressU = Clamp;
    AddressV = Clamp;
    MipFilter = Point;
    MinFilter = Linear;
    MagFilter = Linear;
    SRGBTexture = false;
};
sampler colorLinearSampler
{
    Texture = ReShade::BackBufferTex;
    AddressU = Clamp;
    AddressV = Clamp;
    MipFilter = Point;
    MinFilter = Linear;
    MagFilter = Linear;
    SRGBTexture = true;
};
sampler edgesSampler
{
    Texture = edgesTex;
    AddressU = Clamp;
    AddressV = Clamp;
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
    SRGBTexture = false;
};
sampler blendSampler
{
    Texture = blendTex;
    AddressU = Clamp;
    AddressV = Clamp;
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
    SRGBTexture = false;
};
sampler areaSampler
{
    Texture = areaTex;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
    SRGBTexture = false;
};
sampler searchSampler
{
    Texture = searchTex;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
    MipFilter = Point;
    MinFilter = Point;
    MagFilter = Point;
    SRGBTexture = false;
};

/*----------------.
| :: Functions :: |
'----------------*/

float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

float4 ApplyDFTAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);

    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    const float satAmountH = saturate((Lambda * LumH - 0.1f) / LumH);

    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

float4 BlurMask(float4 mask, float2 tc)
{
    float4 blur = mask * 0.25;
    blur += LoadPixel(BackBuffer, tc + float2(-1.0 * BUFFER_RCP_WIDTH, 0)) * 0.125; // Left
    blur += LoadPixel(BackBuffer, tc + float2(1.0 * BUFFER_RCP_WIDTH, 0)) * 0.125; // Right
    blur += LoadPixel(BackBuffer, tc + float2(0, -BUFFER_RCP_HEIGHT)) * 0.125; // Up
    blur += LoadPixel(BackBuffer, tc + float2(0, BUFFER_RCP_HEIGHT)) * 0.125; // Down
    return saturate(blur);
}

/*-----------------.
| ::   Passes  ::  |
'-----------------*/

// Main DFTAA pass
float4 DFTAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float4 center = LoadPixel(BackBuffer, texcoord);
    const float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * BUFFER_RCP_WIDTH, 0));
    const float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * BUFFER_RCP_WIDTH, 0));

    float4 DFTAA = ApplyDFTAA(center, left, right);

    // Mask calculation for DFTAA
    const float4 maskColorA = float4(1.0, 1.0, 0.0, 1.0);

    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    
    const float maskA = LumH > EdgeThreshold ? 1.0 : 0.0;

    // Mask View A
    if (View_Mode == 1)
    {
        return BlurMask(maskA * maskColorA, texcoord);
    }

    // Mask View B
    const float4 diff = abs(DFTAA - center);
    const float maxDiff = max(max(diff.r, diff.g), diff.b);
    const float maskB = maxDiff > EdgeThreshold ? 1.0 : 0.0;

    if (View_Mode == 2)
    {
        return BlurMask(maskB * float4(1.0, 0.0, 0.0, 1.0), texcoord);
    }

    return DFTAA;
}

// SMAALumaEdgeDetection
float4 SMAALumaEdgeDetectionPS(float2 texcoord)
{
    float4 pixel = LoadPixel(BackBuffer, texcoord);
    float luminance = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
    return float4(luminance, luminance, luminance, 1.0);
}

// FSMAA Edge Detection
float4 FSMAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 edgeDetection = SMAALumaEdgeDetectionPS(texcoord);
    
    return DFTAAPass(position, texcoord);
}

// Main TAA Pass
float4 TAAPass(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    static float2 JitterPattern[4] =
    {
        float2(0.0, 0.0),
        float2(0.25, 0.25),
        float2(0.75, 0.25),
        float2(0.25, 0.75)
    };
    
    static int frameIndex = 0;
    int jitterIndex = frameIndex % 4;
    float2 jitter = JitterPattern[jitterIndex] / float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    float4 previousColor = LoadPixel(texPrevious, texcoord);
    float4 currentColor = LoadPixel(BackBuffer, texcoord + jitter);
    
    frameIndex++;

    return lerp(previousColor, currentColor, 0.5);
}

void SMAAEdgeDetectionWrapVS(
	in uint id : SV_VertexID,
	out float4 position : SV_Position,
	out float2 texcoord : TEXCOORD0,
	out float4 offset[3] : TEXCOORD1)
{
    PostProcessVS(id, position, texcoord);
    SMAAEdgeDetectionVS(texcoord, offset);
}
void SMAABlendingWeightCalculationWrapVS(
	in uint id : SV_VertexID,
	out float4 position : SV_Position,
	out float2 texcoord : TEXCOORD0,
	out float2 pixcoord : TEXCOORD1,
	out float4 offset[3] : TEXCOORD2)
{
    PostProcessVS(id, position, texcoord);
    SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
}
void SMAANeighborhoodBlendingWrapVS(
	in uint id : SV_VertexID,
	out float4 position : SV_Position,
	out float2 texcoord : TEXCOORD0,
	out float4 offset : TEXCOORD1)
{
    PostProcessVS(id, position, texcoord);
    SMAANeighborhoodBlendingVS(texcoord, offset);
}

#ifndef SMAA_DISABLE_DEPTH_BUFFER
float SMAADepthLinearizationPS(
	float4 position : SV_Position,
	float2 texcoord : TEXCOORD) : SV_Target
{
	return ReShade::GetLinearizedDepth(texcoord);
}
#endif

float2 SMAAEdgeDetectionWrapPS(
	float4 position : SV_Position,
	float2 texcoord : TEXCOORD0,
	float4 offset[3] : TEXCOORD1) : SV_Target
{
#ifndef SMAA_DISABLE_DEPTH_BUFFER
	if (EdgeDetectionType == 0 && SMAA_PREDICATION == true)
		return SMAALumaEdgePredicationDetectionPS(texcoord, offset, colorGammaSampler, depthLinearSampler);
	else 
#endif
    if (EdgeDetectionType == 0)
        return SMAALumaEdgeDetectionPS(texcoord, offset, colorGammaSampler);
#ifndef SMAA_DISABLE_DEPTH_BUFFER
	if (EdgeDetectionType == 2)
		return SMAADepthEdgeDetectionPS(texcoord, offset, depthLinearSampler);

	if (SMAA_PREDICATION)
		return SMAAColorEdgePredicationDetectionPS(texcoord, offset, colorGammaSampler, depthLinearSampler);
	else
#endif
    return SMAAColorEdgeDetectionPS(texcoord, offset, colorGammaSampler);
}
float4 SMAABlendingWeightCalculationWrapPS(
	float4 position : SV_Position,
	float2 texcoord : TEXCOORD0,
	float2 pixcoord : TEXCOORD1,
	float4 offset[3] : TEXCOORD2) : SV_Target
{
    return SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edgesSampler, areaSampler, searchSampler, 0.0);
}

float3 SMAANeighborhoodBlendingWrapPS(
	float4 position : SV_Position,
	float2 texcoord : TEXCOORD0,
	float4 offset : TEXCOORD1) : SV_Target
{
    if (DebugOutput == 1)
        return tex2D(edgesSampler, texcoord).rgb;
    if (DebugOutput == 2)
        return tex2D(blendSampler, texcoord).rgb;

    return SMAANeighborhoodBlendingPS(texcoord, offset, colorLinearSampler, blendSampler).rgb;
}

// Vertex shader generating a triangle covering the entire screen
void CustomPostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DFTAA
{
    pass DFTAA_Light
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = DFTAAPass;
    }

    pass FSMAA_Pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = FSMAAPass;
    }

    pass TAA_Pass
    {
        VertexShader = CustomPostProcessVS;
        PixelShader = TAAPass;
    }
#ifndef SMAA_DISABLE_DEPTH_BUFFER
    pass LinearizeDepthPass
    {
        VertexShader = PostProcessVS;
        PixelShader = SMAADepthLinearizationPS;
        RenderTarget = depthTex;
    }
#endif
    pass EdgeDetectionPass
    {
        VertexShader = SMAAEdgeDetectionWrapVS;
        PixelShader = SMAAEdgeDetectionWrapPS;
        RenderTarget = edgesTex;
        ClearRenderTargets = true;
        StencilEnable = true;
        StencilPass = REPLACE;
        StencilRef = 1;
    }
    pass BlendWeightCalculationPass
    {
        VertexShader = SMAABlendingWeightCalculationWrapVS;
        PixelShader = SMAABlendingWeightCalculationWrapPS;
        RenderTarget = blendTex;
        ClearRenderTargets = true;
        StencilEnable = true;
        StencilPass = KEEP;
        StencilFunc = EQUAL;
        StencilRef = 1;
    }
    pass NeighborhoodBlendingPass
    {
        VertexShader = SMAANeighborhoodBlendingWrapVS;
        PixelShader = SMAANeighborhoodBlendingWrapPS;
        StencilEnable = false;
        SRGBWriteEnable = true;
    }
}
