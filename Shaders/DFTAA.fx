/*------------------.
| :: Description :: |
'-------------------/

    DFTAA

    Version 1.3
    Author: Barbatos Bachiko
    Original DLAA by: BlueSkyDefender
    Original FSMAA by: lordbean-git

    License: Creative Commons Attribution 3.0

    About:
    Implementation of DLAA, FSMAA and Simple TAA.

    History:
   (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

   Version 1.3
   + Update DLAA
   + Update TAA
*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

// SMAA 
#if !defined(SMAA_PRESET_LOW) && !defined(SMAA_PRESET_MEDIUM) && !defined(SMAA_PRESET_HIGH) && !defined(SMAA_PRESET_ULTRA)
#define SMAA_PRESET_CUSTOM 
#define SMAA_DISABLE_DIAG_DETECTION
#define SMAA_DISABLE_DEPTH_BUFFER
#endif

uniform int View_Mode < 
    ui_type = "combo";
    ui_items = "DFTAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "Select normal output or debug view."; 
> = 0; 

uniform float LongEdgeSampleSize < 
    ui_category = "DLAA";
    ui_type = "slider";
    ui_label = "Long Edge Sample Size";
    ui_tooltip = "Adjust the sample size for long edge detection."; 
    ui_min = 2.0; 
    ui_max = 12.0; 
    ui_default = 5.0; 
> = 8.5;

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

uniform uint framecount < source = "framecount"; >;

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

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define lambda 3.0f
#define epsilon 0.1f

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

float LI(in float3 value)
{
    return dot(value.ggg, float3(0.333, 0.333, 0.333));
}

float4 LP(float2 tc, float dx, float dy)
{
    return tex2D(ReShade::BackBuffer, tc + float2(dx, dy) * pix.xy);
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

//Information on Slide 44 says to run the edge processing jointly short and Large.
float4 DLAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
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
    
    return DLAA;
}

// SMAALumaEdgeDetection
float4 SMAALumaEdgeDetectionPS(float2 texcoord)
{
    float4 pixel = LoadPixel(BackBuffer, texcoord);
    float luminance = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
    return float4(luminance, luminance, luminance, 1.0);
}

// FSMAA Edge Detection
float4 FSMAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 edgeDetection = SMAALumaEdgeDetectionPS(texcoord);
    
    return DLAA(position, texcoord);
}

//Simple TAA
float4 TAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    static const float2 JitterPattern[4] =
    {
        float2(0.25, 0.25) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT),
        float2(-0.25, -0.25) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT),
        float2(-0.25, 0.25) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT),
        float2(0.25, -0.25) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
    };

    // Get persistent frame counter from ReShade
    uint frameIndex = framecount % 4;
    
    float4 previousColor = tex2D(texPrevious, texcoord);
    float4 currentColor = tex2D(BackBuffer, texcoord + JitterPattern[frameIndex]);

    // Motion-aware blending with color clamping
    float3 velocity = abs(currentColor.rgb - previousColor.rgb);
    float motionFactor = saturate(length(velocity) * 5.0);
    float4 blended = lerp(previousColor, currentColor, 0.1 + motionFactor * 0.4);

    // Clip to neighborhood bounds
    float4 minColor = min(previousColor, currentColor);
    float4 maxColor = max(previousColor, currentColor);
    return clamp(blended, minColor, maxColor);
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

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique DFTAA
{
    pass Pre_Filter
    {
        VertexShader = PostProcessVS;
        PixelShader = PreFilter;
        RenderTarget = SLPtex;
    }
    pass 
    {
        VertexShader = PostProcessVS;
        PixelShader = DLAA;
    }

    pass FSMAA_Pass
    {
        VertexShader = PostProcessVS;
        PixelShader = FSMAA;
    }

    pass TAA_Pass
    {
        VertexShader = PostProcessVS;
        PixelShader = TAA;
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
