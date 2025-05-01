/*------------------.
| :: Description :: |
'-------------------/

    Directional Anti-Aliasing (DAA)
    
    Version 1.5
    Author: Barbatos Bachiko
    License: MIT

    About: Directional Anti-Aliasing (DAA) is an edge-aware spatiotemporal anti-aliasing technique 
    that smooths edges by applying directional blurring based on local gradient detection.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.5
    * Added "Edge Mask Overlay" (red edge highlights over the original image)
    + Quality Mode now uses 3x3 sampling with AABB
    - Removed Sharpening
    + Code maintenance
*/

// Includes
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

// Motion vector configuration
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Utility macros
#define getColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLum(color) dot(color, float3(0.299, 0.587, 0.114))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter= POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

// version-number.fxh
#ifndef _VERSION_NUMBER_H
#define _VERSION_NUMBER_H

#define MAJOR_VERSION 1
#define MINOR_VERSION 5
#define PATCH_VERSION 0

#define BUILD_DOT_VERSION_(mav, miv, pav) #mav "." #miv "." #pav
#define BUILD_DOT_VERSION(mav, miv, pav) BUILD_DOT_VERSION_(mav, miv, pav)
#define DOT_VERSION_STR BUILD_DOT_VERSION(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

#define BUILD_UNDERSCORE_VERSION_(prefix, mav, miv, pav) prefix ## _ ## mav ## _ ## miv ## _ ## pav
#define BUILD_UNDERSCORE_VERSION(p, mav, miv, pav) BUILD_UNDERSCORE_VERSION_(p, mav, miv, pav)
#define APPEND_VERSION_SUFFIX(prefix) BUILD_UNDERSCORE_VERSION(prefix, MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

#endif  //  _VERSION_NUMBER_H

uniform int APPEND_VERSION_SUFFIX(version) <
    ui_text = "Version: "
DOT_VERSION_STR;
    ui_label = " ";
    ui_type = "radio";
>;

    /*-------------------.
    | :: Settings ::    |
    '-------------------*/

uniform int View_Mode
<
    ui_category = "Anti-Aliasing";
    ui_type = "combo";
    ui_items = "Output\0Edge Mask\0Edge Mask Overlay\0Gradient Direction\0Motion Vectors\0";
    ui_label = "View Mode";
> = 0;

uniform float DirectionalStrength
<
    ui_type = "slider";
    ui_label = "Strength";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.05;
    ui_category = "Anti-Aliasing";
> = 2.4;
    
uniform float PixelWidth
<
    ui_type = "slider";
    ui_label = "Pixel Width";
    ui_tooltip = "Pixel width for edge detection.";
    ui_min = 0.5; ui_max = 4.0; ui_step = 0.1;
    ui_category = "Edge Detection";
> = 1.0;

uniform float EdgeThreshold
<
    ui_type = "slider";
    ui_label = "Edge Threshold";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Edge Detection";
> = 2.0;

uniform float EdgeFalloff
<
    ui_type = "slider";
    ui_label = "Edge Falloff";
    ui_tooltip = "Smooth falloff range for edge detection.";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Edge Detection";
> = 2.0;

uniform bool EnableTemporalAA
<
    ui_category = "Temporal";
    ui_type = "checkbox";
    ui_label = "Temporal";
> = false;

uniform int TemporalMode
<
    ui_category = "Temporal";
    ui_type = "combo";
    ui_items = "Blurry\0Standard\0Quality\0";
    ui_label = "TemporalMode";
    ui_tooltip = "";
> = 2;

uniform float TemporalAAFactor
<
    ui_category = "Temporal";
    ui_type = "slider";
    ui_label = "Temporal Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform uint framecount < source = "framecount"; >;

    /*---------------.
    | :: Textures :: |
    '---------------*/

texture2D DAATemporal
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};

texture2D DAAHistory
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};

sampler2D sDAATemporal
{
    Texture = DAATemporal;
};

sampler2D sDAAHistory
{
    Texture = DAAHistory;
};

#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex;  };
}
#elif USE_VORT_MOTION
    texture2D MotVectTexVort {  Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC  };
#else
texture texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler sTexMotionVectorsSampler
{
    Texture = texMotionVectors;
    S_PC
};
#endif

    /*----------------.
    | :: Functions :: |
    '----------------*/

// Calculates the gradient using the Scharr operator
float2 ComputeGradient(float2 texcoord)
{
    const float2 offset = ReShade::PixelSize.xy * PixelWidth;

    float lumTL = GetLum(getColor(float4(texcoord + float2(-offset.x, -offset.y), 0, 0)).rgb);
    float lumT = GetLum(getColor(float4(texcoord + float2(0, -offset.y), 0, 0)).rgb);
    float lumTR = GetLum(getColor(float4(texcoord + float2(offset.x, -offset.y), 0, 0)).rgb);
    float lumL = GetLum(getColor(float4(texcoord + float2(-offset.x, 0), 0, 0)).rgb);
    float lumR = GetLum(getColor(float4(texcoord + float2(offset.x, 0), 0, 0)).rgb);
    float lumBL = GetLum(getColor(float4(texcoord + float2(-offset.x, offset.y), 0, 0)).rgb);
    float lumB = GetLum(getColor(float4(texcoord + float2(0, offset.y), 0, 0)).rgb);
    float lumBR = GetLum(getColor(float4(texcoord + float2(offset.x, offset.y), 0, 0)).rgb);

    float gx = (-3.0 * lumTL - 10.0 * lumL - 3.0 * lumBL) + (3.0 * lumTR + 10.0 * lumR + 3.0 * lumBR);
    float gy = (-3.0 * lumTL - 10.0 * lumT - 3.0 * lumTR) + (3.0 * lumBL + 10.0 * lumB + 3.0 * lumBR);

    return float2(gx, gy);
}

float4 DirectionalAA(float2 texcoord)
{
    float4 originalColor = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0));
    float2 gradient = ComputeGradient(texcoord);
    float edgeStrength = length(gradient);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + EdgeFalloff, edgeStrength);

    if (weight > 0.01)
    {
        float2 blurDir = normalize(float2(-gradient.y, gradient.x));
        float2 blurOffset = blurDir * ReShade::PixelSize.xy * PixelWidth * DirectionalStrength;

        float4 color1 = getColor(float4(texcoord + blurOffset * 0.5, 0, 0));
        float4 color2 = getColor(float4(texcoord - blurOffset * 0.5, 0, 0));
        float4 color3 = getColor(float4(texcoord + blurOffset, 0, 0));
        float4 color4 = getColor(float4(texcoord - blurOffset, 0, 0));
        
        float4 smoothedColor = (color1 + color2) * 0.4 + (color3 + color4) * 0.1;
        return lerp(originalColor, smoothedColor, weight);
    }

    return originalColor;
}

float2 GetMotionVector(float2 texcoord)
{
#if USE_MARTY_LAUNCHPAD_MOTION
    return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
    return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#else
    return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).xy;
#endif
}

float3 HSVtoRGB(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

 // Convert RGB to YCoCg
float3 RGBToYCoCg(float3 rgb)
{
    float Y = dot(rgb, float3(0.25, 0.5, 0.25));
    float Co = dot(rgb, float3(0.5, 0.0, -0.5));
    float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
    return float3(Y, Co, Cg);
}

// Convert YCoCg back to RGB
float3 YCoCgToRGB(float3 ycocg)
{
    float Y = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

//Temporal DAA
float4 PS_TemporalDAA(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 motion = GetMotionVector(texcoord);
    float4 current = DirectionalAA(texcoord);
    
    if (TemporalMode == 0) // Method A - Basic neighborhood averaging = Blurry
    {
        float2 reprojectedTexcoord = texcoord + motion;
        
        // Define offsets for directions
        float2 offsetUp = float2(0.0, -ReShade::PixelSize.y);
        float2 offsetDown = float2(0.0, ReShade::PixelSize.y);
        float2 offsetLeft = float2(-ReShade::PixelSize.x, 0.0);
        float2 offsetRight = float2(ReShade::PixelSize.x, 0.0);
        
        // Samples the history
        float4 historyCenter = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord, 0, 0));
        float4 historyUp = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord + offsetUp, 0, 0));
        float4 historyDown = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord + offsetDown, 0, 0));
        float4 historyLeft = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord + offsetLeft, 0, 0));
        float4 historyRight = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord + offsetRight, 0, 0));
        
        float4 historyAvg = (historyCenter + historyUp + historyDown + historyLeft + historyRight) / 5.0;
        
        float factor = EnableTemporalAA ? TemporalAAFactor : 0.0;
        return lerp(current, historyAvg, factor);
    }
    else if (TemporalMode == 1) // Method B - Simple YCoCg blending 
    {
        float3 currentYCoCg = RGBToYCoCg(current.rgb);
        float2 reprojectedTexcoord = texcoord + motion;
        float4 history = tex2Dlod(sDAAHistory, float4(reprojectedTexcoord, 0, 0));
        float3 historyYCoCg = RGBToYCoCg(history.rgb);
        
        float factor = EnableTemporalAA ? TemporalAAFactor : 0.0;
        float3 blendedYCoCg = lerp(currentYCoCg, historyYCoCg, factor);
        float3 finalRGB = YCoCgToRGB(blendedYCoCg);
        
        return float4(finalRGB, current.a);
    }
    else // Method C – Neighborhood clamping = Quality
    {
        float2 pixelSize = ReShade::PixelSize.xy;
        float3 currentYCoCg = RGBToYCoCg(current.rgb);
        float2 reproTexcoord = texcoord + motion;
    
        static const float FLT_MAX = 3.40282347e+38;
        static const float NEG_FLT_MAX = -3.40282347e+38;

        // Initialize AABB
        float3 minColor = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxColor = float3(NEG_FLT_MAX, NEG_FLT_MAX, NEG_FLT_MAX);
    
        // Sample 3×3 neighborhood, skip (i=0,j=0)
        for (int j = -1; j <= 1; ++j)
        {
            for (int i = -1; i <= 1; ++i)
            {
                if (i != 0 || j != 0)
                {
                    float2 offset = pixelSize * float2(i, j);
                    float3 sampleYCoCg = RGBToYCoCg(
                    getColor(float4(texcoord + offset, 0, 0)).rgb
                );
                    minColor = min(minColor, sampleYCoCg);
                    maxColor = max(maxColor, sampleYCoCg);
                }
            }
        }
    
        // Expand AABB by 50%
        float3 halfSize = (maxColor - minColor) * 0.5;
        minColor -= halfSize;
        maxColor += halfSize;
    
        // Sample history and clamp it
        float3 historyYCoCg = RGBToYCoCg(
        tex2Dlod(sDAAHistory, float4(reproTexcoord, 0, 0)).rgb
    );
        historyYCoCg = clamp(historyYCoCg, minColor, maxColor);
    
       // Compute blend factor based on motion length
        float motionLen = length(motion) * 100.0;
        float blendFactor = EnableTemporalAA
        ? clamp(TemporalAAFactor * (1.0 - motionLen), 0.0, TemporalAAFactor)
        : 0.0;
    
        float3 resultYCoCg = lerp(currentYCoCg, historyYCoCg, blendFactor);
        float3 resultRGB = YCoCgToRGB(resultYCoCg);
        return float4(resultRGB, current.a);
    }
}

// History pass
float4 PS_SaveHistoryDAA(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 temporalResult = tex2Dlod(sDAATemporal, float4(texcoord, 0, 0));
    return temporalResult;
}

// Composite pass
float4 PS_CompositeDAA(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 originalColor = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0));
    float4 daaColor = tex2Dlod(sDAATemporal, float4(texcoord, 0, 0));
    
    float2 gradient = ComputeGradient(texcoord);
    float edgeStrength = length(gradient);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + EdgeFalloff, edgeStrength);
    
    float2 motion = GetMotionVector(texcoord);

    switch (View_Mode)
    {
        case 0: // Output
            return daaColor;
            
        case 1: // Edge Mask
            return float4(weight.xxx, 1.0);
            
        case 2: // Edge Mask Overlay
            float3 overlay = originalColor.rgb;
            overlay = lerp(overlay, float3(1.0, 0.0, 0.0), weight * 0.8); //Red
            return float4(overlay, originalColor.a);
            
        case 3: // Gradient Direction
            float2 normGrad = (edgeStrength > 0.0) ? normalize(gradient) : float2(0.0, 0.0);
            return float4(normGrad.x * 0.5 + 0.5, normGrad.y * 0.5 + 0.5, 0.0, 1.0);
            
        case 4: // Motion Vectors
            float magnitude = length(motion) * 50.0;
            float angle = atan2(motion.y, motion.x);
            float3 hsv = float3((angle / 6.283185) + 0.5, 1.0, saturate(magnitude));
            return float4(HSVtoRGB(hsv), 1.0);
            
        default:
            return originalColor;
    }
}

    /*-------------------.
    | :: Techniques ::   |
    '-------------------*/

technique DAA
<
    ui_tooltip = "Directional Anti-Aliasing. Enable Quark_Motion or any Motion Vectors for Temporal Anti-Aliasing";
>
{
    pass Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_TemporalDAA;
        RenderTarget = DAATemporal;
    }
    pass SaveHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveHistoryDAA;
        RenderTarget = DAAHistory;
    }
    pass Composite
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CompositeDAA;
    }
}
