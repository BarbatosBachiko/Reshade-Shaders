/*------------------.
| :: Description :: |
'-------------------/

    Directional Anti-Aliasing (DAA) 
    Author: Barbatos Bachiko
    License: MIT

    About: Directional Anti-Aliasing (DAA) is an edge-aware spatiotemporal anti-aliasing technique 
    that smooths edges by applying directional blurring based on local gradient detection.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 1.6.5
    + Better lum Gradient
*/

// Includes
#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Utility macros
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define MAX_FRAMES 64

/*-------------------.
| :: Settings ::     |
'-------------------*/

uniform int View_Mode <
    ui_category = "Anti-Aliasing";
    ui_type = "combo";
    ui_items = "Output\0Edge Mask Overlay\0Edge Mask\0Gradient Direction\0";
    ui_label = "View Mode";
> = 0;

uniform float DirectionalStrength <
    ui_type = "drag";
    ui_label = "Strength";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.05;
    ui_category = "Anti-Aliasing";
> = 2.4;
    
uniform float EdgeThreshold <
    ui_type = "slider";
    ui_label = "Edge Threshold";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Edge Detection";
> = 0.260;

uniform float EdgeFalloff <
    ui_type = "slider";
    ui_label = "EdgeFalloff";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Edge Detection";
> = 0.0;

uniform bool EnableTemporalAA <
    ui_category = "Temporal";
    ui_type = "checkbox";
    ui_label = "Enable Temporal AA";
> = false;

uniform int AccumFrames <
    ui_type = "slider";
    ui_label = "Accumulation Frames";
    ui_min = 1; ui_max = 32; ui_step = 1;
    ui_category = "Temporal";
> = 8;

uniform int FRAME_COUNT < source = "framecount"; >;

/*---------------.
| :: Textures :: |
'---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).rg;
    }
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
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};
float2 SampleMotionVectors(float2 texcoord)
{
    return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).rg;
}
#endif

texture2D TEMP
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
texture2D HIS
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler2D sTEMP
{
    Texture = TEMP;
};
sampler2D sHIS
{
    Texture = HIS;
};

/*----------------.
| :: Functions :: |
'----------------*/

float3 RGBToYCoCg(float3 rgb)
{
    float Y = dot(rgb, float3(0.299, 0.587, 0.114));
    float Cb = dot(rgb, float3(-0.169, -0.331, 0.500));
    float Cr = dot(rgb, float3(0.500, -0.419, -0.081));
    return float3(Y, Cb + 0.5019608, Cr + 0.5019608);
}

float3 YCoCgToRGB(float3 ycc)
{
    float3 c = ycc - float3(0.0, 0.5019608, 0.5019608);
    float R = c.x + 1.400 * c.z;
    float G = c.x - 0.343 * c.y - 0.711 * c.z;
    float B = c.x + 1.765 * c.y;
    return float3(R, G, B);
}

float lum(float3 color)
{
    return dot(color, 0.3333333);
}

float2 computeGradient(float2 t) //with sobel
{
    float l0 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(-1, -1)).rgb);
    float l1 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, -1)).rgb);
    float l2 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(1, -1)).rgb);
    float l3 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(-1, 0)).rgb);
    float l5 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(1, 0)).rgb);
    float l6 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(-1, 1)).rgb);
    float l7 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, 1)).rgb);
    float l8 = lum(tex2Doffset(ReShade::BackBuffer, t, int2(1, 1)).rgb);

    float gradX = -l0 - 2.0 * l3 - l6 + l2 + 2.0 * l5 + l8;
    float gradY = -l0 - 2.0 * l1 - l2 + l6 + 2.0 * l7 + l8;

    return float2(gradX, gradY);
}

float4 DAA(float2 t)
{
    float4 original = tex2Dlod(ReShade::BackBuffer, float4(t, 0, 0));
    float2 gradient = computeGradient(t);
    float edgeStrength = length(gradient);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + max(EdgeFalloff, 0.0001), edgeStrength);

    if (weight > 0.001)
    {
        float2 blurDir = normalize(float2(-gradient.y, gradient.x));
        float2 pixelStep = ReShade::PixelSize.xy * DirectionalStrength;
        float2 offset1 = blurDir * pixelStep * 0.5;
        float2 offset2 = blurDir * pixelStep;

        float4 color = (tex2Dlod(ReShade::BackBuffer, float4(t + offset1, 0, 0)) +
                        tex2Dlod(ReShade::BackBuffer, float4(t - offset1, 0, 0)) +
                        tex2Dlod(ReShade::BackBuffer, float4(t + offset2, 0, 0)) * 0.5 +
                        tex2Dlod(ReShade::BackBuffer, float4(t - offset2, 0, 0)) * 0.5);
                        
        color /= 3.0;
        return float4(lerp(original.rgb, color.rgb, weight), weight);
    }
    return float4(original.rgb, 0.0);
}

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

float4 PS_Temporal(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    float4 current = DAA(t);
    float3 currentYCoCg = RGBToYCoCg(current.rgb);

    if (!EnableTemporalAA)
    {
        return current;
    }

    float2 motion = SampleMotionVectors(t);
    float2 reprojected_uv = t + motion;

    bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) && FRAME_COUNT > 1;

    if (validHistory)
    {
        float3 minColor = currentYCoCg;
        float3 maxColor = currentYCoCg;
        float3 meanColor = currentYCoCg;
        float3 m2Color = currentYCoCg * currentYCoCg;
        int sampleCount = 1;

        static const int2 offsets[8] =
        {
            int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
            int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
        };

        [unroll]
        for (int i = 0; i < 8; ++i)
        {
            float3 sampleYCoCg = RGBToYCoCg(tex2Doffset(ReShade::BackBuffer, t, offsets[i]).rgb);
            minColor = min(minColor, sampleYCoCg);
            maxColor = max(maxColor, sampleYCoCg);
            meanColor += sampleYCoCg;
            m2Color += sampleYCoCg * sampleYCoCg;
            sampleCount++;
        }

        meanColor /= sampleCount;
        m2Color /= sampleCount;

        float3 variance = abs(m2Color - meanColor * meanColor);
        float3 stdDev = sqrt(variance);

        float3 expansion = lerp(0.5, 1.5, saturate(length(stdDev)));
        float3 halfSize = (maxColor - minColor) * expansion;
        float3 clampMin = meanColor - halfSize * 0.5;
        float3 clampMax = meanColor + halfSize * 0.5;

        float3 rawHistoryYCoCg = RGBToYCoCg(tex2Dlod(sHIS, float4(reprojected_uv, 0, 0)).rgb);

        float3 center = (clampMin + clampMax) * 0.5;
        float3 extents = (clampMax - clampMin) * 0.5;
        float3 historyYCoCg = center + clamp(rawHistoryYCoCg - center, -extents, extents);

        float alpha = 1.0 / min(FRAME_COUNT, (float) AccumFrames);
        currentYCoCg = lerp(historyYCoCg, currentYCoCg, alpha);
    }

    return float4(YCoCgToRGB(currentYCoCg), current.a);
}

float4 PS_SaveHistory(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    return tex2Dlod(sTEMP, float4(t, 0, 0));
}

float4 OutPut(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    float4 original = tex2Dlod(ReShade::BackBuffer, float4(t, 0, 0));
    float4 daaResult = tex2Dlod(sTEMP, float4(t, 0, 0));
    
    // View modes
    switch (View_Mode)
    {
        case 2: // Edge Mask
            return float4(daaResult.aaa, 1.0);
            
        case 1: // Edge Mask Overlay
            return float4(lerp(original.rgb, float3(1.0, 0.2, 0.2), daaResult.a * 0.7), 1.0);
            
        case 3: // Gradient Direction
            float2 grad = computeGradient(t);
            return float4(normalize(grad) * 0.5 + 0.5, 0.0, 1.0);
            
        default: // Output
            return float4(daaResult.rgb, 1.0);
    }
}

technique DAA
<
    ui_tooltip = "Directional SpatioTemporal Anti-Aliasing.";
>
{
    pass Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Temporal;
        RenderTarget = TEMP;
    }
    pass SaveHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveHistory;
        RenderTarget = HIS;
    }
    pass Output
    {
        VertexShader = PostProcessVS;
        PixelShader = OutPut;
    }
}
