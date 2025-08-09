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

    Version 1.6.3
    + Motion Vector Support
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
#define f float
#define f2 float2
#define f3 float3
#define f4 float4
#define i2 int2
#define t2 texture2D
#define s sampler2D
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

// Motion Vector texture selection
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

t2 TEMP
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
t2 HIS
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
s sTEMP
{
    Texture = TEMP;
};
s sHIS
{
    Texture = HIS;
};

/*----------------.
| :: Functions :: |
'----------------*/

f3 RGBToYCoCg(f3 rgb)
{
    float Y = dot(rgb, f3(0.299, 0.587, 0.114));
    float Cb = dot(rgb, f3(-0.169, -0.331, 0.500));
    float Cr = dot(rgb, f3(0.500, -0.419, -0.081));
    return f3(Y, Cb + 128.0 / 255.0, Cr + 128.0 / 255.0);
}

f3 YCoCgToRGB(f3 ycc)
{
    f3 c = ycc - f3(0.0, 128.0 / 255.0, 128.0 / 255.0);
    f R = c.x + 1.400 * c.z;
    f G = c.x - 0.343 * c.y - 0.711 * c.z;
    f B = c.x + 1.765 * c.y;
    return f3(R, G, B);
}

f lum(f3 color)
{
    return dot(color, 0.3333333);
}

f2 computeGradient(f2 t)
{
    f gradX = lum(tex2Doffset(ReShade::BackBuffer, t, int2(1, 0)).rgb) -
              lum(tex2Doffset(ReShade::BackBuffer, t, int2(-1, 0)).rgb);
    f gradY = lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, 1)).rgb) -
              lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, -1)).rgb);
    
    return f2(gradX, gradY) * 2.0;
}

f4 DAA(f2 t)
{
    f4 original = tex2Dlod(ReShade::BackBuffer, f4(t, 0, 0));
    f2 gradient = computeGradient(t);
    f edgeStrength = length(gradient);
    f weight = smoothstep(EdgeThreshold, EdgeThreshold + EdgeFalloff, edgeStrength);

    if (weight > 0.001)
    {
        f2 blurDir = normalize(f2(-gradient.y, gradient.x));
        f2 pixelStep = ReShade::PixelSize.xy * DirectionalStrength;
        f2 offset1 = blurDir * pixelStep * 0.5;
        f2 offset2 = blurDir * pixelStep;

        f4 color = (tex2Dlod(ReShade::BackBuffer, f4(t + offset1, 0, 0)) +
                    tex2Dlod(ReShade::BackBuffer, f4(t - offset1, 0, 0)) +
                    tex2Dlod(ReShade::BackBuffer, f4(t + offset2, 0, 0)) * 0.5 +
                    tex2Dlod(ReShade::BackBuffer, f4(t - offset2, 0, 0)) * 0.5);
                    
        color /= 3.0;
        return f4(lerp(original.rgb, color.rgb, weight), weight);
    }
    return f4(original.rgb, 0.0);
}

f4 PS_Temporal(f4 pos : SV_Position, f2 t : TEXCOORD) : SV_Target
{
    f4 current = DAA(t);
    f3 currentYCoCg = RGBToYCoCg(current.rgb);

    if (!EnableTemporalAA)
    {
        return current;
    }

    float2 motion = SampleMotionVectors(t);
    float2 reprojected_uv = t + motion;

    bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) && FRAME_COUNT > 1;

    if (validHistory)
    {
        // Neighborhood sampling for variance clamping
        f3 minColor = currentYCoCg;
        f3 maxColor = currentYCoCg;
        f3 meanColor = currentYCoCg;
        f3 m2Color = currentYCoCg * currentYCoCg;
        int sampleCount = 1;

        static const i2 offsets[8] =
        {
            i2(-1, 0), i2(1, 0), i2(0, -1), i2(0, 1),
            i2(-1, -1), i2(1, -1), i2(-1, 1), i2(1, 1)
        };

        [unroll]
        for (int i = 0; i < 8; ++i)
        {
            f3 sampleYCoCg = RGBToYCoCg(tex2Doffset(ReShade::BackBuffer, t, offsets[i]).rgb);
            minColor = min(minColor, sampleYCoCg);
            maxColor = max(maxColor, sampleYCoCg);
            meanColor += sampleYCoCg;
            m2Color += sampleYCoCg * sampleYCoCg;
            sampleCount++;
        }

        meanColor /= sampleCount;
        m2Color /= sampleCount;

        f3 variance = abs(m2Color - meanColor * meanColor);
        f3 stdDev = sqrt(variance);

        // Create expanded clamp box based on variance
        f3 expansion = lerp(0.5, 1.5, saturate(length(stdDev)));
        f3 halfSize = (maxColor - minColor) * expansion;
        f3 clampMin = meanColor - halfSize * 0.5;
        f3 clampMax = meanColor + halfSize * 0.5;

        // Fetch history
        f3 rawHistoryYCoCg = RGBToYCoCg(tex2Dlod(sHIS, f4(reprojected_uv, 0, 0)).rgb);

        // Soft AABB clamping
        f3 center = (clampMin + clampMax) * 0.5;
        f3 extents = (clampMax - clampMin) * 0.5;
        f3 historyYCoCg = center + clamp(rawHistoryYCoCg - center, -extents, extents);

        // Blend
        f alpha = 1.0 / min(FRAME_COUNT, (f) AccumFrames);
        currentYCoCg = lerp(historyYCoCg, currentYCoCg, alpha);
    }

    return f4(YCoCgToRGB(currentYCoCg), current.a);
}

f4 PS_SaveHistory(f4 pos : SV_Position, f2 t : TEXCOORD) : SV_Target
{
    return tex2Dlod(sTEMP, f4(t, 0, 0));
}

f4 PSComposite(f4 pos : SV_Position, f2 t : TEXCOORD) : SV_Target
{
    f4 original = tex2Dlod(ReShade::BackBuffer, f4(t, 0, 0));
    f4 daaResult = tex2Dlod(sTEMP, f4(t, 0, 0));
    
    // View modes
    switch (View_Mode)
    {
        case 2: // Edge Mask
            return f4(daaResult.aaa, 1.0);
            
        case 1: // Edge Mask Overlay
            return f4(lerp(original.rgb, f3(1.0, 0.2, 0.2), daaResult.a * 0.7), 1.0);
            
        case 3: // Gradient Direction
            f2 grad = computeGradient(t);
            return f4(normalize(grad) * 0.5 + 0.5, 0.0, 1.0);
            
        default: // Output
            return f4(daaResult.rgb, 1.0);
    }
}

/*-------------------.
| :: Techniques ::   |
'-------------------*/

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
    pass Composite
    {
        VertexShader = PostProcessVS;
        PixelShader = PSComposite;
    }
}

