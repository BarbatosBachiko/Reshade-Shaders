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

    Version 1.7.0
    + Implemented sub-pixel jittering.
    + Added history rejection based on scene depth to reduce ghosting on disocclusions.
    + Added history rejection based on motion vector length to improve stability.
    + Implemented Catmull-Rom resampling for history buffer to reduce blur.
    + Added a UI difficulty system with Simple and Advanced modes.
*/

#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Utility macros
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define MAX_FRAMES 64

//----------|
// :: UI :: |
//----------|


#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if UI_DIFFICULTY == 0 // Simple Mode

#define EdgeThreshold 0.260
#define EdgeFalloff 0.0
#define AccumFrames 8
#define JitterStrength 1.0
#define DepthThreshold 0.01
#define MotionRejection 0.2
#define VarianceClippingGamma 1.0

    // -- Debug --
uniform int View_Mode <
        ui_category = "Debug";
        ui_type = "combo";
        ui_items = "Output\0Edge Mask Overlay\0Edge Mask\0Gradient Direction\0";
        ui_label = "View Mode";
    > = 0;

uniform float DirectionalStrength <
        ui_type = "drag";
        ui_label = "Strength";
        ui_min = 0.0; ui_max = 3.0; ui_step = 0.05;
    > = 2.4;

uniform bool EnableTemporalAA <
        ui_type = "checkbox";
        ui_label = "Enable Temporal AA";
    > = false;

#elif UI_DIFFICULTY == 1 // Advanced Mode

uniform int View_Mode <
        ui_category = "Debug";
        ui_type = "combo";
        ui_items = "Output\0Edge Mask Overlay\0Edge Mask\0Gradient Direction\0";
        ui_label = "View Mode";
    > = 0;

    // -- Main Settings --
    uniform float DirectionalStrength <
        ui_type = "drag";
        ui_label = "Strength";
        ui_min = 0.0; ui_max = 3.0; ui_step = 0.05;
        ui_category = "Main Settings";
    > = 2.4;
    
    uniform bool EnableTemporalAA <
        ui_category = "Main Settings";
        ui_type = "checkbox";
        ui_label = "Enable Temporal AA";
    > = true;

    // -- Edge Detection --
    uniform float EdgeThreshold <
        ui_type = "slider";
        ui_label = "Edge Threshold";
        ui_min = 0.1; ui_max = 1.0; ui_step = 0.001;
        ui_category = "Edge Detection";
    > = 0.260;

    uniform float EdgeFalloff <
        ui_type = "slider";
        ui_label = "Edge Falloff";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
        ui_category = "Edge Detection";
    > = 0.0;

    // -- Temporal Filtering --
    uniform int AccumFrames <
        ui_type = "drag";
        ui_label = "Accumulation Frames";
        ui_min = 1; ui_max = 16; ui_step = 1;
        ui_category = "Temporal Filtering";
    > = 8;

    uniform float JitterStrength <
        ui_type = "drag";
        ui_label = "Jitter Strength";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
        ui_category = "Temporal Filtering";
    > = 1.0;

    uniform float DepthThreshold <
        ui_type = "slider";
        ui_label = "Depth Rejection Threshold";
        ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
        ui_category = "Temporal Filtering";
    > = 0.01;

    uniform float MotionRejection <
        ui_type = "slider";
        ui_label = "Motion Rejection Threshold";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Temporal Filtering";
    > = 0.2;

    uniform float VarianceClippingGamma <
        ui_type = "slider";
        ui_label = "Variance Clipping Gamma";
        ui_min = 0.5; ui_max = 2.0; ui_step = 0.05;
        ui_category = "Temporal Filtering";
        ui_tooltip = "Controls the size of the color extents for history rectification. Lower values are more aggressive at rejecting history, preventing ghosting but may cause flickering. Higher values are more permissive, improving stability but may allow ghosting.";
    > = 1.0;

#endif

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
        return GetLod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, float4(texcoord, 0, 0)).rg;
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
    return GetLod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).rg;
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
texture2D DEPTH
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R32F;
};

sampler2D sTEMP
{
    Texture = TEMP;
};
sampler2D sHIS
{
    Texture = HIS;
};
sampler2D sDEPTH
{
    Texture = DEPTH;
};

/*----------------.
| :: Functions :: |
'----------------*/

float halton(int index, int base)
{
    float result = 0.0;
    float f = 1.0 / base;
    int i = index;
    while (i > 0)
    {
        result = result + f * (i % base);
        i = floor(i / base);
        f = f / base;
    }
    return result;
}

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

float4 cubic(float4 p0, float4 p1, float4 p2, float4 p3, float t)
{
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)));
}

float4 tex2D_catmullrom(sampler s, float2 uv)
{
    float2 texelSize = ReShade::PixelSize;
    float2 frac_uv = frac(uv / texelSize - 0.5);
    float2 base_uv = (floor(uv / texelSize - 0.5) + 0.5) * texelSize;

    float4 row[4];
    for (int j = -1; j <= 2; ++j)
    {
        float4 p0 = GetLod(s, base_uv + float2(-1, j) * texelSize);
        float4 p1 = GetLod(s, base_uv + float2(0, j) * texelSize);
        float4 p2 = GetLod(s, base_uv + float2(1, j) * texelSize);
        float4 p3 = GetLod(s, base_uv + float2(2, j) * texelSize);
        row[j + 1] = cubic(p0, p1, p2, p3, frac_uv.x);
    }

    return cubic(row[0], row[1], row[2], row[3], frac_uv.y);
}

float4 DAA(float2 t)
{
    float4 original = GetColor(t);
    float2 gradient = computeGradient(t);
    float edgeStrength = length(gradient);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + max(EdgeFalloff, 0.0001), edgeStrength);

    if (weight > 0.001)
    {
        float2 blurDir = normalize(float2(-gradient.y, gradient.x));
        float2 pixelStep = ReShade::PixelSize.xy * DirectionalStrength;
        float2 offset1 = blurDir * pixelStep * 0.5;
        float2 offset2 = blurDir * pixelStep;

        float4 color = (GetColor(t + offset1) +
                        GetColor(t - offset1) +
                        GetColor(t + offset2) * 0.5 +
                        GetColor(t - offset2) * 0.5);
                        
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
    float2 jitter = float2(halton(FRAME_COUNT % MAX_FRAMES, 2), halton(FRAME_COUNT % MAX_FRAMES, 3)) - 0.5;
    float2 jittered_uv = t + (jitter * ReShade::PixelSize * JitterStrength * EnableTemporalAA);

    float4 current = DAA(jittered_uv);
    
    if (!EnableTemporalAA)
    {
        return current;
    }

    float3 currentYCoCg = RGBToYCoCg(current.rgb);
    float2 motion = SampleMotionVectors(t);
    float2 reprojected_uv = t + motion;

    bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) && FRAME_COUNT > 1;
    
    float currentDepth = GetDepth(t);
    float historyDepth = GetLod(sDEPTH, reprojected_uv).r;
    
    if (abs(currentDepth - historyDepth) > DepthThreshold)
    {
        validHistory = false;
    }

    if (validHistory)
    {
        float3 sumColor = 0.0;
        float3 sumColorSq = 0.0;
        int sampleCount = 0;

        static const int2 offsets[9] =
        {
            int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
            int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
        };

        [unroll]
        for (int i = 0; i < 9; ++i)
        {
            float3 sampleYCoCg = RGBToYCoCg(tex2Doffset(ReShade::BackBuffer, jittered_uv, offsets[i]).rgb);
            sumColor += sampleYCoCg;
            sumColorSq += sampleYCoCg * sampleYCoCg;
            sampleCount++;
        }

        float3 mean = sumColor / sampleCount;
        float3 variance = abs(sumColorSq / sampleCount - mean * mean);
        float3 std_dev = sqrt(variance);

        float3 clampMin = mean - VarianceClippingGamma * std_dev;
        float3 clampMax = mean + VarianceClippingGamma * std_dev;

        float3 rawHistoryYCoCg = RGBToYCoCg(tex2D_catmullrom(sHIS, reprojected_uv).rgb);

        float3 center = (clampMin + clampMax) * 0.5;
        float3 extents = (clampMax - clampMin) * 0.5;
        float3 historyYCoCg = center + clamp(rawHistoryYCoCg - center, -extents, extents);

        float alpha = 1.0 / min(FRAME_COUNT, (float) AccumFrames);
        
        float motionLength = length(motion * ReShade::ScreenSize);
        alpha = lerp(alpha, 1.0, smoothstep(0.0, MotionRejection * 100.0, motionLength));

        currentYCoCg = lerp(historyYCoCg, currentYCoCg, alpha);
    }

    return float4(YCoCgToRGB(currentYCoCg), current.a);
}

float4 PS_SaveHistory(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    return GetLod(sTEMP, t);
}

float PS_SaveDepth(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    return GetDepth(t);
}

float4 OutPut(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    float4 original = GetColor(t);
    float4 daaResult = GetLod(sTEMP, t);
    
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
    pass SaveDepth
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveDepth;
        RenderTarget = DEPTH;
    }
    pass Output
    {
        VertexShader = PostProcessVS;
        PixelShader = OutPut;
    }
}

