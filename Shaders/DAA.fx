/*-------------------------------------------------|
| ::       Directional Anti-Aliasing (DAA)      :: |
'--------------------------------------------------|
| Version: 1.8                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: is an edge-aware spatiotemporal     |
|  anti-aliasing technique that smooths edges by   |
|  applying directional blurring based on          | 
|  local gradient detection.                       |
'-------------------------------------------------*/

#include "ReShade.fxh"

#ifndef ENABLE_JITTER_FOR_TAA
#define ENABLE_JITTER_FOR_TAA 1
#endif

// Constants
#define BASE_BLEND_ALPHA 0.12 
#define DEPTH_THRESHOLD 0.01
#define MOTION_REJECTION 0.2
#define VARIANCE_GAMMA 1.0

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

// Utility macros 
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0.0, 0.0))
#define GetLod(s, c) tex2Dlod(s, float4((c).xy, 0.0, 0.0))

//----------|
// :: UI :: |
//----------|

uniform float DirectionalStrength <
    ui_type = "drag";
    ui_label = "DAA Strength";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.05;
    ui_category = "Main Settings";
> = 2.4;

uniform float EdgeThreshold <
    ui_type = "drag";
    ui_label = "Edge Threshold";
    ui_min = 0.0; ui_max = 0.5;
    ui_step = 0.001;
    ui_category = "Edge Detection";
> = 0.100;

uniform float EdgeFalloff <
    ui_type = "slider";
    ui_label = "Edge Falloff";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_category = "Edge Detection";
> = 0.0;

uniform bool EnableTemporalAA <
    ui_type = "checkbox";
    ui_label = "Enable Temporal AA";
    ui_category = "Temporal Filtering";
> = false;

uniform int View_Mode <
    ui_category = "Debug";
    ui_type = "combo";
    ui_items = "Output\0Edge Mask Overlay\0Edge Mask\0Gradient Direction\0";
    ui_label = "View Mode";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;

static const float2 JitterLUT[16] =
{
    float2(0.0000, -0.1667), float2(-0.2500, 0.1667), float2(0.2500, -0.3889), float2(-0.3750, -0.0556),
    float2(0.1250, 0.2778), float2(-0.1250, -0.2778), float2(0.3750, 0.0556), float2(-0.4375, 0.3889),
    float2(0.0625, -0.1481), float2(-0.1875, 0.1852), float2(0.3125, -0.3704), float2(-0.3125, -0.0370),
    float2(0.1875, 0.2963), float2(-0.0625, -0.2593), float2(0.4375, 0.0741), float2(-0.4688, 0.4074)
};

/*---------------.
| :: Textures :: |
'---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, texcoord).rg;
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
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
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

struct Output
{
    float4 Color : SV_Target0;
    float4 Depth : SV_Target1;
};

static const float3 CoefLuma = float3(0.299, 0.587, 0.114);
static const float3 CoefChromaBlue = float3(-0.169, -0.331, 0.500);
static const float3 CoefChromaRed = float3(0.500, -0.419, -0.081);
static const float ChromaBias = 0.5019608; 

float3 RGBToYCoCg(float3 rgb)
{
    float Y = dot(rgb, CoefLuma);
    float Cb = dot(rgb, CoefChromaBlue);
    float Cr = dot(rgb, CoefChromaRed);
    return float3(Y, Cb + ChromaBias, Cr + ChromaBias);
}

float3 YCoCgToRGB(float3 ycc)
{
    float3 c = ycc - float3(0.0, ChromaBias, ChromaBias);
    return float3(
        c.x + 1.400 * c.z,
        c.x - 0.343 * c.y - 0.711 * c.z,
        c.x + 1.765 * c.y
    );
}

float lum(float3 color)
{
    return dot(color, 0.3333333);
}

float2 computeGradient(float2 t)
{
    float l_left = lum(tex2Doffset(ReShade::BackBuffer, t, int2(-1, 0)).rgb);
    float l_right = lum(tex2Doffset(ReShade::BackBuffer, t, int2(1, 0)).rgb);
    float l_up = lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, -1)).rgb);
    float l_down = lum(tex2Doffset(ReShade::BackBuffer, t, int2(0, 1)).rgb);
    return float2(l_right - l_left, l_down - l_up);
}

float GetLum(float2 uv)
{
    return lum(GetColor(uv).rgb);
}

float4 DAA(float2 base_uv, float2 jitter_uv)
{
    float4 original = GetColor(jitter_uv);
    float2 gradient = computeGradient(base_uv);
    float gradLen = length(gradient);
    
    if (gradLen < 0.0001) 
        return float4(original.rgb, 0.0);

    // Second Order Analysis
    float2 dir = gradient / gradLen;
    
    // Sample along the gradient direction to check linearity vs curvature.
    float2 offset = dir * ReShade::PixelSize.xy;
    float l_center = GetLum(base_uv);
    float l_fwd = GetLum(base_uv + offset);
    float l_bwd = GetLum(base_uv - offset);

    // Calculate Second Derivative magnitude (Directional Laplacian)
    // Formula: f''(x) approx f(x+h) - 2f(x) + f(x-h)
    float edgeStrength = abs(l_fwd + l_bwd - 2.0 * l_center);

    float weight = smoothstep(EdgeThreshold, EdgeThreshold + max(EdgeFalloff, 0.0001), edgeStrength);

    if (weight > 0.001)
    {
        // Directional Blur 
        // We blur along the TANGENT (perpendicular to gradient)
        // Rotate gradient 90 degrees: (-y, x)
        float2 blurDir = float2(-dir.y, dir.x);
        
        float2 pixelStep = ReShade::PixelSize.xy * DirectionalStrength;
        float2 offset2 = blurDir * pixelStep;
        float2 offset1 = offset2 * 0.5;

        // Sample taps
        float4 t1 = GetColor(jitter_uv + offset1);
        float4 t2 = GetColor(jitter_uv - offset1);
        float4 t3 = GetColor(jitter_uv + offset2);
        float4 t4 = GetColor(jitter_uv - offset2);

        float4 color = t1 + t2 + (t3 * 0.5) + (t4 * 0.5);
        color *= 0.3333333; 

        return float4(lerp(original.rgb, color.rgb, weight), weight);
    }

    return float4(original.rgb, 0.0);
}

float4 PS_Temporal(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    float2 jitter = JitterLUT[FRAME_COUNT % 16];
    
#if ENABLE_JITTER_FOR_TAA
    float jitterVal = EnableTemporalAA ? 1.0 : 0.0;
#else
    float jitterVal = 0.0;
#endif
    
    float2 jittered_uv = t + (jitter * ReShade::PixelSize * jitterVal);
    
    float4 current = DAA(t, jittered_uv);

    if (!EnableTemporalAA)
        return current;

    float3 currentYCoCg = RGBToYCoCg(current.rgb);
    float2 motion = SampleMotionVectors(t);
    float2 reprojected_uv = t + motion;
    
    // History Validation
    bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) && (FRAME_COUNT > 1);
    
    if (validHistory)
    {
        // Depth rejection
        float currentDepth = GetDepth(t);
        float historyDepth = GetLod(sDEPTH, reprojected_uv).r;
        
        if (abs(currentDepth - historyDepth) > DEPTH_THRESHOLD)
            validHistory = false;
    }

    if (validHistory)
    {
        float3 sumColor = 0.0;
        float3 sumColorSq = 0.0;
        
        // Neighborhood clamping
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            [unroll]
            for (int x = -1; x <= 1; ++x)
            {
                float3 s = RGBToYCoCg(tex2Doffset(ReShade::BackBuffer, jittered_uv, int2(x, y)).rgb);
                sumColor += s;
                sumColorSq += s * s;
            }
        }

        float3 mean = sumColor * 0.111111; 
        float3 variance = abs((sumColorSq * 0.111111) - (mean * mean));
        float3 std_dev = sqrt(variance);
        
        float3 clampMin = mean - VARIANCE_GAMMA * std_dev;
        float3 clampMax = mean + VARIANCE_GAMMA * std_dev;

        float3 historyYCoCg = RGBToYCoCg(tex2D(sHIS, reprojected_uv).rgb);
        historyYCoCg = clamp(historyYCoCg, clampMin, clampMax);
        
        float alpha = (FRAME_COUNT < 8) ? (1.0 / float(FRAME_COUNT)) : BASE_BLEND_ALPHA;
        float motionLength = length(motion * ReShade::ScreenSize);
        
        alpha = saturate(alpha + smoothstep(0.0, MOTION_REJECTION * 100.0, motionLength));

        currentYCoCg = lerp(historyYCoCg, currentYCoCg, alpha);
    }

    return float4(YCoCgToRGB(currentYCoCg), current.a);
}

Output PS_SaveHistoryDepth(float4 pos : SV_Position, float2 t : TEXCOORD)
{
    Output outData;
    outData.Color = GetLod(sTEMP, t);
    float d = GetDepth(t);
    outData.Depth = float4(d, d, d, 1.0);
    return outData;
}

float4 OutPut(float4 pos : SV_Position, float2 t : TEXCOORD) : SV_Target
{
    float4 daaResult = GetLod(sTEMP, t);
    
    switch (View_Mode)
    {
        case 2: // Edge Mask
            return float4(daaResult.aaa, 1.0);
        case 1: // Edge Mask Overlay
            return float4(lerp(GetColor(t).rgb, float3(1.0, 0.2, 0.2), daaResult.a * 0.7), 1.0);
        case 3: // Gradient Direction
            return float4(normalize(computeGradient(t)) * 0.5 + 0.5, 0.0, 1.0);
        default: // Output
            return float4(daaResult.rgb, 1.0);
    }
}

technique DAA
<
    ui_label = "Directional Anti-Aliasing.";
    ui_tooltip = "Directional SpatioTemporal Anti-Aliasing.";
>
{
    pass Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Temporal;
        RenderTarget = TEMP;
    }
    pass Output
    {
        VertexShader = PostProcessVS;
        PixelShader = OutPut;
    }
    pass SaveHistoryDepth
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveHistoryDepth;
        RenderTarget0 = HIS;
        RenderTarget1 = DEPTH;
    }
}
