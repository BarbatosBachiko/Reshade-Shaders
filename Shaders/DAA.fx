/*------------------.
| :: Description :: |
'-------------------/

    Directional Anti-Aliasing (DAA)
    
    Version 1.6
    Author: Barbatos Bachiko
    License: MIT

    About: Directional Anti-Aliasing (DAA) is an edge-aware spatiotemporal anti-aliasing technique 
    that smooths edges by applying directional blurring based on local gradient detection.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.6
    + Temporal
*/

// Includes
#include "ReShade.fxh"

// Utility macros
#define getColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter= POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define SkyDepth 0.99
#define MAX_Frames 64

    /*-------------------.
    | :: Settings ::    |
    '-------------------*/

uniform int View_Mode
<
    ui_category = "Anti-Aliasing";
    ui_type = "combo";
    ui_items = "Output\0Edge Mask\0Edge Mask Overlay\0Gradient Direction\0";
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

static float AccumFrames = 16;
uniform int FRAME_COUNT < source = "framecount"; >;
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

    /*----------------.
    | :: Functions :: |
    '----------------*/

float lum(float3 color)
{
    return (color.r + color.g + color.b) * 0.3333333;
}

// Calculates the gradient using the Scharr operator
float2 ComputeGradient(float2 texcoord)
{
    const float2 offset = ReShade::PixelSize.xy * PixelWidth;

    float lumTL = lum(getColor(float4(texcoord + float2(-offset.x, -offset.y), 0, 0)).rgb);
    float lumT = lum(getColor(float4(texcoord + float2(0, -offset.y), 0, 0)).rgb);
    float lumTR = lum(getColor(float4(texcoord + float2(offset.x, -offset.y), 0, 0)).rgb);
    float lumL = lum(getColor(float4(texcoord + float2(-offset.x, 0), 0, 0)).rgb);
    float lumR = lum(getColor(float4(texcoord + float2(offset.x, 0), 0, 0)).rgb);
    float lumBL = lum(getColor(float4(texcoord + float2(-offset.x, offset.y), 0, 0)).rgb);
    float lumB = lum(getColor(float4(texcoord + float2(0, offset.y), 0, 0)).rgb);
    float lumBR = lum(getColor(float4(texcoord + float2(offset.x, offset.y), 0, 0)).rgb);

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

float3 RGBToYCoCg(float3 rgb)
{
    float Y = .299 * rgb.x + .587 * rgb.y + .114 * rgb.z; // Luminance
    float Cb = -.169 * rgb.x - .331 * rgb.y + .500 * rgb.z; // Chrominance Blue
    float Cr = .500 * rgb.x - .419 * rgb.y - .081 * rgb.z; // Chrominance Red
    return float3(Y, Cb + 128. / 255., Cr + 128. / 255.);
}

float3 YCoCgToRGB(float3 ycc)
{
    float3 c = ycc - float3(0., 128. / 255., 128. / 255.);
    
    float R = c.x + 1.400 * c.z;
    float G = c.x - 0.343 * c.y - 0.711 * c.z;
    float B = c.x + 1.765 * c.y;
    return float3(R, G, B);
}

float4 PS_Temporal(float4 pos : SV_Position, float2 texcoord : TEXCOORD, out float outHistoryLength : SV_Target1) : SV_Target
{

    float4 current = DirectionalAA(texcoord);
    float3 currentYCoCg = RGBToYCoCg(current.rgb);

    float depth = getDepth(texcoord);
    bool validHistory = (depth <= SkyDepth) &&
                        all(saturate(texcoord) == texcoord) &&
                        FRAME_COUNT > 1;

    float3 historyYCoCg = currentYCoCg;

    if (EnableTemporalAA && validHistory)
    {
        float2 pixelSize = ReShade::PixelSize.xy;

        // Neighborhood sampling for AABB clamping
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
            float2 offset = texcoord + pixelSize * offsets[i];
            float3 sampleYCoCg = RGBToYCoCg(getColor(float4(offset, 0, 0)).rgb);

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
        minColor = meanColor - halfSize * 0.5;
        maxColor = meanColor + halfSize * 0.5;

        float3 rawHistoryYCoCg = RGBToYCoCg(
            tex2Dlod(sDAAHistory, float4(texcoord, 0, 0)).rgb
        );

        // Soft AABB clamping
        float3 center = (minColor + maxColor) * 0.5;
        float3 extents = (maxColor - minColor) * 0.5;
        historyYCoCg = center + clamp(rawHistoryYCoCg - center, -extents, extents);

        // Blend
        float alpha = 1.0 / min(FRAME_COUNT, (float) AccumFrames);
        currentYCoCg = lerp(historyYCoCg, currentYCoCg, alpha);
    }

    outHistoryLength = validHistory ? min(FRAME_COUNT, MAX_Frames) : 0;
    return float4(YCoCgToRGB(currentYCoCg), current.a);
}

// History pass
float4 PS_SaveHistory(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 temporalResult = tex2Dlod(sDAATemporal, float4(texcoord, 0, 0));
    return temporalResult;
}

// Composite pass
float4 PS_Composite(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 originalColor = getColor(texcoord);
    float4 daaColor = tex2Dlod(sDAATemporal, float4(texcoord, 0, 0));
    
    float2 gradient = ComputeGradient(texcoord);
    float edgeStrength = length(gradient);
    float weight = smoothstep(EdgeThreshold, EdgeThreshold + EdgeFalloff, edgeStrength);
    

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
            
        default:
            return originalColor;
    }
}

    /*-------------------.
    | :: Techniques ::   |
    '-------------------*/

technique DAA
<
    ui_tooltip = "Directional SpatioTemporal Anti-Aliasing";
>
{
    pass Temporal
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Temporal;
        RenderTarget = DAATemporal;
    }
    pass SaveHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SaveHistory;
        RenderTarget = DAAHistory;
    }
    pass Composite
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Composite;
    }
}
