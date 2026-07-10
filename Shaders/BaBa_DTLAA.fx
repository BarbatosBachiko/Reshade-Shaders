/*----------------------------------------------|
| ::              BaBa DTLAA                 :: |
|-----------------------------------------------|
| Version: 1.1                                  |
| Author: Barbatos                              |
| License: MIT                                  |
| Description: Directionally Localized AA       |
| (Andreev GDC 2011) + optional temporal AA.    |
| Ref: http://and.intercon.ru/releases/talks/   |
|      dlaagdc2011/slides/                      |
'----------------------------------------------*/

#include ".\Includes\bb_reshade.fxh"
#include ".\Includes\bb_common.fxh"
#include ".\Includes\bb_colorspace.fxh"
#include ".\Includes\bb_mv.fxh"
#include ".\Includes\bb_taa.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Strength <
    ui_type = "slider";
    ui_label = "Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform bool EnableTemporal <
    ui_label = "Enable Temporal";
    ui_tooltip = "TAA after spatial DLAA. Prefer BaBa_Flow (or other MV) earlier in the preset.";
> = true;

uniform float TemporalStability <
    ui_type = "slider";
    ui_label = "Temporal Stability";
    ui_min = 0.0; ui_max = 0.99; ui_step = 0.01;
> = 0.90;

uniform float Lambda <
    ui_type = "slider";
    ui_label = "Short Edge Lambda";
    ui_tooltip = "Short-edge effectiveness (TFU2 used 3.0).";
    ui_min = 0.5; ui_max = 8.0; ui_step = 0.1;
> = 3.0;

uniform float Epsilon <
    ui_type = "slider";
    ui_label = "Short Edge Epsilon";
    ui_tooltip = "Bias subtracted in edge crush + short-edge gate. Visible in Edge Alpha debug.";
    ui_min = 0.0; ui_max = 0.5; ui_step = 0.005;
> = 0.05;

uniform float LongEdgeThreshold <
    ui_type = "slider";
    ui_label = "Long Edge Threshold";
    ui_tooltip = "Noise rejection: require abs(longH-longV) above this. Visible in Long Mask debug.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.35;

uniform float EdgeContrast <
    ui_type = "slider";
    ui_label = "Edge Contrast";
    ui_tooltip = "Prefilter high-pass gain. Visible in Edge Alpha / Long Mask debug.";
    ui_min = 0.5; ui_max = 8.0; ui_step = 0.1;
> = 2.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Off\0Long Mask\0Short Mask\0Temporal Blend\0Edge Alpha\0";
    ui_label = "View Mode";
    ui_tooltip = "Long / Short Mask: overlay on color\nTemporal Blend: feedback amount\nEdge Alpha: prefilter edge (replace)\nDebug frames are never written to TAA history.";
> = 0;

//----------------|
// :: Textures :: |
//----------------|

texture TexPreFilter { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler sPreFilter { Texture = TexPreFilter; };

texture TexSpatial { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler sSpatial { Texture = TexSpatial; };

texture HistoryTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler sHistoryTex { Texture = HistoryTex; };

//----------------|
// :: Functions ::|
//----------------|

// Talk §43/§51: green as luminosity for DLAA kernels
float LI(float3 c)
{
    return c.g;
}

float4 LoadPF(float2 uv, float2 offset_px)
{
    return tex2Dlod(sPreFilter, float4(uv + offset_px * bb::PixelSize, 0, 0));
}

float4 LoadBB(float2 uv, float2 offset_px)
{
    return tex2Dlod(bb::BackBuffer, float4(uv + offset_px * bb::PixelSize, 0, 0));
}

// Safe t in (0,1) for luminosity match; 0 = invalid
float SafeLumaT(float blurred, float fromLum, float toLum)
{
    float denom = toLum - fromLum;
    if (abs(denom) < 1e-4)
        return 0.0;
    float t = (blurred - fromLum) / denom;
    return (t > 0.0 && t < 1.0) ? t : 0.0;
}

//---------------------|
// :: Pixel Shaders :: |
//---------------------|

float4 PS_PreFilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float4 center = LoadBB(uv, float2(0, 0));
    float4 left   = LoadBB(uv, float2(-1, 0));
    float4 right  = LoadBB(uv, float2( 1, 0));
    float4 top    = LoadBB(uv, float2( 0,-1));
    float4 bottom = LoadBB(uv, float2( 0, 1));

    // High-pass + crush (talk slide 22): saturate(abs(x)*a - b)
    // Scale *4 restores classic DLAA edge magnitude so Edge Contrast / Epsilon are visible
    float4 edges = 4.0 * abs((left + right + top + bottom) - 4.0 * center);
    float edgesLum = LI(edges.rgb);
    float crushed = saturate(edgesLum * EdgeContrast - Epsilon);

    return float4(center.rgb, crushed);
}

float4 PS_Spatial(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float4 Center = LoadPF(uv, float2( 0,  0));
    float4 Left   = LoadPF(uv, float2(-1,  0));
    float4 Right  = LoadPF(uv, float2( 1,  0));
    float4 Up     = LoadPF(uv, float2( 0, -1));
    float4 Down   = LoadPF(uv, float2( 0,  1));

    // Short edge — talk §43 / TFU2 (5-tap H/V)
    // Horizontal blur along horizontal edges (detected via vertical high-pass)
    // Vertical blur along vertical edges (detected via horizontal high-pass)
    float4 combH = 2.0 * (Left + Right);
    float4 combV = 2.0 * (Up + Down);

    float4 centerDiffH = abs(combH - 4.0 * Center) / 4.0;
    float4 centerDiffV = abs(combV - 4.0 * Center) / 4.0;

    float4 blurredH = (combH + 2.0 * Center) / 6.0;
    float4 blurredV = (combV + 2.0 * Center) / 6.0;

    float lumH  = LI(centerDiffH.rgb);
    float lumV  = LI(centerDiffV.rgb);
    float lumHB = max(LI(blurredH.rgb), 1e-4);
    float lumVB = max(LI(blurredV.rgb), 1e-4);

    // Intensity-independent masks; epsilon scaled to edge magnitude domain
    float satAmountH = saturate((Lambda * lumH - Epsilon) / lumVB);
    float satAmountV = saturate((Lambda * lumV - Epsilon) / lumHB);

    float4 dlaa = Center;
    dlaa = lerp(dlaa, blurredH, satAmountV);
    // Second axis at half weight — matches proven ReShade ports; full weight over-softens
    // diagonals into new stair artifacts
    dlaa = lerp(dlaa, blurredV, satAmountH * 0.5);

    // Long edge sparse samples — talk §44 (reuse ±1 short taps + ±3.5/5.5/7.5)
    float4 HNegA = LoadPF(uv, float2(-3.5, 0));
    float4 HNegB = LoadPF(uv, float2(-5.5, 0));
    float4 HNegC = LoadPF(uv, float2(-7.5, 0));
    float4 HPosA = LoadPF(uv, float2( 3.5, 0));
    float4 HPosB = LoadPF(uv, float2( 5.5, 0));
    float4 HPosC = LoadPF(uv, float2( 7.5, 0));

    float4 VNegA = LoadPF(uv, float2(0, -3.5));
    float4 VNegB = LoadPF(uv, float2(0, -5.5));
    float4 VNegC = LoadPF(uv, float2(0, -7.5));
    float4 VPosA = LoadPF(uv, float2(0,  3.5));
    float4 VPosB = LoadPF(uv, float2(0,  5.5));
    float4 VPosC = LoadPF(uv, float2(0,  7.5));

    float4 avgBlurH = (Left + HNegA + HNegB + HNegC + Right + HPosA + HPosB + HPosC) / 8.0;
    float4 avgBlurV = (Up + VNegA + VNegB + VNegC + Down + VPosA + VPosB + VPosC) / 8.0;

    float longMaskH = saturate(avgBlurH.a * 2.0 - 1.0);
    float longMaskV = saturate(avgBlurV.a * 2.0 - 1.0);
    float longEdgeSep = abs(longMaskH - longMaskV);
    float longGate = longEdgeSep > LongEdgeThreshold ? 1.0 : 0.0;

    // Only one dominant axis, and only where the mask is actually strong
    [branch]
    if (longGate > 0.0 && max(longMaskH, longMaskV) > 0.15)
    {
        float lbH = LI(avgBlurH.rgb);
        float lbV = LI(avgBlurV.rgb);

        float cL = LI(Center.rgb);
        float lL = LI(Left.rgb);
        float rL = LI(Right.rgb);
        float uL = LI(Up.rgb);
        float dL = LI(Down.rgb);

        // Talk §45–47: construct a neighbor color whose luma matches the wide blur
        float4 clrH = Center;
        float tUp = SafeLumaT(lbH, uL, cL);
        float tDn = SafeLumaT(lbH, cL, dL);
        if (tUp > 0.0)
            clrH = lerp(Up, Center, tUp);
        else if (tDn > 0.0)
            clrH = lerp(Center, Down, tDn);

        float4 clrV = Center;
        float tLf = SafeLumaT(lbV, lL, cL);
        float tRt = SafeLumaT(lbV, cL, rL);
        if (tLf > 0.0)
            clrV = lerp(Left, Center, tLf);
        else if (tRt > 0.0)
            clrV = lerp(Center, Right, tRt);

        // Prefer the dominant axis only (noise rejection already requires H≠V)
        if (longMaskH > longMaskV)
            dlaa = lerp(dlaa, clrH, longMaskH);
        else
            dlaa = lerp(dlaa, clrV, longMaskV);
    }

    float3 outRgb = lerp(Center.rgb, dlaa.rgb, Strength);
    // Always write clean AA — debug overlays are applied in Resolve only so
    // HistoryTex never stores a debug frame.
    return float4(outRgb, 1.0);
}

float4 PS_Resolve(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 spatial = tex2Dlod(sSpatial, float4(uv, 0, 0)).rgb;

    // --- Debug views (display only; history still gets clean TexSpatial) ---
    if (ViewMode == 4) // Edge Alpha (last)
    {
        float a = tex2Dlod(sPreFilter, float4(uv, 0, 0)).a;
        return float4(a.xxx, 1.0);
    }

    if (ViewMode == 1 || ViewMode == 2) // Long / Short Mask overlay
    {
        float4 Center = LoadPF(uv, float2( 0,  0));
        float4 Left   = LoadPF(uv, float2(-1,  0));
        float4 Right  = LoadPF(uv, float2( 1,  0));
        float4 Up     = LoadPF(uv, float2( 0, -1));
        float4 Down   = LoadPF(uv, float2( 0,  1));

        float3 overlay = 0.0;

        if (ViewMode == 2) // Short Mask
        {
            float4 combH = 2.0 * (Left + Right);
            float4 combV = 2.0 * (Up + Down);
            float4 centerDiffH = abs(combH - 4.0 * Center) / 4.0;
            float4 centerDiffV = abs(combV - 4.0 * Center) / 4.0;
            float4 blurredH = (combH + 2.0 * Center) / 6.0;
            float4 blurredV = (combV + 2.0 * Center) / 6.0;
            float lumH  = LI(centerDiffH.rgb);
            float lumV  = LI(centerDiffV.rgb);
            float lumHB = max(LI(blurredH.rgb), 1e-4);
            float lumVB = max(LI(blurredV.rgb), 1e-4);
            float satH = saturate((Lambda * lumH - Epsilon) / lumVB);
            float satV = saturate((Lambda * lumV - Epsilon) / lumHB);
            overlay = float3(satH, satV, max(satH, satV));
        }
        else // Long Mask
        {
            float4 HNegA = LoadPF(uv, float2(-3.5, 0));
            float4 HNegB = LoadPF(uv, float2(-5.5, 0));
            float4 HNegC = LoadPF(uv, float2(-7.5, 0));
            float4 HPosA = LoadPF(uv, float2( 3.5, 0));
            float4 HPosB = LoadPF(uv, float2( 5.5, 0));
            float4 HPosC = LoadPF(uv, float2( 7.5, 0));
            float4 VNegA = LoadPF(uv, float2(0, -3.5));
            float4 VNegB = LoadPF(uv, float2(0, -5.5));
            float4 VNegC = LoadPF(uv, float2(0, -7.5));
            float4 VPosA = LoadPF(uv, float2(0,  3.5));
            float4 VPosB = LoadPF(uv, float2(0,  5.5));
            float4 VPosC = LoadPF(uv, float2(0,  7.5));

            float4 avgBlurH = (Left + HNegA + HNegB + HNegC + Right + HPosA + HPosB + HPosC) / 8.0;
            float4 avgBlurV = (Up + VNegA + VNegB + VNegC + Down + VPosA + VPosB + VPosC) / 8.0;
            float longMaskH = saturate(avgBlurH.a * 2.0 - 1.0);
            float longMaskV = saturate(avgBlurV.a * 2.0 - 1.0);
            float longEdgeSep = abs(longMaskH - longMaskV);
            float longGate = longEdgeSep > LongEdgeThreshold ? 1.0 : 0.0;
            overlay = float3(longMaskH * longGate, longMaskV * longGate, longEdgeSep);
        }

        float w = saturate(max(overlay.r, max(overlay.g, overlay.b)));
        return float4(lerp(spatial, overlay, w), 1.0);
    }

    if (!EnableTemporal || FRAME_COUNT <= 1)
        return float4(spatial, 1.0);

    // Sky / no usable depth: skip temporal
    float depth = bb::GetLinearizedDepth(uv);
    if (depth >= 0.98)
        return float4(spatial, 1.0);

    float2 velocity = MV_GetVelocity(uv);
    float2 histUV = uv + velocity;
    bool inside = all(saturate(histUV) == histUV);

    if (!inside)
        return float4(spatial, 1.0);

    float conf = saturate(MV_GetConfidence(uv));
    if (conf < 0.05)
        return float4(spatial, 1.0);

    float3 history = TAA_SampleHistoryCatmullRom(sHistoryTex, histUV, float2(BUFFER_WIDTH, BUFFER_HEIGHT)).rgb;

    float3 cur_c  = TAA_Compress(spatial);
    float3 hist_c = TAA_Compress(history);
    float3 cur_y  = RGBToYCoCg(cur_c);
    float3 hist_y = RGBToYCoCg(hist_c);

    float4 color_min, color_max;
    TAA_ComputeNeighborhoodVariance(sSpatial, uv, float4(spatial, 1.0), bb::PixelSize, color_min, color_max);

    float relax = 0.10 * conf;
    color_min -= relax;
    color_max += relax;

    float3 clipped = TAA_ClipToAABB(color_min.rgb, color_max.rgb, hist_y);

    float color_err = length(clipped - cur_y);
    float color_conf = saturate(1.0 - color_err * 4.0);

    float feedback = TemporalStability * lerp(0.55, 0.95, conf) * color_conf;

    if (ViewMode == 3) // Temporal Blend
        return float4(feedback.xxx, 1.0);

    float3 out_y = lerp(cur_y, clipped, feedback);
    float3 result = TAA_Resolve(YCoCgToRGB(out_y));
    return float4(result, 1.0);
}

float4 PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // Never store debug frames into history — keeps TAA clean when leaving View Mode
    if (ViewMode != 0)
        return tex2Dlod(sSpatial, float4(uv, 0, 0));

    return tex2Dlod(bb::BackBuffer, float4(uv, 0, 0));
}

//-----------------|
// :: Techniques ::|
//-----------------|

technique BaBa_DTLAA <
    ui_label = "BaBa: DTLAA";
    ui_tooltip = "Directionally Localized AA (Andreev) + optional TAA. Run BaBa_Flow before this when using temporal.";
>
{
    pass PreFilter
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_PreFilter;
        RenderTarget = TexPreFilter;
    }
    pass Spatial
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Spatial;
        RenderTarget = TexSpatial;
    }
    pass Resolve
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Resolve;
    }
    pass UpdateHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_UpdateHistory;
        RenderTarget = HistoryTex;
    }
}
