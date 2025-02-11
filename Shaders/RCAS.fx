/*------------------.
| :: Description :: |
'-------------------/

   RCAS (Version 1.1.1)

    Author: Barbatos Bachiko
    License: Copyright © 2024 Jakob Wapenhensch from B A D   U P S C A L I N G   R E P L A C E R (https://creativecommons.org/licenses/by-nc-sa/4.0/)
    About: Adaptive Sharpening filter.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.1.1
    + Code Clean

*/
namespace RCASisCool
{
#define FSR_RCAS_LIMIT (0.18 - (1.0/16.0))
#include "ReShade.fxh"
/*---------------.
| :: Settings :: |
'---------------*/

    uniform float sharpness <
    ui_type = "slider";
    ui_label = "Sharpness Intensity";
    ui_tooltip = "Adjust Sharpness Intensity";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_default = 0.15;
> = 0.6;

    uniform float contrastThreshold <
    ui_type = "slider";
    ui_label = "Contrast Threshold";
    ui_tooltip = "Threshold to enable sharpening on low-contrast areas";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_default = 0.0;
> = 0.0;

/*----------------.
| :: Functions :: |
'----------------*/

    float4 FsrRcasLoadF(sampler2D samp, float2 p)
    {
        return tex2Dlod(samp, float4(p * ReShade::PixelSize, 0, 0));
    }

    float FsrRcasCon(float sharpness)
    {
        return exp2(sharpness);
    }

    float3 FsrRcasF(
    sampler2D samp, float2 ip, float con
)
    {
    // Loads neighboring pixels
        float3 b = FsrRcasLoadF(samp, ip + float2(0, -1)).rgb;
        float3 d = FsrRcasLoadF(samp, ip + float2(-1, 0)).rgb;
        float3 e = FsrRcasLoadF(samp, ip).rgb;
        float3 f = FsrRcasLoadF(samp, ip + float2(1, 0)).rgb;
        float3 h = FsrRcasLoadF(samp, ip + float2(0, 1)).rgb;
    
    // Simple luminance calculation for contrast detection
        float bL = b.g + 0.5 * (b.b + b.r);
        float dL = d.g + 0.5 * (d.b + d.r);
        float eL = e.g + 0.5 * (e.b + e.r);
        float fL = f.g + 0.5 * (f.b + f.r);
        float hL = h.g + 0.5 * (h.b + h.r);
    
    // Check if there is enough contrast to apply sharpening
        float diffBD = max(abs(bL - eL), abs(dL - eL));
        float diffFH = max(abs(fL - eL), abs(hL - eL));
        float maxDiff = max(diffBD, diffFH);
        if (maxDiff < contrastThreshold)
        {
            return e;
        }
    
        float nz = 0.25 * (bL + dL + fL + hL) - eL;
        nz = clamp(
        abs(nz) / (
             max(max(bL, dL), max(eL, max(fL, hL)))
            - min(min(bL, dL), min(eL, min(fL, hL)))
        ),
        0.0,
        1.0
    );
        nz = 1.0 - 0.5 * nz;
    
        float3 mn4 = min(b, min(f, h));
        float3 mx4 = max(b, max(f, h));
        float2 peakC = float2(1.0, -4.0);
        float3 hitMin = mn4 / (4.0 * mx4);
        float3 hitMax = (peakC.x - mx4) / (4.0 * mn4 + peakC.y);
        float3 lobeRGB = max(-hitMin, hitMax);
        float lobe = max(
        -FSR_RCAS_LIMIT,
        min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)
    ) * con;
        lobe *= nz;
    
        return (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
    }

    float4 Rcas(sampler2D samp, float2 texcoord, float sharpness)
    {
        float2 fragCoord = texcoord * tex2Dsize(samp);
        float con = FsrRcasCon(sharpness);
        float3 col = FsrRcasF(samp, fragCoord, con);
        return float4(col, 0);
    }

    float4 Out(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        return Rcas(ReShade::BackBuffer, texcoord, sharpness);
    }

/*-----------------.
| :: Techniques :: |
'-----------------*/

    technique RCAS
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = Out;
        }
    }
}
