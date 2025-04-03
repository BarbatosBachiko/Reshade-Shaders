/*------------------.
| :: Description :: |
'-------------------/

// LICENSE
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

    RCAS
    
    Version 1.3
    Author: Barbatos Bachiko
    About: Adaptive Sharpening filter.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.3
    x Update FsrRcasF.

*/
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
> = 0.4;

    uniform float contrastThreshold <
    ui_type = "slider";
    ui_label = "Contrast Threshold";
    ui_tooltip = "Threshold to enable sharpening on low-contrast areas";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_default = 0.0;
> = 0.0;

#define FSR_RCAS_LIMIT (0.18 - (1.0/16.0))
/*----------------.
| :: Functions :: |
'----------------*/

float4 FsrRcasLoadF(sampler2D samp, float2 p)
{
  return tex2Dlod(samp, float4(p * ReShade::PixelSize, 0, 0));
}

float FsrRcasCon(float sharpness)
{
  return sharpness > 0.0 ? exp2(sharpness) : 0.0;
}

float3 FsrRcasF(float2 ip, float con)
{
    float2 pixelSize = ReShade::PixelSize;

    // Uses offsets relative to screen size
    float3 b = FsrRcasLoadF(ReShade::BackBuffer, ip + float2(0, -1) * pixelSize * ReShade::ScreenSize.xy).rgb;
    float3 d = FsrRcasLoadF(ReShade::BackBuffer, ip + float2(-1, 0) * pixelSize * ReShade::ScreenSize.xy).rgb;
    float3 e = FsrRcasLoadF(ReShade::BackBuffer, ip).rgb;
    float3 f = FsrRcasLoadF(ReShade::BackBuffer, ip + float2(1, 0) * pixelSize * ReShade::ScreenSize.xy).rgb;
    float3 h = FsrRcasLoadF(ReShade::BackBuffer, ip + float2(0, 1) * pixelSize * ReShade::ScreenSize.xy).rgb;

    // Luma times 2.
    float bL = mad(.5, b.b + b.r, b.g);
    float dL = mad(.5, d.b + d.r, d.g);
    float eL = mad(.5, e.b + e.r, e.g);
    float fL = mad(.5, f.b + f.r, f.g);
    float hL = mad(.5, h.b + h.r, h.g);

    // Contrast check to avoid over-sharpening low contrast areas
    float diffBD = max(abs(bL - eL), abs(dL - eL));
    float diffFH = max(abs(fL - eL), abs(hL - eL));
    float maxDiff = max(diffBD, diffFH);
    if (maxDiff < contrastThreshold)
    {
        return e;
    }

    // Noise detection
    float nz = (abs(bL - eL) + abs(dL - eL) + abs(fL - eL) + abs(hL - eL)) * 0.25;
    float range = max(max(bL, dL), max(eL, max(fL, hL))) - min(min(bL, dL), min(eL, min(fL, hL)));
    nz = (range > 0.001) ? (nz / range) : 0.0;
    nz = 1.0 - (nz * 0.5);

    // Calculation of minimum and maximum values ​​around the central pixel
    float3 mn4 = min(b, min(f, h));
    float3 mx4 = max(b, max(f, h));

    // Constants to limit sharpness
    float2 peakC = float2(1., -4.);
    float3 hitMin = mn4 / (4. * mx4);
    float3 hitMax = (peakC.x - mx4) / mad(4., mn4, peakC.y);
    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.)) * con;

    // Applying sharpness while preserving the original tone
    return mad(lobe, b + d + h + f, e) / mad(4., lobe, 1.);
}

float4 Rcas(sampler2D samp, float2 texcoord, float sharpness)
{
   float2 fragCoord = texcoord * tex2Dsize(samp);
   float con = FsrRcasCon(sharpness);
   float3 col = FsrRcasF(fragCoord, con);
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
