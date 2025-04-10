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
    
    Version 1.4
    Author: Barbatos Bachiko
    About: FSR - [RCAS] ROBUST CONTRAST ADAPTIVE SHARPENING

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.4
    + review
*/

#include "ReShade.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform float sharpness <
    ui_type = "slider";
    ui_label = "Sharpness Intensity";
    ui_tooltip = "Controls the strength of the sharpening effect (0.0 to 1.0)";
    ui_min = 0.0;
    ui_max = 1.2;
    ui_step = 0.01;
    ui_default = 0.15;
> = 0.8;

uniform bool enableDenoise <
    ui_label = "Enable Noise Reduction";
    ui_tooltip = "Reduces sharpening in noisy areas";
    ui_default = true;
> = true;


#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))

/*----------------.
| :: Functions :: |
'----------------*/

float4 FsrRcasLoadF(float2 p)
{
    return tex2Dlod(ReShade::BackBuffer, float4(p, 0, 0));
}

float FsrRcasCon(float sharpness)
{
    return sharpness; 
}

float3 FsrRcasInput(float3 rgb)
{
    return rgb;
}

float3 FsrRcasF(float2 ip, float con)
{
    float2 pixelSize = ReShade::PixelSize;
    
    float3 b = FsrRcasInput(FsrRcasLoadF(ip + float2(0, -1) * pixelSize).rgb);
    float3 d = FsrRcasInput(FsrRcasLoadF(ip + float2(-1, 0) * pixelSize).rgb);
    float3 e = FsrRcasInput(FsrRcasLoadF(ip).rgb);
    float3 f = FsrRcasInput(FsrRcasLoadF(ip + float2(1, 0) * pixelSize).rgb);
    float3 h = FsrRcasInput(FsrRcasLoadF(ip + float2(0, 1) * pixelSize).rgb);

    // Calculate luma
    float bL = dot(b, float3(0.2126, 0.7152, 0.0722));
    float dL = dot(d, float3(0.2126, 0.7152, 0.0722));
    float eL = dot(e, float3(0.2126, 0.7152, 0.0722));
    float fL = dot(f, float3(0.2126, 0.7152, 0.0722));
    float hL = dot(h, float3(0.2126, 0.7152, 0.0722));

    // Contrast check 
    float diffBD = max(abs(bL - eL), abs(dL - eL));
    float diffFH = max(abs(fL - eL), abs(hL - eL));
    float maxDiff = max(diffBD, diffFH);
    
    // Noise detection 
    float nz = 0.25 * (bL + dL + fL + hL) - eL;
    float rangeMax = max(max(max(bL, dL), max(fL, hL)), eL);
    float rangeMin = min(min(min(bL, dL), min(fL, hL)), eL);
    nz = saturate(abs(nz) / (rangeMax - rangeMin));
    nz = -0.5 * nz + 1.0;

    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));

    float3 hitMin = mn4 / (4.0 * mx4);
    float3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0);
    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * con;
    
    if (enableDenoise)
    {
        lobe *= nz;
    }

    float rcpL = rcp(4.0 * lobe + 1.0);
    return (e + lobe * (b + d + f + h)) * rcpL;
}

float4 RCAS_Pass(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float con = FsrRcasCon(sharpness);
    float3 color = FsrRcasF(texcoord, con);
    return float4(color, tex2D(ReShade::BackBuffer, texcoord).a);
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique RCAS
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = RCAS_Pass;
    }
}
