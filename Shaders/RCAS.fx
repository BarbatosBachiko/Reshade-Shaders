/*------------------.
| :: Description :: |
'-------------------/

   RCAS (Version 1.0)

    Author: Barbatos Bachiko
    License: Copyright © 2024 Jakob Wapenhensch from B A D   U P S C A L I N G   R E P L A C E R (https://creativecommons.org/licenses/by-nc-sa/4.0/)
    -------------------------------------------------------------------------------------------------------------------------------------------------
    About: The shader is a Adaptive Sharpening filter.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
	
    Version 1.0

*/

uniform float sharpness <
    ui_type = "slider";
    ui_label = "Sharpness Intensity"; 
    ui_tooltip = "Adjust Sharpness Intensity"; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.15; 
> = 0.6;

#define FSR_RCAS_LIMIT (0.18-(1.0/16.0))

#include "ReShade.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

texture2D BackBufferTex : COLOR;
sampler2D BackBuffer
{
    Texture = BackBufferTex;
};

/*----------------.
| :: Functions :: |
'----------------*/

void FsrEasuCon(
    out float4 con0,
    out float4 con1,
    out float4 con2,
    out float4 con3,
    float2 inputViewportInPixels,
    float2 inputSizeInPixels,
    float2 outputSizeInPixels
)
{
    con0 = float4(
        float2(inputViewportInPixels / outputSizeInPixels),
        float2(0.5 * inputViewportInPixels / outputSizeInPixels - 0.5) - 0.5
    );
    con1 = float4(1, 1, 1, -1) / inputSizeInPixels.xyxy;
    con2 = float4(-1, 2, 1, 2) / inputSizeInPixels.xyxy;
    con3 = float4(0, 4, 0, 0) / inputSizeInPixels.xyxy;
}

void FsrEasuTapF(
    inout float3 aC,
    inout float aW,
    float2 off,
    float2 dir,
    float2 len,
    float lob,
    float clp,
    float3 c
)
{
    float2 v = float2(dot(off, dir), dot(off, float2(-dir.y, dir.x)));
    v *= len;
    float d2 = min(dot(v, v), clp);
    float wB = .4 * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 1.5625 * wB - 0.5625;
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void FsrEasuSetF(
    inout float2 dir,
    inout float len,
    float w,
    float lA, float lB, float lC, float lD, float lE
)
{
    float lenX = max(abs(lD - lC), abs(lC - lB));
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) / lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;
    float lenY = max(abs(lE - lC), abs(lC - lA));
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) / lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

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
    float3 b = FsrRcasLoadF(samp, ip + float2(0, -1)).rgb;
    float3 d = FsrRcasLoadF(samp, ip + float2(-1, 0)).rgb;
    float3 e = FsrRcasLoadF(samp, ip).rgb;
    float3 f = FsrRcasLoadF(samp, ip + float2(1, 0)).rgb;
    float3 h = FsrRcasLoadF(samp, ip + float2(0, 1)).rgb;
    float bL = b.g + 0.5 * (b.b + b.r);
    float dL = d.g + 0.5 * (d.b + d.r);
    float eL = e.g + 0.5 * (e.b + e.r);
    float fL = f.g + 0.5 * (f.b + f.r);
    float hL = h.g + 0.5 * (h.b + h.r);
    float nz = 0.25 * (bL + dL + fL + hL) - eL;
    nz = clamp(
        abs(nz) / (
             max(max(bL, dL), max(eL, max(fL, hL)))
            - min(min(bL, dL), min(eL, min(fL, hL)))
        ),
        0.,
        1.
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
    return (lobe * (b + d + h + f) + e) / (4. * lobe + 1.);
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
    {
        return Rcas(BackBuffer, texcoord, sharpness);
    }
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