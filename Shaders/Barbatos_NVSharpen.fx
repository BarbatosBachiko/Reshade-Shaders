/*------------------.
| :: Description :: |
'-------------------/
// The MIT License(MIT)
//
// Copyright(c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,

// subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Barbatos NVSharpen
    
    Version 1.0
    Author: Barbatos Bachiko
    About: NIS Sharpen from NVIDIA Image Scaling SDK - v1.0.3 https://github.com/NVIDIAGameWorks/NVIDIAImageScaling/blob/35e13ba316c98eeecf16f37eae70ce88019911f6/NIS/NIS_Scaler.h

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
*/

#include "ReShade.fxh"

// Kernel configuration
static const int K = 2; // radius of 5x5 kernel
static const int SIZE = 5; // 2*K + 1
#define INDEX(x,y) ((y)*SIZE + (x))

// Sharpening parameters
static const float SharpStrengthMin = 0.0;
static const float SharpLimitMin = 0.3;
static const float SharpLimitScale = 0.0;
static const float HDRCompressionFactor = 1.0;
static const float kDetectRatio = 1.5; // ratio threshold
static const float kDetectThres = 0.01; // minimum edge strength

// NV12 support flag
#ifndef NIS_NV12_SUPPORT
#define NIS_NV12_SUPPORT 0
#endif

#ifndef HDR_BETA
#define HDR_BETA 0
#endif

// Filters
#define kSupportSize 5
#define kEps 1e-4f
#define kBaseEps 1e-4f

// Edge
#define kMinContrastRatio 2.0f
#define kRatioNorm 0.1f
#define kContrastBoost 0.5f

// Adaptative Sharpness
#define kSharpStartY 0.5f
#define kSharpScaleY 1.0f

// HDR
#define HDR_COMPRESSION_FACTOR 1.0
#define kEpsHDR (1e-4f * HDR_COMPRESSION_FACTOR * HDR_COMPRESSION_FACTOR)


/*---------------.
| :: Settings :: |
'---------------*/

uniform float SharpStrength <
    ui_type = "drag";
    ui_label = "Sharpness Strength";
    ui_min = 0.0;
    ui_max = 4.0;
    ui_step = 0.001;
    ui_category = "Sharpen";
> = 2.000;

namespace NEOSPACE
{
    texture ColorTex : COLOR;
    sampler sColor
    {
        Texture = ColorTex;
    };

/*----------------.
| :: Functions :: |
'----------------*/

    float getYLinear(float3 rgb)
    {
        return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
    }

    // YUV -> RGB conversion (NV12 support)
    float3 YUVtoRGB(float3 yuv)
    {
        float y = yuv.x - 16.0f / 255.0f;
        float u = yuv.y - 128.0f / 255.0f;
        float v = yuv.z - 128.0f / 255.0f;
        float3 rgb;
        rgb.x = saturate(1.164f * y + 1.596f * v);
        rgb.y = saturate(1.164f * y - 0.392f * u - 0.813f * v);
        rgb.z = saturate(1.164f * y + 2.017f * u);
        return rgb;
    }

    float CalcLTIFast(float y[5])
    {
        float a_min = min(min(y[0], y[1]), y[2]);
        float a_max = max(max(y[0], y[1]), y[2]);

        float b_min = min(min(y[2], y[3]), y[4]);
        float b_max = max(max(y[2], y[3]), y[4]);

        float a_cont = a_max - a_min;
        float b_cont = b_max - b_min;

        float cont_ratio = max(a_cont, b_cont) / (min(a_cont, b_cont) + kEps);
        return (1.0 - saturate((cont_ratio - kMinContrastRatio) * kRatioNorm)) * kContrastBoost;
    }

    float EvalUSM(float pxl[5], float sharpnessStrength, float sharpnessLimit)
    {
        float y_usm = -0.6001 * pxl[1] + 1.2002 * pxl[2] - 0.6001 * pxl[3];
        y_usm *= sharpnessStrength;
        y_usm = clamp(y_usm, -sharpnessLimit, sharpnessLimit);
        y_usm *= CalcLTIFast(pxl);

        return y_usm;
    }

    // Direcional USM 
    float4 GetDirUSM(float p[SIZE * SIZE])
    {
        float yC = p[INDEX(2, 2)];
        float scaleY = 1.0 - saturate((yC - kSharpStartY) * kSharpScaleY);
        float sharpStrength = scaleY * SharpStrength + SharpStrengthMin;
        float sharpLimit = (scaleY * SharpLimitScale + SharpLimitMin) * yC;
        float4 usm;
    
        float interp0Deg[5], interp90Deg[5], interp45Deg[5], interp135Deg[5];
    
         // 0° 
        interp0Deg[0] = p[INDEX(0, 2)];
        interp0Deg[1] = p[INDEX(1, 2)];
        interp0Deg[2] = p[INDEX(2, 2)];
        interp0Deg[3] = p[INDEX(3, 2)];
        interp0Deg[4] = p[INDEX(4, 2)];
        usm.x = EvalUSM(interp0Deg, sharpStrength, sharpLimit);

        // 90° 
        interp90Deg[0] = p[INDEX(2, 0)];
        interp90Deg[1] = p[INDEX(2, 1)];
        interp90Deg[2] = p[INDEX(2, 2)];
        interp90Deg[3] = p[INDEX(2, 3)];
        interp90Deg[4] = p[INDEX(2, 4)];
        usm.y = EvalUSM(interp90Deg, sharpStrength, sharpLimit);

        // 45° 
        interp45Deg[0] = p[INDEX(1, 1)];
        interp45Deg[1] = lerp(p[INDEX(2, 1)], p[INDEX(1, 2)], 0.5);
        interp45Deg[2] = p[INDEX(2, 2)];
        interp45Deg[3] = lerp(p[INDEX(3, 2)], p[INDEX(2, 3)], 0.5);
        interp45Deg[4] = p[INDEX(3, 3)];
        usm.z = EvalUSM(interp45Deg, sharpStrength, sharpLimit);

         // 135° 
        interp135Deg[0] = p[INDEX(3, 1)];
        interp135Deg[1] = lerp(p[INDEX(3, 2)], p[INDEX(2, 1)], 0.5);
        interp135Deg[2] = p[INDEX(2, 2)];
        interp135Deg[3] = lerp(p[INDEX(2, 3)], p[INDEX(1, 2)], 0.5);
        interp135Deg[4] = p[INDEX(1, 3)];
        usm.w = EvalUSM(interp135Deg, sharpStrength, sharpLimit);

        return usm;
    }

    float4 GetEdgeMap(float p[SIZE * SIZE])
    {
        float g0 = abs(p[INDEX(0, 2)] + p[INDEX(0, 1)] + p[INDEX(0, 0)] - p[INDEX(2, 2)] - p[INDEX(2, 1)] - p[INDEX(2, 0)]);
        float g45 = abs(p[INDEX(1, 2)] + p[INDEX(0, 2)] + p[INDEX(0, 1)] - p[INDEX(2, 1)] - p[INDEX(2, 0)] - p[INDEX(1, 0)]);
        float g90 = abs(p[INDEX(2, 0)] + p[INDEX(1, 0)] + p[INDEX(0, 0)] - p[INDEX(2, 2)] - p[INDEX(1, 2)] - p[INDEX(0, 2)]);
        float g135 = abs(p[INDEX(1, 0)] + p[INDEX(2, 0)] + p[INDEX(2, 1)] - p[INDEX(0, 1)] - p[INDEX(0, 2)] - p[INDEX(1, 2)]);

        float g0_90_max = max(g0, g90);
        float g0_90_min = min(g0, g90);
        float g45_135_max = max(g45, g135);
        float g45_135_min = min(g45, g135);

        if (g0_90_max + g45_135_max == 0)
            return float4(0, 0, 0, 0);

        float e0_90 = min(g0_90_max / (g0_90_max + g45_135_max), 1.0);
        float e45_135 = 1.0 - e0_90;

        bool c0_90 = (g0_90_max > g0_90_min * kDetectRatio) && (g0_90_max > kDetectThres) && (g0_90_max > g45_135_min);
        bool c45_135 = (g45_135_max > g45_135_min * kDetectRatio) && (g45_135_max > kDetectThres) && (g45_135_max > g0_90_min);
        bool cg0_90 = (g0_90_max == g0);
        bool cg45_135 = (g45_135_max == g45);

        float f_e0_90 = (c0_90 && c45_135) ? e0_90 : 1.0;
        float f_e45_135 = (c0_90 && c45_135) ? e45_135 : 1.0;

        float w0 = (c0_90 && cg0_90) ? f_e0_90 : 0.0;
        float w90 = (c0_90 && !cg0_90) ? f_e0_90 : 0.0;
        float w45 = (c45_135 && cg45_135) ? f_e45_135 : 0.0;
        float w135 = (c45_135 && !cg45_135) ? f_e45_135 : 0.0;

        return float4(w0, w90, w45, w135);
    }

    float4 PS_NIS(float4 pos : SV_Position, float2 tex : TEXCOORD) : SV_Target
    {
        float lum[SIZE * SIZE];
    
    [unroll]
        for (int y = -K; y <= K; ++y)
        {
        [unroll]
            for (int x = -K; x <= K; ++x)
            {
#if NIS_NV12_SUPPORT
            // NV12 path: sample YUV and convert
            float2 coord = tex + float2(x, y) * ReShade::PixelSize.xy;
            float yv = tex2D(ReShade::BackBuffer, coord).r;
            float2 uv = tex2D(ReShade::BackBuffer, coord).gb;
            float3 yuv = float3(yv, uv);
            float3 rgb = YUVtoRGB(yuv);
            lum[INDEX(x+K, y+K)] = getYLinear(rgb);
#else
                float4 c = tex2D(ReShade::BackBuffer, tex + float2(x, y) * ReShade::PixelSize.xy);
                lum[INDEX(x+K, y+K)] = getYLinear(c.rgb);
#endif
            }
        }

        float4 usm = GetDirUSM(lum);
        float4 w = GetEdgeMap(lum);
        float usmY = dot(usm, w);

        float4 orig = tex2D(sColor, tex);
        if (HDR_BETA)
        {
            float3 centerRGB = tex2D(ReShade::BackBuffer, tex).rgb;
            float oldY = getYLinear(centerRGB);
            float dynamicEps = 1e-4f * HDRCompressionFactor * HDRCompressionFactor;
            float newY = max(oldY + usmY, 0.0);
            float corr = (newY * newY + dynamicEps) / (oldY * oldY + dynamicEps);
            orig.rgb *= corr;
        }
        else
        {
            orig.rgb += usmY;
        }
        return orig;
    }

    technique Barbatos_NVSharpen <
		ui_label = "Barbatos: NVSharpen";>
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_NIS;
        }
    }
}
