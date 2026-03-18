/*Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
/*-------------------------------------------------|
| ::                   MiAO                     :: |
'--------------------------------------------------|
| Version: 1.5                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: Simple ambient occlusion with repur-|
| posed content from FidelityFX CACAO              |
'-------------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "BaBa_MV.fxh"
#include "BaBa_ColorSpace.fxh"

//--------------------|
// :: Preprocessor :: |
//--------------------|
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))

static const float2 BUFFER_DIM = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
static const float2 BUFFER_RCP = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
static const float2 R2 = float2(0.75487766624669276005, 0.5698402909980532659114);
#define PI 3.1415927
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

//----------|
// :: UI :: |
//----------|
uniform float ShadowPow <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Intensity";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.01;
    ui_tooltip = "Global strength of the ambient occlusion effect.";
> = 1.5;

uniform float Radius <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Radius";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.01;
    ui_tooltip = "World-space radius of the ambient occlusion effect.";
> = 5.0;

uniform float EffectShadowClamp <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Clamp";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Limits the maximum amount of occlusion to prevent excessive darkening.";
> = 0.98;

uniform int QualityLevel <
    ui_type = "combo";
    ui_label = "Quality";
    ui_category = "Performance & Quality";
    ui_items = "Lowest (3 taps)\0Low (5 taps)\0Medium (12 taps)\0High (20 taps)\0Very high (31 taps)\0";
    ui_tooltip = "Controls the number of samples used for AO calculation.";
> = 2;

uniform float RenderScale <
    ui_type = "drag";
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Renders AO at a lower resolution for better performance, then upscales it.";
> = 0.8;

uniform bool EnableTemporal <
    ui_category = "Performance & Quality";
    ui_label = "Reduce Noise (TAA)";
    ui_tooltip = "Reduces flickering and noise using temporal anti-aliasing";
> = true;

uniform bool EnableDistantRadius <
    ui_category = "Advanced Options";
    ui_label = "Enable Distant Radius";
> = false;

uniform float EffectHorizonAngleThreshold <
    ui_type = "drag";
    ui_category = "Advanced Options";
    ui_label = "Horizon Angle Threshold";
    ui_min = 0.0; ui_max = 0.5; ui_step = 0.001;
    ui_tooltip = "Limits errors on slopes and caused by insufficient geometry tessellation.";
> = 0.04;

uniform float FadeOutFrom <
    ui_type = "drag";
    ui_category = "Advanced Options";
    ui_label = "Fade Out Start";
    ui_min = 1.0; ui_max = 500.0; ui_step = 1.0;
    ui_tooltip = "The distance at which the AO effect begins to fade out.";
> = 50.0;

uniform float FadeOutTo <
    ui_type = "drag";
    ui_category = "Advanced Options";
    ui_label = "Fade Out End";
    ui_min = 1.0; ui_max = 550.0; ui_step = 1.0;
    ui_tooltip = "The distance at which the AO effect has completely faded out.";
> = 1.0;

uniform float FOV <
    ui_type = "slider";
    ui_category = "Advanced Options";
    ui_label = "Vertical FOV";
    ui_min = 30.0; ui_max = 120.0;
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 75.0;

uniform int DebugView <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "Debug View";
    ui_items = "None\0Raw SSAO\0View-space Normals\0";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;
static const int g_TapCounts[5] = { 3, 5, 12, 20, 31 };
static const float DEG2RAD = 0.017453292; // PI / 180.0
#define TWO_PI 6.2831854

//----------------|
// :: Textures :: |
//----------------|
#ifndef USE_HILBERT_LUT
#define USE_HILBERT_LUT 1 
#endif

texture texPrevLuma
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R8;
};
sampler sPrevLuma
{
    Texture = texPrevLuma;
};

namespace MiAO150
{
    texture DEPTH
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
    };
    sampler sDEPTH
    {
        Texture = DEPTH;
    };

    texture NORMALS
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sNORMALS
    {
        Texture = NORMALS;
    };

    texture AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
        MipLevels = 4;
    };
    sampler sAO
    {
        Texture = AO;
    };

    texture History0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory0
    {
        Texture = History0;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory1
    {
        Texture = History1;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture RS_Prev
    {
        Width = 1;
        Height = 1;
        Format = R16F;
    };
    sampler sRS_Prev
    {
        Texture = RS_Prev;
        MagFilter = POINT;
        MinFilter = POINT;
    };

#if USE_HILBERT_LUT
    texture texHilbertLUT < source = "Barbatos_Hilbert_RGB.png"; >
    {
        Width = 64;
        Height = 64;
        Format = RGBA8;
    };
    sampler sHilbertLUT
    {
        Texture = texHilbertLUT;
        AddressU = Wrap;
        AddressV = Wrap;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };
#endif
    
//----------------|
// :: Functions ::|
//----------------|
    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };

#if !USE_HILBERT_LUT
    float hilbert(float2 p, int level)
    {
        float d = 0;
        for (int k = 0; k < level; k++)
        {
            int n = level - k - 1;
            float n_pow2 = exp2(n);
            float2 r = fmod(floor(p / n_pow2), 2.0);
            float term = r.y + r.x * (3.0 - 2.0 * r.y);
            d += term * exp2(2 * n);
            if (r.y < 0.5)
            {
                if (r.x > 0.5)
                {
                    p = n_pow2 - 1.0 - p;
                }
                p = p.yx;
            }
        }
        return d;
    }

    uint HilbertIndex(uint x, uint y)
    {
        return (uint)hilbert(float2(x % 64, y % 64), 6);
    }
#endif

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        uint index;
#if USE_HILBERT_LUT
        float4 encodedVal = tex2Dfetch(sHilbertLUT, int2(pixCoord.x % 64, pixCoord.y % 64));
        uint high_byte = (uint) (encodedVal.r * 255.0 + 0.1);
        uint low_byte = (uint) (encodedVal.g * 255.0 + 0.1);
        index = (high_byte * 256) + low_byte;
#else
        index = HilbertIndex(pixCoord.x, pixCoord.y);
#endif
        index += 288 * (temporalIndex % 64);
        return frac(0.5 + index * R2);
    }

    float ScreenSpaceToViewSpaceDepth(float screenDepth)
    {
        return screenDepth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }

    float3 DepthBufferUVToViewSpace(float2 pos, float viewspaceDepth, float2 pScale)
    {
        float3 ret;
        float2 ndc = pos * 2.0 - 1.0;
        ndc.y = -ndc.y;
        
        ret.xy = (pScale * ndc) * viewspaceDepth;
        ret.z = viewspaceDepth;
        return ret;
    }

    float4 CalculateEdges(const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ)
    {
        float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
        float4 edgesLRTBSlopeAdjusted = edgesLRTB + edgesLRTB.yxwz;
        edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
        return saturate(1.3 - (edgesLRTB * 25.0) / centerZ);
    }

    float3 CalculateNormal(const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
    {
        float4 acceptedNormals = float4(edgesLRTB.x * edgesLRTB.z, edgesLRTB.z * edgesLRTB.y, edgesLRTB.y * edgesLRTB.w, edgesLRTB.w * edgesLRTB.x);
        pixLPos = normalize(pixLPos - pixCenterPos);
        pixRPos = normalize(pixRPos - pixCenterPos);
        pixTPos = normalize(pixTPos - pixCenterPos);
        pixBPos = normalize(pixBPos - pixCenterPos);
        float3 pixelNormal = float3(0, 0, -0.0005);
        pixelNormal += (acceptedNormals.x) * cross(pixLPos, pixTPos);
        pixelNormal += (acceptedNormals.y) * cross(pixTPos, pixRPos);
        pixelNormal += (acceptedNormals.z) * cross(pixRPos, pixBPos);
        pixelNormal += (acceptedNormals.w) * cross(pixBPos, pixLPos);
        return normalize(pixelNormal);
    }

    static const float K_Weights[31] =
    {
    2.87855, 3.89337, 2.03928, 2.57912, 2.82997, 2.49643, 1.65273, 1.14063,
    0.93697, 0.95761, 1.14153, 1.45707, 1.20977, 1.51199, 1.65217, 0.95377,
    -0.39005, 0.92968, 1.32165, 0.97041, 0.59024, -0.03566, -1.72452, 1.28163,
    2.27953, 0.24276, 1.67147, 2.05600, 2.10187, 0.73659, 1.63937
    };
    static const float K_MipLevels[31] =
    {
     -2.00, -1.50, -1.00, -1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20,
     -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
    };
    float4 GetSamplePattern(int index, float2 noise)
    {
        float2 offset = frac(R2 * float(index) + noise);
        float radius = sqrt(offset.x);
        float angle = offset.y * TWO_PI;

        float2 samplePos;
        sincos(angle, samplePos.y, samplePos.x);
        samplePos *= radius;
        index = clamp(index, 0, 30);
    
        return float4(samplePos, K_Weights[index], K_MipLevels[index]);
    }

    float CalculatePixelObscurance(float3 pixelNormal, float3 hitDelta, float falloffCalcMulSq)
    {
        float lengthSq = dot(hitDelta, hitDelta);
        float invLength = rsqrt(lengthSq);
        float NdotD = dot(pixelNormal, hitDelta) * invLength;
    
        float falloffMult = max(0.0, lengthSq * falloffCalcMulSq + 1.0);
        return max(0, NdotD - EffectHorizonAngleThreshold) * falloffMult;
    }

    void SSAOTap(
    inout float obscuranceSum, inout float weightSum,
    const int tapIndex, const float2x2 rotScale, const float3 pixCenterPos,
    const float3 pixelNormal, const float2 depthBufferUV, const float falloffCalcMulSq,
    const float2 noise, const float2 pScale)
    {
        float4 newSample = GetSamplePattern(tapIndex, noise);
        float2 sampleOffset = mul(rotScale, newSample.xy);
        float weightMod = newSample.z;

        float2 samplingUV = sampleOffset * BUFFER_RCP + depthBufferUV;
        samplingUV = saturate(samplingUV);
        float viewspaceSampleZ1 = GetLod(sDEPTH, samplingUV).r;
        float3 hitPos1 = DepthBufferUVToViewSpace(samplingUV, viewspaceSampleZ1, pScale);
        float3 hitDelta1 = hitPos1 - pixCenterPos;
        float obscurance1 = CalculatePixelObscurance(pixelNormal, hitDelta1, falloffCalcMulSq);
        float weight1 = 1.0;
    
        obscuranceSum += obscurance1 * weight1 * weightMod;
        weightSum += weight1 * weightMod;

        float2 samplingUV2 = -sampleOffset * BUFFER_RCP + depthBufferUV;
        samplingUV2 = saturate(samplingUV2);
        float viewspaceSampleZ2 = GetLod(sDEPTH, samplingUV2).r;
        float3 hitPos2 = DepthBufferUVToViewSpace(samplingUV2, viewspaceSampleZ2, pScale);
        float3 hitDelta2 = hitPos2 - pixCenterPos;
        float obscurance2 = CalculatePixelObscurance(pixelNormal, hitDelta2, falloffCalcMulSq);
        float weight2 = 1.0;

        obscuranceSum += obscurance2 * weight2 * weightMod;
        weightSum += weight2 * weightMod;
    }

    float SampleHistoryCatmullRom_Scalar(sampler sInput, float2 uv, float2 texSize)
    {
        float2 samplePos = uv * texSize;
        float2 texPos1 = floor(samplePos - 0.5) + 0.5;
        float2 f = samplePos - texPos1;

        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);

        float2 w12 = w1 + w2;
        float2 offset12 = w2 / (w1 + w2);

        float2 texPos0 = texPos1 - 1.0;
        float2 texPos3 = texPos1 + 2.0;
        float2 texPos12 = texPos1 + offset12;

        texPos0 /= texSize;
        texPos3 /= texSize;
        texPos12 /= texSize;

        float result = 0.0;
        result += GetLod(sInput, float2(texPos0.x, texPos0.y)).r * w0.x * w0.y;
        result += GetLod(sInput, float2(texPos12.x, texPos0.y)).r * w12.x * w0.y;
        result += GetLod(sInput, float2(texPos3.x, texPos0.y)).r * w3.x * w0.y;

        result += GetLod(sInput, float2(texPos0.x, texPos12.y)).r * w0.x * w12.y;
        result += GetLod(sInput, float2(texPos12.x, texPos12.y)).r * w12.x * w12.y;
        result += GetLod(sInput, float2(texPos3.x, texPos12.y)).r * w3.x * w12.y;

        result += GetLod(sInput, float2(texPos0.x, texPos3.y)).r * w0.x * w3.y;
        result += GetLod(sInput, float2(texPos12.x, texPos3.y)).r * w12.x * w3.y;
        result += GetLod(sInput, float2(texPos3.x, texPos3.y)).r * w3.x * w3.y;

        return max(result, 0.0);
    }

/*--------------------.
| :: Vertex Shader :: |
'--------------------*/

    void VS_MiAO(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ? 2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
        
        float fov_rad = FOV * DEG2RAD;
        float y = tan(fov_rad * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }

    void VS_MiAO_Even(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_MiAO(id, outStruct);
        if ((FRAME_COUNT % 2) != 0) outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 1.0);
    }

    void VS_MiAO_Odd(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_MiAO(id, outStruct);
        if ((FRAME_COUNT % 2) == 0) outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 1.0);
    }

/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/

    void PS_Prepare(VS_OUTPUT input, out float4 outDepth : SV_Target0, out float4 outNormal : SV_Target1)
    {
        float2 uv = input.uv;
        float linear_depth = GetDepth(uv);
        outDepth = ScreenSpaceToViewSpaceDepth(linear_depth);

        float2 p = BUFFER_RCP;
        float3 p_c = DepthBufferUVToViewSpace(uv, outDepth.r, input.pScale);
        float3 p_l = DepthBufferUVToViewSpace(uv - float2(p.x, 0), ScreenSpaceToViewSpaceDepth(GetDepth(uv - float2(p.x, 0))), input.pScale);
        float3 p_r = DepthBufferUVToViewSpace(uv + float2(p.x, 0), ScreenSpaceToViewSpaceDepth(GetDepth(uv + float2(p.x, 0))), input.pScale);
        float3 p_t = DepthBufferUVToViewSpace(uv - float2(0, p.y), ScreenSpaceToViewSpaceDepth(GetDepth(uv - float2(0, p.y))), input.pScale);
        float3 p_b = DepthBufferUVToViewSpace(uv + float2(0, p.y), ScreenSpaceToViewSpaceDepth(GetDepth(uv + float2(0, p.y))), input.pScale);

        float4 edges = CalculateEdges(p_c.z, p_l.z, p_r.z, p_t.z, p_b.z);
        float3 normal = CalculateNormal(edges, p_c, p_l, p_r, p_t, p_b);
    
        outNormal = float4(normal * 0.5 + 0.5, 1.0);
    }

    float PS_GenerateSSAO(VS_OUTPUT input) : SV_Target
    {
        float2 uv = input.uv;
        float4 vpos = input.vpos;
        
        if (any(uv > RenderScale)) return 1.0;
        float2 scaled_uv = uv / RenderScale;

        float pixZ = tex2D(sDEPTH, scaled_uv).r;
        if (pixZ >= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.999) return 1.0;

        float2 p = BUFFER_RCP;
        float pixLZ = GetLod(sDEPTH, scaled_uv - float2(p.x, 0)).r;
        float pixRZ = GetLod(sDEPTH, scaled_uv + float2(p.x, 0)).r;
        float pixTZ = GetLod(sDEPTH, scaled_uv - float2(0, p.y)).r;
        float pixBZ = GetLod(sDEPTH, scaled_uv + float2(0, p.y)).r;

        float3 pixCenterPos = DepthBufferUVToViewSpace(scaled_uv, pixZ, input.pScale);
        float3 pixelNormal = normalize(tex2D(sNORMALS, scaled_uv).rgb * 2.0 - 1.0);

        float activeRadiusDistanceScale = EnableDistantRadius ? 1.0 : 0.0;
        float effectViewspaceRadius = Radius + (pixZ * activeRadiusDistanceScale);
        float pixelRadius = effectViewspaceRadius / pixZ;
    
        uint2 random_coord = uint2(vpos.xy);
        uint pseudoRandomIndex = (random_coord.y * 2 + random_coord.x) % 5;
        float angle = (pseudoRandomIndex / 5.0) * 2.0 * PI;
        float2 noise = SpatioTemporalNoise(uint2(vpos.xy), FRAME_COUNT);
        angle += noise.x * 2.0 * PI;

        float s, c;
        sincos(angle, s, c);
        float2x2 rotScale = float2x2(c, s, -s, c) * pixelRadius * 100.0;

        float obscuranceSum = 0.0;
        float weightSum = 0.0001;
        float4 edgesLRTB = CalculateEdges(pixZ, pixLZ, pixRZ, pixTZ, pixBZ);
    
        float falloffCalcMulSq = -1.0 / (effectViewspaceRadius * effectViewspaceRadius);
        const int numberOfTaps = g_TapCounts[clamp(QualityLevel, 0, 4)];
        
        for (int i = 0; i < numberOfTaps; i++)
        {
            SSAOTap(obscuranceSum, weightSum, i, rotScale, pixCenterPos, pixelNormal, scaled_uv, falloffCalcMulSq, noise, input.pScale);
        }

        float obscurance = obscuranceSum / weightSum;
        float fadeOut = 1.0;
        
        if (FadeOutTo > FadeOutFrom)
        {
            fadeOut = saturate((FadeOutTo - pixZ) / (FadeOutTo - FadeOutFrom));
        }
    
        float edgeFadeoutFactor = saturate((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35) + saturate((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35);
        fadeOut *= saturate(1.0 - edgeFadeoutFactor);
        obscurance = min(obscurance, EffectShadowClamp);
        obscurance *= fadeOut;
        float occlusion = 1.0 - obscurance;
        occlusion = pow(saturate(occlusion), ShadowPow);

        return occlusion;
    }
    
    float ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        if (any(input.uv > RenderScale)) discard;
        float2 viewUV = input.uv / RenderScale;
        float rawDepth = GetDepth(viewUV);
        
        if (rawDepth >= 0.999) return 1.0;

        float current_ao = GetLod(sAO, input.uv).r;
        float prevRenderScale = tex2Dlod(sRS_Prev, float4(0, 0, 0, 0)).x;
        
        if (abs(RenderScale - prevRenderScale) > 0.001 || !EnableTemporal || FRAME_COUNT < 2)
            return current_ao;

        float2 velocity = MV_GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        float2 reprojected_buffer_uv = reprojected_view_uv * RenderScale;

        if (any(reprojected_view_uv < 0.0) || any(reprojected_view_uv > 1.0))
            return current_ao;

        float history_ao = SampleHistoryCatmullRom_Scalar(sHistoryParams, reprojected_buffer_uv, float2(BUFFER_WIDTH, BUFFER_HEIGHT));

        // Variance Clipping
        float m1 = 0.0;
        float m2 = 0.0;
        float2 pSize = ReShade::PixelSize;

        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                float val = GetLod(sAO, input.uv + float2(x, y) * pSize).r;
                m1 += val;
                m2 += val * val;
            }
        }

        m1 /= 9.0;
        m2 /= 9.0;

        float sigma = sqrt(abs(m2 - m1 * m1));
        float gamma = 1.25;
        float val_min = m1 - gamma * sigma;
        float val_max = m1 + gamma * sigma;

        float flow_magnitude = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float curr_luma = GetLuminance(Input2Linear(GetColor(viewUV).rgb));
        float raw_confidence = saturate(MV_GetConfidenceAO(viewUV, velocity, flow_magnitude, curr_luma, sPrevLuma));

        float relax_amount = 0.15 * raw_confidence;
        val_min -= relax_amount;
        val_max += relax_amount;

        float clipped_history = clamp(history_ao, val_min, val_max);
        float clamp_distance = abs(clipped_history - history_ao);
        float blend_adapt = saturate(1.0 - clamp_distance * 2.0);

        float max_feedback = 0.98;
        float min_feedback = 0.85;
        float final_feedback = lerp(min_feedback, max_feedback, raw_confidence) * lerp(0.8, 1.0, blend_adapt);

        return lerp(current_ao, clipped_history, final_feedback);
    }
    
    void PS_SpatioTemporal0(VS_OUTPUT input, out float outHistory : SV_Target)
    {
        if ((FRAME_COUNT % 2) != 0) discard;
        outHistory = ComputeTAA(input, sHistory1);
    }

    void PS_SpatioTemporal1(VS_OUTPUT input, out float outHistory : SV_Target)
    {
        if ((FRAME_COUNT % 2) == 0) discard;
        outHistory = ComputeTAA(input, sHistory0);
    }

    float JointBilateralUpsample(float2 uv, float highDepth)
    {
        float2 lowResUV = uv * RenderScale;
        float3 highNormal = GetLod(sNORMALS, uv).rgb * 2.0 - 1.0;

        float sumAO = 0.0;
        float sumWeight = 0.0;

        float2 texelSize = ReShade::PixelSize;
        float2 baseUV = (floor(lowResUV / texelSize) + 0.5) * texelSize;

        float depth_weight_factor = 1.0 / (0.1 * highDepth + 1e-6);

        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                float2 sampleUV = baseUV + float2(x, y) * texelSize;
                float sampleAO;

                if ((FRAME_COUNT % 2) == 0)
                    sampleAO = GetLod(sHistory0, sampleUV).r;
                else
                    sampleAO = GetLod(sHistory1, sampleUV).r;

                float2 fullResSampleUV = sampleUV / RenderScale;
                float lowDepth = GetLod(sDEPTH, fullResSampleUV).r;
                float3 lowNormal = GetLod(sNORMALS, fullResSampleUV).rgb * 2.0 - 1.0;

                float wDepth = exp(-abs(highDepth - lowDepth) * depth_weight_factor);
                float dotN = max(0.0, dot(normalize(highNormal), normalize(lowNormal)));
                float wNormal = pow(dotN, 16.0);
                float wSpatial = exp(-0.5 * float(x * x + y * y));

                float weight = wDepth * wNormal * wSpatial;
                sumAO += sampleAO * weight;
                sumWeight += weight;
            }
        }

        if (sumWeight < 1e-6)
        {
            if ((FRAME_COUNT % 2) == 0) return GetLod(sHistory0, lowResUV).r;
            else return GetLod(sHistory1, lowResUV).r;
        }

        return sumAO / sumWeight;
    }

    void PS_StorePrevLuma(VS_OUTPUT input, out float outLuma : SV_Target)
    {
        outLuma = GetLuminance(Input2Linear(GetColor(input.uv).rgb));
    }

    float4 PS_Apply(VS_OUTPUT input) : SV_Target
    {
        float2 uv = input.uv;
        float4 color = GetColor(uv);
        float linear_depth = GetDepth(uv);

        if (linear_depth >= 0.999) return color;

        float view_depth = ScreenSpaceToViewSpaceDepth(linear_depth);
        float ao = 1.0;

        if (RenderScale >= 0.99)
        {
            if ((FRAME_COUNT % 2) == 0)
                ao = GetLod(sHistory0, uv).r;
            else
                ao = GetLod(sHistory1, uv).r;
        }
        else
        {
            ao = JointBilateralUpsample(uv, view_depth);
        }

        if (DebugView == 1) // Raw SSAO
        {
            return float4(ao, ao, ao, 1.0);
        }
        else if (DebugView == 2) // Normals
        {
            float3 debugNormals = GetLod(sNORMALS, uv).rgb;
            debugNormals = debugNormals * 2.0 - 1.0;
            debugNormals.x = -debugNormals.x;
            debugNormals.z = -debugNormals.z;
            
            return float4(debugNormals * 0.5 + 0.5, 1.0);
        }
    
        float3 linearColor = Input2Linear(color.rgb);
        linearColor *= ao;
        
        return float4(Linear2Output(linearColor), color.a);
    }
    
    void PS_SaveScale(VS_OUTPUT input, out float outScale : SV_Target)
    {
        outScale = RenderScale;
    }
    
    technique MiAO
    {
        pass Prepare
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_Prepare;
            RenderTarget0 = DEPTH;
            RenderTarget1 = NORMALS;
        }
        pass GenerateAO
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_GenerateSSAO;
            RenderTarget = AO;
            GenerateMipMaps = true;
        }
        pass SpatioTemporal0
        {
            VertexShader = VS_MiAO_Even;
            PixelShader = PS_SpatioTemporal0;
            RenderTarget = History0;
        }
        pass SpatioTemporal1
        {
            VertexShader = VS_MiAO_Odd;
            PixelShader = PS_SpatioTemporal1;
            RenderTarget = History1;
        }
        pass UpdateLuma
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_StorePrevLuma;
            RenderTarget = texPrevLuma;
        }
        pass Apply
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_Apply;
        }
        pass SaveScale
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_SaveScale;
            RenderTarget = RS_Prev;
        }
    }
}