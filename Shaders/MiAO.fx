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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
/*-------------------------------------------------|
| ::                   MiAO                     :: |
'--------------------------------------------------|
| Version: 1.2                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: Simple ambient occlusion with repur-|
| posed content from FidelityFX CACAO              |
'---------------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//--------------------|
// :: Preprocessor :: |
//--------------------|

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
static const float2 BUFFER_DIM = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
static const float2 BUFFER_RCP = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
static const float2 R2 = float2(0.75487766624669276005, 0.5698402909980532659114);
#define PI 3.1415927
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

//----------|
// :: UI :: |
//----------|

// -- Basic Settings --
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

// -- Performance & Quality --
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

uniform float TemporalBlend <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 0.0; ui_max = 0.99;ui_step = 0.01;
    ui_category = "Performance & Quality";
    ui_label = "Temporal Feedback";
    ui_tooltip = "Controls how much of the previous frame is blended in. Higher values are smoother but can cause more ghosting.";
> = 0.8;

// -- Advanced Options --
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
> = 300.0;

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
 static const float OcclusionSensitivity = 4.0;
//----------------|
// :: Textures :: |
//----------------|

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT; MinFilter=POINT; MipFilter=POINT; AddressU=Clamp; AddressV=Clamp; };
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
#endif

texture tMotionConfidence
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler sMotionConfidence
{
    Texture = tMotionConfidence;
    MagFilter = POINT;
    MinFilter = POINT;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
};

namespace MiAO
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
    };
    sampler sAO
    {
        Texture = AO;
    };

    texture FINAL_AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler sFINAL_AO
    {
        Texture = FINAL_AO;
    };

    texture HISTORY_AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16;
    };
    sampler sHISTORY_AO
    {
        Texture = HISTORY_AO;
    };

//----------------|
// :: Functions ::|
//----------------|

    float2 GetMotion(float2 texcoord)
    {
#if USE_MARTY_LAUNCHPAD_MOTION
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
#elif USE_VORT_MOTION
        return GetLod(sMotVectTexVort, texcoord).rg;
#else
        return GetLod(sTexMotionVectorsSampler, texcoord).rg;
#endif
    }

    float GetConfidence(float2 texcoord)
    {
        return GetLod(sMotionConfidence, texcoord).r;
    }

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
        return hilbert(float2(x % 64, y % 64), 6);
    }

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        uint index = HilbertIndex(pixCoord.x, pixCoord.y);
        index += 288 * (temporalIndex % 64);
        return frac(0.5 + index * R2);
    }

    float2 CameraTanHalfFOV()
    {
        float fov_rad = FOV * (PI / 180.0);
        float tanHalfFOV = tan(fov_rad * 0.5);
        return float2(tanHalfFOV, tanHalfFOV);
    }
    
    float2 NDCToViewMul()
    {
        float2 tanHalfFOV = CameraTanHalfFOV();
        return float2(tanHalfFOV.x * ReShade::AspectRatio, tanHalfFOV.y);
    }

    float ScreenSpaceToViewSpaceDepth(float screenDepth)
    {
        return screenDepth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }

    float3 DepthBufferUVToViewSpace(float2 pos, float viewspaceDepth)
    {
        float3 ret;
        float2 ndc = pos * 2.0 - 1.0;
        ndc.y = -ndc.y;
        ret.xy = (NDCToViewMul() * ndc) * viewspaceDepth;
        ret.z = viewspaceDepth;
        return ret;
    }

    float4 CalculateEdges(const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ)
    {
        float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
        float4 edgesLRTBSlopeAdjusted = edgesLRTB + edgesLRTB.yxwz;
        edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
        return saturate((1.3 - edgesLRTB / (centerZ * 0.040)));
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

    float4 GetSamplePattern(int index, float2 noise)
    {
        float2 offset = frac(R2 * float(index) + noise);
        float radius = sqrt(offset.x);
        float angle = offset.y * 2.0 * PI;
    
        float2 samplePos;
        sincos(angle, samplePos.y, samplePos.x);
        samplePos *= radius;
        float weight = lerp(1.5, 0.6, saturate(float(index) / 32.0));
        float mipLevel;
        if (index < 3)
            mipLevel = -2.0 + float(index) * 0.5;
        else if (index < 12)
            mipLevel = -1.0 + float(index - 3) * 0.1;
        else
            mipLevel = -0.5 + float(index - 12) * 0.05;
    
        return float4(samplePos, weight, mipLevel);
    }

    int GetNumTaps(int qualityLevel)
    {
        switch (qualityLevel)
        {
            case 0:
                return 3;
            case 1:
                return 5;
            case 2:
                return 12;
            case 3:
                return 20;
            case 4:
                return 31;
        }
        return 12;
    }

    float CalculatePixelObscurance(float3 pixelNormal, float3 hitDelta, float falloffCalcMulSq)
    {
        float lengthSq = dot(hitDelta, hitDelta);
        float NdotD = dot(pixelNormal, hitDelta) / sqrt(lengthSq);
        float falloffMult = max(0.0, lengthSq * falloffCalcMulSq + 1.0);
        return max(0, NdotD - EffectHorizonAngleThreshold) * falloffMult;
    }

    void SSAOTap(
    inout float obscuranceSum, inout float weightSum,
    const int tapIndex, const float2x2 rotScale, const float3 pixCenterPos,
    const float3 pixelNormal, const float2 depthBufferUV, const float falloffCalcMulSq,
    const float2 noise)
    {
        float4 newSample = GetSamplePattern(tapIndex, noise);
        float2 sampleOffset = mul(rotScale, newSample.xy);
        float weightMod = newSample.z;

        float2 samplingUV = sampleOffset * BUFFER_RCP + depthBufferUV;
        samplingUV = saturate(samplingUV);

        float viewspaceSampleZ1 = GetLod(sDEPTH, float4(samplingUV, 0, 0)).r;
        float3 hitPos1 = DepthBufferUVToViewSpace(samplingUV, viewspaceSampleZ1);
        float3 hitDelta1 = hitPos1 - pixCenterPos;
        float obscurance1 = CalculatePixelObscurance(pixelNormal, hitDelta1, falloffCalcMulSq);
        float weight1 = 1.0;
    
        obscuranceSum += obscurance1 * weight1 * weightMod;
        weightSum += weight1 * weightMod;

        float2 samplingUV2 = -sampleOffset * BUFFER_RCP + depthBufferUV;
        samplingUV2 = saturate(samplingUV2);
    
        float viewspaceSampleZ2 = GetLod(sDEPTH, float4(samplingUV2, 0, 0)).r;
        float3 hitPos2 = DepthBufferUVToViewSpace(samplingUV2, viewspaceSampleZ2);
        float3 hitDelta2 = hitPos2 - pixCenterPos;
        float obscurance2 = CalculatePixelObscurance(pixelNormal, hitDelta2, falloffCalcMulSq);
        float weight2 = 1.0;

        obscuranceSum += obscurance2 * weight2 * weightMod;
        weightSum += weight2 * weightMod;
    }

/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/

    void PS_Prepare(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outDepth : SV_Target0, out float4 outNormal : SV_Target1)
    {
        float linear_depth = GetDepth(uv);
        outDepth = ScreenSpaceToViewSpaceDepth(linear_depth);

        float2 p = BUFFER_RCP;
        float3 p_c = DepthBufferUVToViewSpace(uv, outDepth.r);
        float3 p_l = DepthBufferUVToViewSpace(uv - float2(p.x, 0), ScreenSpaceToViewSpaceDepth(GetDepth(uv - float2(p.x, 0))));
        float3 p_r = DepthBufferUVToViewSpace(uv + float2(p.x, 0), ScreenSpaceToViewSpaceDepth(GetDepth(uv + float2(p.x, 0))));
        float3 p_t = DepthBufferUVToViewSpace(uv - float2(0, p.y), ScreenSpaceToViewSpaceDepth(GetDepth(uv - float2(0, p.y))));
        float3 p_b = DepthBufferUVToViewSpace(uv + float2(0, p.y), ScreenSpaceToViewSpaceDepth(GetDepth(uv + float2(0, p.y))));

        float4 edges = CalculateEdges(p_c.z, p_l.z, p_r.z, p_t.z, p_b.z);
        float3 normal = CalculateNormal(edges, p_c, p_l, p_r, p_t, p_b);
    
        outNormal = float4(normal * 0.5 + 0.5, 1.0);
    }

    float PS_GenerateSSAO(float4 vpos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 1.0;
        }
        float2 scaled_uv = uv / RenderScale;

        float pixZ = tex2D(sDEPTH, scaled_uv).r;

        if (pixZ >= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.999)
        {
            return 1.0;
        }

        float2 p = BUFFER_RCP;
        float pixLZ = GetLod(sDEPTH, scaled_uv - float2(p.x, 0)).r;
        float pixRZ = GetLod(sDEPTH, scaled_uv + float2(p.x, 0)).r;
        float pixTZ = GetLod(sDEPTH, scaled_uv - float2(0, p.y)).r;
        float pixBZ = GetLod(sDEPTH, scaled_uv + float2(0, p.y)).r;

        float3 pixCenterPos = DepthBufferUVToViewSpace(scaled_uv, pixZ);
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

        const int numberOfTaps = GetNumTaps(QualityLevel);
        for (int i = 0; i < numberOfTaps; i++)
        {
            SSAOTap(obscuranceSum, weightSum, i, rotScale, pixCenterPos, pixelNormal, scaled_uv, falloffCalcMulSq, noise);
        }

        float obscurance = obscuranceSum / weightSum;
    
        float fadeOut = 1.0;
        if (FadeOutTo > FadeOutFrom)
        {
            fadeOut = saturate((FadeOutTo - pixZ) / (FadeOutTo - FadeOutFrom));
        }
    
        float edgeFadeoutFactor = saturate((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35) +
                                  saturate((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35);
        fadeOut *= saturate(1.0 - edgeFadeoutFactor);
        obscurance = min(obscurance, EffectShadowClamp);
        obscurance *= fadeOut;
        float occlusion = 1.0 - obscurance;
        occlusion = pow(saturate(occlusion), ShadowPow);

        return occlusion;
    }
    
    void PS_Upscale(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float2 outUpscaled : SV_Target)
    {
        float current_ao;
        if (RenderScale >= 1.0)
        {
            current_ao = GetLod(sAO, uv).r;
        }
        else
        {
            current_ao = GetLod(sAO, uv * RenderScale).r;
        }

        if (FRAME_COUNT < 2)
        {
            outUpscaled = float2(current_ao, current_ao * current_ao);
            return;
        }

        float2 motion = GetMotion(uv);
        float2 reprojected_uv = uv + motion;

        bool reproj_out_of_bounds = any(reprojected_uv < 0.0) || any(reprojected_uv > 1.0);
        bool large_motion = dot(motion, motion) > (1 * 1);

        float2 history_moments = GetLod(sHISTORY_AO, saturate(reprojected_uv)).rg;
        float hist_m1 = history_moments.r;
        float hist_m2 = history_moments.g;

        float depth_cur = GetDepth(uv);
        float depth_prev = GetLod(sHISTORY_AO, saturate(reprojected_uv)).r;
        bool depth_invalid = (depth_cur <= 0.0) || (depth_prev <= 0.0);

        float depth_diff = abs(depth_cur - depth_prev);
        bool depth_close = depth_diff <= 1;
        bool valid_history = !reproj_out_of_bounds && !large_motion && !depth_invalid && depth_close;
    
        float blend_factor = valid_history ? TemporalBlend: 0.0; //Temporal Blend

        if (blend_factor > 0.0)
        {
            float confidence = GetConfidence(uv) * 1.0;
            if (confidence <= 0.001)
            {
                float aoDiff = abs(current_ao - hist_m1);
                confidence = 1.0 - saturate(aoDiff * OcclusionSensitivity);
            }
            blend_factor *= saturate(confidence);
        }

        if (blend_factor <= 0.0)
        {
            outUpscaled = float2(current_ao, current_ao * current_ao);
            return;
        }
    
        float hist_m2_safe = (hist_m2 > 0.0) ? hist_m2 : (hist_m1 * hist_m1);

        float new_moment1 = lerp(current_ao, hist_m1, blend_factor);
        float new_moment2 = lerp(current_ao * current_ao, hist_m2_safe, blend_factor);
        new_moment2 = max(new_moment2, 0.0);
        outUpscaled = float2(new_moment1, new_moment2);
    }

    void PS_UpdateHistory(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float2 outHistory : SV_Target)
    {
        outHistory = GetLod(sFINAL_AO, uv).rg;
    }

    float4 PS_Apply(float4 vpos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 color = GetColor(uv);
        float linear_depth = GetDepth(uv);

        if (linear_depth >= 0.999)
            return color;

        if (DebugView == 1) // Raw SSAO
        {
            return GetLod(sFINAL_AO, uv).rrrr;
        }
        else if (DebugView == 2) // Normals
        {
            return GetLod(sNORMALS, uv);
        }
    
        float ao = GetLod(sFINAL_AO, uv).r;
        color.rgb *= ao;
    
        return color;
    }
    
    technique MiAO
    {
        pass Prepare
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Prepare;
            RenderTarget0 = DEPTH;
            RenderTarget1 = NORMALS;
        }
        pass GenerateAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GenerateSSAO;
            RenderTarget = AO;
        }
        pass UpdateHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HISTORY_AO;
        }
        pass Upscale
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale;
            RenderTarget = FINAL_AO;
        }
        pass Apply
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Apply;
        }
    }
}
