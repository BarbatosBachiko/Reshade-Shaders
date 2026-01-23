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
| Version: 1.4                                     |
| Author: Barbatos                                 |
| License: MIT                                     |
| Description: Simple ambient occlusion with repur-|
| posed content from FidelityFX CACAO              |
'-------------------------------------------------*/

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

namespace MiAO24
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
        Format = RG16F;
    };
    sampler sHISTORY_AO
    {
        Texture = HISTORY_AO;
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

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }

    float2 GetVelocity(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = GetDepth(texcoord);
        float2 best_coord = texcoord;
        
        const int2 offsets[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };

        [unroll]
        for (int i = 0; i < 4; i++)
        {
            float2 s_coord = texcoord + offsets[i] * pixel_size;
            float s_depth = GetDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                best_coord = s_coord;
            }
        }
        return GetMotion(best_coord);
    }

    //Based on LumaFlow.fx from LumeniteFX CC-BY-NC-4.0
    float Confidence(float2 uv, float2 velocity)
    {
        float2 prev_uv = uv + velocity;
        if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
            return 0.0;

        float curr_luma = GetLuminance(GetColor(uv).rgb);
        float prev_luma = tex2D(sPrevLuma, prev_uv).r;
        float luma_error = abs(curr_luma - prev_luma);
        float flow_magnitude = length(velocity * BUFFER_DIM);
        
        if (flow_magnitude <= 1.0)
            return 1.0;

        float2 destination_velocity = GetMotion(prev_uv);
        float2 diff = velocity - destination_velocity;
        float error = length(diff);
        float normalized_error = error / length(velocity);

        float motion_penalty = flow_magnitude;
        float length_conf = rcp(motion_penalty * 0.002 + 1.0);
        float consistency_conf = rcp(normalized_error + 1.0);
        float photometric_conf = exp(-luma_error * 1.0);

        return (consistency_conf * length_conf * photometric_conf);
    }

    void ComputeNeighborhoodMinMax(sampler2D color_tex, float2 texcoord, float center_val, out float color_min, out float color_max)
    {
        float4 vals = tex2DgatherR(color_tex, texcoord);
        color_min = min(center_val, min(min(vals.x, vals.y), min(vals.z, vals.w)));
        color_max = max(center_val, max(max(vals.x, vals.y), max(vals.z, vals.w)));
    }

    float ComputeTrustFactor(float2 velocity_pixels, float low_threshold = 10.0, float high_threshold = 80.0)
    {
        float vel_mag = length(velocity_pixels);
        return saturate((high_threshold - vel_mag) / (high_threshold - low_threshold));
    }

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
        // Calculate index procedurally (Slower)
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

/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/

    void PS_Prepare(VS_OUTPUT input, out float4 outDepth : SV_Target0, out float4 outNormal : SV_Target1, out float2 outHistory : SV_Target2)
    {
        float2 uv = input.uv;
        float linear_depth = GetDepth(uv);
        outDepth = ScreenSpaceToViewSpaceDepth(linear_depth);
        outHistory = GetLod(sFINAL_AO, uv).rg;

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
    
        float edgeFadeoutFactor = saturate((1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35) +
                                  saturate((1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35);
        fadeOut *= saturate(1.0 - edgeFadeoutFactor);
        obscurance = min(obscurance, EffectShadowClamp);
        obscurance *= fadeOut;
        float occlusion = 1.0 - obscurance;
        occlusion = pow(saturate(occlusion), ShadowPow);

        return occlusion;
    }
    
    void PS_Upscale(VS_OUTPUT input, out float2 outUpscaled : SV_Target)
    {
        float2 uv = input.uv;
        float2 scaled_uv = (RenderScale >= 1.0) ? uv : uv * RenderScale;

        //Denoising
        float myDepth = GetDepth(uv);
        float depthGrad = length(float2(ddx(myDepth), ddy(myDepth)));
        float mip_level = saturate(1.0 - depthGrad * 500.0) * 2.5;
        float current_ao = tex2Dlod(sAO, float4(scaled_uv, 0, mip_level)).r;

        //Temporal Filter
        if (!EnableTemporal || FRAME_COUNT < 2)
        {
            outUpscaled = float2(current_ao, current_ao * current_ao);
            return;
        }

        float2 velocity = GetVelocity(uv);
        
        // Reprojection
        float2 reprojected_uv = uv + velocity;
        
        // Check bounds
        if (any(reprojected_uv < 0.0) || any(reprojected_uv > 1.0))
        {
            outUpscaled = float2(current_ao, current_ao * current_ao);
            return;
        }

        // Confidence & History
        float confidence = Confidence(uv, velocity);
        float2 history_signal = GetLod(sHISTORY_AO, reprojected_uv).rg;

        // Neighborhood Clamping 
        float sig_min, sig_max;
        ComputeNeighborhoodMinMax(sAO, scaled_uv, current_ao, sig_min, sig_max);

        float2 clipped_history = clamp(history_signal, sig_min, sig_max);
        
        // Blend
        float blend_factor = 0.9 * confidence;
        float2 temporal_signal = lerp(float2(current_ao, current_ao * current_ao), clipped_history, blend_factor);

        // Spatial Filter 
        float trust_factor = 1.0;
        if (confidence < 0.9)
        {
            trust_factor = ComputeTrustFactor(velocity * BUFFER_DIM);
        }

        if (trust_factor < 0.95)
        {
            float2 blurred_signal = float2(current_ao, current_ao * current_ao);
            float valid_samples = 1.0;
            const int blur_samples = 3;
            
            [unroll]
            for (int i = 1; i < blur_samples; i++)
            {
                float t = (float) i / (float) (blur_samples - 1);
                float2 blur_coord = uv - velocity * 0.5 * t;
                
                if (all(blur_coord >= 0.0) && all(blur_coord <= 1.0))
                {
                    float sampleAO = GetLod(sAO, (RenderScale >= 1.0 ? blur_coord : blur_coord * RenderScale)).r;
                    blurred_signal += float2(sampleAO, sampleAO * sampleAO);
                    valid_samples += 1.0;
                }
            }
            blurred_signal /= valid_samples;
            temporal_signal = lerp(blurred_signal, temporal_signal, trust_factor);
        }

        outUpscaled = temporal_signal;
    }

    void PS_StorePrevLuma(VS_OUTPUT input, out float outLuma : SV_Target)
    {
        outLuma = GetLuminance(GetColor(input.uv).rgb);
    }

    float4 PS_Apply(VS_OUTPUT input) : SV_Target
    {
        float2 uv = input.uv;
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
            VertexShader = VS_MiAO;
            PixelShader = PS_Prepare;
            RenderTarget0 = DEPTH;
            RenderTarget1 = NORMALS;
            RenderTarget2 = HISTORY_AO;
        }
        pass GenerateAO
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_GenerateSSAO;
            RenderTarget = AO;
            GenerateMipMaps = true;
        }
        pass Upscale
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_Upscale;
            RenderTarget = FINAL_AO;
        }
        pass Apply
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_Apply;
        }
        pass UpdateLuma
        {
            VertexShader = VS_MiAO;
            PixelShader = PS_StorePrevLuma;
            RenderTarget = texPrevLuma;
        }
    }
}
