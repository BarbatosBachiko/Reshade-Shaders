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
| Version: 1.0                                     |
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

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
static const float2 BUFFER_DIM = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
static const float2 BUFFER_RCP = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
#define PI 3.1415927

//----------|
// :: UI :: |
//----------|

#if UI_DIFFICULTY == 0 // Simple Mode
#define EffectShadowClamp 0.98
#define EffectHorizonAngleThreshold 0.04
#define DetailAOStrength 0.5
#define FadeOutFrom 50.0
#define FadeOutTo 300.0
#define FOV 75.0
#define RadiusDistanceScale 0.25
#define ShadowPow 1.5

uniform float Intensity <
    ui_type = "drag";
    ui_label = "Intensity";
    ui_category = "AO Settings";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
    ui_tooltip = "Global strength of the ambient occlusion effect.";
> = 1.0;

uniform float Radius <
    ui_type = "drag";
    ui_label = "Radius";
    ui_category = "AO Settings";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.01;
    ui_tooltip = "World-space radius of the ambient occlusion effect.";
> = 3.0;

uniform int QualityLevel <
    ui_type = "combo";
    ui_label = "Quality";
    ui_category = "Performance & Quality";
    ui_items = "Lowest (3 taps)\0Low (5 taps)\0Medium (12 taps)\0High (20 taps)\0Very high (32 taps)\0";
    ui_tooltip = "Controls the number of samples used for AO calculation.";
> = 2;

uniform float RenderScale <
    ui_type = "drag";
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Renders AO at a lower resolution for better performance, then upscales it.";
> = 0.8;

uniform int DebugView <
    ui_type = "combo";
    ui_label = "Debug View";
    ui_category = "Debug";
    ui_items = "None\0Raw SSAO\0View-space Normals\0";
> = 0;

#elif UI_DIFFICULTY == 1 // Advanced Mode

// -- Main Settings --
uniform float Intensity <
    ui_type = "drag";
    ui_category = "Main Settings";
    ui_label = "Intensity";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
    ui_tooltip = "Global strength of the ambient occlusion effect.";
> = 1.0;

uniform float ShadowPow <
    ui_type = "drag";
    ui_category = "Main Settings";
    ui_label = "Contrast";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.01;
    ui_tooltip = "Controls the contrast of the occlusion. Higher values create a darker, more defined effect.";
> = 1.5;

uniform float Radius <
    ui_type = "drag";
    ui_category = "Main Settings";
    ui_label = "Radius";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.01;
    ui_tooltip = "World-space radius of the ambient occlusion effect.";
> = 3.0;

uniform float EffectShadowClamp <
    ui_type = "drag";
    ui_category = "Main Settings";
    ui_label = "Clamp";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Limits the maximum amount of occlusion to prevent excessive darkening.";
> = 0.98;

// -- Performance & Quality --
uniform int QualityLevel <
    ui_type = "combo";
    ui_label = "Quality";
    ui_category = "Performance & Quality";
    ui_items = "Lowest (3 taps)\0Low (5 taps)\0Medium (12 taps)\0High (20 taps)\0Very high (20 taps)\0";
    ui_tooltip = "Controls the number of samples used for AO calculation.";
> = 2;

uniform float RenderScale <
    ui_type = "drag";
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Renders AO at a lower resolution for better performance, then upscales it.";
> = 0.8;

uniform float DetailAOStrength <
    ui_type = "drag";
    ui_category = "Performance & Quality";
    ui_label = "Detail AO Strength";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.01;
    ui_tooltip = "Adds a high-definition, sharp AO effect based on immediate neighbors. Only active on Low quality and above.";
> = 0.5;

// -- Advanced Settings --
uniform float EffectHorizonAngleThreshold <
    ui_type = "drag";
    ui_category = "Advanced Settings";
    ui_label = "Horizon Angle Threshold";
    ui_min = 0.0; ui_max = 0.5; ui_step = 0.001;
    ui_tooltip = "Limits errors on slopes and caused by insufficient geometry tessellation.";
> = 0.04;

uniform float RadiusDistanceScale <
    ui_type = "drag";
    ui_category = "Advanced Settings";
    ui_label = "Distant Radius Scale";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_tooltip = "Increases the AO radius for distant objects.";
> = 0.25;

uniform float FadeOutFrom <
    ui_type = "drag";
    ui_category = "Advanced Settings";
    ui_label = "Fade Out Start";
    ui_min = 1.0; ui_max = 500.0; ui_step = 1.0;
    ui_tooltip = "The distance at which the AO effect begins to fade out.";
> = 50.0;

uniform float FadeOutTo <
    ui_type = "drag";
    ui_category = "Advanced Settings";
    ui_label = "Fade Out End";
    ui_min = 1.0; ui_max = 500.0; ui_step = 1.0;
    ui_tooltip = "The distance at which the AO effect has completely faded out.";
> = 300.0;

uniform float FOV <
    ui_type = "slider";
    ui_category = "Advanced Settings";
    ui_label = "Vertical FOV";
    ui_min = 30.0; ui_max = 120.0;
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 75.0;

// -- Debug --
uniform int DebugView <
    ui_type = "combo";
    ui_category = "Debug";
    ui_label = "Debug View";
    ui_items = "None\0Raw SSAO\0View-space Normals\0";
> = 0;

#endif

static const bool EnableTemporalFilter = true;
static const float TemporalFeedback = 0.9;
static const float VarianceClippingStrength = 0.4;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|
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
    Format = RG16F;
};
sampler sHISTORY_AO
{
    Texture = HISTORY_AO;
};

//----------------|
// :: Functions ::|
//----------------|

//https://www.shadertoy.com/view/3tB3z3
uint part1by1(uint x)
{
    x = (x & 0x0000ffffu);
    x = ((x ^ (x << 8u)) & 0x00ff00ffu);
    x = ((x ^ (x << 4u)) & 0x0f0f0f0fu);
    x = ((x ^ (x << 2u)) & 0x33333333u);
    x = ((x ^ (x << 1u)) & 0x55555555u);
    return x;
}
    
uint compact1by1(uint x)
{
    x = (x & 0x55555555u);
    x = ((x ^ (x >> 1u)) & 0x33333333u);
    x = ((x ^ (x >> 2u)) & 0x0f0f0f0fu);
    x = ((x ^ (x >> 4u)) & 0x00ff00ffu);
    x = ((x ^ (x >> 8u)) & 0x0000ffffu);
    return x;
}
    
uint pack_morton2x16(uint2 v)
{
    return part1by1(v.x) | (part1by1(v.y) << 1);
}

uint2 unpack_morton2x16(uint p)
{
    return uint2(compact1by1(p), compact1by1(p >> 1));
}

uint inverse_gray32(uint n)
{
    n = n ^ (n >> 1);
    n = n ^ (n >> 2);
    n = n ^ (n >> 4);
    n = n ^ (n >> 8);
    n = n ^ (n >> 16);
    return n;
}

// https://www.shadertoy.com/view/llGcDm
int hilbert(int2 p, int level)
{
    int d = 0;
    for (int k = 0; k < level; k++)
    {
        int n = level - k - 1;
        int2 r = (p >> n) & 1;
        d += ((3 * r.x) ^ r.y) << (2 * n);
        if (r.y == 0)
        {
            if (r.x == 1)
            {
                p = (1 << n) - 1 - p;
            }
            p = p.yx;
        }
    }
    return d;
}

// https://www.shadertoy.com/view/llGcDm
int2 ihilbert(int i, int level)
{
    int2 p = int2(0, 0);
    for (int k = 0; k < level; k++)
    {
        int2 r = int2(i >> 1, i ^ (i >> 1)) & 1;
        if (r.y == 0)
        {
            if (r.x == 1)
            {
                p = (1 << k) - 1 - p;
            }
            p = p.yx;
        }
        p += r << k;
        i >>= 2;
    }
    return p;
}

// knuth's multiplicative hash function (fixed point R1)
uint kmhf(uint x)
{
    return 0x80000000u + 2654435789u * x;
}

uint kmhf_inv(uint x)
{
    return (x - 0x80000000u) * 827988741u;
}

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R1 quasirandom sequence for it
uint hilbert_r1_blue_noise(uint2 p)
{
#if 1
    uint x = uint(hilbert(int2(p), 17)) % (1u << 17u);
#else
    //p = p ^ (p >> 1);
    uint x = pack_morton2x16( p ) % (1u << 17u);    
    //x = x ^ (x >> 1);
    x = inverse_gray32(x);
#endif
#if 0
    // based on http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const float phi = 2.0/(sqrt(5.0)+1.0);
	return frac(0.5+phi*float(x));
#else
    x = kmhf(x);
    return x;
#endif
}

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R1 quasirandom sequence for it
float hilbert_r1_blue_noisef(uint2 p)
{
    uint x = hilbert_r1_blue_noise(p);
#if 0
    return float(x >> 24) / 256.0;
#else
    return float(x) / 4294967296.0;
#endif
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

float4 GetSamplePattern(int index)
{
    switch (index)
    {
        case 0:
            return float4(0.78488064, 0.56661671, 1.500000, -0.126083);
        case 1:
            return float4(0.26022232, -0.29575172, 1.500000, -1.064030);
        case 2:
            return float4(0.10459357, 0.08372527, 1.110000, -2.730563);
        case 3:
            return float4(-0.68286800, 0.04963045, 1.090000, -0.498827);
        case 4:
            return float4(-0.13570161, -0.64190155, 1.250000, -0.532765);
        case 5:
            return float4(-0.26193795, -0.08205118, 0.670000, -1.783245);
        case 6:
            return float4(-0.61177456, 0.66664219, 0.710000, -0.044234);
        case 7:
            return float4(0.43675563, 0.25119025, 0.610000, -1.167283);
        case 8:
            return float4(0.07884444, 0.86618668, 0.640000, -0.459002);
        case 9:
            return float4(-0.12790935, -0.29869005, 0.600000, -1.729424);
        case 10:
            return float4(-0.04031125, 0.02413622, 0.600000, -4.792042);
        case 11:
            return float4(0.16201244, -0.52851415, 0.790000, -1.067055);
        case 12:
            return float4(-0.70991218, 0.47301072, 0.640000, -0.335236);
        case 13:
            return float4(0.03277707, -0.22349690, 0.600000, -1.982384);
        case 14:
            return float4(0.68921727, 0.36800742, 0.630000, -0.266718);
        case 15:
            return float4(0.29251814, 0.37775412, 0.610000, -1.422520);
        case 16:
            return float4(-0.12224089, 0.96582592, 0.600000, -0.426142);
        case 17:
            return float4(0.11071457, -0.16131058, 0.600000, -2.165947);
        case 18:
            return float4(0.46562141, -0.59747696, 0.600000, -0.189760);
        case 19:
            return float4(-0.51548797, 0.11804193, 0.600000, -1.246800);
        case 20:
            return float4(0.89141309, -0.42090443, 0.600000, 0.028192);
        case 21:
            return float4(-0.32402530, -0.01591529, 0.600000, -1.543018);
        case 22:
            return float4(0.60771245, 0.41635221, 0.600000, -0.605411);
        case 23:
            return float4(0.02379565, -0.08239821, 0.600000, -3.809046);
        case 24:
            return float4(0.48951152, -0.23657045, 0.600000, -1.189011);
        case 25:
            return float4(-0.17611565, -0.81696892, 0.600000, -0.513724);
        case 26:
            return float4(-0.33930185, -0.20732205, 0.600000, -1.698047);
        case 27:
            return float4(-0.91974425, 0.05403209, 0.600000, 0.062246);
        case 28:
            return float4(-0.15064627, -0.14949332, 0.600000, -1.896062);
        case 29:
            return float4(0.53180975, -0.35210401, 0.600000, -0.758838);
        case 30:
            return float4(0.41487166, 0.81442589, 0.600000, -0.505648);
        case 31:
            return float4(-0.24106961, -0.32721516, 0.600000, -1.665244);
    }
    return float4(0, 0, 0, 0);
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
    const float3 pixelNormal, const float2 depthBufferUV, const float falloffCalcMulSq)
{
    float4 newSample = GetSamplePattern(tapIndex);
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

float4 cubic(float v)
{
    float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
    float4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return float4(x, y, z, w) * (1.0 / 6.0);
}

float4 textureBicubic(sampler s, float2 texCoords, float2 lowResTexSize)
{
    float2 invTexSize = 1.0 / lowResTexSize;
    texCoords = texCoords * lowResTexSize - 0.5;
    float2 fxy = frac(texCoords);
    texCoords -= fxy;
    float4 xcubic = cubic(fxy.x);
    float4 ycubic = cubic(fxy.y);
    float4 c = float4(texCoords.x, texCoords.x, texCoords.y, texCoords.y) + float4(-0.5, 1.5, -0.5, 1.5);
    float4 ss = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    float4 offset = c + float4(xcubic.yw, ycubic.yw) / ss;
    offset *= float4(invTexSize.x, invTexSize.x, invTexSize.y, invTexSize.y);

    float4 sample0 = GetLod(s, offset.xz * RenderScale);
    float4 sample1 = GetLod(s, offset.yz * RenderScale);
    float4 sample2 = GetLod(s, offset.xw * RenderScale);
    float4 sample3 = GetLod(s, offset.yw * RenderScale);

    float sx = ss.x / (ss.x + ss.y);
    float sy = ss.z / (ss.z + ss.w);
    return lerp(
        lerp(sample3, sample2, sx),
        lerp(sample1, sample0, sx),
        sy
    );
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

    float effectViewspaceRadius = Radius + (pixZ * RadiusDistanceScale);
    float pixelRadius = effectViewspaceRadius / pixZ;
    
    uint2 random_coord = uint2(vpos.xy);
    uint pseudoRandomIndex = (random_coord.y * 2 + random_coord.x) % 5;
    float angle = (pseudoRandomIndex / 5.0) * 2.0 * PI;

    angle += hilbert_r1_blue_noisef(uint2(vpos.xy) + (FRAME_COUNT % 256)) * 2.0 * PI;

    float s, c;
    sincos(angle, s, c);
    
    float2x2 rotScale = float2x2(c, s, -s, c) * pixelRadius * 50.0;

    float obscuranceSum = 0.0;
    float weightSum = 0.0001;

    float4 edgesLRTB = CalculateEdges(pixZ, pixLZ, pixRZ, pixTZ, pixBZ);
    
    float falloffCalcMulSq = -1.0 / (effectViewspaceRadius * effectViewspaceRadius);
    
    if (QualityLevel > 0 && DetailAOStrength > 0.0)
    {
        float3 pixLPos = DepthBufferUVToViewSpace(scaled_uv - float2(p.x, 0), pixLZ);
        float3 pixRPos = DepthBufferUVToViewSpace(scaled_uv + float2(p.x, 0), pixRZ);
        float3 pixTPos = DepthBufferUVToViewSpace(scaled_uv - float2(0, p.y), pixTZ);
        float3 pixBPos = DepthBufferUVToViewSpace(scaled_uv + float2(0, p.y), pixBZ);
        
        float3 pixLDelta = pixLPos - pixCenterPos;
        float3 pixRDelta = pixRPos - pixCenterPos;
        float3 pixTDelta = pixTPos - pixCenterPos;
        float3 pixBDelta = pixBPos - pixCenterPos;

        const float modifiedFalloffCalcMulSq = 4.0 * falloffCalcMulSq;
        float4 additionalObscurance;
        additionalObscurance.x = CalculatePixelObscurance(pixelNormal, pixLDelta, modifiedFalloffCalcMulSq);
        additionalObscurance.y = CalculatePixelObscurance(pixelNormal, pixRDelta, modifiedFalloffCalcMulSq);
        additionalObscurance.z = CalculatePixelObscurance(pixelNormal, pixTDelta, modifiedFalloffCalcMulSq);
        additionalObscurance.w = CalculatePixelObscurance(pixelNormal, pixBDelta, modifiedFalloffCalcMulSq);
        obscuranceSum += DetailAOStrength * dot(additionalObscurance, edgesLRTB);
    }

    const int numberOfTaps = GetNumTaps(QualityLevel);
    for (int i = 0; i < numberOfTaps; i++)
    {
        SSAOTap(obscuranceSum, weightSum, i, rotScale, pixCenterPos, pixelNormal, scaled_uv, falloffCalcMulSq);
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
    obscurance *= Intensity;
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
        float2 lowResSize = float2(BUFFER_WIDTH * RenderScale, BUFFER_HEIGHT * RenderScale);
        current_ao = textureBicubic(sAO, uv, lowResSize).r;
    }

    if (!EnableTemporalFilter || FRAME_COUNT < 2)
    {
        outUpscaled = float2(current_ao, current_ao * current_ao);
        return;
    }

    // Variance Clipping
    float2 low_res_uv = uv * RenderScale;
    float2 pixel_size_low_res = (BUFFER_RCP / RenderScale);

    float moment1 = 0.0; // E[x]
    float moment2 = 0.0; // E[x^2]

    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float2 offset_uv = low_res_uv + float2(x, y) * pixel_size_low_res;
            float neighbor_ao = GetLod(sAO, saturate(offset_uv)).r;
            moment1 += neighbor_ao;
            moment2 += neighbor_ao * neighbor_ao;
        }
    }

    moment1 /= 9.0;
    moment2 /= 9.0;

    float variance = max(0.0, moment2 - (moment1 * moment1));
    float std_dev = sqrt(variance);

    float2 range = moment1 + float2(-VarianceClippingStrength, VarianceClippingStrength) * std_dev;

    float2 history_moments = GetLod(sHISTORY_AO, uv).rg;
    float clamped_history_ao = clamp(history_moments.r, range.x, range.y);

    float new_moment1 = lerp(current_ao, clamped_history_ao, TemporalFeedback);
    float new_moment2 = lerp(current_ao * current_ao, clamped_history_ao * clamped_history_ao, TemporalFeedback);

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