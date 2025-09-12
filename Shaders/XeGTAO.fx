/*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion",
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
//
// Implementation: Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>               (\_/)
// Version:        (see XeGTAO.h)                                                                                  (='.'=)
// Details:        https://github.com/GameTechDev/XeGTAO                                                           (")_(")
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmoothNormals from https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/smoothNormals.fx
/*------------------.
| :: Description :: |
'-------------------/

    XeGTAO
    Version 1.5.2
    Author: Barbatos

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
*/

#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define half min16float
#define half2 min16float2
#define half3 min16float3
#define half4 min16float4
#define half3x3 float3x3

#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;
#define S_LC MagFilter=LINEAR;MinFilter=LINEAR;MipFilter=LINEAR;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define getDepth(coords) (ReShade::GetLinearizedDepth(coords))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define PI 3.1415926535
#define PI_HALF 1.57079632679
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

/*---------.
| :: UI :: |
'---------*/

#define DenoiseBlurAmount 0.85

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if UI_DIFFICULTY == 0
#define EnableDenoise 1
#define RadiusMultiplier        1.0
#define FinalValuePower     0.8
#define FalloffRange            0.6 
#define SampleDistributionPower  8.0
#define ThinOccluderCompensation -0.60
#define ComputeBentNormals      false
#define EnableTemporal          true
#define DepthMultiplier         1.0
#define DepthThreshold          0.999
#define bSmoothNormals          false
#define FOV                     35.0
#define HeightmapIntensity      100.0
#define HeightmapBlendAmount    0.0
#endif

uniform float Intensity <
    ui_category = "General";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float RenderScale <
    ui_category = "General";
    ui_label = "Render Scale";
    ui_tooltip = "Renders the effect at a lower resolution for performance, then upscales using FSR 1.0.";
    ui_type = "drag";
    ui_min = 0.5; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform int QualityLevel <
    ui_category = "General";
    ui_label = "Quality Level";
    ui_tooltip = "Defines the number of directions and samples per direction.";
    ui_type = "combo";
    ui_items = "Low (2 directions, 2 samples)\0Medium (3 directions, 4 samples)\0High (4 directions, 8 samples)\0Ultra (8 directions, 8 samples)\0";
> = 2;

uniform float EffectRadius <
    ui_category = "General";
    ui_label = "Effect Radius";
    ui_tooltip = "Main radius of the AO effect in view-space units.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 50.0; ui_step = 0.01;
> = 2.0;

uniform float4 OcclusionColor <
    ui_category = "General";
    ui_type = "color";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform float TemporalAccumulationFrames <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 32.0;
    ui_step = 1.0;
    ui_category = "Temporal Filtering";
    ui_label = "Temporal Accumulation Frames";
    ui_tooltip = "Number of frames to accumulate. Higher values are smoother but may cause more ghosting on moving objects.";
> = 10.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "Normal\0Normals\0View-Space Depth\0Raw AO\0Denoised AO\0Temporal AO\0Upscaled AO\0Edges\0";
    ui_tooltip = "Selects the debug view mode.";
> = 0;

#if UI_DIFFICULTY == 1
uniform float RadiusMultiplier <
    ui_category = "GTAO";
    ui_label = "Radius Multiplier";
    ui_tooltip = "Additional radius multiplier, useful for scaling the effect without changing the base radius.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform float FinalValuePower <
    ui_category = "GTAO";
    ui_label = "Occlusion Power";
    ui_tooltip = "Final occlusion modifier. Higher values make the occlusion stronger.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 8.0; ui_step = 0.1;
> = 0.8;

uniform float FalloffRange <
    ui_category = "GTAO";
    ui_label = "Falloff Range";
    ui_tooltip = "Controls the smoothness of the AO edge, as a percentage of the radius.";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
> = 0.6;

uniform float SampleDistributionPower <
    ui_category = "GTAO Quality";
    ui_label = "Sample Distribution Power";
    ui_tooltip = "Controls how samples are distributed along the view ray. >1 pushes samples further away.";
    ui_type = "drag";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
> = 8.0;

uniform float ThinOccluderCompensation <
    ui_category = "GTAO Quality";
    ui_label = "Thin Occluder Compensation";
    ui_tooltip = "Reduces self-occlusion on thin objects.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = -0.6;

uniform bool ComputeBentNormals <
    ui_category = "GTAO Quality";
    ui_label = "Compute Bent Normals";
> = false;

uniform bool EnableTemporal <
    ui_category = "Temporal Filtering";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Blends the current frame's AO with previous frames to reduce noise and flickering.";
> = true;

uniform bool EnableDenoise <
    ui_category = "Denoiser";
    ui_type = "checkbox";
    ui_label = "Enable XeGTAO Denoiser";
    ui_tooltip = "Apply a spatial denoiser after the temporal filter to further smooth the result.";
> = true;

uniform float DepthMultiplier <
    ui_category = "Advanced";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
    ui_type = "drag";
> = 1.0;

uniform float DepthThreshold <
    ui_category = "Advanced";
    ui_label = "Sky Threshold";
    ui_tooltip = "Sets the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_type = "drag";
> = 0.999;

uniform bool bSmoothNormals <
    ui_category = "Advanced";
    ui_label = "Smooth Normals";
> = false;

uniform float FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set to your game's vertical Field of View for accurate projection calculations.";
> = 35.0;

uniform float HeightmapIntensity <
    ui_category = "Heightmap Normals";
    ui_label = "Heightmap Intensity";
    ui_tooltip = "Controls the strength of heightmap-based normal perturbation.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 200.0; ui_step = 0.1;
> = 100.0;

uniform float HeightmapBlendAmount <
    ui_category = "Heightmap Normals";
    ui_label = "Heightmap Blend Amount";
    ui_tooltip = "How much to blend heightmap normals with geometric normals.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

#endif // UI_DIFFICULTY == 1

uniform float frametime < source = "frametime"; >;
uniform int FRAME_COUNT < source = "framecount"; >;

/*----------------.
| :: Textures :: |
'----------------*/

// Motion Vectors Texture
#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, texcoord).rg;
    }
#else
texture texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler sTexMotionVectorsSampler
{
    Texture = texMotionVectors;S_PC
};
float2 SampleMotionVectors(float2 texcoord)
{
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace XeGTAO
{
    texture NormalT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = NormalT;S_LC
    };

    texture DepthT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
    };
    sampler sDepth
    {
        Texture = DepthT;S_LC
    };

    texture AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sAO
    {
        Texture = AO;S_LC
    };

    // Denoiser Textures
    texture EdgesT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R8;
    };
    sampler sEdges
    {
        Texture = EdgesT;S_LC
    };

    texture DenoiseT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sDenoise
    {
        Texture = DenoiseT;S_LC
    };

    texture DenoiseT0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sDenoiseT0
    {
        Texture = DenoiseT0;S_LC
    };

    // Temporal Filter Textures
    texture TempT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sTemp
    {
        Texture = TempT;S_LC
    };

    texture HistoryT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sHistory
    {
        Texture = HistoryT;S_LC
    };

    // Upscaling Texture
    texture UpscaledT
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler sUpscaled
    {
        Texture = UpscaledT;S_LC
    };

/*----------------.
| :: Functions :: |
'----------------*/

    //Noise
    // https://www.shadertoy.com/view/llGcDm
#if __SHADERMODEL__ >= 40
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

    uint HilbertIndex(uint x, uint y)
    {
        return hilbert(int2(x % 64, y % 64), 6);
    }
#else
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
#endif

    float2 SpatioTemporalNoise(uint2 pixCoord, uint temporalIndex)
    {
        // Hilbert curve driving R2 (see https://www.shadertoy.com/view/3tB3z3)
        uint index = HilbertIndex(pixCoord.x, pixCoord.y);
        
        // why 288? tried out a few and that's the best so far (with XE_HILBERT_LEVEL 6U) - but there's probably better :)
        index += 288 * (temporalIndex % 64);
        
        // R2 sequence - see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        return frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114));
    }
    
    // http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
#if __SHADERMODEL__ >= 40
    half XeGTAO_FastSqrt(float x)
    {
        return (half) (asfloat(0x1fbd1df5 + (asint(x) >> 1)));
    }
#else
    half XeGTAO_FastSqrt(float x)
    {
        return sqrt(x);
    }
#endif
    
    // input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
    half XeGTAO_FastACos(half inX)
    {
        half x = abs(inX);
        half res = -0.156583 * x + PI_HALF;
        res *= XeGTAO_FastSqrt(1.0 - x);
        return (inX >= 0) ? res : PI - res;
    }

    float3 GetViewPos(float2 uv, float linear_depth)
    {
        float view_z = linear_depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
        float fov_rad = FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / BUFFER_ASPECT_RATIO;

        float2 clip_pos = uv * 2.0 - 1.0;

        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z;
        view_pos.z = view_z;

        return view_pos * DepthMultiplier;
    }
    
    float3 XeGTAO_CalculateNormal(half4 edgesLRTB, float3 CENTER, float3 LEFT, float3 RIGHT, float3 TOP, float3 BOTTOM)
    {
        float3 ddx = RIGHT - LEFT;
        float3 ddy = BOTTOM - TOP;
        return normalize(cross(ddy, ddx));
    }

    float3 XeGTAO_ComputeViewspaceNormal(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
    
        float2 uv_down1 = min(texcoord + float2(0, p.y), 1.0 - p);
        float2 uv_down2 = min(texcoord + float2(0, p.y * 2), 1.0 - p);
        float2 uv_up1 = max(texcoord - float2(0, p.y), p);
        float2 uv_up2 = max(texcoord - float2(0, p.y * 2), p);
        float2 uv_right1 = min(texcoord + float2(p.x, 0), 1.0 - p);
        float2 uv_right2 = min(texcoord + float2(p.x * 2, 0), 1.0 - p);
        float2 uv_left1 = max(texcoord - float2(p.x, 0), p);
        float2 uv_left2 = max(texcoord - float2(p.x * 2, 0), p);
    
        float3 center_pos = GetViewPos(texcoord, getDepth(texcoord));
        float3 pos_down1 = GetViewPos(uv_down1, getDepth(uv_down1));
        float3 pos_down2 = GetViewPos(uv_down2, getDepth(uv_down2));
        float3 extrapolated_down = pos_down1 + (pos_down1 - pos_down2);
        float3 pos_up1 = GetViewPos(uv_up1, getDepth(uv_up1));
        float3 pos_up2 = GetViewPos(uv_up2, getDepth(uv_up2));
        float3 extrapolated_up = pos_up1 + (pos_up1 - pos_up2);
        float3 ddy = pos_down1 - center_pos;
        if (abs(extrapolated_up.z - center_pos.z) < abs(extrapolated_down.z - center_pos.z))
        {
            ddy = center_pos - pos_up1;
        }
        float3 pos_right1 = GetViewPos(uv_right1, getDepth(uv_right1));
        float3 pos_right2 = GetViewPos(uv_right2, getDepth(uv_right2));
        float3 extrapolated_right = pos_right1 + (pos_right1 - pos_right2);
        float3 pos_left1 = GetViewPos(uv_left1, getDepth(uv_left1));
        float3 pos_left2 = GetViewPos(uv_left2, getDepth(uv_left2));
        float3 extrapolated_left = pos_left1 + (pos_left1 - pos_left2);
        float3 ddx = pos_right1 - center_pos;
        if (abs(extrapolated_left.z - center_pos.z) < abs(extrapolated_right.z - center_pos.z))
        {
            ddx = center_pos - pos_left1;
        }
        return normalize(cross(ddy, ddx));
    }

    float3 ComputeHeightmapNormal(float h00, float h10, float h20, float h01, float h11, float h21, float h02, float h12, float h22, const float3 pixelWorldSize)
    {
        // Sobel 3x3
        //  0,0 | 1,0 | 2,0
        // ----+-----+----
        //  0,1 | 1,1 | 2,1
        // ----+-----+----
        //  0,2 | 1,2 | 2,2

        h00 -= h11;
        h10 -= h11;
        h20 -= h11;
        h01 -= h11;
        h21 -= h11;
        h02 -= h11;
        h12 -= h11;
        h22 -= h11;
        
        // The Sobel X kernel is:
        //
        // [ 1.0  0.0  -1.0 ]
        // [ 2.0  0.0  -2.0 ]
        // [ 1.0  0.0  -1.0 ]
        
        float Gx = h00 - h20 + 2.0 * h01 - 2.0 * h21 + h02 - h22;
                
        // The Sobel Y kernel is:
        //
        // [  1.0    2.0    1.0 ]
        // [  0.0    0.0    0.0 ]
        // [ -1.0   -2.0   -1.0 ]
        
        float Gy = h00 + 2.0 * h10 + h20 - h02 - 2.0 * h12 - h22;
        
        float stepX = pixelWorldSize.x;
        float stepY = pixelWorldSize.y;
        float sizeZ = pixelWorldSize.z;
        
        Gx = Gx * stepY * sizeZ;
        Gy = Gy * stepX * sizeZ;
        
        float Gz = stepX * stepY * 8;
        
        return normalize(float3(Gx, Gy, Gz));
    }

    float3 GetHeightmapNormal(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
    
        float h00 = GetColor(texcoord + float2(-p.x, -p.y)).r;
        float h10 = GetColor(texcoord + float2(0, -p.y)).r;
        float h20 = GetColor(texcoord + float2(p.x, -p.y)).r;
    
        float h01 = GetColor(texcoord + float2(-p.x, 0)).r;
        float h11 = GetColor(texcoord).r;
        float h21 = GetColor(texcoord + float2(p.x, 0)).r;
    
        float h02 = GetColor(texcoord + float2(-p.x, p.y)).r;
        float h12 = GetColor(texcoord + float2(0, p.y)).r;
        float h22 = GetColor(texcoord + float2(p.x, p.y)).r;

        float3 pixelWorldSize = float3(p.x, p.y, HeightmapIntensity * 0.001);
    
        return ComputeHeightmapNormal(h00, h10, h20, h01, h11, h21, h02, h12, h22, pixelWorldSize);
    }
    
    float XeGTAO_EncodeVisibilityBentNormal(half visibility, half3 bentNormal)
    {
        float4 unpackedInput = float4(bentNormal * 0.5 + 0.5, visibility);
        unpackedInput = saturate(unpackedInput) * 255.0;
        return floor(unpackedInput.x) + floor(unpackedInput.y) * 256.0 + floor(unpackedInput.z) * 65536.0 + floor(unpackedInput.w) * 16777216.0;
    }

    void XeGTAO_DecodeVisibilityBentNormal(const float packedValue, out half visibility, out half3 bentNormal)
    {
        float val = packedValue;
        float r = fmod(val, 256.0);
        val = floor(val / 256.0);
        float g = fmod(val, 256.0);
        val = floor(val / 256.0);
        float b = fmod(val, 256.0);
        val = floor(val / 256.0);
        float a = fmod(val, 256.0);
        
        half4 decoded = half4(r, g, b, a) / 255.0;
        bentNormal = decoded.xyz * 2.0 - 1.0;
        visibility = decoded.w;
    }

    float ReconstructUint(float4 packedFloat)
    {
        packedFloat = round(packedFloat * 255.0);
        return packedFloat.x + packedFloat.y * 256.0 + packedFloat.z * 65536.0 + packedFloat.w * 16777216.0;
    }
    
    float4 ReconstructFloat4(float val)
    {
        float r = fmod(val, 256.0);
        val = floor(val / 256.0);
        float g = fmod(val, 256.0);
        val = floor(val / 256.0);
        float b = fmod(val, 256.0);
        val = floor(val / 256.0);
        float a = fmod(val, 256.0);
        return float4(r, g, b, a) / 255.0;
    }

    half3x3 XeGTAO_RotFromToMatrix(half3 from, half3 to)
    {
        const half e = dot(from, to);
        const half f = abs(e);

        if (f > half(1.0 - 0.0003))
            return half3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);

        const half3 v = cross(from, to);
        const half h = (1.0) / (1.0 + e);
        const half hvx = h * v.x;
        const half hvz = h * v.z;
        const half hvxy = hvx * v.y;
        const half hvxz = hvx * v.z;
        const half hvyz = hvz * v.y;

        half3x3 mtx;
        mtx[0][0] = e + hvx * v.x;
        mtx[0][1] = hvxy - v.z;
        mtx[0][2] = hvxz + v.y;
        mtx[1][0] = hvxy + v.z;
        mtx[1][1] = e + h * v.y * v.y;
        mtx[1][2] = hvyz - v.x;
        mtx[2][0] = hvxz - v.y;
        mtx[2][1] = hvyz + v.x;
        mtx[2][2] = e + hvz * v.z;
        return mtx;
    }

    half4 XeGTAO_CalculateEdges(const half centerZ, const half leftZ, const half rightZ, const half topZ, const half bottomZ)
    {
        half4 edgesLRTB = half4(leftZ, rightZ, topZ, bottomZ) - centerZ;
        half slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
        half slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
        half4 edgesLRTBSlopeAdjusted = edgesLRTB + half4(slopeLR, -slopeLR, slopeTB, -slopeTB);
        edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
        return saturate((1.25 - edgesLRTB / (centerZ * 0.011)));
    }

    half XeGTAO_PackEdges(half4 edgesLRTB)
    {
        edgesLRTB = round(saturate(edgesLRTB) * 2.9);
        return dot(edgesLRTB, half4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
    }

#if __SHADERMODEL__ >= 40
    half4 XeGTAO_UnpackEdges(half _packedVal)
    {
        uint packedVal = (uint) (_packedVal * 255.5);
        half4 edgesLRTB;
        edgesLRTB.x = half((packedVal >> 6) & 0x03) / 3.0;
        edgesLRTB.y = half((packedVal >> 4) & 0x03) / 3.0;
        edgesLRTB.z = half((packedVal >> 2) & 0x03) / 3.0;
        edgesLRTB.w = half((packedVal >> 0) & 0x03) / 3.0;
        return saturate(edgesLRTB);
    }
#else
    half4 XeGTAO_UnpackEdges(half _packedVal)
    {
        float packedVal = (_packedVal * 255.5);
        half4 edgesLRTB;
        edgesLRTB.x = fmod(floor(packedVal / 64.0), 4.0) / 3.0;
        edgesLRTB.y = fmod(floor(packedVal / 16.0), 4.0) / 3.0;
        edgesLRTB.z = fmod(floor(packedVal / 4.0), 4.0) / 3.0;
        edgesLRTB.w = fmod(packedVal, 4.0) / 3.0;
        return saturate(edgesLRTB);
    }
#endif
    
    void XeGTAO_AddSample(half ssaoValue, half edgeValue, inout half sum, inout half sumWeight)
    {
        half weight = edgeValue;
        sum += (weight * ssaoValue);
        sumWeight += weight;
    }
    
    float XeGTAO_Denoise(float2 texcoord)
    {
        half center_ao = GetLod(sDenoise, float4(texcoord, 0.0, 0.0)).r;
        
        if (!EnableDenoise)
        {
            return center_ao;
        }

        const half diagWeight = 0.85 * 0.5;
        const float2 p = ReShade::PixelSize / RenderScale;

        half4 edgesC_LRTB = XeGTAO_UnpackEdges(GetLod(sEdges, texcoord).r);
        half4 edgesL_LRTB = XeGTAO_UnpackEdges(GetLod(sEdges, texcoord - float2(p.x, 0)).r);
        half4 edgesR_LRTB = XeGTAO_UnpackEdges(GetLod(sEdges, texcoord + float2(p.x, 0)).r);
        half4 edgesT_LRTB = XeGTAO_UnpackEdges(GetLod(sEdges, texcoord - float2(0, p.y)).r);
        half4 edgesB_LRTB = XeGTAO_UnpackEdges(GetLod(sEdges, texcoord + float2(0, p.y)).r);
        
        edgesC_LRTB *= half4(edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z);
        
        half weightTL = diagWeight * (edgesC_LRTB.x * edgesL_LRTB.z + edgesC_LRTB.z * edgesT_LRTB.x);
        half weightTR = diagWeight * (edgesC_LRTB.z * edgesT_LRTB.y + edgesC_LRTB.y * edgesR_LRTB.z);
        half weightBL = diagWeight * (edgesC_LRTB.w * edgesB_LRTB.x + edgesC_LRTB.x * edgesL_LRTB.w);
        half weightBR = diagWeight * (edgesC_LRTB.y * edgesR_LRTB.w + edgesC_LRTB.w * edgesB_LRTB.y);

        half ssaoValueL = tex2Doffset(sDenoise, texcoord, int2(-1, 0)).r;
        half ssaoValueR = tex2Doffset(sDenoise, texcoord, int2(1, 0)).r;
        half ssaoValueT = tex2Doffset(sDenoise, texcoord, int2(0, -1)).r;
        half ssaoValueB = tex2Doffset(sDenoise, texcoord, int2(0, 1)).r;
        half ssaoValueTL = tex2Doffset(sDenoise, texcoord, int2(-1, -1)).r;
        half ssaoValueTR = tex2Doffset(sDenoise, texcoord, int2(1, -1)).r;
        half ssaoValueBL = tex2Doffset(sDenoise, texcoord, int2(-1, 1)).r;
        half ssaoValueBR = tex2Doffset(sDenoise, texcoord, int2(1, 1)).r;

        half sumWeight = 1.0 - DenoiseBlurAmount;
        half sum = center_ao * sumWeight;

        XeGTAO_AddSample(ssaoValueL, edgesC_LRTB.x, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueR, edgesC_LRTB.y, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueT, edgesC_LRTB.z, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueB, edgesC_LRTB.w, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueTL, weightTL, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueTR, weightTR, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueBL, weightBL, sum, sumWeight);
        XeGTAO_AddSample(ssaoValueBR, weightBR, sum, sumWeight);

        return sum / sumWeight;
    }

    float3 blend_normals(float3 n1, float3 n2)
    {
        n1 += float3(0, 0, 1);
        n2 *= float3(-1, -1, 1);
        return n1 * dot(n1, n2) / n1.z - n2;
    }
    
/*--------------------.
| :: Pixel Shaders :: |
'--------------------*/
    
    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 geomNormal = -XeGTAO_ComputeViewspaceNormal(uv);
        float3 heightmapNormal = GetHeightmapNormal(uv);
        float3 blended = blend_normals(heightmapNormal, geomNormal);
        float3 finalNormal = normalize(lerp(geomNormal, blended, HeightmapBlendAmount));
        
        return float4(saturate(finalNormal * 0.5 + 0.5), 1.0);
    }

    float PS_ViewDepth(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return getDepth(uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    }
    
    float PS_Edges(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 0.0;
        }
        float2 scaled_uv = uv / RenderScale;
        float2 p = ReShade::PixelSize / RenderScale;
        
        half centerZ = GetLod(sDepth, float4(scaled_uv, 0, 0)).r;
        half leftZ = GetLod(sDepth, float4(scaled_uv - float2(p.x, 0), 0, 0)).r;
        half rightZ = GetLod(sDepth, float4(scaled_uv + float2(p.x, 0), 0, 0)).r;
        half topZ = GetLod(sDepth, float4(scaled_uv - float2(0, p.y), 0, 0)).r;
        half bottomZ = GetLod(sDepth, float4(scaled_uv + float2(0, p.y), 0, 0)).r;
        
        half4 edgesLRTB = XeGTAO_CalculateEdges(centerZ, leftZ, rightZ, topZ, bottomZ);
        return XeGTAO_PackEdges(edgesLRTB);
    }

    float4 PS_GTAO_Main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 0;
        }
        float2 scaled_uv = uv / RenderScale;

        float viewspaceZ_raw = GetLod(sDepth, float4(scaled_uv, 0, 0)).r;
        float2 p = ReShade::PixelSize / RenderScale;

        int sliceCount, stepsPerSlice;
        if (QualityLevel == 0)
        {
            sliceCount = 2;
            stepsPerSlice = 2;
        }
        else if (QualityLevel == 1)
        {
            sliceCount = 3;
            stepsPerSlice = 4;
        }
        else if (QualityLevel == 2)
        {
            sliceCount = 4;
            stepsPerSlice = 8;
        }
        else
        {
            sliceCount = 8;
            stepsPerSlice = 8;
        }

        float3 pixCenterPos = GetViewPos(scaled_uv, viewspaceZ_raw / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
        float viewspaceZ = pixCenterPos.z;

        if (getDepth(scaled_uv) >= DepthThreshold)
        {
            float encodedValue = XeGTAO_EncodeVisibilityBentNormal(1.0, float3(0, 0, 1));
            return ReconstructFloat4(encodedValue);
        }

        half3 viewspaceNormal = GetLod(sNormal, float4(scaled_uv, 0, 0)).xyz * 2.0 - 1.0;
        const half3 viewVec = (half3) normalize(-pixCenterPos);

        const half effectRadius = (half) EffectRadius * (half) RadiusMultiplier;
        const half falloffRange = (half) FalloffRange * effectRadius;
        const half falloffFrom = effectRadius * (1.0 - (half) FalloffRange);
        const half falloffMul = -1.0 / falloffRange;
        const half falloffAdd = falloffFrom / falloffRange + 1.0;
        const half sampleDistributionPower = (half) SampleDistributionPower;
        const half thinOccluderCompensation = (half) ThinOccluderCompensation;

        half visibility = 0;
        half3 bentNormal = 0;

        uint2 pixCoord = pos.xy;
        float2 localNoise = SpatioTemporalNoise(pixCoord, FRAME_COUNT);

        const half noiseSlice = (half) localNoise.x;
        const half noiseSample = (half) localNoise.y;

        const float fov_rad = FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / BUFFER_ASPECT_RATIO;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * (float2(2.0 / proj_scale_x, 2.0 / proj_scale_y) / (BUFFER_SCREEN_SIZE * RenderScale));

        half screenspaceRadius = effectRadius / (half) pixelDirRBViewspaceSizeAtCenterZ.x;
        
        visibility += saturate((10.0 - screenspaceRadius) / 100.0) * 0.5;

        const half pixelTooCloseThreshold = 1.3;
        const half minS = screenspaceRadius > 0 ? (pixelTooCloseThreshold / screenspaceRadius) : 0.0;

        for (half slice = 0; slice < sliceCount; slice++)
        {
            half sliceK = (slice + noiseSlice) / sliceCount;
            half phi = sliceK * PI;
            half cosPhi, sinPhi;
            sincos(phi, sinPhi, cosPhi);
            half2 omega = half2(cosPhi, -sinPhi);

            omega *= screenspaceRadius;

            const half3 directionVec = half3(cosPhi, sinPhi, 0);
            const half3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
            const half3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
            half3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

            half signNorm = (half) sign(dot(orthoDirectionVec, projectedNormalVec));
            half projectedNormalVecLength = length(projectedNormalVec);
            half cosNorm = saturate(dot(projectedNormalVec, viewVec) / max(projectedNormalVecLength, 1e-4));

            half n = signNorm * XeGTAO_FastACos(cosNorm);

            const half lowHorizonCos0 = cos(n + PI_HALF);
            const half lowHorizonCos1 = cos(n - PI_HALF);

            half horizonCos0 = lowHorizonCos0;
            half horizonCos1 = lowHorizonCos1;

            [unroll]
            for (half step = 0; step < stepsPerSlice; step++)
            {
                const half stepBaseNoise = (slice + step * stepsPerSlice) * 0.61803398875;
                half stepNoise = frac(noiseSample + stepBaseNoise);
                half s = (step + stepNoise) / stepsPerSlice;
                s = pow(s, sampleDistributionPower);
                s += minS;

                float2 sampleOffset = round(s * omega) * (ReShade::PixelSize / RenderScale);

                float2 sampleScreenPos0 = scaled_uv + sampleOffset;
                float SZ0 = GetLod(sDepth, float4(sampleScreenPos0, 0, 0)).r;
                float3 samplePos0 = GetViewPos(sampleScreenPos0, SZ0 / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

                float2 sampleScreenPos1 = scaled_uv - sampleOffset;
                float SZ1 = GetLod(sDepth, float4(sampleScreenPos1, 0, 0)).r;
                float3 samplePos1 = GetViewPos(sampleScreenPos1, SZ1 / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
                
                float3 sampleDelta0 = (samplePos0 - pixCenterPos);
                float3 sampleDelta1 = (samplePos1 - pixCenterPos);
                half sampleDist0 = (half) length(sampleDelta0);
                half sampleDist1 = (half) length(sampleDelta1);

                half3 sampleHorizonVec0 = (half3) (sampleDelta0 / max(sampleDist0, 1e-6));
                half3 sampleHorizonVec1 = (half3) (sampleDelta1 / max(sampleDist1, 1e-6));

                half falloffBase0 = length(half3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1.0 + thinOccluderCompensation)));
                half falloffBase1 = length(half3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1.0 + thinOccluderCompensation)));
                half weight0 = saturate(falloffBase0 * falloffMul + falloffAdd);
                half weight1 = saturate(falloffBase1 * falloffMul + falloffAdd);

                half shc0 = dot(sampleHorizonVec0, viewVec);
                half shc1 = dot(sampleHorizonVec1, viewVec);

                shc0 = lerp(lowHorizonCos0, shc0, weight0);
                shc1 = lerp(lowHorizonCos1, shc1, weight1);

                horizonCos0 = max(horizonCos0, shc0);
                horizonCos1 = max(horizonCos1, shc1);
            }

            projectedNormalVecLength = lerp(projectedNormalVecLength, 1.0, 1.8); // remind

            half h0 = -XeGTAO_FastACos(clamp(horizonCos1, -1.0, 1.0));
            half h1 = XeGTAO_FastACos(clamp(horizonCos0, -1.0, 1.0));

            half iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
            half iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
            half localVisibility = projectedNormalVecLength * (iarc0 + iarc1);
            visibility += localVisibility;

            if (ComputeBentNormals)
            {
                half t0 = (6.0 * sin(h0 - n) - sin(3.0 * h0 - n) + 6.0 * sin(h1 - n) - sin(3.0 * h1 - n) + 16.0 * sin(n) - 3.0 * (sin(h0 + n) + sin(h1 + n))) / 12.0;
                half t1 = (-cos(3.0 * h0 - n) - cos(3.0 * h1 - n) + 8.0 * cos(n) - 3.0 * (cos(h0 + n) + cos(h1 + n))) / 12.0;
                half3 localBentNormal = half3(directionVec.x * t0, directionVec.y * t0, -t1);
                localBentNormal = mul(XeGTAO_RotFromToMatrix(half3(0, 0, -1), viewVec), localBentNormal) * projectedNormalVecLength;
                bentNormal += localBentNormal;
            }
        }
        
        visibility /= (half) sliceCount;
        visibility = pow(saturate(visibility), (half) FinalValuePower);
        visibility = max(0.03, visibility);

        if (ComputeBentNormals)
        {
            bentNormal = normalize(bentNormal + half3(0, 0, 1e-6));
        }
        else
        {
            bentNormal = half3(0, 0, 1);
        }

        float encodedValue = XeGTAO_EncodeVisibilityBentNormal(visibility, bentNormal);
        return ReconstructFloat4(encodedValue);
    }

    float PS_PrepareDenoise(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 1.0;
        }

        float4 packedAO = GetLod(sAO, uv);
        float encodedValue = ReconstructUint(packedAO);
        half visibility, bentNormal;
        XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
        return visibility;
    }
    
    float PS_Denoise(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (any(texcoord > RenderScale))
        {
            return GetLod(sDenoise, float4(texcoord, 0.0, 0.0)).r;
        }
        return XeGTAO_Denoise(texcoord);
    }

    float PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
        {
            return 1.0;
        }

        float currentAO = GetLod(sDenoiseT0, uv).r;

        if (!EnableTemporal)
        {
            return currentAO;
        }
        
        float2 full_res_uv = uv / RenderScale;
        
        float closest_view_depth = GetLod(sDepth, full_res_uv).r;
        float2 motion = SampleMotionVectors(full_res_uv);

        [unroll]
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                if (x == 0 && y == 0)
                    continue;

                float2 offset_uv = full_res_uv + float2(x, y) * ReShade::PixelSize;
                float neighbor_view_depth = GetLod(sDepth, offset_uv).r;

                if (neighbor_view_depth < closest_view_depth)
                {
                    closest_view_depth = neighbor_view_depth;
                    motion = SampleMotionVectors(offset_uv);
                }
            }
        }
        
        float2 reprojected_uv_full = full_res_uv + motion;
        float currentViewDepth = GetLod(sDepth, full_res_uv).r;
        float historyViewDepth = GetLod(sDepth, reprojected_uv_full).r;
        float2 reprojected_uv_low = reprojected_uv_full * RenderScale;

        bool validHistory = all(saturate(reprojected_uv_low) == reprojected_uv_low) &&
                                        FRAME_COUNT > 1 &&
                                        abs(historyViewDepth - currentViewDepth) < (currentViewDepth * 0.02);

        float blendedAO = currentAO;
        if (validHistory)
        {
            float historyAO = GetLod(sHistory, reprojected_uv_low).r;

            float minBox = currentAO, maxBox = currentAO;
            float2 low_res_pixel_size = ReShade::PixelSize / RenderScale;

            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    float2 neighbor_uv = uv + float2(x, y) * low_res_pixel_size;
                    float neighborAO = GetLod(sDenoiseT0, neighbor_uv).r;
                    minBox = min(minBox, neighborAO);
                    maxBox = max(maxBox, neighborAO);
                }
            }

            float center = (minBox + maxBox) * 0.5;
            float extents = (maxBox - minBox) * 0.5;
            extents += 0.01;
            minBox = center - extents;
            maxBox = center + extents;

            float clampedHistoryAO = clamp(historyAO, minBox, maxBox);
            
            float alpha = 1.0 / min((float) FRAME_COUNT, TemporalAccumulationFrames);
            
            float rejection_dist = abs(historyAO - clampedHistoryAO);
            float rejection_factor = saturate(rejection_dist * 8.0);
            alpha = max(alpha, rejection_factor);

            blendedAO = lerp(clampedHistoryAO, currentAO, alpha);
        }
        
        return blendedAO;
    }
    
    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float outHistory : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outHistory = 1.0;
            return;
        }
        outHistory = GetLod(sTemp, uv).r;
    }

    void PS_Upscale(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outUpscaled : SV_Target)
    {
        if (RenderScale >= 1.0)
        {
            outUpscaled = GetLod(sTemp, uv);
            return;
        }
        
        // Simple bilinear upscaling 
        float2 scaled_uv = uv * RenderScale;
        outUpscaled = GetLod(sTemp, scaled_uv);
    }

    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);

        if (ViewMode == 0) // Normal
        {
            if (getDepth(uv) >= DepthThreshold)
                return originalColor;

            half visibility = GetLod(sUpscaled, uv).r;
            float occlusion = 1.0 - visibility;
            occlusion = saturate(occlusion * Intensity);

            float fade = saturate(1.0 - smoothstep(0.95, 1.0, getDepth(uv)));
            occlusion *= fade;

            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, occlusion);
            return originalColor;
        }
        // Debug
        else if (ViewMode == 1) // Normals
        {
            return float4(GetLod(sNormal, uv).rgb, 1.0);
        }
        else if (ViewMode == 2) // View-Space Depth
        {
            float depth = GetLod(sDepth, uv).r / (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * DepthMultiplier);
            return float4(saturate(depth.rrr), 1.0);
        }
        else if (ViewMode == 3) // Raw AO
        {
            float4 packedAO = GetLod(sAO, uv);
            float encodedValue = ReconstructUint(packedAO);
            half visibility, bentNormal;
            XeGTAO_DecodeVisibilityBentNormal(encodedValue, visibility, bentNormal);
            return float4(visibility.xxx, 1.0);
        }
        else if (ViewMode == 4) // Denoised AO
        {
            if (EnableDenoise)
                return float4(GetLod(sDenoiseT0, uv).rrr, 1.0);
            else
                return float4(0.0, 1.0, 0.0, 1.0); // Green screen to indicate denoiser is off
        }
        else if (ViewMode == 5) // Temporal
        {
            if (EnableTemporal)
                return float4(GetLod(sTemp, uv).rrr, 1.0);
            else
                return float4(0.0, 1.0, 0.0, 1.0); // Green screen to indicate temporal is off
        }
        else if (ViewMode == 6) // Upscaled AO
        {
            return float4(GetLod(sUpscaled, uv).rrr, 1.0);
        }
        else if (ViewMode == 7) // Edges
        {
            return float4(XeGTAO_UnpackEdges(GetLod(sEdges, uv).r).xyz, 1.0);
        }

        return originalColor;
    }

    technique XeGTAO< ui_tooltip = "Need Motion Vectors like Zenteon Motion to top"; >
    {
        pass Normal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = NormalT;
        }
        pass Depth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ViewDepth;
            RenderTarget = DepthT;
        }
        pass Edges
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Edges;
            RenderTarget = EdgesT;
        }
        pass GTAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GTAO_Main;
            RenderTarget = AO;
            ClearRenderTargets = true;
        }
        pass Prepare_Denoise
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_PrepareDenoise;
            RenderTarget = DenoiseT;
            ClearRenderTargets = true;
        }
        pass Denoise_O
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Denoise;
            RenderTarget = DenoiseT0;
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Accumulate;
            RenderTarget = TempT;
            ClearRenderTargets = true;
        }
        pass History
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HistoryT;
            ClearRenderTargets = true;
        }
        pass Upscale
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale;
            RenderTarget = UpscaledT;
        }
        
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
