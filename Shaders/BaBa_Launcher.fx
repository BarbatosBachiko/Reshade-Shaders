/*----------------------------------------------|
| :: BaBa Launcher                           :: |
|-----------------------------------------------|
| Version: 1.0.2                                |
| Author: Barbatos                              |
| License: MIT                                  |
|-----------------------------------------------|
| Description:                                  |
| Shared prepass for linear depth, smoothed     |
| normals and dense optical flow.               |
'----------------------------------------------*/
// The Dense Real-time Optical is (Flow shader derived from umar-afzaal' LumaFlow.)

#ifdef _BABA_USE_LAUNCHER
    #undef _BABA_USE_LAUNCHER
#endif
#define _BABA_USE_LAUNCHER 1
#define _BABA_LAUNCHER_PROVIDER 1

#include ".\Includes\bb_reshade.fxh"
#include ".\Includes\bb_common.fxh"
#include ".\Includes\bb_colorspace.fxh"
#include ".\Includes\bb_depth.fxh"
#include ".\Includes\bb_normal.fxh"
#include ".\Includes\bb_launcher.fxh"

#define EPSILON 1e-6

#ifndef BABA_LAUNCHER_DEBUG
    #define BABA_LAUNCHER_DEBUG 0
#endif

#ifndef BABA_FLOW_SEARCH_ITER
    #define BABA_FLOW_SEARCH_ITER 4
#endif

uniform int FLOW_QUALITY <
    ui_type = "slider";
    ui_category = "Quality";
    ui_label = "Optical Flow Quality";
    ui_min = 1; ui_max = 3; ui_step = 1;
    ui_tooltip = "Quality of the shared dense optical flow used for motion vectors and temporal effects.\n"
                 "1 = Low, 2 = Medium, 3 = High.\nHigher quality searches/filters with more samples but is slower.";
> = 3;

uniform int NORMAL_SMOOTHING <
    ui_type = "slider";
    ui_category = "Quality";
    ui_label = "Normal Smoothing";
    ui_min = 1; ui_max = 3; ui_step = 1;
    ui_tooltip = "The shared normal smoothing used by dependent shaders.\n"
                 "1 = Off, 2 = Medium (one a-trous iteration), 3 = High (two iterations).";
> = 3;

#define FLOW_LEVEL (clamp(FLOW_QUALITY, 1, 3) - 1)
#define NORMAL_SMOOTH_LEVEL (clamp(NORMAL_SMOOTHING, 1, 3) - 1)


#if BABA_LAUNCHER_DEBUG
    uniform int DEBUG_VIEW <
        ui_type = "combo";
        ui_category = "Debug";
        ui_items = "Debug Off\0"
                   "Optical Flow\0"
                   "Motion Vectors\0"
                   "Confidence Map\0"
                   "Linear Depth\0"
                   "Smoothed Normals\0"
                   "View Depth\0"
                   "Encoded Normal (Decoded)\0";
    ui_label = "Debug View";
> = 0;
#endif

static const float NORMAL_SMOOTH_THRESHOLD = 0.250;
static const float RECONSTRUCTION_FOV = 60.0;

uniform bool NORMAL_TEMPORAL <
    ui_category = "Normal Map";
    ui_label = "Temporal Normal Stabilization";
    ui_tooltip = "Accumulates the shared normal map using motion, normal agreement and depth rejection. Helps temporal effects ignore depth jitter without spatially smearing edges.";
> = true;

uniform float NORMAL_TEMPORAL_STABILITY <
    ui_type = "slider";
    ui_category = "Normal Map";
    ui_label = "Normal Temporal Stability";
    ui_min = 0.0; ui_max = 0.98; ui_step = 0.01;
    ui_tooltip = "Higher values suppress depth-jitter flicker; lower values react faster to geometry changes.";
> = 0.88;

uniform float TEXTURE_NORMAL_INTENSITY <
    ui_type = "drag";
    ui_category = "Normal Map";
    ui_label = "Textured Normals (Bump Mapping)";
    ui_tooltip = "Perturbs the shared normal map with fine texture detail from the scene, adding micro-bumps that every effect reading the Launcher's normal (AO, GI, etc.) can see.\n"
                 "BaBa: SSR is genuinely excluded: it reads the Launcher's geometric normal and applies its own equivalent (Micro-Surface Details), so this setting never stacks on top of it. Tune the two independently.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 0.0;

uniform float TEXTURE_NORMAL_EDGE_THRESHOLD <
    ui_type = "drag";
    ui_category = "Normal Map";
    ui_label = "Textured Normals Edge Threshold";
    ui_tooltip = "Ignores soft color changes when applying Textured Normals. Higher values mean bumps only appear on harsh texture edges.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.0;

namespace Barbatos_Flow
{
    texture2D tCurrLuma
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
        MipLevels = 8;
    };
    sampler2D sCurrLuma
    {
        Texture = tCurrLuma;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
        AddressU = CLAMP;
        AddressV = CLAMP;
        AddressW = CLAMP;
    };
    texture2D tPrevLuma
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
        MipLevels = 8;
    };
    sampler2D sPrevLuma
    {
        Texture = tPrevLuma;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
        AddressU = CLAMP;
        AddressV = CLAMP;
        AddressW = CLAMP;
    };
    texture2D tCoarseFlowL4 { Width = BUFFER_WIDTH / 128; Height = BUFFER_HEIGHT / 128; Format = RG16F; };
    sampler2D sCoarseFlowL4 { Texture = tCoarseFlowL4; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL3_A { Width = BUFFER_WIDTH / 64; Height = BUFFER_HEIGHT / 64; Format = RG16F;
    };
    sampler2D sCoarseFlowL3_A { Texture = tCoarseFlowL3_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL3_B { Width = BUFFER_WIDTH / 64; Height = BUFFER_HEIGHT / 64; Format = RG16F;
    };
    sampler2D sCoarseFlowL3_B { Texture = tCoarseFlowL3_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL2_A { Width = BUFFER_WIDTH / 32; Height = BUFFER_HEIGHT / 32; Format = RG16F;
    };
    sampler2D sCoarseFlowL2_A { Texture = tCoarseFlowL2_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL2_B { Width = BUFFER_WIDTH / 32; Height = BUFFER_HEIGHT / 32; Format = RG16F;
    };
    sampler2D sCoarseFlowL2_B { Texture = tCoarseFlowL2_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL1_A { Width = BUFFER_WIDTH / 16; Height = BUFFER_HEIGHT / 16; Format = RG16F;
    };
    sampler2D sCoarseFlowL1_A { Texture = tCoarseFlowL1_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL1_B { Width = BUFFER_WIDTH / 16; Height = BUFFER_HEIGHT / 16; Format = RG16F;
    };
    sampler2D sCoarseFlowL1_B { Texture = tCoarseFlowL1_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL0_A { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F;
    };
    sampler2D sCoarseFlowL0_A { Texture = tCoarseFlowL0_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tCoarseFlowL0_B { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F;
    };
    sampler2D sCoarseFlowL0_B { Texture = tCoarseFlowL0_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tDenseFlow_A { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F;
    };
    sampler2D sDenseFlow_A { Texture = tDenseFlow_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };
    texture2D tDenseFlow_M { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F;
    };
    sampler2D sDenseFlow_M { Texture = tDenseFlow_M; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tDenseFlow_B { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F;
    };
    sampler2D sFinalFlow { Texture = tDenseFlow_B; MagFilter = LINEAR; MinFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    texture2D tGlobalFlow { Width = 1; Height = 1; Format = RG16F; };
    sampler2D sGlobalFlow { Texture = tGlobalFlow; };

    texture2D tPrevFlow { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4;
    Format = RG16F; };
    sampler2D sPrevFrameFlow { Texture = tPrevFlow; MagFilter = POINT; MinFilter = POINT; };
    texture2D tConfidence { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = R16F; };
    sampler2D sConfidence { Texture = tConfidence; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };
    texture2D tPrevConfidence { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = R16F; };
    sampler2D sPrevConfidence { Texture = tPrevConfidence; MagFilter = LINEAR; MinFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP;
    AddressW = CLAMP; };

    //----------------|
    // :: Functions ::|
    //----------------|
#if BABA_LAUNCHER_DEBUG
    float3 MotionToColor(float2 motion)
    {
        float angle = atan2(-motion.y, -motion.x) / 6.283 + 0.5;
        float raw_length = length(motion) / (15.0 * bb::PixelSize.x);
        float compressed = raw_length / (1.0 + raw_length * 1.4);
        float boosted = pow(compressed, 0.5);
        float magnitude = saturate(lerp(compressed, boosted, saturate(raw_length * 3.0)));
        float3 hsv = float3(angle, 1, magnitude);
        float4 K = float4(1, 2.0 / 3.0, 1.0 / 3.0, 3);
        float3 p = abs(frac(hsv.xxx + K.xyz) * 6 - K.www);
        return hsv.z * lerp(K.xxx, clamp(p - K.xxx, 0, 1), hsv.y) + 0.1;
    }

    // Distance from a point to a line segment, used to draw the motion-vector arrow shaft.
    float sdSegment(float2 p, float2 a, float2 b)
    {
        float2 pa = p - a;
        float2 ba = b - a;
        float h = saturate(dot(pa, ba) / max(dot(ba, ba), 1e-6));
        return length(pa - ba * h);
    }

    // Signed distance to a triangle (negative inside), based on Inigo Quilez's sdTriangle.
    float sdTriangle(float2 p, float2 p0, float2 p1, float2 p2)
    {
        float2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
        float2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
        float2 pq0 = v0 - e0 * saturate(dot(v0, e0) / dot(e0, e0));
        float2 pq1 = v1 - e1 * saturate(dot(v1, e1) / dot(e1, e1));
        float2 pq2 = v2 - e2 * saturate(dot(v2, e2) / dot(e2, e2));
        float s = sign(e0.x * e2.y - e0.y * e2.x);
        float2 d0 = float2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
        float2 d1 = float2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
        float2 d2 = float2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
        float2 d = min(min(d0, d1), d2);
        return -sqrt(d.x) * sign(d.y);
    }

    float ArrowCoverage(float dist, float aaWidth)
    {
        return saturate(0.5 - dist / max(aaWidth, 1e-4));
    }
#endif // BABA_LAUNCHER_DEBUG

    bool IsOOB(float2 uv) { return any(uv < 0.0) || any(uv > 1.0); }

    static const float FLOW_MAX_MAGNITUDE = 0.25;
    static const float FLOW_SEARCH_EARLY_OUT = 0.01;

    bool IsInvalidFlow(float2 flow)
    {
        return any(flow != flow) || any(abs(flow) > FLOW_MAX_MAGNITUDE);
    }

#if (__RENDERER__ < 0xA000)
    int ZAD_Ring8(int x) { return (x >= 8) ? (x - 8) : x; }
#else
    int ZAD_Ring8(int x) { return x & 7; }
#endif

    //Zero-mean Absolute Difference
    void ZAD_BuildRef(sampler2D cur, float2 pos_a, float2 texel_size, int mip, out float refWindow[9])
    {
        [unroll]
        for (int z = 0; z < 9; z++)
            refWindow[z] = 0.0;

        if (FLOW_LEVEL == 0)
        {
            refWindow[0] = tex2Dlod(cur, float4(pos_a, 0, mip)).r;
        }
        else if (FLOW_LEVEL == 1)
        {
            const int2 offsM[5] = { int2(0,-2), int2(-2,0), int2(0,0), int2(2,0), int2(0,2) };
            float meanM = 0.0;
            [unroll]
            for (int i = 0; i < 5; i++)
            {
                refWindow[i] = tex2Dlod(cur, float4(pos_a + float2(offsM[i]) * texel_size, 0, mip)).r;
                meanM += refWindow[i];
            }
            meanM /= 5.0;
            [unroll]
            for (int i = 0; i < 5; i++)
                refWindow[i] -= meanM;
        }
        else
        {
            const int2 offsH[9] = { int2(0,-3), int2(0,-1), int2(-3,0), int2(-1,0), int2(0,0), int2(1,0), int2(3,0), int2(0,1), int2(0,3) };
            float meanH = 0.0;
            [unroll]
            for (int i = 0; i < 9; i++)
            {
                refWindow[i] = tex2Dlod(cur, float4(pos_a + float2(offsH[i]) * texel_size, 0, mip)).r;
                meanH += refWindow[i];
            }
            meanH /= 9.0;
            [unroll]
            for (int i = 0; i < 9; i++)
                refWindow[i] -= meanH;
        }
    }

    float ZAD_Cost(sampler2D prev, float2 pos_b, float2 texel_size, int mip, const float refWindow[9])
    {
        float result = EPSILON;
        float samples[9];
        float mean = 0.0;
        float err = 0.0;

        if (FLOW_LEVEL == 0)
        {
            result = abs(refWindow[0] - tex2Dlod(prev, float4(pos_b, 0, mip)).r) + EPSILON;
        }
        else if (FLOW_LEVEL == 1)
        {
            const int2 offsM[5] = { int2(0,-2), int2(-2,0), int2(0,0), int2(2,0), int2(0,2) };
            [unroll]
            for (int i = 0; i < 5; i++)
            {
                samples[i] = tex2Dlod(prev, float4(pos_b + float2(offsM[i]) * texel_size, 0, mip)).r;
                mean += samples[i];
            }
            mean /= 5.0;
            [unroll]
            for (int i = 0; i < 5; i++)
                err += abs(refWindow[i] - (samples[i] - mean));
            result = (err / 5.0) + EPSILON;
        }
        else
        {
            const int2 offsH[9] = { int2(0,-3), int2(0,-1), int2(-3,0), int2(-1,0), int2(0,0), int2(1,0), int2(3,0), int2(0,1), int2(0,3) };
            [unroll]
            for (int i = 0; i < 9; i++)
            {
                samples[i] = tex2Dlod(prev, float4(pos_b + float2(offsH[i]) * texel_size, 0, mip)).r;
                mean += samples[i];
            }
            mean /= 9.0;
            [unroll]
            for (int i = 0; i < 9; i++)
                err += abs(refWindow[i] - (samples[i] - mean));
            result = (err / 9.0) + EPSILON;
        }

        return result;
    }

    void ZAD_Consider(float2 candidate, float2 uv, float2 texel_size, int mip, const float refWindow[9],
                      inout float2 best, inout float best_cost)
    {
        float cost = ZAD_Cost(sPrevLuma, uv + candidate, texel_size, mip, refWindow);
        if (cost < best_cost)
        {
            best_cost = cost;
            best = candidate;
        }
    }

    #define BB_S2(a, b) { float2 bb_tmp = (a); (a) = min((a), (b)); (b) = max(bb_tmp, (b)); }
    #define BB_MN3(a, b, c) BB_S2(a, b) BB_S2(a, c)
    #define BB_MX3(a, b, c) BB_S2(b, c) BB_S2(a, c)
    #define BB_MNMX3(a, b, c) BB_MX3(a, b, c) BB_S2(a, b)
    #define BB_MNMX4(a, b, c, d) BB_S2(a, b) BB_S2(c, d) BB_S2(a, c) BB_S2(b, d)
    #define BB_MNMX5(a, b, c, d, e) BB_S2(a, b) BB_S2(c, d) BB_MN3(a, c, e) BB_MX3(b, d, e)
    #define BB_MNMX6(a, b, c, d, e, f) BB_S2(a, d) BB_S2(b, e) BB_S2(c, f) BB_MN3(a, b, c) BB_MX3(d, e, f)

    float2 Median(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
    {
        float2 result = tex2Dlod(motion_tex, float4(uv, 0, mip)).xy;

        if (FLOW_LEVEL == 1)
        {
            const int2 MEDIAN_OFFS[5] = { int2(0,0), int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };
            float2 v[5];
            [unroll]
            for(int i = 0; i < 5; i++)
                v[i] = tex2Dlod(motion_tex, float4(uv + float2(MEDIAN_OFFS[i]) * texel_size, 0, mip)).xy;

            BB_S2(v[0], v[1])
            BB_S2(v[2], v[3])
            BB_S2(v[0], v[2])
            BB_S2(v[1], v[3])
            result = max(min(v[1], v[2]), min(max(v[1], v[2]), v[4]));
        }
        else if (FLOW_LEVEL != 0)
        {
            float2 v[9];
            int idx = 0;
            [unroll]
            for(int dy = -1; dy <= 1; dy++)
            {
                [unroll]
                for(int dx = -1; dx <= 1; dx++)
                {
                    v[idx] = tex2Dlod(motion_tex, float4(uv + float2(dx, dy) * texel_size, 0, mip)).xy;
                    idx++;
                }
            }

            BB_MNMX6(v[0], v[1], v[2], v[3], v[4], v[5])
            BB_MNMX5(v[1], v[2], v[3], v[4], v[6])
            BB_MNMX4(v[2], v[3], v[4], v[7])
            BB_MNMX3(v[3], v[4], v[8])
            result = v[4];
        }

        return result;
    }

    static const float REFINE_LUMA_SIGMA = 0.1;
    static const float REFINE_SPATIAL_SIGMA = 1.5;
    static const float REFINE_DISOCCLUSION_REL = 0.02;
    static const float REFINE_DISOCCLUSION_MIN = 1e-4;
    static const float REFINE_CONFIDENCE_FLOOR = 0.01;
    static const float REFINE_FARPLANE_DEPTH = 0.999;
    static const float FLOW_DEADZONE_MIN_PIXELS = 0.15;
    static const float FLOW_DEADZONE_MAX_PIXELS = 0.50;

    float2 ApplySubpixelDeadzone(float2 flow)
    {
        float magnitude_px = length(flow * bb::ScreenSize);
        return flow * smoothstep(FLOW_DEADZONE_MIN_PIXELS, FLOW_DEADZONE_MAX_PIXELS, magnitude_px);
    }

    void RefineFlowTap(sampler2D motion_tex, float2 uv, float2 texel_size, int mip,
                       int x, int y, float center_luma, float center_depth, float depth_tolerance,
                       float inv_spatial_sigma_sq, float inv_luma_sigma_sq,
                       inout float2 flow_sum, inout float weight_sum)
    {
        float2 sample_uv = uv + float2(x, y) * texel_size;

        float2 neighbor_flow = GetLod(motion_tex, sample_uv).xy;
        float neighbor_luma = tex2Dlod(sCurrLuma, float4(sample_uv, 0, mip)).r;
        float neighbor_depth = tex2Dlod(BaBa_Launcher::sLinearDepth, float4(saturate(sample_uv), 0, mip)).r;

        float spatial_dist_sq = dot(float2(x, y), float2(x, y));
        float spatial_weight = exp(spatial_dist_sq * inv_spatial_sigma_sq);
        float luma_diff = center_luma - neighbor_luma;
        float luma_weight = exp(luma_diff * luma_diff * inv_luma_sigma_sq);

        float abs_depth_diff = abs(center_depth - neighbor_depth);
        float disocclusion_gate = (abs_depth_diff < depth_tolerance) ? 1.0 : 0.0;
        bool center_far   = center_depth   >= REFINE_FARPLANE_DEPTH;
        bool neighbor_far = neighbor_depth >= REFINE_FARPLANE_DEPTH;
        if (center_far != neighbor_far)
            disocclusion_gate = 0.0;
        float neighbor_confidence = GetLod(sConfidence, sample_uv).r;
        neighbor_confidence = (neighbor_confidence == neighbor_confidence) ? saturate(neighbor_confidence) : 0.0;
        float confidence_weight = max(neighbor_confidence * neighbor_confidence, REFINE_CONFIDENCE_FLOOR);
        float total_weight = spatial_weight * luma_weight * disocclusion_gate * confidence_weight;
        flow_sum += neighbor_flow * total_weight;
        weight_sum += total_weight;
    }

    float2 RefineFlow(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
    {
        const float inv_spatial_sigma_sq = -0.5 / (REFINE_SPATIAL_SIGMA * REFINE_SPATIAL_SIGMA);
        const float inv_luma_sigma_sq = -0.5 / (REFINE_LUMA_SIGMA * REFINE_LUMA_SIGMA);

        float2 center_flow = GetLod(motion_tex, uv).xy;
        float center_luma = tex2Dlod(sCurrLuma, float4(uv, 0, mip)).r;
        float center_depth = tex2Dlod(BaBa_Launcher::sLinearDepth, float4(saturate(uv), 0, mip)).r;
        float depth_tolerance = REFINE_DISOCCLUSION_REL * center_depth + REFINE_DISOCCLUSION_MIN;

        float2 flow_sum = 0.0;
        float weight_sum = 0.0;

        if (FLOW_LEVEL == 1)
        {
            [unroll]
            for (int y = -1; y <= 1; ++y)
            {
                [unroll]
                for (int x = -1; x <= 1; ++x)
                    RefineFlowTap(motion_tex, uv, texel_size, mip, x, y, center_luma, center_depth,
                                  depth_tolerance, inv_spatial_sigma_sq, inv_luma_sigma_sq,
                                  flow_sum, weight_sum);
            }
        }
        else if (FLOW_LEVEL != 0)
        {
            [unroll]
            for (int y = -2; y <= 2; ++y)
            {
                [unroll]
                for (int x = -2; x <= 2; ++x)
                    RefineFlowTap(motion_tex, uv, texel_size, mip, x, y, center_luma, center_depth,
                                  depth_tolerance, inv_spatial_sigma_sq, inv_luma_sigma_sq,
                                  flow_sum, weight_sum);
            }
        }

        return (weight_sum > EPSILON) ?
            (flow_sum / weight_sum) : center_flow;
    }

    float2 ComputeFlow(sampler2D source_flow_sampler, float2 uv, int mip1, int mip2)
    {
        if(FRAME_COUNT == 0) return float2(0, 0);
        const int2 c8[8] = {
            int2(-1, 1), int2(0, 1), int2(1, 1),
            int2(-1, 0),             int2(1, 0),
            int2(-1,-1), int2(0,-1), int2(1,-1)
        };
        const int2 c8_it[9] = {
            int2(6, 3), int2(0, 3), int2(0, 5), int2(2, 5),
            int2(2, 7), int2(4, 7), int2(4, 1), int2(6, 1),
            int2(0, 0)
        };
        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip1));
        float2 source_texel_size = rcp(tex2Dsize(source_flow_sampler, 0));

        float predRef[9];
        ZAD_BuildRef(sCurrLuma, uv, texel_size, mip1, predRef);

        float2 center_candidate = GetLod(source_flow_sampler, uv).xy;
        float2 prediction = center_candidate;
        float min_cost = 1e6;

        ZAD_Consider(center_candidate, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(0, -source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(0,  source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv - float2(source_texel_size.x, 0)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(source_texel_size.x, 0)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(-source_texel_size.x, -source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2( source_texel_size.x, -source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(-source_texel_size.x,  source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(source_flow_sampler, uv + float2(source_texel_size.x, source_texel_size.y)).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(tex2Dfetch(sGlobalFlow, int2(0,0), 0).xy, uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(float2(0.0, 0.0), uv, texel_size, mip1, predRef, prediction, min_cost);
        ZAD_Consider(GetLod(sPrevFrameFlow, uv).xy, uv, texel_size, mip1, predRef, prediction, min_cost);

        texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip2));

        float searchRef[9];
        ZAD_BuildRef(sCurrLuma, uv, texel_size, mip2, searchRef);

        float2 residual = 0.0;
        float match_cost = ZAD_Cost(sPrevLuma, uv + prediction + residual, texel_size, mip2, searchRef);
        int match_i = 8;

        const int SEARCH_ITER_MAX = BABA_FLOW_SEARCH_ITER;
        int searchIter = (FLOW_LEVEL == 0) ? 1 : ((FLOW_LEVEL == 1) ? min(3, SEARCH_ITER_MAX) : SEARCH_ITER_MAX);

        if (match_cost >= FLOW_SEARCH_EARLY_OUT)
        {
            bool search_done = false;
            [loop]
            for (int search = 0; search < SEARCH_ITER_MAX; search++)
            {
                if (search < searchIter && !search_done)
                {
                    int i = c8_it[match_i].x;
                    int end = c8_it[match_i].y;
                    float2 search_center = residual;

                    float2 candidate_residual = search_center + float2(c8[i]) * texel_size;
                    float cost = ZAD_Cost(sPrevLuma, uv + prediction + candidate_residual, texel_size, mip2, searchRef);
                    if (cost < match_cost)
                    {
                        residual = candidate_residual;
                        match_i = i;
                        match_cost = cost;
                    }
                    i = ZAD_Ring8(i + 1);
                    [loop]
                    for(int k=0; k<8; k++)
                    {
                        if (i != end)
                        {
                            float2 cand_res = search_center + float2(c8[i]) * texel_size;
                            float c_cost = ZAD_Cost(sPrevLuma, uv + prediction + cand_res, texel_size, mip2, searchRef);
                            if (c_cost < match_cost)
                            {
                                residual = cand_res;
                                match_i = i;
                                match_cost = c_cost;
                            }
                            i = ZAD_Ring8(i + 1);
                        }
                    }

                    if (all(search_center == residual) || match_cost < FLOW_SEARCH_EARLY_OUT)
                        search_done = true;
                }
            }
        }

        float2 integer_match = prediction + residual;
        float cost_left   = ZAD_Cost(sPrevLuma, uv + integer_match - float2(texel_size.x, 0), texel_size, mip2, searchRef);
        float cost_right  = ZAD_Cost(sPrevLuma, uv + integer_match + float2(texel_size.x, 0), texel_size, mip2, searchRef);
        float cost_down   = ZAD_Cost(sPrevLuma, uv + integer_match - float2(0, texel_size.y), texel_size, mip2, searchRef);
        float cost_up     = ZAD_Cost(sPrevLuma, uv + integer_match + float2(0, texel_size.y), texel_size, mip2, searchRef);

        float2 curvature = 2.0 * float2(cost_left + cost_right, cost_down + cost_up) - 4.0 * match_cost;
        float2 gradient = float2(cost_left - cost_right, cost_down - cost_up);

        float min_curvature = max(match_cost * 0.05, EPSILON * 16.0);

        float2 subpixel_offset = 0.0;
        if (curvature.x > min_curvature)
            subpixel_offset.x = clamp(gradient.x / curvature.x, -0.5, 0.5);
        if (curvature.y > min_curvature)
            subpixel_offset.y = clamp(gradient.y / curvature.y, -0.5, 0.5);

        float2 refined = integer_match + subpixel_offset * texel_size;

        if (any(subpixel_offset != 0.0))
        {
            float refined_cost = ZAD_Cost(sPrevLuma, uv + refined, texel_size, mip2, searchRef);
            if (!(refined_cost < match_cost))
                refined = integer_match;
        }

        return refined;
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    float SampleWeightedLuma(float2 uv, float2 texel_size, const int2 offs[13], const float weights[13], const int sampleCount)
    {
        float luma_sum = 0.0;
        float weight_sum = 0.0;

        [loop]
        for(int i = 0; i < 13; i++)
        {
            if (i >= sampleCount) break;
            float2 sample_uv = uv + float2(offs[i]) * texel_size;
            float3 color = GetColor(sample_uv).rgb;
            float luma = GetLuminance(color);
            luma = luma * rcp(1.0 + luma);
            float weight = weights[i];
            luma_sum += luma * weight;
            weight_sum += weight;
        }

        return luma_sum / weight_sum;
    }

    float PS_CurrLuma(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 center_color = GetColor(uv).rgb;
        float center_luma = GetLuminance(center_color);
        center_luma = center_luma * rcp(1.0 + center_luma);

        if (FLOW_LEVEL == 0)
            return center_luma;

        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float smooth_luma;

        if (FLOW_LEVEL == 1)
        {
            const int2 offs[13] = { int2(0,0), int2(-1,0), int2(1,0), int2(0,-1), int2(0,1), 0,0,0,0,0,0,0,0 };
            const float weights[13] = { 6, 4, 4, 4, 4, 0,0,0,0,0,0,0,0 };
            smooth_luma = SampleWeightedLuma(uv, texel_size, offs, weights, 5);
        }
        else
        {
            const int2 offs[13] = { int2(0,-2), int2(-1,-1),int2(0,-1),int2(1,-1), int2(-2,0),int2(-1,0), int2(0,0), int2(1,0), int2(2,0), int2(-1,1), int2(0,1), int2(1,1), int2(0,2) };
            const float weights[13] = { 1, 3, 4, 3, 1, 4, 6, 4, 1, 3, 4, 3, 1 };
            smooth_luma = SampleWeightedLuma(uv, texel_size, offs, weights, 13);
        }

        return lerp(center_luma, smooth_luma, 0.9);
    }

    float2 PS_CoarseFlowL4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        static const int mip = 5;
        if (FRAME_COUNT == 0) return float2(0, 0);
        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip));
        
        float refWindow[9];
        ZAD_BuildRef(sCurrLuma, uv, texel_size, mip, refWindow);

        float2 global_prediction = tex2Dfetch(sGlobalFlow, int2(0,0), 0).xy;
        float2 local_prediction = GetLod(sPrevFrameFlow, uv).xy;
        float2 static_prediction = float2(0, 0);
        float global_cost = ZAD_Cost(sPrevLuma, uv + global_prediction, texel_size, mip, refWindow);
        float local_cost = ZAD_Cost(sPrevLuma, uv + local_prediction, texel_size, mip, refWindow);
        float static_cost = ZAD_Cost(sPrevLuma, uv + static_prediction, texel_size, mip, refWindow);

        float2 prediction = (static_cost < local_cost) ?
            ((static_cost < global_cost) ? static_prediction : global_prediction) :
            ((local_cost < global_cost) ? local_prediction : global_prediction);
        float2 best_flow = prediction;

        float min_cost = min(min(static_cost, local_cost), global_cost);
        if (FLOW_LEVEL > 0)
        {
            const int SEARCH_RADIUS_MAX = 3;
            int searchRadius = (FLOW_LEVEL == 1) ? 1 : SEARCH_RADIUS_MAX;
            bool search_done = false;
            [loop]
            for (int y = -SEARCH_RADIUS_MAX; y <= SEARCH_RADIUS_MAX; ++y)
            {
                if (abs(y) > searchRadius || search_done) continue;
                [loop]
                for (int x = -SEARCH_RADIUS_MAX; x <= SEARCH_RADIUS_MAX; ++x)
                {
                    if (abs(x) > searchRadius || search_done) continue;
                    if (x == 0 &&
                        y == 0) continue;
                    float2 candidate_flow = prediction + float2(x, y) * texel_size;
                    float cost = ZAD_Cost(sPrevLuma, uv + candidate_flow, texel_size, mip, refWindow);
                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        best_flow = candidate_flow;
                        if (min_cost < FLOW_SEARCH_EARLY_OUT) search_done = true;
                    }
                }
            }
        }
        return best_flow;
    }

    float2 PS_CoarseFlowL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL4, uv, 4, 4); }
    float2 PS_CoarseFlowL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL3_B, uv, 3, 3); }
    float2 PS_CoarseFlowL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL2_B, uv, 2, 2); }
    float2 PS_CoarseFlowL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL1_B, uv, 1, 1); }

    float2 PS_DenseFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (FRAME_COUNT == 0) return float2(0, 0);
        return ComputeFlow(sCoarseFlowL0_B, uv, 1, 0);
    }

    float2 PS_GlobalFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        static const float2 SPARSE_SCREEN[32] = {
            float2(0.0625, 0.125), float2(0.1875, 0.125), float2(0.3125, 0.125), float2(0.4375, 0.125),
            float2(0.5625, 0.125), float2(0.6875, 0.125), float2(0.8125, 0.125), float2(0.9375, 0.125),
            float2(0.0625, 0.375), float2(0.1875, 0.375), float2(0.3125, 0.375), float2(0.4375, 0.375),
       
            float2(0.5625, 0.375), float2(0.6875, 0.375), float2(0.8125, 0.375), float2(0.9375, 0.375),
            float2(0.0625, 0.625), float2(0.1875, 0.625), float2(0.3125, 0.625), float2(0.4375, 0.625),
            float2(0.5625, 0.625), float2(0.6875, 0.625), float2(0.8125, 0.625), float2(0.9375, 0.625),
            float2(0.0625, 0.875), float2(0.1875, 0.875), float2(0.3125, 0.875), float2(0.4375, 0.875),
            float2(0.5625, 0.875), float2(0.6875, 0.875), float2(0.8125, 0.875), float2(0.9375, 0.875)
        };
        float x_values[32], y_values[32];
        uint count = 0;

        [loop]
        for(int i = 0; i < 32; i++)
        {
            float depth = BaBa_Launcher_SampleDepth(SPARSE_SCREEN[i]);
            if(depth < 0.999)
            {
                float2 flow = GetLod(sCoarseFlowL0_B, SPARSE_SCREEN[i]).xy;
                x_values[count] = flow.x;
                y_values[count] = flow.y;
                count++;
            }
        }

        if(count < 3u) return 0.0;
        uint mid = count / 2u;
        [loop]
        for(int k = 0; k < 32; k++)
        {
            if(k > mid) break;
            [loop]
            for(int j = 0; j < 32; j++)
            {
                if(j >= count - 1 - k) break;
                if(x_values[j] > x_values[j+1])
                {
                    float tmp = x_values[j];
                    x_values[j] = x_values[j+1];
                    x_values[j+1] = tmp;
                }
                if(y_values[j] > y_values[j+1])
                {
                    float tmp = y_values[j];
                    y_values[j] = y_values[j+1];
                    y_values[j+1] = tmp;
                }
            }
        }
        return float2(x_values[mid], y_values[mid]);
    }

    void PS_CopyFlowHistory(float4 pos : SV_Position, float2 uv : TEXCOORD,
                            out float2 flow : SV_Target0, out float confidence : SV_Target1)
    {
        float2 f = GetLod(sFinalFlow, uv).xy;
        flow = IsInvalidFlow(f) ? 0.0 : f;

        float c = GetLod(sConfidence, uv).r;
        confidence = (c == c) ? saturate(c) : 0.0;
    }

    float PS_CopyHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return GetLod(sCurrLuma, uv).r;
    }

#if BABA_LAUNCHER_DEBUG
    float4 PS_Debug(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (DEBUG_VIEW == 1)
            return float4(MotionToColor(GetLod(BaBa_Launcher::sMotionVectors, uv).xy), 1);

        if (DEBUG_VIEW == 2)
        {
            static const float ARROW_THICKNESS  = 1.2; // shaft half-width, in pixels
            static const float ARROW_OUTLINE     = 1.0; // extra half-width added by the outline
            static const float CELL_SIZE         = 28.0; // spacing between arrows, in pixels
            static const float ARROWHEAD_LENGTH  = 6.0;
            static const float ARROWHEAD_WIDTH   = 3.0; // arrowhead half-width at its base
            static const float AA_WIDTH          = 1.4; // antialiasing softness, in pixels
            static const float MAX_ARROW_LENGTH  = CELL_SIZE - 6.0; // keep arrows inside their own cell

            float3 base_color = GetColor(uv).rgb;
            float2 screen_size = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
            float2 pixel_pos = uv * screen_size;
            // Grid spaced in absolute pixels so arrows stay legibly apart.
            float2 grid_pixel_pos = (floor(pixel_pos / CELL_SIZE) + 0.5) * CELL_SIZE;
            float2 motion_grid = grid_pixel_pos * bb::PixelSize;

            float2 motion = GetLod(BaBa_Launcher::sMotionVectors, motion_grid).xy;
            float2 motion_pixels = motion * screen_size;
            float motion_magnitude = length(motion_pixels);

            if (motion_magnitude < 0.5)
                return float4(base_color, 1.0);

            float arrow_length = clamp(motion_magnitude * 3.0, 8.0, MAX_ARROW_LENGTH);
            float2 arrow_dir  = NM_SafeNormalize(float3(-motion_pixels, 0.0)).xy;
            float2 arrow_perp = float2(-arrow_dir.y, arrow_dir.x);

            // The shaft stops short of the tip so it fuses into one silhouette with the head.
            float shaft_length = max(arrow_length - ARROWHEAD_LENGTH * 0.65, 0.0);
            float2 shaft_start = grid_pixel_pos;
            float2 shaft_end   = grid_pixel_pos + arrow_dir * shaft_length;
            float2 arrow_tip   = grid_pixel_pos + arrow_dir * arrow_length;
            float2 head_base_l = shaft_end + arrow_perp * ARROWHEAD_WIDTH;
            float2 head_base_r = shaft_end - arrow_perp * ARROWHEAD_WIDTH;

            float dist_shaft = sdSegment(pixel_pos, shaft_start, shaft_end) - ARROW_THICKNESS;
            float dist_head  = sdTriangle(pixel_pos, arrow_tip, head_base_l, head_base_r);
            float dist_arrow = min(dist_shaft, dist_head);

            float outline_mask = ArrowCoverage(dist_arrow - ARROW_OUTLINE, AA_WIDTH);
            float fill_mask     = ArrowCoverage(dist_arrow, AA_WIDTH);

            float head_t = saturate(dot(pixel_pos - grid_pixel_pos, arrow_dir) / max(arrow_length, 1e-4));
            float3 arrow_color = MotionToColor(motion) * lerp(0.8, 1.2, head_t);

            float3 result = lerp(base_color, float3(0.02, 0.02, 0.02), outline_mask);
            result = lerp(result, arrow_color, fill_mask);
            return float4(result, 1.0);
        }

        if (DEBUG_VIEW == 3)
        {
        float confidence = GetLod(BaBa_Launcher::sMotionConfidence, uv).x;
            float3 confidenceColor;
            if (confidence < 0.5)
                confidenceColor = lerp(float3(1.0, 0.0, 0.0), float3(1.0, 1.0, 0.0), confidence * 2.0);
            else
                confidenceColor = lerp(float3(1.0, 1.0, 0.0), float3(0.0, 1.0, 0.0), (confidence - 0.5) * 2.0);
            return float4(confidenceColor, 1.0);
        }

        if (DEBUG_VIEW == 4)
        {
            float depth = GetLod(BaBa_Launcher::sLinearDepth, uv).r;
            float visibleDepth = pow(saturate(depth), 0.25);
            return float4(visibleDepth.xxx, 1.0);
        }

        if (DEBUG_VIEW == 5)
        {
            float3 normal = BaBa_Launcher_SampleNormal(uv);
            return float4(normal * 0.5 + 0.5, 1.0);
        }

        if (DEBUG_VIEW == 6)
        {
            float viewDepth = GetLod(BaBa_Launcher::sViewDepth, uv).r;
            float visibleDepth = pow(saturate(viewDepth / FAR_PLANE), 0.25);
            return float4(visibleDepth.xxx, 1.0);
        }

        if (DEBUG_VIEW == 7)
        {
            float3 normal = NM_DecodeNormal(tex2Dlod(
                BaBa_Launcher::sNormalEncoded,
                float4(clamp(uv, 0.0, 1.0), 0.0, 0.0)).rg);
            return float4(normal * 0.5 + 0.5, 1.0);
        }

        return GetColor(uv);
    }
#endif // BABA_LAUNCHER_DEBUG

    float2 PS_SpatialFilterL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median(sCoarseFlowL3_A, uv, rcp(tex2Dsize(sCoarseFlowL3_A, 0)), 6); }
    float2 PS_SpatialFilterL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median(sCoarseFlowL2_A, uv, rcp(tex2Dsize(sCoarseFlowL2_A, 0)), 5); }
    float2 PS_SpatialFilterL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median(sCoarseFlowL1_A, uv, rcp(tex2Dsize(sCoarseFlowL1_A, 0)), 4); }
    float2 PS_SpatialFilterL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median(sCoarseFlowL0_A, uv, rcp(tex2Dsize(sCoarseFlowL0_A, 0)), 3); }
    float2 PS_MedianDenseFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median(sDenseFlow_A, uv, rcp(tex2Dsize(sDenseFlow_A, 0)), 2); }
    float2 PS_SmoothFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return ApplySubpixelDeadzone(RefineFlow(sDenseFlow_M, uv, rcp(tex2Dsize(sDenseFlow_M, 0)), 2));
    }

    // Confidence is accumulated asymmetrically: a genuine drop is followed fast
    // and a rise slowly. The deadband keeps ordinary estimate noise on the slow path.
    static const float CONFIDENCE_ALPHA_RISE = 0.15;  // ~10 frames to settle
    static const float CONFIDENCE_ALPHA_FALL = 0.60;  // ~2 frames to settle
    static const float CONFIDENCE_FALL_DEADBAND = 0.08;

    float PS_Confidence(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (FRAME_COUNT == 0) return 0.0;

        // Measured on the pre-refinement flow, so the refinement can weight by it.
        float2 flow = GetLod(sDenseFlow_M, uv).xy;

        // NaN comparisons do not trip IsOOB, so validate the flow explicitly.
        if (IsInvalidFlow(flow)) return 0.0;

        float2 prev_uv = uv + flow;

        // A reprojection that leaves the screen has no history by definition.
        if (IsOOB(prev_uv)) return 0.0;
        float curr_luma = tex2Dlod(sCurrLuma, float4(uv, 0, 2)).r;
        float prev_luma = tex2Dlod(sPrevLuma, float4(prev_uv, 0, 2)).r;
        float luma_error = abs(curr_luma - prev_luma);

        // Fade the range out instead of a hard cutoff, which flickered on the threshold.
        float photometric_gate = 1.0 - smoothstep(0.10, 0.22, luma_error);
        if (photometric_gate <= 0.0)
            return 0.0;

        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float subpixel_threshold = length(texel_size);
        float flow_magnitude = length(flow);

        float current_confidence;
        if (flow_magnitude <= subpixel_threshold)
        {
            current_confidence = photometric_gate;
        }
        else
        {
            float2 destination_flow = GetLod(sDenseFlow_M, prev_uv).xy;
            if (IsInvalidFlow(destination_flow)) return 0.0;
            float2 flow_difference = flow - destination_flow;
            float error = length(flow_difference);
            float normalized_error = error / flow_magnitude;
            float motion_penalty = flow_magnitude / subpixel_threshold;

            float length_confidence = rcp(motion_penalty * 0.05 + 1.0);
            float consistency_confidence = rcp(normalized_error + 1.0);
            float photometric_confidence = exp(-luma_error * 5.0) * photometric_gate;

            current_confidence = consistency_confidence * length_confidence * photometric_confidence;
        }

        current_confidence = saturate(current_confidence);

        // Nothing to accumulate onto yet.
        if (FRAME_COUNT <= 1) return current_confidence;

        float history = GetLod(sPrevConfidence, prev_uv).r;
        if (history != history) return current_confidence;
        history = saturate(history);

        float alpha = (current_confidence < history - CONFIDENCE_FALL_DEADBAND) ?
            CONFIDENCE_ALPHA_FALL : CONFIDENCE_ALPHA_RISE;
        return saturate(lerp(history, current_confidence, alpha));
    }

    void PS_ExportFlow(float4 pos : SV_Position, float2 uv : TEXCOORD, out float2 flow : SV_Target0, out float confidence : SV_Target1)
    {
        flow = GetLod(sFinalFlow, uv).xy;
        confidence = GetLod(sConfidence, uv).x;

        if (IsInvalidFlow(flow))
            flow = 0.0;
        if (confidence != confidence)
            confidence = 0.0;
        confidence = saturate(confidence);
    }

    struct PrepassVS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };

    void VS_Prepass(in uint id : SV_VertexID, out PrepassVS_OUTPUT output)
    {
        output.uv.x = (id == 2) ? 2.0 : 0.0;
        output.uv.y = (id == 1) ? 2.0 : 0.0;
        output.vpos = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        float y = tan(RECONSTRUCTION_FOV * DEG2RAD * 0.5);
        output.pScale = float2(y * bb::AspectRatio, y);
    }

    void PS_Depth(float4 pos : SV_Position, float2 uv : TEXCOORD,
                          out float4 outLinear : SV_Target0, out float4 outView : SV_Target1)
    {
        float depth = NM_SafeDepth(bb::GetLinearizedDepth(uv));
        outLinear = float4(depth, 0.0, 0.0, 1.0);
        outView = float4(depth * FAR_PLANE, 0.0, 0.0, 1.0);
    }

    void PS_Normal(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float depth = NM_SafeDepth(bb::GetLinearizedDepth(input.uv));
        float3 normal = (depth >= 0.999) ? float3(0.0, 0.0, -1.0) : NM_CalculateNormal(input.uv, input.pScale);
        outNormal = float4(NM_SafeNormalize(normal), depth);
    }

    void PS_NormalSmoothH(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 raw = tex2Dlod(BaBa_Launcher::sNormalRaw, float4(input.uv, 0.0, 0.0));
        if (NORMAL_SMOOTH_LEVEL < 1)
        {
            outNormal = raw;
            return;
        }

        outNormal = NM_ATrousSmoothNormal(
            input.uv, float2(1.0, 0.0), BaBa_Launcher::sNormalRaw,
            4.0, NORMAL_SMOOTH_THRESHOLD, FAR_PLANE);
    }

    void PS_NormalSmoothV(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 normalData = tex2Dlod(BaBa_Launcher::sNormalSmoothH, float4(input.uv, 0.0, 0.0));
        if (NORMAL_SMOOTH_LEVEL < 1)
        {
            outNormal = NM_SanitizeNormalData(normalData);
            return;
        }

        outNormal = NM_ATrousSmoothNormal(
            input.uv, float2(0.0, 1.0), BaBa_Launcher::sNormalSmoothH,
            4.0, NORMAL_SMOOTH_THRESHOLD, FAR_PLANE);
    }

    void PS_NormalAtrousH2(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 stage1 = tex2Dlod(BaBa_Launcher::sNormalSmoothV, float4(input.uv, 0.0, 0.0));
        if (NORMAL_SMOOTH_LEVEL < 2)
        {
            outNormal = NM_SanitizeNormalData(stage1);
            return;
        }

        outNormal = NM_ATrousSmoothNormal(
            input.uv, float2(1.0, 0.0), BaBa_Launcher::sNormalSmoothV,
            8.0, NORMAL_SMOOTH_THRESHOLD, FAR_PLANE);
    }

    void PS_NormalAtrousV2(PrepassVS_OUTPUT input,
                            out float4 outNormal : SV_Target0,
                            out float4 outEncoded : SV_Target1,
                            out float4 outGeometric : SV_Target2)
    {
        float4 normalData = tex2Dlod(BaBa_Launcher::sNormalSmoothH, float4(input.uv, 0.0, 0.0));
        if (NORMAL_SMOOTH_LEVEL >= 2)
        {
            normalData = NM_ATrousSmoothNormal(
                input.uv, float2(0.0, 1.0), BaBa_Launcher::sNormalSmoothH,
                8.0, NORMAL_SMOOTH_THRESHOLD, FAR_PLANE);
        }

        normalData = NM_SanitizeNormalData(normalData);
        float3 normal = NM_SafeNormalize(normalData.rgb);
        float depth = NM_SafeDepth(GetDepth(input.uv));
        outGeometric = float4(normal, depth);

        if (TEXTURE_NORMAL_INTENSITY > 0.0 && depth < 0.999)
        {
            normal = NM_ApplyTextureBump(normal, input.uv, bb::BackBuffer, bb::PixelSize,
                TEXTURE_NORMAL_INTENSITY, TEXTURE_NORMAL_EDGE_THRESHOLD);
        }

        outNormal = float4(normal, depth);
        outEncoded = float4(NM_EncodeNormal(normal), 0.0, depth);
    }

    float4 ComputeTemporalNormal(PrepassVS_OUTPUT input, sampler2D sPreviousNormal)
    {
        float4 currentData = NM_SanitizeNormalData(tex2Dlod(BaBa_Launcher::sNormalMap,
            float4(input.uv, 0.0, 0.0)));

        if (!NORMAL_TEMPORAL || FRAME_COUNT <= 1)
            return currentData;

        float2 velocity = GetLod(sFinalFlow, input.uv).xy;
        if (IsInvalidFlow(velocity))
            return currentData;
        float2 previousUV = input.uv + velocity;
        if (any(previousUV < 0.0) || any(previousUV > 1.0))
            return currentData;

        float4 previousData = NM_SanitizeNormalData(tex2Dlod(sPreviousNormal,
            float4(previousUV, 0.0, 0.0)));
        float3 currentNormal = NM_SafeNormalize(currentData.rgb);
        float3 previousNormal = NM_SafeNormalize(previousData.rgb);

        float2 px = bb::PixelSize;
        float3 nLeft = NM_SafeNormalize(tex2Dlod(BaBa_Launcher::sNormalMap, float4(input.uv + float2(-px.x, 0.0), 0.0, 0.0)).rgb);
        float3 nRight = NM_SafeNormalize(tex2Dlod(BaBa_Launcher::sNormalMap, float4(input.uv + float2(px.x, 0.0), 0.0, 0.0)).rgb);
        float3 nUp = NM_SafeNormalize(tex2Dlod(BaBa_Launcher::sNormalMap, float4(input.uv + float2(0.0, -px.y), 0.0, 0.0)).rgb);
        float3 nDown = NM_SafeNormalize(tex2Dlod(BaBa_Launcher::sNormalMap, float4(input.uv + float2(0.0, px.y), 0.0, 0.0)).rgb);
        float localAgreement = min(min(dot(currentNormal, nLeft), dot(currentNormal, nRight)),
                                    min(dot(currentNormal, nUp), dot(currentNormal, nDown)));
        float localChaos = 1.0 - smoothstep(0.0, 0.6, saturate(localAgreement));

        float normalAgreement = saturate(dot(currentNormal, previousNormal));
        float normalConfidence = lerp(smoothstep(0.25, 0.90, normalAgreement), 1.0, localChaos);

        float depthTolerance = lerp(0.004 + 0.04 * max(currentData.a, previousData.a), 1.0, localChaos);
        float depthConfidence = exp(-abs(currentData.a - previousData.a) /
            max(depthTolerance, 1e-4));
        float flowConfidenceSample = GetLod(sConfidence, input.uv).r;
        float flowConfidence = (flowConfidenceSample == flowConfidenceSample) ?
            saturate(flowConfidenceSample) : 0.0;
        float feedback = NORMAL_TEMPORAL_STABILITY *
            flowConfidence * normalConfidence * depthConfidence;

        float3 stableNormal = NM_SafeNormalize(lerp(currentNormal, previousNormal, saturate(feedback)));
        float stableDepth = NM_SafeDepth(currentData.a);
        return float4(stableNormal, stableDepth);
    }

    void PS_NormalTemporal0(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        outNormal = 0.0;
        if ((uint(FRAME_COUNT) % 2u) != 0u)
            discard;
        outNormal = ComputeTemporalNormal(input, BaBa_Launcher::sNormalHistory1);
    }

    void PS_NormalTemporal1(PrepassVS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        outNormal = 0.0;
        if ((uint(FRAME_COUNT) % 2u) == 0u)
            discard;
        outNormal = ComputeTemporalNormal(input, BaBa_Launcher::sNormalHistory0);
    }

    void PS_PublishNormal(PrepassVS_OUTPUT input,
                                  out float4 outNormal : SV_Target0,
                                  out float4 outEncoded : SV_Target1)
    {
        float4 stableData = ((uint(FRAME_COUNT) % 2u) == 0u) ?
            tex2Dlod(BaBa_Launcher::sNormalHistory0, float4(input.uv, 0.0, 0.0)) :
            tex2Dlod(BaBa_Launcher::sNormalHistory1, float4(input.uv, 0.0, 0.0));

        stableData = NM_SanitizeNormalData(stableData);
        float3 normal = NM_SafeNormalize(stableData.rgb);
        outNormal = float4(normal, stableData.a);
        outEncoded = float4(NM_EncodeNormal(normal), 0.0, stableData.a);
    }

    technique BaBa_Launcher <
        ui_label = "BaBa: Launcher";
        ui_tooltip = "Shared depth, smoothed normals and dense optical flow prepass.";
    >
    {
        //=== Shared depth and normal prepass
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_Depth;
        RenderTarget0 = BaBa_Launcher::LinearDepth;
        RenderTarget1 = BaBa_Launcher::ViewDepth;
        GenerateMipMaps = true; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_Normal; RenderTarget = BaBa_Launcher::NormalRaw; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalSmoothH; RenderTarget = BaBa_Launcher::NormalSmoothH; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalSmoothV; RenderTarget = BaBa_Launcher::NormalSmoothV; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalAtrousH2; RenderTarget = BaBa_Launcher::NormalSmoothH; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalAtrousV2;
        RenderTarget0 = BaBa_Launcher::NormalMap;
        RenderTarget1 = BaBa_Launcher::NormalEncoded;
        RenderTarget2 = BaBa_Launcher::NormalSmoothV;
        GenerateMipMaps = false; }

        //=== Luma pyramid
        // The coarse-to-fine search reads mips 0..5, so the mip chain is required.
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CurrLuma; RenderTarget = tCurrLuma; GenerateMipMaps = true; }

        //=== Optical Flow
        // Coarse Flow Level 4
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CoarseFlowL4; RenderTarget = tCoarseFlowL4; }

        // Coarse Flow Level 3
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CoarseFlowL3; RenderTarget = tCoarseFlowL3_A; }
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_SpatialFilterL3; RenderTarget = tCoarseFlowL3_B; }

        // Coarse Flow Level 2
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CoarseFlowL2; RenderTarget = tCoarseFlowL2_A; }
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_SpatialFilterL2; RenderTarget = tCoarseFlowL2_B; }

        // Coarse Flow Level 1
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CoarseFlowL1; RenderTarget = tCoarseFlowL1_A; }
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_SpatialFilterL1; RenderTarget = tCoarseFlowL1_B; }

        // Coarse Flow Level 0
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CoarseFlowL0; RenderTarget = tCoarseFlowL0_A; }
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_SpatialFilterL0; RenderTarget = tCoarseFlowL0_B; }

        // Dense Flow and Features
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_DenseFlow; RenderTarget = tDenseFlow_A; }
        // Outlier rejection before the bilateral: a median removes a bad vector.
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_MedianDenseFlow; RenderTarget = tDenseFlow_M; }

        // === Confidence Map for the Flow field
        // Runs before the refinement so it can weight by the result.
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_Confidence; RenderTarget = tConfidence; }

        pass { VertexShader = PostProcessVS;
        PixelShader = PS_SmoothFlow; RenderTarget = tDenseFlow_B; }

        // === Global Flow
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_GlobalFlow; RenderTarget = tGlobalFlow; }

        //=== Export the Flow
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_ExportFlow; RenderTarget0 = BaBa_Launcher::MotionVectors; RenderTarget1 = BaBa_Launcher::MotionConfidence; }

        //=== Stabilize normals after motion is available.
        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalTemporal0; RenderTarget = BaBa_Launcher::NormalHistory0; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_NormalTemporal1; RenderTarget = BaBa_Launcher::NormalHistory1; }

        pass { VertexShader = VS_Prepass;
        PixelShader = PS_PublishNormal;
        RenderTarget0 = BaBa_Launcher::NormalMap;
        RenderTarget1 = BaBa_Launcher::NormalEncoded;
        GenerateMipMaps = true; }

        //=== Debug pass 
#if BABA_LAUNCHER_DEBUG
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_Debug; }
#endif

        //=== Save history
        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CopyFlowHistory;
        RenderTarget0 = tPrevFlow;
        RenderTarget1 = tPrevConfidence; }

        pass { VertexShader = PostProcessVS;
        PixelShader = PS_CopyHistory; RenderTarget = tPrevLuma; GenerateMipMaps = true; }
    }
}
