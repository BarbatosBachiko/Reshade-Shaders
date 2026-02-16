/*----------------------------------------------|
| :: BarbatosFlow                            :: |
|-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| Original Author (LumaFlow): Afzaal (Kaidō)    |
| License: CC BY-NC 4.0                         |
|-----------------------------------------------|
| Description:                                  |
| BarbatosFlow is a Dense Real-time Optical     |
| Flow shader derived from LumaFlow.            |
|                                               |
| Original Repository:                          |
| GitHub : https://github.com/umar-afzaal/LumeniteFX
|                                               |
| Original Copyright (c) Afzaal.                |
| All rights reserved.                          |
'----------------------------------------------*/

#include "ReShade.fxh"

#define EPSILON 1e-6

uniform int FRAME_COUNT < source = "framecount"; >;

uniform int QUALITY_PRESET <
    ui_type = "combo";
    ui_items = "Medium\0High\0";
    ui_label = "Quality Preset";
> = 0;

uniform int DEBUG_VIEW <
    ui_type = "combo";
    ui_items = "Debug Off\0"
               "Optical Flow\0"
               "Motion Vectors\0"
               "Confidence Map\0";
    ui_label = "Debug View";
> = 0;

//----------------|
// :: Textures :: |
//----------------|

//=== Export Flow - use these in your shaders to access the final flow field
texture2D texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler2D sTexMotionVectorsSampler
{
    Texture = texMotionVectors;
    MagFilter = POINT;
    MinFilter = POINT;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
};

texture2D tMotionConfidence
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler2D sMotionConfidence
{
    Texture = tMotionConfidence;
    MagFilter = POINT;
    MinFilter = POINT;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
};

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
    sampler2D sCoarseFlowL4 { Texture = tCoarseFlowL4; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL3_A { Width = BUFFER_WIDTH / 64; Height = BUFFER_HEIGHT / 64; Format = RG16F; };
    sampler2D sCoarseFlowL3_A { Texture = tCoarseFlowL3_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL3_B { Width = BUFFER_WIDTH / 64; Height = BUFFER_HEIGHT / 64; Format = RG16F; };
    sampler2D sCoarseFlowL3_B { Texture = tCoarseFlowL3_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL2_A { Width = BUFFER_WIDTH / 32; Height = BUFFER_HEIGHT / 32; Format = RG16F; };
    sampler2D sCoarseFlowL2_A { Texture = tCoarseFlowL2_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL2_B { Width = BUFFER_WIDTH / 32; Height = BUFFER_HEIGHT / 32; Format = RG16F; };
    sampler2D sCoarseFlowL2_B { Texture = tCoarseFlowL2_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL1_A { Width = BUFFER_WIDTH / 16; Height = BUFFER_HEIGHT / 16; Format = RG16F; };
    sampler2D sCoarseFlowL1_A { Texture = tCoarseFlowL1_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL1_B { Width = BUFFER_WIDTH / 16; Height = BUFFER_HEIGHT / 16; Format = RG16F; };
    sampler2D sCoarseFlowL1_B { Texture = tCoarseFlowL1_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL0_A { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F; };
    sampler2D sCoarseFlowL0_A { Texture = tCoarseFlowL0_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tCoarseFlowL0_B { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F; };
    sampler2D sCoarseFlowL0_B { Texture = tCoarseFlowL0_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tDenseFlow_A { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F; };
    sampler2D sDenseFlow_A { Texture = tDenseFlow_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tDenseFlow_B { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F; };
    sampler2D sFinalFlow { Texture = tDenseFlow_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    texture2D tGlobalFlow { Width = 1; Height = 1; Format = RG16F; };
    sampler2D sGlobalFlow { Texture = tGlobalFlow; };

    texture2D tPrevFlow { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F; };
    sampler2D sPrevFrameFlow { Texture = tPrevFlow; MagFilter = POINT; MinFilter = POINT; };

    texture2D tPrevBackBuffer { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
    sampler2D sPrevBackBuffer { Texture = tPrevBackBuffer; MagFilter = LINEAR; MinFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP; };

    texture2D tConfidence { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = R16F; };
    sampler2D sConfidence { Texture = tConfidence; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

    //-----------------|
    // :: Functions :: |
    //-----------------|

    float3 MotionToColor(float2 motion)
    {
        float angle = atan2(-motion.y, -motion.x) / 6.283 + 0.5;
        float raw_length = length(motion) / (15.0 * ReShade::PixelSize.x);
        float compressed = raw_length / (1.0 + raw_length * 1.4);
        float boosted = pow(compressed, 0.5);
        float magnitude = saturate(lerp(compressed, boosted, saturate(raw_length * 3.0)));
        float3 hsv = float3(angle, 1, magnitude);
        float4 K = float4(1, 2.0 / 3.0, 1.0 / 3.0, 3);
        float3 p = abs(frac(hsv.xxx + K.xyz) * 6 - K.www);
        return hsv.z * lerp(K.xxx, clamp(p - K.xxx, 0, 1), hsv.y) + 0.1;
    }

    float GetDepth(float2 xy) { return ReShade::GetLinearizedDepth(xy); }
    bool IsOOB(float2 uv) { return any(uv < 0.0) || any(uv > 1.0); }
    float3 GetColor(float2 uv) { return tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb; }

    // Helper for Median
    void MnMx(inout float2 a, inout float2 b)
    {
        float2 mn = min(a, b);
        float2 mx = max(a, b);
        a = mn;
        b = mx;
    }

    float ZAD(sampler2D cur, sampler2D prev, float2 pos_a, float2 pos_b, float2 texel_size, int mip)
    {
        // MEDIUM 
        if (QUALITY_PRESET == 0)
        {
            static const int2 DIAMOND[5] = { int2(0,0), int2(0,-1), int2(-1,0), int2(1,0), int2(0,1) };
            float err = 0.0;
            [unroll]
            for (int i = 0; i < 5; i++)
            {
                float2 offset = float2(DIAMOND[i]) * texel_size;
                float val_a = tex2Dlod(cur, float4(pos_a + offset, 0, mip)).r;
                float val_b = tex2Dlod(prev, float4(pos_b + offset, 0, mip)).r;
                err += abs(val_a - val_b);
            }
            return ((err * 0.2) + EPSILON);
        }
        // HIGH 
        else
        {
            static const int2 DIAMOND_FULL[9] = {
                int2(0,-3), int2(0,-1), int2(-3,0), int2(-1,0), int2(0,0),
                int2(1,0), int2(3,0), int2(0,1), int2(0,3)
            };
            float samples_a[9], samples_b[9];
            float mean_a = 0.0, mean_b = 0.0;

            [unroll]
            for (int i = 0; i < 9; i++)
            {
                float2 offset = float2(DIAMOND_FULL[i]) * texel_size;
                samples_a[i] = tex2Dlod(cur, float4(pos_a + offset, 0, mip)).r;
                samples_b[i] = tex2Dlod(prev, float4(pos_b + offset, 0, mip)).r;
                mean_a += samples_a[i];
                mean_b += samples_b[i];
            }
            mean_a /= 9.0;
            mean_b /= 9.0;

            float err = 0.0;
            [unroll]
            for (int i = 0; i < 9; i++)
            {
                err += abs((samples_a[i] - mean_a) - (samples_b[i] - mean_b));
            }
            return ((err / 9.0) + EPSILON);
        }
    }

    float2 Median(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
    {
        // MEDIUM - 5 Tap Network
        if (QUALITY_PRESET == 0)
        {
            float2 v[5];
            v[0] = tex2Dlod(motion_tex, float4(uv, 0, mip)).xy;
            v[1] = tex2Dlod(motion_tex, float4(uv + float2(texel_size.x, 0), 0, mip)).xy;
            v[2] = tex2Dlod(motion_tex, float4(uv + float2(-texel_size.x, 0), 0, mip)).xy;
            v[3] = tex2Dlod(motion_tex, float4(uv + float2(0, texel_size.y), 0, mip)).xy;
            v[4] = tex2Dlod(motion_tex, float4(uv + float2(0, -texel_size.y), 0, mip)).xy;

            MnMx(v[0], v[1]); MnMx(v[1], v[2]); MnMx(v[2], v[3]); MnMx(v[3], v[4]);
            MnMx(v[0], v[1]); MnMx(v[1], v[2]); MnMx(v[2], v[3]);
            MnMx(v[0], v[1]); MnMx(v[1], v[2]);
            return v[2];
        }
        // HIGH - 9 Tap Bubble Sort
        else
        {
            float x_values[9], y_values[9];
            int idx = 0;
            [loop]
            for (int dy = -1; dy <= 1; dy++)
            {
                [loop]
                for (int dx = -1; dx <= 1; dx++)
                {
                    float2 vec = tex2Dlod(motion_tex, float4(uv + float2(dx, dy) * texel_size, 0, mip)).xy;
                    x_values[idx] = vec.x; y_values[idx] = vec.y;
                    idx++;
                }
            }
            [loop]
            for (int k = 0; k < 5; k++)
            {
                [loop]
                for (int i = 0; i < 8; i++)
                {
                    if (i >= 8 - k) break;
                    if (x_values[i] > x_values[i + 1]) { float tmp = x_values[i]; x_values[i] = x_values[i + 1]; x_values[i + 1] = tmp; }
                    if (y_values[i] > y_values[i + 1]) { float tmp = y_values[i]; y_values[i] = y_values[i + 1]; y_values[i + 1] = tmp; }
                }
            }
            return float2(x_values[4], y_values[4]);
        }
    }

    float2 RefineFlow(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
    {
        // MEDIUM & HIGH Share this logic structure, but parameters differ implicitly via loop count/cost
        if (QUALITY_PRESET == 0)
        {
            float2 center_flow = tex2Dlod(motion_tex, float4(uv, 0, 0)).xy;
            float center_depth = GetDepth(uv);
            float2 sum_flow = center_flow;
            float sum_weight = 1.0;
            static const int2 OFFSETS[4] = { int2(0,-1), int2(-1,0), int2(1,0), int2(0,1) };

            [unroll]
            for (int i = 0; i < 4; i++)
            {
                float2 off = float2(OFFSETS[i]) * texel_size;
                float2 sample_uv = uv + off;
                float2 neighbor_flow = tex2Dlod(motion_tex, float4(sample_uv, 0, 0)).xy;
                float neighbor_depth = GetDepth(sample_uv);
                float depth_diff = abs(center_depth - neighbor_depth);
                float w = (depth_diff < 0.01) ? 1.0 : 0.0;
                sum_flow += neighbor_flow * w;
                sum_weight += w;
            }
            return sum_flow / (sum_weight + EPSILON);
        }
        else
        {
            // Full Spatial+Luma 
            #define LUMA_SIGMA 0.1
            #define SPATIAL_SIGMA 1.5
            #define DISOCCLUSION_THRESHOLD 0.01
            const float inv_spatial_sigma_sq = -0.5 / (SPATIAL_SIGMA * SPATIAL_SIGMA);
            const float inv_luma_sigma_sq = -0.5 / (LUMA_SIGMA * LUMA_SIGMA);

            float2 center_flow = tex2Dlod(motion_tex, float4(uv, 0, 0)).xy;
            float center_luma = tex2Dlod(sCurrLuma, float4(uv, 0, mip)).r;
            float center_depth = GetDepth(uv);
            float2 flow_sum = 0.0;
            float weight_sum = 0.0;

            [loop]
            for (int y = -2; y <= 2; ++y)
            {
                [loop]
                for (int x = -2; x <= 2; ++x)
                {
                    float2 offset = float2(x, y) * texel_size;
                    float2 sample_uv = uv + offset;
                    float2 neighbor_flow = tex2Dlod(motion_tex, float4(sample_uv, 0, 0)).xy;
                    float neighbor_luma = tex2Dlod(sCurrLuma, float4(sample_uv, 0, mip)).r;
                    float neighbor_depth = GetDepth(sample_uv);
                    float spatial_dist_sq = dot(float2(x, y), float2(x, y));
                    float spatial_weight = exp(spatial_dist_sq * inv_spatial_sigma_sq);
                    float luma_diff = center_luma - neighbor_luma;
                    float luma_weight = exp(luma_diff * luma_diff * inv_luma_sigma_sq);
                    float abs_depth_diff = abs(center_depth - neighbor_depth);
                    float disocclusion_gate = (abs_depth_diff < DISOCCLUSION_THRESHOLD) ? 1.0 : 0.0;
                    float total_weight = spatial_weight * luma_weight * disocclusion_gate;
                    flow_sum += neighbor_flow * total_weight;
                    weight_sum += total_weight;
                }
            }
            return (weight_sum > EPSILON) ? (flow_sum / weight_sum) : center_flow;
        }
    }

    float2 ComputeFlow(sampler2D source_flow_sampler, float2 uv, int mip1, int mip2)
    {
        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip1));
        float2 source_texel_size = rcp(tex2Dsize(source_flow_sampler, 0));
        float min_cost = 1e6;
        float2 prediction = 0.0;

        //=== (1) Get best prediction
        if (QUALITY_PRESET == 0) // MEDIUM
        {
            float2 candidates[6];
            candidates[0] = tex2D(source_flow_sampler, uv).xy;
            candidates[1] = tex2D(sPrevFrameFlow, uv).xy;
            candidates[2] = tex2D(source_flow_sampler, uv + float2(0, -source_texel_size.y)).xy;
            candidates[3] = tex2D(source_flow_sampler, uv + float2(0, source_texel_size.y)).xy;
            candidates[4] = tex2D(source_flow_sampler, uv - float2(source_texel_size.x, 0)).xy;
            candidates[5] = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, 0)).xy;

            prediction = candidates[0];
            [unroll]
            for (int i = 0; i < 6; i++) {
                float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidates[i], texel_size, mip1);
                if (cost < min_cost) { min_cost = cost; prediction = candidates[i]; }
            }
        }
        else // HIGH
        {
            float2 candidates[12];
            candidates[0] = tex2D(source_flow_sampler, uv).xy;
            candidates[1] = tex2D(source_flow_sampler, uv + float2(0, -source_texel_size.y)).xy;
            candidates[2] = tex2D(source_flow_sampler, uv + float2(0, source_texel_size.y)).xy;
            candidates[3] = tex2D(source_flow_sampler, uv - float2(source_texel_size.x, 0)).xy;
            candidates[4] = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, 0)).xy;
            candidates[5] = tex2D(source_flow_sampler, uv + float2(-source_texel_size.x, -source_texel_size.y)).xy;
            candidates[6] = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, -source_texel_size.y)).xy;
            candidates[7] = tex2D(source_flow_sampler, uv + float2(-source_texel_size.x, source_texel_size.y)).xy;
            candidates[8] = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, source_texel_size.y)).xy;
            candidates[9] = tex2Dfetch(sGlobalFlow, int2(0, 0), 0).xy;
            candidates[10] = 0.0;
            candidates[11] = tex2D(sPrevFrameFlow, uv).xy;

            prediction = candidates[0];
            [loop]
            for (int i = 0; i < 12; i++) {
                float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidates[i], texel_size, mip1);
                if (cost < min_cost) { min_cost = cost; prediction = candidates[i]; }
            }
        }

        //=== (2) Residual Integer Search 
        texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip2));
        float2 integer_match = prediction;
        float match_cost = min_cost;

        if (QUALITY_PRESET == 1) // HIGH 
        {
            int2 c8[8] = { int2(-1, 1), int2(0, 1), int2(1, 1), int2(-1, 0), int2(1, 0), int2(-1,-1), int2(0,-1), int2(1,-1) };
            int2 c8_it[9] = { int2(6, 3), int2(0, 3), int2(0, 5), int2(2, 5), int2(2, 7), int2(4, 7), int2(4, 1), int2(6, 1), int2(0, 0) };
            float2 residual = 0.0;
            match_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction + residual, texel_size, mip2);
            int match_i = 8;

            [loop]
            for (int search = 0; search < 10; search++)
            {
                int i = c8_it[match_i].x;
                int end = c8_it[match_i].y;
                float2 search_center = residual;

                float2 candidate_residual = search_center + float2(c8[i]) * texel_size;
                float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction + candidate_residual, texel_size, mip2);
                if (cost < match_cost) { residual = candidate_residual; match_i = i; match_cost = cost; }

                i = (i + 1) & 7;
                [loop]
                for (int k = 0; k < 8; k++) {
                    if (i == end) break;
                    candidate_residual = search_center + float2(c8[i]) * texel_size;
                    cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction + candidate_residual, texel_size, mip2);
                    if (cost < match_cost) { residual = candidate_residual; match_i = i; match_cost = cost; }
                    i = (i + 1) & 7;
                }
                if (all(search_center == residual)) break;
                if (match_cost < 0.01) break;
            }
            integer_match = prediction + residual;
        }

        //=== (3) Sub-pixel refinement
        float cost_left = ZAD(sCurrLuma, sPrevLuma, uv, uv + integer_match - float2(texel_size.x, 0), texel_size, mip2);
        float cost_right = ZAD(sCurrLuma, sPrevLuma, uv, uv + integer_match + float2(texel_size.x, 0), texel_size, mip2);
        float cost_down = ZAD(sCurrLuma, sPrevLuma, uv, uv + integer_match - float2(0, texel_size.y), texel_size, mip2);
        float cost_up = ZAD(sCurrLuma, sPrevLuma, uv, uv + integer_match + float2(0, texel_size.y), texel_size, mip2);

        float2 subpixel_offset;
        subpixel_offset.x = (cost_left - cost_right) / (2.0 * (cost_left + cost_right - 2.0 * match_cost) + EPSILON);
        subpixel_offset.y = (cost_down - cost_up) / (2.0 * (cost_down + cost_up - 2.0 * match_cost) + EPSILON);
        subpixel_offset = clamp(subpixel_offset, -0.5, 0.5);

        return (integer_match + subpixel_offset * texel_size);
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

    float PS_CurrLuma(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        // MEDIUM
        [branch]
        if (QUALITY_PRESET == 0)
        {
            float3 center_color = GetColor(uv);
            float center_luma = dot(center_color, float3(0.2126, 0.7152, 0.0722));
            return center_luma * rcp(1.0 + center_luma);
        }
        // HIGH
        else
        {
            static const int2 DENSE_13[13] = {
                int2(0,-2), int2(-1,-1),int2(0,-1),int2(1,-1),
                int2(-2,0),int2(-1,0), int2(0,0), int2(1,0), int2(2,0),
                int2(-1,1), int2(0,1), int2(1,1), int2(0,2)
            };
            float weights[13] = { 1, 3, 4, 3, 1, 4, 6, 4, 1, 3, 4, 3, 1 };
            float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT));
            float luma_sum = 0.0;
            float weight_sum = 0.0;
            [loop]
            for (int i = 0; i < 13; i++)
            {
                float2 sample_uv = uv + float2(DENSE_13[i]) * texel_size;
                float3 color = GetColor(sample_uv);
                float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
                luma = luma * rcp(1.0 + luma);
                luma_sum += luma * weights[i];
                weight_sum += weights[i];
            }
            float smooth_luma = luma_sum / weight_sum;
            float3 center_color = GetColor(uv);
            float center_luma = dot(center_color, float3(0.2126, 0.7152, 0.0722));
            center_luma = center_luma * rcp(1.0 + center_luma);
            return lerp(center_luma, smooth_luma, 0.9);
        }
    }

    float2 PS_CoarseFlowL4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        static const int mip = 5;
        if (FRAME_COUNT == 0) return float2(0, 0);
        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip));

        // MEDIUM
        if (QUALITY_PRESET == 0)
        {
            float2 local_prediction = tex2D(sPrevFrameFlow, uv).xy;
            float2 static_prediction = float2(0, 0);
            float local_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + local_prediction, texel_size, mip);
            float static_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + static_prediction, texel_size, mip);
            float2 prediction = (static_cost < local_cost) ? static_prediction : local_prediction;
            float2 best_flow = prediction;
            float min_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction, texel_size, mip);

            [unroll]
            for (int y = -1; y <= 1; ++y) {
                [unroll]
                for (int x = -1; x <= 1; ++x) {
                    if (x == 0 && y == 0) continue;
                    float2 candidate_flow = prediction + float2(x, y) * texel_size;
                    float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidate_flow, texel_size, mip);
                    if (cost < min_cost) { min_cost = cost; best_flow = candidate_flow; }
                }
            }
            return best_flow;
        }
        // HIGH
        else
        {
            float2 global_prediction = tex2Dfetch(sGlobalFlow, int2(0, 0), 0).xy;
            float2 local_prediction = tex2D(sPrevFrameFlow, uv).xy;
            float2 static_prediction = float2(0, 0);
            float global_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + global_prediction, texel_size, mip);
            float local_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + local_prediction, texel_size, mip);
            float static_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + static_prediction, texel_size, mip);
            float2 prediction = (static_cost < local_cost) ?
                ((static_cost < global_cost) ? static_prediction : global_prediction) :
                ((local_cost < global_cost) ? local_prediction : global_prediction);
            
            float2 best_flow = prediction;
            float min_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction, texel_size, mip);
            
            [loop]
            for (int y = -3; y <= 3; ++y) {
                [loop]
                for (int x = -3; x <= 3; ++x) {
                    if (x == 0 && y == 0) continue;
                    float2 candidate_flow = prediction + float2(x, y) * texel_size;
                    float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidate_flow, texel_size, mip);
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_flow = candidate_flow;
                        if (min_cost < 0.01) return best_flow;
                    }
                }
            }
            return best_flow;
        }
    }

    float2 PS_CoarseFlowL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return ComputeFlow(sCoarseFlowL4, uv, 4, 4);
    }

    float2 PS_CoarseFlowL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return ComputeFlow(sCoarseFlowL3_B, uv, 3, 3);
    }

    float2 PS_CoarseFlowL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return ComputeFlow(sCoarseFlowL2_B, uv, 2, 2);
    }

    float2 PS_CoarseFlowL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return ComputeFlow(sCoarseFlowL1_B, uv, 1, 1);
    }

    float2 PS_DenseFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (FRAME_COUNT == 0) return float2(0, 0);
        return ComputeFlow(sCoarseFlowL0_B, uv, 1, 0);
    }

    float2 PS_GlobalFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        // MEDIUM (Sparse 8)
        if (QUALITY_PRESET == 0)
        {
            static const float2 SPARSE_SCREEN_LOW[8] = {
                float2(0.2, 0.2), float2(0.5, 0.2), float2(0.8, 0.2),
                float2(0.2, 0.5), float2(0.8, 0.5),
                float2(0.2, 0.8), float2(0.5, 0.8), float2(0.8, 0.8)
            };
            float x_values[8], y_values[8];
            uint count = 0;
            [unroll]
            for (int i = 0; i < 8; i++) {
                if (GetDepth(SPARSE_SCREEN_LOW[i]) < 0.999) {
                    float2 flow = tex2Dlod(sCoarseFlowL0_B, float4(SPARSE_SCREEN_LOW[i], 0, 0)).xy;
                    x_values[count] = flow.x; y_values[count] = flow.y; count++;
                }
            }
            if (count < 2u) return 0.0;
            float sum_x = 0, sum_y = 0;
            for (int k = 0; k < count; k++) { sum_x += x_values[k]; sum_y += y_values[k]; }
            return float2(sum_x, sum_y) / float(count);
        }
        // HIGH (Sparse 32 + Sort)
        else
        {
            static const float2 SPARSE_SCREEN_HIGH[32] = {
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
            for (int i = 0; i < 32; i++) {
                if (GetDepth(SPARSE_SCREEN_HIGH[i]) < 0.999) {
                    float2 flow = tex2Dlod(sCoarseFlowL0_B, float4(SPARSE_SCREEN_HIGH[i], 0, 0)).xy;
                    x_values[count] = flow.x; y_values[count] = flow.y; count++;
                }
            }
            if (count < 3u) return 0.0;
            uint mid = count / 2u;
            [loop]
            for (int k = 0; k < 32; k++) {
                if (k > mid) break;
                [loop]
                for (int j = 0; j < 32; j++) {
                    if (j >= count - 1 - k) break;
                    if (x_values[j] > x_values[j + 1]) { float tmp = x_values[j]; x_values[j] = x_values[j + 1]; x_values[j + 1] = tmp; }
                    if (y_values[j] > y_values[j + 1]) { float tmp = y_values[j]; y_values[j] = y_values[j + 1]; y_values[j + 1] = tmp; }
                }
            }
            return float2(x_values[mid], y_values[mid]);
        }
    }

    float2 PS_CopyFinalFlowToHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sFinalFlow, uv).xy;
    }

    float PS_CopyCurrLumaAsPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sCurrLuma, uv).r;
    }

    float4 PS_CopyCurrColorAsPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target0
    {
        return float4(GetColor(uv), 1);
    }

    float4 PS_Debug(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        switch (DEBUG_VIEW)
        {
            case 0: return float4(GetColor(uv), 1);
            case 1: return float4(MotionToColor(tex2Dlod(sTexMotionVectorsSampler, float4(uv, 0, 0)).xy), 1);
            case 2:
            {
                static const float ARROW_THICKNESS = 1.0;
                static const int GRID_STEP = 2;
                static const float ARROWHEAD_LENGTH = 4.0;
                static const float WING_ANGLE = 0.6;
                float3 base_color = GetColor(uv);
                float2 motion_res = float2(BUFFER_WIDTH, BUFFER_HEIGHT) / 4.0;
                float2 motion_texel_size = rcp(motion_res);
                float2 pixel_pos = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
                float2 motion_grid = floor(uv / motion_texel_size / GRID_STEP) * motion_texel_size * GRID_STEP + motion_texel_size * GRID_STEP * 0.5;
                float2 grid_pixel_pos = motion_grid * float2(BUFFER_WIDTH, BUFFER_HEIGHT);

                float2 motion = tex2D(sTexMotionVectorsSampler, motion_grid).xy;
                float2 motion_pixels = motion * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
                float motion_magnitude = length(motion_pixels);

                if (motion_magnitude < 0.5)
                    return float4(base_color, 1.0);
                
                float arrow_length = clamp(motion_magnitude * 3.0, 8.0, 48.0);
                float2 arrow_dir = normalize(-motion_pixels + float2(EPSILON, EPSILON));
                float2 arrow_tip = grid_pixel_pos + arrow_dir * arrow_length;

                float2 to_pixel = pixel_pos - grid_pixel_pos;
                float proj = dot(to_pixel, arrow_dir);
                float2 closest_shaft = arrow_dir * clamp(proj, 0, arrow_length - ARROWHEAD_LENGTH);
                float dist_shaft = length(to_pixel - closest_shaft);
                bool on_shaft = (dist_shaft < ARROW_THICKNESS) && (proj > 0) && (proj < arrow_length - ARROWHEAD_LENGTH);

                float2 back_dir = -arrow_dir;
                float2 wing_left_dir = float2(
                    back_dir.x * cos(WING_ANGLE) + back_dir.y * sin(WING_ANGLE),
                    -back_dir.x * sin(WING_ANGLE) + back_dir.y * cos(WING_ANGLE)
                );
                float2 wing_right_dir = float2(
                    back_dir.x * cos(WING_ANGLE) - back_dir.y * sin(WING_ANGLE),
                    back_dir.x * sin(WING_ANGLE) + back_dir.y * cos(WING_ANGLE)
                );
                float2 to_tip = pixel_pos - arrow_tip;
                float proj_left = dot(to_tip, wing_left_dir);
                float2 closest_left = wing_left_dir * clamp(proj_left, 0, ARROWHEAD_LENGTH);
                float dist_left = length(to_tip - closest_left);
                bool on_left = (dist_left < ARROW_THICKNESS) && (proj_left > 0) && (proj_left < ARROWHEAD_LENGTH);
                float proj_right = dot(to_tip, wing_right_dir);
                float2 closest_right = wing_right_dir * clamp(proj_right, 0, ARROWHEAD_LENGTH);
                float dist_right = length(to_tip - closest_right);
                bool on_right = (dist_right < ARROW_THICKNESS) && (proj_right > 0) && (proj_right < ARROWHEAD_LENGTH);

                float3 arrow_color = MotionToColor(motion);
                return float4((on_shaft || on_left || on_right) ? arrow_color : base_color, 1.0);
            }
            case 3:
            {
                float confidence = tex2D(sMotionConfidence, uv).x;
                float3 confidenceColor;
                if (confidence < 0.5)
                    confidenceColor = lerp(float3(1.0, 0.0, 0.0), float3(1.0, 1.0, 0.0), confidence * 2.0);
                else
                    confidenceColor = lerp(float3(1.0, 1.0, 0.0), float3(0.0, 1.0, 0.0), (confidence - 0.5) * 2.0);
                return float4(confidenceColor, 1.0);
            }
            default: return float4(GetColor(uv), 1);
        }
    }

    float2 PS_SpatialFilterL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median(sCoarseFlowL3_A, uv, rcp(tex2Dsize(sCoarseFlowL3_A, 0)), 6);
    }

    float2 PS_SpatialFilterL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median(sCoarseFlowL2_A, uv, rcp(tex2Dsize(sCoarseFlowL2_A, 0)), 5);
    }

    float2 PS_SpatialFilterL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median(sCoarseFlowL1_A, uv, rcp(tex2Dsize(sCoarseFlowL1_A, 0)), 4);
    }

    float2 PS_SpatialFilterL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median(sCoarseFlowL0_A, uv, rcp(tex2Dsize(sCoarseFlowL0_A, 0)), 3);
    }

    float2 PS_SmoothFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return RefineFlow(sDenseFlow_A, uv, rcp(tex2Dsize(sDenseFlow_A, 0)), 2);
    }

    float PS_Confidence(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (FRAME_COUNT == 0) return 0.0;
        float2 flow = tex2D(sFinalFlow, uv).xy;

        float2 prev_uv = uv + flow;

        if (IsOOB(prev_uv)) return 0.0;
        float curr_luma = tex2Dlod(sCurrLuma, float4(uv, 0, 2)).r;
        float prev_luma = tex2Dlod(sPrevLuma, float4(prev_uv, 0, 2)).r;
        float luma_error = abs(curr_luma - prev_luma);
        if (luma_error > 0.15)
            return 0.0;

        float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float subpixel_threshold = length(texel_size);
        float flow_magnitude = length(flow);

        if (flow_magnitude <= subpixel_threshold) return 1.0;
        float2 destination_flow = tex2D(sFinalFlow, prev_uv).xy;
        float2 flow_difference = flow - destination_flow;
        float error = length(flow_difference);
        float normalized_error = error / flow_magnitude;
        float motion_penalty = flow_magnitude / subpixel_threshold;

        float length_confidence = rcp(motion_penalty * 0.05 + 1.0);
        float consistency_confidence = rcp(normalized_error + 1.0);
        float photometric_confidence = exp(-luma_error * 5.0);

        return (consistency_confidence * length_confidence * photometric_confidence);
    }

    void PS_ExportFlow(float4 pos : SV_Position, float2 uv : TEXCOORD, out float2 flow : SV_Target0, out float confidence : SV_Target1)
    {
        flow = tex2D(sFinalFlow, uv).xy;
        confidence = tex2D(sConfidence, uv).x;
    }

    technique BarbatosFlow <
        ui_label = "BaBaFlow";
        ui_tooltip = "Dense Real-time Optical Flow.";
    >
    {
        //=== Luma pyramid
        pass { VertexShader = PostProcessVS; PixelShader = PS_CurrLuma; RenderTarget = tCurrLuma; }

        //=== Optical Flow
        // Coarse Flow Level 4
        pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL4; RenderTarget = tCoarseFlowL4; }

        // Coarse Flow Level 3
        pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL3; RenderTarget = tCoarseFlowL3_A; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL3; RenderTarget = tCoarseFlowL3_B; }

        // Coarse Flow Level 2
        pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL2; RenderTarget = tCoarseFlowL2_A; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL2; RenderTarget = tCoarseFlowL2_B; }

        // Coarse Flow Level 1
        pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL1; RenderTarget = tCoarseFlowL1_A; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL1; RenderTarget = tCoarseFlowL1_B; }

        // Coarse Flow Level 0
        pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL0; RenderTarget = tCoarseFlowL0_A; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL0; RenderTarget = tCoarseFlowL0_B; }

        // Dense Flow and Features
        pass { VertexShader = PostProcessVS; PixelShader = PS_DenseFlow; RenderTarget = tDenseFlow_A; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_SmoothFlow; RenderTarget = tDenseFlow_B; }

        // === Global Flow
        pass { VertexShader = PostProcessVS; PixelShader = PS_GlobalFlow; RenderTarget = tGlobalFlow; }

        // === Confidence Map for the Flow field
        pass { VertexShader = PostProcessVS; PixelShader = PS_Confidence; RenderTarget = tConfidence; }

        //=== Export the Flow
        pass { VertexShader = PostProcessVS; PixelShader = PS_ExportFlow; RenderTarget0 = texMotionVectors; RenderTarget1 = tMotionConfidence; }

        //=== Debug pass
        pass { VertexShader = PostProcessVS; PixelShader = PS_Debug; }

        //=== Save history
        pass { VertexShader = PostProcessVS; PixelShader = PS_CopyFinalFlowToHistory; RenderTarget = tPrevFlow; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_CopyCurrLumaAsPrev; RenderTarget = tPrevLuma; }
        pass { VertexShader = PostProcessVS; PixelShader = PS_CopyCurrColorAsPrev; RenderTarget = tPrevBackBuffer; }
    }
}
