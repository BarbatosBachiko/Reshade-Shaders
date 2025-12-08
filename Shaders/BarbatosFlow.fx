/*----------------------------------------------|
| ::               BarbatosFlow              :: |
'-----------------------------------------------|
Version: 0.0.1                                
Author: Barbatos (Original by umar-afzaal - Kaidô)                               
License: CC-BY-NC-4.0 (https://github.com/umar-afzaal/LumeniteFX/blob/mainline/Shaders/LumaFlow.fx)
Description: A performance-focused version of LumaFlow with DX9 compatibility.
The optimization focused on using the shaders present in my repository; other shaders were not tested.
*/

#include "ReShade.fxh"

#define EPSILON 1e-6

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

texture2D tCurrLuma { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = 8; };
sampler2D sCurrLuma { Texture = tCurrLuma; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tPrevLuma { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = 8; };
sampler2D sPrevLuma { Texture = tPrevLuma; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

// <>
texture2D tCoarseFlowL4 { Width = BUFFER_WIDTH/128; Height = BUFFER_HEIGHT/128; Format = RG16F; };
sampler2D sCoarseFlowL4 { Texture = tCoarseFlowL4; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL3_A { Width = BUFFER_WIDTH/64; Height = BUFFER_HEIGHT/64; Format = RG16F; };
sampler2D sCoarseFlowL3_A { Texture = tCoarseFlowL3_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL3_B { Width = BUFFER_WIDTH/64; Height = BUFFER_HEIGHT/64; Format = RG16F; };
sampler2D sCoarseFlowL3_B { Texture = tCoarseFlowL3_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL2_A { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RG16F; };
sampler2D sCoarseFlowL2_A { Texture = tCoarseFlowL2_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL2_B { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RG16F; };
sampler2D sCoarseFlowL2_B { Texture = tCoarseFlowL2_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL1_A { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RG16F; };
sampler2D sCoarseFlowL1_A { Texture = tCoarseFlowL1_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL1_B { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RG16F; };
sampler2D sCoarseFlowL1_B { Texture = tCoarseFlowL1_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL0_A { Width = BUFFER_WIDTH/8; Height = BUFFER_HEIGHT/8; Format = RG16F; };
sampler2D sCoarseFlowL0_A { Texture = tCoarseFlowL0_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tCoarseFlowL0_B { Width = BUFFER_WIDTH/8; Height = BUFFER_HEIGHT/8; Format = RG16F; };
sampler2D sCoarseFlowL0_B { Texture = tCoarseFlowL0_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tDenseFlow_A { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RG16F; };
sampler2D sDenseFlow_A { Texture = tDenseFlow_A; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tDenseFlow_B { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RG16F; };
sampler2D sFinalFlow { Texture = tDenseFlow_B; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

texture2D tGlobalFlow { Width = 1; Height = 1; Format = RG16F; };
sampler2D sGlobalFlow { Texture = tGlobalFlow; };

texture2D tPrevFlow { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RG16F; };
sampler2D sPrevFrameFlow { Texture = tPrevFlow; MagFilter = POINT; MinFilter = POINT; };

texture2D tPrevBackBuffer { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler2D sPrevBackBuffer { Texture = tPrevBackBuffer; MagFilter = LINEAR; MinFilter = LINEAR; AddressU = CLAMP; AddressV = CLAMP; };

texture2D texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sTexMotionVectorsSampler { Texture = texMotionVectors; MagFilter = POINT; MinFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; AddressW = CLAMP; };

//---------------|
// :: Functions::|
//---------------|

float GetDepth(float2 xy) { return ReShade::GetLinearizedDepth(xy); }
float3 GetColor(float2 uv) { return tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb; }

float ZAD(sampler2D cur, sampler2D prev, float2 pos_a, float2 pos_b, float2 texel_size, int mip)
{
    float2 offsets[5] = { float2(0,0), float2(1.5,0), float2(-1.5,0), float2(0,1.5), float2(0,-1.5) };
    float mean_a = 0.0, mean_b = 0.0;

    [unroll]
    for(int i = 0; i < 5; i++)
    {
        float2 off = offsets[i] * texel_size;
        mean_a += tex2Dlod(cur,  float4(pos_a + off, 0, mip)).r;
        mean_b += tex2Dlod(prev, float4(pos_b + off, 0, mip)).r;
    }
    mean_a /= 5.0; mean_b /= 5.0;

    float err = 0.0;
    [unroll]
    for(int j = 0; j < 5; j++)
    {
        float2 off = offsets[j] * texel_size;
        float val_a = tex2Dlod(cur,  float4(pos_a + off, 0, mip)).r;
        float val_b = tex2Dlod(prev, float4(pos_b + off, 0, mip)).r;
        err += abs((val_a - mean_a) - (val_b - mean_b));
    }
    return ((err / 5.0) + EPSILON);
}

float2 Median5(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
{
    float2 v[5];
    v[0] = tex2Dlod(motion_tex, float4(uv, 0, mip)).xy;
    v[1] = tex2Dlod(motion_tex, float4(uv + float2( texel_size.x, 0), 0, mip)).xy;
    v[2] = tex2Dlod(motion_tex, float4(uv + float2(-texel_size.x, 0), 0, mip)).xy;
    v[3] = tex2Dlod(motion_tex, float4(uv + float2(0,  texel_size.y), 0, mip)).xy;
    v[4] = tex2Dlod(motion_tex, float4(uv + float2(0, -texel_size.y), 0, mip)).xy;

    [unroll]
    for (int i = 0; i < 4; i++) {
        [unroll]
        for (int j = 0; j < 4 - i; j++) {
            if (v[j].x > v[j+1].x) { float temp = v[j].x; v[j].x = v[j+1].x; v[j+1].x = temp; }
            if (v[j].y > v[j+1].y) { float temp = v[j].y; v[j].y = v[j+1].y; v[j+1].y = temp; }
        }
    }
    return float2(v[2].x, v[2].y);
}

float2 BilateralFilter(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
{
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

float2 ComputeFlow(sampler2D source_flow_sampler, float2 uv, int mip1, int mip2)
{
    if(FRAME_COUNT == 0) return float2(0, 0);
    #define SEARCH_ITER 4

    int2 c8[8] = { int2(-1, 1), int2(0, 1), int2(1, 1), int2(-1, 0), int2(1, 0), int2(-1,-1), int2(0,-1), int2(1,-1) };
    int2 c8_it[9] = { int2(6, 3), int2(0, 3), int2(0, 5), int2(2, 5), int2(2, 7), int2(4, 7), int2(4, 1), int2(6, 1), int2(0, 0) };
    
    float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip1));
    float2 source_texel_size = rcp(float2(tex2Dsize(source_flow_sampler, 0)));

    float2 candidates[12];
    candidates[0]  = tex2D(source_flow_sampler, uv).xy;
    candidates[1]  = tex2D(source_flow_sampler, uv + float2(0, -source_texel_size.y)).xy;
    candidates[2]  = tex2D(source_flow_sampler, uv + float2(0,  source_texel_size.y)).xy;
    candidates[3]  = tex2D(source_flow_sampler, uv - float2(source_texel_size.x, 0)).xy;
    candidates[4]  = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, 0)).xy;
    candidates[5]  = tex2D(source_flow_sampler, uv + float2(-source_texel_size.x, -source_texel_size.y)).xy;
    candidates[6]  = tex2D(source_flow_sampler, uv + float2( source_texel_size.x, -source_texel_size.y)).xy;
    candidates[7]  = tex2D(source_flow_sampler, uv + float2(-source_texel_size.x,  source_texel_size.y)).xy;
    candidates[8]  = tex2D(source_flow_sampler, uv + float2(source_texel_size.x, source_texel_size.y)).xy;
    candidates[9]  = tex2Dlod(sGlobalFlow, float4(0.5, 0.5, 0, 0)).xy;
    candidates[10] = 0.0;
    candidates[11] = tex2D(sPrevFrameFlow, uv).xy;

    float min_cost = 1e6;
    float2 prediction = candidates[0];

    [unroll]
    for (int i = 0; i < 12; i++)
    {
        float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidates[i], texel_size, mip1);
        if (cost < min_cost) { min_cost = cost; prediction = candidates[i]; }
    }

    texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip2));
    float2 residual = 0.0;
    float match_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction + residual, texel_size, mip2);
    int match_i = 8;

    [loop]
    for (int search = 0; search < SEARCH_ITER; search++)
    {
        int i = c8_it[match_i].x;
        int end = c8_it[match_i].y;
        float2 search_center = residual;

        [loop]
        for(int k=0; k<8; k++)
        {
            float2 candidate_residual = search_center + float2(c8[i]) * texel_size;
            float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction + candidate_residual, texel_size, mip2);
            if (cost < match_cost) { residual = candidate_residual; match_i = i; match_cost = cost; }
            i++; if (i > 7) i = 0; if (i == end) break;
        }
        if (all(search_center == residual)) break;
        if (match_cost < 0.01) break;
    }
    return prediction + residual;
}

//---------------------|
// :: Pixel Shaders  ::|
//---------------------|
float PS_CurrLuma(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float luma_center = dot(GetColor(uv), float3(0.2126, 0.7152, 0.0722));
    float luma_u = dot(GetColor(uv + float2(0, -texel_size.y)), float3(0.2126, 0.7152, 0.0722));
    float luma_d = dot(GetColor(uv + float2(0,  texel_size.y)), float3(0.2126, 0.7152, 0.0722));
    float luma_l = dot(GetColor(uv + float2(-texel_size.x, 0)), float3(0.2126, 0.7152, 0.0722));
    float luma_r = dot(GetColor(uv + float2( texel_size.x, 0)), float3(0.2126, 0.7152, 0.0722));
    return (luma_center * 4.0 + luma_u + luma_d + luma_l + luma_r) * 0.125;
}

float2 PS_CoarseFlowL4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    #define SEARCH_RADIUS 2
    static const int mip = 5;
    if(FRAME_COUNT == 0) return float2(0, 0);

    float2 texel_size = rcp(float2(BUFFER_WIDTH, BUFFER_HEIGHT) / exp2(mip));
    float2 global_prediction = tex2Dlod(sGlobalFlow, float4(0.5, 0.5, 0, 0)).xy; 
    float2 local_prediction = tex2D(sPrevFrameFlow, uv).xy;
    float global_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + global_prediction, texel_size, mip);
    float local_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + local_prediction, texel_size, mip);
    
    float2 prediction = (local_cost < global_cost) ? local_prediction : global_prediction;
    float2 best_flow = prediction;
    float min_cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + prediction, texel_size, mip);
    [loop]
    for (int y = -SEARCH_RADIUS; y <= SEARCH_RADIUS; ++y)
    {
        [loop]
        for (int x = -SEARCH_RADIUS; x <= SEARCH_RADIUS; ++x)
        {
            if (x == 0 && y == 0) continue;
            float2 candidate_flow = prediction + float2(x, y) * texel_size;
            float cost = ZAD(sCurrLuma, sPrevLuma, uv, uv + candidate_flow, texel_size, mip);
            if (cost < min_cost) { min_cost = cost; best_flow = candidate_flow; }
        }
    }
    return best_flow;
}

float2 PS_CoarseFlowL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL4, uv, 4, 4); }
float2 PS_CoarseFlowL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL3_B, uv, 3, 3); }
float2 PS_CoarseFlowL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL2_B, uv, 2, 2); }
float2 PS_CoarseFlowL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return ComputeFlow(sCoarseFlowL1_B, uv, 1, 1); }
float2 PS_DenseFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { if(FRAME_COUNT == 0) return 0.0; return ComputeFlow(sCoarseFlowL0_B, uv, 1, 0); }

float2 PS_GlobalFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    static const float2 SPARSE_SCREEN[16] = {
        float2(0.125, 0.125), float2(0.375, 0.125), float2(0.625, 0.125), float2(0.875, 0.125),
        float2(0.125, 0.375), float2(0.375, 0.375), float2(0.625, 0.375), float2(0.875, 0.375),
        float2(0.125, 0.625), float2(0.375, 0.625), float2(0.625, 0.625), float2(0.875, 0.625),
        float2(0.125, 0.875), float2(0.375, 0.875), float2(0.625, 0.875), float2(0.875, 0.875)
    };
    float2 flow_sum = 0.0;
    float weight_sum = 0.0;
    [unroll]
    for(int i = 0; i < 16; i++)
    {
        float depth = GetDepth(SPARSE_SCREEN[i]);
        if(depth < 0.999) { flow_sum += tex2Dlod(sCoarseFlowL0_B, float4(SPARSE_SCREEN[i], 0, 0)).xy; weight_sum += 1.0; }
    }
    return (weight_sum > 0.0) ? (flow_sum / weight_sum) : 0.0;
}

float PS_CopyFinalFlowToHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return tex2D(sFinalFlow, uv).xy; }
float PS_CopyCurrLumaAsPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return tex2D(sCurrLuma, uv).r; }
float4 PS_CopyCurrColorAsPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target0 { return float4(GetColor(uv), 1); }

float2 PS_SpatialFilterL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median5(sCoarseFlowL3_A, uv, rcp(float2(tex2Dsize(sCoarseFlowL3_A, 0))), 6); }
float2 PS_SpatialFilterL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median5(sCoarseFlowL2_A, uv, rcp(float2(tex2Dsize(sCoarseFlowL2_A, 0))), 5); }
float2 PS_SpatialFilterL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median5(sCoarseFlowL1_A, uv, rcp(float2(tex2Dsize(sCoarseFlowL1_A, 0))), 4); }
float2 PS_SpatialFilterL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return Median5(sCoarseFlowL0_A, uv, rcp(float2(tex2Dsize(sCoarseFlowL0_A, 0))), 3); }

float2 PS_SmoothFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return BilateralFilter(sDenseFlow_A, uv, rcp(float2(tex2Dsize(sDenseFlow_A, 0))), 2); }

float2 PS_ExportFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2D(sFinalFlow, uv).xy;
}

technique BarbatosFlow
{
    pass { VertexShader = PostProcessVS; PixelShader = PS_CurrLuma; RenderTarget = tCurrLuma; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL4; RenderTarget = tCoarseFlowL4; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL3; RenderTarget = tCoarseFlowL3_A; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL3; RenderTarget = tCoarseFlowL3_B; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL2; RenderTarget = tCoarseFlowL2_A; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL2; RenderTarget = tCoarseFlowL2_B; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL1; RenderTarget = tCoarseFlowL1_A; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL1; RenderTarget = tCoarseFlowL1_B; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CoarseFlowL0; RenderTarget = tCoarseFlowL0_A; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_SpatialFilterL0; RenderTarget = tCoarseFlowL0_B; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_DenseFlow; RenderTarget = tDenseFlow_A; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_SmoothFlow; RenderTarget = tDenseFlow_B; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_GlobalFlow; RenderTarget = tGlobalFlow; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_ExportFlow; RenderTarget = texMotionVectors; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CopyFinalFlowToHistory; RenderTarget = tPrevFlow; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CopyCurrLumaAsPrev; RenderTarget = tPrevLuma; }
    pass { VertexShader = PostProcessVS; PixelShader = PS_CopyCurrColorAsPrev; RenderTarget = tPrevBackBuffer; }
}