/*----------------------------------------------|
| ::            BarbatosFlow Lite            :: |
'-----------------------------------------------|
Version: 1.0                               
Author: Barbatos (Original by umar-afzaal - Kaidô)                               
License: CC-BY-NC-4.0 (https://github.com/umar-afzaal/LumeniteFX/blob/mainline/Shaders/LumaFlow.fx)
Description: A performance-focused version of LumaFlow with DX9 compatibility.
The optimization focused on using the shaders present in my repository; other shaders were not tested.
*/

#include "ReShade.fxh"
uniform int FRAME_COUNT < source = "framecount"; >;

texture2D texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};

//----------------|
// :: Textures :: |
//----------------|
namespace Barbatos_Flow_Lite
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
        AddressU = CLAMP;
        AddressV = CLAMP;
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
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tCoarseFlowL4
    {
        Width = BUFFER_WIDTH / 128;
        Height = BUFFER_HEIGHT / 128;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL4
    {
        Texture = tCoarseFlowL4;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tCoarseFlowL3_A
    {
        Width = BUFFER_WIDTH / 64;
        Height = BUFFER_HEIGHT / 64;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL3_A
    {
        Texture = tCoarseFlowL3_A;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    texture2D tCoarseFlowL3_B
    {
        Width = BUFFER_WIDTH / 64;
        Height = BUFFER_HEIGHT / 64;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL3_B
    {
        Texture = tCoarseFlowL3_B;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tCoarseFlowL2_A
    {
        Width = BUFFER_WIDTH / 32;
        Height = BUFFER_HEIGHT / 32;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL2_A
    {
        Texture = tCoarseFlowL2_A;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    texture2D tCoarseFlowL2_B
    {
        Width = BUFFER_WIDTH / 32;
        Height = BUFFER_HEIGHT / 32;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL2_B
    {
        Texture = tCoarseFlowL2_B;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tCoarseFlowL1_A
    {
        Width = BUFFER_WIDTH / 16;
        Height = BUFFER_HEIGHT / 16;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL1_A
    {
        Texture = tCoarseFlowL1_A;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    texture2D tCoarseFlowL1_B
    {
        Width = BUFFER_WIDTH / 16;
        Height = BUFFER_HEIGHT / 16;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL1_B
    {
        Texture = tCoarseFlowL1_B;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tCoarseFlowL0_A
    {
        Width = BUFFER_WIDTH / 8;
        Height = BUFFER_HEIGHT / 8;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL0_A
    {
        Texture = tCoarseFlowL0_A;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    texture2D tCoarseFlowL0_B
    {
        Width = BUFFER_WIDTH / 8;
        Height = BUFFER_HEIGHT / 8;
        Format = RG16F;
    };
    sampler2D sCoarseFlowL0_B
    {
        Texture = tCoarseFlowL0_B;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tDenseFlow_A
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RG16F;
    };
    sampler2D sDenseFlow_A
    {
        Texture = tDenseFlow_A;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    texture2D tDenseFlow_B
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RG16F;
    };
    sampler2D sFinalFlow
    {
        Texture = tDenseFlow_B;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D tGlobalFlow
    {
        Width = 1;
        Height = 1;
        Format = RG16F;
    };
    sampler2D sGlobalFlow
    {
        Texture = tGlobalFlow;
    };

    texture2D tPrevFlow
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RG16F;
    };
    sampler2D sPrevFrameFlow
    {
        Texture = tPrevFlow;
        MagFilter = POINT;
        MinFilter = POINT;
    };


//---------------|
// :: Functions::|
//---------------|
    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }
    float3 GetColor(float2 uv)
    {
        return tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb;
    }

    float Cost(sampler2D cur, sampler2D prev, float2 uv, float2 motion, int mip)
    {
        float c = tex2Dlod(cur, float4(uv, 0, mip)).r;
        float p = tex2Dlod(prev, float4(uv + motion, 0, mip)).r;
        return abs(c - p);
    }

    float2 SmartBlur(sampler2D motion_tex, float2 uv, float2 texel_size)
    {
        float2 center_flow = tex2Dlod(motion_tex, float4(uv, 0, 0)).xy;
        float center_depth = GetDepth(uv);
        float2 flow_sum = center_flow;
        float weight_sum = 1.0;

        float2 offsets[4] = { float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1) };
    
    [unroll]
        for (int i = 0; i < 4; i++)
        {
            float2 sample_uv = uv + offsets[i] * texel_size;
            float depth = GetDepth(sample_uv);
            if (abs(depth - center_depth) < 0.02)
            {
                flow_sum += tex2Dlod(motion_tex, float4(sample_uv, 0, 0)).xy;
                weight_sum += 1.0;
            }
        }
        return flow_sum / weight_sum;
    }

    void MnMx(inout float2 a, inout float2 b)
    {
        float2 mn = min(a, b);
        float2 mx = max(a, b);
        a = mn;
        b = mx;
    }

    float2 Median5(sampler2D motion_tex, float2 uv, float2 texel_size, int mip)
    {
        float2 v[5];
        v[0] = tex2Dlod(motion_tex, float4(uv, 0, mip)).xy;
        v[1] = tex2Dlod(motion_tex, float4(uv + float2(texel_size.x, 0), 0, mip)).xy;
        v[2] = tex2Dlod(motion_tex, float4(uv + float2(-texel_size.x, 0), 0, mip)).xy;
        v[3] = tex2Dlod(motion_tex, float4(uv + float2(0, texel_size.y), 0, mip)).xy;
        v[4] = tex2Dlod(motion_tex, float4(uv + float2(0, -texel_size.y), 0, mip)).xy;
    
        MnMx(v[0], v[1]);
        MnMx(v[1], v[2]);
        MnMx(v[2], v[3]);
        MnMx(v[3], v[4]);
        MnMx(v[0], v[1]);
        MnMx(v[1], v[2]);
        MnMx(v[2], v[3]);
        MnMx(v[0], v[1]);
        MnMx(v[1], v[2]);

        return v[2];
    }

    float2 ComputeFlow(sampler2D source_flow_sampler, float2 uv, int mip1, int mip2)
    {
        if (FRAME_COUNT == 0)
            return 0.0;

        float2 texel_size_mip = BUFFER_PIXEL_SIZE * exp2(mip1);
        float2 src_texel = rcp(float2(tex2Dsize(source_flow_sampler, 0)));

        float2 candidates[6];
        candidates[0] = tex2D(source_flow_sampler, uv).xy; // Center
        candidates[1] = tex2D(source_flow_sampler, uv + float2(-src_texel.x, 0)).xy; // Left
        candidates[2] = tex2D(source_flow_sampler, uv + float2(src_texel.x, 0)).xy; // Right
        candidates[3] = tex2D(source_flow_sampler, uv + float2(0, -src_texel.y)).xy; // Up
        candidates[4] = tex2D(source_flow_sampler, uv + float2(0, src_texel.y)).xy; // Down
        candidates[5] = tex2D(sPrevFrameFlow, uv).xy; // Temporal

        float min_cost = 100.0;
        float2 best_vec = candidates[0];

    [unroll]
        for (int i = 0; i < 6; i++)
        {
            float cost = Cost(sCurrLuma, sPrevLuma, uv, candidates[i], mip1);
            if (cost < min_cost)
            {
                min_cost = cost;
                best_vec = candidates[i];
            }
        }

        float2 refine_size = BUFFER_PIXEL_SIZE * exp2(mip2);
        float2 offsets[4] = { float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1) };
    
    [unroll]
        for (int j = 0; j < 4; j++)
        {
            float2 test_vec = best_vec + offsets[j] * refine_size;
            float cost = Cost(sCurrLuma, sPrevLuma, uv, test_vec, mip2);
            if (cost < min_cost)
            {
                min_cost = cost;
                best_vec = test_vec;
            }
        }

        return best_vec;
    }

//---------------------|
// :: Pixel Shaders  ::|
//---------------------|
    float PS_CurrLuma(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 t = BUFFER_PIXEL_SIZE;
    
        float l = dot(GetColor(uv), 0.333);
        l += dot(GetColor(uv + float2(t.x, 0)), 0.333);
        l += dot(GetColor(uv - float2(t.x, 0)), 0.333);
        l += dot(GetColor(uv + float2(0, t.y)), 0.333);
        l += dot(GetColor(uv - float2(0, t.y)), 0.333);
        return l * 0.2;
    }

    float2 PS_CoarseFlowL4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (FRAME_COUNT == 0)
            return 0.0;
        float2 prev = tex2D(sPrevFrameFlow, uv).xy;
        float2 glob = tex2Dlod(sGlobalFlow, float4(0.5, 0.5, 0, 0)).xy;
    
        float c_prev = Cost(sCurrLuma, sPrevLuma, uv, prev, 5);
        float c_glob = Cost(sCurrLuma, sPrevLuma, uv, glob, 5);
    
        return (c_prev < c_glob) ? prev : glob;
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
        return ComputeFlow(sCoarseFlowL0_B, uv, 1, 0);
    }

    float2 PS_GlobalFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2Dlod(sCoarseFlowL0_B, float4(0.5, 0.5, 0, 0)).xy;
    }

    float2 PS_CopyFinalFlowToHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sFinalFlow, uv).xy;
    }

    float PS_CopyCurrLumaAsPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sCurrLuma, uv).r;
    }

    float2 PS_SpatialFilterL3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median5(sCoarseFlowL3_A, uv, BUFFER_PIXEL_SIZE * 64.0, 6);
    }
    float2 PS_SpatialFilterL2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median5(sCoarseFlowL2_A, uv, BUFFER_PIXEL_SIZE * 32.0, 5);
    }
    float2 PS_SpatialFilterL1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median5(sCoarseFlowL1_A, uv, BUFFER_PIXEL_SIZE * 16.0, 4);
    }
    float2 PS_SpatialFilterL0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return Median5(sCoarseFlowL0_A, uv, BUFFER_PIXEL_SIZE * 8.0, 3);
    }

    float2 PS_SmoothFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return SmartBlur(sDenseFlow_A, uv, BUFFER_PIXEL_SIZE * 4.0);
    }
    float2 PS_ExportFlow(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        return tex2D(sFinalFlow, uv).xy;
    }

    technique BarbatosFlow
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CurrLuma;
            RenderTarget = tCurrLuma;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CoarseFlowL4;
            RenderTarget = tCoarseFlowL4;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CoarseFlowL3;
            RenderTarget = tCoarseFlowL3_A;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SpatialFilterL3;
            RenderTarget = tCoarseFlowL3_B;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CoarseFlowL2;
            RenderTarget = tCoarseFlowL2_A;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SpatialFilterL2;
            RenderTarget = tCoarseFlowL2_B;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CoarseFlowL1;
            RenderTarget = tCoarseFlowL1_A;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SpatialFilterL1;
            RenderTarget = tCoarseFlowL1_B;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CoarseFlowL0;
            RenderTarget = tCoarseFlowL0_A;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SpatialFilterL0;
            RenderTarget = tCoarseFlowL0_B;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenseFlow;
            RenderTarget = tDenseFlow_A;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SmoothFlow;
            RenderTarget = tDenseFlow_B;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GlobalFlow;
            RenderTarget = tGlobalFlow;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ExportFlow;
            RenderTarget = texMotionVectors;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CopyFinalFlowToHistory;
            RenderTarget = tPrevFlow;
        }
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CopyCurrLumaAsPrev;
            RenderTarget = tPrevLuma;
        }
    }
}
