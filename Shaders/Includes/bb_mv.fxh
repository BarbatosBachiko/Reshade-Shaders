/*----------------------------------------------|
| ::      BaBa Motion Vectors Utils          :: |
|----------------------------------------------*/

#pragma once

#include "bb_pipeline.fxh"

#if _BABA_MOTION_PROVIDER == 0
    #include "bb_launcher_resources.fxh"
#endif

#include "bb_depth.fxh"

#ifndef _BABA_MV_CONFIDENCE_SENSITIVITY
    #if defined(MV_CONFIDENCE_SENSITIVITY)
        #define _BABA_MV_CONFIDENCE_SENSITIVITY MV_CONFIDENCE_SENSITIVITY
    #else
        #define _BABA_MV_CONFIDENCE_SENSITIVITY 1.0
    #endif
#endif

//----------------|
// :: Textures :: |
//----------------|

#if _BABA_MOTION_PROVIDER == 0
    // BaBa_Launcher owns this resource and must run before consumers.
#elif _BABA_MOTION_PROVIDER == 2
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
#elif _BABA_MOTION_PROVIDER == 3
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort
    {
        Texture   = MotVectTexVort;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU  = Clamp;
        AddressV  = Clamp;
    };
#elif _BABA_MOTION_PROVIDER == 5
    namespace QuantMotion {
        texture2D tFlow { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F; };
        sampler2D sFlow { Texture = tFlow; MagFilter = POINT; MinFilter = POINT; };

        texture2D tConfidence { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = R16F; };
        sampler2D sConfidence { Texture = tConfidence; };
    }
#elif _BABA_MOTION_PROVIDER == 4
    namespace Kernel {
        texture2D tFlow { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = RG16F; };
        sampler2D sFlow { Texture = tFlow; MagFilter = POINT; MinFilter = POINT; };

        texture2D tConfidence { Width = BUFFER_WIDTH / 8; Height = BUFFER_HEIGHT / 8; Format = R16F; };
        sampler2D sConfidence { Texture = tConfidence; };
    }
#else
    texture texMotionVectors
    {
        Width   = BUFFER_WIDTH;
        Height  = BUFFER_HEIGHT;
        Format  = RG16F;
    };
    sampler sTexMotionVectorsSampler
    {
        Texture   = texMotionVectors;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU  = Clamp;
        AddressV  = Clamp;
    };
#endif

float2 SampleMotionVectors(float2 texcoord)
{
#if _BABA_MOTION_PROVIDER == 0
    return tex2Dlod(BaBa_Launcher::sMotionVectors, float4(texcoord, 0, 0)).rg;
#elif _BABA_MOTION_PROVIDER == 2
    return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
#elif _BABA_MOTION_PROVIDER == 3
    return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).rg;
#elif _BABA_MOTION_PROVIDER == 5
    return tex2Dlod(QuantMotion::sFlow, float4(texcoord, 0, 0)).rg;
#elif _BABA_MOTION_PROVIDER == 4
    return tex2Dlod(Kernel::sFlow, float4(texcoord, 0, 0)).rg;
#else
    return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).rg;
#endif
}

float2 MV_GetVelocity(float2 texcoord)
{
    float2 pixel_size    = bb::PixelSize;
    float  closest_depth = 1.0;
    float2 closest_vel   = 0.0;

    static const float2 offsets[5] = {
        float2( 0,  0),
        float2( 0, -1), float2(-1,  0),
        float2( 1,  0), float2( 0,  1)
    };
    [unroll]
    for (int i = 0; i < 5; i++)
    {
        float2 s = texcoord + offsets[i] * pixel_size;
        float  d = GetDepth(s);
        if (d < closest_depth)
        {
            closest_depth = d;
            closest_vel   = SampleMotionVectors(s);
        }
    }
    return closest_vel;
}

float MV_GetConfidence(float2 texcoord)
{
#if _BABA_MOTION_PROVIDER == 5
    return saturate(tex2Dlod(QuantMotion::sConfidence, float4(texcoord, 0, 0)).r);
#elif _BABA_MOTION_PROVIDER == 4
    return saturate(tex2Dlod(Kernel::sConfidence, float4(texcoord, 0, 0)).r);
#else
    float2 velocity = MV_GetVelocity(texcoord);
    float2 prev_uv  = texcoord + velocity;

    if (any(saturate(prev_uv) != prev_uv))
        return 0.0;

    float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float flow_magnitude = length(velocity * resolution);

    float consistency_conf = 1.0;
    if (flow_magnitude > 0.5)
    {
        float2 dest_velocity = SampleMotionVectors(prev_uv);
        float error = length(velocity - dest_velocity);
        float normalized_error = (error * _BABA_MV_CONFIDENCE_SENSITIVITY * 1.25) / (length(velocity) + 1e-6);
        consistency_conf = rcp(normalized_error + 1.0);
    }

    float length_conf = rcp(flow_magnitude * 0.02 + 1.0);

    float curr_depth = GetDepth(texcoord);
    float dest_depth = GetDepth(prev_uv);
    float depth_conf = exp(-abs(curr_depth - dest_depth) * 100.0);

    return saturate(consistency_conf * length_conf * depth_conf);
#endif
}

float MV_GetConfidenceAO(float2 uv, float2 velocity, float flow_magnitude, float curr_luma, sampler sLumaPrev)
{
#if _BABA_MOTION_PROVIDER == 5
    return saturate(tex2Dlod(QuantMotion::sConfidence, float4(uv, 0, 0)).r);
#elif _BABA_MOTION_PROVIDER == 4
    return saturate(tex2Dlod(Kernel::sConfidence, float4(uv, 0, 0)).r);
#else
    float2 prev_uv = uv + velocity;

    if (any(saturate(prev_uv) != prev_uv))
        return 0.0;

    float consistency_conf = 1.0;
    if (flow_magnitude > 0.5)
    {
        float2 dest_velocity = SampleMotionVectors(prev_uv);
        float error = length(velocity - dest_velocity);
        float normalized_error = (error * _BABA_MV_CONFIDENCE_SENSITIVITY) / (length(velocity) + 1e-6);
        consistency_conf = rcp(normalized_error + 1.0);
    }

    float length_conf = rcp(flow_magnitude * 0.02 + 1.0);

    float prev_luma = tex2Dlod(sLumaPrev, float4(prev_uv, 0, 0)).r;
    float photometric_conf = exp(-abs(curr_luma - prev_luma) * 1.5);

    float curr_depth = GetDepth(uv);
    float dest_depth = GetDepth(prev_uv);
    float depth_conf = exp(-abs(curr_depth - dest_depth) * 100.0);

    return saturate(consistency_conf * length_conf * photometric_conf * depth_conf);
#endif
}
