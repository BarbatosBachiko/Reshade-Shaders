/*----------------------------------------------|
| ::       BaBa Launcher Resources            :: |
|----------------------------------------------*/

#pragma once

#include "bb_pipeline.fxh"

// ReShade shares resources that use the same unique name, so the Launcher can
// produce these once and dependent effects can sample them.
#if _BABA_USE_LAUNCHER
namespace BaBa_Launcher
{
    texture2D LinearDepth
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        // R16F quantizes distant geometry into visible stripes in the normal.
        Format = R32F;
        MipLevels = 6;
    };
    sampler2D sLinearDepth
    {
        Texture = LinearDepth;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D ViewDepth
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R32F;
    };
    sampler2D sViewDepth
    {
        Texture = ViewDepth;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D NormalRaw
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sNormalRaw
    {
        Texture = NormalRaw;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    // Ping-pong pair for the multi-pass (a-trous) normal filter.
    texture2D NormalSmoothH
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sNormalSmoothH
    {
        Texture = NormalSmoothH;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D NormalSmoothV
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sNormalSmoothV
    {
        Texture = NormalSmoothV;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    // Geometric (un-bumped) counterpart of NormalMap. Effects with their own
    // screen-space texture detail (BaBa: SSR, "Micro-Surface Details") must read
    // this instead, so the same Scharr signal is not applied twice. Aliases the
    // NormalSmoothV scratch buffer and is published before temporal stabilization.
    sampler2D sNormalGeometric
    {
        Texture = NormalSmoothV;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    // NormalMap stores xyz in rgb and linear depth in alpha.
    texture2D NormalMap
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
        MipLevels = 6;
    };
    sampler2D sNormalMap
    {
        Texture = NormalMap;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    // Compatibility output for AO shaders that use two-channel encoded normals.
    texture2D NormalEncoded
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
        MipLevels = 6;
    };
    sampler2D sNormalEncoded
    {
        Texture = NormalEncoded;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    // Temporal normal history: stabilized view-space normal in RGB, normalized
    // linear depth in A.
    texture2D NormalHistory0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sNormalHistory0
    {
        Texture = NormalHistory0;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D NormalHistory1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler2D sNormalHistory1
    {
        Texture = NormalHistory1;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D MotionVectors
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RG16F;
    };
    sampler2D sMotionVectors
    {
        Texture = MotionVectors;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

    texture2D MotionConfidence
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R16F;
    };
    sampler2D sMotionConfidence
    {
        Texture = MotionConfidence;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
}

float4 BaBa_Launcher_SampleNormalData(float2 uv)
{
    return tex2Dlod(BaBa_Launcher::sNormalMap,
        float4(clamp(uv, 0.0, 1.0), 0.0, 0.0));
}

#endif
