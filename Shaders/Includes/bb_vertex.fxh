/*----------------------------------------|
| ::        BaBa Vertex Shaders        :: |
|----------------------------------------*/

#pragma once

#include "bb_common.fxh"

struct VS_OUTPUT
{
    float4 vpos : SV_Position;
    float2 uv : TEXCOORD0;
    float2 pScale : TEXCOORD1;
};

void VS_Barbatos_FullScreen(in uint id : SV_VertexID, out VS_OUTPUT outStruct, float fovDegrees)
{
    outStruct.uv.x = (id == 2) ? 2.0 : 0.0;
    outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
    outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    float y = tan(max(0.01, fovDegrees) * DEG2RAD * 0.5);
    outStruct.pScale = float2(y * bb::AspectRatio, y);
}

void VS_Accumulate0(in uint id : SV_VertexID, out VS_OUTPUT outStruct, float fovDegrees)
{
    VS_Barbatos_FullScreen(id, outStruct, fovDegrees);
    if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
    {
        outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
    }
}

void VS_Accumulate1(in uint id : SV_VertexID, out VS_OUTPUT outStruct, float fovDegrees)
{
    VS_Barbatos_FullScreen(id, outStruct, fovDegrees);
    if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
    {
        outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
    }
}
