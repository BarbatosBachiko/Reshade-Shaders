/*--------------------------------------|
| ::     bb Hilbert SpatioTemporal   :: |
|--------------------------------------*/

#pragma once

#ifndef USE_HILBERT_LUT
#define USE_HILBERT_LUT 1
#endif

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

#if !USE_HILBERT_LUT
float hilbert_procedural(float2 p, int level)
{
    float d = 0;
    [unroll]
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
                p = n_pow2 - 1.0 - p;
            p = p.yx;
        }
    }
    return d;
}

uint HilbertIndex_Procedural(uint x, uint y)
{
    return (uint)hilbert_procedural(float2(x % 64, y % 64), 6);
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
    index = HilbertIndex_Procedural(pixCoord.x, pixCoord.y);
#endif

#if defined(BB_HILBERT_TEMPORAL_AND)
#if __RENDERER__ >= 0xa000
    index += 288 * (temporalIndex & 63);
#else
    index += 288 * (temporalIndex % 64);
#endif
#else
    index += 288 * (temporalIndex % 64);
#endif

    return frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114));
}
