/*-------------------------------------------------|
| ::                  uFakeHDR                  :: |
| Version: 3.1                                     |
| Author: Barbatos                                 |
| License: CC0                                     |
'-------------------------------------------------*/

#include "ReShade.fxh"

namespace uFakeHDR
{
    static const int LUT_COUNT = 3;
    static const float LUT_SIZE = 64.0;
    static const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    static const float LUT_OFFSET = 0.5 / LUT_SIZE;
    static const float INV_LUT_SIZE = 1.0 / LUT_SIZE;
    static const float INV_LUT_COUNT = 1.0 / float(LUT_COUNT);
    static const float LUT_SIZE_MINUS_ONE = LUT_SIZE - 1.0;
    static const float DITHER_INTENSITY = 0.00196078431;

//----------|
// :: UI :: |
//----------|
    
    uniform int Preset <
        ui_label = "Color Grade";
        ui_type  = "combo";
        ui_items = "Natural\0Vivid\0FakeHDR\0";
    > = 2;

    uniform float Strength <
        ui_label = "INTENSITY";
        ui_type  = "slider";
        ui_min   = 0.0;
        ui_max   = 2.0;
    > = 1.0; 

//----------------|
// :: Textures :: |
//----------------|
    
    texture texLUTAtlas < source = "Barbatos_LUT_Atlas.png"; >
    {
        Width = 4096;
        Height = 192;
        Format = RGBA8;
    };

    sampler SamplerLUT
    {
        Texture = texLUTAtlas;
        AddressU = CLAMP;
        AddressV = CLAMP;
        MinFilter = LINEAR;
        MagFilter = LINEAR;
    };

/*------------------.
| :: Functions ::   |
'------------------*/
    
    float3 ApplyDither(float3 color, float2 pixelCoord)
    {
        float noise = frac(sin(dot(pixelCoord, float2(12.9898, 78.233))) * 43758.5453);
        return color + (noise * 2.0 - 1.0) * DITHER_INTENSITY;
    }

    float3 ApplyLUT(sampler s, float3 color, int presetIndex)
    {
        float3 uvw = color * LUT_SCALE + LUT_OFFSET;

        float slice = uvw.b * LUT_SIZE;
        float slice0 = floor(slice);
        float sliceFrac = slice - slice0;

        float u_base = uvw.r * INV_LUT_SIZE;
        float u0 = slice0 * INV_LUT_SIZE + u_base;
        float u1 = min(slice0 + 1.0, LUT_SIZE_MINUS_ONE) * INV_LUT_SIZE + u_base;

        float v_atlas = (uvw.g + float(presetIndex)) * INV_LUT_COUNT;

        float3 col0 = tex2D(s, float2(u0, v_atlas)).rgb;
        float3 col1 = tex2D(s, float2(u1, v_atlas)).rgb;

        return lerp(col0, col1, sliceFrac);
    }

    float3 PS_Main(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
    {
        const float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
        const float3 lut_output = ApplyLUT(SamplerLUT, color, Preset);
        const float3 final_color = lerp(color, lut_output, Strength);
        return ApplyDither(final_color, texcoord * ReShade::ScreenSize);
    }

    technique uFakeHDR
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Main;
        }
    }
}
