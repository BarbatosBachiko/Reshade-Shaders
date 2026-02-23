/*-------------------------------------------------|
| ::                  uFakeHDR                  :: |
| Version: 3.2                                     |
| Author: Barbatos                                 |
| License: CC0                                     |
'-------------------------------------------------*/

namespace uFakeHDR
{
    static const int LUT_COUNT = 3;
    static const float LUT_SIZE = 64.0;
    static const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    static const float LUT_OFFSET = 0.5 / LUT_SIZE;
    static const float INV_LUT_SIZE = 1.0 / LUT_SIZE;
    static const float INV_LUT_COUNT = 1.0 / float(LUT_COUNT);
    static const float LUT_SIZE_MINUS_ONE = LUT_SIZE - 1.0;
    static const float DITHER_INTENSITY = 0.00392156862;
    static const float2 ScreenSize = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    
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

    texture BackBufferTex : COLOR;

    sampler Color
    {
        Texture = BackBufferTex;
    };
    
/*------------------.
| :: Functions ::   |
'------------------*/

    struct VSOUT
    {
        float4 vpos : SV_Position;
        float2 texcoord : TEXCOORD0;
        float magicDot : TEXCOORD1;
    };

    VSOUT VS_FHD(in uint id : SV_VertexID)
    {
        VSOUT o;
        
        o.texcoord.x = (id == 2) ? 2.0 : 0.0;
        o.texcoord.y = (id == 1) ? 2.0 : 0.0;
        o.vpos = float4(o.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
        
        const float2 pixelCoord = o.texcoord * ScreenSize;
        o.magicDot = dot(pixelCoord, float2(0.06711056, 0.00583715));
        
        return o;
    }

    float3 ApplyDither(const float3 color, const float magicDot)
    {
        const float noise1 = frac(52.9829189 * frac(magicDot));
        const float noise2 = frac(52.9829189 * frac(magicDot + 0.036473855));
        const float triangleNoise = noise1 + noise2 - 1.0;
        
        return color + (triangleNoise * DITHER_INTENSITY);
    }

    float3 ApplyLUT(sampler s, const float3 color, const int presetIndex)
    {
        const float3 clampedColor = saturate(color);
        const float3 uvw = clampedColor * LUT_SCALE + LUT_OFFSET;
        
        const float slice = clampedColor.b * LUT_SIZE_MINUS_ONE;
        const float slice0 = floor(slice);
        const float sliceFrac = frac(slice);
        
        const float u0 = (slice0 + uvw.r) * INV_LUT_SIZE;
        const float u1 = (min(slice0 + 1.0, LUT_SIZE_MINUS_ONE) + uvw.r) * INV_LUT_SIZE;

        const float v_atlas = (uvw.g + float(presetIndex)) * INV_LUT_COUNT;
        
        const float3 col0 = tex2D(s, float2(u0, v_atlas)).rgb;
        const float3 col1 = tex2D(s, float2(u1, v_atlas)).rgb;

        return lerp(col0, col1, sliceFrac);
    }

    float3 PS_FHD(VSOUT i) : SV_Target
    {
        const float3 color = tex2D(Color, i.texcoord).rgb;
        const float3 lut_output = ApplyLUT(SamplerLUT, color, Preset);
        const float3 final_color = lerp(color, lut_output, Strength);
        
        return ApplyDither(final_color, i.magicDot);
    }

    technique uFakeHDR
    {
        pass
        {
            VertexShader = VS_FHD;
            PixelShader = PS_FHD;
        }
    }
}
