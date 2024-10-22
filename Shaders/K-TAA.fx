/*------------------.
| :: Description :: |
'-------------------/

    TAA Kuwahara Filter Shader (version 0.1)

    Author: BarbatosBachiko
    Originaly Ported to ReShade by Eideren

    License: Follow the original license from www.kyprianidis.com

    About:
    A combination of Temporal Anti-Aliasing with a Kuwahara filter

    Ideas for future improvement:
    * Optimize for better performance with high iteration counts
    * Add dynamic sample distance based on screen resolution or user preference
    * Investigate alternative blending methods for sharper TAA results
    
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 0.1
    * Initial texture preservation and TAA blending

*/

/*---------------.
| :: Includes :: |
'---------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

/*---------------.
| :: Settings :: |
'---------------*/

uniform int Iterations
<
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 16.0;
    ui_label = "Iterations";
>
= 4;

uniform float SampleDistance
<
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 10.0;
    ui_label = "SampleDistance";
>
= 4.0;

uniform float BlendFactor
<
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "Blend Factor";
>
= 0.5;

uniform bool EnableTexturePreservation
<
    ui_type = "checkbox";
    ui_label = "Enable Texture Preservation";
    ui_tooltip = "Preserve texture detail to avoid oily appearance.";
>
= true;

/*---------------.
| :: Textures :: |
'---------------*/

#define TexChannel0 (ReShade::BackBuffer)
#define TexChannel1 (ReShade::BackBuffer) // For TAA blending
#define TexChannel0_size (ReShade::ScreenSize.xy)

/*---------------.
| :: Samplers :: |
'---------------*/

#define SAMPLE(s, uv) tex2Dlod(s, float4(uv, 0, 0))

/*----------------.
| :: Functions :: |
'----------------*/

// Function to preserve texture detail
float3 PreserveTexture(float2 uv)
{
    float3 colorCenter = SAMPLE(TexChannel0, uv).rgb;
    float3 colorUp = SAMPLE(TexChannel0, uv + float2(0.0, 1.0) / TexChannel0_size).rgb;
    float3 colorDown = SAMPLE(TexChannel0, uv + float2(0.0, -1.0) / TexChannel0_size).rgb;
    float3 colorLeft = SAMPLE(TexChannel0, uv + float2(-1.0, 0.0) / TexChannel0_size).rgb;
    float3 colorRight = SAMPLE(TexChannel0, uv + float2(1.0, 0.0) / TexChannel0_size).rgb;

    float3 detail = (colorUp + colorDown + colorLeft + colorRight) * 0.25;
    return lerp(colorCenter, detail, 0.5);
}

float4 KuwaharaTaaPass(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
    float2 uv = tex.xy;
    float n = float((Iterations + 1) * (Iterations + 1));

    float3 m[4] = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };
    float3 s[4] = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };

    float scaledSampleDistance = SampleDistance / (Iterations + 1);

    for (int j = -Iterations; j <= 0; ++j)
    {
        for (int i = -Iterations; i <= 0; ++i)
        {
            float3 c = SAMPLE(TexChannel0, uv + float2(i, j) / TexChannel0_size * scaledSampleDistance).rgb;
            m[0] += c;
            s[0] += c * c;
        }
    }
    
    for (int j = -Iterations; j <= 0; ++j)
    {
        for (int i = 0; i <= Iterations; ++i)
        {
            float3 c = SAMPLE(TexChannel0, uv + float2(i, j) / TexChannel0_size * scaledSampleDistance).rgb;
            m[1] += c;
            s[1] += c * c;
        }
    }
    
    for (int j = 0; j <= Iterations; ++j)
    {
        for (int i = 0; i <= Iterations; ++i)
        {
            float3 c = SAMPLE(TexChannel0, uv + float2(i, j) / TexChannel0_size * scaledSampleDistance).rgb;
            m[2] += c;
            s[2] += c * c;
        }
    }
    
    for (int j = 0; j <= Iterations; ++j)
    {
        for (int i = -Iterations; i <= 0; ++i)
        {
            float3 c = SAMPLE(TexChannel0, uv + float2(i, j) / TexChannel0_size * scaledSampleDistance).rgb;
            m[3] += c;
            s[3] += c * c;
        }
    }

    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float min_sigma2 = 1e+2;

    for (int k = 0; k < 4; ++k)
    {
        m[k] /= n;
        s[k] = abs(s[k] / n - m[k] * m[k]);
        
        float sigma2 = s[k].r + s[k].g + s[k].b;

        if (sigma2 < min_sigma2)
        {
            min_sigma2 = sigma2;
            fragColor = float4(m[k], 1.0);
        }
    }

    float3 prevColor = SAMPLE(TexChannel1, uv).rgb;
    fragColor.rgb = lerp(prevColor, fragColor.rgb, BlendFactor);

    if (EnableTexturePreservation)
    {
        fragColor.rgb = PreserveTexture(uv);
    }

    return fragColor;
}

/*-----------------.
| :: Techniques :: |
'-----------------*/

technique K_TAA
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = KuwaharaTaaPass;
    }
}

/*-------------.
| :: Footer :: |
'--------------/

References:
* Anisotropic Kuwahara Filtering on the GPU - Kyprianidis et al.
* http://www.kyprianidis.com/p/pg2009/jkyprian-pg2009.pdf

*/
