/*------------------.
| :: Description :: |
'-------------------/

    GBloom
    Version 1.1
    Author: Barbatos Bachiko
    License: MIT

    About: Simple Bloom Effect

    History:
    (*) Feature   (+) Improvement   (x) Bugfix   (-) Information   (!) Compatibility

    Version 1.1
    * Implemented downsampling to half and quarter resolution.
    * Replaced the 5-tap Gaussian blur (RGBA16) with an optimized 3-tap blur (RGBA8).
    * Added temporal filter.
    * Integration of blend modes (Additive, Multiplicative, Alpha Blend) into the composition.
*/
    #include "ReShade.fxh"

    /*---------------.
    | :: Settings :: |
    '---------------*/
    uniform float BloomIntensity <
        ui_type = "slider";
        ui_label = "Bloom Intensity";
        ui_tooltip = "Strength of the bloom effect";
        ui_min = 0.0;
        ui_max = 5.0;
        ui_step = 0.01;
        ui_default = 1.5;
    > = 1.0;

    uniform float LuminanceThreshold <
        ui_type = "slider";
        ui_label = "Luminance Threshold";
        ui_tooltip = "Minimum brightness to generate bloom";
        ui_min = 0.0;
        ui_max = 2.0;
        ui_step = 0.01;
        ui_default = 0.75;
    > = 0.60;

    uniform float BlurRadius <
        ui_type = "slider";
        ui_label = "Blur Radius";
        ui_tooltip = "Size of the blur effect";
        ui_min = 0.5;
        ui_max = 10.0;
        ui_step = 0.1;
        ui_default = 3.0;
    > = 2.0;

    uniform bool EnableTemporal <
        ui_category = "Temporal";
        ui_type = "checkbox";
        ui_label = "Temporal Filtering";
        ui_tooltip = "Enable temporal filtering for bloom effect";
    > = true;

    uniform float TemporalFilterStrength <
        ui_category = "Temporal";
        ui_type = "slider";
        ui_label = "Temporal Strength";
        ui_min = 0.0;
        ui_max = 1.0;
        ui_step = 0.01;
        ui_tooltip = "Blend factor between current bloom and history";
    > = 0.25;

    uniform int BlendMode <
        ui_category = "Blend";
        ui_type = "combo";
        ui_label = "Blend Mode";
        ui_items = "Additive\0Multiplicative\0Alpha Blend\0";
    > = 1;

uniform int ViewMode <
        ui_category = "Blend";
        ui_type = "combo";
        ui_label = "View Mode";
        ui_items = "Normal\0Color\0";
    > = 0;

    /*---------------.
    | :: Textures :: |
    '---------------*/
    
    texture2D DownsampledTex
    {
        Width = BUFFER_WIDTH / 2;
        Height = BUFFER_HEIGHT / 2;
        Format = RGBA8;
    };

    texture2D FDTex
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RGBA8;
    };
    
    texture2D FDTempTex
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RGBA8;
    };

    texture2D BloomTemporal
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RGBA8;
    };

    texture2D BloomHistory
    {
        Width = BUFFER_WIDTH / 4;
        Height = BUFFER_HEIGHT / 4;
        Format = RGBA8;
    };

    sampler2D sDownsampled
    {
        Texture = DownsampledTex;
    };

    sampler2D sFD
    {
        Texture = FDTex;
    };

    sampler2D sFDTemp
    {
        Texture = FDTempTex;
    };

    sampler2D sBloomTemporal
    {
        Texture = BloomTemporal;
    };

    sampler2D sBloomHistory
    {
        Texture = BloomHistory;
    };

    // Kernel 3-tap
    static const float weight3_center = 0.5;
    static const float weight3_side = 0.25;

    /*----------------.
    | :: Functions :: |
    '----------------*/
    
    float4 BrightDownsample(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 fullTexel = float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
        float3 c0 = tex2D(ReShade::BackBuffer, uv).rgb;
        float3 c1 = tex2D(ReShade::BackBuffer, uv + float2(fullTexel.x, 0)).rgb;
        float3 c2 = tex2D(ReShade::BackBuffer, uv + float2(0, fullTexel.y)).rgb;
        float3 c3 = tex2D(ReShade::BackBuffer, uv + fullTexel).rgb;
        float3 avg = (c0 + c1 + c2 + c3) * 0.25;

        float luminance = dot(avg, float3(0.2126, 0.7152, 0.0722));
        float bright = smoothstep(LuminanceThreshold, LuminanceThreshold + 0.2, luminance);

        return float4(avg * bright, 1.0);
    }

    float4 DownsampleFurther(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 halfTexel = float2(1.0 / (BUFFER_WIDTH / 2), 1.0 / (BUFFER_HEIGHT / 2));
        float3 c0 = tex2D(sDownsampled, uv).rgb;
        float3 c1 = tex2D(sDownsampled, uv + float2(halfTexel.x, 0)).rgb;
        float3 c2 = tex2D(sDownsampled, uv + float2(0, halfTexel.y)).rgb;
        float3 c3 = tex2D(sDownsampled, uv + halfTexel).rgb;
        float3 avg = (c0 + c1 + c2 + c3) * 0.25;
        return float4(avg, 1.0);
    }

    float4 BlurHorizontal_FD(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 fdResolution = float2(BUFFER_WIDTH / 4, BUFFER_HEIGHT / 4);
        float2 texel = BlurRadius / fdResolution;
        
        float3 center = tex2D(sFD, uv).rgb;
        float3 right = tex2D(sFD, uv + float2(texel.x, 0)).rgb;
        float3 left = tex2D(sFD, uv - float2(texel.x, 0)).rgb;
        
        float3 result = center * weight3_center + (right + left) * weight3_side;
        return float4(result, 1.0);
    }

    float4 BlurVertical_FD(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float2 fdResolution = float2(BUFFER_WIDTH / 4, BUFFER_HEIGHT / 4);
        float2 texel = BlurRadius / fdResolution;
        
        float3 center = tex2D(sFDTemp, uv).rgb;
        float3 up = tex2D(sFDTemp, uv + float2(0, texel.y)).rgb;
        float3 down = tex2D(sFDTemp, uv - float2(0, texel.y)).rgb;
        
        float3 result = center * weight3_center + (up + down) * weight3_side;
        return float4(result, 1.0);
    }

    float4 PS_TemporalBloom(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 currentBloom = tex2D(sFD, uv).rgb;
        float3 historyBloom = tex2D(sBloomHistory, uv).rgb;
        float3 filteredBloom = EnableTemporal ? lerp(currentBloom, historyBloom, TemporalFilterStrength) : currentBloom;
        return float4(filteredBloom, 1.0);
    }

    float4 PS_SaveHistoryBloom(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 bloom = tex2D(sBloomTemporal, uv).rgb;
        return float4(bloom, 1.0);
    }
    
    float4 ABlendMode(float4 originalColor, float4 bloomColor, int viewMode, int blendMode)
    {
        if (viewMode == 0)
        {
            if (blendMode == 0) // Additive
                return float4(originalColor.rgb + bloomColor.rgb, originalColor.a);
            else if (blendMode == 1) // Multiplicative
                return float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - bloomColor.rgb), originalColor.a);
            else if (blendMode == 2) // Alpha Blend
            {
                float blendFactor = saturate(bloomColor.r);
                return float4(lerp(originalColor.rgb, bloomColor.rgb, blendFactor), originalColor.a);
            }
        }
        return originalColor;
    }

    float4 PS_CompositeBloom(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 bloom = EnableTemporal ? tex2D(sBloomTemporal, uv).rgb : tex2D(sFD, uv).rgb;
        float2 fdResolution = float2(BUFFER_WIDTH / 4, BUFFER_HEIGHT / 4);
        float2 texel = float2(1.0 / fdResolution.x, 1.0 / fdResolution.y);
        
        // Upsampling filter 3-tap
        float3 bloomCenter = bloom;
        float3 bloomRight = EnableTemporal ? tex2D(sBloomTemporal, uv + float2(texel.x, 0)).rgb : tex2D(sFD, uv + float2(texel.x, 0)).rgb;
        float3 bloomLeft = EnableTemporal ? tex2D(sBloomTemporal, uv - float2(texel.x, 0)).rgb : tex2D(sFD, uv - float2(texel.x, 0)).rgb;
        float3 bloomUpsampled = bloomCenter * weight3_center + (bloomRight + bloomLeft) * weight3_side;
    
        float3 scene = tex2D(ReShade::BackBuffer, uv).rgb;
        
        //Blend
        float4 finalColor = ABlendMode(float4(scene, 1.0), float4(bloomUpsampled * BloomIntensity, 1.0), ViewMode, BlendMode);
        return finalColor;
    }

    /*-----------------.
    | :: Techniques :: |
    '-----------------*/
    technique GBloom
    {
        pass BrightDownsample
        {
            VertexShader = PostProcessVS;
            PixelShader = BrightDownsample;
            RenderTarget = DownsampledTex;
        }
        
        pass DownsampleFurther
        {
            VertexShader = PostProcessVS;
            PixelShader = DownsampleFurther;
            RenderTarget = FDTex;
        }
        
        pass HorizontalBlur_FD
        {
            VertexShader = PostProcessVS;
            PixelShader = BlurHorizontal_FD;
            RenderTarget = FDTempTex;
        }
        
        pass VerticalBlur_FD
        {
            VertexShader = PostProcessVS;
            PixelShader = BlurVertical_FD;
            RenderTarget = FDTex;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TemporalBloom;
            RenderTarget = BloomTemporal;
            ClearRenderTargets = true;
        }
        pass SaveHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SaveHistoryBloom;
            RenderTarget = BloomHistory;
        }
        
        pass Composite
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_CompositeBloom;
        }
    }
