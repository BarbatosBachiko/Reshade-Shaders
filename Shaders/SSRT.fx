/*------------------.
| :: Description :: |
'-------------------/

    SSRT

    Version 1.0
    Author: Barbatos Bachiko
    Original: jebbyk
    License: GNU Affero General Public License v3.0
    
    About: Open source screen space ray tracing shader for reshade.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.0
    x Trace accumulating too much light causing an explosion when the temporal factor was high
    * Debug Modes
     
*/

#include "ReShade.fxh"

    /*-------------------.
    | :: Settings ::    |
    '-------------------*/

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "Geral";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode";
    ui_items = "Combine\0GI Debug\0Normal Debug\0Depth Debug\0";
> = 0;

uniform float BASE_RAYS_LENGTH <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Base ray length";
    ui_tooltip = "Increases distance of light spreading, decreases intersections detection quality";
    ui_category = "Ray Tracing";
> = 2.5;

uniform int RAYS_AMOUNT <
    ui_type = "drag";
    ui_min = 1; ui_max = 256;
    ui_step = 1;
    ui_label = "Rays amount";
    ui_tooltip = "Decreases noise amount";
    ui_category = "Ray Tracing";
> = 4;

uniform int STEPS_PER_RAY <
    ui_type = "drag";
    ui_min = 1; ui_max = 256;
    ui_step = 1;
    ui_label = "Steps per ray";
    ui_tooltip = "Increases quality of intersections detection";
    ui_category = "Ray Tracing";
> = 32;

uniform float EFFECT_INTENSITY <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Effect intensity";
    ui_tooltip = "Power of effect";
    ui_category = "Ray Tracing";
> = 2.0;

uniform float DEPTH_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_label = "Depth Threshold";
    ui_tooltip = "Less accurate tracing but less noise at the same time";
    ui_category = "Ray Tracing";
> = 0.002;

uniform float NORMAL_THRESHOLD <
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_label = "Normal Threshold";
    ui_tooltip = "More accurate tracing but more noise at the same time";
    ui_category = "Ray Tracing";
> = 0.0;

uniform float TEMPORAL_FACTOR <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 0.9;
    ui_step = 0.1;
    ui_label = "Temporal factor";
    ui_tooltip = "Less noise but more ghosting";
    ui_category = "Filtering";
> = 0.9;

uniform float BLURING_AMOUNT <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 8.0;
    ui_step = 0.1;
    ui_label = "Bluring amount";
    ui_tooltip = "Less noise but less details";
    ui_category = "Filtering";
> = 1.0;

uniform float DEGHOSTING_TRESHOLD <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 0.1;
    ui_step = 0.001;
    ui_label = "Deghosting treshold";
    ui_tooltip = "Smaller number decreases ghosting caused by temporal gi blending but increases noise during movement";
    ui_category = "Filtering";
> = 0.002;

uniform float PERSPECTIVE_COEFFITIENT <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Perpective coeffitient";
    ui_tooltip = "Testing";
    ui_category = "Experimental";
> = 2.0;

uniform float TONE <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Tone";
    ui_tooltip = "Testing";
    ui_category = "Experimental";
> = 1.0;

uniform int FRAME_COUNT < source = "framecount"; >;

    /*---------------.
    | :: Textures :: |
    '---------------*/

texture fGiTexture0
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler giTexture0
{
    Texture = fGiTexture0;
};

texture fGiTexture1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler giTexture1
{
    Texture = fGiTexture1;
};

texture fBlurTexture0
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler blurTexture0
{
    Texture = fBlurTexture0;
};

texture fBlurTexture1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler blurTexture1
{
    Texture = fBlurTexture1;
};

texture fBlurTexture2
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA8;
};
sampler blurTexture2
{
    Texture = fBlurTexture2;
};

texture fNoiseTexture < source = "SSRT_bluenoise.png"; >
{
    Width = 32;
    Height = 32;
    Format = RGBA8;
};
sampler noiseTexture
{
    Texture = fNoiseTexture;
    AddressU = WRAP;
    AddressV = WRAP;
};

    /*----------------.
    | :: Functions :: |
    '----------------*/

// Depth
float GetLinearizedDepth(float2 texcoord)
{
    return ReShade::GetLinearizedDepth(texcoord);
}

// Normal From DisplayDepth.fx
float3 GetScreenSpaceNormal(float2 texcoord)
{
    float3 offset = float3(BUFFER_PIXEL_SIZE, 0.0);
    float2 posCenter = texcoord.xy;
    float2 posNorth = posCenter - offset.zy;
    float2 posEast = posCenter + offset.xz;

    float3 vertCenter = float3(posCenter - 0.5, 1) * GetLinearizedDepth(posCenter);
    float3 vertNorth = float3(posNorth - 0.5, 1) * GetLinearizedDepth(posNorth);
    float3 vertEast = float3(posEast - 0.5, 1) * GetLinearizedDepth(posEast);

    return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast));
}

float nrand(float2 uv)
{
    return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float3 rand3d(float2 uv)
{
    return tex2D(noiseTexture, uv.xy).rgb;
}

float2 getPixelSize()
{
    return float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
}

float3 uvz_to_xyz(float2 uv, float z)
{
    uv -= float2(0.5, 0.5);
    return float3(uv.x * z * PERSPECTIVE_COEFFITIENT, uv.y * z * PERSPECTIVE_COEFFITIENT, z);
}

float2 xyz_to_uv(float3 pos)
{
    float2 uv = float2(pos.x / (pos.z * PERSPECTIVE_COEFFITIENT), pos.y / (pos.z * PERSPECTIVE_COEFFITIENT));
    return uv + float2(0.5, 0.5);
}

// Ray Tracing
float4 Trace(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    float perspectiveCoeff = PERSPECTIVE_COEFFITIENT;
    float depth = GetLinearizedDepth(texcoord).x;
    float3 normal = GetScreenSpaceNormal(texcoord);
    float2 centredTexCoord = texcoord - float2(0.5, 0.5);
    
    float4 accumulatedColor = float4(0.0, 0.0, 0.0, 0.0);
    const int RAYS_AMOUNT = 4;
    const float invRays = 1.0 / (RAYS_AMOUNT + 1);
    
    float4 oldGI = tex2Dlod(giTexture1, float4(texcoord.xy, 0.0, 0.0));
    oldGI.xyz *= TEMPORAL_FACTOR;
    
    [unroll]
    for (int j = 0; j < RAYS_AMOUNT; j++)
    {
        float j1 = j + 1.0;
        float3 selfPosition = float3(centredTexCoord.x * depth * perspectiveCoeff,
                                     centredTexCoord.y * depth * perspectiveCoeff,
                                     depth);
        
        float3 rand = normalize(rand3d(texcoord * (32.0 * j1) + frac(FRAME_COUNT / 4.0)) - float3(0.5, 0.5, 0.5));
        float3 rayDir = normalize(-normal - rand);
        float3 step = rayDir * (0.01 * BASE_RAYS_LENGTH / STEPS_PER_RAY);
        
        [loop]
        for (int i = 0; i < STEPS_PER_RAY; i++)
        {
            selfPosition += step;
            float newZ = selfPosition.z;
            float2 newTexCoord = float2(selfPosition.x / (newZ * perspectiveCoeff),
                                        selfPosition.y / (newZ * perspectiveCoeff));
            float2 newTexCoordCentred = newTexCoord + float2(0.5, 0.5);
            
            if (newTexCoordCentred.x < 0.0 || newTexCoordCentred.x > 1.0 ||
                newTexCoordCentred.y < 0.0 || newTexCoordCentred.y > 1.0)
            {
                continue;
            }
            
            float newDepth = GetLinearizedDepth(newTexCoordCentred).x;
            if (newZ <= newDepth || newZ >= newDepth + DEPTH_THRESHOLD)
            {
                continue;
            }
            
            float3 newNormal = GetScreenSpaceNormal(newTexCoordCentred);
            if (dot(newNormal, rayDir) <= NORMAL_THRESHOLD)
            {
                continue;
            }
            
            float3 photon = tex2Dlod(ReShade::BackBuffer, float4(newTexCoordCentred.xy, 0.0, 0.0)).rgb;
            photon = photon * photon * photon;
            accumulatedColor += float4(photon * invRays, 0.0);
            
            break;
        }
    }
    
    if (depth < oldGI.w + DEGHOSTING_TRESHOLD * 0.02 && depth > oldGI.w - DEGHOSTING_TRESHOLD * 0.02)
    {
        accumulatedColor = accumulatedColor * (1.0 - TEMPORAL_FACTOR) + float4(oldGI.rgb, 0.0);
    }
    accumulatedColor.w = depth;
    
    return accumulatedColor;
}

// Final Image
float4 Combine(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    if (ViewMode == 1)
    {
        float3 gi = tex2D(blurTexture2, texcoord.xy).rgb * EFFECT_INTENSITY;
        return float4(gi, 1.0);
    }
    else if (ViewMode == 2)
    {
        float3 normal = GetScreenSpaceNormal(texcoord);
        normal = normal * 0.5 + 0.5;
        return float4(normal, 1.0);
    }
    else if (ViewMode == 3)
    {
        float depth = GetLinearizedDepth(texcoord).x;
        return float4(depth, depth, depth, 1.0);
    }
    else
    {
        float depth = GetLinearizedDepth(texcoord).x;
        float3 color = tex2D(ReShade::BackBuffer, texcoord.xy).rgb;
        float giDepth = tex2D(blurTexture2, texcoord.xy).w;
        float3 gi = tex2D(blurTexture2, texcoord.xy).rgb * EFFECT_INTENSITY;
        gi = (color * gi * 0.9) + gi * 0.1;
        gi = gi / (gi + TONE);
        color += gi;
        return float4(color, 1.0);
    }
}

float4 SaveGI(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2D(giTexture0, texcoord);
}

// Downsampling Function - 0
float4 Downsample0(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 pixelSize = ReShade::PixelSize * BLURING_AMOUNT;
    float4 color = tex2D(giTexture0, texcoord + float2(-pixelSize.x, -pixelSize.y));
    color += tex2D(giTexture0, texcoord + float2(pixelSize.x, -pixelSize.y));
    color += tex2D(giTexture0, texcoord + float2(-pixelSize.x, pixelSize.y));
    color += tex2D(giTexture0, texcoord + float2(pixelSize.x, pixelSize.y));
    color *= 0.25;
    return color;
}

// Downsampling Function - 1
float4 Downsample1(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 pixelSize = ReShade::PixelSize * 2 * BLURING_AMOUNT;
    float4 color = tex2D(blurTexture0, texcoord + float2(-pixelSize.x, -pixelSize.y));
    color += tex2D(blurTexture0, texcoord + float2(pixelSize.x, -pixelSize.y));
    color += tex2D(blurTexture0, texcoord + float2(-pixelSize.x, pixelSize.y));
    color += tex2D(blurTexture0, texcoord + float2(pixelSize.x, pixelSize.y));
    color *= 0.25;
    return color;
}

// Downsampling Function - 2
float4 Downsample2(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 pixelSize = ReShade::PixelSize * 4 * BLURING_AMOUNT;
    float4 color = tex2D(blurTexture1, texcoord + float2(-pixelSize.x, -pixelSize.y));
    color += tex2D(blurTexture1, texcoord + float2(pixelSize.x, -pixelSize.y));
    color += tex2D(blurTexture1, texcoord + float2(-pixelSize.x, pixelSize.y));
    color += tex2D(blurTexture1, texcoord + float2(pixelSize.x, pixelSize.y));
    color *= 0.25;
    return color;
}

technique SSRT <
    ui_tooltip = "Open source screen space ray tracing shader for reshade.\n";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Trace;
        RenderTarget0 = fGiTexture0;
    }

    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Downsample0;
        RenderTarget0 = fBlurTexture0;
    }

    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Downsample1;
        RenderTarget0 = fBlurTexture1;
    }

    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Downsample2;
        RenderTarget0 = fBlurTexture2;
    }

    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = Combine;
    }

    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = SaveGI;
        RenderTarget0 = fGiTexture1;
    }
}
