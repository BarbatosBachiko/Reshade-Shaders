/*------------------.
| :: Description :: |
'-------------------/

remove the last character to use the shader, it is currently not compiling in Directx9 and I still haven't figured out why

    SSRT

    Version 1.1
    Author: Barbatos Bachiko
    Original: jebbyk
    License: GNU Affero General Public License v3.0
    
    About: Open source screen space ray tracing shader for reshade.

    History:
    (*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility

    Version 1.1
    * Implemented new denoising algorithm
    + Pespective Improvement 
     
*/

#include "ReShade.fxh"

    /*-------------------.
    | :: Settings ::    |
    '-------------------*/

uniform int ViewMode <
    ui_type = "combo";
    ui_category = "General";
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
> = 1;

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
> = 10.0;

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
> = 0.0;

uniform float DEGHOSTING_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 0.1;
    ui_step = 0.001;
    ui_label = "Deghosting threshold";
    ui_tooltip = "Smaller number decreases ghosting caused by temporal gi blending but increases noise during movement";
    ui_category = "Filtering";
> = 0.002;

uniform float PERSPECTIVE_COEFFICIENT <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Perspective coefficient";
    ui_tooltip = "Controls perspective projection";
    ui_category = "Experimental";
> = 2.0;

uniform float TONE <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Tone";
    ui_tooltip = "Controls tone mapping intensity";
    ui_category = "Experimental";
> = 0.3;

uniform float BLURING_AMOUNT <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 8.0;
    ui_step = 0.1;
    ui_label = "Bluring amount";
    ui_tooltip = "Less noise but less details";
    ui_category = "Denoising";
> = 1.0;

uniform float EDGE_STOPPING_NORMAL <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Normal Edge Stopping";
    ui_tooltip = "Controls how much normals affect edge detection in denoising";
    ui_category = "Denoising";
> = 0.5;

uniform float EDGE_STOPPING_DEPTH <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Depth Edge Stopping";
    ui_tooltip = "Controls how much depth affects edge detection in denoising";
    ui_category = "Denoising";
> = 0.5;

uniform int DENOISE_ITERATIONS <
    ui_type = "drag";
    ui_min = 1; ui_max = 5;
    ui_step = 1;
    ui_label = "Denoise Iterations";
    ui_tooltip = "Number of denoising passes to apply";
    ui_category = "Denoising";
> = 3;

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
    Format = RGBA16F; 
};
sampler blurTexture0
{
    Texture = fBlurTexture0;
};

texture fBlurTexture1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F; 
};
sampler blurTexture1
{
    Texture = fBlurTexture1;
};

texture fBlurTexture2
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler blurTexture2
{
    Texture = fBlurTexture2;
};


texture fEdgeTexture
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler edgeTexture
{
    Texture = fEdgeTexture;
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

    float3 normal = normalize(cross(vertCenter - vertNorth, vertCenter - vertEast));
    
    if (any(isnan(normal)) || any(isinf(normal)))
    {
        normal = float3(0, 0, 1);
    }
    
    return normal;
}

float nrand(float2 uv)
{
    return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float3 rand3d(float2 uv)
{
    return tex2Dlod(noiseTexture, float4(uv.xy, 0.0, 0.0)).rgb;
}

float2 getPixelSize()
{
    return float2(1.0 / BUFFER_WIDTH, 1.0 / BUFFER_HEIGHT);
}

float3 uvz_to_xyz(float2 uv, float z)
{
    uv -= float2(0.5, 0.5);
    return float3(uv.x * z * PERSPECTIVE_COEFFICIENT, uv.y * z * PERSPECTIVE_COEFFICIENT, z);
}

float2 xyz_to_uv(float3 pos)
{
    float2 uv = float2(pos.x / (pos.z * PERSPECTIVE_COEFFICIENT), pos.y / (pos.z * PERSPECTIVE_COEFFICIENT));
    return uv + float2(0.5, 0.5);
}

float calculateLuminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

float4 Trace(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    float perspectiveCoeff = PERSPECTIVE_COEFFICIENT;
    float depth = GetLinearizedDepth(texcoord).x;
    float3 normal = GetScreenSpaceNormal(texcoord);
    float2 centredTexCoord = texcoord - float2(0.5, 0.5);
    
    float4 accumulatedColor = float4(0.0, 0.0, 0.0, 0.0);
    const float invRays = 1.0 / (RAYS_AMOUNT);
    
    float4 oldGI = tex2Dlod(giTexture1, float4(texcoord.xy, 0.0, 0.0));
    
    float3 selfPosition = float3(centredTexCoord.x * depth * perspectiveCoeff,
                                 centredTexCoord.y * depth * perspectiveCoeff,
                                 depth);
    
    [loop]
    for (int j = 0; j < RAYS_AMOUNT; j++)
    {
        float j1 = j + 1.0;
        float3 currentPosition = selfPosition;
        
        float3 rand = normalize(rand3d(texcoord * (32.0 * j1) + frac(FRAME_COUNT / 4.0)) * 2.0 - 1.0);
        float3 rayDir = normalize(reflect(normalize(currentPosition), normal + rand * 0.1));
        
        float stepSize = 0.01 * BASE_RAYS_LENGTH / STEPS_PER_RAY;
        float3 step = rayDir * stepSize;
        
        bool hitFound = false;
        
        [loop]
        for (int i = 0; i < STEPS_PER_RAY; i++)
        {
            currentPosition += step;
            float newZ = currentPosition.z;
            
            // Skip calculations for positions outside screen space
            if (newZ <= 0.0)
                continue;
            
            float2 newTexCoord = float2(currentPosition.x / (newZ * perspectiveCoeff),
                                      currentPosition.y / (newZ * perspectiveCoeff));
            float2 newTexCoordCentred = newTexCoord + float2(0.5, 0.5);
            
            // Better boundary check with early exit
            if (any(newTexCoordCentred < 0.0) || any(newTexCoordCentred > 1.0))
            {
                continue;
            }
            
            float newDepth = GetLinearizedDepth(newTexCoordCentred).x;
            
            // Skip if current depth is too far or too close
            if (newZ <= newDepth || newZ >= newDepth + DEPTH_THRESHOLD)
            {
                continue;
            }
            
            float3 newNormal = GetScreenSpaceNormal(newTexCoordCentred);
            float normalAlignment = dot(newNormal, rayDir);
            
            // Skip if normal is facing wrong direction
            if (normalAlignment <= NORMAL_THRESHOLD)
            {
                continue;
            }
            
            float3 photon = tex2Dlod(ReShade::BackBuffer, float4(newTexCoordCentred.xy, 0.0, 0.0)).rgb;
            float correctionFactor = max(0.2, abs(normalAlignment));
            
            photon = photon * photon; 
            accumulatedColor.rgb += photon * invRays * correctionFactor;
            
            hitFound = true;
            break;
        }
        
        if (!hitFound && j == 0)
        {
            float3 skyColor = float3(0.1, 0.12, 0.15);
            accumulatedColor.rgb += skyColor * 0.1 * invRays;
        }
    }
    
    accumulatedColor.a = depth;
    
    accumulatedColor.rgb = min(accumulatedColor.rgb, 10.0);
    
    if (depth < oldGI.w + DEGHOSTING_THRESHOLD && depth > oldGI.w - DEGHOSTING_THRESHOLD)
    {
        accumulatedColor.rgb = lerp(accumulatedColor.rgb, oldGI.rgb, TEMPORAL_FACTOR);
    }
    
    return accumulatedColor;
}

float4 StoreEdges(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float depth = GetLinearizedDepth(texcoord);
    float3 normal = GetScreenSpaceNormal(texcoord);
    return float4(normal, depth);
}

float4 Denoise(float4 position : SV_Position, float2 texcoord : TEXCOORD, sampler sourceSampler, float stepSize) : SV_Target
{
    const float2 pixelSize = ReShade::PixelSize;
    const int radius = 2;
    
    float4 centerColor = tex2Dlod(sourceSampler, float4(texcoord, 0.0, 0.0));
    float4 centerEdge = tex2Dlod(edgeTexture, float4(texcoord, 0.0, 0.0));
    float centerDepth = centerEdge.a;
    float3 centerNormal = centerEdge.rgb;
    
    float4 result = float4(0, 0, 0, 0);
    float totalWeight = 0;
    
    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            float2 offset = float2(x, y) * stepSize * pixelSize;
            float2 sampleCoord = texcoord + offset;
            
            float4 sampleColor = tex2Dlod(sourceSampler, float4(sampleCoord, 0.0, 0.0));
            float4 sampleEdge = tex2Dlod(edgeTexture, float4(sampleCoord, 0.0, 0.0));
            
            // Edge stopping functions based on depth and normal
            float depthWeight = exp(-abs(centerDepth - sampleEdge.a) / (EDGE_STOPPING_DEPTH * 0.1 + 0.0001));
            float normalWeight = pow(max(0, dot(centerNormal, sampleEdge.rgb)), EDGE_STOPPING_NORMAL * 8.0 + 1.0);
            float spatialWeight = exp(-(x * x + y * y) / (radius + 0.001));
            
            float weight = depthWeight * normalWeight * spatialWeight;
            
            result += sampleColor * weight;
            totalWeight += weight;
        }
    }
    
    result /= max(totalWeight, 0.0001);
    result.a = centerColor.a;
    
    return result;
}

float4 Denoise0(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return Denoise(position, texcoord, giTexture0, 1.0 * BLURING_AMOUNT);
}

float4 Denoise1(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return Denoise(position, texcoord, blurTexture0, 2.0 * BLURING_AMOUNT);
}

float4 Denoise2(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return Denoise(position, texcoord, blurTexture1, 4.0 * BLURING_AMOUNT);
}

float4 Downsample0(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float kernelWeights[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    const float stepSize = 1.0 * BLURING_AMOUNT;
    const float2 pixelSize = ReShade::PixelSize;
    
    float4 color = 0;
    
    [unroll]
    for (int x = -2; x <= 2; x++)
    {
        [unroll]
        for (int y = -2; y <= 2; y++)
        {
            const float2 offset = float2(x, y) * stepSize * pixelSize;
            const float weight = kernelWeights[x + 2] * kernelWeights[y + 2];
            
            color += tex2Dlod(giTexture0, float4(texcoord + offset, 0.0, 0.0)) * weight;
        }
    }
    
    return color;
}

float4 Downsample1(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float kernelWeights[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    const float stepSize = 2.0 * BLURING_AMOUNT;
    const float2 pixelSize = ReShade::PixelSize;
    
    float4 color = 0;
    
    [unroll]
    for (int x = -2; x <= 2; x++)
    {
        [unroll]
        for (int y = -2; y <= 2; y++)
        {
            const float2 offset = float2(x, y) * stepSize * pixelSize;
            const float weight = kernelWeights[x + 2] * kernelWeights[y + 2];
            color += tex2Dlod(blurTexture0, float4(texcoord + offset, 0.0, 0.0)) * weight;
        }
    }
    
    return color;
}

float4 Downsample2(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float kernelWeights[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    const float stepSize = 4.0 * BLURING_AMOUNT;
    const float2 pixelSize = ReShade::PixelSize;
    
    float4 color = 0;
    
    [unroll]
    for (int x = -2; x <= 2; x++)
    {
        [unroll]
        for (int y = -2; y <= 2; y++)
        {
            const float2 offset = float2(x, y) * stepSize * pixelSize;
            const float weight = kernelWeights[x + 2] * kernelWeights[y + 2];
            color += tex2Dlod(blurTexture1, float4(texcoord + offset, 0.0, 0.0)) * weight;
        }
    }
    
    return color;
}

float4 Combine(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    if (ViewMode == 1)
    {
        float3 gi = tex2Dlod(blurTexture2, float4(texcoord.xy, 0.0, 0.0)).rgb * EFFECT_INTENSITY;
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
        float3 color = tex2Dlod(ReShade::BackBuffer, float4(texcoord.xy, 0.0, 0.0)).rgb;
        float3 gi = tex2Dlod(blurTexture2, float4(texcoord.xy, 0.0, 0.0)).rgb * EFFECT_INTENSITY;
        
        color = lerp(color, color + gi, TONE);
        
        return float4(color, 1.0);
    }
}

float4 SaveGI(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2Dlod(giTexture0, float4(texcoord, 0.0, 0.0));
}

/*------------------.
| :: Techniques :: |
'------------------*/

technique SSRT <
    ui_tooltip = "Open source screen space ray tracing";
>
{
    pass Trace
    {
        VertexShader = PostProcessVS;
        PixelShader = Trace;
        RenderTarget0 = fGiTexture0;
    }
    
    pass StoreEdges
    {
        VertexShader = PostProcessVS;
        PixelShader = StoreEdges;
        RenderTarget0 = fEdgeTexture;
    }

    pass Denoise0
    {
        VertexShader = PostProcessVS;
        PixelShader = Denoise0;
        RenderTarget0 = fBlurTexture0;
    }

    pass Denoise1
    {
        VertexShader = PostProcessVS;
        PixelShader = Denoise1;
        RenderTarget0 = fBlurTexture1;
    }

    pass Denoise2
    {
        VertexShader = PostProcessVS;
        PixelShader = Denoise2;
        RenderTarget0 = fBlurTexture2;
    }

    pass Combine
    {
        VertexShader = PostProcessVS;
        PixelShader = Combine;
    }

    pass SaveGI
    {
        VertexShader = PostProcessVS;
        PixelShader = SaveGI;
        RenderTarget0 = fGiTexture1;
    }
  }
} //<--- REMOVE
