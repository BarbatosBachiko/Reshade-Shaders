/*----------------------------------------------|
| ::                NeoSSAO                  :: |
|-----------------------------------------------|
| Version: 2.1                                  |
| Author: Barbatos                              |
| License: MIT                                  |
|----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Intensity";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 1.0;

uniform float Thickness <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Thickness";
    ui_tooltip = "Assumed thickness of objects. Allows light to pass behind thin surfaces.";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.01;
> = 0.5;

uniform float AORadius <
    ui_type = "drag";
    ui_category = "Basic Settings";
    ui_label = "Radius";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
> = 1.0;

uniform float RenderScale <
    ui_type = "drag";
    ui_category = "Performance";
    ui_label = "Render Resolution";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform int RaySteps <
    ui_type = "slider";
    ui_category = "Performance";
    ui_label = "Ray Steps";
    ui_min = 3;
    ui_max = 32;
> = 8;

uniform int SampleCount <
    ui_type = "slider";
    ui_category = "Performance";
    ui_label = "Slice Count";
    ui_min = 1; ui_max = 16;
> = 4;

uniform float FadeStart <
    ui_type = "slider";
    ui_category = "Fade Settings";
    ui_label = "Fade Start";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float FadeEnd <
    ui_type = "slider";
    ui_category = "Fade Settings";
    ui_label = "Fade End";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform float BlurSharpness <
    ui_type = "drag";
    ui_category = "Denoising";
    ui_label = "Blur Sharpness";
    ui_tooltip = "Higher values preserve more edges but reduce blur strength.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

uniform bool EnableDepthMultiplier <
    ui_type = "checkbox";
    ui_category = "Depth & Normals";
    ui_label = "Enable Depth Multiplier";
> = false;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_category = "Depth & Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform float DepthThreshold <
    ui_type = "slider";
    ui_category = "Depth & Normals";
    ui_label = "Sky Threshold";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.999;

uniform float FOV <
    ui_type = "slider";
    ui_category = "Depth & Normals";
    ui_label = "Field of View (Vertical)";
    ui_min = 1.0; ui_max = 120.0;
> = 75.0;

uniform float4 OcclusionColor <
    ui_type = "color";
    ui_category = "Debug & Style";
    ui_label = "Occlusion Color";
> = float4(0.0, 0.0, 0.0, 1.0);

uniform int ViewMode < 
    ui_type = "combo";
    ui_category = "Debug & Style";
    ui_label = "View Mode";
    ui_items = "None\0AO Only\0Depth\0Normals\0";
> = 0;

// Defines
#define PI 3.1415926535
#define HALF_PI 1.57079632679
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define SECTOR_COUNT 32 
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
static const int BlurRadius = 2;
//----------------|
// :: Textures :: |
//----------------|

namespace Barbatos_NeoSSAO2
{
    texture2D normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = normalTex;
    };

    texture2D AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R8;
    };
    sampler2D sAO
    {
        Texture = AO;
    };
    
    texture2D AOBlur
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = R8;
    };
    sampler2D sAOBlur
    {
        Texture = AOBlur;
    };

    //-------------|
    // :: Utility::|
    //-------------|

    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };

    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }
    
    void VS_NeoSSAO(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ? 2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        float fov_rad = FOV * (PI / 180.0);
        float y = tan(fov_rad * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }

    float3 UVToViewPos(float2 uv, float view_z, float2 pScale, float depthMult)
    {
        float2 ndc = uv * 2.0 - 1.0;
        return float3(ndc.x * pScale.x, -ndc.y * pScale.y, 1.0) * (view_z * depthMult);
    }

    float4 CalculateEdges(const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ)
    {
        float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
        float4 edgesLRTBSlopeAdjusted = edgesLRTB + edgesLRTB.yxwz;
        edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
        return saturate((1.3 - edgesLRTB / (centerZ * 0.040)));
    }

    float3 CalculateNormal(const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos)
    {
        float4 acceptedNormals = float4(edgesLRTB.x * edgesLRTB.z, edgesLRTB.z * edgesLRTB.y, edgesLRTB.y * edgesLRTB.w, edgesLRTB.w * edgesLRTB.x);
        pixLPos = normalize(pixLPos - pixCenterPos);
        pixRPos = normalize(pixRPos - pixCenterPos);
        pixTPos = normalize(pixTPos - pixCenterPos);
        pixBPos = normalize(pixBPos - pixCenterPos);
        float3 pixelNormal = float3(0, 0, -0.0005);
        pixelNormal += (acceptedNormals.x) * cross(pixLPos, pixTPos);
        pixelNormal += (acceptedNormals.y) * cross(pixTPos, pixRPos);
        pixelNormal += (acceptedNormals.z) * cross(pixRPos, pixBPos);
        pixelNormal += (acceptedNormals.w) * cross(pixBPos, pixLPos);
        return normalize(pixelNormal);
    }

    float3 getNormalFromTex(float2 coords)
    {
        return tex2Dlod(sNormal, float4(coords, 0, 0)).xyz;
    }

    float GetBayer8x8(float2 uv)
    {
        int2 pixelPos = int2(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        static const int bayer[64] =
        {
            0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
            12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
      
            15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21
        };
        return float(bayer[(pixelPos.x % 8) + (pixelPos.y % 8) * 8]) * (1.0 / 64.0);
    }

    float GTAOFastAcos(float x)
    {
        float res = -0.156583 * abs(x) + HALF_PI;
        res *= sqrt(1.0 - abs(x));
        return x >= 0 ? res : PI - res;
    }
    
    float2 GTAOFastAcos(float2 x)
    {
        float2 res = -0.156583 * abs(x) + HALF_PI;
        res *= sqrt(1.0 - abs(x));
        return x >= 0 ? res : PI - res;
    }

    uint UpdateSectors(float minHorizon, float maxHorizon, uint globalOccludedBitfield)
    {
        uint startHorizonInt = minHorizon * SECTOR_COUNT;
        uint angleHorizonInt = ceil((maxHorizon - minHorizon) * SECTOR_COUNT);
        uint angleHorizonBitfield = angleHorizonInt > 0 ?
            (0xFFFFFFFF >> (SECTOR_COUNT - angleHorizonInt)) : 0;
        uint currentOccludedBitfield = angleHorizonBitfield << startHorizonInt;
        return globalOccludedBitfield | currentOccludedBitfield;
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    float4 PS_GenNormals(VS_OUTPUT input) : SV_Target
    {
        float depth = GetDepth(input.uv);
        if (depth >= DepthThreshold)
            return float4(0, 0, 1, 1);
        float realDepthMult = EnableDepthMultiplier ? lerp(0.1, 5.0, DepthMultiplier) : 1.0;
        float2 p = ReShade::PixelSize;
        float3 p_c = UVToViewPos(input.uv, depth * FAR_PLANE, input.pScale, realDepthMult);

        float2 uvL = input.uv - float2(p.x, 0);
        float2 uvR = input.uv + float2(p.x, 0);
        float2 uvT = input.uv - float2(0, p.y);
        float2 uvB = input.uv + float2(0, p.y);

        float depthL = GetDepth(uvL);
        float depthR = GetDepth(uvR);
        float depthT = GetDepth(uvT);
        float depthB = GetDepth(uvB);

        float3 p_l = UVToViewPos(uvL, depthL * FAR_PLANE, input.pScale, realDepthMult);
        float3 p_r = UVToViewPos(uvR, depthR * FAR_PLANE, input.pScale, realDepthMult);
        float3 p_t = UVToViewPos(uvT, depthT * FAR_PLANE, input.pScale, realDepthMult);
        float3 p_b = UVToViewPos(uvB, depthB * FAR_PLANE, input.pScale, realDepthMult);

        float4 edges = CalculateEdges(p_c.z, p_l.z, p_r.z, p_t.z, p_b.z);
        float3 normal = CalculateNormal(edges, p_c, p_l, p_r, p_t, p_b);

        return float4(normal, 1.0);
    }

    float4 PS_SSAO(VS_OUTPUT input) : SV_Target
    {
        float2 scaled_uv = input.uv / RenderScale;
        if (any(scaled_uv > 1.0))
            discard;

        float center_depth = GetDepth(scaled_uv);
        if (center_depth >= DepthThreshold)
            return 1.0;
        float realDepthMult = EnableDepthMultiplier ? lerp(0.1, 5.0, DepthMultiplier) : 1.0;
        float2 invPScale = 1.0 / input.pScale;
        float3 positionVS = UVToViewPos(scaled_uv, center_depth * FAR_PLANE, input.pScale, realDepthMult);
        float3 normalVS = getNormalFromTex(scaled_uv);
        
        positionVS += normalVS * (0.005 * realDepthMult);
        float3 V = normalize(-positionVS);

        float random_val = GetBayer8x8(input.uv);
        float stepDist = (max(AORadius, 0.01)) / float(RaySteps);
        
        float totalVisibility = 0.0;
        [loop]
        for (int i = 0; i < SampleCount; i++)
        {
            float angle = (float(i) + random_val) * (PI / float(SampleCount));
            float2 dir = float2(cos(angle), sin(angle));
            
            float3 sliceN = normalize(cross(float3(dir, 0.0), V));
            float3 projN = normalVS - sliceN * dot(normalVS, sliceN);
            float projNLen = length(projN);
            float cosN = dot(projN / (projNLen + 1e-6), V);
            float3 T = cross(V, sliceN);
            float N_angle = -sign(dot(projN, T)) * GTAOFastAcos(cosN);
            
            uint globalOccludedBitfield = 0;
            [unroll]
            for (int side = -1; side <= 1; side += 2)
            {
                float currentDist = stepDist * (random_val + 0.1);
                float2 rayDir = dir * float(side);
                
                [loop]
                for (int j = 0; j < RaySteps; j++)
                {
                    float3 samplePosRay = positionVS + (float3(rayDir, 0) * currentDist);
                    float invZ = 1.0 / (samplePosRay.z + 1e-6);
                    float2 projected = samplePosRay.xy * invZ * invPScale;
                    float2 sample_uv = projected * float2(0.5, -0.5) + 0.5;

                    if (all(saturate(sample_uv) == sample_uv))
                    {
                        float sampleDepth = GetDepth(sample_uv);
                        float3 samplePosVS = UVToViewPos(sample_uv, sampleDepth * FAR_PLANE, input.pScale, realDepthMult);
                        
                        float3 deltaPos = samplePosVS - positionVS;
                        float3 deltaPosBackface = deltaPos - V * Thickness;

                        float2 frontBackHorizon = float2(dot(normalize(deltaPos), V), dot(normalize(deltaPosBackface), V));
                        frontBackHorizon = GTAOFastAcos(frontBackHorizon);
                        float2 horizonAngles = (float(side) * -frontBackHorizon - N_angle + HALF_PI) / PI;
                        horizonAngles = saturate(horizonAngles);

                        float minH = min(horizonAngles.x, horizonAngles.y);
                        float maxH = max(horizonAngles.x, horizonAngles.y);
                        
                        globalOccludedBitfield = UpdateSectors(minH, maxH, globalOccludedBitfield);
                    }
                    currentDist += stepDist;
                }
            }
            
            float occludedCount = countbits(globalOccludedBitfield);
            totalVisibility += 1.0 - (occludedCount / float(SECTOR_COUNT));
        }

        float visibility = totalVisibility / float(SampleCount);
        float occlusion = 1.0 - visibility;
        occlusion = pow(saturate(occlusion * Intensity), 2.0);
        
        float fade = smoothstep(FadeEnd * 2.0, FadeStart, center_depth);
        occlusion *= fade;

        return 1.0 - saturate(occlusion);
    }
    
    float4 PS_BilateralBlur(VS_OUTPUT input) : SV_Target
    {
        float2 scaled_uv = input.uv / RenderScale;
        if (any(scaled_uv > 1.0))
            discard;

        float2 texelSize = ReShade::PixelSize * 1.5; 
        float centerDepth = GetDepth(scaled_uv);
        
        float totalWeight = 1.0;
        float totalAO = tex2D(sAO, input.uv).r; 

        const int2 offsets[14] =
        {
            int2(0, 1), int2(0, -1), 
            int2(0, 2), int2(0, -2),
            int2(1, 0), int2(-1, 0),
            int2(1, 1), int2(1, -1), int2(-1, 1), int2(-1, -1), 
            int2(1, 2), int2(1, -2), int2(-1, 2), int2(-1, -2) 
        };

        const float spatialWeights[14] =
        {
            0.8825, 0.8825, // w1
            0.6065, 0.6065, // w3
            0.8825, 0.8825, // w1
            0.7788, 0.7788, 0.7788, 0.7788, // w2
            0.5353, 0.5353, 0.5353, 0.5353 // w4
        };

        float sharpnessMult = 1000.0 * BlurSharpness;

        [unroll]
        for (int k = 0; k < 14; k++)
        {
            float2 sampleUV = input.uv + (float2(offsets[k]) * texelSize);
            float sampleDepth = GetDepth(sampleUV / RenderScale);
            float sampleAO = tex2D(sAO, sampleUV).r;

            float depthDiff = abs(centerDepth - sampleDepth);
            float weight = spatialWeights[k] * exp(-depthDiff * sharpnessMult);

            totalAO += sampleAO * weight;
            totalWeight += weight;
        }
        
        return totalAO / totalWeight;
    }

    float4 PS_Output(VS_OUTPUT input) : SV_Target
    {
        float4 originalColor = GetColor(input.uv);
        float occlusion = tex2D(sAOBlur, input.uv * RenderScale).r;

        if (ViewMode == 0) // Normal
        {
            originalColor.rgb *= occlusion;
            originalColor.rgb = lerp(originalColor.rgb, OcclusionColor.rgb, 1.0 - occlusion);
            return originalColor;
        }
        else if (ViewMode == 1) // AO Only
            return occlusion;
        else if (ViewMode == 2) // Depth
            return GetDepth(input.uv);
        else if (ViewMode == 3) // Normals
            return float4(getNormalFromTex(input.uv) * 0.5 + 0.5, 1.0);
        return originalColor;
    }

    technique NeoSSAO
    {
        pass GenNormals
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_GenNormals;
            RenderTarget = normalTex;
        }
        pass SSAOPass
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_SSAO;
            RenderTarget = AO;
        }
        pass DenoisePass
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_BilateralBlur;
            RenderTarget = AOBlur;
        }
        pass OutputPass
        {
            VertexShader = VS_NeoSSAO;
            PixelShader = PS_Output;
        }
    }
}
