/*------------------.
| :: Description :: |
'-------------------/

  ___ ___  ___ ___ 
 / __/ __|/ __|_ _|
 \__ \__ \ (_ || | 
 |___/___/\___|___|
                   
    Author: Barbatos Bachiko
    License: MIT

    About: Simulates indirect light and AO.

    History:
	(*) Feature (+) Improvement	(x) Bugfix (-) Information (!) Compatibility
    
    Version 2.0
    * Remake
*/
#include "ReShade.fxh"

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace ScreenSpaceGIJonson
{
/*---------------.
| :: Settings :: |
'---------------*/

    uniform int viewMode
    < 
        ui_type = "combo";
        ui_label = "View Mode";
        ui_items = "Normal\0GI Debug\0Depth\0Normals\0Specular\0";
        ui_category = "Visualization";
    > = 0;

    uniform float giIntensity <
        ui_type = "slider";
        ui_label = "GI Strength";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
        ui_category = "Global Illumination";
    > = 0.3;

    uniform float sampleRadius <
        ui_type = "slider";
        ui_label = "Sample Radius";
        ui_min = 0.1; ui_max = 5.0; ui_step = 0.01;
        ui_category = "Global Illumination";
    > = 1.0;

    uniform int numSamples <
        ui_type = "slider";
        ui_label = "Sample Count";
        ui_min = 4; ui_max = 64; ui_step = 2;
        ui_category = "Global Illumination";
    > = 8;


/*---------------.
| :: GI Settings :: |
'---------------*/

    uniform float diffuseIntensity
    <
        ui_type = "slider";
        ui_label = "Diffuse Intensity";
        ui_tooltip = "Adjust the intensity of diffuse reflections.";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
        ui_category = "GI Settings";
    >
    = 1.0;

    uniform float specularIntensity
    <
        ui_type = "slider";
        ui_label = "Specular Intensity";
        ui_tooltip = "Adjust the intensity of specular reflections.";
        ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
        ui_category = "GI Settings";
    >
    = 0.5;

    uniform float falloffDistance
    <
        ui_type = "slider";
        ui_label = "Falloff Distance";
        ui_tooltip = "Adjust the distance at which indirect light falls off.";
        ui_min = 0.1; ui_max = 10.0; ui_step = 0.1;
        ui_category = "GI Settings";
    >
    = 10.0;

/*---------------.
| :: AO Settings :: |
'---------------*/

    uniform float aoIntensity
    < 
        ui_type = "slider";
        ui_label = "AO Intensity";
        ui_tooltip = "Adjust the intensity of ambient occlusion.";
        ui_min = 0.0; ui_max = 20.0; ui_step = 0.05;
        ui_category = "AO Settings";
    >
    = 2.0;

    uniform float aoRadius
    < 
        ui_type = "slider";
        ui_label = "AO Radius";
        ui_tooltip = "Adjust the radius for ambient occlusion sampling.";
        ui_min = 0.001; ui_max = 10.0; ui_step = 0.01;
        ui_category = "AO Settings";
    >
    = 0.05;

    uniform float rayJitter <
        ui_type = "slider";
        ui_label = "Ray Jitter";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Ray Tracing";
    > = 0.25;

    uniform int rayTraceDepth <
        ui_type = "slider";
        ui_label = "Trace Depth";
        ui_min = 1; ui_max = 4; ui_step = 1;
        ui_category = "Ray Tracing";
    > = 1;

    uniform float roughness <
        ui_type = "slider";
        ui_label = "Surface Roughness";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_category = "Material Properties";
    > = 0.5;

    uniform float metallic <
    ui_type = "slider";
    ui_label = "Metallic";
    ui_tooltip = "Adjust the metallic property of the material.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Material Properties";
> = 0.5;
/*---------------.
| :: Textures :: |
'---------------*/

    texture ColorTex : COLOR;
    texture DepthTex : DEPTH;
    
    sampler ColorSampler
    {
        Texture = ColorTex;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };
    
    sampler DepthSampler
    {
        Texture = DepthTex;
        AddressU = CLAMP;
        AddressV = CLAMP;
    };

/*----------------.
| :: Functions :: |
'----------------*/

    float GetLinearDepth(float2 coords)
    {
        return ReShade::GetLinearizedDepth(coords);
    }
    
    float3 GetPosition(float2 coords)
    {
        float EyeDepth = GetLinearDepth(coords.xy) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
        return float3((coords.xy * 2.0 - 1.0) * EyeDepth, EyeDepth);
    }

    float3 GetNormalFromDepth(float2 coords)
    {
        float2 texelSize = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float depthCenter = GetLinearDepth(coords);
        float depthX = GetLinearDepth(coords + float2(texelSize.x, 0.0));
        float depthY = GetLinearDepth(coords + float2(0.0, texelSize.y));
        float3 deltaX = float3(texelSize.x, 0.0, depthX - depthCenter);
        float3 deltaY = float3(0.0, texelSize.y, depthY - depthCenter);
        return normalize(cross(deltaX, deltaY));
    }

    float3 SampleDiffuse(float2 coord)
    {
        return tex2Dlod(ColorSampler, float4(clamp(coord, 0.0, 1.0), 0, 0)).rgb;
    }

    float2 Hammersley(int i, int N)
    {
        float u = float(i) / float(N);
        float v = 0.0;
        for (int bits = i, j = 0; j < 32; j++)
        {
            v += float(bits & 1) * pow(2.0, -float(j + 1));
            bits = bits >> 1;
        }
        return float2(u, v);
    }

    float3 GetJitter(float2 texcoord, int seed)
    {
        float3 jitter = 0.0;
        float noise = frac(sin(dot(texcoord, float2(12.9898, 78.233)) + seed) * 43758.5453);
        jitter.xy = float2(cos(noise * 6.283), sin(noise * 6.283)) * rayJitter;
        return jitter;
    }

    float rand(float2 co)
    {
        return frac(sin(dot(co, float2(12.9898, 78.233))) * 43758.5453);
    }

    struct Ray
    {
        float3 Origin;
        float3 Direction;
    };
    struct HitRecord
    {
        float3 Point;
        float3 Normal;
        float T;
        bool Hit;
    };
    
    Ray GetRay(float3 origin, float3 direction)
    {
        Ray r;
        r.Origin = origin;
        r.Direction = direction;
        return r;
    }

    bool HitScene(Ray r, inout HitRecord rec)
    {
        const float maxDistance = 25.0;
        float t = 0.0;
        float stepSize = 0.15;
        bool hitFound = false;

    [loop]
        for (int i = 0; i < 32; i++)
        {
            float3 currentPos = r.Origin + r.Direction * t;
            float2 screenPos = (currentPos.xy / currentPos.z) * 0.5 + 0.5;
        
            if (any(saturate(screenPos) != screenPos))
                return false;

            float sceneDepth = GetLinearDepth(screenPos);
            float depthDelta = currentPos.z - sceneDepth;

            if (depthDelta > 0.001)
            {
            // Binary search refinement
                float low = max(t - stepSize, 0.0);
                float high = t;
            [unroll]
                for (int j = 0; j < 4; j++)
                {
                    float mid = (low + high) * 0.5;
                    float3 midPos = r.Origin + r.Direction * mid;
                    float2 midScreen = (midPos.xy / midPos.z) * 0.5 + 0.5;
                    float midDepth = GetLinearDepth(midScreen);
                
                    (midPos.z > midDepth + 0.001) ? high = mid : low = mid;
                }
            
                rec.T = high;
                rec.Point = r.Origin + high * r.Direction;
                rec.Normal = GetNormalFromDepth((rec.Point.xy / rec.Point.z) * 0.5 + 0.5);
                rec.Hit = true;
                return true;
            }
        
        // Adaptive step size
            stepSize *= 1.15;
            t += stepSize;
        
            if (t > maxDistance)
                break;
        }
    
        return false;
    }

    float3 EnergyCompensation(float3 reflectance, float specularStrength)
    {
        float3 energy = 1.0 - (reflectance * specularStrength);
        return 1.0 / max(energy, 0.0001);
    }

    float3 RayColor(Ray r, int maxDepth, float2 texcoord)
    {
        float3 color = float3(1.0, 1.0, 1.0);
        float3 throughput = float3(1.0, 1.0, 1.0);
        Ray currentRay = r;

    [loop]
        for (int depth = 0; depth < maxDepth; depth++)
        {
            HitRecord rec;
            rec.T = 1e6;
            rec.Hit = false;

            if (HitScene(currentRay, rec))
            {
            // Surface properties
                float2 hitUV = (rec.Point.xy / rec.Point.z) * 0.5 + 0.5;
                float3 albedo = SampleDiffuse(hitUV);
                float3 normal = normalize(rec.Normal);
            
            // Material properties
                float3 viewDir = -currentRay.Direction;
                float fresnel = pow(saturate(1.0 - dot(viewDir, normal)), 3.0);
                float specularStrength = saturate(fresnel * specularIntensity * 2.0);
            
            // Scattering directions
                float3 specularDir = reflect(currentRay.Direction, normal);
                float3 diffuseDir = normalize(normal + float3(
                rand(texcoord + depth + 0.1),
                rand(texcoord + depth + 0.2),
                rand(texcoord + depth + 0.3)
            ));
    
                float3 scatterDir = normalize(lerp(diffuseDir, specularDir, specularStrength));
            
            // Energy conservation
                float3 reflectance = lerp(float3(0.04, 0.04, 0.04), albedo, specularStrength);
                float3 energyComp = EnergyCompensation(reflectance, specularStrength);
                float3 attenuation = lerp(albedo * (1.0 - specularStrength), reflectance, specularStrength) * energyComp;
            
                throughput *= attenuation *
                lerp(saturate(dot(scatterDir, normal)), 1.0, specularStrength);

            // Prepare next ray
                currentRay.Origin = rec.Point + normal * 0.001;
                currentRay.Direction = normalize(scatterDir +
                rayJitter * float3(
                    rand(texcoord + depth + 0.4) - 0.5,
                    rand(texcoord + depth + 0.5) - 0.5,
                    rand(texcoord + depth + 0.6) - 0.5
                ));
            }
            else
            {
            // Environmental lighting
                float3 envColor = float3(0.4, 0.6, 1.0) * (1.0 - currentRay.Direction.y) * 2.0;
                envColor += float3(1.0, 0.9, 0.8) *
                pow(saturate(dot(currentRay.Direction,
                    normalize(float3(0.5, 0.3, 0.2)))), 8.0);
            
                color *= throughput * envColor * exp(-rec.T * 0.1);
                break;
            }
            
            if (depth > 2)
            {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (rand(texcoord + depth) > p)
                    break;
                throughput /= p;
            }
        }

        return color;
    }
    
    float CalculateAmbientOcclusion(float2 texcoord, float3 position, float3 normal)
    {
        float occlusion = 0.0;
        float radiusOverSamples = aoRadius / numSamples;

    [loop]
        for (int i = 0; i < numSamples; ++i)
        {
            float2 rand2D = Hammersley(i, numSamples);
            float angle = rand2D.x * 6.28318530718;
            float radiusScale = sqrt(rand2D.y);

            float3 jitter = GetJitter(texcoord, i) * radiusScale;
            float2 sampleDir = float2(cos(angle), sin(angle)) * radiusScale + jitter.xy;
            sampleDir *= radiusOverSamples;

            float2 sampleCoord = texcoord + sampleDir;
            float3 samplePos = GetPosition(sampleCoord);

            if (GetLinearDepth(sampleCoord) < position.z)
            {
                float3 dirToSample = normalize(samplePos - position);
                float NdotS = max(dot(normal, dirToSample), 0.0);
                float dist = length(samplePos - position);

                float falloff = exp(-dist / falloffDistance);

                occlusion += NdotS * falloff * (1.0 / (1.0 + dist));
            }
        }

        occlusion = 1.0 - (occlusion / numSamples);
        return pow(saturate(occlusion), aoIntensity * 2.0);
    }
    
    float3 FresnelSchlick(float cosTheta, float3 F0)
    {
        return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    }

    float DistributionGGX(float3 N, float3 H, float roughness)
    {
        float a = roughness * roughness;
        float a2 = a * a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH * NdotH;

        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;

        return a2 / denom;
    }

    float GeometrySchlickGGX(float NdotV, float roughness)
    {
        float r = (roughness + 1.0);
        float k = (r * r) / 8.0;

        float denom = NdotV * (1.0 - k) + k;
        return NdotV / denom;
    }

    float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
    {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx1 = GeometrySchlickGGX(NdotV, roughness);
        float ggx2 = GeometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }

    float3 CalculateSpecular(float3 N, float3 V, float3 L, float3 albedo, float roughness, float metallic)
    {
        float3 H = normalize(V + L);
        float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic); 
        float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);

        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        float3 specular = numerator / denominator;

        return specular;
    }
    
    float3 ImportanceSampleGGX(float2 xi, float3 N, float roughness)
    {
        float a = roughness * roughness;

        float phi = 2.0 * PI * xi.x;
        float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        float3 H;
        H.x = sinTheta * cos(phi);
        H.y = sinTheta * sin(phi);
        H.z = cosTheta;

        float3 upVector = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
        float3 tangentX = normalize(cross(upVector, N));
        float3 tangentY = cross(N, tangentX);

        return normalize(tangentX * H.x + tangentY * H.y + N * H.z);
    }
    
    float3 TraceSpecular(Ray ray, float3 N, float3 albedo, float roughness, float metallic, int maxDepth, float2 texcoord)
    {
        float3 color = 0.0;
        float3 throughput = 1.0;

        for (int depth = 0; depth < maxDepth; depth++)
        {
            HitRecord rec;
            rec.T = 1e6;
            rec.Hit = false;

            if (HitScene(ray, rec))
            {
                float2 hitUV = (rec.Point.xy / rec.Point.z) * 0.5 + 0.5;
                float3 hitAlbedo = SampleDiffuse(hitUV);
                float3 hitNormal = normalize(rec.Normal);

                float3 V = -ray.Direction;
                float3 L = ImportanceSampleGGX(Hammersley(depth, numSamples), hitNormal, roughness);
                float3 H = normalize(V + L);

                float3 specular = CalculateSpecular(hitNormal, V, L, hitAlbedo, roughness, metallic); // Pass metallic
                color += throughput * specular;

            // Prepare next ray
                throughput *= specular;
                ray.Origin = rec.Point + hitNormal * 0.001;
                ray.Direction = reflect(ray.Direction, hitNormal);
            }
            else
            {
            // Environmental lighting
                float3 envColor = float3(0.4, 0.6, 1.0) * (1.0 - ray.Direction.y) * 2.0;
                envColor += float3(1.0, 0.9, 0.8) * pow(saturate(dot(ray.Direction, normalize(float3(0.5, 0.3, 0.2)))), 8.0);
                color += throughput * envColor;
                break;
            }

            if (depth > 2)
            {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (rand(texcoord + depth) > p)
                    break;
                throughput /= p;
            }
        }

        return color;
    }
    
    float3 CalculateIndirectLight(float2 texcoord, float radius)
    {
        float3 indirect = 0.0;
        float3 viewPos = GetPosition(texcoord);
        float3 normal = GetNormalFromDepth(texcoord);
        float randSeed = rand(texcoord);

    [loop]
        for (int i = 0; i < numSamples; ++i)
        {
        // Importance-sampled hemisphere
            float2 xi = Hammersley(i, numSamples);
            float3 jitter = rayJitter * float3(
            rand(texcoord + i + 0.1),
            rand(texcoord + i + 0.2),
            rand(texcoord + i + 0.3)
        );

            float3 sampleDir = ImportanceSampleGGX(xi, normal, 0.5);
            Ray giRay = GetRay(viewPos, normalize(sampleDir + jitter));

        // Trace multi-bounce light transport
            float3 rayColor = RayColor(giRay, rayTraceDepth, texcoord);
            float NdotS = dot(normal, sampleDir);

            float distanceToSample = length(viewPos - giRay.Origin);
            float falloff = exp(-distanceToSample / falloffDistance); 
            indirect += rayColor * NdotS * falloff * diffuseIntensity;
        }
        indirect *= giIntensity / numSamples;
        indirect *= CalculateAmbientOcclusion(texcoord, viewPos, normal);

        return indirect;
    }
    
    float4 GlobalIlluminationPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
    {
        float3 originalColor = SampleDiffuse(texcoord);
        float3 indirectLight = CalculateIndirectLight(texcoord, sampleRadius);
        float3 finalColor = originalColor + giIntensity * indirectLight * 1.5;

        if (viewMode == 0)
        {
            return float4(finalColor, 1.0);
        }
        else if (viewMode == 1)
        {
            return float4(indirectLight, 1.0);
        }
        else if (viewMode == 2)
        {
            float depth = GetLinearDepth(texcoord);
            return float4(depth, depth, depth, 1.0);
        }
        else if (viewMode == 3)
        {
            float3 normal = GetNormalFromDepth(texcoord);
            return float4(normal * 0.5 + 0.5, 1.0);
        }
        if (viewMode == 4) // Specular Debug
        {
            float3 normal = GetNormalFromDepth(texcoord);
            float3 viewDir = -normalize(GetPosition(texcoord));
            float3 albedo = SampleDiffuse(texcoord);

            float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic); 
            float3 fresnel = FresnelSchlick(max(dot(normal, viewDir), 0.0), F0);

            float3 specular = CalculateSpecular(normal, viewDir, normalize(reflect(-viewDir, normal)), albedo, roughness, metallic);
            return float4(specular * fresnel, 1.0);
        }
        return float4(originalColor, 1.0);
    }
    
    void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
    {
        texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
        position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    }

/*-----------------.
| :: Techniques :: |
'-----------------*/

    technique SSGI
    {
        pass
        {
            VertexShader = PostProcessVS;
            PixelShader = GlobalIlluminationPS;
        }
    }
}
