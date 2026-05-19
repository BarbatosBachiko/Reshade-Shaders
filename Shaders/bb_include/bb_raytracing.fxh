/*-------------------------------|
| ::     BaBa Ray Tracing     :: |
|-------------------------------*/

#pragma once

#include "bb_depth.fxh"

// Structs
struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitResult
{
    bool found;
    float3 viewPos;
    float2 uv;
};

struct SurfaceData
{
    float3 viewPos;
    float3 normal;
    float3 viewDir;
    float depth;
    float2 uv;
};

// Fresnel (Schlick)
float3 RT_F_Schlick(float VdotH, float3 f0)
{
    float t = 1.0 - VdotH;
    float t2 = t * t;
    return f0 + (1.0 - f0) * (t2 * t2 * t);
}

// Thickness Calculation
float RT_GetThickness(float2 uv, float3 normal, float3 viewDir, float depth, float thicknessThreshold)
{
    float NdotV = abs(dot(normal, -viewDir));
    float geometricScale = 1.0 / max(NdotV, 0.2);
    float depthDerivative = fwidth(depth);
    float edgeMask = 1.0 - smoothstep(0.0, 0.002, depthDerivative);
    return thicknessThreshold * geometricScale * edgeMask;
}

bool RT_TraceRay(float3 origin, float3 dir, float2 pScale, int num_steps, float jitter,
                 float maxDist, float thickness, bool quadratic, int binIters,
                 out float2 hitUV, out float3 hitPos, out float hitDist)
{
    float3 lastPos = origin;
    float3 current = origin;
    float2 startUV  = ViewPosToUV(origin, pScale);
    float2 prevUV   = startUV;
    float  prevZ    = origin.z;
    float3 endPos   = origin + dir * maxDist;
    float2 endUV    = ViewPosToUV(endPos, pScale);
    float2 deltaUV  = endUV - startUV;

    // Early out for degenerate rays
    if (dot(deltaUV, deltaUV) < 0.0001)
    {
        hitUV = 0.0; hitPos = 0.0; hitDist = 0.0;
        return false;
    }

    float startK = 1.0 / origin.z;
    float endK   = 1.0 / endPos.z;
    float deltaK = endK - startK;
    float stepSize = 1.0 / (float)num_steps;

    if (!quadratic)
    {
        // Linear UV stepping
        float t0 = stepSize * jitter;
        float2 currUV = startUV + deltaUV * t0;
        float  currK  = startK  + deltaK  * t0;
        float2 stepUV = deltaUV * stepSize;
        float  stepK  = deltaK  * stepSize;
        float  prevRayDepth = 1.0 / max(abs(currK - stepK), 1e-10);

        // Early exit counters for warp coherence
        int consecutiveMisses = 0;
        static const int MAX_MISSES = 4;

        [loop]
        for (int i = 0; i < num_steps; ++i)
        {
            if (any(currUV < 0.0) || any(currUV > 1.0))
                break;

            float rayDepth   = 1.0 / currK;
            float sceneDepth = GetDepth(currUV);
            float depthDiff  = rayDepth - sceneDepth;
            float rayStepSizeZ = abs(rayDepth - prevRayDepth);
            float adaptiveThickness = max(thickness, rayStepSizeZ * 1.5);
            adaptiveThickness *= (1.0 + rayDepth * 0.2);

            if (depthDiff > 0.0 && depthDiff < adaptiveThickness)
            {
                // Reset miss counter on potential hit
                consecutiveMisses = 0;

                // Binary search in UV space (2 iterations, SSR only)
                float2 loUV = currUV - stepUV;
                float2 hiUV = currUV;
                [unroll]
                for (int j = 0; j < 2; j++)
                {
                    float2 midUV = (loUV + hiUV) * 0.5;
                    float midRayDepth = 1.0 / (currK - stepK * 0.5);
                    if (midRayDepth > GetDepth(midUV))
                        hiUV = midUV;
                    else
                        loUV = midUV;
                }
                hitUV = hiUV;
                hitPos = UVToViewPos(hiUV, GetDepth(hiUV), pScale);
                hitDist = length(hitPos - origin);
                return true;
            }
            else if (depthDiff < -adaptiveThickness * 3.0)
            {
                // Deep behind geometry
                consecutiveMisses++;
                if (consecutiveMisses >= MAX_MISSES)
                    break;
            }

            prevRayDepth = rayDepth;
            currUV += stepUV;
            currK  += stepK;
        }
    }
    else
    {
        // Quadratic view-space stepping (GI)
        int consecutiveMisses = 0;
        static const int MAX_MISSES_GI = 4;

        [loop]
        for (int i = 1; i <= num_steps; i++)
        {
            float t = (float(i) - 1.0 + jitter) / (float)num_steps;
            t = t * t;
            float distScale = maxDist * t;
            current = origin + dir * distScale;
            float2 cuv = ViewPosToUV(current, pScale);

            if (any(cuv < 0.0) || any(cuv > 1.0))
            {
                hitUV = 0.0; hitPos = 0.0; hitDist = 0.0;
                return false;
            }

            float zScene = GetDepth(cuv);
            float zRay   = current.z;
            float depthDiff = zRay - zScene;

            if (zScene < 0.999 && depthDiff > 0.0 && depthDiff < thickness)
            {
                consecutiveMisses = 0;

                // Binary search in view-space (4 iterations, GI)
                float3 startPos = lastPos;
                float3 endPos   = current;
                [loop]
                for (int r = 0; r < 4; r++)
                {
                    float3 midPos = (startPos + endPos) * 0.5;
                    float2 midUV = ViewPosToUV(midPos, pScale);
                    if (any(midUV < 0.0) || any(midUV > 1.0))
                        break;
                    float midDepth = GetDepth(midUV);
                    if (midPos.z > midDepth)
                        endPos = midPos;
                    else
                        startPos = midPos;
                }
                current = endPos;
                hitUV   = ViewPosToUV(current, pScale);
                hitPos  = current;
                hitDist = length(current - origin);
                return true;
            }
            else if (depthDiff < -thickness * 3.0)
            {
                consecutiveMisses++;
                if (consecutiveMisses >= MAX_MISSES_GI)
                {
                    hitUV = 0.0; hitPos = 0.0; hitDist = 0.0;
                    return false;
                }
            }

            lastPos = current;
        }
    }

    hitUV = 0.0; hitPos = 0.0; hitDist = 0.0;
    return false;
}

// Ray Marching (SSR)
HitResult RT_TraceRay2D(Ray r, int num_steps, float max_dist, float2 pScale, float jitter, float geoThickness)
{
    HitResult result;
    float2 hitUV;
    float3 hitPos;
    float  hitDist;
    result.found = RT_TraceRay(r.origin, r.direction, pScale, num_steps, jitter, max_dist,
                                geoThickness, false, 2, hitUV, hitPos, hitDist);
    result.viewPos = hitPos;
    result.uv = hitUV;
    return result;
}

// Ray Marching (GI)
bool RT_TraceRayGI(float3 origin, float3 dir, float2 pScale, int steps, float jitter,
                   float maxRayDist, float currentThickness,
                   out float2 hitUV, out float3 hitPos, out float hitDist)
{
    return RT_TraceRay(origin, dir, pScale, steps, jitter, maxRayDist,
                        currentThickness, true, 4, hitUV, hitPos, hitDist);
}

// GGX Importance Sampling
float3 RT_ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    
    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// VNDF Importance Sampling by RicoJuly
float3 RT_ImportanceSampleGGX_VNDF(float2 Xi, float3 N, float3 V, float roughness)
{
    float alpha = roughness * roughness;
    Xi = clamp(Xi, 0.001, 0.999);
    
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    
    float3 V_local = float3(dot(V, T), dot(V, B), dot(V, N));
    float3 V_stretch = normalize(float3(alpha * V_local.x, alpha * V_local.y, V_local.z));
    
    float3 T1 = (V_stretch.z < 0.9999) ?
        normalize(cross(V_stretch, float3(0.0, 0.0, 1.0))) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(T1, V_stretch);
    float a = 1.0 / (1.0 + V_stretch.z);
    float r = sqrt(Xi.x);
    float phi = (Xi.y < a) ?
        (Xi.y / a) * PI : PI + (Xi.y - a) / (1.0 - a) * PI;
    float P1 = r * cos(phi);
    float P2 = r * sin(phi) * ((Xi.y < a) ? 1.0 : V_stretch.z);
    float3 N_local = P1 * T1 + P2 * T2 + sqrt(max(0.0, 1.0 - P1 * P1 - P2 * P2)) * V_stretch;
    N_local = normalize(float3(alpha * N_local.x, alpha * N_local.y, max(0.0, N_local.z)));
    return normalize(T * N_local.x + B * N_local.y + N * N_local.z);
}

// Cosine-weighted Hemisphere Sampling
void RT_GenTB(float3 N, out float3 T, out float3 B)
{
    float s = N.z < 0.0 ? -1.0 : 1.0;
    float a = -1.0 / (s + N.z);
    float b = N.x * N.y * a;
    T = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
    B = float3(b, s + N.y * N.y * a, -N.y);
}

float3 RT_CosineSample(float3 N, float2 r)
{
    float3 T, B;
    RT_GenTB(N, T, B);
    r.x *= 2.0 * PI;
    float s = sqrt(max(0.0, 1.0 - r.y));
    float2 sincos_rx;
    sincos(r.x, sincos_rx.x, sincos_rx.y);
    return T * (sincos_rx.y * s) + B * (sincos_rx.x * s) + N * sqrt(r.y);
}

// Glossy
float RT_SpecularPowerToConeAngle(float specularPower)
{
    if (specularPower >= 4096.0)
        return 0.0;
    float exponent = rcp(specularPower + 1.0);
    return acos(clamp(pow(abs(0.244), exponent), -1.0, 1.0));
}

float RT_IsoscelesTriangleOpposite(float adjacentLength, float coneTheta)
{
    return 2.0 * tan(coneTheta) * adjacentLength;
}

float RT_IsoscelesTriangleInRadius(float a, float h)
{
    float a2 = a * a;
    float fh2 = 4.0 * h * h;
    return (a * (sqrt(a2 + fh2) - a)) / max(4.0 * h, 1e-6);
}
