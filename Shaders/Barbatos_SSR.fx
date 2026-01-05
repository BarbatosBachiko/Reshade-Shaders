/*----------------------------------------------|
| :: Barbatos SSR (Screen-Space Reflections) :: |
|-----------------------------------------------|
| Version: 0.6.0                                |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "Blending.fxh"

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Intensity";
    ui_tooltip = "Overall intensity of reflections";
> = 1.0;

uniform float THICKNESS_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.6; ui_step = 0.001;
    ui_category = "Basic Settings";
    ui_label = "Thickness";
    ui_tooltip = "Controls how 'thick' surfaces are before a ray passes through them.";
> = 0.066;

uniform float FadeDistance <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Basic Settings";
    ui_label = "Fade Distance";
    ui_tooltip = "How far away reflections start to fade out";
> = 4.999;

uniform int ReflectionMode <
    ui_type = "combo";
    ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
    ui_category = "Basic Settings";
    ui_label = "Surfaces";
    ui_tooltip = "Choose which surfaces show reflections";
> = 4;

uniform float SurfaceGlossiness <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Material";
    ui_label = "Surface Glossiness";
    ui_tooltip = "Controls surface smoothness.\n0 = Sharp reflections (Defined)\n1 = Total roughness";
> = 0.25;

uniform float MetallicFactor <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Material";
    ui_label = "Metallic";
    ui_tooltip = "Make surfaces look more metallic (0=non-metal, 1=metal)";
> = 0.2;

uniform float RoughnessDetection <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Material";
    ui_label = "Roughness Detection";
    ui_tooltip = "Estimates roughness based on local color contrast.\nHigher values make detailed/noisy textures appear rougher.";
> = 1.0;

uniform float SurfaceDetails <
    ui_category = "Material";
    ui_label = "Details";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_tooltip = "Adds small surface details to reflections";
> = 0.1;

uniform float RenderResolution <
    ui_type = "drag";
    ui_min = 0.3; ui_max = 1.0; ui_step = 0.05;
    ui_category = "Performance";
    ui_label = "Resolution %";
    ui_tooltip = "Lower values = better performance but less details";
> = 0.8;

uniform bool EnableSmoothing <
    ui_category = "Performance";
    ui_label = "Reduce Reflection Noise (TAA)";
    ui_tooltip = "Reduces flickering and noise in reflections using temporal anti-aliasing";
> = false;

BLENDING_COMBO(g_BlendMode, "Blending Mode", "Select how reflections are blended with the scene.", "Color Adjustments", false, 0, 0)

uniform float g_Contrast <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Color Adjustments";
    ui_label = "Contrast";
> = 1.0;

uniform float g_Saturation <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Color Adjustments";
    ui_label = "Saturation";
> = 1.0;

uniform int SmartSurfaceMode <
    ui_type = "combo";
    ui_items = "Performance\0Balanced\0Quality\0";
    ui_category = "Advanced Options";
    ui_label = "Smart Surface Detection";
    ui_tooltip = "Quality of the normal smoothing filter.";
> = 0;

uniform float VERTICAL_FOV <
    __UNIFORM_DRAG_FLOAT1
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced Options";
    ui_label = "Vertical FOV";
> = 60.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0Motion\0";
    ui_category = "Advanced Options";
    ui_label = "Debug View";
    ui_tooltip = "Special views";
> = 0;

// Defines & Constants
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

static const float DEG2RAD = 0.017453292;
// PI / 180.0
static const float SobelEdgeThreshold = 0.03;
static const float Smooth_Threshold = 0.5;
static const int GlossySamples = 10;
static const float OrientationThreshold = 0.5;
static const float GeoCorrectionIntensity = -0.01;
static const float EDGE_MASK_THRESHOLD = 0.2;
static const float DIELECTRIC_REFLECTANCE = 0.04;
static const int STEPS_PER_RAY_WALLS = 32;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 GetMV(float2 texcoord) { return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
}
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 GetMV(float2 texcoord) { return GetLod(sMotVectTexVort, texcoord).rg;
}
#else
texture texMotionVectors
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};
sampler sTexMotionVectorsSampler
{
    Texture = texMotionVectors;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};
float2 GetMV(float2 texcoord)
{
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace Barbatos_SSR401
{
    texture Normal
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = Normal;
    };

    texture Normal1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal1
    {
        Texture = Normal1;
    };
    
    texture Reflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sReflection
    {
        Texture = Reflection;
        AddressU = Clamp;
        AddressV = Clamp;
    };

    texture History0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sHistory0
    {
        Texture = History0;
    };

    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16;
    };
    sampler sHistory1
    {
        Texture = History1;
    };

    texture TexColorCopy
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
        MipLevels = 8;
    };
    sampler sTexColorCopy
    {
        Texture = TexColorCopy;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
        MipFilter = LINEAR;
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
    
    void VS_Barbatos_SSR(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ?
        2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        float y = tan(VERTICAL_FOV * DEG2RAD * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }

    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }

    float3 F_Schlick(float VdotH, float3 f0)
    {
        float t = 1.0 - VdotH;
        float t2 = t * t;
        return f0 + (1.0 - f0) * (t2 * t2 * t);
    }
    
    float3 HSVToRGB(float3 c)
    {
        float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }

    float2 ConcentricSquareMapping(float2 u)
    {
        float2 ab = 2.0 * u - 1.0;
        float2 ab2 = ab * ab;
        
        float r, phi;
        if (ab2.x > ab2.y)
        {
            r = ab.x;
            phi = (PI / 4.0) * (ab.y / ab.x);
        }
        else
        {
            r = ab.y;
            phi = (ab.y != 0.0) ? (PI / 2.0) - (PI / 4.0) * (ab.x / ab.y) : 0.0;
        }
        
        float2 sincosPhi;
        sincos(phi, sincosPhi.y, sincosPhi.x);
        return r * sincosPhi;
    }

    float3 AdjustContrast(float3 color, float contrast)
    {
        return (color - 0.5) * contrast + 0.5;
    }

    float3 AdjustSaturation(float3 color, float saturation)
    {
        float lum = GetLuminance(color);
        return lerp(lum.xxx, color, saturation);
    }

    float GetSpatialTemporalNoise(float2 pos)
    {
        float time = fmod((float) FRAME_COUNT, 64.0);
        return frac(52.9829189 * frac(0.06711056 * pos.x + 0.00583715 * pos.y + 0.006237 * time));
    }
    
    float GetSpatialNoise(float2 pos)
    {
        return frac(52.9829189 * frac(0.06711056 * pos.x + 0.00583715 * pos.y));
    }

    //------------------------------------|
    // :: View Space & Normal Functions ::|
    //------------------------------------|
    float3 UVToViewPos(float2 uv, float view_z, float2 pScale)
    {
        float2 ndc = uv * 2.0 - 1.0;
        return float3(ndc.x * pScale.x * view_z, -ndc.y * pScale.y * view_z, view_z);
    }

    float2 ViewPosToUV(float3 view_pos, float2 pScale)
    {
        if (abs(view_pos.z) < 1e-6)
            return 0.5;
        float2 ndc = view_pos.xy / (view_pos.z * pScale);
        return float2(ndc.x, -ndc.y) * 0.5 + 0.5;
    }
    
    float3 GVPFUV(float2 uv, float2 pScale)
    {
        return UVToViewPos(uv, GetDepth(uv), pScale);
    }

    float3 CalculateNormal(float2 texcoord, float2 pScale)
    {
        float3 offset_x = GVPFUV(texcoord + float2(ReShade::PixelSize.x, 0.0), pScale);
        float3 offset_y = GVPFUV(texcoord + float2(0.0, ReShade::PixelSize.y), pScale);
        float3 center = GVPFUV(texcoord, pScale);
        
        return normalize(cross(center - offset_x, center - offset_y));
    }
    
    float3 ApplySurfaceDetails(float2 texcoord, float3 normal)
    {
        if (SurfaceDetails == 0.0 && GeoCorrectionIntensity == 0.0)
            return normal;
        float4 pTL = tex2DgatherG(ReShade::BackBuffer, texcoord - ReShade::PixelSize * 0.5);
        float4 pTR = tex2DgatherG(ReShade::BackBuffer, texcoord + float2(ReShade::PixelSize.x, -ReShade::PixelSize.y) * 0.5);
        float4 pBL = tex2DgatherG(ReShade::BackBuffer, texcoord + float2(-ReShade::PixelSize.x, ReShade::PixelSize.y) * 0.5);
        float4 pBR = tex2DgatherG(ReShade::BackBuffer, texcoord + ReShade::PixelSize * 0.5);
        float Gx = (pTR.z + 2.0 * pBR.z + pBR.y) - (pTL.w + 2.0 * pTL.x + pBL.x);
        float Gy = (pBL.x + 2.0 * pBR.w + pBR.y) - (pTL.w + 2.0 * pTL.z + pTR.z);
        float3 finalNormal = normal;

        if (SurfaceDetails > 0.0 && dot(float2(Gx, Gy), float2(Gx, Gy)) >= (SobelEdgeThreshold * SobelEdgeThreshold))
        {
            float2 slope = float2(Gx, Gy) * SurfaceDetails;
            float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 T = normalize(cross(up, normal));
            float3 B = cross(normal, T);
            finalNormal = normalize(finalNormal + T * slope.x - B * slope.y);
        }
    
        if (GeoCorrectionIntensity != 0.0)
        {
            float3 bumpNormal = normalize(float3(Gx, Gy, 1.0));
            finalNormal = normalize(finalNormal + bumpNormal * GeoCorrectionIntensity);
        }

        return finalNormal;
    }
    
    float4 ComputeSmoothedNormal(float2 uv, float2 direction, sampler sInput)
    {
        float4 color = GetLod(sInput, uv);
        float SNWidth = (SmartSurfaceMode == 0) ? 5.5 : ((SmartSurfaceMode == 1) ? 2.5 : 1.0);
        int SNSamples = (SmartSurfaceMode == 0) ? 1 : ((SmartSurfaceMode == 1) ? 3 : 30);
        float2 p = ReShade::PixelSize * SNWidth * direction;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        float4 s1 = 0.0;
        float sc = 0.0;
        
        [loop]
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            float4 s = GetLod(sInput, uv + (p * x));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T);
            s1 += s * diff;
            sc += diff;
        }
        return (sc > 0.0001) ? (s1 / sc) : color;
    }
    
    float4 SampleGBuffer(float2 uv)
    {
        return GetLod(sNormal, uv);
    }

    //-------------------|
    // :: Ray Tracing  ::|
    //-------------------|
    HitResult TraceRay2D(Ray r, int num_steps, float2 pScale, float jitter)
    {
        HitResult result;
        result.found = false;
        result.viewPos = 0.0;
        result.uv = 0.0;

        //Max Ray Distance
        float maxDist = 10.0;
        float3 endPos = r.origin + r.direction * maxDist;

        float2 startUV = ViewPosToUV(r.origin, pScale);
        float2 endUV = ViewPosToUV(endPos, pScale);
        float startK = 1.0 / r.origin.z;
        float endK = 1.0 / endPos.z;
        //Compute Deltas
        float2 deltaUV = endUV - startUV;
        float deltaK = endK - startK;

        if (dot(deltaUV, deltaUV) < 0.0001)
            return result;
        float stepSize = 1.0 / (float) num_steps;
        float t = stepSize * jitter;
        float2 currUV = startUV + deltaUV * t;
        float currK = startK + deltaK * t;
        float2 stepUV = deltaUV * stepSize;
        float stepK = deltaK * stepSize;
        [loop]
        for (int i = 0; i < num_steps; ++i)
        {
            if (any(currUV < 0.0) || any(currUV > 1.0))
                break;
            // Recover View Depth
            float rayDepth = 1.0 / currK;
            float sceneDepth = GetDepth(currUV);

            // Intersection Check
            float depthDiff = rayDepth - sceneDepth;
            // Adaptive thickness
            float prevRayDepth = 1.0 / (currK - stepK);
            float rayStepSizeZ = abs(rayDepth - prevRayDepth);
            float adaptiveThickness = max(THICKNESS_THRESHOLD, rayStepSizeZ * 1.5);
            adaptiveThickness *= (1.0 + rayDepth * 0.1);
            if (depthDiff > 0.0 && depthDiff < adaptiveThickness)
            {
                //Binary Search Refinement
                float2 loUV = currUV - stepUV;
                float2 hiUV = currUV;
                float2 midUV;
                float midRayDepth;
            
                [unroll]
                for (int j = 0; j < 4; j++)
                {
                    midUV = (loUV + hiUV) * 0.5;
                    float t_mid = t - (stepSize * 0.5) + (stepSize * (float) j * 0.0);
                    midRayDepth = 1.0 / (currK - stepK * 0.5);
                
                    if (midRayDepth > GetDepth(midUV))
                        hiUV = midUV;
                    else
                        loUV = midUV;
                }

                result.found = true;
                result.uv = hiUV;
                result.viewPos = UVToViewPos(hiUV, GetDepth(hiUV), pScale);
                return result;
            }

            // Advance Ray
            currUV += stepUV;
            currK += stepK;
            t += stepSize;
        }

        return result;
    }

    //---------------|
    // :: Glossy  :: |
    //---------------|
    float GetLocalRoughness(float2 uv)
    {
        float3 center = GetLod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb;
        float lumaC = GetLuminance(center);
    
        float2 p = ReShade::PixelSize;
        
        float lumaN = GetLuminance(GetLod(ReShade::BackBuffer, float4(uv + float2(p.x, 0), 0, 0)).rgb);
        float lumaS = GetLuminance(GetLod(ReShade::BackBuffer, float4(uv - float2(p.x, 0), 0, 0)).rgb);
        float lumaE = GetLuminance(GetLod(ReShade::BackBuffer, float4(uv + float2(0, p.y), 0, 0)).rgb);
        float lumaW = GetLuminance(GetLod(ReShade::BackBuffer, float4(uv - float2(0, p.y), 0, 0)).rgb);
        float variance = abs(lumaN - lumaC) + abs(lumaS - lumaC) + abs(lumaE - lumaC) + abs(lumaW - lumaC);
        return saturate(variance * 10.0);
    }
    
    float specularPowerToConeAngle(float specularPower)
    {
        if (specularPower >= 4096.0) // exp2(12.0)
            return 0.0;
        float exponent = rcp(specularPower + 1.0);
        return acos(clamp(pow(0.244, exponent), -1.0, 1.0));
    }
    
    float isoscelesTriangleOpposite(float adjacentLength, float coneTheta)
    {
        return 2.0 * tan(coneTheta) * adjacentLength;
    }
    
    float isoscelesTriangleInRadius(float a, float h)
    {
        float a2 = a * a;
        float fh2 = 4.0 * h * h;
        return (a * (sqrt(a2 + fh2) - a)) / max(4.0 * h, 1e-6);
    }

    float3 GetGlossySample(float2 sample_uv, float2 pixel_uv, float local_roughness)
    {
        float netRoughness = saturate(SurfaceGlossiness + (local_roughness * RoughnessDetection));
        if (netRoughness <= 0.001)
            return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, 0)).rgb;
        float specularPower = pow(2.0, 10.0 * (1.0 - netRoughness) + 1.0);
        float coneTheta = specularPowerToConeAngle(specularPower) * 0.5;
        float2 deltaP = (sample_uv - pixel_uv) * BUFFER_SCREEN_SIZE;
        float adjacentLength = length(deltaP);
        float oppositeLength = isoscelesTriangleOpposite(adjacentLength, coneTheta);
        float incircleSize = isoscelesTriangleInRadius(oppositeLength, adjacentLength);

        float rawMip = log2(max(1.0, incircleSize));
        float mipLevel = clamp(rawMip - 1.5, 0.0, 4.0);
        int adaptedSamples = (mipLevel > 1.0) ? max(6, GlossySamples / 2) : GlossySamples;

        float3 reflectionColor = 0.0;
        float noise = GetSpatialTemporalNoise(pixel_uv * BUFFER_SCREEN_SIZE * RenderResolution);
        float blurRadiusUV = incircleSize * ReShade::PixelSize.x;
        [loop]
        for (int i = 0; i < adaptedSamples; ++i)
        {
            float2 u = float2(
                frac(noise + float(i) * 0.61803398875),
                frac(noise + float(i) * 0.83462692011)
            );
            float2 offset = ConcentricSquareMapping(u) * blurRadiusUV;
            reflectionColor += tex2Dlod(sTexColorCopy, float4(sample_uv + offset, 0, mipLevel)).rgb;
        }

        return reflectionColor / float(adaptedSamples);
    }
    
    //------------|
    // :: TAA  :: |
    //------------|
    
    float4 GetActiveHistory(float2 uv)
    {
        return (fmod((float) FRAME_COUNT, 2.0) < 0.5) ?
        GetLod(sHistory0, float4(uv, 0, 0)) : GetLod(sHistory1, float4(uv, 0, 0));
    }

    float3 ClipToAABB(float3 aabb_min, float3 aabb_max, float3 history_sample)
    {
        float3 p_clip = 0.5 * (aabb_max + aabb_min);
        float3 e_clip = 0.5 * (aabb_max - aabb_min) + 1e-6;
        float3 v_clip = history_sample - p_clip;
        float3 v_unit = v_clip / e_clip;
        float3 a_unit = abs(v_unit);
        float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
        return (ma_unit > 1.0) ? (p_clip + v_clip / ma_unit) : history_sample;
    }
    
    float2 GetVelocityFromClosestFragment(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0.0;

        static const float2 offsets[5] = { float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1) };
        [unroll]
        for (int i = 0; i < 5; i++)
        {
            float2 s_coord = texcoord + offsets[i] * pixel_size;
            float s_depth = GetDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                closest_velocity = GetMV(s_coord);
            }
        }
        return closest_velocity;
    }
    
    void ComputeNeighborhoodMinMax(sampler2D color_tex, float2 texcoord, float3 center_color, out float3 color_min, out float3 color_max)
    {
        float4 r_quad = tex2DgatherR(color_tex, texcoord);
        float4 g_quad = tex2DgatherG(color_tex, texcoord);
        float4 b_quad = tex2DgatherB(color_tex, texcoord);

        color_min.r = min(min(r_quad.x, r_quad.y), min(r_quad.z, r_quad.w));
        color_min.g = min(min(g_quad.x, g_quad.y), min(g_quad.z, g_quad.w));
        color_min.b = min(min(b_quad.x, b_quad.y), min(b_quad.z, b_quad.w));
    
        color_max.r = max(max(r_quad.x, r_quad.y), max(r_quad.z, r_quad.w));
        color_max.g = max(max(g_quad.x, g_quad.y), max(g_quad.z, g_quad.w));
        color_max.b = max(max(b_quad.x, b_quad.y), max(b_quad.z, b_quad.w));

        color_min = min(center_color, color_min);
        color_max = max(center_color, color_max);
    }

    float4 ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
            return 0.0;
        float2 packed_uv = input.uv * RenderResolution;
        float4 current_reflection = GetLod(sReflection, float4(packed_uv, 0, 0));
        if (!EnableSmoothing)
            return current_reflection;
        // Reprojection
        float2 velocity = GetVelocityFromClosestFragment(input.uv);
        float2 reprojected_uv = input.uv + velocity;
        if (any(saturate(reprojected_uv) != reprojected_uv) || FRAME_COUNT <= 1)
            return current_reflection;
        float4 history_reflection = GetLod(sHistoryParams, float4(reprojected_uv, 0, 0));

        // Depth Check
        float history_depth = GetDepth(reprojected_uv);
        if (abs(history_depth - depth) > 0.005)
            return current_reflection;
        // AABB Clip
        float3 color_min, color_max;
        ComputeNeighborhoodMinMax(sReflection, packed_uv, current_reflection.rgb, color_min, color_max);
        float3 clipped_history_rgb = ClipToAABB(color_min, color_max, history_reflection.rgb);

        // Confidence
        float curr_luma = GetLuminance(current_reflection.rgb);
        float hist_luma = GetLuminance(clipped_history_rgb);
        float luma_diff = abs(curr_luma - hist_luma);

        float velocity_weight = saturate(1.0 - length(velocity * RenderResolution * 100.0));
        float raw_confidence = (1.0 - saturate(luma_diff * 8.0)) * velocity_weight;
        // Confidence Tonemapping
        float compressed_confidence = saturate(raw_confidence + log2(2.0 - raw_confidence) * 0.5);
        // Feedback 
        float feedback = compressed_confidence * 0.95;
        float3 temporal_rgb = lerp(current_reflection.rgb, clipped_history_rgb, feedback);
        float temporal_a = lerp(current_reflection.a, history_reflection.a, feedback);
        
        return float4(temporal_rgb, temporal_a);
    }
    
    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_CopyColor(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        float3 color = tex2D(ReShade::BackBuffer, input.uv).rgb;
        float roughness = GetLocalRoughness(input.uv);
        outColor = float4(color, roughness);
    }
    
    bool CheckSkyNormal(float2 uv, out float4 outNormal)
    {
        if (GetDepth(uv) >= 1.0)
        {
            outNormal = float4(0.0, 0.0, 1.0, 1.0);
            return true;
        }
        return false;
    }
    
    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (CheckSkyNormal(input.uv, outNormal))
            return;
        float3 normal = CalculateNormal(input.uv, input.pScale);
        outNormal = float4(normal, GetDepth(input.uv));
    }
    
    void PS_SmoothNormals_H(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (CheckSkyNormal(input.uv, outNormal))
            return;
        outNormal = ComputeSmoothedNormal(input.uv, float2(1, 0), sNormal);
    }

    void PS_SmoothNormals_V(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (CheckSkyNormal(input.uv, outNormal))
            return;
        float4 smoothed = ComputeSmoothedNormal(input.uv, float2(0, 1), sNormal1);
        float3 finalNormal = ApplySurfaceDetails(input.uv, smoothed.rgb);
        outNormal = float4(finalNormal, smoothed.a);
    }
    
    void PS_TraceReflections(VS_OUTPUT input, out float4 outReflection : SV_Target)
    {
        float2 scaled_uv = input.uv / RenderResolution;
        if (any(scaled_uv < 0.001) || any(scaled_uv > 0.999))
        {
            outReflection = 0;
            return;
        }

        float4 gbuffer = SampleGBuffer(scaled_uv);
        float depth = gbuffer.a;
        if (depth >= 1.0)
        {
            outReflection = 0.0;
            return;
        }
        
        float2 pScale = input.pScale;
        float3 normal = normalize(gbuffer.rgb);
        float3 viewPos = UVToViewPos(scaled_uv, depth, pScale);
        float3 viewDir = -normalize(viewPos);
        bool showFloor = (ReflectionMode == 0 || ReflectionMode == 3 || ReflectionMode == 4);
        bool showWall = (ReflectionMode == 1 || ReflectionMode == 4);
        bool showCeil = (ReflectionMode == 2 || ReflectionMode == 3 || ReflectionMode == 4);

        bool isFloor = normal.y > OrientationThreshold;
        bool isCeiling = normal.y < -OrientationThreshold;
        bool isWall = abs(normal.y) <= OrientationThreshold;
        float orientationIntensity = (isFloor * (float) showFloor) + (isWall * (float) showWall) + (isCeiling * (float) showCeil);
        if (orientationIntensity <= 0.0)
        {
            outReflection = 0;
            return;
        }

        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(-viewDir, normal));
        float bias = 0.0005 + (depth * 0.02);
        r.origin += r.direction * bias;
        
        float VdotN = dot(viewDir, normal);
        if (VdotN > 0.9 || r.direction.z < 0.0)
        {
            outReflection = 0.0;
            return;
        }

        HitResult hit;
        float jitter = GetSpatialNoise(scaled_uv * BUFFER_SCREEN_SIZE);
        if (isWall)
            hit = TraceRay2D(r, STEPS_PER_RAY_WALLS, pScale, jitter);
        else
            hit = TraceRay2D(r, 64, pScale, jitter);
        float3 reflectionColor = 0;
        float reflectionAlpha = 0.0;
        float estimatedRoughness = GetLod(sTexColorCopy, float4(scaled_uv, 0, 0)).a;
        if (hit.found)
        {
            reflectionColor = GetGlossySample(hit.uv, scaled_uv, estimatedRoughness);
            // Distance Fading
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / 10.0);
            float fadeRange = max(FadeDistance, 0.001);
            float depthFade = saturate((FadeDistance - depth) / fadeRange);
            depthFade *= depthFade;
            // Screen Edge Fade
            float2 edgeDist = min(hit.uv, 1.0 - hit.uv);
            float screenFade = smoothstep(0.0, 0.001, min(edgeDist.x, edgeDist.y));
            reflectionAlpha = distFactor * depthFade * screenFade;
            // Geometry Masking
            float3 nR = SampleGBuffer(scaled_uv + float2(ReShade::PixelSize.x, 0)).rgb;
            float3 nD = SampleGBuffer(scaled_uv + float2(0, ReShade::PixelSize.y)).rgb;
            float edgeDelta = length(normal - nR) + length(normal - nD);
            float geoMask = 1.0 - smoothstep(0.05, EDGE_MASK_THRESHOLD, edgeDelta);
            
            reflectionAlpha *= geoMask;
        }
        else
        {
            float adaptiveDist = min(depth * 1.2 + 0.012, 10.0);
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos, pScale).xy);
            reflectionColor = GetGlossySample(uvFb, scaled_uv, estimatedRoughness);
            reflectionAlpha = smoothstep(0.0, 0.2, 1.0 - scaled_uv.y);
        }
        
        reflectionAlpha *= pow(saturate(dot(-viewDir, r.direction)), 2.0);
        reflectionAlpha *= orientationIntensity;
        outReflection = float4(reflectionColor, reflectionAlpha);
    }
    
    void PS_Accumulate0(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
            discard;
        outBlended = ComputeTAA(input, sHistory1);
    }

    void PS_Accumulate1(VS_OUTPUT input, out float4 outBlended : SV_Target)
    {
        if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
            discard;
        outBlended = ComputeTAA(input, sHistory0);
    }
    
    void PS_Output(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Reflections Only
             
                    outColor = float4(GetActiveHistory(input.uv).rgb, 1.0);
                    return;
                case 2: // Normals
                    outColor = float4(SampleGBuffer(input.uv).rgb * 0.5 + 0.5, 1.0);
                    return;
                case 3: // Depth
                    outColor = SampleGBuffer(input.uv).aaaa;
                    return;
                case 4: // Motion Vectors
                { 
                        float2 m = GetMV(input.uv);
                        float v_mag = length(m) * 100.0;
                        float a = atan2(m.y, m.x);
                        float3 hsv_color = HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, 1.0));
                        float3 final_color = lerp(float3(0.5, 0.5, 0.5), hsv_color, saturate(v_mag));
                        outColor = float4(final_color, 1.0);
                        return;
                    }
            }
        }

        float3 color = GetColor(input.uv).rgb;
        float4 gbuffer = SampleGBuffer(input.uv);
        float depth = gbuffer.a;

        if (depth >= 1.0)
        {
            outColor = float4(color, 1.0);
            return;
        }
        
        float3 normal = gbuffer.rgb;
        float4 reflectionSample = GetActiveHistory(input.uv);

        float3 reflectionColor = reflectionSample.rgb;
        float reflectionMask = reflectionSample.a;
        // Colors Adjust
        reflectionColor = AdjustContrast(reflectionColor, g_Contrast);
        reflectionColor = AdjustSaturation(reflectionColor, g_Saturation);
        // PBR
        float2 pScale = input.pScale;
        float3 viewDir = -normalize(UVToViewPos(input.uv, depth, pScale));
        float VdotN = saturate(dot(viewDir, normal));
        
        float3 f0 = lerp(DIELECTRIC_REFLECTANCE.xxx, color, MetallicFactor);
        float3 F = F_Schlick(VdotN, f0);

        float3 finalColor;
        if (g_BlendMode == 0)
        {
            float3 kS = F;
            float effectiveIntensity = saturate(Intensity);
            float3 kD = (1.0 - kS * effectiveIntensity) * (1.0 - MetallicFactor * effectiveIntensity);
            float3 diffuseComponent = color * kD;
            float3 specularComponent = reflectionColor * kS * Intensity;
            finalColor = lerp(color, diffuseComponent + specularComponent, reflectionMask);
        }
        else // Blending.fxh
        {
            float blendAmount = dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask;
            finalColor = ComHeaders::Blending::Blend(g_BlendMode, color, reflectionColor, blendAmount * Intensity);
        }
    
        outColor = float4(finalColor, 1.0);
    }

    technique Barbatos_SSR
    {
        pass GenNormals
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_GenNormals;
            RenderTarget = Normal;
        }
        pass SmoothNormals_H
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_SmoothNormals_H;
            RenderTarget = Normal1;
        }
        pass SmoothNormals_V
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_SmoothNormals_V;
            RenderTarget = Normal;
        }
        pass CopyColorGenMips
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_CopyColor;
            RenderTarget = TexColorCopy;
        }
        pass TraceReflections
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_TraceReflections;
            RenderTarget = Reflection;
        }
        pass Accumulate0
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass Output
        {
            VertexShader = VS_Barbatos_SSR;
            PixelShader = PS_Output;
        }
    }
}
