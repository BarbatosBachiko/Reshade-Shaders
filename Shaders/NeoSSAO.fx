/*------------------.
| :: Description :: |
'-------------------/

 NeoSSAO
                                                                       
    Version 1.9
    Author: Barbatos Bachiko
    License: MIT
    Smooth Normals use AlucardDH MIT License : https://github.com/AlucardDH/dh-reshade-shaders-mit/blob/master/LICENSE

    About: Screen-Space Ambient Occlusion using ray marching.
    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility
    
    Version 1.9
    + Quality of Life
*/ 

#include "ReShade.fxh"

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif
#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define S_PC MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp;AddressW=Clamp;

#define getDepth(coords)      (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define fov 28.6
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE

#define MVErrorTolerance 0.96
#define SkyDepth 0.99
#define MAX_Frames 64
#define UseEdgeDirection 

static const float PI2div360 = 0.01745329;
#define rad(x) (x * PI2div360)

//----------|
// :: UI :: |
//----------|

uniform float Intensity <
    ui_category = "General";
    ui_type = "drag";
    ui_label = "AO Intensity";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform float MaxRayDistance <
    ui_type = "slider";
    ui_category = "Ray Marching";
    ui_label = "Max Ray Distance";
    ui_tooltip = "Maximum distance for ray marching";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
> = 0.015;

uniform float FadeStart <
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade Start";
    ui_tooltip = "Distance at which AO starts to fade out.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform float FadeEnd <
    ui_category = "Fade";
    ui_type = "slider";
    ui_label = "Fade End";
    ui_tooltip = "Distance at which AO completely fades out.";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 1.0;

uniform bool EnableTemporal <
    ui_category = "Temporal";
    ui_type = "checkbox";
    ui_label = "Temporal Filtering";
> = true;

uniform float AccumFrames <
    ui_type = "slider";
    ui_category = "Temporal";
    ui_label = "AO Temporal";
    ui_min = 1.0; ui_max = 16.0; ui_step = 1.0;
> = 4.0;

uniform float DepthMultiplier <
    ui_type = "slider";
    ui_category = "Depth/Normals";
    ui_label = "Depth Multiplier";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.1;
> = 1.0;

uniform float DepthThreshold <
    ui_type = "slider";
    ui_category = "Depth/Normals";
    ui_label = "Sky Threshold";
    ui_tooltip = "Set the depth threshold to ignore the sky.";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.99;

uniform bool bSmoothNormals <
    ui_category = "Depth/Normals";
    ui_label = "Smooth Normals";
> = false;

uniform float BrightnessThreshold <
    ui_category = "Visibility";
    ui_type = "slider";
    ui_label = "Brightness Threshold";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform float4 OcclusionColor <
    ui_category = "Extra";
    ui_type = "color";
    ui_label = "Occlusion Color";
    ui_tooltip = "Select the color for ambient occlusion.";
> = float4(0.0, 0.0, 0.0, 1.0);

// SGSR Settings
uniform float RenderScale <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.01;
    ui_category = "SGSR Settings";
    ui_label = "Render Scale";
    ui_tooltip = "Renders AO at a lower resolution for performance, then upscales using SGSR.";
> = 1.0;

uniform float EdgeSharpness <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "SGSR Settings";
    ui_label = "Edge Sharpness";
> = 0.0;

uniform float EdgeThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 16.0; ui_step = 0.1;
    ui_category = "SGSR Settings";
    ui_label = "Edge Threshold";
> = 8.0;

uniform int ViewMode <
    ui_category = "Debug";
    ui_type = "combo";
    ui_label = "View Mode";
    ui_tooltip = "Select the view mode for SSAO";
    ui_items = "None\0AO Debug\0Depth\0Sky Debug\0Normal Debug\0Raw Low-Res AO\0";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;
static const float SampleRadius = 1.0;
static const int SampleCount = 8;
static const float RayScale = 0.222;
static const float EnableBrightnessThreshold = true;
static const float DepthSmoothEpsilon = 0.0003;

    /*---------------.
    | :: Textures :: |
    '---------------*/

#if USE_MARTY_LAUNCHPAD_MOTION
namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
}
float2 sampleMotion(float2 texcoord) {
    return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
}

#elif USE_VORT_MOTION
texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sMotVectTexVort { Texture = MotVectTexVort; S_PC };
float2 sampleMotion(float2 texcoord) {
    return GetLod(sMotVectTexVort, texcoord).rg;
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
    Texture = texMotionVectors;S_PC
};
float2 sampleMotion(float2 texcoord)
{
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace NEOSPACEAO
{
    texture2D AO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };

    texture2D TEMP
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };

    texture2D HISTORY
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };

    sampler2D sAO
    {
        Texture = AO;
        SRGBTexture = false;
    };

    sampler2D sTEMP
    {
        Texture = TEMP;
        SRGBTexture = false;
    };

    sampler2D sHISTORY
    {
        Texture = HISTORY;
        SRGBTexture = false;
    };

    texture normalTex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sNormal
    {
        Texture = normalTex;S_PC
    };

    texture UpscaledAO
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sUpscaledAO
    {
        Texture = UpscaledAO;
    };

    // :: Functions ::
    
    float3 ClipToAABB(float3 aabb_min, float3 aabb_max, float3 q)
    {
        float3 center = 0.5 * (aabb_max + aabb_min);
        float3 extents = 0.5 * (aabb_max - aabb_min);
        extents = max(extents, 1.0 / 255.0);

        float3 v = q - center;
        float max_dist = max(abs(v.x / extents.x), max(abs(v.y / extents.y), abs(v.z / extents.z)));

        if (max_dist > 1.0)
        {
            return center + v / max_dist;
        }

        return q;
    }

    float lum(float3 color)
    {
        return (color.r + color.g + color.b) * 0.3333333;
    }

    float3 UVtoPos(float2 texcoord, float depth)
    {
        float3 scrncoord = float3(texcoord.xy * 2 - 1, depth * FAR_PLANE);
        scrncoord.xy *= scrncoord.z;
        scrncoord.x *= ReShade::AspectRatio;
        scrncoord *= rad(fov);
        return scrncoord.xyz;
    }
    float3 UVtoPos(float2 texcoord)
    {
        return UVtoPos(texcoord, getDepth(texcoord));
    }

    float2 PostoUV(float3 position)
    {
        float2 scrnpos = position.xy;
        scrnpos /= rad(fov);
        scrnpos.x /= ReShade::AspectRatio;
        scrnpos /= position.z;
        return scrnpos / 2 + 0.5;
    }

    float3 computeNormal(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float3 u, d, l, r;
        u = UVtoPos(texcoord + float2(0, p.y));
        d = UVtoPos(texcoord - float2(0, p.y));
        l = UVtoPos(texcoord + float2(p.x, 0));
        r = UVtoPos(texcoord - float2(p.x, 0));
        float3 c = UVtoPos(texcoord);
        float3 v = u - c;
        float3 h = r - c;
        if (abs(d.z - c.z) < abs(u.z - c.z))
            v = c - d;
        if (abs(l.z - c.z) < abs(r.z - c.z))
            h = c - l;
        return normalize(cross(v, h));
    }

    // SmoothNormal by AlucardDH MIT Licence
    float3 GetNormal(float2 texcoord)
    {
        float3 offset = float3(ReShade::PixelSize, 0.0);
        float3 normal = computeNormal(texcoord);

        if (bSmoothNormals)
        {
            float2 offset2 = ReShade::PixelSize * 7.5 * (1.0 - getDepth(texcoord));

            float3 normalTop = computeNormal(texcoord - float2(0, offset2.y));
            float3 normalBottom = computeNormal(texcoord + float2(0, offset2.y));
            float3 normalLeft = computeNormal(texcoord - float2(offset2.x, 0));
            float3 normalRight = computeNormal(texcoord + float2(offset2.x, 0));

            float weightTop = smoothstep(1, 0, distance(normal, normalTop) * 1.5) * 2;
            float weightBottom = smoothstep(1, 0, distance(normal, normalBottom) * 1.5) * 2;
            float weightLeft = smoothstep(1, 0, distance(normal, normalLeft) * 1.5) * 2;
            float weightRight = smoothstep(1, 0, distance(normal, normalRight) * 1.5) * 2;

            float4 weightedNormal =
                float4(normal, 1.0) +
                float4(normalTop * weightTop, weightTop) +
                float4(normalBottom * weightBottom, weightBottom) +
                float4(normalLeft * weightLeft, weightLeft) +
                float4(normalRight * weightRight, weightRight);

            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }
        return normal;
    }

    float3 getNormal(float2 coords)
    {
        float3 normal = -(GetLod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2;
        return normalize(normal);
    }

    float RayMarching(in float2 texcoord, in float3 rayDir)
    {
        float occlusion = 0.0;
        float depthValue = getDepth(texcoord);
        float3 normal = getNormal(texcoord);

        float stepSize = ReShade::PixelSize.x / RayScale;
        int numSteps = max(int(MaxRayDistance / stepSize), 2);

        [loop]
        for (int i = 0; i < numSteps; i++)
        {
            float t = float(i) * rcp(float(numSteps - 1));
            float sampleDistance = mad(t, t * MaxRayDistance, 0.0);
            float2 sampleCoord = mad(rayDir.xy, sampleDistance, texcoord);

            sampleCoord = clamp(sampleCoord, 0.0, 1.0);

            float sampleDepth = getDepth(sampleCoord);
            float depthDiff = depthValue - sampleDepth;
            float hitFactor = saturate(depthDiff * rcp(DepthSmoothEpsilon + 1e-6));

            if (hitFactor > 0.01)
            {
                float angleFactor = saturate(dot(normal, rayDir));
                float weight = (1.0 - (sampleDistance / MaxRayDistance)) * hitFactor * angleFactor;
                occlusion += weight;
                if (hitFactor < 0.001)
                    break;
            }
        }

        return occlusion;
    }

    static const float3 hemisphereSamples[12] =
    {
        float3(0.5381, 0.1856, 0.4319), float3(0.1379, 0.2486, 0.6581),
        float3(-0.3371, 0.5679, 0.6981), float3(-0.7250, 0.4233, 0.5429),
        float3(-0.4571, -0.5329, 0.7116), float3(0.0649, -0.9270, 0.3706),
        float3(0.3557, -0.6380, 0.6827), float3(0.6494, -0.2861, 0.7065),
        float3(0.7969, 0.5845, 0.1530), float3(-0.0195, 0.8512, 0.5234),
        float3(-0.5890, -0.7287, 0.3487), float3(-0.6729, 0.2057, 0.7117)
    };

    float2 GetMotionVector(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float2 MV = sampleMotion(texcoord);

        if (MVErrorTolerance < 1)
        {
            if (abs(MV.x) < p.x && abs(MV.y) < p.y)
                MV = 0;
        }

#if USE_MARTY_LAUNCHPAD_MOTION
        MV = GetLod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).xy;
#elif USE_VORT_MOTION
        MV = GetLod(sMotVectTexVort, float4(texcoord, 0, 0)).xy;
#endif

        return MV;
    }
    
//-----------|
// :: SGSR ::|
//-----------|
    float fastLanczos2(float x)
    {
        float wA = x - float(4.0);
        float wB = x * wA - wA;
        wA *= wA;
        return wB * wA;
    }

#if defined(UseEdgeDirection)
    float2 weightY(float dx, float dy, float c, float3 data)
#else
    float2 weightY(float dx, float dy, float c, float data)
#endif
    {
#if defined(UseEdgeDirection)
        float std = data.x;
        float2 dir = data.yz;

        float edgeDis = ((dx * dir.y) + (dy * dir.x));
        float x = (((dx * dx) + (dy * dy)) + ((edgeDis * edgeDis) * ((clamp(((c * c) * std), 0.0, 1.0) * 0.7) + -1.0)));
#else
        float std = data;
        float x = ((dx*dx)+(dy* dy))* float(0.5) + clamp(abs(c)*std, 0.0, 1.0);
#endif

        float w = fastLanczos2(x);
        return float2(w, w * c);
    }

    float2 edgeDirection(float4 left, float4 right)
    {
        float2 dir;
        float RxLz = (right.x + (-left.z));
        float RwLy = (right.w + (-left.y));
        float2 delta;
        delta.x = (RxLz + RwLy);
        delta.y = (RxLz + (-RwLy));
        float lengthInv = rsqrt((delta.x * delta.x + 3.075740e-05) + (delta.y * delta.y));
        dir.x = (delta.x * lengthInv);
        dir.y = (delta.y * lengthInv);
        return dir;
    }

    float4 SGSRH(float2 p)
    {
        float g_tl = GetLod(sTEMP, p).g;
        float g_tr = tex2Doffset(sTEMP, p, int2(1, 0)).g;
        float g_bl = tex2Doffset(sTEMP, p, int2(0, 1)).g;
        float g_br = tex2Doffset(sTEMP, p, int2(1, 1)).g;
        return float4(g_tr, g_br, g_bl, g_tl);
    }

    float3 SgsrYuvH(float2 uv, float4 con1)
    {
        float edgeThreshold = EdgeThreshold / 255.0;
        float edgeSharpness = EdgeSharpness;

        float3 pix = GetLod(sTEMP, uv * RenderScale).rrr;

        float2 imgCoord = (uv.xy * con1.zw) - 0.5;
        float2 imgCoordPixel = floor(imgCoord);

        float2 coord_corner = imgCoordPixel * con1.xy;
        float2 coord_center = coord_corner + (0.5 * con1.xy);
        float2 pl = frac(imgCoord);
        float4 left = SGSRH(coord_center * RenderScale);

        float edgeVote = abs(left.z - left.y) + abs(pix[1] - left.y) + abs(pix[1] - left.z);
        if (edgeVote > edgeThreshold)
        {
            float2 right_coord_center = coord_center + float2(con1.x, 0.0);
            float4 right = SGSRH(right_coord_center * RenderScale);

            float2 up_coord_center = coord_center + float2(0.0, -con1.y);
            float2 down_coord_center = coord_center + float2(0.0, con1.y);

            float4 upDown;
            upDown.xy = SGSRH(up_coord_center * RenderScale).wz;
            upDown.zw = SGSRH(down_coord_center * RenderScale).yx;

            float mean = (left.y + left.z + right.x + right.w) * float(0.25);
            left = left - float4(mean, mean, mean, mean);
            right = right - float4(mean, mean, mean, mean);
            upDown = upDown - float4(mean, mean, mean, mean);
            float pix_G = pix[1] - mean;

            float sum = (((((abs(left.x) + abs(left.y)) + abs(left.z)) + abs(left.w)) + (((abs(right.x) + abs(right.y)) + abs(right.z)) + abs(right.w))) + (((abs(upDown.x) + abs(upDown.y)) + abs(upDown.z)) + abs(upDown.w)));
            float sumMean = 1.014185e+01 / sum;
            float std = (sumMean * sumMean);

#if defined(UseEdgeDirection)
            float3 data = float3(std, edgeDirection(left, right));
#else
            float data = std;
#endif

            float2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, data);
            aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, data);
            aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, data);
            aWY += weightY(pl.x, pl.y - 2.0, upDown.w, data);
            aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, data);
            aWY += weightY(pl.x, pl.y - 1.0, left.y, data);
            aWY += weightY(pl.x, pl.y, left.z, data);
            aWY += weightY(pl.x + 1.0, pl.y, left.w, data);
            aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, data);
            aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, data);
            aWY += weightY(pl.x - 2.0, pl.y, right.z, data);
            aWY += weightY(pl.x - 1.0, pl.y, right.w, data);

            float finalY = aWY.y / aWY.x;
            float max4 = max(max(left.y, left.z), max(right.x, right.w));
            float min4 = min(min(left.y, left.z), min(right.x, right.w));
            finalY = clamp(edgeSharpness * finalY, min4, max4);

            float deltaY = finalY - pix_G;
            pix = saturate(pix + deltaY);
        }
        return pix;
    }

//--------------------|
// :: Pixel Shaders ::|
//--------------------|
    
    float4 PS_SSAO(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            discard;
        float2 scaled_uv = uv / RenderScale;

        float depthValue = getDepth(scaled_uv);
        float3 normal = getNormal(scaled_uv);
        float3 originalColor = GetColor(scaled_uv).rgb;

        float brightness = lum(originalColor);
        float brightnessFactor = 1.0;
        if (EnableBrightnessThreshold)
        {
            brightnessFactor = saturate(1.0 - smoothstep(BrightnessThreshold - 0.1, BrightnessThreshold + 0.1, brightness));
        }

        float3 tangent = normalize(abs(normal.z) < 0.999 ? cross(normal, float3(0.0, 0.0, 1.0)) : float3(1.0, 0.0, 0.0));
        float3 bitangent = cross(normal, tangent);
        float3x3 TBN = float3x3(tangent, bitangent, normal);

        int sampleCount = clamp(SampleCount, 1, 12);
        float occlusion = 0.0;

        for (int i = 0; i < sampleCount; i++)
        {
            float3 sampleDir = mul(TBN, hemisphereSamples[i]);
            occlusion += RayMarching(scaled_uv, sampleDir * SampleRadius);
        }

        occlusion = (occlusion / sampleCount) * Intensity;
        occlusion *= brightnessFactor;
        float fade = (depthValue < FadeStart) ? 1.0 : saturate((FadeEnd - depthValue) / (FadeEnd - FadeStart));
        occlusion *= fade;

        return float4(occlusion, occlusion, occlusion, occlusion);
    }

    float4 PS_ApplyTemporalFilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            return 0;
        float2 full_res_uv = uv / RenderScale;

        float4 currentSpec = GetLod(sAO, uv);
        
        if (!EnableTemporal)
        {
            return currentSpec;
        }
        
        float currentDepth = getDepth(full_res_uv);

        float2 motion = GetMotionVector(full_res_uv);
        float2 reprojectedUV_full = full_res_uv + motion;
        float2 reprojectedUV_low = reprojectedUV_full * RenderScale;

        float historyDepth = getDepth(reprojectedUV_full);
        bool validHistory = all(saturate(reprojectedUV_low) == reprojectedUV_low) &&
                            FRAME_COUNT > 1 &&
                            abs(historyDepth - currentDepth) < 0.01;

        float4 blendedSpec = currentSpec;
        if (validHistory)
        {
            float4 historySpec = GetLod(sHISTORY, float4(reprojectedUV_low, 0, 0));

            float3 minBox = currentSpec.rgb, maxBox = currentSpec.rgb;
            const int2 offsets[8] = { int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1) };
            float2 low_res_pixel_size = ReShade::PixelSize / RenderScale;

            [unroll]
            for (int i = 0; i < 8; i++)
            {
                float2 neighbor_uv = uv + offsets[i] * low_res_pixel_size;
                float3 neighborSpec = tex2Dlod(sAO, float4(neighbor_uv, 0, 0)).rgb;
                minBox = min(minBox, neighborSpec);
                maxBox = max(maxBox, neighborSpec);
            }
            float3 center = (minBox + maxBox) * 0.5;
            float3 extents = (maxBox - minBox) * 0.5;
            extents += 0.01;
            minBox = center - extents;
            maxBox = center + extents;

            float3 historyRGB = historySpec.rgb;
            float3 processedHistoryRGB = ClipToAABB(minBox, maxBox, historyRGB);
            
            float alpha = 1.0 / min((float) FRAME_COUNT, AccumFrames);
            
            float rejection_dist = distance(historyRGB, processedHistoryRGB);
            float rejection_factor = saturate(rejection_dist * 8.0);
            alpha = max(alpha, rejection_factor);

            blendedSpec.rgb = lerp(processedHistoryRGB, currentSpec.rgb, alpha);
            blendedSpec.a = lerp(historySpec.a, currentSpec.a, alpha);
        }
        
        return blendedSpec;
    }

    float4 PS_Normals(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float3 normal = GetNormal(uv);
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float4 PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (any(uv > RenderScale))
            return 0;

        return GetLod(sTEMP, float4(uv, 0, 0));
    }
    
    float4 PS_Upscale_SGSR(float4 vpos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        if (RenderScale >= 1.0)
        {
            return GetLod(sTEMP, float4(uv, 0, 0));
        }

        float2 inputSize = BUFFER_SCREEN_SIZE * RenderScale;
        float2 inputPixelSize = 1.0 / inputSize;
        float4 con1 = float4(inputPixelSize, inputSize);

        float3 sgsrColor = SgsrYuvH(uv, con1);
        
        return float4(sgsrColor, 1.0);
    }

    // Final Image
    float4 PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
    {
        float4 originalColor = GetColor(uv);

        float depthValue = getDepth(uv);
        float3 normal = getNormal(uv);

        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // AO Debug
                {
                        float ao = GetLod(sUpscaledAO, float4(uv, 0, 0)).r;
                        return float4(1.0 - ao, 1.0 - ao, 1.0 - ao, 1.0);
                    }
                case 2: // Depth
                    return float4(depthValue, depthValue, depthValue, 1.0);

                case 3: // Sky Debug
                    return (depthValue >= DepthThreshold)
                        ? float4(1.0, 0.0, 0.0, 1.0)
                        : float4(depthValue, depthValue, depthValue, 1.0);

                case 4: // Normal Debug
                    return float4(normal * 0.5 + 0.5, 1.0);
                case 5: // Raw Low-Res AO
                    return GetLod(sAO, float4(uv, 0, 0));
            }
        }
        
        if (depthValue >= DepthThreshold)
        {
            return originalColor;
        }

        float occlusion;
        
        if (RenderScale < 1.0)
        {
            occlusion = GetLod(sUpscaledAO, float4(uv, 0, 0)).r;
        }
        else if (EnableTemporal)
        {
            occlusion = GetLod(sTEMP, float4(uv, 0, 0)).r;
        }
        else
        {
            occlusion = GetLod(sAO, float4(uv, 0, 0)).r;
        }

        return originalColor * (1.0 - saturate(occlusion)) + OcclusionColor * saturate(occlusion);
    }

    technique NeoSSAO
    <
        ui_tooltip = "RT Ambient Occlusion";
    >
    {
        pass NormalPass
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Normals;
            RenderTarget = normalTex;
        }
        pass SSAO
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SSAO;
            RenderTarget = AO;
            ClearRenderTargets = true;
        }
        pass Temporal
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_ApplyTemporalFilter;
            RenderTarget = TEMP;
            ClearRenderTargets = true;
        }
        pass UpdateHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = HISTORY;
        }
        pass Upscale_SGSR
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale_SGSR;
            RenderTarget = UpscaledAO;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
