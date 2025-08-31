/*-------------------|
| :: Description ::  |
'--------------------|

SGSR code is based on the Snapdragon™ Game Super Resolution. SPDX-License-Identifier: BSD-3-Clause
https://github.com/SnapdragonStudios/snapdragon-gsr/blob/main/sgsr/v1/include/hlsl/sgsr1_shader_mobile.hlsl
https://github.com/Blinue/Magpie/blob/dev/src/Effects/SGSR.hlsl

    Barbatos SSR_T

    Barbatos SSR, but focused for testing

    Version: 0.0.1
    Author: Barbatos
    License: MIT
*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "Blending.fxh"

//--------------------|
// :: Preprocessor :: |
//--------------------|

// Macros
static const float2 LOD_MASK = float2(0.0, 1.0);
static const float2 ZERO_LOD = float2(0.0, 0.0);
#define GetLod(s,c) tex2Dlod(s, ((c).xyyy * LOD_MASK.yyxx + ZERO_LOD.xxxy))
#define PI 3.1415927
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords))

static const int STEPS_PER_RAY_WALLS = 32;

#define fReflectFloorsIntensity 1
#define fReflectWallsIntensity 0
#define fReflectCeilingsIntensity 0

//----------|
// :: UI :: |
//----------|

uniform float SPIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Intensity";
> = 1.1;

uniform float FadeEnd <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.010;
    ui_category = "Reflection Settings";
    ui_label = "Fade Distance";
> = 4.999;

uniform float THICKNESS_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.02; ui_step = 0.001;
    ui_category = "Reflection Settings";
    ui_label = "Thickness";
> = 0.01;

uniform float BumpIntensity <
    ui_label = "Bump Mapping Intensity";
    ui_type = "drag";
    ui_category = "Surface & Normals";
    ui_min = 0.0; ui_max = 2.0;
> = 0.003;

uniform float OrientationThreshold <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Orientation";
    ui_label = "Orientation Threshold";
    ui_tooltip = "Controls the sensitivity for detecting floors, walls, and ceilings. Lower values are stricter.";
> = 0.90;

uniform float GeoCorrectionIntensity <
    ui_type = "drag";
    ui_min = -0.1; ui_max = 0.01;
    ui_step = 0.01;
    ui_category = "Orientation";
    ui_label = "Geometry Correction Intensity";
> = -0.01;

uniform bool EnableTemporal <
    ui_category = "Temporal Filtering";
    ui_label = "Enable Temporal Accumulation";
    ui_tooltip = "Blends the current frame's reflection with previous frames to reduce noise and flickering.";
> = true;

uniform float AccumFramesSG <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 32.0;
    ui_step = 1.0;
    ui_category = "Temporal Filtering";
    ui_label = "Temporal Accumulation Frames";
    ui_tooltip = "Number of frames to accumulate. Higher values are smoother but may cause more ghosting on moving objects.";
> = 2.0;

uniform int Quality <
    ui_type = "combo";
    ui_items = "Quality\0Performance\0";
    ui_category = "Performance & Quality";
    ui_label = "Quality Preset";
    ui_tooltip = "Choose between a higher quality 'Heavy' preset or a faster 'Light' preset.";
> = 0;

uniform float RenderScale <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_tooltip = "Renders reflections at a lower resolution for performance, then upscales using SGSR.";
> = 0.8;

uniform float EdgeSharpness <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0; ui_step = 0.1;
    ui_category = "SGSR Settings";
    ui_label = "Edge Sharpness";
> = 2.0;

uniform float EdgeThreshold <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 16.0; ui_step = 0.1;
    ui_category = "SGSR Settings";
    ui_label = "Edge Threshold";
> = 8.0;

uniform float VERTICAL_FOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
> = 37.0;

BLENDING_COMBO(BlendMode, "Blend Mode", "How the final reflections are blended with the original image.", "Blending & Output", false, 0, 6)

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0Motion Vectors\0Final Reflection\0Normals\0Depth\0Raw Low-Res Reflection\0Reflection Mask\0";
    ui_category = "Debug";
    ui_label = "Debug View Mode";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

sampler samplerColor
{
    Texture = ReShade::BackBufferTex;
    SRGBTexture = true;
};

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

namespace Barbatos_SSR_TEST
{
    texture Reflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sReflection
    {
        Texture = Reflection;
    };

    texture Temp
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sTemp
    {
        Texture = Temp;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU = Clamp;
        AddressV = Clamp;
    };

    texture History
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sHistory
    {
        Texture = History;
    };

    texture TNormal
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = TNormal;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };

    texture UpscaledReflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sUpscaledReflection
    {
        Texture = UpscaledReflection;
    };

//---------------|
// :: Structs :: |
//---------------|

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

//---------------|
// :: Utility :: |
//---------------|

    float3 GetColor(float2 uv)
    {
        return tex2D(samplerColor, uv).rgb;
    }

    float3 RGBToYCoCg(float3 rgb)
    {
        float Y = dot(rgb, float3(0.25, 0.5, 0.25));
        float Co = dot(rgb, float3(0.5, 0, -0.5));
        float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
        return float3(Y, Co, Cg);
    }

    float3 YCoCgToRGB(float3 ycocg)
    {
        float r = ycocg.x + ycocg.y - ycocg.z;
        float g = ycocg.x + ycocg.z;
        float b = ycocg.x - ycocg.y - ycocg.z;
        return float3(r, g, b);
    }
    
    float3 HSVToRGB(float3 c)
    {
        const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }

    float GetLuminance(float3 linearColor)
    {
        return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
    }

//------------------------------------|
// :: View Space & Normal Functions ::|
//------------------------------------|

    float3 UVToViewPos(float2 uv, float view_z)
    {
        float fov_rad = VERTICAL_FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;

        float2 clip_pos = uv * 2.0 - 1.0;

        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z;
        view_pos.z = view_z;

        return view_pos;
    }

    float2 ViewPosToUV(float3 view_pos)
    {
        float fov_rad = VERTICAL_FOV * (PI / 180.0);
        float proj_scale_y = 1.0 / tan(fov_rad * 0.5);
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;

        float2 clip_pos;
        clip_pos.x = view_pos.x * proj_scale_x / view_pos.z;
        clip_pos.y = -view_pos.y * proj_scale_y / view_pos.z;

        return clip_pos * 0.5 + 0.5;
    }

    float3 GVPFUV(float2 uv)
    {
        float depth = GetDepth(uv);
        return UVToViewPos(uv, depth);
    }

    float3 Normal(float2 texcoord)
    {
        float3 offset_x = GVPFUV(texcoord + float2(ReShade::PixelSize.x, 0.0));
        float3 offset_y = GVPFUV(texcoord + float2(0.0, ReShade::PixelSize.y));
        float3 center = GVPFUV(texcoord);

        float3 dx = center - offset_x;
        float3 dy = center - offset_y;

        return normalize(cross(dx, dy));
    }

    float3 ApplyBumpMapping(float2 texcoord, float3 normal)
    {
        if (BumpIntensity == 0.0)
            return normal;

        float h_c = GetLuminance(tex2D(ReShade::BackBuffer, texcoord).rgb);
        float h_x = GetLuminance(tex2Doffset(ReShade::BackBuffer, texcoord, int2(1, 0)).rgb);
        float h_y = GetLuminance(tex2Doffset(ReShade::BackBuffer, texcoord, int2(0, 1)).rgb);

        float2 slope = float2(h_x - h_c, h_y - h_c) * BumpIntensity * 100.0;

        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T = normalize(cross(up, normal));
        float3 B = cross(normal, T);
        float3 bumpedNormal = normal + T * slope.x - B * slope.y;

        return normalize(bumpedNormal);
    }

    float3 GeometryCorrection(float2 texcoord, float3 normal)
    {
        if (GeoCorrectionIntensity == 0.0)
            return normal;

        float lumCenter = GetLuminance(GetColor(texcoord).rgb);
        float lumRight = GetLuminance(tex2Doffset(samplerColor, texcoord, int2(1, 0)).rgb);
        float lumDown = GetLuminance(tex2Doffset(samplerColor, texcoord, int2(0, 1)).rgb);

        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));

        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }
    
    float2 SampleMotionVectors(float2 texcoord)
    {
        return GetLod(sTexMotionVectorsSampler, texcoord).rg;
    }

//-------------------|
// :: Ray Tracing  ::|
//-------------------|

    HitResult TraceRay(Ray r, int num_steps)
    {
        HitResult result;
        result.found = false;

        float max_trace_dist, step_scale, min_step_size, max_step_size;
        int refinement_steps;

        if (Quality == 1)
        {
            max_trace_dist = 2.0;
            refinement_steps = 5;
            step_scale = 0.7;
            min_step_size = 0.001;
            max_step_size = 1.0;
        }
        else 
        {
            max_trace_dist = 2.0;
            refinement_steps = 5;
            step_scale = 0.1;
            min_step_size = 0.0001;
            max_step_size = 1.0;
        }

        float stepSize = min_step_size;
        float totalDist = 0.0;
        float3 prevPos = r.origin;

        [loop]
        for (int i = 0; i < num_steps; ++i)
        {
            float3 currPos = prevPos + r.direction * stepSize;
            totalDist += stepSize;

            float2 uvCurr = ViewPosToUV(currPos);
            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) || totalDist > max_trace_dist)
                break;

            float sceneDepth = GetDepth(uvCurr);
            float thickness = abs(currPos.z - sceneDepth);

            if (currPos.z < sceneDepth || thickness > THICKNESS_THRESHOLD)
            {
                prevPos = currPos;
                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * step_scale, min_step_size, max_step_size);
                continue;
            }

            float3 lo = prevPos, hi = currPos;
            [unroll]
            for (int ref_step = 0; ref_step < refinement_steps; ++ref_step)
            {
                float3 mid = 0.5 * (lo + hi);
                float midDepth = GetDepth(ViewPosToUV(mid));
                if (mid.z >= midDepth)
                    hi = mid;
                else
                    lo = mid;
            }
            result.viewPos = hi;
            result.uv = ViewPosToUV(result.viewPos);
            result.found = true;
            return result;
        }
        return result;
    }

//-----------|
// :: SGSR ::|
//-----------|

#define UseEdgeDirection

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
        float g_tl = tex2D(sTemp, p).g;
        float g_tr = tex2Doffset(sTemp, p, int2(1, 0)).g;
        float g_bl = tex2Doffset(sTemp, p, int2(0, 1)).g;
        float g_br = tex2Doffset(sTemp, p, int2(1, 1)).g;
        return float4(g_tr, g_br, g_bl, g_tl);
    }
    
    float3 SgsrYuvH(float2 uv, float4 con1)
    {
        float3 pix;
        float edgeThreshold = EdgeThreshold / 255.0;
        float edgeSharpness = EdgeSharpness;
        
        pix = tex2D(sTemp, uv * RenderScale).xyz;

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

    void PS_GBuffer(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outNormal = 0;
            return;
        }

        float2 source_uv = uv / RenderScale;

        float3 normal = Normal(source_uv);
        float depth = GetDepth(source_uv);

        // Apply bump mapping
        normal = ApplyBumpMapping(source_uv, normal);
        normal = GeometryCorrection(source_uv, normal);

        outNormal.rgb = normal * 0.5 + 0.5;
        outNormal.a = depth;
    }

    void PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outReflection : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outReflection = 0;
            return;
        }

        float2 scaled_uv = uv / RenderScale;

        float depth = GetDepth(scaled_uv);
        if (depth >= 1.0)
        {
            outReflection = 0;
            return;
        }

        float3 viewPos = UVToViewPos(scaled_uv, depth);
        float3 viewDir = -normalize(viewPos);

        float3 normal = (GetLod(sNormal, uv).xyz - 0.5) * 2.0;
        normal = normalize(normal);

        bool isFloor = normal.y > OrientationThreshold;
        bool isCeiling = normal.y < -OrientationThreshold;
        bool isWall = abs(normal.y) <= OrientationThreshold;

        float orientationIntensity = (isFloor * fReflectFloorsIntensity) +
                                     (isWall * fReflectWallsIntensity) +
                                     (isCeiling * fReflectCeilingsIntensity);

        if (orientationIntensity <= 0.0)
        {
            outReflection = 0;
            return;
        }

        float3 eyeDir = -viewDir;

        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(eyeDir, normal));
        r.origin += r.direction * 0.0001;

        HitResult hit;
        if (isWall)
        {
            hit = TraceRay(r, STEPS_PER_RAY_WALLS);
        }
        else
        {
            int steps_per_ray = (Quality == 1) ? 128 : 256;
            hit = TraceRay(r, steps_per_ray);
        }

        float3 reflectionColor = 0;
        float reflectionAlpha = 0.0;
        float fresnel = pow(1.0 - saturate(dot(eyeDir, normal)), 3.0);

        if (hit.found)
        {
            reflectionColor = GetColor(hit.uv).rgb;
            float max_trace_dist = 2.0;
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / max_trace_dist);
            float fadeRange = max(FadeEnd, 0.001);
            float depthFade = saturate((FadeEnd - depth) / fadeRange);
            depthFade *= depthFade;

            reflectionAlpha = distFactor * depthFade * fresnel;
        }
        else // Fallback for missed rays
        {
            float adaptiveDist = depth * 1.2 + 0.012;
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos));
            reflectionColor = GetColor(uvFb).rgb;
            float vertical_fade = pow(saturate(1.0 - scaled_uv.y), 2.0);
            reflectionAlpha = fresnel * vertical_fade;
        }

        float angleWeight = pow(saturate(dot(-viewDir, r.direction)), 2.0);
        reflectionAlpha *= angleWeight;
        reflectionAlpha *= orientationIntensity;

        outReflection = float4(reflectionColor, reflectionAlpha);
    }

    void PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outBlended : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outBlended = 0;
            return;
        }

        float4 currentSpec = GetLod(sReflection, uv);

        if (!EnableTemporal)
        {
            outBlended = currentSpec;
            return;
        }
        
        float2 full_res_uv = uv / RenderScale;
        
        float2 motion = SampleMotionVectors(full_res_uv);
        float2 reprojected_uv_full = full_res_uv + motion;

        float currentDepth = GetDepth(full_res_uv);
        float historyDepth = GetDepth(reprojected_uv_full);

        float2 reprojected_uv_low = reprojected_uv_full * RenderScale;

        bool validHistory = all(saturate(reprojected_uv_low) == reprojected_uv_low) &&
                                      FRAME_COUNT > 1 &&
                                      abs(historyDepth - currentDepth) < 0.01;

        float4 blendedSpec = currentSpec;
        if (validHistory)
        {
            float4 historySpec = GetLod(sHistory, reprojected_uv_low);

            float3 minBox = RGBToYCoCg(currentSpec.rgb), maxBox = minBox;
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    
                    float2 neighbor_uv = uv + float2(x, y) * ReShade::PixelSize;
                    float3 neighborSpec = RGBToYCoCg(GetLod(sReflection, neighbor_uv).rgb);
                    minBox = min(minBox, neighborSpec);
                    maxBox = max(maxBox, neighborSpec);
                }
            }
            
            float3 clampedHistorySpec = clamp(RGBToYCoCg(historySpec.rgb), minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, AccumFramesSG);
            blendedSpec.rgb = YCoCgToRGB(lerp(clampedHistorySpec, RGBToYCoCg(currentSpec.rgb), alpha));
            blendedSpec.a = lerp(historySpec.a, currentSpec.a, alpha);
        }
        
        outBlended = blendedSpec;
    }

    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outHistory : SV_Target)
    {
        outHistory = GetLod(sTemp, uv);
    }

    void PS_Upscale_SGSR(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        if (RenderScale >= 1.0)
        {
            outColor = GetLod(sTemp, uv);
            return;
        }
       
        float2 inputSize = BUFFER_SCREEN_SIZE * RenderScale;
        float2 inputPixelSize = 1.0 / inputSize;
        float4 con1 = float4(inputPixelSize, inputSize);
        
        float3 sgsrColor = SgsrYuvH(uv, con1);
        float2 scaled_uv = uv * RenderScale;
        float2 texel_size = ReShade::PixelSize;
        float2 uv_floored = floor(scaled_uv / texel_size) * texel_size;
        float2 f = frac(scaled_uv / texel_size);

        float a_tl = tex2Dlod(sTemp, float4(uv_floored, 0, 0)).a;
        float a_tr = tex2Dlod(sTemp, float4(uv_floored + float2(texel_size.x, 0), 0, 0)).a;
        float a_bl = tex2Dlod(sTemp, float4(uv_floored + float2(0, texel_size.y), 0, 0)).a;
        float a_br = tex2Dlod(sTemp, float4(uv_floored + texel_size, 0, 0)).a;

        float a_top = lerp(a_tl, a_tr, f.x);
        float a_bot = lerp(a_bl, a_br, f.x);
        float final_alpha = lerp(a_top, a_bot, f.y);

        outColor = float4(sgsrColor, final_alpha);
    }

    void PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        // Debug 
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float2 motion = SampleMotionVectors(uv);
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / (2.0 * PI)) + 0.5, 1.0, saturate(velocity));
                    outColor = float4(HSVToRGB(hsv), 1.0);
                    return;
                case 2: // Final Reflection (Upscaled)
                    outColor = float4(GetLod(sUpscaledReflection, uv).rgb, 1.0);
                    return;
                case 3: // Normals
                    float3 normal = (GetLod(sNormal, uv).xyz - 0.5) * 2.0;
                    outColor = float4(normalize(normal) * 0.5 + 0.5, 1.0);
                    return;
                case 4: // Depth
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5: // Raw Low-Res Reflection
                    outColor = float4(GetLod(sReflection, uv).rgb, 1.0);
                    return;
                case 6: // Reflection Mask
                    outColor = GetLod(sUpscaledReflection, uv).aaaa;
                    return;
            }
        }

        float3 originalColor = GetColor(uv);

        if (GetDepth(uv) >= 1.0)
        {
            outColor = float4(originalColor, 1.0);
            return;
        }

        float4 reflectionSample = GetLod(sUpscaledReflection, uv);
        float3 specularGI = reflectionSample.rgb;
        float blendFactor = reflectionSample.a;

        specularGI *= SPIntensity;

        float3 finalColor = ComHeaders::Blending::Blend(BlendMode, originalColor.rgb, specularGI, blendFactor);
        
        outColor = float4(finalColor, 1.0);
    }

    technique Barbatos_SSR_TEST
    {
        pass GBuffer
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GBuffer;
            RenderTarget = TNormal;
            ClearRenderTargets = true;
        }
        pass TraceReflections
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_TraceReflections;
            RenderTarget = Reflection;
            ClearRenderTargets = true;
        }
        pass Accumulate
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Accumulate;
            RenderTarget = Temp;
        }
        pass UpdateHistory
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateHistory;
            RenderTarget = History;
        }
        pass Upscale_SGSR
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale_SGSR;
            RenderTarget = UpscaledReflection;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
            SRGBWriteEnable = true;
        }
    }
}
