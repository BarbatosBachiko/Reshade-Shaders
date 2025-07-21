/*--------------------.
| :: Description ::  |
'--------------------/

    SSRT - Screen Space Ray Traced Reflections

    Version: 1.6.92
    Author: Barbatos Bachiko & Gemini
    Original SSRT by jebbyk: https://github.com/jebbyk/SSRT-for-reshade/

    License: GNU Affero General Public License v3.0
    (https://github.com/jebbyk/SSRT-for-reshade/blob/main/LICENSE)

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 1.6.92
    + Readibility
*/

#include "ReShade.fxh"

//-------------------------------------------|
// :: Preprocessor Definitions & Constants ::|
//-------------------------------------------|

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)

// Ray Marching Constants
#define MAX_TRACE_DISTANCE 2.0
static const int STEPS_PER_RAY = 128;
static const int REFINEMENT_STEPS = 5;

// Adaptive Step Constants
#define STEP_SCALE 0.7
#define MIN_STEP_SIZE 0.001
#define MAX_STEP_SIZE 1.0

//----------|
// :: UI :: |
//----------|

uniform float SPIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Intensity";
    ui_tooltip = "Controls the overall brightness and intensity of the reflections.";
> = 1.1;

uniform float NormalBias <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Normal Bias";
    ui_tooltip = "Prevents self-reflection artifacts by comparing the hit surface normal with the origin surface normal.";
> = 0.2;

uniform float FadeStart <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Fade Start";
    ui_tooltip = "Distance at which reflections begin to fade out.";
> = 0.0;

uniform float FadeEnd <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0;
    ui_step = 0.010;
    ui_category = "Reflection Settings";
    ui_label = "Fade End";
    ui_tooltip = "Distance at which reflections completely disappear.";
> = 4.999;

uniform float BumpIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Surface & Normals";
    ui_label = "Bump Mapping Intensity";
    ui_tooltip = "Add texture to reflections.";
> = 0.002;

uniform float GeoCorrectionIntensity <
    ui_type = "drag";
    ui_min = -0.070; ui_max = 0.150;
    ui_step = 0.001;
    ui_category = "Surface & Normals";
    ui_label = "Geometry Correction";
> = 0.009;

uniform bool bSmoothNormals <
    ui_category = "Surface & Normals";
    ui_label = "Enable Smooth Normals";
    ui_tooltip = "Smooths out the calculated surface normals to reduce blocky artifacts, especially on flat surfaces.";
> = true;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.1;
    ui_category = "Surface & Normals";
    ui_label = "Depth Multiplier";
> = 1.0;

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
> = 4.0;

uniform int BlendMode <
    ui_type = "combo";
    ui_items = "Additive\0Screen\0Alpha Blend\0";
    ui_category = "Blending & Output";
    ui_label = "Blend Mode";
    ui_tooltip = "How the final reflections are blended with the original image.";
> = 1;

uniform float3 Adjustments <
    ui_type = "drag";
    ui_category = "Blending & Output";
    ui_label = "Saturation / Exposure / Contrast";
    ui_tooltip = "Adjusts the color properties of the final reflection.";
> = float3(1.5, 1.0, 1.1);

uniform bool EnableACES <
    ui_category = "Blending & Output";
    ui_label = "Enable ACES Tone Mapping";
    ui_tooltip = "Applies ACES tone mapping to the reflections for a more filmic look.";
> = false;

uniform bool AssumeSRGB <
    ui_category = "Blending & Output";
    ui_label = "Assume sRGB Input";
    ui_tooltip = "Assumes the input color is in sRGB space and linearizes it. Keep disabled unless you know the game outputs non-linear color.";
> = false;

uniform float RenderScale <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_tooltip = "Renders reflections at a lower resolution to improve performance. 0.5 means half resolution.";
> = 0.8;

uniform float ThicknessThreshold <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.01;
    ui_step = 0.001;
    ui_category = "Performance & Quality";
    ui_label = "Thickness Threshold";
    ui_tooltip = "Determines how 'thick' a surface is. Helps the ray to not pass through thin objects.";
> = 0.010;

uniform float VerticalFOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
    ui_tooltip = "Set this to your game's vertical Field of View for accurate projection calculations.";
> = 37.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0Motion Vectors\0Final Reflection\0Normals\0Depth\0Raw Reflection\0";
    ui_category = "Debug";
    ui_label = "Debug View Mode";
> = 0;

uniform int FRAME_COUNT < source = "framecount"; >;

//----------------|
// :: Textures :: |
//----------------|

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2D(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2D(sMotVectTexVort, texcoord).rg;
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
float2 SampleMotionVectors(float2 texcoord)
{
    return tex2D(sTexMotionVectorsSampler, texcoord).rg;
}
#endif

namespace SSRTnan
{
    texture SSR_Tex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sSSR
    {
        Texture = SSR_Tex;
    };

    texture Temp_Tex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sTemp
    {
        Texture = Temp_Tex;
    };

    texture History_Tex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sHistory
    {
        Texture = History_Tex;
    };
    
    texture Normal_Tex
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sNormal
    {
        Texture = Normal_Tex;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };

    //-------------------|
    // :: Functions    ::|
    //-------------------|

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

    float3 LinearizeSRGB(float3 color)
    {
        return pow(color, 2.2);
    }

    static const float3x3 g_sRGBToACEScg = float3x3(
        0.613117812906440, 0.341181995855625, 0.045787344282337,
        0.069934082307513, 0.918103037508582, 0.011932775530201,
        0.020462992637737, 0.106768663382511, 0.872715910619442
    );

    static const float3x3 g_ACEScgToSRGB = float3x3(
        1.704887331049502, -0.624157274479025, -0.080886773895704,
        -0.129520935348888, 1.138399326040076, -0.008779241755018,
        -0.024127059936902, -0.124620612286390, 1.148822109913262
    );
    
    float3 sRGBToACEScg(float3 srgb)
    {
        return mul(g_sRGBToACEScg, srgb);
    }
    float3 ACEScgToSRGB(float3 acescg)
    {
        return mul(g_ACEScgToSRGB, acescg);
    }

    float3 Apply_ACES(float3 color)
    {
        if (!EnableACES)
            return color;
        
        float3 acescg = sRGBToACEScg(color);

        const float A = 2.51, B = 0.03, C = 2.43, D = 0.59, E = 0.14;
        float3 toneMapped = (acescg * (A * acescg + B)) / (acescg * (C * acescg + D) + E);

        return ACEScgToSRGB(toneMapped);
    }

    float3 UVToViewPos(float2 uv, float view_z)
    {
        float2 clip_pos = uv * 2.0 - 1.0;
        float proj_scale_y = 1.0 / tan(radians(VerticalFOV * 0.5));
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;

        float3 view_pos;
        view_pos.x = clip_pos.x / proj_scale_x * view_z;
        view_pos.y = -clip_pos.y / proj_scale_y * view_z; 
        view_pos.z = view_z;

        return view_pos;
    }

    float2 ViewPosToUV(float3 view_pos)
    {
        float proj_scale_y = 1.0 / tan(radians(VerticalFOV * 0.5));
        float proj_scale_x = proj_scale_y / ReShade::AspectRatio;

        float2 clip_pos;
        clip_pos.x = view_pos.x * proj_scale_x / view_pos.z;
        clip_pos.y = -view_pos.y * proj_scale_y / view_pos.z;

        return clip_pos * 0.5 + 0.5;
    }

    float3 CalculateNormalFromDepth(float2 texcoord)
    {
        float2 p = ReShade::PixelSize;
        float center_depth = GetDepth(texcoord);

        float3 center_pos = UVToViewPos(texcoord, center_depth);
        float3 right_pos = UVToViewPos(texcoord + float2(p.x, 0), GetDepth(texcoord + float2(p.x, 0)));
        float3 up_pos = UVToViewPos(texcoord + float2(0, p.y), GetDepth(texcoord + float2(0, p.y)));

        return normalize(cross(up_pos - center_pos, right_pos - center_pos));
    }

    float3 ApplyBumpMapping(float2 texcoord, float3 normal)
    {
        if (BumpIntensity == 0.0)
            return normal;

        float2 px = ReShade::PixelSize;
        float h_c = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb);
        float h_x = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(px.x, 0), 0, 0)).rgb);
        float h_y = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, px.y), 0, 0)).rgb);

        float2 slope = float2(h_x - h_c, h_y - h_c) * BumpIntensity * 100.0;

        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T = normalize(cross(up, normal));
        float3 B = cross(normal, T);
        float3 bumpedNormal = normal + T * slope.x - B * slope.y;

        return normalize(bumpedNormal);
    }
    
    float3 ApplyGeometryCorrection(float2 texcoord, float3 normal)
    {
        if (GeoCorrectionIntensity == 0.0)
            return normal;

        float2 px = ReShade::PixelSize;
        float lumCenter = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb);
        float lumRight = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(px.x, 0), 0, 0)).rgb);
        float lumDown = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, px.y), 0, 0)).rgb);

        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));

        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }

    float3 GetSurfaceNormal(float2 texcoord)
    {
        float3 normal = CalculateNormalFromDepth(texcoord);

        if (bSmoothNormals)
        {
            float2 px = ReShade::PixelSize;
            float2 offset = px * 7.5 * (1.0 - GetDepth(texcoord));

            float3 n_t = CalculateNormalFromDepth(texcoord - float2(0, offset.y));
            float3 n_b = CalculateNormalFromDepth(texcoord + float2(0, offset.y));
            float3 n_l = CalculateNormalFromDepth(texcoord - float2(offset.x, 0));
            float3 n_r = CalculateNormalFromDepth(texcoord + float2(offset.x, 0));

            float w_t = smoothstep(1, 0, distance(normal, n_t) * 1.5) * 2;
            float w_b = smoothstep(1, 0, distance(normal, n_b) * 1.5) * 2;
            float w_l = smoothstep(1, 0, distance(normal, n_l) * 1.5) * 2;
            float w_r = smoothstep(1, 0, distance(normal, n_r) * 1.5) * 2;

            float4 weightedNormal = float4(normal, 1.0)
                                  + float4(n_t * w_t, w_t)
                                  + float4(n_b * w_b, w_b)
                                  + float4(n_l * w_l, w_l)
                                  + float4(n_r * w_r, w_r);

            if (weightedNormal.a > 0)
            {
                normal = normalize(weightedNormal.xyz / weightedNormal.a);
            }
        }

        normal = ApplyBumpMapping(texcoord, normal);
        normal = ApplyGeometryCorrection(texcoord, normal);

        return normal;
    }

    float3 SampleNormalFromTexture(float2 coords)
    {
        float3 normal = -(tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2.0;
        return normalize(normal);
    }

    bool TraceRay(in float3 rayOrigin, in float3 rayDir, out float3 hitViewPos, out float2 uvHit)
    {
        float stepSize = MIN_STEP_SIZE;
        float totalDist = 0.0;
        float3 prevPos = rayOrigin;

        [loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + rayDir * stepSize;
            totalDist += stepSize;

            float2 uvCurr = ViewPosToUV(currPos);
            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) || totalDist > MAX_TRACE_DISTANCE)
                break;

            float sceneDepth = GetDepth(uvCurr);
            float thickness = abs(currPos.z - sceneDepth);

            // If the ray is behind the surface or the surface is too thick, continue marching
            if (currPos.z < sceneDepth || thickness > ThicknessThreshold)
            {
                prevPos = currPos;
                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * STEP_SCALE, MIN_STEP_SIZE, MAX_STEP_SIZE);
                continue;
            }

            // --- Intersection found, perform refinement ---
            float3 lo = prevPos, hi = currPos;
            [unroll]
            for (int r = 0; r < REFINEMENT_STEPS; ++r)
            {
                float3 mid = 0.5 * (lo + hi);
                float midDepth = GetDepth(ViewPosToUV(mid));
                if (mid.z >= midDepth)
                    hi = mid;
                else
                    lo = mid;
            }
            hitViewPos = hi;
            uvHit = ViewPosToUV(hitViewPos);
            return true;
        }

        return false; // No intersection found
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

    void PS_GenerateNormals(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (any(uv > RenderScale))
            discard;
        float2 screen_uv = uv / RenderScale;
        float3 normal = GetSurfaceNormal(screen_uv);
        outNormal = float4(normal * 0.5 + 0.5, 1.0);
    }

    void PS_TraceReflections(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outReflection : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outReflection = 0;
            return;
        }

        float2 screen_uv = uv / RenderScale;
        float depth = GetDepth(screen_uv);
        float3 viewPos = UVToViewPos(screen_uv, depth);
        float3 viewDir = -normalize(viewPos);
        float3 normal = SampleNormalFromTexture(uv);

        float3 eyeDir = -viewDir;
        float3 rayDir = normalize(reflect(eyeDir, normal));

        float3 hitViewPos;
        float2 uvHit;
        bool hit = TraceRay(viewPos, rayDir, hitViewPos, uvHit);

        float3 reflectionColor = 0;
        if (hit)
        {
            float3 hitNormal = SampleNormalFromTexture(uvHit * RenderScale);
            // Check normal bias to avoid self-reflection
            if (distance(hitNormal, normal) >= NormalBias)
            {
                float distFactor = saturate(1.0 - length(hitViewPos - viewPos) / MAX_TRACE_DISTANCE);
                reflectionColor = tex2Dlod(ReShade::BackBuffer, float4(uvHit, 0, 0)).rgb * distFactor;
            }
        }
        else // Fallback logic for missed rays
        {
            float adaptiveDist = depth * 1.2 + 0.003;
            float3 fbViewPos = viewPos + rayDir * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos));
            bool isSky = GetDepth(uvFb) >= 1.0;
            float3 fbColor = tex2Dlod(ReShade::BackBuffer, float4(uvFb, 0, 0)).rgb;

            if (isSky)
            {
                reflectionColor = fbColor;
            }
            else
            {
                float depthFactor = saturate(1.0 - depth / MAX_TRACE_DISTANCE);
                float vertical_fade = 1.0 - screen_uv.y;
                reflectionColor = fbColor * depthFactor * vertical_fade;
            }
        }

        float fresnel = pow(1.0 - saturate(dot(eyeDir, normal)), 3.0);
        float angleWeight = pow(saturate(dot(-viewDir, rayDir)), 2.0);
        float fadeRange = max(FadeEnd - FadeStart, 0.001);
        float depthFade = saturate((FadeEnd - depth) / fadeRange);
        depthFade *= depthFade;

        reflectionColor *= fresnel * angleWeight * depthFade;
        outReflection = float4(reflectionColor, depth);
    }

    void PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outBlended : SV_Target)
    {
        float2 sample_uv = uv * RenderScale;
        float3 currentSpec = tex2D(sSSR, sample_uv).rgb;

        if (!EnableTemporal)
        {
            outBlended = float4(currentSpec, 1.0);
            return;
        }

        float2 motion = SampleMotionVectors(uv);
        float2 reprojected_uv = uv + motion;

        float currentDepth = GetDepth(uv);
        float historyDepth = GetDepth(reprojected_uv);

        bool validHistory = all(saturate(reprojected_uv) == reprojected_uv) &&
                            FRAME_COUNT > 1 &&
                            abs(historyDepth - currentDepth) < 0.01;

        float3 blendedSpec = currentSpec;
        if (validHistory)
        {
            float3 historySpec = tex2D(sHistory, reprojected_uv).rgb;
            
            // Use YCoCg color space and neighborhood clamping to reduce ghosting
            float3 minBox = RGBToYCoCg(currentSpec);
            float3 maxBox = minBox;

            [unroll]
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    float2 neighbor_uv = uv + float2(x, y) * ReShade::PixelSize;
                    float3 neighborSpec = RGBToYCoCg(tex2D(sSSR, neighbor_uv * RenderScale).rgb);
                    minBox = min(minBox, neighborSpec);
                    maxBox = max(maxBox, neighborSpec);
                }
            }
            
            float3 clampedHistorySpec = clamp(RGBToYCoCg(historySpec), minBox, maxBox);
            float alpha = 1.0 / min(FRAME_COUNT, AccumFramesSG);

            blendedSpec = YCoCgToRGB(lerp(clampedHistorySpec, RGBToYCoCg(currentSpec), alpha));
        }
        
        outBlended = float4(blendedSpec, 1.0);
    }

    void PS_UpdateHistory(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outHistory : SV_Target)
    {
        outHistory = tex2D(sTemp, uv);
    }

    void PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        // --- Debug Views ---
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float2 motion = SampleMotionVectors(uv);
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / 6.283185) + 0.5, 1.0, saturate(velocity));
                    outColor = float4(HSVToRGB(hsv), 1.0);
                    return;
                case 2: // Final Reflection
                    outColor = float4(tex2D(sTemp, uv).rgb, 1.0);
                    return;
                case 3: // Normals
                    outColor = float4(GetSurfaceNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 4: // Depth
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5: // Raw Reflection
                    outColor = float4(tex2D(sSSR, uv * RenderScale).rgb, 1.0);
                    return;
            }
        }

        // --- Final Composition ---
        float3 originalColor = tex2D(ReShade::BackBuffer, uv).rgb;
        float3 specularGI = tex2D(sTemp, uv).rgb;

        // Apply color adjustments
        specularGI *= Adjustments.g; // Exposure
        if (AssumeSRGB)
            specularGI = LinearizeSRGB(specularGI);
        if (EnableACES)
            specularGI = Apply_ACES(specularGI);
        
        float luminance = GetLuminance(specularGI);
        specularGI = lerp(luminance.xxx, specularGI, Adjustments.r); // Saturation
        specularGI = (specularGI - 0.5) * Adjustments.b + 0.5; // Contrast

        specularGI *= SPIntensity;

        // Blend with original image
        switch (BlendMode)
        {
            case 0: // Additive
                outColor = float4(originalColor.rgb + specularGI, 1.0);
                break;
            case 1: // Screen
                outColor = float4(1.0 - (1.0 - originalColor.rgb) * (1.0 - specularGI), 1.0);
                break;
            case 2: // Alpha Blend
                float giLuminance = GetLuminance(saturate(specularGI));
                outColor = float4(lerp(originalColor.rgb, specularGI, giLuminance), 1.0);
                break;
        }
    }

technique SSRT < ui_tooltip = "Screen Space Ray Traced Reflections"; >
{
    pass GenerateNormals
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_GenerateNormals;
        RenderTarget = Normal_Tex;
        ClearRenderTargets = true;
    }
    pass TraceReflections
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_TraceReflections;
        RenderTarget = SSR_Tex;
        ClearRenderTargets = true;
    }
    pass Accumulate
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Accumulate;
        RenderTarget = Temp_Tex;
    }
    pass UpdateHistory
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_UpdateHistory;
        RenderTarget = History_Tex;
    }
    pass Output
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Output;
    }
  }
}
