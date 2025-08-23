/*-------------------|
| :: Description ::  |
'--------------------|
    FSR 1.0 code is derived from the FidelityFX SDK, provided under the MIT license.
    Copyright (C) 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

    Barbatos SSR

    Version: 0.2.2
    Author: Barbatos Bachiko
    License: MIT

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 0.2.2
    + Ray Marching reworked
    + Bump Mapping improved
    + Fallback rethought.
    - Removed Roughness and jitter
*/

#include "ReShade.fxh"
#include "Blending.fxh"
#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#define PI 3.1415927
#define PI2 (2.0 * PI)
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

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

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

#if UI_DIFFICULTY == 0  // Simple Mode

#define NormalBias 0.0
#define FadeStart 0.0
#define SmoothMode 2
#define Smooth_Threshold 0.5
#define GeoCorrectionIntensity -0.01
#define DepthMultiplier 1.0
#define EnableTemporal true
#define Adjustments float3(1.5, 1.0, 1.1)
#define EnableACES false
#define AssumeSRGB false
#define ThicknessThreshold 0.10
#define VerticalFOV 37.0
#define c_phi 0.01
#define n_phi 5.0
#define p_phi 1.0
#define BumpEdgeThreshold 0.55

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

uniform float BumpIntensity <
    ui_label = "Bump Mapping Intensity";
    ui_type = "drag";
    ui_category = "Surface & Normals";
    ui_min = 0.0; ui_max = 1.0;
> = 0.036;

uniform float RenderScale <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_tooltip = "Renders reflections at a lower resolution for performance, then upscales using FSR 1.0.";
> = 0.8;

uniform bool bEnableDenoise <
    ui_category = "Denoiser";
    ui_type = "checkbox";
    ui_label = "Enable A-Trous Denoiser";
> = false;

uniform float AccumFramesSG <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 32.0;
    ui_step = 1.0;
    ui_category = "Temporal Filtering";
    ui_label = "Temporal Accumulation Frames";
    ui_tooltip = "Number of frames to accumulate. Higher values are smoother but may cause more ghosting on moving objects.";
> = 4.0;

BLENDING_COMBO(BlendMode, "Blend Mode", "How the final reflections are blended with the original image.", "Blending & Output", false, 0, 6)

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0Motion Vectors\0Final Reflection (Upscaled)\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0";
    ui_category = "Debug";
    ui_label = "Debug View Mode";
> = 0;

#endif // UI_DIFFICULTY == 0

#if UI_DIFFICULTY == 1 // Advanced Mode

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
> = 0.0;

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

uniform int SmoothMode <
    ui_label = "Smooth Normals Mode";
    ui_type = "combo";
    ui_items = "Off\0Fast (13x13)\0Medium (16x16)\0High Quality (31x31)\0";
    ui_category = "Surface & Normals";
> = 2;

uniform float Smooth_Threshold <
    ui_label = "Smooth Normals Threshold";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_category = "Surface & Normals";
> = 0.5;

uniform float BumpIntensity <
    ui_label = "Bump Mapping Intensity";
    ui_type = "drag";
    ui_category = "Surface & Normals";
    ui_min = 0.0; ui_max = 1.0;
> = 0.036;

uniform float BumpEdgeThreshold <
    ui_label = "Bump Edge Threshold";
    ui_type = "drag";
    ui_category = "Surface & Normals";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 0.55;

uniform float GeoCorrectionIntensity <
    ui_type = "drag";
    ui_min = -0.1; ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Surface & Normals";
    ui_label = "Geometry Correction Intensity";
> = -0.01;

uniform float DepthMultiplier <
    ui_type = "drag";
    ui_min = 0.1; ui_max = 5.0;
    ui_step = 0.1;
    ui_category = "Surface & Normals";
    ui_label = "Depth Multiplier";
> = 1.0;

// --- Denoiser UI ---
uniform bool bEnableDenoise  <
    ui_category = "Denoiser";
    ui_type = "checkbox";
    ui_label = "Enable A-Trous Denoiser";
> = false;

uniform float c_phi <
    ui_category = "Denoiser";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
    ui_label = "Color Sigma";
    ui_tooltip = "Controls the influence of color similarity in the denoiser. Lower values consider only very similar colors.";
> = 0.01;

uniform float n_phi <
    ui_category = "Denoiser";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 5.0; ui_step = 0.01;
    ui_label = "Normals Sigma";
    ui_tooltip = "Controls the influence of normal similarity. Lower values restrict filtering to surfaces with similar orientation.";
> = 5.0;

uniform float p_phi <
    ui_category = "Denoiser";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 10.0; ui_step = 0.01;
    ui_label = "Position (Depth) Sigma";
    ui_tooltip = "Controls the influence of world-space position similarity. Lower values restrict filtering to nearby pixels.";
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

BLENDING_COMBO(BlendMode, "Blend Mode", "How the final reflections are blended with the original image.\n'Normal' mode uses a custom luminance-based alpha blend.", "Blending & Output", false, 0, 6)

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
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Performance & Quality";
    ui_label = "Render Scale";
    ui_tooltip = "Renders reflections at a lower resolution for performance. 1.0 is full resolution.";
> = 0.8;

uniform float ThicknessThreshold <
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.1;
    ui_step = 0.001;
    ui_category = "Performance & Quality";
    ui_label = "Thickness Threshold";
    ui_tooltip = "Determines how 'thick' a surface is. Helps the ray to not pass through thin objects.";
> = 0.10;

uniform float VerticalFOV <
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0;
    ui_step = 0.1;
    ui_category = "Advanced";
    ui_label = "Vertical FOV";
> = 37.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0Motion Vectors\0Final Reflection (Upscaled)\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0";
    ui_category = "Debug";
    ui_label = "Debug View Mode";
> = 0;

#endif // UI_DIFFICULTY == 1

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
        return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 SampleMotionVectors(float2 texcoord) {
        return tex2Dlod(sMotVectTexVort, float4(texcoord, 0, 0)).rg;
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
    return tex2Dlod(sTexMotionVectorsSampler, float4(texcoord, 0, 0)).rg;
}
#endif

namespace Barbatos_SSR
{
    // Main SSR textures
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
    
    // G-Buffer textures
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

    texture NormTex_Pass1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormTex_Pass1
    {
        Texture = NormTex_Pass1;
    };

    texture NormTex_Pass2
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16f;
    };
    sampler sNormTex_Pass2
    {
        Texture = NormTex_Pass2;
    };

    // Denoiser textures
    texture DenoiseTex0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sDenoiseTex0
    {
        Texture = DenoiseTex0;
    };

    texture DenoiseTex1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA8;
    };
    sampler sDenoiseTex1
    {
        Texture = DenoiseTex1;
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

    //-----------------|
    // :: Structs     ::|
    //-----------------|

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
    
    // FSR
    struct RectificationBox
    {
        float3 boxCenter;
        float3 boxVec;
        float3 aabbMin;
        float3 aabbMax;
        float fBoxCenterWeight;
    };

    //-------------------|
    // :: Functions     ::|
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

    float3 ACES(float3 color)
    {
        if (!EnableACES)
            return color;
        
        const float3x3 ACES_INPUT_MAT = float3x3(
            0.59719, 0.35458, 0.04823,
            0.07600, 0.90834, 0.01566,
            0.02840, 0.13383, 0.83777
        );
        const float3x3 ACES_OUTPUT_MAT = float3x3(
            1.60475, -0.53108, -0.07367,
            -0.10208, 1.10813, -0.00605,
            -0.00327, -0.07276, 1.07602
        );
        
        float3 v = mul(ACES_INPUT_MAT, color);
        float3 a = v * (v + 0.0245786) - 0.000090537;
        float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
        float3 result = mul(ACES_OUTPUT_MAT, (a / b));
        
        return pow(saturate(result), 1.0 / 2.2);
    }
    
    float3 UVToViewPos(float2 uv, float view_z)
    {
        float fov_rad = VerticalFOV * (PI / 180.0);
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
        float fov_rad = VerticalFOV * (PI / 180.0);
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
        const float2 p = ReShade::PixelSize;
        float3 u, d, l, r, u2, d2, l2, r2;

        u = GVPFUV(texcoord + float2(0, p.y));
        d = GVPFUV(texcoord - float2(0, p.y));
        l = GVPFUV(texcoord - float2(p.x, 0));
        r = GVPFUV(texcoord + float2(-p.x, 0));

        u2 = GVPFUV(texcoord + float2(0, p.y * 2.0));
        d2 = GVPFUV(texcoord - float2(0, p.y * 2.0));
        l2 = GVPFUV(texcoord + float2(p.x * 2.0, 0));
        r2 = GVPFUV(texcoord + float2(p.x * 2.0, 0));

        u2 = u + (u - u2);
        d2 = d + (d - d2);
        l2 = l + (l - l2);
        r2 = r + (r - r2);

        float3 c = GVPFUV(texcoord);

        float3 v = u - c;
        float3 h = r - c;

        if (abs(d2.z - c.z) < abs(u2.z - c.z))
            v = c - d;
        if (abs(l2.z - c.z) < abs(r2.z - c.z))
            h = c - l;
        return normalize(cross(v, h));
    }

    float2 ComputeGradient(float2 texcoord)
    {
        const float2 offset = ReShade::PixelSize.xy;

        float3 colorTL = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-offset.x, -offset.y), 0, 0)).rgb;
        float3 colorT = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, -offset.y), 0, 0)).rgb;
        float3 colorTR = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(offset.x, -offset.y), 0, 0)).rgb;
        float3 colorL = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-offset.x, 0), 0, 0)).rgb;
        float3 colorR = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(offset.x, 0), 0, 0)).rgb;
        float3 colorBL = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-offset.x, offset.y), 0, 0)).rgb;
        float3 colorB = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, offset.y), 0, 0)).rgb;
        float3 colorBR = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(offset.x, offset.y), 0, 0)).rgb;
        
        float lumTL = GetLuminance(colorTL);
        float lumT = GetLuminance(colorT);
        float lumTR = GetLuminance(colorTR);
        float lumL = GetLuminance(colorL);
        float lumR = GetLuminance(colorR);
        float lumBL = GetLuminance(colorBL);
        float lumB = GetLuminance(colorB);
        float lumBR = GetLuminance(colorBR);

        float gx = (-3.0 * lumTL - 10.0 * lumL - 3.0 * lumBL) + (3.0 * lumTR + 10.0 * lumR + 3.0 * lumBR);
        float gy = (-3.0 * lumTL - 10.0 * lumT - 3.0 * lumTR) + (3.0 * lumBL + 10.0 * lumB + 3.0 * lumBR);

        float2 gradient = float2(gx, gy);

        if (BumpEdgeThreshold > 0.0 && length(gradient) < BumpEdgeThreshold)
        {
            gradient = 0.0;
        }

        return gradient;
    }

    float3 Bump(float2 texcoord, float intensity)
    {
        float2 gradient = ComputeGradient(texcoord);
        float3 bumpNormal = float3(gradient.x * -intensity, gradient.y * -intensity, 1.0);
        return normalize(bumpNormal);
    }

    float3 blend_normals(float3 n1, float3 n2)
    {
        n1 += float3(0, 0, 1);
        n2 *= float3(-1, -1, 1);
        return n1 * dot(n1, n2) / n1.z - n2;
    }

    float3 GeometryCorrection(float2 texcoord, float3 normal)
    {
        if (GeoCorrectionIntensity == 0.0)
            return normal;

        float lumCenter = GetLuminance(tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb);
        float lumRight = GetLuminance(tex2Doffset(ReShade::BackBuffer, texcoord, int2(1, 0)).rgb);
        float lumDown = GetLuminance(tex2Doffset(ReShade::BackBuffer, texcoord, int2(0, 1)).rgb);

        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));

        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }

    float3 SampleNormal(float2 coords)
    {
        float3 normal = (tex2Dlod(sNormal, float4(coords, 0, 0)).xyz - 0.5) * 2.0;
        return normalize(normal);
    }
    
    // Ray Marching
    HitResult TraceRay(Ray r)
    {
        HitResult result;
        result.found = false;

        float stepSize = MIN_STEP_SIZE;
        float totalDist = 0.0;
        float3 prevPos = r.origin;
        float depthBias = NormalBias * 0.005;

        [loop]
        for (int i = 0; i < STEPS_PER_RAY; ++i)
        {
            float3 currPos = prevPos + r.direction * stepSize;
            totalDist += stepSize;

            float2 uvCurr = ViewPosToUV(currPos);
            if (any(uvCurr < 0.0) || any(uvCurr > 1.0) || totalDist > MAX_TRACE_DISTANCE)
                break;

            float sceneDepth = GetDepth(uvCurr);
            float thickness = abs(currPos.z - sceneDepth);

            if (currPos.z < sceneDepth - depthBias || thickness > ThicknessThreshold)
            {
                prevPos = currPos;
                float distToDepth = abs(currPos.z - sceneDepth);
                stepSize = clamp(distToDepth * STEP_SCALE, MIN_STEP_SIZE, MAX_STEP_SIZE);
                continue;
            }
            
            float3 lo = prevPos, hi = currPos;
            [unroll]
            for (int ref_step = 0; ref_step < REFINEMENT_STEPS; ++ref_step)
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

    // FSR
    void RectificationBoxAddSample(inout RectificationBox box, bool bInitialSample, float3 fSample, float fWeight)
    {
        if (bInitialSample)
        {
            box.boxCenter = fSample * fWeight;
            box.boxVec = fSample * fSample * fWeight;
            box.aabbMin = fSample;
            box.aabbMax = fSample;
            box.fBoxCenterWeight = fWeight;
        }
        else
        {
            box.boxCenter += fSample * fWeight;
            box.boxVec += fSample * fSample * fWeight;
            box.aabbMin = min(box.aabbMin, fSample);
            box.aabbMax = max(box.aabbMax, fSample);
            box.fBoxCenterWeight += fWeight;
        }
    }

    void RectificationBoxComputeVarianceBoxData(inout RectificationBox box)
    {
        const float fBoxCenterWeightRcp = rcp(max(1e-6, box.fBoxCenterWeight));
        box.boxCenter *= fBoxCenterWeightRcp;
        box.boxVec = max(0.0, box.boxVec * fBoxCenterWeightRcp - (box.boxCenter * box.boxCenter));
    }

    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|

    void PS_GBuffer_NoSmooth(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode != 0)
            discard;
        outNormal.rgb = Normal(uv.xy);
        outNormal.a = GetDepth(uv.xy);
        outNormal.rgb = blend_normals(Bump(uv, -BumpIntensity), outNormal.rgb);
        outNormal.rgb = GeometryCorrection(uv, outNormal.rgb);
        outNormal.rgb = outNormal.rgb * 0.5 + 0.5;
    }

    void PS_GBuffer_WithSmooth(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
        outNormal.rgb = Normal(uv.xy);
        outNormal.a = GetDepth(uv.xy);
    }

    void PS_SmoothNormals_H(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
        float4 color = tex2Dlod(sNormTex_Pass1, float4(uv, 0, 0));
        float4 s, s1;
        float sc;
        
        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = tex2Dlod(sNormTex_Pass1, float4(uv.xy + float2(x * p.x, 0), 0, 0));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T);
            s1 += s * diff;
            sc += diff;
        }
        outNormal = s1.rgba / sc;
    }

    void PS_SmoothNormals_V(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outNormal : SV_Target)
    {
        if (SmoothMode == 0)
            discard;
        float4 color = tex2Dlod(sNormTex_Pass2, float4(uv, 0, 0));
        float4 s, s1;
        float sc;

        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = tex2Dlod(sNormTex_Pass2, float4(uv + float2(0, x * p.y), 0, 0));
            float diff = dot(0.333, abs(s.rgb - color.rgb)) + abs(s.a - color.a) * (FAR_PLANE * Smooth_Threshold);
            diff = 1 - saturate(diff * T * 2);
            s1 += s * diff;
            sc += diff;
        }
        
        s1.rgba = s1.rgba / sc;
        s1.rgb = blend_normals(Bump(uv, -BumpIntensity), s1.rgb);
        s1.rgb = GeometryCorrection(uv, s1.rgb);
        outNormal = float4(s1.rgb * 0.5 + 0.5, s1.a);
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
            outReflection = float4(tex2Dlod(ReShade::BackBuffer, float4(scaled_uv, 0, 0)).rgb, depth);
            return;
        }

        float3 viewPos = UVToViewPos(scaled_uv, depth);
        float3 viewDir = -normalize(viewPos);
        float3 normal = SampleNormal(scaled_uv);

        float3 eyeDir = -viewDir;
        
        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(eyeDir, normal));
        
        HitResult hit = TraceRay(r);
        float3 surfaceFallbackColor = tex2Dlod(ReShade::BackBuffer, float4(scaled_uv, 0, 0)).rgb;

        float3 reflectionColor = 0;
        float reflectionFade = 1.0;

        if (hit.found)
        {
            float3 hitNormal = SampleNormal(hit.uv);
            if (distance(hitNormal, normal) >= NormalBias)
            {
                reflectionColor = tex2Dlod(ReShade::BackBuffer, float4(hit.uv, 0, 0)).rgb;
                float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / MAX_TRACE_DISTANCE);
                float fadeRange = max(FadeEnd - FadeStart, 0.001);
                float depthFade = saturate((FadeEnd - depth) / fadeRange);
                depthFade *= depthFade;

                reflectionFade = distFactor * depthFade;
            }
            else
            {
                reflectionColor = surfaceFallbackColor;
            }
        }
        else
        {
            float adaptiveDist = depth * 1.2 + 0.012;
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos));
            reflectionColor = tex2Dlod(ReShade::BackBuffer, float4(uvFb, 0, 0)).rgb;
        }

        float fresnel = pow(1.0 - saturate(dot(eyeDir, normal)), 3.0);
        reflectionColor *= fresnel * reflectionFade;
        
        float angleWeight = pow(saturate(dot(-viewDir, r.direction)), 2.0);
        float3 finalReflection = lerp(surfaceFallbackColor, reflectionColor, angleWeight);
        
        outReflection = float4(finalReflection, depth);
    }

    void PS_Accumulate(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outBlended : SV_Target)
    {
        if (any(uv > RenderScale))
        {
            outBlended = 0;
            return;
        }

        float3 currentSpec = tex2Dlod(sReflection, float4(uv, 0, 0)).rgb;

        if (!EnableTemporal)
        {
            outBlended = float4(currentSpec, 1.0);
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

        float3 blendedSpec = currentSpec;
        if (validHistory)
        {
            float3 historySpec = tex2Dlod(sHistory, float4(reprojected_uv_low, 0, 0)).rgb;

            float3 minBox = RGBToYCoCg(currentSpec), maxBox = minBox;
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    if (x == 0 && y == 0)
                        continue;
                    
                    float2 neighbor_uv = uv + float2(x, y) * ReShade::PixelSize;
                    float3 neighborSpec = RGBToYCoCg(tex2Dlod(sReflection, float4(neighbor_uv, 0, 0)).rgb);
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
        outHistory = tex2Dlod(sTemp, float4(uv, 0, 0));
    }

    float4 PS_DenoisePass(float4 vpos : SV_Position, float2 uv : TEXCOORD, int level, sampler input_sampler)
    {
        if (any(uv > RenderScale))
        {
            return tex2Dlod(input_sampler, float4(uv, 0.0, 0.0));
        }

        float4 center_color = tex2Dlod(input_sampler, float4(uv, 0.0, 0.0));
        
        float2 full_res_uv = uv / RenderScale;
        float center_depth = GetDepth(full_res_uv);
        
        if (center_depth / FAR_PLANE >= 0.999)
            return center_color;

        float3 center_normal = SampleNormal(full_res_uv);
        float3 center_pos = UVToViewPos(full_res_uv, center_depth);

        float4 sum = 0.0;
        float cum_w = 0.0;
        
        const float2 step_size = ReShade::PixelSize * exp2(level);

        static const float2 atrous_offsets[9] =
        {
            float2(-1, -1), float2(0, -1), float2(1, -1),
            float2(-1, 0), float2(0, 0), float2(1, 0),
            float2(-1, 1), float2(0, 1), float2(1, 1)
        };

        [loop]
        for (int i = 0; i < 9; i++)
        {
            const float2 neighbor_uv_low = uv + atrous_offsets[i] * step_size;

            if (any(neighbor_uv_low < 0.0) || any(neighbor_uv_low > RenderScale))
                continue;

            const float4 sample_color = tex2Dlod(input_sampler, float4(neighbor_uv_low, 0.0, 0.0));
            
            const float2 neighbor_uv_full = neighbor_uv_low / RenderScale;
            const float sample_depth = GetDepth(neighbor_uv_full);

            if (sample_depth / FAR_PLANE >= 0.999)
                continue;

            const float3 sample_normal = SampleNormal(neighbor_uv_full);
            const float3 sample_pos = UVToViewPos(neighbor_uv_full, sample_depth);
            
            float diff_c = distance(center_color.rgb, sample_color.rgb);
            float w_c = exp(-(diff_c * diff_c) / c_phi);
            
            float diff_n = dot(center_normal, sample_normal);
            float w_n = pow(saturate(diff_n), n_phi);
            
            float diff_p = distance(center_pos, sample_pos);
            float w_p = exp(-(diff_p * diff_p) / p_phi);

            const float weight = w_c * w_n * w_p;

            sum += sample_color * weight;
            cum_w += weight;
        }

        return cum_w > 1e-6 ? (sum / cum_w) : center_color;
    }

    float4 PS_DenoisePass0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return tex2D(sTemp, texcoord);
        return PS_DenoisePass(vpos, texcoord, 0, sTemp);
    }

    float4 PS_DenoisePass1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return tex2D(sDenoiseTex0, texcoord);
        return PS_DenoisePass(vpos, texcoord, 1, sDenoiseTex0);
    }

    float4 PS_DenoisePass2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return tex2D(sDenoiseTex1, texcoord);
        return PS_DenoisePass(vpos, texcoord, 2, sDenoiseTex1);
    }

    void PS_Upscale_FSR(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        if (RenderScale >= 1.0)
        {
            outColor = tex2Dlod(sTemp, float4(uv, 0, 0));
            return;
        }

        const float2 fDownscaleFactor = float2(RenderScale, RenderScale);
        const float2 fRenderSize = BUFFER_SCREEN_SIZE * fDownscaleFactor;

        const float2 fDstOutputPos = uv * BUFFER_SCREEN_SIZE + 0.5f;
        const float2 fSrcOutputPos = fDstOutputPos * fDownscaleFactor;
        const int2 iSrcInputPos = int2(floor(fSrcOutputPos));
        
        const float2 fSrcUnjitteredPos = (float2(iSrcInputPos) + 0.5f);
        const float2 fBaseSampleOffset = fSrcUnjitteredPos - fSrcOutputPos;

        int2 offsetTL;
        offsetTL.x = (fSrcUnjitteredPos.x > fSrcOutputPos.x) ? -2 : -1;
        offsetTL.y = (fSrcUnjitteredPos.y > fSrcOutputPos.y) ? -2 : -1;

        const bool bFlipCol = fSrcUnjitteredPos.x > fSrcOutputPos.x;
        const bool bFlipRow = fSrcUnjitteredPos.y > fSrcOutputPos.y;

        RectificationBox clippingBox;
        float3 fSamples[9];
        int iSampleIndex = 0;

        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                const int2 sampleColRow = int2(bFlipCol ? (2 - col) : col, bFlipRow ? (2 - row) : row);
                const int2 iSrcSamplePos = iSrcInputPos + offsetTL + sampleColRow;
                
                const float2 sample_uv = ((iSrcSamplePos + 0.5) * rcp(fRenderSize)) * RenderScale;
                
                float3 s = tex2Dlod(sTemp, float4(sample_uv, 0, 0)).rgb;
                fSamples[iSampleIndex] = RGBToYCoCg(s);
                iSampleIndex++;
            }
        }

        iSampleIndex = 0;
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                const int2 sampleColRow = int2(bFlipCol ? (2 - col) : col, bFlipRow ? (2 - row) : row);
                const float2 fOffset = (float2) offsetTL + (float2) sampleColRow;
                const float2 fSrcSampleOffset = fBaseSampleOffset + fOffset;

                const float fRectificationCurveBias = -2.3f;
                const float fSrcSampleOffsetSq = dot(fSrcSampleOffset, fSrcSampleOffset);
                const float fBoxSampleWeight = exp(fRectificationCurveBias * fSrcSampleOffsetSq);

                const bool bInitialSample = (row == 0) && (col == 0);
                RectificationBoxAddSample(clippingBox, bInitialSample, fSamples[iSampleIndex], fBoxSampleWeight);
                iSampleIndex++;
            }
        }
        
        RectificationBoxComputeVarianceBoxData(clippingBox);

        float3 finalColorYCoCg = clippingBox.boxCenter;
        outColor.rgb = YCoCgToRGB(finalColorYCoCg);
        outColor.a = 1.0;
    }

    void PS_Output(float4 pos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        if (ViewMode != 0)
        {
            switch (ViewMode)
            {
                case 1: // Motion vectors
                    float2 motion = SampleMotionVectors(uv);
                    float velocity = length(motion) * 100.0;
                    float angle = atan2(motion.y, motion.x);
                    float3 hsv = float3((angle / PI2) + 0.5, 1.0, saturate(velocity));
                    outColor = float4(HSVToRGB(hsv), 1.0);
                    return;
                case 2: // Final Reflection (Upscaled)
                    outColor = float4(tex2Dlod(sUpscaledReflection, float4(uv, 0, 0)).rgb, 1.0);
                    return;
                case 3: // Normals
                    outColor = float4(SampleNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 4: // Depth
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5: // Raw Low-Res Reflection
                    outColor = float4(tex2Dlod(sReflection, float4(uv, 0, 0)).rgb, 1.0);
                    return;
                case 6: // Denoised Low-Res Reflection
                    outColor = float4(tex2Dlod(sTemp, float4(uv, 0, 0)).rgb, 1.0);
                    return;
            }
        }

        float3 originalColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0, 0)).rgb;

        if (GetDepth(uv) >= 1.0)
        {
            outColor = float4(originalColor, 1.0);
            return;
        }

        float3 specularGI = tex2Dlod(sUpscaledReflection, float4(uv, 0, 0)).rgb;

        specularGI *= Adjustments.y; // Exposure
        if (AssumeSRGB)
            specularGI = LinearizeSRGB(specularGI);
        if (EnableACES)
            specularGI = ACES(specularGI);
        
        float luminance = GetLuminance(specularGI);
        specularGI = lerp(luminance.xxx, specularGI, Adjustments.x); // Saturation
        float3 SSR = specularGI;

        specularGI = (specularGI - 0.5) * Adjustments.z + 0.5;
        specularGI = saturate(specularGI);
        specularGI *= SPIntensity;

        float3 finalColor;

        if (BlendMode == 0)
        {
            float giLuminance = GetLuminance(saturate(SSR * SPIntensity));
            finalColor = lerp(originalColor.rgb, specularGI, giLuminance);
        }
        else
        {
            finalColor = ComHeaders::Blending::Blend(BlendMode, originalColor.rgb, specularGI, 1.0);
        }

        outColor = float4(finalColor, 1.0);
    }

//-------------------|
// :: Technique     ::|
//-------------------|
    technique Barbatos_SSR
    {
        pass GBuffer_NoSmooth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GBuffer_NoSmooth;
            RenderTarget = TNormal;
        }
        pass GBuffer_WithSmooth
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_GBuffer_WithSmooth;
            RenderTarget = NormTex_Pass1;
        }
        pass SmoothNormals_H
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SmoothNormals_H;
            RenderTarget = NormTex_Pass2;
        }
        pass SmoothNormals_V
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_SmoothNormals_V;
            RenderTarget = TNormal;
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
        pass DenoisePass0
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass0;
            RenderTarget = DenoiseTex0;
        }
        pass DenoisePass1
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass1;
            RenderTarget = DenoiseTex1;
        }
        pass DenoisePass2
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_DenoisePass2;
            RenderTarget = Temp;
        }
        pass Upscale_FSR
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Upscale_FSR;
            RenderTarget = UpscaledReflection;
        }
        pass Output
        {
            VertexShader = PostProcessVS;
            PixelShader = PS_Output;
        }
    }
}
