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
furnished to do so, to the following conditions :

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

    Version: 0.2.4
    Author: Barbatos Bachiko
    License: MIT
    HDR/Color functions from Glamarye Fast Effects by Robert Jessop, used under MIT license.

    History:
    (*) Feature (+) Improvement (x) Bugfix (-) Information (!) Compatibility

    Version 0.2.4
    x Fixed Fallback Fade
    * Added reflection orientation limiter (Floors, Walls, Ceilings).
    + Integrated color space handling from Glamarye Fast Effects.
    + Added support for sRGB, scRGB, and HDR10 (PQ) color spaces.
    + Added UI controls for color space and tonemapping.
    + Other improvements
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
#define PI2 (2.0 * PI)
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetDepth(coords) (ReShade::GetLinearizedDepth(coords) * DepthMultiplier)
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

#define MAX_TRACE_DISTANCE 2.0
static const int STEPS_PER_RAY = 128;
static const int REFINEMENT_STEPS = 5;

#define STEP_SCALE 0.7
#define MIN_STEP_SIZE 0.001
#define MAX_STEP_SIZE 1.0

#ifndef UI_DIFFICULTY
#define UI_DIFFICULTY 0
#endif

// Color Space from Glamarye
#ifndef HDR_WHITELEVEL
#define HDR_WHITELEVEL 203
#endif

#ifndef OVERRIDE_COLOR_SPACE
#define OVERRIDE_COLOR_SPACE 0
#endif

#if OVERRIDE_COLOR_SPACE > 0
#undef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE OVERRIDE_COLOR_SPACE
#endif

#if BUFFER_COLOR_SPACE > 0
#else
#if BUFFER_COLOR_BIT_DEPTH == 8
#undef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 1
#elif BUFFER_COLOR_BIT_DEPTH == 16
#undef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 2
#elif __RENDERER__ < 0xb000

#undef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 1
#endif
#endif

#if BUFFER_COLOR_BIT_DEPTH == 10 &&  __RENDERER__ >= 0xb000 && __RESHADE__ <= 50100 && OVERRIDE_COLOR_SPACE == 0
#define MANUAL_COLOR_SPACE_SELECT 1
#else
#define MANUAL_COLOR_SPACE_SELECT 0
#endif

#if BUFFER_COLOR_BIT_DEPTH > 8 || BUFFER_COLOR_SPACE > 1
#define IS_HDR_OR_EXTENDED_RANGE 1
#else
#define IS_HDR_OR_EXTENDED_RANGE 0
#endif

//----------|
// :: UI :: |
//----------|

#if MANUAL_COLOR_SPACE_SELECT
uniform int select_color_space <
    ui_category = "Color Space & HDR";
    ui_type = "combo";
    ui_label = "Color Space (CHECK THIS!)";
    ui_tooltip = "Your version of ReShade cannot auto-detect the color space for 10-bit output. Please select it manually.\n\n* If your game and screen are in HDR mode, set to HDR Perceptual Quantizer.\n* If not, set it to SDR sRGB.";
    ui_items = "SDR sRGB (PC standard for non-HDR screens)\0HDR Perceptual Quantizer (SMTPE ST2084)\0";
> = 0;
#define GLAMAYRE_COLOR_SPACE (select_color_space * 2 + 1)
#else
uniform int show_color_space <
    ui_category = "Color Space & HDR";
    ui_type = "combo";
    ui_label = "Detected Color Space";
    ui_tooltip = "ReShade or the shader has detected which color space the game is using. To override, set OVERRIDE_COLOR_SPACE pre-processor definition.\n\n1 = sRGB (standard)\n2 = scRGB (linear)\n3 = PQ (HDR10)\n4 = HLG (HDR)";
    #if BUFFER_COLOR_SPACE == 1
        ui_items = "1 sRGB (autodetected)\0";
    #elif BUFFER_COLOR_SPACE == 2
        ui_items = "2 scRGB (autodetected)\0";
    #elif BUFFER_COLOR_SPACE == 3
        ui_items = "3 Perceptual Quantizer (autodetected)\0";
    #elif BUFFER_COLOR_SPACE == 4
        ui_items = "4 Hybrid Log–Gamma (autodetected)\0";
    #else
        ui_items = "Unknown! (assuming sRGB)\0";
    #endif
> = 0;
#define GLAMAYRE_COLOR_SPACE BUFFER_COLOR_SPACE
#endif

#if IS_HDR_OR_EXTENDED_RANGE
uniform int fast_color_space_conversion <
    ui_category = "Color Space & HDR";
    ui_type = "combo";
    ui_label = "Transfer Function Precision";
    ui_items = "Accurate\0Fast Approximation\0";
    ui_tooltip = "Correct color space conversions (especially for HDR) can be slow. A fast approximation is often sufficient as inaccuracies mostly cancel out.";
> = 1;
#else
#define fast_color_space_conversion 2
#endif

uniform float tone_map < __UNIFORM_SLIDER_FLOAT1
	ui_category = "Color Space & HDR";
	ui_min = 1;ui_max = 9; ui_step = .1;
	ui_tooltip = "Note: this is ignored in HDR modes.\n\nIn SDR, games use tone mapping to fit a wide brightness range onto the screen. This function attempts to reverse that for more accurate lighting calculations.\n\nOptimal value depends on the game's tone mapping method. The default is a good starting point.";
	ui_label = "SDR Tone Mapping Compensation";
> = 1;

#if UI_DIFFICULTY == 0 // Simple Mode
#define NormalBias 0.0
#define FadeStart 0.0
#define SmoothMode 2
#define Smooth_Threshold 0.5
#define GeoCorrectionIntensity -0.01
#define DepthMultiplier 1.0
#define EnableTemporal true
#define Adjustments float3(1.0, 1.0, 1.0)
#define EnableACES false
#define ThicknessThreshold 0.10
#define VerticalFOV 37.0
#define c_phi 0.1
#define n_phi 5.0
#define p_phi 1.0
#define BumpEdgeThreshold 0.55
#define fReflectFloorsIntensity 1.0
#define fReflectWallsIntensity 0.5
#define fReflectCeilingsIntensity 0.5
#define OrientationThreshold 0.85

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
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;

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
> = 2.0;

BLENDING_COMBO(BlendMode, "Blend Mode", "How the final reflections are blended with the original image.", "Blending & Output", false, 0, 6)

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "None\0Motion Vectors\0Final Reflection (Upscaled)\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0Reflection Mask\0";
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

uniform float fReflectFloorsIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Floor Reflection Intensity";
> = 1.0;

uniform float fReflectWallsIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Wall Reflection Intensity";
> = 0.5;

uniform float fReflectCeilingsIntensity <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Ceiling Reflection Intensity";
> = 0.5;

uniform float OrientationThreshold <
    ui_type = "drag";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Reflection Settings";
    ui_label = "Orientation Threshold";
    ui_tooltip = "Controls the sensitivity for detecting floors, walls, and ceilings. Lower values are stricter.";
> = 0.85;

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
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;

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
> = 0.1;

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
> = 2.0;

BLENDING_COMBO(BlendMode, "Blend Mode", "How the final reflections are blended with the original image.", "Blending & Output", false, 0, 6)

uniform float3 Adjustments <
    ui_type = "drag";
    ui_category = "Blending & Output";
    ui_label = "Saturation / Exposure / Contrast";
    ui_tooltip = "Adjusts the color properties of the final reflection.";
> = float3(1.0, 1.0, 1.0);

uniform bool EnableACES <
    ui_category = "Blending & Output";
    ui_label = "Enable ACES Tone Mapping";
    ui_tooltip = "Applies ACES tone mapping to the reflections for a more filmic look.";
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
    ui_items = "None\0Motion Vectors\0Final Reflection (Upscaled)\0Normals\0Depth\0Raw Low-Res Reflection\0Denoised Low-Res Reflection\0Reflection Mask\0";
    ui_category = "Debug";
    ui_label = "Debug View Mode";
> = 0;

#endif // UI_DIFFICULTY == 1

uniform int FRAME_COUNT < source = "framecount"; >;

//--------------------------|
// :: Color Space & HDR ::  |
//--------------------------|

float3 undoTonemap(float3 c)
{
    if (GLAMAYRE_COLOR_SPACE < 2)
    {
        c = saturate(c);
        c = c / (1.0 - (1.0 - rcp(tone_map)) * c);
    }
    return c;
}

float3 reapplyTonemap(float3 c)
{
    if (GLAMAYRE_COLOR_SPACE < 2)
    {
        c = c / ((1 - rcp(tone_map)) * c + 1.0);
    }
    return c;
}

float3 sRGBtoLinearAccurate(float3 r)
{
    return (r <= .04045) ? (r / 12.92) : pow(abs(r + .055) / 1.055, 2.4);
}
float3 sRGBtoLinearFastApproximation(float3 r)
{
    return max(r / 12.92, r * r);
}
float3 sRGBtoLinear(float3 r)
{
    if (fast_color_space_conversion == 1)
        r = sRGBtoLinearFastApproximation(r);
    else if (fast_color_space_conversion == 0)
        r = sRGBtoLinearAccurate(r);
    return r;
}
float3 linearToSRGBAccurate(float3 r)
{
    return (r <= .0031308) ? (r * 12.92) : (1.055 * pow(abs(r), 1.0 / 2.4) - .055);
}
float3 linearToSRGBFastApproximation(float3 r)
{
    return min(r * 12.92, sqrt(r));
}
float3 linearToSRGB(float3 r)
{
    if (fast_color_space_conversion == 1)
        r = linearToSRGBFastApproximation(r);
    else if (fast_color_space_conversion == 0)
        r = linearToSRGBAccurate(r);
    return r;
}

// Perceptual Quantizer (HDR10) Conversions
float3 PQtoLinearAccurate(float3 r)
{
    const float m1 = 1305.0 / 8192.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 107.0 / 128.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    float3 powr = pow(max(r, 0), 1.0 / m2);
    r = pow(max(max(powr - c1, 0) / (c2 - c3 * powr), 0), 1.0 / m1);
    return r * 10000.0 / HDR_WHITELEVEL;
}
float3 PQtoLinearFastApproximation(float3 r)
{
    float3 square = r * r;
    float3 quad = square * square;
    float3 oct = quad * quad;
    r = max(max(square / 340.0, quad / 6.0), oct);
    return r * 10000.0 / HDR_WHITELEVEL;
}
float3 PQtoLinear(float3 r)
{
    if (fast_color_space_conversion)
        r = PQtoLinearFastApproximation(r);
    else
        r = PQtoLinearAccurate(r);
    return r;
}
float3 linearToPQAccurate(float3 r)
{
    const float m1 = 1305.0 / 8192.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 107.0 / 128.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    r = r * (HDR_WHITELEVEL / 10000.0);
    float3 powr = pow(max(r, 0), m1);
    r = pow(max((c1 + c2 * powr) / (1 + c3 * powr), 0), m2);
    return r;
}
float3 linearToPQFastApproximation(float3 r)
{
    r = r * (HDR_WHITELEVEL / 10000.0);
    float3 squareroot = sqrt(r);
    float3 quadroot = sqrt(squareroot);
    float3 octroot = sqrt(quadroot);
    r = min(octroot, min(sqrt(sqrt(6.0)) * quadroot, sqrt(340.0) * squareroot));
    return r;
}
float3 linearToPQ(float3 r)
{
    if (fast_color_space_conversion)
        r = linearToPQFastApproximation(r);
    else
        r = linearToPQAccurate(r);
    return r;
}

// Hybrid Log Gamma Conversions
float3 linearToHLG(float3 r)
{
    r = r * HDR_WHITELEVEL / 1000;
    float a = 0.17883277;
    float b = 0.28466892;
    float c = 0.55991073;
    float3 s = sqrt(3 * r);
    return (s < .5) ? s : (log(12 * r - b) * a + c);
}
float3 HLGtoLinear(float3 r)
{
    float a = 0.17883277;
    float b = 0.28466892;
    float c = 0.55991073;
    r = (r < .5) ? r * r / 3.0 : ((exp((r - c) / a) + b) / 12.0);
    return r * 1000 / HDR_WHITELEVEL;
}

// Main conversion wrapper functions
float3 toLinearColorspace(float3 r)
{
    if (GLAMAYRE_COLOR_SPACE == 2)
        r = r * (80.0 / HDR_WHITELEVEL); // scRGB
    else if (GLAMAYRE_COLOR_SPACE == 3)
        r = PQtoLinear(r); // HDR10 PQ
    else if (GLAMAYRE_COLOR_SPACE == 4)
        r = HLGtoLinear(r); // HLG
    else
        r = sRGBtoLinear(r); // sRGB
    return r;
}
float3 toOutputColorspace(float3 r)
{
    if (GLAMAYRE_COLOR_SPACE == 2)
        r = r * (HDR_WHITELEVEL / 80.0); // scRGB
    else if (GLAMAYRE_COLOR_SPACE == 3)
        r = linearToPQ(r); // HDR10 PQ
    else if (GLAMAYRE_COLOR_SPACE == 4)
        r = linearToHLG(r); // HLG
    else
        r = linearToSRGB(r); // sRGB
    return r;
}

//----------------|
// :: Textures :: |
//----------------|

sampler samplerColor
{
    Texture = ReShade::BackBufferTex;
#if IS_HDR_OR_EXTENDED_RANGE
	SRGBTexture = false;
#else
    SRGBTexture = true;
#endif
};

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
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
#endif

namespace Barbatos_SSR31
{
    texture Reflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F; 
    };
    sampler sReflection
    {
        Texture = Reflection;
    };

    texture Temp
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F; 
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
        Format = RGBA16F; 
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

    texture DenoiseTex0
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sDenoiseTex0
    {
        Texture = DenoiseTex0;
    };

    texture DenoiseTex1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F; 
    };
    sampler sDenoiseTex1
    {
        Texture = DenoiseTex1;
    };

    texture UpscaledReflection
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
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
    
    struct RectificationBox
    {
        float3 boxCenter;
        float3 boxVec;
        float3 aabbMin;
        float3 aabbMax;
        float fBoxCenterWeight;
    };

//---------------|
// :: Utility :: |
//---------------|
    
    float3 GetColor(float2 uv)
    {
        float3 color = GetLod(samplerColor, uv).rgb;
        return toLinearColorspace(color);
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

//------------------------------------|
// :: View Space & Normal Functions ::|
//------------------------------------|

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

        float3 colorTL = GetColor(texcoord + float2(-offset.x, -offset.y)).rgb;
        float3 colorT = GetColor(texcoord + float2(0, -offset.y)).rgb;
        float3 colorTR = GetColor(texcoord + float2(offset.x, -offset.y)).rgb;
        float3 colorL = GetColor(texcoord + float2(-offset.x, 0)).rgb;
        float3 colorR = GetColor(texcoord + float2(offset.x, 0)).rgb;
        float3 colorBL = GetColor(texcoord + float2(-offset.x, offset.y)).rgb;
        float3 colorB = GetColor(texcoord + float2(0, offset.y)).rgb;
        float3 colorBR = GetColor(texcoord + float2(offset.x, offset.y)).rgb;
        
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

        float lumCenter = GetLuminance(GetColor(texcoord).rgb);
        float lumRight = GetLuminance(toLinearColorspace(tex2Doffset(samplerColor, texcoord, int2(1, 0)).rgb));
        float lumDown = GetLuminance(toLinearColorspace(tex2Doffset(samplerColor, texcoord, int2(0, 1)).rgb));

        float3 bumpNormal = normalize(float3(lumRight - lumCenter, lumDown - lumCenter, 1.0));

        return normalize(normal + bumpNormal * GeoCorrectionIntensity);
    }

    float3 SampleNormal(float2 coords)
    {
        float3 normal = (GetLod(sNormal, coords).xyz - 0.5) * 2.0;
        return normalize(normal);
    }

#if !USE_MARTY_LAUNCHPAD_MOTION && !USE_VORT_MOTION
    float2 SampleMotionVectors(float2 texcoord)
    {
        return GetLod(sTexMotionVectorsSampler, texcoord).rg;
    }
#elif USE_MARTY_LAUNCHPAD_MOTION
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
#elif USE_VORT_MOTION
    float2 SampleMotionVectors(float2 texcoord) {
        return GetLod(sMotVectTexVort, texcoord).rg;
    }
#endif

//-------------------|
// :: Ray Tracing  ::|
//-------------------|

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

//-----------|
// :: FSR  ::|
//-----------|

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
        float4 color = GetLod(sNormTex_Pass1, uv);
        float4 s, s1;
        float sc;
        
        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = GetLod(sNormTex_Pass1, uv.xy + float2(x * p.x, 0));
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
        float4 color = GetLod(sNormTex_Pass2, uv);
        float4 s, s1;
        float sc;

        float SNWidth = (SmoothMode == 1) ? 5.5 : (SmoothMode == 2) ? 2.5 : 1.0;
        int SNSamples = (SmoothMode == 1) ? 1 : (SmoothMode == 2) ? 3 : 30;

        float2 p = ReShade::PixelSize * SNWidth;
        float T = rcp(max(Smooth_Threshold * saturate(2 * (1 - color.a)), 0.0001));
        
        for (int x = -SNSamples; x <= SNSamples; x++)
        {
            s = GetLod(sNormTex_Pass2, uv + float2(0, x * p.y));
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
            outReflection = 0;
            return;
        }

        float3 viewPos = UVToViewPos(scaled_uv, depth);
        float3 viewDir = -normalize(viewPos);

        float3 normal_un = (GetLod(sNormal, scaled_uv).xyz - 0.5) * 2.0;
        float normal_len_sq = dot(normal_un, normal_un);
        float3 normal = normal_un * rsqrt(max(1e-6, normal_len_sq));

        float3 eyeDir = -viewDir;
    
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
    
        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(eyeDir, normal));
    
        HitResult hit = TraceRay(r);

        float3 reflectionColor = 0;
        float reflectionAlpha = 0.0;
        float fresnel = pow(1.0 - saturate(dot(eyeDir, normal)), 3.0);

        if (hit.found)
        {
            float3 hitNormal = (GetLod(sNormal, hit.uv).xyz - 0.5) * 2.0;
            float hitNormal_len_sq = dot(hitNormal, hitNormal);

            float dotp = dot(normal, hitNormal) * rsqrt(max(1e-6, normal_len_sq * hitNormal_len_sq));

            if (dotp <= (1.0 - 0.5 * NormalBias * NormalBias))
            {
                reflectionColor = GetColor(hit.uv).rgb;
                float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / MAX_TRACE_DISTANCE);
                float fadeRange = max(FadeEnd - FadeStart, 0.001);
                float depthFade = saturate((FadeEnd - depth) / fadeRange);
                depthFade *= depthFade;

                reflectionAlpha = distFactor * depthFade * fresnel;
            }
        }
        else
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

    float4 PS_DenoisePass(float4 vpos : SV_Position, float2 uv : TEXCOORD, int level, sampler input_sampler)
    {
        if (any(uv > RenderScale))
        {
            return GetLod(input_sampler, uv);
        }

        float4 center_color = GetLod(input_sampler, uv);
        
        float2 full_res_uv = uv / RenderScale;
        float center_depth = GetDepth(full_res_uv);
        
        if (center_depth / FAR_PLANE >= 0.999)
            return center_color;

        float3 center_normal = (GetLod(sNormal, full_res_uv).xyz - 0.5) * 2.0;
        float center_normal_len_sq = dot(center_normal, center_normal);
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

            const float4 sample_color = GetLod(input_sampler, neighbor_uv_low);
            
            const float2 neighbor_uv_full = neighbor_uv_low / RenderScale;
            const float sample_depth = GetDepth(neighbor_uv_full);

            if (sample_depth / FAR_PLANE >= 0.999)
                continue;

            const float3 sample_normal = (GetLod(sNormal, neighbor_uv_full).xyz - 0.5) * 2.0;
            const float3 sample_pos = UVToViewPos(neighbor_uv_full, sample_depth);
            
            float diff_c = distance(center_color.rgb, sample_color.rgb);
            float w_c = exp(-(diff_c * diff_c) / c_phi);
            
            const float sample_normal_len_sq = dot(sample_normal, sample_normal);
            float diff_n = dot(center_normal, sample_normal) * rsqrt(max(1e-6, center_normal_len_sq * sample_normal_len_sq));
            float w_n = exp(n_phi * (saturate(diff_n) - 1.0));
            
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
            return GetLod(sTemp, texcoord);
        return PS_DenoisePass(vpos, texcoord, 0, sTemp);
    }

    float4 PS_DenoisePass1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return GetLod(sDenoiseTex0, texcoord);
        return PS_DenoisePass(vpos, texcoord, 1, sDenoiseTex0);
    }

    float4 PS_DenoisePass2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
    {
        if (!bEnableDenoise || RenderScale >= 1.0)
            return GetLod(sDenoiseTex1, texcoord);
        return PS_DenoisePass(vpos, texcoord, 2, sDenoiseTex1);
    }

    void PS_Upscale_FSR(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 outColor : SV_Target)
    {
        if (RenderScale >= 1.0)
        {
            outColor = GetLod(sTemp, uv);
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
        float fSamplesA[9];
        int iSampleIndex = 0;

        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                const int2 sampleColRow = int2(bFlipCol ? (2 - col) : col, bFlipRow ? (2 - row) : row);
                const int2 iSrcSamplePos = iSrcInputPos + offsetTL + sampleColRow;
                
                const float2 sample_uv = ((iSrcSamplePos + 0.5) * rcp(fRenderSize)) * RenderScale;
                
                float4 s = GetLod(sTemp, sample_uv);
                fSamples[iSampleIndex] = RGBToYCoCg(s.rgb);
                fSamplesA[iSampleIndex] = s.a;
                iSampleIndex++;
            }
        }

        iSampleIndex = 0;
        float fAccumulatedAlpha = 0.0;
        float fAccumulatedWeight = 0.0;

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

                fAccumulatedAlpha += fSamplesA[iSampleIndex] * fBoxSampleWeight;
                fAccumulatedWeight += fBoxSampleWeight;

                iSampleIndex++;
            }
        }
        
        RectificationBoxComputeVarianceBoxData(clippingBox);

        float3 finalColorYCoCg = clippingBox.boxCenter;
        outColor.rgb = YCoCgToRGB(finalColorYCoCg);
        outColor.a = fAccumulatedAlpha / max(1e-6, fAccumulatedWeight);
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
                    outColor = float4(toOutputColorspace(HSVToRGB(hsv)), 1.0);
                    return;
                case 2: // Final Reflection (Upscaled)
                    outColor = float4(toOutputColorspace(GetLod(sUpscaledReflection, uv).rgb), 1.0);
                    return;
                case 3: // Normals
                    outColor = float4(SampleNormal(uv) * 0.5 + 0.5, 1.0);
                    return;
                case 4: // Depth
                    outColor = GetDepth(uv).xxxx;
                    return;
                case 5: // Raw Low-Res Reflection
                    outColor = float4(toOutputColorspace(GetLod(sReflection, uv).rgb), 1.0);
                    return;
                case 6: // Denoised Low-Res Reflection
                    outColor = float4(toOutputColorspace(GetLod(sTemp, uv).rgb), 1.0);
                    return;
                case 7: // Reflection Mask
                    outColor = GetLod(sUpscaledReflection, uv).aaaa;
                    return;
            }
        }

        float3 originalColor = GetColor(uv);

        if (GetDepth(uv) >= 1.0)
        {
            outColor = float4(toOutputColorspace(originalColor), 1.0);
            return;
        }

        float4 reflectionSample = GetLod(sUpscaledReflection, uv);
        float3 specularGI = reflectionSample.rgb;
        float blendFactor = reflectionSample.a;

        specularGI *= Adjustments.y; // Exposure
        
        if (EnableACES)
            specularGI = ACES(specularGI);
        
        float luminance = GetLuminance(specularGI);
        specularGI = lerp(luminance.xxx, specularGI, Adjustments.x); // Saturation

        specularGI = (specularGI - 0.5) * Adjustments.z + 0.5; // Contrast
        specularGI = saturate(specularGI);
        specularGI *= SPIntensity;

        float3 finalColor = ComHeaders::Blending::Blend(BlendMode, originalColor.rgb, specularGI, blendFactor);
        finalColor = reapplyTonemap(finalColor);
        finalColor = toOutputColorspace(finalColor);

        outColor = float4(finalColor, 1.0);
    }

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
#if IS_HDR_OR_EXTENDED_RANGE
                SRGBWriteEnable = false;
#else
            SRGBWriteEnable = true;
#endif
        }
    }
}
