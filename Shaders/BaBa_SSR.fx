/*----------------------------------------------|
| :: Barbatos SSR (Screen-Space Reflections) :: |
|-----------------------------------------------|
| Version: 1.5                                  |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "Blending.fxh"

//----------|
// :: UI :: |
//----------|

uniform int HDR_Input_Format <
    ui_category = "HDR";
    ui_label = "Input Format";
    ui_tooltip = "Auto = Detect automatically (recommended)\nRaw = No conversion";
    ui_type = "combo";
    ui_items = "Auto\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0Raw (No Conversion)\0";
> = 0;
uniform float HDR_Peak_Nits <
    ui_category = "HDR";
    ui_label = "HDR Peak Brightness (Nits)";
    ui_type = "drag";
    ui_min = 400.0; ui_max = 10000.0; ui_step = 10.0;
> = 1000.0;

//Reflections

uniform float Intensity <
    ui_category = "Reflections";
    ui_label = "Intensity";
    ui_type = "drag";
    ui_tooltip = "Overall intensity of reflections";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
> = 1.0;
uniform float THICKNESS_THRESHOLD <
    ui_category = "Reflections";
    ui_label = "Base Thickness";
    ui_tooltip = "Base thickness of objects.";
    ui_type = "drag";
    ui_min = 0.001; ui_max = 0.6;
    ui_step = 0.001;
> = 0.001;
uniform float FadeDistance <
    ui_category = "Reflections";
    ui_label = "Fade Distance";
    ui_type = "drag";
    ui_tooltip = "How far away reflections start to fade out";
    ui_min = 0.0;
    ui_max = 5.0; ui_step = 0.01;
> = 4.999;

uniform int ReflectionMode <
    ui_category = "Reflections";
    ui_label = "Surface Mode";
    ui_type = "combo";
    ui_tooltip = "Choose which surfaces show reflections";
    ui_items = "Floors Only\0Walls Only\0Ceilings Only\0Floors & Ceilings\0All Surfaces\0";
> = 4;
uniform float OrientationThreshold <
    ui_category = "Reflections";
    ui_label = "Surface Orientation Threshold";
    ui_tooltip = "Determines the angle separation for Walls vs Floors vs Ceilings.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0; ui_step = 0.01;
> = 0.5;

uniform int RayTraceQuality <
    ui_category = "Reflections";
    ui_label = "Quality";
    ui_type = "combo";
    ui_tooltip = "Controls ray tracing steps and max distance.";
    ui_items = "Normal\0High\0Extreme (Perfect)\0";
> = 0;

uniform float RenderResolution <
    ui_category = "Reflections";
    ui_label = "Resolution";
    ui_type = "drag";
    ui_min = 0.3;
    ui_max = 1.0; ui_step = 0.05;
> = 0.8;
uniform float VERTICAL_FOV <
    ui_category = "Reflections";
    ui_label = "Vertical FOV";
    ui_type = "drag";
    ui_min = 15.0;
    ui_max = 120.0; ui_step = 0.1;
> = 70.0;

uniform bool EnableSmoothing <
    ui_category = "Reflections";
    ui_label = "Reduce Noise (TAA)";
> = true;

uniform bool EnableTAAUpscaling <
    ui_category = "Reflections";
    ui_label = "TAA Upscaling (Sub-pixel Jitter)";
> = true;

uniform bool EnableAntiSmear <
    ui_category = "Reflections";
    ui_label = "Anti-Smearing (Reduce TAA on Motion)";
> = false;

//Material
uniform float SurfaceGlossiness <
    ui_category = "Material";
    ui_label = "Glossiness";
    ui_tooltip = "0 = Sharp reflections\n1 = Total roughness";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.3;

uniform float Anisotropy <
    ui_category = "Material";
    ui_label = "Anisotropy";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;
uniform float Metallic <
    ui_category = "Material";
    ui_label = "Metallic";
    ui_type = "drag";
    ui_tooltip = "Make surfaces look more metallic (0=non-metal, 1=metal)";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;

uniform float DIELECTRIC_REFLECTANCE <
    ui_category = "Material";
    ui_label = "Dielectric Reflectance (F0)";
    ui_tooltip = "Base reflectivity for non-metallic surfaces.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.04;

uniform float RoughnessDetection <
    ui_category = "Material";
    ui_label = "Roughness Detection";
    ui_type = "drag";
    ui_tooltip = "Estimates roughness based on local color contrast.\nHigher values make detailed/noisy textures appear rougher.";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
> = 0.0;

uniform float SurfaceDetails <
    ui_category = "Material";
    ui_label = "Details";
    ui_type = "drag";
    ui_tooltip = "Adds small surface details to reflections";
    ui_min = 0.0; ui_max = 5.0;
    ui_step = 0.01;
> = 0.3;

uniform float SobelEdgeThreshold <
    ui_category = "Material";
    ui_label = "Details: Edge Detection Threshold";
    ui_tooltip = "Threshold for surface detail detection.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0; ui_step = 0.001;
> = 0.0;

uniform float GeoCorrectionIntensity <
    ui_category = "Material";
    ui_label = "Geometric Correction";
    ui_tooltip = "Adjusts normal bending based on geometry.";
    ui_type = "drag";
    ui_min = -1.0;
    ui_max = 1.0; ui_step = 0.01;
> = -0.01;

//Color Grading
BLENDING_COMBO(BlendMode, "Blend Mode", "Select how reflections are blended with the scene.", "Color Grading", false, 0, 0)

uniform float Preserve_Scene_Highlights <
    ui_category = "Color Grading";
    ui_label = "Preserve Scene Highlights";
    ui_tooltip = "Protects bright scene objects from being darkened or overwritten by reflections.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool Use_Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Use Color Temperature";
> = false;

uniform float Color_Temperature <
    ui_category = "Color Grading";
    ui_label = "Temperature (Kelvin)";
    ui_type = "drag";
    ui_min = 1500.0; ui_max = 15000.0; ui_step = 10.0;
> = 6500.0;
uniform float SSR_Vibrance <
    ui_category = "Color Grading";
    ui_label = "Vibrance";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 10.0; ui_step = 0.1;
> = 1.0;

uniform float SSR_Contrast <
    ui_category = "Color Grading";
    ui_label = "Contrast";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;
uniform float3 SSR_Tint <
    ui_category = "Color Grading";
    ui_label = "Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Shadow_Tint <
    ui_category = "Color Grading";
    ui_label = "Shadow Color";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float3 SSR_Highlight_Tint <
    ui_category = "Color Grading";
    ui_label = "Highlight Color";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);
uniform float SSR_Split_Balance <
    ui_category = "Color Grading";
    ui_label = "Split Balance";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

//Advanced
uniform int SmartSurfaceMode <
    ui_category = "Advanced";
    ui_label = "Surface Detection Quality";
    ui_type = "combo";
    ui_items = "Performance\0Balanced\0Quality\0";
> = 0;
uniform float Smooth_Threshold <
    ui_category = "Advanced";
    ui_label = "Normal Smooth Threshold";
    ui_tooltip = "Threshold for smoothing surface normals to reduce noise. Lower values = less smoothing.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;
uniform float Vegetation_Protection <
    ui_category = "Advanced";
    ui_label = "Vegetation Protection";
    ui_tooltip = "Hacks the normal map to remove reflections from noisy geometry (leaves/grass).\nHigher values remove more vegetation.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;
uniform float EDGE_MASK_THRESHOLD <
    ui_category = "Advanced";
    ui_label = "Edge Mask Threshold";
    ui_tooltip = "Reduces artifacts on geometry edges.";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.2;

uniform int ViewMode <
    ui_category = "Advanced";
    ui_label = "Debug View";
    ui_type = "combo";
    ui_items = "Off\0Reflections Only\0Surface Normals\0Depth View\0Motion\0Confidence\0";
> = 0;

#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 0
#endif

// Defines & Constants
#define PI 3.1415927
#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
#define GetColor(c) tex2Dlod(ReShade::BackBuffer, float4((c).xy, 0, 0))
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

static const float DEG2RAD = 0.017453292;
// PI / 180.0
static const float2 TAA_Offsets[5] =
{
    float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1)
};
#define Anisotropy_Rotation 90
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
float GetFlowConf(float2 texcoord) { 
    float2 mv = GetMV(texcoord);
    float2 prev_uv = texcoord + mv;

    if (any(prev_uv < 0.0) || any(prev_uv > 1.0)) 
        return 0.0;

    return 1.0; 
}

#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 GetMV(float2 texcoord) { return GetLod(sMotVectTexVort, texcoord).rg;
}
float GetFlowConf(float2 texcoord) { 
    float2 mv = GetMV(texcoord);
    float2 prev_uv = texcoord + mv;

    if (any(prev_uv < 0.0) || any(prev_uv > 1.0)) 
        return 0.0;

    return 1.0; 
}

#else

texture2D texMotionVectors
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
    
texture2D tMotionConfidence
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};
sampler sMotionConfidence
{
    Texture = tMotionConfidence;
    MagFilter = POINT;
    MinFilter = POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

float2 GetMV(float2 texcoord)
{
    return GetLod(sTexMotionVectorsSampler, texcoord).rg;
}
    
float GetFlowConf(float2 texcoord)
{
    return GetLod(sMotionConfidence, texcoord).r;
}
#endif

namespace Barbatos_SSR110
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
        Format = RGBA16F;
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
        Format = RGBA16F;
    };
    sampler sHistory0
    {
        Texture = History0;
    };

    texture History1
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
    };
    sampler sHistory1
    {
        Texture = History1;
    };

    texture TexColorCopy
    {
        Width = BUFFER_WIDTH;
        Height = BUFFER_HEIGHT;
        Format = RGBA16F;
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
    
	//-------------------------|
    // :: Blue Noise Texture ::|
    //-------------------------|
    texture TexBlueNoise < source = "SS_BN.png"; >
    {
        Width = 1024;
        Height = 1024;
        Format = RGBA8;
    };
    sampler sTexBlueNoise
    {
        Texture = TexBlueNoise;
        AddressU = Repeat;
        AddressV = Repeat;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
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

    void VS_Accumulate0(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_SSR(id, outStruct);
        if (fmod((float) FRAME_COUNT, 2.0) > 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
    }
    
    void VS_Accumulate1(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        VS_Barbatos_SSR(id, outStruct);
        if (fmod((float) FRAME_COUNT, 2.0) < 0.5)
        {
            outStruct.vpos = float4(-10000.0, -10000.0, 0.0, 0.0);
        }
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
        const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }

    //------------------------|
    // :: Color Management :: |
    //------------------------|

    static const float3 LUMA_709 = float3(0.2126, 0.7152, 0.0722);
    static const float3 LUMA_2020 = float3(0.2627, 0.6780, 0.0593);

    int GetHDRMode()
    {
        if (HDR_Input_Format != 0)
            return HDR_Input_Format;
#if BUFFER_COLOR_SPACE == 1
        return 1;
#elif BUFFER_COLOR_SPACE == 2
        return 2;
#elif BUFFER_COLOR_SPACE == 3
        return 3;
#else
        return 1;
#endif
    }

    float3 PQ2Linear(float3 color)
    {
        float m1 = 0.1593017578125;
        float m2 = 78.84375;
        float c1 = 0.8359375;
        float c2 = 18.8515625;
        float c3 = 18.6875;
        float3 val = max(pow(abs(color), 1.0 / m2) - c1, 0.0);
        float3 den = c2 - c3 * pow(abs(color), 1.0 / m2);
        float3 linearHdr = pow(abs(val / den), 1.0 / m1);
        return linearHdr * (10000.0 / HDR_Peak_Nits);
    }

    float3 Linear2PQ(float3 linearColor)
    {
        float m1 = 0.1593017578125;
        float m2 = 78.84375;
        float c1 = 0.8359375;
        float c2 = 18.8515625;
        float c3 = 18.6875;
        float3 Y = max(0.0, linearColor * (HDR_Peak_Nits / 10000.0));
        float3 num = c1 + c2 * pow(Y, m1);
        float3 den = 1.0 + c3 * pow(Y, m1);
        return pow(num / den, m2);
    }

	#define SDR_Inverse_Power 0.05

	float3 sRGB2Linear(float3 x)
    {
        float3 linear_srgb = (x < 0.04045) ? (x / 12.92) : pow(abs((x + 0.055) / 1.055), 2.4);
        float3 safe_rgb = min(linear_srgb, 0.95); 
        
        // Per-channel Inverse Reinhard
        float3 expanded_rgb = (safe_rgb / max(1.0 - safe_rgb, 0.001));
        return expanded_rgb * SDR_Inverse_Power;
    }

    float3 Linear2sRGB(float3 x)
    {
        x = max(x, 0.0);
        x = x / max(SDR_Inverse_Power, 0.001);
        x = x / (1.0 + x);
        return (x < 0.0031308) ? (12.92 * x) : (1.055 * pow(abs(x), 1.0 / 2.4) - 0.055);
    }
	
    float3 Input2Linear(float3 color)
    {
        int mode = GetHDRMode();
        if (mode == 4)
            return color;
        else if (mode == 2)
            return color * (80.0 / HDR_Peak_Nits);
        else if (mode == 3)
            return PQ2Linear(color);
        else
            return sRGB2Linear(color);
    }

    float3 Linear2Output(float3 color)
    {
        int mode = GetHDRMode();
        if (mode == 4)
            return color;
        else if (mode == 2)
            return color * (HDR_Peak_Nits / 80.0);
        else if (mode == 3)
            return Linear2PQ(color);
        else
            return Linear2sRGB(color);
    }

    float GetLuminance(float3 color)
    {
        int mode = GetHDRMode();
        float3 lumaCoeff = (mode == 2 || mode == 3) ? LUMA_2020 : LUMA_709;
        return dot(color, lumaCoeff);
    }

    float3 KelvinToRGB(float k)
    {
        float3 color;
        k = clamp(k, 1000.0, 40000.0) / 100.0;
        if (k <= 66.0)
        {
            color.r = 255.0;
            color.g = 99.4708025861 * log(k) - 161.1195681661;
            if (k <= 19.0)
                color.b = 0.0;
            else
                color.b = 138.5177312231 * log(k - 10.0) - 305.0447927307;
        }
        else
        {
            color.r = 329.698727446 * pow(k - 60.0, -0.1332047592);
            color.g = 288.1221695283 * pow(k - 60.0, -0.0755148492);
            color.b = 255.0;
        }
        return saturate(color / 255.0);
    }
    
    //------------|
    // :: Noise ::|
    //------------|
    float GetSpatialTemporalNoise(float2 pos)
    {
        float time = fmod((float) FRAME_COUNT, 64.0);
        return frac(52.9829189 * frac(0.06711056 * pos.x + 0.00583715 * pos.y + 0.006237 * time));
    }
    
    float goldenSequence(uint i)
    {
        return float(2654435769u * i) / 4294967296.0;
    }
    float2 plasticSequence(uint i)
    {
        return float2(3242174889u * i, 2447445414u * i) / 4294967296.0;
    }
    float3 sequence3D(uint i)
    {
        return float3(plasticSequence(i), goldenSequence(i));
    }
    float3 toroidalJitter(float3 x, float3 jitter)
    {
        return 2.0 * abs(frac(x + jitter) - 0.5);
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
        float z_safe = max(1e-6, view_pos.z);
        float2 ndc = view_pos.xy / (z_safe * pScale);
        return float2(ndc.x, -ndc.y) * 0.5 + 0.5;
    }

    float3 CalculateNormal(float2 uv, float2 pScale)
    {
        float3 center = UVToViewPos(uv, GetDepth(uv), pScale);
        float3 offset_x = UVToViewPos(uv + float2(ReShade::PixelSize.x, 0), GetDepth(uv + float2(ReShade::PixelSize.x, 0)), pScale);
        float3 offset_y = UVToViewPos(uv + float2(0, ReShade::PixelSize.y), GetDepth(uv + float2(0, ReShade::PixelSize.y)), pScale);
        float3 n = cross(center - offset_x, center - offset_y);
        float lenSq = dot(n, n);
        return (lenSq > 1e-25) ?
            n * rsqrt(lenSq) : float3(0, 0, -1);
    }
    
    float3 ApplySurfaceDetails(float2 texcoord, float3 normal, float depth)
{
    if (SurfaceDetails <= 0.0 && GeoCorrectionIntensity == 0.0)
        return normal;
         
    float radius = lerp(3.0, 0.5, saturate(depth * 20.0));
    float2 off = ReShade::PixelSize * radius;

    float L = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-off.x, 0), 0, 0)).g;
    float R = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(off.x, 0), 0, 0)).g;
    float T = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, off.y), 0, 0)).g;
    float B = tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0, -off.y), 0, 0)).g;
    
    float Gx = (R - L) * 3.0; 
    float Gy = (B - T) * 3.0;
    
    float3 finalNormal = normal;

    if (SurfaceDetails > 0.0 && (Gx * Gx + Gy * Gy) >= (SobelEdgeThreshold * SobelEdgeThreshold))
    {
        float detailFade = lerp(1.0, 0.2, saturate(depth * 25.0));
        float2 slope = float2(Gx, Gy) * (SurfaceDetails * 10) * detailFade;
        
        float3 up = abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 T_vec = normalize(cross(up, normal));
        float3 B_vec = cross(normal, T_vec);
        
        finalNormal = normalize(finalNormal + T_vec * slope.x - B_vec * slope.y);
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
    float GetThickness(float2 uv, float3 normal, float3 viewDir, float depth)
    {
        // Calculate angle between view and surface
        float NdotV = abs(dot(normal, -viewDir));
        //Volume Expansion Factor
        float geometricScale = 1.0 / max(NdotV, 0.2);
        //Edge Guard 
        float depthDerivative = fwidth(depth);
        float edgeMask = 1.0 - smoothstep(0.0, 0.002, depthDerivative);

        return THICKNESS_THRESHOLD * geometricScale * edgeMask;
    }

    HitResult TraceRay2D(Ray r, int num_steps, float max_dist, float2 pScale, float jitter, float geoThickness)
    {
        HitResult result;
        result.found = false;
        result.viewPos = 0.0;
        result.uv = 0.0;
        
        float3 endPos = r.origin + r.direction * max_dist;

        float2 startUV = ViewPosToUV(r.origin, pScale);
        float2 endUV = ViewPosToUV(endPos, pScale);
        float2 deltaUV = endUV - startUV;
        if (dot(deltaUV, deltaUV) < 0.0001)
            return result;
        float startK = 1.0 / r.origin.z;
        float endK = 1.0 / endPos.z;
        float deltaK = endK - startK;
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
            float rayDepth = 1.0 / currK;
            float sceneDepth = GetDepth(currUV);
            float depthDiff = rayDepth - sceneDepth;
            
            // Adaptive thickness 
            float prevRayDepth = 1.0 / (currK - stepK);
            float rayStepSizeZ = abs(rayDepth - prevRayDepth);
            float adaptiveThickness = max(geoThickness, rayStepSizeZ * 1.5);
            adaptiveThickness *= (1.0 + rayDepth * 0.2);
            
            if (depthDiff > 0.0 && depthDiff < adaptiveThickness)
            {
                float2 loUV = currUV - stepUV;
                float2 hiUV = currUV;
                float2 midUV;
                
                [unroll]
                for (int j = 0; j < 2; j++)
                {
                    midUV = (loUV + hiUV) * 0.5;
                    float midRayDepth = 1.0 / (currK - stepK * 0.5);
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
        float3 center = GetColor(uv).rgb;
        float lumaC = GetLuminance(center);
        float2 p = ReShade::PixelSize;
        
        float lumaN = GetLuminance(GetColor(uv + float2(p.x, 0)).rgb);
        float lumaS = GetLuminance(GetColor(uv - float2(p.x, 0)).rgb);
        float lumaE = GetLuminance(GetColor(uv + float2(0, p.y)).rgb);
        float lumaW = GetLuminance(GetColor(uv - float2(0, p.y)).rgb);
        
        return saturate((abs(lumaN - lumaC) + abs(lumaS - lumaC) + abs(lumaE - lumaC) + abs(lumaW - lumaC)) * 10.0);
    }
    
    float specularPowerToConeAngle(float specularPower)
    {
        if (specularPower >= 4096.0)
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

    float3 GetGlossySample(float2 sample_uv, float2 pixel_uv, float local_roughness, float2 rand_noise)
    {
        float netRoughness = saturate(SurfaceGlossiness + (local_roughness * RoughnessDetection));
        if (netRoughness <= 0.001)
            return tex2Dlod(sTexColorCopy, float4(sample_uv, 0, 0)).rgb;

        float specularPower = pow(2.0, 10.0 * (1.0 - netRoughness) + 1.0);
        float coneTheta = specularPowerToConeAngle(specularPower) * 0.5;
        
        float2 screen_size = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float2 deltaP = (sample_uv - pixel_uv) * screen_size;
        
        float adjacentLength = length(deltaP);
        float oppositeLength = isoscelesTriangleOpposite(adjacentLength, coneTheta);
        float incircleSize = isoscelesTriangleInRadius(oppositeLength, adjacentLength);

        float rawMip = log2(max(1.0, incircleSize));
        float mipLevel = clamp(rawMip - 1.5, 0.0, 4.0);
        float2 blurRadiusUV = incircleSize * ReShade::PixelSize;

        float anisoScaleX = 1.0; float anisoScaleY = 1.0;
        float sinRot = 0.0; float cosRot = 1.0;
        bool useAnisotropy = (Anisotropy > 0.01);
        
        if (useAnisotropy)
        {
            anisoScaleX = 1.0 + (Anisotropy * 15.0);
            anisoScaleY = 1.0 / (1.0 + Anisotropy * 2.0);
            float angle = radians(Anisotropy_Rotation);
            sincos(angle, sinRot, cosRot);
        }

        float2 offset = ConcentricSquareMapping(rand_noise);
        float2 bn_uv = (pixel_uv * screen_size) / 1024.0;

        float2 golden_offset = float2(0.61803398875, 0.73205080757) * fmod((float)FRAME_COUNT, 64.0);
        bn_uv += golden_offset;

        float blue_noise_val = tex2Dlod(sTexBlueNoise, float4(bn_uv, 0, 0)).r;

        float scrambleAngle = blue_noise_val * 2.0 * PI;
        float s_scram, c_scram;
        sincos(scrambleAngle, s_scram, c_scram);
        
        float2 scrambledOffset;
        scrambledOffset.x = offset.x * c_scram - offset.y * s_scram;
        scrambledOffset.y = offset.x * s_scram + offset.y * c_scram;
        offset = scrambledOffset;

        if (useAnisotropy)
        {
            offset.x *= anisoScaleX;
            offset.y *= anisoScaleY;
            float2 rotatedOffset;
            rotatedOffset.x = offset.x * cosRot - offset.y * sinRot;
            rotatedOffset.y = offset.x * sinRot + offset.y * cosRot;
            offset = rotatedOffset;
        }

        offset = clamp(offset, -1.0, 1.0);
        
        return tex2Dlod(sTexColorCopy, float4(sample_uv + offset * blurRadiusUV, 0, mipLevel)).rgb;
    }
    
    //------------|
    // :: TAA  :: |
    //------------|
	float3 TAA_Compress(float3 color)
	{
    return color / (10.0 + color);
	}

	float3 TAA_Resolve(float3 color)
	{
    return (color * 10.0) / max(1e-6, 1.0 - color);
	}

    float4 GetActiveHistory(float2 uv)
    {
        return (fmod((float) FRAME_COUNT, 2.0) < 0.5) ?
            GetLod(sHistory0, uv) : GetLod(sHistory1, uv);
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
    
    float2 GetVelocity(float2 texcoord)
    {
        float2 pixel_size = ReShade::PixelSize;
        float closest_depth = 1.0;
        float2 closest_velocity = 0.0;

        [unroll]
        for (int i = 0; i < 5; i++)
        {
            float2 s_coord = texcoord + TAA_Offsets[i] * pixel_size;
            float s_depth = GetDepth(s_coord);
            if (s_depth < closest_depth)
            {
                closest_depth = s_depth;
                closest_velocity = GetMV(s_coord);
            }
        }
        return closest_velocity;
    }
    
    void ComputeNeighborhoodMinMax(sampler sInput, float2 texcoord, out float4 color_min, out float4 color_max)
    {
        float2 pSize = ReShade::PixelSize;
        float4 center_color = GetLod(sInput, texcoord);
        
        center_color.rgb = TAA_Compress(center_color.rgb);
        
        color_min = center_color;
        color_max = center_color;
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            [unroll]
            for (int y = -1; y <= 1; y++)
            {
                if (x == 0 && y == 0)
         
                    continue;
                float4 s = GetLod(sInput, texcoord + float2(x,y) * pSize);
                s.rgb = TAA_Compress(s.rgb);

                color_min = min(color_min, s);
                color_max = max(color_max, s);
            }
        }
    }
    
    float4 ComputeTAA(VS_OUTPUT input, sampler sHistoryParams)
    {
        float2 viewUV = input.uv;
        float depth = GetDepth(viewUV);
        
        if (depth >= 0.999)
            return 0.0;
            
        float2 lowres_uv = viewUV * RenderResolution;
        float4 current_reflection = GetLod(sReflection, lowres_uv);
        
        if (!EnableSmoothing)
            return current_reflection;
            
        float2 velocity = GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        
        if (any(saturate(reprojected_view_uv) != reprojected_view_uv) || FRAME_COUNT <= 1)
            return current_reflection;
            
        float4 history_reflection = GetLod(sHistoryParams, reprojected_view_uv);

        float history_depth = GetDepth(reprojected_view_uv);
        if (abs(history_depth - depth) > 0.05) 
            return current_reflection;

        // Tone Compression
        float3 current_compressed = TAA_Compress(current_reflection.rgb);
        float3 history_compressed = TAA_Compress(history_reflection.rgb);
        
        // Anti-Smear Logic
        float vel_mag = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float motion_sensitivity = EnableAntiSmear ? 3.0 : 0.5;
        float static_factor = saturate(1.0 - vel_mag * motion_sensitivity);

        // Neighborhood Clamping
        float4 color_min, color_max;
        ComputeNeighborhoodMinMax(sReflection, lowres_uv, color_min, color_max);
        
        float relax_amount = 0.15 * static_factor;
        color_min -= relax_amount;
        color_max += relax_amount;
        
        float3 clipped_history_rgb = ClipToAABB(color_min.rgb, color_max.rgb, history_compressed);
        float clipped_history_a = clamp(history_reflection.a, color_min.a, color_max.a);

        // Anti-Ghosting
        float3 diff = abs(current_compressed - clipped_history_rgb);
        float luma_diff = max(diff.r, max(diff.g, diff.b));

        float min_feedback = EnableAntiSmear ? 0.30 : 0.85;
        float dynamic_feedback = lerp(0.97, min_feedback, saturate(luma_diff * 30.0));
        
        float final_static_factor = EnableAntiSmear ? static_factor : (static_factor * static_factor);
        float final_feedback = lerp(dynamic_feedback, 0.99, final_static_factor);
        
        // Confidence
        float flowConfidence = GetFlowConf(viewUV);
        float boosted_conf = saturate(flowConfidence + log2(2.0 - flowConfidence) * 0.3);
        
        final_feedback *= boosted_conf;
        final_feedback = clamp(final_feedback, 0.0, 0.93);
        
        // Final Blend
        float3 result_compressed = lerp(current_compressed, clipped_history_rgb, final_feedback);
        float result_alpha = lerp(current_reflection.a, clipped_history_a, final_feedback);

        return float4(TAA_Resolve(result_compressed), result_alpha);
    }
    
    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_CopyColor(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        outColor = float4(GetColor(input.uv).rgb, GetLocalRoughness(input.uv));
    }
    
    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
        {
            outNormal = float4(0.0, 0.0, 1.0, 1.0);
            return;
        }
        
        float3 normal = CalculateNormal(input.uv, input.pScale);
        if (Vegetation_Protection > 0.0)
        {
            float2 p = ReShade::PixelSize * 1.5;
            float dX = abs(GetDepth(input.uv + float2(p.x, 0)) - depth);
            float dY = abs(GetDepth(input.uv + float2(0, p.y)) - depth);
            float dX_inv = abs(GetDepth(input.uv - float2(p.x, 0)) - depth);
            float dY_inv = abs(GetDepth(input.uv - float2(0, p.y)) - depth);
            float depthNoise = dX + dY + dX_inv + dY_inv;
            float threshold = (1.0 - Vegetation_Protection) * 0.05;
            if (depthNoise > threshold)
            {
                normal = float3(0.0, 0.0, -1.0);
            }
        }

        outNormal = float4(normal, depth);
    }
    
    void PS_SmoothNormals_H(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        float4 centerNormal = GetLod(sNormal, input.uv);
        if (centerNormal.a >= 0.999)
        {
            outNormal = centerNormal;
            return;
        }
        outNormal = ComputeSmoothedNormal(input.uv, float2(1, 0), sNormal);
    }

    void PS_SmoothNormals_V(VS_OUTPUT input, out float4 outNormal : SV_Target)
	{
		float4 centerNormal = GetLod(sNormal1, input.uv);
		if (centerNormal.a >= 0.999)
		{
			outNormal = centerNormal;
			return;
		}
    
		float4 smoothed = ComputeSmoothedNormal(input.uv, float2(0, 1), sNormal1);
    
		float depth = smoothed.a; 
    
		float3 finalNormal = ApplySurfaceDetails(input.uv, smoothed.rgb, depth);
		outNormal = float4(finalNormal, depth);
	}
    
    void PS_TraceReflections(VS_OUTPUT input, out float4 outReflection : SV_Target)
    {
        // Virtual Resolution
        float2 scaled_uv = input.uv / RenderResolution;
        
        // ==========================================
        // TAA Upscaling Jitter
        // ==========================================
        if (EnableTAAUpscaling)
        {
            // 8-Phase Halton Sequence
            const float2 jitterOffsets[8] = {
                float2( 0.125, -0.375), float2(-0.125,  0.375),
                float2(-0.375, -0.125), float2( 0.375,  0.125),
                float2( 0.375, -0.375), float2(-0.375,  0.375),
                float2( 0.125,  0.125), float2(-0.125, -0.125)
            };
            
            uint jitter_idx = uint(fmod((float)FRAME_COUNT, 8.0));
            
            float2 jitter = jitterOffsets[jitter_idx] * (ReShade::PixelSize / RenderResolution);
            
            scaled_uv += jitter;
        }

        if (any(scaled_uv < 0.001) || any(scaled_uv > 0.999))
        {
            outReflection = 0.0;
            return;
        }

        float depth = GetDepth(scaled_uv);
        if (depth >= 1.0)
        {
            outReflection = 0.0;
            return;
        }
        
        //GBuffer 
        float4 gbuffer = SampleGBuffer(scaled_uv);
        float2 pScale = input.pScale;
        float3 normal = normalize(gbuffer.rgb);
        float3 viewPos = UVToViewPos(scaled_uv, depth, pScale);
        float3 viewDir = -normalize(viewPos);
        //Surface Orientation
        bool showFloor = (ReflectionMode == 0 || ReflectionMode == 3 || ReflectionMode == 4);
        bool showWall = (ReflectionMode == 1 || ReflectionMode == 4);
        bool showCeil = (ReflectionMode == 2 || ReflectionMode == 3 || ReflectionMode == 4);

        float orientationIntensity = 0.0;
        if (normal.y > OrientationThreshold && showFloor)
            orientationIntensity = 1.0;
        else if (normal.y < -OrientationThreshold && showCeil)
            orientationIntensity = 1.0;
        else if (abs(normal.y) <= OrientationThreshold && showWall)
            orientationIntensity = 1.0;
        if (orientationIntensity <= 0.0)
        {
            outReflection = 0.0;
            return;
        }

        //Ray Setup
        Ray r;
        r.origin = viewPos;
        r.direction = normalize(reflect(-viewDir, normal));
        
        float bias = 0.0005 + (depth * 0.02);
        // Tenho que analisar isso melhor, as vezes bom, as vezes ruim...
        r.origin += r.direction * bias;
        float VdotN = dot(viewDir, normal);
        if (VdotN > 0.9 || r.direction.z < 0.0)
        {
            outReflection = 0.0;
            return;
        }

        // Blue Noise Jitter 
        uint pixelIndex = uint((input.vpos.y / RenderResolution) * BUFFER_WIDTH + (input.vpos.x / RenderResolution));
        uint perFrameSeedBase = uint(FRAME_COUNT);

        float3 blueNoiseSeed = float3(
            frac(pixelIndex * 0.1031),
            frac(pixelIndex * 0.11369),
            frac(pixelIndex * 0.13787)
        );
        float3 rand = toroidalJitter(sequence3D(perFrameSeedBase), blueNoiseSeed);
        float jitter = rand.z;
        
        // Quality 
        int ray_steps = 16;
        float max_dist = 4.0;
        
        if (RayTraceQuality == 1)
        {
            ray_steps = 32;
            max_dist = 12.0;
        }
        else if (RayTraceQuality == 2)
        {
            ray_steps = 128;
            max_dist = 100.0;
        }

        //Ray Tracing
        float geoThickness = GetThickness(scaled_uv, normal, normalize(viewPos), depth);
        HitResult hit = TraceRay2D(r, ray_steps, max_dist, pScale, jitter, geoThickness);

        float3 reflectionColor = 0.0;
        float reflectionAlpha = 0.0;
        float estimatedRoughness = GetLod(sTexColorCopy, scaled_uv).a;
        
        if (hit.found)
        {
            reflectionColor = GetGlossySample(hit.uv, scaled_uv, estimatedRoughness, rand.xy); 
            // Calculate Fading
            float distFactor = saturate(1.0 - length(hit.viewPos - viewPos) / 10.0);
            float fadeRange = max(FadeDistance, 0.001);
            float depthFade = saturate((FadeDistance - depth) / fadeRange);
            depthFade *= depthFade;
            float2 edgeDist = min(hit.uv, 1.0 - hit.uv);
            float screenFade = smoothstep(0.0, 0.001, min(edgeDist.x, edgeDist.y));
            reflectionAlpha = distFactor * depthFade * screenFade;

            // Masking
            float3 nR = SampleGBuffer(scaled_uv + float2(ReShade::PixelSize.x, 0)).rgb;
            float3 nD = SampleGBuffer(scaled_uv + float2(0, ReShade::PixelSize.y)).rgb;
            float edgeDelta = length(normal - nR) + length(normal - nD);
            float geoMask = 1.0 - smoothstep(0.05, EDGE_MASK_THRESHOLD, edgeDelta);
            
            reflectionAlpha *= geoMask;
        }
		else
        {
            // Fallback 
            float adaptiveDist = min(depth * 1.2 + 0.012, 10.0);
            float3 fbViewPos = viewPos + r.direction * adaptiveDist;
            float2 uvFb = saturate(ViewPosToUV(fbViewPos, pScale).xy);

            reflectionColor = GetGlossySample(uvFb, scaled_uv, estimatedRoughness, rand.xy);
            
            float baseAlpha = smoothstep(0.0, 0.2, 1.0 - scaled_uv.y);
            float ghostKiller = smoothstep(0.0, 0.4, SurfaceGlossiness + (estimatedRoughness * 0.5));
            reflectionAlpha = baseAlpha * ghostKiller;
        }
        
        float fresnelFade = pow(saturate(dot(-viewDir, r.direction)), 2.0);
        outReflection = float4(Input2Linear(reflectionColor), reflectionAlpha * fresnelFade * orientationIntensity);
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
        // Debug Views
        if (ViewMode != 0)
        {
            if (ViewMode == 1)
                outColor = float4(GetActiveHistory(input.uv).rgb, 1.0);
            else if (ViewMode == 2)
                outColor = float4(SampleGBuffer(input.uv).rgb * 0.5 + 0.5, 1.0);
            else if (ViewMode == 3)
                outColor = SampleGBuffer(input.uv).aaaa;
            else if (ViewMode == 4)
            {
                float2 m = GetMV(input.uv);
                float v_mag = length(m) * 100.0;
                float a = atan2(m.y, m.x);
                float3 hsv_color = HSVToRGB(float3((a / (2.0 * PI)) + 0.5, 1.0, 1.0));
                outColor = float4(lerp(float3(0.5, 0.5, 0.5), hsv_color, saturate(v_mag)), 1.0);
            }
            else if (ViewMode == 5)
            {
                float conf = GetFlowConf(input.uv);
                outColor = float4(conf.xxx, 1.0);
            }
            return;
        }

        float depth = GetDepth(input.uv);
        if (depth >= 1.0)
        {
            outColor = GetColor(input.uv);
            return;
        }

        float3 rawScene = GetColor(input.uv).rgb;
        float3 scene = Input2Linear(rawScene);
        float4 gbuffer = SampleGBuffer(input.uv);
        float3 normal = gbuffer.rgb;
        
        float4 reflectionSample = GetActiveHistory(input.uv);
        float3 reflectionColor = reflectionSample.rgb;
        float reflectionMask = reflectionSample.a;

        // Color Grading
        float3 tint = Use_Color_Temperature ?
            KelvinToRGB(Color_Temperature) : SSR_Tint;
        reflectionColor *= tint;
        
        float paper_white_norm = 80.0 / HDR_Peak_Nits;
        float mid_gray = paper_white_norm * 0.18;
        // Contrast
        reflectionColor = (reflectionColor - mid_gray) * SSR_Contrast + mid_gray;
        reflectionColor = max(0.0, reflectionColor);
        
        // Vibrance
        float reflLum = GetLuminance(reflectionColor);
        float3 chroma = reflectionColor - reflLum;
        reflectionColor = reflLum + chroma * (SSR_Vibrance);
        float luma_normalized = saturate(reflLum / (paper_white_norm * 3.0));
        float shadowCurve = 1.0 - smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);
        float highlightCurve = smoothstep(SSR_Split_Balance - 0.2, SSR_Split_Balance + 0.2, luma_normalized);
        
        // Split Tint
        float3 splitTint = shadowCurve * SSR_Shadow_Tint + highlightCurve * SSR_Highlight_Tint;
        reflectionColor = reflectionColor * splitTint;
        float splitTintAlpha = max(splitTint.r, max(splitTint.g, splitTint.b));
        reflectionMask *= saturate(splitTintAlpha);

        //Scene Highlight Preservation
        float sceneLuma = GetLuminance(scene);
        float highlightProtectionMask = smoothstep(paper_white_norm, paper_white_norm * 4.0, sceneLuma);
        reflectionMask *= saturate(1.0 - (highlightProtectionMask * Preserve_Scene_Highlights));

        // PBR & Blending
        float3 viewDir = -normalize(UVToViewPos(input.uv, depth, input.pScale));
        float VdotN = saturate(dot(viewDir, normal));
        float3 f0 = lerp(DIELECTRIC_REFLECTANCE.xxx, scene, Metallic);
        float3 F = F_Schlick(VdotN, f0);

        float3 finalColor;
        if (BlendMode == 0) // Default PBR 
        {
            float validReflection = reflectionMask * saturate(Intensity);
            float3 kD = 1.0 - (F * validReflection);
            kD *= lerp(1.0, 1.0 - Metallic, validReflection);
            float3 specularLight = reflectionColor * F * Intensity * reflectionMask;
            finalColor = (scene * kD) + specularLight;
        }
        else // Legacy Blending modes
        {
            float blendAmount = dot(F, float3(0.333, 0.333, 0.334)) * reflectionMask;
            finalColor = ComHeaders::Blending::Blend(BlendMode, scene, reflectionColor, blendAmount * Intensity);
        }
    
        outColor = float4(Linear2Output(finalColor), 1.0);
    }
    
    technique BaBa_SSR
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
            VertexShader = VS_Accumulate0;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Accumulate1;
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
