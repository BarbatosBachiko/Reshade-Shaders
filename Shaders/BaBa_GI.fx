/*----------------------------------------------|
| ::              Barbatos GI                :: |
|-----------------------------------------------|
| Version: 1.0                                  |
| Author: Barbatos                              |
| License: MIT                                  |
'----------------------------------------------*/
#include "ReShade.fxh"

//----------|
// :: UI :: |
//----------|
//HDR
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
//GI
uniform float Intensity <
    ui_category = "Global Lighting";
    ui_label = "GI Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 1.0;
uniform float Roughness <
    ui_category = "Global Lighting";
    ui_label = "Roughness";
    ui_tooltip = "Controls the glossiness of the reflections.\n0.0 = Mirror, 1.0 = Diffuse.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0; ui_step = 0.01;
> = 1.0;

uniform float Thickness <
    ui_category = "Global Lighting";
    ui_label = "Thickness";
    ui_type = "drag";
    ui_min = 0.01;
    ui_max = 0.2; ui_step = 0.01;
> = 0.02;
uniform float GI_RenderDistance <
    ui_category = "Global Lighting";
    ui_label = "Render Distance";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = 1.0;
uniform int RayCount <
    ui_category = "Global Lighting";
    ui_label = "Rays per Pixel";
    ui_type = "drag";
    ui_min = 1; ui_max = 16; ui_step = 1.0;
> = 2;
uniform int RaySteps <
    ui_category = "Global Lighting";
    ui_label = "Ray Steps";
    ui_type = "drag";
    ui_min = 2;
    ui_max = 16;
> = 6;

uniform float Near_Intensity <
    ui_category = "Global Lighting";
    ui_label = "Near Field Intensity";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
> = 0.8;
uniform float MaxRayDistance <
    ui_category =  "Global Lighting";
    ui_label = "Max Ray Distance";
    ui_type = "drag";
    ui_min = 0.01;
    ui_max = 0.5; ui_step = 0.001;
> = 0.100;
//AO
uniform float AO_Intensity <
    ui_category = "Ambient Occlusion";
    ui_label = "AO Intensity";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0; ui_step = 0.01;
> = 1.0;
uniform float AO_Radius <
    ui_category = "Ambient Occlusion";
    ui_label = "AO Radius";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 5.0;
> = 0.1;

uniform int AO_BlendMode <
    ui_category = "Ambient Occlusion";
    ui_label = "AO Blend Mode";
    ui_type = "combo";
    ui_items = "Multiplicative\0Luminance Masked\0";
> = 0;
// Color Grading
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
uniform float GI_Color_Bleed <
    ui_category = "Color Grading";
    ui_label = "Material Color Bleed";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.8;
uniform float3 GI_Color <
    ui_category = "Color Grading";
    ui_label = "Tint";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float GI_Vibrance <
    ui_category = "Color Grading";
    ui_label = "Saturation";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 2.0;
uniform float GI_Contrast <
    ui_category = "Color Grading";
    ui_label = "Contrast";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
> = 1.0;

uniform float3 GI_Shadow_Tint <
    ui_category = "Color Grading";
    ui_label = "Shadow Color";
    ui_tooltip = "Color of GI in dark areas. Set to Black to disable GI in shadows.";
    ui_type = "color";
> = float3(0.5, 0.5, 0.5);

uniform float3 GI_Highlight_Tint <
    ui_category = "Color Grading";
    ui_label = "Highlight Color";
    ui_tooltip = "Color of GI in bright areas. Set to Black to disable GI in highlights.";
    ui_type = "color";
> = float3(1.0, 1.0, 1.0);

uniform float GI_Split_Balance <
    ui_category = "Color Grading";
    ui_label = "Split Balance";
    ui_tooltip = "Determines the separation point between Shadows and Highlights.";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
> = 0.5;

// Manual Position
uniform bool SSS_Enabled <
    ui_category = "Manual Light";
    ui_label = "Enable Screen Space Shadows";
    ui_tooltip = "Adds directional shadows";
> = false;
uniform float SSS_Intensity <
    ui_category = "Manual Light";
    ui_label = "Shadow Intensity";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
> = 1.0;

uniform bool Manual_Sun_Enabled <
    ui_category = "Manual Light";
    ui_label = "Enable Manual Sun";
> = false;

uniform float Sun_Azimuth <
    ui_category = "Manual Light";
    ui_label = "Sun Rotation (Azimuth)";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 360.0;
    ui_step = 1.0;
> = 175.0;
uniform float Sun_Elevation <
    ui_category = "Manual Light";
    ui_label = "Sun Elevation (Altitude)";
    ui_type = "drag";
    ui_min = -15.0; ui_max = 90.0; ui_step = 1.0;
> = 22.0;

uniform float Shadow_Softness <
    ui_category = "Manual Light";
    ui_label = "Sun Shadow Softness";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
> = 0.1;

uniform float Sun_Shadow_Fill <
    ui_category = "Manual Light";
    ui_label = "Ambient Fill";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;

uniform bool Show_Sun_Widget <
    ui_category = "Manual Light";
    ui_label = "Show Sun Widget";
> = true;

uniform float2 Sun_Widget_Pos <
    ui_category = "Manual Light";
    ui_label = "Widget Position";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = float2(0.88, 0.15);

uniform float Sun_Widget_Scale <
    ui_category = "Manual Light";
    ui_label = "Widget Scale";
    ui_type = "drag";
    ui_min = 0.05; ui_max = 0.5;
    ui_step = 0.001;
> = 0.100;

uniform float RenderScale <
   ui_category = "Advanced";
    ui_label = "Resolution";
    ui_tooltip = "Scales the rendering resolution of GI.";
    ui_type = "drag";
    ui_min = 0.1; ui_max = 1.0; ui_step = 0.001;
> = 0.333;

// Tech
uniform float VERTICAL_FOV <
    ui_category = "Advanced";
    ui_label = "Camera Vertical FOV";
    ui_type = "drag";
    ui_min = 15.0; ui_max = 120.0; ui_step = 0.1;
> = 60.0;

uniform int ViewMode <
    ui_category = "Advanced";
    ui_label = "Debug View";
    ui_type = "combo";
    ui_items = "Off\0GI Only\0AO Only\0Surface Normals\0Motion Vectors\0Raw LowRes GI\0White World\0Luminance\0";
> = 0;

// Defines
#define PI 3.1415927
#define GetLod(s,c) tex2Dlod(s, float4((c).xy, 0, 0))
#define GetColor(c) GetLod(ReShade::BackBuffer, c)
#define fmod(x, y) (frac((x)*rcp(y)) * (y))

#define GW BUFFER_WIDTH
#define GH BUFFER_HEIGHT

static const float DEG2RAD = 0.017453292;
static const float2 TAA_Offsets[5] = { float2(0, 0), float2(0, -1), float2(-1, 0), float2(1, 0), float2(0, 1) };
uniform int FRAME_COUNT < source = "framecount"; >;
uniform float TIMER < source = "timer"; >;
#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 0
#endif

#ifndef USE_MARTY_LAUNCHPAD_MOTION
#define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
#define USE_VORT_MOTION 0
#endif

//----------------|
// :: Textures :: |
//----------------|
// Motion Vectors
#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH;
            Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }
    float2 GetMV(float2 texcoord) { return GetLod(Deferred::sMotionVectorsTex, texcoord).rg;
    }
    
    // Dummy confidence for launchpad 
    float GetFlowConf(float2 texcoord) { return 0.5;
    } 

#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort { Texture = MotVectTexVort; MagFilter=POINT;MinFilter=POINT;MipFilter=POINT;AddressU=Clamp;AddressV=Clamp; };
    float2 GetMV(float2 texcoord) { return GetLod(sMotVectTexVort, texcoord).rg;
    }
    
    // Dummy confidence for vort
    float GetFlowConf(float2 texcoord) { return 0.5;
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

texture tMotionConfidence
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

namespace Barbatos_GI_100
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
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture Accum
    {
        Width = GW;
        Height = GH;
        Format = RGBA16F;
    };
    sampler sAccum
    {
        Texture = Accum;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture History0
    {
        Width = GW;
        Height = GH;
        Format = RGBA16F;
    };
    sampler sHistory0
    {
        Texture = History0;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture History1
    {
        Width = GW;
        Height = GH;
        Format = RGBA16F;
    };
    sampler sHistory1
    {
        Texture = History1;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };

    texture DNA
    {
        Width = GW;
        Height = GH;
        Format = RGBA16F;
    };
    sampler sDNA
    {
        Texture = DNA;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
    };
    texture DNB
    {
        Width = GW;
        Height = GH;
        Format = RGBA16F;
    };
    sampler sDNB
    {
        Texture = DNB;
        AddressU = Clamp;
        AddressV = Clamp;
        MagFilter = LINEAR;
        MinFilter = LINEAR;
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

    //---------------------|
    // :: Vertex Shaders ::|
    //---------------------|
    struct VS_OUTPUT
    {
        float4 vpos : SV_Position;
        float2 uv : TEXCOORD0;
        float2 pScale : TEXCOORD1;
    };
    
    void VS_Barbatos_PTGI(in uint id : SV_VertexID, out VS_OUTPUT outStruct)
    {
        outStruct.uv.x = (id == 2) ?
            2.0 : 0.0;
        outStruct.uv.y = (id == 1) ? 2.0 : 0.0;
        outStruct.vpos = float4(outStruct.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
        
        float y = tan(max(0.01, VERTICAL_FOV) * DEG2RAD * 0.5);
        outStruct.pScale = float2(y * ReShade::AspectRatio, y);
    }
    
    //-----------------|
    // :: Functions :: |
    //-----------------|

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

    float3 sRGB2Linear(float3 x)
    {
        return (x < 0.04045) ?
            (x / 12.92) : pow(abs((x + 0.055) / 1.055), 2.4);
    }

    float3 Linear2sRGB(float3 x)
    {
        return (x < 0.0031308) ?
            (12.92 * x) : (1.055 * pow(abs(x), 1.0 / 2.4) - 0.055);
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

    float3 GetFalseColor(float luminance)
    {
        float3 color = float3(0.0, 0.0, 0.0);
        if (luminance < 0.25)
            color = lerp(float3(0, 0, 1), float3(0, 1, 1), luminance * 4.0);
        else if (luminance < 0.5)
            color = lerp(float3(0, 1, 1), float3(0, 1, 0), (luminance - 0.25) * 4.0);
        else if (luminance < 0.75)
            color = lerp(float3(0, 1, 0), float3(1, 1, 0), (luminance - 0.5) * 4.0);
        else
            color = lerp(float3(1, 1, 0), float3(1, 0, 0), (luminance - 0.75) * 4.0);
        return color;
    }

    float3 GetSunVector()
    {
        float az = radians(Sun_Azimuth);
        float el = radians(Sun_Elevation);
        float x = sin(az) * cos(el);
        float y = sin(el);
        float z = cos(az) * cos(el);
        return normalize(float3(x, y, z));
    }

    float GetDepth(float2 xy)
    {
        return ReShade::GetLinearizedDepth(xy);
    }

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
    
    //------------|
    // :: Noise ::|
    //------------|
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

    //---------------------------|
    // :: View Space & Normal :: |
    //---------------------------|
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

    void genTB(float3 N, out float3 T, out float3 B)
    {
        float s = N.z < 0.0 ?
            -1.0 : 1.0;
        float a = -1.0 / (s + N.z);
        float b = N.x * N.y * a;
        T = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
        B = float3(b, s + N.y * N.y * a, -N.y);
    }

    float3 cosineSample(float3 N, float2 r)
    {
        float3 T, B;
        genTB(N, T, B);
        r.x *= 2.0 * PI;
        float s = sqrt(max(0.0, 1.0 - r.y));
        float2 sincos_rx;
        sincos(r.x, sincos_rx.x, sincos_rx.y);
        return T * (sincos_rx.y * s) + B * (sincos_rx.x * s) + N * sqrt(r.y);
    }

    // GGX Importance Sampling 
    float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
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

    //-------------------|
    // :: Ray Tracing  ::|
    //-------------------|
    bool TraceRay(float3 origin, float3 dir, float2 pScale, int steps, float jitter, out float2 hitUV, out float3 hitPos, out float hitDist)
    {
        float3 lastPos = origin;
        float3 current = origin;

        [loop]
        for (int i = 1; i <= steps; i++)
        {
            float t = (float(i) - 1.0 + jitter) / float(steps);
            t = t * t;
            float distScale = MaxRayDistance * t;
            current = origin + dir * distScale;
            hitUV = ViewPosToUV(current, pScale);
        
            if (any(hitUV < 0.0) || any(hitUV > 1.0))
                return false;
            float zScene = GetDepth(hitUV);
            float zRay = current.z;
            float depthDiff = zRay - zScene;
            if (zScene < 0.999 && depthDiff > 0.0 && depthDiff < Thickness)
            {
                // BINARY SEARCH
                float3 startPos = lastPos;
                float3 endPos = current;
                float3 midPos;
       
                for (int r = 0; r < 4; r++)
                {
                    midPos = (startPos + endPos) * 0.5;
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
                hitUV = ViewPosToUV(current, pScale);
                hitPos = current;
                hitDist = length(current - origin);
                return true;
            }
            lastPos = current;
        }
        return false;
    }

    //------------|
    // :: TAA  :: |
    //------------|
    
    float3 TAA_Compress(float3 color)
    {
        return color / (1.0 + color);
    }

    float3 TAA_Resolve(float3 color)
    {
        return color / max(1e-6, 1.0 - color);
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
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float depth = GetDepth(viewUV);
        if (depth >= 0.999)
            return float4(0.0, 0.0, 0.0, 1.0);
        float4 current_gi = GetLod(sAccum, input.uv);
        
        float2 velocity = GetVelocity(viewUV);
        float2 reprojected_view_uv = viewUV + velocity;
        float2 reprojected_buffer_uv = reprojected_view_uv * RenderScale;

        if (any(saturate(reprojected_view_uv) != reprojected_view_uv) || FRAME_COUNT <= 1)
            return current_gi;
        float4 history_gi = GetLod(sHistoryParams, reprojected_buffer_uv);
        
        float3 current_compressed = TAA_Compress(current_gi.rgb);
        float3 history_compressed = TAA_Compress(history_gi.rgb);

        float vel_mag = length(velocity * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
        float static_factor = saturate(1.0 - vel_mag * 0.5);
        float4 color_min, color_max;
        ComputeNeighborhoodMinMax(sAccum, input.uv, color_min, color_max);
        float relax_amount = 0.15 * static_factor;
        color_min -= relax_amount;
        color_max += relax_amount;
        float3 clipped_history_rgb = ClipToAABB(color_min.rgb, color_max.rgb, history_compressed);
        float clipped_history_a = clamp(history_gi.a, color_min.a, color_max.a);

        float3 diff = abs(current_compressed - clipped_history_rgb);
        float luma_diff = max(diff.r, max(diff.g, diff.b));
        float dynamic_feedback = lerp(0.97, 0.85, saturate(luma_diff * 30.0));
        float final_feedback = lerp(dynamic_feedback, 0.99, static_factor * static_factor);
        float3 result_compressed = lerp(current_compressed, clipped_history_rgb, final_feedback);
        float result_alpha = lerp(current_gi.a, clipped_history_a, final_feedback);
        
        return float4(TAA_Resolve(result_compressed), result_alpha);
    }
    
    //--------------------|
    // :: Pixel Shaders ::|
    //--------------------|
    void PS_CopyColor(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        outColor = GetColor(input.uv);
    }

    void PS_GenNormals(VS_OUTPUT input, out float4 outNormal : SV_Target)
    {
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float d = GetDepth(viewUV);
        if (d >= 0.999)
        {
            outNormal = float4(0, 0, 1, d);
            return;
        }
        float3 normal = CalculateNormal(viewUV, input.pScale);
        outNormal = float4(normal, d);
    }
    
    void PS_Trace(VS_OUTPUT input, out float4 outGI : SV_Target)
    {
        if (any(input.uv > RenderScale))
            discard;
        float2 viewUV = input.uv / RenderScale;

        float4 gbuffer = GetLod(sNormal, input.uv);
        float depth = gbuffer.a;
        if (depth >= 0.999 || depth > GI_RenderDistance)
        {
            outGI = float4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        float3 normal = gbuffer.rgb;
        float3 viewPos = UVToViewPos(viewUV, depth, input.pScale);
        float3 totalRadiance = 0.0;
        float totalVisibility = 0.0;

        uint pixelIndex = uint((input.vpos.y / RenderScale) * BUFFER_WIDTH + (input.vpos.x / RenderScale));
        uint perFrameSeedBase = uint(FRAME_COUNT) * RayCount;

        float3 blueNoiseSeed = float3(
        frac(pixelIndex * 0.1031),
        frac(pixelIndex * 0.11369),
        frac(pixelIndex * 0.13787)
    );
        float bias = (depth * 0.002) + 0.0005;
        float3 rayOrigin = viewPos + normal * bias;
        float3 sunDir = float3(0, 1, 0);
        
        // View Vector for Reflection Calculation
        float3 V = normalize(-viewPos);
        if (Manual_Sun_Enabled || SSS_Enabled)
            sunDir = GetSunVector();
        [loop]
        for (int s = 0; s < RayCount; s++)
        {
            uint currentSeed = perFrameSeedBase + s;
            float3 rand = toroidalJitter(sequence3D(currentSeed), blueNoiseSeed);
            float3 rayDir;

            if (Manual_Sun_Enabled)
            {
                float3 jitter = (rand - 0.5) * Shadow_Softness;
                rayDir = normalize(sunDir + jitter);
                if (dot(normal, rayDir) <= 0.0)
                {
                    totalVisibility += Sun_Shadow_Fill;
                    continue;
                }
            }
            else
            {
                // GGX Sampling 
                float3 H = ImportanceSampleGGX(rand.xy, normal, Roughness);
                rayDir = reflect(-V, H);
                
                // Fallback 
                if (dot(normal, rayDir) <= 0.0)
                    rayDir = cosineSample(normal, rand.xy);
            }
    
            float2 hitUV;
            float3 hitPos;
            float hitDist;

            if (TraceRay(rayOrigin, rayDir, input.pScale, RaySteps, rand.z, hitUV, hitPos, hitDist))
            {
                float3 hitNormal = tex2Dlod(sNormal, float4(hitUV, 0, 0)).rgb;
                bool validHit = Manual_Sun_Enabled ? true : (dot(rayDir, hitNormal) < 0.1);
                if (validHit)
                {
                    float3 rawAlbedo = tex2Dlod(sTexColorCopy, float4(hitUV, 0, 3.0)).rgb;
                    if (!Manual_Sun_Enabled)
                        totalRadiance += Input2Linear(rawAlbedo);
                    else
                        totalRadiance += Input2Linear(rawAlbedo) * Sun_Shadow_Fill;
                }
        
                float distFactor = saturate(hitDist / max(0.001, AO_Radius));
                float weight_falloff = saturate(1.0 - distFactor * distFactor); // Quadratic
                float weight = Manual_Sun_Enabled ?
                    Sun_Shadow_Fill : weight_falloff;

                totalVisibility += weight;
            }
            else
            {
                if (Manual_Sun_Enabled)
                {
                    totalVisibility += 1.0;
                    totalRadiance += 0.1;
                }
            }
        }

        float invRays = 1.0 / float(max(1, RayCount));
        float finalVisibility;

        if (Manual_Sun_Enabled)
        {
            finalVisibility = totalVisibility * invRays;
            finalVisibility = lerp(1.0, finalVisibility, AO_Intensity);
        }
        else
        {
            finalVisibility = 1.0 - saturate(totalVisibility * invRays * AO_Intensity);
            if (SSS_Enabled)
            {
                float3 rand = toroidalJitter(sequence3D(perFrameSeedBase), blueNoiseSeed);
                float3 jitter = (rand - 0.5) * Shadow_Softness;
                float3 shadowRayDir = normalize(sunDir + jitter);
                if (dot(normal, shadowRayDir) > 0.0)
                {
                    float2 sUV;
                    float3 sPos;
                    float sDist;

                    bool hit = TraceRay(rayOrigin, shadowRayDir, input.pScale, RaySteps, rand.z, sUV, sPos, sDist);
                    if (hit)
                    {
                        finalVisibility *= (1.0 - SSS_Intensity);
                    }
                }
                else
                {
                    finalVisibility *= (1.0 - (SSS_Intensity * 0.5));
                }
            }
        }

        outGI = float4(totalRadiance * invRays, finalVisibility);
    }

    float4 AtrousFilter(VS_OUTPUT input, sampler sInputTex, float stepWidth)
    {
        if (any(input.uv > RenderScale))
            discard;
        float4 c_data = GetLod(sInputTex, input.uv);
        float3 c_val = c_data.rgb;
        float c_ao = c_data.a;
        
        float4 c_gbuffer = GetLod(sNormal, input.uv);
        float3 c_norm = c_gbuffer.rgb;
        float c_depth = c_gbuffer.a;
        
        static const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };
        float4 sum = float4(c_val, c_ao);
        float cum_w = 1.0;
        
        float2 px = ReShade::PixelSize * stepWidth;
        float depth_weight_factor = 1.0 / (0.1 * c_depth + 1e-6);
        [unroll]
        for (int x = -2; x <= 2; x++)
        {
            [unroll]
            for (int y = -2; y <= 2; y++)
            {
                if (x == 0 && y == 0)
         
                    continue;
                float2 uv_offset = input.uv + float2(x, y) * px;
                float4 s_data = GetLod(sInputTex, uv_offset);
                float4 s_gbuffer = GetLod(sNormal, uv_offset);
                float3 s_norm = s_gbuffer.rgb;
                float s_depth = s_gbuffer.a;

                float w_z = exp(-abs(c_depth - s_depth) * depth_weight_factor);
                float dotN = max(0.0, dot(c_norm, s_norm));
                dotN = pow(dotN, 4.0);
                float w_n = dotN;
                
                float k_w = kernel[abs(x)] * kernel[abs(y)];
                float weight = w_z * w_n * k_w;
                
                sum += s_data * weight;
                cum_w += weight;
            }
        }
        return sum / max(cum_w, 0.0001);
    }
    
    void PS_Accumulate0(VS_OUTPUT input, out float4 outAccum : SV_Target)
    {
        if (FRAME_COUNT % 2 != 0)
            discard;
        outAccum = ComputeTAA(input, sHistory1);
    }

    void PS_Accumulate1(VS_OUTPUT input, out float4 outAccum : SV_Target)
    {
        if (FRAME_COUNT % 2 == 0)
            discard;
        outAccum = ComputeTAA(input, sHistory0);
    }

    void PS_Atrous1(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        if (FRAME_COUNT % 2 == 0)
            outColor = AtrousFilter(input, sHistory0, 1.0);
        else
            outColor = AtrousFilter(input, sHistory1, 1.0);
    }
    
    void PS_AtrousFinal(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        outColor = AtrousFilter(input, sDNA, 3.0);
    }

    float sdTorus(float3 p, float2 t)
    {
        float2 q = float2(length(p.xz) - t.x, p.y);
        return length(q) - t.y;
    }
    
    float GetBit(int n, int b)
    {
        return fmod(floor(float(n) / exp2(float(b))), 2.0);
    }
    
    float GetDigit(float2 uv, int d)
    {
        int font[10] =
        {
            31599, // 0
            9362, // 1
            29671, // 2
            29391, // 3
            23497, // 4
            31183, // 5
            31215, // 6
            29257, // 7
            31727, // 8
            31695 // 9
        };
        
        int2 ip = int2(floor(uv * float2(3.0, 5.0)));
        if (ip.x < 0 || ip.x > 2 || ip.y < 0 || ip.y > 4)
            return 0.0;
        
        int bit = (4 - ip.y) * 3 + (2 - ip.x);
        return GetBit(font[d], bit);
    }
    
    float PrintNumber(float2 uv, float2 pos, float size, int number)
    {
        float2 localUV = (uv - pos) / size;
        if (localUV.y < 0.0 || localUV.y > 1.0)
            return 0.0;
        
        float res = 0.0;
        int d1 = (number / 100) % 10;
        int d2 = (number / 10) % 10;
        int d3 = number % 10;
       
        float spacing = 0.4; 
        
        // Digit 1
        float2 digitUV = localUV;
        digitUV.x = (localUV.x) * 2.0;
        if (number >= 100)
            if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
                res += GetDigit(digitUV, d1);
             
        // Digit 2
        digitUV.x = (localUV.x - 0.6) * 2.0;
        if (number >= 10)
            if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
                res += GetDigit(digitUV, d2);
             
        // Digit 3
        digitUV.x = (localUV.x - 1.2) * 2.0;
        if (digitUV.x >= 0.0 && digitUV.x <= 1.0)
            res += GetDigit(digitUV, d3);
        
        return res;
    }

    float4 DrawSunWidget(float2 texcoord, float3 sunDir, float4 sceneColorLinear)
    {
        float2 uv = texcoord - Sun_Widget_Pos;
        uv.x *= ReShade::AspectRatio;
        uv /= Sun_Widget_Scale;

        float distCenter = length(uv);
        if (distCenter > 2.0)
            return sceneColorLinear;

        float3 ro = float3(0, 0, -3.5);
        float3 rd = normalize(float3(uv, 2.0));
        
        // Tilt camera to see the floor
        float thV = radians(30.0);
        float cV = cos(thV);
        float sV = sin(thV);
        float3x3 mTilt = float3x3(1, 0, 0, 0, cV, -sV, 0, sV, cV);
        ro = mul(mTilt, ro);
        rd = mul(mTilt, rd);

        float3 p = ro;
        float t = 0.0;
        bool hit = false;
        int objID = 0;
        float glowAcc = 0.0;
        
        float radAz = radians(-Sun_Azimuth);
        float radEl = radians(Sun_Elevation);
        
        // Calculate Sun Position 
        float3 sunPos = float3(
            sin(radAz) * cos(radEl),
            sin(radEl),
            cos(radAz) * cos(radEl)
        ) * 1.2;

        for (int i = 0; i < 60; i++)
        {
            p = ro + rd * t;
            
            // Center Cross 
            float dCross = min(length(p.xy), min(length(p.xz), length(p.yz))) - 0.01;
            float dAnchor = max(length(p) - 0.2, dCross);
            
            //Compass Ring
            float dCompass = abs(length(p.xz) - 1.2) - 0.02;
            dCompass = max(dCompass, abs(p.y) - 0.01);
            
            // Elevation Ring 
            float3 pElv = p;
            float cA = cos(radAz);
            float sA = sin(radAz);
            pElv.xz = float2(pElv.x * cA - pElv.z * sA, pElv.x * sA + pElv.z * cA);
            float dElvRing = length(float2(length(pElv.zy) - 1.2, pElv.x)) - 0.015;
            
            // Sun Sphere
            float dSun = length(p - sunPos) - 0.15;
            
            float dScene = min(dAnchor, min(dCompass, min(dElvRing, dSun)));
            
            glowAcc += 1.0 / (1.0 + dSun * dSun * 100.0);
            
            if (dScene < 0.002)
            {
                hit = true;
                if (dScene == dSun)
                    objID = 1; // Sun
                else if (dScene == dCompass)
                    objID = 2; // Horizontal Ring
                else if (dScene == dElvRing)
                    objID = 3; // Vertical Ring
                else
                    objID = 4; // Center
                break;
            }
            t += dScene * 0.8;
            if (t > 8.0)
                break;
        }

        float shadowMask = smoothstep(1.8, 0.4, distCenter) * 0.6;
        float3 finalColor = sceneColorLinear.rgb * (1.0 - shadowMask);

        if (hit)
        {
            float3 N = normalize(p);
            // Fake simple lighting
            float3 L = normalize(float3(0.5, 1.0, -0.5));
            float NdotL = max(0.2, dot(N, L));
            
            float3 objColor = float3(0, 0, 0);
            
            if (objID == 1) // Sun
            {
                objColor = float3(1.0, 0.9, 0.5) * 4.0; // Emission
            }
            else if (objID == 2) // Compass Ring 
            {
                objColor = float3(1.0, 0.5, 0.0);
            }
            else if (objID == 3) // Vertical Ring 
            {
                objColor = float3(0.0, 0.8, 1.0);
            }
            else if (objID == 4) // Center 
            {
                objColor = float3(0.5, 0.5, 0.5);
            }
            
            finalColor = objColor;
        }

        float3 glowColor = float3(1.0, 0.6, 0.2) * glowAcc * 0.05;
        finalColor += glowColor;
        
        // Text: Rotation 
        float numMask = PrintNumber(uv, float2(0.0, -1.2), 0.15, int(Sun_Azimuth));
        // Text: Elevation
        numMask += PrintNumber(uv, float2(0.0, 1.2), 0.15, int(Sun_Elevation));
        
        finalColor = lerp(finalColor, float3(1.0, 1.0, 1.0), numMask);
        
        float edgeFade = 1.0 - smoothstep(1.4, 1.6, distCenter);
        return lerp(sceneColorLinear, float4(finalColor, 1.0), edgeFade);
    }

    void PS_Output(VS_OUTPUT input, out float4 outColor : SV_Target)
    {
        float depth = GetDepth(input.uv);
        float3 rawScene = GetColor(input.uv).rgb;
        float3 scene = Input2Linear(rawScene);
        float3 finalColor = scene;
        if (depth < 0.99)
        {
            float4 giData = tex2Dlod(sDNB, float4(input.uv * RenderScale, 0, 0));
            // Tint
            float3 tint = Use_Color_Temperature ?
                KelvinToRGB(Color_Temperature) : GI_Color;
            float3 processedGI = giData.rgb * tint;
            
            //HDR Contrast
            float paper_white_norm = 80.0 / HDR_Peak_Nits;
            float mid_gray = paper_white_norm * 0.18;
            processedGI = (processedGI - mid_gray) * GI_Contrast + mid_gray;
            processedGI = max(0.0, processedGI);
            // Vibrance
            float lum = GetLuminance(processedGI);
            float3 chroma = processedGI - lum;
            processedGI = lum + chroma * (1.0 + GI_Vibrance);
            float fadeStart = GI_RenderDistance * 0.9;
            float fade = 1.0 - smoothstep(fadeStart, GI_RenderDistance, depth);
            float depthWeight = lerp(Near_Intensity, 1.0, saturate(depth * 10.0));

            //  Debug
            if (ViewMode != 0)
            {
                if (ViewMode == 6) // White World
                {
                    scene = float3(0.8, 0.8, 0.8);
                }
                else
                {
                    if (ViewMode == 1)
                        outColor = float4(Linear2Output(processedGI), 1.0);
                    else if (ViewMode == 2)
                        outColor = float4(giData.aaa, 1.0);
                    else if (ViewMode == 3)
                        outColor = float4(GetLod(sNormal, input.uv * RenderScale).rgb * 0.5 + 0.5, 1.0);
                    else if (ViewMode == 4)
                    {
                        float2 mv = GetMV(input.uv);
                        outColor = float4(saturate(float3(mv.x, mv.y, 0.0) * 50.0 + 0.5), 1.0);
                    }
                    else if (ViewMode == 5)
                        outColor = float4((FRAME_COUNT % 2 == 0 ? GetLod(sHistory0, input.uv * RenderScale).rgb : GetLod(sHistory1, input.uv * RenderScale).rgb), 1.0);
                    else if (ViewMode == 7)
                    {
                        float lum = GetLuminance(processedGI);
                        outColor = float4(GetFalseColor(saturate(lum)), 1.0);
                    }

                    if (Manual_Sun_Enabled && Show_Sun_Widget)
                    {
                        float3 debugLinear = Input2Linear(outColor.rgb);
                        float4 widgetRes = DrawSunWidget(input.uv, GetSunVector(), float4(debugLinear, 1.0));
                        outColor = float4(Linear2Output(widgetRes.rgb), 1.0);
                    }
                    return;
                }
            }
        
            float rawAO = saturate(giData.a);
            float finalAO = 1.0;
        
            if (AO_BlendMode == 0)
            {
                finalAO = lerp(1.0, rawAO, AO_Intensity);
            }
            else
            {
                float sceneLum = GetLuminance(scene);
                float brightMask = saturate(sceneLum / paper_white_norm);
                finalAO = lerp(1.0, lerp(rawAO, 1.0, brightMask), AO_Intensity);
            }

            finalAO = lerp(1.0, finalAO, depthWeight);
            finalAO = lerp(1.0, finalAO, fade);
            float3 occludedScene = scene * finalAO;
            float3 bouncedLight = processedGI * Intensity * depthWeight * fade;
            //Split Toning
            float sceneLuma = GetLuminance(scene);
            float luma_normalized = saturate(sceneLuma / (paper_white_norm * 3.0));

            float shadowCurve = 1.0 - smoothstep(GI_Split_Balance - 0.2, GI_Split_Balance + 0.2, luma_normalized);
            float highlightCurve = smoothstep(GI_Split_Balance - 0.2, GI_Split_Balance + 0.2, luma_normalized);

            float3 shadowIntegration = (lerp(1.0, scene, GI_Color_Bleed) + 0.02);
            float3 shadowLight = bouncedLight * shadowIntegration * shadowCurve * GI_Shadow_Tint;
            float3 litLight = bouncedLight * scene * highlightCurve * GI_Highlight_Tint;
            finalColor = occludedScene + shadowLight + litLight;
        }

        // Widget Overlay
        if (Manual_Sun_Enabled && Show_Sun_Widget)
        {
            float4 widgetRes = DrawSunWidget(input.uv, GetSunVector(), float4(finalColor, 1.0));
            finalColor = widgetRes.rgb;
        }

        outColor = float4(Linear2Output(finalColor), 1.0);
    }
    
    technique Barbatos_GI
    {
        pass CopyColorGenMips
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_CopyColor;
            RenderTarget = TexColorCopy;
        }
        pass Normals
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_GenNormals;
            RenderTarget = Normal;
        }
        pass Trace
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Trace;
            RenderTarget = Accum;
        }
        pass Accumulate0
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Accumulate0;
            RenderTarget = History0;
        }
        pass Accumulate1
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Accumulate1;
            RenderTarget = History1;
        }
        pass DenoiseStep1
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Atrous1;
            RenderTarget = DNA;
        }
        pass DenoiseStep2
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_AtrousFinal;
            RenderTarget = DNB;
        }
        pass Output
        {
            VertexShader = VS_Barbatos_PTGI;
            PixelShader = PS_Output;
        }
    }
}
