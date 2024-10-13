////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//* Directionally Fast Temporal Anti-Aliasing (DFTAA) by Barbatos Bachiko                                                                                                        
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uniform int View_Mode < 
    ui_type = "combo";
    ui_items = "DFTAA Out\0Mask View A\0Mask View B\0"; 
    ui_label = "View Mode"; 
    ui_tooltip = "Select normal output or debug view."; 
> = 0; // Default view mode as DFTAA Out

uniform float EdgeThreshold < 
    ui_type = "slider";
    ui_label = "Edge Threshold"; 
    ui_tooltip = "Adjust the edge threshold for mask creation."; 
    ui_min = 0.0; 
    ui_max = 1.0; 
    ui_default = 0.1; 
> = 0.1; // Adjustable edge threshold

uniform float Lambda < 
    ui_type = "slider";
    ui_label = "Lambda"; 
    ui_tooltip = "Adjust the lambda for saturation amount."; 
    ui_min = 0.0; 
    ui_max = 10.0; 
    ui_default = 3.0; 
> = 3.0; // Adjustable lambda

#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)

texture BackBufferTex : COLOR;
sampler BackBuffer
{
    Texture = BackBufferTex;
};

texture PreviousFrameTex : COLOR;
sampler texPrevious
{
    Texture = PreviousFrameTex;
};

// Function to load pixels
float4 LoadPixel(sampler tex, float2 tc)
{
    return tex2D(tex, tc);
}

// Function to apply DFTAA 
float4 ApplyDFTAA(float4 center, float4 left, float4 right)
{
    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);

    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    const float satAmountH = saturate((Lambda * LumH - 0.1f) / LumH);

    return lerp(center, (combH + center) / 3.0, satAmountH * 0.5f);
}

// Function to apply a blur to the mask
float4 BlurMask(float4 mask, float2 tc)
{
    float4 blur = mask * 0.25; // Start with current mask weight
    blur += LoadPixel(BackBuffer, tc + float2(-pix.x, 0)) * 0.125; // Left
    blur += LoadPixel(BackBuffer, tc + float2(pix.x, 0)) * 0.125; // Right
    blur += LoadPixel(BackBuffer, tc + float2(0, -pix.y)) * 0.125; // Up
    blur += LoadPixel(BackBuffer, tc + float2(0, pix.y)) * 0.125; // Down
    return saturate(blur); // Prevent overflow
}

// Main DFTAA pass
float4 Out(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    const float4 center = LoadPixel(BackBuffer, texcoord);
    const float4 left = LoadPixel(BackBuffer, texcoord + float2(-1.0 * pix.x, 0));
    const float4 right = LoadPixel(BackBuffer, texcoord + float2(1.0 * pix.x, 0));

    float4 DFTAA = ApplyDFTAA(center, left, right);

    // Mask calculation for DFTAA
    const float4 maskColorA = float4(1.0, 1.0, 0.0, 1.0); // Yellow for Mask A

    const float4 combH = left + right;
    const float4 centerDiffH = abs(combH - 2.0 * center);
    const float LumH = dot(centerDiffH.rgb, float3(0.333, 0.333, 0.333));
    
    const float maskA = LumH > EdgeThreshold ? 1.0 : 0.0;

    // Mask View A
    if (View_Mode == 1)
    {
        return BlurMask(maskA * maskColorA, texcoord); // Returns blurred mask A in yellow
    }

    // Logic for Mask View B (additional edge detection)
    const float4 diff = abs(DFTAA - center);
    const float maxDiff = max(max(diff.r, diff.g), diff.b);
    const float maskB = maxDiff > EdgeThreshold ? 1.0 : 0.0;

    if (View_Mode == 2)
    {
        return BlurMask(maskB * float4(1.0, 0.0, 0.0, 1.0), texcoord); // Returns blurred mask B in red
    }

    return DFTAA; // Final DFTAA output
}

// FSMAA implementation
float4 SMAALumaEdgeDetectionPS(float2 texcoord)
{
    float4 pixel = LoadPixel(BackBuffer, texcoord);
    float luminance = dot(pixel.rgb, float3(0.299, 0.587, 0.114)); // Luminance calculation
    return float4(luminance, luminance, luminance, 1.0); // Returns luminance as grayscale for edge detection
}

// FSMAA Edge Detection
float4 FSAAMain(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 edgeDetection = SMAALumaEdgeDetectionPS(texcoord);
    
    // Just return DFTAA output by default
    return Out(position, texcoord); // Final DFTAA output
}

// TAA implementation
float4 TAA(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
   
    static float2 JitterPattern[4] =
    {
        float2(0.0, 0.0),
        float2(0.25, 0.25),
        float2(0.75, 0.25),
        float2(0.25, 0.75)
    };
    
    static int frameIndex = 0; 
    int jitterIndex = frameIndex % 4;
    float2 jitter = JitterPattern[jitterIndex] / float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT); 

    float4 previousColor = tex2D(texPrevious, texcoord);
    float4 currentColor = tex2D(BackBuffer, texcoord + jitter);
    
    frameIndex++; 

    return lerp(previousColor, currentColor, 0.5); 
}

// Vertex shader generating a triangle covering the entire screen
void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Unique technique for optimized DLAA + FSMAA + TAA
technique DFTAA
{
    pass DFTAA_Light
    {
        VertexShader = PostProcessVS;
        PixelShader = Out; // Use the DFTAA output
    }

    pass FSMAA_Pass
    {
        VertexShader = PostProcessVS;
        PixelShader = FSAAMain; // Use the FSMAA output
    }

    pass TAA_Pass
    {
        VertexShader = PostProcessVS;
        PixelShader = TAA; // Use the TAA output
    }
}
