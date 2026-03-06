/*----------------------------------------------|
| ::      Barbatos Motion Vectors Utils      :: |
|----------------------------------------------*/

#ifndef USE_MARTY_LAUNCHPAD_MOTION
    #define USE_MARTY_LAUNCHPAD_MOTION 0
#endif

#ifndef USE_VORT_MOTION
    #define USE_VORT_MOTION 0
#endif

#ifndef MV_CONFIDENCE_SENSITIVITY
    #define MV_CONFIDENCE_SENSITIVITY 1.0
#endif

//----------------|
// :: Textures :: |
//----------------|

#if USE_MARTY_LAUNCHPAD_MOTION
    namespace Deferred {
        texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
        sampler sMotionVectorsTex { Texture = MotionVectorsTex; };
    }

#elif USE_VORT_MOTION
    texture2D MotVectTexVort { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
    sampler2D sMotVectTexVort
    {
        Texture   = MotVectTexVort;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU  = Clamp;
        AddressV  = Clamp;
    };

#else
    texture texMotionVectors
    {
        Width   = BUFFER_WIDTH;
        Height  = BUFFER_HEIGHT;
        Format  = RG16F;
    };
    sampler sTexMotionVectorsSampler
    {
        Texture   = texMotionVectors;
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
        AddressU  = Clamp;
        AddressV  = Clamp;
    };

    texture tMotionConfidence
    {
        Width   = BUFFER_WIDTH;
        Height  = BUFFER_HEIGHT;
        Format  = R16F;
    };
    sampler sMotionConfidence
    {
        Texture   = tMotionConfidence;
        MagFilter = POINT;
        MinFilter = POINT;
        AddressU  = Clamp;
        AddressV  = Clamp;
    };
#endif

float2 SampleMotionVectors(float2 texcoord)
{
#if USE_MARTY_LAUNCHPAD_MOTION
    return tex2Dlod(Deferred::sMotionVectorsTex, float4(texcoord, 0, 0)).rg;
#elif USE_VORT_MOTION
    return tex2Dlod(sMotVectTexVort,              float4(texcoord, 0, 0)).rg;
#else
    return tex2Dlod(sTexMotionVectorsSampler,     float4(texcoord, 0, 0)).rg;
#endif
}

float2 MV_GetVelocity(float2 texcoord)
{
    float2 pixel_size    = ReShade::PixelSize;
    float  closest_depth = 1.0;
    float2 closest_vel   = 0.0;

    static const float2 offsets[5] = {
        float2( 0,  0),
        float2( 0, -1), float2(-1,  0),
        float2( 1,  0), float2( 0,  1)
    };

    [unroll]
    for (int i = 0; i < 5; i++)
    {
        float2 s = texcoord + offsets[i] * pixel_size;
        float  d = ReShade::GetLinearizedDepth(s);
        if (d < closest_depth)
        {
            closest_depth = d;
            closest_vel   = SampleMotionVectors(s);
        }
    }
    return closest_vel;
}

float MV_GetConfidence(float2 texcoord)
{
#if USE_MARTY_LAUNCHPAD_MOTION || USE_VORT_MOTION
    float2 velocity    = SampleMotionVectors(texcoord);
    float2 prev_uv     = texcoord + velocity;

    // Out-of-bounds reprojection → zero confidence.
    if (any(saturate(prev_uv) != prev_uv))
        return 0.0;

    float2 resolution     = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float  flow_magnitude = length(velocity * resolution);

    // Sub-pixel motion → always trusted.
    if (flow_magnitude <= 0.5)
        return 1.0;

    // Bidirectional consistency.
    float2 dest_velocity    = SampleMotionVectors(prev_uv);
    float  error            = length(velocity - dest_velocity);
    float  normalized_error = (error * 1.25) / (length(velocity) + 1e-6);
    float  consistency_conf = rcp(normalized_error + 1.0);

    // Motion length penalty.
    float length_conf = rcp(flow_magnitude * 0.02 + 1.0);

    // Depth discontinuity penalty.
    float curr_depth  = ReShade::GetLinearizedDepth(texcoord);
    float dest_depth  = ReShade::GetLinearizedDepth(prev_uv);
    float depth_conf  = exp(-abs(curr_depth - dest_depth) * 100.0);

    return saturate(consistency_conf * length_conf * depth_conf);

#else
    // BaBa_Flow: confidence texture.
    return tex2Dlod(sMotionConfidence, float4(texcoord, 0, 0)).r;
#endif
}

float MV_GetConfidenceAO(float2 uv, float2 velocity, float flow_magnitude,
                          float curr_luma, sampler sLumaPrev)
{
    float2 prev_uv = uv + velocity;

    // Out-of-bounds reprojection.
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
        return 0.0;

    // Sub-pixel motion — always trusted.
    if (flow_magnitude <= 0.5)
        return 1.0;

    // Bidirectional consistency.
    float2 dest_velocity    = SampleMotionVectors(prev_uv);
    float  error            = length(velocity - dest_velocity);
    float  normalized_error = (error * MV_CONFIDENCE_SENSITIVITY) / (length(velocity) + 1e-6);
    float  consistency_conf = rcp(normalized_error + 1.0);

    // Motion length penalty.
    float length_conf = rcp(flow_magnitude * 0.02 + 1.0);

    // Photometric luma guard — catches disocclusions / lighting changes.
    float prev_luma        = tex2D(sLumaPrev, prev_uv).r;
    float photometric_conf = exp(-abs(curr_luma - prev_luma) * 1.5);

    return consistency_conf * length_conf * photometric_conf;
}