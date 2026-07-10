# Barbatos ReShade Shaders

ReShade effects for lighting, reflections, sharpening, tone, motion, and anti-aliasing.

Author: **Barbatos**

---

## Contents

### Global illumination and ambient occlusion

| Shader | What it does |
|---|---|
| **BaBa_GI.fx** | Screen-space global illumination with optional AO and directional shadows. |
| **BaBa_XeGTAO.fx** | Ambient occlusion based on Intel XeGTAO. |
| **BaBa_NeoSSAO.fx** | Screen-space ambient occlusion with ray-traced sampling. |
| **BaBa_MiAO.fx** | Lighter ambient occlusion (FidelityFX CACAO–inspired). |

### Screen-space reflections

| Shader | What it does |
|---|---|
| **BaBa_SSR.fx** | Full SSR with material controls, glossy sampling, temporal stability, and color grading. |
| **BaBa_SSR_Lite.fx** | Lighter SSR with temporal denoise and color grading. Lower cost than the full SSR. |

### Sharpening, anti-aliasing, and cleanup

| Shader | What it does |
|---|---|
| **BaBa_Sharpen_NIS.fx** | Adaptive sharpening based on NVIDIA Image Scaling (NIS). |
| **BaBa_Sharpen_Neural.fx** | Neural-network-based sharpening with selectable models. |
| **BaBa_Sharpen_Residual.fx** | Multi-scale residual / unsharp-style sharpening. |
| **BaBa_Deband.fx** | Reduces color banding in gradients (skies, soft lighting, etc.). |
| **BaBa_DTLAA.fx** | Directionally Localized Anti-Aliasing, with optional temporal accumulation. Enable a motion-vector effect earlier in the preset when using temporal mode. |

### Tone and color

| Shader | What it does |
|---|---|
| **BaBa_PHDR.fx** | Luminance balancing and highlight/shadow control for SDR displays. Not true HDR. |
| **BaBa_VividTone.fx** | Exposure, contrast, and related tone controls. |
| **BaBa_FakeHDR.fx** | Contrast / saturation / luminance look using LUT-based grading. |

### Motion and optical flow

| Shader | What it does |
|---|---|
| **BaBa_Flow.fx** | Optical-flow motion estimation for temporal effects (GI, SSR, AO, DTLAA). |
| **BaBa_Flow_Lite.fx** | Faster, lighter optical-flow variant. |

### Stylized

| Shader | What it does |
|---|---|
| **BaBa_Outline.fx** | Depth and/or color outlines, with an optional wobble animation. |

---

## Installation

1. Copy the contents of `Shaders/` (the `.fx` files **and** the `Includes/` folder) into your ReShade `reshade-shaders\Shaders` folder. Keep the `Includes/` layout so includes resolve correctly.
2. Copy the contents of `Textures/` into `reshade-shaders\Textures`.
3. In-game, open the ReShade menu and enable the effects you want.

---

## Suggested preset order

1. Motion vectors (`BaBa_Flow` or `BaBa_Flow_Lite`, or another supported provider below)
2. Lighting / AO / SSR
3. Anti-aliasing (`BaBa_DTLAA`)
4. Tone / color
5. Sharpen, deband, outline last

Exact order depends on the look you want.

---

## Motion vectors (optional providers)

Several effects use motion vectors for temporal stability. Enable **one** provider and place it **before** GI / SSR / AO / DTLAA.

| Provider | How to enable |
|---|---|
| **BaBa_Flow** / **BaBa_Flow_Lite** | Default — no extra setup. |
| **Lumenite Kernel** | Install [LumeniteFX](https://github.com/umar-afzaal/LumeniteFX), enable `lumenite_Kernel.fx`, and set global preprocessor `USE_LUMENITE_KERNEL_MOTION=1`. |
| **Lumenite QuantMotion** | Same pack: enable `lumenite_QuantMotion.fx`, set `USE_LUMENITE_QUANTMOTION=1`. |
| **Marty / Launchpad** | Set `USE_MARTY_LAUNCHPAD_MOTION=1`. |
| **Vort** | Set `USE_VORT_MOTION=1`. |

Preprocessor macros go in ReShade → **Edit global preprocessor definitions**. Do not enable more than one motion provider at once. Lumenite is a separate package and is not included here.

---

## Depth buffer

GI, AO, SSR, and outlines need a correct depth buffer. In ReShade → **Edit global preprocessor definitions**, set as needed for your game:

- `RESHADE_DEPTH_LINEARIZATION_FAR_PLANE`
- `RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN`
- `RESHADE_DEPTH_INPUT_IS_REVERSED`
- `RESHADE_DEPTH_INPUT_IS_LOGARITHMIC`

Guide: [ReShade Depth Guide — Marty's Mods](https://guides.martysmods.com/reshade/depth/)

---

## Credits

- **Barbatos Bachiko** — development and adaptation
- **Intel Corporation** — XeGTAO (MIT)
- **NVIDIA Corporation** — NIS sharpen (MIT)
- **Umar-afzaal (Kaidô)** — optical flow (LumaFlow); LumeniteFX compatibility

**Discord:** [https://discord.gg/7Cq5jvSamu](https://discord.gg/7Cq5jvSamu)
