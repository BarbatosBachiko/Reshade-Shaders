# Barbatos ReShade Shaders

ReShade effects for lighting, reflections, sharpening, tone, motion, and anti-aliasing.

Author: **Barbatos**

---

## Contents

File names and in-game names differ in a few places, so both are listed.

### Shared pipeline

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_Launcher.fx** | BaBa: Launcher | Builds depth, surface normals, and motion vectors once and shares them with every effect below. Enable it **first**. |

### Global illumination and ambient occlusion

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_GI.fx** | BaBa: GI | Screen-space global illumination with optional AO and directional shadows. |
| **BaBa_XeGTAO.fx** | BaBa: XeGTAO | Ambient occlusion based on Intel XeGTAO. |
| **BaBa_NeoSSAO.fx** | BaBa: NeoSSAO | Screen-space ambient occlusion with ray-traced sampling. |
| **BaBa_MiAO.fx** | BaBa: MiAO | Lighter ambient occlusion (FidelityFX CACAO–inspired). |

### Screen-space reflections

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_SSR.fx** | BaBa: SSR | Full SSR with material controls, glossy sampling, temporal stability, and color grading. |
| **BaBa_SSR_Lite.fx** | BaBa: SSR Lite | Lighter SSR with multi-ray tracing, temporal accumulation, masking, and color grading. |

### Sharpening, anti-aliasing, and cleanup

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_Sharpen_NIS.fx** | BaBa: NVSharpen | Adaptive sharpening based on NVIDIA Image Scaling (NIS). |
| **BaBa_Sharpen_Neural.fx** | BaBa: Neural Sharpen | Neural-network-based sharpening with selectable models. |
| **BaBa_Sharpen_Residual.fx** | BaBa: Sharpen | Multi-scale residual / unsharp-style sharpening. |
| **BaBa_Deband.fx** | BaBa: Deband | Reduces color banding in gradients (skies, soft lighting, etc.). |
| **BaBa_DLAA-T.fx** | BaBa: DTLAA | Directionally Localized Anti-Aliasing, with optional temporal accumulation. |

### Tone and color

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_PHDR.fx** | BaBa: PHDR | Luminance balancing and highlight/shadow control for SDR displays. Not true HDR. |
| **BaBa_VividTone.fx** | BaBa: Vivid Tone | Exposure, contrast, and related tone controls. |
| **BaBa_FakeHDR.fx** | BaBa: Fake HDR | Contrast / saturation / luminance look using LUT-based grading. |

### Stylized

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_Outline.fx** | BaBa: S Outline | Depth and/or color outlines, with an optional wobble animation. |

### Legacy

| Shader | In the ReShade menu | What it does |
|---|---|---|
| **BaBa_Flow.fx** | BaBa: Flow (Legacy) | Standalone optical-flow motion estimation. Superseded by the Launcher. |
| **BaBa_Flow_Lite.fx** | BaBa: Flow Lite (Legacy) | Faster, lighter optical-flow variant. Superseded by the Launcher. |

---

## Installation

1. Copy the contents of `Shaders/` (the `.fx` files **and** the `Includes/` folder) into your ReShade `reshade-shaders\Shaders` folder. Keep the `Includes/` layout so includes resolve correctly.
2. Copy the contents of `Textures/` into `reshade-shaders\Textures`.
3. In-game, open the ReShade menu and enable the effects you want.

If you are updating from an older release, delete the old `.fx` files and the old `bb_include/` folder first. Most shaders were renamed, so leaving them behind gives you two copies of the same effect.

---

## Suggested preset order

1. `BaBa_Launcher.fx` — always first
2. Lighting / AO / SSR
3. Anti-aliasing (`BaBa_DLAA-T.fx`)
4. Tone / color
5. Sharpen, deband, outline last

Exact order depends on the look you want.

---

## Motion vectors

`BaBa_Launcher.fx` is the motion vector provider, and it also prepares the shared depth and normal buffers. Enable it above GI / SSR / AO / DTLAA and you are done — no preprocessor setup required.

Do not run another motion provider alongside it. That includes `BaBa_Flow.fx`, `BaBa_Flow_Lite.fx`, Marty Launchpad, Vort, and Lumenite Kernel / QuantMotion.

### Presets made before the Launcher

Add this in ReShade → **Edit global preprocessor definitions**:

```
BABA_USE_LEGACY_PIPELINE=1
```

That restores the old behavior, where each effect builds its own depth and normals and `BaBa_Flow` supplies motion. Old `USE_MARTY_LAUNCHPAD_MOTION`, `USE_VORT_MOTION`, `USE_LUMENITE_KERNEL_MOTION` and `USE_LUMENITE_QUANTMOTION` definitions are still recognized in this mode.

Legacy mode is a migration bridge, not a second pipeline — don't combine it with the Launcher in the same preset. Lumenite is a separate package and is not included here.

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
- **Umar-afzaal (Kaidô)** — [LumaFlow](https://github.com/umar-afzaal/LumeniteFX), the original optical flow. The dense optical flow in `BaBa_Launcher.fx`, `BaBa_Flow.fx` and `BaBa_Flow_Lite.fx` is derived from it. Also LumeniteFX compatibility.

**Discord:** [https://discord.gg/7Cq5jvSamu](https://discord.gg/7Cq5jvSamu)
