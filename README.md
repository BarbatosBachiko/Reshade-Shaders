# Barbatos ReShade Shaders

A collection of shaders for ReShade, focused on Global Illumination (GI), Ambient Occlusion (AO), Screen-Space Reflections (SSR), Sharpening, and visual enhancements.

Developed and adapted by **Barbatos**.
> **Note:** Artificial Intelligence is used as an assistant in the development, optimization, and coding of these shaders.

## 📂 Package Contents

### ☀️ Global Illumination & Ambient Occlusion

* **BaBa_GI.fx**
    * Screen-Space Global Illumination for high performance.
* **BaBa_XeGTAO.fx**
    * Implementation based on **Intel XeGTAO** (Ground Truth Ambient Occlusion).
* **NeoSSAO.fx**
    * A Screen-Space Ambient Occlusion using Ray Tracing.
* **MiAO.fx**
    * A simple and performant ambient occlusion implementation, repurposing concepts from FidelityFX CACAO.
    * Ideal for those seeking a balance between quality and performance.
* **DepthDarknening.fx**
    * Enhances depth perception by applying *unsharp masking* to the depth buffer.

### ✨ Screen-Space Reflections (SSR)

* **BaBa_SSR.fx**
    * Full, high-quality screen-space reflections.
    * Supports PBR materials (metallic/roughness), TAA (Temporal Anti-Aliasing) for stability, and normal-based surface details.
* **BaBa_SSR_Lite.fx**
    * Optimized version of SSR focused on mobile GPUs or systems with fewer resources.
    * Maintains essential functionality with lower processing cost.

### 🗡️ Sharpening, Anti-Aliasing & Image Quality

* **BaBa_NVSharpen.fx**
    * Implementation of the **NVIDIA Image Scaling (NIS)** algorithm.
    * High-quality adaptive and directional sharpening with edge detection to avoid artifacts.
* **BaBa_NeuralSharpen.fx**
    * Advanced sharpening technique designed to enhance fine details intelligently without over-sharpening noise or introducing heavy ringing.
* **BaBa_Deband.fx**
    * Smooths out color banding artifacts (gradient stepping) often found in compressed skies or low-bit-depth textures.
* **JaSharpen.fx**
    * Sharpening based on *convolution kernels* and *unsharp masking*. Simple and effective.
* **DAA.fx (Directional Anti-Aliasing)**
    * Edge-aware spatiotemporal anti-aliasing technique.
    * Smoothes jaggies by applying directional blur based on local gradient detection.

### 🎨 Tone Mapping & Colors

* **PHDR.fx**
    * Implementation to balance luminance, recover shadow details, and manage highlights for to LDR monitors. It's not true HDR.
* **VividTone.fx**
    * Transforms ordinary visuals into vibrant, high-contrast scenes.
    * Includes static/adaptive exposure, *shadow lift*, and fake HDR controls.
* **uFakeHDR.fx**
    * Adding saturation, contrast, and luminance using LUTs created from neural networks.

### 🌊 Motion & Optical Flow

* **BaBa_Flow.fx**
    * High-quality motion estimation (Optical Flow). Generates motion vectors that can be used by other shaders for temporal effects.
    * Based on LumaFlow.
* **BaBa_Flow_Lite.fx**
    * A lightweight, faster version of the optical flow estimation, balancing performance and motion accuracy.
* **SoftMotion.fx**
    * Emulates frame interpolation or *motion blur* using the generated motion vectors.
    * Can create a sense of fluidity or stylized motion blur.

### 🖌️ Stylized Effects

* **S_Outline.fx**
    * Outline detection based on Depth and/or Color.
    * Includes an animated "Wobble" effect.

---

## ⚙️ Installation

1. Download the repository.
2. Copy the `.fx` and `.fxh` files from the `Shaders` folder to the `reshade-shaders\Shaders` folder of your ReShade installation.
3. Copy the `Textures` folder (containing assets like `SS_BN.png`, `Barbatos_Hilbert_RGB.png`, etc.) to the `reshade-shaders\Textures` folder.
4. In-game, open the ReShade menu and enable the desired effects.

## ⚠️ Depth Buffer Requirements

Most of these shaders (GI, XeGTAO, NeoSSAO, SSR, Depth Darkening, Outlines) **require** correct access to the game's Depth Buffer.

Ensure you configure the global preprocessor definitions correctly in the "Edit Global Preprocessor Definitions" tab of ReShade:

* `RESHADE_DEPTH_LINEARIZATION_FAR_PLANE`: Set the game's maximum render distance.
* `RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN`: Set to 0 or 1 depending on whether the depth buffer appears vertically inverted.
* `RESHADE_DEPTH_INPUT_IS_REVERSED`: Set to 1 if the game uses a reversed depth buffer (common in modern DX11/12 games).
* `RESHADE_DEPTH_INPUT_IS_LOGARITHMIC`: Required for some games with a logarithmic buffer.

For a complete guide, refer to: [ReShade Depth Guide - Marty's Mods](https://guides.martysmods.com/reshade/depth/)

## 📝 Credits and Licenses

* **Barbatos Bachiko**: Original development, adaptation, and optimization of the listed shaders.
* **Intel Corporation**: Original XeGTAO code (MIT License).
* **NVIDIA Corporation**: Original NIS Sharpen code (MIT License).
* **Umar-afzaal (Kaidô)**: Original Optical Flow code (LumaFlow).

---

**Community and Support:**
Join our Discord to discuss shaders, ask questions, and share your work!
[**DISCORD**](https://discord.gg/7Cq5jvSamu)
