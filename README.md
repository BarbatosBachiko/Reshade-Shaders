# Barbatos ReShade Shaders

A collection of advanced shaders for ReShade, focused on Ambient Occlusion (AO), Screen-Space Reflections (SSR), Sharpening, and visual enhancements.

Developed and adapted by **Barbatos Bachiko**.
> **Note:** Artificial Intelligence is used as an assistant in the development, optimization, and coding of these shaders.

## 📂 Package Contents

### 🌑 Ambient Occlusion

* **Barbatos_XeGTAO.fx**
    * Implementation based on **Intel XeGTAO** (Ground Truth Ambient Occlusion).
* **NeoSSAO.fx**
    * A Screen-Space Ambient Occlusion using *ray marching*.
* **MiAO.fx**
    * A simple and performant ambient occlusion implementation, repurposing concepts from FidelityFX CACAO.
    * Ideal for those seeking a balance between quality and performance.
* **DepthDarkening.fx**
    * Enhances depth perception by applying *unsharp masking* to the depth buffer.
    * Based on the paper by Luft, Colditz, and Deussen. Creates a subtle contact shadowing effect.

### ✨ Screen-Space Reflections (SSR)

* **Barbatos_SSR.fx**
    * Full, high-quality screen-space reflections.
    * Supports PBR materials (metallic/roughness), TAA (Temporal Anti-Aliasing) for stability, and normal-based surface details.
* **Barbatos_SSR_Lite.fx**
    * Optimized version of SSR focused on mobile GPUs or systems with fewer resources.
    * Maintains essential functionality with lower processing cost.

### 🗡️ Sharpening & Anti-Aliasing

* **Barbatos_NVSharpen.fx**
    * Implementation of the **NVIDIA Image Scaling (NIS)** algorithm.
    * High-quality adaptive and directional sharpening with edge detection to avoid artifacts.
* **JaSharpen.fx**
    * Sharpening based on *convolution kernels* and *unsharp masking*. Simple and effective.
* **DAA.fx (Directional Anti-Aliasing)**
    * Edge-aware spatiotemporal anti-aliasing technique.
    * Smoothes jaggies by applying directional blur based on local gradient detection.

### 🎨 Tone Mapping & Colors

* **VividTone.fx**
    * Transforms ordinary visuals into vibrant, high-contrast scenes.
    * Includes static/adaptive exposure, *shadow lift*, and fake HDR controls.
* **uFakeHDR.fx**
    * Simple luminance-based HDR emulation to remove the "gray" look from some games.

### 🌊 Motion & Optical Flow

* **BarbatosFlow.fx**
    * Motion estimation (Optical Flow). Generates motion vectors that can be used by other shaders for temporal effects.
    * Based on LumaFlow.
* **SoftMotion.fx**
    * Emulates frame interpolation or *motion blur* using the generated motion vectors.
    * Can create a sense of fluidity or stylized motion blur.

### 🖌️ Stylized Effects

* **S_Outline.fx**
    * Outline detection based on Depth and/or Color.
    * Includes an animated "Wobble" effect.

---

## ⚙️ Installation

1.  Download the repository.
2.  Copy the `.fx` files from the `Shaders` folder to the `Shaders` folder of your ReShade installation.
3.  Copy the `Textures` folder (if present, e.g., `SS_BN.png`) to the `Textures` folder of your ReShade installation.
4.  In-game, open the ReShade menu and enable the desired effects.

## ⚠️ Depth Buffer Requirements

Most of these shaders (XeGTAO, NeoSSAO, SSR, DepthDarkening) **require** correct access to the game's Depth Buffer.

Ensure you configure the global preprocessor definitions correctly in the "Edit Global Preprocessor Definitions" tab of ReShade:

* `RESHADE_DEPTH_LINEARIZATION_FAR_PLANE`: Set the game's maximum render distance.
* `RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN`: Set to 0 or 1 depending on whether the depth buffer appears vertically inverted.
* `RESHADE_DEPTH_INPUT_IS_REVERSED`: Set to 1 if the game uses a reversed depth buffer (common in modern DX11/12 games).
* `RESHADE_DEPTH_INPUT_IS_LOGARITHMIC`: Required for some games with a logarithmic buffer.

For a complete guide, refer to: [ReShade Depth Guide - Marty's Mods](https://guides.martysmods.com/reshade/depth/)

## 📝 Credits and Licenses

* **Barbatos Bachiko**: Adaptation, optimization, and development of the listed shaders.
* **Intel Corporation**: Original XeGTAO code (MIT License).
* **NVIDIA Corporation**: Original NIS Sharpen code (MIT License).
* **AlucardDH**: Normal smoothing code (MIT License).
* **Umar-afzaal (Kaidô)**: Original Optical Flow code (LumaFlow).
* **Thomas Luft, Carsten Colditz, Oliver Deussen**: Theoretical basis for Depth Darkening.

---

**Community and Support:**
Join our Discord to discuss shaders, ask questions, and share your work!
[**DISCORD**](https://discord.gg/7Cq5jvSamu)
