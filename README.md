## Shaders

### 1. Barbatos NVSharpen
* A ReShade implementation of the NVIDIA Image Scaling (NIS) SDK's sharpening algorithm. This shader uses the original NVIDIA source code (under MIT license). Its goal is to increase image clarity and detail through a directional and adaptive Unsharp Mask (USM) method that intelligently applies the effect based on scene characteristics.
* **Repository:** https://github.com/BarbatosBachiko/Reshade-Shaders/blob/main/Shaders/BarbatosNVSharpen.fx
* **Demo:**
   ![DarkSoulsRemastered 2025-07-07 23-54-58_633 After](https://github.com/user-attachments/assets/000e4f43-39d0-48e5-8b8d-622afc8fc43d)

Important depth buffer settings you need to know:

* `RESHADE_DEPTH_LINEARIZATION_FAR_PLANE`
* `RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN`
* `RESHADE_DEPTH_INPUT_IS_REVERSED`
* `RESHADE_DEPTH_INPUT_IS_LOGARITHMIC`

For a complete guide on how to properly configure depth buffer access, see the guide:
**[ReShade Depth Guide - Marty's Mods](https://guides.martysmods.com/reshade/depth/)**

## Where to find me
Join our Discord community to discuss shaders, ask questions, and share your work!
https://discord.gg/99BtcdSUA4
