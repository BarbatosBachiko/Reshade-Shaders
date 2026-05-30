# AGENTS.md

## Cursor Cloud specific instructions

### What this repository is

Barbatos ReShade Shaders is an **asset pack** (HLSL/ReShade FX + PNG textures), not a web app or service. There is no `package.json`, Docker Compose, database, or dev server. Development and testing happen **in a game** with [ReShade](https://reshade.me/) installed.

### Cloud VM limitations

A Cursor Cloud VM can validate **repository integrity** (files, includes, bundled textures, install layout) but **cannot** run the real end-to-end workflow without:

- ReShade injected into a compatible game
- A GPU and graphics drivers
- (For depth-based effects) correct global depth preprocessor definitions per game

### Standard commands (no install step)

There is nothing to `npm install` or `pip install`. After clone/pull, the tree is ready to copy into a ReShade install per [README.md](README.md).

### Quick static validation (optional)

From the repo root, confirm bundled assets and local includes:

```bash
# Bundled textures referenced by shaders
test -f Textures/Barbatos/SS_BN3.png
test -f Textures/Barbatos/Barbatos_Hilbert_RGB.png
test -f Textures/Barbatos/Barbatos_LUT_Atlas.png

# Effect count
find Shaders -maxdepth 1 -name '*.fx' | wc -l   # expect 19
```

### Lint / test / build

| Step | In this repo |
|------|----------------|
| Lint | None configured |
| Unit tests | None configured |
| Build | None; shaders compile at runtime inside ReShade |

### Running the “application” (in-game)

1. Copy `Shaders/` → `<game>/reshade-shaders/Shaders` (merge with existing shaders).
2. Copy `Textures/` → `<game>/reshade-shaders/Textures`.
3. **`BaBa_SSR_Lite.fx`** also needs standard headers from the official [reshade-shaders](https://github.com/crosire/reshade-shaders) pack (`ReShade.fxh`, `ReShadeUI.fxh`, `Blending.fxh`) in the same `Shaders` folder if not already present.
4. Launch the game, open the ReShade overlay, enable an effect.

**Suggested hello-world effects** (no depth buffer required): `JaSharpen.fx`, `uFakeHDR.fx`, `BaBa_Deband.fx`, `PHDR.fx`.

**Depth-dependent effects** (require depth preprocessor setup): `BaBa_GI.fx`, `BaBa_XeGTAO.fx`, `NeoSSAO.fx`, `BaBa_SSR.fx`, `MiAO.fx`, `S_Outline.fx`, `DepthDarknening.fx`. See README depth section and [Marty’s depth guide](https://guides.martysmods.com/reshade/depth/).

### Shader ↔ bundled texture map

| Shaders | Texture |
|---------|---------|
| `BaBa_GI.fx`, `BaBa_SSR.fx` | `SS_BN3.png` |
| `BaBa_XeGTAO.fx`, `MiAO.fx` | `Barbatos_Hilbert_RGB.png` |
| `uFakeHDR.fx` | `Barbatos_LUT_Atlas.png` |

### Gotchas

- `BaBa_NeuralSharpen.fx` includes `bb_reshade.fxh` and `ModelAB_weights.fxh` from `bb_include/`; keep that folder alongside the `.fx` files.
- Most shaders use `.\bb_include\...` includes; paths are Windows-style but work with ReShade on Linux when laid out like the README.
- `SS_BN2.png` is present in the repo but not referenced by current shaders (optional asset).
