# ComfyUI Remote MiniMax H3

[English](README.md) | [日本語](README_JP.md)

**Run MiniMax H3's CLIP / Text Encoder processing on another PC to reduce VRAM usage on your main ComfyUI machine.**

This custom node allows you to split MiniMax H3 processing across two PCs connected over a local network.

Instead of loading the large MiniMax H3 Text Encoder and related models on the same GPU as the UNET, this node sends the required inputs to a second ComfyUI instance, performs the H3 processing there, and returns the resulting data to the main PC.

The large model files remain on the remote PC. Only the processing results are transmitted over the network.

---

## Why?

MiniMax H3 requires a very large Text Encoder and additional models.

On a GPU with limited VRAM, loading everything into a single ComfyUI instance can cause:

- VRAM exhaustion
- Frequent model offloading
- GPU ↔ CPU memory transfers
- Severe performance degradation
- ComfyUI UI becoming sluggish or unresponsive
- Difficulty running the UNET and upscaling/refining stages

If you have a second PC with enough VRAM for the H3 models, you can use it to offload the H3 processing.

### Before

```text
┌──────────────────────────────┐
│          PC-A                 │
│                              │
│  CLIP / Text Encoder         │
│  MiniMax H3 Text Encoder     │
│  Video VAE                   │
│  UNET                        │
│  Upscaler / Refiner          │
│                              │
│       VRAM OVERLOAD          │
└──────────────────────────────┘
```

### With Remote MiniMax H3

```text
┌─────────────────────┐
│       PC-A          │
│                     │
│       UNET          │
│  Upscaler / Refiner │
│                     │
└──────────┬──────────┘
           │
           │ LAN
           │
           │ Processing data
           │
           ▼
┌─────────────────────┐
│       PC-B          │
│                     │
│  CLIP               │
│  H3 Text Encoder    │
│  Video VAE          │
│                     │
└─────────────────────┘
```

This allows PC-A's GPU to focus primarily on the UNET and sampling workload.

---

## How it works

The main PC sends the required information to the remote ComfyUI instance.

PC-B then:

1. Loads the specified CLIP
2. Loads the specified VAE
3. Performs MiniMax H3 processing
4. Generates the required conditioning / latent data
5. Sends the results back to PC-A

The large model files remain on PC-B.

```text
PC-A
  │
  │ Prompt
  │ CLIP/VAE names
  │ First Frame
  │ Last Frame
  │ Width / Height
  │ Length
  │
  ▼
PC-B
  │
  │ CLIP
  │ H3 Text Encoder
  │ Video VAE
  │ MiniMax H3 processing
  │
  ▼
PC-A
  │
  │ Conditioning / Latent
  ▼
UNET / Sampler
```

---

## Network traffic

One of the important advantages of this approach is that **the large models are never transferred between PCs**.

In one actual test:

```text
PC-A → PC-B : 4.60 MB
PC-B → PC-A : 15.01 MB

Total       : 19.61 MB
```

The MiniMax H3 Text Encoder used in the test was approximately **25.9 GB staged on PC-B**, while the returned result was about **15 MB**.

This makes the approach practical for a local high-speed network.

---

## Example hardware configuration

A typical configuration might look like:

```text
PC-A
GPU: RTX 3090 24GB

Responsible for:
- UNET
- Sampling
- Latent Upscaling
- Refiner
```

```text
PC-B
GPU: GPU with sufficient VRAM for MiniMax H3

Responsible for:
- CLIP
- MiniMax H3 Text Encoder
- MiniMax H3 Video VAE
- H3 preprocessing
```

The two machines run separate ComfyUI instances.

PC-B should be started with:

```text
--listen
```

so that PC-A can communicate with it over the network.

---

## Performance / VRAM

The main purpose of this project is **VRAM reduction**, rather than simply increasing raw GPU compute performance.

By moving the large H3-related models to PC-B, PC-A can keep significantly more VRAM available for UNET processing.

In testing, a 24GB RTX 3090 was able to perform a time-sliced sampling workflow using approximately:

```text
~17 GB VRAM
```

after the H3-related processing was moved to the second PC.

This can significantly reduce VRAM pressure, model offloading, and the resulting performance degradation.

---

## Seed variations

The CLIP / Text Encoder result does not normally depend on the sampling seed.

Therefore, when generating multiple variations with different seeds, the expensive H3 conditioning calculation can potentially be reused while the seed-dependent sampling remains on PC-A.

Conceptually:

```text
                 ┌─ Seed 1 ─┐
                 ├─ Seed 2 ─┤
H3 Conditioning ─┼─ Seed 3 ─┼─ UNET / Sampling
                 ├─ Seed 4 ─┤
                 └─ Seed 5 ─┘
```

This is particularly useful when performing large numbers of seed variations.

---

## Requirements

- ComfyUI
- MiniMax H3 support installed on both PCs
- Two PCs connected through a network
- PC-B running ComfyUI with `--listen`
- Sufficient VRAM on PC-B to load the MiniMax H3 models
- Network connectivity between PC-A and PC-B

Both PCs should have compatible ComfyUI environments and the required MiniMax H3 models installed.

---

## Installation

Clone or copy this repository into the `custom_nodes` directory of the ComfyUI installation.

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI_RemoteMiniMaxH3/
        ├── __init__.py
        └── ...
```

Install the custom node on both PCs if both instances need the node.

Restart ComfyUI after installation.

---

## Configuration

The node requires two inputs:

| Input | Description |
|---|---|
| `PC Name` | Hostname or IP address of the PC running the remote ComfyUI |
| `CLIP File Name` | CLIP model used by MiniMax H3 |

Example:

```text
PC Name:
Z840

CLIP:
MiniMax\qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

The corresponding VAE is handled by the remote H3 processing node.

---

## Important

This project is designed primarily for **local/private network use**.

Do not expose the remote ComfyUI server directly to the public Internet without appropriate network security measures.

---

## Project Status

**Experimental / Work in Progress**

This project was created to solve a specific VRAM limitation when running MiniMax H3 across multiple PCs.

It has been tested successfully with a two-PC setup where:

- PC-A handles the UNET / sampling workload
- PC-B handles CLIP / H3 Text Encoder / Video VAE processing
- Only processing data is transferred between the machines

Your hardware, ComfyUI version, model versions, and network configuration may produce different results.

---

## Why I made this

The project started with a simple problem:

> **"One GPU doesn't have enough VRAM for everything."**

Instead of transferring huge models between PCs, the idea was to let each GPU specialize in the part of the workflow it can handle best.

The result was a surprisingly effective distributed MiniMax H3 workflow:

```text
        PC-A                         PC-B
┌─────────────────┐          ┌─────────────────┐
│      UNET       │          │      CLIP       │
│    Sampling     │◄────────►│  H3 Text Encoder│
│    Upscaling    │   LAN    │   Video VAE     │
│     Refiner     │          │                 │
└─────────────────┘          └─────────────────┘
```

**Instead of moving the model, move the result.**

---

## Notes

This project was created primarily to reduce VRAM usage when running MiniMax H3 workflows across multiple PCs.

The amount of data transferred between PCs is intended to be much smaller than transferring the large model weights themselves. The exact amount of transferred data depends on the workflow and ComfyUI implementation.

This project is provided **as-is**.

It is primarily a personal/experimental utility and is not intended to imply ongoing maintenance or support. Compatibility with future versions of ComfyUI is not guaranteed.

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

This project depends on the MiniMax H3 implementation included in ComfyUI (`comfy_extras.nodes_minimax_h3`). ComfyUI and its components are subject to their respective licenses.

This project does not include or redistribute MiniMax H3 model weights. MiniMax H3 models are subject to their respective licenses.

See [`LICENSE`](LICENSE) for the full GPL-3.0 license text.

## Acknowledgements

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- MiniMax H3 and its associated model/software components
