# Requirements (confirmed contract)

Status: **confirmed with the user**, and implemented. This document is the
contract the implementation is measured against; what has actually been measured
is in `validation.md`. Anything not listed here is out of scope until it is added
here.

## 1. Identity

| Field | Value |
| --- | --- |
| Repository | `ComfyUI-MiniMax-H3-RAVEN-Streaming` |
| Package name | `comfyui-minimax-h3-raven-streaming` |
| Display name | `MiniMax H3 RAVEN Streaming` |
| Runtime Python package | `raven_streaming` |
| Version | `0.1.0` |
| License | MIT (code only) |
| Public repository | **None.** Local checkout only — no remote is configured and no GitHub project exists, so no repository URL is published anywhere in this package. |
| Comfy Registry PublisherId | `yanzuolu` |
| Publishing | **Manual only.** No publish GitHub Action, no CI publishing. |

## 2. Functional requirements

### R1 — T2VA only
Text prompt in, joint video + audio out. The official H3 image-conditioned modes
(first/last keyframe `fl2va`, reference `ref2va`) are **not** exposed in 0.1.x.
The official `MiniMaxH3ImageToVideo` node is reused in its T2VA form, i.e. with
no `first_frame` / `last_frame` inputs connected.

### R2 — Built on ComfyUI's official H3
The model, VAEs, text encoder, conditioning and latent formats come from
ComfyUI's official MiniMax H3 support. This project does **not** fork or vendor
the model implementation.

### R3 — Two nodes only

**N1. `RAVEN Model Loader`**
- Selects the **full, non-pruned BF16 H3 DiT**.
- Selects the **mandatory RAVEN PEFT LoRA**. Without it the node cannot produce
  a model; this is an error, not a warning.
- Returns a **standard `MODEL`**, so stock model nodes keep working downstream.

**N2. `RAVEN Streaming Sampler`**
- Inputs: `MODEL`, official `CONDITIONING`, official H3 AV `LATENT`, video
  `VAE`, audio `VAE`, plus the parameters in R6.
- **Takes no prompt and no CLIP; performs no text encoding.** Conditioning and
  the AV latent are produced upstream by the official node.
- Internally runs a chunk-major consistency loop (not a stock sampler).
- Outputs: `LATENT`, `IMAGE`, `AUDIO`.

No third node is added.

### R4 — Reuse of official nodes
The workflow is completed with stock ComfyUI nodes:

| Official node | Role |
| --- | --- |
| `CLIPLoader` | H3 (Qwen3-VL) text encoder. |
| `MiniMaxH3ImageToVideo` (T2VA, no frame inputs) | `CONDITIONING` + empty AV `LATENT`. |
| `VAELoader` x2 | Video VAE and audio VAE. |
| `LoraLoaderModelOnly` | Optional extra official LoRAs, chained after N1. |

### R5 — Official LoRA chaining
Because N1 returns a standard `MODEL`, additional official H3 LoRAs are stacked
with the stock `LoraLoaderModelOnly`. The package does not own LoRA loading
policy beyond making RAVEN mandatory in N1.

### R6 — Parameters and constraints

| Parameter | Default | Constraint |
| --- | --- | --- |
| `width` | `1376` | Multiple of 32. |
| `height` | `768` | Multiple of 32. |
| area | — | `width * height <= 1376 * 768`. |
| `frames` | `192` | Must satisfy `17k + 5` with **`k >= 1`**: minimum `22`, maximum `362`. `k = 0` (5 frames / 2 latents) is **not supported in v0.1** and must fail loud. Above `192`: allowed but experimental, must emit a warning. |
| `steps` | `4` | Consistency NFE count. |
| `video_shift` | `12` | Video-stream timestep shift. |
| `audio_shift` | `3` | Audio-stream timestep shift. |
| `sink` | `2` | Attention-sink chunks retained. |
| `window` | `2` | Sliding-window chunks retained. |
| sampler / scheduler | — | **Not selectable.** Stock samplers/schedulers are not applicable (see R8). |
| CFG / negative prompt | — | **Not exposed.** |
| batch size | `1` | Fixed. |

Constraints that must be enforced with explicit errors: RAVEN LoRA present,
batch size 1, 32-multiple width/height, the `1376 * 768` area cap, `17k + 5`
frame alignment with `k >= 1` (i.e. `frames >= 22`; a `k = 0` / 5-frame,
2-latent request is rejected, never silently promoted), and the `362` frame
maximum.

### R7 — Streaming
- Chunks are emitted **in order** and an emitted chunk is **never revised**.
- **Video decode:** 5 main latents + 2 lookahead latents per decode, with a
  **5-frame overlap** between consecutive decodes.
- **Latency profile:** one-chunk startup delay; in steady state each new DiT
  chunk releases the **previous 17 frames**; a final **5-frame flush** ends the
  stream.
- **Audio:** overlap-save decoding; the margin was **measured at M0** as 17
  latents (0.425 s of lookahead) at a 28-latent block, matching a full decode to
  `< 2.5e-6`.
- **Transport:** fMP4 segments muxed with PyAV, delivered for Media Source
  Extensions playback; **silence is inserted automatically** when a segment
  carries no audio.
- Streaming means incremental availability only. **No latency or throughput
  guarantee, no real-time SLA.**

### R8 — No stock sampler reuse
A standard ComfyUI sigma-major sampler cannot drive this model configuration.
The sampler node executes the RAVEN **chunk-major, fresh-noise consistency
sampler** internally. Consequences that are part of the contract:

- The incoming `MODEL` must be a **`CoreModelPatcher`**.
- The RAVEN LoRA is applied as an **FP32 activation residual**, i.e. computed on
  activations in FP32 — **not** by fusing `B @ A` into the base weights.

### R9 — Outputs
- `LATENT` — the full H3 AV latent (nested video + audio tensors).
- `IMAGE` — decoded video frames, `[N, H, W, 3]`, 24 fps.
- `AUDIO` — a standard ComfyUI audio dict (`waveform`, `sample_rate`).

## 3. Non-functional requirements

### N1 — Hardware baseline
- **1x NVIDIA H200, single GPU** — the box every measurement in `validation.md`
  was taken on. The reference runs were **not** made on a physically small card:
  an external holder process allocated VRAM on the H200 until only **24.100 GiB**
  was free, so the run had to plan against a real 24 GiB of headroom. That 24 GiB
  is the envelope this package sizes itself against — a simulated ceiling on a
  big card, not a 24 GiB GPU.
- **System RAM: budget 160 GiB available, 192 GB physical.** Measured peak host
  RSS was **129.854 GiB at 192 frames** and **135.025 GiB at 362 frames**, so a
  128 GiB machine is not sufficient — expect swapping or an OOM kill there. The
  host carries the offloaded BF16 base weights plus the FP32 adapter, the KV
  cache (22.62 GB steady / 30.10 GB peak at 192 frames) and the output buffers
  (`IMAGE` 2.43 GB at 192 frames, 4.59 GB at 362); Comfy's reserve accounts for
  none of it.
- **Comfy native partial CPU offload** (stock `ModelPatcher` full/partial CPU
  offload) for the diffusion model. This is the required baseline; cross-device
  LoRA residual application is part of it.
- **DynamicVRAM / aimdo: not supported in v0.1.** The loader forces the static
  patcher and the sampler refuses a `ModelPatcherDynamic` `MODEL` rather than
  run an unverified path. Unavailable on the cu128 environment (vr-1) where the
  stock offload path passed, and absent from cu130 builds as well.
- **NVFP4 text encoder**: GPU encode, then offloaded to CPU.

### N2 — No SLA
No real-time guarantee, no fixed frames-per-second target, no latency budget.

### N3 — Models are external
No weights are distributed here. Model licenses are independent of this
repository's MIT license. Files, folders, URLs and SHA256 checksums are listed
in `README.md`.

### N4 — Compatibility policy
Audited against ComfyUI `0.33.0` at commit
`c67885b14556cf3e4e061862925282d403d09862`, frontend `1.49.6`, `av>=16.0.0`.
The declared target minimum ComfyUI is `0.30`, **pending the feature test** in
`COMPATIBILITY.md` — it is a target, not a verified floor. Behaviour never
branches on version strings.

### N5 — Honest status reporting
Until a milestone is actually validated, documentation says so. No claims of
"tested", "verified" or "published" before the corresponding evidence exists.

## 4. Explicit non-goals

- Real-time playback / latency guarantees.
- Image-conditioned or reference-conditioned generation.
- Multi-GPU, and any guaranteed low-VRAM strategy beyond the Comfy native
  partial CPU offload described in N1.
- `k = 0` (5-frame / 2-latent) generation.
- Training or LoRA extraction tooling.
- Automated Registry publishing.
