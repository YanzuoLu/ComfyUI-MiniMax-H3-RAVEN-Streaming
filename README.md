# ComfyUI-MiniMax-H3-RAVEN-Streaming

Streaming **text-to-video-and-audio (T2VA)** custom nodes for **MiniMax H3**,
built on top of ComfyUI's *official* H3 implementation and a **mandatory RAVEN
LoRA**.

> **Status: implemented, locally validated, not published.** The nodes, the
> chunk-major sampler, the streaming preview and the workflows in `workflows/`
> run end to end on real weights: 1376x768 / 192 frames inside a simulated
> 24 GiB VRAM envelope, the 362-frame maximum, the real Qwen3-VL text lane, a
> public third-party H3 LoRA through the official `LoraLoaderModelOnly`, and
> CPU-pinned/GPU KV modes with bitwise-identical `LATENT` / `IMAGE` / `AUDIO`.
> A delivered-chunk cancellation followed by three normal runs also leaves no
> buffered media and produces bitwise-identical outputs with zero CUDA plateau
> growth. Every measurement is listed in [`docs/validation.md`](docs/validation.md).
> There is no Comfy Registry release and no public git remote.

- Display name: **MiniMax H3 RAVEN Streaming**
- Package name: `comfyui-minimax-h3-raven-streaming`
- Runtime Python package: `raven_streaming/`
- Version: `0.1.0`
- License: MIT (see `LICENSE`) — **model weights are licensed separately, see below**
- Repository: **not published.** There is no public clone URL yet; this tree is
  local only.

---

## Installation

Not published to the Comfy Registry, and **not available from a public git
remote** — do not try to `git clone` it. Install by copying this directory into
ComfyUI's `custom_nodes`:

```bash
cp -R /path/to/ComfyUI-MiniMax-H3-RAVEN-Streaming ComfyUI/custom_nodes/
```

(or symlink it, if you want to keep editing in place:
`ln -s /path/to/ComfyUI-MiniMax-H3-RAVEN-Streaming ComfyUI/custom_nodes/`).

When there *is* a release — a Comfy Registry entry, or a published repository —
this section becomes the one-line install for it. Until then the copy above is
the only supported path.

Then restart ComfyUI. The nodes register through the repository root
`__init__.py` (V1 mappings plus `WEB_DIRECTORY`), so no extra step is needed to
get the in-node preview widget.

### Requirements

| | |
|---|---|
| Python | `>= 3.10` (follows upstream's `requires-python`) |
| PyAV | **`av >= 16.0.0`** — the only runtime dependency this package adds, used for fMP4 muxing of the preview stream |
| Everything else | ComfyUI's own requirements; nothing is vendored |

PyAV ships with ComfyUI's requirements at the pinned baseline, so in a normal
ComfyUI environment there is nothing to install. If you maintain a stripped
environment:

```bash
pip install "av>=16.0.0"
```

The **preview** lane is the only thing PyAV is needed for. If PyAV, the H.264
encoder or the AAC encoder is missing, the node logs the reason and samples
anyway; the `LATENT` / `IMAGE` / `AUDIO` outputs are unaffected.

### Versions

| Component | Version | Status |
|---|---|---|
| ComfyUI | `0.33.0` @ `c67885b14556cf3e4e061862925282d403d09862` | **pinned and verified** — every measurement in `docs/validation.md` was taken here |
| `comfyui-frontend-package` | `1.49.6` | **pinned and verified** — the preview protocol was read and tested against this frontend |
| PyAV | `av >= 16.0.0` (measured on `17.1.0` and `18.1.0`) | **verified** |
| ComfyUI `0.30` | declared as `requires-comfyui = ">=0.30.0"` | **target only, NOT verified.** Never run. The feature probe in [`COMPATIBILITY.md`](COMPATIBILITY.md) decides, not the version string. |

---

## Models

Weights are **not** distributed with this repository, are **not** downloaded
automatically, and are **not** covered by this repository's MIT license. MiniMax
H3 weights and the RAVEN LoRA carry their own upstream licenses and usage terms;
read and accept them before use.

Everything goes in the **standard ComfyUI model directories** — no custom folder,
no custom path config:

```
ComfyUI/models/
├── diffusion_models/   # MiniMax H3 DiT (full, non-pruned BF16)
├── text_encoders/      # Qwen3-VL-32B NVFP4 AWQ text encoder
├── vae/                # H3 video VAE + H3 audio VAE
└── loras/              # RAVEN streaming LoRA (mandatory) + optional official LoRAs
```

### Downloads

**RAVEN streaming LoRA (mandatory, public)** -> `models/loras/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors`

- <https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA/resolve/main/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors>
- SHA256 `99de2e6b1ff69c49c3ca4b1126e5679409037fd2ccd0442b8323dc310d328f30`
- 266 LoRA modules, rank 128, alpha 128, ~5 GB of FP32 A/B tensors.

**H3 DiT (official Comfy-Org repackage)** -> `models/diffusion_models/minimax_h3_fl2va_bf16.safetensors`

- <https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors>
- SHA256 `907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182`
- This is the **full, non-pruned BF16** checkpoint. The pruned / adaln-curve
  variant is rejected by the loader: it has no `time_embedder` for the RAVEN
  adapter's 266-module mapping to attach to.

**Text encoder** -> `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

- <https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors>
- SHA256 `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`

**Video VAE** -> `models/vae/minimax_h3_video_vae_fp16.safetensors`

- <https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors>
- SHA256 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`

**Audio VAE** -> `models/vae/minimax_h3_audio_vae_fp32.safetensors`

- <https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors>
- SHA256 `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`

---

## The workflow

This package adds exactly **two** nodes. Everything else is stock ComfyUI.

```
[RAVEN Model Loader]
   (full non-pruned BF16 DiT + mandatory RAVEN LoRA, strength fixed at 1.0)
        │
        └─> MODEL ──> (optional) LoraLoaderModelOnly ... ─────────────────┐
                                                                          │
CLIPLoader (type: minimax) ──> CLIP ──┐                                   │
                                      ├─> MiniMaxH3ImageToVideo           │
VAELoader (video VAE) ────────────────┘   (T2VA: no first/last frame)     │
   │                                        ├─> CONDITIONING ────────────>┤
   │                                        └─> LATENT (empty AV) ───────>┤
   ├─────────────────────────────────────────────  video_vae ────────────>┤
VAELoader (audio VAE) ──────────────────────────── audio_vae ────────────>┤
                                                                          │
                                        [RAVEN Streaming Sampler] <───────┘
                                                 ├─> LATENT
                                                 ├─> IMAGE  -> SaveAnimatedWEBP / CreateVideo
                                                 └─> AUDIO  -> SaveAudio / CreateVideo
```

Ready-made graphs live in [`workflows/`](workflows/):

- `workflows/raven_t2va_streaming.json` — a **UI workflow**: drag it onto the
  ComfyUI canvas (or *Workflow -> Open*) and it loads as a graph you can edit.
- `workflows/raven_t2va_streaming_api.json` — the same graph in **API prompt**
  format, for `POST /prompt`. This one does **not** open in the UI.

Both reference the public filenames listed above; if your files are named
differently, re-pick them in the combo widgets.

### 1. `RAVEN Model Loader`

Loads the full non-pruned BF16 H3 DiT together with the **mandatory** RAVEN PEFT
LoRA and returns a **standard `MODEL`**.

- Inputs: `unet_name` (from `diffusion_models`), `lora_name` (from `loras`),
  `weight_dtype`.
- The RAVEN residual strength is **fixed at 1.0** and is not a node input. The
  adapter is what makes the model a RAVEN model; "off" is not a state this node
  can be put in. (A programmatic caller can still pass `strength=` to
  `loader.load_raven_diffusion_model`.)
- The LoRA is applied as an **FP32 activation residual** — computed on
  activations and added to the base output, never fused as `B @ A` into the BF16
  weights. That is what survives partial CPU offload, where base weights and
  adapter can sit on different devices.
- The returned `MODEL` is a **stock static `ModelPatcher`**
  (`disable_dynamic=True`), so stock `LoraLoaderModelOnly` chains after it and
  Comfy's own offload accounting applies unchanged.
- There is deliberately **no FP8/INT8 `weight_dtype`**: `comfy.ops` fuses
  quantised linears and would silently skip the residual.

### 2. `RAVEN Streaming Sampler`

Runs the chunk-major RAVEN rollout and produces the final media.

- Inputs: `MODEL`, official `CONDITIONING`, the official H3 AV `LATENT`, the
  **video VAE** and the **audio VAE**, plus the parameters below.
- **It takes no prompt and no CLIP and encodes no text.** Conditioning and the
  empty AV latent come from the official `MiniMaxH3ImageToVideo` node used in
  T2VA form (no `first_frame` / `last_frame` connected).
- **It has no width / height / frames inputs.** The canvas and the frame count
  are read off the incoming `LATENT`, so there is exactly one place they are
  set.
- A non-empty latent is refused: every chunk starts from its own fresh noise.
- Both VAE sockets are feature-probed by name (`MiniMaxH3VideoVAE` /
  `MiniMaxH3AudioVAE`, channel counts, sample rate, the decoder entry points the
  streaming lane drives). Swapping the two sockets is reported as exactly that.
- Both VAEs are driven **per chunk, through their inner modules**, and never
  through `VAE.decode` on the whole clip. The `IMAGE` and `AUDIO` outputs come
  out of those per-chunk decodes; see *Outputs*.

### Reused official nodes

| Official node | Role |
|---|---|
| `CLIPLoader` (`type: minimax`) | Loads the H3 (Qwen3-VL) text encoder. |
| `MiniMaxH3ImageToVideo` (T2VA: no frame inputs) | Prompt encoding -> `CONDITIONING` + empty AV `LATENT`. Its `vae` input takes the **video** VAE. |
| `VAELoader` x2 | Video VAE and audio VAE, one loader each. |
| `LoraLoaderModelOnly` (optional) | Third-party / additional official H3 LoRAs, chained **after** `RAVEN Model Loader` and before the sampler. Stack as many as you like; the mandatory RAVEN adapter is already inside the `MODEL` and is not affected by their strengths. |
| `SaveAnimatedWEBP`, `SaveAudio` (or `CreateVideo` -> `SaveVideo`) | Saving the `IMAGE` and `AUDIO` outputs, see below. |

---

## Parameters and limits

| Parameter | Default | Constraint |
|---|---|---|
| `seed` | `0` | Seeds a private generator; global RNG is never touched. |
| `steps` | `4` | Consistency NFEs per chunk. RAVEN's published preview adapter is distilled for 4; more is not a free quality win. |
| `video_shift` | `12.0` | Shift of the video stream's trailing sigma grid. |
| `audio_shift` | `3.0` | Shift of the audio stream's own grid. The two streams run independent grids. |
| `sink` | `2` | Attention-sink cache chunks pinned from the start. Chunk 0 is the text prefill, so `2` means text + the first media chunk. |
| `window` | `2` | Most recent cache chunks kept besides the sinks. `0` keeps only the sinks. |
| `kv_cache_storage` | **`cpu_pinned`** | Where the retained chunk KV cache lives: `cpu_pinned` (default, page-locked host memory), `cpu`, or `gpu`. It changes *where bytes live*, not what is computed. At 1376x768 / 192 frames the default holds one layer's slot on the card — **0.561 GiB** — instead of the whole cache's ~21.1 GiB steady / ~28.0 GiB peak. |
| `width` / `height` | `1376` / `768` (set on the H3 latent node) | Multiples of **32**, and `width * height <= 1376 * 768`. |
| `frames` | `192` (set on the H3 latent node) | Must satisfy **`17k + 5` with `k >= 1`**: minimum `22`, hard maximum `362`. `k = 0` (5 frames / 2 latents) is **not supported** and fails loud. Above `192` is allowed but **experimental** and warns. |
| batch size | `1` | Fixed. |
| sampler / scheduler | — | **Not selectable.** The node runs the RAVEN chunk-major consistency loop; stock samplers and schedulers do not apply. |
| CFG / negative prompt | — | **Not exposed.** One conditioning branch only. |

Scope limits that are part of the contract:

- **T2VA only.** The official image-conditioned modes (`fl2va`, `ref2va`) are out
  of scope for 0.1.x.
- **Single GPU only.** No multi-GPU, no model parallel, no distributed sampling.
- **Full, non-pruned BF16 DiT only.**
- **No per-chunk reroll.** There is no way to reject chunk *i* and resample it:
  a chunk is committed into the KV cache the moment it finishes, and the next
  chunk is already conditioned on it. There is no widget, no button and no
  protocol message for it anywhere in the UI, and adding one would mean
  rewinding the cache, the media clock and fMP4 fragments that have already
  been appended to a `SourceBuffer` and can never be revised.
- **No real-time / latency SLA.** "Streaming" means incremental availability,
  and nothing more. The measured 192-frame run took **301.30 s of sampling** to
  produce **8 s of clip**; the first fragment reached the sink **51.46 s** in.
  No claim is made — here or anywhere else in this repository — that output
  keeps up with playback.

---

## Outputs

The node returns `LATENT`, `IMAGE`, `AUDIO`. **There is no whole-clip VAE decode
anywhere in this package** — not for the video, not for the audio. Both media
outputs are built incrementally, in the same per-chunk callback that feeds the
preview:

| Output | How it is produced |
|---|---|
| `LATENT` | The finished AV latent, the same nested (video, audio) structure as the input. |
| `IMAGE` | The **video collector**. Each chunk's latents are decoded in the official order (`to(device, vae_dtype)` -> denormalize -> `_adaptive_decode` -> `blend` -> `_finalize_pixels`) and the finalized frames are `memcpy`'d straight into a pre-allocated `[T, H, W, 3]` host buffer. That buffer *is* the `IMAGE`. |
| `AUDIO` | The **audio collector**. Each chunk's audio latents are decoded overlap-save and copied into a pre-allocated `[1, 2, N]` host waveform. At the end that raw waveform gets the official whole-clip normalisation — `std = std(x, dim=[1,2]) * 5`, `std[std < 1] = 1`, `x /= std`, the tail of `comfy_extras.nodes_audio.vae_decode_audio`, reproduced expression for expression. |

Why there is no full decode: both of the calls this replaced died on real
hardware. `video_vae.decode` OOMed at 39 frames on a 141 GiB card (130.22 GiB
allocated), and at 192 frames on a 24 GiB card the whole-clip
`vae_decode_audio` OOMed even with the DiT *and* the video VAE already evicted
— and its own OOM fallback made it worse, because the generic tiled path is
written for 4-D latents and the H3 audio latent is `[B, 32, 2, T]`. Both calls
only re-derived data these lanes already had, chunk by chunk.

**The two collectors are therefore not optional and not best-effort.** They run
on every chunk whether or not a browser is attached, and a failure in either one
fails the run — with no full decode to fall back on, dropping a chunk would mean
silently returning a partly-black `IMAGE` or a waveform with a hole in it.

**The preview is a third lane, on top of them.** It reads frames and PCM back
out of the two collectors, muxes fMP4 and sends it; it is lossy on purpose
(H.264 + AAC) and exists to show progress. It can fail at any point without
touching `LATENT` / `IMAGE` / `AUDIO`.

One consequence worth knowing: **the preview's audio is the raw PCM, before
normalisation.** The `/= max(1, std*5)` divisor is a whole-clip statistic that
simply does not exist yet while the clip is still being made, so a
sample-for-sample comparison of what was streamed against what was returned
differs by exactly that divisor. The video has no such split — the frames the
preview encodes are the same frames the `IMAGE` buffer holds.

### Saving

The workflows in `workflows/` end in the official `SaveAnimatedWEBP` (video,
24 fps) and `SaveAudio` (FLAC), because those two take `IMAGE` and `AUDIO`
directly and carry only plain widgets. Note that upstream marks `SaveAudio`
**deprecated** at the pinned baseline — it still loads and still writes FLAC;
`SaveAudioAdvanced` is its replacement if you want to pick the format.

For a single muxed file instead, replace both with the official `CreateVideo`
(`images` + `audio`, `fps = 24`) -> `SaveVideo` pair: `CreateVideo` is exactly
the node upstream provides for an `IMAGE` + `AUDIO` pair.

---

## Streaming preview

The sampler streams into its own node widget while it samples. Contract:

- **Video appears one chunk behind sampling.** The video VAE is causal in
  latents with a **2-latent lookahead**: 7 latents (5 main + 2 lookahead)
  finalize 17 frames, consecutive decodes overlap by 5 frames, and the clip ends
  with a **5-frame flush**. So there is a one-chunk startup delay — chunk 0
  produces no frames at all — then each new DiT chunk releases the previous
  17 frames.
- **Audio trails by a further 0.425 s.** The audio VAE is decoded overlap-save
  with a **17-latent margin** (17 x 800 samples at 32 kHz), the smallest margin
  at which the streamed waveform matches a full-sequence decode to within the
  decoder's own noise (`< 2.5e-6`, measured at M0).
- **The tail is bigger than the lookahead, and it is the block size that makes
  it so.** Audio is decoded in 28-latent blocks, and a block can only be emitted
  once `block + margin` further latents exist. The last `block - 1 + margin` =
  **44 latents (1.1 s)** of a clip therefore cannot be decoded until
  `finish()`, and because the lane never substitutes silence, the video frames
  covering that span wait with them. `finish()` flushes exactly that last
  boundary and nothing else — everything before it already went out during
  sampling. Measured at 1376x768 / 192 frames: **216 of 251 fragments (86 %)**
  and **168 of 192 muxed frames (88 %)** were already sent when `finish()` was
  called, the first fragment landing after chunk 1 at **51.46 s**. Shorter
  clips pay proportionally more for the same fixed tail: at 22 frames nothing
  streams before `finish()` at all.
- **Transport is every-frame fMP4.** H.264 (High@4.0) + AAC-LC muxed by PyAV
  with `frag_every_frame+empty_moov+default_base_moof` and
  `min_frag_duration=1`, delivered as `video/mp4; codecs="avc1.640028,mp4a.40.2"`
  to a Media Source Extensions `SourceBuffer`. One fragment per frame means the
  first picture reaches the browser one frame after it is muxed instead of one
  segment. Those fragments are decodable **in order only**, which is exactly what
  a sequentially appended `SourceBuffer` needs; one forced IDR per 17-frame chunk
  keeps a late joiner resynchronisable. Silence is inserted automatically when a
  segment carries no audio, so the two timelines stay aligned.
- **Playback starts muted and autoplays.** Browsers refuse unmuted autoplay; the
  widget starts muted with an unmute button, and if autoplay is refused even
  while muted it offers an explicit play control.
- **Cost: base64 inside JSON, +33 %.** The pinned frontend drops unknown *binary*
  websocket frames, so media travels as base64 in JSON messages
  (`raven.preview`). `n` bytes cost `ceil(n/3)*4` characters plus a ~200–300 byte
  envelope. This is a deliberate, documented trade — see
  [`web/PROTOCOL.md`](web/PROTOCOL.md) §2.
- **A preview failure never affects the outputs.** No preview session, no PyAV,
  no encoder, a mid-run send failure, a closed browser tab: each is logged and
  sampling continues. The returned `LATENT` / `IMAGE` / `AUDIO` are byte-for-byte
  what they would have been with no browser attached — they come from the two
  collectors, which are a layer *below* the preview and always run.
- **What you hear is not quite what you get.** The preview carries the
  collector's raw chunk-wise PCM; the returned `AUDIO` is that same buffer after
  the official whole-clip `/= max(1, std*5)`. See *Outputs* above.
- **Cancel cleans up.** Interrupting from ComfyUI ends the session with
  `reason: cancelled`; the client drops its queue, aborts the source buffer,
  revokes the object URL and removes its listeners, and the backend releases the
  session's resources through its own context manager on every exit path,
  including exceptions.

### Attention backend

The causal DiT resolves its attention backend once, in RAVEN's own priority
order: **FlashAttention 3** (`flash_attn_interface`) -> **FlashAttention 2**
(`flash_attn`) -> **PyTorch SDPA**, the packed transcription of RAVEN's
`_sdpa_varlen`. All three are optional; SDPA is always available, so a box
without FlashAttention runs without any configuration. `FLASH_ATTN_3_AVAILABLE=0`
/ `FLASH_ATTN_2_AVAILABLE=0` force a backend down the chain.

---

## Hardware and memory

- **1x NVIDIA H200, single GPU** — that is the box every measurement in
  `docs/validation.md` was taken on. The 192-frame reference run did **not** run
  on a physically small card: an external process
  (`.cache/hold_vram.py`) allocated VRAM on the H200 until only **24.100 GiB**
  was free, so upstream saw, and had to plan against, a real 24 GiB of headroom.
  That is the envelope this package sizes itself against; it is a simulated
  ceiling on a big card, not a 24 GiB GPU.
- **System RAM: budget 160 GiB available, 192 GB physical.** This is the number
  most likely to bite you, and the older ">= 128 GB" figure is **not** good
  enough — measured peak host RSS was **129.854 GiB at 192 frames** and
  **135.025 GiB at 362**, both of which clear 128 GiB outright. On a 128 GiB box
  expect swapping or an OOM kill.

  Where it goes: ~66 GB of BF16 base weights plus ~5 GB of FP32 adapter (which
  live on the host whenever Comfy offloads them), the load path, the KV cache
  — **22.62 GB steady / 30.10 GB peak** at 192 frames, on the host by default
  — and the output buffers, 2.43 GB of `IMAGE` at 192 frames and 4.59 GB at 362.
  Those last two are the trade this architecture makes deliberately: the outputs
  are built in host RAM instead of being decoded from VRAM at the end. Comfy's
  reserve accounts for none of it.
- **Stock Comfy partial CPU offload is the memory strategy**, and the only one
  v0.1 makes a claim about. The loader returns a static `ModelPatcher`; nothing
  in this package calls `.to()`, `patch_model()`, `partially_load()` or
  `cleanup_models()`. Comfy owns residency. Measured on `vr-1` in all three
  residency states — fully resident, fully offloaded, and split across
  CPU/GPU — with the cross-device LoRA residual working in the split state
  (`docs/validation.md`, M1).
- **The DiT and the two VAEs are never co-resident.** Every chunk alternates two
  phases through ordinary `comfy.model_management.load_models_gpu` calls: a
  **DiT phase** for the forwards, then a **VAE phase** for the decode inside
  `on_chunk`. Each phase asks for its own `memory_required` and upstream decides
  what to evict; this package never moves a weight itself. The last chunk stops
  in the VAE phase on purpose, because `finish()` still has the tail flush to
  decode. Measured at 192 frames: **12 chunks, 59 forward passes, 12 DiT-phase
  loads and 12 VAE-phase loads.**
- **The KV cache lives on the host by default.** `kv_cache_storage` defaults to
  `cpu_pinned`: the committed chunks sit in page-locked host memory and one
  layer's retained rows are copied back per block. That is **0.561 GiB** of VRAM
  at 1376x768 / 192 frames instead of the ~21.1 GiB the whole cache would hold
  steady (~28.0 GiB at its peak) on the card. `gpu` is available for cards with
  room to spare; it changes where bytes live, not what is computed.
- **The reserve is the rollout workspace, not the weights.** The node prices the
  KV slot, the rollout buffers, the widest forward and the decode workspace
  before it loads anything, and hands that number to Comfy as `memory_required`.
  For the published request (1376x768, 192 frames, sink 2, window 2,
  `cpu_pinned`) it comes to **2.296 GiB**:

  ```
  2.296 GiB = KV 0.000 + buffers 0.074 + max(forward 1.529, decode 0.635) + safety 0.692
  ```

  KV is `0.000` because with `cpu_pinned` the cache is not on the card at all.
  The number is an estimate computed from this repository's own structures and
  logged at `INFO` on every run — and it has now been run against real weights:
  the same request peaked at **21.342 GiB allocated / 21.721 GiB reserved**,
  inside a 24 GiB envelope. Term-by-term detail in `docs/validation.md`.
- **There is deliberately no VRAM-cap and no weight-residency widget.** A "max
  resident GB" input would be a second, worse estimate of free memory sitting in
  front of upstream's real one; every time the two disagreed, the user's number
  would be the one turning a run that fitted into an OOM, or a run that fitted
  into a refusal. The 24 GiB reference figure is a constant used for *reporting*
  ("would this have fitted the smallest card we target?"), and the 24 GiB path
  is tested by making the device actually that small from outside the process,
  never by the node pretending a bigger card is smaller.
- **The NVFP4 text encoder** runs its encode on the GPU and is then offloaded to
  CPU by ComfyUI, so it holds no VRAM during sampling. Measured, not assumed:
  after the encode, `loaded_size = 0` and all **51 509 571 044 B** of its
  parameters are back on `cpu` — but they are back on *the host*, which is part
  of why the RAM figure above is what it is (that run peaked at 131.632 GiB RSS
  on a 39-frame clip).
- **DynamicVRAM / aimdo is not supported in v0.1.** The loader forces the static
  patcher and the sampler rejects a `ModelPatcherDynamic` `MODEL` outright. It
  was never exercised: on the cu128 measurement box the aimdo path is
  unavailable, and a cu130 build has no DynamicVRAM support either.

---

## Documentation

- [`docs/validation.md`](docs/validation.md) — what was measured, with the
  numbers and the artifacts. **Read this before believing anything else.**
- [`docs/requirements.md`](docs/requirements.md) — the confirmed requirements
  contract.
- [`docs/architecture.md`](docs/architecture.md) — module layout, the
  chunk-major loop, the KV cache, the two collectors, the preview lane, the
  memory budget.
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — pinned baseline, the feature probe,
  and the known risks.
- [`web/PROTOCOL.md`](web/PROTOCOL.md) — the preview wire protocol (v1).
- [`NOTICE`](NOTICE) — third-party attribution.

All of these links are **repository-relative**, because there is no published
repository for an absolute link to point at. Note that `docs/` and
`COMPATIBILITY.md` are excluded from the Registry archive by `.comfyignore`, so
if this package is ever published these four links stop resolving from the
packaged README and have to be re-pointed at the repository — that is a release
step, tracked as M5. `workflows/`, `web/`, `LICENSE` and `NOTICE` ship with the
package.
