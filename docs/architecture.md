# Architecture

How the package is put together, as built. Where a number here came from a
measurement, the measurement is in [`validation.md`](validation.md); where it
came from arithmetic over our own structures, this document says so.

Baseline read for every upstream reference below: ComfyUI `0.33.0`, commit
`c67885b14556cf3e4e061862925282d403d09862` (see `../COMPATIBILITY.md`).

---

## 1. Shape of the problem

MiniMax H3 generates video and audio jointly. Its latent is a *nested pair*:

- video `[B, 24, T_v, H/16, W/16]`, where `T_v = 5k + 2` tokens cover `17k + 5`
  pixel frames at 24 fps (`FRAME_PER_TOKEN` / `FRAME_RESCALE` upstream),
- audio `[B, 32, 2, T_a]`, at 40 audio-latent frames per second.

v0.1 requires **`k >= 1`** (`frames >= 22`). The degenerate `k = 0` case (5
frames / 2 video latents) is rejected loudly: it leaves no room for the
streaming loop's context or the decoder's lookahead.

Upstream ships single-shot nodes: build an empty AV latent, encode the prompt
with the Qwen3-VL-based H3 CLIP, sample the whole clip with a stock sampler,
then decode. RAVEN changes the sampling regime itself, so the stock sampling
path cannot be reused (§4).

---

## 2. Nodes (2)

### Node 1 — `RAVEN Model Loader` (`raven_streaming/loader.py`, `nodes.py`)

Full non-pruned BF16 H3 DiT + the mandatory RAVEN PEFT LoRA -> a **standard
`MODEL`**.

- Inputs: `unet_name` (`diffusion_models`), `lora_name` (`loras`),
  `weight_dtype`. No strength input: the residual is fixed at `1.0`, because the
  adapter is what the 4-NFE schedule was distilled with, not a quality knob.
- The loader walks upstream's own chain (old-quant conversion, prefix
  detection, `model_config_from_unet`, dtype / manual-cast / operations
  selection, `load_model_weights`, device selection, patcher construction,
  `cached_patcher_init`) and injects the chunk-causal DiT class as `unet_model`
  through an MRO shim that still runs the official `model_base.MiniMaxH3.__init__`.
- `force_static_patcher=True` / `disable_dynamic=True`: the result is a stock
  static `ModelPatcher`, the same class upstream selects for
  `disable_dynamic=True`. It is part of the spec, not of the call, so a rebuild
  triggered by `deepclone_multigpu` stays static too.
- Because the output is a plain `MODEL` with untouched `diffusion_model.*.weight`
  keys, stock `LoraLoaderModelOnly` chains after it and Comfy's caching, offload
  and memory accounting all keep working.

### Node 2 — `RAVEN Streaming Sampler` (`raven_streaming/nodes.py`)

- Inputs: `MODEL`, `CONDITIONING` (official), `LATENT` (official H3 AV latent),
  video `VAE`, audio `VAE`, and `seed`, `steps`, `video_shift`, `audio_shift`,
  `sink`, `window`, `kv_cache_storage`. Hidden: `unique_id`.
- **No prompt, no CLIP, no text encoding. No width / height / frames** — the
  canvas and frame count are read off the `LATENT`, so they are set in exactly
  one place.
- **No VRAM cap and no weight-residency input.** Both would be a second, worse
  estimate of free memory placed in front of the one `comfy.model_management`
  actually measures. `GPU_HARD_CAP_BYTES` (24 GiB) is a *reporting* constant,
  not an enforced ceiling; the 24 GiB path is exercised by shrinking the device
  from outside the process, never by the node pretending otherwise.
- **No per-chunk reroll.** A chunk is committed into the KV cache as soon as it
  finishes and the next chunk is conditioned on it, so "resample chunk *i*"
  would mean rewinding the cache, the media clock and fMP4 fragments that have
  already been appended to a `SourceBuffer`. It is not supported, and there is
  no widget, button or protocol message for it.
- Outputs: `LATENT`, `IMAGE`, `AUDIO`.
- Order of operations inside `sample()`: resolve and feature-probe both VAEs ->
  parse conditioning and latent -> build the layout -> **price the rollout
  reserve** and the two-phase offload plan -> open the preview session -> build
  the pipeline (two collectors + optional preview) -> run the chunk-major
  rollout, whose `on_chunk` is the **phase-swap coordinator** (VAE phase ->
  decode + collect + mux + send -> DiT phase for the next forward) -> announce
  `finalizing` -> `finish()` flushes the last boundary -> `finalize_image()` and
  `finalize_audio()` hand over the buffers the collectors already filled.
  **There is no whole-clip decode step at the end** (§6).

Everything else in the workflow is stock ComfyUI: `CLIPLoader`,
`MiniMaxH3ImageToVideo` in T2VA form, and two `VAELoader` instances.

---

## 3. Dataflow

```
[Node 1] RAVEN Model Loader
   (full non-pruned BF16 DiT + mandatory RAVEN PEFT LoRA, strength 1.0)
        │
        └─> MODEL ──> LoraLoaderModelOnly (optional, extra LoRAs) ────────┐
                                                                          │
CLIPLoader (minimax) ──> CLIP ──┐                                         │
prompt (STRING) ────────────────┴─> MiniMaxH3ImageToVideo (T2VA)          │
VAELoader (video VAE) ─────────────> vae                                  │
                             ├─> CONDITIONING ───────────────────────────>┤
                             └─> LATENT (empty AV, nested) ──────────────>┤
VAELoader (video VAE) ────────────────────────────────────────────────────┤
VAELoader (audio VAE) ────────────────────────────────────────────────────┤
                                                                          │
                                   [Node 2] RAVEN Streaming Sampler <─────┘
                                                │
                                                ├─> LATENT
                                                ├─> IMAGE
                                                └─> AUDIO
```

Inside Node 2, per chunk:

```
for chunk i in 0..N-1:                       # chunk-major, not sigma-major
    ├── [DiT phase]  load_models_gpu([DiT], memory_required=<DiT workspace>)
    ├── fresh noise for chunk i (private generator, seeded; global RNG untouched)
    ├── `steps` consistency NFEs on chunk i against the retained context:
    │       `sink` pinned cache chunks + `window` most recent ones
    ├── independent trailing sigma grids per stream (video_shift / audio_shift)
    ├── clean forward of chunk i -> committed into the KV cache as context
    └── on_chunk (one callback, three lanes, in this order):
            ├── [VAE phase] load_models_gpu([audio VAE, video VAE],
            │                              memory_required=<decode workspace>)
            ├── decode this chunk's video latents (official operator order)
            │     └── memcpy the finalized frames into the IMAGE buffer   [always]
            ├── decode this chunk's audio latents (overlap-save)
            │     └── memcpy the samples into the raw waveform            [always]
            └── read both back out, mux fMP4, send                      [best effort]

finish():   flush the last boundary only — the 5-frame video tail and the edge
            audio blocks. Everything earlier already went out during sampling.
finalize_image() / finalize_audio(): hand over the two buffers. No decode here.
```

The DiT and the two VAEs are **never co-resident**: `PhaseSwapCoordinator`
alternates them with ordinary `load_models_gpu` calls and lets upstream decide
what to evict. The last chunk deliberately stops in the VAE phase, because
`finish()` still decodes through both VAEs.

Emission timing (a property of the video VAE's geometry, §6, and the audio
block size):

- one-**chunk** startup delay before the first frames exist — chunk 0 emits
  nothing,
- steady state: each newly sampled DiT chunk releases the **previous 17 frames**,
- end of clip: a final **flush of 5 video frames**, plus the audio edge blocks —
  the last `28 - 1 + 17 = 44` audio latents (**1.1 s**), and the video frames
  covering them, since the lane never substitutes silence,
- chunks are emitted in order and are **never revised**.

The two collectors are one whole chunk *ahead* of the preview: none of that
gating touches them, so the `IMAGE` and `AUDIO` buffers are complete the moment
`finish()` returns.

---

## 4. Why the stock sampler cannot be reused

ComfyUI's samplers are **sigma-major**: one denoising step across the whole
latent, then the next sigma. RAVEN streaming is **chunk-major**: a chunk is
carried to completion with fresh noise and a small NFE budget before the next
chunk starts, with only `sink` + `window` chunks kept as context. The two loop
orders are not interchangeable, so:

- `KSampler` / `SamplerCustom` and the stock scheduler list do not apply, and no
  sampler or scheduler selector is exposed.
- `raven_streaming/consistency.py` runs the rollout: a port of RAVEN's
  `ConsistencySampler` + `TrailingSamplingTimesteps` driven by
  `CausalMiniMaxH3Base._rollout_latents`, onto ComfyUI's official H3 model.
- The incoming `MODEL` must be a core **static** `ModelPatcher`. A
  `ModelPatcherDynamic` is refused with an explicit message rather than run on an
  unverified path.
- The RAVEN LoRA is applied as an **FP32 activation residual**
  (`runtime_linear.py`):
  `out = (base_out + B(A(x.float())) * alpha / r * strength).to(base_out.dtype)`,
  which is exactly `peft.tuners.lora.layer.Linear.forward`. `B @ A` is never
  materialised, `.weight` / `.bias` are never touched, and the residual works
  cross-device — which is what makes it compatible with partial CPU offload.

---

## 5. Module layout (as built)

```
ComfyUI-MiniMax-H3-RAVEN-Streaming/
├── __init__.py               # V1 entry point: NODE_*_MAPPINGS + WEB_DIRECTORY
├── raven_streaming/
│   ├── __init__.py           # metadata + PEP 562 lazy mappings, import-light
│   ├── nodes.py              # the 2 node classes, VAE probing, memory budget,
│   │                         #   the DiT/VAE phase swap, output handover
│   ├── loader.py             # DiT + RAVEN PEFT selection, static ModelPatcher
│   ├── lora.py               # safetensors parse + strict PEFT->Comfy key map
│   ├── runtime_linear.py     # FP32 activation-residual attachment to comfy Linear
│   ├── causal_model.py       # chunk-causal MiniMaxH3Model + attention backends
│   ├── cache.py              # per-layer chunk KV cache, sink + sliding window
│   ├── consistency.py        # chunk-major fresh-noise consistency rollout
│   ├── layout.py             # T2VA packed geometry, 17k+5 grid, chunk table
│   ├── contracts.py          # socket-boundary parsing (CONDITIONING/LATENT/MODEL)
│   ├── compat.py             # upstream feature detection, no version branching
│   ├── streaming_pipeline.py # the 2 collectors (= IMAGE/AUDIO) + preview lane
│   ├── preview.py            # facade the node imports
│   ├── preview_session.py    # sessions, seq ordering, replay, TTL (stdlib only)
│   ├── preview_server.py     # PromptServer.send_sync + optional resume route
│   └── media/
│       ├── clock.py          # single integer tick grid, lcm(24, 32000) = 96000
│       ├── video_stream.py   # incremental video VAE coordinator (5+2 / 17+5)
│       ├── audio_stream.py   # overlap-save audio decoding
│       ├── codecs.py         # H.264 / AAC encoder preference chains
│       ├── mp4_writer.py     # fragmented MP4 muxer over a write-only sink
│       ├── mp4_boxes.py      # incremental ISO-BMFF box scanner / segmenter
│       └── fakes.py          # numpy stand-ins for both VAEs (no GPU, no torch)
├── web/                      # the in-node preview extension (see PROTOCOL.md)
│   ├── raven_streaming_preview.js
│   ├── preview.css
│   └── lib/{controller,mse,sequencer,protocol,states,identity,ui}.js
├── workflows/                # importable UI workflow + API prompt
├── tools/                    # probes and harnesses (not shipped)
├── tests/                    # pure unit/contract tests (not shipped)
└── docs/
```

`raven_streaming/__init__.py` imports nothing but the standard library;
`NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` are served by a module-level
`__getattr__`, so metadata reads and the pure tests run in a bare interpreter
while ComfyUI's `hasattr` check still triggers the real import. The repository
root `__init__.py` re-exports those plus `WEB_DIRECTORY = "./web"` and installs
the preview route, guarded so a preview failure can never hide the nodes.

---

## 6. The streaming lane

### Three lanes off one callback

`StreamingPipeline` runs **three** lanes off `on_chunk`, and only one of them is
optional:

| Lane | Optional? | What it is |
|---|---|---|
| video collector | **no** | decodes each chunk's video latents in the official order and `memcpy`s the finalized frames into a pre-allocated `[T, H, W, 3]` host buffer. That buffer **is** the node's `IMAGE`. |
| audio collector | **no** | decodes each chunk's audio latents overlap-save and copies the samples into a pre-allocated `[1, 2, N]` host waveform. That buffer **is** the node's `AUDIO`, after one final normalisation. |
| preview | **yes** | reads frames and PCM back out of the two collectors, muxes fMP4, sends it. Every failure here — no sink, no PyAV, a dead socket, an oversized fragment — disables the preview and nothing else. |

A failure in either collector **propagates**: it is a failure of the run. That
is a deliberate consequence of removing the full decode — with nothing to fall
back on, a dropped chunk would mean silently returning a partly-black `IMAGE` or
a waveform with a hole in it.

### Why there is no whole-clip decode

The node used to decode the whole clip twice over: once here for the preview,
and once again through `video_vae.decode` / `vae_decode_audio` for the outputs.
Both of those died on real hardware:

- 39 frames, 141 GiB card: `video_vae.decode` OOMed at 130.22 GiB allocated /
  139.12 GiB reserved.
- 192 frames, 24 GiB card: with the DiT *and* the video VAE already unloaded and
  only the audio VAE resident, the whole-clip `vae_decode_audio` still OOMed —
  and its OOM fallback made it worse, because the generic tiled path is written
  for 4-D latents and the H3 audio latent is `[B, 32, 2, T]`, so the retry died
  in an `IndexError` instead of a memory error.

Both calls only re-derived data these lanes already produced chunk by chunk. So
they are gone. The VRAM peak becomes one 7-latent video chunk plus one
overlap-save audio block (both priced by upstream's own `memory_used_decode`),
and the two host buffers that *are* the outputs: 2.43 GB of `IMAGE` at 192
frames (4.59 GB at 362) and 2.5 MB of waveform, in host RAM rather than VRAM.
The integration probe asserts this directly — `decode_images` x0,
`decode_audio` x0, `VAE.decode` calls `{'video': 0, 'audio': 0}`.

### Audio: the returned signal is not the streamed signal

`finalize_audio()` reproduces the tail of
`comfy_extras.nodes_audio.vae_decode_audio` expression for expression on the
collected raw waveform:

```python
std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
std[std < 1.0] = 1.0
audio /= std
```

That divisor is a **whole-clip statistic**, which does not exist while the clip
is still being made. So the preview necessarily carries the raw, un-normalised
PCM, and a sample-for-sample comparison of what was streamed against what was
returned differs by exactly that divisor. The divide is in place and the result
is cached, so repeat calls hand back the same dict rather than normalising
twice. Video has no equivalent split.

### Video — causal in latents with a 2-latent lookahead

The geometry is read off the upstream VAE (`clip_length=17`, `vae_ratio_t=4`,
`token_drop=3`), never assumed:

```
tokens_chunk_size = ceil(17 / 4) = 5      frame_pre_padding = (-17) % 4 = 3
token_overlap     = (-3) % 5    = 2       frame_overlap     = max(2*4 - 3, 0) = 5
chunk_dec         = 5 * 4       = 20
```

Chunk `i` reads latents `[5i, 5i+7)`, the decoder emits 28 frames, split at
`chunk_dec`: 17 finalized frames (cross-faded over 5 with the previous chunk's
tail) and 5 held as `dec_overlap` for the next chunk, flushed after the last.
`IncrementalVideoDecoder` runs that machine without ever materialising the full
output: its entire state is the pending latents (trimmed to the <= 7 still
needed), 5 decoded frames, and integer counters.
`reference_decode_temporal` is a port of upstream `decode_temporal`, used to
prove the streaming machine is bit-identical.

### Audio — overlap-save, 17-latent margin

The BigVGAN decoder is not causal, so a slice decode is not a slice of the full
decode. The fix is classic overlap-save: decode `[a-m, b+m)` and keep `[a, b)`.
`m = 17` latents with 28-latent blocks was **measured** at M0 as the smallest
margin whose result matches a full decode to within the decoder's own noise
(`< 2.5e-6`). 17 x 800 = 13 600 samples = **0.425 s** of lookahead, which is why
preview audio trails preview video by roughly one chunk. Block size 28 is a
steady-state chunk's audio latent count (the 29 / 28 / 28 cadence of the shared
85/3 clock), so a block boundary lands on a chunk boundary.

The two numbers do different jobs, and confusing them understates the tail.
**Correctness** depends on the margin (17, the receptive field). **Latency**
depends on the block: a block is only emitted once `block + margin` further
latents exist, so the last `block - 1 + margin` = **44 latents (1.1 s)** of a
clip can only be decoded by the edge blocks at `finish()`, and the video frames
covering that span wait with them — the lane never substitutes silence, because
an fMP4 fragment is never revised and the gap would be permanent. What that
costs, by clip length: 22 frames streams 0 % before `finish()`, 39 frames 41 %,
90 frames 74 %, 192 frames 87.5 %, 362 frames 93 %. A smaller block would stream
sooner at the cost of decoding more overlap (28 decodes 62 latents per 28
emitted, 2.2x; 7 would decode 41 per 7, 5.9x), so changing it is a measurement,
not an edit.

### Clock

Everything is scheduled on one integer tick grid — `lcm(24, 32000) = 96000`
ticks/s, 4000 per frame, 3 per sample — so no float drift can accumulate over a
long stream.

### Mux and transport

`FragmentedMP4Muxer` writes into a `WriteOnlySink` (no `seek`/`tell`, so PyAV
treats it as non-seekable), with
`frag_every_frame+empty_moov+default_base_moof` and `min_frag_duration=1`.
`mp4_boxes.py` splits the resulting byte stream at box boundaries into an init
segment (`ftyp`+`moov`) and fragments (`moof`+`mdat`) as bytes arrive. Encoders
come from a probed preference chain (`h264_nvenc` -> `libx264` -> `libopenh264`;
AAC likewise), never hardcoded. Delivery is `PromptServer.send_sync` with
base64 bodies inside JSON — the pinned frontend drops unknown binary frames
(`web/PROTOCOL.md` §1.1).

Measured on the 192-frame run: **251 fragments / 6 531 496 B**, largest 137 679 B,
0 oversize, 0 send failures; **216 fragments and 168 of 192 muxed frames were
already out when `finish()` was called**, first fragment after chunk 1 at
51.46 s.

### Preview sessions

`preview_session.py` is stdlib-only: one `seq` counter covering `open`, `init`,
`segment`, `status` and `end`; one state machine; one terminal message; a
bounded replay buffer; TTL-based expiry; idempotent cleanup through
zero-argument finalizers. `preview_server.py` holds everything that touches
ComfyUI and imports it lazily inside the call. Four properties the split exists
to guarantee: a preview failure never touches sampling; a session never pins GPU
memory; ordering is one rule; cancellation is observed, never caused.

---

## 7. Memory and offload

- **DiT**: stock `ModelPatcher` full/partial CPU offload. Required baseline, and
  the only strategy v0.1 claims. Nothing in this package calls `.to()`,
  `patch_model()`, `partially_load()` or `cleanup_models()` — Comfy owns
  residency.
- **Two phases, never co-resident.** The DiT and the two VAEs alternate on the
  card, through ordinary `comfy.model_management.load_models_gpu` calls, once
  per chunk: a **DiT phase** for the forwards, a **VAE phase** for the decode
  inside `on_chunk`. Each phase passes its own `memory_required` and upstream
  decides what to evict — that *is* the swap. The last chunk stops in the VAE
  phase, because `finish()` still has the tail flush to decode through both
  VAEs; reloading the DiT for a forward that never happens would evict exactly
  the models about to be used. Inside `enter_vae_phase` the audio VAE is listed
  first and the video VAE last, because `load_models_gpu` reverses the list and
  sizes each model against the memory left at that point, so the last entry is
  served first and the video VAE is by far the larger. Measured at 192 frames:
  12 chunks, 59 forwards, 12 DiT-phase loads, 12 VAE-phase loads.
- **The KV cache defaults to host memory.** `kv_cache_storage` is
  `cpu_pinned` by default: committed chunks live in page-locked host memory and
  one layer's retained rows are copied back per block, so the card holds a
  single **0.561 GiB** slot instead of the whole cache (~21.1 GiB steady,
  ~28.0 GiB peak at 1376x768 / 192 frames). `cpu` is the same without
  page-locking; `gpu` keeps it resident. This moves bytes, it does not change
  what is computed.
- **The reserve is the rollout workspace.** `nodes.estimate_rollout_budget` is a
  pure function of the request, the retention policy and the DiT's measured
  shape:

  ```
  kv_bytes_per_row = 2 * layers * heads * head_dim * compute_dtype_size
  kv_cache         = kv_peak_rows * kv_bytes_per_row      # 0 when KV is on the host
  kv_slot          = one layer's retained rows            # what cpu_pinned costs instead
  kv_gather        = 2 * kv_gathered_rows * heads * head_dim * dtype
  activations      = widest_rows * dtype * (4*hidden + 3*inner + inner + 3*ffn)
  forward          = activations + kv_gather + lora_fp32_temporaries
  buffers          = 3 * (video_clip + audio_clip) * latent_dtype
                   + 6 * (video_chunk + audio_chunk) * latent_dtype
  subtotal         = kv_cache + buffers + max(forward, decode)
  total            = subtotal + 0.12 * subtotal + 512 MiB
  ```

  `max(forward, decode)` and not a sum: the VAE decode runs inside `on_chunk`,
  i.e. between DiT forwards, never during one. The KV peak is not re-derived —
  `ChunkKVCache.retained_index_set` is asked which chunks it would keep, so if
  eviction changes the estimate changes with it. The DiT's shape is measured off
  the live module tree, with the published full-size numbers as a fallback, and
  which fields were measured is reported rather than hidden.

  For the published request (1376x768, 192 frames, sink 2, window 2, text 128,
  BF16, `cpu_pinned`) this is **2.296 GiB** =
  `KV 0.000 + buffers 0.074 + max(forward 1.529, decode 0.635) + safety 0.692`.
  It is still an *estimate*, but it is no longer unchecked: the same request
  peaked at **21.342 GiB allocated / 21.721 GiB reserved** against a 24 GiB
  envelope. Term by term in `validation.md`.
- **Neither a cap nor a residency widget exists.** `GPU_HARD_CAP_BYTES` (24 GiB)
  and `PLANNING_BUDGET_BYTES` (22 GiB, i.e. the cap less 2 GiB of headroom for
  one allocation this package does not model) are *yardsticks* used by the
  residency record and `hard_cap_watch` to say whether a run would have fitted
  the smallest card this pack targets. They enforce nothing: the allocator and
  `comfy.model_management` already do that, with numbers this node cannot see.
- **The outputs are host buffers, and they are part of the RAM cost.** 2.43 GB
  of `IMAGE` at 192 frames (4.59 GB at 362) plus 2.5 MB of waveform live in host
  RAM for the length of the run, and Comfy's reserve does not cover host RAM.
- **FP32 LoRA residual**: ~5 GB of FP32 A/B tensors, counted by Comfy's memory
  accounting from the first `model_size()` call, applied cross-device so a split
  base/adapter residency is correct rather than merely tolerated.
- **Text encoder**: NVFP4 AWQ Qwen3-VL-32B, GPU encode then CPU offload by
  ComfyUI; holds no VRAM during sampling.
- **Handover after the run.** `prepare_final_decode` unloads the DiT once the
  outputs are in hand, so the next node in the graph is not left waiting behind
  60+ GiB of parked weights. Measured: 13.92 GiB freed in 10.33 s.
- **System RAM: 160 GiB available / 192 GB physical is the honest number.**
  Measured peak RSS was **129.854 GiB at 192 frames** and **135.025 GiB at 362**
  — both above a 128 GiB box, so 128 GiB is a swap-or-OOM configuration, not a
  baseline. The total is ~66 GB base + ~5 GB FP32 adapter on the host whenever
  Comfy offloads them, plus the host KV cache (22.62 GB steady / 30.10 GB peak
  at 192 frames under the `cpu_pinned` default), plus the output buffers, plus
  the load path. Moving the KV cache off the card is what makes the 24 GiB VRAM
  envelope work — it does not make the bytes disappear, it moves them into the
  number above.
- No multi-GPU. No real-time SLA.

---

## 8. Milestones

| ID | Milestone | State |
| --- | --- | --- |
| **M0** | Media + LoRA probes: fMP4/MSE muxing with PyAV, audio overlap-save margin, video VAE streaming geometry, RAVEN PEFT FP32 activation-residual probe. | **done**, `validation.md` §M0 |
| **M1** | Loader + offload: full non-pruned BF16 DiT under stock partial CPU offload on one H200, in all three residency states; node schema frozen. | **done**, `validation.md` §M1 |
| **M2** | Causal parity: chunk-major fresh-noise consistency sampling with `sink`/`window` reproduces the reference, all 50 blocks. | **done**, `validation.md` §M2 |
| **M3** | Sampler + AV outputs: per-chunk video decode (5+2, 5-frame overlap) and audio overlap-save feeding the two collectors, correct `LATENT` / `IMAGE` / `AUDIO`, no whole-clip decode. | **done**, `validation.md` §Integration (1376x768 / 192 frames on a 24 GiB envelope, 86/86) |
| **M4** | WebSocket + MSE delivery: fMP4 over the ComfyUI websocket, one-chunk startup delay, 17-frame cadence, tail flush observed end to end. | **done**, `validation.md` §Integration (251 fragments, 216 before `finish()`) |
| **M5** | Release: docs carrying measured behaviour, `.comfyignore` verified against a package dry run, manual Comfy Registry publish under `yanzuolu`. | **in progress.** Metadata validates (`comfy node validate`: all checks passed; upstream's own `extract_node_configuration` parses publisher `yanzuolu`, `>=0.30.0`, OS Independent, NVIDIA CUDA, `web = None`) — `validation.md` §Packaging. `comfy node pack` still **pending**: it needs a git history this tree does not have yet. Nothing is published and there is no public remote. |

Every milestone that makes a claim ("passes", "matches", "verified") lands with
the evidence that supports it; until then `validation.md` says `pending`.

---

## 9. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Chunk-major loop has no upstream equivalent | We own the whole sampling path; upstream changes to sampling internals can break it silently | The loop is one module; the model contract is asserted at the socket; feature probe at registration. |
| `comfy_api.latest` and sampling internals are moving surfaces | Node registration or sampling breaks on upgrade | V1 registration only, feature table in `COMPATIBILITY.md`, hard errors instead of silent fallbacks. |
| Attention backend differences (FA3 / FA2 / SDPA) | Numerically different rollouts on different boxes | One resolution point, cached and logged; parity measured on FA3, SDPA is the always-available transcription of RAVEN's own `_sdpa_varlen`. |
| FP32 activation residual vs. CPU offload | Precision loss or thrashing at 4 NFE | Never fuse `B @ A`; cross-device residual measured working in the split residency state (M1). |
| Memory reserve is an estimate | Under-reserving lets Comfy fill the card with weights and OOM mid-rollout | Deliberately conservative (12 % + 512 MiB floor); every term exposed in `detail`. Compared against a real 192-frame run on a 24 GiB envelope: peak 21.342 GiB allocated, inside budget. |
| `frames > 192` | Untested regime | Allowed to 362, warned as experimental; 192 default (and measured); `k = 0` rejected outright. 362 has **not** been run on this architecture — see `validation.md`. |
| Host RAM, not VRAM, is the binding constraint | The `IMAGE` buffer and the host KV cache are GB that Comfy's reserve does not account for, on top of offloaded weights | Sized and stated (`IMAGE` 2.43 GB at 192 / 4.59 GB at 362; KV 22.62 GB steady / 30.10 GB peak at 192). Measured peak RSS **129.854 GiB at 192, 135.025 GiB at 362**, so the guidance is 160 GiB available / 192 GB physical — 128 GiB is explicitly called out as swap-or-OOM. |
| No whole-clip decode to fall back on | A dropped chunk would silently produce a black band or an audio hole | The collectors are mandatory, not best-effort: a failure in either propagates and fails the run. A cancelled run returns *no* partial output at all. |
| Preview transport cost (base64, +33 %) | Socket pressure on long clips | Fragment-sized messages, size guidance in `web/PROTOCOL.md` §2; an out-of-band body is the documented v2 change if a measurement ever demands it. |
| Browser/MSE variation | Preview fails despite correct generation | Preview is strictly optional and isolated; the final outputs never depend on it. |
| `NestedTensor` conventions may change | Latent slicing/concat breaks | Nested handling is isolated in `contracts.py`; shapes asserted at chunk boundaries. |
| "Streaming" read as "real-time" | Wrong expectations | No SLA is stated anywhere; README and requirements say so explicitly. |
