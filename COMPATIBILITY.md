# Compatibility

What upstream this package was built against, how it decides at runtime whether
it can run, and what was actually observed on the measurement box.

Measured results are recorded once, in [`docs/validation.md`](docs/validation.md);
this file states the policy and points there rather than restating numbers.

## Pinned baseline — verified

Every design decision, and every measurement in `docs/validation.md`, was made
against this exact revision:

| Item | Value | Status |
| --- | --- | --- |
| ComfyUI commit | `c67885b14556cf3e4e061862925282d403d09862` | **verified** |
| ComfyUI version (`comfyui_version.py`) | `0.33.0` | **verified** |
| `comfyui-frontend-package` | `1.49.6` | **verified** |
| PyAV | `av>=16.0.0`; probes ran on `17.1.0` (`vr-1`) and `18.1.0` (local) | **verified** |
| Python | `>=3.10` (follows upstream `requires-python`) | **verified** |
| Torch | `2.11.0+cu128` on the measurement box | environment note |

A checkout of that revision is expected at `.cache/upstream/ComfyUI` for local
inspection and tests. That path is gitignored; it is a working cache, never a
vendored copy.

## Declared support range — a target, not a claim

`pyproject.toml` declares:

```toml
[tool.comfy]
requires-comfyui = ">=0.30.0"
```

`0.30` is a **target**. It has never been run. Nothing in this repository
asserts that `0.30` works, and no measurement in `docs/validation.md` was taken
on anything but `0.33.0`. The authoritative answer comes from the feature probe
below; if the probe fails on a version inside the declared range, the
declaration is corrected — the probe wins, not the string.

## Feature probe (the actual gate)

Rather than branching on version numbers, the package probes for the upstream
capabilities it binds to and fails fast with an actionable message when one is
missing (`raven_streaming/compat.py`).

| Upstream surface | Why we need it |
| --- | --- |
| `comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo` | Reused in T2VA form (no frame inputs) to produce our `CONDITIONING` + AV `LATENT` inputs. |
| `comfy.ldm.minimax.model.FRAME_PER_TOKEN`, `FRAME_RESCALE` | Mapping between video latent tokens and pixel frames; drives chunk boundaries. |
| `comfy.ldm.minimax.model.MiniMaxH3Model` | Base class of the chunk-causal DiT the loader injects as `unet_model`. |
| `comfy.nested_tensor.NestedTensor` | H3 latents are a nested (video, audio) tensor pair. |
| `comfy.model_patcher.ModelPatcher` | The `MODEL` must be a core **static** patcher; hooks, offload and Comfy's memory accounting depend on it. |
| `comfy.model_patcher.ModelPatcherDynamic` (**detected, then refused**) | DynamicVRAM / aimdo. v0.1 forces the static patcher and rejects a dynamic `MODEL` explicitly rather than running an unverified path. |
| Stock `ModelPatcher` full/partial CPU offload (**required**) | The guaranteed way to run the full non-pruned BF16 DiT on a single GPU, including cross-device LoRA residual application. |
| `comfy.model_management.load_models_gpu` / `throw_exception_if_processing_interrupted` | The **DiT phase** and the **VAE phase** each load through this call with their own `memory_required`, once per chunk; the DiT and the two VAEs are never co-resident and upstream decides what to evict. Cancellation observed, never caused. |
| `comfy.model_management.unload_model_and_clones` | Handover after the run: the DiT is released once the outputs are in hand, so the next node in the graph is not left behind 60+ GiB of parked weights. |
| `comfy.sd.VAE` with the H3 inner modules (`MiniMaxH3VideoVAE`, `MiniMaxH3AudioVAE`) | The two collectors drive `_adaptive_decode` / `blend` / `_finalize_pixels` on the **inner modules, per chunk**, and read `clip_length` / `vae_ratio_t` / `token_drop` off the model. `VAE.decode` — the whole-clip entry point — is deliberately **never called**; the integration probe asserts a call count of 0 on both sockets. |
| `comfy_extras.nodes_audio.vae_decode_audio` (**read, not called**) | The whole-clip helper OOMed at 192 frames on a 24 GiB card and its tiled fallback then died in an `IndexError` (the generic path assumes a 4-D latent; H3 audio is `[B, 32, 2, T]`). Its **normalisation tail** is instead reproduced expression for expression on the collected waveform: `std = std(x, dim=[1,2]) * 5; std[std < 1] = 1; x /= std`. If upstream changes that expression, this is the line that has to change with it. |
| NVFP4 text-encoder support (`CLIPLoader` type `minimax`) | Loading the NVFP4 AWQ Qwen3-VL-32B encoder; GPU encode then CPU offload. |
| Stock `LoraLoaderModelOnly` compatibility | Optional extra LoRAs chained after our loader's standard `MODEL`. |
| `folder_paths` | `diffusion_models` / `loras` / `text_encoders` / `vae` resolution. |
| `av >= 16` + an H.264 and an AAC encoder | fMP4 segment muxing for the preview lane (and auto-silence tracks). **Optional**: absence disables the preview, never the sampling. |
| `PromptServer.send_sync` + `PromptServer.routes` | Preview delivery and the optional resume route. **Optional**, same rule. |
| `comfy_execution.utils.get_executing_context` | `(node_id, prompt_id)` for the preview session. **Optional.** |

Rules:

1. A missing or renamed **required** symbol is a hard, explicit error at node
   registration or node execution time — never a silent fallback that changes
   results.
2. A missing **optional** symbol degrades the preview only, is logged with the
   reason, and leaves `LATENT` / `IMAGE` / `AUDIO` identical.
3. Version strings are used only for user-facing messages, never as the branch
   condition for behaviour.
4. When upstream changes one of these surfaces, this table and the pinned
   baseline are updated in the same change that adapts the code.

## Measured environment notes

Measurement box: `vr-1`, **1x NVIDIA H200**, **cu128**. Every artifact that
records a device name records `NVIDIA H200`; there is no H100 anywhere in this
record. Full numbers in `docs/validation.md`.

The 24 GiB figure that appears throughout is a **simulated envelope, not a
physical card**: an external process allocates VRAM on the H200 until only
24.100 GiB is free, so `comfy.model_management` plans against real scarcity.
Nothing in this package fakes a smaller device to itself.

- **Stock `ModelPatcher` full/partial CPU offload: passed**, in all three
  residency states (fully resident, fully offloaded to CPU, and split), with the
  full non-pruned BF16 DiT plus the ~5 GB FP32 adapter.
- **Cross-device LoRA residual application: passed** in the split state.
- **DynamicVRAM / aimdo: unavailable, and unsupported by v0.1.** On this cu128
  environment the aimdo path does not initialise — ComfyUI logs a warning and
  continues on the stock patcher. A **cu130** build has no DynamicVRAM support
  either, so there is currently no environment in which this path could be
  exercised. It was never requested by any probe
  (`dynamic.status = "not requested"`), and the package refuses it rather than
  guessing.
- **`h264_nvenc`: present but not usable** inside the probe container
  (`PermissionError` from `avcodec_open2`). The encoder preference chain falls
  through to `libx264`, which is what every fMP4 measurement used.
- **PyAV fragment cadence:** in `frag_keyframe` mode the muxer holds a fragment
  until the first frame of the *next* segment, which at 1376x768 with 17-frame
  segments left the last fragment unflushed until close (2 probe failures, by
  design of the muxer). The preview lane therefore uses
  `frag_every_frame` + `min_frag_duration=1`, whose measured delay is <= 1 frame.

These are environment observations, not performance claims.

## Known compatibility risks

- The V3 node API (`comfy_api.latest`) is a moving target. This package
  registers through **V1** mappings from the repository root `__init__.py`
  (plus `WEB_DIRECTORY`) precisely so the live schema does not depend on which
  branch upstream's loader takes; there is deliberately no `comfy_entrypoint`
  beside it.
- `NestedTensor` semantics for the (video, audio) pair are new and may change
  shape conventions.
- The H3 nodes' canvas / frame-grid helpers are private module-level functions
  upstream (`align_frame_count`, `video_latent_t`, `temporal_shape`,
  `adapt_canvas`). We reimplement the parts we need rather than importing
  private helpers, and cross-check them against upstream in tests
  (`tests/test_layout_official_parity.py`).
- The sampling path is **not** shared with upstream: the RAVEN chunk-major
  fresh-noise consistency loop is ours (`docs/architecture.md` §4), so upstream
  sampler changes will not help us and upstream model-sampling changes can break
  us silently. Assumptions are asserted, not inferred.
- The attention backend is resolved at runtime (FA3 -> FA2 -> SDPA). Parity was
  measured on FA3; SDPA is the always-available fallback and is what the
  reference dumps were produced with in the earlier, superseded runs.
- Frontend/WebSocket transport for incremental fMP4 delivery is tied to
  `comfyui-frontend-package`; `1.49.6` is the only frontend the protocol was
  read and tested against. Binary websocket frames are unusable there (see
  `web/PROTOCOL.md` §1.1), so a future frontend that gains a raw-frame hook
  would be an opportunity, not a break.
