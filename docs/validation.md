# Validation record

What has actually been run, with the numbers it produced. Nothing on this page
is a projection: a row is either a measurement with its artifact named, or it is
marked **pending** and carries no number at all.

Owner of this file's results: the **main agent**. Rows marked *(to be refreshed
by the main agent)* hold the last locally known value and must be replaced with
the final run before anything is published.

What this page describes is the **final architecture**: two mandatory
collectors that decode each chunk inside `on_chunk` and hand their buffers over
as `IMAGE` and `AUDIO`, an optional preview lane on top of them, and **no
whole-clip VAE decode anywhere**. Any earlier number that assumed a full decode
at the end, or a GPU-resident KV cache, has been removed rather than carried
forward — those were measurements of a product that no longer exists.

Environment for every measured row below, unless the row says otherwise:

| | |
|---|---|
| Box | `vr-1`, **1x NVIDIA H200**, single GPU, **cu128** build. Every artifact that records a device name records `NVIDIA H200` (M2 dumps and all four integration runs); the M1 loader probes do not record one, but ran on this same box. There is no H100 in this record. |
| ComfyUI | `0.33.0` @ `c67885b14556cf3e4e061862925282d403d09862` |
| Frontend package | `1.49.6` |
| PyAV | `17.1.0` on `vr-1` (FFmpeg 8 / libavformat 62); `18.1.0` locally |
| Torch | `2.11.0+cu128` (M2 and the integration runs) |
| Python | `3.10.20` (integration runs) |

---

## M0 — media, VAE and LoRA probes

| Probe | Result | Evidence |
|---|---|---|
| PyAV / fMP4, 848x480 | **passed**, 0 failures. mp4 muxer present; `libx264` usable with *behaviourally verified* forced IDR; AAC encoder usable; init segment is `ftyp`+`moov`; every fragment a complete `moof`+`mdat`; no segment starts on a non-keyframe; 3 of 4 fragments (12 974 of 17 284 bytes) delivered **before** close. | `.cache/probe_pyav_vr.json` |
| PyAV / fMP4, 1376x768, 17-frame segments | **2 failures**, both `fragments delivered BEFORE close (1 of 2)` — in `frag_keyframe` mode the muxer holds fragment *N* until the first frame of segment *N+1*, so a 2-segment clip only flushes its last fragment at close. This is the measurement that drove the preview lane to `frag_every_frame` + `min_frag_duration=1` (delay <= 1 frame). `h264_nvenc` was present but not usable in that container (`PermissionError` from `avcodec_open2`); `libx264` was. | `.cache/probe_pyav_1376.json` |
| Audio VAE overlap-save margin | **17 latents** at a 28-latent block: streamed decode matches the full-sequence decode to `max|diff| < 2.5e-6`, the decoder's own run-to-run noise. 17 x 800 = 13 600 samples = **0.425 s** of lookahead. | real `MiniMaxH3AudioVAE`, `tools/probe_audio_overlap.py`; constant and derivation pinned in `raven_streaming/streaming_pipeline.py` (`AUDIO_MARGIN_LATENTS`) |
| Video VAE streaming geometry | 5 main + 2 lookahead latents per decode, 5-frame overlap, `5k+2` latents -> `17k+5` frames, read off the VAE's own `clip_length` / `vae_ratio_t` / `token_drop` rather than assumed; the incremental coordinator is checked bit-identical against a port of upstream `decode_temporal`. | `tools/probe_video_vae_streaming.py` (real `MiniMaxH3VideoVAE`, `--tol 1e-6`, grid 7/22/57/107, run via `.cache/run_vr_vae_probes.sh`); `tests/test_video_vae_stream.py` for the parity port. Result JSON stays on the probe host — *(to be refreshed by the main agent)* |
| RAVEN LoRA topology | 266 modules, rank 128, alpha 128.0, strength 1.0; category split core 208 / adaln 51 / time 2 / boundary 5; official topology confirmed against a 1 543-entry key map; 535 base keys loaded, 0 left over. FP32 A/B tensors, 5 057 413 120 bytes, applied as an activation residual (never fused). | `.cache/probe_m1_loader_*.json` (`build.official_key_hits`), `tools/inspect_raven_lora.py` |

## M1 — loader and offload (full non-pruned BF16 DiT + RAVEN LoRA)

Common to all three runs: `ModelPatcher` (static, `force_static_patcher=True`,
`is_core_patcher=True`), `MiniMaxH3` / `MiniMaxH3Model`, `torch.bfloat16`,
no manual cast, 33 122 992 912 parameters, `model_size` 71 337 847 264 B
(**~66 GB base + ~5 GB FP32 adapter**), `vram_state=LOW_VRAM`, peak host RSS
~133.5 GB. That was the *loader's* peak; the full rollout is higher again — see
"Host RAM" under Integration for the measured 129.854 / 135.025 GiB peaks and
the RAM guidance they imply.

The VRAM figures in the table below are **held allocations on the H200**, the
same `hold_vram.py` technique the integration runs use: "24 GiB of VRAM held"
means an external allocation left roughly that much in use, not that the card
was 24 GiB.

| Residency mode | What happened | Evidence |
|---|---|---|
| **Full** (24 GiB of VRAM held by a live allocation) | Everything resident on `cuda:0`: 66 280 430 080 B base + 5 057 413 120 B LoRA, 534 + 532 tensors, `fully_loaded=true`. Build 76.8 s, load 37.0 s. Hooked forward ran on `cuda:0`. | `.cache/probe_m1_loader_vr.json` |
| **CPU / partial** (no reservation, forced low VRAM) | `model_lowvram=true`, `loaded_size=0`, all 66 + 5 GB on `cpu`, `fully_loaded=false`. Hooked forward ran on `cpu` and produced the expected `[8, 5376]` FP32 output. Build 11.7 s, load 6.0 s. | `.cache/probe_m1_loader_partial_vr.json` |
| **Mixed** (76 GiB of VRAM held, forcing a split) | 61 532 969 984 B base + 4 662 312 960 B LoRA on `cuda:0`, 4 747 460 096 B base + 395 100 160 B LoRA left on `cpu`; `model_lowvram=true`, `fully_loaded=false`. **Cross-device LoRA residual applied correctly in this split state.** Build 74.7 s, load 38.2 s. | `.cache/probe_m1_loader_mixed_vr.json` |

DynamicVRAM / aimdo was **not requested and not exercised** in any of the three
(`dynamic.status = "not requested"`). See "Optional paths" below.

## M2 — causal parity against the official H3 DiT

Reported result of the FA3 parity run (full, non-pruned, **all 50 blocks**,
BF16, `sink=2` / `window=2`, 39 frames at 512x288, text_len 128, 3 chunks,
300 attention calls):

| | |
|---|---|
| Gating checks | **16 / 16 passed** |
| Q / K / V dumps | **bitwise identical** to the reference |
| video `x0` | **bitwise identical** |
| audio `x0` | `max|diff| = 9.54e-7` |

Harness: `tools/raven_parity_harness.py` (reference side) and
`tools/probe_causal_parity.py --mode real` (this package's causal model), driven
by `.cache/run_vr_m2_fa3_parity.sh` with `FLASH_ATTN_3_AVAILABLE=1`. The final
result JSON stays on the probe host — *(to be refreshed by the main agent)*.

Superseded intermediate runs are kept in `.cache/` for the record and must not
be quoted as the M2 result: `m2_full_comfy.json` and `m2_full_comfy_exact.json`
(SDPA reference, FP32-island variants), `m2_full_comfy_fa3_layout.json`
(1 gating failure), `m2_operator_block*.json`, `m2_embedding_off.json`,
`m2_dump_analysis.log`.

## Local test suite

| | |
|---|---|
| Final measured | **2624 passed, 12 skipped** with `RAVEN_ROOT` set to the real RAVEN checkout; **2621 passed, 15 skipped** without it. **0 failures** either way. |
| Scope | `tests/` — unit, contract, protocol, workflow and probe tests. No GPU or weights are required for the bulk of them; upstream/RAVEN-dependent cases skip when their checkout is unavailable. |
| Commands | `env -u RAVEN_ROOT .cache/venv/bin/python -m pytest tests -q` and `RAVEN_ROOT=/Users/ol125/Documents/RAVEN .cache/venv/bin/python -m pytest tests -q` |
| Other final checks | Web protocol **32/32**; Web controller **29/29**; `compileall` clean; workflow structural checker passed; Ruff fatal rules `E9,F63,F7,F82` passed. |

The delta between the two columns is the cross-checks against a real RAVEN
checkout (`test_causal_embedding_probe`, `test_causal_operator_probe`,
`test_causal_parity_dump`): `RAVEN_ROOT` unlocks 3 of them, and without it they
skip.

The suite proves *internal* consistency (geometry, cache retention, budget
arithmetic, mp4 box structure, protocol/session behaviour, node schema). It says
nothing about generation quality, which is what the integration rows below are
for.

## Integration — end to end on real weights

The end-to-end lane has **run**. Seven final artifacts back the rows below, all
`ok: true` with **0 failures**:

| Artifact | Lane | Checks |
|---|---|---|
| `.cache/probe_raven_integration_24g_192f_vr.json` | synthetic, simulated 24 GiB envelope | 86/86 |
| `.cache/probe_raven_integration_362_final_vr.json` | synthetic, 362-frame maximum | 86/86 |
| `.cache/probe_raven_integration_text_final_vr.json` | **real text encoder** | 100/100 |
| `.cache/probe_raven_integration_thirdparty_final_vr.json` | official `LoraLoaderModelOnly` | 101/101 |
| `.cache/probe_raven_integration_kv_cpu_final_vr.json` | `cpu_pinned` KV, two runs | 172/172 |
| `.cache/probe_raven_integration_kv_gpu_final_vr.json` | GPU KV, two runs | 172/172 |
| `.cache/probe_raven_integration_cancel_chunk2_repeat3_final_vr.json` | delivered-chunk cancel + three runs | 301/301 |

Driver: `tools/probe_raven_integration.py`. Box: `NVIDIA H200`, cu128, Torch
`2.11.0+cu128`, Python `3.10.20`, ComfyUI `0.33.0` @ `c67885b`, PyAV `17.1.0`.

> **Which lane proves what.** The two large runs (192 and 362 frames) use
> **synthetic conditioning**: no text encoder is loaded, and the `CONDITIONING`
> is a deterministic `[1, 128, 5120]` tensor drawn from
> `manual_seed(seed ^ 0x52415645)` with constant token tags. They therefore
> verify the **DiT / VAE / media / node** lane at scale and make no claim about
> the text encoder. The text encoder has its **own** run — 39 frames, a real
> prompt, the official `CLIPLoader` and `MiniMaxH3ImageToVideo` — recorded
> immediately below.

**Official third-party LoRA chaining.** The public Apache-2.0
`drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` dynamic-rank LoRA (298 177 224 B)
was loaded by upstream's own `nodes.LoraLoaderModelOnly` at strength 0.05 after
the mandatory RAVEN loader. The gate observed 208 non-zero base-weight patches,
0 unmatched keys, 0 residual-parameter targets, the same 266-module RAVEN
attachment, and a complete 39-frame generation.

**CPU/GPU KV parity.** Independent two-run reports used `cpu_pinned` and `gpu`
KV storage. Direct tensor comparisons of the last warm runs matched bit-for-bit
for video latent, audio latent, IMAGE and AUDIO. Moving the canonical BF16 KV
bytes to host memory changes
residency and transfer cost, not the result.

### 192 frames — 1376x768, inside a 24 GiB envelope

The headline run, and the one the memory design is accepted on. The card is an
H200; the 24 GiB is a **simulated envelope**. An external process
(`.cache/hold_vram.py`) allocated VRAM until only **24.100 GiB was free**, so
`comfy.model_management` planned against real scarcity — the node was never told
to pretend, and the probe reports the device's true total (139.8 GiB) alongside
the free figure it actually had.

| | |
|---|---|
| Request | 1376x768, 192 frames, `steps=4`, `video_shift=12`, `audio_shift=3`, `sink=2`, `window=2`, text_len 128, `kv_cache_storage=cpu_pinned` |
| Rollout | **12 chunks, 59 forward passes**, 12 DiT-phase loads / 12 VAE-phase loads, last phase `vae` |
| Wall clock | `sample()` **301.30 s**; model load **50.67 s**; build 9.75 s; preview flush 0.56 s; `finalize_image()` ~2 µs and `finalize_audio()` 0.58 ms — both are handovers, not decodes |
| Peak VRAM | **21.342 GiB allocated / 21.721 GiB reserved** (22 915 754 496 / 23 322 427 392 B). `hard_cap_watch` at the DiT-phase load: `over_reference: false` |
| Host RSS | 124.383 GiB after, **129.854 GiB peak** — i.e. this run does **not** fit a 128 GiB machine (see "Host RAM" below) |
| Reserve asked for | **2.296 GiB** = `KV 0.000 + buffers 0.074 + max(forward 1.529, decode 0.635) + safety 0.692` — KV is 0 on the card because `cpu_pinned` keeps the cache on the host (one 0.561 GiB layer slot instead of 21.066 GiB steady / 28.033 GiB peak) |
| `IMAGE` | `[192, 768, 1376, 3]` fp32 on `cpu`, 2 434 793 472 B, **192/192 frames collected**, finite, in `[0, 1]` |
| `AUDIO` | `[1, 2, 256000]` @ 32 kHz, **256000/256000 samples collected** (320 audio latents x 800), finite, `std = 0.019231`, peak `0.147245` |
| `LATENT` | video `[1, 24, 57, 48, 86]`, audio `[1, 32, 2, 320]`, both finite |
| No whole-clip decode | `decode_images` **x0**, `decode_audio` **x0**, `VAE.decode` calls `{'video': 0, 'audio': 0}`, `finalize_image` x1, `finalize_audio` x1 |
| Preview | 257 messages, one session, `seq 0..256` contiguous, one `open` / one `init` / one `end` (`complete`), **251 fragments / 6 531 496 B**, largest 137 679 B, 0 oversize, 0 send failures |
| Streamed before `finish()` | **216 of 251 fragments (86 %)**, **168 of 192 muxed frames (88 %)**, 224 000 audio samples. First fragment after **chunk 1, at 51.46 s**. Chunk 0 emits nothing (the one-chunk startup delay) and chunk 11 emits nothing (the tail flush) |
| Muxed stream | decodes back as 1 video + 1 audio stream, h264 1376x768 + AAC 32 kHz, 192 frames |
| Handover | DiT unloaded after the outputs were in hand: **13.92 GiB freed in 10.33 s** |

### 362 frames — 1376x768, the documented maximum

The long-clip run, on the same final architecture (two collectors, no whole-clip
decode). **Read the memory row carefully: this one was *not* constrained.**

| | |
|---|---|
| Request | 1376x768, **362 frames** (`k = 21`), same sampler settings, `kv_cache_storage=cpu_pinned` |
| Rollout | **22 chunks, 109 forward passes**, 22 DiT-phase loads / 22 VAE-phase loads |
| Wall clock | `sample()` **250.965 s**; model load **18.06 s** |
| Peak VRAM | **73.315 GiB allocated / 75.008 GiB reserved.** No VRAM holder ran, so upstream kept the whole DiT resident and `hard_cap_watch` reports **`over_reference: true`** at the DiT-phase load. This run says the 362-frame path *works*; it does **not** say 362 frames fit a 24 GiB card, and no such claim is made |
| Host RSS | **127.547 GiB after, 135.025 GiB peak** — the largest host figure in this record. The 4.59 GB `IMAGE` buffer is part of why |
| Reserve asked for | **2.358 GiB** = `KV 0.000 + buffers 0.130 + max(forward 1.529, decode 0.635) + safety 0.699` |
| `IMAGE` | `[362, 768, 1376, 3]` fp32 on `cpu`, 4 590 600 192 B, **362/362 frames collected** |
| `AUDIO` | `[1, 2, 482400]` @ 32 kHz, **482400/482400 samples collected**, `std = 0.035250`, peak `0.198219` |
| No whole-clip decode | `decode_images` x0, `decode_audio` x0, `VAE.decode` `{'video': 0, 'audio': 0}` |
| Preview | **473 fragments / 11 960 218 B**, largest 165 858 B, 0 oversize, 0 send failures; muxed stream decodes as 362 frames + AAC 32 kHz |
| Streamed before `finish()` | **435 of 473 fragments (92 %)**, **336 of 362 muxed frames (93 %)**, 448 000 audio samples. First fragment after **chunk 1, at 36.17 s** |
| Handover | 71 530 812 928 B (66.62 GiB) freed in 36.00 s |

The two runs together are what the tail-flush arithmetic predicts: the flush is a
fixed 44 audio latents / 1.1 s plus the 5-frame video tail, so the longer the
clip the smaller its share — 86 % of fragments out before `finish()` at 192
frames, 92 % at 362.

### 39 frames — the real Qwen3-VL text-encoder lane

The run that closes the gap the two synthetic runs leave open: the prompt goes
through the **official** nodes, end to end, on the real NVFP4 encoder.
`ok: true`, **100 checks, 0 failures**. 512x288, 39 frames, prompt *"a cat
playing a trumpet on a rooftop at sunset, jazzy soundtrack"* (64 characters).

**Text lane** — official nodes, not a re-implementation:

| | |
|---|---|
| Loader | official `CLIPLoader`, `type='minimax'`, `device='default'` — both read from its own `INPUT_TYPES`, not hard-coded |
| Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, **load 4.27 s**, `model_size` 15 686 911 147 B |
| Encode | **5.58 s**, with `load_models_gpu` putting the encoder on `['cuda:0']` |
| **Residency during sampling** | `loaded_size = 0`, all **51 509 571 044 B** of parameters (1003 tensors) back on `cpu`. The encoder holds **no** VRAM while the rollout runs — which is the claim the README makes and this is the measurement behind it. Offload via `unload_model_and_clones`, 7.73 s |
| Conditioning | exactly 1 entry; context **`[1, 15, 5120]`** fp32 and finite. 5120 is the raw Qwen3-VL width the DiT knows (already-refined states would be 5376) |
| Token tags | `[15]`, all 15 tagged `text(1)`, 0 video / 0 audio — one tag per token |
| T2VA shape confirmed | extras are `['minimax_token_tags', 'pooled_output']` only: **no `minimax_keyframes`, no `minimax_refs`**, because no frames were passed |
| Latent | official `MiniMaxH3ImageToVideo` built `(39, 512, 288, latent_t 12, audio_t 65)` — exactly the requested grid — and it is **empty**, as the sampler requires |

What this row does **not** say: that the RAVEN LoRA is text-only. It records the
shape of *this* run — a T2VA graph, so no condition/reference extras were
produced. The sampler refuses `minimax_keyframes` / `minimax_refs` because its
causal packed layout does not implement condition rows, which is a property of
this implementation and not a measured LoRA capability. Note also that the
empty-latent requirement is unrelated: the official image- and
reference-conditioned nodes carry their conditioning in the **conditioning
extras** and still hand over an empty target latent.

**Rollout and outputs:**

| | |
|---|---|
| Rollout | 3 chunks, 14 forward passes; `sample()` **57.15 s**, model load 12.56 s |
| Peak VRAM | **72.263 GiB allocated / 72.654 GiB reserved.** No VRAM holder ran here, so this figure is what an unconstrained card does, not a 24 GiB result |
| Host RSS | 73.685 GiB before, 72.139 GiB after, **131.632 GiB peak** |
| `IMAGE` | `[39, 288, 512, 3]`, **39/39 frames** |
| `AUDIO` | `[1, 2, 52000]` @ 32 kHz, **52000/52000 samples** (65 audio latents x 800), `std = 0.157287`, peak `0.531706` |
| No whole-clip decode | `decode_images` x0, `decode_audio` x0, `VAE.decode` `{'video': 0, 'audio': 0}` |
| Preview | **52 fragments / 1 349 920 B**, 0 send failures, 0 errors |
| Streamed before `finish()` | **18 of 52 fragments (35 %)**, 16 of 39 muxed frames (41 %), 21 333 audio samples; first fragment after **chunk 1, at 18.79 s** |

The low pre-`finish()` percentage is the expected one for a short clip, not a
regression: the tail flush is a fixed 44 audio latents / 1.1 s, so at 39 frames
it swallows most of the clip (41 % streams early), at 192 frames 88 %, and at
362 frames 93 %.

### Repeat, determinism and delivered-chunk cancellation

`ok: true`, **301 checks, 0 failures**. 256x160, 39 frames, synthetic
conditioning, plan `cancel_chunk, sample, sample, sample` in one process.

**Cancellation** landed only after two chunks had reached the media pipeline:

- 9 DiT forwards and 2 delivered chunks ran before `SamplingCancelled`.
- The real incremental decoders had produced **17 frames / 22 400 samples**;
  the browser had already received **18 fMP4 fragments / 360 135 B**.
- Before cancellation the video decoder held `_pending` + `_dec_overlap`, the
  audio decoder held `_buffer`, `pending_frames = 1`, and
  `samples_available = 22400` — so this is a real post-decode cleanup test.
- `abort()` ran once on each decoder; afterwards both decoder buffer lists were
  empty, `pending_frames = 0`, `samples_available = 0`, and the IMAGE/AUDIO
  buffers reported 0 bytes. No partial output was returned.
- Staged KV was discarded once; preview ended `cancelled`; no session remained.

Three normal runs then succeeded in the **same process**. Their complete
pairwise matrices are all ones:

| Output | Result |
|---|---|
| video `LATENT` | **bitwise identical** across all three runs |
| audio `LATENT` | **bitwise identical** across all three runs |
| `IMAGE` | **bitwise identical** across all three warm runs |
| `AUDIO` | **bitwise identical** across all three runs |

The earlier cold-first IMAGE difference was CUDA backend initialization, not a
sampler or VAE-state leak: the delivered-chunk cancellation exercises and warms
the real VAE before the three gated runs, which then produce bitwise-identical IMAGE tensors.

**Memory plateau**, gated on the last two runs:

| | |
|---|---|
| CUDA allocated growth | **0 B** |
| CUDA reserved growth | **0 B** |
| Host RSS growth | **−35 581 952 B** (it went down) |

### Host RAM — the measured peaks, and why 128 GiB is not a baseline

Stated separately because it is the number most likely to bite a user, and
because the older ">= 128 GB system RAM" line in this repository was **wrong**:
both final runs exceed it.

| Run | RSS before | RSS after | **RSS peak** |
|---|---|---|---|
| 192 frames, 1376x768 | 72.955 GiB | 124.383 GiB | **129.854 GiB** |
| 362 frames, 1376x768 | 72.955 GiB | 127.547 GiB | **135.025 GiB** |
| 39 frames, 512x288, real text encoder | 73.685 GiB | 72.139 GiB | **131.632 GiB** |

Note the third row: a *short* clip still peaks above 128 GiB, because the
NVFP4 encoder's 51.5 GB of host-side parameters land on top of the DiT's. Frame
count is not the only driver.

What is in it, at 192 frames:

| Item | Host bytes |
|---|---|
| BF16 base weights + FP32 adapter (on the host whenever Comfy offloads them) | 71 337 847 264 B (66.44 GiB) |
| KV cache under the `cpu_pinned` default | **22 622 208 000 B steady (21.07 GiB) / 30 099 865 600 B peak (28.03 GiB)** |
| `IMAGE` output buffer | 2 434 793 472 B (2.27 GiB); 4 590 600 192 B (4.28 GiB) at 362 frames |
| `AUDIO` output buffer | 2 048 000 B |

Those overlap in time rather than summing cleanly — the point is the shape of
the cost, and the peaks in the table above are what was actually measured.

**Guidance: budget at least 160 GiB of *available* RAM; 192 GB physical is the
comfortable configuration.** A 128 GiB machine is a swap-or-OOM configuration
for both measured requests, not a supported baseline. Note the coupling: moving
the KV cache off the card is what makes the 24 GiB VRAM envelope work, and it
pays for that by putting 21-28 GiB on the host, on top of offloaded weights.
Choosing `kv_cache_storage=gpu` reverses the trade.

## Memory reserve — estimate, now checked against a run

For the published request — 1376x768, 192 frames, `sink=2`, `window=2`, text_len
128, BF16, 50 x 56 x 128, `kv_cache_storage=cpu_pinned` —
`nodes.estimate_rollout_budget` reserves:

```
2.296 GiB = KV 0.000 + buffers 0.074 + max(forward 1.529, decode 0.635) + safety 0.692
            [20996 peak KV rows, 50x56x128 @ 2B; KV slot 0.561 GiB on the card,
             21.066 GiB steady / 28.033 GiB peak held on the host instead]
```

The arithmetic is still arithmetic over this repository's own structures
(`tests/test_nodes_memory.py` reproduces it without a GPU). What has changed is
that it is no longer *unchecked*: the same request ran on a card held to
24.100 GiB free and peaked at **21.342 GiB allocated / 21.721 GiB reserved**,
with `over_reference: false`. The `detail` dict carries every term, the joint
offload plan for both phases, and the measured device facts, so the comparison
can be redone term by term from the artifact.

## Packaging — Registry dry run

The source repository is public, but no tag, GitHub Release or Comfy Registry
entry exists. The checks below validate the package without publishing it.

| Check | Result | How |
|---|---|---|
| `comfy node validate` | **"All validation checks passed successfully"** (configuration + security checks) | `comfy-cli 1.16.0` installed into an isolated, project-local `.cache/registry-venv`, run as `.cache/registry-venv/bin/comfy --json node validate` |
| Upstream's own parser | **passes** | the real `comfy_config.config_parser.extract_node_configuration('.')` from the pinned ComfyUI checkout — not a re-implementation |
| Parsed metadata | `publisher_id = yanzuolu`, `supported_comfyui_version = >=0.30.0`, `supported_os = ['OS Independent']`, `supported_accelerators = ['GPU :: NVIDIA CUDA']`, **`web = None`** | same call |
| `comfy node pack` | **passed**: 45 files | packed with the custom-node template JSON/JPG and separate API prompt, then moved to `.cache/registry-pack/comfyui-minimax-h3-raven-streaming-0.1.0.zip` |
| Archive contents | required runtime, `web/`, `example_workflows/`, `api_workflows/`, README/LICENSE/NOTICE/metadata present; **no** `tests/`, `tools/`, `docs/`, `.cache/`, model or IO trees | `unzip -Z1` allow/exclusion audit; full listing in `.cache/final_pack_listing.txt` |

`web = None` is the intended value, not an omission: `WEB_DIRECTORY = "./web"`
lives in the repository root `__init__.py`, and setting `[tool.comfy] web` as
well would make upstream register the folder twice under two keys and load every
file in it twice (`web/PROTOCOL.md` §6.1).

The venv and packed ZIP are deliberately inside gitignored `.cache/`, so neither
contaminates the runtime environment or git history. `pack` is a local dry run:
it creates no tag, release or Registry entry.

## Optional paths, explicitly unverified

- **DynamicVRAM / aimdo (`ModelPatcherDynamic`, `comfy-aimdo`).** Never
  exercised. On the cu128 measurement box the aimdo path is unavailable and
  ComfyUI logs a warning rather than enabling it; a cu130 build has no
  DynamicVRAM support to fall back on either. v0.1 does not merely treat it as
  optional — the loader forces the **static** patcher (`disable_dynamic=True`)
  and the sampler **rejects** a `ModelPatcherDynamic` `MODEL` with an explicit
  error. Supporting it is a future change with its own evidence, not a
  configuration switch.
- **`h264_nvenc`.** Present on `vr-1` but not usable from inside the probe
  container; the preview lane falls back through its encoder preference chain to
  `libx264`, which is what every measured fMP4 number above was produced with.
- **ComfyUI `0.30`.** Declared as a *target* minimum only. Never run. The
  feature probe in `COMPATIBILITY.md` is the authority, not the version string.
