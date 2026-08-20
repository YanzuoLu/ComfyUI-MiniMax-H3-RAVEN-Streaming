# Workflows

Two files, same graph, different formats.

| File | Format | How to use it |
|---|---|---|
| `raven_t2va_streaming.json` | **UI workflow** (litegraph, `version: 0.4`) | Drag it onto the ComfyUI canvas, or *Workflow -> Open*. Loads as an editable graph. |
| `raven_t2va_streaming_api.json` | **API prompt** | `POST /prompt` with `{"prompt": <this file>}`. Does **not** open in the UI. |

Both reference the public filenames from the repository `README.md`:

```
models/diffusion_models/minimax_h3_fl2va_bf16.safetensors
models/loras/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
```

If your copies are named differently, re-pick them in the combo widgets (UI) or
edit the strings (API).

## The graph

```
RAVEN Model Loader ──MODEL─────────────────────────┐
                     (insert LoraLoaderModelOnly    │
                      here for extra LoRAs)         │
CLIPLoader (minimax) ──CLIP──┐                      │
VAELoader (video) ──VAE──────┴─> MiniMaxH3ImageToVideo (T2VA)
       │                            ├─CONDITIONING──┤
       │                            └─LATENT────────┤
       └──────────────────────── video_vae ─────────┤
VAELoader (audio) ───────────── audio_vae ──────────┤
                                                    │
                              RAVEN Streaming Sampler
                                    ├─ LATENT  (unconnected)
                                    ├─ IMAGE ──> SaveAnimatedWEBP (fps 24)
                                    └─ AUDIO ──> SaveAudio (FLAC)
```

Notes on the wiring:

- The **video** VAE goes to two places: the H3 node's `vae` input (required by
  the official schema even in T2VA form) and the sampler's `video_vae`.
- `first_frame` / `last_frame` stay unconnected. Connecting them is `fl2va`,
  which is out of scope for 0.1.x.
- Extra LoRAs are stacked with stock `LoraLoaderModelOnly` on the `MODEL` wire,
  between the loader and the sampler. The mandatory RAVEN adapter is already
  inside the `MODEL` at strength 1.0 and is unaffected.
- Width, height and length are set on the H3 node, not on the sampler: multiples
  of 32, `width * height <= 1376 * 768`, `length = 17k + 5` with `k >= 1`
  (22 min, 362 max; above 192 is experimental).
- The sampler's `LATENT` output is left unconnected in both files. Connect it if
  you want to re-decode or inspect the finished AV latent.
- Saving: `SaveAnimatedWEBP` + `SaveAudio` take `IMAGE` and `AUDIO` directly.
  Upstream marks `SaveAudio` deprecated at the pinned baseline — it still works;
  `SaveAudioAdvanced` is its replacement. For a single muxed file, swap both for
  `CreateVideo` (`images` + `audio`, `fps 24`) -> `SaveVideo`.

## Preview

The live preview appears **inside** the `RAVEN Streaming Sampler` node while it
samples; no preview node is needed and none exists. It starts muted, video
becomes visible one chunk behind sampling and audio trails it by a further
0.425 s. If the preview cannot start, the run is unaffected — see the repository
`README.md`.
