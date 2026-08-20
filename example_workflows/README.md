# Example workflows

One graph, two formats, in two directories — and the split is not cosmetic.

| File | Format | How to use it |
|---|---|---|
| `example_workflows/minimax_h3_raven_streaming_t2va.json` | **UI workflow** (litegraph, `version: 0.4`) | *Workflow -> Browse Templates -> Extensions*, search `minimax_h3_raven_streaming_t2va`. Or drag it onto the canvas / *Workflow -> Open*. |
| `../api_workflows/raven_t2va_streaming_api.json` | **API prompt** | `POST /prompt` with `{"prompt": <this file>}`. Does **not** open in the UI. |

ComfyUI (`app/custom_node_manager.py` at the pinned 0.33.0 baseline) globs
`<custom_node>/example_workflows/*.json` — plus the legacy aliases `example`,
`examples`, `workflow`, `workflows` — and offers **every** JSON it finds as a
template, named after the file. That is why:

- the UI workflow is named `minimax_h3_raven_streaming_t2va.json`: the file name
  is the template title and the string you search for. The matching
  `minimax_h3_raven_streaming_t2va.jpg` next to it is the template thumbnail.
- the API prompt lives in `api_workflows/`, a name the scanner does not know. It
  would otherwise be listed as a second template and open as a broken graph.
- there is no `workflows/` directory any more. It was one of the scanned
  aliases, so keeping it would have re-introduced exactly that problem.

Only `*.json` is globbed, so this README is not a template. Non-JSON files in
this folder are served but never listed.

Both files reference the public filenames from the repository `README.md`
(nothing is downloaded automatically):

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
- `kv_cache_storage` is `cpu_pinned` in both files. It keeps the retained KV
  cache in page-locked host RAM (~0.56 GiB on the card instead of ~21-28 GiB),
  which is what makes 1376x768 x 192 fit a 24 GiB envelope. It changes where
  bytes live, not what is computed.
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
