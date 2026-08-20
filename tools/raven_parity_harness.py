#!/usr/bin/env python3
"""RAVEN-side reference for the M2 causal parity probe.

Runs **RAVEN's own** chunk-causal stack -- ``CausalMiniMaxH3DiTModel`` inside
``MiniMaxH3X0Model``, driven by ``CausalMiniMaxH3Base._text_cache_fill`` /
``_chunk_forward`` against ``utils.naive_cache.NaiveCache`` -- on the shared
inputs, taps every layer's varlen attention call, and writes a dump in
``tools/probe_causal_parity.py``'s ``KV_DUMP_SCHEMA``.

Nothing here re-implements the reference. The only code this file owns is glue:

* a packing shim, so RAVEN's real dataset packer
  (``CausalTextOnlyT2AVDataset._pack_sample``) runs without a tokenizer file --
  the text encoder is replaced by the shared inputs' fixed random states, which
  is exactly what "same inputs on both sides" requires;
* a ``ForwardInput`` built by RAVEN's own ``_build_inputs``;
* tensor-layout conversions between the Comfy convention in the shared input
  file (``[1, C, T, H, W]`` / ``[1, C, 2, A]``) and RAVEN's per-sample one
  (``[C, T, H, W]`` / ``[2, C, A]``);
* a tap on the module-level ``_CAUSAL_FLASH_ATTENTION`` singleton, which is
  where RAVEN's causal attention hands the merged ``[retained | current]`` K/V
  to its backend -- the same point the Comfy probe taps.

Two processes, never two resident models::

    # 1. shared inputs (either side can write them; they are just tensors)
    python tools/probe_causal_parity.py --mode inputs --arch full \\
        --frames 39 --width 512 --height 288 --text-len 128 \\
        --emit-inputs inputs.pt
    #    ... or, on a box with no ComfyUI:
    python tools/raven_parity_harness.py --emit-inputs inputs.pt --arch full \\
        --frames 39 --width 512 --height 288 --text-len 128

    # 2. RAVEN process (this file)
    python tools/raven_parity_harness.py --raven-root /root/Jarvis \\
        --weights h3_bf16.safetensors --inputs inputs.pt --arch full \\
        --dtype bf16 --device cuda --emit-dump raven_kv.pt --json raven.json

    # 3. ComfyUI process
    python tools/probe_causal_parity.py --mode real --arch full \\
        --dit h3_bf16.safetensors --inputs inputs.pt \\
        --compare-dump raven_kv.pt --json parity.json

Weights: a **runtime-layout** checkpoint (fused QKV), loaded with
``load_state_dict`` under the ``dit.`` prefix. No QKV reinterleave, no key
remap, no LoRA -- this is base-vs-base parity of the causal mechanism.

**BF16 only.** RAVEN's vendored blocks hard-cast every modulation result to
``_BF16_DTYPE`` (``projects/minimax_h3/modeling/transformer/model.py``), so an
fp32 or fp16 placement dies in the first block on a dtype mismatch. Both sides
therefore run bf16, and the comparison tolerance is a bf16 tolerance.

Status: exercised end to end against the Comfy probe on the ``tiny``
architecture (CPU, bf16), where the two implementations agree to
``atol 8e-3 / rtol 2e-2`` on K/V and on the per-chunk x0 -- about 2-3 bf16 ULP,
flat across layers and chunks. The full 50-block BF16 run needs the real
checkpoint and a vr-* box; it has not been run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _entry in (str(PROJECT_ROOT), str(TOOLS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Schema owner: the probe. Importing it (rather than restating it) is what keeps
# the two dumps in one format. It imports torch/ComfyUI lazily, inside
# functions, so this stays safe in a RAVEN-only environment.
from probe_causal_parity import (  # noqa: E402
    ARCHS,
    INPUTS_SCHEMA,
    KV_DUMP_SCHEMA,
    build_shared_inputs,
    load_inputs,
    load_state_dict_file,
    record_entry,
    runtime_meta,
    save_inputs,
    select_layers,
    write_dump,
)

#: The parity prompt is never tokenised for real: the shared inputs carry fixed
#: random text states in place of the encoder, so both sides condition on the
#: exact same rows without either loading Qwen3-VL.
PARITY_PROMPT = "raven causal parity probe"


# --- RAVEN import ------------------------------------------------------------


class RavenModules:
    """Everything this harness needs out of a RAVEN checkout, imported once."""

    def __init__(self, root: str) -> None:
        root_path = Path(root).expanduser()
        if not (root_path / "projects" / "minimax_h3").is_dir():
            raise SystemExit(
                f"--raven-root {root_path} does not look like a RAVEN checkout "
                "(no projects/minimax_h3)"
            )
        # RAVEN owns top-level ``utils``/``common`` packages, and so does
        # ComfyUI. They must never share a process; this file is the RAVEN one.
        sys.path.insert(0, str(root_path))
        self.root = str(root_path)

        from projects.minimax_h3.data.causal_text_only import CausalTextOnlyT2AVDataset
        from projects.minimax_h3.meta_models.causal_minimax_h3_base import (
            CausalMiniMaxH3Base,
        )
        from projects.minimax_h3.modeling.transformer import causal_model as causal_module
        from projects.minimax_h3.modeling.transformer.causal_model import (
            CausalMiniMaxH3DiTModel,
        )
        from projects.minimax_h3.modeling.transformer.config import (
            MiniMaxH3DiTArchConfig,
            MiniMaxH3DiTConfig,
        )
        from projects.minimax_h3.modeling.transformer.model import (
            MINIMAX_H3_FP32_BUFFER_NAMES,
            MINIMAX_H3_FP32_PARAM_NAMES,
        )
        from projects.minimax_h3.modeling.transformer.x0_model import MiniMaxH3X0Model
        from utils.naive_cache import NaiveCache

        self.dataset_cls = CausalTextOnlyT2AVDataset
        self.base_cls = CausalMiniMaxH3Base
        self.causal_module = causal_module
        self.dit_cls = CausalMiniMaxH3DiTModel
        self.arch_config_cls = MiniMaxH3DiTArchConfig
        self.dit_config_cls = MiniMaxH3DiTConfig
        self.x0_cls = MiniMaxH3X0Model
        self.cache_cls = NaiveCache
        self.fp32_params = MINIMAX_H3_FP32_PARAM_NAMES
        self.fp32_buffers = MINIMAX_H3_FP32_BUFFER_NAMES


# --- packing shim ------------------------------------------------------------


class _FixedLengthTokenizer:
    """Stands in for ``MiniMaxH3Tokenizer`` at a pinned prompt length.

    ``_pack_sample`` uses the tokenizer for exactly one thing: the token count
    and the ids it ships to the text encoder. The encoder is not run here (the
    shared inputs carry its output), so only the count is load-bearing, and it
    has to be the count the Comfy side used.
    """

    def __init__(self, text_len: int) -> None:
        self.text_len = int(text_len)

    def encode(self, prompt: str):
        import torch

        del prompt
        return torch.zeros(self.text_len, dtype=torch.long), self.text_len


class _PackerShim:
    """The attribute surface ``CausalTextOnlyT2AVDataset._pack_sample`` reads.

    Deliberately not a real dataset instance: the constructor wants a tokenizer
    file, a resume context and a worker topology, none of which exist in a
    parity run. The packing *code* is still RAVEN's -- ``_pack_sample`` is
    called unbound against this object -- so the chunk cut, the position ids and
    the flex-attention rectangles all come from the reference implementation.
    """

    def __init__(self, *, text_len, latent_t, latent_h, latent_w, audio_t,
                 latents_dim, audio_latents_dim, sink, window):
        self.tokenizer = _FixedLengthTokenizer(text_len)
        self.latent_t = int(latent_t)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.audio_t = int(audio_t)
        self.sink = int(sink)
        self.window_size = None if window is None else int(window)
        self.latent_shape = (int(latents_dim), int(latent_t), int(latent_h), int(latent_w))
        self.audio_shape = (2, int(audio_latents_dim), int(audio_t))


def build_batch(raven: RavenModules, request: Dict[str, Any]) -> Dict[str, List[Any]]:
    """One batch-of-one, packed by RAVEN's own dataset code."""
    shim = _PackerShim(
        text_len=request["text_len"],
        latent_t=request["latent_t"],
        latent_h=request["latent_h"],
        latent_w=request["latent_w"],
        audio_t=request["audio_t"],
        latents_dim=request["latents_dim"],
        audio_latents_dim=request["audio_latents_dim"],
        sink=request["sink"],
        window=request["window"],
    )
    sample = raven.dataset_cls._pack_sample(shim, prompt=PARITY_PROMPT)
    return {key: [value] for key, value in sample.items()}


# --- attention tap -----------------------------------------------------------


class RavenKVTap:
    """Wraps the module-level varlen attention singleton and records its inputs.

    RAVEN's ``CausalMiniMaxH3Attention`` merges the cache into ``k``/``v`` and
    then calls ``_CAUSAL_FLASH_ATTENTION(q, k, v, ...)``. Replacing that object
    for the duration of the run records exactly the tensors the backend sees,
    in block order, without touching the model.
    """

    def __init__(self, module: Any, full_layers: Sequence[int] = (), row_stride: int = 1,
                 attribute: str = "_CAUSAL_FLASH_ATTENTION") -> None:
        self.module = module
        self.attribute = attribute
        self.full_layers = {int(index) for index in full_layers}
        self.row_stride = max(1, int(row_stride))
        self.entries: List[Dict[str, Any]] = []
        self.forward_name = "?"
        self._original = None

    def install(self) -> "RavenKVTap":
        self._original = getattr(self.module, self.attribute)
        tap = self

        class _Traced:
            def __call__(self, q, k, v, *args, **kwargs):
                tap.record(q, k, v)
                return tap._original(q, k, v, *args, **kwargs)

        setattr(self.module, self.attribute, _Traced())
        return self

    def remove(self) -> None:
        if self._original is not None:
            setattr(self.module, self.attribute, self._original)
            self._original = None

    def __enter__(self) -> "RavenKVTap":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    def forward(self, name: str) -> "RavenKVTap":
        self.forward_name = name
        return self

    def record(self, q, k, v) -> None:
        layer = sum(1 for entry in self.entries if entry["forward"] == self.forward_name)
        self.entries.append(record_entry(
            self.forward_name, layer, q, k, v,
            store_full=layer in self.full_layers, row_stride=self.row_stride,
        ))


# --- tensor conventions ------------------------------------------------------


def video_to_raven(latent, start: int, stop: int):
    """Comfy ``[1, C, T, H, W]`` -> RAVEN per-sample ``[C, t, H, W]``."""
    return latent[0][:, start:stop].contiguous()


def audio_to_raven(latent, start: int, stop: int):
    """Comfy ``[1, C, 2, A]`` -> RAVEN per-sample ``[2, C, a]``."""
    return latent[0].permute(1, 0, 2)[:, :, start:stop].contiguous()


def video_to_comfy(latent):
    """RAVEN ``[C, t, H, W]`` -> Comfy ``[1, C, t, H, W]``."""
    return latent.unsqueeze(0)


def audio_to_comfy(latent):
    """RAVEN ``[2, C, a]`` -> Comfy ``[1, C, 2, a]``."""
    return latent.permute(1, 0, 2).unsqueeze(0)


# --- model -------------------------------------------------------------------


def build_model(raven: RavenModules, arch: str, dtype: str, device: str):
    """RAVEN's causal DiT inside the x0 wrapper, at the requested placement.

    The fp32 island (patch projections, time embedder, output heads, rope
    inverse frequencies) is kept in fp32 exactly as the checkpoint contract
    says, which is also how ComfyUI builds those modules.
    """
    import torch

    config = ARCHS[arch]
    arch_config = raven.arch_config_cls(
        num_layers=config["num_layers"],
        token_refiner_num_layers=config["token_refiner_num_layers"],
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        ffn_hidden_size=config["ffn_hidden_size"],
        latents_dim=config["latents_dim"],
        audio_latents_dim=config["audio_latents_dim"],
        text_dim=config["text_dim"],
        timestep_input_dim=config["timestep_input_dim"],
        time_embed_hidden_size=config["time_embed_hidden_size"],
        time_embed_dim=config["time_embed_dim"],
        adaln_out_features=18 * config["hidden_size"],
        final_adaln_out_features=2 * config["hidden_size"],
        rope_inv_freq_len=config["rope_inv_freq_len"],
    )
    dit = raven.dit_cls(config=raven.dit_config_cls(arch_config=arch_config), hf_config={})
    model = raven.x0_cls(dit)

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = model.to(device=device, dtype=torch_dtype)
    for name, parameter in dit.named_parameters():
        if name in raven.fp32_params:
            parameter.data = parameter.data.float()
    for name, buffer in dit.named_buffers():
        if name in raven.fp32_buffers:
            buffer.data = buffer.data.float()
    model.requires_grad_(False)
    model.eval()
    return model


def load_weights(raven: RavenModules, model, path: str) -> Dict[str, Any]:
    """Load a runtime-layout checkpoint under the ``dit.`` prefix, strictly."""
    state = load_state_dict_file(path)
    target = model.state_dict()
    prefixed = {}
    for key, value in state.items():
        name = key if key.startswith("dit.") else f"dit.{key}"
        reference = target.get(name)
        if reference is None:
            prefixed[name] = value
            continue
        prefixed[name] = value.to(dtype=reference.dtype, device=reference.device)
    missing, unexpected = model.load_state_dict(prefixed, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected), "keys": len(state)}


# --- rollout -----------------------------------------------------------------


def run_rollout(raven: RavenModules, model, inputs_payload: Dict[str, Any], *,
                tap: RavenKVTap, device: str, dtype: str) -> Dict[str, Any]:
    """text prefill -> per chunk: noise (cache read-only) then clean fill.

    The loop mirrors ``CausalMiniMaxH3Base._rollout_latents``, minus the
    sampler: one denoise forward per chunk at the shared timestep, then the
    cache-fill on the shared clean content. Every forward is RAVEN's.
    """
    import torch

    request = inputs_payload["request"]
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    batch = build_batch(raven, request)
    base = object.__new__(raven.base_cls)  # the glue below reads no config
    context = inputs_payload["context"][0].to(device=device, dtype=torch_dtype)
    forward_inputs = raven.base_cls._build_inputs(base, batch, [context])

    layout = forward_inputs.layouts[0]
    chunk_table = [
        {
            "video": (chunk.video_start, chunk.video_stop),
            "audio": (chunk.audio_start, chunk.audio_stop),
            "rows": chunk.rows,
            "has_clean": chunk.clean_start is not None,
        }
        for chunk in layout.chunks
    ]

    def to_device(name):
        return inputs_payload[name].to(device=device, dtype=torch_dtype)

    video_xt, audio_xt = to_device("video_xt"), to_device("audio_xt")
    video_x0, audio_x0 = to_device("video_x0"), to_device("audio_x0")
    video_eps, audio_eps = to_device("video_eps"), to_device("audio_eps")

    num_layers = raven.base_cls._num_layers(model)
    cache = raven.cache_cls(
        num_layers, 1, sink=[int(request["sink"])],
        window_size=[None if request["window"] is None else int(request["window"])],
    )

    tap.forward("text")
    base._text_cache_fill(model, forward_inputs, cache)

    outputs: List[Dict[str, Any]] = []
    video_sigma = float(request["video_sigma"])
    audio_sigma = float(request["audio_sigma"])
    for index, chunk in enumerate(layout.chunks):
        v0, v1 = chunk.video_start, chunk.video_stop
        a0, a1 = chunk.audio_start, chunk.audio_stop

        tap.forward(f"chunk{index}:noise")
        video_pred, audio_pred = base._chunk_forward(
            model, forward_inputs,
            chunk_index=index, role="noise",
            video_rows_source=[video_to_raven(video_xt, v0, v1)],
            audio_rows_source=[audio_to_raven(audio_xt, a0, a1)],
            video_timesteps=torch.tensor([video_sigma], device=device),
            audio_timesteps=torch.tensor([audio_sigma], device=device),
            cache=cache, update_cache=False,
        )
        outputs.append({
            "video_x0": video_to_comfy(video_pred[0]).detach().to("cpu").float(),
            "audio_x0": audio_to_comfy(audio_pred[0]).detach().to("cpu").float(),
        })

        if chunk.clean_start is None:
            break  # last chunk: nothing after it reads this history
        tap.forward(f"chunk{index}:clean")
        base._chunk_forward(
            model, forward_inputs,
            chunk_index=index, role="clean",
            video_rows_source=[video_to_raven(video_x0, v0, v1)],
            audio_rows_source=[audio_to_raven(audio_x0, a0, a1)],
            video_timesteps=torch.zeros(1, device=device),
            audio_timesteps=torch.zeros(1, device=device),
            cache=cache, update_cache=True,
            video_eps_source=[video_to_raven(video_eps, v0, v1)],
            audio_eps_source=[audio_to_raven(audio_eps, a0, a1)],
        )

    return {
        "outputs": outputs,
        "chunk_table": chunk_table,
        "num_layers": num_layers,
        "text_len": layout.text_len,
    }


def check_chunk_table(request: Dict[str, Any], chunk_table: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-check RAVEN's chunk cut against this repo's layout module.

    A pure assertion: nothing in the rollout is derived from it. If the two
    chunkings ever disagree, every downstream number is comparing different
    rows and the dump must not be trusted.
    """
    from raven_streaming.layout import T2VALayout

    layout = T2VALayout.from_request(
        text_len=request["text_len"], frames=request["frames"],
        width=request["width"], height=request["height"],
    )
    ours = [
        {"video": (c.video_start, c.video_stop), "audio": (c.audio_start, c.audio_stop),
         "rows": c.rows}
        for c in layout.chunks
    ]
    theirs = [{k: v for k, v in entry.items() if k != "has_clean"} for entry in chunk_table]
    return {
        "agrees": ours == theirs,
        "ours": ours,
        "raven": theirs,
    }


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raven-root", default=os.environ.get("RAVEN_ROOT", "/root/Jarvis"),
                        help="RAVEN checkout (default: $RAVEN_ROOT or /root/Jarvis)")
    parser.add_argument("--weights", default=None,
                        help="runtime-layout H3 checkpoint (fused QKV, no remap)")
    parser.add_argument("--inputs", default=None, help="shared input file to read")
    parser.add_argument("--emit-inputs", default=None,
                        help="write a shared input file and exit (no RAVEN needed)")
    parser.add_argument("--arch", choices=tuple(ARCHS), default="full")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                        help="bf16 only in practice; see the module docstring")
    parser.add_argument("--emit-dump", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--kv-layers", default="first,mid,last",
                        help="which blocks keep full K/V ('all', 'none', '0,7,49')")
    parser.add_argument("--kv-row-stride", type=int, default=1)
    parser.add_argument("--attention-backend", choices=("auto", "sdpa"), default="auto",
                        help="sdpa forces RAVEN's packed SDPA fallback (no FlashAttention), "
                             "which is what pairs with the probe's --attention-backend pytorch")

    grid = parser.add_argument_group("request grid (only for --emit-inputs)")
    grid.add_argument("--frames", type=int, default=39)
    grid.add_argument("--width", type=int, default=512)
    grid.add_argument("--height", type=int, default=288)
    grid.add_argument("--text-len", type=int, default=128)
    grid.add_argument("--sink", type=int, default=2)
    grid.add_argument("--window", type=int, default=2, help="-1 means None (no eviction)")
    grid.add_argument("--sigma", type=float, default=0.6)
    grid.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.window is not None and args.window < 0:
        args.window = None
    if args.dtype != "bf16" and args.weights:
        raise SystemExit(
            f"--dtype {args.dtype} cannot run: RAVEN's blocks hard-cast modulation "
            "output to bf16 (_BF16_DTYPE in modeling/transformer/model.py), so any "
            "other placement raises a dtype mismatch inside the first block. Run "
            "both sides in bf16."
        )
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Placement and backend must be decided before torch / RAVEN are imported:
    # ``get_device()`` reads CUDA availability and ``utils.flash_attn`` binds its
    # backend at import time.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.attention_backend == "sdpa":
        os.environ["FLASH_ATTN_3_AVAILABLE"] = "0"
        os.environ["FLASH_ATTN_2_AVAILABLE"] = "0"

    if args.emit_inputs:
        payload = build_shared_inputs(
            frames=args.frames, width=args.width, height=args.height,
            text_len=args.text_len, seed=args.seed, sink=args.sink, window=args.window,
            video_sigma=args.sigma, audio_sigma=None, arch=args.arch,
        )
        save_inputs(args.emit_inputs, payload)
        print(f"inputs: {args.emit_inputs}")
        print(json.dumps(payload["request"], indent=2))
        if not args.weights:
            return 0

    if not args.weights:
        raise SystemExit("--weights is required (runtime-layout checkpoint)")
    if not args.inputs and not args.emit_inputs:
        raise SystemExit("--inputs is required: both sides must read the same tensors")

    import torch

    raven = RavenModules(args.raven_root)
    from common.distributed.ops import get_device

    device = str(get_device())
    payload = load_inputs(args.inputs or args.emit_inputs)
    request = payload["request"]
    if request["arch"] != args.arch:
        raise SystemExit(
            f"shared inputs were built for arch {request['arch']!r}, "
            f"this run asks for {args.arch!r}"
        )

    model = build_model(raven, args.arch, args.dtype, device)
    load_report = load_weights(raven, model, args.weights)
    if load_report["missing"] or load_report["unexpected"]:
        raise SystemExit(
            f"checkpoint does not fit the {args.arch} architecture: "
            f"{len(load_report['missing'])} missing, "
            f"{len(load_report['unexpected'])} unexpected "
            f"(first missing: {load_report['missing'][:3]}, "
            f"first unexpected: {load_report['unexpected'][:3]})"
        )

    num_layers = raven.base_cls._num_layers(model)
    tap = RavenKVTap(raven.causal_module,
                     full_layers=select_layers(args.kv_layers, num_layers),
                     row_stride=args.kv_row_stride)
    with torch.no_grad(), tap:
        result = run_rollout(raven, model, payload, tap=tap, device=device, dtype=args.dtype)

    chunk_check = check_chunk_table(request, result["chunk_table"])
    meta = {
        "producer": "raven",
        "runtime": runtime_meta(),
        "raven_root": raven.root,
        "weights": str(args.weights),
        "arch": args.arch,
        "dtype": args.dtype,
        "device": device,
        "request": request,
        "parity_scope": (
            "base vs base: runtime-layout checkpoint, no LoRA, no QKV reinterleave"
        ),
        "kv_selection": {"layers": select_layers(args.kv_layers, num_layers),
                         "row_stride": args.kv_row_stride},
        "attention_backend": args.attention_backend,
        "flash_attn_env": {
            "FLASH_ATTN_2_AVAILABLE": os.environ.get("FLASH_ATTN_2_AVAILABLE"),
            "FLASH_ATTN_3_AVAILABLE": os.environ.get("FLASH_ATTN_3_AVAILABLE"),
        },
        "weights_load": load_report,
        "chunk_table": result["chunk_table"],
        "chunk_table_agrees_with_raven_streaming": chunk_check["agrees"],
        "num_layers": num_layers,
        "attention_calls": len(tap.entries),
        "inputs_schema_version": INPUTS_SCHEMA["version"],
    }

    print(f"RAVEN: {raven.root}")
    print(f"device={device} dtype={args.dtype} arch={args.arch} layers={num_layers}")
    print(f"chunks={len(result['chunk_table'])} attention_calls={len(tap.entries)}")
    print(f"chunk table agrees with raven_streaming.layout: {chunk_check['agrees']}")
    if not chunk_check["agrees"]:
        print(json.dumps(chunk_check, indent=2))

    if args.emit_dump:
        write_dump(args.emit_dump, "raven", tap.entries, result["outputs"], meta)
        print(f"dump: {args.emit_dump}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"meta": meta, "kv_dump_schema": KV_DUMP_SCHEMA}, indent=2, default=str))
        print(f"report: {args.json}")
    return 0 if chunk_check["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
