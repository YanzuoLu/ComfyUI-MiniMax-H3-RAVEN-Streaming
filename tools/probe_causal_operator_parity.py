#!/usr/bin/env python3
"""Operator-level audit: pinned ComfyUI vs real RAVEN, one DiT block, stage by stage.

Why this exists
---------------
``tools/probe_causal_parity.py`` measures the whole causal rollout and reports
where it lands: on the real 50-block BF16 model, block 0's Q/K/V already differ
by ~0.5-1% relative L2 and the chunk-0 ``video_x0`` ends at ~9% (cosine
0.99596). That is a *smooth* accumulation, not a wiring bug -- which means the
answer is in the operators, and a rollout probe cannot say which one.

This probe takes **one block** (default 49, ``--block 0`` for the shallow one),
feeds both implementations the *same* deterministic BF16 inputs, and compares
each stage separately:

===========================  ===================================================
stage                        Comfy                       RAVEN
===========================  ===================================================
``rope_freqs``               ``MiniMaxH3Model.rope_freqs``  ``MiniMaxH3Rope``
``qkv_gemm.{q,k,v}``         ``Attention.qkv_proj``         ``MiniMaxH3Attention.qkv_proj``
``qk_norm.{q,k}``            ``Attention.q_norm/k_norm``    ``_apply_qk_norm``
``qk_norm_rope.{q,k}``       ``ck.rms_rope_split_half_``    ``_apply_qk_norm`` + ``_apply_rope_qk``
``attention.out``            ``optimized_attention``        ``_CAUSAL_FLASH_ATTENTION``
``attention.proj_out``       ``Attention.out_proj``         ``RowParallelLinear``
``adaln_gemm.*``             ``AdalnProj.linear``           ``MiniMaxH3AdalnProj.project_local``
``adaln_native.*``           ``AdalnProj`` (silu in bf16)   ``silu(t_emb)`` fp32 -> bf16
``modulate.scale_shift``     ``_mod_scale_shift``           ``_modulate_scale_shift``
``modulate.gate``            ``_mod_gate``                  ``_modulate_gate``
``mlp.fc1`` / ``swiglu`` / ``out``  ``MLP`` + ``linear_input_act``  ``MiniMaxH3MLP`` + ``_silu_mul``
===========================  ===================================================

Every number comes from the implementation's own function or module -- nothing
here re-implements either side. Two of the stages are only *separable* on one
side, and the dump says so:

* Comfy fuses q/k RMSNorm and RoPE into one kernel, so ``qk_norm`` (norm alone)
  is that same ``RMSNorm`` module invoked on its own, while ``qk_norm_rope`` is
  the production fused call;
* Comfy fuses SwiGLU into the ``fc2`` matmul, so ``mlp.swiglu`` is Comfy's own
  ``INPUT_ACT_EAGER['swiglu']`` -- the exact function ``linear_input_act`` calls
  on a non-INT8 weight -- and ``mlp.out`` is the real ``MLP.forward``.

Each side also runs a **self-check**: the staged replay of the attention path is
compared against that side's real attention ``forward`` on the same input. If
the replay ever stops being the production path, the self-check says so before
any cross-side number is believed.

Memory: only ``blocks.<N>.*`` plus ``rope.inv_freq`` is read out of the
safetensors file (``safe_open`` + per-key ``get_tensor``), so a 66 GB checkpoint
costs one block (~1.2 GB in BF16 for the full arch). The full model is never
constructed.

Process model: ComfyUI and RAVEN both own top-level ``utils`` / ``common``
packages and cannot share an interpreter, so ``--side both`` (the default) is an
orchestrator: it writes one shared input file and spawns ``--side comfy`` and
``--side raven`` as subprocesses, then compares their dumps. Each subprocess
holds one block, never two models.

That separation has to be enforced through the **environment**, not through
``sys.path`` order. RAVEN's ``utils`` is a namespace package (no ``__init__``)
and ComfyUI's is a regular one, and Python's finder keeps scanning past a
namespace portion until it finds a regular package -- so an inherited
``PYTHONPATH`` entry pointing at ComfyUI wins over ``sys.path.insert(0, raven)``
and ``from utils.flash_attn import FlashAttention`` dies with
ModuleNotFoundError. Observed on vr. Each side therefore gets an explicit
``PYTHONPATH`` (its own root, then this project), its own ``cwd``, and its own
interpreter (``--comfy-python`` / ``--raven-python``, both defaulting to
``--python``); the ``both`` run prints all three before it launches anything.

Usage::

    # everything at once (spawns both sides)
    python tools/probe_causal_operator_parity.py \\
        --weights h3_bf16.safetensors --block 49 \\
        --raven-root /root/Jarvis --comfyui-path /path/to/ComfyUI \\
        --device cuda --json operator_block49.json

    # or by hand, one side at a time
    python tools/probe_causal_operator_parity.py --side inputs --emit-inputs op_inputs.pt
    python tools/probe_causal_operator_parity.py --side comfy --inputs op_inputs.pt \\
        --weights h3_bf16.safetensors --block 49 --emit-dump comfy_ops.pt
    python tools/probe_causal_operator_parity.py --side raven --inputs op_inputs.pt \\
        --weights h3_bf16.safetensors --block 49 --raven-root /root/Jarvis \\
        --emit-dump raven_ops.pt
    python tools/probe_causal_operator_parity.py --side compare \\
        --comfy-dump comfy_ops.pt --raven-dump raven_ops.pt --json operator.json

Status: the stage plumbing runs on CPU with the ``tiny`` architecture (both
sides). The real block-49 BF16 CUDA audit is a vr-* run and has not happened.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _entry in (str(PROJECT_ROOT), str(TOOLS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from probe_causal_parity import (  # noqa: E402
    ARCHS,
    Check,
    Report,
    COSINE_MIN,
    _metric_detail,
    configure_attention,
    find_comfyui,
    load_state_dict_file,
    metrics_pass,
    runtime_meta,
    tensor_metrics,
)

OPERATOR_INPUTS_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "request": "arch/block-independent geometry: rows, kv_rows, dims, seed, dtype",
    "x": "[rows, hidden] float32 CPU, the block input",
    "positions": "[kv_rows, 3] float64 absolute (t, h, w); the query rows are the last `rows`",
    "attn_q": "[rows, heads, head_dim] float32 CPU",
    "attn_k": "[kv_rows, heads, head_dim] float32 CPU (merged [retained | current])",
    "attn_v": "[kv_rows, heads, head_dim] float32 CPU",
    "t_emb": "[M, t_dim] float32 CPU, pre-SiLU time embedding rows",
    "adaln_input": "[M, t_dim] float32 CPU, already SiLU'd (isolates the GEMM)",
    "mod_rows": "[rows] int64, AdaLN mod-row index per token (t_row * 3 + tag)",
    "mod_x": "[rows, hidden] float32 CPU, modulation input",
    "mod_other": "[rows, hidden] float32 CPU, the gated branch output",
    "shift/scale/gate": "[M*3, hidden] float32 CPU, shared modulation vectors",
    "mlp_x": "[rows, hidden] float32 CPU",
}

OPERATOR_DUMP_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "side": "'comfy' or 'raven'",
    "block": "int block index the weights came from",
    "stages": "dict stage-name -> float32 CPU tensor",
    "sources": "dict stage-name -> the function/module that produced it",
    "meta": "runtime, dtypes, resolved backends, self-check metrics",
}

#: Stage gate. The measured rollout sits at 0.5-1% rel_l2 by block 0, so a
#: single operator above 1% is the thing to look at first.
STAGE_REL_L2_MAX = 0.01

STAGE_ORDER: Tuple[str, ...] = (
    "rope_freqs",
    "qkv_gemm.q", "qkv_gemm.k", "qkv_gemm.v",
    "qk_norm.q", "qk_norm.k",
    "qk_norm_rope.q", "qk_norm_rope.k",
    "attention.out", "attention.proj_out",
    "adaln_gemm.shift_msa", "adaln_gemm.scale_msa", "adaln_gemm.gate_msa",
    "adaln_gemm.shift_mlp", "adaln_gemm.scale_mlp", "adaln_gemm.gate_mlp",
    "adaln_native.shift_msa", "adaln_native.scale_msa", "adaln_native.gate_msa",
    "adaln_native.shift_mlp", "adaln_native.scale_mlp", "adaln_native.gate_mlp",
    "modulate.scale_shift", "modulate.gate",
    "mlp.fc1", "mlp.swiglu", "mlp.out",
)

ADALN_NAMES = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")


# --- shared inputs -----------------------------------------------------------


def build_operator_inputs(
    *,
    arch: str,
    frames: int,
    width: int,
    height: int,
    text_len: int,
    seed: int,
    rows: Optional[int] = None,
    kv_rows: Optional[int] = None,
    unique_timesteps: int = 2,
) -> Dict[str, Any]:
    """Deterministic operator inputs, on the geometry the real lane uses.

    Positions are the real thing: the text rows followed by chunk 0's rows of a
    T2VA layout, i.e. exactly the ``[retained | current]`` frame a cached chunk
    forward attends over. Query rows are the last ``rows`` of that.
    """
    import torch

    from raven_streaming.layout import T2VALayout

    config = ARCHS[arch]
    layout = T2VALayout.from_request(text_len=text_len, frames=frames, width=width,
                                     height=height)
    positions = torch.cat([layout.text_position_ids(), layout.chunk_position_ids(0)])
    available = int(positions.shape[0])
    chunk_rows = layout.chunks[0].rows
    rows = int(rows) if rows else chunk_rows
    kv_rows = int(kv_rows) if kv_rows else available
    if rows > kv_rows or kv_rows > available:
        raise SystemExit(
            f"rows/kv_rows must satisfy rows <= kv_rows <= {available} "
            f"(text {text_len} + chunk0 {chunk_rows}), got {rows}/{kv_rows}"
        )
    positions = positions[:kv_rows]

    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    t_dim = config["time_embed_dim"]
    generator = torch.Generator().manual_seed(int(seed))

    def draw(shape, scale=1.0):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    modality = 3
    mod_rows = torch.zeros(rows, dtype=torch.long)
    audio_rows = layout.chunks[0].audio_rows if rows >= layout.chunks[0].rows else rows // 4
    # audio rows first, then video rows: the packed order of a real chunk
    mod_rows[:audio_rows] = 1 * modality + 2      # t_row 1, audio tag
    mod_rows[audio_rows:] = 0 * modality + 0      # t_row 0, video tag

    return {
        "version": OPERATOR_INPUTS_SCHEMA["version"],
        "request": {
            "arch": arch, "frames": frames, "width": width, "height": height,
            "text_len": text_len, "seed": int(seed), "rows": rows, "kv_rows": kv_rows,
            "hidden": hidden, "heads": heads, "head_dim": head_dim, "t_dim": t_dim,
            "ffn": config["ffn_hidden_size"], "unique_timesteps": unique_timesteps,
            "audio_rows": int(audio_rows),
        },
        # activations are drawn at RMSNorm-ish scale so bf16 rounding is realistic
        "x": draw((rows, hidden)),
        "positions": positions,
        "attn_q": draw((rows, heads, head_dim)),
        "attn_k": draw((kv_rows, heads, head_dim)),
        "attn_v": draw((kv_rows, heads, head_dim)),
        "t_emb": draw((unique_timesteps, t_dim)),
        "adaln_input": draw((unique_timesteps, t_dim)),
        "mod_rows": mod_rows,
        "mod_x": draw((rows, hidden)),
        "mod_other": draw((rows, hidden)),
        "shift": draw((unique_timesteps * modality, hidden), 0.1),
        "scale": draw((unique_timesteps * modality, hidden), 0.1),
        "gate": draw((unique_timesteps * modality, hidden), 0.1),
        "mlp_x": draw((rows, hidden)),
    }


def save_operator_inputs(path: str, inputs: Dict[str, Any]) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, path)


def load_operator_inputs(path: str) -> Dict[str, Any]:
    import torch

    inputs = torch.load(path, map_location="cpu", weights_only=False)
    if int(inputs.get("version", -1)) != OPERATOR_INPUTS_SCHEMA["version"]:
        raise SystemExit(
            f"{path}: operator inputs schema {inputs.get('version')} != "
            f"{OPERATOR_INPUTS_SCHEMA['version']}"
        )
    return inputs


# --- partial checkpoint read -------------------------------------------------


def read_block_state(path: str, block: int,
                     extra_keys: Sequence[str] = ("rope.inv_freq",)) -> Dict[str, Any]:
    """Read only ``blocks.<block>.*`` (plus ``extra_keys``) out of a checkpoint.

    A safetensors file is read key by key through ``safe_open``, so a 66 GB
    checkpoint costs one block. Anything else (a ``.pt`` used by the tiny
    smoke runs) is loaded and filtered, which is only acceptable because those
    files are small.
    """
    prefix = f"blocks.{int(block)}."
    wanted_extra = set(extra_keys)
    if str(path).endswith(".safetensors"):
        from safetensors import safe_open

        state: Dict[str, Any] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                if key.startswith(prefix) or key in wanted_extra:
                    state[key] = handle.get_tensor(key)
        if not any(k.startswith(prefix) for k in state):
            blocks = sorted({int(k.split(".")[1]) for k in keys if k.startswith("blocks.")})
            raise SystemExit(
                f"{path} has no {prefix}* tensors; blocks present: "
                f"{blocks[:3]}..{blocks[-3:] if blocks else []}"
            )
        return state
    full = load_state_dict_file(path)
    state = {k: v for k, v in full.items() if k.startswith(prefix) or k in wanted_extra}
    if not any(k.startswith(prefix) for k in state):
        raise SystemExit(f"{path} has no {prefix}* tensors")
    return state


def strip_block_prefix(state: Dict[str, Any], block: int) -> Dict[str, Any]:
    prefix = f"blocks.{int(block)}."
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _store(tensor) -> Any:
    return tensor.detach().to("cpu", copy=True).float().contiguous()


def _segments_from_rows(mod_rows) -> List[Tuple[int, int, int]]:
    """Per-token mod-row indices -> Comfy's contiguous ``(start, stop, row)`` runs."""
    values = mod_rows.view(-1).tolist()
    segments: List[Tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            segments.append((start, index, int(values[start])))
            start = index
    return segments


# --- Comfy side --------------------------------------------------------------


def run_comfy_side(args, inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    import comfy.model_management
    import comfy.ops
    import comfy.quant_ops
    from comfy.ldm.minimax.model import (
        AdalnProj,
        Attention,
        MLP,
        MiniMaxH3Model,
        _mod_gate,
        _mod_scale_shift,
        rope_rotation_table,
    )
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    config = ARCHS[args.arch]
    request = inputs["request"]
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    ops = comfy.ops.disable_weight_init

    block_state = read_block_state(args.weights, args.block)
    local = strip_block_prefix(block_state, args.block)
    rope_inv_freq = block_state.get("rope.inv_freq")
    if rope_inv_freq is None:
        raise SystemExit("checkpoint has no rope.inv_freq; cannot build RoPE frequencies")

    attn = Attention(config["hidden_size"], config["num_attention_heads"],
                     config["attention_head_dim"], 1e-5,
                     dtype=dtype, device=device, operations=ops)
    mlp = MLP(config["hidden_size"], config["ffn_hidden_size"],
              dtype=dtype, device=device, operations=ops)
    adaln = AdalnProj(config["time_embed_dim"], config["hidden_size"], 6, 3,
                      apply_silu=True, dtype=dtype, device=device, operations=ops)
    _load_subset(attn, local, "attn.")
    _load_subset(mlp, local, "mlp.")
    _load_subset(adaln, local, "adaln_proj.")
    for module in (attn, mlp, adaln):
        module.to(device=device).requires_grad_(False).eval()

    rows, kv_rows = request["rows"], request["kv_rows"]
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]

    def to_device(name, cast=True):
        tensor = inputs[name].to(device)
        return tensor.to(dtype) if cast else tensor

    x = to_device("x")
    positions_q = inputs["positions"][-rows:].to(device)

    stages: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    # -- rope frequencies (fp32): the model's own method, on a shim that carries
    #    only the buffer it reads
    class _RopeShim:
        pass

    shim = _RopeShim()
    shim.rope = _RopeShim()
    shim.rope.inv_freq = rope_inv_freq.to(device=device, dtype=torch.float32)
    freqs = MiniMaxH3Model.rope_freqs(shim, positions_q, device)
    stages["rope_freqs"] = _store(freqs)
    sources["rope_freqs"] = "comfy.ldm.minimax.model.MiniMaxH3Model.rope_freqs"
    rope_table = rope_rotation_table(freqs, dtype)

    # -- (a) fused QKV projection
    qkv = attn.qkv_proj(x)
    q_raw, k_raw, v_raw = qkv.split(heads * head_dim, dim=-1)
    q_raw = q_raw.view(rows, heads, head_dim)
    k_raw = k_raw.view(rows, heads, head_dim)
    v_raw = v_raw.view(rows, heads, head_dim)
    stages["qkv_gemm.q"] = _store(q_raw)
    stages["qkv_gemm.k"] = _store(k_raw)
    stages["qkv_gemm.v"] = _store(v_raw)
    for name in ("q", "k", "v"):
        sources[f"qkv_gemm.{name}"] = "Attention.qkv_proj (operations.Linear)"

    # -- (b) q/k RMSNorm alone. Comfy's production path fuses norm with RoPE, so
    #    this is that same RMSNorm module invoked on its own, for localisation.
    stages["qk_norm.q"] = _store(attn.q_norm(q_raw.clone()))
    stages["qk_norm.k"] = _store(attn.k_norm(k_raw.clone()))
    sources["qk_norm.q"] = "Attention.q_norm (module, not the production fused path)"
    sources["qk_norm.k"] = "Attention.k_norm (module, not the production fused path)"

    # -- (c) production norm+RoPE: the exact fused call Attention.forward makes
    q_nr, k_nr, fused_used = _comfy_norm_rope(attn, q_raw, k_raw, rope_table, rows,
                                              heads, head_dim, x.device)
    stages["qk_norm_rope.q"] = _store(q_nr)
    stages["qk_norm_rope.k"] = _store(k_nr)
    kernel = ("comfy.quant_ops.ck.rms_rope_split_half_"
              if fused_used else "q_norm/k_norm + rope_rotation_table (in_training branch)")
    sources["qk_norm_rope.q"] = kernel
    sources["qk_norm_rope.k"] = kernel

    # -- (d) attention over the shared q/k/v, then the output projection
    attn_out = _comfy_attention(optimized_attention, AttentionTensorContainer,
                                to_device("attn_q"), to_device("attn_k"),
                                to_device("attn_v"), heads)
    stages["attention.out"] = _store(attn_out)
    sources["attention.out"] = "comfy.ldm.modules.attention.optimized_attention"
    stages["attention.proj_out"] = _store(attn.out_proj(attn_out))
    sources["attention.proj_out"] = "Attention.out_proj (operations.Linear)"

    # -- (e) AdaLN: the GEMM alone, then the production path from raw t_emb
    adaln_input = to_device("adaln_input")
    projected = adaln.linear(adaln_input)
    projected = projected.view(projected.shape[0] * adaln.modalities,
                               adaln.expand * adaln.hidden)
    for name, tensor in zip(ADALN_NAMES, projected.chunk(adaln.expand, dim=-1)):
        stages[f"adaln_gemm.{name}"] = _store(tensor)
        sources[f"adaln_gemm.{name}"] = "AdalnProj.linear + view/chunk (no SiLU)"
    for name, tensor in zip(ADALN_NAMES, adaln(to_device("t_emb"))):
        stages[f"adaln_native.{name}"] = _store(tensor)
        sources[f"adaln_native.{name}"] = "AdalnProj.forward (SiLU at the module dtype)"

    # -- (f) modulation and gate, on shared shift/scale/gate vectors
    segments = _segments_from_rows(inputs["mod_rows"])
    scale_shift = _mod_scale_shift(to_device("mod_x").clone(), to_device("shift"),
                                   to_device("scale"), segments)
    stages["modulate.scale_shift"] = _store(scale_shift)
    sources["modulate.scale_shift"] = "comfy.ldm.minimax.model._mod_scale_shift"
    gated = _mod_gate(to_device("mod_x").clone(), to_device("gate"),
                      to_device("mod_other"), segments)
    stages["modulate.gate"] = _store(gated)
    sources["modulate.gate"] = "comfy.ldm.minimax.model._mod_gate"

    # -- (g) MLP
    mlp_x = to_device("mlp_x")
    fc1 = mlp.fc1(mlp_x)
    stages["mlp.fc1"] = _store(fc1)
    sources["mlp.fc1"] = "MLP.fc1 (operations.Linear)"
    stages["mlp.swiglu"] = _store(comfy.ops.INPUT_ACT_EAGER["swiglu"](fc1.clone()))
    sources["mlp.swiglu"] = "comfy.ops.INPUT_ACT_EAGER['swiglu'] (what linear_input_act calls)"
    stages["mlp.out"] = _store(mlp(mlp_x))
    sources["mlp.out"] = "MLP.forward (comfy.ops.linear_input_act, fc2 fused)"

    # -- self-check: staged replay vs the real Attention.forward
    replay_q, replay_k, _ = _comfy_norm_rope(attn, q_raw, k_raw, rope_table, rows,
                                             heads, head_dim, x.device)
    replay = attn.out_proj(_comfy_attention(optimized_attention, AttentionTensorContainer,
                                            replay_q, replay_k, v_raw.clone(), heads))
    real = attn(x, rope_freqs=rope_table, transformer_options={})
    selfcheck = tensor_metrics(_store(replay), _store(real))

    meta = {
        "side": "comfy",
        "block": int(args.block),
        "arch": args.arch,
        "dtype": args.dtype,
        "device": str(device),
        "runtime": runtime_meta(),
        "attention_backend": _comfy_backend_name(),
        "fused_qknorm_rope": bool(fused_used),
        "comfy_kitchen": _comfy_kitchen_meta(),
        "weights": str(args.weights),
        "loaded_keys": sorted(local),
        "selfcheck.attention": selfcheck,
        "request": request,
    }
    return {"version": OPERATOR_DUMP_SCHEMA["version"], "side": "comfy",
            "block": int(args.block), "stages": stages, "sources": sources, "meta": meta}


def _load_subset(module, local_state: Dict[str, Any], prefix: str) -> None:
    """Strictly load the ``prefix``-scoped slice of one block's state dict."""
    import torch

    subset = {k[len(prefix):]: v for k, v in local_state.items() if k.startswith(prefix)}
    target = module.state_dict()
    missing = sorted(set(target) - set(subset))
    unexpected = sorted(set(subset) - set(target))
    if missing or unexpected:
        raise SystemExit(
            f"block slice {prefix!r} does not fit {type(module).__name__}: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    module.load_state_dict({k: v.to(dtype=target[k].dtype) for k, v in subset.items()},
                           strict=True)
    del torch


def _comfy_norm_rope(attn, q_raw, k_raw, rope_table, rows, heads, head_dim, device):
    """The fused q/k RMSNorm + split-half RoPE exactly as ``Attention.forward`` runs it."""
    import comfy.model_management
    import comfy.quant_ops

    q = q_raw.clone().view(1, rows, heads, head_dim)
    k = k_raw.clone().view(1, rows, heads, head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=device)
    rot = rope_table.shape[-3] * 2
    fused = not comfy.model_management.in_training
    if fused:
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_table, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    else:
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q, k, rope_table, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0], k[0], fused


def _comfy_attention(optimized_attention, container_cls, q, k, v, heads):
    """``optimized_attention`` with the model's own container/layout conventions."""
    out = optimized_attention(
        container_cls(q.transpose(0, 1).unsqueeze(0)),
        container_cls(k.transpose(0, 1).unsqueeze(0)),
        container_cls(v.transpose(0, 1).unsqueeze(0)),
        heads, mask=None, skip_reshape=True, transformer_options={},
    )
    return out.squeeze(0)


def _comfy_backend_name() -> Optional[str]:
    try:
        from comfy.ldm.modules.attention import optimized_attention

        return getattr(getattr(optimized_attention, "__wrapped__", optimized_attention),
                       "__name__", None)
    except Exception:  # pragma: no cover - environment dependent
        return None


def _comfy_kitchen_meta() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        import comfy_kitchen

        meta["available"] = True
        meta["module"] = getattr(comfy_kitchen, "__file__", None)
        if hasattr(comfy_kitchen, "list_backends"):
            meta["backends"] = {k: str(v) for k, v in comfy_kitchen.list_backends().items()}
    except Exception as exc:  # pragma: no cover - environment dependent
        meta["available"] = False
        meta["error"] = repr(exc)
    return meta


# --- RAVEN side --------------------------------------------------------------


def run_raven_side(args, inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    root = Path(args.raven_root).expanduser()
    if not (root / "projects" / "minimax_h3").is_dir():
        raise SystemExit(f"--raven-root {root} is not a RAVEN checkout")
    sys.path.insert(0, str(root))

    import torch.nn.functional as F
    from projects.minimax_h3.modeling.transformer import causal_model as causal_module
    from projects.minimax_h3.modeling.transformer.causal_model import (
        CausalMiniMaxH3Attention,
    )
    from projects.minimax_h3.modeling.transformer.config import MiniMaxH3DiTArchConfig
    from projects.minimax_h3.modeling.transformer.model import (
        _BF16_DTYPE,
        _apply_qk_norm,
        _apply_rope_qk,
        _modulate_gate,
        _modulate_scale_shift,
        _rope_cos_sin_cache,
        _silu_mul,
        MiniMaxH3DiTBlock,
        MiniMaxH3Rope,
    )
    from utils.naive_cache import NaiveCache

    config = ARCHS[args.arch]
    request = inputs["request"]
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)

    arch = MiniMaxH3DiTArchConfig(
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

    block_state = read_block_state(args.weights, args.block)
    local = strip_block_prefix(block_state, args.block)
    rope_inv_freq = block_state.get("rope.inv_freq")
    if rope_inv_freq is None:
        raise SystemExit("checkpoint has no rope.inv_freq; cannot build RoPE frequencies")

    block = MiniMaxH3DiTBlock(arch, None, prefix=f"blocks.{args.block}")
    block = block.to(device=device, dtype=dtype)
    target = block.state_dict()
    missing = sorted(set(target) - set(local))
    unexpected = sorted(set(local) - set(target))
    if missing or unexpected:
        raise SystemExit(
            f"block {args.block} does not fit RAVEN's block: missing={missing[:4]} "
            f"unexpected={unexpected[:4]}"
        )
    block.load_state_dict({k: v.to(device=device, dtype=target[k].dtype)
                           for k, v in local.items()}, strict=True)
    block.requires_grad_(False).eval()
    attn = block.attn

    rope = MiniMaxH3Rope(config["rope_inv_freq_len"]).to(device)
    rope.inv_freq.data = rope_inv_freq.to(device=device, dtype=torch.float32)

    rows, kv_rows = request["rows"], request["kv_rows"]
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]

    def to_device(name, cast=True):
        tensor = inputs[name].to(device)
        return tensor.to(dtype) if cast else tensor

    x = to_device("x")
    positions_q = inputs["positions"][-rows:].to(device)

    stages: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    freqs = rope(positions_q.unsqueeze(0))
    stages["rope_freqs"] = _store(freqs)
    sources["rope_freqs"] = "projects.minimax_h3...model.MiniMaxH3Rope.forward"
    cos_sin_cache = _rope_cos_sin_cache(freqs, dtype=dtype)
    positions_index = torch.arange(rows, device=device, dtype=torch.long)

    qkv, _ = attn.qkv_proj(x)
    q_raw, k_raw, v_raw = qkv.split(attn.local_inner_dim, dim=-1)
    q_raw = q_raw.view(rows, heads, head_dim)
    k_raw = k_raw.view(rows, heads, head_dim)
    v_raw = v_raw.view(rows, heads, head_dim)
    stages["qkv_gemm.q"] = _store(q_raw)
    stages["qkv_gemm.k"] = _store(k_raw)
    stages["qkv_gemm.v"] = _store(v_raw)
    for name in ("q", "k", "v"):
        sources[f"qkv_gemm.{name}"] = "MiniMaxH3Attention.qkv_proj (MergedColumnParallelLinear)"

    q_n, k_n = _apply_qk_norm(q_raw.clone(), k_raw.clone(), attn.q_norm, attn.k_norm,
                              attn.head_dim)
    stages["qk_norm.q"] = _store(q_n)
    stages["qk_norm.k"] = _store(k_n)
    sources["qk_norm.q"] = "model._apply_qk_norm (eager branch)"
    sources["qk_norm.k"] = "model._apply_qk_norm (eager branch)"

    q_nr, k_nr = _apply_rope_qk(q_n.clone(), k_n.clone(), cos_sin_cache, positions_index)
    stages["qk_norm_rope.q"] = _store(q_nr)
    stages["qk_norm_rope.k"] = _store(k_nr)
    sources["qk_norm_rope.q"] = "model._apply_qk_norm + model._apply_rope_qk (eager)"
    sources["qk_norm_rope.k"] = "model._apply_qk_norm + model._apply_rope_qk (eager)"

    attn_out = _raven_attention(causal_module, to_device("attn_q"), to_device("attn_k"),
                                to_device("attn_v"), rows, kv_rows, attn.softmax_scale,
                                device)
    stages["attention.out"] = _store(attn_out)
    sources["attention.out"] = "causal_model._CAUSAL_FLASH_ATTENTION (utils.flash_attn)"
    proj_out, _ = attn.out_proj(attn_out)
    stages["attention.proj_out"] = _store(proj_out)
    sources["attention.proj_out"] = "MiniMaxH3Attention.out_proj (RowParallelLinear)"

    adaln = block.adaln_proj
    projected = adaln.project_local(to_device("adaln_input"))
    for name, tensor in zip(ADALN_NAMES, adaln.split_output(projected)):
        stages[f"adaln_gemm.{name}"] = _store(tensor)
        sources[f"adaln_gemm.{name}"] = "MiniMaxH3AdalnProj.project_local + split_output"
    native_input = F.silu(inputs["t_emb"].to(device)).to(_BF16_DTYPE)
    for name, tensor in zip(ADALN_NAMES, adaln(native_input)):
        stages[f"adaln_native.{name}"] = _store(tensor)
        sources[f"adaln_native.{name}"] = "silu(t_emb) in fp32 -> bf16 -> MiniMaxH3AdalnProj"

    mod_rows = inputs["mod_rows"].to(device)
    scale_shift = _modulate_scale_shift(to_device("mod_x").clone(), to_device("shift"),
                                        to_device("scale"), mod_rows, dtype=_BF16_DTYPE)
    stages["modulate.scale_shift"] = _store(scale_shift)
    sources["modulate.scale_shift"] = "model._modulate_scale_shift (eager branch)"
    gated = _modulate_gate(to_device("mod_x").clone(), to_device("gate"),
                           to_device("mod_other"), mod_rows, dtype=_BF16_DTYPE)
    stages["modulate.gate"] = _store(gated)
    sources["modulate.gate"] = "model._modulate_gate (eager branch)"

    mlp_x = to_device("mlp_x")
    fc1, _ = block.mlp.fc1(mlp_x)
    stages["mlp.fc1"] = _store(fc1)
    sources["mlp.fc1"] = "MiniMaxH3MLP.fc1 (MergedColumnParallelLinear)"
    stages["mlp.swiglu"] = _store(_silu_mul(fc1.clone(), reuse_input=False))
    sources["mlp.swiglu"] = "model._silu_mul (eager branch)"
    stages["mlp.out"] = _store(block.mlp(mlp_x))
    sources["mlp.out"] = "MiniMaxH3MLP.forward"

    # self-check: staged replay vs the causal attention module's own forward
    selfcheck = _raven_selfcheck(
        CausalMiniMaxH3Attention, NaiveCache, causal_module, arch, attn, x,
        cos_sin_cache, positions_index, rows, heads, head_dim, device, dtype,
        {k[len("attn."):]: v for k, v in local.items() if k.startswith("attn.")},
    )

    meta = {
        "side": "raven",
        "block": int(args.block),
        "arch": args.arch,
        "dtype": args.dtype,
        "device": str(device),
        "runtime": runtime_meta(),
        "attention_backend": _raven_backend_name(),
        "fused_qknorm_rope": bool(getattr(attn, "_use_fused_qknorm_rope", False)),
        "softmax_scale": float(attn.softmax_scale),
        "weights": str(args.weights),
        "loaded_keys": sorted(local),
        "selfcheck.attention": selfcheck,
        "request": request,
    }
    return {"version": OPERATOR_DUMP_SCHEMA["version"], "side": "raven",
            "block": int(args.block), "stages": stages, "sources": sources, "meta": meta}


def _raven_attention(causal_module, q, k, v, rows, kv_rows, softmax_scale, device):
    import torch

    return causal_module._CAUSAL_FLASH_ATTENTION(
        q, k, v,
        q_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        k_lens=torch.tensor([kv_rows], dtype=torch.int32, device=device),
        softmax_scale=softmax_scale,
        causal=False,
    ).reshape(rows, -1)


def _raven_selfcheck(attention_cls, cache_cls, causal_module, arch, attn, x,
                     cos_sin_cache, positions_index, rows, heads, head_dim,
                     device, dtype, attn_state):
    """Replay vs ``CausalMiniMaxH3Attention.forward`` with an empty cache."""
    import torch

    from projects.minimax_h3.modeling.transformer.model import (
        _apply_qk_norm,
        _apply_rope_qk,
    )

    try:
        module = attention_cls(arch, None, prefix="probe.attn", layer_idx=0)
        module = module.to(device=device, dtype=dtype)
        target = module.state_dict()
        module.load_state_dict({k: v.to(device=device, dtype=target[k].dtype)
                                for k, v in attn_state.items()}, strict=True)
        module.requires_grad_(False).eval()

        cache = cache_cls(1, 1, sink=[1], window_size=[None])
        lens = torch.tensor([rows], dtype=torch.int32, device=device)
        real = module(
            x, rope_cache=(cos_sin_cache, positions_index),
            sample_lens=lens, key_value_lens=lens,
            past_key_values=cache, update_past_key_values=False,
            packed_query_indexes=None, packed_past_key_value_indexes=None,
        )
        qkv, _ = attn.qkv_proj(x)
        q, k, v = qkv.split(attn.local_inner_dim, dim=-1)
        q = q.view(rows, heads, head_dim)
        k = k.view(rows, heads, head_dim)
        v = v.view(rows, heads, head_dim)
        q, k = _apply_qk_norm(q, k, attn.q_norm, attn.k_norm, attn.head_dim)
        q, k = _apply_rope_qk(q, k, cos_sin_cache, positions_index)
        replay = _raven_attention(causal_module, q, k, v, rows, rows,
                                  attn.softmax_scale, device)
        replay, _ = attn.out_proj(replay)
        return tensor_metrics(_store(replay), _store(real))
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": f"{type(exc).__name__}: {exc}"}


def _raven_backend_name() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        import utils.flash_attn as flash

        meta["flash_attn_3"] = bool(getattr(flash, "FLASH_ATTN_3_AVAILABLE", False))
        meta["flash_attn_2"] = bool(getattr(flash, "FLASH_ATTN_2_AVAILABLE", False))
        meta["fallback"] = not (meta["flash_attn_3"] or meta["flash_attn_2"])
    except Exception as exc:  # pragma: no cover
        meta["error"] = repr(exc)
    return meta


# --- comparison --------------------------------------------------------------


def compare_operator_dumps(report: Report, comfy: Dict[str, Any], raven: Dict[str, Any],
                           *, rel_l2_max: float, cos_min: float) -> None:
    """Stage-by-stage metrics, with each stage its own gate."""
    for side, dump in (("comfy", comfy), ("raven", raven)):
        if int(dump.get("version", -1)) != OPERATOR_DUMP_SCHEMA["version"]:
            report.add(Check(f"{side}.schema_version", False,
                             {"got": dump.get("version")}))
            return

    report.meta["comfy"] = comfy["meta"]
    report.meta["raven"] = raven["meta"]
    report.meta["sources"] = {"comfy": comfy["sources"], "raven": raven["sources"]}
    report.meta["gate"] = {"stage_rel_l2_max": rel_l2_max, "cosine_min": cos_min}

    report.add(Check("setup.same_block",
                     int(comfy["block"]) == int(raven["block"]),
                     {"comfy": comfy["block"], "raven": raven["block"]}))
    report.add(Check("setup.same_dtype",
                     comfy["meta"]["dtype"] == raven["meta"]["dtype"],
                     {"comfy": comfy["meta"]["dtype"], "raven": raven["meta"]["dtype"]}))
    report.add(Check("setup.same_request",
                     comfy["meta"]["request"] == raven["meta"]["request"],
                     {"rows": comfy["meta"]["request"].get("rows"),
                      "kv_rows": comfy["meta"]["request"].get("kv_rows")}))

    for side, dump in (("comfy", comfy), ("raven", raven)):
        selfcheck = dump["meta"].get("selfcheck.attention", {})
        if "error" in selfcheck:
            report.skip(f"{side}.selfcheck_attention", selfcheck["error"])
        else:
            report.add(Check(f"{side}.selfcheck_attention",
                             metrics_pass(selfcheck, rel_l2_max=1e-6, cos_min=1.0 - 1e-9),
                             _metric_detail(selfcheck)))

    stages = [name for name in STAGE_ORDER
              if name in comfy["stages"] and name in raven["stages"]]
    missing = sorted((set(comfy["stages"]) | set(raven["stages"])) - set(stages))
    if missing:
        report.skip("stages.missing_on_one_side", ", ".join(missing))

    for name in stages:
        metrics = tensor_metrics(comfy["stages"][name], raven["stages"][name])
        report.metrics.append(dict(metrics, stage=name,
                                   comfy_source=comfy["sources"].get(name),
                                   raven_source=raven["sources"].get(name)))
        report.add(Check(
            f"stage.{name}",
            metrics_pass(metrics, rel_l2_max=rel_l2_max, cos_min=cos_min),
            dict(_metric_detail(metrics), rel_l2_max=rel_l2_max, cos_min=cos_min),
        ))


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("both", "inputs", "comfy", "raven", "compare"),
                        default="both")
    parser.add_argument("--weights", default=None,
                        help="runtime-layout checkpoint; only one block is read")
    parser.add_argument("--block", type=int, default=49,
                        help="which DiT block to audit (0 for the shallowest)")
    parser.add_argument("--arch", choices=tuple(ARCHS), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--emit-inputs", default=None)
    parser.add_argument("--emit-dump", default=None)
    parser.add_argument("--comfy-dump", default=None)
    parser.add_argument("--raven-dump", default=None)
    parser.add_argument("--work-dir", default=None,
                        help="where --side both puts its intermediates (default: temp)")
    parser.add_argument("--json", default=None)
    parser.add_argument("--comfyui-path", default=None)
    parser.add_argument("--raven-root", default=os.environ.get("RAVEN_ROOT", "/root/Jarvis"))
    parser.add_argument("--python", default=sys.executable,
                        help="default interpreter for both side subprocesses")
    parser.add_argument("--comfy-python", default=None,
                        help="interpreter for the ComfyUI side (default: --python)")
    parser.add_argument("--raven-python", default=None,
                        help="interpreter for the RAVEN side (default: --python)")
    parser.add_argument("--rel-l2-max", type=float, default=STAGE_REL_L2_MAX)
    parser.add_argument("--cos-min", type=float, default=COSINE_MIN)

    backends = parser.add_argument_group("kernel pinning (both sides, recorded either way)")
    backends.add_argument("--attention-backend",
                          choices=("pytorch", "split", "quad", "sage", "flash"),
                          default=None, help="Comfy side: pin optimized_attention")
    backends.add_argument("--disable-fused-kernels", action="store_true",
                          help="Comfy side: disable comfy-kitchen triton/cuda backends")
    backends.add_argument("--raven-attention", choices=("auto", "sdpa"), default="auto",
                          help="RAVEN side: sdpa forces the packed SDPA fallback")

    grid = parser.add_argument_group("input geometry")
    grid.add_argument("--frames", type=int, default=39)
    grid.add_argument("--width", type=int, default=512)
    grid.add_argument("--height", type=int, default=288)
    grid.add_argument("--text-len", type=int, default=128)
    grid.add_argument("--rows", type=int, default=None,
                      help="query rows (default: chunk 0's row count)")
    grid.add_argument("--kv-rows", type=int, default=None,
                      help="key/value rows (default: text + chunk 0)")
    grid.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def _resolve_inputs(args) -> Dict[str, Any]:
    if args.inputs:
        return load_operator_inputs(args.inputs)
    return build_operator_inputs(
        arch=args.arch, frames=args.frames, width=args.width, height=args.height,
        text_len=args.text_len, seed=args.seed, rows=args.rows, kv_rows=args.kv_rows,
    )


def _run_side(args, side: str) -> Dict[str, Any]:
    import torch

    if not args.weights:
        raise SystemExit(f"--side {side} needs --weights")
    if not args.inputs:
        raise SystemExit(f"--side {side} needs --inputs (both sides must read the same file)")
    inputs = load_operator_inputs(args.inputs)
    if inputs["request"]["arch"] != args.arch:
        raise SystemExit(
            f"inputs were built for arch {inputs['request']['arch']!r}, "
            f"this run asks for {args.arch!r}"
        )
    if side == "comfy":
        comfyui = find_comfyui(args.comfyui_path)
        if str(comfyui) not in sys.path:
            sys.path.insert(0, str(comfyui))
        print(f"ComfyUI: {comfyui}")
        # before anything imports comfy.ldm: the backend is bound at import time
        attention_meta = configure_attention(args.attention_backend,
                                             args.disable_fused_kernels)
        with torch.no_grad():
            dump = run_comfy_side(args, inputs)
        dump["meta"]["attention_pinning"] = attention_meta
    else:
        print(f"RAVEN: {args.raven_root}")
        with torch.no_grad():
            dump = run_raven_side(args, inputs)
    if args.emit_dump:
        Path(args.emit_dump).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dump, args.emit_dump)
        print(f"dump: {args.emit_dump}")
    selfcheck = dump["meta"].get("selfcheck.attention", {})
    print(f"stages: {len(dump['stages'])}  selfcheck: {selfcheck}")
    return dump


# --- subprocess isolation ----------------------------------------------------
#
# ComfyUI and RAVEN both ship a top-level ``utils`` package (and RAVEN adds
# ``common``/``projects``, ComfyUI adds ``comfy``/``folder_paths``). A single
# inherited PYTHONPATH entry pointing at the wrong root is enough to make
# ``from utils.flash_attn import FlashAttention`` resolve inside ComfyUI and die
# with ModuleNotFoundError -- observed on vr with ``--side both``.
#
# Ordering does not fix it: RAVEN's ``utils`` has no ``__init__.py``, so it is a
# namespace portion, and the import machinery keeps scanning the rest of the
# path for a *regular* package -- ComfyUI's, which then wins no matter how early
# RAVEN sits on sys.path. The only reliable fix is to keep the other checkout
# out of the child's environment entirely. So each side gets an *explicit*
# PYTHONPATH: its own root first, this project second, and every inherited entry
# that could answer for one of those names dropped.

#: Top-level names either checkout claims. An inherited path entry offering one
#: of them is ambiguous by construction and is not carried into a side process.
CONFLICTING_TOP_LEVEL: Tuple[str, ...] = (
    "utils", "common", "projects", "comfy", "comfy_extras", "folder_paths",
)


@dataclass
class SpawnPlan:
    """Exactly how one side is launched: interpreter, cwd, path, command."""

    side: str
    interpreter: str
    cwd: str
    pythonpath: List[str]
    dropped: List[Dict[str, str]]
    cmd: List[str]
    env: Dict[str, str]

    def describe(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "interpreter": self.interpreter,
            "cwd": self.cwd,
            "pythonpath": list(self.pythonpath),
            "dropped_pythonpath": list(self.dropped),
        }


def provides_conflicting_package(entry: str) -> Optional[str]:
    """The ambiguous top-level name an entry offers, or ``None``."""
    path = Path(entry)
    for name in CONFLICTING_TOP_LEVEL:
        if (path / name).is_dir() or (path / f"{name}.py").is_file():
            return name
    return None


def side_pythonpath(
    side: str,
    *,
    own_root: str,
    project_root: str = str(PROJECT_ROOT),
    inherited: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """``(entries, dropped)`` for one side's PYTHONPATH.

    ``own_root`` comes first so its ``utils`` wins, then this project (which has
    no conflicting top-level names). Inherited entries are kept only when they
    cannot answer for one of :data:`CONFLICTING_TOP_LEVEL` -- a venv or an extra
    library path survives, the other checkout does not.
    """
    roots = [str(Path(own_root).resolve()), str(Path(project_root).resolve())]
    entries = list(roots)
    dropped: List[Dict[str, str]] = []
    raw = os.environ.get("PYTHONPATH", "") if inherited is None else os.pathsep.join(inherited)
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        resolved = str(Path(entry).resolve()) if Path(entry).exists() else entry
        if resolved in entries:
            continue
        conflict = provides_conflicting_package(entry)
        if conflict is not None:
            dropped.append({"entry": entry, "provides": conflict, "side": side})
            continue
        entries.append(resolved)
    return entries, dropped


def build_spawn_plan(args, side: str, inputs_path: str, dump_path: str, *,
                     comfy_root: Optional[str] = None,
                     raven_root: Optional[str] = None,
                     inherited: Optional[Sequence[str]] = None,
                     script: Optional[str] = None,
                     extra_args: Optional[Sequence[str]] = None) -> SpawnPlan:
    """Everything needed to run one side in its own, unambiguous environment.

    Paths are absolutised because the child runs with ``cwd`` set to its own
    checkout: a relative ``--weights`` would otherwise resolve against the wrong
    directory. ``script`` lets a sibling probe reuse this isolation (the
    embedding probe does); it defaults to this file.
    """
    if side == "comfy":
        own_root = comfy_root
        interpreter = args.comfy_python or args.python
    elif side == "raven":
        own_root = raven_root
        interpreter = args.raven_python or args.python
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown side {side!r}")
    if not own_root:
        raise SystemExit(f"--side both needs the {side} root to launch that side")

    entries, dropped = side_pythonpath(side, own_root=own_root, inherited=inherited)

    def absolute(value: Any) -> str:
        return str(Path(str(value)).expanduser().resolve())

    cmd = [interpreter, str(Path(script).resolve() if script else Path(__file__).resolve()),
           "--side", side, "--weights", absolute(args.weights),
           "--arch", args.arch, "--device", args.device, "--dtype", args.dtype,
           "--inputs", absolute(inputs_path), "--emit-dump", absolute(dump_path)]
    if getattr(args, "block", None) is not None and script is None:
        cmd += ["--block", str(args.block)]
    cmd += list(extra_args or ())

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if side == "comfy":
        cmd += ["--comfyui-path", absolute(own_root)]
        # optional kernel pinning: a sibling probe reusing this helper may not
        # expose those flags at all
        if getattr(args, "attention_backend", None):
            cmd += ["--attention-backend", args.attention_backend]
        if getattr(args, "disable_fused_kernels", False):
            cmd += ["--disable-fused-kernels"]
        env["COMFYUI_PATH"] = absolute(own_root)
        # nothing on this side may reach for RAVEN
        env.pop("RAVEN_ROOT", None)
    else:
        cmd += ["--raven-root", absolute(own_root)]
        if getattr(args, "raven_attention", None):
            cmd += ["--raven-attention", args.raven_attention]
        env["RAVEN_ROOT"] = absolute(own_root)
        # ... and nothing on this side may reach for ComfyUI
        env.pop("COMFYUI_PATH", None)
        env.pop("COMFYUI_UPSTREAM_PATH", None)

    return SpawnPlan(side=side, interpreter=interpreter, cwd=absolute(own_root),
                     pythonpath=entries, dropped=dropped, cmd=cmd, env=env)


def _spawn_side(plan: SpawnPlan) -> None:
    print(f"\n[{plan.side}] interpreter: {plan.interpreter}")
    print(f"[{plan.side}] cwd:         {plan.cwd}")
    print(f"[{plan.side}] PYTHONPATH:  {os.pathsep.join(plan.pythonpath)}")
    for item in plan.dropped:
        print(f"[{plan.side}] dropped {item['entry']} (provides {item['provides']})")
    print(f"$ {' '.join(plan.cmd)}")
    done = subprocess.run(plan.cmd, capture_output=True, text=True,
                          cwd=plan.cwd, env=plan.env)
    sys.stdout.write(done.stdout)
    if done.returncode != 0:
        sys.stderr.write(done.stderr)
        raise SystemExit(f"{plan.side} side failed with exit code {done.returncode}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # ``utils.flash_attn`` binds its backend at import time, so this has to
    # happen before the RAVEN side imports anything.
    if args.raven_attention == "sdpa":
        os.environ["FLASH_ATTN_3_AVAILABLE"] = "0"
        os.environ["FLASH_ATTN_2_AVAILABLE"] = "0"

    if args.side == "inputs":
        inputs = _resolve_inputs(args)
        if not args.emit_inputs:
            raise SystemExit("--side inputs needs --emit-inputs")
        save_operator_inputs(args.emit_inputs, inputs)
        print(f"inputs: {args.emit_inputs}")
        print(json.dumps(inputs["request"], indent=2))
        return 0

    if args.side in ("comfy", "raven"):
        _run_side(args, args.side)
        return 0

    import torch

    report = Report(mode=f"operator/block{args.block}", device=args.device)
    report.meta["operator_dump_schema"] = OPERATOR_DUMP_SCHEMA
    report.meta["operator_inputs_schema"] = OPERATOR_INPUTS_SCHEMA

    if args.side == "compare":
        if not args.comfy_dump or not args.raven_dump:
            raise SystemExit("--side compare needs --comfy-dump and --raven-dump")
        comfy = torch.load(args.comfy_dump, map_location="cpu", weights_only=False)
        raven = torch.load(args.raven_dump, map_location="cpu", weights_only=False)
    else:
        work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(
            prefix="raven_operator_parity_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = args.inputs or str(work_dir / "operator_inputs.pt")
        if not args.inputs:
            save_operator_inputs(inputs_path, _resolve_inputs(args))
            print(f"inputs: {inputs_path}")
        comfy_path = args.comfy_dump or str(work_dir / "comfy_ops.pt")
        raven_path = args.raven_dump or str(work_dir / "raven_ops.pt")

        # Resolve both roots here, in the parent: each child needs the *other*
        # one kept off its PYTHONPATH, and a missing checkout should fail before
        # anything is built rather than inside an import.
        comfy_root = str(find_comfyui(args.comfyui_path))
        raven_root = Path(args.raven_root).expanduser()
        if not (raven_root / "projects" / "minimax_h3").is_dir():
            raise SystemExit(f"--raven-root {raven_root} is not a RAVEN checkout")

        plans = [build_spawn_plan(args, side, inputs_path,
                                  comfy_path if side == "comfy" else raven_path,
                                  comfy_root=comfy_root, raven_root=str(raven_root))
                 for side in ("comfy", "raven")]
        report.meta["spawn"] = {plan.side: plan.describe() for plan in plans}
        # one block per process, one process at a time: never two models resident
        for plan in plans:
            _spawn_side(plan)
        comfy = torch.load(comfy_path, map_location="cpu", weights_only=False)
        raven = torch.load(raven_path, map_location="cpu", weights_only=False)
        report.meta["work_dir"] = str(work_dir)

    print()
    compare_operator_dumps(report, comfy, raven,
                           rel_l2_max=args.rel_l2_max, cos_min=args.cos_min)

    gating = [c for c in report.checks if c.gate]
    print(f"\nGATE {'PASS' if report.passed else 'FAIL'}: "
          f"{sum(1 for c in gating if c.passed)}/{len(gating)} gating checks, "
          f"{report.diagnostics_failed} diagnostic difference(s), "
          f"{len(report.skipped)} skipped")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2, default=str))
        print(f"report: {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
