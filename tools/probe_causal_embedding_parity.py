#!/usr/bin/env python3
"""Embedding-path audit: everything upstream of the DiT block stack.

Why this exists
---------------
The 50-block exact rerun still shows ~1.0-1.17% relative L2 on **block 0's**
Q/K/V, while ``tools/probe_causal_operator_parity.py`` has already shown that
inside a block the QKV GEMM, q/k RMSNorm, RoPE, the AdaLN GEMM, the modulation
and the MLP are exact. Q/K/V at block 0 are a linear function of that block's
*input*, so the disagreement is upstream: the packed decoder input, the AdaLN
vectors that modulate it, or both.

This probe audits exactly that region, on both implementations, from the same
shared inputs:

======================================  ==========================================
what                                    where it comes from
======================================  ==========================================
video patch projection                  ``video_patch_proj`` input / raw fp32 out / bf16 cast
audio patch projection                  ``audio_patch_proj`` input / raw fp32 out / bf16 cast
time embedder                           frequency embedding (``proj_in`` input),
                                        ``proj_in`` out, SiLU (``proj_out`` input),
                                        ``t_emb``, and the AdaLN input the block
                                        actually receives (Comfy SiLUs a bf16
                                        ``t_emb``; RAVEN SiLUs the fp32 one)
text conditioning                       ``condition_proj``, each of the two real
                                        refiner blocks, ``final_norm``
packed decoder input                    block 0's input rows, for the text prefill
                                        and for chunk 0
block 0 pre-attention                   ``norm1`` output and the modulated ``h``
                                        the attention actually consumes
======================================  ==========================================

**Nothing here is replayed.** Both sides run their real production entry points
-- Comfy ``RavenCausalMiniMaxH3Model.prefill_text`` / ``forward_chunk``, RAVEN
``CausalMiniMaxH3Base._text_cache_fill`` / ``_chunk_forward`` through
``MiniMaxH3X0Model`` -- and every number is captured with a **forward hook** on
the module that produced it. There is no staged reimplementation to drift, so
there is no self-check to fake: the per-side ``selfcheck`` block instead records
that the two production forwards ran and how many taps fired.

The models are built with ``num_layers=1``. Everything this probe measures is
upstream of block 1, so one block is all that has to exist -- and only the
subtrees it needs (``video_patch_proj``, ``audio_patch_proj``, ``condition_proj``,
``time_embedder``, ``rope``, ``token_refiner``, ``blocks.0``, ``final_layer``)
are read out of the checkpoint, key by key, through ``safe_open``. A 66 GB file
costs ~3 GB.

``--comfy-fp32-island {declared,cast}`` exists because it is a live hypothesis:
``probe_causal_parity --mode real`` casts *every* float tensor to ``--dtype``
before loading, and the H3 checkpoint has an fp32 island (patch projections,
time embedder, output heads). Loading bf16-rounded values into those fp32
parameters is not what production does, and it moves exactly the tensors this
probe measures. ``declared`` (the default) keeps each parameter's own dtype;
``cast`` reproduces the old behaviour so the two can be compared directly.

Attention backend
-----------------
The causal lane now dispatches **FA3 -> FA2 -> SDPA**, the same chain RAVEN's
wrapper uses, and its SDPA step disables ``allow_fp16_bf16_reduction_math_sdp``
for that one call and restores it in ``finally``. So the two sides can now be on
the *same* kernel -- and when they are, every shared stage here is expected to
be bit-identical.

When they are **not** on the same kernel (FA3 on one side, SDPA on the other),
every stage from the first attention call onward carries that kernel's float
error. This probe records both sides' backends in ``meta.attention_backends``,
classifies those stages as ``kernel_float_error``, and reports them **without
gating**: two kernels disagreeing on the last bits of a bf16 reduction is
arithmetic, not a logic difference, and calling it one would be wrong. Stages
*before* the first attention call -- embeddings, projections, norms, and the
SDPA seam's own q/k/v inputs -- stay gated either way, because nothing about
the kernel can excuse those.

To compare logic rather than kernels, export ``FLASH_ATTN_3_AVAILABLE=0`` and
``FLASH_ATTN_2_AVAILABLE=0`` in *both* processes (the causal lane honours the
same switches RAVEN does), or pass ``--math-sdp-reduced-precision off``.

The ``env/sdpa_control`` stage stays as a **process-level** control: a canned
SDPA run outside the lane. It is diagnostic only now -- the lane neutralises the
process flag around its own call, so the control can differ while every lane
stage is exact.

Process model: same two-process isolation as the operator probe (ComfyUI and
RAVEN both own a top-level ``utils``; RAVEN's is a namespace package and loses
to ComfyUI's regular one wherever both are importable), reusing its
:func:`build_spawn_plan`.

Usage::

    # one shot, both sides
    python tools/probe_causal_embedding_parity.py \\
        --weights h3_bf16.safetensors --inputs m2_full_inputs.pt \\
        --raven-root /root/Jarvis --comfyui-path /path/to/ComfyUI \\
        --device cuda --dtype bf16 --work-dir $OUT/emb --json $OUT/embedding.json

    # or side by side
    python tools/probe_causal_embedding_parity.py --side comfy --weights ... \\
        --inputs m2_full_inputs.pt --emit-dump comfy_emb.pt
    python tools/probe_causal_embedding_parity.py --side raven --weights ... \\
        --inputs m2_full_inputs.pt --raven-root /root/Jarvis --emit-dump raven_emb.pt
    python tools/probe_causal_embedding_parity.py --side compare \\
        --comfy-dump comfy_emb.pt --raven-dump raven_emb.pt --json embedding.json

Status: runs on CPU with the ``tiny`` architecture against a real RAVEN
checkout. The full BF16 CUDA run is a vr-* job and has not happened here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
    COSINE_MIN,
    Report,
    _metric_detail,
    find_comfyui,
    load_inputs,
    load_state_dict_file,
    metrics_pass,
    runtime_meta,
    tensor_metrics,
)
from probe_causal_operator_parity import (  # noqa: E402
    _store,
    _torch_dtype,
    build_spawn_plan,
    _spawn_side,
)

EMBEDDING_DUMP_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "side": "'comfy' or 'raven'",
    "stages": "dict '<phase>/<module path>.<in|out>' -> float32 CPU tensor",
    "dtypes": "dict stage -> the dtype the tensor actually had before storing",
    "sources": "dict stage -> the module or derivation that produced it",
    "meta": "runtime, weight placement, tap counts, request",
}

#: Subtrees the probe reads. Everything else in the checkpoint stays on disk.
REQUIRED_PREFIXES: Tuple[str, ...] = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "condition_proj.",
    "time_embedder.",
    "token_refiner.",
    "blocks.0.",
    "final_layer.",
)
REQUIRED_KEYS: Tuple[str, ...] = ("rope.inv_freq",)

#: Modules tapped on both sides. The paths are identical in both module trees,
#: which is what makes the two dumps comparable stage for stage.
TAP_SPECS: Tuple[Tuple[str, str], ...] = (
    ("video_patch_proj", "in_out"),
    ("audio_patch_proj", "in_out"),
    ("condition_proj", "out"),
    ("time_embedder", "out"),
    ("time_embedder.proj_in", "in_out"),
    ("time_embedder.proj_out", "in_out"),
    ("token_refiner.blocks.0", "out"),
    ("token_refiner.blocks.1", "out"),
    ("token_refiner.final_norm", "in_out"),
    ("blocks.0", "in"),
    ("blocks.0.norm1", "out"),
    ("blocks.0.adaln_proj.linear", "in"),
    ("blocks.0.attn.qkv_proj", "in"),
)


def refiner_tap_specs(num_blocks: int) -> Tuple[Tuple[str, str], ...]:
    """Per-refiner-block taps, on modules that exist and are *called* on both sides.

    Neither side's refiner is a single tappable call:

    * RAVEN runs ``MiniMaxH3TokenRefinerBlock.forward`` --
      ``x + attn(norm1(x))`` then ``x + mlp(norm2(x))``;
    * the causal lane replays that same body over the official modules
      (``_causal_refine_text``), so ``block.__call__`` and ``attn.__call__``
      never fire there.

    What *both* invoke is the leaf modules, so every stage below is keyed off a
    leaf. The block boundaries come out of the neighbours' **inputs**:
    ``norm1.in`` is the block input, ``norm2.in`` is the attention residual, and
    the block output is the next consumer's input (``blocks.N+1.norm1.in``, or
    ``final_norm.in`` for the last block).

    ``mlp.fc2.in`` is the SwiGLU output on both sides: comfy's
    ``linear_input_act`` calls ``fc2(INPUT_ACT_EAGER['swiglu'](x))`` on a
    non-INT8 weight, RAVEN's ``MiniMaxH3MLP.forward`` calls ``fc2(_silu_mul(x))``.
    """
    specs: List[Tuple[str, str]] = []
    for index in range(int(num_blocks)):
        base = f"token_refiner.blocks.{index}"
        specs += [
            (f"{base}.norm1", "in_out"),          # block input, norm1 output
            (f"{base}.attn.qkv_proj", "out"),     # fused QKV, split below
            (f"{base}.attn.q_norm", "out"),
            (f"{base}.attn.k_norm", "out"),
            (f"{base}.attn.out_proj", "in_out"),  # in == SDPA out, reshaped
            (f"{base}.norm2", "in_out"),          # attention residual, norm2 output
            (f"{base}.mlp", "out"),
            (f"{base}.mlp.fc1", "out"),
            (f"{base}.mlp.fc2", "in"),            # the SwiGLU output
        ]
    return tuple(specs)


#: Execution order inside one refiner block, for "which stage moved first".
REFINER_STEP_ORDER: Tuple[str, ...] = (
    "norm1.in",
    "norm1.out",
    "attn.qkv_proj.out",
    "attn.qkv_proj.out.q",
    "attn.qkv_proj.out.k",
    "attn.qkv_proj.out.v",
    "attn.q_norm.out",
    "attn.k_norm.out",
    "attn.sdpa.q",
    "attn.sdpa.k",
    "attn.sdpa.v",
    "attn.sdpa.out",
    "attn.out_proj.in",
    "attn.out_proj.out",
    "norm2.in",
    "norm2.out",
    "mlp.fc1.out",
    "mlp.fc2.in",
    "mlp.out",
)

#: Stage-level gate. Block 0's Q/K/V sit at ~1% today, so an upstream stage over
#: this budget is a candidate cause rather than noise.
STAGE_REL_L2_MAX = 0.005

#: What to say when a specific stage is the one that moved. Reported, never
#: applied: this probe does not touch runtime code.
FIX_SUGGESTIONS: Dict[str, str] = {
    "attn.sdpa": (
        "The divergence starts at the SDPA call itself, with bit-identical q/k/v "
        "going in, so nothing upstream of attention is responsible. First check "
        "env.attention_backend_matches: RAVEN dispatches FA3 -> FA2 -> SDPA and "
        "the causal lane now reproduces that chain, so the two sides can be on "
        "different kernels (FA3 vs SDPA). Different kernels have different float "
        "error on identical inputs -- that is arithmetic, not logic, and this "
        "probe classifies it as kernel_float_error rather than gating on it. To "
        "compare the *logic*, pin both sides to one kernel with "
        "FLASH_ATTN_3_AVAILABLE=0 / FLASH_ATTN_2_AVAILABLE=0 (both processes) "
        "and re-run: every attention stage should then be bit-identical. If it "
        "is not, and the backends agree, that is a real difference. "
        "(The older cause -- comfy.model_management enabling "
        "allow_fp16_bf16_reduction_math_sdp process-wide -- is handled inside the "
        "lane's SDPA fallback, which disables and restores it per call; "
        "env.math_sdp_reduced_precision_matches still reports the process state.)"
    ),
    "video_patch_proj": (
        "video_patch_proj is an fp32 island in the checkpoint. Check that the "
        "loader keeps it fp32 (comfy declares dtype=torch.float32 for it) and "
        "that the latent reaches it as fp32 rows: comfy patchifies "
        "video_x.to(float32), RAVEN casts the packed rows with .to(fp32) in "
        "_embed. If the probe's own loader cast the weights to bf16, fix the "
        "probe first (--comfy-fp32-island declared)."
    ),
    "audio_patch_proj": (
        "audio_patch_proj is the same fp32 island as video_patch_proj; see that "
        "suggestion."
    ),
    "time_embedder": (
        "time_embedder is fp32 in both implementations and computes the same "
        "closed form (cos before sin), so a difference here is about its *input* "
        "or its weights, never the formula. Measured residual at ~1e-8: the "
        "unique timestep value itself. The causal lane computes 1 - sigma as a "
        "python float while RAVEN's x0 wrapper computes it as a tensor op, and "
        "the two round differently in the last fp32 bit. Harmless at that size; "
        "if bit-parity is wanted, build the timestep with the same tensor "
        "expression on both sides. Anything larger than ~1e-6 here is a "
        "weight-dtype problem instead -- see setup.same_weight_values."
    ),
    "blocks.0.adaln_proj.linear.in": (
        "This is the known AdaLN SiLU placement difference: comfy casts t_emb to "
        "the model dtype and SiLUs in bf16 (AdalnProj.forward with "
        "apply_silu=True), RAVEN SiLUs the fp32 t_emb and casts afterwards "
        "(adaln_input = F.silu(t_emb).to(bf16)). Minimal runtime fix, if parity "
        "is wanted: feed the AdaLN projection silu(t_emb_fp32).to(dtype) instead "
        "of silu(t_emb_bf16) -- i.e. keep the fp32 t_emb around in the causal "
        "forward and apply the SiLU before the cast. It changes nothing else."
    ),
    "condition_proj": (
        "condition_proj is bf16 on both sides; a difference is a text input or "
        "weight-load difference, not an operator difference."
    ),
    "token_refiner": (
        "The refiner runs two real attention blocks. Comfy uses "
        "optimized_attention, RAVEN varlen flash/SDPA -- the same backend "
        "difference the operator probe measures at attention.out. Pin both "
        "backends before drawing a conclusion."
    ),
    "blocks.0.attn.qkv_proj.in": (
        "This is the attention input h itself. Whatever moved it is upstream: "
        "compare blocks.0.in (the packed decoder input) and "
        "blocks.0.adaln_proj.linear.in (the AdaLN vectors) to attribute it."
    ),
}


# --- partial checkpoint read -------------------------------------------------


def read_embedding_state(path: str, *, prefixes: Sequence[str] = REQUIRED_PREFIXES,
                         keys: Sequence[str] = REQUIRED_KEYS) -> Dict[str, Any]:
    """Read only the embedding subtrees (plus block 0 and the head) key by key."""
    wanted_keys = set(keys)

    def wanted(name: str) -> bool:
        return name in wanted_keys or any(name.startswith(p) for p in prefixes)

    if str(path).endswith(".safetensors"):
        from safetensors import safe_open

        state: Dict[str, Any] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if wanted(name):
                    state[name] = handle.get_tensor(name)
    else:
        state = {k: v for k, v in load_state_dict_file(path).items() if wanted(k)}

    missing = [p for p in prefixes if not any(k.startswith(p) for k in state)]
    if missing or any(k not in state for k in wanted_keys):
        raise SystemExit(
            f"{path} is missing the embedding subtree(s): {missing} "
            f"(and keys {[k for k in wanted_keys if k not in state]})"
        )
    return state


def place_weights(state: Dict[str, Any], target: Dict[str, Any], *,
                  fp32_island: str) -> Dict[str, Any]:
    """Cast a checkpoint slice for loading.

    ``declared`` respects each destination parameter's dtype, which is what a
    real loader does and what keeps the checkpoint's fp32 island fp32.
    ``cast`` reproduces the blanket ``.to(--dtype)`` some probe paths used, so
    the effect of degrading that island can be measured rather than argued.
    """
    import torch

    placed: Dict[str, Any] = {}
    for name, value in state.items():
        reference = target.get(name)
        if reference is None:
            placed[name] = value
            continue
        if (fp32_island != "declared" and value.is_floating_point()
                and reference.dtype == torch.float32):
            # round-trip through the compute dtype on purpose, so the cost of a
            # blanket cast is a number rather than an argument
            placed[name] = value.to(dtype=_torch_dtype(fp32_island)).to(torch.float32)
        else:
            placed[name] = value.to(dtype=reference.dtype)
    return placed


# --- taps --------------------------------------------------------------------


class ModuleTap:
    """Forward hooks on the shared module paths, tagged by rollout phase.

    Hooks, not a replay: what lands in the dump is what the production forward
    computed, so there is no second implementation to keep honest.
    """

    def __init__(self) -> None:
        self.stages: Dict[str, Any] = {}
        self.dtypes: Dict[str, str] = {}
        self.sources: Dict[str, str] = {}
        self.phase = "?"
        self.handles: List[Any] = []
        self.fired = 0
        #: which attention body is currently running, for the SDPA seam. The
        #: k_norm tap sets it: both sides call that module right before the
        #: attention call, and the causal lane's own ``site`` label (when
        #: present) takes precedence.
        self.pending_site: Optional[Tuple[str, int]] = None
        self.seam_calls: List[Dict[str, Any]] = []
        self.seam_counts: Dict[Tuple[str, str], int] = {}

    def _first_tensor(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = self._first_tensor(item)
                if found is not None:
                    return found
        return None

    def record(self, name: str, tensor, source: str) -> None:
        import torch

        if not isinstance(tensor, torch.Tensor):
            return
        if name.endswith(".attn.k_norm.out") or name.endswith(".attn.q_norm.out"):
            self.pending_site = _site_from_path(name)
        stage = f"{self.phase}/{name}"
        if stage in self.stages:  # a module called twice in one phase
            stage = f"{stage}#{sum(1 for k in self.stages if k.startswith(stage))}"
        self.stages[stage] = _store(tensor)
        self.dtypes[stage] = str(tensor.dtype)
        self.sources[stage] = source
        self.fired += 1

    def attach(self, model, specs=TAP_SPECS) -> "ModuleTap":
        """Inputs on a *pre*-hook, outputs on a forward hook.

        The distinction is load-bearing: the DiT block adds its attention
        residual into its own input buffer in place (``_mod_gate`` ->
        ``addcmul_``), so a forward hook reading ``args`` would hand back the
        block's input *after* the block already overwrote it. That is not a
        subtle risk -- it showed up as a 6x inflated ``blocks.0.in`` difference
        the first time this probe ran.
        """
        for path, mode in specs:
            module = _resolve_module(model, path)
            if module is None:
                continue
            name = type(module).__name__
            if "in" in mode:
                self.handles.append(module.register_forward_pre_hook(
                    _make_pre_hook(self, path, name)))
            if "out" in mode:
                self.handles.append(module.register_forward_hook(
                    _make_hook(self, path, name)))
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> "ModuleTap":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


class SeamTap:
    """Temporarily wrap a module-level attention callable and record its output.

    The SDPA output *before* the output projection is not a module boundary on
    either side -- it is a plain function called from inside an attention body:

    * causal lane: ``raven_streaming.causal_model.raven_packed_attention``
      (which also carries a ``site`` label, ``('text_refiner', i)`` or
      ``('dit', i)``);
    * RAVEN: the module-level ``_MINIMAX_H3_FLASH_ATTENTION`` singleton used by
      the refiner and ``_CAUSAL_FLASH_ATTENTION`` used by the causal blocks.

    The wrapper delegates to the real callable and stores what it returned, so
    nothing is recomputed; the attribute is restored in ``finally``.
    """

    def __init__(self, tap: "ModuleTap", module: Any, attribute: str,
                 default_kind: str) -> None:
        self.tap = tap
        self.module = module
        self.attribute = attribute
        self.default_kind = default_kind
        self._original = None
        self.calls = 0
        self.wrapped = False

    def __enter__(self) -> "SeamTap":
        self._original = getattr(self.module, self.attribute, None)
        if self._original is None:
            return self
        seam = self

        def traced(q, k, v, *args, **kwargs):
            site = seam.resolve_site(kwargs.get("site"))
            source = f"{self_name(seam._original)} (module-level seam, wrapped)"
            # the seam's *inputs* too: when the outputs differ, the first
            # question is always whether the two calls were handed the same
            # tensors or the same op behaved differently
            base = _sdpa_stage_name(site)[: -len("out")]
            for part, tensor in (("q", q), ("k", k), ("v", v)):
                seam.tap.record(f"{base}{part}", tensor, source + " input")
            out = seam._original(q, k, v, *args, **kwargs)
            seam.calls += 1
            seam.tap.record(_sdpa_stage_name(site), out, source)
            # strides and contiguity decide which SDPA kernel torch picks, so a
            # seam that differs only in layout still differs in output
            seam.tap.seam_calls.append({
                "phase": seam.tap.phase, "seam": seam.attribute, "site": list(site),
                "scale": float(kwargs.get("scale") or kwargs.get("softmax_scale") or 0.0),
                "kwargs": sorted(k for k in kwargs if k != "site"),
                "tensors": {
                    name: {"shape": list(t.shape), "stride": list(t.stride()),
                           "contiguous": bool(t.is_contiguous()), "dtype": str(t.dtype)}
                    for name, t in (("q", q), ("k", k), ("v", v))
                },
                "out": {"shape": list(out.shape), "stride": list(out.stride()),
                        "contiguous": bool(out.is_contiguous()), "dtype": str(out.dtype)},
            })
            return out

        setattr(self.module, self.attribute, traced)
        self.wrapped = True
        return self

    def resolve_site(self, explicit) -> Tuple[str, int]:
        """Which stack this call belongs to, and which block of it.

        Priority: the caller's own label (the causal lane passes one), then the
        last q/k-norm tap *if it belongs to this seam's stack*, then a per-phase
        counter. The middle rule is what stops a DiT call inheriting the
        refiner's index: RAVEN reaches the two stacks through two different
        module-level singletons, so the seam identity already fixes the kind.
        """
        if explicit:
            return (str(explicit[0]), int(explicit[1]))
        pending = self.tap.pending_site
        if pending is not None and pending[0] == self.default_kind:
            return pending
        key = (self.attribute, self.tap.phase)
        index = self.tap.seam_counts.get(key, 0)
        self.tap.seam_counts[key] = index + 1
        return (self.default_kind, index)

    def __exit__(self, *exc) -> None:
        if self._original is not None:
            setattr(self.module, self.attribute, self._original)
            self._original = None


def self_name(obj: Any) -> str:
    return getattr(obj, "__name__", None) or type(obj).__name__


def _sdpa_stage_name(site: Tuple[str, int]) -> str:
    kind, index = site
    if kind == "text_refiner":
        return f"token_refiner.blocks.{int(index)}.attn.sdpa.out"
    return f"{kind}.{int(index)}.attn.sdpa.out"


#: Shape of the canned SDPA control. Small, fixed, and identical on both sides.
SDPA_CONTROL_SHAPE: Tuple[int, int, int] = (64, 2, 16)
SDPA_CONTROL_SEED = 20250820


def sdpa_control(tap: "ModuleTap", device, dtype) -> None:
    """One fixed SDPA call, recorded on both sides, as an environment control.

    Measured on CPU: the *same* ``scaled_dot_product_attention`` on the same
    bf16 tensors returns different values depending on whether ComfyUI has been
    imported into the process (rel_l2 ~4.8e-3, identical torch version, thread
    count, mkldnn flag and matmul precision). Importing a framework changes
    kernel dispatch, and this lane runs its two sides in two different
    processes *by construction* -- one with ComfyUI imported, one with RAVEN.

    So before any stage difference can be blamed on the model code, this canned
    call has to agree. Its inputs come from a fixed seed on CPU and are then
    moved, so both sides start from bit-identical numbers.
    """
    import torch
    import torch.nn.functional as F

    rows, heads, head_dim = SDPA_CONTROL_SHAPE
    generator = torch.Generator().manual_seed(SDPA_CONTROL_SEED)
    q, k, v = (torch.randn(rows, heads, head_dim, generator=generator).to(
        device=device, dtype=dtype) for _ in range(3))
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=head_dim ** -0.5,
    ).transpose(0, 1)
    phase = tap.phase
    tap.phase = "env"
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        tap.record(f"sdpa_control.{name}", tensor,
                   "torch.nn.functional.scaled_dot_product_attention on canned inputs")
    tap.phase = phase


def apply_math_sdp_precision(choice: str) -> Dict[str, Any]:
    """Pin the math-SDPA reduction precision, the measured cause of the split.

    ``comfy.model_management`` calls
    ``torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`` at import
    (model_management.py:553). It is a **process-global** switch on the math
    SDPA path's accumulation precision, so every attention in a ComfyUI process
    accumulates in reduced precision while the same call in a RAVEN process
    accumulates in fp32. Measured on CPU with identical bf16 inputs: checksum
    740.935364 with the flag on, 741.044983 with it off -- which is exactly the
    ComfyUI-vs-RAVEN split this probe sees at the SDPA seam.

    ``auto`` leaves whatever the process set (i.e. reproduces production),
    ``on``/``off`` force both sides onto the same footing.
    """
    import torch

    applied: Dict[str, Any] = {"requested": choice}
    setter = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    getter = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
    if setter is not None and choice in ("on", "off"):
        try:
            setter(choice == "on")
        except Exception as exc:  # pragma: no cover - environment dependent
            applied["error"] = repr(exc)
    if getter is not None:
        try:
            applied["allowed"] = bool(getter())
        except Exception as exc:  # pragma: no cover
            applied["get_error"] = repr(exc)
    return applied


def apply_sdp_backend(choice: str) -> Dict[str, Any]:
    """Pin torch's SDPA backend so both processes cannot pick different kernels."""
    import torch

    applied: Dict[str, Any] = {"requested": choice}
    if choice == "auto":
        return applied
    wanted = {"flash": "flash", "mem_efficient": "mem_efficient", "math": "math"}[choice]
    for name, setter in (("flash", "enable_flash_sdp"),
                         ("mem_efficient", "enable_mem_efficient_sdp"),
                         ("math", "enable_math_sdp"),
                         ("cudnn", "enable_cudnn_sdp")):
        function = getattr(torch.backends.cuda, setter, None)
        if function is None:
            continue
        try:
            function(name == wanted)
            applied[name] = name == wanted
        except Exception as exc:  # pragma: no cover - environment dependent
            applied[f"{name}_error"] = repr(exc)
    return applied


def _derive_qkv_splits(tap: "ModuleTap", heads: int, head_dim: int) -> None:
    """Split every tapped fused QKV output into q/k/v.

    The split is upstream's own -- ``qkv.split(heads * head_dim, dim=-1)`` in
    comfy's ``Attention.forward`` and ``qkv.split(self.local_inner_dim, dim=-1)``
    in RAVEN's -- so this is a view of a recorded tensor, not a recomputation.
    """
    inner = int(heads) * int(head_dim)
    for stage in [name for name in tap.stages if name.endswith("attn.qkv_proj.out")]:
        fused = tap.stages[stage]
        if fused.shape[-1] != 3 * inner:
            continue
        rows = fused.shape[0]
        for offset, part in enumerate(("q", "k", "v")):
            derived = f"{stage}.{part}"
            tap.stages[derived] = fused[:, offset * inner:(offset + 1) * inner].reshape(
                rows, heads, head_dim).contiguous()
            tap.dtypes[derived] = tap.dtypes[stage]
            tap.sources[derived] = "derived: upstream's own qkv.split(heads*head_dim)"


def _make_pre_hook(tap: ModuleTap, path: str, module_name: str):
    def pre_hook(module, args):
        tensor = tap._first_tensor(args)
        if tensor is not None:
            tap.record(f"{path}.in", tensor, f"{module_name} forward pre-hook (input)")
    return pre_hook


def _make_hook(tap: ModuleTap, path: str, module_name: str):
    def hook(module, args, output):
        tensor = tap._first_tensor(output)
        if tensor is not None:
            tap.record(f"{path}.out", tensor, f"{module_name} forward hook (output)")
    return hook


def _site_from_path(name: str) -> Optional[Tuple[str, int]]:
    """``token_refiner.blocks.1.attn.k_norm.out`` -> ``('text_refiner', 1)``."""
    parts = name.split(".")
    for prefix, kind in (("token_refiner", "text_refiner"), ("blocks", "dit")):
        if parts[0] == prefix:
            for index, part in enumerate(parts):
                if part == "blocks" and index + 1 < len(parts) and parts[index + 1].isdigit():
                    return (kind, int(parts[index + 1]))
            if parts[0] == "blocks" and parts[1].isdigit():
                return (kind, int(parts[1]))
    return None


def _resolve_module(root, path: str):
    module = root
    for part in path.split("."):
        if part.isdigit():
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            module = getattr(module, part, None)
        if module is None:
            return None
    return module


def _derive_casts(tap: ModuleTap, dtype_name: str) -> None:
    """Add the bf16 cast each side applies to the fp32 projection output.

    ``video_embed.to(bf16)`` (comfy) and ``video_embed.to(_BF16_DTYPE)`` (RAVEN)
    are the same one-line step on both sides; deriving it here makes the
    quantisation visible next to the raw fp32 GEMM output it comes from.
    """
    import torch

    dtype = _torch_dtype(dtype_name)
    for stage in list(tap.stages):
        for name in ("video_patch_proj.out", "audio_patch_proj.out"):
            if stage.endswith(name):
                cast = tap.stages[stage].to(dtype).to(torch.float32)
                derived = stage.replace(".out", ".cast_" + dtype_name)
                tap.stages[derived] = cast
                tap.dtypes[derived] = str(dtype)
                tap.sources[derived] = "projection output cast into the bf16 decoder buffer"


# --- Comfy side --------------------------------------------------------------


def run_comfy_side(args, inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import comfy.ops

    from raven_streaming.cache import ChunkKVCache
    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model
    from raven_streaming.layout import T2VALayout

    request = inputs["request"]
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    config = dict(ARCHS[args.arch], num_layers=1)

    model = RavenCausalMiniMaxH3Model(
        **config, dtype=dtype, device=device,
        operations=comfy.ops.disable_weight_init,
    )
    state = read_embedding_state(args.weights)
    target = model.state_dict()
    placed = place_weights(state, target, fp32_island=args.comfy_fp32_island)
    missing, unexpected = model.load_state_dict(placed, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"embedding subtree does not fit a 1-block model: missing={list(missing)[:4]} "
            f"unexpected={list(unexpected)[:4]}"
        )
    model.to(device).requires_grad_(False).eval()

    layout = T2VALayout.from_request(text_len=request["text_len"], frames=request["frames"],
                                     width=request["width"], height=request["height"])

    def to_device(name):
        return inputs[name].to(device=device, dtype=dtype)

    import raven_streaming.causal_model as causal_module

    sdp = apply_sdp_backend(args.sdp_backend)
    # after comfy is imported: model_management flips this at import time
    math_sdp = apply_math_sdp_precision(args.math_sdp_reduced_precision)
    specs = TAP_SPECS + refiner_tap_specs(config["token_refiner_num_layers"])
    tap = ModuleTap().attach(model, specs=specs)
    seam = SeamTap(tap, causal_module, "raven_packed_attention", "dit")
    with torch.no_grad(), tap, seam:
        cache = ChunkKVCache(1, sink=int(request["sink"]), window=request["window"])
        tap.phase = "text"
        model.prefill_text(to_device("context"), cache=cache)
        tap.phase = "chunk0"
        model.forward_chunk(
            video_latent=layout.video_chunk_latent(to_device("video_xt"), 0),
            audio_latent=layout.audio_chunk_latent(to_device("audio_xt"), 0),
            layout=layout, chunk_index=0, cache=cache, role="noise",
            video_sigma=float(request["video_sigma"]),
            audio_sigma=float(request["audio_sigma"]),
        )
    sdpa_control(tap, device, dtype)
    _derive_casts(tap, args.dtype)
    _derive_qkv_splits(tap, config["num_attention_heads"], config["attention_head_dim"])

    meta = {
        "side": "comfy",
        "sdp_backend": sdp,
        "math_sdp_reduced_precision": math_sdp,
        "arch": args.arch,
        "dtype": args.dtype,
        "device": str(device),
        "runtime": runtime_meta(),
        "weights": str(args.weights),
        "fp32_island": args.comfy_fp32_island,
        "attention": {
            "seam": "raven_streaming.causal_model.raven_packed_attention",
            "seam_wrapped": seam.wrapped,
            "seam_calls": tap.seam_calls,
            "comfy_optimized_attention": _comfy_backend_name(),
            # the FA3 -> FA2 -> SDPA chain the causal lane resolved, and what it
            # actually ran; ``backend`` is the normalised name
            "lane_backend": _comfy_attention_backend(),
            "backend": (_comfy_attention_backend() or {}).get("backend"),
        },
        "weight_dtypes": _weight_dtypes(model),
        "weight_fingerprints": _weight_fingerprints(model),
        "loaded_keys": len(placed),
        "taps_fired": tap.fired,
        "entry_points": ["RavenCausalMiniMaxH3Model.prefill_text",
                         "RavenCausalMiniMaxH3Model.forward_chunk"],
        "request": request,
    }
    return _dump("comfy", tap, meta)


#: The checkpoint's fp32 island, plus one bf16 witness. A blanket cast to the
#: compute dtype before loading leaves the *declared* dtype fp32 while silently
#: destroying the mantissa, so dtype alone cannot detect it -- the fingerprint
#: below can.
WATCHED_WEIGHTS: Tuple[str, ...] = (
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_out.weight",
    "final_layer.video_out.weight",
    "condition_proj.weight",
    "blocks.0.attn.qkv_proj.weight",
    "rope.inv_freq",
)


def _comfy_backend_name() -> Optional[str]:
    """Whatever comfy bound as ``optimized_attention`` (dense lane / refiner fallback)."""
    try:
        from comfy.ldm.modules.attention import optimized_attention

        return getattr(getattr(optimized_attention, "__wrapped__", optimized_attention),
                       "__name__", None)
    except Exception:  # pragma: no cover - environment dependent
        return None


def _raven_flash_flags() -> Dict[str, Any]:
    """Which varlen backend RAVEN's wrapper bound, and the env that decided it.

    ``backend`` is normalised to the same vocabulary the causal lane reports
    (``fa3`` / ``fa2`` / ``sdpa``), because "FA3 vs SDPA" is the one comparison
    that decides whether a difference downstream of attention is a kernel's
    float error or a logic difference.
    """
    flags: Dict[str, Any] = {
        "FLASH_ATTN_2_AVAILABLE": os.environ.get("FLASH_ATTN_2_AVAILABLE"),
        "FLASH_ATTN_3_AVAILABLE": os.environ.get("FLASH_ATTN_3_AVAILABLE"),
    }
    try:
        import utils.flash_attn as flash

        flags["flash_attn_3"] = bool(getattr(flash, "FLASH_ATTN_3_AVAILABLE", False))
        flags["flash_attn_2"] = bool(getattr(flash, "FLASH_ATTN_2_AVAILABLE", False))
        flags["sdpa_fallback"] = not (flags["flash_attn_3"] or flags["flash_attn_2"])
        flags["backend"] = ("fa3" if flags["flash_attn_3"]
                            else "fa2" if flags["flash_attn_2"] else "sdpa")
    except Exception as exc:  # pragma: no cover - environment dependent
        flags["error"] = repr(exc)
        flags["backend"] = None
    return flags


def _comfy_attention_backend() -> Dict[str, Any]:
    """The causal lane's own backend snapshot, plus the normalised name.

    ``raven_streaming.causal_model.raven_attention_backend()`` reports what the
    FA3 -> FA2 -> SDPA chain resolved to and what the *last* call actually ran.
    """
    snapshot: Dict[str, Any] = {}
    try:
        from raven_streaming.causal_model import raven_attention_backend

        snapshot = dict(raven_attention_backend() or {})
    except Exception as exc:  # pragma: no cover - runtime dependent
        return {"error": repr(exc), "backend": None}
    last = snapshot.get("last") or {}
    resolved = snapshot.get("resolved") or {}
    backend = last.get("backend")
    if backend is None and resolved.get("order"):
        backend = resolved["order"][0]
    snapshot["backend"] = backend
    return snapshot


def normalised_backend(meta: Dict[str, Any]) -> Optional[str]:
    """``'fa3'`` / ``'fa2'`` / ``'sdpa'`` for either side's dump, or ``None``."""
    attention = meta.get("attention") or {}
    for key in ("backend", "comfy_backend"):
        value = attention.get(key)
        if isinstance(value, str):
            return value
    for key in ("lane_backend", "flash_attn"):
        block = attention.get(key)
        if isinstance(block, dict) and isinstance(block.get("backend"), str):
            return block["backend"]
    return None


def _weight_dtypes(model) -> Dict[str, str]:
    state = model.state_dict()
    return {name: str(state[name].dtype) for name in WATCHED_WEIGHTS if name in state}


def _weight_fingerprints(model) -> Dict[str, Dict[str, Any]]:
    """Per-weight evidence that an fp32 tensor still holds fp32 information.

    ``bf16_exact_fraction`` is the share of elements that survive a bf16 round
    trip unchanged. A genuine fp32 checkpoint tensor sits far below 1.0; a
    tensor that was cast to bf16 and back sits at exactly 1.0 while still
    *reporting* dtype float32.
    """
    import torch

    out: Dict[str, Dict[str, Any]] = {}
    state = model.state_dict()
    for name in WATCHED_WEIGHTS:
        value = state.get(name)
        if value is None or not value.is_floating_point():
            continue
        as_bf16 = value.to(torch.bfloat16).to(value.dtype)
        out[name] = {
            "dtype": str(value.dtype),
            "bf16_exact_fraction": round(float((value == as_bf16).float().mean()), 6),
            "absmax": round(float(value.abs().max()), 6),
        }
    return out


def _dump(side: str, tap: ModuleTap, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": EMBEDDING_DUMP_SCHEMA["version"],
        "side": side,
        "stages": tap.stages,
        "dtypes": tap.dtypes,
        "sources": tap.sources,
        "meta": meta,
    }


# --- RAVEN side --------------------------------------------------------------


def run_raven_side(args, inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    import raven_parity_harness as harness

    raven = harness.RavenModules(args.raven_root)
    from projects.minimax_h3.modeling.transformer.config import (  # noqa: E402
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTConfig,
    )

    request = inputs["request"]
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    config = dict(ARCHS[args.arch], num_layers=1)

    arch = MiniMaxH3DiTArchConfig(
        num_layers=1,
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
    dit = raven.dit_cls(config=MiniMaxH3DiTConfig(arch_config=arch), hf_config={})
    model = raven.x0_cls(dit)
    model = model.to(device=device, dtype=dtype)
    for name, parameter in dit.named_parameters():
        if name in raven.fp32_params:
            parameter.data = parameter.data.float()
    for name, buffer in dit.named_buffers():
        if name in raven.fp32_buffers:
            buffer.data = buffer.data.float()

    state = read_embedding_state(args.weights)
    target = dit.state_dict()
    placed = {k: v.to(device=device, dtype=target[k].dtype)
              for k, v in state.items() if k in target}
    missing = sorted(set(target) - set(placed))
    unexpected = sorted(set(state) - set(target))
    if missing or unexpected:
        raise SystemExit(
            f"embedding subtree does not fit RAVEN's 1-block DiT: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    dit.load_state_dict(placed, strict=True)
    model.requires_grad_(False).eval()
    if not hasattr(dit.token_refiner, "gradient_checkpointing"):
        dit.token_refiner.gradient_checkpointing = False

    batch = harness.build_batch(raven, request)
    base = object.__new__(raven.base_cls)
    context = inputs["context"][0].to(device=device, dtype=dtype)
    forward_inputs = raven.base_cls._build_inputs(base, batch, [context])
    layout = forward_inputs.layouts[0]
    chunk = layout.chunks[0]

    def to_device(name):
        return inputs[name].to(device=device, dtype=dtype)

    cache = raven.cache_cls(
        1, 1, sink=[int(request["sink"])],
        window_size=[None if request["window"] is None else int(request["window"])],
    )

    from projects.minimax_h3.modeling.transformer import model as raven_model_module
    from projects.minimax_h3.modeling.transformer import causal_model as raven_causal_module

    sdp = apply_sdp_backend(args.sdp_backend)
    math_sdp = apply_math_sdp_precision(args.math_sdp_reduced_precision)
    specs = TAP_SPECS + refiner_tap_specs(config["token_refiner_num_layers"])
    tap = ModuleTap().attach(dit, specs=specs)
    refiner_seam = SeamTap(tap, raven_model_module, "_MINIMAX_H3_FLASH_ATTENTION",
                           "text_refiner")
    dit_seam = SeamTap(tap, raven_causal_module, "_CAUSAL_FLASH_ATTENTION", "dit")
    with torch.no_grad(), tap, refiner_seam, dit_seam:
        tap.phase = "text"
        base._text_cache_fill(model, forward_inputs, cache)
        tap.phase = "chunk0"
        base._chunk_forward(
            model, forward_inputs, chunk_index=0, role="noise",
            video_rows_source=[harness.video_to_raven(to_device("video_xt"),
                                                      chunk.video_start, chunk.video_stop)],
            audio_rows_source=[harness.audio_to_raven(to_device("audio_xt"),
                                                      chunk.audio_start, chunk.audio_stop)],
            video_timesteps=torch.tensor([float(request["video_sigma"])], device=device),
            audio_timesteps=torch.tensor([float(request["audio_sigma"])], device=device),
            cache=cache, update_cache=False,
        )
    sdpa_control(tap, device, dtype)
    _derive_casts(tap, args.dtype)
    _derive_qkv_splits(tap, config["num_attention_heads"], config["attention_head_dim"])

    meta = {
        "side": "raven",
        "sdp_backend": sdp,
        "math_sdp_reduced_precision": math_sdp,
        "arch": args.arch,
        "dtype": args.dtype,
        "device": str(device),
        "runtime": runtime_meta(),
        "weights": str(args.weights),
        "fp32_island": "declared",
        "attention": {
            "refiner_seam": "projects...transformer.model._MINIMAX_H3_FLASH_ATTENTION",
            "dit_seam": "projects...transformer.causal_model._CAUSAL_FLASH_ATTENTION",
            "seam_wrapped": {"refiner": refiner_seam.wrapped, "dit": dit_seam.wrapped},
            "seam_calls": tap.seam_calls,
            "flash_attn": _raven_flash_flags(),
            "backend": _raven_flash_flags().get("backend"),
        },
        "weight_dtypes": _weight_dtypes(dit),
        "weight_fingerprints": _weight_fingerprints(dit),
        "loaded_keys": len(placed),
        "taps_fired": tap.fired,
        "entry_points": ["CausalMiniMaxH3Base._text_cache_fill",
                         "CausalMiniMaxH3Base._chunk_forward"],
        "raven_root": raven.root,
        "request": request,
    }
    return _dump("raven", tap, meta)


# --- comparison --------------------------------------------------------------


def compare_embedding_dumps(report: Report, comfy: Dict[str, Any], raven: Dict[str, Any],
                            *, rel_l2_max: float, cos_min: float) -> None:
    for side, dump in (("comfy", comfy), ("raven", raven)):
        if int(dump.get("version", -1)) != EMBEDDING_DUMP_SCHEMA["version"]:
            report.add(Check(f"{side}.schema_version", False,
                             {"got": dump.get("version")}))
            return

    report.meta["comfy"] = comfy["meta"]
    report.meta["raven"] = raven["meta"]
    report.meta["sources"] = {"comfy": comfy["sources"], "raven": raven["sources"]}
    report.meta["dtypes"] = {"comfy": comfy["dtypes"], "raven": raven["dtypes"]}
    report.meta["gate"] = {"stage_rel_l2_max": rel_l2_max, "cosine_min": cos_min}

    report.add(Check("setup.same_dtype",
                     comfy["meta"]["dtype"] == raven["meta"]["dtype"],
                     {"comfy": comfy["meta"]["dtype"], "raven": raven["meta"]["dtype"]}))
    report.add(Check("setup.same_request",
                     comfy["meta"]["request"] == raven["meta"]["request"],
                     {"text_len": comfy["meta"]["request"].get("text_len")}))
    report.add(Check("setup.taps_fired",
                     comfy["meta"]["taps_fired"] > 0 and raven["meta"]["taps_fired"] > 0,
                     {"comfy": comfy["meta"]["taps_fired"],
                      "raven": raven["meta"]["taps_fired"]}))

    # the fp32 island is a load-time property, and a mismatch explains more than
    # any single stage would
    island_diff = {name: {"comfy": dtype, "raven": raven["meta"]["weight_dtypes"].get(name)}
                   for name, dtype in comfy["meta"]["weight_dtypes"].items()
                   if raven["meta"]["weight_dtypes"].get(name) != dtype}
    report.add(Check("setup.same_weight_dtypes", not island_diff,
                     island_diff or {"checked": len(comfy["meta"]["weight_dtypes"])}))

    # dtype equality is not enough: a tensor cast to bf16 and back still reports
    # float32. Compare the fingerprints, and say plainly when a side's fp32
    # island carries no fp32 information at all.
    ours = comfy["meta"].get("weight_fingerprints", {})
    theirs = raven["meta"].get("weight_fingerprints", {})
    fingerprint_diff = {
        name: {"comfy": value["bf16_exact_fraction"],
               "raven": theirs[name]["bf16_exact_fraction"]}
        for name, value in ours.items()
        if name in theirs
        and abs(value["bf16_exact_fraction"] - theirs[name]["bf16_exact_fraction"]) > 1e-6
    }
    report.add(Check("setup.same_weight_values", not fingerprint_diff,
                     fingerprint_diff or {"checked": len(ours)}))
    for side, prints in (("comfy", ours), ("raven", theirs)):
        degraded = sorted(name for name, value in prints.items()
                          if value["dtype"] == "torch.float32"
                          and value["bf16_exact_fraction"] >= 1.0)
        if degraded:
            report.add(Check(f"{side}.fp32_island_intact", False,
                             {"bf16_representable": degraded}, gate=False))

    # RAVEN's glue re-runs the refiner on every chunk forward and projects
    # zero-filled latent rows during the text fill; comfy's causal lane does
    # neither. Both are structural, not numerical -- but only if the recomputed
    # refiner output is the same one the text phase produced, so check it.
    for side, dump in (("comfy", comfy), ("raven", raven)):
        text_refined = dump["stages"].get("text/token_refiner.final_norm.out")
        chunk_refined = dump["stages"].get("chunk0/token_refiner.final_norm.out")
        if text_refined is not None and chunk_refined is not None:
            metrics = tensor_metrics(chunk_refined, text_refined)
            report.add(Check(f"{side}.refiner_recompute_is_identical",
                             metrics["max_abs"] == 0.0,
                             _metric_detail(metrics), gate=False))
        # within-side consistency: the text rows the block stack receives are
        # exactly what the refiner produced. This is what caught the tap reading
        # a block input after the block had written its residual into it.
        block_in = dump["stages"].get("text/blocks.0.in")
        if text_refined is not None and block_in is not None:
            metrics = tensor_metrics(block_in, text_refined)
            report.add(Check(f"{side}.text_decoder_input_is_refiner_output",
                             metrics["max_abs"] == 0.0, _metric_detail(metrics)))

    # The environment control decides whether any stage number means anything:
    # if the same canned SDPA call disagrees, the two *processes* disagree and
    # the model code has not been measured yet.
    control = {}
    for name in ("q", "k", "v", "out"):
        stage = f"env/sdpa_control.{name}"
        if stage in comfy["stages"] and stage in raven["stages"]:
            control[name] = tensor_metrics(comfy["stages"][stage], raven["stages"][stage])
    if control:
        inputs_identical = all(control[name]["max_abs"] == 0.0 for name in ("q", "k", "v")
                               if name in control)
        report.add(Check("env.sdpa_control_inputs_identical", inputs_identical,
                         {name: control[name]["max_abs"] for name in control
                          if name != "out"}))
        if "out" in control:
            # Diagnostic, not a gate: this is a *process-level* control, run
            # outside the causal lane. The lane turns the math-SDPA reduction
            # off around its own SDPA call and restores it in finally, so the
            # two processes can still differ here while every lane stage is
            # bit-identical. What decides the run is the stages themselves.
            report.add(Check("env.sdpa_control_matches",
                             control["out"]["max_abs"] == 0.0,
                             dict(_metric_detail(control["out"]),
                                  comfy_sdp=comfy["meta"].get("sdp_backend"),
                                  raven_sdp=raven["meta"].get("sdp_backend"),
                                  scope="process-level control, outside the lane"),
                             gate=False))
            report.meta["sdpa_control"] = control["out"]

    # the specific process-global switch that was measured to split them
    ours = (comfy["meta"].get("math_sdp_reduced_precision") or {}).get("allowed")
    theirs = (raven["meta"].get("math_sdp_reduced_precision") or {}).get("allowed")
    if ours is not None or theirs is not None:
        report.add(Check("env.math_sdp_reduced_precision_matches", ours == theirs,
                         {"comfy": ours, "raven": theirs,
                          "note": "comfy.model_management enables it at import and "
                                  "RAVEN does not; the causal lane's SDPA fallback "
                                  "disables and restores it per call, so this is a "
                                  "process fact, not a lane difference"},
                         gate=False))

    # Which attention kernel each side ran. Two different kernels produce two
    # different float errors on identical inputs; that is not a logic
    # difference, and this probe must not report it as one.
    comfy_backend = normalised_backend(comfy["meta"])
    raven_backend = normalised_backend(raven["meta"])
    same_backend = (comfy_backend is not None and comfy_backend == raven_backend)
    report.meta["attention_backends"] = {
        "comfy": comfy_backend, "raven": raven_backend, "match": same_backend,
        "comfy_detail": (comfy["meta"].get("attention") or {}).get("lane_backend"),
        "raven_detail": (raven["meta"].get("attention") or {}).get("flash_attn"),
    }
    report.add(Check("env.attention_backend_matches", same_backend,
                     {"comfy": comfy_backend, "raven": raven_backend,
                      "effect": ("same kernel: attention stages are gated"
                                 if same_backend else
                                 "different kernels: stages from the first "
                                 "attention call on are classified "
                                 "kernel_float_error and reported, not gated")},
                     gate=False))
    report.meta["parity_scope"] = (
        "same attention kernel on both sides: every shared stage is comparable"
        if same_backend else
        f"different attention kernels ({comfy_backend} vs {raven_backend}): only "
        "stages before the first attention call are comparable; everything after "
        "it carries that kernel's float error and is reported as "
        "kernel_float_error, which is not evidence of a logic difference"
    )

    shared = [name for name in comfy["stages"] if name in raven["stages"]]
    only_comfy = sorted(set(comfy["stages"]) - set(raven["stages"]))
    only_raven = sorted(set(raven["stages"]) - set(comfy["stages"]))
    report.meta["one_sided_stages"] = {"comfy_only": only_comfy, "raven_only": only_raven}
    if only_comfy or only_raven:
        report.skip("stages.one_sided",
                    f"{len(only_comfy)} comfy-only, {len(only_raven)} raven-only "
                    "(see meta.one_sided_stages; RAVEN's _embed always projects the "
                    "latent rows and its _chunk_forward re-runs the refiner)")

    ordered = sorted(shared, key=_stage_sort_key)
    for name in ordered:
        metrics = tensor_metrics(comfy["stages"][name], raven["stages"][name])
        classification = classify_stage(
            metrics, same_backend=same_backend,
            attention_dependent=is_attention_dependent(name),
            is_control=name.startswith("env/"))
        row = dict(metrics, stage=name, classification=classification,
                   comfy_dtype=comfy["dtypes"].get(name),
                   raven_dtype=raven["dtypes"].get(name),
                   comfy_source=comfy["sources"].get(name),
                   raven_source=raven["sources"].get(name))
        report.metrics.append(row)
        passed = metrics_pass(metrics, rel_l2_max=rel_l2_max, cos_min=cos_min)
        detail = dict(_metric_detail(metrics), rel_l2_max=rel_l2_max, cos_min=cos_min,
                      classification=classification)
        if comfy["dtypes"].get(name) != raven["dtypes"].get(name):
            detail["dtype"] = f"{comfy['dtypes'].get(name)} vs {raven['dtypes'].get(name)}"
        if classification == "kernel_float_error":
            # measured under two different attention kernels: report it, do not
            # let it decide the run, and never call it a logic difference
            detail["backends"] = f"{comfy_backend} vs {raven_backend}"
            report.add(Check(f"stage.{name}", True, detail, gate=False))
        elif classification == "process_control":
            # the canned control lives outside the lane; env.* already reports it
            report.add(Check(f"stage.{name}", passed, detail, gate=False))
        else:
            report.add(Check(f"stage.{name}", passed, detail))

    report.meta["first_divergence"] = first_divergence(
        report.metrics, rel_l2_max=rel_l2_max)
    # the same question restricted to what this run can actually conclude from
    comparable = [row for row in report.metrics
                  if row.get("classification") not in ("kernel_float_error",
                                                       "process_control")]
    report.meta["first_divergence_comparable"] = first_divergence(
        comparable, rel_l2_max=rel_l2_max)
    report.meta["ranked_contributors"] = rank_contributors(report)
    report.meta["suggestions"] = suggest_fixes(report)


def rank_contributors(report: Report, top: int = 8) -> List[Dict[str, Any]]:
    """The stages that moved most, largest relative L2 first.

    The gate answers "is anything out of budget"; this answers "where should the
    next hour go", which is the actual question when block 0 is at 1%.
    """
    rows = [row for row in report.metrics if "rel_l2" in row]
    rows.sort(key=lambda row: (-(row["rel_l2"] if row["rel_l2"] == row["rel_l2"] else 0.0),
                               row["stage"]))
    return [{"stage": row["stage"], "rel_l2": row["rel_l2"], "cosine": row["cosine"],
             "p99_abs_over_ref_absmax": row["p99_abs_over_ref_absmax"],
             "comfy_dtype": row.get("comfy_dtype"), "raven_dtype": row.get("raven_dtype")}
            for row in rows[:top]]


#: Coarse execution order of the stage groups, so "which stage moved first" is a
#: question about the forward pass rather than about alphabetical order.
STAGE_GROUP_ORDER: Tuple[Tuple[str, int], ...] = (
    ("condition_proj", 10),
    ("token_refiner.blocks.", 20),
    ("token_refiner.final_norm", 30),
    ("video_patch_proj", 40),
    ("audio_patch_proj", 41),
    ("time_embedder", 50),
    ("blocks.0.adaln_proj", 60),
    ("blocks.0", 61),
    ("dit.", 62),
)


def execution_key(name: str) -> Tuple[int, int, int, int, str]:
    """``(phase, group, block, step, name)`` in forward-pass order."""
    phase, _, rest = name.partition("/")
    phase_rank = 0 if phase == "text" else 1
    if rest.startswith("token_refiner.blocks."):
        parts = rest.split(".")
        block = int(parts[2]) if parts[2].isdigit() else 0
        suffix = ".".join(parts[3:])
        step = (REFINER_STEP_ORDER.index(suffix) if suffix in REFINER_STEP_ORDER
                else len(REFINER_STEP_ORDER))
        return (phase_rank, 20, block, step, rest)
    for prefix, group in STAGE_GROUP_ORDER:
        if rest.startswith(prefix):
            return (phase_rank, group, 0, 0, rest)
    return (phase_rank, 99, 0, 0, rest)


#: Below this a difference is bf16/fp32 rounding rather than anything to chase.
FLOAT_ERROR_REL_L2 = 1e-6


#: Stage suffixes whose value is computed *from* an attention output. Position
#: in the forward order is not enough to decide this -- the time embedder runs
#: "after" the refiner in stage order and depends on nothing it produced -- so
#: the dependency is written out instead of inferred.
ATTENTION_DEPENDENT_SUFFIXES: Tuple[str, ...] = (
    "attn.sdpa.out",
    "attn.out_proj.in",
    "attn.out_proj.out",
    "norm2.in",
    "norm2.out",
    "mlp.fc1.out",
    "mlp.fc2.in",
    "mlp.out",
)


def is_attention_dependent(stage: str) -> bool:
    """Does this stage carry a value that came out of an attention call?

    True for the attention output and everything the block computes from it,
    for any refiner block after the first, for the refiner tail, and -- in the
    **text** phase only -- for the DiT stack, whose input *is* the refiner
    output. False for the SDPA seam's own q/k/v, for the projections and norms
    that feed it, for the time embedder and for the patch projections.
    """
    phase, _, rest = stage.partition("/")
    if phase == "env":
        return False
    if any(rest.endswith(suffix) for suffix in ATTENTION_DEPENDENT_SUFFIXES):
        return True
    if rest.startswith("token_refiner.final_norm"):
        return True
    if rest.startswith("token_refiner.blocks."):
        parts = rest.split(".")
        return parts[2].isdigit() and int(parts[2]) > 0
    if phase == "text" and (rest.startswith("blocks.") or rest.startswith("dit.")):
        # the text prefill feeds the refiner's output straight into the stack,
        # except the AdaLN input, which comes from the time embedder
        return not rest.startswith("blocks.0.adaln_proj")
    return False


def classify_stage(metrics: Dict[str, Any], *, same_backend: bool,
                   attention_dependent: bool, is_control: bool = False) -> str:
    """``exact`` / ``float_error`` / ``kernel_float_error`` / ``process_control`` /
    ``unexplained``.

    ``kernel_float_error`` is reserved for the one case it means: the two sides
    ran *different attention kernels* and this stage's value came out of one.
    It is a statement about arithmetic, never about logic.
    """
    if is_control:
        return "process_control"
    if metrics.get("rel_l2") == 0.0:
        return "exact"
    if attention_dependent and not same_backend:
        return "kernel_float_error"
    if metrics.get("rel_l2", float("inf")) <= FLOAT_ERROR_REL_L2:
        return "float_error"
    return "unexplained"


def first_divergence(metrics: List[Dict[str, Any]], *, rel_l2_max: float,
                     epsilon: float = 0.0) -> Dict[str, Any]:
    """The earliest stage that stops being exact, and the earliest out of budget.

    Both are reported: the first *non-zero* stage says where the two
    implementations start to disagree at all, which is the diagnostic question;
    the first stage over the gate says where it starts to matter.
    """
    ordered = sorted((row for row in metrics if "rel_l2" in row
                      and not str(row.get("stage", "")).startswith("env/")),
                     key=lambda row: execution_key(row["stage"]))
    out: Dict[str, Any] = {"first_nonzero": None, "first_over_gate": None,
                           "exact_prefix": []}
    for row in ordered:
        if row["rel_l2"] > epsilon and out["first_nonzero"] is None:
            out["first_nonzero"] = {k: row[k] for k in
                                    ("stage", "rel_l2", "cosine", "comfy_dtype",
                                     "raven_dtype") if k in row}
        if out["first_nonzero"] is None:
            out["exact_prefix"].append(row["stage"])
        if row["rel_l2"] > rel_l2_max and out["first_over_gate"] is None:
            out["first_over_gate"] = {k: row[k] for k in
                                      ("stage", "rel_l2", "cosine") if k in row}
    return out


def _stage_sort_key(name: str) -> Tuple[int, int, int, int, str]:
    return execution_key(name)


def suggest_fixes(report: Report) -> List[Dict[str, str]]:
    """Map failing stages onto the concrete, minimal runtime change they imply."""
    out: List[Dict[str, str]] = []
    seen = set()
    for check in report.checks:
        if check.passed or not check.name.startswith("stage."):
            continue
        stage = check.name[len("stage."):]
        _, _, module = stage.partition("/")
        for key, text in FIX_SUGGESTIONS.items():
            # a seam name (``attn.sdpa``) sits *inside* a module path, a stack
            # name (``token_refiner``) starts one; check both, most specific
            # first, which is the order FIX_SUGGESTIONS is written in
            if module.startswith(key) or module == key or f".{key}" in module:
                if key in seen:
                    break
                seen.add(key)
                out.append({"stage": stage, "module": key, "suggestion": text})
                break
    # An environment split is worth reporting even when no single stage cleared
    # the gate: it is the reason the stages below it cannot be trusted.
    env_checks = ("env.sdpa_control_matches", "env.math_sdp_reduced_precision_matches",
                  "env.sdpa_control_inputs_identical")
    failed_env = [c.name for c in report.checks if c.name in env_checks and not c.passed]
    if failed_env and "attn.sdpa" not in seen:
        seen.add("attn.sdpa")
        out.insert(0, {"stage": ", ".join(failed_env), "module": "attn.sdpa",
                       "suggestion": FIX_SUGGESTIONS["attn.sdpa"]})

    loader_checks = ("setup.same_weight_dtypes", "setup.same_weight_values",
                     "comfy.fp32_island_intact", "raven.fp32_island_intact")
    failed_loader = [c.name for c in report.checks
                     if c.name in loader_checks and not c.passed]
    if failed_loader:
        out.insert(0, {
            "stage": ", ".join(failed_loader),
            "module": "loader",
            "suggestion": (
                "The two sides hold different parameter dtypes. The H3 checkpoint "
                "has an fp32 island (patch projections, time embedder, output "
                "heads); a blanket .to(bf16) before load degrades it on one side "
                "only. Load with each parameter's declared dtype "
                "(--comfy-fp32-island declared) and re-measure before reading "
                "anything into the stage numbers."
            ),
        })
    return out


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("both", "comfy", "raven", "compare"),
                        default="both")
    parser.add_argument("--weights", default=None,
                        help="runtime-layout checkpoint; only the embedding subtrees "
                             "plus block 0 and the head are read")
    parser.add_argument("--inputs", default=None,
                        help="shared inputs from probe_causal_parity --mode inputs "
                             "(e.g. m2_full_inputs.pt)")
    parser.add_argument("--arch", choices=tuple(ARCHS), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--comfy-fp32-island", choices=("declared", "bf16"),
                        default="declared",
                        help="'declared' keeps the checkpoint's fp32 island fp32; "
                             "'bf16' reproduces a blanket cast so its effect is measurable")
    parser.add_argument("--emit-dump", default=None)
    parser.add_argument("--comfy-dump", default=None)
    parser.add_argument("--raven-dump", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--comfyui-path", default=None)
    parser.add_argument("--raven-root", default=os.environ.get("RAVEN_ROOT", "/root/Jarvis"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--comfy-python", default=None)
    parser.add_argument("--raven-python", default=None)
    parser.add_argument("--rel-l2-max", type=float, default=STAGE_REL_L2_MAX)
    parser.add_argument("--cos-min", type=float, default=COSINE_MIN)
    parser.add_argument("--sdp-backend", choices=("auto", "flash", "mem_efficient", "math"),
                        default="auto",
                        help="pin torch's SDPA kernel on both sides; 'auto' leaves "
                             "each process to pick, which is how they can disagree")
    parser.add_argument("--math-sdp-reduced-precision", choices=("auto", "on", "off"),
                        default="auto",
                        help="torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp. "
                             "comfy.model_management turns it ON at import and RAVEN "
                             "never does; 'auto' reproduces that production split, "
                             "'off'/'on' put both sides on the same footing")
    args = parser.parse_args(argv)
    args.block = None  # the spawn helper is shared with the operator probe
    return args


def _run_side(args, side: str) -> Dict[str, Any]:
    import torch

    if not args.weights:
        raise SystemExit(f"--side {side} needs --weights")
    if not args.inputs:
        raise SystemExit(f"--side {side} needs --inputs (both sides read the same file)")
    inputs = load_inputs(args.inputs)
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
        dump = run_comfy_side(args, inputs)
    else:
        print(f"RAVEN: {args.raven_root}")
        dump = run_raven_side(args, inputs)
    if args.emit_dump:
        Path(args.emit_dump).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dump, args.emit_dump)
        print(f"dump: {args.emit_dump}")
    print(f"stages: {len(dump['stages'])}  taps fired: {dump['meta']['taps_fired']}")
    print(f"weight dtypes: {dump['meta']['weight_dtypes']}")
    return dump


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.side in ("comfy", "raven"):
        _run_side(args, args.side)
        return 0

    import torch

    report = Report(mode="embedding", device=args.device)
    report.meta["embedding_dump_schema"] = EMBEDDING_DUMP_SCHEMA

    if args.side == "compare":
        if not args.comfy_dump or not args.raven_dump:
            raise SystemExit("--side compare needs --comfy-dump and --raven-dump")
        comfy = torch.load(args.comfy_dump, map_location="cpu", weights_only=False)
        raven = torch.load(args.raven_dump, map_location="cpu", weights_only=False)
    else:
        if not args.inputs:
            raise SystemExit("--side both needs --inputs")
        work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(
            prefix="raven_embedding_parity_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        comfy_path = args.comfy_dump or str(work_dir / "comfy_emb.pt")
        raven_path = args.raven_dump or str(work_dir / "raven_emb.pt")

        comfy_root = str(find_comfyui(args.comfyui_path))
        raven_root = Path(args.raven_root).expanduser()
        if not (raven_root / "projects" / "minimax_h3").is_dir():
            raise SystemExit(f"--raven-root {raven_root} is not a RAVEN checkout")

        plans = []
        for side in ("comfy", "raven"):
            # every knob that changes numerics must reach *both* children
            extra = ["--sdp-backend", args.sdp_backend,
                     "--math-sdp-reduced-precision", args.math_sdp_reduced_precision]
            if side == "comfy":
                extra += ["--comfy-fp32-island", args.comfy_fp32_island]
            plans.append(build_spawn_plan(
                args, side, args.inputs,
                comfy_path if side == "comfy" else raven_path,
                comfy_root=comfy_root, raven_root=str(raven_root),
                script=str(Path(__file__).resolve()), extra_args=extra))
        report.meta["spawn"] = {plan.side: plan.describe() for plan in plans}
        for plan in plans:
            _spawn_side(plan)
        comfy = torch.load(comfy_path, map_location="cpu", weights_only=False)
        raven = torch.load(raven_path, map_location="cpu", weights_only=False)
        report.meta["work_dir"] = str(work_dir)

    print()
    compare_embedding_dumps(report, comfy, raven,
                            rel_l2_max=args.rel_l2_max, cos_min=args.cos_min)

    gating = [c for c in report.checks if c.gate]
    print(f"\nGATE {'PASS' if report.passed else 'FAIL'}: "
          f"{sum(1 for c in gating if c.passed)}/{len(gating)} gating checks, "
          f"{report.diagnostics_failed} diagnostic difference(s), "
          f"{len(report.skipped)} skipped")
    backends = report.meta.get("attention_backends", {})
    if backends:
        print(f"\nattention backend: comfy={backends.get('comfy')} "
              f"raven={backends.get('raven')} "
              f"({'same kernel' if backends.get('match') else 'DIFFERENT kernels'})")
        print(f"  scope: {report.meta.get('parity_scope')}")

    divergence = report.meta.get("first_divergence", {})
    first = divergence.get("first_nonzero")
    print("\nfirst stage that stops being exact (forward order):")
    if first is None:
        print("  none -- every shared stage is bit-identical")
    else:
        print(f"  {first['stage']}  rel_l2={first['rel_l2']:.3e} "
              f"cos={first['cosine']:.6f}")
        print(f"  ({len(divergence.get('exact_prefix', []))} stage(s) before it "
              f"were exact)")
    over = divergence.get("first_over_gate")
    if over is not None:
        print(f"first stage over the gate: {over['stage']}  "
              f"rel_l2={over['rel_l2']:.3e}")
    if backends and not backends.get("match"):
        comparable = report.meta.get("first_divergence_comparable", {})
        first_comparable = comparable.get("first_nonzero")
        print("comparable stages only (upstream of the first attention call): "
              + ("all exact" if first_comparable is None else
                 f"{first_comparable['stage']} rel_l2={first_comparable['rel_l2']:.3e}"))
        counts: Dict[str, int] = {}
        for row in report.metrics:
            counts[row.get("classification", "?")] = \
                counts.get(row.get("classification", "?"), 0) + 1
        print(f"stage classification: {counts}")

    print("\nlargest upstream differences (rel_l2):")
    for row in report.meta.get("ranked_contributors", []):
        print(f"  {row['rel_l2']:.3e}  cos={row['cosine']:.6f}  {row['stage']}"
              + (f"  [{row['comfy_dtype']} vs {row['raven_dtype']}]"
                 if row["comfy_dtype"] != row["raven_dtype"] else ""))

    refiner = [row for row in report.metrics
               if "token_refiner.blocks." in row.get("stage", "")]
    if refiner:
        print("\nrefiner, in forward order:")
        for row in sorted(refiner, key=lambda r: execution_key(r["stage"])):
            print(f"  {row['rel_l2']:.3e}  cos={row['cosine']:.6f}  {row['stage']}")
    for item in report.meta.get("suggestions", []):
        print(f"\n[suggestion] {item['module']} (first seen at {item['stage']}):\n"
              f"  {item['suggestion']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2, default=str))
        print(f"\nreport: {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
