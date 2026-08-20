"""Chunk-causal MiniMax H3 DiT: state-dict-neutral subclasses of the official model.

Scope (M2)
----------
T2VA only, single batch, single GPU, no CFG, no references, no keyframes, no
sequence/FSDP parallelism, no training path. What lands here is the *causal
lane*: a text prefill, a per-chunk forward against a KV cache, and nothing else.
The sampler (M3) drives it; no node, no sampler and no web code lives here.

Three classes, each a subclass of its official counterpart in
``comfy.ldm.minimax.model``:

``RavenCausalAttention(Attention)``
    Same parameters, same names, same shapes -- ``layer_idx`` is a plain int,
    not a module. QKV projection, per-head RMSNorm and *absolute* RoPE are the
    official ones (including the fused ``rms_rope_split_half`` kernel), then the
    retained cached K/V are assembled in front of this chunk's own K/V -- one
    merged buffer per layer, filled in place by the cache (see "The merged
    slot" below) -- while Q stays current-only. Attention itself is ``mask``-free: the current chunk
    is fully visible inside itself and fully attends to whatever the cache
    retained, which is exactly the visibility RAVEN's training mask encodes for
    a non-diagonal chunk. Cached keys keep the absolute positions they were
    rotated with, so no re-basing is ever needed.

``RavenCausalDiTBlock(DiTBlock)``
    The official block with ``self.attn`` swapped for the causal one. Same
    module keys, same norm / AdaLN / gate / MLP order -- the forward body is a
    verbatim fork with the cache kwargs threaded through.

``RavenCausalMiniMaxH3Model(MiniMaxH3Model)``
    The official model, whose 50 blocks are rebuilt as causal blocks after
    construction (one at a time, so peak memory stays at one extra block). The
    state dict is unchanged, key for key and shape for shape -- construction
    asserts it. The inherited dense ``forward``/``_forward`` keeps working (the
    causal kwargs default to "no cache"), and two explicit APIs are added:
    :meth:`prefill_text` and :meth:`forward_chunk`.

Two lanes, one state dict
-------------------------
Everything here is dual-path, and the switch is **the cache**:

* ``cache is None`` -- the *dense* lane. Every forward body defers to its
  official parent (``super().forward``, ``AdalnProj.forward``,
  ``FinalLayer.forward``, ``optimized_attention``, ``TokenRefiner.forward``),
  so the inherited dense ``forward``/``_forward`` stays bit-identical to
  upstream. ``tests/test_causal_model_parity.py`` pins that.
* ``cache is not None`` -- the *causal* lane, which reproduces **RAVEN's**
  numerical order rather than Comfy's, because the checkpoint and the LoRA were
  trained through RAVEN's operators. The four places the two implementations
  disagree, all measured on a real CUDA BF16 single-block audit:

  ``attention``
      RAVEN hands packed ``[rows, heads, dim]`` q/k/v to
      ``utils.flash_attn.FlashAttention``, which dispatches **FlashAttention 3,
      then FlashAttention 2, then a packed SDPA fallback**. Comfy's
      ``optimized_attention`` instead dispatches a 4-D
      ``[1, heads, rows, dim]`` call through whichever backend its build picked
      (relative L2 0.0024-0.0028 against RAVEN).
      :func:`raven_packed_attention` is the RAVEN path, single document,
      batch 1, same priority chain -- see "Attention backend" below.
  ``AdaLN input``
      RAVEN computes ``silu(t_emb)`` **in fp32** once per forward and casts the
      result to the compute dtype; ``AdalnProj.forward`` runs the SiLU at the
      module dtype, i.e. on an already-rounded BF16 ``t_emb`` (0.0035-0.0052).
      The causal lane therefore builds the shared ``adaln_input`` itself and
      calls ``adaln_proj.linear`` directly -- ``apply_silu`` is never mutated,
      because that flag is also read by the dense lane.
  ``gate``
      RAVEN's ``_modulate_gate`` is out of place, ``(x + gate * other).to(dt)``,
      rounding the product before the sum; Comfy's ``_mod_gate`` fuses them
      into ``addcmul_`` (0.000886). :func:`_raven_mod_gate` is the RAVEN form.
  ``text refiner``
      Same attention split as above, so :meth:`prefill_text` replays the
      refiner block body over the official modules.

  Everything else -- qkv GEMM, QK norm, RoPE, the AdaLN GEMM, scale/shift, the
  SwiGLU MLP -- was measured **bit-identical** and is left alone: the causal
  lane keeps calling the official modules and the official
  ``_mod_scale_shift``.

Every weight is still reached through its module's ``__call__``. That is not
style: :mod:`raven_streaming.runtime_linear` attaches the RAVEN LoRA as a
``register_forward_hook`` on each ``Linear``, and ComfyUI's partial offload
casts weights inside ``forward``. Touching ``.weight`` directly, or calling
``module.forward(...)``, would silently drop the adapter and read an offloaded
tensor.

Attention backend
-----------------
A strict vr audit settled what the remaining attention gap actually is. Once
both sides were forced onto the *same* math SDPA with
``allow_fp16_bf16_reduction_math_sdp(False)``, every stage -- refiner, DiT
blocks, embeddings -- came out **bit-identical**. So the production difference
was never the operator order: it was that RAVEN runs an external varlen
**FlashAttention 3** kernel while this lane ran PyTorch SDPA.

:func:`raven_packed_attention` therefore reproduces RAVEN's dispatch, not just
its fallback:

1. ``flash_attn_interface.flash_attn_varlen_func`` (FA3), then
2. ``flash_attn.flash_attn_varlen_func`` (FA2), then
3. PyTorch SDPA, the packed transcription of ``utils.flash_attn._sdpa_varlen``.

The first two are used only on CUDA, only for fp16/bf16, only for
``head_dim <= 256`` -- RAVEN's own assertions -- and are called with RAVEN's
exact keyword set (``cu_seqlens_q``/``cu_seqlens_k`` built by the same cumsum,
``max_seqlen_q``/``max_seqlen_k``, ``softmax_scale``, ``causal=False``,
``deterministic=False``, FA2's ``dropout_p=0.0`` and ``window_size=(-1, -1)``,
FA3's ``seqused_q=None``/``seqused_k=None``). Nothing is imported from RAVEN;
the two packages are imported directly, lazily, and cached.

``FLASH_ATTN_3_AVAILABLE=0`` / ``FLASH_ATTN_2_AVAILABLE=0`` disable a backend,
exactly as they do in RAVEN, which is how a parity run pins both sides to the
same kernel.

The SDPA step is where the vr finding is honoured: it saves
``fp16_bf16_reduction_math_sdp_allowed()``, turns the reduction off **for that
one call**, and restores it in ``finally`` -- including on exception -- so a
Comfy-wide setting is never left changed.

What may fall back and what may not is a deliberate line. A missing package, a
disabled env switch, an unmet precondition, or a kernel that says it does not
support this build/shape (see :func:`_unsupported_reason`) fall back and are
recorded. Anything else -- OOM, a CUDA fault, an assertion from inside the
kernel -- propagates: swallowing it would silently change the numbers that this
whole module exists to keep. :func:`raven_attention_backend` reports what was
resolved and what the last call actually ran.

The merged slot
---------------
A cached layer needs ``[retained | current]`` keys and values on the compute
device, and nothing else. It gets them as **one** allocation:
:meth:`ChunkKVCache.retained_spec` says how many rows the history is,
``k.new_empty((past + rows, heads, head_dim))`` reserves the merged pair, and
:meth:`ChunkKVCache.copy_retained_into` writes the retained rows straight into
the prefix while this chunk's own rows are copied into the tail.

The old spelling -- ``torch.cat((cache.retained(...), k))`` -- allocated the
gathered history *and* the merge, so two device buffers were live at the peak;
with the cache's canonical copy on the host it would have been three. The
assembly API removes both extra buffers, which is what makes a host-side KV
cache cost one merged slot of device memory at a time instead of the whole
history. The values and the layout are unchanged: contiguous, row-major,
retained rows first, exactly what ``cat`` produced and what RAVEN's own scatter
produces.

A layer with **no** history allocates nothing: ``k`` stays the freshly
materialised contiguous tensor and ``v`` stays the fused-QKV view, which is
RAVEN's no-history layout (see the stride note in
:meth:`RavenCausalAttention.forward`).

The path is the same for every ``storage`` the cache was built with -- a
device-resident cache copies device-to-device into the same prefix -- so
``gpu``, ``cpu_pinned`` and ``cpu`` are bit-for-bit the same rollout, and only
the residency and the bandwidth differ.

Timestep convention
-------------------
The sampler side speaks the RAVEN repo convention: ``t_repo = sigma``, so
``t_repo = 0`` is clean. This module converts at its boundary:

* a **noise** chunk carries its own ``sigma`` per stream (video and audio run
  independent shifted grids) and maps to H3 time ``t_h3 = 1 - sigma``;
* a **clean** chunk (context written into the cache) maps to
  ``t_h3 = 0.999`` -- the checkpoint's attested condition timestep -- and its
  rows are re-noised as ``t * x0 + (1 - t) * eps`` per stream, with ``eps``
  supplied by the caller so a later pass can reproduce the exact context;
* text is prefilled alone at the same ``0.999``, never merged with media rows.

Every one of those numbers is evaluated in **float32**, because RAVEN evaluates
them in float32 (see :func:`_fp32_one_minus`): ``t_h3 = 1 - sigma`` is an fp32
subtraction, the condition constant is the fp32 ``0.999``, and the mix's eps
coefficient is ``1 - fp32(0.999) = 0.00099998713``, *not* ``fp32(0.001)``. The
sigma schedule itself is untouched -- only the conversion at this boundary.

Distinct timesteps become distinct ``t_emb`` rows, and each row expands to the
official three AdaLN modality rows (video 0, text 1, audio 2), so the AdaLN
semantics are the dense model's.

Returned velocity
-----------------
:meth:`forward_chunk` returns the **native H3 velocity** ``v = x0 - eps`` for
the chunk's own rows, i.e. the raw head output. The official dense ``forward``
returns ``-v`` (Comfy's sampler convention); the causal lane deliberately does
not negate, because the RAVEN consistency step consumes H3 velocity directly.
:func:`velocity_to_x0` is the matching conversion.

It is returned in **fp32**, in the head's own dtype, and is *not* rounded back
to the latent's. That is the checkpoint's contract, not a preference:
``FinalLayer.video_out``/``audio_out`` are built ``dtype=torch.float32``
("the checkpoint's fp32 island", upstream's words) exactly like
``video_patch_proj``/``audio_patch_proj`` and the time embedder, and RAVEN's
``MiniMaxH3FinalLayer`` does the same -- it casts the modulated activations up
with ``h.to(_FP32_DTYPE)`` and both heads are ``params_dtype=_FP32_DTYPE``.
RAVEN's ``MiniMaxH3X0Model.forward`` then converts velocity to ``x0`` in fp32
(``minimax_h3_rf_v_to_x0`` on the fp32 logits) and never rounds in between.

This used to be ``velocity.to(video_latent.dtype)``. With a ComfyUI ``LATENT``
(fp32) that cast was a no-op, which is why it survived; with a **BF16** latent
-- what the parity harness runs, and what a BF16 ``LATENT`` would be -- it put
one bf16 ULP of *relative* error (2**-9, ~1.7e-3) on every value the sampler
consumes. It was the last measurable difference against the RAVEN harness once
the operators and the attention backend matched: with the cast, per-chunk
``video_x0`` sat at rel_l2 1.67e-3; without it, **0.0** (audio 1.2e-8, fp32
noise from the permute). ``unpatchify_video`` and ``unpack_audio`` are pure
reshape/permute, so nothing between the head and the caller changes a bit.

What crosses into fp32 is therefore only the *output* side: the velocity, the
sampler's ``x0``/``x_t`` transition arithmetic, and the latent it finally
delivers. The model's own compute stays in the compute dtype -- the packed rows,
the blocks, and the cached K/V are all BF16 -- and the next chunk's embedding
steps re-enter through ``patchify_video(x.to(torch.float32))`` /
``pack_audio``, i.e. the fp32 patch projections the production dense forward
uses too, whose output is cast back to the compute dtype before the blocks. An
fp32 latent arriving from the sampler is therefore what those projections
already expect; a BF16 one is silently upcast by the same line.

Inference only
--------------
:meth:`prefill_text` and :meth:`forward_chunk` run under ``torch.no_grad``:
RAVEN's rollout is a no-grad path, upstream's fused in-place RoPE kernel refuses
autograd outright, and a backward through a cached forward would recompute the
cache merge against an already-advanced cache.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.minimax.model import (
    Attention,
    DiTBlock,
    MiniMaxH3Model,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
# Private upstream helpers, imported rather than re-implemented.
# ``_mod_scale_shift`` and ``_mod_row`` are used by the *causal* lane too: the
# audit measured them bit-identical to RAVEN's ``_modulate_scale_shift`` (both
# round the product and then the sum, in the compute dtype). ``_mod_gate`` is
# not, and is deliberately not imported -- see :func:`_raven_mod_gate`.
#
# Neither is ``optimized_attention``. It used to be this module's attention
# call, and ``tools/probe_causal_parity.py``'s ``KVTap`` still wraps
# ``raven_streaming.causal_model.optimized_attention`` to record what every
# layer's backend is handed. That tap must move to :func:`raven_packed_attention`
# (plain tensors now, not ``AttentionTensorContainer``s); until it does it
# raises ``AttributeError`` on install, which is the intended loud failure --
# leaving the name importable would have made the probe silently record nothing.
from comfy.ldm.minimax.model import _mod_row, _mod_scale_shift

from raven_streaming import layout as layout_mod
from raven_streaming.cache import ChunkKVCache

__all__ = [
    "CausalModelError",
    "CLEAN_TIMESTEP_VIDEO",
    "CLEAN_TIMESTEP_TEXT",
    "CLEAN_TIMESTEP_AUDIO",
    "velocity_to_x0",
    "raven_packed_attention",
    "raven_attention_backend",
    "RavenCausalAttention",
    "RavenCausalDiTBlock",
    "RavenCausalMiniMaxH3Model",
]

_LOG = logging.getLogger(__name__)


class CausalModelError(RuntimeError):
    """A causal forward that violates the chunk/cache contract."""


#: Checkpoint fidelity: H3's condition rows are attested at 0.999. Do not tune.
CLEAN_TIMESTEP_VIDEO = 0.999
#: Text natively inherits the checkpoint's video timestep.
CLEAN_TIMESTEP_TEXT = 0.999
#: RAVEN's choice for predicted audio context; tune this first if audio drifts.
CLEAN_TIMESTEP_AUDIO = 0.999


def velocity_to_x0(x_t: torch.Tensor, velocity: torch.Tensor, t_h3: float) -> torch.Tensor:
    """``x0 = x_t + (1 - t_h3) * v`` for native H3 velocity ``v``.

    RAVEN's ``_minimax_h3_rf_v_to_x0`` casts the timestep to ``x_t``'s dtype and
    subtracts there. With an fp32 ``x_t`` (what :meth:`forward_chunk` now feeds
    the sampler) and a ``t_h3`` that is exactly representable in fp32 (what
    :func:`_fp32_scalar` guarantees for every timestep this module produces),
    that is the same single rounding as the subtraction below: the double
    difference of two exact fp32 values is itself exact, so rounding it at the
    multiply lands on the same fp32 number the fp32 subtraction would.
    """
    return x_t + (1.0 - float(t_h3)) * velocity


# --- timesteps, in RAVEN's precision ----------------------------------------
#
# RAVEN's timesteps live in float32 tensors from the moment they enter the
# model: the rollout builds ``torch.tensor([sigma])`` (fp32), ``MiniMaxH3X0Model``
# does ``repo_t = unique_timesteps.float()`` and then
# ``h3_t = where(repo_t == 0, clean_by_tag[tags], 1.0 - repo_t)`` -- an fp32
# subtraction -- and the *same* fp32 values go on to the time embedder and to
# the clean-context mix ``t * x + (1 - t) * eps``.
#
# Spelling that as ``1.0 - float(sigma)`` in Python instead keeps the schedule's
# full double precision through the subtraction and only rounds afterwards,
# which is a different number: the embedding probe measured ~1e-8 on
# ``time_embedder.out`` from exactly this. Every timestep this module hands to
# the embedder or to a mix therefore goes through the two helpers below, so a
# Python float is only ever a *carrier* for a value that is already float32.


def _fp32_scalar(value: float) -> float:
    """``value`` rounded to float32, carried back as an exact Python float.

    ``float(torch.tensor(x, dtype=torch.float32))`` is a double holding exactly
    the fp32 number, so ``torch.tensor([...], dtype=torch.float32)`` rebuilds it
    bit for bit and set/dict keys made from it cannot split one fp32 timestep
    into two rows.
    """
    return float(torch.tensor(float(value), dtype=torch.float32))


def _fp32_one_minus(value: float) -> float:
    """``1 - value`` evaluated in float32, RAVEN's two ``1 - x`` sites.

    Those sites are ``1.0 - repo_t`` (repo sigma -> H3 timestep) and the
    ``(1 - t)`` coefficient of the clean-context mix. Both run on fp32 tensors
    there, so both run in fp32 here.

    Done on the host: this is one IEEE-754 subtraction of a scalar, which is
    exact-to-the-same-bit on every device, and doing it on the compute device
    would cost a launch and a sync per chunk for a number the layout code
    already needs in Python.
    """
    return float(1.0 - torch.tensor(float(value), dtype=torch.float32))


# --- RAVEN primitives -------------------------------------------------------
#
# The RAVEN counterparts of the Comfy operators a real-CUDA BF16 audit showed to
# differ (attention wrapper, AdaLN input, gate), plus the two small replays that
# drive them over the official modules. They are module-level (not methods) for
# the same reason RAVEN keeps ``_CAUSAL_FLASH_ATTENTION`` at module level: they
# hold no state, they must stay out of the module tree, and a probe can wrap the
# module attribute to tap them.


#: ``(name, module, attribute, env switch)`` for the two external kernels, in
#: RAVEN's own priority order. The env switches are RAVEN's, spelled the same
#: way, so one export pins both implementations to the same backend.
_FLASH_BACKENDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("fa3", "flash_attn_interface", "flash_attn_varlen_func", "FLASH_ATTN_3_AVAILABLE"),
    ("fa2", "flash_attn", "flash_attn_varlen_func", "FLASH_ATTN_2_AVAILABLE"),
)

#: Resolution is cached: importing ``flash_attn`` costs hundreds of ms and a
#: rollout calls the seam 50 times per chunk forward.
_ATTENTION_STATE: Dict[str, Any] = {"resolved": None, "last": None}

#: A kernel message matching one of these is a capability statement ("this
#: build/GPU/shape is not supported"), which is a legitimate reason to fall
#: back to the next backend.
_UNSUPPORTED_MARKERS: Tuple[str, ...] = (
    "only support", "not support", "unsupported", "no kernel", "not compiled",
    "not built", "is not available", "requires", "ampere", "hopper", "sm_",
    "sm80", "sm90", "head_dim", "headdim", "unexpected keyword argument",
    "required positional argument",
)

#: ... and one matching any of these never is, whatever else it says. A kernel
#: that ran out of memory or faulted must not be turned into a quiet backend
#: switch that changes the result.
_FATAL_MARKERS: Tuple[str, ...] = (
    "out of memory", "cuda error", "illegal memory access", "device-side assert",
    "misaligned address", "an illegal instruction", "driver shutting down",
)

#: Exception types a capability failure can plausibly arrive as. Anything else
#: propagates untouched.
_UNSUPPORTED_TYPES: Tuple[type, ...] = (
    NotImplementedError, RuntimeError, ValueError, AssertionError, TypeError,
)


def _env_switch(name: str) -> bool:
    """RAVEN's ``bool(int(os.environ.get(name, "1")))``, with a readable error."""
    raw = os.environ.get(name, "1")
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        raise CausalModelError(
            f"{name}={raw!r} is not an integer; RAVEN reads these switches as "
            f"bool(int(...)), so use 0 or 1"
        ) from None


def _resolve_attention_backends(force: bool = False) -> Dict[str, Any]:
    """Import FA3/FA2 once and remember the outcome (and why).

    RAVEN resolves these at *import* time; here it is deferred to the first
    attention call, because importing ``flash_attn`` inside a ComfyUI custom
    node's module import would cost every user hundreds of milliseconds whether
    or not they run this lane. The observable behaviour is the same: one
    resolution per process, env switches honoured, no per-call import.

    A broken install (``ImportError`` on a missing CUDA symbol, ``OSError`` on a
    shared object that will not load) is treated as "not present" rather than
    propagated -- RAVEN only guards ``ModuleNotFoundError``, but there the
    import is at module scope and the process is a RAVEN process; here it would
    take down an unrelated ComfyUI startup path.
    """
    state = _ATTENTION_STATE
    if state["resolved"] is not None and not force:
        return state["resolved"]

    resolved: Dict[str, Any] = {"order": [name for name, *_ in _FLASH_BACKENDS],
                                "available": {}, "why": {}, "func": {}}
    for name, module_name, attribute, env in _FLASH_BACKENDS:
        enabled = _env_switch(env)
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            resolved["why"][name] = f"{module_name} not importable: {type(exc).__name__}"
            resolved["available"][name] = False
            continue
        func = getattr(module, attribute, None)
        if func is None:
            resolved["why"][name] = f"{module_name} has no {attribute}"
            resolved["available"][name] = False
            continue
        if not enabled:
            # RAVEN checks the switch only after a successful import, so an
            # env-disabled backend still reports as installed.
            resolved["why"][name] = f"disabled by {env}=0"
            resolved["available"][name] = False
            continue
        resolved["func"][name] = func
        resolved["available"][name] = True
        resolved["why"][name] = f"{module_name}.{attribute}"
    state["resolved"] = resolved
    return resolved


def _reset_attention_backends() -> None:
    """Drop the cached resolution. Test seam; not part of the public surface."""
    _ATTENTION_STATE["resolved"] = None
    _ATTENTION_STATE["last"] = None


def raven_attention_backend() -> Dict[str, Any]:
    """Read-only snapshot: what is available, and what the last call ran.

    ``{'resolved': {'order', 'available', 'why'}, 'last': {'backend', 'reason',
    'site', 'rows', 'kv_rows'}}``. Purely diagnostic -- nothing here is a
    module attribute, a buffer or a parameter, so the state dict is untouched
    and two models in one process share one resolution.

    A parity run should record it: "FA3 vs SDPA" is exactly the difference the
    vr audit isolated, so a dump that does not say which one ran cannot be
    compared against another.
    """
    resolved = _ATTENTION_STATE["resolved"]
    snapshot: Dict[str, Any] = {
        "resolved": None if resolved is None else {
            "order": list(resolved["order"]),
            "available": dict(resolved["available"]),
            "why": dict(resolved["why"]),
        },
        "last": None if _ATTENTION_STATE["last"] is None else dict(_ATTENTION_STATE["last"]),
    }
    return snapshot


def _unsupported_reason(exc: BaseException) -> Optional[str]:
    """``str`` when ``exc`` is a capability failure, ``None`` when it is real.

    The line this draws is the whole safety of the fallback chain. "This kernel
    does not support sm80 / this head_dim / this build" is a routing fact and
    the next backend is the right answer. "CUDA out of memory" or a device-side
    assert is a real failure whose result would silently change if it were
    answered by quietly running a different kernel, so it propagates.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return None
    if not isinstance(exc, _UNSUPPORTED_TYPES):
        return None
    message = str(exc).lower()
    if any(marker in message for marker in _FATAL_MARKERS):
        return None
    for marker in _UNSUPPORTED_MARKERS:
        if marker in message:
            return f"{type(exc).__name__}: {exc}"
    return None


def _flash_preconditions(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Optional[str]:
    """RAVEN's own asserts, as a reason string instead of an assert.

    ``FlashAttention.forward`` asserts ``q.device.type == 'cuda'`` and
    ``q.size(-1) <= 256`` and takes ``half_dtypes = (float16, bfloat16)``. RAVEN
    would *cast* a non-half input to bf16 (its ``half()`` helper); this lane
    refuses instead and stays on SDPA, because silently attending in bf16 when
    the caller asked for fp32 is a numerical change, not a backend choice.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return f"not cuda (q on {q.device})"
    if not (q.dtype == k.dtype == v.dtype):
        return f"mixed dtypes q={q.dtype}, k={k.dtype}, v={v.dtype}"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype} is not fp16/bf16"
    if q.shape[-1] > 256:
        return f"head_dim {q.shape[-1]} > 256"
    return None


def _cu_seqlens(lengths: Sequence[int], device) -> torch.Tensor:
    """RAVEN's cumulative sequence bounds, built the same way.

    ``torch.cat([lens.new_zeros([1]), lens]).cumsum(0, dtype=torch.int32)`` on a
    CPU ``int32`` length vector, then moved to the device -- the length vector
    is built on the host in RAVEN too (``_minimax_h3_attention_core_impl``
    reads ``cu_seqlens_host``), which is what keeps a packed call from syncing
    the device once per block.
    """
    lens = torch.tensor(list(lengths), dtype=torch.int32)
    return torch.cat([lens.new_zeros([1]), lens]).cumsum(0, dtype=torch.int32).to(
        device, non_blocking=True)


def _flash_varlen(name: str, func, q, k, v, *, scale: float,
                  rows: int, kv_rows: int) -> torch.Tensor:
    """One varlen call with RAVEN's exact keyword set for that backend.

    Transcribed from ``utils/flash_attn.py::FlashAttention.forward`` on its
    packed (``q.ndim == 3``) single-document path, with RAVEN's defaults
    inlined: ``dropout_p=0.0``, ``q_scale=None``, ``causal=False``,
    ``window_size=(-1, -1)``, ``deterministic=False``. ``lq``/``lk`` are
    ``int(q_lens.cpu().max())`` there, i.e. this document's own lengths.

    The ``q.to(v.dtype)`` / ``k.to(v.dtype)`` pair is RAVEN's and is a no-op
    here (the precondition check already required one dtype); it is kept so the
    transcription stays readable against the source.
    """
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    cu_seqlens_q = _cu_seqlens([rows], q.device)
    cu_seqlens_k = _cu_seqlens([kv_rows], q.device)
    if name == "fa3":
        out = func(q=q, k=k, v=v,
                   cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                   seqused_q=None, seqused_k=None,
                   max_seqlen_q=rows, max_seqlen_k=kv_rows,
                   softmax_scale=scale, causal=False, deterministic=False)
    else:
        out = func(q=q, k=k, v=v,
                   cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                   max_seqlen_q=rows, max_seqlen_k=kv_rows,
                   dropout_p=0.0, softmax_scale=scale, causal=False,
                   window_size=(-1, -1), deterministic=False)
    if isinstance(out, (tuple, list)):
        # some flash_attn_interface builds return (out, softmax_lse); element 0
        # is the output in every one of them, so this cannot change a value
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise CausalModelError(
            f"{name} varlen attention returned {type(out).__name__}, not a tensor"
        )
    if tuple(out.shape) != tuple(q.shape):
        raise CausalModelError(
            f"{name} varlen attention returned {tuple(out.shape)}, expected "
            f"{tuple(q.shape)}"
        )
    return out.type(q.dtype)


def _sdpa_packed(q, k, v, *, scale: float) -> torch.Tensor:
    """``utils/flash_attn.py::_sdpa_varlen`` for one document, reduction off.

    ``out = torch.empty_like(q)``, one ``scaled_dot_product_attention`` on the
    **3-D** ``[heads, rows, dim]`` transpose with ``attn_mask=None``,
    ``dropout_p=0.0``, ``is_causal=False`` and the caller's ``softmax_scale``,
    transposed back and copied into the output buffer.

    ``allow_fp16_bf16_reduction_math_sdp(False)`` wraps exactly this call. The
    vr audit showed that with the reduction disabled on both sides the two
    implementations agree bit for bit; leaving it enabled makes the math kernel
    accumulate in bf16 and reintroduces the difference. The previous value is
    restored in ``finally`` -- this is a process-wide Comfy setting and must not
    be left changed, not even when the kernel raises.
    """
    getter = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
    setter = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    previous = None
    if getter is not None and setter is not None:
        previous = bool(getter())
        setter(False)
    try:
        out = torch.empty_like(q)
        segment = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ).transpose(0, 1)
        out.copy_(segment)
        return out
    finally:
        if previous is not None:
            setter(previous)


def _record_backend(backend: str, reason: str, site, rows: int, kv_rows: int) -> None:
    _ATTENTION_STATE["last"] = {"backend": backend, "reason": reason,
                                "site": site, "rows": rows, "kv_rows": kv_rows}


def _disable_attention_backend(name: str, reason: str) -> None:
    """Retire a backend that said it cannot serve this build, loudly once.

    A capability failure is not transient: it would raise on every one of the
    remaining calls in the rollout. Retiring it keeps the cost at one failed
    call, and the warning is what makes "parity ran on SDPA, not FA3" visible
    without reading a dump.
    """
    resolved = _resolve_attention_backends()
    resolved["func"].pop(name, None)
    resolved["available"][name] = False
    resolved["why"][name] = f"disabled after an unsupported error: {reason}"
    _LOG.warning("raven_streaming: %s varlen attention is unusable (%s); "
                 "falling back to the next backend", name, reason)


def raven_packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    site: Optional[Tuple[str, int]] = None,
) -> torch.Tensor:
    """One packed document through SDPA, exactly as RAVEN's fallback runs it.

    ``q`` is ``[rows, heads, dim]`` and ``k``/``v`` are ``[kv_rows, heads, dim]``
    -- the merged ``[retained | current]`` history, so ``kv_rows >= rows``. The
    return is ``[rows, heads, dim]``.

    ``site`` is a **diagnostic label only** and never touches the math:
    ``('dit', layer_idx)`` or ``('text_refiner', block_idx)``. Two different
    stacks share this seam -- the 50 DiT blocks and, during
    :meth:`RavenCausalMiniMaxH3Model.prefill_text`, the token refiner -- and a
    probe wrapping this function has no other way to tell them apart. Counting
    calls would renumber every DiT block by the refiner calls in front of it,
    so the label carries the index instead of the probe inferring it
    (``tools/probe_causal_parity.py``'s ``KVTap``). Each caller holds its label
    as a constant, so labelling costs no per-call allocation.

    Backend priority is RAVEN's: FA3 (``flash_attn_interface``), then FA2
    (``flash_attn``), then the packed SDPA transcription of
    ``utils.flash_attn._sdpa_varlen``. See "Attention backend" in the module
    docstring for what may fall back and what may not; :func:`_flash_varlen`
    and :func:`_sdpa_packed` hold the two call transcriptions.

    ``is_causal`` is False everywhere even though the lane is chunk-causal:
    visibility is expressed by *what is in the cache*, not by a triangle. RAVEN
    rejects the triangle outright on the cached path
    (``_prepare_cache_routing`` raises on a non-zero ``attn_type_map``) because
    ``q_rows != kv_rows`` makes torch's top-left-aligned triangle disagree with
    FlashAttention's bottom-right one.

    Not ``optimized_attention``: Comfy's dispatcher takes a 4-D
    ``[1, heads, rows, dim]`` batch and picks a backend by build, which the
    audit put 0.0024-0.0028 relative L2 away from this call.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise CausalModelError(
            "packed attention takes [rows, heads, dim] tensors, got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    rows, kv_rows = int(q.shape[0]), int(k.shape[0])

    blocked = _flash_preconditions(q, k, v)
    if blocked is None:
        resolved = _resolve_attention_backends()
        for name in resolved["order"]:
            func = resolved["func"].get(name)
            if func is None:
                continue
            try:
                out = _flash_varlen(name, func, q, k, v, scale=scale,
                                    rows=rows, kv_rows=kv_rows)
            except BaseException as exc:  # noqa: BLE001 - re-raised unless routing
                reason = _unsupported_reason(exc)
                if reason is None:
                    raise
                _disable_attention_backend(name, reason)
                continue
            _record_backend(name, resolved["why"][name], site, rows, kv_rows)
            return out
        blocked = "no flash backend available"

    _record_backend("sdpa", blocked, site, rows, kv_rows)
    return _sdpa_packed(q, k, v, scale=scale)


def _raven_mod_gate(x, gate, other, segments):
    """RAVEN's ``_modulate_gate``: ``(x + gate[row] * other).to(x.dtype)``.

    The difference from upstream's ``_mod_gate`` is the rounding, not the
    algebra: ``addcmul_`` computes the product and the sum in one pass, RAVEN
    rounds ``gate * other`` to the compute dtype and only then adds. On BF16
    that disagrees on ~23% of elements, and it is one of the three residual
    gaps the operator audit found.

    ``segments`` is the official ``[(start, stop, mod_row)]`` table and each
    segment is written back into ``x`` in place, exactly as ``_mod_gate`` does:
    the segments tile the rows, so the result is RAVEN's whole-tensor
    expression element for element, without materialising a per-row gate.
    """
    for a, b, row in segments:
        x[a:b] = (x[a:b] + _mod_row(gate, row, x.dtype) * other[a:b]).to(x.dtype)
    return x


def _raven_adaln_input(t_emb: torch.Tensor, dtype, *, apply_silu: bool) -> torch.Tensor:
    """The AdaLN input every block and the final layer share, RAVEN's way.

    RAVEN's model builds it once per forward, right after the time embedder::

        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)

    with ``t_emb`` still fp32 (the time embedder is an fp32 island in the
    checkpoint). ``AdalnProj.forward`` instead applies the SiLU to whatever
    dtype it is handed, and the dense forward hands it a ``t_emb`` already cast
    to the compute dtype -- a BF16 SiLU on a BF16 input, 0.0035-0.0052 away.

    ``apply_silu`` mirrors the module's own flag: the curve-form checkpoint
    folds the SiLU into its stored table and its AdaLN runs at fp32, so there is
    nothing to redo and the fp32 ``t_emb`` is passed straight through. RAVEN has
    no curve form; that branch stays the official one.
    """
    if not apply_silu:
        return t_emb
    return F.silu(t_emb.to(torch.float32)).to(dtype)


def _raven_adaln_params(adaln, adaln_input: torch.Tensor):
    """``AdalnProj.forward`` minus its SiLU: linear -> modality view -> chunk.

    ``adaln.linear`` is invoked, not ``F.linear`` on ``adaln.weight``, because
    the AdaLN projections are 51 of the RAVEN LoRA's 266 targets and the
    adapter is a forward hook. The view/chunk that follows is upstream's own,
    and is what RAVEN's ``split_output`` does too (``[M, expand*H*modalities]``
    -> ``[M*modalities, expand*H]`` -> ``expand`` chunks).
    """
    x = adaln.linear(adaln_input)
    x = x.view(x.shape[0] * adaln.modalities, adaln.expand * adaln.hidden)
    return x.chunk(adaln.expand, dim=-1)


def _raven_refiner_attention(attn, x: torch.Tensor, site: Tuple[str, int]) -> torch.Tensor:
    """The refiner's attention, RAVEN's way: no RoPE, packed 3-D SDPA.

    A replay of ``Attention.forward``'s no-RoPE branch over the block's own
    modules -- ``qkv_proj``, ``q_norm``/``k_norm``, ``out_proj``, all through
    ``__call__`` so the LoRA hooks and the offload casts fire -- with
    :func:`raven_packed_attention` in place of ``optimized_attention``. RAVEN's
    refiner block calls the same ``MiniMaxH3Attention`` as its DiT blocks with
    ``rope_cache=None``, so ``_apply_qk_norm`` alone runs and the packed varlen
    call is the same one.

    Single document, batch 1: the prompt is one text sample.
    """
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
    k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.view(s, attn.heads, attn.head_dim)
    out = raven_packed_attention(q, k, v, scale=attn.head_dim ** -0.5, site=site)
    return attn.out_proj(out.reshape(s, attn.heads * attn.head_dim))


def _materialise_rows(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous ``[rows, heads, dim]`` buffer this forward owns outright.

    Two properties, both load-bearing:

    *Layout* -- RAVEN's eager ``_apply_qk_norm``/``_apply_rope_qk`` allocate q
    and k, so its varlen kernel sees ``[heads*dim, dim, 1]``. Upstream's fused
    kernel rewrites them *inside* the 3x QKV buffer instead, leaving
    ``[3*heads*dim, dim, 1]``, and FA3 takes a different path for it.

    *Ownership* -- ``contiguous()`` returns ``self`` when the view already
    happens to be contiguous, which for a single-row chunk it is (a size-1
    leading dim makes any stride contiguous). Staging that into the cache would
    pin the whole QKV buffer, 3x the rows it holds, for the life of the record.
    So the storage is checked too, and a view that owns more than its own rows
    is copied even when its layout is already right.
    """
    owns_exactly = (
        tensor.untyped_storage().size() == tensor.numel() * tensor.element_size()
    )
    if tensor.is_contiguous():
        return tensor if owns_exactly else tensor.clone()
    return tensor.contiguous()


def _reject_attention_overrides(transformer_options: Dict[str, Any]) -> None:
    """Fail on ``transformer_options`` the packed attention cannot honour.

    ``contracts.resolve_transformer_options`` already rejects patches/wrappers
    for the whole rollout, but this is the point where honouring them would
    matter, and a direct caller of :meth:`forward_chunk` never passes through
    the contract.
    """
    if transformer_options.get("optimized_attention_override") is not None:
        raise CausalModelError(
            "transformer_options['optimized_attention_override'] is set, but the "
            "causal lane deliberately does not run comfy's optimized_attention: "
            "it reproduces RAVEN's packed SDPA call instead"
        )


class RavenCausalAttention(Attention):
    """Official ``Attention`` with a chunk KV cache in front of K/V.

    Adds exactly one attribute, the plain-int ``layer_idx``; the parameter set
    (``qkv_proj``, ``q_norm``, ``k_norm``, ``out_proj``) is the parent's, under
    the parent's names and shapes, so a bidirectional checkpoint loads unchanged.

    Ordering inside the forward is the parent's and is load-bearing:
    QKV -> QK RMSNorm -> absolute RoPE -> *then* the cache merge. What is staged
    is therefore the post-norm / post-RoPE key and the raw value, which is what
    a later chunk needs in order to attend to this one without recomputing it.

    Without a cache this **is** the parent: ``super().forward`` runs, so the
    dense lane keeps Comfy's ``optimized_attention`` and its backend, its
    ``v.clone()`` and its strides. With a cache, the projection / norm / RoPE
    prologue is still the parent's (audited bit-identical to RAVEN's
    ``_apply_qk_norm`` + ``_apply_rope_qk``), the attention call becomes
    RAVEN's :func:`raven_packed_attention`, and q/k/v are put in RAVEN's
    *memory layout* before the call -- see the comments in :meth:`forward`.

    The history is read through the cache's assembly API rather than through
    ``retained()``: one merged ``[past + current, heads, head_dim]`` buffer per
    layer, filled in place. See "The merged slot" in the module docstring for
    why, and why it makes the cache's ``storage`` invisible to the numbers.
    """

    def __init__(
        self,
        hidden,
        heads,
        head_dim,
        eps,
        *,
        layer_idx: int,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__(
            hidden, heads, head_dim, eps, dtype=dtype, device=device, operations=operations
        )
        if int(layer_idx) != layer_idx or layer_idx < 0:
            raise CausalModelError(f"layer_idx must be a non-negative int, got {layer_idx!r}")
        self.layer_idx = int(layer_idx)
        # constant probe label (see raven_packed_attention); a plain tuple
        # attribute, so it stays out of the state dict
        self._attention_site = ("dit", self.layer_idx)

    def forward(
        self,
        x,
        rope_freqs=None,
        transformer_options={},
        *,
        cache: Optional[ChunkKVCache] = None,
        update_cache: bool = False,
    ):
        if cache is None:
            if update_cache:
                raise CausalModelError("update_cache=True without a cache")
            # Dense lane: the official body verbatim, backend and all. Calling
            # the parent rather than re-forking it is what makes dense parity a
            # property of the class hierarchy instead of a diff review.
            return super().forward(
                x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )

        _reject_attention_overrides(transformer_options)

        # --- verbatim fork of Attention.forward (projection / QK norm / RoPE) ---
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        # --- end fork; the parent's ``v = v.clone()`` is deliberately not here ---

        # Match RAVEN's *memory layout* at the attention seam, not just its
        # values. A full-model FA3 run put layer-0 q/k/v at 5e-5..9e-5 relative
        # L2 with identical arithmetic, and the seam evidence was the strides:
        #
        #   RAVEN   q/k [7168, 128, 1] contiguous   v [21504, 128, 1] view
        #   here    q/k [21504, 128, 1] view        v [7168, 128, 1] cloned
        #
        # -- exactly swapped. RAVEN's ``_apply_qk_norm``/``_apply_rope_qk`` are
        # eager and return freshly allocated q/k, while its ``v`` stays the
        # fused-QKV view; upstream's fused kernel instead rewrites q/k *inside*
        # the QKV buffer and upstream clones v. FA3 is stride-sensitive, so the
        # two layouts take different kernel paths.
        #
        # q/k are therefore materialised here, before attention and before the
        # cache routing (one copy each, the same allocation RAVEN's eager norm
        # makes), and v is handed on as the view.
        q = _materialise_rows(q)
        k = _materialise_rows(k)

        spec = cache.retained_spec(self.layer_idx)
        if update_cache:
            # Staged before the merge, and before attention: a record holds this
            # chunk's own rows, computed but not yet merged with anything.
            #
            # With a host-side canonical cache this is also where the D2H
            # happens, synchronously, so the chunk's K/V are already on the host
            # when the merged slot is allocated -- the two never bid for device
            # memory at the same time, and nothing is in flight afterwards.
            #
            # ``k`` is this forward's own contiguous [rows, heads, dim] tensor
            # -- not a view of anything, and nothing writes to it after this
            # point (the merge below allocates and copies) -- so a ``gpu`` cache
            # can take it as is. That is one copy fewer than before, and it is
            # the same tensor the backend sees when there is no history.
            #
            # ``v`` is the fused-QKV view and must be copied: keeping it would
            # pin 3x its own size for the life of the record. The copy is the
            # one unavoidable allocation of this path, and it lands contiguous
            # ([rows, heads, dim], storage exactly numel), so the record stays a
            # compact owned buffer. A host-side cache copies both, so its
            # records own exactly their rows whatever was staged.
            cache.stage(self.layer_idx, k, v, copy_key=False, copy_value=True)
        if spec is not None:
            if spec.dtype != k.dtype:
                raise CausalModelError(
                    f"layer {self.layer_idx}: cached K/V are {spec.dtype} but this "
                    f"chunk is {k.dtype}; the cache must be filled and read in one "
                    "compute dtype"
                )
            if not cache.canonical_on_host and spec.device != k.device:
                # A device-resident cache and a chunk on another device is a
                # rollout that has half moved; a *host* cache is off-device by
                # design and says so through ``canonical_on_host``.
                raise CausalModelError(
                    f"layer {self.layer_idx}: cached K/V are on {spec.device} but "
                    f"this chunk is on {k.device}; a device-resident cache must be "
                    "filled and read on one device"
                )
            if (spec.heads, spec.head_dim) != tuple(k.shape[1:]):
                raise CausalModelError(
                    f"layer {self.layer_idx}: cached K head shape "
                    f"{(spec.heads, spec.head_dim)} != current {tuple(k.shape[1:])}"
                )
            # One merged slot per layer, allocated once and filled in place:
            # retained rows into the prefix (an H2D out of the canonical host
            # buffers, or a D2D out of the device ones), this chunk's own rows
            # into the tail. Nothing gathers the history into a second device
            # buffer first, so the peak is this pair and only this pair, and the
            # cache keeps no reference to either -- they die with this call.
            #
            # The result is contiguous [merged, heads, dim], which is the layout
            # ``torch.cat`` used to produce and the one RAVEN's own merge
            # produces (it scatters into ``past_keys.new_zeros([merged, n, d])``).
            merged = spec.rows + s
            merged_k = k.new_empty((merged, self.heads, self.head_dim))
            merged_v = k.new_empty((merged, self.heads, self.head_dim))
            cache.copy_retained_into(
                self.layer_idx, merged_k[:spec.rows], merged_v[:spec.rows]
            )
            merged_k[spec.rows:].copy_(k)
            merged_v[spec.rows:].copy_(v)
            k, v = merged_k, merged_v

        # RAVEN's packed call, on the merged history: q stays current-only, so
        # this is one document of ``rows`` queries over ``kv_rows`` keys. Called
        # through the module global, so a probe can wrap the module attribute
        # and see exactly what every layer hands the backend.
        #
        # The layout the backend sees is RAVEN's, per stream position:
        #   no history -- q/k contiguous, v the fused-QKV view;
        #   with history -- q contiguous, k/v contiguous out of the merge.
        out = raven_packed_attention(q, k, v, scale=self.head_dim ** -0.5,
                                     site=self._attention_site)
        return self.out_proj(out.reshape(s, self.heads * self.head_dim))


class RavenCausalDiTBlock(DiTBlock):
    """Official ``DiTBlock`` whose attention is the causal one.

    ``self.attn`` is replaced after the parent's ``__init__`` rather than the
    body being re-run: the causal attention registers the same parameters under
    the same names, so the swap is state-dict neutral.

    Two forward bodies, selected by the cache. Without one, the parent's runs
    untouched. With one, the body is RAVEN's ``CausalMiniMaxH3DiTBlock.forward``
    expressed over the official modules: the shared pre-SiLU ``adaln_input``
    goes straight into ``adaln_proj.linear`` (the module's own SiLU is skipped,
    not disabled), scale/shift stay upstream's audited-identical
    ``_mod_scale_shift``, and the two gated residuals use
    :func:`_raven_mod_gate` instead of ``addcmul_``.
    """

    def __init__(
        self,
        hidden,
        heads,
        head_dim,
        ffn,
        t_dim,
        eps,
        qk_eps,
        apply_silu=True,
        adaln_dtype=None,
        *,
        layer_idx: int,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__(
            hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
            apply_silu=apply_silu, adaln_dtype=adaln_dtype,
            dtype=dtype, device=device, operations=operations,
        )
        self.attn = RavenCausalAttention(
            hidden, heads, head_dim, qk_eps, layer_idx=layer_idx,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        *,
        cache: Optional[ChunkKVCache] = None,
        update_cache: bool = False,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if cache is None:
            if update_cache:
                raise CausalModelError("update_cache=True without a cache")
            # Dense lane: the parent's body, including ``AdalnProj.forward``'s
            # own SiLU and upstream's fused ``_mod_gate``.
            return super().forward(x, t_emb, mod_segments, rope_freqs,
                                   transformer_options=transformer_options)

        if adaln_input is None:
            raise CausalModelError(
                "the causal lane needs the shared pre-SiLU adaln_input: RAVEN "
                "computes silu(t_emb) once in fp32 per forward, and it cannot be "
                "recovered from a t_emb already cast to the compute dtype. "
                "RavenCausalMiniMaxH3Model builds it; pass it through."
            )

        # Fork of RAVEN's CausalMiniMaxH3DiTBlock.forward over Comfy's modules.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            _raven_adaln_params(self.adaln_proj, adaln_input)
        residual = x
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        h = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options,
                      cache=cache, update_cache=update_cache)
        x = _raven_mod_gate(residual, gate_msa, h, mod_segments)
        residual = x
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _raven_mod_gate(residual, gate_mlp, self.mlp(h), mod_segments)


def _official_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the official ``__init__`` arguments, defaults included.

    Read off ``MiniMaxH3Model.__init__``'s signature rather than copied, so a
    changed upstream default cannot silently desynchronise the rebuilt blocks
    from the ones the parent constructed.
    """
    signature = inspect.signature(MiniMaxH3Model.__init__)
    bound = signature.bind_partial(None, **kwargs)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    resolved.pop("self", None)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            resolved.pop(name, None)
    return resolved


class RavenCausalMiniMaxH3Model(MiniMaxH3Model):
    """Chunk-causal MiniMax H3 DiT with an exact-parity state dict.

    Construction: the official ``__init__`` runs untouched, then every block is
    replaced by its causal counterpart *one index at a time* -- the whole
    ModuleList rebuilt at once would double the block memory of a 50-layer BF16
    model for the duration of the loop. Each replacement is checked against the
    block it replaces, key for key and shape for shape, so any upstream change
    that would break checkpoint compatibility fails at construction.

    The inherited dense ``forward`` / ``_forward`` are untouched and stay usable:
    the causal kwargs are keyword-only with "no cache" defaults.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = _official_config(kwargs)

        dtype = config["dtype"]
        device = config["device"]
        operations = config["operations"]
        adaln_dtype = torch.float32 if self.use_adaln_curves else dtype
        apply_silu = not self.use_adaln_curves

        num_layers = int(config["num_layers"])
        if len(self.blocks) != num_layers:
            raise CausalModelError(
                f"official model built {len(self.blocks)} blocks for num_layers="
                f"{num_layers}; the causal rebuild cannot map them"
            )

        for index in range(num_layers):
            official_block = self.blocks[index]
            causal_block = RavenCausalDiTBlock(
                config["hidden_size"],
                config["num_attention_heads"],
                config["attention_head_dim"],
                config["ffn_hidden_size"],
                config["time_embed_dim"],
                config["norm_eps"],
                config["qk_norm_eps"],
                apply_silu=apply_silu,
                adaln_dtype=adaln_dtype,
                layer_idx=index,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            _assert_same_parameters(official_block, causal_block, index)
            self.blocks[index] = causal_block
            del official_block

    # -- helpers ---------------------------------------------------------

    def audio_sigma_from_video(self, video_sigma: float, transformer_options: Dict[str, Any] = {}) -> float:
        """Map a video sigma onto the audio stream's shifted grid (official rule)."""
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma = torch.as_tensor(float(video_sigma), dtype=torch.float32).clamp(min=1e-6)
        return float(time_shift_sigma(sigma, shift_v, shift_a))

    def _time_embeddings_native(self, unique_t: Sequence[float], device) -> torch.Tensor:
        """Fork of the dense forward's ``t_emb``, *before* the compute-dtype cast.

        Both checkpoint forms produce fp32 here: the time embedder is an fp32
        island, and the curve table is stored fp32. The dense forward casts the
        non-curve result to the compute dtype immediately (see
        :meth:`_time_embeddings`); the causal lane keeps this one so the shared
        AdaLN SiLU can run in fp32, which is what RAVEN does.
        """
        t_vals = torch.tensor(list(unique_t), dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        return self.time_embedder(t_vals)

    def _time_embeddings(self, unique_t: Sequence[float], device, dtype) -> torch.Tensor:
        """The dense forward's ``t_emb``: fp32 for the curve form, compute dtype otherwise."""
        t_emb = self._time_embeddings_native(unique_t, device)
        return t_emb if self.use_adaln_curves else t_emb.to(dtype)

    def _causal_time_embeddings(self, unique_t: Sequence[float], device, dtype):
        """``(t_emb, adaln_input)`` for one causal forward.

        ``t_emb`` is the dense lane's tensor (used when the caller passes no
        cache); ``adaln_input`` is RAVEN's shared, once-per-forward
        ``silu(t_emb)`` computed in fp32 and cast once -- see
        :func:`_raven_adaln_input`. Both come off the same native fp32
        embedding, so the time embedder runs exactly once.
        """
        native = self._time_embeddings_native(unique_t, device)
        t_emb = native if self.use_adaln_curves else native.to(dtype)
        adaln_input = _raven_adaln_input(
            native, dtype, apply_silu=not self.use_adaln_curves
        )
        return t_emb, adaln_input

    def _causal_refine_text(self, text_states: torch.Tensor, transformer_options) -> torch.Tensor:
        """``condition_proj`` + token refiner in RAVEN's numerical order.

        The official ``TokenRefiner.forward`` is not called: its blocks would
        route attention through ``optimized_attention``. Instead the refiner's
        own body is replayed over its own modules --
        ``x + attn(norm1(x))`` then ``x + mlp(norm2(x))``, then ``final_norm``
        -- which is RAVEN's ``MiniMaxH3TokenRefinerBlock.forward`` verbatim,
        with :func:`_raven_refiner_attention` supplying the packed SDPA call.

        Nothing is replaced or monkey-patched: ``self.token_refiner`` keeps its
        official ``forward`` for the dense lane, and every parameter here is
        still reached through its module's ``__call__`` (the eight refiner
        Linears are LoRA targets).
        """
        h = self.condition_proj(text_states)
        for index, block in enumerate(self.token_refiner.blocks):
            h = h + _raven_refiner_attention(block.attn, block.norm1(h),
                                             ("text_refiner", index))
            h = h + block.mlp(block.norm2(h))
        return self.token_refiner.final_norm(h)

    def _causal_final_layer(self, h, adaln_input, video_seg, audio_seg):
        """``FinalLayer.forward`` driven by RAVEN's shared pre-SiLU AdaLN input.

        The body is upstream's, with two deliberate choices:

        * ``adaln_proj.linear`` is called directly (module ``__call__``, so the
          51st AdaLN LoRA target still fires), skipping ``AdalnProj.forward``'s
          BF16 SiLU;
        * the modulation expression is upstream's own -- and for the
          non-curve checkpoint it *is* RAVEN's, because ``scale.dtype`` equals
          the compute dtype there, so both round the product and then the sum in
          BF16. For the curve form ``scale`` is fp32 and this keeps the official
          fp32 arithmetic; RAVEN has no curve form to match.

        The two heads run on their own row slices rather than on every row.
        RAVEN projects all rows and selects afterwards because its packed
        sequence carries text and condition rows it must drop; a causal chunk
        has only the two target segments, so the slices *are* the live rows.
        """
        final = self.final_layer
        shift, scale = _raven_adaln_params(final.adaln_proj, adaln_input)

        def mod(seg):
            a, b, row = seg
            return (final.norm(h[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                    + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        return final.video_out(mod(video_seg)), final.audio_out(mod(audio_seg))

    def _rope_table(self, position_ids: torch.Tensor, device, dtype) -> torch.Tensor:
        return rope_rotation_table(self.rope_freqs(position_ids, device), dtype)

    def _run_blocks(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments,
        rope_freqs: torch.Tensor,
        transformer_options: Dict[str, Any],
        cache: Optional[ChunkKVCache],
        update_cache: bool,
        adaln_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The official block loop, prefetch queue included (CPU offload path).

        ``adaln_input`` is RAVEN's shared ``silu(t_emb)``, built once by the
        caller and handed to all 50 blocks -- the same tensor object, exactly as
        RAVEN's forward does it. It is ignored by a cache-free (dense-lane)
        block, which re-derives its own from ``t_emb``.

        The dense forward's ``patches_replace['dit']`` hook is deliberately not
        honoured here: a replacement block receives the official argument dict
        and would run without the cache, silently turning a cached chunk into a
        context-free one. Block replacement stays available on the dense path.
        """
        device = h.device
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
            list(self.blocks), device, transformer_options
        )
        try:
            for block in self.blocks:
                comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
                h = block(h, t_emb, mod_segments, rope_freqs,
                          transformer_options=transformer_options,
                          cache=cache, update_cache=update_cache,
                          adaln_input=adaln_input)
            if prefetch_queue is not None:
                comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        except BaseException:
            if cache is not None and update_cache:
                cache.discard_pending()
            raise
        return h

    # -- causal API ------------------------------------------------------

    @torch.no_grad()
    def prefill_text(
        self,
        context: torch.Tensor,
        *,
        cache: ChunkKVCache,
        transformer_options: Dict[str, Any] = {},
        text_token_tags: Optional[torch.Tensor] = None,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> int:
        """Write the text rows into the cache as chunk 0, alone.

        Text gets its own cached forward: folding it into the first media chunk
        would let text rows attend media rows, and the cached text keys would
        stop being the ones every later chunk assumes.

        ``context`` is ``[1, L, text_dim]`` Qwen states, or ``[1, L, hidden]``
        when the refiner already ran (Comfy's ``extra_conds`` pre-runs it once
        per sampling). ``text_token_tags`` is the official per-token modality
        tag vector; without it every text row carries the text tag.

        Returns the number of text rows written.
        """
        if cache.committed_chunks != 0:
            raise CausalModelError(
                f"prefill_text into a cache that already holds "
                f"{cache.committed_chunks} chunk(s); the text is chunk 0"
            )
        if context.ndim != 3 or context.shape[0] != 1:
            raise CausalModelError(
                f"context must be [1, L, dim], got {tuple(context.shape)}"
            )

        device = context.device
        dtype = compute_dtype if compute_dtype is not None else context.dtype
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            # RAVEN casts the encoder rows to the compute dtype *before*
            # condition_proj (``refine_prompt_embeds``); in production the two
            # dtypes already agree, so this is a no-op there.
            text_states = self._causal_refine_text(text_states.to(dtype), transformer_options)
        h = text_states.to(dtype)
        if h.data_ptr() == context.data_ptr():
            # The block stack accumulates its residuals into this buffer (both
            # lanes do -- upstream's ``_mod_gate`` in place, RAVEN's gate
            # written back per segment). ``.to()`` is a no-op when the caller
            # already handed us the compute dtype, which for a pre-refined
            # ``context`` means the buffer *is* theirs: without this copy the
            # first block would overwrite the caller's conditioning, and the
            # damage would only show on the next rollout that reuses it.
            h = h.clone()
        text_len = int(h.shape[0])

        positions = layout_mod.text_position_ids(text_len)
        rope_freqs = self._rope_table(positions, device, dtype)
        t_emb, adaln_input = self._causal_time_embeddings(
            [_fp32_scalar(CLEAN_TIMESTEP_TEXT)], device, dtype)
        mod_segments = _text_mod_segments(text_len, text_token_tags)

        self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options,
                         cache, True, adaln_input=adaln_input)
        cache.commit(role="text")
        return text_len

    @torch.no_grad()
    def forward_chunk(
        self,
        *,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        layout: layout_mod.T2VALayout,
        chunk_index: int,
        cache: Optional[ChunkKVCache] = None,
        role: str = "noise",
        video_sigma: Optional[float] = None,
        audio_sigma: Optional[float] = None,
        video_eps: Optional[torch.Tensor] = None,
        audio_eps: Optional[torch.Tensor] = None,
        update_cache: Optional[bool] = None,
        transformer_options: Dict[str, Any] = {},
        compute_dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """One cached forward over chunk ``chunk_index`` only.

        ``role='noise'``
            ``video_latent``/``audio_latent`` are the chunk's ``x_t`` at the
            given repo sigmas (the two streams run independent shifted grids).
            The cache is read-only.

        ``role='clean'``
            ``video_latent``/``audio_latent`` are the chunk's ``x0``; both
            streams are re-noised to ``0.999 * x0 + 0.001 * eps`` with the
            caller's ``eps`` -- in fp32, as RAVEN does it -- and the pass exists
            for the K/V it writes.

        Both roles take the latents in any float dtype: the rows enter through
        the fp32 patch projections (``patchify_video(x.to(torch.float32))``),
        and everything from the block stack onwards runs in ``compute_dtype``,
        so the cached K/V stay BF16 whatever the caller hands in.

        Returns ``[video_velocity, audio_velocity]`` in **native H3** sign,
        shaped like the chunk's latents (``[1, 24, t, H, W]`` and
        ``[1, 32, 2, a]``) and in **fp32** -- the output heads' own dtype, not
        the input latents'. See "Returned velocity" in the module docstring:
        rounding it back would put one bf16 ULP (~1.7e-3 relative) on every
        value the sampler consumes and is the one thing that kept the tiny
        cross-implementation ``x0`` off 0.0. ``role='clean'`` returns them too;
        the sampler is free to drop them.
        """
        chunk = _resolve_chunk(layout, chunk_index)
        role = str(role)
        if role not in ("noise", "clean"):
            raise CausalModelError(f"role must be 'noise' or 'clean', got {role!r}")
        if update_cache is None:
            update_cache = role == "clean"
        if update_cache and role != "clean":
            raise CausalModelError(
                "only a clean chunk may write the cache: context K/V must carry "
                "the 0.999 condition timestep, not a noisy one"
            )
        if update_cache and cache is None:
            raise CausalModelError("update_cache=True without a cache")

        _check_chunk_latents(video_latent, audio_latent, layout, chunk)

        if role == "noise":
            if video_eps is not None or audio_eps is not None:
                raise CausalModelError("eps is only used by a clean chunk")
            if video_sigma is None or audio_sigma is None:
                raise CausalModelError(
                    "a noise chunk needs both video_sigma and audio_sigma "
                    "(the streams run independent shifted grids)"
                )
            # ``1 - sigma`` in fp32, which is where RAVEN evaluates it
            t_v = _fp32_one_minus(video_sigma)
            t_a = _fp32_one_minus(audio_sigma)
            video_x, audio_x = video_latent, audio_latent
        else:
            if video_sigma is not None or audio_sigma is not None:
                raise CausalModelError(
                    "a clean chunk carries repo t = 0; it takes no sigma"
                )
            if video_eps is None or audio_eps is None:
                raise CausalModelError(
                    "a clean chunk needs video_eps and audio_eps: its rows are "
                    "0.999 * x0 + 0.001 * eps, and a later pass must be able to "
                    "reproduce exactly what the cache saw"
                )
            if video_eps.shape != video_latent.shape or audio_eps.shape != audio_latent.shape:
                raise CausalModelError(
                    "eps must match its stream's chunk latent shape: "
                    f"{tuple(video_eps.shape)} vs {tuple(video_latent.shape)}, "
                    f"{tuple(audio_eps.shape)} vs {tuple(audio_latent.shape)}"
                )
            # RAVEN's ``clean_by_tag`` is an fp32 tensor, so the condition
            # timestep the embedder and the mix see is the fp32 0.999.
            #
            # This rounding and the ``_fp32_one_minus`` below are one fix in two
            # halves and neither is redundant: ``1 - t`` is exactly representable
            # once ``t`` is an fp32 number (Sterbenz), so with the rounding here
            # a double subtraction would land on the same bits -- but with the
            # raw ``0.999`` double it does not, and that is the old behaviour.
            # Dropping either half alone is invisible; dropping both regresses.
            t_v = _fp32_scalar(CLEAN_TIMESTEP_VIDEO)
            t_a = _fp32_scalar(CLEAN_TIMESTEP_AUDIO)
            # In fp32, because RAVEN's is. ``MiniMaxH3X0Model.forward`` mixes
            # with ``t = h3_t.view(1, -1, 1)``, an fp32 tensor, so
            # ``t * x + (1 - t) * eps`` promotes BF16 rows to fp32 and the
            # result is never rounded back -- ``_embed`` casts to fp32 anyway.
            # A python float does not promote, so spelling it the obvious way
            # would round the augmented rows to BF16 and cost ~3e-3 relative L2
            # on every cached context row (measured against the RAVEN harness).
            # ``(1 - t)`` is RAVEN's second fp32 subtraction: it evaluates
            # ``t * x + (1 - t) * eps`` with ``t`` an fp32 tensor, so the eps
            # coefficient is ``1 - fp32(0.999) = 0.00099998713`` and not the
            # ``fp32(1.0 - 0.999) = 0.00100000005`` the double spelling gives --
            # 1.3e-5 of each other, i.e. ~1.3e-8 of the mixed row.
            video_x = (t_v * video_latent.to(torch.float32)
                       + _fp32_one_minus(t_v) * video_eps.to(torch.float32))
            audio_x = (t_a * audio_latent.to(torch.float32)
                       + _fp32_one_minus(t_a) * audio_eps.to(torch.float32))

        if cache is not None:
            expected = chunk_index + 1  # text is cache chunk 0
            if cache.committed_chunks != expected:
                raise CausalModelError(
                    f"chunk {chunk_index} expects {expected} committed cache chunk(s) "
                    f"(text + {chunk_index} clean fills), found {cache.committed_chunks}"
                )
            if cache.has_pending:
                raise CausalModelError(
                    "cache has a partially staged chunk; a previous forward did not commit"
                )

        device = video_latent.device
        dtype = compute_dtype if compute_dtype is not None else self.dtype
        if dtype is None:
            dtype = video_latent.dtype

        # embed: audio rows first, video rows second (RAVEN's chunk order)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        video_embed = self.video_patch_proj(video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows).to(dtype)
        h = torch.cat((audio_embed, video_embed), dim=0)

        audio_n = chunk.audio_rows
        rows = chunk.rows
        if h.shape[0] != rows:
            raise CausalModelError(
                f"chunk {chunk_index} embedded {h.shape[0]} rows, layout says {rows}"
            )

        # unique timesteps -> t_emb rows, each expanding to 3 AdaLN modality rows
        unique_t = sorted({t_v, t_a})
        t_row = {value: index for index, value in enumerate(unique_t)}
        mod_segments = [
            (0, audio_n, t_row[t_a] * 3 + layout_mod.AUDIO_TAG),
            (audio_n, rows, t_row[t_v] * 3 + layout_mod.VIDEO_TAG),
        ]
        t_emb, adaln_input = self._causal_time_embeddings(unique_t, device, dtype)

        positions = layout.chunk_position_ids(chunk_index)
        rope_freqs = self._rope_table(positions, device, dtype)

        h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options,
                             cache, bool(update_cache), adaln_input=adaln_input)
        if update_cache:
            # once, after the last layer staged: every layer must evict against
            # the same history, so the step advances here and nowhere else
            cache.commit(role="clean")

        video_seg = (audio_n, rows, t_row[t_v])
        audio_seg = (0, audio_n, t_row[t_a])
        if cache is None:
            # No cache means no causal lane at all (see the class docstring):
            # this forward is the dense computation restricted to one chunk, so
            # its head stays the official one.
            video_out, audio_out = self.final_layer(h, t_emb, video_seg, audio_seg)
        else:
            video_out, audio_out = self._causal_final_layer(
                h, adaln_input, video_seg, audio_seg)

        video_velocity = unpatchify_video(
            video_out, chunk.video_latents, layout.latent_h // 2, layout.latent_w // 2,
            self.latents_dim, self.patch_size,
        )
        audio_velocity = unpack_audio(audio_out)
        # fp32 out, deliberately: see "Returned velocity" in the module docstring.
        # ``unpatchify_video`` / ``unpack_audio`` are pure reshape + permute, so
        # this is the head's own fp32 output, unrounded.
        return [video_velocity, audio_velocity]


# --- construction / validation helpers --------------------------------------


def _assert_same_parameters(official: nn.Module, causal: nn.Module, index: int) -> None:
    """Fail loudly when the causal block is not state-dict identical."""
    official_shapes = {k: tuple(v.shape) for k, v in official.state_dict().items()}
    causal_shapes = {k: tuple(v.shape) for k, v in causal.state_dict().items()}
    if official_shapes != causal_shapes:
        missing = sorted(set(official_shapes) - set(causal_shapes))
        extra = sorted(set(causal_shapes) - set(official_shapes))
        changed = sorted(
            k for k in set(official_shapes) & set(causal_shapes)
            if official_shapes[k] != causal_shapes[k]
        )
        raise CausalModelError(
            f"block {index}: causal rebuild is not state-dict neutral "
            f"(missing={missing}, unexpected={extra}, reshaped={changed})"
        )


def _text_mod_segments(text_len: int, text_token_tags: Optional[torch.Tensor]):
    """Text AdaLN segments: one run per tag, at the single text ``t_emb`` row."""
    if text_token_tags is None:
        return [(0, text_len, layout_mod.TEXT_TAG)]
    tags = text_token_tags.view(-1).tolist()
    if len(tags) != text_len:
        raise CausalModelError(
            f"text_token_tags covers {len(tags)} rows, context has {text_len}"
        )
    segments = []
    run_start = 0
    for i in range(1, text_len + 1):
        if i == text_len or tags[i] != tags[run_start]:
            segments.append((run_start, i, int(tags[run_start])))
            run_start = i
    return segments


def _resolve_chunk(layout: layout_mod.T2VALayout, chunk_index: int) -> layout_mod.Chunk:
    if not isinstance(layout, layout_mod.T2VALayout):
        raise CausalModelError(
            f"layout must be a T2VALayout, got {type(layout).__name__}"
        )
    if not (0 <= chunk_index < layout.num_chunks):
        raise CausalModelError(
            f"chunk_index {chunk_index} outside [0, {layout.num_chunks})"
        )
    return layout.chunks[chunk_index]


def _check_chunk_latents(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    layout: layout_mod.T2VALayout,
    chunk: layout_mod.Chunk,
) -> None:
    if video_latent.ndim != 5 or video_latent.shape[0] != 1:
        raise CausalModelError(
            f"video latent must be [1, C, t, H, W], got {tuple(video_latent.shape)}"
        )
    if audio_latent.ndim != 4 or audio_latent.shape[0] != 1:
        raise CausalModelError(
            f"audio latent must be [1, C, 2, t], got {tuple(audio_latent.shape)}"
        )
    expected_video = (chunk.video_latents, layout.latent_h, layout.latent_w)
    if tuple(video_latent.shape[2:]) != expected_video:
        raise CausalModelError(
            f"chunk {chunk.index} video latent {tuple(video_latent.shape[2:])} != "
            f"layout {expected_video}"
        )
    if layout.latent_h % 2 or layout.latent_w % 2:
        raise CausalModelError(
            f"latent grid {layout.latent_h}x{layout.latent_w} is not a multiple of "
            "the 2x2 DiT patch; the causal lane does not pad"
        )
    expected_audio = (layout_mod.AUDIO_CHANNELS, chunk.audio_latents)
    if tuple(audio_latent.shape[2:]) != expected_audio:
        raise CausalModelError(
            f"chunk {chunk.index} audio latent {tuple(audio_latent.shape[2:])} != "
            f"layout {expected_audio}"
        )
    if video_latent.device != audio_latent.device:
        raise CausalModelError("video and audio chunk latents are on different devices")
