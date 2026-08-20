#!/usr/bin/env python3
"""Probe: does the M2 causal lane behave like the model it subclasses, and like RAVEN?

Three modes.

``tiny`` (runs anywhere, CPU, seconds)
    Builds a tiny official ``MiniMaxH3Model`` and the tiny
    ``RavenCausalMiniMaxH3Model`` from *identical* random weights and checks:

    1. the state dict is identical (keys, shapes, dtypes, order);
    2. the inherited **dense** forward is bit-identical to the official one;
    3. the chunk positions agree with the official ``PackedLayout`` rows;
    4. every layer's attention consumes exactly ``retained_rows + chunk_rows``
       keys -- the cache merge is observed at the attention call, not assumed;
    5. retained history changes the next chunk's output (visibility), and an
       evicted chunk provably does not (eviction);
    6. the repo clean timestep maps to H3 ``0.999`` with the
       ``0.999 * x0 + 0.001 * eps`` augmentation;
    7. a repeated rollout is bit-identical (determinism).

``inputs`` (no model, no ComfyUI weights)
    Writes the **shared input file** both sides read, and optionally a random
    checkpoint for the ``tiny`` architecture. Nothing else in this lane is
    allowed to invent its own tensors: two processes cannot be trusted to draw
    the same numbers from a seed, so the numbers are drawn once and shipped.

``real``
    Loads a real (or tiny) checkpoint into the causal model, runs the chunked
    rollout on the shared inputs, taps every layer's attention, and can

    * ``--emit-dump`` the per-layer K/V and the per-chunk x0 in
      :data:`KV_DUMP_SCHEMA`, and/or
    * ``--compare-dump`` against a dump produced by
      ``tools/raven_parity_harness.py`` on a RAVEN checkout.

What decides a comparison
-------------------------
The **gate** is scale-free and whole-tensor:

* block 0 of every forward: ``rel_l2 <= 0.01`` and ``cosine >= 0.999``;
* per-chunk ``video_x0`` / ``audio_x0``: ``rel_l2 <= 0.03`` and ``cosine >= 0.999``.

Everything else is a **diagnostic**: deeper blocks (fifty bf16 layers of
accumulation), the element-wise ``allclose`` and the per-call statistics are
measured, printed and stored, but they do not decide the run. Element-wise
``max_rel`` in particular is reported and never gated on -- one reference
element at 1e-12 turns two bf16 ULP into a relative error of 1e8.

When a gate fails, ``tools/probe_causal_operator_parity.py`` is the next step:
it audits one block's operators stage by stage against the same RAVEN code.

Two-process protocol (never two 66 GB models in one process)::

    # 1. once, on either box
    python tools/probe_causal_parity.py --mode inputs --arch full \\
        --frames 39 --width 512 --height 288 --text-len 128 \\
        --emit-inputs inputs.pt

    # 2. on the RAVEN box (/root/Jarvis), RAVEN process, no ComfyUI
    python tools/raven_parity_harness.py --raven-root /root/Jarvis \\
        --weights h3_bf16.safetensors --inputs inputs.pt \\
        --emit-dump raven_kv.pt --json raven.json

    # 3. on the ComfyUI box, Comfy process, no RAVEN
    python tools/probe_causal_parity.py --mode real --arch full \\
        --dit h3_bf16.safetensors --inputs inputs.pt \\
        --compare-dump raven_kv.pt --json parity.json

Both sides load the **same runtime-layout checkpoint** (fused QKV, no
reinterleave) and neither applies a LoRA: this is base-vs-base parity of the
causal mechanism, not a RAVEN-adapter comparison. Both run **bf16** -- RAVEN's
vendored blocks hard-cast modulation to bf16, so no other placement is even
runnable there.

Status
------
``tiny`` and ``inputs`` run anywhere. ``real`` has been exercised end to end
against ``tools/raven_parity_harness.py`` on the **tiny** architecture, CPU,
bf16, 39 frames / 512x288 / 128 text rows / sink 2 / window 2: every gate holds
(worst block-0 ``rel_l2`` 9.8e-3 on Q, outputs at 1.7e-3).

The **full 50-block BF16** run has been done by the orchestrator and does *not*
pass this gate: block 0 sits at ~0.5-1% ``rel_l2`` on Q/K/V and smooths out over
the stack, while chunk-0 ``video_x0`` lands at ``rel_l2`` 9% with cosine
0.99596 and ``audio_x0`` at 2.56% with cosine 0.99969. So audio clears the 3%
output budget and video does not. That is the recorded state of M2 parity; the
numbers are what the operator probe is for, not something to widen the gate
around.

Environment: ``COMFYUI_PATH`` / ``COMFYUI_UPSTREAM_PATH`` locate the checkout,
``RAVEN_PROBE_DEVICE`` sets the default device.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Per-layer K/V dump. Both producers write this exact structure; the
#: comparator reads nothing else. Tensor layout is canonical
#: ``[rows, heads, head_dim]`` so a Comfy ``[1, H, S, D]`` container and a RAVEN
#: varlen ``[S, H, D]`` pack land in the same frame.
#:
#: Schema note (additive, version stays 2): ``entries`` has always meant *DiT
#: block* attention calls and still does -- key for key, layer for layer -- so a
#: RAVEN dump produced before the causal lane moved to RAVEN's packed attention
#: is still a valid reference and does not need regenerating. What is new is the
#: optional ``refiner_entries`` list on the Comfy side: the token refiner now
#: shares the DiT's attention seam, and its calls are separated out here rather
#: than being allowed to shift the DiT layer numbering. The comparator never
#: reads ``refiner_entries`` (RAVEN's refiner runs through a wrapper the harness
#: does not tap, so there is nothing on the other side to compare it against),
#: and a consumer that does not know the key simply ignores it.
KV_DUMP_SCHEMA: Dict[str, Any] = {
    "version": 2,
    "producer": "'comfy' or 'raven'",
    "entries": "list, one per DiT block attention call, in forward order then block order",
    "refiner_entries": "optional, comfy only: stats-only token-refiner calls, "
                       "forward '<name>:refiner'. Diagnostic; never compared",
    "entry": {
        "forward": "'text' | 'chunk<i>:noise' | 'chunk<i>:clean'",
        "layer": "int block index inside that forward, read off the call's own label",
        "stats": "{'q'|'k'|'v': {shape, dtype, mean, std, absmax, sum}} always present",
        "q": "[q_rows, heads, head_dim] float32 CPU, only for --kv-layers",
        "k": "[kv_rows, heads, head_dim] float32 CPU, merged [retained | current]",
        "v": "same shape as k",
        "row_stride": "int: rows were subsampled by this stride before storing",
    },
    "outputs": "list, one per noise chunk: {'video_x0': [1,C,t,H,W], 'audio_x0': [1,Ca,2,a]}",
    "meta": "request / layout / cache / timestep / kv-selection settings",
}

#: Shared input file. Drawn once, read by both processes: two torch builds are
#: not guaranteed to produce the same numbers from the same seed, and a silent
#: input difference would look exactly like a parity failure.
INPUTS_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "request": "frames/width/height/text_len/seed/sink/window/sigmas/dims/arch",
    "context": "[1, L, text_dim] float32 CPU text states (random stand-in for the encoder)",
    "video_xt": "[1, C, T, H, W] float32 CPU, the noise-chunk input for the whole clip",
    "audio_xt": "[1, Ca, 2, A] float32 CPU",
    "video_x0": "[1, C, T, H, W] float32 CPU, the clean-fill content",
    "audio_x0": "[1, Ca, 2, A] float32 CPU",
    "video_eps": "[1, C, T, H, W] float32 CPU, the clean-fill augmentation noise",
    "audio_eps": "[1, Ca, 2, A] float32 CPU",
}

#: Element-wise tolerances, kept for the ``allclose`` **diagnostic** only. They
#: are no longer a gate: one bf16 ULP on one element of a 100M-element tensor is
#: not a parity failure, and letting it decide the run hid the metrics that are.
DEFAULT_ATOL = 8e-3
DEFAULT_RTOL = 2e-2

#: The semantic gate. Deliberately conservative on the first pass:
#:
#: * block 0 is the shallowest measurement, so it is still about *operators*.
#:   The measured 50-block BF16 run sits at rel_l2 ~0.5-1% there, i.e. right at
#:   this budget -- which is the point: it may not be loosened silently.
#: * the outputs are what the sampler consumes. The measured run is at 9%
#:   rel_l2 on ``video_x0`` (cosine 0.99596) and 2.56% on ``audio_x0``
#:   (cosine 0.99969), so this budget **fails video today**. That is the honest
#:   state of M2 parity, not something to tune away.
LAYER0_REL_L2_MAX = 0.01
OUTPUT_REL_L2_MAX = 0.03
COSINE_MIN = 0.999

#: Architectures both sides understand. ``full`` is the released non-pruned H3.
TINY_CONFIG: Dict[str, Any] = dict(
    hidden_size=32,
    num_layers=3,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=24,
    ffn_hidden_size=64,
    latents_dim=24,
    audio_latents_dim=32,
    text_dim=16,
    timestep_input_dim=16,
    time_embed_hidden_size=32,
    time_embed_dim=32,
    rope_inv_freq_len=4,
)

FULL_CONFIG: Dict[str, Any] = dict(
    hidden_size=5376,
    num_layers=50,
    token_refiner_num_layers=2,
    num_attention_heads=56,
    attention_head_dim=128,
    ffn_hidden_size=14336,
    latents_dim=24,
    audio_latents_dim=32,
    text_dim=5120,
    timestep_input_dim=256,
    time_embed_hidden_size=5376,
    time_embed_dim=2688,
    rope_inv_freq_len=16,
)

ARCHS: Dict[str, Dict[str, Any]] = {"tiny": TINY_CONFIG, "full": FULL_CONFIG}


# --- reporting ---------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    #: ``False`` marks a *diagnostic*: it is measured, printed and stored, but it
    #: does not decide the run. Element-wise ``allclose`` on deep-layer K/V is
    #: the motivating case -- one bf16 ULP on one element out of millions is not
    #: a parity failure, and treating it as one hides the metrics that are.
    gate: bool = True

    def line(self) -> str:
        if self.gate:
            status = "PASS" if self.passed else "FAIL"
        else:
            status = "ok/diag" if self.passed else "DIFF/diag"
        extra = ", ".join(f"{k}={v}" for k, v in self.detail.items())
        return f"[{status}] {self.name}" + (f"  ({extra})" if extra else "")


@dataclass
class Report:
    mode: str
    device: str
    checks: List[Check] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    #: per-forward / per-layer metric tables, for reading rather than gating
    metrics: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.gate)

    @property
    def diagnostics_failed(self) -> int:
        return sum(1 for check in self.checks if not check.gate and not check.passed)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        print(check.line())
        return check

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append({"name": name, "reason": reason})
        print(f"[SKIP] {name}  ({reason})")

    def to_json(self) -> Dict[str, Any]:
        gating = [c for c in self.checks if c.gate]
        return {
            "mode": self.mode,
            "device": self.device,
            "passed": self.passed,
            "gate_summary": {
                "gating_checks": len(gating),
                "gating_failed": sum(1 for c in gating if not c.passed),
                "diagnostics": len(self.checks) - len(gating),
                "diagnostics_failed": self.diagnostics_failed,
            },
            "checks": [asdict(check) for check in self.checks],
            "skipped": self.skipped,
            "metrics": self.metrics,
            "meta": self.meta,
            "kv_dump_schema": KV_DUMP_SCHEMA,
            "inputs_schema": INPUTS_SCHEMA,
        }


# --- metrics -----------------------------------------------------------------
#
# ``max_rel`` (element-wise |a-b|/|b|) is unusable as a gate: one reference
# element at 1e-12 turns 2 bf16 ULP into a relative error of 1e8, which says
# nothing about the tensor. The gate therefore runs on whole-tensor quantities
# that are stable near zero:
#
#   rel_l2  = ||a - b||_2 / ||b||_2          (scale-free, dominated by real mass)
#   cosine  = <a, b> / (||a|| ||b||)         (direction only, ignores gain)
#   max_abs / absmax(ref), p99(|d|) / absmax(ref)
#                                            (element-wise, but normalised by the
#                                             tensor's own scale rather than by
#                                             whatever the smallest element was)

#: Above this element count the p99 is taken on a deterministic subsample; a
#: full sort of a 100M-element tensor is not worth the seconds it costs.
P99_MAX_ELEMENTS = 1_000_000


def tensor_metrics(ours, ref, *, p99_max_elements: int = P99_MAX_ELEMENTS) -> Dict[str, Any]:
    """Scale-free agreement metrics between two tensors, computed in float64.

    Degenerate cases are defined, not approximated: two all-zero tensors agree
    perfectly (``rel_l2 = 0``, ``cosine = 1``); a zero reference against a
    non-zero candidate disagrees completely (``rel_l2 = inf``, ``cosine = 0``).
    """
    import torch

    a = ours.detach().to(torch.float64).reshape(-1)
    b = ref.detach().to(torch.float64).reshape(-1)
    if a.shape != b.shape:
        return {
            "shape_mismatch": [list(ours.shape), list(ref.shape)],
            "rel_l2": float("inf"),
            "cosine": 0.0,
        }

    diff = (a - b).abs()
    ref_norm = float(b.norm())
    our_norm = float(a.norm())
    ref_absmax = float(b.abs().max()) if b.numel() else 0.0
    max_abs = float(diff.max()) if diff.numel() else 0.0

    if ref_norm == 0.0 and our_norm == 0.0:
        rel_l2, cosine = 0.0, 1.0
    elif ref_norm == 0.0:
        rel_l2, cosine = float("inf"), 0.0
    elif our_norm == 0.0:
        rel_l2, cosine = float(diff.norm() / ref_norm), 0.0
    else:
        rel_l2 = float(diff.norm() / ref_norm)
        cosine = float(torch.dot(a, b) / (our_norm * ref_norm))
        cosine = max(-1.0, min(1.0, cosine))

    if diff.numel() == 0:
        p99 = 0.0
        subsampled = False
    elif diff.numel() > p99_max_elements:
        # deterministic stride, not an RNG: two processes must agree
        stride = (diff.numel() + p99_max_elements - 1) // p99_max_elements
        p99 = float(torch.quantile(diff[::stride].contiguous(), 0.99))
        subsampled = True
    else:
        p99 = float(torch.quantile(diff, 0.99))
        subsampled = False

    scale = ref_absmax if ref_absmax > 0.0 else float("nan")
    return {
        "rel_l2": rel_l2,
        "cosine": cosine,
        "max_abs": max_abs,
        "p99_abs": p99,
        "max_abs_over_ref_absmax": (max_abs / scale) if ref_absmax > 0.0 else 0.0,
        "p99_abs_over_ref_absmax": (p99 / scale) if ref_absmax > 0.0 else 0.0,
        "ref_absmax": ref_absmax,
        "ref_l2": ref_norm,
        "elements": int(a.numel()),
        "p99_subsampled": subsampled,
    }


def metrics_pass(metrics: Dict[str, Any], *, rel_l2_max: float, cos_min: float) -> bool:
    """The gate: relative L2 under budget *and* direction preserved."""
    if "shape_mismatch" in metrics:
        return False
    import math

    rel_l2 = metrics["rel_l2"]
    cosine = metrics["cosine"]
    if math.isnan(rel_l2) or math.isnan(cosine):
        return False
    return rel_l2 <= rel_l2_max and cosine >= cos_min


def runtime_meta() -> Dict[str, Any]:
    """Everything about the numerics environment that can move a comparison.

    TF32, the matmul precision knob and the SDP kernel switches change results
    by more than the differences this lane is trying to measure, so a dump that
    does not carry them cannot be compared against another one.
    """
    import torch

    meta: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": None,
        "float32_matmul_precision": None,
        "tf32_matmul": None,
        "tf32_cudnn": None,
        "sdp": {},
        "device_name": None,
        "device_capability": None,
    }
    with contextlib.suppress(Exception):
        meta["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    with contextlib.suppress(Exception):
        meta["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    with contextlib.suppress(Exception):
        meta["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    with contextlib.suppress(Exception):
        meta["cudnn"] = torch.backends.cudnn.version()
    for name, getter in (
        ("flash", "flash_sdp_enabled"),
        ("math", "math_sdp_enabled"),
        ("mem_efficient", "mem_efficient_sdp_enabled"),
        ("cudnn", "cudnn_sdp_enabled"),
    ):
        with contextlib.suppress(Exception):
            meta["sdp"][name] = bool(getattr(torch.backends.cuda, getter)())
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            meta["device_name"] = torch.cuda.get_device_name(0)
            meta["device_capability"] = list(torch.cuda.get_device_capability(0))
    return meta


# --- shared input protocol ---------------------------------------------------


def build_shared_inputs(
    *,
    frames: int,
    width: int,
    height: int,
    text_len: int,
    seed: int,
    sink: int,
    window: Optional[int],
    video_sigma: float,
    audio_sigma: Optional[float],
    arch: str,
) -> Dict[str, Any]:
    """Draw every tensor both sides consume, once.

    ``audio_sigma`` defaults to the official shifted mapping of ``video_sigma``
    (12.0 -> 3.0), which is what a real rollout uses; it is stored explicitly so
    neither side has to re-derive it.
    """
    import torch

    from raven_streaming.layout import T2VALayout

    config = ARCHS[arch]
    layout = T2VALayout.from_request(text_len=text_len, frames=frames, width=width,
                                     height=height)
    if audio_sigma is None:
        audio_sigma = _shifted_audio_sigma(video_sigma)

    generator = torch.Generator().manual_seed(int(seed))

    def draw(shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    video_shape = layout.video_latent_shape(config["latents_dim"])
    audio_shape = layout.audio_latent_shape(config["audio_latents_dim"])
    return {
        "version": INPUTS_SCHEMA["version"],
        "request": {
            "frames": int(frames),
            "width": int(width),
            "height": int(height),
            "text_len": int(text_len),
            "seed": int(seed),
            "sink": int(sink),
            "window": None if window is None else int(window),
            "video_sigma": float(video_sigma),
            "audio_sigma": float(audio_sigma),
            "arch": arch,
            "latents_dim": int(config["latents_dim"]),
            "audio_latents_dim": int(config["audio_latents_dim"]),
            "text_dim": int(config["text_dim"]),
            "latent_t": layout.latent_t,
            "latent_h": layout.latent_h,
            "latent_w": layout.latent_w,
            "audio_t": layout.audio_t,
            "num_chunks": layout.num_chunks,
        },
        "context": draw((1, text_len, config["text_dim"])),
        "video_xt": draw(video_shape),
        "audio_xt": draw(audio_shape),
        "video_x0": draw(video_shape),
        "audio_x0": draw(audio_shape),
        "video_eps": draw(video_shape),
        "audio_eps": draw(audio_shape),
    }


def _shifted_audio_sigma(video_sigma: float, shift_v: float = 12.0, shift_a: float = 3.0) -> float:
    """``time_shift_sigma`` in closed form, without importing ComfyUI."""
    base = video_sigma / (shift_v + video_sigma * (1.0 - shift_v))
    return shift_a * base / (1.0 + (shift_a - 1.0) * base)


def save_inputs(path: str, inputs: Dict[str, Any]) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, path)


def load_inputs(path: str) -> Dict[str, Any]:
    import torch

    inputs = torch.load(path, map_location="cpu", weights_only=False)
    if int(inputs.get("version", -1)) != INPUTS_SCHEMA["version"]:
        raise SystemExit(
            f"{path}: inputs schema version {inputs.get('version')} != "
            f"{INPUTS_SCHEMA['version']}"
        )
    for key in ("context", "video_xt", "audio_xt", "video_x0", "audio_x0",
                "video_eps", "audio_eps"):
        if key not in inputs:
            raise SystemExit(f"{path}: shared inputs are missing {key!r}")
    return inputs


# --- weight IO (no ComfyUI, so the RAVEN side can reuse it) ------------------


def save_state_dict(path: str, state: Dict[str, Any]) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in state.items()}, path)
    else:
        torch.save(state, path)


def load_state_dict_file(path: str) -> Dict[str, Any]:
    """Load a runtime-layout checkpoint. No QKV reinterleave, no key remap."""
    import torch

    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    return state


def random_state_dict(arch: str, seed: int = 0, dtype: str = "fp32") -> Dict[str, Any]:
    """A checkpoint-shaped random state dict for ``arch``, in runtime layout.

    Built from the *official* model's own topology so the file is exactly what a
    real checkpoint would be, only smaller and random.
    """
    import torch
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    config = ARCHS[arch]
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    model = MiniMaxH3Model(**config, dtype=torch_dtype, device=torch.device("cpu"),
                           operations=comfy.ops.disable_weight_init)
    generator = torch.Generator().manual_seed(int(seed))
    state = {
        name: (torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.05).to(value.dtype)
        for name, value in model.state_dict().items()
    }
    length = config["rope_inv_freq_len"]
    state["rope.inv_freq"] = 1.0 / (
        10000.0 ** (torch.arange(length, dtype=torch.float32) / length)
    )
    return state


# --- environment -------------------------------------------------------------


def find_comfyui(explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for var in ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH"):
        value = os.environ.get(var)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(PROJECT_ROOT / ".cache" / "upstream" / "ComfyUI")
    for candidate in candidates:
        if (candidate / "comfy").is_dir() and (candidate / "folder_paths.py").exists():
            return candidate.resolve()
    raise SystemExit(
        "No ComfyUI checkout found. Pass --comfyui-path or set COMFYUI_PATH."
    )


def configure_attention(backend: Optional[str], disable_fused: bool) -> Dict[str, Any]:
    """Pin the attention backend / kernel fusion, and report what actually stuck.

    Must run *before* anything imports ``comfy.ldm``: upstream binds
    ``optimized_attention`` at import time from the cli args.
    """
    applied: Dict[str, Any] = {"requested_backend": backend, "disable_fused": disable_fused}
    if backend:
        try:
            from comfy.cli_args import args as comfy_args

            flags = {
                "pytorch": "use_pytorch_cross_attention",
                "split": "use_split_cross_attention",
                "quad": "use_quad_cross_attention",
                "sage": "use_sage_attention",
                "flash": "use_flash_attention",
            }
            if backend not in flags:
                raise SystemExit(f"unknown --attention-backend {backend!r}")
            for name in flags.values():
                if hasattr(comfy_args, name):
                    setattr(comfy_args, name, False)
            setattr(comfy_args, flags[backend], True)
            applied["backend_set"] = backend
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - environment dependent
            applied["backend_error"] = repr(exc)
    if disable_fused:
        try:
            import comfy_kitchen

            disabled = []
            for name in ("triton", "cuda"):
                try:
                    comfy_kitchen.registry.disable(name)
                    disabled.append(name)
                except Exception as exc:  # pragma: no cover
                    applied[f"disable_{name}_error"] = repr(exc)
            applied["kitchen_disabled"] = disabled
        except ImportError as exc:  # pragma: no cover
            applied["kitchen_error"] = repr(exc)
    return applied


def resolved_attention_backend() -> Optional[str]:
    """Name of the backend upstream actually bound at import time."""
    try:
        from comfy.ldm.modules.attention import optimized_attention

        return getattr(getattr(optimized_attention, "__wrapped__", optimized_attention),
                       "__name__", None)
    except Exception:  # pragma: no cover - environment dependent
        return None


# --- attention tap -----------------------------------------------------------


def tensor_stats(tensor: Any) -> Dict[str, Any]:
    t = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(t.mean()),
        "std": float(t.std()) if t.numel() > 1 else 0.0,
        "absmax": float(t.abs().max()) if t.numel() else 0.0,
        "sum": float(t.sum()),
    }


def canonical_kv(tensor: Any) -> Any:
    """Any attention-input layout -> ``[rows, heads, head_dim]`` float32 CPU.

    Accepts ``[1, heads, rows, head_dim]`` (Comfy's container layout) and
    ``[rows, heads, head_dim]`` (RAVEN's varlen pack). Anything else is a bug in
    the caller, not something to guess at.
    """
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {list(tensor.shape)}")
        tensor = tensor[0].transpose(0, 1)
    elif tensor.dim() != 3:
        raise ValueError(f"cannot canonicalise attention tensor {list(tensor.shape)}")
    return tensor.detach().to("cpu", copy=True).float().contiguous()


def select_layers(spec: str, num_layers: int) -> List[int]:
    """``'first,mid,last'`` / ``'all'`` / ``'none'`` / ``'0,7,49'`` -> indices."""
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return list(range(num_layers))
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "first":
            out.append(0)
        elif token == "mid":
            out.append(num_layers // 2)
        elif token == "last":
            out.append(num_layers - 1)
        else:
            index = int(token)
            out.append(index if index >= 0 else num_layers + index)
    return sorted({i for i in out if 0 <= i < num_layers})


class KVTap:
    """Records what every layer's attention backend is actually handed.

    The tap sits on ``raven_streaming.causal_model.raven_packed_attention``,
    the causal lane's single attention seam, i.e. *after* the cache merge: ``k``
    and ``v`` are the merged ``[retained | current]`` tensors and ``q`` is the
    current chunk only. RAVEN's harness taps the equivalent point (its varlen
    flash call inside ``causal_model``), which is what makes the two dumps
    comparable.

    It used to tap ``causal_model.optimized_attention``. That name is gone: the
    causal lane no longer routes through Comfy's 4-D dispatcher, it calls
    RAVEN's packed 3-D SDPA directly, and the tensors arrive here already in the
    canonical ``[rows, heads, head_dim]`` layout (no ``AttentionTensorContainer``
    and therefore no ``peek()``).

    **Two call sites share the seam**, and they are not the same measurement:

    * the 50 DiT blocks -- ``site = ('dit', layer_idx)`` -- which is what
      :data:`KV_DUMP_SCHEMA`'s ``entries`` are and what the RAVEN dump can be
      compared against;
    * the token refiner, during ``prefill_text`` only --
      ``site = ('text_refiner', block_idx)``. RAVEN's refiner runs through a
      *different* module-level wrapper (``model._MINIMAX_H3_FLASH_ATTENTION``)
      which the harness does not tap, so a refiner entry has no counterpart on
      the reference side. They are kept in :attr:`refiner_entries` -- a
      diagnostic, never compared, never mixed into the DiT layer numbering.

    The layer index is read off the call's own ``site`` label, not counted:
    counting would renumber every DiT block by however many refiner calls
    preceded it, and a silent off-by-two is exactly the failure this dump exists
    to catch. The sequential position is still checked against the label, and
    any disagreement is recorded in :attr:`order_violations` (surfaced as the
    ``attention.tap_layer_order`` check) rather than being papered over.
    """

    def __init__(self, full_layers: Sequence[int] = (), row_stride: int = 1) -> None:
        self.full_layers = set(int(i) for i in full_layers)
        self.row_stride = max(1, int(row_stride))
        self.entries: List[Dict[str, Any]] = []
        self.refiner_entries: List[Dict[str, Any]] = []
        self.order_violations: List[Dict[str, Any]] = []
        self.unlabelled_calls = 0
        self.forward_name = "?"
        self._original = None

    @contextlib.contextmanager
    def install(self):
        import raven_streaming.causal_model as cm

        self._original = cm.raven_packed_attention

        def traced(q, k, v, *, scale, site=None):
            self._record(q, k, v, site)
            return self._original(q, k, v, scale=scale, site=site)

        cm.raven_packed_attention = traced
        try:
            yield self
        finally:
            cm.raven_packed_attention = self._original
            self._original = None

    @contextlib.contextmanager
    def forward(self, name: str):
        previous, self.forward_name = self.forward_name, name
        try:
            yield
        finally:
            self.forward_name = previous

    def _record(self, q, k, v, site) -> None:
        kind, index = self._resolve_site(site)
        if kind == "text_refiner":
            # stats only: the refiner is a diagnostic and the full arch would
            # store two extra text-length tensors per rollout for nothing
            self.refiner_entries.append(
                record_entry(f"{self.forward_name}:refiner", index, q, k, v,
                             store_full=False, row_stride=self.row_stride))
            return
        position = sum(1 for e in self.entries if e["forward"] == self.forward_name)
        if index != position:
            self.order_violations.append(
                {"forward": self.forward_name, "label": index, "position": position})
        self.entries.append(
            record_entry(self.forward_name, index, q, k, v,
                         store_full=index in self.full_layers,
                         row_stride=self.row_stride))

    def _resolve_site(self, site) -> Tuple[str, int]:
        """``('dit'|'text_refiner', index)`` from the call's label.

        An unlabelled call is counted (``unlabelled_calls``) and treated as a
        DiT block at its sequential position -- the pre-label behaviour -- so a
        plugin build older than the label still produces a readable dump instead
        of a crash. ``attention.tap_labels`` gates on the count being zero.
        """
        if site is None:
            self.unlabelled_calls += 1
            return "dit", sum(1 for e in self.entries
                              if e["forward"] == self.forward_name)
        kind, index = site
        return str(kind), int(index)

    def by_forward(self, name: str) -> List[Dict[str, Any]]:
        return [entry for entry in self.entries if entry["forward"] == name]

    def refiners_by_forward(self, name: str) -> List[Dict[str, Any]]:
        return [entry for entry in self.refiner_entries
                if entry["forward"] == f"{name}:refiner"]


def add_tap_classification_checks(report: "Report", tap: KVTap, *, num_layers: int,
                                  refiner_layers: int, text_rows: int) -> None:
    """Prove the tap split DiT blocks from refiner blocks, and did not renumber.

    Both call sites share one attention seam, so "the dump has the right number
    of entries" is not enough: a refiner call landing in ``entries`` would shift
    every DiT layer index in the text forward by one and still count out. These
    checks pin the split itself.

    * ``attention.tap_labels`` -- every call carried a ``site`` label, so no
      layer index was inferred by counting;
    * ``attention.tap_layer_order`` -- each label matched its sequential
      position, i.e. the blocks ran in order and none was recorded twice;
    * ``attention.dit_layers_complete`` -- every forward recorded exactly
      ``0..num_layers-1``;
    * ``attention.refiner_calls_separated`` -- refiner calls exist only for the
      text prefill, number ``refiner_layers``, and are self-attention over the
      text rows (``q_rows == k_rows == text_rows``), which is what tells them
      apart from a DiT call that reads the cache.
    """
    report.add(Check("attention.tap_labels", tap.unlabelled_calls == 0,
                     {"unlabelled_calls": tap.unlabelled_calls}))
    report.add(Check("attention.tap_layer_order", not tap.order_violations,
                     {"violations": tap.order_violations[:8],
                      "count": len(tap.order_violations)}))

    by_forward: Dict[str, List[int]] = {}
    for entry in tap.entries:
        by_forward.setdefault(entry["forward"], []).append(int(entry["layer"]))
    expected = list(range(num_layers))
    bad = {name: layers for name, layers in by_forward.items() if layers != expected}
    report.add(Check("attention.dit_layers_complete", not bad,
                     {"forwards": len(by_forward), "layers_per_forward": num_layers,
                      "mismatched": {k: v[:8] for k, v in list(bad.items())[:4]}}))

    refiner_forwards = sorted({e["forward"] for e in tap.refiner_entries})
    shapes_ok = all(
        e["stats"]["q"]["shape"][0] == text_rows
        and e["stats"]["k"]["shape"][0] == text_rows
        for e in tap.refiner_entries
    )
    layers_ok = ([int(e["layer"]) for e in tap.refiner_entries]
                 == list(range(refiner_layers)))
    report.add(Check(
        "attention.refiner_calls_separated",
        refiner_forwards == ["text:refiner"] and shapes_ok and layers_ok,
        {"forwards": refiner_forwards, "calls": len(tap.refiner_entries),
         "expected_calls": refiner_layers, "text_rows": text_rows,
         "shapes_ok": shapes_ok, "layers_ok": layers_ok,
         "in_dit_entries": sum(1 for e in tap.entries if ":refiner" in e["forward"])},
    ))


def record_entry(forward: str, layer: int, q, k, v, *, store_full: bool,
                 row_stride: int = 1) -> Dict[str, Any]:
    """One :data:`KV_DUMP_SCHEMA` entry from one attention call's inputs."""
    canon = {"q": canonical_kv(q), "k": canonical_kv(k), "v": canonical_kv(v)}
    entry: Dict[str, Any] = {
        "forward": forward,
        "layer": int(layer),
        "stats": {name: tensor_stats(value) for name, value in canon.items()},
        "row_stride": int(row_stride),
    }
    if store_full:
        for name, value in canon.items():
            entry[name] = value[:: max(1, int(row_stride))].contiguous()
    return entry


def write_dump(path: str, producer: str, entries: Sequence[Dict[str, Any]],
               outputs: Sequence[Dict[str, Any]], meta: Dict[str, Any],
               refiner_entries: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": KV_DUMP_SCHEMA["version"],
        "producer": producer,
        "entries": list(entries),
        "outputs": list(outputs),
        "meta": meta,
    }
    if refiner_entries:
        # additive key: the comparator ignores it, and a producer without a
        # refiner tap (RAVEN's harness) simply never passes one
        payload["refiner_entries"] = list(refiner_entries)
    torch.save(payload, path)


# --- comparison --------------------------------------------------------------


def compare_tensors(ours, theirs, atol: float, rtol: float) -> Dict[str, Any]:
    """``|ours - ref| <= atol + rtol * |ref|`` with the numbers that decided it."""
    import torch

    a = ours.detach().float()
    b = theirs.detach().float()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)], "allclose": False}
    diff = (a - b).abs()
    denom = b.abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / denom.clamp(min=1e-12)).max()) if diff.numel() else 0.0
    allowed = atol + rtol * denom
    allclose = bool(torch.all(diff <= allowed))
    over = int((diff > allowed).sum())
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
        "over_tolerance_elements": over,
        "elements": int(a.numel()),
        "ref_absmax": float(denom.max()) if denom.numel() else 0.0,
    }


def compare_dumps(report: Report, ours: Dict[str, Any], reference: Dict[str, Any],
                  atol: float, rtol: float, *,
                  layer0_rel_l2_max: float = LAYER0_REL_L2_MAX,
                  output_rel_l2_max: float = OUTPUT_REL_L2_MAX,
                  cos_min: float = COSINE_MIN) -> None:
    """Layer-by-layer K/V and per-chunk x0 comparison of two dumps.

    What decides the run (the *gate*):

    * block 0 of every forward -- the shallowest measurement available, so the
      one that is still about operators rather than accumulation -- must hold
      ``rel_l2 <= layer0_rel_l2_max`` and ``cosine >= cos_min`` on Q, K and V;
    * the per-chunk ``video_x0`` / ``audio_x0`` must hold
      ``rel_l2 <= output_rel_l2_max`` and ``cosine >= cos_min``.

    Everything else -- deeper blocks, per-element ``allclose``, the stats-only
    entries -- is measured, printed and stored as a diagnostic. A deep block
    drifting is expected (it is fifty bf16 layers of accumulation); an operator
    disagreeing at block 0, or the x0 the sampler actually consumes moving, is
    not.
    """
    if int(reference.get("version", -1)) != KV_DUMP_SCHEMA["version"]:
        report.add(Check("dump.schema_version", False,
                         {"ref": reference.get("version"),
                          "expected": KV_DUMP_SCHEMA["version"]}))
        return
    report.meta["reference_producer"] = reference.get("producer")
    report.meta["reference_meta"] = reference.get("meta")

    mine, theirs = ours["entries"], reference["entries"]
    if len(mine) != len(theirs):
        report.add(Check("dump.call_count", False,
                         {"ours": len(mine), "theirs": len(theirs)}))
        return
    report.add(Check("dump.call_count", True, {"calls": len(mine)}))

    order_ok = all(a["forward"] == b["forward"] and a["layer"] == b["layer"]
                   for a, b in zip(mine, theirs))
    report.add(Check("dump.call_order", order_ok))
    if not order_ok:
        return

    report.meta["gate"] = {
        "layer0_rel_l2_max": layer0_rel_l2_max,
        "output_rel_l2_max": output_rel_l2_max,
        "cosine_min": cos_min,
        "allclose": f"diagnostic only (atol={atol}, rtol={rtol})",
    }
    _compare_runtime(report, reference)

    worst_allclose: Dict[str, Dict[str, Any]] = {}
    compared = {"q": 0, "k": 0, "v": 0}
    # worst metric per (key, depth class); layer 0 gates, deeper layers report
    worst_metric: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # stats are the only thing available for layers that stored no full tensor,
    # and they are judged by the same rule: |ours - ref| <= atol + rtol * |ref|
    stats_worst: Dict[str, Dict[str, Any]] = {
        key: {"delta": 0.0, "at": None, "violations": 0} for key in ("q", "k", "v")
    }
    for a, b in zip(mine, theirs):
        where = f"{a['forward']}#{a['layer']}"
        depth = "layer0" if int(a["layer"]) == 0 else "deep"
        row: Dict[str, Any] = {"forward": a["forward"], "layer": int(a["layer"]),
                               "depth": depth}
        for key in ("q", "k", "v"):
            # the shape is always comparable, even when only stats were stored
            if a["stats"][key]["shape"] != b["stats"][key]["shape"]:
                report.add(Check(f"dump.shape[{where}.{key}]", False,
                                 {"ours": a["stats"][key]["shape"],
                                  "theirs": b["stats"][key]["shape"]}))
                return
            for stat in ("mean", "std", "absmax"):
                ours_stat = a["stats"][key][stat]
                ref_stat = b["stats"][key][stat]
                delta = abs(ours_stat - ref_stat)
                if delta > atol + rtol * abs(ref_stat):
                    stats_worst[key]["violations"] += 1
                if delta > stats_worst[key]["delta"]:
                    stats_worst[key].update(delta=delta, at=f"{where}.{stat}",
                                            ref=ref_stat)
            if key in a and key in b:
                metrics = tensor_metrics(a[key], b[key])
                row[key] = metrics
                current = worst_metric.get((key, depth))
                if current is None or metrics["rel_l2"] > current["rel_l2"]:
                    worst_metric[(key, depth)] = dict(metrics, at=where)

                result = compare_tensors(a[key], b[key], atol, rtol)
                compared[key] += 1
                current_allclose = worst_allclose.get(key)
                if current_allclose is None or result["max_abs"] > current_allclose["max_abs"]:
                    worst_allclose[key] = dict(result, at=where)
        if any(key in row for key in ("q", "k", "v")):
            report.metrics.append(row)

    for key in ("q", "k", "v"):
        layer0 = worst_metric.get((key, "layer0"))
        if layer0 is not None:
            report.add(Check(
                f"gate.layer0_{key}",
                metrics_pass(layer0, rel_l2_max=layer0_rel_l2_max, cos_min=cos_min),
                dict(_metric_detail(layer0), rel_l2_max=layer0_rel_l2_max,
                     cos_min=cos_min),
            ))
        else:
            report.skip(f"gate.layer0_{key}",
                        "block 0 stored no full tensor on both sides "
                        "(use --kv-layers first,... on both runs)")
        deep = worst_metric.get((key, "deep"))
        if deep is not None:
            report.add(Check(f"depth.worst_{key}", True, _metric_detail(deep), gate=False))

        if compared[key]:
            detail = dict(worst_allclose[key], compared_tensors=compared[key],
                          atol=atol, rtol=rtol)
            report.add(Check(f"dump.{key}_within_tolerance", bool(detail["allclose"]),
                             detail, gate=False))
        else:
            report.skip(f"dump.{key}_within_tolerance",
                        "no full tensors on both sides (--kv-layers disjoint or 'none')")
        report.add(Check(f"dump.{key}_stats_within_tolerance",
                         stats_worst[key]["violations"] == 0,
                         dict(stats_worst[key], atol=atol, rtol=rtol), gate=False))

    ref_outputs = reference.get("outputs", [])
    our_outputs = ours.get("outputs", [])
    if ref_outputs and our_outputs and len(ref_outputs) == len(our_outputs):
        for name in ("video_x0", "audio_x0"):
            worst_out: Optional[Dict[str, Any]] = None
            worst_allclose_out: Optional[Dict[str, Any]] = None
            for index, (a, b) in enumerate(zip(our_outputs, ref_outputs)):
                if name not in a or name not in b:
                    continue
                metrics = tensor_metrics(a[name], b[name])
                report.metrics.append(dict(metrics, forward=f"chunk{index}",
                                           tensor=name, depth="output"))
                if worst_out is None or metrics["rel_l2"] > worst_out["rel_l2"]:
                    worst_out = dict(metrics, at=f"chunk{index}")
                result = compare_tensors(a[name], b[name], atol, rtol)
                if (worst_allclose_out is None
                        or result["max_abs"] > worst_allclose_out["max_abs"]):
                    worst_allclose_out = dict(result, at=f"chunk{index}")
            if worst_out is not None:
                report.add(Check(
                    f"gate.output_{name}",
                    metrics_pass(worst_out, rel_l2_max=output_rel_l2_max, cos_min=cos_min),
                    dict(_metric_detail(worst_out), rel_l2_max=output_rel_l2_max,
                         cos_min=cos_min),
                ))
                report.add(Check(f"dump.{name}_within_tolerance",
                                 bool(worst_allclose_out["allclose"]),
                                 dict(worst_allclose_out, atol=atol, rtol=rtol),
                                 gate=False))
    else:
        report.skip("gate.outputs",
                    f"outputs missing or length mismatch "
                    f"({len(our_outputs)} vs {len(ref_outputs)})")


def _compare_runtime(report: Report, reference: Dict[str, Any]) -> None:
    """Diagnostic: did both sides run under the same numerics environment?

    Not a gate -- the two boxes may legitimately differ -- but TF32 or a
    different SDP kernel moves results by more than this lane measures, so a
    mismatch has to be visible in the report rather than inferred later.
    """
    ours = report.meta.get("runtime") or {}
    theirs = (reference.get("meta") or {}).get("runtime")
    if not ours or not theirs:
        report.skip("env.runtime_recorded",
                    "one side's dump carries no runtime metadata")
        return
    differences = {}
    for key in ("torch", "torch_cuda", "float32_matmul_precision", "tf32_matmul",
                "tf32_cudnn", "sdp", "device_name"):
        if ours.get(key) != theirs.get(key):
            differences[key] = {"ours": ours.get(key), "reference": theirs.get(key)}
    report.add(Check("env.runtime_matches", not differences,
                     differences or {"torch": ours.get("torch"),
                                     "tf32_matmul": ours.get("tf32_matmul"),
                                     "sdp": ours.get("sdp")},
                     gate=False))


def _metric_detail(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a metric dict worth printing on one line."""
    keys = ("at", "rel_l2", "cosine", "max_abs_over_ref_absmax",
            "p99_abs_over_ref_absmax", "max_abs", "ref_absmax", "elements")
    return {key: _round(metrics[key]) for key in keys if key in metrics}


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value


# --- rollout shared by tiny and real ----------------------------------------


def rollout(model, layout, *, video_xt, audio_xt, video_x0, audio_x0,
            video_eps, audio_eps, context, cache, video_sigma, audio_sigma,
            tap=None, transformer_options=None):
    """text prefill -> per chunk: noise (read-only) then clean fill.

    Returns the per-chunk ``(velocity_video, velocity_audio, x_t_video, x_t_audio)``
    of the noise passes, which is everything a dump needs.
    """
    transformer_options = transformer_options or {}

    def named(name):
        return tap.forward(name) if tap is not None else contextlib.nullcontext()

    with named("text"):
        model.prefill_text(context, cache=cache, transformer_options=transformer_options)

    results = []
    for index in range(layout.num_chunks):
        chunk_xt_v = layout.video_chunk_latent(video_xt, index)
        chunk_xt_a = layout.audio_chunk_latent(audio_xt, index)
        with named(f"chunk{index}:noise"):
            velocity = model.forward_chunk(
                video_latent=chunk_xt_v, audio_latent=chunk_xt_a,
                layout=layout, chunk_index=index, cache=cache, role="noise",
                video_sigma=video_sigma, audio_sigma=audio_sigma,
                transformer_options=transformer_options,
            )
        results.append((velocity[0], velocity[1], chunk_xt_v, chunk_xt_a))
        if index == layout.num_chunks - 1:
            break  # nothing after the last chunk reads its history
        with named(f"chunk{index}:clean"):
            model.forward_chunk(
                video_latent=layout.video_chunk_latent(video_x0, index),
                audio_latent=layout.audio_chunk_latent(audio_x0, index),
                layout=layout, chunk_index=index, cache=cache, role="clean",
                video_eps=layout.video_chunk_latent(video_eps, index),
                audio_eps=layout.audio_chunk_latent(audio_eps, index),
                transformer_options=transformer_options,
            )
    return results


def outputs_as_x0(results, video_sigma: float, audio_sigma: float) -> List[Dict[str, Any]]:
    """Native H3 velocity -> x0, the form RAVEN's wrapper returns."""
    from raven_streaming.causal_model import velocity_to_x0

    out = []
    for v_vel, a_vel, v_xt, a_xt in results:
        out.append({
            "video_x0": velocity_to_x0(v_xt, v_vel, 1.0 - video_sigma).detach().to("cpu").float(),
            "audio_x0": velocity_to_x0(a_xt, a_vel, 1.0 - audio_sigma).detach().to("cpu").float(),
        })
    return out


# --- tiny mode ---------------------------------------------------------------


def build_tiny(device: str, seed: int = 0) -> Tuple[Any, Any]:
    import torch
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model

    kwargs = dict(TINY_CONFIG, dtype=torch.float32, device=torch.device(device),
                  operations=comfy.ops.disable_weight_init)
    official = MiniMaxH3Model(**kwargs)
    causal = RavenCausalMiniMaxH3Model(**kwargs)
    state = random_state_dict("tiny", seed=seed)
    official.load_state_dict(state)
    causal.load_state_dict(state)
    official.to(device).requires_grad_(False).eval()
    causal.to(device).requires_grad_(False).eval()
    return official, causal


def _tiny_rollout(causal, layout, video, audio, context, cache, tap=None, sigma=0.6,
                  clean_scale=1.0, eps_seed=7):
    import torch

    generator = torch.Generator().manual_seed(eps_seed)
    return rollout(
        causal, layout,
        video_xt=video, audio_xt=audio,
        video_x0=video * clean_scale, audio_x0=audio * clean_scale,
        video_eps=torch.randn(video.shape, generator=generator),
        audio_eps=torch.randn(audio.shape, generator=generator),
        context=context, cache=cache,
        video_sigma=sigma, audio_sigma=causal.audio_sigma_from_video(sigma),
        tap=tap,
    )


def run_tiny(args) -> Report:
    import torch
    from comfy.ldm.minimax.model import PackedLayout

    from raven_streaming.cache import ChunkKVCache
    from raven_streaming.causal_model import (
        CLEAN_TIMESTEP_AUDIO,
        CLEAN_TIMESTEP_VIDEO,
        _fp32_one_minus,
        _fp32_scalar,
    )
    from raven_streaming.layout import T2VALayout

    report = Report(mode="tiny", device=args.device)
    official, causal = build_tiny(args.device, seed=args.seed)
    layers = TINY_CONFIG["num_layers"]

    layout = T2VALayout.from_request(
        text_len=args.text_len, frames=args.frames, width=args.width, height=args.height
    )
    report.meta.update(
        config=TINY_CONFIG,
        runtime=runtime_meta(),
        parity_scope="base (no LoRA) causal lane vs the official dense model",
        request=dict(frames=args.frames, width=args.width, height=args.height,
                     text_len=args.text_len, sink=args.sink, window=args.window),
        layout=dict(latent_t=layout.latent_t, audio_t=layout.audio_t,
                    latent_h=layout.latent_h, latent_w=layout.latent_w,
                    chunks=[dict(video=(c.video_start, c.video_stop),
                                 audio=(c.audio_start, c.audio_stop), rows=c.rows)
                            for c in layout.chunks]),
    )

    generator = torch.Generator().manual_seed(args.seed + 1)
    video = torch.randn(*layout.video_latent_shape(TINY_CONFIG["latents_dim"]),
                        generator=generator).to(args.device)
    audio = torch.randn(*layout.audio_latent_shape(TINY_CONFIG["audio_latents_dim"]),
                        generator=generator).to(args.device)
    context = torch.randn(1, layout.text_len, TINY_CONFIG["text_dim"],
                          generator=generator).to(args.device)

    # 1 / 2: state dict
    theirs = official.state_dict()
    ours = causal.state_dict()
    report.add(Check("state_dict.keys", list(theirs) == list(ours), {"n": len(theirs)}))
    shapes_match = {k: (tuple(v.shape), str(v.dtype)) for k, v in theirs.items()} == {
        k: (tuple(v.shape), str(v.dtype)) for k, v in ours.items()
    }
    report.add(Check("state_dict.shapes_dtypes", shapes_match))

    # 3: dense forward
    with torch.no_grad():
        dense_official = official(x=[video, audio], timestep=torch.tensor([500.0]),
                                  context=context)
        dense_causal = causal(x=[video, audio], timestep=torch.tensor([500.0]),
                              context=context)
    dense_diff = max(float((a - b).abs().max()) for a, b in zip(dense_official, dense_causal))
    report.add(Check("dense_forward.bit_identical", dense_diff == 0.0,
                     {"max_abs_diff": dense_diff}))

    # 4: chunk positions vs the official packed layout
    packed = PackedLayout(layout.text_len, layout.latent_t, layout.latent_h,
                          layout.latent_w, layout.audio_t)
    frame_rows, audio_t = layout.frame_rows, layout.audio_t
    audio_base, video_base = layout.text_len, layout.text_len + audio_t * 2
    position_diff = 0.0
    for index, chunk in enumerate(layout.chunks):
        rows = torch.cat((
            torch.arange(audio_base + chunk.audio_start, audio_base + chunk.audio_stop),
            torch.arange(audio_base + audio_t + chunk.audio_start,
                         audio_base + audio_t + chunk.audio_stop),
            torch.arange(video_base + chunk.video_start * frame_rows,
                         video_base + chunk.video_stop * frame_rows),
        ))
        diff = (layout.chunk_position_ids(index) - packed.position_ids.index_select(0, rows)).abs()
        position_diff = max(position_diff, float(diff.max()))
    report.add(Check("layout.positions_match_official", position_diff <= 1e-11,
                     {"max_abs_diff": position_diff}))

    # 5: the merged key count every layer sees
    tap = KVTap(full_layers=range(layers), row_stride=1)
    cache = ChunkKVCache(layers, sink=args.sink, window=args.window)
    with torch.no_grad(), tap.install():
        base_results = _tiny_rollout(causal, layout, video, audio, context, cache, tap=tap)

    observed_ok = True
    calls = 0
    committed = 0
    chunk_lens: List[int] = []

    def policy_rows(lens, sink, window, n):
        keep = set(ChunkKVCache(1, sink=sink, window=window).retained_index_set(n))
        return sum(length for i, length in enumerate(lens) if i in keep)

    for name in ["text"] + [f"chunk{i}:{role}" for i in range(layout.num_chunks)
                            for role in ("noise", "clean")]:
        entries = tap.by_forward(name)
        if not entries:
            continue
        if name == "text":
            q_rows, retained_before = layout.text_len, 0
        else:
            index = int(name.split(":")[0][5:])
            q_rows = layout.chunks[index].rows
            retained_before = policy_rows(chunk_lens, args.sink, args.window, committed)
        for entry in entries:
            calls += 1
            if entry["stats"]["k"]["shape"][0] != retained_before + q_rows:
                observed_ok = False
            if entry["stats"]["q"]["shape"][0] != q_rows:
                observed_ok = False
        if name == "text" or name.endswith(":clean"):
            chunk_lens.append(q_rows)
            committed += 1
    report.add(Check("attention.key_rows_match_cache", observed_ok,
                     {"calls": calls, "expected_calls": len(tap.entries)}))
    # ... and the count above must be every DiT call, not a subset that happened
    # to line up because refiner calls were silently dropped somewhere
    report.add(Check("attention.key_rows_cover_every_call", calls == len(tap.entries),
                     {"gated": calls, "recorded": len(tap.entries)}))
    add_tap_classification_checks(report, tap, num_layers=layers,
                                  refiner_layers=len(causal.token_refiner.blocks),
                                  text_rows=layout.text_len)

    # 6: determinism
    cache_b = ChunkKVCache(layers, sink=args.sink, window=args.window)
    with torch.no_grad():
        repeat = _tiny_rollout(causal, layout, video, audio, context, cache_b)
    repeat_diff = max(
        float((a - b).abs().max())
        for first, second in zip(base_results, repeat)
        for a, b in zip(first[:2], second[:2])
    )
    report.add(Check("rollout.deterministic", repeat_diff == 0.0,
                     {"max_abs_diff": repeat_diff}))

    # 7: visibility -- the retained context must matter
    cache_c = ChunkKVCache(layers, sink=8, window=None)
    cache_d = ChunkKVCache(layers, sink=8, window=None)
    with torch.no_grad():
        altered = _tiny_rollout(causal, layout, video, audio, context, cache_c, clean_scale=-3.0)
        kept = _tiny_rollout(causal, layout, video, audio, context, cache_d, clean_scale=1.0)
    visible_diff = min(
        float((a[0] - b[0]).abs().max()) for a, b in zip(altered[1:], kept[1:])
    ) if layout.num_chunks > 1 else float("nan")
    output_scale = max(float(a[0].abs().max()) for a in kept)
    report.add(Check("cache.history_is_visible", visible_diff > 0.0,
                     {"min_abs_diff_over_chunks": visible_diff,
                      "output_absmax": output_scale}))

    # 8: eviction -- an evicted chunk must be invisible
    evicted = []
    for scale in (1.0, -3.0):
        cache_e = ChunkKVCache(layers, sink=1, window=0)
        with torch.no_grad():
            evicted.append(_tiny_rollout(causal, layout, video, audio, context, cache_e,
                                         clean_scale=scale))
    evicted_diff = max(
        float((a - b).abs().max())
        for first, second in zip(evicted[0][1:], evicted[1][1:])
        for a, b in zip(first[:2], second[:2])
    ) if layout.num_chunks > 1 else 0.0
    report.add(Check("cache.evicted_history_is_invisible", evicted_diff == 0.0,
                     {"max_abs_diff": evicted_diff}))

    # 9: clean timestep mapping
    cache_f = ChunkKVCache(layers, sink=8, window=None)
    cache_g = ChunkKVCache(layers, sink=8, window=None)
    with torch.no_grad():
        causal.prefill_text(context, cache=cache_f)
        causal.prefill_text(context, cache=cache_g)
        x0_v = layout.video_chunk_latent(video, 0)
        x0_a = layout.audio_chunk_latent(audio, 0)
        eps_v = torch.randn(x0_v.shape, generator=torch.Generator().manual_seed(3))
        eps_a = torch.randn(x0_a.shape, generator=torch.Generator().manual_seed(4))
        from_clean = causal.forward_chunk(
            video_latent=x0_v, audio_latent=x0_a, layout=layout, chunk_index=0,
            cache=cache_f, role="clean", video_eps=eps_v, audio_eps=eps_a)
        # In the lane's own float32 arithmetic, because that is RAVEN's: the
        # condition timestep is the fp32 0.999 and the eps coefficient is
        # ``1 - fp32(0.999)``, not ``fp32(1 - 0.999)``. The repo sigma that maps
        # back onto it is that same ``1 - t``, exactly -- ``1 - t`` is
        # representable for t near 1 (Sterbenz), so ``1 - (1 - t) == t`` in fp32
        # and the two forwards really do run at one timestep.
        t_v = _fp32_scalar(CLEAN_TIMESTEP_VIDEO)
        t_a = _fp32_scalar(CLEAN_TIMESTEP_AUDIO)
        sigma_v, sigma_a = _fp32_one_minus(t_v), _fp32_one_minus(t_a)
        from_noise = causal.forward_chunk(
            video_latent=t_v * x0_v + sigma_v * eps_v,
            audio_latent=t_a * x0_a + sigma_a * eps_a,
            layout=layout, chunk_index=0, cache=cache_g, role="noise",
            video_sigma=sigma_v, audio_sigma=sigma_a)
    clean_diff = max(float((a - b).abs().max()) for a, b in zip(from_clean, from_noise))
    report.add(Check("timestep.clean_maps_to_0999", clean_diff == 0.0,
                     {"max_abs_diff": clean_diff}))
    return report


# --- real mode ---------------------------------------------------------------


def check_full_nonpruned(report: Report, state: Dict[str, Any], arch: str) -> bool:
    """Refuse a pruned / distilled / curve-form checkpoint before anything else.

    RAVEN is trained against the full non-pruned BF16 DiT, so a parity run
    against anything else is measuring the wrong model.
    """
    config = ARCHS[arch]
    blocks = {int(name.split(".")[1]) for name in state if name.startswith("blocks.")}
    expected = int(config["num_layers"])
    detail: Dict[str, Any] = {"blocks": len(blocks), "expected": expected, "arch": arch}

    ok = len(blocks) == expected and blocks == set(range(expected))
    if "adaln_t_table" in state:
        ok = False
        detail["curve_form"] = True
    if "time_embedder.proj_in.weight" not in state:
        ok = False
        detail["time_embedder"] = "missing"
    hidden = state.get("blocks.0.norm1.weight")
    if hidden is not None:
        detail["hidden_size"] = int(hidden.shape[0])
        if int(hidden.shape[0]) != int(config["hidden_size"]):
            ok = False
    ffn = state.get("blocks.0.mlp.fc2.weight")
    if ffn is not None:
        detail["ffn_hidden_size"] = int(ffn.shape[1])
        if int(ffn.shape[1]) != int(config["ffn_hidden_size"]):
            ok = False
    report.add(Check("weights.full_nonpruned", ok, detail))
    return ok


def _fp32_island_fingerprints(model) -> Dict[str, Any]:
    """Evidence that the fp32 parameters still carry fp32 information.

    ``bf16_exact_fraction == 1.0`` on a float32 parameter means it went through
    bf16 at some point: the dtype says fp32, the values do not.
    """
    import torch

    watched = ("video_patch_proj.weight", "audio_patch_proj.weight",
               "time_embedder.proj_in.weight", "time_embedder.proj_out.weight",
               "final_layer.video_out.weight", "rope.inv_freq")
    state = model.state_dict()
    out: Dict[str, Any] = {}
    for name in watched:
        value = state.get(name)
        if value is None or not value.is_floating_point():
            continue
        as_bf16 = value.to(torch.bfloat16).to(value.dtype)
        out[name] = {
            "dtype": str(value.dtype),
            "bf16_exact_fraction": round(float((value == as_bf16).float().mean()), 6),
        }
    return out


def check_no_lora(report: Report, model) -> None:
    """This lane compares base against base; a patched model would not be that."""
    from raven_streaming.runtime_linear import HOOK_ATTR

    hooked = [name for name, module in model.named_modules() if hasattr(module, HOOK_ATTR)]
    lora_params = [name for name, _ in model.named_parameters() if "raven_lora" in name]
    report.add(Check("weights.base_no_lora", not hooked and not lora_params,
                     {"hooked_modules": len(hooked), "lora_params": len(lora_params)}))


def run_real(args) -> Report:
    import torch
    import comfy.ops

    from raven_streaming.cache import ChunkKVCache
    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model
    from raven_streaming.layout import T2VALayout

    report = Report(mode="real", device=args.device)
    report.meta["attention"] = dict(getattr(args, "_attention_meta", {}),
                                    resolved=resolved_attention_backend())
    report.meta["parity_scope"] = (
        "base vs base: the same runtime-layout checkpoint on both sides, no LoRA, "
        "no QKV reinterleave, no key remap"
    )
    report.meta["runtime"] = runtime_meta()
    if not args.dit:
        raise SystemExit("--mode real needs --dit <checkpoint>")

    inputs = _resolve_inputs(args)
    request = inputs["request"]
    if request["arch"] != args.arch:
        raise SystemExit(
            f"shared inputs were built for arch {request['arch']!r}, "
            f"this run asks for {args.arch!r}"
        )
    report.meta["request"] = request
    report.meta["inputs_file"] = args.inputs

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    state = load_state_dict_file(args.dit)
    if not check_full_nonpruned(report, state, args.arch):
        # A pruned / distilled / curve-form checkpoint is a different model; a
        # dump from it would look like a parity result and mean nothing.
        report.skip("rollout.completed",
                    "checkpoint is not the full non-pruned architecture; refusing "
                    "to produce a dump from it")
        return report

    config = ARCHS[args.arch]
    model = RavenCausalMiniMaxH3Model(
        **config, dtype=dtype, device=torch.device(args.device),
        operations=comfy.ops.manual_cast if args.manual_cast else comfy.ops.disable_weight_init,
    )
    # The checkpoint has an fp32 island (patch projections, time embedder, output
    # heads). A blanket cast to the compute dtype leaves those parameters
    # *declared* fp32 while destroying their mantissa, which is not what a real
    # loader does and moves every tensor downstream of them. Default to each
    # parameter's declared dtype; ``--fp32-island cast`` restores the old
    # blanket behaviour for comparison.
    target = model.state_dict()
    if args.fp32_island == "declared":
        placed = {k: (v.to(target[k].dtype) if k in target else v) for k, v in state.items()}
    else:
        placed = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(placed, strict=False)
    report.meta["weights"] = {
        "fp32_island": args.fp32_island,
        "fingerprints": _fp32_island_fingerprints(model),
    }
    loaded_ok = not missing and not unexpected
    report.add(Check("weights.loaded", loaded_ok,
                     {"missing": len(missing), "unexpected": len(unexpected),
                      "keys": len(state),
                      "first_missing": list(missing)[:3],
                      "first_unexpected": list(unexpected)[:3]}))
    if not loaded_ok:
        report.skip("rollout.completed",
                    "checkpoint does not fit the requested architecture")
        return report
    model.to(args.device).requires_grad_(False).eval()
    check_no_lora(report, model)

    layout = T2VALayout.from_request(text_len=request["text_len"], frames=request["frames"],
                                     width=request["width"], height=request["height"])
    report.add(Check("layout.matches_inputs",
                     layout.latent_t == request["latent_t"]
                     and layout.audio_t == request["audio_t"]
                     and layout.num_chunks == request["num_chunks"],
                     {"latent_t": layout.latent_t, "audio_t": layout.audio_t,
                      "chunks": layout.num_chunks}))

    def to_device(name):
        return inputs[name].to(device=args.device, dtype=dtype)

    full_layers = select_layers(args.kv_layers, len(model.blocks))
    tap = KVTap(full_layers=full_layers, row_stride=args.kv_row_stride)
    cache = ChunkKVCache(len(model.blocks), sink=int(request["sink"]),
                         window=request["window"])
    with torch.no_grad(), tap.install():
        results = rollout(
            model, layout,
            video_xt=to_device("video_xt"), audio_xt=to_device("audio_xt"),
            video_x0=to_device("video_x0"), audio_x0=to_device("audio_x0"),
            video_eps=to_device("video_eps"), audio_eps=to_device("audio_eps"),
            context=to_device("context"), cache=cache,
            video_sigma=float(request["video_sigma"]),
            audio_sigma=float(request["audio_sigma"]),
            tap=tap,
        )
    outputs = outputs_as_x0(results, float(request["video_sigma"]),
                            float(request["audio_sigma"]))
    # Which attention kernel the causal lane actually ran. RAVEN dispatches
    # FA3 -> FA2 -> SDPA and so does this lane, and the vr audit showed that
    # *this* is the production difference once both sides are on the same math
    # SDPA -- so a dump that does not say which kernel produced it cannot be
    # compared against another one.
    from raven_streaming.causal_model import raven_attention_backend

    report.meta["causal_attention_backend"] = raven_attention_backend()
    report.add(Check("rollout.completed", True,
                     {"chunks": layout.num_chunks,
                      "attention_calls": len(tap.entries),
                      "refiner_attention_calls": len(tap.refiner_entries),
                      "attention_backend": (report.meta["causal_attention_backend"]
                                            ["last"] or {}).get("backend"),
                      "full_kv_layers": full_layers, "row_stride": args.kv_row_stride}))
    add_tap_classification_checks(report, tap, num_layers=len(model.blocks),
                                  refiner_layers=len(model.token_refiner.blocks),
                                  text_rows=layout.text_len)

    report.meta["kv_selection"] = {"layers": full_layers, "row_stride": args.kv_row_stride}
    dump = {
        "version": KV_DUMP_SCHEMA["version"],
        "producer": "comfy",
        "entries": tap.entries,
        "refiner_entries": tap.refiner_entries,
        "outputs": outputs,
        "meta": report.meta,
    }
    if args.emit_dump:
        write_dump(args.emit_dump, "comfy", tap.entries, outputs, report.meta,
                   refiner_entries=tap.refiner_entries)
        report.meta["emitted_dump"] = str(args.emit_dump)
        print(f"dump: {args.emit_dump}")

    if args.compare_dump:
        reference = torch.load(args.compare_dump, map_location="cpu", weights_only=False)
        compare_dumps(report, dump, reference, args.atol, args.rtol,
                      layer0_rel_l2_max=args.gate_layer0_rel_l2,
                      output_rel_l2_max=args.gate_output_rel_l2,
                      cos_min=args.gate_cosine)
    else:
        report.skip("dump.comparison",
                    "no --compare-dump: this run only produced the Comfy side")
    return report


def _resolve_inputs(args) -> Dict[str, Any]:
    if args.inputs:
        return load_inputs(args.inputs)
    print("[warn] no --inputs: drawing tensors from the seed. Two processes only "
          "agree if they run the same torch build; prefer --mode inputs --emit-inputs.")
    return build_shared_inputs(
        frames=args.frames, width=args.width, height=args.height,
        text_len=args.text_len, seed=args.seed, sink=args.sink, window=args.window,
        video_sigma=args.sigma, audio_sigma=None, arch=args.arch,
    )


def run_inputs(args) -> Report:
    report = Report(mode="inputs", device=args.device)
    report.meta["runtime"] = runtime_meta()
    inputs = build_shared_inputs(
        frames=args.frames, width=args.width, height=args.height,
        text_len=args.text_len, seed=args.seed, sink=args.sink, window=args.window,
        video_sigma=args.sigma, audio_sigma=None, arch=args.arch,
    )
    report.meta["request"] = inputs["request"]
    if args.emit_inputs:
        save_inputs(args.emit_inputs, inputs)
        report.meta["emitted_inputs"] = str(args.emit_inputs)
        print(f"inputs: {args.emit_inputs}")
    report.add(Check("inputs.built", True, {
        "chunks": inputs["request"]["num_chunks"],
        "video": list(inputs["video_xt"].shape),
        "audio": list(inputs["audio_xt"].shape),
        "context": list(inputs["context"].shape),
    }))
    if args.emit_weights:
        state = random_state_dict(args.arch, seed=args.seed, dtype=args.dtype)
        save_state_dict(args.emit_weights, state)
        report.meta["emitted_weights"] = str(args.emit_weights)
        report.add(Check("weights.emitted", True,
                         {"keys": len(state), "path": str(args.emit_weights)}))
        print(f"weights: {args.emit_weights}")
    if not args.emit_inputs and not args.emit_weights:
        report.skip("inputs.write", "neither --emit-inputs nor --emit-weights given")
    return report


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("tiny", "inputs", "real"), default="tiny")
    parser.add_argument("--device", default=os.environ.get("RAVEN_PROBE_DEVICE", "cpu"))
    parser.add_argument("--comfyui-path", default=None)
    parser.add_argument("--json", default=None, help="write the report to this path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=tuple(ARCHS), default="full",
                        help="tiny is the CPU-sized stand-in; full is the released H3")

    grid = parser.add_argument_group("request grid (small by default: this is a "
                                     "parity probe, not a quality run)")
    grid.add_argument("--frames", type=int, default=39)
    grid.add_argument("--width", type=int, default=512)
    grid.add_argument("--height", type=int, default=288)
    grid.add_argument("--text-len", type=int, default=128)
    grid.add_argument("--sink", type=int, default=2)
    grid.add_argument("--window", type=int, default=2,
                      help="sliding-window chunks; -1 means None (no eviction)")
    grid.add_argument("--sigma", type=float, default=0.6)

    shared = parser.add_argument_group("shared inputs (two-process protocol)")
    shared.add_argument("--emit-inputs", default=None)
    shared.add_argument("--inputs", default=None)
    shared.add_argument("--emit-weights", default=None,
                        help="write a random checkpoint for --arch (tiny runs)")

    real = parser.add_argument_group("real mode")
    real.add_argument("--dit", default=None, help="runtime-layout H3 checkpoint")
    real.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    real.add_argument("--manual-cast", action="store_true")
    real.add_argument("--fp32-island", choices=("declared", "cast"), default="declared",
                      help="'declared' loads each parameter at its own dtype (what a "
                           "real loader does); 'cast' reproduces the old blanket "
                           "cast to --dtype, which degrades the checkpoint's fp32 island")
    real.add_argument("--attention-backend",
                      choices=("pytorch", "split", "quad", "sage", "flash"), default=None)
    real.add_argument("--disable-fused-kernels", action="store_true",
                      help="disable comfy-kitchen triton/cuda backends before running")
    real.add_argument("--emit-dump", default=None)
    real.add_argument("--compare-dump", default=None)
    real.add_argument("--kv-layers", default="first,mid,last",
                      help="which blocks keep full K/V ('all', 'none', 'first,mid,last', '0,7,49')")
    real.add_argument("--kv-row-stride", type=int, default=1,
                      help="store every Nth row of the full K/V tensors")
    real.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                      help="element-wise allclose diagnostic only, never a gate")
    real.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                      help="element-wise allclose diagnostic only, never a gate")

    gate = parser.add_argument_group("semantic gate (what actually decides the run)")
    gate.add_argument("--gate-layer0-rel-l2", type=float, default=LAYER0_REL_L2_MAX)
    gate.add_argument("--gate-output-rel-l2", type=float, default=OUTPUT_REL_L2_MAX)
    gate.add_argument("--gate-cosine", type=float, default=COSINE_MIN)

    args = parser.parse_args(argv)
    if args.window is not None and args.window < 0:
        args.window = None
    if args.mode == "tiny":
        args.arch = "tiny"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    comfyui = find_comfyui(args.comfyui_path)
    if str(comfyui) not in sys.path:
        sys.path.insert(0, str(comfyui))
    print(f"ComfyUI: {comfyui}")
    # Before anything imports comfy.ldm: upstream binds ``optimized_attention``
    # at import time from the cli args, so a later switch would be a no-op.
    args._attention_meta = configure_attention(args.attention_backend,
                                               args.disable_fused_kernels)

    if args.mode == "tiny":
        report = run_tiny(args)
    elif args.mode == "inputs":
        report = run_inputs(args)
    else:
        report = run_real(args)

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
