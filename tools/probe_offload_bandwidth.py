#!/usr/bin/env python3
"""Micro-benchmark: what does host<->device copy actually cost on this box?

The rollout plan for the published request (1376x768, 192 frames, 50 blocks,
56 x 128 heads, BF16) does not fit a 24 GiB card with the KV cache resident:
`docs/validation.md` estimates 34.12 GiB, ~84 % of it KV. Every design that
does fit -- KV in host memory, block weights streamed in, or both -- pays for
it in PCIe traffic, and whether that is survivable is a **bandwidth** question,
not an architecture question. This probe measures the bandwidth instead of
assuming it, and then applies the measured numbers to the real rollout shape.

What is measured (each with CUDA events *and* wall clock around an explicit
``synchronize``, ``--warmup`` discarded, ``--iters`` samples kept, reported as
p50/p95 in GB/s and GiB/s):

``h2d_pageable_blocking``
    pageable host memory -> GPU. The floor: the driver has to stage it.
``h2d_pinned_blocking`` / ``h2d_pinned_nonblocking``
    pinned host memory -> GPU, once as a blocking copy and once issued on a
    dedicated copy stream. The gap between the two is what asynchrony buys.
``d2h_pinned_nonblocking``
    GPU -> pinned host. Evicting KV rows is this direction, and on most parts
    it is not symmetric with H2D.
``pipeline_single_buffer`` / ``pipeline_double_buffer``
    ``--pipeline-blocks`` transfers feeding a ``--compute-ms`` consumer on a
    second stream. With one landing buffer, copy *i* cannot start until the
    consumer released the buffer; with two, it can. The difference is the
    entire argument for double buffering, measured rather than asserted.
``parallel_two_copy_streams`` / ``serial_one_copy_stream``
    a block of weights (``--block-gib``) and one layer of KV issued on two copy
    streams at once, against the same two transfers issued back to back. H2D
    usually has one DMA engine, so this is the check for whether splitting the
    traffic buys anything at all.
``compute_overlap``
    a ``--compute-ms`` CUDA workload (``torch.cuda._sleep`` when available,
    otherwise a calibrated matmul loop -- never a CPU sleep, which would not
    occupy the GPU) on a compute stream while the next layer's KV lands on a
    copy stream. Reports what fraction of the copy is hidden.

Then it applies the measured bandwidth to the rollout: ``--forwards`` DiT
forwards, ``--layers`` x (K+V) of ``--rows`` retained rows per forward fetched
from host memory, plus the weight bytes that a ``--vram-cap-gib`` card cannot
hold. Both a serial and an ideal-overlap **lower bound** are reported; neither
is a prediction of the run, they bracket it.

Usage::

    python tools/probe_offload_bandwidth.py --device cuda \\
        --json .cache/probe_offload_bandwidth.json

    # no GPU: geometry + estimates only, from an assumed bandwidth
    python tools/probe_offload_bandwidth.py --device cpu --assume-h2d-gb-s 20 \\
        --json .cache/probe_offload_bandwidth_assumed.json

Allocation is bounded on purpose (``--max-gpu-gib``, ``--max-host-gib``): this
probe is meant to run next to other work, and a benchmark that OOMs the box it
is measuring has measured nothing. Pinned-allocation failure and OOM are loud
(non-zero exit, the failure recorded in the report) but never silent: the JSON
report is written atomically, inside this repository, even when a scenario
raises.

Exit codes: 0 all requested scenarios ran, 1 something failed, 2 CUDA was not
available so nothing could be measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ENVIRONMENT = 2

GB = 1000 ** 3
GIB = 1024 ** 3

DEVICE_ENV_VAR = "RAVEN_PROBE_DEVICE"
DEFAULT_REPORT = ".cache/probe_offload_bandwidth.json"

#: The published full-size RAVEN shape (docs/validation.md "Memory reserve").
DEFAULT_ROWS = 20996
DEFAULT_HEADS = 56
DEFAULT_HEAD_DIM = 128
DEFAULT_LAYERS = 50
DEFAULT_BLOCK_GIB = 1.29
DEFAULT_FORWARDS = 59

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

SCENARIOS = (
    "h2d_pageable_blocking",
    "h2d_pinned_blocking",
    "h2d_pinned_nonblocking",
    "d2h_pinned_nonblocking",
    "pipeline_single_buffer",
    "pipeline_double_buffer",
    "parallel_two_copy_streams",
    "serial_one_copy_stream",
    "compute_overlap",
)


class ProbeError(RuntimeError):
    """Something the probe cannot honestly measure or is not allowed to do."""


# --------------------------------------------------------------------------
# geometry: bytes per layer of KV, from the model shape
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KVGeometry:
    """K+V byte accounting for one attention layer of ``rows`` retained rows.

    ``ChunkKVCache`` stores ``keys``/``values`` as ``[rows, heads, head_dim]``
    per layer (``raven_streaming/cache.py``), so one layer costs *two* such
    tensors and a forward that reads the whole retained context touches
    ``layers`` of them.
    """

    rows: int
    heads: int
    head_dim: int
    dtype: str = "bf16"
    layers: int = DEFAULT_LAYERS

    def __post_init__(self) -> None:
        for name in ("rows", "heads", "head_dim", "layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ProbeError(f"{name} must be a positive int, got {value!r}")
        if self.dtype not in DTYPE_BYTES:
            raise ProbeError(
                f"dtype must be one of {sorted(DTYPE_BYTES)}, got {self.dtype!r}"
            )

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def elements_per_layer(self) -> int:
        """K and V together, one layer."""
        return 2 * self.rows * self.heads * self.head_dim

    @property
    def per_layer_bytes(self) -> int:
        return self.elements_per_layer * self.dtype_bytes

    @property
    def all_layers_bytes(self) -> int:
        return self.per_layer_bytes * self.layers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "layers": self.layers,
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "per_layer_bytes": self.per_layer_bytes,
            "per_layer_gib": self.per_layer_bytes / GIB,
            "all_layers_bytes": self.all_layers_bytes,
            "all_layers_gib": self.all_layers_bytes / GIB,
        }


# --------------------------------------------------------------------------
# statistics (pure; no CUDA, no torch)
# --------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 1]."""
    if not values:
        raise ProbeError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ProbeError(f"q must be in [0, 1], got {q!r}")
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return data[low]
    return data[low] + (data[high] - data[low]) * (pos - low)


def bandwidth(nbytes: int, milliseconds: float) -> Dict[str, float]:
    """GB/s (10^9) and GiB/s (2^30) for ``nbytes`` moved in ``milliseconds``."""
    if milliseconds <= 0.0:
        raise ProbeError(f"non-positive duration {milliseconds!r} ms")
    seconds = milliseconds / 1000.0
    return {
        "gb_per_s": nbytes / GB / seconds,
        "gib_per_s": nbytes / GIB / seconds,
    }


def describe(samples: Sequence[float]) -> Dict[str, float]:
    data = [float(v) for v in samples]
    if not data:
        raise ProbeError("describe() of an empty sample")
    return {
        "samples": len(data),
        "min": min(data),
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "max": max(data),
        "mean": sum(data) / len(data),
    }


def summarize_transfer(
    nbytes: int,
    event_ms: Sequence[float],
    wall_ms: Sequence[float],
    *,
    note: str = "",
) -> Dict[str, Any]:
    """Per-transfer timing plus the bandwidth it implies.

    Bandwidth is quoted from the **CUDA event** p50 (what the copy engine did)
    and from the p95 (the tail a pipeline would actually have to live with);
    wall clock is reported alongside so launch and synchronize overhead stays
    visible instead of being folded into the headline number.
    """
    events = describe(event_ms)
    walls = describe(wall_ms)
    out: Dict[str, Any] = {
        "bytes": int(nbytes),
        "gib": nbytes / GIB,
        "event_ms": events,
        "wall_ms": walls,
        "bandwidth_p50": bandwidth(nbytes, events["p50"]),
        "bandwidth_p95": bandwidth(nbytes, events["p95"]),
        "wall_bandwidth_p50": bandwidth(nbytes, walls["p50"]),
    }
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------
# rollout arithmetic (pure)
# --------------------------------------------------------------------------
def weight_offload_bytes(
    total_weight_bytes: int,
    vram_cap_bytes: int,
    other_resident_bytes: int = 0,
) -> int:
    """Weight bytes that do not fit and therefore stream in every forward.

    ``other_resident_bytes`` is whatever the card must hold besides weights
    (activations, a resident KV working set, the allocator's own slack). It is
    a knob, not a measurement: the caller states it, the report echoes it.
    """
    for name, value in (
        ("total_weight_bytes", total_weight_bytes),
        ("vram_cap_bytes", vram_cap_bytes),
        ("other_resident_bytes", other_resident_bytes),
    ):
        if value < 0:
            raise ProbeError(f"{name} must be >= 0, got {value!r}")
    room = max(0, int(vram_cap_bytes) - int(other_resident_bytes))
    return max(0, int(total_weight_bytes) - room)


def estimate_rollout(
    *,
    forwards: int,
    layers: int,
    per_layer_kv_bytes: int,
    weight_bytes_per_forward: int,
    h2d_gb_per_s: Optional[float],
    d2h_gb_per_s: Optional[float] = None,
    kv_writeback_bytes_per_forward: int = 0,
    forward_compute_ms: float = 0.0,
    bandwidth_source: str = "measured",
) -> Dict[str, Any]:
    """Copy-time bounds for a whole rollout at a given bandwidth.

    Two numbers, both bounds and neither a prediction:

    ``serial_seconds``
        nothing overlaps -- every byte is copied while the GPU is idle, and
        every forward computes while the link is idle. Upper bound.
    ``overlap_lower_bound_seconds``
        perfect overlap -- ``max(copy, compute)``. No implementation beats it,
        so if this number is already unacceptable, no amount of pipelining
        engineering will rescue the design.
    """
    if forwards <= 0 or layers <= 0:
        raise ProbeError("forwards and layers must be positive")
    if per_layer_kv_bytes < 0 or weight_bytes_per_forward < 0:
        raise ProbeError("byte counts must be >= 0")

    kv_per_forward = int(per_layer_kv_bytes) * int(layers)
    kv_total = kv_per_forward * int(forwards)
    weight_total = int(weight_bytes_per_forward) * int(forwards)
    writeback_total = int(kv_writeback_bytes_per_forward) * int(forwards)
    compute_total_s = (float(forward_compute_ms) / 1000.0) * int(forwards)

    result: Dict[str, Any] = {
        "forwards": int(forwards),
        "layers": int(layers),
        "bandwidth_source": bandwidth_source,
        "h2d_gb_per_s": h2d_gb_per_s,
        "d2h_gb_per_s": d2h_gb_per_s,
        "kv_bytes_per_forward": kv_per_forward,
        "kv_gib_per_forward": kv_per_forward / GIB,
        "kv_bytes_total": kv_total,
        "kv_gib_total": kv_total / GIB,
        "weight_bytes_per_forward": int(weight_bytes_per_forward),
        "weight_gib_per_forward": weight_bytes_per_forward / GIB,
        "weight_bytes_total": weight_total,
        "weight_gib_total": weight_total / GIB,
        "kv_writeback_bytes_total": writeback_total,
        "compute_seconds_total": compute_total_s,
        "forward_compute_ms": float(forward_compute_ms),
    }

    if not h2d_gb_per_s or h2d_gb_per_s <= 0:
        result["available"] = False
        result["reason"] = (
            "no host->device bandwidth available (nothing measured and no "
            "--assume-h2d-gb-s given); the byte counts above stand on their own"
        )
        return result

    h2d_bytes_per_s = float(h2d_gb_per_s) * GB
    h2d_seconds = (kv_total + weight_total) / h2d_bytes_per_s
    if writeback_total and d2h_gb_per_s and d2h_gb_per_s > 0:
        d2h_seconds = writeback_total / (float(d2h_gb_per_s) * GB)
    else:
        d2h_seconds = 0.0
    copy_total_s = h2d_seconds + d2h_seconds

    result.update(
        {
            "available": True,
            "h2d_seconds_total": h2d_seconds,
            "d2h_seconds_total": d2h_seconds,
            "copy_seconds_total": copy_total_s,
            "copy_seconds_per_forward": copy_total_s / forwards,
            "serial_seconds": copy_total_s + compute_total_s,
            "overlap_lower_bound_seconds": max(copy_total_s, compute_total_s),
            "copy_bound": copy_total_s >= compute_total_s,
        }
    )
    return result


# --------------------------------------------------------------------------
# report output: inside the repo, atomic, always written
# --------------------------------------------------------------------------
def resolve_report_path(value: str, root: Path = ROOT) -> Path:
    """Absolute path for ``--json``, refused if it escapes ``root``.

    A probe writes exactly one artifact and it belongs in the repository
    (``.cache/`` is gitignored and is the intended home for run output).
    """
    root = Path(root).expanduser().resolve()
    candidate = Path(os.path.expanduser(str(value)))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = Path(os.path.normpath(str(candidate)))
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ProbeError(
            "the report path must be inside {} (got {}); probes write their "
            "artifacts into the repository, and .cache/ is gitignored for "
            "exactly this".format(root, resolved)
        ) from None
    return resolved


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write ``payload`` as JSON with no observable half-written state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


# --------------------------------------------------------------------------
# allocation, under an explicit ceiling
# --------------------------------------------------------------------------
class MemoryBudget:
    """A hard ceiling on what this probe may hold at once.

    Both device and pinned host memory are scarce and *shared*: pinned pages
    are locked out of the page cache for everyone, and the GPU may be running
    somebody else's job. The ceiling is enforced before the allocator is asked,
    so exceeding it is a probe bug reported as one, not an OOM blamed on the
    box.
    """

    def __init__(self, gpu_limit_bytes: int, host_limit_bytes: int) -> None:
        if gpu_limit_bytes <= 0 or host_limit_bytes <= 0:
            raise ProbeError("memory limits must be positive")
        self.gpu_limit_bytes = int(gpu_limit_bytes)
        self.host_limit_bytes = int(host_limit_bytes)
        self._gpu: Dict[str, int] = {}
        self._host: Dict[str, int] = {}

    @property
    def gpu_reserved(self) -> int:
        return sum(self._gpu.values())

    @property
    def host_reserved(self) -> int:
        return sum(self._host.values())

    def reserve(self, name: str, nbytes: int, kind: str) -> None:
        if kind not in ("gpu", "host"):
            raise ProbeError(f"kind must be 'gpu' or 'host', got {kind!r}")
        book = self._gpu if kind == "gpu" else self._host
        limit = self.gpu_limit_bytes if kind == "gpu" else self.host_limit_bytes
        if name in book:
            raise ProbeError(f"{kind} reservation {name!r} already exists")
        if nbytes <= 0:
            raise ProbeError(f"reservation {name!r} must be positive, got {nbytes!r}")
        current = self.gpu_reserved if kind == "gpu" else self.host_reserved
        if current + nbytes > limit:
            raise ProbeError(
                "{} budget exceeded by {!r}: {:.3f} GiB held + {:.3f} GiB "
                "requested > {:.3f} GiB limit (raise --max-{}-gib only if this "
                "box can really spare it)".format(
                    kind, name, current / GIB, nbytes / GIB, limit / GIB,
                    "gpu" if kind == "gpu" else "host",
                )
            )
        book[name] = int(nbytes)

    def release(self, name: str, kind: str) -> None:
        book = self._gpu if kind == "gpu" else self._host
        book.pop(name, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_limit_gib": self.gpu_limit_bytes / GIB,
            "host_limit_gib": self.host_limit_bytes / GIB,
            "gpu_reserved_gib": self.gpu_reserved / GIB,
            "host_reserved_gib": self.host_reserved / GIB,
        }


class Arena:
    """Scoped allocations that are always released, budget included."""

    def __init__(self, budget: MemoryBudget, device: "torch.device") -> None:
        self.budget = budget
        self.device = device
        self._live: List[Tuple[str, str]] = []
        self._tensors: Dict[str, "torch.Tensor"] = {}

    def __enter__(self) -> "Arena":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.free_all()

    def device_bytes(self, name: str, nbytes: int) -> "torch.Tensor":
        self.budget.reserve(name, nbytes, "gpu")
        try:
            tensor = torch.empty(int(nbytes), dtype=torch.uint8, device=self.device)
        except RuntimeError as exc:  # includes torch.cuda.OutOfMemoryError
            self.budget.release(name, "gpu")
            raise ProbeError(
                "device allocation {!r} of {:.3f} GiB failed on {}: {}: {}".format(
                    name, nbytes / GIB, self.device, type(exc).__name__, exc
                )
            ) from exc
        self._live.append((name, "gpu"))
        self._tensors[name] = tensor
        return tensor

    def pinned_bytes(self, name: str, nbytes: int, fill: bool = True) -> "torch.Tensor":
        self.budget.reserve(name, nbytes, "host")
        try:
            tensor = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
        except RuntimeError as exc:
            self.budget.release(name, "host")
            raise ProbeError(
                "pinned host allocation {!r} of {:.3f} GiB failed: {}: {} -- "
                "page-locked memory is a global resource; this is a real "
                "failure, not something to retry pageable".format(
                    name, nbytes / GIB, type(exc).__name__, exc
                )
            ) from exc
        if fill:
            tensor.fill_(1)
        self._live.append((name, "host"))
        self._tensors[name] = tensor
        return tensor

    def pageable_bytes(self, name: str, nbytes: int) -> "torch.Tensor":
        self.budget.reserve(name, nbytes, "host")
        try:
            tensor = torch.ones(int(nbytes), dtype=torch.uint8)
        except RuntimeError as exc:
            self.budget.release(name, "host")
            raise ProbeError(
                "pageable host allocation {!r} of {:.3f} GiB failed: {}: {}".format(
                    name, nbytes / GIB, type(exc).__name__, exc
                )
            ) from exc
        self._live.append((name, "host"))
        self._tensors[name] = tensor
        return tensor

    def adopt(self, name: str, tensor: "torch.Tensor", kind: str = "gpu") -> "torch.Tensor":
        """Take ownership of an already-allocated tensor, budget included."""
        nbytes = tensor.numel() * tensor.element_size()
        self.budget.reserve(name, nbytes, kind)
        self._live.append((name, kind))
        self._tensors[name] = tensor
        return tensor

    def free_all(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        for name, kind in self._live:
            self._tensors.pop(name, None)
            self.budget.release(name, kind)
        self._live.clear()
        self._tensors.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# a CUDA-resident workload of a chosen duration (never a CPU sleep)
# --------------------------------------------------------------------------
class ComputeWorkload:
    """Occupy the GPU for ~``target_ms`` on a caller-chosen stream.

    ``torch.cuda._sleep`` spins in a kernel for a cycle count, which is exactly
    what is wanted: the GPU is busy, the CPU is not, and nothing depends on
    matmul heuristics. When it is missing, a matmul loop is calibrated to the
    same duration. A ``time.sleep`` would prove nothing -- the copy engine
    would be idle and every overlap number would be a fiction.
    """

    def __init__(self, device: "torch.device", target_ms: float, arena: Arena) -> None:
        if target_ms <= 0:
            raise ProbeError(f"compute target must be positive ms, got {target_ms!r}")
        self.device = device
        self.target_ms = float(target_ms)
        self._a: Optional["torch.Tensor"] = None
        self._b: Optional["torch.Tensor"] = None
        self._repeats = 0
        self._cycles = 0
        if hasattr(torch.cuda, "_sleep"):
            self.mode = "cuda_sleep"
            self._cycles = self._calibrate_sleep()
        else:
            self.mode = "matmul"
            self._repeats = self._calibrate_matmul(arena)

    # -- calibration -----------------------------------------------------
    def _time_ms(self, fn: Callable[[], None]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        fn()  # warm the launch path
        torch.cuda.synchronize(self.device)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(self.device)
        return float(start.elapsed_time(end))

    def _calibrate_sleep(self) -> int:
        probe_cycles = 10_000_000
        elapsed = self._time_ms(lambda: torch.cuda._sleep(probe_cycles))
        if elapsed <= 0:
            raise ProbeError("torch.cuda._sleep calibration measured 0 ms")
        cycles_per_ms = probe_cycles / elapsed
        return max(1, int(cycles_per_ms * self.target_ms))

    def _calibrate_matmul(self, arena: Arena) -> int:
        size = 2048
        self._a = arena.adopt(
            "workload_a",
            torch.randn(size, size, device=self.device, dtype=torch.float16),
        )
        self._b = arena.adopt(
            "workload_b",
            torch.randn(size, size, device=self.device, dtype=torch.float16),
        )
        one = self._time_ms(lambda: torch.mm(self._a, self._b))
        if one <= 0:
            raise ProbeError("matmul calibration measured 0 ms")
        return max(1, int(math.ceil(self.target_ms / one)))

    # -- use -------------------------------------------------------------
    def launch(self) -> None:
        """Enqueue the workload on the *current* stream."""
        if self.mode == "cuda_sleep":
            torch.cuda._sleep(self._cycles)
            return
        assert self._a is not None and self._b is not None
        for _ in range(self._repeats):
            torch.mm(self._a, self._b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "target_ms": self.target_ms,
            "sleep_cycles": self._cycles,
            "matmul_repeats": self._repeats,
        }


# --------------------------------------------------------------------------
# measurement primitives
# --------------------------------------------------------------------------
def measure(
    body: Callable[[], None],
    *,
    device: "torch.device",
    warmup: int,
    iters: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tuple[List[float], List[float]]:
    """Run ``body`` ``warmup + iters`` times; return (event ms, wall ms).

    ``body`` enqueues work on ``stream`` (or the current stream). Every
    iteration ends in a device ``synchronize``, which is inside the wall-clock
    window and outside the event window -- so the two columns together say how
    much of the cost is the copy and how much is launch plus sync.
    """
    event_ms: List[float] = []
    wall_ms: List[float] = []
    for index in range(warmup + iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        if stream is not None:
            with torch.cuda.stream(stream):
                start.record(stream)
                body()
                end.record(stream)
        else:
            start.record()
            body()
            end.record()
        torch.cuda.synchronize(device)
        wall_end = time.perf_counter()
        if index >= warmup:
            event_ms.append(float(start.elapsed_time(end)))
            wall_ms.append((wall_end - wall_start) * 1000.0)
    return event_ms, wall_ms


def measure_wall(
    body: Callable[[], None],
    *,
    device: "torch.device",
    warmup: int,
    iters: int,
) -> List[float]:
    """Wall-clock ms for multi-stream bodies, ``synchronize`` included."""
    wall_ms: List[float] = []
    for index in range(warmup + iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        body()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        if index >= warmup:
            wall_ms.append((end - start) * 1000.0)
    return wall_ms


def hidden_fraction(copy_ms: float, compute_ms: float, both_ms: float) -> float:
    """Fraction of ``copy_ms`` that ran underneath compute, in [0, 1].

    The overlapped duration is ``copy + compute - both``, clamped into
    ``[0, min(copy, compute)]`` -- it cannot be negative, and it cannot exceed
    the shorter of the two, so a copy twice as long as the compute it hides
    behind reports 0.5 even when the wall time is already at its floor. That
    ceiling is the point: it says how much of the transfer the pipeline still
    has to pay for in the open.
    """
    if copy_ms <= 0:
        raise ProbeError("copy duration must be positive")
    overlapped = max(0.0, min(copy_ms + compute_ms - both_ms,
                              min(copy_ms, compute_ms)))
    return overlapped / copy_ms


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
@dataclass
class Context:
    device: "torch.device"
    budget: MemoryBudget
    geometry: KVGeometry
    block_bytes: int
    warmup: int
    iters: int
    compute_ms: float
    pipeline_blocks: int


def scenario_h2d_pageable_blocking(ctx: Context) -> Dict[str, Any]:
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.pageable_bytes("pageable_src", ctx.block_bytes)
        dst = arena.device_bytes("device_dst", ctx.block_bytes)
        event_ms, wall_ms = measure(
            lambda: dst.copy_(src, non_blocking=False),
            device=ctx.device, warmup=ctx.warmup, iters=ctx.iters,
        )
    return summarize_transfer(
        ctx.block_bytes, event_ms, wall_ms,
        note="pageable host memory: the driver stages through its own pinned "
             "buffer, so this is the floor, not the link speed",
    )


def scenario_h2d_pinned_blocking(ctx: Context) -> Dict[str, Any]:
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.pinned_bytes("pinned_src", ctx.block_bytes)
        dst = arena.device_bytes("device_dst", ctx.block_bytes)
        event_ms, wall_ms = measure(
            lambda: dst.copy_(src, non_blocking=False),
            device=ctx.device, warmup=ctx.warmup, iters=ctx.iters,
        )
    return summarize_transfer(
        ctx.block_bytes, event_ms, wall_ms,
        note="pinned source, blocking copy on the default stream",
    )


def scenario_h2d_pinned_nonblocking(ctx: Context) -> Dict[str, Any]:
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.pinned_bytes("pinned_src", ctx.block_bytes)
        dst = arena.device_bytes("device_dst", ctx.block_bytes)
        stream = torch.cuda.Stream(device=ctx.device)
        event_ms, wall_ms = measure(
            lambda: dst.copy_(src, non_blocking=True),
            device=ctx.device, warmup=ctx.warmup, iters=ctx.iters, stream=stream,
        )
    return summarize_transfer(
        ctx.block_bytes, event_ms, wall_ms,
        note="pinned source issued on a dedicated copy stream",
    )


def scenario_d2h_pinned_nonblocking(ctx: Context) -> Dict[str, Any]:
    nbytes = ctx.geometry.per_layer_bytes
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.device_bytes("device_src", nbytes)
        dst = arena.pinned_bytes("pinned_dst", nbytes)
        stream = torch.cuda.Stream(device=ctx.device)
        event_ms, wall_ms = measure(
            lambda: dst.copy_(src, non_blocking=True),
            device=ctx.device, warmup=ctx.warmup, iters=ctx.iters, stream=stream,
        )
    return summarize_transfer(
        nbytes, event_ms, wall_ms,
        note="one layer of K+V evicted to pinned host memory",
    )


def _pipeline(ctx: Context, buffers: int) -> Dict[str, Any]:
    """``pipeline_blocks`` copies feeding a consumer, with N landing buffers.

    The consumer is a real CUDA workload on its own stream and it *owns* the
    buffer it reads: the next copy into that buffer waits on the consumer's
    event. With one buffer that dependency is with the immediately preceding
    consumer, so copy and compute serialise; with two it is one step further
    back, so they can overlap. Nothing else differs between the two runs.
    """
    if buffers not in (1, 2):
        raise ProbeError(f"buffers must be 1 or 2, got {buffers!r}")
    blocks = ctx.pipeline_blocks
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.pinned_bytes("pinned_src", ctx.block_bytes)
        dsts = [
            arena.device_bytes(f"device_dst_{i}", ctx.block_bytes)
            for i in range(buffers)
        ]
        workload = ComputeWorkload(ctx.device, ctx.compute_ms, arena)
        copy_stream = torch.cuda.Stream(device=ctx.device)
        compute_stream = torch.cuda.Stream(device=ctx.device)

        # solo costs, for the hidden-fraction arithmetic
        solo_copy_ms, _ = measure(
            lambda: dsts[0].copy_(src, non_blocking=True),
            device=ctx.device, warmup=ctx.warmup, iters=max(2, ctx.iters // 2),
            stream=copy_stream,
        )
        solo_compute_ms, _ = measure(
            workload.launch,
            device=ctx.device, warmup=1, iters=max(2, ctx.iters // 2),
            stream=compute_stream,
        )

        def run_pipeline() -> None:
            consumed: List[Optional[torch.cuda.Event]] = [None] * blocks
            for i in range(blocks):
                slot = i % buffers
                dependency = consumed[i - buffers] if i >= buffers else None
                with torch.cuda.stream(copy_stream):
                    if dependency is not None:
                        copy_stream.wait_event(dependency)
                    dsts[slot].copy_(src, non_blocking=True)
                    copy_done = torch.cuda.Event()
                    copy_done.record(copy_stream)
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(copy_done)
                    workload.launch()
                    released = torch.cuda.Event()
                    released.record(compute_stream)
                consumed[i] = released

        wall_ms = measure_wall(
            run_pipeline, device=ctx.device, warmup=1, iters=max(2, ctx.iters // 2)
        )
        workload_info = workload.to_dict()

    copy_ms = percentile(solo_copy_ms, 0.50)
    compute_ms = percentile(solo_compute_ms, 0.50)
    total_ms = percentile(wall_ms, 0.50)
    total_bytes = ctx.block_bytes * blocks
    return {
        "buffers": buffers,
        "blocks": blocks,
        "bytes_per_block": ctx.block_bytes,
        "bytes_total": total_bytes,
        "gib_total": total_bytes / GIB,
        "solo_copy_ms": describe(solo_copy_ms),
        "solo_compute_ms": describe(solo_compute_ms),
        "pipeline_wall_ms": describe(wall_ms),
        "serial_reference_ms": blocks * (copy_ms + compute_ms),
        "ideal_overlap_ms": blocks * max(copy_ms, compute_ms),
        "effective_bandwidth": bandwidth(total_bytes, total_ms),
        "copy_hidden_fraction": hidden_fraction(
            blocks * copy_ms, blocks * compute_ms, total_ms
        ),
        "workload": workload_info,
    }


def scenario_pipeline_single_buffer(ctx: Context) -> Dict[str, Any]:
    return _pipeline(ctx, buffers=1)


def scenario_pipeline_double_buffer(ctx: Context) -> Dict[str, Any]:
    return _pipeline(ctx, buffers=2)


def _weight_and_kv(ctx: Context, parallel: bool) -> Dict[str, Any]:
    """A block of weights and one layer of KV, on two streams or on one."""
    kv_bytes = ctx.geometry.per_layer_bytes
    with Arena(ctx.budget, ctx.device) as arena:
        w_src = arena.pinned_bytes("pinned_weight", ctx.block_bytes)
        w_dst = arena.device_bytes("device_weight", ctx.block_bytes)
        kv_src = arena.pinned_bytes("pinned_kv", kv_bytes)
        kv_dst = arena.device_bytes("device_kv", kv_bytes)
        stream_a = torch.cuda.Stream(device=ctx.device)
        stream_b = torch.cuda.Stream(device=ctx.device) if parallel else stream_a

        def body() -> None:
            with torch.cuda.stream(stream_a):
                w_dst.copy_(w_src, non_blocking=True)
            with torch.cuda.stream(stream_b):
                kv_dst.copy_(kv_src, non_blocking=True)

        wall_ms = measure_wall(
            body, device=ctx.device, warmup=ctx.warmup, iters=ctx.iters
        )
    total_bytes = ctx.block_bytes + kv_bytes
    return {
        "parallel": parallel,
        "streams": 2 if parallel else 1,
        "weight_bytes": ctx.block_bytes,
        "kv_bytes": kv_bytes,
        "bytes_total": total_bytes,
        "gib_total": total_bytes / GIB,
        "wall_ms": describe(wall_ms),
        "aggregate_bandwidth_p50": bandwidth(total_bytes, percentile(wall_ms, 0.50)),
        "aggregate_bandwidth_p95": bandwidth(total_bytes, percentile(wall_ms, 0.95)),
        "note": (
            "two copy streams issuing at once; most parts have a single H2D DMA "
            "engine, so agreement with the serial control is the expected result"
            if parallel else
            "the control: the same two transfers, back to back on one stream"
        ),
    }


def scenario_parallel_two_copy_streams(ctx: Context) -> Dict[str, Any]:
    return _weight_and_kv(ctx, parallel=True)


def scenario_serial_one_copy_stream(ctx: Context) -> Dict[str, Any]:
    return _weight_and_kv(ctx, parallel=False)


def scenario_compute_overlap(ctx: Context) -> Dict[str, Any]:
    """Next layer's KV lands while the current layer computes."""
    kv_bytes = ctx.geometry.per_layer_bytes
    with Arena(ctx.budget, ctx.device) as arena:
        src = arena.pinned_bytes("pinned_kv", kv_bytes)
        dst = arena.device_bytes("device_kv", kv_bytes)
        workload = ComputeWorkload(ctx.device, ctx.compute_ms, arena)
        copy_stream = torch.cuda.Stream(device=ctx.device)
        compute_stream = torch.cuda.Stream(device=ctx.device)

        copy_only_ms, copy_wall_ms = measure(
            lambda: dst.copy_(src, non_blocking=True),
            device=ctx.device, warmup=ctx.warmup, iters=ctx.iters, stream=copy_stream,
        )
        compute_only_ms, _ = measure(
            workload.launch,
            device=ctx.device, warmup=1, iters=ctx.iters, stream=compute_stream,
        )

        def both() -> None:
            with torch.cuda.stream(copy_stream):
                dst.copy_(src, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                workload.launch()

        both_wall_ms = measure_wall(
            both, device=ctx.device, warmup=ctx.warmup, iters=ctx.iters
        )
        workload_info = workload.to_dict()

    copy_ms = percentile(copy_only_ms, 0.50)
    compute_ms = percentile(compute_only_ms, 0.50)
    both_ms = percentile(both_wall_ms, 0.50)
    return {
        "kv_bytes": kv_bytes,
        "kv_gib": kv_bytes / GIB,
        "copy_only": summarize_transfer(kv_bytes, copy_only_ms, copy_wall_ms),
        "compute_only_ms": describe(compute_only_ms),
        "overlapped_wall_ms": describe(both_wall_ms),
        "serial_reference_ms": copy_ms + compute_ms,
        "ideal_overlap_ms": max(copy_ms, compute_ms),
        "copy_hidden_fraction": hidden_fraction(copy_ms, compute_ms, both_ms),
        "compute_hidden_fraction": hidden_fraction(compute_ms, copy_ms, both_ms),
        "workload": workload_info,
        "note": (
            "the compute stream runs a real CUDA workload (never a CPU sleep, "
            "which would leave the GPU idle and fake the overlap)"
        ),
    }


SCENARIO_FUNCS: Dict[str, Callable[[Context], Dict[str, Any]]] = {
    "h2d_pageable_blocking": scenario_h2d_pageable_blocking,
    "h2d_pinned_blocking": scenario_h2d_pinned_blocking,
    "h2d_pinned_nonblocking": scenario_h2d_pinned_nonblocking,
    "d2h_pinned_nonblocking": scenario_d2h_pinned_nonblocking,
    "pipeline_single_buffer": scenario_pipeline_single_buffer,
    "pipeline_double_buffer": scenario_pipeline_double_buffer,
    "parallel_two_copy_streams": scenario_parallel_two_copy_streams,
    "serial_one_copy_stream": scenario_serial_one_copy_stream,
    "compute_overlap": scenario_compute_overlap,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def device_info(device: "torch.device") -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "requested": str(device),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": props.total_memory / GIB,
                "capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return info


def measured_bandwidths(measurements: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Headline GB/s used for the estimates: pinned async, both directions."""
    h2d = measurements.get("h2d_pinned_nonblocking") or measurements.get(
        "h2d_pinned_blocking"
    )
    d2h = measurements.get("d2h_pinned_nonblocking")

    def pick(entry: Any) -> Optional[float]:
        if not isinstance(entry, dict):
            return None
        band = entry.get("bandwidth_p50")
        if isinstance(band, dict):
            return float(band["gb_per_s"])
        return None

    return {"h2d_gb_per_s": pick(h2d), "d2h_gb_per_s": pick(d2h)}


def build_estimates(args: argparse.Namespace, geometry: KVGeometry,
                    measurements: Dict[str, Any]) -> Dict[str, Any]:
    measured = measured_bandwidths(measurements)
    h2d = measured["h2d_gb_per_s"]
    d2h = measured["d2h_gb_per_s"]
    source = "measured"
    if h2d is None and args.assume_h2d_gb_s:
        h2d = float(args.assume_h2d_gb_s)
        source = "assumed (--assume-h2d-gb-s)"
    if d2h is None and args.assume_d2h_gb_s:
        d2h = float(args.assume_d2h_gb_s)

    block_bytes = int(round(args.block_gib * GIB))
    total_weight_bytes = (
        int(round(args.weight_total_gib * GIB))
        if args.weight_total_gib is not None
        else block_bytes * geometry.layers
    )
    if args.weight_offload_gib is not None:
        offload_bytes = int(round(args.weight_offload_gib * GIB))
        offload_source = "explicit (--weight-offload-gib)"
    else:
        offload_bytes = weight_offload_bytes(
            total_weight_bytes,
            int(round(args.vram_cap_gib * GIB)),
            int(round(args.vram_other_gib * GIB)),
        )
        offload_source = (
            "derived: max(0, weight_total - (vram_cap - vram_other))"
        )

    all_kv_on_cpu = estimate_rollout(
        forwards=args.forwards,
        layers=geometry.layers,
        per_layer_kv_bytes=geometry.per_layer_bytes,
        weight_bytes_per_forward=0,
        h2d_gb_per_s=h2d,
        d2h_gb_per_s=d2h,
        forward_compute_ms=args.forward_compute_ms,
        bandwidth_source=source,
    )
    kv_and_weights = estimate_rollout(
        forwards=args.forwards,
        layers=geometry.layers,
        per_layer_kv_bytes=geometry.per_layer_bytes,
        weight_bytes_per_forward=offload_bytes,
        h2d_gb_per_s=h2d,
        d2h_gb_per_s=d2h,
        forward_compute_ms=args.forward_compute_ms,
        bandwidth_source=source,
    )
    return {
        "bandwidth_used": {"h2d_gb_per_s": h2d, "d2h_gb_per_s": d2h,
                           "source": source},
        "weight_offload": {
            "vram_cap_gib": args.vram_cap_gib,
            "vram_other_gib": args.vram_other_gib,
            "weight_total_gib": total_weight_bytes / GIB,
            "weight_offload_bytes_per_forward": offload_bytes,
            "weight_offload_gib_per_forward": offload_bytes / GIB,
            "source": offload_source,
        },
        "all_kv_on_cpu": all_kv_on_cpu,
        "all_kv_on_cpu_plus_weight_offload": kv_and_weights,
        "caveat": (
            "copy-time bounds only. serial_seconds assumes nothing overlaps; "
            "overlap_lower_bound_seconds is max(copy, compute) and no "
            "implementation beats it. Neither includes VAE decode, sampling "
            "overhead or allocator churn."
        ),
    }


def render(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    info = report["device"]
    lines.append(
        "device: {} ({}), torch {} / cuda {}".format(
            info.get("name", info["requested"]), info["requested"],
            info["torch"], info["torch_cuda"],
        )
    )
    geo = report["geometry"]
    lines.append(
        "KV geometry: {rows} rows x {heads} heads x {head_dim} @ {dtype} -> "
        "{per_layer_gib:.4f} GiB/layer (K+V), {all_layers_gib:.2f} GiB x "
        "{layers} layers".format(**geo)
    )
    for name in SCENARIOS:
        entry = report["measurements"].get(name)
        if entry is None:
            skip = report["skipped"].get(name)
            if skip:
                lines.append("[SKIP] {}: {}".format(name, skip))
            continue
        if "bandwidth_p50" in entry:
            lines.append(
                "[OK]   {:<26} {:.3f} GiB  p50 {:.2f} GB/s ({:.2f} GiB/s)  "
                "p95 {:.2f} GB/s".format(
                    name, entry["gib"],
                    entry["bandwidth_p50"]["gb_per_s"],
                    entry["bandwidth_p50"]["gib_per_s"],
                    entry["bandwidth_p95"]["gb_per_s"],
                )
            )
        elif "copy_hidden_fraction" in entry:
            lines.append(
                "[OK]   {:<26} copy hidden {:.1%}  (serial {:.1f} ms, ideal "
                "{:.1f} ms, measured p50 {:.1f} ms)".format(
                    name, entry["copy_hidden_fraction"],
                    entry["serial_reference_ms"], entry["ideal_overlap_ms"],
                    entry.get("overlapped_wall_ms", entry.get("pipeline_wall_ms"))["p50"],
                )
            )
        elif "aggregate_bandwidth_p50" in entry:
            lines.append(
                "[OK]   {:<26} {:.3f} GiB in p50 {:.1f} ms -> {:.2f} GB/s".format(
                    name, entry["gib_total"], entry["wall_ms"]["p50"],
                    entry["aggregate_bandwidth_p50"]["gb_per_s"],
                )
            )
    for name, detail in report["failed"].items():
        lines.append("[FAIL] {}: {}".format(name, detail))
    for key in ("all_kv_on_cpu", "all_kv_on_cpu_plus_weight_offload"):
        est = report["estimates"][key]
        if not est.get("available"):
            lines.append("estimate {}: {}".format(key, est.get("reason", "unavailable")))
            continue
        lines.append(
            "estimate {:<34} {:.1f} GiB over {} forwards -> copy {:.1f} s, "
            "serial {:.1f} s, ideal-overlap floor {:.1f} s".format(
                key,
                est["kv_gib_total"] + est["weight_gib_total"],
                est["forwards"], est["copy_seconds_total"],
                est["serial_seconds"], est["overlap_lower_bound_seconds"],
            )
        )
    lines.append("RESULT: {}".format("ok" if report["ok"] else "FAILURES PRESENT"))
    return "\n".join(lines)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shape = parser.add_argument_group("KV geometry")
    shape.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS,
                       help="retained KV rows seen by one forward (default: %(default)s)")
    shape.add_argument("--heads", type=positive_int, default=DEFAULT_HEADS)
    shape.add_argument("--head-dim", type=positive_int, default=DEFAULT_HEAD_DIM)
    shape.add_argument("--layers", type=positive_int, default=DEFAULT_LAYERS)
    shape.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    shape.add_argument("--block-gib", type=positive_float, default=DEFAULT_BLOCK_GIB,
                       help="one transformer block's weights, GiB (default: %(default)s)")

    run = parser.add_argument_group("measurement")
    run.add_argument("--device", default=os.environ.get(DEVICE_ENV_VAR) or "cuda")
    run.add_argument("--warmup", type=non_negative_int, default=3)
    run.add_argument("--iters", type=positive_int, default=10)
    run.add_argument("--compute-ms", type=positive_float, default=8.0,
                     help="target duration of the simulated per-layer compute")
    run.add_argument("--pipeline-blocks", type=positive_int, default=6)
    run.add_argument("--only", default=None,
                     help="comma-separated subset of: " + ", ".join(SCENARIOS))
    run.add_argument("--max-gpu-gib", type=positive_float, default=6.0,
                     help="ceiling on simultaneously held device memory")
    run.add_argument("--max-host-gib", type=positive_float, default=8.0,
                     help="ceiling on simultaneously held host memory")

    est = parser.add_argument_group("rollout estimate")
    est.add_argument("--forwards", type=positive_int, default=DEFAULT_FORWARDS,
                     help="DiT forwards in the rollout (default: %(default)s, "
                          "the 192-frame published request)")
    est.add_argument("--vram-cap-gib", type=positive_float, default=24.0)
    est.add_argument("--vram-other-gib", type=non_negative_float, default=0.0,
                     help="non-weight bytes the card must hold at the same time")
    est.add_argument("--weight-total-gib", type=positive_float, default=None,
                     help="whole-model weight bytes (default: layers x --block-gib)")
    est.add_argument("--weight-offload-gib", type=non_negative_float, default=None,
                     help="override the derived per-forward weight streaming volume")
    est.add_argument("--forward-compute-ms", type=non_negative_float, default=0.0,
                     help="per-forward GPU compute, for the overlap bound")
    est.add_argument("--assume-h2d-gb-s", type=positive_float, default=None,
                     help="bandwidth for the estimates when nothing was measured")
    est.add_argument("--assume-d2h-gb-s", type=positive_float, default=None)

    parser.add_argument("--json", default=DEFAULT_REPORT,
                        help="report path, must be inside the repository "
                             "(default: %(default)s)")
    return parser


def selected_scenarios(only: Optional[str]) -> List[str]:
    if not only:
        return list(SCENARIOS)
    names = [part.strip() for part in only.split(",") if part.strip()]
    unknown = [name for name in names if name not in SCENARIO_FUNCS]
    if unknown:
        raise ProbeError(
            "unknown scenario(s) {}; known: {}".format(unknown, ", ".join(SCENARIOS))
        )
    return names


def run_probe(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    geometry = KVGeometry(
        rows=args.rows, heads=args.heads, head_dim=args.head_dim,
        dtype=args.dtype, layers=args.layers,
    )
    device = torch.device(args.device)
    budget = MemoryBudget(
        int(round(args.max_gpu_gib * GIB)), int(round(args.max_host_gib * GIB))
    )
    report: Dict[str, Any] = {
        "schema": 1,
        "tool": "probe_offload_bandwidth",
        "config": {
            key: getattr(args, key)
            for key in (
                "rows", "heads", "head_dim", "layers", "dtype", "block_gib",
                "device", "warmup", "iters", "compute_ms", "pipeline_blocks",
                "only", "max_gpu_gib", "max_host_gib", "forwards",
                "vram_cap_gib", "vram_other_gib", "weight_total_gib",
                "weight_offload_gib", "forward_compute_ms",
                "assume_h2d_gb_s", "assume_d2h_gb_s", "json",
            )
        },
        "device": device_info(device),
        "geometry": geometry.to_dict(),
        "budget": budget.to_dict(),
        "measurements": {},
        "skipped": {},
        "failed": {},
    }

    names = selected_scenarios(args.only)
    report["requested_scenarios"] = names

    if device.type != "cuda" or not torch.cuda.is_available():
        reason = (
            "device {} is not a usable CUDA device (torch.cuda.is_available()="
            "{}); host<->device bandwidth cannot be measured here".format(
                device, torch.cuda.is_available()
            )
        )
        for name in names:
            report["skipped"][name] = reason
        report["estimates"] = build_estimates(args, geometry, report["measurements"])
        report["ok"] = False
        report["reason"] = reason
        return report, EXIT_ENVIRONMENT

    ctx = Context(
        device=device,
        budget=budget,
        geometry=geometry,
        block_bytes=int(round(args.block_gib * GIB)),
        warmup=args.warmup,
        iters=args.iters,
        compute_ms=args.compute_ms,
        pipeline_blocks=args.pipeline_blocks,
    )
    for name in names:
        try:
            report["measurements"][name] = SCENARIO_FUNCS[name](ctx)
        except Exception as exc:  # noqa: BLE001 - the failure is the evidence
            report["failed"][name] = "{}: {}".format(type(exc).__name__, exc)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    report["peak_memory"] = {
        "torch_max_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "torch_max_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
    }
    report["estimates"] = build_estimates(args, geometry, report["measurements"])
    report["ok"] = not report["failed"] and bool(report["measurements"])
    return report, (EXIT_OK if report["ok"] else EXIT_FAILED)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report_path = resolve_report_path(args.json)
    except ProbeError as exc:
        print("PROBE REFUSED: {}".format(exc), file=sys.stderr)
        return EXIT_FAILED

    try:
        report, code = run_probe(args)
    except ProbeError as exc:
        report = {
            "schema": 1,
            "tool": "probe_offload_bandwidth",
            "ok": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
        code = EXIT_FAILED
        print("PROBE FAILED: {}".format(exc), file=sys.stderr)
        atomic_write_json(report_path, report)
        return code
    except Exception as exc:  # noqa: BLE001 - still leave evidence behind
        atomic_write_json(
            report_path,
            {
                "schema": 1,
                "tool": "probe_offload_bandwidth",
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            },
        )
        raise

    print(render(report))
    atomic_write_json(report_path, report)
    print("report: {}".format(report_path))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
