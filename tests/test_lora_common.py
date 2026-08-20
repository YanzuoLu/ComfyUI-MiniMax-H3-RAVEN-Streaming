"""Shared helpers for the RAVEN M0 LoRA lane tests.

Collected by pytest (the file name matches ``test_lora_*.py``) but contains no
tests of its own - only synthetic fixtures: a toy DiT with the same module
paths as the official model, a synthetic PEFT safetensors writer, and a
reference implementation of ``peft.tuners.lora.layer.Linear.forward``.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.utils._python_dispatch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raven_streaming import lora as rlora  # noqa: E402

# A tiny stand-in for the official topology: same module paths / categories,
# small dimensions. 1 refiner block + 2 DiT blocks -> 12 core, 3 adaln, 2 time,
# 5 boundary = 22 modules.
TOY_CONFIG = rlora.RavenBaseConfig(
    hidden_size=16,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=4,
    ffn_hidden_size=12,
    latents_dim=4,
    audio_latents_dim=6,
    patch_size=(1, 2, 2),
    text_dim=10,
    timestep_input_dim=8,
    time_embed_hidden_size=16,
    time_embed_dim=14,
)

TOY_COUNTS = {"core": 12, "adaln": 3, "time": 2, "boundary": 5}

_ST_NAMES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
}


# --------------------------------------------------------------------------
# toy base model
# --------------------------------------------------------------------------
def _ensure_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        child = getattr(node, part, None)
        if child is None:
            child = nn.Module()
            node.add_module(part, child)
        node = child
    return node, parts[-1]


def build_toy_dit(
    config: rlora.RavenBaseConfig = TOY_CONFIG,
    dtype: torch.dtype = torch.float32,
    bias: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> nn.Module:
    """A module tree with exactly the base model's LoRA-target paths."""
    gen = torch.Generator().manual_seed(seed)
    root = nn.Module()
    for path, entry in config.modules().items():
        parent, leaf = _ensure_parent(root, path)
        lin = nn.Linear(entry.in_features, entry.out_features, bias=bias, dtype=dtype, device=device)
        if device is None or device.type != "meta":
            with torch.no_grad():
                lin.weight.copy_(
                    torch.randn(entry.weight_shape, generator=gen, dtype=torch.float32).to(dtype)
                    * 0.05
                )
                if bias:
                    lin.bias.copy_(
                        torch.randn(entry.out_features, generator=gen, dtype=torch.float32).to(dtype)
                        * 0.05
                    )
        parent.add_module(leaf, lin)
    # the full non-pruned model has no adaln_t_table buffer
    root.register_buffer("rope_inv_freq", torch.zeros(4))
    return root


def make_pruned_dit(config: rlora.RavenBaseConfig = TOY_CONFIG) -> nn.Module:
    """Curve-form / pruned stand-in: adaln_t_table buffer, no time_embedder."""
    root = build_toy_dit(config)
    del root._modules["time_embedder"]
    root.register_buffer("adaln_t_table", torch.zeros(8, config.time_embed_dim))
    return root


class ToyWrapper(nn.Module):
    """Mimics ``comfy.model_base.BaseModel``: ``.diffusion_model`` submodule."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.diffusion_model = dit


# --------------------------------------------------------------------------
# synthetic PEFT payloads
# --------------------------------------------------------------------------
def synthetic_lora_tensors(
    config: rlora.RavenBaseConfig = TOY_CONFIG,
    rank: int = 4,
    dtype: torch.dtype = torch.float32,
    prefix: str = rlora.PEFT_PREFIX,
    adapter: Optional[str] = None,
    seed: int = 1234,
    scale: float = 0.1,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    suffix = ".weight" if adapter is None else ".{}.weight".format(adapter)
    for path, entry in config.modules().items():
        a = torch.randn(rank, entry.in_features, generator=gen, dtype=torch.float32) * scale
        b = torch.randn(entry.out_features, rank, generator=gen, dtype=torch.float32) * scale
        out["{}{}.lora_A{}".format(prefix, path, suffix)] = a.to(dtype)
        out["{}{}.lora_B{}".format(prefix, path, suffix)] = b.to(dtype)
    return out


def synthetic_weight_pairs(
    config: rlora.RavenBaseConfig = TOY_CONFIG, rank: int = 4, seed: int = 7, scale: float = 0.1
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """``module path -> (A, B)`` without going through a file."""
    gen = torch.Generator().manual_seed(seed)
    pairs = {}
    for path, entry in config.modules().items():
        a = torch.randn(rank, entry.in_features, generator=gen) * scale
        b = torch.randn(entry.out_features, rank, generator=gen) * scale
        pairs[path] = (a, b)
    return pairs


def write_safetensors(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Optional[Mapping[str, str]] = None
) -> Path:
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs: List[bytes] = []
    offset = 0
    for name in tensors:
        t = tensors[name].detach().cpu().contiguous()
        raw = t.flatten().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": _ST_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for raw in blobs:
            fh.write(raw)
    return path


def header_from_shapes(
    shapes: Mapping[str, Sequence[int]],
    dtype: str = "F32",
    metadata: Optional[Mapping[str, str]] = None,
) -> rlora.SafetensorsHeader:
    """Build a header object directly - no file, no tensor memory."""
    tensors: Dict[str, rlora.TensorInfo] = {}
    offset = 0
    itemsize = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}[dtype]
    for name, shape in shapes.items():
        n = 1
        for d in shape:
            n *= int(d)
        nbytes = n * itemsize
        tensors[name] = rlora.TensorInfo(
            name=name, dtype=dtype, shape=tuple(int(d) for d in shape), begin=offset, end=offset + nbytes
        )
        offset += nbytes
    return rlora.SafetensorsHeader(
        tensors=tensors, metadata=dict(metadata or {}), data_offset=8
    )


def full_scale_shapes(
    config: Optional[rlora.RavenBaseConfig] = None,
    rank: int = 128,
    prefix: str = rlora.PEFT_PREFIX,
) -> Dict[str, Tuple[int, int]]:
    """Metadata-only shapes for the published 532-tensor / 266-module file."""
    cfg = config or rlora.RavenBaseConfig()
    shapes: Dict[str, Tuple[int, int]] = {}
    for path, entry in cfg.modules().items():
        shapes["{}{}.lora_A.weight".format(prefix, path)] = (rank, entry.in_features)
        shapes["{}{}.lora_B.weight".format(prefix, path)] = (entry.out_features, rank)
    return shapes


# --------------------------------------------------------------------------
# PEFT reference
# --------------------------------------------------------------------------
def peft_reference_forward(
    base_out: torch.Tensor,
    x: torch.Tensor,
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    scalings: Iterable[float],
) -> torch.Tensor:
    """Mirror of ``peft.tuners.lora.layer.Linear.forward``.

    ``result = base(x)``; for each active adapter
    ``result = result + lora_B(lora_A(x.to(A.dtype))) * scaling``; finally
    ``result.to(torch_result_dtype)``.
    """
    result = base_out
    dtype = result.dtype
    for (a, b), scaling in zip(pairs, scalings):
        xa = x.to(a.dtype)
        result = result + torch.nn.functional.linear(
            torch.nn.functional.linear(xa, a), b
        ) * scaling
    return result.to(dtype)


class NoBigAllocation(torch.overrides.TorchFunctionMode):
    """Fail if any torch op produces a tensor larger than ``limit`` elements.

    Used as a dense-``B @ A`` sentinel: for the published shapes a merged delta
    would be e.g. 21504x5376 = 115M elements, far above the limit, while the
    activation-side path only ever produces [tokens, rank] and [tokens, out].
    """

    def __init__(self, limit: int = 10_000_000):
        super().__init__()
        self.limit = limit
        self.max_seen = 0

    def _check(self, obj):
        if isinstance(obj, torch.Tensor):
            n = obj.numel()
            self.max_seen = max(self.max_seen, n)
            if n > self.limit:
                raise AssertionError(
                    "dense allocation sentinel: tensor with {} elements (shape {}) "
                    "exceeds the {} element limit".format(n, tuple(obj.shape), self.limit)
                )
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self._check(o)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._check(out)
        return out


def _returns_alias(func) -> bool:
    """True for view/in-place ops whose result aliases an input (not an allocation)."""
    try:
        return any(r.alias_info is not None for r in func._schema.returns)
    except Exception:  # noqa: BLE001 - not all overloads carry a schema
        return False


class ChunkAllocationSentinel(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch-level guard: no *allocation* may exceed ``limit`` elements.

    Views and in-place writes (which alias an existing buffer, e.g. the base
    output being written back chunk by chunk) are exempt; every genuinely new
    tensor - the promoted activation, the low-rank intermediate, the residual,
    the FP32 accumulator - must stay inside one row chunk. A merged ``B @ A``
    would blow straight through it.
    """

    def __init__(self, limit: int, forbidden_shapes=()):
        super().__init__()
        self.limit = int(limit)
        self.forbidden_shapes = {tuple(s) for s in forbidden_shapes}
        self.max_alloc = 0
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if _returns_alias(func):
            return out
        for t in (out if isinstance(out, (list, tuple)) else (out,)):
            if not isinstance(t, torch.Tensor):
                continue
            self.ops.append((str(func), tuple(t.shape)))
            self.max_alloc = max(self.max_alloc, t.numel())
            if tuple(t.shape) in self.forbidden_shapes:
                raise AssertionError(
                    "forbidden dense allocation {} from {}".format(tuple(t.shape), func)
                )
            if t.numel() > self.limit:
                raise AssertionError(
                    "{} allocated {} elements (shape {}), limit is {}".format(
                        func, t.numel(), tuple(t.shape), self.limit
                    )
                )
        return out
