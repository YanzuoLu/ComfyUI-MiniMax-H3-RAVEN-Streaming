"""RAVEN M0 LoRA lane: safetensors parsing, strict PEFT->Comfy key mapping, attach.

Scope (M0)
----------
Load an already-published PEFT LoRA for MiniMax-H3 (RAVEN streaming) onto the
*official, full, non-pruned* ComfyUI MiniMaxH3 diffusion model:

* the published file is a plain ``safetensors`` PEFT dump whose keys carry the
  ``base_model.model.dit.`` prefix, 532 tensors / 266 modules, FP32 ``lora_A`` /
  ``lora_B``, ``r == lora_alpha == 128``;
* the base model is the one built by ``comfy/ldm/minimax/model.py``
  (``MiniMaxH3Model``), reachable through ``ModelPatcher.model.diffusion_model``,
  so the patch keys are ``diffusion_model.<module path>.weight``;
* ``B @ A`` is **never** materialised: the delta is applied activation-side at
  runtime (see :mod:`raven_streaming.runtime_linear`), so the base module keeps
  its own ``.weight`` / ``.bias`` and the official ``LoraLoaderModelOnly``
  generic-format LoRA path keeps hitting the exact same keys afterwards;
* no QKV reinterleaving and no ``fc1`` half swap are performed. The published
  adapter was trained against this exact layout; the module inventory below
  (208 core + 51 adaln + 2 time + 5 boundary = 266) matches the Comfy port 1:1,
  which is what makes the identity mapping legitimate. Any deviation must fail
  loudly rather than be silently "fixed" by a permutation.

Everything that can go wrong (unknown key, missing module, duplicate tensor,
wrong shape, inconsistent rank, pruned/curve-form base) raises; nothing is
skipped with a warning.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from raven_streaming import runtime_linear

__all__ = [
    "RavenLoraError",
    "SafetensorsFormatError",
    "UnexpectedKeyError",
    "DuplicateTensorError",
    "MissingCoverageError",
    "ShapeMismatchError",
    "PrunedBaseError",
    "TensorInfo",
    "SafetensorsHeader",
    "parse_safetensors_header",
    "read_safetensors_header",
    "load_tensor",
    "load_tensors",
    "RavenBaseConfig",
    "ModuleEntry",
    "ModuleLora",
    "RavenLoraManifest",
    "PEFT_PREFIX",
    "BASE_KEY_PREFIX",
    "DEFAULT_ALPHA",
    "EXPECTED_CATEGORY_COUNTS",
    "EXPECTED_MODULE_COUNT",
    "EXPECTED_TENSOR_COUNT",
    "parse_peft_key",
    "build_manifest",
    "manifest_from_file",
    "resolve_alpha",
    "resolve_dit_root",
    "is_model_patcher",
    "assert_patcher_attachable",
    "lora_aware_patcher_init",
    "register_lora_aware_patcher_factory",
    "has_lora_aware_patcher_factory",
    "assert_base_not_pruned",
    "check_base_modules",
    "load_lora_weights",
    "attach_raven_lora",
]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
class RavenLoraError(Exception):
    """Base class for every RAVEN LoRA lane failure."""


class SafetensorsFormatError(RavenLoraError):
    """The file is not a well formed safetensors container."""


class UnexpectedKeyError(RavenLoraError):
    """A tensor key does not map onto a known base module."""


class DuplicateTensorError(RavenLoraError):
    """Two tensors claim the same (module, side) slot."""


class MissingCoverageError(RavenLoraError):
    """A base module expected to be covered has no (complete) LoRA pair."""


class ShapeMismatchError(RavenLoraError):
    """A/B shapes disagree with the base module or with each other."""


class PrunedBaseError(RavenLoraError):
    """The base model is the pruned / adaln-curve form, which M0 refuses."""


# --------------------------------------------------------------------------
# safetensors header / tensor reader (no safetensors dependency, no full read)
# --------------------------------------------------------------------------
_MAX_HEADER_BYTES = 256 * 1024 * 1024

_ST_DTYPES: Dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _ST_DTYPES["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _ST_DTYPES["F8_E5M2"] = torch.float8_e5m2


@dataclass(frozen=True)
class TensorInfo:
    """One entry of a safetensors header."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    begin: int  # byte offset relative to the start of the data section
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return _ST_DTYPES[self.dtype]
        except KeyError:
            raise SafetensorsFormatError(
                "unsupported safetensors dtype {!r} for tensor {!r}".format(self.dtype, self.name)
            ) from None

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: Dict[str, TensorInfo]
    metadata: Dict[str, str]
    data_offset: int  # absolute file offset of the data section

    def total_data_bytes(self) -> int:
        return max((t.end for t in self.tensors.values()), default=0)


def safetensors_header_length(first_eight: bytes) -> int:
    """Number of JSON header bytes announced by the first 8 bytes of the file."""
    if len(first_eight) < 8:
        raise SafetensorsFormatError("file shorter than the 8 byte safetensors length prefix")
    (n,) = struct.unpack("<Q", first_eight[:8])
    if n <= 0 or n > _MAX_HEADER_BYTES:
        raise SafetensorsFormatError("implausible safetensors header length: {}".format(n))
    return int(n)


def parse_safetensors_header(blob: bytes) -> SafetensorsHeader:
    """Parse ``blob`` = first ``8 + header_len`` bytes of a safetensors file."""
    header_len = safetensors_header_length(blob)
    if len(blob) < 8 + header_len:
        raise SafetensorsFormatError(
            "need {} header bytes, got {}".format(8 + header_len, len(blob))
        )
    try:
        raw = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise SafetensorsFormatError("safetensors header is not valid JSON: {}".format(exc)) from exc
    if not isinstance(raw, dict):
        raise SafetensorsFormatError("safetensors header is not a JSON object")

    metadata = raw.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        raise SafetensorsFormatError("__metadata__ is not a JSON object")

    tensors: Dict[str, TensorInfo] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise SafetensorsFormatError("header entry {!r} is not an object".format(name))
        try:
            dtype = str(spec["dtype"])
            shape = tuple(int(x) for x in spec["shape"])
            begin, end = (int(x) for x in spec["data_offsets"])
        except Exception as exc:  # noqa: BLE001
            raise SafetensorsFormatError(
                "header entry {!r} is malformed: {}".format(name, exc)
            ) from exc
        if end < begin:
            raise SafetensorsFormatError("header entry {!r} has inverted offsets".format(name))
        tensors[name] = TensorInfo(name=name, dtype=dtype, shape=shape, begin=begin, end=end)

    return SafetensorsHeader(
        tensors=tensors,
        metadata={str(k): str(v) for k, v in metadata.items()},
        data_offset=8 + header_len,
    )


def read_safetensors_header(path: str) -> SafetensorsHeader:
    """Read only the header of a local safetensors file (no tensor data)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        header_len = safetensors_header_length(prefix)
        return parse_safetensors_header(prefix + fh.read(header_len))


def _tensor_from_buffer(buf: bytearray, info: TensorInfo) -> torch.Tensor:
    dtype = info.torch_dtype
    expected = info.numel() * dtype.itemsize if dtype != torch.bool else info.numel()
    if len(buf) != info.nbytes:
        raise SafetensorsFormatError(
            "short read for {!r}: {} of {} bytes".format(info.name, len(buf), info.nbytes)
        )
    if info.nbytes != expected:
        raise SafetensorsFormatError(
            "tensor {!r} announces {} bytes but shape {} dtype {} needs {}".format(
                info.name, info.nbytes, info.shape, info.dtype, expected
            )
        )
    flat = torch.frombuffer(buf, dtype=dtype)
    return flat.reshape(info.shape)


def load_tensor(path: str, info: TensorInfo, data_offset: int) -> torch.Tensor:
    with open(path, "rb") as fh:
        return _read_one(fh, info, data_offset)


def _read_one(fh, info: TensorInfo, data_offset: int) -> torch.Tensor:
    fh.seek(data_offset + info.begin)
    return _tensor_from_buffer(bytearray(fh.read(info.nbytes)), info)


def load_tensors(
    path: str, infos: Sequence[TensorInfo], data_offset: int
) -> Dict[str, torch.Tensor]:
    """Read the given tensors in file order, one open file handle."""
    out: Dict[str, torch.Tensor] = {}
    with open(path, "rb") as fh:
        for info in sorted(infos, key=lambda i: i.begin):
            out[info.name] = _read_one(fh, info, data_offset)
    return out


# --------------------------------------------------------------------------
# base module inventory (structural, from the Comfy MiniMaxH3Model definition)
# --------------------------------------------------------------------------
PEFT_PREFIX = "base_model.model.dit."
BASE_KEY_PREFIX = "diffusion_model."
DEFAULT_ALPHA = 128.0
DEFAULT_RANK = 128

CATEGORY_CORE = "core"
CATEGORY_ADALN = "adaln"
CATEGORY_TIME = "time"
CATEGORY_BOUNDARY = "boundary"
CATEGORY_ORDER = (CATEGORY_CORE, CATEGORY_ADALN, CATEGORY_TIME, CATEGORY_BOUNDARY)

EXPECTED_CATEGORY_COUNTS: Dict[str, int] = {
    CATEGORY_CORE: 208,
    CATEGORY_ADALN: 51,
    CATEGORY_TIME: 2,
    CATEGORY_BOUNDARY: 5,
}
EXPECTED_MODULE_COUNT = sum(EXPECTED_CATEGORY_COUNTS.values())  # 266
EXPECTED_TENSOR_COUNT = 2 * EXPECTED_MODULE_COUNT  # 532


@dataclass(frozen=True)
class ModuleEntry:
    """One LoRA-eligible ``nn.Linear`` of the base DiT."""

    path: str  # relative to the DiT root, e.g. "blocks.0.attn.qkv_proj"
    category: str
    out_features: int
    in_features: int

    @property
    def base_key(self) -> str:
        return "{}{}.weight".format(BASE_KEY_PREFIX, self.path)

    @property
    def weight_shape(self) -> Tuple[int, int]:
        return (self.out_features, self.in_features)


@dataclass(frozen=True)
class RavenBaseConfig:
    """Shape/topology of the official full non-pruned MiniMax-H3 DiT.

    Defaults mirror ``comfy.ldm.minimax.model.MiniMaxH3Model.__init__``.
    """

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        p = self.patch_size
        return self.latents_dim * p[0] * p[1] * p[2]

    def _attn_mlp(self, prefix: str) -> List[ModuleEntry]:
        h, inner, ffn = self.hidden_size, self.inner_dim, self.ffn_hidden_size
        return [
            ModuleEntry(prefix + ".attn.qkv_proj", CATEGORY_CORE, inner * 3, h),
            ModuleEntry(prefix + ".attn.out_proj", CATEGORY_CORE, h, inner),
            ModuleEntry(prefix + ".mlp.fc1", CATEGORY_CORE, ffn * 2, h),
            ModuleEntry(prefix + ".mlp.fc2", CATEGORY_CORE, h, ffn),
        ]

    def modules(self) -> "Dict[str, ModuleEntry]":
        """Ordered ``path -> ModuleEntry`` inventory of every LoRA target."""
        h = self.hidden_size
        entries: List[ModuleEntry] = [
            ModuleEntry("video_patch_proj", CATEGORY_BOUNDARY, h, self.video_patch_dim),
            ModuleEntry("audio_patch_proj", CATEGORY_BOUNDARY, h, self.audio_latents_dim),
            ModuleEntry("condition_proj", CATEGORY_BOUNDARY, h, self.text_dim),
            ModuleEntry("time_embedder.proj_in", CATEGORY_TIME, self.time_embed_hidden_size, self.timestep_input_dim),
            ModuleEntry("time_embedder.proj_out", CATEGORY_TIME, self.time_embed_dim, self.time_embed_hidden_size),
        ]
        for i in range(self.token_refiner_num_layers):
            entries.extend(self._attn_mlp("token_refiner.blocks.{}".format(i)))
        for i in range(self.num_layers):
            prefix = "blocks.{}".format(i)
            entries.extend(self._attn_mlp(prefix))
            entries.append(
                ModuleEntry(prefix + ".adaln_proj.linear", CATEGORY_ADALN, 6 * h * 3, self.time_embed_dim)
            )
        entries.append(
            ModuleEntry("final_layer.adaln_proj.linear", CATEGORY_ADALN, 2 * h * 1, self.time_embed_dim)
        )
        entries.append(ModuleEntry("final_layer.video_out", CATEGORY_BOUNDARY, self.video_patch_dim, h))
        entries.append(ModuleEntry("final_layer.audio_out", CATEGORY_BOUNDARY, self.audio_latents_dim, h))

        out: Dict[str, ModuleEntry] = {}
        for e in entries:
            if e.path in out:
                raise RavenLoraError("duplicated inventory path {!r}".format(e.path))
            out[e.path] = e
        return out


def category_counts(entries: Iterable[ModuleEntry]) -> Dict[str, int]:
    counts = {c: 0 for c in CATEGORY_ORDER}
    for e in entries:
        counts[e.category] = counts.get(e.category, 0) + 1
    return counts


# --------------------------------------------------------------------------
# PEFT key parsing / mapping
# --------------------------------------------------------------------------
# base_model.model.dit.<path>.lora_A.weight
# base_model.model.dit.<path>.lora_A.default.weight   (named adapter)
_LORA_KEY_RE = re.compile(
    r"^(?P<path>.+?)\.lora_(?P<side>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$"
)

def parse_peft_key(key: str, prefix: str = PEFT_PREFIX) -> Tuple[str, str, Optional[str]]:
    """``base_model.model.dit.<path>.lora_A[.<adapter>].weight`` -> (path, "A", adapter).

    Raises :class:`UnexpectedKeyError` for anything else. No renaming, no
    permutation: the remaining path is used verbatim as the base module path.
    """
    if not key.startswith(prefix):
        raise UnexpectedKeyError(
            "key {!r} does not start with the expected PEFT prefix {!r}".format(key, prefix)
        )
    rest = key[len(prefix) :]
    m = _LORA_KEY_RE.match(rest)
    if m is None:
        raise UnexpectedKeyError(
            "key {!r} is not a PEFT lora_A/lora_B weight (unsupported entry, e.g. "
            "lora_embedding/lora_magnitude/bias)".format(key)
        )
    return m.group("path"), m.group("side"), m.group("adapter")


@dataclass(frozen=True)
class ModuleLora:
    """One mapped module: base key + the A/B tensor descriptors."""

    path: str
    base_key: str
    category: str
    rank: int
    a: TensorInfo
    b: TensorInfo
    entry: ModuleEntry

    @property
    def a_shape(self) -> Tuple[int, ...]:
        return self.a.shape

    @property
    def b_shape(self) -> Tuple[int, ...]:
        return self.b.shape

    def numel(self) -> int:
        return self.a.numel() + self.b.numel()


@dataclass
class RavenLoraManifest:
    """Validated mapping of a published PEFT file onto ``diffusion_model.*``."""

    modules: "Dict[str, ModuleLora]"
    metadata: Dict[str, str]
    alpha: float
    rank: int
    counts: Dict[str, int]
    adapter_names: Tuple[str, ...] = ()
    source: str = ""
    data_offset: int = 0

    @property
    def module_count(self) -> int:
        return len(self.modules)

    @property
    def tensor_count(self) -> int:
        return 2 * len(self.modules)

    def base_keys(self) -> List[str]:
        return [m.base_key for m in self.modules.values()]

    def parameter_numel(self) -> int:
        return sum(m.numel() for m in self.modules.values())

    def summary(self) -> str:
        parts = ["modules={} tensors={}".format(self.module_count, self.tensor_count)]
        parts.append(" ".join("{}={}".format(c, self.counts.get(c, 0)) for c in CATEGORY_ORDER))
        parts.append("rank={} alpha={}".format(self.rank, self.alpha))
        return " | ".join(parts)


def resolve_alpha(
    metadata: Mapping[str, str],
    override: Optional[float] = None,
    adapter_config: Optional[Mapping[str, object]] = None,
    default: float = DEFAULT_ALPHA,
) -> float:
    """Precedence: explicit argument > safetensors ``__metadata__`` > adapter_config > 128."""
    if override is not None:
        return float(override)
    for key in ("lora_alpha", "alpha", "raven_lora_alpha"):
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                raise RavenLoraError(
                    "safetensors metadata {}={!r} is not a number".format(key, metadata[key])
                ) from None
    if adapter_config:
        for key in ("lora_alpha", "alpha"):
            if key in adapter_config:
                return float(adapter_config[key])  # type: ignore[arg-type]
    return float(default)


def _read_adapter_config(lora_path: str) -> Optional[Dict[str, object]]:
    cfg = os.path.join(os.path.dirname(os.path.abspath(lora_path)), "adapter_config.json")
    if not os.path.isfile(cfg):
        return None
    with open(cfg, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _fmt_sample(items: Iterable[str], limit: int = 8) -> str:
    items = list(items)
    head = ", ".join(repr(x) for x in items[:limit])
    if len(items) > limit:
        head += ", ... (+{} more)".format(len(items) - limit)
    return head


def build_manifest(
    header: SafetensorsHeader,
    config: Optional[RavenBaseConfig] = None,
    *,
    alpha: Optional[float] = None,
    prefix: str = PEFT_PREFIX,
    allowed_dtypes: Sequence[str] = ("F32",),
    require_full_coverage: bool = True,
    expected_counts: Optional[Mapping[str, int]] = EXPECTED_CATEGORY_COUNTS,
    adapter_config: Optional[Mapping[str, object]] = None,
    source: str = "",
) -> RavenLoraManifest:
    """Strictly map a parsed header onto the base module inventory.

    Fails loud on: unexpected keys, duplicate (module, side) tensors, missing
    A/B halves, unknown modules, A/B shapes that disagree with the base module
    or with each other, inconsistent rank, unexpected dtype, and (optionally)
    incomplete coverage / wrong per-category counts.
    """
    cfg = config or RavenBaseConfig()
    inventory = cfg.modules()

    pairs: Dict[str, Dict[str, TensorInfo]] = {}
    adapters: List[str] = []
    unexpected: List[str] = []
    unknown_modules: List[str] = []
    bad_dtype: List[str] = []

    for name in sorted(header.tensors):
        info = header.tensors[name]
        try:
            path, side, adapter = parse_peft_key(name, prefix=prefix)
        except UnexpectedKeyError:
            unexpected.append(name)
            continue
        if adapter is not None and adapter not in adapters:
            adapters.append(adapter)
        if path not in inventory:
            unknown_modules.append(name)
            continue
        if allowed_dtypes and info.dtype not in allowed_dtypes:
            bad_dtype.append("{} ({})".format(name, info.dtype))
            continue
        slot = pairs.setdefault(path, {})
        if side in slot:
            raise DuplicateTensorError(
                "module {!r} has two lora_{} tensors: {!r} and {!r}".format(
                    path, side, slot[side].name, name
                )
            )
        slot[side] = info

    if unexpected:
        raise UnexpectedKeyError(
            "{} tensor key(s) are not PEFT lora_A/lora_B weights under prefix {!r}: {}".format(
                len(unexpected), prefix, _fmt_sample(unexpected)
            )
        )
    if unknown_modules:
        raise UnexpectedKeyError(
            "{} tensor key(s) target modules that do not exist in the official full "
            "non-pruned MiniMax-H3 DiT: {}".format(len(unknown_modules), _fmt_sample(unknown_modules))
        )
    if bad_dtype:
        raise RavenLoraError(
            "expected dtype(s) {} for LoRA tensors, got: {}".format(
                list(allowed_dtypes), _fmt_sample(bad_dtype)
            )
        )

    incomplete = sorted(p for p, s in pairs.items() if len(s) != 2)
    if incomplete:
        raise MissingCoverageError(
            "{} module(s) have only one of lora_A/lora_B: {}".format(
                len(incomplete), _fmt_sample(incomplete)
            )
        )

    modules: Dict[str, ModuleLora] = {}
    ranks: Dict[int, List[str]] = {}
    for path in inventory:  # inventory order, deterministic
        if path not in pairs:
            continue
        entry = inventory[path]
        a, b = pairs[path]["A"], pairs[path]["B"]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ShapeMismatchError(
                "module {!r}: expected 2D lora_A/lora_B, got {} and {}".format(path, a.shape, b.shape)
            )
        rank = int(a.shape[0])
        if int(b.shape[1]) != rank:
            raise ShapeMismatchError(
                "module {!r}: lora_A rank {} != lora_B rank {} (A={}, B={})".format(
                    path, rank, b.shape[1], a.shape, b.shape
                )
            )
        if int(a.shape[1]) != entry.in_features or int(b.shape[0]) != entry.out_features:
            raise ShapeMismatchError(
                "module {!r}: base weight is {} but lora_A={} lora_B={} "
                "(expected A=[r, {}] and B=[{}, r]); refusing any QKV re-interleave or "
                "fc1 half swap".format(
                    path, entry.weight_shape, a.shape, b.shape, entry.in_features, entry.out_features
                )
            )
        ranks.setdefault(rank, []).append(path)
        modules[path] = ModuleLora(
            path=path,
            base_key=entry.base_key,
            category=entry.category,
            rank=rank,
            a=a,
            b=b,
            entry=entry,
        )

    if not modules:
        raise MissingCoverageError("no LoRA module survived mapping (empty file?)")
    if len(ranks) != 1:
        raise ShapeMismatchError(
            "inconsistent LoRA rank across modules: {}".format(
                {r: _fmt_sample(v, 3) for r, v in sorted(ranks.items())}
            )
        )
    rank = next(iter(ranks))

    if require_full_coverage:
        missing = [p for p in inventory if p not in modules]
        if missing:
            raise MissingCoverageError(
                "{} base module(s) are not covered by the LoRA: {}".format(
                    len(missing), _fmt_sample(missing)
                )
            )

    counts = category_counts(m.entry for m in modules.values())
    if expected_counts is not None:
        mismatch = {c: (counts.get(c, 0), n) for c, n in expected_counts.items() if counts.get(c, 0) != n}
        if mismatch:
            raise MissingCoverageError(
                "category counts {} (got, expected) do not match the published "
                "208/51/2/5 layout; full counts: {}".format(mismatch, counts)
            )
        total = sum(expected_counts.values())
        if len(modules) != total:
            raise MissingCoverageError(
                "expected {} modules / {} tensors, mapped {} / {}".format(
                    total, 2 * total, len(modules), 2 * len(modules)
                )
            )

    return RavenLoraManifest(
        modules=modules,
        metadata=dict(header.metadata),
        alpha=resolve_alpha(header.metadata, alpha, adapter_config),
        rank=rank,
        counts=counts,
        adapter_names=tuple(adapters),
        source=source,
        data_offset=header.data_offset,
    )


def manifest_from_file(
    path: str,
    config: Optional[RavenBaseConfig] = None,
    *,
    alpha: Optional[float] = None,
    use_adapter_config: bool = True,
    **kwargs,
) -> RavenLoraManifest:
    """Header-only mapping of a local safetensors file (no tensor data read)."""
    header = read_safetensors_header(path)
    adapter_config = _read_adapter_config(path) if use_adapter_config else None
    return build_manifest(
        header, config, alpha=alpha, adapter_config=adapter_config, source=path, **kwargs
    )


def load_lora_weights(
    path: str, manifest: RavenLoraManifest, dtype: torch.dtype = torch.float32
) -> "Dict[str, Tuple[torch.Tensor, torch.Tensor]]":
    """Read the A/B tensors of ``manifest`` from ``path``.

    Returns ``module path -> (A, B)``; ``B @ A`` is never formed.
    """
    infos: List[TensorInfo] = []
    for m in manifest.modules.values():
        infos.extend((m.a, m.b))
    raw = load_tensors(path, infos, manifest.data_offset)
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for p, m in manifest.modules.items():
        a = raw[m.a.name]
        b = raw[m.b.name]
        out[p] = (a.to(dtype).contiguous(), b.to(dtype).contiguous())
    return out


# --------------------------------------------------------------------------
# base model side
# --------------------------------------------------------------------------
def is_model_patcher(target) -> bool:
    """Duck-typed ``comfy.model_patcher.ModelPatcher`` (incl. the dynamic one)."""
    return (
        hasattr(target, "model")
        and hasattr(target, "load_device")
        and hasattr(target, "offload_device")
        and callable(getattr(target, "model_size", None))
    )


def resolve_dit_root(target) -> Tuple[torch.nn.Module, str]:
    """Accept a ModelPatcher / BaseModel / raw DiT and return (root, key prefix).

    The key prefix is always ``diffusion_model.`` so that reported base keys are
    exactly the ones the official generic-format LoRA loader uses.
    """
    inner = getattr(target, "model", None)
    if inner is not None and hasattr(inner, "diffusion_model"):
        return inner.diffusion_model, BASE_KEY_PREFIX
    if hasattr(target, "diffusion_model"):
        return target.diffusion_model, BASE_KEY_PREFIX
    if isinstance(target, torch.nn.Module):
        return target, BASE_KEY_PREFIX
    raise RavenLoraError("cannot resolve a DiT root from {!r}".format(type(target)))


# --------------------------------------------------------------------------
# ModelPatcher safety: attach must happen before the patcher measures or loads
# --------------------------------------------------------------------------
def assert_patcher_attachable(patcher, *, allow_cached_patcher_init: bool = False) -> None:
    """Refuse to attach to a patcher that has already measured or loaded itself.

    ``ModelPatcher.model_size()`` memoises into ``self.size``; ``load()`` /
    ``partially_load()`` record ``model_loaded_weight_memory``. Attaching after
    either would leave the LoRA bytes unaccounted (dynamic VRAM would then
    over-commit) or leave A/B behind on the offload device with no owner.
    Silently resetting ``patcher.size`` is *not* acceptable - the caller must
    attach earlier instead (preferably to the raw ``BaseModel``/DiT, before the
    patcher is constructed at all).
    """
    size = getattr(patcher, "size", 0) or 0
    if size > 0:
        raise runtime_linear.RavenAttachError(
            "ModelPatcher.model_size() was already cached ({} bytes): the RAVEN LoRA "
            "must be attached before the patcher is constructed / before its first "
            "model_size() call, otherwise the A/B parameters stay invisible to VRAM "
            "accounting. Attach to the BaseModel/DiT first, then build the patcher "
            "(clearing patcher.size here would silently corrupt the memory ledger).".format(size)
        )
    model = getattr(patcher, "model", None)
    loaded = int(getattr(model, "model_loaded_weight_memory", 0) or 0)
    try:
        loaded = max(loaded, int(patcher.loaded_size() or 0))
    except Exception:  # noqa: BLE001 - dynamic patchers may need a live vbar
        pass
    if loaded > 0 or bool(getattr(model, "model_lowvram", False)):
        raise runtime_linear.RavenAttachError(
            "model is already (partially) loaded: {} bytes resident, lowvram={}. "
            "Attach the RAVEN LoRA before the first load; a post-load attach leaves "
            "A/B unloaded and unaccounted.".format(
                loaded, bool(getattr(model, "model_lowvram", False))
            )
        )
    if not allow_cached_patcher_init and getattr(patcher, "cached_patcher_init", None) is not None:
        raise runtime_linear.RavenAttachError(
            "patcher.cached_patcher_init is set: clone(disable_dynamic=True) and "
            "deepclone_multigpu() would rebuild the model from that factory and the "
            "rebuilt model would silently have no RAVEN LoRA. M0 refuses this; the M1 "
            "loader must register a LoRA-aware factory (see "
            "register_lora_aware_patcher_factory) or attach before the factory is set. "
            "Do not simply clear cached_patcher_init - multigpu/non-dynamic delegates "
            "depend on it."
        )


def lora_aware_patcher_init(cached_patcher_init, reattach):
    """Wrap a ``cached_patcher_init`` tuple so rebuilt patchers get the LoRA back.

    ``cached_patcher_init`` is ``(factory, args)`` or ``(factory, args, index)``;
    comfy calls ``factory(*args)`` (optionally with ``disable_dynamic=True``) and
    picks ``result[index]``. The wrapper re-attaches to whatever patcher the
    factory produced, so ``deepclone_multigpu`` / non-dynamic delegates keep the
    RAVEN residuals.
    """
    if cached_patcher_init is None:
        raise RavenLoraError("cached_patcher_init is None: nothing to wrap")
    factory = cached_patcher_init[0]
    index = cached_patcher_init[2] if len(cached_patcher_init) > 2 else None

    def wrapped(*args, **kwargs):
        produced = factory(*args, **kwargs)
        target = produced
        if index is not None and isinstance(produced, (list, tuple)):
            target = produced[index]
        reattach(target)
        return produced

    wrapped.raven_lora_wrapped_factory = factory  # type: ignore[attr-defined]
    return (wrapped,) + tuple(cached_patcher_init[1:])


def register_lora_aware_patcher_factory(patcher, reattach) -> None:
    """Install :func:`lora_aware_patcher_init` on ``patcher`` (M1 loader hook)."""
    patcher.cached_patcher_init = lora_aware_patcher_init(
        getattr(patcher, "cached_patcher_init", None), reattach
    )


def has_lora_aware_patcher_factory(patcher) -> bool:
    init = getattr(patcher, "cached_patcher_init", None)
    return init is not None and hasattr(init[0], "raven_lora_wrapped_factory")


def assert_base_not_pruned(root: torch.nn.Module) -> None:
    """Refuse the pruned / adaln-curve checkpoint form.

    Curve-form checkpoints replace ``time_embedder`` with a shared
    ``adaln_t_table`` basis buffer and change the adaln weights, so the
    published 266-module adapter cannot be applied.
    """
    buffers = {n for n, _ in root.named_buffers()}
    if "adaln_t_table" in buffers or hasattr(root, "adaln_t_table"):
        raise PrunedBaseError(
            "base model exposes 'adaln_t_table': this is the pruned / adaln-curve "
            "checkpoint form, which the M0 LoRA lane refuses (needs the full "
            "non-pruned model with a time_embedder)"
        )
    if not hasattr(root, "time_embedder"):
        raise PrunedBaseError(
            "base model has no 'time_embedder': not the official full non-pruned "
            "MiniMax-H3 DiT expected by the published adapter"
        )


def check_base_modules(
    root: torch.nn.Module, manifest: RavenLoraManifest
) -> "Dict[str, torch.nn.Module]":
    """Resolve every mapped path on the live model and verify weight shapes."""
    resolved: Dict[str, torch.nn.Module] = {}
    missing: List[str] = []
    bad: List[str] = []
    for path, m in manifest.modules.items():
        try:
            mod = root.get_submodule(path)
        except AttributeError:
            missing.append(path)
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or tuple(weight.shape) != m.entry.weight_shape:
            bad.append(
                "{} base={} expected={}".format(
                    path, None if weight is None else tuple(weight.shape), m.entry.weight_shape
                )
            )
            continue
        resolved[path] = mod
    if missing:
        raise MissingCoverageError(
            "{} mapped module(s) do not exist on the base model: {}".format(
                len(missing), _fmt_sample(missing)
            )
        )
    if bad:
        raise ShapeMismatchError(
            "{} base module(s) have unexpected weight shapes: {}".format(len(bad), _fmt_sample(bad))
        )
    return resolved


# --------------------------------------------------------------------------
# public attach entry point
# --------------------------------------------------------------------------
def attach_raven_lora(
    target,
    lora: "str | RavenLoraManifest",
    *,
    strength: float = 1.0,
    alpha: Optional[float] = None,
    weights: "Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]" = None,
    config: Optional[RavenBaseConfig] = None,
    name: Optional[str] = None,
    manifest_kwargs: Optional[Mapping[str, object]] = None,
    row_chunk: Optional[int] = None,
    allow_cached_patcher_init: bool = False,
) -> runtime_linear.RavenLoraAttachment:
    """Attach a published RAVEN PEFT LoRA activation-side onto a Comfy model.

    ``target`` should preferably be the raw ``BaseModel`` / ``MiniMaxH3Model``
    **before** any ``ModelPatcher`` is built around it. A ``ModelPatcher`` is
    accepted too, but only while it is still pristine: see
    :func:`assert_patcher_attachable` (cached ``size``, resident weights or a
    ``cached_patcher_init`` factory are all hard errors).

    ``lora`` is a path to the safetensors file or an already built
    :class:`RavenLoraManifest` (then ``weights`` must be supplied, e.g. for
    synthetic tests). An ``alpha`` override never mutates a shared manifest: a
    copy is made instead.

    ``strength`` is mandatory-with-default: it is always applied as
    ``scale = strength * alpha / rank``. ``row_chunk`` overrides the automatic
    residual row chunking (default: derived from an FP32 temp budget).

    Neither the base ``.weight``/``.bias`` keys nor the module names change, so
    the official ``LoraLoaderModelOnly`` weight-patch path still resolves the
    exact same ``diffusion_model.*.weight`` keys afterwards.
    """
    mkwargs = dict(manifest_kwargs or {})
    if isinstance(lora, str):
        manifest = manifest_from_file(lora, config, alpha=alpha, **mkwargs)  # type: ignore[arg-type]
    else:
        manifest = lora
        if alpha is not None and float(alpha) != float(manifest.alpha):
            # never mutate a manifest the caller may share between attaches
            manifest = dataclasses.replace(manifest, alpha=float(alpha))

    if is_model_patcher(target):
        assert_patcher_attachable(
            target,
            allow_cached_patcher_init=(
                allow_cached_patcher_init or has_lora_aware_patcher_factory(target)
            ),
        )

    root, _prefix = resolve_dit_root(target)
    assert_base_not_pruned(root)
    modules = check_base_modules(root, manifest)

    if weights is None:
        if not manifest.source:
            raise RavenLoraError("no weights given and the manifest has no source file")
        weights = load_lora_weights(manifest.source, manifest)

    missing = [p for p in manifest.modules if p not in weights]
    if missing:
        raise MissingCoverageError(
            "{} module(s) have no A/B tensors: {}".format(len(missing), _fmt_sample(missing))
        )
    extra = [p for p in weights if p not in manifest.modules]
    if extra:
        raise UnexpectedKeyError(
            "{} weight entr(ies) are not in the manifest: {}".format(len(extra), _fmt_sample(extra))
        )

    plan: List[runtime_linear.ResidualPlan] = []
    for path, m in manifest.modules.items():
        a, b = weights[path]
        plan.append(
            runtime_linear.ResidualPlan(
                path=path,
                module=modules[path],
                a=a,
                b=b,
                alpha=manifest.alpha,
                rank=m.rank,
                strength=strength,
                base_key=m.base_key,
            )
        )
    return runtime_linear.attach_residuals(
        plan,
        name=name or (os.path.basename(manifest.source) if manifest.source else "raven_lora"),
        row_chunk=row_chunk,
    )
