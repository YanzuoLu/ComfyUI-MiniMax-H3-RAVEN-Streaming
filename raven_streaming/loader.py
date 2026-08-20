"""RAVEN M1 loader lane: official H3 DiT + mandatory RAVEN PEFT -> standard MODEL.

What this is
------------
A faithful replication of ``comfy.sd.load_diffusion_model_state_dict`` (pinned
at ComfyUI ``c67885b``) with **one deliberate reordering**:

    upstream:  build BaseModel -> build ModelPatcher -> load base weights
    here:      build BaseModel -> load base weights -> attach RAVEN LoRA
                                                    -> build ModelPatcher

The reordering is the whole point. ``ModelPatcher.model_size()`` memoises into
``self.size`` and ``load()``/``partially_load()`` record
``model_loaded_weight_memory``; both happen off the *live* module tree. If the
FP32 activation residual (``raven_lora_A_* / raven_lora_B_*``, registered as
direct parameters of the base ``Linear`` leaves, see
:mod:`raven_streaming.runtime_linear`) is attached after the patcher has
measured or loaded itself, those bytes are invisible to Comfy's memory ledger
and the A/B parameters are never moved by partial CPU offload. So the LoRA is
attached to the raw ``BaseModel``/DiT *before* the patcher exists, and the
patcher's very first ``model_size()`` already counts it. Nothing here clears
``patcher.size`` or otherwise rewrites upstream bookkeeping.

Everything else follows upstream exactly: ``load_torch_file`` with metadata,
old-quant conversion, unet prefix detection/strip, ``model_config_from_unet``,
dtype / manual-cast / operations selection, ``model_config.get_model``,
``load_model_weights(assign=<patcher is dynamic>)``, load/offload device
selection, patcher construction and a ``cached_patcher_init`` factory. The
result is a **standard** ``ModelPatcher``: no custom MODEL type, so stock
partial CPU offload, cached reloads and a downstream official
``LoraLoaderModelOnly`` all keep working - the base ``diffusion_model.*.weight``
keys are untouched.

Which patcher (v0.1 release contract)
-------------------------------------
The v0.1 release baseline is **stock Comfy ``ModelPatcher`` partial CPU
offload**: that is the path the full non-pruned BF16 DiT is required to run on,
and the only one this package makes any claim about. DynamicVRAM / aimdo
(``ModelPatcherDynamic``, which ``CoreModelPatcher`` is rebound to by ``main.py``
when comfy-aimdo initialises) is **optional and unverified here** - it has never
been exercised with this loader.

So :class:`RavenLoaderSpec` carries ``force_static_patcher``, defaulting to
``True``, which selects the same class upstream's ``disable_dynamic=True`` does.
It is part of the spec, not of the call, which means the
``cached_patcher_init`` factory reproduces it: a rebuild triggered by
``deepclone_multigpu`` (plain ``factory(*args)``) stays static, and upstream's
explicit ``factory(*args, disable_dynamic=True)`` can only be *more* strict,
never less. A programmatic caller may set ``force_static_patcher=False`` to try
the dynamic path; nothing about it is validated, and the probe reports it as
optional.

M1 vs M2
--------
M1 builds the official bidirectional ``comfy.ldm.minimax.model.MiniMaxH3Model``.
M2 needs a causal/streaming variant, so :class:`RavenLoaderSpec` carries an
optional ``unet_model_cls`` (injected into ``BaseModel(unet_model=...)`` through
an MRO shim that still runs the official ``model_base.MiniMaxH3.__init__``) and
an optional ``base_model_factory`` for a full BaseModel replacement. Neither
requires a change to the loader's main chain, and the ``cached_patcher_init``
factory carries the choice along, so a rebuilt patcher is the same model with
the same adapter.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch

from raven_streaming import compat
from raven_streaming import lora as rlora
from raven_streaming import runtime_linear

__all__ = [
    "RavenLoaderError",
    "PrunedCheckpointError",
    "UnsupportedCheckpointError",
    "DIFFUSION_MODEL_FOLDER",
    "LORA_FOLDER",
    "WEIGHT_DTYPE_CHOICES",
    "PRUNED_STATE_DICT_KEYS",
    "REQUIRED_STATE_DICT_KEYS",
    "LoaderInput",
    "RAVEN_LOADER_INPUTS",
    "RAVEN_LOADER_RETURN_TYPES",
    "loader_input_schema",
    "RavenLoaderSpec",
    "RavenLoadReport",
    "resolve_model_path",
    "resolve_diffusion_model_path",
    "resolve_lora_path",
    "model_options_for_weight_dtype",
    "assert_full_nonpruned_state_dict",
    "assert_full_nonpruned_unet_config",
    "raven_config_from_unet_config",
    "expected_category_counts",
    "make_unet_injected_model_class",
    "patcher_class_for",
    "patcher_would_be_dynamic",
    "build_raven_patcher",
    "load_raven_model",
    "load_raven_model_with_report",
    "load_raven_diffusion_model",
    "rebuild_raven_patcher",
    "get_raven_attachment",
    "get_raven_manifest",
    "official_lora_key_map",
    "official_lora_key_hits",
]

LOG = logging.getLogger(__name__)

DIFFUSION_MODEL_FOLDER = "diffusion_models"
LORA_FOLDER = "loras"

#: Node-facing weight dtype choices. No FP8 entries: RAVEN is trained against
#: the full non-pruned BF16 model, and INT8/FP8 quantised weights take
#: ``comfy.ops.linear_input_act``'s fused path, which bypasses ``__call__`` and
#: would silently skip the residual hook (see runtime_linear).
WEIGHT_DTYPE_CHOICES = ("default", "bf16", "fp32")

#: Keys that identify the pruned / adaln-curve checkpoint form.
PRUNED_STATE_DICT_KEYS = ("adaln_t_table",)
#: Keys the official full non-pruned H3 DiT must expose.
REQUIRED_STATE_DICT_KEYS = (
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "condition_proj.weight",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_out.weight",
    "final_layer.video_out.weight",
    "final_layer.audio_out.weight",
)


class RavenLoaderError(RuntimeError):
    """The RAVEN model could not be built."""


class UnsupportedCheckpointError(RavenLoaderError):
    """The file is not the official MiniMax-H3 diffusion model."""


class PrunedCheckpointError(UnsupportedCheckpointError, rlora.PrunedBaseError):
    """The checkpoint is the pruned / adaln-curve form, which RAVEN refuses."""


# --------------------------------------------------------------------------
# node schema (frozen at M1; the node class itself lands with node registration)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoaderInput:
    """One node input of ``RAVEN Model Loader``."""

    name: str
    kind: str  # "combo" | "float" | "string"
    required: bool = True
    default: Any = None
    folder: Optional[str] = None  # folder_paths key for combo inputs
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    advanced: bool = False
    tooltip: str = ""


RAVEN_LOADER_INPUTS: Tuple[LoaderInput, ...] = (
    LoaderInput(
        name="unet_name",
        kind="combo",
        folder=DIFFUSION_MODEL_FOLDER,
        tooltip="Official full, non-pruned BF16 MiniMax-H3 DiT. Pruned/adaln-curve "
                "checkpoints are rejected.",
    ),
    LoaderInput(
        name="lora_name",
        kind="combo",
        folder=LORA_FOLDER,
        tooltip="Mandatory RAVEN PEFT LoRA (532 tensors / 266 modules, FP32 A/B). "
                "Without it this node cannot produce a model.",
    ),
    LoaderInput(
        name="strength",
        kind="float",
        default=1.0,
        minimum=-10.0,
        maximum=10.0,
        step=0.01,
        tooltip="RAVEN residual strength; applied as strength * alpha / rank.",
    ),
    LoaderInput(
        name="weight_dtype",
        kind="combo",
        choices=WEIGHT_DTYPE_CHOICES,
        default="default",
        advanced=True,
        tooltip="'default' lets comfy.model_management pick (BF16 for H3). 'fp32' "
                "doubles the weights to 132 GB+ for the full non-pruned model, on "
                "top of the FP32 RAVEN residual - expect heavy offload traffic or "
                "OOM unless you know the box can hold it.",
    ),
)

RAVEN_LOADER_RETURN_TYPES: Tuple[str, ...] = ("MODEL",)


def loader_input_schema() -> Dict[str, Any]:
    """Serialisable description of the node inputs (schema test / docs)."""
    return {
        "return_types": list(RAVEN_LOADER_RETURN_TYPES),
        "inputs": [dataclasses.asdict(i) for i in RAVEN_LOADER_INPUTS],
    }


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def resolve_model_path(
    name: str, folder: str, *, folder_paths_module: Any = None, mods: Optional[compat.ComfyModules] = None
) -> str:
    """Resolve a node-selected file name, or accept an absolute/existing path.

    Node inputs come from ``folder_paths`` combos, so the normal path is
    ``folder_paths.get_full_path_or_raise(folder, name)``. Probes and tests hand
    in an absolute path instead, which is returned as-is - that is why the core
    functions never require ``folder_paths`` to be importable.
    """
    if not name:
        raise RavenLoaderError("empty {} path".format(folder))
    expanded = os.path.expanduser(str(name))
    if os.path.isabs(expanded):
        if not os.path.isfile(expanded):
            raise RavenLoaderError("{}: no such file: {}".format(folder, expanded))
        return expanded
    if os.path.isfile(expanded):  # relative but real (probe convenience)
        return os.path.abspath(expanded)

    module = folder_paths_module
    if module is None:
        module = (mods or compat.import_comfy_modules()).get("folder_paths")
    if module is None:
        raise RavenLoaderError(
            "{!r} is not an absolute path and folder_paths is not importable, so the "
            "{!r} folder cannot be searched".format(name, folder)
        )
    return module.get_full_path_or_raise(folder, name)


def resolve_diffusion_model_path(name: str, **kwargs) -> str:
    return resolve_model_path(name, DIFFUSION_MODEL_FOLDER, **kwargs)


def resolve_lora_path(name: str, **kwargs) -> str:
    return resolve_model_path(name, LORA_FOLDER, **kwargs)


def model_options_for_weight_dtype(weight_dtype: str = "default") -> Dict[str, Any]:
    """Node ``weight_dtype`` choice -> ``model_options`` (upstream convention)."""
    if weight_dtype in (None, "", "default"):
        return {}
    if weight_dtype == "bf16":
        return {"dtype": torch.bfloat16}
    if weight_dtype == "fp32":
        return {"dtype": torch.float32}
    raise RavenLoaderError(
        "unsupported weight_dtype {!r}; expected one of {}".format(
            weight_dtype, list(WEIGHT_DTYPE_CHOICES)
        )
    )


# --------------------------------------------------------------------------
# checkpoint guards
# --------------------------------------------------------------------------
def assert_full_nonpruned_state_dict(state_dict: Mapping[str, Any], source: str = "") -> None:
    """Refuse the pruned / adaln-curve checkpoint and non-H3 files.

    Runs on the *prefix-stripped* diffusion-model state dict.
    """
    keys = state_dict.keys()
    present = [k for k in PRUNED_STATE_DICT_KEYS if k in keys]
    if present:
        raise PrunedCheckpointError(
            "{}: checkpoint contains {} - this is the pruned / adaln-curve MiniMax-H3 "
            "form (shared time-curve basis instead of a time_embedder). The published "
            "RAVEN adapter is trained against the full non-pruned BF16 model and its "
            "266-module mapping cannot be applied here.".format(source or "checkpoint", present)
        )
    missing = [k for k in REQUIRED_STATE_DICT_KEYS if k not in keys]
    if missing:
        raise UnsupportedCheckpointError(
            "{}: not the official full non-pruned MiniMax-H3 DiT; {} required key(s) "
            "missing: {}".format(source or "checkpoint", len(missing), missing)
        )


def assert_full_nonpruned_unet_config(unet_config: Mapping[str, Any], source: str = "") -> None:
    """Same guard on the detected config (``adaln_curve_grid`` == pruned form)."""
    if unet_config.get("image_model") != "minimax_h3":
        raise UnsupportedCheckpointError(
            "{}: detected image_model={!r}, expected 'minimax_h3'".format(
                source or "checkpoint", unet_config.get("image_model")
            )
        )
    if unet_config.get("adaln_curve_grid") is not None:
        raise PrunedCheckpointError(
            "{}: detected adaln_curve_grid={!r}: pruned / adaln-curve MiniMax-H3 form, "
            "refused by the RAVEN loader".format(source or "checkpoint", unet_config.get("adaln_curve_grid"))
        )
    for key in ("timestep_input_dim", "time_embed_hidden_size", "time_embed_dim"):
        if unet_config.get(key) is None:
            raise PrunedCheckpointError(
                "{}: detected config has no {!r}: the time embedder is missing, which "
                "means a pruned checkpoint".format(source or "checkpoint", key)
            )


def raven_config_from_unet_config(unet_config: Mapping[str, Any]) -> rlora.RavenBaseConfig:
    """Derive the LoRA module inventory from the *detected* checkpoint config.

    ``RavenBaseConfig``'s defaults are the published full-size numbers; using
    the detected config instead means the A/B shape checks are made against the
    model that is actually being loaded, so a dimension mismatch is reported as
    a mismatch rather than as a confusing shape error against a constant.
    ``patch_size`` is not part of the detected config - upstream's H3 detection
    hard-codes the 1x2x2 patch - so the structural default is kept.
    """
    defaults = rlora.RavenBaseConfig()
    fields = (
        "hidden_size", "num_layers", "token_refiner_num_layers", "num_attention_heads",
        "attention_head_dim", "ffn_hidden_size", "latents_dim", "audio_latents_dim",
        "text_dim", "timestep_input_dim", "time_embed_hidden_size", "time_embed_dim",
    )
    values: Dict[str, Any] = {}
    for name in fields:
        value = unet_config.get(name, None)
        values[name] = getattr(defaults, name) if value is None else int(value)
    return dataclasses.replace(defaults, **values)


def expected_category_counts(config: rlora.RavenBaseConfig) -> Dict[str, int]:
    """Per-category module counts the adapter must cover for ``config``.

    For the published full-size model this is exactly the 208/51/2/5 layout the
    M0 lane pins. For any other (e.g. a structurally identical toy model used by
    a probe dry-run) it is that model's own inventory, so coverage still has to
    be complete - nothing is skipped either way.
    """
    if config == rlora.RavenBaseConfig():
        return dict(rlora.EXPECTED_CATEGORY_COUNTS)
    return rlora.category_counts(config.modules().values())


# --------------------------------------------------------------------------
# M2 injection point: BaseModel(unet_model=...)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def make_unet_injected_model_class(
    base_model_cls: type, root_cls: type, unet_model_cls: type
) -> type:
    """Subclass of ``base_model_cls`` that builds ``unet_model_cls`` as the DiT.

    ``comfy.model_base.MiniMaxH3.__init__`` hard-codes
    ``unet_model=comfy.ldm.minimax.model.MiniMaxH3Model`` in its ``super()``
    call. Rather than skipping that ``__init__`` (which would silently drop any
    future H3-specific setup), a shim class is spliced into the MRO *between*
    ``MiniMaxH3`` and ``BaseModel``: the official ``__init__`` runs unchanged
    and its ``super().__init__(..., unet_model=...)`` lands on the shim, which
    swaps the class and forwards to ``BaseModel.__init__``.

    ``root_cls`` must be ``comfy.model_base.BaseModel`` (the class that owns the
    ``unet_model`` parameter); ``base_model_cls`` the H3 BaseModel subclass.
    """
    if not issubclass(base_model_cls, root_cls):
        raise RavenLoaderError(
            "{} is not a subclass of {}: cannot splice the unet_model override into "
            "the MRO".format(base_model_cls.__name__, root_cls.__name__)
        )

    class _RavenUnetOverride(root_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["unet_model"] = unet_model_cls
            super().__init__(*args, **kwargs)

    injected = type(
        "Raven{}".format(base_model_cls.__name__),
        (base_model_cls, _RavenUnetOverride),
        {
            "raven_unet_model_cls": unet_model_cls,
            "__doc__": "{} with diffusion_model = {}".format(
                base_model_cls.__name__, getattr(unet_model_cls, "__name__", unet_model_cls)
            ),
        },
    )
    return injected


# --------------------------------------------------------------------------
# loader spec / report
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RavenLoaderSpec:
    """Everything needed to build (or *re*build) the exact same RAVEN model.

    This is what the ``cached_patcher_init`` factory closes over, so a rebuilt
    patcher reads the same base file, the same LoRA file, the same strength and
    the same ``unet_model_cls`` - the adapter can never be silently dropped.
    """

    unet_path: str
    lora_path: str
    strength: float = 1.0
    alpha: Optional[float] = None
    row_chunk: Optional[int] = None
    model_options: Mapping[str, Any] = field(default_factory=dict)
    unet_model_cls: Optional[type] = None
    base_model_factory: Optional[Callable[..., Any]] = None
    lora_config: Optional[rlora.RavenBaseConfig] = None
    manifest_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Build the stock ``ModelPatcher`` (upstream's ``disable_dynamic=True``
    #: class choice). ``True`` is the v0.1 release contract: stock partial CPU
    #: offload is the only supported path. Setting it to ``False`` opts into the
    #: optional, unverified DynamicVRAM/aimdo patcher. Because it lives on the
    #: spec, every rebuild through ``cached_patcher_init`` keeps the choice.
    force_static_patcher: bool = True

    def resolved(self, *, folder_paths_module: Any = None, mods: Optional[compat.ComfyModules] = None) -> "RavenLoaderSpec":
        """Copy with both paths turned into absolute file paths."""
        return dataclasses.replace(
            self,
            unet_path=resolve_diffusion_model_path(
                self.unet_path, folder_paths_module=folder_paths_module, mods=mods
            ),
            lora_path=resolve_lora_path(
                self.lora_path, folder_paths_module=folder_paths_module, mods=mods
            ),
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "unet_path": self.unet_path,
            "lora_path": self.lora_path,
            "strength": float(self.strength),
            "alpha": self.alpha,
            "row_chunk": self.row_chunk,
            "model_options": {k: str(v) for k, v in dict(self.model_options).items()},
            "unet_model_cls": getattr(self.unet_model_cls, "__name__", None),
            "base_model_factory": getattr(self.base_model_factory, "__name__", None),
            "force_static_patcher": bool(self.force_static_patcher),
        }


@dataclass
class RavenLoadReport:
    """Everything the M1 probe reports about one build."""

    spec: Optional[RavenLoaderSpec] = None
    patcher_class: str = ""
    model_class: str = ""
    unet_model_class: str = ""
    unet_dtype: str = ""
    manual_cast_dtype: str = ""
    operations: str = ""
    load_device: str = ""
    offload_device: str = ""
    parameters: int = 0
    checkpoint_weight_dtype: str = ""
    base_key_count: int = 0
    left_over_keys: List[str] = field(default_factory=list)
    assign_weights: bool = False
    is_dynamic: bool = False
    force_static_patcher: bool = True
    requested_disable_dynamic: bool = False
    effective_disable_dynamic: bool = True
    model_size: int = 0
    lora_bytes: int = 0
    lora_modules: int = 0
    lora_rank: int = 0
    lora_alpha: float = 0.0
    lora_strength: float = 1.0
    lora_category_counts: Dict[str, int] = field(default_factory=dict)
    official_topology: bool = False
    official_key_hits: Dict[str, int] = field(default_factory=dict)
    build_seconds: float = 0.0
    #: live objects, not serialised
    attachment: Optional[runtime_linear.RavenLoraAttachment] = None
    manifest: Optional[rlora.RavenLoraManifest] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.describe() if self.spec is not None else None,
            "patcher_class": self.patcher_class,
            "model_class": self.model_class,
            "unet_model_class": self.unet_model_class,
            "unet_dtype": self.unet_dtype,
            "manual_cast_dtype": self.manual_cast_dtype,
            "operations": self.operations,
            "load_device": self.load_device,
            "offload_device": self.offload_device,
            "parameters": self.parameters,
            "checkpoint_weight_dtype": self.checkpoint_weight_dtype,
            "base_key_count": self.base_key_count,
            "left_over_keys": list(self.left_over_keys),
            "assign_weights": self.assign_weights,
            "is_dynamic": self.is_dynamic,
            "force_static_patcher": self.force_static_patcher,
            "requested_disable_dynamic": self.requested_disable_dynamic,
            "effective_disable_dynamic": self.effective_disable_dynamic,
            "model_size": self.model_size,
            "lora_bytes": self.lora_bytes,
            "lora_modules": self.lora_modules,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_strength": self.lora_strength,
            "lora_category_counts": dict(self.lora_category_counts),
            "official_topology": self.official_topology,
            "official_key_hits": dict(self.official_key_hits),
            "build_seconds": self.build_seconds,
        }


# --------------------------------------------------------------------------
# patcher class selection (upstream rules, evaluated before construction)
# --------------------------------------------------------------------------
def patcher_class_for(mods: compat.ComfyModules, disable_dynamic: bool = False) -> type:
    """``comfy.sd``'s rule: ``ModelPatcher`` when dynamic is disabled.

    ``CoreModelPatcher`` is whatever ``main.py`` last bound it to -
    ``ModelPatcher`` by default, ``ModelPatcherDynamic`` once comfy-aimdo
    initialises. ``disable_dynamic`` picks the stock class explicitly, which is
    what ``RavenLoaderSpec.force_static_patcher`` asks for.
    """
    module = mods.require("model_patcher")
    return module.ModelPatcher if disable_dynamic else module.CoreModelPatcher


def patcher_would_be_dynamic(patcher_cls: type, load_device: Any, mods: compat.ComfyModules) -> bool:
    """Predict ``patcher.is_dynamic()`` *before* the patcher exists.

    Upstream passes ``assign=model_patcher.is_dynamic()`` to
    ``load_model_weights``; we load weights first, so the answer is needed
    earlier. Two rules, both read off upstream: the class-level ``is_dynamic``
    return value, and ``ModelPatcherDynamic.__new__`` rerouting CPU load devices
    back to the plain ``ModelPatcher``. The prediction is verified against the
    constructed patcher in :func:`build_raven_patcher`, so a change in either
    rule fails loudly instead of loading weights the wrong way.
    """
    is_dynamic = getattr(patcher_cls, "is_dynamic", None)
    if is_dynamic is None:
        return False
    try:
        declared = bool(is_dynamic(None))
    except Exception as exc:  # noqa: BLE001 - an is_dynamic needing a live self is not predictable
        raise RavenLoaderError(
            "{}.is_dynamic() cannot be evaluated on the class, so the value of "
            "load_model_weights(assign=...) cannot be predicted before the patcher "
            "exists: {}".format(getattr(patcher_cls, "__name__", patcher_cls), exc)
        ) from exc
    if not declared:
        return False
    return not mods.require("model_management").is_device_cpu(load_device)


# --------------------------------------------------------------------------
# base model construction
# --------------------------------------------------------------------------
def _instantiate_base_model(
    model_config: Any, state_dict: Mapping[str, Any], spec: RavenLoaderSpec, mods: compat.ComfyModules
):
    model_base = mods.require("model_base")
    official_cls = getattr(model_base, "MiniMaxH3")

    if spec.base_model_factory is not None:
        model = spec.base_model_factory(model_config, state_dict, device=None)
        if model is None:
            raise RavenLoaderError("base_model_factory returned None")
    elif spec.unet_model_cls is not None:
        injected = make_unet_injected_model_class(
            official_cls, model_base.BaseModel, spec.unet_model_cls
        )
        model = injected(model_config, device=None)
    else:
        model = model_config.get_model(state_dict, "")
        if model is None:
            raise RavenLoaderError("model_config.get_model() returned None")

    if not isinstance(model, model_base.BaseModel):
        raise RavenLoaderError(
            "expected a comfy.model_base.BaseModel, got {}".format(type(model).__name__)
        )
    if getattr(model, "diffusion_model", None) is None:
        raise RavenLoaderError(
            "{} has no diffusion_model (disable_unet_model_creation?); the RAVEN "
            "residual needs the live DiT".format(type(model).__name__)
        )
    if spec.unet_model_cls is not None and not isinstance(model.diffusion_model, spec.unet_model_cls):
        raise RavenLoaderError(
            "unet_model_cls={} was requested but diffusion_model is {}".format(
                getattr(spec.unet_model_cls, "__name__", spec.unet_model_cls),
                type(model.diffusion_model).__name__,
            )
        )
    return model


def _move_attachment_to(attachment: runtime_linear.RavenLoraAttachment, device: Any) -> None:
    """Follow the base weights onto a non-CPU offload device.

    The A/B tensors are read from the safetensors file on CPU. Upstream moves
    the whole model to a non-CPU offload device before loading weights; the
    residual parameters have to make the same trip, otherwise the patcher's load
    plan would find them on the wrong device.
    """
    for entry in attachment.entries:
        for pname in (entry.spec.a_param, entry.spec.b_param):
            param = getattr(entry.module, pname)
            if param.device != device:
                param.data = param.data.to(device)


# --------------------------------------------------------------------------
# core build
# --------------------------------------------------------------------------
def build_raven_patcher(
    state_dict: Dict[str, Any],
    spec: RavenLoaderSpec,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    disable_dynamic: bool = False,
    mods: Optional[compat.ComfyModules] = None,
) -> Tuple[Any, RavenLoadReport]:
    """Replicate the official diffusion-model load, LoRA-first.

    ``state_dict`` is consumed the way upstream consumes it (keys are popped by
    ``load_model_weights``). The patcher class is decided by
    ``spec.force_static_patcher OR disable_dynamic``: the spec's release default
    already selects the stock ``ModelPatcher``, and an explicit
    ``disable_dynamic`` from upstream can only agree with it. Returns
    ``(patcher, report)``.
    """
    started = time.perf_counter()
    mods = mods or compat.require_features()
    utils = mods.require("utils")
    detection = mods.require("model_detection")
    mm = mods.require("model_management")

    # v0.1 ships on the stock patcher; an explicit disable_dynamic from upstream
    # can only make the choice stricter, never re-enable the dynamic class
    force_static = bool(getattr(spec, "force_static_patcher", True))
    effective_disable_dynamic = bool(disable_dynamic) or force_static

    report = RavenLoadReport(
        spec=spec,
        lora_strength=float(spec.strength),
        force_static_patcher=force_static,
        requested_disable_dynamic=bool(disable_dynamic),
        effective_disable_dynamic=effective_disable_dynamic,
    )
    model_options = dict(spec.model_options or {})
    dtype = model_options.get("dtype", None)
    custom_operations = model_options.get("custom_operations", None)

    # -- 1. old quant conversion (upstream order: before and after the strip) --
    if custom_operations is None:
        state_dict, metadata = utils.convert_old_quants(state_dict, "", metadata=metadata)

    # -- 2. unet prefix detection / strip -----------------------------------
    diffusion_model_prefix = detection.unet_prefix_from_state_dict(state_dict)
    temp_sd = utils.state_dict_prefix_replace(
        state_dict, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        state_dict = temp_sd
        if custom_operations is None:
            state_dict, metadata = utils.convert_old_quants(state_dict, "", metadata=metadata)

    # -- 3. the RAVEN checkpoint guard, on the stripped keys ----------------
    assert_full_nonpruned_state_dict(state_dict, spec.unet_path)

    parameters = utils.calculate_parameters(state_dict)
    weight_dtype = utils.weight_dtype(state_dict)
    report.parameters = int(parameters)
    report.checkpoint_weight_dtype = str(weight_dtype)
    report.base_key_count = len(state_dict)

    load_device = model_options.get("load_device", mm.get_torch_device())
    offload_device = model_options.get("offload_device", mm.unet_offload_device())
    report.load_device = str(load_device)
    report.offload_device = str(offload_device)

    # -- 4. model config ----------------------------------------------------
    model_config = detection.model_config_from_unet(state_dict, "", metadata=metadata)
    if model_config is None:
        raise UnsupportedCheckpointError(
            "{}: comfy.model_detection could not detect a model config. The RAVEN "
            "loader takes the official MiniMax-H3 diffusion model only (no diffusers "
            "unet/mmdit conversion path).".format(spec.unet_path)
        )
    assert_full_nonpruned_unet_config(model_config.unet_config, spec.unet_path)
    supported = mods.get("supported_models")
    official_config_cls = getattr(supported, "MiniMaxH3", None) if supported else None
    if official_config_cls is not None and not isinstance(model_config, official_config_cls):
        raise UnsupportedCheckpointError(
            "{}: detected model config {} is not comfy.supported_models.MiniMaxH3".format(
                spec.unet_path, type(model_config).__name__
            )
        )
    if getattr(model_config, "quant_config", None) is not None:
        raise UnsupportedCheckpointError(
            "{}: quantised checkpoint (quant_config detected). comfy.ops fuses "
            "quantised linears and bypasses Linear.__call__, so the FP32 activation "
            "residual would silently not run. RAVEN needs the full BF16 model.".format(
                spec.unet_path
            )
        )

    # -- 5. dtype / manual cast / operations --------------------------------
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if dtype is None:
        unet_dtype = mm.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype
    # upstream picks manual_cast from `None` instead of `unet_dtype` when a
    # quant_config is present; quantised checkpoints are rejected above, so this
    # is the one remaining branch.
    manual_cast_dtype = mm.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)
    if custom_operations is not None:
        model_config.custom_operations = custom_operations
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True
    report.unet_dtype = str(unet_dtype)
    report.manual_cast_dtype = str(manual_cast_dtype)

    # -- 6. patcher class + assign, decided before the patcher exists -------
    patcher_cls = patcher_class_for(mods, disable_dynamic=effective_disable_dynamic)
    assign = patcher_would_be_dynamic(patcher_cls, load_device, mods)
    report.assign_weights = bool(assign)

    # -- 7. raw BaseModel ---------------------------------------------------
    model = _instantiate_base_model(model_config, state_dict, spec, mods)
    report.model_class = type(model).__name__
    report.unet_model_class = type(model.diffusion_model).__name__

    if not mm.is_device_cpu(offload_device):
        model.to(offload_device)

    # -- 8. base weights ----------------------------------------------------
    model.load_model_weights(state_dict, "", assign=assign)
    left_over = list(state_dict.keys())
    report.left_over_keys = left_over
    if left_over:
        LOG.info("left over keys in diffusion model: %s", left_over)

    # -- 9. RAVEN residual, still before any patcher exists -----------------
    lora_config = spec.lora_config or raven_config_from_unet_config(model_config.unet_config)
    manifest_kwargs = dict(spec.manifest_kwargs or {})
    manifest_kwargs.setdefault("expected_counts", expected_category_counts(lora_config))
    manifest = rlora.manifest_from_file(
        spec.lora_path,
        lora_config,
        alpha=spec.alpha,
        **manifest_kwargs,
    )
    attachment = rlora.attach_raven_lora(
        model,
        manifest,
        strength=float(spec.strength),
        row_chunk=spec.row_chunk,
        name=os.path.basename(spec.lora_path),
    )
    if not mm.is_device_cpu(offload_device):
        _move_attachment_to(attachment, offload_device)
    model.raven_lora_attachment = attachment
    model.raven_lora_manifest = manifest
    model.raven_loader_spec = spec
    report.attachment = attachment
    report.manifest = manifest
    report.lora_bytes = attachment.parameter_bytes()
    report.lora_modules = len(attachment)
    report.lora_rank = manifest.rank
    report.lora_alpha = float(manifest.alpha)
    report.lora_category_counts = dict(manifest.counts)
    report.official_topology = dict(manifest.counts) == dict(rlora.EXPECTED_CATEGORY_COUNTS)
    if attachment.entries:  # comfy.ops class actually in use, read off a live linear
        linear_cls = type(attachment.entries[0].module)
        report.operations = "{}.{}".format(linear_cls.__module__, linear_cls.__qualname__)

    # -- 10. patcher --------------------------------------------------------
    patcher = patcher_cls(model, load_device=load_device, offload_device=offload_device)
    report.patcher_class = type(patcher).__name__
    report.is_dynamic = bool(patcher.is_dynamic())
    if report.is_dynamic != assign:
        raise RavenLoaderError(
            "predicted is_dynamic={} but {} reports {}: base weights were loaded with "
            "assign={}, which no longer matches upstream's rule. Refusing the model "
            "instead of shipping a mis-loaded one.".format(
                assign, type(patcher).__name__, report.is_dynamic, assign
            )
        )

    # first size measurement of this patcher; it must already see the residual
    size = patcher.model_size()
    report.model_size = int(size)
    if size < report.lora_bytes:
        raise RavenLoaderError(
            "ModelPatcher.model_size()={} is smaller than the attached RAVEN LoRA "
            "({} bytes): the residual parameters are not visible to Comfy's memory "
            "accounting.".format(size, report.lora_bytes)
        )

    # official generic-format LoRA reachability of the very same base keys
    report.official_key_hits = official_lora_key_hits(patcher, manifest, mods=mods)

    # -- 11. LoRA-aware cached factory --------------------------------------
    patcher.cached_patcher_init = (rebuild_raven_patcher, (spec,))

    report.build_seconds = time.perf_counter() - started
    return patcher, report


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def load_raven_model_with_report(
    spec: RavenLoaderSpec,
    *,
    disable_dynamic: bool = False,
    mods: Optional[compat.ComfyModules] = None,
) -> Tuple[Any, RavenLoadReport]:
    """Read both files and build the model; returns ``(patcher, report)``.

    The patcher class comes from ``spec.force_static_patcher`` (stock
    ``ModelPatcher`` by default) OR-ed with ``disable_dynamic``.
    """
    mods = mods or compat.require_features()
    spec = spec.resolved(mods=mods)
    utils = mods.require("utils")
    state_dict, metadata = utils.load_torch_file(spec.unet_path, return_metadata=True)
    return build_raven_patcher(
        state_dict, spec, metadata=metadata, disable_dynamic=disable_dynamic, mods=mods
    )


def load_raven_model(
    spec: RavenLoaderSpec,
    *,
    disable_dynamic: bool = False,
    mods: Optional[compat.ComfyModules] = None,
) -> Any:
    """Build the RAVEN model and return the standard ``MODEL`` patcher."""
    patcher, _report = load_raven_model_with_report(
        spec, disable_dynamic=disable_dynamic, mods=mods
    )
    return patcher


def rebuild_raven_patcher(spec: RavenLoaderSpec, disable_dynamic: bool = False) -> Any:
    """``cached_patcher_init`` factory: same base, same LoRA, same DiT class.

    Comfy calls this as ``factory(*args)`` or ``factory(*args,
    disable_dynamic=True)`` - the first form from ``deepclone_multigpu``, the
    second from the non-dynamic delegate path - and both signatures are
    satisfied here (one positional spec, one optional keyword). It re-runs the
    full loader from the immutable :class:`RavenLoaderSpec`, so a rebuilt
    patcher always carries the adapter *and* the same patcher-class choice:
    ``spec.force_static_patcher`` is honoured even when the caller passes no
    ``disable_dynamic`` at all, so a rebuild can never silently become dynamic.

    The factory is *not* a wrapper around another factory: rebuilt patchers get
    the very same ``(rebuild_raven_patcher, (spec,))`` tuple, so no nesting can
    accumulate across rebuilds.
    """
    return load_raven_model(spec, disable_dynamic=disable_dynamic)


#: Makes ``rlora.has_lora_aware_patcher_factory`` recognise our factory, so a
#: later attach against a rebuilt patcher is not rejected for having one.
rebuild_raven_patcher.raven_lora_wrapped_factory = load_raven_model  # type: ignore[attr-defined]


def load_raven_diffusion_model(
    unet_name: str,
    lora_name: str,
    *,
    strength: float = 1.0,
    weight_dtype: str = "default",
    alpha: Optional[float] = None,
    row_chunk: Optional[int] = None,
    model_options: Optional[Mapping[str, Any]] = None,
    unet_model_cls: Optional[type] = None,
    base_model_factory: Optional[Callable[..., Any]] = None,
    lora_config: Optional[rlora.RavenBaseConfig] = None,
    manifest_kwargs: Optional[Mapping[str, Any]] = None,
    force_static_patcher: bool = True,
    disable_dynamic: bool = False,
    mods: Optional[compat.ComfyModules] = None,
) -> Any:
    """Node-level entry point: names (or absolute paths) in, ``MODEL`` out.

    ``force_static_patcher`` defaults to ``True``: the node always builds the
    stock ``ModelPatcher`` (v0.1's supported partial CPU offload path). It is
    deliberately not a node input - the DynamicVRAM alternative is optional,
    unverified with this loader, and not something a workflow should toggle.
    Programmatic callers can still pass ``False`` to try it.
    """
    options = dict(model_options or {})
    for key, value in model_options_for_weight_dtype(weight_dtype).items():
        options.setdefault(key, value)
    spec = RavenLoaderSpec(
        unet_path=unet_name,
        lora_path=lora_name,
        strength=float(strength),
        alpha=alpha,
        row_chunk=row_chunk,
        model_options=options,
        unet_model_cls=unet_model_cls,
        base_model_factory=base_model_factory,
        lora_config=lora_config,
        manifest_kwargs=dict(manifest_kwargs or {}),
        force_static_patcher=bool(force_static_patcher),
    )
    return load_raven_model(spec, disable_dynamic=disable_dynamic, mods=mods)


# --------------------------------------------------------------------------
# introspection helpers
# --------------------------------------------------------------------------
def _model_of(target: Any) -> Any:
    model = getattr(target, "model", None)
    return target if model is None else model


def get_raven_attachment(target: Any) -> Optional[runtime_linear.RavenLoraAttachment]:
    """The attachment carried by a patcher (or its model); ``None`` if absent."""
    return getattr(_model_of(target), "raven_lora_attachment", None)


def get_raven_manifest(target: Any) -> Optional[rlora.RavenLoraManifest]:
    return getattr(_model_of(target), "raven_lora_manifest", None)


def official_lora_key_map(target: Any, mods: Optional[compat.ComfyModules] = None) -> Dict[str, str]:
    """``comfy.lora.model_lora_keys_unet`` for this model (generic LoRA format)."""
    mods = mods or compat.import_comfy_modules()
    lora_module = mods.require("lora")
    return lora_module.model_lora_keys_unet(_model_of(target), {})


def official_lora_key_hits(
    target: Any,
    manifest: Optional[rlora.RavenLoraManifest] = None,
    mods: Optional[compat.ComfyModules] = None,
) -> Dict[str, int]:
    """How many RAVEN base keys an official generic-format LoRA can still patch.

    A stock ``LoraLoaderModelOnly`` after this loader resolves
    ``lora_unet_<path>`` -> ``diffusion_model.<path>.weight``. Because the RAVEN
    residual never renames or fuses anything, every mapped base key must still
    be reachable - the 208 ``core`` modules above all.
    """
    manifest = manifest or get_raven_manifest(target)
    if manifest is None:
        raise RavenLoaderError("no RAVEN manifest on {}".format(type(target).__name__))
    key_map = official_lora_key_map(target, mods=mods)
    reachable = set(key_map.values())
    hits: Dict[str, int] = {"total": 0}
    for module in manifest.modules.values():
        category = module.category
        hits.setdefault(category, 0)
        if module.base_key in reachable:
            hits[category] += 1
            hits["total"] += 1
    hits["key_map_size"] = len(key_map)
    return hits
