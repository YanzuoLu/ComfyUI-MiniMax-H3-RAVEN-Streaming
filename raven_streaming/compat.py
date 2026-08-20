"""Upstream feature detection for the surfaces this package binds to.

This module does **feature detection only**. It imports upstream ComfyUI
modules, probes for the exact symbols/attributes the RAVEN lanes call, and
reports what is missing. It never branches behaviour on a version string, never
patches upstream, and never touches private DynamicVRAM internals (no
``dynamic_vbar`` / ``_vbar_get`` / ``LowVramPatch`` poking).

Pinned audit baseline
---------------------
Everything here was read against ComfyUI commit
``c67885b14556cf3e4e061862925282d403d09862`` (``comfyui_version.py`` reports
``0.33.0``). ``pyproject.toml`` *declares* ``requires-comfyui = ">=0.30.0"``;
that lower bound is a target, **not a verified claim** - nothing in this
repository has been exercised on ``0.30``. The feature report below is the
authority: if a probe fails on some version, that version is unsupported no
matter what the declaration says, and if it passes, the version string is not
consulted at all.

Versions are collected only for user-facing messages.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "CompatError",
    "MissingFeatureError",
    "PINNED_COMFY_COMMIT",
    "PINNED_COMFY_VERSION",
    "DECLARED_MIN_COMFYUI",
    "SUPPORT_NOTE",
    "MODULE_NAMES",
    "OPTIONAL_MODULES",
    "ComfyModules",
    "FeatureCheck",
    "FeatureReport",
    "import_comfy_modules",
    "check_features",
    "require_features",
    "comfy_version",
]


#: Commit the design and this probe set were read against.
PINNED_COMFY_COMMIT = "c67885b14556cf3e4e061862925282d403d09862"
#: ``comfyui_version.__version__`` at that commit.
PINNED_COMFY_VERSION = "0.33.0"
#: What ``pyproject.toml`` declares. A target, pending evidence - not verified.
DECLARED_MIN_COMFYUI = "0.30.0"

SUPPORT_NOTE = (
    "Audited against ComfyUI {version} ({commit}). The declared lower bound "
    "{declared} is a target and has not been verified on any run; the feature "
    "report in raven_streaming.compat is the authority, not the version string."
).format(
    version=PINNED_COMFY_VERSION, commit=PINNED_COMFY_COMMIT[:7], declared=DECLARED_MIN_COMFYUI
)


class CompatError(RuntimeError):
    """Base class for upstream-compatibility failures."""


class MissingFeatureError(CompatError):
    """A required upstream module/symbol/attribute is missing or renamed."""


# --------------------------------------------------------------------------
# module table
# --------------------------------------------------------------------------
#: attribute name -> importable module name.
MODULE_NAMES: Dict[str, str] = {
    "utils": "comfy.utils",
    "sd": "comfy.sd",
    "lora": "comfy.lora",
    "model_detection": "comfy.model_detection",
    "model_management": "comfy.model_management",
    "model_patcher": "comfy.model_patcher",
    "model_base": "comfy.model_base",
    "supported_models": "comfy.supported_models",
    "latent_formats": "comfy.latent_formats",
    "minimax_ldm": "comfy.ldm.minimax.model",
    "folder_paths": "folder_paths",
}

#: Modules the core loader can work without (absolute paths bypass them).
OPTIONAL_MODULES = ("folder_paths",)


@dataclass
class ComfyModules:
    """The upstream modules the RAVEN lanes bind to.

    Instances are produced by :func:`import_comfy_modules`. Tests inject tiny
    fakes by pre-populating ``sys.modules`` (the import below is a plain
    ``importlib.import_module``) or by passing ``overrides``.
    """

    utils: Any = None
    sd: Any = None
    lora: Any = None
    model_detection: Any = None
    model_management: Any = None
    model_patcher: Any = None
    model_base: Any = None
    supported_models: Any = None
    latent_formats: Any = None
    minimax_ldm: Any = None
    folder_paths: Any = None
    #: import failures, attribute name -> "<ExcType>: <message>"
    import_errors: Dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        return getattr(self, name, None)

    def require(self, name: str) -> Any:
        module = self.get(name)
        if module is None:
            raise MissingFeatureError(
                "upstream module {!r} ({}) is not importable: {}. {}".format(
                    name,
                    MODULE_NAMES.get(name, "?"),
                    self.import_errors.get(name, "not imported"),
                    SUPPORT_NOTE,
                )
            )
        return module


def import_comfy_modules(overrides: Optional[Mapping[str, Any]] = None) -> ComfyModules:
    """Import every upstream module, recording (not raising) import failures."""
    mods = ComfyModules()
    overrides = dict(overrides or {})
    for attr, module_name in MODULE_NAMES.items():
        if attr in overrides:
            setattr(mods, attr, overrides[attr])
            continue
        try:
            setattr(mods, attr, importlib.import_module(module_name))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            mods.import_errors[attr] = "{}: {}".format(type(exc).__name__, exc)
    return mods


def comfy_version(mods: Optional[ComfyModules] = None) -> str:
    """``comfyui_version.__version__`` if available - message use only."""
    try:
        module = importlib.import_module("comfyui_version")
    except Exception:  # noqa: BLE001
        return "unknown"
    return str(getattr(module, "__version__", "unknown"))


# --------------------------------------------------------------------------
# feature checks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureCheck:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True

    @property
    def status(self) -> str:
        return "PASS" if self.ok else ("FAIL" if self.required else "MISSING (optional)")

    def line(self) -> str:
        return "[{}] {}{}".format(self.status, self.name, " - " + self.detail if self.detail else "")


@dataclass
class FeatureReport:
    checks: List[FeatureCheck] = field(default_factory=list)
    comfy_version: str = "unknown"
    import_errors: Dict[str, str] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: Any = "", required: bool = True) -> FeatureCheck:
        check = FeatureCheck(name, bool(ok), str(detail), bool(required))
        self.checks.append(check)
        return check

    @property
    def failures(self) -> List[FeatureCheck]:
        return [c for c in self.checks if c.required and not c.ok]

    @property
    def optional_missing(self) -> List[FeatureCheck]:
        return [c for c in self.checks if not c.required and not c.ok]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = "ComfyUI version: {} | {}".format(self.comfy_version, SUPPORT_NOTE)
        body = "\n".join(c.line() for c in self.checks)
        tail = "RESULT: {}".format("all required features present" if self.ok else "MISSING FEATURES")
        return "\n".join((head, body, tail))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comfy_version": self.comfy_version,
            "pinned_commit": PINNED_COMFY_COMMIT,
            "pinned_version": PINNED_COMFY_VERSION,
            "declared_min": DECLARED_MIN_COMFYUI,
            "support_note": SUPPORT_NOTE,
            "ok": self.ok,
            "import_errors": dict(self.import_errors),
            "checks": [dataclasses.asdict(c) for c in self.checks],
        }


def _has_attrs(obj: Any, names: Tuple[str, ...]) -> Tuple[bool, str]:
    missing = [n for n in names if not hasattr(obj, n)]
    return (not missing), ("missing: " + ", ".join(missing) if missing else "ok")


def _accepts(func: Callable, parameter: str) -> Tuple[bool, str]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as exc:  # pragma: no cover - builtins
        return False, "no signature: {}".format(exc)
    if parameter in sig.parameters:
        return True, str(sig)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True, "accepted via **kwargs: {}".format(sig)
    return False, "signature {} has no {!r} parameter".format(sig, parameter)


def _sets_attribute(func: Callable, attribute: str) -> Tuple[bool, str]:
    """Does ``func``'s bytecode reference ``self.<attribute>``?

    Used for attributes that only exist on instances (``cached_patcher_init``
    is assigned in ``ModelPatcher.__init__``), where ``hasattr`` on the class
    says nothing.
    """
    code = getattr(func, "__code__", None)
    if code is None:
        return False, "no bytecode"
    ok = attribute in code.co_names
    return ok, "{} in {}.co_names: {}".format(attribute, getattr(func, "__qualname__", func), ok)


def check_features(mods: Optional[ComfyModules] = None) -> FeatureReport:
    """Probe every upstream surface the loader/LoRA lanes need."""
    mods = mods or import_comfy_modules()
    report = FeatureReport(comfy_version=comfy_version(mods), import_errors=dict(mods.import_errors))

    for attr, module_name in MODULE_NAMES.items():
        required = attr not in OPTIONAL_MODULES
        module = mods.get(attr)
        report.add(
            "import {}".format(module_name),
            module is not None,
            mods.import_errors.get(attr, "" if module is not None else "not imported"),
            required=required,
        )

    utils = mods.get("utils")
    if utils is not None:
        ok, detail = _has_attrs(
            utils,
            ("load_torch_file", "save_torch_file", "convert_old_quants",
             "state_dict_prefix_replace", "calculate_parameters", "weight_dtype"),
        )
        report.add("comfy.utils state-dict helpers", ok, detail)
        if hasattr(utils, "load_torch_file"):
            ok, detail = _accepts(utils.load_torch_file, "return_metadata")
            report.add("comfy.utils.load_torch_file(return_metadata=True)", ok, detail)

    detection = mods.get("model_detection")
    if detection is not None:
        ok, detail = _has_attrs(
            detection, ("unet_prefix_from_state_dict", "model_config_from_unet")
        )
        report.add("comfy.model_detection entry points", ok, detail)
        if hasattr(detection, "model_config_from_unet"):
            ok, detail = _accepts(detection.model_config_from_unet, "metadata")
            report.add("model_config_from_unet(metadata=...)", ok, detail)

    mm = mods.get("model_management")
    if mm is not None:
        ok, detail = _has_attrs(
            mm,
            ("get_torch_device", "unet_offload_device", "unet_dtype", "unet_manual_cast",
             "is_device_cpu", "module_size", "load_models_gpu"),
        )
        report.add("comfy.model_management device/dtype helpers", ok, detail)

    patcher_module = mods.get("model_patcher")
    if patcher_module is not None:
        ok, detail = _has_attrs(patcher_module, ("ModelPatcher", "CoreModelPatcher"))
        report.add("comfy.model_patcher.{ModelPatcher,CoreModelPatcher}", ok, detail)
        patcher_cls = getattr(patcher_module, "ModelPatcher", None)
        if patcher_cls is not None:
            ok, detail = _has_attrs(
                patcher_cls,
                ("model_size", "loaded_size", "is_dynamic", "clone", "partially_load",
                 "partially_unload", "model_state_dict", "add_patches"),
            )
            report.add("ModelPatcher size/load/patch API", ok, detail)
            ok, detail = _sets_attribute(patcher_cls.__init__, "cached_patcher_init")
            report.add("ModelPatcher.cached_patcher_init factory slot", ok, detail)
        report.add(
            "comfy.model_patcher.ModelPatcherDynamic (optional: DynamicVRAM)",
            hasattr(patcher_module, "ModelPatcherDynamic"),
            "v0.1 ships on the stock ModelPatcher partial CPU offload path; the "
            "dynamic patcher is optional and unverified with this loader, so its "
            "absence changes nothing",
            required=False,
        )

    model_base = mods.get("model_base")
    if model_base is not None:
        ok, detail = _has_attrs(model_base, ("BaseModel", "MiniMaxH3", "ModelType"))
        report.add("comfy.model_base.{BaseModel,MiniMaxH3,ModelType}", ok, detail)
        base_cls = getattr(model_base, "BaseModel", None)
        if base_cls is not None:
            ok, detail = _accepts(base_cls.__init__, "unet_model")
            report.add("BaseModel(unet_model=...) injection point", ok, detail)
            ok, detail = _has_attrs(base_cls, ("load_model_weights",))
            report.add("BaseModel.load_model_weights", ok, detail)
            if hasattr(base_cls, "load_model_weights"):
                ok, detail = _accepts(base_cls.load_model_weights, "assign")
                report.add("BaseModel.load_model_weights(assign=...)", ok, detail)
        model_type = getattr(model_base, "ModelType", None)
        report.add(
            "comfy.model_base.ModelType.FLOW_AV (H3 sampling type)",
            model_type is not None and hasattr(model_type, "FLOW_AV"),
            "" if model_type is None else str(getattr(model_type, "FLOW_AV", "missing")),
        )

    supported = mods.get("supported_models")
    latent_formats = mods.get("latent_formats")
    if supported is not None:
        config_cls = getattr(supported, "MiniMaxH3", None)
        report.add(
            "comfy.supported_models.MiniMaxH3", config_cls is not None,
            "official H3 model config",
        )
        if config_cls is not None:
            unet_config = getattr(config_cls, "unet_config", {}) or {}
            report.add(
                "MiniMaxH3.unet_config image_model == 'minimax_h3'",
                unet_config.get("image_model") == "minimax_h3",
                str(unet_config),
            )
            sampling = getattr(config_cls, "sampling_settings", {}) or {}
            report.add(
                "MiniMaxH3.sampling_settings carries shift + audio_shift",
                "shift" in sampling and "audio_shift" in sampling,
                str(sampling),
            )
            latent_format = getattr(config_cls, "latent_format", None)
            expected_latent = getattr(latent_formats, "MiniMaxH3AV", None) if latent_formats else None
            report.add(
                "MiniMaxH3.latent_format is comfy.latent_formats.MiniMaxH3AV",
                latent_format is not None
                and expected_latent is not None
                and latent_format is expected_latent,
                "{} vs {}".format(
                    getattr(latent_format, "__name__", latent_format),
                    getattr(expected_latent, "__name__", expected_latent),
                ),
            )
            dtypes = list(getattr(config_cls, "supported_inference_dtypes", []) or [])
            names = [getattr(d, "__str__", lambda: d)() for d in dtypes]
            report.add(
                "MiniMaxH3 supports bfloat16 inference (the published RAVEN base dtype)",
                any("bfloat16" in str(d) for d in dtypes),
                ", ".join(str(n) for n in names),
            )
            report.add(
                "MiniMaxH3.get_model factory", callable(getattr(config_cls, "get_model", None)), ""
            )

    ldm = mods.get("minimax_ldm")
    if ldm is not None:
        ok, detail = _has_attrs(ldm, ("MiniMaxH3Model", "FRAME_PER_TOKEN", "FRAME_RESCALE"))
        report.add("comfy.ldm.minimax.model surface", ok, detail)
        unet_cls = getattr(ldm, "MiniMaxH3Model", None)
        if unet_cls is not None:
            ok, detail = _accepts(unet_cls.__init__, "operations")
            report.add("MiniMaxH3Model(operations=...)", ok, detail)

    lora_module = mods.get("lora")
    if lora_module is not None:
        report.add(
            "comfy.lora.model_lora_keys_unet (official generic LoRA key map)",
            callable(getattr(lora_module, "model_lora_keys_unet", None)),
            "",
        )

    sd = mods.get("sd")
    if sd is not None:
        ok, detail = _has_attrs(
            sd, ("load_diffusion_model", "load_diffusion_model_state_dict", "load_lora_for_models")
        )
        report.add("comfy.sd loader entry points (replication reference)", ok, detail)

    folder_paths = mods.get("folder_paths")
    if folder_paths is not None:
        ok, detail = _has_attrs(
            folder_paths, ("get_full_path_or_raise", "get_filename_list", "get_folder_paths")
        )
        report.add("folder_paths helpers (optional: absolute paths bypass them)", ok, detail,
                   required=False)

    return report


def require_features(
    mods: Optional[ComfyModules] = None, *, report: Optional[FeatureReport] = None
) -> ComfyModules:
    """Return the modules, raising :class:`MissingFeatureError` on any failure."""
    mods = mods or import_comfy_modules()
    report = report or check_features(mods)
    if not report.ok:
        raise MissingFeatureError(
            "incompatible ComfyUI (version {}): {} required feature(s) missing.\n{}\n{}".format(
                report.comfy_version,
                len(report.failures),
                "\n".join(c.line() for c in report.failures),
                SUPPORT_NOTE,
            )
        )
    return mods
