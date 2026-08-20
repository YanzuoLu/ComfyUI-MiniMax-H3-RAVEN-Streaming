#!/usr/bin/env python3
"""Probe: are RAVEN LoRA A/B parameters seen by the Comfy model patchers?

Two scenarios, both against a toy DiT built from *real* ``comfy.ops`` Linear
modules at a pinned ComfyUI commit:

``static``
    ``comfy.model_patcher.ModelPatcher`` (what ``CoreModelPatcher`` binds to
    when DynamicVRAM is off, e.g. on CPU/MPS/AMD). Checks ``model_size()``
    accounting, ``_load_list()`` membership as *direct* params of a still-leaf
    module, ``load(full_load=True)``, cold partial load, cross-device residual,
    ``partially_unload`` -> ``partially_load``, and base-key stability.

``dynamic``
    ``comfy.model_patcher.ModelPatcherDynamic``, which is what stock ComfyUI
    rebinds ``CoreModelPatcher`` to on NVIDIA/ROCm (``main.py``:
    ``CoreModelPatcher = ModelPatcherDynamic`` + ``aimdo_enabled = True``).
    This is the default production path, so the probe reproduces that rebinding
    instead of trusting the static alias, drives the model through
    ``comfy.model_management.load_models_gpu`` (never ``full_load``), and checks
    that A/B land on the GPU in FP32, that they are counted in
    ``loaded_size()`` / ``model_loaded_weight_memory``, that the residual is
    numerically right, and that two load/unload rounds neither drop nor rename
    them. Requires CUDA + a working ``comfy_aimdo``; otherwise the scenario is
    reported as skipped with the reason.

It also records the counter-example that motivates the design: attaching A/B as
*child modules* makes ``_load_list`` classify the base Linear as a non-leaf
"default random weights" module and drop it from the loading plan entirely.

Usage::

    python tools/probe_lora_offload.py                     # RAVEN_PROBE_DEVICE or cpu
    python tools/probe_lora_offload.py --device cuda       # on vr-1: static + dynamic
    python tools/probe_lora_offload.py --scenario dynamic --json report.json

Environment: ``COMFYUI_PATH`` / ``COMFYUI_UPSTREAM_PATH`` locate the checkout,
``RAVEN_PROBE_DEVICE`` sets the default device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from raven_streaming import lora as rlora  # noqa: E402
from raven_streaming import runtime_linear as rrl  # noqa: E402

DEFAULT_COMFY = ROOT / ".cache" / "upstream" / "ComfyUI"
COMFY_ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")
DEVICE_ENV_VAR = "RAVEN_PROBE_DEVICE"


def resolve_comfy_path(explicit: Optional[str] = None) -> Path:
    """``--comfy`` > ``COMFYUI_PATH`` > ``COMFYUI_UPSTREAM_PATH`` > bundled cache."""
    candidates = [explicit] + [os.environ.get(v) for v in COMFY_ENV_VARS] + [str(DEFAULT_COMFY)]
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand).expanduser()
        if (path / "comfy" / "model_patcher.py").is_file():
            return path
    return Path(explicit) if explicit else DEFAULT_COMFY


def default_device() -> str:
    return os.environ.get(DEVICE_ENV_VAR) or "cpu"

# small but structurally identical to the official model
PROBE_CONFIG = rlora.RavenBaseConfig(
    hidden_size=64,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=4,
    attention_head_dim=8,
    ffn_hidden_size=48,
    latents_dim=4,
    audio_latents_dim=6,
    text_dim=32,
    timestep_input_dim=16,
    time_embed_hidden_size=64,
    time_embed_dim=48,
)
PROBE_COUNTS = {"core": 12, "adaln": 3, "time": 2, "boundary": 5}
PROBE_RANK = 8
TARGET = "blocks.0.mlp.fc1"


# --------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    scenario: str = "static"
    skipped: bool = False

    @property
    def status(self) -> str:
        return "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")

    def line(self) -> str:
        return "[{}] ({}) {}{}".format(
            self.status, self.scenario, self.name,
            " - " + self.detail if self.detail else "")


@dataclass
class Report:
    comfy_path: str = ""
    comfy_commit: str = ""
    device: str = ""
    checks: List[Check] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    patcher_classes: Dict[str, str] = field(default_factory=dict)
    scenarios: Dict[str, str] = field(default_factory=dict)
    scenario: str = "static"

    def add(self, name: str, ok: bool, detail: Any = "", scenario: Optional[str] = None) -> Check:
        c = Check(name, bool(ok), str(detail), scenario or self.scenario)
        self.checks.append(c)
        return c

    def skip(self, name: str, detail: Any = "", scenario: Optional[str] = None) -> Check:
        c = Check(name, True, str(detail), scenario or self.scenario, skipped=True)
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if not c.skipped)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comfy_path": self.comfy_path,
            "comfy_commit": self.comfy_commit,
            "device": self.device,
            "ok": self.ok,
            "patcher_classes": self.patcher_classes,
            "scenarios": self.scenarios,
            "checks": [
                {"scenario": c.scenario, "name": c.name, "ok": c.ok,
                 "skipped": c.skipped, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
            "notes": self.notes,
        }

    def render(self) -> str:
        head = "ComfyUI {} @ {}  device={}".format(self.comfy_path, self.comfy_commit or "?", self.device)
        classes = "patcher classes: {}".format(self.patcher_classes or "{}")
        status = "scenarios: {}".format(self.scenarios or "{}")
        body = "\n".join(c.line() for c in self.checks)
        notes = "\n".join("note: " + n for n in self.notes)
        tail = "RESULT: {}".format("all checks passed" if self.ok else "FAILURES PRESENT")
        return "\n".join(x for x in (head, classes, status, body, notes, tail) if x)


# --------------------------------------------------------------------------
def import_comfy(comfy_path: Path):
    if not (comfy_path / "comfy" / "model_patcher.py").is_file():
        raise RuntimeError("no ComfyUI checkout at {}".format(comfy_path))
    p = str(comfy_path)
    if p not in sys.path:
        sys.path.insert(0, p)
    import comfy.model_management  # noqa: F401
    import comfy.model_patcher  # noqa: F401
    import comfy.ops  # noqa: F401

    return sys.modules["comfy"]


def comfy_commit(comfy_path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(comfy_path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _ensure_parent(root: nn.Module, path: str):
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        child = getattr(node, part, None)
        if child is None:
            child = nn.Module()
            node.add_module(part, child)
        node = child
    return node, parts[-1]


class ProbeModel(nn.Module):
    """Stand-in for ``comfy.model_base.BaseModel``: has ``.diffusion_model``."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.diffusion_model = dit


def build_probe_model(comfy, ops=None, dtype=torch.float32, seed: int = 0) -> ProbeModel:
    ops = ops or comfy.ops.manual_cast
    gen = torch.Generator().manual_seed(seed)
    dit = nn.Module()
    for path, entry in PROBE_CONFIG.modules().items():
        parent, leaf = _ensure_parent(dit, path)
        lin = ops.Linear(entry.in_features, entry.out_features, bias=True, dtype=dtype,
                         device=torch.device("cpu"))
        if getattr(lin, "weight", None) is None:  # aimdo lazy-init path
            lin.weight = nn.Parameter(torch.empty(entry.weight_shape, dtype=dtype))
            lin.bias = nn.Parameter(torch.empty(entry.out_features, dtype=dtype))
        with torch.no_grad():
            lin.weight.copy_(torch.randn(entry.weight_shape, generator=gen).to(dtype) * 0.05)
            lin.bias.copy_(torch.randn(entry.out_features, generator=gen).to(dtype) * 0.05)
        lin.weight.requires_grad_(False)
        lin.bias.requires_grad_(False)
        parent.add_module(leaf, lin)
    return ProbeModel(dit)


def probe_weights(seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    pairs = {}
    for path, entry in PROBE_CONFIG.modules().items():
        pairs[path] = (
            torch.randn(PROBE_RANK, entry.in_features, generator=gen) * 0.1,
            torch.randn(entry.out_features, PROBE_RANK, generator=gen) * 0.1,
        )
    return pairs


def probe_manifest() -> rlora.RavenLoraManifest:
    shapes = {}
    for path, entry in PROBE_CONFIG.modules().items():
        shapes[rlora.PEFT_PREFIX + path + ".lora_A.weight"] = (PROBE_RANK, entry.in_features)
        shapes[rlora.PEFT_PREFIX + path + ".lora_B.weight"] = (entry.out_features, PROBE_RANK)
    tensors = {}
    offset = 0
    for name, shape in shapes.items():
        nbytes = 4 * shape[0] * shape[1]
        tensors[name] = rlora.TensorInfo(name, "F32", tuple(shape), offset, offset + nbytes)
        offset += nbytes
    header = rlora.SafetensorsHeader(tensors=tensors, metadata={}, data_offset=8)
    return rlora.build_manifest(header, PROBE_CONFIG, expected_counts=PROBE_COUNTS)


def _find_module(patcher, path: str):
    return patcher.model.get_submodule("diffusion_model." + path)


def _load_entry(patcher, path: str):
    want = "diffusion_model." + path
    for item in patcher._load_list():
        if item[-3] == want:
            return item
    return None


# --------------------------------------------------------------------------
def run_probe(
    comfy_path: Path = DEFAULT_COMFY,
    device: Optional[str] = None,
    scenarios: Sequence[str] = ("static", "dynamic"),
) -> Report:
    comfy_path = resolve_comfy_path(str(comfy_path) if comfy_path else None)
    comfy = import_comfy(comfy_path)
    device = device or default_device()
    report = Report(comfy_path=str(comfy_path), comfy_commit=comfy_commit(comfy_path),
                    device=device)

    bound = comfy.model_patcher.CoreModelPatcher
    report.patcher_classes["CoreModelPatcher (as imported)"] = getattr(bound, "__name__", str(bound))
    report.patcher_classes["ModelPatcherDynamic available"] = str(
        hasattr(comfy.model_patcher, "ModelPatcherDynamic"))
    report.notes.append(
        "main.py rebinds CoreModelPatcher to ModelPatcherDynamic when DynamicVRAM is "
        "supported (NVIDIA / ROCm>=7.14) and comfy_aimdo initialises; importing comfy "
        "without main.py leaves the static alias, so the dynamic scenario reproduces "
        "that rebinding explicitly."
    )

    manifest = probe_manifest()
    weights = probe_weights()

    if "static" in scenarios:
        report.scenario = "static"
        _static_scenario(comfy, report, torch.device(device), torch.device("cpu"),
                         manifest, weights)
    if "dynamic" in scenarios:
        report.scenario = "dynamic"
        _dynamic_scenario(comfy, report, device, manifest, weights)
    report.scenario = "static"
    return report


def _static_scenario(comfy, report: Report, dev, offload, manifest, weights) -> None:
    MP = comfy.model_patcher.ModelPatcher
    report.patcher_classes["static"] = MP.__name__
    report.scenarios["static"] = "ran on {}".format(dev)

    # ---- 1. model_size accounting -------------------------------------
    model = build_probe_model(comfy)
    baseline = MP(model, load_device=dev, offload_device=offload).model_size()
    att = rlora.attach_raven_lora(model, manifest, strength=1.0, weights=weights)
    patcher = MP(model, load_device=dev, offload_device=offload)
    with_lora = patcher.model_size()
    report.add(
        "model_size() counts the LoRA parameters",
        with_lora - baseline == att.parameter_bytes(),
        "{} - {} = {} bytes, attachment reports {}".format(
            with_lora, baseline, with_lora - baseline, att.parameter_bytes()),
    )
    report.notes.append(
        "ModelPatcher.model_size() caches into .size, so the LoRA must be attached "
        "before the patcher is built; attach_raven_lora() now refuses a patcher whose "
        "size is already cached / which is already loaded (never clears it silently)."
    )
    try:
        rlora.attach_raven_lora(patcher, manifest, strength=1.0, weights=weights)
        late_attach_error = ""
    except rrl.RavenAttachError as exc:
        late_attach_error = str(exc)
    report.add(
        "attaching to an already-measured patcher fails loud",
        bool(late_attach_error),
        late_attach_error[:120],
    )

    # ---- 2. _load_list sees them as direct params of a leaf module -----
    entry = _load_entry(patcher, TARGET)
    module = _find_module(patcher, TARGET)
    params = entry[-1] if entry is not None else {}
    report.add(
        "_load_list() still contains the LoRA'd module",
        entry is not None,
        "entry for diffusion_model.{}".format(TARGET),
    )
    report.add(
        "_load_list() exposes raven_lora_A_0/raven_lora_B_0 as direct params",
        {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"} == set(params),
        "params={}".format(sorted(params)),
    )
    if entry is not None:
        module_mem = entry[-4]
        expected_mem = comfy.model_management.module_size(module)
        lora_bytes = sum(
            getattr(module, n).numel() * 4 for n in ("raven_lora_A_0", "raven_lora_B_0")
        )
        report.add(
            "module_size() of the LoRA'd module includes A/B",
            module_mem == expected_mem and module_mem > lora_bytes,
            "module_mem={} lora_bytes={}".format(module_mem, lora_bytes),
        )

    # ---- 3. base keys unchanged ---------------------------------------
    base_keys = [m.base_key for m in manifest.modules.values()]
    sd = patcher.model_state_dict()
    report.add(
        "all base diffusion_model.*.weight keys still present",
        all(k in sd for k in base_keys),
        "{} keys".format(len(base_keys)),
    )
    report.add(
        "LoRA params do not masquerade as patchable .weight keys",
        not any(k.endswith(".weight") or k.endswith(".bias") for k in att.state_dict_keys()),
        att.state_dict_keys()[:2],
    )

    # ---- 4. full load --------------------------------------------------
    x = torch.randn(4, PROBE_CONFIG.hidden_size)
    reference = _reference_output(model, weights, manifest, x)
    patcher.load(dev, lowvram_model_memory=0, full_load=True)
    module = _find_module(patcher, TARGET)
    devices = {n: str(p.device) for n, p in module.named_parameters(recurse=False)}
    report.add(
        "full load moves A/B onto the load device with the base weight",
        len(set(devices.values())) == 1 and dev.type in next(iter(devices.values())),
        str(devices),
    )
    out = module(x.to(dev))
    report.add(
        "residual is applied after a full load",
        torch.allclose(out.cpu(), reference, atol=1e-5),
        "max|diff|={:.3e}".format(float((out.cpu() - reference).abs().max())),
    )

    patcher.detach()

    # ---- 5. partial load, from cold ------------------------------------
    # a fresh model/patcher: load() never moves weights back, so a partial load
    # is only meaningful before any full load happened.
    model = build_probe_model(comfy)
    att = rlora.attach_raven_lora(model, manifest, strength=1.0, weights=weights)
    patcher = MP(model, load_device=dev, offload_device=offload)
    total = patcher.model_size()
    patcher.load(dev, lowvram_model_memory=total // 3, full_load=False)
    lowvram = bool(getattr(patcher.model, "model_lowvram", False))
    module = _find_module(patcher, TARGET)
    names = {n for n, _ in module.named_parameters(recurse=False)}
    report.add(
        "partial load keeps every LoRA parameter (no loss, no rename)",
        {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"} == names,
        "lowvram={} params={}".format(lowvram, sorted(names)),
    )
    missing = [
        k for k in att.state_dict_keys() if k not in patcher.model_state_dict()
    ]
    report.add(
        "partial load: every LoRA key is still in the state dict",
        not missing,
        "missing={}".format(missing[:4]),
    )
    out = module(x.to(dev))
    report.add(
        "residual is applied after a partial load",
        torch.allclose(out.cpu(), reference, atol=1e-5),
        "max|diff|={:.3e} lowvram={} devices={}".format(
            float((out.cpu() - reference).abs().max()), lowvram,
            {n: str(p.device) for n, p in module.named_parameters(recurse=False)}),
    )

    # a module that stayed on the offload device: input on the load device must
    # still produce the right residual (A/B are cast per call, like the base weight)
    offloaded_path = None
    for path in PROBE_CONFIG.modules():
        m = _find_module(patcher, path)
        if m.weight.device.type != dev.type:
            offloaded_path = path
            break
    if offloaded_path is None:
        report.skip(
            "cross-device residual: offloaded module, input on the load device",
            "no module remained on the offload device during the partial load "
            "(load_device == offload_device on this box)")
    else:
        m = _find_module(patcher, offloaded_path)
        entry = PROBE_CONFIG.modules()[offloaded_path]
        xo = torch.randn(3, entry.in_features)
        a, b = weights[offloaded_path]
        ref = torch.nn.functional.linear(
            xo, m.weight.detach().float().cpu(), m.bias.detach().float().cpu()
        ) + torch.nn.functional.linear(
            torch.nn.functional.linear(xo, a), b
        ) * (manifest.alpha / manifest.rank)
        got = m(xo.to(dev))
        report.add(
            "cross-device residual: offloaded module, input on the load device",
            torch.allclose(got.cpu(), ref, atol=1e-5),
            "{} weight on {}, out on {}, max|diff|={:.3e}".format(
                offloaded_path, m.weight.device, got.device,
                float((got.cpu() - ref).abs().max())),
        )

    freed = patcher.partially_unload(offload, memory_to_free=total)
    module = _find_module(patcher, TARGET)
    names = {n for n, _ in module.named_parameters(recurse=False)}
    report.add(
        "partially_unload keeps every LoRA parameter",
        {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"} == names,
        "freed={} bytes".format(freed),
    )
    loaded = patcher.partially_load(dev, extra_memory=total)
    out = module(x.to(dev))
    report.add(
        "residual survives partially_unload -> partially_load",
        torch.allclose(out.cpu(), reference, atol=1e-5),
        "reloaded={} bytes, max|diff|={:.3e}".format(
            loaded, float((out.cpu() - reference).abs().max())),
    )
    patcher.detach()

    # ---- 6. counter-example: child modules disappear -------------------
    report.checks.append(_child_module_counterexample(comfy, dev, offload))


# --------------------------------------------------------------------------
# dynamic (stock ComfyUI on NVIDIA/ROCm) scenario
# --------------------------------------------------------------------------
def enable_dynamic_vram(comfy) -> str:
    """Reproduce main.py's DynamicVRAM enablement. Returns "" on success."""
    if not hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        return "this ComfyUI has no ModelPatcherDynamic"
    try:
        import comfy_aimdo.control  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return "comfy_aimdo unavailable: {}".format(exc)
    if comfy.model_management.torch_version_numeric < (2, 8):
        return "DynamicVRAM needs torch >= 2.8"
    try:
        devices = list(comfy.model_management.get_all_torch_devices())
        try:
            # (device index, extra headroom bytes) - 0 is the ComfyUI default
            ok = comfy_aimdo.control.init_devices((d.index, 0) for d in devices)
        except TypeError:  # comfy-aimdo 0.4.9 protocol
            ok = comfy_aimdo.control.init_devices(d.index for d in devices)
    except Exception as exc:  # noqa: BLE001
        return "comfy_aimdo.init_devices failed: {}".format(exc)
    if not ok:
        return "comfy_aimdo.init_devices returned False"
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    return ""


def _dynamic_scenario(comfy, report: Report, device: str, manifest, weights) -> None:
    dev = torch.device(device)
    if dev.type != "cuda":
        reason = (
            "device={} is not CUDA; ModelPatcherDynamic.__new__ reroutes CPU load "
            "devices back to ModelPatcher and comfy_aimdo needs a CUDA index".format(device)
        )
        report.scenarios["dynamic"] = "skipped: " + reason
        report.skip("dynamic ModelPatcherDynamic path", reason)
        return
    reason = enable_dynamic_vram(comfy)
    if reason:
        report.scenarios["dynamic"] = "skipped: " + reason
        report.skip("dynamic ModelPatcherDynamic path", reason)
        return

    MPD = comfy.model_patcher.ModelPatcherDynamic
    report.patcher_classes["dynamic"] = MPD.__name__
    report.patcher_classes["CoreModelPatcher (after rebind)"] = getattr(
        comfy.model_patcher.CoreModelPatcher, "__name__", "?")
    report.scenarios["dynamic"] = "ran on {}".format(dev)

    offload = comfy.model_management.unet_offload_device()
    model = build_probe_model(comfy, dtype=torch.bfloat16)
    att = rlora.attach_raven_lora(model, manifest, strength=1.0, weights=weights)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, load_device=dev, offload_device=offload)
    report.add(
        "CoreModelPatcher instantiates ModelPatcherDynamic on CUDA",
        isinstance(patcher, MPD) and patcher.is_dynamic(),
        "{} is_dynamic={}".format(type(patcher).__name__, patcher.is_dynamic()),
    )
    report.add(
        "dynamic model_size() counts the LoRA parameters",
        patcher.model_size() > att.parameter_bytes(),
        "model_size={} lora_bytes={}".format(patcher.model_size(), att.parameter_bytes()),
    )

    entry = PROBE_CONFIG.modules()[TARGET]
    x = torch.randn(4, entry.in_features, dtype=torch.bfloat16)
    module = _find_module(patcher, TARGET)
    ref = _bf16_reference(module, weights[TARGET], manifest, x)

    names_expected = {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"}
    for round_index in (1, 2):
        # the production entry point: no full_load, no direct load() call
        comfy.model_management.load_models_gpu([patcher], memory_required=0)
        module = _find_module(patcher, TARGET)
        params = {n: p for n, p in module.named_parameters(recurse=False)}
        report.add(
            "round {}: LoRA parameters kept (no loss, no rename)".format(round_index),
            set(params) == names_expected,
            sorted(params),
        )
        a = params.get("raven_lora_A_0")
        b = params.get("raven_lora_B_0")
        report.add(
            "round {}: A/B are FP32 on the GPU".format(round_index),
            a is not None and b is not None
            and a.dtype == torch.float32 and b.dtype == torch.float32
            and a.device.type == "cuda" and b.device.type == "cuda",
            "A: {} {} B: {} {}".format(
                None if a is None else a.dtype, None if a is None else a.device,
                None if b is None else b.dtype, None if b is None else b.device),
        )
        loaded_size = patcher.loaded_size()
        weight_mem = int(getattr(patcher.model, "model_loaded_weight_memory", 0))
        report.add(
            "round {}: A/B counted in loaded_size()/model_loaded_weight_memory".format(round_index),
            loaded_size > 0 and weight_mem >= att.parameter_bytes(),
            "loaded_size={} model_loaded_weight_memory={} lora_bytes={}".format(
                loaded_size, weight_mem, att.parameter_bytes()),
        )
        out = module(x.to(dev))
        diff = float((out.float().cpu() - ref).abs().max())
        # bf16 GEMM accumulation differs from the FP32 reference by a few bf16 ULP
        tol = 8 * torch.finfo(torch.bfloat16).eps * max(1.0, float(ref.abs().max()))
        report.add(
            "round {}: residual is numerically correct under dynamic loading".format(round_index),
            diff <= tol,
            "max|diff|={:.3e} tol={:.3e} (bf16 output)".format(diff, tol),
        )
        report.add(
            "round {}: hook ran".format(round_index),
            att.call_counts()[TARGET] >= round_index,
            "calls={} chunks={} row_chunk={}".format(
                att.call_counts()[TARGET], att.chunk_counts()[TARGET],
                att.row_chunks()[TARGET]),
        )
        comfy.model_management.unload_all_models()

    sd = patcher.model_state_dict()
    report.add(
        "dynamic: base keys unchanged after two load/unload rounds",
        all(m.base_key in sd for m in manifest.modules.values()),
        "{} base keys".format(len(manifest.modules)),
    )
    patcher.detach()


def _bf16_reference(module, ab, manifest, x: torch.Tensor) -> torch.Tensor:
    a, b = ab
    w = module.weight.detach().float().cpu()
    bias = module.bias.detach().float().cpu()
    xf = x.float()
    base = torch.nn.functional.linear(xf, w, bias).to(torch.bfloat16).float()
    resid = torch.nn.functional.linear(torch.nn.functional.linear(xf, a), b)
    return (base + resid * (manifest.alpha / manifest.rank)).to(torch.bfloat16).float()


def _reference_output(model, weights, manifest, x: torch.Tensor) -> torch.Tensor:
    """PEFT reference for the probed module, computed from the raw parameters."""
    module = model.get_submodule("diffusion_model." + TARGET)
    a, b = weights[TARGET]
    w = module.weight.detach().float().cpu()
    bias = module.bias.detach().float().cpu()
    base = torch.nn.functional.linear(x, w, bias)
    scale = manifest.alpha / manifest.rank
    return base + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * scale


def _child_module_counterexample(comfy, dev, offload) -> Check:
    """Show why A/B must be direct parameters, not child modules."""
    model = build_probe_model(comfy)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=dev, offload_device=offload)
    before = _load_entry(patcher, TARGET) is not None
    module = model.get_submodule("diffusion_model." + TARGET)
    module.add_module("lora_A", nn.Linear(module.in_features, 4, bias=False))
    module.add_module("lora_B", nn.Linear(4, module.out_features, bias=False))
    after = _load_entry(patcher, TARGET) is not None
    return Check(
        "child-module attachment would drop the module from _load_list "
        "(counter-example, expected to be dropped)",
        before and not after,
        "in _load_list before={} after={}".format(before, after),
    )


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comfy", default=None,
                    help="ComfyUI checkout (default: COMFYUI_PATH / COMFYUI_UPSTREAM_PATH / cache)")
    ap.add_argument("--device", default=default_device(),
                    help="load device (cpu / mps / cuda / cuda:0); env RAVEN_PROBE_DEVICE")
    ap.add_argument("--scenario", default="both", choices=("static", "dynamic", "both"))
    ap.add_argument("--json", default=None, help="write the report as JSON to this path")
    args = ap.parse_args(argv)

    scenarios = ("static", "dynamic") if args.scenario == "both" else (args.scenario,)
    try:
        report = run_probe(resolve_comfy_path(args.comfy), args.device, scenarios)
    except Exception as exc:  # noqa: BLE001 - the probe itself is the evidence
        print("PROBE FAILED TO RUN: {}: {}".format(type(exc).__name__, exc))
        print(
            "If the ModelPatcher API cannot host the A/B parameters at all, the fallback is "
            "an explicit side-car: keep A/B in a separate nn.Module registered on the model "
            "root and wrapped in its own CoreModelPatcher (additional_models), accepting "
            "independent offload granularity."
        )
        raise

    print(report.render())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
