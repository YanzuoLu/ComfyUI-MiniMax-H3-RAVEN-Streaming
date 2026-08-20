#!/usr/bin/env python3
"""Probe: build the RAVEN M1 model and (optionally) load it under Comfy offload.

What it answers
---------------
1. Does :func:`raven_streaming.loader.load_raven_model` build a **standard**
   ``MODEL`` from the official full non-pruned BF16 MiniMax-H3 DiT plus the
   mandatory RAVEN PEFT LoRA, with the residual attached *before* the patcher
   measures anything (``model_size()`` must already include the LoRA bytes)?
2. Do the official generic-format LoRA keys still resolve afterwards
   (``comfy.lora.model_lora_keys_unet`` -> the 208 ``core``
   ``diffusion_model.*.weight`` keys, plus adaln/time/boundary)?
3. What does Comfy's **stock** ``ModelPatcher`` offload actually do with it -
   full load, partial load, lowvram - and where do the FP32 A/B parameters end
   up?

The release gate is the static path
-----------------------------------
v0.1 ships on the stock ``ModelPatcher`` partial CPU offload path, so that is
what this probe runs by default (``spec.force_static_patcher=True``, i.e.
upstream's ``disable_dynamic=True`` class choice) and that is the only result
that decides the exit code. ``--dynamic`` additionally reproduces ``main.py``'s
DynamicVRAM/aimdo enablement and repeats the build on ``ModelPatcherDynamic``;
it is **optional and unverified**, so when it cannot run (no aimdo, no CUDA,
older torch) it is reported as ``SKIP`` with the reason and the static gate is
unaffected. A ``--dynamic`` failure is likewise recorded, never merged into the
static verdict.

Memory expectations for the real run
------------------------------------
The official full non-pruned BF16 DiT is ~66 GB of weights and the published
RAVEN adapter adds ~5 GB of FP32 residual parameters. Even ``--load-mode none``
materialises all of that in **host RAM** (>= 128 GB is the documented baseline),
because the residual is attached to the live module tree before the patcher is
built. ``--weight-dtype fp32`` doubles the base weights to 132 GB+ on top of
that and is a debugging option only.

It reports build time, ``force_static``/effective patcher class, model size,
LoRA bytes, base key counts, official LoRA key hits, per-device weight placement
after loading, peak host RSS and peak VRAM. Nothing is sampled: ``--load-mode
none`` builds only, and a forward pass happens only with ``--forward`` (one tiny
GEMM through a single LoRA'd linear - never a DiT forward, which at the official
bidirectional geometry would be far heavier than the M1 gate needs).

Usage::

    # build only (host RAM >= 128 GB for the real 66 GB base + 5 GB adapter)
    python tools/probe_model_loader.py --base /models/h3.safetensors \
        --lora /models/raven.safetensors --load-mode none

    # vr / H100-H200: stock partial CPU offload with 24 GiB of VRAM taken away
    python tools/probe_model_loader.py --base ... --lora ... --device cuda \
        --reserve-vram 24 --load-mode auto --json report.json

    # force the lowvram regime regardless of free VRAM
    python tools/probe_model_loader.py --base ... --lora ... --device cuda \
        --force-lowvram --load-mode auto

    # optional, unverified: also try the DynamicVRAM patcher (SKIPs cleanly)
    python tools/probe_model_loader.py --base ... --lora ... --device cuda \
        --load-mode auto --dynamic

Environment: ``COMFYUI_PATH`` / ``COMFYUI_UPSTREAM_PATH`` locate the checkout,
``RAVEN_PROBE_DEVICE`` sets the default device.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

DEFAULT_COMFY = ROOT / ".cache" / "upstream" / "ComfyUI"
COMFY_ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")
DEVICE_ENV_VAR = "RAVEN_PROBE_DEVICE"
LOAD_MODES = ("none", "auto", "full", "partial")


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


def import_comfy(comfy_path: Path):
    if not (comfy_path / "comfy" / "model_patcher.py").is_file():
        raise RuntimeError("no ComfyUI checkout at {}".format(comfy_path))
    p = str(comfy_path)
    if p not in sys.path:
        sys.path.insert(0, p)
    import comfy.model_management  # noqa: F401
    import comfy.model_patcher  # noqa: F401

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


# --------------------------------------------------------------------------
def peak_host_rss_bytes() -> int:
    """``ru_maxrss`` normalised to bytes (KiB on Linux, bytes on macOS)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if platform.system() == "Darwin" else int(raw) * 1024


def cuda_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
    }


def device_histogram(module: torch.nn.Module) -> Dict[str, Dict[str, int]]:
    """Parameter bytes per device, split into base weights and RAVEN A/B."""
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {"base_bytes": 0, "lora_bytes": 0,
                                                          "base_tensors": 0, "lora_tensors": 0})
    for name, param in module.named_parameters():
        key = str(param.device)
        bucket = "lora" if ".raven_lora_" in name or name.startswith("raven_lora_") else "base"
        out[key]["{}_bytes".format(bucket)] += param.numel() * param.element_size()
        out[key]["{}_tensors".format(bucket)] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


@dataclass
class Report:
    comfy_path: str = ""
    comfy_commit: str = ""
    comfy_version: str = ""
    device: str = ""
    offload_device: str = ""
    load_mode: str = "none"
    force_lowvram: bool = False
    reserve_vram_bytes: int = 0
    vram_state: str = ""
    #: the release gate: stock ModelPatcher (upstream's disable_dynamic class)
    force_static: bool = True
    features_ok: bool = False
    feature_failures: List[str] = field(default_factory=list)
    build: Dict[str, Any] = field(default_factory=dict)
    load: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    forward: Dict[str, Any] = field(default_factory=dict)
    #: optional, unverified DynamicVRAM scenario; never part of :attr:`ok`
    dynamic: Dict[str, Any] = field(default_factory=lambda: {"status": "not requested"})
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Only the static (stock ModelPatcher) scenario decides the verdict."""
        return self.features_ok and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "comfy_path": self.comfy_path,
            "comfy_commit": self.comfy_commit,
            "comfy_version": self.comfy_version,
            "device": self.device,
            "offload_device": self.offload_device,
            "load_mode": self.load_mode,
            "force_lowvram": self.force_lowvram,
            "reserve_vram_bytes": self.reserve_vram_bytes,
            "vram_state": self.vram_state,
            "force_static": self.force_static,
            "features_ok": self.features_ok,
            "feature_failures": list(self.feature_failures),
            "build": self.build,
            "load": self.load,
            "memory": self.memory,
            "forward": self.forward,
            "dynamic": self.dynamic,
            "notes": list(self.notes),
            "errors": list(self.errors),
        }

    def render(self) -> str:
        lines = [
            "ComfyUI {} @ {} (version {})".format(self.comfy_path, self.comfy_commit, self.comfy_version),
            "device={} offload={} load_mode={} lowvram_forced={} reserve={} vram_state={}".format(
                self.device, self.offload_device, self.load_mode, self.force_lowvram,
                _gib(self.reserve_vram_bytes), self.vram_state),
            "scenario: static (stock ModelPatcher partial CPU offload) force_static={}".format(
                self.force_static),
            "features: {}".format("ok" if self.features_ok else "MISSING: " + "; ".join(self.feature_failures)),
        ]
        if self.build:
            lines += [
                "build: {:.2f}s patcher={} model={} dit={} dtype={} manual_cast={}".format(
                    self.build.get("build_seconds", 0.0), self.build.get("patcher_class"),
                    self.build.get("model_class"), self.build.get("unet_model_class"),
                    self.build.get("unet_dtype"), self.build.get("manual_cast_dtype")),
                "model_size={} lora_bytes={} lora_modules={} rank={} alpha={} strength={}".format(
                    _gib(self.build.get("model_size", 0)), _gib(self.build.get("lora_bytes", 0)),
                    self.build.get("lora_modules"), self.build.get("lora_rank"),
                    self.build.get("lora_alpha"), self.build.get("lora_strength")),
                "base keys={} left_over={} parameters={} ckpt_dtype={}".format(
                    self.build.get("base_key_count"), len(self.build.get("left_over_keys", [])),
                    self.build.get("parameters"), self.build.get("checkpoint_weight_dtype")),
                "official LoRA key hits: {}".format(self.build.get("official_key_hits")),
            ]
        if self.load:
            lines += [
                "load: {:.2f}s loaded_size={} lowvram={} current_device={}".format(
                    self.load.get("load_seconds", 0.0), _gib(self.load.get("loaded_size", 0)),
                    self.load.get("model_lowvram"), self.load.get("current_loaded_device")),
                "placement: {}".format(json.dumps(self.load.get("devices", {}))),
            ]
        if self.memory:
            lines.append("memory: peak_rss={} cuda={}".format(
                _gib(self.memory.get("peak_rss_bytes", 0)),
                {k: _gib(v) for k, v in self.memory.get("cuda_after_load", {}).items()}))
        if self.forward:
            lines.append("forward: {}".format(json.dumps(self.forward)))
        status = self.dynamic.get("status", "not requested")
        lines.append("dynamic (optional, unverified): {}{}".format(
            status.upper(),
            " - " + str(self.dynamic.get("reason")) if self.dynamic.get("reason") else ""))
        if self.dynamic.get("build"):
            lines.append("  dynamic build: patcher={} is_dynamic={} model_size={}".format(
                self.dynamic["build"].get("patcher_class"),
                self.dynamic["build"].get("is_dynamic"),
                _gib(self.dynamic["build"].get("model_size", 0))))
        lines += ["note: " + n for n in self.notes]
        lines += ["ERROR: " + e for e in self.errors]
        lines.append("RESULT: {}".format("ok" if self.ok else "FAILED"))
        return "\n".join(lines)


def _gib(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "{:.3f}GiB".format(n / (1024 ** 3))


# --------------------------------------------------------------------------
def enable_dynamic_vram(comfy) -> str:
    """Reproduce ``main.py``'s DynamicVRAM enablement. Returns "" on success.

    Optional path only: every failure mode returns a reason string, which the
    caller turns into a SKIP. Nothing here is required for the release gate.
    """
    if not hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        return "this ComfyUI has no ModelPatcherDynamic"
    try:
        import comfy_aimdo.control  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return "comfy_aimdo unavailable: {}".format(exc)
    if getattr(comfy.model_management, "torch_version_numeric", (0, 0)) < (2, 8):
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


def _build_load_forward(
    comfy,
    loader,
    spec,
    *,
    dev: torch.device,
    load_mode: str,
    forward: bool,
) -> Dict[str, Any]:
    """One scenario: build, optionally load, optionally one residual GEMM.

    Returns a payload with ``build`` / ``load`` / ``forward`` / ``memory`` /
    ``errors``; it never raises, so a scenario can fail without taking the
    process (or another scenario) with it.
    """
    mm = comfy.model_management
    payload: Dict[str, Any] = {
        "build": {}, "load": {}, "forward": {}, "memory": {}, "errors": [], "patcher": None,
    }
    payload["memory"]["cuda_before_build"] = cuda_stats(dev)

    try:
        patcher, build_report = loader.load_raven_model_with_report(spec)
    except Exception as exc:  # noqa: BLE001 - the probe is the evidence
        payload["errors"].append("build failed: {}: {}".format(type(exc).__name__, exc))
        payload["memory"]["peak_rss_bytes"] = peak_host_rss_bytes()
        return payload

    payload["patcher"] = patcher
    payload["build"] = build_report.to_dict()
    payload["build"]["is_core_patcher"] = isinstance(patcher, comfy.model_patcher.ModelPatcher)
    payload["build"]["lora_param_devices_after_build"] = sorted(
        set(build_report.attachment.devices().values())
    ) if build_report.attachment else []
    payload["memory"]["cuda_after_build"] = cuda_stats(dev)

    if load_mode != "none":
        started = time.perf_counter()
        try:
            if load_mode == "auto":
                mm.load_models_gpu([patcher], memory_required=0)
            elif load_mode == "full":
                mm.load_models_gpu([patcher], force_full_load=True)
            elif load_mode == "partial":
                patcher.partially_load(dev, extra_memory=0)
            else:
                raise ValueError("unknown load mode {!r}".format(load_mode))
        except Exception as exc:  # noqa: BLE001
            payload["errors"].append("load failed: {}: {}".format(type(exc).__name__, exc))
        seconds = time.perf_counter() - started
        model = patcher.model
        payload["load"] = {
            "load_seconds": seconds,
            "loaded_size": int(patcher.loaded_size() or 0),
            "model_loaded_weight_memory": int(getattr(model, "model_loaded_weight_memory", 0) or 0),
            "model_lowvram": bool(getattr(model, "model_lowvram", False)),
            "lowvram_patch_counter": int(getattr(model, "lowvram_patch_counter", 0) or 0),
            "current_loaded_device": str(patcher.current_loaded_device()),
            "devices": device_histogram(model),
            "fully_loaded": int(patcher.loaded_size() or 0) >= int(patcher.model_size()),
        }

    payload["memory"]["cuda_after_load"] = cuda_stats(dev)
    payload["memory"]["peak_rss_bytes"] = peak_host_rss_bytes()
    if dev.type == "cuda":
        payload["memory"]["cuda_free_total"] = [int(x) for x in torch.cuda.mem_get_info(dev)]

    if forward:
        attachment = loader.get_raven_attachment(patcher)
        entry = attachment.entries[0] if attachment and attachment.entries else None
        if entry is None:
            payload["errors"].append("no RAVEN residual to forward through")
        else:
            module = entry.module
            weight = module.weight
            x = torch.randn(8, weight.shape[1], dtype=torch.float32, device=weight.device)
            cast = getattr(weight, "dtype", torch.float32)
            out = module(x.to(cast) if cast.is_floating_point else x)
            payload["forward"] = {
                "path": entry.path,
                "input": list(x.shape),
                "output": list(out.shape),
                "output_dtype": str(out.dtype),
                "hook_calls": entry.hook.calls,
                "hook_chunks": entry.hook.chunks,
                "device": str(out.device),
            }
    return payload


def run_probe(
    base_path: str,
    lora_path: str,
    *,
    comfy_path: Path = DEFAULT_COMFY,
    device: Optional[str] = None,
    offload_device: Optional[str] = None,
    strength: float = 1.0,
    weight_dtype: str = "default",
    load_mode: str = "auto",
    reserve_vram_gib: float = 0.0,
    force_lowvram: bool = False,
    forward: bool = False,
    force_static: bool = True,
    try_dynamic: bool = False,
) -> Report:
    comfy_path = resolve_comfy_path(str(comfy_path) if comfy_path else None)
    comfy = import_comfy(comfy_path)

    from raven_streaming import compat, loader  # imported after comfy is importable

    device = device or default_device()
    dev = torch.device(device)
    off = torch.device(offload_device) if offload_device else comfy.model_management.unet_offload_device()

    report = Report(
        comfy_path=str(comfy_path),
        comfy_commit=comfy_commit(comfy_path),
        comfy_version=compat.comfy_version(),
        device=str(dev),
        offload_device=str(off),
        load_mode=load_mode,
        force_lowvram=bool(force_lowvram),
        force_static=bool(force_static),
    )
    report.notes.append(compat.SUPPORT_NOTE)
    report.notes.append(
        "the release gate is the stock ModelPatcher partial CPU offload path "
        "(force_static={}); DynamicVRAM/aimdo is optional and unverified with this "
        "loader".format(bool(force_static))
    )
    report.notes.append(
        "host RAM: the real base is ~66 GB BF16 and the adapter ~5 GB FP32, so even "
        "--load-mode none needs the documented >= 128 GB baseline"
    )

    features = compat.check_features()
    report.features_ok = features.ok
    report.feature_failures = [c.line() for c in features.failures]
    if not features.ok:
        return report

    mm = comfy.model_management
    if force_lowvram:
        mm.vram_state = mm.VRAMState.LOW_VRAM
        report.notes.append(
            "comfy.model_management.vram_state forced to LOW_VRAM (public module state; "
            "no private DynamicVRAM internals are touched)"
        )
    report.vram_state = str(getattr(mm, "vram_state", "?"))

    reserve_tensor = None
    reserve_bytes = int(reserve_vram_gib * (1024 ** 3))
    report.reserve_vram_bytes = reserve_bytes
    if reserve_bytes > 0 and dev.type == "cuda":
        # a real allocation, so Comfy's free-memory query sees the pressure
        reserve_tensor = torch.empty(reserve_bytes, dtype=torch.uint8, device=dev)
        report.notes.append("reserved {} of VRAM with a live allocation".format(_gib(reserve_bytes)))
    elif reserve_bytes > 0:
        report.notes.append("--reserve-vram ignored on a non-CUDA device")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def make_spec(static: bool):
        return loader.RavenLoaderSpec(
            unet_path=base_path,
            lora_path=lora_path,
            strength=float(strength),
            model_options=dict(
                loader.model_options_for_weight_dtype(weight_dtype),
                load_device=dev,
                offload_device=off,
            ),
            force_static_patcher=bool(static),
        )

    # ---- the gate: stock ModelPatcher --------------------------------------
    payload = _build_load_forward(
        comfy, loader, make_spec(force_static), dev=dev, load_mode=load_mode, forward=forward
    )
    report.build = payload["build"]
    report.load = payload["load"]
    report.forward = payload["forward"]
    report.memory = payload["memory"]
    report.errors.extend(payload["errors"])
    patcher = payload["patcher"]

    # ---- optional, unverified: DynamicVRAM ---------------------------------
    report.dynamic = _dynamic_scenario(
        comfy, loader, make_spec(False), dev=dev, load_mode=load_mode,
        forward=forward, requested=try_dynamic, static_patcher=patcher,
    )

    del reserve_tensor
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _dynamic_scenario(
    comfy,
    loader,
    spec,
    *,
    dev: torch.device,
    load_mode: str,
    forward: bool,
    requested: bool,
    static_patcher: Any = None,
) -> Dict[str, Any]:
    """Optional ``ModelPatcherDynamic`` repeat; SKIPs never touch the gate."""
    if not requested:
        return {"status": "not requested",
                "reason": "pass --dynamic to try the optional DynamicVRAM path"}
    if dev.type != "cuda":
        return {"status": "skipped",
                "reason": "device={} is not CUDA; ModelPatcherDynamic.__new__ reroutes CPU "
                          "load devices back to ModelPatcher and comfy_aimdo needs a CUDA "
                          "index".format(dev)}
    reason = enable_dynamic_vram(comfy)
    if reason:
        return {"status": "skipped", "reason": reason}

    # free the static model first: two copies of the full DiT will not fit
    try:
        comfy.model_management.unload_all_models()
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": "unload_all_models failed: {}".format(exc)}
    if static_patcher is not None:
        try:
            static_patcher.detach()
        except Exception:  # noqa: BLE001 - best effort only
            pass
    gc.collect()
    torch.cuda.empty_cache()

    payload = _build_load_forward(
        comfy, loader, spec, dev=dev, load_mode=load_mode, forward=forward
    )
    out: Dict[str, Any] = {
        "status": "failed" if payload["errors"] else "ran",
        "reason": "; ".join(payload["errors"]),
        "core_model_patcher_after_rebind": getattr(
            comfy.model_patcher.CoreModelPatcher, "__name__", "?"),
        "build": payload["build"],
        "load": payload["load"],
        "forward": payload["forward"],
        "memory": payload["memory"],
        "errors": payload["errors"],
        "note": "optional and unverified: no release claim is made about this path",
    }
    return out


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.epilog = (
        "Release gate = the static scenario (stock ModelPatcher partial CPU offload). "
        "--dynamic is optional/unverified and can only SKIP or FAIL on its own. "
        "Real run sizes: ~66 GB BF16 base + ~5 GB FP32 adapter, held in host RAM even "
        "for --load-mode none, so the documented >= 128 GB host RAM baseline applies; "
        "--weight-dtype fp32 doubles the base to 132 GB+."
    )
    ap.add_argument("--base", required=True,
                    help="official full non-pruned BF16 MiniMax-H3 DiT (~66 GB; absolute "
                         "path or a name inside the diffusion_models folder)")
    ap.add_argument("--lora", required=True,
                    help="mandatory RAVEN PEFT LoRA (~5 GB FP32; absolute path or a name "
                         "in loras)")
    ap.add_argument("--comfy", default=None,
                    help="ComfyUI checkout (default: COMFYUI_PATH / COMFYUI_UPSTREAM_PATH / cache)")
    ap.add_argument("--device", default=default_device(), help="load device; env RAVEN_PROBE_DEVICE")
    ap.add_argument("--offload-device", default=None,
                    help="offload device (default: comfy.model_management.unet_offload_device())")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--weight-dtype", default="default", choices=("default", "bf16", "fp32"),
                    help="'default' lets comfy.model_management pick (BF16 for H3). 'fp32' "
                         "doubles the base weights to 132 GB+ on top of the FP32 residual - "
                         "debugging only")
    ap.add_argument("--load-mode", default="auto", choices=LOAD_MODES,
                    help="none = build only (valid M1 gate: build + optional load, no "
                         "sampling); auto = load_models_gpu; full = force_full_load; "
                         "partial = patcher.partially_load")
    ap.add_argument("--reserve-vram", type=float, default=0.0, metavar="GIB",
                    help="GiB of VRAM to allocate and hold for the whole probe (CUDA only). "
                         "It is a real allocation, so Comfy's free-memory query sees less "
                         "room and must partially offload: '--reserve-vram 24' takes 24 GiB "
                         "away from the model. Ignored on non-CUDA devices.")
    ap.add_argument("--force-lowvram", action="store_true",
                    help="set comfy.model_management.vram_state = LOW_VRAM before loading")
    ap.add_argument("--forward", action="store_true",
                    help="run one tiny GEMM through a single LoRA'd linear. Never a DiT "
                         "forward: at the official bidirectional geometry that is far more "
                         "expensive than the M1 build/load gate requires")
    ap.add_argument("--dynamic", action="store_true",
                    help="also try the OPTIONAL, UNVERIFIED DynamicVRAM/aimdo patcher "
                         "(reproduces main.py's enablement). Reported separately; a SKIP or "
                         "FAIL here never changes the static exit code")
    ap.add_argument("--allow-dynamic-default", action="store_true",
                    help="do NOT force the stock ModelPatcher for the main scenario, i.e. "
                         "use whatever CoreModelPatcher is bound to. Debug only; the "
                         "release path is the forced-static one")
    ap.add_argument("--json", default=None, help="write the report as JSON to this path")
    args = ap.parse_args(argv)

    report = run_probe(
        args.base,
        args.lora,
        comfy_path=resolve_comfy_path(args.comfy),
        device=args.device,
        offload_device=args.offload_device,
        strength=args.strength,
        weight_dtype=args.weight_dtype,
        load_mode=args.load_mode,
        reserve_vram_gib=args.reserve_vram,
        force_lowvram=args.force_lowvram,
        forward=args.forward,
        force_static=not args.allow_dynamic_default,
        try_dynamic=args.dynamic,
    )
    print(report.render())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
