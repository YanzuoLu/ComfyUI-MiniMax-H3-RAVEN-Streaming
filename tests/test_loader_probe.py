"""The M1 loader probe, dry-run on a tiny real model (no big weights).

The real H200/H100 run (66 GB base + 5 GB adapter, VRAM reserve / lowvram) is a
separate, manual probe; this test only pins that ``tools/probe_model_loader.py``
runs, reports every field the milestone asks for, and does not sample anything.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_loader_fakes as fakes  # noqa: E402
from test_loader_official import TINY_KWARGS  # noqa: E402

try:
    from conftest import find_upstream_comfyui
except Exception:  # noqa: BLE001 - direct execution fallback

    def find_upstream_comfyui():
        for var in ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH"):
            value = os.environ.get(var)
            if value and (Path(value) / "comfy").is_dir():
                return Path(value)
        default = ROOT / ".cache" / "upstream" / "ComfyUI"
        return default if (default / "comfy").is_dir() else None


COMFY = find_upstream_comfyui()
pytestmark = pytest.mark.skipif(
    COMFY is None, reason="no ComfyUI checkout (set COMFYUI_PATH / COMFYUI_UPSTREAM_PATH)"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_model_loader", ROOT / "tools" / "probe_model_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


@pytest.fixture(scope="module")
def comfy(probe):
    try:
        return probe.import_comfy(Path(COMFY))
    except Exception as exc:  # noqa: BLE001 - missing optional deps
        pytest.skip("cannot import ComfyUI: {}: {}".format(type(exc).__name__, exc))


@pytest.fixture(scope="module")
def files(comfy, tmp_path_factory):
    import comfy.ldm.minimax.model  # noqa: F401
    import comfy.utils

    tmp = tmp_path_factory.mktemp("raven_probe")
    model = comfy.ldm.minimax.model.MiniMaxH3Model(
        **TINY_KWARGS, dtype=torch.bfloat16, device=torch.device("cpu"), operations=torch.nn
    )
    state = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    base = str(tmp / "h3_tiny.safetensors")
    comfy.utils.save_torch_file(state, base)
    del model
    lora = fakes.write_tiny_lora(str(tmp / "raven_tiny.safetensors"))
    return base, lora


def test_probe_builds_and_reports_every_milestone_field(probe, files):
    base, lora = files
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="auto")
    assert report.ok, report.render()
    payload = report.to_dict()
    json.loads(json.dumps(payload, default=str))

    build = payload["build"]
    assert build["build_seconds"] > 0
    assert build["patcher_class"] and build["model_class"] and build["unet_model_class"]
    assert build["model_size"] > build["lora_bytes"] > 0
    assert build["base_key_count"] > 0 and build["left_over_keys"] == []
    assert build["official_key_hits"]["total"] == build["lora_modules"]
    assert build["is_core_patcher"] is True
    assert build["lora_param_devices_after_build"] == ["cpu"]
    # the release gate is explicit in the JSON
    assert payload["force_static"] is True
    assert build["force_static_patcher"] is True
    assert build["effective_disable_dynamic"] is True
    assert build["is_dynamic"] is False
    assert build["spec"]["force_static_patcher"] is True

    load = payload["load"]
    assert load["load_seconds"] >= 0
    assert "cpu" in load["devices"]
    assert load["devices"]["cpu"]["lora_bytes"] == build["lora_bytes"]
    assert load["current_loaded_device"]
    assert payload["memory"]["peak_rss_bytes"] > 0
    assert payload["forward"] == {}  # nothing was sampled


def test_probe_default_is_the_static_release_path(probe, files, comfy, monkeypatch):
    """Even with CoreModelPatcher rebound, the gate scenario stays stock."""
    base, lora = files
    if hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        monkeypatch.setattr(
            comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcherDynamic
        )
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none")
    assert report.ok, report.render()
    assert report.force_static is True
    assert report.build["patcher_class"] == "ModelPatcher"
    assert report.build["is_dynamic"] is False
    assert "stock ModelPatcher" in report.render()


def test_dynamic_scenario_is_optional_and_never_gates(probe, files):
    base, lora = files
    # not requested: recorded, no effect
    default_report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                                     offload_device="cpu", load_mode="none")
    assert default_report.dynamic["status"] == "not requested"
    assert default_report.ok

    # requested on CPU: skipped with a reason, static verdict untouched
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none", try_dynamic=True)
    assert report.dynamic["status"] == "skipped"
    assert report.dynamic["reason"]
    assert report.ok and not report.errors
    rendered = report.render()
    assert "dynamic (optional, unverified): SKIPPED" in rendered
    assert "RESULT: ok" in rendered


def test_dynamic_enablement_helper_reports_why_it_cannot_run(probe, comfy):
    reason = probe.enable_dynamic_vram(comfy)
    # on this box it is expected to be unavailable; if it ever is available the
    # helper must have rebound CoreModelPatcher instead of returning a reason
    if reason:
        assert isinstance(reason, str) and reason
    else:
        assert comfy.model_patcher.CoreModelPatcher is comfy.model_patcher.ModelPatcherDynamic


def test_probe_can_build_without_loading(probe, files):
    base, lora = files
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none")
    assert report.ok, report.render()
    assert report.build and not report.load


def test_probe_forward_flag_runs_one_residual(probe, files):
    base, lora = files
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none", forward=True)
    assert report.ok, report.render()
    assert report.forward["hook_calls"] >= 1
    assert report.forward["path"]


def test_probe_reserve_and_lowvram_switches(probe, files, comfy, monkeypatch):
    base, lora = files
    monkeypatch.setattr(comfy.model_management, "vram_state",
                        comfy.model_management.vram_state, raising=False)
    report = probe.run_probe(base, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none",
                             reserve_vram_gib=1.0, force_lowvram=True)
    assert report.ok, report.render()
    assert report.force_lowvram is True
    assert report.reserve_vram_bytes == 1024 ** 3
    assert "LOW_VRAM" in report.vram_state
    assert any("reserve-vram ignored on a non-CUDA device" in n for n in report.notes)


def test_probe_reports_a_failed_build_instead_of_raising(probe, files, tmp_path):
    import comfy.utils

    base, lora = files
    state = comfy.utils.load_torch_file(base)
    state["adaln_t_table"] = torch.zeros(16, TINY_KWARGS["time_embed_dim"])
    pruned = str(tmp_path / "h3_pruned.safetensors")
    comfy.utils.save_torch_file(state, pruned)

    report = probe.run_probe(pruned, lora, comfy_path=Path(COMFY), device="cpu",
                             offload_device="cpu", load_mode="none")
    assert not report.ok
    assert any("PrunedCheckpointError" in e for e in report.errors)


def test_probe_cli_writes_json(probe, files, tmp_path, capsys):
    base, lora = files
    out = tmp_path / "report.json"
    code = probe.main([
        "--base", base, "--lora", lora, "--device", "cpu", "--offload-device", "cpu",
        "--load-mode", "none", "--json", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["ok"] and payload["build"]["model_size"] > 0
    printed = capsys.readouterr().out
    assert "RESULT: ok" in printed


def test_probe_documents_the_real_run_costs(probe):
    doc = probe.__doc__
    assert "66 GB" in doc and "5 GB" in doc and "128 GB" in doc
    assert "132 GB" in doc
    assert "optional and unverified" in doc


def test_probe_cli_help_explains_reserve_and_dynamic(probe, capsys):
    with pytest.raises(SystemExit):
        probe.main(["--help"])
    text = capsys.readouterr().out
    assert "--reserve-vram GIB" in text
    assert "GiB of VRAM to allocate and hold" in text
    assert "--dynamic" in text and "UNVERIFIED" in text
    assert "128 GB" in text and "66 GB" in text
    assert "132 GB" in text


def test_probe_helpers(probe):
    assert probe.peak_host_rss_bytes() > 0
    assert probe.LOAD_MODES == ("none", "auto", "full", "partial")
    linear = torch.nn.Linear(4, 4)
    linear.raven_lora_A_0 = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    hist = probe.device_histogram(linear)
    assert hist["cpu"]["lora_bytes"] == 32
    assert hist["cpu"]["base_bytes"] == (16 + 4) * 4
