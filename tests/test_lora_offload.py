"""Run the model-patcher offload probe as a test (skipped without ComfyUI).

Environment: ``COMFYUI_PATH`` / ``COMFYUI_UPSTREAM_PATH`` locate the checkout
(same contract as ``tests/conftest.py``); ``RAVEN_PROBE_DEVICE`` selects the
device, so a GPU box runs the dynamic scenario by exporting
``RAVEN_PROBE_DEVICE=cuda``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

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


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_lora_offload", ROOT / "tools" / "probe_lora_offload.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


probe = _load_probe()
COMFY = find_upstream_comfyui()
DEVICE = os.environ.get("RAVEN_PROBE_DEVICE") or "cpu"

pytestmark = pytest.mark.skipif(
    COMFY is None, reason="no ComfyUI checkout (set COMFYUI_PATH / COMFYUI_UPSTREAM_PATH)"
)


@pytest.fixture(scope="module")
def report():
    try:
        probe.import_comfy(Path(COMFY))
    except Exception as exc:  # noqa: BLE001 - missing comfy deps in this env
        pytest.skip("cannot import ComfyUI: {}: {}".format(type(exc).__name__, exc))
    return probe.run_probe(Path(COMFY), DEVICE)


def test_probe_all_checks_pass(report):
    failures = [c.line() for c in report.checks if not c.ok and not c.skipped]
    assert not failures, "\n".join(failures)


def test_probe_covers_the_required_evidence(report):
    names = " | ".join(c.name for c in report.checks if c.scenario == "static")
    for fragment in (
        "model_size()",
        "_load_list()",
        "full load",
        "partial load",
        "partially_unload",
        "child-module attachment",
        "already-measured patcher fails loud",
    ):
        assert fragment in names, fragment


def test_probe_reports_the_patcher_classes(report):
    assert report.patcher_classes.get("static") == "ModelPatcher"
    assert report.patcher_classes.get("ModelPatcherDynamic available") == "True"
    assert "dynamic" in report.scenarios
    payload = report.to_dict()
    assert payload["patcher_classes"] and payload["scenarios"]
    assert all("scenario" in c and "status" in c for c in payload["checks"])


def test_dynamic_scenario_runs_or_says_why(report):
    dynamic = [c for c in report.checks if c.scenario == "dynamic"]
    assert dynamic, "the dynamic scenario must appear in the report"
    if DEVICE.startswith("cuda"):
        assert report.patcher_classes.get("dynamic") == "ModelPatcherDynamic"
        names = " | ".join(c.name for c in dynamic)
        for fragment in ("FP32 on the GPU", "loaded_size()", "round 2"):
            assert fragment in names, fragment
    else:
        assert all(c.skipped for c in dynamic)
        assert "skipped" in report.scenarios["dynamic"]


def test_probe_reports_the_pinned_commit(report):
    assert report.comfy_commit
    assert report.comfy_path.endswith("ComfyUI")
