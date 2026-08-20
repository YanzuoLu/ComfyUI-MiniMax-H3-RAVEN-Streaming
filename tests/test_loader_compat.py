"""Feature detection for the loader lane: fakes always, real upstream if present.

``raven_streaming.compat`` is the gate the package opens on; these tests pin
what it probes, that a missing symbol is a loud failure, and that it makes no
unverified version claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_loader_fakes as fakes  # noqa: E402
from raven_streaming import compat  # noqa: E402

try:
    from conftest import find_upstream_comfyui
except Exception:  # noqa: BLE001 - direct execution fallback
    import os

    def find_upstream_comfyui():
        for var in ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH"):
            value = os.environ.get(var)
            if value and (Path(value) / "comfy").is_dir():
                return Path(value)
        default = ROOT / ".cache" / "upstream" / "ComfyUI"
        return default if (default / "comfy").is_dir() else None


@pytest.fixture
def fake_comfy(monkeypatch):
    return fakes.install_fake_modules(monkeypatch)


# --------------------------------------------------------------------------
# against the fakes
# --------------------------------------------------------------------------
def test_fake_modules_satisfy_every_required_feature(fake_comfy):
    report = compat.check_features()
    assert report.ok, report.render()
    assert not report.failures
    compat.require_features()


def test_report_is_serialisable_and_carries_the_pinned_baseline(fake_comfy):
    payload = compat.check_features().to_dict()
    json.loads(json.dumps(payload, default=str))
    assert payload["pinned_commit"] == compat.PINNED_COMFY_COMMIT
    assert payload["pinned_version"] == "0.33.0"
    assert payload["declared_min"] == "0.30.0"
    assert payload["checks"]


def test_a_renamed_symbol_is_a_loud_failure(fake_comfy, monkeypatch):
    monkeypatch.delattr(fake_comfy["comfy.model_detection"], "unet_prefix_from_state_dict")
    report = compat.check_features()
    assert not report.ok
    assert any("model_detection" in c.name for c in report.failures)
    with pytest.raises(compat.MissingFeatureError, match="unet_prefix_from_state_dict"):
        compat.require_features()


def test_a_missing_injection_point_is_a_loud_failure(fake_comfy, monkeypatch):
    class NoInjection:
        def __init__(self, model_config, model_type=None, device=None):
            pass

        load_model_weights = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(fake_comfy["comfy.model_base"], "BaseModel", NoInjection)
    report = compat.check_features()
    failing = [c.name for c in report.failures]
    assert "BaseModel(unet_model=...) injection point" in failing


def test_a_changed_h3_model_config_is_a_loud_failure(fake_comfy, monkeypatch):
    config_cls = fake_comfy["comfy.supported_models"].MiniMaxH3
    monkeypatch.setattr(config_cls, "unet_config", {"image_model": "something_else"})
    monkeypatch.setattr(config_cls, "sampling_settings", {"shift": 12.0})
    monkeypatch.setattr(config_cls, "supported_inference_dtypes", [])
    failing = [c.name for c in compat.check_features().failures]
    assert "MiniMaxH3.unet_config image_model == 'minimax_h3'" in failing
    assert "MiniMaxH3.sampling_settings carries shift + audio_shift" in failing
    assert "MiniMaxH3 supports bfloat16 inference (the published RAVEN base dtype)" in failing


def test_a_foreign_latent_format_is_a_loud_failure(fake_comfy, monkeypatch):
    class OtherLatent:
        latent_channels = 16

    monkeypatch.setattr(fake_comfy["comfy.supported_models"].MiniMaxH3, "latent_format", OtherLatent)
    failing = [c.name for c in compat.check_features().failures]
    assert "MiniMaxH3.latent_format is comfy.latent_formats.MiniMaxH3AV" in failing


def test_folder_paths_and_dynamic_vram_are_optional(fake_comfy, monkeypatch):
    # ``None`` in sys.modules makes importlib raise, which is what an
    # environment without folder_paths looks like (deleting the entry would let
    # a real checkout on sys.path be imported instead)
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    monkeypatch.delattr(fake_comfy["comfy.model_patcher"], "ModelPatcherDynamic")
    report = compat.check_features()
    assert report.ok, report.render()
    optional = [c.name for c in report.optional_missing]
    assert any("folder_paths" in name for name in optional)
    assert any("ModelPatcherDynamic" in name for name in optional)


def test_cached_patcher_init_slot_is_probed(fake_comfy, monkeypatch):
    class NoFactorySlot(fakes.FakeModelPatcher):
        def __init__(self, model, load_device=None, offload_device=None, size=0,
                     weight_inplace_update=False):
            super().__init__(model, load_device, offload_device, size, weight_inplace_update)

    # a subclass whose own __init__ does not mention the attribute still passes,
    # because the probe reads the class it is asked about
    monkeypatch.setattr(fake_comfy["comfy.model_patcher"], "ModelPatcher", NoFactorySlot)
    failing = [c.name for c in compat.check_features().failures]
    assert "ModelPatcher.cached_patcher_init factory slot" in failing


# --------------------------------------------------------------------------
# claims and hygiene
# --------------------------------------------------------------------------
def test_no_unverified_version_claim():
    note = compat.SUPPORT_NOTE
    assert compat.DECLARED_MIN_COMFYUI in note
    assert "not been verified" in note
    assert "0.33.0" in note
    # the declared lower bound must never be described as tested/verified/supported
    lowered = note.lower()
    for phrase in ("verified on 0.30", "supported on 0.30", "tested on 0.30"):
        assert phrase not in lowered


def _referenced_identifiers(source: str):
    """Identifiers, attributes and non-docstring literals used by the code."""
    import ast

    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                used.add(node.value)
    return used


def test_no_private_dynamic_vram_internals_are_touched():
    """Prose may name them; the code must never reach for them."""
    for name in ("compat.py", "loader.py"):
        source = (ROOT / "raven_streaming" / name).read_text(encoding="utf-8")
        used = _referenced_identifiers(source)
        for forbidden in ("dynamic_vbar", "dynamic_vbars", "_vbar_get", "_v",
                          "LowVramPatch", "dynamic_pins", "dynamic_patchers"):
            assert forbidden not in used, "{} references {}".format(name, forbidden)


def test_import_errors_are_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    for module_name in compat.MODULE_NAMES.values():
        monkeypatch.setitem(sys.modules, module_name, None)
    mods = compat.import_comfy_modules()
    assert mods.import_errors
    with pytest.raises(compat.MissingFeatureError):
        mods.require("model_patcher")


def test_overrides_bypass_imports():
    sentinel = object()
    mods = compat.import_comfy_modules({"utils": sentinel})
    assert mods.utils is sentinel
    assert mods.require("utils") is sentinel


# --------------------------------------------------------------------------
# against the real upstream, when a checkout is available
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_comfy():
    path = find_upstream_comfyui()
    if path is None:
        pytest.skip("no ComfyUI checkout (set COMFYUI_PATH)")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        import comfy.model_patcher  # noqa: F401
        import comfy.ldm.minimax.model  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - missing optional deps
        pytest.skip("cannot import ComfyUI: {}: {}".format(type(exc).__name__, exc))
    return sys.modules["comfy"]


def test_real_upstream_passes_the_feature_probe(real_comfy):
    report = compat.check_features()
    assert report.ok, report.render()
    assert report.comfy_version != "unknown"


def test_real_upstream_reports_the_pinned_version(real_comfy):
    version = compat.comfy_version()
    if version != compat.PINNED_COMFY_VERSION:
        pytest.skip(
            "checkout is ComfyUI {}, the audited baseline is {}".format(
                version, compat.PINNED_COMFY_VERSION
            )
        )
    assert compat.check_features().ok
