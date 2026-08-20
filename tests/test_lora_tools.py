"""Tests for tools/inspect_raven_lora.py (header-only inspection)."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from raven_streaming import lora as rlora  # noqa: E402
from test_lora_common import full_scale_shapes  # noqa: E402


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "inspect_raven_lora", ROOT / "tools" / "inspect_raven_lora.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _write_header_only(path: Path, entries, dtype="F32", itemsize=4, metadata=None):
    """Write a safetensors file that has a full header but no tensor data."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, shape in entries.items():
        n = itemsize
        for d in shape:
            n *= int(d)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


@pytest.fixture(scope="module")
def full_lora(tmp_path_factory):
    path = tmp_path_factory.mktemp("lora") / "adapter_model.safetensors"
    return _write_header_only(path, full_scale_shapes(), metadata={"format": "pt"})


def test_inspect_full_file_is_header_only(full_lora):
    # a 5 GB adapter described by a <100 kB file: nothing but the header is read
    assert full_lora.stat().st_size < 200 * 1024
    diag = tool.diagnose(rlora.read_safetensors_header(str(full_lora)), rlora.RavenBaseConfig())
    assert diag["tensors"] == 532
    assert diag["modules_covered"] == diag["modules_expected"] == 266
    assert diag["coverage_by_category"] == {
        "core": {"covered": 208, "expected": 208},
        "adaln": {"covered": 51, "expected": 51},
        "time": {"covered": 2, "expected": 2},
        "boundary": {"covered": 5, "expected": 5},
    }
    assert diag["rank_histogram"] == {128: 266}
    assert diag["total_data_bytes"] > 4 * 1024**3
    assert not diag["unexpected_keys"] and not diag["missing_modules"]


def test_inspect_reports_problems_without_raising(tmp_path):
    shapes = dict(full_scale_shapes())
    shapes["transformer.blocks.0.mlp.fc1.lora_A.weight"] = (128, 5376)
    shapes[rlora.PEFT_PREFIX + "blocks.99.mlp.fc1.lora_A.weight"] = (128, 5376)
    del shapes[rlora.PEFT_PREFIX + "time_embedder.proj_in.lora_B.weight"]
    path = _write_header_only(tmp_path / "broken.safetensors", shapes)

    diag = tool.diagnose(rlora.read_safetensors_header(str(path)), rlora.RavenBaseConfig())
    assert diag["unexpected_keys"] == ["transformer.blocks.0.mlp.fc1.lora_A.weight"]
    assert diag["unknown_modules"] == [rlora.PEFT_PREFIX + "blocks.99.mlp.fc1.lora_A.weight"]
    assert diag["half_pairs"] == ["time_embedder.proj_in"]
    assert diag["missing_modules"] == ["time_embedder.proj_in"]
    assert diag["coverage_by_category"]["time"] == {"covered": 1, "expected": 2}


def test_inspect_detects_pruned_base(tmp_path):
    cfg = rlora.RavenBaseConfig()
    entries = {
        "model.diffusion_model.{}.weight".format(p): e.weight_shape
        for p, e in cfg.modules().items()
        if not p.startswith("time_embedder")
    }
    entries["model.diffusion_model.adaln_t_table"] = (1024, cfg.time_embed_dim)
    path = _write_header_only(tmp_path / "base.safetensors", entries, dtype="BF16", itemsize=2)

    info = tool.check_base_file(
        str(path), list(cfg.modules()), ["", "diffusion_model.", "model.diffusion_model."]
    )
    assert info["pruned"] is True
    assert info["prefix"] == "model.diffusion_model."
    assert info["missing_base_key_count"] == 2


def test_inspect_cli_end_to_end(full_lora, tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = tool.main(["--file", str(full_lora), "--json", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "modules covered: 266/266" in printed
    assert "strict mapping: OK" in printed
    data = json.loads(out.read_text())
    assert data["manifest"]["counts"] == {"core": 208, "adaln": 51, "time": 2, "boundary": 5}
    assert data["manifest"]["alpha"] == 128.0
    assert data["strict_error"] is None


def test_inspect_cli_reports_failure_exit_code(tmp_path, capsys):
    shapes = dict(full_scale_shapes())
    del shapes[rlora.PEFT_PREFIX + "final_layer.video_out.lora_A.weight"]
    del shapes[rlora.PEFT_PREFIX + "final_layer.video_out.lora_B.weight"]
    path = _write_header_only(tmp_path / "partial.safetensors", shapes)
    rc = tool.main(["--file", str(path)])
    assert rc == 1
    printed = capsys.readouterr().out
    assert "MissingCoverageError" in printed
    assert "boundary  4/5" in printed


def test_hf_url_builder():
    assert tool.hf_url("org/name", "adapter_model.safetensors").endswith(
        "/org/name/resolve/main/adapter_model.safetensors"
    )
    assert "/resolve/abc123/" in tool.hf_url("org/name", "f.safetensors", "abc123")
