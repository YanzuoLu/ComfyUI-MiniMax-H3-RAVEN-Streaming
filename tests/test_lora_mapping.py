"""Header parsing + strict PEFT -> diffusion_model.* mapping tests."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from raven_streaming import lora as rlora  # noqa: E402
from test_lora_common import (  # noqa: E402
    TOY_CONFIG,
    TOY_COUNTS,
    NoBigAllocation,
    full_scale_shapes,
    header_from_shapes,
    synthetic_lora_tensors,
    write_safetensors,
)

TOY_KW = dict(expected_counts=TOY_COUNTS)


# --------------------------------------------------------------------------
# safetensors header / tensor reader
# --------------------------------------------------------------------------
def test_header_roundtrip_and_tensor_read(tmp_path):
    tensors = {
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    path = write_safetensors(tmp_path / "t.safetensors", tensors, {"lora_alpha": "64"})

    header = rlora.read_safetensors_header(str(path))
    assert set(header.tensors) == {"a", "b"}
    assert header.metadata["lora_alpha"] == "64"
    assert header.tensors["a"].shape == (2, 3)
    assert header.tensors["a"].dtype == "F32"
    assert header.tensors["a"].nbytes == 24
    assert header.total_data_bytes() == 32

    loaded = rlora.load_tensors(
        str(path), list(header.tensors.values()), header.data_offset
    )
    assert torch.equal(loaded["a"], tensors["a"])
    assert torch.equal(loaded["b"], tensors["b"])


def test_header_only_read_does_not_touch_data(tmp_path):
    # a header describing far more data than the file actually contains is still
    # parseable: proof that no tensor bytes are read during inspection.
    header = {"x": {"dtype": "F32", "shape": [1 << 20, 1 << 10], "data_offsets": [0, 4 << 30]}}
    blob = json.dumps(header).encode("utf-8")
    path = tmp_path / "sparse.safetensors"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    parsed = rlora.read_safetensors_header(str(path))
    assert parsed.tensors["x"].nbytes == 4 << 30
    assert path.stat().st_size < 1024


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x00" * 4,
        struct.pack("<Q", 0),
        struct.pack("<Q", 1 << 60),
    ],
)
def test_bad_header_length_fails_loud(blob):
    with pytest.raises(rlora.SafetensorsFormatError):
        rlora.parse_safetensors_header(blob)


def test_bad_header_json_fails_loud():
    body = b"not json"
    with pytest.raises(rlora.SafetensorsFormatError):
        rlora.parse_safetensors_header(struct.pack("<Q", len(body)) + body)


def test_truncated_tensor_read_fails_loud(tmp_path):
    path = write_safetensors(
        tmp_path / "t.safetensors", {"a": torch.zeros(4, dtype=torch.float32)}
    )
    with open(path, "r+b") as fh:
        fh.truncate(path.stat().st_size - 4)
    header = rlora.read_safetensors_header(str(path))
    with pytest.raises(rlora.SafetensorsFormatError):
        rlora.load_tensors(str(path), list(header.tensors.values()), header.data_offset)


# --------------------------------------------------------------------------
# key parsing
# --------------------------------------------------------------------------
def test_parse_peft_key_variants():
    assert rlora.parse_peft_key(
        "base_model.model.dit.blocks.0.attn.qkv_proj.lora_A.weight"
    ) == ("blocks.0.attn.qkv_proj", "A", None)
    assert rlora.parse_peft_key(
        "base_model.model.dit.final_layer.adaln_proj.linear.lora_B.default.weight"
    ) == ("final_layer.adaln_proj.linear", "B", "default")


@pytest.mark.parametrize(
    "key",
    [
        "dit.blocks.0.attn.qkv_proj.lora_A.weight",  # wrong prefix
        "base_model.model.dit.blocks.0.attn.qkv_proj.weight",  # not a lora tensor
        "base_model.model.dit.blocks.0.attn.qkv_proj.lora_A.bias",
        "base_model.model.dit.blocks.0.attn.qkv_proj.lora_magnitude_vector.weight",
        "base_model.model.dit.blocks.0.attn.qkv_proj.lora_C.weight",
    ],
)
def test_parse_peft_key_rejects(key):
    with pytest.raises(rlora.UnexpectedKeyError):
        rlora.parse_peft_key(key)


# --------------------------------------------------------------------------
# inventory / published-scale mapping (metadata only)
# --------------------------------------------------------------------------
def test_official_inventory_counts():
    inv = rlora.RavenBaseConfig().modules()
    counts = rlora.category_counts(inv.values())
    assert counts == rlora.EXPECTED_CATEGORY_COUNTS
    assert len(inv) == rlora.EXPECTED_MODULE_COUNT == 266
    assert 2 * len(inv) == rlora.EXPECTED_TENSOR_COUNT == 532
    # spot-check the shapes that matter for "no QKV re-interleave / no fc1 swap"
    assert inv["blocks.0.attn.qkv_proj"].weight_shape == (21504, 5376)
    assert inv["blocks.0.attn.out_proj"].weight_shape == (5376, 7168)
    assert inv["blocks.0.mlp.fc1"].weight_shape == (28672, 5376)
    assert inv["blocks.0.mlp.fc2"].weight_shape == (5376, 14336)
    assert inv["blocks.0.adaln_proj.linear"].weight_shape == (96768, 2688)
    assert inv["final_layer.adaln_proj.linear"].weight_shape == (10752, 2688)
    assert inv["token_refiner.blocks.1.mlp.fc2"].category == "core"
    assert inv["time_embedder.proj_out"].category == "time"
    assert inv["final_layer.video_out"].category == "boundary"


def test_full_scale_manifest_maps_to_diffusion_model_keys():
    header = header_from_shapes(full_scale_shapes())
    assert len(header.tensors) == 532
    manifest = rlora.build_manifest(header)
    assert manifest.module_count == 266
    assert manifest.tensor_count == 532
    assert manifest.counts == {"core": 208, "adaln": 51, "time": 2, "boundary": 5}
    assert manifest.rank == 128
    assert manifest.alpha == 128.0
    keys = manifest.base_keys()
    assert len(keys) == 266
    assert all(k.startswith("diffusion_model.") and k.endswith(".weight") for k in keys)
    assert "diffusion_model.blocks.49.mlp.fc2.weight" in keys
    assert "diffusion_model.time_embedder.proj_in.weight" in keys
    # generic-format LoRA key that comfy.lora.model_lora_keys_unet builds
    assert all(k[: -len(".weight")].startswith("diffusion_model.") for k in keys)


def test_full_scale_mapping_never_allocates_dense_delta():
    shapes = full_scale_shapes()
    with NoBigAllocation(limit=1_000_000) as sentinel:
        header = header_from_shapes(shapes)
        manifest = rlora.build_manifest(header)
    assert manifest.module_count == 266
    # sanity: the merged qkv delta would have been 115M elements
    assert 21504 * 5376 > 1_000_000
    assert sentinel.max_seen == 0


# --------------------------------------------------------------------------
# strict failure modes (toy scale)
# --------------------------------------------------------------------------
def _toy_shapes(**kw):
    return full_scale_shapes(TOY_CONFIG, rank=kw.pop("rank", 4), **kw)


def test_toy_manifest_ok():
    manifest = rlora.build_manifest(header_from_shapes(_toy_shapes()), TOY_CONFIG, **TOY_KW)
    assert manifest.counts == TOY_COUNTS
    assert manifest.module_count == 22
    assert manifest.rank == 4


def test_unexpected_prefix_fails_loud():
    shapes = _toy_shapes()
    k = next(iter(shapes))
    shapes["transformer." + k] = shapes[k]
    with pytest.raises(rlora.UnexpectedKeyError, match="PEFT prefix|lora_A/lora_B"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_unknown_module_fails_loud():
    shapes = _toy_shapes()
    shapes[rlora.PEFT_PREFIX + "blocks.99.attn.qkv_proj.lora_A.weight"] = (4, 16)
    shapes[rlora.PEFT_PREFIX + "blocks.99.attn.qkv_proj.lora_B.weight"] = (16, 4)
    with pytest.raises(rlora.UnexpectedKeyError, match="do not exist"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_duplicate_tensor_fails_loud():
    shapes = _toy_shapes()
    shapes[rlora.PEFT_PREFIX + "blocks.0.mlp.fc1.lora_A.default.weight"] = shapes[
        rlora.PEFT_PREFIX + "blocks.0.mlp.fc1.lora_A.weight"
    ]
    with pytest.raises(rlora.DuplicateTensorError):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_missing_half_fails_loud():
    shapes = _toy_shapes()
    del shapes[rlora.PEFT_PREFIX + "blocks.1.mlp.fc2.lora_B.weight"]
    with pytest.raises(rlora.MissingCoverageError, match="only one of"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_missing_module_fails_loud():
    shapes = _toy_shapes()
    del shapes[rlora.PEFT_PREFIX + "time_embedder.proj_in.lora_A.weight"]
    del shapes[rlora.PEFT_PREFIX + "time_embedder.proj_in.lora_B.weight"]
    with pytest.raises(rlora.MissingCoverageError, match="not covered"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_category_count_mismatch_fails_loud():
    shapes = _toy_shapes()
    del shapes[rlora.PEFT_PREFIX + "time_embedder.proj_in.lora_A.weight"]
    del shapes[rlora.PEFT_PREFIX + "time_embedder.proj_in.lora_B.weight"]
    with pytest.raises(rlora.MissingCoverageError):
        rlora.build_manifest(
            header_from_shapes(shapes),
            TOY_CONFIG,
            require_full_coverage=False,
            expected_counts=TOY_COUNTS,
        )


def test_shape_mismatch_fails_loud():
    shapes = _toy_shapes()
    key = rlora.PEFT_PREFIX + "blocks.0.attn.qkv_proj.lora_B.weight"
    out, r = shapes[key]
    shapes[key] = (out + 1, r)
    with pytest.raises(rlora.ShapeMismatchError, match="refusing any QKV"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_transposed_pair_fails_loud():
    """A/B swapped (i.e. someone "helpfully" transposed the adapter) must fail."""
    shapes = _toy_shapes()
    a = rlora.PEFT_PREFIX + "blocks.0.mlp.fc1.lora_A.weight"
    b = rlora.PEFT_PREFIX + "blocks.0.mlp.fc1.lora_B.weight"
    shapes[a], shapes[b] = shapes[b], shapes[a]
    with pytest.raises(rlora.ShapeMismatchError):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_inconsistent_rank_fails_loud():
    shapes = _toy_shapes()
    a = rlora.PEFT_PREFIX + "blocks.1.attn.out_proj.lora_A.weight"
    b = rlora.PEFT_PREFIX + "blocks.1.attn.out_proj.lora_B.weight"
    shapes[a] = (8, shapes[a][1])
    shapes[b] = (shapes[b][0], 8)
    with pytest.raises(rlora.ShapeMismatchError, match="inconsistent LoRA rank"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_rank_mismatch_between_a_and_b_fails_loud():
    shapes = _toy_shapes()
    b = rlora.PEFT_PREFIX + "blocks.1.attn.out_proj.lora_B.weight"
    shapes[b] = (shapes[b][0], 8)
    with pytest.raises(rlora.ShapeMismatchError, match="rank"):
        rlora.build_manifest(header_from_shapes(shapes), TOY_CONFIG, **TOY_KW)


def test_non_fp32_dtype_fails_loud():
    header = header_from_shapes(_toy_shapes(), dtype="BF16")
    with pytest.raises(rlora.RavenLoraError, match="expected dtype"):
        rlora.build_manifest(header, TOY_CONFIG, **TOY_KW)
    # ...unless explicitly allowed
    manifest = rlora.build_manifest(
        header, TOY_CONFIG, allowed_dtypes=("F32", "BF16"), **TOY_KW
    )
    assert manifest.module_count == 22


# --------------------------------------------------------------------------
# alpha / strength resolution
# --------------------------------------------------------------------------
def test_alpha_default_and_sources():
    assert rlora.resolve_alpha({}) == 128.0
    assert rlora.resolve_alpha({"lora_alpha": "64"}) == 64.0
    assert rlora.resolve_alpha({"alpha": "32"}) == 32.0
    assert rlora.resolve_alpha({"lora_alpha": "64"}, override=16) == 16.0
    assert rlora.resolve_alpha({}, adapter_config={"lora_alpha": 256}) == 256.0
    with pytest.raises(rlora.RavenLoraError):
        rlora.resolve_alpha({"lora_alpha": "not-a-number"})


def test_manifest_from_file_uses_metadata_and_adapter_config(tmp_path):
    tensors = synthetic_lora_tensors(TOY_CONFIG, rank=4)
    path = write_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    m = rlora.manifest_from_file(str(path), TOY_CONFIG, **TOY_KW)
    assert m.alpha == 128.0  # default

    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 64, "r": 4}))
    m = rlora.manifest_from_file(str(path), TOY_CONFIG, **TOY_KW)
    assert m.alpha == 64.0

    path2 = write_safetensors(
        tmp_path / "meta.safetensors", tensors, metadata={"lora_alpha": "99"}
    )
    m = rlora.manifest_from_file(str(path2), TOY_CONFIG, **TOY_KW)
    assert m.alpha == 99.0  # safetensors metadata wins over adapter_config
    m = rlora.manifest_from_file(str(path2), TOY_CONFIG, alpha=8.0, **TOY_KW)
    assert m.alpha == 8.0  # explicit argument wins over everything


def test_load_lora_weights_roundtrip(tmp_path):
    tensors = synthetic_lora_tensors(TOY_CONFIG, rank=4)
    path = write_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    manifest = rlora.manifest_from_file(str(path), TOY_CONFIG, **TOY_KW)
    weights = rlora.load_lora_weights(str(path), manifest)
    assert set(weights) == set(manifest.modules)
    for p, (a, b) in weights.items():
        assert torch.equal(a, tensors[rlora.PEFT_PREFIX + p + ".lora_A.weight"])
        assert torch.equal(b, tensors[rlora.PEFT_PREFIX + p + ".lora_B.weight"])
        assert a.dtype == b.dtype == torch.float32


def test_named_adapter_keys_are_accepted(tmp_path):
    tensors = synthetic_lora_tensors(TOY_CONFIG, rank=4, adapter="default")
    path = write_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    manifest = rlora.manifest_from_file(str(path), TOY_CONFIG, **TOY_KW)
    assert manifest.adapter_names == ("default",)
    assert manifest.module_count == 22
