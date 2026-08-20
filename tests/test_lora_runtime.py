"""Activation-side residual semantics, key stability and attach guards."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from raven_streaming import lora as rlora  # noqa: E402
from raven_streaming import runtime_linear as rrl  # noqa: E402
from test_lora_common import (  # noqa: E402
    TOY_CONFIG,
    TOY_COUNTS,
    ChunkAllocationSentinel,
    NoBigAllocation,
    ToyWrapper,
    build_toy_dit,
    make_pruned_dit,
    peft_reference_forward,
    synthetic_lora_tensors,
    synthetic_weight_pairs,
    write_safetensors,
)

TOY_KW = dict(expected_counts=TOY_COUNTS)
PATH = "blocks.0.mlp.fc1"


def _manifest(rank=4):
    from test_lora_common import full_scale_shapes, header_from_shapes

    return rlora.build_manifest(
        header_from_shapes(full_scale_shapes(TOY_CONFIG, rank=rank)), TOY_CONFIG, **TOY_KW
    )


def _attach(root, pairs, strength=1.0, alpha=128.0, rank=4, name="lora"):
    plan = []
    inv = TOY_CONFIG.modules()
    for path, (a, b) in pairs.items():
        plan.append(
            rrl.ResidualPlan(
                path=path,
                module=root.get_submodule(path),
                a=a,
                b=b,
                alpha=alpha,
                rank=rank,
                strength=strength,
                base_key=inv[path].base_key,
            )
        )
    return rrl.attach_residuals(plan, name=name)


# --------------------------------------------------------------------------
# exact PEFT semantics
# --------------------------------------------------------------------------
def test_fp32_residual_matches_peft_reference_exactly():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    att = _attach(root, pairs, strength=1.0, alpha=128.0, rank=4)
    assert len(att) == 22

    x = torch.randn(5, TOY_CONFIG.hidden_size)
    a, b = pairs[PATH]
    scaling = 128.0 / 4.0
    expected = peft_reference_forward(reference.get_submodule(PATH)(x), x, [(a, b)], [scaling])
    got = root.get_submodule(PATH)(x)
    assert torch.equal(got, expected)
    assert att.call_counts()[PATH] == 1


def test_strength_scales_the_residual():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    att = _attach(root, pairs, strength=0.25, alpha=128.0, rank=4)
    assert att.strength == 0.25

    x = torch.randn(3, TOY_CONFIG.hidden_size)
    a, b = pairs[PATH]
    base = reference.get_submodule(PATH)(x)
    expected = peft_reference_forward(base, x, [(a, b)], [0.25 * 128.0 / 4.0])
    assert torch.equal(root.get_submodule(PATH)(x), expected)

    att.set_strength(1.0)
    expected = peft_reference_forward(base, x, [(a, b)], [128.0 / 4.0])
    assert torch.equal(root.get_submodule(PATH)(x), expected)

    att.set_strength(0.0)
    assert torch.equal(root.get_submodule(PATH)(x), base)


def test_alpha_over_rank_scaling():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs(rank=4)
    reference = copy.deepcopy(root)
    _attach(root, pairs, strength=1.0, alpha=16.0, rank=4)
    x = torch.randn(2, TOY_CONFIG.hidden_size)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x), x, [pairs[PATH]], [16.0 / 4.0]
    )
    assert torch.equal(root.get_submodule(PATH)(x), expected)


def test_multiple_residuals_follow_registration_order_and_peft_accumulation():
    root = build_toy_dit()
    p1 = synthetic_weight_pairs(seed=1)
    p2 = synthetic_weight_pairs(seed=2)
    reference = copy.deepcopy(root)

    a1 = _attach(root, p1, strength=0.5, alpha=128.0, rank=4, name="first")
    a2 = _attach(root, p2, strength=2.0, alpha=64.0, rank=4, name="second")

    module = root.get_submodule(PATH)
    specs = rrl.module_residual_specs(module)
    assert [s.name for s in specs] == ["first", "second"]
    assert [s.a_param for s in specs] == ["raven_lora_A_0", "raven_lora_A_1"]

    x = torch.randn(4, TOY_CONFIG.hidden_size)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x),
        x,
        [p1[PATH], p2[PATH]],
        [0.5 * 128.0 / 4.0, 2.0 * 64.0 / 4.0],
    )
    assert torch.equal(module(x), expected)

    # one forward hook per module, shared by both adapters
    assert module._forward_hooks and len(module._forward_hooks) == 1
    assert a1.call_counts()[PATH] == a2.call_counts()[PATH]

    # detaching the first leaves the second intact and correctly scaled
    a1.detach()
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x), x, [p2[PATH]], [2.0 * 64.0 / 4.0]
    )
    assert torch.equal(module(x), expected)
    assert not hasattr(module, "raven_lora_A_0")
    assert hasattr(module, "raven_lora_A_1")


def test_bf16_base_output_casts_like_peft():
    root = build_toy_dit(dtype=torch.bfloat16)
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    _attach(root, pairs, strength=1.0, alpha=128.0, rank=4)

    x = torch.randn(6, TOY_CONFIG.hidden_size, dtype=torch.bfloat16)
    base = reference.get_submodule(PATH)(x)
    assert base.dtype == torch.bfloat16
    a, b = pairs[PATH]
    expected = peft_reference_forward(base, x, [(a, b)], [128.0 / 4.0])
    got = root.get_submodule(PATH)(x)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)
    # and it is *not* the same as rounding the residual to bf16 before adding
    naive = base + (
        torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a), b) * (128.0 / 4.0)
    ).to(torch.bfloat16)
    assert torch.equal(got, expected) and not torch.equal(naive, expected)


def test_residual_uses_fp32_math_even_for_bf16_ab():
    root = build_toy_dit()
    a = torch.randn(4, TOY_CONFIG.hidden_size).to(torch.bfloat16).float()
    b = torch.randn(TOY_CONFIG.ffn_hidden_size * 2, 4).to(torch.bfloat16).float()
    module = root.get_submodule(PATH)
    rrl.attach_residual(module, a, b, path=PATH, alpha=128.0, rank=4)
    x = torch.randn(3, TOY_CONFIG.hidden_size)
    out = module(x)
    assert out.dtype == torch.float32
    assert torch.equal(
        rrl.lora_residual(x, a, b, 32.0),
        torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 32.0,
    )


# --------------------------------------------------------------------------
# key / state-dict stability + official LoRA compatibility
# --------------------------------------------------------------------------
def test_base_state_dict_keys_and_values_unchanged():
    root = build_toy_dit()
    before = {k: v.clone() for k, v in root.state_dict().items()}
    att = _attach(root, synthetic_weight_pairs())
    after = root.state_dict()

    assert set(before).issubset(set(after))
    for k, v in before.items():
        assert torch.equal(after[k], v), k

    extra = sorted(set(after) - set(before))
    assert len(extra) == 2 * 22
    assert extra == sorted(att.parameter_names())
    assert all(("raven_lora_A_0" in k) or ("raven_lora_B_0" in k) for k in extra)
    # the new keys must not look like patchable weights to comfy.lora
    assert not any(k.endswith(".weight") or k.endswith(".bias") for k in extra)

    att.detach()
    assert set(root.state_dict()) == set(before)


def test_generic_lora_key_mapping_still_resolves():
    """Mirror of comfy.lora.model_lora_keys_unet over the wrapped model."""
    model = ToyWrapper(build_toy_dit())
    manifest = _manifest()
    att = rlora.attach_raven_lora(
        model, manifest, strength=1.0, weights=synthetic_weight_pairs()
    )

    sd = model.state_dict()
    key_map = {}
    for k in sd:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_map[k[: -len(".weight")]] = k
            else:
                key_map[k] = k

    for base_key in att.base_keys():
        assert base_key in sd
        assert key_map[base_key[: -len(".weight")]] == base_key
    # our parameters never shadow a base weight key
    for k in att.state_dict_keys():
        assert not k.endswith(".weight")
        assert k in sd


def test_official_weight_patch_and_raven_residual_stack():
    """LoraLoaderModelOnly patches .weight in place; the residual rides on top."""
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    att = _attach(root, pairs, strength=1.0, alpha=128.0, rank=4)
    module = root.get_submodule(PATH)

    x = torch.randn(4, TOY_CONFIG.hidden_size)
    before_patch = module(x)

    # emulate comfy patch_weight_to_device -> set_attr_param(model, key, out_weight)
    other_a = torch.randn(2, TOY_CONFIG.hidden_size) * 0.01
    other_b = torch.randn(TOY_CONFIG.ffn_hidden_size * 2, 2) * 0.01
    delta = (other_b @ other_a) * 0.5
    patched_weight = module.weight.data + delta
    module.weight = nn.Parameter(patched_weight.clone(), requires_grad=False)

    base_patched = torch.nn.functional.linear(x, patched_weight, module.bias)
    expected = peft_reference_forward(base_patched, x, [pairs[PATH]], [128.0 / 4.0])
    got = module(x)
    assert torch.equal(got, expected)
    assert not torch.equal(got, before_patch)
    # the lora params survived the weight replacement untouched
    assert torch.equal(getattr(module, "raven_lora_A_0"), pairs[PATH][0])
    assert att.call_counts()[PATH] == 2


def test_detach_restores_exact_base_behaviour():
    root = build_toy_dit()
    reference = copy.deepcopy(root)
    att = _attach(root, synthetic_weight_pairs())
    x = torch.randn(3, TOY_CONFIG.hidden_size)
    module = root.get_submodule(PATH)
    assert not torch.equal(module(x), reference.get_submodule(PATH)(x))

    att.detach()
    assert att.detached
    assert torch.equal(module(x), reference.get_submodule(PATH)(x))
    assert not module._forward_hooks
    assert not hasattr(module, rrl.HOOK_ATTR)
    assert not any("raven_lora" in n for n, _ in module.named_parameters())


# --------------------------------------------------------------------------
# structural requirements for ModelPatcher._load_list
# --------------------------------------------------------------------------
def test_lora_params_are_direct_and_module_stays_leaf():
    root = build_toy_dit()
    att = _attach(root, synthetic_weight_pairs())
    module = root.get_submodule(PATH)

    assert list(module.children()) == []
    direct = {n for n, _ in module.named_parameters(recurse=False)}
    recursive = {n for n, _ in module.named_parameters(recurse=True)}
    # this is exactly the ModelPatcher._load_list "not a leaf" test
    assert recursive == direct
    assert {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"} == direct
    assert att.parameter_numel() > 0


def test_module_size_accounting_matches_parameters():
    root = build_toy_dit()
    module = root.get_submodule(PATH)
    before = sum(p.numel() * p.element_size() for p in module.parameters())
    att = _attach(root, synthetic_weight_pairs())
    after = sum(p.numel() * p.element_size() for p in module.parameters())
    a, b = getattr(module, "raven_lora_A_0"), getattr(module, "raven_lora_B_0")
    assert after - before == (a.numel() + b.numel()) * 4
    assert att.parameter_bytes() == sum(
        (getattr(root.get_submodule(p), "raven_lora_A_0").numel()
         + getattr(root.get_submodule(p), "raven_lora_B_0").numel()) * 4
        for p in att.paths()
    )


def test_module_to_moves_lora_params_with_the_base_weight():
    root = build_toy_dit()
    _attach(root, synthetic_weight_pairs())
    module = root.get_submodule(PATH)
    module.to(torch.float64)  # stand-in for a device move on a CPU-only box
    assert getattr(module, "raven_lora_A_0").dtype == torch.float64
    assert module.weight.dtype == torch.float64
    x = torch.randn(2, TOY_CONFIG.hidden_size, dtype=torch.float64)
    assert module(x).dtype == torch.float64


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def test_pruned_base_is_rejected():
    root = make_pruned_dit()
    with pytest.raises(rlora.PrunedBaseError, match="adaln_t_table"):
        rlora.attach_raven_lora(root, _manifest(), weights=synthetic_weight_pairs())


def test_missing_time_embedder_is_rejected():
    root = build_toy_dit()
    del root._modules["time_embedder"]
    with pytest.raises(rlora.PrunedBaseError, match="time_embedder"):
        rlora.assert_base_not_pruned(root)


def test_base_shape_mismatch_is_rejected():
    root = build_toy_dit()
    parent = root.get_submodule("blocks.0.mlp")
    parent.fc1 = nn.Linear(TOY_CONFIG.hidden_size, 3)
    with pytest.raises(rlora.ShapeMismatchError):
        rlora.attach_raven_lora(root, _manifest(), weights=synthetic_weight_pairs())


def test_missing_base_module_is_rejected():
    root = build_toy_dit()
    del root.get_submodule("blocks.0.mlp")._modules["fc1"]
    with pytest.raises(rlora.MissingCoverageError, match="do not exist"):
        rlora.attach_raven_lora(root, _manifest(), weights=synthetic_weight_pairs())


def test_non_leaf_target_is_rejected():
    module = nn.Linear(8, 4)
    module.add_module("child", nn.Linear(4, 4))
    with pytest.raises(rrl.RavenAttachError, match="leaf"):
        rrl.attach_residual(
            module, torch.zeros(2, 8), torch.zeros(4, 2), path="x", alpha=1.0, rank=2
        )


def test_quantized_weight_is_rejected():
    class FakeQuantized(torch.Tensor):
        _layout_cls = "TensorWiseINT8Layout"

    module = nn.Linear(8, 4)
    module.weight = nn.Parameter(
        torch.zeros(4, 8).as_subclass(FakeQuantized), requires_grad=False
    )
    with pytest.raises(rrl.RavenAttachError, match="quantized"):
        rrl.attach_residual(
            module, torch.zeros(2, 8), torch.zeros(4, 2), path="x", alpha=1.0, rank=2
        )


def test_attach_rolls_back_on_failure():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    bad_path = "blocks.1.mlp.fc2"
    pairs[bad_path] = (torch.zeros(4, 3), pairs[bad_path][1])
    with pytest.raises(rrl.RavenAttachError):
        _attach(root, pairs)
    for path in TOY_CONFIG.modules():
        module = root.get_submodule(path)
        assert not any("raven_lora" in n for n, _ in module.named_parameters())
        assert not module._forward_hooks


def test_missing_or_extra_weights_are_rejected():
    root = build_toy_dit()
    manifest = _manifest()
    pairs = synthetic_weight_pairs()
    dropped = dict(pairs)
    dropped.pop(PATH)
    with pytest.raises(rlora.MissingCoverageError):
        rlora.attach_raven_lora(root, manifest, weights=dropped)

    extra = dict(pairs)
    extra["blocks.9.mlp.fc1"] = pairs[PATH]
    with pytest.raises(rlora.UnexpectedKeyError):
        rlora.attach_raven_lora(root, manifest, weights=extra)


def test_non_tensor_output_raises():
    class TupleLinear(nn.Linear):
        def forward(self, x):
            return (super().forward(x),)

    module = TupleLinear(8, 4)
    rrl.attach_residual(
        module, torch.zeros(2, 8), torch.zeros(4, 2), path="x", alpha=1.0, rank=2
    )
    with pytest.raises(rrl.RavenResidualError, match="plain tensor"):
        module(torch.zeros(1, 8))


# --------------------------------------------------------------------------
# end-to-end through a file
# --------------------------------------------------------------------------
def test_attach_from_file(tmp_path):
    tensors = synthetic_lora_tensors(TOY_CONFIG, rank=4)
    path = write_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    model = ToyWrapper(build_toy_dit())
    reference = copy.deepcopy(model)
    att = rlora.attach_raven_lora(
        model, str(path), strength=0.75, config=TOY_CONFIG, manifest_kwargs=TOY_KW
    )
    assert len(att) == 22
    assert att.strength == 0.75
    assert att.alpha == 128.0 and att.rank == 4

    x = torch.randn(3, TOY_CONFIG.hidden_size)
    full = "diffusion_model." + PATH
    module = model.get_submodule(full)
    a = tensors[rlora.PEFT_PREFIX + PATH + ".lora_A.weight"]
    b = tensors[rlora.PEFT_PREFIX + PATH + ".lora_B.weight"]
    expected = peft_reference_forward(
        reference.get_submodule(full)(x), x, [(a, b)], [0.75 * 128.0 / 4.0]
    )
    assert torch.equal(module(x), expected)


# --------------------------------------------------------------------------
# ModelPatcher attach-order guards (no ComfyUI needed: duck-typed patcher)
# --------------------------------------------------------------------------
class FakePatcher:
    """Minimal duck-type of comfy.model_patcher.ModelPatcher."""

    def __init__(self, model, size=0, loaded=0, lowvram=False, cached_patcher_init=None):
        self.model = model
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.size = size
        self.cached_patcher_init = cached_patcher_init
        model.model_loaded_weight_memory = loaded
        model.model_lowvram = lowvram

    def model_size(self):
        if self.size == 0:
            self.size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory


def test_attach_to_pristine_patcher_is_allowed():
    patcher = FakePatcher(ToyWrapper(build_toy_dit()))
    assert rlora.is_model_patcher(patcher)
    att = rlora.attach_raven_lora(patcher, _manifest(), weights=synthetic_weight_pairs())
    assert len(att) == 22


def test_attach_after_model_size_is_cached_fails_loud():
    patcher = FakePatcher(ToyWrapper(build_toy_dit()))
    patcher.model_size()  # what ModelPatcher.__init__/clone/load do internally
    with pytest.raises(rrl.RavenAttachError, match="model_size\\(\\) was already cached"):
        rlora.attach_raven_lora(patcher, _manifest(), weights=synthetic_weight_pairs())
    # and the guard must not "fix" it silently
    assert patcher.size > 0


@pytest.mark.parametrize("kwargs", [{"loaded": 4096}, {"lowvram": True}])
def test_attach_to_loaded_patcher_fails_loud(kwargs):
    patcher = FakePatcher(ToyWrapper(build_toy_dit()), **kwargs)
    with pytest.raises(rrl.RavenAttachError, match="already"):
        rlora.attach_raven_lora(patcher, _manifest(), weights=synthetic_weight_pairs())


def test_attach_with_cached_patcher_init_fails_loud():
    patcher = FakePatcher(ToyWrapper(build_toy_dit()), cached_patcher_init=(lambda: None, ()))
    with pytest.raises(rrl.RavenAttachError, match="cached_patcher_init"):
        rlora.attach_raven_lora(patcher, _manifest(), weights=synthetic_weight_pairs())
    # the factory is left untouched: clearing it would break multigpu/delegates
    assert patcher.cached_patcher_init is not None


def test_lora_aware_patcher_factory_reattaches_on_rebuild():
    weights = synthetic_weight_pairs()
    manifest = _manifest()

    def build_patcher(disable_dynamic=False):
        # stands in for a core loader: fresh weights straight from "disk"
        return [FakePatcher(ToyWrapper(build_toy_dit())), "other"]

    patcher = FakePatcher(ToyWrapper(build_toy_dit()), cached_patcher_init=(build_patcher, (), 0))

    def reattach(p):
        rlora.attach_raven_lora(p, manifest, weights=weights, allow_cached_patcher_init=True)

    rlora.register_lora_aware_patcher_factory(patcher, reattach)
    assert rlora.has_lora_aware_patcher_factory(patcher)
    # now the attach is allowed, because a rebuild re-attaches
    att = rlora.attach_raven_lora(patcher, manifest, weights=weights)
    assert len(att) == 22

    factory, args, index = patcher.cached_patcher_init
    rebuilt = factory(*args, disable_dynamic=True)[index]
    module = rebuilt.model.get_submodule("diffusion_model." + PATH)
    assert hasattr(module, "raven_lora_A_0")
    assert torch.equal(getattr(module, "raven_lora_A_0"), weights[PATH][0])


def test_plain_clone_sharing_the_model_keeps_the_lora():
    """A normal ModelPatcher.clone() shares .model, so nothing is lost."""
    model = ToyWrapper(build_toy_dit())
    weights = synthetic_weight_pairs()
    rlora.attach_raven_lora(model, _manifest(), weights=weights)
    clone = FakePatcher(model)  # same model object, as clone() does
    module = clone.model.get_submodule("diffusion_model." + PATH)
    assert hasattr(module, "raven_lora_A_0")
    assert clone.model_size() > 0


def test_alpha_override_does_not_mutate_a_shared_manifest():
    manifest = _manifest()
    root_a = build_toy_dit()
    root_b = build_toy_dit()
    weights = synthetic_weight_pairs()
    att_a = rlora.attach_raven_lora(root_a, manifest, weights=weights, alpha=32.0)
    assert manifest.alpha == 128.0  # untouched
    att_b = rlora.attach_raven_lora(root_b, manifest, weights=weights)
    assert att_a.alpha == 32.0 and att_b.alpha == 128.0


# --------------------------------------------------------------------------
# row chunking
# --------------------------------------------------------------------------
def test_row_chunk_is_derived_from_the_fp32_temp_budget():
    cfg = rlora.RavenBaseConfig()
    fc1 = cfg.modules()["blocks.0.mlp.fc1"]  # 28672 x 5376
    chunk = rrl.compute_row_chunk(fc1.in_features, fc1.out_features, 128)
    per_row = 4 * (fc1.in_features + 128 + 2 * fc1.out_features)
    assert chunk * per_row <= rrl.DEFAULT_TEMP_BUDGET_BYTES
    assert rrl.MIN_ROW_CHUNK <= chunk <= rrl.MAX_ROW_CHUNK
    # tiny modules are capped, not unbounded
    assert rrl.compute_row_chunk(32, 96, 128) == rrl.MAX_ROW_CHUNK
    # an explicit budget shrinks the chunk
    small = rrl.compute_row_chunk(
        fc1.in_features, fc1.out_features, 128, budget_bytes=1 << 20, minimum=1
    )
    assert small < chunk


@pytest.mark.parametrize("row_chunk", [1, 7, 64, 4096])
def test_chunked_residual_matches_peft_reference(row_chunk):
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    att = _attach(root, pairs, strength=0.8, alpha=128.0, rank=4)
    att.set_row_chunk(row_chunk)

    x = torch.randn(1000, TOY_CONFIG.hidden_size)
    module = root.get_submodule(PATH)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x), x, [pairs[PATH]], [0.8 * 128.0 / 4.0]
    )
    got = module(x)
    # Row tiling is mathematically a no-op (each output row depends only on its
    # own input row), but it changes the GEMM shape (M=row_chunk instead of
    # M=1000) and therefore which blocked kernel/accumulation order the backend
    # picks. The result is a handful of FP32 ULP, not a semantic difference, so
    # this is allclose rather than torch.equal.
    ulp = torch.finfo(torch.float32).eps * float(expected.detach().abs().max())
    assert float((got - expected).detach().abs().max()) <= 8 * ulp
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert att.chunk_counts()[PATH] == -(-1000 // row_chunk)


def test_chunking_preserves_leading_dims_and_is_exact_in_one_chunk():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    att = _attach(root, pairs)
    att.set_row_chunk(4096)

    x = torch.randn(2, 3, 5, TOY_CONFIG.hidden_size)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x), x, [pairs[PATH]], [128.0 / 4.0]
    )
    got = root.get_submodule(PATH)(x)
    assert got.shape == expected.shape == (2, 3, 5, TOY_CONFIG.ffn_hidden_size * 2)
    assert torch.equal(got, expected)
    assert att.chunk_counts()[PATH] == 1


def test_empty_row_batch_is_a_no_op():
    root = build_toy_dit()
    att = _attach(root, synthetic_weight_pairs())
    module = root.get_submodule(PATH)
    out = module(torch.zeros(0, TOY_CONFIG.hidden_size))
    assert out.shape == (0, TOY_CONFIG.ffn_hidden_size * 2)
    assert att.chunk_counts()[PATH] == 0


def test_chunked_bf16_matches_peft_reference():
    root = build_toy_dit(dtype=torch.bfloat16)
    pairs = synthetic_weight_pairs()
    reference = copy.deepcopy(root)
    att = _attach(root, pairs)
    att.set_row_chunk(37)

    x = torch.randn(300, TOY_CONFIG.hidden_size, dtype=torch.bfloat16)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x), x, [pairs[PATH]], [128.0 / 4.0]
    )
    got = root.get_submodule(PATH)(x)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_chunked_multi_adapter_accumulates_per_chunk_like_peft():
    root = build_toy_dit(dtype=torch.bfloat16)
    p1 = synthetic_weight_pairs(seed=1)
    p2 = synthetic_weight_pairs(seed=2)
    reference = copy.deepcopy(root)
    a1 = _attach(root, p1, strength=0.5, alpha=128.0, rank=4, name="first")
    _attach(root, p2, strength=2.0, alpha=64.0, rank=4, name="second")
    a1.set_row_chunk(13)  # one hook per module: both adapters share the chunking

    x = torch.randn(200, TOY_CONFIG.hidden_size, dtype=torch.bfloat16)
    expected = peft_reference_forward(
        reference.get_submodule(PATH)(x),
        x,
        [p1[PATH], p2[PATH]],
        [0.5 * 128.0 / 4.0, 2.0 * 64.0 / 4.0],
    )
    got = root.get_submodule(PATH)(x)
    # two adapters accumulated in FP32 per chunk, one cast back at the end; the
    # FP32 GEMM tiling difference can flip the last bf16 mantissa bit
    ulp = torch.finfo(torch.bfloat16).eps * float(expected.detach().abs().max())
    assert float((got - expected).detach().abs().max()) <= ulp
    assert (got != expected).float().mean() < 0.01


def test_inference_path_writes_into_the_base_output_in_place():
    """No second full-size buffer during inference: the base output is reused."""
    root = build_toy_dit(dtype=torch.bfloat16)
    pairs = synthetic_weight_pairs()
    att = _attach(root, pairs)
    att.set_row_chunk(16)
    module = root.get_submodule(PATH)
    hook = getattr(module, rrl.HOOK_ATTR)

    with torch.no_grad():
        x = torch.randn(50, TOY_CONFIG.hidden_size, dtype=torch.bfloat16)
        base_out = torch.nn.functional.linear(x, module.weight, module.bias)
        before = base_out.clone()
        returned = hook(module, (x,), base_out)
    assert returned is base_out  # in-place write-back
    assert not torch.equal(base_out, before)
    expected = peft_reference_forward(before, x, [pairs[PATH]], [128.0 / 4.0])
    assert torch.equal(returned, expected)


def test_grad_enabled_path_does_not_write_in_place():
    root = build_toy_dit()
    pairs = synthetic_weight_pairs()
    att = _attach(root, pairs)
    att.set_row_chunk(8)
    module = root.get_submodule(PATH)
    hook = getattr(module, rrl.HOOK_ATTR)

    x = torch.randn(20, TOY_CONFIG.hidden_size, requires_grad=True)
    base_out = torch.nn.functional.linear(x, module.weight, module.bias)
    out = hook(module, (x,), base_out)
    assert out is not base_out
    assert out.shape == base_out.shape
    out.sum().backward()
    assert x.grad is not None


def test_huge_streaming_shape_bounds_every_temporary():
    """S=60000 rows through fc1 (out=28672): only row-chunk sized temporaries."""
    cfg = rlora.RavenBaseConfig()
    entry = cfg.modules()["blocks.0.mlp.fc1"]
    meta = torch.device("meta")
    module = nn.Linear(entry.in_features, entry.out_features, bias=False, device=meta)
    rrl.attach_residual(
        module,
        torch.empty(128, entry.in_features, device=meta),
        torch.empty(entry.out_features, 128, device=meta),
        path="blocks.0.mlp.fc1",
        alpha=128.0,
        rank=128,
    )
    hook = getattr(module, rrl.HOOK_ATTR)
    rows = 60000
    chunk = hook.effective_row_chunk(entry.in_features, entry.out_features, hook.residuals)
    assert chunk <= rrl.MAX_ROW_CHUNK

    x = torch.empty(rows, entry.in_features, dtype=torch.bfloat16, device=meta)
    base_out = torch.empty(rows, entry.out_features, dtype=torch.bfloat16, device=meta)
    limit = chunk * entry.out_features  # the largest legal temporary
    dense = (entry.out_features, entry.in_features)
    with ChunkAllocationSentinel(limit, forbidden_shapes=[dense, dense[::-1]]) as sentinel:
        out = hook(module, (x,), base_out)
    assert out is base_out
    assert out.shape == (rows, entry.out_features)
    assert sentinel.max_alloc <= limit
    # the full-size FP32 promotion (60000 x 28672) is 60x over the limit
    assert rows * entry.out_features > 60 * limit
    assert hook.chunks == -(-rows // chunk)


def test_huge_streaming_shape_with_leading_dims():
    cfg = rlora.RavenBaseConfig()
    entry = cfg.modules()["blocks.0.attn.qkv_proj"]
    meta = torch.device("meta")
    module = nn.Linear(entry.in_features, entry.out_features, bias=False, device=meta)
    rrl.attach_residual(
        module,
        torch.empty(128, entry.in_features, device=meta),
        torch.empty(entry.out_features, 128, device=meta),
        path="blocks.0.attn.qkv_proj",
        alpha=128.0,
        rank=128,
        row_chunk=512,
    )
    hook = getattr(module, rrl.HOOK_ATTR)
    x = torch.empty(2, 30000, entry.in_features, dtype=torch.bfloat16, device=meta)
    base_out = torch.empty(2, 30000, entry.out_features, dtype=torch.bfloat16, device=meta)
    limit = 512 * entry.out_features
    with ChunkAllocationSentinel(
        limit, forbidden_shapes=[(entry.out_features, entry.in_features)]
    ) as sentinel:
        out = hook(module, (x,), base_out)
    assert out.shape == (2, 30000, entry.out_features)
    assert sentinel.max_alloc <= limit


def test_full_scale_forward_shapes_on_meta_without_dense_mm():
    cfg = rlora.RavenBaseConfig()
    entry = cfg.modules()["blocks.0.attn.qkv_proj"]
    meta = torch.device("meta")
    module = nn.Linear(entry.in_features, entry.out_features, bias=False, device=meta)
    a = torch.empty(128, entry.in_features, device=meta)
    b = torch.empty(entry.out_features, 128, device=meta)
    rrl.attach_residual(module, a, b, path="blocks.0.attn.qkv_proj", alpha=128.0, rank=128)

    tokens = 8
    x = torch.empty(tokens, entry.in_features, device=meta)
    limit = 4 * tokens * entry.out_features  # room for activations, not for B@A
    assert entry.out_features * entry.in_features > limit
    with NoBigAllocation(limit=limit) as sentinel:
        out = module(x)
    assert out.shape == (tokens, entry.out_features)
    assert out.device.type == "meta"
    assert sentinel.max_seen <= limit


def test_full_scale_attach_never_forms_dense_delta():
    cfg = rlora.RavenBaseConfig()
    inv = cfg.modules()
    meta = torch.device("meta")
    paths = ["blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1", "blocks.0.adaln_proj.linear"]
    # build the meta modules and A/B *outside* the sentinel: only the attach path
    # itself is under scrutiny (a merged B@A would be built there).
    built = []
    for path in paths:
        entry = inv[path]
        built.append(
            (
                path,
                entry,
                nn.Linear(entry.in_features, entry.out_features, bias=False, device=meta),
                torch.empty(128, entry.in_features, device=meta),
                torch.empty(entry.out_features, 128, device=meta),
            )
        )
    limit = 20_000_000  # > the largest A/B (96768x128), << any merged B@A
    assert all(e.out_features * e.in_features > limit for _, e, *_ in built)
    with NoBigAllocation(limit=limit) as sentinel:
        for path, entry, module, a, b in built:
            rrl.attach_residual(module, a, b, path=path, alpha=128.0, rank=128)
            assert getattr(module, "raven_lora_A_0").shape == (128, entry.in_features)
    assert sentinel.max_seen <= limit
