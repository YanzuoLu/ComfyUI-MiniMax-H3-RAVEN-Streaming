"""M1 loader lane: build order, guards, factory, schema - against tiny fakes.

Nothing here reads a real checkpoint: the fake ComfyUI modules in
``tests/test_loader_fakes.py`` provide a structurally identical (but tiny) H3
DiT, so the ordering and bookkeeping contract can be pinned in milliseconds.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_loader_fakes as fakes  # noqa: E402
from raven_streaming import compat  # noqa: E402
from raven_streaming import loader  # noqa: E402
from raven_streaming import lora as rlora  # noqa: E402
from raven_streaming import runtime_linear  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture
def fake_comfy(monkeypatch):
    """Fake ComfyUI in ``sys.modules`` + a fresh event log."""
    return fakes.install_fake_modules(monkeypatch)


@pytest.fixture
def mods(fake_comfy):
    return compat.require_features(compat.import_comfy_modules())


@pytest.fixture
def tiny_lora(tmp_path, fake_comfy):
    # depends on fake_comfy so the fake registries are cleared *before* the file
    # is registered, whatever order a test lists its fixtures in
    return fakes.write_tiny_lora(str(tmp_path / "raven_tiny.safetensors"))


@pytest.fixture
def tiny_spec(tmp_path, tiny_lora, fake_comfy):
    base = fakes.register_file(
        str(tmp_path / "h3_tiny.safetensors"), fakes.build_tiny_state_dict()
    )
    return loader.RavenLoaderSpec(
        unet_path=base,
        lora_path=tiny_lora,
        strength=1.0,
        model_options={"load_device": CPU, "offload_device": CPU},
    )


def _build(spec, mods, **kwargs):
    return loader.load_raven_model_with_report(spec, mods=mods, **kwargs)


# --------------------------------------------------------------------------
# 1. the order: weights -> LoRA -> patcher
# --------------------------------------------------------------------------
def test_lora_is_attached_after_weights_and_before_the_patcher(tiny_spec, mods, monkeypatch):
    original = rlora.attach_raven_lora

    def traced(*args, **kwargs):
        fakes.record("attach_raven_lora", kwargs.get("strength"))
        return original(*args, **kwargs)

    monkeypatch.setattr(rlora, "attach_raven_lora", traced)
    patcher, report = _build(tiny_spec, mods)

    names = fakes.event_names()
    assert names.index("load_model_weights") < names.index("attach_raven_lora")
    assert names.index("attach_raven_lora") < names.index("patcher_init")
    # the patcher's first size measurement happens after the patcher exists,
    # i.e. after the attach - never before
    assert names.index("patcher_init") < names.index("model_size")
    assert report.build_seconds > 0


def test_patcher_first_model_size_already_counts_the_lora(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    attachment = loader.get_raven_attachment(patcher)
    assert attachment is not None

    bare = fakes.fake_module_size(patcher.model) - attachment.parameter_bytes()
    assert report.model_size == patcher.size == fakes.fake_module_size(patcher.model)
    assert report.lora_bytes == attachment.parameter_bytes() > 0
    assert report.model_size - bare == report.lora_bytes


def test_attaching_to_the_built_patcher_would_now_be_refused(tiny_spec, mods):
    """The ordering is not merely preferred: the late path is a hard error."""
    patcher, report = _build(tiny_spec, mods)
    with pytest.raises(runtime_linear.RavenAttachError):
        rlora.attach_raven_lora(patcher, report.manifest, strength=1.0)


def test_base_weights_are_loaded_with_the_predicted_assign_flag(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    assign_events = [payload for name, payload in fakes.EVENTS if name == "load_model_weights"]
    assert assign_events == [report.assign_weights] == [patcher.is_dynamic()] == [False]


def test_dynamic_patcher_class_flips_assign_and_stays_consistent(tiny_spec, mods, monkeypatch):
    """Opting out of force_static: a dynamic patcher loads with ``assign=True``."""
    monkeypatch.setattr(
        mods.model_patcher, "CoreModelPatcher", fakes.FakeModelPatcherDynamic, raising=False
    )
    # a non-CPU load device, so ModelPatcherDynamic's CPU reroute does not fire;
    # the offload device stays CPU so no weights are actually moved
    spec = dataclasses.replace(
        tiny_spec,
        model_options={"load_device": torch.device("meta"), "offload_device": CPU},
        force_static_patcher=False,
    )
    patcher, report = _build(spec, mods)
    assert isinstance(patcher, fakes.FakeModelPatcherDynamic)
    assert report.assign_weights is True and report.is_dynamic is True
    assert report.force_static_patcher is False
    assert report.effective_disable_dynamic is False
    assert [p for n, p in fakes.EVENTS if n == "load_model_weights"] == [True]


def test_mismatched_dynamic_prediction_is_refused(tiny_spec, mods, monkeypatch):
    """If upstream's routing changed, loading must fail, not ship a wrong assign."""
    class Liar(fakes.FakeModelPatcher):
        def is_dynamic(self):  # predicted False (class rule), reports True
            return True

    monkeypatch.setattr(mods.model_patcher, "CoreModelPatcher", Liar, raising=False)
    # class-level prediction: Liar.is_dynamic(None) -> True, but CPU reroute is
    # not implemented on Liar, so prediction is False for a CPU load device
    monkeypatch.setattr(mods.model_management, "is_device_cpu", lambda d: True, raising=False)
    spec = dataclasses.replace(tiny_spec, force_static_patcher=False)
    with pytest.raises(loader.RavenLoaderError, match="predicted is_dynamic"):
        _build(spec, mods)


# --------------------------------------------------------------------------
# 1b. the v0.1 release contract: stock ModelPatcher, always
# --------------------------------------------------------------------------
@pytest.fixture
def dynamic_core(mods, monkeypatch):
    """Bind ``CoreModelPatcher`` to the dynamic class, as ``main.py`` does."""
    monkeypatch.setattr(
        mods.model_patcher, "CoreModelPatcher", fakes.FakeModelPatcherDynamic, raising=False
    )
    return mods


def _gpu_spec(spec, **kwargs):
    """Same spec on a non-CPU load device (no reroute), CPU offload (no moves)."""
    return dataclasses.replace(
        spec,
        model_options={"load_device": torch.device("meta"), "offload_device": CPU},
        **kwargs,
    )


def test_spec_forces_the_stock_patcher_by_default(tiny_spec, dynamic_core):
    patcher, report = _build(_gpu_spec(tiny_spec), dynamic_core)
    assert tiny_spec.force_static_patcher is True
    assert type(patcher) is fakes.FakeModelPatcher  # not the dynamic class
    assert report.force_static_patcher is True
    assert report.effective_disable_dynamic is True
    assert report.requested_disable_dynamic is False
    assert report.is_dynamic is False and report.assign_weights is False
    assert report.patcher_class == "FakeModelPatcher"


def test_upstream_disable_dynamic_can_only_be_stricter(tiny_spec, dynamic_core):
    # force_static=False + explicit disable_dynamic -> still static
    _patcher, report = _build(
        _gpu_spec(tiny_spec, force_static_patcher=False), dynamic_core, disable_dynamic=True
    )
    assert report.requested_disable_dynamic is True
    assert report.effective_disable_dynamic is True
    assert report.is_dynamic is False
    # force_static=True + explicit disable_dynamic -> still static, no conflict
    _patcher2, report2 = _build(_gpu_spec(tiny_spec), dynamic_core, disable_dynamic=True)
    assert report2.effective_disable_dynamic is True and report2.is_dynamic is False


def test_plain_factory_rebuild_never_becomes_dynamic(tiny_spec, dynamic_core):
    """``deepclone_multigpu`` calls the factory without ``disable_dynamic``."""
    patcher, report = _build(_gpu_spec(tiny_spec), dynamic_core)
    factory, args = patcher.cached_patcher_init
    assert args[0].force_static_patcher is True

    classes = [type(patcher)]
    current = patcher
    for _ in range(3):
        current = factory(*current.cached_patcher_init[1])  # no disable_dynamic
        classes.append(type(current))
        assert loader.get_raven_attachment(current) is not None
    assert set(classes) == {fakes.FakeModelPatcher}
    assert all(c.is_dynamic(None) is False for c in classes)
    # and the strict call form agrees
    strict = factory(*args, disable_dynamic=True)
    assert type(strict) is fakes.FakeModelPatcher
    assert report.model_size == strict.model_size()


def test_node_entry_point_forces_static(tmp_path, tiny_lora, dynamic_core):
    base = fakes.register_file(
        str(tmp_path / "h3_static.safetensors"), fakes.build_tiny_state_dict()
    )
    fakes.register_folder_file("diffusion_models", "h3.safetensors", base)
    fakes.register_folder_file("loras", "raven.safetensors", tiny_lora)
    patcher = loader.load_raven_diffusion_model(
        "h3.safetensors", "raven.safetensors",
        model_options={"load_device": torch.device("meta"), "offload_device": CPU},
        mods=dynamic_core,
    )
    assert type(patcher) is fakes.FakeModelPatcher
    assert patcher.cached_patcher_init[1][0].force_static_patcher is True


def test_programmatic_opt_out_is_possible_and_reported(tiny_spec, dynamic_core):
    patcher = loader.load_raven_model(
        _gpu_spec(tiny_spec, force_static_patcher=False), mods=dynamic_core
    )
    assert isinstance(patcher, fakes.FakeModelPatcherDynamic)
    # the opt-out travels with the spec, so rebuilds keep trying dynamic
    rebuilt = patcher.cached_patcher_init[0](*patcher.cached_patcher_init[1])
    assert isinstance(rebuilt, fakes.FakeModelPatcherDynamic)


def test_spec_and_report_expose_the_patcher_choice(tiny_spec, mods):
    _patcher, report = _build(tiny_spec, mods)
    payload = report.to_dict()
    assert payload["force_static_patcher"] is True
    assert payload["effective_disable_dynamic"] is True
    assert payload["requested_disable_dynamic"] is False
    assert payload["spec"]["force_static_patcher"] is True


def test_cached_patcher_init_signature_matches_upstream_call_forms(tiny_spec, mods):
    """Upstream calls ``factory(*args)`` and ``factory(*args, disable_dynamic=True)``."""
    import inspect

    patcher, _report = _build(tiny_spec, mods)
    init = patcher.cached_patcher_init
    assert isinstance(init, tuple) and len(init) == 2  # (factory, args), no index
    factory, args = init
    assert isinstance(args, tuple) and len(args) == 1
    assert isinstance(args[0], loader.RavenLoaderSpec)
    signature = inspect.signature(factory)
    bound = signature.bind(*args)
    assert list(bound.arguments) == ["spec"]
    signature.bind(*args, disable_dynamic=True)  # the non-dynamic delegate form
    assert signature.parameters["disable_dynamic"].default is False


# --------------------------------------------------------------------------
# 2. checkpoint guards
# --------------------------------------------------------------------------
def test_pruned_adaln_curve_checkpoint_is_refused(tmp_path, tiny_lora, mods):
    base = fakes.register_file(
        str(tmp_path / "h3_pruned.safetensors"), fakes.build_tiny_state_dict(pruned=True)
    )
    spec = loader.RavenLoaderSpec(
        unet_path=base, lora_path=tiny_lora,
        model_options={"load_device": CPU, "offload_device": CPU},
    )
    with pytest.raises(loader.PrunedCheckpointError, match="adaln_t_table"):
        _build(spec, mods)


def test_pruned_error_is_also_a_lora_lane_pruned_error(tmp_path, tiny_lora, mods):
    base = fakes.register_file(
        str(tmp_path / "h3_pruned2.safetensors"), fakes.build_tiny_state_dict(pruned=True)
    )
    spec = loader.RavenLoaderSpec(
        unet_path=base, lora_path=tiny_lora,
        model_options={"load_device": CPU, "offload_device": CPU},
    )
    with pytest.raises(rlora.PrunedBaseError):
        _build(spec, mods)


def test_missing_time_embedder_in_the_detected_config_is_refused():
    with pytest.raises(loader.PrunedCheckpointError, match="adaln_curve_grid"):
        loader.assert_full_nonpruned_unet_config(
            {"image_model": "minimax_h3", "adaln_curve_grid": 16, "time_embed_dim": 48}
        )
    with pytest.raises(loader.PrunedCheckpointError, match="time_embed_dim"):
        loader.assert_full_nonpruned_unet_config(
            {"image_model": "minimax_h3", "timestep_input_dim": 16, "time_embed_hidden_size": 64}
        )


def test_non_h3_checkpoint_is_refused(tmp_path, tiny_lora, mods):
    sd = fakes.build_tiny_state_dict()
    sd.pop("video_patch_proj.weight")
    base = fakes.register_file(str(tmp_path / "not_h3.safetensors"), sd)
    spec = loader.RavenLoaderSpec(
        unet_path=base, lora_path=tiny_lora,
        model_options={"load_device": CPU, "offload_device": CPU},
    )
    with pytest.raises(loader.UnsupportedCheckpointError):
        _build(spec, mods)


def test_quantised_checkpoints_are_refused(tiny_spec, mods, monkeypatch):
    original = fakes.fake_model_config_from_unet

    def quantised(*args, **kwargs):
        config = original(*args, **kwargs)
        config.quant_config = {"blocks.0.mlp.fc1": "int8"}
        return config

    monkeypatch.setattr(mods.model_detection, "model_config_from_unet", quantised)
    with pytest.raises(loader.UnsupportedCheckpointError, match="quant"):
        _build(tiny_spec, mods)


# --------------------------------------------------------------------------
# 3. cached factory / clone
# --------------------------------------------------------------------------
def test_cached_factory_rebuilds_the_same_model_with_the_lora(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    factory, args = patcher.cached_patcher_init
    assert factory is loader.rebuild_raven_patcher
    assert args == (tiny_spec.resolved(mods=mods),)

    rebuilt = factory(*args)
    rebuilt_attachment = loader.get_raven_attachment(rebuilt)
    assert rebuilt_attachment is not None
    assert rebuilt.model is not patcher.model  # a genuine rebuild
    assert len(rebuilt_attachment) == len(loader.get_raven_attachment(patcher))
    assert rebuilt_attachment.strength == report.lora_strength
    assert rebuilt.model_size() == report.model_size


def test_cached_factory_does_not_nest_across_rebuilds(tiny_spec, mods):
    patcher, _ = _build(tiny_spec, mods)
    factory, args = patcher.cached_patcher_init
    rebuilt = factory(*args)
    again = rebuilt.cached_patcher_init
    assert again[0] is factory and again[1] == args
    # no wrapper-of-wrapper: the factory is the module-level function itself
    assert getattr(factory, "raven_lora_wrapped_factory", None) is loader.load_raven_model
    assert rlora.has_lora_aware_patcher_factory(rebuilt)
    third = again[0](*again[1])
    assert third.cached_patcher_init[0] is factory


def test_cached_factory_honours_disable_dynamic(tiny_spec, mods):
    patcher, _ = _build(tiny_spec, mods)
    factory, args = patcher.cached_patcher_init
    non_dynamic = factory(*args, disable_dynamic=True)
    assert type(non_dynamic) is fakes.FakeModelPatcher
    assert loader.get_raven_attachment(non_dynamic) is not None


def test_cached_factory_keeps_the_injected_unet_model_cls(tiny_spec, mods):
    spec = dataclasses.replace(tiny_spec, unet_model_cls=fakes.FakeCausalMiniMaxH3Model)
    patcher, report = _build(spec, mods)
    assert report.unet_model_class == "FakeCausalMiniMaxH3Model"
    rebuilt = patcher.cached_patcher_init[0](*patcher.cached_patcher_init[1])
    assert isinstance(rebuilt.model.diffusion_model, fakes.FakeCausalMiniMaxH3Model)
    assert loader.get_raven_attachment(rebuilt) is not None


def test_clone_shares_the_model_and_therefore_the_residual(tiny_spec, mods):
    patcher, _ = _build(tiny_spec, mods)
    clone = patcher.clone()
    assert clone.model is patcher.model
    assert loader.get_raven_attachment(clone) is loader.get_raven_attachment(patcher)
    assert clone.cached_patcher_init == patcher.cached_patcher_init


def test_a_stronger_strength_survives_the_rebuild(tmp_path, tiny_lora, mods):
    base = fakes.register_file(
        str(tmp_path / "h3_strength.safetensors"), fakes.build_tiny_state_dict()
    )
    spec = loader.RavenLoaderSpec(
        unet_path=base, lora_path=tiny_lora, strength=0.75,
        model_options={"load_device": CPU, "offload_device": CPU},
    )
    patcher, _ = _build(spec, mods)
    assert loader.get_raven_attachment(patcher).strength == pytest.approx(0.75)
    rebuilt = patcher.cached_patcher_init[0](*patcher.cached_patcher_init[1])
    assert loader.get_raven_attachment(rebuilt).strength == pytest.approx(0.75)


# --------------------------------------------------------------------------
# 4. official generic-format LoRA still chains
# --------------------------------------------------------------------------
def test_official_key_map_still_reaches_every_raven_base_key(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    hits = loader.official_lora_key_hits(patcher, mods=mods)
    assert hits["total"] == report.lora_modules == len(report.manifest.modules)
    for category, expected in fakes.TINY_COUNTS.items():
        assert hits[category] == expected
    assert report.official_key_hits == hits


def test_raven_ab_parameters_are_not_reachable_under_a_lora_unet_name(tiny_spec, mods):
    """No published LoRA name can land on the residual parameters.

    Upstream also maps every non-``.weight`` ``diffusion_model.*`` key onto
    itself, so ``diffusion_model.<path>.raven_lora_A_0`` *is* in the key map
    under that verbatim name. That is upstream's generic escape hatch, not a
    name any published adapter uses; what matters is that none of the
    ``lora_unet_*`` / base-weight names resolve to a residual parameter.
    """
    patcher, _ = _build(tiny_spec, mods)
    key_map = loader.official_lora_key_map(patcher, mods=mods)
    assert not [k for k in key_map if k.startswith("lora_unet_") and "raven_lora" in k]
    assert not [k for k, v in key_map.items() if "raven_lora" in v and k != v]
    for key, value in key_map.items():
        if "raven_lora" in value:
            assert key == value  # verbatim generic form only


def test_a_synthetic_official_lora_patches_the_untouched_base_keys(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    key_map = loader.official_lora_key_map(patcher, mods=mods)
    state = patcher.model_state_dict()

    lora = {}
    targets = []
    for module in list(report.manifest.modules.values())[:6]:
        name = "lora_unet_" + module.path.replace(".", "_")
        assert key_map[name] == module.base_key
        weight = state[module.base_key]
        lora[name + ".lora_up.weight"] = torch.zeros(weight.shape[0], 2)
        lora[name + ".lora_down.weight"] = torch.randn(2, weight.shape[1]) * 0.01
        targets.append(module.base_key)

    patched, _clip = mods.sd.load_lora_for_models(patcher, None, lora, 1.0, 0.0)
    assert sorted(patched.patches) == sorted(targets)
    # the RAVEN residual rides along on the shared model
    assert loader.get_raven_attachment(patched) is loader.get_raven_attachment(patcher)


# --------------------------------------------------------------------------
# 5. path resolution
# --------------------------------------------------------------------------
def test_absolute_paths_bypass_folder_paths(tmp_path, tiny_lora, mods):
    assert loader.resolve_lora_path(tiny_lora, mods=mods) == tiny_lora
    with pytest.raises(loader.RavenLoaderError, match="no such file"):
        loader.resolve_diffusion_model_path(str(tmp_path / "nope.safetensors"), mods=mods)


def test_names_are_resolved_through_folder_paths(tmp_path, tiny_lora, mods):
    fakes.register_folder_file("loras", "raven.safetensors", tiny_lora)
    assert loader.resolve_lora_path("raven.safetensors", mods=mods) == tiny_lora
    assert fakes.fake_get_filename_list("loras") == ["raven.safetensors"]
    with pytest.raises(FileNotFoundError):
        loader.resolve_lora_path("missing.safetensors", mods=mods)


def test_names_resolve_end_to_end_through_the_node_entry_point(tmp_path, tiny_lora, mods):
    base = fakes.register_file(
        str(tmp_path / "h3_named.safetensors"), fakes.build_tiny_state_dict()
    )
    fakes.register_folder_file("diffusion_models", "h3.safetensors", base)
    fakes.register_folder_file("loras", "raven.safetensors", tiny_lora)
    patcher = loader.load_raven_diffusion_model(
        "h3.safetensors",
        "raven.safetensors",
        strength=1.0,
        model_options={"load_device": CPU, "offload_device": CPU},
        mods=mods,
    )
    assert loader.get_raven_attachment(patcher) is not None
    assert patcher.cached_patcher_init[1][0].unet_path == base


def test_missing_folder_paths_only_breaks_name_lookups(tiny_lora):
    bare = compat.ComfyModules()
    assert loader.resolve_lora_path(tiny_lora, mods=bare) == tiny_lora
    with pytest.raises(loader.RavenLoaderError, match="folder_paths"):
        loader.resolve_lora_path("raven.safetensors", mods=bare)


def test_weight_dtype_choices(mods):
    assert loader.model_options_for_weight_dtype("default") == {}
    assert loader.model_options_for_weight_dtype("bf16") == {"dtype": torch.bfloat16}
    assert loader.model_options_for_weight_dtype("fp32") == {"dtype": torch.float32}
    with pytest.raises(loader.RavenLoaderError):
        loader.model_options_for_weight_dtype("fp8_e4m3fn")


def test_weight_dtype_reaches_the_model(tmp_path, tiny_lora, mods):
    base = fakes.register_file(
        str(tmp_path / "h3_fp32.safetensors"), fakes.build_tiny_state_dict()
    )
    fakes.register_folder_file("diffusion_models", "h3.safetensors", base)
    fakes.register_folder_file("loras", "raven.safetensors", tiny_lora)
    patcher = loader.load_raven_diffusion_model(
        "h3.safetensors", "raven.safetensors", weight_dtype="fp32",
        model_options={"load_device": CPU, "offload_device": CPU}, mods=mods,
    )
    assert patcher.model.get_dtype() is torch.float32


# --------------------------------------------------------------------------
# 6. M2 injection point
# --------------------------------------------------------------------------
def test_unet_model_cls_is_injected_without_skipping_the_official_init(tiny_spec, mods):
    spec = dataclasses.replace(tiny_spec, unet_model_cls=fakes.FakeCausalMiniMaxH3Model)
    patcher, report = _build(spec, mods)
    model = patcher.model
    assert isinstance(model, fakes.FakeMiniMaxH3BaseModel)  # official BaseModel subclass
    assert isinstance(model.diffusion_model, fakes.FakeCausalMiniMaxH3Model)
    assert model.model_type is fakes.FakeModelType.FLOW_AV  # set by the official __init__
    assert report.model_class == "RavenFakeMiniMaxH3BaseModel"
    assert loader.get_raven_attachment(patcher) is not None
    # the shim sits between the H3 BaseModel and BaseModel in the MRO
    mro = type(model).__mro__
    assert mro.index(fakes.FakeMiniMaxH3BaseModel) < mro.index(fakes.FakeBaseModel)


def test_injected_class_is_cached(mods):
    first = loader.make_unet_injected_model_class(
        fakes.FakeMiniMaxH3BaseModel, fakes.FakeBaseModel, fakes.FakeCausalMiniMaxH3Model
    )
    second = loader.make_unet_injected_model_class(
        fakes.FakeMiniMaxH3BaseModel, fakes.FakeBaseModel, fakes.FakeCausalMiniMaxH3Model
    )
    assert first is second


def test_base_model_factory_replaces_the_whole_base_model(tiny_spec, mods):
    seen = {}

    def factory(model_config, state_dict, device=None):
        seen["config"] = model_config
        seen["keys"] = len(state_dict)
        return fakes.FakeMiniMaxH3BaseModel(model_config, device=device)

    spec = dataclasses.replace(tiny_spec, base_model_factory=factory)
    patcher, report = _build(spec, mods)
    assert seen["config"].unet_config["image_model"] == "minimax_h3"
    assert seen["keys"] > 0
    assert loader.get_raven_attachment(patcher) is not None
    assert "get_model" not in fakes.event_names()  # the official factory was bypassed


def test_a_factory_returning_the_wrong_thing_fails_loud(tiny_spec, mods):
    spec = dataclasses.replace(tiny_spec, base_model_factory=lambda *a, **k: torch.nn.Linear(2, 2))
    with pytest.raises(loader.RavenLoaderError, match="BaseModel"):
        _build(spec, mods)


# --------------------------------------------------------------------------
# 7. inventory derivation and node schema
# --------------------------------------------------------------------------
def test_lora_inventory_is_derived_from_the_detected_config(tiny_spec, mods):
    _patcher, report = _build(tiny_spec, mods)
    assert report.lora_category_counts == fakes.TINY_COUNTS
    assert report.official_topology is False  # tiny model, not the published one
    assert report.lora_rank == fakes.TINY_RANK
    assert report.lora_alpha == fakes.TINY_ALPHA


def test_published_topology_keeps_the_208_51_2_5_layout():
    published = rlora.RavenBaseConfig()
    assert loader.expected_category_counts(published) == rlora.EXPECTED_CATEGORY_COUNTS
    assert loader.expected_category_counts(published)["core"] == 208
    derived = loader.raven_config_from_unet_config(
        {
            "hidden_size": 5376, "num_layers": 50, "token_refiner_num_layers": 2,
            "num_attention_heads": 56, "attention_head_dim": 128, "ffn_hidden_size": 14336,
            "latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120,
            "timestep_input_dim": 256, "time_embed_hidden_size": 5376, "time_embed_dim": 2688,
        }
    )
    assert derived == published


def test_derived_config_follows_the_checkpoint(mods):
    derived = loader.raven_config_from_unet_config(
        {"image_model": "minimax_h3", "hidden_size": 64, "num_layers": 2,
         "token_refiner_num_layers": 1, "num_attention_heads": 4, "attention_head_dim": 16,
         "ffn_hidden_size": 48, "latents_dim": 4, "audio_latents_dim": 6, "text_dim": 32,
         "timestep_input_dim": 16, "time_embed_hidden_size": 64, "time_embed_dim": 48}
    )
    assert derived == fakes.TINY_CONFIG
    assert loader.expected_category_counts(derived) == fakes.TINY_COUNTS


def test_report_schema_is_serialisable(tiny_spec, mods):
    _patcher, report = _build(tiny_spec, mods)
    payload = report.to_dict()
    import json

    json.loads(json.dumps(payload, default=str))
    for key in (
        "patcher_class", "model_class", "unet_model_class", "unet_dtype", "manual_cast_dtype",
        "load_device", "offload_device", "parameters", "base_key_count", "left_over_keys",
        "assign_weights", "is_dynamic", "model_size", "lora_bytes", "lora_modules",
        "lora_rank", "lora_alpha", "lora_strength", "lora_category_counts",
        "official_key_hits", "official_topology", "build_seconds", "spec",
    ):
        assert key in payload, key
    assert payload["spec"]["unet_path"] == tiny_spec.unet_path
    assert payload["left_over_keys"] == []
    assert payload["patcher_class"] == "FakeModelPatcher"


def test_node_input_schema_is_frozen():
    schema = loader.loader_input_schema()
    assert schema["return_types"] == ["MODEL"]
    names = [i["name"] for i in schema["inputs"]]
    assert names == ["unet_name", "lora_name", "strength", "weight_dtype"]
    by_name = {i["name"]: i for i in schema["inputs"]}
    assert by_name["unet_name"]["folder"] == "diffusion_models"
    assert by_name["lora_name"]["folder"] == "loras"
    assert by_name["lora_name"]["required"] is True  # RAVEN is mandatory
    assert by_name["strength"]["default"] == 1.0
    assert tuple(by_name["weight_dtype"]["choices"]) == loader.WEIGHT_DTYPE_CHOICES
    assert "fp8_e4m3fn" not in by_name["weight_dtype"]["choices"]
    # the fp32 memory risk is stated where the user picks it
    assert "132 GB" in by_name["weight_dtype"]["tooltip"]
    # the patcher choice is a product decision, not a workflow toggle
    assert "force_static_patcher" not in names
    assert not [i for i in schema["inputs"] if "static" in i["name"]]


# --------------------------------------------------------------------------
# 8. offload-friendliness of the attached parameters
# --------------------------------------------------------------------------
def test_ab_parameters_are_direct_params_of_the_base_leaves(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    module = patcher.model.get_submodule("diffusion_model.blocks.0.mlp.fc1")
    names = {n for n, _ in module.named_parameters(recurse=False)}
    assert names == {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"}
    assert not list(module.named_children())  # still a leaf: stays in _load_list
    a = module.raven_lora_A_0
    assert a.dtype is torch.float32 and a.requires_grad is False
    assert report.lora_bytes == sum(
        p.numel() * p.element_size()
        for n, p in patcher.model.named_parameters()
        if "raven_lora_" in n
    )


def test_partial_load_moves_the_residual_with_the_base_weights(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    mods.model_management.load_models_gpu([patcher])
    assert patcher.loaded_size() == report.model_size
    devices = set(loader.get_raven_attachment(patcher).devices().values())
    assert devices == {"cpu"}


def test_state_dict_carries_the_residual_but_not_as_weight_keys(tiny_spec, mods):
    patcher, report = _build(tiny_spec, mods)
    sd = patcher.model_state_dict()
    for module in report.manifest.modules.values():
        assert module.base_key in sd
    residual_keys = [k for k in sd if "raven_lora_" in k]
    assert len(residual_keys) == 2 * report.lora_modules
    assert not [k for k in residual_keys if k.endswith(".weight")]
