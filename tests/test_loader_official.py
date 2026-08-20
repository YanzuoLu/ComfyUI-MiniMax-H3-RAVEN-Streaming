"""M1 loader against the *real* upstream ComfyUI, on a tiny synthetic model.

Skipped unless a ComfyUI checkout is importable (``COMFYUI_PATH`` /
``COMFYUI_UPSTREAM_PATH`` / ``.cache/upstream/ComfyUI``).

No large weights are read: a structurally real but tiny
``comfy.ldm.minimax.model.MiniMaxH3Model`` is built in-process, saved as a
safetensors file, and loaded back through the RAVEN loader. That exercises the
genuine ``comfy.utils`` / ``comfy.model_detection`` / ``comfy.model_base`` /
``comfy.model_patcher`` chain, the real ``comfy.lora.model_lora_keys_unet`` and
the real ``comfy.sd.load_lora_for_models``.

The 208-core-key claim is checked against the *published* full-size inventory
too, by handing upstream's key-map function the published base keys (names
only, empty tensors) - no 66 GB model required.
"""

from __future__ import annotations

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
from raven_streaming import compat  # noqa: E402
from raven_streaming import loader  # noqa: E402
from raven_streaming import lora as rlora  # noqa: E402

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


CPU = torch.device("cpu")
TINY = fakes.TINY_CONFIG
TINY_KWARGS = dict(
    hidden_size=TINY.hidden_size,
    num_layers=TINY.num_layers,
    token_refiner_num_layers=TINY.token_refiner_num_layers,
    num_attention_heads=TINY.num_attention_heads,
    attention_head_dim=TINY.attention_head_dim,
    ffn_hidden_size=TINY.ffn_hidden_size,
    latents_dim=TINY.latents_dim,
    audio_latents_dim=TINY.audio_latents_dim,
    text_dim=TINY.text_dim,
    timestep_input_dim=TINY.timestep_input_dim,
    time_embed_hidden_size=TINY.time_embed_hidden_size,
    time_embed_dim=TINY.time_embed_dim,
    rope_inv_freq_len=fakes.ROPE_INV_FREQ_LEN,
)


@pytest.fixture(scope="module")
def comfy():
    path = find_upstream_comfyui()
    if path is None:
        pytest.skip("no ComfyUI checkout (set COMFYUI_PATH)")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        import comfy.ldm.minimax.model  # noqa: F401
        import comfy.lora  # noqa: F401
        import comfy.model_patcher  # noqa: F401
        import comfy.sd  # noqa: F401
        import comfy.utils  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - missing optional deps
        pytest.skip("cannot import ComfyUI: {}: {}".format(type(exc).__name__, exc))
    return sys.modules["comfy"]


@pytest.fixture(scope="module")
def mods(comfy):
    return compat.require_features(compat.import_comfy_modules())


@pytest.fixture(scope="module")
def tiny_checkpoint(comfy, tmp_path_factory):
    """A real (tiny) MiniMaxH3Model serialised as a real safetensors file."""
    tmp = tmp_path_factory.mktemp("raven_official")
    model = comfy.ldm.minimax.model.MiniMaxH3Model(
        **TINY_KWARGS, dtype=torch.bfloat16, device=CPU, operations=torch.nn
    )
    state = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    path = str(tmp / "h3_tiny.safetensors")
    comfy.utils.save_torch_file(state, path)
    del model
    return path


@pytest.fixture(scope="module")
def tiny_lora(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raven_official_lora")
    return fakes.write_tiny_lora(str(tmp / "raven_tiny.safetensors"))


@pytest.fixture
def spec(tiny_checkpoint, tiny_lora):
    return loader.RavenLoaderSpec(
        unet_path=tiny_checkpoint,
        lora_path=tiny_lora,
        strength=1.0,
        model_options={"load_device": CPU, "offload_device": CPU},
    )


@pytest.fixture
def built(spec, mods):
    return loader.load_raven_model_with_report(spec, mods=mods)


# --------------------------------------------------------------------------
def test_loader_returns_the_stock_model_patcher(built, comfy):
    """v0.1 gate: the stock ``ModelPatcher``, not whatever CoreModelPatcher is."""
    patcher, report = built
    assert type(patcher) is comfy.model_patcher.ModelPatcher
    assert patcher.is_dynamic() is False
    assert report.force_static_patcher is True
    assert report.effective_disable_dynamic is True
    assert report.assign_weights is False
    assert isinstance(patcher.model, comfy.model_base.MiniMaxH3)
    assert isinstance(patcher.model.diffusion_model, comfy.ldm.minimax.model.MiniMaxH3Model)
    assert report.patcher_class == type(patcher).__name__
    assert report.left_over_keys == []
    assert report.model_size == patcher.model_size() > report.lora_bytes > 0


def test_real_model_size_counts_the_residual(built, comfy):
    patcher, report = built
    attachment = loader.get_raven_attachment(patcher)
    without = comfy.model_management.module_size(patcher.model) - attachment.parameter_bytes()
    assert patcher.model_size() - without == attachment.parameter_bytes()
    assert report.lora_bytes == attachment.parameter_bytes()


def test_real_official_key_map_reaches_every_raven_base_key(built, comfy):
    patcher, report = built
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    for module in report.manifest.modules.values():
        name = "lora_unet_" + module.path.replace(".", "_")
        assert key_map.get(name) == module.base_key
    assert report.official_key_hits["total"] == len(report.manifest.modules)


def test_residual_parameters_have_no_lora_unet_name(built, comfy):
    """Upstream's non-``.weight`` branch exposes them verbatim - and only so.

    ``model_lora_keys_unet`` maps every ``diffusion_model.*`` key that does not
    end in ``.weight`` onto itself, so ``...raven_lora_A_0`` is present under
    exactly that name. No ``lora_unet_*`` / lycoris / diffusers name resolves to
    a residual parameter, which is what a published adapter would use.
    """
    patcher, _report = built
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    residual_targets = {k: v for k, v in key_map.items() if "raven_lora" in v}
    assert residual_targets, "expected upstream's verbatim generic entries"
    for key, value in residual_targets.items():
        assert key == value
    assert not [k for k in key_map if k.startswith("lora_unet_") and "raven_lora" in k]


def test_published_inventory_maps_208_core_keys(comfy):
    """The full-size claim, checked through the real upstream key-map function."""
    published = rlora.RavenBaseConfig()
    inventory = published.modules()
    assert loader.expected_category_counts(published)["core"] == 208

    class KeysOnly:
        # unet_to_diffusers() is a no-op without num_res_blocks, exactly as for
        # the real H3 config, so only the generic branch contributes keys
        model_config = type("Cfg", (), {"unet_config": {"image_model": "minimax_h3"}})()

        def state_dict(self):
            sd = {entry.base_key: torch.empty(0) for entry in inventory.values()}
            # norm keys ride along in a real model; they must not confuse us
            sd["diffusion_model.blocks.0.attn.q_norm.weight"] = torch.empty(0)
            return sd

    key_map = comfy.lora.model_lora_keys_unet(KeysOnly(), {})
    core = [e for e in inventory.values() if e.category == rlora.CATEGORY_CORE]
    assert len(core) == 208
    for entry in core:
        assert key_map["lora_unet_" + entry.path.replace(".", "_")] == entry.base_key
    lora_unet_names = [k for k in key_map if k.startswith("lora_unet_")]
    assert len(lora_unet_names) == len(inventory) + 1  # + the q_norm weight


def test_official_lora_loader_still_patches_after_raven(built, comfy):
    patcher, report = built
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    state = patcher.model_state_dict()

    lora = {}
    targets = []
    for module in list(report.manifest.modules.values())[:5]:
        name = "lora_unet_" + module.path.replace(".", "_")
        weight = state[key_map[name]]
        lora[name + ".lora_up.weight"] = torch.zeros(weight.shape[0], 2)
        lora[name + ".lora_down.weight"] = torch.randn(2, weight.shape[1]) * 0.01
        targets.append(module.base_key)

    patched, _clip = comfy.sd.load_lora_for_models(patcher, None, lora, 1.0, 0.0)
    assert sorted(patched.patches) == sorted(targets)
    assert patched.model is patcher.model
    assert loader.get_raven_attachment(patched) is loader.get_raven_attachment(patcher)


def test_real_cached_factory_rebuilds_with_the_lora(built, spec, mods):
    patcher, report = built
    factory, args = patcher.cached_patcher_init
    assert factory is loader.rebuild_raven_patcher
    rebuilt = factory(*args)
    assert rebuilt.model is not patcher.model
    attachment = loader.get_raven_attachment(rebuilt)
    assert attachment is not None and len(attachment) == report.lora_modules
    assert rebuilt.model_size() == report.model_size
    assert rebuilt.cached_patcher_init[0] is factory
    # the patcher class is stable across plain and strict rebuilds
    strict = factory(*args, disable_dynamic=True)
    assert type(rebuilt) is type(patcher) is type(strict)
    assert args[0].force_static_patcher is True


def test_force_static_picks_the_stock_class_even_when_core_is_dynamic(comfy, mods, monkeypatch):
    """The class choice happens before ``ModelPatcherDynamic.__new__``'s reroute."""
    if not hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        pytest.skip("this ComfyUI has no ModelPatcherDynamic")
    monkeypatch.setattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcherDynamic
    )
    assert loader.patcher_class_for(mods, disable_dynamic=True) is comfy.model_patcher.ModelPatcher
    assert (
        loader.patcher_class_for(mods, disable_dynamic=False)
        is comfy.model_patcher.ModelPatcherDynamic
    )


def test_real_build_under_a_dynamic_core_binding_stays_static(comfy, spec, mods, monkeypatch):
    if not hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        pytest.skip("this ComfyUI has no ModelPatcherDynamic")
    monkeypatch.setattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcherDynamic
    )
    patcher, report = loader.load_raven_model_with_report(spec, mods=mods)
    assert type(patcher) is comfy.model_patcher.ModelPatcher
    assert report.is_dynamic is False and report.assign_weights is False
    assert loader.get_raven_attachment(patcher) is not None


def test_real_partial_load_keeps_the_residual_with_its_module(built, comfy):
    patcher, report = built
    comfy.model_management.load_models_gpu([patcher], memory_required=0)
    module = patcher.model.get_submodule("diffusion_model.blocks.0.mlp.fc1")
    names = {n for n, _ in module.named_parameters(recurse=False)}
    expected = {"weight", "raven_lora_A_0", "raven_lora_B_0"}
    if module.bias is not None:
        expected.add("bias")
    assert names == expected
    assert not list(module.named_children())  # still a leaf for _load_list
    assert module.raven_lora_A_0.device == module.weight.device
    comfy.model_management.unload_all_models()


def test_the_residual_hook_runs_on_a_real_comfy_ops_linear(built):
    patcher, _report = built
    attachment = loader.get_raven_attachment(patcher)
    entry = [e for e in attachment.entries if e.path == "blocks.0.mlp.fc1"][0]
    module = entry.module
    calls_before = entry.hook.calls
    x = torch.randn(4, module.weight.shape[1], dtype=module.weight.dtype)
    out = module(x)
    assert out.shape == (4, module.weight.shape[0])
    assert entry.hook.calls == calls_before + 1


def test_real_detection_rejects_the_pruned_form(comfy, spec, mods, tmp_path):
    state = comfy.utils.load_torch_file(spec.unet_path)
    state["adaln_t_table"] = torch.zeros(16, TINY.time_embed_dim, dtype=torch.float32)
    path = str(tmp_path / "h3_pruned.safetensors")
    comfy.utils.save_torch_file(state, path)
    pruned_spec = loader.RavenLoaderSpec(
        unet_path=path,
        lora_path=spec.lora_path,
        model_options={"load_device": CPU, "offload_device": CPU},
    )
    with pytest.raises(loader.PrunedCheckpointError):
        loader.load_raven_model_with_report(pruned_spec, mods=mods)


def test_unet_model_cls_injection_against_the_real_base_model(comfy, spec, mods):
    import dataclasses

    class CausalTinyH3(comfy.ldm.minimax.model.MiniMaxH3Model):
        """Stand-in for the M2 causal DiT."""

    patcher, report = loader.load_raven_model_with_report(
        dataclasses.replace(spec, unet_model_cls=CausalTinyH3), mods=mods
    )
    model = patcher.model
    assert isinstance(model, comfy.model_base.MiniMaxH3)
    assert isinstance(model.diffusion_model, CausalTinyH3)
    assert model.model_type is comfy.model_base.ModelType.FLOW_AV
    assert report.unet_model_class == "CausalTinyH3"
    assert loader.get_raven_attachment(patcher) is not None
    mro = type(model).__mro__
    assert mro.index(comfy.model_base.MiniMaxH3) < mro.index(comfy.model_base.BaseModel)
