"""The node's ComfyUI-facing seams, against a real checkout. Opt-in.

Every test here asks for the ``comfyui_on_syspath`` fixture and skips when no
checkout is available (``tests/conftest.py``). Nothing is faked: the point is to
find out whether the surfaces the node binds to still exist and still behave the
way ``COMPATIBILITY.md``'s pinned baseline says they do -- ``comfy.sd.VAE``,
``comfy.nested_tensor.NestedTensor``, ``comfy_execution.utils``,
``comfy.model_management``'s interrupt and
``comfy_extras.nodes_audio.vae_decode_audio``.

A failure here is an upstream drift report, not a bug in the node.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as layout_mod, nodes  # noqa: E402

# The official-shaped fakes live next door; pytest imports test modules under
# their bare names, so this is the same module object it collected.
from test_nodes_sampler import (  # noqa: E402
    FakePatcher,
    StubRollout,
    audio_vae,
    t2va_conditioning,
    video_vae,
)


def _import_or_skip(name: str):
    try:
        return __import__(name, fromlist=["_"])
    except Exception as exc:  # noqa: BLE001 - an incomplete checkout is not a failure
        pytest.skip(f"upstream {name} is not importable here: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# the VAE socket
# --------------------------------------------------------------------------


def test_the_vae_isinstance_gate_resolves_to_the_real_class(comfyui_on_syspath):
    comfy_sd = _import_or_skip("comfy.sd")
    assert nodes._comfy_vae_class() is comfy_sd.VAE

    # With the real class available, a stand-in is refused by type before any
    # feature probe runs -- which is what stops a duck-typed "VAE" from a
    # third-party pack reaching the streaming decode.
    with pytest.raises(nodes.NodeInputError, match="comfy.sd.VAE"):
        nodes.resolve_video_vae(video_vae())
    with pytest.raises(nodes.NodeInputError, match="comfy.sd.VAE"):
        nodes.resolve_audio_vae(audio_vae())


def test_the_h3_vae_geometry_this_node_checks_is_the_one_upstream_sets(comfyui_on_syspath):
    source = (comfyui_on_syspath / "comfy" / "sd.py").read_text(encoding="utf-8")
    video = source[source.index("# MiniMax H3 video VAE") : source.index("# MiniMax H3 audio VAE")]
    assert "self.latent_channels = 24" in video
    assert "self.latent_dim = 3" in video

    audio = source[source.index("# MiniMax H3 audio VAE") :][:2000]
    assert "self.latent_channels = 32" in audio
    assert "self.output_channels = 2" in audio
    assert "self.audio_sample_rate = 32000" in audio
    assert "self.upscale_ratio = 800" in audio

    # The streaming lane drives the inner modules directly and therefore skips
    # the wrapper's process_output. That is only safe while it is the identity
    # for both H3 VAEs, which is what these two lines say.
    assert "self.process_output = lambda image: image" in video
    assert "self.process_output = lambda audio: audio" in audio

    ldm_vae = _import_or_skip("comfy.ldm.minimax.vae")
    ldm_audio = _import_or_skip("comfy.ldm.minimax.audio_vae")
    assert hasattr(ldm_vae, nodes.VIDEO_VAE_INNER_CLASS)
    assert hasattr(ldm_audio, nodes.AUDIO_VAE_INNER_CLASS)
    inner = getattr(ldm_vae, nodes.VIDEO_VAE_INNER_CLASS)
    for attribute in ("_adaptive_decode", "blend", "_finalize_pixels"):
        assert callable(getattr(inner, attribute, None)), attribute


# --------------------------------------------------------------------------
# the final outputs
# --------------------------------------------------------------------------


def test_the_official_image_flatten_is_the_one_this_node_reproduces(comfyui_on_syspath):
    source = (comfyui_on_syspath / "nodes.py").read_text(encoding="utf-8")
    decode = source[source.index("class VAEDecode:") : source.index("class VAEDecodeTiled:")]
    assert "latent.is_nested" in decode
    assert "latent.unbind()[0]" in decode
    assert (
        "images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])" in decode
    )


def test_the_pinned_audio_helper_still_normalises_and_returns_a_waveform(comfyui_on_syspath):
    nodes_audio = _import_or_skip("comfy_extras.nodes_audio")
    assert nodes._default_audio_helper() is nodes_audio.vae_decode_audio

    nested = _import_or_skip("comfy.nested_tensor").NestedTensor
    samples = 4096
    latent = {
        "samples": nested(
            (torch.zeros(1, 24, 7, 4, 4), torch.zeros(1, 32, 2, samples // 800))
        )
    }

    class StubAudioVAE:
        audio_sample_rate = 32000

        def decode(self, z):
            # VAE.decode hands back [B, samples, channels]
            return torch.full((1, samples, 2), 8.0)

    out = nodes.decode_audio(StubAudioVAE(), latent)
    assert set(out) == {"waveform", "sample_rate"}
    assert out["sample_rate"] == 32000
    assert out["waveform"].shape == (1, 2, samples)
    # the helper owns the loudness normalisation; a constant signal has zero
    # std, so the floor at 1.0 leaves it untouched
    assert torch.allclose(out["waveform"], torch.full((1, 2, samples), 8.0))


# --------------------------------------------------------------------------
# execution context, interrupt, folders
# --------------------------------------------------------------------------


def test_the_interrupt_hook_is_the_thrower_the_sampler_expects(comfyui_on_syspath):
    model_management = _import_or_skip("comfy.model_management")
    cancel_check = nodes._resolve_cancel_check()
    assert cancel_check is model_management.throw_exception_if_processing_interrupted
    # a thrower, not a predicate: it returns None when nothing is interrupted,
    # which the sampler's cancel points must not read as "cancelled"
    assert cancel_check() is None


def test_the_unload_api_is_public_and_has_the_pinned_signature(comfyui_on_syspath):
    import inspect

    model_management = _import_or_skip("comfy.model_management")
    unload = getattr(model_management, "unload_model_and_clones", None)
    assert callable(unload), "the targeted unload disappeared from comfy.model_management"

    signature = inspect.signature(unload)
    parameters = list(signature.parameters)
    assert parameters[:3] == ["model", "unload_additional_models", "all_devices"]
    assert signature.parameters["unload_additional_models"].default is True
    assert signature.parameters["all_devices"].default is False
    assert callable(getattr(model_management, "soft_empty_cache", None))


def test_the_loader_this_node_drives_has_the_pinned_signature(comfyui_on_syspath):
    """Everything the phase swap rests on is one function's public signature."""
    import inspect

    model_management = _import_or_skip("comfy.model_management")
    loader = getattr(model_management, "load_models_gpu", None)
    assert callable(loader)

    parameters = inspect.signature(loader).parameters
    assert list(parameters)[0] == "models"
    for name in ("memory_required", "force_full_load", "minimum_memory_required"):
        assert name in parameters, name
    assert parameters["memory_required"].default == 0
    assert parameters["force_full_load"].default is False

    # the accounting the residency record reads back
    assert callable(getattr(model_management, "extra_reserved_memory", None))
    assert callable(getattr(model_management, "get_total_memory", None))
    assert callable(getattr(model_management, "get_free_memory", None))
    assert isinstance(model_management.NUM_STREAMS, int)


def test_memory_required_is_what_upstream_keeps_free(comfyui_on_syspath):
    """Why the node's ``memory_required`` is a workspace and never weights.

    ``load_models_gpu`` turns the argument into head-room it refuses to fill
    with weights (``lowvram_model_memory = current_free_mem -
    minimum_memory_required``). So every byte this node puts in it is a byte of
    model that gets streamed instead of resident -- which is correct for the
    tensors the rollout is about to allocate, and wrong for anything else.
    """
    source = (comfyui_on_syspath / "comfy" / "model_management.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def load_models_gpu") :]
    body = body[: body.index("\ndef ", 10)]
    assert "extra_mem = max(inference_memory, memory_required + extra_reserved_memory())" in body
    assert "lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required)" in body
    # and it frees only as much as the incoming models need, which is what
    # makes the phase swap a swap rather than a purge
    assert "free_memory(total_memory_required[device] * 1.1 + extra_mem" in body


def test_the_last_model_in_the_list_is_the_one_upstream_serves_first(comfyui_on_syspath):
    """``models.reverse()``: the coordinator puts the video VAE last on purpose."""
    source = (comfyui_on_syspath / "comfy" / "model_management.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def load_models_gpu") :]
    body = body[: body.index("\ndef ", 10)]
    assert "models.reverse()" in body
    # residency is decided per model, in that order, against the memory left
    assert "for loaded_model in models_to_load:" in body
    assert "current_free_mem = get_free_memory(torch_dev) + loaded_memory" in body

    import inspect

    order = inspect.getsource(nodes.PhaseSwapCoordinator.enter_vae_phase)
    assert "(self.audio, include_audio), (self.video, True)" in order


def test_the_residency_numbers_this_node_reads_are_real_patcher_api(comfyui_on_syspath):
    model_patcher = _import_or_skip("comfy.model_patcher")
    patcher = model_patcher.ModelPatcher(
        torch.nn.Linear(4, 4), load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    assert isinstance(patcher.model_size(), int)
    assert isinstance(patcher.loaded_size(), int)
    # set by ModelPatcher.load / partially_unload; the node only reads it
    assert hasattr(patcher.model, "model_offload_buffer_memory")

    assert nodes._model_size(patcher) == patcher.model_size()
    assert nodes._loaded_size(patcher) == patcher.loaded_size()
    assert nodes._offload_buffer_bytes(patcher) == patcher.model.model_offload_buffer_memory


def test_the_pinned_host_allocation_api_the_cpu_cache_needs_exists():
    """``kv_cache_storage='cpu_pinned'`` is a promise about host memory.

    The cache lane owns the allocation; what is checked here is that the API it
    is named after is the one torch actually has, so the widget's default is not
    describing something that cannot be done.
    """
    assert callable(getattr(torch.Tensor, "pin_memory", None))
    assert callable(getattr(torch.Tensor, "is_pinned", None))
    # torch.empty is a builtin: exercise the keyword rather than its signature
    tensor = torch.empty(4, dtype=torch.uint8, device="cpu", pin_memory=False)
    assert tensor.is_pinned() is False


def test_the_targeted_unload_keeps_everything_that_is_not_a_clone(comfyui_on_syspath):
    """The property the final-decode handover rests on, read off upstream.

    ``unload_model_and_clones`` builds a *keep* list of every loaded model whose
    ``clone_base_uuid`` differs from the given one (and from its nested
    additional models), then frees only what is left. That is what makes it
    safe to call with the two VAEs resident: they were loaded from different
    patchers, so they are kept.
    """
    source = (comfyui_on_syspath / "comfy" / "model_management.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def unload_model_and_clones") :]
    body = body[: body.index("\ndef ", 10)]
    assert "model.clone_base_uuid == loaded_model.model.clone_base_uuid" in body
    assert "get_nested_additional_models()" in body
    assert "keep_loaded.append(loaded_model)" in body
    assert "free_memory(1e30, get_torch_device(), keep_loaded)" in body

    # a clone -- what LoraLoaderModelOnly hands downstream -- carries the same
    # uuid, which is why passing the clone evicts the family
    model_patcher = _import_or_skip("comfy.model_patcher")
    patcher = model_patcher.ModelPatcher(
        torch.nn.Linear(2, 2), load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    assert patcher.clone().clone_base_uuid == patcher.clone_base_uuid


def test_the_handover_runs_against_the_real_model_management(comfyui_on_syspath):
    """No stub at the ComfyUI boundary: the real call, on a real patcher."""
    model_patcher = _import_or_skip("comfy.model_patcher")
    model_management = _import_or_skip("comfy.model_management")
    cpu = torch.device("cpu")
    patcher = model_patcher.ModelPatcher(
        torch.nn.Linear(2, 2), load_device=cpu, offload_device=cpu
    )

    loaded_before = list(model_management.current_loaded_models)
    handover = nodes.prepare_final_decode(patcher)

    assert handover.strategy == nodes.FINAL_DECODE_UNLOAD_STRATEGY
    assert handover.seconds >= 0.0
    # a model nobody loaded is a no-op, and nothing else was disturbed
    assert list(model_management.current_loaded_models) == loaded_before


def test_the_execution_context_supplies_the_prompt_and_node_id(comfyui_on_syspath):
    utils = _import_or_skip("comfy_execution.utils")
    with utils.CurrentNodeContext(prompt_id="prompt-1", node_id="41"):
        assert nodes.executing_identity(None) == ("41", "prompt-1")
        # the hidden unique_id wins when both are present; it is the one the
        # client addresses, including inside a subgraph
        assert nodes.executing_identity("12:41") == ("12:41", "prompt-1")
    assert nodes.executing_identity(None) == (None, None)


def test_the_client_id_helper_reads_the_prompt_server(comfyui_on_syspath):
    preview_server = _import_or_skip("raven_streaming.preview_server")
    server = SimpleNamespace(client_id="sid-9")
    assert preview_server.current_client_id(server) == "sid-9"
    # no server at all is not an error; it means "broadcast"
    assert preview_server.current_client_id(SimpleNamespace(client_id=None)) is None


def test_the_loader_combos_come_from_folder_paths(comfyui_on_syspath):
    folder_paths = _import_or_skip("folder_paths")
    assert callable(folder_paths.get_filename_list)
    for folder in (nodes.loader_mod.DIFFUSION_MODEL_FOLDER, nodes.loader_mod.LORA_FOLDER):
        assert isinstance(nodes._filename_list(folder), list)


# --------------------------------------------------------------------------
# a real NestedTensor through the node
# --------------------------------------------------------------------------


def test_an_official_nested_latent_survives_the_round_trip(comfyui_on_syspath, monkeypatch):
    """The node must work with the ``NestedTensor`` class it never imports."""
    nested_module = _import_or_skip("comfy.nested_tensor")
    frames, width, height = 39, 64, 64
    video = torch.zeros(
        [1, 24, layout_mod.video_latent_t(frames), height // 16, width // 16]
    )
    audio = torch.zeros([1, 32, 2, layout_mod.audio_latent_t(frames)])
    latent = {"samples": nested_module.NestedTensor((video, audio))}

    from raven_streaming import contracts

    request = contracts.parse_latent(latent)
    assert (request.frames, request.width, request.height) == (frames, width, height)

    out = contracts.build_output_latent(request, video, audio)
    assert isinstance(out["samples"], nested_module.NestedTensor)
    assert out["samples"].unbind()[0] is video

    # ... and the sampler node consumes exactly that structure
    monkeypatch.setattr(nodes, "_comfy_vae_class", lambda: None)
    monkeypatch.setattr(nodes.consistency, "sample_streaming", StubRollout(out))
    monkeypatch.setattr(
        nodes, "_default_load_models_gpu", lambda: (lambda *a, **k: None)
    )
    monkeypatch.setattr(
        nodes,
        "_default_audio_helper",
        lambda: (lambda vae, latent_in: {"waveform": torch.zeros(1, 2, 16), "sample_rate": 32000}),
    )
    result = nodes.RAVENStreamingSampler().sample(
        model=FakePatcher(),
        positive=t2va_conditioning(),
        latent=latent,
        video_vae=video_vae(),
        audio_vae=audio_vae(),
        seed=0,
        unique_id="7",
    )
    assert result[0] is out
    assert result[1].shape == (frames, height, width, 3)


# --------------------------------------------------------------------------
# registration, as upstream performs it
# --------------------------------------------------------------------------


def test_upstream_would_pick_up_this_package(comfyui_on_syspath):
    import raven_streaming

    loader_source = (comfyui_on_syspath / "nodes.py").read_text(encoding="utf-8")
    assert 'hasattr(module, "NODE_CLASS_MAPPINGS")' in loader_source
    assert 'hasattr(module, "WEB_DIRECTORY")' in loader_source

    # the three attribute reads upstream makes, in the order it makes them
    assert hasattr(raven_streaming, "WEB_DIRECTORY")
    assert getattr(raven_streaming, "WEB_DIRECTORY") is not None
    assert hasattr(raven_streaming, "NODE_CLASS_MAPPINGS")
    assert getattr(raven_streaming, "NODE_CLASS_MAPPINGS") is not None
    assert hasattr(raven_streaming, "NODE_DISPLAY_NAME_MAPPINGS")
    for name, node_cls in raven_streaming.NODE_CLASS_MAPPINGS.items():
        assert callable(getattr(node_cls, "INPUT_TYPES", None)), name
        assert isinstance(node_cls.RETURN_TYPES, tuple), name
        assert isinstance(node_cls.FUNCTION, str), name
        assert callable(getattr(node_cls, node_cls.FUNCTION, None)), name
        assert isinstance(node_cls.CATEGORY, str), name
