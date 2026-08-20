"""What the two nodes actually do when they execute.

Still no ComfyUI, still nothing faked into ``sys.modules``. What is substituted
is substituted *by injection*, through the seams the node module already has:
the loader entry point, the causal DiT class, ``load_models_gpu``, the audio
decode helper and the preview installer. Everything else -- the contract
parsers, the pipeline, the preview session and its state machine -- is the real
implementation.

The claims:

* the loader passes RAVEN's fixed strength, the causal DiT class and the static
  patcher, and lets its own errors through untouched;
* the sampler resolves both VAE sockets by feature, loudly, including the
  swapped-socket case;
* the DiT and the VAEs are never handed to Comfy together: each chunk is
  sampled in a DiT phase and decoded in a VAE phase, each phase is one ordinary
  ``load_models_gpu`` call, and the ``memory_required`` it carries is that
  phase's workspace and nothing else;
* the preview lane produces one session per execution with the documented
  message order, and a preview that cannot start or dies mid-run changes
  neither the outputs nor the fact that the run finishes;
* the final IMAGE is the official full decode reshaped the way ``VAEDecode``
  reshapes it, and the final AUDIO is whatever the official helper returned.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import consistency, layout as layout_mod, nodes  # noqa: E402
from raven_streaming import streaming_pipeline as sp  # noqa: E402
from raven_streaming.media.audio_stream import (  # noqa: E402
    AudioLatentGeometry,
    OverlapSaveAudioDecoder,
)
from raven_streaming.media.fakes import (  # noqa: E402
    FakeVideoChunkDecoder,
    FiniteRFAudioDecoder,
)
from raven_streaming.media.video_stream import IncrementalVideoDecoder  # noqa: E402
from raven_streaming.preview_session import (  # noqa: E402
    PreviewManager,
    RecordingSender,
)


# --------------------------------------------------------------------------
# official-shaped inputs
# --------------------------------------------------------------------------


class NestedTensor:
    """``comfy.nested_tensor.NestedTensor``, reproduced field for field.

    The contract layer duck-types on ``is_nested`` / ``unbind`` / ``tensors``
    and rebuilds the *input's own class* for the output, so a stand-in has to
    carry all three -- and the node has to work with a class it never imported.
    """

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def empty_av_latent(width: int = 64, height: int = 64, frames: int = 39):
    """``EmptyMiniMaxH3LatentAV`` output, on the same grid upstream builds."""
    latent_t = layout_mod.video_latent_t(frames)
    audio_t = layout_mod.audio_latent_t(frames)
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16])
    audio = torch.zeros([1, 32, 2, audio_t])
    return {"samples": NestedTensor((video, audio))}


def t2va_conditioning(text_len: int = 6, dim: int = 32):
    tags = torch.full((text_len,), layout_mod.TEXT_TAG, dtype=torch.int64)
    return [[torch.zeros(1, text_len, dim), {"minimax_token_tags": tags}]]


class FakeAttention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim


class FakeMLP(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        # fc1 emits gate + up, exactly like comfy.ldm.minimax.model.MLP
        self.fc1 = torch.nn.Linear(hidden, ffn * 2, bias=False)


class FakeBlock(torch.nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int):
        super().__init__()
        self.attn = FakeAttention(heads, head_dim)
        self.mlp = FakeMLP(hidden, ffn)


class FakeDiT(torch.nn.Module):
    """Shaped like the real DiT where the budget reads it, tiny where it does not."""

    def __init__(self, layers: int = 2, hidden: int = 64, heads: int = 4, head_dim: int = 8,
                 ffn: int = 128, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden
        self.blocks = torch.nn.ModuleList(
            FakeBlock(hidden, heads, head_dim, ffn) for _ in range(layers)
        )
        self.dtype = dtype

    def prefill_text(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the stubbed sampler must not run the model")

    def forward_chunk(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("the stubbed sampler must not run the model")


class FakePatcher:
    """Enough of a ``ModelPatcher`` for ``contracts.resolve_model``.

    ``clone_base_uuid`` and ``get_nested_additional_models`` are here because
    the final-decode handover hands this object to the real
    ``unload_model_and_clones`` wherever a checkout is available.

    ``loaded_size`` / ``model_size`` / ``model_offload_buffer_memory`` are the
    three numbers the residency record reads back off upstream. They are plain
    attributes here so a test can say "upstream made 3 GiB resident" without
    pretending to run its loader.
    """

    def __init__(self, diffusion_model=None, loaded=0, size=0):
        self.model = SimpleNamespace(
            diffusion_model=diffusion_model if diffusion_model is not None else FakeDiT(),
            model_offload_buffer_memory=0,
        )
        self.model_options = {"transformer_options": {}}
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.wrappers = {}
        self.clone_base_uuid = uuid.uuid4()
        self.parent = None
        self.patches = {}
        self.loaded = loaded
        self.size = size

    def is_dynamic(self):
        return False

    def get_nested_additional_models(self):
        return []

    def loaded_size(self):
        return self.loaded

    def model_size(self):
        return self.size


# -- the two VAE sockets ---------------------------------------------------


class MiniMaxH3VideoVAE:  # noqa: N801 - the class name is the thing being checked
    """The inner video VAE's decode surface: the geometry, and a cheap decode.

    It really decodes -- 1 latent frame -> 4 pixel frames, 16x spatial -- because
    the collector is not stubbed out in these tests: it is the IMAGE output.
    """

    clip_length = 17
    vae_ratio_t = 4
    token_drop = 3

    def __init__(self):
        self.latents_mean = torch.zeros(24)
        self.latents_std = torch.ones(24)

    def _adaptive_decode(self, z):
        pixels = z[:, :3].repeat_interleave(self.vae_ratio_t, dim=2)
        return pixels.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        weights = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [blend_extent if i == dim % a.ndim else 1 for i in range(a.ndim)]
        weight_b = (weights / blend_extent).reshape(shape)
        head = [slice(None)] * a.ndim
        head[dim] = slice(a.shape[dim] - blend_extent, None)
        tail = [slice(None)] * b.ndim
        tail[dim] = slice(0, blend_extent)
        blended = a[tuple(head)] * (1 - weight_b) + b[tuple(tail)] * weight_b
        if blend_extent < b.shape[dim]:
            rest = [slice(None)] * b.ndim
            rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(rest)]], dim=dim)
        return blended

    def _finalize_pixels(self, part):
        return part.float().clamp(0.0, 1.0)


class MiniMaxH3AudioVAE:  # noqa: N801
    """The inner audio VAE's decode surface: the geometry, and a cheap decode.

    Like the video one, it really decodes -- 1 latent -> 800 stereo samples --
    because the audio collector is the AUDIO output and stubbing it out would
    hide the failure that matters.
    """

    samples_per_latent = 800
    sample_rate = 32000

    def decode(self, z):
        # [1, 32, 2, T] -> [1, 2, T * 800], the shape VAE.decode hands back
        batch, _channels, stereo, latents = z.shape
        pooled = z.mean(dim=1)  # [1, 2, T]
        return pooled.repeat_interleave(self.samples_per_latent, dim=-1).reshape(
            batch, stereo, latents * self.samples_per_latent
        )


class FakeVAEPatcher:
    """A VAE's ``ModelPatcher``: identity, the two size numbers, and the
    eviction surface.

    ``clone_base_uuid`` / ``get_nested_additional_models`` are here for the same
    reason ``FakePatcher`` carries them: the final-audio handover passes this
    object to the *real* ``unload_model_and_clones`` wherever a checkout is
    available, and a real ``comfy.sd.VAE.patcher`` is a ModelPatcher.
    """

    def __init__(self, name, size=0, loaded=0):
        self.name = name
        self.size = size
        self.loaded = loaded
        self.clone_base_uuid = uuid.uuid4()
        self.parent = None

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.loaded

    def get_nested_additional_models(self):
        return []


class FakeVAEWrapper:
    """The ``comfy.sd.VAE`` surface the node reads, and nothing more."""

    def __init__(
        self,
        inner,
        *,
        decode_memory=1,
        events=None,
        tag="",
        decode_raises=None,
        model_bytes=0,
        **attributes
    ):
        self.first_stage_model = inner
        self.patcher = FakeVAEPatcher(type(inner).__name__, size=model_bytes)
        self.device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.output_device = torch.device("cpu")
        self.decode_calls = []
        self._decode_memory = decode_memory
        self.memory_shapes = []
        self._events = events
        self._tag = tag or type(inner).__name__
        self._decode_raises = decode_raises
        for key, value in attributes.items():
            setattr(self, key, value)

    def memory_used_decode(self, shape, dtype):
        self.memory_shapes.append((tuple(shape), dtype))
        return self._decode_memory

    def decode(self, samples, vae_options={}):
        self.decode_calls.append(samples)
        if self._events is not None:
            self._events.append("{}-decode".format(self._tag))
        if self._decode_raises is not None:
            raise self._decode_raises
        # [B, C, T, H, W] latents -> [B, T, H*16, W*16, 3], as VAE.decode does
        b, _c, t, h, w = samples.shape
        frames = max(1, (t - 2) // 5 * 17 + 5)
        return torch.zeros((b, frames, h * 16, w * 16, 3))


def video_vae(**kwargs):
    kwargs.setdefault("tag", "video")
    return FakeVAEWrapper(
        MiniMaxH3VideoVAE(), latent_channels=24, latent_dim=3, **kwargs
    )


def audio_vae(**kwargs):
    kwargs.setdefault("tag", "audio")
    return FakeVAEWrapper(
        MiniMaxH3AudioVAE(),
        latent_channels=32,
        output_channels=2,
        audio_sample_rate=32000,
        upscale_ratio=800,
        latent_dim=2,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def no_upstream_vae_class(monkeypatch):
    """These VAEs are stand-ins, not ``comfy.sd.VAE`` instances.

    The isinstance gate is real and is covered against the real class in
    ``tests/test_nodes_upstream.py``; here it is switched off explicitly rather
    than left to depend on whether some earlier test imported ``comfy.sd``.
    """
    monkeypatch.setattr(nodes, "_comfy_vae_class", lambda: None)


# --------------------------------------------------------------------------
# Node 1: the loader
# --------------------------------------------------------------------------


class RecordedLoad:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else object()
        self.error = error
        self.calls = []

    def __call__(self, unet_name, lora_name, **kwargs):
        self.calls.append((unet_name, lora_name, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def test_loader_forces_strength_the_causal_dit_and_the_static_patcher(monkeypatch):
    recorder = RecordedLoad()
    sentinel = type("SentinelCausalDiT", (), {})
    monkeypatch.setattr(nodes.loader_mod, "load_raven_diffusion_model", recorder)
    monkeypatch.setattr(nodes, "_causal_model_class", lambda: sentinel)

    out = nodes.RAVENModelLoader().load_model("dit.safetensors", "raven.safetensors", "bf16")

    assert out == (recorder.result,)
    (unet, lora, kwargs) = recorder.calls[0]
    assert (unet, lora) == ("dit.safetensors", "raven.safetensors")
    assert kwargs["strength"] == 1.0
    assert kwargs["weight_dtype"] == "bf16"
    assert kwargs["unet_model_cls"] is sentinel
    assert kwargs["force_static_patcher"] is True
    assert kwargs["disable_dynamic"] is True


def test_loader_errors_reach_the_user_unchanged(monkeypatch):
    failure = nodes.loader_mod.PrunedCheckpointError("pruned / adaln-curve form")
    monkeypatch.setattr(
        nodes.loader_mod, "load_raven_diffusion_model", RecordedLoad(error=failure)
    )
    monkeypatch.setattr(nodes, "_causal_model_class", lambda: object)

    with pytest.raises(nodes.loader_mod.PrunedCheckpointError) as caught:
        nodes.RAVENModelLoader().load_model("dit.safetensors", "raven.safetensors", "default")
    assert caught.value is failure


# --------------------------------------------------------------------------
# VAE feature probes
# --------------------------------------------------------------------------


def test_both_vae_sockets_are_probed_by_feature():
    video = nodes.resolve_video_vae(video_vae())
    assert video.kind == "video" and video.inner is video.vae.first_stage_model
    audio = nodes.resolve_audio_vae(audio_vae())
    assert audio.kind == "audio" and audio.patcher is audio.vae.patcher


def test_swapped_vae_sockets_say_so():
    with pytest.raises(nodes.NodeInputError, match="sockets are swapped"):
        nodes.resolve_video_vae(audio_vae())
    with pytest.raises(nodes.NodeInputError, match="sockets are swapped"):
        nodes.resolve_audio_vae(video_vae())


def test_a_foreign_vae_is_refused_by_name():
    class AutoencoderKL:
        pass

    wrapper = FakeVAEWrapper(AutoencoderKL(), latent_channels=4, latent_dim=2)
    with pytest.raises(nodes.NodeInputError, match="MiniMaxH3VideoVAE"):
        nodes.resolve_video_vae(wrapper)
    with pytest.raises(nodes.NodeInputError, match="MiniMaxH3AudioVAE"):
        nodes.resolve_audio_vae(wrapper)


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        ("latent_channels", 16, "latent"),
        ("latent_dim", 2, "latent_dim"),
    ],
)
def test_a_video_vae_with_the_wrong_geometry_is_refused(attribute, value, message):
    vae = video_vae()
    setattr(vae, attribute, value)
    with pytest.raises(nodes.NodeInputError, match=message):
        nodes.resolve_video_vae(vae)


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        ("latent_channels", 24, "latent channels"),
        ("output_channels", 1, "stereo"),
        ("audio_sample_rate", 44100, "32000"),
        ("upscale_ratio", 512, "800"),
    ],
)
def test_an_audio_vae_with_the_wrong_geometry_is_refused(attribute, value, message):
    vae = audio_vae()
    setattr(vae, attribute, value)
    with pytest.raises(nodes.NodeInputError, match=message):
        nodes.resolve_audio_vae(vae)


def test_an_unloaded_vae_is_refused():
    vae = video_vae()
    vae.first_stage_model = None
    with pytest.raises(nodes.NodeInputError, match="not a loaded ComfyUI VAE"):
        nodes.resolve_video_vae(vae)
    with pytest.raises(nodes.NodeInputError, match="required"):
        nodes.resolve_audio_vae(None)


# --------------------------------------------------------------------------
# the two phases
# --------------------------------------------------------------------------


def fake_plan(dit_workspace=4096, vae_workspace=2048, **kwargs):
    """A :class:`JointOffloadPlan` with the two workspace numbers pinned."""
    return nodes.JointOffloadPlan(
        dit=nodes.PhasePlan(
            name="dit", workspace_bytes=dit_workspace, model_bytes=kwargs.get("dit_model", 0)
        ),
        vae=nodes.PhasePlan(
            name="vae", workspace_bytes=vae_workspace, model_bytes=kwargs.get("vae_model", 0)
        ),
        facts=nodes.DeviceMemoryFacts(),
    )


def test_the_load_closure_hands_over_the_dit_alone():
    """The VAEs are 5.4 GiB that no forward touches; they do not ride along."""
    video = nodes.resolve_video_vae(video_vae())
    audio = nodes.resolve_audio_vae(audio_vae())
    calls = []
    loaded = []

    def fake_load_models_gpu(models, memory_required=0, force_full_load=False):
        calls.append((list(models), memory_required, force_full_load))

    closure = nodes.make_load_models(
        video,
        audio,
        memory_required=0,
        plan=fake_plan(dit_workspace=4096),
        load_models_gpu=fake_load_models_gpu,
        on_loaded=lambda: loaded.append(1),
    )
    patcher = FakePatcher()
    closure([patcher], memory_required=0, force_full_load=False)

    assert closure.calls == 1 and loaded == [1]
    models, memory, force_full = calls[0]
    assert models == [patcher]
    assert video.patcher not in models and audio.patcher not in models
    # the DiT phase's workspace, not the whole-run reserve and not the decode
    assert memory == 4096 == closure.memory_required
    # the full non-pruned BF16 DiT is expected to be partially offloaded
    assert force_full is False


def test_the_load_closure_only_announces_the_first_load():
    """Every chunk reloads the DiT; only the first is a phase the client sees."""
    loaded = []
    closure = nodes.make_load_models(
        None,
        None,
        plan=fake_plan(),
        load_models_gpu=lambda models, memory_required=0, force_full_load=False: None,
        on_loaded=lambda: loaded.append(1),
    )
    for _ in range(3):
        closure([FakePatcher()])
    assert closure.calls == 3 and loaded == [1]


def test_the_load_closure_records_what_upstream_made_resident():
    resident = FakePatcher()
    resident.loaded = 3 * 1024 ** 3
    resident.model.model_offload_buffer_memory = 512 * 1024 ** 2
    closure = nodes.make_load_models(
        None,
        None,
        plan=fake_plan(dit_workspace=1024 ** 3),
        load_models_gpu=lambda models, memory_required=0, force_full_load=False: None,
    )
    closure([resident])

    residency = closure.residency
    assert residency.phase == "dit"
    assert residency.dit_loaded_bytes == 3 * 1024 ** 3
    assert residency.offload_buffer_bytes == 512 * 1024 ** 2
    assert residency.workspace_bytes == 1024 ** 3
    # what upstream did, plus what this node is about to allocate
    assert residency.predicted_peak_bytes == (
        3 * 1024 ** 3 + 512 * 1024 ** 2 + 1024 ** 3
    )
    assert residency.within_planning is True
    assert residency.to_dict()["model_offload_buffer_memory"] == 512 * 1024 ** 2


def test_a_residency_over_the_reference_budget_is_reported_not_refused(caplog):
    """Upstream sized this against real free memory; the node only says so."""
    fat = FakePatcher()
    fat.loaded = 60 * 1024 ** 3
    closure = nodes.make_load_models(
        None,
        None,
        plan=fake_plan(),
        load_models_gpu=lambda models, memory_required=0, force_full_load=False: None,
    )
    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        closure([fat])  # no raise
    assert closure.residency.within_planning is False
    assert closure.residency.over_bytes > 0
    assert any("over by" in r.getMessage() for r in caplog.records)
    # nothing is refused and nothing is unloaded: upstream sized this against
    # the memory that was actually free
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_the_workspace_is_the_larger_of_the_two_decode_estimates():
    video = nodes.resolve_video_vae(video_vae(decode_memory=900))
    audio = nodes.resolve_audio_vae(audio_vae(decode_memory=1500))
    config = sp.PipelineConfig(frames=39, width=64, height=64)

    workspace = nodes.decode_workspace_bytes(video, audio, config, latent_h=4, latent_w=4)
    assert workspace == 1500  # max, not sum: the two decodes never overlap

    # and the shapes handed to upstream's estimators are the real ones
    video_shape, _dtype = video.vae.memory_shapes[0]
    assert video_shape == (1, 24, 7, 4, 4)  # 5 latents + the 2-latent lookahead
    audio_shape, _dtype = audio.vae.memory_shapes[0]
    assert audio_shape == (1, 32, 2, 28 + 2 * 17)


def test_a_broken_memory_estimator_costs_the_estimate_not_the_run():
    video = video_vae()
    video.memory_used_decode = lambda shape, dtype: 1 / 0
    resolved_video = nodes.resolve_video_vae(video)
    resolved_audio = nodes.resolve_audio_vae(audio_vae(decode_memory=77))
    config = sp.PipelineConfig(frames=39, width=64, height=64)
    assert nodes.decode_workspace_bytes(resolved_video, resolved_audio, config, 4, 4) == 77


# --------------------------------------------------------------------------
# final outputs
# --------------------------------------------------------------------------


def test_final_image_is_the_official_full_decode_flattened():
    vae = video_vae()
    latent = empty_av_latent(width=64, height=64, frames=39)
    images = nodes.decode_images(vae, latent)

    # the video stream of the nested pair, never the audio one
    assert vae.decode_calls[0] is latent["samples"].tensors[0]
    # [1, T, H, W, 3] -> [T, H, W, 3], exactly what nodes.VAEDecode does
    assert images.shape == (39, 64, 64, 3)


def test_final_audio_is_whatever_the_official_helper_returned():
    vae = audio_vae()
    latent = empty_av_latent()
    sentinel = {"waveform": torch.zeros(1, 2, 8), "sample_rate": 32000}
    seen = []

    def helper(passed_vae, passed_latent):
        seen.append((passed_vae, passed_latent))
        return sentinel

    out = nodes.decode_audio(vae, latent, helper=helper)
    assert out is sentinel  # normalisation and the dict shape are the helper's
    assert seen == [(vae, latent)]


def test_the_default_audio_helper_is_the_pinned_upstream_one():
    import inspect

    source = inspect.getsource(nodes._default_audio_helper)
    assert "comfy_extras.nodes_audio" in source
    assert "vae_decode_audio" in source


# --------------------------------------------------------------------------
# handing the card over to the final decode
# --------------------------------------------------------------------------


def test_the_handover_calls_the_pinned_api_with_the_pinned_arguments():
    model_management = FakeModelManagement()
    model = FakePatcher()

    handover = nodes.prepare_final_decode(model, model_management=model_management)

    assert model_management.unload_calls == [(model, True, False)]
    assert model_management.soft_empty_calls == 1
    # the cache is emptied *after* the unload, not before
    assert model_management.events == ["unload", "soft_empty_cache"]
    assert handover.seconds >= 0.0
    assert handover.strategy == nodes.FINAL_DECODE_UNLOAD_STRATEGY
    assert "unload_model_and_clones" in handover.strategy


def test_the_handover_only_ever_evicts_this_runs_model():
    model_management = FakeModelManagement()
    model = FakePatcher()
    nodes.prepare_final_decode(model, model_management=model_management)

    (evicted, additional, all_devices) = model_management.unload_calls[0]
    assert evicted is model                 # not "every model", not the VAEs
    assert additional is True               # nested additional models go too
    assert all_devices is False             # only the device this run used
    assert model_management.blunt_calls == []


def test_the_node_never_reaches_for_the_process_wide_sledgehammers():
    import inspect

    source = inspect.getsource(nodes)
    body = source.split("def prepare_final_decode")[1]
    call_sites = [
        line
        for line in body.splitlines()
        if ("unload_all_models" in line or "cleanup_models" in line)
        and "``" not in line
        and not line.strip().startswith("*")
    ]
    assert call_sites == [], call_sites


def test_the_handover_reports_what_it_freed():
    model_management = FakeModelManagement(free=(8 * 1024 ** 3, 96 * 1024 ** 3))
    handover = nodes.prepare_final_decode(FakePatcher(), model_management=model_management)

    assert handover.free_before == 8 * 1024 ** 3
    assert handover.free_after == 96 * 1024 ** 3
    assert handover.freed_bytes == 88 * 1024 ** 3
    assert handover.device == "cuda:0"

    payload = handover.to_dict()
    assert payload["final_decode_unload_seconds"] == handover.seconds
    assert payload["final_decode_unload_strategy"] == nodes.FINAL_DECODE_UNLOAD_STRATEGY
    assert payload["final_decode_freed_bytes"] == 88 * 1024 ** 3
    text = handover.describe()
    assert "88.00 GiB" in text and "VAEs stay resident" in text


def test_a_measurement_failure_does_not_stop_the_handover():
    class NoMeasurement(FakeModelManagement):
        def get_free_memory(self, device):
            raise RuntimeError("no such device")

    model_management = NoMeasurement()
    handover = nodes.prepare_final_decode(FakePatcher(), model_management=model_management)
    assert model_management.unload_calls  # the eviction still happened
    assert handover.freed_bytes is None
    assert "GiB" not in handover.describe()


def test_a_missing_unload_api_is_loud():
    class WithoutTheApi(FakeModelManagement):
        unload_model_and_clones = None

    with pytest.raises(nodes.NodeInputError, match="unload_model_and_clones"):
        nodes.prepare_final_decode(FakePatcher(), model_management=WithoutTheApi())

    class WithoutSoftEmpty(FakeModelManagement):
        soft_empty_cache = None

    with pytest.raises(nodes.NodeInputError, match="soft_empty_cache"):
        nodes.prepare_final_decode(FakePatcher(), model_management=WithoutSoftEmpty())


def test_an_unload_failure_propagates_untouched():
    failure = RuntimeError("model is in use")
    model_management = FakeModelManagement(fail=failure)
    with pytest.raises(RuntimeError) as caught:
        nodes.prepare_final_decode(FakePatcher(), model_management=model_management)
    assert caught.value is failure
    # and the cache was never emptied, because the unload never completed
    assert model_management.soft_empty_calls == 0


# --------------------------------------------------------------------------
# node id / execution context
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (7, "7"),
        ("7", "7"),
        (["7"], "7"),
        ("12:7", "12:7"),          # a node inside a subgraph
        ("  9  ", "9"),
        (None, None),
        ("", None),
        ("-1", None),              # litegraph before configure(); never matches
        ([], None),
    ],
)
def test_unique_id_is_normalised_to_the_protocols_node_id(raw, expected):
    assert nodes.normalise_node_id(raw) == expected


def test_identity_falls_back_to_the_execution_context():
    # no ComfyUI here, so there is no context: the hidden input is all there is
    assert nodes.executing_identity("12:7") == ("12:7", None)
    assert nodes.executing_identity(None) == (None, None)


def test_no_node_id_means_no_session_and_no_sink():
    with nodes.preview_sink(None) as sink:
        assert sink is None


def test_a_preview_that_cannot_install_still_yields_a_sink_slot(monkeypatch):
    import raven_streaming.preview as preview_mod

    def explode(*args, **kwargs):
        raise RuntimeError("no server here")

    monkeypatch.setattr(preview_mod, "install", explode)
    with nodes.preview_sink("7") as sink:
        assert sink is None


# --------------------------------------------------------------------------
# the sampler node, end to end with the rollout stubbed
# --------------------------------------------------------------------------


class StubSegment:
    def __init__(self, index):
        self.kind = "fragment"
        self.index = index
        self.data = bytes([index % 251]) * 64


class StubMuxer:
    """Records what was muxed and emits one fragment per frame, like PyAV does.

    It emits rather than swallowing because the node-level claim under test is
    "fragments leave during the rollout": a muxer that produced nothing would
    make that claim untestable from here.
    """

    def __init__(self, on_close=None):
        self.frames = []
        self.audio = []
        self.closes = 0
        self._on_close = on_close
        self._pending = []
        self._index = 0
        self._init_taken = False

    def write_video_frame(self, image, force_keyframe=None):
        self.frames.append(image)
        self._pending.append(StubSegment(self._index))
        self._index += 1

    def write_audio(self, pcm):
        self.audio.append(pcm)

    def take_init_segment(self):
        if self._init_taken or not self.frames:
            return None
        self._init_taken = True
        return b"ftyp+moov"

    def take_fragments(self):
        out = self._pending
        self._pending = []
        return out

    def close(self):
        self.closes += 1
        if self._on_close is not None:
            self._on_close()


class StubDecoder:
    """Accepts latents, produces nothing. Stands in for the *audio* lane only."""

    def __init__(self):
        self.pushed = []
        self.finished = 0

    def push(self, z):
        self.pushed.append(z)
        return []

    def finish(self):
        self.finished += 1
        return []


def assert_audio(payload, frames: int = 39, sample_rate: int = 32000) -> None:
    """The AUDIO output's shape contract, as a Comfy consumer would read it."""
    assert set(payload) == {"waveform", "sample_rate"}
    waveform = payload["waveform"]
    assert waveform.dtype == torch.float32
    assert waveform.shape == (1, 2, round(frames / 24 * 40) * 800)
    assert payload["sample_rate"] == sample_rate


def make_audio_collector(config) -> OverlapSaveAudioDecoder:
    """A real overlap-save collector over a fake VAE, at the published margins.

    Not stubbed for the same reason the video one is not: it is the AUDIO
    output, and a stub that produced no samples would hide the failure that
    matters.
    """
    return OverlapSaveAudioDecoder(
        FiniteRFAudioDecoder(radius=2, samples_per_latent=800, latent_channels=32),
        margin=config.audio_margin_latents,
        block_latents=config.audio_block_latents,
        geometry=AudioLatentGeometry(800, 32000),
    )


def make_collector() -> IncrementalVideoDecoder:
    """A real incremental collector over a fake VAE.

    ``spatial_scale=16`` is the H3 video VAE's spatial upscale, so the frames
    come out at exactly the canvas the latent grid implies -- which is what the
    collector checks its buffer against.

    The collector is never stubbed in these tests: it is the IMAGE output, so a
    stub that produced no frames would hide exactly the failure that matters.
    """
    return IncrementalVideoDecoder(
        FakeVideoChunkDecoder(vae_ratio_t=4, out_channels=3, spatial_scale=16)
    )


class StubRollout:
    """Stands in for ``consistency.sample_streaming``; records how it was called.

    The chunks it delivers are the *real* ones for the latent it was given --
    same grid, same counts -- because the collector downstream checks that they
    add up to the requested frame count.
    """

    def __init__(self, latent, fail=None, stop_after=None):
        self.latent = latent
        self.fail = fail
        self.stop_after = stop_after
        self.kwargs = None
        self.delivered = []

    def _plan(self):
        video, _audio = self.latent["samples"].unbind()
        _batch, _channels, latent_t, latent_h, latent_w = video.shape
        frames = (latent_t - 2) // 5 * 17 + 5
        return layout_mod.T2VALayout.from_request(
            text_len=8,
            frames=frames,
            width=latent_w * 16,
            height=latent_h * 16,
            warn_experimental=False,
        )

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        kwargs["load_models"]([FakePatcher()], memory_required=0, force_full_load=False)
        plan = self._plan()
        for chunk in plan.chunks:
            if self.stop_after is not None and chunk.index >= self.stop_after:
                break
            output = consistency.ChunkOutput(
                index=chunk.index,
                is_last=chunk.index == len(plan.chunks) - 1,
                video_start=chunk.video_start,
                video_stop=chunk.video_stop,
                audio_start=chunk.audio_start,
                audio_stop=chunk.audio_stop,
                video_x0=torch.zeros(
                    1, 24, chunk.video_latents, plan.latent_h, plan.latent_w
                ),
                audio_x0=torch.zeros(1, 32, 2, chunk.audio_latents),
            )
            self.delivered.append(output)
            kwargs["on_chunk"](output)
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(latent=self.latent, layout=None, config=None)


class FakeModelManagement:
    """The ``comfy.model_management`` surface the node is allowed to touch.

    The two process-wide sledgehammers are present precisely so a test can show
    they are never reached. ``total`` is a parameter because "does this node
    behave the same on a 24 GiB card and a 141 GiB one?" is a question the
    residency design has to answer with "yes".
    """

    def __init__(
        self,
        events=None,
        fail=None,
        free=(8 * 1024 ** 3, 96 * 1024 ** 3),
        total=141 * 1024 ** 3,
        num_streams=2,
        extra_reserved=400 * 1024 ** 2,
    ):
        self.events = events if events is not None else []
        self.fail = fail
        self.unload_calls = []
        self.soft_empty_calls = 0
        self.blunt_calls = []
        self._free = list(free)
        self.device = "cuda:0"
        self.total = total
        self.NUM_STREAMS = num_streams
        self._extra_reserved = extra_reserved

    # -- what the node may call -----------------------------------------

    def get_torch_device(self):
        return self.device

    def get_total_memory(self, device=None):
        return self.total

    def extra_reserved_memory(self):
        return self._extra_reserved

    def get_free_memory(self, device=None):
        # The last reading is sticky: the node reads free memory for its own
        # report as well as for the handover, and a queue that ran dry would
        # turn a diagnostic into a different answer for the code under test.
        if not self._free:
            return 0
        if len(self._free) == 1:
            return self._free[0]
        return self._free.pop(0)

    def unload_model_and_clones(self, model, unload_additional_models=True, all_devices=False):
        self.unload_calls.append((model, unload_additional_models, all_devices))
        self.events.append("unload")
        if self.fail is not None:
            raise self.fail

    def soft_empty_cache(self, force=False):
        self.soft_empty_calls += 1
        self.events.append("soft_empty_cache")

    # -- what it must not ------------------------------------------------

    def unload_all_models(self):  # pragma: no cover - reaching it is the failure
        self.blunt_calls.append("unload_all_models")

    def cleanup_models(self, *args, **kwargs):  # pragma: no cover
        self.blunt_calls.append("cleanup_models")

    def unload_all_models_and_clones(self, *args, **kwargs):  # pragma: no cover
        self.blunt_calls.append("unload_all_models_and_clones")

    def free_memory(self, *args, **kwargs):  # pragma: no cover
        self.blunt_calls.append("free_memory")


@pytest.fixture
def wired(monkeypatch):
    """The sampler node with the rollout, the loader and the preview injected."""
    sender = RecordingSender()
    manager = PreviewManager(sender=sender)
    import raven_streaming.preview as preview_mod

    monkeypatch.setattr(preview_mod, "install", lambda *a, **k: (manager, False))
    monkeypatch.setattr(preview_mod, "current_client_id", lambda *a, **k: "client-1")

    events = []
    loads = []

    def fake_load_models_gpu(models, memory_required=0, force_full_load=False):
        given = list(models)
        loads.append((given, memory_required, force_full_load))
        # Which phase a load *is* is decided by what it loads, exactly as it is
        # in the product: there is no phase flag on the call.
        kinds = {
            "MiniMaxH3VideoVAE": "video-vae",
            "MiniMaxH3AudioVAE": "audio-vae",
        }
        names = sorted(kinds.get(getattr(m, "name", ""), "dit") for m in given)
        events.append("load[{}]".format("+".join(names)))

    monkeypatch.setattr(nodes, "_default_load_models_gpu", lambda: fake_load_models_gpu)

    model_management = FakeModelManagement(events=events)
    monkeypatch.setattr(nodes, "_model_management", lambda: model_management)

    # Whether ComfyUI's interrupt hook resolves depends on what else this
    # session imported, so it is pinned here; that it is the real thrower is
    # checked against a real checkout in tests/test_nodes_upstream.py.
    def cancel_check():
        return None

    monkeypatch.setattr(nodes, "_resolve_cancel_check", lambda: cancel_check)

    # The node no longer calls this -- the AUDIO comes from the collector --
    # but it stays patched so that a regression *back* to the whole-clip decode
    # shows up as an unexpected 'audio-decode' event rather than as an import.
    def audio_helper(vae, latent):
        events.append("audio-decode")
        raise AssertionError("the node must not call the whole-clip audio decode")

    monkeypatch.setattr(nodes, "_default_audio_helper", lambda: audio_helper)

    built = {}
    marks = {}

    def phases_now():
        return [body["phase"] for body in sender.bodies if body["event"] == "status"]

    def on_muxer_close():
        marks.setdefault("phases_at_close", phases_now())
        events.append("finish")

    def fake_build(
        *, video_vae, audio_vae, config, sink=None, log=None, muxer=None, memory_budget=None
    ):
        built["memory_budget"] = memory_budget
        built["muxer"] = StubMuxer(on_close=on_muxer_close)
        built["video"] = make_collector()
        built["audio"] = make_audio_collector(config)
        built["pipeline"] = sp.StreamingPipeline(
            config=config,
            video_decoder=built["video"],
            audio_decoder=built["audio"],
            muxer=built["muxer"],
            sink=sink,
            log=log,
            memory_budget=memory_budget,
            decode_policy={"order": "stubbed"},
        )
        return built["pipeline"]

    monkeypatch.setattr(nodes.pipeline_mod, "build_media_pipeline", fake_build)

    return SimpleNamespace(
        sender=sender,
        manager=manager,
        cancel_check=cancel_check,
        marks=marks,
        phases_now=phases_now,
        events=events,
        model_management=model_management,
        loads=loads,
        built=built,
        monkeypatch=monkeypatch,
    )


def run_sampler(wired, *, latent=None, rollout=None, unique_id="7", **overrides):
    latent = latent if latent is not None else empty_av_latent()
    rollout = rollout if rollout is not None else StubRollout(latent)
    wired.monkeypatch.setattr(consistency, "sample_streaming", rollout)
    wired.rollout = rollout
    arguments = dict(
        model=FakePatcher(),
        positive=t2va_conditioning(),
        latent=latent,
        video_vae=video_vae(decode_memory=900, events=wired.events),
        audio_vae=audio_vae(decode_memory=1500, events=wired.events),
        seed=1234,
        steps=4,
        video_shift=12.0,
        audio_shift=3.0,
        sink=2,
        window=2,
        unique_id=unique_id,
    )
    arguments.update(overrides)
    wired.arguments = arguments
    return nodes.RAVENStreamingSampler().sample(**arguments)


def test_sampler_returns_latent_image_and_audio(wired):
    latent = empty_av_latent(width=64, height=64, frames=39)
    out = run_sampler(wired, latent=latent)

    assert len(out) == 3
    assert out[0] is latent                       # the sampler's own result
    assert out[1].shape == (39, 64, 64, 3)        # official full decode
    assert_audio(out[2])                          # from the audio collector


def test_sampler_calls_the_rollout_with_the_streaming_wiring(wired):
    run_sampler(wired, steps=6, video_shift=9.5, audio_shift=2.5, sink=3, window=1, seed=99)
    kwargs = wired.rollout.kwargs

    # not the pipeline's own callback any more: the decode inside it needs the
    # VAEs resident and the forward after it needs the DiT back
    assert kwargs["on_chunk"] != wired.built["pipeline"].on_chunk
    assert getattr(kwargs["on_chunk"], "__self__", None).__class__ is nodes.PhaseSwapCoordinator
    assert kwargs["on_chunk"].__self__.pipeline is wired.built["pipeline"]
    assert callable(kwargs["load_models"])
    assert kwargs["warn_experimental"] is False
    # the sampler polls ComfyUI's interrupt, not a hook of our own
    assert kwargs["cancel_check"] is wired.cancel_check

    config = kwargs["config"]
    assert isinstance(config, consistency.SamplerConfig)
    assert (config.steps, config.seed) == (6, 99)
    assert (config.video_shift, config.audio_shift) == (9.5, 2.5)
    assert (config.sink, config.window) == (3, 1)

    # the model / conditioning / latent go through untouched
    assert kwargs["model"] is wired.arguments["model"]
    assert kwargs["positive"] is wired.arguments["positive"]
    assert kwargs["latent"] is wired.arguments["latent"]


VAE_PATCHER_NAMES = ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE")


def _is_vae_patcher(model):
    return getattr(model, "name", None) in VAE_PATCHER_NAMES


def _is_dit_load(models):
    """A load is a DiT load exactly when no VAE patcher is in it."""
    return not any(_is_vae_patcher(m) for m in models)


def test_no_load_ever_makes_the_dit_and_a_vae_co_resident(wired):
    """5.4 GiB of VAE weights are idle during every forward; they do not ride along."""
    run_sampler(wired)
    assert wired.loads, "nothing was loaded at all"
    for models, _memory, force_full in wired.loads:
        kinds = {_is_vae_patcher(m) for m in models}
        assert len(kinds) == 1, [getattr(m, "name", m) for m in models]
        assert force_full is False


def test_each_phase_asks_for_its_own_workspace_and_nothing_else(wired):
    run_sampler(wired)
    dit_loads = [call for call in wired.loads if _is_dit_load(call[0])]
    vae_loads = [call for call in wired.loads if not _is_dit_load(call[0])]
    assert dit_loads and vae_loads

    budget = wired.built["pipeline"].memory_budget
    plan = budget["detail"]["joint_offload_plan"]
    dit_required = plan["phases"]["dit"]["memory_required"]
    vae_required = plan["phases"]["vae"]["memory_required"]

    assert {call[1] for call in dit_loads} == {dit_required}
    assert {call[1] for call in vae_loads} == {vae_required}

    # the DiT phase pays for the KV slot, the activations and the buffers ...
    dit_items = plan["phases"]["dit"]["items"]
    assert dit_items["kv_slot_bytes"] > 0
    assert "decode_workspace_bytes" not in dit_items
    # ... and the VAE phase pays for the decode, which is the larger of the two
    vae_items = plan["phases"]["vae"]["items"]
    assert vae_items["decode_workspace_bytes"] == 1500
    assert budget["decode_workspace_bytes"] == 1500
    # neither asks for weights: that is upstream's own accounting
    assert dit_required == sum(dit_items.values())
    assert vae_required == sum(vae_items.values())


def test_the_report_carries_the_budget_and_the_decode_policy(wired):
    run_sampler(wired)
    report = wired.built["pipeline"].report().to_dict()
    budget = report["memory_budget"]
    assert set(budget) >= {
        "kv_cache_bytes",
        "forward_workspace_bytes",
        "rollout_buffer_bytes",
        "decode_workspace_bytes",
        "safety_bytes",
        "total_bytes",
        "detail",
    }
    detail = budget["detail"]
    assert detail["sink"] == 2 and detail["window"] == 2
    assert detail["frames"] == 39
    assert detail["kv_peak_rows"] > 0
    # measured off the fake DiT rather than assumed
    assert detail["num_layers"] == 2 and detail["num_heads"] == 4
    assert "num_layers" in detail["measured"]
    assert report["decode_policy"] == {"order": "stubbed"}


def test_every_chunk_reaches_both_lanes_and_the_media_lane_is_closed(wired):
    run_sampler(wired)
    pipeline = wired.built["pipeline"]
    assert pipeline.chunks == 3  # 39 frames: 5 / 5 / 2 latents
    assert wired.built["muxer"].closes == 1
    assert pipeline.finished is True
    # both collectors ran to completion and their buffers are the outputs
    report = pipeline.report()
    assert report.collected_frames == report.expected_frames == 39
    assert report.image_shape == (39, 64, 64, 3)
    assert report.collected_samples == report.expected_samples == 65 * 800
    assert report.audio_complete is True
    # one emission record per chunk, written as the chunks arrived
    assert [e.chunk for e in report.chunk_emissions] == [0, 1, 2]


def test_the_node_streams_while_it_samples(wired):
    """Fragments reach the client during the rollout, not at the end of it.

    The rollout stub calls ``on_chunk`` and nothing else, so any ``segment``
    that has been sent by the time chunk *n* returns was sent from inside that
    callback -- which is the property the user asked to be able to check.
    """
    sent_per_chunk = []
    rollout = StubRollout(empty_av_latent())
    original = wired.built  # populated by fake_build during the run

    class WatchingRollout(StubRollout):
        def __call__(self, **kwargs):
            on_chunk = kwargs["on_chunk"]

            def watched(chunk):
                on_chunk(chunk)
                sent_per_chunk.append(
                    len([b for b in wired.sender.bodies if b["event"] == "segment"])
                )

            kwargs["on_chunk"] = watched
            return StubRollout.__call__(self, **kwargs)

    run_sampler(wired, rollout=WatchingRollout(rollout.latent))

    # chunk 0 has nothing decidable yet (both decoders are filling context);
    # chunk 1 puts frames on the wire, before the run is anywhere near over
    assert sent_per_chunk[0] == 0
    assert sent_per_chunk[1] > 0
    assert sent_per_chunk == sorted(sent_per_chunk)

    report = original["pipeline"].report()
    assert report.fragments_before_finish == sent_per_chunk[-1]
    assert report.fragments_before_finish > 0


def test_the_node_logs_the_per_chunk_emission_table(wired, caplog):
    with caplog.at_level(logging.INFO, logger="raven_streaming.nodes"):
        run_sampler(wired)
    messages = [record.getMessage() for record in caplog.records]

    table = next(m for m in messages if m.startswith("raven emission log"))
    assert "chunk   0" in table and "chunk   2" in table
    assert "before finish()" in table

    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        caplog.clear()
        run_sampler(wired)
    record = next(
        m for m in (r.getMessage() for r in caplog.records)
        if m.startswith("raven memory record:")
    )
    # the same table, machine-readable, next to the memory numbers
    assert "emission_log" in record and "fragments_before_finish" in record


def test_the_session_reports_the_documented_phases_in_order(wired):
    run_sampler(wired)
    bodies = wired.sender.bodies
    events = [body["event"] for body in bodies]

    # the control messages, in order, with the media interleaved between them
    assert [e for e in events if e != "segment"] == [
        "open",
        "status",       # model_loading
        "status",       # sampling
        "init",         # the moment the first frames were muxed, mid-rollout
        "status",       # finalizing
        "end",
    ]
    assert events.count("segment") > 0
    # init lands before finalizing: the stream started while sampling was still
    # going, which is the whole point of the lane
    assert events.index("init") < len(events) - 2
    assert [body["seq"] for body in bodies] == list(range(len(bodies)))
    assert [b["phase"] for b in bodies if b["event"] == "status"] == [
        "model_loading",
        "sampling",
        "finalizing",
    ]
    assert bodies[-1]["reason"] == "complete"
    # one session, addressed to one node and one client
    assert {body["node_id"] for body in bodies} == {"7"}
    assert len({body["session_id"] for body in bodies}) == 1
    assert {message[2] for message in wired.sender.messages} == {"client-1"}


def test_a_subgraph_node_id_survives_into_the_session(wired):
    run_sampler(wired, unique_id="12:7")
    assert {body["node_id"] for body in wired.sender.bodies} == {"12:7"}


def test_each_execution_gets_its_own_session(wired):
    run_sampler(wired)
    first = {body["session_id"] for body in wired.sender.bodies}
    run_sampler(wired)
    sessions = {body["session_id"] for body in wired.sender.bodies}
    assert len(sessions) == 2 and first < sessions


def test_status_before_sampling_says_the_models_are_loading(wired):
    run_sampler(wired)
    events = [body["event"] for body in wired.sender.bodies]
    statuses = [body for body in wired.sender.bodies if body["event"] == "status"]
    # 'sampling' is sent by the load closure, i.e. after the weights are there
    assert statuses[0]["phase"] == "model_loading"
    assert statuses[1]["phase"] == "sampling"
    assert events.index("open") == 0


def test_finalizing_is_announced_before_the_flush_and_the_finalisers(wired):
    seen = {}
    original = sp.StreamingPipeline.finalize_audio

    def recording_finalize(self, vae=None, **kwargs):
        seen["phases_at_finalize_audio"] = wired.phases_now()
        return original(self, vae, **kwargs)

    wired.monkeypatch.setattr(sp.StreamingPipeline, "finalize_audio", recording_finalize)
    run_sampler(wired)

    # the two tail flushes and the normalisation all take real time; a client
    # left on 'sampling' through them cannot tell them from a hang
    assert wired.marks["phases_at_close"] == ["model_loading", "sampling", "finalizing"]
    assert seen["phases_at_finalize_audio"] == ["model_loading", "sampling", "finalizing"]


def test_the_dit_is_evicted_between_the_flush_and_the_final_decode(wired):
    """The 39-frame E2E failure, in order -- minus the decode that caused it."""
    out = run_sampler(wired)

    assert wired.events == [
        "load[dit]",                       # chunk 0's forward
        "load[audio-vae+video-vae]",       # chunk 0's decode, both collectors
        "load[dit]",                       # chunk 1's forward
        "load[audio-vae+video-vae]",
        "load[dit]",                       # chunk 2 -- the last one
        "load[audio-vae+video-vae]",
        "finish",           # both tail flushes, through those VAEs
        "unload",           # this run's DiT and its clones, nothing else
        "soft_empty_cache",
    ]
    # the last chunk deliberately does not reload the DiT: nothing else samples
    assert "load[dit]" not in wired.events[6:]
    # neither whole-clip decode happens any more: both outputs were collected
    assert "audio-decode" not in wired.events
    # there is no whole-clip video decode any more: the IMAGE came from the
    # collector, frame by frame, during the rollout
    assert "video-decode" not in wired.events
    assert wired.arguments["video_vae"].decode_calls == []
    # the first eviction targeted the MODEL this execution was given, the
    # second the video VAE whose work is finished
    assert wired.model_management.unload_calls == [
        (wired.arguments["model"], True, False),
    ]
    assert wired.model_management.blunt_calls == []
    # the audio VAE is never handed to the unloader; it is what runs next
    evicted = [call[0] for call in wired.model_management.unload_calls]
    assert wired.arguments["video_vae"] not in evicted   # the wrapper, not the patcher
    assert wired.arguments["audio_vae"] not in evicted
    assert wired.arguments["audio_vae"].patcher not in evicted
    # ... and the outputs are unchanged by any of it
    assert out[0] is wired.arguments["latent"]
    assert out[1].shape[0] == 39
    assert_audio(out[2])


# --------------------------------------------------------------------------
# the phase swap itself
# --------------------------------------------------------------------------


class StubPhasePipeline:
    """Records when its ``on_chunk`` ran, relative to the loads around it."""

    def __init__(self, events, preview_disabled=False, fail=None):
        self.events = events
        self.preview_disabled = preview_disabled
        self.fail = fail

    def on_chunk(self, chunk):
        self.events.append(("decode", chunk.index))
        if self.fail is not None:
            raise self.fail


def phase_chunk(index, is_last=False):
    return SimpleNamespace(index=index, is_last=is_last)


def make_coordinator(events, *, chunks=3, preview_disabled=False, fail=None, audio=True):
    video = nodes.resolve_video_vae(video_vae(model_bytes=4 * 1024 ** 3))
    resolved_audio = (
        nodes.resolve_audio_vae(audio_vae(model_bytes=1024 ** 3)) if audio else None
    )
    pipeline = StubPhasePipeline(events, preview_disabled=preview_disabled, fail=fail)

    def loader(models, memory_required=0, force_full_load=False):
        events.append(("vae-load", [m.name for m in models], memory_required))

    def load_dit(models, memory_required=0, force_full_load=False):
        events.append(("dit-load", memory_required))

    coordinator = nodes.PhaseSwapCoordinator(
        model=FakePatcher(),
        video=video,
        audio=resolved_audio,
        pipeline=pipeline,
        plan=fake_plan(dit_workspace=111, vae_workspace=222),
        load_dit=load_dit,
        load_models_gpu=loader,
        needs_audio=lambda: not pipeline.preview_disabled,
    )
    return coordinator, pipeline


def test_the_coordinator_loads_the_vaes_decodes_then_takes_the_dit_back():
    events = []
    coordinator, _pipeline = make_coordinator(events)
    for index in range(3):
        coordinator.on_chunk(phase_chunk(index, is_last=index == 2))

    assert events == [
        ("vae-load", ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"], 222),
        ("decode", 0),
        ("dit-load", 0),
        ("vae-load", ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"], 222),
        ("decode", 1),
        ("dit-load", 0),
        ("vae-load", ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"], 222),
        ("decode", 2),
        # ... and nothing after the last chunk: the VAEs stay for the tail
        # flush and the final audio decode, and no forward is left to run
    ]
    assert coordinator.last_phase == "vae"
    assert (coordinator.chunks, coordinator.vae_loads, coordinator.dit_loads) == (3, 3, 2)


def test_the_video_vae_is_the_one_upstream_serves_first():
    """``load_models_gpu`` reverses the list, so the last entry is served first.

    The video VAE decodes the IMAGE on every chunk; the audio VAE only feeds the
    preview. On a card that cannot hold both, that is the order the two must be
    squeezed in.
    """
    events = []
    coordinator, _pipeline = make_coordinator(events)
    coordinator.on_chunk(phase_chunk(0))
    (_kind, names, _memory) = events[0]
    assert names[-1] == "MiniMaxH3VideoVAE"


def test_without_a_preview_the_audio_vae_waits_for_the_last_chunk():
    """No preview, no audio decode -- until the final AUDIO has to come out."""
    events = []
    coordinator, _pipeline = make_coordinator(events, preview_disabled=True)
    coordinator.on_chunk(phase_chunk(0))
    coordinator.on_chunk(phase_chunk(1, is_last=True))

    loads = [event[1] for event in events if event[0] == "vae-load"]
    assert loads[0] == ["MiniMaxH3VideoVAE"]          # collector only
    assert loads[1] == ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]
    assert coordinator.audio_vae_loads == 1


def test_a_preview_that_dies_mid_run_stops_loading_the_audio_vae():
    events = []
    coordinator, pipeline = make_coordinator(events)
    coordinator.on_chunk(phase_chunk(0))
    pipeline.preview_disabled = True  # what StreamingPipeline does on failure
    coordinator.on_chunk(phase_chunk(1))

    loads = [event[1] for event in events if event[0] == "vae-load"]
    assert loads == [
        ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"],
        ["MiniMaxH3VideoVAE"],
    ]


def test_a_failing_decode_propagates_and_leaves_the_vaes_loaded():
    """The DiT is not brought back for a forward the run will never reach."""
    events = []
    failure = RuntimeError("video vae decode failed")
    coordinator, _pipeline = make_coordinator(events, fail=failure)
    with pytest.raises(RuntimeError) as caught:
        coordinator.on_chunk(phase_chunk(0))

    assert caught.value is failure
    assert [kind for (kind, *_rest) in events] == ["vae-load", "decode"]
    assert coordinator.last_phase == "vae"
    assert coordinator.dit_loads == 0


def test_a_cancelled_chunk_leaves_the_run_in_the_vae_phase():
    events = []
    cancelled = consistency.SamplingCancelled("cancelled at on_chunk")
    coordinator, _pipeline = make_coordinator(events, fail=cancelled)
    with pytest.raises(consistency.SamplingCancelled):
        coordinator.on_chunk(phase_chunk(0))
    assert coordinator.dit_loads == 0
    assert coordinator.to_dict()["phase_swap_last_phase"] == "vae"


def test_a_full_length_run_swaps_once_per_chunk(wired):
    """192 frames is 12 chunks: 12 VAE phases and 11 DiT reloads, no more."""
    run_sampler(wired, latent=empty_av_latent(width=64, height=64, frames=192))

    dit_loads = [call for call in wired.loads if _is_dit_load(call[0])]
    vae_loads = [call for call in wired.loads if not _is_dit_load(call[0])]
    assert wired.built["pipeline"].chunks == 12
    assert len(vae_loads) == 12
    assert len(dit_loads) == 12  # the first one, plus one per non-last chunk

    record = wired.built["pipeline"].report().to_dict()
    assert record["chunks"] == 12


@pytest.mark.parametrize(
    "total_gib, free_gib", [(24, 21), (80, 76), (141, 137)]
)
def test_the_same_request_asks_for_the_same_thing_on_every_card(wired, total_gib, free_gib):
    """Nothing this node asks for is a function of how big the card is.

    If ``memory_required`` moved with the device, a workflow that ran on one box
    would be making a different request on another, and the difference would be
    this node second-guessing upstream's own measurement of free memory. The
    24 GiB path is exercised by making a device actually that small, not by
    arithmetic here.
    """
    wired.model_management.total = total_gib * 1024 ** 3
    wired.model_management._free = [free_gib * 1024 ** 3]
    run_sampler(wired)

    asked = [(_is_dit_load(models), memory) for models, memory, _f in wired.loads]
    plan = wired.built["pipeline"].memory_budget["detail"]["joint_offload_plan"]
    dit_required = plan["phases"]["dit"]["memory_required"]
    vae_required = plan["phases"]["vae"]["memory_required"]
    assert asked == [
        (True, dit_required),
        (False, vae_required),
        (True, dit_required),
        (False, vae_required),
        (True, dit_required),
        (False, vae_required),
    ]
    # the card's size is recorded as a fact, never used as an input to the plan
    assert plan["facts"]["total_bytes"] == total_gib * 1024 ** 3


def test_the_card_size_only_ever_reaches_the_report(wired):
    """Same request, two very different cards, byte-identical plans."""
    wired.model_management.total = 24 * 1024 ** 3
    wired.model_management._free = [21 * 1024 ** 3]
    run_sampler(wired)
    small = wired.built["pipeline"].memory_budget["detail"]["joint_offload_plan"]

    wired.model_management.total = 141 * 1024 ** 3
    wired.model_management._free = [137 * 1024 ** 3]
    run_sampler(wired)
    large = wired.built["pipeline"].memory_budget["detail"]["joint_offload_plan"]

    assert small["phases"] == large["phases"]
    assert small["planned_peak_bytes"] == large["planned_peak_bytes"]
    assert small["facts"]["total_bytes"] != large["facts"]["total_bytes"]


# --------------------------------------------------------------------------
# where the KV cache lives
# --------------------------------------------------------------------------


@dataclass
class SamplerConfigWithStorage:
    """``consistency.SamplerConfig`` as the cache lane is about to ship it."""

    steps: int = 4
    video_shift: float = 12.0
    audio_shift: float = 3.0
    sink: int = 2
    window: int = 2
    seed: int = 0
    kv_cache_storage: str = "cpu_pinned"


@pytest.mark.parametrize("storage", ["cpu_pinned", "cpu", "gpu"])
def test_the_storage_choice_reaches_the_sampler_config(wired, storage):
    wired.monkeypatch.setattr(consistency, "SamplerConfig", SamplerConfigWithStorage)
    run_sampler(wired, kv_cache_storage=storage)

    config = wired.rollout.kwargs["config"]
    assert config.kv_cache_storage == storage
    # ... and the budget describes the cache the rollout is actually building
    budget = wired.built["pipeline"].memory_budget
    assert budget["kv_cache_storage"] == storage
    if storage == "gpu":
        assert budget["kv_cache_bytes"] > 0 and budget["cpu_kv_peak_bytes"] == 0
    else:
        assert budget["kv_cache_bytes"] == 0 and budget["cpu_kv_peak_bytes"] > 0


def test_the_default_is_the_host_backed_cache(wired):
    # The node's own default, not run_sampler's: this is the value a workflow
    # that never touched the widget gets.
    wired.monkeypatch.setattr(consistency, "SamplerConfig", SamplerConfigWithStorage)
    latent = empty_av_latent()
    wired.monkeypatch.setattr(consistency, "sample_streaming", StubRollout(latent))
    wired.rollout = consistency.sample_streaming
    nodes.RAVENStreamingSampler().sample(
        model=FakePatcher(),
        positive=t2va_conditioning(),
        latent=latent,
        video_vae=video_vae(),
        audio_vae=audio_vae(),
        seed=0,
        unique_id="7",
    )
    assert wired.rollout.kwargs["config"].kv_cache_storage == "cpu_pinned"


def test_a_config_without_the_field_is_told_where_its_cache_will_be(wired, caplog):
    """An older cache lane still runs -- loudly, because the numbers change."""

    @dataclass
    class OldSamplerConfig:
        steps: int = 4
        video_shift: float = 12.0
        audio_shift: float = 3.0
        sink: int = 2
        window: int = 2
        seed: int = 0

    wired.monkeypatch.setattr(consistency, "SamplerConfig", OldSamplerConfig)
    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        out = run_sampler(wired, kv_cache_storage="cpu_pinned")

    assert out[1].shape[0] == 39  # the run is unaffected
    assert not hasattr(wired.rollout.kwargs["config"], "kv_cache_storage")
    assert any("kv_cache_storage" in r.getMessage() for r in caplog.records)
    assert any("28 GiB" in r.getMessage() for r in caplog.records)


def test_an_unknown_storage_is_refused_before_anything_loads(wired):
    with pytest.raises(nodes.NodeInputError, match="kv_cache_storage"):
        run_sampler(wired, kv_cache_storage="nvme")
    assert wired.loads == []


# --------------------------------------------------------------------------
# stacked official LoRAs
# --------------------------------------------------------------------------


def test_stacked_official_loras_get_the_bitwise_warning(wired, caplog):
    model = FakePatcher()
    model.patches = {"blocks.0.mlp.fc1.weight": [object()]}
    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        run_sampler(wired, model=model)
    message = next(m for m in (r.getMessage() for r in caplog.records) if "patched weight" in m)
    assert "last bits" in message
    # RAVEN's own adapter is an activation residual and is explicitly excluded
    assert "activation residual" in message


def test_a_raven_only_model_is_not_warned_about(wired, caplog):
    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        run_sampler(wired)
    assert not any("patched weight" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# after the run: what is evicted, and what is deliberately not decoded
# --------------------------------------------------------------------------


def test_the_whole_clip_audio_decode_never_runs(wired):
    """The 192-frame OOM, removed rather than worked around.

    ``vae_decode_audio`` over the whole clip is what died on the 24 GiB card --
    with the DiT *and* the video VAE already evicted, and then again inside
    Comfy's tiled retry, which is written for 4-D image latents and raised an
    ``IndexError`` on the H3 audio latent. The node does not call it at all any
    more: the AUDIO is the audio collector's waveform. The fixture's helper
    raises if it is ever reached, so this test is the tripwire.
    """
    out = run_sampler(wired)

    assert "audio-decode" not in wired.events
    assert wired.arguments["audio_vae"].decode_calls == []
    assert_audio(out[2])


def test_only_the_dit_is_evicted_after_the_run(wired):
    """Nothing decodes after finish, so nothing else needs the card."""
    run_sampler(wired)

    evicted = [call[0] for call in wired.model_management.unload_calls]
    assert evicted == [wired.arguments["model"]]
    # both VAEs stay: they were needed on every chunk and through the flush,
    # and evicting them here would only cost the next execution a reload
    assert wired.arguments["video_vae"].patcher not in evicted
    assert wired.arguments["audio_vae"].patcher not in evicted
    assert wired.model_management.blunt_calls == []

    order = [e for e in wired.events if e in ("finish", "unload")]
    assert order == ["finish", "unload"]


def test_no_final_audio_handover_is_recorded_any_more(wired, caplog):
    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        run_sampler(wired)
    messages = [record.getMessage() for record in caplog.records]

    record = next(m for m in messages if m.startswith("raven memory record:"))
    assert "final_decode_unload_seconds" in record  # the DiT handover stays
    # ... and the video VAE one is gone with the decode it was making room for
    assert "final_audio_unload_seconds" not in record
    assert not any("final audio handover" in message for message in messages)


def test_the_39_frame_outputs_are_unchanged_by_the_extra_handover(wired):
    latent = empty_av_latent(width=64, height=64, frames=39)
    out = run_sampler(wired, latent=latent)

    assert out[0] is latent
    assert out[1].shape == (39, 64, 64, 3)
    assert_audio(out[2])
    assert wired.built["pipeline"].report().collected_frames == 39
    assert wired.sender.bodies[-1]["reason"] == "complete"


    latent = empty_av_latent(width=64, height=64, frames=39)
    out = run_sampler(wired, latent=latent)

    assert out[0] is latent
    assert out[1].shape == (39, 64, 64, 3)
    assert_audio(out[2])
    assert wired.built["pipeline"].report().collected_frames == 39
    assert wired.sender.bodies[-1]["reason"] == "complete"


def test_the_audio_handover_is_the_pinned_api_with_the_pinned_arguments():
    model_management = FakeModelManagement()
    video = nodes.resolve_video_vae(video_vae())

    handover = nodes.prepare_final_audio_decode(video, model_management=model_management)

    assert model_management.unload_calls == [(video.patcher, True, False)]
    assert model_management.events == ["unload", "soft_empty_cache"]
    assert handover.strategy == nodes.FINAL_AUDIO_UNLOAD_STRATEGY
    assert "video_vae.patcher" in handover.strategy
    assert handover.phase == "final_audio"
    assert set(handover.to_dict()) == {
        "final_audio_unload_seconds",
        "final_audio_unload_strategy",
        "final_audio_free_before",
        "final_audio_free_after",
        "final_audio_freed_bytes",
        "final_audio_device",
    }


def test_no_video_vae_means_no_audio_handover():
    model_management = FakeModelManagement()
    assert nodes.prepare_final_audio_decode(None, model_management=model_management) is None
    assert model_management.unload_calls == []


def test_a_missing_unload_api_is_loud_for_the_audio_handover_too():
    class WithoutTheApi(FakeModelManagement):
        unload_model_and_clones = None

    video = nodes.resolve_video_vae(video_vae())
    with pytest.raises(nodes.NodeInputError, match="unload_model_and_clones"):
        nodes.prepare_final_audio_decode(video, model_management=WithoutTheApi())


def test_a_downstream_lora_clone_is_evicted_with_the_model_it_came_from(wired):
    """``LoraLoaderModelOnly`` hands us a clone; upstream matches by uuid.

    The node's job is to pass *the MODEL it was given* -- the clone -- and let
    ``unload_model_and_clones`` gather the rest of the family. Reaching past
    the socket for some "original" would evict the wrong set.
    """
    base = FakePatcher()
    clone = FakePatcher()
    clone.clone_base_uuid = base.clone_base_uuid = "shared-uuid"
    run_sampler(wired, model=clone)

    (evicted, _additional, _all_devices) = wired.model_management.unload_calls[0]
    assert evicted is clone
    assert evicted.clone_base_uuid == base.clone_base_uuid


def test_the_preview_is_complete_before_the_eviction_and_survives_it(wired):
    run_sampler(wired)

    # every preview message is already out, the stream is still 'finalizing'
    # (not ended), and the muxer was closed before the unload
    phases = [b["phase"] for b in wired.sender.bodies if b["event"] == "status"]
    assert phases == ["model_loading", "sampling", "finalizing"]
    assert wired.marks["phases_at_close"] == phases
    assert wired.events.index("finish") < wired.events.index("unload")
    assert wired.built["pipeline"].finished is True
    assert wired.sender.bodies[-1]["reason"] == "complete"


def test_a_failed_handover_stops_the_run_before_the_final_decode(wired):
    wired.model_management.fail = RuntimeError("model is in use")

    with pytest.raises(RuntimeError, match="model is in use"):
        run_sampler(wired)

    # no decode was attempted in the memory state that is known to OOM
    assert "video-decode" not in wired.events
    assert "audio-decode" not in wired.events
    terminal = wired.sender.bodies[-1]
    assert terminal["event"] == "end" and terminal["reason"] == "error"
    assert "model is in use" in terminal["message"]


def test_the_handover_timing_is_recorded_with_the_memory_budget(wired, caplog):
    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        run_sampler(wired)
    messages = [record.getMessage() for record in caplog.records]
    assert any("rollout reserve" in message for message in messages)
    assert any("final decode handover" in message for message in messages)
    assert any("unloaded the DiT and its clones" in message for message in messages)

    # one record carrying every half of the memory story
    record = next(m for m in messages if m.startswith("raven memory record:"))
    for key in (
        "kv_cache_bytes",
        "total_bytes",
        "final_decode_unload_seconds",
        "final_decode_unload_strategy",
        "kv_peak_rows",
        "joint_offload_plan",
        "phase_swap_chunks",
        "dit_phase_residency",
        "dit_loaded_bytes",
        "hard_cap_reference",
    ):
        assert key in record, key


def test_a_wrapped_load_closure_without_diagnostics_still_finishes(wired, caplog):
    """The measured 39-frame run died here, after producing all three outputs.

    An integration harness wrapped ``make_load_models`` to count loads. Its
    wrapper is a plain function, so the closure's ``residency`` attribute --
    a *diagnostic* hung off the returned callable -- was not on it, and reading
    it as if it were part of the contract raised ``AttributeError`` on the last
    line before the node returned. The record is optional; the outputs are not.
    """
    real = nodes.make_load_models
    wrapped_calls = []

    def instrumented(*args, **kwargs):
        closure = real(*args, **kwargs)

        def wrapper(models, memory_required=0, force_full_load=False):
            # deliberately copies no function attributes, exactly as the
            # harness did
            wrapped_calls.append(len(list(models)))
            return closure(
                models, memory_required=memory_required, force_full_load=force_full_load
            )

        return wrapper

    wired.monkeypatch.setattr(nodes, "make_load_models", instrumented)
    latent = empty_av_latent(width=64, height=64, frames=39)
    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        out = run_sampler(wired, latent=latent)

    # all three outputs, unchanged
    assert out[0] is latent
    assert out[1].shape == (39, 64, 64, 3)
    assert_audio(out[2])
    # the wrapper really was in the path, for every DiT phase
    assert wrapped_calls == [1, 1, 1]

    # the record is still written -- one line short, and that is all
    record = next(
        m
        for m in (r.getMessage() for r in caplog.records)
        if m.startswith("raven memory record:")
    )
    assert "dit_phase_residency" not in record
    assert "phase_swap_chunks" in record and "hard_cap_reference" in record


def test_the_diagnostics_are_read_as_optional_everywhere():
    """No attribute access on the load closure outside the closure itself."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(nodes.RAVENStreamingSampler.sample)))
    reads = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "load_models"
    ]
    assert reads == [], reads


def test_a_sampling_cancelled_rollout_reports_cancelled(wired):
    """The sampler's own cancellation, not ComfyUI's, still reads as cancelled."""
    failure = consistency.SamplingCancelled("cancelled at before_noise_forward")
    with pytest.raises(consistency.SamplingCancelled):
        run_sampler(wired, rollout=StubRollout(empty_av_latent(), fail=failure))

    terminal = wired.sender.bodies[-1]
    assert terminal["event"] == "end"
    assert terminal["reason"] == "cancelled"
    assert "SamplingCancelled" in terminal["message"]


def test_the_cancellation_names_cover_both_interrupt_paths():
    from raven_streaming.preview_session import (
        CANCELLATION_EXCEPTION_NAMES,
        _looks_like_cancellation,
    )

    assert {"InterruptProcessingException", "SamplingCancelled"} <= set(
        CANCELLATION_EXCEPTION_NAMES
    )
    assert _looks_like_cancellation(consistency.SamplingCancelled("stop"))
    assert _looks_like_cancellation(KeyboardInterrupt())
    assert not _looks_like_cancellation(RuntimeError("CUDA out of memory"))
    assert not _looks_like_cancellation(consistency.SamplerError("bad grid"))


def test_a_cancelled_run_reports_cancelled_and_re_raises(wired):
    class InterruptProcessingException(Exception):
        """Named the way ComfyUI names it; that is how it is classified."""

    failure = InterruptProcessingException()
    with pytest.raises(InterruptProcessingException) as caught:
        run_sampler(wired, rollout=StubRollout(empty_av_latent(), fail=failure))
    assert caught.value is failure

    terminal = wired.sender.bodies[-1]
    assert terminal["event"] == "end" and terminal["reason"] == "cancelled"
    # cancellation still releases the media lane
    assert wired.built["muxer"].closes == 1
    assert wired.built["pipeline"].finished is True


def test_a_failed_run_reports_error_and_re_raises(wired):
    failure = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_sampler(wired, rollout=StubRollout(empty_av_latent(), fail=failure))
    terminal = wired.sender.bodies[-1]
    assert terminal["event"] == "end" and terminal["reason"] == "error"
    assert "CUDA out of memory" in terminal["message"]
    assert wired.built["muxer"].closes == 1


def test_a_preview_that_cannot_start_does_not_stop_the_run(wired):
    """No PyAV: no audio decoder, no muxer -- and still a full IMAGE."""

    def fake_build(
        *, video_vae, audio_vae, config, sink=None, log=None, muxer=None, memory_budget=None
    ):
        wired.built["pipeline"] = sp.StreamingPipeline(
            config=config,
            video_decoder=make_collector(),
            # still built: it is the AUDIO output, not part of the preview
            audio_decoder=make_audio_collector(config),
            muxer=None,
            sink=sink,
            log=log,
            memory_budget=memory_budget,
            preview_disabled_reason="preview unavailable: EncoderUnavailable: no libx264",
        )
        return wired.built["pipeline"]

    wired.monkeypatch.setattr(nodes.pipeline_mod, "build_media_pipeline", fake_build)
    latent = empty_av_latent(width=64, height=64, frames=39)
    out = run_sampler(wired, latent=latent)

    assert out[0] is latent and out[1].shape == (39, 64, 64, 3)
    assert wired.built["pipeline"].report().collected_frames == 39
    bodies = wired.sender.bodies
    # the session still opened, still explained itself, and still ended cleanly
    assert bodies[0]["event"] == "open"
    assert "preview unavailable" in bodies[1]["message"]
    assert bodies[-1]["reason"] == "complete"


def test_a_collector_that_cannot_be_built_fails_the_run(wired):
    """The collector is the output, so its failure is the run's failure."""

    def explode(**kwargs):
        raise RuntimeError("the video VAE is unusable")

    wired.monkeypatch.setattr(nodes.pipeline_mod, "build_media_pipeline", explode)
    with pytest.raises(RuntimeError, match="the video VAE is unusable"):
        run_sampler(wired)

    # it failed before the stream was opened, so the client was never told
    # about a session that produced nothing; nothing was loaded either
    assert wired.sender.bodies == []
    assert wired.loads == []


def test_a_preview_that_dies_mid_run_does_not_stop_the_run(wired):
    latent = empty_av_latent(width=64, height=64, frames=39)

    class DyingMuxer(StubMuxer):
        def write_video_frame(self, image, force_keyframe=None):
            raise RuntimeError("the encoder died mid-clip")

    def fake_build(
        *, video_vae, audio_vae, config, sink=None, log=None, muxer=None, memory_budget=None
    ):
        wired.built["muxer"] = DyingMuxer()
        wired.built["pipeline"] = sp.StreamingPipeline(
            config=config,
            video_decoder=make_collector(),
            audio_decoder=make_audio_collector(config),
            muxer=wired.built["muxer"],
            sink=sink,
            log=log,
            memory_budget=memory_budget,
        )
        return wired.built["pipeline"]

    wired.monkeypatch.setattr(nodes.pipeline_mod, "build_media_pipeline", fake_build)
    out = run_sampler(wired, latent=latent)

    assert out[0] is latent and out[1].shape == (39, 64, 64, 3)
    assert wired.built["pipeline"].preview_disabled is True
    assert wired.built["pipeline"].report().collected_frames == 39
    assert wired.sender.bodies[-1]["reason"] == "complete"


def test_a_collector_that_dies_mid_run_fails_the_run(wired):
    class DyingCollector:
        def push(self, z):
            raise RuntimeError("video vae decode failed")

        def finish(self):
            return []

    def fake_build(
        *, video_vae, audio_vae, config, sink=None, log=None, muxer=None, memory_budget=None
    ):
        wired.built["muxer"] = StubMuxer()
        wired.built["pipeline"] = sp.StreamingPipeline(
            config=config,
            video_decoder=DyingCollector(),
            audio_decoder=StubDecoder(),
            muxer=wired.built["muxer"],
            sink=sink,
            log=log,
        )
        return wired.built["pipeline"]

    wired.monkeypatch.setattr(nodes.pipeline_mod, "build_media_pipeline", fake_build)
    with pytest.raises(RuntimeError, match="video vae decode failed"):
        run_sampler(wired)

    # no output was produced, the session says error, and the buffers are gone
    assert "audio-decode" not in wired.events
    assert "unload" not in wired.events
    assert wired.sender.bodies[-1]["reason"] == "error"
    with pytest.raises(sp.PipelineError):
        wired.built["pipeline"].finalize_image()


def test_no_preview_at_all_still_samples(wired):
    import raven_streaming.preview as preview_mod

    def explode(*args, **kwargs):
        raise RuntimeError("no server")

    wired.monkeypatch.setattr(preview_mod, "install", explode)
    latent = empty_av_latent(width=64, height=64, frames=39)
    out = run_sampler(wired, latent=latent)

    assert out[0] is latent and out[1].shape == (39, 64, 64, 3)
    assert wired.sender.bodies == []
    # no session, so no preview lane at all -- but the collector still ran,
    # because it is the IMAGE output rather than a preview convenience
    pipeline = wired.built["pipeline"]
    assert pipeline.preview_disabled is True
    assert pipeline.report().collected_frames == 39
    assert wired.rollout.kwargs is not None  # ... and the rollout still ran


def test_the_pipeline_canvas_comes_from_the_latent(wired):
    run_sampler(wired, latent=empty_av_latent(width=96, height=64, frames=56))
    config = wired.built["pipeline"].config
    assert (config.width, config.height, config.frames) == (96, 64, 56)
    assert float(config.fps) == 24.0 and config.sample_rate == 32000


@pytest.mark.parametrize("frames", [22, 192, 362])
def test_the_supported_frame_range_is_accepted_end_to_end(wired, frames):
    latent = empty_av_latent(width=64, height=64, frames=frames)
    out = run_sampler(wired, latent=latent)
    assert out[1].shape == (frames, 64, 64, 3)
    assert wired.built["pipeline"].config.frames == frames


def test_a_frame_count_off_the_grid_is_refused_before_anything_loads(wired):
    latent = empty_av_latent(width=64, height=64, frames=39)
    # 39 frames is 12 video latents; hand it 13 and the grid no longer closes
    video, audio = latent["samples"].tensors
    latent["samples"] = NestedTensor((torch.zeros(1, 24, 13, 4, 4), audio))
    with pytest.raises(Exception) as caught:
        run_sampler(wired, latent=latent)
    assert "5k + 2" in str(caught.value)
    assert wired.loads == []
    assert wired.sender.bodies == []


def test_a_non_empty_latent_is_refused(wired):
    latent = empty_av_latent(width=64, height=64, frames=39)
    latent["samples"].tensors[0][:] = 0.5
    with pytest.raises(Exception, match="not empty"):
        run_sampler(wired, latent=latent)
    assert wired.loads == []


def test_a_negative_conditioning_pair_is_refused(wired):
    positive = t2va_conditioning() + t2va_conditioning()
    with pytest.raises(Exception, match="no CFG, no negative"):
        run_sampler(wired, positive=positive)
    assert wired.loads == []
