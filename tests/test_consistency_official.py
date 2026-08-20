"""The M3 sampler against the *real* pinned upstream types, on CPU.

The fake-driven tests pin the loop's ordering; these pin that the loop still
fits the objects ComfyUI actually hands it:

* the official ``comfy.nested_tensor.NestedTensor`` and the official empty-AV
  geometry (``comfy.latent_formats.MiniMaxH3AV.fix_empty_latent``, the same
  code path ``EmptyMiniMaxH3LatentAV`` produces) survive a full round trip;
* a real ``comfy.model_patcher.ModelPatcher`` passes ``resolve_model`` and the
  stock bidirectional ``MiniMaxH3Model`` does not;
* the whole rollout runs end to end against the real
  :class:`RavenCausalMiniMaxH3Model` and the real
  :class:`~raven_streaming.cache.ChunkKVCache` -- text prefill, cached noise
  forwards, clean fills, eviction -- on a tiny CPU model with the official
  channel counts.

Opt-in: skipped without a local ComfyUI checkout (see ``tests/conftest.py``).
Nothing here needs a GPU or a checkpoint.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conftest  # noqa: E402

_UPSTREAM = conftest.find_upstream_comfyui()
if _UPSTREAM is None:  # pragma: no cover - environment without a checkout
    pytest.skip("No local ComfyUI checkout found", allow_module_level=True)
conftest.add_to_sys_path(_UPSTREAM)

import comfy.latent_formats  # noqa: E402
import comfy.ldm.minimax.model  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.nested_tensor  # noqa: E402

from raven_streaming.consistency import (  # noqa: E402
    SamplerConfig,
    sample_streaming,
)
from raven_streaming.contracts import (  # noqa: E402
    ContractError,
    build_output_latent,
    parse_latent,
    resolve_model,
)
from test_causal_common import (  # noqa: E402
    TINY_CONFIG,
    tiny_models,  # noqa: F401  (fixture re-export)
)
from test_consistency_common import LoadRecorder  # noqa: E402

#: k = 2 -> 12 video latents -> chunks (0,5) (5,10) (10,12)
FRAMES = 39
WIDTH = 64
HEIGHT = 64
TEXT_LEN = 5


# --------------------------------------------------------------------------
# official latent geometry
# --------------------------------------------------------------------------
def official_empty_latent(frames=FRAMES, width=WIDTH, height=HEIGHT):
    """The official empty AV latent, built the way upstream builds it.

    ``fix_empty_latent`` derives the audio length from the video token count
    through ``FRAME_PER_TOKEN``/``FRAME_RESCALE``, i.e. by a different route
    than :func:`raven_streaming.layout.audio_latent_t`. Agreement between the
    two is the point of using it here.
    """
    latent_t = ((frames - 5) // 17) * 5 + 2
    packed = torch.zeros(1, 32, latent_t, height // 16, width // 16)
    samples = comfy.latent_formats.MiniMaxH3AV().fix_empty_latent(packed)
    return {"samples": samples}


def test_official_empty_latent_parses():
    request = parse_latent(official_empty_latent())
    assert request.frames == FRAMES
    assert (request.width, request.height) == (WIDTH, HEIGHT)
    assert request.latent_t == 12
    assert request.nested_cls is comfy.nested_tensor.NestedTensor


def test_official_audio_length_agrees_with_our_clock():
    for frames in (22, 39, 124, 192):
        latent = official_empty_latent(frames=frames, width=64, height=64)
        request = parse_latent(latent, warn_experimental=False)
        assert request.frames == frames
        assert request.audio_t == latent["samples"].tensors[1].shape[-1]


def test_nested_tensor_round_trip_through_the_output_latent():
    latent = official_empty_latent()
    request = parse_latent(latent)
    video = torch.randn(1, 24, request.latent_t, request.latent_h, request.latent_w)
    audio = torch.randn(1, 32, 2, request.audio_t)

    out = build_output_latent(request, video, audio)
    samples = out["samples"]
    assert isinstance(samples, comfy.nested_tensor.NestedTensor)
    assert samples.is_nested is True
    assert samples.shape == video.shape          # NestedTensor reports stream 0
    assert samples.device == video.device
    assert samples.dtype == video.dtype
    streams = samples.unbind()
    assert torch.equal(streams[0], video)
    assert torch.equal(streams[1], audio)

    # the round trip survives the operations the rest of the graph performs
    moved = samples.to(dtype=torch.float16)
    assert moved.unbind()[1].dtype == torch.float16


# --------------------------------------------------------------------------
# official ModelPatcher
# --------------------------------------------------------------------------
class _BaseModelStub(torch.nn.Module):
    """Stands in for ``comfy.model_base.MiniMaxH3``: it owns the DiT."""

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model


def official_patcher(diffusion_model):
    return comfy.model_patcher.ModelPatcher(
        _BaseModelStub(diffusion_model),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def test_real_model_patcher_with_the_causal_dit_resolves(tiny_models):
    _official, causal = tiny_models
    resolved = resolve_model(official_patcher(causal))
    assert resolved.diffusion_model is causal
    assert resolved.num_layers == TINY_CONFIG["num_layers"]
    assert resolved.transformer_options == {}
    assert resolved.load_device == torch.device("cpu")


def test_real_model_patcher_with_the_stock_dit_is_rejected(tiny_models):
    official, _causal = tiny_models
    assert isinstance(official, comfy.ldm.minimax.model.MiniMaxH3Model)
    with pytest.raises(ContractError, match="prefill_text"):
        resolve_model(official_patcher(official))


def test_real_dit_block_replacement_is_rejected(tiny_models):
    _official, causal = tiny_models
    patcher = official_patcher(causal)
    patcher.set_model_patch_replace(lambda *a, **k: None, "dit", "double_block", 0)
    with pytest.raises(ContractError, match="patches_replace"):
        resolve_model(patcher)


def test_real_unet_function_wrapper_is_rejected(tiny_models):
    _official, causal = tiny_models
    patcher = official_patcher(causal)
    patcher.set_model_unet_function_wrapper(lambda *a, **k: None)
    with pytest.raises(ContractError, match="model_function_wrapper"):
        resolve_model(patcher)


def test_real_patcher_wrapper_is_rejected(tiny_models):
    _official, causal = tiny_models
    patcher = official_patcher(causal)
    patcher.add_wrapper("apply_model", lambda *a, **k: None)
    with pytest.raises(ContractError, match="patcher wrappers"):
        resolve_model(patcher)


# --------------------------------------------------------------------------
# end to end against the real causal model
# --------------------------------------------------------------------------
def official_conditioning(text_len=TEXT_LEN, seed=3):
    generator = torch.Generator().manual_seed(seed)
    cond = torch.randn(1, text_len, TINY_CONFIG["text_dim"], generator=generator)
    return [[cond, {"pooled_output": None,
                    "minimax_token_tags": torch.ones(text_len, dtype=torch.long)}]]


def run_official(causal, *, config, on_chunk=None):
    load = LoadRecorder()
    # No compute_dtype: the sampler must read it off the DiT itself, so that the
    # text prefill and the chunk forwards fill and read the cache in one dtype.
    result = sample_streaming(
        model=official_patcher(causal),
        positive=official_conditioning(),
        latent=official_empty_latent(),
        config=config,
        on_chunk=on_chunk,
        load_models=load,
    )
    return result, load


def test_end_to_end_rollout_against_the_real_causal_model(tiny_models):
    _official, causal = tiny_models
    chunks = []
    result, load = run_official(
        causal, config=SamplerConfig(steps=2, seed=20260807), on_chunk=chunks.append
    )

    assert len(load.calls) == 1
    assert result.num_chunks == 3
    assert result.noise_forwards == 6            # 3 chunks x 2 steps
    assert result.clean_forwards == 2            # never the last chunk
    assert [c.index for c in chunks] == [0, 1, 2]

    video, audio = result.latent["samples"].unbind()
    assert tuple(video.shape) == (1, 24, 12, 4, 4)
    assert tuple(audio.shape) == (1, 32, 2, result.layout.audio_t)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    # a real rollout writes every row: no chunk was left at zero
    for chunk in result.layout.chunks:
        assert not torch.equal(
            video[:, :, chunk.video_start:chunk.video_stop],
            torch.zeros_like(video[:, :, chunk.video_start:chunk.video_stop]),
        )


def test_end_to_end_is_deterministic(tiny_models):
    _official, causal = tiny_models
    config = SamplerConfig(steps=2, seed=99)
    first, _ = run_official(causal, config=config)
    second, _ = run_official(causal, config=config)
    a = first.latent["samples"].unbind()
    b = second.latent["samples"].unbind()
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_end_to_end_seed_actually_matters(tiny_models):
    _official, causal = tiny_models
    first, _ = run_official(causal, config=SamplerConfig(steps=2, seed=1))
    second, _ = run_official(causal, config=SamplerConfig(steps=2, seed=2))
    assert not torch.equal(
        first.latent["samples"].unbind()[0], second.latent["samples"].unbind()[0]
    )


def test_the_last_chunk_leaves_the_cache_one_shorter(tiny_models):
    """Three chunks, two clean fills, so the cache ever holds text + 2 records."""
    _official, causal = tiny_models
    result, _ = run_official(causal, config=SamplerConfig(steps=1, seed=4))
    assert result.clean_forwards == result.num_chunks - 1


def test_conditioning_width_decides_whether_the_refiner_runs(tiny_models):
    """A pre-refined [1, L, hidden] context is accepted as-is by prefill_text."""
    _official, causal = tiny_models
    generator = torch.Generator().manual_seed(11)
    refined = torch.randn(1, TEXT_LEN, TINY_CONFIG["hidden_size"], generator=generator)
    result = sample_streaming(
        model=official_patcher(causal),
        positive=[[refined, {}]],
        latent=official_empty_latent(),
        config=SamplerConfig(steps=1, seed=5),
        load_models=LoadRecorder(),
        compute_dtype=torch.float32,
    )
    video, _audio = result.latent["samples"].unbind()
    assert torch.isfinite(video).all()
