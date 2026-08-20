"""End-to-end causal lane: text prefill, read-only denoise, clean fill, eviction.

These are the behavioural claims M2 rests on:

* the text is cache chunk 0 and is written alone;
* a noise (denoise) forward never touches the cache and is reproducible;
* a clean forward writes exactly one chunk, all layers, once;
* the next chunk actually *sees* what the clean fill wrote;
* an evicted chunk is provably invisible -- changing it changes nothing;
* the repo timestep convention maps as specified (clean == 0.999 with the
  ``0.999 * x0 + 0.001 * eps`` augmentation).

Requires a local ComfyUI checkout (see ``tests/conftest.py``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conftest  # noqa: E402

# ``raven_streaming.causal_model`` imports ComfyUI at module scope (it subclasses
# the official model), so the checkout has to be on sys.path before collection.
_UPSTREAM = conftest.find_upstream_comfyui()
if _UPSTREAM is None:  # pragma: no cover - environment without a checkout
    pytest.skip("No local ComfyUI checkout found", allow_module_level=True)
conftest.add_to_sys_path(_UPSTREAM)

from raven_streaming.cache import CacheError, ChunkKVCache  # noqa: E402
from raven_streaming.causal_model import (  # noqa: E402
    CLEAN_TIMESTEP_AUDIO,
    CLEAN_TIMESTEP_VIDEO,
    CausalModelError,
    velocity_to_x0,
)
from test_causal_common import (  # noqa: E402
    TINY_CONFIG,
    random_inputs,
    tiny_layout,
    tiny_models,  # noqa: F401  (fixture re-export)
)

NUM_LAYERS = TINY_CONFIG["num_layers"]


def _noise(model, layout, video, audio, index, cache, sigma=0.6):
    return model.forward_chunk(
        video_latent=layout.video_chunk_latent(video, index),
        audio_latent=layout.audio_chunk_latent(audio, index),
        layout=layout,
        chunk_index=index,
        cache=cache,
        role="noise",
        video_sigma=sigma,
        audio_sigma=model.audio_sigma_from_video(sigma),
    )


def _clean(model, layout, video, audio, index, cache, eps_seed=7):
    generator = torch.Generator().manual_seed(eps_seed)
    v = layout.video_chunk_latent(video, index)
    a = layout.audio_chunk_latent(audio, index)
    return model.forward_chunk(
        video_latent=v,
        audio_latent=a,
        layout=layout,
        chunk_index=index,
        cache=cache,
        role="clean",
        video_eps=torch.randn(v.shape, generator=generator),
        audio_eps=torch.randn(a.shape, generator=generator),
    )


# --- text prefill ------------------------------------------------------------


def test_text_prefill_is_cache_chunk_zero(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    _, _, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)

    rows = causal.prefill_text(context, cache=cache)
    assert rows == layout.text_len
    assert cache.committed_chunks == 1
    assert cache.retained_indices == [0]
    assert cache.chunk_lens == [layout.text_len]
    for layer in range(NUM_LAYERS):
        keys, values = cache.retained(layer)
        assert keys.shape == (layout.text_len, TINY_CONFIG["num_attention_heads"],
                              TINY_CONFIG["attention_head_dim"])
        assert values.shape == keys.shape


def test_text_prefill_accepts_prerefined_states(tiny_models):
    official, causal = tiny_models
    layout = tiny_layout()
    _, _, context = random_inputs(layout)
    refined = official.preprocess_text_embeds(context)

    raw_cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    refined_cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    causal.prefill_text(context, cache=raw_cache)
    causal.prefill_text(refined, cache=refined_cache)
    for layer in range(NUM_LAYERS):
        assert torch.equal(raw_cache.retained(layer)[0], refined_cache.retained(layer)[0])


def test_text_prefill_twice_is_rejected(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    _, _, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    causal.prefill_text(context, cache=cache)
    with pytest.raises(CausalModelError, match="chunk 0"):
        causal.prefill_text(context, cache=cache)


def test_chunk_before_prefill_is_rejected(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, _ = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    with pytest.raises(CausalModelError, match="committed cache chunk"):
        _noise(causal, layout, video, audio, 0, cache)


def test_chunks_must_arrive_in_order(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    causal.prefill_text(context, cache=cache)
    with pytest.raises(CausalModelError, match="committed cache chunk"):
        _noise(causal, layout, video, audio, 1, cache)


# --- read-only denoise -------------------------------------------------------


def test_noise_forward_is_read_only_and_reproducible(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    causal.prefill_text(context, cache=cache)

    before = [cache.retained(layer)[0].clone() for layer in range(NUM_LAYERS)]
    first = _noise(causal, layout, video, audio, 0, cache)
    second = _noise(causal, layout, video, audio, 0, cache)

    assert cache.committed_chunks == 1
    assert cache.retained_rows == layout.text_len
    assert not cache.has_pending
    for layer in range(NUM_LAYERS):
        assert torch.equal(cache.retained(layer)[0], before[layer])
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_noise_forward_may_not_write_the_cache(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=None)
    causal.prefill_text(context, cache=cache)
    with pytest.raises(CausalModelError, match="only a clean chunk"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="noise",
            video_sigma=0.5, audio_sigma=0.5, update_cache=True,
        )
    assert cache.committed_chunks == 1


def test_chunk_output_shapes_follow_the_layout(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=2, window=None)
    causal.prefill_text(context, cache=cache)
    for index, chunk in enumerate(layout.chunks):
        v, a = _noise(causal, layout, video, audio, index, cache)
        assert tuple(v.shape) == (1, TINY_CONFIG["latents_dim"], chunk.video_latents,
                                  layout.latent_h, layout.latent_w)
        assert tuple(a.shape) == (1, TINY_CONFIG["audio_latents_dim"], 2, chunk.audio_latents)
        if index < layout.num_chunks - 1:
            _clean(causal, layout, video, audio, index, cache)


# --- clean fill --------------------------------------------------------------


def test_clean_forward_commits_exactly_one_chunk(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=cache)

    _clean(causal, layout, video, audio, 0, cache)
    assert cache.committed_chunks == 2
    assert cache.chunk_lens == [layout.text_len, layout.chunks[0].rows]
    assert not cache.has_pending
    for layer in range(NUM_LAYERS):
        assert cache.retained(layer)[0].shape[0] == layout.text_len + layout.chunks[0].rows


def test_committed_kv_survives_later_forwards(tiny_models):
    """The staged V is handed over uncopied when a prefix exists; nothing may touch it."""
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=cache)
    _clean(causal, layout, video, audio, 0, cache)

    snapshot = [(k.clone(), v.clone()) for k, v in
                (cache.retained(layer) for layer in range(NUM_LAYERS))]
    _noise(causal, layout, video, audio, 1, cache)
    _clean(causal, layout, video, audio, 1, cache)
    _noise(causal, layout, video, audio, 2, cache)

    for layer, (keys, values) in enumerate(snapshot):
        current_k, current_v = cache.retained(layer)
        assert torch.equal(current_k[: keys.shape[0]], keys)
        assert torch.equal(current_v[: values.shape[0]], values)


def test_clean_forward_requires_eps_and_refuses_sigma(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=cache)

    with pytest.raises(CausalModelError, match="needs video_eps"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="clean",
        )
    with pytest.raises(CausalModelError, match="takes no sigma"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="clean",
            video_sigma=0.5, audio_sigma=0.5,
            video_eps=torch.zeros_like(layout.video_chunk_latent(video, 0)),
            audio_eps=torch.zeros_like(layout.audio_chunk_latent(audio, 0)),
        )
    assert cache.committed_chunks == 1


def test_noise_forward_requires_both_sigmas(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=cache)
    with pytest.raises(CausalModelError, match="video_sigma and audio_sigma"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="noise",
            video_sigma=0.5,
        )


def test_clean_timestep_equals_a_noise_pass_at_sigma_0_001(tiny_models):
    """The repo's ``t = 0`` maps to H3 ``0.999`` with ``t*x0 + (1-t)*eps``.

    Written in the module's own float32 arithmetic, because that is RAVEN's:
    the condition timestep is the fp32 ``0.999`` and the eps coefficient is
    ``1 - fp32(0.999)``, which is ``0.00099998713`` and not ``fp32(0.001)``.
    The repo sigma that maps back onto it is that same ``1 - t`` -- exactly, by
    Sterbenz: ``1 - t`` is representable for ``t`` near 1, so the round trip
    ``1 - (1 - t) == t`` holds in fp32 and the two passes really do see one
    timestep.
    """
    from raven_streaming.causal_model import _fp32_one_minus, _fp32_scalar

    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    generator = torch.Generator().manual_seed(7)
    x0_v = layout.video_chunk_latent(video, 0)
    x0_a = layout.audio_chunk_latent(audio, 0)
    eps_v = torch.randn(x0_v.shape, generator=generator)
    eps_a = torch.randn(x0_a.shape, generator=generator)

    t_v = _fp32_scalar(CLEAN_TIMESTEP_VIDEO)
    t_a = _fp32_scalar(CLEAN_TIMESTEP_AUDIO)
    sigma_v, sigma_a = _fp32_one_minus(t_v), _fp32_one_minus(t_a)
    assert _fp32_one_minus(sigma_v) == t_v and _fp32_one_minus(sigma_a) == t_a

    clean_cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    noise_cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=clean_cache)
    causal.prefill_text(context, cache=noise_cache)

    from_clean = causal.forward_chunk(
        video_latent=x0_v, audio_latent=x0_a, layout=layout, chunk_index=0,
        cache=clean_cache, role="clean", video_eps=eps_v, audio_eps=eps_a,
    )
    from_noise = causal.forward_chunk(
        video_latent=t_v * x0_v + sigma_v * eps_v,
        audio_latent=t_a * x0_a + sigma_a * eps_a,
        layout=layout, chunk_index=0, cache=noise_cache, role="noise",
        video_sigma=sigma_v, audio_sigma=sigma_a,
    )
    for a, b in zip(from_clean, from_noise):
        assert torch.equal(a, b)
    # ... and the write is what distinguishes them
    assert clean_cache.committed_chunks == 2
    assert noise_cache.committed_chunks == 1

    # the double spelling of the same formula is a *different* mix
    assert (1.0 - CLEAN_TIMESTEP_VIDEO) != sigma_v


def test_streams_carry_independent_timesteps(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=cache)

    common = dict(
        video_latent=layout.video_chunk_latent(video, 0),
        audio_latent=layout.audio_chunk_latent(audio, 0),
        layout=layout, chunk_index=0, cache=cache, role="noise",
    )
    base = causal.forward_chunk(video_sigma=0.6, audio_sigma=0.3, **common)
    other_audio = causal.forward_chunk(video_sigma=0.6, audio_sigma=0.9, **common)
    same = causal.forward_chunk(video_sigma=0.6, audio_sigma=0.3, **common)

    assert torch.equal(base[0], same[0]) and torch.equal(base[1], same[1])
    assert not torch.equal(base[1], other_audio[1])
    # a different audio timestep also moves video, through attention
    assert not torch.equal(base[0], other_audio[0])


def test_audio_sigma_helper_matches_the_official_shift_rule(tiny_models, comfyui_on_syspath):
    from comfy.ldm.minimax.model import time_shift_sigma

    _, causal = tiny_models
    expected = float(time_shift_sigma(torch.tensor(0.6), causal.sigma_shift_video,
                                      causal.sigma_shift_audio))
    assert causal.audio_sigma_from_video(0.6) == pytest.approx(expected, rel=1e-6)


# --- visibility --------------------------------------------------------------


def test_next_chunk_sees_the_clean_fill(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)

    with_history = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=with_history)
    _clean(causal, layout, video, audio, 0, with_history)
    seen = _noise(causal, layout, video, audio, 1, with_history)

    # same rollout, but chunk 0's context carries different content
    other = ChunkKVCache(NUM_LAYERS, sink=4, window=None)
    causal.prefill_text(context, cache=other)
    _clean(causal, layout, video * -1.0, audio * -1.0, 0, other)
    different = _noise(causal, layout, video, audio, 1, other)

    assert not torch.equal(seen[0], different[0])
    assert not torch.equal(seen[1], different[1])


def test_evicted_history_is_provably_invisible(tiny_models):
    # sink=1 (text only) + window=0: chunk 0's clean fill is evicted the moment
    # it is committed, so chunk 1 must not depend on it at all.
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)

    outputs = []
    for scale in (1.0, -3.0):
        cache = ChunkKVCache(NUM_LAYERS, sink=1, window=0)
        causal.prefill_text(context, cache=cache)
        _clean(causal, layout, video * scale, audio * scale, 0, cache)
        assert cache.retained_indices == [0]
        outputs.append(_noise(causal, layout, video, audio, 1, cache))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


def test_window_retains_the_recent_chunks_only(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout(frames=90)  # 6 chunks
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=1, window=1)
    causal.prefill_text(context, cache=cache)

    expected_rows = layout.text_len
    for index in range(layout.num_chunks - 1):
        _noise(causal, layout, video, audio, index, cache)
        _clean(causal, layout, video, audio, index, cache)
        expected_rows = layout.text_len + layout.chunks[index].rows
        assert cache.retained_indices == [0, index + 1]
        assert cache.retained_rows == expected_rows


def test_full_rollout_row_accounting_with_no_eviction(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=0, window=None)
    causal.prefill_text(context, cache=cache)

    rows = layout.text_len
    for index in range(layout.num_chunks):
        _noise(causal, layout, video, audio, index, cache)
        if index == layout.num_chunks - 1:
            break  # the last chunk needs no clean fill: nothing reads it
        _clean(causal, layout, video, audio, index, cache)
        rows += layout.chunks[index].rows
        assert cache.retained_rows == rows
    assert cache.committed_chunks == layout.num_chunks


# --- misc contract -----------------------------------------------------------


def test_velocity_to_x0_round_trips(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=2, window=None)
    causal.prefill_text(context, cache=cache)
    velocity = _noise(causal, layout, video, audio, 0, cache, sigma=0.4)
    x_t = layout.video_chunk_latent(video, 0)
    x0 = velocity_to_x0(x_t, velocity[0], 1.0 - 0.4)
    assert torch.allclose(x0, x_t + 0.4 * velocity[0])


def test_chunk_latent_shape_is_checked(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=2, window=None)
    causal.prefill_text(context, cache=cache)
    with pytest.raises(CausalModelError, match="video latent"):
        causal.forward_chunk(
            video_latent=video,  # whole clip instead of the chunk
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="noise",
            video_sigma=0.5, audio_sigma=0.5,
        )
    with pytest.raises(CausalModelError, match="outside"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=layout.num_chunks, cache=cache,
            role="noise", video_sigma=0.5, audio_sigma=0.5,
        )


def test_chunk_without_a_cache_runs_context_free(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, _ = random_inputs(layout)
    out = _noise(causal, layout, video, audio, 0, None)
    assert tuple(out[0].shape[2:]) == (layout.chunks[0].video_latents,
                                       layout.latent_h, layout.latent_w)


def test_cache_layer_count_must_match_the_model(tiny_models):
    _, causal = tiny_models
    layout = tiny_layout()
    _, _, context = random_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS - 1, sink=1, window=None)
    with pytest.raises(CacheError):
        causal.prefill_text(context, cache=cache)
