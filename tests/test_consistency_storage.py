"""``kv_cache_storage``: the sampler's residency knob, end to end on the fakes.

The claim is narrow and load-bearing: the storage a rollout picks changes where
the K/V live and what it reports, and **nothing else**. Same seed, same draws,
same chunks, bit for bit.

The default is pinned here too: ``SamplerConfig()`` must still mean ``'gpu'``,
so a caller written before this knob existed keeps its exact memory profile.

No ComfyUI, no weights, no GPU -- the fakes from ``test_consistency_common``
drive a real :class:`raven_streaming.cache.ChunkKVCache`, which is the object
under test.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import cache as cache_mod  # noqa: E402
from raven_streaming import consistency  # noqa: E402
from raven_streaming.consistency import (  # noqa: E402
    DEFAULT_KV_CACHE_STORAGE,
    KV_CACHE_STORAGES,
    SamplerConfig,
    SamplerError,
    SamplingCancelled,
    sample_streaming,
)
from test_consistency_common import (  # noqa: E402
    FakeCausalDiT,
    FakePatcher,
    LoadRecorder,
    empty_av_latent,
    text_conditioning,
)


@pytest.fixture
def force_pinned(monkeypatch):
    """Same seam as the cache tests: make the pinned branch reachable here."""
    monkeypatch.setattr(
        cache_mod, "_empty_pinned",
        lambda shape, dtype: torch.empty(tuple(shape), dtype=dtype, device="cpu"))
    monkeypatch.setattr(cache_mod, "_is_pinned", lambda tensor: True)


def run(storage=DEFAULT_KV_CACHE_STORAGE, *, frames=39, seed=1234, steps=2,
        cancel_check=None, collect=None, **config_kwargs):
    dit = FakeCausalDiT(num_layers=2)
    on_chunk = None
    if collect is not None:
        def on_chunk(chunk):
            collect.append((chunk.index, chunk.video_x0.clone(),
                            chunk.audio_x0.clone()))
    result = sample_streaming(
        model=FakePatcher(dit),
        positive=text_conditioning(),
        latent=empty_av_latent(frames=frames),
        config=SamplerConfig(steps=steps, seed=seed, kv_cache_storage=storage,
                             **config_kwargs),
        on_chunk=on_chunk,
        cancel_check=cancel_check,
        load_models=LoadRecorder(),
        require_upstream_class=False,
    )
    return result, dit


# --- the config knob ----------------------------------------------------------


def test_the_default_is_gpu_so_existing_callers_are_unchanged():
    assert DEFAULT_KV_CACHE_STORAGE == "gpu"
    assert SamplerConfig().kv_cache_storage == "gpu"
    assert KV_CACHE_STORAGES == ("gpu", "cpu_pinned", "cpu")


@pytest.mark.parametrize("storage", KV_CACHE_STORAGES)
def test_every_storage_is_accepted_and_described(storage):
    config = SamplerConfig(kv_cache_storage=storage)
    assert config.describe()["kv_cache_storage"] == storage


def test_an_unknown_storage_is_rejected_before_anything_is_loaded():
    with pytest.raises(SamplerError, match="kv_cache_storage"):
        SamplerConfig(kv_cache_storage="ssd")


# --- the rollout uses it ------------------------------------------------------


@pytest.mark.parametrize("storage", ("cpu_pinned", "cpu"))
def test_the_rollout_builds_the_cache_with_the_configured_storage(
        storage, force_pinned):
    result, _ = run(storage)
    stats = result.kv_cache
    assert stats["requested_storage"] == storage
    assert stats["actual_storage"] == storage
    assert stats["pin_fallback"] is False
    assert stats["d2h_calls"] > 0 and stats["d2h_bytes"] > 0
    assert stats["canonical_cpu_bytes"] > 0
    # The fake has no attention, so nothing reads the history back and the H2D
    # side stays at zero here: that half is pinned by ``test_cache_storage.py``
    # and ``test_causal_cache_storage.py``, against the real merge.
    assert stats["h2d_calls"] == 0 and stats["peak_gpu_slot_bytes"] == 0
    assert result.describe()["kv_cache"]["actual_storage"] == storage


def test_a_gpu_rollout_reports_no_host_traffic_at_all():
    result, _ = run("gpu")
    stats = result.kv_cache
    assert stats["actual_storage"] == "gpu"
    assert (stats["d2h_bytes"], stats["h2d_bytes"]) == (0, 0)
    assert stats["canonical_cpu_bytes"] == 0
    assert result.config.kv_cache_storage == "gpu"


def test_the_storage_does_not_move_a_single_bit(force_pinned):
    """Same seed, same schedule: the chunks are identical across storages."""
    reference = []
    baseline, _ = run("gpu", collect=reference)
    for storage in ("cpu_pinned", "cpu"):
        collected = []
        result, _ = run(storage, collect=collected)
        assert [c[0] for c in collected] == [c[0] for c in reference]
        for (_, ref_v, ref_a), (_, got_v, got_a) in zip(reference, collected):
            assert torch.equal(ref_v, got_v)
            assert torch.equal(ref_a, got_a)
        assert result.draws == baseline.draws
        assert result.noise_forwards == baseline.noise_forwards
        assert result.clean_forwards == baseline.clean_forwards


def test_eviction_and_call_order_are_unchanged_by_the_storage(force_pinned):
    traces = {}
    for storage in KV_CACHE_STORAGES:
        _, dit = run(storage, frames=73, steps=1, sink=2, window=1)
        traces[storage] = [
            (call.kind, call.chunk_index, tuple(call.retained_before))
            for call in dit.calls
        ]
    assert traces["cpu_pinned"] == traces["gpu"]
    assert traces["cpu"] == traces["gpu"]


# --- degradation and cancellation ---------------------------------------------


def test_a_pin_failure_is_reported_through_the_result(monkeypatch):
    monkeypatch.setattr(
        cache_mod, "_empty_pinned",
        lambda shape, dtype: (_ for _ in ()).throw(RuntimeError("no pinned pages")))
    result, _ = run("cpu_pinned")
    stats = result.kv_cache
    assert stats["requested_storage"] == "cpu_pinned"
    assert stats["actual_storage"] == "cpu"
    assert stats["pin_fallback"] is True
    assert "no pinned pages" in stats["pin_fallback_reason"]
    assert stats["warnings"] and "pageable" in stats["warnings"][0]


@pytest.mark.parametrize("storage", ("cpu_pinned", "cpu"))
def test_cancelling_a_host_backed_rollout_returns_nothing_partial(
        storage, force_pinned):
    def cancel(point):
        return point == "before_clean"

    with pytest.raises(SamplingCancelled, match="before_clean"):
        run(storage, cancel_check=cancel)


@pytest.mark.parametrize("storage", ("cpu_pinned", "cpu"))
def test_an_exception_mid_stack_leaves_no_pending_host_buffers(
        storage, force_pinned, monkeypatch):
    """The rollout's ``discard_pending`` is safe: no transfer is ever in flight."""
    caches = []
    original = consistency.ChunkKVCache

    def record(*args, **kwargs):
        cache = original(*args, **kwargs)
        caches.append(cache)
        return cache

    monkeypatch.setattr(consistency, "ChunkKVCache", record)

    class Exploding(FakeCausalDiT):
        def _stage_and_commit(self, cache, rows, role):
            if role == "clean":
                cache.stage(0, torch.zeros(rows, 2, 4), torch.zeros(rows, 2, 4))
                raise RuntimeError("boom mid-stack")
            super()._stage_and_commit(cache, rows, role)

    dit = Exploding(num_layers=2)
    with pytest.raises(RuntimeError, match="boom mid-stack"):
        sample_streaming(
            model=FakePatcher(dit),
            positive=text_conditioning(),
            latent=empty_av_latent(frames=39),
            config=SamplerConfig(steps=1, seed=5, kv_cache_storage=storage),
            load_models=LoadRecorder(),
            require_upstream_class=False,
        )

    (cache,) = caches
    assert not cache.has_pending
    assert cache.committed_chunks == 1          # the text prefill only
    assert cache.stats()["canonical_cpu_bytes"] > 0
