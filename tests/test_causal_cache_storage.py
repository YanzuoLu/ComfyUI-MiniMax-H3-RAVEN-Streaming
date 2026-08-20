"""The causal lane against an off-device KV cache: same bits, one merged slot.

``storage`` is a residency choice and nothing else. These tests drive the real
tiny H3 model, at BF16 (where a stray cast would show), and claim:

* a rollout on ``storage='cpu'`` / ``'cpu_pinned'`` produces **bit-identical**
  velocities to one on ``'gpu'``, for several sink/window policies -- including
  the ones that evict;
* the attention assembles ``[retained | current]`` into one merged buffer per
  layer, filled in place, with the retained rows in the prefix and this chunk's
  own rows in the tail, contiguous and row-major -- the layout RAVEN's varlen
  kernel is handed;
* the merged buffer is the caller's and the cache keeps no reference to it, so
  nothing device-resident outlives one layer's attention call;
* a clean/text fill has already copied this layer's K/V off the device *before*
  that layer's attention runs;
* a host record owns exactly its rows -- never the 3x fused QKV buffer the
  value is a view of;
* a machine that cannot pin still runs, still bit-identical, and says so.

Requires a local ComfyUI checkout (see ``tests/conftest.py``).
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

from raven_streaming import cache as cache_mod  # noqa: E402
from raven_streaming import causal_model as cm  # noqa: E402
from raven_streaming.cache import KV_STORAGES, ChunkKVCache  # noqa: E402
from test_causal_common import (  # noqa: E402
    TINY_CONFIG,
    random_inputs,
    tiny_bf16_models,  # noqa: F401  (fixture re-export)
    tiny_layout,
)

NUM_LAYERS = TINY_CONFIG["num_layers"]
HEADS = TINY_CONFIG["num_attention_heads"]
HEAD_DIM = TINY_CONFIG["attention_head_dim"]
BF16 = torch.bfloat16
#: contiguous [rows, heads, head_dim]
ROW_MAJOR = (HEADS * HEAD_DIM, HEAD_DIM, 1)
HOST_STORAGES = ("cpu_pinned", "cpu")


@pytest.fixture
def force_pinned(monkeypatch):
    """Reach the pinned branch on a host whose allocator cannot page-lock.

    Same seam as ``tests/test_cache_storage.py``: the allocation is real, only
    the machine-dependent "is this pinned" answer is stubbed. Without it every
    ``cpu_pinned`` test here would silently be a ``cpu`` test.
    """
    monkeypatch.setattr(
        cache_mod, "_empty_pinned",
        lambda shape, dtype: torch.empty(tuple(shape), dtype=dtype, device="cpu"))
    monkeypatch.setattr(cache_mod, "_is_pinned", lambda tensor: True)


def bf16_inputs(layout, seed=1):
    video, audio, context = random_inputs(layout, seed=seed)
    return video.to(BF16), audio.to(BF16), context.to(BF16)


def rollout(causal, layout, storage, *, sink=8, window=None, chunks=None, seed=1):
    """Prefill + (noise, clean)* over the first ``chunks`` chunks.

    Returns ``(cache, outputs)`` where ``outputs`` is every velocity pair the
    noise forwards produced, in order -- the tensors a sampler would consume.
    """
    video, audio, context = bf16_inputs(layout, seed=seed)
    cache = ChunkKVCache(NUM_LAYERS, sink=sink, window=window, storage=storage)
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)

    count = layout.num_chunks if chunks is None else int(chunks)
    outputs = []
    for index in range(count):
        v = layout.video_chunk_latent(video, index)
        a = layout.audio_chunk_latent(audio, index)
        outputs.append(causal.forward_chunk(
            video_latent=v, audio_latent=a, layout=layout, chunk_index=index,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16))
        if index == count - 1:
            continue
        causal.forward_chunk(
            video_latent=v, audio_latent=a, layout=layout, chunk_index=index,
            cache=cache, role="clean", compute_dtype=BF16,
            video_eps=torch.zeros_like(v), audio_eps=torch.zeros_like(a))
    return cache, outputs


def capture_seam(monkeypatch, extra=None):
    """Record what every attention call is handed (and optionally more)."""
    calls = []
    original = cm.raven_packed_attention

    def spy(q, k, v, *, scale, site=None):
        entry = {
            "site": site,
            "rows": int(q.shape[0]),
            "kv_rows": int(k.shape[0]),
            "k": k,
            "v": v,
            "k_stride": tuple(k.stride()),
            "v_stride": tuple(v.stride()),
        }
        if extra is not None:
            entry.update(extra(site))
        calls.append(entry)
        return original(q, k, v, scale=scale, site=site)

    monkeypatch.setattr(cm, "raven_packed_attention", spy)
    return calls


# --- parity across storages ---------------------------------------------------


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_a_host_cache_produces_the_same_bits_as_a_device_one(
        tiny_bf16_models, storage, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    reference_cache, reference = rollout(causal, layout, "gpu")
    offloaded_cache, offloaded = rollout(causal, layout, storage)

    assert len(reference) == layout.num_chunks
    for (ref_v, ref_a), (got_v, got_a) in zip(reference, offloaded):
        assert torch.equal(ref_v, got_v)
        assert torch.equal(ref_a, got_a)
    # ... and so are the records themselves, bit for bit in BF16
    for layer in range(NUM_LAYERS):
        for ref, got in zip(reference_cache.retained(layer),
                            offloaded_cache.retained(layer)):
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16))
    assert offloaded_cache.actual_storage == storage


@pytest.mark.parametrize("sink,window", [(1, 0), (1, 1), (2, 1), (2, None), (8, None)])
def test_every_retention_policy_is_storage_independent(
        tiny_bf16_models, sink, window, force_pinned):
    """Including the policies that evict: eviction is unchanged by residency."""
    _, causal = tiny_bf16_models
    layout = tiny_layout(frames=90)  # 6 chunks, so a window really bites
    results = {}
    for storage in KV_STORAGES:
        cache, outputs = rollout(causal, layout, storage, sink=sink, window=window)
        results[storage] = (cache.retained_indices, cache.chunk_lens, outputs)

    reference_indices, reference_lens, reference = results["gpu"]
    for storage in HOST_STORAGES:
        indices, lens, outputs = results[storage]
        assert indices == reference_indices
        assert lens == reference_lens
        for (ref_v, ref_a), (got_v, got_a) in zip(reference, outputs):
            assert torch.equal(ref_v, got_v)
            assert torch.equal(ref_a, got_a)


# --- the merged slot ----------------------------------------------------------


@pytest.mark.parametrize("storage", KV_STORAGES)
def test_the_backend_sees_prefix_history_then_this_chunks_own_rows(
        tiny_bf16_models, storage, monkeypatch, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)
    expected = {layer: cache.retained(layer) for layer in range(NUM_LAYERS)}

    calls = capture_seam(monkeypatch)
    causal.forward_chunk(
        video_latent=layout.video_chunk_latent(video, 0),
        audio_latent=layout.audio_chunk_latent(audio, 0),
        layout=layout, chunk_index=0, cache=cache, role="noise",
        video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)

    rows = layout.chunks[0].rows
    text_len = layout.text_len
    assert len(calls) == NUM_LAYERS
    for layer, call in enumerate(calls):
        assert call["site"] == ("dit", layer)
        assert call["rows"] == rows
        assert call["kv_rows"] == text_len + rows
        # the prefix is the retained history, bit for bit, in time order
        past_k, past_v = expected[layer]
        assert torch.equal(call["k"][:text_len].view(torch.int16),
                           past_k.view(torch.int16))
        assert torch.equal(call["v"][:text_len].view(torch.int16),
                           past_v.view(torch.int16))
        # ... and one merged allocation, in RAVEN's layout
        assert call["k_stride"] == ROW_MAJOR and call["v_stride"] == ROW_MAJOR
        assert call["k"].is_contiguous() and call["v"].is_contiguous()
        assert call["k"].untyped_storage().nbytes() == (
            call["k"].numel() * call["k"].element_size())


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_no_history_keeps_ravens_original_layout(
        tiny_bf16_models, storage, monkeypatch, force_pinned):
    """An empty cache assembles nothing: k is the forward's, v the fused view."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
    calls = capture_seam(monkeypatch)

    causal.prefill_text(context, cache=cache, compute_dtype=BF16)

    blocks = [c for c in calls if c["site"][0] == "dit"]
    assert len(blocks) == NUM_LAYERS
    for call in blocks:
        assert call["kv_rows"] == layout.text_len
        assert call["k_stride"] == ROW_MAJOR
        # v is still the 3x fused-QKV view -- the host copy read through it and
        # did not replace it
        assert call["v_stride"] == (3 * HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert cache.stats()["peak_gpu_slot_bytes"] == 0


def test_the_merged_slot_is_the_closed_form_and_belongs_to_the_caller(
        tiny_bf16_models, monkeypatch, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    cache, _ = rollout(causal, layout, "cpu_pinned", sink=1, window=1, chunks=2)

    itemsize = torch.tensor([], dtype=BF16).element_size()
    # sink=1 + window=1 keeps the text and the newest chunk, so the widest merge
    # of this rollout is chunk 1's: [text | chunk 0] in front of its own rows.
    widest = layout.text_len + layout.chunks[0].rows + layout.chunks[1].rows
    expected = 2 * widest * HEADS * HEAD_DIM * itemsize
    assert cache.stats()["peak_gpu_slot_bytes"] == expected

    # nothing the cache holds is one of those merged buffers
    merged_pointers = set()
    calls = capture_seam(monkeypatch)
    video, audio, _ = bf16_inputs(layout)
    causal.forward_chunk(
        video_latent=layout.video_chunk_latent(video, 1),
        audio_latent=layout.audio_chunk_latent(audio, 1),
        layout=layout, chunk_index=1, cache=cache, role="noise",
        video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)
    for call in calls:
        merged_pointers.add(call["k"].data_ptr())
        merged_pointers.add(call["v"].data_ptr())
    for layer in range(NUM_LAYERS):
        for tensor in cache.retained(layer):
            assert tensor.data_ptr() not in merged_pointers


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_the_cache_holds_nothing_but_owned_host_rows(
        tiny_bf16_models, storage, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    cache, _ = rollout(causal, layout, storage, chunks=2)

    assert cache.canonical_on_host is True
    for record in cache._records:  # noqa: SLF001 - the point is what is stored
        for table in (record.keys, record.values):
            assert sorted(table) == list(range(NUM_LAYERS))
            for tensor in table.values():
                assert tensor.device.type == "cpu"
                assert tensor.is_contiguous()
                assert tuple(tensor.stride()) == ROW_MAJOR
                # exactly its own rows: never the 3x fused QKV buffer
                assert tensor.untyped_storage().nbytes() == (
                    tensor.numel() * tensor.element_size())
    expected_bytes = sum(
        2 * NUM_LAYERS * record.rows * HEADS * HEAD_DIM * 2
        for record in cache._records  # noqa: SLF001
    )
    assert cache.stats()["canonical_cpu_bytes"] == expected_bytes


# --- ordering: the copy happens before the attention --------------------------


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_a_cache_filling_layer_copies_off_device_before_it_attends(
        tiny_bf16_models, storage, monkeypatch, force_pinned):
    """Staging is synchronous and comes first, so the two never coexist.

    ``d2h_calls`` at the moment layer *i*'s attention is entered must already
    include that layer's own K and V -- two copies per layer, layer 0 first.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)

    calls = capture_seam(
        monkeypatch, extra=lambda site: {"d2h_calls": cache.stats()["d2h_calls"]})
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)

    blocks = [c for c in calls if c["site"][0] == "dit"]
    assert [c["d2h_calls"] for c in blocks] == [2 * (i + 1) for i in range(NUM_LAYERS)]
    assert cache.stats()["d2h_calls"] == 2 * NUM_LAYERS
    assert cache.stats()["d2h_bytes"] == (
        2 * NUM_LAYERS * layout.text_len * HEADS * HEAD_DIM * 2)


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_a_read_only_forward_copies_nothing_off_device(
        tiny_bf16_models, storage, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)
    staged = cache.stats()["d2h_calls"]

    causal.forward_chunk(
        video_latent=layout.video_chunk_latent(video, 0),
        audio_latent=layout.audio_chunk_latent(audio, 0),
        layout=layout, chunk_index=0, cache=cache, role="noise",
        video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)

    assert cache.stats()["d2h_calls"] == staged      # read-only really is
    assert cache.stats()["h2d_calls"] == 2 * NUM_LAYERS   # one record per layer


# --- lifetime and failure -----------------------------------------------------


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_the_host_records_survive_the_forwards_that_follow(
        tiny_bf16_models, storage, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)
    before = [tuple(t.clone() for t in cache.retained(layer))
              for layer in range(NUM_LAYERS)]

    for _ in range(2):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="noise",
            video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)

    for layer, (keys, values) in enumerate(before):
        current_k, current_v = cache.retained(layer)
        assert torch.equal(current_k, keys)
        assert torch.equal(current_v, values)


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_an_aborted_forward_leaves_no_half_chunk_on_the_host(
        tiny_bf16_models, storage, monkeypatch, force_pinned):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_inputs(layout)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
    causal.prefill_text(context, cache=cache, compute_dtype=BF16)
    settled = cache.stats()["canonical_cpu_bytes"]

    original = cm.raven_packed_attention
    state = {"calls": 0}

    def explode(q, k, v, *, scale, site=None):
        state["calls"] += 1
        if state["calls"] > 2:      # let a layer or two land first
            raise RuntimeError("cancelled mid-stack")
        return original(q, k, v, scale=scale, site=site)

    monkeypatch.setattr(cm, "raven_packed_attention", explode)
    with pytest.raises(RuntimeError, match="cancelled mid-stack"):
        causal.forward_chunk(
            video_latent=layout.video_chunk_latent(video, 0),
            audio_latent=layout.audio_chunk_latent(audio, 0),
            layout=layout, chunk_index=0, cache=cache, role="clean",
            video_eps=torch.zeros_like(layout.video_chunk_latent(video, 0)),
            audio_eps=torch.zeros_like(layout.audio_chunk_latent(audio, 0)),
            compute_dtype=BF16)

    # ``_run_blocks`` discards the partial stage set; nothing was in flight, so
    # the host buffers are released immediately and the cache is reusable
    assert not cache.has_pending
    assert cache.stats()["canonical_cpu_bytes"] == settled
    assert cache.committed_chunks == 1


def test_a_host_without_pinning_still_runs_and_still_matches(
        tiny_bf16_models, monkeypatch):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    reference_cache, reference = rollout(causal, layout, "gpu", chunks=2)

    monkeypatch.setattr(
        cache_mod, "_empty_pinned",
        lambda shape, dtype: (_ for _ in ()).throw(RuntimeError("no pinned pages")))
    cache, outputs = rollout(causal, layout, "cpu_pinned", chunks=2)

    assert cache.requested_storage == "cpu_pinned"
    assert cache.actual_storage == "cpu"
    assert cache.pin_fallback is True
    for (ref_v, ref_a), (got_v, got_a) in zip(reference, outputs):
        assert torch.equal(ref_v, got_v)
        assert torch.equal(ref_a, got_a)
    for layer in range(NUM_LAYERS):
        for ref, got in zip(reference_cache.retained(layer), cache.retained(layer)):
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


def test_a_dtype_mismatch_across_the_host_boundary_is_refused(tiny_bf16_models):
    """One cache, one compute dtype -- the host copy never casts it back.

    Driven at the attention module, because a whole ``forward_chunk`` in the
    wrong dtype dies in the first AdaLN GEMM instead; the guard being tested is
    the one in front of the merge.
    """
    from comfy.ldm.minimax.model import rope_rotation_table

    _, causal = tiny_bf16_models
    layout = tiny_layout()
    rows = layout.chunks[0].rows
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage="cpu")
    torch.manual_seed(11)
    for layer in range(NUM_LAYERS):
        # an fp32 history, which a BF16 chunk may not be merged with
        cache.stage(layer, torch.randn(4, HEADS, HEAD_DIM),
                    torch.randn(4, HEADS, HEAD_DIM))
    cache.commit()

    x = torch.randn(rows, TINY_CONFIG["hidden_size"]).to(BF16)
    rope_table = rope_rotation_table(
        causal.rope_freqs(layout.chunk_position_ids(0), x.device), BF16)
    with pytest.raises(cm.CausalModelError, match="one compute dtype"):
        causal.blocks[0].attn(x, rope_freqs=rope_table, cache=cache)


# --- CUDA, opt-in -------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_rollout_matches_the_device_resident_one(tiny_bf16_models):
    _, causal = tiny_bf16_models
    causal = causal.to("cuda")
    layout = tiny_layout()

    def run(storage):
        video, audio, context = bf16_inputs(layout)
        video, audio = video.cuda(), audio.cuda()
        cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None, storage=storage)
        causal.prefill_text(context.cuda(), cache=cache, compute_dtype=BF16)
        outputs = []
        for index in range(2):
            v = layout.video_chunk_latent(video, index)
            a = layout.audio_chunk_latent(audio, index)
            outputs.append(causal.forward_chunk(
                video_latent=v, audio_latent=a, layout=layout, chunk_index=index,
                cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
                compute_dtype=BF16))
            if index == 0:
                causal.forward_chunk(
                    video_latent=v, audio_latent=a, layout=layout, chunk_index=index,
                    cache=cache, role="clean", compute_dtype=BF16,
                    video_eps=torch.zeros_like(v), audio_eps=torch.zeros_like(a))
        return cache, outputs

    reference_cache, reference = run("gpu")
    cache, outputs = run("cpu_pinned")
    for (ref_v, ref_a), (got_v, got_a) in zip(reference, outputs):
        assert torch.equal(ref_v, got_v)
        assert torch.equal(ref_a, got_a)
    for layer in range(NUM_LAYERS):
        assert cache.retained(layer)[0].device.type == "cpu"
        assert cache.retained(layer)[0].is_pinned()
        for ref, got in zip(reference_cache.retained(layer), cache.retained(layer)):
            assert torch.equal(ref.cpu().view(torch.int16), got.view(torch.int16))
