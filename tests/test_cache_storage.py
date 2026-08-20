"""Canonical KV storage: host offload, in-place merge assembly, accounting.

Pure torch, no ComfyUI, no CUDA required. What is claimed here:

* where a record physically is, and that it owns exactly its own rows -- a
  staged fused-QKV view must never become the record, whatever the storage;
* that :meth:`ChunkKVCache.copy_retained_into` fills the *prefix* of the
  caller's merged buffer and leaves the tail alone, so the attention path can
  assemble ``[retained | current]`` in one allocation;
* that the one device allocation this design permits is the merged slot, and
  that its accounting is the closed form ``2 * (past + current) * heads *
  head_dim * itemsize``;
* that ``gpu``, ``cpu_pinned`` and ``cpu`` produce **bit-identical** merged
  tensors, in BF16, for every sink/window combination -- the host round trip is
  a copy, never a cast;
* that a machine which cannot page-lock memory degrades to pageable host
  memory, reports it, and still produces the same bits;
* lifetime and mutation: a record is independent of the caller's buffers, and
  a discarded or evicted chunk releases them.

The CUDA-only claims (a real D2H/H2D round trip, really pinned memory) are
opt-in at the bottom and skip on a machine without a GPU.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import cache as cache_mod  # noqa: E402
from raven_streaming.cache import (  # noqa: E402
    DEFAULT_KV_STORAGE,
    KV_STORAGES,
    CacheError,
    ChunkKVCache,
)

HEADS, HEAD_DIM = 2, 4
HOST_STORAGES = ("cpu_pinned", "cpu")
#: row-major [rows, heads, head_dim]; the layout RAVEN's varlen kernel expects
ROW_MAJOR = (HEADS * HEAD_DIM, HEAD_DIM, 1)


def _kv(rows: int, seed: int, dtype: torch.dtype = torch.float32):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(rows, HEADS, HEAD_DIM, generator=g).to(dtype)
    v = torch.randn(rows, HEADS, HEAD_DIM, generator=g).to(dtype)
    return k, v


def _commit(cache: ChunkKVCache, rows: int, seed: int = 0, role: str = "clean",
            dtype: torch.dtype = torch.float32):
    for layer in range(cache.num_layers):
        cache.stage(layer, *_kv(rows, seed * 100 + layer, dtype))
    return cache.commit(role=role)


def merge_like_attention(
    cache: ChunkKVCache,
    layer: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """What ``RavenCausalAttention.forward`` does, in miniature.

    Kept in the test rather than imported so that a change to the attention
    module cannot quietly redefine what this file is asserting; the tiny-model
    tests in ``test_causal_cache_storage.py`` pin the real one against it.
    """
    spec = cache.retained_spec(layer)
    if spec is None:
        return key, value
    rows = key.shape[0]
    merged_k = key.new_empty((spec.rows + rows, spec.heads, spec.head_dim))
    merged_v = key.new_empty((spec.rows + rows, spec.heads, spec.head_dim))
    cache.copy_retained_into(layer, merged_k[:spec.rows], merged_v[:spec.rows])
    merged_k[spec.rows:].copy_(key)
    merged_v[spec.rows:].copy_(value)
    return merged_k, merged_v


@pytest.fixture
def force_pinned(monkeypatch):
    """Make the pinned path reachable on a host with no pinning allocator.

    ``torch.empty(pin_memory=True)`` raises without an accelerator, and on at
    least one platform it succeeds and then reports the result as unpinned --
    either way ``cpu_pinned`` would fall back and the pinned branch would never
    be exercised off a GPU box. The allocation itself is real; only the two
    machine-dependent answers are stubbed.
    """
    calls = []

    def fake_pinned(shape, dtype):
        calls.append((tuple(shape), dtype))
        return torch.empty(tuple(shape), dtype=dtype, device="cpu")

    monkeypatch.setattr(cache_mod, "_empty_pinned", fake_pinned)
    monkeypatch.setattr(cache_mod, "_is_pinned", lambda tensor: True)
    return calls


# --- construction ------------------------------------------------------------


def test_default_storage_is_gpu_so_old_callers_are_unaffected():
    cache = ChunkKVCache(2, sink=1, window=1)
    assert DEFAULT_KV_STORAGE == "gpu"
    assert cache.requested_storage == "gpu"
    assert cache.actual_storage == "gpu"
    assert cache.canonical_on_host is False
    assert cache.pin_fallback is False
    assert cache.warnings == ()


def test_unknown_storage_is_rejected():
    with pytest.raises(CacheError, match="storage must be one of"):
        ChunkKVCache(2, sink=1, window=1, storage="nvme")
    assert KV_STORAGES == ("gpu", "cpu_pinned", "cpu")


# --- canonical location and ownership ----------------------------------------


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_host_storage_records_are_owned_contiguous_host_buffers(storage, force_pinned):
    cache = ChunkKVCache(2, sink=4, window=None, storage=storage)
    k, v = _kv(3, 7, torch.bfloat16)
    for layer in range(2):
        cache.stage(layer, k, v)
    record = cache.commit()

    for layer in range(2):
        for tensor, source in ((record.keys[layer], k), (record.values[layer], v)):
            assert tensor.device.type == "cpu"
            assert tensor.is_contiguous()
            assert tuple(tensor.stride()) == ROW_MAJOR
            assert tensor.data_ptr() != source.data_ptr()
            # owns exactly its rows: a view would keep a bigger buffer alive
            assert tensor.untyped_storage().nbytes() == (
                tensor.numel() * tensor.element_size())
            assert torch.equal(tensor, source)
    assert cache.canonical_on_host is True


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_a_fused_three_times_view_never_becomes_a_host_record(storage, force_pinned):
    """The attention stages V as a view of the 3x fused QKV buffer."""
    cache = ChunkKVCache(1, sink=4, window=None, storage=storage)
    qkv = torch.randn(5, 3 * HEADS * HEAD_DIM).to(torch.bfloat16)
    view = qkv.split(HEADS * HEAD_DIM, dim=-1)[2].view(5, HEADS, HEAD_DIM)
    assert view.untyped_storage().nbytes() == 3 * view.numel() * view.element_size()

    # copy_value=False is the ownership-transfer flag; a host cache must copy
    # regardless, because "ownership" of a view would be ownership of 3x.
    cache.stage(0, torch.zeros_like(view), view, copy_value=False)
    record = cache.commit()
    stored = record.values[0]
    assert stored.untyped_storage().nbytes() == stored.numel() * stored.element_size()
    assert torch.equal(stored, view)
    qkv.zero_()
    assert float(stored.abs().sum()) > 0.0


def test_gpu_storage_still_transfers_ownership_when_asked():
    cache = ChunkKVCache(1, sink=1, window=None)
    k, v = _kv(3, 0)
    cache.stage(0, k, v, copy_key=True, copy_value=False)
    record = cache.commit()
    assert record.values[0].data_ptr() == v.data_ptr()
    assert record.keys[0].data_ptr() != k.data_ptr()


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_host_storage_ignores_the_ownership_flags(storage, force_pinned):
    cache = ChunkKVCache(1, sink=1, window=None, storage=storage)
    k, v = _kv(3, 1)
    cache.stage(0, k, v, copy_key=False, copy_value=False)
    record = cache.commit()
    assert record.keys[0].data_ptr() != k.data_ptr()
    assert record.values[0].data_ptr() != v.data_ptr()
    # ... and the caller's tensors are untouched, which is what lets the
    # attention module go on using them for the merge after staging
    assert torch.equal(record.keys[0], k) and torch.equal(record.values[0], v)


# --- direct merged assembly ---------------------------------------------------


@pytest.mark.parametrize("storage", KV_STORAGES)
def test_copy_retained_into_fills_the_prefix_and_leaves_the_tail(storage, force_pinned):
    cache = ChunkKVCache(1, sink=4, window=None, storage=storage)
    parts = []
    for index, rows in enumerate((2, 3)):
        k, v = _kv(rows, index)
        cache.stage(0, k, v)
        cache.commit()
        parts.append((k, v))

    current_k, current_v = _kv(4, 99)
    merged_k, merged_v = merge_like_attention(cache, 0, current_k, current_v)

    expected_k = torch.cat([p[0] for p in parts] + [current_k], dim=0)
    expected_v = torch.cat([p[1] for p in parts] + [current_v], dim=0)
    assert torch.equal(merged_k, expected_k)
    assert torch.equal(merged_v, expected_v)
    assert merged_k.is_contiguous() and tuple(merged_k.stride()) == ROW_MAJOR
    assert merged_v.is_contiguous() and tuple(merged_v.stride()) == ROW_MAJOR
    # and it agrees with the gather API it replaces
    gathered_k, gathered_v = cache.retained(0)
    assert torch.equal(merged_k[:5], gathered_k)
    assert torch.equal(merged_v[:5], gathered_v)


@pytest.mark.parametrize("storage", KV_STORAGES)
def test_past_rows_is_the_per_layer_spelling_of_the_history(storage, force_pinned):
    cache = ChunkKVCache(3, sink=4, window=None, storage=storage)
    assert cache.past_rows(0) == 0
    assert cache.retained_spec(0) is None
    _commit(cache, rows=6, seed=1)
    _commit(cache, rows=4, seed=2)
    for layer in range(3):
        assert cache.past_rows(layer) == 10 == cache.retained_rows
        spec = cache.retained_spec(layer)
        assert (spec.rows, spec.heads, spec.head_dim) == (10, HEADS, HEAD_DIM)
        assert spec.dtype == torch.float32
        assert spec.device.type == "cpu"
    with pytest.raises(CacheError):
        cache.past_rows(3)


def test_an_empty_history_copies_nothing_and_needs_no_prefix():
    cache = ChunkKVCache(1, sink=1, window=None, storage="cpu")
    empty = torch.zeros(0, HEADS, HEAD_DIM)
    assert cache.copy_retained_into(0, empty, empty) == 0
    assert cache.stats()["h2d_calls"] == 0


def test_a_mis_sized_or_mis_typed_destination_is_refused():
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu")
    _commit(cache, rows=5, seed=3)
    good = torch.zeros(5, HEADS, HEAD_DIM)
    with pytest.raises(CacheError, match="rows but layer"):
        cache.copy_retained_into(0, torch.zeros(4, HEADS, HEAD_DIM), good)
    with pytest.raises(CacheError, match=r"\[rows, heads, head_dim\]"):
        cache.copy_retained_into(0, torch.zeros(5, HEADS * HEAD_DIM), good)
    with pytest.raises(CacheError, match="head shape"):
        cache.copy_retained_into(0, torch.zeros(5, HEADS + 1, HEAD_DIM),
                                 torch.zeros(5, HEADS + 1, HEAD_DIM))
    with pytest.raises(CacheError, match="never casts"):
        cache.copy_retained_into(0, good.to(torch.bfloat16), good.to(torch.bfloat16))


# --- single-slot peak ---------------------------------------------------------


@pytest.mark.parametrize("storage", KV_STORAGES)
def test_peak_slot_is_the_closed_form_of_one_merged_pair(storage, force_pinned):
    """``2 * (past + current) * heads * head_dim * itemsize``, and nothing else."""
    dtype = torch.bfloat16
    itemsize = torch.tensor([], dtype=dtype).element_size()
    chunk_rows = [5, 7, 6, 9]
    cache = ChunkKVCache(1, sink=1, window=1, storage=storage)

    expected_peak = 0
    for index, rows in enumerate(chunk_rows):
        past = cache.past_rows(0)
        current_k, current_v = _kv(rows, index, dtype)
        merge_like_attention(cache, 0, current_k, current_v)
        if past:
            # No history means no merged slot at all: the forward keeps its own
            # k/v (RAVEN's no-history layout) and nothing is assembled.
            expected_peak = max(expected_peak,
                                2 * (past + rows) * HEADS * HEAD_DIM * itemsize)
        cache.stage(0, current_k, current_v)
        cache.commit()
        assert cache.stats()["peak_gpu_slot_bytes"] == expected_peak

    # sink=1 + window=1 caps the history at two chunks, so the slot is capped
    # too: it never grows with the length of the rollout. Longhand, the four
    # merges are 0+5, 5+7, 12+6 and 11+9 rows, so the widest is 20.
    assert expected_peak == 2 * 20 * HEADS * HEAD_DIM * itemsize
    assert cache.stats()["peak_gpu_slot_bytes"] == expected_peak


def test_the_slot_is_measured_on_the_whole_merged_buffer_not_the_prefix():
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu")
    _commit(cache, rows=3, seed=1)
    current_k, current_v = _kv(5, 2)
    merge_like_attention(cache, 0, current_k, current_v)
    element = current_k.element_size()
    assert cache.stats()["peak_gpu_slot_bytes"] == 2 * 8 * HEADS * HEAD_DIM * element


# --- bit-for-bit parity across storages ---------------------------------------


@pytest.mark.parametrize("sink", [0, 1, 2, 3])
@pytest.mark.parametrize("window", [None, 0, 1, 2, 5])
def test_every_storage_assembles_the_same_bits(sink, window, force_pinned):
    """BF16, every retention policy: the host round trip is a copy, not a cast."""
    dtype = torch.bfloat16
    chunk_rows = [5, 2, 7, 4, 6, 3]
    results = {}
    for storage in KV_STORAGES:
        cache = ChunkKVCache(2, sink=sink, window=window, storage=storage)
        merged = []
        for index, rows in enumerate(chunk_rows):
            current_k, current_v = _kv(rows, index + 1, dtype)
            for layer in range(2):
                merged.append(merge_like_attention(cache, layer, current_k, current_v))
                cache.stage(layer, current_k, current_v)
            cache.commit()
            assert cache.retained_indices == cache.retained_index_set(index + 1)
        results[storage] = merged

    reference = results["gpu"]
    for storage in ("cpu_pinned", "cpu"):
        for (ref_k, ref_v), (got_k, got_v) in zip(reference, results[storage]):
            assert torch.equal(ref_k, got_k)
            assert torch.equal(ref_v, got_v)
            assert got_k.dtype == dtype and got_v.dtype == dtype


def test_a_bf16_host_round_trip_is_bitwise():
    dtype = torch.bfloat16
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu")
    k, v = _kv(11, 5, dtype)
    cache.stage(0, k, v)
    cache.commit()
    out_k = torch.empty(11, HEADS, HEAD_DIM, dtype=dtype)
    out_v = torch.empty(11, HEADS, HEAD_DIM, dtype=dtype)
    cache.copy_retained_into(0, out_k, out_v)
    assert torch.equal(out_k.view(torch.int16), k.view(torch.int16))
    assert torch.equal(out_v.view(torch.int16), v.view(torch.int16))


# --- lifetime, mutation, cancellation, eviction -------------------------------


@pytest.mark.parametrize("storage", KV_STORAGES)
def test_a_record_is_independent_of_the_caller_buffers(storage, force_pinned):
    cache = ChunkKVCache(1, sink=4, window=None, storage=storage)
    k, v = _kv(3, 0)
    expected_k, expected_v = k.clone(), v.clone()
    cache.stage(0, k, v)
    cache.commit()
    k.zero_()
    v.zero_()
    keys, values = cache.retained(0)
    assert torch.equal(keys, expected_k)
    assert torch.equal(values, expected_v)


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_a_host_record_survives_later_merges(storage, force_pinned):
    cache = ChunkKVCache(1, sink=4, window=None, storage=storage)
    _commit(cache, rows=4, seed=1)
    before = [t.clone() for t in cache.retained(0)]
    for index in range(3):
        current_k, current_v = _kv(2, 50 + index)
        merged_k, merged_v = merge_like_attention(cache, 0, current_k, current_v)
        merged_k.zero_()          # writing the merged slot must not reach back
        merged_v.zero_()
    after = cache.retained(0)
    assert torch.equal(after[0], before[0]) and torch.equal(after[1], before[1])


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_discarding_a_pending_chunk_releases_its_host_buffers(storage, force_pinned):
    cache = ChunkKVCache(3, sink=4, window=None, storage=storage)
    _commit(cache, rows=4, seed=1)
    settled = cache.canonical_cpu_bytes
    cache.stage(0, *_kv(6, 2))
    cache.stage(1, *_kv(6, 3))
    assert cache.canonical_cpu_bytes > settled
    assert cache.has_pending

    cache.discard_pending()      # what a cancel / an exception does
    assert not cache.has_pending
    assert cache.canonical_cpu_bytes == settled
    # nothing was in flight, so the cache is immediately usable again
    _commit(cache, rows=4, seed=4)
    assert cache.committed_chunks == 2


@pytest.mark.parametrize("storage", HOST_STORAGES)
def test_eviction_frees_host_bytes(storage, force_pinned):
    cache = ChunkKVCache(2, sink=1, window=1, storage=storage)
    for index, rows in enumerate((5, 40, 40, 40)):
        _commit(cache, rows=rows, seed=index)
    stats = cache.stats()
    per_row = 2 * 2 * HEADS * HEAD_DIM * 4      # K+V, 2 layers, fp32
    assert cache.retained_indices == [0, 3]
    assert stats["canonical_cpu_bytes"] == 45 * per_row
    # The peak is the staging moment, where the chunk being staged is resident
    # beside the history it will evict: 5 + 40 retained + 40 pending.
    assert stats["peak_canonical_cpu_bytes"] == 85 * per_row


def test_reset_forgets_records_but_keeps_the_traffic_report(force_pinned):
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")
    _commit(cache, rows=4, seed=1)
    moved = cache.stats()["d2h_bytes"]
    cache.reset()
    assert cache.committed_chunks == 0
    assert cache.retained(0) is None
    assert cache.canonical_cpu_bytes == 0
    assert cache.stats()["d2h_bytes"] == moved
    assert cache.actual_storage == "cpu_pinned"


# --- pin failure --------------------------------------------------------------


def test_pin_failure_falls_back_to_pageable_and_says_so(monkeypatch, caplog):
    def refuse(shape, dtype):
        raise RuntimeError("Need to provide pin_memory allocator to use pin memory.")

    monkeypatch.setattr(cache_mod, "_empty_pinned", refuse)
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")
    with caplog.at_level("WARNING"):
        k, v = _kv(3, 0)
        cache.stage(0, k, v)
        cache.commit()

    assert cache.requested_storage == "cpu_pinned"
    assert cache.actual_storage == "cpu"
    assert cache.pin_fallback is True
    assert "pin_memory allocator" in cache.pin_fallback_reason
    assert len(cache.warnings) == 1
    assert "pageable" in cache.warnings[0]
    assert any("pinned host memory" in r.getMessage() for r in caplog.records)
    # ... and the values are exactly the ones a pinned cache would have held
    assert torch.equal(cache.retained(0)[0], k)
    stats = cache.stats()
    assert stats["requested_storage"] == "cpu_pinned"
    assert stats["actual_storage"] == "cpu"


def test_the_fallback_is_taken_once_not_per_tensor(monkeypatch):
    attempts = []

    def refuse(shape, dtype):
        attempts.append(tuple(shape))
        raise RuntimeError("out of pinnable pages")

    monkeypatch.setattr(cache_mod, "_empty_pinned", refuse)
    cache = ChunkKVCache(2, sink=4, window=None, storage="cpu_pinned")
    _commit(cache, rows=3, seed=0)
    _commit(cache, rows=3, seed=1)
    assert len(attempts) == 1
    assert len(cache.warnings) == 1


def test_memory_that_reports_itself_unpinned_counts_as_a_fallback(monkeypatch):
    """Some hosts accept ``pin_memory=True`` and hand back ordinary pages."""
    monkeypatch.setattr(cache_mod, "_empty_pinned",
                        lambda shape, dtype: torch.empty(tuple(shape), dtype=dtype))
    monkeypatch.setattr(cache_mod, "_is_pinned", lambda tensor: False)
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")
    _commit(cache, rows=2, seed=0)
    assert cache.actual_storage == "cpu"
    assert "not page-locked" in cache.pin_fallback_reason
    assert cache.retained(0)[0].shape[0] == 2


def test_a_pageable_request_never_asks_for_pinned_memory(monkeypatch):
    def explode(shape, dtype):  # pragma: no cover - must not be reached
        raise AssertionError("storage='cpu' must not try to pin")

    monkeypatch.setattr(cache_mod, "_empty_pinned", explode)
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu")
    _commit(cache, rows=2, seed=0)
    assert cache.actual_storage == "cpu"
    assert cache.pin_fallback is False


def test_the_pinned_path_allocates_pinned_buffers(force_pinned):
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")
    _commit(cache, rows=3, seed=0)
    assert cache.actual_storage == "cpu_pinned"
    assert cache.pin_fallback is False
    assert force_pinned == [((3, HEADS, HEAD_DIM), torch.float32)] * 2


# --- stats / report -----------------------------------------------------------


def test_stats_account_for_every_transfer(force_pinned):
    cache = ChunkKVCache(2, sink=4, window=None, storage="cpu_pinned")
    element = 4
    rows = 3
    _commit(cache, rows=rows, seed=1)                # 2 layers x (K, V)
    staged_bytes = 2 * 2 * rows * HEADS * HEAD_DIM * element

    stats = cache.stats()
    assert stats["d2h_calls"] == 4
    assert stats["d2h_bytes"] == staged_bytes
    assert stats["h2d_calls"] == 0 and stats["h2d_bytes"] == 0
    assert stats["canonical_cpu_bytes"] == staged_bytes

    for layer in range(2):
        merge_like_attention(cache, layer, *_kv(2, 9))
    stats = cache.stats()
    assert stats["h2d_calls"] == 2 * 2                 # one record, K and V, per layer
    assert stats["h2d_bytes"] == staged_bytes          # the whole history, once per layer
    assert stats["requested_storage"] == "cpu_pinned"
    assert stats["actual_storage"] == "cpu_pinned"
    assert stats["pin_fallback"] is False
    assert stats["retained_rows"] == rows
    assert stats["sink"] == 4 and stats["window"] is None


def test_a_device_resident_cache_reports_no_host_traffic():
    cache = ChunkKVCache(2, sink=4, window=None)
    _commit(cache, rows=3, seed=1)
    for layer in range(2):
        merge_like_attention(cache, layer, *_kv(2, 9))
    stats = cache.stats()
    assert stats["canonical_cpu_bytes"] == 0
    assert stats["peak_canonical_cpu_bytes"] == 0
    assert (stats["h2d_bytes"], stats["h2d_calls"]) == (0, 0)
    assert (stats["d2h_bytes"], stats["d2h_calls"]) == (0, 0)
    # the merged slot is still accounted: it is device memory either way
    assert stats["peak_gpu_slot_bytes"] > 0


def test_report_is_one_line_and_names_the_fallback(monkeypatch):
    monkeypatch.setattr(cache_mod, "_empty_pinned",
                        lambda shape, dtype: (_ for _ in ()).throw(RuntimeError("nope")))
    cache = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")
    _commit(cache, rows=3, seed=0)
    line = cache.report()
    assert "\n" not in line
    assert "storage=cpu (requested cpu_pinned)" in line
    assert "GiB" in line


# --- CUDA, opt-in -------------------------------------------------------------


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="needs a CUDA device")


@cuda_only
def test_cuda_host_offload_keeps_nothing_on_the_device():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache = ChunkKVCache(2, sink=4, window=None, storage="cpu_pinned")
    k = torch.randn(6, HEADS, HEAD_DIM, device=device).to(dtype)
    v = torch.randn(6, HEADS, HEAD_DIM, device=device).to(dtype)
    for layer in range(2):
        cache.stage(layer, k, v, copy_key=False, copy_value=True)
    cache.commit()

    for layer in range(2):
        for tensor in cache.retained(layer):
            assert tensor.device.type == "cpu"
            assert tensor.is_pinned()
    assert cache.actual_storage == "cpu_pinned"
    assert cache.pin_fallback is False


@cuda_only
def test_cuda_round_trip_is_bitwise_and_assembles_in_place():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    reference = ChunkKVCache(1, sink=4, window=None)
    offloaded = ChunkKVCache(1, sink=4, window=None, storage="cpu_pinned")

    merged = {}
    for index in range(3):
        k = torch.randn(4 + index, HEADS, HEAD_DIM, device=device).to(dtype)
        v = torch.randn(4 + index, HEADS, HEAD_DIM, device=device).to(dtype)
        for name, cache in (("gpu", reference), ("cpu_pinned", offloaded)):
            merged.setdefault(name, []).append(merge_like_attention(cache, 0, k, v))
            cache.stage(0, k, v, copy_key=True, copy_value=True)
            cache.commit()

    for (ref_k, ref_v), (got_k, got_v) in zip(merged["gpu"], merged["cpu_pinned"]):
        assert got_k.device.type == "cuda" and got_v.device.type == "cuda"
        assert torch.equal(ref_k.view(torch.int16), got_k.view(torch.int16))
        assert torch.equal(ref_v.view(torch.int16), got_v.view(torch.int16))
    assert offloaded.stats()["h2d_bytes"] > 0
    assert offloaded.stats()["d2h_bytes"] > 0
