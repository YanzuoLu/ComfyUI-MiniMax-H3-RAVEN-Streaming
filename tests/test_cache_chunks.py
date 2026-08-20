"""Chunk KV cache: commit protocol, retention policy, eviction alignment.

Pure torch, no ComfyUI. The end-to-end behaviour against a real model (text
prefill -> read-only noise pass -> clean write -> next chunk sees it) lives in
``test_causal_cache_flow.py``.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.cache import CacheError, ChunkKVCache  # noqa: E402

HEADS, HEAD_DIM = 2, 4


def _kv(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(rows, HEADS, HEAD_DIM, generator=g)
    v = torch.randn(rows, HEADS, HEAD_DIM, generator=g)
    return k, v


def _commit(cache: ChunkKVCache, rows: int, seed: int = 0, role: str = "clean"):
    for layer in range(cache.num_layers):
        cache.stage(layer, *_kv(rows, seed * 100 + layer))
    return cache.commit(role=role)


def _raven_retained(sink: int, window: Optional[int], num_chunks: int) -> List[int]:
    """RAVEN's ``NaiveCache`` rule, replayed: pop one chunk per step.

    ``utils/naive_cache.py``: once ``current_step > sink + window`` the chunk at
    ``current_step - window - 1`` is evicted. This is the behaviour the
    chunk-record cache must reproduce.
    """
    retained: List[int] = []
    for step in range(1, num_chunks + 1):
        retained.append(step - 1)
        if window is not None and step > sink + window:
            retained.remove(step - window - 1)
    return retained


# --- construction ------------------------------------------------------------


def test_rejects_bad_construction():
    with pytest.raises(CacheError):
        ChunkKVCache(0, sink=1, window=1)
    with pytest.raises(CacheError):
        ChunkKVCache(2, sink=-1, window=1)
    with pytest.raises(CacheError):
        ChunkKVCache(2, sink=1, window=-1)


def test_empty_cache_reads_as_no_prefix():
    cache = ChunkKVCache(3, sink=1, window=1)
    assert cache.retained(0) is None
    assert cache.committed_chunks == 0
    assert cache.retained_rows == 0


# --- retention policy --------------------------------------------------------


@pytest.mark.parametrize("sink", [0, 1, 2, 3])
@pytest.mark.parametrize("window", [None, 0, 1, 2, 5])
@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 7, 12])
def test_retention_matches_the_raven_pop_rule(sink, window, num_chunks):
    cache = ChunkKVCache(1, sink=sink, window=window)
    for index in range(num_chunks):
        _commit(cache, rows=index + 1, seed=index)
    assert cache.retained_indices == _raven_retained(sink, window, num_chunks)
    assert cache.committed_chunks == num_chunks


def test_window_none_never_evicts():
    cache = ChunkKVCache(2, sink=0, window=None)
    for index in range(6):
        _commit(cache, rows=3, seed=index)
    assert cache.retained_indices == list(range(6))
    assert cache.retained_rows == 18


def test_window_zero_keeps_only_the_sink():
    cache = ChunkKVCache(2, sink=2, window=0)
    for index in range(5):
        _commit(cache, rows=2, seed=index)
    assert cache.retained_indices == [0, 1]


def test_sink_zero_window_zero_retains_nothing_after_the_first_step():
    cache = ChunkKVCache(1, sink=0, window=0)
    _commit(cache, rows=4)
    assert cache.retained_indices == []
    assert cache.retained(0) is None


def test_eviction_accounts_for_unequal_chunk_lengths():
    # text (5 rows) + chunks of 78 / 76 / 24 rows, sink=1 window=1: the text is
    # pinned, only the newest media chunk is kept.
    cache = ChunkKVCache(2, sink=1, window=1)
    for rows in (5, 78, 76, 24):
        _commit(cache, rows=rows, seed=rows)
    assert cache.retained_indices == [0, 3]
    assert cache.chunk_lens == [5, 24]
    assert cache.retained_rows == 29
    keys, values = cache.retained(0)
    assert keys.shape[0] == 29 and values.shape[0] == 29


def test_retained_index_set_is_a_pure_prediction():
    cache = ChunkKVCache(1, sink=2, window=2)
    assert cache.retained_index_set(6) == [0, 1, 4, 5]
    assert cache.retained_index_set(3) == [0, 1, 2]
    assert cache.retained_index_set(0) == []


# --- reads -------------------------------------------------------------------


def test_retained_concatenates_in_time_order():
    cache = ChunkKVCache(1, sink=3, window=None)
    parts = []
    for index, rows in enumerate((2, 3, 4)):
        k, v = _kv(rows, index)
        cache.stage(0, k, v)
        cache.commit()
        parts.append(k)
    keys, _ = cache.retained(0)
    assert torch.equal(keys, torch.cat(parts, dim=0))


def test_records_are_independent_of_the_caller_buffers():
    cache = ChunkKVCache(1, sink=1, window=None)
    k, v = _kv(3, 0)
    cache.stage(0, k, v)
    cache.commit()
    k.zero_()
    v.zero_()
    keys, values = cache.retained(0)
    assert float(keys.abs().sum()) > 0.0
    assert float(values.abs().sum()) > 0.0


def test_copy_false_transfers_ownership_of_the_exact_tensor():
    # The attention module uses this for V, which is already a private buffer:
    # one copy instead of two. The contract is ownership, so the cache holds the
    # very tensor that was handed over.
    cache = ChunkKVCache(1, sink=1, window=None)
    k, v = _kv(3, 0)
    expected = v.clone()
    cache.stage(0, k, v, copy_key=True, copy_value=False)
    cache.commit()
    keys, values = cache.retained(0)
    assert values.data_ptr() == v.data_ptr()
    assert torch.equal(values, expected)
    # the key was copied, so writing through the caller's buffer cannot reach it
    k.zero_()
    assert float(keys.abs().sum()) > 0.0


def test_copy_false_still_materialises_a_non_contiguous_tensor():
    cache = ChunkKVCache(1, sink=1, window=None)
    k, v = _kv(4, 1)
    view = v.transpose(0, 1)[:, ::2]  # non-contiguous, and a view of a bigger buffer
    expected = view.clone().transpose(0, 1).contiguous()
    cache.stage(0, k[::2], view.transpose(0, 1), copy_value=False)
    cache.commit()
    _, values = cache.retained(0)
    assert values.is_contiguous()
    assert torch.equal(values, expected)
    v.zero_()
    assert float(values.abs().sum()) > 0.0


def test_layer_index_bounds_are_checked():
    cache = ChunkKVCache(2, sink=1, window=1)
    with pytest.raises(CacheError):
        cache.retained(2)
    with pytest.raises(CacheError):
        cache.stage(5, *_kv(1, 0))


# --- commit protocol ---------------------------------------------------------


def test_commit_requires_every_layer():
    cache = ChunkKVCache(3, sink=1, window=1)
    cache.stage(0, *_kv(2, 0))
    cache.stage(2, *_kv(2, 1))
    with pytest.raises(CacheError, match="unstaged"):
        cache.commit()
    assert cache.committed_chunks == 0


def test_double_stage_of_one_layer_is_rejected():
    cache = ChunkKVCache(2, sink=1, window=1)
    cache.stage(0, *_kv(2, 0))
    with pytest.raises(CacheError, match="staged twice"):
        cache.stage(0, *_kv(2, 1))


def test_layers_must_agree_on_the_chunk_length():
    cache = ChunkKVCache(2, sink=1, window=1)
    cache.stage(0, *_kv(4, 0))
    with pytest.raises(CacheError, match="rows"):
        cache.stage(1, *_kv(5, 1))


def test_stage_shape_validation():
    cache = ChunkKVCache(1, sink=1, window=1)
    with pytest.raises(CacheError, match=r"\[rows, heads, head_dim\]"):
        cache.stage(0, torch.zeros(3, 4), torch.zeros(3, 4))
    with pytest.raises(CacheError, match="row counts differ"):
        cache.stage(0, torch.zeros(3, 2, 4), torch.zeros(4, 2, 4))


def test_all_layers_advance_together():
    # The eviction step happens once, on commit; a mid-chunk read still sees the
    # previous history, so layer 0 and layer N attend to the same context.
    cache = ChunkKVCache(3, sink=1, window=0)
    _commit(cache, rows=5, seed=1)  # text
    before = [cache.retained(layer)[0].shape[0] for layer in range(3)]
    for layer in range(3):
        # staging layer by layer must not move anything for the later layers
        cache.stage(layer, *_kv(7, 10 + layer))
        assert [cache.retained(i)[0].shape[0] for i in range(3)] == before
    cache.commit()
    assert cache.retained_indices == [0]
    assert all(cache.retained(layer)[0].shape[0] == 5 for layer in range(3))


def test_discard_pending_clears_a_partial_chunk():
    cache = ChunkKVCache(3, sink=1, window=None)
    cache.stage(0, *_kv(2, 0))
    assert cache.has_pending
    cache.discard_pending()
    assert not cache.has_pending
    _commit(cache, rows=2, seed=1)
    assert cache.committed_chunks == 1


def test_reset_forgets_the_step_counter():
    cache = ChunkKVCache(1, sink=1, window=1)
    _commit(cache, rows=2)
    cache.reset()
    assert cache.committed_chunks == 0
    assert cache.retained(0) is None


def test_record_carries_role_and_row_count():
    cache = ChunkKVCache(2, sink=2, window=None)
    record = _commit(cache, rows=6, role="text")
    assert record.role == "text"
    assert record.rows == 6
    assert record.index == 0
    assert sorted(record.keys) == [0, 1]
