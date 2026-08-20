"""Per-layer chunk KV cache with attention-sink + sliding-window eviction (M2).

Model
-----
The cache is a list of **chunk records**, in time order. Chunk 0 is the text
prefill; every media chunk that is committed appends one more record. A record
holds, for every layer, the chunk's own post-QK-norm / post-RoPE keys and its
raw (unnormalised, unrotated) values -- exactly what the attention module
computed for that chunk's rows, never a merged tensor.

Retention is counted in chunks, matching RAVEN's ``utils/naive_cache.py``:

* the first ``sink`` chunks are pinned forever (chunk 0, the text, is inside
  that count -- ``sink=2`` means text + first media chunk);
* the most recent ``window`` chunks form the sliding history;
* ``window=None`` disables eviction entirely, ``window=0`` keeps only the sink.

For ``n`` committed chunks the retained set is therefore
``[0, sink) | [max(sink, n - window), n)``, which is what RAVEN's one-chunk-
per-step pop converges to.

Commit protocol
---------------
A cached forward stages one ``(key, value)`` pair per layer while the block
stack runs, and the model commits **once**, after the last layer:

    for every layer: cache.stage(layer_idx, k, v)
    cache.commit()          # appends the record, evicts, advances one step

Advancing per layer instead would let layer 0 evict against ``n + 1`` chunks
while layer 49 still evicts against ``n`` -- the layers would drift out of
alignment and each would be attending to a different history. ``commit()``
refuses a partial stage set, and a read-only (noise) forward never stages at
all, so a dropped chunk cannot half-land.

The last chunk of a clip needs no clean fill: nothing after it reads that
history. That decision belongs to the sampler; this module only refuses to
commit something inconsistent.

Where the records live: ``storage``
-----------------------------------
The *canonical* copy of every record can sit on the compute device or on the
host. The choice is one constructor argument and changes nothing else -- not
the dtype, not the byte count, not the positions the keys were rotated with,
not the retention policy:

``storage='gpu'`` (default)
    The record is whatever the attention module staged, on the device it
    staged it from. This is the historical behaviour and the default so that
    every existing caller keeps its old memory profile until it opts in.
``storage='cpu_pinned'``
    Every staged tensor is copied into an **owned, contiguous, page-locked**
    host buffer by a *blocking* D2H copy, and the device tensor is dropped.
    Page-locked is what makes the H2D read back a DMA instead of a staging
    copy through a bounce buffer.
``storage='cpu'``
    The same, in ordinary pageable host memory.

A ``cpu_pinned`` cache that cannot get page-locked memory (no accelerator, a
host allocator that refuses, a driver that has run out of pinnable pages) does
**not** fail the rollout: it falls back to pageable host memory, flips
:attr:`ChunkKVCache.actual_storage` to ``'cpu'``, and records the reason in
:attr:`ChunkKVCache.warnings` and in :meth:`ChunkKVCache.stats`. The fallback
is sticky -- one failed allocation is enough to conclude the allocator will not
serve this process -- and it is a *reported* degradation, never a silent one:
the numbers are identical, only the bandwidth is not.

Reading a history back: the single merged slot
----------------------------------------------
:meth:`ChunkKVCache.retained` concatenates the retained records for one layer
and hands back one pair of tensors. That is fine for a probe or a test, but as
the attention path it costs **two** device buffers at once -- the gathered
``past`` and then the ``cat((past, current))`` merge -- and with a host-side
canonical copy it would cost a third (the H2D landing buffer).

The attention path therefore uses the assembly API instead:

    spec = cache.retained_spec(layer_idx)            # rows/heads/dim/dtype
    merged_k = k.new_empty((spec.rows + rows, heads, dim))
    merged_v = k.new_empty((spec.rows + rows, heads, dim))
    cache.copy_retained_into(layer_idx, merged_k[:spec.rows], merged_v[:spec.rows])
    merged_k[spec.rows:].copy_(k)                    # this chunk's own rows
    merged_v[spec.rows:].copy_(v)

One allocation per layer, filled in place: the retained rows land straight in
the prefix (H2D out of the canonical host buffers, or D2D out of the device
ones) and the current chunk's rows in the tail. Nothing but that one merged
pair is ever device-resident, and the cache keeps **no** reference to it -- it
is the caller's buffer, alive for exactly one layer's attention call.

Both backends go through the same two calls, so a ``gpu`` cache and a ``cpu``
cache hand the attention backend the same values in the same layout. Every copy
on this path is blocking (``non_blocking`` is never set): after
:meth:`copy_retained_into` or :meth:`stage` returns, no DMA is in flight, so a
cancellation or an exception can drop the pending set and free buffers without
waiting for anything.

Import weight: torch only. ComfyUI is *not* imported here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "CacheError",
    "ChunkRecord",
    "ChunkKVCache",
    "RetainedSpec",
    "KV_STORAGES",
    "DEFAULT_KV_STORAGE",
]

_LOG = logging.getLogger(__name__)

#: Where the canonical copy of a record may live.
KV_STORAGES: Tuple[str, ...] = ("gpu", "cpu_pinned", "cpu")

#: The historical behaviour, and the default, so that a caller that does not
#: know about offload keeps exactly the memory profile it had.
DEFAULT_KV_STORAGE = "gpu"


class CacheError(RuntimeError):
    """A cache operation that violates the commit or retention protocol."""


def _own(tensor: torch.Tensor, copy: bool) -> torch.Tensor:
    """Return a tensor the cache may keep for the life of a chunk record."""
    tensor = tensor.detach()
    if copy:
        return tensor.clone()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _empty_pinned(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Allocate page-locked host memory.

    A module-level function on purpose: it is the seam a test monkeypatches to
    prove that the pin-failure fallback is reached, and pinning is the one step
    of the offload path whose availability is a property of the machine rather
    than of this code.
    """
    return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)


def _empty_pageable(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=dtype, device="cpu")


def _is_pinned(tensor: torch.Tensor) -> bool:
    """``tensor.is_pinned()``, defensively.

    Some builds answer this by asking an accelerator that is not there and
    raise instead of returning False; and at least one (torch on macOS with the
    MPS host allocator) accepts ``pin_memory=True`` and then reports the result
    as unpinned. Both are "not page-locked as far as this process can tell",
    which is exactly the condition the fallback exists to report.
    """
    try:
        return bool(tensor.is_pinned())
    except (RuntimeError, NotImplementedError):
        return False


def _nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _storage_nbytes(tensor: torch.Tensor) -> int:
    """Bytes of the *whole* buffer ``tensor`` is a view of.

    Used for the merged-slot accounting: the caller hands
    :meth:`ChunkKVCache.copy_retained_into` the prefix slice of a merged
    ``[past + current, heads, dim]`` buffer, so the slice's own size would
    under-report the allocation that is actually resident.
    """
    storage = tensor.untyped_storage()
    nbytes = getattr(storage, "nbytes", None)
    return int(nbytes() if callable(nbytes) else storage.size())


@dataclass
class ChunkRecord:
    """One committed chunk: its row count and per-layer K/V.

    ``keys``/``values`` map ``layer_idx -> [rows, heads, head_dim]``. Keys are
    post-QK-norm and post-RoPE (absolute positions, so they never need
    re-basing); values are the raw projection output. Which device they are on
    is the cache's ``storage`` choice; the shapes, the dtype and the values are
    the same either way.
    """

    index: int
    rows: int
    role: str
    keys: Dict[int, torch.Tensor]
    values: Dict[int, torch.Tensor]


@dataclass(frozen=True)
class RetainedSpec:
    """What one layer's retained history is, without materialising it.

    Enough for the caller to allocate the merged slot: the row count it has to
    reserve in front of its own rows, the head geometry those rows must match,
    and the dtype/device the canonical copy is held in. ``device`` is the
    *canonical* device -- with a host-side cache it is ``cpu`` while the
    attention runs on CUDA, which is the point of the whole arrangement and not
    a mismatch.
    """

    rows: int
    heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device


class ChunkKVCache:
    """Chunk-record K/V cache for a single-batch chunk-causal rollout.

    Parameters
    ----------
    num_layers:
        Number of DiT blocks that will stage into this cache.
    sink:
        Chunks pinned from the start of the rollout, text chunk included.
    window:
        Most recent chunks retained. ``None`` disables eviction; ``0`` keeps
        only the sink.
    storage:
        Where the canonical copy of every record lives: ``'gpu'`` (default,
        the historical behaviour), ``'cpu_pinned'`` or ``'cpu'``. See "Where
        the records live" in the module docstring.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        sink: int,
        window: Optional[int],
        storage: str = DEFAULT_KV_STORAGE,
    ) -> None:
        if int(num_layers) <= 0:
            raise CacheError(f"num_layers must be positive, got {num_layers!r}")
        if int(sink) != sink or sink < 0:
            raise CacheError(f"sink must be a non-negative integer, got {sink!r}")
        if window is not None and (int(window) != window or window < 0):
            raise CacheError(f"window must be None or a non-negative integer, got {window!r}")
        if storage not in KV_STORAGES:
            raise CacheError(
                f"storage must be one of {KV_STORAGES}, got {storage!r}"
            )

        self.num_layers = int(num_layers)
        self.sink = int(sink)
        self.window: Optional[int] = None if window is None else int(window)

        self._requested_storage = str(storage)
        #: Resolved lazily-but-stickily: ``cpu_pinned`` stays ``cpu_pinned``
        #: until an allocation proves the host allocator will not serve it.
        self._actual_storage = str(storage)
        self._pin_fallback_reason: Optional[str] = None
        self._warnings: List[str] = []

        self._records: List[ChunkRecord] = []
        self._committed = 0  # chunks ever committed, evictions included
        self._pending_keys: Dict[int, torch.Tensor] = {}
        self._pending_values: Dict[int, torch.Tensor] = {}

        # transfer accounting; see stats()
        self._d2h_bytes = 0
        self._d2h_calls = 0
        self._h2d_bytes = 0
        self._h2d_calls = 0
        self._peak_gpu_slot_bytes = 0
        self._peak_canonical_cpu_bytes = 0

    # -- storage introspection -------------------------------------------

    @property
    def requested_storage(self) -> str:
        """What the caller asked for."""
        return self._requested_storage

    @property
    def actual_storage(self) -> str:
        """What the records are really in; differs after a pin fallback."""
        return self._actual_storage

    @property
    def canonical_on_host(self) -> bool:
        """True when records live in host memory, whatever staged them."""
        return self._actual_storage != "gpu"

    @property
    def pin_fallback(self) -> bool:
        return self._pin_fallback_reason is not None

    @property
    def pin_fallback_reason(self) -> Optional[str]:
        return self._pin_fallback_reason

    @property
    def warnings(self) -> Tuple[str, ...]:
        """Degradations that were taken rather than raised."""
        return tuple(self._warnings)

    # -- introspection --------------------------------------------------

    @property
    def committed_chunks(self) -> int:
        """Chunks ever committed, including ones later evicted."""
        return self._committed

    @property
    def retained_chunks(self) -> int:
        return len(self._records)

    @property
    def retained_indices(self) -> List[int]:
        """Time-ordered indices of the chunks still held."""
        return [record.index for record in self._records]

    @property
    def retained_rows(self) -> int:
        """Rows held across all retained chunks (every layer holds the same)."""
        return sum(record.rows for record in self._records)

    @property
    def chunk_lens(self) -> List[int]:
        """Row counts of the retained chunks, in time order."""
        return [record.rows for record in self._records]

    @property
    def has_pending(self) -> bool:
        return bool(self._pending_keys)

    @property
    def canonical_cpu_bytes(self) -> int:
        """Host bytes the canonical records (and any pending stage) occupy.

        Zero for a ``gpu`` cache, by definition: nothing of it is on the host.
        """
        if not self.canonical_on_host:
            return 0
        total = 0
        for record in self._records:
            for table in (record.keys, record.values):
                for tensor in table.values():
                    total += _nbytes(tensor)
        for table in (self._pending_keys, self._pending_values):
            for tensor in table.values():
                total += _nbytes(tensor)
        return total

    def retained_index_set(self, num_chunks: int) -> List[int]:
        """Indices ``[0, n)`` would keep under this policy, for ``n`` chunks."""
        n = int(num_chunks)
        if self.window is None:
            return list(range(n))
        head = min(self.sink, n)
        tail_start = max(head, n - self.window)
        return list(range(head)) + list(range(tail_start, n))

    # -- reads ----------------------------------------------------------

    def retained(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Concatenated retained ``(keys, values)`` for one layer, in time order.

        ``None`` when nothing is retained, which is what an empty prefix looks
        like to the attention module.

        The tensors come back on the **canonical** device, so with a host-side
        cache this is a host concatenation and not an implicit H2D. It is the
        introspection read (probes, tests, the parity harness); the attention
        path uses :meth:`retained_spec` + :meth:`copy_retained_into` instead, so
        that no second device buffer is ever needed. See "Reading a history
        back" in the module docstring.
        """
        self._check_layer(layer_idx)
        if not self._records:
            return None
        keys = [record.keys[layer_idx] for record in self._records]
        values = [record.values[layer_idx] for record in self._records]
        if len(keys) == 1:
            return keys[0], values[0]
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def past_rows(self, layer_idx: int) -> int:
        """Rows one layer's retained history would contribute to a merge.

        The per-layer spelling of the :attr:`retained_rows` property: every
        layer stages the same rows for a chunk (``commit()`` enforces it), so
        the number is the same, but the caller that is about to size a merged
        slot for *this* layer should ask about this layer.
        """
        self._check_layer(layer_idx)
        return sum(record.rows for record in self._records)

    def retained_spec(self, layer_idx: int) -> Optional[RetainedSpec]:
        """Shape/dtype of one layer's retained history, or ``None`` if empty.

        Reads the first record's key tensor for the head geometry; every record
        of a rollout carries the same one, and :meth:`copy_retained_into`
        re-checks every record it copies, so a lie here cannot become a silent
        mis-assembly.
        """
        self._check_layer(layer_idx)
        if not self._records:
            return None
        first = self._records[0].keys[layer_idx]
        return RetainedSpec(
            rows=self.past_rows(layer_idx),
            heads=int(first.shape[1]),
            head_dim=int(first.shape[2]),
            dtype=first.dtype,
            device=first.device,
        )

    def copy_retained_into(
        self,
        layer_idx: int,
        key_dst: torch.Tensor,
        value_dst: torch.Tensor,
    ) -> int:
        """Fill ``key_dst``/``value_dst`` with one layer's retained history.

        The destinations are normally the **prefix slices** of the caller's
        merged ``[past + current, heads, head_dim]`` buffers, so the retained
        rows land directly in the tensor attention will read and no second
        device buffer is ever allocated. They must have exactly
        :meth:`past_rows` rows, and the same head geometry and dtype as the
        records -- no cast happens here, because the whole point of the host
        round trip is that it is bit-for-bit.

        Every copy is blocking. With a pinned canonical copy that is a DMA the
        call waits on; with a pageable one it is a staged copy; with a ``gpu``
        cache it is a device-to-device copy. In all three cases nothing is in
        flight when this returns, which is what lets a cancellation free the
        buffers immediately.

        Returns the number of rows copied.
        """
        self._check_layer(layer_idx)
        rows = self.past_rows(layer_idx)
        for name, dst in (("key_dst", key_dst), ("value_dst", value_dst)):
            if dst.ndim != 3:
                raise CacheError(
                    f"{name} must be [rows, heads, head_dim], got {tuple(dst.shape)}"
                )
            if int(dst.shape[0]) != rows:
                raise CacheError(
                    f"{name} has {int(dst.shape[0])} rows but layer {layer_idx} "
                    f"retains {rows}; the destination must be exactly the merged "
                    "buffer's prefix"
                )
        # The slot is what the caller allocated, not what this call writes: the
        # destinations are views into the merged K/V that stay resident for the
        # whole attention call.
        slot = _storage_nbytes(key_dst) + _storage_nbytes(value_dst)
        if slot > self._peak_gpu_slot_bytes:
            self._peak_gpu_slot_bytes = slot
        if rows == 0:
            return 0
        if key_dst.shape[1:] != value_dst.shape[1:]:
            raise CacheError(
                f"key_dst {tuple(key_dst.shape)} and value_dst "
                f"{tuple(value_dst.shape)} disagree on the head geometry"
            )

        offset = 0
        moved = 0
        for record in self._records:
            key = record.keys[layer_idx]
            value = record.values[layer_idx]
            if key.shape[1:] != key_dst.shape[1:]:
                raise CacheError(
                    f"layer {layer_idx}: record {record.index} holds head shape "
                    f"{tuple(key.shape[1:])}, destination is {tuple(key_dst.shape[1:])}"
                )
            if key.dtype != key_dst.dtype or value.dtype != value_dst.dtype:
                raise CacheError(
                    f"layer {layer_idx}: record {record.index} is "
                    f"{key.dtype}/{value.dtype} but the destination is "
                    f"{key_dst.dtype}/{value_dst.dtype}; the cache is filled and "
                    "read in one dtype, and this copy never casts"
                )
            stop = offset + record.rows
            key_dst[offset:stop].copy_(key)
            value_dst[offset:stop].copy_(value)
            moved += _nbytes(key) + _nbytes(value)
            offset = stop
        if offset != rows:  # pragma: no cover - commit() makes this unreachable
            raise CacheError(
                f"layer {layer_idx}: copied {offset} rows for a {rows}-row history"
            )
        if self.canonical_on_host:
            self._h2d_bytes += moved
            self._h2d_calls += 2 * len(self._records)
        return rows

    # -- writes ---------------------------------------------------------

    def stage(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        copy_key: bool = True,
        copy_value: bool = True,
    ) -> None:
        """Stage one layer's K/V for the chunk being committed.

        With ``storage='gpu'`` both tensors are copied by default, because the
        caller normally hands the same buffers to the attention backend, which
        owns them from that point on.

        ``copy_key=False`` / ``copy_value=False`` transfer *ownership* instead:
        the cache keeps the tensor as given and the caller must guarantee that
        (a) nothing else writes to or frees it, and (b) it is not a view of a
        larger buffer -- a view would pin the whole storage (the fused QKV
        buffer is 3x the size of K alone) for the life of the record. A
        non-contiguous tensor is made contiguous, which is a copy anyway.

        With a **host** canonical storage the two flags are inert: there is
        nothing to hand over, because the record is a freshly allocated
        contiguous host buffer that owns exactly its own rows and is filled by a
        blocking D2H copy. A staged fused-QKV *view* is read through and
        dropped, so a host record can never pin 3x its rows either. The
        caller's device tensor is untouched and remains its own, which is what
        lets the attention module go on using ``k``/``v`` for the merge after
        staging them.
        """
        self._check_layer(layer_idx)
        if layer_idx in self._pending_keys:
            raise CacheError(
                f"layer {layer_idx} staged twice before commit; a cached forward "
                "must stage each layer exactly once"
            )
        if key.ndim != 3 or value.ndim != 3:
            raise CacheError(
                "staged K/V must be [rows, heads, head_dim], got "
                f"{tuple(key.shape)} / {tuple(value.shape)}"
            )
        if key.shape[0] != value.shape[0]:
            raise CacheError(
                f"staged K/V row counts differ: {key.shape[0]} vs {value.shape[0]}"
            )
        if self._pending_keys:
            expected = next(iter(self._pending_keys.values())).shape[0]
            if key.shape[0] != expected:
                raise CacheError(
                    f"layer {layer_idx} staged {key.shape[0]} rows, but this chunk "
                    f"staged {expected} rows for an earlier layer"
                )
        self._pending_keys[layer_idx] = self._canonical(key, copy_key)
        self._pending_values[layer_idx] = self._canonical(value, copy_value)
        self._note_canonical_bytes()

    def discard_pending(self) -> None:
        """Drop a partial stage set (a forward that aborted before commit).

        Safe from anywhere, including an exception handler: every transfer this
        cache makes is blocking, so a dropped buffer is never the target of an
        in-flight DMA.
        """
        self._pending_keys.clear()
        self._pending_values.clear()

    def commit(self, *, role: str = "clean") -> ChunkRecord:
        """Append the staged chunk, evict, and advance the rollout one step.

        Called once per cached forward, after every layer has staged. Purely
        bookkeeping: the transfers already happened in :meth:`stage`, so this
        starts nothing asynchronous and there is nothing to wait for.
        """
        missing = [i for i in range(self.num_layers) if i not in self._pending_keys]
        if missing:
            raise CacheError(
                f"commit with {len(missing)} of {self.num_layers} layers unstaged "
                f"(first missing: {missing[0]}); all layers must stage before the "
                "cache advances"
            )
        rows = next(iter(self._pending_keys.values())).shape[0]
        record = ChunkRecord(
            index=self._committed,
            rows=int(rows),
            role=str(role),
            keys=self._pending_keys,
            values=self._pending_values,
        )
        self._pending_keys = {}
        self._pending_values = {}

        self._records.append(record)
        self._committed += 1
        self._evict()
        self._note_canonical_bytes()
        return record

    def _evict(self) -> None:
        keep = set(self.retained_index_set(self._committed))
        if len(keep) == len(self._records):
            return
        # Records are dropped, not sliced: releasing the tensors is the point.
        self._records = [record for record in self._records if record.index in keep]

    def reset(self) -> None:
        """Forget everything, including the step counter.

        The transfer counters are *not* reset: they measure what this cache
        object has moved over its lifetime, which is what a report at the end of
        a run wants. The storage resolution (including a pin fallback that
        already happened) is not re-litigated either.
        """
        self._records = []
        self._committed = 0
        self.discard_pending()

    # -- reporting ------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Everything the node/sampler layer needs to report about this cache.

        ``h2d``/``d2h`` are named for the production case and are counted only
        when the canonical copy is on the host -- a ``gpu`` cache moves nothing
        across the PCIe bus, so both stay zero. ``d2h`` is charged in
        :meth:`stage` (one call per tensor, so two per layer per chunk),
        ``h2d`` in :meth:`copy_retained_into` (two per retained record).

        ``peak_gpu_slot_bytes`` is the largest merged K+V slot a caller has
        handed to :meth:`copy_retained_into`, measured on the *whole* buffer the
        destination is a view of, i.e. ``2 * (past + current) * heads *
        head_dim * itemsize``. It is the one device allocation this design
        allows to exist at a time, so it is the number a memory budget needs.
        """
        return {
            "requested_storage": self._requested_storage,
            "actual_storage": self._actual_storage,
            "pin_fallback": self.pin_fallback,
            "pin_fallback_reason": self._pin_fallback_reason,
            "warnings": list(self._warnings),
            "canonical_cpu_bytes": self.canonical_cpu_bytes,
            "peak_canonical_cpu_bytes": self._peak_canonical_cpu_bytes,
            "peak_gpu_slot_bytes": self._peak_gpu_slot_bytes,
            "h2d_bytes": self._h2d_bytes,
            "h2d_calls": self._h2d_calls,
            "d2h_bytes": self._d2h_bytes,
            "d2h_calls": self._d2h_calls,
            "committed_chunks": self._committed,
            "retained_chunks": len(self._records),
            "retained_rows": self.retained_rows,
            "num_layers": self.num_layers,
            "sink": self.sink,
            "window": self.window,
        }

    def report(self) -> str:
        """One log line: where the K/V are, how big, and what they cost."""
        stats = self.stats()
        gib = float(1 << 30)
        storage = stats["actual_storage"]
        if stats["pin_fallback"]:
            storage = f"{storage} (requested {stats['requested_storage']})"
        return (
            f"KV cache: storage={storage}, "
            f"chunks={stats['retained_chunks']}/{stats['committed_chunks']}, "
            f"rows={stats['retained_rows']}, "
            f"host={stats['canonical_cpu_bytes'] / gib:.3f} GiB "
            f"(peak {stats['peak_canonical_cpu_bytes'] / gib:.3f} GiB), "
            f"peak merged slot={stats['peak_gpu_slot_bytes'] / gib:.3f} GiB, "
            f"h2d={stats['h2d_bytes'] / gib:.3f} GiB in {stats['h2d_calls']} copies, "
            f"d2h={stats['d2h_bytes'] / gib:.3f} GiB in {stats['d2h_calls']} copies"
        )

    # -- internals ------------------------------------------------------

    def _canonical(self, tensor: torch.Tensor, copy: bool) -> torch.Tensor:
        """The tensor this cache will keep for the life of a record."""
        tensor = tensor.detach()
        if not self.canonical_on_host:
            return _own(tensor, copy)
        destination = self._allocate_host(tensor.shape, tensor.dtype)
        # Blocking, and from whatever layout the caller staged: a fused-QKV
        # view is read through here and never becomes the record.
        destination.copy_(tensor)
        self._d2h_bytes += _nbytes(destination)
        self._d2h_calls += 1
        return destination

    def _allocate_host(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """An owned, contiguous host buffer of exactly ``shape``.

        Page-locked while ``actual_storage == 'cpu_pinned'``. The first failure
        -- an allocator that raises, or one that returns memory it will not call
        pinned -- flips the whole cache to pageable and is reported; it is not
        retried per tensor, because the answer would not change and each retry
        costs an allocation.
        """
        if self._actual_storage == "cpu_pinned":
            try:
                buffer = _empty_pinned(shape, dtype)
            except (RuntimeError, NotImplementedError, MemoryError) as exc:
                self._fall_back_to_pageable(f"{type(exc).__name__}: {exc}")
            else:
                if _is_pinned(buffer):
                    return buffer
                self._fall_back_to_pageable(
                    "the host allocator accepted pin_memory=True but reports the "
                    "result as not page-locked"
                )
                return buffer
        return _empty_pageable(shape, dtype)

    def _fall_back_to_pageable(self, reason: str) -> None:
        self._actual_storage = "cpu"
        self._pin_fallback_reason = reason
        message = (
            f"pinned host memory for the KV cache is unavailable ({reason}); "
            "falling back to pageable host memory -- the values are identical, "
            "the H2D transfers are slower"
        )
        self._warnings.append(message)
        _LOG.warning("raven_streaming: %s", message)

    def _note_canonical_bytes(self) -> None:
        if not self.canonical_on_host:
            return
        current = self.canonical_cpu_bytes
        if current > self._peak_canonical_cpu_bytes:
            self._peak_canonical_cpu_bytes = current

    def _check_layer(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or not (0 <= layer_idx < self.num_layers):
            raise CacheError(
                f"layer index {layer_idx!r} outside [0, {self.num_layers})"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ChunkKVCache(layers={self.num_layers}, sink={self.sink}, "
            f"window={self.window}, storage={self._actual_storage}, "
            f"committed={self._committed}, "
            f"retained={self.retained_indices}, rows={self.retained_rows})"
        )
