"""Incremental (streaming) coordinator for the MiniMax H3 video VAE decoder.

Geometry
--------
The upstream VAE (``comfy.ldm.minimax.vae.MiniMaxH3VideoVAE``) derives its
temporal chunking from ``clip_length=17``, ``vae_ratio_t=4`` and
``token_drop=3``::

    tokens_chunk_size = ceil(17 / 4)        = 5
    token_overlap     = (-3) % 5            = 2
    frame_pre_padding = (-17) % 4           = 3
    frame_overlap     = max(2*4 - 3, 0)     = 5
    chunk_dec         = 5 * 4               = 20

Chunk ``i`` reads latents ``[5i, 5i+7)`` -> the decoder emits ``7*4 = 28``
frames, which are split at ``chunk_dec``:

* ``j=0``: frames ``[0, 20)`` minus the 3 pre-padding frames -> **17 frames**
  (cross-faded over ``frame_overlap=5`` with the previous chunk's tail)
* ``j=1``: frames ``[20, 28)`` minus the 3 pre-padding frames -> **5 frames**
  held as ``dec_overlap`` for the next chunk, and flushed after the last one.

So the pipeline is causal in latents with a **2-latent lookahead**: 7 latents
finalize 17 frames, and the stream ends with a 5-frame flush.  A ``5k+2``
latent stream yields exactly ``17k+5`` frames.

What this module adds
---------------------
:class:`IncrementalVideoDecoder` runs that machine without ever materialising
the full output tensor.  Its entire state is

* pending latents (never more than what the caller pushed at once, and trimmed
  down to the ``<= 7`` still needed),
* ``dec_overlap``: exactly 5 decoded frames,
* an integer frame plan (counters only).

All heavy operators come from an injected decoder object implementing
:class:`VideoChunkDecoder` (``_adaptive_decode`` and ``blend``, optionally
``_finalize_pixels``), so the coordinator itself is framework-agnostic and can
be tested against a pure-numpy fake.

:func:`reference_decode_temporal` is a faithful port of upstream
``decode_temporal`` used to prove the streaming machine is bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = [
    "MIN_SUPPORTED_LATENTS",
    "MIN_SUPPORTED_FRAMES",
    "ShortVideoStreamError",
    "VideoChunkParams",
    "VideoChunkDecoder",
    "FrameBatch",
    "IncrementalVideoDecoder",
    "reference_decode_temporal",
    "minimax_decoder_adapter",
    "ComparisonVerdict",
    "summarize_decode_comparison",
    "torch_cat",
]

# A concat callable: (parts, dim) -> concatenated.  Defaults to torch.cat.
ConcatFn = Callable[[Sequence[Any], int], Any]

#: Shortest clip the product supports: k = 1 on the encoder's 5k+2 latent grid.
#: k = 0 (2 latents -> 5 frames) is deliberately out of scope for v0.1.
MIN_SUPPORTED_LATENTS = 7
MIN_SUPPORTED_FRAMES = 22


class ShortVideoStreamError(ValueError):
    """Raised when a stream is shorter than the minimum supported clip."""


def torch_cat(parts: Sequence[Any], dim: int) -> Any:
    import torch  # local import: the coordinator itself does not need torch

    return torch.cat(list(parts), dim=dim)


@dataclass(frozen=True)
class VideoChunkParams:
    """Temporal geometry of the MiniMax H3 video VAE decoder."""

    clip_length: int = 17
    vae_ratio_t: int = 4
    token_drop: int = 3

    @classmethod
    def from_vae(cls, model: Any) -> "VideoChunkParams":
        """Read the geometry off a real ``MiniMaxH3VideoVAE`` (or its wrapper)."""
        inner = getattr(model, "first_stage_model", model)
        return cls(
            clip_length=int(getattr(inner, "clip_length", 17)),
            vae_ratio_t=int(getattr(inner, "vae_ratio_t", 4)),
            token_drop=int(getattr(inner, "token_drop", 3)),
        )

    # -- derived ------------------------------------------------------------

    @property
    def tokens_chunk_size(self) -> int:
        return math.ceil(self.clip_length / self.vae_ratio_t)

    @property
    def token_overlap(self) -> int:
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_pre_padding(self) -> int:
        return (-self.clip_length) % self.vae_ratio_t

    @property
    def frame_overlap(self) -> int:
        return max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

    @property
    def chunk_dec(self) -> int:
        return self.tokens_chunk_size * self.vae_ratio_t

    @property
    def split_count(self) -> int:
        return int(self.token_drop > 0) + 1

    @property
    def latents_per_step(self) -> int:
        """Latents consumed per finalized chunk (5)."""
        return self.tokens_chunk_size

    @property
    def latents_needed(self) -> int:
        """Latents that must be available before a chunk can run (7)."""
        return self.tokens_chunk_size + self.token_overlap

    @property
    def lookahead_latents(self) -> int:
        """Latents of lookahead beyond the chunk being finalized (2)."""
        return self.token_overlap

    @property
    def frames_per_step(self) -> int:
        """Frames finalized per chunk (17)."""
        return min(self.chunk_dec, self.latents_needed * self.vae_ratio_t) - self.frame_pre_padding

    @property
    def tail_frames(self) -> int:
        """Frames flushed after the last chunk (5)."""
        return self.frame_overlap

    # -- supported range ----------------------------------------------------

    @property
    def min_supported_latents(self) -> int:
        """Shortest supported clip in latents: one full chunk plus lookahead."""
        return self.latents_needed

    @property
    def min_supported_frames(self) -> int:
        """Frames the shortest supported clip decodes to (22)."""
        return self.total_frames(self.min_supported_latents)

    def short_stream_message(self, z_len: int) -> str:
        return (
            "video stream is too short: {} latent(s) would decode to {} frame(s). "
            "The minimum supported clip is {} latents / {} frames (k=1 on the "
            "encoder's 5k+2 grid). k=0 (2 latents / 5 frames) is not supported in "
            "v0.1 - generate at least {} latents.".format(
                z_len, self.total_frames(z_len), self.min_supported_latents,
                self.min_supported_frames, self.min_supported_latents,
            )
        )

    def check_stream_length(self, z_len: int) -> None:
        """Raise :class:`ShortVideoStreamError` for an unsupported clip length."""
        if 0 < z_len < self.min_supported_latents:
            raise ShortVideoStreamError(self.short_stream_message(z_len))

    # -- upstream plan (faithful ports) -------------------------------------

    def temporal_chunks(self, z_len: int) -> Tuple[int, int]:
        """Port of ``_decode_temporal_chunks``: (pad_tokens, num_chunks)."""
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens

        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        return pad_tokens, num_chunks

    def pad_frames(self, z_len: int, pad_tokens: int) -> int:
        """Port of ``_decode_temporal_pad_frames``.

        ``z_len`` is the *padded* latent count.
        """
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t

        z_len_before_pad = z_len - pad_tokens
        return sum(
            (
                intra_tail
                if (z_len_before_pad + k) % self.tokens_chunk_size == 0
                else self.vae_ratio_t
            )
            for k in range(pad_tokens)
        )

    def frame_plan(self, z_len: int, num_chunks: int, pad_tokens: int) -> int:
        """Port of ``_decode_temporal_frame_plan``.  ``z_len`` is padded."""
        total_frames = 0
        final_overlap_frames = 0

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t

            for j in range(self.split_count):
                f_start_idx = j * self.chunk_dec
                f_end_idx = min(f_start_idx + self.chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        return total_frames - self.pad_frames(z_len, pad_tokens)

    def total_frames(self, z_len: int) -> int:
        """Frames the upstream decoder produces for ``z_len`` input latents."""
        if z_len <= 0:
            return 0
        if z_len == 1:
            return 1
        pad_tokens, num_chunks = self.temporal_chunks(z_len)
        return self.frame_plan(z_len + pad_tokens, num_chunks, pad_tokens)

    def eager_chunks_available(self, latents_seen: int) -> int:
        """Chunks runnable from real latents only (no tail padding needed)."""
        if latents_seen < self.latents_needed:
            return 0
        return (latents_seen - self.latents_needed) // self.latents_per_step + 1


class VideoChunkDecoder:
    """Structural protocol for the injected decoder.

    Any object providing these is accepted; the real one is
    ``MiniMaxH3VideoVAE`` itself (see :func:`minimax_decoder_adapter`).
    """

    def _adaptive_decode(self, z: Any) -> Any:  # pragma: no cover - protocol
        raise NotImplementedError

    def blend(self, a: Any, b: Any, blend_extent: int, dim: int) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _finalize_pixels(self, part: Any) -> Any:  # optional
        return part


@dataclass
class FrameBatch:
    """A contiguous run of finalized frames."""

    start_frame: int
    frames: Any  # [B, C, T, H, W]
    count: int
    chunk_index: int
    is_tail: bool = False

    @property
    def stop_frame(self) -> int:
        return self.start_frame + self.count


def _frames_of(part: Any) -> int:
    return int(part.shape[2])


def _slice_t(part: Any, start: int, stop: Optional[int] = None) -> Any:
    if stop is None:
        return part[:, :, start:]
    return part[:, :, start:stop]


class IncrementalVideoDecoder:
    """Streaming driver for the video VAE temporal chunk machine.

    ``push`` accepts latents ``[B, C, T, H, W]`` in any chunking; batches of
    finalized frames come back as soon as they are decidable.  ``finish``
    handles tail padding and the final 5-frame flush.

    A stream ends one of two ways, and they are not interchangeable:
    :meth:`finish` (the clip is complete: pad, decode the last chunk, flush the
    5-frame tail) or :meth:`abort` (the clip is being thrown away: decode
    nothing, emit nothing, drop the buffers).
    """

    def __init__(
        self,
        decoder: Any,
        params: Optional[VideoChunkParams] = None,
        concat: ConcatFn = torch_cat,
        denormalize: Optional[Callable[[Any], Any]] = None,
        allow_short_stream: bool = False,
    ) -> None:
        """``allow_short_stream`` re-enables clips below the product minimum.

        It exists for upstream-equivalence testing only - it is not part of the
        public streaming contract.  With it off (the default) anything shorter
        than :attr:`VideoChunkParams.min_supported_latents` fails loud at
        :meth:`finish`, including the single-latent still-image path.
        """
        self.decoder = decoder
        self.params = params if params is not None else VideoChunkParams.from_vae(decoder)
        self._concat = concat
        self._denormalize = denormalize
        self._allow_short_stream = bool(allow_short_stream)
        self._finalize = getattr(decoder, "_finalize_pixels", None)

        self._pending: Any = None
        self._pending_start = 0  # absolute latent index of _pending[..., 0, :, :]
        self._latents_seen = 0
        self._chunk_index = 0
        self._dec_overlap: Any = None
        self._frames_emitted = 0
        self._frame_limit: Optional[int] = None
        self._finished = False

    # -- introspection ------------------------------------------------------

    @property
    def latents_seen(self) -> int:
        return self._latents_seen

    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted

    @property
    def chunks_done(self) -> int:
        return self._chunk_index

    @property
    def pending_latents(self) -> int:
        return 0 if self._pending is None else _frames_of(self._pending)

    @property
    def lookahead_latents(self) -> int:
        return self.params.lookahead_latents

    def expected_total_frames(self, total_latents: Optional[int] = None) -> int:
        n = self._latents_seen if total_latents is None else total_latents
        return self.params.total_frames(n)

    # -- streaming ----------------------------------------------------------

    def push(self, z: Any) -> List[FrameBatch]:
        """Append latents and finalize every chunk that is now decidable."""
        if self._finished:
            raise RuntimeError("decoder already finished")
        n = _frames_of(z)
        if n <= 0:
            return []
        if self._denormalize is not None:
            z = self._denormalize(z)
        if self._pending is None:
            self._pending = z
            self._pending_start = self._latents_seen
        else:
            self._pending = self._concat([self._pending, z], 2)
        self._latents_seen += n
        return self._run_eager()

    def finish(self) -> List[FrameBatch]:
        """Close the stream: tail padding, last chunk, and the 5-frame flush."""
        if self._finished:
            return []
        self._finished = True
        total = self._latents_seen
        if total == 0:
            return []

        params = self.params
        if not self._allow_short_stream:
            params.check_stream_length(total)

        if total == 1:
            # upstream `decode` short-circuits a single latent to a single frame
            frames = self._emit_pixels(_slice_t(self.decoder._adaptive_decode(self._pending), -1))
            batch = FrameBatch(
                start_frame=0, frames=frames, count=_frames_of(frames), chunk_index=0, is_tail=True
            )
            self._frames_emitted += batch.count
            self._pending = None
            return [batch]

        pad_tokens, num_chunks = params.temporal_chunks(total)
        self._frame_limit = params.frame_plan(total + pad_tokens, num_chunks, pad_tokens)

        out: List[FrameBatch] = []
        if pad_tokens > 0 and self._chunk_index < num_chunks:
            last = _slice_t(self._pending, _frames_of(self._pending) - 1)
            self._pending = self._concat([self._pending] + [last] * pad_tokens, 2)

        while self._chunk_index < num_chunks:
            out.extend(self._run_chunk())

        if self._dec_overlap is not None:
            tail = self._emit_frames(self._dec_overlap, chunk_index=self._chunk_index - 1, is_tail=True)
            self._dec_overlap = None
            if tail is not None:
                out.append(tail)

        self._pending = None
        return out

    def abort(self) -> None:
        """Discard the stream without decoding, padding or emitting anything.

        The terminal path for a cancelled or failed run.  Every tensor the
        coordinator holds - the pending latents and the 5-frame ``dec_overlap``
        that lives on the *decode device* between chunks - is dropped here,
        which is the point: on a cancel those are the only things keeping a GPU
        allocation alive, and waiting for the object to be dereferenced leaves
        them resident for as long as the caller's traceback holds a frame.

        Deliberately *not* a flush.  ``finish`` would pad the tail and run one
        more decode to produce frames nobody asked for; the whole reason this
        exists as a separate method is that a cancel must cost zero decoder
        work.

        Idempotent, and harmless after :meth:`finish` (which has already
        emptied the same fields).  Afterwards :meth:`push` raises and
        :meth:`finish` returns ``[]``, exactly as after a normal finish - a
        cancelled stream must never hand back a partial clip.  The counters
        (:attr:`latents_seen`, :attr:`frames_emitted`, :attr:`chunks_done`) are
        left alone: they are the record of what happened, not buffered data.
        """
        self._pending = None
        self._dec_overlap = None
        self._frame_limit = None
        self._finished = True

    # -- internals ----------------------------------------------------------

    def _run_eager(self) -> List[FrameBatch]:
        out: List[FrameBatch] = []
        runnable = self.params.eager_chunks_available(self._latents_seen)
        while self._chunk_index < runnable:
            out.extend(self._run_chunk())
        return out

    def _run_chunk(self) -> List[FrameBatch]:
        params = self.params
        i = self._chunk_index
        t_start = i * params.tokens_chunk_size
        t_end = t_start + params.latents_needed

        local_start = t_start - self._pending_start
        local_end = t_end - self._pending_start
        if local_start < 0:
            raise RuntimeError("latents for chunk {} were already discarded".format(i))
        clip_z = _slice_t(self._pending, local_start, local_end)

        clip_dec = self.decoder._adaptive_decode(clip_z)
        out: List[FrameBatch] = []
        for j in range(params.split_count):
            f_start = j * params.chunk_dec
            f_end = min(f_start + params.chunk_dec, _frames_of(clip_dec))
            part = _slice_t(clip_dec, f_start, f_end)
            part = _slice_t(part, params.frame_pre_padding)
            if j == 0:
                if self._dec_overlap is not None:
                    part = self.decoder.blend(
                        self._dec_overlap, part, params.frame_overlap, -3
                    )
                    self._dec_overlap = None
                batch = self._emit_frames(part, chunk_index=i)
                if batch is not None:
                    out.append(batch)
            else:
                self._dec_overlap = _contiguous(part)

        self._chunk_index += 1
        self._trim_pending()
        return out

    def _trim_pending(self) -> None:
        """Drop latents no future chunk can reference."""
        keep_from = self._chunk_index * self.params.tokens_chunk_size
        drop = keep_from - self._pending_start
        if drop <= 0 or self._pending is None:
            return
        available = _frames_of(self._pending)
        drop = min(drop, available)
        self._pending = _slice_t(self._pending, drop)
        self._pending_start += drop

    def _emit_pixels(self, part: Any) -> Any:
        if self._finalize is not None:
            return self._finalize(part)
        return part

    def _emit_frames(self, part: Any, chunk_index: int, is_tail: bool = False) -> Optional[FrameBatch]:
        """Mirror of upstream ``decode_temporal.write_part``, operator for operator.

        Upstream is::

            part_frames = part.shape[2]
            if part_frames <= 0: return
            part = self._finalize_pixels(part)                  # whole part
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0: dec[...].copy_(part[:, :, :copy_frames])

        so the *whole* part is finalized and only then cropped to the room left
        in the output plan.  ``_finalize_pixels`` happens to be pointwise today
        (it broadcasts ``pixel_mean``/``pixel_std`` of shape ``[1,3,1,1,1]``),
        which makes crop-then-finalize algebraically equivalent - but relying on
        that would silently break the moment the hook gains any cross-frame or
        shape-dependent term.  The order is kept identical instead, including
        finalizing a part that is then entirely discarded.
        """
        count = _frames_of(part)
        if count <= 0:
            return None

        frames = self._emit_pixels(part)

        if self._frame_limit is not None:
            room = max(0, self._frame_limit - self._frames_emitted)
            if room == 0:
                return None
            if count > room:
                frames = _slice_t(frames, 0, room)
                count = room

        batch = FrameBatch(
            start_frame=self._frames_emitted,
            frames=frames,
            count=count,
            chunk_index=chunk_index,
            is_tail=is_tail,
        )
        self._frames_emitted += count
        return batch


def _contiguous(part: Any) -> Any:
    fn = getattr(part, "contiguous", None)
    if callable(fn):
        return fn()
    return part


# -- reference ------------------------------------------------------------


def reference_decode_temporal(
    decoder: Any,
    z: Any,
    params: Optional[VideoChunkParams] = None,
    concat: ConcatFn = torch_cat,
    denormalize: Optional[Callable[[Any], Any]] = None,
) -> List[Any]:
    """Faithful port of upstream ``MiniMaxH3VideoVAE.decode_temporal``.

    Returns the finalized frame parts in order (concatenate along dim 2 for the
    full tensor).  Written to mirror the upstream control flow line for line so
    that the streaming implementation can be proven equivalent.
    """
    if params is None:
        params = VideoChunkParams.from_vae(decoder)
    if denormalize is not None:
        z = denormalize(z)

    finalize = getattr(decoder, "_finalize_pixels", None)
    z_len = _frames_of(z)

    if z_len == 1:
        part = _slice_t(decoder._adaptive_decode(z), -1)
        return [finalize(part) if finalize is not None else part]

    chunk_dec = params.chunk_dec
    split_count = params.split_count
    pad_tokens, num_chunks = params.temporal_chunks(z_len)
    total_frames = params.frame_plan(z_len + pad_tokens, num_chunks, pad_tokens)

    if pad_tokens > 0:
        last = _slice_t(z, z_len - 1)
        z = concat([z] + [last] * pad_tokens, 2)

    parts: List[Any] = []
    written = 0
    dec_overlap = None

    def write_part(part: Any) -> None:
        nonlocal written
        part_frames = _frames_of(part)
        if part_frames <= 0:
            return
        part = finalize(part) if finalize is not None else part
        copy_frames = min(part_frames, max(0, total_frames - written))
        if copy_frames > 0:
            parts.append(_slice_t(part, 0, copy_frames))
            written += copy_frames

    for i in range(num_chunks):
        t_start_idx = i * params.tokens_chunk_size
        t_end_idx = t_start_idx + params.tokens_chunk_size + params.token_overlap
        clip_z = _slice_t(z, t_start_idx, t_end_idx)

        clip_dec = decoder._adaptive_decode(clip_z)

        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, _frames_of(clip_dec))
            clip_dec_chunk = _slice_t(clip_dec, f_start_idx, f_end_idx)
            clip_dec_chunk = _slice_t(clip_dec_chunk, params.frame_pre_padding)

            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = decoder.blend(
                        dec_overlap, clip_dec_chunk, params.frame_overlap, -3
                    )
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = _contiguous(clip_dec_chunk)

        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None

    return parts


@dataclass
class ComparisonVerdict:
    """Outcome of comparing the incremental decode against the official one."""

    passed: bool
    notes: List[str]
    cold_within_tolerance: Optional[bool] = None
    warm_within_tolerance: Optional[bool] = None
    official_reproducible: Optional[bool] = None
    incremental_reproducible: Optional[bool] = None


def _fmt_diff(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.3e}".format(value)


def summarize_decode_comparison(
    cold: dict,
    warm: dict,
    official_self: Optional[dict] = None,
    incremental_self: Optional[dict] = None,
    tolerance: float = 1e-6,
) -> ComparisonVerdict:
    """Turn raw diff measurements into a pass/fail plus evidence.

    Each argument is a mapping with ``max_abs_diff``, ``exact`` and optionally
    ``shape_mismatch``:

    * ``cold`` - incremental vs official on the **first** pass of each
    * ``warm`` - incremental vs official on the **last** pass of each
    * ``official_self`` - official vs official, run to run
    * ``incremental_self`` - incremental vs incremental, run to run

    The gate is the warm pair against ``tolerance``; the tolerance is never
    widened.  A cold-only failure and a non-reproducible official decoder are
    reported as *evidence* instead, because both are properties of the model and
    its kernels rather than of the streaming coordinator.
    """
    notes: List[str] = []

    def within(entry: Optional[dict]) -> Optional[bool]:
        if not entry or entry.get("max_abs_diff") is None:
            return None
        return bool(entry["max_abs_diff"] <= tolerance)

    warm_ok = within(warm)
    cold_ok = within(cold)
    passed = bool(warm_ok)

    if warm.get("shape_mismatch"):
        notes.append(
            "SHAPE MISMATCH: incremental {} vs official {}".format(*warm["shape_mismatch"])
        )

    if cold_ok is False and passed:
        notes.append(
            "COLD-KERNEL EVIDENCE: the first pass differed by {} (> tol {:g}) while a "
            "later pass is within tolerance (max|diff|={}). The coordinator is not the "
            "cause; the first decode after model load pays cuDNN autotune / lazy init. "
            "Keep --warmup >= 1 to remove it.".format(
                _fmt_diff(cold.get("max_abs_diff")), tolerance,
                _fmt_diff(warm.get("max_abs_diff")))
        )

    official_reproducible = None
    if official_self is not None and official_self.get("max_abs_diff") is not None:
        official_reproducible = bool(official_self.get("exact"))
        if not official_reproducible:
            notes.append(
                "OFFICIAL DECODER IS NOT BITWISE REPRODUCIBLE on this box: two identical "
                "model.decode() calls differ by {}. Any incremental-vs-official "
                "difference at or below that magnitude is a property of the model and "
                "its kernels, not of the streaming coordinator.".format(
                    _fmt_diff(official_self.get("max_abs_diff")))
            )

    incremental_reproducible = None
    if incremental_self is not None and incremental_self.get("max_abs_diff") is not None:
        incremental_reproducible = bool(incremental_self.get("exact"))
        if not incremental_reproducible:
            notes.append(
                "the incremental path is not bitwise reproducible either "
                "(max|diff|={}), consistent with kernel-level nondeterminism.".format(
                    _fmt_diff(incremental_self.get("max_abs_diff")))
            )

    if warm.get("exact"):
        notes.append("incremental output is BITWISE IDENTICAL to the official decode.")
    elif not passed:
        notes.append(
            "FAILED: max|diff|={} exceeds tol {:g}. The tolerance is deliberately NOT "
            "relaxed - check that every part is finalized before it is cropped, exactly "
            "as upstream write_part() does.".format(
                _fmt_diff(warm.get("max_abs_diff")), tolerance)
        )

    return ComparisonVerdict(
        passed=passed,
        notes=notes,
        cold_within_tolerance=cold_ok,
        warm_within_tolerance=warm_ok,
        official_reproducible=official_reproducible,
        incremental_reproducible=incremental_reproducible,
    )


def minimax_decoder_adapter(model: Any) -> Any:
    """Wrap a real ``MiniMaxH3VideoVAE`` so latents are denormalized like ``decode``."""

    class _Adapter(object):
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            # keep the geometry visible so VideoChunkParams.from_vae still works
            self.clip_length = int(getattr(inner, "clip_length", 17))
            self.vae_ratio_t = int(getattr(inner, "vae_ratio_t", 4))
            self.token_drop = int(getattr(inner, "token_drop", 3))

        def _adaptive_decode(self, z: Any) -> Any:
            return self._inner._adaptive_decode(z)

        def blend(self, a: Any, b: Any, blend_extent: int, dim: int) -> Any:
            return self._inner.blend(a, b, blend_extent, dim)

        def _finalize_pixels(self, part: Any) -> Any:
            return self._inner._finalize_pixels(part)

        def denormalize(self, z: Any) -> Any:
            mean = self._inner.latents_mean.view(1, -1, 1, 1, 1).to(z)
            std = self._inner.latents_std.view(1, -1, 1, 1, 1).to(z)
            return z * std + mean

    inner = getattr(model, "first_stage_model", model)
    return _Adapter(inner)
