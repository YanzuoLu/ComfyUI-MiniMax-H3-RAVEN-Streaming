"""Overlap-save streaming for the MiniMax H3 audio VAE (BigVGAN decoder).

The audio decoder is **not causal**: it is a stack of dilated convolutions and
anti-aliased up/down samplers, so every output sample depends on latents on
both sides of it.  Decoding latents ``[a, b)`` in isolation is therefore *not*
equal to the corresponding slice of a full decode - the difference is exactly
the implicit zero padding at the slice edges leaking inward.

The receptive field is finite, so the fix is classic overlap-save: decode
``[a-m, b+m)`` and keep only the samples belonging to ``[a, b)``.  Once ``m``
covers the receptive field the result is bit-comparable to the full decode.
At the true stream edges no context is dropped, which reproduces the full
decode's own zero padding exactly.

The right margin ``m`` is unavoidable **lookahead**: the stream cannot emit
audio for latent ``b-1`` before latent ``b+m-1`` exists.  ``m`` latents is
``m * 800`` samples = ``m / 40`` seconds at 32 kHz.  :func:`search_latent_margin`
measures the smallest ``m`` that meets a tolerance on a real decoder, so the
required latency is measured rather than guessed.

Everything here is pure integer planning plus generic ``[..., T]`` slicing, so
it works with torch tensors, numpy arrays, or any object supporting those.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, List, Optional, Sequence

__all__ = [
    "AudioLatentGeometry",
    "DecodeRequest",
    "OverlapSavePlanner",
    "OverlapSaveAudioDecoder",
    "MarginSearchResult",
    "BlockDiff",
    "search_latent_margin",
    "decode_overlap_save",
    "diff_block_by_block",
    "probe_shift_equivariance",
    "diff_stats",
    "saturation_stats",
    "max_abs_diff",
]


@dataclass(frozen=True)
class AudioLatentGeometry:
    """Sample/latent geometry of the audio VAE."""

    samples_per_latent: int = 800
    sample_rate: int = 32000

    @classmethod
    def from_vae(cls, model: Any) -> "AudioLatentGeometry":
        inner = getattr(model, "first_stage_model", model)
        return cls(
            samples_per_latent=int(
                getattr(inner, "samples_per_latent", getattr(inner, "hop_length", 800))
            ),
            sample_rate=int(getattr(inner, "sample_rate", 32000)),
        )

    @property
    def latents_per_second(self) -> Fraction:
        return Fraction(self.sample_rate, self.samples_per_latent)

    def latents_to_samples(self, latents: int) -> int:
        return int(latents) * self.samples_per_latent

    def latents_to_seconds(self, latents: int) -> Fraction:
        return Fraction(int(latents) * self.samples_per_latent, self.sample_rate)


@dataclass(frozen=True)
class DecodeRequest:
    """One overlap-save decode step.

    Decode latents ``[latent_start, latent_stop)``, then keep decoded samples
    ``[take_start, take_stop)`` and place them at absolute output sample index
    ``out_start``.
    """

    index: int
    latent_start: int
    latent_stop: int
    take_start: int
    take_stop: int
    out_start: int
    samples_per_latent: int = 800
    is_final: bool = False

    @property
    def latent_count(self) -> int:
        return self.latent_stop - self.latent_start

    @property
    def decoded_samples(self) -> int:
        return self.latent_count * self.samples_per_latent

    @property
    def out_samples(self) -> int:
        return self.take_stop - self.take_start

    @property
    def out_stop(self) -> int:
        return self.out_start + self.out_samples

    @property
    def left_context_samples(self) -> int:
        """Decoded samples discarded from the head of this block."""
        return self.take_start

    @property
    def right_context_samples(self) -> int:
        """Decoded samples discarded from the tail of this block."""
        return self.decoded_samples - self.take_stop


class OverlapSavePlanner:
    """Pure integer planner for overlap-save decoding.

    Offline: :meth:`plan` for a known latent count.
    Streaming: :meth:`push` as latents arrive, then :meth:`finish`.
    """

    def __init__(
        self,
        margin: int,
        block_latents: int,
        geometry: Optional[AudioLatentGeometry] = None,
        left_margin: Optional[int] = None,
        phase_align: int = 1,
        prefix_mode: bool = False,
    ) -> None:
        """``phase_align`` rounds every decode's ``latent_start`` *down* to a
        multiple of it.  Extra left context is always safe (it is trimmed away),
        so this is free insurance against a decoder whose internal strided
        resamplers are only equivariant to shifts that are a multiple of some
        period.  ``phase_align=1`` (the default) means "no constraint".

        ``prefix_mode`` decodes ``z[0:hi]`` for every block instead of a
        windowed slice: the left edge is then always the true start of the
        stream, so left-boundary error is zero *by construction* rather than by
        assumption.  It costs O(n^2) decode work, so it is a diagnostic and a
        fallback, not the default.
        """
        if margin < 0:
            raise ValueError("margin must be >= 0")
        if block_latents < 1:
            raise ValueError("block_latents must be >= 1")
        if phase_align < 1:
            raise ValueError("phase_align must be >= 1")
        self.margin = int(margin)
        self.left_margin = int(margin if left_margin is None else left_margin)
        self.block_latents = int(block_latents)
        self.phase_align = int(phase_align)
        self.prefix_mode = bool(prefix_mode)
        self.geometry = geometry or AudioLatentGeometry()
        self._next_block = 0
        self._latents_seen = 0
        self._finished = False

    # -- properties ---------------------------------------------------------

    @property
    def lookahead_latents(self) -> int:
        return self.margin

    @property
    def lookahead_samples(self) -> int:
        return self.geometry.latents_to_samples(self.margin)

    @property
    def lookahead_seconds(self) -> Fraction:
        return self.geometry.latents_to_seconds(self.margin)

    @property
    def latents_seen(self) -> int:
        return self._latents_seen

    @property
    def blocks_done(self) -> int:
        return self._next_block

    @property
    def finished(self) -> bool:
        """True once the stream was closed, by :meth:`finish` or :meth:`abort`."""
        return self._finished

    # -- planning -----------------------------------------------------------

    def _block(self, index: int, total: int) -> Optional[DecodeRequest]:
        spl = self.geometry.samples_per_latent
        a = index * self.block_latents
        if a >= total:
            return None
        b = min(a + self.block_latents, total)
        if self.prefix_mode:
            lo = 0
        else:
            lo = max(0, a - self.left_margin)
            lo -= lo % self.phase_align  # only ever adds context, never removes
        hi = min(total, b + self.margin)
        return DecodeRequest(
            index=index,
            latent_start=lo,
            latent_stop=hi,
            take_start=(a - lo) * spl,
            take_stop=(b - lo) * spl,
            out_start=a * spl,
            samples_per_latent=spl,
            is_final=b >= total,
        )

    def plan(self, total_latents: int) -> List[DecodeRequest]:
        """Full offline plan for a known-length latent sequence."""
        out: List[DecodeRequest] = []
        index = 0
        while True:
            req = self._block(index, total_latents)
            if req is None:
                break
            out.append(req)
            index += 1
        return out

    # -- streaming ----------------------------------------------------------

    def push(self, n_latents: int) -> List[DecodeRequest]:
        """Register ``n_latents`` new latents; return now-satisfiable blocks."""
        if self._finished:
            raise RuntimeError("planner already finished")
        self._latents_seen += int(n_latents)
        out: List[DecodeRequest] = []
        while True:
            a = self._next_block * self.block_latents
            b = a + self.block_latents
            # a full block plus its right margin must be available
            if b + self.margin > self._latents_seen:
                break
            req = self._block(self._next_block, self._latents_seen)
            if req is None:
                break
            # a mid-stream block must not be marked final
            req = DecodeRequest(
                index=req.index,
                latent_start=req.latent_start,
                latent_stop=req.latent_stop,
                take_start=req.take_start,
                take_stop=req.take_stop,
                out_start=req.out_start,
                samples_per_latent=req.samples_per_latent,
                is_final=False,
            )
            out.append(req)
            self._next_block += 1
        return out

    def finish(self) -> List[DecodeRequest]:
        """Close the stream: emit the remaining (edge) blocks."""
        if self._finished:
            return []
        self._finished = True
        out: List[DecodeRequest] = []
        total = self._latents_seen
        while True:
            req = self._block(self._next_block, total)
            if req is None:
                break
            out.append(req)
            self._next_block += 1
        return out

    def abort(self) -> None:
        """Close the stream *without* planning the remaining edge blocks.

        :meth:`finish` exists to emit the blocks that are still short of their
        right context - the last ``block - 1 + margin`` latents of a clip.  On a
        cancel those latents are being thrown away, so planning decodes for them
        would be work spent on output nobody will read.  This marks the plan
        closed instead: :meth:`push` then raises and :meth:`finish` returns
        ``[]``, so no request can be produced afterwards by either door.

        Idempotent, and harmless after :meth:`finish`.  ``latents_seen`` and
        ``blocks_done`` stay put - they are the record of what was planned, not
        pending state.
        """
        self._finished = True

    def required_history_latents(self) -> int:
        """Latents that must stay buffered behind the write cursor."""
        if self.prefix_mode:
            return self._latents_seen
        return self.left_margin + self.phase_align - 1


class OverlapSaveAudioDecoder:
    """Drives :class:`OverlapSavePlanner` over an injected decode callable.

    ``decode_fn(z) -> waveform`` where ``z`` is a latent slice along the last
    axis and the waveform's last axis is samples.  Nothing about the tensor
    library is assumed beyond ``[..., a:b]`` slicing and ``.shape``.
    """

    def __init__(
        self,
        decode_fn: Callable[[Any], Any],
        margin: int,
        block_latents: int,
        geometry: Optional[AudioLatentGeometry] = None,
        concat: Optional[Callable[[Sequence[Any], int], Any]] = None,
        left_margin: Optional[int] = None,
    ) -> None:
        self.decode_fn = decode_fn
        self.planner = OverlapSavePlanner(margin, block_latents, geometry, left_margin)
        self._concat = concat or _default_concat
        self._buffer: Any = None
        self._buffer_start = 0  # absolute latent index of buffer[..., 0]
        self._samples_emitted = 0

    @property
    def samples_emitted(self) -> int:
        return self._samples_emitted

    @property
    def lookahead_seconds(self) -> Fraction:
        return self.planner.lookahead_seconds

    @property
    def finished(self) -> bool:
        """True once the stream was closed, by :meth:`finish` or :meth:`abort`."""
        return self.planner.finished

    def push(self, z: Any) -> List[Any]:
        if self.planner.finished:
            # checked before the buffer is touched: a rejected push must not
            # leave latents behind in a decoder that is already closed
            raise RuntimeError("decoder already finished")
        n = int(z.shape[-1])
        if self._buffer is None:
            self._buffer = z
            self._buffer_start = self.planner.latents_seen
        else:
            self._buffer = self._concat([self._buffer, z], -1)
        requests = self.planner.push(n)
        return self._run(requests)

    def finish(self) -> List[Any]:
        return self._run(self.planner.finish())

    def abort(self) -> None:
        """Drop every buffered latent and close the stream without decoding.

        The terminal path for a cancelled or failed run.  What it releases is
        the overlap-save history: up to ``block + 2 * margin`` latents held so
        the edge blocks can still see their context.  On a cancel that context
        is only ever used to decode audio that will be discarded, so the buffer
        is dropped and the planner is closed, and ``decode_fn`` is never called
        again - not on the way out, and not by a later :meth:`finish`.

        Idempotent, and harmless after :meth:`finish`.  Afterwards
        :meth:`push` raises and :meth:`finish` returns ``[]``;
        ``samples_emitted`` is left as the record of what was actually decoded.

        ``decode_fn`` itself is kept: it belongs to the caller (the pipeline
        drops its whole decoder on cancel), and clearing it would turn a stale
        call into an ``AttributeError`` instead of the explicit
        ``RuntimeError`` above.
        """
        self._buffer = None
        self._buffer_start = 0
        self.planner.abort()

    def _run(self, requests: Sequence[DecodeRequest]) -> List[Any]:
        out: List[Any] = []
        for req in requests:
            lo = req.latent_start - self._buffer_start
            hi = req.latent_stop - self._buffer_start
            if lo < 0:
                raise RuntimeError(
                    "latent {} needed for block {} was already discarded".format(
                        req.latent_start, req.index
                    )
                )
            decoded = self.decode_fn(self._buffer[..., lo:hi])
            wave = decoded[..., req.take_start:req.take_stop]
            out.append(wave)
            self._samples_emitted += req.out_samples
            self._trim(req)
        return out

    def _trim(self, req: DecodeRequest) -> None:
        # the next block starts at (req.index + 1) * block_latents and reaches
        # back by left_margin; nothing before that can ever be referenced again
        keep_from = max(
            0,
            (req.index + 1) * self.planner.block_latents
            - self.planner.required_history_latents(),
        )
        drop = keep_from - self._buffer_start
        if drop <= 0 or self._buffer is None:
            return
        drop = min(drop, int(self._buffer.shape[-1]))
        self._buffer = self._buffer[..., drop:]
        self._buffer_start += drop


def _default_concat(parts: Sequence[Any], dim: int) -> Any:
    module = type(parts[0]).__module__ or ""
    if module.split(".")[0] == "torch":
        import torch

        return torch.cat(list(parts), dim=dim)
    import numpy as np

    return np.concatenate(list(parts), axis=dim)


def _abs(d: Any) -> Any:
    abs_fn = getattr(d, "abs", None)
    return abs_fn() if callable(abs_fn) else abs(d)


def diff_stats(a: Any, b: Any) -> dict:
    """max/mean |a - b| plus the worst sample position, over the common prefix."""
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    if n == 0:
        return {"samples": 0, "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                "worst_sample": None, "exact": True}
    d = _abs(a[..., :n] - b[..., :n])
    flat = d.reshape(-1)
    worst_flat = int(flat.argmax())
    return {
        "samples": n,
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "worst_sample": worst_flat % n,  # position along the time axis
        "exact": float(d.max()) == 0.0,
    }


def max_abs_diff(a: Any, b: Any) -> float:
    """Max |a - b| over the common prefix, for torch tensors or numpy arrays."""
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    d = _abs(a[..., :n] - b[..., :n])
    return float(d.max())


def saturation_stats(waveform: Any, limit: float = 1.0, eps: float = 1e-6) -> dict:
    """How much of a decoded waveform sits on the clipping rails?

    The BigVGAN decoder clamps its output to [-1, 1].  Latents drawn from
    ``randn`` are far off the encoder's manifold, and the decoder's response to
    them tends to rail: large stretches pinned at exactly +/-1.

    That matters for a margin search.  On a railed signal the clamp destroys the
    very differences the search is trying to measure in some places and leaves
    them intact in others, so ``max|diff|`` stops being a smooth function of the
    context length - it can plateau and wobble for reasons that have nothing to
    do with the receptive field.  A margin measured on a saturated reference is
    not a measurement of the decoder.
    """
    a = _abs(waveform)
    total = 1
    for dim in waveform.shape:
        total *= int(dim)
    at_rail = (a >= (limit - eps))
    # works for torch and numpy: bool -> sum
    try:
        rail_count = int(at_rail.sum())
    except TypeError:  # pragma: no cover - defensive
        rail_count = int(at_rail.astype("int64").sum())
    peak = float(a.max()) if total else 0.0
    try:
        rms = float(((waveform.astype("float64") if hasattr(waveform, "astype")
                      else waveform) ** 2).mean()) ** 0.5
    except Exception:  # pragma: no cover - defensive
        rms = float("nan")
    return {
        "samples": total,
        "peak_abs": peak,
        "rms": rms,
        "samples_at_rail": rail_count,
        "fraction_at_rail": (rail_count / float(total)) if total else 0.0,
        "is_saturated": bool(total and (rail_count / float(total)) > 0.01),
    }


@dataclass
class BlockDiff:
    """How one overlap-save block compares with the full-sequence decode."""

    request: DecodeRequest
    total_latents: int
    max_abs_diff: float
    mean_abs_diff: float
    worst_sample_in_block: Optional[int]
    exact: bool

    @property
    def is_full_sequence(self) -> bool:
        """True when this block decoded the *entire* latent sequence.

        Such a request feeds ``decode_fn`` the identical tensor the reference
        used, so any non-zero difference cannot be an overlap-save/boundary
        effect: it is nondeterminism or a harness bug.
        """
        return self.request.latent_start == 0 and self.request.latent_stop == self.total_latents

    @property
    def worst_sample_absolute(self) -> Optional[int]:
        if self.worst_sample_in_block is None:
            return None
        return self.request.out_start + self.worst_sample_in_block

    @property
    def touches_stream_start(self) -> bool:
        return self.request.latent_start == 0

    @property
    def touches_stream_end(self) -> bool:
        return self.request.latent_stop == self.total_latents

    def to_dict(self) -> dict:
        req = self.request
        return {
            "index": req.index,
            "latent_start": req.latent_start,
            "latent_stop": req.latent_stop,
            "latent_count": req.latent_count,
            "out_start": req.out_start,
            "out_samples": req.out_samples,
            "left_context_samples": req.left_context_samples,
            "right_context_samples": req.right_context_samples,
            "is_full_sequence": self.is_full_sequence,
            "touches_stream_start": self.touches_stream_start,
            "touches_stream_end": self.touches_stream_end,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "worst_sample_in_block": self.worst_sample_in_block,
            "worst_sample_absolute": self.worst_sample_absolute,
            "exact": self.exact,
        }

    def describe(self) -> str:
        marks = []
        if self.is_full_sequence:
            marks.append("FULL-SEQUENCE")
        else:
            if self.touches_stream_start:
                marks.append("at-start")
            if self.touches_stream_end:
                marks.append("at-end")
        return (
            "  block {:>3} lat[{:>5},{:>5}) out[{:>8},{:>8}) ctx L{:>6}/R{:>6}  "
            "max={:.3e} mean={:.3e} worst@{}  {}".format(
                self.request.index, self.request.latent_start, self.request.latent_stop,
                self.request.out_start, self.request.out_stop,
                self.request.left_context_samples, self.request.right_context_samples,
                self.max_abs_diff, self.mean_abs_diff, self.worst_sample_absolute,
                " ".join(marks),
            )
        )


def diff_block_by_block(
    decode_fn: Callable[[Any], Any],
    latents: Any,
    reference: Any,
    margin: int,
    block_latents: int,
    geometry: Optional[AudioLatentGeometry] = None,
    phase_align: int = 1,
    prefix_mode: bool = False,
    left_margin: Optional[int] = None,
) -> List[BlockDiff]:
    """Compare every overlap-save block against the full-sequence decode.

    This localises error instead of collapsing it into a single number: which
    block, how far into it, and - crucially - whether the offending block was a
    *full-sequence* request, which would rule out overlap-save entirely.
    """
    geom = geometry or AudioLatentGeometry()
    total = int(latents.shape[-1])
    planner = OverlapSavePlanner(
        margin, block_latents, geom, left_margin=left_margin,
        phase_align=phase_align, prefix_mode=prefix_mode,
    )
    out: List[BlockDiff] = []
    for req in planner.plan(total):
        decoded = decode_fn(latents[..., req.latent_start:req.latent_stop])
        got = decoded[..., req.take_start:req.take_stop]
        want = reference[..., req.out_start:req.out_stop]
        stats = diff_stats(got, want)
        out.append(
            BlockDiff(
                request=req,
                total_latents=total,
                max_abs_diff=stats["max_abs_diff"],
                mean_abs_diff=stats["mean_abs_diff"],
                worst_sample_in_block=stats["worst_sample"],
                exact=stats["exact"],
            )
        )
    return out


def probe_shift_equivariance(
    decode_fn: Callable[[Any], Any],
    latents: Any,
    latent_start: int,
    latent_stop: int,
    shifts: Sequence[int] = (1, 2, 3, 4, 5),
    geometry: Optional[AudioLatentGeometry] = None,
    inset_latents: int = 8,
) -> List[dict]:
    """Does moving the slice's left edge change the *interior* of the output?

    A stack of convolutions and transposed convolutions is translation
    covariant: shifting the input window by ``k`` latents shifts the output by
    exactly ``k * samples_per_latent``, so the shared interior must agree once
    the boundary-affected margins are inset.  If it does not, the decoder has a
    genuine phase dependence (a strided resampler whose decimation grid is
    anchored to the tensor's origin) and no amount of margin can fix it - the
    slice start would have to be aligned to that period instead.

    The comparison is **length controlled**: the window is slid, not grown, so
    both decodes see a tensor of exactly the same shape.  Without that, a
    decoder whose kernels are selected per input *length* would show up here as
    a phase dependence, which is a different problem with a different fix.
    """
    geom = geometry or AudioLatentGeometry()
    spl = geom.samples_per_latent
    length = latent_stop - latent_start
    base = decode_fn(latents[..., latent_start:latent_stop])

    results: List[dict] = []
    for shift in shifts:
        lo = latent_start - shift
        if lo < 0:
            continue
        # same length, slid left by `shift` latents
        shifted = decode_fn(latents[..., lo:latent_stop - shift])
        common = length - shift  # latents covered by both windows, from `latent_start`
        a_lo = inset_latents * spl
        a_hi = (common - inset_latents) * spl
        if a_hi <= a_lo:
            continue
        got_base = base[..., a_lo:a_hi]
        got_shift = shifted[..., a_lo + shift * spl:a_hi + shift * spl]
        stats = diff_stats(got_base, got_shift)
        results.append({
            "shift_latents": shift,
            "latent_start": lo,
            "window_latents": length,
            "parity": lo % 2,
            "compared_samples": stats["samples"],
            "max_abs_diff": stats["max_abs_diff"],
            "mean_abs_diff": stats["mean_abs_diff"],
            "exact": stats["exact"],
        })
    return results


@dataclass
class MarginSearchResult:
    """Outcome of a latent-margin search."""

    margin: Optional[int]
    tolerance: float
    errors: List[tuple]  # [(margin, max_abs_diff), ...] in probe order
    block_latents: int
    samples_per_latent: int
    sample_rate: int
    max_margin: int
    total_latents: Optional[int] = None
    phase_align: int = 1
    prefix_mode: bool = False

    @property
    def found(self) -> bool:
        return self.margin is not None

    @property
    def full_sequence_error(self) -> Optional[float]:
        """Error at the control rung where every block decodes everything."""
        if self.total_latents is None:
            return None
        for margin, err in self.errors:
            if margin >= self.total_latents:
                return err
        return None

    @property
    def full_sequence_is_exact(self) -> Optional[bool]:
        err = self.full_sequence_error
        return None if err is None else err == 0.0

    @property
    def lookahead_samples(self) -> Optional[int]:
        if self.margin is None:
            return None
        return self.margin * self.samples_per_latent

    @property
    def lookahead_seconds(self) -> Optional[Fraction]:
        if self.margin is None:
            return None
        return Fraction(self.margin * self.samples_per_latent, self.sample_rate)

    def describe(self) -> str:
        lines = ["margin search (block={} latents, tol={:g}):".format(self.block_latents, self.tolerance)]
        for margin, err in self.errors:
            mark = "OK " if err <= self.tolerance else "   "
            lines.append("  {} margin={:<4d} max|diff|={:.3e}".format(mark, margin, err))
        fs_err = self.full_sequence_error
        if fs_err is not None:
            lines.append(
                "  control: margin >= total ({}) decodes the whole sequence per block "
                "-> max|diff|={:.3e} ({})".format(
                    self.total_latents, fs_err,
                    "exact, as it must be" if fs_err == 0.0
                    else "NOT EXACT - see below")
            )
        if self.found:
            lines.append(
                "  -> minimum margin {} latents = {} samples = {:.4f} s of lookahead".format(
                    self.margin, self.lookahead_samples, float(self.lookahead_seconds)
                )
            )
        else:
            searched = [e for m, e in self.errors if m <= self.max_margin]
            best = min(searched) if searched else float("nan")
            lines.append(
                "  -> NO margin <= {} met tol {:g} (best max|diff|={:.3e}); "
                "one chunk of lookahead is NOT enough at this tolerance".format(
                    self.max_margin, self.tolerance, best)
            )
            if not self.is_monotone:
                lines.append(
                    "     the error is NON-MONOTONE in the margin, which a receptive-field "
                    "effect cannot be: suspect run-to-run nondeterminism (cuDNN autotune "
                    "selecting different algorithms per input length)."
                )
            if fs_err is not None and fs_err > 0.0:
                lines.append(
                    "     and the full-sequence control is ALSO non-zero, so this is NOT "
                    "an overlap-save/receptive-field result: the decode is "
                    "nondeterministic or the harness is wrong."
                )
        return "\n".join(lines)

    @property
    def is_monotone(self) -> bool:
        """Does the error fall (weakly) as the margin grows?

        A real receptive-field effect is monotone.  Non-monotone errors point at
        run-to-run nondeterminism (cuDNN autotune picking different algorithms
        per input length) rather than at missing context.
        """
        ordered = sorted(self.errors, key=lambda item: item[0])
        tol = 1e-12
        return all(b <= a + tol for (_, a), (_, b) in zip(ordered, ordered[1:]))

    @property
    def searched_errors(self) -> List[tuple]:
        """Ladder rungs that were actually eligible for the search."""
        return [(m, e) for m, e in self.errors if m <= self.max_margin]


def search_latent_margin(
    decode_fn: Callable[[Any], Any],
    latents: Any,
    reference: Any,
    tolerance: float = 1e-5,
    block_latents: int = 5,
    max_margin: int = 64,
    margins: Optional[Sequence[int]] = None,
    geometry: Optional[AudioLatentGeometry] = None,
    concat: Optional[Callable[[Sequence[Any], int], Any]] = None,
    on_step: Optional[Callable[[int, float], None]] = None,
    phase_align: int = 1,
    prefix_mode: bool = False,
    include_full_sequence: bool = True,
) -> MarginSearchResult:
    """Find the smallest overlap-save margin meeting ``tolerance``.

    ``reference`` is the full-sequence decode of ``latents``.  Margins are tried
    from small to large (the growth is what makes the answer a *measured*
    latency rather than a guess).

    ``include_full_sequence`` appends ``total_latents`` to the ladder so the
    search always ends with the control case where every block decodes the whole
    sequence.  That rung must come out exact; if it does not, the harness or the
    decoder is nondeterministic and no margin would ever have converged.
    """
    geom = geometry or AudioLatentGeometry()
    total = int(latents.shape[-1])
    candidates = list(margins) if margins is not None else _default_margin_ladder(max_margin)

    def _run(margin: int) -> Any:
        return decode_overlap_save(
            decode_fn, latents, margin, block_latents, geom, concat=concat,
            phase_align=phase_align, prefix_mode=prefix_mode,
        )

    errors: List[tuple] = []
    best: Optional[int] = None
    for margin in candidates:
        if margin > max_margin:
            break
        streamed = _run(margin)
        err = max_abs_diff(streamed, reference)
        errors.append((margin, err))
        if on_step is not None:
            on_step(margin, err)
        if err <= tolerance:
            best = margin
            break

    if best is not None and best > 0:
        # tighten: the ladder may have skipped values below `best`
        lo = max(0, _previous_candidate(candidates, best) + 1)
        hi = best
        while lo < hi:
            mid = (lo + hi) // 2
            streamed = _run(mid)
            err = max_abs_diff(streamed, reference)
            errors.append((mid, err))
            if on_step is not None:
                on_step(mid, err)
            if err <= tolerance:
                hi = mid
            else:
                lo = mid + 1
        best = hi

    # Control rung: margin >= total makes every block decode the whole sequence
    # and slice it, so it is the full decode by construction.  It is recorded as
    # evidence only - it can never satisfy the search, which is bounded by
    # max_margin.
    if include_full_sequence and not any(m >= total for m, _ in errors):
        err = max_abs_diff(_run(total), reference)
        errors.append((total, err))
        if on_step is not None:
            on_step(total, err)

    return MarginSearchResult(
        margin=best,
        tolerance=tolerance,
        errors=errors,
        block_latents=block_latents,
        samples_per_latent=geom.samples_per_latent,
        sample_rate=geom.sample_rate,
        max_margin=max_margin,
        total_latents=total,
        phase_align=phase_align,
        prefix_mode=prefix_mode,
    )


def _default_margin_ladder(max_margin: int) -> List[int]:
    ladder = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    out = [m for m in ladder if m <= max_margin]
    if not out or out[-1] != max_margin:
        out.append(max_margin)
    return out


def _previous_candidate(candidates: Sequence[int], value: int) -> int:
    prev = -1
    for c in candidates:
        if c < value:
            prev = max(prev, c)
    return prev


def decode_overlap_save(
    decode_fn: Callable[[Any], Any],
    latents: Any,
    margin: int,
    block_latents: int,
    geometry: Optional[AudioLatentGeometry] = None,
    concat: Optional[Callable[[Sequence[Any], int], Any]] = None,
    phase_align: int = 1,
    prefix_mode: bool = False,
    left_margin: Optional[int] = None,
) -> Any:
    """Decode a whole latent sequence via overlap-save and concatenate.

    With ``margin >= total_latents`` every block decodes the entire sequence and
    then slices it, so the result is the full decode **by construction**.  That
    makes it the decisive control: if it is not exact, the difference cannot be
    an overlap-save artefact.
    """
    geom = geometry or AudioLatentGeometry()
    cat = concat or _default_concat
    total = int(latents.shape[-1])
    planner = OverlapSavePlanner(
        margin, block_latents, geom, left_margin=left_margin,
        phase_align=phase_align, prefix_mode=prefix_mode,
    )
    parts: List[Any] = []
    for req in planner.plan(total):
        decoded = decode_fn(latents[..., req.latent_start:req.latent_stop])
        parts.append(decoded[..., req.take_start:req.take_stop])
    if not parts:
        return latents[..., :0]
    return cat(parts, -1)
