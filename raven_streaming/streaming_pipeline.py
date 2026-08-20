"""Streaming video and audio collectors + optional fMP4 preview, off ``on_chunk``.

Two collectors and a preview
----------------------------
One :class:`StreamingPipeline` lives for exactly one node execution and runs
**three lanes** off the same chunk callback, two of which are the node's actual
outputs:

* the **video collector**, always on. It decodes each chunk's video latents in
  the official order and copies the finalized frames into a pre-allocated
  ``IMAGE`` buffer. :meth:`StreamingPipeline.finalize_image` hands that buffer
  over as the node's IMAGE.
* the **audio collector**, always on. It decodes each chunk's audio latents
  overlap-save and copies the samples into a pre-allocated raw waveform.
  :meth:`StreamingPipeline.finalize_audio` normalises it exactly the way
  ``comfy_extras.nodes_audio.vae_decode_audio`` does and hands over the AUDIO.
* the **preview lane**, optional. It reads frames and PCM back out of the two
  collectors, muxes fMP4 fragments and pushes them at a
  :class:`~raven_streaming.preview.PreviewMediaSink`.

A failure in either collector **propagates** -- it is a failure of the run.
Every failure in the preview -- no sink, no PyAV, a dead websocket, an
oversized fragment -- **disables the preview and nothing else**.

Why this shape (and not a full decode at the end)
-------------------------------------------------
The node used to finish by decoding the whole clip twice over: once here for
the preview, and once again through ``video_vae.decode`` / ``vae_decode_audio``
for the outputs. Both of those died on real hardware.

* 39 frames, 141 GiB card: ``video_vae.decode`` OOMed at 130.22 GiB allocated /
  139.12 GiB reserved.
* 192 frames, 24 GiB card: with the DiT *and* the video VAE already unloaded
  and only the audio VAE resident, the whole-clip ``vae_decode_audio`` still
  OOMed -- and its OOM fallback made it worse, because the generic tiled path
  is written for 4-D latents and the H3 audio latent is ``[B, 32, 2, T]``, so
  the retry died in an ``IndexError`` instead of a memory error.

Both calls only re-derive data these lanes already produced chunk by chunk. So
they are gone: the peak becomes one 7-latent video chunk and one overlap-save
audio block (both priced by upstream's own ``memory_used_decode``), plus two
host buffers that are exactly the outputs -- 2.43 GB of IMAGE at 192 frames
(4.59 GB at 362) and 2.5 MB of waveform, in host RAM rather than VRAM.

That is also why the collectors cannot be optional or best-effort: with no full
decode to fall back on, dropping a chunk would mean silently returning a short
or partly-black IMAGE, or a clip with a hole in the audio.

Why the preview lane waits for both clocks
------------------------------------------
The video VAE finalizes 17 frames per 5 latents with a 2-latent lookahead; the
audio VAE is not causal at all and needs :data:`AUDIO_MARGIN_LATENTS` latents of
right context per block. The two therefore become available at different times,
and a muxer fed video that is ahead of the audio would either stall or need
silence inserted. Inserting silence is *not* an option: it would be permanent
(fMP4 fragments are never revised), so the preview would carry a gap the final
AUDIO output does not have. Instead the coordinator holds frames until real PCM
covers the same media-clock range -- see :meth:`_drain`. None of that gating
touches the collector, which is always a whole chunk ahead of the preview.

Import weight: standard library plus :mod:`raven_streaming.media` (pure integer
planning) and :mod:`raven_streaming.preview_session` (standard library only).
torch, numpy, PyAV and ComfyUI are imported lazily, inside the calls that need
them -- the collectors do need torch, because their buffers *are* the node's
tensors, but nothing is imported at module scope.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from raven_streaming.media.audio_stream import (
    AudioLatentGeometry,
    OverlapSaveAudioDecoder,
)
from raven_streaming.media.clock import (
    AUDIO_SAMPLES_PER_LATENT,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_VIDEO_FPS,
    VIDEO_FRAMES_PER_CHUNK,
    MediaClock,
)
from raven_streaming.media.video_stream import (
    IncrementalVideoDecoder,
    VideoChunkParams,
    minimax_decoder_adapter,
)
from raven_streaming.preview_session import MAX_RAW_PAYLOAD_BYTES

__all__ = [
    "AUDIO_MARGIN_LATENTS",
    "AUDIO_BLOCK_LATENTS",
    "AUDIO_CHANNELS",
    "PREVIEW_MIME",
    "PREVIEW_FRAGMENT_MODE",
    "PREVIEW_IDR_INTERVAL_FRAMES",
    "PipelineError",
    "PipelineConfig",
    "PipelineReport",
    "StreamingPipeline",
    "detach_to_cpu",
    "frames_to_arrays",
    "pcm_to_array",
    "build_video_decoder",
    "build_audio_decoder",
    "build_muxer_config",
    "build_muxer",
    "build_media_pipeline",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# published constants
# --------------------------------------------------------------------------

#: Overlap-save right/left context for the audio VAE, in latents.
#:
#: Measured at M0 against the real ``MiniMaxH3AudioVAE`` (BigVGAN decoder), not
#: guessed: ``tools/probe_audio_overlap.py`` grows the margin until the streamed
#: waveform matches a full-sequence decode. At margin 17 with 28-latent blocks
#: the measured ``max|diff|`` against the full decode is below ``2.5e-6`` -- the
#: same order as the decoder's own run-to-run noise, i.e. the boundary error is
#: gone. 17 latents is ``17 * 800 = 13600`` samples = **0.425 s of lookahead**,
#: which is why the preview's audio trails its video by roughly one chunk.
AUDIO_MARGIN_LATENTS = 17

#: Latents decoded per overlap-save block. 28 is the audio latent count of a
#: steady-state H3 chunk (the 29 / 28 / 28 cadence the shared 85/3 clock
#: produces), so a block boundary lands on a chunk boundary instead of cutting
#: across one, and the M0 measurement was taken at exactly this block size.
#:
#: It is also the preview's audio *granularity*, and that is a trade-off worth
#: stating. A block is emitted once ``block + margin`` further latents exist,
#: so the last ``block - 1 + margin`` latents of a clip (44 here, 1.1 s) can
#: only be decoded by the edge blocks at :meth:`StreamingPipeline.finish`, and
#: the frames they cover wait with them (the lane never substitutes silence).
#: Measured cost of that, per clip length: 22 frames streams 0 % before finish,
#: 39 frames 41 %, 90 frames 74 %, 192 frames 87.5 %, 362 frames 93 %.
#:
#: Correctness depends on ``margin`` -- the receptive field -- not on the block
#: size, so a smaller block would stream sooner at the cost of decoding more
#: overlap: 28 decodes 62 latents per 28 emitted (2.2x), 7 would decode 41 per
#: 7 (5.9x). Changing it is a measurement, not an edit; the number here is the
#: one M0 actually ran.
AUDIO_BLOCK_LATENTS = 28

#: The H3 audio VAE is stereo. Not configurable: the latent carries 2 rows.
AUDIO_CHANNELS = 2

#: MSE type sent in ``open``. H.264 High@4.0 + AAC-LC, which is what
#: :mod:`raven_streaming.media.codecs`' preference chains select.
PREVIEW_MIME = 'video/mp4; codecs="avc1.640028,mp4a.40.2"'

#: ``frag_every_frame``: a fragment per frame, so the first picture reaches the
#: browser one frame after it is muxed instead of one segment. The fragments are
#: only decodable **in order** (see ``media/mp4_writer.py``), which is exactly
#: what an MSE ``SourceBuffer`` fed sequentially needs, and is why ``segment``
#: messages from this lane never claim ``keyframe``.
PREVIEW_FRAGMENT_MODE = "every_frame"

#: One forced IDR per video chunk (17 frames), so a client that joins late can
#: still be resynchronised by a fresh stream at chunk granularity.
PREVIEW_IDR_INTERVAL_FRAMES = VIDEO_FRAMES_PER_CHUNK

#: How many frames the muxer itself may still be holding when a callback
#: returns. This is PyAV's, not ours: in ``frag_every_frame`` mode the measured
#: steady-state delay is 1 frame video-only and a little more with an audio
#: track interleaved (``media/mp4_writer.py``). Anything above this means the
#: pipeline is sitting on data instead of pushing it, which is the failure this
#: number exists to make visible.
PREVIEW_MAX_MUX_DELAY_FRAMES = 3


class PipelineError(RuntimeError):
    """The streaming pipeline was wired up wrong."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the media lane needs to know about one request."""

    frames: int
    width: int
    height: int
    fps: Fraction = DEFAULT_VIDEO_FPS
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    channels: int = AUDIO_CHANNELS
    audio_margin_latents: int = AUDIO_MARGIN_LATENTS
    audio_block_latents: int = AUDIO_BLOCK_LATENTS
    samples_per_latent: int = AUDIO_SAMPLES_PER_LATENT
    idr_interval_frames: int = PREVIEW_IDR_INTERVAL_FRAMES
    fragment_mode: str = PREVIEW_FRAGMENT_MODE
    mime: str = PREVIEW_MIME
    video_bitrate: int = 6_000_000
    audio_bitrate: int = 128_000

    def __post_init__(self) -> None:
        for name in ("frames", "width", "height"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise PipelineError(f"{name} must be a positive integer, got {value!r}")
        if self.width % 2 or self.height % 2:
            raise PipelineError(
                f"H.264 yuv420p needs even dimensions, got {self.width}x{self.height}"
            )
        if self.audio_margin_latents < 0 or self.audio_block_latents < 1:
            raise PipelineError("audio margin must be >= 0 and block size >= 1")
        object.__setattr__(self, "fps", Fraction(self.fps))

    @property
    def clock(self) -> MediaClock:
        return MediaClock(self.fps, self.sample_rate)

    @property
    def duration_seconds(self) -> float:
        return float(Fraction(int(self.frames), 1) / Fraction(self.fps))

    @property
    def audio_decode_latents(self) -> int:
        """Latents one overlap-save block hands the decoder at its widest."""
        return int(self.audio_block_latents + 2 * self.audio_margin_latents)

    @property
    def audio_latents(self) -> int:
        """Audio latents this clip carries (``round(frames / 24 * 40)``).

        Taken from :func:`raven_streaming.layout.audio_latent_t` rather than
        re-derived, so the collector's expected length cannot drift from the
        grid the sampler wrote.
        """
        from raven_streaming.layout import audio_latent_t

        return int(audio_latent_t(int(self.frames)))

    @property
    def audio_samples(self) -> int:
        return self.audio_latents * int(self.samples_per_latent)

    def describe(self) -> Dict[str, Any]:
        return {
            "frames": int(self.frames),
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "audio_margin_latents": int(self.audio_margin_latents),
            "audio_block_latents": int(self.audio_block_latents),
            "samples_per_latent": int(self.samples_per_latent),
            "fragment_mode": self.fragment_mode,
            "idr_interval_frames": int(self.idr_interval_frames),
            "mime": self.mime,
        }


@dataclass(frozen=True)
class ChunkEmission:
    """What one ``on_chunk`` call decoded and, crucially, *sent*.

    One of these is recorded per chunk, whether or not a preview is running, so
    that "was it streamed as it was produced, or did it all arrive at the end?"
    is answered by the run's own record rather than by reading the code.
    """

    chunk: int
    frames: int          # frames the collector wrote this chunk
    samples: int         # samples the collector wrote this chunk
    muxed_frames: int    # frames handed to the muxer this chunk
    muxed_samples: int
    fragments: int       # fragments sent to the sink this chunk
    fragment_bytes: int
    held_frames: int     # decoded frames still waiting for their audio
    seconds: float       # since the pipeline was built

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk,
            "frames": self.frames,
            "samples": self.samples,
            "muxed_frames": self.muxed_frames,
            "muxed_samples": self.muxed_samples,
            "fragments": self.fragments,
            "fragment_bytes": self.fragment_bytes,
            "held_frames": self.held_frames,
            "seconds": round(self.seconds, 4),
        }


@dataclass
class PipelineReport:
    """What one execution's two lanes actually did. For the node's log."""

    chunks: int = 0
    frames_decoded: int = 0
    samples_decoded: int = 0
    #: collector: frames written into the IMAGE buffer, and what it cost
    collected_frames: int = 0
    expected_frames: int = 0
    image_bytes: int = 0
    image_shape: Tuple[int, ...] = ()
    image_device: str = ""
    image_dtype: str = ""
    #: collector: samples written into the raw waveform, and what it cost
    collected_samples: int = 0
    expected_samples: int = 0
    audio_bytes: int = 0
    audio_shape: Tuple[int, ...] = ()
    audio_device: str = ""
    audio_dtype: str = ""
    frames_muxed: int = 0
    samples_muxed: int = 0
    init_bytes: int = 0
    fragment_sizes: List[int] = field(default_factory=list)
    first_fragment_chunk: Optional[int] = None
    first_fragment_latency: Optional[float] = None
    oversize_fragments: int = 0
    send_failures: int = 0
    errors: int = 0
    preview_disabled: bool = False
    disabled_reason: str = ""
    #: one entry per chunk: what was decoded and what went out, in order
    chunk_emissions: List[ChunkEmission] = field(default_factory=list)
    #: fragments that had been sent before ``finish()`` was called
    fragments_before_finish: int = 0
    #: the node's itemised rollout reserve, carried so one report explains both
    #: what was streamed and what the run was allowed to allocate
    memory_budget: Dict[str, Any] = field(default_factory=dict)
    #: device/dtype order the preview's latents actually took
    decode_policy: Dict[str, Any] = field(default_factory=dict)

    @property
    def fragments(self) -> int:
        return len(self.fragment_sizes)

    @property
    def fragment_bytes(self) -> int:
        return int(sum(self.fragment_sizes))

    @property
    def largest_fragment(self) -> int:
        return max(self.fragment_sizes) if self.fragment_sizes else 0

    @property
    def image_complete(self) -> bool:
        return bool(self.expected_frames) and self.collected_frames == self.expected_frames

    @property
    def audio_complete(self) -> bool:
        return bool(self.expected_samples) and self.collected_samples == self.expected_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": self.chunks,
            "frames_decoded": self.frames_decoded,
            "samples_decoded": self.samples_decoded,
            "collected_frames": self.collected_frames,
            "expected_frames": self.expected_frames,
            "image_bytes": self.image_bytes,
            "image_shape": list(self.image_shape),
            "image_device": self.image_device,
            "image_dtype": self.image_dtype,
            "image_complete": self.image_complete,
            "collected_samples": self.collected_samples,
            "expected_samples": self.expected_samples,
            "audio_bytes": self.audio_bytes,
            "audio_shape": list(self.audio_shape),
            "audio_device": self.audio_device,
            "audio_dtype": self.audio_dtype,
            "audio_complete": self.audio_complete,
            "frames_muxed": self.frames_muxed,
            "samples_muxed": self.samples_muxed,
            "init_bytes": self.init_bytes,
            "fragments": self.fragments,
            "fragment_bytes": self.fragment_bytes,
            "largest_fragment": self.largest_fragment,
            "fragment_sizes": list(self.fragment_sizes),
            "first_fragment_chunk": self.first_fragment_chunk,
            "first_fragment_latency": self.first_fragment_latency,
            "oversize_fragments": self.oversize_fragments,
            "send_failures": self.send_failures,
            "errors": self.errors,
            "fragments_before_finish": self.fragments_before_finish,
            "chunk_emissions": [e.to_dict() for e in self.chunk_emissions],
            "preview_disabled": self.preview_disabled,
            "disabled_reason": self.disabled_reason,
            "memory_budget": dict(self.memory_budget),
            "decode_policy": dict(self.decode_policy),
        }

    def describe(self) -> str:
        collector = (
            "raven collectors: {frames}/{expected_frames} frame(s) into {shape} {dtype} "
            "on {device} ({mib:.0f} MiB host), {samples}/{expected_samples} sample(s) "
            "({audio_mib:.1f} MiB)".format(
                frames=self.collected_frames,
                expected_frames=self.expected_frames,
                shape=tuple(self.image_shape) or "-",
                dtype=self.image_dtype or "-",
                device=self.image_device or "-",
                mib=self.image_bytes / (1024.0 ** 2),
                samples=self.collected_samples,
                expected_samples=self.expected_samples,
                audio_mib=self.audio_bytes / (1024.0 ** 2),
            )
        )
        if self.preview_disabled and not self.fragment_sizes:
            preview = "preview: not streamed ({})".format(
                self.disabled_reason or "disabled"
            )
        else:
            latency = (
                "n/a"
                if self.first_fragment_latency is None
                else "{:.2f}s".format(self.first_fragment_latency)
            )
            preview = (
                "preview: {chunks} chunk(s), {frames} frame(s), {samples} sample(s), "
                "{fragments} fragment(s) / {bytes_} B (largest {largest} B), first "
                "fragment after chunk {chunk} at {latency}{tail}".format(
                    chunks=self.chunks,
                    frames=self.frames_muxed,
                    samples=self.samples_muxed,
                    fragments=self.fragments,
                    bytes_=self.fragment_bytes,
                    largest=self.largest_fragment,
                    chunk=self.first_fragment_chunk,
                    latency=latency,
                    tail=(
                        ""
                        if not self.preview_disabled
                        else "; preview stopped early ({})".format(self.disabled_reason)
                    ),
                )
            )
        return collector + "; " + preview

    def describe_emissions(self) -> str:
        """The per-chunk table a real run should be accepted on.

        What it answers: did each chunk push what it produced, or did the
        stream arrive in one lump at the end? ``finish`` is the last row and
        should be the smallest -- the 5-frame video tail and the encoder's own
        tails, nothing more.
        """
        header = (
            "raven emission log (chunk: frames/samples decoded -> muxed, "
            "fragments sent, held for audio, at t)"
        )
        rows = [
            "  chunk {chunk:>3}: {frames:>4}f/{samples:>7}s -> {muxed_frames:>4}f/"
            "{muxed_samples:>7}s, {fragments:>4} frag / {bytes_:>8} B, held {held:>3}"
            ", t={seconds:.2f}s".format(
                chunk=e.chunk,
                frames=e.frames,
                samples=e.samples,
                muxed_frames=e.muxed_frames,
                muxed_samples=e.muxed_samples,
                fragments=e.fragments,
                bytes_=e.fragment_bytes,
                held=e.held_frames,
                seconds=e.seconds,
            )
            for e in self.chunk_emissions
        ]
        tail = "  {} of {} fragment(s) were sent before finish()".format(
            self.fragments_before_finish, self.fragments
        )
        return "\n".join([header] + rows + [tail])


# --------------------------------------------------------------------------
# tensor -> CPU array helpers (duck-typed: torch tensors or numpy arrays)
# --------------------------------------------------------------------------


def detach_to_cpu(tensor: Any) -> Any:
    """Return a CPU, autograd-free copy of ``tensor``; pass anything else through.

    The single place this package severs its link to the sampler's device
    memory. A torch tensor is detached, moved to CPU and widened to float32
    (the VAEs are re-fed in their own dtype at decode time); a numpy array --
    what the fakes use -- is already all three of those things.
    """
    detach = getattr(tensor, "detach", None)
    cpu = getattr(tensor, "cpu", None)
    if not (callable(detach) and callable(cpu)):
        return tensor
    out = detach()
    out = out.cpu()
    to_float = getattr(out, "float", None)
    if callable(to_float):
        out = to_float()
    return out


def _to_numpy(value: Any) -> Any:
    import numpy as np  # local: the coordinator itself never needs numpy

    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn) and hasattr(value, "detach"):
        value = detach_to_cpu(value)
        return value.numpy()
    return np.asarray(value)


def frames_to_arrays(frames: Any, channels: int = 3) -> List[Any]:
    """``[B, C, T, H, W]`` pixels in [0, 1] -> a list of ``[H, W, C]`` arrays.

    One array per frame, contiguous and host-resident, which is what
    ``FragmentedMP4Muxer.write_video_frame`` takes. Batch entries beyond the
    first are dropped: the streaming sampler is batch size 1 by contract.

    The pipeline itself no longer goes through here -- the preview reads its
    frames back out of the collector's IMAGE buffer, which is already
    ``[T, H, W, C]``, so that what is shown is what is returned. This stays as
    the conversion for anything holding raw decoder output (diagnostics, and the
    tests that pin the shape contract).
    """
    import numpy as np

    array = _to_numpy(frames)
    if array.ndim != 5:
        raise PipelineError(
            f"decoded frames must be [B, C, T, H, W], got shape {tuple(array.shape)}"
        )
    array = array[0]  # [C, T, H, W]
    if array.shape[0] < channels:
        raise PipelineError(
            f"decoded frames carry {array.shape[0]} channel(s), need {channels}"
        )
    array = array[:channels]
    array = np.transpose(array, (1, 2, 3, 0))  # [T, H, W, C]
    return [np.ascontiguousarray(array[i]) for i in range(array.shape[0])]


def pcm_to_array(wave: Any, channels: int = AUDIO_CHANNELS) -> Any:
    """``[B, S, N]`` (or ``[S, N]``) decoded audio -> a ``[channels, N]`` array."""
    import numpy as np

    array = _to_numpy(wave)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise PipelineError(
            f"decoded audio must be [B, channels, samples] or [channels, samples], "
            f"got shape {tuple(array.shape)}"
        )
    if array.shape[0] == 1 and channels > 1:
        array = np.repeat(array, channels, axis=0)
    if array.shape[0] != channels:
        raise PipelineError(
            f"decoded audio has {array.shape[0]} channel(s), expected {channels}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

_CHUNK_FIELDS = (
    "index",
    "is_last",
    "video_start",
    "video_stop",
    "audio_start",
    "audio_stop",
    "video_x0",
    "audio_x0",
)


def _chunk_output_class() -> Optional[type]:
    """``consistency.ChunkOutput`` when it is importable (it needs torch)."""
    try:
        from raven_streaming.consistency import ChunkOutput
    except Exception:  # noqa: BLE001 - a bare environment is a supported mode
        return None
    return ChunkOutput


def _require_chunk_output(chunk: Any) -> None:
    """Accept the sampler's own ``ChunkOutput`` and nothing else.

    This is the one error the pipeline lets escape into the sampler callback:
    it can only be reached by wiring ``on_chunk`` to something that is not the
    streaming sampler, which is a bug in this package rather than a preview
    failure a user should have to live with.
    """
    expected = _chunk_output_class()
    if expected is not None:
        if isinstance(chunk, expected):
            return
        raise TypeError(
            "on_chunk takes a raven_streaming.consistency.ChunkOutput, got "
            f"{type(chunk).__name__}"
        )
    missing = [name for name in _CHUNK_FIELDS if not hasattr(chunk, name)]
    if missing:
        raise TypeError(
            "on_chunk takes a raven_streaming.consistency.ChunkOutput; "
            f"{type(chunk).__name__} is missing {missing}"
        )


class StreamingPipeline:
    """One execution's video collector plus its optional preview lane.

    Not reusable and not thread safe: the sampler calls :meth:`on_chunk` from
    its own loop, in order, and the node calls :meth:`finish` (or
    :meth:`cancel`) exactly once afterwards, then :meth:`finalize_image`.

    Lifecycle contract, stated once:

    * :meth:`on_chunk` -- collector first, then the preview. Collector errors
      propagate; preview errors disable the preview.
    * :meth:`finish` -- idempotent. Flushes the collector's 5-frame tail,
      checks the frame count, then flushes and closes the preview. A collector
      failure here still propagates; a second call is a no-op that returns the
      same report.
    * :meth:`finalize_image` -- only after :meth:`finish`, only when the frame
      count is exactly right, and never after :meth:`cancel`. Repeat calls hand
      back the same tensor.
    * :meth:`cancel` -- idempotent, and the *only* path that drops the
      collected frames. A cancelled or failed run returns no partial IMAGE.
      It also *aborts* both decoders before letting go of them: no decode, no
      tail padding, no flush, and the device-resident overlap they hold is
      released at a known point rather than whenever the last reference dies.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig,
        video_decoder: Any,
        audio_decoder: Any = None,
        muxer: Any = None,
        sink: Any = None,
        log: Optional[logging.Logger] = None,
        clock_fn: Callable[[], float] = time.monotonic,
        preview_disabled_reason: str = "",
        image_device: Any = None,
        image_dtype: Any = None,
        memory_budget: Optional[Dict[str, Any]] = None,
        decode_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        if video_decoder is None or audio_decoder is None:
            raise PipelineError(
                "a StreamingPipeline needs both decoders: the two collectors are "
                "the node's IMAGE and AUDIO outputs, not optional extras"
            )
        self.config = config
        self.clock = config.clock
        self._video = video_decoder
        self._audio = audio_decoder
        self._muxer = muxer
        self._sink = sink
        self._log = log if log is not None else logger
        self._now = clock_fn
        self._started_at = clock_fn()
        self.memory_budget: Dict[str, Any] = dict(memory_budget or {})
        if decode_policy is None:
            policy = getattr(getattr(video_decoder, "decoder", None), "policy", None)
            decode_policy = policy() if callable(policy) else None
        self.decode_policy: Dict[str, Any] = dict(decode_policy or {})

        # -- collector state ----------------------------------------------
        self._image: Any = None
        self._image_device = image_device
        self._image_dtype = image_dtype
        self._collected = 0
        self._waveform: Any = None
        self._collected_samples = 0
        self._audio_payload: Optional[Dict[str, Any]] = None
        self._released = False

        # -- preview coordination state; host arrays and integers only ----
        self._frames: Deque[Tuple[int, Any]] = deque()
        self._pcm: Deque[Any] = deque()
        self._pcm_available = 0
        self._frames_muxed = 0
        self._samples_muxed = 0

        # -- bookkeeping ---------------------------------------------------
        self.chunk_emissions: List[ChunkEmission] = []
        self.fragments_before_finish = 0
        self.chunks = 0
        self.frames_decoded = 0
        self.samples_decoded = 0
        self.init_bytes = 0
        self.fragment_sizes: List[int] = []
        self.first_fragment_chunk: Optional[int] = None
        self.first_fragment_latency: Optional[float] = None
        self.oversize_fragments = 0
        self.send_failures = 0
        self.errors = 0

        self._current_chunk = -1
        self._finished = False
        self._media_closed = False
        # The preview needs a sink and a muxer on top of the collectors that
        # already exist. Missing either is "no preview", never "no output".
        incomplete = sink is None or muxer is None
        self._preview_disabled = bool(incomplete)
        self._preview_reason = (
            preview_disabled_reason
            if preview_disabled_reason
            else ("no preview sink" if incomplete else "")
        )

    # -- introspection ---------------------------------------------------

    @property
    def preview_disabled(self) -> bool:
        return self._preview_disabled

    @property
    def preview_disabled_reason(self) -> str:
        return self._preview_reason

    @property
    def _preview_active(self) -> bool:
        return not self._preview_disabled and self._muxer is not None and self._sink is not None

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def collected_frames(self) -> int:
        """Frames written into the IMAGE buffer so far."""
        return self._collected

    @property
    def collected_samples(self) -> int:
        """Samples written into the raw waveform so far."""
        return self._collected_samples

    @property
    def frames_muxed(self) -> int:
        return self._frames_muxed

    @property
    def samples_muxed(self) -> int:
        return self._samples_muxed

    @property
    def samples_available(self) -> int:
        """Decoded audio samples the coordinator may still draw on."""
        return self._pcm_available

    @property
    def pending_frames(self) -> int:
        """Decoded frames held back because the audio has not caught up."""
        return len(self._frames)

    @staticmethod
    def _buffer_bytes(buffer: Any) -> int:
        if buffer is None:
            return 0
        try:
            return int(buffer.numel()) * int(buffer.element_size())
        except Exception:  # noqa: BLE001 - numpy fakes and the like
            try:
                return int(buffer.nbytes)
            except Exception:  # noqa: BLE001
                return 0

    def report(self) -> PipelineReport:
        return PipelineReport(
            chunks=self.chunks,
            frames_decoded=self.frames_decoded,
            samples_decoded=self.samples_decoded,
            collected_frames=self._collected,
            expected_frames=int(self.config.frames),
            image_bytes=self._buffer_bytes(self._image),
            image_shape=tuple(self._image.shape) if self._image is not None else (),
            image_device=str(getattr(self._image, "device", "")) if self._image is not None else "",
            image_dtype=str(getattr(self._image, "dtype", "")) if self._image is not None else "",
            collected_samples=self._collected_samples,
            expected_samples=self._expected_samples(),
            audio_bytes=self._buffer_bytes(self._waveform),
            audio_shape=tuple(self._waveform.shape) if self._waveform is not None else (),
            audio_device=(
                str(getattr(self._waveform, "device", "")) if self._waveform is not None else ""
            ),
            audio_dtype=(
                str(getattr(self._waveform, "dtype", "")) if self._waveform is not None else ""
            ),
            frames_muxed=self._frames_muxed,
            samples_muxed=self._samples_muxed,
            init_bytes=self.init_bytes,
            fragment_sizes=list(self.fragment_sizes),
            first_fragment_chunk=self.first_fragment_chunk,
            first_fragment_latency=self.first_fragment_latency,
            oversize_fragments=self.oversize_fragments,
            send_failures=self.send_failures,
            errors=self.errors,
            chunk_emissions=list(self.chunk_emissions),
            fragments_before_finish=self.fragments_before_finish,
            preview_disabled=self._preview_disabled,
            disabled_reason=self._preview_reason,
            memory_budget=dict(self.memory_budget),
            decode_policy=dict(self.decode_policy),
        )

    # -- preview control -------------------------------------------------

    def open_preview(self) -> bool:
        """Send the session's ``open``. Returns False when there is no preview."""
        if self._sink is None:
            return False
        return bool(
            self._sink.on_open(
                self.config.mime,
                width=int(self.config.width),
                height=int(self.config.height),
                fps=float(self.config.fps),
                audio={
                    "sample_rate": int(self.config.sample_rate),
                    "channels": int(self.config.channels),
                },
                duration_hint=self.config.duration_seconds,
            )
        )

    def status(self, phase: str, **kwargs: Any) -> bool:
        """Forward a backend phase to the sink. Never raises."""
        if self._sink is None:
            return False
        return bool(self._sink.on_status(phase, **kwargs))

    # -- the sampler callback --------------------------------------------

    def on_chunk(self, chunk: Any) -> None:
        """``consistency.sample_streaming(on_chunk=...)``.

        Collectors first, preview second, **all of it before this call
        returns**. Whatever the two decoders finalized for this chunk is muxed
        and pushed at the sink here, in this callback: nothing decidable is
        parked for :meth:`finish`, which exists only to flush the video tail
        and the encoders' own tails. The collectors' copies are memcpys into
        buffers that already exist, so they cost the send nothing.

        What *is* held back is the frames whose audio has not been decoded yet
        (see :meth:`_drain`) -- and only until the block that covers them
        arrives, which is the next chunk. That number is recorded per chunk as
        ``held_frames`` so the delay is visible rather than assumed.

        The collectors' exceptions travel back into the sampler on purpose --
        their frames and samples are the IMAGE and AUDIO outputs, and a run
        that cannot produce them must fail rather than return a short clip.
        The preview's are caught here and nowhere else.

        Consumes no randomness (it draws none) and never mutates the chunk.
        """
        _require_chunk_output(chunk)
        self.chunks += 1
        self._current_chunk = int(chunk.index)
        if self._finished:
            return
        before = self._emission_marks()

        # Copy off the compute device first, then let go: from here on this
        # object holds host data only, so it cannot keep a GPU allocation (or
        # an autograd graph) alive past the callback.
        video_latent = detach_to_cpu(chunk.video_x0)
        batches = self._video.push(video_latent)
        del video_latent
        frames_written = self._collect(batches)
        del batches

        audio_latent = detach_to_cpu(chunk.audio_x0)
        blocks = self._audio.push(audio_latent)
        del audio_latent
        samples_written = self._collect_audio(blocks)
        del blocks

        if not self._preview_active:
            self._record_emission(before)
            return
        try:
            self._queue_preview_frames(frames_written)
            self._queue_preview_pcm(samples_written)
            # Mux and push inside the callback: the point of the lane is that a
            # frame reaches the browser while the next chunk is being sampled,
            # not when the clip is over.
            self._drain(final=False)
            self._pump()
        except Exception as exc:  # noqa: BLE001 - a preview failure is not a run failure
            self._disable_preview("streaming preview failed", exc)
        self._record_emission(before)

    # -- lifecycle -------------------------------------------------------

    def finish(self) -> PipelineReport:
        """Flush both lanes. Idempotent; collector failures still propagate."""
        if self._finished:
            return self.report()
        self._finished = True
        self.fragments_before_finish = len(self.fragment_sizes)

        # The collectors' tails are the last 5 frames of the clip and its final
        # audio blocks; without them the outputs are short, so neither flush is
        # guarded.
        batches = self._video.finish()
        frames_written = self._collect(batches)
        del batches
        expected = int(self.config.frames)
        if self._collected != expected:
            raise PipelineError(
                f"the streaming decode produced {self._collected} frame(s) for a "
                f"{expected}-frame request. The IMAGE output is decoded once, "
                "chunk by chunk, so a miscount here would be returned as a short "
                "or partly-empty clip rather than caught later."
            )

        blocks = self._audio.finish()
        samples_written = self._collect_audio(blocks)
        del blocks
        expected_samples = self._expected_samples()
        if self._collected_samples != expected_samples:
            raise PipelineError(
                f"the streaming decode produced {self._collected_samples} audio "
                f"sample(s) for a clip of {expected_samples}. The AUDIO output is "
                "decoded once, block by block, so a miscount here would be returned "
                "as a clip with a hole in it."
            )

        if self._preview_active:
            try:
                self._queue_preview_frames(frames_written)
                self._queue_preview_pcm(samples_written)
                self._drain(final=True)
                self._close_media()
                self._pump()
            except Exception as exc:  # noqa: BLE001
                self._disable_preview("streaming preview flush failed", exc)
        self._close_media()
        self._release_preview()
        return self.report()

    def finalize_image(self) -> Any:
        """The node's ``IMAGE``: ``[frames, H, W, 3]`` float32. Only after finish.

        Hands over the very buffer the collector has been writing into, so there
        is no second full-size allocation anywhere in this lane. Repeat calls
        return the same tensor.
        """
        if not self._finished:
            raise PipelineError(
                "finalize_image() before finish(): the last 5 frames are only "
                "decoded by the tail flush"
            )
        if self._image is None:
            raise PipelineError(
                "the collected frames were released: a cancelled or failed run "
                "returns no partial IMAGE"
                + (" ({})".format(self._preview_reason) if self._released else "")
            )
        expected = int(self.config.frames)
        if self._collected != expected:
            raise PipelineError(
                f"only {self._collected} of {expected} frame(s) were collected"
            )
        return self._image

    def finalize_audio(self, vae: Any = None, *, sample_rate: Any = None) -> Dict[str, Any]:
        """The node's ``AUDIO``: the collected waveform, normalised officially.

        The tail of ``comfy_extras.nodes_audio.vae_decode_audio``, reproduced
        expression for expression on the raw waveform this lane collected::

            std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            audio /= std

        It is a **whole-clip** statistic, which is exactly why the preview's
        chunk-wise PCM can never be normalised the same way and why this waits
        for :meth:`finish`. The divide is in place, so the result is cached and
        repeat calls hand back the same dict rather than normalising twice.

        The sample rate follows upstream's resolution order:
        ``vae.audio_sample_rate_output``, then ``vae.audio_sample_rate``, then
        44100 -- and an explicit ``sample_rate`` (what a LATENT carrying one
        would supply) overrides all of it.
        """
        if not self._finished:
            raise PipelineError(
                "finalize_audio() before finish(): the last overlap-save blocks "
                "are only decoded by the tail flush"
            )
        if self._audio_payload is not None:
            return self._audio_payload
        if self._waveform is None:
            raise PipelineError(
                "the collected waveform was released: a cancelled or failed run "
                "returns no partial AUDIO"
            )
        expected = self._expected_samples()
        if self._collected_samples != expected:
            raise PipelineError(
                f"only {self._collected_samples} of {expected} audio sample(s) "
                "were collected"
            )

        import torch

        waveform = self._waveform
        std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        waveform /= std

        rate = sample_rate
        if rate is None:
            rate = getattr(
                vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100)
            )
        self._audio_payload = {"waveform": waveform, "sample_rate": rate}
        return self._audio_payload

    def cancel(self, reason: str = "cancelled") -> PipelineReport:
        """Terminal path for a cancelled or failed run. Idempotent.

        Deliberately does *not* flush: a cancelled run has no more media to
        show, and the fragments still sitting in the muxer describe frames the
        user asked to stop producing. Both lanes' buffers go, the collector's
        included -- there is no such thing as a partial IMAGE output here.

        The two decoders are *aborted* rather than merely dereferenced, and
        before the references go (see :meth:`_abort_decoders`): dropping the
        last reference is not the same as dropping the tensors, and the ones
        the video coordinator holds live on the decode device.
        """
        if self._finished:
            self._close_media()
            self._release_preview()
            self._abort_decoders()
            self._release_collector()
            return self.report()
        self._finished = True
        if not self._preview_disabled and not self._preview_reason:
            self._preview_disabled = True
            self._preview_reason = str(reason)
        self._close_media()
        self._release_preview()
        self._abort_decoders()
        self._release_collector()
        return self.report()

    def _abort_decoders(self) -> None:
        """Tell both decoders to throw their buffers away without decoding.

        Why explicitly, when :meth:`_release_collector` is about to drop both
        references anyway: what those references keep alive is the video
        coordinator's ``dec_overlap`` (5 decoded frames, on the *decode device*)
        and the audio decoder's overlap-save history. Until the last reference
        goes they stay allocated, and on a cancel there are usually several --
        the traceback that caused the cancel, the node holding the decoders it
        built, a test asserting on them. ``abort`` makes the release happen at a
        point this code controls instead of at whatever point the garbage
        collector gets there.

        Feature-probed, not assumed: a decoder without ``abort`` (an older one,
        or a stand-in in a test) is simply skipped, and dereferencing is all it
        gets -- the same behaviour as before this method existed.

        Both are attempted even if the first raises, and a failure is recorded
        and logged rather than propagated. This runs on the path a *failure*
        already took; letting a cleanup error escape here would replace the
        reason for the cancel with a second, less interesting one, and would
        skip :meth:`_release_collector` -- which is what guarantees no partial
        IMAGE or AUDIO survives a cancelled run.
        """
        for lane, decoder in (("video", self._video), ("audio", self._audio)):
            abort = getattr(decoder, "abort", None)
            if not callable(abort):
                continue
            try:
                abort()
            except Exception as exc:  # noqa: BLE001 - cleanup never raises
                self.errors += 1
                self._log.warning(
                    "raven streaming: aborting the %s decoder raised (%s: %s); "
                    "its buffers are dropped with the decoder itself",
                    lane,
                    type(exc).__name__,
                    exc,
                )

    # -- emission accounting ---------------------------------------------

    def _emission_marks(self) -> Tuple[int, int, int, int, int]:
        return (
            self._collected,
            self._collected_samples,
            self._frames_muxed,
            self._samples_muxed,
            len(self.fragment_sizes),
        )

    def _record_emission(self, before: Tuple[int, int, int, int, int]) -> None:
        after = self._emission_marks()
        fragments = self.fragment_sizes[before[4]:]
        self.chunk_emissions.append(
            ChunkEmission(
                chunk=self._current_chunk,
                frames=after[0] - before[0],
                samples=after[1] - before[1],
                muxed_frames=after[2] - before[2],
                muxed_samples=after[3] - before[3],
                fragments=len(fragments),
                fragment_bytes=int(sum(fragments)),
                held_frames=len(self._frames),
                seconds=self._now() - self._started_at,
            )
        )

    # -- the collector ---------------------------------------------------

    def _allocate_image(self, sample: Any) -> None:
        """Pre-allocate the whole IMAGE once, from the first decoded batch.

        Shape and dtype come from the request and from what the VAE actually
        produced, and the two are checked against each other: the canvas is
        derived from the latent grid, so a disagreement means the decode is not
        the decode this request described.
        """
        frames = int(self.config.frames)
        height, width, channels = (int(n) for n in sample.shape[-3:])
        if height != int(self.config.height) or width != int(self.config.width):
            raise PipelineError(
                f"the video VAE produced {height}x{width} frames for a "
                f"{self.config.height}x{self.config.width} request"
            )
        if channels != 3:
            raise PipelineError(
                f"decoded frames carry {channels} channel(s); a Comfy IMAGE is RGB"
            )
        import torch

        # Always a torch tensor: this buffer *is* the node's IMAGE, and a Comfy
        # IMAGE is a float32 torch tensor whatever the decoder handed back.
        device = self._image_device
        if device is None:
            device = getattr(sample, "device", None) or "cpu"
        self._image = torch.empty(
            (frames, height, width, channels),
            dtype=self._image_dtype if self._image_dtype is not None else torch.float32,
            device=device,
        )

    def _collect(self, batches: Any) -> List[Tuple[int, int]]:
        """Copy finalized frames into the IMAGE buffer, in decode order.

        Returns the ``[start, stop)`` frame ranges written, which is what the
        preview lane then reads back out of the buffer -- so what is previewed
        is, frame for frame, what is returned.
        """
        written: List[Tuple[int, int]] = []
        for batch in batches or ():
            # [B, C, T, H, W] -> [T, H, W, C]; a view, not a copy
            frames = batch.frames
            if int(frames.shape[0]) != 1:
                raise PipelineError(
                    f"decoded batch has batch size {frames.shape[0]}; the streaming "
                    "sampler runs batch size 1"
                )
            permute = getattr(frames, "permute", None)
            if callable(permute):
                view = frames[0].permute(1, 2, 3, 0)
            else:  # numpy fakes
                view = frames[0].transpose(1, 2, 3, 0)
            count = int(view.shape[0])
            if count <= 0:
                continue
            if self._image is None:
                self._allocate_image(view)
            start = int(batch.start_frame)
            stop = start + count
            if stop > int(self.config.frames):
                raise PipelineError(
                    f"the streaming decode produced frame {stop - 1} for a "
                    f"{self.config.frames}-frame request"
                )
            import torch

            # one copy, cross-device if the buffer lives elsewhere; the
            # decoder's own tensor is the only other full-size thing alive
            self._image[start:stop].copy_(
                view if hasattr(view, "copy_") else torch.as_tensor(view)
            )
            self._collected += count
            self.frames_decoded += count
            written.append((start, stop))
        return written

    def _release_collector(self) -> None:
        self._image = None
        self._waveform = None
        self._audio_payload = None
        self._video = None
        self._audio = None
        self._released = True

    # -- the audio collector ---------------------------------------------

    def _samples_per_latent(self) -> int:
        """The decoder's own geometry when it exposes one, else the H3 default."""
        geometry = getattr(getattr(self._audio, "planner", None), "geometry", None)
        value = getattr(geometry, "samples_per_latent", None)
        if isinstance(value, int) and value > 0:
            return value
        return int(self.config.samples_per_latent)

    def _expected_samples(self) -> int:
        return int(self.config.audio_latents) * self._samples_per_latent()

    def _allocate_waveform(self, sample: Any) -> None:
        """Pre-allocate the whole raw waveform once, from the first block.

        ``[1, channels, audio_latents * samples_per_latent]`` float32 on the
        host: the shape ``vae_decode_audio`` normalises and every AUDIO
        consumer reads. It is small -- 2.5 MB for a 192-frame clip -- so it is
        allocated whole rather than grown.
        """
        import torch

        channels = int(sample.shape[-2])
        if channels != int(self.config.channels):
            raise PipelineError(
                f"the audio VAE produced {channels} channel(s), expected "
                f"{self.config.channels}"
            )
        self._waveform = torch.empty(
            (1, channels, self._expected_samples()), dtype=torch.float32
        )

    def _collect_audio(self, blocks: Any) -> List[Tuple[int, int]]:
        """Copy decoded samples into the waveform, in overlap-save order."""
        import torch

        written: List[Tuple[int, int]] = []
        for block in blocks or ():
            tensor = block if hasattr(block, "copy_") else torch.as_tensor(block)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                raise PipelineError(
                    "decoded audio must be [1, channels, samples], got "
                    f"{tuple(tensor.shape)}"
                )
            count = int(tensor.shape[-1])
            if count <= 0:
                continue
            if self._waveform is None:
                self._allocate_waveform(tensor)
            start = self._collected_samples
            stop = start + count
            if stop > int(self._waveform.shape[-1]):
                raise PipelineError(
                    f"the streaming decode produced audio sample {stop - 1} for a "
                    f"clip of {self._waveform.shape[-1]}"
                )
            self._waveform[..., start:stop].copy_(tensor)
            self._collected_samples = stop
            self.samples_decoded += count
            written.append((start, stop))
        return written

    # -- the preview lane ------------------------------------------------

    def _queue_preview_frames(self, written: Any) -> None:
        """Queue the frames the collector just wrote, as host arrays."""
        for start, stop in written or ():
            arrays = _to_numpy(self._image[start:stop])
            for offset in range(int(arrays.shape[0])):
                self._frames.append((start + offset, arrays[offset]))

    def _queue_preview_pcm(self, written: Any) -> None:
        """Queue the samples the collector just wrote, as ``[channels, n]``.

        Read back out of the collector's buffer for the same reason the frames
        are: what is previewed is then, sample for sample, what is returned --
        modulo the whole-clip normalisation, which no stream can know in
        advance.
        """
        import numpy as np

        for start, stop in written or ():
            block = np.ascontiguousarray(
                _to_numpy(self._waveform[0, :, start:stop]), dtype=np.float32
            )
            samples = int(block.shape[-1])
            if samples <= 0:
                continue
            self._pcm.append(block)
            self._pcm_available += samples

    def _drain(self, final: bool) -> None:
        """Mux everything both lanes can account for.

        A frame is written only once real PCM covers the media clock through
        the end of that frame. ``final`` lifts the audio gate because the audio
        stream is over: whatever exists is all there will ever be, and the
        alternative (padding with silence to reach the video's length) would
        write a gap into fragments that can never be revised. The tail is at
        most a few samples: ``audio_t`` is ``round(frames / 24 * 40)``, so the
        two lanes agree to within one audio latent.
        """
        while self._frames:
            index, frame = self._frames[0]
            needed = self.clock.samples_for_frames(index + 1)
            if not final and needed > self._pcm_available:
                break
            self._frames.popleft()
            self._muxer.write_video_frame(
                frame,
                force_keyframe=(index % int(self.config.idr_interval_frames) == 0),
            )
            self._frames_muxed += 1
            self._write_audio_upto(min(needed, self._pcm_available))
        if final:
            self._write_audio_upto(self._pcm_available)

    def _write_audio_upto(self, target: int) -> None:
        while self._samples_muxed < target and self._pcm:
            block = self._pcm[0]
            available = int(block.shape[-1])
            room = target - self._samples_muxed
            if available <= room:
                self._pcm.popleft()
                part = block
                taken = available
            else:
                part = block[..., :room]
                self._pcm[0] = block[..., room:]
                taken = room
            self._muxer.write_audio(part)
            self._samples_muxed += taken

    def _pump(self) -> None:
        """Pull finished bytes out of the muxer and hand them to the sink."""
        if self._sink is None or self._muxer is None:
            return
        take_init = getattr(self._muxer, "take_init_segment", None)
        if callable(take_init):
            init = take_init()
            if init:
                self.init_bytes = len(init)
                if not self._sink.on_init(bytes(init)):
                    self.send_failures += 1
        take_fragments = getattr(self._muxer, "take_fragments", None)
        if not callable(take_fragments):
            return
        for segment in take_fragments() or ():
            data = getattr(segment, "data", segment)
            payload = bytes(data)
            self.fragment_sizes.append(len(payload))
            if len(payload) > MAX_RAW_PAYLOAD_BYTES:
                # Protocol v1 has no part field, so the sink refuses it loudly
                # in the log and drops that one fragment. Recorded here so the
                # node's report can say the preview was lossy.
                self.oversize_fragments += 1
            if self.first_fragment_chunk is None:
                self.first_fragment_chunk = self._current_chunk
                self.first_fragment_latency = self._now() - self._started_at
            index = getattr(segment, "index", None)
            ok = self._sink.on_fragment(
                payload,
                index=index if isinstance(index, int) and index >= 0 else None,
                # ``every_frame`` fragments are decodable in order only, so the
                # client is never told one is a seek point.
                keyframe=False,
            )
            if not ok:
                self.send_failures += 1

    def _disable_preview(self, what: str, exc: BaseException) -> None:
        """Stop previewing, say why, and leave the collector completely alone.

        The collector's buffer, its decoder and its frame count are untouched
        here: losing the preview must not cost the output. That is the whole
        point of the split, and ``tests/test_streaming_pipeline.py`` pins it by
        comparing the IMAGE against a run where nothing failed.
        """
        self.errors += 1
        if not self._preview_disabled:
            self._preview_disabled = True
            self._preview_reason = "{}: {}: {}".format(what, type(exc).__name__, exc)
        self._log.warning(
            "raven streaming preview: %s (%s: %s); sampling and the IMAGE output "
            "continue without a preview",
            what,
            type(exc).__name__,
            exc,
        )
        if self._sink is not None:
            # ``sampling`` because the run itself is still going; the message
            # is what tells the user the picture stopped for a reason.
            self._sink.on_status(
                "sampling", message="preview stopped: {}".format(self._preview_reason)
            )
        self._close_media()
        self._release_preview()

    def _close_media(self) -> None:
        if self._media_closed:
            return
        self._media_closed = True
        muxer = self._muxer
        close = getattr(muxer, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - cleanup never raises
                self.errors += 1
                self._log.warning(
                    "raven streaming preview: closing the muxer raised (%s: %s)",
                    type(exc).__name__,
                    exc,
                )

    def _release_preview(self) -> None:
        """Drop the preview lane's buffers. Idempotent; collectors untouched.

        ``_pcm_available`` goes with the deque it describes. It is the audio
        gate's running total -- how much decoded PCM :meth:`_drain` has been
        handed -- so leaving it set after clearing ``_pcm`` would leave the
        pipeline reporting samples it is no longer holding, which is exactly
        what :attr:`samples_available` is read for after a cancel. Zeroing it
        cannot affect the gate: every caller of this method has already ended
        the preview, so ``_drain`` never runs again.

        ``_frames`` needs no such reset -- :attr:`pending_frames` is the
        deque's own length, so clearing it is the whole release.

        Note what is *not* dropped: ``self._audio``. The overlap-save decoder
        used to belong to the preview, and clearing it here was correct then;
        it is a collector now, and clearing it would silently end the AUDIO
        output the moment a preview failed.

        Nor are ``_frames_muxed`` / ``_samples_muxed``: those are the report's
        record of what *was* sent, not a buffer.
        """
        self._frames.clear()
        self._pcm.clear()
        self._pcm_available = 0
        self._muxer = None


# --------------------------------------------------------------------------
# factories (the only place torch / numpy / PyAV / ComfyUI are touched)
# --------------------------------------------------------------------------


class _DeviceBoundVideoDecoder:
    """``minimax_decoder_adapter`` plus the VAE wrapper's device/dtype policy.

    ``comfy.sd.VAE.decode`` cannot be used for the streaming lane: it calls
    ``load_models_gpu([self.patcher], ...)`` itself, which would evict the DiT
    between chunks and undo the single co-resident load the node performs. So
    the inner module is driven directly and the wrapper's device and dtype are
    reproduced here instead: latents in on the VAE's compute device and dtype,
    pixels out on its ``output_device`` in float32 (which is what
    ``_finalize_pixels`` produces anyway, and where the muxer wants them).

    Two things are deliberately *not* reproduced, because for this VAE they are
    nothing: ``process_output`` is the identity for the H3 video VAE (the decode
    already finalizes to [0, 1]), and the OOM tiling fallback -- the incremental
    coordinator already hands over one 7-latent chunk at a time, which is the
    unit upstream's own memory estimate is built on.

    Only *tensors* are moved here. No module is ever ``.to()``-ed, patched or
    unloaded: residency belongs to ``load_models_gpu``.

    What does stay on the decode device between chunks is the coordinator's
    5-frame ``dec_overlap`` -- the cross-fade tail the temporal machine is
    built on. It is bounded, it is the same tensor upstream holds, and the
    pipeline itself never touches it: everything *it* buffers is host memory.
    """

    def __init__(self, vae: Any) -> None:
        self._adapter = minimax_decoder_adapter(vae)
        self._device = getattr(vae, "device", None)
        self._dtype = getattr(vae, "vae_dtype", None)
        self._output_device = getattr(vae, "output_device", None)
        # geometry, so VideoChunkParams.from_vae still reads the real numbers
        self.clip_length = self._adapter.clip_length
        self.vae_ratio_t = self._adapter.vae_ratio_t
        self.token_drop = self._adapter.token_drop
        self.decode_calls = 0
        self.last_input_device: Optional[str] = None
        self.last_input_dtype: Optional[str] = None

    def _adaptive_decode(self, z: Any) -> Any:
        """Cast, then denormalize, then decode -- upstream's order exactly.

        ``VAE.decode`` casts the latents to the VAE's device and dtype and only
        then calls ``MiniMaxH3VideoVAE.decode``, whose first act is
        ``z * latents_std + latents_mean`` in that same dtype
        (``comfy/ldm/minimax/vae.py``). Denormalizing on the host in float32
        first and casting afterwards is a *different* rounding of every latent,
        and the difference survives the decoder -- so the preview would drift
        from the returned IMAGE for no reason anyone could see.

        Doing it per clip rather than once for the whole tensor is exact:
        the expression is elementwise per channel, so slicing commutes with it
        bit for bit.
        """
        import torch

        with torch.no_grad():
            if self._device is not None or self._dtype is not None:
                z = z.to(device=self._device, dtype=self._dtype)
            self.decode_calls += 1
            self.last_input_device = str(getattr(z, "device", ""))
            self.last_input_dtype = str(getattr(z, "dtype", ""))
            z = self._adapter.denormalize(z)
            return self._adapter._adaptive_decode(z)

    def blend(self, a: Any, b: Any, blend_extent: int, dim: int) -> Any:
        """Cross-fade on the decode device, in the decode dtype (as upstream)."""
        return self._adapter.blend(a, b, blend_extent, dim)

    def _finalize_pixels(self, part: Any) -> Any:
        """The last operator before the frames leave the decode device."""
        return self._adapter._finalize_pixels(part)

    def denormalize(self, z: Any) -> Any:
        """Exposed for tests/diagnostics; the decode path calls it itself."""
        return self._adapter.denormalize(z)

    def policy(self) -> Dict[str, Any]:
        """What the preview lane did with the latents, for the node's report."""
        return {
            "decode_device": str(self._device),
            "decode_dtype": str(self._dtype),
            "output_device": str(self._output_device),
            "order": "to(device,vae_dtype) -> denormalize -> _adaptive_decode "
                     "-> blend -> _finalize_pixels -> host",
            "denormalize": "on the decode device, in vae_dtype (official order)",
        }


def build_video_decoder(video_vae: Any) -> IncrementalVideoDecoder:
    """Streaming video decoder around a ``comfy.sd.VAE`` holding the H3 video VAE.

    No ``denormalize=`` is passed on purpose. ``IncrementalVideoDecoder`` would
    apply it to the latents *as pushed* -- on the host, in whatever dtype they
    arrived in -- whereas upstream denormalizes after the cast to the VAE's
    dtype. The step is done inside :meth:`_DeviceBoundVideoDecoder._adaptive_decode`
    instead, which is where upstream does it.
    """
    decoder = _DeviceBoundVideoDecoder(video_vae)
    return IncrementalVideoDecoder(decoder, params=VideoChunkParams.from_vae(decoder))


def build_audio_decoder(
    audio_vae: Any, config: PipelineConfig
) -> OverlapSaveAudioDecoder:
    """Overlap-save audio decoder around a ``comfy.sd.VAE`` holding the H3 audio VAE.

    Same reason as the video lane for bypassing ``VAE.decode``: it would call
    ``load_models_gpu`` per block, and its whole-clip form is what OOMed on the
    24 GiB card. The inner ``decode`` already denormalizes the latents and
    clamps to [-1, 1], and returns ``[B, 2, samples]`` -- samples last, which is
    the axis the overlap-save planner slices. The wrapper's ``process_output``
    is the identity for this VAE, so nothing is skipped.

    What is *not* applied per block is the per-clip loudness normalisation:
    ``vae_decode_audio`` divides by a standard deviation taken over the whole
    clip, which a stream does not have until it ends. The collector therefore
    holds raw decoder output and
    :meth:`StreamingPipeline.finalize_audio` applies that expression once, at
    the end, to the finished waveform.
    """
    inner = getattr(audio_vae, "first_stage_model", audio_vae)
    device = getattr(audio_vae, "device", None)
    dtype = getattr(audio_vae, "vae_dtype", None)
    output_device = getattr(audio_vae, "output_device", None)

    def decode_fn(z: Any) -> Any:
        import torch

        with torch.no_grad():
            if device is not None or dtype is not None:
                z = z.to(device=device, dtype=dtype)
            out = inner.decode(z)
            if output_device is not None:
                out = out.to(device=output_device, dtype=torch.float32)
            else:
                out = out.to(dtype=torch.float32)
        return out

    geometry = AudioLatentGeometry.from_vae(audio_vae)
    return OverlapSaveAudioDecoder(
        decode_fn,
        margin=int(config.audio_margin_latents),
        block_latents=int(config.audio_block_latents),
        geometry=geometry,
    )


def build_muxer_config(config: PipelineConfig) -> Any:
    """The ``MuxerConfig`` for a preview stream (no PyAV needed to build it)."""
    from raven_streaming.media.mp4_writer import MuxerConfig

    return MuxerConfig(
        width=int(config.width),
        height=int(config.height),
        fps=config.fps,
        sample_rate=int(config.sample_rate),
        channels=int(config.channels),
        with_audio=True,
        video_bitrate=int(config.video_bitrate),
        audio_bitrate=int(config.audio_bitrate),
        segment_frames=int(config.idr_interval_frames),
        fragment_mode=config.fragment_mode,
        # A preview never promises random access, so an encoder that cannot
        # prove a forced IDR is still usable; and a missed IDR must not abort a
        # run that is otherwise fine.
        strict_idr=False,
    )


def build_muxer(config: PipelineConfig) -> Any:
    """Open a fragmented MP4 muxer for the preview stream. Needs PyAV."""
    from raven_streaming.media.mp4_writer import FragmentedMP4Muxer

    return FragmentedMP4Muxer(build_muxer_config(config)).open()


def build_media_pipeline(
    *,
    video_vae: Any,
    audio_vae: Any,
    config: PipelineConfig,
    sink: Any = None,
    log: Optional[logging.Logger] = None,
    muxer: Any = None,
    memory_budget: Optional[Dict[str, Any]] = None,
) -> StreamingPipeline:
    """Build all three lanes: both collectors always, the preview if it can be had.

    The two collectors are constructed first and unguarded -- without them the
    node has no IMAGE and no AUDIO to return, and pretending otherwise would
    show up much later as a missing output. Only what the *preview* adds on top
    (PyAV, an encoder, a sink) sits behind a try/except: no sink, no PyAV, no
    libx264 all mean "no preview", never "no run".
    """
    log_ = log if log is not None else logger
    video_decoder = build_video_decoder(video_vae)
    audio_decoder = build_audio_decoder(audio_vae, config)
    image_device = getattr(video_vae, "output_device", None)

    preview_muxer = None
    reason = ""
    if sink is None:
        reason = "no preview sink"
    else:
        try:
            preview_muxer = muxer if muxer is not None else build_muxer(config)
        except Exception as exc:  # noqa: BLE001 - the preview is the optional half
            preview_muxer = None
            reason = "preview unavailable: {}: {}".format(type(exc).__name__, exc)
            log_.warning(
                "raven preview: could not start the media lane (%s: %s); sampling "
                "and the IMAGE/AUDIO outputs continue without a preview",
                type(exc).__name__,
                exc,
            )

    return StreamingPipeline(
        config=config,
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
        muxer=preview_muxer,
        sink=sink,
        log=log,
        image_device=image_device,
        memory_budget=memory_budget,
        preview_disabled_reason=reason,
        decode_policy=video_decoder.decoder.policy(),
    )
