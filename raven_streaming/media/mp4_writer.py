"""Fragmented MP4 muxer writing into a custom, non-seekable byte sink.

Design constraints this module exists to prove out:

* PyAV's MP4 muxer must accept a *write-only* Python object (no ``seek``), so
  that bytes can be handed to a consumer the instant they are produced.  That
  requires ``movflags=frag_keyframe+empty_moov+default_base_moof``: without
  ``empty_moov`` the muxer wants to rewrite the ``moov`` at close time.
* Every segment must start on an IDR frame, otherwise a consumer that joins at
  a fragment boundary cannot decode it.  ``frag_keyframe`` cuts a fragment at
  each keyframe, so segment control reduces to forcing IDR on the first frame
  of each segment.
* The muxer API is deliberately **backpressure-free**: nothing here blocks,
  queues, or drops.  Bytes accumulate in :class:`FragmentedMP4Muxer` and the
  caller pulls them with :meth:`take_init_segment` / :meth:`take_fragments`.
  Any flow-control policy lives entirely in the caller.

The output is a normal fragmented MP4: concatenating the init segment with all
fragments in order yields a file that decodes as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

from .clock import DEFAULT_AUDIO_SAMPLE_RATE, DEFAULT_VIDEO_FPS, MediaClock
from .codecs import (
    AAC_ENCODER_CANDIDATES,
    EncoderCandidate,
    EncoderUnavailable,
    H264_ENCODER_CANDIDATES,
    force_idr,
    import_av,
    packet_frame_index,
    select_encoder,
)
from .mp4_boxes import FragmentedMP4Segmenter, Segment

__all__ = [
    "DEFAULT_MOVFLAGS",
    "SEGMENT_MOVFLAGS",
    "PREVIEW_MOVFLAGS",
    "FRAGMENT_MODES",
    "FRAGMENT_MODE_OPTIONS",
    "PREVIEW_CONTAINER_OPTIONS",
    "WriteOnlySink",
    "MuxerConfig",
    "FragmentedMP4Muxer",
    "FragmentRecord",
    "IDRViolation",
    "concat_stream",
]

#: Base flags shared by both fragment modes.
#: ``empty_moov``          - no sample tables in moov, so no seeking needed
#: ``default_base_moof``   - self-contained fragments (ISO-BMFF byte streams)
_BASE_MOVFLAGS = "empty_moov+default_base_moof"

#: ``frag_keyframe`` - cut a fragment at every video keyframe.  Measured
#: behaviour (PyAV 18.1 / libx264, FFmpeg 8): fragment N is written when the
#: *first frame of segment N+1* is muxed, i.e. a one-segment mux delay.  Every
#: fragment is independently decodable after the init segment.
SEGMENT_MOVFLAGS = "frag_keyframe+" + _BASE_MOVFLAGS

#: ``frag_every_frame`` - one fragment per frame.  Measured delay is <= 1 frame
#: (video-only), but only the fragments that happen to carry an IDR are
#: independently decodable; the rest are valid **only when appended in order**
#: after the init segment.  That is enough for a preview lane (MSE-style
#: sequential append) and is not enough for random access.
PREVIEW_MOVFLAGS = "frag_every_frame+" + _BASE_MOVFLAGS

FRAGMENT_MODES = {
    "keyframe": SEGMENT_MOVFLAGS,
    "every_frame": PREVIEW_MOVFLAGS,
}

#: Measured on PyAV 18.1 / FFmpeg 8 (libavformat 62): with ``frag_every_frame``
#: *and* an audio track, one-sample fragments make the muxer mis-assign sample
#: durations - the first video packet jumps to the wrong pts and the last two
#: packets collide on one pts, so the stream decodes one frame short.  Setting
#: ``min_frag_duration=1`` (1 microsecond, i.e. still a fragment per frame)
#: restores correct per-fragment timing: all frames decode, no duplicate pts,
#: and the <= 1 frame delay is unchanged.  Video-only is unaffected either way.
PREVIEW_CONTAINER_OPTIONS = {"min_frag_duration": "1"}

FRAGMENT_MODE_OPTIONS = {
    "keyframe": {},
    "every_frame": dict(PREVIEW_CONTAINER_OPTIONS),
}

#: Backwards-compatible default (segment mode).
DEFAULT_MOVFLAGS = SEGMENT_MOVFLAGS


class WriteOnlySink:
    """A file-like object exposing only ``write``.

    Deliberately has no ``seek``/``tell``: PyAV inspects the object and marks
    the AVIO context non-seekable, which is what forces FFmpeg down the
    streaming code path instead of the rewrite-at-close path.
    """

    def __init__(self, on_write) -> None:
        self._on_write = on_write
        self.bytes_written = 0
        self.write_calls = 0

    def write(self, data) -> int:
        payload = bytes(data)
        self.bytes_written += len(payload)
        self.write_calls += 1
        self._on_write(payload)
        return len(payload)

    def flush(self) -> None:  # pragma: no cover - FFmpeg may or may not call it
        return None


@dataclass
class FragmentRecord:
    """When a fragment became available, measured at *production* time."""

    index: int
    size: int
    frames_written: int
    samples_written: int
    bytes_written: int
    after_close: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "size": self.size,
            "frames_written": self.frames_written,
            "samples_written": self.samples_written,
            "bytes_written": self.bytes_written,
            "after_close": self.after_close,
        }


@dataclass
class IDRViolation:
    """A segment whose first video packet was not a keyframe."""

    segment_index: int
    frame_index: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "frame_index": self.frame_index,
            "reason": self.reason,
        }


@dataclass
class MuxerConfig:
    width: int
    height: int
    fps: Fraction = DEFAULT_VIDEO_FPS
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    channels: int = 2
    with_audio: bool = True
    video_bitrate: int = 6_000_000
    audio_bitrate: int = 128_000
    #: frames per segment; the first frame of each segment is forced to IDR
    segment_frames: int = 24
    video_encoder: Optional[str] = None
    audio_encoder: Optional[str] = None
    #: "keyframe" (one fragment per segment, independently decodable) or
    #: "every_frame" (one fragment per frame, sequential append only)
    fragment_mode: str = "keyframe"
    #: explicit override; when None it is derived from ``fragment_mode``
    movflags: Optional[str] = None
    #: flush the AVIO buffer after every packet so bytes reach the sink early
    flush_packets: bool = True
    #: apply the mode's measured container-option workarounds (see
    #: :data:`PREVIEW_CONTAINER_OPTIONS`); turn off only to reproduce the bug
    apply_mode_options: bool = True
    #: raise the moment a segment's first video packet is not a keyframe
    strict_idr: bool = True
    #: require the encoder to pass the behavioural forced-IDR probe
    require_forced_idr: Optional[bool] = None
    extra_container_options: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fps = Fraction(self.fps)
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p requires even width/height")
        if self.segment_frames < 1:
            raise ValueError("segment_frames must be >= 1")
        if self.fragment_mode not in FRAGMENT_MODES:
            raise ValueError(
                "fragment_mode must be one of {}, got {!r}".format(
                    sorted(FRAGMENT_MODES), self.fragment_mode
                )
            )
        if self.movflags is None:
            self.movflags = FRAGMENT_MODES[self.fragment_mode]
        if self.require_forced_idr is None:
            # preview mode never promises random access, so an encoder without a
            # provable forced IDR is acceptable there and only there
            self.require_forced_idr = self.fragment_mode == "keyframe"

    @property
    def clock(self) -> MediaClock:
        return MediaClock(self.fps, self.sample_rate)

    @property
    def fragments_are_independently_decodable(self) -> bool:
        """Only segment mode promises per-fragment random access."""
        return self.fragment_mode == "keyframe"

    def container_options(self) -> Dict[str, str]:
        """Every option handed to ``av.open``, in precedence order."""
        options: Dict[str, str] = {"movflags": self.movflags}
        if self.flush_packets:
            options["flush_packets"] = "1"
        if self.apply_mode_options:
            options.update(FRAGMENT_MODE_OPTIONS.get(self.fragment_mode, {}))
        options.update(self.extra_container_options)
        return options


class FragmentedMP4Muxer:
    """Pure fMP4 muxer: frames in, appendable byte segments out."""

    def __init__(self, config: MuxerConfig) -> None:
        self.config = config
        self.clock = config.clock
        self._av = import_av()
        self._segmenter = FragmentedMP4Segmenter(on_fragment=self._on_fragment)
        self._sink = WriteOnlySink(self._segmenter.feed)
        self._container = None
        self._video_stream = None
        self._audio_stream = None
        self._video_candidate: Optional[EncoderCandidate] = None
        self._audio_candidate: Optional[EncoderCandidate] = None
        self._video_probe = None
        self._audio_probe = None
        self._frame_index = 0
        self._sample_index = 0
        self._audio_buffer = None  # numpy array [channels, n], float32
        self._closed = False
        self._opened = False
        self._container_options: Dict[str, str] = {}
        self._fragments_before_close = 0
        # -- evidence collected while muxing --------------------------------
        self._fragment_log: List[FragmentRecord] = []
        self._idr_violations: List[IDRViolation] = []
        self._segment_first_packet: Dict[int, bool] = {}
        self._forced_idr_frames: List[int] = []
        self._video_packets = 0
        self._audio_packets = 0
        self._first_video_pts_ticks: Optional[int] = None
        self._first_audio_pts_samples: Optional[int] = None
        self._audio_frames_encoded = 0
        self._audio_padded_tail = 0
        self._bytes_before_close: Optional[int] = None

    # -- instrumentation ----------------------------------------------------

    def _on_fragment(self, segment: Segment) -> None:
        """Fires inside ``mux()``, i.e. at fragment *production* time."""
        self._fragment_log.append(
            FragmentRecord(
                index=segment.index,
                size=len(segment.data),
                frames_written=self._frame_index,
                samples_written=self._sample_index,
                bytes_written=self._sink.bytes_written,
                after_close=self._closed,
            )
        )

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "FragmentedMP4Muxer":
        if self._opened:
            return self
        av = self._av
        cfg = self.config

        self._video_candidate, self._video_probe = select_encoder(
            H264_ENCODER_CANDIDATES,
            preferred=cfg.video_encoder,
            require_forced_idr=bool(cfg.require_forced_idr),
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            segment_frames=min(cfg.segment_frames, 6),
            segments=3,
        )
        if cfg.with_audio:
            self._audio_candidate, self._audio_probe = select_encoder(
                AAC_ENCODER_CANDIDATES,
                preferred=cfg.audio_encoder,
                sample_rate=cfg.sample_rate,
                channels=cfg.channels,
            )

        options = cfg.container_options()
        self._container = av.open(self._sink, mode="w", format="mp4", options=options)
        self._container_options = options

        vs = self._container.add_stream(
            self._video_candidate.name,
            rate=cfg.fps,
            options=self._video_candidate.encoder_options,
        )
        vs.width = cfg.width
        vs.height = cfg.height
        vs.pix_fmt = self._video_candidate.pix_fmt or "yuv420p"
        vs.bit_rate = cfg.video_bitrate
        vs.time_base = Fraction(1, self.clock.ticks_per_second)
        # one IDR per segment; FFmpeg also needs gop_size so it does not insert
        # extra IDRs of its own inside a segment
        try:
            vs.codec_context.gop_size = cfg.segment_frames
        except Exception:
            pass
        self._video_stream = vs

        if cfg.with_audio:
            aud = self._container.add_stream(
                self._audio_candidate.name,
                rate=cfg.sample_rate,
                options=dict(self._audio_candidate.options),
            )
            aud.bit_rate = cfg.audio_bitrate
            try:
                aud.layout = "stereo" if cfg.channels == 2 else "mono"
            except Exception:
                pass
            aud.time_base = Fraction(1, cfg.sample_rate)
            self._audio_stream = aud

        self._opened = True
        return self

    def __enter__(self) -> "FragmentedMP4Muxer":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- introspection ------------------------------------------------------

    @property
    def video_encoder_name(self) -> Optional[str]:
        return self._video_candidate.name if self._video_candidate else None

    @property
    def audio_encoder_name(self) -> Optional[str]:
        return self._audio_candidate.name if self._audio_candidate else None

    @property
    def audio_frame_size(self) -> int:
        if self._audio_stream is None:
            return 0
        return int(self._audio_stream.codec_context.frame_size or 1024)

    @property
    def frames_written(self) -> int:
        return self._frame_index

    @property
    def samples_written(self) -> int:
        return self._sample_index

    @property
    def bytes_written(self) -> int:
        return self._sink.bytes_written

    # -- video --------------------------------------------------------------

    def is_segment_start(self, frame_index: Optional[int] = None) -> bool:
        idx = self._frame_index if frame_index is None else frame_index
        return idx % self.config.segment_frames == 0

    def write_video_frame(self, image, force_keyframe: Optional[bool] = None) -> int:
        """Encode one RGB frame.

        ``image`` is either an ``av.VideoFrame`` or an HxWx3 uint8 array.
        Returns the frame index that was written.
        """
        self._require_open()
        av = self._av
        idx = self._frame_index

        if isinstance(image, av.VideoFrame):
            frame = image
        else:
            frame = av.VideoFrame.from_ndarray(_as_rgb_uint8(image), format="rgb24")

        if force_keyframe is None:
            force_keyframe = self.is_segment_start(idx)
        if force_keyframe:
            if not force_idr(frame):
                raise RuntimeError(
                    "PyAV rejected pict_type=I on frame {}: cannot force an IDR, so "
                    "segment boundaries cannot be guaranteed".format(idx)
                )
            self._forced_idr_frames.append(idx)

        frame.pts = self.clock.frames_to_ticks(idx)
        frame.time_base = self._video_stream.time_base
        for packet in self._video_stream.encode(frame):
            self._track_video_packet(packet)
            self._container.mux(packet)

        self._frame_index += 1
        return idx

    def _track_video_packet(self, packet) -> None:
        """Verify segment-start packets are keyframes, mapping by pts not order.

        The encoder may hold a frame back and emit its packet later, so the
        packet in flight is *not* necessarily the frame just handed in.  The
        frame index is recovered from ``pts * time_base * fps`` and only the
        packet that really belongs to a segment's first frame is checked.
        """
        self._video_packets += 1
        frame_index = packet_frame_index(packet, self.config.fps)
        if frame_index is None:
            return
        if self._first_video_pts_ticks is None:
            self._first_video_pts_ticks = self.clock.frames_to_ticks(frame_index)
        if frame_index % self.config.segment_frames != 0:
            return

        segment_index = frame_index // self.config.segment_frames
        is_key = bool(packet.is_keyframe)
        # first packet seen for this segment start wins
        if segment_index in self._segment_first_packet:
            return
        self._segment_first_packet[segment_index] = is_key
        if is_key:
            return

        violation = IDRViolation(
            segment_index=segment_index,
            frame_index=frame_index,
            reason="first packet of segment {} (frame {}) is not a keyframe; "
            "encoder {} ignored the forced IDR".format(
                segment_index, frame_index, self.video_encoder_name
            ),
        )
        self._idr_violations.append(violation)
        if self.config.strict_idr:
            raise RuntimeError(violation.reason)

    # -- audio --------------------------------------------------------------

    def write_audio(self, pcm) -> int:  # noqa: D401
        """Buffer interleaved-by-plane PCM and encode whole AAC frames.

        ``pcm`` is a ``[channels, samples]`` float32 array in [-1, 1] (a
        ``[samples]`` mono array is accepted and broadcast).  Returns the number
        of samples appended.
        """
        self._require_open()
        if self._audio_stream is None:
            raise RuntimeError("muxer was configured with with_audio=False")
        np = _numpy()
        data = _as_planar_float32(np, pcm, self.config.channels)
        if self._audio_buffer is None or self._audio_buffer.shape[1] == 0:
            self._audio_buffer = data
        else:
            self._audio_buffer = np.concatenate([self._audio_buffer, data], axis=1)
        self._drain_audio(partial=False)
        return int(data.shape[1])

    def _drain_audio(self, partial: bool) -> None:
        np = _numpy()
        av = self._av
        if self._audio_buffer is None:
            return
        size = self.audio_frame_size or 1024
        buf = self._audio_buffer
        pos = 0
        while buf.shape[1] - pos >= size:
            block = buf[:, pos:pos + size]
            self._encode_audio_block(av, np, block, size)
            pos += size
        if partial and buf.shape[1] - pos > 0:
            tail = buf[:, pos:]
            padded = np.zeros((tail.shape[0], size), dtype=np.float32)
            padded[:, : tail.shape[1]] = tail
            self._encode_audio_block(av, np, padded, tail.shape[1])
            pos = buf.shape[1]
        self._audio_buffer = buf[:, pos:]

    def _encode_audio_block(self, av, np, block, valid_samples: int) -> None:
        fmt = self._audio_stream.codec_context.format.name
        layout = "stereo" if self.config.channels == 2 else "mono"
        if fmt.startswith("s16"):
            arr = np.clip(block, -1.0, 1.0)
            arr = (arr * 32767.0).astype(np.int16)
            if fmt == "s16":  # packed
                arr = arr.T.reshape(1, -1)
        else:
            arr = np.ascontiguousarray(block, dtype=np.float32)
            if fmt == "flt":  # packed
                arr = arr.T.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format=fmt, layout=layout)
        frame.sample_rate = self.config.sample_rate
        frame.time_base = self._audio_stream.time_base
        frame.pts = self._sample_index
        self._audio_frames_encoded += 1
        if valid_samples < block.shape[1]:
            self._audio_padded_tail += int(block.shape[1] - valid_samples)
        for packet in self._audio_stream.encode(frame):
            self._audio_packets += 1
            if self._first_audio_pts_samples is None and packet.pts is not None:
                tb = packet.time_base
                try:
                    seconds = Fraction(int(packet.pts)) * Fraction(tb)
                    self._first_audio_pts_samples = int(seconds * self.config.sample_rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    self._first_audio_pts_samples = int(packet.pts)
            self._container.mux(packet)
        self._sample_index += int(valid_samples)

    # -- output -------------------------------------------------------------

    def take_init_segment(self) -> Optional[bytes]:
        return self._segmenter.take_init_segment()

    def take_fragments(self) -> List[Segment]:
        return self._segmenter.take_fragments()

    def take_trailer(self) -> List[Segment]:
        return self._segmenter.take_trailer()

    def close(self) -> None:
        if self._closed:
            return
        self._bytes_before_close = self._sink.bytes_written
        self._fragments_before_close = len(self._fragment_log)
        self._closed = True
        if self._container is not None:
            try:
                if self._audio_stream is not None:
                    self._drain_audio(partial=True)
                    for packet in self._audio_stream.encode(None):
                        self._audio_packets += 1
                        self._container.mux(packet)
                if self._video_stream is not None:
                    for packet in self._video_stream.encode(None):
                        self._track_video_packet(packet)
                        self._container.mux(packet)
            finally:
                self._container.close()
                self._container = None
        self._segmenter.close()

    # -- evidence -----------------------------------------------------------

    @property
    def fragment_log(self) -> List[FragmentRecord]:
        """Fragment production events, timestamped by frames/samples written."""
        return list(self._fragment_log)

    @property
    def idr_violations(self) -> List[IDRViolation]:
        return list(self._idr_violations)

    @property
    def bytes_before_close(self) -> Optional[int]:
        return self._bytes_before_close

    @property
    def close_only_bytes(self) -> Optional[int]:
        if self._bytes_before_close is None:
            return None
        return self._sink.bytes_written - self._bytes_before_close

    @property
    def first_fragment_after_frames(self) -> Optional[int]:
        for record in self._fragment_log:
            if not record.after_close:
                return record.frames_written
        return None

    @property
    def steady_fragment_delay_frames(self) -> Optional[int]:
        """Frames between consecutive pre-close fragments, once in steady state.

        In ``keyframe`` mode this should equal ``segment_frames`` (one segment
        of mux delay); in ``every_frame`` mode it should be 1.
        """
        pre = [r.frames_written for r in self._fragment_log if not r.after_close]
        # several fragments can land at the same frame count (audio interleave),
        # so collapse to distinct cursor positions before measuring the cadence
        distinct: List[int] = []
        for value in pre:
            if not distinct or value != distinct[-1]:
                distinct.append(value)
        if len(distinct) < 2:
            return None
        deltas = sorted(b - a for a, b in zip(distinct, distinct[1:]))
        return deltas[len(deltas) // 2]

    def av_first_pts_skew_samples(self) -> Optional[int]:
        """|first video pts - first audio pts| in samples, at the *encoder*.

        For AAC this is the encoder priming delay: the first packet carries a
        negative pts of one frame (1024 samples = 32 ms at 32 kHz), which the
        muxer normalises away so the container's ``start_time`` is 0.  A value
        above one AAC frame means the lanes were fed out of step.
        """
        if self._first_video_pts_ticks is None or self._first_audio_pts_samples is None:
            return None
        video_samples = self.clock.ticks_to_samples_floor(self._first_video_pts_ticks)
        return abs(video_samples - self._first_audio_pts_samples)

    @property
    def audio_priming_samples(self) -> Optional[int]:
        """Encoder delay: ``-first_audio_pts`` when the encoder primes (AAC)."""
        if self._first_audio_pts_samples is None:
            return None
        return max(0, -self._first_audio_pts_samples)

    def report(self) -> Dict[str, Any]:
        """A JSON-serialisable summary of everything the mux proved."""
        cfg = self.config
        return {
            "video_encoder": self.video_encoder_name,
            "audio_encoder": self.audio_encoder_name,
            "fragment_mode": cfg.fragment_mode,
            "movflags": cfg.movflags,
            "container_options": dict(self._container_options),
            "flush_packets": cfg.flush_packets,
            "segment_frames": cfg.segment_frames,
            "fragments_independently_decodable": cfg.fragments_are_independently_decodable,
            "frames_written": self._frame_index,
            "samples_written": self._sample_index,
            "video_packets": self._video_packets,
            "audio_packets": self._audio_packets,
            "forced_idr_frames": list(self._forced_idr_frames),
            "segment_first_packet_is_keyframe": dict(self._segment_first_packet),
            "idr_violations": [v.to_dict() for v in self._idr_violations],
            "fragments": [r.to_dict() for r in self._fragment_log],
            "first_fragment_after_frames": self.first_fragment_after_frames,
            "steady_fragment_delay_frames": self.steady_fragment_delay_frames,
            "fragments_before_close": sum(
                1 for r in self._fragment_log if not r.after_close
            ),
            "bytes_before_close": self._bytes_before_close,
            "close_only_bytes": self.close_only_bytes,
            "total_bytes": self._sink.bytes_written,
            "sink_write_calls": self._sink.write_calls,
            "first_video_pts_ticks": self._first_video_pts_ticks,
            "first_audio_pts_samples": self._first_audio_pts_samples,
            "av_first_pts_skew_samples": self.av_first_pts_skew_samples(),
            "audio_frame_size": self.audio_frame_size,
            "audio_frames_encoded": self._audio_frames_encoded,
            "audio_padded_tail_samples": self._audio_padded_tail,
            "audio_priming_samples": self.audio_priming_samples,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("muxer is closed")
        if not self._opened:
            self.open()


# -- helpers ---------------------------------------------------------------


def _numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise EncoderUnavailable("numpy is required for array input: {}".format(exc)) from exc
    return np


def _as_rgb_uint8(image):
    np = _numpy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 image, got shape {}".format(arr.shape))
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) if arr.dtype.kind == "f" else arr
        if arr.dtype.kind == "f":
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_planar_float32(np, pcm, channels: int):
    arr = np.asarray(pcm, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.repeat(arr[None, :], channels, axis=0)
    if arr.ndim != 2:
        raise ValueError("expected [channels, samples] PCM, got shape {}".format(arr.shape))
    if arr.shape[0] != channels:
        if arr.shape[1] == channels:  # caller passed [samples, channels]
            arr = arr.T
        elif arr.shape[0] == 1:
            arr = np.repeat(arr, channels, axis=0)
        else:
            raise ValueError(
                "PCM has {} channels, muxer configured for {}".format(arr.shape[0], channels)
            )
    return np.ascontiguousarray(arr, dtype=np.float32)


def concat_stream(init: Optional[bytes], fragments: Sequence[Segment]) -> bytes:
    """Concatenate an init segment and fragments back into a playable file."""
    parts: List[bytes] = []
    if init:
        parts.append(init)
    parts.extend(f.data for f in fragments)
    return b"".join(parts)
