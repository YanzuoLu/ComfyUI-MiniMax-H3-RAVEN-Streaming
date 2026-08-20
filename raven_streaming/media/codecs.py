"""H.264 / AAC encoder discovery for the streaming media lane.

PyAV ships whatever encoders its bundled FFmpeg was built with, and that set
differs wildly between a Comfy wheel on a workstation and an H100 node.  The
media lane therefore never hardcodes an encoder: it probes a preference chain
and reports precisely what it found.

Preference chain for H.264:

1. ``h264_nvenc``   - hardware, keeps the GPU-resident frames off the CPU
2. ``libx264``      - the usual software fallback
3. ``libopenh264``  - permissively licensed fallback shipped by some builds

``av`` is imported lazily so this module stays importable (and testable) in an
environment without PyAV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EncoderCandidate",
    "EncoderProbeResult",
    "EncoderUnavailable",
    "H264_ENCODER_CANDIDATES",
    "AAC_ENCODER_CANDIDATES",
    "import_av",
    "av_version",
    "describe_environment",
    "container_format_available",
    "codec_available",
    "packet_frame_index",
    "force_idr",
    "probe_encoder",
    "probe_encoders",
    "probe_forced_idr",
    "select_encoder",
]


class EncoderUnavailable(RuntimeError):
    """No encoder in the preference chain could be opened."""


def import_av():
    """Import PyAV, raising a helpful error instead of a bare ImportError."""
    try:
        import av  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise EncoderUnavailable(
            "PyAV is not importable ({}). The media lane expects the PyAV that "
            "ships with ComfyUI; run with that interpreter or pass --comfy-root.".format(exc)
        ) from exc
    return av


def av_version() -> Tuple[str, Tuple[int, ...]]:
    av = import_av()
    raw = getattr(av, "__version__", "0")
    parts: List[int] = []
    for chunk in str(raw).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return str(raw), tuple(parts)


def container_format_available(name: str) -> Tuple[Optional[bool], str]:
    """Is ``name`` a muxable container format?  Compatible with PyAV 15..18+.

    ``av.formats.available`` was removed (PyAV 18 has no ``av.formats`` module
    at all), so several APIs are tried in order and *none* of them are allowed
    to raise.  Returns ``(available_or_None, how_it_was_determined)``; ``None``
    means "could not be determined", which is not the same as "missing".
    """
    av = import_av()

    # 1. ContainerFormat has been stable across every PyAV generation
    try:
        from av.format import ContainerFormat  # type: ignore

        fmt = ContainerFormat(name, "w")
        return (fmt.name == name or name in str(fmt.name)), "av.format.ContainerFormat"
    except Exception:
        pass

    # 2. module-level set of names (PyAV 9..17)
    for attr_path in (("formats_available",), ("format", "formats_available"),
                      ("formats", "available")):
        try:
            obj: Any = av
            for attr in attr_path:
                obj = getattr(obj, attr)
            return (name in obj), "av." + ".".join(attr_path)
        except Exception:
            continue

    return None, "undetermined"


def codec_available(name: str, mode: str = "w") -> Tuple[bool, Optional[str]]:
    """Is ``name`` present as an encoder (``w``) / decoder (``r``)?"""
    av = import_av()
    try:
        av.codec.Codec(name, mode)
    except Exception as exc:
        return False, "{}: {}".format(type(exc).__name__, exc)
    return True, None


def describe_environment() -> Dict[str, Any]:
    """A JSON-serialisable description of the PyAV/FFmpeg build.

    Never raises: every lookup that a given PyAV generation might not support
    degrades to ``None`` with the reason recorded.
    """
    info: Dict[str, Any] = {
        "pyav_version": None,
        "pyav_version_tuple": None,
        "library_versions": None,
        "mp4_muxer": None,
        "mp4_muxer_probe": None,
        "errors": [],
    }
    try:
        raw, parts = av_version()
        info["pyav_version"] = raw
        info["pyav_version_tuple"] = list(parts)
    except Exception as exc:
        info["errors"].append("av_version: {}".format(exc))
        return info

    av = import_av()
    try:
        versions = getattr(av, "library_versions", None)
        if versions is not None:
            info["library_versions"] = {k: list(v) for k, v in dict(versions).items()}
    except Exception as exc:
        info["errors"].append("library_versions: {}".format(exc))

    try:
        available, how = container_format_available("mp4")
        info["mp4_muxer"] = available
        info["mp4_muxer_probe"] = how
    except Exception as exc:  # pragma: no cover - defensive
        info["errors"].append("container_format_available: {}".format(exc))
    return info


def packet_frame_index(packet: Any, fps: Any) -> Optional[int]:
    """Frame index a muxed packet belongs to, from its own pts *and* time_base.

    Encoders reorder, and PyAV rescales pts at ``mux()`` time (the encoder's
    ``CodecContext.time_base`` is ``1/fps`` while the stream's may be the media
    clock's), so the frame a packet belongs to must be recovered from
    ``pts * time_base * fps`` rather than assumed from the write order.
    """
    pts = getattr(packet, "pts", None)
    if pts is None:
        return None
    time_base = getattr(packet, "time_base", None)
    if time_base is None:
        return None
    try:
        seconds = Fraction(int(pts)) * Fraction(time_base)
        value = seconds * Fraction(fps)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    # round half up; pts is exact on the media grid so this only guards jitter
    return int((value * 2 + 1) // 2)


def force_idr(frame: Any) -> bool:
    """Mark a video frame as an I frame.  Returns True if the flag stuck.

    PyAV 18's ``pict_type`` setter rejects plain strings (it wants the enum or
    an int), older releases accepted ``"I"``; both paths are tried.
    """
    try:
        from av.video.frame import PictureType  # type: ignore

        frame.pict_type = PictureType.I
        return True
    except Exception:
        pass
    for value in ("I", 1):
        try:
            frame.pict_type = value
            return True
        except Exception:
            continue
    return False


@dataclass(frozen=True)
class EncoderCandidate:
    """One entry in a preference chain."""

    name: str
    kind: str  # "video" | "audio"
    options: Mapping[str, str] = field(default_factory=dict)
    pix_fmt: Optional[str] = None
    sample_fmt: Optional[str] = None
    hardware: bool = False
    notes: str = ""
    #: Encoder options that make a forced I frame a *closed-GOP IDR*.  FFmpeg
    #: silently ignores unknown codec options, so these are re-probed by
    #: behaviour (:func:`probe_forced_idr`) and dropped if they don't open.
    idr_options: Mapping[str, str] = field(default_factory=dict)
    #: How IDR is requested; "unverified" means the encoder has no documented
    #: forced-IDR control and must prove itself behaviourally or be excluded.
    idr_control: str = "pict_type"

    def _replace(self, **changes: Any) -> "EncoderCandidate":
        fields = dict(
            name=self.name,
            kind=self.kind,
            options=dict(self.options),
            pix_fmt=self.pix_fmt,
            sample_fmt=self.sample_fmt,
            hardware=self.hardware,
            notes=self.notes,
            idr_options=dict(self.idr_options),
            idr_control=self.idr_control,
        )
        fields.update(changes)
        return EncoderCandidate(**fields)

    def with_options(self, **extra: str) -> "EncoderCandidate":
        merged: Dict[str, str] = dict(self.options)
        merged.update({k: str(v) for k, v in extra.items()})
        return self._replace(options=merged)

    def without_idr_options(self) -> "EncoderCandidate":
        return self._replace(idr_options={}, idr_control="pict_type")

    @property
    def encoder_options(self) -> Dict[str, str]:
        """Options actually handed to the encoder (base + forced-IDR)."""
        merged: Dict[str, str] = dict(self.options)
        merged.update(self.idr_options)
        return merged


#: Low-latency-first H.264 preference chain.
H264_ENCODER_CANDIDATES: Tuple[EncoderCandidate, ...] = (
    EncoderCandidate(
        name="h264_nvenc",
        kind="video",
        options={
            "preset": "p1",
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "delay": "0",
            "bf": "0",
            "repeat_spspps": "1",
        },
        # nvenc documents -forced-idr: "If forcing keyframes, force them as IDR"
        idr_options={"forced-idr": "1", "strict_gop": "1"},
        idr_control="forced-idr",
        pix_fmt="yuv420p",
        hardware=True,
        notes="NVIDIA hardware encoder; requires an NVENC-capable GPU and driver.",
    ),
    EncoderCandidate(
        name="libx264",
        kind="video",
        options={
            "preset": "veryfast",
            "tune": "zerolatency",
            "x264-params": "bframes=0:scenecut=0:open-gop=0:repeat-headers=1",
        },
        # libx264 also exposes -forced-idr in current FFmpeg
        idr_options={"forced-idr": "1"},
        idr_control="forced-idr",
        pix_fmt="yuv420p",
        notes="Software x264, GPL build.",
    ),
    EncoderCandidate(
        name="libopenh264",
        kind="video",
        options={"allow_skip_frames": "0"},
        # OpenH264 exposes no forced-IDR control: it must prove that a forced
        # pict_type=I actually yields an IDR, or it is not usable for
        # independently decodable fragments.
        idr_options={},
        idr_control="unverified",
        pix_fmt="yuv420p",
        notes="Cisco OpenH264; no forced-IDR option, must be verified by behaviour.",
    ),
)

#: AAC preference chain.
AAC_ENCODER_CANDIDATES: Tuple[EncoderCandidate, ...] = (
    EncoderCandidate(
        name="libfdk_aac",
        kind="audio",
        options={},
        sample_fmt="s16",
        notes="Higher quality AAC; only in non-free FFmpeg builds.",
    ),
    EncoderCandidate(
        name="aac",
        kind="audio",
        options={},
        sample_fmt="fltp",
        notes="FFmpeg native AAC, always available.",
    ),
)


@dataclass
class EncoderProbeResult:
    """Outcome of actually opening an encoder and pushing one frame at it."""

    name: str
    kind: str
    present: bool = False
    usable: bool = False
    hardware: bool = False
    pix_fmt: Optional[str] = None
    sample_fmt: Optional[str] = None
    error: Optional[str] = None
    packets: int = 0
    notes: str = ""
    #: forced-IDR behaviour (video only): None = not probed
    forced_idr_ok: Optional[bool] = None
    forced_idr_error: Optional[str] = None
    idr_control: Optional[str] = None
    idr_options_accepted: bool = True
    keyframe_frames: Optional[List[int]] = None
    expected_keyframe_frames: Optional[List[int]] = None

    def describe(self) -> str:
        if not self.present:
            return "{:<14} MISSING: {}".format(self.name, self.error)
        if not self.usable:
            return "{:<14} PRESENT BUT UNUSABLE: {}".format(self.name, self.error)
        bits = ["packets={}".format(self.packets)]
        if self.hardware:
            bits.append("hardware")
        if self.forced_idr_ok is True:
            bits.append("forced-IDR verified via {}".format(self.idr_control))
        elif self.forced_idr_ok is False:
            bits.append("FORCED-IDR FAILED: {}".format(self.forced_idr_error))
        if not self.idr_options_accepted:
            bits.append("idr options rejected, fell back")
        return "{:<14} {}  ({})".format(
            self.name, "OK  " if self.forced_idr_ok is not False else "IDR!", ", ".join(bits)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "present": self.present,
            "usable": self.usable,
            "hardware": self.hardware,
            "pix_fmt": self.pix_fmt,
            "sample_fmt": self.sample_fmt,
            "error": self.error,
            "packets": self.packets,
            "notes": self.notes,
            "forced_idr_ok": self.forced_idr_ok,
            "forced_idr_error": self.forced_idr_error,
            "idr_control": self.idr_control,
            "idr_options_accepted": self.idr_options_accepted,
            "keyframe_frames": self.keyframe_frames,
            "expected_keyframe_frames": self.expected_keyframe_frames,
        }


def _codec_present(name: str) -> Tuple[bool, Optional[str]]:
    return codec_available(name, "w")


def _open_video_context(av, candidate: EncoderCandidate, width, height, fps, options):
    rate = Fraction(fps)
    ctx = av.codec.CodecContext.create(candidate.name, "w")
    ctx.width = width
    ctx.height = height
    ctx.pix_fmt = candidate.pix_fmt or "yuv420p"
    ctx.time_base = Fraction(rate.denominator, rate.numerator)
    try:
        ctx.framerate = rate
    except Exception:
        pass
    ctx.options = dict(options)
    ctx.open()
    return ctx


def _probe_video(
    candidate: EncoderCandidate,
    width: int,
    height: int,
    fps,
    segment_frames: int = 6,
    segments: int = 3,
) -> EncoderProbeResult:
    av = import_av()
    result = EncoderProbeResult(
        name=candidate.name,
        kind="video",
        hardware=candidate.hardware,
        pix_fmt=candidate.pix_fmt,
        notes=candidate.notes,
        idr_control=candidate.idr_control,
    )
    present, err = _codec_present(candidate.name)
    result.present = present
    if not present:
        result.error = err
        return result

    # FFmpeg ignores unknown codec options silently, so "accepted" here only
    # means the context opened; the real test is the behavioural one below.
    options = candidate.encoder_options
    ctx = None
    try:
        ctx = _open_video_context(av, candidate, width, height, fps, options)
    except Exception as exc:
        result.idr_options_accepted = False
        options = dict(candidate.options)
        try:
            ctx = _open_video_context(av, candidate, width, height, fps, options)
            result.error = "idr options rejected ({}), retried without them".format(exc)
        except Exception as exc2:
            result.error = "{}: {}".format(type(exc2).__name__, exc2)
            return result

    try:
        packets = 0
        for i in range(3):
            frame = _solid_video_frame(av, width, height, ctx.pix_fmt, i)
            frame.pts = i
            frame.time_base = ctx.time_base
            packets += len(ctx.encode(frame))
        packets += len(ctx.encode(None))
        result.packets = packets
        result.usable = packets > 0
        if not result.usable:
            result.error = "encoder produced no packets"
    except Exception as exc:
        result.error = "{}: {}".format(type(exc).__name__, exc)
        return result
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    if result.usable:
        idr = probe_forced_idr(
            candidate,
            width=width,
            height=height,
            fps=fps,
            segment_frames=segment_frames,
            segments=segments,
            options=options,
        )
        result.forced_idr_ok = idr["ok"]
        result.forced_idr_error = idr["error"]
        result.keyframe_frames = idr["keyframe_frames"]
        result.expected_keyframe_frames = idr["expected"]
    return result


def probe_forced_idr(
    candidate: EncoderCandidate,
    width: int = 64,
    height: int = 64,
    fps=24,
    segment_frames: int = 6,
    segments: int = 3,
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Prove *by behaviour* that forcing ``pict_type=I`` yields keyframe packets.

    FFmpeg accepts unknown codec options without complaint, so a declared
    ``forced-idr`` flag means nothing on its own.  This encodes ``segments``
    segments, forces an I frame at each segment start, and checks that the
    resulting packets are keyframes exactly there - mapping packets back to
    frames through ``pts * time_base * fps`` so encoder reordering cannot make
    a late packet look like a boundary one.
    """
    av = import_av()
    opts = dict(candidate.encoder_options if options is None else options)
    expected = [i * segment_frames for i in range(segments)]
    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "keyframe_frames": None,
        "expected": expected,
        "pict_type_set": None,
    }

    ctx = None
    try:
        ctx = _open_video_context(av, candidate, width, height, fps, opts)
        packets: List[Tuple[Optional[int], bool]] = []
        pict_type_set = True
        total = segment_frames * segments
        for i in range(total):
            frame = _solid_video_frame(av, width, height, ctx.pix_fmt, i)
            frame.pts = i
            frame.time_base = ctx.time_base
            if i % segment_frames == 0 and not force_idr(frame):
                pict_type_set = False
            for packet in ctx.encode(frame):
                packets.append((packet_frame_index(packet, fps), bool(packet.is_keyframe)))
        for packet in ctx.encode(None):
            packets.append((packet_frame_index(packet, fps), bool(packet.is_keyframe)))

        out["pict_type_set"] = pict_type_set
        keyframes = sorted(idx for idx, is_key in packets if is_key and idx is not None)
        out["keyframe_frames"] = keyframes
        if not pict_type_set:
            out["error"] = "PyAV rejected every pict_type assignment"
        elif len(packets) < total:
            out["error"] = "encoder returned {} packets for {} frames".format(
                len(packets), total
            )
        elif not set(expected).issubset(set(keyframes)):
            out["error"] = "segment starts {} are not all keyframes (keyframes at {})".format(
                sorted(set(expected) - set(keyframes)), keyframes
            )
        else:
            out["ok"] = True
    except Exception as exc:
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
    return out


def _solid_video_frame(av, width: int, height: int, pix_fmt: str, seed: int):
    frame = av.VideoFrame(width, height, pix_fmt)
    for plane_idx, plane in enumerate(frame.planes):
        value = (16 + 40 * seed + 30 * plane_idx) % 240
        plane.update(bytes([value]) * plane.buffer_size)
    return frame


def _probe_audio(candidate: EncoderCandidate, sample_rate: int, channels: int) -> EncoderProbeResult:
    av = import_av()
    result = EncoderProbeResult(
        name=candidate.name,
        kind="audio",
        sample_fmt=candidate.sample_fmt,
        notes=candidate.notes,
    )
    present, err = _codec_present(candidate.name)
    result.present = present
    if not present:
        result.error = err
        return result

    ctx = None
    try:
        ctx = av.codec.CodecContext.create(candidate.name, "w")
        ctx.sample_rate = sample_rate
        layout = "stereo" if channels == 2 else "mono"
        ctx.layout = layout
        ctx.format = candidate.sample_fmt or "fltp"
        ctx.time_base = Fraction(1, sample_rate)
        ctx.options = dict(candidate.options)
        ctx.open()
        frame_size = ctx.frame_size or 1024
        packets = 0
        for i in range(3):
            frame = _silence_audio_frame(av, ctx.format.name, layout, frame_size, channels)
            frame.sample_rate = sample_rate
            frame.time_base = ctx.time_base
            frame.pts = i * frame_size
            packets += len(ctx.encode(frame))
        packets += len(ctx.encode(None))
        result.packets = packets
        result.usable = packets > 0
        if not result.usable:
            result.error = "encoder produced no packets"
    except Exception as exc:
        result.error = "{}: {}".format(type(exc).__name__, exc)
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
    return result


def _silence_audio_frame(av, fmt: str, layout: str, samples: int, channels: int):
    frame = av.AudioFrame(format=fmt, layout=layout, samples=samples)
    for plane in frame.planes:
        plane.update(bytes(plane.buffer_size))
    return frame


def probe_encoder(
    candidate: EncoderCandidate,
    *,
    width: int = 64,
    height: int = 64,
    fps=24,
    sample_rate: int = 32000,
    channels: int = 2,
    segment_frames: int = 6,
    segments: int = 3,
) -> EncoderProbeResult:
    """Open the encoder for real and push a few frames through it."""
    if candidate.kind == "video":
        return _probe_video(candidate, width, height, fps, segment_frames, segments)
    if candidate.kind == "audio":
        return _probe_audio(candidate, sample_rate, channels)
    raise ValueError("unknown candidate kind: {!r}".format(candidate.kind))


def probe_encoders(
    candidates: Sequence[EncoderCandidate] = H264_ENCODER_CANDIDATES, **kwargs: Any
) -> List[EncoderProbeResult]:
    return [probe_encoder(c, **kwargs) for c in candidates]


def select_encoder(
    candidates: Sequence[EncoderCandidate] = H264_ENCODER_CANDIDATES,
    preferred: Optional[str] = None,
    require_forced_idr: bool = True,
    **kwargs: Any
) -> Tuple[EncoderCandidate, EncoderProbeResult]:
    """Return the first candidate that actually encodes.

    ``preferred`` pins a specific encoder name; if it is not usable the error
    names it explicitly rather than silently falling back.

    With ``require_forced_idr`` (the default for video) an encoder that cannot
    be *shown* to turn a forced ``pict_type=I`` into a keyframe packet is
    rejected: claiming independently decodable fragments on such an encoder
    would be a lie.  Set it to False only for a preview lane that appends
    fragments sequentially and never seeks.
    """
    chain: Sequence[EncoderCandidate] = candidates
    if preferred is not None:
        chain = [c for c in candidates if c.name == preferred]
        if not chain:
            chain = [EncoderCandidate(name=preferred, kind=candidates[0].kind if candidates else "video")]

    failures: List[EncoderProbeResult] = []
    for candidate in chain:
        result = probe_encoder(candidate, **kwargs)
        if not result.usable:
            failures.append(result)
            continue
        if candidate.kind == "video" and require_forced_idr and result.forced_idr_ok is not True:
            failures.append(result)
            continue
        if not result.idr_options_accepted:
            candidate = candidate.without_idr_options()
        return candidate, result

    raise EncoderUnavailable(
        "no usable encoder in chain [{}]{}:\n  {}".format(
            ", ".join(c.name for c in chain),
            " with verified forced IDR" if require_forced_idr else "",
            "\n  ".join(r.describe() for r in failures),
        )
    )
