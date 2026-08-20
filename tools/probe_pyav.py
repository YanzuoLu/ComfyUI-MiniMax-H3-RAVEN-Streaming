#!/usr/bin/env python3
"""M0 probe 1: can PyAV produce an appendable fragmented MP4 from a custom sink?

Falsification targets, in order:

1. PyAV imports and its FFmpeg has an H.264 encoder somewhere in the chain
   ``h264_nvenc -> libx264 -> libopenh264`` (and an AAC encoder).  Forced IDR is
   verified *by behaviour*, because FFmpeg accepts unknown codec options
   silently and a declared ``forced-idr`` flag proves nothing on its own.
2. ``av.open(write_only_object, 'w', format='mp4')`` works with no ``seek``,
   given ``movflags=frag_keyframe+empty_moov+default_base_moof``.
3. The byte stream splits cleanly into an init segment plus fragments, and
   fragments arrive **before** the container is closed (with the delay measured,
   not assumed), for video-only and for video+audio.
4. Concatenating init + all fragments decodes back to the exact frame count,
   and A/V start times agree to within one AAC frame.
5. In ``keyframe`` mode every fragment is *independently* decodable after the
   init segment.  In ``every_frame`` (preview) mode only sequential append is
   claimed, and that is what gets checked.

Every step prints PASS/FAIL.  Exit codes: 0 all passed, 1 a check failed,
2 the environment could not support the probe at all.

Usage::

    python tools/probe_pyav.py
    python tools/probe_pyav.py --comfy-root /path/to/ComfyUI   # use Comfy's PyAV
    python tools/probe_pyav.py --json report.json --frames 68 --segment-frames 17
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from fractions import Fraction
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ENVIRONMENT = 2


def _resolve_comfy_root(explicit: Optional[str]) -> Optional[str]:
    """``--comfy-root`` then COMFYUI_PATH then COMFYUI_UPSTREAM_PATH."""
    candidates = [explicit] + [os.environ.get(v) for v in ENV_VARS]
    for candidate in candidates:
        if not candidate:
            continue
        root = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(root):
            if candidate is explicit:
                raise SystemExit("--comfy-root does not exist: {}".format(root))
            continue
        return root
    return None


def _add_comfy_root(root: Optional[str]) -> Optional[str]:
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


class Report(object):
    def __init__(self) -> None:
        self.failures = 0
        self.skips = 0
        self.checks: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {}
        self._section = "general"

    def section(self, title: str) -> None:
        self._section = title
        print("\n== {} ==".format(title))

    def ok(self, message: str) -> None:
        self.checks.append({"section": self._section, "status": "pass", "message": message})
        print("  PASS  {}".format(message))

    def fail(self, message: str) -> None:
        self.failures += 1
        self.checks.append({"section": self._section, "status": "fail", "message": message})
        print("  FAIL  {}".format(message))

    def skip(self, message: str) -> None:
        self.skips += 1
        self.checks.append({"section": self._section, "status": "skip", "message": message})
        print("  SKIP  {}".format(message))

    def info(self, message: str) -> None:
        print("        {}".format(message))

    def check(self, condition: bool, message: str) -> bool:
        if condition:
            self.ok(message)
        else:
            self.fail(message)
        return bool(condition)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failures": self.failures,
            "skips": self.skips,
            "passed": self.failures == 0,
            "checks": self.checks,
            "data": self.data,
        }


# -- environment -----------------------------------------------------------


def probe_environment(report: Report) -> bool:
    from raven_streaming.media.codecs import EncoderUnavailable, describe_environment

    report.section("environment")
    try:
        info = describe_environment()
    except EncoderUnavailable as exc:
        report.fail(str(exc))
        report.data["environment"] = {"error": str(exc)}
        return False
    except Exception as exc:  # pragma: no cover - defensive, must never crash
        report.fail("environment probe raised {}: {}".format(type(exc).__name__, exc))
        report.data["environment"] = {"error": repr(exc)}
        return False

    report.data["environment"] = info
    if not info.get("pyav_version"):
        report.fail("PyAV is not importable: {}".format(info.get("errors")))
        return False

    report.ok("PyAV {}".format(info["pyav_version"]))
    parts = info.get("pyav_version_tuple") or []
    if parts and parts[0] < 16:
        report.info("NOTE: target is PyAV >= 16, found {}".format(info["pyav_version"]))
    for name, version in sorted((info.get("library_versions") or {}).items()):
        report.info("{:<16} {}".format(name, ".".join(str(v) for v in version)))
    for err in info.get("errors") or ():
        report.info("environment note: {}".format(err))

    # av.formats was removed in PyAV 18; container_format_available copes
    report.check(
        info.get("mp4_muxer") is not False,
        "mp4 muxer available (determined via {})".format(info.get("mp4_muxer_probe")),
    )
    return True


def probe_encoders(report: Report, args):
    from raven_streaming.media.codecs import (
        AAC_ENCODER_CANDIDATES,
        H264_ENCODER_CANDIDATES,
        probe_encoders as run_probe,
    )

    report.section("H.264 encoder chain (h264_nvenc -> libx264 -> libopenh264)")
    video = run_probe(
        H264_ENCODER_CANDIDATES,
        width=args.width,
        height=args.height,
        fps=args.fps,
        segment_frames=min(args.segment_frames, 6),
        segments=3,
    )
    for result in video:
        print("  " + result.describe())
        if result.usable and result.keyframe_frames is not None:
            report.info(
                "  {}: keyframe packets at frames {} (expected {})".format(
                    result.name, result.keyframe_frames, result.expected_keyframe_frames
                )
            )
    report.data["video_encoders"] = [r.to_dict() for r in video]

    usable = [r for r in video if r.usable]
    report.check(bool(usable), "at least one H.264 encoder is usable")
    idr_ok = [r for r in usable if r.forced_idr_ok]
    report.check(
        bool(idr_ok),
        "at least one H.264 encoder has behaviourally verified forced IDR",
    )
    for result in usable:
        if result.forced_idr_ok is False:
            report.info(
                "{} cannot be used for independently decodable fragments: {}".format(
                    result.name, result.forced_idr_error
                )
            )
    if idr_ok:
        report.info("selected for segment mode: {}".format(idr_ok[0].name))

    report.section("AAC encoder chain")
    audio = run_probe(AAC_ENCODER_CANDIDATES, sample_rate=args.sample_rate, channels=2)
    for result in audio:
        print("  " + result.describe())
    report.data["audio_encoders"] = [r.to_dict() for r in audio]
    usable_audio = [r for r in audio if r.usable]
    report.check(bool(usable_audio), "at least one AAC encoder is usable")

    return (idr_ok[0].name if idr_ok else (usable[0].name if usable else None),
            usable_audio[0].name if usable_audio else None)


# -- muxing ----------------------------------------------------------------


def _frame(np, width: int, height: int, i: int):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = (i * 7) % 256
    img[:, :, 1] = (i * 13) % 256
    img[(i * 3) % height, :, 2] = 255
    return img


def _pcm(np, n: int, start: int, sample_rate: int):
    t = (np.arange(n, dtype=np.float32) + start) / float(sample_rate)
    return np.stack(
        [0.2 * np.sin(2 * np.pi * 440.0 * t), 0.2 * np.sin(2 * np.pi * 660.0 * t)], axis=0
    ).astype(np.float32)


def run_mux(report: Report, args, video_encoder, audio_encoder, fragment_mode: str,
            with_audio: bool):
    import numpy as np

    from raven_streaming.media.clock import MediaClock
    from raven_streaming.media.mp4_writer import (
        FragmentedMP4Muxer,
        MuxerConfig,
        concat_stream,
    )

    clock = MediaClock(Fraction(args.fps), args.sample_rate)
    cfg = MuxerConfig(
        width=args.width,
        height=args.height,
        fps=Fraction(args.fps),
        sample_rate=args.sample_rate,
        with_audio=with_audio,
        segment_frames=args.segment_frames,
        fragment_mode=fragment_mode,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder if with_audio else None,
    )

    muxer = FragmentedMP4Muxer(cfg)
    init = None
    fragments = []
    try:
        muxer.open()
        group_frames = clock.sync_period_frames
        group_samples = clock.sync_period_samples
        with muxer:
            for i in range(args.frames):
                muxer.write_video_frame(_frame(np, args.width, args.height, i))
                if with_audio and (i + 1) % group_frames == 0:
                    start = ((i + 1) // group_frames - 1) * group_samples
                    muxer.write_audio(_pcm(np, group_samples, start, args.sample_rate))
                init = init or muxer.take_init_segment()
                fragments.extend(muxer.take_fragments())
        init = init or muxer.take_init_segment()
        fragments.extend(muxer.take_fragments())
        trailer = muxer.take_trailer()
    except Exception as exc:
        report.fail("muxing ({}, audio={}) raised {}: {}".format(
            fragment_mode, with_audio, type(exc).__name__, exc))
        traceback.print_exc()
        return None

    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)
    return {
        "muxer": muxer,
        "report": muxer.report(),
        "init": init,
        "fragments": fragments,
        "trailer": trailer,
        "blob": blob,
        "clock": clock,
        "with_audio": with_audio,
        "fragment_mode": fragment_mode,
    }


def probe_muxer(report: Report, args, video_encoder, audio_encoder, fragment_mode: str,
                with_audio: bool):
    from raven_streaming.media.mp4_boxes import iter_boxes

    label = "{} / {}".format(fragment_mode, "video+audio" if with_audio else "video-only")
    report.section("fragmented MP4 through a write-only sink [{}]".format(label))
    if video_encoder is None:
        report.skip("no H.264 encoder; cannot mux")
        return None

    muxed = run_mux(report, args, video_encoder, audio_encoder, fragment_mode, with_audio)
    if muxed is None:
        return None

    info = muxed["report"]
    init, fragments = muxed["init"], muxed["fragments"]
    report.data.setdefault("mux", {})[label] = info

    report.ok("muxed with video={} audio={} movflags={}".format(
        info["video_encoder"], info["audio_encoder"], info["movflags"]))
    report.check(init is not None, "init segment produced")
    if init:
        types = [b.type for b in iter_boxes(init)]
        report.check(types[:1] == ["ftyp"], "init starts with ftyp (got {})".format(types))
        report.check("moov" in types, "init contains moov")
    report.check(len(fragments) > 1, "multiple fragments produced ({})".format(len(fragments)))
    report.check(
        all("moof" in f.box_types and "mdat" in f.box_types for f in fragments),
        "every fragment is a complete moof+mdat pair",
    )
    report.check(info["idr_violations"] == [], "no segment started on a non-keyframe packet")
    if info["idr_violations"]:
        for violation in info["idr_violations"]:
            report.info(violation["reason"])

    # -- the timing evidence -------------------------------------------------
    report.check(
        info["fragments_before_close"] >= 2,
        "fragments delivered BEFORE close ({} of {})".format(
            info["fragments_before_close"], len(fragments)),
    )
    report.check(
        (info["bytes_before_close"] or 0) > 0,
        "bytes delivered before close ({} of {})".format(
            info["bytes_before_close"], info["total_bytes"]),
    )
    report.info("first_fragment_after_frames  = {}".format(info["first_fragment_after_frames"]))
    report.info("steady_fragment_delay_frames = {}".format(info["steady_fragment_delay_frames"]))
    report.info("close_only_bytes             = {} of {} total".format(
        info["close_only_bytes"], info["total_bytes"]))
    report.info("sink writes                  = {}".format(info["sink_write_calls"]))

    delay = info["steady_fragment_delay_frames"]
    if fragment_mode == "keyframe" and not with_audio and delay is not None:
        report.check(
            delay == args.segment_frames,
            "keyframe mode costs exactly one segment of mux delay ({} frames)".format(delay),
        )
    if fragment_mode == "every_frame" and not with_audio and delay is not None:
        report.check(delay <= 1, "preview mode delivers within <= 1 frame (delay={})".format(delay))

    if with_audio and info["audio_encoder"]:
        report.info("audio priming   = {} samples ({} AAC frame)".format(
            info["audio_priming_samples"], info["audio_frame_size"]))
        report.info("audio padded tail = {} samples".format(info["audio_padded_tail_samples"]))
        skew = info["av_first_pts_skew_samples"]
        report.check(
            skew is not None and skew <= 1024,
            "A/V first-PTS skew at the encoder is {} samples (<= 1024 = {:.1f} ms)".format(
                skew, 1024 * 1000.0 / args.sample_rate),
        )

    if args.out and fragment_mode == "keyframe" and with_audio:
        with open(args.out, "wb") as handle:
            handle.write(muxed["blob"])
        report.info("wrote {} ({} bytes)".format(args.out, len(muxed["blob"])))
    return muxed


# -- decoding --------------------------------------------------------------


def _decode_frames(av, blob: bytes) -> int:
    with av.open(io.BytesIO(blob)) as container:
        return sum(1 for _ in container.decode(video=0))


def _first_packet_is_keyframe(av, blob: bytes) -> bool:
    with av.open(io.BytesIO(blob)) as container:
        for packet in container.demux(video=0):
            if packet.size == 0:
                continue
            return bool(packet.is_keyframe)
    return False


def probe_decode(report: Report, args, muxed):
    import av

    label = "{} / {}".format(muxed["fragment_mode"], "video+audio" if muxed["with_audio"] else "video-only")
    report.section("decode the concatenated stream [{}]".format(label))
    blob = muxed["blob"]
    try:
        with av.open(io.BytesIO(blob)) as container:
            vstreams = [s for s in container.streams if s.type == "video"]
            astreams = [s for s in container.streams if s.type == "audio"]
            report.check(len(vstreams) == 1, "exactly one video stream")
            report.check(
                len(astreams) == (1 if muxed["with_audio"] else 0),
                "audio stream present == {}".format(muxed["with_audio"]),
            )
        frames = _decode_frames(av, blob)
        report.check(frames == args.frames,
                     "decoded {} frames (expected {})".format(frames, args.frames))
        report.info("video duration {:.4f} s at {} fps".format(
            float(Fraction(frames, 1) / Fraction(args.fps)), args.fps))
    except Exception as exc:
        report.fail("video decode raised {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        return

    if not muxed["with_audio"]:
        return

    try:
        with av.open(io.BytesIO(blob)) as container:
            video = container.streams.video[0]
            audio = container.streams.audio[0]
            v_start = Fraction(video.start_time or 0) * Fraction(video.time_base)
            a_start = Fraction(audio.start_time or 0) * Fraction(audio.time_base)
            skew = abs(v_start - a_start) * args.sample_rate
            report.check(
                skew <= 1024,
                "container start_time skew {:.1f} samples (<= 1024)".format(float(skew)),
            )
            report.info("video.start_time={} audio.start_time={}".format(
                video.start_time, audio.start_time))
        with av.open(io.BytesIO(blob)) as container:
            v_pkt = next(p for p in container.demux(video=0) if p.size)
            v_pts = Fraction(v_pkt.pts) * Fraction(v_pkt.time_base)
        with av.open(io.BytesIO(blob)) as container:
            a_pkt = next(p for p in container.demux(audio=0) if p.size)
            a_pts = Fraction(a_pkt.pts) * Fraction(a_pkt.time_base)
        pkt_skew = abs(v_pts - a_pts) * args.sample_rate
        report.check(pkt_skew <= 1024,
                     "first demuxed packet skew {:.1f} samples (<= 1024)".format(float(pkt_skew)))

        with av.open(io.BytesIO(blob)) as container:
            samples = sum(f.samples for f in container.decode(audio=0))
        expected = muxed["clock"].samples_for_frames(args.frames)
        padded = muxed["report"]["audio_padded_tail_samples"]
        slack = 2 * 1024 + padded
        report.check(
            abs(samples - expected) <= slack,
            "decoded {} audio samples (expected ~{}, slack {} = 2 AAC frames + {} pad)".format(
                samples, expected, slack, padded),
        )
        report.info("audio duration {:.4f} s".format(float(Fraction(samples, args.sample_rate))))
    except Exception as exc:
        report.fail("audio checks raised {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()


def probe_independent_fragments(report: Report, muxed):
    """Keyframe mode: each fragment must stand alone after the init segment."""
    import av

    report.section("fragment independence [{}]".format(muxed["fragment_mode"]))
    init, fragments = muxed["init"], muxed["fragments"]
    independent = muxed["report"]["fragments_independently_decodable"]

    bad_keyframe: List[int] = []
    standalone = 0
    for idx, frag in enumerate(fragments):
        blob = init + frag.data
        # NOTE: keyframe inspection and decoding must use separate containers.
        # Consuming the IDR with demux() and then calling decode() on the same
        # container starves the decoder and looks like a muxer bug.
        try:
            if not _first_packet_is_keyframe(av, blob):
                bad_keyframe.append(idx)
            elif _decode_frames(av, blob) > 0:
                standalone += 1
        except Exception:
            bad_keyframe.append(idx)

    report.data.setdefault("independence", {})[muxed["fragment_mode"]] = {
        "fragments": len(fragments),
        "standalone": standalone,
        "claimed_independent": independent,
    }

    if independent:
        report.check(
            standalone == len(fragments) and not bad_keyframe,
            "all {} fragments start on an IDR and decode standalone".format(len(fragments)),
        )
        if bad_keyframe:
            report.info("non-IDR fragments: {}".format(bad_keyframe))
    else:
        report.ok(
            "preview mode: {}/{} fragments happen to stand alone; only sequential "
            "append is claimed".format(standalone, len(fragments))
        )


def probe_prefix_growth(report: Report, muxed):
    """Sequential append must never invalidate the stream (both modes)."""
    import av

    from raven_streaming.media.mp4_writer import concat_stream

    report.section("sequential append [{}]".format(muxed["fragment_mode"]))
    init, fragments = muxed["init"], muxed["fragments"]
    step = max(1, len(fragments) // 24)
    seen = 0
    monotone = True
    for n in range(1, len(fragments) + 1, step):
        blob = concat_stream(init, fragments[:n])
        try:
            count = _decode_frames(av, blob)
        except Exception as exc:
            report.fail("prefix of {} fragments failed to open: {}".format(n, exc))
            return
        if count < seen:
            monotone = False
        seen = count
    final = _decode_frames(av, concat_stream(init, fragments))
    report.check(monotone, "frame count is monotonically non-decreasing over prefixes")
    report.check(final == muxed["report"]["frames_written"],
                 "full concatenation decodes {} frames".format(final))


# -- main ------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comfy-root", default=None,
                        help="ComfyUI checkout to put on sys.path (for its PyAV); "
                             "falls back to COMFYUI_PATH / COMFYUI_UPSTREAM_PATH")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--frames", type=int, default=68,
                        help="frames to mux (68 = 4 video VAE chunks of 17)")
    parser.add_argument("--segment-frames", type=int, default=17,
                        help="frames per segment; the first is forced to IDR")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-preview-mode", action="store_true",
                        help="skip the frag_every_frame (preview) measurements")
    parser.add_argument("--out", default=None, help="also write the muxed bytes here")
    parser.add_argument("--json", default=None, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    comfy_root = _add_comfy_root(_resolve_comfy_root(args.comfy_root))

    report = Report()
    report.data["args"] = vars(args)
    report.data["interpreter"] = sys.executable
    report.data["comfy_root"] = comfy_root
    print("probe_pyav: fragmented MP4 / custom sink falsification")
    print("interpreter: {}".format(sys.executable))
    if comfy_root:
        print("comfy root : {}".format(comfy_root))

    def finish(code: int) -> int:
        payload = report.to_dict()
        payload["exit_code"] = code
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(payload, handle, indent=2, default=str)
            print("\nwrote JSON report to {}".format(args.json))
        print("\nRESULT: {} ({} failures, {} skips)".format(
            "PASSED" if code == EXIT_OK else "FAILED", report.failures, report.skips))
        return code

    try:
        import numpy  # noqa: F401
    except Exception as exc:
        report.section("environment")
        report.fail("numpy is required: {}".format(exc))
        return finish(EXIT_ENVIRONMENT)

    if not probe_environment(report):
        return finish(EXIT_ENVIRONMENT)

    video_encoder, audio_encoder = probe_encoders(report, args)
    if video_encoder is None:
        return finish(EXIT_ENVIRONMENT)

    modes = ["keyframe"] if args.no_preview_mode else ["keyframe", "every_frame"]
    audio_variants = [False] if args.no_audio else [False, True]

    for mode in modes:
        for with_audio in audio_variants:
            muxed = probe_muxer(report, args, video_encoder, audio_encoder, mode, with_audio)
            if muxed is None:
                continue
            probe_decode(report, args, muxed)
            if not with_audio:
                probe_independent_fragments(report, muxed)
                probe_prefix_growth(report, muxed)

    _summarise(report)
    return finish(EXIT_OK if report.failures == 0 else EXIT_FAILED)


def _summarise(report: Report) -> None:
    report.section("summary: measured streaming contract")
    mux = report.data.get("mux") or {}
    if not mux:
        report.info("nothing was muxed")
        return
    print("  {:<34} {:>10} {:>10} {:>12} {:>12}".format(
        "configuration", "1st frag", "delay", "pre-close", "close-only"))
    for label, info in mux.items():
        print("  {:<34} {:>10} {:>10} {:>12} {:>12}".format(
            label,
            str(info["first_fragment_after_frames"]),
            str(info["steady_fragment_delay_frames"]),
            "{}B/{}f".format(info["bytes_before_close"], info["fragments_before_close"]),
            str(info["close_only_bytes"]),
        ))
    encoders = report.data.get("video_encoders") or []
    usable = [e for e in encoders if e["usable"]]
    if usable:
        report.info("video encoder: {} (forced IDR via {}, verified={})".format(
            usable[0]["name"], usable[0]["idr_control"], usable[0]["forced_idr_ok"]))


if __name__ == "__main__":
    raise SystemExit(main())
