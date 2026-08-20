"""Falsification tests for the fragmented MP4 muxer over a PyAV custom sink.

These require PyAV *and* a usable H.264 encoder.  They skip (loudly) otherwise;
run ``tools/probe_pyav.py`` to see exactly what the interpreter has.

A note on a trap this suite used to fall into: ``next(container.demux(video=0))``
*consumes* the first packet.  Calling ``container.decode()`` on the same
container afterwards starts at the second packet, so the decoder never receives
the IDR and returns zero frames - which looks exactly like a broken muxer but
is a broken test.  Keyframe inspection and decoding therefore always happen on
separate containers (or by decoding demuxed packets by hand).
"""

from __future__ import annotations

import io
import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.media.mp4_boxes import iter_boxes  # noqa: E402

av = pytest.importorskip("av", reason="PyAV not installed in this interpreter")
np = pytest.importorskip("numpy")

from raven_streaming.media.codecs import (  # noqa: E402
    H264_ENCODER_CANDIDATES,
    EncoderUnavailable,
    codec_available,
    container_format_available,
    describe_environment,
    packet_frame_index,
    probe_forced_idr,
    select_encoder,
)
from raven_streaming.media.mp4_writer import (  # noqa: E402
    DEFAULT_MOVFLAGS,
    PREVIEW_CONTAINER_OPTIONS,
    PREVIEW_MOVFLAGS,
    SEGMENT_MOVFLAGS,
    FragmentedMP4Muxer,
    MuxerConfig,
    WriteOnlySink,
    concat_stream,
)

WIDTH, HEIGHT, FPS = 64, 64, 24
SEGMENT_FRAMES = 17  # one video VAE chunk
SAMPLE_RATE = 32000


# -- helpers ---------------------------------------------------------------


def _require_h264(require_forced_idr: bool = True):
    try:
        candidate, result = select_encoder(
            H264_ENCODER_CANDIDATES,
            require_forced_idr=require_forced_idr,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
        )
    except EncoderUnavailable as exc:
        pytest.skip("no usable H.264 encoder: {}".format(exc))
    return candidate, result


def _frame(i: int):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :, 0] = (i * 11) % 256
    img[:, :, 1] = (i * 29) % 256
    img[i % HEIGHT, :, 2] = 255  # a moving line so frames differ
    return img


def _pcm(n: int, start: int):
    t = (np.arange(n, dtype=np.float32) + start) / float(SAMPLE_RATE)
    left = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.2 * np.sin(2 * np.pi * 660.0 * t)
    return np.stack([left, right], axis=0).astype(np.float32)


def _decode_frames(blob: bytes) -> int:
    """Decode a blob on a *fresh* container (never share with a demux pass)."""
    with av.open(io.BytesIO(blob)) as container:
        return sum(1 for _ in container.decode(video=0))


def _first_packet_is_keyframe(blob: bytes) -> bool:
    with av.open(io.BytesIO(blob)) as container:
        for packet in container.demux(video=0):
            if packet.size == 0:
                continue
            return bool(packet.is_keyframe)
    return False


def _decode_demuxed_packets(blob: bytes):
    """Demux and decode by hand, so keyframe flags and frames come from one pass."""
    frames = 0
    first_key = None
    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            if first_key is None:
                first_key = bool(packet.is_keyframe)
            frames += len(packet.decode())
        # flush
        for frame in stream.codec_context.decode(None):
            frames += 1
    return first_key, frames


def _run_muxer(frames: int, with_audio: bool, fragment_mode: str = "keyframe", **kw):
    cfg = MuxerConfig(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        sample_rate=SAMPLE_RATE,
        with_audio=with_audio,
        segment_frames=SEGMENT_FRAMES,
        fragment_mode=fragment_mode,
        **kw
    )
    muxer = FragmentedMP4Muxer(cfg)
    try:
        muxer.open()
    except EncoderUnavailable as exc:
        pytest.skip("encoder unavailable: {}".format(exc))

    init = None
    fragments = []
    # audio is fed in exact 3-frame sync groups: 3 frames == 4000 samples
    group_frames, group_samples = 3, 4000
    with muxer:
        for i in range(frames):
            muxer.write_video_frame(_frame(i))
            if with_audio and (i + 1) % group_frames == 0:
                start = ((i + 1) // group_frames - 1) * group_samples
                muxer.write_audio(_pcm(group_samples, start))
            init = init or muxer.take_init_segment()
            fragments.extend(muxer.take_fragments())
    init = init or muxer.take_init_segment()
    fragments.extend(muxer.take_fragments())
    trailer = muxer.take_trailer()
    return muxer, init, fragments, trailer


# -- environment -----------------------------------------------------------


def test_environment_report_never_crashes():
    """PyAV 18 removed av.formats; the report must degrade, not explode."""
    info = describe_environment()
    assert info["pyav_version"]
    assert info["errors"] == []
    assert info["mp4_muxer"] is True
    assert info["mp4_muxer_probe"] != "undetermined"


def test_container_format_probe_is_version_agnostic():
    available, how = container_format_available("mp4")
    assert available is True
    assert how in (
        "av.format.ContainerFormat",
        "av.formats_available",
        "av.format.formats_available",
        "av.formats.available",
    )
    missing, _ = container_format_available("definitely-not-a-format")
    assert missing in (False, None)


def test_codec_availability_reports_reasons():
    present, err = codec_available("libx264")
    if present:
        assert err is None
    missing, err = codec_available("definitely_not_a_codec")
    assert missing is False and err


def test_write_only_sink_has_no_seek():
    """PyAV keys off the absence of seek to pick the non-seekable code path."""
    sink = WriteOnlySink(lambda data: None)
    assert not hasattr(sink, "seek")
    assert not hasattr(sink, "tell")
    assert sink.write(b"abcd") == 4
    assert sink.bytes_written == 4


def test_movflags_sets_are_the_streaming_ones():
    for flags in (SEGMENT_MOVFLAGS, PREVIEW_MOVFLAGS):
        assert "empty_moov" in flags
        assert "default_base_moof" in flags
    assert "frag_keyframe" in SEGMENT_MOVFLAGS
    assert "frag_every_frame" in PREVIEW_MOVFLAGS
    assert DEFAULT_MOVFLAGS == SEGMENT_MOVFLAGS


# -- forced IDR ------------------------------------------------------------


def test_selected_encoder_has_behaviourally_verified_forced_idr():
    candidate, result = _require_h264()
    assert result.forced_idr_ok is True, result.forced_idr_error
    assert result.keyframe_frames is not None
    assert set(result.expected_keyframe_frames).issubset(set(result.keyframe_frames))


def test_forced_idr_probe_maps_packets_by_pts_not_by_order():
    candidate, _ = _require_h264()
    out = probe_forced_idr(candidate, width=WIDTH, height=HEIGHT, fps=FPS,
                           segment_frames=5, segments=4)
    assert out["ok"], out["error"]
    assert out["expected"] == [0, 5, 10, 15]
    assert set(out["expected"]).issubset(set(out["keyframe_frames"]))


def test_packet_frame_index_handles_either_time_base():
    class FakePacket(object):
        def __init__(self, pts, time_base):
            self.pts = pts
            self.time_base = time_base

    # encoder time_base (1/fps): pts is the frame index
    assert packet_frame_index(FakePacket(7, Fraction(1, 24)), 24) == 7
    # media-clock time_base (1/96000): pts is in ticks
    assert packet_frame_index(FakePacket(7 * 4000, Fraction(1, 96000)), 24) == 7
    assert packet_frame_index(FakePacket(None, Fraction(1, 24)), 24) is None
    assert packet_frame_index(FakePacket(0, None), 24) is None


def test_muxer_records_a_keyframe_for_every_segment_start():
    muxer, _, _, _ = _run_muxer(frames=SEGMENT_FRAMES * 3, with_audio=False)
    report = muxer.report()
    assert report["idr_violations"] == []
    assert report["forced_idr_frames"] == [0, 17, 34]
    assert report["segment_first_packet_is_keyframe"] == {0: True, 1: True, 2: True}


def test_strict_idr_raises_loudly_on_a_non_keyframe_segment_start():
    """Guard the guard: a violation must abort, not be silently recorded."""
    from raven_streaming.media.mp4_writer import IDRViolation

    muxer, _, _, _ = _run_muxer(frames=SEGMENT_FRAMES, with_audio=False)
    assert muxer.config.strict_idr is True

    class FakePacket(object):
        pts = 0
        time_base = Fraction(1, 24)
        is_keyframe = False

    fresh = FragmentedMP4Muxer(
        MuxerConfig(width=WIDTH, height=HEIGHT, fps=FPS, with_audio=False,
                    segment_frames=SEGMENT_FRAMES)
    )
    with pytest.raises(RuntimeError, match="not a keyframe"):
        fresh._track_video_packet(FakePacket())
    assert isinstance(fresh.idr_violations[0], IDRViolation)


# -- decodability ----------------------------------------------------------


@pytest.mark.parametrize("with_audio", [False, True])
def test_fragmented_mp4_roundtrips_through_pyav(with_audio):
    _require_h264()
    frames = 51
    muxer, init, fragments, trailer = _run_muxer(frames, with_audio)

    assert init, "no init segment was produced"
    types = [b.type for b in iter_boxes(init)]
    assert types[0] == "ftyp" and "moov" in types
    assert fragments, "no media fragments were produced"
    for frag in fragments:
        assert "moof" in frag.box_types
        assert "mdat" in frag.box_types

    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)
    with av.open(io.BytesIO(blob)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        astreams = [s for s in container.streams if s.type == "audio"]
        assert len(vstreams) == 1
        assert len(astreams) == (1 if with_audio else 0)
        assert vstreams[0].codec_context.name == "h264"
    assert _decode_frames(blob) == frames


def test_every_fragment_starts_on_an_idr_and_decodes_standalone():
    """Keyframe/decode checks use separate containers - see the module docstring."""
    _require_h264()
    frames = SEGMENT_FRAMES * 4
    muxer, init, fragments, _ = _run_muxer(frames, with_audio=False)

    assert len(fragments) >= 2, "expected multiple fragments, got {}".format(len(fragments))
    assert muxer.config.fragments_are_independently_decodable

    for idx, frag in enumerate(fragments):
        blob = init + frag.data
        assert _first_packet_is_keyframe(blob), "fragment {} does not start on a keyframe".format(idx)
        assert _decode_frames(blob) > 0, "fragment {} decoded to nothing".format(idx)
        # and again in a single demux+decode pass, to prove the two agree
        first_key, decoded = _decode_demuxed_packets(blob)
        assert first_key is True
        assert decoded > 0


def test_demux_then_decode_on_one_container_is_the_known_trap():
    """Documents why the old test was red: it is a test bug, not a muxer bug."""
    _require_h264()
    _, init, fragments, _ = _run_muxer(SEGMENT_FRAMES * 2, with_audio=False)
    blob = init + fragments[0].data

    with av.open(io.BytesIO(blob)) as container:
        first = next(container.demux(video=0))
        assert first.is_keyframe
        starved = sum(1 for _ in container.decode(video=0))
    # the IDR was consumed by demux(), so the decoder never got a key packet
    assert starved < _decode_frames(blob)
    assert _decode_frames(blob) > 0


def test_incremental_append_equals_the_whole_file():
    """init + fragments[:n] must always be a valid decodable prefix."""
    _require_h264()
    frames = SEGMENT_FRAMES * 3
    _, init, fragments, _ = _run_muxer(frames, with_audio=False)

    seen = 0
    for n in range(1, len(fragments) + 1):
        count = _decode_frames(concat_stream(init, fragments[:n]))
        assert count >= seen, "prefix of {} fragments lost frames".format(n)
        seen = count
    assert seen == frames


# -- pre-close timing evidence ---------------------------------------------


@pytest.mark.parametrize("with_audio", [False, True])
def test_fragments_are_available_before_close(with_audio):
    """The whole point of streaming: bytes must flow before the container closes."""
    _require_h264()
    frames = SEGMENT_FRAMES * 3
    muxer, init, fragments, _ = _run_muxer(frames, with_audio)
    report = muxer.report()

    assert report["fragments_before_close"] >= 2, (
        "only {} fragments before close".format(report["fragments_before_close"]))
    assert report["bytes_before_close"] > 0
    assert report["first_fragment_after_frames"] is not None
    # a fragment must land well before the last frame is written
    assert report["first_fragment_after_frames"] < frames
    # close writes only the tail (last fragment + trailer), not the bulk
    assert report["close_only_bytes"] < report["bytes_before_close"]


def test_keyframe_mode_costs_one_segment_of_mux_delay():
    """Measured contract for frag_keyframe, video-only (no audio interleave)."""
    _require_h264()
    frames = SEGMENT_FRAMES * 4
    muxer, _, _, _ = _run_muxer(frames, with_audio=False)
    report = muxer.report()

    # fragment N is written when the first frame of segment N+1 is muxed
    assert report["first_fragment_after_frames"] == SEGMENT_FRAMES
    assert report["steady_fragment_delay_frames"] == SEGMENT_FRAMES
    pre = [f["frames_written"] for f in report["fragments"] if not f["after_close"]]
    assert pre == [SEGMENT_FRAMES * (i + 1) for i in range(len(pre))]


def test_audio_interleave_adds_at_most_one_audio_group_of_delay():
    """With audio the muxer must also have PCM for the segment before cutting."""
    _require_h264()
    frames = SEGMENT_FRAMES * 4
    muxer, _, _, _ = _run_muxer(frames, with_audio=True)
    report = muxer.report()

    first = report["first_fragment_after_frames"]
    assert first is not None
    assert SEGMENT_FRAMES <= first <= SEGMENT_FRAMES + 6, (
        "audio interleave delay out of contract: {}".format(first))
    assert report["steady_fragment_delay_frames"] <= SEGMENT_FRAMES + 6


def test_preview_mode_delivers_a_fragment_within_one_frame():
    """frag_every_frame is the low-latency preview lane."""
    _require_h264()
    frames = SEGMENT_FRAMES * 2
    muxer, init, fragments, _ = _run_muxer(frames, with_audio=False,
                                           fragment_mode="every_frame")
    report = muxer.report()

    assert report["movflags"] == PREVIEW_MOVFLAGS
    assert report["first_fragment_after_frames"] <= 1
    assert report["steady_fragment_delay_frames"] == 1
    assert len(fragments) >= frames - 1
    # sequential append still reproduces the whole video exactly
    assert _decode_frames(concat_stream(init, fragments)) == frames


def test_preview_mode_with_audio_decodes_every_frame():
    """Regression: one-sample fragments + audio used to drop the last frame."""
    _require_h264()
    frames = SEGMENT_FRAMES * 4
    muxer, init, fragments, trailer = _run_muxer(frames, with_audio=True,
                                                 fragment_mode="every_frame")
    assert muxer.report()["container_options"]["min_frag_duration"] == "1"
    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)
    assert _decode_frames(blob) == frames

    # ... and no two video packets may share a pts
    with av.open(io.BytesIO(blob)) as container:
        ptss = [p.pts for p in container.demux(container.streams.video[0]) if p.size]
    assert len(ptss) == frames
    assert len(set(ptss)) == frames, "duplicate video pts: {}".format(
        sorted(p for p in set(ptss) if ptss.count(p) > 1))


def test_preview_workaround_is_load_bearing_and_documented():
    """Guard the guard: without min_frag_duration the bug is still there.

    If a future FFmpeg fixes this, the test flips to green-with-a-note rather
    than silently letting the workaround rot.
    """
    _require_h264()
    frames = SEGMENT_FRAMES * 4
    assert PREVIEW_CONTAINER_OPTIONS == {"min_frag_duration": "1"}

    muxer, init, fragments, trailer = _run_muxer(
        frames, with_audio=True, fragment_mode="every_frame", apply_mode_options=False
    )
    assert "min_frag_duration" not in muxer.report()["container_options"]
    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)
    without = _decode_frames(blob)

    muxer2, init2, fragments2, trailer2 = _run_muxer(
        frames, with_audio=True, fragment_mode="every_frame"
    )
    blob2 = concat_stream(init2, fragments2) + b"".join(t.data for t in trailer2)
    assert _decode_frames(blob2) == frames
    if without == frames:
        pytest.skip("this FFmpeg no longer needs min_frag_duration; workaround is inert")
    assert without < frames


def test_preview_mode_does_not_promise_independent_fragments():
    """Only sequential append is claimed; assert the honest, weaker contract."""
    _require_h264()
    frames = SEGMENT_FRAMES * 2
    muxer, init, fragments, _ = _run_muxer(frames, with_audio=False,
                                           fragment_mode="every_frame")
    assert muxer.config.fragments_are_independently_decodable is False

    standalone = sum(1 for f in fragments if _decode_frames(init + f.data) > 0)
    # only the IDR-carrying fragments stand alone - which is exactly why the
    # preview lane must never be advertised as random-access
    assert standalone < len(fragments)

    # but every growing prefix decodes, and never goes backwards
    seen = 0
    for n in range(1, len(fragments) + 1, 4):
        count = _decode_frames(concat_stream(init, fragments[:n]))
        assert count >= seen
        seen = count
    assert _decode_frames(concat_stream(init, fragments)) == frames


def test_preview_mode_segment_starts_are_still_idr():
    _require_h264()
    muxer, _, _, _ = _run_muxer(SEGMENT_FRAMES * 3, with_audio=False,
                                fragment_mode="every_frame")
    report = muxer.report()
    assert report["idr_violations"] == []
    assert all(report["segment_first_packet_is_keyframe"].values())


# -- AAC timing ------------------------------------------------------------


def test_aac_priming_and_first_pts_are_accounted_for():
    _require_h264()
    frames = SEGMENT_FRAMES * 3
    muxer, init, fragments, trailer = _run_muxer(frames, with_audio=True)
    report = muxer.report()
    if report["audio_encoder"] is None:
        pytest.skip("no AAC encoder")

    frame_size = report["audio_frame_size"]
    assert frame_size > 0
    # AAC primes with one frame of encoder delay: a negative first pts
    assert report["first_audio_pts_samples"] is not None
    assert report["audio_priming_samples"] == frame_size == 1024
    # A/V first-PTS skew at the encoder must not exceed one AAC frame
    skew = report["av_first_pts_skew_samples"]
    assert skew is not None
    assert skew <= 1024, "A/V first PTS skew {} samples > 1024".format(skew)
    assert skew / float(SAMPLE_RATE) <= 1024.0 / 32000.0
    # the zero-padded final AAC frame is recorded, not hidden
    assert 0 <= report["audio_padded_tail_samples"] < frame_size


def test_container_start_times_are_aligned_after_muxing():
    """What a player sees: the muxer normalises priming away to start_time 0."""
    _require_h264()
    frames = SEGMENT_FRAMES * 3
    muxer, init, fragments, trailer = _run_muxer(frames, with_audio=True)
    if muxer.report()["audio_encoder"] is None:
        pytest.skip("no AAC encoder")
    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)

    with av.open(io.BytesIO(blob)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        v_start = Fraction(video.start_time or 0) * Fraction(video.time_base)
        a_start = Fraction(audio.start_time or 0) * Fraction(audio.time_base)
        skew_samples = abs(v_start - a_start) * SAMPLE_RATE
        assert skew_samples <= 1024, "container start_time skew {} samples".format(
            float(skew_samples))

    # and the first *demuxed* packet of each lane, on separate containers
    with av.open(io.BytesIO(blob)) as container:
        v_pkt = next(p for p in container.demux(video=0) if p.size)
        v_pts = Fraction(v_pkt.pts) * Fraction(v_pkt.time_base)
    with av.open(io.BytesIO(blob)) as container:
        a_pkt = next(p for p in container.demux(audio=0) if p.size)
        a_pts = Fraction(a_pkt.pts) * Fraction(a_pkt.time_base)
    assert abs(v_pts - a_pts) * SAMPLE_RATE <= 1024


def test_audio_duration_matches_the_video_on_the_tick_grid():
    _require_h264()
    frames = SEGMENT_FRAMES * 3
    muxer, init, fragments, trailer = _run_muxer(frames, with_audio=True)
    report = muxer.report()
    if report["audio_encoder"] is None:
        pytest.skip("no AAC encoder")
    blob = concat_stream(init, fragments) + b"".join(t.data for t in trailer)

    with av.open(io.BytesIO(blob)) as container:
        decoded = sum(f.samples for f in container.decode(audio=0))
    expected = muxer.clock.samples_for_frames(frames)
    # decoder returns priming + padded tail on top of the fed samples
    assert abs(decoded - expected) <= 2 * 1024 + report["audio_padded_tail_samples"]


# -- config ----------------------------------------------------------------


def test_odd_dimensions_are_rejected_early():
    with pytest.raises(ValueError):
        MuxerConfig(width=65, height=64)


def test_unknown_fragment_mode_is_rejected():
    with pytest.raises(ValueError, match="fragment_mode"):
        MuxerConfig(width=64, height=64, fragment_mode="whenever")


def test_preview_mode_relaxes_the_forced_idr_requirement_only_there():
    keyframe = MuxerConfig(width=64, height=64, fragment_mode="keyframe")
    preview = MuxerConfig(width=64, height=64, fragment_mode="every_frame")
    assert keyframe.require_forced_idr is True
    assert preview.require_forced_idr is False
    assert keyframe.fragments_are_independently_decodable is True
    assert preview.fragments_are_independently_decodable is False


def test_muxer_reports_which_encoders_it_picked():
    candidate, _ = _require_h264()
    cfg = MuxerConfig(width=WIDTH, height=HEIGHT, fps=FPS, with_audio=False)
    muxer = FragmentedMP4Muxer(cfg)
    try:
        muxer.open()
    except EncoderUnavailable as exc:
        pytest.skip("encoder unavailable: {}".format(exc))
    with muxer:
        assert muxer.video_encoder_name == candidate.name
        assert muxer.audio_encoder_name is None
