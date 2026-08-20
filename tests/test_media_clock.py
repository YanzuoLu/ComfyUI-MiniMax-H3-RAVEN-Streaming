"""Exactness tests for the shared 24 fps / 32 kHz rational media clock."""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.media.clock import (  # noqa: E402
    AUDIO_SAMPLES_PER_LATENT,
    RAVEN_CLOCK,
    VIDEO_FRAMES_PER_CHUNK,
    MediaClock,
    StreamCursor,
)


def test_raven_grid_is_the_expected_integer_grid():
    clock = RAVEN_CLOCK
    assert clock.ticks_per_second == 96000  # lcm(24, 32000)
    assert clock.ticks_per_frame == 4000
    assert clock.ticks_per_sample == 3
    assert clock.time_base == Fraction(1, 96000)


def test_frame_and_sample_conversions_are_exact_integers():
    clock = RAVEN_CLOCK
    for frames in range(0, 500):
        ticks = clock.frames_to_ticks(frames)
        assert isinstance(ticks, int)
        assert clock.ticks_to_frames_floor(ticks) == frames
        assert clock.frames_to_seconds(frames) == Fraction(frames, 24)
    for samples in range(0, 5000, 7):
        ticks = clock.samples_to_ticks(samples)
        assert clock.ticks_to_samples_floor(ticks) == samples


def test_one_frame_is_not_a_whole_number_of_samples():
    """The whole point of the tick grid: 24 fps and 32 kHz do not align per frame."""
    clock = RAVEN_CLOCK
    assert not clock.is_frame_sample_aligned(1)
    assert not clock.is_frame_sample_aligned(2)
    assert clock.is_frame_sample_aligned(3)
    assert clock.sync_period_frames == 3
    assert clock.sync_period_samples == 4000
    assert clock.sync_period_ticks == 12000


def test_no_drift_accumulates_over_a_long_stream():
    """Float 4000/3 would drift; integer ticks must not."""
    clock = RAVEN_CLOCK
    cursor = StreamCursor(clock)
    # one hour of 3-frame sync groups
    groups = 24 * 60 * 60 // 3
    for _ in range(groups):
        cursor.advance_frames(3)
        cursor.advance_samples(4000)
    assert cursor.drift_ticks == 0
    assert cursor.frames == 24 * 60 * 60
    assert cursor.samples == 32000 * 60 * 60


def test_drift_is_reported_exactly_when_lanes_are_uneven():
    clock = RAVEN_CLOCK
    cursor = StreamCursor(clock)
    cursor.advance_frames(1)  # 4000 ticks
    cursor.advance_samples(1000)  # 3000 ticks
    assert cursor.drift_ticks == 1000
    assert cursor.drift_seconds == Fraction(1000, 96000)


def test_samples_for_frames_matches_chunk_geometry():
    clock = RAVEN_CLOCK
    # 17 frames is not a whole number of samples (17 * 4000 / 3)
    assert clock.samples_for_frames(17) == 22666
    assert not clock.is_frame_sample_aligned(17)
    # 3 chunks (51 frames) is
    assert clock.is_frame_sample_aligned(51)
    assert clock.samples_for_frames(51) == 68000


def test_chunk_alignment_period():
    align = RAVEN_CLOCK.chunk_alignment(VIDEO_FRAMES_PER_CHUNK, AUDIO_SAMPLES_PER_LATENT)
    assert align.video_chunks == 3
    assert align.video_frames == 51
    assert align.audio_latents == 85
    assert align.audio_samples == 68000
    assert align.seconds == Fraction(51, 24) == Fraction(68000, 32000)


def test_pts_helpers_are_monotonic_and_exact():
    cursor = StreamCursor(RAVEN_CLOCK)
    assert cursor.frame_pts(0) == 0
    assert cursor.frame_pts(1) == 4000
    assert cursor.frame_pts(10_000) == 40_000_000
    assert cursor.sample_pts(32000) == 96000


@pytest.mark.parametrize(
    "fps,rate,tps,tpf,tps_sample",
    [
        (24, 32000, 96000, 4000, 3),
        (30, 48000, 48000, 1600, 1),
        (25, 44100, 44100, 1764, 1),
        (Fraction(24000, 1001), 48000, 48000, 2002, 1),
    ],
)
def test_grid_generalizes(fps, rate, tps, tpf, tps_sample):
    clock = MediaClock(fps, rate)
    assert clock.ticks_per_second == tps
    assert clock.ticks_per_frame == tpf
    assert clock.ticks_per_sample == tps_sample
    assert clock.frames_to_seconds(1) == Fraction(1, 1) / Fraction(fps)


def test_rejects_nonsense():
    with pytest.raises(ValueError):
        MediaClock(0, 32000)
    with pytest.raises(ValueError):
        MediaClock(24, 0)
    with pytest.raises(ValueError):
        StreamCursor(RAVEN_CLOCK).advance_frames(-1)
