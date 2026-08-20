"""Shared rational media clock for the RAVEN streaming media lane.

Everything in the media lane (video frames out of the video VAE, PCM out of the
audio VAE, MP4 timestamps) is scheduled on a single *integer* tick grid so that
no floating point drift can ever accumulate over a long stream.

For the RAVEN defaults (24 fps video, 32000 Hz audio) the grid is::

    ticks per second   = lcm(24, 32000) = 96000
    ticks per frame    = 96000 / 24     = 4000
    ticks per sample   = 96000 / 32000  = 3

Note that a single video frame is *not* an integral number of audio samples
(4000/3 = 1333.33 samples), so exact A/V alignment only happens every 3 frames
(= 4000 samples = 1/8 s).  All the helpers below expose that structure
explicitly instead of hiding it behind a float.

This module has no third-party dependencies on purpose: it is pure integer /
:class:`fractions.Fraction` arithmetic and is importable without torch or PyAV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, Union

__all__ = [
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_AUDIO_SAMPLE_RATE",
    "VIDEO_LATENTS_PER_CHUNK",
    "VIDEO_FRAMES_PER_CHUNK",
    "AUDIO_SAMPLES_PER_LATENT",
    "MediaClock",
    "AVChunkAlignment",
    "StreamCursor",
    "RAVEN_CLOCK",
]

DEFAULT_VIDEO_FPS = Fraction(24, 1)
DEFAULT_AUDIO_SAMPLE_RATE = 32000

#: Video VAE temporal geometry: 5 latents in -> 17 frames out (see video_stream).
VIDEO_LATENTS_PER_CHUNK = 5
VIDEO_FRAMES_PER_CHUNK = 17

#: Audio VAE temporal geometry: 1 latent frame -> 800 samples @ 32 kHz (40 Hz).
AUDIO_SAMPLES_PER_LATENT = 800

FpsLike = Union[int, float, Fraction, Tuple[int, int]]


def _as_fraction(fps: FpsLike) -> Fraction:
    if isinstance(fps, Fraction):
        value = fps
    elif isinstance(fps, tuple):
        if len(fps) != 2:
            raise ValueError("fps tuple must be (numerator, denominator)")
        value = Fraction(int(fps[0]), int(fps[1]))
    elif isinstance(fps, int):
        value = Fraction(fps, 1)
    elif isinstance(fps, float):
        # limit_denominator keeps 23.976 -> 24000/1001 style rates exact-ish
        value = Fraction(fps).limit_denominator(1001000)
    else:
        raise TypeError("unsupported fps type: {!r}".format(type(fps)))
    if value <= 0:
        raise ValueError("fps must be positive")
    return value


def _lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


@dataclass(frozen=True)
class AVChunkAlignment:
    """Smallest repeating period where video *chunks* and audio latents align.

    A video VAE chunk is 17 frames and an audio latent is 800 samples; neither
    is a multiple of the other on the tick grid, so a scheduler that wants to
    interleave whole chunks needs the least common period.
    """

    ticks: int
    ticks_per_second: int
    video_chunks: int
    video_frames: int
    audio_latents: int
    audio_samples: int

    @property
    def seconds(self) -> Fraction:
        return Fraction(self.ticks, self.ticks_per_second)


@dataclass(frozen=True)
class MediaClock:
    """Exact integer clock shared by the video and audio lanes."""

    fps: Fraction = DEFAULT_VIDEO_FPS
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "fps", _as_fraction(self.fps))
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    # -- grid ---------------------------------------------------------------

    @property
    def ticks_per_second(self) -> int:
        """Common timebase denominator: integral for both frames and samples."""
        return _lcm(self.fps.numerator, self.sample_rate)

    @property
    def ticks_per_frame(self) -> int:
        tps = self.ticks_per_second
        num, den = self.fps.numerator, self.fps.denominator
        assert (tps * den) % num == 0
        return tps * den // num

    @property
    def ticks_per_sample(self) -> int:
        tps = self.ticks_per_second
        assert tps % self.sample_rate == 0
        return tps // self.sample_rate

    @property
    def time_base(self) -> Fraction:
        """Timebase as a Fraction (seconds per tick), for PyAV stream setup."""
        return Fraction(1, self.ticks_per_second)

    # -- conversions --------------------------------------------------------

    def frames_to_ticks(self, frames: int) -> int:
        return int(frames) * self.ticks_per_frame

    def samples_to_ticks(self, samples: int) -> int:
        return int(samples) * self.ticks_per_sample

    def ticks_to_frames_floor(self, ticks: int) -> int:
        return int(ticks) // self.ticks_per_frame

    def ticks_to_frames_ceil(self, ticks: int) -> int:
        return -((-int(ticks)) // self.ticks_per_frame)

    def ticks_to_samples_floor(self, ticks: int) -> int:
        return int(ticks) // self.ticks_per_sample

    def ticks_to_samples_ceil(self, ticks: int) -> int:
        return -((-int(ticks)) // self.ticks_per_sample)

    def ticks_to_seconds(self, ticks: int) -> Fraction:
        return Fraction(int(ticks), self.ticks_per_second)

    def frames_to_seconds(self, frames: int) -> Fraction:
        return Fraction(int(frames), 1) / self.fps

    def samples_to_seconds(self, samples: int) -> Fraction:
        return Fraction(int(samples), self.sample_rate)

    def samples_for_frames(self, frames: int) -> int:
        """Audio samples covering exactly ``frames`` video frames (floored).

        Exact whenever ``frames`` is a multiple of :attr:`sync_period_frames`.
        """
        return self.ticks_to_samples_floor(self.frames_to_ticks(frames))

    def frames_for_samples(self, samples: int) -> int:
        return self.ticks_to_frames_floor(self.samples_to_ticks(samples))

    def is_frame_sample_aligned(self, frames: int) -> bool:
        return self.frames_to_ticks(frames) % self.ticks_per_sample == 0

    # -- periods ------------------------------------------------------------

    @property
    def sync_period_ticks(self) -> int:
        """Smallest tick count that is a whole number of frames *and* samples."""
        return _lcm(self.ticks_per_frame, self.ticks_per_sample)

    @property
    def sync_period_frames(self) -> int:
        return self.sync_period_ticks // self.ticks_per_frame

    @property
    def sync_period_samples(self) -> int:
        return self.sync_period_ticks // self.ticks_per_sample

    def chunk_alignment(
        self,
        frames_per_chunk: int = VIDEO_FRAMES_PER_CHUNK,
        samples_per_latent: int = AUDIO_SAMPLES_PER_LATENT,
    ) -> AVChunkAlignment:
        """Period over which whole video chunks and whole audio latents align."""
        chunk_ticks = self.frames_to_ticks(frames_per_chunk)
        latent_ticks = self.samples_to_ticks(samples_per_latent)
        period = _lcm(chunk_ticks, latent_ticks)
        chunks = period // chunk_ticks
        latents = period // latent_ticks
        return AVChunkAlignment(
            ticks=period,
            ticks_per_second=self.ticks_per_second,
            video_chunks=chunks,
            video_frames=chunks * frames_per_chunk,
            audio_latents=latents,
            audio_samples=latents * samples_per_latent,
        )


@dataclass
class StreamCursor:
    """Tracks how far each lane has been emitted, in exact ticks.

    ``drift_ticks`` is positive when video is ahead of audio.  A scheduler can
    use it to decide which lane to advance next without touching floats.
    """

    clock: MediaClock = None  # type: ignore[assignment]
    frames: int = 0
    samples: int = 0

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = MediaClock()

    def advance_frames(self, n: int) -> int:
        if n < 0:
            raise ValueError("cannot rewind the stream cursor")
        self.frames += int(n)
        return self.frames

    def advance_samples(self, n: int) -> int:
        if n < 0:
            raise ValueError("cannot rewind the stream cursor")
        self.samples += int(n)
        return self.samples

    @property
    def video_ticks(self) -> int:
        return self.clock.frames_to_ticks(self.frames)

    @property
    def audio_ticks(self) -> int:
        return self.clock.samples_to_ticks(self.samples)

    @property
    def drift_ticks(self) -> int:
        return self.video_ticks - self.audio_ticks

    @property
    def drift_seconds(self) -> Fraction:
        return self.clock.ticks_to_seconds(self.drift_ticks)

    def frame_pts(self, frame_index: int) -> int:
        return self.clock.frames_to_ticks(frame_index)

    def sample_pts(self, sample_index: int) -> int:
        return self.clock.samples_to_ticks(sample_index)


#: The concrete clock used by the RAVEN media lane.
RAVEN_CLOCK = MediaClock(DEFAULT_VIDEO_FPS, DEFAULT_AUDIO_SAMPLE_RATE)
