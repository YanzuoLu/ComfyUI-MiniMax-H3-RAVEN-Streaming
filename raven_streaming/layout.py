"""T2VA packed-sequence geometry for the chunk-causal MiniMax H3 lane (M2).

Scope
-----
**T2VA only.** No keyframes (``fl2va``), no references (``ref2va``), no
condition rows, batch size 1. Those layouts exist upstream in
``comfy.ldm.minimax.model.PackedLayout``; this module deliberately implements
only the target-stream geometry the RAVEN streaming sampler needs, cut into
time chunks.

What this module owns
---------------------
* the request grid: ``frames = 17k + 5`` with ``k >= 1``, 32-multiple canvas
  under the area cap, ``latent_t = 5k + 2``, ``audio_t = round(frames / 24 * 40)``;
* the chunk cut: ``5`` video latents per chunk plus a ``2``-latent tail chunk,
  and the audio rows each chunk owns, derived from the shared ``85/3`` clock
  (the 29 / 28 / 28 cadence is a *consequence*, never a hard-coded pattern);
* per-chunk absolute ``(t, h, w)`` positions, audio rows first and video rows
  second, on the same float64 grid the dense model uses, so a chunk's RoPE
  angles are the ones the dense layout would have given those rows;
* the stereo row permutation, because the model's native audio pack is
  channel-major over the *whole clip* (``[L(0..T-1) | R(0..T-1)]``) while a
  chunk carries ``[L(a..b) | R(a..b)]``.

Relation to upstream
--------------------
The geometry is re-derived here in closed form rather than sliced out of
``PackedLayout``: a chunk must be constructible on its own, and upstream only
ever materialises the whole clip. The closed form is the same one RAVEN's
training layout uses (``projects/minimax_h3/data/causal_text_only.py``).

Spatial coordinates are bitwise identical to upstream's (same expression, same
float64 order). Video times differ from upstream's cumulative sum by float
rounding only -- measured at ``1.7e-13`` absolute over 57 latents, i.e. ~1 ulp
of a ~300-wide coordinate. ``tests/test_layout_official_parity.py`` pins both
claims against the pinned checkout.

Import weight: torch only. ComfyUI is *not* imported here.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

__all__ = [
    "LayoutError",
    "FPS",
    "AUDIO_LATENT_FPS",
    "FRAME_PER_TOKEN",
    "FRAME_RESCALE",
    "VIDEO_LATENTS_PER_CHUNK",
    "CHUNK_T_SPAN",
    "AUDIO_CHANNELS",
    "PATCH_SIZE",
    "CANVAS_MULTIPLE",
    "MAX_AREA",
    "MIN_FRAMES",
    "MAX_FRAMES",
    "EXPERIMENTAL_FRAMES",
    "VIDEO_TAG",
    "TEXT_TAG",
    "AUDIO_TAG",
    "validate_frames",
    "validate_canvas",
    "video_latent_t",
    "audio_latent_t",
    "video_position_start",
    "video_chunk_ranges",
    "audio_chunk_ranges",
    "axis_from_sqrt_area",
    "frame_grid",
    "text_position_ids",
    "chunk_position_ids",
    "stereo_chunk_indices",
    "gather_stereo_chunk",
    "scatter_stereo_chunks",
    "Chunk",
    "T2VALayout",
]


# --- family constants (mirrors of comfy.ldm.minimax.model / the H3 checkpoint)

FPS = 24
AUDIO_LATENT_FPS = 40
#: pixel frames each video latent of a 5-latent group stands for
FRAME_PER_TOKEN: Tuple[int, ...] = (1, 4, 4, 4, 4)
#: position-grid units per pixel frame
FRAME_RESCALE = 5.0 / 3.0
#: one causal chunk is one full 5-latent group ...
VIDEO_LATENTS_PER_CHUNK = 5
#: ... which spans exactly ``FRAME_RESCALE * sum(FRAME_PER_TOKEN) == 85 / 3``
#: units of the shared clock, the same clock audio latents advance on at 1/frame.
CHUNK_T_SPAN = 85.0 / 3.0
AUDIO_CHANNELS = 2
PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)

#: AdaLN modality tags, from the dense model's ``seg_tag``.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

# --- request constraints (docs/requirements.md R6)

CANVAS_MULTIPLE = 32
MAX_AREA = 1376 * 768
#: ``k >= 1``; ``k == 0`` (5 frames / 2 latents) is rejected, never promoted.
MIN_FRAMES = 22
MAX_FRAMES = 362
#: above this the request is allowed but warned as experimental
EXPERIMENTAL_FRAMES = 192


class LayoutError(ValueError):
    """A request that does not sit on the H3 T2VA grid."""


def validate_frames(frames: int, *, warn_experimental: bool = True) -> int:
    """Return ``k`` for a valid ``frames = 17k + 5`` request, or raise.

    ``k == 0`` (5 frames, 2 video latents) is rejected outright: it leaves no
    room for the streaming loop's context, so v0.1 fails loud instead of
    silently rounding up.
    """
    if int(frames) != frames:
        raise LayoutError(f"frames must be an integer, got {frames!r}")
    frames = int(frames)
    if frames % 17 != 5:
        raise LayoutError(
            f"frames must satisfy 17k + 5, got {frames} "
            f"(nearest valid: {frames - (frames - 5) % 17}, "
            f"{frames + (17 - (frames - 5) % 17) % 17})"
        )
    k = (frames - 5) // 17
    if k < 1:
        raise LayoutError(
            f"frames={frames} is k={k}; v0.1 requires k >= 1 (frames >= {MIN_FRAMES}). "
            "The 5-frame / 2-latent case is not supported."
        )
    if frames > MAX_FRAMES:
        raise LayoutError(f"frames must be <= {MAX_FRAMES}, got {frames}")
    if warn_experimental and frames > EXPERIMENTAL_FRAMES:
        warnings.warn(
            f"frames={frames} exceeds the {EXPERIMENTAL_FRAMES}-frame audited "
            "range; allowed but experimental.",
            RuntimeWarning,
            stacklevel=2,
        )
    return k


def validate_canvas(width: int, height: int) -> None:
    """Enforce the 32-multiple canvas and the area cap, or raise."""
    for name, value in (("width", width), ("height", height)):
        if int(value) != value or value <= 0:
            raise LayoutError(f"{name} must be a positive integer, got {value!r}")
        if int(value) % CANVAS_MULTIPLE != 0:
            raise LayoutError(
                f"{name} must be a multiple of {CANVAS_MULTIPLE}, got {value}"
            )
    if int(width) * int(height) > MAX_AREA:
        raise LayoutError(
            f"width * height must be <= {MAX_AREA} ({MAX_AREA // 768}x768), "
            f"got {width}x{height} = {int(width) * int(height)}"
        )


def video_latent_t(frames: int) -> int:
    """``frames = 17k + 5`` pixel frames -> ``5k + 2`` video latents."""
    validate_frames(frames, warn_experimental=False)
    return ((frames - 5) // 17) * VIDEO_LATENTS_PER_CHUNK + 2


def audio_latent_t(frames: int) -> int:
    """Audio latent count at 40 Hz for a 24 fps clip of ``frames`` frames."""
    validate_frames(frames, warn_experimental=False)
    return round(frames / FPS * AUDIO_LATENT_FPS)


# --- temporal grid -----------------------------------------------------------


def video_position_start(index: int, origin: float = 0.0) -> float:
    """Shared-clock start of video latent ``index``, without building priors.

    Closed form of upstream's exclusive cumulative sum over
    ``FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]``: whole 5-latent groups advance by
    exactly ``CHUNK_T_SPAN``, and the offset inside a group is the partial sum.
    """
    group, offset = divmod(int(index), VIDEO_LATENTS_PER_CHUNK)
    return (
        float(origin)
        + group * CHUNK_T_SPAN
        + FRAME_RESCALE * float(sum(FRAME_PER_TOKEN[:offset]))
    )


def video_chunk_ranges(latent_t: int) -> List[Tuple[int, int]]:
    """Cut ``5k + 2`` video latents into ``k`` full chunks plus the 2-row tail."""
    if latent_t < 2 or (latent_t - 2) % VIDEO_LATENTS_PER_CHUNK != 0:
        raise LayoutError(f"latent_t must be 5k + 2, got {latent_t}")
    return [
        (start, min(start + VIDEO_LATENTS_PER_CHUNK, latent_t))
        for start in range(0, latent_t, VIDEO_LATENTS_PER_CHUNK)
    ]


def audio_chunk_ranges(latent_t: int, audio_t: int) -> List[Tuple[int, int]]:
    """Assign audio latents to video chunks on the shared 40 Hz clock.

    Audio latent ``j`` belongs to chunk ``c`` exactly when ``T_c <= j < T_{c+1}``
    for the chunk's clock span. The strict ``<`` at the upper bound is
    load-bearing: at an integer boundary the audio latent stays with the *next*
    chunk, so audio is never generated ahead of the video it belongs to. The
    29 / 28 / 28 cadence falls out of the ``85/3`` span; it is never stored as a
    pattern.
    """
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for video_start, video_stop in video_chunk_ranges(latent_t):
        chunk_start = video_position_start(video_start, 0.0)
        chunk_stop = video_position_start(video_stop, 0.0)
        while cursor < audio_t and float(cursor) < chunk_start:
            cursor += 1
        start = cursor
        while cursor < audio_t and float(cursor) < chunk_stop:
            cursor += 1
        ranges.append((start, cursor))
    if cursor != audio_t:
        raise LayoutError(
            f"audio chunking left {audio_t - cursor} of {audio_t} audio latents "
            "unassigned; latent_t and audio_t do not describe the same clip"
        )
    return ranges


# --- spatial grid ------------------------------------------------------------


def axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One area-normalised spatial axis.

    Character-for-character the upstream expression
    (``comfy.ldm.minimax.model._axis_from_sqrt_area``); any deviation would move
    every RoPE angle.
    """
    ratio = dim / sqrt_area
    n = dim // patch
    return (
        torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0
    ) * 32.0


def frame_grid(latent_h: int, latent_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``([rows, 2] float64 (h, w) of one latent frame's 2x2-patch rows, w axis)``."""
    area = math.sqrt(latent_h * latent_w)
    h_axis = axis_from_sqrt_area(latent_h, PATCH_SIZE[1], area)
    w_axis = axis_from_sqrt_area(latent_w, PATCH_SIZE[2], area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


# --- packed positions --------------------------------------------------------


def text_position_ids(text_len: int) -> torch.Tensor:
    """``[text_len, 3]`` float64: ``t`` counts tokens, ``h``/``w`` stay 0."""
    g = torch.zeros(int(text_len), 3, dtype=torch.float64)
    g[:, 0] = torch.arange(int(text_len), dtype=torch.float64)
    return g


def chunk_position_ids(
    *,
    video_start: int,
    video_stop: int,
    audio_start: int,
    audio_stop: int,
    latent_h: int,
    latent_w: int,
    origin: float,
) -> torch.Tensor:
    """One chunk's ``[rows, 3]`` float64 positions, audio rows then video rows.

    ``origin`` is the target timeline origin, i.e. ``text_len`` for T2VA (the
    dense layout starts the media clock right after the text rows).
    """
    frame, w_axis = frame_grid(latent_h, latent_w)
    frame_rows = int(frame.shape[0])

    audio_n = int(audio_stop) - int(audio_start)
    audio = torch.zeros(audio_n * AUDIO_CHANNELS, 3, dtype=torch.float64)
    audio[:, 0] = (
        float(origin)
        + torch.arange(int(audio_start), int(audio_stop), dtype=torch.float64)
    ).repeat(AUDIO_CHANNELS)
    audio[:audio_n, 2] = float(w_axis[0])
    audio[audio_n:, 2] = float(w_axis[-1])

    video_t = torch.tensor(
        [
            video_position_start(index, origin)
            for index in range(int(video_start), int(video_stop))
        ],
        dtype=torch.float64,
    )
    video = torch.empty(video_t.numel(), frame_rows, 3, dtype=torch.float64)
    video[:, :, 0] = video_t[:, None]
    video[:, :, 1:] = frame[None]

    return torch.cat((audio, video.reshape(-1, 3)), dim=0)


# --- stereo row permutation --------------------------------------------------


def stereo_chunk_indices(audio_t: int, start: int, stop: int) -> torch.Tensor:
    """Rows of ``[L(start:stop) | R(start:stop)]`` inside the clip-wide pack.

    The model's native audio pack is channel-major over the whole clip
    (``pack_audio``: ``[L(0..T-1) | R(0..T-1)]``), so a chunk is two disjoint
    spans, not one slice.
    """
    left = torch.arange(int(start), int(stop), dtype=torch.long)
    right = torch.arange(int(audio_t) + int(start), int(audio_t) + int(stop), dtype=torch.long)
    return torch.cat((left, right))


def gather_stereo_chunk(native_rows: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Gather one chunk's stereo rows out of clip-wide channel-major rows."""
    audio_t = int(native_rows.shape[0]) // AUDIO_CHANNELS
    index = stereo_chunk_indices(audio_t, start, stop).to(native_rows.device)
    return native_rows.index_select(0, index)


def scatter_stereo_chunks(
    chunks: Sequence[torch.Tensor],
    chunk_ranges: Sequence[Tuple[int, int]],
    audio_t: int,
) -> torch.Tensor:
    """Invert :func:`gather_stereo_chunk` for a full set of chunks."""
    if len(chunks) != len(chunk_ranges):
        raise LayoutError(
            f"got {len(chunks)} chunks for {len(chunk_ranges)} ranges"
        )
    covered = sum(stop - start for start, stop in chunk_ranges)
    if covered != int(audio_t):
        raise LayoutError(
            f"chunk ranges cover {covered} audio latents, expected {audio_t}"
        )
    native = chunks[0].new_empty((int(audio_t) * AUDIO_CHANNELS, *chunks[0].shape[1:]))
    for chunk, (start, stop) in zip(chunks, chunk_ranges):
        index = stereo_chunk_indices(audio_t, start, stop).to(chunk.device)
        native.index_copy_(0, index, chunk)
    return native


# --- layout objects ----------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One time chunk: its video latents, its audio latents, its row counts."""

    index: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int
    frame_rows: int

    @property
    def video_latents(self) -> int:
        return self.video_stop - self.video_start

    @property
    def audio_latents(self) -> int:
        return self.audio_stop - self.audio_start

    @property
    def video_rows(self) -> int:
        return self.video_latents * self.frame_rows

    @property
    def audio_rows(self) -> int:
        return self.audio_latents * AUDIO_CHANNELS

    @property
    def rows(self) -> int:
        """Total packed rows: audio first, video second."""
        return self.audio_rows + self.video_rows


@dataclass(frozen=True)
class T2VALayout:
    """Chunk-major T2VA geometry for one request.

    ``chunks`` are time-ordered; the last one is the 2-latent tail. The packed
    order inside a chunk is audio rows then video rows, matching RAVEN's
    training layout.
    """

    text_len: int
    frames: int
    width: int
    height: int
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int
    chunks: Tuple[Chunk, ...]

    @classmethod
    def from_request(
        cls,
        *,
        text_len: int,
        frames: int,
        width: int,
        height: int,
        warn_experimental: bool = True,
    ) -> "T2VALayout":
        validate_frames(frames, warn_experimental=warn_experimental)
        validate_canvas(width, height)
        if int(text_len) <= 0:
            raise LayoutError(f"text_len must be positive, got {text_len!r}")

        latent_t = video_latent_t(frames)
        audio_t = audio_latent_t(frames)
        latent_h = int(height) // 16
        latent_w = int(width) // 16
        if latent_h % PATCH_SIZE[1] or latent_w % PATCH_SIZE[2]:
            raise LayoutError(
                f"latent grid {latent_h}x{latent_w} is not a multiple of the "
                f"{PATCH_SIZE[1]}x{PATCH_SIZE[2]} DiT patch"
            )
        frame_rows = (latent_h // PATCH_SIZE[1]) * (latent_w // PATCH_SIZE[2])

        video_ranges = video_chunk_ranges(latent_t)
        audio_ranges = audio_chunk_ranges(latent_t, audio_t)
        chunks = tuple(
            Chunk(
                index=i,
                video_start=v0,
                video_stop=v1,
                audio_start=a0,
                audio_stop=a1,
                frame_rows=frame_rows,
            )
            for i, ((v0, v1), (a0, a1)) in enumerate(zip(video_ranges, audio_ranges))
        )
        return cls(
            text_len=int(text_len),
            frames=int(frames),
            width=int(width),
            height=int(height),
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            chunks=chunks,
        )

    # -- derived shapes

    @property
    def frame_rows(self) -> int:
        return (self.latent_h // PATCH_SIZE[1]) * (self.latent_w // PATCH_SIZE[2])

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def video_latent_shape(self, latents_dim: int = 24) -> Tuple[int, int, int, int, int]:
        return (1, latents_dim, self.latent_t, self.latent_h, self.latent_w)

    def audio_latent_shape(self, audio_latents_dim: int = 32) -> Tuple[int, int, int, int]:
        return (1, audio_latents_dim, AUDIO_CHANNELS, self.audio_t)

    # -- positions

    def text_position_ids(self) -> torch.Tensor:
        return text_position_ids(self.text_len)

    def chunk(self, chunk_index: int) -> Chunk:
        """Bounds-checked chunk lookup."""
        if not (0 <= int(chunk_index) < len(self.chunks)):
            raise LayoutError(
                f"chunk_index {chunk_index} outside [0, {len(self.chunks)})"
            )
        return self.chunks[int(chunk_index)]

    def chunk_position_ids(self, chunk_index: int) -> torch.Tensor:
        chunk = self.chunk(chunk_index)
        return chunk_position_ids(
            video_start=chunk.video_start,
            video_stop=chunk.video_stop,
            audio_start=chunk.audio_start,
            audio_stop=chunk.audio_stop,
            latent_h=self.latent_h,
            latent_w=self.latent_w,
            origin=float(self.text_len),
        )

    # -- latent slicing

    def video_chunk_latent(self, latent: torch.Tensor, chunk_index: int) -> torch.Tensor:
        """``[B, C, T, H, W]`` -> this chunk's ``[B, C, t, H, W]`` view."""
        chunk = self.chunk(chunk_index)
        return latent[:, :, chunk.video_start : chunk.video_stop]

    def audio_chunk_latent(self, latent: torch.Tensor, chunk_index: int) -> torch.Tensor:
        """``[B, C, 2, T]`` -> this chunk's ``[B, C, 2, t]`` view."""
        chunk = self.chunk(chunk_index)
        return latent[:, :, :, chunk.audio_start : chunk.audio_stop]

    def audio_chunk_ranges(self) -> List[Tuple[int, int]]:
        return [(c.audio_start, c.audio_stop) for c in self.chunks]

    def video_chunk_ranges(self) -> List[Tuple[int, int]]:
        return [(c.video_start, c.video_stop) for c in self.chunks]
