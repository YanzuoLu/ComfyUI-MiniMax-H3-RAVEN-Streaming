"""T2VA chunk layout: the request grid, the chunk cut, positions, stereo rows.

Pure geometry: torch only, no ComfyUI. Agreement with the official packed
layout is a separate module (``test_layout_official_parity.py``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as L  # noqa: E402


# --- request grid ------------------------------------------------------------


@pytest.mark.parametrize("frames,k", [(22, 1), (39, 2), (192, 11), (362, 21)])
def test_valid_frame_grid(frames, k):
    assert L.validate_frames(frames, warn_experimental=False) == k
    assert L.video_latent_t(frames) == 5 * k + 2


def test_k_zero_is_rejected_not_promoted():
    with pytest.raises(L.LayoutError, match="k >= 1"):
        L.validate_frames(5)


@pytest.mark.parametrize("frames", [21, 23, 100, 191])
def test_off_grid_frames_rejected(frames):
    with pytest.raises(L.LayoutError, match=r"17k \+ 5"):
        L.validate_frames(frames)


def test_frame_maximum_enforced():
    with pytest.raises(L.LayoutError, match="<= 362"):
        L.validate_frames(379, warn_experimental=False)


def test_above_192_warns_but_is_allowed():
    with pytest.warns(RuntimeWarning, match="experimental"):
        assert L.validate_frames(209) == 12


def test_canvas_rules():
    L.validate_canvas(1376, 768)
    with pytest.raises(L.LayoutError, match="multiple of 32"):
        L.validate_canvas(1000, 768)
    with pytest.raises(L.LayoutError, match="area|<="):
        L.validate_canvas(1408, 1024)


@pytest.mark.parametrize(
    "frames,audio_t", [(22, 37), (39, 65), (192, 320), (362, 603)]
)
def test_audio_latent_count(frames, audio_t):
    assert L.audio_latent_t(frames) == round(frames / 24 * 40) == audio_t


# --- chunk cut ---------------------------------------------------------------


def test_video_chunks_are_five_plus_a_two_tail():
    ranges = L.video_chunk_ranges(57)
    assert len(ranges) == 12
    assert all(stop - start == 5 for start, stop in ranges[:-1])
    assert ranges[-1] == (55, 57)


def test_video_chunk_ranges_reject_off_grid_latents():
    with pytest.raises(L.LayoutError, match=r"5k \+ 2"):
        L.video_chunk_ranges(56)


def test_audio_cadence_is_derived_not_hardcoded():
    # 29 / 28 / 28 falls out of the 85/3 clock span; the tail chunk takes what
    # is left. Nothing in layout.py stores this pattern.
    lengths = [b - a for a, b in L.audio_chunk_ranges(57, 320)]
    assert lengths[:6] == [29, 28, 28, 29, 28, 28]
    assert sum(lengths) == 320
    assert lengths[-1] == 8


def test_audio_boundary_is_strict_so_audio_never_leads_video():
    # chunk 3 starts at t = 3 * 85/3 = 85 exactly; audio latent 85 must belong
    # to chunk 3, not to chunk 2.
    ranges = L.audio_chunk_ranges(57, 320)
    assert ranges[2][1] == 85
    assert ranges[3][0] == 85


def test_audio_ranges_partition_the_clip():
    for frames in (22, 39, 90, 192, 362):
        latent_t = L.video_latent_t(frames)
        audio_t = L.audio_latent_t(frames)
        ranges = L.audio_chunk_ranges(latent_t, audio_t)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == audio_t
        for (_, prev_stop), (start, _) in zip(ranges, ranges[1:]):
            assert prev_stop == start


def test_audio_assignment_matches_the_shared_clock_definition():
    latent_t, audio_t = 57, 320
    ranges = L.audio_chunk_ranges(latent_t, audio_t)
    video_ranges = L.video_chunk_ranges(latent_t)
    for (a0, a1), (v0, v1) in zip(ranges, video_ranges):
        lo = L.video_position_start(v0, 0.0)
        hi = L.video_position_start(v1, 0.0)
        for j in range(a0, a1):
            assert lo <= float(j) < hi or j == audio_t - 1 or hi > float(j)


def test_chunk_clock_span_is_exactly_85_over_3():
    assert L.video_position_start(5, 0.0) == pytest.approx(85.0 / 3.0, abs=0.0)
    assert L.video_position_start(10, 0.0) == pytest.approx(170.0 / 3.0, rel=1e-15)


# --- positions ---------------------------------------------------------------


def test_text_positions_count_tokens_on_the_t_axis():
    pos = L.text_position_ids(4)
    assert pos.dtype == torch.float64
    assert torch.equal(pos[:, 0], torch.arange(4, dtype=torch.float64))
    assert float(pos[:, 1:].abs().max()) == 0.0


def test_chunk_positions_are_audio_first_video_second_and_absolute():
    layout = L.T2VALayout.from_request(text_len=7, frames=39, width=64, height=64)
    chunk = layout.chunks[1]
    pos = layout.chunk_position_ids(1)
    assert pos.shape == (chunk.rows, 3)

    audio = pos[: chunk.audio_rows]
    video = pos[chunk.audio_rows :]
    # audio: t = origin + latent index, repeated per stereo channel, h = 0
    expected_t = 7.0 + torch.arange(
        chunk.audio_start, chunk.audio_stop, dtype=torch.float64
    ).repeat(2)
    assert torch.equal(audio[:, 0], expected_t)
    assert float(audio[:, 1].abs().max()) == 0.0
    # stereo channels are pinned to the two extremes of the w grid
    _, w_axis = L.frame_grid(layout.latent_h, layout.latent_w)
    assert float(audio[0, 2]) == float(w_axis[0])
    assert float(audio[-1, 2]) == float(w_axis[-1])

    # video: one frame's spatial grid repeated per latent, absolute t
    frame, _ = L.frame_grid(layout.latent_h, layout.latent_w)
    video = video.reshape(chunk.video_latents, chunk.frame_rows, 3)
    for row, index in enumerate(range(chunk.video_start, chunk.video_stop)):
        assert float(video[row, 0, 0]) == L.video_position_start(index, 7.0)
        assert torch.equal(video[row, :, 1:], frame)


def test_chunk_positions_are_a_partition_of_the_clip_positions():
    layout = L.T2VALayout.from_request(text_len=3, frames=90, width=96, height=64)
    seen_audio, seen_video = [], []
    for i in range(layout.num_chunks):
        chunk = layout.chunks[i]
        pos = layout.chunk_position_ids(i)
        seen_audio.append(pos[: chunk.audio_rows])
        seen_video.append(pos[chunk.audio_rows :])
    total = sum(int(p.shape[0]) for p in seen_audio) + sum(
        int(p.shape[0]) for p in seen_video
    )
    assert total == layout.audio_t * 2 + layout.latent_t * layout.frame_rows


# --- stereo permutation ------------------------------------------------------


def test_stereo_chunk_indices_are_two_disjoint_spans():
    index = L.stereo_chunk_indices(audio_t=10, start=3, stop=6)
    assert index.tolist() == [3, 4, 5, 13, 14, 15]


def test_stereo_gather_scatter_round_trip():
    audio_t = 65
    rows = torch.arange(audio_t * 2 * 4, dtype=torch.float32).reshape(audio_t * 2, 4)
    layout = L.T2VALayout.from_request(text_len=2, frames=39, width=64, height=64)
    ranges = layout.audio_chunk_ranges()
    chunks = [L.gather_stereo_chunk(rows, start, stop) for start, stop in ranges]
    assert [int(c.shape[0]) for c in chunks] == [2 * (b - a) for a, b in ranges]
    assert torch.equal(L.scatter_stereo_chunks(chunks, ranges, audio_t), rows)


def test_scatter_rejects_incomplete_coverage():
    rows = torch.zeros(8, 2)
    with pytest.raises(L.LayoutError):
        L.scatter_stereo_chunks([rows], [(0, 3)], audio_t=4)


# --- layout object -----------------------------------------------------------


def test_layout_shapes_and_chunk_rows():
    layout = L.T2VALayout.from_request(text_len=11, frames=192, width=1376, height=768)
    assert layout.latent_t == 57
    assert layout.audio_t == 320
    assert (layout.latent_h, layout.latent_w) == (48, 86)
    assert layout.frame_rows == 24 * 43
    assert layout.video_latent_shape() == (1, 24, 57, 48, 86)
    assert layout.audio_latent_shape() == (1, 32, 2, 320)
    assert layout.num_chunks == 12
    for chunk in layout.chunks:
        assert chunk.rows == chunk.audio_rows + chunk.video_rows
        assert chunk.audio_rows == 2 * chunk.audio_latents
        assert chunk.video_rows == chunk.video_latents * layout.frame_rows
    assert sum(c.video_latents for c in layout.chunks) == layout.latent_t
    assert sum(c.audio_latents for c in layout.chunks) == layout.audio_t


def test_layout_latent_slicing_matches_the_chunk_ranges():
    layout = L.T2VALayout.from_request(text_len=5, frames=39, width=64, height=64)
    video = torch.randn(*layout.video_latent_shape(24))
    audio = torch.randn(*layout.audio_latent_shape(32))
    rebuilt_video = torch.cat(
        [layout.video_chunk_latent(video, i) for i in range(layout.num_chunks)], dim=2
    )
    rebuilt_audio = torch.cat(
        [layout.audio_chunk_latent(audio, i) for i in range(layout.num_chunks)], dim=3
    )
    assert torch.equal(rebuilt_video, video)
    assert torch.equal(rebuilt_audio, audio)


def test_layout_rejects_bad_requests():
    with pytest.raises(L.LayoutError):
        L.T2VALayout.from_request(text_len=1, frames=5, width=64, height=64)
    with pytest.raises(L.LayoutError):
        L.T2VALayout.from_request(text_len=1, frames=22, width=60, height=64)
    with pytest.raises(L.LayoutError):
        L.T2VALayout.from_request(text_len=0, frames=22, width=64, height=64)
