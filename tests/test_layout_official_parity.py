"""The chunk layout against upstream's whole-clip ``PackedLayout``.

The causal lane cuts the packed sequence into chunks, so it rebuilds the
geometry in closed form instead of slicing a clip-wide materialisation. These
tests pin that the rebuild lands on the same grid as the pinned checkout:

* spatial coordinates: **bitwise** identical (same expression, same order);
* text positions: **bitwise** identical;
* video times: identical up to float64 rounding -- upstream accumulates a
  cumulative sum, the closed form does not. Measured deviation over 57 latents
  is ~1.7e-13 on coordinates of magnitude ~300, i.e. ~1 ulp.

Requires a local ComfyUI checkout (see ``tests/conftest.py``); skipped without one.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as L  # noqa: E402

#: absolute tolerance on the shared clock, ~1 ulp of a 300-wide float64 coordinate
CLOCK_ATOL = 1e-11


@pytest.fixture
def official(comfyui_on_syspath):
    import comfy.ldm.minimax.model as model

    return model


def _official_layout(model, layout: L.T2VALayout):
    return model.PackedLayout(
        layout.text_len, layout.latent_t, layout.latent_h, layout.latent_w, layout.audio_t
    )


def test_spatial_axis_is_bitwise_identical(official):
    for dim in (4, 24, 48, 86):
        for area in (10.0, 64.0, 2048.0):
            ours = L.axis_from_sqrt_area(dim, 2, area)
            theirs = official._axis_from_sqrt_area(dim, 2, area)
            assert torch.equal(ours, theirs)


def test_frame_grid_is_bitwise_identical(official):
    for h, w in ((4, 4), (48, 86), (24, 40)):
        ours, ours_w = L.frame_grid(h, w)
        theirs, theirs_w = official._frame_grid(h, w)
        assert torch.equal(ours, theirs)
        assert torch.equal(ours_w, theirs_w)


def test_video_latent_count_matches_the_official_node_rule(official):
    # comfy_extras pulls in the whole node graph (torchsde, scipy, ...); the rule
    # itself is what matters, so a missing optional dep skips rather than fails.
    temporal_shape = pytest.importorskip(
        "comfy_extras.nodes_minimax_h3", reason="official node module needs the full ComfyUI dep set"
    ).temporal_shape

    for frames in (22, 39, 192, 362):
        aligned, latent_t, audio_t = temporal_shape(frames)
        assert aligned == frames
        assert latent_t == L.video_latent_t(frames)
        assert audio_t == L.audio_latent_t(frames)


def test_text_positions_are_bitwise_identical(official):
    layout = L.T2VALayout.from_request(text_len=9, frames=39, width=64, height=64)
    packed = _official_layout(official, layout)
    assert torch.equal(layout.text_position_ids(), packed.position_ids[: layout.text_len])


def test_video_time_closed_form_matches_the_official_cumsum(official):
    grid = official._video_t_grid(57, 7.0)
    ours = torch.tensor(
        [L.video_position_start(i, 7.0) for i in range(57)], dtype=torch.float64
    )
    assert torch.allclose(ours, grid, rtol=0.0, atol=CLOCK_ATOL)


@pytest.mark.parametrize("frames,width,height", [(22, 64, 64), (39, 96, 64), (90, 64, 96)])
def test_chunk_positions_match_the_official_packed_rows(official, frames, width, height):
    layout = L.T2VALayout.from_request(text_len=6, frames=frames, width=width, height=height)
    packed = _official_layout(official, layout)
    text_len, audio_t, frame_rows = layout.text_len, layout.audio_t, layout.frame_rows
    audio_base = text_len
    video_base = text_len + audio_t * 2

    for index, chunk in enumerate(layout.chunks):
        ours = layout.chunk_position_ids(index)
        # upstream packs [text | L(all) R(all) | video(all)]
        rows = torch.cat(
            (
                torch.arange(audio_base + chunk.audio_start, audio_base + chunk.audio_stop),
                torch.arange(
                    audio_base + audio_t + chunk.audio_start,
                    audio_base + audio_t + chunk.audio_stop,
                ),
                torch.arange(
                    video_base + chunk.video_start * frame_rows,
                    video_base + chunk.video_stop * frame_rows,
                ),
            )
        )
        theirs = packed.position_ids.index_select(0, rows)
        assert ours.shape == theirs.shape
        # audio rows and the spatial columns are exact; only the video clock rounds
        assert torch.equal(ours[: chunk.audio_rows], theirs[: chunk.audio_rows])
        assert torch.equal(ours[:, 1:], theirs[:, 1:])
        assert torch.allclose(ours[:, 0], theirs[:, 0], rtol=0.0, atol=CLOCK_ATOL)


def test_stereo_permutation_matches_the_official_audio_pack(official):
    layout = L.T2VALayout.from_request(text_len=4, frames=39, width=64, height=64)
    audio = torch.randn(1, 32, 2, layout.audio_t)
    native = official.pack_audio(audio)
    ranges = layout.audio_chunk_ranges()
    chunks = [official.pack_audio(layout.audio_chunk_latent(audio, i))
              for i in range(layout.num_chunks)]
    # a chunk packed on its own equals the clip-wide pack gathered by the
    # stereo permutation, which is the whole reason that permutation exists
    for chunk, (start, stop) in zip(chunks, ranges):
        assert torch.equal(chunk, L.gather_stereo_chunk(native, start, stop))
    assert torch.equal(L.scatter_stereo_chunks(chunks, ranges, layout.audio_t), native)
