"""Falsification tests for the incremental video VAE decode coordinator.

The bar is exact equality with a faithful port of upstream
``MiniMaxH3VideoVAE.decode_temporal`` over a fake decoder, for every latent
count in a wide grid - including the 5k+2 grid the encoder actually produces
and every tail-padding case.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from raven_streaming.media.fakes import (  # noqa: E402
    FakeVideoChunkDecoder,
    make_latents,
    numpy_concat,
)
from raven_streaming.media.video_stream import (  # noqa: E402
    MIN_SUPPORTED_FRAMES,
    MIN_SUPPORTED_LATENTS,
    IncrementalVideoDecoder,
    ShortVideoStreamError,
    VideoChunkParams,
    reference_decode_temporal,
    summarize_decode_comparison,
)

EXACT = {"max_abs_diff": 0.0, "exact": True}

PARAMS = VideoChunkParams()


# -- geometry --------------------------------------------------------------


def test_geometry_matches_the_official_vae_constants():
    p = PARAMS
    assert p.tokens_chunk_size == 5
    assert p.token_overlap == 2
    assert p.frame_pre_padding == 3
    assert p.frame_overlap == 5
    assert p.chunk_dec == 20
    assert p.split_count == 2
    assert p.latents_needed == 7
    assert p.latents_per_step == 5
    assert p.frames_per_step == 17
    assert p.tail_frames == 5
    assert p.lookahead_latents == 2


def test_from_vae_reads_a_duck_typed_model():
    class Model(object):
        clip_length = 17
        vae_ratio_t = 4
        token_drop = 3

    assert VideoChunkParams.from_vae(Model()) == PARAMS

    class Wrapper(object):
        first_stage_model = Model()

    assert VideoChunkParams.from_vae(Wrapper()) == PARAMS


@pytest.mark.parametrize("k", list(range(1, 13)))
def test_5k_plus_2_grid_yields_17k_plus_5_frames(k):
    """The encoder's natural output length, matching upstream upscale_ratio."""
    t = 5 * k + 2
    assert PARAMS.total_frames(t) == 17 * k + 5
    # upstream: upscale_ratio[0] = lambda a: max(1, (a - 2) // 5 * 17 + 5)
    assert PARAMS.total_frames(t) == max(1, (t - 2) // 5 * 17 + 5)
    # exactly k chunks, no tail padding
    pad_tokens, num_chunks = PARAMS.temporal_chunks(t)
    assert (pad_tokens, num_chunks) == (0, k)


def test_single_latent_still_image_is_internal_only():
    """The 1-latent path exists upstream; it is not a v0.1 product path."""
    assert PARAMS.total_frames(1) == 1  # internal geometry, still true
    assert PARAMS.total_frames(0) == 0

    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    stream.push(make_latents(1, seed=1))
    with pytest.raises(ShortVideoStreamError):
        stream.finish()

    # reachable only behind the explicit internal flag
    internal = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS,
                                       concat=numpy_concat, allow_short_stream=True)
    internal.push(make_latents(1, seed=1))
    assert sum(b.count for b in internal.finish()) == 1


@pytest.mark.parametrize("t", list(range(2, 60)))
def test_chunk_plan_is_self_consistent(t):
    pad_tokens, num_chunks = PARAMS.temporal_chunks(t)
    padded = t + pad_tokens
    # the padded stream is always an exact number of whole chunks + overlap
    assert padded == num_chunks * PARAMS.tokens_chunk_size + PARAMS.token_overlap
    assert num_chunks >= 1
    # streaming may never run more chunks than the offline plan
    assert PARAMS.eager_chunks_available(t) <= num_chunks
    assert num_chunks - PARAMS.eager_chunks_available(t) <= 1


# -- streaming equivalence -------------------------------------------------


def _reference_frames(decoder, z):
    parts = reference_decode_temporal(decoder, z, PARAMS, concat=numpy_concat)
    if not parts:
        return np.zeros((1, 3, 0, 2, 2))
    return np.concatenate(parts, axis=2)


def _streamed_frames(decoder, z, push_sizes, allow_short_stream=True):
    # allow_short_stream: these are upstream-equivalence tests over the whole
    # internal length domain, not product claims.  The product minimum is
    # enforced separately (see the rejection tests).
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat,
                                     allow_short_stream=allow_short_stream)
    batches = []
    pos = 0
    for size in push_sizes:
        if pos >= z.shape[2]:
            break
        chunk = z[:, :, pos:pos + size]
        pos += chunk.shape[2]
        batches.extend(stream.push(chunk))
    if pos < z.shape[2]:
        batches.extend(stream.push(z[:, :, pos:]))
    batches.extend(stream.finish())
    return batches, stream


def _concat_batches(batches):
    if not batches:
        return np.zeros((1, 3, 0, 2, 2))
    return np.concatenate([b.frames for b in batches], axis=2)


@pytest.mark.parametrize("finalize_mode", ["pointwise", "cross_frame"])
@pytest.mark.parametrize("t", list(range(MIN_SUPPORTED_LATENTS, 60)))
def test_streaming_equals_reference_for_every_supported_length(t, finalize_mode):
    z = make_latents(t, seed=t)
    ref = _reference_frames(FakeVideoChunkDecoder(finalize_mode=finalize_mode), z)
    batches, stream = _streamed_frames(
        FakeVideoChunkDecoder(finalize_mode=finalize_mode), z, [t]
    )
    got = _concat_batches(batches)

    assert got.shape == ref.shape
    assert np.array_equal(got, ref)
    assert got.shape[2] == PARAMS.total_frames(t)
    assert stream.frames_emitted == PARAMS.total_frames(t)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("push", [1, 2, 3, 5, 7, 11, 1000])
def test_push_chunking_does_not_change_the_output(k, push):
    """5k+2 grid, arbitrary arrival granularity: byte-identical output."""
    t = 5 * k + 2
    z = make_latents(t, seed=k)
    ref = _reference_frames(FakeVideoChunkDecoder(), z)
    batches, stream = _streamed_frames(FakeVideoChunkDecoder(), z, [push] * 1000)
    got = _concat_batches(batches)

    assert np.array_equal(got, ref)
    assert got.shape[2] == 17 * k + 5
    assert stream.chunks_done == k


def test_batches_are_contiguous_and_correctly_indexed():
    t = 5 * 4 + 2
    z = make_latents(t, seed=4)
    batches, stream = _streamed_frames(FakeVideoChunkDecoder(), z, [1] * 100)
    cursor = 0
    for batch in batches:
        assert batch.start_frame == cursor
        assert batch.count == batch.frames.shape[2]
        cursor = batch.stop_frame
    assert cursor == stream.frames_emitted == PARAMS.total_frames(t)
    # 4 chunks of 17 frames, then the 5-frame tail flush
    assert [b.count for b in batches] == [17, 17, 17, 17, 5]
    assert [b.is_tail for b in batches] == [False, False, False, False, True]


def test_frames_are_emitted_with_only_two_latents_of_lookahead():
    """7 latents in -> 17 frames out; nothing before that, nothing extra after."""
    decoder = FakeVideoChunkDecoder()
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat)
    z = make_latents(17, seed=1)

    emitted = []
    for i in range(17):
        batches = stream.push(z[:, :, i:i + 1])
        emitted.append(sum(b.count for b in batches))

    # first output only once latent index 6 (the 7th) has arrived
    assert emitted[:7] == [0, 0, 0, 0, 0, 0, 17]
    # then one chunk of 17 frames every 5 latents
    assert emitted[7:12] == [0, 0, 0, 0, 17]
    assert emitted[12:17] == [0, 0, 0, 0, 17]
    assert stream.chunks_done == 3


def test_no_full_output_tensor_is_ever_allocated():
    """State stays bounded: <= 7 pending latents and a 5-frame overlap."""
    decoder = FakeVideoChunkDecoder()
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat)
    z = make_latents(202, seed=7)  # 5*40+2 -> 685 frames

    total = 0
    max_pending = 0
    for i in range(z.shape[2]):
        for batch in stream.push(z[:, :, i:i + 1]):
            total += batch.count
            # each batch is released immediately; nothing accumulates
            assert batch.frames.shape[2] <= PARAMS.frames_per_step
        max_pending = max(max_pending, stream.pending_latents)
    for batch in stream.finish():
        total += batch.count

    assert max_pending <= PARAMS.latents_needed
    assert total == PARAMS.total_frames(202) == 17 * 40 + 5


def test_decoder_work_is_not_duplicated():
    """Each chunk decodes exactly 7 latents once - no re-decoding of overlap."""
    t = 5 * 6 + 2
    z = make_latents(t, seed=3)
    decoder = FakeVideoChunkDecoder()
    _streamed_frames(decoder, z, [t])
    assert decoder.decode_calls == 6
    assert decoder.decoded_latents == 6 * 7

    ref_decoder = FakeVideoChunkDecoder()
    _reference_frames(ref_decoder, z)
    assert decoder.decode_calls == ref_decoder.decode_calls
    assert decoder.decoded_latents == ref_decoder.decoded_latents


@pytest.mark.parametrize("finalize_mode", ["pointwise", "cross_frame"])
@pytest.mark.parametrize("t", [8, 9, 10, 11, 13, 14, 16, 18, 23, 24, 26])
def test_tail_padding_paths_match_reference(t, finalize_mode):
    """Lengths off the 5k+2 grid go through the pad+truncate path."""
    pad_tokens, num_chunks = PARAMS.temporal_chunks(t)
    assert pad_tokens > 0  # these all need padding
    z = make_latents(t, seed=100 + t)
    ref = _reference_frames(FakeVideoChunkDecoder(finalize_mode=finalize_mode), z)
    batches, _ = _streamed_frames(
        FakeVideoChunkDecoder(finalize_mode=finalize_mode), z, [2]
    )
    assert np.array_equal(_concat_batches(batches), ref)


# -- finalize / crop ordering ---------------------------------------------
#
# Upstream write_part() finalizes the *whole* part and only then copies as many
# frames as fit in the output plan.  The real _finalize_pixels broadcasts
# pixel_mean/pixel_std of shape [1, 3, 1, 1, 1], so it is pointwise and cannot
# distinguish the two orders - which means a pointwise fake cannot either.  The
# cross_frame fake can, and these tests pin the order with it.


def test_cross_frame_fake_is_actually_order_sensitive():
    """Guard the guard: crop-then-finalize must differ from finalize-then-crop."""
    decoder = FakeVideoChunkDecoder(finalize_mode="cross_frame")
    part = make_latents(17, channels=3, seed=5)  # [1, 3, 17, h, w]

    finalize_then_crop = decoder._finalize_pixels(part)[:, :, :5]
    crop_then_finalize = decoder._finalize_pixels(part[:, :, :5])
    assert not np.allclose(finalize_then_crop, crop_then_finalize)

    # the pointwise fake, like the real hook today, cannot tell them apart
    pointwise = FakeVideoChunkDecoder(finalize_mode="pointwise")
    assert np.array_equal(
        pointwise._finalize_pixels(part)[:, :, :5],
        pointwise._finalize_pixels(part[:, :, :5]),
    )


def test_two_latent_clip_is_rejected_as_a_product_path():
    """v0.1 does not support k=0. 2 latents / 5 frames must fail loud."""
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    stream.push(make_latents(2, seed=42))
    with pytest.raises(ShortVideoStreamError) as excinfo:
        stream.finish()
    message = str(excinfo.value)
    assert "2 latent(s)" in message and "5 frame(s)" in message
    assert "7 latents / 22 frames" in message
    assert "k=0" in message


def test_whole_part_is_finalized_before_it_is_cropped():
    """Upstream finalizes 17 frames then copies 5; count 17, not 5."""
    decoder = FakeVideoChunkDecoder(finalize_mode="cross_frame")
    z = make_latents(2, seed=7)
    batches, _ = _streamed_frames(decoder, z, [2])

    assert sum(b.count for b in batches) == 5
    # j=0 part (17 frames) and the discarded tail (5 frames) are both finalized
    # whole, exactly as upstream's write_part does
    assert decoder.finalized_frames == 17 + 5
    assert decoder.finalize_calls == 2

    reference = FakeVideoChunkDecoder(finalize_mode="cross_frame")
    _reference_frames(reference, z)
    assert decoder.finalized_frames == reference.finalized_frames
    assert decoder.finalize_calls == reference.finalize_calls


def test_truncated_tail_flush_matches_reference():
    """Lengths where the 5-frame tail flush is partly or wholly discarded."""
    for t in range(MIN_SUPPORTED_LATENTS, 40):
        pad_tokens, num_chunks = PARAMS.temporal_chunks(t)
        if pad_tokens == 0:
            continue
        z = make_latents(t, seed=200 + t)
        ref_decoder = FakeVideoChunkDecoder(finalize_mode="cross_frame")
        ref = _reference_frames(ref_decoder, z)
        got_decoder = FakeVideoChunkDecoder(finalize_mode="cross_frame")
        batches, _ = _streamed_frames(got_decoder, z, [1] * 100)
        assert np.array_equal(_concat_batches(batches), ref), "t={}".format(t)
        # identical amount of finalize work, i.e. identical operator sequence
        assert got_decoder.finalized_frames == ref_decoder.finalized_frames
        assert got_decoder.finalize_calls == ref_decoder.finalize_calls


def test_finalize_hook_is_applied_exactly_once():
    """On the untruncated 5k+2 grid every finalized frame is also emitted."""
    t = 5 * 3 + 2
    assert PARAMS.temporal_chunks(t)[0] == 0  # no padding, so no cropping
    decoder = FakeVideoChunkDecoder()
    batches, _ = _streamed_frames(decoder, make_latents(t, seed=9), [t])

    assert decoder.finalized_frames == sum(b.count for b in batches)
    assert decoder.finalized_frames == PARAMS.total_frames(t) == 56
    # 3 chunks + the tail flush
    assert decoder.finalize_calls == 4


def test_misalignment_would_be_detected():
    """Guard the guard: the fake decoder is sensitive to temporal shifts."""
    z = make_latents(12, seed=11)
    decoder = FakeVideoChunkDecoder()
    a = _reference_frames(decoder, z)
    shifted = np.concatenate([z[:, :, 1:], z[:, :, :1]], axis=2)
    b = _reference_frames(decoder, shifted)
    assert not np.allclose(a, b)


def test_finish_is_idempotent_and_push_after_finish_raises():
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    stream.push(make_latents(7, seed=2))
    stream.finish()
    assert stream.finish() == []
    with pytest.raises(RuntimeError):
        stream.push(make_latents(1, seed=2))


def test_empty_stream_produces_nothing():
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    assert stream.finish() == []
    assert stream.frames_emitted == 0


# -- abort -----------------------------------------------------------------
#
# The cancel path. What makes it worth its own tests is that the coordinator is
# holding the two things a cancelled run most wants gone: the pending latents,
# and the 5-frame `dec_overlap` which on real hardware sits on the decode
# device between chunks.


def test_abort_drops_every_buffer_without_decoding_anything():
    decoder = FakeVideoChunkDecoder()
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat)
    # 9 latents: chunk 0 has run (so `dec_overlap` holds 5 decoded frames) and
    # latents 5..8 are still pending. Both kinds of state are live.
    stream.push(make_latents(9, seed=11))
    assert stream.chunks_done == 1
    assert stream._dec_overlap is not None and stream._pending is not None
    decodes = decoder.decode_calls

    stream.abort()

    assert stream._pending is None
    assert stream._dec_overlap is None
    assert stream._frame_limit is None
    # not a flush: no padding, no last chunk, no tail, no decoder work at all
    assert decoder.decode_calls == decodes
    assert stream.frames_emitted == 17  # only what chunk 0 already emitted


def test_abort_is_terminal_in_both_directions():
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    stream.push(make_latents(9, seed=12))
    stream.abort()

    # no partial clip through either door
    assert stream.finish() == []
    with pytest.raises(RuntimeError):
        stream.push(make_latents(1, seed=12))


def test_abort_is_idempotent_and_harmless_after_finish():
    decoder = FakeVideoChunkDecoder()
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat)
    stream.push(make_latents(7, seed=13))
    batches = stream.finish()
    decodes, emitted = decoder.decode_calls, stream.frames_emitted
    assert sum(b.count for b in batches) == PARAMS.tail_frames  # the 5-frame flush
    assert emitted == PARAMS.total_frames(7)  # the whole clip came out first

    stream.abort()
    stream.abort()

    assert decoder.decode_calls == decodes
    assert stream.frames_emitted == emitted  # the record survives; the buffers do not
    assert stream._pending is None and stream._dec_overlap is None


def test_abort_before_anything_was_pushed_is_a_no_op():
    decoder = FakeVideoChunkDecoder()
    stream = IncrementalVideoDecoder(decoder, PARAMS, concat=numpy_concat)
    stream.abort()
    assert stream.finish() == []
    assert decoder.decode_calls == 0
    assert stream.frames_emitted == 0


# -- probe verdict logic ---------------------------------------------------
#
# The GPU probe's judgement is pure arithmetic on measured diffs, so it is
# tested here rather than only on a node with a checkpoint.


def test_verdict_passes_only_when_the_warm_pair_is_within_tolerance():
    good = summarize_decode_comparison(cold=EXACT, warm=EXACT, tolerance=1e-6)
    assert good.passed
    assert any("BITWISE IDENTICAL" in n for n in good.notes)

    bad = summarize_decode_comparison(
        cold={"max_abs_diff": 1e-3, "exact": False},
        warm={"max_abs_diff": 1e-3, "exact": False},
        tolerance=1e-6,
    )
    assert not bad.passed
    assert any("NOT" in n and "relaxed" in n for n in bad.notes)
    assert any("finalized before it is cropped" in n for n in bad.notes)


def test_verdict_reports_cold_kernel_evidence_without_relaxing_tolerance():
    """The vr-1 grid-2 signature: first pass off, later pass exact."""
    verdict = summarize_decode_comparison(
        cold={"max_abs_diff": 0.002417, "exact": False},
        warm=EXACT,
        official_self=EXACT,
        incremental_self=EXACT,
        tolerance=1e-6,
    )
    assert verdict.passed
    assert verdict.cold_within_tolerance is False
    assert verdict.warm_within_tolerance is True
    assert any("COLD-KERNEL EVIDENCE" in n for n in verdict.notes)
    assert any("2.417e-03" in n for n in verdict.notes)


def test_verdict_flags_a_nondeterministic_official_decoder():
    verdict = summarize_decode_comparison(
        cold={"max_abs_diff": 2e-3, "exact": False},
        warm={"max_abs_diff": 2e-3, "exact": False},
        official_self={"max_abs_diff": 2e-3, "exact": False},
        incremental_self={"max_abs_diff": 2e-3, "exact": False},
        tolerance=1e-6,
    )
    # still a failure: the gate is not widened by the excuse
    assert not verdict.passed
    assert verdict.official_reproducible is False
    assert verdict.incremental_reproducible is False
    assert any("NOT BITWISE REPRODUCIBLE" in n for n in verdict.notes)


def test_verdict_reports_shape_mismatch():
    verdict = summarize_decode_comparison(
        cold={"max_abs_diff": None, "exact": False,
              "shape_mismatch": [[1, 3, 5, 2, 2], [1, 3, 22, 2, 2]]},
        warm={"max_abs_diff": None, "exact": False,
              "shape_mismatch": [[1, 3, 5, 2, 2], [1, 3, 22, 2, 2]]},
        tolerance=1e-6,
    )
    assert not verdict.passed
    assert any("SHAPE MISMATCH" in n for n in verdict.notes)


def test_verdict_tolerance_is_a_hard_gate():
    """Nothing in the evidence path may turn a real failure into a pass."""
    for official in (EXACT, {"max_abs_diff": 5e-3, "exact": False}):
        for cold in (EXACT, {"max_abs_diff": 5e-3, "exact": False}):
            verdict = summarize_decode_comparison(
                cold=cold,
                warm={"max_abs_diff": 1.1e-6, "exact": False},
                official_self=official,
                tolerance=1e-6,
            )
            assert verdict.passed is False


# -- v0.1 supported range --------------------------------------------------
#
# The product minimum is k=1 on the encoder's 5k+2 grid: 7 latents -> 22 frames.
# k=0 (2 latents -> 5 frames) is out of scope for v0.1.


def test_minimum_supported_clip_is_k_equals_one():
    assert MIN_SUPPORTED_LATENTS == 7
    assert MIN_SUPPORTED_FRAMES == 22
    assert PARAMS.min_supported_latents == MIN_SUPPORTED_LATENTS
    assert PARAMS.min_supported_frames == MIN_SUPPORTED_FRAMES
    # k=1 on the 5k+2 grid, and it is exactly one chunk plus the lookahead
    assert PARAMS.min_supported_latents == 5 * 1 + 2 == PARAMS.latents_needed
    assert PARAMS.min_supported_frames == 17 * 1 + 5


@pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6])
def test_streams_below_the_minimum_fail_loud(t):
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    stream.push(make_latents(t, seed=t))
    with pytest.raises(ShortVideoStreamError) as excinfo:
        stream.finish()
    message = str(excinfo.value)
    assert "{} latent(s)".format(t) in message
    assert "minimum supported clip is 7 latents / 22 frames" in message


@pytest.mark.parametrize("t", [7, 8, 12, 22, 57, 107])
def test_streams_at_or_above_the_minimum_are_accepted(t):
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    frames = sum(b.count for b in stream.push(make_latents(t, seed=t)))
    frames += sum(b.count for b in stream.finish())
    assert frames == PARAMS.total_frames(t)
    assert frames >= MIN_SUPPORTED_FRAMES


def test_the_shortest_supported_clip_is_exactly_22_frames():
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    batches = list(stream.push(make_latents(MIN_SUPPORTED_LATENTS, seed=0)))
    batches += list(stream.finish())
    assert [b.count for b in batches] == [17, 5]
    assert sum(b.count for b in batches) == MIN_SUPPORTED_FRAMES


def test_empty_stream_is_not_an_error():
    """Zero latents is a no-op, not a too-short clip."""
    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), PARAMS, concat=numpy_concat)
    assert stream.finish() == []


def test_check_stream_length_is_usable_before_any_decoding():
    """Lets a caller reject a request before spending GPU time on it."""
    PARAMS.check_stream_length(0)   # empty is fine
    PARAMS.check_stream_length(7)
    PARAMS.check_stream_length(107)
    for bad in (1, 2, 6):
        with pytest.raises(ShortVideoStreamError):
            PARAMS.check_stream_length(bad)


def test_real_probe_default_grid_is_supported():
    """The documented real grid must not contain a rejected length."""
    grid = [7, 22, 57, 107]
    for t in grid:
        PARAMS.check_stream_length(t)
        assert (t - 2) % 5 == 0, "grid should stay on the encoder's 5k+2 latents"
    assert [PARAMS.total_frames(t) for t in grid] == [22, 73, 192, 362]
