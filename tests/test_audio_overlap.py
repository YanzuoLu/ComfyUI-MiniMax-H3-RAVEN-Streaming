"""Falsification tests for the audio overlap-save planner.

The audio VAE decoder is non-causal, so a chunk decoded in isolation is *not*
a slice of the full decode.  These tests use a fake decoder whose receptive
field radius is known exactly, and assert both directions:

* margin >= radius  -> overlap-save reproduces the full decode exactly
* margin <  radius  -> it provably does not (so the margin search is measuring
  something real, not a tautology)
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from raven_streaming.media.audio_stream import (  # noqa: E402
    AudioLatentGeometry,
    OverlapSaveAudioDecoder,
    OverlapSavePlanner,
    decode_overlap_save,
    diff_block_by_block,
    diff_stats,
    max_abs_diff,
    probe_shift_equivariance,
    saturation_stats,
    search_latent_margin,
)
from raven_streaming.media.fakes import (  # noqa: E402
    FiniteRFAudioDecoder,
    JitteryAudioDecoder,
    NondeterministicAudioDecoder,
    PhaseSensitiveAudioDecoder,
    make_audio_latents,
)

SPL = 8  # samples per latent in the fake
GEOM = AudioLatentGeometry(samples_per_latent=SPL, sample_rate=SPL * 40)
REAL_GEOM = AudioLatentGeometry()


# -- geometry --------------------------------------------------------------


def test_real_geometry_matches_the_official_audio_vae():
    assert REAL_GEOM.samples_per_latent == 800
    assert REAL_GEOM.sample_rate == 32000
    assert REAL_GEOM.latents_per_second == 40
    assert REAL_GEOM.latents_to_samples(40) == 32000
    assert REAL_GEOM.latents_to_seconds(40) == 1


def test_geometry_from_duck_typed_vae():
    class Model(object):
        hop_length = 800
        samples_per_latent = 800
        sample_rate = 32000

    assert AudioLatentGeometry.from_vae(Model()) == REAL_GEOM


# -- planner ---------------------------------------------------------------


@pytest.mark.parametrize("total", list(range(1, 40)))
@pytest.mark.parametrize("block", [1, 3, 5, 40])
@pytest.mark.parametrize("margin", [0, 1, 3])
def test_plan_tiles_the_output_exactly_once(total, block, margin):
    plan = OverlapSavePlanner(margin, block, GEOM).plan(total)
    cursor = 0
    for req in plan:
        assert req.out_start == cursor
        assert req.take_start >= 0
        assert req.take_stop <= req.decoded_samples
        assert req.out_samples > 0
        assert 0 <= req.latent_start <= req.latent_stop <= total
        cursor = req.out_stop
    assert cursor == total * SPL
    assert plan[-1].is_final


def test_plan_context_is_clamped_at_the_stream_edges():
    plan = OverlapSavePlanner(margin=3, block_latents=5, geometry=GEOM).plan(20)
    first, last = plan[0], plan[-1]
    # no left context available at the very start; none dropped either
    assert first.latent_start == 0
    assert first.left_context_samples == 0
    assert first.right_context_samples == 3 * SPL
    # no right context at the very end
    assert last.latent_stop == 20
    assert last.right_context_samples == 0
    assert last.left_context_samples == 3 * SPL


def test_streaming_plan_equals_offline_plan():
    for margin in (0, 1, 2, 5):
        for block in (1, 4, 5):
            for total in (1, 5, 7, 13, 40, 41):
                offline = OverlapSavePlanner(margin, block, GEOM).plan(total)
                streaming = OverlapSavePlanner(margin, block, GEOM)
                got = []
                for _ in range(total):
                    got.extend(streaming.push(1))
                got.extend(streaming.finish())
                assert [(r.latent_start, r.latent_stop, r.take_start, r.take_stop, r.out_start)
                        for r in got] == [
                    (r.latent_start, r.latent_stop, r.take_start, r.take_stop, r.out_start)
                    for r in offline
                ]


def test_lookahead_is_reported_in_real_units():
    planner = OverlapSavePlanner(margin=6, block_latents=5, geometry=REAL_GEOM)
    assert planner.lookahead_latents == 6
    assert planner.lookahead_samples == 4800
    assert planner.lookahead_seconds == Fraction(6, 40)
    assert float(planner.lookahead_seconds) == pytest.approx(0.15)


def test_a_block_is_only_emitted_once_its_right_context_exists():
    planner = OverlapSavePlanner(margin=3, block_latents=5, geometry=GEOM)
    emitted = [len(planner.push(1)) for _ in range(10)]
    # block 0 covers latents [0,5) but needs latents up to index 7 -> 8 pushes
    assert emitted == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]


# -- correctness against a known receptive field ---------------------------


@pytest.mark.parametrize("radius", [0, 1, 2, 5])
@pytest.mark.parametrize("block", [1, 3, 5, 7])
def test_overlap_save_is_exact_when_margin_covers_the_receptive_field(radius, block):
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=radius)
    z = make_audio_latents(37, channels=decoder.latent_channels, seed=radius)
    full = decoder(z)
    streamed = decode_overlap_save(decoder, z, radius, block, GEOM, concat=np.concatenate)
    assert streamed.shape == full.shape
    assert max_abs_diff(streamed, full) == 0.0


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_insufficient_margin_provably_breaks(radius):
    """Without enough lookahead the seams are wrong - this is the falsification."""
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=radius)
    z = make_audio_latents(37, channels=decoder.latent_channels, seed=radius)
    full = decoder(z)
    for margin in range(0, radius):
        streamed = decode_overlap_save(decoder, z, margin, 5, GEOM, concat=np.concatenate)
        assert max_abs_diff(streamed, full) > 1e-6, "margin {} should be too small".format(margin)


def test_zero_lookahead_is_not_enough_for_a_non_causal_decoder():
    decoder = FiniteRFAudioDecoder(3, samples_per_latent=SPL, seed=0)
    z = make_audio_latents(20, channels=decoder.latent_channels, seed=0)
    naive = np.concatenate([decoder(z[..., i:i + 5]) for i in range(0, 20, 5)], axis=-1)
    assert max_abs_diff(naive, decoder(z)) > 1e-6


# -- streaming driver ------------------------------------------------------


@pytest.mark.parametrize("push_size", [1, 2, 5, 13])
def test_streaming_decoder_matches_full_decode(push_size):
    radius = 3
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=1)
    z = make_audio_latents(41, channels=decoder.latent_channels, seed=1)
    full = decoder(z)

    stream = OverlapSaveAudioDecoder(decoder, radius, 5, GEOM, concat=np.concatenate)
    parts = []
    for i in range(0, z.shape[-1], push_size):
        parts.extend(stream.push(z[..., i:i + push_size]))
    parts.extend(stream.finish())
    got = np.concatenate(parts, axis=-1)

    assert got.shape == full.shape
    assert max_abs_diff(got, full) == 0.0
    assert stream.samples_emitted == full.shape[-1]


def test_streaming_decoder_keeps_only_a_bounded_history():
    radius, block = 3, 5
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=2)
    z = make_audio_latents(200, channels=decoder.latent_channels, seed=2)
    stream = OverlapSaveAudioDecoder(decoder, radius, block, GEOM, concat=np.concatenate)
    peak = 0
    for i in range(z.shape[-1]):
        stream.push(z[..., i:i + 1])
        peak = max(peak, int(stream._buffer.shape[-1]))
    stream.finish()
    # worst case: one block + both margins still in flight
    assert peak <= block + 2 * radius + 1


# -- abort -----------------------------------------------------------------
#
# The cancel path. `finish` exists to decode the edge blocks -- the last
# `block - 1 + margin` latents, the ones still short of their right context.
# On a cancel that is work spent on samples nobody will hear, so `abort` closes
# the stream without it and drops the overlap-save history with it.


def test_abort_drops_the_buffer_without_decoding_anything():
    radius, block = 3, 5
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=4)
    z = make_audio_latents(23, channels=decoder.latent_channels, seed=4)

    stream = OverlapSaveAudioDecoder(decoder, radius, block, GEOM, concat=np.concatenate)
    emitted = stream.push(z)
    # blocks 0..3 are complete; latents 17..22 are held for the edge block
    assert len(emitted) == 4
    assert stream._buffer is not None and int(stream._buffer.shape[-1]) > 0
    calls, samples = decoder.calls, stream.samples_emitted

    stream.abort()

    assert stream._buffer is None
    assert stream.finished and stream.planner.finished
    assert decoder.calls == calls  # no edge block, no PCM
    assert stream.samples_emitted == samples  # the record survives; the buffer does not


def test_abort_is_terminal_in_both_directions():
    decoder = FiniteRFAudioDecoder(2, samples_per_latent=SPL, seed=5)
    z = make_audio_latents(19, channels=decoder.latent_channels, seed=5)
    stream = OverlapSaveAudioDecoder(decoder, 2, 5, GEOM, concat=np.concatenate)
    stream.push(z)
    stream.abort()
    calls = decoder.calls

    assert stream.finish() == []
    with pytest.raises(RuntimeError):
        stream.push(z[..., :1])
    # the rejected push left nothing behind either
    assert stream._buffer is None
    assert decoder.calls == calls


def test_abort_is_idempotent_and_harmless_after_finish():
    decoder = FiniteRFAudioDecoder(2, samples_per_latent=SPL, seed=6)
    z = make_audio_latents(19, channels=decoder.latent_channels, seed=6)
    stream = OverlapSaveAudioDecoder(decoder, 2, 5, GEOM, concat=np.concatenate)
    stream.push(z)
    stream.finish()
    calls, samples = decoder.calls, stream.samples_emitted
    assert samples == 19 * SPL  # the whole clip was emitted before the abort

    stream.abort()
    stream.abort()

    assert decoder.calls == calls
    assert stream.samples_emitted == samples
    assert stream._buffer is None and stream.finished


def test_planner_abort_emits_no_edge_blocks():
    planner = OverlapSavePlanner(margin=3, block_latents=5, geometry=GEOM)
    planner.push(23)
    done = planner.blocks_done
    assert planner.plan(23)[done:], "there were edge blocks left to plan"

    planner.abort()

    assert planner.finished
    assert planner.finish() == []
    assert planner.blocks_done == done  # nothing was planned on the way out
    with pytest.raises(RuntimeError):
        planner.push(1)


# -- margin search ---------------------------------------------------------


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 7])
def test_margin_search_finds_the_true_receptive_field(radius):
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=radius)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=radius)
    result = search_latent_margin(
        decoder, z, decoder(z), tolerance=1e-12, block_latents=5,
        max_margin=32, geometry=GEOM, concat=np.concatenate,
    )
    assert result.found
    assert result.margin == radius
    assert result.lookahead_samples == radius * SPL
    assert "minimum margin" in result.describe()


def test_margin_search_reports_failure_when_lookahead_is_insufficient():
    decoder = FiniteRFAudioDecoder(9, samples_per_latent=SPL, seed=5)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=5)
    result = search_latent_margin(
        decoder, z, decoder(z), tolerance=1e-12, block_latents=5,
        max_margin=4, geometry=GEOM, concat=np.concatenate,
    )
    assert not result.found
    assert result.margin is None
    assert "NOT enough" in result.describe()


def test_margin_search_honours_a_loose_tolerance():
    """A tolerance larger than the seam error accepts a smaller margin."""
    decoder = FiniteRFAudioDecoder(4, samples_per_latent=SPL, seed=6)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=6)
    ref = decoder(z)
    tight = search_latent_margin(
        decoder, z, ref, tolerance=1e-12, block_latents=5,
        max_margin=32, geometry=GEOM, concat=np.concatenate,
    )
    err_at_2 = max_abs_diff(
        decode_overlap_save(decoder, z, 2, 5, GEOM, concat=np.concatenate), ref
    )
    loose = search_latent_margin(
        decoder, z, ref, tolerance=err_at_2, block_latents=5,
        max_margin=32, geometry=GEOM, concat=np.concatenate,
    )
    assert tight.margin == 4
    assert loose.found and loose.margin is not None and loose.margin <= 2


# -- diagnostics: localise the error instead of collapsing it --------------


def test_full_sequence_control_is_exact_by_construction():
    """margin >= total makes every block decode the whole z, then slice it.

    There is no overlap-save left in that configuration, so a non-zero result
    can only mean nondeterminism or a harness bug.  This is the decisive
    control the real-VAE probe leans on.
    """
    total = 37
    for radius in (0, 3, 9):
        decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=radius)
        z = make_audio_latents(total, channels=decoder.latent_channels, seed=radius)
        full = decoder(z)
        for block in (5, 28, 29):
            streamed = decode_overlap_save(
                decoder, z, total, block, GEOM, concat=np.concatenate
            )
            assert max_abs_diff(streamed, full) == 0.0


def test_margin_search_always_records_the_full_sequence_control():
    decoder = FiniteRFAudioDecoder(9, samples_per_latent=SPL, seed=3)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=3)
    result = search_latent_margin(
        decoder, z, decoder(z), tolerance=1e-12, block_latents=28,
        max_margin=4, geometry=GEOM, concat=np.concatenate,
    )
    assert not result.found  # 4 < radius 9
    assert result.total_latents == 60
    assert result.full_sequence_error == 0.0
    assert result.full_sequence_is_exact is True
    assert "exact, as it must be" in result.describe()
    # the control must never be mistaken for a usable margin
    assert all(m <= 4 for m, _ in result.searched_errors)


def test_block_by_block_diff_localises_the_error():
    radius, block = 4, 10
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=8)
    z = make_audio_latents(50, channels=decoder.latent_channels, seed=8)
    full = decoder(z)

    diffs = diff_block_by_block(decoder, z, full, margin=0, block_latents=block,
                                geometry=GEOM)
    assert len(diffs) == 5
    assert all(d.request.out_start == i * block * SPL for i, d in enumerate(diffs))
    # with no margin every interior block is wrong, and we can say where
    assert any(d.max_abs_diff > 0 for d in diffs)
    for d in diffs:
        if not d.exact:
            assert 0 <= d.worst_sample_in_block < d.request.out_samples
            assert d.worst_sample_absolute == d.request.out_start + d.worst_sample_in_block
    # nothing here decoded the whole sequence
    assert not any(d.is_full_sequence for d in diffs)

    good = diff_block_by_block(decoder, z, full, margin=radius, block_latents=block,
                               geometry=GEOM)
    assert all(d.exact for d in good)


def test_block_by_block_flags_full_sequence_requests():
    total, block = 30, 10
    decoder = FiniteRFAudioDecoder(2, samples_per_latent=SPL, seed=1)
    z = make_audio_latents(total, channels=decoder.latent_channels, seed=1)
    diffs = diff_block_by_block(decoder, z, decoder(z), margin=total,
                                block_latents=block, geometry=GEOM)
    assert diffs and all(d.is_full_sequence for d in diffs)
    assert all(d.exact for d in diffs)
    assert all("FULL-SEQUENCE" in d.describe() for d in diffs)
    assert all(d.to_dict()["is_full_sequence"] for d in diffs)


def test_edge_blocks_are_marked():
    decoder = FiniteRFAudioDecoder(2, samples_per_latent=SPL, seed=2)
    z = make_audio_latents(30, channels=decoder.latent_channels, seed=2)
    diffs = diff_block_by_block(decoder, z, decoder(z), margin=2, block_latents=10,
                                geometry=GEOM)
    assert diffs[0].touches_stream_start and not diffs[0].touches_stream_end
    assert diffs[-1].touches_stream_end and not diffs[-1].touches_stream_start
    assert not diffs[1].touches_stream_start and not diffs[1].touches_stream_end


# -- nondeterminism vs receptive field -------------------------------------


def test_shape_keyed_jitter_plateaus_non_monotonically_but_control_stays_exact():
    """Fingerprint A: cuDNN autotune keyed on input shape.

    Growing the margin never converges and the error wobbles, yet the
    full-sequence control is exact - because there every block decodes a tensor
    of the same length as the reference.
    """
    decoder = JitteryAudioDecoder(0, samples_per_latent=SPL, seed=4, jitter=1e-2)
    z = make_audio_latents(120, channels=decoder.latent_channels, seed=4)
    reference = decoder(z)

    result = search_latent_margin(
        decoder, z, reference, tolerance=1e-5, block_latents=28,
        max_margin=64, geometry=GEOM, concat=np.concatenate,
    )
    assert not result.found
    errors = [e for _, e in result.searched_errors]
    assert min(errors) > 1e-5, "jitter never falls below tolerance"
    assert not result.is_monotone
    assert "NON-MONOTONE" in result.describe()
    # the control IS exact -> the planner is fine, the decode is shape-keyed
    assert result.full_sequence_error == 0.0
    assert "NOT an overlap-save" not in result.describe()


def test_run_to_run_nondeterminism_breaks_even_the_full_sequence_control():
    """Fingerprint B: the same tensor decoded twice gives different answers."""
    decoder = NondeterministicAudioDecoder(0, samples_per_latent=SPL, seed=4, jitter=1e-2)
    z = make_audio_latents(120, channels=decoder.latent_channels, seed=4)
    reference = decoder(z)

    result = search_latent_margin(
        decoder, z, reference, tolerance=1e-5, block_latents=28,
        max_margin=64, geometry=GEOM, concat=np.concatenate,
    )
    assert not result.found
    assert result.full_sequence_error is not None and result.full_sequence_error > 0.0
    assert result.full_sequence_is_exact is False
    text = result.describe()
    assert "NOT an overlap-save" in text
    assert "nondeterministic" in text

    # and a block-by-block view names the offending full-sequence requests
    diffs = diff_block_by_block(decoder, z, reference, margin=120,
                                block_latents=28, geometry=GEOM)
    assert all(d.is_full_sequence for d in diffs)
    assert any(not d.exact for d in diffs)


def test_a_receptive_field_result_is_monotone_and_converges():
    decoder = FiniteRFAudioDecoder(6, samples_per_latent=SPL, seed=4)
    z = make_audio_latents(120, channels=decoder.latent_channels, seed=4)
    result = search_latent_margin(
        decoder, z, decoder(z), tolerance=1e-12, block_latents=28,
        max_margin=64, geometry=GEOM, concat=np.concatenate,
    )
    assert result.found and result.margin == 6
    assert result.full_sequence_error == 0.0
    assert "NOT an overlap-save" not in result.describe()


# -- phase -----------------------------------------------------------------


def _phase_decode_fn(decoder, z):
    """Wire latent_start into the phase-sensitive fake, as a real slice would."""
    total = int(z.shape[-1])

    def fn(chunk):
        n = int(chunk.shape[-1])
        # locate the chunk inside z by identity of its first column
        for start in range(0, total - n + 1):
            if np.array_equal(np.asarray(chunk), np.asarray(z[..., start:start + n])):
                decoder.latent_start = start
                break
        return decoder(chunk)

    return fn


def test_phase_sensitive_decoder_defeats_any_margin():
    """Guard the guard: a phase-dependent decoder cannot be fixed by context."""
    decoder = PhaseSensitiveAudioDecoder(2, period=2, samples_per_latent=SPL, seed=0)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=0)
    fn = _phase_decode_fn(decoder, z)
    decoder.latent_start = 0
    reference = decoder(z)

    # block 5 with an odd margin lands on odd latent starts -> broken
    broken = decode_overlap_save(fn, z, 3, 5, GEOM, concat=np.concatenate)
    assert max_abs_diff(broken, reference) > 1e-6


def test_phase_align_repairs_a_phase_sensitive_decoder():
    decoder = PhaseSensitiveAudioDecoder(2, period=2, samples_per_latent=SPL, seed=0)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=0)
    fn = _phase_decode_fn(decoder, z)
    decoder.latent_start = 0
    reference = decoder(z)

    aligned = decode_overlap_save(fn, z, 3, 5, GEOM, concat=np.concatenate,
                                  phase_align=2)
    assert max_abs_diff(aligned, reference) == 0.0


def test_phase_align_only_ever_adds_left_context():
    for phase in (1, 2, 4, 5):
        plain = OverlapSavePlanner(3, 7, GEOM).plan(60)
        aligned = OverlapSavePlanner(3, 7, GEOM, phase_align=phase).plan(60)
        assert len(plain) == len(aligned)
        for a, b in zip(plain, aligned):
            assert b.latent_start <= a.latent_start        # never less context
            assert b.latent_start % phase == 0
            assert b.out_start == a.out_start              # same output tiling
            assert b.out_samples == a.out_samples


def test_shift_equivariance_probe_detects_phase_dependence():
    z = make_audio_latents(60, channels=4, seed=0)

    clean = FiniteRFAudioDecoder(2, samples_per_latent=SPL, seed=0)
    results = probe_shift_equivariance(clean, z, 20, 50, shifts=(1, 2, 3),
                                       geometry=GEOM, inset_latents=4)
    assert results and all(r["exact"] for r in results), "conv stacks are shift-equivariant"

    phased = PhaseSensitiveAudioDecoder(2, period=2, samples_per_latent=SPL, seed=0)
    results = probe_shift_equivariance(_phase_decode_fn(phased, z), z, 20, 50,
                                       shifts=(1, 2, 3), geometry=GEOM, inset_latents=4)
    by_shift = {r["shift_latents"]: r for r in results}
    assert not by_shift[1]["exact"], "an odd shift must break a period-2 decoder"
    assert by_shift[2]["exact"], "an even shift must not"


def test_shift_equivariance_probe_is_length_controlled():
    """A length-keyed decoder must NOT be misreported as phase dependent.

    The two failure modes need different fixes (pin the decode length vs align
    the slice start), so the probe slides a fixed-size window instead of growing
    one.
    """
    z = make_audio_latents(60, channels=4, seed=0)
    jittery = JitteryAudioDecoder(2, samples_per_latent=SPL, seed=0, jitter=1e-2)

    results = probe_shift_equivariance(jittery, z, 20, 50, shifts=(1, 2, 3),
                                       geometry=GEOM, inset_latents=4)
    assert results
    assert all(r["window_latents"] == 30 for r in results), "window size must be fixed"
    assert all(r["exact"] for r in results), (
        "length-keyed jitter cancels when both windows have the same shape")

    # ... while its margin search still fails, which is the correct attribution
    result = search_latent_margin(
        jittery, z, jittery(z), tolerance=1e-5, block_latents=10,
        max_margin=16, geometry=GEOM, concat=np.concatenate,
    )
    assert not result.found


# -- prefix (cumulative) decode -------------------------------------------


@pytest.mark.parametrize("radius", [0, 2, 5])
@pytest.mark.parametrize("block", [5, 28, 29])
def test_prefix_mode_removes_all_left_boundary_error(radius, block):
    """Decoding z[0:hi] every time makes the left edge the true stream start."""
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=radius)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=radius)
    full = decoder(z)
    streamed = decode_overlap_save(decoder, z, radius, block, GEOM,
                                   concat=np.concatenate, prefix_mode=True)
    assert max_abs_diff(streamed, full) == 0.0


def test_prefix_mode_needs_no_left_margin_at_all():
    """Only the right margin remains; left context is exact by construction."""
    radius = 4
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=1)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=1)
    full = decoder(z)

    plan = OverlapSavePlanner(radius, 10, GEOM, prefix_mode=True).plan(60)
    assert all(r.latent_start == 0 for r in plan)

    streamed = decode_overlap_save(decoder, z, radius, 10, GEOM,
                                   concat=np.concatenate, prefix_mode=True)
    assert max_abs_diff(streamed, full) == 0.0

    # and it also survives a phase-sensitive decoder, since every slice starts at 0
    phased = PhaseSensitiveAudioDecoder(2, period=2, samples_per_latent=SPL, seed=0)
    fn = _phase_decode_fn(phased, z)
    phased.latent_start = 0
    ref = phased(z)
    got = decode_overlap_save(fn, z, 2, 10, GEOM, concat=np.concatenate, prefix_mode=True)
    assert max_abs_diff(got, ref) == 0.0


def test_prefix_mode_streaming_history_is_the_whole_stream():
    planner = OverlapSavePlanner(3, 10, GEOM, prefix_mode=True)
    planner.push(40)
    assert planner.required_history_latents() == 40  # O(n) memory, O(n^2) work


def test_diff_stats_reports_position():
    a = np.zeros((1, 2, 20))
    b = np.zeros((1, 2, 20))
    b[0, 1, 13] = 0.5
    stats = diff_stats(a, b)
    assert stats["max_abs_diff"] == 0.5
    assert stats["worst_sample"] == 13
    assert stats["exact"] is False
    assert diff_stats(a, a)["exact"] is True


# -- harness confounds: clipping and stereo -------------------------------


def test_saturation_stats_detects_a_railed_reference():
    clean = np.stack([np.sin(np.linspace(0, 20, 4000)) * 0.4] * 2)[None]
    stats = saturation_stats(clean)
    assert stats["fraction_at_rail"] == 0.0
    assert not stats["is_saturated"]
    assert 0.0 < stats["rms"] < 1.0
    assert stats["peak_abs"] <= 1.0

    railed = np.clip(np.stack([np.sin(np.linspace(0, 20, 4000)) * 4.0] * 2)[None], -1, 1)
    stats = saturation_stats(railed)
    assert stats["is_saturated"]
    assert stats["fraction_at_rail"] > 0.5
    assert stats["peak_abs"] == pytest.approx(1.0)
    assert stats["samples"] == railed.size


def test_clipping_can_fake_a_plateau():
    """Why the probe refuses to read a margin off a clipped reference.

    Once the clamp is applied, growing the context stops changing the railed
    regions at all, so the error can sit flat for reasons unrelated to the
    receptive field.
    """
    radius = 6
    decoder = FiniteRFAudioDecoder(radius, samples_per_latent=SPL, seed=0)
    z = make_audio_latents(60, channels=decoder.latent_channels, seed=0) * 50.0

    def clipped(chunk):
        return np.clip(decoder(chunk), -1.0, 1.0)

    reference = clipped(z)
    assert saturation_stats(reference)["is_saturated"]

    errors = [
        max_abs_diff(
            decode_overlap_save(clipped, z, m, 10, GEOM, concat=np.concatenate), reference
        )
        for m in range(0, radius)
    ]
    # the true margin still works, but the approach to it is flattened by the clamp
    assert max_abs_diff(
        decode_overlap_save(clipped, z, radius, 10, GEOM, concat=np.concatenate), reference
    ) == 0.0
    assert len(set(round(e, 6) for e in errors)) < len(errors), (
        "clipping should collapse distinct error levels into a plateau")


def test_overlap_save_keeps_the_stereo_axis_intact():
    """A wrong permute/reshape would swap or mix the two channels."""
    decoder = FiniteRFAudioDecoder(3, samples_per_latent=SPL, seed=0)
    z = make_audio_latents(40, channels=decoder.latent_channels, seed=0)
    # make the two stereo channels unmistakably different
    z[:, :, 1, :] = z[:, :, 0, :] * -3.0

    full = decoder(z)
    streamed = decode_overlap_save(decoder, z, 3, 7, GEOM, concat=np.concatenate)
    assert streamed.shape == full.shape
    assert max_abs_diff(streamed, full) == 0.0

    # channels must not be interchangeable: a swap has to be detectable
    swapped = full[:, ::-1, :]
    assert max_abs_diff(swapped, full) > 1e-6
    # and each channel must match its own reference independently
    for ch in (0, 1):
        assert max_abs_diff(streamed[:, ch:ch + 1, :], full[:, ch:ch + 1, :]) == 0.0


def test_block_diffs_are_reported_per_stereo_pair_not_flattened():
    decoder = FiniteRFAudioDecoder(3, samples_per_latent=SPL, seed=1)
    z = make_audio_latents(40, channels=decoder.latent_channels, seed=1)
    z[:, :, 1, :] *= 2.0
    diffs = diff_block_by_block(decoder, z, decoder(z), margin=0, block_latents=10,
                                geometry=GEOM)
    for d in diffs:
        # worst_sample is a position on the time axis, never a flattened index
        if d.worst_sample_in_block is not None:
            assert 0 <= d.worst_sample_in_block < d.request.out_samples
