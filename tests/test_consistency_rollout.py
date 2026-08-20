"""The chunk-major loop itself: call order, cache order, RNG order, cancellation.

Everything here runs against the fakes in ``test_consistency_common.py``, so it
is exact rather than approximate: the sampler's output is compared *tensor for
tensor* against a second, longhand implementation of RAVEN's rollout. That is
what makes the claims falsifiable -- if the draw order, the ``x0`` conversion or
the ``(1 - s) x0 + s eps`` step changed, the comparison would fail, not drift.

No ComfyUI, no weights, no GPU.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import consistency  # noqa: E402
from raven_streaming.consistency import (  # noqa: E402
    CANCEL_POINTS,
    SamplerConfig,
    SamplerError,
    SamplingCancelled,
    sample_streaming,
    step_pairs,
)
from test_consistency_common import (  # noqa: E402
    FakeCausalDiT,
    FakePatcher,
    LoadRecorder,
    empty_av_latent,
    reference_rollout,
    text_conditioning,
)


def run(
    *,
    config=None,
    frames=22,
    on_chunk=None,
    cancel_check=None,
    dit=None,
    patcher=None,
    load=None,
    text_len=5,
    latent=None,
):
    dit = dit or FakeCausalDiT(num_layers=2)
    patcher = patcher or FakePatcher(dit)
    load = load or LoadRecorder()
    result = sample_streaming(
        model=patcher,
        positive=text_conditioning(text_len=text_len),
        latent=empty_av_latent(frames=frames) if latent is None else latent,
        config=config or SamplerConfig(steps=4, seed=1234),
        on_chunk=on_chunk,
        cancel_check=cancel_check,
        load_models=load,
        require_upstream_class=False,
    )
    return result, dit, patcher, load


# --------------------------------------------------------------------------
# call trace
# --------------------------------------------------------------------------
def test_call_trace_is_chunk_major_with_text_first():
    result, dit, _, _ = run()
    kinds = [(c.kind, c.chunk_index) for c in dit.calls]
    assert kinds == [
        ("prefill", None),
        ("noise", 0), ("noise", 0), ("noise", 0), ("noise", 0),
        ("clean", 0),
        ("noise", 1), ("noise", 1), ("noise", 1), ("noise", 1),
        # no clean on the last chunk
    ]
    assert result.noise_forwards == 8
    assert result.clean_forwards == 1
    assert result.num_chunks == 2


def test_last_chunk_is_never_written_to_the_cache():
    _, dit, _, _ = run(frames=39)          # 3 chunks
    cleans = [c.chunk_index for c in dit.calls if c.kind == "clean"]
    assert cleans == [0, 1]                # never 2


def test_prefill_runs_exactly_once_per_rollout():
    _, dit, _, _ = run(frames=39)
    assert sum(1 for c in dit.calls if c.kind == "prefill") == 1
    assert dit.calls[0].kind == "prefill"


def test_text_is_cache_chunk_zero_and_every_clean_appends_one():
    _, dit, _, _ = run(frames=39, config=SamplerConfig(steps=2, seed=5))
    prefill = dit.calls[0]
    assert prefill.committed_before == 0
    for call in dit.calls[1:]:
        # text + one clean fill per completed chunk before this one
        assert call.committed_before == call.chunk_index + 1


def test_sink_and_window_evict_the_way_the_cache_policy_says():
    # sink=2 pins text + chunk 0; window=1 keeps only the newest
    _, dit, _, _ = run(
        frames=73,  # k=4 -> 22 video latents -> 5 chunks
        config=SamplerConfig(steps=1, seed=2, sink=2, window=1),
    )
    retained = {
        c.chunk_index: list(c.retained_before)
        for c in dit.calls if c.kind == "noise"
    }
    assert retained[0] == [0]              # the text only
    assert retained[1] == [0, 1]           # text + chunk 0 (both in the sink)
    assert retained[2] == [0, 1, 2]        # sink + newest
    assert retained[3] == [0, 1, 3]        # chunk 2's record evicted
    assert retained[4] == [0, 1, 4]


def test_noise_forwards_carry_the_two_independent_shifted_grids():
    config = SamplerConfig(steps=4, seed=9, video_shift=12.0, audio_shift=3.0)
    _, dit, _, _ = run(config=config)
    video_sigmas = [c.video_sigma for c in dit.calls if c.kind == "noise"]
    audio_sigmas = [c.audio_sigma for c in dit.calls if c.kind == "noise"]
    assert video_sigmas == list(config.video_sigmas) * 2
    assert audio_sigmas == list(config.audio_sigmas) * 2


def test_clean_fill_carries_eps_and_no_sigma():
    _, dit, _, _ = run()
    clean = [c for c in dit.calls if c.kind == "clean"][0]
    assert clean.update_cache is True
    assert clean.has_video_eps and clean.has_audio_eps
    assert clean.video_sigma is None and clean.audio_sigma is None


def test_chunk_shapes_follow_the_layout():
    result, dit, _, _ = run()
    chunks = result.layout.chunks
    for call in dit.calls:
        if call.chunk_index is None:
            continue
        chunk = chunks[call.chunk_index]
        assert call.video_shape == (1, 24, chunk.video_latents, 2, 2)
        assert call.audio_shape == (1, 32, 2, chunk.audio_latents)


# --------------------------------------------------------------------------
# model loading / transformer_options
# --------------------------------------------------------------------------
def test_load_models_gpu_is_called_once_before_any_forward():
    calls = []

    class Recorder(LoadRecorder):
        def __call__(self, *args, **kwargs):
            calls.append(("load", len(dit.calls)))
            super().__call__(*args, **kwargs)

    dit = FakeCausalDiT()
    load = Recorder()
    _result, dit, patcher, _ = run(dit=dit, load=load)

    assert len(load.calls) == 1
    args, kwargs = load.calls[0]
    assert args[0] == [patcher]
    assert kwargs == {"memory_required": 0, "force_full_load": False}
    # ... and it happened before the first model call
    assert calls == [("load", 0)]


def test_every_call_reuses_one_transformer_options_object():
    _, dit, _, _ = run(frames=39)
    ids = {c.options_id for c in dit.calls}
    assert len(ids) == 1


def test_one_compute_dtype_is_used_for_the_prefill_and_every_chunk():
    """The cache is filled and read in one dtype, so it must be decided once.

    ``forward_chunk`` would otherwise default to the DiT's dtype while
    ``prefill_text`` defaults to the *context's*, and the attention module
    refuses a cache whose K/V dtype differs from the current chunk's.
    """
    dit = FakeCausalDiT(dtype=torch.bfloat16)
    _result, dit, _, _ = run(dit=dit, frames=39)
    dtypes = {c.compute_dtype for c in dit.calls}
    assert dtypes == {torch.bfloat16}


def test_an_explicit_compute_dtype_wins():
    dit = FakeCausalDiT(dtype=torch.bfloat16)
    sample_streaming(
        model=FakePatcher(dit),
        positive=text_conditioning(),
        latent=empty_av_latent(),
        config=SamplerConfig(steps=1, seed=1),
        load_models=LoadRecorder(),
        compute_dtype=torch.float32,
        require_upstream_class=False,
    )
    assert {c.compute_dtype for c in dit.calls} == {torch.float32}


def test_a_model_without_a_dtype_falls_back_to_the_latent_dtype():
    _result, dit, _, _ = run()
    assert {c.compute_dtype for c in dit.calls} == {torch.float32}


def test_transformer_options_is_a_copy_of_the_patchers_and_carries_its_content():
    dit = FakeCausalDiT()
    original = {"marker": object()}
    patcher = FakePatcher(dit, transformer_options=original)
    run(dit=dit, patcher=patcher)
    seen = {c.options_id for c in dit.calls}
    assert len(seen) == 1
    assert seen != {id(original)}          # never the patcher's own dict
    assert original == {"marker": original["marker"]}   # and it was not mutated


# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------
def test_draw_order_and_shapes_are_the_rollout_contract():
    result, _, _, _ = run(config=SamplerConfig(steps=2, seed=77))
    layout = result.layout
    video_full = tuple(layout.video_latent_shape(24))
    audio_full = tuple(layout.audio_latent_shape(32))

    expected = [
        ("video_initial_noise", video_full),
        ("audio_initial_noise", audio_full),
        ("video_clean_eps", video_full),
        ("audio_clean_eps", audio_full),
    ]
    for chunk in layout.chunks:
        for _ in range(2):
            expected.append(("video_step_eps", (1, 24, chunk.video_latents, 2, 2)))
            expected.append(("audio_step_eps", (1, 32, 2, chunk.audio_latents)))

    assert [(d.label, d.shape) for d in result.draws] == expected


def test_the_four_full_clip_draws_precede_every_step_draw():
    result, _, _, _ = run()
    labels = [d.label for d in result.draws]
    assert labels[:4] == [
        "video_initial_noise", "audio_initial_noise",
        "video_clean_eps", "audio_clean_eps",
    ]
    assert "video_initial_noise" not in labels[4:]
    assert "video_clean_eps" not in labels[4:]


def test_step_draws_happen_even_when_the_next_sigma_is_zero():
    # steps=1 -> the only step's next sigma is 0, so the eps is multiplied away
    result, _, _, _ = run(config=SamplerConfig(steps=1, seed=3))
    step_draws = [d for d in result.draws if d.label.endswith("step_eps")]
    assert len(step_draws) == 2 * result.num_chunks
    assert step_pairs(result.config.video_sigmas)[-1][1] == 0.0


def test_a_zeroed_step_still_advances_the_stream():
    """Two chunks with steps=1: chunk 1's noise must be the *5th/6th* draw.

    If the s == 0 draw were skipped as an optimisation, chunk 1 would silently
    receive different noise, so this pins the advance rather than the value.
    """
    result, _, _, _ = run(config=SamplerConfig(steps=1, seed=42))
    generator = torch.Generator().manual_seed(42)
    for draw in result.draws:
        torch.randn(draw.shape, generator=generator)
    # the replay consumed exactly the same draws in the same order
    replay = torch.Generator().manual_seed(42)
    for draw in result.draws:
        torch.randn(draw.shape, generator=replay)
    assert torch.equal(generator.get_state(), replay.get_state())
    assert len(result.draws) == 4 + 2 * result.num_chunks


def test_output_matches_an_independent_replay_of_ravens_rollout():
    config = SamplerConfig(steps=4, seed=20260807)
    result, _, _, _ = run(config=config)
    video, audio = reference_rollout(result.layout, config)
    streams = result.latent["samples"].unbind()
    assert torch.equal(streams[0], video)
    assert torch.equal(streams[1], audio)


def test_output_matches_the_replay_for_a_three_chunk_clip():
    config = SamplerConfig(steps=2, seed=11)
    result, _, _, _ = run(frames=39, config=config)
    video, audio = reference_rollout(result.layout, config)
    streams = result.latent["samples"].unbind()
    assert torch.equal(streams[0], video)
    assert torch.equal(streams[1], audio)


def test_same_seed_is_bit_identical():
    first, _, _, _ = run(config=SamplerConfig(steps=3, seed=5))
    second, _, _, _ = run(config=SamplerConfig(steps=3, seed=5))
    a = first.latent["samples"].unbind()
    b = second.latent["samples"].unbind()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_a_different_seed_changes_the_result():
    first, _, _, _ = run(config=SamplerConfig(steps=3, seed=5))
    second, _, _, _ = run(config=SamplerConfig(steps=3, seed=6))
    a = first.latent["samples"].unbind()
    b = second.latent["samples"].unbind()
    assert not torch.equal(a[0], b[0])
    assert not torch.equal(a[1], b[1])


def test_the_rollout_never_touches_global_rng():
    torch.manual_seed(999)
    before = torch.get_rng_state()
    run()
    assert torch.equal(before, torch.get_rng_state())


# --------------------------------------------------------------------------
# output latent
# --------------------------------------------------------------------------
def test_output_latent_has_the_input_structure_and_dtype():
    latent = empty_av_latent(frames=22)
    dit = FakeCausalDiT()
    patcher = FakePatcher(dit)
    result = sample_streaming(
        model=patcher,
        positive=text_conditioning(),
        latent=latent,
        config=SamplerConfig(steps=1, seed=1),
        load_models=LoadRecorder(),
        require_upstream_class=False,
    )
    out = result.latent["samples"]
    assert type(out) is type(latent["samples"])
    video, audio = out.unbind()
    assert video.shape == latent["samples"].tensors[0].shape
    assert audio.shape == latent["samples"].tensors[1].shape
    assert video.dtype == torch.float32
    assert not torch.equal(video, torch.zeros_like(video))


def test_a_bf16_latent_keeps_its_dtype_out_while_the_step_runs_in_fp32():
    """The fp32 island reaches the sampler; the ``LATENT`` contract does not move.

    ``forward_chunk`` returns the output heads' fp32 velocity unrounded (see
    "Returned velocity" in ``raven_streaming.causal_model``), so ``x0`` and the
    ``(1 - s) x0 + s eps`` transition are computed in fp32 even when the caller
    hands in BF16 latents. What the node returns is still cast back to the
    request's dtype, so nothing downstream sees a changed ``LATENT``.
    """
    latent = empty_av_latent(frames=39, dtype=torch.bfloat16)
    seen = []
    result, _, _, _ = run(frames=39, latent=latent, on_chunk=seen.append,
                          config=SamplerConfig(steps=2, seed=5))

    video, audio = result.latent["samples"].unbind()
    assert video.dtype == torch.bfloat16 and audio.dtype == torch.bfloat16
    assert video.shape == latent["samples"].tensors[0].shape

    # ... and every chunk handed to the pipeline carries the fp32 transition
    assert [c.index for c in seen] == [0, 1, 2]
    for chunk in seen:
        assert chunk.video_x0.dtype == torch.float32
        assert chunk.audio_x0.dtype == torch.float32
        assert torch.isfinite(chunk.video_x0).all()


def test_an_fp32_latent_is_unchanged_by_the_fp32_velocity():
    """The ComfyUI default path: fp32 in, fp32 through, fp32 out."""
    seen = []
    result, _, _, _ = run(frames=22, on_chunk=seen.append,
                          config=SamplerConfig(steps=2, seed=5))
    video, audio = result.latent["samples"].unbind()
    assert video.dtype == torch.float32 and audio.dtype == torch.float32
    assert all(c.video_x0.dtype == torch.float32 for c in seen)


def test_the_streaming_pipeline_accepts_the_fp32_chunk_the_sampler_emits():
    """The decode lane's own entry check, on the tensors a BF16 run now produces."""
    from raven_streaming import streaming_pipeline

    seen = []
    run(frames=22, latent=empty_av_latent(frames=22, dtype=torch.bfloat16),
        on_chunk=seen.append, config=SamplerConfig(steps=1, seed=3))
    assert seen and seen[0].video_x0.dtype == torch.float32
    for chunk in seen:
        streaming_pipeline._require_chunk_output(chunk)  # raises if it would not
        # what ``comfy.sd.VAE.decode`` does to a latent before decoding it
        for tensor in (chunk.video_x0, chunk.audio_x0):
            moved = streaming_pipeline.detach_to_cpu(tensor).to(dtype=torch.float16)
            assert moved.shape == tensor.shape
            assert torch.isfinite(moved).all()


# --------------------------------------------------------------------------
# chunk callback
# --------------------------------------------------------------------------
def test_chunk_callback_fires_once_per_chunk_in_order_with_the_final_x0():
    seen = []
    result, _, _, _ = run(
        frames=39, config=SamplerConfig(steps=2, seed=8), on_chunk=seen.append
    )
    assert [c.index for c in seen] == [0, 1, 2]
    assert [c.is_last for c in seen] == [False, False, True]

    video, audio = result.latent["samples"].unbind()
    for chunk in seen:
        assert torch.equal(
            chunk.video_x0, video[:, :, chunk.video_start:chunk.video_stop])
        assert torch.equal(
            chunk.audio_x0, audio[:, :, :, chunk.audio_start:chunk.audio_stop])


def test_chunk_callback_runs_after_the_scatter_and_before_the_clean_fill():
    order = []
    dit = FakeCausalDiT()

    def on_chunk(chunk):
        order.append(("chunk", chunk.index, len(dit.calls)))

    run(dit=dit, config=SamplerConfig(steps=1, seed=4), on_chunk=on_chunk)
    # chunk 0 is delivered after its single noise forward (prefill + 1 call)
    # and before the clean fill that follows it
    assert order[0] == ("chunk", 0, 2)
    kinds = [c.kind for c in dit.calls]
    assert kinds == ["prefill", "noise", "clean", "noise"]


def test_chunk_callback_failure_aborts_and_returns_nothing():
    dit = FakeCausalDiT()
    boom = RuntimeError("consumer exploded")

    def on_chunk(chunk):
        raise boom

    with pytest.raises(RuntimeError, match="consumer exploded"):
        run(dit=dit, on_chunk=on_chunk)
    # it stopped at the first chunk: no clean fill, no second chunk
    assert [c.kind for c in dit.calls] == ["prefill"] + ["noise"] * 4


def test_a_callback_that_consumes_the_rollout_rng_fails_loudly():
    holder = {}
    real_normal = consistency.RolloutRNG.normal

    def capture(self, shape, **kwargs):
        holder["rng"] = self
        return real_normal(self, shape, **kwargs)

    def greedy(chunk):
        torch.randn(2, generator=holder["rng"].generator)

    consistency.RolloutRNG.normal = capture
    try:
        with pytest.raises(SamplerError, match="consumed the sampler's RNG"):
            run(on_chunk=greedy)
    finally:
        consistency.RolloutRNG.normal = real_normal


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------
def test_every_documented_cancel_point_is_polled():
    points = []
    run(cancel_check=lambda point: points.append(point) or False)
    assert set(points) == set(CANCEL_POINTS)


def test_cancel_points_occur_in_the_documented_order():
    points = []
    result, _, _, _ = run(
        frames=39,
        config=SamplerConfig(steps=2, seed=1),
        cancel_check=lambda point: points.append(point) or False,
    )
    assert points == list(result.cancel_points)
    assert points[0] == "before_model_load"
    assert points[1:3] == ["before_prefill", "after_prefill"]
    assert points[-1] == "before_return"
    # one noise-forward pair per step, one clean pair per non-final chunk
    assert points.count("before_noise_forward") == result.noise_forwards
    assert points.count("after_noise_forward") == result.noise_forwards
    assert points.count("after_step_update") == result.noise_forwards
    assert points.count("before_clean") == result.clean_forwards
    assert points.count("before_chunk_delivery") == result.num_chunks


def test_nothing_is_polled_between_the_video_and_audio_step_draws():
    events = []
    real_normal = consistency.RolloutRNG.normal

    def spy(self, shape, **kwargs):
        events.append(("draw", kwargs["label"]))
        return real_normal(self, shape, **kwargs)

    consistency.RolloutRNG.normal = spy
    try:
        run(
            frames=39,
            config=SamplerConfig(steps=2, seed=1),
            cancel_check=lambda point: events.append(("cancel", point)) or False,
            on_chunk=lambda chunk: events.append(("chunk", chunk.index)),
        )
    finally:
        consistency.RolloutRNG.normal = real_normal

    for index, event in enumerate(events):
        if event == ("draw", "video_step_eps"):
            assert events[index + 1] == ("draw", "audio_step_eps"), (
                "a hook ran between the two step draws"
            )


@pytest.mark.parametrize("point", CANCEL_POINTS)
def test_cancelling_at_any_point_raises_and_returns_nothing(point):
    def cancel(seen):
        return seen == point

    with pytest.raises(SamplingCancelled, match=point):
        run(cancel_check=cancel)


def test_cancellation_leaves_no_pending_cache_rows():
    """A hook that stops the run mid-clean-fill must not leave a half chunk."""
    cache_holder = {}

    class SpyDiT(FakeCausalDiT):
        def _stage_and_commit(self, cache, rows, role):
            cache_holder["cache"] = cache
            if role == "clean":
                # stage one layer only, then let the cancel hook fire
                cache.stage(0, torch.zeros(rows, 2, 4), torch.zeros(rows, 2, 4))
                raise SamplingCancelled("cancelled inside the clean fill")
            super()._stage_and_commit(cache, rows, role)

    with pytest.raises(SamplingCancelled):
        run(dit=SpyDiT())
    assert cache_holder["cache"].has_pending is False


def test_a_hook_returning_none_never_cancels():
    """``throw_exception_if_processing_interrupted`` returns None on every call."""
    calls = []
    result, _, _, _ = run(cancel_check=lambda point: calls.append(point))
    assert len(calls) == len(result.cancel_points)
    assert result.noise_forwards == 8


def test_a_zero_argument_cancel_hook_is_supported():
    """``comfy.model_management.throw_exception_if_processing_interrupted`` style."""
    state = {"n": 0}

    def interrupt():
        state["n"] += 1
        if state["n"] == 3:
            raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run(cancel_check=interrupt)


def test_a_cancel_hook_that_consumes_rng_fails_loudly():
    holder = {}
    real_normal = consistency.RolloutRNG.normal

    def capture(self, shape, **kwargs):
        holder["rng"] = self
        return real_normal(self, shape, **kwargs)

    def greedy(point):
        if "rng" in holder:
            torch.randn(1, generator=holder["rng"].generator)
        return False

    consistency.RolloutRNG.normal = capture
    try:
        with pytest.raises(SamplerError, match="consumed the sampler's RNG"):
            run(cancel_check=greedy)
    finally:
        consistency.RolloutRNG.normal = real_normal


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def test_result_describes_the_run():
    result, _, _, _ = run()
    described = result.describe()
    assert described["frames"] == 22
    assert described["num_chunks"] == 2
    assert described["noise_forwards"] == 8
    assert described["clean_forwards"] == 1
    assert described["config"]["video_sigmas"][0] == 1.0
