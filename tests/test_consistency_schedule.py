"""The shifted trailing sigma grid, pinned three ways.

Against literal values (so a refactor cannot quietly move the schedule),
against RAVEN's own ``TrailingSamplingTimesteps`` expression -- both replayed
in pure Python and, when numpy is installed, evaluated with the very same
``np.arange`` call -- and against the structural claims the rollout depends on:
``N`` forwards, ``sigma_0 == 1``, and a final ``next`` sigma of exactly ``0``.

Pure math; no torch model, no ComfyUI.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.consistency import (  # noqa: E402
    DEFAULT_AUDIO_SHIFT,
    DEFAULT_STEPS,
    DEFAULT_VIDEO_SHIFT,
    SamplerConfig,
    SamplerError,
    shifted_trailing_sigmas,
    step_pairs,
)

def raven_trailing_python(num_sampling_steps: int, shift: float):
    """``TrailingSamplingTimesteps.set_timesteps`` at T=1.0, final_linear_steps=0.

    ``np.arange(1.0, 0, -1/N)`` yields ``start + i * step`` for
    ``i in range(N)``; the shift line below is copied character for character
    from ``common/diffusion/timestep/sampling/trailing.py``.
    """
    step = -1.0 / num_sampling_steps
    t = [1.0 + i * step for i in range(num_sampling_steps)]
    return [shift * x / (1 + (shift - 1) * x) for x in t]


def test_published_defaults_are_the_4nfe_preview_trial():
    assert DEFAULT_STEPS == 4
    assert DEFAULT_VIDEO_SHIFT == 12.0
    assert DEFAULT_AUDIO_SHIFT == 3.0
    config = SamplerConfig()
    assert config.steps == 4
    assert config.sink == 2
    assert config.window == 2


def test_video_grid_golden():
    assert shifted_trailing_sigmas(4, 12.0) == (
        1.0,
        0.972972972972973,     # 9 / 9.25
        0.9230769230769231,    # 6 / 6.5
        0.8,                   # 3 / 3.75
    )


def test_audio_grid_golden():
    assert shifted_trailing_sigmas(4, 3.0) == (1.0, 0.9, 0.75, 0.5)


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 8, 20, 50])
@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0, 12.0, 17.0])
def test_matches_raven_trailing_expression(steps, shift):
    ours = shifted_trailing_sigmas(steps, shift)
    theirs = raven_trailing_python(steps, shift)
    assert len(ours) == len(theirs)
    # The two differ only by float rounding in ``1 - i/N`` vs ``1 + i*(-1/N)``;
    # 1e-12 is far below any difference that could change a sample.
    assert all(abs(a - b) <= 1e-12 for a, b in zip(ours, theirs))


@pytest.mark.parametrize("steps", [1, 3, 4, 20])
@pytest.mark.parametrize("shift", [3.0, 12.0])
def test_matches_ravens_actual_numpy_call(steps, shift):
    np = pytest.importorskip("numpy")
    t = np.arange(1.0, 0, -1.0 / steps)
    theirs = shift * t / (1 + (shift - 1) * t)
    ours = np.array(shifted_trailing_sigmas(steps, shift))
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("shift", [0.5, 1.0, 3.0, 12.0, 100.0])
def test_first_sigma_is_exactly_one_for_any_shift(shift):
    # shift * 1 / (1 + (shift - 1) * 1) == 1 exactly; a chunk starts at pure noise
    assert shifted_trailing_sigmas(4, shift)[0] == 1.0


@pytest.mark.parametrize("steps", [1, 2, 4, 7])
def test_step_count_is_the_forward_count(steps):
    assert len(shifted_trailing_sigmas(steps, 12.0)) == steps
    assert len(step_pairs(shifted_trailing_sigmas(steps, 12.0))) == steps


def test_sigmas_are_strictly_decreasing():
    for shift in (1.0, 3.0, 12.0):
        sigmas = shifted_trailing_sigmas(8, shift)
        assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_last_next_sigma_is_zero_and_others_chain():
    sigmas = shifted_trailing_sigmas(4, 12.0)
    pairs = step_pairs(sigmas)
    assert [p[0] for p in pairs] == list(sigmas)
    assert [p[1] for p in pairs] == list(sigmas[1:]) + [0.0]
    assert pairs[-1][1] == 0.0


def test_single_step_goes_straight_to_zero():
    assert step_pairs(shifted_trailing_sigmas(1, 12.0)) == ((1.0, 0.0),)


def test_config_exposes_both_independent_grids():
    config = SamplerConfig(steps=4, video_shift=12.0, audio_shift=3.0)
    assert config.video_sigmas == shifted_trailing_sigmas(4, 12.0)
    assert config.audio_sigmas == shifted_trailing_sigmas(4, 3.0)
    # the two streams are NOT the same grid remapped: only sigma_0 coincides
    assert config.video_sigmas[0] == config.audio_sigmas[0] == 1.0
    assert config.video_sigmas[1:] != config.audio_sigmas[1:]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(steps=0),
        dict(steps=-1),
        dict(steps=1.5),
        dict(video_shift=0.0),
        dict(audio_shift=-1.0),
        dict(sink=0),          # cache chunk 0 is the text; every chunk attends it
        dict(sink=-1),
        dict(window=-1),
        dict(seed=1.5),
    ],
)
def test_config_rejects_impossible_settings(kwargs):
    with pytest.raises(SamplerError):
        SamplerConfig(**kwargs)


def test_config_accepts_no_eviction():
    assert SamplerConfig(window=None).window is None
    assert SamplerConfig(window=0).window == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_schedule_rejects_impossible_steps(bad):
    with pytest.raises(SamplerError):
        shifted_trailing_sigmas(bad, 12.0)


def test_schedule_rejects_impossible_shift():
    with pytest.raises(SamplerError):
        shifted_trailing_sigmas(4, 0.0)
    with pytest.raises(SamplerError):
        shifted_trailing_sigmas(4, -3.0)


def test_step_pairs_rejects_empty_grid():
    with pytest.raises(SamplerError):
        step_pairs([])


def test_describe_is_serialisable():
    described = SamplerConfig(seed=7).describe()
    assert described["seed"] == 7
    assert described["video_sigmas"][0] == 1.0
    assert described["window"] == 2
