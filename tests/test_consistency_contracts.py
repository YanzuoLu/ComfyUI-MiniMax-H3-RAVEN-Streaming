"""What the node boundary accepts, and everything it refuses by name.

The rule these tests encode: a feature the chunk-major loop does not execute
must be *rejected*, never silently ignored. Accepting a negative prompt, a
ControlNet or a keyframe and then not running it looks like a quality bug with
no error anywhere, which is the worst possible failure mode for a sampler.

Pure fakes; no ComfyUI import. ``tests/test_consistency_official.py`` runs the
same parsers against the real pinned upstream types.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import contracts  # noqa: E402
from raven_streaming import layout as layout_mod  # noqa: E402
from raven_streaming.contracts import (  # noqa: E402
    ContractError,
    build_output_latent,
    parse_conditioning,
    parse_latent,
    resolve_model,
    resolve_transformer_options,
)
from test_consistency_common import (  # noqa: E402
    FakeCausalDiT,
    FakeNestedTensor,
    FakePatcher,
    empty_av_latent,
    text_conditioning,
)


# --------------------------------------------------------------------------
# CONDITIONING
# --------------------------------------------------------------------------
def test_accepts_one_positive_t2va_entry():
    parsed = parse_conditioning(text_conditioning(text_len=7, dim=16))
    assert parsed.text_len == 7
    assert tuple(parsed.cross_attn.shape) == (1, 7, 16)
    assert parsed.token_tags is not None
    assert tuple(parsed.token_tags.shape) == (7,)


def test_token_tags_are_preserved_not_dropped():
    tags = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)
    cond = text_conditioning(text_len=5)
    cond[0][1]["minimax_token_tags"] = tags
    parsed = parse_conditioning(cond)
    assert torch.equal(parsed.token_tags, tags)


def test_conditioning_without_tags_is_accepted():
    parsed = parse_conditioning(text_conditioning(tags=False))
    assert parsed.token_tags is None


def test_rejects_two_entries_no_cfg_no_negative_no_combine():
    cond = text_conditioning() + text_conditioning()
    with pytest.raises(ContractError, match="exactly one positive entry"):
        parse_conditioning(cond)


def test_rejects_empty_conditioning():
    with pytest.raises(ContractError, match="empty"):
        parse_conditioning([])


def test_rejects_none_conditioning():
    with pytest.raises(ContractError):
        parse_conditioning(None)


def test_rejects_batched_conditioning():
    cond = text_conditioning()
    cond[0][0] = torch.randn(2, 5, 16)
    with pytest.raises(ContractError, match="batch"):
        parse_conditioning(cond)


@pytest.mark.parametrize(
    "key, value",
    [
        ("minimax_keyframes", [{"resolved_frame_index": 0}]),
        ("minimax_refs", [{"latent": None}]),
        ("minimax_visual_cond_noise_aug", 0.05),
        ("minimax_audio_cond_noise_aug", 0.05),
        ("control", object()),
        ("gligen", object()),
        ("area", (1, 1, 0, 0)),
        ("mask", torch.ones(1, 1)),
        ("strength", 0.5),
        ("start_percent", 0.0),
        ("end_percent", 1.0),
        ("clip_start_percent", 0.0),
        ("hooks", object()),
        ("something_new_upstream_added", 1),
    ],
)
def test_rejects_unsupported_conditioning_extras(key, value):
    cond = text_conditioning()
    cond[0][1][key] = value
    with pytest.raises(ContractError, match="unsupported conditioning extras"):
        parse_conditioning(cond)


@pytest.mark.parametrize(
    "key, value",
    [
        ("minimax_keyframes", [{"resolved_frame_index": 0}]),
        ("minimax_refs", [{"latent": None}]),
    ],
)
def test_condition_row_refusal_is_an_implementation_limit_not_a_lora_claim(key, value):
    """The refusal must name *this sampler* as the limit.

    Keyframe / reference conditioning is refused because the causal packed
    layout for condition rows is not implemented or verified here -- not
    because the RAVEN LoRA is text-only. The message has to say so, otherwise
    the error reads as a capability claim about the adapter that nothing in
    this repository measured.
    """
    cond = text_conditioning()
    cond[0][1][key] = value
    with pytest.raises(ContractError) as excinfo:
        parse_conditioning(cond)

    message = str(excinfo.value)
    lowered = message.lower()
    assert "not implemented or verified" in lowered
    assert "this streaming sampler" in lowered
    assert "not a claim about what the raven lora supports" in lowered
    # No absolute "T2VA only" phrasing anywhere in the reason.
    assert "t2va only" not in lowered
    assert "only t2va" not in lowered
    assert "out of scope" not in lowered


def test_no_unsupported_reason_claims_the_lora_is_t2va_only():
    for key, reason in contracts.UNSUPPORTED_CONDITIONING_KEYS.items():
        lowered = reason.lower()
        assert "t2va only" not in lowered, key
        assert "only t2va" not in lowered, key
        assert "out of scope" not in lowered, key


def test_rejects_wrong_token_tag_length():
    cond = text_conditioning(text_len=5)
    cond[0][1]["minimax_token_tags"] = torch.ones(4, dtype=torch.long)
    with pytest.raises(ContractError, match="covers 4 tokens"):
        parse_conditioning(cond)


def test_rejects_out_of_range_token_tags():
    cond = text_conditioning(text_len=5)
    cond[0][1]["minimax_token_tags"] = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
    with pytest.raises(ContractError, match=r"tag\(s\) \[3\]"):
        parse_conditioning(cond)


def test_rejects_float_token_tags():
    cond = text_conditioning(text_len=5)
    cond[0][1]["minimax_token_tags"] = torch.ones(5)
    with pytest.raises(ContractError, match="integer"):
        parse_conditioning(cond)


# --------------------------------------------------------------------------
# LATENT
# --------------------------------------------------------------------------
def test_accepts_official_empty_av_latent():
    request = parse_latent(empty_av_latent(frames=22, width=32, height=32))
    assert (request.frames, request.width, request.height) == (22, 32, 32)
    assert (request.latent_t, request.latent_h, request.latent_w) == (7, 2, 2)
    assert request.audio_t == layout_mod.audio_latent_t(22)
    assert request.nested_cls is FakeNestedTensor


def test_latent_derives_the_same_layout_the_sampler_uses():
    request = parse_latent(empty_av_latent(frames=39, width=64, height=64))
    built = request.layout(text_len=5)
    assert built.frames == 39
    assert built.num_chunks == 3          # (0,5) (5,10) (10,12)
    assert built.latent_t == 12
    assert built.audio_t == request.audio_t


def test_rejects_non_empty_latent_loudly():
    latent = empty_av_latent()
    latent["samples"].tensors[0][0, 0, 0, 0, 0] = 1e-6
    with pytest.raises(ContractError, match="not empty"):
        parse_latent(latent)


def test_rejects_non_empty_audio_stream():
    latent = empty_av_latent()
    latent["samples"].tensors[1][0, 0, 0, 0] = -1.0
    with pytest.raises(ContractError, match="audio latent is not empty"):
        parse_latent(latent)


def test_rejects_plain_tensor_latent():
    with pytest.raises(ContractError, match="NestedTensor"):
        parse_latent({"samples": torch.zeros(1, 24, 7, 2, 2)})


def test_rejects_single_stream_nested_latent():
    with pytest.raises(ContractError, match=r"holds 1 stream"):
        parse_latent({"samples": FakeNestedTensor((torch.zeros(1, 24, 7, 2, 2),))})


def test_rejects_batch_two():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(2, 24, 7, 2, 2), torch.zeros(2, 32, 2, 37)))}
    with pytest.raises(ContractError, match="batch size must be 1"):
        parse_latent(latent)


def test_rejects_wrong_video_channels():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 16, 7, 2, 2), torch.zeros(1, 32, 2, 37)))}
    with pytest.raises(ContractError, match="video latent has 16 channels"):
        parse_latent(latent)


def test_rejects_mono_audio():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 1, 37)))}
    with pytest.raises(ContractError, match="stereo"):
        parse_latent(latent)


def test_rejects_audio_length_that_is_not_the_clip_clock():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 40)))}
    with pytest.raises(ContractError, match="carries 37"):
        parse_latent(latent)


def test_rejects_off_grid_video_length():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 24, 6, 2, 2), torch.zeros(1, 32, 2, 32)))}
    with pytest.raises(ContractError, match=r"5k \+ 2"):
        parse_latent(latent)


def test_rejects_the_degenerate_two_latent_clip():
    # latent_t == 2 is k == 0: five frames, no room for the streaming context
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 8)))}
    with pytest.raises(ContractError):
        parse_latent(latent)


def test_rejects_canvas_off_the_32_grid():
    latent = {"samples": FakeNestedTensor(
        (torch.zeros(1, 24, 7, 3, 2), torch.zeros(1, 32, 2, 37)))}
    with pytest.raises(ContractError, match="multiple of 32"):
        parse_latent(latent)


@pytest.mark.parametrize("key", ["noise_mask", "batch_index", "unknown_thing"])
def test_rejects_unsupported_latent_keys(key):
    latent = empty_av_latent()
    latent[key] = torch.ones(1)
    with pytest.raises(ContractError, match="unsupported LATENT keys"):
        parse_latent(latent)


def test_rejects_missing_samples():
    with pytest.raises(ContractError, match="no 'samples'"):
        parse_latent({})


# --------------------------------------------------------------------------
# output LATENT
# --------------------------------------------------------------------------
def test_output_latent_roundtrips_the_input_structure():
    latent = empty_av_latent()
    request = parse_latent(latent)
    video = torch.randn(1, 24, request.latent_t, request.latent_h, request.latent_w)
    audio = torch.randn(1, 32, 2, request.audio_t)
    out = build_output_latent(request, video, audio)

    assert set(out) == {"samples"}
    assert type(out["samples"]) is type(latent["samples"])
    assert out["samples"].is_nested
    streams = out["samples"].unbind()
    assert len(streams) == 2
    assert torch.equal(streams[0], video)
    assert torch.equal(streams[1], audio)
    # and it parses back as the same geometry (minus the empty check)
    assert tuple(streams[0].shape) == tuple(latent["samples"].tensors[0].shape)
    assert tuple(streams[1].shape) == tuple(latent["samples"].tensors[1].shape)


def test_output_latent_rejects_a_mis_shaped_result():
    request = parse_latent(empty_av_latent())
    with pytest.raises(ContractError, match="output video latent"):
        build_output_latent(
            request,
            torch.randn(1, 24, 5, 2, 2),
            torch.randn(1, 32, 2, request.audio_t),
        )
    with pytest.raises(ContractError, match="output audio latent"):
        build_output_latent(
            request,
            torch.randn(1, 24, request.latent_t, 2, 2),
            torch.randn(1, 32, 2, 3),
        )


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
def test_resolves_a_static_patcher_with_the_causal_dit():
    dit = FakeCausalDiT(num_layers=3)
    patcher = FakePatcher(dit)
    resolved = resolve_model(patcher, require_upstream_class=False)
    assert resolved.diffusion_model is dit
    assert resolved.num_layers == 3
    assert resolved.load_device == torch.device("cpu")
    assert resolved.transformer_options == {}


def test_transformer_options_are_a_copy_not_the_patchers_dict():
    original = {"sample_key": 1}
    patcher = FakePatcher(transformer_options=original)
    resolved = resolve_model(patcher, require_upstream_class=False)
    assert resolved.transformer_options == original
    assert resolved.transformer_options is not original


def test_rejects_a_dynamic_patcher():
    patcher = FakePatcher(dynamic=True)
    with pytest.raises(ContractError, match="is_dynamic"):
        resolve_model(patcher, require_upstream_class=False)


def test_rejects_a_stock_bidirectional_dit():
    class Stock:
        blocks = [object()]

    patcher = FakePatcher(Stock())
    with pytest.raises(ContractError, match="prefill_text"):
        resolve_model(patcher, require_upstream_class=False)


def test_rejects_a_model_without_a_diffusion_model():
    patcher = FakePatcher()
    patcher.model.diffusion_model = None
    with pytest.raises(ContractError, match="diffusion_model"):
        resolve_model(patcher, require_upstream_class=False)


def test_rejects_dit_block_replacement():
    patcher = FakePatcher(
        transformer_options={"patches_replace": {"dit": {("double_block", 0): object()}}}
    )
    with pytest.raises(ContractError, match=r"patches_replace"):
        resolve_model(patcher, require_upstream_class=False)


def test_empty_patches_replace_is_fine():
    patcher = FakePatcher(transformer_options={"patches_replace": {"dit": {}}})
    resolve_model(patcher, require_upstream_class=False)


@pytest.mark.parametrize(
    "key",
    [
        "model_function_wrapper",
        "sampler_cfg_function",
        "sampler_post_cfg_function",
        "sampler_pre_cfg_function",
        "sampler_calc_cond_batch_function",
        "denoise_mask_function",
    ],
)
def test_rejects_unsupported_model_option_hooks(key):
    patcher = FakePatcher()
    patcher.model_options[key] = lambda *a, **k: None
    with pytest.raises(ContractError, match="cannot honour"):
        resolve_transformer_options(patcher)


@pytest.mark.parametrize("key", ["wrappers", "patches"])
def test_rejects_unsupported_transformer_options(key):
    patcher = FakePatcher(transformer_options={key: {"apply_model": {None: [lambda: None]}}})
    with pytest.raises(ContractError, match="cannot honour"):
        resolve_transformer_options(patcher)


def test_rejects_patcher_level_wrappers():
    patcher = FakePatcher(wrappers={"apply_model": {None: [lambda: None]}})
    with pytest.raises(ContractError, match="patcher wrappers"):
        resolve_transformer_options(patcher)


def test_rejects_a_non_model_object():
    class NotAModel:
        model_options = {"transformer_options": {}}

    with pytest.raises(ContractError, match="has no .model"):
        resolve_model(NotAModel(), require_upstream_class=False)
