"""State-dict neutrality and dense-path parity of the causal subclasses.

The causal model must be a drop-in for the official one at the checkpoint
level: same keys, same shapes, same dtypes, in the same order -- and its
inherited dense forward must still produce the official result bit for bit,
because M2 only *adds* a lane, it does not replace the one that already works.

Requires a local ComfyUI checkout (see ``tests/conftest.py``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_causal_common import (  # noqa: E402
    TINY_CONFIG,
    build_models,
    random_inputs,
    tiny_layout,
    tiny_models,  # noqa: F401  (fixture re-export)
)


# --- state dict --------------------------------------------------------------


def test_state_dict_keys_are_identical(tiny_models):
    official, causal = tiny_models
    assert list(official.state_dict()) == list(causal.state_dict())


def test_state_dict_shapes_and_dtypes_are_identical(tiny_models):
    official, causal = tiny_models
    theirs = {k: (tuple(v.shape), v.dtype) for k, v in official.state_dict().items()}
    ours = {k: (tuple(v.shape), v.dtype) for k, v in causal.state_dict().items()}
    assert ours == theirs


def test_official_checkpoint_loads_into_the_causal_model(tiny_models):
    official, causal = tiny_models
    missing, unexpected = causal.load_state_dict(official.state_dict(), strict=True)
    assert list(missing) == []
    assert list(unexpected) == []
    for name, value in official.state_dict().items():
        assert torch.equal(causal.state_dict()[name], value)


def test_blocks_are_causal_and_carry_their_layer_index(tiny_models):
    from raven_streaming.causal_model import RavenCausalAttention, RavenCausalDiTBlock

    _, causal = tiny_models
    assert len(causal.blocks) == TINY_CONFIG["num_layers"]
    for index, block in enumerate(causal.blocks):
        assert isinstance(block, RavenCausalDiTBlock)
        assert isinstance(block.attn, RavenCausalAttention)
        assert block.attn.layer_idx == index


def test_layer_index_is_not_a_parameter_or_buffer(tiny_models):
    _, causal = tiny_models
    names = set(causal.state_dict())
    assert not any("layer_idx" in name for name in names)


def test_causal_attention_matches_the_official_parameter_set(comfyui_on_syspath):
    import comfy.ops
    from comfy.ldm.minimax.model import Attention

    from raven_streaming.causal_model import RavenCausalAttention

    kwargs = dict(dtype=torch.float32, device=torch.device("cpu"),
                  operations=comfy.ops.disable_weight_init)
    official = Attention(32, 2, 24, 1e-5, **kwargs)
    causal = RavenCausalAttention(32, 2, 24, 1e-5, layer_idx=7, **kwargs)
    assert {k: tuple(v.shape) for k, v in official.state_dict().items()} == {
        k: tuple(v.shape) for k, v in causal.state_dict().items()
    }


def test_curve_form_checkpoint_also_rebuilds(comfyui_on_syspath):
    # The curve-form checkpoint replaces the time embedder and stores adaln at
    # fp32; the rebuild reads that off the parent instead of assuming a form.
    official, causal = build_models(config=dict(TINY_CONFIG, adaln_curve_grid=8))
    assert causal.use_adaln_curves
    assert {k: tuple(v.shape) for k, v in official.state_dict().items()} == {
        k: tuple(v.shape) for k, v in causal.state_dict().items()
    }
    assert causal.blocks[0].adaln_proj.linear.weight.dtype == torch.float32


# --- dense path --------------------------------------------------------------


@pytest.mark.parametrize("sigma", [999.0, 500.0, 1.0])
def test_dense_forward_is_bit_identical(tiny_models, sigma):
    official, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    timestep = torch.tensor([sigma])
    with torch.no_grad():
        theirs = official(x=[video, audio], timestep=timestep, context=context)
        ours = causal(x=[video, audio], timestep=timestep, context=context)
    for a, b in zip(ours, theirs):
        assert a.shape == b.shape
        assert torch.equal(a, b)


def test_dense_forward_is_bit_identical_with_prerefined_text(tiny_models):
    official, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    with torch.no_grad():
        refined = official.preprocess_text_embeds(context)
        theirs = official(x=[video, audio], timestep=torch.tensor([700.0]), context=refined)
        ours = causal(x=[video, audio], timestep=torch.tensor([700.0]), context=refined)
    assert refined.shape[-1] == TINY_CONFIG["hidden_size"]
    for a, b in zip(ours, theirs):
        assert torch.equal(a, b)


def test_dense_forward_leaves_no_cache_state(tiny_models):
    # the dense path must not stage anything anywhere: it takes no cache at all
    _, causal = tiny_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    with torch.no_grad():
        causal(x=[video, audio], timestep=torch.tensor([500.0]), context=context)
    for block in causal.blocks:
        assert not hasattr(block.attn, "_pending")
