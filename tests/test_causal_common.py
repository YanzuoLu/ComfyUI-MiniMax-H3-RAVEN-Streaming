"""Shared fixtures for the M2 causal-lane tests.

Collected by pytest (the name matches ``test_causal_*.py``) but holds no tests
of its own: only the tiny H3 configuration both causal test modules build, and
the helpers that give the official and the causal model identical weights.

The tiny model keeps every *structural* property of the real one -- 2x2 patch,
24/32 latent channels, stereo audio, the same RoPE axis split, the same AdaLN
modality expansion -- and shrinks only widths and depth, so state-dict and
forward parity mean what they say while running on CPU in seconds.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 3 axes x ``rope_inv_freq_len`` x 2 halves must fit inside ``attention_head_dim``
TINY_CONFIG: Dict[str, Any] = dict(
    hidden_size=32,
    num_layers=3,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=24,
    ffn_hidden_size=64,
    latents_dim=24,
    audio_latents_dim=32,
    text_dim=16,
    timestep_input_dim=16,
    time_embed_hidden_size=32,
    time_embed_dim=32,
    rope_inv_freq_len=4,
)

TINY_REQUEST = dict(frames=39, width=64, height=64, text_len=5)


def build_models(seed: int = 0, config: Optional[Dict[str, Any]] = None,
                 dtype: torch.dtype = torch.float32) -> Tuple[Any, Any]:
    """``(official, causal)`` tiny models carrying identical random weights.

    ``dtype`` is the *compute* dtype the model is built at. It is a real
    parameter and not a constant because the numerical differences the causal
    lane exists to close (a SiLU rounded to BF16, a fused ``addcmul_``) are
    invisible at fp32: both spellings agree bit for bit there.
    """
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model

    cfg = dict(TINY_CONFIG if config is None else config)
    kwargs = dict(cfg, dtype=dtype, device=torch.device("cpu"),
                  operations=comfy.ops.disable_weight_init)
    official = MiniMaxH3Model(**kwargs)
    causal = RavenCausalMiniMaxH3Model(**kwargs)

    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, value in official.state_dict().items():
        # drawn in fp32 and cast, so the same seed gives the same weights at
        # every compute dtype (a no-op for the fp32 build)
        state[name] = (torch.randn(value.shape, generator=generator,
                                   dtype=torch.float32) * 0.05).to(value.dtype)
    # a plausible inverse-frequency ladder rather than noise, so RoPE angles are
    # in the range the real checkpoint uses
    length = cfg["rope_inv_freq_len"]
    state["rope.inv_freq"] = 1.0 / (10000.0 ** (torch.arange(length, dtype=torch.float32) / length))

    official.load_state_dict(state)
    causal.load_state_dict(state)
    official.requires_grad_(False)
    causal.requires_grad_(False)
    official.eval()
    causal.eval()
    return official, causal


def tiny_layout(**overrides: Any):
    from raven_streaming.layout import T2VALayout

    request = dict(TINY_REQUEST)
    request.update(overrides)
    return T2VALayout.from_request(**request)


def random_inputs(layout, seed: int = 1):
    """``(video, audio, context)`` for one tiny clip."""
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(*layout.video_latent_shape(TINY_CONFIG["latents_dim"]),
                        generator=generator)
    audio = torch.randn(*layout.audio_latent_shape(TINY_CONFIG["audio_latents_dim"]),
                        generator=generator)
    context = torch.randn(1, layout.text_len, TINY_CONFIG["text_dim"], generator=generator)
    return video, audio, context


@pytest.fixture
def tiny_models(comfyui_on_syspath):
    return build_models()


@pytest.fixture
def tiny_bf16_models(comfyui_on_syspath):
    """The same tiny pair at BF16, where RAVEN's and Comfy's roundings diverge."""
    return build_models(dtype=torch.bfloat16)
