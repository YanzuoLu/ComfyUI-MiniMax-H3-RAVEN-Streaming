"""Fakes shared by the M3 sampler tests: no ComfyUI, no weights, no GPU.

Collected by pytest (the name matches ``test_consistency_*.py``) but holds no
tests of its own.

The sampler's contract is an *ordering* contract -- draw order, call order,
cache order, cancel order -- so what it needs to be tested against is a model
that records what it was asked to do and a latent that has the official
structure, not 66 GB of BF16 weights. The one place that is not enough is
"does this still line up with the real upstream types", which
``tests/test_consistency_official.py`` covers against the pinned checkout.

The fake DiT deliberately implements the *same* cache protocol as
:mod:`raven_streaming.causal_model`: stage every layer, commit exactly once,
never on a noise forward. A real :class:`raven_streaming.cache.ChunkKVCache` is
used, so the sampler's cache handling is exercised for real even though the
attention is not.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as layout_mod  # noqa: E402
from raven_streaming.consistency import SamplerConfig, step_pairs  # noqa: E402

#: A clip small enough to run in milliseconds and still have >1 chunk plus the
#: 2-latent tail: 22 frames -> 7 video latents -> chunks (0,5) and (5,7).
TINY_REQUEST = dict(frames=22, width=32, height=32)
TINY_TEXT_LEN = 5
#: fake head widths; nothing reads them except the cache's shape checks
FAKE_HEADS = 2
FAKE_HEAD_DIM = 4


# --------------------------------------------------------------------------
# latent / conditioning
# --------------------------------------------------------------------------
class FakeNestedTensor:
    """The two attributes :mod:`raven_streaming.contracts` duck-types on."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def empty_av_latent(
    frames: int = TINY_REQUEST["frames"],
    width: int = TINY_REQUEST["width"],
    height: int = TINY_REQUEST["height"],
    *,
    nested_cls: type = FakeNestedTensor,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """An empty H3 AV latent with the official geometry."""
    latent_t = layout_mod.video_latent_t(frames)
    audio_t = layout_mod.audio_latent_t(frames)
    video = torch.zeros(1, 24, latent_t, height // 16, width // 16, dtype=dtype)
    audio = torch.zeros(1, 32, 2, audio_t, dtype=dtype)
    return {"samples": nested_cls((video, audio))}


def text_conditioning(
    text_len: int = TINY_TEXT_LEN,
    dim: int = 16,
    *,
    tags: bool = True,
    seed: int = 3,
) -> List[List[Any]]:
    """One positive T2VA conditioning entry, as the official CLIP produces it."""
    generator = torch.Generator().manual_seed(seed)
    cond = torch.randn(1, text_len, dim, generator=generator)
    extras: Dict[str, Any] = {"pooled_output": None}
    if tags:
        extras["minimax_token_tags"] = torch.ones(text_len, dtype=torch.long)
    return [[cond, extras]]


# --------------------------------------------------------------------------
# fake model + patcher
# --------------------------------------------------------------------------
@dataclass
class Call:
    """One recorded model call."""

    kind: str                      # "prefill" | "noise" | "clean"
    chunk_index: Optional[int] = None
    video_sigma: Optional[float] = None
    audio_sigma: Optional[float] = None
    video_shape: Tuple[int, ...] = ()
    audio_shape: Tuple[int, ...] = ()
    update_cache: bool = False
    committed_before: int = 0
    retained_before: Tuple[int, ...] = ()
    options_id: int = 0
    has_video_eps: bool = False
    has_audio_eps: bool = False
    compute_dtype: Any = None


def fake_velocity(video: torch.Tensor, audio: torch.Tensor):
    """A deterministic stand-in for the DiT head.

    Any pure function of the input would do; this one is affine so the test's
    reference implementation can reproduce it exactly and the whole loop --
    draw order included -- can be compared tensor for tensor.
    """
    return [video * -0.5 + 0.25, audio * -0.25 + 0.5]


class FakeCausalDiT:
    """Duck-type of ``RavenCausalMiniMaxH3Model``'s causal API."""

    def __init__(self, num_layers: int = 2, dtype: Any = None) -> None:
        self.blocks = [object() for _ in range(num_layers)]
        self.calls: List[Call] = []
        if dtype is not None:
            #: the real model exposes its compute dtype; the sampler reads it
            self.dtype = dtype

    # -- cache protocol, same shape as the real block stack --
    def _stage_and_commit(self, cache, rows: int, role: str) -> None:
        for layer in range(len(self.blocks)):
            key = torch.zeros(rows, FAKE_HEADS, FAKE_HEAD_DIM)
            value = torch.zeros(rows, FAKE_HEADS, FAKE_HEAD_DIM)
            cache.stage(layer, key, value)
        cache.commit(role=role)

    def prefill_text(
        self,
        context,
        *,
        cache,
        transformer_options=None,
        text_token_tags=None,
        compute_dtype=None,
    ) -> int:
        self.calls.append(
            Call(
                kind="prefill",
                video_shape=tuple(context.shape),
                committed_before=cache.committed_chunks,
                retained_before=tuple(cache.retained_indices),
                options_id=id(transformer_options),
                has_video_eps=text_token_tags is not None,
                compute_dtype=compute_dtype,
            )
        )
        text_len = int(context.shape[1])
        self._stage_and_commit(cache, text_len, "text")
        return text_len

    def forward_chunk(
        self,
        *,
        video_latent,
        audio_latent,
        layout,
        chunk_index,
        cache=None,
        role="noise",
        video_sigma=None,
        audio_sigma=None,
        video_eps=None,
        audio_eps=None,
        update_cache=None,
        transformer_options=None,
        compute_dtype=None,
    ):
        chunk = layout.chunks[chunk_index]
        self.calls.append(
            Call(
                kind=str(role),
                chunk_index=int(chunk_index),
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                video_shape=tuple(video_latent.shape),
                audio_shape=tuple(audio_latent.shape),
                update_cache=bool(update_cache),
                committed_before=cache.committed_chunks if cache is not None else -1,
                retained_before=(
                    tuple(cache.retained_indices) if cache is not None else ()
                ),
                options_id=id(transformer_options),
                has_video_eps=video_eps is not None,
                has_audio_eps=audio_eps is not None,
                compute_dtype=compute_dtype,
            )
        )
        if role == "clean":
            assert update_cache, "a clean fill exists for the K/V it writes"
            assert video_eps is not None and audio_eps is not None
            self._stage_and_commit(cache, chunk.rows, "clean")
            return [torch.zeros_like(video_latent, dtype=torch.float32),
                    torch.zeros_like(audio_latent, dtype=torch.float32)]
        assert not update_cache, "a noise forward must never write the cache"
        assert video_sigma is not None and audio_sigma is not None
        video_v, audio_v = fake_velocity(video_latent, audio_latent)
        # The real ``forward_chunk`` returns its fp32 output heads unrounded --
        # ``FinalLayer.video_out``/``audio_out`` are the checkpoint's fp32
        # island and RAVEN's x0 model converts in fp32 -- so this duck-type
        # returns fp32 too, whatever dtype the latents came in. For an fp32
        # LATENT (the ComfyUI default, and every other test here) it is a no-op.
        return [video_v.to(torch.float32), audio_v.to(torch.float32)]


class FakeBaseModel:
    def __init__(self, diffusion_model) -> None:
        self.diffusion_model = diffusion_model


class FakePatcher:
    """Enough of ``comfy.model_patcher.ModelPatcher`` for the contract checks."""

    def __init__(
        self,
        diffusion_model=None,
        *,
        transformer_options: Optional[Dict[str, Any]] = None,
        model_options: Optional[Dict[str, Any]] = None,
        wrappers: Optional[Dict[str, Any]] = None,
        dynamic: bool = False,
        device: Any = "cpu",
    ) -> None:
        self.model = FakeBaseModel(diffusion_model or FakeCausalDiT())
        self.model_options = model_options if model_options is not None else {
            "transformer_options": transformer_options if transformer_options is not None else {}
        }
        self.wrappers = wrappers if wrappers is not None else {}
        self.load_device = torch.device(device)
        self.offload_device = torch.device("cpu")
        self._dynamic = bool(dynamic)

    def is_dynamic(self) -> bool:
        return self._dynamic


@dataclass
class LoadRecorder:
    """Stands in for ``comfy.model_management.load_models_gpu``."""

    calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = field(default_factory=list)

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


# --------------------------------------------------------------------------
# an independent replay of the loop, for tensor-for-tensor comparison
# --------------------------------------------------------------------------
def reference_rollout(
    layout, config: SamplerConfig, *, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RAVEN's rollout, written out longhand against :func:`fake_velocity`.

    Deliberately a second implementation rather than a call into the sampler:
    it is what makes the draw order, the ``x0 = x_t + sigma * v`` conversion and
    the ``(1 - s) * x0 + s * eps`` step falsifiable.
    """
    generator = torch.Generator().manual_seed(int(config.seed))
    video_shape = layout.video_latent_shape(24)
    audio_shape = layout.audio_latent_shape(32)

    video_noise = torch.randn(video_shape, generator=generator, dtype=dtype)
    audio_noise = torch.randn(audio_shape, generator=generator, dtype=dtype)
    torch.randn(video_shape, generator=generator, dtype=dtype)  # video clean eps
    torch.randn(audio_shape, generator=generator, dtype=dtype)  # audio clean eps

    video_x0 = torch.zeros_like(video_noise)
    audio_x0 = torch.zeros_like(audio_noise)
    video_steps = step_pairs(config.video_sigmas)
    audio_steps = step_pairs(config.audio_sigmas)

    for chunk in layout.chunks:
        video_xt = video_noise[:, :, chunk.video_start:chunk.video_stop].clone()
        audio_xt = audio_noise[:, :, :, chunk.audio_start:chunk.audio_stop].clone()
        for step in range(config.steps):
            video_sigma, video_next = video_steps[step]
            audio_sigma, audio_next = audio_steps[step]
            video_v, audio_v = fake_velocity(video_xt, audio_xt)
            video_pred = video_xt + video_sigma * video_v
            audio_pred = audio_xt + audio_sigma * audio_v
            video_eps = torch.randn(tuple(video_xt.shape), generator=generator, dtype=dtype)
            audio_eps = torch.randn(tuple(audio_xt.shape), generator=generator, dtype=dtype)
            video_xt = (1.0 - video_next) * video_pred + video_next * video_eps
            audio_xt = (1.0 - audio_next) * audio_pred + audio_next * audio_eps
        video_x0[:, :, chunk.video_start:chunk.video_stop] = video_xt
        audio_x0[:, :, :, chunk.audio_start:chunk.audio_stop] = audio_xt
    return video_x0, audio_x0
