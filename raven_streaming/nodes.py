"""The two ComfyUI nodes: ``RAVEN Model Loader`` and ``RAVEN Streaming Sampler``.

Node contract (``docs/architecture.md`` §2), stated once:

* **exactly two nodes**, registered through the V1 ``NODE_CLASS_MAPPINGS``
  surface. There is deliberately no ``comfy_entrypoint`` / V3 extension here:
  upstream tries V1 first (``nodes.py::load_custom_node``), and defining both
  would make "which schema is live?" depend on load order;
* the loader takes the **full non-pruned BF16 H3 DiT** and the **mandatory
  RAVEN LoRA** and returns a stock ``MODEL``;
* the sampler takes that ``MODEL``, the official T2VA ``CONDITIONING``, the
  official **empty** H3 AV ``LATENT``, and the two H3 VAEs -- and nothing else.
  No negative, no CFG, no sampler/scheduler pickers, and no ``width`` /
  ``height`` / ``frames``: every one of those either does not exist in this
  sampling regime or is already carried by the latent, and offering a second
  place to state it is how a workflow ends up silently generating something
  other than what its latent says. Keyframe / reference conditioning extras are
  refused because this sampler has not implemented the causal packed layout for
  condition rows -- a runtime limit of the implementation, not a claim about
  what the RAVEN LoRA can do.

Import weight
-------------
This module is importable in a bare interpreter: torch and the RAVEN lanes come
in at import, ComfyUI does not. ``folder_paths``, ``comfy.model_management``,
``comfy_execution.utils`` and ``comfy_extras.nodes_audio`` are all resolved
lazily inside the call that needs them, so the schema and every helper below can
be tested without a ComfyUI checkout.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from raven_streaming import consistency
from raven_streaming import contracts
from raven_streaming import layout as layout_mod
from raven_streaming import loader as loader_mod
from raven_streaming import lora as lora_mod
from raven_streaming import runtime_linear
from raven_streaming import streaming_pipeline as pipeline_mod
from raven_streaming.cache import ChunkKVCache

__all__ = [
    "RAVEN_LORA_STRENGTH",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VIDEO_VAE_INNER_CLASS",
    "AUDIO_VAE_INNER_CLASS",
    "NodeInputError",
    "RAVENModelLoader",
    "RAVENStreamingSampler",
    "resolve_video_vae",
    "resolve_audio_vae",
    "decode_workspace_bytes",
    "DiTDimensions",
    "RolloutMemoryBudget",
    "dit_dimensions",
    "dtype_size",
    "estimate_rollout_budget",
    "rollout_memory_budget",
    "make_load_models",
    "KV_CACHE_STORAGE_CHOICES",
    "DEFAULT_KV_CACHE_STORAGE",
    "GPU_HARD_CAP_BYTES",
    "CAP_PLANNING_HEADROOM_BYTES",
    "PLANNING_BUDGET_BYTES",
    "planning_bytes",
    "kv_on_gpu",
    "kv_host_pinned",
    "DeviceMemoryFacts",
    "PhasePlan",
    "JointOffloadPlan",
    "plan_joint_offload",
    "offload_envelope_bytes",
    "largest_module_bytes",
    "PhaseResidency",
    "measure_residency",
    "report_residency",
    "gpu_memory_state",
    "hard_cap_watch",
    "PhaseSwapCoordinator",
    "FINAL_DECODE_UNLOAD_STRATEGY",
    "FINAL_AUDIO_UNLOAD_STRATEGY",
    "FinalDecodeHandover",
    "prepare_final_decode",
    "prepare_final_audio_decode",
    "decode_images",
    "decode_audio",
    "normalise_node_id",
    "executing_identity",
    "preview_sink",
]

LOG = logging.getLogger(__name__)


#: RAVEN's residual strength, fixed at the published value.
#:
#: Not a node input. The adapter is what makes the model a RAVEN model at all
#: (``docs/requirements.md``: mandatory, not a quality knob), so 0 -- "run the
#: base H3 DiT through a chunk-major consistency loop it was never trained for"
#: -- is not a state this node can be put in. A programmatic caller can still
#: reach ``loader.load_raven_diffusion_model(strength=...)``.
RAVEN_LORA_STRENGTH = 1.0

#: Inner modules the two ``VAE`` sockets must be carrying.
VIDEO_VAE_INNER_CLASS = "MiniMaxH3VideoVAE"
AUDIO_VAE_INNER_CLASS = "MiniMaxH3AudioVAE"

GIB = 1024 ** 3

#: The reference card this package measures itself against, in bytes.
#: **A constant, and deliberately not an input.**
#:
#: It is not a cap: this node does not enforce a VRAM ceiling, because the
#: allocator and ``comfy.model_management`` already do that with numbers this
#: node cannot see. What the constant is for is a yardstick -- the residency
#: record and :func:`hard_cap_watch` say whether a run would have fitted the
#: smallest card this pack targets. The 24 GiB path itself is *tested* by making
#: a device actually that small from outside the process (a VRAM reserve process
#: on the big box), never by this node pretending a bigger card is smaller.
#:
#: There is no widget for it, and no widget for weight residency either. A
#: "max resident GB" input would be a second, worse estimate of free memory
#: sitting in front of upstream's real one: every time the two disagreed, the
#: user's number would be the one turning a run that fitted into an OOM, or a
#: run that fitted into a refusal.
GPU_HARD_CAP_BYTES = 24 * GIB

#: Reporting sits *below* the reference card by this much, so "within budget"
#: means "fitted with room for one allocation this package does not model"
#: (an allocator block, a cuBLAS workspace) rather than "fitted exactly".
CAP_PLANNING_HEADROOM_BYTES = 2 * GIB

#: What a phase's planned peak is compared against in the log: 22 GiB.
PLANNING_BUDGET_BYTES = GPU_HARD_CAP_BYTES - CAP_PLANNING_HEADROOM_BYTES

#: Where the chunk KV cache lives (``consistency.SamplerConfig.kv_cache_storage``).
#:
#: ``cpu_pinned`` -- host memory, page-locked, so the per-chunk copies overlap
#: with compute; ``cpu`` -- plain host memory; ``gpu`` -- the whole cache stays
#: on the compute device, which costs ~28 GiB at the published request on its
#: own and is why the host-backed default exists.
KV_CACHE_STORAGE_CHOICES = ("cpu_pinned", "cpu", "gpu")
DEFAULT_KV_CACHE_STORAGE = "cpu_pinned"


def kv_on_gpu(storage: Any) -> bool:
    """Does this storage mode keep the committed KV cache on the card?"""
    return str(storage) == "gpu"


def kv_host_pinned(storage: Any) -> bool:
    """Does this storage mode page-lock the host copy?"""
    return str(storage) == "cpu_pinned"


def planning_bytes() -> int:
    """The reference budget a phase's planned peak is reported against."""
    return int(PLANNING_BUDGET_BYTES)

_LOADER_TOOLTIPS: Dict[str, str] = {
    entry.name: entry.tooltip for entry in loader_mod.RAVEN_LOADER_INPUTS
}


class NodeInputError(ValueError):
    """A socket carries something this node cannot use."""


# --------------------------------------------------------------------------
# schema helpers
# --------------------------------------------------------------------------


def _filename_list(folder: str) -> List[str]:
    """``folder_paths.get_filename_list(folder)``, or ``[]`` outside ComfyUI.

    An empty combo is what upstream's own loaders show when a folder is empty;
    it is not an error, and it must not stop this module from importing.
    """
    try:
        import folder_paths  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - a bare environment is a supported mode
        return []
    try:
        return list(folder_paths.get_filename_list(folder))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("raven: could not list %s (%s: %s)", folder, type(exc).__name__, exc)
        return []


# --------------------------------------------------------------------------
# Node 1: RAVEN Model Loader
# --------------------------------------------------------------------------


def _causal_model_class() -> type:
    """The chunk-causal DiT class the loader injects as ``unet_model``.

    Behind a function because :mod:`raven_streaming.causal_model` subclasses
    ``comfy.ldm.minimax.model`` and therefore imports ComfyUI at module scope;
    keeping that off this module's import path is what lets the schema be read
    (and tested) without a checkout.
    """
    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model

    return RavenCausalMiniMaxH3Model


class RAVENModelLoader:
    """Full non-pruned BF16 H3 DiT + the mandatory RAVEN LoRA -> stock ``MODEL``."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "unet_name": (
                    _filename_list(loader_mod.DIFFUSION_MODEL_FOLDER),
                    {
                        "tooltip": _LOADER_TOOLTIPS["unet_name"]
                        + " The RAVEN adapter is trained against the full BF16 "
                        "weights; the pruned/adaln-curve checkpoint has no "
                        "time_embedder for its 266-module mapping to attach to."
                    },
                ),
                "lora_name": (
                    _filename_list(loader_mod.LORA_FOLDER),
                    {
                        "tooltip": _LOADER_TOOLTIPS["lora_name"]
                        + " About 5 GB of FP32 A/B tensors, applied as an "
                        "activation residual (never fused into the BF16 weights) "
                        "and counted by Comfy's memory accounting from the first "
                        "model_size() call."
                    },
                ),
                "weight_dtype": (
                    list(loader_mod.WEIGHT_DTYPE_CHOICES),
                    {
                        "default": "default",
                        "tooltip": _LOADER_TOOLTIPS["weight_dtype"]
                        + " There is no FP8/INT8 choice on purpose: comfy.ops "
                        "fuses quantised linears and would silently skip the "
                        "residual.",
                    },
                ),
            }
        }

    RETURN_TYPES = loader_mod.RAVEN_LOADER_RETURN_TYPES  # ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    OUTPUT_TOOLTIPS = (
        "A standard ComfyUI MODEL (stock static ModelPatcher) whose diffusion_model "
        "is the chunk-causal RAVEN DiT. Stock LoraLoaderModelOnly can be chained "
        "after it, and nothing here restricts it to one generation mode - other "
        "official H3 workflows can be explored with it, unverified and unsupported. "
        "The RAVEN Streaming Sampler in this pack is the narrower part: it refuses "
        "conditioning carrying condition/reference rows it has not implemented.",
    )
    FUNCTION = "load_model"
    CATEGORY = "model/loaders/raven"
    DESCRIPTION = (
        "Loads the official full, non-pruned BF16 MiniMax H3 diffusion model together "
        "with the mandatory RAVEN PEFT LoRA and returns a standard MODEL.\n\n"
        "The RAVEN residual strength is fixed at 1.0: the adapter is what the "
        "streaming sampler's 4-NFE consistency schedule was trained with, not a "
        "quality knob, so there is no 'off' setting.\n\n"
        "Weights land on the stock ModelPatcher partial CPU offload path (v0.1's "
        "only supported memory strategy). 'fp32' doubles the base weights to 132 GB+ "
        "on top of the FP32 residual - expect heavy offload traffic or OOM."
    )
    SEARCH_ALIASES = ["raven", "minimax h3", "raven lora", "h3 loader"]

    def load_model(self, unet_name: str, lora_name: str, weight_dtype: str = "default"):
        # No try/except anywhere in here on purpose: a refused checkpoint, a
        # LoRA whose 266-module mapping does not fit, a missing file -- each
        # already carries the reason it failed, and wrapping it would only
        # bury that under a generic message.
        model = loader_mod.load_raven_diffusion_model(
            unet_name,
            lora_name,
            strength=RAVEN_LORA_STRENGTH,
            weight_dtype=weight_dtype,
            unet_model_cls=_causal_model_class(),
            force_static_patcher=True,
            disable_dynamic=True,
        )
        return (model,)


# --------------------------------------------------------------------------
# VAE resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedVAE:
    """A validated ``VAE`` socket: the wrapper, its inner module, its geometry."""

    vae: Any
    inner: Any
    kind: str

    @property
    def patcher(self) -> Any:
        return getattr(self.vae, "patcher", None)


def _comfy_vae_class() -> Optional[type]:
    try:
        import comfy.sd  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    return getattr(comfy.sd, "VAE", None)


def _require_vae_wrapper(vae: Any, socket: str) -> Any:
    """The socket must carry a ``comfy.sd.VAE``, and it must be usable."""
    if vae is None:
        raise NodeInputError(f"{socket} is required")
    expected = _comfy_vae_class()
    if expected is not None and not isinstance(vae, expected):
        raise NodeInputError(
            f"{socket} must be a comfy.sd.VAE (the output of VAELoader), got "
            f"{type(vae).__name__}"
        )
    missing = [
        name
        for name in ("first_stage_model", "patcher", "decode", "memory_used_decode")
        if getattr(vae, name, None) is None
    ]
    if missing:
        raise NodeInputError(
            f"{socket} ({type(vae).__name__}) is missing {missing}; this is not a "
            "loaded ComfyUI VAE"
        )
    inner = vae.first_stage_model
    if inner is None:
        raise NodeInputError(
            f"{socket} has no weights loaded (first_stage_model is None)"
        )
    return inner


def _swapped_hint(inner_name: str, socket: str) -> str:
    if socket == "video_vae" and inner_name == AUDIO_VAE_INNER_CLASS:
        return " -- this is the audio VAE; the two VAE sockets are swapped"
    if socket == "audio_vae" and inner_name == VIDEO_VAE_INNER_CLASS:
        return " -- this is the video VAE; the two VAE sockets are swapped"
    return ""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise NodeInputError(message)


def resolve_video_vae(video_vae: Any) -> ResolvedVAE:
    """Feature-probe the ``video_vae`` socket. Loud, by name, on every mismatch.

    Everything checked here is something the streaming lane *uses*: the inner
    class (whose ``_adaptive_decode`` / ``blend`` / ``_finalize_pixels`` the
    incremental coordinator drives directly), the 24 latent channels the sampler
    produces, and the temporal geometry the 5+2 chunk machine is derived from.
    """
    inner = _require_vae_wrapper(video_vae, "video_vae")
    name = type(inner).__name__
    _check(
        name == VIDEO_VAE_INNER_CLASS,
        f"video_vae holds a {name}, expected the MiniMax H3 video VAE "
        f"({VIDEO_VAE_INNER_CLASS}){_swapped_hint(name, 'video_vae')}",
    )
    _check(
        int(getattr(video_vae, "latent_channels", 0)) == contracts.VIDEO_LATENT_CHANNELS,
        f"video_vae reports {getattr(video_vae, 'latent_channels', None)} latent "
        f"channels, expected {contracts.VIDEO_LATENT_CHANNELS}",
    )
    _check(
        int(getattr(video_vae, "latent_dim", 0)) == 3,
        f"video_vae reports latent_dim={getattr(video_vae, 'latent_dim', None)}, "
        "expected 3 (a video VAE)",
    )
    for attribute in ("_adaptive_decode", "blend", "_finalize_pixels"):
        _check(
            callable(getattr(inner, attribute, None)),
            f"video_vae's {name} has no {attribute}(); the incremental decoder "
            "drives the temporal chunk machine directly and cannot run without it",
        )
    for attribute in ("clip_length", "vae_ratio_t", "token_drop"):
        _check(
            getattr(inner, attribute, None) is not None,
            f"video_vae's {name} has no {attribute}; the 5+2 chunk geometry is "
            "read off the model, never assumed",
        )
    return ResolvedVAE(vae=video_vae, inner=inner, kind="video")


def resolve_audio_vae(audio_vae: Any) -> ResolvedVAE:
    """Feature-probe the ``audio_vae`` socket."""
    inner = _require_vae_wrapper(audio_vae, "audio_vae")
    name = type(inner).__name__
    _check(
        name == AUDIO_VAE_INNER_CLASS,
        f"audio_vae holds a {name}, expected the MiniMax H3 audio VAE "
        f"({AUDIO_VAE_INNER_CLASS}){_swapped_hint(name, 'audio_vae')}",
    )
    _check(
        int(getattr(audio_vae, "latent_channels", 0)) == contracts.AUDIO_LATENT_CHANNELS,
        f"audio_vae reports {getattr(audio_vae, 'latent_channels', None)} latent "
        f"channels, expected {contracts.AUDIO_LATENT_CHANNELS}",
    )
    _check(
        int(getattr(audio_vae, "output_channels", 0)) == pipeline_mod.AUDIO_CHANNELS,
        f"audio_vae reports {getattr(audio_vae, 'output_channels', None)} output "
        f"channel(s), expected stereo ({pipeline_mod.AUDIO_CHANNELS})",
    )
    _check(
        int(getattr(audio_vae, "audio_sample_rate", 0)) == 32000,
        f"audio_vae reports {getattr(audio_vae, 'audio_sample_rate', None)} Hz, "
        "expected the H3 audio VAE's 32000",
    )
    _check(
        int(getattr(audio_vae, "upscale_ratio", 0)) == 800,
        f"audio_vae reports upscale_ratio={getattr(audio_vae, 'upscale_ratio', None)}, "
        "expected 800 samples per latent",
    )
    _check(
        callable(getattr(inner, "decode", None)),
        f"audio_vae's {name} has no decode(); the overlap-save decoder calls it "
        "per block",
    )
    return ResolvedVAE(vae=audio_vae, inner=inner, kind="audio")


# --------------------------------------------------------------------------
# model loading: one co-resident load for the whole execution
# --------------------------------------------------------------------------


def decode_workspace_bytes(
    video: ResolvedVAE,
    audio: ResolvedVAE,
    config: pipeline_mod.PipelineConfig,
    latent_h: int,
    latent_w: int,
) -> int:
    """VRAM to reserve for the streaming decodes, using upstream's own estimates.

    The larger of one video chunk (7 latents, the widest the incremental
    coordinator ever hands over) and one overlap-save audio block (block plus
    both margins). ``max``, not a sum: the two decodes never run at the same
    time, and reserving both would evict DiT weights that are needed every step.

    A **whole-clip** video decode is deliberately not priced here, because the
    node no longer performs one: the video collector writes the IMAGE frame by
    frame during the rollout, so the largest video allocation this execution
    ever makes is the one chunk below. The finished IMAGE itself is host
    memory, not VRAM (``PipelineReport.image_bytes``: 2.43 GB at 192 frames,
    4.59 GB at 362), and Comfy's reserve does not cover host RAM.

    The audio term stands whether or not a preview runs -- the preview's
    overlap-save blocks and the final ``vae_decode_audio`` are the same order
    of magnitude, and the reserve is held for the whole execution either way.
    """
    total = 0
    try:
        from raven_streaming.media.video_stream import VideoChunkParams

        params = VideoChunkParams.from_vae(video.inner)
        video_shape = (
            1,
            contracts.VIDEO_LATENT_CHANNELS,
            params.latents_needed,
            int(latent_h),
            int(latent_w),
        )
        total = max(
            total,
            int(video.vae.memory_used_decode(video_shape, video.vae.vae_dtype)),
        )
    except Exception as exc:  # noqa: BLE001 - an estimate, never a hard failure
        LOG.warning(
            "raven: could not estimate the video decode workspace (%s: %s); "
            "reserving nothing for it",
            type(exc).__name__,
            exc,
        )
    try:
        audio_shape = (
            1,
            contracts.AUDIO_LATENT_CHANNELS,
            pipeline_mod.AUDIO_CHANNELS,
            config.audio_decode_latents,
        )
        total = max(
            total,
            int(audio.vae.memory_used_decode(audio_shape, audio.vae.vae_dtype)),
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "raven: could not estimate the audio decode workspace (%s: %s); "
            "reserving nothing for it",
            type(exc).__name__,
            exc,
        )
    return int(total)


# --------------------------------------------------------------------------
# rollout VRAM budget
# --------------------------------------------------------------------------

#: Activation copies of one chunk that are live at once inside a DiT block.
#: Read off ``comfy.ldm.minimax.model.DiTBlock`` / ``Attention`` / ``MLP``:
#: the block's input ``x`` and the residual it is added back to, the attention
#: branch's output and the MLP branch's output -- four hidden-wide tensors.
HIDDEN_ACTIVATION_COPIES = 4
#: ``Attention.qkv_proj`` produces one fused ``3 * inner`` buffer, and the
#: attention output is one more ``inner``-wide tensor before ``out_proj``.
QKV_ACTIVATION_COPIES = 3
ATTENTION_OUTPUT_COPIES = 1
#: ``MLP.fc1`` writes ``2 * ffn`` (gate + up) and the gated activation is one
#: more ``ffn``-wide tensor.
MLP_ACTIVATION_COPIES = 3
#: Full-clip tensors the rollout holds for its whole life, per stream: the
#: initial noise, the clean-context eps and the x0 accumulator
#: (``consistency._Rollout._run``).
FULL_CLIP_TENSORS = 3
#: Chunk-sized tensors live inside one consistency step: ``x_t``, the model's
#: velocity, ``pred_x0``, the fresh eps, the next ``x_t``, plus one spare for
#: the temporary the expression builds.
STEP_TENSORS = 6
#: Fraction of the estimate added as head-room, and the floor under it. The
#: estimate models the tensors this package knows about; allocator
#: fragmentation, cuBLAS/cuDNN workspaces and the attention backend's own
#: scratch are not individually modelled, and under-reserving is the expensive
#: mistake (Comfy would load weights into space the rollout then needs).
SAFETY_FRACTION = 0.12
SAFETY_FLOOR_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DiTDimensions:
    """The shape numbers the budget needs, measured off the live DiT.

    ``measured`` lists what was actually read from the model; anything else
    fell back to the published full-size numbers
    (:class:`raven_streaming.lora.RavenBaseConfig`). The distinction is
    reported rather than hidden: a budget computed from defaults is a guess
    about *which model this is*, not just about its memory.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    ffn_hidden_size: int
    compute_dtype_size: int
    measured: Tuple[str, ...] = ()

    @property
    def inner_dim(self) -> int:
        return int(self.num_heads) * int(self.head_dim)

    def describe(self) -> Dict[str, Any]:
        return {
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "head_dim": int(self.head_dim),
            "inner_dim": self.inner_dim,
            "hidden_size": int(self.hidden_size),
            "ffn_hidden_size": int(self.ffn_hidden_size),
            "compute_dtype_size": int(self.compute_dtype_size),
            "measured": list(self.measured),
        }


def dtype_size(dtype: Any, default: int = 2) -> int:
    """Bytes per element of a torch dtype, without importing torch here."""
    if dtype is None:
        return int(default)
    size = getattr(dtype, "itemsize", None)  # torch >= 2.1
    if isinstance(size, int) and size > 0:
        return size
    try:
        import torch

        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:  # noqa: BLE001 - an estimate, never a hard failure
        return int(default)


def dit_dimensions(model: Any) -> DiTDimensions:
    """Measure the DiT's shape off the live module tree.

    Deliberately forgiving: every number has a published fallback, because this
    feeds a *reservation*. A model whose blocks are not introspectable should
    still run -- with a budget computed from the full-size numbers, which is the
    conservative direction -- rather than fail at the socket.
    """
    defaults = lora_mod.RavenBaseConfig()
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", model)
    measured: List[str] = []

    def _measure(name: str, value: Any, fallback: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return int(fallback)
        if number <= 0:
            return int(fallback)
        measured.append(name)
        return number

    blocks = getattr(diffusion_model, "blocks", None)
    num_layers = _measure(
        "num_layers", len(blocks) if blocks is not None else None, defaults.num_layers
    )
    block = blocks[0] if blocks is not None and len(blocks) else None
    attention = getattr(block, "attn", None)
    num_heads = _measure(
        "num_heads", getattr(attention, "heads", None), defaults.num_attention_heads
    )
    head_dim = _measure(
        "head_dim", getattr(attention, "head_dim", None), defaults.attention_head_dim
    )
    hidden_size = _measure(
        "hidden_size", getattr(diffusion_model, "hidden_size", None), defaults.hidden_size
    )
    # fc1 emits gate + up, i.e. twice the FFN width
    fc1_out = getattr(getattr(getattr(block, "mlp", None), "fc1", None), "out_features", None)
    ffn = _measure(
        "ffn_hidden_size",
        None if fc1_out is None else int(fc1_out) // 2,
        defaults.ffn_hidden_size,
    )
    compute_dtype = getattr(diffusion_model, "dtype", None)
    dtype_bytes = dtype_size(compute_dtype)
    if compute_dtype is not None:
        measured.append("compute_dtype_size")

    return DiTDimensions(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn,
        compute_dtype_size=dtype_bytes,
        measured=tuple(measured),
    )


@dataclass(frozen=True)
class RolloutMemoryBudget:
    """Itemised VRAM the rollout needs *besides* the weights.

    ``total_bytes`` is what goes to ``load_models_gpu(memory_required=...)``:
    everything Comfy must keep free after it has decided how much of the DiT
    and the two VAEs to make resident.

    ``kv_cache_bytes`` is **GPU-resident** KV only. With a host-backed cache
    (``kv_cache_storage`` ``cpu`` / ``cpu_pinned``) it is 0, the committed
    chunks are priced in ``cpu_kv_steady_bytes`` / ``cpu_kv_peak_bytes`` -- host
    memory, which no VRAM cap governs -- and the card only ever carries
    ``kv_slot_bytes``: the one merged (retained + current) K/V slot that the
    layer currently executing gathers back onto the device.
    """

    kv_cache_bytes: int
    forward_workspace_bytes: int
    rollout_buffer_bytes: int
    decode_workspace_bytes: int
    safety_bytes: int
    total_bytes: int
    kv_slot_bytes: int = 0
    cpu_kv_steady_bytes: int = 0
    cpu_kv_peak_bytes: int = 0
    kv_cache_storage: str = "gpu"
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def kv_host_pinned(self) -> bool:
        return kv_host_pinned(self.kv_cache_storage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_cache_bytes": int(self.kv_cache_bytes),
            "forward_workspace_bytes": int(self.forward_workspace_bytes),
            "rollout_buffer_bytes": int(self.rollout_buffer_bytes),
            "decode_workspace_bytes": int(self.decode_workspace_bytes),
            "safety_bytes": int(self.safety_bytes),
            "total_bytes": int(self.total_bytes),
            "kv_slot_bytes": int(self.kv_slot_bytes),
            "cpu_kv_steady_bytes": int(self.cpu_kv_steady_bytes),
            "cpu_kv_peak_bytes": int(self.cpu_kv_peak_bytes),
            "kv_cache_storage": str(self.kv_cache_storage),
            "detail": dict(self.detail),
        }

    def describe(self) -> str:
        gib = float(GIB)
        kv = (
            "KV {:.2f} (gpu)".format(self.kv_cache_bytes / gib)
            if kv_on_gpu(self.kv_cache_storage)
            else "KV slot {slot:.3f} (host {peak:.2f} peak / {steady:.2f} steady, {pin})".format(
                slot=self.kv_slot_bytes / gib,
                peak=self.cpu_kv_peak_bytes / gib,
                steady=self.cpu_kv_steady_bytes / gib,
                pin="pinned" if self.kv_host_pinned else "pageable",
            )
        )
        return (
            "raven rollout reserve: {total:.2f} GiB = {kv} + buffers {buf:.2f} "
            "+ max(forward {fwd:.2f}, decode {dec:.2f}) + safety {safe:.2f} "
            "[{rows} peak KV rows, {layers}x{heads}x{head_dim} @ {dtype}B]".format(
                total=self.total_bytes / gib,
                kv=kv,
                buf=self.rollout_buffer_bytes / gib,
                fwd=self.forward_workspace_bytes / gib,
                dec=self.decode_workspace_bytes / gib,
                safe=self.safety_bytes / gib,
                rows=self.detail.get("kv_peak_rows"),
                layers=self.detail.get("num_layers"),
                heads=self.detail.get("num_heads"),
                head_dim=self.detail.get("head_dim"),
                dtype=self.detail.get("compute_dtype_size"),
            )
        )


def _committed_chunk_rows(layout: layout_mod.T2VALayout, text_len: int) -> List[int]:
    """Rows of every chunk that is ever written into the KV cache, in order.

    Cache chunk 0 is the text prefill. The rollout's **last** media chunk is
    never committed -- nothing reads that history -- so it is not here either
    (``consistency._Rollout._run``).
    """
    rows = [max(1, int(text_len))]
    rows.extend(int(chunk.rows) for chunk in layout.chunks[:-1])
    return rows


def _kv_peak_rows(
    chunk_rows: Sequence[int], sink: int, window: Optional[int]
) -> Tuple[int, int, int]:
    """``(peak rows held, peak rows gathered, rows retained at rest)``.

    The retention policy is not re-derived here: :class:`ChunkKVCache` is asked,
    with the node's own ``sink`` / ``window``, which chunk indices it would keep
    after ``n`` commits (``retained_index_set`` is pure integer arithmetic and
    allocates nothing). If eviction ever changes, this estimate changes with it
    instead of drifting away from it.

    The peak is ``retained + the chunk being staged``: ``stage()`` copies a
    chunk's K/V for every layer before ``commit()`` releases the evicted ones,
    so both are resident at the same moment. ``gathered`` is the separate
    ``torch.cat`` of the retained keys/values that ``retained()`` builds for one
    layer at a time.
    """
    policy = ChunkKVCache(1, sink=int(sink), window=window)
    peak_held = 0
    peak_gathered = 0
    for step in range(len(chunk_rows)):
        retained = sum(chunk_rows[i] for i in policy.retained_index_set(step))
        peak_gathered = max(peak_gathered, retained)
        peak_held = max(peak_held, retained + chunk_rows[step])
    settled = sum(chunk_rows[i] for i in policy.retained_index_set(len(chunk_rows)))
    return max(peak_held, settled), max(peak_gathered, settled), settled


def estimate_rollout_budget(
    *,
    layout: layout_mod.T2VALayout,
    text_len: int,
    sink: int,
    window: Optional[int],
    dims: DiTDimensions,
    latent_dtype_size: int = 4,
    decode_workspace_bytes: int = 0,
    lora_temp_bytes: int = runtime_linear.DEFAULT_TEMP_BUDGET_BYTES,
    safety_fraction: float = SAFETY_FRACTION,
    safety_floor_bytes: int = SAFETY_FLOOR_BYTES,
    kv_cache_storage: str = "gpu",
) -> RolloutMemoryBudget:
    """Pure estimate of the VRAM one rollout needs beside the weights.

    Formula, all of it derived from code in this repository rather than from a
    measurement (see ``detail`` for every input)::

        kv_bytes_per_row = 2 * layers * heads * head_dim * compute_dtype_size
        kv_cache         = kv_peak_rows * kv_bytes_per_row
        kv_gather        = 2 * kv_gathered_rows * heads * head_dim * dtype
        activations      = widest_rows * dtype * (4*hidden + 3*inner + inner
                                                  + 3*ffn)
        forward          = activations + kv_gather + lora_fp32_temporaries
        buffers          = 3 * (video_clip + audio_clip) * latent_dtype
                         + 6 * (video_chunk + audio_chunk) * latent_dtype
        subtotal         = kv_cache + buffers + max(forward, decode)
        total            = subtotal + safety_fraction * subtotal + floor

    Why ``max`` and not a sum for the last pair: the streaming VAE decode runs
    inside ``on_chunk``, i.e. *between* DiT forwards, never during one. Adding
    them would reserve peak memory for a state the rollout is never in.

    Why the KV term dominates and must be counted at all: a retained chunk
    costs ``rows * 2 * layers * inner * dtype``, which for the published model
    at 1376x768 is about 7 GB **per chunk**. Reserving only the VAE's decode
    workspace -- a few hundred MB -- lets Comfy fill the card with weights and
    then OOM three chunks in, which reads like a model bug rather than a
    budgeting one.

    ``kv_cache_storage`` decides *where* that cache lives, and therefore which
    of the two shapes this estimate has:

    * ``gpu`` -- the pre-cap arrangement. The whole cache is device memory
      (``kv_cache_bytes``, ~28 GiB at the published request) and a cached
      forward additionally gathers the retained rows of one layer
      (``kv_gather_bytes``).
    * ``cpu`` / ``cpu_pinned`` -- the committed chunks are host tensors, so the
      *persistent* GPU cost is zero and the card only ever holds one merged
      ``retained + current`` K/V slot for the layer being executed
      (``kv_slot_bytes``: 0.56 GiB at 192 frames with a 128-token prompt). The
      host cost is reported as well, because it is real -- it is just not what
      a VRAM cap governs.
    """
    chunk_rows = _committed_chunk_rows(layout, text_len)
    peak_rows, gathered_rows, steady_rows = _kv_peak_rows(chunk_rows, sink, window)

    dtype = int(dims.compute_dtype_size)
    inner = dims.inner_dim
    kv_bytes_per_row = 2 * int(dims.num_layers) * inner * dtype
    kv_gather_bytes = 2 * gathered_rows * inner * dtype
    # One layer's merged (retained + the chunk being staged) K and V: what a
    # host-backed cache copies back onto the card to run one block.
    kv_slot_bytes = 2 * peak_rows * inner * dtype

    on_gpu = kv_on_gpu(kv_cache_storage)
    kv_cache_bytes = peak_rows * kv_bytes_per_row if on_gpu else 0
    cpu_kv_peak_bytes = 0 if on_gpu else peak_rows * kv_bytes_per_row
    cpu_kv_steady_bytes = 0 if on_gpu else steady_rows * kv_bytes_per_row

    # The prefill is one "chunk" too, and for a long prompt it can be the
    # widest one; the workspace has to cover whichever is.
    widest_rows = max([int(text_len)] + [int(chunk.rows) for chunk in layout.chunks])
    per_row_activation = (
        HIDDEN_ACTIVATION_COPIES * int(dims.hidden_size)
        + QKV_ACTIVATION_COPIES * inner
        + ATTENTION_OUTPUT_COPIES * inner
        + MLP_ACTIVATION_COPIES * int(dims.ffn_hidden_size)
    )
    activation_bytes = widest_rows * per_row_activation * dtype
    # With the cache on the host the gather *is* the slot: the retained rows are
    # copied up alongside the current chunk's, and nothing else of the cache is
    # on the card.
    kv_forward_bytes = kv_gather_bytes if on_gpu else kv_slot_bytes
    forward_workspace_bytes = activation_bytes + kv_forward_bytes + int(lora_temp_bytes)

    video_clip = (
        contracts.VIDEO_LATENT_CHANNELS * layout.latent_t * layout.latent_h * layout.latent_w
    )
    audio_clip = contracts.AUDIO_LATENT_CHANNELS * layout_mod.AUDIO_CHANNELS * layout.audio_t
    widest_chunk = max(layout.chunks, key=lambda c: c.video_latents)
    video_chunk = (
        contracts.VIDEO_LATENT_CHANNELS
        * widest_chunk.video_latents
        * layout.latent_h
        * layout.latent_w
    )
    audio_chunk = (
        contracts.AUDIO_LATENT_CHANNELS
        * layout_mod.AUDIO_CHANNELS
        * max(chunk.audio_latents for chunk in layout.chunks)
    )
    rollout_buffer_bytes = int(latent_dtype_size) * (
        FULL_CLIP_TENSORS * (video_clip + audio_clip)
        + STEP_TENSORS * (video_chunk + audio_chunk)
    )

    working = max(int(forward_workspace_bytes), int(decode_workspace_bytes))
    subtotal = int(kv_cache_bytes) + int(rollout_buffer_bytes) + working
    safety_bytes = int(math.ceil(subtotal * float(safety_fraction))) + int(safety_floor_bytes)

    detail: Dict[str, Any] = {
        "kv_peak_rows": int(peak_rows),
        "kv_gathered_rows": int(gathered_rows),
        "kv_steady_rows": int(steady_rows),
        "kv_bytes_per_row": int(kv_bytes_per_row),
        "kv_gather_bytes": int(kv_gather_bytes),
        "kv_slot_bytes": int(kv_slot_bytes),
        "kv_forward_bytes": int(kv_forward_bytes),
        "kv_cache_storage": str(kv_cache_storage),
        "kv_host_pinned": bool(kv_host_pinned(kv_cache_storage)),
        "canonical_cpu_kv_steady_bytes": int(cpu_kv_steady_bytes),
        "canonical_cpu_kv_peak_bytes": int(cpu_kv_peak_bytes),
        "activation_bytes": int(activation_bytes),
        "lora_fp32_temp_bytes": int(lora_temp_bytes),
        "widest_chunk_rows": int(widest_rows),
        "committed_chunks": len(chunk_rows),
        "chunk_rows": [int(r) for r in chunk_rows],
        "text_len": int(text_len),
        "sink": int(sink),
        "window": window,
        "frames": int(layout.frames),
        "width": int(layout.width),
        "height": int(layout.height),
        "frame_rows": int(layout.frame_rows),
        "latent_dtype_size": int(latent_dtype_size),
        "safety_fraction": float(safety_fraction),
        "safety_floor_bytes": int(safety_floor_bytes),
    }
    detail.update(dims.describe())

    return RolloutMemoryBudget(
        kv_cache_bytes=int(kv_cache_bytes),
        forward_workspace_bytes=int(forward_workspace_bytes),
        rollout_buffer_bytes=int(rollout_buffer_bytes),
        decode_workspace_bytes=int(decode_workspace_bytes),
        safety_bytes=int(safety_bytes),
        total_bytes=int(subtotal + safety_bytes),
        kv_slot_bytes=int(kv_slot_bytes),
        cpu_kv_steady_bytes=int(cpu_kv_steady_bytes),
        cpu_kv_peak_bytes=int(cpu_kv_peak_bytes),
        kv_cache_storage=str(kv_cache_storage),
        detail=detail,
    )


def rollout_memory_budget(
    *,
    model: Any,
    layout: layout_mod.T2VALayout,
    text_len: int,
    config: consistency.SamplerConfig,
    video: ResolvedVAE,
    audio: ResolvedVAE,
    pipeline_config: pipeline_mod.PipelineConfig,
    latent_dtype: Any = None,
    kv_cache_storage: Optional[str] = None,
) -> RolloutMemoryBudget:
    """The node's budget: measure the model, price the decode, add it up.

    ``kv_cache_storage`` defaults to whatever the ``config`` carries, so the
    budget can never describe a different cache from the one the rollout is
    about to build. A config too old to have the field is priced as ``gpu``,
    which is what such a build does.
    """
    if kv_cache_storage is None:
        kv_cache_storage = str(getattr(config, "kv_cache_storage", "gpu"))
    return estimate_rollout_budget(
        layout=layout,
        text_len=int(text_len),
        sink=int(config.sink),
        window=config.window,
        dims=dit_dimensions(model),
        latent_dtype_size=dtype_size(latent_dtype, default=4),
        decode_workspace_bytes=decode_workspace_bytes(
            video, audio, pipeline_config, layout.latent_h, layout.latent_w
        ),
        kv_cache_storage=str(kv_cache_storage),
    )


def _model_management():
    """``comfy.model_management``, imported at the point of use."""
    import comfy.model_management as model_management  # type: ignore[import-not-found]

    return model_management


def _default_load_models_gpu():
    loader = getattr(_model_management(), "load_models_gpu", None)
    if not callable(loader):
        raise NodeInputError(
            "comfy.model_management.load_models_gpu is missing; this ComfyUI cannot "
            "load the model onto the compute device"
        )
    return loader


# --------------------------------------------------------------------------
# residency: two phases, both driven by upstream's own loader
# --------------------------------------------------------------------------
#
# The whole strategy, stated once, because every function below is only a
# detail of it:
#
# The DiT and the two VAEs are **never co-resident**. A chunk is sampled with
# the DiT loaded and the VAEs wherever Comfy last left them; that chunk is then
# decoded with the VAEs loaded, which is a second ``load_models_gpu`` call and
# therefore Comfy's own opportunity to make room by offloading the DiT; the
# next chunk loads the DiT again, which does the same to the VAEs. Nothing in
# this module calls ``.to()``, ``partially_load()`` or ``partially_unload()``,
# and nothing here decides how much of a model may stay resident.
#
# That last sentence is the point. Residency is a function of how much memory
# is actually free at the moment of the load, which upstream measures and this
# node cannot: a plugin-side cap would be a *second*, worse estimate of the
# same quantity, and every time the two disagreed the plugin's would be the one
# that produced a wrong answer -- an OOM on a card that had room, or a run
# refused on a card that would have fitted. What this node owes upstream is an
# honest ``memory_required``: how much VRAM the code *it* is about to run will
# allocate on top of the weights. That is what the two phase plans below are.
#
# The 24 GiB number that appears here is a **diagnostic**, not a cap
# (:data:`GPU_HARD_CAP_BYTES`): it is logged so a run can be compared against
# the smallest card this pack targets. The 24 GiB path itself is exercised by
# constraining the device from outside the process (a VRAM reserve process on
# the H200 box), which is the only way to test it without this node pretending
# to be an allocator.


def _patcher_of(target: Any) -> Any:
    """The ``ModelPatcher`` behind a MODEL socket or a :class:`ResolvedVAE`."""
    if target is None:
        return None
    if isinstance(target, ResolvedVAE):
        return target.patcher
    return target


def _int_or(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return number


def _loaded_size(target: Any) -> int:
    """``ModelPatcher.loaded_size()`` -- weights currently on the compute device."""
    patcher = _patcher_of(target)
    getter = getattr(patcher, "loaded_size", None)
    if not callable(getter):
        return 0
    try:
        return max(0, _int_or(getter()))
    except Exception:  # noqa: BLE001 - a measurement failing is not a run failing
        return 0


def _model_size(target: Any) -> int:
    """``ModelPatcher.model_size()`` -- the weights in full, wherever they are."""
    patcher = _patcher_of(target)
    getter = getattr(patcher, "model_size", None)
    if not callable(getter):
        return 0
    try:
        return max(0, _int_or(getter()))
    except Exception:  # noqa: BLE001
        return 0


def _offload_buffer_bytes(target: Any) -> int:
    """``model.model_offload_buffer_memory``: the staging room Comfy reserved.

    ``ModelPatcher.load`` sets this to the largest ``module + the next
    NUM_STREAMS modules`` it had to stream, and keeps it up to date as it
    unloads. It is VRAM this execution holds and nothing else can use, so a
    residency report that left it out would understate the peak -- but it is
    *Comfy's* reserve, decided inside its own load loop, which is why it is
    reported here and never added to what this node asks for.
    """
    patcher = _patcher_of(target)
    inner = getattr(patcher, "model", None)
    return max(0, _int_or(getattr(inner, "model_offload_buffer_memory", 0)))


def largest_module_bytes(dims: DiTDimensions) -> int:
    """The widest single module Comfy ever streams for this DiT.

    ``MLP.fc1`` (``hidden -> 2 * ffn``) is the largest tensor in a block --
    308 MB in BF16 at the published size, against 154 MB for ``qkv_proj``.
    """
    dtype = int(dims.compute_dtype_size)
    fc1 = int(dims.hidden_size) * 2 * int(dims.ffn_hidden_size) * dtype
    qkv = int(dims.hidden_size) * 3 * dims.inner_dim * dtype
    return int(max(fc1, qkv))


def offload_envelope_bytes(dims: DiTDimensions, num_streams: int) -> int:
    """Comfy's async weight-stream reserve, at the *measured* stream count.

    ``ModelPatcher.load`` computes, for every module it leaves offloaded::

        potential_offload = module_offload_mem + sum(next NUM_STREAMS module sizes)

    and keeps the largest as ``model_offload_buffer_memory``. Modelled here as
    ``(NUM_STREAMS + 1) * widest module``: the module being cast plus the ones
    already in flight.

    Reported, not reserved. Upstream subtracts this from its own residency
    decision while it loads, so adding it to ``memory_required`` would charge
    the same bytes twice and offload weights that had room. The stream count is
    likewise **read** and never set: changing
    ``comfy.model_management.NUM_STREAMS`` from a node would silently re-tune
    every other model in the process.
    """
    return int(max(0, int(num_streams)) + 1) * largest_module_bytes(dims)


@dataclass(frozen=True)
class DeviceMemoryFacts:
    """What the compute device and Comfy say, at the moment the plan is made.

    Diagnostics. Nothing in the plan is computed from ``total_bytes`` or
    ``free_bytes`` -- if it were, this node's ``memory_required`` would change
    with the card, and a request that ran on one box would ask for something
    different on another.

    ``baseline_used_bytes`` (``U_base``) is everything already on the card that
    is **not** this execution's models: other processes, another workflow's
    resident weights, the CUDA context. It is reported on its own line and
    never folded into this node's own numbers -- charging another process's
    allocation to this plugin's budget would make the plugin look like it grew
    every time something else did.
    """

    device: str = ""
    total_bytes: int = 0
    free_bytes: int = 0
    extra_reserved_bytes: int = 0
    num_streams: int = 0
    loaded_bytes: Dict[str, int] = field(default_factory=dict)
    measured: bool = False

    @property
    def ours_loaded_bytes(self) -> int:
        return int(sum(int(v) for v in self.loaded_bytes.values()))

    @property
    def baseline_used_bytes(self) -> int:
        return max(0, int(self.total_bytes) - int(self.free_bytes) - self.ours_loaded_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "total_bytes": int(self.total_bytes),
            "free_bytes": int(self.free_bytes),
            "extra_reserved_bytes": int(self.extra_reserved_bytes),
            "num_streams": int(self.num_streams),
            "loaded_bytes": {str(k): int(v) for k, v in self.loaded_bytes.items()},
            "baseline_used_bytes": int(self.baseline_used_bytes),
            "measured": bool(self.measured),
        }

    def describe(self) -> str:
        gib = float(GIB)
        return (
            "raven device facts[{device}]: total {total:.2f} GiB, free {free:.2f} GiB, "
            "already ours {ours:.2f} GiB, U_base (not ours) {base:.2f} GiB, "
            "comfy extra_reserved {extra:.2f} GiB, NUM_STREAMS {streams}{measured}".format(
                device=self.device or "?",
                total=self.total_bytes / gib,
                free=self.free_bytes / gib,
                ours=self.ours_loaded_bytes / gib,
                base=self.baseline_used_bytes / gib,
                extra=self.extra_reserved_bytes / gib,
                streams=self.num_streams,
                measured="" if self.measured else " (unmeasured: no ComfyUI device)",
            )
        )

    @classmethod
    def probe(
        cls,
        targets: Optional[Mapping[str, Any]] = None,
        *,
        model_management: Any = None,
    ) -> "DeviceMemoryFacts":
        """Read the device, Comfy's reserve and the stream count. Never fatal.

        Outside ComfyUI (or on a build where any of these moved) the facts come
        back ``measured=False`` with zeroed totals. Since nothing is *decided*
        from them, that costs a log line and nothing else.
        """
        loaded = {
            str(name): _loaded_size(target)
            for name, target in (targets or {}).items()
            if target is not None
        }
        try:
            mm = model_management if model_management is not None else _model_management()
        except Exception:  # noqa: BLE001 - a bare interpreter is a supported mode
            return cls(loaded_bytes=loaded)
        try:
            device = mm.get_torch_device()
            total = _int_or(mm.get_total_memory(device))
            free = _int_or(mm.get_free_memory(device))
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "raven: could not read the device's memory (%s: %s); the residency "
                "report will be short a few numbers, the run is unaffected",
                type(exc).__name__,
                exc,
            )
            return cls(loaded_bytes=loaded)
        extra = 0
        getter = getattr(mm, "extra_reserved_memory", None)
        if callable(getter):
            try:
                extra = _int_or(getter())
            except Exception:  # noqa: BLE001
                extra = 0
        return cls(
            device=str(device),
            total_bytes=total,
            free_bytes=free,
            extra_reserved_bytes=extra,
            num_streams=_int_or(getattr(mm, "NUM_STREAMS", 0)),
            loaded_bytes=loaded,
            measured=total > 0,
        )


@dataclass(frozen=True)
class PhasePlan:
    """One residency phase: what it runs, and what it asks Comfy to keep free.

    A *phase* is a state the execution is actually in for a stretch of time:
    ``dit`` while a chunk is sampled, ``vae`` while that chunk is decoded. They
    are planned separately -- and the run's planned peak is the larger of the
    two, never the sum -- because they are never simultaneous. Summing them
    would ask Comfy to keep free memory for a state that does not exist, and
    upstream would answer by offloading weights that had room.

    ``workspace_bytes`` is what goes to ``load_models_gpu(memory_required=...)``:
    everything the phase allocates *besides* weights.
    """

    name: str
    workspace_bytes: int
    model_bytes: int
    items: Dict[str, int] = field(default_factory=dict)
    planning_bytes: int = PLANNING_BUDGET_BYTES

    @property
    def memory_required(self) -> int:
        return int(self.workspace_bytes)

    @property
    def planned_peak_bytes(self) -> int:
        """Workspace plus the weights this phase wants resident.

        A diagnostic upper bound: whether the weights *are* all resident is
        upstream's decision, taken against real free memory. On a card that
        cannot hold them the DiT is partially offloaded and the real peak is
        lower than this, not higher.
        """
        return int(self.workspace_bytes) + int(self.model_bytes)

    @property
    def within_planning(self) -> bool:
        return self.planned_peak_bytes <= int(self.planning_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "workspace_bytes": int(self.workspace_bytes),
            "model_bytes": int(self.model_bytes),
            "memory_required": int(self.memory_required),
            "planned_peak_bytes": int(self.planned_peak_bytes),
            "planning_bytes": int(self.planning_bytes),
            "within_planning": bool(self.within_planning),
            "items": {str(k): int(v) for k, v in self.items.items()},
        }

    def itemised(self) -> str:
        gib = float(GIB)
        return ", ".join(
            "{}={:.3f} GiB".format(name, int(value) / gib)
            for name, value in sorted(self.items.items(), key=lambda kv: -int(kv[1]))
            if int(value)
        )

    def describe(self) -> str:
        gib = float(GIB)
        return (
            "{name} phase: memory_required {req:.2f} GiB + weights {model:.2f} GiB "
            "= planned peak {peak:.2f} GiB [{items}]".format(
                name=self.name,
                req=self.memory_required / gib,
                model=self.model_bytes / gib,
                peak=self.planned_peak_bytes / gib,
                items=self.itemised(),
            )
        )


@dataclass(frozen=True)
class JointOffloadPlan:
    """Both phases of one execution, and the facts they were priced against."""

    dit: PhasePlan
    vae: PhasePlan
    facts: DeviceMemoryFacts
    kv_cache_storage: str = DEFAULT_KV_CACHE_STORAGE
    hard_cap_bytes: int = GPU_HARD_CAP_BYTES
    offload_envelope_bytes: int = 0

    @property
    def planned_peak_bytes(self) -> int:
        return max(self.dit.planned_peak_bytes, self.vae.planned_peak_bytes)

    @property
    def within_planning(self) -> bool:
        return self.dit.within_planning and self.vae.within_planning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": {"dit": self.dit.to_dict(), "vae": self.vae.to_dict()},
            "planned_peak_bytes": int(self.planned_peak_bytes),
            "within_planning": bool(self.within_planning),
            "planning_bytes": int(PLANNING_BUDGET_BYTES),
            "hard_cap_bytes": int(self.hard_cap_bytes),
            "kv_cache_storage": str(self.kv_cache_storage),
            "offload_envelope_bytes": int(self.offload_envelope_bytes),
            "facts": self.facts.to_dict(),
        }

    def describe(self) -> str:
        gib = float(GIB)
        return (
            "raven joint plan: phase swap on (the DiT and the VAEs are never "
            "co-resident), KV on {kv}, planned peak {peak:.2f} GiB vs the {plan:.2f} "
            "GiB reference budget ({verdict} the {cap:.0f} GiB reference card), "
            "comfy offload envelope ~{env:.2f} GiB. {facts} | {dit} | {vae}".format(
                kv=self.kv_cache_storage,
                peak=self.planned_peak_bytes / gib,
                plan=PLANNING_BUDGET_BYTES / gib,
                verdict="within" if self.within_planning else "OVER",
                cap=self.hard_cap_bytes / gib,
                env=self.offload_envelope_bytes / gib,
                facts=self.facts.describe(),
                dit=self.dit.describe(),
                vae=self.vae.describe(),
            )
        )

    def diagnose(self, log: Optional[logging.Logger] = None) -> None:
        """Log the plan, and warn about the one thing offloading cannot fix.

        A phase's *weights* being larger than the reference budget is normal and
        is not warned about: that is exactly the case upstream handles by
        streaming them, at a cost in bandwidth rather than correctness. A
        phase's **workspace** being larger is different -- those are live
        tensors, no residency decision makes them smaller, and a card that size
        simply cannot run the request. That is worth a warning even though it is
        still not a refusal: the card this is running on may well be bigger than
        the reference, and only the allocator knows.
        """
        log_ = log if log is not None else LOG
        log_.info("%s", self.describe())
        for phase in (self.dit, self.vae):
            if phase.memory_required <= PLANNING_BUDGET_BYTES:
                continue
            log_.warning(
                "raven: the %s phase's workspace alone is %.2f GiB, above the "
                "%.2f GiB reference budget. These are live tensors, so no amount "
                "of weight offloading makes them fit -- a card this size cannot "
                "run this request. Terms: %s",
                phase.name,
                phase.memory_required / float(GIB),
                PLANNING_BUDGET_BYTES / float(GIB),
                phase.itemised(),
            )


def plan_joint_offload(
    *,
    budget: RolloutMemoryBudget,
    dims: DiTDimensions,
    facts: DeviceMemoryFacts,
    dit_model_bytes: int = 0,
    video_model_bytes: int = 0,
    audio_model_bytes: int = 0,
    kv_cache_storage: Optional[str] = None,
) -> JointOffloadPlan:
    """Price the two phases: what each allocates besides weights.

    The DiT phase carries, on top of DiT weights:

    * the KV slot -- one merged retained+current K/V for a host-backed cache,
      or the whole cache when it is on the card;
    * the widest chunk's activations, and the FP32 LoRA temporaries;
    * the rollout's own latent buffers, which live for the whole run;
    * the safety head-room the estimate already carries.

    The VAE phase carries the larger of the two decode workspaces and the same
    rollout buffers -- the latents do not go anywhere while a chunk decodes.

    Neither phase's ``memory_required`` includes the other's terms, and neither
    includes weights: those are the two mistakes that make upstream offload
    something it did not have to.
    """
    storage = (
        str(kv_cache_storage)
        if kv_cache_storage is not None
        else str(budget.kv_cache_storage)
    )
    detail = budget.detail
    on_gpu = kv_on_gpu(storage)
    kv_resident = int(budget.kv_cache_bytes)  # 0 unless the cache is on the card
    kv_slot = int(detail.get("kv_gather_bytes", 0)) if on_gpu else int(budget.kv_slot_bytes)

    dit_items: Dict[str, int] = {
        "kv_resident_bytes": kv_resident,
        "kv_slot_bytes": kv_slot,
        "activation_bytes": int(detail.get("activation_bytes", 0)),
        "lora_fp32_temp_bytes": int(detail.get("lora_fp32_temp_bytes", 0)),
        "rollout_buffer_bytes": int(budget.rollout_buffer_bytes),
        "safety_bytes": int(budget.safety_bytes),
    }
    vae_items: Dict[str, int] = {
        "decode_workspace_bytes": int(budget.decode_workspace_bytes),
        "rollout_buffer_bytes": int(budget.rollout_buffer_bytes),
        "kv_resident_bytes": kv_resident,
        "safety_bytes": int(budget.safety_bytes),
    }

    return JointOffloadPlan(
        dit=PhasePlan(
            name="dit",
            workspace_bytes=int(sum(dit_items.values())),
            model_bytes=int(dit_model_bytes),
            items=dit_items,
        ),
        vae=PhasePlan(
            name="vae",
            workspace_bytes=int(sum(vae_items.values())),
            model_bytes=int(video_model_bytes) + int(audio_model_bytes),
            items=vae_items,
        ),
        facts=facts,
        kv_cache_storage=storage,
        offload_envelope_bytes=offload_envelope_bytes(dims, facts.num_streams),
    )


# --------------------------------------------------------------------------
# what actually landed
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseResidency:
    """Measured residency after a load: weights, staging buffer, likely peak.

    A record, not a gate. Every number is upstream's own accounting of what it
    just did -- ``loaded_size()`` for weight bytes on the compute device,
    ``model_offload_buffer_memory`` for the staging room it reserved -- and the
    only thing this node does with them is write them down and, when the total
    is above the reference budget, say so.
    """

    phase: str
    dit_loaded_bytes: int
    video_loaded_bytes: int
    audio_loaded_bytes: int
    offload_buffer_bytes: int
    workspace_bytes: int
    planning_bytes: int = PLANNING_BUDGET_BYTES

    @property
    def model_bytes(self) -> int:
        return (
            int(self.dit_loaded_bytes)
            + int(self.video_loaded_bytes)
            + int(self.audio_loaded_bytes)
        )

    @property
    def predicted_peak_bytes(self) -> int:
        return self.model_bytes + int(self.offload_buffer_bytes) + int(self.workspace_bytes)

    @property
    def within_planning(self) -> bool:
        return self.predicted_peak_bytes <= int(self.planning_bytes)

    @property
    def over_bytes(self) -> int:
        return max(0, self.predicted_peak_bytes - int(self.planning_bytes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": str(self.phase),
            "dit_loaded_bytes": int(self.dit_loaded_bytes),
            "video_loaded_bytes": int(self.video_loaded_bytes),
            "audio_loaded_bytes": int(self.audio_loaded_bytes),
            "model_offload_buffer_memory": int(self.offload_buffer_bytes),
            "workspace_bytes": int(self.workspace_bytes),
            "planning_bytes": int(self.planning_bytes),
            "predicted_peak_bytes": int(self.predicted_peak_bytes),
            "over_bytes": int(self.over_bytes),
            "within_planning": bool(self.within_planning),
        }

    def describe(self) -> str:
        gib = float(GIB)
        return (
            "raven {phase} residency: DiT {dit:.2f} + video VAE {video:.2f} + audio VAE "
            "{audio:.2f} + comfy offload buffer {buffer:.2f} + workspace {work:.2f} "
            "= {peak:.2f} GiB (reference budget {plan:.2f} GiB, {verdict})".format(
                phase=self.phase,
                dit=self.dit_loaded_bytes / gib,
                video=self.video_loaded_bytes / gib,
                audio=self.audio_loaded_bytes / gib,
                buffer=self.offload_buffer_bytes / gib,
                work=self.workspace_bytes / gib,
                peak=self.predicted_peak_bytes / gib,
                plan=self.planning_bytes / gib,
                verdict=(
                    "within"
                    if self.within_planning
                    else "over by {:.2f} GiB".format(self.over_bytes / gib)
                ),
            )
        )


def measure_residency(
    phase: str,
    *,
    model: Any = None,
    video: Any = None,
    audio: Any = None,
    workspace_bytes: int = 0,
    planning_bytes: int = PLANNING_BUDGET_BYTES,
) -> PhaseResidency:
    """Read ``loaded_size()`` and the offload buffer back off the patchers."""
    return PhaseResidency(
        phase=str(phase),
        dit_loaded_bytes=_loaded_size(model),
        video_loaded_bytes=_loaded_size(video),
        audio_loaded_bytes=_loaded_size(audio),
        offload_buffer_bytes=_offload_buffer_bytes(model),
        workspace_bytes=int(workspace_bytes),
        planning_bytes=int(planning_bytes),
    )


def report_residency(
    residency: PhaseResidency, *, log: Optional[logging.Logger] = None
) -> PhaseResidency:
    """Log what a load actually produced. Returns it, so callers can record it.

    Info, not a warning, in both directions. Upstream sized this against the
    memory that was actually free; a run that ended up above the reference
    budget did so because the card had the room, and shouting about it every
    chunk would train the reader to ignore the line that carries the numbers.
    The verdict is in the message either way, and
    :func:`hard_cap_watch` is what raises its voice.
    """
    log_ = log if log is not None else LOG
    log_.info("%s", residency.describe())
    return residency


# --------------------------------------------------------------------------
# the 24 GiB diagnostic
# --------------------------------------------------------------------------


def gpu_memory_state(model_management: Any = None) -> Dict[str, Any]:
    """``(reserved by this process, used on the device, total)``. Never fatal.

    Two numbers, deliberately not one:

    * ``reserved_bytes`` -- ``torch.cuda.memory_reserved()``, i.e. what *this*
      process holds. This is the number the diagnostic is about, because it is
      the only part of the card this node has anything to do with.
    * ``driver_used_bytes`` -- total minus what the driver reports free, so it
      includes every other process on the card. Kept apart from the first on
      purpose: reporting somebody else's allocation as this node's usage is how
      a memory report starts lying.
    """
    state: Dict[str, Any] = {
        "device": "",
        "reserved_bytes": None,
        "driver_used_bytes": None,
        "total_bytes": None,
    }
    try:
        mm = model_management if model_management is not None else _model_management()
        device = mm.get_torch_device()
        state["device"] = str(device)
    except Exception:  # noqa: BLE001 - no ComfyUI: nothing to measure
        return state
    try:
        total = _int_or(mm.get_total_memory(device))
        free = _int_or(mm.get_free_memory(device))
        state["total_bytes"] = total
        state["driver_used_bytes"] = max(0, total - free)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if str(getattr(device, "type", device)).startswith("cuda"):
            state["reserved_bytes"] = int(torch.cuda.memory_reserved(device))
    except Exception:  # noqa: BLE001
        pass
    return state


def hard_cap_watch(
    where: str,
    *,
    model_management: Any = None,
    state: Optional[Dict[str, Any]] = None,
    cap_bytes: int = GPU_HARD_CAP_BYTES,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Note, once per call site, when this process passes the 24 GiB reference.

    Deliberately **not** an abort. This node does not enforce a VRAM cap: the
    allocator upstream already refuses what does not fit, and a plugin-side
    abort at 24 GiB on a 141 GiB card would stop a run that was going to
    succeed. What the number is good for is telling the difference between "this
    run would fit the smallest supported card" and "this run needs the big one",
    which is a fact worth having in the log next to the residency record.

    The 24 GiB path is *tested* by making the device actually that small from
    outside the process, not by this function.
    """
    log_ = log if log is not None else LOG
    reading = state if state is not None else gpu_memory_state(model_management)
    reading = dict(reading)
    reading["where"] = str(where)
    reading["cap_bytes"] = int(cap_bytes)
    reserved = reading.get("reserved_bytes")
    reading["over_reference"] = bool(
        reserved is not None and int(cap_bytes) > 0 and int(reserved) >= int(cap_bytes)
    )
    if reading["over_reference"]:
        driver = reading.get("driver_used_bytes")
        log_.warning(
            "raven: at %s this process has %.2f GiB reserved on %s, at or above the "
            "%.2f GiB reference card%s. The run continues -- the cap is a yardstick, "
            "not a limit this node enforces.",
            where,
            int(reserved) / float(GIB),
            reading.get("device") or "the compute device",
            int(cap_bytes) / float(GIB),
            (
                ""
                if driver is None
                else " (%.2f GiB in use on the device in total, other processes "
                "included -- that part is not this node's)" % (int(driver) / float(GIB))
            ),
        )
    return reading


def make_load_models(
    video: Optional[ResolvedVAE],
    audio: Optional[ResolvedVAE],
    memory_required: int = 0,
    *,
    load_models_gpu: Any = None,
    on_loaded: Any = None,
    log: Optional[logging.Logger] = None,
    plan: Optional[JointOffloadPlan] = None,
):
    """The sampler's ``load_models`` closure: **the DiT alone**, every DiT phase.

    This used to hand the DiT and both VAEs over in one call, so that the
    streaming decode could run between forwards without a reload. It no longer
    does, and the reason is the arithmetic: the two H3 VAEs are 5.4 GiB of
    weights that are idle for the whole of every forward, and the DiT is the
    model that has to be streamed off the host when the card is small. Keeping
    them co-resident spends the scarcest memory in the run on the model that is
    not running. So the execution alternates -- DiT phase, VAE phase -- and each
    phase is a plain ``load_models_gpu`` call for the models that phase uses
    (:class:`PhaseSwapCoordinator`).

    What makes the alternation safe rather than thrash is that upstream, not
    this node, decides what to evict: ``load_models_gpu`` frees only as much as
    the incoming models need, so on a card with room for everything nothing is
    evicted at all and the swap costs one no-op call per chunk. On a card
    without room, the model that is idle is the one that gives way. Neither
    outcome is chosen here.

    ``memory_required`` is the DiT phase's **workspace** -- KV slot,
    activations, LoRA temporaries, rollout buffers, safety (:class:`PhasePlan`)
    -- and nothing else. It must not include the decode workspace (a different
    phase) or any weights (upstream's own accounting), because either would make
    Comfy offload weights that had room.

    ``force_full_load=False`` always: the full non-pruned BF16 DiT is expected
    to be partially offloaded. Nothing here calls ``.to()``, ``patch_model()``,
    ``partially_load()``, ``partially_unload()`` or ``cleanup_models()``.

    The returned closure carries ``calls`` / ``memory_required`` / ``residency``
    / ``hard_cap`` as **diagnostics**, not as a contract. Anything that wraps
    this closure -- an integration harness counting loads, a profiler,
    ``functools.partial`` -- is a plain function without them, so every reader
    outside this function uses ``getattr(..., None)``. A memory record that
    cannot be written is a missing line in a log; it is not a reason for a run
    that has already produced its LATENT/IMAGE/AUDIO to fail.
    """
    log_ = log if log is not None else LOG
    # Bound once, under a name the inner signature cannot shadow: the sampler
    # calls the closure with its own ``memory_required=0``.
    reserve = int(plan.dit.memory_required) if plan is not None else int(memory_required)

    def load_models(models, memory_required: int = 0, force_full_load: bool = False):
        loader = load_models_gpu if load_models_gpu is not None else _default_load_models_gpu()
        targets = list(models)
        required = max(int(memory_required), reserve)
        log_.debug(
            "raven: dit phase load, %d model(s), memory_required=%d bytes "
            "(workspace only; weights are upstream's decision)",
            len(targets),
            required,
        )
        result = loader(targets, memory_required=required, force_full_load=False)
        load_models.calls += 1
        residency = measure_residency(
            "dit",
            model=targets[0] if targets else None,
            video=video,
            audio=audio,
            workspace_bytes=required,
        )
        load_models.residency = report_residency(residency, log=log_)
        load_models.hard_cap = hard_cap_watch("the dit phase load", log=log_)
        if on_loaded is not None and load_models.calls == 1:
            # Only the first load is a phase change the user can see; the
            # per-chunk reloads happen while the status is already 'sampling'.
            on_loaded()
        return result

    load_models.calls = 0
    load_models.memory_required = reserve
    load_models.residency = None
    load_models.hard_cap = None
    return load_models


# --------------------------------------------------------------------------
# the phase swap
# --------------------------------------------------------------------------


class PhaseSwapCoordinator:
    """Alternates DiT and VAE residency around every ``on_chunk``.

    One chunk, in order:

    1. ``load_models_gpu([the VAEs this chunk needs], memory_required=<decode
       workspace>)``. If the card cannot hold the DiT as well, upstream offloads
       the DiT to make room -- that is the swap, and it is upstream's call.
    2. ``pipeline.on_chunk(chunk)``: the incremental video collector and the
       overlap-save audio collector -- which *are* the IMAGE and AUDIO outputs.
    3. Not the last chunk: ``load_models_gpu([the DiT], memory_required=<DiT
       workspace>)`` for the next forward, which is the same swap in reverse.

    The last chunk deliberately stops after step 2 and leaves the VAEs loaded:
    ``pipeline.finish()`` still has both tail flushes to decode through them,
    and reloading the DiT for a forward that never happens would evict exactly
    the models about to be used.

    Both VAEs are needed on **every** chunk. That was not always true -- the
    audio lane used to exist only for the preview -- but the audio collector is
    now the AUDIO output, decoded block by block as the rollout runs, so a
    chunk that skipped it would leave a hole in the returned waveform.
    ``needs_audio`` survives for programmatic callers that genuinely have no
    audio lane; the node passes a predicate that is always true.
    """

    def __init__(
        self,
        *,
        model: Any,
        video: Optional[ResolvedVAE],
        audio: Optional[ResolvedVAE],
        pipeline: Any,
        plan: JointOffloadPlan,
        load_dit: Any,
        load_models_gpu: Any = None,
        needs_audio: Optional[Callable[[], bool]] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.video = video
        self.audio = audio
        self.pipeline = pipeline
        self.plan = plan
        self._load_dit = load_dit
        self._loader = load_models_gpu
        self._needs_audio = needs_audio
        self._log = log if log is not None else LOG
        self.chunks = 0
        self.vae_loads = 0
        self.dit_loads = 0
        self.audio_vae_loads = 0
        self.last_phase = "dit"

    # -- the callback the rollout sees ----------------------------------

    def on_chunk(self, chunk: Any) -> None:
        is_last = bool(getattr(chunk, "is_last", False))
        self.chunks += 1
        self.enter_vae_phase(include_audio=is_last or self._audio_wanted())
        # Not wrapped: a collector failure is the run's failure (its frames are
        # the IMAGE), and a cancelled rollout raises through here on purpose.
        # Residency is left in the VAE phase in both cases, which is the state
        # the finish/decode path wants anyway.
        self.pipeline.on_chunk(chunk)
        if not is_last:
            self.enter_dit_phase()

    # -- the two phases --------------------------------------------------

    def enter_vae_phase(self, *, include_audio: bool = True) -> None:
        # Audio first, video last, and that order is load-bearing:
        # ``load_models_gpu`` does ``models.reverse()`` and then sizes each
        # model's residency in turn against the memory left at that point, so
        # the **last** entry is the one served first. The video VAE goes last
        # because it is by far the larger of the two and its decode is the
        # bigger allocation; the audio VAE is 0.56 GiB and both are now needed
        # on every chunk.
        targets = []
        for resolved, wanted in ((self.audio, include_audio), (self.video, True)):
            patcher = resolved.patcher if resolved is not None else None
            if wanted and patcher is not None:
                targets.append(patcher)
        if not targets:
            return
        loader = self._loader if self._loader is not None else _default_load_models_gpu()
        required = int(self.plan.vae.memory_required)
        self._log.debug(
            "raven: vae phase load, %d model(s), memory_required=%d bytes",
            len(targets),
            required,
        )
        loader(targets, memory_required=required, force_full_load=False)
        self.vae_loads += 1
        if include_audio and self.audio is not None:
            self.audio_vae_loads += 1
        self.last_phase = "vae"
        report_residency(
            measure_residency(
                "vae",
                model=self.model,
                video=self.video,
                audio=self.audio,
                workspace_bytes=required,
            ),
            log=self._log,
        )

    def enter_dit_phase(self) -> None:
        self._load_dit([self.model], memory_required=0, force_full_load=False)
        self.dit_loads += 1
        self.last_phase = "dit"

    # -- reporting -------------------------------------------------------

    def _audio_wanted(self) -> bool:
        if self.audio is None:
            return False
        if self._needs_audio is None:
            return True
        try:
            return bool(self._needs_audio())
        except Exception:  # noqa: BLE001 - a predicate failing is not a run failing
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_swap_chunks": int(self.chunks),
            "phase_swap_vae_loads": int(self.vae_loads),
            "phase_swap_audio_vae_loads": int(self.audio_vae_loads),
            "phase_swap_dit_loads": int(self.dit_loads),
            "phase_swap_last_phase": str(self.last_phase),
        }

    def describe(self) -> str:
        return (
            "raven phase swap: {chunks} chunk(s), {vae} VAE phase load(s) "
            "({audio} with the audio VAE), {dit} DiT reload(s); ended in the "
            "{last} phase".format(
                chunks=self.chunks,
                vae=self.vae_loads,
                audio=self.audio_vae_loads,
                dit=self.dit_loads,
                last=self.last_phase,
            )
        )


def _resolve_cancel_check():
    """``comfy.model_management.throw_exception_if_processing_interrupted``.

    A *thrower*, not a predicate: it returns ``None`` and raises ComfyUI's
    ``InterruptProcessingException`` by itself, which the sampler's cancel-point
    protocol already understands. ``None`` outside ComfyUI, where there is no
    interrupt to observe.
    """
    try:
        import comfy.model_management as model_management  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    return fn if callable(fn) else None


# --------------------------------------------------------------------------
# handing the card over to the final decode
# --------------------------------------------------------------------------

#: Exactly what :func:`prepare_final_decode` does, in one string, so the log
#: line and the report say the same thing the code does.
FINAL_DECODE_UNLOAD_STRATEGY = (
    "comfy.model_management.unload_model_and_clones(model, "
    "unload_additional_models=True, all_devices=False) then soft_empty_cache()"
)

#: The same call, one model later: the video VAE, once its last frame is out.
#:
#: Measured, at 192 frames on a 24 GiB card: the rollout, the collector and the
#: preview all completed (peak 22.85 / 23.17 GiB), and the run then died in the
#: final ``audio_vae.decode`` -- with the video VAE (4.85 GiB) and the audio VAE
#: (0.56 GiB) both still resident, because the last chunk deliberately keeps
#: them for the tail flush. The OOM was not even the visible failure: Comfy
#: caught it, fell back to its generic tiled decode, and that path is written
#: for 5-D image latents, so a 4-D audio latent came back as an ``IndexError``
#: several frames from the actual problem.
FINAL_AUDIO_UNLOAD_STRATEGY = (
    "comfy.model_management.unload_model_and_clones(video_vae.patcher, "
    "unload_additional_models=True, all_devices=False) then soft_empty_cache()"
)


@dataclass(frozen=True)
class FinalDecodeHandover:
    """What one targeted eviction before a final decode cost and freed.

    Used for both handovers at the end of an execution -- the DiT before the
    audio phase, and the video VAE before the audio decode itself. ``phase`` is
    the key prefix a memory report splices in verbatim, so the two records sit
    side by side in one dict instead of overwriting each other.
    """

    seconds: float
    strategy: str = FINAL_DECODE_UNLOAD_STRATEGY
    free_before: Optional[int] = None
    free_after: Optional[int] = None
    device: str = ""
    phase: str = "final_decode"
    subject: str = "the DiT and its clones"
    keeps: str = "the two VAEs stay resident"

    @property
    def freed_bytes(self) -> Optional[int]:
        if self.free_before is None or self.free_after is None:
            return None
        return int(self.free_after) - int(self.free_before)

    def to_dict(self) -> Dict[str, Any]:
        # Key names are the ones a memory report should splice in verbatim.
        prefix = str(self.phase)
        return {
            "{}_unload_seconds".format(prefix): float(self.seconds),
            "{}_unload_strategy".format(prefix): self.strategy,
            "{}_free_before".format(prefix): self.free_before,
            "{}_free_after".format(prefix): self.free_after,
            "{}_freed_bytes".format(prefix): self.freed_bytes,
            "{}_device".format(prefix): self.device,
        }

    def describe(self) -> str:
        gib = 1024.0 ** 3
        freed = self.freed_bytes
        return (
            "raven {label} handover: unloaded {subject} in "
            "{seconds:.2f}s{freed}{free_after}; {keeps} "
            "[{strategy}]".format(
                label=str(self.phase).replace("_", " "),
                subject=self.subject,
                seconds=self.seconds,
                freed="" if freed is None else ", freeing {:.2f} GiB".format(freed / gib),
                free_after=(
                    ""
                    if self.free_after is None
                    else " ({:.2f} GiB free on {})".format(self.free_after / gib, self.device)
                ),
                keeps=self.keeps,
                strategy=self.strategy,
            )
        )


def _free_memory_now(model_management: Any) -> Tuple[Optional[int], str]:
    """``(free bytes, device)`` for the log. Diagnostics only, never fatal."""
    try:
        device = model_management.get_torch_device()
        return int(model_management.get_free_memory(device)), str(device)
    except Exception:  # noqa: BLE001 - a measurement failing is not a run failing
        return None, ""


def prepare_final_decode(
    model: Any,
    *,
    log: Optional[logging.Logger] = None,
    model_management: Any = None,
) -> FinalDecodeHandover:
    """Evict **this run's** DiT before the official full decode. Loud on failure.

    Why this exists, measured rather than assumed: a 39-frame end-to-end run
    completed the rollout *and* the whole streaming preview, then died in
    ``video_vae.decode`` at 130.22 GiB allocated / 139.12 GiB reserved. Nothing
    was leaking. The DiT was simply still resident -- this node deliberately
    loads it co-resident with both VAEs (:func:`make_load_models`) so the
    streaming decode can run between forwards -- and ``VAE.decode``'s own
    implicit ``load_models_gpu([self.patcher], memory_required=<its estimate>)``
    frees only as much as *that* estimate asks for. The estimate is for one
    chunked decode; the final decode is the whole clip at once, and with the
    DiT holding the card there is nothing left to give it.

    So the handover is explicit, and it is **targeted**:

    * ``unload_model_and_clones(model, unload_additional_models=True)`` frees
      this ``MODEL`` and everything sharing its ``clone_base_uuid`` -- which is
      exactly the set a downstream ``LoraLoaderModelOnly`` produces, since
      ``ModelPatcher.clone()`` copies that uuid -- plus its nested additional
      models. Every other loaded model, the two VAEs above all, is left alone:
      they are the models about to be used.
    * ``all_devices=False`` keeps it to the compute device this run used.
    * ``unload_all_models()`` / ``cleanup_models()`` are **not** used. They are
      process-wide: they would evict the VAEs we are about to call and any
      model another node in the same prompt still needs, turning one node's
      memory problem into everyone's reload.

    Nothing here is permanent. The patcher is untouched as an object, so the
    next execution that wants this MODEL gets it back through Comfy's normal
    load path -- at the cost of a reload, which is the price of the final decode
    fitting at all.

    Failure is **loud**: if the API is missing or the unload raises, the run
    stops here. Continuing would mean running the final decode in exactly the
    memory state that is known to OOM, and an OOM three lines later reads like
    a VAE bug instead of a handover that did not happen.
    """
    return _targeted_handover(
        model,
        phase="final_decode",
        subject="the DiT and its clones",
        keeps="the two VAEs stay resident",
        strategy=FINAL_DECODE_UNLOAD_STRATEGY,
        what="this run's DiT",
        before="the final decode",
        log=log,
        model_management=model_management,
    )


def _targeted_handover(
    patcher: Any,
    *,
    phase: str,
    subject: str,
    keeps: str,
    strategy: str,
    what: str,
    before: str,
    log: Optional[logging.Logger] = None,
    model_management: Any = None,
) -> FinalDecodeHandover:
    """One targeted ``unload_model_and_clones`` + ``soft_empty_cache``, timed.

    Shared by the two end-of-execution handovers so that they cannot drift
    apart: same API, same ``all_devices=False``, same refusal to reach for a
    process-wide sledgehammer, same loud failure. Only the model and the words
    differ.
    """
    log_ = log if log is not None else LOG
    started = time.perf_counter()
    mm = model_management if model_management is not None else _model_management()

    unload = getattr(mm, "unload_model_and_clones", None)
    if not callable(unload):
        raise NodeInputError(
            "comfy.model_management.unload_model_and_clones is missing, so {what} "
            "cannot be evicted before {before}. It is public API at the audited "
            "baseline (ComfyUI 0.33.0, c67885b); refusing to run the decode with it "
            "still resident.".format(what=what, before=before)
        )
    soft_empty_cache = getattr(mm, "soft_empty_cache", None)
    if not callable(soft_empty_cache):
        raise NodeInputError(
            "comfy.model_management.soft_empty_cache is missing; refusing to run "
            "{before} without returning the freed blocks to the allocator".format(
                before=before
            )
        )

    free_before, device = _free_memory_now(mm)
    # Not wrapped: an eviction that failed leaves the exact memory state the
    # decode after it is known to OOM in.
    unload(patcher, unload_additional_models=True, all_devices=False)
    soft_empty_cache()
    free_after, device_after = _free_memory_now(mm)

    handover = FinalDecodeHandover(
        seconds=time.perf_counter() - started,
        strategy=strategy,
        free_before=free_before,
        free_after=free_after,
        device=device or device_after,
        phase=phase,
        subject=subject,
        keeps=keeps,
    )
    log_.info("%s", handover.describe())
    return handover


def prepare_final_audio_decode(
    video: Optional[ResolvedVAE],
    *,
    log: Optional[logging.Logger] = None,
    model_management: Any = None,
) -> Optional[FinalDecodeHandover]:
    """Evict the **video** VAE so the audio decode has the card to itself.

    The last thing an execution does is one whole-clip ``vae_decode_audio``, and
    it is the only decode left that is not chunked. Measured at 192 frames on a
    24 GiB card: everything before it succeeded -- rollout, collector and
    preview, peaking at 22.85 / 23.17 GiB -- and this call went out of memory
    with the video VAE's 4.85 GiB and the audio VAE's 0.56 GiB both still
    resident. They are both resident *by design*: the last chunk keeps the VAEs
    loaded because ``pipeline.finish()`` still has a tail flush to decode
    through the video one.

    By the time this runs, that is over: ``finalize_image()`` has already handed
    the IMAGE buffer (host memory) to the caller, so the video VAE has no work
    left this execution. Unloading it -- and nothing else -- is what turns the
    audio decode from "share 5.4 GiB of VAEs plus a whole-clip workspace" into
    "one 0.56 GiB model with the room to itself".

    Why it is worth a *targeted* eviction rather than trusting the OOM: Comfy
    catches an OOM in ``VAE.decode`` and retries through its generic tiled path,
    which is written for 5-D image latents. A 4-D audio latent goes into it and
    comes back as an ``IndexError`` -- so the failure a user sees is not "out of
    memory" at all, and points at a tiling helper this node never asked for.

    Returns ``None`` when there is no video VAE to evict (a programmatic caller
    that never passed one); otherwise the handover record, and a failure here is
    loud for the same reason the DiT handover's is.
    """
    patcher = video.patcher if video is not None else None
    if patcher is None:
        return None
    return _targeted_handover(
        patcher,
        phase="final_audio",
        subject="the video VAE and its clones",
        keeps="the audio VAE stays resident, alone, for the decode",
        strategy=FINAL_AUDIO_UNLOAD_STRATEGY,
        what="this run's video VAE",
        before="the final audio decode",
        log=log,
        model_management=model_management,
    )


# --------------------------------------------------------------------------
# final outputs
# --------------------------------------------------------------------------


def _unbind(samples: Any, index: int) -> Any:
    if getattr(samples, "is_nested", False):
        return samples.unbind()[index]
    return samples


def decode_images(video_vae: Any, latent: Dict[str, Any]) -> Any:
    """Whole-clip video decode. **Diagnostics and tests only.**

    Operator for operator what ``nodes.VAEDecode.decode`` does, including the
    5-D flatten: ``VAE.decode`` returns ``[1, T, H, W, 3]`` for a video VAE and
    the official node reshapes it to ``[T, H, W, 3]``.

    The sampler node does **not** call this. Its IMAGE comes from the streaming
    collector (``StreamingPipeline.finalize_image``), which decodes the same
    frames chunk by chunk as the rollout produces them. Calling this instead
    means decoding the clip a second time, at a whole-clip peak -- which is the
    allocation a measured 39-frame run died on (130.22 GiB allocated / 139.12
    GiB reserved). It stays here because it is the reference the collector is
    checked against, bit for bit, in ``tests/test_streaming_pipeline.py``.
    """
    video = _unbind(latent["samples"], 0)
    images = video_vae.decode(video)
    if len(images.shape) == 5:  # combine batches, exactly as VAEDecode does
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _default_audio_helper():
    from comfy_extras.nodes_audio import vae_decode_audio  # type: ignore[import-not-found]

    return vae_decode_audio


def decode_audio(audio_vae: Any, latent: Dict[str, Any], *, helper: Any = None) -> Any:
    """Final ``AUDIO``: the pinned ``comfy_extras.nodes_audio.vae_decode_audio``.

    Called rather than reimplemented, and its result is returned untouched. It
    owns two things this node must not diverge from: the per-clip loudness
    normalisation (``std * 5``, floored at 1) and the ``{waveform, sample_rate}``
    shape every AUDIO consumer reads. Both are properties of the *whole* clip,
    which is also why the preview's chunk-wise PCM can never be reused here.
    """
    decode = helper if helper is not None else _default_audio_helper()
    return decode(audio_vae, latent)


# --------------------------------------------------------------------------
# preview session wiring
# --------------------------------------------------------------------------


def normalise_node_id(value: Any) -> Optional[str]:
    """The hidden ``unique_id`` as the protocol's ``node_id`` string.

    ComfyUI hands hidden inputs through as whatever the prompt carried -- an
    int, a string, or (for a node inside a subgraph) a colon path like
    ``"12:7"``. The path is kept intact: the client matches the full string
    first and falls back to a suffix match, so rewriting it here would only
    lose information.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return None
    text = str(value).strip()
    if not text or text == "-1":
        # litegraph uses -1 before a node is configured; the client never
        # matches it, so a session for it would go nowhere.
        return None
    return text


def executing_identity(unique_id: Any = None) -> Tuple[Optional[str], Optional[str]]:
    """``(node_id, prompt_id)`` for this execution, snapshotted once.

    ``comfy_execution.utils.get_executing_context()`` is a ``ContextVar`` that
    upstream sets around each node call; reading it once at the top of
    ``execute`` gives a stable pair even if the sampler later runs work on
    another thread, where the context would be unset.
    """
    context = None
    try:
        from comfy_execution.utils import get_executing_context  # type: ignore[import-not-found]

        context = get_executing_context()
    except Exception:  # noqa: BLE001 - no ComfyUI, or no execution in flight
        context = None

    node_id = normalise_node_id(unique_id)
    if node_id is None and context is not None:
        node_id = normalise_node_id(getattr(context, "node_id", None))
    prompt_id = None
    if context is not None:
        raw = getattr(context, "prompt_id", None)
        prompt_id = None if raw is None else str(raw)
    return node_id, prompt_id


@contextlib.contextmanager
def preview_sink(
    node_id: Optional[str],
    *,
    prompt_id: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Iterator[Any]:
    """Scope one preview session and yield its sink, or ``None``.

    Yields ``None`` -- and samples exactly the same way -- when there is no node
    id, no preview package, or no server. When a session *is* created, the
    manager's context manager owns the terminal message: ``complete`` on a clean
    exit, ``cancelled`` for an interrupt, ``error`` otherwise, with the original
    exception re-raised untouched in both failure cases.
    """
    log_ = log if log is not None else LOG
    if node_id is None:
        yield None
        return
    try:
        from raven_streaming import preview as preview_mod

        manager, _registered = preview_mod.install()
        client_id = preview_mod.current_client_id()
    except Exception as exc:  # noqa: BLE001 - previewing is optional, sampling is not
        log_.warning(
            "raven preview: not available (%s: %s); sampling continues",
            type(exc).__name__,
            exc,
        )
        yield None
        return

    with manager.session(node_id, client_id=client_id, prompt_id=prompt_id) as session:
        yield preview_mod.PreviewMediaSink(session)


# --------------------------------------------------------------------------
# Node 2: RAVEN Streaming Sampler
# --------------------------------------------------------------------------


def _sampler_config(**kwargs: Any) -> Any:
    """``consistency.SamplerConfig``, tolerating a build without ``kv_cache_storage``.

    The cache's storage mode is the cache lane's field; this node only chooses a
    string for it. Against a build that predates the field, the widget still has
    to mean something -- so the config is built without it and the run is told,
    once, that its cache will be on the card whatever the widget said. Silently
    dropping it would leave a workflow claiming a 0.5 GiB KV footprint while
    allocating 28 GiB of one.
    """
    config_cls = consistency.SamplerConfig
    try:
        names = {f.name for f in dataclass_fields(config_cls)}
    except TypeError:  # noqa: BLE001 - not a dataclass: let the call decide
        names = set(kwargs)
    storage = kwargs.get("kv_cache_storage")
    if "kv_cache_storage" in kwargs and "kv_cache_storage" not in names:
        kwargs.pop("kv_cache_storage")
        LOG.warning(
            "raven: this build of raven_streaming.consistency has no "
            "kv_cache_storage, so the KV cache stays on the compute device "
            "whatever kv_cache_storage=%r asked for. At the published request "
            "that is ~28 GiB of VRAM instead of ~0.56 GiB.",
            storage,
        )
    return config_cls(**kwargs)


def _warn_about_stacked_loras(model: Any, log: Optional[logging.Logger] = None) -> bool:
    """Warn when official weight patches are stacked on top of the RAVEN model.

    Not a refusal, and not about RAVEN's own adapter: that one is an *activation*
    residual (``runtime_linear``), computed in FP32 on every forward from tensors
    that are never fused into the weights, so it produces the same numbers
    wherever the weights happen to live.

    A stock ``LoraLoaderModelOnly`` is different. Its patches are merged into the
    weight as it is made resident, in that weight's dtype, which means a module
    that was streamed and a module that stayed resident can round differently.
    With residency now decided per phase against real free memory, the *same*
    workflow on a fuller card can therefore produce a bit-for-bit different
    result. Same distribution, same quality, different last bits -- worth saying
    once, because "my seed stopped reproducing" otherwise looks like a bug in
    this node.
    """
    log_ = log if log is not None else LOG
    patches = getattr(_patcher_of(model), "patches", None)
    if not patches:
        return False
    log_.warning(
        "raven: %d patched weight key(s) are stacked on this MODEL (an official "
        "LoRA loader). Those patches are merged in the weight's dtype as each "
        "module is made resident, so a run whose residency differs -- a fuller "
        "card, a different prompt queue -- can differ in the last bits. RAVEN's "
        "own mandatory adapter is an activation residual and is not affected.",
        len(patches),
    )
    return True


class RAVENStreamingSampler:
    """Chunk-major RAVEN rollout with a streaming preview; LATENT/IMAGE/AUDIO out."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "A MODEL from RAVEN Model Loader (optionally with "
                        "official LoRAs stacked after it). A stock bidirectional H3 "
                        "model is rejected: the loop needs the chunk-causal DiT."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "The positive CONDITIONING from MiniMax H3 Image to "
                        "Video used in T2VA form. There is no negative input and no "
                        "CFG: the chunk-major loop runs one conditioning branch, so a "
                        "second one would be silently ignored. Keyframe (fl2va) and "
                        "reference (ref2va) extras are refused with an explicit error: "
                        "this sampler has not implemented or verified the causal packed "
                        "layout for condition rows, so refusing beats dropping them "
                        "silently. That is an implementation limit here, not a "
                        "statement about the RAVEN LoRA."
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "The empty AV latent from the same node (or Empty "
                        "MiniMax H3 AV Latent). It defines the frame count and canvas; "
                        "this node deliberately has no width/height/frames inputs. A "
                        "non-empty latent is refused - every chunk starts from its own "
                        "fresh noise."
                    },
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "The MiniMax H3 video VAE (24 latent channels)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "The MiniMax H3 audio VAE (32 channels, stereo, 32 kHz)."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seeds a private generator; the rollout never touches "
                        "global RNG, so the same seed is the same clip.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": consistency.DEFAULT_STEPS,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Consistency NFEs per chunk. RAVEN's published "
                        "preview trial is 4; more steps is not a free quality win, "
                        "the schedule was distilled for this budget.",
                    },
                ),
                "video_shift": (
                    "FLOAT",
                    {
                        "default": consistency.DEFAULT_VIDEO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Shift of the video stream's trailing sigma grid.",
                    },
                ),
                "audio_shift": (
                    "FLOAT",
                    {
                        "default": consistency.DEFAULT_AUDIO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Shift of the audio stream's own trailing sigma grid. "
                        "The two streams run independent grids, not one remapped grid.",
                    },
                ),
                "sink": (
                    "INT",
                    {
                        "default": consistency.DEFAULT_SINK,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Attention-sink cache chunks pinned from the start. "
                        "Chunk 0 is the text prefill, so 2 means text + the first media "
                        "chunk.",
                    },
                ),
                "window": (
                    "INT",
                    {
                        "default": (
                            consistency.DEFAULT_WINDOW
                            if consistency.DEFAULT_WINDOW is not None
                            else 2
                        ),
                        "min": 0,
                        "max": 64,
                        "tooltip": "Most recent cache chunks kept besides the sinks. "
                        "0 keeps only the sinks.",
                    },
                ),
                "kv_cache_storage": (
                    list(KV_CACHE_STORAGE_CHOICES),
                    {
                        "default": DEFAULT_KV_CACHE_STORAGE,
                        "tooltip": "Where the retained chunk KV cache lives. "
                        "'cpu_pinned' (default) keeps it in page-locked host memory "
                        "and copies one layer's retained rows back per block: about "
                        "0.56 GiB of VRAM at 192 frames instead of the ~28 GiB the "
                        "whole cache costs on the card. 'cpu' is the same without "
                        "page-locking (slower copies, no pinned-memory pressure). "
                        "'gpu' keeps it resident and is only for cards with room to "
                        "spare. This changes where bytes live, not what is computed.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("LATENT", "IMAGE", "AUDIO")
    OUTPUT_TOOLTIPS = (
        "The finished AV latent, same nested (video, audio) structure as the input.",
        "Every frame, in order, as written by the incremental video collector while "
        "the rollout ran. The clip is not decoded again at the end.",
        "The stereo waveform, as written by the incremental (overlap-save) audio "
        "collector while the rollout ran, with the official whole-clip loudness "
        "normalisation applied once at finalize.",
    )
    FUNCTION = "sample"
    CATEGORY = "model/sampling/raven"
    DESCRIPTION = (
        "Runs RAVEN's chunk-major fresh-noise consistency rollout over MiniMax H3 and "
        "streams the result into the node while it samples.\n\n"
        "This is not a stock sampler: a chunk is carried to completion in `steps` NFEs "
        "and written into the KV cache before the next one starts, so no sampler or "
        "scheduler selector applies and CFG/negative conditioning do not exist here.\n\n"
        "Frame count and canvas come from the LATENT. Video becomes visible one chunk "
        "behind sampling (the decoder needs 2 latents of lookahead) and the audio VAE "
        "adds 0.425 s of its own; the clip ends with a 5-frame flush.\n\n"
        "Memory: the DiT and the two VAEs are never co-resident. Each chunk is sampled "
        "with the DiT loaded and decoded with the VAEs loaded, through ordinary "
        "comfy.model_management.load_models_gpu calls, so ComfyUI decides what is "
        "resident against the memory that is actually free. There is no VRAM cap or "
        "residency input here on purpose - it would only be a second guess at a number "
        "ComfyUI already measures.\n\n"
        "The preview is best-effort: if it cannot start, or fails mid-run, sampling and "
        "the returned LATENT/IMAGE/AUDIO are unaffected."
    )
    SEARCH_ALIASES = ["raven", "streaming", "minimax h3", "t2va"]

    def sample(
        self,
        model: Any,
        positive: Any,
        latent: Any,
        video_vae: Any,
        audio_vae: Any,
        seed: int,
        steps: int = consistency.DEFAULT_STEPS,
        video_shift: float = consistency.DEFAULT_VIDEO_SHIFT,
        audio_shift: float = consistency.DEFAULT_AUDIO_SHIFT,
        sink: int = consistency.DEFAULT_SINK,
        window: int = 2,
        kv_cache_storage: str = DEFAULT_KV_CACHE_STORAGE,
        unique_id: Any = None,
    ):
        video = resolve_video_vae(video_vae)
        audio = resolve_audio_vae(audio_vae)
        storage = str(kv_cache_storage)
        _check(
            storage in KV_CACHE_STORAGE_CHOICES,
            f"kv_cache_storage must be one of {list(KV_CACHE_STORAGE_CHOICES)}, got "
            f"{kv_cache_storage!r}",
        )

        # Parsed here as well as inside the sampler: the request grid decides
        # the preview's canvas and duration, and a bad input should be rejected
        # before a session is opened or a weight is touched.
        conditioning = contracts.parse_conditioning(positive)
        request = contracts.parse_latent(latent)
        # warn_experimental=False: parse_latent has already warned about a long
        # clip, and saying it twice per execution reads like two problems.
        layout = request.layout(conditioning.text_len, warn_experimental=False)

        config = _sampler_config(
            steps=int(steps),
            video_shift=float(video_shift),
            audio_shift=float(audio_shift),
            sink=int(sink),
            window=int(window),
            seed=int(seed),
            kv_cache_storage=storage,
        )
        pipeline_config = pipeline_mod.PipelineConfig(
            frames=request.frames, width=request.width, height=request.height
        )
        node_id, prompt_id = executing_identity(unique_id)

        # Priced before the session opens, because the number is what the very
        # first thing inside it (the model load) is driven by.
        budget = rollout_memory_budget(
            model=model,
            layout=layout,
            text_len=conditioning.text_len,
            config=config,
            video=video,
            audio=audio,
            pipeline_config=pipeline_config,
            latent_dtype=request.dtype,
            kv_cache_storage=storage,
        )
        LOG.info("%s", budget.describe())

        # Both phases priced before anything loads, and logged as one record:
        # the DiT phase's workspace is what the first load asks Comfy for, and
        # the VAE phase's is what every chunk's decode asks for.
        facts = DeviceMemoryFacts.probe(
            {"dit": model, "video": video.patcher, "audio": audio.patcher}
        )
        plan = plan_joint_offload(
            budget=budget,
            dims=dit_dimensions(model),
            facts=facts,
            dit_model_bytes=_model_size(model),
            video_model_bytes=_model_size(video.patcher),
            audio_model_bytes=_model_size(audio.patcher),
            kv_cache_storage=storage,
        )
        plan.diagnose(LOG)
        _warn_about_stacked_loras(model)
        budget.detail["joint_offload_plan"] = plan.to_dict()

        with preview_sink(node_id, prompt_id=prompt_id) as sink_obj:
            # Unguarded: this builds the video collector, which *is* the IMAGE
            # output. Only the preview half of it is best-effort, and that
            # try/except lives inside build_media_pipeline.
            pipeline = pipeline_mod.build_media_pipeline(
                video_vae=video.vae,
                audio_vae=audio.vae,
                config=pipeline_config,
                sink=sink_obj,
                log=LOG,
                memory_budget=budget.to_dict(),
            )
            try:
                pipeline.open_preview()
                pipeline.status(
                    "model_loading",
                    message=(
                        None
                        if not pipeline.preview_disabled
                        else pipeline.preview_disabled_reason
                    ),
                )
                load_models = make_load_models(
                    video,
                    audio,
                    plan=plan,
                    on_loaded=lambda: pipeline.status("sampling"),
                )
                # The rollout's on_chunk is the coordinator's, not the
                # pipeline's: the decode inside it needs the VAEs, and the
                # forward after it needs the DiT.
                phases = PhaseSwapCoordinator(
                    model=model,
                    video=video,
                    audio=audio,
                    pipeline=pipeline,
                    plan=plan,
                    load_dit=load_models,
                    # Always: the audio collector is the AUDIO output now, not
                    # a preview extra, so every chunk decodes through it.
                    needs_audio=lambda: True,
                    log=LOG,
                )
                result = consistency.sample_streaming(
                    model=model,
                    positive=positive,
                    latent=latent,
                    config=config,
                    on_chunk=phases.on_chunk,
                    cancel_check=_resolve_cancel_check(),
                    load_models=load_models,
                    # already validated (and warned about) above
                    warn_experimental=False,
                )
            except BaseException:
                # Cancellation and failure take the same cleanup path; the
                # session's own context manager classifies and re-raises.
                pipeline.cancel()
                raise

            # Announced *before* the tail flush, the DiT handover and the audio
            # decode, not after: those take real time, and a client left on
            # 'sampling' through them reads as a hang.
            pipeline.status("finalizing")
            # The last chunk left the VAEs loaded on purpose (see
            # PhaseSwapCoordinator): the tail flush decodes through them.
            report = pipeline.finish()
            LOG.info("%s", report.describe())
            LOG.info("%s", phases.describe())
            # The per-chunk record: what each chunk decoded and what went out
            # while it was decoding. This is what a real run is accepted on --
            # "it streamed as it sampled" is a claim with a table behind it.
            LOG.info("%s", report.describe_emissions())

            # Both outputs are already decoded -- frame by frame and block by
            # block, as the rollout produced them. There is deliberately no
            # whole-clip decode of either stream here: those are the two calls
            # that OOMed on real hardware (video at 39 frames on a 141 GiB
            # card; audio at 192 frames on a 24 GiB card, with the DiT *and*
            # the video VAE already evicted), and both would only re-derive
            # data these lanes already hold.
            images = pipeline.finalize_image()
            waveform = pipeline.finalize_audio(
                audio.vae, sample_rate=result.latent.get("sample_rate")
            )

            # Nothing else in this execution needs the card. The DiT still goes,
            # because the next node in the graph is about to want the room and
            # Comfy would otherwise keep 60+ GiB of it parked here.
            handover = prepare_final_decode(model)
            # One complete memory record per execution: the reserve that was
            # asked for, and what the handover before the final decode cost.
            # (It cannot ride on PipelineReport.memory_budget -- the pipeline
            # copied that dict before the rollout even started.)
            budget.detail.update(handover.to_dict())
            budget.detail.update(phases.to_dict())
            # ``getattr``, not attribute access: the loader here is whatever the
            # caller handed us. ``make_load_models`` hangs its diagnostics off
            # the closure, but a wrapper -- an integration harness counting
            # calls, a profiler, ``functools.partial`` -- is a plain function
            # with none of them, and a *record* that cannot be written must not
            # be able to fail a run that has already produced its outputs.
            # These attributes are diagnostics, so they are read as optional;
            # everything the run actually depends on goes through the call.
            residency = getattr(load_models, "residency", None)
            if residency is not None:
                budget.detail["dit_phase_residency"] = residency.to_dict()
            hard_cap = getattr(load_models, "hard_cap", None)
            if hard_cap is not None:
                budget.detail["dit_phase_hard_cap"] = hard_cap

            # No handover for the audio decode any more: there is no audio
            # decode. ``prepare_final_audio_decode`` existed to give a
            # whole-clip ``vae_decode_audio`` the card to itself, and that call
            # is gone -- evicting the video VAE here would buy nothing and cost
            # the next execution a reload. The helper stays for diagnostics.
            budget.detail["emission_log"] = [
                emission.to_dict() for emission in report.chunk_emissions
            ]
            budget.detail["fragments_before_finish"] = report.fragments_before_finish
            budget.detail["hard_cap_reference"] = hard_cap_watch(
                "after both collectors finalised"
            )
            LOG.debug("raven memory record: %s", budget.to_dict())

        return (result.latent, images, waveform)


# --------------------------------------------------------------------------
# registration (V1)
# --------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "RAVENModelLoader": RAVENModelLoader,
    "RAVENStreamingSampler": RAVENStreamingSampler,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "RAVENModelLoader": "RAVEN Model Loader",
    "RAVENStreamingSampler": "RAVEN Streaming Sampler",
}
