"""Node-boundary contracts for the RAVEN streaming sampler (M3).

What this module owns
---------------------
Everything that crosses the ComfyUI socket boundary, parsed **strictly** and
failing **loudly**:

* the official ``CONDITIONING`` produced by ``MiniMaxH3ImageToVideo`` in its
  T2VA form -- exactly one positive entry, no negative, no CFG, no scheduling,
  no hooks, no keyframes / references / guides / controls;
* the official ``LATENT`` produced by ``EmptyMiniMaxH3LatentAV`` /
  ``MiniMaxH3ImageToVideo`` -- a two-stream ``NestedTensor`` pair
  ``(video [1, 24, T, H/16, W/16], audio [1, 32, 2, T40])`` at batch size 1,
  **empty** (all-zero), on the H3 grid;
* the ``MODEL`` -- a stock static ``comfy.model_patcher.ModelPatcher`` whose
  ``diffusion_model`` is the RAVEN causal DiT, with the one
  ``transformer_options`` dict the whole rollout reuses.

Why strict
----------
The streaming sampler runs its own chunk-major loop: it never calls
``comfy.samplers``, so every feature that lives in the stock sampling path
(CFG, negative conditioning, conditioning schedules, area/mask conditioning,
ControlNet, block replacement, model wrappers) is simply *not executed*.
Accepting such an input and ignoring it would look like a quality bug, so each
one is rejected by name with the reason.

``minimax_token_tags`` is the one H3-specific extra that IS honoured: it is the
per-token modality tag vector the AdaLN path needs, and it is passed straight
through to :meth:`RavenCausalMiniMaxH3Model.prefill_text`.

Deliberately not done here
--------------------------
``BaseModel.process_latent_in`` / ``process_latent_out`` are **not** applied.
For ``comfy.latent_formats.MiniMaxH3AV`` the format scaling is the identity
(``scale_factor == 1.0``); the only thing those two methods add is the stock
sampler's **audio scale compensation**, which exists because the stock
sigma-major path carries the audio stream on the *video* schedule. The RAVEN
loop runs the audio stream on its own shifted grid, so applying that
compensation would rescale the audio latent against a schedule mapping it never
took. Latents therefore cross this boundary in native model space, which for
this family is also Comfy's latent space.

Import weight: torch plus :mod:`raven_streaming.layout` (torch-only). ComfyUI is
**not** imported at module scope, and every upstream type this module checks is
either duck-typed or imported lazily, so the parsers are pure functions that run
against fakes in a bare Python environment.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from raven_streaming import layout as layout_mod

__all__ = [
    "ContractError",
    "VIDEO_LATENT_CHANNELS",
    "AUDIO_LATENT_CHANNELS",
    "ALLOWED_CONDITIONING_KEYS",
    "UNSUPPORTED_CONDITIONING_KEYS",
    "ALLOWED_LATENT_KEYS",
    "UNSUPPORTED_LATENT_KEYS",
    "UNSUPPORTED_MODEL_OPTIONS",
    "UNSUPPORTED_TRANSFORMER_OPTIONS",
    "TextConditioning",
    "LatentRequest",
    "ResolvedModel",
    "parse_conditioning",
    "parse_latent",
    "build_output_latent",
    "resolve_model",
    "resolve_transformer_options",
]


class ContractError(ValueError):
    """An input that this node's contract does not accept."""


#: Official H3 stream widths (``comfy_extras/nodes_minimax_h3.py``).
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32


# --------------------------------------------------------------------------
# CONDITIONING
# --------------------------------------------------------------------------
#: Extras a plain T2VA encode legitimately carries.
ALLOWED_CONDITIONING_KEYS: frozenset = frozenset({"pooled_output", "minimax_token_tags"})

#: Extras we know about and refuse, with the reason shown to the user.
UNSUPPORTED_CONDITIONING_KEYS: Dict[str, str] = {
    "minimax_keyframes": "keyframe (fl2va) conditioning; the streaming sampler is T2VA only",
    "minimax_refs": "reference (ref2va) conditioning; the streaming sampler is T2VA only",
    "minimax_visual_cond_noise_aug": "condition-row noise augmentation; there are no condition rows in T2VA",
    "minimax_audio_cond_noise_aug": "condition-row noise augmentation; there are no condition rows in T2VA",
    "control": "ControlNet; the chunk-major loop never runs comfy.samplers' control path",
    "control_apply_to_uncond": "ControlNet; the chunk-major loop never runs comfy.samplers' control path",
    "gligen": "GLIGEN conditioning is not part of the H3 lane",
    "area": "area conditioning is a comfy.samplers feature; the streaming loop does not run it",
    "strength": "per-conditioning strength is a comfy.samplers feature; the streaming loop does not run it",
    "mask": "masked conditioning is a comfy.samplers feature; the streaming loop does not run it",
    "mask_strength": "masked conditioning is a comfy.samplers feature; the streaming loop does not run it",
    "set_area_to_bounds": "area conditioning is a comfy.samplers feature; the streaming loop does not run it",
    "start_percent": "conditioning schedules (ConditioningSetTimestepRange) are not supported",
    "end_percent": "conditioning schedules (ConditioningSetTimestepRange) are not supported",
    "clip_start_percent": "scheduled CLIP hooks are not supported",
    "clip_end_percent": "scheduled CLIP hooks are not supported",
    "hooks": "CLIP/model hook groups are not supported",
    "timestep_start": "conditioning schedules are not supported",
    "timestep_end": "conditioning schedules are not supported",
}


@dataclass(frozen=True)
class TextConditioning:
    """The one positive T2VA conditioning entry, unpacked.

    ``cross_attn`` is ``[1, L, dim]``: either raw Qwen3-VL hidden states
    (``dim == text_dim``) or already-refined states (``dim == hidden_size``).
    Which one it is is decided by the model, not here --
    :meth:`RavenCausalMiniMaxH3Model.prefill_text` runs ``condition_proj`` +
    the token refiner only when the width says it has not run yet.
    """

    cross_attn: torch.Tensor
    token_tags: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None

    @property
    def text_len(self) -> int:
        return int(self.cross_attn.shape[1])


def parse_conditioning(conditioning: Any) -> TextConditioning:
    """Strictly unpack an official positive ``CONDITIONING`` for T2VA.

    Accepts exactly one entry, ``[cross_attn, extras]``. Everything the stock
    sampling path would have consumed is rejected by name.
    """
    if conditioning is None:
        raise ContractError("positive conditioning is required")
    if isinstance(conditioning, (str, bytes)) or not isinstance(conditioning, Sequence):
        raise ContractError(
            "conditioning must be a list of [cond, extras] entries, got "
            f"{type(conditioning).__name__}"
        )
    if len(conditioning) == 0:
        raise ContractError("conditioning is empty")
    if len(conditioning) != 1:
        raise ContractError(
            f"conditioning carries {len(conditioning)} entries; the streaming sampler "
            "takes exactly one positive entry (no CFG, no negative, no conditioning "
            "combine/schedule)"
        )

    entry = conditioning[0]
    if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
        raise ContractError(
            "conditioning entry must be [cond, extras], got "
            f"{type(entry).__name__} of length "
            f"{len(entry) if isinstance(entry, Sequence) else 'n/a'}"
        )
    cross_attn, extras = entry[0], entry[1]

    if not isinstance(cross_attn, torch.Tensor):
        raise ContractError(
            f"conditioning[0][0] must be a tensor, got {type(cross_attn).__name__}"
        )
    if cross_attn.ndim != 3:
        raise ContractError(
            f"conditioning tensor must be [B, L, dim], got {tuple(cross_attn.shape)}"
        )
    if cross_attn.shape[0] != 1:
        raise ContractError(
            f"conditioning batch is {cross_attn.shape[0]}; the streaming sampler runs "
            "batch size 1 (encode one prompt per run)"
        )
    if cross_attn.shape[1] < 1:
        raise ContractError("conditioning has zero text tokens")

    if extras is None:
        extras = {}
    if not isinstance(extras, Mapping):
        raise ContractError(
            f"conditioning[0][1] must be a dict, got {type(extras).__name__}"
        )

    rejected = []
    for key in extras:
        if key in ALLOWED_CONDITIONING_KEYS:
            continue
        reason = UNSUPPORTED_CONDITIONING_KEYS.get(
            key, "unknown conditioning extra; the streaming sampler refuses inputs it "
                 "cannot honour rather than silently dropping them"
        )
        rejected.append(f"{key!r}: {reason}")
    if rejected:
        raise ContractError(
            "unsupported conditioning extras:\n  " + "\n  ".join(sorted(rejected))
        )

    tags = extras.get("minimax_token_tags", None)
    if tags is not None:
        if not isinstance(tags, torch.Tensor):
            raise ContractError(
                f"minimax_token_tags must be a tensor, got {type(tags).__name__}"
            )
        tags = tags.reshape(-1)
        if int(tags.numel()) != int(cross_attn.shape[1]):
            raise ContractError(
                f"minimax_token_tags covers {int(tags.numel())} tokens but the "
                f"conditioning has {int(cross_attn.shape[1])}"
            )
        if tags.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ContractError(
                f"minimax_token_tags must be an integer tensor, got {tags.dtype}"
            )
        valid = (layout_mod.VIDEO_TAG, layout_mod.TEXT_TAG, layout_mod.AUDIO_TAG)
        bad = sorted({int(v) for v in tags.tolist()} - set(valid))
        if bad:
            raise ContractError(
                f"minimax_token_tags carries tag(s) {bad}; the official AdaLN "
                f"modality tags are {list(valid)}"
            )

    pooled = extras.get("pooled_output", None)
    if pooled is not None and not isinstance(pooled, torch.Tensor):
        raise ContractError(
            f"pooled_output must be a tensor, got {type(pooled).__name__}"
        )
    return TextConditioning(cross_attn=cross_attn, token_tags=tags, pooled_output=pooled)


# --------------------------------------------------------------------------
# LATENT
# --------------------------------------------------------------------------
#: Keys an official empty AV latent legitimately carries.
ALLOWED_LATENT_KEYS: frozenset = frozenset({"samples"})

UNSUPPORTED_LATENT_KEYS: Dict[str, str] = {
    "noise_mask": "latent masking / inpainting is not supported by the streaming loop",
    "batch_index": "batch indexing implies batch > 1; the streaming sampler runs batch size 1",
}


@dataclass(frozen=True)
class LatentRequest:
    """One validated empty T2VA request, plus the request grid it implies."""

    video: torch.Tensor
    audio: torch.Tensor
    frames: int
    width: int
    height: int
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int
    #: the concrete ``NestedTensor`` class the input used, so the output is the
    #: same type without this module importing ComfyUI
    nested_cls: type = tuple
    extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.video.device

    @property
    def dtype(self) -> torch.dtype:
        return self.video.dtype

    def layout(self, text_len: int, *, warn_experimental: bool = True) -> layout_mod.T2VALayout:
        built = layout_mod.T2VALayout.from_request(
            text_len=int(text_len),
            frames=self.frames,
            width=self.width,
            height=self.height,
            warn_experimental=warn_experimental,
        )
        # The grid is re-derived from ``frames``; if it disagreed with the
        # tensors we measured, the sampler would silently write the wrong rows.
        if (built.latent_t, built.latent_h, built.latent_w, built.audio_t) != (
            self.latent_t, self.latent_h, self.latent_w, self.audio_t
        ):
            raise ContractError(
                "the latent's measured grid "
                f"(t={self.latent_t}, h={self.latent_h}, w={self.latent_w}, "
                f"audio_t={self.audio_t}) does not match the T2VA layout derived "
                f"from frames={self.frames} (t={built.latent_t}, h={built.latent_h}, "
                f"w={built.latent_w}, audio_t={built.audio_t})"
            )
        return built


def _unbind_nested(samples: Any) -> Tuple[torch.Tensor, ...]:
    """The two streams of a Comfy ``NestedTensor``, duck-typed."""
    if not getattr(samples, "is_nested", False):
        raise ContractError(
            "LATENT['samples'] must be a MiniMax H3 AV NestedTensor (video, audio) "
            f"pair, got {type(samples).__name__}. Use 'Empty MiniMax H3 AV Latent' "
            "or the latent output of 'MiniMax H3 Image to Video'."
        )
    tensors = getattr(samples, "tensors", None)
    if tensors is None and hasattr(samples, "unbind"):
        tensors = samples.unbind()
    if tensors is None:
        raise ContractError(
            f"{type(samples).__name__} exposes neither .tensors nor .unbind(); it is "
            "not a comfy.nested_tensor.NestedTensor"
        )
    return tuple(tensors)


def parse_latent(latent: Any, *, warn_experimental: bool = True) -> LatentRequest:
    """Strictly unpack an official **empty** H3 AV ``LATENT``.

    A non-empty latent is refused: the streaming loop starts every chunk from
    its own fresh noise, so an incoming latent would be discarded. img2img /
    latent-continuation semantics are not implemented, and pretending otherwise
    would silently ignore the user's input.
    """
    if latent is None:
        raise ContractError("latent is required")
    if not isinstance(latent, Mapping):
        raise ContractError(f"LATENT must be a dict, got {type(latent).__name__}")
    if "samples" not in latent:
        raise ContractError("LATENT has no 'samples' key")

    rejected = []
    for key in latent:
        if key in ALLOWED_LATENT_KEYS:
            continue
        reason = UNSUPPORTED_LATENT_KEYS.get(
            key, "unknown LATENT key; the streaming sampler refuses inputs it cannot honour"
        )
        rejected.append(f"{key!r}: {reason}")
    if rejected:
        raise ContractError("unsupported LATENT keys:\n  " + "\n  ".join(sorted(rejected)))

    samples = latent["samples"]
    streams = _unbind_nested(samples)
    if len(streams) != 2:
        raise ContractError(
            f"LATENT['samples'] holds {len(streams)} stream(s); the H3 AV latent is a "
            "(video, audio) pair"
        )
    video, audio = streams
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise ContractError("both AV latent streams must be tensors")

    if video.ndim != 5:
        raise ContractError(
            f"video latent must be [B, 24, T, H, W], got {tuple(video.shape)}"
        )
    if audio.ndim != 4:
        raise ContractError(
            f"audio latent must be [B, 32, 2, T], got {tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ContractError(
            f"batch size must be 1, got video {video.shape[0]} / audio {audio.shape[0]}"
        )
    if video.shape[1] != VIDEO_LATENT_CHANNELS:
        raise ContractError(
            f"video latent has {video.shape[1]} channels, expected {VIDEO_LATENT_CHANNELS}"
        )
    if audio.shape[1] != AUDIO_LATENT_CHANNELS:
        raise ContractError(
            f"audio latent has {audio.shape[1]} channels, expected {AUDIO_LATENT_CHANNELS}"
        )
    if audio.shape[2] != layout_mod.AUDIO_CHANNELS:
        raise ContractError(
            f"audio latent has {audio.shape[2]} audio channel(s), expected stereo "
            f"({layout_mod.AUDIO_CHANNELS})"
        )
    if video.device != audio.device:
        raise ContractError("the video and audio latent streams are on different devices")
    if video.dtype != audio.dtype:
        raise ContractError(
            f"the two latent streams have different dtypes: {video.dtype} vs {audio.dtype}"
        )

    latent_t = int(video.shape[2])
    latent_h = int(video.shape[3])
    latent_w = int(video.shape[4])
    audio_t = int(audio.shape[3])

    if latent_t < 2 or (latent_t - 2) % layout_mod.VIDEO_LATENTS_PER_CHUNK != 0:
        raise ContractError(
            f"video latent has {latent_t} temporal latents; the H3 grid is 5k + 2"
        )
    k = (latent_t - 2) // layout_mod.VIDEO_LATENTS_PER_CHUNK
    frames = 17 * k + 5
    try:
        layout_mod.validate_frames(frames, warn_experimental=warn_experimental)
    except layout_mod.LayoutError as exc:
        raise ContractError(str(exc)) from exc

    expected_audio_t = layout_mod.audio_latent_t(frames)
    if audio_t != expected_audio_t:
        raise ContractError(
            f"audio latent has {audio_t} latents; a {frames}-frame clip at "
            f"{layout_mod.FPS} fps carries {expected_audio_t} at "
            f"{layout_mod.AUDIO_LATENT_FPS} Hz"
        )

    width = latent_w * 16
    height = latent_h * 16
    try:
        layout_mod.validate_canvas(width, height)
    except layout_mod.LayoutError as exc:
        raise ContractError(str(exc)) from exc

    for name, tensor in (("video", video), ("audio", audio)):
        if not torch.is_floating_point(tensor):
            raise ContractError(f"{name} latent must be a floating point tensor, got {tensor.dtype}")
        if bool(torch.any(tensor != 0)):
            raise ContractError(
                f"the {name} latent is not empty. The RAVEN streaming sampler starts "
                "every chunk from its own fresh noise, so an incoming latent would be "
                "discarded; connect 'Empty MiniMax H3 AV Latent' (or the latent output "
                "of 'MiniMax H3 Image to Video') instead of a sampled/encoded latent."
            )

    return LatentRequest(
        video=video,
        audio=audio,
        frames=int(frames),
        width=int(width),
        height=int(height),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        nested_cls=type(samples),
        extra={k2: v for k2, v in latent.items() if k2 != "samples"},
    )


def build_output_latent(
    request: LatentRequest, video: torch.Tensor, audio: torch.Tensor
) -> Dict[str, Any]:
    """Same structure in, same structure out: ``{'samples': NestedTensor(v, a)}``.

    The ``NestedTensor`` class is taken from the *input* latent rather than
    imported, so the output is bit-for-bit the type the rest of the graph
    already handles and this module keeps its "no ComfyUI import" property.
    """
    expected_video = (1, VIDEO_LATENT_CHANNELS, request.latent_t, request.latent_h, request.latent_w)
    expected_audio = (1, AUDIO_LATENT_CHANNELS, layout_mod.AUDIO_CHANNELS, request.audio_t)
    if tuple(video.shape) != expected_video:
        raise ContractError(
            f"output video latent {tuple(video.shape)} != request {expected_video}"
        )
    if tuple(audio.shape) != expected_audio:
        raise ContractError(
            f"output audio latent {tuple(audio.shape)} != request {expected_audio}"
        )
    out: Dict[str, Any] = dict(request.extra)
    out["samples"] = request.nested_cls((video, audio))
    return out


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
#: ``patcher.model_options`` entries that hook the stock sampling path.
UNSUPPORTED_MODEL_OPTIONS: Dict[str, str] = {
    "model_function_wrapper": "set_model_unet_function_wrapper",
    "sampler_cfg_function": "set_model_sampler_cfg_function",
    "sampler_post_cfg_function": "set_model_sampler_post_cfg_function",
    "sampler_pre_cfg_function": "set_model_sampler_pre_cfg_function",
    "sampler_calc_cond_batch_function": "set_model_sampler_calc_cond_batch_function",
    "denoise_mask_function": "set_model_denoise_mask_function",
}

#: ``transformer_options`` entries the causal lane cannot honour.
UNSUPPORTED_TRANSFORMER_OPTIONS: Dict[str, str] = {
    "wrappers": "comfy.patcher_extension wrappers (apply_model / diffusion_model / "
                "predict_noise ...) wrap comfy.samplers, which the chunk-major loop "
                "never calls",
    "patches": "transformer patches (attn1/attn2/middle ...) are applied by the dense "
               "forward, not by the causal chunk forward",
}


@dataclass(frozen=True)
class ResolvedModel:
    """A validated ``MODEL``: the patcher, the causal DiT, and the one options dict."""

    patcher: Any
    base_model: Any
    diffusion_model: Any
    transformer_options: Dict[str, Any]
    load_device: Any
    offload_device: Any
    num_layers: int


def _upstream_patcher_classes() -> Tuple[Optional[type], Optional[type]]:
    """``(ModelPatcher, ModelPatcherDynamic)`` if ComfyUI is importable, else ``(None, None)``."""
    try:
        module = importlib.import_module("comfy.model_patcher")
    except Exception:  # noqa: BLE001 - a bare environment is a supported test mode
        return None, None
    return getattr(module, "ModelPatcher", None), getattr(module, "ModelPatcherDynamic", None)


def resolve_transformer_options(patcher: Any) -> Dict[str, Any]:
    """One ``transformer_options`` dict for the whole rollout, validated.

    A **copy** of the patcher's own dict (so nothing this run does leaks back
    into the MODEL) built exactly once, because every cached forward of one
    rollout must see the same options object: a per-call dict would let a
    downstream consumer stash per-call state and silently lose it, and it makes
    "did anything change mid-rollout?" untestable.

    ``patches_replace['dit']`` is the load-bearing rejection: a replacement
    block is called with the official argument dict and would run **without the
    KV cache**, turning every cached chunk into a context-free one -- a quality
    regression with no error anywhere.
    """
    model_options = getattr(patcher, "model_options", None)
    if model_options is None:
        raise ContractError(
            f"{type(patcher).__name__} has no model_options; this is not a ComfyUI MODEL"
        )
    if not isinstance(model_options, Mapping):
        raise ContractError(
            f"model_options must be a dict, got {type(model_options).__name__}"
        )

    problems = []
    for key, setter in UNSUPPORTED_MODEL_OPTIONS.items():
        if model_options.get(key, None) is not None:
            problems.append(
                f"model_options[{key!r}] is set (via {setter}); the RAVEN streaming "
                "sampler does not run comfy.samplers, so it would never be called"
            )

    to = model_options.get("transformer_options", {}) or {}
    if not isinstance(to, Mapping):
        raise ContractError(
            f"model_options['transformer_options'] must be a dict, got {type(to).__name__}"
        )

    patches_replace = to.get("patches_replace", {}) or {}
    if isinstance(patches_replace, Mapping):
        dit_patches = patches_replace.get("dit", None)
        if dit_patches:
            problems.append(
                "transformer_options['patches_replace']['dit'] holds "
                f"{len(dit_patches)} block replacement(s) "
                f"({sorted(map(str, dit_patches))}); a replacement block receives the "
                "official argument dict and would run without the KV cache, silently "
                "turning cached chunks into context-free ones"
            )
        other = sorted(k for k, v in patches_replace.items() if k != "dit" and v)
        if other:
            problems.append(
                f"transformer_options['patches_replace'] holds unsupported entries {other}"
            )

    for key, why in UNSUPPORTED_TRANSFORMER_OPTIONS.items():
        if to.get(key, None):
            problems.append(f"transformer_options[{key!r}] is set: {why}")

    wrappers = getattr(patcher, "wrappers", None)
    if wrappers:
        problems.append(
            f"the MODEL carries patcher wrappers {sorted(map(str, wrappers))}; they wrap "
            "comfy.samplers entry points that the chunk-major loop never calls"
        )

    if problems:
        raise ContractError(
            "the MODEL carries hooks the RAVEN streaming sampler cannot honour:\n  "
            + "\n  ".join(problems)
        )
    return dict(to)


def resolve_model(model: Any, *, require_upstream_class: bool = True) -> ResolvedModel:
    """Validate a ``MODEL`` and hand back everything the rollout needs.

    Requirements, each failing loudly:

    * a **stock static** ``comfy.model_patcher.ModelPatcher`` -- v0.1's release
      contract is stock partial CPU offload (``loader.RavenLoaderSpec.
      force_static_patcher``), and the DynamicVRAM patcher has never been
      exercised with the causal lane;
    * a ``diffusion_model`` that is the RAVEN causal DiT, i.e. exposes
      ``prefill_text`` / ``forward_chunk``. Duck-typed on purpose: it keeps this
      function usable against a fake in a bare environment, and the real check
      that matters (does the cached forward exist at all?) is exactly this one.

    ``require_upstream_class=False`` skips the ``isinstance`` check against the
    real upstream class; it exists for tests that drive fakes while a real
    ComfyUI happens to be importable.
    """
    if model is None:
        raise ContractError("MODEL is required")

    patcher_cls, dynamic_cls = _upstream_patcher_classes()
    type_name = type(model).__name__
    if type_name == "ModelPatcherDynamic" or (
        dynamic_cls is not None and isinstance(model, dynamic_cls)
    ):
        raise ContractError(
            "the MODEL is a ModelPatcherDynamic (DynamicVRAM / comfy-aimdo). v0.1 runs "
            "on the stock ModelPatcher partial CPU offload path only; load the model "
            "with the RAVEN Model Loader, which forces the static patcher."
        )
    if require_upstream_class and patcher_cls is not None and not isinstance(model, patcher_cls):
        raise ContractError(
            f"MODEL must be a comfy.model_patcher.ModelPatcher, got {type_name}"
        )

    is_dynamic = getattr(model, "is_dynamic", None)
    if callable(is_dynamic) and bool(is_dynamic()):
        raise ContractError(
            f"{type_name}.is_dynamic() is True; v0.1 runs on the stock static "
            "ModelPatcher only"
        )

    base_model = getattr(model, "model", None)
    if base_model is None:
        raise ContractError(
            f"{type_name} has no .model (BaseModel); this is not a ComfyUI MODEL"
        )
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None:
        raise ContractError(
            f"{type(base_model).__name__} has no .diffusion_model; the streaming "
            "sampler needs the live DiT"
        )

    missing = [
        name for name in ("prefill_text", "forward_chunk")
        if not callable(getattr(diffusion_model, name, None))
    ]
    if missing:
        raise ContractError(
            f"the MODEL's diffusion_model ({type(diffusion_model).__name__}) has no "
            f"{missing}; the RAVEN streaming sampler needs the chunk-causal DiT built "
            "by the RAVEN Model Loader, not the stock bidirectional MiniMaxH3Model"
        )

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise ContractError(
            f"{type(diffusion_model).__name__} has no .blocks; the KV cache needs the "
            "layer count"
        )
    num_layers = len(blocks)
    if num_layers <= 0:
        raise ContractError("the DiT reports zero blocks")

    return ResolvedModel(
        patcher=model,
        base_model=base_model,
        diffusion_model=diffusion_model,
        transformer_options=resolve_transformer_options(model),
        load_device=getattr(model, "load_device", None),
        offload_device=getattr(model, "offload_device", None),
        num_layers=int(num_layers),
    )
