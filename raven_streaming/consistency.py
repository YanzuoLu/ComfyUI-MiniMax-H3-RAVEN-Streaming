"""Chunk-major fresh-noise consistency sampling for the RAVEN causal lane (M3).

What this is
------------
A faithful port of RAVEN's rollout -- ``ConsistencySampler`` +
``TrailingSamplingTimesteps`` driven by
``CausalMiniMaxH3Base._rollout_latents`` -- onto ComfyUI's official MiniMax H3
model, with the KV cache and the cached forward supplied by
:mod:`raven_streaming.cache` and :mod:`raven_streaming.causal_model`.

The loop is **chunk-major**, not sigma-major: a chunk is carried to completion
in ``steps`` NFEs, written into the cache as clean context, and only then does
the next chunk start. Stock ComfyUI samplers cannot express this, which is why
none of ``comfy.samplers`` is called (see ``docs/architecture.md`` §4).

The schedule
------------
Both streams run a *shifted trailing* grid, RAVEN's
``TrailingSamplingTimesteps`` with ``T = 1.0``, ``final_linear_steps = 0``::

    u_i   = 1 - i / N                       i = 0 .. N-1
    sigma = shift * u / (1 + (shift - 1) u)

so there are exactly ``N`` forwards per chunk, ``sigma_0 == 1`` for any shift
(the chunk starts at pure noise), and the last step's *next* sigma is ``0``
(RAVEN's ``get_next_timesteps`` clamps past the end and substitutes the bound).
Video and audio run **independent** grids -- ``shift`` 12 and 3 -- advanced by a
shared step index, exactly as the trial YAML configures
``sampling_timesteps`` / ``audio_sampling_timesteps``. This is *not* the stock
model's ``time_shift_sigma`` remap of one grid onto the other.

The step is RAVEN's consistency transition, in repo convention (``t = sigma``,
``t = 0`` is clean), on a ``LinearInterpolationSchedule``::

    x0     = x_t + sigma * v          # v is native H3 velocity, x0 - eps
    x_next = (1 - s) * x0 + s * eps   # eps is FRESH noise, never reused

The ``x0`` conversion carries **no negation**. H3's head predicts data-ward
velocity ``v = x0 - eps``; RAVEN's ``pred_type: x_0`` config feeds the sampler
an x0 that ``MiniMaxH3X0Model.forward`` has already produced through
``x0 = x_t + (1 - t_h3) * v`` (``projects/minimax_h3/modeling/scheduler.py::
minimax_h3_rf_v_to_x0``), with ``1 - t_h3 == sigma``. Our
:meth:`forward_chunk` returns that same raw H3 velocity (the official dense
``forward`` negates it for Comfy's convention; the causal lane deliberately does
not), so the identical expression applies here.

RNG contract
------------
One private device ``torch.Generator``, seeded from the node's ``seed``. Nothing
in this module ever touches global RNG state, and the draw order is fixed:

1. video initial noise (whole clip)
2. audio initial noise (whole clip)
3. video clean-context eps (whole clip)
4. audio clean-context eps (whole clip)
5. then, per chunk, per step: video step eps, audio step eps

The four full-clip draws happen **before any forward**, at full-clip shape, and
are then *sliced* per chunk. Drawing them per chunk instead would change every
sample that follows, so the shapes and the order are part of the reproducibility
contract, not an implementation detail -- this mirrors
``_rollout_latents`` exactly.

A step draws its fresh eps **even when the next sigma is 0**. The draw is
multiplied by zero and thrown away, but skipping it would desynchronise the
stream for every later chunk. The video and audio step draws are adjacent by
construction: no cancel check, no callback and no other draw may run between
them.

Where the KV cache lives
------------------------
``SamplerConfig.kv_cache_storage`` is handed straight to
:class:`raven_streaming.cache.ChunkKVCache` and decides whether the retained
K/V records sit on the compute device (``'gpu'``) or in host memory
(``'cpu_pinned'``, ``'cpu'``). It is a *residency* knob only: the transfers
never cast and never reorder, and the attention path assembles its merged
``[retained | current]`` slot the same way for every storage, so the same seed
and the same schedule produce the same tensors bit for bit. What changes is the
VRAM profile -- an off-device cache leaves one merged slot resident per layer
instead of the whole history -- and the PCIe traffic, which
:attr:`StreamingResult.kv_cache` reports.

The default is ``'gpu'`` so that a caller that predates the knob keeps its old
behaviour; the node passes the storage it wants explicitly.

Cancellation and delivery
-------------------------
``cancel_check`` is polled at every point where a long operation is about to
start or has just finished (:data:`CANCEL_POINTS`). On cancellation -- or on any
exception, including one raised by ``on_chunk`` -- the cache's partially staged
chunk is discarded and the exception propagates: **no partial result is ever
returned**.

``on_chunk`` is called once per completed chunk, after the chunk's ``x0`` has
been scattered into the full accumulators, and gets that chunk's own ``x0``
tensors. It is the hook the streaming decode/emission path (M3/M4) hangs off.
Neither ``on_chunk`` nor ``cancel_check`` may consume RNG; the generator state
is compared across every such call and a change fails loudly.

Import weight: torch, :mod:`raven_streaming.layout`, :mod:`raven_streaming.cache`
and :mod:`raven_streaming.contracts` -- all torch-only. ComfyUI is imported
lazily, and only to reach ``load_models_gpu``; every other upstream object
arrives through the arguments. The whole loop therefore runs against fakes in a
bare Python environment.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from raven_streaming import contracts
from raven_streaming import layout as layout_mod
from raven_streaming.cache import KV_STORAGES, ChunkKVCache

__all__ = [
    "SamplerError",
    "SamplingCancelled",
    "DEFAULT_STEPS",
    "DEFAULT_VIDEO_SHIFT",
    "DEFAULT_AUDIO_SHIFT",
    "DEFAULT_SINK",
    "DEFAULT_WINDOW",
    "DEFAULT_KV_CACHE_STORAGE",
    "KV_CACHE_STORAGES",
    "CANCEL_POINTS",
    "shifted_trailing_sigmas",
    "step_pairs",
    "SamplerConfig",
    "NoiseDraw",
    "RolloutRNG",
    "ChunkOutput",
    "StreamingResult",
    "sample_streaming",
]


class SamplerError(RuntimeError):
    """A streaming-sampler invariant was violated."""


class SamplingCancelled(RuntimeError):
    """``cancel_check`` asked for the run to stop; nothing partial is returned."""


#: RAVEN's published 4-NFE preview trial
#: (``minimax_h3_raven_streaming_lora_4nfe_preview.yaml``).
DEFAULT_STEPS = 4
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
#: cache chunks pinned from the start; chunk 0 is the text, so 2 == text + chunk 0
DEFAULT_SINK = 2
#: most recent cache chunks retained
DEFAULT_WINDOW: Optional[int] = 2

#: Where the KV cache's canonical records live. Re-exported from
#: :mod:`raven_streaming.cache` so a node can build its combo box from the
#: sampler's own surface without importing the cache module.
KV_CACHE_STORAGES: Tuple[str, ...] = KV_STORAGES
#: ``'gpu'``, deliberately: a caller that constructs a :class:`SamplerConfig`
#: without naming a storage gets exactly the residency it got before this knob
#: existed. The node passes ``'cpu_pinned'`` explicitly.
DEFAULT_KV_CACHE_STORAGE = "gpu"

#: Every point at which ``cancel_check`` is polled, in the order they occur.
#: Nothing sits between ``video step eps`` and ``audio step eps`` -- inserting a
#: poll there would invite a caller to consume RNG mid-step.
CANCEL_POINTS: Tuple[str, ...] = (
    "before_model_load",
    "before_prefill",
    "after_prefill",
    "before_noise_forward",
    "after_noise_forward",
    "after_step_update",
    "before_chunk_delivery",
    "before_clean",
    "after_clean",
    "before_return",
)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
def shifted_trailing_sigmas(steps: int, shift: float) -> Tuple[float, ...]:
    """RAVEN's shifted trailing grid, in closed form.

    Equivalent to ``TrailingSamplingTimesteps.set_timesteps(steps)`` with
    ``T = 1.0`` and ``final_linear_steps = 0``: ``np.arange(1.0, 0, -1/N)``
    followed by ``shift * t / (1 + (shift - 1) * t)``. The closed form avoids
    ``np.arange``'s float accumulation, which is why
    ``tests/test_consistency_schedule.py`` pins it against the numpy expression
    as well as against literal values.
    """
    steps = int(steps)
    if steps < 1:
        raise SamplerError(f"steps must be >= 1, got {steps}")
    shift = float(shift)
    if not (shift > 0.0):
        raise SamplerError(f"shift must be > 0, got {shift}")
    sigmas = []
    for i in range(steps):
        u = 1.0 - i / steps
        sigmas.append(shift * u / (1.0 + (shift - 1.0) * u))
    return tuple(sigmas)


def step_pairs(sigmas: Sequence[float]) -> Tuple[Tuple[float, float], ...]:
    """``[(sigma_i, sigma_{i+1})]`` with the last ``next`` pinned to ``0.0``.

    RAVEN's ``get_next_timesteps`` clamps the index at the end of the grid and
    then substitutes the continuous bound ``0.0``, so the final step lands
    exactly on ``x0``.
    """
    values = [float(s) for s in sigmas]
    if not values:
        raise SamplerError("empty sigma grid")
    return tuple(
        (values[i], values[i + 1] if i + 1 < len(values) else 0.0)
        for i in range(len(values))
    )


@dataclass(frozen=True)
class SamplerConfig:
    """The streaming sampler's knobs, with RAVEN's published defaults."""

    steps: int = DEFAULT_STEPS
    video_shift: float = DEFAULT_VIDEO_SHIFT
    audio_shift: float = DEFAULT_AUDIO_SHIFT
    sink: int = DEFAULT_SINK
    window: Optional[int] = DEFAULT_WINDOW
    seed: int = 0
    #: ``'gpu' | 'cpu_pinned' | 'cpu'`` -- where the KV cache keeps its
    #: canonical records. Purely a residency choice: every storage produces
    #: bit-identical results (the copies never cast and never reorder), so this
    #: trades host RAM and PCIe bandwidth for VRAM and nothing else. See
    #: :mod:`raven_streaming.cache`.
    kv_cache_storage: str = DEFAULT_KV_CACHE_STORAGE

    def __post_init__(self) -> None:
        if int(self.steps) != self.steps or self.steps < 1:
            raise SamplerError(f"steps must be a positive integer, got {self.steps!r}")
        for name in ("video_shift", "audio_shift"):
            value = getattr(self, name)
            if not (float(value) > 0.0):
                raise SamplerError(f"{name} must be > 0, got {value!r}")
        if int(self.sink) != self.sink or self.sink < 1:
            raise SamplerError(
                f"sink must be an integer >= 1, got {self.sink!r}: cache chunk 0 is the "
                "text prefill, and every media chunk attends it"
            )
        if self.window is not None and (int(self.window) != self.window or self.window < 0):
            raise SamplerError(
                f"window must be None (no eviction) or a non-negative integer, got "
                f"{self.window!r}"
            )
        if int(self.seed) != self.seed:
            raise SamplerError(f"seed must be an integer, got {self.seed!r}")
        if self.kv_cache_storage not in KV_CACHE_STORAGES:
            raise SamplerError(
                f"kv_cache_storage must be one of {KV_CACHE_STORAGES}, got "
                f"{self.kv_cache_storage!r}"
            )

    @property
    def video_sigmas(self) -> Tuple[float, ...]:
        return shifted_trailing_sigmas(self.steps, self.video_shift)

    @property
    def audio_sigmas(self) -> Tuple[float, ...]:
        return shifted_trailing_sigmas(self.steps, self.audio_shift)

    def video_steps(self) -> Tuple[Tuple[float, float], ...]:
        return step_pairs(self.video_sigmas)

    def audio_steps(self) -> Tuple[Tuple[float, float], ...]:
        return step_pairs(self.audio_sigmas)

    def describe(self) -> Dict[str, Any]:
        return {
            "steps": int(self.steps),
            "video_shift": float(self.video_shift),
            "audio_shift": float(self.audio_shift),
            "sink": int(self.sink),
            "window": self.window,
            "seed": int(self.seed),
            "kv_cache_storage": str(self.kv_cache_storage),
            "video_sigmas": list(self.video_sigmas),
            "audio_sigmas": list(self.audio_sigmas),
        }


# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------
class NoiseDraw(NamedTuple):
    """One recorded draw: what it was for, and its shape."""

    label: str
    shape: Tuple[int, ...]
    chunk: Optional[int] = None
    step: Optional[int] = None


class RolloutRNG:
    """A private device generator plus a log of every draw it served.

    Private on purpose: ``torch.randn`` without a generator would consume the
    process-global stream, so two runs of the same graph would differ depending
    on what else ComfyUI did in between. The log is what makes the draw *order*
    -- not just the values -- testable.
    """

    def __init__(self, seed: int, device: Any = "cpu") -> None:
        self.device = torch.device(device)
        self.seed = int(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.draws: List[NoiseDraw] = []

    def normal(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        label: str,
        chunk: Optional[int] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        shape = tuple(int(s) for s in shape)
        self.draws.append(NoiseDraw(label, shape, chunk, step))
        return torch.randn(
            shape, generator=self.generator, device=self.device, dtype=dtype
        )

    def state(self) -> torch.Tensor:
        return self.generator.get_state()


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkOutput:
    """One finished chunk, handed to ``on_chunk`` after it has been scattered.

    ``video_x0`` / ``audio_x0`` are the chunk's own tensors on the compute
    device (``[1, 24, t, H, W]`` and ``[1, 32, 2, a]``), not views into the
    accumulators, so a consumer may keep or move them freely. They are final:
    chunks are emitted in order and never revised.
    """

    index: int
    is_last: bool
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int
    video_x0: torch.Tensor
    audio_x0: torch.Tensor


@dataclass
class StreamingResult:
    """Everything one rollout produced."""

    latent: Dict[str, Any]
    layout: layout_mod.T2VALayout
    config: SamplerConfig
    video_sigmas: Tuple[float, ...] = ()
    audio_sigmas: Tuple[float, ...] = ()
    noise_forwards: int = 0
    clean_forwards: int = 0
    draws: Tuple[NoiseDraw, ...] = ()
    cancel_points: Tuple[str, ...] = ()
    #: :meth:`raven_streaming.cache.ChunkKVCache.stats` at the end of the run:
    #: where the K/V actually lived (a requested ``cpu_pinned`` may have fallen
    #: back to ``cpu``), how much host memory they took, the peak merged device
    #: slot, and the H2D/D2H traffic. Empty only for a result built by hand.
    kv_cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return self.layout.num_chunks

    def describe(self) -> Dict[str, Any]:
        return {
            "config": self.config.describe(),
            "frames": self.layout.frames,
            "num_chunks": self.num_chunks,
            "noise_forwards": self.noise_forwards,
            "clean_forwards": self.clean_forwards,
            "draws": [(d.label, list(d.shape)) for d in self.draws],
            "kv_cache": dict(self.kv_cache),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _takes_argument(func: Callable) -> bool:
    """Does ``func`` accept a positional argument (the cancel point name)?"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return True
    return False


def _default_load_models_gpu() -> Callable:
    """``comfy.model_management.load_models_gpu``, imported only when needed."""
    try:
        module = importlib.import_module("comfy.model_management")
    except Exception as exc:  # noqa: BLE001
        raise SamplerError(
            "comfy.model_management is not importable, so the model cannot be loaded "
            f"onto the compute device: {type(exc).__name__}: {exc}. Pass "
            "load_models=... to drive the sampler outside ComfyUI."
        ) from exc
    loader = getattr(module, "load_models_gpu", None)
    if not callable(loader):
        raise SamplerError("comfy.model_management.load_models_gpu is missing")
    return loader


class _Rollout:
    """One run of the chunk-major loop. Not reusable; one instance per sample."""

    def __init__(
        self,
        *,
        resolved: contracts.ResolvedModel,
        conditioning: contracts.TextConditioning,
        request: contracts.LatentRequest,
        layout: layout_mod.T2VALayout,
        config: SamplerConfig,
        device: torch.device,
        dtype: torch.dtype,
        compute_dtype: Optional[torch.dtype],
        on_chunk: Optional[Callable[[ChunkOutput], Any]],
        cancel_check: Optional[Callable[..., Any]],
    ) -> None:
        self.resolved = resolved
        self.conditioning = conditioning
        self.request = request
        self.layout = layout
        self.config = config
        self.device = device
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.on_chunk = on_chunk
        self.cancel_check = cancel_check
        self.cancel_takes_point = (
            cancel_check is not None and _takes_argument(cancel_check)
        )
        self.rng = RolloutRNG(config.seed, device)
        self.cache = ChunkKVCache(
            resolved.num_layers,
            sink=config.sink,
            window=config.window,
            storage=config.kv_cache_storage,
        )
        self.polled: List[str] = []
        self.noise_forwards = 0
        self.clean_forwards = 0

    # -- guarded callbacks ---------------------------------------------

    def _without_rng(self, func: Callable, *args: Any) -> Any:
        """Run a caller-supplied callable and prove it consumed no RNG.

        A hook that draws from our generator would shift every later chunk's
        noise, so the same seed would stop meaning the same video depending on
        whether a preview/cancel hook was connected. Comparing the generator
        state is cheap next to a DiT forward and turns that into a loud error.
        """
        before = self.rng.state()
        result = func(*args)
        after = self.rng.state()
        if not torch.equal(before, after):
            raise SamplerError(
                f"{getattr(func, '__name__', func)!r} consumed the sampler's RNG. "
                "Cancel hooks and chunk callbacks must not draw from the rollout "
                "generator: the seed would no longer determine the result."
            )
        return result

    def cancel_point(self, point: str) -> None:
        """Poll the cancel hook at one of :data:`CANCEL_POINTS`.

        Two shapes are supported, because both exist upstream: a *predicate*
        (``comfy.model_management.processing_interrupted``), whose truthy return
        stops the run, and a *thrower*
        (``throw_exception_if_processing_interrupted``), which returns ``None``
        and raises by itself. ``None`` is therefore never a cancellation.
        """
        self.polled.append(point)
        if self.cancel_check is None:
            return
        if self.cancel_takes_point:
            result = self._without_rng(self.cancel_check, point)
        else:
            result = self._without_rng(self.cancel_check)
        if result is not None and bool(result):
            raise SamplingCancelled(f"cancelled at {point}")

    # -- the loop ------------------------------------------------------

    def run(self, load_models: Callable) -> StreamingResult:
        try:
            return self._run(load_models)
        except BaseException:
            # A forward that aborted mid-block leaves staged K/V behind; the
            # next commit would then mix two chunks' rows. Nothing partial is
            # returned either way, so the cache is left in a clean state and the
            # original exception propagates untouched.
            self.cache.discard_pending()
            raise

    def _run(self, load_models: Callable) -> StreamingResult:
        model = self.resolved.diffusion_model
        options = self.resolved.transformer_options
        layout = self.layout
        config = self.config

        self.cancel_point("before_model_load")
        # Exactly once, for the whole rollout. Comfy's loader decides how much
        # of the model fits and keeps the rest offloaded; the streaming loop
        # never calls partially_load / partially_unload / cleanup itself, so it
        # cannot fight that decision chunk by chunk.
        load_models([self.resolved.patcher], memory_required=0, force_full_load=False)

        video_shape = layout.video_latent_shape(contracts.VIDEO_LATENT_CHANNELS)
        audio_shape = layout.audio_latent_shape(contracts.AUDIO_LATENT_CHANNELS)

        # --- the four full-clip draws, in order, before any forward ---
        video_noise = self.rng.normal(
            video_shape, dtype=self.dtype, label="video_initial_noise")
        audio_noise = self.rng.normal(
            audio_shape, dtype=self.dtype, label="audio_initial_noise")
        # Clean-context eps is drawn up front and carried: a chunk's cache-fill
        # rows are 0.999 * x0 + 0.001 * eps, and reproducing a run means
        # reproducing exactly those rows.
        video_clean_eps = self.rng.normal(
            video_shape, dtype=self.dtype, label="video_clean_eps")
        audio_clean_eps = self.rng.normal(
            audio_shape, dtype=self.dtype, label="audio_clean_eps")

        self.cancel_point("before_prefill")
        # Text is cache chunk 0 and is written alone, once per rollout. The
        # context is cast to the compute dtype here, the way upstream's
        # ``MiniMaxH3.extra_conds`` casts before running the refiner: the CLIP
        # hands back whatever dtype it encoded in, and cached text K/V in a
        # different dtype than the chunks' would be refused by the attention
        # module (one cache, one compute dtype, one device).
        model.prefill_text(
            self.conditioning.cross_attn.to(device=self.device, dtype=self.compute_dtype),
            cache=self.cache,
            transformer_options=options,
            text_token_tags=(
                None if self.conditioning.token_tags is None
                else self.conditioning.token_tags.to(self.device)
            ),
            compute_dtype=self.compute_dtype,
        )
        self.cancel_point("after_prefill")

        video_x0 = torch.zeros_like(video_noise)
        audio_x0 = torch.zeros_like(audio_noise)

        video_steps = config.video_steps()
        audio_steps = config.audio_steps()

        for chunk in layout.chunks:
            index = chunk.index
            is_last = index == layout.num_chunks - 1
            video_xt = video_noise[:, :, chunk.video_start:chunk.video_stop].clone()
            audio_xt = audio_noise[:, :, :, chunk.audio_start:chunk.audio_stop].clone()

            for step in range(config.steps):
                video_sigma, video_next = video_steps[step]
                audio_sigma, audio_next = audio_steps[step]

                self.cancel_point("before_noise_forward")
                video_v, audio_v = model.forward_chunk(
                    video_latent=video_xt,
                    audio_latent=audio_xt,
                    layout=layout,
                    chunk_index=index,
                    cache=self.cache,
                    role="noise",
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                    update_cache=False,
                    transformer_options=options,
                    compute_dtype=self.compute_dtype,
                )
                self.noise_forwards += 1
                self.cancel_point("after_noise_forward")

                # native H3 velocity: x0 = x_t + sigma * v, no negation
                video_pred_x0 = video_xt + video_sigma * video_v
                audio_pred_x0 = audio_xt + audio_sigma * audio_v

                # Both draws here, back to back and in this order. Nothing may
                # be inserted between them, even at s == 0: the draw still
                # happens (and is then multiplied by zero) because skipping it
                # would shift the stream for every later chunk.
                video_eps = self.rng.normal(
                    video_xt.shape, dtype=self.dtype,
                    label="video_step_eps", chunk=index, step=step)
                audio_eps = self.rng.normal(
                    audio_xt.shape, dtype=self.dtype,
                    label="audio_step_eps", chunk=index, step=step)

                video_xt = (1.0 - video_next) * video_pred_x0 + video_next * video_eps
                audio_xt = (1.0 - audio_next) * audio_pred_x0 + audio_next * audio_eps
                self.cancel_point("after_step_update")

            # The last step's next sigma is 0, so x_t is exactly x0 here.
            self.cancel_point("before_chunk_delivery")
            video_x0[:, :, chunk.video_start:chunk.video_stop] = video_xt
            audio_x0[:, :, :, chunk.audio_start:chunk.audio_stop] = audio_xt
            if self.on_chunk is not None:
                self._without_rng(
                    self.on_chunk,
                    ChunkOutput(
                        index=index,
                        is_last=is_last,
                        video_start=chunk.video_start,
                        video_stop=chunk.video_stop,
                        audio_start=chunk.audio_start,
                        audio_stop=chunk.audio_stop,
                        video_x0=video_xt,
                        audio_x0=audio_xt,
                    ),
                )

            if is_last:
                # Nothing after it will read this history, so the last chunk is
                # never written into the cache. Writing it would cost a full
                # forward and one more chunk record for no reader.
                continue

            self.cancel_point("before_clean")
            model.forward_chunk(
                video_latent=video_xt,
                audio_latent=audio_xt,
                layout=layout,
                chunk_index=index,
                cache=self.cache,
                role="clean",
                video_eps=video_clean_eps[:, :, chunk.video_start:chunk.video_stop],
                audio_eps=audio_clean_eps[:, :, :, chunk.audio_start:chunk.audio_stop],
                update_cache=True,
                transformer_options=options,
                compute_dtype=self.compute_dtype,
            )
            self.clean_forwards += 1
            self.cancel_point("after_clean")

        self.cancel_point("before_return")
        out_video = video_x0.to(device=self.request.device, dtype=self.request.dtype)
        out_audio = audio_x0.to(device=self.request.device, dtype=self.request.dtype)
        return StreamingResult(
            latent=contracts.build_output_latent(self.request, out_video, out_audio),
            layout=layout,
            config=config,
            video_sigmas=config.video_sigmas,
            audio_sigmas=config.audio_sigmas,
            noise_forwards=self.noise_forwards,
            clean_forwards=self.clean_forwards,
            draws=tuple(self.rng.draws),
            cancel_points=tuple(self.polled),
            kv_cache=self.cache.stats(),
        )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def sample_streaming(
    *,
    model: Any,
    positive: Any,
    latent: Any,
    config: Optional[SamplerConfig] = None,
    on_chunk: Optional[Callable[[ChunkOutput], Any]] = None,
    cancel_check: Optional[Callable[..., Any]] = None,
    device: Any = None,
    compute_dtype: Optional[torch.dtype] = None,
    load_models: Optional[Callable] = None,
    require_upstream_class: bool = True,
    warn_experimental: bool = True,
) -> StreamingResult:
    """Run one chunk-major RAVEN rollout and return the finished AV latent.

    ``model`` / ``positive`` / ``latent`` are the node's raw sockets; they are
    parsed by :mod:`raven_streaming.contracts` and every unsupported feature is
    rejected before a single weight is touched.

    ``device`` defaults to the patcher's ``load_device``, which is where
    ``load_models_gpu`` will have put the model. ``compute_dtype`` defaults to
    the DiT's own dtype and is used for **both** the text prefill and every
    chunk forward, so the KV cache is filled and read in one dtype.
    ``load_models`` defaults to ``comfy.model_management.load_models_gpu`` and
    is injectable so the loop can be driven against a fake model with no
    ComfyUI present.

    Raises rather than returning anything partial: :class:`SamplingCancelled`
    when ``cancel_check`` asks to stop, :class:`contracts.ContractError` for a
    rejected input, and whatever ``on_chunk`` raised if it did.
    """
    config = config or SamplerConfig()
    resolved = contracts.resolve_model(model, require_upstream_class=require_upstream_class)
    conditioning = contracts.parse_conditioning(positive)
    request = contracts.parse_latent(latent, warn_experimental=warn_experimental)
    layout = request.layout(conditioning.text_len, warn_experimental=False)

    if device is None:
        device = resolved.load_device if resolved.load_device is not None else request.device
    device = torch.device(device)

    if compute_dtype is None:
        # Resolve it here rather than letting each call fall back on its own:
        # ``forward_chunk`` defaults to the DiT's dtype while ``prefill_text``
        # defaults to the *context's*, and a cache filled in one dtype and read
        # in another is refused mid-rollout. One value, decided once.
        compute_dtype = getattr(resolved.diffusion_model, "dtype", None)
    if compute_dtype is None:
        compute_dtype = request.dtype

    if load_models is None:
        load_models = _default_load_models_gpu()

    rollout = _Rollout(
        resolved=resolved,
        conditioning=conditioning,
        request=request,
        layout=layout,
        config=config,
        device=device,
        dtype=request.dtype,
        compute_dtype=compute_dtype,
        on_chunk=on_chunk,
        cancel_check=cancel_check,
    )
    return rollout.run(load_models)
