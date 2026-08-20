"""Deterministic numpy fakes standing in for the real VAE decoders.

These exist so the streaming state machines can be falsified *without* a GPU,
a checkpoint, or even torch.  They are not approximations of the real models -
they only have to reproduce the structural properties the coordinators rely on:

* :class:`FakeVideoChunkDecoder` - ``_adaptive_decode`` maps ``T`` latents to
  ``T * vae_ratio_t`` frames with every output frame depending on *every* input
  latent of the clip, so any temporal misalignment changes the result.
* :class:`FiniteRFAudioDecoder` - a decoder with an exactly known finite
  receptive field of ``radius`` latents and implicit zero padding, so
  overlap-save is exact iff ``margin >= radius``.

numpy is the only dependency.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "numpy_concat",
    "FakeVideoChunkDecoder",
    "make_latents",
    "FiniteRFAudioDecoder",
    "PhaseSensitiveAudioDecoder",
    "JitteryAudioDecoder",
    "NondeterministicAudioDecoder",
    "make_audio_latents",
]


def numpy_concat(parts: Sequence[Any], dim: int) -> Any:
    return np.concatenate(list(parts), axis=dim)


class FakeVideoChunkDecoder:
    """Stand-in for ``MiniMaxH3VideoVAE``'s ``_adaptive_decode`` / ``blend``."""

    #: ``finalize_mode`` values accepted by the constructor.
    FINALIZE_MODES = ("pointwise", "cross_frame")

    def __init__(
        self,
        vae_ratio_t: int = 4,
        out_channels: int = 3,
        spatial_scale: int = 1,
        finalize_mode: str = "pointwise",
    ) -> None:
        self.vae_ratio_t = int(vae_ratio_t)
        self.out_channels = int(out_channels)
        self.spatial_scale = int(spatial_scale)
        if finalize_mode not in self.FINALIZE_MODES:
            raise ValueError(
                "finalize_mode must be one of {}, got {!r}".format(
                    self.FINALIZE_MODES, finalize_mode
                )
            )
        self.finalize_mode = finalize_mode
        self.decode_calls = 0
        self.decoded_latents = 0
        self.finalize_calls = 0
        self.finalized_frames = 0

    # -- protocol -----------------------------------------------------------

    def _adaptive_decode(self, z: Any) -> Any:
        z = np.asarray(z, dtype=np.float64)
        b, c, t, h, w = z.shape
        self.decode_calls += 1
        self.decoded_latents += t
        frames = t * self.vae_ratio_t
        # dense (frame, latent) mixing: no output frame is a copy of one latent
        weights = np.empty((frames, t), dtype=np.float64)
        for f in range(frames):
            for k in range(t):
                weights[f, k] = math.cos(0.7 * f + 1.3 * k) + 0.5 * math.sin(0.11 * f * (k + 1))
        # [b, c, t, h, w] x [f, t] -> [b, c, f, h, w]
        mixed = np.einsum("bcthw,ft->bcfhw", z, weights)
        if c >= self.out_channels:
            mixed = mixed[:, : self.out_channels]
        else:
            mixed = np.repeat(mixed, math.ceil(self.out_channels / c), axis=1)[
                :, : self.out_channels
            ]
        bias = np.arange(frames, dtype=np.float64).reshape(1, 1, frames, 1, 1) * 1e-3
        out = mixed + bias
        if self.spatial_scale > 1:
            out = np.repeat(np.repeat(out, self.spatial_scale, axis=-2), self.spatial_scale, axis=-1)
        return out

    def blend(self, a: Any, b: Any, blend_extent: int, dim: int) -> Any:
        """Port of ``MiniMaxH3VideoVAE.blend`` in numpy."""
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = np.arange(blend_extent, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.reshape(shape)
        weight_b = weight_b.reshape(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(a.shape[dim] - blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent < b.shape[dim]:
            slice_rest = [slice(None)] * b.ndim
            slice_rest[dim] = slice(blend_extent, None)
            return np.concatenate([blended, b[tuple(slice_rest)]], axis=dim)
        return blended

    def _finalize_pixels(self, part: Any) -> Any:
        """Stand-in for ``MiniMaxH3VideoVAE._finalize_pixels``.

        ``pointwise`` mirrors the real hook's algebra: the real one broadcasts
        ``pixel_mean``/``pixel_std`` of shape ``[1, 3, 1, 1, 1]``, so it is
        strictly elementwise and *cannot* tell finalize-then-crop from
        crop-then-finalize.

        ``cross_frame`` deliberately can.  Every output frame depends on the sum
        over **all** frames of the part (including later ones) and on the part's
        own length, so ``finalize(x)[:k] != finalize(x[:k])`` whenever ``k < n``.
        That is what pins the coordinator to upstream's operator order rather
        than to an algebraic coincidence that a future hook could break.
        """
        self.finalize_calls += 1
        self.finalized_frames += int(part.shape[2])
        base = part * 0.5 + 0.25
        if self.finalize_mode == "pointwise":
            # monotone and unclamped, so a wrong frame is never masked by saturation
            return base
        n = int(part.shape[2])
        whole_part = np.sum(part, axis=2, keepdims=True)  # depends on every frame
        return base + whole_part / float(n + 1)  # ... and on the part length


def make_latents(t: int, channels: int = 24, h: int = 2, w: int = 2, seed: int = 0) -> Any:
    """Deterministic latent tensor ``[1, channels, t, h, w]``."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal((1, channels, t, h, w)).astype(np.float64)


class FiniteRFAudioDecoder:
    """Non-causal decoder with an exactly known receptive field.

    ``out[..., n*spl:(n+1)*spl] = sum_{d=-R..R} K[d] @ z[..., n+d]`` with the
    signal treated as zero outside the given slice - which is precisely the
    implicit padding a real conv stack applies at a slice edge.  Overlap-save
    with ``margin >= R`` therefore reproduces the full decode exactly, and with
    ``margin < R`` it cannot.
    """

    def __init__(
        self,
        radius: int,
        samples_per_latent: int = 8,
        latent_channels: int = 4,
        seed: int = 0,
    ) -> None:
        self.radius = int(radius)
        self.samples_per_latent = int(samples_per_latent)
        self.latent_channels = int(latent_channels)
        rng = np.random.RandomState(seed)
        # kernels[d] : [samples_per_latent, latent_channels]
        self.kernels = [
            rng.standard_normal((self.samples_per_latent, self.latent_channels))
            for _ in range(2 * self.radius + 1)
        ]
        self.calls = 0
        self.decoded_latents = 0

    def __call__(self, z: Any) -> Any:
        """``z``: ``[B, C, S, T]`` -> waveform ``[B, S, T * samples_per_latent]``."""
        z = np.asarray(z, dtype=np.float64)
        b, c, s, t = z.shape
        self.calls += 1
        self.decoded_latents += t
        spl = self.samples_per_latent
        out = np.zeros((b, s, t * spl), dtype=np.float64)
        if t == 0:
            return out
        for d in range(-self.radius, self.radius + 1):
            k = self.kernels[d + self.radius]  # [spl, c]
            lo = max(0, -d)
            hi = min(t, t - d)
            if lo >= hi:
                continue
            src = z[:, :, :, lo + d:hi + d]  # [b, c, s, n]
            contrib = np.einsum("bcsn,pc->bsnp", src, k)  # [b, s, n, spl]
            n = hi - lo
            out[:, :, lo * spl:hi * spl] += contrib.reshape(b, s, n * spl)
        return out


class PhaseSensitiveAudioDecoder(FiniteRFAudioDecoder):
    """A decoder that is only shift-equivariant to multiples of ``period``.

    Real anti-aliased vocoders contain strided resamplers whose decimation grid
    is anchored to the tensor's origin.  If such a stage is ever *not* balanced
    by a matching upsample, slicing the latents at an unaligned index changes
    the whole output rather than just its edges - and no margin can repair that,
    because it is not a boundary effect.

    This fake reproduces exactly that failure mode: the output picks up a term
    that depends on ``latent_start mod period``, which the caller cannot see.
    It is the falsification target for ``OverlapSavePlanner(phase_align=...)``.
    """

    def __init__(
        self,
        radius: int,
        period: int = 2,
        samples_per_latent: int = 8,
        latent_channels: int = 4,
        seed: int = 0,
        strength: float = 1.0,
    ) -> None:
        super().__init__(radius, samples_per_latent, latent_channels, seed)
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = int(period)
        self.strength = float(strength)
        #: set by the harness to say where this slice starts in the full stream
        self.latent_start = 0

    def __call__(self, z: Any) -> Any:
        out = super().__call__(z)
        phase = self.latent_start % self.period
        if phase == 0:
            return out
        # a global, non-boundary perturbation: exactly what margin cannot fix
        return out + self.strength * (phase / float(self.period))


class JitteryAudioDecoder(FiniteRFAudioDecoder):
    """A decoder whose output depends on the *length* of its input.

    Stands in for cuDNN autotune picking a different convolution algorithm per
    input shape: the perturbation is tiny, global, and - critically - varies
    non-monotonically with the margin, so growing the context never converges.

    Because it is a pure function of the input length, the full-sequence control
    (``margin >= total``) still comes out **exact**: every block then decodes a
    tensor of the same length as the reference.  That combination - plateau,
    non-monotone, but full-sequence exact - is the fingerprint of shape-keyed
    algorithm selection rather than of run-to-run nondeterminism.
    """

    def __init__(
        self,
        radius: int,
        samples_per_latent: int = 8,
        latent_channels: int = 4,
        seed: int = 0,
        jitter: float = 1e-2,
    ) -> None:
        super().__init__(radius, samples_per_latent, latent_channels, seed)
        self.jitter = float(jitter)

    def __call__(self, z: Any) -> Any:
        out = super().__call__(z)
        n = int(z.shape[-1])
        # deterministic per length, but with no trend in n
        wobble = math.sin(12.9898 * n) * 43758.5453
        return out + self.jitter * (wobble - math.floor(wobble) - 0.5)


class NondeterministicAudioDecoder(FiniteRFAudioDecoder):
    """A decoder that returns something slightly different on every call.

    Stands in for genuine run-to-run nondeterminism (atomics, split-k reduction
    order, autotune re-benchmarking).  Unlike :class:`JitteryAudioDecoder`, this
    one makes even the full-sequence control non-zero, which is the signal that
    the problem is not overlap-save at all.
    """

    def __init__(
        self,
        radius: int,
        samples_per_latent: int = 8,
        latent_channels: int = 4,
        seed: int = 0,
        jitter: float = 1e-2,
    ) -> None:
        super().__init__(radius, samples_per_latent, latent_channels, seed)
        self.jitter = float(jitter)

    def __call__(self, z: Any) -> Any:
        out = super().__call__(z)
        wobble = math.sin(7.13 * self.calls) * 0.5
        return out + self.jitter * wobble


def make_audio_latents(t: int, channels: int = 4, stereo: int = 2, seed: int = 0) -> Any:
    """Deterministic latent tensor ``[1, channels, stereo, t]``."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal((1, channels, stereo, t)).astype(np.float64)
