#!/usr/bin/env python3
"""M0 probe 3: how much lookahead does the non-causal audio VAE actually need?

The MiniMax H3 audio decoder (BigVGAN) is not causal, so a chunk decoded on its
own is *not* a slice of the full decode.  Padding-based guesswork is not an
option; the required context has to be measured.

This probe grows the overlap-save latent margin from small to large against a
real audio VAE and reports the smallest margin at which chunked decoding
matches a full decode to within ``--tol`` (default 1e-5).  Output:

* the minimum margin in latents, samples and **seconds of added latency**
* an explicit verdict if a single chunk of lookahead is not enough, naming the
  delay the pipeline would have to accept

No model path is hardcoded::

    python tools/probe_audio_overlap.py \\
        --comfy-root /workspace/ComfyUI \\
        --vae /workspace/ComfyUI/models/vae/minimax_h3_audio_vae.safetensors \\
        --device cuda --latents 400 --block 40 --tol 1e-5

``--fake`` runs the whole search against a numpy decoder with a known receptive
field, which validates the harness without a GPU or a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.media.audio_stream import (  # noqa: E402
    AudioLatentGeometry,
    OverlapSavePlanner,
    decode_overlap_save,
    diff_block_by_block,
    diff_stats,
    max_abs_diff,
    probe_shift_equivariance,
    saturation_stats,
    search_latent_margin,
)

#: Two co-prime block sizes: if the required margin were an artefact of the
#: block grid rather than the decoder's receptive field, these would disagree.
DEFAULT_BLOCKS = "28,29"

ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ENVIRONMENT = 2


def _resolve_comfy_root(explicit: Optional[str]) -> Optional[str]:
    """``--comfy-root`` then COMFYUI_PATH then COMFYUI_UPSTREAM_PATH."""
    for candidate in [explicit] + [os.environ.get(v) for v in ENV_VARS]:
        if not candidate:
            continue
        root = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(root):
            if candidate == explicit:
                raise SystemExit("--comfy-root does not exist: {}".format(root))
            continue
        return root
    return None


def _add_comfy_root(comfy_root: Optional[str]) -> Optional[str]:
    root = _resolve_comfy_root(comfy_root)
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


def _resolve_vae(vae: str, comfy_root: Optional[str]) -> str:
    path = os.path.abspath(os.path.expanduser(vae))
    if os.path.isfile(path):
        return path
    if comfy_root:
        candidate = os.path.join(comfy_root, "models", "vae", vae)
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit("VAE checkpoint not found: {}".format(vae))


def _verdict(result, geometry: AudioLatentGeometry, block: int, video_chunk_seconds: Fraction) -> bool:
    print()
    print(result.describe())
    print()
    if not result.found:
        print("VERDICT: one chunk of lookahead is NOT sufficient at this tolerance.")
        print("         Increase --max-margin to find the real figure, or relax --tol.")
        return False

    latency = result.lookahead_seconds
    print("VERDICT: overlap-save needs {} latent(s) of right context.".format(result.margin))
    print("         = {} samples = {:.4f} s of added audio latency.".format(
        result.lookahead_samples, float(latency)))
    print("         Left context must also be retained: {} latent(s) behind the "
          "write cursor.".format(result.margin))
    if result.margin > block:
        print("         WARNING: margin ({}) exceeds the block size ({}), so a single "
              "block of lookahead is NOT enough; the stream must buffer "
              "{:.4f} s (> {:.4f} s per block).".format(
                  result.margin, block, float(latency),
                  float(geometry.latents_to_seconds(block))))
    print("         For reference one video chunk (17 frames @ 24 fps) is {:.4f} s; "
          "audio lookahead is {:.2f} video chunks.".format(
              float(video_chunk_seconds), float(latency / video_chunk_seconds)))
    return True


def _result_dict(result, geometry: AudioLatentGeometry, block: int) -> Dict[str, Any]:
    return {
        "block_latents": block,
        "margin": result.margin,
        "found": result.found,
        "tolerance": result.tolerance,
        "max_margin": result.max_margin,
        "errors": [[m, e] for m, e in result.errors],
        "lookahead_samples": result.lookahead_samples,
        "lookahead_seconds": (float(result.lookahead_seconds)
                              if result.lookahead_seconds is not None else None),
        "samples_per_latent": geometry.samples_per_latent,
        "sample_rate": geometry.sample_rate,
        "margin_exceeds_block": (result.margin is not None and result.margin > block),
    }


def _blocks(args) -> List[int]:
    return [int(x) for x in str(args.block).split(",") if x.strip()]


def run_fake(args) -> Dict[str, Any]:
    import numpy as np

    from raven_streaming.media.fakes import FiniteRFAudioDecoder, make_audio_latents

    spl = 8
    geometry = AudioLatentGeometry(samples_per_latent=spl, sample_rate=spl * 40)
    decoder = FiniteRFAudioDecoder(args.fake_radius, samples_per_latent=spl, seed=0)
    z = make_audio_latents(args.latents, channels=decoder.latent_channels, seed=0)
    reference = decoder(z)

    print("mode: FAKE (numpy, true receptive field radius = {} latents)".format(args.fake_radius))
    out: Dict[str, Any] = {"mode": "fake", "fake_radius": args.fake_radius, "blocks": []}
    all_ok = True
    for block in _blocks(args):
        print("\n--- block_latents = {} ---".format(block))
        result = search_latent_margin(
            decoder, z, reference, tolerance=args.tol, block_latents=block,
            max_margin=args.max_margin, geometry=geometry, concat=np.concatenate,
            phase_align=args.phase_align, prefix_mode=bool(args.prefix_mode),
        )
        ok = _verdict(result, geometry, block, Fraction(17, 24))
        correct = result.margin == args.fake_radius
        print("harness self-check: recovered margin {} (true {}) -> {}".format(
            result.margin, args.fake_radius, "OK" if correct else "MISMATCH"))
        entry = _result_dict(result, geometry, block)
        entry["recovered_true_radius"] = correct
        entry["passed"] = bool(ok and correct)
        out["blocks"].append(entry)
        all_ok = all_ok and entry["passed"]

    _compare_blocks(out["blocks"])
    out["passed"] = all_ok
    print("\nRESULT: {}".format("PASSED" if all_ok else "FAILED"))
    return out


def _compare_blocks(entries: List[Dict[str, Any]]) -> None:
    """A receptive field is a property of the decoder, not of the block grid."""
    if len(entries) < 2:
        return
    margins = {e["block_latents"]: e["margin"] for e in entries}
    print("\nmargin by block size: {}".format(margins))
    distinct = set(m for m in margins.values() if m is not None)
    if len(distinct) <= 1:
        print("  -> consistent across block sizes: the margin is a property of the "
              "decoder's receptive field, not of the block grid.")
    else:
        print("  -> WARNING: block-size dependent margins {}. Use the largest ({}) "
              "and investigate: a real receptive field should not vary with the "
              "block grid.".format(sorted(distinct), max(distinct)))


def _import_comfy():
    """Import ComfyUI without letting its CLI parser see our argv."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore

        return comfy
    finally:
        sys.argv = saved


def _pin_determinism(torch) -> Dict[str, Any]:
    """Remove every source of shape-keyed / run-to-run kernel variation we can.

    cuDNN benchmark mode autotunes per input shape, so a windowed decode and a
    full decode can legitimately land on different convolution algorithms and
    disagree by ~1e-2 with no bug anywhere.  That would masquerade as a
    receptive-field result that never converges, so it has to be off before any
    margin number means anything.
    """
    flags: Dict[str, Any] = {}
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        flags["cudnn.benchmark"] = bool(torch.backends.cudnn.benchmark)
        flags["cudnn.deterministic"] = bool(torch.backends.cudnn.deterministic)
    except Exception as exc:
        flags["cudnn"] = "unavailable: {}".format(exc)
    for path, name in (("cudnn", "allow_tf32"), ("cuda.matmul", "allow_tf32")):
        try:
            obj = torch.backends
            for part in path.split("."):
                obj = getattr(obj, part)
            setattr(obj, name, False)
            flags["{}.{}".format(path, name)] = bool(getattr(obj, name))
        except Exception as exc:
            flags["{}.{}".format(path, name)] = "unavailable: {}".format(exc)
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        flags["cuda.matmul.fp32_precision"] = "ieee"
    except Exception:
        pass
    return flags


def _abba_determinism(torch, decode_fn, z, block_latents: int) -> Dict[str, Any]:
    """Is the decoder reproducible at all?  Same tensor, and same slice, twice."""
    full_a = decode_fn(z)
    full_b = decode_fn(z)
    total = int(z.shape[-1])
    lo = min(block_latents, max(0, total - block_latents))
    hi = min(total, lo + block_latents)
    slice_a = decode_fn(z[..., lo:hi])
    slice_b = decode_fn(z[..., lo:hi])
    out = {
        "full_vs_full": diff_stats(full_a, full_b),
        "slice_vs_slice": diff_stats(slice_a, slice_b),
        "slice_latents": [lo, hi],
    }
    print("determinism (ABBA):")
    print("  full decode twice   : max|diff|={:.3e} exact={}".format(
        out["full_vs_full"]["max_abs_diff"], out["full_vs_full"]["exact"]))
    print("  same slice twice    : max|diff|={:.3e} exact={}".format(
        out["slice_vs_slice"]["max_abs_diff"], out["slice_vs_slice"]["exact"]))
    out["deterministic"] = bool(
        out["full_vs_full"]["exact"] and out["slice_vs_slice"]["exact"])
    if not out["deterministic"]:
        print("  -> the decoder is NOT reproducible run to run; no margin search on "
              "this box can mean anything until that is fixed.")
    return out


def _full_sequence_control(decode_fn, z, reference, block_latents, geometry, concat) -> Dict[str, Any]:
    """margin >= total: every block decodes the whole z, then slices it."""
    total = int(z.shape[-1])
    streamed = decode_overlap_save(
        decode_fn, z, total, block_latents, geometry, concat=concat
    )
    stats = diff_stats(streamed, reference)
    print("full-sequence control (margin={} >= total): max|diff|={:.3e} exact={}".format(
        total, stats["max_abs_diff"], stats["exact"]))
    if not stats["exact"]:
        print("  -> DECISIVE: this configuration performs no windowing at all - each "
              "block decodes the identical tensor the reference used and is then "
              "sliced. A non-zero result CANNOT be an overlap-save or "
              "receptive-field effect; it is nondeterminism or a harness bug.")
    else:
        print("  -> the planner and slicing are exact; any margin error is genuinely "
              "about missing context (or about shape-keyed kernel selection).")
    return stats


def _report_blocks(decode_fn, z, reference, margin, block_latents, geometry,
                   phase_align, prefix_mode, limit=12) -> List[Dict[str, Any]]:
    diffs = diff_block_by_block(
        decode_fn, z, reference, margin=margin, block_latents=block_latents,
        geometry=geometry, phase_align=phase_align, prefix_mode=prefix_mode,
    )
    print("per-block diff (margin={}, block={}, phase_align={}, prefix={}):".format(
        margin, block_latents, phase_align, prefix_mode))
    for d in diffs[:limit]:
        print(d.describe())
    if len(diffs) > limit:
        print("  ... {} more blocks".format(len(diffs) - limit))
    worst = max(diffs, key=lambda d: d.max_abs_diff) if diffs else None
    if worst is not None:
        print("  worst block: {} (max={:.3e} at absolute sample {}{})".format(
            worst.request.index, worst.max_abs_diff, worst.worst_sample_absolute,
            ", FULL-SEQUENCE" if worst.is_full_sequence else ""))
    full_seq_bad = [d for d in diffs if d.is_full_sequence and not d.exact]
    if full_seq_bad:
        print("  !! {} full-sequence request(s) are non-zero -> harness/nondeterminism, "
              "not overlap-save".format(len(full_seq_bad)))
    return [d.to_dict() for d in diffs]


def _report_phase(decode_fn, z, geometry, block_latents) -> List[Dict[str, Any]]:
    """Does shifting the window's left edge change the shared interior?"""
    total = int(z.shape[-1])
    start = min(2 * block_latents, max(8, total // 3))
    stop = min(total, start + max(3 * block_latents, 24))
    if stop - start < 20:
        return []
    results = probe_shift_equivariance(
        decode_fn, z, start, stop, shifts=(1, 2, 3, 4, 5),
        geometry=geometry, inset_latents=8,
    )
    print("shift-equivariance (fixed {}-latent window slid left, interior inset by 8; "
          "only meaningful if the decoder is reproducible):".format(stop - start))
    for r in results:
        print("  shift {:>2} latents (start {:>4}, parity {}): max|diff|={:.3e} exact={}".format(
            r["shift_latents"], r["latent_start"], r["parity"],
            r["max_abs_diff"], r["exact"]))
    if results:
        odd = [r for r in results if r["shift_latents"] % 2 == 1]
        even = [r for r in results if r["shift_latents"] % 2 == 0]
        odd_bad = [r for r in odd if not r["exact"]]
        even_ok = [r for r in even if r["exact"]]
        if odd_bad and even_ok and len(even_ok) == len(even):
            print("  -> PHASE DEPENDENCE with period 2: odd shifts break the interior, "
                  "even shifts do not. Use OverlapSavePlanner(phase_align=2) so every "
                  "decode starts on an even latent index.")
        elif all(r["exact"] for r in results):
            print("  -> the decoder is shift-equivariant: slicing at any latent index is "
                  "phase-safe, so a residual error is a boundary/context effect, not phase.")
        else:
            print("  -> interior differs for some shifts; inspect before trusting any "
                  "windowed scheme (try --prefix-mode, which never shifts the left edge).")
    return results


def _make_latents(torch, model, args, device, dtype):
    """Latents to probe with, and where they came from.

    ``randn`` latents are far off the encoder's manifold; the decoder's response
    to them tends to sit on the +/-1 clamp, which makes ``max|diff|`` a poor
    convergence metric.  ``encode`` runs a synthetic but *plausible* waveform
    through the real encoder first, so the margin is measured on latents the
    decoder was actually trained to see.
    """
    total = args.latents
    if args.signal == "randn":
        torch.manual_seed(args.seed)
        z = torch.randn(1, 32, 2, total, device=device, dtype=dtype)
        return z, {"source": "randn", "note": "off-manifold; expect clipping"}

    spl = int(getattr(model, "samples_per_latent", getattr(model, "hop_length", 800)))
    rate = int(getattr(model, "sample_rate", 32000))
    n = total * spl
    t = torch.arange(n, device=device, dtype=torch.float32) / float(rate)
    torch.manual_seed(args.seed)
    noise = torch.randn(n, device=device, dtype=torch.float32) * 0.02
    left = 0.4 * torch.sin(2 * math.pi * 220.0 * t) + 0.2 * torch.sin(2 * math.pi * 440.0 * t)
    right = 0.35 * torch.sin(2 * math.pi * 330.0 * t) + 0.15 * torch.sin(2 * math.pi * 660.0 * t)
    # a slow amplitude envelope so the signal is not stationary
    env = 0.6 + 0.4 * torch.sin(2 * math.pi * 0.7 * t)
    wave = torch.stack([(left + noise) * env, (right + noise) * env], dim=0).unsqueeze(0)
    wave = wave.clamp(-1.0, 1.0).to(dtype)
    with torch.no_grad():
        z = model.encode(wave)
    z = z[..., :total].to(dtype)
    got = int(z.shape[-1])
    return z, {
        "source": "encode",
        "waveform_samples": n,
        "latents_returned": got,
        "note": "on-manifold: synthetic waveform through the real encoder",
    }


def run_real(args) -> Dict[str, Any]:
    comfy_root = _add_comfy_root(args.comfy_root)
    vae_path = _resolve_vae(args.vae, comfy_root)

    import torch

    comfy = _import_comfy()

    print("comfy root : {}".format(comfy_root))
    print("vae        : {}".format(vae_path))
    print("device     : {}".format(args.device))

    flags = _pin_determinism(torch)
    print("determinism flags pinned: {}".format(flags))

    state_dict = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=state_dict)
    model = vae.first_stage_model
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = model.to(device=device, dtype=dtype).eval()
    print("dtype      : {} (model and latents unified)".format(args.dtype))

    geometry = AudioLatentGeometry.from_vae(model)
    print("geometry   : {} samples/latent @ {} Hz ({} latents/s)".format(
        geometry.samples_per_latent, geometry.sample_rate, geometry.latents_per_second))

    z, signal_info = _make_latents(torch, model, args, device, dtype)
    args.latents = int(z.shape[-1])
    print("signal     : {} ({})".format(signal_info["source"], signal_info["note"]))

    def decode_fn(chunk):
        with torch.no_grad():
            return model.decode(chunk)

    cat = lambda parts, dim: torch.cat(list(parts), dim=dim)  # noqa: E731

    out: Dict[str, Any] = {
        "mode": "real",
        "comfy_root": comfy_root,
        "vae": vae_path,
        "device": args.device,
        "dtype": args.dtype,
        "determinism_flags": flags,
        "latents": args.latents,
        "samples_per_latent": geometry.samples_per_latent,
        "sample_rate": geometry.sample_rate,
        "phase_align": args.phase_align,
        "prefix_mode": bool(args.prefix_mode),
        "signal": signal_info,
        "blocks": [],
    }

    print("reference  : full decode of {} latents ({:.3f} s of audio)".format(
        args.latents, float(geometry.latents_to_seconds(args.latents))))
    reference = decode_fn(z)
    print("             waveform shape {}".format(tuple(reference.shape)))

    sat = saturation_stats(reference)
    out["reference_saturation"] = sat
    print("             peak={:.4f} rms={:.4f} at-rail={:.2%} of samples".format(
        sat["peak_abs"], sat["rms"], sat["fraction_at_rail"]))
    if sat["is_saturated"]:
        advice = ("Re-run with --signal encode." if args.signal != "encode" else
                  "Even the encoded signal rails: lower the synthetic waveform's level "
                  "or check the latent scaling before reading any margin off this.")
        print("             !! the reference is CLIPPED. max|diff| on a railed signal "
              "is not a smooth function of context length, so a plateau here may be "
              "the clamp, not the receptive field. " + advice)

    # 1. is the decoder reproducible at all?
    print()
    out["determinism"] = _abba_determinism(torch, decode_fn, z, _blocks(args)[0])

    # 2. is the planner/slicing itself exact?
    print()
    out["full_sequence_control"] = _full_sequence_control(
        decode_fn, z, reference, _blocks(args)[0], geometry, cat)

    # 3. is the decoder phase dependent?
    print()
    out["shift_equivariance"] = _report_phase(decode_fn, z, geometry, _blocks(args)[0])

    all_ok = True
    for block in _blocks(args):
        print("\n" + "-" * 70)
        print("--- block_latents = {} ({:.3f} s per block) ---".format(
            block, float(geometry.latents_to_seconds(block))))

        naive = torch.cat(
            [decode_fn(z[..., i:i + block]) for i in range(0, args.latents, block)], dim=-1
        )
        naive_err = max_abs_diff(naive, reference)
        print("no-context : max|diff| = {:.3e} {}".format(
            naive_err,
            "(decoder behaves causally at this tolerance!)" if naive_err <= args.tol
            else "-> decoder is non-causal, as expected"))

        def on_step(margin, err):
            print("  probing margin {:<4d} -> max|diff| = {:.3e}".format(margin, err))

        result = search_latent_margin(
            decode_fn, z, reference, tolerance=args.tol, block_latents=block,
            max_margin=args.max_margin, geometry=geometry, concat=cat,
            on_step=on_step, phase_align=args.phase_align,
            prefix_mode=bool(args.prefix_mode),
        )

        ok = _verdict(result, geometry, block, Fraction(17, 24))
        entry = _result_dict(result, geometry, block)
        entry["no_context_max_abs_diff"] = naive_err
        entry["decoder_is_non_causal"] = bool(naive_err > args.tol)
        entry["is_monotone"] = result.is_monotone
        entry["full_sequence_error"] = result.full_sequence_error
        entry["full_sequence_is_exact"] = result.full_sequence_is_exact

        report_margin = args.report_margin
        if report_margin is None:
            report_margin = result.margin if result.found else min(args.max_margin,
                                                                   args.latents)
        print()
        entry["block_diffs"] = _report_blocks(
            decode_fn, z, reference, report_margin, block, geometry,
            args.phase_align, bool(args.prefix_mode))
        entry["report_margin"] = report_margin

        entry["passed"] = bool(ok)
        if result.found:
            planner = OverlapSavePlanner(result.margin, block, geometry,
                                         phase_align=args.phase_align,
                                         prefix_mode=bool(args.prefix_mode))
            entry["required_history_latents"] = planner.required_history_latents()
            print("\nplanner    : block={} latents, margin={}, lookahead={:.4f} s, "
                  "history={} latents".format(
                      block, result.margin, float(planner.lookahead_seconds),
                      planner.required_history_latents()))
        out["blocks"].append(entry)
        all_ok = all_ok and ok

    _compare_blocks(out["blocks"])
    _diagnose(out)
    out["passed"] = all_ok
    print("\nRESULT: {}".format("PASSED" if all_ok else "FAILED"))
    return out


def _diagnose(out: Dict[str, Any]) -> None:
    """Say what the evidence implies, without touching the tolerance."""
    print("\n=== diagnosis ===")
    det = out.get("determinism") or {}
    control = out.get("full_sequence_control") or {}
    sat = out.get("reference_saturation") or {}
    blocks = out.get("blocks") or []
    notes: List[str] = []

    if sat.get("is_saturated"):
        notes.append(
            "The reference waveform is clipped ({:.1%} of samples on the +/-1 rail). "
            "The decoder's clamp both hides and creates differences depending on where "
            "the signal rails, so max|diff| is not a smooth function of the margin here. "
            "{}".format(
                sat.get("fraction_at_rail", 0.0),
                "Re-run with --signal encode before trusting any plateau."
                if out.get("signal", {}).get("source") != "encode" else
                "This is already the encoded signal, so the clipping is real: "
                "reduce the probe waveform's level before reading any margin."))
    if det and not det.get("deterministic", True):
        notes.append(
            "The decoder is not reproducible run to run even on an identical tensor "
            "(full-vs-full max|diff|={:.3e}). Every margin number below is noise at or "
            "above that level. Fix determinism first.".format(
                det.get("full_vs_full", {}).get("max_abs_diff", float("nan"))))
    if control and not control.get("exact", True):
        notes.append(
            "The full-sequence control is non-zero (max|diff|={:.3e}). No windowing "
            "happens in that configuration, so this is NOT an overlap-save or "
            "receptive-field result - it is nondeterminism or a harness bug.".format(
                control.get("max_abs_diff", float("nan"))))
    elif control:
        notes.append(
            "The full-sequence control is exact, so the planner, the slicing and the "
            "reference all agree; the residual is genuinely about context or about "
            "shape-keyed kernel selection.")

    non_monotone = [b for b in blocks if b.get("is_monotone") is False]
    unconverged = [b for b in blocks if not b.get("found")]
    if unconverged and non_monotone and control.get("exact"):
        notes.append(
            "Margins plateau and wobble while the control is exact: that is the "
            "fingerprint of convolution algorithms being chosen per input SHAPE "
            "(cuDNN autotune). With benchmark/TF32 already pinned off, the remaining "
            "options are --prefix-mode (left edge is always the true stream start, so "
            "only the right margin matters) or accepting a fixed decode length.")
    if unconverged and not non_monotone and control.get("exact"):
        notes.append(
            "Margins fall monotonically but have not reached the tolerance within "
            "--max-margin: this looks like a genuinely large receptive field. Re-run "
            "with a bigger --max-margin to measure it rather than widening --tol.")

    phase = out.get("shift_equivariance") or []
    reproducible = det.get("deterministic", True)
    if phase and all(r.get("exact") for r in phase):
        notes.append(
            "The decoder is shift-equivariant, so slice phase is not the problem and "
            "phase_align would not help.")
    elif phase and not reproducible:
        notes.append(
            "The shift-equivariance probe also disagrees, but it cannot mean anything "
            "while the decoder is nondeterministic: its differences are at the same "
            "magnitude as the run-to-run noise. Re-run it after determinism is fixed "
            "before concluding anything about phase.")
    elif phase:
        bad = [r["shift_latents"] for r in phase if not r.get("exact")]
        exact_shifts = [r["shift_latents"] for r in phase if r.get("exact")]
        notes.append(
            "Interior output changes for shifts {} (a fixed-size window was slid, so "
            "this is phase, not length). There IS a phase dependence: set "
            "--phase-align to {}, or use --prefix-mode.".format(
                bad, min(exact_shifts) if exact_shifts else "the smallest exact shift"))

    for note in notes:
        print("  * {}".format(note))
    out["diagnosis"] = notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comfy-root", default=None,
                        help="ComfyUI checkout (also used to resolve --vae); falls back "
                             "to COMFYUI_PATH / COMFYUI_UPSTREAM_PATH")
    parser.add_argument("--vae", default=None,
                        help="audio VAE checkpoint path, or a filename under models/vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latents", type=int, default=400,
                        help="latent frames to test (40 per second of audio)")
    parser.add_argument("--block", default=DEFAULT_BLOCKS,
                        help="comma separated latents-per-block sizes to test "
                             "(default: {}; co-prime so a block-grid artefact would "
                             "show up as disagreement)".format(DEFAULT_BLOCKS))
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-margin", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal", default="encode", choices=["encode", "randn"],
                        help="where the probe latents come from: 'encode' runs a "
                             "synthetic waveform through the real encoder (on-manifold, "
                             "default), 'randn' is off-manifold and tends to clip")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="model and latents are unified to this dtype")
    parser.add_argument("--phase-align", type=int, default=1,
                        help="round every decode's latent_start down to a multiple of "
                             "this (only ever adds left context); use if the "
                             "shift-equivariance probe reports a phase dependence")
    parser.add_argument("--prefix-mode", action="store_true",
                        help="decode z[0:hi] for every block instead of a window, so "
                             "the left edge is always the true stream start "
                             "(exact left context by construction, O(n^2) work)")
    parser.add_argument("--report-margin", type=int, default=None,
                        help="margin used for the per-block diff report "
                             "(default: the found margin, else max-margin)")
    parser.add_argument("--fake", action="store_true",
                        help="run against a numpy decoder with a known receptive field")
    parser.add_argument("--fake-radius", type=int, default=6)
    parser.add_argument("--json", default=None, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    print("probe_audio_overlap: measuring the audio VAE's required lookahead")
    print("interpreter: {}".format(sys.executable))

    if not args.fake and not args.vae:
        parser.error("--vae is required (or use --fake)")

    payload: Dict[str, Any] = {
        "tool": "probe_audio_overlap",
        "interpreter": sys.executable,
        "args": vars(args),
    }

    def finish(code: int) -> int:
        payload["exit_code"] = code
        payload["passed"] = code == EXIT_OK
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(payload, handle, indent=2, default=str)
            print("\nwrote JSON report to {}".format(args.json))
        return code

    try:
        result = run_fake(args) if args.fake else run_real(args)
    except ImportError as exc:
        print("\nENVIRONMENT ERROR: {}".format(exc))
        payload["error"] = str(exc)
        return finish(EXIT_ENVIRONMENT)

    payload["result"] = result
    return finish(EXIT_OK if result.get("passed") else EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
