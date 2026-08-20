#!/usr/bin/env python3
"""M0 probe 2: does the incremental video decoder equal the official full decode?

Runs :class:`IncrementalVideoDecoder` against a *real* MiniMax H3 video VAE and
compares, tensor for tensor, with ``VAE.decode`` / ``decode_temporal`` on the
same latents.  The target is exact equality (max|diff| == 0); anything up to
``--tol`` (default 1e-6) is still reported as a pass, anything above is a
falsification of the streaming design.

It also measures what the streaming machine actually buys: peak resident frames
versus the full output tensor, and the latent lookahead (2 latents) before the
first 17 frames can be emitted.

No model path is hardcoded.  Point it at any ComfyUI checkout and any VAE
checkpoint:

    python tools/probe_video_vae_streaming.py \\
        --comfy-root /workspace/ComfyUI \\
        --vae /workspace/ComfyUI/models/vae/minimax_h3_video_vae.safetensors \\
        --device cuda --latents 22 --height 32 --width 32

``--vae`` may be an absolute path, or a bare filename resolved through the
ComfyUI checkout's ``models/vae`` directory.  With ``--fake`` the probe runs the
whole comparison on a numpy fake, which is useful to sanity check the harness
itself on a machine with no GPU and no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.media.video_stream import (  # noqa: E402
    MIN_SUPPORTED_FRAMES,
    MIN_SUPPORTED_LATENTS,
    IncrementalVideoDecoder,
    ShortVideoStreamError,
    VideoChunkParams,
    reference_decode_temporal,
    summarize_decode_comparison,
)

#: Latent counts on the encoder's natural 5k+2 grid (k = 1, 4, 11, 21), i.e.
#: 22, 73, 192 and 362 output frames.  k=0 (2 latents / 5 frames) is not a
#: supported clip length in v0.1 and is rejected rather than measured.
DEFAULT_LATENT_GRID = "7,22,57,107"

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


# -- fake mode -------------------------------------------------------------


def run_fake(args) -> Dict[str, Any]:
    import numpy as np

    from raven_streaming.media.fakes import (
        FakeVideoChunkDecoder,
        make_latents,
        numpy_concat,
    )

    params = VideoChunkParams()
    params.check_stream_length(args.latents)
    print("mode: FAKE (numpy) - harness self-check, not a model result")
    z = make_latents(args.latents, channels=24, h=args.height // 16 or 1, w=args.width // 16 or 1)

    ref_parts = reference_decode_temporal(FakeVideoChunkDecoder(), z, params, concat=numpy_concat)
    ref = np.concatenate(ref_parts, axis=2)

    stream = IncrementalVideoDecoder(FakeVideoChunkDecoder(), params, concat=numpy_concat)
    got_parts: List = []
    for i in range(0, z.shape[2], args.push):
        for batch in stream.push(z[:, :, i:i + args.push]):
            got_parts.append(batch.frames)
    for batch in stream.finish():
        got_parts.append(batch.frames)
    got = np.concatenate(got_parts, axis=2)

    diff = float(np.abs(got - ref).max()) if got.size else 0.0
    expected = params.total_frames(args.latents)
    ok = bool(got.shape == ref.shape and diff <= args.tol and got.shape[2] == expected)
    print("  latents={} frames={} (expected {})".format(args.latents, got.shape[2], expected))
    print("  max|diff| = {:.3e}".format(diff))
    print("\nRESULT: {}".format("PASSED" if ok else "FAILED"))
    return {
        "mode": "fake",
        "latents": args.latents,
        "frames": int(got.shape[2]),
        "expected_frames": expected,
        "max_abs_diff": diff,
        "exact": diff == 0.0,
        "tolerance": args.tol,
        "passed": ok,
    }


# -- real mode -------------------------------------------------------------


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


def _build_incremental(torch, model, params, args):
    """Fresh coordinator wired to the real VAE's operators."""
    from raven_streaming.media.video_stream import minimax_decoder_adapter

    adapter = minimax_decoder_adapter(model)
    return adapter, IncrementalVideoDecoder(
        adapter,
        params,
        concat=lambda parts, dim: torch.cat(list(parts), dim=dim),
        denormalize=adapter.denormalize,
    )


def _decode_incremental(torch, model, params, z, args, stats=None):
    """Run the streaming coordinator over ``z`` and concatenate the batches."""
    _, stream = _build_incremental(torch, model, params, args)
    parts: List = []
    peak_batch_frames = 0
    first_output_after = None
    with torch.no_grad():
        for i in range(0, z.shape[2], args.push):
            for batch in stream.push(z[:, :, i:i + args.push]):
                if first_output_after is None:
                    first_output_after = stream.latents_seen
                peak_batch_frames = max(peak_batch_frames, batch.count)
                parts.append(batch.frames.to("cpu") if args.offload else batch.frames)
        for batch in stream.finish():
            if first_output_after is None:
                first_output_after = stream.latents_seen
            peak_batch_frames = max(peak_batch_frames, batch.count)
            parts.append(batch.frames.to("cpu") if args.offload else batch.frames)
    if stats is not None:
        stats["peak_batch_frames"] = peak_batch_frames
        stats["first_output_after_latents"] = first_output_after
    return torch.cat(parts, dim=2)


def _diff(torch, a, b) -> Dict[str, Any]:
    """max/mean |a - b| plus bitwise equality, without upcasting the whole tensor."""
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_mismatch": [list(a.shape), list(b.shape)],
                "max_abs_diff": None, "mean_abs_diff": None, "exact": False}
    exact = bool(torch.equal(a, b))
    d = (a.to(torch.float32) - b.to(torch.float32)).abs_()
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "exact": exact,
    }


def _warmup(torch, model, params, args, latent_h, latent_w, dtype, device) -> None:
    """Burn cuDNN autotune / lazy-init on the *exact* per-chunk shape.

    Every chunk decodes ``tokens_chunk_size + token_overlap`` = 7 latents, so a
    7-latent warmup exercises the same kernels, the same tile split and the same
    workspace as the measured runs.  Nothing here is compared or timed.
    """
    if args.warmup <= 0:
        return
    print("warmup     : {} pass(es) on the {}-latent chunk shape "
          "(not compared, not timed)".format(args.warmup, params.latents_needed))
    warm = torch.randn(1, 24, params.latents_needed, latent_h, latent_w,
                       device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(args.warmup):
            model.decode(warm)
            _decode_incremental(torch, model, params, warm, args)
    _sync(torch, device)
    del warm
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_real(args) -> Dict[str, Any]:
    comfy_root = _add_comfy_root(args.comfy_root)
    vae_path = _resolve_vae(args.vae, comfy_root)

    import torch  # noqa: F401

    comfy = _import_comfy()

    print("comfy root : {}".format(comfy_root))
    print("vae        : {}".format(vae_path))
    print("device     : {}".format(args.device))

    state_dict = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=state_dict)
    model = vae.first_stage_model
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    params = VideoChunkParams.from_vae(model)
    print("geometry   : tokens_chunk_size={} token_overlap={} frame_pre_padding={} "
          "frame_overlap={} (lookahead {} latents)".format(
              params.tokens_chunk_size, params.token_overlap, params.frame_pre_padding,
              params.frame_overlap, params.lookahead_latents))

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    latent_h = max(1, args.height // 16)
    latent_w = max(1, args.width // 16)
    z = torch.randn(1, 24, args.latents, latent_h, latent_w, device=device, dtype=dtype)

    params.check_stream_length(args.latents)
    expected_frames = params.total_frames(args.latents)
    pad_tokens, num_chunks = params.temporal_chunks(args.latents)
    truncates = pad_tokens > 0
    print("input      : latents={} [{}x{}] -> expected {} frames "
          "(pad_tokens={}, chunks={}, output {} truncated)".format(
              args.latents, latent_h, latent_w, expected_frames, pad_tokens, num_chunks,
              "IS" if truncates else "is not"))

    _warmup(torch, model, params, args, latent_h, latent_w, dtype, device)

    # ABBA: full, incremental, incremental, full.  Comparing pass 1 of each
    # against pass 2 of each separates a genuine algorithmic difference from a
    # cold-kernel one, and full-vs-full answers "is the official decoder even
    # bitwise reproducible on this box?" - which no tolerance change could.
    schedule = ["full", "incr", "incr", "full"] if args.abba else ["full", "incr"]
    outputs: Dict[str, List] = {"full": [], "incr": []}
    timings: Dict[str, List[float]] = {"full": [], "incr": []}
    stream_stats: Dict[str, Any] = {}

    for step, kind in enumerate(schedule):
        t0 = time.time()
        if kind == "full":
            with torch.no_grad():
                out = model.decode(z)
        else:
            out = _decode_incremental(torch, model, params, z, args, stream_stats)
        _sync(torch, device)
        elapsed = time.time() - t0
        timings[kind].append(elapsed)
        outputs[kind].append(out.detach().to("cpu"))
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("  pass {} {:<4} shape={} in {:.2f}s".format(
            step + 1, kind, tuple(outputs[kind][-1].shape), elapsed))

    full_1, incr_1 = outputs["full"][0], outputs["incr"][0]
    full_n, incr_n = outputs["full"][-1], outputs["incr"][-1]

    result: Dict[str, Any] = {
        "mode": "real",
        "comfy_root": comfy_root,
        "vae": vae_path,
        "device": args.device,
        "dtype": args.dtype,
        "latents": args.latents,
        "expected_frames": expected_frames,
        "pad_tokens": pad_tokens,
        "num_chunks": num_chunks,
        "output_is_truncated": truncates,
        "official_shape": list(full_1.shape),
        "incremental_shape": list(incr_1.shape),
        "warmup_passes": args.warmup,
        "abba": bool(args.abba),
        "schedule": schedule,
        "official_seconds": timings["full"],
        "incremental_seconds": timings["incr"],
        "first_output_after_latents": stream_stats.get("first_output_after_latents"),
        "peak_batch_frames": stream_stats.get("peak_batch_frames"),
        "lookahead_latents": params.lookahead_latents,
        "tolerance": args.tol,
    }

    peak = stream_stats.get("peak_batch_frames") or 1
    result["resident_frame_ratio"] = float(full_1.shape[2]) / max(1, peak)
    print("             first frames emitted after {} latents "
          "(chunk {} + lookahead {})".format(
              result["first_output_after_latents"], params.tokens_chunk_size,
              params.lookahead_latents))
    print("             peak frames in one batch: {} (full tensor holds {})"
          " -> {:.1f}x fewer resident frames per step".format(
              peak, full_1.shape[2], result["resident_frame_ratio"]))

    if full_1.shape[2] != expected_frames:
        print("  NOTE: official decode returned {} frames, plan predicted {}".format(
            full_1.shape[2], expected_frames))

    # -- comparisons ---------------------------------------------------------
    cold = _diff(torch, incr_1, full_1)
    warm = _diff(torch, incr_n, full_n)
    official_self = _diff(torch, full_1, full_n) if len(outputs["full"]) > 1 else None
    incr_self = _diff(torch, incr_1, incr_n) if len(outputs["incr"]) > 1 else None

    result.update({
        "compare_cold": cold,
        "compare_warm": warm,
        "official_self_consistency": official_self,
        "incremental_self_consistency": incr_self,
    })

    print("\ncomparison (tol={:g}, NOT relaxed):".format(args.tol))
    print("  incremental vs official, first pass  : exact={} max|diff|={}".format(
        cold["exact"], _fmt(cold["max_abs_diff"])))
    print("  incremental vs official, last pass   : exact={} max|diff|={}".format(
        warm["exact"], _fmt(warm["max_abs_diff"])))
    if official_self is not None:
        print("  official vs official (run-to-run)    : exact={} max|diff|={}".format(
            official_self["exact"], _fmt(official_self["max_abs_diff"])))
    if incr_self is not None:
        print("  incremental vs itself (run-to-run)   : exact={} max|diff|={}".format(
            incr_self["exact"], _fmt(incr_self["max_abs_diff"])))

    verdict = summarize_decode_comparison(
        cold, warm, official_self, incr_self, tolerance=args.tol
    )
    ok = verdict.passed
    if not ok and tuple(incr_n.shape) == tuple(full_n.shape):
        d = (incr_n.to(torch.float32) - full_n.to(torch.float32)).abs_()
        result["worst_frame"] = int(d.amax(dim=(0, 1, 3, 4)).argmax())
        print("  worst frame index: {}".format(result["worst_frame"]))

    for note in verdict.notes:
        print("  * {}".format(note))
    result["notes"] = verdict.notes
    result["cold_within_tolerance"] = verdict.cold_within_tolerance
    result["warm_within_tolerance"] = verdict.warm_within_tolerance
    result["official_reproducible"] = verdict.official_reproducible
    result["incremental_reproducible"] = verdict.incremental_reproducible
    result["passed"] = bool(ok)
    print("\nRESULT: {}".format("PASSED" if ok else "FAILED"))
    return result


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return "{:.3e}".format(value)


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comfy-root", default=None,
                        help="ComfyUI checkout (also used to resolve --vae); falls back "
                             "to COMFYUI_PATH / COMFYUI_UPSTREAM_PATH")
    parser.add_argument("--vae", default=None,
                        help="VAE checkpoint path, or a filename under <comfy-root>/models/vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--latents", type=int, default=22,
                        help="latent frame count; the encoder's natural grid is 5k+2, "
                             "minimum {} (= {} frames)".format(
                                 MIN_SUPPORTED_LATENTS, MIN_SUPPORTED_FRAMES))
    parser.add_argument("--height", type=int, default=256, help="pixel height (16x latent)")
    parser.add_argument("--width", type=int, default=256, help="pixel width (16x latent)")
    parser.add_argument("--push", type=int, default=1,
                        help="latents delivered per push() call")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", action="store_true",
                        help="move each emitted batch to CPU (what the real lane does)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup passes on the 7-latent chunk shape before any "
                             "comparison, to burn cuDNN autotune / lazy init (default 1)")
    parser.add_argument("--abba", dest="abba", action="store_true", default=True,
                        help="run full/incr/incr/full and report run-to-run "
                             "reproducibility of both paths (default on)")
    parser.add_argument("--no-abba", dest="abba", action="store_false",
                        help="single full + single incremental pass only")
    parser.add_argument("--fake", action="store_true",
                        help="run on the numpy fake decoder (harness self-check)")
    parser.add_argument("--grid", default=DEFAULT_LATENT_GRID,
                        help="comma separated latent counts to sweep "
                             "(default: {} = the 5k+2 grid)".format(DEFAULT_LATENT_GRID))
    parser.add_argument("--json", default=None, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    print("probe_video_vae_streaming: incremental vs official full decode")
    print("interpreter: {}".format(sys.executable))

    counts = [int(x) for x in args.grid.split(",") if x.strip()] if args.grid else [args.latents]

    # Reject an unsupported clip length before loading a model or touching a GPU.
    reference_params = VideoChunkParams()
    unsupported = []
    for count in counts:
        try:
            reference_params.check_stream_length(count)
        except ShortVideoStreamError as exc:
            unsupported.append((count, str(exc)))
    if unsupported:
        for count, message in unsupported:
            print("\nUNSUPPORTED --grid entry {}: {}".format(count, message))
        parser.error(
            "grid contains {} unsupported clip length(s); the minimum is {} latents "
            "/ {} frames (k=1)".format(
                len(unsupported), MIN_SUPPORTED_LATENTS, MIN_SUPPORTED_FRAMES))

    payload: Dict[str, Any] = {
        "tool": "probe_video_vae_streaming",
        "interpreter": sys.executable,
        "args": vars(args),
        "grid": counts,
        "runs": [],
    }

    def finish(code: int) -> int:
        payload["exit_code"] = code
        payload["passed"] = code == EXIT_OK
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(payload, handle, indent=2, default=str)
            print("\nwrote JSON report to {}".format(args.json))
        print("\nOVERALL: {} ({}/{} latent counts passed)".format(
            "PASSED" if code == EXIT_OK else "FAILED",
            sum(1 for r in payload["runs"] if r.get("passed")), len(payload["runs"])))
        return code

    if not args.fake and not args.vae:
        parser.error("--vae is required (or use --fake)")

    try:
        for count in counts:
            args.latents = count
            print("\n" + "=" * 70)
            payload["runs"].append(run_fake(args) if args.fake else run_real(args))
    except ImportError as exc:
        print("\nENVIRONMENT ERROR: {}".format(exc))
        payload["error"] = str(exc)
        return finish(EXIT_ENVIRONMENT)

    ok = all(r.get("passed") for r in payload["runs"])
    return finish(EXIT_OK if ok else EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
