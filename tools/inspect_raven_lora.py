#!/usr/bin/env python3
"""Inspect a published RAVEN / MiniMax-H3 PEFT LoRA without downloading it.

Reads only the safetensors header (8 byte length + JSON) - locally, or from a
Hugging Face repo / arbitrary URL through HTTP range requests - and reports how
the keys map onto the official full non-pruned Comfy model
(``diffusion_model.*``): coverage per category (core / adaln / time / boundary),
rank, alpha, dtypes, byte size, plus every unexpected / missing / duplicate key.

Examples::

    python tools/inspect_raven_lora.py --file /models/loras/raven.safetensors
    python tools/inspect_raven_lora.py --hf MiniMaxAI/Some-Repo --hf-file adapter_model.safetensors
    python tools/inspect_raven_lora.py --url https://.../adapter_model.safetensors --json out.json

With ``--base <comfy model .safetensors>`` the base checkpoint header is read
too (again header-only) to verify that every mapped key exists there and that
the checkpoint is not the pruned / adaln-curve form.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raven_streaming import lora as rlora  # noqa: E402

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


# --------------------------------------------------------------------------
# remote header fetch (range requests only)
# --------------------------------------------------------------------------
def _http_range(url: str, start: int, end: int, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes={}-{}".format(start, end))
    req.add_header("User-Agent", "raven-streaming-lora-inspect/0")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "huggingface" in url:
        req.add_header("Authorization", "Bearer {}".format(token))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) > end - start + 1:  # server ignored the Range header
        data = data[: end - start + 1]
    return data


def fetch_remote_header(url: str, timeout: float = 30.0) -> Tuple[rlora.SafetensorsHeader, int]:
    prefix = _http_range(url, 0, 7, timeout)
    n = rlora.safetensors_header_length(prefix)
    body = _http_range(url, 8, 8 + n - 1, timeout)
    return rlora.parse_safetensors_header(prefix + body), 8 + n


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return "{}/{}/resolve/{}/{}".format(HF_ENDPOINT.rstrip("/"), repo, revision, filename)


def read_header(args) -> Tuple[rlora.SafetensorsHeader, str, int]:
    if args.file:
        header = rlora.read_safetensors_header(args.file)
        return header, args.file, header.data_offset
    url = args.url or hf_url(args.hf, args.hf_file, args.revision)
    header, offset = fetch_remote_header(url, args.timeout)
    return header, url, offset


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def diagnose(
    header: rlora.SafetensorsHeader,
    config: rlora.RavenBaseConfig,
    prefix: str = rlora.PEFT_PREFIX,
) -> Dict[str, Any]:
    """Tolerant pass over the header: never raises, reports everything."""
    inventory = config.modules()
    covered: Dict[str, Dict[str, rlora.TensorInfo]] = {}
    unexpected: List[str] = []
    unknown: List[str] = []
    duplicates: List[str] = []
    dtypes = Counter()
    ranks = Counter()
    total_bytes = 0

    for name in sorted(header.tensors):
        info = header.tensors[name]
        total_bytes += info.nbytes
        dtypes[info.dtype] += 1
        try:
            path, side, _adapter = rlora.parse_peft_key(name, prefix=prefix)
        except rlora.UnexpectedKeyError:
            unexpected.append(name)
            continue
        if path not in inventory:
            unknown.append(name)
            continue
        slot = covered.setdefault(path, {})
        if side in slot:
            duplicates.append("{} vs {}".format(slot[side].name, name))
            continue
        slot[side] = info
        if side == "A" and len(info.shape) == 2:
            ranks[int(info.shape[0])] += 1

    complete = {p: s for p, s in covered.items() if len(s) == 2}
    shape_errors: List[str] = []
    for path, slot in sorted(complete.items()):
        entry = inventory[path]
        a, b = slot["A"], slot["B"]
        if len(a.shape) != 2 or len(b.shape) != 2:
            shape_errors.append("{}: non-2D A{} B{}".format(path, a.shape, b.shape))
            continue
        if a.shape[0] != b.shape[1]:
            shape_errors.append("{}: rank A={} B={}".format(path, a.shape[0], b.shape[1]))
        if a.shape[1] != entry.in_features or b.shape[0] != entry.out_features:
            shape_errors.append(
                "{}: A{} B{} vs base weight {}".format(
                    path, tuple(a.shape), tuple(b.shape), entry.weight_shape)
            )

    per_cat_total = Counter(e.category for e in inventory.values())
    per_cat_covered = Counter(inventory[p].category for p in complete)
    missing = [p for p in inventory if p not in complete]

    return {
        "prefix": prefix,
        "tensors": len(header.tensors),
        "total_data_bytes": total_bytes,
        "dtypes": dict(dtypes),
        "metadata": dict(header.metadata),
        "modules_expected": len(inventory),
        "modules_covered": len(complete),
        "coverage_by_category": {
            c: {"covered": per_cat_covered.get(c, 0), "expected": per_cat_total.get(c, 0)}
            for c in rlora.CATEGORY_ORDER
        },
        "rank_histogram": dict(ranks),
        "half_pairs": sorted(p for p, s in covered.items() if len(s) != 2),
        "missing_modules": missing,
        "unexpected_keys": unexpected,
        "unknown_modules": unknown,
        "duplicates": duplicates,
        "shape_errors": shape_errors,
        "base_keys_sample": [inventory[p].base_key for p in list(complete)[:3]],
    }


def check_base_file(base_path: str, mapped_paths: List[str], prefixes: List[str]) -> Dict[str, Any]:
    """Header-only validation against a base checkpoint (no tensor read)."""
    header = rlora.read_safetensors_header(base_path)
    keys = set(header.tensors)
    pruned = [k for k in keys if k.endswith("adaln_t_table")]
    chosen, missing = None, None
    for pref in prefixes:
        want = ["{}{}.weight".format(pref, p) for p in mapped_paths]
        absent = [k for k in want if k not in keys]
        if chosen is None or len(absent) < len(missing or []):
            chosen, missing = pref, absent
        if not absent:
            break
    return {
        "path": base_path,
        "tensors": len(keys),
        "prefix": chosen,
        "missing_base_keys": (missing or [])[:16],
        "missing_base_key_count": len(missing or []),
        "adaln_t_table": pruned,
        "pruned": bool(pruned),
    }


# --------------------------------------------------------------------------
def render(diag: Dict[str, Any], strict: Optional[str], base: Optional[Dict[str, Any]]) -> str:
    lines = []
    lines.append("tensors: {}  ({} bytes of tensor data)".format(
        diag["tensors"], diag["total_data_bytes"]))
    lines.append("dtypes: {}".format(diag["dtypes"]))
    lines.append("metadata: {}".format(diag["metadata"] or "{}"))
    lines.append("modules covered: {}/{}".format(diag["modules_covered"], diag["modules_expected"]))
    for cat, v in diag["coverage_by_category"].items():
        lines.append("  {:<9} {}/{}".format(cat, v["covered"], v["expected"]))
    lines.append("rank histogram: {}".format(diag["rank_histogram"]))
    for label, key in (
        ("unexpected keys", "unexpected_keys"),
        ("unknown modules", "unknown_modules"),
        ("incomplete pairs", "half_pairs"),
        ("duplicates", "duplicates"),
        ("shape errors", "shape_errors"),
        ("missing modules", "missing_modules"),
    ):
        items = diag[key]
        if items:
            lines.append("{}: {} -> {}".format(label, len(items), items[:6]))
    lines.append("example base keys: {}".format(diag["base_keys_sample"]))
    if base is not None:
        lines.append("base checkpoint {}: {} tensors, prefix={!r}, pruned={}".format(
            base["path"], base["tensors"], base["prefix"], base["pruned"]))
        if base["adaln_t_table"]:
            lines.append("  REJECT: adaln_t_table present -> pruned / adaln-curve form")
        if base["missing_base_key_count"]:
            lines.append("  missing {} base keys, e.g. {}".format(
                base["missing_base_key_count"], base["missing_base_keys"][:4]))
    lines.append("strict mapping: {}".format(strict or "OK"))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="local safetensors file")
    src.add_argument("--hf", help="Hugging Face repo id, e.g. org/name")
    src.add_argument("--url", help="direct URL to a safetensors file")
    ap.add_argument("--hf-file", default="adapter_model.safetensors")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--prefix", default=rlora.PEFT_PREFIX)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--base", default=None, help="base model safetensors (header-only check)")
    ap.add_argument(
        "--base-prefix", action="append", default=None,
        help="candidate key prefix inside the base checkpoint (repeatable)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.hf and not args.hf_file:
        ap.error("--hf requires --hf-file")

    header, source, _offset = read_header(args)
    config = rlora.RavenBaseConfig()
    diag = diagnose(header, config, prefix=args.prefix)

    strict_error: Optional[str] = None
    manifest_info: Dict[str, Any] = {}
    try:
        manifest = rlora.build_manifest(
            header, config, alpha=args.alpha, prefix=args.prefix, source=source)
        manifest_info = {
            "modules": manifest.module_count,
            "tensors": manifest.tensor_count,
            "counts": manifest.counts,
            "rank": manifest.rank,
            "alpha": manifest.alpha,
            "adapters": list(manifest.adapter_names),
            "summary": manifest.summary(),
        }
    except rlora.RavenLoraError as exc:
        strict_error = "{}: {}".format(type(exc).__name__, exc)

    base_info = None
    if args.base:
        prefixes = args.base_prefix or ["", "diffusion_model.", "model.diffusion_model."]
        mapped = list(config.modules())
        base_info = check_base_file(args.base, mapped, prefixes)

    print("source: {}".format(source))
    print(render(diag, strict_error, base_info))
    if manifest_info:
        print("manifest: {}".format(manifest_info["summary"]))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"source": source, "diagnostic": diag, "manifest": manifest_info,
             "strict_error": strict_error, "base": base_info},
            indent=2))
    return 0 if strict_error is None and (base_info is None or not base_info["pruned"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
