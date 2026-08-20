"""Smoke test for ``tools/probe_causal_parity.py`` in its tiny CPU mode.

The probe is the artefact M2's parity claim is made with, so it has to actually
run in the environment the claim is made from. This test runs it end to end on
the smallest legal request and checks that every check passed and that the JSON
report carries them.

Requires a local ComfyUI checkout (see ``tests/conftest.py``).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conftest  # noqa: E402

_UPSTREAM = conftest.find_upstream_comfyui()
if _UPSTREAM is None:  # pragma: no cover - environment without a checkout
    pytest.skip("No local ComfyUI checkout found", allow_module_level=True)


def _probe():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import probe_causal_parity

    return probe_causal_parity


def test_tiny_mode_passes_every_check(tmp_path):
    probe = _probe()
    report_path = tmp_path / "parity.json"
    code = probe.main([
        "--mode", "tiny", "--frames", "22", "--width", "64", "--height", "64",
        "--text-len", "3", "--sink", "1", "--window", "1",
        "--json", str(report_path),
    ])
    assert code == 0

    report = json.loads(report_path.read_text())
    assert report["mode"] == "tiny"
    assert report["passed"] is True
    names = {check["name"] for check in report["checks"]}
    assert {
        "state_dict.keys",
        "state_dict.shapes_dtypes",
        "dense_forward.bit_identical",
        "layout.positions_match_official",
        "attention.key_rows_match_cache",
        "attention.key_rows_cover_every_call",
        # the DiT blocks and the token refiner share one attention seam; these
        # pin that the tap told them apart instead of renumbering the blocks
        "attention.tap_labels",
        "attention.tap_layer_order",
        "attention.dit_layers_complete",
        "attention.refiner_calls_separated",
        "cache.history_is_visible",
        "cache.evicted_history_is_invisible",
        "timestep.clean_maps_to_0999",
    } <= names
    assert all(check["passed"] for check in report["checks"])

    by_name = {check["name"]: check for check in report["checks"]}
    refiner = by_name["attention.refiner_calls_separated"]["detail"]
    assert refiner["forwards"] == ["text:refiner"]
    assert refiner["in_dit_entries"] == 0
    assert refiner["calls"] == refiner["expected_calls"]
    # every DiT call was gated on its merged key count, none dropped
    cover = by_name["attention.key_rows_cover_every_call"]["detail"]
    assert cover["gated"] == cover["recorded"] > 0


def test_window_minus_one_means_no_eviction():
    probe = _probe()
    args = probe.parse_args(["--mode", "tiny", "--window", "-1"])
    assert args.window is None
