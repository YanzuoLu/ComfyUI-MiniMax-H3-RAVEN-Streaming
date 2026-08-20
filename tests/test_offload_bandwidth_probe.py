"""Tests for tools/probe_offload_bandwidth.py.

No CUDA anywhere: every measurement scenario in the probe needs a device, so
what is testable off-GPU is the part that decides *what* gets measured and
*what the numbers mean* -- the KV byte arithmetic, the statistics, the rollout
bounds, the CLI surface, the allocation ceiling, and the report write. Those
are also the parts a wrong answer would be hardest to notice in a benchmark
log, which is why they are pinned here rather than eyeballed on the box.

The no-CUDA path of ``main()`` is exercised end to end: it must still produce a
report (skips named, byte counts intact, estimates from an assumed bandwidth)
and exit with the environment code rather than pretending to have measured
something.
"""

from __future__ import annotations

import argparse
import importlib.util
import contextlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "probe_offload_bandwidth", ROOT / "tools" / "probe_offload_bandwidth.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


tool = _load_tool()

GIB = 1024 ** 3
GB = 1000 ** 3


# --------------------------------------------------------------------------
# KV geometry
# --------------------------------------------------------------------------
def test_per_layer_bytes_is_k_plus_v_over_the_published_shape():
    """20996 x 56 x 128 BF16, K and V: the number the probe transfers."""
    geo = tool.KVGeometry(rows=20996, heads=56, head_dim=128, dtype="bf16", layers=50)
    assert geo.dtype_bytes == 2
    assert geo.elements_per_layer == 2 * 20996 * 56 * 128
    assert geo.per_layer_bytes == 2 * 20996 * 56 * 128 * 2 == 601_997_312
    # ~0.56 GiB per layer, ~28 GiB for all 50 -- the figure docs/validation.md
    # quotes as the reason the request does not fit a 24 GiB card.
    assert geo.per_layer_bytes / GIB == pytest.approx(0.5607, abs=5e-4)
    assert geo.all_layers_bytes == geo.per_layer_bytes * 50
    assert geo.all_layers_bytes / GIB == pytest.approx(28.03, abs=0.02)


def test_geometry_scales_with_dtype_and_layers():
    base = tool.KVGeometry(rows=1024, heads=8, head_dim=64, dtype="bf16", layers=4)
    fp32 = tool.KVGeometry(rows=1024, heads=8, head_dim=64, dtype="fp32", layers=4)
    assert fp32.per_layer_bytes == 2 * base.per_layer_bytes
    assert base.all_layers_bytes == 4 * base.per_layer_bytes
    assert tool.KVGeometry(rows=1024, heads=8, head_dim=64, dtype="fp16",
                           layers=4).per_layer_bytes == base.per_layer_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"rows": -1},
        {"heads": 0},
        {"head_dim": -8},
        {"layers": 0},
        {"dtype": "int4"},
    ],
)
def test_geometry_rejects_nonsense(kwargs):
    fields = dict(rows=16, heads=2, head_dim=8, dtype="bf16", layers=2)
    fields.update(kwargs)
    with pytest.raises(tool.ProbeError):
        tool.KVGeometry(**fields)


def test_geometry_dict_carries_both_bytes_and_gib():
    geo = tool.KVGeometry(rows=20996, heads=56, head_dim=128)
    payload = geo.to_dict()
    assert payload["per_layer_bytes"] == geo.per_layer_bytes
    assert payload["all_layers_gib"] == geo.all_layers_bytes / GIB
    assert payload["dtype"] == "bf16"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def test_percentile_interpolates_and_brackets():
    data = [1.0, 2.0, 3.0, 4.0]
    assert tool.percentile(data, 0.0) == 1.0
    assert tool.percentile(data, 1.0) == 4.0
    assert tool.percentile(data, 0.5) == pytest.approx(2.5)
    # position 0.95 * 3 = 2.85 -> 3.0 + 0.85 * 1.0
    assert tool.percentile(data, 0.95) == pytest.approx(3.85)
    assert tool.percentile([7.0], 0.5) == 7.0
    # unsorted input must not change the answer
    assert tool.percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_percentile_rejects_q_outside_unit_interval(bad):
    with pytest.raises(tool.ProbeError):
        tool.percentile([1.0, 2.0], bad)


def test_percentile_rejects_empty_sample():
    with pytest.raises(tool.ProbeError):
        tool.percentile([], 0.5)


def test_bandwidth_uses_decimal_and_binary_units_consistently():
    band = tool.bandwidth(GB, 1000.0)  # 1e9 bytes in one second
    assert band["gb_per_s"] == pytest.approx(1.0)
    assert band["gib_per_s"] == pytest.approx(GB / GIB)
    faster = tool.bandwidth(GB, 500.0)
    assert faster["gb_per_s"] == pytest.approx(2.0)


def test_bandwidth_refuses_a_zero_duration():
    with pytest.raises(tool.ProbeError):
        tool.bandwidth(1024, 0.0)


def test_describe_reports_the_tail_not_just_the_average():
    samples = [10.0, 10.0, 10.0, 10.0, 50.0]
    stats = tool.describe(samples)
    assert stats["samples"] == 5
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p50"] == 10.0
    assert stats["p95"] > stats["p50"]
    assert stats["mean"] == pytest.approx(18.0)


def test_summarize_transfer_quotes_p50_and_the_slower_p95():
    event_ms = [100.0, 100.0, 100.0, 200.0]
    wall_ms = [110.0, 112.0, 111.0, 215.0]
    entry = tool.summarize_transfer(GB, event_ms, wall_ms, note="hi")
    assert entry["bytes"] == GB
    assert entry["gib"] == pytest.approx(GB / GIB)
    assert entry["event_ms"]["p50"] == 100.0
    assert entry["bandwidth_p50"]["gb_per_s"] == pytest.approx(10.0)
    # the p95 tail is slower, so its bandwidth must be lower, never higher
    assert entry["bandwidth_p95"]["gb_per_s"] < entry["bandwidth_p50"]["gb_per_s"]
    # wall clock includes the synchronize, so it can only be slower
    assert entry["wall_bandwidth_p50"]["gb_per_s"] < entry["bandwidth_p50"]["gb_per_s"]
    assert entry["note"] == "hi"


def test_summarize_transfer_omits_an_empty_note():
    entry = tool.summarize_transfer(1024, [1.0], [2.0])
    assert "note" not in entry


def test_hidden_fraction_spans_none_to_all():
    # nothing overlapped: the wall time is the sum
    assert tool.hidden_fraction(10.0, 5.0, 15.0) == pytest.approx(0.0)
    # compute longer than the copy and fully concurrent: the copy is free
    assert tool.hidden_fraction(10.0, 20.0, 20.0) == pytest.approx(1.0)
    # compute shorter than the copy: at best min(copy, compute) is hidden, so
    # half of this copy is still exposed even at the ideal wall time
    assert tool.hidden_fraction(10.0, 5.0, 10.0) == pytest.approx(0.5)
    assert tool.hidden_fraction(10.0, 5.0, 12.5) == pytest.approx(0.25)
    assert tool.hidden_fraction(10.0, 5.0, 10.0) > tool.hidden_fraction(10.0, 5.0, 12.5)
    # measurement noise under the ideal must not invent extra overlap
    assert tool.hidden_fraction(10.0, 5.0, 8.0) == pytest.approx(0.5)
    assert tool.hidden_fraction(10.0, 20.0, 15.0) == pytest.approx(1.0)


def test_hidden_fraction_requires_a_real_copy():
    with pytest.raises(tool.ProbeError):
        tool.hidden_fraction(0.0, 5.0, 5.0)


# --------------------------------------------------------------------------
# rollout arithmetic
# --------------------------------------------------------------------------
def test_weight_offload_is_what_the_cap_cannot_hold():
    cap = 24 * GIB
    total = 64 * GIB
    assert tool.weight_offload_bytes(total, cap) == 40 * GIB
    # anything else resident eats into the room for weights
    assert tool.weight_offload_bytes(total, cap, 4 * GIB) == 44 * GIB
    # a model that fits streams nothing
    assert tool.weight_offload_bytes(8 * GIB, cap) == 0
    # over-subscribed "other" cannot produce negative room
    assert tool.weight_offload_bytes(total, cap, 100 * GIB) == total


def test_weight_offload_rejects_negative_inputs():
    with pytest.raises(tool.ProbeError):
        tool.weight_offload_bytes(-1, 1)
    with pytest.raises(tool.ProbeError):
        tool.weight_offload_bytes(1, -1)
    with pytest.raises(tool.ProbeError):
        tool.weight_offload_bytes(1, 1, -1)


def test_estimate_rollout_is_serial_upper_and_overlap_lower_bound():
    geo = tool.KVGeometry(rows=20996, heads=56, head_dim=128, layers=50)
    est = tool.estimate_rollout(
        forwards=59,
        layers=50,
        per_layer_kv_bytes=geo.per_layer_bytes,
        weight_bytes_per_forward=0,
        h2d_gb_per_s=20.0,
        forward_compute_ms=100.0,
    )
    assert est["available"] is True
    assert est["kv_bytes_per_forward"] == geo.per_layer_bytes * 50
    assert est["kv_bytes_total"] == geo.per_layer_bytes * 50 * 59
    expected_copy = est["kv_bytes_total"] / (20.0 * GB)
    assert est["copy_seconds_total"] == pytest.approx(expected_copy)
    assert est["compute_seconds_total"] == pytest.approx(5.9)
    assert est["serial_seconds"] == pytest.approx(expected_copy + 5.9)
    assert est["overlap_lower_bound_seconds"] == pytest.approx(max(expected_copy, 5.9))
    assert est["overlap_lower_bound_seconds"] <= est["serial_seconds"]
    assert est["copy_bound"] is True


def test_estimate_rollout_counts_streamed_weights_on_top_of_kv():
    geo = tool.KVGeometry(rows=1024, heads=8, head_dim=64, layers=4)
    without = tool.estimate_rollout(
        forwards=10, layers=4, per_layer_kv_bytes=geo.per_layer_bytes,
        weight_bytes_per_forward=0, h2d_gb_per_s=10.0,
    )
    with_weights = tool.estimate_rollout(
        forwards=10, layers=4, per_layer_kv_bytes=geo.per_layer_bytes,
        weight_bytes_per_forward=GIB, h2d_gb_per_s=10.0,
    )
    assert with_weights["weight_bytes_total"] == 10 * GIB
    assert with_weights["copy_seconds_total"] > without["copy_seconds_total"]
    delta = with_weights["copy_seconds_total"] - without["copy_seconds_total"]
    assert delta == pytest.approx(10 * GIB / (10.0 * GB))


def test_estimate_rollout_adds_writeback_only_with_a_d2h_number():
    common = dict(forwards=4, layers=2, per_layer_kv_bytes=GIB,
                  weight_bytes_per_forward=0, h2d_gb_per_s=10.0,
                  kv_writeback_bytes_per_forward=GIB)
    no_d2h = tool.estimate_rollout(**common)
    with_d2h = tool.estimate_rollout(d2h_gb_per_s=5.0, **common)
    assert no_d2h["d2h_seconds_total"] == 0.0
    assert with_d2h["d2h_seconds_total"] == pytest.approx(4 * GIB / (5.0 * GB))
    assert with_d2h["copy_seconds_total"] > no_d2h["copy_seconds_total"]


def test_estimate_rollout_without_bandwidth_still_reports_the_bytes():
    est = tool.estimate_rollout(
        forwards=59, layers=50, per_layer_kv_bytes=GIB,
        weight_bytes_per_forward=0, h2d_gb_per_s=None,
    )
    assert est["available"] is False
    assert "reason" in est
    assert est["kv_bytes_total"] == 59 * 50 * GIB
    assert "serial_seconds" not in est


@pytest.mark.parametrize("kwargs", [{"forwards": 0}, {"layers": 0},
                                    {"per_layer_kv_bytes": -1},
                                    {"weight_bytes_per_forward": -1}])
def test_estimate_rollout_rejects_nonsense(kwargs):
    fields = dict(forwards=2, layers=2, per_layer_kv_bytes=1024,
                  weight_bytes_per_forward=0, h2d_gb_per_s=1.0)
    fields.update(kwargs)
    with pytest.raises(tool.ProbeError):
        tool.estimate_rollout(**fields)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_defaults_are_the_published_shape():
    args = tool.build_parser().parse_args([])
    assert args.rows == 20996
    assert args.heads == 56
    assert args.head_dim == 128
    assert args.layers == 50
    assert args.dtype == "bf16"
    assert args.block_gib == pytest.approx(1.29)
    assert args.forwards == 59
    assert args.vram_cap_gib == pytest.approx(24.0)
    assert args.warmup == 3
    assert args.iters == 10
    assert args.json == tool.DEFAULT_REPORT
    assert args.max_gpu_gib == pytest.approx(6.0)
    assert args.max_host_gib == pytest.approx(8.0)


def test_cli_overrides_are_typed():
    args = tool.build_parser().parse_args(
        ["--rows", "1024", "--heads", "8", "--head-dim", "64", "--dtype", "fp32",
         "--block-gib", "0.5", "--warmup", "0", "--iters", "2",
         "--json", ".cache/x.json"]
    )
    assert (args.rows, args.heads, args.head_dim) == (1024, 8, 64)
    assert args.dtype == "fp32"
    assert args.block_gib == pytest.approx(0.5)
    assert args.warmup == 0  # warmup may legitimately be zero
    assert args.iters == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--dtype", "int4"],
        ["--rows", "0"],
        ["--rows", "-5"],
        ["--iters", "0"],
        ["--warmup", "-1"],
        ["--block-gib", "0"],
        ["--compute-ms", "-1"],
        ["--vram-other-gib", "-1"],
        ["--rows", "not-a-number"],
    ],
)
def test_cli_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        tool.build_parser().parse_args(argv)


def test_scenario_selection_round_trips_and_rejects_unknown_names():
    assert tool.selected_scenarios(None) == list(tool.SCENARIOS)
    assert tool.selected_scenarios("") == list(tool.SCENARIOS)
    assert tool.selected_scenarios("compute_overlap, h2d_pinned_blocking") == [
        "compute_overlap", "h2d_pinned_blocking"
    ]
    with pytest.raises(tool.ProbeError):
        tool.selected_scenarios("h2d_pinned_blocking,does_not_exist")


def test_every_declared_scenario_has_an_implementation():
    assert set(tool.SCENARIOS) == set(tool.SCENARIO_FUNCS)


# --------------------------------------------------------------------------
# report path and atomic write
# --------------------------------------------------------------------------
def test_report_path_must_stay_inside_the_repository():
    inside = tool.resolve_report_path(".cache/probe.json")
    assert inside.is_absolute()
    assert str(inside).startswith(str(ROOT))
    with pytest.raises(tool.ProbeError):
        tool.resolve_report_path("/tmp/elsewhere.json")
    with pytest.raises(tool.ProbeError):
        tool.resolve_report_path(str(ROOT / ".." / "escaped.json"))


def test_atomic_write_creates_parents_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "nested" / "report.json"
    tool.atomic_write_json(target, {"ok": True, "value": 1})
    assert json.loads(target.read_text()) == {"ok": True, "value": 1}
    assert list(tmp_path.glob("**/*.tmp-*")) == []


def test_atomic_write_replaces_a_previous_report_wholesale(tmp_path):
    target = tmp_path / "report.json"
    tool.atomic_write_json(target, {"run": 1, "padding": "x" * 10000})
    tool.atomic_write_json(target, {"run": 2})
    assert json.loads(target.read_text()) == {"run": 2}
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_serialises_unknown_objects_instead_of_raising(tmp_path):
    target = tmp_path / "report.json"
    tool.atomic_write_json(target, {"path": Path("/x"), "inf": None})
    assert json.loads(target.read_text())["path"] == "/x"


# --------------------------------------------------------------------------
# memory ceiling
# --------------------------------------------------------------------------
def test_budget_admits_what_fits_and_refuses_what_does_not():
    budget = tool.MemoryBudget(6 * GIB, 8 * GIB)
    budget.reserve("weights", 4 * GIB, "gpu")
    budget.reserve("kv", 1 * GIB, "gpu")
    assert budget.gpu_reserved == 5 * GIB
    with pytest.raises(tool.ProbeError) as excinfo:
        budget.reserve("second_block", 2 * GIB, "gpu")
    assert "gpu budget exceeded" in str(excinfo.value)
    # the refused reservation left no trace
    assert budget.gpu_reserved == 5 * GIB
    budget.release("weights", "gpu")
    budget.reserve("second_block", 2 * GIB, "gpu")
    assert budget.gpu_reserved == 3 * GIB


def test_budget_tracks_host_and_device_separately():
    budget = tool.MemoryBudget(6 * GIB, 8 * GIB)
    budget.reserve("pinned", 7 * GIB, "host")
    assert budget.host_reserved == 7 * GIB
    assert budget.gpu_reserved == 0
    budget.reserve("device", 5 * GIB, "gpu")  # unaffected by the host pressure
    with pytest.raises(tool.ProbeError):
        budget.reserve("more_pinned", 2 * GIB, "host")


def test_budget_rejects_duplicate_names_and_bad_kinds():
    budget = tool.MemoryBudget(GIB, GIB)
    budget.reserve("a", 1024, "gpu")
    with pytest.raises(tool.ProbeError):
        budget.reserve("a", 1024, "gpu")
    with pytest.raises(tool.ProbeError):
        budget.reserve("b", 1024, "vram")
    with pytest.raises(tool.ProbeError):
        budget.reserve("c", 0, "gpu")


def test_budget_limits_must_be_positive():
    with pytest.raises(tool.ProbeError):
        tool.MemoryBudget(0, GIB)
    with pytest.raises(tool.ProbeError):
        tool.MemoryBudget(GIB, -1)


def test_default_budget_keeps_the_probe_within_six_and_eight_gib():
    """The defaults must actually fit the largest scenario's footprint.

    Double buffering holds two block-sized landing buffers on the device and
    one pinned source; the weight+KV scenario holds a block plus a KV layer on
    both sides. If a default ever drifts past the ceiling the probe would fail
    on the box instead of here.
    """
    args = tool.build_parser().parse_args([])
    geo = tool.KVGeometry(rows=args.rows, heads=args.heads,
                          head_dim=args.head_dim, layers=args.layers)
    block = int(round(args.block_gib * GIB))
    worst_gpu = max(2 * block, block + geo.per_layer_bytes)
    worst_host = max(block, block + geo.per_layer_bytes)
    assert worst_gpu <= args.max_gpu_gib * GIB
    assert worst_host <= args.max_host_gib * GIB


# --------------------------------------------------------------------------
# estimates wired to the CLI
# --------------------------------------------------------------------------
def _args(**overrides) -> argparse.Namespace:
    argv = []
    for key, value in overrides.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    return tool.build_parser().parse_args(argv)


def test_build_estimates_derives_weight_offload_from_the_cap():
    args = _args(assume_h2d_gb_s=20.0)
    geo = tool.KVGeometry(rows=args.rows, heads=args.heads,
                          head_dim=args.head_dim, layers=args.layers)
    estimates = tool.build_estimates(args, geo, {})
    offload = estimates["weight_offload"]
    # default weight total is layers x block-gib = 50 x 1.29 GiB
    assert offload["weight_total_gib"] == pytest.approx(50 * 1.29, abs=1e-6)
    assert offload["weight_offload_gib_per_forward"] == pytest.approx(
        50 * 1.29 - 24.0, abs=1e-3
    )
    assert "derived" in offload["source"]
    assert estimates["bandwidth_used"]["source"].startswith("assumed")
    assert estimates["all_kv_on_cpu"]["weight_bytes_total"] == 0
    assert (
        estimates["all_kv_on_cpu_plus_weight_offload"]["serial_seconds"]
        > estimates["all_kv_on_cpu"]["serial_seconds"]
    )


def test_build_estimates_honours_an_explicit_offload_volume():
    args = _args(assume_h2d_gb_s=20.0, weight_offload_gib=2.0)
    geo = tool.KVGeometry(rows=64, heads=2, head_dim=8, layers=2)
    estimates = tool.build_estimates(args, geo, {})
    assert estimates["weight_offload"]["weight_offload_gib_per_forward"] == pytest.approx(2.0)
    assert "explicit" in estimates["weight_offload"]["source"]


def test_build_estimates_prefers_measured_bandwidth_over_the_assumption():
    args = _args(assume_h2d_gb_s=1.0)
    geo = tool.KVGeometry(rows=64, heads=2, head_dim=8, layers=2)
    measurements = {
        "h2d_pinned_nonblocking": tool.summarize_transfer(GB, [50.0], [55.0]),
        "d2h_pinned_nonblocking": tool.summarize_transfer(GB, [100.0], [105.0]),
    }
    estimates = tool.build_estimates(args, geo, measurements)
    assert estimates["bandwidth_used"]["source"] == "measured"
    assert estimates["bandwidth_used"]["h2d_gb_per_s"] == pytest.approx(20.0)
    assert estimates["bandwidth_used"]["d2h_gb_per_s"] == pytest.approx(10.0)


def test_measured_bandwidth_falls_back_to_the_blocking_h2d_number():
    only_blocking = {"h2d_pinned_blocking": tool.summarize_transfer(GB, [100.0], [110.0])}
    picked = tool.measured_bandwidths(only_blocking)
    assert picked["h2d_gb_per_s"] == pytest.approx(10.0)
    assert picked["d2h_gb_per_s"] is None
    assert tool.measured_bandwidths({}) == {"h2d_gb_per_s": None, "d2h_gb_per_s": None}


# --------------------------------------------------------------------------
# the no-CUDA path, end to end
# --------------------------------------------------------------------------
@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(tool.torch.cuda, "is_available", lambda: False)


def test_main_without_cuda_writes_a_report_and_says_so(tmp_path, monkeypatch, no_cuda):
    report_path = ROOT / ".cache" / "tests" / "offload_bandwidth_no_cuda.json"
    if report_path.exists():
        report_path.unlink()
    code = tool.main(
        ["--device", "cpu", "--assume-h2d-gb-s", "20",
         "--json", str(report_path.relative_to(ROOT))]
    )
    assert code == tool.EXIT_ENVIRONMENT
    report = json.loads(report_path.read_text())

    assert report["ok"] is False
    assert report["measurements"] == {}
    # every requested scenario is named as skipped, with a reason
    assert set(report["skipped"]) == set(tool.SCENARIOS)
    assert all("CUDA" in reason for reason in report["skipped"].values())
    assert report["failed"] == {}
    # the geometry and the estimates survive the absence of a GPU
    assert report["geometry"]["per_layer_bytes"] == 601_997_312
    est = report["estimates"]["all_kv_on_cpu"]
    assert est["available"] is True
    assert est["forwards"] == 59
    assert est["kv_bytes_total"] == 601_997_312 * 50 * 59
    assert est["serial_seconds"] > 0
    assert est["overlap_lower_bound_seconds"] <= est["serial_seconds"]
    assert report["estimates"]["bandwidth_used"]["source"].startswith("assumed")
    report_path.unlink()


def test_main_without_cuda_or_an_assumption_reports_bytes_but_no_seconds(no_cuda):
    report_path = ROOT / ".cache" / "tests" / "offload_bandwidth_no_bandwidth.json"
    code = tool.main(["--device", "cpu", "--json", str(report_path.relative_to(ROOT))])
    assert code == tool.EXIT_ENVIRONMENT
    report = json.loads(report_path.read_text())
    est = report["estimates"]["all_kv_on_cpu"]
    assert est["available"] is False
    assert "reason" in est
    assert est["kv_gib_total"] > 0
    report_path.unlink()


def test_main_refuses_a_report_path_outside_the_repository(capsys, no_cuda):
    assert tool.main(["--device", "cpu", "--json", "/tmp/nope.json"]) == tool.EXIT_FAILED
    assert "PROBE REFUSED" in capsys.readouterr().err


def test_run_probe_skips_scenarios_without_touching_cuda(no_cuda, monkeypatch):
    """The skip must be decided before any scenario runs, not inside one."""
    for name in tool.SCENARIO_FUNCS:
        monkeypatch.setitem(
            tool.SCENARIO_FUNCS, name,
            lambda ctx: pytest.fail("a scenario ran without CUDA"),
        )
    args = _args(device="cpu")
    report, code = tool.run_probe(args)
    assert code == tool.EXIT_ENVIRONMENT
    assert report["device"]["cuda_available"] is False
    assert report["requested_scenarios"] == list(tool.SCENARIOS)


def test_run_probe_honours_only_when_listing_skips(no_cuda):
    args = _args(device="cpu", only="compute_overlap")
    report, _ = tool.run_probe(args)
    assert report["requested_scenarios"] == ["compute_overlap"]
    assert set(report["skipped"]) == {"compute_overlap"}


def test_render_survives_a_report_with_nothing_measured(no_cuda):
    args = _args(device="cpu", assume_h2d_gb_s=20.0)
    report, _ = tool.run_probe(args)
    text = tool.render(report)
    assert "KV geometry" in text
    assert "[SKIP]" in text
    assert "FAILURES PRESENT" in text
    assert "estimate all_kv_on_cpu" in text


def test_render_formats_a_measured_report():
    """Feed render() a synthetic but structurally real measurement set."""
    args = _args(device="cpu", assume_h2d_gb_s=20.0)
    geo = tool.KVGeometry(rows=args.rows, heads=args.heads,
                          head_dim=args.head_dim, layers=args.layers)
    measurements = {
        "h2d_pinned_nonblocking": tool.summarize_transfer(GB, [50.0, 52.0], [55.0, 57.0]),
        "compute_overlap": {
            "copy_hidden_fraction": 0.8,
            "serial_reference_ms": 30.0,
            "ideal_overlap_ms": 20.0,
            "overlapped_wall_ms": tool.describe([22.0, 23.0]),
        },
        "parallel_two_copy_streams": {
            "gib_total": 1.85,
            "wall_ms": tool.describe([100.0, 101.0]),
            "aggregate_bandwidth_p50": tool.bandwidth(2 * GB, 100.0),
        },
    }
    report = {
        "device": tool.device_info(tool.torch.device("cpu")),
        "geometry": geo.to_dict(),
        "measurements": measurements,
        "skipped": {"d2h_pinned_nonblocking": "not requested"},
        "failed": {"h2d_pageable_blocking": "ProbeError: pinned allocation failed"},
        "estimates": tool.build_estimates(args, geo, measurements),
        "ok": False,
    }
    text = tool.render(report)
    assert "h2d_pinned_nonblocking" in text
    assert "copy hidden 80.0%" in text
    assert "[FAIL] h2d_pageable_blocking" in text
    assert "[SKIP] d2h_pinned_nonblocking" in text


def test_device_info_is_honest_about_a_missing_gpu(no_cuda):
    info = tool.device_info(tool.torch.device("cuda"))
    assert info["cuda_available"] is False
    assert "name" not in info
    assert info["torch"] == tool.torch.__version__


# --------------------------------------------------------------------------
# the timing loop and the failure bookkeeping, against a fake CUDA
# --------------------------------------------------------------------------
class _Clock:
    """A virtual clock, so timings are asserted rather than approximated."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def advance(self, seconds: float) -> None:
        self.seconds += seconds

    def __call__(self) -> float:  # stands in for time.perf_counter
        return self.seconds


class _FakeEvent:
    """A CUDA event whose "device time" is the virtual clock."""

    clock: _Clock = None  # set by the fixture

    def __init__(self, enable_timing: bool = False) -> None:
        self.enable_timing = enable_timing
        self.stamp = None

    def record(self, stream=None) -> None:
        self.stamp = type(self).clock.seconds

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert self.stamp is not None and other.stamp is not None
        return (other.stamp - self.stamp) * 1000.0


class _FakeStream:
    def wait_event(self, event) -> None:  # pragma: no cover - not exercised here
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    """Enough of ``torch.cuda`` to drive the timing loop with no GPU present."""
    clock = _Clock()
    _FakeEvent.clock = clock
    cuda = tool.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "Event", _FakeEvent)
    monkeypatch.setattr(cuda, "Stream", lambda device=None: _FakeStream())
    monkeypatch.setattr(cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda device=None: 3 * GIB)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda device=None: 4 * GIB)
    monkeypatch.setattr(
        cuda, "get_device_properties",
        lambda index: SimpleNamespace(
            name="FakeGPU", total_memory=24 * GIB, major=8, minor=9,
            multi_processor_count=108,
        ),
    )
    monkeypatch.setattr(tool.time, "perf_counter", clock)
    return clock


def test_measure_discards_warmup_and_keeps_one_sample_per_iteration(fake_cuda):
    calls = {"n": 0}

    def body():
        calls["n"] += 1
        fake_cuda.advance(0.010)  # 10 ms of "device" time per transfer

    event_ms, wall_ms = measure_under_fake(body, warmup=2, iters=3)
    assert calls["n"] == 5  # warmup ran, it just was not recorded
    assert len(event_ms) == len(wall_ms) == 3
    assert event_ms == [pytest.approx(10.0)] * 3
    assert wall_ms == [pytest.approx(10.0)] * 3


def measure_under_fake(body, *, warmup, iters):
    return tool.measure(
        body, device=tool.torch.device("cuda"), warmup=warmup, iters=iters
    )


def test_measure_with_zero_warmup_records_everything(fake_cuda):
    event_ms, _ = measure_under_fake(lambda: fake_cuda.advance(0.005),
                                     warmup=0, iters=4)
    assert event_ms == [pytest.approx(5.0)] * 4


def test_measure_wall_discards_warmup_too(fake_cuda):
    calls = {"n": 0}

    def body():
        calls["n"] += 1
        fake_cuda.advance(0.020)

    wall_ms = tool.measure_wall(
        body, device=tool.torch.device("cuda"), warmup=1, iters=2
    )
    assert calls["n"] == 3
    assert wall_ms == [pytest.approx(20.0)] * 2


def test_measured_timings_feed_the_reported_bandwidth(fake_cuda):
    """End to end over the fake: 1 GB in 50 ms must read as 20 GB/s."""
    event_ms, wall_ms = measure_under_fake(
        lambda: fake_cuda.advance(0.050), warmup=1, iters=2
    )
    entry = tool.summarize_transfer(GB, event_ms, wall_ms)
    assert entry["bandwidth_p50"]["gb_per_s"] == pytest.approx(20.0)


def test_run_probe_records_a_failing_scenario_without_losing_the_others(
    fake_cuda, monkeypatch
):
    """A pinned-allocation failure is loud, named, and non-fatal to the report.

    This is the OOM path: the scenario raises, the probe keeps the failure in
    the report, still finishes the other scenarios, still computes the
    estimates from what it did measure, and exits non-zero.
    """
    good = tool.summarize_transfer(GB, [50.0, 51.0], [55.0, 56.0])

    def boom(ctx):
        raise tool.ProbeError(
            "pinned host allocation 'pinned_src' of 1.290 GiB failed: "
            "RuntimeError: CUDA driver error: out of memory"
        )

    monkeypatch.setattr(
        tool, "SCENARIO_FUNCS",
        {
            "h2d_pinned_nonblocking": lambda ctx: good,
            "h2d_pageable_blocking": boom,
        },
    )
    monkeypatch.setattr(tool, "SCENARIOS", ("h2d_pinned_nonblocking",
                                            "h2d_pageable_blocking"))
    args = _args(device="cuda", only="h2d_pinned_nonblocking,h2d_pageable_blocking")
    report, code = tool.run_probe(args)

    assert code == tool.EXIT_FAILED
    assert report["ok"] is False
    assert report["measurements"]["h2d_pinned_nonblocking"] == good
    failure = report["failed"]["h2d_pageable_blocking"]
    assert failure.startswith("ProbeError:")
    assert "out of memory" in failure
    # measured bandwidth still drives the estimates
    assert report["estimates"]["bandwidth_used"]["source"] == "measured"
    assert report["estimates"]["all_kv_on_cpu"]["available"] is True
    assert report["peak_memory"]["torch_max_allocated_gib"] == pytest.approx(3.0)
    assert report["device"]["name"] == "FakeGPU"


def test_main_writes_the_report_even_when_a_scenario_failed(fake_cuda, monkeypatch):
    def boom(ctx):
        raise tool.ProbeError("pinned host allocation failed: out of memory")

    monkeypatch.setattr(tool, "SCENARIO_FUNCS", {"h2d_pinned_blocking": boom})
    monkeypatch.setattr(tool, "SCENARIOS", ("h2d_pinned_blocking",))
    report_path = ROOT / ".cache" / "tests" / "offload_bandwidth_failure.json"
    code = tool.main(
        ["--device", "cuda", "--only", "h2d_pinned_blocking",
         "--json", str(report_path.relative_to(ROOT))]
    )
    assert code == tool.EXIT_FAILED
    report = json.loads(report_path.read_text())
    assert report["ok"] is False
    assert "out of memory" in report["failed"]["h2d_pinned_blocking"]
    report_path.unlink()


def test_main_reports_success_when_every_scenario_ran(fake_cuda, monkeypatch):
    good = tool.summarize_transfer(GB, [50.0], [55.0])
    monkeypatch.setattr(tool, "SCENARIO_FUNCS", {"h2d_pinned_nonblocking": lambda ctx: good})
    monkeypatch.setattr(tool, "SCENARIOS", ("h2d_pinned_nonblocking",))
    report_path = ROOT / ".cache" / "tests" / "offload_bandwidth_ok.json"
    code = tool.main(
        ["--device", "cuda", "--only", "h2d_pinned_nonblocking",
         "--json", str(report_path.relative_to(ROOT))]
    )
    assert code == tool.EXIT_OK
    report = json.loads(report_path.read_text())
    assert report["ok"] is True
    assert report["failed"] == {}
    assert report["estimates"]["bandwidth_used"]["h2d_gb_per_s"] == pytest.approx(20.0)
    report_path.unlink()


# --------------------------------------------------------------------------
# the arena refuses to allocate past the ceiling, without a GPU in sight
# --------------------------------------------------------------------------
def test_arena_pageable_allocation_is_budgeted_and_released():
    budget = tool.MemoryBudget(GIB, 16 * 1024 * 1024)
    device = tool.torch.device("cpu")
    with tool.Arena(budget, device) as arena:
        buffer = arena.pageable_bytes("src", 4 * 1024 * 1024)
        assert buffer.numel() == 4 * 1024 * 1024
        assert budget.host_reserved == 4 * 1024 * 1024
        with pytest.raises(tool.ProbeError):
            arena.pageable_bytes("too_big", 32 * 1024 * 1024)
    assert budget.host_reserved == 0


def test_arena_adopt_books_an_existing_tensor():
    budget = tool.MemoryBudget(GIB, GIB)
    with tool.Arena(budget, tool.torch.device("cpu")) as arena:
        tensor = tool.torch.zeros(1024, dtype=tool.torch.float32)
        arena.adopt("workload", tensor, kind="gpu")
        assert budget.gpu_reserved == 4096
    assert budget.gpu_reserved == 0


def test_compute_workload_rejects_a_non_positive_target():
    budget = tool.MemoryBudget(GIB, GIB)
    with tool.Arena(budget, tool.torch.device("cpu")) as arena:
        with pytest.raises(tool.ProbeError):
            tool.ComputeWorkload(tool.torch.device("cpu"), 0.0, arena)


def test_probe_never_uses_a_cpu_sleep_for_the_overlap_workload():
    """A ``time.sleep`` on the compute stream would fake every overlap number."""
    source = (ROOT / "tools" / "probe_offload_bandwidth.py").read_text()
    assert "time.sleep(" not in source  # prose may mention it; code may not
    assert "torch.cuda._sleep" in source
    assert tool.ComputeWorkload.launch.__doc__
