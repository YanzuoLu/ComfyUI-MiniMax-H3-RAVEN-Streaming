"""The two-process parity protocol: shared inputs, K/V dumps, tolerance maths.

What is pinned here:

* the shared input file round-trips and carries everything both sides need --
  the whole point of shipping tensors instead of a seed;
* ``compare_tensors`` really is ``|ours - ref| <= atol + rtol * |ref|``, so a
  purely *relative* difference passes or fails on ``rtol`` and not on ``atol``
  (an atol-only comparator would silently pass a 3% error on small tensors and
  fail it on large ones);
* a RAVEN-shaped tap (varlen ``[rows, heads, head_dim]``) and the Comfy tap
  produce entries in the same frame, which is what makes the two dumps
  comparable at all. Both are now literally the same layout: the causal lane
  calls RAVEN's packed 3-D attention, so ``canonical_kv`` has nothing to
  transpose on either side;
* the Comfy tap separates the token refiner from the DiT blocks. They share one
  attention seam and only the DiT calls have a counterpart in the RAVEN dump, so
  a refiner call left in ``entries`` would silently renumber every DiT layer of
  the text forward;
* the harness's tensor-convention conversions invert each other and agree with
  the model's own audio packing.

The real RAVEN model is *not* imported here: ComfyUI and RAVEN both ship
top-level ``utils``/``common`` packages and cannot share a process. The
end-to-end cross-implementation run is a subprocess test, opt-in through
``RAVEN_ROOT`` (see ``test_raven_harness_cross_check``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import conftest  # noqa: E402
import probe_causal_parity as probe  # noqa: E402
import raven_parity_harness as harness  # noqa: E402


# --- shared inputs -----------------------------------------------------------


def test_shared_inputs_carry_every_tensor_both_sides_read(tmp_path):
    inputs = probe.build_shared_inputs(
        frames=39, width=512, height=288, text_len=16, seed=3,
        sink=2, window=2, video_sigma=0.6, audio_sigma=None, arch="tiny",
    )
    request = inputs["request"]
    assert request["arch"] == "tiny"
    assert request["num_chunks"] == 3
    assert request["latent_t"] == 12 and request["audio_t"] == 65
    assert inputs["context"].shape == (1, 16, probe.TINY_CONFIG["text_dim"])
    for name in ("video_xt", "video_x0", "video_eps"):
        assert inputs[name].shape == (1, 24, 12, 18, 32)
    for name in ("audio_xt", "audio_x0", "audio_eps"):
        assert inputs[name].shape == (1, 32, 2, 65)

    path = tmp_path / "inputs.pt"
    probe.save_inputs(str(path), inputs)
    loaded = probe.load_inputs(str(path))
    assert loaded["request"] == request
    for name in ("context", "video_xt", "audio_eps"):
        assert torch.equal(loaded[name], inputs[name])


def test_audio_sigma_defaults_to_the_shifted_grid():
    inputs = probe.build_shared_inputs(
        frames=22, width=512, height=288, text_len=8, seed=0, sink=1, window=None,
        video_sigma=0.6, audio_sigma=None, arch="tiny",
    )
    assert inputs["request"]["audio_sigma"] != inputs["request"]["video_sigma"]
    assert inputs["request"]["audio_sigma"] == pytest.approx(
        probe._shifted_audio_sigma(0.6), rel=1e-12
    )
    assert inputs["request"]["window"] is None


def test_loading_a_foreign_schema_version_is_refused(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"version": 999}, path)
    with pytest.raises(SystemExit, match="schema version"):
        probe.load_inputs(str(path))


def test_shifted_audio_sigma_matches_the_official_rule(comfyui_on_syspath):
    from comfy.ldm.minimax.model import time_shift_sigma

    for sigma in (0.999, 0.6, 0.1, 0.001):
        expected = float(time_shift_sigma(torch.tensor(sigma), 12.0, 3.0))
        assert probe._shifted_audio_sigma(sigma) == pytest.approx(expected, rel=1e-6)


# --- canonical layouts -------------------------------------------------------


def test_canonical_kv_accepts_both_producer_layouts():
    rows, heads, dim = 5, 2, 4
    varlen = torch.randn(rows, heads, dim)              # RAVEN
    container = varlen.transpose(0, 1).unsqueeze(0)     # Comfy
    assert torch.equal(probe.canonical_kv(varlen), probe.canonical_kv(container))
    assert probe.canonical_kv(container).shape == (rows, heads, dim)


def test_canonical_kv_refuses_anything_else():
    with pytest.raises(ValueError):
        probe.canonical_kv(torch.randn(3, 4))
    with pytest.raises(ValueError, match="batch 1"):
        probe.canonical_kv(torch.randn(2, 2, 5, 4))


@pytest.mark.parametrize(
    "spec,expected",
    [("all", [0, 1, 2, 3]), ("none", []), ("", []), ("first,mid,last", [0, 2, 3]),
     ("0,2", [0, 2]), ("-1", [3]), ("7", [])],
)
def test_select_layers(spec, expected):
    assert probe.select_layers(spec, 4) == expected


# --- tolerance maths ---------------------------------------------------------


def test_compare_tensors_reports_abs_rel_and_allclose():
    ref = torch.tensor([1.0, 10.0, 100.0])
    ours = torch.tensor([1.01, 10.0, 100.0])
    result = probe.compare_tensors(ours, ref, atol=0.0, rtol=0.02)
    assert result["max_abs"] == pytest.approx(0.01, rel=1e-5)
    assert result["max_rel"] == pytest.approx(0.01, rel=1e-5)
    assert result["allclose"] is True
    assert result["over_tolerance_elements"] == 0


def test_relative_error_is_judged_by_rtol_not_atol():
    """A 3% error: atol alone cannot see it, rtol must."""
    ref = torch.tensor([100.0, 200.0])
    ours = ref * 1.03
    tight = probe.compare_tensors(ours, ref, atol=1e-3, rtol=0.02)
    assert tight["allclose"] is False
    assert tight["max_rel"] == pytest.approx(0.03, rel=1e-5)
    assert tight["max_abs"] == pytest.approx(6.0, rel=1e-5)
    loose = probe.compare_tensors(ours, ref, atol=1e-3, rtol=0.05)
    assert loose["allclose"] is True


def test_atol_covers_near_zero_references():
    ref = torch.tensor([0.0, 1e-9])
    ours = torch.tensor([1e-4, 1e-4])
    assert probe.compare_tensors(ours, ref, atol=1e-3, rtol=0.0)["allclose"] is True
    strict = probe.compare_tensors(ours, ref, atol=1e-6, rtol=0.5)
    assert strict["allclose"] is False
    assert strict["max_rel"] > 1e3  # meaningless on its own, which is the point


def test_shape_mismatch_is_reported_not_raised():
    result = probe.compare_tensors(torch.zeros(2, 2), torch.zeros(3, 2), atol=1, rtol=1)
    assert result["allclose"] is False
    assert result["shape_mismatch"] == [[2, 2], [3, 2]]


# --- scale-free metrics ------------------------------------------------------


def test_metrics_on_identical_tensors():
    x = torch.randn(64, 8)
    metrics = probe.tensor_metrics(x, x)
    assert metrics["rel_l2"] == 0.0
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["max_abs"] == 0.0
    assert metrics["max_abs_over_ref_absmax"] == 0.0
    assert metrics["p99_abs_over_ref_absmax"] == 0.0


def test_metrics_are_stable_when_both_tensors_are_zero():
    """The near-zero case that makes element-wise ``max_rel`` useless."""
    zeros = torch.zeros(16, 4)
    metrics = probe.tensor_metrics(zeros, zeros)
    assert metrics["rel_l2"] == 0.0
    assert metrics["cosine"] == 1.0
    assert probe.metrics_pass(metrics, rel_l2_max=0.0, cos_min=1.0)


def test_metrics_flag_a_zero_reference_with_a_non_zero_candidate():
    metrics = probe.tensor_metrics(torch.ones(4), torch.zeros(4))
    assert metrics["rel_l2"] == float("inf")
    assert metrics["cosine"] == 0.0
    assert not probe.metrics_pass(metrics, rel_l2_max=1e9, cos_min=-1.0)


def test_metrics_ignore_a_lone_near_zero_reference_element():
    """One 1e-12 reference element must not dominate the verdict."""
    ref = torch.tensor([1.0, 2.0, 3.0, 1e-12])
    ours = torch.tensor([1.0, 2.0, 3.0, 1e-12 + 1e-6])
    metrics = probe.tensor_metrics(ours, ref)
    assert metrics["rel_l2"] < 1e-6
    assert metrics["cosine"] > 0.999999
    # the element-wise view of the same pair is 1e6 relative error
    legacy = probe.compare_tensors(ours, ref, atol=0.0, rtol=0.0)
    assert legacy["max_rel"] > 1e5
    assert probe.metrics_pass(metrics, rel_l2_max=0.01, cos_min=0.999)


def test_pure_scale_error_shows_in_rel_l2_but_not_in_cosine():
    ref = torch.randn(128, 4)
    metrics = probe.tensor_metrics(ref * 1.09, ref)
    assert metrics["rel_l2"] == pytest.approx(0.09, rel=1e-5)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert not probe.metrics_pass(metrics, rel_l2_max=0.03, cos_min=0.999)


def test_direction_error_shows_in_cosine():
    ref = torch.tensor([1.0, 0.0])
    metrics = probe.tensor_metrics(torch.tensor([0.0, 1.0]), ref)
    assert metrics["cosine"] == pytest.approx(0.0, abs=1e-12)
    assert not probe.metrics_pass(metrics, rel_l2_max=10.0, cos_min=0.999)


def test_metrics_report_shape_mismatch_instead_of_raising():
    metrics = probe.tensor_metrics(torch.zeros(4, 2), torch.zeros(4, 3))
    assert metrics["shape_mismatch"] == [[4, 2], [4, 3]]
    assert not probe.metrics_pass(metrics, rel_l2_max=1.0, cos_min=-1.0)


def test_p99_subsamples_deterministically_on_large_tensors():
    ref = torch.randn(4096, 8)
    ours = ref + 1e-3
    small = probe.tensor_metrics(ours, ref)
    assert small["p99_subsampled"] is False
    big = probe.tensor_metrics(ours, ref, p99_max_elements=1024)
    again = probe.tensor_metrics(ours, ref, p99_max_elements=1024)
    assert big["p99_subsampled"] is True
    assert big["p99_abs"] == again["p99_abs"]  # stride, not RNG
    assert big["p99_abs"] == pytest.approx(small["p99_abs"], rel=1e-6)


def test_runtime_meta_records_the_numerics_environment():
    meta = probe.runtime_meta()
    assert meta["torch"] == torch.__version__
    assert set(meta) >= {"torch", "torch_cuda", "float32_matmul_precision",
                         "tf32_matmul", "tf32_cudnn", "sdp", "device_name"}
    assert isinstance(meta["sdp"], dict)
    if torch.backends.cuda.is_built():
        assert set(meta["sdp"]) >= {"flash", "math", "mem_efficient"}


# --- dump comparison ---------------------------------------------------------


def _fake_dump(producer, scale=1.0, seed=0, layers=2, forwards=("text", "chunk0:noise")):
    generator = torch.Generator().manual_seed(seed)
    entries = []
    for forward in forwards:
        for layer in range(layers):
            q = torch.randn(4, 2, 3, generator=generator)
            k = torch.randn(6, 2, 3, generator=generator)
            v = torch.randn(6, 2, 3, generator=generator)
            entry = probe.record_entry(forward, layer, q * scale, k * scale, v * scale,
                                       store_full=True)
            entries.append(entry)
    outputs = [{"video_x0": torch.randn(1, 24, 5, 4, 4, generator=generator) * scale,
                "audio_x0": torch.randn(1, 32, 2, 7, generator=generator) * scale}]
    return {"version": probe.KV_DUMP_SCHEMA["version"], "producer": producer,
            "entries": entries, "outputs": outputs, "meta": {}}


def test_identical_dumps_compare_clean():
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, _fake_dump("comfy"), _fake_dump("raven"), 1e-9, 0.0)
    assert report.passed
    names = {check.name for check in report.checks}
    assert {"dump.call_count", "dump.call_order", "gate.layer0_k",
            "gate.output_video_x0", "dump.k_within_tolerance"} <= names


def test_dump_comparison_uses_rtol_for_the_allclose_diagnostic():
    ours = _fake_dump("comfy", scale=1.03)
    reference = _fake_dump("raven", scale=1.0)

    strict = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(strict, ours, reference, atol=1e-4, rtol=0.02)
    detail = next(c.detail for c in strict.checks if c.name == "dump.k_within_tolerance")
    assert detail["allclose"] is False
    assert detail["max_rel"] == pytest.approx(0.03, rel=1e-4)
    assert detail["over_tolerance_elements"] > 0

    loose = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(loose, ours, reference, atol=1e-4, rtol=0.05)
    loose_detail = next(c.detail for c in loose.checks
                        if c.name == "dump.k_within_tolerance")
    assert loose_detail["allclose"] is True


def test_allclose_never_decides_the_run():
    """A 3% scale error: the diagnostic can be tuned away, the gate cannot."""
    ours = _fake_dump("comfy", scale=1.03)
    reference = _fake_dump("raven", scale=1.0)
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, ours, reference, atol=1e-4, rtol=0.05)

    allclose_checks = [c for c in report.checks if "_within_tolerance" in c.name]
    assert allclose_checks and all(c.passed for c in allclose_checks)
    assert all(not c.gate for c in allclose_checks)
    # ... and the run still fails, because block 0 moved by 3%
    assert not report.passed
    assert any(c.name == "gate.layer0_k" and not c.passed for c in report.checks)


def test_deep_layer_drift_is_recorded_but_does_not_fail_the_run():
    """Fifty bf16 layers accumulate; that is not a parity failure by itself."""
    ours = _fake_dump("comfy")
    reference = _fake_dump("raven")
    for entry in ours["entries"]:
        if entry["layer"] != 0:
            for key in ("q", "k", "v"):
                entry[key] = entry[key] * 1.10

    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, ours, reference, atol=1e-4, rtol=0.02)
    assert report.passed
    deep = next(c for c in report.checks if c.name == "depth.worst_k")
    assert deep.gate is False
    assert deep.detail["rel_l2"] == pytest.approx(0.10, rel=1e-3)
    # the diagnostic saw it too, and did not sink the run
    assert not next(c for c in report.checks
                    if c.name == "dump.k_within_tolerance").passed


def test_the_measured_nine_percent_video_x0_does_not_pass():
    """The real 50-block run: video_x0 rel_l2 9% / cosine 0.99596, audio 2.56%.

    Pinned as a test because it is the one number the gate exists to refuse.
    """
    ours = _fake_dump("comfy")
    reference = _fake_dump("raven")
    generator = torch.Generator().manual_seed(5)

    def perturb(tensor, rel_l2):
        noise = torch.randn(tensor.shape, generator=generator)
        noise = noise / noise.norm() * tensor.norm() * rel_l2
        return tensor + noise

    ours["outputs"][0]["video_x0"] = perturb(reference["outputs"][0]["video_x0"], 0.09)
    ours["outputs"][0]["audio_x0"] = perturb(reference["outputs"][0]["audio_x0"], 0.0256)

    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, ours, reference, atol=1e9, rtol=1e9)

    video = next(c for c in report.checks if c.name == "gate.output_video_x0")
    audio = next(c for c in report.checks if c.name == "gate.output_audio_x0")
    assert video.detail["rel_l2"] == pytest.approx(0.09, rel=1e-3)
    assert not video.passed, "a 9% video_x0 must never be reported as parity"
    assert audio.detail["rel_l2"] == pytest.approx(0.0256, rel=1e-3)
    assert audio.passed
    assert not report.passed
    # the element-wise diagnostic, tuned wide open, passes -- and changes nothing
    assert next(c for c in report.checks
                if c.name == "dump.video_x0_within_tolerance").passed


def test_gate_thresholds_are_configurable_and_reported():
    ours = _fake_dump("comfy", scale=1.02)
    reference = _fake_dump("raven")
    loose = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(loose, ours, reference, 1e-4, 0.05,
                        layer0_rel_l2_max=0.05, output_rel_l2_max=0.05, cos_min=0.99)
    assert loose.passed
    assert loose.meta["gate"]["layer0_rel_l2_max"] == 0.05

    tight = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(tight, ours, reference, 1e-4, 0.05,
                        layer0_rel_l2_max=0.001, output_rel_l2_max=0.001, cos_min=0.999)
    assert not tight.passed


def test_runtime_environment_difference_is_reported_as_a_diagnostic():
    ours = _fake_dump("comfy")
    reference = _fake_dump("raven")
    reference["meta"] = {"runtime": dict(probe.runtime_meta(), tf32_matmul=True)}
    report = probe.Report(mode="test", device="cpu")
    report.meta["runtime"] = dict(probe.runtime_meta(), tf32_matmul=False)
    probe.compare_dumps(report, ours, reference, 1e-9, 0.0)

    check = next(c for c in report.checks if c.name == "env.runtime_matches")
    assert check.gate is False
    assert check.passed is False
    assert check.detail["tf32_matmul"] == {"ours": False, "reference": True}
    assert report.passed  # a diagnostic, not a gate


def test_missing_runtime_metadata_is_skipped_not_assumed_equal():
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, _fake_dump("comfy"), _fake_dump("raven"), 1e-9, 0.0)
    assert any(item["name"] == "env.runtime_recorded" for item in report.skipped)


def test_report_json_separates_gates_from_diagnostics():
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, _fake_dump("comfy"), _fake_dump("raven"), 1e-9, 0.0)
    payload = report.to_json()
    assert payload["gate_summary"]["gating_checks"] > 0
    assert payload["gate_summary"]["diagnostics"] > 0
    assert payload["gate_summary"]["gating_failed"] == 0
    assert any(row.get("depth") == "layer0" for row in payload["metrics"])
    assert any(row.get("depth") == "output" for row in payload["metrics"])


def test_dump_comparison_catches_call_count_and_order():
    short = _fake_dump("comfy", forwards=("text",))
    long = _fake_dump("raven")
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, short, long, 1e-3, 1e-3)
    assert not report.passed
    assert report.checks[0].name == "dump.call_count"

    swapped = _fake_dump("raven")
    swapped["entries"] = list(reversed(swapped["entries"]))
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, _fake_dump("comfy"), swapped, 1e-3, 1e-3)
    assert not report.passed
    assert any(c.name == "dump.call_order" and not c.passed for c in report.checks)


def test_stats_only_entries_skip_the_full_tensor_check():
    ours = _fake_dump("comfy")
    reference = _fake_dump("raven")
    for entry in reference["entries"]:
        for key in ("q", "k", "v"):
            entry.pop(key)
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, ours, reference, 1e-6, 0.0)
    skipped = {item["name"] for item in report.skipped}
    assert {"dump.q_within_tolerance", "dump.k_within_tolerance",
            "dump.v_within_tolerance"} <= skipped


def test_dump_round_trips_through_disk(tmp_path):
    dump = _fake_dump("comfy")
    path = tmp_path / "kv.pt"
    probe.write_dump(str(path), "comfy", dump["entries"], dump["outputs"], {"x": 1})
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["producer"] == "comfy"
    assert loaded["version"] == probe.KV_DUMP_SCHEMA["version"]
    assert "refiner_entries" not in loaded  # additive: only written when there are any
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, dump, loaded, 0.0, 0.0)
    assert report.passed


# --- the shared attention seam: DiT blocks vs the token refiner ---------------
#
# One function (``causal_model.raven_packed_attention``) now serves both the 50
# DiT blocks and, during the text prefill, the token refiner. RAVEN's harness
# taps only the DiT one, so a refiner call that leaked into ``entries`` would
# renumber every DiT layer of the text forward and break the comparison in a way
# that still counts out.


class _FakePackedAttention:
    """Stands in for ``raven_packed_attention``: 3-D packed, keyword-only scale."""

    def __init__(self):
        self.sites = []

    def __call__(self, q, k, v, *, scale, site=None):
        self.sites.append(site)
        return q


def _install_fake_seam(monkeypatch):
    import raven_streaming.causal_model as cm

    fake = _FakePackedAttention()
    monkeypatch.setattr(cm, "raven_packed_attention", fake)
    return cm, fake


def _call(cm, rows=4, kv_rows=6, site=None):
    q = torch.randn(rows, 2, 3)
    k = torch.randn(kv_rows, 2, 3)
    cm.raven_packed_attention(q, k, k.clone(), scale=0.5, site=site)


def test_tap_splits_refiner_calls_out_of_the_dit_entries(monkeypatch):
    cm, fake = _install_fake_seam(monkeypatch)
    tap = probe.KVTap(full_layers=[0], row_stride=1)
    with tap.install():
        with tap.forward("text"):
            _call(cm, rows=5, kv_rows=5, site=("text_refiner", 0))
            _call(cm, rows=5, kv_rows=5, site=("text_refiner", 1))
            for layer in range(3):
                _call(cm, rows=5, kv_rows=5, site=("dit", layer))
        with tap.forward("chunk0:noise"):
            for layer in range(3):
                _call(cm, rows=4, kv_rows=9, site=("dit", layer))

    # the refiner never touches the DiT numbering
    assert [e["layer"] for e in tap.by_forward("text")] == [0, 1, 2]
    assert [e["layer"] for e in tap.by_forward("chunk0:noise")] == [0, 1, 2]
    assert [e["forward"] for e in tap.refiner_entries] == ["text:refiner"] * 2
    assert [e["layer"] for e in tap.refiner_entries] == [0, 1]
    assert len(tap.refiners_by_forward("text")) == 2
    assert tap.unlabelled_calls == 0
    assert tap.order_violations == []
    # refiner entries are stats-only, even for a selected full layer
    assert all(key not in tap.refiner_entries[0] for key in ("q", "k", "v"))
    assert "k" in tap.by_forward("text")[0]
    # and the seam is restored
    assert cm.raven_packed_attention is fake
    assert fake.sites[0] == ("text_refiner", 0)


def test_tap_reports_a_layer_that_arrives_out_of_order(monkeypatch):
    cm, _ = _install_fake_seam(monkeypatch)
    tap = probe.KVTap()
    with tap.install():
        with tap.forward("text"):
            _call(cm, site=("dit", 0))
            _call(cm, site=("dit", 2))  # block 1 never ran, or ran unlabelled
    assert [e["layer"] for e in tap.entries] == [0, 2]  # the label wins, not the count
    assert tap.order_violations == [{"forward": "text", "label": 2, "position": 1}]


def test_tap_falls_back_to_counting_for_an_unlabelled_call(monkeypatch):
    cm, _ = _install_fake_seam(monkeypatch)
    tap = probe.KVTap()
    with tap.install():
        with tap.forward("text"):
            _call(cm, site=None)
            _call(cm, site=None)
    assert [e["layer"] for e in tap.entries] == [0, 1]
    assert tap.unlabelled_calls == 2  # gated by attention.tap_labels


def test_tap_classification_checks_gate_on_the_split():
    tap = probe.KVTap()
    tap.entries = [probe.record_entry("text", layer, torch.randn(5, 2, 3),
                                      torch.randn(5, 2, 3), torch.randn(5, 2, 3),
                                      store_full=False) for layer in range(3)]
    tap.refiner_entries = [probe.record_entry("text:refiner", 0, torch.randn(5, 2, 3),
                                              torch.randn(5, 2, 3), torch.randn(5, 2, 3),
                                              store_full=False)]
    report = probe.Report(mode="test", device="cpu")
    probe.add_tap_classification_checks(report, tap, num_layers=3, refiner_layers=1,
                                        text_rows=5)
    assert report.passed

    # a refiner call recorded as a DiT block: the count still "works", the split
    # does not
    leaked = probe.KVTap()
    leaked.entries = list(tap.entries) + [
        probe.record_entry("text", 3, torch.randn(5, 2, 3), torch.randn(5, 2, 3),
                           torch.randn(5, 2, 3), store_full=False)]
    bad = probe.Report(mode="test", device="cpu")
    probe.add_tap_classification_checks(bad, leaked, num_layers=3, refiner_layers=1,
                                        text_rows=5)
    assert not bad.passed
    failed = {c.name for c in bad.checks if not c.passed}
    assert "attention.dit_layers_complete" in failed
    assert "attention.refiner_calls_separated" in failed


def test_refiner_entries_ride_along_without_disturbing_a_comparison(tmp_path):
    """A Comfy dump with refiner entries still compares against a RAVEN dump."""
    ours = _fake_dump("comfy")
    ours["refiner_entries"] = [
        probe.record_entry("text:refiner", 0, torch.randn(5, 2, 3),
                           torch.randn(5, 2, 3), torch.randn(5, 2, 3),
                           store_full=False)]
    reference = _fake_dump("raven")
    assert "refiner_entries" not in reference
    report = probe.Report(mode="test", device="cpu")
    probe.compare_dumps(report, ours, reference, 1e-9, 0.0)
    assert report.passed
    call_count = next(c for c in report.checks if c.name == "dump.call_count")
    assert call_count.detail["calls"] == len(ours["entries"])

    path = tmp_path / "kv.pt"
    probe.write_dump(str(path), "comfy", ours["entries"], ours["outputs"], {},
                     refiner_entries=ours["refiner_entries"])
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert len(loaded["refiner_entries"]) == 1
    assert loaded["version"] == probe.KV_DUMP_SCHEMA["version"]


# --- RAVEN-side adapters (no RAVEN import) -----------------------------------


class _FakeRavenAttention:
    """Stands in for ``_CAUSAL_FLASH_ATTENTION``: varlen [rows, heads, dim]."""

    def __init__(self):
        self.calls = 0

    def __call__(self, q, k, v, **kwargs):
        self.calls += 1
        return q.clone()


class _FakeRavenModule:
    def __init__(self):
        self._CAUSAL_FLASH_ATTENTION = _FakeRavenAttention()


def test_raven_tap_records_the_same_frame_as_the_comfy_tap():
    module = _FakeRavenModule()
    inner = module._CAUSAL_FLASH_ATTENTION
    tap = harness.RavenKVTap(module, full_layers=[0, 1], row_stride=1)

    q = torch.randn(4, 2, 3)
    k = torch.randn(6, 2, 3)
    v = torch.randn(6, 2, 3)
    with tap:
        tap.forward("text")
        module._CAUSAL_FLASH_ATTENTION(q, k, v, q_lens=None, k_lens=None)
        tap.forward("chunk0:noise")
        module._CAUSAL_FLASH_ATTENTION(q, k, v)
    assert module._CAUSAL_FLASH_ATTENTION is inner  # restored
    assert inner.calls == 2
    assert [e["forward"] for e in tap.entries] == ["text", "chunk0:noise"]
    assert [e["layer"] for e in tap.entries] == [0, 0]
    assert tap.entries[0]["k"].shape == (6, 2, 3)

    # the same tensors seen through the Comfy container layout must land on the
    # same entry, which is what makes a cross-producer comparison legitimate
    comfy_entry = probe.record_entry(
        "text", 0,
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), store_full=True,
    )
    for key in ("q", "k", "v"):
        assert torch.equal(comfy_entry[key], tap.entries[0][key])
        assert comfy_entry["stats"][key] == tap.entries[0]["stats"][key]


def test_raven_tap_row_stride_subsamples_full_tensors():
    module = _FakeRavenModule()
    tap = harness.RavenKVTap(module, full_layers=[0], row_stride=3)
    with tap:
        tap.forward("text")
        module._CAUSAL_FLASH_ATTENTION(torch.randn(9, 2, 3), torch.randn(9, 2, 3),
                                       torch.randn(9, 2, 3))
    assert tap.entries[0]["k"].shape == (3, 2, 3)
    assert tap.entries[0]["stats"]["k"]["shape"] == [9, 2, 3]  # stats stay full


def test_video_and_audio_conventions_invert():
    video = torch.randn(1, 24, 12, 4, 6)
    chunk = harness.video_to_raven(video, 5, 10)
    assert chunk.shape == (24, 5, 4, 6)
    assert torch.equal(harness.video_to_comfy(chunk), video[:, :, 5:10])

    audio = torch.randn(1, 32, 2, 20)
    chunk = harness.audio_to_raven(audio, 4, 9)
    assert chunk.shape == (2, 32, 5)
    assert torch.equal(harness.audio_to_comfy(chunk), audio[:, :, :, 4:9])


def test_audio_convention_matches_the_models_own_packing(comfyui_on_syspath):
    """RAVEN's row packing of a chunk must equal Comfy's ``pack_audio`` of it."""
    from comfy.ldm.minimax.model import pack_audio

    audio = torch.randn(1, 32, 2, 20)
    start, stop = 4, 9
    raven_chunk = harness.audio_to_raven(audio, start, stop)
    # RAVEN: [2, C, a] -> rows [L(a..b) | R(a..b)]
    raven_rows = raven_chunk.permute(0, 2, 1).reshape(-1, raven_chunk.shape[1])
    comfy_rows = pack_audio(audio[:, :, :, start:stop])
    assert torch.equal(raven_rows, comfy_rows)


def test_packer_shim_exposes_exactly_what_the_packer_reads():
    shim = harness._PackerShim(
        text_len=7, latent_t=12, latent_h=18, latent_w=32, audio_t=65,
        latents_dim=24, audio_latents_dim=32, sink=2, window=2,
    )
    ids, count = shim.tokenizer.encode("anything")
    assert count == 7 and ids.shape == (7,)
    assert shim.latent_shape == (24, 12, 18, 32)
    assert shim.audio_shape == (2, 32, 65)  # RAVEN's stereo-first convention
    assert shim.window_size == 2


def test_harness_rejects_a_non_bf16_placement():
    with pytest.raises(SystemExit, match="bf16"):
        harness.parse_args(["--weights", "w.safetensors", "--inputs", "i.pt",
                            "--dtype", "fp32"])


def test_harness_rejects_a_directory_that_is_not_raven(tmp_path):
    with pytest.raises(SystemExit, match="RAVEN checkout"):
        harness.RavenModules(str(tmp_path))


# --- end-to-end cross-implementation run (opt-in) ----------------------------


@pytest.mark.skipif(not os.environ.get("RAVEN_ROOT"),
                    reason="set RAVEN_ROOT to a RAVEN checkout to run the real "
                           "cross-implementation parity (separate process: RAVEN "
                           "and ComfyUI both own a top-level 'utils' package)")
def test_raven_harness_cross_check(tmp_path):
    comfyui = conftest.find_upstream_comfyui()
    if comfyui is None:
        pytest.skip("no local ComfyUI checkout")
    inputs = tmp_path / "inputs.pt"
    weights = tmp_path / "tiny.safetensors"
    raven_dump = tmp_path / "raven_kv.pt"
    report = tmp_path / "parity.json"
    python = sys.executable
    env = dict(os.environ, COMFYUI_PATH=str(comfyui))

    subprocess.run(
        [python, str(ROOT / "tools" / "probe_causal_parity.py"), "--mode", "inputs",
         "--arch", "tiny", "--frames", "22", "--width", "512", "--height", "288",
         "--text-len", "32", "--sink", "2", "--window", "2",
         "--emit-inputs", str(inputs), "--emit-weights", str(weights)],
        check=True, env=env, capture_output=True,
    )
    subprocess.run(
        [python, str(ROOT / "tools" / "raven_parity_harness.py"),
         "--raven-root", os.environ["RAVEN_ROOT"], "--weights", str(weights),
         "--inputs", str(inputs), "--arch", "tiny", "--device", "cpu",
         "--dtype", "bf16", "--attention-backend", "sdpa", "--kv-layers", "all",
         "--emit-dump", str(raven_dump)],
        check=True, env=env, capture_output=True,
    )
    done = subprocess.run(
        [python, str(ROOT / "tools" / "probe_causal_parity.py"), "--mode", "real",
         "--arch", "tiny", "--dtype", "bf16", "--dit", str(weights),
         "--inputs", str(inputs), "--kv-layers", "all",
         "--attention-backend", "pytorch", "--compare-dump", str(raven_dump),
         "--atol", "0.008", "--rtol", "0.02", "--json", str(report)],
        check=False, env=env, capture_output=True, text=True,
    )
    payload = json.loads(report.read_text())
    assert payload["passed"], done.stdout + done.stderr
    assert payload["meta"]["reference_producer"] == "raven"
    assert done.returncode == 0
