"""``tools/probe_causal_operator_parity.py``: inputs, partial reads, stage gates.

The operator probe exists to localise the ~0.5-1% block-0 disagreement the
rollout probe measures, so the things that have to be right here are:

* the shared operator inputs are deterministic and carry the real geometry;
* only the requested block is read out of the checkpoint (the whole point of the
  tool is that it never materialises 66 GB);
* the per-stage comparison gates on ``rel_l2`` / ``cosine`` and names the stage
  that fails, rather than reporting one number for the whole block;
* the Comfy side runs end to end on CPU with the tiny architecture, and its
  staged replay is *bit-identical* to the production ``Attention.forward`` --
  without that self-check the staged numbers would be a re-implementation.

The RAVEN side needs a checkout and its own process (ComfyUI and RAVEN both own
a top-level ``utils`` package), so the cross-side run is opt-in through
``RAVEN_ROOT``.
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
import probe_causal_operator_parity as op  # noqa: E402
import probe_causal_parity as probe  # noqa: E402


# --- shared inputs -----------------------------------------------------------


def test_operator_inputs_carry_the_real_geometry():
    inputs = op.build_operator_inputs(arch="tiny", frames=39, width=512, height=288,
                                      text_len=128, seed=0)
    request = inputs["request"]
    # chunk 0 of a 39-frame 512x288 clip: 29 audio latents (x2 rows) + 5 video
    # latents x (18/2 * 32/2) rows
    assert request["rows"] == 58 + 5 * 144
    assert request["kv_rows"] == 128 + request["rows"]
    assert request["audio_rows"] == 58
    assert inputs["x"].shape == (request["rows"], 32)
    assert inputs["positions"].shape == (request["kv_rows"], 3)
    assert inputs["positions"].dtype == torch.float64
    assert inputs["attn_k"].shape == (request["kv_rows"], 2, 24)
    assert inputs["mod_rows"].shape == (request["rows"],)
    assert set(inputs["mod_rows"].tolist()) == {0, 5}


def test_operator_inputs_are_deterministic_and_round_trip(tmp_path):
    first = op.build_operator_inputs(arch="tiny", frames=22, width=512, height=288,
                                     text_len=16, seed=7)
    second = op.build_operator_inputs(arch="tiny", frames=22, width=512, height=288,
                                      text_len=16, seed=7)
    for key in ("x", "attn_q", "t_emb", "shift", "mlp_x"):
        assert torch.equal(first[key], second[key])

    path = tmp_path / "op_inputs.pt"
    op.save_operator_inputs(str(path), first)
    loaded = op.load_operator_inputs(str(path))
    assert loaded["request"] == first["request"]
    assert torch.equal(loaded["attn_v"], first["attn_v"])


def test_operator_inputs_reject_impossible_row_counts():
    with pytest.raises(SystemExit, match="rows <= kv_rows"):
        op.build_operator_inputs(arch="tiny", frames=22, width=512, height=288,
                                 text_len=16, seed=0, rows=10_000)


def test_operator_inputs_schema_version_is_checked(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"version": 99}, path)
    with pytest.raises(SystemExit, match="operator inputs schema"):
        op.load_operator_inputs(str(path))


# --- partial checkpoint read -------------------------------------------------


@pytest.fixture
def tiny_checkpoint(comfyui_on_syspath, tmp_path):
    state = probe.random_state_dict("tiny", seed=1)
    path = tmp_path / "tiny.safetensors"
    probe.save_state_dict(str(path), state)
    return path


def test_read_block_state_reads_only_that_block(tiny_checkpoint):
    state = op.read_block_state(str(tiny_checkpoint), 1)
    assert "rope.inv_freq" in state
    block_keys = {k for k in state if k.startswith("blocks.")}
    assert block_keys and all(k.startswith("blocks.1.") for k in block_keys)
    assert not any(k.startswith("blocks.0.") for k in state)
    # the projections that every stage needs are there
    local = op.strip_block_prefix(state, 1)
    assert {"attn.qkv_proj.weight", "attn.q_norm.weight", "attn.out_proj.weight",
            "mlp.fc1.weight", "mlp.fc2.weight", "adaln_proj.linear.weight"} <= set(local)


def test_read_block_state_rejects_a_missing_block(tiny_checkpoint):
    with pytest.raises(SystemExit, match="no blocks.49"):
        op.read_block_state(str(tiny_checkpoint), 49)


def test_read_block_state_supports_plain_torch_files(tmp_path, comfyui_on_syspath):
    state = probe.random_state_dict("tiny", seed=2)
    path = tmp_path / "tiny.pt"
    probe.save_state_dict(str(path), state)
    read = op.read_block_state(str(path), 0)
    assert all(k.startswith("blocks.0.") or k == "rope.inv_freq" for k in read)


def test_segments_from_rows_matches_the_models_run_encoding():
    rows = torch.tensor([5, 5, 5, 0, 0, 3])
    assert op._segments_from_rows(rows) == [(0, 3, 5), (3, 5, 0), (5, 6, 3)]
    assert op._segments_from_rows(torch.tensor([2])) == [(0, 1, 2)]


# --- stage comparison --------------------------------------------------------


def _mock_dump(side, *, scale=None, dtype="bf16", block=3, selfcheck=None):
    """A dump shaped exactly like a side's output, with controllable content."""
    generator = torch.Generator().manual_seed(11)
    stages = {}
    for name in ("rope_freqs", "qkv_gemm.q", "attention.out", "mlp.out"):
        stages[name] = torch.randn(8, 4, generator=generator)
    if scale:
        for name, factor in scale.items():
            stages[name] = stages[name] * factor
    metrics = selfcheck if selfcheck is not None else {
        "rel_l2": 0.0, "cosine": 1.0, "max_abs": 0.0, "p99_abs": 0.0,
        "max_abs_over_ref_absmax": 0.0, "p99_abs_over_ref_absmax": 0.0,
        "ref_absmax": 1.0, "ref_l2": 1.0, "elements": 32, "p99_subsampled": False,
    }
    return {
        "version": op.OPERATOR_DUMP_SCHEMA["version"],
        "side": side,
        "block": block,
        "stages": stages,
        "sources": {name: f"{side}.{name}" for name in stages},
        "meta": {"side": side, "block": block, "dtype": dtype, "device": "cpu",
                 "request": {"rows": 8, "kv_rows": 8},
                 "selfcheck.attention": metrics, "runtime": {"torch": torch.__version__}},
    }


def test_identical_operator_dumps_pass_every_stage():
    report = probe.Report(mode="operator", device="cpu")
    op.compare_operator_dumps(report, _mock_dump("comfy"), _mock_dump("raven"),
                              rel_l2_max=1e-9, cos_min=1.0 - 1e-12)
    assert report.passed
    names = {check.name for check in report.checks}
    assert {"stage.rope_freqs", "stage.attention.out", "comfy.selfcheck_attention"} <= names


def test_a_single_bad_stage_is_named_and_fails_alone():
    report = probe.Report(mode="operator", device="cpu")
    op.compare_operator_dumps(report, _mock_dump("comfy", scale={"attention.out": 1.05}),
                              _mock_dump("raven"), rel_l2_max=0.01, cos_min=0.999)
    failed = {c.name for c in report.checks if not c.passed}
    assert failed == {"stage.attention.out"}
    assert not report.passed
    detail = next(c.detail for c in report.checks if c.name == "stage.attention.out")
    assert detail["rel_l2"] == pytest.approx(0.05, rel=1e-3)
    assert detail["cosine"] == pytest.approx(1.0, abs=1e-9)  # scale, not direction


def test_setup_mismatches_are_caught_before_the_numbers():
    report = probe.Report(mode="operator", device="cpu")
    op.compare_operator_dumps(report, _mock_dump("comfy", dtype="fp32"),
                              _mock_dump("raven", dtype="bf16"),
                              rel_l2_max=1.0, cos_min=-1.0)
    assert not report.passed
    assert any(c.name == "setup.same_dtype" and not c.passed for c in report.checks)


def test_a_failed_selfcheck_is_a_gate_not_a_footnote():
    report = probe.Report(mode="operator", device="cpu")
    broken = {"rel_l2": 0.02, "cosine": 0.99, "max_abs": 1.0, "p99_abs": 0.5,
              "max_abs_over_ref_absmax": 1.0, "p99_abs_over_ref_absmax": 0.5,
              "ref_absmax": 1.0, "ref_l2": 1.0, "elements": 32, "p99_subsampled": False}
    op.compare_operator_dumps(report, _mock_dump("comfy", selfcheck=broken),
                              _mock_dump("raven"), rel_l2_max=0.01, cos_min=0.999)
    assert not report.passed
    assert any(c.name == "comfy.selfcheck_attention" and not c.passed for c in report.checks)


def test_selfcheck_errors_are_skipped_not_silently_passed():
    report = probe.Report(mode="operator", device="cpu")
    op.compare_operator_dumps(report, _mock_dump("comfy", selfcheck={"error": "boom"}),
                              _mock_dump("raven"), rel_l2_max=1.0, cos_min=-1.0)
    assert any(item["name"] == "comfy.selfcheck_attention" for item in report.skipped)


def test_stage_metrics_are_recorded_with_their_sources():
    report = probe.Report(mode="operator", device="cpu")
    op.compare_operator_dumps(report, _mock_dump("comfy"), _mock_dump("raven"),
                              rel_l2_max=1.0, cos_min=-1.0)
    stages = {row["stage"] for row in report.metrics}
    assert "attention.out" in stages
    row = next(r for r in report.metrics if r["stage"] == "attention.out")
    assert row["comfy_source"] == "comfy.attention.out"
    assert row["raven_source"] == "raven.attention.out"


# --- CLI on CPU / tiny -------------------------------------------------------


def test_comfy_side_runs_on_cpu_and_its_replay_is_the_production_path(tmp_path, tiny_checkpoint):
    inputs_path = tmp_path / "op_inputs.pt"
    dump_path = tmp_path / "comfy_ops.pt"
    assert op.main(["--side", "inputs", "--arch", "tiny", "--frames", "22",
                    "--width", "512", "--height", "288", "--text-len", "16",
                    "--emit-inputs", str(inputs_path)]) == 0
    assert op.main(["--side", "comfy", "--arch", "tiny", "--device", "cpu",
                    "--dtype", "bf16", "--block", "1",
                    "--weights", str(tiny_checkpoint), "--inputs", str(inputs_path),
                    "--emit-dump", str(dump_path),
                    "--comfyui-path", str(conftest.find_upstream_comfyui())]) == 0

    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    assert dump["side"] == "comfy"
    assert set(op.STAGE_ORDER) == set(dump["stages"])
    # the staged replay must BE Attention.forward, not an approximation of it
    selfcheck = dump["meta"]["selfcheck.attention"]
    assert selfcheck["max_abs"] == 0.0
    assert selfcheck["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert dump["meta"]["runtime"]["torch"] == torch.__version__
    assert "sdp" in dump["meta"]["runtime"]


def test_compare_side_reads_two_dumps_and_reports(tmp_path, tiny_checkpoint):
    inputs_path = tmp_path / "op_inputs.pt"
    comfy_path = tmp_path / "comfy_ops.pt"
    mock_raven_path = tmp_path / "raven_ops.pt"
    report_path = tmp_path / "report.json"
    op.main(["--side", "inputs", "--arch", "tiny", "--frames", "22", "--width", "512",
             "--height", "288", "--text-len", "16", "--emit-inputs", str(inputs_path)])
    op.main(["--side", "comfy", "--arch", "tiny", "--device", "cpu", "--dtype", "bf16",
             "--block", "1", "--weights", str(tiny_checkpoint),
             "--inputs", str(inputs_path), "--emit-dump", str(comfy_path),
             "--comfyui-path", str(conftest.find_upstream_comfyui())])

    # a stand-in for the RAVEN side: same numbers, different producer label
    dump = torch.load(comfy_path, map_location="cpu", weights_only=False)
    dump["side"] = "raven"
    dump["meta"] = dict(dump["meta"], side="raven")
    torch.save(dump, mock_raven_path)

    assert op.main(["--side", "compare", "--block", "1",
                    "--comfy-dump", str(comfy_path), "--raven-dump", str(mock_raven_path),
                    "--json", str(report_path)]) == 0
    payload = json.loads(report_path.read_text())
    assert payload["passed"] is True
    assert payload["meta"]["gate"]["stage_rel_l2_max"] == op.STAGE_REL_L2_MAX
    assert any(row["stage"] == "qk_norm_rope.q" for row in payload["metrics"])


def test_missing_weights_and_inputs_fail_loudly(tmp_path):
    with pytest.raises(SystemExit, match="needs --weights"):
        op.main(["--side", "comfy"])
    with pytest.raises(SystemExit, match="needs --inputs"):
        op.main(["--side", "comfy", "--weights", str(tmp_path / "nope.safetensors")])
    with pytest.raises(SystemExit, match="needs --comfy-dump"):
        op.main(["--side", "compare"])


# --- subprocess isolation ----------------------------------------------------
#
# The vr failure this guards: with the ComfyUI root inherited on PYTHONPATH, the
# RAVEN child resolved ``utils`` to ComfyUI's package and died on
# ``from utils.flash_attn import FlashAttention``. Ordering cannot fix it --
# RAVEN's ``utils`` is a namespace package and ComfyUI's is a regular one, and a
# regular package found later on the path beats a namespace portion found
# earlier -- so the fixtures below reproduce exactly that asymmetry.


@pytest.fixture
def fake_roots(tmp_path):
    """Two checkouts that both answer to ``utils``, as the real ones do."""
    comfy = tmp_path / "fake_comfy"
    (comfy / "comfy").mkdir(parents=True)
    (comfy / "comfy" / "__init__.py").write_text("")
    (comfy / "folder_paths.py").write_text("")
    (comfy / "utils").mkdir()
    # regular package, exactly like ComfyUI's
    (comfy / "utils" / "__init__.py").write_text("SIDE = 'comfy'\n")

    raven = tmp_path / "fake_raven"
    (raven / "projects" / "minimax_h3").mkdir(parents=True)
    (raven / "common").mkdir()
    (raven / "common" / "__init__.py").write_text("")
    (raven / "utils").mkdir()
    # namespace package, exactly like RAVEN's: no __init__.py
    (raven / "utils" / "flash_attn.py").write_text("SIDE = 'raven'\n")
    return comfy, raven


def _plan(tmp_path, side, comfy, raven, *, extra_args=(), inherited=None):
    """A spawn plan for one side, against the fake checkouts."""
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"")
    inputs = tmp_path / "in.pt"
    inputs.write_bytes(b"")
    args = op.parse_args(["--arch", "tiny", "--device", "cpu", "--dtype", "bf16",
                          "--block", "1", "--weights", str(weights),
                          *extra_args])
    return op.build_spawn_plan(args, side, str(inputs), str(tmp_path / f"{side}.pt"),
                               comfy_root=str(comfy), raven_root=str(raven),
                               inherited=inherited)


def _import_side(plan, script):
    done = subprocess.run([sys.executable, "-c", script], env=plan.env, cwd=plan.cwd,
                          capture_output=True, text=True)
    return done


def test_raven_plan_drops_the_comfy_root_that_broke_vr(tmp_path, fake_roots):
    comfy, raven = fake_roots
    plan = _plan(tmp_path, "raven", comfy, raven, inherited=[str(comfy)])
    assert plan.pythonpath[0] == str(raven.resolve())
    assert plan.pythonpath[1] == str(ROOT.resolve())
    assert str(comfy.resolve()) not in plan.pythonpath
    assert plan.dropped == [{"entry": str(comfy), "provides": "utils", "side": "raven"}]
    assert plan.env["PYTHONPATH"].split(os.pathsep)[0] == str(raven.resolve())
    assert "COMFYUI_PATH" not in plan.env


def test_comfy_plan_drops_the_raven_root(tmp_path, fake_roots):
    comfy, raven = fake_roots
    plan = _plan(tmp_path, "comfy", comfy, raven, inherited=[str(raven)])
    assert plan.pythonpath[0] == str(comfy.resolve())
    assert str(raven.resolve()) not in plan.pythonpath
    assert plan.dropped[0]["provides"] in op.CONFLICTING_TOP_LEVEL
    assert plan.env["COMFYUI_PATH"] == str(comfy.resolve())
    assert "RAVEN_ROOT" not in plan.env


def test_harmless_inherited_entries_survive(tmp_path, fake_roots):
    comfy, raven = fake_roots
    extra = tmp_path / "site-packages-ish"
    (extra / "somelib").mkdir(parents=True)
    plan = _plan(tmp_path, "raven", comfy, raven,
                 inherited=[str(comfy), str(extra), ""])
    assert str(extra.resolve()) in plan.pythonpath
    assert [item["entry"] for item in plan.dropped] == [str(comfy)]


def test_the_naive_environment_really_does_break_raven(tmp_path, fake_roots):
    """The regression this guards: without filtering, ``utils`` is ComfyUI's.

    A namespace ``utils`` (RAVEN) loses to a regular ``utils`` (ComfyUI) even
    when RAVEN comes first, which is why ordering was never the fix.
    """
    comfy, raven = fake_roots
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(raven), str(comfy)])
    done = subprocess.run(
        [sys.executable, "-c", "import utils.flash_attn as m; print(m.SIDE)"],
        env=env, cwd=str(tmp_path), capture_output=True, text=True)
    assert done.returncode != 0
    assert "No module named 'utils.flash_attn'" in done.stderr


def test_the_raven_plan_resolves_utils_to_raven(tmp_path, fake_roots):
    comfy, raven = fake_roots
    plan = _plan(tmp_path, "raven", comfy, raven, inherited=[str(comfy)])
    done = _import_side(plan, "import utils.flash_attn as m; print(m.SIDE)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "raven"


def test_the_comfy_plan_resolves_utils_to_comfy(tmp_path, fake_roots):
    comfy, raven = fake_roots
    plan = _plan(tmp_path, "comfy", comfy, raven, inherited=[str(raven)])
    done = _import_side(plan, "import utils; print(utils.SIDE)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "comfy"
    # and the RAVEN-only packages must not be reachable from this side
    missing = _import_side(plan, "import projects")
    assert missing.returncode != 0


def test_each_side_runs_in_its_own_checkout_with_absolute_paths(tmp_path, fake_roots):
    comfy, raven = fake_roots
    comfy_plan = _plan(tmp_path, "comfy", comfy, raven)
    raven_plan = _plan(tmp_path, "raven", comfy, raven)
    assert comfy_plan.cwd == str(comfy.resolve())
    assert raven_plan.cwd == str(raven.resolve())
    for plan in (comfy_plan, raven_plan):
        for flag in ("--weights", "--inputs", "--emit-dump"):
            value = plan.cmd[plan.cmd.index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} must survive a cwd change"
    assert "--comfyui-path" in comfy_plan.cmd
    assert "--raven-root" in raven_plan.cmd


def test_per_side_interpreters(tmp_path, fake_roots):
    comfy, raven = fake_roots
    default = _plan(tmp_path, "raven", comfy, raven)
    assert default.interpreter == sys.executable

    shared = _plan(tmp_path, "raven", comfy, raven,
                   extra_args=["--python", "/usr/bin/python3"])
    assert shared.interpreter == "/usr/bin/python3"
    assert shared.cmd[0] == "/usr/bin/python3"

    split_comfy = _plan(tmp_path, "comfy", comfy, raven,
                        extra_args=["--python", "/usr/bin/python3",
                                    "--comfy-python", "/opt/comfy/bin/python",
                                    "--raven-python", "/opt/raven/bin/python"])
    split_raven = _plan(tmp_path, "raven", comfy, raven,
                        extra_args=["--python", "/usr/bin/python3",
                                    "--comfy-python", "/opt/comfy/bin/python",
                                    "--raven-python", "/opt/raven/bin/python"])
    assert split_comfy.interpreter == "/opt/comfy/bin/python"
    assert split_raven.interpreter == "/opt/raven/bin/python"


def test_plan_description_is_what_the_report_records(tmp_path, fake_roots):
    comfy, raven = fake_roots
    plan = _plan(tmp_path, "raven", comfy, raven, inherited=[str(comfy)])
    described = plan.describe()
    assert described["side"] == "raven"
    assert described["interpreter"] == plan.interpreter
    assert described["cwd"] == str(raven.resolve())
    assert described["pythonpath"] == plan.pythonpath
    assert described["dropped_pythonpath"][0]["provides"] == "utils"


def test_build_spawn_plan_needs_the_side_root(tmp_path, fake_roots):
    comfy, _ = fake_roots
    with pytest.raises(SystemExit, match="needs the raven root"):
        _plan(tmp_path, "raven", comfy, "")


def test_conflicting_package_detection(tmp_path, fake_roots):
    comfy, raven = fake_roots
    assert op.provides_conflicting_package(str(comfy)) in ("utils", "comfy", "folder_paths")
    assert op.provides_conflicting_package(str(raven)) in ("utils", "common", "projects")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert op.provides_conflicting_package(str(plain)) is None
    assert op.provides_conflicting_package(str(tmp_path / "does-not-exist")) is None


# --- the real cross-side run (opt-in) ----------------------------------------


@pytest.mark.skipif(not os.environ.get("RAVEN_ROOT"),
                    reason="set RAVEN_ROOT to a RAVEN checkout to run the real "
                           "operator audit (RAVEN and ComfyUI need separate processes)")
def test_operator_both_sides_cross_check(tmp_path):
    comfyui = conftest.find_upstream_comfyui()
    if comfyui is None:
        pytest.skip("no local ComfyUI checkout")
    weights = tmp_path / "tiny.safetensors"
    report_path = tmp_path / "operator.json"
    # reproduce the vr environment: the parent carries the ComfyUI root on
    # PYTHONPATH, which is what sent the RAVEN child into ComfyUI's ``utils``
    env = dict(os.environ, COMFYUI_PATH=str(comfyui),
               PYTHONPATH=os.pathsep.join(
                   [str(comfyui), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep))

    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "probe_causal_parity.py"),
         "--mode", "inputs", "--arch", "tiny", "--frames", "22", "--width", "512",
         "--height", "288", "--text-len", "16", "--emit-weights", str(weights)],
        check=True, env=env, capture_output=True,
    )
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "probe_causal_operator_parity.py"),
         "--arch", "tiny", "--device", "cpu", "--dtype", "bf16", "--block", "1",
         "--weights", str(weights), "--raven-root", os.environ["RAVEN_ROOT"],
         "--frames", "22", "--width", "512", "--height", "288", "--text-len", "16",
         "--work-dir", str(tmp_path / "work"), "--json", str(report_path)],
        check=False, env=env, capture_output=True, text=True,
    )
    payload = json.loads(report_path.read_text())
    assert payload["passed"], done.stdout + done.stderr
    assert done.returncode == 0
    # both sides must have proven their replay is the production path
    for side in ("comfy", "raven"):
        selfcheck = payload["meta"][side]["selfcheck.attention"]
        assert selfcheck["max_abs"] == 0.0

    # and each side must have run in its own environment
    spawn = payload["meta"]["spawn"]
    assert spawn["raven"]["cwd"] == str(Path(os.environ["RAVEN_ROOT"]).resolve())
    assert spawn["comfy"]["cwd"] == str(comfyui)
    assert str(comfyui) not in spawn["raven"]["pythonpath"]
    assert os.environ["RAVEN_ROOT"] not in spawn["comfy"]["pythonpath"]
    assert spawn["raven"]["dropped_pythonpath"], "the polluted entry must be reported"
    assert "[raven] PYTHONPATH:" in done.stdout
