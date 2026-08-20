"""``tools/probe_causal_embedding_parity.py``: the path upstream of block 0.

Block 0's Q/K/V are a linear function of the block's *input*, so when the
operator probe says the block internals are exact and the rollout still shows
~1% at block 0, the answer is in the embedding path. What has to be right here:

* only the embedding subtrees are read out of the checkpoint (a 66 GB file must
  cost one block, not a model);
* the taps read *inputs* through pre-hooks -- the DiT block writes its attention
  residual into its own input buffer, so a forward hook reading ``args`` returns
  a mutated tensor. That bug was real, and there is a regression test for it;
* an fp32 island that was cast through bf16 is detected by **value**, not by
  dtype, because a round trip leaves the dtype saying ``float32``;
* the comparison names the stage that moved and turns it into a concrete
  suggestion.

The RAVEN side needs its own process and a checkout, so the cross-side run is
opt-in through ``RAVEN_ROOT``.
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
import probe_causal_embedding_parity as emb  # noqa: E402
import probe_causal_parity as probe  # noqa: E402

TINY_REQUEST = dict(frames=22, width=512, height=288, text_len=16)


@pytest.fixture
def tiny_checkpoint(comfyui_on_syspath, tmp_path):
    state = probe.random_state_dict("tiny", seed=3)
    path = tmp_path / "tiny.safetensors"
    probe.save_state_dict(str(path), state)
    return path


@pytest.fixture
def tiny_inputs(comfyui_on_syspath, tmp_path):
    inputs = probe.build_shared_inputs(
        seed=5, sink=2, window=2, video_sigma=0.6, audio_sigma=None, arch="tiny",
        **TINY_REQUEST)
    path = tmp_path / "inputs.pt"
    probe.save_inputs(str(path), inputs)
    return path


# --- partial checkpoint read -------------------------------------------------


def test_only_the_embedding_subtrees_are_read(tiny_checkpoint):
    state = emb.read_embedding_state(str(tiny_checkpoint))
    assert "rope.inv_freq" in state
    assert any(k.startswith("video_patch_proj.") for k in state)
    assert any(k.startswith("token_refiner.") for k in state)
    assert any(k.startswith("blocks.0.") for k in state)
    # the rest of the stack must stay on disk
    assert not any(k.startswith("blocks.1.") for k in state)
    assert not any(k.startswith("blocks.2.") for k in state)


def test_a_checkpoint_without_the_subtrees_fails_loudly(tmp_path, comfyui_on_syspath):
    partial = {k: v for k, v in probe.random_state_dict("tiny", seed=1).items()
               if k.startswith("blocks.")}
    path = tmp_path / "no_embedding.safetensors"
    probe.save_state_dict(str(path), partial)
    with pytest.raises(SystemExit, match="missing the embedding subtree"):
        emb.read_embedding_state(str(path))


def test_place_weights_respects_declared_dtypes():
    state = {"w": torch.randn(4, 4, dtype=torch.float32)}
    target = {"w": torch.zeros(4, 4, dtype=torch.float32)}
    declared = emb.place_weights(state, target, fp32_island="declared")
    assert torch.equal(declared["w"], state["w"])

    degraded = emb.place_weights(state, target, fp32_island="bf16")
    assert degraded["w"].dtype == torch.float32          # dtype unchanged ...
    assert not torch.equal(degraded["w"], state["w"])    # ... values are not
    assert torch.equal(degraded["w"], state["w"].to(torch.bfloat16).to(torch.float32))


# --- taps --------------------------------------------------------------------


class _InPlaceResidualBlock(torch.nn.Module):
    """A block that adds into its own input buffer, exactly like the DiT one."""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.Identity()

    def forward(self, x):
        x.add_(1.0)  # the residual write that made a forward hook lie
        return x


def test_input_taps_are_pre_hooks_and_survive_an_in_place_block():
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([_InPlaceResidualBlock()])
    tap = emb.ModuleTap()
    tap.attach(model, specs=(("blocks.0", "in"), ("blocks.0.norm1", "out")))
    tap.phase = "text"
    x = torch.zeros(2, 3)
    with tap:
        model.blocks[0](x)
    assert float(tap.stages["text/blocks.0.in"].abs().max()) == 0.0, (
        "the tap must see the block input, not the residual the block wrote into it"
    )
    assert "forward pre-hook" in tap.sources["text/blocks.0.in"]


def test_taps_are_tagged_by_phase_and_handle_tuple_outputs():
    class _TupleLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.inner(x), None  # RAVEN's linears return (out, bias)

    model = torch.nn.Module()
    model.proj = _TupleLinear()
    tap = emb.ModuleTap().attach(model, specs=(("proj", "in_out"),))
    with tap:
        tap.phase = "text"
        model.proj(torch.randn(2, 3))
        tap.phase = "chunk0"
        model.proj(torch.randn(4, 3))
    assert set(tap.stages) == {"text/proj.in", "text/proj.out",
                               "chunk0/proj.in", "chunk0/proj.out"}
    assert tap.stages["chunk0/proj.out"].shape == (4, 3)


def test_repeated_calls_in_one_phase_do_not_overwrite_each_other():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(3, 3)
    tap = emb.ModuleTap().attach(model, specs=(("proj", "out"),))
    tap.phase = "chunk0"
    with tap:
        model.proj(torch.randn(2, 3))
        model.proj(torch.randn(2, 3))
    assert "chunk0/proj.out" in tap.stages
    assert any(name.startswith("chunk0/proj.out#") for name in tap.stages)


def test_derived_bf16_cast_stage_is_added():
    tap = emb.ModuleTap()
    tap.stages["chunk0/video_patch_proj.out"] = torch.tensor([[1.0009765, 2.5]])
    tap.dtypes["chunk0/video_patch_proj.out"] = "torch.float32"
    tap.sources["chunk0/video_patch_proj.out"] = "x"
    emb._derive_casts(tap, "bf16")
    cast = tap.stages["chunk0/video_patch_proj.cast_bf16"]
    assert torch.equal(cast, tap.stages["chunk0/video_patch_proj.out"]
                       .to(torch.bfloat16).to(torch.float32))


# --- comparison --------------------------------------------------------------


def _dump(side, *, stages=None, fingerprints=None, dtype="bf16", taps=4,
          math_sdp=False, control=None, backend="sdpa"):
    generator = torch.Generator().manual_seed(9)
    refined = torch.randn(4, 8, generator=generator)
    base = {
        "env/sdpa_control.q": torch.ones(2, 2),
        "env/sdpa_control.k": torch.ones(2, 2),
        "env/sdpa_control.v": torch.ones(2, 2),
        "env/sdpa_control.out": (control if control is not None else torch.full((2, 2), 0.5)),
        # the real invariant: the text rows entering the stack are the refiner's
        # own output, so the mock has to satisfy it too
        "text/token_refiner.final_norm.out": refined,
        "text/blocks.0.in": refined.clone(),
        "chunk0/video_patch_proj.out": torch.randn(4, 8, generator=generator),
        "chunk0/blocks.0.attn.qkv_proj.in": torch.randn(4, 8, generator=generator),
        "chunk0/blocks.0.adaln_proj.linear.in": torch.randn(2, 8, generator=generator),
    }
    if stages:
        base.update(stages)
    prints = fingerprints if fingerprints is not None else {
        "video_patch_proj.weight": {"dtype": "torch.float32",
                                    "bf16_exact_fraction": 0.5, "absmax": 1.0},
    }
    return {
        "version": emb.EMBEDDING_DUMP_SCHEMA["version"],
        "side": side,
        "stages": base,
        "dtypes": {name: "torch.bfloat16" for name in base},
        "sources": {name: f"{side}.{name}" for name in base},
        "meta": {"side": side, "dtype": dtype, "request": {"text_len": 16},
                 "taps_fired": taps, "weight_dtypes": {"video_patch_proj.weight": "torch.float32"},
                 "weight_fingerprints": prints, "runtime": {"torch": torch.__version__},
                 "sdp_backend": {"requested": "auto"},
                 "attention": {"backend": backend},
                 "math_sdp_reduced_precision": {"requested": "auto", "allowed": math_sdp}},
    }


def test_identical_dumps_pass_every_stage():
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy"), _dump("raven"),
                                rel_l2_max=1e-9, cos_min=1.0 - 1e-12)
    assert report.passed
    names = {c.name for c in report.checks}
    assert "stage.chunk0/video_patch_proj.out" in names
    assert "setup.same_weight_values" in names


def test_a_degraded_fp32_island_is_caught_by_value_not_dtype():
    degraded = {"video_patch_proj.weight": {"dtype": "torch.float32",
                                            "bf16_exact_fraction": 1.0, "absmax": 1.0}}
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", fingerprints=degraded),
                                _dump("raven"), rel_l2_max=1.0, cos_min=-1.0)
    assert not report.passed
    values = next(c for c in report.checks if c.name == "setup.same_weight_values")
    assert not values.passed
    assert values.detail["video_patch_proj.weight"] == {"comfy": 1.0, "raven": 0.5}
    # the dtype check, on its own, sees nothing wrong
    assert next(c for c in report.checks if c.name == "setup.same_weight_dtypes").passed
    intact = next(c for c in report.checks if c.name == "comfy.fp32_island_intact")
    assert not intact.passed and intact.gate is False
    assert any(item["module"] == "loader" for item in report.meta["suggestions"])


def test_a_moved_stage_is_named_and_turned_into_a_suggestion():
    moved = {"chunk0/blocks.0.adaln_proj.linear.in": torch.full((2, 8), 3.0)}
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", stages=moved), _dump("raven"),
                                rel_l2_max=0.005, cos_min=0.999)
    failed = {c.name for c in report.checks if not c.passed and c.gate}
    assert failed == {"stage.chunk0/blocks.0.adaln_proj.linear.in"}
    suggestion = next(item for item in report.meta["suggestions"]
                      if item["module"] == "blocks.0.adaln_proj.linear.in")
    assert "SiLU" in suggestion["suggestion"]


def test_ranked_contributors_order_the_search():
    stages = {
        "chunk0/video_patch_proj.out": torch.zeros(4, 8),
        "text/blocks.0.in": torch.zeros(4, 8),
    }
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", stages=stages), _dump("raven"),
                                rel_l2_max=10.0, cos_min=-1.0)
    ranked = report.meta["ranked_contributors"]
    assert ranked[0]["rel_l2"] >= ranked[-1]["rel_l2"]
    assert {row["stage"] for row in ranked[:2]} == set(stages)


def test_one_sided_stages_are_listed_not_silently_dropped():
    report = probe.Report(mode="embedding", device="cpu")
    raven = _dump("raven", stages={"text/video_patch_proj.out": torch.zeros(2, 2)})
    emb.compare_embedding_dumps(report, _dump("comfy"), raven,
                                rel_l2_max=1.0, cos_min=-1.0)
    assert report.meta["one_sided_stages"]["raven_only"] == ["text/video_patch_proj.out"]
    assert any(item["name"] == "stages.one_sided" for item in report.skipped)


def test_within_side_consistency_is_checked():
    """The text rows entering the stack must be the refiner's own output."""
    broken = {"text/blocks.0.in": torch.full((4, 8), 7.0)}
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", stages=broken),
                                _dump("raven", stages=broken),
                                rel_l2_max=1.0, cos_min=-1.0)
    for side in ("comfy", "raven"):
        check = next(c for c in report.checks
                     if c.name == f"{side}.text_decoder_input_is_refiner_output")
        assert not check.passed


# --- refiner staging, ordering and the SDPA seam ------------------------------


def test_refiner_tap_specs_cover_every_stage_of_a_block():
    specs = dict(op_specs := emb.refiner_tap_specs(2))
    assert len(op_specs) == 18  # nine leaves per block, two blocks
    for index in (0, 1):
        base = f"token_refiner.blocks.{index}"
        assert specs[f"{base}.norm1"] == "in_out"        # block input + norm1 out
        assert specs[f"{base}.norm2"] == "in_out"        # attention residual
        assert specs[f"{base}.attn.qkv_proj"] == "out"
        assert specs[f"{base}.attn.q_norm"] == "out"
        assert specs[f"{base}.attn.k_norm"] == "out"
        assert specs[f"{base}.attn.out_proj"] == "in_out"
        assert specs[f"{base}.mlp.fc1"] == "out"
        assert specs[f"{base}.mlp.fc2"] == "in"          # the SwiGLU output
        assert specs[f"{base}.mlp"] == "out"


def test_execution_order_follows_the_forward_pass():
    stages = [
        "text/token_refiner.blocks.1.norm1.in",
        "text/token_refiner.blocks.0.attn.sdpa.out",
        "text/token_refiner.blocks.0.norm1.in",
        "text/condition_proj.out",
        "chunk0/video_patch_proj.out",
        "text/token_refiner.final_norm.out",
        "text/token_refiner.blocks.0.attn.q_norm.out",
    ]
    ordered = sorted(stages, key=emb.execution_key)
    assert ordered == [
        "text/condition_proj.out",
        "text/token_refiner.blocks.0.norm1.in",
        "text/token_refiner.blocks.0.attn.q_norm.out",
        "text/token_refiner.blocks.0.attn.sdpa.out",
        "text/token_refiner.blocks.1.norm1.in",
        "text/token_refiner.final_norm.out",
        "chunk0/video_patch_proj.out",
    ]


def test_first_divergence_reports_the_earliest_stage_not_the_largest():
    metrics = [
        {"stage": "text/condition_proj.out", "rel_l2": 0.0, "cosine": 1.0},
        {"stage": "text/token_refiner.blocks.0.attn.q_norm.out", "rel_l2": 0.0,
         "cosine": 1.0},
        {"stage": "text/token_refiner.blocks.0.attn.sdpa.out", "rel_l2": 2.6e-3,
         "cosine": 0.999997},
        {"stage": "text/token_refiner.blocks.0.mlp.out", "rel_l2": 9.0e-1,
         "cosine": 0.5},
    ]
    found = emb.first_divergence(metrics, rel_l2_max=0.005)
    assert found["first_nonzero"]["stage"].endswith("attn.sdpa.out")
    assert found["exact_prefix"] == ["text/condition_proj.out",
                                     "text/token_refiner.blocks.0.attn.q_norm.out"]
    assert found["first_over_gate"]["stage"].endswith("mlp.out")


class _FakeSeamModule:
    def __init__(self):
        self.calls = 0

        def seam(q, k, v, **kwargs):
            self.calls += 1
            return q * 2

        seam.__name__ = "fake_seam"
        self.seam = seam


def test_seam_tap_records_inputs_and_output_then_restores():
    module = _FakeSeamModule()
    original = module.seam
    tap = emb.ModuleTap()
    tap.phase = "text"
    q = torch.ones(4, 2, 3)
    with emb.SeamTap(tap, module, "seam", "text_refiner") as seam:
        module.seam(q, q, q, scale=0.5, site=("text_refiner", 1))
    assert module.seam is original, "the real callable must come back"
    assert module.calls == 1 and seam.calls == 1
    base = "text/token_refiner.blocks.1.attn.sdpa."
    assert {base + p for p in ("q", "k", "v", "out")} <= set(tap.stages)
    assert torch.equal(tap.stages[base + "out"], q * 2)
    call = tap.seam_calls[0]
    assert call["site"] == ["text_refiner", 1]
    assert call["tensors"]["q"]["contiguous"] is True
    assert call["scale"] == 0.5


def test_seam_tap_restores_even_when_the_body_raises():
    module = _FakeSeamModule()
    original = module.seam
    tap = emb.ModuleTap()
    with pytest.raises(RuntimeError):
        with emb.SeamTap(tap, module, "seam", "dit"):
            raise RuntimeError("boom")
    assert module.seam is original


def test_seam_output_is_copied_before_the_caller_can_mutate_it():
    """The seam's return is often a buffer the caller writes into afterwards."""
    module = _FakeSeamModule()
    tap = emb.ModuleTap()
    tap.phase = "text"
    with emb.SeamTap(tap, module, "seam", "dit"):
        out = module.seam(torch.ones(2, 2, 2), torch.ones(2, 2, 2), torch.ones(2, 2, 2))
    out.zero_()  # exactly what an in-place residual would do
    assert float(tap.stages["text/dit.0.attn.sdpa.out"].abs().max()) == 2.0


def test_seam_site_resolution_prefers_the_label_then_the_stack_then_a_counter():
    module = _FakeSeamModule()
    tap = emb.ModuleTap()
    tap.phase = "text"
    seam = emb.SeamTap(tap, module, "seam", "dit")
    # 1. the caller's own label wins
    assert seam.resolve_site(("text_refiner", 3)) == ("text_refiner", 3)
    # 2. a pending site of a *different* stack is ignored (this is what kept a
    #    DiT call from inheriting the refiner's block index)
    tap.pending_site = ("text_refiner", 1)
    assert seam.resolve_site(None) == ("dit", 0)
    # 3. a pending site of the same stack supplies the index
    refiner_seam = emb.SeamTap(tap, module, "seam", "text_refiner")
    assert refiner_seam.resolve_site(None) == ("text_refiner", 1)
    # 4. otherwise a per-phase counter
    tap.pending_site = None
    assert seam.resolve_site(None) == ("dit", 1)
    tap.phase = "chunk0"
    assert seam.resolve_site(None) == ("dit", 0)


def test_qk_norm_taps_set_the_pending_site():
    tap = emb.ModuleTap()
    tap.phase = "text"
    tap.record("token_refiner.blocks.1.attn.k_norm.out", torch.zeros(2, 2), "x")
    assert tap.pending_site == ("text_refiner", 1)
    tap.record("blocks.0.attn.q_norm.out", torch.zeros(2, 2), "x")
    assert tap.pending_site == ("dit", 0)


def test_qkv_split_derives_q_k_v_from_the_fused_output():
    tap = emb.ModuleTap()
    fused = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3 * 4 * 3)
    tap.stages["text/token_refiner.blocks.0.attn.qkv_proj.out"] = fused
    tap.dtypes["text/token_refiner.blocks.0.attn.qkv_proj.out"] = "torch.bfloat16"
    emb._derive_qkv_splits(tap, heads=3, head_dim=4)
    base = "text/token_refiner.blocks.0.attn.qkv_proj.out."
    assert tap.stages[base + "q"].shape == (2, 3, 4)
    assert torch.equal(tap.stages[base + "q"], fused[:, :12].reshape(2, 3, 4))
    assert torch.equal(tap.stages[base + "v"], fused[:, 24:].reshape(2, 3, 4))
    assert "upstream's own" in tap.sources[base + "k"]


# --- the environment control -------------------------------------------------


def test_sdpa_control_is_deterministic_and_recorded_under_env():
    first, second = emb.ModuleTap(), emb.ModuleTap()
    for tap in (first, second):
        tap.phase = "chunk0"
        emb.sdpa_control(tap, torch.device("cpu"), torch.bfloat16)
        assert tap.phase == "chunk0", "the control must not leak its phase"
    for name in ("q", "k", "v", "out"):
        assert torch.equal(first.stages[f"env/sdpa_control.{name}"],
                           second.stages[f"env/sdpa_control.{name}"])


def test_math_sdp_reduced_precision_flag_moves_the_control():
    """The measured cause: a process-global torch flag ComfyUI sets at import."""
    getter = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
    if getter is None:
        pytest.skip("this torch build has no math-SDP reduction switch")
    before = bool(getter())
    try:
        outputs = {}
        for choice in ("off", "on"):
            applied = emb.apply_math_sdp_precision(choice)
            assert applied["allowed"] is (choice == "on")
            tap = emb.ModuleTap()
            emb.sdpa_control(tap, torch.device("cpu"), torch.bfloat16)
            outputs[choice] = tap.stages["env/sdpa_control.out"]
        assert not torch.equal(outputs["on"], outputs["off"]), (
            "if this ever passes, the flag stopped mattering and the probe's "
            "headline finding needs re-measuring"
        )
    finally:
        emb.apply_math_sdp_precision("on" if before else "off")


def test_control_mismatch_is_reported_but_does_not_decide_the_run():
    """The lane disables the reduction around its own SDPA call and restores it.

    So the two *processes* can still differ on the global flag while every lane
    stage is bit-identical. The control says so; it does not gate.
    """
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(
        report,
        _dump("comfy", math_sdp=True, control=torch.full((2, 2), 0.4)),
        _dump("raven", math_sdp=False),
        rel_l2_max=1.0, cos_min=-1.0)
    control = next(c for c in report.checks if c.name == "env.sdpa_control_matches")
    assert not control.passed and control.gate is False
    assert control.detail["scope"] == "process-level control, outside the lane"
    flag = next(c for c in report.checks
                if c.name == "env.math_sdp_reduced_precision_matches")
    assert not flag.passed and flag.gate is False
    assert "per call" in flag.detail["note"]
    assert report.passed, "a process-level flag must not fail the run on its own"


def test_matching_control_passes_and_inputs_are_checked_separately():
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy"), _dump("raven"),
                                rel_l2_max=1.0, cos_min=-1.0)
    assert next(c for c in report.checks
                if c.name == "env.sdpa_control_inputs_identical").passed
    assert next(c for c in report.checks if c.name == "env.sdpa_control_matches").passed


def test_same_backend_gates_attention_stages():
    moved = {"text/token_refiner.blocks.0.attn.sdpa.out": torch.full((4, 8), 9.0)}
    reference = _dump("raven", backend="sdpa")
    reference["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.zeros(4, 8)
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", stages=moved, backend="sdpa"),
                                reference, rel_l2_max=0.005, cos_min=0.999)
    assert report.meta["attention_backends"]["match"] is True
    check = next(c for c in report.checks
                 if c.name == "stage.text/token_refiner.blocks.0.attn.sdpa.out")
    assert check.gate is True and not check.passed
    assert check.detail["classification"] == "unexplained"
    assert not report.passed


def test_different_backends_classify_downstream_as_kernel_float_error():
    """FA3 vs SDPA is arithmetic. It must be named, not gated, and not implied
    to be a logic error."""
    moved = {"text/token_refiner.blocks.0.attn.sdpa.out": torch.full((4, 8), 9.0)}
    reference = _dump("raven", backend="fa3")
    reference["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.zeros(4, 8)
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, _dump("comfy", stages=moved, backend="sdpa"),
                                reference, rel_l2_max=0.005, cos_min=0.999)

    backends = report.meta["attention_backends"]
    assert backends == dict(backends, comfy="sdpa", raven="fa3", match=False)
    match_check = next(c for c in report.checks if c.name == "env.attention_backend_matches")
    assert not match_check.passed and match_check.gate is False

    check = next(c for c in report.checks
                 if c.name == "stage.text/token_refiner.blocks.0.attn.sdpa.out")
    assert check.gate is False, "a kernel difference must not gate"
    assert check.passed is True
    assert check.detail["classification"] == "kernel_float_error"
    assert check.detail["backends"] == "sdpa vs fa3"
    assert "not evidence of a logic difference" in report.meta["parity_scope"]
    assert report.passed

    # everything upstream of the kernel is still gated and still exact
    upstream = next(c for c in report.checks
                    if c.name == "stage.chunk0/video_patch_proj.out")
    assert upstream.gate is True and upstream.passed
    comparable = report.meta["first_divergence_comparable"]["first_nonzero"]
    assert comparable is None


def test_sdpa_seam_inputs_stay_gated_when_backends_differ():
    """``attn.sdpa.q/k/v`` are upstream of the kernel; no kernel excuses them."""
    moved = {"text/token_refiner.blocks.0.attn.sdpa.q": torch.full((4, 8), 5.0)}
    reference = _dump("raven", backend="fa3")
    reference["stages"]["text/token_refiner.blocks.0.attn.sdpa.q"] = torch.zeros(4, 8)
    comfy = _dump("comfy", stages=moved, backend="sdpa")
    comfy["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.zeros(4, 8)
    reference["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.zeros(4, 8)
    report = probe.Report(mode="embedding", device="cpu")
    emb.compare_embedding_dumps(report, comfy, reference, rel_l2_max=0.005, cos_min=0.999)
    check = next(c for c in report.checks
                 if c.name == "stage.text/token_refiner.blocks.0.attn.sdpa.q")
    assert check.gate is True and not check.passed
    assert not report.passed


@pytest.mark.parametrize(
    "rel_l2,same_backend,dependent,expected",
    [
        (0.0, True, True, "exact"),
        (0.0, False, True, "exact"),
        (1e-8, True, False, "float_error"),
        (2e-3, False, True, "kernel_float_error"),
        (2e-3, False, False, "unexplained"),
        (2e-3, True, True, "unexplained"),
    ],
)
def test_stage_classification(rel_l2, same_backend, dependent, expected):
    assert emb.classify_stage({"rel_l2": rel_l2}, same_backend=same_backend,
                              attention_dependent=dependent) == expected


def test_control_stages_are_classified_as_process_control():
    assert emb.classify_stage({"rel_l2": 4e-3}, same_backend=True,
                              attention_dependent=False,
                              is_control=True) == "process_control"


@pytest.mark.parametrize(
    "stage,dependent",
    [
        # produced by an attention call, or computed from one
        ("text/token_refiner.blocks.0.attn.sdpa.out", True),
        ("text/token_refiner.blocks.0.attn.out_proj.out", True),
        ("text/token_refiner.blocks.0.norm2.in", True),
        ("text/token_refiner.blocks.0.mlp.out", True),
        ("text/token_refiner.final_norm.out", True),
        ("text/token_refiner.blocks.1.norm1.in", True),   # second block's input
        ("text/blocks.0.in", True),                       # text prefill feeds the stack
        ("text/dit.0.attn.sdpa.out", True),
        # upstream of every attention call, whatever the kernels are
        ("text/token_refiner.blocks.0.attn.sdpa.q", False),
        ("text/token_refiner.blocks.0.attn.qkv_proj.out", False),
        ("text/token_refiner.blocks.0.attn.q_norm.out", False),
        ("text/token_refiner.blocks.0.norm1.in", False),
        ("text/condition_proj.out", False),
        ("chunk0/video_patch_proj.out", False),
        ("chunk0/blocks.0.in", False),                    # chunk rows come from the patch proj
        ("chunk0/time_embedder.out", False),
        ("text/time_embedder.out", False),
        ("text/blocks.0.adaln_proj.linear.in", False),    # comes from the time embedder
        ("env/sdpa_control.out", False),
    ],
)
def test_attention_dependency_is_declared_not_inferred_from_order(stage, dependent):
    assert emb.is_attention_dependent(stage) is dependent


def test_normalised_backend_reads_either_dump_shape():
    assert emb.normalised_backend({"attention": {"backend": "fa2"}}) == "fa2"
    assert emb.normalised_backend(
        {"attention": {"flash_attn": {"backend": "fa3"}}}) == "fa3"
    assert emb.normalised_backend(
        {"attention": {"lane_backend": {"backend": "sdpa"}}}) == "sdpa"
    assert emb.normalised_backend({"attention": {}}) is None
    assert emb.normalised_backend({}) is None


def test_sdpa_stage_gets_the_measured_suggestion():
    moved = {"chunk0/blocks.0.adaln_proj.linear.in": torch.zeros(2, 8)}
    report = probe.Report(mode="embedding", device="cpu")
    dump = _dump("comfy", stages=moved)
    dump["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.full((4, 8), 9.0)
    reference = _dump("raven")
    reference["stages"]["text/token_refiner.blocks.0.attn.sdpa.out"] = torch.zeros(4, 8)
    emb.compare_embedding_dumps(report, dump, reference, rel_l2_max=0.005, cos_min=0.999)
    suggestion = next(item for item in report.meta["suggestions"]
                      if item["module"] == "attn.sdpa")
    assert "allow_fp16_bf16_reduction_math_sdp" in suggestion["suggestion"]


# --- CLI on CPU / tiny -------------------------------------------------------


def test_comfy_side_runs_on_cpu_and_taps_the_real_forwards(tmp_path, tiny_checkpoint,
                                                           tiny_inputs):
    dump_path = tmp_path / "comfy_emb.pt"
    assert emb.main(["--side", "comfy", "--arch", "tiny", "--device", "cpu",
                     "--dtype", "bf16", "--weights", str(tiny_checkpoint),
                     "--inputs", str(tiny_inputs), "--emit-dump", str(dump_path),
                     "--comfyui-path", str(conftest.find_upstream_comfyui())]) == 0
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    assert dump["side"] == "comfy"
    assert dump["meta"]["entry_points"] == [
        "RavenCausalMiniMaxH3Model.prefill_text",
        "RavenCausalMiniMaxH3Model.forward_chunk",
    ]
    for stage in ("text/condition_proj.out", "text/token_refiner.final_norm.out",
                  "text/blocks.0.in", "text/blocks.0.attn.qkv_proj.in",
                  "chunk0/video_patch_proj.in", "chunk0/video_patch_proj.out",
                  "chunk0/video_patch_proj.cast_bf16", "chunk0/audio_patch_proj.out",
                  "chunk0/time_embedder.proj_in.in", "chunk0/time_embedder.out",
                  "chunk0/blocks.0.adaln_proj.linear.in", "chunk0/blocks.0.norm1.out",
                  "chunk0/blocks.0.attn.qkv_proj.in"):
        assert stage in dump["stages"], stage
    # the patch projections are the checkpoint's fp32 island and must stay fp32
    assert dump["dtypes"]["chunk0/video_patch_proj.in"] == "torch.float32"
    assert dump["dtypes"]["chunk0/video_patch_proj.out"] == "torch.float32"
    assert dump["dtypes"]["chunk0/blocks.0.attn.qkv_proj.in"] == "torch.bfloat16"
    assert dump["meta"]["weight_fingerprints"]["video_patch_proj.weight"]["dtype"] \
        == "torch.float32"


def test_declared_and_cast_islands_differ_measurably(tmp_path, tiny_checkpoint,
                                                     tiny_inputs):
    """The knob has to actually move the tensors it claims to move."""
    dumps = {}
    for mode in ("declared", "bf16"):
        path = tmp_path / f"comfy_{mode}.pt"
        emb.main(["--side", "comfy", "--arch", "tiny", "--device", "cpu",
                  "--dtype", "bf16", "--comfy-fp32-island", mode,
                  "--weights", str(tiny_checkpoint), "--inputs", str(tiny_inputs),
                  "--emit-dump", str(path),
                  "--comfyui-path", str(conftest.find_upstream_comfyui())])
        dumps[mode] = torch.load(path, map_location="cpu", weights_only=False)

    declared = dumps["declared"]["meta"]["weight_fingerprints"]["video_patch_proj.weight"]
    cast = dumps["bf16"]["meta"]["weight_fingerprints"]["video_patch_proj.weight"]
    assert declared["dtype"] == cast["dtype"] == "torch.float32"
    assert cast["bf16_exact_fraction"] == 1.0
    assert declared["bf16_exact_fraction"] < 1.0
    metrics = probe.tensor_metrics(dumps["bf16"]["stages"]["chunk0/video_patch_proj.out"],
                                   dumps["declared"]["stages"]["chunk0/video_patch_proj.out"])
    assert metrics["rel_l2"] > 0.0


def test_compare_side_reads_two_dumps(tmp_path, tiny_checkpoint, tiny_inputs):
    comfy_path = tmp_path / "comfy_emb.pt"
    mock_raven = tmp_path / "raven_emb.pt"
    report_path = tmp_path / "embedding.json"
    emb.main(["--side", "comfy", "--arch", "tiny", "--device", "cpu", "--dtype", "bf16",
              "--weights", str(tiny_checkpoint), "--inputs", str(tiny_inputs),
              "--emit-dump", str(comfy_path),
              "--comfyui-path", str(conftest.find_upstream_comfyui())])
    dump = torch.load(comfy_path, map_location="cpu", weights_only=False)
    dump["side"] = "raven"
    dump["meta"] = dict(dump["meta"], side="raven")
    torch.save(dump, mock_raven)

    assert emb.main(["--side", "compare", "--comfy-dump", str(comfy_path),
                     "--raven-dump", str(mock_raven), "--json", str(report_path)]) == 0
    payload = json.loads(report_path.read_text())
    assert payload["passed"] is True
    assert payload["meta"]["gate"]["stage_rel_l2_max"] == emb.STAGE_REL_L2_MAX
    assert payload["meta"]["ranked_contributors"][0]["rel_l2"] == 0.0


def test_missing_arguments_fail_loudly(tmp_path):
    with pytest.raises(SystemExit, match="needs --weights"):
        emb.main(["--side", "comfy"])
    with pytest.raises(SystemExit, match="needs --inputs"):
        emb.main(["--side", "comfy", "--weights", str(tmp_path / "w.safetensors")])
    with pytest.raises(SystemExit, match="needs --comfy-dump"):
        emb.main(["--side", "compare"])


def test_side_both_uses_the_isolated_spawn_plans(tmp_path, fake_roots_for_embedding):
    """The embedding probe reuses the operator probe's env isolation."""
    comfy, raven = fake_roots_for_embedding
    args = emb.parse_args(["--arch", "tiny", "--device", "cpu", "--dtype", "bf16",
                           "--weights", str(tmp_path / "w.safetensors"),
                           "--inputs", str(tmp_path / "in.pt")])
    (tmp_path / "w.safetensors").write_bytes(b"")
    (tmp_path / "in.pt").write_bytes(b"")
    from probe_causal_operator_parity import build_spawn_plan

    plan = build_spawn_plan(args, "raven", str(tmp_path / "in.pt"),
                            str(tmp_path / "raven.pt"),
                            comfy_root=str(comfy), raven_root=str(raven),
                            inherited=[str(comfy)],
                            script=str(Path(emb.__file__).resolve()))
    assert plan.cmd[1].endswith("probe_causal_embedding_parity.py")
    assert "--block" not in plan.cmd
    assert str(comfy.resolve()) not in plan.pythonpath
    assert plan.cwd == str(raven.resolve())


@pytest.fixture
def fake_roots_for_embedding(tmp_path):
    comfy = tmp_path / "fake_comfy"
    (comfy / "comfy").mkdir(parents=True)
    (comfy / "folder_paths.py").write_text("")
    (comfy / "utils").mkdir()
    (comfy / "utils" / "__init__.py").write_text("SIDE = 'comfy'\n")
    raven = tmp_path / "fake_raven"
    (raven / "projects" / "minimax_h3").mkdir(parents=True)
    (raven / "utils").mkdir()
    (raven / "utils" / "flash_attn.py").write_text("SIDE = 'raven'\n")
    return comfy, raven


# --- the real cross-side run (opt-in) ----------------------------------------


@pytest.mark.skipif(not os.environ.get("RAVEN_ROOT"),
                    reason="set RAVEN_ROOT to a RAVEN checkout to run the real "
                           "embedding audit (separate processes)")
def test_embedding_both_sides_cross_check(tmp_path):
    """Both sides, for real, twice -- judged by which attention kernel each ran.

    The causal lane now dispatches FA3 -> FA2 -> SDPA like RAVEN's wrapper, and
    its SDPA step disables/restores the math-SDPA reduction flag per call. So
    the rule is no longer "auto must diverge at the seam", it is:

    * **same backend on both sides** -> every shared attention/refiner stage is
      bit-identical (or inside the gate), and the only stage allowed to move at
      all is the time embedder, at ~1e-8 (``1 - sigma`` as a python float on one
      side and a tensor op on the other);
    * **different backends** -> the dump must name both, and the stages from the
      first attention call on must be classified ``kernel_float_error`` and
      reported without gating. A kernel's float error is not a logic error.

    Run 2 pins the reduction off on both sides; with the same kernel that must
    still be bit-identical.
    """
    comfyui = conftest.find_upstream_comfyui()
    if comfyui is None:
        pytest.skip("no local ComfyUI checkout")
    weights = tmp_path / "tiny.safetensors"
    inputs = tmp_path / "inputs.pt"
    env = dict(os.environ, COMFYUI_PATH=str(comfyui),
               PYTHONPATH=os.pathsep.join(
                   [str(comfyui), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep))

    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "probe_causal_parity.py"), "--mode", "inputs",
         "--arch", "tiny", "--frames", "22", "--width", "512", "--height", "288",
         "--text-len", "16", "--emit-inputs", str(inputs), "--emit-weights", str(weights)],
        check=True, env=env, capture_output=True)

    def run(mode: str):
        report_path = tmp_path / f"embedding_{mode}.json"
        done = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "probe_causal_embedding_parity.py"),
             "--arch", "tiny", "--device", "cpu", "--dtype", "bf16",
             "--math-sdp-reduced-precision", mode,
             "--weights", str(weights), "--inputs", str(inputs),
             "--raven-root", os.environ["RAVEN_ROOT"],
             "--work-dir", str(tmp_path / f"work_{mode}"), "--json", str(report_path)],
            check=False, env=env, capture_output=True, text=True)
        return json.loads(report_path.read_text()), done

    #: stages that must be exact whatever the kernels are: they are upstream of
    #: the first attention call, including the seam's own q/k/v inputs
    upstream = ("text/token_refiner.blocks.0.norm1.in",
                "text/token_refiner.blocks.0.norm1.out",
                "text/token_refiner.blocks.0.attn.qkv_proj.out",
                "text/token_refiner.blocks.0.attn.qkv_proj.out.q",
                "text/token_refiner.blocks.0.attn.qkv_proj.out.k",
                "text/token_refiner.blocks.0.attn.qkv_proj.out.v",
                "text/token_refiner.blocks.0.attn.q_norm.out",
                "text/token_refiner.blocks.0.attn.k_norm.out",
                "text/token_refiner.blocks.0.attn.sdpa.q",
                "text/token_refiner.blocks.0.attn.sdpa.k",
                "text/token_refiner.blocks.0.attn.sdpa.v",
                "chunk0/video_patch_proj.out", "chunk0/audio_patch_proj.out")

    def assert_report(payload, stdout, *, mode):
        backends = payload["meta"]["attention_backends"]
        assert backends["comfy"] is not None and backends["raven"] is not None, (
            f"[{mode}] the dump must name both backends: {backends}"
        )
        stages = {row["stage"]: row for row in payload["metrics"]}
        for stage in upstream:
            assert stages[stage]["rel_l2"] == 0.0, f"[{mode}] {stage}"

        if backends["match"]:
            # same kernel: attention and refiner must agree bit for bit, and the
            # only stage allowed to move at all is the time embedder
            for stage, row in stages.items():
                if "token_refiner" in stage or "attn.sdpa" in stage \
                        or "patch_proj" in stage:
                    assert row["rel_l2"] == 0.0, f"[{mode}] {stage}"
            first = payload["meta"]["first_divergence"]["first_nonzero"]
            assert first is None or (first["stage"].endswith("time_embedder.out")
                                     and first["rel_l2"] < 1e-6), f"[{mode}] {first}"
            assert payload["passed"], stdout
        else:
            # different kernels: named, classified, reported -- never implied to
            # be a logic difference, and never gating
            assert "kernel" in payload["meta"]["parity_scope"]
            assert "not evidence of a logic difference" in payload["meta"]["parity_scope"]
            downstream = [row for row in payload["metrics"]
                          if row.get("classification") == "kernel_float_error"]
            assert downstream, f"[{mode}] a backend split must classify its stages"
            for row in downstream:
                check = next(c for c in payload["checks"]
                             if c["name"] == f"stage.{row['stage']}")
                assert check["gate"] is False and check["passed"], f"[{mode}] {row['stage']}"
            comparable = payload["meta"]["first_divergence_comparable"]["first_nonzero"]
            assert comparable is None or comparable["rel_l2"] < 1e-6, (
                f"[{mode}] a stage upstream of attention moved: {comparable}"
            )
        return backends

    production, done = run("auto")
    backends = assert_report(production, done.stdout + done.stderr, mode="auto")
    assert production["meta"]["raven"]["entry_points"][0].endswith("_text_cache_fill")
    assert str(comfyui) not in production["meta"]["spawn"]["raven"]["pythonpath"]
    assert next(c for c in production["checks"]
                if c["name"] == "setup.same_weight_values")["passed"]

    pinned, done = run("off")
    pinned_backends = assert_report(pinned, done.stdout + done.stderr, mode="off")
    if pinned_backends["match"]:
        assert pinned["passed"] and done.returncode == 0
    # the reduction flag is a process fact, never the verdict
    for name in ("env.sdpa_control_matches", "env.math_sdp_reduced_precision_matches"):
        check = next((c for c in pinned["checks"] if c["name"] == name), None)
        if check is not None:
            assert check["gate"] is False, name
    assert backends["comfy"] == pinned_backends["comfy"]
