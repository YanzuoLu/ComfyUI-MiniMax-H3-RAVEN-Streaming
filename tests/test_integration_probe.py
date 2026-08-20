"""``tools/probe_raven_integration.py`` without the 71 GB of weights.

The probe's job is to run the real plugin on real models, which no test box
here has. What *can* be pinned locally is everything the probe decides:

* the CLI surface, and that a failed probe still writes its report and exits 1;
* the ``17k + 5`` geometry it builds latents on, and the deterministic
  conditioning it stands in for the text encoder;
* the preview envelope checks -- fed **real** fMP4 produced by the real muxer
  and pushed through a real ``PreviewSession``, so "moof + mdat" and "the
  stream decodes" are claims about actual bytes, not about a fixture;
* the metric and determinism gates, including that a near-miss FAILS rather
  than being absorbed into a tolerance;
* the atomic report write, and that the report path cannot escape the repo;
* the cancel and repeat logic, driven through ``execute_run`` with a fake
  sampler that still uses the real preview lane, the real muxer and the real
  streaming pipeline.

What is deliberately *not* tested here: that the real DiT loads, samples and
decodes. That is the probe's own output on a GPU box, and it is unverified
until someone runs it there.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from raven_streaming import layout as layout_mod  # noqa: E402
from raven_streaming import nodes as nodes_mod  # noqa: E402
from raven_streaming import preview as preview_mod  # noqa: E402
from raven_streaming import streaming_pipeline as pipeline_mod  # noqa: E402
from raven_streaming.consistency import ChunkOutput, SamplingCancelled  # noqa: E402

PROBE_PATH = os.path.join(ROOT, "tools", "probe_raven_integration.py")

# This module must never touch the process-global RNG. Other tests in this
# suite draw unseeded tensors and then assert *bitwise* results, so consuming
# (or reseeding) the global stream from here would change their inputs and make
# them fail for reasons that have nothing to do with them. Everything random in
# this file therefore comes from a private generator.
_RNG = torch.Generator(device="cpu")
_RNG.manual_seed(0x5241564E)


def _randn(*shape, seed=None):
    if seed is not None:
        _RNG.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=_RNG)


def _rand(*shape, seed=None):
    if seed is not None:
        _RNG.manual_seed(int(seed))
    return torch.rand(tuple(shape), generator=_RNG)


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_raven_integration", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


@pytest.fixture
def repo_report_dir(request):
    """A scratch directory *inside* the repo (the probe refuses anything else).

    ``.cache/`` is gitignored, which is where a run artifact belongs; the
    directory is removed again so a test run leaves nothing behind.
    """
    import shutil

    path = os.path.join(ROOT, ".cache", "probe_integration_tests", str(os.getpid()))
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(os.path.join(ROOT, ".cache", "probe_integration_tests"), ignore_errors=True)


# ==========================================================================
# CLI
# ==========================================================================


REQUIRED_ARGS = [
    "--base", "/models/base.safetensors",
    "--lora", "/models/lora.safetensors",
    "--video-vae", "/models/video.safetensors",
    "--audio-vae", "/models/audio.safetensors",
]


def test_cli_defaults_are_the_documented_ones(probe):
    args = probe.build_parser().parse_args(REQUIRED_ARGS)
    assert (args.width, args.height, args.frames) == (512, 288, 39)
    assert args.text_len == 128
    assert args.steps == 4
    assert (args.video_shift, args.audio_shift) == (12.0, 3.0)
    assert (args.sink, args.window) == (2, 2)
    assert args.repeat == 1
    assert args.cancel_after_forward == 0
    assert args.json == probe.DEFAULT_REPORT
    assert args.weight_dtype == "default"


def test_cli_requires_every_model_path(probe, capsys):
    for drop in ("--base", "--lora", "--video-vae", "--audio-vae"):
        argv = list(REQUIRED_ARGS)
        index = argv.index(drop)
        del argv[index : index + 2]
        with pytest.raises(SystemExit):
            probe.build_parser().parse_args(argv)
        assert drop in capsys.readouterr().err


def test_cli_accepts_the_full_flag_set(probe):
    args = probe.build_parser().parse_args(
        REQUIRED_ARGS
        + [
            "--comfy-root", "/opt/ComfyUI",
            "--width", "256", "--height", "256", "--frames", "22",
            "--text-len", "16", "--seed", "7", "--steps", "2",
            "--video-shift", "5", "--audio-shift", "1.5",
            "--sink", "1", "--window", "0",
            "--device", "cuda", "--repeat", "2",
            "--cancel-after-forward", "3",
            "--json", ".cache/report.json",
        ]
    )
    assert args.comfy_root == "/opt/ComfyUI"
    assert (args.width, args.height, args.frames, args.text_len) == (256, 256, 22, 16)
    assert (args.seed, args.steps) == (7, 2)
    assert (args.video_shift, args.audio_shift) == (5.0, 1.5)
    assert (args.sink, args.window) == (1, 0)
    assert args.device == "cuda"
    assert args.repeat == 2
    assert args.cancel_after_forward == 3
    assert args.json == ".cache/report.json"


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
def test_cli_accepts_the_whole_repeat_range(probe, count):
    args = probe.build_parser().parse_args(REQUIRED_ARGS + ["--repeat", str(count)])
    assert args.repeat == count


@pytest.mark.parametrize("bad", ["0", "-1", "11", "100", "two"])
def test_cli_rejects_a_repeat_outside_the_range(probe, bad, capsys):
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(REQUIRED_ARGS + ["--repeat", bad])
    assert "--repeat" in capsys.readouterr().err


def test_cli_compare_official_video_is_off_by_default(probe):
    args = probe.build_parser().parse_args(REQUIRED_ARGS)
    assert args.compare_official_video is False
    args = probe.build_parser().parse_args(REQUIRED_ARGS + ["--compare-official-video"])
    assert args.compare_official_video is True


def test_cli_help_states_what_the_probe_does_not_verify(probe, capsys):
    with pytest.raises(SystemExit):
        probe.main(["--help"])
    text = capsys.readouterr().out
    # argparse re-wraps every help string, so long claims are matched against
    # the whitespace-normalised text rather than against a particular wrapping
    flat = " ".join(text.split())
    assert "never downloads" in flat
    assert "no text encoder is loaded" in flat
    assert "synthetic" in flat
    assert "--cancel-after-forward N" in text
    # the optional text lane, and what it costs
    assert "--text-encoder PATH" in text
    assert "--prompt TEXT" in text
    assert "NVFP4" in flat and "host RAM" in flat
    assert "128 GB" in flat
    # the plateau story is documented where the flag is
    assert "plateau" in flat
    # the optional whole-clip reference decode, and why it is bounded
    assert "--compare-official-video" in text
    assert "OOMed a measured 39-frame run" in flat
    assert "362-frame requests are refused, not attempted" in flat
    assert "An OOM here is reported as a FAILURE" in flat


def test_docstring_declares_the_text_encoder_caveat(probe):
    doc = probe.__doc__ or ""
    assert "no text encoder is loaded" in doc
    assert "T2VA" in doc and "unload_model_and_clones" in doc
    assert "17k + 5" in doc
    assert "PyAV" in doc and "ffmpeg" in doc
    assert "warm-up from a leak" in doc
    # where the IMAGE comes from now, and what the optional diagnostic costs
    assert "finalize_image" in doc
    assert "--compare-official-video" in doc
    assert "prepare_final_decode" in doc


def test_run_plan_puts_the_cancelled_run_first(probe):
    assert probe.run_plan(repeat=1, cancel_after_forward=0) == ["sample"]
    assert probe.run_plan(repeat=2, cancel_after_forward=0) == ["sample", "sample"]
    assert probe.run_plan(repeat=1, cancel_after_forward=2) == ["cancel", "sample"]
    assert probe.run_plan(repeat=2, cancel_after_forward=5) == ["cancel", "sample", "sample"]
    assert probe.run_plan(repeat=10, cancel_after_forward=0) == ["sample"] * 10
    assert probe.run_plan(repeat=3, cancel_after_forward=1) == ["cancel"] + ["sample"] * 3


def test_run_plan_carries_a_chunk_cancel_the_same_way(probe):
    assert probe.run_plan(repeat=1, cancel_after_forward=0, cancel_after_chunk=0) == ["sample"]
    assert probe.run_plan(repeat=1, cancel_after_forward=0, cancel_after_chunk=1) == [
        "cancel_chunk", "sample",
    ]
    assert probe.run_plan(repeat=3, cancel_after_forward=0, cancel_after_chunk=2) == [
        "cancel_chunk", "sample", "sample", "sample",
    ]
    # the CLI refuses both, but the function must still pick exactly one run
    assert probe.run_plan(repeat=1, cancel_after_forward=2, cancel_after_chunk=2) == [
        "cancel", "sample",
    ]


# ==========================================================================
# --kv-cache-storage: offered by the probe, honoured by the node
# ==========================================================================


def test_cli_kv_cache_storage_defaults_to_the_nodes_own_default(probe):
    args = probe.build_parser().parse_args(REQUIRED_ARGS)
    assert args.kv_cache_storage == probe.DEFAULT_KV_CACHE_STORAGE == "cpu_pinned"


@pytest.mark.parametrize("mode", ["cpu_pinned", "cpu", "gpu"])
def test_cli_accepts_every_storage_mode(probe, mode):
    args = probe.build_parser().parse_args(REQUIRED_ARGS + ["--kv-cache-storage", mode])
    assert args.kv_cache_storage == mode


def test_cli_refuses_a_storage_mode_the_node_does_not_have(probe, capsys):
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(REQUIRED_ARGS + ["--kv-cache-storage", "disk"])
    assert "--kv-cache-storage" in capsys.readouterr().err


def test_the_probes_storage_choices_are_the_nodes(probe):
    """The parser's copy has to stay the node's list, or a run dies after the build."""
    assert set(probe.KV_CACHE_STORAGE_CHOICES) == set(nodes_mod.KV_CACHE_STORAGE_CHOICES)
    assert probe.DEFAULT_KV_CACHE_STORAGE == nodes_mod.DEFAULT_KV_CACHE_STORAGE


# ==========================================================================
# --cancel-after-chunk: the later cancellation point
# ==========================================================================


def test_cli_cancel_after_chunk_is_off_by_default(probe):
    args = probe.build_parser().parse_args(REQUIRED_ARGS)
    assert args.cancel_after_chunk == 0
    args = probe.build_parser().parse_args(REQUIRED_ARGS + ["--cancel-after-chunk", "2"])
    assert args.cancel_after_chunk == 2
    assert args.cancel_after_forward == 0


def test_cli_refuses_both_cancellation_points(probe, capsys):
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(
            REQUIRED_ARGS + ["--cancel-after-forward", "1", "--cancel-after-chunk", "1"]
        )
    assert "not allowed with argument" in capsys.readouterr().err


def test_the_two_cancel_points_expect_different_dit_load_counts(probe):
    """A chunk cancel lands before the coordinator reloads the DiT; a forward
    cancel lands after it. One off-by-one either way is a reported failure."""
    # 5 chunks, cancelled with 2 delivered
    assert probe.dit_loads_after_cancel(2, 5) == 3
    assert probe.dit_loads_after_chunk_cancel(2, 5) == 2
    # nothing delivered yet: only the rollout's own first load happened
    assert probe.dit_loads_after_cancel(0, 5) == 1
    assert probe.dit_loads_after_chunk_cancel(0, 5) == 1
    assert probe.dit_loads_after_chunk_cancel(1, 5) == 1
    # the last chunk never reloads, whichever way the run stopped
    assert probe.dit_loads_after_cancel(5, 5) == 5
    assert probe.dit_loads_after_chunk_cancel(5, 5) == 5


# ==========================================================================
# --stacked-lora-name / --stacked-lora-strength: the CLI half
# ==========================================================================


def test_cli_stacked_lora_is_off_by_default(probe):
    args = probe.build_parser().parse_args(REQUIRED_ARGS)
    assert args.stacked_lora_name is None
    assert args.stacked_lora_strength is None
    assert probe.stacked_lora_requested(args) is False


def test_cli_accepts_the_stacked_lora_flags(probe):
    args = probe.build_parser().parse_args(
        REQUIRED_ARGS
        + ["--stacked-lora-name", "extra/style.safetensors", "--stacked-lora-strength", "0.75"]
    )
    assert args.stacked_lora_name == "extra/style.safetensors"
    assert args.stacked_lora_strength == 0.75
    assert probe.stacked_lora_requested(args) is True
    assert probe.stacked_lora_strength(args) == 0.75


def test_the_stacked_strength_defaults_to_the_nodes_own(probe):
    args = probe.build_parser().parse_args(
        REQUIRED_ARGS + ["--stacked-lora-name", "style.safetensors"]
    )
    assert args.stacked_lora_strength is None
    assert probe.stacked_lora_strength(args) == 1.0


def test_main_refuses_a_stacked_strength_without_a_lora(probe, capsys):
    with pytest.raises(SystemExit):
        probe.main(REQUIRED_ARGS + ["--stacked-lora-strength", "0.5"])
    assert "needs --stacked-lora-name" in capsys.readouterr().err


def test_main_refuses_a_zero_stacked_strength(probe, capsys):
    """At strength 0 the official node returns the model untouched."""
    with pytest.raises(SystemExit):
        probe.main(
            REQUIRED_ARGS
            + ["--stacked-lora-name", "style.safetensors", "--stacked-lora-strength", "0"]
        )
    assert "refused" in capsys.readouterr().err


def test_help_documents_the_three_new_lanes(probe, capsys):
    with pytest.raises(SystemExit):
        probe.main(["--help"])
    text = capsys.readouterr().out
    flat = " ".join(text.split())
    assert "--stacked-lora-name NAME" in text
    assert "--stacked-lora-strength F" in text
    assert "LoraLoaderModelOnly" in flat
    assert "RELATIVE name" in flat
    assert "--kv-cache-storage" in text
    assert "--cancel-after-chunk N" in text
    assert "DELIVERED" in flat


# ==========================================================================
# the two lanes are mutually exclusive
# ==========================================================================


def test_text_lane_needs_both_flags(probe):
    parse = probe.build_parser().parse_args
    assert probe.text_lane_requested(parse(REQUIRED_ARGS)) is False
    assert probe.text_lane_requested(
        parse(REQUIRED_ARGS + ["--text-encoder", "/te.safetensors", "--prompt", "a cat"])
    ) is True


@pytest.mark.parametrize(
    "half", [["--text-encoder", "/te.safetensors"], ["--prompt", "a cat"]]
)
def test_main_refuses_half_a_text_lane(probe, half, capsys):
    with pytest.raises(SystemExit):
        probe.main(REQUIRED_ARGS + half)
    error = capsys.readouterr().err
    assert "all-or-nothing" in error


# ==========================================================================
# report path and atomic write
# ==========================================================================


def test_report_path_must_stay_inside_the_repository(probe, tmp_path):
    inside = probe.resolve_report_path(".cache/x.json")
    assert str(inside).startswith(ROOT)
    with pytest.raises(probe.ProbeError, match="must be inside"):
        probe.resolve_report_path(str(tmp_path / "escape.json"))
    with pytest.raises(probe.ProbeError):
        probe.resolve_report_path("../outside.json")


def test_report_path_accepts_an_alternative_root(probe, tmp_path):
    assert probe.resolve_report_path(str(tmp_path / "r.json"), root=tmp_path) == (
        tmp_path / "r.json"
    )


def test_atomic_write_leaves_no_partial_file(probe, repo_report_dir):
    path = os.path.join(repo_report_dir, "nested", "report.json")
    probe.atomic_write_json(path, {"ok": True, "value": 1})
    assert json.loads(open(path).read())["value"] == 1
    probe.atomic_write_json(path, {"ok": False, "value": 2})
    assert json.loads(open(path).read())["value"] == 2
    # the temporary is gone: a reader only ever sees a whole report
    assert [n for n in os.listdir(os.path.dirname(path)) if ".tmp-" in n] == []


def test_atomic_write_survives_non_json_values(probe, repo_report_dir):
    path = os.path.join(repo_report_dir, "report.json")
    probe.atomic_write_json(path, {"dtype": torch.float32, "device": torch.device("cpu")})
    payload = json.loads(open(path).read())
    assert payload["dtype"] == "torch.float32"


def test_main_writes_a_report_and_exits_1_when_the_probe_raises(probe, repo_report_dir):
    out = os.path.join(repo_report_dir, "failed.json")
    code = probe.main(
        REQUIRED_ARGS
        + ["--comfy-root", os.path.join(repo_report_dir, "no-comfy-here"), "--json", out]
    )
    assert code == 1
    payload = json.loads(open(out).read())
    assert payload["ok"] is False
    assert payload["errors"] and "no ComfyUI checkout" in payload["errors"][0]
    # the arguments are echoed, so a report identifies the run that produced it
    assert payload["args"]["frames"] == 39
    assert payload["args"]["json"] == out


def test_a_crash_keeps_the_checks_made_before_it(probe, monkeypatch):
    """A probe that dies half way still reports what it had established."""
    env = probe.ComfyEnv(
        root="/fake",
        comfy=SimpleNamespace(
            model_management=SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
        ),
        folder_paths=None,
        upstream_nodes=None,
    )
    monkeypatch.setattr(probe, "import_comfy", lambda root: env)

    def boom(*args, **kwargs):
        raise RuntimeError("geometry exploded")

    monkeypatch.setattr(probe, "latent_geometry", boom)
    args = probe.build_parser().parse_args(REQUIRED_ARGS + ["--device", "cpu"])
    report = probe.Report()
    with pytest.raises(RuntimeError, match="geometry exploded"):
        probe.run_probe(args, report)
    assert report.environment["device"] == "cpu"
    assert any("environment" in check["name"] for check in report.checks)
    assert report.notes


def test_main_rejects_a_report_path_outside_the_repo(probe, tmp_path, capsys):
    with pytest.raises(SystemExit):
        probe.main(REQUIRED_ARGS + ["--json", str(tmp_path / "x.json")])
    assert "must be inside" in capsys.readouterr().err


# ==========================================================================
# geometry and inputs
# ==========================================================================


def test_geometry_is_the_17k_plus_5_grid(probe):
    assert probe.latent_geometry(39, 512, 288) == {
        "frames": 39, "k": 2, "width": 512, "height": 288,
        "latent_t": 12, "latent_h": 18, "latent_w": 32, "audio_t": 65,
    }
    assert probe.latent_geometry(22, 64, 64)["latent_t"] == 7
    assert probe.latent_geometry(192, 1376, 768)["k"] == 11


def test_geometry_refuses_an_off_grid_request(probe):
    with pytest.raises(layout_mod.LayoutError, match="17k \\+ 5"):
        probe.latent_geometry(40, 512, 288)
    with pytest.raises(layout_mod.LayoutError, match="k=0"):
        probe.latent_geometry(5, 512, 288)
    with pytest.raises(layout_mod.LayoutError, match="multiple of 32"):
        probe.latent_geometry(39, 500, 288)


class FakeNestedTensor:
    """The two things ``contracts`` reads off ``comfy.nested_tensor``."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def test_empty_latent_is_the_official_shape_and_is_empty(probe):
    geometry = probe.latent_geometry(39, 512, 288)
    latent = probe.build_empty_latent(geometry, FakeNestedTensor)
    video, audio = latent["samples"].unbind()
    assert list(video.shape) == [1, 24, 12, 18, 32]
    assert list(audio.shape) == [1, 32, 2, 65]
    assert not video.any() and not audio.any()
    assert list(latent.keys()) == ["samples"]


def test_empty_latent_is_accepted_by_the_node_contract(probe):
    from raven_streaming import contracts

    geometry = probe.latent_geometry(22, 64, 64)
    latent = probe.build_empty_latent(geometry, FakeNestedTensor)
    request = contracts.parse_latent(latent)
    assert request.frames == 22
    assert (request.latent_t, request.audio_t) == (7, 37)


def test_conditioning_is_deterministic_and_seed_dependent(probe):
    first, describe = probe.build_conditioning(16, 0)
    again, describe_again = probe.build_conditioning(16, 0)
    other, describe_other = probe.build_conditioning(16, 1)

    # the same seed is the same tensor, bitwise -- checked on the tensors
    # themselves, which is the only thing that settles it
    assert torch.equal(first[0][0], again[0][0])
    assert torch.equal(first[0][1]["minimax_token_tags"], again[0][1]["minimax_token_tags"])
    assert not torch.equal(first[0][0], other[0][0])
    assert describe["seed"] == describe_again["seed"] == 0
    assert describe_other["seed"] == 1
    assert describe["shape"] == [1, 16, probe.TEXT_EMBED_DIM]
    assert describe["device"] == "cpu"
    assert "text encoder" in describe["caveat"]


def test_the_conditioning_report_describes_the_draw_without_hashing_it(probe):
    """Shape, dtype, seed and generator -- everything needed to redraw it."""
    _, describe = probe.build_conditioning(16, 7)
    assert describe["dtype"] == "torch.float32"
    assert describe["seed"] == 7
    assert "manual_seed" in describe["generator"]
    assert set(describe) == {
        "source", "shape", "dtype", "device", "generator", "seed",
        "token_tags", "caveat",
    }
    assert set(describe["token_tags"]) == {"value", "shape", "dtype"}
    assert describe["token_tags"]["shape"] == [16]
    assert describe["token_tags"]["dtype"] == "torch.int64"


def test_conditioning_is_accepted_by_the_node_contract(probe):
    from raven_streaming import contracts

    conditioning, _describe = probe.build_conditioning(9, 3)
    parsed = contracts.parse_conditioning(conditioning)
    assert parsed.text_len == 9
    assert parsed.token_tags is not None
    assert set(parsed.token_tags.tolist()) == {layout_mod.TEXT_TAG}


def test_the_two_generators_are_recorded_separately(probe):
    """The conditioning draw and the rollout draw are different generators."""
    _cond, conditioning = probe.build_conditioning(4, 11)
    rollout = probe.describe_rollout_rng(11, torch.device("cuda:0"))
    assert conditioning["device"] == "cpu"
    assert rollout["generator_device"] == "cuda:0"
    assert rollout["source"].endswith("RolloutRNG")
    assert rollout["private"] is True
    assert rollout["seed"] == 11


def test_conditioning_refuses_an_empty_prompt(probe):
    with pytest.raises(probe.ProbeError, match="text_len"):
        probe.build_conditioning(0, 0)


def test_geometry_cross_check_skips_without_upstream(probe, monkeypatch):
    checks = probe.Checks()
    monkeypatch.setitem(sys.modules, "comfy_extras", None)
    probe.check_geometry_against_upstream(probe.latent_geometry(22, 64, 64), checks)
    assert checks.ok
    assert "skipped" in checks.items[0].name


# ==========================================================================
# the optional real text lane, driven against fake official nodes
# ==========================================================================


class FakeFolderPaths:
    """The four ``folder_paths`` entry points the probe is allowed to use."""

    def __init__(self, shadow=None):
        self.folders = {}
        self.shadow = dict(shadow or {})
        self.filename_list_cache = {"vae": (["stale"], {}, 0.0),
                                    "text_encoders": (["stale"], {}, 0.0)}
        self.registered = []
        self.cleared = 0

    def add_model_folder_path(self, folder, path, is_default=False):
        self.registered.append((folder, path, is_default))
        paths = self.folders.setdefault(folder, [])
        if is_default:
            paths.insert(0, path)
        else:
            paths.append(path)

    def get_full_path(self, folder, name):
        if (folder, name) in self.shadow:
            return self.shadow[(folder, name)]
        for directory in self.folders.get(folder, []):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        return None


def _touch(path, content=b"weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_register_model_files_returns_basenames_and_registers_dirs(probe, tmp_path):
    first = _touch(tmp_path / "a" / "video.safetensors")
    second = _touch(tmp_path / "b" / "audio.safetensors")
    folder_paths = FakeFolderPaths()
    names = probe.register_model_files(folder_paths, "vae", [first, second])
    assert names == ["video.safetensors", "audio.safetensors"]
    assert folder_paths.registered == [
        ("vae", str(tmp_path / "a"), True),
        ("vae", str(tmp_path / "b"), True),
    ]
    # the stale combo cache for that folder is dropped, others are untouched
    assert "vae" not in folder_paths.filename_list_cache
    assert "text_encoders" in folder_paths.filename_list_cache


def test_register_model_files_registers_the_text_encoder_folder(probe, tmp_path):
    encoder = _touch(tmp_path / "te" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    folder_paths = FakeFolderPaths()
    names = probe.register_model_files(folder_paths, probe.TEXT_ENCODER_FOLDER, [encoder])
    assert names == [encoder.name]
    assert folder_paths.registered == [("text_encoders", str(tmp_path / "te"), True)]


def test_register_model_files_catches_a_shadowing_file(probe, tmp_path):
    encoder = _touch(tmp_path / "mine" / "qwen.safetensors")
    other = _touch(tmp_path / "theirs" / "qwen.safetensors")
    folder_paths = FakeFolderPaths(
        shadow={("text_encoders", "qwen.safetensors"): str(other)}
    )
    with pytest.raises(probe.ProbeError, match="shadowing"):
        probe.register_model_files(folder_paths, "text_encoders", [encoder])


def test_register_model_files_refuses_a_missing_file(probe, tmp_path):
    with pytest.raises(probe.ProbeError, match="nothing is downloaded"):
        probe.register_model_files(
            FakeFolderPaths(), "text_encoders", [tmp_path / "absent.safetensors"]
        )


# ==========================================================================
# the stacked standard LoRA: the pieces, then the real official node
# ==========================================================================


def test_file_info_is_the_resolved_path_and_size(probe, tmp_path):
    path = _touch(tmp_path / "loras" / "style.safetensors", b"not really a lora")
    info = probe.file_info(path)
    assert info["path"] == str(path.resolve())
    assert info["bytes"] == len(b"not really a lora")
    assert set(info) == {"path", "bytes"}


def test_file_info_does_not_read_the_file(probe, tmp_path, monkeypatch):
    """A LoRA can be gigabytes; the size comes from the metadata, not a read."""
    blob = bytes(range(256)) * 40
    path = _touch(tmp_path / "big.safetensors", blob)

    real_open = builtins.open

    def refuse(file, *args, **kwargs):
        if pathlib.Path(str(file)) in (path, path.resolve()):
            raise AssertionError("file_info must not open the file it describes")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse)
    info = probe.file_info(path)
    assert info["bytes"] == len(blob)
    assert info["path"] == str(path.resolve())


class _Counted:
    """Something with a method worth counting, that must still run."""

    calls = []

    def work(self, name, strength):
        _Counted.calls.append((name, strength))
        return ("did", name, strength)


def test_count_calls_counts_calls_through_and_restores(probe):
    _Counted.calls = []
    original = _Counted.work
    with probe.count_calls(_Counted, "work") as counter:
        assert _Counted().work("a.safetensors", 0.5) == ("did", "a.safetensors", 0.5)
        assert _Counted().work("b.safetensors", 1.0) == ("did", "b.safetensors", 1.0)
    assert counter.calls == 2
    # the original really ran, both times
    assert _Counted.calls == [("a.safetensors", 0.5), ("b.safetensors", 1.0)]
    assert _Counted.work is original


def test_count_calls_keeps_the_scalars_and_drops_the_tensors(probe):
    """A counter that outlived the call must not keep a LoRA state dict alive."""
    _Counted.calls = []
    big = torch.zeros(4, 4)
    with probe.count_calls(_Counted, "work") as counter:
        _Counted().work(big, 0.25)
    assert counter.calls == 1
    (recorded,) = counter.scalar_args
    assert 0.25 in recorded
    assert not any(isinstance(item, torch.Tensor) for item in recorded)


def test_capture_lora_warnings_sees_comfys_own_unmatched_key_reports(probe):
    import logging

    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    with probe.capture_lora_warnings() as messages:
        logging.warning("lora key not loaded: %s", "lora_unet_nope.lora_up.weight")
        logging.warning("NOT LOADED %s", "diffusion_model.nope.weight")
        logging.warning("something else entirely")
    assert root.handlers == before_handlers and root.level == before_level
    unmatched = probe.unmatched_lora_messages(messages)
    assert len(unmatched) == 2
    assert "lora_unet_nope.lora_up.weight" in unmatched[0]
    assert probe.unmatched_lora_messages(["something else entirely"]) == []


def test_capture_lora_warnings_lowers_a_silenced_root_and_puts_it_back(probe):
    import logging

    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.CRITICAL)
    try:
        with probe.capture_lora_warnings() as messages:
            logging.warning("lora key not loaded: x")
        assert probe.unmatched_lora_messages(messages)
        assert root.level == logging.CRITICAL
    finally:
        root.setLevel(previous)


def _lora_patch(up=None, down=None, strength_patch=1.0, strength_model=1.0):
    """One entry of ``ModelPatcher.patches[key]``, in upstream's own shape."""
    up = torch.ones(4, 1) if up is None else up
    down = torch.ones(1, 8) if down is None else down
    adapter = SimpleNamespace(weights=(up, down, 1.0, None, None, None))
    return (strength_patch, adapter, strength_model, None, None)


def test_describe_patch_reads_a_real_lora_entry(probe):
    described = probe.describe_patch(_lora_patch())
    assert described["nonzero"] is True
    assert described["finite"] is True
    assert described["tensors"] == [[1, 8], [4, 1]] or described["tensors"] == [[4, 1], [1, 8]]
    assert described["strength_patch"] == 1.0


def test_a_lora_whose_up_is_zero_contributes_nothing(probe):
    """A freshly initialised adapter is exactly this, and it changes no pixel."""
    described = probe.describe_patch(_lora_patch(up=torch.zeros(4, 1)))
    assert described["nonzero"] is False


@pytest.mark.parametrize("kwargs", [{"strength_patch": 0.0}, {"strength_model": 0.0}])
def test_a_patch_applied_at_zero_strength_is_not_a_hit(probe, kwargs):
    assert probe.describe_patch(_lora_patch(**kwargs))["nonzero"] is False


def test_describe_patch_understands_the_plain_diff_tuple(probe):
    entry = (1.0, ("diff", (torch.ones(2, 2),)), 1.0, None, None)
    described = probe.describe_patch(entry)
    assert described["kind"] == "diff"
    assert described["nonzero"] is True
    assert described["tensors"] == [[2, 2]]


def _patcher_with_patches(patches, state_keys=()):
    return SimpleNamespace(
        patches=patches,
        model=SimpleNamespace(state_dict=lambda: {k: torch.zeros(1) for k in state_keys}),
    )


def test_summarise_stacked_patches_attributes_only_what_was_added(probe):
    base_key = "diffusion_model.blocks.0.attn.out_proj.weight"
    other = "diffusion_model.blocks.1.mlp.fc1.weight"
    patcher = _patcher_with_patches(
        {base_key: [_lora_patch(), _lora_patch()], other: [_lora_patch()]},
        state_keys=(base_key, other),
    )
    summary = probe.summarise_stacked_patches(
        {base_key: 1}, patcher, residual_prefixes=("raven_lora_A_", "raven_lora_B_")
    )
    assert summary["target_keys"] == sorted([base_key, other])
    assert summary["patch_count"] == 2  # one new on the first key, one on the second
    assert summary["nonzero_base_key_hits"] == sorted([base_key, other])
    assert summary["residual_targets"] == []
    assert summary["targets_outside_the_state_dict"] == []


def test_summarise_stacked_patches_flags_a_residual_target(probe):
    residual = "diffusion_model.blocks.0.attn.out_proj.raven_lora_A_0"
    patcher = _patcher_with_patches({residual: [_lora_patch()]}, state_keys=(residual,))
    summary = probe.summarise_stacked_patches(
        {}, patcher, residual_prefixes=probe.raven_residual_prefixes()
    )
    assert summary["residual_targets"] == [residual]
    assert summary["nonzero_base_key_hits"] == []


def test_summarise_stacked_patches_separates_zero_patches(probe):
    key = "diffusion_model.final_layer.video_out.weight"
    patcher = _patcher_with_patches(
        {key: [_lora_patch(up=torch.zeros(4, 1))]}, state_keys=(key,)
    )
    summary = probe.summarise_stacked_patches({}, patcher, residual_prefixes=())
    assert summary["zero_keys"] == [key]
    assert summary["nonzero_base_key_hits"] == []
    assert summary["patch_count"] == 1


def test_the_residual_prefixes_come_from_the_runtime(probe):
    from raven_streaming import runtime_linear

    prefixes = probe.raven_residual_prefixes()
    assert prefixes == (
        runtime_linear.A_PARAM_TEMPLATE.split("{")[0],
        runtime_linear.B_PARAM_TEMPLATE.split("{")[0],
    )
    assert all("{" not in p for p in prefixes)


# -- the gates, as a pure function of what was observed --------------------


def _stacked_record(**overrides):
    """A record of a stacked LoRA that did everything right."""
    record = {
        "lora_name": "style.safetensors",
        "path": "/models/loras/style.safetensors",
        "bytes": 4096,
        "strength": 0.8,
        "node_module": "nodes",
        "node_class": "LoraLoaderModelOnly",
        "node_function": "load_lora_model_only",
        "node_return_types": ["MODEL"],
        "node_calls": 1,
        "load_lora_for_models_calls": 1,
        "returned_a_clone": True,
        "patcher_class": "ModelPatcher",
        "patcher_class_before": "ModelPatcher",
        "patcher_class_unchanged": True,
        "patcher_is_stock": True,
        "patch_count": 3,
        "target_key_count": 3,
        "nonzero_base_key_hits": ["diffusion_model.blocks.0.attn.out_proj.weight"],
        "zero_keys": [],
        "unmatched_warnings": [],
        "targets_outside_the_state_dict": [],
        "residual_targets": [],
        "attachment_same_object": True,
        "attachment_modules": 266,
        "attachment_modules_before": 266,
        "expected_attachment_modules": 266,
        "diffusion_model_same_object": True,
    }
    record.update(overrides)
    return record


def _stacked_failures(probe, record):
    checks = probe.Checks()
    probe.check_stacked_lora(record, checks)
    return [c.name for c in checks.failures]


def test_a_clean_stacked_lora_passes_every_gate(probe):
    checks = probe.Checks()
    probe.check_stacked_lora(_stacked_record(), checks)
    assert checks.ok
    # the file is evidence, so it is in the report whatever happened
    assert any(
        "/models/loras/style.safetensors" in c.detail and "4096 bytes" in c.detail
        for c in checks.items
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"node_calls": 0}, "stacked lora: the official node ran exactly once"),
        ({"node_calls": 2}, "stacked lora: the official node ran exactly once"),
        (
            {"node_function": "apply_lora"},
            "stacked lora: the official node is the one upstream advertises",
        ),
        (
            {"node_return_types": ["MODEL", "CLIP"]},
            "stacked lora: the official node is the one upstream advertises",
        ),
        (
            {"returned_a_clone": False},
            "stacked lora: the node returned a fresh MODEL, not the one it was given",
        ),
        (
            {"patcher_is_stock": False},
            "stacked lora: the patcher downstream is still a stock ModelPatcher",
        ),
        (
            {"patcher_class_unchanged": False},
            "stacked lora: the patcher downstream is still a stock ModelPatcher",
        ),
        (
            {"patch_count": 0, "target_key_count": 0, "nonzero_base_key_hits": []},
            "stacked lora: the official load produced patches",
        ),
        (
            {"nonzero_base_key_hits": []},
            "stacked lora: at least one non-zero patch landed on a base key",
        ),
        (
            {"unmatched_warnings": ["lora key not loaded: lora_unet_nope.lora_up.weight"]},
            "stacked lora: every key of the file was matched",
        ),
        (
            {"targets_outside_the_state_dict": ["diffusion_model.ghost.weight"]},
            "stacked lora: every target is a real key of the model",
        ),
        (
            {"residual_targets": ["diffusion_model.blocks.0.attn.out_proj.raven_lora_A_0"]},
            "stacked lora: nothing targeted the RAVEN residual's own parameters",
        ),
        (
            {"attachment_same_object": False},
            "stacked lora: the mandatory RAVEN attachment came through untouched",
        ),
        (
            {"attachment_modules": 265},
            "stacked lora: the mandatory RAVEN attachment came through untouched",
        ),
        (
            {"attachment_modules": 200, "attachment_modules_before": 200},
            "stacked lora: the mandatory RAVEN attachment came through untouched",
        ),
        (
            {"diffusion_model_same_object": False},
            "stacked lora: the DiT object downstream is the same one the loader built",
        ),
    ],
)
def test_a_broken_stacked_lora_is_a_gate_failure(probe, overrides, expected):
    assert expected in _stacked_failures(probe, _stacked_record(**overrides))


def test_the_zero_patch_count_is_reported_alongside_the_hit(probe):
    checks = probe.Checks()
    probe.check_stacked_lora(
        _stacked_record(zero_keys=["diffusion_model.blocks.1.mlp.fc1.weight"]), checks
    )
    detail = [
        c.detail
        for c in checks.items
        if c.name == "stacked lora: at least one non-zero patch landed on a base key"
    ][0]
    assert "1 target(s) patched to zero" in detail


def test_a_stacked_record_survives_the_json_report(probe, repo_report_dir):
    """The record is only useful if it reaches the report intact."""
    report = probe.Report()
    report.setup = {"stacked_lora": _stacked_record()}
    report.environment = {"kv_cache_storage": "gpu"}
    path = os.path.join(repo_report_dir, "stacked.json")
    probe.atomic_write_json(path, report.to_dict())
    payload = json.loads(open(path).read())
    stacked = payload["setup"]["stacked_lora"]
    assert stacked["node_class"] == "LoraLoaderModelOnly"
    assert stacked["path"] == "/models/loras/style.safetensors"
    assert stacked["bytes"] == 4096
    assert stacked["strength"] == 0.8
    assert stacked["attachment_modules"] == 266
    rendered = report.render()
    assert "stacked lora: LoraLoaderModelOnly.load_lora_model_only" in rendered
    assert "kv cache gpu" in rendered


# -- and now the real upstream node ---------------------------------------
#
# Everything above pins the probe's own reasoning. These drive
# ``nodes.LoraLoaderModelOnly`` itself -- the real class, its real
# ``load_lora_model_only``, real ``comfy.sd.load_lora_for_models`` underneath,
# real ``folder_paths`` resolution and a real safetensors file -- against a
# miniature model that carries real base keys and a real RAVEN-shaped residual
# parameter. Nothing is downloaded and nothing weighs more than a few KB.


@pytest.fixture(scope="module")
def upstream_lora_node():
    """``nodes.LoraLoaderModelOnly`` from a local ComfyUI, or a skip."""
    from conftest import find_upstream_comfyui

    path = find_upstream_comfyui()
    if path is None:
        pytest.skip("no ComfyUI checkout (set COMFYUI_PATH / COMFYUI_UPSTREAM_PATH)")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        import comfy.model_patcher  # noqa: F401
        import comfy.sd  # noqa: F401
        import folder_paths  # type: ignore[import-not-found]
        import nodes as upstream_nodes  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - missing optional deps
        pytest.skip("cannot import ComfyUI nodes: {}: {}".format(type(exc).__name__, exc))
    finally:
        sys.argv = saved_argv
    return SimpleNamespace(
        nodes=upstream_nodes,
        comfy=sys.modules["comfy"],
        folder_paths=folder_paths,
        node_cls=upstream_nodes.LoraLoaderModelOnly,
    )


TINY_OUT, TINY_IN = 8, 6
#: the base module the miniature model exposes, spelled the way the real H3
#: inventory spells one (``raven_streaming.lora.RavenBaseConfig.modules()``)
TINY_PATH = "blocks.0.attn.out_proj"
TINY_BASE_KEY = "diffusion_model.{}.weight".format(TINY_PATH)
TINY_LORA_PREFIX = "lora_unet_{}".format(TINY_PATH.replace(".", "_"))
TINY_RESIDUAL_KEY = "diffusion_model.{}.raven_lora_A_0".format(TINY_PATH)


def _tiny_raven_model():
    """A model with a real base weight and a real RAVEN residual parameter.

    The residual is registered exactly as ``runtime_linear`` registers it: a
    direct parameter of the base leaf module, whose name deliberately does not
    end in ``.weight``. That detail is the whole reason the "nothing targeted
    the residual" gate exists -- ``model_lora_keys_unet`` still exposes such a
    key under the generic format.
    """
    from raven_streaming import lora as raven_lora
    from raven_streaming import runtime_linear

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Linear(TINY_IN, TINY_OUT, bias=False)
            self.out_proj.register_parameter(
                runtime_linear.A_PARAM_TEMPLATE.format(0),
                torch.nn.Parameter(torch.zeros(2, TINY_IN)),
            )

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    class RavenCausalMiniMaxH3Model(torch.nn.Module):  # noqa: N801 - the name is the check
        """``_run_probe`` identifies the chunk-causal DiT by class name."""

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])
            self.dtype = torch.float32
            self.hidden_size = TINY_OUT

        def forward_chunk(self, **kwargs):  # pragma: no cover - never sampled here
            raise AssertionError("the stacked-lora tests never sample")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = RavenCausalMiniMaxH3Model()
            self.model_config = SimpleNamespace(unet_config={})

    model = Model()
    torch.nn.init.ones_(model.diffusion_model.blocks[0].attn.out_proj.weight)
    # the mandatory adapter's handle, with the published module count
    model.raven_lora_attachment = SimpleNamespace(
        entries=[object()] * raven_lora.EXPECTED_MODULE_COUNT
    )
    model.raven_lora_manifest = SimpleNamespace(name="raven-4nfe-preview")
    assert TINY_BASE_KEY in model.state_dict()
    assert TINY_RESIDUAL_KEY in model.state_dict()
    return model


def _write_lora(upstream, directory, tensors, name="stacked_test_lora.safetensors"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    upstream.comfy.utils.save_torch_file({k: v for k, v in tensors.items()}, str(path))
    return path


def _rank1_lora(scale=1.0):
    """The minimal non-zero rank-1 LoRA for ``TINY_BASE_KEY``.

    This is the exact key shape ``comfy.weight_adapter.LoRAAdapter`` looks for
    first (``regular_lora``): ``<mapped key>.lora_up.weight`` [out, r] and
    ``<mapped key>.lora_down.weight`` [r, in], with the optional ``.alpha``.
    """
    return {
        "{}.lora_up.weight".format(TINY_LORA_PREFIX): torch.full((TINY_OUT, 1), scale),
        "{}.lora_down.weight".format(TINY_LORA_PREFIX): torch.full((1, TINY_IN), scale),
        "{}.alpha".format(TINY_LORA_PREFIX): torch.tensor(1.0),
    }


@pytest.fixture
def stacked_lora_env(probe, upstream_lora_node, tmp_path):
    """A real ModelPatcher, a real loras folder, and a real official node.

    ``folder_paths`` is the *real* module here, because that is what
    ``LoraLoaderModelOnly`` itself resolves through -- registering a fake would
    only fool the probe, not the node. The registration is undone afterwards so
    the rest of the suite sees the folder it started with.
    """
    folder_paths = upstream_lora_node.folder_paths
    directory = tmp_path / "loras"
    directory.mkdir(parents=True, exist_ok=True)
    saved = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.add_model_folder_path("loras", str(directory), is_default=True)
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop("loras", None)

    model = _tiny_raven_model()
    patcher = upstream_lora_node.comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    env = probe.ComfyEnv(
        root="/fake/ComfyUI",
        comfy=upstream_lora_node.comfy,
        folder_paths=folder_paths,
        upstream_nodes=upstream_lora_node.nodes,
    )
    try:
        yield SimpleNamespace(
            env=env,
            model=patcher,
            inner=model,
            directory=directory,
            upstream=upstream_lora_node,
        )
    finally:
        if saved is None:
            folder_paths.folder_names_and_paths.pop("loras", None)
        else:
            folder_paths.folder_names_and_paths["loras"] = saved
        if isinstance(cache, dict):
            cache.pop("loras", None)


def test_the_official_node_stacks_onto_the_raven_model(probe, stacked_lora_env):
    path = _write_lora(
        stacked_lora_env.upstream, stacked_lora_env.directory, _rank1_lora()
    )
    checks = probe.Checks()
    stacked, record = probe.stack_official_lora(
        env=stacked_lora_env.env,
        model=stacked_lora_env.model,
        lora_name=path.name,
        strength=0.5,
        checks=checks,
    )

    assert checks.ok, [c.line() for c in checks.failures]
    # the node, as upstream defines it -- not a reimplementation of it
    assert record["node_class"] == "LoraLoaderModelOnly"
    assert record["node_function"] == "load_lora_model_only"
    assert record["node_calls"] == 1
    assert record["load_lora_for_models_calls"] == 1
    assert record["node_call_args"] == [[path.name, 0.5]]
    # the file, identified
    assert record["path"] == str(path.resolve())
    assert record["bytes"] == path.stat().st_size
    assert record["strength"] == 0.5
    # what the official load did
    assert record["target_keys"] == [TINY_BASE_KEY]
    assert record["patch_count"] == 1
    assert record["nonzero_base_key_hits"] == [TINY_BASE_KEY]
    assert record["unmatched_warnings"] == []
    assert record["residual_targets"] == []
    # ... and what it left alone
    assert record["returned_a_clone"] is True
    assert record["patcher_is_stock"] is True
    assert record["attachment_same_object"] is True
    assert record["attachment_modules"] == 266
    assert stacked.model is stacked_lora_env.inner
    assert stacked.patches[TINY_BASE_KEY][0][0] == 0.5
    # the RAVEN residual parameter is still there, unpatched
    assert TINY_RESIDUAL_KEY in stacked.model.state_dict()
    assert TINY_RESIDUAL_KEY not in stacked.patches


def test_the_stacked_strength_is_the_one_the_node_was_given(probe, stacked_lora_env):
    path = _write_lora(
        stacked_lora_env.upstream, stacked_lora_env.directory, _rank1_lora()
    )
    checks = probe.Checks()
    stacked, record = probe.stack_official_lora(
        env=stacked_lora_env.env,
        model=stacked_lora_env.model,
        lora_name=path.name,
        strength=0.25,
        checks=checks,
    )
    assert checks.ok, [c.line() for c in checks.failures]
    assert record["strength"] == 0.25
    assert stacked.patches[TINY_BASE_KEY][0][0] == 0.25
    assert record["patch_detail"][0]["patches"][0]["strength_patch"] == 0.25
    assert record["patch_detail"][0]["patches"][0]["nonzero"] is True


def test_a_lora_for_another_model_fails_the_gates(probe, stacked_lora_env):
    """Comfy reports unmatched keys by logging; the probe has to catch that."""
    path = _write_lora(
        stacked_lora_env.upstream,
        stacked_lora_env.directory,
        {
            "lora_unet_some_other_model_block.lora_up.weight": torch.ones(4, 1),
            "lora_unet_some_other_model_block.lora_down.weight": torch.ones(1, 4),
        },
        name="wrong_model.safetensors",
    )
    checks = probe.Checks()
    _stacked, record = probe.stack_official_lora(
        env=stacked_lora_env.env,
        model=stacked_lora_env.model,
        lora_name=path.name,
        strength=1.0,
        checks=checks,
    )
    failed = [c.name for c in checks.failures]
    assert "stacked lora: every key of the file was matched" in failed
    assert "stacked lora: the official load produced patches" in failed
    assert "stacked lora: at least one non-zero patch landed on a base key" in failed
    assert record["unmatched_warnings"]
    assert record["target_keys"] == []
    # the node still ran exactly once, and the residual is still intact
    assert record["node_calls"] == 1
    assert record["attachment_same_object"] is True


def test_a_lora_that_targets_the_raven_residual_fails(probe, stacked_lora_env):
    """The residual's parameters do not end in ``.weight``, so the generic
    format still maps them -- which is exactly the hole being watched."""
    path = _write_lora(
        stacked_lora_env.upstream,
        stacked_lora_env.directory,
        {"{}.diff".format(TINY_RESIDUAL_KEY): torch.ones(2, TINY_IN)},
        name="residual_hijack.safetensors",
    )
    checks = probe.Checks()
    _stacked, record = probe.stack_official_lora(
        env=stacked_lora_env.env,
        model=stacked_lora_env.model,
        lora_name=path.name,
        strength=1.0,
        checks=checks,
    )
    assert record["residual_targets"] == [TINY_RESIDUAL_KEY]
    assert (
        "stacked lora: nothing targeted the RAVEN residual's own parameters"
        in [c.name for c in checks.failures]
    )


def test_a_zeroed_lora_is_not_counted_as_a_hit(probe, stacked_lora_env):
    path = _write_lora(
        stacked_lora_env.upstream,
        stacked_lora_env.directory,
        dict(
            _rank1_lora(),
            **{"{}.lora_up.weight".format(TINY_LORA_PREFIX): torch.zeros(TINY_OUT, 1)},
        ),
        name="all_zero.safetensors",
    )
    checks = probe.Checks()
    _stacked, record = probe.stack_official_lora(
        env=stacked_lora_env.env,
        model=stacked_lora_env.model,
        lora_name=path.name,
        strength=1.0,
        checks=checks,
    )
    assert record["target_keys"] == [TINY_BASE_KEY]
    assert record["nonzero_base_key_hits"] == []
    assert record["zero_keys"] == [TINY_BASE_KEY]
    assert (
        "stacked lora: at least one non-zero patch landed on a base key"
        in [c.name for c in checks.failures]
    )


def test_the_probe_refuses_a_name_folder_paths_cannot_resolve(probe, stacked_lora_env):
    with pytest.raises(probe.ProbeError, match="loras"):
        probe.stack_official_lora(
            env=stacked_lora_env.env,
            model=stacked_lora_env.model,
            lora_name="not_here.safetensors",
            strength=1.0,
            checks=probe.Checks(),
        )


def test_the_probe_refuses_a_comfyui_without_the_official_node(probe, stacked_lora_env):
    env = probe.ComfyEnv(
        root="/fake",
        comfy=stacked_lora_env.env.comfy,
        folder_paths=stacked_lora_env.env.folder_paths,
        upstream_nodes=SimpleNamespace(),
    )
    with pytest.raises(probe.ProbeError, match="LoraLoaderModelOnly"):
        probe.stack_official_lora(
            env=env,
            model=stacked_lora_env.model,
            lora_name="whatever.safetensors",
            strength=1.0,
            checks=probe.Checks(),
        )


def test_the_pinned_official_lora_node_still_has_this_shape():
    """Runs without ComfyUI importable: the source is read, not executed."""
    tree = _upstream_source("nodes.py")
    node = _find_class(tree, "LoraLoaderModelOnly")
    args, _defaults = _signature(_find_function(node, "load_lora_model_only"))
    assert args == ["self", "model", "lora_name", "strength_model"], (
        "the probe calls load_lora_model_only(model, name, strength) positionally"
    )
    import ast

    source = ast.dump(node)
    assert "'loras'" in source, "the combo no longer comes from folder_paths('loras')"
    assert "'load_lora_model_only'" in source, "FUNCTION was renamed"
    assert "'MODEL'" in source


# -- CLIPLoader's live schema ---------------------------------------------


UPSTREAM_CLIP_TYPES = ["stable_diffusion", "wan", "qwen_image", "minimax"]


class FakeCLIPLoader:
    """``nodes.CLIPLoader``'s surface: the V1 schema and ``load_clip``."""

    types = UPSTREAM_CLIP_TYPES
    devices = ["default", "cpu"]
    calls = []
    clip_factory = None

    @classmethod
    def INPUT_TYPES(cls):
        schema = {"required": {"clip_name": (list(cls.types),), "type": (list(cls.types),)}}
        if cls.devices is not None:
            schema["optional"] = {"device": (list(cls.devices), {"advanced": True})}
        return schema

    # the parameter really is called ``type`` upstream; the AST test below pins it
    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        FakeCLIPLoader.calls.append(
            {"clip_name": clip_name, "type": type, "device": device}
        )
        return (FakeCLIPLoader.clip_factory(),)


def test_choose_clip_type_reads_the_live_schema(probe):
    clip_type, device = probe.choose_clip_type(FakeCLIPLoader)
    assert clip_type == "minimax"
    assert device == "default"


def test_choose_clip_type_is_case_insensitive(probe):
    class Upper(FakeCLIPLoader):
        types = ["stable_diffusion", "MiniMax"]

    assert probe.choose_clip_type(Upper)[0] == "MiniMax"


def test_choose_clip_type_refuses_a_comfyui_without_minimax(probe):
    class NoMiniMax(FakeCLIPLoader):
        types = ["stable_diffusion", "wan"]

    with pytest.raises(probe.ProbeError, match="no MiniMax H3 text-encoder type"):
        probe.choose_clip_type(NoMiniMax)


def test_choose_clip_type_refuses_a_schema_without_a_default_device(probe):
    class NoDefault(FakeCLIPLoader):
        devices = ["cpu"]

    with pytest.raises(probe.ProbeError, match="no 'default' entry"):
        probe.choose_clip_type(NoDefault)


def test_choose_clip_type_tolerates_a_schema_without_a_device_input(probe):
    class NoDeviceInput(FakeCLIPLoader):
        devices = None

    assert probe.choose_clip_type(NoDeviceInput) == ("minimax", "default")


def test_choose_clip_type_refuses_an_unrecognisable_schema(probe):
    class Weird:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"something_else": (["a"],)}}

    with pytest.raises(probe.ProbeError, match="refuses to guess"):
        probe.choose_clip_type(Weird)


# -- watching the encode's load device ------------------------------------


class FakePatcher:
    def __init__(self, load_device="cuda:0", offload_device="cpu", size=1024, model=None):
        self.load_device = torch.device(load_device)
        self.offload_device = torch.device(offload_device)
        self.is_clip = True
        self.model = model if model is not None else torch.nn.Linear(4, 4)
        self._size = size
        self._loaded = 0

    def loaded_size(self):
        return self._loaded

    def model_size(self):
        return self._size

    def current_loaded_device(self):
        return self.load_device if self._loaded else self.offload_device


class FakeCLIP:
    def __init__(self, patcher=None):
        self.patcher = patcher if patcher is not None else FakePatcher()


def test_watch_model_loads_records_and_restores(probe):
    patcher = FakePatcher()
    seen = []

    def load_models_gpu(models, memory_required=0, **kwargs):
        seen.append(memory_required)
        for model in models:
            model._loaded = model.model_size()

    module = SimpleNamespace(load_models_gpu=load_models_gpu)
    with probe.watch_model_loads(module) as records:
        module.load_models_gpu([patcher], memory_required=99)
    assert module.load_models_gpu is load_models_gpu  # restored
    assert seen == [99]  # the real call still happened, untouched
    assert len(records) == 1
    entry = records[0]["models"][0]
    assert records[0]["memory_required"] == 99
    assert entry["load_device"] == "cuda:0"
    assert entry["is_clip"] is True
    assert entry["current_loaded_device"] == "cuda:0"
    assert entry["loaded_size"] == 1024
    assert entry["id"] == id(patcher)


def test_parameter_histogram_counts_buffers_too(probe):
    module = torch.nn.Linear(4, 4, bias=False)  # 16 float32 params
    module.register_buffer("scales", torch.zeros(8, dtype=torch.float32))
    histogram = probe.parameter_device_histogram(module)
    assert histogram["cpu"]["parameter_bytes"] == 64
    assert histogram["cpu"]["buffer_bytes"] == 32
    assert histogram["cpu"]["parameters"] == 1 and histogram["cpu"]["buffers"] == 1


def test_residency_reports_what_comfy_sees(probe):
    clip = FakeCLIP()
    clip.patcher._loaded = 512
    residency = probe.text_encoder_residency(clip)
    assert residency["load_device"] == "cuda:0"
    assert residency["loaded_size"] == 512
    assert residency["current_loaded_device"] == "cuda:0"
    assert residency["devices"]["cpu"]["parameters"] >= 1


def test_offload_uses_comfys_targeted_unload(probe):
    clip = FakeCLIP()
    clip.patcher._loaded = 4096
    unloaded = []

    def unload_model_and_clones(patcher):
        unloaded.append(patcher)
        patcher._loaded = 0

    module = SimpleNamespace(unload_model_and_clones=unload_model_and_clones)
    result = probe.offload_text_encoder(module, clip)
    assert result["method"] == "unload_model_and_clones"
    assert unloaded == [clip.patcher]
    assert probe.text_encoder_residency(clip)["loaded_size"] == 0


def test_offload_reports_an_older_comfyui_instead_of_hand_rolling_one(probe):
    result = probe.offload_text_encoder(SimpleNamespace(), FakeCLIP())
    assert result["method"] is None
    assert "evicted naturally" in result["reason"]


def test_offload_reports_a_failing_unload(probe):
    def boom(patcher):
        raise RuntimeError("nope")

    result = probe.offload_text_encoder(
        SimpleNamespace(unload_model_and_clones=boom), FakeCLIP()
    )
    assert result["error"].startswith("RuntimeError")


# -- the official T2VA node ------------------------------------------------


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


#: Official Qwen3-VL width, mirrored here so the fixture does not depend on the
#: probe module being importable.
TEXT_DIM = 5120


def official_t2va_outputs(geometry=None, *, extras=None):
    """What ``MiniMaxH3ImageToVideo`` returns for a T2VA prompt."""
    geometry = geometry or FAKE_GEOMETRY
    text_len = 11
    context = _randn(1, text_len, TEXT_DIM)
    tags = torch.tensor(
        [layout_mod.TEXT_TAG] * (text_len - 2) + [layout_mod.VIDEO_TAG, layout_mod.AUDIO_TAG],
        dtype=torch.int64,
    )
    payload = {"minimax_token_tags": tags, "pooled_output": torch.zeros(1, 8)}
    payload.update(extras or {})
    conditioning = [[context, payload]]
    latent = {
        "samples": FakeNestedTensor(
            (
                torch.zeros(
                    1, 24, geometry["latent_t"], geometry["latent_h"], geometry["latent_w"]
                ),
                torch.zeros(1, 32, 2, geometry["audio_t"]),
            )
        )
    }
    return conditioning, latent


class FakeImageToVideo:
    calls = []
    outputs = None
    raises = None

    @classmethod
    def execute(cls, **kwargs):
        FakeImageToVideo.calls.append(dict(kwargs))
        if FakeImageToVideo.raises is not None:
            raise FakeImageToVideo.raises
        conditioning, latent = FakeImageToVideo.outputs
        return FakeNodeOutput(conditioning, latent)


#: ``_text_lane_env`` wraps ``execute`` to record the load; the fixture below
#: puts the original back so the wrapper cannot stack across tests.
_ORIGINAL_EXECUTE = FakeImageToVideo.__dict__["execute"]


@pytest.fixture(autouse=True)
def _reset_fake_official_nodes():
    FakeCLIPLoader.calls = []
    FakeCLIPLoader.types = list(UPSTREAM_CLIP_TYPES)
    FakeCLIPLoader.devices = ["default", "cpu"]
    FakeCLIPLoader.clip_factory = FakeCLIP
    FakeImageToVideo.calls = []
    FakeImageToVideo.outputs = None
    FakeImageToVideo.raises = None
    FakeImageToVideo.execute = _ORIGINAL_EXECUTE
    yield
    FakeImageToVideo.execute = _ORIGINAL_EXECUTE


def test_encode_t2va_never_passes_a_keyframe(probe):
    FakeImageToVideo.outputs = official_t2va_outputs()
    clip, vae = FakeCLIP(), object()
    conditioning, latent = probe.encode_t2va(
        FakeImageToVideo, clip=clip, vae=vae, prompt="a cat", width=64, height=64, length=22
    )
    call = FakeImageToVideo.calls[0]
    assert call == {
        "clip": clip, "vae": vae, "prompt": "a cat",
        "width": 64, "height": 64, "length": 22,
    }
    assert "first_frame" not in call and "last_frame" not in call
    assert conditioning is FakeImageToVideo.outputs[0]
    assert latent is FakeImageToVideo.outputs[1]


def test_encode_t2va_accepts_a_plain_tuple_result(probe):
    conditioning, latent = official_t2va_outputs()

    class TupleNode:
        @classmethod
        def execute(cls, **kwargs):
            return (conditioning, latent)

    assert probe.encode_t2va(
        TupleNode, clip=None, vae=None, prompt="x", width=64, height=64, length=22
    ) == (conditioning, latent)


def test_encode_t2va_refuses_an_unexpected_return(probe):
    class WrongNode:
        @classmethod
        def execute(cls, **kwargs):
            return FakeNodeOutput("only one")

    with pytest.raises(probe.ProbeError, match="expected \\(CONDITIONING, LATENT\\)"):
        probe.encode_t2va(
            WrongNode, clip=None, vae=None, prompt="x", width=64, height=64, length=22
        )


def test_clone_latent_copies_the_official_tensors(probe):
    _conditioning, latent = official_t2va_outputs()
    copy = probe.clone_latent(latent, FakeNestedTensor)
    original_video, original_audio = latent["samples"].unbind()
    video, audio = copy["samples"].unbind()
    assert torch.equal(video, original_video) and torch.equal(audio, original_audio)
    assert video is not original_video and audio is not original_audio
    video += 1.0
    assert not torch.equal(video, original_video)  # runs cannot affect each other


# -- what the node handed back --------------------------------------------


def test_official_t2va_outputs_pass_the_contract_checks(probe):
    conditioning, latent = official_t2va_outputs()
    summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    failures = [c.line() for c in checks if c.gate and not c.ok]
    assert not failures, failures
    assert summary["text_len"] == 11
    assert summary["context"]["shape"] == [1, 11, 5120]
    assert summary["context"]["finite"] is True
    assert summary["token_tags"]["histogram"] == {
        "video(0)": 1, "text(1)": 9, "audio(2)": 1,
    }
    assert summary["token_tags"]["shape"] == [11]
    assert summary["extras"] == ["minimax_token_tags", "pooled_output"]
    assert summary["latent"]["frames"] == 22 and summary["latent"]["empty"] is True
    assert summary["latent"]["nested_class"] == "FakeNestedTensor"


def test_a_keyframe_conditioning_is_refused(probe):
    conditioning, latent = official_t2va_outputs(
        extras={"minimax_keyframes": [{"resolved_frame_index": 0}]}
    )
    _summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert any("no minimax_keyframes" in name for name in failed)
    # and the sampler's own contract rejects it too
    assert any("accepts this CONDITIONING" in name for name in failed)


def test_a_reference_conditioning_is_refused(probe):
    conditioning, latent = official_t2va_outputs(extras={"minimax_refs": [1]})
    _summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    assert any(
        "no minimax_refs" in c.name for c in checks if c.gate and not c.ok
    )


def test_a_latent_off_the_requested_grid_is_caught(probe):
    conditioning, latent = official_t2va_outputs()
    other = dict(FAKE_GEOMETRY, frames=39, latent_t=12, audio_t=65)
    _summary, checks = probe.check_text_lane_outputs(conditioning, latent, geometry=other)
    assert any(
        "sits on the requested grid" in c.name for c in checks if c.gate and not c.ok
    )


def test_a_non_empty_official_latent_is_caught(probe):
    conditioning, latent = official_t2va_outputs()
    latent["samples"].tensors[0] += 0.5
    _summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    failed = [c for c in checks if c.gate and not c.ok]
    # the sampler's own contract is what refuses it, and it says why
    assert any("accepts this LATENT" in c.name for c in failed)
    assert any("is not empty" in c.detail for c in failed)


def test_a_missing_tag_vector_is_reported_but_does_not_gate(probe):
    conditioning, latent = official_t2va_outputs()
    del conditioning[0][1]["minimax_token_tags"]
    summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert summary["token_tags"] is None
    warned = [c for c in checks if not c.ok and not c.gate]
    assert any("minimax_token_tags" in c.name for c in warned)


def test_a_context_of_the_wrong_width_is_caught(probe):
    conditioning, latent = official_t2va_outputs()
    conditioning[0][0] = _randn(1, 11, 777)
    _summary, checks = probe.check_text_lane_outputs(
        conditioning, latent, geometry=FAKE_GEOMETRY
    )
    assert any("context width" in c.name for c in checks if c.gate and not c.ok)


def test_the_dit_hidden_size_comes_from_the_package(probe):
    from raven_streaming import lora as lora_mod

    assert probe.dit_hidden_size() == lora_mod.RavenBaseConfig().hidden_size


# -- the whole lane, in order ---------------------------------------------


def _text_lane_env(tmp_path, *, load_device="cuda:0", with_unload=True):
    encoder = _touch(tmp_path / "te" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    folder_paths = FakeFolderPaths()
    order = []
    patcher = FakePatcher(load_device=load_device)
    clip = FakeCLIP(patcher)

    def clip_factory():
        order.append("load_clip")
        return clip

    FakeCLIPLoader.clip_factory = clip_factory
    FakeImageToVideo.outputs = official_t2va_outputs()

    def load_models_gpu(models, memory_required=0, **kwargs):
        order.append("load_models_gpu")
        for model in models:
            model._loaded = model.model_size()

    module = SimpleNamespace(load_models_gpu=load_models_gpu)
    if with_unload:
        def unload_model_and_clones(target):
            order.append("unload")
            target._loaded = 0

        module.unload_model_and_clones = unload_model_and_clones

    original_execute = FakeImageToVideo.execute.__func__

    def execute(cls, **kwargs):
        order.append("encode")
        module.load_models_gpu([patcher], memory_required=123)
        return original_execute(cls, **kwargs)

    FakeImageToVideo.execute = classmethod(execute)
    return SimpleNamespace(
        encoder=encoder, folder_paths=folder_paths, order=order,
        clip=clip, patcher=patcher, model_management=module,
    )


def test_the_text_lane_runs_in_the_workflow_order(probe, tmp_path):
    env = _text_lane_env(tmp_path)
    checks = probe.Checks()
    video_vae = object()
    conditioning, latent, summary = probe.run_text_lane(
        clip_loader_cls=FakeCLIPLoader,
        image_to_video_cls=FakeImageToVideo,
        folder_paths=env.folder_paths,
        model_management=env.model_management,
        text_encoder_path=env.encoder,
        prompt="a cat playing a trumpet",
        geometry=FAKE_GEOMETRY,
        video_vae=video_vae,
        device=torch.device("cuda:0"),
        checks=checks,
    )
    failures = [c.line() for c in checks.items if c.gate and not c.ok]
    assert not failures, failures

    # order: register -> load the encoder -> encode -> offload
    assert env.order == ["load_clip", "encode", "load_models_gpu", "unload"]
    assert env.folder_paths.registered == [
        ("text_encoders", str(env.encoder.parent), True)
    ]

    # the loader got the basename and the type/device the schema offered
    assert FakeCLIPLoader.calls == [
        {"clip_name": env.encoder.name, "type": "minimax", "device": "default"}
    ]
    # the T2VA node got the loaded VAE object itself, not a path or a copy
    call = FakeImageToVideo.calls[0]
    assert call["vae"] is video_vae and call["clip"] is env.clip
    assert call["prompt"] == "a cat playing a trumpet"
    assert (call["width"], call["height"], call["length"]) == (64, 64, 22)

    assert conditioning is FakeImageToVideo.outputs[0]
    assert latent is FakeImageToVideo.outputs[1]

    # the report identifies the encode
    assert summary["name"] == env.encoder.name
    assert summary["clip_type"] == "minimax"
    assert summary["prompt"] == "a cat playing a trumpet"
    assert summary["prompt_characters"] == len("a cat playing a trumpet")
    assert summary["load_seconds"] >= 0 and summary["encode_seconds"] >= 0
    assert summary["text_len"] == 11
    assert summary["context"]["shape"] == [1, 11, 5120]
    assert summary["token_tags"]["histogram"]["text(1)"] == 9
    assert summary["latent"]["frames"] == 22
    assert summary["offload"]["method"] == "unload_model_and_clones"
    assert summary["residency_after_encode"]["loaded_size"] == 1024
    assert summary["residency_after_offload"]["loaded_size"] == 0
    assert summary["encoder_loads"][0]["current_loaded_device"] == "cuda:0"

    names = [c.name for c in checks.items]
    assert any("the encode ran with the encoder loaded onto cuda" in n for n in names)
    assert any("holds no cuda weights during sampling" in n for n in names)


def test_the_text_lane_fails_when_the_encode_ran_on_the_wrong_device(probe, tmp_path):
    env = _text_lane_env(tmp_path, load_device="cpu")
    checks = probe.Checks()
    probe.run_text_lane(
        clip_loader_cls=FakeCLIPLoader,
        image_to_video_cls=FakeImageToVideo,
        folder_paths=env.folder_paths,
        model_management=env.model_management,
        text_encoder_path=env.encoder,
        prompt="a cat",
        geometry=FAKE_GEOMETRY,
        video_vae=object(),
        device=torch.device("cuda:0"),
        checks=checks,
    )
    failed = [c.name for c in checks.items if c.gate and not c.ok]
    assert any("the encode ran with the encoder loaded onto cuda" in name for name in failed)


def test_the_text_lane_notes_a_comfyui_without_targeted_unload(probe, tmp_path):
    env = _text_lane_env(tmp_path, with_unload=False)
    checks = probe.Checks()
    _cond, _latent, summary = probe.run_text_lane(
        clip_loader_cls=FakeCLIPLoader,
        image_to_video_cls=FakeImageToVideo,
        folder_paths=env.folder_paths,
        model_management=env.model_management,
        text_encoder_path=env.encoder,
        prompt="a cat",
        geometry=FAKE_GEOMETRY,
        video_vae=object(),
        device=torch.device("cuda:0"),
        checks=checks,
    )
    assert [c.name for c in checks.items if c.gate and not c.ok] == []
    assert summary["offload"]["method"] is None
    assert any("not offloaded explicitly" in c.name for c in checks.items)


def test_the_text_lane_propagates_an_encode_failure(probe, tmp_path):
    env = _text_lane_env(tmp_path)
    FakeImageToVideo.raises = RuntimeError("the encoder exploded")
    with pytest.raises(RuntimeError, match="the encoder exploded"):
        probe.run_text_lane(
            clip_loader_cls=FakeCLIPLoader,
            image_to_video_cls=FakeImageToVideo,
            folder_paths=env.folder_paths,
            model_management=env.model_management,
            text_encoder_path=env.encoder,
            prompt="a cat",
            geometry=FAKE_GEOMETRY,
            video_vae=object(),
            device=torch.device("cuda:0"),
            checks=probe.Checks(),
        )
    # the wrapper is still put back even though the encode raised
    assert env.model_management.load_models_gpu.__name__ == "load_models_gpu"


# -- the upstream signatures this lane is pinned to ------------------------


def _upstream_source(relative):
    from conftest import find_upstream_comfyui

    root = find_upstream_comfyui()
    if root is None:
        pytest.skip("no ComfyUI checkout (set COMFYUI_PATH / COMFYUI_UPSTREAM_PATH)")
    path = os.path.join(str(root), relative)
    if not os.path.isfile(path):
        pytest.skip("{} is not in this checkout".format(relative))
    import ast

    return ast.parse(open(path, encoding="utf-8").read())


def _find_class(tree, name):
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("{} is not in this checkout".format(name))


def _find_function(scope, name):
    import ast

    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError("{} is not in this scope".format(name))


def _signature(function):
    import ast

    args = [a.arg for a in function.args.args]
    defaults = {}
    for arg, default in zip(args[len(args) - len(function.args.defaults):],
                            function.args.defaults):
        defaults[arg] = ast.literal_eval(default) if isinstance(
            default, ast.Constant) else "<expr>"
    return args, defaults


def test_pinned_signature_of_upstream_clip_loader():
    tree = _upstream_source("nodes.py")
    loader = _find_class(tree, "CLIPLoader")
    args, defaults = _signature(_find_function(loader, "load_clip"))
    assert args == ["self", "clip_name", "type", "device"], (
        "the probe calls load_clip(name, type, device) positionally"
    )
    assert defaults == {"type": "stable_diffusion", "device": "default"}


def test_pinned_clip_loader_schema_offers_minimax_and_a_default_device():
    import ast

    tree = _upstream_source("nodes.py")
    source = ast.dump(_find_function(_find_class(tree, "CLIPLoader"), "INPUT_TYPES"))
    assert "'minimax'" in source, "CLIPLoader no longer offers the MiniMax type"
    assert "'text_encoders'" in source
    assert "'default'" in source and "'cpu'" in source


def test_pinned_signature_of_the_official_t2va_node():
    tree = _upstream_source(os.path.join("comfy_extras", "nodes_minimax_h3.py"))
    node = _find_class(tree, "MiniMaxH3ImageToVideo")
    args, defaults = _signature(_find_function(node, "execute"))
    assert args == [
        "cls", "clip", "vae", "prompt", "width", "height", "length",
        "first_frame", "last_frame",
    ]
    # T2VA is exactly "do not pass these two"
    assert defaults == {"first_frame": None, "last_frame": None}


def test_pinned_signature_of_the_official_empty_av_latent():
    tree = _upstream_source(os.path.join("comfy_extras", "nodes_minimax_h3.py"))
    args, defaults = _signature(_find_function(tree, "_empty_av_latent"))
    assert args == ["width", "height", "length", "batch_size"]
    assert defaults == {"batch_size": 1}


def test_pinned_upstream_still_exposes_the_targeted_unload():
    tree = _upstream_source(os.path.join("comfy", "model_management.py"))
    args, _defaults = _signature(_find_function(tree, "unload_model_and_clones"))
    assert args[0] == "model"
    _find_function(tree, "loaded_models")


# ==========================================================================
# a real preview stream, built from real fMP4
# ==========================================================================


def _muxed_preview(session, *, frames=12, width=64, height=64):
    """Push real muxer output through a real ``PreviewMediaSink``."""
    import numpy as np

    from raven_streaming.media.mp4_writer import FragmentedMP4Muxer, MuxerConfig

    sink = preview_mod.PreviewMediaSink(session)
    config = pipeline_mod.PipelineConfig(frames=frames, width=width, height=height)
    sink.on_open(
        config.mime,
        width=width,
        height=height,
        fps=float(config.fps),
        audio={"sample_rate": config.sample_rate, "channels": config.channels},
        duration_hint=config.duration_seconds,
    )
    sink.on_status("model_loading")
    muxer = FragmentedMP4Muxer(
        MuxerConfig(
            width=width,
            height=height,
            fps=config.fps,
            sample_rate=config.sample_rate,
            channels=config.channels,
            segment_frames=frames,
            fragment_mode="every_frame",
            strict_idr=False,
        )
    ).open()
    sink.on_status("sampling")
    samples = config.clock.samples_for_frames(frames)
    for index in range(frames):
        picture = np.full((height, width, 3), index / max(1, frames), dtype=np.float32)
        muxer.write_video_frame(picture)
    muxer.write_audio(np.zeros((2, samples), dtype=np.float32))
    sink.on_status("finalizing")
    muxer.close()
    sink.pump_muxer(muxer)
    return sink


@pytest.fixture(scope="module")
def recorded_stream():
    """One complete, genuine preview session: real encoder, real envelopes."""
    pytest.importorskip("av")
    sender = preview_mod.RecordingSender()
    manager = preview_mod.PreviewManager(sender=sender)
    with manager.session("node-7", client_id="sid") as session:
        _muxed_preview(session)
    return sender.messages


def test_a_real_session_passes_every_envelope_check(probe, recorded_stream):
    summary, checks, blob = probe.check_preview_messages(
        recorded_stream, node_id="node-7", expect_end="complete", expect_media=True,
        width=64, height=64,
    )
    failures = [c for c in checks if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]
    assert summary["events"][0] == "open" and summary["events"][-1] == "end"
    assert summary["end_reason"] == "complete"
    assert summary["segments"] >= 1
    assert summary["init_boxes"][:1] == ["ftyp"] and "moov" in summary["init_boxes"]
    assert "moof" in summary["first_fragment_boxes"]
    assert "mdat" in summary["first_fragment_boxes"]
    assert summary["phases"] == ["model_loading", "sampling", "finalizing"]
    assert blob.startswith(b"\x00")  # a box length, i.e. real MP4 bytes
    assert len(blob) == summary["init_bytes"] + summary["segment_bytes"]


def test_the_recorded_stream_really_decodes(probe, recorded_stream):
    _summary, _checks, blob = probe.check_preview_messages(
        recorded_stream, node_id="node-7", expect_end="complete", expect_media=True,
    )
    media, checks = probe.decode_preview_media(blob, expect_frames=12)
    failures = [c for c in checks if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]
    assert media["video_streams"] == 1 and media["audio_streams"] == 1
    assert media["video_frames"] == 12
    assert media["audio_samples"] > 0


def test_decode_reports_a_frame_count_mismatch(probe, recorded_stream):
    _summary, _checks, blob = probe.check_preview_messages(
        recorded_stream, node_id="node-7", expect_end="complete", expect_media=True,
    )
    _media, checks = probe.decode_preview_media(blob, expect_frames=999)
    assert [c for c in checks if not c.ok and "every requested frame" in c.name]


def test_decode_of_nothing_is_a_failure_not_a_pass(probe):
    summary, checks = probe.decode_preview_media(b"")
    assert summary == {"bytes": 0}
    assert [c for c in checks if not c.ok]


def test_decode_of_garbage_is_reported_not_raised(probe):
    pytest.importorskip("av")
    summary, checks = probe.decode_preview_media(b"not an mp4 at all" * 64)
    assert [c for c in checks if not c.ok]
    assert "traceback" in summary or summary.get("video_streams") in (0, None)


# -- the negative envelope cases, with hand-built messages ------------------


def _envelope(seq, event, **body):
    payload = {"v": 1, "event": event, "session_id": "s", "node_id": "n", "seq": seq, "t": 0.0}
    payload.update(body)
    return ("raven.preview", payload, "sid")


#: Structurally valid, semantically empty MP4 boxes: enough for the box-level
#: checks, and small enough to read.
INIT_BOXES = b"\x00\x00\x00\x10ftypiso5\x00\x00\x00\x00" + b"\x00\x00\x00\x08moov"
FRAGMENT_BOXES = b"\x00\x00\x00\x08moof" + b"\x00\x00\x00\x08mdat"


def _payload(event, seq, blob, **extra):
    return _envelope(
        seq, event, encoding="base64", bytes=len(blob),
        data=base64.b64encode(blob).decode("ascii"), **extra
    )


def _minimal_stream(**overrides):
    messages = [
        _envelope(0, "open", mime='video/mp4; codecs="avc1.640028,mp4a.40.2"',
                  width=64, height=64, fps=24.0,
                  audio={"sample_rate": 32000, "channels": 2}),
        _payload("init", 1, INIT_BOXES),
        _payload("segment", 2, FRAGMENT_BOXES, index=0),
        _envelope(3, "end", reason="complete", segments=1),
    ]
    return [overrides.get(i, m) for i, m in enumerate(messages)]


def _failed(probe, messages, **kwargs):
    options = {
        "node_id": "n", "expect_end": "complete", "expect_media": True,
        "width": 64, "height": 64,
    }
    options.update(kwargs)
    _summary, checks, _blob = probe.check_preview_messages(messages, **options)
    return [c.name for c in checks if c.gate and not c.ok]


def test_a_hand_built_minimal_stream_passes(probe):
    assert _failed(probe, _minimal_stream()) == []


def test_a_seq_gap_is_caught(probe):
    messages = _minimal_stream()
    messages[2][1]["seq"] = 5
    assert any("seq is contiguous" in name for name in _failed(probe, messages))


def test_a_wrong_end_reason_is_caught(probe):
    messages = _minimal_stream()
    messages[3][1]["reason"] = "error"
    names = _failed(probe, messages)
    assert any("end reason" in name for name in names)


def test_a_cancelled_stream_is_only_accepted_when_expected(probe):
    messages = _minimal_stream()
    messages[3][1]["reason"] = "cancelled"
    assert any("end reason" in name for name in _failed(probe, messages))
    assert _failed(probe, messages, expect_end="cancelled") == []


def test_a_segment_before_init_is_caught(probe):
    messages = _minimal_stream()
    messages[1], messages[2] = messages[2], messages[1]
    messages[1][1]["seq"], messages[2][1]["seq"] = 1, 2
    assert any("init" in name for name in _failed(probe, messages))


def test_a_missing_end_is_caught(probe):
    messages = _minimal_stream()[:-1]
    assert any("exactly one end" in name for name in _failed(probe, messages))


def test_a_broken_payload_is_caught(probe):
    messages = _minimal_stream()
    messages[2][1]["data"] = "!!!! not base64 !!!!"
    assert any("base64" in name for name in _failed(probe, messages))


def test_a_lying_byte_count_is_caught(probe):
    messages = _minimal_stream()
    messages[2][1]["bytes"] = 999
    assert any("base64" in name for name in _failed(probe, messages))


def test_a_fragment_without_moof_is_caught(probe):
    messages = _minimal_stream()
    messages[2] = _payload("segment", 2, b"\x00\x00\x00\x08free" + b"\x00\x00\x00\x08skip", index=0)
    assert any("moof + mdat" in name for name in _failed(probe, messages))


def test_an_init_without_ftyp_is_caught(probe):
    messages = _minimal_stream()
    messages[1] = _payload("init", 1, b"\x00\x00\x00\x08moov")
    assert any("init segment" in name for name in _failed(probe, messages))


def test_the_wrong_node_id_is_caught(probe):
    assert any("node id" in name for name in _failed(probe, _minimal_stream(), node_id="other"))


def test_a_canvas_mismatch_is_caught(probe):
    assert any(
        "canvas" in name for name in _failed(probe, _minimal_stream(), width=1376, height=768)
    )


def test_an_empty_session_is_a_failure(probe):
    assert any("produced messages" in name for name in _failed(probe, []))


def test_a_cancelled_run_may_have_no_media(probe):
    messages = [
        _envelope(0, "open", mime='video/mp4; codecs="avc1.640028,mp4a.40.2"',
                  audio={"sample_rate": 32000, "channels": 2}),
        _envelope(1, "status", phase="model_loading"),
        _envelope(2, "end", reason="cancelled", segments=0),
    ]
    assert _failed(probe, messages, expect_end="cancelled", expect_media=False,
                   width=None, height=None) == []


# ==========================================================================
# metrics and the determinism gate
# ==========================================================================


def test_tensor_metrics_separate_bitwise_from_close(probe):
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert probe.tensor_metrics(a, a.clone())["bitwise"] is True
    b = a.clone()
    b[0, 0] += 1e-6
    metrics = probe.tensor_metrics(a, b)
    assert metrics["bitwise"] is False
    assert metrics["max_abs"] == pytest.approx(1e-6, rel=1e-3)
    assert metrics["rel_l2"] > 0
    assert probe.tensor_metrics(a, a.reshape(4, 3))["comparable"] is False
    assert probe.tensor_metrics(a, None)["comparable"] is False


def _run_entry(video, audio, images, waveform, *, rss=1000, cuda=None):
    memory = {"rss_after": rss}
    if cuda is not None:
        memory["cuda_allocated_after"] = cuda
        memory["cuda_reserved_after"] = cuda
    return {
        "artifacts": {
            "video_latent": video, "audio_latent": audio,
            "images": images, "waveform": waveform,
        },
        "memory": memory,
    }


def test_identical_runs_pass_the_determinism_gate(probe):
    video = _randn(1, 24, 7, 4, 4)
    audio = _randn(1, 32, 2, 37)
    images = _rand(22, 64, 64, 3)
    waveform = _randn(1, 2, 29600)
    first = _run_entry(video, audio, images, waveform, rss=1000, cuda=100)
    second = _run_entry(
        video.clone(), audio.clone(), images.clone(), waveform.clone(), rss=1200, cuda=150
    )
    summary, checks = probe.compare_runs(first, second)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert summary["latent"]["video"]["bitwise"] is True
    assert summary["rss_growth_bytes"] == 200
    assert summary["cuda_allocated_growth_bytes"] == 50


def test_a_near_miss_fails_the_determinism_gate_and_is_quantified(probe):
    video = _randn(1, 24, 7, 4, 4)
    drifted = video.clone()
    drifted[0, 0, 0, 0, 0] += 1e-7
    audio = _randn(1, 32, 2, 37)
    first = _run_entry(video, audio, None, None)
    second = _run_entry(drifted, audio.clone(), None, None)
    summary, checks = probe.compare_runs(first, second)
    failures = [c for c in checks if c.gate and not c.ok]
    assert len(failures) == 1
    assert "video latent is bitwise identical" in failures[0].name
    assert "max|d|" in failures[0].detail
    assert summary["latent"]["video"]["max_abs"] > 0
    # the audio stream is untouched, so only the video gate falls over
    assert summary["latent"]["audio"]["bitwise"] is True


def test_image_and_audio_differences_are_reported_but_do_not_gate(probe):
    video = _randn(1, 24, 7, 4, 4)
    audio = _randn(1, 32, 2, 37)
    images = _rand(22, 64, 64, 3)
    first = _run_entry(video, audio, images, _randn(1, 2, 100))
    second = _run_entry(video.clone(), audio.clone(), images + 0.5, _randn(1, 2, 100))
    _summary, checks = probe.compare_runs(first, second)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    image_check = [c for c in checks if "IMAGE metrics" in c.name][0]
    assert image_check.gate is False
    assert "bitwise=False" in image_check.detail


def _series_entry(index, *, artifacts=None, rss=0, cuda=0):
    return {
        "index": index,
        "artifacts": artifacts if artifacts is not None else {},
        "memory": {
            "rss_after": rss,
            "cuda_allocated_after": cuda,
            "cuda_reserved_after": cuda,
        },
    }


def _identical_artifacts(seed=0):
    video = _randn(1, 24, 7, 4, 4, seed=seed)
    audio = _randn(1, 32, 2, 37)
    images = _rand(22, 64, 64, 3)
    waveform = _randn(1, 2, 29600)

    def make():
        return {
            "video_latent": video.clone(), "audio_latent": audio.clone(),
            "images": images.clone(), "waveform": waveform.clone(),
        }

    return make


# ==========================================================================
# the outputs a run publishes, and the CPU copies later runs are compared to
# ==========================================================================


def test_check_outputs_hands_back_cpu_copies_of_every_artifact(probe):
    """The comparison material is the tensors themselves, nothing derived."""
    latent, images, audio = _outputs()
    summary, checks, artifacts = probe.check_outputs(
        latent, images, audio, geometry=GEOMETRY
    )
    assert not [c for c in checks if c.gate and not c.ok]
    assert sorted(artifacts) == ["audio_latent", "images", "video_latent", "waveform"]
    for tensor in artifacts.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
    # the copies really are the values that were checked, bitwise
    assert torch.equal(artifacts["images"], images.to("cpu"))
    assert torch.equal(artifacts["waveform"], audio["waveform"].to("cpu"))


def test_the_output_summary_describes_the_tensors_and_stays_json(probe):
    latent, images, audio = _outputs()
    summary, _checks, _artifacts = probe.check_outputs(
        latent, images, audio, geometry=GEOMETRY
    )
    assert summary["video_latent"]["shape"]
    assert summary["images"]["shape"]
    assert summary["audio"]["shape"]
    assert json.loads(json.dumps(summary)) == summary


def test_series_with_one_run_has_nothing_to_gate(probe):
    summary, checks = probe.compare_run_series([_series_entry(1)])
    assert summary["memory"]["mode"] == "single"
    assert summary["comparisons"] == []
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert checks[0].gate is False


def test_series_with_two_runs_keeps_the_strict_gate(probe):
    make = _identical_artifacts()
    runs = [
        _series_entry(1, artifacts=make(), rss=0, cuda=0),
        _series_entry(2, artifacts=make(), rss=10, cuda=10),
    ]
    summary, checks = probe.compare_run_series(runs)
    assert summary["memory"]["mode"] == "strict"
    assert summary["memory"]["gate_window"] == "run 1 -> run 2"
    assert [c.name for c in checks if c.gate and not c.ok] == []
    gated = [c.name for c in checks if c.gate]
    assert "repeat: host RSS did not grow materially" in gated
    assert "repeat: CUDA allocation did not grow materially" in gated


def test_two_runs_still_fail_the_strict_gate_when_they_grow(probe):
    make = _identical_artifacts()
    runs = [
        _series_entry(1, artifacts=make(), rss=0, cuda=0),
        _series_entry(
            2, artifacts=make(),
            rss=probe.RSS_GROWTH_TOLERANCE_BYTES + 1,
            cuda=probe.CUDA_GROWTH_TOLERANCE_BYTES + 1,
        ),
    ]
    _summary, checks = probe.compare_run_series(runs)
    failed = {c.name for c in checks if c.gate and not c.ok}
    assert any("host RSS" in name for name in failed)
    assert any("CUDA allocation" in name for name in failed)


def test_three_runs_gate_the_plateau_and_forgive_the_warm_up(probe):
    """Run 1 -> 2 growth is the ModelPatcher/VAE residency, not a leak."""
    make = _identical_artifacts()
    warm_up = 40 * (1024 ** 3)  # the DiT and both VAEs becoming resident
    runs = [
        _series_entry(1, artifacts=make(), rss=0, cuda=0),
        _series_entry(2, artifacts=make(), rss=warm_up, cuda=warm_up),
        _series_entry(3, artifacts=make(), rss=warm_up + 1024, cuda=warm_up + 1024),
    ]
    summary, checks = probe.compare_run_series(runs)
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    memory = summary["memory"]
    assert memory["mode"] == "plateau"
    assert memory["gate_window"] == "run 2 -> run 3"
    assert memory["gated"]["cuda_allocated_bytes"] == 1024
    # the warm-up is not hidden: it is reported as a diagnostic
    assert memory["diagnostic_total"]["cuda_allocated_bytes"] == warm_up + 1024
    diagnostics = [c for c in checks if not c.gate and "total growth" in c.name]
    assert diagnostics and all(d.gate is False for d in diagnostics)
    assert any(not d.ok for d in diagnostics)  # over tolerance, but not a failure
    assert "warm-up" in memory["note"]


def test_three_runs_fail_when_growth_never_stops(probe):
    make = _identical_artifacts()
    step = probe.CUDA_GROWTH_TOLERANCE_BYTES * 2
    runs = [
        _series_entry(1, artifacts=make(), rss=0, cuda=0),
        _series_entry(2, artifacts=make(), rss=step, cuda=step),
        _series_entry(3, artifacts=make(), rss=2 * step, cuda=2 * step),
    ]
    summary, checks = probe.compare_run_series(runs)
    failed = {c.name for c in checks if c.gate and not c.ok}
    assert any("plateaued once the process was warm" in name for name in failed)
    assert summary["memory"]["gated"]["cuda_allocated_bytes"] == step


def test_a_series_compares_every_run_against_the_first(probe):
    make = _identical_artifacts()
    drifted = make()
    drifted["video_latent"][0, 0, 0, 0, 0] += 1e-6
    runs = [
        _series_entry(1, artifacts=make()),
        _series_entry(2, artifacts=make()),
        _series_entry(3, artifacts=drifted),
    ]
    summary, checks = probe.compare_run_series(runs)
    assert [c["run"] for c in summary["comparisons"]] == [2, 3]
    assert all(c["against"] == 1 for c in summary["comparisons"])
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert failed == ["determinism: video latent is bitwise identical between run 3 and run 1"]
    assert summary["comparisons"][0]["latent"]["video"]["bitwise"] is True
    assert summary["comparisons"][1]["latent"]["video"]["bitwise"] is False


def test_a_drift_that_only_shows_up_late_is_still_caught(probe):
    """Neighbour-comparison would miss a slow drift; run-1 comparison does not."""
    make = _identical_artifacts()
    runs = [_series_entry(1, artifacts=make())]
    for index in range(2, 5):
        artifacts = make()
        # float32 eps near 1.0 is ~1.2e-7; anything smaller would not change a bit
        artifacts["video_latent"][0, 0, 0, 0, 0] += 1e-5 * index
        runs.append(_series_entry(index, artifacts=artifacts))
    _summary, checks = probe.compare_run_series(runs)
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert len(failed) == 3  # runs 2, 3 and 4 all differ from run 1


# -- the three views: vs-first, adjacent, and the full pairwise matrix ------


def _series_of(probe, count, mutate=None):
    make = _identical_artifacts(seed=21)
    runs = []
    for index in range(1, count + 1):
        artifacts = make()
        if mutate is not None:
            mutate(index, artifacts)
        runs.append(_series_entry(index, artifacts=artifacts))
    return probe.compare_run_series(runs)


def test_a_series_reports_adjacent_pairs_as_well_as_the_first(probe):
    summary, _checks = _series_of(probe, 4)
    assert [(c["against"], c["run"]) for c in summary["adjacent"]] == [
        (1, 2), (2, 3), (3, 4),
    ]
    assert [(c["against"], c["run"]) for c in summary["comparisons"]] == [
        (1, 2), (1, 3), (1, 4),
    ]


def test_a_series_records_the_full_pairwise_matrix(probe):
    summary, _checks = _series_of(probe, 4)
    assert [(p["a"], p["b"]) for p in summary["pairwise"]] == [
        (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    ]
    for name in ("video_latent", "audio_latent", "images", "audio"):
        grid = summary["bitwise_matrix"][name]
        assert len(grid) == 4 and all(len(row) == 4 for row in grid)
        assert all(grid[i][i] is True for i in range(4))
        assert all(all(row) for row in grid)  # identical runs agree everywhere


def test_the_matrix_localises_which_runs_disagree(probe):
    def mutate(index, artifacts):
        if index == 3:
            artifacts["images"][0, 0, 0, 0] += 1e-3

    summary, _checks = _series_of(probe, 4, mutate)
    grid = summary["bitwise_matrix"]["images"]
    assert grid[2][2] is True  # run 3 against itself
    assert grid[0][2] is False and grid[2][0] is False
    assert grid[2][3] is False and grid[3][2] is False
    assert grid[0][1] is True and grid[0][3] is True
    # the latents were untouched, so their matrix is still all True
    assert all(all(row) for row in summary["bitwise_matrix"]["video_latent"])


def test_the_warm_plateau_gates_image_and_audio(probe):
    """With three or more runs the last pair is warm, so it is a gate."""
    def mutate(index, artifacts):
        if index == 3:
            artifacts["images"][0, 0, 0, 0] += 1e-3
            artifacts["waveform"][0, 0, 0] += 1e-3

    summary, checks = _series_of(probe, 3, mutate)
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert "determinism: IMAGE is bitwise identical between the last two warm runs (2 and 3)" in failed
    assert "determinism: AUDIO is bitwise identical between the last two warm runs (2 and 3)" in failed
    assert summary["image_reproducibility"]["gated"] is True
    assert summary["image_reproducibility"]["gated_pair"] == [2, 3]
    assert summary["image_reproducibility"]["images_bitwise"] is False
    # the latents are untouched, so the latent gate is silent
    assert not any("latent" in name for name in failed)


def test_a_cold_first_image_is_only_a_diagnostic(probe):
    """Run 1's decode is where a kernel picks its algorithm; that is not a bug."""
    def mutate(index, artifacts):
        if index == 1:
            artifacts["images"][0, 0, 0, 0] += 1e-3
            artifacts["waveform"][0, 0, 0] += 1e-3

    summary, checks = _series_of(probe, 3, mutate)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    # ... and it is still visible, as a number, in both the checks and the report
    diagnostics = [
        c for c in checks if not c.gate and c.name.startswith("determinism: IMAGE metrics")
    ]
    assert diagnostics and any("bitwise=False" in c.detail for c in diagnostics)
    assert summary["bitwise_matrix"]["images"][0][1] is False
    assert summary["image_reproducibility"]["images_bitwise"] is True  # runs 2 and 3 agree


def test_two_runs_leave_image_reproducibility_ungated(probe):
    """One pair, and it is cold-vs-warm: reported, not gated."""
    def mutate(index, artifacts):
        if index == 2:
            artifacts["images"][0, 0, 0, 0] += 1e-3

    summary, checks = _series_of(probe, 2, mutate)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert summary["image_reproducibility"]["gated"] is False
    assert summary["image_reproducibility"]["gated_pair"] is None
    assert "--repeat 3" in summary["image_reproducibility"]["note"]
    assert summary["image_reproducibility"]["images_bitwise"] is False


def test_a_series_compares_the_tensors_themselves(probe):
    """Identical runs match bitwise -- decided on the tensors, not on a hash."""
    summary, _checks = _series_of(probe, 3)
    assert summary["runs"] == 3
    for name in ("video_latent", "audio_latent", "images", "audio"):
        grid = summary["bitwise_matrix"][name]
        assert all(all(row) for row in grid), name
    assert all(entry["latent"]["video"]["bitwise"] for entry in summary["pairwise"])
    assert set(summary) == {
        "runs", "comparisons", "adjacent", "pairwise", "bitwise_matrix",
        "memory", "image_reproducibility",
    }


def test_a_single_run_has_nothing_to_compare_and_says_so(probe):
    """``--repeat 1`` compares nothing: one run is not a determinism claim."""
    summary, _checks = _series_of(probe, 1)
    assert summary["memory"]["mode"] == "single"
    assert summary["comparisons"] == []
    assert set(summary) == {
        "runs", "comparisons", "adjacent", "pairwise", "bitwise_matrix", "memory",
    }


def test_the_series_summary_is_json_and_renders(probe):
    summary, _checks = _series_of(probe, 3)
    report = probe.Report()
    report.determinism = summary
    assert json.loads(json.dumps(report.to_dict(), default=str))
    rendered = report.render()
    assert "IMAGE/AUDIO reproducibility: pair [2, 3] GATED" in rendered
    assert "bitwise matrix images:" in rendered


def test_memory_growth_beyond_tolerance_fails(probe):
    video = torch.zeros(1, 24, 7, 4, 4)
    audio = torch.zeros(1, 32, 2, 37)
    first = _run_entry(video, audio, None, None, rss=0, cuda=0)
    second = _run_entry(
        video.clone(), audio.clone(), None, None,
        rss=probe.RSS_GROWTH_TOLERANCE_BYTES + 1,
        cuda=probe.CUDA_GROWTH_TOLERANCE_BYTES + 1,
    )
    _summary, checks = probe.compare_runs(first, second)
    failed = {c.name for c in checks if c.gate and not c.ok}
    assert any("host RSS" in name for name in failed)
    assert any("CUDA allocation" in name for name in failed)


# ==========================================================================
# output checks
# ==========================================================================


GEOMETRY = {"frames": 22, "k": 1, "width": 64, "height": 64,
            "latent_t": 7, "latent_h": 4, "latent_w": 4, "audio_t": 37}


def _outputs(geometry=GEOMETRY):
    latent = {
        "samples": FakeNestedTensor(
            (
                torch.zeros(1, 24, geometry["latent_t"], geometry["latent_h"], geometry["latent_w"]),
                torch.zeros(1, 32, 2, geometry["audio_t"]),
            )
        )
    }
    images = _rand(geometry["frames"], geometry["height"], geometry["width"], 3)
    audio = {"waveform": torch.zeros(1, 2, geometry["audio_t"] * 800), "sample_rate": 32000}
    return latent, images, audio


def test_well_formed_outputs_pass(probe):
    latent, images, audio = _outputs()
    summary, checks, artifacts = probe.check_outputs(latent, images, audio, geometry=GEOMETRY)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert summary["images"]["shape"] == [22, 64, 64, 3]
    assert summary["audio"]["sample_rate"] == 32000
    assert set(artifacts) == {"video_latent", "audio_latent", "images", "waveform"}
    assert artifacts["images"].device.type == "cpu"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda l, i, a: (l, i[:5], a), "IMAGE shape"),
        (lambda l, i, a: (l, i * 3.0, a), "IMAGE is in [0, 1]"),
        (lambda l, i, a: (l, i.masked_fill(torch.zeros_like(i, dtype=torch.bool), 0) * float("nan"), a), "IMAGE is finite"),
        (lambda l, i, a: (l, i, {"waveform": a["waveform"], "sample_rate": 44100}), "sample rate"),
        (lambda l, i, a: (l, i, {"waveform": a["waveform"][:, :1], "sample_rate": 32000}), "[1, 2, N]"),
        (lambda l, i, a: (l, i, {"waveform": a["waveform"][:, :, :10], "sample_rate": 32000}), "length matches"),
    ],
)
def test_broken_outputs_are_caught(probe, mutate, expected):
    latent, images, audio = mutate(*_outputs())
    _summary, checks, _artifacts = probe.check_outputs(latent, images, audio, geometry=GEOMETRY)
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert any(expected in name for name in failed), failed


# -- the collector half of the pipeline report is now an output gate -------


AUDIO_SAMPLES = GEOMETRY["audio_t"] * 800


def _collector_report(**overrides):
    report = {
        "collected_frames": 22,
        "expected_frames": 22,
        "image_complete": True,
        "image_shape": [22, 64, 64, 3],
        "image_bytes": 22 * 64 * 64 * 3 * 4,
        "image_dtype": "torch.float32",
        "image_device": "cpu",
        "collected_samples": AUDIO_SAMPLES,
        "expected_samples": AUDIO_SAMPLES,
        "audio_complete": True,
        "audio_shape": [1, 2, AUDIO_SAMPLES],
        "audio_bytes": 1 * 2 * AUDIO_SAMPLES * 4,
        "audio_dtype": "torch.float32",
        "audio_device": "cpu",
    }
    report.update(overrides)
    return report


def test_a_complete_collector_report_passes(probe):
    checks = probe.check_collector_report(_collector_report(), geometry=GEOMETRY)
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert any("where the IMAGE was built" in c.name and not c.gate for c in checks)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"collected_frames": 17, "image_complete": False}, "every frame was collected"),
        ({"expected_frames": 39}, "expected_frames is the requested frame count"),
        ({"image_shape": [22, 64, 64]}, "IMAGE buffer is [frames, H, W, 3]"),
        ({"image_shape": [22, 48, 64, 3]}, "IMAGE buffer is [frames, H, W, 3]"),
        ({"image_bytes": 0}, "image_bytes accounts for the whole buffer"),
        ({"image_bytes": 12345}, "image_bytes accounts for the whole buffer"),
    ],
)
def test_a_broken_collector_report_is_a_gate_failure(probe, overrides, expected):
    checks = probe.check_collector_report(_collector_report(**overrides), geometry=GEOMETRY)
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert any(expected in name for name in failed), failed


def test_image_bytes_are_checked_against_the_reported_dtype(probe):
    half = _collector_report(
        image_dtype="torch.float16", image_bytes=22 * 64 * 64 * 3 * 2
    )
    assert [c for c in probe.check_collector_report(half, geometry=GEOMETRY)
            if c.gate and not c.ok] == []


def test_an_unknown_dtype_only_requires_a_non_zero_size(probe):
    odd = _collector_report(image_dtype="torch.uint8", image_bytes=7)
    assert [c for c in probe.check_collector_report(odd, geometry=GEOMETRY)
            if c.gate and not c.ok] == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"collected_samples": 100, "audio_complete": False}, "every audio sample was collected"),
        ({"expected_samples": 12345}, "expected_samples is the clip's audio length"),
        ({"audio_shape": [2, AUDIO_SAMPLES]}, "waveform buffer is [1, 2, samples]"),
        ({"audio_shape": [1, 1, AUDIO_SAMPLES]}, "waveform buffer is [1, 2, samples]"),
        ({"audio_bytes": 0}, "audio_bytes accounts for the whole buffer"),
        ({"audio_bytes": 999}, "audio_bytes accounts for the whole buffer"),
    ],
)
def test_a_broken_audio_collector_report_is_a_gate_failure(probe, overrides, expected):
    checks = probe.check_collector_report(_collector_report(**overrides), geometry=GEOMETRY)
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert any(expected in name for name in failed), failed


def test_the_audio_collector_half_passes_when_it_is_whole(probe):
    checks = probe.check_collector_report(_collector_report(), geometry=GEOMETRY)
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert any("where the AUDIO was built" in c.name and not c.gate for c in checks)


def test_a_cancelled_run_keeps_no_partial_image(probe):
    checks = probe.check_collector_report(
        _collector_report(collected_frames=0, image_shape=[], image_bytes=0,
                          image_complete=False),
        geometry=GEOMETRY, cancelled=True,
    )
    assert [c for c in checks if c.gate and not c.ok] == []
    assert any("kept no partial IMAGE" in c.name for c in checks)


# -- the optional whole-clip reference decode -----------------------------


def test_official_compare_is_refused_at_the_sizes_that_oom(probe):
    ok, reason = probe.official_compare_allowed(probe.latent_geometry(39, 512, 288))
    assert ok is True and "within" in reason

    for frames, width, height in [(192, 512, 288), (362, 512, 288), (39, 1376, 768)]:
        ok, reason = probe.official_compare_allowed(
            probe.latent_geometry(frames, width, height)
        )
        assert ok is False
        assert "OOMed" in reason and "--frames 39" in reason


def test_official_compare_skips_without_running_anything(probe):
    called = []
    summary, checks = probe.compare_official_video(
        video_vae=object(),
        latent={},
        images=torch.zeros(1),
        geometry=probe.latent_geometry(192, 512, 288),
        device=torch.device("cpu"),
        decode_images=lambda vae, latent: called.append(1),
    )
    assert called == []
    assert summary["allowed"] is False
    assert [c for c in checks if c.gate] == []
    assert any("skipped at this geometry" in c.name for c in checks)


def test_official_compare_reports_a_bitwise_match(probe):
    images = _rand(39, 64, 64, 3)
    summary, checks = probe.compare_official_video(
        video_vae=object(),
        latent={},
        images=images,
        geometry=probe.latent_geometry(39, 64, 64),
        device=torch.device("cpu"),
        decode_images=lambda vae, latent: images.clone(),
    )
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert summary["metrics"]["bitwise"] is True
    assert summary["seconds"] >= 0
    match = [c for c in checks if "matches the whole-clip decode" in c.name][0]
    assert match.ok is True and match.gate is False


def test_official_compare_reports_a_mismatch_without_failing(probe):
    images = _rand(39, 64, 64, 3)
    summary, checks = probe.compare_official_video(
        video_vae=object(),
        latent={},
        images=images,
        geometry=probe.latent_geometry(39, 64, 64),
        device=torch.device("cpu"),
        decode_images=lambda vae, latent: images + 1e-3,
    )
    # a different kernel for a different shape is a number, not a verdict
    assert [c.name for c in checks if c.gate and not c.ok] == []
    match = [c for c in checks if "matches the whole-clip decode" in c.name][0]
    assert match.ok is False and match.gate is False
    assert summary["metrics"]["max_abs"] > 0
    assert "rel_l2" in match.detail


def test_official_compare_catches_a_shape_disagreement(probe):
    summary, checks = probe.compare_official_video(
        video_vae=object(),
        latent={},
        images=_rand(39, 64, 64, 3),
        geometry=probe.latent_geometry(39, 64, 64),
        device=torch.device("cpu"),
        decode_images=lambda vae, latent: _rand(34, 64, 64, 3),
    )
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert any("the collector's shape" in name for name in failed)
    assert summary["shape"] == [34, 64, 64, 3]


def test_an_out_of_memory_reference_decode_is_a_failure(probe):
    def oom(vae, latent):
        raise RuntimeError("CUDA out of memory. Tried to allocate 130.22 GiB")

    summary, checks = probe.compare_official_video(
        video_vae=object(),
        latent={},
        images=_rand(39, 64, 64, 3),
        geometry=probe.latent_geometry(39, 64, 64),
        device=torch.device("cpu"),
        decode_images=oom,
    )
    failed = [c for c in checks if c.gate and not c.ok]
    assert len(failed) == 1
    assert "reference decode completed" in failed[0].name
    assert "out of memory" in failed[0].detail
    assert summary["out_of_memory"] is True
    assert "handover had already run" in failed[0].detail


def test_any_reference_decode_failure_is_reported(probe):
    def boom(vae, latent):
        raise ValueError("nope")

    summary, checks = probe.compare_official_video(
        video_vae=object(), latent={}, images=_rand(39, 64, 64, 3),
        geometry=probe.latent_geometry(39, 64, 64),
        device=torch.device("cpu"), decode_images=boom,
    )
    assert summary["out_of_memory"] is False
    assert [c for c in checks if c.gate and not c.ok]


def test_the_official_audio_normalisation_is_checked(probe):
    """``/= max(1, std*5)`` leaves a loud clip at exactly std 0.2."""
    latent, images, audio = _outputs()
    loud = _randn(1, 2, GEOMETRY["audio_t"] * 800, seed=5)
    scaled = loud / (float(torch.std(loud)) * 5.0)  # what finalize_audio does
    summary, checks, _artifacts = probe.check_outputs(
        latent, images, {"waveform": scaled, "sample_rate": 32000}, geometry=GEOMETRY
    )
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert summary["audio"]["std"] == pytest.approx(0.2, abs=1e-3)
    assert "vae_decode_audio tail" in summary["audio"]["normalised"]


def test_a_quiet_clip_is_left_alone_and_still_passes(probe):
    latent, images, audio = _outputs()
    quiet = _randn(1, 2, GEOMETRY["audio_t"] * 800, seed=6) * 0.01
    _summary, checks, _artifacts = probe.check_outputs(
        latent, images, {"waveform": quiet, "sample_rate": 32000}, geometry=GEOMETRY
    )
    # std*5 < 1 so the divisor is 1: unchanged, and below the ceiling
    assert [c.line() for c in checks if c.gate and not c.ok] == []


def test_an_unnormalised_audio_output_is_caught(probe):
    latent, images, audio = _outputs()
    raw = _randn(1, 2, GEOMETRY["audio_t"] * 800, seed=7)  # std ~1.0, never divided
    summary, checks, _artifacts = probe.check_outputs(
        latent, images, {"waveform": raw, "sample_rate": 32000}, geometry=GEOMETRY
    )
    failed = [c for c in checks if c.gate and not c.ok]
    assert any("official whole-clip normalisation" in c.name for c in failed)
    assert summary["audio"]["std"] > 0.2


def test_a_wrong_latent_shape_is_caught(probe):
    latent, images, audio = _outputs()
    latent["samples"].tensors[0] = torch.zeros(1, 24, 9, 4, 4)
    _summary, checks, _artifacts = probe.check_outputs(latent, images, audio, geometry=GEOMETRY)
    assert any("video latent shape" in c.name for c in checks if not c.ok)


def test_a_latent_that_is_not_nested_is_caught(probe):
    _latent, images, audio = _outputs()
    _summary, checks, artifacts = probe.check_outputs(
        {"samples": torch.zeros(1, 24, 7, 4, 4)}, images, audio, geometry=GEOMETRY
    )
    assert artifacts == {}
    assert any("NestedTensor" in c.name for c in checks if not c.ok)


# ==========================================================================
# the streaming cadence: when bytes went out, not just how many
# ==========================================================================


def _emission(chunk, *, fragments=0, muxed_frames=0, muxed_samples=0, held_frames=0):
    return {
        "chunk": chunk, "frames": 17, "samples": 22400,
        "muxed_frames": muxed_frames, "muxed_samples": muxed_samples,
        "fragments": fragments, "fragment_bytes": fragments * 900,
        "held_frames": held_frames, "seconds": 0.5 * (chunk + 1),
    }


def _cadence_report(emissions, *, fragments, before_finish, frames_muxed=192):
    return {
        "chunk_emissions": emissions,
        "fragments": fragments,
        "fragments_before_finish": before_finish,
        "frames_muxed": frames_muxed,
        "first_fragment_chunk": next(
            (e["chunk"] for e in emissions if e["fragments"]), None
        ),
        "first_fragment_latency": 1.25,
    }


def test_a_streaming_run_reports_its_cadence_without_gating_a_ratio(probe):
    emissions = [_emission(0)] + [
        _emission(i, fragments=21, muxed_frames=17, muxed_samples=22400)
        for i in range(1, 11)
    ] + [_emission(11)]
    report = _cadence_report(emissions, fragments=251, before_finish=216)
    summary, checks = probe.check_streaming_cadence(
        report, num_chunks=12, preview_disabled=False
    )
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert summary["emitting_chunks"] == list(range(1, 11))
    assert summary["intermediate_chunks"] == list(range(1, 11))
    assert summary["fragments_before_finish"] == 216
    assert summary["fragments_at_finish"] == 35
    assert summary["fragments_before_finish_ratio"] == pytest.approx(216 / 251)
    assert summary["frames_before_finish_ratio"] == pytest.approx(170 / 192)
    assert summary["emission_log"] == emissions
    # the proportion is measured, never asserted against a number made up here
    ratio_check = [c for c in checks if "how much of the clip" in c.name][0]
    assert ratio_check.gate is False
    assert "Measured, not expected" in ratio_check.detail


def test_a_silent_intermediate_chunk_is_a_failure(probe):
    emissions = [_emission(0)] + [
        _emission(i, fragments=0 if i == 5 else 21, muxed_frames=17)
        for i in range(1, 11)
    ] + [_emission(11)]
    _summary, checks = probe.check_streaming_cadence(
        _cadence_report(emissions, fragments=230, before_finish=195),
        num_chunks=12, preview_disabled=False,
    )
    failed = [c for c in checks if c.gate and not c.ok]
    assert any("every intermediate chunk emitted" in c.name for c in failed)
    assert "[5]" in failed[0].detail


def test_a_run_that_only_emitted_at_the_flush_is_a_failure(probe):
    emissions = [_emission(i) for i in range(12)]
    _summary, checks = probe.check_streaming_cadence(
        _cadence_report(emissions, fragments=251, before_finish=0),
        num_chunks=12, preview_disabled=False,
    )
    failed = [c.name for c in checks if c.gate and not c.ok]
    assert "streaming: fragments went out before the tail flush" in failed
    assert any("every intermediate chunk emitted" in name for name in failed)


def test_a_two_chunk_clip_is_all_startup_delay_and_is_not_gated(probe):
    """22 frames is shorter than the cadence it would be measured against."""
    emissions = [_emission(0), _emission(1)]
    summary, checks = probe.check_streaming_cadence(
        _cadence_report(emissions, fragments=30, before_finish=0, frames_muxed=22),
        num_chunks=2, preview_disabled=False,
    )
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    early = [c for c in checks if "before the tail flush" in c.name][0]
    assert early.gate is False and "startup delay" in early.detail
    assert summary["intermediate_chunks"] == []


def test_a_missing_emission_record_is_a_failure(probe):
    _summary, checks = probe.check_streaming_cadence(
        _cadence_report([_emission(0)], fragments=10, before_finish=5),
        num_chunks=3, preview_disabled=False,
    )
    assert any(
        "every chunk is accounted for" in c.name for c in checks if c.gate and not c.ok
    )


def test_a_disabled_preview_has_no_cadence_to_measure(probe):
    summary, checks = probe.check_streaming_cadence(
        {"chunk_emissions": [], "fragments": 0, "fragments_before_finish": 0},
        num_chunks=12, preview_disabled=True,
    )
    assert [c for c in checks if c.gate] == []
    assert checks[0].name == "streaming: cadence not measurable"
    assert summary["fragments"] == 0


@pytest.mark.parametrize(
    "frames, chunks, expect_early",
    [(22, 2, False), (39, 3, True), (192, 12, True)],
)
def test_the_real_pipeline_cadence_at_each_size(probe, frames, chunks, expect_early):
    """The cadence claims, measured against the real collector and muxer.

    Not a fixture: this drives the actual ``StreamingPipeline`` over the fake
    VAEs, so the emission pattern is the one this repository's code produces.
    """
    pytest.importorskip("av")
    geometry = probe.latent_geometry(frames, 64, 64)
    assert probe.layout_num_chunks(geometry) == chunks

    sender = preview_mod.RecordingSender()
    manager = preview_mod.PreviewManager(sender=sender)
    inst = probe.Instrumentation()
    with manager.session("cadence") as session:
        sink = preview_mod.PreviewMediaSink(session)
        with probe.instrument(inst):
            pipeline, report, image = _run_a_real_collector(
                probe, inst, sink=sink, geometry=geometry
            )
            audio = pipeline.finalize_audio(socketed_audio_vae())

    payload = report.to_dict()
    summary, checks = probe.check_streaming_cadence(
        payload, num_chunks=chunks, preview_disabled=report.preview_disabled
    )
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert summary["chunks_recorded"] == chunks
    assert (summary["fragments_before_finish"] > 0) is expect_early
    if expect_early:
        # every intermediate chunk emitted, which is the "as soon as it can" claim
        assert summary["emitting_chunks"] == list(range(1, chunks - 1))
        assert 0 < summary["frames_before_finish_ratio"] < 1
    # and both collectors are complete regardless of the cadence
    assert payload["collected_frames"] == payload["expected_frames"] == frames
    assert payload["collected_samples"] == payload["expected_samples"]
    assert list(image.shape) == [frames, 64, 64, 3]
    assert list(audio["waveform"].shape) == [1, 2, payload["expected_samples"]]
    assert audio["sample_rate"] == 32000
    assert inst.finalize_image_calls == 1 and inst.finalize_audio_calls == 1
    assert inst.timings["audio_finalize_seconds"] >= 0


# ==========================================================================
# the phase swap: how many DiT-phase loads a run makes
# ==========================================================================


@pytest.mark.parametrize(
    "frames, chunks",
    [
        (22, 2),    # k=1: one full chunk plus the 2-latent tail
        (39, 3),    # the probe's own request -- what the vr run measured
        (192, 12),  # the published request
        (362, 22),  # the documented maximum
    ],
)
def test_the_chunk_count_comes_from_the_layout(probe, frames, chunks):
    geometry = probe.latent_geometry(frames, 64, 64)
    assert probe.layout_num_chunks(geometry) == chunks
    # ... and it is a property of the video grid, not of the prompt length
    assert probe.layout_num_chunks(geometry, text_len=512) == chunks


@pytest.mark.parametrize("chunks", [2, 3, 12, 22])
def test_a_completed_rollout_loads_the_dit_once_per_chunk(probe, chunks):
    """Initial load + one reload after each non-last chunk == num_chunks."""
    assert probe.expected_dit_loads(chunks) == chunks
    assert probe.expected_dit_loads(chunks) == 1 + (chunks - 1)


def test_a_single_chunk_rollout_still_loads_once(probe):
    assert probe.expected_dit_loads(1) == 1
    assert probe.expected_dit_loads(0) == 1  # nothing sensible is below one


@pytest.mark.parametrize(
    "delivered, chunks, expected",
    [
        (0, 3, 1),    # cancelled inside the first chunk's forwards
        (1, 3, 2),    # one chunk delivered, its reload done
        (2, 3, 3),    # both non-last chunks delivered
        (3, 3, 3),    # even if the last one landed, it never reloads
        (11, 12, 12),
        (21, 22, 22),
        (0, 1, 1),
    ],
)
def test_a_cancelled_rollout_derives_its_count_from_what_finished(
    probe, delivered, chunks, expected
):
    assert probe.dit_loads_after_cancel(delivered, chunks) == expected


def test_the_39_frame_vr_geometry_expects_three_dit_loads(probe):
    """The exact case the old gate failed on: 3, not 1."""
    geometry = probe.latent_geometry(39, 512, 288)
    assert probe.layout_num_chunks(geometry) == 3
    assert probe.expected_dit_loads(probe.layout_num_chunks(geometry)) == 3


# ==========================================================================
# instrumentation: transparent wrappers that put everything back
# ==========================================================================


def test_instrument_restores_every_wrapped_attribute(probe):
    from raven_streaming import cache as cache_mod

    # ``StreamingPipeline.inert`` is gone from the runtime: there is no such
    # thing as a pipeline that does not collect, because the collector is the
    # IMAGE. Nothing here may reach for it.
    assert not hasattr(pipeline_mod.StreamingPipeline, "inert")

    before = (
        nodes_mod.rollout_memory_budget,
        nodes_mod.make_load_models,
        nodes_mod.decode_images,
        nodes_mod.decode_audio,
        pipeline_mod.build_media_pipeline,
        cache_mod.ChunkKVCache.__dict__["discard_pending"],
    )
    inst = probe.Instrumentation()
    with probe.instrument(inst):
        assert nodes_mod.decode_images is not before[2]
    after = (
        nodes_mod.rollout_memory_budget,
        nodes_mod.make_load_models,
        nodes_mod.decode_images,
        nodes_mod.decode_audio,
        pipeline_mod.build_media_pipeline,
        cache_mod.ChunkKVCache.__dict__["discard_pending"],
    )
    assert before == after


def _run_a_real_collector(probe, inst, *, sink=None, muxer=None, geometry=None):
    """Drive a real ``StreamingPipeline`` over fake VAEs, under instrumentation."""
    geometry = geometry or FAKE_GEOMETRY
    config = pipeline_mod.PipelineConfig(
        frames=geometry["frames"], width=geometry["width"], height=geometry["height"]
    )
    layout = layout_mod.T2VALayout.from_request(
        text_len=8, frames=geometry["frames"],
        width=geometry["width"], height=geometry["height"],
    )
    kwargs = {} if muxer is None else {"muxer": muxer}
    pipeline = pipeline_mod.build_media_pipeline(
        video_vae=FakeVAEWrapper(FakeVideoInner(geometry["height"], geometry["width"])),
        audio_vae=FakeVAEWrapper(FakeAudioInner()),
        config=config,
        sink=sink,
        **kwargs,
    )
    if sink is not None:
        pipeline.open_preview()
    for chunk in layout.chunks:
        pipeline.on_chunk(
            ChunkOutput(
                index=chunk.index,
                is_last=chunk.index == layout.num_chunks - 1,
                video_start=chunk.video_start,
                video_stop=chunk.video_stop,
                audio_start=chunk.audio_start,
                audio_stop=chunk.audio_stop,
                video_x0=torch.zeros(
                    1, 24, chunk.video_latents, geometry["latent_h"], geometry["latent_w"]
                ),
                audio_x0=torch.zeros(1, 32, 2, chunk.audio_latents),
            )
        )
    report = pipeline.finish()
    image = pipeline.finalize_image()
    return pipeline, report, image


def test_instrument_captures_the_pipeline_and_times_both_flushes(probe):
    inst = probe.Instrumentation()
    with probe.instrument(inst):
        pipeline, report, image = _run_a_real_collector(probe, inst)
    assert inst.pipeline is pipeline
    assert inst.timings["preview_flush_seconds"] >= 0.0
    # the collector handover is timed under its own name; there is no
    # whole-clip video decode to time any more
    assert inst.timings["image_finalize_seconds"] >= 0.0
    assert "final_video_decode_seconds" not in inst.timings
    assert list(image.shape) == [22, 64, 64, 3]
    assert report.collected_frames == report.expected_frames == 22
    assert report.image_complete is True


def test_instrument_counts_the_whole_clip_decode_instead_of_timing_it(probe):
    from raven_streaming.cache import ChunkKVCache

    calls = []
    inst = probe.Instrumentation()
    with probe.instrument(inst):
        ChunkKVCache(2, sink=2, window=2).discard_pending()
        nodes_mod.decode_audio(None, {"samples": None}, helper=lambda vae, latent: "audio")
        assert nodes_mod.decode_images is not None
        nodes_mod.decode_images(
            SimpleNamespace(decode=lambda video: calls.append(video) or torch.zeros(3, 4, 4, 3)),
            {"samples": FakeNestedTensor((torch.zeros(1, 24, 7, 4, 4), torch.zeros(1, 32, 2, 37)))},
        )
    assert inst.discard_pending_calls == 1
    # both whole-clip helpers are counted, so "the node never does this" is
    # checkable; neither is timed, because the product path has no such call
    assert inst.decode_images_calls == 1
    assert inst.decode_audio_calls == 1
    assert "final_video_decode_seconds" not in inst.timings
    assert "final_audio_decode_seconds" not in inst.timings
    assert len(calls) == 1


def test_watch_vae_decodes_counts_and_restores(probe):
    video = socketed_video_vae()
    audio = socketed_audio_vae()
    original = video.decode
    counts = {}
    with probe.watch_vae_decodes(counts, video=video, audio=audio):
        video.decode(torch.zeros(1, 24, 7, 4, 4))
        video.decode(torch.zeros(1, 24, 7, 4, 4))
    assert counts == {"video": 2, "audio": 0}
    assert video.decode.__func__ is original.__func__
    assert "decode" not in vars(video)


def test_watch_vae_decodes_tolerates_a_socket_without_decode(probe):
    counts = {}
    with probe.watch_vae_decodes(counts, video=SimpleNamespace()):
        pass
    assert counts == {"video": 0}


def test_instrument_wraps_load_models_without_changing_it(probe):
    calls = []
    inst = probe.Instrumentation()
    with probe.instrument(inst):
        closure = nodes_mod.make_load_models(
            None, None, 4242,
            load_models_gpu=lambda models, memory_required=0, force_full_load=False: calls.append(
                memory_required
            ),
        )
        closure(["model"])
    assert calls == [4242]  # the reserve still reaches Comfy untouched
    assert inst.load_calls == 1
    assert inst.timings["model_load_seconds"] >= 0.0


# ==========================================================================
# cancel / repeat
# ==========================================================================


class Forwarder:
    def __init__(self):
        self.calls = 0

    def forward_chunk(self, **kwargs):
        self.calls += 1
        return "velocity"


def test_cancel_after_forwards_lets_n_real_forwards_through(probe):
    target = Forwarder()
    with probe.cancel_after_forwards(target, 3) as counter:
        assert target.forward_chunk() == "velocity"
        assert target.forward_chunk() == "velocity"
        with pytest.raises(SamplingCancelled, match="after 3"):
            target.forward_chunk()
    assert counter.calls == 3
    assert counter.raised is True
    # the third forward really ran before the cancellation
    assert target.calls == 3
    # and the instance attribute is gone again
    assert "forward_chunk" not in vars(target)


def test_count_forwards_never_interrupts(probe):
    target = Forwarder()
    with probe.count_forwards(target) as counter:
        for _ in range(5):
            target.forward_chunk()
    assert counter.calls == 5 and counter.raised is False


def test_the_cancellation_is_the_packages_own_exception(probe):
    target = Forwarder()
    with probe.cancel_after_forwards(target, 1):
        with pytest.raises(SamplingCancelled) as excinfo:
            target.forward_chunk()
    # classified as "the user stopped this", not as a fault
    from raven_streaming.preview_session import _looks_like_cancellation

    assert _looks_like_cancellation(excinfo.value)


# ==========================================================================
# execute_run, against a fake sampler that uses the real preview lane
# ==========================================================================


FAKE_GEOMETRY = {"frames": 22, "k": 1, "width": 64, "height": 64,
                 "latent_t": 7, "latent_h": 4, "latent_w": 4, "audio_t": 37}


class FakeVideoInner:
    """Enough of ``MiniMaxH3VideoVAE`` for the real streaming decoder."""

    clip_length = 17
    vae_ratio_t = 4
    token_drop = 3

    def __init__(self, height=64, width=64):
        self.height = height
        self.width = width
        self.latents_mean = torch.zeros(24)
        self.latents_std = torch.ones(24)

    def _adaptive_decode(self, z):
        b, _c, t, _h, _w = z.shape
        frames = t * self.vae_ratio_t
        ramp = torch.linspace(0.0, 1.0, steps=frames).reshape(1, 1, frames, 1, 1)
        return ramp.expand(b, 3, frames, self.height, self.width).clone()

    def blend(self, a, b, blend_extent, dim):
        extent = min(a.shape[dim], b.shape[dim], blend_extent)
        weights = torch.arange(extent, dtype=b.dtype) / extent
        shape = [1] * a.ndim
        shape[dim] = extent
        weight_b = weights.reshape(shape)
        tail = a.narrow(dim, a.shape[dim] - extent, extent)
        head = b.narrow(dim, 0, extent)
        blended = tail * (1 - weight_b) + head * weight_b
        if extent < b.shape[dim]:
            rest = b.narrow(dim, extent, b.shape[dim] - extent)
            return torch.cat([blended, rest], dim=dim)
        return blended

    def _finalize_pixels(self, part):
        return part.clamp(0.0, 1.0)


class FakeAudioInner:
    samples_per_latent = 800
    sample_rate = 32000

    def decode(self, z):
        b, _c, channels, t = z.shape
        return torch.zeros(b, channels, t * self.samples_per_latent)


class BrokenMuxer:
    """A muxer that dies on the first frame. The collector must not care."""

    def take_init_segment(self):
        return None

    def take_fragments(self):
        return []

    def write_video_frame(self, image, force_keyframe=None):
        raise RuntimeError("libx264 fell over")

    def write_audio(self, pcm):
        return 0

    def close(self):
        return None


class FakeVAEWrapper:
    """The ``comfy.sd.VAE`` surface both the collector and the node read.

    The socket attributes (``latent_channels`` and friends) are real values
    rather than stubs so ``nodes.resolve_video_vae`` / ``resolve_audio_vae``
    can run for real against this object -- the probe's VAE gate is not
    bypassed anywhere in these tests.
    """

    def __init__(self, inner, **attributes):
        self.first_stage_model = inner
        self.device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.output_device = torch.device("cpu")
        self.patcher = SimpleNamespace(name=type(inner).__name__)
        for key, value in attributes.items():
            setattr(self, key, value)

    def memory_used_decode(self, shape, dtype):
        return 1024

    def decode(self, samples, vae_options={}):
        # [B, C, T, H, W] -> [B, T, H*16, W*16, 3], as comfy.sd.VAE.decode does
        b, _c, t, h, w = samples.shape
        frames = max(1, (int(t) - 2) // 5 * 17 + 5)
        return torch.zeros((b, frames, h * 16, w * 16, 3))


class MiniMaxH3VideoVAE(FakeVideoInner):  # noqa: N801 - the class NAME is the check
    """``nodes.resolve_video_vae`` identifies the VAE by its class name."""


class MiniMaxH3AudioVAE(FakeAudioInner):  # noqa: N801
    """``nodes.resolve_audio_vae`` identifies the VAE by its class name."""


def socketed_video_vae(height=64, width=64):
    """A video VAE the node's own feature probe accepts, unmodified."""
    return FakeVAEWrapper(
        MiniMaxH3VideoVAE(height, width), latent_channels=24, latent_dim=3
    )


def socketed_audio_vae():
    return FakeVAEWrapper(
        MiniMaxH3AudioVAE(),
        latent_channels=32,
        latent_dim=2,
        output_channels=2,
        audio_sample_rate=32000,
        upscale_ratio=800,
    )


@pytest.fixture
def stand_in_vae_class(monkeypatch):
    """Let a stand-in VAE past the node's ``isinstance(vae, comfy.sd.VAE)`` gate.

    ``nodes._require_vae_wrapper`` applies that check only when ComfyUI happens
    to be importable, and whether it is depends on what some *other* test in
    the session imported first -- which is not a property these tests are
    about. Switching it off explicitly is what ``tests/test_nodes_sampler.py``
    does for the same reason, and it is the only thing switched off: every
    feature probe (inner class name, latent channels, latent_dim, the decoder
    methods, the audio geometry) still runs for real against these objects, and
    the real class is covered in ``tests/test_nodes_upstream.py``.
    """
    monkeypatch.setattr(nodes_mod, "_comfy_vae_class", lambda: None)


def test_the_socketed_fakes_pass_the_nodes_own_vae_gate(probe, stand_in_vae_class):
    """The regression tests below must not be leaning on a bypassed gate."""
    assert nodes_mod.resolve_video_vae(socketed_video_vae()).kind == "video"
    assert nodes_mod.resolve_audio_vae(socketed_audio_vae()).kind == "audio"


@dataclass
class FakeSampler:
    """A miniature of ``RAVENStreamingSampler.sample``.

    It calls the same collaborators the real node does -- ``preview_sink``,
    ``build_media_pipeline``, ``make_load_models``, the DiT's
    ``forward_chunk``, ``pipeline.finish``, ``pipeline.finalize_image`` -- so
    ``execute_run`` sees the shape of a real execution. What it does *not* do is
    sample: the latents it hands the pipeline are zeros.

    Like the real node, it never calls ``nodes.decode_images``: the IMAGE is
    whatever the collector built.
    """

    diffusion_model: object
    geometry: dict
    fail_with: BaseException = None
    #: a muxer stub that breaks mid-run, to exercise "preview died, collector
    #: carried on"
    broken_muxer: bool = False
    #: what the node splices into the budget detail from PhaseSwapCoordinator
    phase_record: dict = field(default_factory=dict)
    #: write that record *before* sampling, i.e. as a runtime that recorded the
    #: swap on the cancel path too would
    phase_record_early: bool = False
    #: report a pre-flush fragment count that the pipeline report contradicts
    corrupt_emission_record: bool = False
    #: every ``kv_cache_storage`` this fake was handed, so the probe's own
    #: pass-through can be asserted rather than assumed
    kv_cache_storages: list = field(default_factory=list)

    def sample(self, *, model, positive, latent, video_vae, audio_vae, seed, steps,
               video_shift, audio_shift, sink, window, unique_id,
               kv_cache_storage=nodes_mod.DEFAULT_KV_CACHE_STORAGE):
        from raven_streaming import consistency
        from raven_streaming.cache import ChunkKVCache

        # The real node validates this before it opens a session; so does this
        # miniature, so a probe that stopped passing the flag through would
        # fail here rather than silently sample in the default mode.
        assert kv_cache_storage in nodes_mod.KV_CACHE_STORAGE_CHOICES, kv_cache_storage
        self.kv_cache_storages.append(str(kv_cache_storage))

        # the rollout's own contract: whatever escapes, the staged K/V goes
        # (``consistency._Rollout.run``)
        cache = ChunkKVCache(1, sink=int(sink), window=int(window))
        config = pipeline_mod.PipelineConfig(
            frames=self.geometry["frames"],
            width=self.geometry["width"],
            height=self.geometry["height"],
        )
        layout = layout_mod.T2VALayout.from_request(
            text_len=8,
            frames=self.geometry["frames"],
            width=self.geometry["width"],
            height=self.geometry["height"],
        )
        # the node's own sockets and its own price for the rollout, so the
        # budget the probe captures is a real one and the phase record has
        # somewhere to land
        video = nodes_mod.resolve_video_vae(video_vae)
        audio = nodes_mod.resolve_audio_vae(audio_vae)
        budget = nodes_mod.rollout_memory_budget(
            model=model,
            layout=layout,
            text_len=8,
            config=consistency.SamplerConfig(
                steps=int(steps), video_shift=float(video_shift),
                audio_shift=float(audio_shift), sink=int(sink),
                window=int(window), seed=int(seed),
            ),
            video=video,
            audio=audio,
            pipeline_config=config,
            latent_dtype=torch.float32,
        )

        if self.phase_record_early:
            budget.detail.update(self.phase_record)

        extra = {"muxer": BrokenMuxer()} if self.broken_muxer else {}
        with nodes_mod.preview_sink(str(unique_id)) as media_sink:
            pipeline = pipeline_mod.build_media_pipeline(
                video_vae=video_vae, audio_vae=audio_vae, config=config,
                sink=media_sink,
                memory_budget=budget.to_dict(),
                **extra,
            )
            try:
                pipeline.open_preview()
                pipeline.status("model_loading")
                load_models = nodes_mod.make_load_models(
                    None, None, 1234,
                    load_models_gpu=lambda models, memory_required=0, force_full_load=False: None,
                    on_loaded=lambda: pipeline.status("sampling"),
                )
                # The rollout's own first load, before any forward.
                load_models([model], memory_required=0, force_full_load=False)
                for chunk in layout.chunks:
                    for _ in range(int(steps)):
                        self.diffusion_model.forward_chunk(chunk_index=chunk.index)
                    is_last = chunk.index == layout.num_chunks - 1
                    # PhaseSwapCoordinator.on_chunk: VAE phase, deliver, and --
                    # for every chunk but the last -- the DiT back for the next
                    # forward. The VAE-phase load goes through Comfy directly,
                    # not through this closure, so only the reload is counted.
                    pipeline.on_chunk(
                        ChunkOutput(
                            index=chunk.index,
                            is_last=is_last,
                            video_start=chunk.video_start,
                            video_stop=chunk.video_stop,
                            audio_start=chunk.audio_start,
                            audio_stop=chunk.audio_stop,
                            video_x0=torch.zeros(
                                1, 24, chunk.video_latents,
                                self.geometry["latent_h"], self.geometry["latent_w"],
                            ),
                            audio_x0=torch.zeros(1, 32, 2, chunk.audio_latents),
                        )
                    )
                    if not is_last:
                        load_models([model], memory_required=0, force_full_load=False)
                if self.fail_with is not None:
                    raise self.fail_with
            except BaseException:
                cache.discard_pending()
                pipeline.cancel()
                raise
            pipeline.status("finalizing")
            pipeline.finish()
            # PhaseSwapCoordinator.to_dict(), spliced into the budget detail on
            # the way out -- exactly where the node puts it
            budget.detail["emission_log"] = [
                emission.to_dict() for emission in pipeline.report().chunk_emissions
            ]
            budget.detail["fragments_before_finish"] = (
                999 if self.corrupt_emission_record
                else pipeline.report().fragments_before_finish
            )
            budget.detail.update(self.phase_record or {
                "phase_swap_chunks": layout.num_chunks,
                "phase_swap_vae_loads": layout.num_chunks,
                "phase_swap_audio_vae_loads": layout.num_chunks,
                "phase_swap_dit_loads": layout.num_chunks - 1,
                "phase_swap_last_phase": "vae",
            })
            # both outputs the workflow gets, straight out of the collectors
            images = pipeline.finalize_image()
            audio_out = pipeline.finalize_audio(audio_vae)
            latent_out, _decoded_images, _decoded_audio = _outputs(self.geometry)
            return (latent_out, images, audio_out)


def _make_fake_context(probe, *, geometry=None, frames=22, extra_args=(), cancel_args=None):
    """A ProbeContext whose sampler is a faithful miniature of the node.

    ``cancel_args`` defaults to the forward-cancel the cancel tests use. The
    two cancellation points are mutually exclusive on the command line, so a
    chunk-cancel context has to replace it rather than add to it.
    """
    geometry = geometry or FAKE_GEOMETRY
    manager = preview_mod.default_manager()
    recorder = preview_mod.RecordingSender()
    previous = manager.sender
    manager.set_sender(recorder)
    forwarder = Forwarder()
    cancel_args = ["--cancel-after-forward", "3"] if cancel_args is None else list(cancel_args)
    args = probe.build_parser().parse_args(
        REQUIRED_ARGS
        + ["--frames", str(geometry["frames"]),
           "--width", str(geometry["width"]), "--height", str(geometry["height"]),
           "--steps", "2"]
        + cancel_args
        + list(extra_args)
    )
    context = probe.ProbeContext(
        args=args,
        env=probe.ComfyEnv(
            root="/nowhere",
            comfy=SimpleNamespace(
                nested_tensor=SimpleNamespace(NestedTensor=FakeNestedTensor),
                model_management=SimpleNamespace(intermediate_device=lambda: torch.device("cpu")),
            ),
            folder_paths=None,
            upstream_nodes=None,
        ),
        device=torch.device("cpu"),
        geometry=geometry,
        conditioning=probe.build_conditioning(8, 0)[0],
        model=SimpleNamespace(),
        video_vae=socketed_video_vae(geometry["height"], geometry["width"]),
        audio_vae=socketed_audio_vae(),
        manager=manager,
        recorder=recorder,
        sampler=FakeSampler(diffusion_model=forwarder, geometry=geometry),
        diffusion_model=forwarder,
    )
    return context, manager, previous


@contextlib.contextmanager
def fake_context_at(probe, geometry):
    context, manager, previous = _make_fake_context(probe, geometry=geometry)
    try:
        yield context
    finally:
        manager.set_sender(previous)
        for session in list(manager.active_sessions):
            manager.cleanup(session.session_id)


@pytest.fixture
def fake_context(probe, stand_in_vae_class):
    pytest.importorskip("av")
    with fake_context_at(probe, FAKE_GEOMETRY) as context:
        yield context


def test_execute_run_records_a_whole_successful_run(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "sample")
    failures = [c for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]
    assert payload["ok"] is True
    assert payload["kind"] == "sample"
    # one DiT-phase load per chunk: the initial one plus one after each
    # non-last chunk (the phase swap), derived from the layout not hard-coded
    assert payload["num_chunks"] == 2
    assert payload["dit_phase_loads"] == probe.expected_dit_loads(2) == 2
    assert payload["forward_chunk_calls"] == 4  # 2 chunks x 2 steps
    assert payload["timings"]["sample_seconds"] > 0
    assert payload["timings"]["preview_flush_seconds"] >= 0
    # the media lane really ran: fragments, muxed frames and a decodable stream
    assert payload["pipeline"]["fragments"] >= 1
    assert payload["pipeline"]["frames_muxed"] == 22
    assert payload["pipeline"]["first_fragment_chunk"] is not None
    # the pipeline carries the budget the node priced, and it is a real one
    assert payload["pipeline"]["memory_budget"]["total_bytes"] > 0
    assert payload["memory_budget"]["detail"]["phase_swap_chunks"] == 2
    assert payload["preview"]["end_reason"] == "complete"
    assert payload["media"]["video_frames"] == 22
    assert payload["media"]["audio_samples"] > 0
    assert payload["sessions"]["active"] == 0
    assert set(payload["artifacts"]) == {"video_latent", "audio_latent", "images", "waveform"}
    assert payload["memory"]["rss_after"] > 0


def test_execute_run_gates_the_collector_and_the_image(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "sample")
    names = [c.name for c in payload["_checks"]]
    for gate in (
        "collector: expected_frames is the requested frame count",
        "collector: every frame was collected",
        "collector: the IMAGE buffer is [frames, H, W, 3]",
        "collector: image_bytes accounts for the whole buffer",
    ):
        check = [c for c in payload["_checks"] if c.name == gate]
        assert check and check[0].gate is True and check[0].ok is True, gate
    assert "run: the node called no whole-clip decode helper" in names
    assert "run: neither VAE's whole-clip decode() was called" in names
    # the IMAGE really came out of the collector, and is the buffer it reports
    assert payload["pipeline"]["image_shape"] == [22, 64, 64, 3]
    assert payload["outputs"]["images"]["shape"] == [22, 64, 64, 3]
    assert payload["timings"]["image_finalize_seconds"] >= 0
    assert "final_video_decode_seconds" not in payload["timings"]
    assert payload["preview_disabled"] is False
    assert payload["preview_disabled_reason"] == ""


def test_execute_run_gates_both_collectors_and_both_finalisers(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "sample")
    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures

    for gate in (
        "collector: expected_samples is the clip's audio length",
        "collector: every audio sample was collected",
        "collector: the waveform buffer is [1, 2, samples]",
        "collector: audio_bytes accounts for the whole buffer",
        "run: each collector handed over exactly once",
        "run: the node called no whole-clip decode helper",
        "run: neither VAE's whole-clip decode() was called",
    ):
        check = [c for c in payload["_checks"] if c.name == gate]
        assert check and check[0].gate is True and check[0].ok is True, gate

    samples = FAKE_GEOMETRY["audio_t"] * 800
    assert payload["pipeline"]["collected_samples"] == samples
    assert payload["pipeline"]["audio_shape"] == [1, 2, samples]
    assert payload["pipeline"]["audio_complete"] is True
    assert payload["outputs"]["audio"]["shape"] == [1, 2, samples]
    assert payload["timings"]["audio_finalize_seconds"] >= 0
    assert "final_audio_decode_seconds" not in payload["timings"]
    # the emission log and the streaming summary are recorded for every run
    assert payload["streaming"]["chunks_recorded"] == 2
    assert payload["streaming"]["emission_log"]
    # the node's own copy of the record agrees with the pipeline report
    assert payload["streaming"]["node_emission_log_entries"] == 2
    assert payload["streaming"]["node_fragments_before_finish"] == 0
    agreement = [c for c in payload["_checks"]
                 if c.name == "streaming: the node's own emission record matches the report"]
    assert agreement and agreement[0].ok is True
    assert payload["pipeline"]["fragments_before_finish"] == 0  # 2 chunks: all flush


def test_a_disagreeing_emission_record_is_a_failure(probe, fake_context):
    """Two accounts of the same cadence must not contradict each other."""
    fake_context.sampler.corrupt_emission_record = True
    payload = probe.execute_run(fake_context, 1, "sample")
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert "streaming: the node's own emission record matches the report" in failed
    assert payload["streaming"]["node_fragments_before_finish"] == 999
    assert payload["pipeline"]["fragments_before_finish"] == 0


def test_a_node_that_decodes_a_whole_stream_fails_the_gate(probe, fake_context):
    """Either helper, or either VAE socket, is enough to fail it."""
    original = FakeSampler.sample

    def sample_and_decode(self, **kwargs):
        outputs = original(self, **kwargs)
        kwargs["audio_vae"].decode(torch.zeros(1, 32, 2, 4))
        return outputs

    fake_context.sampler.__class__.sample = sample_and_decode
    try:
        payload = probe.execute_run(fake_context, 1, "sample")
    finally:
        fake_context.sampler.__class__.sample = original

    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert "run: neither VAE's whole-clip decode() was called" in failed
    assert payload["_checks"]  # and the count is in the detail
    detail = [c.detail for c in payload["_checks"]
              if c.name == "run: neither VAE's whole-clip decode() was called"][0]
    assert "'audio': 1" in detail


def test_the_finaliser_counters_count_every_handover(probe):
    """The gate pins the *count*, so the counter has to be one per call."""
    pytest.importorskip("av")
    inst = probe.Instrumentation()
    with probe.instrument(inst):
        pipeline, _report, image = _run_a_real_collector(probe, inst)
        again = pipeline.finalize_image()          # idempotent, but counted
        audio = pipeline.finalize_audio(socketed_audio_vae())
        audio_again = pipeline.finalize_audio(socketed_audio_vae())
    assert inst.finalize_image_calls == 2
    assert inst.finalize_audio_calls == 2
    # the runtime hands back the same objects, which is why the probe counts
    # calls rather than trusting that a second one is harmless
    assert again is image
    assert audio_again is audio


def test_the_streaming_gates_fire_at_a_size_that_has_a_cadence(probe, stand_in_vae_class):
    """39 frames: chunk 1 must emit before the flush."""
    pytest.importorskip("av")
    geometry = probe.latent_geometry(39, 64, 64)
    with fake_context_at(probe, geometry) as context:
        payload = probe.execute_run(context, 1, "sample")
    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures
    streaming = payload["streaming"]
    assert streaming["fragments_before_finish"] > 0
    assert streaming["emitting_chunks"] == [1]
    assert 0 < streaming["frames_before_finish_ratio"] < 1
    gate = [c for c in payload["_checks"]
            if c.name == "streaming: fragments went out before the tail flush"][0]
    assert gate.gate is True and gate.ok is True


def test_a_dead_preview_does_not_take_the_image_with_it(probe, fake_context):
    """The collector is the IMAGE; the preview is the optional half."""
    fake_context.sampler.broken_muxer = True
    payload = probe.execute_run(fake_context, 1, "sample")

    # the preview really did die, and the probe says so
    assert payload["preview_disabled"] is True
    assert "libx264 fell over" in payload["preview_disabled_reason"]
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert "preview: the media lane stayed enabled" in failed

    # ... and the collector still produced the whole IMAGE
    collector_failures = [name for name in failed if name.startswith("collector:")]
    assert collector_failures == []
    assert payload["pipeline"]["collected_frames"] == 22
    assert payload["pipeline"]["image_complete"] is True
    assert payload["outputs"]["images"]["shape"] == [22, 64, 64, 3]
    assert [name for name in failed if name.startswith("output:")] == []
    assert payload["artifacts"]["images"] is not None


def test_a_node_that_ran_the_whole_clip_decode_fails_the_gate(probe, fake_context):
    original = FakeSampler.sample

    def sample_with_a_second_decode(self, **kwargs):
        outputs = original(self, **kwargs)
        nodes_mod.decode_images(
            SimpleNamespace(decode=lambda video: torch.zeros(1, 22, 64, 64, 3)),
            outputs[0],
        )
        return outputs

    fake_context.sampler.__class__.sample = sample_with_a_second_decode
    try:
        payload = probe.execute_run(fake_context, 1, "sample")
    finally:
        fake_context.sampler.__class__.sample = original
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert "run: the node called no whole-clip decode helper" in failed


def test_execute_run_can_compare_against_the_official_decode(probe, fake_context):
    fake_context.args.compare_official_video = True
    # the fake video VAE grows a whole-clip decode, as comfy.sd.VAE has
    fake_context.video_vae.decode = lambda video: torch.zeros(
        1, FAKE_GEOMETRY["frames"], FAKE_GEOMETRY["height"], FAKE_GEOMETRY["width"], 3
    )
    payload = probe.execute_run(fake_context, 1, "sample")

    compare = payload["official_compare"]
    assert compare["allowed"] is True
    assert compare["seconds"] >= 0
    assert compare["shape"] == [22, 64, 64, 3]
    assert compare["metrics"]["comparable"] is True
    # the reference is all zeros and the collector's ramp is not, so this is a
    # reported difference, not a failure
    assert compare["metrics"]["bitwise"] is False
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert failed == []
    # the diagnostic ran outside the instrumented window, so the "the node
    # never calls it" gate is untouched
    assert "run: the node called no whole-clip decode helper" in [
        c.name for c in payload["_checks"] if c.ok
    ]


def test_the_official_comparison_is_off_unless_asked_for(probe, fake_context):
    assert fake_context.args.compare_official_video is False
    payload = probe.execute_run(fake_context, 1, "sample")
    assert "official_compare" not in payload


def test_a_cancelled_run_never_compares_against_the_official_decode(probe, fake_context):
    fake_context.args.compare_official_video = True
    payload = probe.execute_run(fake_context, 1, "cancel")
    assert "official_compare" not in payload


def test_the_dit_phase_gate_follows_the_chunk_count(probe, stand_in_vae_class):
    """3 chunks means 3 DiT-phase loads, and the gate derives that itself.

    39 frames is the geometry the vr run used, where the old
    ``load_models_calls == 1`` gate failed against a correct phase swap.
    """
    pytest.importorskip("av")
    geometry = probe.latent_geometry(39, 64, 64)
    assert probe.layout_num_chunks(geometry) == 3
    with fake_context_at(probe, geometry) as context:
        payload = probe.execute_run(context, 1, "sample")

    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures
    assert payload["num_chunks"] == 3
    assert payload["dit_phase_loads"] == 3
    assert payload["phase_swap"]["observed_dit_loads"] == 3
    assert payload["phase_swap"]["delivered_chunks"] == 3
    assert payload["phase_swap"]["phase_swap_dit_loads"] == 2  # the reloads only
    assert payload["phase_swap"]["phase_swap_last_phase"] == "vae"
    gate = [c for c in payload["_checks"]
            if c.name == "run: the DiT phase was entered once per chunk"][0]
    assert gate.ok and "one after each non-last" in gate.detail


def test_the_dit_phase_gate_fails_when_a_reload_goes_missing(probe, fake_context):
    """A run that loads once for a 2-chunk clip is the old, wrong behaviour."""
    original = FakeSampler.sample

    def sample_without_the_swap(self, **kwargs):
        # drop the per-chunk reload by pretending every chunk is the last
        loads = []
        real_make = nodes_mod.make_load_models

        def once(*args, **kw):
            closure = real_make(*args, **kw)

            def guarded(models, memory_required=0, force_full_load=False):
                if loads:
                    return None
                loads.append(1)
                return closure(models, memory_required, force_full_load)

            return guarded

        nodes_mod.make_load_models = once
        try:
            return original(self, **kwargs)
        finally:
            nodes_mod.make_load_models = real_make

    fake_context.sampler.__class__.sample = sample_without_the_swap
    try:
        payload = probe.execute_run(fake_context, 1, "sample")
    finally:
        fake_context.sampler.__class__.sample = original

    failed = [c for c in payload["_checks"] if c.gate and not c.ok]
    assert any("the DiT phase was entered once per chunk" in c.name for c in failed)
    assert payload["dit_phase_loads"] == 1
    assert payload["num_chunks"] == 2


def test_a_disagreeing_phase_record_is_a_failure(probe, fake_context):
    """Two accounts of the same run must not contradict each other."""
    fake_context.sampler.phase_record = {
        "phase_swap_chunks": 7,
        "phase_swap_vae_loads": 7,
        "phase_swap_audio_vae_loads": 7,
        "phase_swap_dit_loads": 6,
        "phase_swap_last_phase": "dit",
    }
    payload = probe.execute_run(fake_context, 1, "sample")
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert any("the node's record agrees with the layout" in name for name in failed)
    assert any("ended in the VAE phase" in name for name in failed)
    # the count this probe made itself is still right
    assert payload["dit_phase_loads"] == 2


def test_a_cancelled_run_derives_its_phase_count(probe, fake_context):
    """steps=2, cancel after 3 forwards: chunk 0 delivered, chunk 1 interrupted."""
    payload = probe.execute_run(fake_context, 1, "cancel")
    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures

    delivered = payload["phase_swap"]["delivered_chunks"]
    assert delivered == 1  # only chunk 0 reached the pipeline
    assert payload["num_chunks"] == 2
    assert payload["dit_phase_loads"] == probe.dit_loads_after_cancel(delivered, 2) == 2
    # a cancelled run has no phase record to cross-check against; the count is
    # derived from what finished instead
    assert "phase_swap_chunks" not in payload["phase_swap"]
    gate = [c for c in payload["_checks"]
            if c.name == "cancel: the DiT phase count matches the chunks that finished"][0]
    assert gate.ok


def test_a_cancelled_run_that_does_carry_a_record_is_checked_loosely(probe, fake_context):
    """If a future runtime writes the record on the cancel path too.

    The strict layout comparison cannot apply -- a cancelled run delivered
    fewer chunks than the layout has -- so only the invariant that survives
    any cancellation point is checked.
    """
    fake_context.sampler.phase_record = {
        "phase_swap_chunks": 1,
        "phase_swap_vae_loads": 1,
        "phase_swap_audio_vae_loads": 1,
        "phase_swap_dit_loads": 1,
        "phase_swap_last_phase": "dit",
    }
    fake_context.sampler.phase_record_early = True
    payload = probe.execute_run(fake_context, 1, "cancel")
    names = [c.name for c in payload["_checks"]]
    assert "cancel: the phase record accounts for every DiT call counted" in names
    # the strict, completed-run comparisons must not have been applied
    assert "phase swap: the run ended in the VAE phase" not in names
    assert "phase swap: the node's record agrees with the layout and the count" not in names


def test_a_run_cancelled_before_any_chunk_still_loaded_the_dit_once(probe, stand_in_vae_class):
    pytest.importorskip("av")
    with fake_context_at(probe, FAKE_GEOMETRY) as context:
        context.args.cancel_after_forward = 1  # inside chunk 0's first forward
        payload = probe.execute_run(context, 1, "cancel")
    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures
    assert payload["phase_swap"]["delivered_chunks"] == 0
    assert payload["dit_phase_loads"] == 1
    assert probe.dit_loads_after_cancel(0, payload["num_chunks"]) == 1


def test_execute_run_reports_a_failing_run_without_raising(probe, fake_context):
    fake_context.sampler.fail_with = RuntimeError("boom")
    payload = probe.execute_run(fake_context, 1, "sample")
    assert payload["ok"] is False
    assert "RuntimeError: boom" in payload["exception"]
    assert "Traceback" in payload["traceback"]
    assert payload["preview"]["end_reason"] == "error"
    assert payload["sessions"]["active"] == 0


def test_execute_run_cancels_a_real_run_and_cleans_up(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "cancel")
    failures = [c for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]
    assert payload["cancelled"] is True
    assert payload["forward_chunk_calls"] == 3
    assert "SamplingCancelled" in payload["exception"]
    assert "artifacts" not in payload or payload["artifacts"] == {}
    assert payload["preview"]["end_reason"] == "cancelled"
    assert payload["media"]["skipped"]
    assert payload["sessions"]["active"] == 0


# -- the later cancellation point: --cancel-after-chunk --------------------


@pytest.fixture
def chunk_cancel_context(probe, stand_in_vae_class):
    """A fake context whose args ask for a cancel after one delivered chunk."""
    context, manager, previous = _make_fake_context(
        probe, cancel_args=["--cancel-after-chunk", "1"]
    )
    try:
        yield context
    finally:
        manager.set_sender(previous)
        for session in list(manager.active_sessions):
            manager.cleanup(session.session_id)


def test_a_chunk_cancel_stops_after_the_chunk_reached_the_decoders(probe, chunk_cancel_context):
    payload = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    failures = [c for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]

    assert payload["cancelled"] is True
    assert "SamplingCancelled" in payload["exception"]
    assert payload["chunks_delivered"] == 1
    cancel = payload["chunk_cancel"]
    assert cancel["requested_after_chunks"] == 1
    assert cancel["raised"] is True

    # the chunk really reached both collectors: it is in their buffers
    before = cancel["state_before"]
    assert before["chunks"] == 1
    assert before["holds_decoder_buffers"] is True
    assert before["decoders"]["video"]["latents_seen"] > 0

    # ... and afterwards the pipeline is holding nothing at all
    after = cancel["state_after"]
    assert after["pending_frames"] == 0
    assert after["samples_available"] == 0
    assert after["image_bytes"] == 0 and after["audio_bytes"] == 0
    assert after["holds_decoder_buffers"] is False

    # no partial product, and the preview ended as a cancellation
    assert "artifacts" not in payload or payload["artifacts"] == {}
    assert payload["preview"]["end_reason"] == "cancelled"
    assert payload["media"]["skipped"]
    assert payload["sessions"]["active"] == 0


def test_a_chunk_cancel_after_two_chunks_has_really_used_the_vae(probe, stand_in_vae_class):
    """The video lane buffers its lookahead, so the *second* chunk is the one
    that has provably been through ``VAE.decode``. The probe derives that
    expectation from the decoder's own state rather than assuming either way."""
    context, manager, previous = _make_fake_context(
        probe, cancel_args=["--cancel-after-chunk", "2"]
    )
    try:
        payload = probe.execute_run(context, 1, "cancel_chunk")
    finally:
        manager.set_sender(previous)
        for session in list(manager.active_sessions):
            manager.cleanup(session.session_id)

    failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
    assert not failures, failures
    before = payload["chunk_cancel"]["state_before"]
    assert payload["chunks_delivered"] == 2
    assert before["frames_decoded"] > 0
    video = before["decoders"]["video"]
    assert video["vae_decode_calls"] >= 1
    assert video["consumed_latents"] > 0
    # and the frames it decoded are gone again after the cancel
    after = payload["chunk_cancel"]["state_after"]
    assert after["image_bytes"] == 0 and after["pending_frames"] == 0


def test_the_decode_gate_is_vacuous_only_while_nothing_was_due(probe, chunk_cancel_context):
    """At ``--cancel-after-chunk 1`` no decode is due, and the report says so
    instead of quietly passing a gate that checked nothing."""
    payload = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    detail = [
        c.detail
        for c in payload["_checks"]
        if c.name == "cancel: every latent the video decoder consumed had really been decoded"
    ][0]
    assert "nothing was due yet at --cancel-after-chunk 1" in detail
    assert payload["chunk_cancel"]["state_before"]["decoders"]["video"]["consumed_latents"] == 0


def test_a_chunk_cancel_aborts_both_decoders(probe, chunk_cancel_context):
    """Feature-probed: where ``abort()`` exists, the cancel must have used it."""
    payload = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    cancel = payload["chunk_cancel"]
    names = [c.name for c in payload["_checks"]]
    for lane in ("video", "audio"):
        available = cancel["decoder_abort_available"][lane]
        if available:
            assert cancel["decoder_aborts"][lane] >= 1
            assert "cancel: the {} decoder was aborted".format(lane) in names
        else:
            assert "cancel: the {} decoder has no abort()".format(lane) in names
    # this runtime does have both, so the gates above are real ones
    assert cancel["decoder_abort_available"] == {"video": True, "audio": True}


def test_the_decoders_were_holding_something_when_the_cancel_landed(probe, chunk_cancel_context):
    payload = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    before = payload["chunk_cancel"]["state_before"]
    assert before["holds_decoder_buffers"] is True, before["decoders"]
    held = {lane: entry["buffers_held"] for lane, entry in before["decoders"].items()}
    assert any(held.values()), held
    # the same probe finds nothing afterwards
    after = payload["chunk_cancel"]["state_after"]
    assert all(not entry["buffers_held"] for entry in after["decoders"].values())


def test_a_chunk_cancel_expects_one_dit_load_fewer(probe, chunk_cancel_context):
    """It lands before the coordinator reloads the DiT for the next forward."""
    payload = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    delivered = payload["phase_swap"]["delivered_chunks"]
    assert payload["dit_phase_loads"] == probe.dit_loads_after_chunk_cancel(
        delivered, payload["num_chunks"]
    )
    # the forward-cancel arithmetic would have expected one more, and saying so
    # is what makes the two points distinguishable rather than interchangeable
    assert payload["dit_phase_loads"] != probe.dit_loads_after_cancel(
        delivered, payload["num_chunks"]
    )
    failed = [c.name for c in payload["_checks"] if c.gate and not c.ok]
    assert "cancel: the DiT phase count matches the chunks that finished" not in failed


def test_a_chunk_cancel_is_followed_by_a_working_run(probe, chunk_cancel_context):
    cancelled = probe.execute_run(chunk_cancel_context, 1, "cancel_chunk")
    assert cancelled["ok"] is True
    after = probe.execute_run(chunk_cancel_context, 2, "sample")
    failures = [c.line() for c in after["_checks"] if c.gate and not c.ok]
    assert not failures, failures
    assert after["preview"]["end_reason"] == "complete"
    assert after["media"]["video_frames"] == 22
    # the second run is not armed: the instrumentation is per-run
    assert "chunk_cancel" not in after


def test_a_normal_run_delivers_every_chunk_and_arms_nothing(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "sample")
    assert payload["chunks_delivered"] == payload["num_chunks"]
    assert "chunk_cancel" not in payload


# -- --kv-cache-storage reaches the node -----------------------------------


def test_the_storage_mode_reaches_the_sampler_and_the_report(probe, fake_context):
    payload = probe.execute_run(fake_context, 1, "sample")
    assert payload["kv_cache_storage"] == "cpu_pinned"
    assert fake_context.sampler.kv_cache_storages == ["cpu_pinned"]


@pytest.mark.parametrize("mode", ["gpu", "cpu", "cpu_pinned"])
def test_every_storage_mode_is_passed_through_unchanged(probe, stand_in_vae_class, mode):
    context, manager, previous = _make_fake_context(
        probe, extra_args=["--kv-cache-storage", mode]
    )
    try:
        payload = probe.execute_run(context, 1, "sample")
    finally:
        manager.set_sender(previous)
        for session in list(manager.active_sessions):
            manager.cleanup(session.session_id)
    assert context.sampler.kv_cache_storages == [mode]
    assert payload["kv_cache_storage"] == mode
    # the run still passes every gate whichever mode it used
    assert [c.line() for c in payload["_checks"] if c.gate and not c.ok] == []


def test_a_cancelled_run_is_followed_by_a_working_one(probe, fake_context):
    cancelled = probe.execute_run(fake_context, 1, "cancel")
    assert cancelled["ok"] is True  # the cancellation behaved as specified
    after = probe.execute_run(fake_context, 2, "sample")
    failures = [c for c in after["_checks"] if c.gate and not c.ok]
    assert not failures, [c.line() for c in failures]
    assert after["preview"]["end_reason"] == "complete"
    assert after["media"]["video_frames"] == 22


def test_cancel_then_three_normal_runs_all_pass(probe, fake_context):
    """The plan ``--repeat 3 --cancel-after-forward N`` actually executes."""
    plan = probe.run_plan(repeat=3, cancel_after_forward=3)
    assert plan == ["cancel", "sample", "sample", "sample"]

    completed = []
    for index, kind in enumerate(plan, start=1):
        payload = probe.execute_run(fake_context, index, kind)
        failures = [c.line() for c in payload["_checks"] if c.gate and not c.ok]
        assert not failures, (kind, failures)
        completed.append({"kind": kind, "payload": payload})

    assert [entry["payload"]["preview"]["end_reason"] for entry in completed] == [
        "cancelled", "complete", "complete", "complete",
    ]
    assert fake_context.manager.active_sessions == []

    sampled = [
        {"artifacts": entry["payload"]["artifacts"], "memory": entry["payload"]["memory"]}
        for entry in completed
        if entry["kind"] == "sample"
    ]
    assert len(sampled) == 3
    summary, checks = probe.compare_run_series(sampled)
    assert [c.line() for c in checks if c.gate and not c.ok] == []
    assert summary["memory"]["mode"] == "plateau"
    assert [c["run"] for c in summary["comparisons"]] == [2, 3]


def test_execute_run_records_what_comfy_still_holds(probe, fake_context):
    resident = FakePatcher(load_device="cpu")
    resident._loaded = 2048
    fake_context.env.comfy.model_management.loaded_models = lambda: [resident]
    payload = probe.execute_run(fake_context, 1, "sample")
    assert payload["loaded_models"] == [
        {
            "patcher": "FakePatcher",
            "model": "Linear",
            "is_clip": True,
            "loaded_size": 2048,
            "model_size": 1024,
            "current_device": "cpu",
        }
    ]
    note = [c for c in payload["_checks"] if "models resident" in c.name]
    assert note and note[0].gate is False


def test_loaded_model_snapshot_survives_an_older_comfyui(probe):
    assert probe.loaded_model_snapshot(SimpleNamespace()) == []
    assert probe.loaded_model_snapshot(None) == []

    def boom():
        raise RuntimeError("no")

    assert probe.loaded_model_snapshot(SimpleNamespace(loaded_models=boom)) == []


def test_execute_run_uses_the_injected_latent_factory(probe, fake_context):
    """The text lane hands over copies of the official node's own latent."""
    built = []

    def factory():
        latent = probe.build_empty_latent(FAKE_GEOMETRY, FakeNestedTensor)
        built.append(latent)
        return latent

    fake_context.latent_factory = factory
    payload = probe.execute_run(fake_context, 1, "sample")
    assert payload["ok"] is True
    assert len(built) == 1


def test_two_runs_are_bitwise_comparable_and_leak_no_session(probe, fake_context):
    first = probe.execute_run(fake_context, 1, "sample")
    second = probe.execute_run(fake_context, 2, "sample")
    assert fake_context.manager.active_sessions == []
    assert first["preview"]["session_ids"] != second["preview"]["session_ids"]
    summary, checks = probe.compare_runs(first, second)
    assert [c.name for c in checks if c.gate and not c.ok] == []
    assert summary["latent"]["video"]["bitwise"] is True


# ==========================================================================
# run_probe end to end: the setup path itself, not a fake standing in for it
# ==========================================================================
#
# These exist because a real vr run died with ``NameError: setup is not
# defined`` in ``_run_probe`` -- before a single weight was read, and with
# every fake-driven test in this file passing, because they all entered below
# the setup. So these drive the *real* ``run_probe``: the real geometry, the
# real path checks, the real ``folder_paths`` registration, the real
# ``VAELoader`` call sequence, the node's real VAE feature probes, the real
# text lane, and the real preview wiring. Only the two things that would need
# 71 GB of weights -- the upstream loader nodes -- are fakes, and they are
# fakes of the *loader*, not of the code under test.


class FakeCausalDiT:
    pass


class RavenCausalMiniMaxH3Model(FakeCausalDiT):  # noqa: N801 - the name is the check
    """``_run_probe`` identifies the chunk-causal DiT by class name.

    It carries ``forward_chunk`` because the real one does and the probe wraps
    it to count forwards; nothing in these tests gets far enough to call it
    (the real sampler refuses this stand-in patcher first).
    """

    def __init__(self):
        self.dtype = torch.bfloat16
        self.hidden_size = 64
        self.blocks = []

    def forward_chunk(self, **kwargs):  # pragma: no cover - never reached here
        raise AssertionError("the setup regression tests never sample")


class FakeRAVENModelLoader:
    """``nodes.RAVENModelLoader`` without the 71 GB."""

    calls = []
    order = []
    raises = None

    def load_model(self, unet_name, lora_name, weight_dtype="default"):
        FakeRAVENModelLoader.calls.append((unet_name, lora_name, weight_dtype))
        FakeRAVENModelLoader.order.append("load_model")
        if FakeRAVENModelLoader.raises is not None:
            raise FakeRAVENModelLoader.raises
        diffusion_model = RavenCausalMiniMaxH3Model()
        inner = SimpleNamespace(
            diffusion_model=diffusion_model,
            raven_lora_manifest=SimpleNamespace(name="raven-4nfe-preview"),
            raven_lora_attachment=SimpleNamespace(entries=[object()] * 266),
        )
        patcher = SimpleNamespace(
            model=inner,
            model_size=lambda: 71 * (1024 ** 3),
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        return (patcher,)


class FakeVAELoaderNode:
    calls = []
    order = []

    def load_vae(self, vae_name):
        FakeVAELoaderNode.calls.append(vae_name)
        FakeVAELoaderNode.order.append("load_vae")
        if "audio" in vae_name:
            return (socketed_audio_vae(),)
        return (socketed_video_vae(),)


@pytest.fixture
def probe_env(probe, repo_report_dir, monkeypatch, stand_in_vae_class):
    """Real ``run_probe``, fake upstream loaders, files that really exist."""
    root = os.path.join(repo_report_dir, "models")
    paths = {
        name: _touch(pathlib.Path(root) / name)
        for name in (
            "minimax_h3_dit.safetensors",
            "raven_lora.safetensors",
            "minimax_h3_video_vae.safetensors",
            "minimax_h3_audio_vae.safetensors",
            "qwen3vl_text_encoder.safetensors",
        )
    }
    FakeRAVENModelLoader.calls = []
    FakeRAVENModelLoader.order = []
    FakeRAVENModelLoader.raises = None
    FakeVAELoaderNode.calls = []
    FakeVAELoaderNode.order = FakeRAVENModelLoader.order  # one shared timeline
    FakeCLIPLoader.calls = []

    folder_paths = FakeFolderPaths()
    order = FakeRAVENModelLoader.order

    def clip_factory():
        order.append("load_clip")
        return FakeCLIP(FakePatcher(load_device="cpu"))

    FakeCLIPLoader.clip_factory = clip_factory
    FakeImageToVideo.outputs = official_t2va_outputs(FAKE_GEOMETRY)

    def load_models_gpu(models, memory_required=0, **kwargs):
        for model in models:
            model._loaded = model.model_size()

    env = probe.ComfyEnv(
        root="/fake/ComfyUI",
        comfy=SimpleNamespace(
            model_management=SimpleNamespace(
                get_torch_device=lambda: torch.device("cpu"),
                intermediate_device=lambda: torch.device("cpu"),
                load_models_gpu=load_models_gpu,
                unload_model_and_clones=lambda patcher, **kw: setattr(
                    patcher, "_loaded", 0
                ),
            ),
            nested_tensor=SimpleNamespace(NestedTensor=FakeNestedTensor),
            sd=SimpleNamespace(VAE=FakeVAEWrapper),
        ),
        folder_paths=folder_paths,
        upstream_nodes=SimpleNamespace(
            VAELoader=FakeVAELoaderNode, CLIPLoader=FakeCLIPLoader
        ),
        minimax_nodes=SimpleNamespace(MiniMaxH3ImageToVideo=FakeImageToVideo),
        commit="deadbeef",
        version="0.33.0",
    )
    monkeypatch.setattr(probe, "import_comfy", lambda root: env)
    monkeypatch.setattr(nodes_mod, "RAVENModelLoader", FakeRAVENModelLoader)
    return SimpleNamespace(
        env=env, paths=paths, folder_paths=folder_paths, order=order, root=root
    )


def _probe_args(probe, probe_env, *extra):
    return probe.build_parser().parse_args(
        [
            "--base", str(probe_env.paths["minimax_h3_dit.safetensors"]),
            "--lora", str(probe_env.paths["raven_lora.safetensors"]),
            "--video-vae", str(probe_env.paths["minimax_h3_video_vae.safetensors"]),
            "--audio-vae", str(probe_env.paths["minimax_h3_audio_vae.safetensors"]),
            "--frames", "22", "--width", "64", "--height", "64",
            "--text-len", "8", "--steps", "1", "--device", "cpu",
        ]
        + list(extra)
    )


def _setup_of(report):
    return report.setup


def test_run_probe_fills_setup_for_the_synthetic_lane(probe, probe_env):
    """Regression: ``NameError: setup is not defined`` before any weight loads."""
    report = probe.Report()
    args = _probe_args(probe, probe_env)

    probe.run_probe(args, report)  # must not raise

    setup = _setup_of(report)
    # the VAE half -- this is where the vr run died
    assert setup["video_vae_name"] == "minimax_h3_video_vae.safetensors"
    assert setup["audio_vae_name"] == "minimax_h3_audio_vae.safetensors"
    assert setup["video_vae_class"] == "MiniMaxH3VideoVAE"
    assert setup["audio_vae_class"] == "MiniMaxH3AudioVAE"
    assert setup["video_vae_seconds"] >= 0 and setup["audio_vae_seconds"] >= 0
    # ... and the model half, which only exists if the loader block ran at all
    assert setup["model_build_seconds"] >= 0
    assert setup["diffusion_model_class"] == "RavenCausalMiniMaxH3Model"
    assert setup["model_size_bytes"] == 71 * (1024 ** 3)
    assert setup["lora_modules"] == 266
    assert setup["lora_manifest"] == "raven-4nfe-preview"
    assert setup["patcher_class"] and setup["model_class"]
    assert setup["weight_dtype"] == "default"

    # the loader was driven with the absolute paths, in the documented order:
    # both VAEs first, then the DiT
    assert FakeRAVENModelLoader.calls == [
        (str(probe_env.paths["minimax_h3_dit.safetensors"]),
         str(probe_env.paths["raven_lora.safetensors"]),
         "default")
    ]
    assert probe_env.order == ["load_vae", "load_vae", "load_model"]
    assert probe_env.folder_paths.registered == [
        ("vae", probe_env.root, True), ("vae", probe_env.root, True)
    ]

    # the setup checks themselves passed; nothing here is a NameError
    names = {c["name"]: c for c in report.checks}
    assert names["vae: the node accepted both sockets"]["ok"] is True
    assert names["loader: the node returned a chunk-causal DiT"]["ok"] is True
    assert names["loader: the RAVEN residual is attached"]["ok"] is True
    assert "NameError" not in json.dumps(report.to_dict(), default=str)
    assert report.inputs["lane"] == "synthetic"
    assert report.inputs["conditioning"]["shape"] == [1, 8, 5120]

    # The probe got all the way into the real node and stopped where a fake
    # patcher must stop it -- not earlier, and not with a NameError. That is
    # what makes the assertions above evidence that the setup path ran.
    assert len(report.runs) == 1
    run = report.runs[0]
    assert "ContractError" in run["exception"]
    assert report.ok is False


def test_run_probe_fills_setup_for_the_text_lane(probe, probe_env):
    report = probe.Report()
    args = _probe_args(
        probe, probe_env,
        "--text-encoder", str(probe_env.paths["qwen3vl_text_encoder.safetensors"]),
        "--prompt", "a cat playing a trumpet",
    )

    probe.run_probe(args, report)  # must not raise

    setup = _setup_of(report)
    assert setup["video_vae_class"] == "MiniMaxH3VideoVAE"
    assert setup["audio_vae_class"] == "MiniMaxH3AudioVAE"
    assert setup["diffusion_model_class"] == "RavenCausalMiniMaxH3Model"
    assert setup["model_size_bytes"] == 71 * (1024 ** 3)

    # the text lane really ran, between the VAEs and the DiT
    assert probe_env.order == ["load_vae", "load_vae", "load_clip", "load_model"]
    text_lane = report.inputs["text_lane"]
    assert text_lane["clip_type"] == "minimax"
    assert text_lane["name"] == "qwen3vl_text_encoder.safetensors"
    assert text_lane["text_len"] == 11
    assert text_lane["released_after_encode"] is True
    assert report.inputs["lane"] == "text-encoder"
    assert report.inputs["text_len"] == 11
    assert "conditioning" not in report.inputs  # the synthetic lane never ran
    assert FakeImageToVideo.calls[0]["prompt"] == "a cat playing a trumpet"
    assert "NameError" not in json.dumps(report.to_dict(), default=str)

    # same as the synthetic lane: it stopped inside the real node, not before it
    assert len(report.runs) == 1
    assert "ContractError" in report.runs[0]["exception"]


def test_run_probe_keeps_the_vae_setup_when_the_loader_dies(probe, probe_env):
    """``setup`` is attached to the report before anything fills it."""
    FakeRAVENModelLoader.raises = RuntimeError("the checkpoint is pruned")
    report = probe.Report()
    with pytest.raises(RuntimeError, match="pruned"):
        probe.run_probe(_probe_args(probe, probe_env), report)

    setup = _setup_of(report)
    assert setup["video_vae_name"] and setup["audio_vae_name"]
    assert "model_build_seconds" not in setup  # the loader never got that far
    # and the checks made before the crash survived the ``finally`` flush
    assert any(c["name"].startswith("vae:") for c in report.checks)


def test_run_probe_stops_at_a_swapped_vae_socket_with_setup_intact(probe, probe_env):
    def swapped(self, vae_name):
        FakeVAELoaderNode.order.append("load_vae")
        return (socketed_audio_vae(),) if "video" in vae_name else (socketed_video_vae(),)

    # Put the class's own method back rather than ``del``-ing the attribute:
    # ``load_vae`` is defined in the class body, so deleting it removes the
    # real one and every later test in this file gets an AttributeError.
    original_load_vae = FakeVAELoaderNode.load_vae
    FakeVAELoaderNode.load_vae = swapped
    try:
        report = probe.Report()
        probe.run_probe(_probe_args(probe, probe_env), report)
    finally:
        FakeVAELoaderNode.load_vae = original_load_vae

    setup = _setup_of(report)
    assert setup["video_vae_class"] == "MiniMaxH3AudioVAE"  # what it actually got
    assert "model_build_seconds" not in setup  # refused before loading 71 GB
    failed = [c for c in report.checks if c["gate"] and not c["ok"]]
    assert any("the node accepted both sockets" in c["name"] for c in failed)
    assert any("sockets are swapped" in c["detail"] for c in failed)
    assert report.ok is False


def test_run_probe_reports_a_missing_file_before_touching_a_loader(probe, probe_env):
    args = _probe_args(probe, probe_env)
    args.lora = os.path.join(probe_env.root, "absent.safetensors")
    report = probe.Report()
    with pytest.raises(probe.ProbeError, match="--lora does not exist"):
        probe.run_probe(args, report)
    assert FakeVAELoaderNode.calls == []
    assert FakeRAVENModelLoader.calls == []
    assert report.setup == {}


def test_run_probe_stacks_nothing_when_the_flag_is_absent(probe, probe_env):
    """The lane is off by default, and leaves no trace claiming otherwise."""
    report = probe.Report()
    probe.run_probe(_probe_args(probe, probe_env), report)
    assert "stacked_lora" not in _setup_of(report)
    assert not [c for c in report.checks if c["name"].startswith("stacked lora:")]
    assert not any("stacked lora lane" in note for note in report.notes)
    assert "stacked lora:" not in report.render()


def test_run_probe_records_the_kv_storage_mode_it_was_given(probe, probe_env):
    report = probe.Report()
    probe.run_probe(_probe_args(probe, probe_env, "--kv-cache-storage", "gpu"), report)
    assert report.environment["kv_cache_storage"] == "gpu"
    names = {c["name"]: c for c in report.checks}
    assert names["kv cache: the probe offers exactly the node's storage modes"]["ok"] is True
    assert "kv cache gpu" in report.render()


def test_run_probe_refuses_a_stacked_lora_without_the_official_node(probe, probe_env):
    """The fake upstream here has no ``LoraLoaderModelOnly``; the probe stops."""
    report = probe.Report()
    args = _probe_args(probe, probe_env, "--stacked-lora-name", "whatever.safetensors")
    with pytest.raises(probe.ProbeError, match="LoraLoaderModelOnly"):
        probe.run_probe(args, report)
    # the DiT was built first, so the refusal is about the node and not the order
    assert _setup_of(report)["model_build_seconds"] >= 0
    assert "stacked_lora" not in _setup_of(report)


def test_run_probe_drives_the_real_official_node_and_reports_it(
    probe, probe_env, upstream_lora_node, monkeypatch, tmp_path
):
    """The whole lane through ``run_probe``: real node, real file, real report.

    The only fakes left are the two loaders that would otherwise need 71 GB.
    The model the official node patches is a real ``ModelPatcher`` around a
    miniature module carrying real base keys and a real RAVEN residual.
    """
    folder_paths = upstream_lora_node.folder_paths
    directory = tmp_path / "loras"
    directory.mkdir(parents=True, exist_ok=True)
    path = _write_lora(upstream_lora_node, directory, _rank1_lora())
    saved = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.add_model_folder_path("loras", str(directory), is_default=True)

    inner = _tiny_raven_model()
    assert type(inner.diffusion_model).__name__ == "RavenCausalMiniMaxH3Model"
    patcher = upstream_lora_node.comfy.model_patcher.ModelPatcher(
        inner, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )

    class RealPatcherLoader:
        def load_model(self, unet_name, lora_name, weight_dtype="default"):
            FakeRAVENModelLoader.order.append("load_model")
            return (patcher,)

    monkeypatch.setattr(nodes_mod, "RAVENModelLoader", RealPatcherLoader)
    # the probe resolves the name through env.folder_paths; the node resolves it
    # through the real module. In production they are the same object -- here
    # the fixture's fake stands in for the VAE combos, so both are told.
    probe_env.folder_paths.add_model_folder_path("loras", str(directory))
    probe_env.env.comfy.sd.load_lora_for_models = (
        upstream_lora_node.comfy.sd.load_lora_for_models
    )
    probe_env.env.comfy.model_patcher = upstream_lora_node.comfy.model_patcher
    probe_env.env.upstream_nodes.LoraLoaderModelOnly = upstream_lora_node.node_cls

    report = probe.Report()
    try:
        probe.run_probe(
            _probe_args(
                probe, probe_env,
                "--stacked-lora-name", path.name,
                "--stacked-lora-strength", "0.6",
            ),
            report,
        )
    finally:
        if saved is None:
            folder_paths.folder_names_and_paths.pop("loras", None)
        else:
            folder_paths.folder_names_and_paths["loras"] = saved

    stacked = _setup_of(report)["stacked_lora"]
    assert stacked["node_class"] == "LoraLoaderModelOnly"
    assert stacked["node_calls"] == 1
    assert stacked["strength"] == 0.6
    assert stacked["target_keys"] == [TINY_BASE_KEY]
    assert stacked["nonzero_base_key_hits"] == [TINY_BASE_KEY]
    assert stacked["attachment_modules"] == 266
    assert stacked["patcher_is_stock"] is True

    # every stacked gate passed, and the note says what was done
    stacked_checks = [c for c in report.checks if c["name"].startswith("stacked lora:")]
    assert stacked_checks
    assert [c for c in stacked_checks if c["gate"] and not c["ok"]] == []
    assert any("stacked lora lane" in note for note in report.notes)

    # the report survives the round trip a GPU box would post back
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["setup"]["stacked_lora"]["path"] == stacked["path"]
    assert payload["setup"]["stacked_lora"]["bytes"] == stacked["bytes"]
    assert "stacked lora: LoraLoaderModelOnly.load_lora_model_only" in report.render()

    # ... and the sampler was handed the node's clone, not the original MODEL
    assert report.runs, "the probe went on to sample with the stacked model"

    # the residual is re-checked after the run(s), against the object recorded
    # before them rather than against itself
    assert stacked["attachment_same_object_after_runs"] is True
    assert stacked["attachment_modules_after_runs"] == 266
    assert stacked["patch_keys_after_runs"] == stacked["patch_keys_after"]
    survived = [
        c for c in report.checks
        if c["name"] == "stacked lora: the patches and the RAVEN residual survived the run(s)"
    ]
    assert survived and survived[0]["ok"] is True


def test_the_probe_module_has_no_use_before_assignment(probe):
    """A static audit of every function, in the shape of the vr failure.

    ``NameError: setup is not defined`` was a *textual* ordering mistake: a
    block that filled a dict was moved above the block that created it. Nothing
    in this file's fake-driven tests could see it, so the guard is static.
    """
    import ast

    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def comprehension_targets(fn):
        names = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for generator in node.generators:
                    for sub in ast.walk(generator.target):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
        return names

    def audit(fn):
        bound, loaded = {}, {}
        arguments = fn.args
        skip = comprehension_targets(fn)
        for arg in (list(arguments.posonlyargs) + list(arguments.args)
                    + list(arguments.kwonlyargs)):
            skip.add(arg.arg)
        if arguments.vararg:
            skip.add(arguments.vararg.arg)
        if arguments.kwarg:
            skip.add(arguments.kwarg.arg)
        for node in ast.walk(fn):
            if isinstance(node, ast.Global):
                skip.update(node.names)
            if isinstance(node, ast.Name):
                target = bound if isinstance(node.ctx, ast.Store) else loaded
                target.setdefault(node.id, node.lineno)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.setdefault((alias.asname or alias.name).split(".")[0], node.lineno)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.setdefault(node.name, node.lineno)
        return [
            "{}: {!r} is used on line {} but first assigned on line {}".format(
                fn.name, name, line, bound[name]
            )
            for name, line in loaded.items()
            if name not in skip and name in bound and bound[name] > line
        ]

    problems = [
        problem
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for problem in audit(node)
    ]
    assert problems == [], problems




def test_report_is_json_safe_and_renders(probe):
    report = probe.Report()
    report.environment = {"comfy_root": "/opt/ComfyUI", "device": "cuda:0"}
    report.inputs = {"geometry": probe.latent_geometry(39, 512, 288), "text_len": 128}
    report.args = {"steps": 4, "seed": 0}
    report.setup = {"model_build_seconds": 1.5, "video_vae_seconds": 0.2, "audio_vae_seconds": 0.1}
    report.runs = [
        {
            "index": 1, "kind": "sample",
            "timings": {"sample_seconds": 12.0},
            "preview": {"messages": 40, "segments": 39, "segment_bytes": 1000, "end_reason": "complete"},
            "media": {"video_frames": 39, "audio_samples": 52000},
            "memory": {"cuda_peak_allocated": 1 << 30, "rss_after": 1 << 30, "rss_peak": 1 << 30},
        }
    ]
    report.checks = [
        {"name": "a", "ok": True, "detail": "", "gate": True},
        {"name": "b", "ok": False, "detail": "why", "gate": True},
    ]
    text = report.render()
    assert "RESULT: FAILED" in text
    assert "[FAIL] b - why" in text
    assert "lane: synthetic" in text
    assert report.ok is False
    json.loads(json.dumps(report.to_dict(), default=str))
    assert report.to_dict()["failures"][0]["name"] == "b"


def test_report_renders_the_text_lane_and_the_plateau_gate(probe):
    report = probe.Report()
    report.inputs = {
        "lane": "text-encoder",
        "geometry": probe.latent_geometry(22, 64, 64),
        "text_len": 11,
        "text_lane": {
            "name": "qwen3vl.safetensors",
            "load_seconds": 42.5,
            "encode_seconds": 3.25,
            "token_tags": {"histogram": {"video(0)": 1, "text(1)": 9, "audio(2)": 1}},
        },
    }
    runs = [_series_entry(i, artifacts=_identical_artifacts()()) for i in (1, 2, 3)]
    report.determinism, _checks = probe.compare_run_series(runs)
    report.runs = [{
        "index": 1, "kind": "sample", "timings": {"sample_seconds": 1.0},
        "preview": {}, "media": {}, "memory": {},
        "phase_swap": {
            "observed_dit_loads": 3, "layout_chunks": 3, "delivered_chunks": 3,
            "phase_swap_vae_loads": 3, "phase_swap_last_phase": "vae",
        },
    }]
    text = report.render()
    assert "phase swap: 3 DiT phase load(s) for 3 chunk(s) (3 delivered)" in text
    assert "ended in the vae phase" in text
    assert "lane: text-encoder" in text
    assert "qwen3vl.safetensors" in text and "load 42.50s" in text
    assert "encode 3.25s" in text
    assert "determinism: run 2 vs run 1" in text
    assert "determinism: run 3 vs run 1" in text
    assert "memory gate: plateau over run 2 -> run 3" in text
    json.loads(json.dumps(report.to_dict(), default=str))


def test_a_report_with_only_warnings_is_ok(probe):
    report = probe.Report()
    report.checks = [{"name": "w", "ok": False, "detail": "", "gate": False}]
    assert report.ok is True
    assert "RESULT: ok" in report.render()


def test_checks_collect_gates_and_notes(probe):
    checks = probe.Checks()
    checks.expect("one", True, "fine")
    checks.note("two", "informational")
    checks.expect("three", False, "broken", gate=False)
    assert checks.ok is True
    checks.fail("four", "really broken")
    assert checks.ok is False
    assert [c["name"] for c in checks.to_list()] == ["one", "two", "three", "four"]
    assert probe.prefixed(checks.items, "run 1")[0].name == "run 1: one"
