"""The node schema and the ComfyUI registration surface.

No ComfyUI is present here and none is faked: the point of these tests is that
the schema, the mappings and the web-directory declaration are readable in a
bare interpreter, because that is the state upstream itself is in when it
imports a node pack (``nodes.py::load_custom_node`` execs the module before
anything of ours has run).

The subprocess checks are deliberate. ``sys.modules`` is shared across a pytest
session, so an earlier test that put a ComfyUI checkout on the path would make
an in-process "is torch imported?" assertion meaningless.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raven_streaming import consistency, loader as loader_mod, nodes  # noqa: E402


def run_python(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that can see only this package."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )


# --------------------------------------------------------------------------
# registration surface
# --------------------------------------------------------------------------


def test_exactly_two_nodes_are_registered():
    assert set(nodes.NODE_CLASS_MAPPINGS) == {"RAVENModelLoader", "RAVENStreamingSampler"}
    assert nodes.NODE_CLASS_MAPPINGS["RAVENModelLoader"] is nodes.RAVENModelLoader
    assert nodes.NODE_CLASS_MAPPINGS["RAVENStreamingSampler"] is nodes.RAVENStreamingSampler
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS == {
        "RAVENModelLoader": "RAVEN Model Loader",
        "RAVENStreamingSampler": "RAVEN Streaming Sampler",
    }


def test_the_package_exposes_what_comfyui_reads():
    import raven_streaming

    assert raven_streaming.WEB_DIRECTORY == "./web"
    assert raven_streaming.NODE_CLASS_MAPPINGS is nodes.NODE_CLASS_MAPPINGS
    assert raven_streaming.NODE_DISPLAY_NAME_MAPPINGS is nodes.NODE_DISPLAY_NAME_MAPPINGS
    # hasattr is what upstream actually calls, and it must work through the
    # lazy __getattr__
    assert hasattr(raven_streaming, "NODE_CLASS_MAPPINGS")
    assert hasattr(raven_streaming, "WEB_DIRECTORY")
    with pytest.raises(AttributeError):
        raven_streaming.NODE_CONFIG_MAPPINGS  # noqa: B018 - the point is the raise


def test_no_v3_entrypoint_competes_with_the_v1_mappings():
    import raven_streaming

    for module in (raven_streaming, nodes):
        assert not hasattr(module, "comfy_entrypoint")
        assert not hasattr(module, "ComfyExtension")
    source = (ROOT / "raven_streaming" / "nodes.py").read_text(encoding="utf-8")
    assert "comfy_entrypoint" not in source.replace("``comfy_entrypoint``", "")


def test_the_sampler_is_registered_under_a_name_the_client_matches():
    identity = (ROOT / "web" / "lib" / "identity.js").read_text(encoding="utf-8")
    names = re.findall(r"'([^']+)'", identity.split("SAMPLER_NODE_NAMES")[1].split("]")[0])
    assert "RAVENStreamingSampler" in names
    assert "RAVENStreamingSampler" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["RAVENStreamingSampler"] in names


def test_importing_the_package_stays_import_light(tmp_path):
    result = run_python(
        "import sys, json, raven_streaming as rs;"
        "assert rs.__version__ and rs.WEB_DIRECTORY;"
        "print(json.dumps([m for m in ('torch', 'numpy', 'av', 'comfy') "
        "if m in sys.modules]))",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_reading_the_mappings_loads_the_nodes_without_comfyui(tmp_path):
    result = run_python(
        "import sys, json, raven_streaming as rs;"
        "mappings = rs.NODE_CLASS_MAPPINGS;"
        "print(json.dumps({"
        "  'names': sorted(mappings),"
        "  'comfy': [m for m in sys.modules if m == 'comfy' or m.startswith('comfy.')],"
        "  'torch': 'torch' in sys.modules,"
        "}))",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["names"] == ["RAVENModelLoader", "RAVENStreamingSampler"]
    assert payload["comfy"] == []
    assert payload["torch"] is True  # the node module owns the torch dependency


def test_reading_the_mappings_installs_the_preview_once(tmp_path):
    result = run_python(
        "import raven_streaming as rs;"
        "calls = [];"
        "rs.install_preview = lambda *a, **k: calls.append(1);"
        "rs.NODE_CLASS_MAPPINGS;"
        "rs.__getattr__('NODE_CLASS_MAPPINGS');"
        "rs._install_preview_once();"
        "print(len(calls))",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "1"


def test_the_node_module_imports_with_no_comfyui_present(tmp_path):
    result = run_python(
        "import sys;"
        "from raven_streaming import nodes;"
        "print([m for m in sys.modules if m == 'comfy' or m.startswith('comfy.')])",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


# --------------------------------------------------------------------------
# Node 1 schema
# --------------------------------------------------------------------------


def test_loader_takes_exactly_a_dit_a_lora_and_a_dtype():
    schema = nodes.RAVENModelLoader.INPUT_TYPES()
    assert set(schema) == {"required"}
    assert list(schema["required"]) == ["unet_name", "lora_name", "weight_dtype"]


def test_loader_does_not_expose_the_raven_strength():
    schema = nodes.RAVENModelLoader.INPUT_TYPES()
    assert "strength" not in schema["required"]
    assert "lora_strength" not in schema["required"]
    # v0.1: mandatory means mandatory. There is no 0 / "disable" state.
    assert nodes.RAVEN_LORA_STRENGTH == 1.0


def test_loader_output_and_placement():
    assert nodes.RAVENModelLoader.RETURN_TYPES == ("MODEL",)
    assert nodes.RAVENModelLoader.CATEGORY == "model/loaders/raven"
    assert nodes.RAVENModelLoader.FUNCTION == "load_model"
    assert callable(getattr(nodes.RAVENModelLoader, "load_model"))


def test_loader_combos_come_from_the_right_folders_and_dtypes():
    schema = nodes.RAVENModelLoader.INPUT_TYPES()["required"]
    # No ComfyUI here, so folder_paths yields nothing; the point is that it is a
    # combo (a list) rather than a free-text path.
    assert isinstance(schema["unet_name"][0], list)
    assert isinstance(schema["lora_name"][0], list)
    assert schema["weight_dtype"][0] == list(loader_mod.WEIGHT_DTYPE_CHOICES)
    assert schema["weight_dtype"][0] == ["default", "bf16", "fp32"]
    assert schema["weight_dtype"][1]["default"] == "default"
    # FP8/INT8 would silently bypass the activation residual
    assert not [choice for choice in schema["weight_dtype"][0] if "8" in choice]


def test_loader_tooltips_state_the_three_things_that_cost_the_user_time():
    schema = nodes.RAVENModelLoader.INPUT_TYPES()["required"]
    assert "full, non-pruned BF16" in schema["unet_name"][1]["tooltip"]
    assert "5 GB" in schema["lora_name"][1]["tooltip"]
    dtype_tooltip = schema["weight_dtype"][1]["tooltip"]
    assert "132 GB" in dtype_tooltip
    assert "OOM" in dtype_tooltip
    assert "1.0" in nodes.RAVENModelLoader.DESCRIPTION


# --------------------------------------------------------------------------
# Node 2 schema
# --------------------------------------------------------------------------


def test_sampler_required_inputs_are_exactly_the_contract():
    schema = nodes.RAVENStreamingSampler.INPUT_TYPES()
    assert list(schema["required"]) == [
        "model",
        "positive",
        "latent",
        "video_vae",
        "audio_vae",
        "seed",
        "steps",
        "video_shift",
        "audio_shift",
        "sink",
        "window",
        "kv_cache_storage",
    ]
    assert schema["hidden"] == {"unique_id": "UNIQUE_ID"}
    assert set(schema) == {"required", "hidden"}


@pytest.mark.parametrize(
    "absent",
    [
        "negative",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "width",
        "height",
        "frames",
        "length",
        "fps",
    ],
)
def test_sampler_refuses_to_offer_what_it_cannot_honour(absent):
    schema = nodes.RAVENStreamingSampler.INPUT_TYPES()
    assert absent not in schema["required"]
    assert absent not in schema.get("optional", {})


@pytest.mark.parametrize(
    "absent",
    [
        "gpu_memory_cap_gb",
        "gpu_memory_cap",
        "vram_cap_gb",
        "max_resident_gb",
        "max_vram_gb",
        "reserve_vram",
        "offload",
        "lowvram",
        "keep_model_loaded",
        "memory_strategy",
    ],
)
def test_the_sampler_exposes_no_residency_or_cap_input(absent):
    """Residency is ComfyUI's decision, so there is nothing here to state it with.

    ``load_models_gpu`` sizes weight residency against the memory that is
    actually free at the moment of the load. A widget for it would be a second,
    worse estimate of that same quantity sitting in front of the real one: state
    it too low and a run that fitted is offloaded into a crawl, too high and one
    that did not fit OOMs. The node's job is to declare its *workspace*
    truthfully, which it does per phase and without asking.
    """
    schema = nodes.RAVENStreamingSampler.INPUT_TYPES()
    assert absent not in schema["required"]
    assert absent not in schema.get("optional", {})
    signature = inspect.signature(nodes.RAVENStreamingSampler.sample)
    assert absent not in signature.parameters


def test_the_only_memory_input_is_where_the_kv_cache_lives():
    """One memory-shaped widget, and it moves bytes rather than capping them."""
    required = nodes.RAVENStreamingSampler.INPUT_TYPES()["required"]
    memory_inputs = [
        name
        for name, spec in required.items()
        if any(
            word in spec[1].get("tooltip", "").lower()
            for word in ("vram", "memory", "resident")
        )
    ]
    assert memory_inputs == ["kv_cache_storage"]

    choices, options = required["kv_cache_storage"]
    assert choices == list(nodes.KV_CACHE_STORAGE_CHOICES) == ["cpu_pinned", "cpu", "gpu"]
    assert options["default"] == nodes.DEFAULT_KV_CACHE_STORAGE == "cpu_pinned"
    # it says what it changes, and what it does not
    assert "not what is computed" in options["tooltip"]


def test_the_24_gib_reference_is_a_constant_and_says_it_is_not_a_cap():
    assert nodes.GPU_HARD_CAP_BYTES == 24 * 1024 ** 3
    assert nodes.PLANNING_BUDGET_BYTES == 22 * 1024 ** 3
    assert nodes.planning_bytes() == nodes.PLANNING_BUDGET_BYTES

    source = (ROOT / "raven_streaming" / "nodes.py").read_text(encoding="utf-8")
    constant = source.split("GPU_HARD_CAP_BYTES = ")[0].rsplit("#: The reference card", 1)
    assert len(constant) == 2, "the constant lost the comment explaining itself"
    assert "not a cap" in constant[1]
    # ... and the node never turns it into one
    assert "raise" not in inspect.getsource(nodes.hard_cap_watch)


def test_the_node_never_decides_residency_itself():
    """No ``.to()``, no ``partially_load/unload``, no ``force_full_load=True``.

    Checked against the parse tree rather than the text, so that the prose
    explaining *why* these calls are absent does not count as making them.
    """
    import ast

    tree = ast.parse((ROOT / "raven_streaming" / "nodes.py").read_text(encoding="utf-8"))
    forbidden = {"partially_unload", "partially_load", "patch_model", "model_load", "to"}
    called = set()
    full_load = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
            called.add(node.func.attr)
        for keyword in node.keywords:
            if keyword.arg in ("force_full_load", "full_load"):
                full_load.append(ast.unparse(keyword.value))
    assert called == set(), called
    # every load this node makes is a partial one
    assert full_load and set(full_load) == {"False"}, full_load


def test_sampler_socket_types_are_the_official_ones():
    required = nodes.RAVENStreamingSampler.INPUT_TYPES()["required"]
    assert required["model"][0] == "MODEL"
    assert required["positive"][0] == "CONDITIONING"
    assert required["latent"][0] == "LATENT"
    assert required["video_vae"][0] == "VAE"
    assert required["audio_vae"][0] == "VAE"


def test_sampler_defaults_are_ravens_published_trial():
    required = nodes.RAVENStreamingSampler.INPUT_TYPES()["required"]
    assert required["steps"][1]["default"] == 4 == consistency.DEFAULT_STEPS
    assert required["video_shift"][1]["default"] == 12.0 == consistency.DEFAULT_VIDEO_SHIFT
    assert required["audio_shift"][1]["default"] == 3.0 == consistency.DEFAULT_AUDIO_SHIFT
    assert required["sink"][1]["default"] == 2 == consistency.DEFAULT_SINK
    assert required["window"][1]["default"] == 2 == consistency.DEFAULT_WINDOW
    assert required["sink"][1]["min"] == 1  # chunk 0 is the text prefill
    assert required["window"][1]["min"] == 0
    assert required["steps"][1]["min"] == 1
    assert required["seed"][1]["control_after_generate"] is True
    # the host-backed cache is the default: it is what makes the published
    # request fit a card that cannot hold 28 GiB of KV
    assert required["kv_cache_storage"][1]["default"] == "cpu_pinned"


def test_sampler_outputs_and_placement():
    assert nodes.RAVENStreamingSampler.RETURN_TYPES == ("LATENT", "IMAGE", "AUDIO")
    assert nodes.RAVENStreamingSampler.RETURN_NAMES == ("LATENT", "IMAGE", "AUDIO")
    assert len(nodes.RAVENStreamingSampler.OUTPUT_TOOLTIPS) == 3
    assert nodes.RAVENStreamingSampler.CATEGORY == "model/sampling/raven"
    assert nodes.RAVENStreamingSampler.FUNCTION == "sample"


def test_sampler_description_says_where_the_geometry_comes_from():
    description = nodes.RAVENStreamingSampler.DESCRIPTION
    assert "LATENT" in description
    assert "CFG" in description
    latent_tooltip = nodes.RAVENStreamingSampler.INPUT_TYPES()["required"]["latent"][1][
        "tooltip"
    ]
    assert "no width/height/frames" in latent_tooltip
    assert "non-empty latent is refused" in latent_tooltip
