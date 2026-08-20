from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / ".cache" / "upstream" / "ComfyUI"


def _load_entrypoint():
    name = "_raven_root_entrypoint_test"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old_path = list(sys.path)
    try:
        if COMFY.is_dir():
            sys.path.insert(0, str(COMFY))
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.modules.pop(name, None)


def test_root_entrypoint_exposes_exact_v1_contract():
    module = _load_entrypoint()
    assert module.WEB_DIRECTORY == "./web"
    assert sorted(module.NODE_CLASS_MAPPINGS) == [
        "RAVENModelLoader",
        "RAVENStreamingSampler",
    ]
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {
        "RAVENModelLoader": "RAVEN Model Loader",
        "RAVENStreamingSampler": "RAVEN Streaming Sampler",
    }
    assert not hasattr(module, "comfy_entrypoint")


def test_root_web_directory_exists():
    module = _load_entrypoint()
    assert (ROOT / module.WEB_DIRECTORY).resolve() == (ROOT / "web").resolve()
    assert (ROOT / module.WEB_DIRECTORY).is_dir()


def test_root_entrypoint_keeps_one_runtime_class_identity():
    module = _load_entrypoint()
    from raven_streaming import consistency, streaming_pipeline

    assert streaming_pipeline._chunk_output_class() is consistency.ChunkOutput
    assert module.NODE_CLASS_MAPPINGS["RAVENStreamingSampler"].__module__ == (
        "raven_streaming.nodes"
    )
    duplicate = [
        name
        for name in sys.modules
        if name.endswith(".raven_streaming.consistency")
        and name != "raven_streaming.consistency"
    ]
    assert duplicate == []
