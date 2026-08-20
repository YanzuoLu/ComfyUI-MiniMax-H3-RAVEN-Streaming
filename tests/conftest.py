"""Pytest configuration.

Deliberately import-light: this module must not import torch, ComfyUI, or any
other heavy dependency at collection time, so that pure unit tests (grid math,
canvas rules, chunk boundaries) run in a bare Python environment.

Tests that genuinely need upstream ComfyUI ask for the ``comfyui_on_syspath``
fixture (or call ``require_upstream_comfyui()``), which puts a local ComfyUI
checkout on ``sys.path`` and skips the test when it is absent.

The checkout is expected at ``.cache/upstream/ComfyUI`` (gitignored working
cache) or wherever ``COMFYUI_PATH`` / ``COMFYUI_UPSTREAM_PATH`` points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = PROJECT_ROOT / ".cache" / "upstream" / "ComfyUI"
ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")

# Marker files that identify a directory as a real ComfyUI checkout.
_MARKERS = ("folder_paths.py", "nodes.py", "comfy")


def _looks_like_comfyui(path: Path) -> bool:
    return all((path / marker).exists() for marker in _MARKERS)


def find_upstream_comfyui() -> Path | None:
    """Return the path to a local ComfyUI checkout, or ``None`` if unavailable.

    Resolution order: ``COMFYUI_PATH``, ``COMFYUI_UPSTREAM_PATH``, then
    ``<project root>/.cache/upstream/ComfyUI``.
    """
    candidates = []
    for var in ENV_VARS:
        value = os.environ.get(var)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(DEFAULT_UPSTREAM)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and _looks_like_comfyui(resolved):
            return resolved
    return None


def require_upstream_comfyui() -> Path:
    """Return the ComfyUI checkout path, or skip the calling test."""
    path = find_upstream_comfyui()
    if path is None:
        pytest.skip(
            "No local ComfyUI checkout found. Clone one to "
            f"{DEFAULT_UPSTREAM} or set COMFYUI_PATH."
        )
    return path


def add_to_sys_path(path: Path) -> bool:
    """Prepend ``path`` to ``sys.path``. Returns True if it was added."""
    entry = str(path)
    if entry in sys.path:
        return False
    sys.path.insert(0, entry)
    return True


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def upstream_comfyui_path() -> Path:
    """Path to a local ComfyUI checkout; skips the test when there is none."""
    return require_upstream_comfyui()


@pytest.fixture
def comfyui_on_syspath(upstream_comfyui_path: Path):
    """Put upstream ComfyUI on ``sys.path`` for the duration of one test.

    Nothing is imported here; the test decides what (if anything) to import.
    ``sys.path`` is restored afterwards, but modules imported by the test stay
    in ``sys.modules`` — that is intentional, since re-importing ComfyUI
    repeatedly is expensive and side-effectful.
    """
    added = add_to_sys_path(upstream_comfyui_path)
    try:
        yield upstream_comfyui_path
    finally:
        if added:
            entry = str(upstream_comfyui_path)
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


@pytest.fixture(scope="session")
def raven_streaming_on_syspath() -> Path:
    """Make the repository root importable so ``raven_streaming`` can be imported."""
    add_to_sys_path(PROJECT_ROOT)
    return PROJECT_ROOT
