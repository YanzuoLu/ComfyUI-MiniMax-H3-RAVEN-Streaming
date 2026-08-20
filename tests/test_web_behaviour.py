"""Run the JavaScript behaviour harnesses under Node.

The logic they cover - envelope validation, sequence ordering, de-duplication,
gap handling and resume, the MediaSource append queue, session isolation and
teardown - is written as browser-free modules precisely so it can be exercised
here with mock events instead of only by clicking around a browser.

Node is optional: without it these tests skip, and the static contract tests in
``test_web_static.py`` still run.  What no harness can cover is real playback
and real layout; those remain browser-only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / "web" / "tests"
HARNESSES = ("run_checks.mjs", "run_controller_checks.mjs")

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@requires_node
@pytest.mark.parametrize("harness", HARNESSES)
def test_javascript_harness_passes(harness):
    path = HARNESS_DIR / harness
    assert path.is_file(), harness

    result = subprocess.run(
        [NODE, str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = result.stdout + result.stderr
    assert result.returncode == 0, f"\n{report}"
    assert "checks passed" in report

    failures = [line for line in report.splitlines() if line.startswith("FAIL")]
    assert not failures, "\n".join(failures)


@requires_node
@pytest.mark.parametrize("harness", HARNESSES)
def test_harness_actually_runs_checks(harness):
    """Guard against a harness that silently stops asserting anything."""
    result = subprocess.run(
        [NODE, str(HARNESS_DIR / harness)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = [line for line in result.stdout.splitlines() if line.startswith("ok ")]
    assert len(passed) >= 20, f"only {len(passed)} checks ran in {harness}"


@requires_node
@pytest.mark.parametrize("module", sorted(p.name for p in (PROJECT_ROOT / "web" / "lib").glob("*.js")))
def test_every_library_module_parses_and_imports_cleanly(module):
    """Import each module on its own: none may need a browser at import time.

    ComfyUI's ``/extensions`` glob loads every ``*.js`` in this pack, including
    the helpers, so an import-time reference to ``window`` or ``document``
    would break for real.
    """
    target = (PROJECT_ROOT / "web" / "lib" / module).as_uri()
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", f"await import({target!r})"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{module}: {result.stderr}"


@requires_node
def test_entry_point_parses():
    """The entry point cannot be imported (its ComfyUI imports do not resolve
    outside the server), so check its syntax instead."""
    entry = PROJECT_ROOT / "web" / "raven_streaming_preview.js"
    result = subprocess.run(
        [NODE, "--check", str(entry)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
