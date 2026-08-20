"""Static contract tests for the browser extension under ``web/``.

Pure text inspection: no browser, no Node, no ComfyUI import.  These tests are
about the things that silently rot - a CSS class renamed in one file only, an
event listener registered too late to receive anything, a stray ``innerHTML``,
a relative import that resolves to the wrong depth when ComfyUI serves the
files from ``/extensions/<pack>/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB = PROJECT_ROOT / "web"
ENTRY = WEB / "raven_streaming_preview.js"
CSS = WEB / "preview.css"
LIB = WEB / "lib"

LIB_MODULES = ("protocol.js", "sequencer.js", "states.js", "identity.js", "mse.js", "ui.js", "controller.js")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def js_sources() -> dict[str, str]:
    return {path.name: read(path) for path in sorted(WEB.rglob("*.js"))}


# -- layout ----------------------------------------------------------------


def test_expected_files_exist():
    assert ENTRY.is_file()
    assert CSS.is_file()
    assert (WEB / "PROTOCOL.md").is_file()
    for name in LIB_MODULES:
        assert (LIB / name).is_file(), name


def test_only_the_entry_point_is_a_top_level_js_file():
    """ComfyUI loads every ``*.js`` under the web dir as an extension module.

    Helper modules live in ``lib/`` and must stay side-effect free, so being
    loaded standalone by that glob is a no-op.
    """
    top_level = sorted(p.name for p in WEB.glob("*.js"))
    assert top_level == ["raven_streaming_preview.js"]


def test_helper_modules_register_nothing_at_import_time():
    for name in LIB_MODULES:
        source = read(LIB / name)
        assert "registerExtension" not in source, name
        assert "scripts/app.js" not in source, name
        assert "scripts/api.js" not in source, name


def test_test_harness_is_not_served_to_browsers():
    """``/extensions`` globs ``*.js`` only; the harness must not be picked up."""
    harnesses = sorted(p.name for p in (WEB / "tests").glob("*"))
    assert harnesses, "the Node harness is missing"
    for name in harnesses:
        assert name.endswith(".mjs"), name


def test_web_package_json_declares_es_modules():
    data = json.loads(read(WEB / "package.json"))
    assert data["type"] == "module"
    assert data["private"] is True


# -- ComfyUI integration points -------------------------------------------


def test_entry_imports_app_and_api_from_the_served_shims():
    source = read(ENTRY)
    assert "import { app } from '../../scripts/app.js'" in source
    assert "import { api } from '../../scripts/api.js'" in source


def test_entry_imports_helpers_by_relative_path():
    source = read(ENTRY)
    for module in ("controller.js", "identity.js", "protocol.js"):
        assert f"./lib/{module}'" in source, module


def test_custom_message_listener_is_registered_in_setup():
    """The pinned frontend drops a custom JSON type unless a listener already
    exists for it, so registration cannot be deferred to node creation."""
    source = read(ENTRY)
    setup_start = source.index("async setup()")
    setup_end = source.index("async beforeRegisterNodeDef")
    setup_body = source[setup_start:setup_end]
    assert "api.addEventListener(MESSAGE_TYPE" in setup_body
    for event in (
        "execution_start",
        "execution_interrupted",
        "execution_error",
        "executed",
        "reconnecting",
        "reconnected",
        "graphCleared",
    ):
        assert f"api.addEventListener('{event}'" in setup_body, event


def test_widget_is_created_through_the_dom_widget_api():
    source = read(ENTRY)
    assert "node.addDOMWidget(" in source
    assert "hideOnZoom: true" in source
    assert "getMinHeight:" in source


def test_widget_value_is_kept_out_of_workflow_and_prompt():
    """`options.serialize` controls the API prompt, `widget.serialize` the
    workflow JSON. A preview is neither an input nor state worth saving."""
    source = read(ENTRY)
    assert "serialize: false" in source
    assert "widget.serialize = false" in source


def test_node_lifecycle_hooks_are_chained_not_replaced():
    source = read(ENTRY)
    assert "const onNodeCreated = nodeType.prototype.onNodeCreated" in source
    assert "onNodeCreated ? onNodeCreated.apply(this, args)" in source
    assert "const onRemoved = nodeType.prototype.onRemoved" in source
    assert "onRemoved ? onRemoved.apply(this, args)" in source


def test_cleanup_paths_exist_for_removal_and_graph_reload():
    source = read(ENTRY)
    assert "function detach(" in source
    assert "detachAll()" in source
    assert "pruneDetachedNodes" in source
    assert "afterConfigureGraph()" in source


def test_every_comfy_callback_is_guarded():
    """A thrown preview error must never reach ComfyUI's dispatch, even though
    the frontend also wraps listeners itself."""
    source = read(ENTRY)
    assert source.count("safely(") >= 6
    assert "function safely(fn) {\n  try {" in source


# -- hygiene ---------------------------------------------------------------


@pytest.mark.parametrize("banned", ["innerHTML", "outerHTML", "document.write", "eval(", "new Function("])
def test_no_unsafe_dom_or_dynamic_code(banned):
    for name, source in js_sources().items():
        assert banned not in source, f"{name} uses {banned}"


def test_no_external_network_references():
    """Everything must load from the ComfyUI server; no CDN, no font service."""
    pattern = re.compile(r"https?://(?!www\.w3\.org)", re.IGNORECASE)
    for name, source in js_sources().items():
        assert not pattern.search(source), f"{name} references an external URL"
    assert not pattern.search(read(CSS))


def test_object_urls_are_revoked_where_they_are_created():
    source = read(LIB / "mse.js")
    assert "createObjectURL" in source
    assert "revokeObjectURL" in source


def test_media_pipeline_has_no_ffmpeg_or_wasm_dependency():
    for name, source in js_sources().items():
        lowered = source.lower()
        assert "ffmpeg" not in lowered, name
        assert ".wasm" not in lowered, name
    assert "MediaSource" in read(LIB / "mse.js")


# -- CSS / DOM parity ------------------------------------------------------


def css_class_names(css: str) -> set[str]:
    return set(re.findall(r"\.(rvp[\w-]*)", css))


def js_class_names() -> set[str]:
    names: set[str] = set()
    for source in js_sources().values():
        names.update(re.findall(r"'(rvp[\w-]*(?:__[\w-]+)?[^']*)'", source))
    out: set[str] = set()
    for value in names:
        out.update(token for token in value.split() if token.startswith("rvp"))
    return out


def test_every_class_used_by_the_dom_is_styled():
    styled = css_class_names(read(CSS))
    used = js_class_names()
    assert used, "no widget classes found in the JS"
    missing = sorted(used - styled)
    assert not missing, f"classes with no styling: {missing}"


def test_every_styled_class_is_actually_used():
    styled = css_class_names(read(CSS))
    used = js_class_names()
    # `is-muted` is toggled through classList and carries no `rvp` prefix.
    unused = sorted(styled - used)
    assert not unused, f"dead CSS classes: {unused}"


def test_state_tones_have_styling():
    """Each tone the state table can produce needs a visual treatment."""
    states = read(LIB / "states.js")
    tones = set(re.findall(r"tone:\s*'(\w+)'", states))
    css = read(CSS)
    for tone in tones:
        if tone == "idle":
            continue  # the default dot colour
        assert f"data-tone='{tone}'" in css, tone


def test_colours_fall_back_when_comfy_variables_are_absent():
    css = read(CSS)
    for var in ("--comfy-input-bg", "--comfy-menu-bg", "--border-color", "--input-text", "--descrip-text"):
        match = re.search(rf"var\({var},\s*([^)]+)\)", css)
        assert match, f"{var} is used without a fallback"


# -- accessibility and responsiveness -------------------------------------


def test_status_is_announced_and_controls_are_labelled():
    ui = read(LIB / "ui.js")
    assert "role: 'status'" in ui
    assert "'aria-live': 'polite'" in ui
    assert "aria-pressed" in ui
    assert "aria-label" in ui
    assert "role === 'error' ? 'alert'" in ui or "'alert'" in ui


def test_decorative_icons_are_hidden_from_assistive_tech():
    ui = read(LIB / "ui.js")
    assert "svg.setAttribute('aria-hidden', 'true')" in ui
    assert "svg.setAttribute('focusable', 'false')" in ui


def test_focus_is_visible_and_motion_can_be_turned_off():
    css = read(CSS)
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "forced-colors" in css


def test_native_playback_controls_stay_available():
    ui = read(LIB / "ui.js")
    assert "controls: true" in ui
    assert "playsinline: true" in ui
    assert "video.muted = true" in ui
    assert "video.autoplay = true" in ui


def test_layout_reacts_to_node_width():
    css = read(CSS)
    assert "container-type: inline-size" in css
    assert "@container" in css
    ui = read(LIB / "ui.js")
    assert "ResizeObserver" in ui
    assert "rvp--narrow" in ui


def test_the_widget_reports_a_height_for_the_node():
    ui = read(LIB / "ui.js")
    assert "preferredHeight(" in ui
    assert "--rvp-aspect" in ui


# -- tone ------------------------------------------------------------------


def test_labels_are_plain_and_unexcited():
    """No marketing voice in a node widget."""
    text = read(LIB / "states.js") + read(LIB / "ui.js")
    for word in ("amazing", "awesome", "magic", "stunning", "blazing", "seamless", "🚀", "✨"):
        assert word not in text.lower(), word
    shouted = re.findall(r"'[^']*!'", text) + re.findall(r'`[^`]*!`', text)
    assert not shouted, f"exclamation marks in user-facing strings: {shouted}"
