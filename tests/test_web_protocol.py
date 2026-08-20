"""``web/PROTOCOL.md`` has to agree with the client that implements it.

The document is the only thing the Python side will be written against, so a
constant that drifts out of it is a bug waiting for a backend author.  These
tests compare the document with the JavaScript, and check that the claims it
makes about the pinned ComfyUI are still true of the checkout in
``.cache/upstream/ComfyUI`` when one is present.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import find_upstream_comfyui

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB = PROJECT_ROOT / "web"
DOC = WEB / "PROTOCOL.md"
PROTOCOL_JS = WEB / "lib" / "protocol.js"
STATES_JS = WEB / "lib" / "states.js"
IDENTITY_JS = WEB / "lib" / "identity.js"

PINNED_COMMIT = "c67885b14556cf3e4e061862925282d403d09862"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def js_string_list(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}\s*=\s*Object\.freeze\(\[(.*?)\]\)", source, re.S)
    assert match, f"{name} not found"
    return re.findall(r"'([^']+)'", match.group(1))


@pytest.fixture(scope="module")
def doc() -> str:
    return read(DOC)


# -- constants -------------------------------------------------------------


def test_message_type_matches(doc):
    source = read(PROTOCOL_JS)
    match = re.search(r"MESSAGE_TYPE\s*=\s*'([^']+)'", source)
    assert match
    message_type = match.group(1)
    assert message_type == "raven.preview"
    assert f"`{message_type}`" in doc


def test_protocol_version_matches(doc):
    source = read(PROTOCOL_JS)
    version = int(re.search(r"PROTOCOL_VERSION\s*=\s*(\d+)", source).group(1))
    assert f"(v{version})" in doc
    assert f"currently `{version}`" in doc


def test_every_event_kind_has_a_section(doc):
    kinds = js_string_list(read(PROTOCOL_JS), "EVENT_KINDS")
    assert kinds == ["open", "init", "segment", "status", "end"]
    for kind in kinds:
        assert re.search(rf"^### 3\.\d+ `{kind}`", doc, re.M), kind


def test_backend_phases_match(doc):
    phases = js_string_list(read(PROTOCOL_JS), "BACKEND_PHASES")
    # The cell escapes its separators as ``\|``, so take the whole line.
    documented = re.search(r"^\| `phase` \| string \| yes \|(.*)$", doc, re.M).group(1)
    for phase in phases:
        assert f"`{phase}`" in documented, phase
    # Client-derived states must not be listed as things a backend may send.
    for client_only in ("buffering", "live", "reconnecting"):
        assert f"`{client_only}`" not in documented


def test_end_reasons_match(doc):
    reasons = js_string_list(read(PROTOCOL_JS), "END_REASONS")
    assert reasons == ["complete", "cancelled", "error"]
    for reason in reasons:
        assert re.search(rf"^- `{reason}` — ", doc, re.M), reason


def test_payload_encoding_matches(doc):
    encodings = js_string_list(read(PROTOCOL_JS), "PAYLOAD_ENCODINGS")
    assert encodings == ["base64"]
    assert "`base64`" in doc


def test_resume_route_matches(doc):
    route = re.search(r"RESUME_ROUTE\s*=\s*'([^']+)'", read(PROTOCOL_JS)).group(1)
    assert route in doc


def test_display_states_are_documented_as_client_side(doc):
    states = js_string_list(read(STATES_JS), "STATES")
    for state in ("buffering", "live", "reconnecting"):
        assert state in states
    assert "**client-side** states" in doc


def test_accepted_node_names_match(doc):
    names = js_string_list(read(IDENTITY_JS), "SAMPLER_NODE_NAMES")
    for name in names:
        assert name in doc, name
    assert "unique_id" in doc


# -- the reasoning the document rests on -----------------------------------


def test_document_states_why_binary_frames_are_not_used(doc):
    assert "Unknown binary websocket message of type" in doc
    assert "protocol.py" in doc
    assert "server.py" in doc
    assert PINNED_COMMIT in doc


def test_document_quantifies_the_base64_overhead(doc):
    assert "ceil(n/3) * 4" in doc
    assert "33.3" in doc
    assert re.search(r"\+33\.3 ?%", doc)


def test_document_lists_what_the_backend_must_do(doc):
    section = doc[doc.index("## 6. What the backend must provide") :]
    assert "WEB_DIRECTORY" in section
    assert "[tool.comfy]" in section
    assert "send_sync" in section
    assert "unique_id" in section
    # The root V1 entry point now owns registration; the document must state
    # that fact rather than preserving the pre-implementation warning.
    assert "repository root `__init__.py`" in section
    assert 'WEB_DIRECTORY = "./web"' in section
    assert "Neither is set" not in section


def test_document_covers_loss_duplicates_and_resume(doc):
    section = doc[doc.index("## 4. Loss, duplicates and resume") : doc.index("## 5.")]
    assert "duplicate" in section
    assert "last_seq" in section
    assert "404" in section
    assert "resync" in section


def test_document_covers_session_and_node_isolation(doc):
    section = doc[doc.index("## 5. Session and node isolation") : doc.index("## 6.")]
    assert "session_id" in section
    assert "node_id" in section
    assert "-1" in section


def test_document_states_the_preview_cannot_break_sampling(doc):
    assert "must not affect" in doc.lower() or "must not break" in doc.lower()


# -- claims about the pinned upstream --------------------------------------


def test_claims_about_upstream_hold_where_the_checkout_is_available():
    upstream = find_upstream_comfyui()
    if upstream is None:
        pytest.skip("no ComfyUI checkout available")

    server = (upstream / "server.py").read_text(encoding="utf-8")
    protocol = (upstream / "protocol.py").read_text(encoding="utf-8")

    # Binary sends require an integer event id, and the four ids the frontend
    # decodes are core ones we must not reuse.
    assert "Binary event types must be integers" in server
    assert "def send_sync(self, event, data, sid=None)" in server
    assert '{"type": event, "data": data}' in server
    for name in ("PREVIEW_IMAGE", "UNENCODED_PREVIEW_IMAGE", "TEXT", "PREVIEW_IMAGE_WITH_METADATA"):
        assert name in protocol

    # The extension glob that serves this directory.
    assert "'extensions/**/*.js'" in server
    nodes = (upstream / "nodes.py").read_text(encoding="utf-8")
    assert "EXTENSION_WEB_DIRS" in nodes
    assert 'hasattr(module, "WEB_DIRECTORY")' in nodes


def test_pinned_frontend_version_is_the_one_that_was_audited():
    upstream = find_upstream_comfyui()
    if upstream is None:
        pytest.skip("no ComfyUI checkout available")
    requirements = (upstream / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"comfyui-frontend-package==([\d.]+)", requirements)
    assert match, "the pinned frontend version disappeared from requirements.txt"
    assert match.group(1) == "1.49.6", (
        "the frontend pin moved; re-check the websocket and DOM widget APIs "
        "documented in web/PROTOCOL.md against the new version"
    )
