"""Transport and resume-route behaviour, against a fake ``PromptServer``.

Two things are checked here that a fake alone cannot establish:

* the resume decisions of ``PreviewManager.handle_resume`` (pure, no HTTP), and
* that the fake's shape is the *real* one -- the last section reads the pinned
  ComfyUI checkout and asserts the signatures this module binds to, so a fake
  that drifts from upstream fails instead of passing quietly.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import find_upstream_comfyui  # noqa: E402
from raven_streaming import preview  # noqa: E402
from raven_streaming.preview_session import (  # noqa: E402
    API_RESUME_ROUTE,
    MESSAGE_TYPE,
    RESUME_BAD_REQUEST,
    RESUME_EXPIRED,
    RESUME_MISMATCH,
    RESUME_REPLAYED,
    RESUME_RESYNC,
    RESUME_ROUTE,
    RESUME_TERMINAL,
    RESUME_UNKNOWN,
    RESUME_UP_TO_DATE,
    PreviewManager,
    PreviewSendError,
    RecordingSender,
)
from raven_streaming.preview_server import (  # noqa: E402
    PromptServerSender,
    current_client_id,
    is_route_registered,
    make_resume_handler,
    register_resume_route,
    resolve_prompt_server,
)

MIME = 'video/mp4; codecs="avc1.640028,mp4a.40.2"'
PINNED_COMMIT = "c67885b14556cf3e4e061862925282d403d09862"


# -- fakes -----------------------------------------------------------------


class FakeRouteTableDef:
    """Stand-in for ``aiohttp.web.RouteTableDef``.

    Upstream's real one is iterable and yields ``RouteDef`` objects carrying
    ``method`` / ``path`` / ``handler``; ``server.add_routes`` relies on exactly
    those attributes (``server.py:1233-1241``).
    """

    class RouteDef:
        def __init__(self, method, path, handler):
            self.method = method
            self.path = path
            self.handler = handler
            self.kwargs = {}

    def __init__(self):
        self._items = []

    def __iter__(self):
        return iter(self._items)

    def route(self, method, path, **kwargs):
        def decorator(handler):
            self._items.append(self.RouteDef(method, path, handler))
            return handler

        return decorator

    def post(self, path, **kwargs):
        return self.route("POST", path, **kwargs)


class FakeResource:
    def __init__(self, canonical, method, handler):
        self.canonical = canonical
        self.method = method
        self.handler = handler


class FakeRouter:
    def __init__(self):
        self._resources = []
        self.frozen = False

    def resources(self):
        return list(self._resources)

    def add_route(self, method, path, handler):
        if self.frozen:
            raise RuntimeError("Cannot register a route after the app has started")
        for resource in self._resources:
            if resource.canonical == path:
                raise ValueError(f"duplicate route {path}")
        resource = FakeResource(path, method, handler)
        self._resources.append(resource)
        return resource


class FakeApp:
    def __init__(self):
        self.router = FakeRouter()


class FakePromptServer:
    """The surface this pack binds to, and nothing else."""

    def __init__(self, *, client_id=None):
        self.routes = FakeRouteTableDef()
        self.app = FakeApp()
        self.client_id = client_id
        self.sockets = {}
        self.sent = []
        self.fail = False

    def send_sync(self, event, data, sid=None):
        if self.fail:
            raise RuntimeError("loop is closed")
        self.sent.append((event, data, sid))

    def add_routes(self):
        """The pinned behaviour: copy the table into the app under /api too."""
        for route in self.routes:
            self.app.router.add_route(route.method, "/api" + route.path, route.handler)
            self.app.router.add_route(route.method, route.path, route.handler)


class FakeRequest:
    def __init__(self, payload, *, raise_on_json=False):
        self._payload = payload
        self._raise = raise_on_json

    async def json(self):
        if self._raise:
            raise ValueError("not JSON")
        return self._payload


def run(coro):
    return asyncio.run(coro)


def responses(body, status):
    return {"body": body, "status": status}


@pytest.fixture
def sender():
    return RecordingSender()


def open_stream(session):
    session.send_open(MIME)
    session.send_init(b"init")


# -- sender ----------------------------------------------------------------


def test_sender_uses_send_sync_with_type_body_and_sid():
    server = FakePromptServer()
    send = PromptServerSender(server)
    send(MESSAGE_TYPE, {"v": 1, "event": "open"}, "cid")
    assert server.sent == [("raven.preview", {"v": 1, "event": "open"}, "cid")]
    assert send.sent == 1


def test_sender_without_a_server_raises_preview_send_error():
    send = PromptServerSender(server=None)
    # No ComfyUI on the path in this environment, so resolution yields nothing.
    if resolve_prompt_server() is None:
        with pytest.raises(PreviewSendError):
            send(MESSAGE_TYPE, {}, None)
        assert send.failures == 1


def test_sender_failure_is_wrapped_and_isolated_by_the_session():
    server = FakePromptServer()
    server.fail = True
    manager = PreviewManager(PromptServerSender(server))
    session = manager.create_session("7", client_id="cid")
    session.send_open(MIME)  # must not raise into the sampler
    assert session.send_failures == 1
    assert session.messages_sent == 0


def test_current_client_id_reads_the_executing_client():
    assert current_client_id(FakePromptServer(client_id="abc")) == "abc"
    assert current_client_id(FakePromptServer(client_id=None)) is None


# -- route registration ----------------------------------------------------


def test_route_is_registered_on_the_route_table_before_add_routes(sender):
    server = FakePromptServer()
    manager = PreviewManager(sender)
    assert register_resume_route(manager, server) is True
    paths = [(r.method, r.path) for r in server.routes]
    assert paths == [("POST", RESUME_ROUTE)]
    # Upstream's /api mirror is what the client actually calls.
    server.add_routes()
    assert any(r.canonical == API_RESUME_ROUTE for r in server.app.router.resources())


def test_registration_is_idempotent(sender):
    server = FakePromptServer()
    manager = PreviewManager(sender)
    assert register_resume_route(manager, server) is True
    assert register_resume_route(manager, server) is False
    assert register_resume_route(manager, server) is False
    assert len(list(server.routes)) == 1
    assert is_route_registered(server) is True


def test_registration_after_add_routes_goes_straight_to_the_app(sender):
    server = FakePromptServer()
    server.routes.post("/prompt")(lambda request: None)
    server.add_routes()  # the table is no longer consulted from here on
    manager = PreviewManager(sender)
    assert register_resume_route(manager, server) is True
    assert any(r.canonical == API_RESUME_ROUTE for r in server.app.router.resources())
    assert register_resume_route(manager, server) is False


def test_registration_on_a_frozen_app_degrades_instead_of_raising(sender, caplog):
    server = FakePromptServer()
    server.routes.post("/prompt")(lambda request: None)
    server.add_routes()
    server.app.router.frozen = True
    manager = PreviewManager(sender)
    assert register_resume_route(manager, server) is False  # logged, not raised


def test_install_is_idempotent_and_returns_the_manager(sender):
    server = FakePromptServer()
    manager = PreviewManager(sender)
    got, registered = preview.install(manager, server)
    assert got is manager and registered is True
    got, registered = preview.install(manager, server)
    assert got is manager and registered is False


def test_package_level_helper_is_lazy_and_import_light():
    import raven_streaming

    server = FakePromptServer()
    manager = PreviewManager(RecordingSender())
    got, registered = raven_streaming.install_preview(manager, server)
    assert got is manager and registered is True


def test_no_route_and_no_server_is_not_an_error(sender):
    manager = PreviewManager(sender)
    if resolve_prompt_server() is None:
        assert register_resume_route(manager, None) is False


# -- resume decisions ------------------------------------------------------


def make_manager_with_session(sender, *, segments=4, client_id="cid", **kwargs):
    manager = PreviewManager(sender, **kwargs)
    session = manager.create_session("7", client_id=client_id, prompt_id="p1")
    open_stream(session)
    for i in range(segments):
        session.send_segment(b"frag-%d" % i, index=i)
    return manager, session


def test_resume_replays_the_gap(sender):
    manager, session = make_manager_with_session(sender)
    before = len(sender.messages)
    result = manager.handle_resume(
        {
            "session_id": session.session_id,
            "node_id": "7",
            "last_seq": 2,
            "client_id": "cid",
            "reason": "segment 3 did not arrive",
        }
    )
    assert result.status == RESUME_REPLAYED
    assert result.http_status == 200
    assert result.body()["resent"] == 3
    assert [b["seq"] for _, b, _ in sender.messages[before:]] == [3, 4, 5]


def test_resume_from_minus_one_replays_everything(sender):
    manager, session = make_manager_with_session(sender, segments=2)
    before = len(sender.messages)
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": -1}
    )
    assert result.status == RESUME_REPLAYED
    assert [b["seq"] for _, b, _ in sender.messages[before:]] == [0, 1, 2, 3]


def test_resume_when_nothing_is_missing(sender):
    manager, session = make_manager_with_session(sender, segments=1)
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": 2}
    )
    assert result.status == RESUME_UP_TO_DATE
    assert result.body()["next_seq"] == 3


def test_resume_on_a_finished_session_says_terminal(sender):
    manager, session = make_manager_with_session(sender, segments=1)
    session.complete()
    manager.retire(session)
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": session.next_seq - 1}
    )
    assert result.status == RESUME_TERMINAL
    assert result.body()["terminal_reason"] == "complete"


def test_resume_beyond_the_replay_window_asks_for_a_resync(sender):
    manager, session = make_manager_with_session(
        sender, segments=20, max_replay_messages=4
    )
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": 0}
    )
    assert result.status == RESUME_RESYNC
    body = result.body()
    assert body["resync"] is True
    assert body["available_from"] == session.replay_span[0]


def test_resume_for_an_expired_session_is_200_not_404(sender):
    clock_now = [1000.0]
    manager = PreviewManager(sender, clock=lambda: clock_now[0], terminal_ttl=10.0)
    session = manager.create_session("7", client_id="cid")
    open_stream(session)
    session.complete()
    manager.retire(session)
    clock_now[0] += 11
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": 0}
    )
    # 404/405/501 would make the client give up on resume for the whole run.
    assert result.http_status == 200
    assert result.status == RESUME_UNKNOWN
    assert result.body()["resync"] is True


def test_resume_for_a_closed_session_reports_expired(sender):
    manager, session = make_manager_with_session(sender)
    session.close()
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "7", "last_seq": 0}
    )
    assert result.status == RESUME_EXPIRED
    assert result.http_status == 200


def test_resume_validates_the_node(sender):
    manager, session = make_manager_with_session(sender)
    result = manager.handle_resume(
        {"session_id": session.session_id, "node_id": "9", "last_seq": 0}
    )
    assert result.status == RESUME_MISMATCH


def test_resume_validates_the_client(sender):
    manager, session = make_manager_with_session(sender, client_id="cid")
    before = len(sender.messages)
    result = manager.handle_resume(
        {
            "session_id": session.session_id,
            "node_id": "7",
            "last_seq": 0,
            "client_id": "another-tab",
        }
    )
    assert result.status == RESUME_MISMATCH
    assert len(sender.messages) == before  # no stream leaked to the other tab


def test_resume_rejects_malformed_bodies(sender):
    manager, session = make_manager_with_session(sender)
    bad_bodies = [
        None,
        [],
        {"node_id": "7", "last_seq": 0},
        {"session_id": "", "node_id": "7", "last_seq": 0},
        {"session_id": session.session_id, "last_seq": 0},
        {"session_id": session.session_id, "node_id": "7"},
        {"session_id": session.session_id, "node_id": "7", "last_seq": "3"},
        {"session_id": session.session_id, "node_id": "7", "last_seq": True},
        {"session_id": session.session_id, "node_id": "7", "last_seq": -2},
        {"session_id": session.session_id, "node_id": "7", "last_seq": 0, "client_id": 5},
        {"session_id": session.session_id, "node_id": "7", "last_seq": 0, "reason": 5},
    ]
    for body in bad_bodies:
        result = manager.handle_resume(body)
        assert result.status == RESUME_BAD_REQUEST, body
        assert result.http_status == 400


def test_resume_for_an_unknown_session(sender):
    manager = PreviewManager(sender)
    result = manager.handle_resume(
        {"session_id": "nope", "node_id": "7", "last_seq": 3}
    )
    assert result.status == RESUME_UNKNOWN
    assert result.http_status == 200


# -- the HTTP handler ------------------------------------------------------


def test_handler_renders_the_manager_decision(sender):
    manager, session = make_manager_with_session(sender)
    handler = make_resume_handler(manager, response_factory=responses)
    out = run(
        handler(
            FakeRequest(
                {"session_id": session.session_id, "node_id": "7", "last_seq": 1}
            )
        )
    )
    assert out["status"] == 200
    assert out["body"]["status"] == RESUME_REPLAYED
    assert out["body"]["v"] == 1


def test_handler_answers_400_for_a_body_that_is_not_json(sender):
    manager = PreviewManager(sender)
    handler = make_resume_handler(manager, response_factory=responses)
    out = run(handler(FakeRequest(None, raise_on_json=True)))
    assert out["status"] == 400
    assert out["body"]["status"] == "bad_request"


def test_handler_never_raises_when_the_manager_does(sender):
    class Boom(PreviewManager):
        def handle_resume(self, payload):
            raise RuntimeError("bug in the preview lane")

    handler = make_resume_handler(Boom(sender), response_factory=responses)
    out = run(handler(FakeRequest({})))
    assert out["status"] == 200
    assert out["body"]["status"] == "error"


def test_registered_handler_is_the_one_that_answers(sender):
    server = FakePromptServer()
    manager, session = make_manager_with_session(sender)
    register_resume_route(manager, server, response_factory=responses)
    server.add_routes()
    handler = next(
        r.handler
        for r in server.app.router.resources()
        if r.canonical == API_RESUME_ROUTE
    )
    out = run(
        handler(
            FakeRequest(
                {"session_id": session.session_id, "node_id": "7", "last_seq": 0}
            )
        )
    )
    assert out["body"]["status"] == RESUME_REPLAYED


# -- the fake must match the pinned upstream -------------------------------


@pytest.fixture(scope="module")
def pinned_server_ast():
    path = find_upstream_comfyui()
    if path is None:
        pytest.skip("no local ComfyUI checkout; set COMFYUI_PATH")
    source = (path / "server.py").read_text(encoding="utf-8")
    return ast.parse(source), path


def _class_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls, name):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found on {cls.name}")


def test_pinned_prompt_server_send_sync_signature(pinned_server_ast):
    """``send_sync(event, data, sid=None)`` -- what PromptServerSender calls."""
    tree, _ = pinned_server_ast
    cls = _class_node(tree, "PromptServer")
    fn = _method(cls, "send_sync")
    assert [a.arg for a in fn.args.args] == ["self", "event", "data", "sid"]
    assert len(fn.args.defaults) == 1
    assert isinstance(fn.args.defaults[0], ast.Constant) and fn.args.defaults[0].value is None
    assert not isinstance(fn, ast.AsyncFunctionDef), "send_sync must be callable from a thread"
    # It only schedules onto the loop: no delivery confirmation exists.
    body = ast.dump(fn)
    assert "call_soon_threadsafe" in body and "put_nowait" in body


def test_pinned_prompt_server_sets_instance_and_route_table(pinned_server_ast):
    tree, _ = pinned_server_ast
    cls = _class_node(tree, "PromptServer")
    init = _method(cls, "__init__")
    dumped = ast.dump(init)
    assert "PromptServer" in dumped and "instance" in dumped
    assert "RouteTableDef" in dumped
    assert "client_id" in dumped


def test_pinned_add_routes_mirrors_every_route_under_api(pinned_server_ast):
    """The client posts to ``/api/...``; that prefix is upstream's doing."""
    tree, path = pinned_server_ast
    cls = _class_node(tree, "PromptServer")
    fn = _method(cls, "add_routes")
    source = (path / "server.py").read_text(encoding="utf-8").splitlines()
    text = "\n".join(source[fn.lineno - 1 : (fn.end_lineno or fn.lineno)])
    assert '"/api" + route.path' in text
    assert "RouteDef" in text  # only non-static routes are mirrored


def test_pinned_send_json_drops_unknown_sids_and_swallows_errors(pinned_server_ast):
    """Why delivery is unconfirmed, and why sessions need a TTL and resume."""
    tree, path = pinned_server_ast
    cls = _class_node(tree, "PromptServer")
    fn = _method(cls, "send_json")
    source = (path / "server.py").read_text(encoding="utf-8").splitlines()
    text = "\n".join(source[fn.lineno - 1 : (fn.end_lineno or fn.lineno)])
    assert "elif sid in self.sockets:" in text  # unknown sid: silently dropped
    assert "send_socket_catch_exception" in text  # socket errors: swallowed


def test_pinned_websocket_handler_has_no_disconnect_hook(pinned_server_ast):
    """No upstream callback fires when a client goes away, so none is faked."""
    _, path = pinned_server_ast
    text = (path / "server.py").read_text(encoding="utf-8")
    assert "self.sockets.pop(sid, None)" in text
    for invented in ("on_disconnect", "on_client_disconnect", "disconnect_handlers"):
        assert invented not in text


def test_pinned_custom_nodes_load_before_add_routes(pinned_server_ast):
    """Registering at import time lands in the table add_routes copies."""
    _, path = pinned_server_ast
    text = (path / "main.py").read_text(encoding="utf-8")
    init_at = text.index("init_extra_nodes(")
    routes_at = text.index("add_routes()")
    assert init_at < routes_at


def test_pinned_commit_is_the_documented_one(pinned_server_ast):
    _, path = pinned_server_ast
    head = Path(path, ".git", "HEAD")
    if not head.exists():
        pytest.skip("checkout has no .git")
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        ref_path = Path(path, ".git", ref.split(" ", 1)[1])
        if not ref_path.exists():
            pytest.skip("packed ref")
        ref = ref_path.read_text(encoding="utf-8").strip()
    assert ref == PINNED_COMMIT


def test_importing_the_preview_lane_stays_light():
    """No torch, no aiohttp, no ComfyUI at import time.

    The whole lane has to import in a bare interpreter, both because the tests
    run there and because a node pack that drags aiohttp/torch into module
    scope breaks metadata scans.
    """
    root = str(Path(__file__).resolve().parents[1])
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        import raven_streaming
        from raven_streaming import preview
        heavy = [m for m in ('torch', 'aiohttp', 'comfy', 'nodes', 'av', 'numpy')
                 if m in sys.modules]
        print(','.join(heavy))
        """
        % root
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"preview import pulled in {out.stdout.strip()}"


def test_our_sender_signature_matches_what_send_sync_expects():
    sig = inspect.signature(PromptServerSender.__call__)
    assert list(sig.parameters) == ["self", "message_type", "body", "client_id"]
