"""ComfyUI transport for the preview lane: ``send_sync`` and the resume route.

Everything that touches ``PromptServer`` lives here, and it is all lazy:
importing this module pulls in nothing but the standard library and
:mod:`raven_streaming.preview_session`. ComfyUI and aiohttp are imported inside
the functions that actually need them, so the whole preview lane -- including
the route handler's decision logic -- is testable in a bare interpreter with a
fake server injected.

Pinned baseline
---------------
Read against ComfyUI ``c67885b14556cf3e4e061862925282d403d09862`` (0.33.0).
The three upstream facts this module is built on:

* ``PromptServer.instance`` is a class attribute assigned in ``__init__``
  (``server.py:217``), so there is exactly one server object to find.
* ``PromptServer.send_sync(event, data, sid=None)`` (``server.py:1392-1394``)
  does ``loop.call_soon_threadsafe(messages.put_nowait, (event, data, sid))``.
  It is **fire and forget**: it returns before anything is serialised, it is
  safe from any thread, and it reports nothing about delivery. The eventual
  ``send_json`` drops the message when ``sid`` is not in ``self.sockets`` and
  swallows socket errors inside ``send_socket_catch_exception``
  (``server.py:1382-1390``). So a send that "succeeded" here may still never
  arrive, which is exactly why the client drives recovery over the resume
  route and why sessions expire on a TTL.
* ``PromptServer.routes`` is an ``aiohttp.web.RouteTableDef``
  (``server.py:262-263``) that ``add_routes()`` (``server.py:1220-1240``)
  copies into the app **with an extra ``/api`` prefix**. ``main.py`` calls
  ``init_extra_nodes()`` (line 531) before ``add_routes()`` (line 545), so a
  node pack registering at import time lands in that table and gets the
  ``/api`` mirror the client calls, for free.

There is no upstream hook for "this client went away" -- the websocket handler
pops the socket in its own ``finally`` (``server.py:325-326``) and fires
nothing. This module therefore does not fake one; see
:meth:`raven_streaming.preview_session.PreviewManager.prune`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from raven_streaming.preview_session import (
    API_RESUME_ROUTE,
    MESSAGE_TYPE,
    RESUME_ROUTE,
    PreviewManager,
    PreviewSendError,
)

__all__ = [
    "MESSAGE_TYPE",
    "RESUME_ROUTE",
    "API_RESUME_ROUTE",
    "resolve_prompt_server",
    "current_client_id",
    "PromptServerSender",
    "make_resume_handler",
    "register_resume_route",
    "is_route_registered",
    "default_manager",
    "install",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Finding the server
# --------------------------------------------------------------------------


def resolve_prompt_server(server: Any = None) -> Optional[Any]:
    """Return the live ``PromptServer``, or ``None`` outside ComfyUI.

    ``server`` may be passed explicitly (tests, or a caller that already has
    it). Otherwise ``server.PromptServer.instance`` is looked up lazily; a
    missing module or a server that has not been constructed yet is not an
    error, it just means there is nobody to send to.
    """
    if server is not None:
        return server
    try:
        import server as comfy_server  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - no ComfyUI on the path
        return None
    return getattr(getattr(comfy_server, "PromptServer", None), "instance", None)


def current_client_id(server: Any = None) -> Optional[str]:
    """The sid of the client that queued the running prompt, when known.

    ``PromptServer.client_id`` (``server.py:265``) is set from the prompt's
    ``extra_data``. Passing it as ``sid`` keeps the stream off every other tab
    (``PROTOCOL.md`` §6.4). ``None`` means "broadcast", which upstream also
    accepts.
    """
    instance = resolve_prompt_server(server)
    if instance is None:
        return None
    sid = getattr(instance, "client_id", None)
    return str(sid) if sid else None


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------


class PromptServerSender:
    """A :data:`~raven_streaming.preview_session.Sender` backed by ``send_sync``.

    Raises :class:`PreviewSendError` when there is no server or the call fails;
    :class:`~raven_streaming.preview_session.PreviewSession` counts that and
    carries on, because a preview failure must not affect sampling.
    """

    __slots__ = ("_server", "_log", "sent", "failures")

    def __init__(self, server: Any = None, *, log: Optional[logging.Logger] = None) -> None:
        # Held only when explicitly supplied; otherwise resolved per call so a
        # server that starts later (or restarts) is picked up without a stale
        # reference living in every session.
        self._server = server
        self._log = log if log is not None else logger
        self.sent = 0
        self.failures = 0

    def __call__(
        self, message_type: str, body: Dict[str, Any], client_id: Optional[str]
    ) -> None:
        instance = resolve_prompt_server(self._server)
        if instance is None:
            self.failures += 1
            raise PreviewSendError("no PromptServer instance; preview message dropped")
        send_sync = getattr(instance, "send_sync", None)
        if not callable(send_sync):
            self.failures += 1
            raise PreviewSendError(
                "PromptServer has no callable send_sync; refusing to guess a transport"
            )
        try:
            # Thread-safe by construction upstream: it only schedules onto the
            # server loop. Nothing here waits for delivery -- there is nothing
            # to wait for.
            send_sync(message_type, body, client_id)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            raise PreviewSendError(f"send_sync failed: {type(exc).__name__}: {exc}") from exc
        self.sent += 1


# --------------------------------------------------------------------------
# The resume route (PROTOCOL.md §4)
# --------------------------------------------------------------------------


def _default_response_factory(body: Dict[str, Any], status: int) -> Any:
    from aiohttp import web  # imported here so this module stays import-light

    return web.json_response(body, status=status)


def make_resume_handler(
    manager: PreviewManager,
    *,
    response_factory: Optional[Callable[[Dict[str, Any], int], Any]] = None,
    log: Optional[logging.Logger] = None,
) -> Callable[[Any], Any]:
    """Build the ``POST /api/raven_streaming/preview/resume`` handler.

    Thin on purpose: parse, delegate to
    :meth:`PreviewManager.handle_resume`, render. The interesting decisions
    (replay / resync / expired / terminal) are in the manager, where they are
    testable without HTTP.

    Note on status codes: the client treats **404, 405 and 501** as "this
    backend has no resume support" and then stops asking for the rest of the
    run (``web/raven_streaming_preview.js``). So every real answer here is a
    200 with a ``status`` field, and a malformed body is a 400.
    """
    factory = response_factory if response_factory is not None else _default_response_factory
    log_ = log if log is not None else logger

    async def resume_handler(request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001 - any body that is not JSON
            log_.debug("raven preview: resume request had no JSON body (%s)", exc)
            return factory({"v": 1, "status": "bad_request", "detail": "invalid JSON body"}, 400)
        try:
            result = manager.handle_resume(payload)
        except Exception as exc:  # noqa: BLE001 - a preview bug is not a 500 for the user
            log_.warning(
                "raven preview: resume handler failed (%s: %s)", type(exc).__name__, exc
            )
            return factory(
                {"v": 1, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}, 200
            )
        return factory(result.body(), result.http_status)

    resume_handler.__name__ = "raven_preview_resume"
    return resume_handler


def _route_table_has(routes: Any, method: str, path: str) -> bool:
    try:
        items = list(routes)
    except TypeError:  # pragma: no cover - not iterable
        return False
    for item in items:
        if (
            getattr(item, "method", None) == method
            and getattr(item, "path", None) == path
        ):
            return True
    return False


def _router_has(app: Any, path: str) -> bool:
    router = getattr(app, "router", None)
    resources = getattr(router, "resources", None)
    if not callable(resources):
        return False
    try:
        for resource in resources():
            if getattr(resource, "canonical", None) == path:
                return True
    except Exception:  # noqa: BLE001 - diagnostics only
        return False
    return False


def is_route_registered(server: Any = None) -> bool:
    """True when the resume route is already on this server, either way it
    can get there (the route table, or the running app's router)."""
    instance = resolve_prompt_server(server)
    if instance is None:
        return False
    routes = getattr(instance, "routes", None)
    if routes is not None and _route_table_has(routes, "POST", RESUME_ROUTE):
        return True
    app = getattr(instance, "app", None)
    return app is not None and _router_has(app, API_RESUME_ROUTE)


def register_resume_route(
    manager: PreviewManager,
    server: Any = None,
    *,
    routes: Any = None,
    response_factory: Optional[Callable[[Dict[str, Any], int], Any]] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Register the resume route. Idempotent; never raises.

    Returns True when this call is what put the route on the server.

    Two placements, chosen by inspecting the server rather than by guessing:

    * **Before** ``PromptServer.add_routes()`` (the normal case -- ``main.py``
      loads custom nodes first): append to ``PromptServer.routes``, the
      ``RouteTableDef``. ``add_routes`` then installs it both bare and under
      ``/api``, which is the path the client calls.
    * **After** ``add_routes()`` has already run (a late import, a reload):
      the table is no longer consulted, so add straight to ``app.router`` under
      the ``/api`` path. If the app is already frozen aiohttp refuses, and that
      refusal is logged, not raised -- the route is optional by protocol, and
      the client degrades to "waiting for the backend to resend".
    """
    log_ = log if log is not None else logger
    instance = resolve_prompt_server(server)
    table = routes if routes is not None else getattr(instance, "routes", None)

    if table is None and instance is None:
        log_.debug("raven preview: no PromptServer; resume route not registered")
        return False

    if _route_table_has(table, "POST", RESUME_ROUTE):
        return False
    app = getattr(instance, "app", None)
    if app is not None and _router_has(app, API_RESUME_ROUTE):
        return False

    handler = make_resume_handler(manager, response_factory=response_factory, log=log_)

    # ``/api/prompt`` only exists once add_routes() has copied the table into
    # the app, so it is a precise "am I late?" signal on the pinned server.
    already_installed = app is not None and (
        _router_has(app, "/api/prompt") or _router_has(app, "/prompt")
    )

    if table is not None and not already_installed:
        try:
            table.post(RESUME_ROUTE)(handler)
        except Exception as exc:  # noqa: BLE001
            log_.warning(
                "raven preview: could not add the resume route to PromptServer.routes "
                "(%s: %s); the preview stays usable without resume",
                type(exc).__name__,
                exc,
            )
            return False
        log_.debug("raven preview: resume route queued as POST %s", API_RESUME_ROUTE)
        return True

    router = getattr(app, "router", None)
    add_route = getattr(router, "add_route", None)
    if not callable(add_route):
        log_.warning("raven preview: no way to register the resume route on this server")
        return False
    try:
        add_route("POST", API_RESUME_ROUTE, handler)
    except Exception as exc:  # noqa: BLE001 - frozen app, duplicate, anything
        log_.warning(
            "raven preview: could not register %s late (%s: %s); the preview stays "
            "usable without resume",
            API_RESUME_ROUTE,
            type(exc).__name__,
            exc,
        )
        return False
    log_.debug("raven preview: resume route registered late as POST %s", API_RESUME_ROUTE)
    return True


# --------------------------------------------------------------------------
# Process-wide wiring
# --------------------------------------------------------------------------

_DEFAULT_MANAGER: Optional[PreviewManager] = None


def default_manager() -> PreviewManager:
    """The manager a node uses when it is not given one. Created once."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = PreviewManager(sender=PromptServerSender(), log=logger)
    return _DEFAULT_MANAGER


def install(
    manager: Optional[PreviewManager] = None,
    server: Any = None,
    *,
    response_factory: Optional[Callable[[Dict[str, Any], int], Any]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[PreviewManager, bool]:
    """Wire the preview lane into a ComfyUI process. Idempotent.

    Returns ``(manager, registered_now)``. Safe to call from a node module's
    import, from ``__init__``, or lazily on first execution: repeat calls do
    nothing beyond returning the same manager.
    """
    mgr = manager if manager is not None else default_manager()
    registered = register_resume_route(
        mgr, server, response_factory=response_factory, log=log
    )
    return mgr, registered
