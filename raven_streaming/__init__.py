"""MiniMax H3 RAVEN Streaming — runtime package and ComfyUI entry point.

Streaming text-to-video-and-audio (T2VA) nodes for MiniMax H3, built on
ComfyUI's official H3 support and a mandatory RAVEN LoRA.

What ComfyUI reads here
-----------------------
``WEB_DIRECTORY`` plus the two V1 mappings (``nodes.py:2285-2298`` at the pinned
commit). ``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` are served
by a module-level ``__getattr__`` (PEP 562), which is what keeps two properties
that would otherwise be in conflict:

* **import-light**: importing ``raven_streaming`` pulls in nothing but the
  standard library, so metadata reads and the pure unit tests still run in a
  bare interpreter with no torch and no ComfyUI;
* **registered anyway**: ``hasattr(module, "NODE_CLASS_MAPPINGS")`` triggers the
  lazy import, so by the time upstream reads the mapping the node classes exist.
  ``install_preview()`` runs on that same first access -- exactly once, and
  never during a bare import.

There is deliberately no ``comfy_entrypoint`` (V3 extension) beside this. V1 is
checked first and returns, so shipping both would make the live schema depend on
which branch upstream happens to take.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "__version__",
    "DISPLAY_NAME",
    "PACKAGE_NAME",
    "REPOSITORY_URL",
    "WEB_DIRECTORY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "install_preview",
]

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from raven_streaming.preview_session import PreviewManager

__version__ = "0.1.0"

PACKAGE_NAME = "comfyui-minimax-h3-raven-streaming"
DISPLAY_NAME = "MiniMax H3 RAVEN Streaming"
REPOSITORY_URL = "https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming"

#: Served at ``/extensions/...`` by ComfyUI; holds the in-node preview client.
WEB_DIRECTORY = "./web"

_LOG = logging.getLogger(__name__)

#: Names resolved by importing :mod:`raven_streaming.nodes` on first access.
_LAZY_NODE_ATTRIBUTES = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")

_preview_installed = False


def install_preview(manager: "PreviewManager | None" = None, server: Any = None, **kwargs: Any):
    """Wire the in-node preview lane into a running ComfyUI. Idempotent.

    Lazy on purpose: the import happens inside the call, so this package stays
    import-light and a process that never previews never pays for it. Returns
    ``(manager, registered_now)``; see :func:`raven_streaming.preview.install`.
    """
    from raven_streaming.preview import install

    return install(manager, server, **kwargs)


def _install_preview_once() -> None:
    """Register the preview route at node-registration time. Never raises.

    Upstream loads custom nodes *before* ``PromptServer.add_routes()``, so this
    is the moment at which appending to the route table still works. A failure
    here costs the resume route, not the nodes.
    """
    global _preview_installed
    if _preview_installed:
        return
    _preview_installed = True
    try:
        install_preview()
    except Exception as exc:  # noqa: BLE001 - previewing is optional, the nodes are not
        _LOG.warning(
            "raven preview: could not install the preview lane (%s: %s); the nodes "
            "still work, without an in-node preview",
            type(exc).__name__,
            exc,
        )


def __getattr__(name: str) -> Any:
    if name in _LAZY_NODE_ATTRIBUTES:
        from raven_streaming import nodes

        mappings: Dict[str, Any] = {
            "NODE_CLASS_MAPPINGS": nodes.NODE_CLASS_MAPPINGS,
            "NODE_DISPLAY_NAME_MAPPINGS": nodes.NODE_DISPLAY_NAME_MAPPINGS,
        }
        globals().update(mappings)
        _install_preview_once()
        return mappings[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(list(globals()) + list(_LAZY_NODE_ATTRIBUTES)))
