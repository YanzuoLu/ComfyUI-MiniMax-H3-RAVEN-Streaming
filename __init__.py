"""ComfyUI V1 custom-node entry point for MiniMax H3 RAVEN Streaming."""

from __future__ import annotations

import logging

try:  # Normal ComfyUI package loading.
    from .raven_streaming import install_preview
    from .raven_streaming.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # Pytest/importlib may load a hyphenated repo root as plain __init__.
    from raven_streaming import install_preview
    from raven_streaming.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    install_preview()
except Exception as exc:  # Preview failure must never hide the nodes.
    logging.getLogger(__name__).warning(
        "RAVEN preview route was not installed (%s: %s); generation remains available",
        type(exc).__name__,
        exc,
    )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
