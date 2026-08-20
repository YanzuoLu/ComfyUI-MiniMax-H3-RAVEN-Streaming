"""ComfyUI V1 custom-node entry point for MiniMax H3 RAVEN Streaming."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ComfyUI executes a custom-node root under a generated package name. The
# repository directory contains hyphens, so it cannot be the stable import name
# used by the runtime's absolute imports. Put this directory on ``sys.path`` and
# import exactly one top-level ``raven_streaming`` package. Aliasing a relative
# package object is not equivalent: its ``__spec__.name`` keeps the generated
# name and can make Python load submodules twice, producing distinct class
# identities from the same source file.
_PLUGIN_ROOT = str(Path(__file__).resolve().parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

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
