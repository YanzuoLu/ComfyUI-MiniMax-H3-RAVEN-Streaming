"""ComfyUI V1 custom-node entry point for MiniMax H3 RAVEN Streaming."""

from __future__ import annotations

import logging
import sys

# ComfyUI loads a custom-node directory under a generated package name. Because
# this repository name contains hyphens, the bundled runtime is initially known
# only as ``<generated-name>.raven_streaming``. Its modules intentionally use
# the stable public name ``raven_streaming`` internally, so publish that alias
# before importing ``nodes``. A normal Python/package installation already has
# the top-level name and takes the fallback branch.
try:
    from . import raven_streaming as _runtime
except (ImportError, ValueError):  # plain ``__init__.py`` import, no package parent
    import raven_streaming as _runtime
else:
    sys.modules.setdefault("raven_streaming", _runtime)

install_preview = _runtime.install_preview
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
