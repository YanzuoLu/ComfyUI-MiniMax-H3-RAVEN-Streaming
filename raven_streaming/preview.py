"""In-node preview lane: the backend half of ``web/PROTOCOL.md`` (v1).

This is the facade the sampler node imports. The implementation is split in
two, along the only seam that matters -- whether ComfyUI is involved:

* :mod:`raven_streaming.preview_session` -- sessions, ordering, replay,
  lifetime, the media sink. Standard library only, no ComfyUI, no torch.
* :mod:`raven_streaming.preview_server` -- ``PromptServer.send_sync`` and the
  optional resume route. Imports ComfyUI and aiohttp lazily, inside the calls
  that need them.

What the node does
------------------

.. code-block:: python

    from raven_streaming import preview

    manager, _ = preview.install()          # idempotent; registers the route

    with manager.session(
        node_id=unique_id,
        client_id=preview.current_client_id(),
        prompt_id=prompt_id,
    ) as session:
        sink = preview.PreviewMediaSink(session)
        sink.on_open('video/mp4; codecs="avc1.640028,mp4a.40.2"', width=848, height=480, fps=24)
        sink.on_status("model_loading")
        ...
        sink.pump_muxer(muxer)              # bytes out of the fMP4 muxer
        ...
        # the context manager sends the terminal `end` and cleans up on every
        # exit path, including cancellation and exceptions

Four properties this lane is built to have, all of them testable:

1. **A preview failure never touches sampling.** The sink and
   :meth:`PreviewSession.isolated` swallow and log; the context manager still
   ends the stream and still releases resources when the node raises.
2. **A session never pins GPU memory.** It holds strings, ints and base64
   text. Resources are held as zero-argument finalizers that run exactly once.
3. **Ordering is one rule.** A single ``seq`` covers ``open``, ``init``,
   ``segment``, ``status`` and ``end``, so gap detection, de-duplication and
   replay need no per-event reasoning.
4. **Cancellation is observed, never caused.** Nothing here calls ComfyUI's
   interrupt. The sampler's own ``cancel_check`` raises, and this lane's only
   job is to report ``end: cancelled`` and clean up.
"""

from __future__ import annotations

from raven_streaming.preview_server import (
    PromptServerSender,
    current_client_id,
    default_manager,
    install,
    is_route_registered,
    make_resume_handler,
    register_resume_route,
    resolve_prompt_server,
)
from raven_streaming.preview_session import (
    API_RESUME_ROUTE,
    BACKEND_PHASES,
    DEFAULT_ACTIVE_IDLE_TTL,
    DEFAULT_REPLAY_BYTES,
    DEFAULT_REPLAY_MESSAGES,
    DEFAULT_TERMINAL_TTL,
    END_REASONS,
    EVENT_KINDS,
    MAX_RAW_PAYLOAD_BYTES,
    MESSAGE_TYPE,
    PAYLOAD_ENCODING,
    PROTOCOL_VERSION,
    RESUME_BAD_REQUEST,
    RESUME_EXPIRED,
    RESUME_MISMATCH,
    RESUME_REPLAYED,
    RESUME_RESYNC,
    RESUME_ROUTE,
    RESUME_TERMINAL,
    RESUME_UNKNOWN,
    RESUME_UP_TO_DATE,
    MediaSink,
    NullSender,
    PreviewError,
    PreviewManager,
    PreviewMediaSink,
    PreviewPayloadTooLarge,
    PreviewSendError,
    PreviewSession,
    PreviewStateError,
    RecordingSender,
    ReplayReport,
    ResumeResult,
    SessionState,
    base64_cost,
)

__all__ = [
    # protocol constants
    "PROTOCOL_VERSION",
    "MESSAGE_TYPE",
    "RESUME_ROUTE",
    "API_RESUME_ROUTE",
    "EVENT_KINDS",
    "BACKEND_PHASES",
    "END_REASONS",
    "PAYLOAD_ENCODING",
    "MAX_RAW_PAYLOAD_BYTES",
    "DEFAULT_REPLAY_MESSAGES",
    "DEFAULT_REPLAY_BYTES",
    "DEFAULT_TERMINAL_TTL",
    "DEFAULT_ACTIVE_IDLE_TTL",
    "base64_cost",
    # resume statuses
    "RESUME_REPLAYED",
    "RESUME_UP_TO_DATE",
    "RESUME_RESYNC",
    "RESUME_EXPIRED",
    "RESUME_TERMINAL",
    "RESUME_UNKNOWN",
    "RESUME_MISMATCH",
    "RESUME_BAD_REQUEST",
    # errors
    "PreviewError",
    "PreviewStateError",
    "PreviewPayloadTooLarge",
    "PreviewSendError",
    # core
    "SessionState",
    "PreviewSession",
    "PreviewManager",
    "ReplayReport",
    "ResumeResult",
    "MediaSink",
    "PreviewMediaSink",
    "NullSender",
    "RecordingSender",
    # ComfyUI transport
    "PromptServerSender",
    "resolve_prompt_server",
    "current_client_id",
    "make_resume_handler",
    "register_resume_route",
    "is_route_registered",
    "default_manager",
    "install",
]
