"""Server side of the ``raven.preview`` protocol: sessions, replay, lifetime.

This module implements the backend half of ``web/PROTOCOL.md`` (v1) and nothing
else. It owns:

* :class:`PreviewSession` -- one sampler execution's stream. Strictly ordered
  ``seq``, one state machine, one terminal message, a bounded replay buffer.
* :class:`PreviewManager` -- the set of live and recently-finished sessions,
  their TTLs, idempotent cleanup, and the ``resume`` decision logic.
* the ``sink`` seam (:class:`PreviewMediaSink`) that the fMP4 muxer and the
  streaming decoders push **bytes and control only** through.

Import weight
-------------
Standard library only. No torch, no ComfyUI, no aiohttp, no
:mod:`raven_streaming.media` import at module scope -- the media objects are
duck-typed at the sink boundary. That keeps the whole preview lane testable in
a bare interpreter, which is also what makes the tests in
``tests/test_preview_session.py`` runnable without a ComfyUI checkout.

Three rules this module exists to enforce
-----------------------------------------
1. **A preview failure must not affect sampling** (``PROTOCOL.md`` §6.5). The
   strict API raises loudly, but every call the node makes goes through an
   isolating wrapper (:meth:`PreviewSession.isolated`, :class:`PreviewMediaSink`)
   that logs and swallows. Nothing here ever raises into the sampler loop, and
   nothing here ever interrupts it (see :mod:`~raven_streaming.preview_server`
   and the cancellation note on :meth:`PreviewManager.session`).
2. **No references that could pin GPU memory.** A session holds strings, ints
   and ``bytes``-derived base64 text. It never holds a tensor, a model, a
   patcher, a decoder or a muxer; resource ownership is expressed as opaque
   zero-argument finalizers that run exactly once.
3. **The wire format is not negotiable here.** ``PROTOCOL.md`` v1 has no
   fragment-splitting field, so an oversized media payload is a *loud* backend
   error (:class:`PreviewPayloadTooLarge`), never a silently truncated or
   privately re-framed message. Fixing it means muxing smaller fragments, or a
   protocol v2 with an explicit part field -- not a change in this file.
"""

from __future__ import annotations

import base64
import collections.abc as _abc
import contextlib
import itertools
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
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
    "base64_cost",
    # errors
    "PreviewError",
    "PreviewStateError",
    "PreviewPayloadTooLarge",
    "PreviewSendError",
    # sending
    "Sender",
    "NullSender",
    "RecordingSender",
    # session
    "SessionState",
    "ReplayReport",
    "PreviewSession",
    # manager
    "ResumeResult",
    "PreviewManager",
    # sinks
    "MediaSink",
    "PreviewMediaSink",
    "CANCELLATION_EXCEPTION_NAMES",
]


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Protocol constants (web/PROTOCOL.md §2, §3; mirrored in web/lib/protocol.js)
# --------------------------------------------------------------------------

#: ``v`` in every envelope. A client that speaks a different version drops us.
PROTOCOL_VERSION = 1

#: The single websocket JSON message type. Binary frames are unusable on the
#: pinned frontend (``PROTOCOL.md`` §1.1): it decodes exactly four integer event
#: ids and throws on anything else, and the throw is swallowed, so a custom
#: binary event is silently dropped. There is no "raw frame" extension hook to
#: work around it. Hence: JSON only, base64 payloads, no binary listener.
MESSAGE_TYPE = "raven.preview"

#: Route as registered on ``PromptServer.routes``; ComfyUI mirrors every route
#: under ``/api`` (``server.py:1233-1241``), which is what the client calls.
RESUME_ROUTE = "/raven_streaming/preview/resume"
API_RESUME_ROUTE = "/api" + RESUME_ROUTE

EVENT_KINDS: Tuple[str, ...] = ("open", "init", "segment", "status", "end")
BACKEND_PHASES: Tuple[str, ...] = ("waiting", "model_loading", "sampling", "finalizing")
END_REASONS: Tuple[str, ...] = ("complete", "cancelled", "error")

#: The only payload encoding v1 defines.
PAYLOAD_ENCODING = "base64"

#: ``PROTOCOL.md`` §2 "Size guidance": keep one message under ~256 KB of raw
#: media (~342 KB base64). aiohttp does not cap outbound frames, but one huge
#: message blocks the socket for every other node's progress traffic while it is
#: written. v1 has **no** part/fragment-index field to split a payload across
#: messages, so exceeding this is a backend bug to fix upstream of here (mux
#: smaller fragments), not something this module may paper over.
MAX_RAW_PAYLOAD_BYTES = 256 * 1024

#: Default bounds on a session's replay buffer. The client holds at most 64
#: out-of-order messages before it declares overflow (``web/lib/sequencer.js``),
#: so retaining a few hundred is generous; the byte cap is what actually keeps
#: a long run from growing without bound.
DEFAULT_REPLAY_MESSAGES = 256
DEFAULT_REPLAY_BYTES = 8 * 1024 * 1024

#: How long a finished session stays resumable after its terminal message.
DEFAULT_TERMINAL_TTL = 120.0

#: How long an *active* session may go without any activity before
#: :meth:`PreviewManager.prune` treats it as abandoned. Sessions are normally
#: retired by their context manager; this only catches a node that died in a way
#: that skipped its ``finally``.
DEFAULT_ACTIVE_IDLE_TTL = 1800.0


def base64_cost(n: int) -> int:
    """Bytes on the wire for ``n`` raw bytes of base64 payload (no envelope)."""
    return ((n + 2) // 3) * 4


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class PreviewError(RuntimeError):
    """Base class for every preview-lane failure."""


class PreviewStateError(PreviewError):
    """A message was offered that the session's state machine forbids."""


class PreviewPayloadTooLarge(PreviewError, ValueError):
    """A media payload exceeds what protocol v1 can carry in one message.

    Raised loudly on purpose. v1 defines no part field, so the backend cannot
    split the payload without inventing wire format the client does not parse.
    """

    def __init__(self, size: int, limit: int, what: str) -> None:
        super().__init__(
            f"{what} payload is {size} bytes ({base64_cost(size)} base64), over the "
            f"protocol v1 per-message limit of {limit} raw bytes. Protocol v1 has no "
            f"fragment-part field, so this cannot be split on the wire: mux smaller "
            f"fMP4 fragments, or extend the protocol (client included) with an "
            f"explicit part field first."
        )
        self.size = size
        self.limit = limit
        self.what = what


class PreviewSendError(PreviewError):
    """The transport refused a message. Recorded, never propagated to sampling."""


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------

#: A sender is ``fn(message_type, body, client_id) -> None``. It must not block
#: for long: the real one is ``PromptServer.send_sync``, which only schedules
#: onto the server loop. Failures raise; the session isolates them.
Sender = Callable[[str, Dict[str, Any], Optional[str]], None]


class NullSender:
    """Drops everything. The default when no ComfyUI server is around."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, message_type: str, body: Dict[str, Any], client_id: Optional[str]) -> None:
        self.count += 1


class RecordingSender:
    """Keeps every message. For tests and for offline inspection."""

    __slots__ = ("messages", "_lock")

    def __init__(self) -> None:
        self.messages: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
        self._lock = threading.Lock()

    def __call__(self, message_type: str, body: Dict[str, Any], client_id: Optional[str]) -> None:
        with self._lock:
            self.messages.append((message_type, body, client_id))

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [body for _, body, _ in self.messages]

    def events(self) -> List[str]:
        return [body["event"] for body in self.bodies]

    def seqs(self) -> List[int]:
        return [body["seq"] for body in self.bodies]


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class SessionState(str, Enum):
    """States of one preview stream.

    ``IDLE -> OPENED -> STREAMING -> ENDED -> CLOSED``, with ``ENDED`` reachable
    from ``OPENED`` too (a run that fails before its init segment).
    """

    IDLE = "idle"          # created, no ``open`` sent yet
    OPENED = "opened"      # ``open`` sent, no ``init`` yet
    STREAMING = "streaming"  # ``init`` sent, segments allowed
    ENDED = "ended"        # terminal ``end`` sent; replay still served
    CLOSED = "closed"      # resources released, replay dropped


_TERMINAL_STATES = (SessionState.ENDED, SessionState.CLOSED)


@dataclass(frozen=True)
class ReplayReport:
    """Outcome of a replay attempt."""

    sent: int
    first: Optional[int]
    last: Optional[int]
    failed: int = 0

    @property
    def empty(self) -> bool:
        return self.sent == 0


@dataclass
class _Retained:
    """One message kept for replay: JSON-ready body, no bytes objects."""

    seq: int
    body: Dict[str, Any]
    cost: int


class PreviewSession:
    """One sampler execution's preview stream.

    Thread and async safe: every public method takes one re-entrant lock and
    does only non-blocking work under it, so ordering on the wire matches the
    ``seq`` order even when the sampler emits from a worker thread while an
    aiohttp handler replays from the event loop. The lock is held across the
    send precisely so those two cannot interleave.

    Isolation: identity is ``(session_id, node_id, client_id)`` and a message is
    addressed to ``client_id`` only, so a second tab, a second node, or a stale
    run from a cancelled execution can never append into this stream
    (``PROTOCOL.md`` §5).
    """

    __slots__ = (
        "session_id",
        "node_id",
        "client_id",
        "prompt_id",
        "_sender",
        "_log",
        "_clock",
        "_wall",
        "_lock",
        "_seq",
        "seq_start",
        "_state",
        "_terminal_reason",
        "_terminal_at",
        "_last_activity",
        "_replay",
        "_replay_bytes",
        "_replay_low",
        "max_replay_messages",
        "max_replay_bytes",
        "max_payload_bytes",
        "messages_sent",
        "segments_sent",
        "send_failures",
        "rejected",
        "replayed_messages",
        "_finalizers",
        "_finalized",
        "__weakref__",
    )

    def __init__(
        self,
        node_id: Any,
        *,
        session_id: Optional[str] = None,
        client_id: Optional[str] = None,
        prompt_id: Optional[Any] = None,
        sender: Optional[Sender] = None,
        seq_start: int = 0,
        max_replay_messages: int = DEFAULT_REPLAY_MESSAGES,
        max_replay_bytes: int = DEFAULT_REPLAY_BYTES,
        max_payload_bytes: int = MAX_RAW_PAYLOAD_BYTES,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        log: Optional[logging.Logger] = None,
    ) -> None:
        if node_id is None:
            raise ValueError("node_id is required: the client addresses one node by id")
        if seq_start < 0:
            raise ValueError("seq_start must be >= 0")
        self.session_id = str(session_id) if session_id else uuid.uuid4().hex
        # ``PROTOCOL.md`` §3: node_id is the hidden ``unique_id``, as a string.
        self.node_id = str(node_id)
        self.client_id = str(client_id) if client_id is not None else None
        self.prompt_id = str(prompt_id) if prompt_id is not None else None

        self._sender: Sender = sender if sender is not None else NullSender()
        self._log = log if log is not None else logger
        self._clock = clock
        self._wall = wall_clock
        self._lock = threading.RLock()

        self.seq_start = int(seq_start)
        self._seq = int(seq_start)
        self._state = SessionState.IDLE
        self._terminal_reason: Optional[str] = None
        self._terminal_at: Optional[float] = None
        self._last_activity = clock()

        self._replay: Deque[_Retained] = deque()
        self._replay_bytes = 0
        self._replay_low: Optional[int] = None
        self.max_replay_messages = int(max_replay_messages)
        self.max_replay_bytes = int(max_replay_bytes)
        self.max_payload_bytes = int(max_payload_bytes)

        self.messages_sent = 0
        self.segments_sent = 0
        self.send_failures = 0
        self.rejected = 0
        self.replayed_messages = 0

        self._finalizers: List[Tuple[str, Callable[[], Any]]] = []
        self._finalized = False

    # -- introspection ----------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<PreviewSession {self.session_id[:8]} node={self.node_id} "
            f"state={self._state.value} seq={self._seq}>"
        )

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def next_seq(self) -> int:
        """The seq the next message will carry."""
        with self._lock:
            return self._seq

    @property
    def last_seq(self) -> Optional[int]:
        with self._lock:
            return self._seq - 1 if self._seq > self.seq_start else None

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in _TERMINAL_STATES

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._state is SessionState.CLOSED

    @property
    def terminal_reason(self) -> Optional[str]:
        with self._lock:
            return self._terminal_reason

    @property
    def terminal_at(self) -> Optional[float]:
        with self._lock:
            return self._terminal_at

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    @property
    def replay_span(self) -> Optional[Tuple[int, int]]:
        """``(lowest, highest)`` seq still replayable, or ``None`` when empty."""
        with self._lock:
            if not self._replay:
                return None
            return (self._replay[0].seq, self._replay[-1].seq)

    @property
    def replay_bytes(self) -> int:
        with self._lock:
            return self._replay_bytes

    def snapshot(self) -> Dict[str, Any]:
        """A JSON-safe view for diagnostics and for the resume response body."""
        with self._lock:
            span = self.replay_span
            return {
                "session_id": self.session_id,
                "node_id": self.node_id,
                "client_id": self.client_id,
                "prompt_id": self.prompt_id,
                "state": self._state.value,
                "seq_start": self.seq_start,
                "next_seq": self._seq,
                "terminal_reason": self._terminal_reason,
                "messages_sent": self.messages_sent,
                "segments_sent": self.segments_sent,
                "send_failures": self.send_failures,
                "rejected": self.rejected,
                "replayed_messages": self.replayed_messages,
                "replay_messages": len(self._replay),
                "replay_bytes": self._replay_bytes,
                "replay_from": None if span is None else span[0],
                "replay_to": None if span is None else span[1],
            }

    # -- state machine ----------------------------------------------------

    def _reject(self, message: str) -> None:
        self.rejected += 1
        raise PreviewStateError(message)

    def _check(self, event: str) -> None:
        """Validate ``event`` against the current state. Caller holds the lock."""
        state = self._state
        if state is SessionState.CLOSED:
            self._reject(
                f"session {self.session_id} is closed; {event!r} rejected"
            )
        if state is SessionState.ENDED:
            self._reject(
                f"session {self.session_id} already ended "
                f"({self._terminal_reason}); {event!r} rejected"
            )
        if event == "open":
            if state is not SessionState.IDLE:
                self._reject(
                    f"session {self.session_id} already sent open (state {state.value})"
                )
            return
        if state is SessionState.IDLE:
            self._reject(
                f"session {self.session_id}: {event!r} before open"
            )
        if event == "init":
            if state is not SessionState.OPENED:
                self._reject(
                    f"session {self.session_id}: init may be sent exactly once, "
                    f"directly after open (state {state.value})"
                )
            return
        if event == "segment":
            if state is not SessionState.STREAMING:
                self._reject(
                    f"session {self.session_id}: segment before init (state {state.value})"
                )
            return
        # 'status' and 'end' are legal in OPENED and STREAMING.

    def _advance(self, event: str, reason: Optional[str]) -> None:
        """Apply the state transition. Caller holds the lock."""
        if event == "open":
            self._state = SessionState.OPENED
        elif event == "init":
            self._state = SessionState.STREAMING
        elif event == "end":
            self._state = SessionState.ENDED
            self._terminal_reason = reason
            self._terminal_at = self._clock()

    # -- sending ----------------------------------------------------------

    def _emit(self, event: str, body: Dict[str, Any], *, reason: Optional[str] = None) -> int:
        """Stamp, send, retain and transition. Returns the seq used."""
        with self._lock:
            self._check(event)
            seq = self._seq
            envelope: Dict[str, Any] = {
                "v": PROTOCOL_VERSION,
                "event": event,
                "session_id": self.session_id,
                "node_id": self.node_id,
                "seq": seq,
                "t": round(self._wall(), 6),
            }
            if self.prompt_id is not None:
                envelope["prompt_id"] = self.prompt_id
            envelope.update(body)

            # The seq is consumed and the transition applied whether or not the
            # send succeeds. A failed send is a *delivery* problem the client
            # recovers from with a resume; renumbering around it would break the
            # one invariant the client relies on.
            self._seq = seq + 1
            self._advance(event, reason)
            if event == "segment":
                self.segments_sent += 1
            self._retain(seq, envelope)
            self._last_activity = self._clock()

            try:
                self._sender(MESSAGE_TYPE, envelope, self.client_id)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                self.send_failures += 1
                self._log.warning(
                    "raven preview: send failed for session %s seq %d (%s: %s); "
                    "message retained for resume",
                    self.session_id,
                    seq,
                    type(exc).__name__,
                    exc,
                )
            else:
                self.messages_sent += 1
            return seq

    def send_open(
        self,
        mime: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        audio: Optional[Mapping[str, Any]] = None,
        duration_hint: Optional[float] = None,
        resync: bool = False,
        label: Optional[str] = None,
    ) -> int:
        """``PROTOCOL.md`` §3.1. Must be the first message of the session."""
        mime = str(mime)
        # The client rejects a mime without codecs, so reject it here rather
        # than shipping a stream that cannot possibly be set up.
        low = mime.lower().strip()
        if not low.startswith("video/mp4") or ";" not in mime or "codecs" not in low:
            raise ValueError(
                "open.mime must be a video/mp4 type carrying a codecs parameter, "
                'e.g. video/mp4; codecs="avc1.640028,mp4a.40.2"'
            )
        body: Dict[str, Any] = {"mime": mime}
        if width is not None:
            body["width"] = int(width)
        if height is not None:
            body["height"] = int(height)
        if fps is not None:
            body["fps"] = float(fps)
        if audio is not None:
            body["audio"] = {k: v for k, v in dict(audio).items()}
        if duration_hint is not None:
            body["duration_hint"] = float(duration_hint)
        if resync:
            body["resync"] = True
        if label is not None:
            body["label"] = str(label)
        return self._emit("open", body)

    def send_status(
        self,
        phase: str,
        *,
        message: Optional[str] = None,
        progress: Optional[Any] = None,
    ) -> int:
        """``PROTOCOL.md`` §3.4. ``buffering``/``live``/``reconnecting`` are
        client-derived and must never be sent from here."""
        phase = str(phase)
        if phase not in BACKEND_PHASES:
            raise ValueError(
                f"unknown status phase {phase!r}; the backend may send one of "
                f"{', '.join(BACKEND_PHASES)}"
            )
        body: Dict[str, Any] = {"phase": phase}
        if message is not None:
            body["message"] = str(message)
        if progress is not None:
            body["progress"] = _normalise_progress(progress)
        return self._emit("status", body)

    def send_init(self, data: bytes) -> int:
        """``PROTOCOL.md`` §3.2. ``ftyp`` + ``moov``, exactly once per open."""
        payload = self._encode(data, "init")
        body = {"encoding": PAYLOAD_ENCODING, "bytes": payload[1], "data": payload[0]}
        return self._emit("init", body)

    def send_segment(
        self,
        data: bytes,
        *,
        index: Optional[int] = None,
        keyframe: bool = False,
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> int:
        """``PROTOCOL.md`` §3.3. One ``moof`` + ``mdat`` fragment, decode order."""
        payload = self._encode(data, "segment")
        body: Dict[str, Any] = {
            "encoding": PAYLOAD_ENCODING,
            "bytes": payload[1],
            "data": payload[0],
        }
        if index is not None:
            body["index"] = int(index)
        if keyframe:
            body["keyframe"] = True
        if start is not None:
            body["start"] = float(start)
        if duration is not None:
            body["duration"] = float(duration)
        return self._emit("segment", body)

    def send_end(
        self,
        reason: str,
        *,
        message: Optional[str] = None,
        segments: Optional[int] = None,
    ) -> int:
        """``PROTOCOL.md`` §3.5. The single terminal message."""
        reason = str(reason)
        if reason not in END_REASONS:
            raise ValueError(
                f"unknown end reason {reason!r}; expected one of {', '.join(END_REASONS)}"
            )
        with self._lock:
            body: Dict[str, Any] = {
                "reason": reason,
                "segments": int(segments) if segments is not None else self.segments_sent,
            }
            if message is not None:
                body["message"] = str(message)
            return self._emit("end", body, reason=reason)

    # Convenience terminals -------------------------------------------------

    def complete(self, *, message: Optional[str] = None) -> int:
        return self.send_end("complete", message=message)

    def cancelled(self, *, message: Optional[str] = None) -> int:
        """Terminal for a cancelled run.

        This is a *report*, not an action: nothing here calls ComfyUI's
        interrupt machinery. Real cancellation is detected by the sampler's own
        ``cancel_check`` and the node then tells the preview about it.
        """
        return self.send_end("cancelled", message=message)

    def failed(self, message: Optional[str] = None) -> int:
        return self.send_end("error", message=message)

    def _encode(self, data: Any, what: str) -> Tuple[str, int]:
        if isinstance(data, (bytes, bytearray, memoryview)):
            raw = bytes(data)
        else:
            raise TypeError(f"{what} payload must be bytes, got {type(data).__name__}")
        if not raw:
            raise ValueError(f"{what} payload is empty")
        if len(raw) > self.max_payload_bytes:
            raise PreviewPayloadTooLarge(len(raw), self.max_payload_bytes, what)
        return base64.b64encode(raw).decode("ascii"), len(raw)

    # -- replay -----------------------------------------------------------

    def _retain(self, seq: int, envelope: Dict[str, Any]) -> None:
        """Store one message for replay, then trim to the bounds."""
        data = envelope.get("data")
        cost = len(data) if isinstance(data, str) else 0
        cost += 256  # envelope text, roughly; keeps control messages accounted
        self._replay.append(_Retained(seq=seq, body=envelope, cost=cost))
        self._replay_bytes += cost
        if self._replay_low is None:
            self._replay_low = seq
        while self._replay and (
            len(self._replay) > self.max_replay_messages
            or self._replay_bytes > self.max_replay_bytes
        ):
            dropped = self._replay.popleft()
            self._replay_bytes -= dropped.cost
            # Drop the reference to the base64 text as soon as it leaves the
            # buffer, so a retained envelope elsewhere cannot keep it alive.
            dropped.body = {}
            self._replay_low = self._replay[0].seq if self._replay else seq + 1

    def can_replay_from(self, last_seq: int) -> bool:
        """True when everything after ``last_seq`` is still buffered."""
        with self._lock:
            want = last_seq + 1
            if want >= self._seq:
                return True  # nothing missing
            if not self._replay:
                return False
            return self._replay[0].seq <= want

    def replay(self, last_seq: int, *, client_id: Optional[str] = None) -> ReplayReport:
        """Resend every retained message with ``seq > last_seq``.

        Original ``seq`` values and bodies are reused verbatim -- option 1 of
        ``PROTOCOL.md`` §4. The session's own sequencer is untouched, so a
        replay racing with live sending cannot renumber the live stream.
        """
        target = client_id if client_id is not None else self.client_id
        with self._lock:
            if self._state is SessionState.CLOSED:
                raise PreviewStateError(
                    f"session {self.session_id} is closed; nothing left to replay"
                )
            pending = [item for item in self._replay if item.seq > last_seq]
            sent = 0
            failed = 0
            for item in pending:
                try:
                    self._sender(MESSAGE_TYPE, item.body, target)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    self._log.warning(
                        "raven preview: replay send failed for session %s seq %d (%s: %s)",
                        self.session_id,
                        item.seq,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    sent += 1
            self.replayed_messages += sent
            self._last_activity = self._clock()
            first = pending[0].seq if pending else None
            last = pending[-1].seq if pending else None
            return ReplayReport(sent=sent, first=first, last=last, failed=failed)

    # -- resources --------------------------------------------------------

    def add_finalizer(self, fn: Callable[[], Any], *, name: str = "") -> None:
        """Register a zero-argument release hook (writer close, callback drop).

        Finalizers run exactly once, in LIFO order, each isolated. Register the
        *release* of a resource here, never the resource itself: the session
        must not be what keeps a muxer, decoder, tensor or model alive.
        """
        if not callable(fn):
            raise TypeError("finalizer must be callable")
        with self._lock:
            if self._finalized:
                # Late registration would silently never run; run it now so the
                # resource is still released, and say so.
                self._log.warning(
                    "raven preview: finalizer %r registered after session %s was "
                    "released; running it immediately",
                    name or getattr(fn, "__name__", "fn"),
                    self.session_id,
                )
                _run_guarded(fn, self._log, f"finalizer {name}")
                return
            self._finalizers.append((name or getattr(fn, "__name__", "fn"), fn))

    def release(self) -> bool:
        """Run the finalizers exactly once. Replay stays available.

        Called the moment a session reaches a terminal state: the heavy things
        (muxer, encoder handles, decoder callbacks) go now, while the small
        base64 replay buffer lives on for the TTL so a late client can resume.
        """
        with self._lock:
            if self._finalized:
                return False
            self._finalized = True
            pending = list(reversed(self._finalizers))
            self._finalizers = []
        for name, fn in pending:
            _run_guarded(fn, self._log, f"finalizer {name}")
        return True

    def close(self) -> bool:
        """Release resources *and* drop the replay buffer. Idempotent."""
        released = self.release()
        with self._lock:
            if self._state is SessionState.CLOSED and not released:
                return False
            for item in self._replay:
                item.body = {}
            self._replay.clear()
            self._replay_bytes = 0
            self._replay_low = self._seq
            self._state = SessionState.CLOSED
            if self._terminal_at is None:
                self._terminal_at = self._clock()
            return True

    # -- isolation --------------------------------------------------------

    def isolated(self) -> "_IsolatedSession":
        """A view whose methods never raise. What the node should hold.

        Every call is wrapped: a protocol error, an oversized payload or a dead
        transport is logged and swallowed, because a preview failure must not
        affect sampling (``PROTOCOL.md`` §6.5).
        """
        return _IsolatedSession(self)


class _IsolatedSession:
    """Non-raising proxy over :class:`PreviewSession`."""

    __slots__ = ("_session", "errors")

    def __init__(self, session: PreviewSession) -> None:
        self._session = session
        self.errors = 0

    @property
    def session(self) -> PreviewSession:
        return self._session

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._session, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                self.errors += 1
                self._session._log.warning(
                    "raven preview: %s failed for session %s (%s: %s); sampling "
                    "continues",
                    name,
                    self._session.session_id,
                    type(exc).__name__,
                    exc,
                )
                return None

        wrapper.__name__ = name
        return wrapper


def _run_guarded(fn: Callable[[], Any], log: logging.Logger, what: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "raven preview: %s raised (%s: %s); continuing cleanup",
            what,
            type(exc).__name__,
            exc,
        )


def _normalise_progress(progress: Any) -> Dict[str, Any]:
    if isinstance(progress, _abc.Mapping):
        value = progress.get("value")
        maximum = progress.get("max")
    elif isinstance(progress, _abc.Sequence) and not isinstance(progress, (str, bytes)):
        if len(progress) != 2:
            raise ValueError("progress sequence must be (value, max)")
        value, maximum = progress
    else:
        raise TypeError("progress must be a mapping or a (value, max) pair")
    out: Dict[str, Any] = {}
    if value is not None:
        out["value"] = float(value) if isinstance(value, float) else int(value)
    if maximum is not None:
        out["max"] = float(maximum) if isinstance(maximum, float) else int(maximum)
    return out


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------


#: Resume outcomes. None of them is a 404/405/501: the client reads those three
#: as "this backend has no resume route" and stops asking forever
#: (``web/raven_streaming_preview.js``), so a real answer must never use them.
RESUME_REPLAYED = "replayed"
RESUME_UP_TO_DATE = "up_to_date"
RESUME_RESYNC = "resync"
RESUME_EXPIRED = "expired"
RESUME_TERMINAL = "terminal"
RESUME_UNKNOWN = "unknown_session"
RESUME_MISMATCH = "mismatch"
RESUME_BAD_REQUEST = "bad_request"


@dataclass(frozen=True)
class ResumeResult:
    """What the HTTP route should answer."""

    status: str
    http_status: int = 200
    detail: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def body(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"v": PROTOCOL_VERSION, "status": self.status}
        if self.detail:
            out["detail"] = self.detail
        out.update(self.extra)
        return out


class PreviewManager:
    """Owns every live and recently-finished session.

    Lifetime, stated once:

    * active while the node runs;
    * terminal (``end`` sent) but still resumable for ``terminal_ttl`` seconds,
      with resources already released;
    * closed and forgotten after that, or the moment :meth:`cleanup` is called.

    Every path -- normal completion, ``end``, cancellation, error, a client that
    went away, an exception that skipped the node's own cleanup -- converges on
    the same idempotent :meth:`cleanup`.
    """

    def __init__(
        self,
        sender: Optional[Sender] = None,
        *,
        terminal_ttl: float = DEFAULT_TERMINAL_TTL,
        active_idle_ttl: Optional[float] = DEFAULT_ACTIVE_IDLE_TTL,
        max_replay_messages: int = DEFAULT_REPLAY_MESSAGES,
        max_replay_bytes: int = DEFAULT_REPLAY_BYTES,
        max_payload_bytes: int = MAX_RAW_PAYLOAD_BYTES,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._sender: Sender = sender if sender is not None else NullSender()
        self.terminal_ttl = float(terminal_ttl)
        self.active_idle_ttl = None if active_idle_ttl is None else float(active_idle_ttl)
        self.max_replay_messages = int(max_replay_messages)
        self.max_replay_bytes = int(max_replay_bytes)
        self.max_payload_bytes = int(max_payload_bytes)
        self._clock = clock
        self._wall = wall_clock
        self._log = log if log is not None else logger
        self._lock = threading.RLock()
        self._active: Dict[str, PreviewSession] = {}
        self._terminal: Dict[str, PreviewSession] = {}
        self._counter = itertools.count(1)
        self.cleanups = 0
        self.pruned = 0

    # -- sender -----------------------------------------------------------

    @property
    def sender(self) -> Sender:
        return self._sender

    def set_sender(self, sender: Optional[Sender]) -> None:
        """Swap the transport. Existing sessions keep the one they were made
        with, so a swap mid-run cannot reorder a live stream."""
        with self._lock:
            self._sender = sender if sender is not None else NullSender()

    # -- creation ---------------------------------------------------------

    def create_session(
        self,
        node_id: Any,
        *,
        client_id: Optional[str] = None,
        prompt_id: Optional[Any] = None,
        session_id: Optional[str] = None,
        seq_start: int = 0,
        sender: Optional[Sender] = None,
    ) -> PreviewSession:
        """One session per node execution (``PROTOCOL.md`` §5)."""
        session = PreviewSession(
            node_id,
            session_id=session_id,
            client_id=client_id,
            prompt_id=prompt_id,
            sender=sender if sender is not None else self._sender,
            seq_start=seq_start,
            max_replay_messages=self.max_replay_messages,
            max_replay_bytes=self.max_replay_bytes,
            max_payload_bytes=self.max_payload_bytes,
            clock=self._clock,
            wall_clock=self._wall,
            log=self._log,
        )
        with self._lock:
            next(self._counter)
            # A node that starts a new run supersedes its own previous stream:
            # the client tears the old one down on the new ``open`` anyway, so
            # holding it here only wastes memory.
            superseded = [
                s
                for s in self._active.values()
                if s.node_id == session.node_id and s.client_id == session.client_id
            ]
            for old in superseded:
                self._active.pop(old.session_id, None)
                self._terminal.pop(old.session_id, None)
            self._active[session.session_id] = session
        # Finalizers run outside the manager lock: they are caller-supplied and
        # may well call back in (a sink's close, a writer's flush).
        for old in superseded:
            self._log.debug(
                "raven preview: node %s started a new session; retiring %s",
                session.node_id,
                old.session_id,
            )
            old.close()
        self.prune()
        return session

    @contextlib.contextmanager
    def session(
        self,
        node_id: Any,
        *,
        client_id: Optional[str] = None,
        prompt_id: Optional[Any] = None,
        session_id: Optional[str] = None,
        seq_start: int = 0,
        sender: Optional[Sender] = None,
        auto_complete: bool = True,
    ) -> Iterator[PreviewSession]:
        """Scoped session for a node's ``execute``.

        Guarantees exactly one terminal message and one cleanup on every exit
        path, including exceptions and cancellation.

        Cancellation is only ever *observed* here. ComfyUI's interrupt is
        raised by the sampler's own ``cancel_check``; this block classifies the
        escaping exception so the client is told ``cancelled`` rather than
        ``error``, then re-raises it untouched. It never calls
        ``model_management.interrupt_current_processing`` or any other global
        interrupt -- the preview is an observer of cancellation, never a source.
        """
        session = self.create_session(
            node_id,
            client_id=client_id,
            prompt_id=prompt_id,
            session_id=session_id,
            seq_start=seq_start,
            sender=sender,
        )
        try:
            yield session
        except BaseException as exc:  # noqa: BLE001 - classify, report, re-raise
            reason = "cancelled" if _looks_like_cancellation(exc) else "error"
            self.finish(session, reason, message=_short_reason(exc))
            raise
        else:
            if auto_complete:
                self.finish(session, "complete")
        finally:
            self.retire(session)

    # -- lifetime ---------------------------------------------------------

    def get(self, session_id: Optional[str]) -> Optional[PreviewSession]:
        if not session_id:
            return None
        with self._lock:
            return self._active.get(session_id) or self._terminal.get(session_id)

    @property
    def active_sessions(self) -> List[PreviewSession]:
        with self._lock:
            return list(self._active.values())

    @property
    def terminal_sessions(self) -> List[PreviewSession]:
        with self._lock:
            return list(self._terminal.values())

    def finish(
        self,
        session: PreviewSession,
        reason: str = "complete",
        *,
        message: Optional[str] = None,
        segments: Optional[int] = None,
    ) -> bool:
        """Send the terminal message if there is not one already. Never raises."""
        if session.is_terminal:
            return False
        try:
            session.send_end(reason, message=message, segments=segments)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "raven preview: could not send end(%s) for session %s (%s: %s)",
                reason,
                session.session_id,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def retire(self, session: PreviewSession) -> None:
        """Move a session out of the active set: release resources, keep replay.

        Idempotent, and safe for a session that never reached a terminal state
        (it is closed outright in that case -- there is nothing to resume to).
        """
        keep = (
            not session.is_closed
            and session.state is SessionState.ENDED
            and self.terminal_ttl > 0
        )
        with self._lock:
            self._active.pop(session.session_id, None)
            if keep:
                self._terminal[session.session_id] = session
            else:
                self._terminal.pop(session.session_id, None)
        # Outside the lock: finalizers are caller code.
        if keep:
            session.release()
        else:
            session.close()

    def cleanup(self, session_id: Any, *, reason: Optional[str] = None) -> bool:
        """Forget a session completely. Idempotent; safe to call from anywhere.

        ``reason`` sends a terminal message first when the session has none --
        that is the disconnect / cancel / error path. Returns True the first
        time it actually did something.
        """
        key = (
            session_id.session_id
            if isinstance(session_id, PreviewSession)
            else str(session_id)
        )
        with self._lock:
            session = self._active.pop(key, None) or self._terminal.pop(key, None)
        if session is None:
            return False
        if reason is not None:
            self.finish(session, reason)
        session.close()
        self.cleanups += 1
        return True

    def prune(self, now: Optional[float] = None) -> List[str]:
        """Drop expired sessions. Returns the ids that were dropped.

        This is the whole disconnect story. The pinned ComfyUI has **no public
        hook for a client going away**: ``server.py`` pops the socket in the
        websocket handler's ``finally`` (``server.py:325-326``) and fires
        nothing, and ``send_json`` swallows socket errors inside
        ``send_socket_catch_exception``, so a send to a dead client is not even
        an error we can see. Rather than invent a disconnect callback that does
        not exist, a stream is bounded by: send failures (counted, never fatal),
        this TTL, and the client-driven resume route.

        Call it periodically (``create_session`` does, and the route does).
        """
        now = self._clock() if now is None else now
        expired: List[PreviewSession] = []
        abandoned: List[PreviewSession] = []
        with self._lock:
            for session in list(self._terminal.values()):
                at = session.terminal_at
                if at is None or now - at >= self.terminal_ttl:
                    self._terminal.pop(session.session_id, None)
                    expired.append(session)
            if self.active_idle_ttl is not None:
                for session in list(self._active.values()):
                    if now - session.last_activity >= self.active_idle_ttl:
                        self._active.pop(session.session_id, None)
                        abandoned.append(session)
        # Sending and finalizing happen outside the lock.
        for session in abandoned:
            self._log.warning(
                "raven preview: session %s idle for %.0fs; abandoning it",
                session.session_id,
                now - session.last_activity,
            )
            self.finish(session, "error", message="preview stream timed out")
        dropped = [s.session_id for s in expired + abandoned]
        for session in expired + abandoned:
            session.close()
        self.pruned += len(dropped)
        return dropped

    def shutdown(self) -> None:
        """Close everything. Idempotent."""
        with self._lock:
            sessions = list(self._active.values()) + list(self._terminal.values())
            self._active.clear()
            self._terminal.clear()
        for session in sessions:
            session.close()

    # -- resume -----------------------------------------------------------

    def handle_resume(self, payload: Any) -> ResumeResult:
        """Decide what a ``POST .../preview/resume`` should do.

        Pure: no aiohttp, no ComfyUI. The HTTP layer in
        :mod:`raven_streaming.preview_server` only parses the body and renders
        the result, so this is directly testable.

        ``last_seq`` is the last *contiguously delivered* seq (``PROTOCOL.md``
        §4); the client may legitimately send ``-1`` when it has delivered
        nothing yet.
        """
        if not isinstance(payload, _abc.Mapping):
            return ResumeResult(RESUME_BAD_REQUEST, 400, "body must be a JSON object")
        session_id = payload.get("session_id")
        node_id = payload.get("node_id")
        last_seq = payload.get("last_seq")
        client_id = payload.get("client_id")
        reason = payload.get("reason")

        if not isinstance(session_id, str) or not session_id:
            return ResumeResult(RESUME_BAD_REQUEST, 400, "session_id must be a non-empty string")
        if node_id is None or not str(node_id):
            return ResumeResult(RESUME_BAD_REQUEST, 400, "node_id is required")
        if isinstance(last_seq, bool) or not isinstance(last_seq, int):
            return ResumeResult(RESUME_BAD_REQUEST, 400, "last_seq must be an integer")
        if last_seq < -1:
            return ResumeResult(RESUME_BAD_REQUEST, 400, "last_seq must be >= -1")
        if client_id is not None and not isinstance(client_id, str):
            return ResumeResult(RESUME_BAD_REQUEST, 400, "client_id must be a string or null")
        if reason is not None and not isinstance(reason, str):
            return ResumeResult(RESUME_BAD_REQUEST, 400, "reason must be a string or null")

        self.prune()
        session = self.get(session_id)
        if session is None:
            # Deliberately 200, not 404: the client reads 404/405/501 as "no
            # resume support on this backend" and never asks again.
            return ResumeResult(
                RESUME_UNKNOWN,
                200,
                "no such session; it expired or belonged to an earlier run",
                {"session_id": session_id, "resync": True},
            )
        if str(node_id) != session.node_id:
            return ResumeResult(
                RESUME_MISMATCH,
                200,
                f"session {session_id} belongs to node {session.node_id}, not {node_id}",
                {"session_id": session_id},
            )
        if (
            client_id is not None
            and session.client_id is not None
            and client_id != session.client_id
        ):
            return ResumeResult(
                RESUME_MISMATCH,
                200,
                "session belongs to another client",
                {"session_id": session_id},
            )
        if session.is_closed:
            return ResumeResult(
                RESUME_EXPIRED,
                200,
                "session is closed; its buffered messages are gone",
                {"session_id": session_id, "resync": True},
            )

        target = client_id if client_id is not None else session.client_id
        extra: Dict[str, Any] = {
            "session_id": session.session_id,
            "node_id": session.node_id,
            "last_seq": last_seq,
            "next_seq": session.next_seq,
        }
        if session.terminal_reason is not None:
            extra["terminal_reason"] = session.terminal_reason

        if last_seq + 1 >= session.next_seq:
            status = RESUME_TERMINAL if session.is_terminal else RESUME_UP_TO_DATE
            return ResumeResult(status, 200, "nothing to resend", extra)

        if not session.can_replay_from(last_seq):
            span = session.replay_span
            extra["available_from"] = None if span is None else span[0]
            extra["resync"] = True
            # Option 2 of PROTOCOL.md §4: the honest answer when the fragments
            # the client is missing have already been evicted. Restarting the
            # stream is the sampler's call, not ours.
            return ResumeResult(
                RESUME_RESYNC,
                200,
                "the missing messages are no longer buffered; a fresh stream is needed",
                extra,
            )

        report = session.replay(last_seq, client_id=target)
        extra.update(
            {
                "resent": report.sent,
                "from": report.first,
                "to": report.last,
                "failed": report.failed,
            }
        )
        status = RESUME_REPLAYED if report.sent else RESUME_UP_TO_DATE
        return ResumeResult(status, 200, None, extra)


#: Exception names that mean "the user stopped this", not "this broke".
#: Matched by name so nothing has to be imported: ``CancelledError`` is
#: asyncio's, ``InterruptProcessingException`` is ComfyUI's, and
#: ``SamplingCancelled`` is what
#: :func:`raven_streaming.consistency.sample_streaming` raises when its
#: ``cancel_check`` returns truthy -- a cancelled rollout that reported itself
#: as ``error`` would send the user looking for a fault that is not there.
CANCELLATION_EXCEPTION_NAMES: frozenset = frozenset(
    {
        "CancelledError",
        "InterruptProcessingException",
        "ProcessingInterrupted",
        "SamplingCancelled",
    }
)


def _looks_like_cancellation(exc: BaseException) -> bool:
    """Classify an escaping exception as cancellation, without importing Comfy.

    Recognises ``KeyboardInterrupt``, ``asyncio.CancelledError``, ComfyUI's
    ``InterruptProcessingException`` and this package's ``SamplingCancelled``
    by name (:data:`CANCELLATION_EXCEPTION_NAMES`). Classification only: no
    interrupt is raised, requested or cleared here.
    """
    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & CANCELLATION_EXCEPTION_NAMES)


def _short_reason(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    head = text[0] if text else ""
    label = f"{type(exc).__name__}: {head}" if head else type(exc).__name__
    return label[:200]


# --------------------------------------------------------------------------
# Sinks: the seam to the media lane
# --------------------------------------------------------------------------


class MediaSink:
    """The interface the media lane pushes through. Bytes and control only.

    Deliberately narrow. The muxer hands over finished fMP4 byte ranges; the
    streaming decoders hand over *counts* (frames emitted, samples emitted) for
    progress. Nothing here accepts a tensor, an array, a frame object or a
    model, so no preview object can ever be what keeps GPU memory alive, and
    this module never has to import :mod:`raven_streaming.media`, torch or PyAV.

    Encoding, muxing and decoding stay entirely in the media lane; this is only
    the transport seam.
    """

    def on_open(self, mime: str, **kwargs: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_init(self, data: bytes) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_fragment(self, data: bytes, **kwargs: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_status(self, phase: str, **kwargs: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_end(self, reason: str, **kwargs: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class PreviewMediaSink(MediaSink):
    """Adapts a :class:`PreviewSession` to the media lane.

    Every call is isolated: a preview failure is logged and counted, and the
    muxer, the decoders and the sampler carry on. That includes
    :class:`PreviewPayloadTooLarge`, which is loud in the log (the fix is a
    smaller fragment cadence) but still not fatal to a run.

    Wiring, all duck-typed so nothing is imported from the media lane:

    * ``FragmentedMP4Muxer`` / ``FragmentedMP4Segmenter``: pass
      :meth:`fragment_callback` as their ``on_fragment``, or pull with
      :meth:`pump_muxer` after each write.
    * ``IncrementalVideoDecoder`` / ``OverlapSaveAudioDecoder``: call
      :meth:`progress` with counts. Their outputs (tensors) never come here.
    """

    __slots__ = ("_session", "_log", "errors", "_closed", "_index")

    def __init__(
        self,
        session: PreviewSession,
        *,
        log: Optional[logging.Logger] = None,
        register_finalizer: bool = True,
    ) -> None:
        self._session = session
        self._log = log if log is not None else logger
        self.errors = 0
        self._closed = False
        self._index = 0
        if register_finalizer:
            session.add_finalizer(self.close, name="preview-sink")

    @property
    def session(self) -> PreviewSession:
        return self._session

    @property
    def closed(self) -> bool:
        return self._closed

    def _guard(self, what: str, fn: Callable[[], Any]) -> bool:
        if self._closed:
            return False
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a preview failure is not a run failure
            self.errors += 1
            self._log.warning(
                "raven preview: %s failed for session %s (%s: %s); sampling continues",
                what,
                self._session.session_id,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    # -- MediaSink --------------------------------------------------------

    def on_open(self, mime: str, **kwargs: Any) -> bool:
        return self._guard("open", lambda: self._session.send_open(mime, **kwargs))

    def on_init(self, data: bytes) -> bool:
        return self._guard("init", lambda: self._session.send_init(data))

    def on_fragment(
        self,
        data: bytes,
        *,
        index: Optional[int] = None,
        keyframe: bool = False,
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> bool:
        if index is None:
            index = self._index
        self._index = index + 1
        return self._guard(
            "segment",
            lambda: self._session.send_segment(
                data, index=index, keyframe=keyframe, start=start, duration=duration
            ),
        )

    def on_status(
        self,
        phase: str,
        *,
        message: Optional[str] = None,
        progress: Optional[Any] = None,
    ) -> bool:
        return self._guard(
            "status",
            lambda: self._session.send_status(phase, message=message, progress=progress),
        )

    def on_end(
        self,
        reason: str,
        *,
        message: Optional[str] = None,
        segments: Optional[int] = None,
    ) -> bool:
        return self._guard(
            "end",
            lambda: self._session.send_end(reason, message=message, segments=segments),
        )

    # -- media-lane adapters ---------------------------------------------

    def fragment_callback(self) -> Callable[[Any], None]:
        """A ``on_fragment(segment)`` callable for the muxer/segmenter.

        Reads ``.data`` / ``.kind`` / ``.index`` off whatever it is given and
        keeps no reference to it. ``kind == "init"`` routes to the init message,
        ``"fragment"`` to a segment, and a ``"trailer"`` is ignored (MSE has no
        use for it; the client ends the stream on ``end``).
        """

        def callback(segment: Any) -> None:
            kind = getattr(segment, "kind", "fragment")
            data = getattr(segment, "data", None)
            if not isinstance(data, (bytes, bytearray, memoryview)):
                self.errors += 1
                self._log.warning(
                    "raven preview: fragment callback got %s without usable bytes",
                    type(segment).__name__,
                )
                return
            if kind == "init":
                self.on_init(bytes(data))
            elif kind == "fragment":
                index = getattr(segment, "index", None)
                self.on_fragment(
                    bytes(data),
                    index=index if isinstance(index, int) and index >= 0 else None,
                    keyframe=True,  # the muxer forces an IDR at every boundary
                )

        return callback

    def pump_muxer(self, muxer: Any) -> int:
        """Pull whatever the muxer has ready and forward it. Returns the count.

        Pull-based sibling of :meth:`fragment_callback`, for a caller that
        prefers to drain after each write. The muxer reference is a parameter,
        never stored.
        """
        sent = 0
        take_init = getattr(muxer, "take_init_segment", None)
        if callable(take_init) and self._session.state is SessionState.OPENED:
            init = take_init()
            if init:
                if self.on_init(bytes(init)):
                    sent += 1
        take_fragments = getattr(muxer, "take_fragments", None)
        if callable(take_fragments):
            for segment in take_fragments() or ():
                data = getattr(segment, "data", segment)
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    continue
                index = getattr(segment, "index", None)
                if self.on_fragment(
                    bytes(data),
                    index=index if isinstance(index, int) and index >= 0 else None,
                    keyframe=True,
                ):
                    sent += 1
        return sent

    def progress(
        self,
        phase: str = "sampling",
        *,
        value: Optional[float] = None,
        maximum: Optional[float] = None,
        message: Optional[str] = None,
    ) -> bool:
        """Control-only progress, e.g. from a decoder's emitted-frame count."""
        progress = None
        if value is not None or maximum is not None:
            progress = {"value": value, "max": maximum}
        return self.on_status(phase, message=message, progress=progress)

    def close(self) -> None:
        """Stop forwarding. Idempotent; never raises."""
        self._closed = True
