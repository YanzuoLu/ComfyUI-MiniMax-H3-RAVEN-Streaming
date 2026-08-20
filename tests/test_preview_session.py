"""Session-level behaviour of the ``raven.preview`` backend.

Everything here runs in a bare interpreter: no ComfyUI, no aiohttp, no torch.
The transport is a fake sender that records what it was handed, which is also
what lets the ordering and isolation claims be checked instead of asserted in
prose.
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.preview_session import (  # noqa: E402
    BACKEND_PHASES,
    END_REASONS,
    EVENT_KINDS,
    MAX_RAW_PAYLOAD_BYTES,
    MESSAGE_TYPE,
    PROTOCOL_VERSION,
    PreviewManager,
    PreviewMediaSink,
    PreviewPayloadTooLarge,
    PreviewSession,
    PreviewStateError,
    RecordingSender,
    SessionState,
    base64_cost,
)

MIME = 'video/mp4; codecs="avc1.640028,mp4a.40.2"'
WEB = Path(__file__).resolve().parents[1] / "web"


class FakeClock:
    """Monotonic clock the test drives by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class ExplodingSender:
    """A transport that fails, the way a dead websocket eventually does."""

    def __init__(self, fail_after: int = 0) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.delivered = []

    def __call__(self, message_type, body, client_id):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("socket is gone")
        self.delivered.append(body)


@pytest.fixture
def sender() -> RecordingSender:
    return RecordingSender()


@pytest.fixture
def session(sender: RecordingSender) -> PreviewSession:
    return PreviewSession("7", sender=sender, client_id="cid", prompt_id="p1")


def open_stream(session: PreviewSession) -> None:
    session.send_open(MIME, width=848, height=480, fps=24)
    session.send_init(b"ftypmoov-init")


# -- the constants come from the protocol, not from here -------------------


def js_string_list(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}\s*=\s*Object\.freeze\(\[(.*?)\]\)", source, re.S)
    assert match, f"{name} not found in the client"
    return re.findall(r"'([^']+)'", match.group(1))


def test_backend_constants_match_the_client():
    js = (WEB / "lib" / "protocol.js").read_text(encoding="utf-8")
    from raven_streaming.preview_session import (
        PAYLOAD_ENCODING,
        RESUME_ROUTE,
    )

    assert re.search(r"MESSAGE_TYPE\s*=\s*'([^']+)'", js).group(1) == MESSAGE_TYPE
    assert int(re.search(r"PROTOCOL_VERSION\s*=\s*(\d+)", js).group(1)) == PROTOCOL_VERSION
    assert js_string_list(js, "EVENT_KINDS") == list(EVENT_KINDS)
    assert js_string_list(js, "BACKEND_PHASES") == list(BACKEND_PHASES)
    assert js_string_list(js, "END_REASONS") == list(END_REASONS)
    assert js_string_list(js, "PAYLOAD_ENCODINGS") == [PAYLOAD_ENCODING]
    assert re.search(r"RESUME_ROUTE\s*=\s*'([^']+)'", js).group(1) == RESUME_ROUTE


def test_payload_limit_is_the_documented_size_guidance():
    doc = (WEB / "PROTOCOL.md").read_text(encoding="utf-8")
    assert "~256 KB of raw media" in doc
    assert MAX_RAW_PAYLOAD_BYTES == 256 * 1024
    # The document has no fragment-part field to split an oversized payload on.
    assert "part field" not in doc.split("## 3. Message format")[1].split("## 4.")[0]


# -- envelope --------------------------------------------------------------


def test_message_type_and_version_are_the_protocol_ones(session, sender):
    session.send_open(MIME)
    message_type, body, sid = sender.messages[0]
    assert message_type == MESSAGE_TYPE == "raven.preview"
    assert body["v"] == PROTOCOL_VERSION == 1
    assert sid == "cid"


def test_every_message_carries_the_full_envelope(session, sender):
    open_stream(session)
    session.send_status("sampling", progress={"value": 1, "max": 40})
    session.send_segment(b"moofmdat", index=0, keyframe=True, start=0.0, duration=0.875)
    session.send_end("complete")
    for body in sender.bodies:
        assert body["v"] == 1
        assert body["event"] in EVENT_KINDS
        assert body["session_id"] == session.session_id
        assert body["node_id"] == "7"
        assert body["prompt_id"] == "p1"
        assert isinstance(body["seq"], int)
        assert isinstance(body["t"], float)


def test_bodies_are_json_serialisable(session, sender):
    open_stream(session)
    session.send_status("finalizing", message="wrapping up")
    session.send_segment(b"abc")
    session.send_end("complete")
    for body in sender.bodies:
        json.loads(json.dumps(body))  # would raise on bytes or a tensor


def test_node_id_is_stringified(sender):
    session = PreviewSession(7, sender=sender)
    session.send_open(MIME)
    assert sender.bodies[0]["node_id"] == "7"


# -- sequencing ------------------------------------------------------------


def test_one_seq_covers_every_event_kind(session, sender):
    session.send_open(MIME)
    session.send_status("model_loading")
    session.send_init(b"init")
    session.send_status("sampling")
    session.send_segment(b"frag-0")
    session.send_segment(b"frag-1")
    session.send_end("complete")
    assert sender.events() == [
        "open",
        "status",
        "init",
        "status",
        "segment",
        "segment",
        "end",
    ]
    assert sender.seqs() == [0, 1, 2, 3, 4, 5, 6]


def test_seq_start_offsets_the_whole_session(sender):
    session = PreviewSession("7", sender=sender, seq_start=100)
    open_stream(session)
    assert sender.seqs() == [100, 101]


def test_send_failure_still_consumes_the_seq(sender):
    bad = ExplodingSender(fail_after=1)
    session = PreviewSession("7", sender=bad)
    session.send_open(MIME)  # delivered
    session.send_init(b"init")  # fails
    session.send_segment(b"frag")  # fails
    assert session.send_failures == 2
    assert session.messages_sent == 1
    assert session.next_seq == 3  # no renumbering around the hole


def test_concurrent_senders_produce_a_dense_ordered_seq(sender):
    session = PreviewSession("7", sender=sender)
    open_stream(session)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                session.send_segment(b"x" * 32)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    seqs = sender.seqs()
    assert seqs == sorted(seqs), "wire order must match seq order"
    assert seqs == list(range(len(seqs)))
    assert len(seqs) == 2 + 8 * 50
    assert session.segments_sent == 400


# -- payloads --------------------------------------------------------------


def test_media_payloads_are_base64_with_a_verified_length(session, sender):
    payload = bytes(range(256)) * 4
    open_stream(session)
    session.send_segment(payload)
    body = sender.bodies[-1]
    assert body["encoding"] == "base64"
    assert body["bytes"] == len(payload)
    assert base64.b64decode(body["data"]) == payload
    assert len(body["data"]) == base64_cost(len(payload))


def test_init_payload_round_trips(session, sender):
    session.send_open(MIME)
    session.send_init(b"\x00\x00\x00\x1cftypiso5")
    body = sender.bodies[-1]
    assert body["event"] == "init"
    assert base64.b64decode(body["data"]) == b"\x00\x00\x00\x1cftypiso5"


def test_oversized_payload_fails_loudly_rather_than_being_reframed(session, sender):
    open_stream(session)
    too_big = b"x" * (MAX_RAW_PAYLOAD_BYTES + 1)
    with pytest.raises(PreviewPayloadTooLarge) as excinfo:
        session.send_segment(too_big)
    # Protocol v1 has no part field; the message must not be split or truncated.
    assert "part field" in str(excinfo.value)
    assert sender.events() == ["open", "init"]
    assert session.next_seq == 2


def test_payload_must_be_bytes(session):
    session.send_open(MIME)
    with pytest.raises(TypeError):
        session.send_init("not bytes")  # type: ignore[arg-type]


def test_open_requires_a_mime_with_codecs(sender):
    session = PreviewSession("7", sender=sender)
    with pytest.raises(ValueError):
        session.send_open("video/mp4")
    with pytest.raises(ValueError):
        session.send_open('video/webm; codecs="vp9"')
    assert sender.messages == []


def test_only_backend_phases_are_accepted(session):
    session.send_open(MIME)
    for phase in BACKEND_PHASES:
        session.send_status(phase)
    for client_only in ("buffering", "live", "reconnecting"):
        with pytest.raises(ValueError):
            session.send_status(client_only)


def test_only_protocol_end_reasons_are_accepted(session):
    session.send_open(MIME)
    with pytest.raises(ValueError):
        session.send_end("kaput")
    assert set(END_REASONS) == {"complete", "cancelled", "error"}


# -- state machine ---------------------------------------------------------


def test_open_must_come_first(session):
    with pytest.raises(PreviewStateError):
        session.send_init(b"init")
    with pytest.raises(PreviewStateError):
        session.send_status("sampling")
    with pytest.raises(PreviewStateError):
        session.send_segment(b"frag")


def test_open_only_once(session):
    session.send_open(MIME)
    with pytest.raises(PreviewStateError):
        session.send_open(MIME)


def test_init_only_once(session):
    open_stream(session)
    with pytest.raises(PreviewStateError):
        session.send_init(b"init-again")


def test_segment_only_after_init(session):
    session.send_open(MIME)
    with pytest.raises(PreviewStateError):
        session.send_segment(b"frag")
    session.send_init(b"init")
    session.send_segment(b"frag")  # now fine


def test_single_terminal_and_nothing_after_it(session, sender):
    open_stream(session)
    session.send_end("complete")
    assert session.is_terminal
    assert session.terminal_reason == "complete"
    for call in (
        lambda: session.send_end("error"),
        lambda: session.send_status("sampling"),
        lambda: session.send_segment(b"frag"),
        lambda: session.send_init(b"init"),
        lambda: session.send_open(MIME),
    ):
        with pytest.raises(PreviewStateError):
            call()
    assert sender.events().count("end") == 1
    assert session.rejected == 5


def test_end_can_follow_open_without_init(session, sender):
    session.send_open(MIME)
    session.failed("model did not load")
    assert sender.events() == ["open", "end"]
    assert sender.bodies[-1]["reason"] == "error"


def test_end_reports_the_segment_count(session, sender):
    open_stream(session)
    for _ in range(3):
        session.send_segment(b"frag")
    session.complete()
    assert sender.bodies[-1]["segments"] == 3


def test_closed_session_rejects_everything(session):
    session.send_open(MIME)
    session.close()
    assert session.state is SessionState.CLOSED
    with pytest.raises(PreviewStateError):
        session.send_status("sampling")


def test_states_progress_as_documented(session):
    assert session.state is SessionState.IDLE
    session.send_open(MIME)
    assert session.state is SessionState.OPENED
    session.send_init(b"init")
    assert session.state is SessionState.STREAMING
    session.complete()
    assert session.state is SessionState.ENDED
    session.close()
    assert session.state is SessionState.CLOSED


# -- isolation -------------------------------------------------------------


def test_isolated_view_never_raises(session, caplog):
    isolated = session.isolated()
    with caplog.at_level(logging.WARNING):
        assert isolated.send_segment(b"frag") is None  # before open: state error
        assert isolated.send_open(MIME) == 0
        assert isolated.send_init(b"x" * (MAX_RAW_PAYLOAD_BYTES + 1)) is None
    assert isolated.errors == 2
    assert any("sampling continues" in r.getMessage() for r in caplog.records)


def test_send_failures_do_not_propagate(sender):
    session = PreviewSession("7", sender=ExplodingSender(fail_after=0))
    session.send_open(MIME)  # transport raises inside, call returns normally
    session.send_init(b"init")
    session.complete()
    assert session.send_failures == 3
    assert session.messages_sent == 0
    assert session.is_terminal  # state still advanced


# -- replay ----------------------------------------------------------------


def test_replay_resends_the_originals_verbatim(session, sender):
    open_stream(session)
    for i in range(4):
        session.send_segment(b"frag-%d" % i, index=i)
    before = len(sender.messages)

    report = session.replay(last_seq=2)

    assert report.sent == 3
    assert (report.first, report.last) == (3, 5)
    resent = sender.messages[before:]
    assert [b["seq"] for _, b, _ in resent] == [3, 4, 5]
    assert [b["event"] for _, b, _ in resent] == ["segment"] * 3
    # The live sequencer is untouched by a replay.
    assert session.next_seq == 6


def test_replay_targets_the_requesting_client(session, sender):
    open_stream(session)
    session.replay(last_seq=-1, client_id="other-tab")
    assert {sid for _, _, sid in sender.messages[2:]} == {"other-tab"}


def test_replay_bound_by_message_count(sender):
    session = PreviewSession("7", sender=sender, max_replay_messages=4)
    open_stream(session)
    for i in range(10):
        session.send_segment(b"frag", index=i)
    span = session.replay_span
    assert span == (8, 11)
    assert not session.can_replay_from(0)
    assert session.can_replay_from(7)


def test_replay_bound_by_bytes(sender):
    session = PreviewSession(
        "7", sender=sender, max_replay_messages=10_000, max_replay_bytes=4096
    )
    open_stream(session)
    for i in range(20):
        session.send_segment(b"x" * 1024, index=i)
    assert session.replay_bytes <= 4096
    assert len(session.replay_span) == 2


def test_replay_after_terminal_still_works(session):
    open_stream(session)
    session.send_segment(b"frag")
    session.complete()
    report = session.replay(last_seq=1)
    assert report.sent == 2  # segment + end


def test_replay_on_a_closed_session_is_refused(session):
    open_stream(session)
    session.close()
    with pytest.raises(PreviewStateError):
        session.replay(last_seq=0)


def test_replay_send_failures_are_counted_not_raised(sender):
    bad = ExplodingSender(fail_after=2)
    session = PreviewSession("7", sender=bad)
    open_stream(session)
    report = session.replay(last_seq=-1)
    assert report.sent == 0
    assert report.failed == 2


# -- resources -------------------------------------------------------------


def test_finalizers_run_exactly_once_in_lifo_order(session):
    calls: list[str] = []
    session.add_finalizer(lambda: calls.append("writer"), name="writer")
    session.add_finalizer(lambda: calls.append("encoder"), name="encoder")

    assert session.release() is True
    assert calls == ["encoder", "writer"]
    assert session.release() is False
    session.close()
    session.close()
    assert calls == ["encoder", "writer"]


def test_a_raising_finalizer_does_not_stop_the_others(session, caplog):
    calls: list[str] = []

    def boom() -> None:
        raise RuntimeError("close failed")

    session.add_finalizer(lambda: calls.append("first"), name="first")
    session.add_finalizer(boom, name="boom")
    with caplog.at_level(logging.WARNING):
        session.close()
    assert calls == ["first"]
    assert any("continuing cleanup" in r.getMessage() for r in caplog.records)


def test_late_finalizer_runs_immediately(session):
    calls: list[str] = []
    session.release()
    session.add_finalizer(lambda: calls.append("late"), name="late")
    assert calls == ["late"]


def test_close_drops_the_replay_payloads(session):
    open_stream(session)
    session.send_segment(b"x" * 4096)
    assert session.replay_bytes > 4096
    session.close()
    assert session.replay_span is None
    assert session.replay_bytes == 0


def test_session_holds_no_media_or_model_references(session):
    """A session must never be what keeps a tensor, model or muxer alive."""

    class FakeTensor:
        device = "cuda:0"

        def cuda(self):  # pragma: no cover - never called
            return self

    class FakeMuxer:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    tensor = FakeTensor()
    muxer = FakeMuxer()
    open_stream(session)
    session.send_segment(bytes(memoryview(b"frag-from-muxer")))
    session.add_finalizer(muxer.close, name="muxer")

    reachable = _reachable_objects(session)
    assert not any(isinstance(obj, FakeTensor) for obj in reachable)
    assert tensor is not None  # keep it alive so the check above is meaningful
    assert not any(hasattr(obj, "device") and hasattr(obj, "cuda") for obj in reachable)

    session.close()
    assert muxer.closed is True
    # After close the bound method is gone too.
    assert not any(obj is muxer for obj in _reachable_objects(session))


def _reachable_objects(root, depth: int = 4):
    """Objects reachable from ``root`` within ``depth`` hops of gc references."""
    seen = {id(root): root}
    frontier = [root]
    for _ in range(depth):
        nxt = []
        for obj in frontier:
            for ref in gc.get_referents(obj):
                if id(ref) not in seen:
                    seen[id(ref)] = ref
                    nxt.append(ref)
        frontier = nxt
    return list(seen.values())


# -- manager ---------------------------------------------------------------


def test_manager_context_sends_a_terminal_and_cleans_up(sender):
    clock = FakeClock()
    manager = PreviewManager(sender, clock=clock)
    with manager.session("7", client_id="cid") as session:
        open_stream(session)
        session.send_segment(b"frag")
    assert sender.events()[-1] == "end"
    assert sender.bodies[-1]["reason"] == "complete"
    assert manager.active_sessions == []
    assert [s.session_id for s in manager.terminal_sessions] == [session.session_id]


def test_manager_context_reports_errors_and_re_raises(sender):
    manager = PreviewManager(sender)
    with pytest.raises(ValueError):
        with manager.session("7") as session:
            open_stream(session)
            raise ValueError("sampler blew up")
    body = sender.bodies[-1]
    assert body["event"] == "end"
    assert body["reason"] == "error"
    assert "ValueError" in body["message"]


def test_manager_context_reports_cancellation_without_interrupting(sender):
    """Cancellation is observed here, never caused: no interrupt is invoked."""

    class InterruptProcessingException(Exception):
        pass

    manager = PreviewManager(sender)
    with pytest.raises(InterruptProcessingException):
        with manager.session("7") as session:
            open_stream(session)
            raise InterruptProcessingException()
    assert sender.bodies[-1]["reason"] == "cancelled"


def test_manager_context_keeps_an_explicit_terminal(sender):
    manager = PreviewManager(sender)
    with manager.session("7") as session:
        open_stream(session)
        session.cancelled(message="user cancelled")
    assert sender.events().count("end") == 1
    assert sender.bodies[-1]["reason"] == "cancelled"


def test_manager_context_releases_resources_on_every_path(sender):
    manager = PreviewManager(sender)
    released: list[str] = []
    with pytest.raises(RuntimeError):
        with manager.session("7") as session:
            session.add_finalizer(lambda: released.append("writer"), name="writer")
            raise RuntimeError("boom")
    assert released == ["writer"]


def test_cleanup_is_idempotent(sender):
    manager = PreviewManager(sender)
    session = manager.create_session("7")
    open_stream(session)
    assert manager.cleanup(session.session_id) is True
    assert manager.cleanup(session.session_id) is False
    assert manager.cleanup("never-existed") is False
    assert session.is_closed


def test_cleanup_can_send_a_terminal_for_a_disconnect(sender):
    manager = PreviewManager(sender)
    session = manager.create_session("7", client_id="cid")
    open_stream(session)
    manager.cleanup(session.session_id, reason="cancelled")
    assert sender.bodies[-1]["reason"] == "cancelled"
    # Second call must not emit a second terminal.
    manager.cleanup(session.session_id, reason="cancelled")
    assert sender.events().count("end") == 1


def test_terminal_sessions_expire_on_the_ttl(sender):
    clock = FakeClock()
    manager = PreviewManager(sender, clock=clock, terminal_ttl=60.0)
    with manager.session("7") as session:
        open_stream(session)
    assert manager.get(session.session_id) is session
    clock.advance(59)
    assert manager.prune() == []
    clock.advance(2)
    assert manager.prune() == [session.session_id]
    assert manager.get(session.session_id) is None
    assert session.is_closed


def test_abandoned_active_sessions_are_pruned(sender):
    clock = FakeClock()
    manager = PreviewManager(sender, clock=clock, active_idle_ttl=30.0)
    session = manager.create_session("7")
    open_stream(session)
    clock.advance(31)
    assert manager.prune() == [session.session_id]
    assert sender.bodies[-1]["reason"] == "error"
    assert session.is_closed


def test_a_new_run_supersedes_the_nodes_previous_session(sender):
    manager = PreviewManager(sender)
    first = manager.create_session("7", client_id="cid")
    open_stream(first)
    second = manager.create_session("7", client_id="cid")
    assert first.is_closed
    assert manager.active_sessions == [second]
    assert first.session_id != second.session_id


def test_shutdown_closes_everything(sender):
    manager = PreviewManager(sender)
    a = manager.create_session("7")
    b = manager.create_session("8")
    manager.shutdown()
    assert a.is_closed and b.is_closed
    assert manager.active_sessions == []
    manager.shutdown()  # idempotent


# -- sink ------------------------------------------------------------------


def test_sink_forwards_bytes_and_control_only(sender):
    session = PreviewSession("7", sender=sender)
    sink = PreviewMediaSink(session)
    assert sink.on_open(MIME, width=848, height=480, fps=24) is True
    assert sink.on_status("model_loading") is True
    assert sink.on_init(b"init-segment") is True
    assert sink.on_fragment(b"frag-0", keyframe=True, start=0.0, duration=0.875) is True
    assert sink.progress("sampling", value=1, maximum=40) is True
    assert sink.on_end("complete") is True
    assert sender.events() == ["open", "status", "init", "segment", "status", "end"]
    assert sender.bodies[3]["index"] == 0
    assert sender.bodies[4]["progress"] == {"value": 1, "max": 40}


def test_sink_isolates_every_failure(sender, caplog):
    session = PreviewSession("7", sender=sender)
    sink = PreviewMediaSink(session)
    with caplog.at_level(logging.WARNING):
        assert sink.on_fragment(b"frag") is False  # before open
        sink.on_open(MIME)
        assert sink.on_init(b"x" * (MAX_RAW_PAYLOAD_BYTES + 1)) is False
    assert sink.errors == 2
    assert session.state is SessionState.OPENED  # the run is untouched


def test_sink_fragment_callback_matches_the_muxer_shape(sender):
    class Segment:
        def __init__(self, kind, data, index=-1):
            self.kind = kind
            self.data = data
            self.index = index
            self.box_types = ()

    session = PreviewSession("7", sender=sender)
    sink = PreviewMediaSink(session)
    sink.on_open(MIME)
    callback = sink.fragment_callback()
    callback(Segment("init", b"ftyp+moov"))
    callback(Segment("fragment", b"moof+mdat", index=0))
    callback(Segment("trailer", b"mfra"))  # ignored: MSE has no use for it
    assert sender.events() == ["open", "init", "segment"]
    assert sender.bodies[-1]["index"] == 0
    assert sender.bodies[-1]["keyframe"] is True


def test_sink_pump_muxer_pulls_init_then_fragments(sender):
    class Segment:
        def __init__(self, data, index):
            self.data = data
            self.index = index

    class FakeMuxer:
        def __init__(self):
            self.init = b"ftyp+moov"
            self.fragments = [Segment(b"f0", 0), Segment(b"f1", 1)]

        def take_init_segment(self):
            init, self.init = self.init, None
            return init

        def take_fragments(self):
            out, self.fragments = self.fragments, []
            return out

    session = PreviewSession("7", sender=sender)
    sink = PreviewMediaSink(session)
    sink.on_open(MIME)
    muxer = FakeMuxer()
    assert sink.pump_muxer(muxer) == 3
    assert sender.events() == ["open", "init", "segment", "segment"]
    assert sink.pump_muxer(muxer) == 0
    assert session.segments_sent == 2


def test_sink_close_is_registered_as_a_finalizer_and_stops_forwarding(sender):
    session = PreviewSession("7", sender=sender)
    sink = PreviewMediaSink(session)
    sink.on_open(MIME)
    session.close()
    assert sink.closed is True
    assert sink.on_status("sampling") is False
    assert sender.events() == ["open"]
