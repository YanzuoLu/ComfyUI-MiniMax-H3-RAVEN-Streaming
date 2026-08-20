"""Incremental ISO-BMFF (MP4) box scanner and fragmented-MP4 segmenter.

The fragmented MP4 muxer writes into a non-seekable, write-only sink.  What
comes out is a byte stream, but a streaming consumer needs it split at *box*
boundaries:

    ftyp moov | (styp? sidx? moof mdat)* | mfra?
    \\_ init _/  \\______ fragment ______/

Only whole boxes may be handed to a consumer, so the split has to be done
incrementally as bytes arrive from the muxer.  This module does exactly that
and nothing else: it is pure ``bytes`` handling with no PyAV / torch imports,
so it can be unit tested anywhere.

Box header layout (ISO/IEC 14496-12):

* ``uint32 size`` then ``char[4] type``
* ``size == 1``  -> ``uint64 largesize`` follows the type (16 byte header)
* ``size == 0``  -> the box runs to end of stream (only legal for the last box)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Box",
    "IncrementalBoxParser",
    "FragmentedMP4Segmenter",
    "Segment",
    "iter_boxes",
    "MP4ParseError",
    "INIT_BOX_TYPES",
    "FRAGMENT_START_TYPES",
    "FRAGMENT_END_TYPES",
    "TRAILER_BOX_TYPES",
]


class MP4ParseError(ValueError):
    """Raised when the byte stream is not a well formed sequence of boxes."""


#: Boxes that make up the init segment (everything before the first fragment).
INIT_BOX_TYPES: Tuple[str, ...] = ("ftyp", "moov")

#: A new media fragment starts at one of these.  ``sidx``/``styp`` precede the
#: ``moof`` they belong to, so a fragment is only cut when the box in flight
#: already contains a ``moof``.
FRAGMENT_START_TYPES: Tuple[str, ...] = ("styp", "sidx", "moof")

#: A ``moof`` followed by a complete ``mdat`` *is* a media fragment (ISO/IEC
#: 14496-12 section 8.8).  Emitting on that pair rather than waiting for the
#: next ``moof`` is what keeps the segmenter from adding a whole fragment of
#: latency on top of the muxer's own.
FRAGMENT_END_TYPES: Tuple[str, ...] = ("mdat",)

#: Boxes emitted after the last fragment; not appendable media data.
TRAILER_BOX_TYPES: Tuple[str, ...] = ("mfra",)

_MIN_HEADER = 8
_LARGE_HEADER = 16


@dataclass(frozen=True)
class Box:
    """A complete top-level box."""

    type: str
    offset: int
    size: int
    header_size: int
    data: bytes

    @property
    def payload(self) -> bytes:
        return self.data[self.header_size:]

    @property
    def end(self) -> int:
        return self.offset + self.size


class IncrementalBoxParser:
    """Feed bytes in, get complete top-level boxes out."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._offset = 0  # absolute stream offset of self._buf[0]
        self._closed = False

    @property
    def consumed(self) -> int:
        """Absolute number of bytes already emitted as complete boxes."""
        return self._offset

    @property
    def pending_bytes(self) -> int:
        """Bytes buffered but not yet part of a complete box."""
        return len(self._buf)

    def feed(self, data: bytes) -> List[Box]:
        if self._closed:
            raise MP4ParseError("parser is closed")
        if not data:
            return []
        self._buf.extend(data)
        return self._drain()

    def close(self) -> List[Box]:
        """Finish the stream.

        Handles a trailing ``size == 0`` box (runs to EOF).  Any other leftover
        is a truncated box and raises.
        """
        if self._closed:
            return []
        boxes = self._drain()
        self._closed = True
        if len(self._buf) >= _MIN_HEADER:
            size, raw_type = struct.unpack_from(">I4s", self._buf, 0)
            if size == 0:
                box = Box(
                    type=_decode_type(raw_type),
                    offset=self._offset,
                    size=len(self._buf),
                    header_size=_MIN_HEADER,
                    data=bytes(self._buf),
                )
                self._offset += len(self._buf)
                self._buf.clear()
                boxes.append(box)
                return boxes
        if self._buf:
            raise MP4ParseError(
                "stream ended inside a box: {} trailing bytes at offset {}".format(
                    len(self._buf), self._offset
                )
            )
        return boxes

    # -- internals ----------------------------------------------------------

    def _drain(self) -> List[Box]:
        out: List[Box] = []
        while True:
            if len(self._buf) < _MIN_HEADER:
                return out
            size, raw_type = struct.unpack_from(">I4s", self._buf, 0)
            header = _MIN_HEADER
            if size == 1:
                if len(self._buf) < _LARGE_HEADER:
                    return out
                (size,) = struct.unpack_from(">Q", self._buf, 8)
                header = _LARGE_HEADER
            elif size == 0:
                # runs to EOF: only resolvable in close()
                return out
            if size < header:
                raise MP4ParseError(
                    "box {!r} at offset {} declares size {} < header {}".format(
                        _decode_type(raw_type), self._offset, size, header
                    )
                )
            if len(self._buf) < size:
                return out
            data = bytes(self._buf[:size])
            out.append(
                Box(
                    type=_decode_type(raw_type),
                    offset=self._offset,
                    size=size,
                    header_size=header,
                    data=data,
                )
            )
            del self._buf[:size]
            self._offset += size


def _decode_type(raw: bytes) -> str:
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


@dataclass(frozen=True)
class Segment:
    """An appendable byte range with the box types it contains."""

    kind: str  # "init" | "fragment" | "trailer"
    data: bytes
    box_types: Tuple[str, ...]
    offset: int
    index: int = -1

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)


class FragmentedMP4Segmenter:
    """Split a fragmented-MP4 byte stream into init + appendable fragments.

    Usage is intentionally pull-based so that it composes with any backpressure
    policy: :meth:`feed` never blocks and never drops data, the caller decides
    when to :meth:`take_init_segment` / :meth:`take_fragments`.
    """

    def __init__(self, on_fragment=None) -> None:
        """``on_fragment(segment)`` fires the instant a fragment is complete.

        The callback is what lets a caller timestamp fragment *production*
        (inside ``mux()``) rather than fragment *collection* (whenever it next
        polls :meth:`take_fragments`).
        """
        self._parser = IncrementalBoxParser()
        self._init_boxes: List[Box] = []
        self._init_segment: Optional[bytes] = None
        self._init_taken = False
        self._current: List[Box] = []
        self._fragments: List[Segment] = []
        self._trailer: List[Segment] = []
        self._closed = False
        self._fragment_index = 0
        self._on_fragment = on_fragment
        self._on_init = None

    # -- input --------------------------------------------------------------

    def feed(self, data: bytes) -> None:
        for box in self._parser.feed(data):
            self._accept(box)

    def close(self) -> None:
        if self._closed:
            return
        for box in self._parser.close():
            self._accept(box)
        self._flush_current()
        self._closed = True

    # -- output -------------------------------------------------------------

    @property
    def has_init_segment(self) -> bool:
        return self._init_segment is not None

    def peek_init_segment(self) -> Optional[bytes]:
        return self._init_segment

    def take_init_segment(self) -> Optional[bytes]:
        """Return the init segment (ftyp+moov) once; ``None`` until complete."""
        if self._init_segment is None or self._init_taken:
            return None
        self._init_taken = True
        return self._init_segment

    def take_fragments(self) -> List[Segment]:
        out = self._fragments
        self._fragments = []
        return out

    def take_trailer(self) -> List[Segment]:
        out = self._trailer
        self._trailer = []
        return out

    @property
    def pending_bytes(self) -> int:
        return self._parser.pending_bytes + sum(b.size for b in self._current)

    # -- internals ----------------------------------------------------------

    def _accept(self, box: Box) -> None:
        if self._init_segment is None:
            if box.type in INIT_BOX_TYPES or box.type in ("free", "skip", "wide"):
                self._init_boxes.append(box)
                if box.type == "moov":
                    self._init_segment = b"".join(b.data for b in self._init_boxes)
                return
            raise MP4ParseError(
                "unexpected box {!r} before moov (init segment incomplete)".format(box.type)
            )

        if box.type in TRAILER_BOX_TYPES:
            self._flush_current()
            self._trailer.append(
                Segment(
                    kind="trailer",
                    data=box.data,
                    box_types=(box.type,),
                    offset=box.offset,
                    index=len(self._trailer),
                )
            )
            return

        if box.type in FRAGMENT_START_TYPES and self._contains_moof():
            # unusual layout (e.g. moof with no mdat): cut at the next start box
            self._flush_current()
        self._current.append(box)
        if box.type in FRAGMENT_END_TYPES and self._contains_moof():
            # normal case: moof + mdat is a complete, appendable fragment
            self._flush_current()

    def _contains_moof(self) -> bool:
        return any(b.type == "moof" for b in self._current)

    def _flush_current(self) -> None:
        if not self._current:
            return
        boxes = self._current
        self._current = []
        segment = Segment(
            kind="fragment",
            data=b"".join(b.data for b in boxes),
            box_types=tuple(b.type for b in boxes),
            offset=boxes[0].offset,
            index=self._fragment_index,
        )
        self._fragments.append(segment)
        self._fragment_index += 1
        if self._on_fragment is not None:
            self._on_fragment(segment)


def iter_boxes(data: bytes) -> List[Box]:
    """Parse a complete in-memory MP4 into its top-level boxes."""
    parser = IncrementalBoxParser()
    boxes = parser.feed(data)
    boxes.extend(parser.close())
    return boxes


def box_types(data: bytes) -> Sequence[str]:
    return [b.type for b in iter_boxes(data)]
