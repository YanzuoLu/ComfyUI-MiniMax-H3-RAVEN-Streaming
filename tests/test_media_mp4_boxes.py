"""Tests for incremental ISO-BMFF box scanning and fMP4 segmentation.

Pure bytes: these run in any interpreter, with or without PyAV.  The tests that
need a real muxer live in ``test_media_mp4_writer.py``.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming.media.mp4_boxes import (  # noqa: E402
    FragmentedMP4Segmenter,
    IncrementalBoxParser,
    MP4ParseError,
    iter_boxes,
)


# -- synthetic box streams -------------------------------------------------


def box(box_type: str, payload: bytes = b"", large: bool = False) -> bytes:
    raw = box_type.encode("ascii")
    if large:
        return struct.pack(">I4sQ", 1, raw, 16 + len(payload)) + payload
    return struct.pack(">I4s", 8 + len(payload), raw) + payload


def test_parses_simple_boxes():
    data = box("ftyp", b"isom") + box("moov", b"x" * 40)
    boxes = iter_boxes(data)
    assert [b.type for b in boxes] == ["ftyp", "moov"]
    assert boxes[0].payload == b"isom"
    assert boxes[1].size == 48
    assert boxes[1].offset == 12
    assert boxes[1].end == 60


def test_parses_64bit_largesize_boxes():
    data = box("mdat", b"y" * 100, large=True)
    boxes = iter_boxes(data)
    assert boxes[0].type == "mdat"
    assert boxes[0].header_size == 16
    assert boxes[0].size == 116
    assert boxes[0].payload == b"y" * 100


def test_size_zero_box_runs_to_eof():
    data = box("ftyp", b"isom") + struct.pack(">I4s", 0, b"mdat") + b"z" * 30
    boxes = iter_boxes(data)
    assert [b.type for b in boxes] == ["ftyp", "mdat"]
    assert boxes[1].size == 38


@pytest.mark.parametrize("step", [1, 2, 3, 7, 13, 64, 100000])
def test_parsing_is_byte_granular(step):
    """Boxes must come out identically no matter how the stream is chopped."""
    data = box("ftyp", b"isom") + box("moov", b"m" * 50) + box("moof", b"f" * 20) + box(
        "mdat", b"d" * 200, large=True
    )
    parser = IncrementalBoxParser()
    boxes = []
    for i in range(0, len(data), step):
        boxes.extend(parser.feed(data[i:i + step]))
    boxes.extend(parser.close())
    assert [b.type for b in boxes] == ["ftyp", "moov", "moof", "mdat"]
    assert b"".join(b.data for b in boxes) == data
    assert parser.consumed == len(data)
    assert parser.pending_bytes == 0


def test_truncated_box_is_reported():
    parser = IncrementalBoxParser()
    parser.feed(box("ftyp", b"isom") + box("moov", b"m" * 20)[:10])
    with pytest.raises(MP4ParseError):
        parser.close()


def test_bogus_size_is_reported():
    parser = IncrementalBoxParser()
    with pytest.raises(MP4ParseError):
        parser.feed(struct.pack(">I4s", 4, b"ftyp") + b"junkjunk")


# -- fragment segmentation -------------------------------------------------


def _fmp4_bytes(fragments: int = 3) -> bytes:
    data = box("ftyp", b"isom") + box("moov", b"m" * 30)
    for i in range(fragments):
        data += box("moof", bytes([i]) * 24) + box("mdat", bytes([i]) * 128)
    data += box("mfra", b"r" * 16)
    return data


@pytest.mark.parametrize("step", [1, 5, 17, 64, 1000000])
def test_segmenter_splits_init_and_fragments(step):
    data = _fmp4_bytes(3)
    seg = FragmentedMP4Segmenter()
    init = None
    fragments = []
    for i in range(0, len(data), step):
        seg.feed(data[i:i + step])
        init = init or seg.take_init_segment()
        fragments.extend(seg.take_fragments())
    seg.close()
    init = init or seg.take_init_segment()
    fragments.extend(seg.take_fragments())
    trailer = seg.take_trailer()

    assert init == box("ftyp", b"isom") + box("moov", b"m" * 30)
    assert len(fragments) == 3
    for frag in fragments:
        assert frag.box_types == ("moof", "mdat")
        assert frag.data[4:8] == b"moof"  # every fragment starts at a moof
    assert len(trailer) == 1 and trailer[0].box_types == ("mfra",)
    # init + fragments + trailer reconstructs the original stream
    assert init + b"".join(f.data for f in fragments) + trailer[0].data == data


def test_init_segment_is_only_handed_out_once():
    seg = FragmentedMP4Segmenter()
    seg.feed(_fmp4_bytes(1))
    assert seg.take_init_segment() is not None
    assert seg.take_init_segment() is None
    assert seg.has_init_segment


def test_styp_and_sidx_stay_with_their_fragment():
    data = box("ftyp", b"isom") + box("moov", b"m" * 10)
    data += box("styp", b"msdh") + box("sidx", b"s" * 20) + box("moof", b"a" * 8) + box("mdat", b"a" * 16)
    data += box("styp", b"msdh") + box("moof", b"b" * 8) + box("mdat", b"b" * 16)
    seg = FragmentedMP4Segmenter()
    seg.feed(data)
    seg.close()
    seg.take_init_segment()
    frags = seg.take_fragments()
    assert len(frags) == 2
    assert frags[0].box_types == ("styp", "sidx", "moof", "mdat")
    assert frags[1].box_types == ("styp", "moof", "mdat")


def test_media_data_before_moov_is_rejected():
    seg = FragmentedMP4Segmenter()
    with pytest.raises(MP4ParseError):
        seg.feed(box("ftyp", b"isom") + box("moof", b"x" * 8))
