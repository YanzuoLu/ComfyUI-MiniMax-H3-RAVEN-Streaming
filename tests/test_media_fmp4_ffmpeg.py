"""End-to-end check of the fMP4 segmenter against real muxer output.

PyAV is not available in every interpreter, but the *byte stream* it produces
is the same fragmented MP4 that the ``ffmpeg`` CLI produces with the same
``movflags``.  So when an ``ffmpeg`` binary is around we can falsify the
segmentation design without PyAV at all:

* the segmenter must reconstruct the original file byte for byte,
* every fragment must start on a ``moof``,
* ``init + fragments[i]`` alone must decode, and its first frame must be a
  keyframe - which is exactly what a streaming consumer relies on.

Skips (loudly) when ``ffmpeg``/``ffprobe`` are absent.  Artifacts go under the
gitignored ``.cache/`` directory inside the project.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from raven_streaming.media.mp4_boxes import FragmentedMP4Segmenter, iter_boxes  # noqa: E402
from raven_streaming.media.mp4_writer import DEFAULT_MOVFLAGS  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH"
)

WIDTH = HEIGHT = 128
FPS = 24
SEGMENT_FRAMES = 12
DURATION = 3
WORKDIR = os.path.join(PROJECT_ROOT, ".cache", "m0probe")


@pytest.fixture(scope="module")
def fragmented_mp4() -> bytes:
    os.makedirs(WORKDIR, exist_ok=True)
    path = os.path.join(WORKDIR, "ffmpeg_frag.mp4")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size={}x{}:rate={}:duration={}".format(
            WIDTH, HEIGHT, FPS, DURATION),
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration={}".format(DURATION),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-g", str(SEGMENT_FRAMES),
        "-force_key_frames", "expr:eq(mod(n,{}),0)".format(SEGMENT_FRAMES),
        "-c:a", "aac", "-ar", "32000", "-ac", "2",
        "-movflags", DEFAULT_MOVFLAGS,
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        pytest.skip("ffmpeg could not produce a fragmented MP4: {}".format(
            proc.stderr.decode("utf-8", "replace")[:400]))
    with open(path, "rb") as handle:
        return handle.read()


def _segment(data: bytes, step: int):
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
    return init, fragments, seg.take_trailer()


def test_real_fmp4_has_the_expected_box_layout(fragmented_mp4):
    types = [b.type for b in iter_boxes(fragmented_mp4)]
    assert types[0] == "ftyp"
    assert types[1] == "moov"
    assert types[2:4] == ["moof", "mdat"]
    assert types.count("moof") >= 2


@pytest.mark.parametrize("step", [1, 3, 997, 65536, 1 << 30])
def test_segmenter_reconstructs_the_file_byte_for_byte(fragmented_mp4, step):
    init, fragments, trailer = _segment(fragmented_mp4, step)
    rebuilt = init + b"".join(f.data for f in fragments) + b"".join(t.data for t in trailer)
    assert rebuilt == fragmented_mp4
    assert all(f.data[4:8] == b"moof" for f in fragments)
    assert len(fragments) >= 2


def test_segment_counts_are_independent_of_chunking(fragmented_mp4):
    baseline = _segment(fragmented_mp4, 1 << 30)
    for step in (1, 7, 4096):
        init, fragments, trailer = _segment(fragmented_mp4, step)
        assert init == baseline[0]
        assert [f.data for f in fragments] == [f.data for f in baseline[1]]
        assert [t.data for t in trailer] == [t.data for t in baseline[2]]


def _ffprobe(path: str, *entries: str) -> str:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0"] + list(entries)
        + ["-of", "csv=p=0", path],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return proc.stdout.decode("utf-8", "replace")


def test_each_fragment_decodes_standalone_after_the_init_segment(fragmented_mp4):
    init, fragments, _ = _segment(fragmented_mp4, 1 << 30)
    frag_dir = os.path.join(WORKDIR, "fragments")
    os.makedirs(frag_dir, exist_ok=True)

    for idx, frag in enumerate(fragments):
        path = os.path.join(frag_dir, "frag{:02d}.mp4".format(idx))
        with open(path, "wb") as handle:
            handle.write(init + frag.data)

        counted = _ffprobe(path, "-count_frames", "-show_entries", "stream=nb_read_frames")
        frames = int(counted.strip().split(",")[0])
        assert frames > 0, "fragment {} decoded to nothing".format(idx)

        first = _ffprobe(path, "-show_frames", "-show_entries", "frame=key_frame")
        first_key = first.strip().splitlines()[0].split(",")[0]
        assert first_key == "1", "fragment {} does not start on a keyframe".format(idx)


def test_growing_prefix_never_loses_frames(fragmented_mp4):
    init, fragments, _ = _segment(fragmented_mp4, 1 << 30)
    prefix_dir = os.path.join(WORKDIR, "prefixes")
    os.makedirs(prefix_dir, exist_ok=True)

    seen = 0
    for n in range(1, len(fragments) + 1):
        path = os.path.join(prefix_dir, "prefix{:02d}.mp4".format(n))
        with open(path, "wb") as handle:
            handle.write(init + b"".join(f.data for f in fragments[:n]))
        counted = _ffprobe(path, "-count_frames", "-show_entries", "stream=nb_read_frames")
        frames = int(counted.strip().split(",")[0])
        assert frames >= seen, "prefix of {} fragments lost frames".format(n)
        seen = frames
    assert seen == FPS * DURATION
