"""RAVEN streaming media lane (M0).

The M0 scope is deliberately *falsification-first*: before any UI or transport
exists, three things have to be shown to actually work.

1. ``mp4_boxes`` / ``mp4_writer`` / ``codecs`` - fragmented MP4 out of PyAV into
   a custom, non-seekable sink, split into an init segment plus appendable
   fragments, with a forced IDR at every segment boundary.
2. ``video_stream`` - an incremental coordinator for the MiniMax H3 video VAE
   temporal chunk machine that never allocates the full output tensor.
3. ``audio_stream`` - an overlap-save planner for the non-causal audio VAE,
   plus a tool that *measures* the required lookahead instead of guessing it.

``clock`` ties the two lanes to one exact integer tick grid.

Submodules that need PyAV (:mod:`codecs`, :mod:`mp4_writer`) import it lazily,
so importing this package never requires PyAV, torch, or numpy.
"""

from __future__ import annotations

from .audio_stream import (
    AudioLatentGeometry,
    DecodeRequest,
    MarginSearchResult,
    OverlapSaveAudioDecoder,
    OverlapSavePlanner,
    decode_overlap_save,
    max_abs_diff,
    search_latent_margin,
)
from .clock import (
    AUDIO_SAMPLES_PER_LATENT,
    RAVEN_CLOCK,
    VIDEO_FRAMES_PER_CHUNK,
    VIDEO_LATENTS_PER_CHUNK,
    AVChunkAlignment,
    MediaClock,
    StreamCursor,
)
from .mp4_boxes import (
    Box,
    FragmentedMP4Segmenter,
    IncrementalBoxParser,
    MP4ParseError,
    Segment,
    iter_boxes,
)
from .video_stream import (
    ComparisonVerdict,
    FrameBatch,
    IncrementalVideoDecoder,
    VideoChunkParams,
    minimax_decoder_adapter,
    reference_decode_temporal,
    summarize_decode_comparison,
)

__all__ = [
    # clock
    "MediaClock",
    "StreamCursor",
    "AVChunkAlignment",
    "RAVEN_CLOCK",
    "VIDEO_LATENTS_PER_CHUNK",
    "VIDEO_FRAMES_PER_CHUNK",
    "AUDIO_SAMPLES_PER_LATENT",
    # mp4
    "Box",
    "Segment",
    "IncrementalBoxParser",
    "FragmentedMP4Segmenter",
    "MP4ParseError",
    "iter_boxes",
    # video
    "VideoChunkParams",
    "IncrementalVideoDecoder",
    "FrameBatch",
    "reference_decode_temporal",
    "minimax_decoder_adapter",
    "ComparisonVerdict",
    "summarize_decode_comparison",
    # audio
    "AudioLatentGeometry",
    "DecodeRequest",
    "OverlapSavePlanner",
    "OverlapSaveAudioDecoder",
    "MarginSearchResult",
    "search_latent_margin",
    "decode_overlap_save",
    "max_abs_diff",
]
