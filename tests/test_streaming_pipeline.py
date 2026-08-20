"""Behaviour of the streaming decode / mux / preview pipeline.

Everything here runs without ComfyUI. Nothing is faked *into* ``sys.modules``:
the coordinator, the incremental video decoder, the overlap-save audio decoder,
the preview session and the media sink are all the real implementations. Only
the two VAE decoders and the muxer are stand-ins, because those are the pieces
that need a checkpoint and PyAV -- and the one test that does want the real
muxer asks for it and skips when PyAV cannot provide an encoder.

The load-bearing claims:

* a chunk's frames are not visible until the *next* chunk's latents arrive
  (the video VAE's 2-latent lookahead) and the clip ends with a 5-frame flush;
* no frame is muxed before real decoded PCM covers the same media-clock range,
  and silence is never substituted for audio that has not arrived;
* every failure mode of the preview lane -- decode, mux, oversized payload,
  dead socket -- stops the preview and nothing else;
* the pipeline holds no tensor from the sampler's device once ``on_chunk``
  returns.
"""

from __future__ import annotations

import gc
import os
import sys
import weakref

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as layout_mod  # noqa: E402
from raven_streaming import streaming_pipeline as sp  # noqa: E402
from raven_streaming.consistency import ChunkOutput  # noqa: E402
from raven_streaming.streaming_pipeline import _to_numpy  # noqa: E402
from raven_streaming.media.audio_stream import (  # noqa: E402
    AudioLatentGeometry,
    OverlapSaveAudioDecoder,
)
from raven_streaming.media.clock import RAVEN_CLOCK  # noqa: E402
from raven_streaming.media.fakes import (  # noqa: E402
    FakeVideoChunkDecoder,
    FiniteRFAudioDecoder,
)
from raven_streaming.media.video_stream import (  # noqa: E402
    IncrementalVideoDecoder,
    minimax_decoder_adapter,
    reference_decode_temporal,
)
from raven_streaming.preview_session import (  # noqa: E402
    MAX_RAW_PAYLOAD_BYTES,
    PreviewMediaSink,
    PreviewSession,
    RecordingSender,
)

# A 39-frame clip (k = 2): 12 video latents cut 5 / 5 / 2, and 65 audio latents
# cut 29 / 28 / 8 by the shared 85/3 clock. Long enough to have a steady state.
FRAMES = 39
VIDEO_CHUNKS = (5, 5, 2)
AUDIO_CHUNKS = (29, 28, 8)
SPATIAL = 8  # fake decoder upscale: 2x2 latents -> 16x16 pixels
WIDTH = HEIGHT = 16


# --------------------------------------------------------------------------
# fakes for the two pieces that would need a checkpoint / PyAV
# --------------------------------------------------------------------------


class FakeSegment:
    """What ``FragmentedMP4Segmenter`` hands back, structurally."""

    def __init__(self, index: int, data: bytes) -> None:
        self.kind = "fragment"
        self.index = index
        self.data = data


class FakeMuxer:
    """Records what was muxed and emits one fragment per video frame."""

    def __init__(self, fragment_bytes: int = 96, fail_on_frame=None) -> None:
        self.frames = []          # [(array, force_keyframe)]
        self.audio_blocks = []    # [np.ndarray[2, n]]
        self.fragment_bytes = int(fragment_bytes)
        self.fail_on_frame = fail_on_frame
        self.closes = 0
        self._pending = []
        self._index = 0
        self._init_taken = False

    # -- muxer surface --------------------------------------------------

    def write_video_frame(self, image, force_keyframe=None):
        if self.fail_on_frame is not None and len(self.frames) == self.fail_on_frame:
            raise RuntimeError("encoder exploded")
        self.frames.append((image, force_keyframe))
        self._pending.append(
            FakeSegment(self._index, bytes([self._index % 251]) * self.fragment_bytes)
        )
        self._index += 1

    def write_audio(self, pcm):
        self.audio_blocks.append(np.array(pcm, copy=True))

    def take_init_segment(self):
        if self._init_taken or not self.frames:
            return None
        self._init_taken = True
        return b"ftyp+moov"

    def take_fragments(self):
        out = self._pending
        self._pending = []
        return out

    def close(self):
        self.closes += 1

    # -- assertions -----------------------------------------------------

    @property
    def samples_written(self) -> int:
        return int(sum(block.shape[-1] for block in self.audio_blocks))

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def concatenated_audio(self):
        if not self.audio_blocks:
            return np.zeros((2, 0), dtype=np.float32)
        return np.concatenate(self.audio_blocks, axis=-1)


class NoAbortDecoder:
    """A decoder from before ``abort`` existed: ``push`` and ``finish``, full stop.

    ``getattr(decoder, "abort", None)`` has to come back ``None`` for this, so
    nothing here delegates by ``__getattr__`` -- that would hand back the inner
    decoder's ``abort`` and the test would prove nothing.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    @property
    def planner(self):  # what _samples_per_latent() reads off an audio decoder
        return self.inner.planner

    def push(self, z):
        return self.inner.push(z)

    def finish(self):
        return self.inner.finish()


class AbortRaisesDecoder(NoAbortDecoder):
    """...and one whose ``abort`` raises, which cleanup must survive."""

    def __init__(self, inner) -> None:
        super().__init__(inner)
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1
        raise RuntimeError("abort exploded")


class TrackedTensor:
    """A stand-in for a tensor that lives on the compute device.

    ``detach()``/``cpu()`` hand back a plain CPU tensor, so anything that keeps
    a ``TrackedTensor`` alive kept the *device* tensor alive.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.detach_calls = 0
        self.cpu_calls = 0

    def detach(self):
        self.detach_calls += 1
        return _CpuHop(self._tensor, self)

    def __getattr__(self, name):  # pragma: no cover - diagnostics only
        return getattr(self._tensor, name)


class _CpuHop:
    def __init__(self, tensor, owner):
        self._tensor = tensor
        self._owner = owner

    def cpu(self):
        self._owner.cpu_calls += 1
        return self._tensor


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def make_config(**kwargs) -> sp.PipelineConfig:
    params = dict(frames=FRAMES, width=WIDTH, height=HEIGHT)
    params.update(kwargs)
    return sp.PipelineConfig(**params)


def make_video_decoder(spatial: int = SPATIAL) -> IncrementalVideoDecoder:
    return IncrementalVideoDecoder(
        FakeVideoChunkDecoder(vae_ratio_t=4, out_channels=3, spatial_scale=spatial)
    )


def make_audio_decoder(config: sp.PipelineConfig) -> OverlapSaveAudioDecoder:
    return OverlapSaveAudioDecoder(
        FiniteRFAudioDecoder(radius=2, samples_per_latent=800, latent_channels=32),
        margin=config.audio_margin_latents,
        block_latents=config.audio_block_latents,
        geometry=AudioLatentGeometry(800, 32000),
    )


def make_sink(sender=None):
    sender = sender if sender is not None else RecordingSender()
    session = PreviewSession("7", sender=sender, client_id="cid", prompt_id="p1")
    return PreviewMediaSink(session), session, sender


def make_pipeline(
    config=None,
    muxer=None,
    sink=None,
    spatial: int = SPATIAL,
    video_decoder=None,
    audio_decoder=None,
    preview: bool = True,
):
    """A pipeline whose two collectors are always real; the preview is optional."""
    config = config if config is not None else make_config()
    if preview:
        muxer = muxer if muxer is not None else FakeMuxer()
        if sink is None:
            sink, _session, _sender = make_sink()
    else:
        muxer = None
        sink = None
    return (
        sp.StreamingPipeline(
            config=config,
            video_decoder=(
                video_decoder if video_decoder is not None else make_video_decoder(spatial)
            ),
            # always built: it is the AUDIO output, not part of the preview
            audio_decoder=audio_decoder if audio_decoder is not None else make_audio_decoder(config),
            muxer=muxer,
            sink=sink,
        ),
        muxer,
        sink,
    )


def run_pipeline(pipeline, chunks: int = 3):
    """Feed a whole clip and finish. Returns the finished IMAGE."""
    pipeline.open_preview()
    for index in range(chunks):
        pipeline.on_chunk(make_chunk(index))
    pipeline.finish()
    return pipeline.finalize_image()


def run_pipeline_av(pipeline, chunks: int = 3, vae=None):
    """Feed a whole clip and finish. Returns ``(IMAGE, AUDIO)``."""
    image = run_pipeline(pipeline, chunks)
    return image, pipeline.finalize_audio(vae if vae is not None else FakeAudioVAE())


class FakeAudioVAE:
    """Only what ``finalize_audio`` reads off a ``comfy.sd.VAE``."""

    def __init__(self, sample_rate=32000, output_rate=None):
        self.audio_sample_rate = sample_rate
        if output_rate is not None:
            self.audio_sample_rate_output = output_rate


def chunk_latents(index: int, wrap=None):
    """One chunk's ``(video, audio)`` latents, deterministic per index."""
    generator = torch.Generator().manual_seed(1000 + index)
    video = torch.randn(
        (1, 24, VIDEO_CHUNKS[index], 2, 2), generator=generator, dtype=torch.float32
    )
    audio = torch.randn(
        (1, 32, 2, AUDIO_CHUNKS[index]), generator=generator, dtype=torch.float32
    )
    if wrap is not None:
        video, audio = wrap(video), wrap(audio)
    return video, audio


def clip_chunks(frames: int, latent_size: int = 2, seed: int = 7):
    """Every chunk of a whole clip, on the real T2VA grid."""
    plan = layout_mod.T2VALayout.from_request(
        text_len=8,
        frames=frames,
        width=latent_size * 16,
        height=latent_size * 16,
        warn_experimental=False,
    )
    for chunk in plan.chunks:
        generator = torch.Generator().manual_seed(seed + chunk.index)
        yield ChunkOutput(
            index=chunk.index,
            is_last=chunk.index == len(plan.chunks) - 1,
            video_start=chunk.video_start,
            video_stop=chunk.video_stop,
            audio_start=chunk.audio_start,
            audio_stop=chunk.audio_stop,
            video_x0=torch.randn(
                (1, 24, chunk.video_latents, latent_size, latent_size),
                generator=generator,
                dtype=torch.float32,
            ),
            audio_x0=torch.randn(
                (1, 32, 2, chunk.audio_latents), generator=generator, dtype=torch.float32
            ),
        )


def make_chunk(index: int, wrap=None) -> ChunkOutput:
    video, audio = chunk_latents(index, wrap=wrap)
    video_start = sum(VIDEO_CHUNKS[:index])
    audio_start = sum(AUDIO_CHUNKS[:index])
    return ChunkOutput(
        index=index,
        is_last=index == len(VIDEO_CHUNKS) - 1,
        video_start=video_start,
        video_stop=video_start + VIDEO_CHUNKS[index],
        audio_start=audio_start,
        audio_stop=audio_start + AUDIO_CHUNKS[index],
        video_x0=video,
        audio_x0=audio,
    )


# --------------------------------------------------------------------------
# published constants
# --------------------------------------------------------------------------


def test_audio_overlap_constants_are_the_measured_ones():
    import inspect

    assert sp.AUDIO_MARGIN_LATENTS == 17
    assert sp.AUDIO_BLOCK_LATENTS == 28
    geometry = AudioLatentGeometry(800, 32000)
    # 17 latents of lookahead is 0.425 s, which is what the preview's audio
    # trails its video by. If the constant moves, this number must move with it.
    assert geometry.latents_to_samples(sp.AUDIO_MARGIN_LATENTS) == 13600
    assert float(geometry.latents_to_seconds(sp.AUDIO_MARGIN_LATENTS)) == pytest.approx(0.425)
    # ... and the number must stay attached to the measurement it came from,
    # rather than becoming a constant nobody can re-derive.
    source = inspect.getsource(sp)
    assert "probe_audio_overlap" in source
    assert "2.5e-6" in source


def test_preview_uses_one_fragment_per_frame():
    config = make_config()
    assert config.fragment_mode == "every_frame"
    muxer_config = sp.build_muxer_config(config)
    assert muxer_config.fragment_mode == "every_frame"
    assert "frag_every_frame" in muxer_config.movflags
    # a fragment per frame is not a seek point per frame; only the forced IDRs
    # are, one per 17-frame video chunk
    assert muxer_config.segment_frames == 17
    assert muxer_config.fragments_are_independently_decodable is False
    assert muxer_config.container_options()["min_frag_duration"] == "1"


def test_open_carries_the_media_description():
    pipeline, _muxer, sink = make_pipeline()
    assert pipeline.open_preview() is True
    body = sink.session._replay[0].body
    assert body["event"] == "open"
    assert body["mime"] == sp.PREVIEW_MIME
    assert body["width"] == WIDTH and body["height"] == HEIGHT
    assert body["fps"] == 24.0
    assert body["audio"] == {"sample_rate": 32000, "channels": 2}
    assert body["duration_hint"] == pytest.approx(FRAMES / 24.0)


# --------------------------------------------------------------------------
# the video lookahead and the tail flush
# --------------------------------------------------------------------------


def test_first_frames_need_the_second_sampling_chunk():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()

    pipeline.on_chunk(make_chunk(0))
    # 5 latents is one chunk of latents but not one chunk of *frames*: the
    # decoder finalizes 17 frames only once the 2-latent lookahead exists.
    assert pipeline.frames_decoded == 0
    assert muxer.frame_count == 0

    pipeline.on_chunk(make_chunk(1))
    assert pipeline.frames_decoded == 17


def test_finish_flushes_the_five_frame_tail():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    # chunks 0..2 = 12 latents = two 17-frame chunks; the last 5 frames are the
    # decoder's held-back overlap and only exist after finish()
    assert pipeline.frames_decoded == 34
    report = pipeline.finish()
    assert pipeline.frames_decoded == FRAMES
    assert report.frames_muxed == FRAMES
    assert muxer.frame_count == FRAMES


def test_forced_idr_lands_on_every_video_chunk_boundary():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    pipeline.finish()
    forced = [i for i, (_frame, key) in enumerate(muxer.frames) if key]
    assert forced == [0, 17, 34]


def test_muxed_frames_are_hwc_pixel_arrays():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    pipeline.finish()
    frame, _key = muxer.frames[0]
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (HEIGHT, WIDTH, 3)


# --------------------------------------------------------------------------
# the A/V coordinator
# --------------------------------------------------------------------------


def test_audio_blocks_wait_for_their_right_context():
    pipeline, _muxer, _sink = make_pipeline()
    pipeline.open_preview()

    pipeline.on_chunk(make_chunk(0))
    # a 28-latent block needs 17 latents of right context: 29 seen is not enough
    assert pipeline.samples_decoded == 0

    pipeline.on_chunk(make_chunk(1))
    assert pipeline.samples_decoded == 28 * 800


def test_video_never_outruns_the_decoded_audio():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))

    # 17 frames are decoded, 22400 samples of real PCM exist. Frame 16 covers
    # the clock through 17 * 4000 / 3 = 22666 samples, which have not been
    # decoded yet, so it stays queued.
    assert pipeline.frames_decoded == 17
    assert pipeline.samples_available == 22400
    assert RAVEN_CLOCK.samples_for_frames(17) == 22666
    assert muxer.frame_count == 16
    assert pipeline.pending_frames == 1
    assert muxer.samples_written == RAVEN_CLOCK.samples_for_frames(16)

    # a third chunk brings more video but no new audio block: still gated
    pipeline.on_chunk(make_chunk(2))
    assert pipeline.frames_decoded == 34
    assert pipeline.samples_available == 22400
    assert muxer.frame_count == 16


def test_no_silence_is_ever_substituted_for_missing_audio():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    # every sample handed to the muxer came out of the decoder, and every
    # decoded sample was handed over exactly once
    assert report.samples_decoded == 65 * 800
    assert muxer.samples_written == report.samples_decoded == report.samples_muxed

    # the reference decode of the whole latent sequence, sample for sample
    decoder = FiniteRFAudioDecoder(radius=2, samples_per_latent=800, latent_channels=32)
    latents = torch.cat([chunk_latents(i)[1] for i in range(3)], dim=-1)
    reference = decoder(latents)[0]
    written = muxer.concatenated_audio()
    assert written.shape[-1] == reference.shape[-1]
    assert np.max(np.abs(written - reference)) < 1e-4
    # ... and no run of inserted zeros: a silence patch would show up as a
    # block that is exactly zero while the reference is not
    assert not np.any(
        np.all(written == 0.0, axis=0) & (np.max(np.abs(reference), axis=0) > 1e-6)
    )


def test_the_two_lanes_stay_on_one_integer_clock():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
        # at every point the audio written covers exactly the frames written
        assert muxer.samples_written == min(
            RAVEN_CLOCK.samples_for_frames(muxer.frame_count),
            pipeline.samples_available,
        )
    pipeline.finish()


# --------------------------------------------------------------------------
# the preview lane
# --------------------------------------------------------------------------


def test_init_and_segments_reach_the_sink_in_order():
    sink, session, sender = make_sink()
    pipeline, muxer, _sink = make_pipeline(sink=sink)
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    events = sender.events()
    assert events[0] == "open"
    assert events[1] == "init"
    assert set(events[2:]) == {"segment"}
    assert events.count("segment") == report.fragments == FRAMES
    assert sender.seqs() == list(range(len(events)))

    # the fragments are sent as base64 of exactly the muxer's bytes
    assert report.init_bytes == len(b"ftyp+moov")
    assert report.fragment_sizes == [96] * FRAMES
    segments = [b for b in sender.bodies if b["event"] == "segment"]
    assert [b["index"] for b in segments] == list(range(FRAMES))
    # every_frame fragments are append-only, so none of them claims to be a
    # seek point
    assert not any(b.get("keyframe") for b in segments)
    assert report.send_failures == 0
    assert report.first_fragment_chunk == 1
    assert report.first_fragment_latency is not None


# --------------------------------------------------------------------------
# emit as it is produced: the whole point of the lane
# --------------------------------------------------------------------------


def test_each_chunk_sends_everything_it_could_send(  # noqa: C901 - one story
):
    """Whatever a chunk decodes is muxed and pushed inside that callback."""
    sink, _session, sender = make_sink()
    pipeline, muxer, _sink = make_pipeline(sink=sink)
    pipeline.open_preview()

    sent_before = 0
    for index in range(3):
        produced_before = muxer._index  # fragments the muxer has produced
        pipeline.on_chunk(make_chunk(index))

        # nothing the muxer produced during this callback is still sitting in
        # it: the pump ran before on_chunk returned
        assert muxer.take_fragments() == [], f"chunk {index} left fragments unsent"
        sent_now = len([b for b in sender.bodies if b["event"] == "segment"])
        assert sent_now - sent_before == muxer._index - produced_before
        sent_before = sent_now

        # ... and every frame that could be muxed was: what is left is exactly
        # the frames whose audio has not been decoded yet
        for held_index, _frame in pipeline._frames:
            needed = RAVEN_CLOCK.samples_for_frames(held_index + 1)
            assert needed > pipeline.samples_available


def streamable_frames(frames: int, config=None) -> int:
    """Frames that *can* be muxed before ``finish()``, derived not guessed.

    Overlap-save needs ``margin`` latents of right context, so the last
    ``margin`` latents of the clip have no audio until the stream ends and the
    planner emits its edge blocks. Everything before that is fair game::

        blocks during the run = floor((audio_latents - margin) / block)
        samples               = blocks * block * 800
        frames                = the frames that many samples cover on the clock

    The 5-frame video tail is behind that same line: it only exists after the
    decoder's own flush.
    """
    config = config or sp.PipelineConfig(frames=frames, width=WIDTH, height=HEIGHT)
    blocks = (config.audio_latents - config.audio_margin_latents) // config.audio_block_latents
    samples = blocks * config.audio_block_latents * config.samples_per_latent
    covered = 0
    while covered < frames and RAVEN_CLOCK.samples_for_frames(covered + 1) <= samples:
        covered += 1
    return covered


def test_the_stream_does_not_wait_for_finish():
    sink, _session, sender = make_sink()
    pipeline, _muxer, _sink = make_pipeline(sink=sink)
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))

    before_finish = len([b for b in sender.bodies if b["event"] == "segment"])
    # everything the audio clock could cover has already gone out; what is left
    # is the audio the overlap-save margin cannot decode until the clip ends
    assert before_finish == streamable_frames(FRAMES)

    report = pipeline.finish()
    assert report.fragments_before_finish == before_finish
    assert report.frames_muxed == FRAMES


def test_a_long_clip_streams_almost_all_of_itself_before_finish():
    """The short clip is the worst case; a real one is nearly all steady state.

    39 frames is three chunks, which is about as long as the audio lane's own
    latency -- so a big fraction of it is inherently end-loaded. At 192 frames
    the steady state dominates and the run streams as it samples, which is what
    the lane is for.
    """
    frames = 192
    config = sp.PipelineConfig(frames=frames, width=2, height=2)
    sink, _session, sender = make_sink()
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=make_video_decoder(spatial=1),
        audio_decoder=make_audio_decoder(config),
        muxer=FakeMuxer(),
        sink=sink,
    )
    pipeline.open_preview()
    for chunk in clip_chunks(frames):
        pipeline.on_chunk(chunk)
    report = pipeline.finish()

    expected = streamable_frames(frames, config)
    assert report.fragments_before_finish == expected
    assert expected >= 0.85 * frames, expected
    # and it arrived spread over the run, not in one lump: every chunk from the
    # second onwards sent something
    streaming_chunks = [e for e in report.chunk_emissions if e.fragments > 0]
    assert len(streaming_chunks) >= len(report.chunk_emissions) - 2
    assert max(e.fragments for e in report.chunk_emissions) <= 2 * 17


def test_the_first_frames_go_out_one_chunk_after_they_exist():
    sink, _session, sender = make_sink()
    pipeline, _muxer, _sink = make_pipeline(sink=sink)
    pipeline.open_preview()

    pipeline.on_chunk(make_chunk(0))
    # nothing is decidable yet: the video decoder needs its 2-latent lookahead
    # and the audio decoder its right margin
    assert [b["event"] for b in sender.bodies] == ["open"]

    pipeline.on_chunk(make_chunk(1))
    # ... and the moment they are, they are on the wire -- not at finish
    events = [b["event"] for b in sender.bodies]
    assert events[1] == "init"
    assert events.count("segment") == 16
    assert pipeline.report().chunk_emissions[1].fragments == 16


def test_the_emission_log_is_per_chunk_and_ends_small():
    pipeline, _muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    emissions = report.chunk_emissions
    assert [e.chunk for e in emissions] == [0, 1, 2]
    # chunk 0 decodes nothing (both decoders are still filling their context)
    assert (emissions[0].frames, emissions[0].fragments) == (0, 0)
    # chunk 1 decodes 17 frames and sends what the audio clock covers
    assert emissions[1].frames == 17 and emissions[1].fragments == 16
    assert emissions[1].held_frames == 1
    assert emissions[1].samples == 28 * 800
    # every chunk's decode is copied into the outputs and pushed in the same
    # callback: nothing accumulates for the end
    assert sum(e.frames for e in emissions) == 34  # the tail is finish's
    assert sum(e.fragments for e in emissions) == report.fragments_before_finish
    assert report.fragments_before_finish == streamable_frames(FRAMES)

    text = report.describe_emissions()
    assert "chunk   1" in text and "before finish()" in text


def test_the_emission_log_is_kept_even_with_no_preview():
    pipeline, _muxer, _sink = make_pipeline(preview=False)
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    # the collectors still ran, so the record still says what was produced --
    # it just has nothing to report as sent
    assert [e.frames for e in report.chunk_emissions] == [0, 17, 17]
    assert all(e.fragments == 0 for e in report.chunk_emissions)
    assert report.fragments_before_finish == 0


def test_a_collector_copy_never_delays_the_send():
    """The collector's copy happens beside the send, not in front of it.

    Ordering, per chunk: decode -> copy into the outputs -> mux -> push. The
    copy is a memcpy into a buffer that already exists, so "the collector
    blocked the stream" would show up as fragments arriving a chunk late. This
    pins the ordering directly instead.
    """
    order = []

    class OrderedMuxer(FakeMuxer):
        def write_video_frame(self, image, force_keyframe=None):
            order.append("mux")
            return super().write_video_frame(image, force_keyframe)

    pipeline, muxer, _sink = make_pipeline(muxer=OrderedMuxer())
    original_collect = pipeline._collect

    def watched_collect(batches):
        written = original_collect(batches)
        if written:
            order.append("collect")
        return written

    pipeline._collect = watched_collect
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))

    assert order[0] == "collect"
    assert "mux" in order[:3]  # muxing starts in the same callback


def test_report_records_the_evidence_a_run_should_be_judged_on():
    pipeline, _muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()
    payload = report.to_dict()
    assert payload["chunks"] == 3
    assert payload["frames_decoded"] == FRAMES
    assert payload["samples_decoded"] == 65 * 800
    assert payload["fragments"] == FRAMES
    assert payload["fragment_bytes"] == 96 * FRAMES
    assert payload["largest_fragment"] == 96
    assert payload["first_fragment_chunk"] == 1
    assert payload["oversize_fragments"] == 0
    assert payload["errors"] == 0
    assert "fragment" in report.describe()


def test_an_oversized_fragment_is_dropped_not_split_and_not_fatal():
    big = MAX_RAW_PAYLOAD_BYTES + 1
    sink, _session, sender = make_sink()
    pipeline, muxer, _sink = make_pipeline(muxer=FakeMuxer(fragment_bytes=big), sink=sink)
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    # sampling saw nothing: the pipeline ran to completion
    assert report.frames_muxed == FRAMES
    assert report.oversize_fragments == FRAMES
    assert report.send_failures == FRAMES
    # protocol v1 cannot split a payload, so nothing went out as a segment
    assert "segment" not in sender.events()
    assert sink.errors == FRAMES
    # ... and the preview lane was never marked broken; only that message was
    assert report.preview_disabled is False


def test_a_dead_socket_never_reaches_the_sampler():
    class ExplodingSender:
        def __init__(self):
            self.calls = 0

        def __call__(self, message_type, body, client_id):
            self.calls += 1
            raise RuntimeError("socket is gone")

    sender = ExplodingSender()
    sink, session, _sender = make_sink(sender=sender)
    pipeline, _muxer, _sink = make_pipeline(sink=sink)
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    assert report.frames_muxed == FRAMES  # muxing carried on regardless
    assert session.send_failures == sender.calls > 0
    assert report.preview_disabled is False


def test_a_mux_failure_disables_the_preview_and_nothing_else():
    sink, _session, sender = make_sink()
    pipeline, muxer, _sink = make_pipeline(muxer=FakeMuxer(fail_on_frame=3), sink=sink)
    pipeline.open_preview()

    for index in range(3):
        assert pipeline.on_chunk(make_chunk(index)) is None  # never raises
    report = pipeline.finish()

    assert report.preview_disabled is True
    assert "encoder exploded" in report.disabled_reason
    assert report.errors >= 1
    assert muxer.closes == 1  # the muxer was closed on the way out
    # the user was told, on the same session, and the stream was not ended
    statuses = [b for b in sender.bodies if b["event"] == "status"]
    assert statuses and statuses[-1]["phase"] == "sampling"
    assert "preview stopped" in statuses[-1]["message"]
    # later chunks are accepted and ignored
    assert pipeline.chunks == 3


def test_an_audio_collector_failure_reaches_the_sampler():
    """The audio lane is an output now: its failures are the run's failures."""

    class ExplodingAudioDecoder:
        def push(self, z):
            raise RuntimeError("audio vae OOM")

        def finish(self):
            return []

    pipeline, _muxer, _sink = make_pipeline(audio_decoder=ExplodingAudioDecoder())
    pipeline.open_preview()
    with pytest.raises(RuntimeError, match="audio vae OOM"):
        pipeline.on_chunk(make_chunk(0))
    assert pipeline.preview_disabled is False  # the preview did nothing wrong


def test_an_audio_collector_flush_failure_reaches_the_caller():
    class ExplodingAudioTail:
        def push(self, z):
            return []

        def finish(self):
            raise RuntimeError("audio tail decode failed")

    pipeline, _muxer, _sink = make_pipeline(audio_decoder=ExplodingAudioTail())
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    # the video flush runs first and succeeds; the audio flush is what fails,
    # and it is not swallowed either
    with pytest.raises(RuntimeError, match="audio tail decode failed"):
        pipeline.finish()
    assert pipeline.collected_frames == FRAMES


def test_a_short_waveform_is_refused_rather_than_returned():
    class ShortAudioDecoder:
        """Drops its last block, the way a silently wrong flush would."""

        def __init__(self, config):
            self._inner = make_audio_decoder(config)

        def push(self, z):
            return self._inner.push(z)

        def finish(self):
            return self._inner.finish()[:-1]

    config = make_config()
    pipeline, _muxer, _sink = make_pipeline(
        config=config, audio_decoder=ShortAudioDecoder(config)
    )
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    with pytest.raises(sp.PipelineError, match=r"audio sample\(s\) for a clip of"):
        pipeline.finish()


def test_on_chunk_takes_a_chunkoutput_and_nothing_else():
    pipeline, _muxer, _sink = make_pipeline()
    for bad in (None, {"index": 0}, (0, 1), object()):
        with pytest.raises(TypeError, match="ChunkOutput"):
            pipeline.on_chunk(bad)


# --------------------------------------------------------------------------
# lifetime
# --------------------------------------------------------------------------


def test_finish_is_idempotent_and_releases_only_the_preview():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))

    first = pipeline.finish()
    second = pipeline.finish()
    assert muxer.closes == 1
    assert second.to_dict() == first.to_dict()
    # the preview lane is gone ...
    assert pipeline._muxer is None
    assert pipeline.pending_frames == 0
    # ... but the audio collector is not: it is an output, not a preview part
    assert pipeline._audio is not None
    # ... and the collected frames are not
    assert pipeline.finalize_image() is pipeline.finalize_image()
    assert pipeline.finalize_image().shape == (FRAMES, HEIGHT, WIDTH, 3)

    # a late chunk after finish is accepted and dropped, not an error
    assert pipeline.on_chunk(make_chunk(0)) is None
    assert pipeline.report().collected_frames == FRAMES


def test_cancel_closes_without_flushing_and_is_idempotent():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))
    muxed_before = muxer.frame_count

    report = pipeline.cancel("interrupted")
    pipeline.cancel("interrupted")
    pipeline.finish()
    assert muxer.closes == 1
    assert muxer.frame_count == muxed_before  # the tail was never flushed
    assert report.preview_disabled is True
    assert report.disabled_reason == "interrupted"
    assert pipeline._muxer is None


def test_cancel_leaves_no_pcm_the_pipeline_is_not_holding():
    """``samples_available`` is a claim about held data, so a cancel must clear it.

    Two chunks in, the preview lane is genuinely mid-flight: 17 frames decoded,
    22400 samples of PCM queued, 16 frames muxed against 21333 of those samples
    -- so a partly consumed block is still in ``_pcm`` and one frame is still
    held for audio that does not exist yet. That is the state a cancel has to
    release, and the only way to see that it did is the counter: the deque and
    the running total are one buffer described twice, and a total left standing
    after the deque is cleared is the pipeline reporting samples it no longer
    has.
    """
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))

    # there is something to release: decoded, partly muxed, partly held
    assert pipeline.frames_decoded == 17 and pipeline.samples_decoded == 22400
    assert muxer.frame_count == 16
    assert muxer.samples_written == RAVEN_CLOCK.samples_for_frames(16) == 21333
    assert pipeline.samples_available == 22400
    assert pipeline.pending_frames == 1
    assert pipeline._pcm and sum(int(b.shape[-1]) for b in pipeline._pcm) > 0

    report = pipeline.cancel("interrupted")

    assert pipeline.samples_available == 0
    assert pipeline.pending_frames == 0
    assert not pipeline._pcm and not pipeline._frames
    assert pipeline._muxer is None
    # what was *sent* is still on the record: those are not buffers
    assert report.frames_muxed == 16 and report.samples_muxed == 21333

    # a second cancel neither resurrects nor disturbs any of it
    again = pipeline.cancel("interrupted")
    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0
    assert not pipeline._pcm and not pipeline._frames
    assert again.to_dict() == report.to_dict()


def test_cancel_with_no_preview_holds_nothing_either():
    """No preview means nothing was ever queued; cancel must still report zero."""
    pipeline, _muxer, _sink = make_pipeline(preview=False)
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))
    assert pipeline.samples_decoded == 22400  # the collector ran regardless
    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0

    pipeline.cancel("interrupted")

    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0
    assert not pipeline._pcm and not pipeline._frames


def test_an_abort_that_raises_still_empties_the_preview_queues():
    """The cleanup that follows a failed abort is the one that must not be skipped."""
    config = make_config()
    video_decoder = AbortRaisesDecoder(make_video_decoder())
    pipeline, _muxer, _sink = make_pipeline(
        config=config, video_decoder=video_decoder
    )
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))
    assert pipeline.samples_available == 22400 and pipeline.pending_frames == 1

    report = pipeline.cancel("interrupted")

    assert video_decoder.abort_calls == 1 and report.errors == 1
    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0
    assert not pipeline._pcm and not pipeline._frames


def test_a_preview_failure_mid_run_drops_the_queues_it_was_holding():
    """``_disable_preview`` releases the same lane, and the counter goes with it."""
    pipeline, _muxer, _sink = make_pipeline(muxer=FakeMuxer(fail_on_frame=8))
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))

    assert pipeline.preview_disabled is True
    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0
    assert not pipeline._pcm and not pipeline._frames
    # and the collectors carried on: this is a preview failure, not a run failure
    assert pipeline.frames_decoded == 17 and pipeline.samples_decoded == 22400
    pipeline.on_chunk(make_chunk(2))
    pipeline.finish()
    assert pipeline.finalize_image().shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert pipeline.samples_available == 0 and pipeline.pending_frames == 0


def test_cancel_aborts_both_decoders_before_it_lets_go_of_them():
    """The tensors go at a point this code controls, not when the GC gets there.

    Both decoders are held here for the length of the test, exactly as a
    traceback or a node holds them during a real cancel: if ``cancel`` only
    dropped the pipeline's references, the video coordinator's 5-frame
    ``dec_overlap`` (on the decode device on real hardware) and the audio
    decoder's overlap-save history would still be alive at this point.
    """
    video_decoder = make_video_decoder()
    audio_decoder = make_audio_decoder(make_config())
    pipeline, _muxer, _sink = make_pipeline(
        video_decoder=video_decoder, audio_decoder=audio_decoder
    )
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))

    # both lanes are mid-stream: the video coordinator is holding a decoded
    # overlap *and* pending latents, the audio decoder its context window
    assert video_decoder._dec_overlap is not None and video_decoder._pending is not None
    assert audio_decoder._buffer is not None
    video_decodes = video_decoder.decoder.decode_calls
    audio_decodes = audio_decoder.decode_fn.calls

    report = pipeline.cancel("interrupted")

    # every tensor/array the two decoders were holding is gone ...
    assert video_decoder._pending is None
    assert video_decoder._dec_overlap is None
    assert video_decoder._frame_limit is None
    assert audio_decoder._buffer is None
    assert audio_decoder.finished
    # ... and not one of them was decoded, padded or flushed on the way out
    assert video_decoder.decoder.decode_calls == video_decodes
    assert audio_decoder.decode_fn.calls == audio_decodes
    assert video_decoder.finish() == [] and audio_decoder.finish() == []
    assert report.errors == 0

    # the pipeline let go afterwards, and returns no partial output
    assert pipeline._video is None and pipeline._audio is None
    with pytest.raises(sp.PipelineError, match="no partial IMAGE"):
        pipeline.finalize_image()
    with pytest.raises(sp.PipelineError, match="no partial AUDIO"):
        pipeline.finalize_audio(FakeAudioVAE())

    pipeline.cancel("again")  # idempotent, and still no decoder work
    assert video_decoder.decoder.decode_calls == video_decodes
    assert audio_decoder.decode_fn.calls == audio_decodes


def test_cancel_after_finish_aborts_the_decoders_too():
    """Finishing already emptied them; aborting again must be a no-op, not a redo."""
    video_decoder = make_video_decoder()
    audio_decoder = make_audio_decoder(make_config())
    pipeline, _muxer, _sink = make_pipeline(
        video_decoder=video_decoder, audio_decoder=audio_decoder
    )
    run_pipeline(pipeline)
    video_decodes = video_decoder.decoder.decode_calls
    audio_decodes = audio_decoder.decode_fn.calls

    report = pipeline.cancel("downstream failed")

    assert video_decoder._pending is None and video_decoder._dec_overlap is None
    assert audio_decoder._buffer is None and audio_decoder.finished
    assert video_decoder.decoder.decode_calls == video_decodes
    assert audio_decoder.decode_fn.calls == audio_decodes
    assert report.errors == 0


def test_cancel_works_with_a_decoder_that_has_no_abort():
    """``abort`` is feature-probed: an older decoder just gets dereferenced."""
    config = make_config()
    video_decoder = NoAbortDecoder(make_video_decoder())
    audio_decoder = NoAbortDecoder(make_audio_decoder(config))
    pipeline, _muxer, _sink = make_pipeline(
        config=config, video_decoder=video_decoder, audio_decoder=audio_decoder
    )
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))

    report = pipeline.cancel("interrupted")

    assert report.errors == 0  # a missing abort is not a failure
    assert report.disabled_reason == "interrupted"
    assert pipeline._video is None and pipeline._audio is None
    with pytest.raises(sp.PipelineError, match="no partial IMAGE"):
        pipeline.finalize_image()


def test_an_abort_that_raises_does_not_stop_the_other_lane_or_the_cleanup():
    config = make_config()
    video_decoder = AbortRaisesDecoder(make_video_decoder())
    audio_decoder = make_audio_decoder(config)
    pipeline, muxer, _sink = make_pipeline(
        config=config, video_decoder=video_decoder, audio_decoder=audio_decoder
    )
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    pipeline.on_chunk(make_chunk(1))
    audio_decodes = audio_decoder.decode_fn.calls

    report = pipeline.cancel("interrupted")

    assert video_decoder.abort_calls == 1
    # the second lane was still aborted, without decoding anything
    assert audio_decoder._buffer is None and audio_decoder.finished
    assert audio_decoder.decode_fn.calls == audio_decodes
    # the preview was still closed and the collectors were still released
    assert muxer.closes == 1
    assert pipeline._muxer is None
    assert pipeline._video is None and pipeline._audio is None
    with pytest.raises(sp.PipelineError, match="no partial IMAGE"):
        pipeline.finalize_image()
    # the failure is recorded rather than raised: it must not replace the
    # reason the run was cancelled in the first place
    assert report.errors == 1
    assert report.disabled_reason == "interrupted"


def test_a_pipeline_built_after_a_cancel_starts_from_zero():
    """A cancelled run leaves nothing behind for the next one to inherit."""
    cancelled, _muxer, _sink = make_pipeline()
    cancelled.open_preview()
    cancelled.on_chunk(make_chunk(0))
    cancelled.on_chunk(make_chunk(1))
    cancelled.cancel("interrupted")

    pipeline, _muxer2, _sink2 = make_pipeline()
    fresh = pipeline.report()
    assert (fresh.chunks, fresh.collected_frames, fresh.collected_samples) == (0, 0, 0)
    assert (fresh.frames_muxed, fresh.samples_muxed, fresh.fragments) == (0, 0, 0)
    assert (fresh.errors, fresh.image_bytes, fresh.audio_bytes) == (0, 0, 0)
    assert pipeline._image is None and pipeline._waveform is None

    image = run_pipeline(pipeline)
    assert image.shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert pipeline.report().collected_frames == FRAMES


def test_a_pipeline_with_no_preview_still_collects_the_image():
    pipeline, _muxer, _sink = make_pipeline(preview=False)
    assert pipeline.preview_disabled and pipeline.preview_disabled_reason == "no preview sink"
    assert pipeline.open_preview() is False
    assert pipeline.status("sampling") is False

    image = run_pipeline(pipeline)

    report = pipeline.report()
    assert report.chunks == 3
    assert report.collected_frames == FRAMES
    assert report.fragments == 0 and report.frames_muxed == 0
    assert image.shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert "not streamed" in report.describe()
    assert "collector" in report.describe()


def test_a_pipeline_whose_preview_lane_is_missing_still_reports_status():
    sink, _session, sender = make_sink()
    config = make_config()
    # what build_media_pipeline produces when PyAV is unavailable: a sink and a
    # collector, but no audio decoder and no muxer
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=make_video_decoder(),
        audio_decoder=make_audio_decoder(config),
        muxer=None,
        sink=sink,
        preview_disabled_reason="preview unavailable: EncoderUnavailable: no libx264",
    )
    assert pipeline.open_preview() is True
    assert pipeline.status("model_loading", message=pipeline.preview_disabled_reason) is True

    image = run_pipeline(pipeline)

    assert sender.events() == ["open", "status"]
    assert image.shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert pipeline.report().collected_frames == FRAMES


# --------------------------------------------------------------------------
# no GPU references
# --------------------------------------------------------------------------


def test_the_pipeline_keeps_no_reference_to_the_sampler_tensors():
    pipeline, muxer, _sink = make_pipeline()
    pipeline.open_preview()

    tracked = []

    def wrap(tensor):
        holder = TrackedTensor(tensor)
        tracked.append(holder)
        return holder

    for index in range(2):
        pipeline.on_chunk(make_chunk(index, wrap=wrap))

    assert all(holder.detach_calls == 1 and holder.cpu_calls == 1 for holder in tracked)

    references = [weakref.ref(holder) for holder in tracked]
    chunk_tensors = [holder._tensor for holder in tracked]
    tracked.clear()
    gc.collect()
    assert all(reference() is None for reference in references), (
        "the pipeline is still holding a device tensor from a finished chunk"
    )

    # nothing buffered for the muxer is a torch tensor either
    for _index, frame in pipeline._frames:
        assert isinstance(frame, np.ndarray)
    for block in pipeline._pcm:
        assert isinstance(block, np.ndarray)
    for frame, _key in muxer.frames:
        assert isinstance(frame, np.ndarray)
    del chunk_tensors


def test_detach_to_cpu_leaves_non_tensors_alone():
    array = np.zeros((2, 3))
    assert sp.detach_to_cpu(array) is array
    tensor = torch.zeros(2, 3, dtype=torch.float64)
    out = sp.detach_to_cpu(tensor)
    assert out.dtype == torch.float32 and out.device.type == "cpu"


def test_frame_and_pcm_conversion_reject_the_wrong_shape():
    with pytest.raises(sp.PipelineError, match=r"\[B, C, T, H, W\]"):
        sp.frames_to_arrays(np.zeros((3, 4, 5)))
    with pytest.raises(sp.PipelineError, match="channel"):
        sp.frames_to_arrays(np.zeros((1, 2, 3, 4, 4)))
    with pytest.raises(sp.PipelineError, match="channel"):
        sp.pcm_to_array(np.zeros((1, 5, 100)))


def test_pipeline_config_rejects_an_unencodable_canvas():
    with pytest.raises(sp.PipelineError, match="even"):
        sp.PipelineConfig(frames=39, width=31, height=16)
    with pytest.raises(sp.PipelineError, match="positive"):
        sp.PipelineConfig(frames=0, width=16, height=16)


# --------------------------------------------------------------------------
# the video collector: always on, and it is the IMAGE output
# --------------------------------------------------------------------------


def same_image(first, second) -> bool:
    left, right = _to_numpy(first), _to_numpy(second)
    return left.shape == right.shape and np.array_equal(left, right)


def reference_image():
    """The IMAGE a run with a perfectly healthy preview produces."""
    pipeline, _muxer, _sink = make_pipeline()
    return run_pipeline(pipeline)


def test_the_collector_produces_a_standard_comfy_image():
    pipeline, _muxer, _sink = make_pipeline()
    image = run_pipeline(pipeline)

    assert image.shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert str(image.dtype).endswith("float32")
    report = pipeline.report()
    assert report.collected_frames == report.expected_frames == FRAMES
    assert report.image_complete is True
    assert report.image_shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert report.image_bytes == FRAMES * HEIGHT * WIDTH * 3 * 4
    assert report.image_device and report.image_dtype
    assert "collector" in report.describe()


def reference_outputs():
    """The IMAGE and AUDIO a run with a perfectly healthy preview produces."""
    pipeline, _muxer, _sink = make_pipeline()
    return run_pipeline_av(pipeline)


def preview_failure_pipelines():
    """Every way the preview can be absent or die, as ``(name, pipeline)``."""
    # 1. no sink at all
    yield "no sink", make_pipeline(preview=False)[0]

    # 2. no PyAV: a sink and both collectors, but no muxer
    yield "no PyAV", sp.StreamingPipeline(
        config=make_config(),
        video_decoder=make_video_decoder(),
        audio_decoder=make_audio_decoder(make_config()),
        muxer=None,
        sink=make_sink()[0],
        preview_disabled_reason="preview unavailable: no PyAV",
    )

    # 3. every fragment too big for protocol v1
    yield "oversize fragments", make_pipeline(
        muxer=FakeMuxer(fragment_bytes=MAX_RAW_PAYLOAD_BYTES + 1)
    )[0]

    # 4. the websocket is gone
    class ExplodingSender:
        def __call__(self, message_type, body, client_id):
            raise RuntimeError("socket is gone")

    yield "dead socket", make_pipeline(sink=make_sink(sender=ExplodingSender())[0])[0]

    # 5. the muxer throws mid-clip
    yield "broken muxer", make_pipeline(muxer=FakeMuxer(fail_on_frame=3))[0]


def test_both_outputs_are_the_same_whatever_the_preview_does():
    """Five ways for the preview to fail; one IMAGE and one AUDIO, bit for bit."""
    expected_image, expected_audio = reference_outputs()

    for name, pipeline in preview_failure_pipelines():
        image, audio = run_pipeline_av(pipeline)
        assert same_image(image, expected_image), name
        assert torch.equal(audio["waveform"], expected_audio["waveform"]), name
        assert audio["sample_rate"] == expected_audio["sample_rate"], name
        assert pipeline.report().audio_complete is True, name


def test_a_collector_decode_failure_reaches_the_sampler():
    class ExplodingVideoDecoder:
        def __init__(self):
            self.finished = 0

        def push(self, z):
            raise RuntimeError("video vae OOM")

        def finish(self):
            self.finished += 1
            return []

    pipeline, _muxer, _sink = make_pipeline(video_decoder=ExplodingVideoDecoder())
    pipeline.open_preview()
    # not swallowed: without these frames there is no IMAGE to return
    with pytest.raises(RuntimeError, match="video vae OOM"):
        pipeline.on_chunk(make_chunk(0))
    assert pipeline.preview_disabled is False  # the preview did nothing wrong


def test_a_collector_flush_failure_reaches_the_caller():
    class ExplodingTail:
        def push(self, z):
            return []

        def finish(self):
            raise RuntimeError("tail decode failed")

    pipeline, _muxer, _sink = make_pipeline(video_decoder=ExplodingTail())
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    with pytest.raises(RuntimeError, match="tail decode failed"):
        pipeline.finish()


def test_a_short_clip_is_refused_rather_than_returned():
    class ShortDecoder:
        """Drops the 5-frame tail, the way a silently wrong flush would."""

        def __init__(self):
            self._inner = make_video_decoder()

        def push(self, z):
            return self._inner.push(z)

        def finish(self):
            batches = self._inner.finish()
            return [batch for batch in batches if not batch.is_tail]

    pipeline, _muxer, _sink = make_pipeline(video_decoder=ShortDecoder())
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    with pytest.raises(sp.PipelineError, match=r"34 frame\(s\) for a 39-frame"):
        pipeline.finish()


# --------------------------------------------------------------------------
# the audio collector and the official normalisation
# --------------------------------------------------------------------------


@pytest.fixture
def m0_tolerance() -> float:
    """The M0 overlap-save tolerance: streamed vs full-sequence decode.

    ``tools/probe_audio_overlap.py`` measured ``max|diff| < 2.5e-6`` against the
    real BigVGAN decoder at margin 17 / block 28. The fake used here has an
    exactly known finite receptive field, so with margin >= radius the same
    comparison is *exact* -- this fixture is what keeps the assertion phrased
    as "within the published tolerance" rather than as a number that happens to
    be zero for a fake.
    """
    return 2.5e-6


def full_sequence_waveform(latents):
    """What a whole-clip ``decode`` of the same latents produces."""
    decoder = FiniteRFAudioDecoder(radius=2, samples_per_latent=800, latent_channels=32)
    return torch.as_tensor(decoder(latents), dtype=torch.float32)


def test_the_collected_waveform_matches_a_full_decode(m0_tolerance):
    pipeline, _muxer, _sink = make_pipeline()
    run_pipeline(pipeline)

    latents = torch.cat([chunk_latents(i)[1] for i in range(3)], dim=-1)
    reference = full_sequence_waveform(latents)
    collected = pipeline._waveform

    assert collected.shape == reference.shape == (1, 2, 65 * 800)
    assert float(torch.max(torch.abs(collected - reference))) <= m0_tolerance


def test_finalize_audio_is_the_official_expression(m0_tolerance):
    pipeline, _muxer, _sink = make_pipeline()
    run_pipeline(pipeline)
    raw = pipeline._waveform.clone()

    payload = pipeline.finalize_audio(FakeAudioVAE(sample_rate=32000))

    # comfy_extras/nodes_audio.py::vae_decode_audio, expression for expression
    expected = raw.clone()
    std = torch.std(expected, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    expected /= std

    assert set(payload) == {"waveform", "sample_rate"}
    assert torch.equal(payload["waveform"], expected)
    assert payload["sample_rate"] == 32000
    assert payload["waveform"].dtype == torch.float32
    assert payload["waveform"].shape == (1, 2, 65 * 800)


def test_a_quiet_clip_is_not_amplified():
    """The floor at 1.0 is what stops a near-silent clip being blown up."""

    class QuietDecoder:
        def __init__(self, config):
            self._inner = make_audio_decoder(config)

        def push(self, z):
            return [block * 1e-6 for block in self._inner.push(z)]

        def finish(self):
            return [block * 1e-6 for block in self._inner.finish()]

    config = make_config()
    pipeline, _muxer, _sink = make_pipeline(config=config, audio_decoder=QuietDecoder(config))
    run_pipeline(pipeline)
    raw = pipeline._waveform.clone()

    payload = pipeline.finalize_audio(FakeAudioVAE())
    assert torch.equal(payload["waveform"], raw)  # divided by exactly 1.0


def test_finalize_audio_normalises_once_however_often_it_is_called():
    pipeline, _muxer, _sink = make_pipeline()
    run_pipeline(pipeline)

    first = pipeline.finalize_audio(FakeAudioVAE())
    second = pipeline.finalize_audio(FakeAudioVAE())
    assert second is first  # the divide is in place; twice would be wrong


@pytest.mark.parametrize(
    "vae, override, expected",
    [
        (FakeAudioVAE(sample_rate=32000), None, 32000),
        (FakeAudioVAE(sample_rate=32000, output_rate=48000), None, 48000),
        (object(), None, 44100),
        (FakeAudioVAE(sample_rate=32000), 16000, 16000),
    ],
)
def test_the_sample_rate_follows_upstreams_resolution_order(vae, override, expected):
    pipeline, _muxer, _sink = make_pipeline()
    run_pipeline(pipeline)
    payload = pipeline.finalize_audio(vae, sample_rate=override)
    assert payload["sample_rate"] == expected


def test_finalize_audio_is_refused_before_finish_and_after_cancel():
    pipeline, _muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))

    with pytest.raises(sp.PipelineError, match="before finish"):
        pipeline.finalize_audio(FakeAudioVAE())

    pipeline.cancel("interrupted")
    with pytest.raises(sp.PipelineError, match="no partial AUDIO"):
        pipeline.finalize_audio(FakeAudioVAE())
    assert pipeline._waveform is None
    assert pipeline.report().audio_bytes == 0


def test_the_preview_pcm_is_read_back_out_of_the_collector():
    pipeline, muxer, _sink = make_pipeline()
    run_pipeline(pipeline)
    raw = pipeline._waveform

    written = muxer.concatenated_audio()
    assert written.shape[-1] == raw.shape[-1]
    # what was muxed is what was collected -- before the whole-clip
    # normalisation, which a stream cannot know in advance
    assert np.allclose(written, _to_numpy(raw)[0], atol=0)


def test_finalize_image_is_refused_before_finish_and_after_cancel():
    pipeline, _muxer, _sink = make_pipeline()
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))

    with pytest.raises(sp.PipelineError, match="before finish"):
        pipeline.finalize_image()

    pipeline.cancel("interrupted")
    with pytest.raises(sp.PipelineError, match="no partial IMAGE"):
        pipeline.finalize_image()
    # the buffer is gone, not merely hidden
    assert pipeline._image is None
    assert pipeline.report().collected_frames == 34
    assert pipeline.report().image_bytes == 0


def test_cancel_after_finish_still_releases_the_collected_frames():
    pipeline, _muxer, _sink = make_pipeline()
    run_pipeline(pipeline)
    assert pipeline.finalize_image() is not None

    pipeline.cancel("downstream failed")
    with pytest.raises(sp.PipelineError, match="no partial IMAGE"):
        pipeline.finalize_image()
    pipeline.cancel("again")  # idempotent


def test_the_preview_reads_back_exactly_what_the_collector_wrote():
    pipeline, muxer, _sink = make_pipeline()
    image = run_pipeline(pipeline)

    assert muxer.frame_count == FRAMES
    for index, (frame, _key) in enumerate(muxer.frames):
        assert np.array_equal(frame, _to_numpy(image[index]))


def test_the_collector_never_decodes_the_whole_clip_at_once():
    decoder = make_video_decoder()
    pipeline, _muxer, _sink = make_pipeline(video_decoder=decoder)
    pipeline.open_preview()
    pipeline.on_chunk(make_chunk(0))
    # chunk 0 finalizes no frames (the decoder wants its 2-latent lookahead),
    # so the buffer is allocated on the first batch, at chunk 1
    assert pipeline._image is None
    pipeline.on_chunk(make_chunk(1))
    first_buffer = pipeline._image
    assert first_buffer is not None
    pipeline.on_chunk(make_chunk(2))
    pipeline.finish()

    fake = decoder.decoder
    # every call was one 7-latent chunk: 5 to finalize plus the 2-latent
    # lookahead, which is the unit upstream's memory estimate prices
    assert fake.decode_calls > 0
    assert fake.decoded_latents == 7 * fake.decode_calls
    # one buffer, allocated once, handed over without a copy
    assert pipeline._image is first_buffer
    assert pipeline.finalize_image() is first_buffer


@pytest.mark.parametrize("frames", [22, 39, 192, 362])
def test_the_collector_covers_the_whole_supported_range(frames):
    config = sp.PipelineConfig(frames=frames, width=2, height=2)
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=make_video_decoder(spatial=1),
        audio_decoder=make_audio_decoder(config),
        sink=None,
    )
    for chunk in clip_chunks(frames):
        pipeline.on_chunk(chunk)
    pipeline.finish()
    image = pipeline.finalize_image()
    audio = pipeline.finalize_audio(FakeAudioVAE())

    assert image.shape == (frames, 2, 2, 3)
    report = pipeline.report()
    assert report.collected_frames == frames
    assert report.image_bytes == frames * 2 * 2 * 3 * 4
    # the host cost the node reports is the output and nothing else
    assert report.image_bytes == int(np.prod(image.shape)) * 4

    # the audio lane covers the same clip: round(frames / 24 * 40) latents,
    # 800 samples each
    expected_samples = round(frames / 24 * 40) * 800
    assert audio["waveform"].shape == (1, 2, expected_samples)
    assert report.collected_samples == expected_samples
    assert report.audio_complete is True
    assert report.audio_bytes == 2 * expected_samples * 4


def test_the_published_canvas_costs_what_the_report_says():
    """The host cost the module documents, at the two clip lengths that matter."""
    # not allocated here -- just the arithmetic PipelineReport.image_bytes does
    for frames, gigabytes in ((192, 2.43), (362, 4.59)):
        image_bytes = frames * 768 * 1376 * 3 * 4
        assert round(image_bytes / 1e9, 2) == gigabytes
    assert "2.43 GB of IMAGE at 192 frames" in sp.__doc__
    assert "4.59 GB at 362" in sp.__doc__


# --------------------------------------------------------------------------
# denormalisation order: cast first, exactly like the official decode
# --------------------------------------------------------------------------


class TorchVideoVAEInner:
    """``MiniMaxH3VideoVAE``'s decode surface, in miniature but dtype-faithful.

    Only the parts the streaming coordinator touches, and every one of them
    keeps the dtype it was given -- which is the whole point here: the test is
    about *when* the latents are rounded to the VAE's dtype, so a fake that
    silently computed in float32 would prove nothing.
    """

    def __init__(self, channels: int = 24) -> None:
        # deliberately non-integer, non-uniform: a mean/std that happened to be
        # exactly representable would make both orders agree by luck
        index = torch.arange(channels, dtype=torch.float32)
        self.latents_mean = 0.1373 + 0.0217 * index
        self.latents_std = 1.0731 + 0.0371 * index
        self.clip_length = 17
        self.vae_ratio_t = 4
        self.token_drop = 3
        self.decode_dtypes = []

    def _adaptive_decode(self, z):
        self.decode_dtypes.append((str(z.device), str(z.dtype)))
        x = z[:, :3].repeat_interleave(self.vae_ratio_t, dim=2)
        weights = torch.linspace(
            0.5, 1.5, x.shape[2], dtype=x.dtype, device=x.device
        ).view(1, 1, -1, 1, 1)
        return x * weights

    def blend(self, a, b, blend_extent, dim):
        """Port of ``MiniMaxH3VideoVAE.blend``."""
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = (1 - positions / blend_extent).reshape(
            [blend_extent if i == dim % a.ndim else 1 for i in range(a.ndim)]
        )
        weight_b = (positions / blend_extent).reshape(weight_a.shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(a.shape[dim] - blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent < b.shape[dim]:
            rest = [slice(None)] * b.ndim
            rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(rest)]], dim=dim)
        return blended

    def _finalize_pixels(self, part):
        # the real hook multiplies by fp32 buffers, so the result is fp32
        return part.float() * 0.5 + 0.25


class TorchVideoVAE:
    """The ``comfy.sd.VAE`` attributes the device-bound decoder reads."""

    def __init__(self, dtype=torch.float16) -> None:
        self.first_stage_model = TorchVideoVAEInner()
        self.device = torch.device("cpu")
        self.vae_dtype = dtype
        self.output_device = torch.device("cpu")


class HostDenormDecoder:
    """The order this lane used to have: denormalise on the host, cast after."""

    def __init__(self, vae) -> None:
        self._adapter = minimax_decoder_adapter(vae)
        self._vae = vae
        self.clip_length = self._adapter.clip_length
        self.vae_ratio_t = self._adapter.vae_ratio_t
        self.token_drop = self._adapter.token_drop

    def _adaptive_decode(self, z):
        return self._adapter._adaptive_decode(
            z.to(device=self._vae.device, dtype=self._vae.vae_dtype)
        )

    def blend(self, a, b, blend_extent, dim):
        return self._adapter.blend(a, b, blend_extent, dim)

    def _finalize_pixels(self, part):
        return self._adapter._finalize_pixels(part)

    def denormalize(self, z):
        return self._adapter.denormalize(z)


def video_latents(latent_t: int = 12, channels: int = 24, size: int = 2):
    generator = torch.Generator().manual_seed(4242)
    return torch.randn(
        (1, channels, latent_t, size, size), generator=generator, dtype=torch.float32
    )


def official_full_decode(vae, latents):
    """``VAE.decode`` -> ``MiniMaxH3VideoVAE.decode``, in that exact order."""
    inner = vae.first_stage_model
    z = latents.to(device=vae.device, dtype=vae.vae_dtype)  # comfy/sd.py
    mean = inner.latents_mean.view(1, -1, 1, 1, 1).to(z)     # comfy/ldm/minimax/vae.py
    std = inner.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * std + mean
    return torch.cat(reference_decode_temporal(inner, z), dim=2)


def run_incremental(decoder, latents, sizes=(5, 5, 2)):
    batches = []
    start = 0
    for size in sizes:
        batches.extend(decoder.push(latents[:, :, start:start + size]))
        start += size
    batches.extend(decoder.finish())
    return torch.cat([batch.frames for batch in batches], dim=2)


def test_the_streaming_decode_matches_the_official_one_bit_for_bit():
    vae = TorchVideoVAE(dtype=torch.float16)
    latents = video_latents()

    streamed = run_incremental(sp.build_video_decoder(vae), latents)
    official = official_full_decode(vae, latents)

    assert streamed.shape == official.shape
    assert torch.equal(streamed, official)


def test_denormalising_on_the_host_would_not_match():
    vae = TorchVideoVAE(dtype=torch.float16)
    latents = video_latents()
    official = official_full_decode(vae, latents)

    legacy_decoder = HostDenormDecoder(vae)
    legacy = IncrementalVideoDecoder(
        legacy_decoder, denormalize=legacy_decoder.denormalize
    )
    host_first = run_incremental(legacy, latents)

    # same shape, same machine, different rounding: the latents were scaled in
    # float32 and only then squeezed into the VAE's dtype
    assert host_first.shape == official.shape
    assert not torch.equal(host_first, official)
    assert torch.max(torch.abs(host_first - official)) > 0


def test_the_streaming_decoder_does_not_denormalise_on_push():
    vae = TorchVideoVAE()
    decoder = sp.build_video_decoder(vae)
    # IncrementalVideoDecoder's own hook is unused: the step belongs on the
    # decode device, after the cast
    assert decoder._denormalize is None
    assert isinstance(decoder.decoder, sp._DeviceBoundVideoDecoder)


def test_the_latents_reach_the_vae_on_its_device_in_its_dtype():
    vae = TorchVideoVAE(dtype=torch.float16)
    decoder = sp.build_video_decoder(vae)
    run_incremental(decoder, video_latents())

    assert vae.first_stage_model.decode_dtypes
    assert set(vae.first_stage_model.decode_dtypes) == {("cpu", "torch.float16")}
    bound = decoder.decoder
    assert bound.decode_calls == len(vae.first_stage_model.decode_dtypes)
    assert bound.last_input_dtype == "torch.float16"

    policy = bound.policy()
    assert policy["decode_dtype"] == "torch.float16"
    assert policy["decode_device"] == "cpu"
    assert "denormalize -> _adaptive_decode" in policy["order"]
    assert "official order" in policy["denormalize"]


def test_frames_leave_the_decode_device_only_after_finalisation():
    vae = TorchVideoVAE(dtype=torch.float16)
    config = make_config(width=2, height=2)
    muxer = FakeMuxer()
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=sp.build_video_decoder(vae),
        audio_decoder=make_audio_decoder(config),
        muxer=muxer,
        sink=make_sink()[0],
    )
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    pipeline.finish()

    # blend and _finalize_pixels ran in the decoder; what the pipeline buffered
    # and handed to the muxer is plain host memory
    assert muxer.frame_count == FRAMES
    for frame, _key in muxer.frames:
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.float32 or frame.dtype == np.float64
    assert pipeline.report().decode_policy["decode_dtype"] == "torch.float16"


def torch_vae_chunks(latents, sizes=VIDEO_CHUNKS, audio_sizes=AUDIO_CHUNKS):
    """Slice one latent tensor into the chunks the sampler would emit."""
    start = 0
    audio_start = 0
    for index, size in enumerate(sizes):
        yield ChunkOutput(
            index=index,
            is_last=index == len(sizes) - 1,
            video_start=start,
            video_stop=start + size,
            audio_start=audio_start,
            audio_stop=audio_start + audio_sizes[index],
            video_x0=latents[:, :, start:start + size],
            audio_x0=torch.zeros(1, 32, 2, audio_sizes[index]),
        )
        start += size
        audio_start += audio_sizes[index]


def test_the_collected_image_equals_the_official_full_decode_bit_for_bit():
    """The claim that lets the node stop calling ``video_vae.decode``."""
    vae = TorchVideoVAE(dtype=torch.float16)
    latents = video_latents(latent_t=12)

    config = sp.PipelineConfig(frames=FRAMES, width=2, height=2)
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=sp.build_video_decoder(vae),
        audio_decoder=make_audio_decoder(config),
        sink=None,
        image_device=vae.output_device,
    )
    for chunk in torch_vae_chunks(latents):
        pipeline.on_chunk(chunk)
    pipeline.finish()
    collected = pipeline.finalize_image()

    official = official_full_decode(TorchVideoVAE(dtype=torch.float16), latents)
    official = official[0].permute(1, 2, 3, 0)  # [1,3,T,H,W] -> [T,H,W,3]

    assert collected.shape == official.shape == (FRAMES, 2, 2, 3)
    assert collected.dtype == torch.float32
    assert torch.equal(collected, official)


def test_the_report_carries_the_budget_it_was_given():
    pipeline, _muxer, _sink = make_pipeline()
    assert pipeline.report().memory_budget == {}
    with_budget = sp.StreamingPipeline(
        config=make_config(),
        video_decoder=make_video_decoder(),
        audio_decoder=make_audio_decoder(make_config()),
        sink=None,
        memory_budget={"total_bytes": 123, "detail": {"kv_peak_rows": 7}},
    )
    payload = with_budget.report().to_dict()
    assert payload["memory_budget"]["total_bytes"] == 123
    assert payload["memory_budget"]["detail"]["kv_peak_rows"] == 7


# --------------------------------------------------------------------------
# the real muxer, when the environment can provide one
# --------------------------------------------------------------------------


def _real_muxer_or_skip(config):
    try:
        return sp.build_muxer(config)
    except Exception as exc:  # noqa: BLE001 - no PyAV, no encoder, no codec
        pytest.skip(f"no usable fMP4 encoder here: {type(exc).__name__}: {exc}")


def test_end_to_end_against_a_real_fragmented_mp4_muxer():
    config = make_config(width=64, height=64)
    muxer = _real_muxer_or_skip(config)
    sink, _session, sender = make_sink()
    pipeline = sp.StreamingPipeline(
        config=config,
        video_decoder=make_video_decoder(spatial=32),
        audio_decoder=make_audio_decoder(config),
        muxer=muxer,
        sink=sink,
    )
    pipeline.open_preview()
    for index in range(3):
        pipeline.on_chunk(make_chunk(index))
    report = pipeline.finish()

    assert report.preview_disabled is False, report.disabled_reason
    assert report.frames_muxed == FRAMES
    assert report.init_bytes > 0
    assert report.fragments > 0

    # the real muxer holds at most a frame or two of its own: whatever it had
    # produced by the end of each callback was pushed inside that callback
    for emission in report.chunk_emissions:
        assert emission.muxed_frames - emission.fragments <= sp.PREVIEW_MAX_MUX_DELAY_FRAMES, (
            emission.to_dict()
        )
    assert report.fragments_before_finish >= streamable_frames(FRAMES) - (
        sp.PREVIEW_MAX_MUX_DELAY_FRAMES
    )
    assert report.first_fragment_chunk is not None
    # the whole point of the 256 KB guidance: one frame per fragment keeps
    # every message small enough for the protocol to carry it
    assert report.largest_fragment <= MAX_RAW_PAYLOAD_BYTES
    events = sender.events()
    assert events[0] == "open" and events[1] == "init"
    assert events.count("segment") == report.fragments
