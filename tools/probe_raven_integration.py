#!/usr/bin/env python3
"""Probe: the whole plugin, end to end, through its own two nodes.

What it answers
---------------
Every other probe in ``tools/`` isolates one lane. This one runs the lane the
user actually gets, with nothing faked below the node surface:

1. :class:`raven_streaming.nodes.RAVENModelLoader` builds the ``MODEL`` from the
   real full BF16 H3 DiT plus the real RAVEN LoRA -- the node's own
   ``load_model``, not the loader function underneath it;
2. upstream's own ``VAELoader`` builds **two real** ``comfy.sd.VAE`` objects
   from the two real H3 VAE files. The node takes a name out of a
   ``folder_paths`` combo, so the probe registers the files' directories with
   ``folder_paths.add_model_folder_path`` and hands over the basename -- the
   loader is driven, never bypassed;
3. :meth:`raven_streaming.nodes.RAVENStreamingSampler.sample` runs the rollout
   with a real ``NestedTensor`` empty AV latent on the ``17k + 5`` grid, and its
   ``LATENT`` / ``IMAGE`` / ``AUDIO`` outputs are checked for shape, dtype,
   finiteness and range. The ``IMAGE`` now comes out of the **streaming
   collector** (``StreamingPipeline.finalize_image``), so the collector half of
   the pipeline report -- ``collected_frames`` / ``expected_frames`` /
   ``image_bytes`` / ``image_shape`` -- is gated rather than noted: it describes
   the output the workflow gets, and a short clip has nowhere else to show up.
   Both outputs now come out of the streaming collectors
   (``finalize_image`` / ``finalize_audio``), so the collector half of the
   pipeline report -- frames *and* samples -- is gated, and the probe asserts
   the node ran **no** whole-clip decode of either stream: neither of the two
   kept diagnostic helpers, and neither ``comfy.sd.VAE.decode`` entry point;
4. the preview lane is captured at the transport seam
   (``PreviewManager.set_sender``, a public injection point) and every
   ``raven.preview`` envelope is checked: one contiguous ``seq``, the
   ``open -> init -> segment* / status* -> end`` order, base64 that decodes, an
   init segment that really is ``ftyp`` + ``moov``, fragments that really are
   ``moof`` + ``mdat``, and the terminal ``end`` the run deserved;
5. the concatenated preview bytes are decoded **in process with PyAV** -- no
   external ``ffmpeg`` binary -- to prove the stream a browser would receive is
   a real, playable fMP4 with both streams in it.

The text lane is optional, and off by default
--------------------------------------------
Without ``--text-encoder`` / ``--prompt`` **no text encoder is loaded**. The
``CONDITIONING`` is then *synthetic but deterministic*: a fixed-seed CPU draw of
``[1, L, 5120]`` (the official Qwen3-VL width, so ``prefill_text`` still runs
``condition_proj`` and the token refiner exactly as it would on real states)
plus an all-text ``minimax_token_tags`` vector, both described in the report by
shape, dtype and the exact generator that drew them, so the input is
reproducible from the report alone.
What that mode verifies is the DiT / VAE / media / node lane; a claim about text
conditioning is **not** one of its outputs.

Given both flags, the probe instead runs the real thing, in the workflow's own
order and through upstream's own nodes:

* ``CLIPLoader`` -- with the ``type`` and ``device`` read off its **live**
  ``INPUT_TYPES()`` rather than hard-coded, because a wrong ``type`` silently
  loads the encoder as another family;
* ``comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo`` in its **T2VA** form
  (no ``first_frame``, no ``last_frame``), which owns the tokenizer, the encode
  and the empty AV latent. Nothing about the prompt is reimplemented here;
* the encode's load device is observed through ``load_models_gpu``, and the
  encoder is then offloaded with Comfy's own targeted
  ``unload_model_and_clones`` -- never a hand-written ``.to()``, which would
  leave Comfy's accounting describing a residency that no longer exists.

All of it happens *before* the 66 GB DiT is built, which is the memory story
``docs/requirements.md`` describes.

Nothing here downloads anything: every path is given, absolute, and must
already exist.

The old whole-clip decode, on request
-------------------------------------
``--compare-official-video`` (off by default) decodes the finished latent a
second time with ``nodes.decode_images`` -- the helper kept for exactly this --
and compares it to the collector's IMAGE. It runs **after** the node has
returned, i.e. after ``prepare_final_decode`` has already evicted the DiT, so
the reference decode gets the memory state it was always meant to have. It is
only attempted at small geometries (the recommended 39-frame 512x288 probe
request); 192- and 362-frame requests are refused rather than attempted,
because the whole-clip decode is the allocation that OOMed a measured run. An
OOM here is reported as a failure; a bitwise mismatch is reported as a number,
since a different kernel for a different tensor shape is a hardware fact. The
product outputs are not touched either way.

Determinism, memory and cancellation
------------------------------------
``--repeat N`` (1-10) runs the sampler N times **in the same process** with the
same seed and compares every run's final latent *bitwise against the first*.
Non-bitwise results are not softened into a tolerance: the max-abs and
relative-L2 differences are recorded and the determinism gate FAILS, because
"the same seed is the same clip" is a documented property of this node
(``nodes.py``: the rollout uses a private generator).

The memory gate distinguishes **warm-up from a leak**, which is why more runs
are better than two. Run 1 is where ``load_models_gpu`` makes the DiT's
ModelPatcher and both VAE patchers resident and where the caching allocator
grows its pools; that residency is supposed to survive into run 2 and is
reported (``loaded_models()`` per run, with each patcher's ``loaded_size`` and
device) rather than counted as a leak. So with three or more runs the gate is
the growth between the **last two** runs -- the plateau -- and the
first-to-last total is a diagnostic. With only two runs there is no plateau to
measure and the old strict gate stands. Every run must also leave no preview
session behind.

``--cancel-after-forward N`` wraps the causal DiT's ``forward_chunk``, lets
exactly ``N`` real forwards through, then raises the real
:class:`raven_streaming.consistency.SamplingCancelled`. The run must then
produce **no** partial output, end the preview with ``cancelled``, discard the
staged KV rows, leave no active session -- and be followed by a normal run that
succeeds, in the same process.

``--cancel-after-chunk N`` (mutually exclusive with it) is the *later*
cancellation point, and it answers a different question. A forward cancel stops
before anything has been decoded; this one lets ``N`` chunks go all the way
through ``StreamingPipeline.on_chunk`` -- both collectors, both real VAEs --
and raises from inside that callback. So when it lands there are pixels, PCM,
held frames and overlap state in memory, and the gates are about what happens
to them: at least one chunk really went through the VAEs, the decoders were
holding buffers at that instant, and afterwards the pipeline holds *nothing* --
no pending frames, no audio history, no IMAGE, no AUDIO. The decoders' own
``abort()`` is feature-probed rather than assumed: where it exists it must have
been called, and where it does not the effect gate still stands.

The KV cache lane
-----------------
``--kv-cache-storage {cpu_pinned,cpu,gpu}`` is passed straight through to
``RAVENStreamingSampler.sample`` and recorded, and every run publishes the
shape, dtype and statistics of its four output tensors -- both latent streams,
the IMAGE and the AUDIO. Whether the storage mode changed the result is
settled inside one process by ``--repeat``, which compares the tensors
themselves bitwise; the report describes the outputs, it does not stand in for
them.

A standard Comfy LoRA stacked on top, on request
------------------------------------------------
``--stacked-lora-name`` (off by default) takes a **``folder_paths`` "loras"
relative name**, exactly what the combo in a workflow holds, and chains
upstream's own ``nodes.LoraLoaderModelOnly`` after the RAVEN loader -- the node
is instantiated and its own ``FUNCTION`` (``load_lora_model_only``) is called.
``comfy.sd.load_lora_for_models`` is *observed*, never called directly: calling
it here would prove that Comfy can patch this model, not that the node a user
wires up can.

What that answers is the claim ``README.md`` makes -- "stock
``LoraLoaderModelOnly`` chains after our loader" -- and it is gated, not noted:

* the official node ran **exactly once**, and it is the class upstream still
  advertises (``FUNCTION`` / ``RETURN_TYPES`` read off the live class);
* at least one **non-zero** standard patch landed on a base ``diffusion_model.*``
  key. An empty patch set, or a patch set that is all zeros, fails;
* **no** key of the file went unmatched. Comfy reports those by logging
  (``lora key not loaded`` / ``NOT LOADED``), which is captured for the duration
  of the call, so a LoRA for another model is a failure rather than a warning
  scrolling past;
* **nothing** targets the RAVEN residual's own parameters. They do not end in
  ``.weight``, so ``model_lora_keys_unet`` still exposes them under the generic
  format, and a file carrying ``...raven_lora_A_0.diff`` would patch the
  mandatory adapter itself;
* the mandatory RAVEN attachment is still the **same object** with its 266
  modules (the node clones the patcher, not the model), and the patcher handed
  downstream is still a stock ``comfy.model_patcher.ModelPatcher``;
* the full sample then runs on the stacked ``MODEL`` and has to produce the
  whole product surface -- LATENT, IMAGE, AUDIO and a complete preview stream.

The file's resolved path and size go into the report, because "a LoRA" is not a
fact and *that* file is.

Usage
-----
::

    # synthetic conditioning; cancel + three sampled runs, so the memory gate
    # can tell warm-up from a leak
    python tools/probe_raven_integration.py \
        --comfy-root /root/ComfyUI \
        --base   /models/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
        --lora   /models/loras/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors \
        --video-vae /models/vae/minimax_h3_video_vae_fp16.safetensors \
        --audio-vae /models/vae/minimax_h3_audio_vae_fp32.safetensors \
        --device cuda --repeat 3 --cancel-after-forward 3 \
        --json .cache/probe_raven_integration.json

    # ... and the same thing with the real text lane switched on
    python tools/probe_raven_integration.py \
        ... \
        --text-encoder /models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
        --prompt "a cat playing a trumpet on a rooftop at sunset" \
        --json .cache/probe_raven_integration_text.json

    # ... and with a standard Comfy LoRA stacked on the MODEL wire by the
    # official LoraLoaderModelOnly (the name is folder_paths-relative, i.e.
    # <ComfyUI>/models/loras/<name>)
    python tools/probe_raven_integration.py \
        ... \
        --stacked-lora-name h3_extra_style.safetensors \
        --stacked-lora-strength 0.8 \
        --json .cache/probe_raven_integration_stacked.json

The report path must be inside this repository (``.cache/`` is gitignored and
is the intended home for run artifacts). It is written **atomically**, and it
is written even when the probe raises -- a crashed probe that leaves no
evidence is worth nothing.

Exit code: 0 only when every gate check passed and nothing raised.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import gc
import io
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

DEFAULT_COMFY = ROOT / ".cache" / "upstream" / "ComfyUI"
COMFY_ENV_VARS = ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH")
DEVICE_ENV_VAR = "RAVEN_PROBE_DEVICE"

#: Default report path, relative to the repository root. ``.cache/`` is
#: gitignored, which is where a run artifact belongs.
DEFAULT_REPORT = os.path.join(".cache", "probe_raven_integration.json")

#: Official Qwen3-VL-32B hidden width (``comfy/text_encoders/minimax.py``:
#: ``embedding_size=5120``). The synthetic conditioning uses it so the DiT's
#: ``condition_proj`` + token refiner run, exactly as they would on real states.
TEXT_EMBED_DIM = 5120

#: ``folder_paths`` folders the two upstream loader nodes read their combos from.
VAE_FOLDER = "vae"
TEXT_ENCODER_FOLDER = "text_encoders"

#: Where ``LoraLoaderModelOnly``'s combo comes from. ``--stacked-lora-name`` is
#: a name *relative to this folder*, which is what a workflow actually carries;
#: the probe resolves it the same way the node does rather than registering a
#: directory of its own, so "the name in the workflow" and "the file measured"
#: cannot drift apart.
LORA_FOLDER = "loras"

#: The official stacking node, and the method name it is *expected* to expose.
#: The method actually called is read off the live class's ``FUNCTION``; this
#: constant only says what upstream had at the pinned commit, so a rename is a
#: reported finding instead of an ``AttributeError``.
OFFICIAL_LORA_NODE = "LoraLoaderModelOnly"
OFFICIAL_LORA_FUNCTION = "load_lora_model_only"

#: How Comfy reports a LoRA key it could not place: ``comfy.lora.load_lora``
#: logs "lora key not loaded: <k>" for a key no adapter claimed, and
#: ``comfy.sd.load_lora_for_models`` logs "NOT LOADED <k>" for a patch that
#: matched no model key. Both are warnings on the root logger, i.e. invisible
#: to a return value -- which is why the probe captures them.
UNMATCHED_LORA_LOG_MARKERS = ("lora key not loaded", "not loaded")

#: Fallback names of the RAVEN residual's own parameters, used when
#: ``raven_streaming.runtime_linear`` cannot be imported to supply the real
#: templates. Nothing a stacked LoRA does may target these.
RAVEN_RESIDUAL_PARAM_PREFIXES = ("raven_lora_A_", "raven_lora_B_")

#: Per-key patch detail kept in the report. The target-key list is complete
#: (it is the evidence); the tensor-level detail is capped, because a real H3
#: LoRA has hundreds of modules and the shapes repeat.
STACKED_PATCH_DETAIL_LIMIT = 16

#: Where the sampler keeps the committed chunk KV cache
#: (``nodes.KV_CACHE_STORAGE_CHOICES`` / ``nodes.DEFAULT_KV_CACHE_STORAGE``).
#: Mirrored here so ``build_parser`` stays importable without the package, and
#: cross-checked against the node's *live* tuple once the run starts -- a
#: silently renamed mode would otherwise be rejected deep inside ``sample()``
#: instead of by the flag that offered it.
KV_CACHE_STORAGE_CHOICES = ("cpu_pinned", "cpu", "gpu")
DEFAULT_KV_CACHE_STORAGE = "cpu_pinned"

#: What ``CLIPLoader``'s ``type`` combo calls the MiniMax family. Matched
#: case-insensitively against the *live* schema; never assumed to be present.
MINIMAX_CLIP_TYPE_NAMES = frozenset({"minimax"})

#: Salt mixed into the conditioning seed so the text draw cannot accidentally
#: coincide with the rollout's own noise stream for the same ``--seed``.
CONDITIONING_SEED_SALT = 0x5241_5645  # "RAVE"

#: Node id used for the preview session. Any stable non-empty string works;
#: ``normalise_node_id`` only rejects ``None`` and litegraph's ``-1``.
PROBE_NODE_ID = "raven-integration-probe"

#: The AUDIO contract: the H3 audio VAE is 32 kHz stereo.
AUDIO_SAMPLE_RATE = 32000
AUDIO_OUTPUT_CHANNELS = 2

#: Pixels are expected in [0, 1]; a hair of slack for the VAE's own arithmetic.
PIXEL_RANGE_TOLERANCE = 1e-4

#: Samples one H3 audio latent decodes to. Published, and the number the
#: collector's own ``expected_samples`` is built from.
AUDIO_SAMPLES_PER_LATENT = 800

#: What the official whole-clip normalisation leaves behind. ``vae_decode_audio``
#: divides by ``max(1, std * 5)``, so a clip loud enough to be scaled comes out
#: at exactly ``std == 0.2`` and a quiet one is left below it. Either way the
#: result cannot exceed this.
AUDIO_NORMALISED_STD = 0.2
AUDIO_NORMALISED_STD_TOLERANCE = 1e-3

#: "No material growth" between two runs in one process. Both are deliberately
#: generous: the point is to catch a leak that scales with runs (a retained KV
#: cache, a session that was never released), not allocator noise.
CUDA_GROWTH_TOLERANCE_BYTES = 256 * 1024 * 1024
RSS_GROWTH_TOLERANCE_BYTES = 2 * 1024 * 1024 * 1024

#: Upper bound on ``--repeat``. Each run is a full rollout and a full decode.
MAX_REPEAT = 10

#: Ceiling on the optional whole-clip reference decode
#: (``--compare-official-video``), in decoded pixels: the recommended 39-frame
#: 512x288 probe geometry, and nothing larger. 192 and 362 frames are far past
#: it, which is the point -- the whole-clip decode is the allocation the
#: product path removed after it OOMed a measured run.
OFFICIAL_COMPARE_MAX_PIXELS = 39 * 512 * 288


class ProbeError(RuntimeError):
    """The probe cannot run as asked (a missing file, a mis-wired socket)."""


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


@dataclass
class Check:
    """One observation. ``gate`` decides whether it can fail the probe."""

    name: str
    ok: bool
    detail: str = ""
    gate: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "gate": self.gate}

    def line(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.gate else "warn")
        tail = " - {}".format(self.detail) if self.detail else ""
        return "[{}] {}{}".format(mark, self.name, tail)


class Checks:
    """An ordered list of :class:`Check`, with the two ways of adding one."""

    def __init__(self) -> None:
        self.items: List[Check] = []

    def expect(self, name: str, condition: Any, detail: str = "", *, gate: bool = True) -> bool:
        ok = bool(condition)
        self.items.append(Check(name=name, ok=ok, detail=detail, gate=gate))
        return ok

    def note(self, name: str, detail: str = "") -> None:
        """Something worth recording that cannot fail anything."""
        self.items.append(Check(name=name, ok=True, detail=detail, gate=False))

    def fail(self, name: str, detail: str = "", *, gate: bool = True) -> None:
        self.items.append(Check(name=name, ok=False, detail=detail, gate=gate))

    def extend(self, items: Sequence[Check]) -> None:
        self.items.extend(items)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.items if c.gate and not c.ok]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.items]


def prefixed(items: Sequence[Check], prefix: str) -> List[Check]:
    """Re-label checks so two runs' identical names stay distinguishable."""
    return [
        Check(name="{}: {}".format(prefix, c.name), ok=c.ok, detail=c.detail, gate=c.gate)
        for c in items
    ]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@dataclass
class Report:
    """Everything the probe saw. Written even when the probe raised."""

    args: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    setup: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    determinism: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c.get("gate", True) and not c.get("ok", False)]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "probe": "raven_integration",
            "args": dict(self.args),
            "environment": dict(self.environment),
            "inputs": dict(self.inputs),
            "setup": dict(self.setup),
            "runs": list(self.runs),
            "determinism": dict(self.determinism),
            "checks": list(self.checks),
            "failures": self.failures,
            "notes": list(self.notes),
            "errors": list(self.errors),
        }

    def render(self) -> str:
        lines: List[str] = []
        env = self.environment
        if env:
            lines.append(
                "ComfyUI {} @ {} (version {}) | torch {} | device {} | kv cache {}".format(
                    env.get("comfy_root"),
                    env.get("comfy_commit"),
                    env.get("comfy_version"),
                    env.get("torch_version"),
                    env.get("device"),
                    env.get("kv_cache_storage", self.args.get("kv_cache_storage")),
                )
            )
        if self.inputs:
            geometry = self.inputs.get("geometry", {})
            lines.append(
                "lane: {}".format(self.inputs.get("lane", "synthetic"))
                + (
                    ""
                    if not self.inputs.get("text_lane")
                    else " | text encoder {} load {:.2f}s encode {:.2f}s | tags {}".format(
                        self.inputs["text_lane"].get("name"),
                        self.inputs["text_lane"].get("load_seconds", 0.0),
                        self.inputs["text_lane"].get("encode_seconds", 0.0),
                        (self.inputs["text_lane"].get("token_tags") or {}).get("histogram"),
                    )
                )
            )
            lines.append(
                "request: {}x{} frames={} (k={}) latent_t={} audio_t={} text_len={} "
                "steps={} seed={}".format(
                    geometry.get("width"),
                    geometry.get("height"),
                    geometry.get("frames"),
                    geometry.get("k"),
                    geometry.get("latent_t"),
                    geometry.get("audio_t"),
                    self.inputs.get("text_len"),
                    self.args.get("steps"),
                    self.args.get("seed"),
                )
            )
        if self.setup:
            lines.append(
                "setup: model build {:.2f}s, video VAE {:.2f}s, audio VAE {:.2f}s".format(
                    self.setup.get("model_build_seconds", 0.0),
                    self.setup.get("video_vae_seconds", 0.0),
                    self.setup.get("audio_vae_seconds", 0.0),
                )
            )
            stacked = self.setup.get("stacked_lora")
            if stacked:
                lines.append(
                    "stacked lora: {}.{}({!r}, strength={}) -> {} patched key(s), "
                    "{} non-zero, {} unmatched | {} ({} bytes)".format(
                        stacked.get("node_class"),
                        stacked.get("node_function"),
                        stacked.get("lora_name"),
                        stacked.get("strength"),
                        stacked.get("target_key_count"),
                        stacked.get("nonzero_key_count"),
                        len(stacked.get("unmatched_warnings") or []),
                        stacked.get("path"),
                        stacked.get("bytes"),
                    )
                )
        for run in self.runs:
            timings = run.get("timings", {})
            preview = run.get("preview", {})
            media = run.get("media", {})
            lines.append(
                "run {} [{}]: sample {:.2f}s (load {:.2f}s, preview flush {:.2f}s, "
                "image finalize {:.4f}s, audio finalize {:.4f}s)".format(
                    run.get("index"),
                    run.get("kind"),
                    timings.get("sample_seconds", 0.0),
                    timings.get("model_load_seconds", 0.0),
                    timings.get("preview_flush_seconds", 0.0),
                    timings.get("image_finalize_seconds", 0.0),
                    timings.get("audio_finalize_seconds", 0.0),
                )
            )
            pipeline = run.get("pipeline", {})
            if pipeline:
                lines.append(
                    "        collector: {}/{} frame(s) into {} {} on {}".format(
                        pipeline.get("collected_frames"),
                        pipeline.get("expected_frames"),
                        pipeline.get("image_shape"),
                        pipeline.get("image_dtype"),
                        pipeline.get("image_device"),
                    )
                )
            streaming = run.get("streaming", {})
            if streaming:
                lines.append(
                    "        streaming: {}/{} fragment(s) out before the flush, "
                    "{}/{} muxed frame(s); emitting chunks {}".format(
                        streaming.get("fragments_before_finish"),
                        streaming.get("fragments"),
                        streaming.get("frames_muxed_before_finish"),
                        streaming.get("frames_muxed_total"),
                        streaming.get("emitting_chunks"),
                    )
                )
            cancel = run.get("chunk_cancel")
            if cancel:
                lines.append(
                    "        chunk cancel: raised after {} delivered chunk(s); "
                    "decoded {} frame(s)/{} sample(s) before, held {} after; "
                    "aborts {}".format(
                        cancel.get("chunks_delivered"),
                        (cancel.get("state_before") or {}).get("frames_decoded"),
                        (cancel.get("state_before") or {}).get("samples_decoded"),
                        (cancel.get("state_after") or {}).get("pending_frames"),
                        cancel.get("decoder_aborts"),
                    )
                )
            phase = run.get("phase_swap", {})
            if phase:
                lines.append(
                    "        phase swap: {} DiT phase load(s) for {} chunk(s) "
                    "({} delivered){}".format(
                        phase.get("observed_dit_loads"),
                        phase.get("layout_chunks"),
                        phase.get("delivered_chunks"),
                        ""
                        if "phase_swap_vae_loads" not in phase
                        else "; {} VAE phase load(s), ended in the {} phase".format(
                            phase.get("phase_swap_vae_loads"),
                            phase.get("phase_swap_last_phase"),
                        ),
                    )
                )
            compare = run.get("official_compare")
            if compare:
                metrics = compare.get("metrics", {})
                lines.append(
                    "        official decode: {} | {} | extra peak {}".format(
                        compare.get("error")
                        or "{:.2f}s".format(compare.get("seconds", 0.0)),
                        "skipped"
                        if not compare.get("allowed")
                        else "bitwise={} max|d|={}".format(
                            metrics.get("bitwise"), metrics.get("max_abs")
                        ),
                        _gib(compare.get("extra_peak_allocated", 0)),
                    )
                )
            lines.append(
                "        preview: {} message(s), {} segment(s) / {} B, end={} | "
                "decoded {} frame(s), {} audio sample(s)".format(
                    preview.get("messages", 0),
                    preview.get("segments", 0),
                    preview.get("segment_bytes", 0),
                    preview.get("end_reason"),
                    media.get("video_frames"),
                    media.get("audio_samples"),
                )
            )
            memory = run.get("memory", {})
            lines.append(
                "        memory: cuda peak alloc {} / reserved {} | rss {} (peak {})".format(
                    _gib(memory.get("cuda_peak_allocated", 0)),
                    _gib(memory.get("cuda_peak_reserved", 0)),
                    _gib(memory.get("rss_after", 0)),
                    _gib(memory.get("rss_peak", 0)),
                )
            )
        if self.determinism:
            for comparison in self.determinism.get("comparisons", []):
                latent = comparison.get("latent", {})
                lines.append(
                    "determinism: run {} vs run 1 -- video bitwise={} audio bitwise={} "
                    "(max|d| {} / {})".format(
                        comparison.get("run"),
                        latent.get("video", {}).get("bitwise"),
                        latent.get("audio", {}).get("bitwise"),
                        latent.get("video", {}).get("max_abs"),
                        latent.get("audio", {}).get("max_abs"),
                    )
                )
            reproducibility = self.determinism.get("image_reproducibility")
            if reproducibility:
                lines.append(
                    "IMAGE/AUDIO reproducibility: pair {} {} -- IMAGE bitwise={} "
                    "AUDIO bitwise={}".format(
                        reproducibility.get("gated_pair"),
                        "GATED" if reproducibility.get("gated") else "diagnostic",
                        reproducibility.get("images_bitwise"),
                        reproducibility.get("audio_bitwise"),
                    )
                )
            for name, grid in (self.determinism.get("bitwise_matrix") or {}).items():
                lines.append(
                    "bitwise matrix {}: {}".format(
                        name, " ".join("".join("1" if v else "." for v in row) for row in grid)
                    )
                )
            memory = self.determinism.get("memory", {})
            if memory:
                lines.append(
                    "memory gate: {} over {} -- gated {}, total {}".format(
                        memory.get("mode"),
                        memory.get("gate_window", "n/a"),
                        memory.get("gated"),
                        memory.get("diagnostic_total", "n/a"),
                    )
                )
        lines.extend("note: " + n for n in self.notes)
        lines.append("")
        lines.extend(Check(**c).line() for c in self.checks)
        lines.extend("ERROR: " + e.splitlines()[0] for e in self.errors)
        lines.append("")
        lines.append(
            "RESULT: {} ({} check(s), {} failure(s), {} error(s))".format(
                "ok" if self.ok else "FAILED",
                len(self.checks),
                len(self.failures),
                len(self.errors),
            )
        )
        return "\n".join(lines)


def _gib(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "{:.3f}GiB".format(number / (1024 ** 3))


# --------------------------------------------------------------------------
# report output: inside the repo, atomic, always written
# --------------------------------------------------------------------------


def resolve_report_path(value: str, root: Path = ROOT) -> Path:
    """Absolute path for ``--json``, refused if it escapes ``root``.

    A probe writes exactly one artifact and it belongs in the repository (which
    gitignores ``.cache/``). Anything else -- a shared filesystem, a home
    directory, ``/tmp`` -- is a write the caller did not ask for and cannot
    find later.
    """
    root = Path(root).expanduser().resolve()
    candidate = Path(os.path.expanduser(str(value)))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = Path(os.path.normpath(str(candidate)))
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ProbeError(
            "the report path must be inside {} (got {}); probes write their "
            "artifacts into the repository, and .cache/ is gitignored for exactly "
            "this".format(root, resolved)
        ) from None
    return resolved


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write ``payload`` as JSON with no observable half-written state.

    Same-directory temporary plus ``os.replace``: a reader either sees the
    previous report or the whole new one, never a truncated file, even if the
    probe is killed mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def resolve_comfy_root(explicit: Optional[str] = None) -> Path:
    """``--comfy-root`` > ``COMFYUI_PATH`` > ``COMFYUI_UPSTREAM_PATH`` > cache.

    An explicit ``--comfy-root`` is **not** allowed to fall back: silently
    probing a different checkout than the one named on the command line is how
    a result ends up describing code nobody ran. It is returned as given and
    :func:`import_comfy` says why it is unusable.
    """
    if explicit:
        return Path(explicit).expanduser()
    for candidate in [os.environ.get(v) for v in COMFY_ENV_VARS] + [str(DEFAULT_COMFY)]:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "comfy" / "model_patcher.py").is_file():
            return path.resolve()
    return DEFAULT_COMFY


def default_device() -> str:
    return os.environ.get(DEVICE_ENV_VAR) or ("cuda" if torch.cuda.is_available() else "cpu")


def comfy_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - diagnostics only
        return "unknown"


@dataclass
class ComfyEnv:
    """The upstream modules this probe drives, imported once."""

    root: Path
    comfy: Any
    folder_paths: Any
    #: upstream ``nodes.py`` -- ``VAELoader`` and ``CLIPLoader``
    upstream_nodes: Any
    #: ``comfy_extras.nodes_minimax_h3`` -- the official T2VA node. ``None``
    #: when this ComfyUI has no H3 support, which the text lane refuses to
    #: work around.
    minimax_nodes: Any = None
    commit: str = "unknown"
    version: str = ""


def import_comfy(root: Path) -> ComfyEnv:
    """Import upstream ComfyUI, including ``nodes.py`` (for ``VAELoader``).

    ``sys.argv`` is masked for the duration: ``comfy.cli_args`` parses the
    process argv at import time and would choke on this probe's flags.
    """
    if not (root / "comfy" / "model_patcher.py").is_file():
        raise ProbeError("no ComfyUI checkout at {}".format(root))
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        import comfy.model_management  # noqa: F401
        import comfy.nested_tensor  # noqa: F401
        import comfy.sd  # noqa: F401
        import comfy.utils  # noqa: F401
        import folder_paths  # type: ignore[import-not-found]
        import nodes as upstream_nodes  # type: ignore[import-not-found]

        try:
            from comfy_extras import nodes_minimax_h3 as minimax_nodes  # type: ignore
        except Exception:  # noqa: BLE001 - only the optional text lane needs it
            minimax_nodes = None
    finally:
        sys.argv = saved_argv

    version = ""
    try:
        from raven_streaming import compat

        version = compat.comfy_version()
    except Exception:  # noqa: BLE001 - a version string is not a gate
        version = ""
    return ComfyEnv(
        root=root,
        comfy=sys.modules["comfy"],
        folder_paths=folder_paths,
        upstream_nodes=upstream_nodes,
        minimax_nodes=minimax_nodes,
        commit=comfy_commit(root),
        version=version,
    )


# --------------------------------------------------------------------------
# memory accounting
# --------------------------------------------------------------------------


def peak_host_rss_bytes() -> int:
    """``ru_maxrss`` normalised to bytes (KiB on Linux, bytes on macOS)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if platform.system() == "Darwin" else int(raw) * 1024


def current_host_rss_bytes() -> int:
    """Resident set *now*. ``ru_maxrss`` is a high-water mark and never falls,
    so it cannot answer "did the second run grow the process?"."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - not Linux, or no procfs
        pass
    return peak_host_rss_bytes()


def cuda_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


# --------------------------------------------------------------------------
# inputs: the synthetic conditioning and the official empty AV latent
# --------------------------------------------------------------------------


def build_conditioning(
    text_len: int, seed: int, *, dim: int = TEXT_EMBED_DIM
) -> Tuple[List[List[Any]], Dict[str, Any]]:
    """A deterministic stand-in for one official T2VA ``CONDITIONING`` entry.

    Drawn on the **CPU** from a private generator, never the global stream, and
    never on the GPU: a CPU draw is bit-identical across devices and driver
    versions, so the seed and the generator recorded in the report are enough to
    reproduce the input everywhere. The rollout moves it to the compute device
    and casts it itself, exactly as it does with real encoder output.

    ``minimax_token_tags`` is all-``TEXT_TAG``: the official tag vector marks
    the vision spans of a multimodal prompt, and this probe sends none.
    """
    from raven_streaming import layout as layout_mod

    if int(text_len) < 1:
        raise ProbeError("text_len must be >= 1, got {!r}".format(text_len))
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) ^ CONDITIONING_SEED_SALT) & 0xFFFF_FFFF_FFFF_FFFF)
    cross_attn = torch.randn(
        (1, int(text_len), int(dim)), generator=generator, dtype=torch.float32
    )
    tags = torch.full((int(text_len),), layout_mod.TEXT_TAG, dtype=torch.int64)
    conditioning = [[cross_attn, {"minimax_token_tags": tags}]]
    describe = {
        "source": "synthetic (no text encoder is loaded by this probe)",
        "shape": list(cross_attn.shape),
        "dtype": str(cross_attn.dtype),
        "device": "cpu",
        "generator": "torch.Generator(device='cpu').manual_seed(seed ^ 0x{:X})".format(
            CONDITIONING_SEED_SALT
        ),
        "seed": int(seed),
        "token_tags": {
            "value": int(layout_mod.TEXT_TAG),
            "shape": list(tags.shape),
            "dtype": str(tags.dtype),
        },
        "caveat": (
            "this probe verifies the DiT / VAE / media / node lane only; it makes no "
            "claim about the Qwen3-VL text encoder"
        ),
    }
    return conditioning, describe


def describe_rollout_rng(seed: int, device: Any) -> Dict[str, Any]:
    """How the *rollout's* noise is drawn, recorded next to the input's own draw.

    Two different generators decide a run and they are not interchangeable:
    the conditioning above is a CPU draw made by this probe, while every noise
    tensor inside the rollout comes from ``consistency.RolloutRNG``, a private
    ``torch.Generator`` on the **compute device**. On CUDA that makes bitwise
    reproducibility a property of the device RNG *and* of the kernels, which is
    exactly what ``--repeat 2`` gates rather than assumes.
    """
    return {
        "seed": int(seed),
        "generator": "torch.Generator(device={!r}).manual_seed(seed)".format(str(device)),
        "generator_device": str(device),
        "private": True,
        "source": "raven_streaming.consistency.RolloutRNG",
        "note": (
            "the rollout never touches global RNG; the draw order is recorded by the "
            "sampler itself and the same seed must give the same clip, which --repeat 2 "
            "checks bitwise"
        ),
    }


def latent_geometry(frames: int, width: int, height: int) -> Dict[str, int]:
    """The ``17k + 5`` grid, taken from the package rather than re-derived.

    ``layout.video_latent_t`` / ``audio_latent_t`` validate the request on the
    way through, so an off-grid ``--frames`` fails here with the node's own
    message instead of producing a latent the sampler will later refuse.
    """
    from raven_streaming import layout as layout_mod

    k = layout_mod.validate_frames(int(frames), warn_experimental=False)
    layout_mod.validate_canvas(int(width), int(height))
    return {
        "frames": int(frames),
        "k": int(k),
        "width": int(width),
        "height": int(height),
        "latent_t": int(layout_mod.video_latent_t(int(frames))),
        "latent_h": int(height) // 16,
        "latent_w": int(width) // 16,
        "audio_t": int(layout_mod.audio_latent_t(int(frames))),
    }


def layout_num_chunks(geometry: Dict[str, int], text_len: int = 1) -> int:
    """How many chunks the rollout will deliver for this request.

    Read off :class:`raven_streaming.layout.T2VALayout`, never counted by hand:
    the cut is ``5`` video latents per chunk plus a ``2``-latent tail, so
    ``latent_t = 5k + 2`` gives ``k + 1`` chunks -- 3 at 39 frames, 12 at 192,
    22 at 362. Deriving it here is what lets the phase-swap gate below be an
    expectation rather than a hard-coded number.
    """
    from raven_streaming import layout as layout_mod

    layout = layout_mod.T2VALayout.from_request(
        text_len=max(1, int(text_len)),
        frames=int(geometry["frames"]),
        width=int(geometry["width"]),
        height=int(geometry["height"]),
        warn_experimental=False,
    )
    return int(layout.num_chunks)


def expected_dit_loads(num_chunks: int) -> int:
    """DiT-phase load calls a *completed* rollout makes: one per chunk.

    The phase swap (``nodes.PhaseSwapCoordinator``) alternates residency around
    every chunk, so the DiT closure is called:

    * **once** by ``consistency.sample_streaming`` before the first forward, and
    * **once after each non-last chunk**, to get the DiT back for the next
      forward -- the last chunk deliberately leaves the VAEs loaded for the
      video/audio collector tail flush, which is where the last overlap-save
      blocks of *both* streams are decoded. ``finalize_image`` /
      ``finalize_audio`` afterwards decode nothing: they normalise and return
      host buffers the flush already filled.

    That is ``1 + (num_chunks - 1) == num_chunks``. It is a count of *calls*,
    not of evictions: on a card that can hold everything at once upstream moves
    nothing and each call is a no-op, which is a property of the machine, not
    of the node.
    """
    return max(1, int(num_chunks))


def dit_loads_after_cancel(delivered_chunks: int, num_chunks: int) -> int:
    """The same count for a rollout that stopped mid-flight.

    A cancelled run has made the initial load plus one reload per chunk it
    actually delivered (capped at the non-last chunks, since the last one never
    reloads). Derived from what the run reports having done rather than
    assumed, because where a cancellation lands is the caller's choice.

    This is the ``--cancel-after-forward`` shape: the cancel lands in a forward,
    i.e. *after* the coordinator has already reloaded the DiT for it.
    """
    delivered = max(0, int(delivered_chunks))
    non_last = min(delivered, max(0, int(num_chunks) - 1))
    return 1 + non_last


def dit_loads_after_chunk_cancel(delivered_chunks: int, num_chunks: int) -> int:
    """The same count for ``--cancel-after-chunk``, which stops one step earlier.

    ``PhaseSwapCoordinator.on_chunk`` is *VAE phase -> deliver -> DiT phase*,
    and the chunk cancel is raised out of the delivery. So the reload that
    follows the last delivered chunk never happens: the run has made the
    initial load plus one reload per chunk it delivered **before** the last one.

    Being one lower than :func:`dit_loads_after_cancel` at the same chunk is the
    whole point of the distinction; using the wrong one would report a missing
    DiT load as a finding.
    """
    delivered = max(0, int(delivered_chunks))
    non_last = min(max(0, delivered - 1), max(0, int(num_chunks) - 1))
    return 1 + non_last


def build_empty_latent(
    geometry: Dict[str, int],
    nested_cls: Any,
    *,
    device: Any = "cpu",
    dtype: Any = torch.float32,
) -> Dict[str, Any]:
    """The official empty AV ``LATENT``: a ``NestedTensor`` of two zero tensors.

    Same construction as ``comfy_extras.nodes_minimax_h3._empty_av_latent``:
    ``[1, 24, latent_t, H/16, W/16]`` video and ``[1, 32, 2, audio_t]`` audio,
    both zero, wrapped in the *official* ``NestedTensor`` class -- which is
    passed in rather than imported so this function is testable without a
    checkout.
    """
    from raven_streaming import contracts

    video = torch.zeros(
        [
            1,
            contracts.VIDEO_LATENT_CHANNELS,
            int(geometry["latent_t"]),
            int(geometry["latent_h"]),
            int(geometry["latent_w"]),
        ],
        device=device,
        dtype=dtype,
    )
    audio = torch.zeros(
        [1, contracts.AUDIO_LATENT_CHANNELS, AUDIO_OUTPUT_CHANNELS, int(geometry["audio_t"])],
        device=device,
        dtype=dtype,
    )
    return {"samples": nested_cls((video, audio))}


def check_geometry_against_upstream(geometry: Dict[str, int], checks: Checks) -> None:
    """Cross-check the grid against upstream's own ``temporal_shape``.

    The geometry is computed from this package's layout module; upstream
    computes it in ``comfy_extras.nodes_minimax_h3``. If the two ever disagree,
    every latent this probe builds is the wrong shape and nothing downstream
    would say so.
    """
    try:
        from comfy_extras import nodes_minimax_h3  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        checks.note(
            "geometry cross-check skipped",
            "comfy_extras.nodes_minimax_h3 not importable ({}: {})".format(
                type(exc).__name__, exc
            ),
        )
        return
    frame_count, latent_t, audio_t = nodes_minimax_h3.temporal_shape(int(geometry["frames"]))
    checks.expect(
        "geometry matches upstream temporal_shape",
        (frame_count, latent_t, audio_t)
        == (int(geometry["frames"]), int(geometry["latent_t"]), int(geometry["audio_t"])),
        "upstream says frames={} latent_t={} audio_t={}, probe says frames={} "
        "latent_t={} audio_t={}".format(
            frame_count,
            latent_t,
            audio_t,
            geometry["frames"],
            geometry["latent_t"],
            geometry["audio_t"],
        ),
    )


# --------------------------------------------------------------------------
# model sockets: drive upstream's own loaders, never bypass them
# --------------------------------------------------------------------------


def register_model_files(folder_paths: Any, folder: str, paths: Sequence[Path]) -> List[str]:
    """Make ``folder_paths`` able to resolve these exact files, return basenames.

    Upstream's loader nodes (``VAELoader``, ``CLIPLoader``) take a *name* out of
    a combo and resolve it with ``folder_paths.get_full_path_or_raise(folder,
    name)``. Registering the files' directories (upstream's documented
    extension point, ``add_model_folder_path``) and handing over the basename is
    what lets a probe use arbitrary absolute paths **through** the real node
    instead of around it.

    The resolution is then verified: a same-named file already registered
    somewhere else would otherwise shadow ours and the probe would silently
    measure a different file. That check is the whole point of doing this here
    rather than trusting ``is_default=True`` to win.
    """
    resolved: List[Path] = []
    for path in paths:
        candidate = Path(os.path.expanduser(str(path)))
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_file():
            raise ProbeError(
                "{} file does not exist: {} (nothing is downloaded)".format(folder, candidate)
            )
        resolved.append(candidate.resolve())

    for candidate in resolved:
        folder_paths.add_model_folder_path(folder, str(candidate.parent), is_default=True)

    # The filename list is cached per folder; a stale entry only affects the
    # combo, but clearing it keeps INPUT_TYPES honest if anything reads it.
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop(folder, None)
    helper = getattr(folder_paths, "cache_helper", None)
    clear = getattr(helper, "clear", None)
    if callable(clear):
        with contextlib.suppress(Exception):
            clear()

    names: List[str] = []
    for candidate in resolved:
        found = folder_paths.get_full_path(folder, candidate.name)
        if found is None or Path(found).resolve() != candidate:
            raise ProbeError(
                "folder_paths resolves '{}/{}' to {!r}, not {}; another file of the "
                "same name is shadowing it".format(folder, candidate.name, found, candidate)
            )
        names.append(candidate.name)
    return names


# --------------------------------------------------------------------------
# the optional stacked standard LoRA (--stacked-lora-name), through the
# official LoraLoaderModelOnly and nothing else
# --------------------------------------------------------------------------


def stacked_lora_requested(args: argparse.Namespace) -> bool:
    """Is the optional stacked-LoRA lane switched on?"""
    return bool(getattr(args, "stacked_lora_name", None))


def stacked_lora_strength(args: argparse.Namespace) -> float:
    """``--stacked-lora-strength``, defaulting to the node's own default.

    ``1.0`` is what ``LoraLoaderModelOnly``'s schema offers, so an unspecified
    strength means "the workflow default", not "whatever the probe felt like".
    """
    value = getattr(args, "stacked_lora_strength", None)
    return 1.0 if value is None else float(value)


def raven_residual_prefixes() -> Tuple[str, ...]:
    """The parameter-name prefixes the RAVEN residual registers on a base linear.

    Read off ``raven_streaming.runtime_linear`` rather than hard-coded, so a
    rename there cannot silently turn this probe's "nothing targeted the
    residual" gate into a check of a string that no longer exists.
    """
    try:
        from raven_streaming import runtime_linear

        prefixes = tuple(
            template.split("{", 1)[0]
            for template in (runtime_linear.A_PARAM_TEMPLATE, runtime_linear.B_PARAM_TEMPLATE)
        )
    except Exception:  # noqa: BLE001 - the fallback is the published spelling
        return RAVEN_RESIDUAL_PARAM_PREFIXES
    return tuple(p for p in prefixes if p) or RAVEN_RESIDUAL_PARAM_PREFIXES


def file_info(path: Path) -> Dict[str, Any]:
    """Resolved path and size of a file the probe was pointed at.

    "a LoRA was stacked" is not a fact anyone can re-check; *which* file was is.
    Both facts come from the filesystem metadata -- the file itself is never
    read here, because a LoRA can be gigabytes and the loader is about to read
    it anyway.
    """
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
    }


class CallCounter:
    """How many times a wrapped callable actually ran."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.calls = 0
        #: Only the scalar arguments of each call. Tensors and model objects are
        #: deliberately *not* retained: a LoRA state dict passed to
        #: ``load_lora_for_models`` is gigabytes, and a counter that outlives the
        #: call would keep it resident.
        self.scalar_args: List[Tuple[Any, ...]] = []


@contextlib.contextmanager
def count_calls(owner: Any, attribute: str, counter: Optional[CallCounter] = None) -> Iterator[CallCounter]:
    """Count calls to ``owner.attribute`` **while still calling the original**.

    This is instrumentation, not substitution: the wrapper's only job is to
    increment and record, and every call reaches upstream's own implementation.
    It is what lets the probe say "the official node ran exactly once" about a
    node it did not reimplement.
    """
    counter = counter or CallCounter("{}.{}".format(getattr(owner, "__name__", owner), attribute))
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counter.calls += 1
        counter.scalar_args.append(
            tuple(a for a in args if a is None or isinstance(a, (str, int, float, bool)))
        )
        return original(*args, **kwargs)

    setattr(owner, attribute, wrapper)
    try:
        yield counter
    finally:
        setattr(owner, attribute, original)


@contextlib.contextmanager
def capture_lora_warnings() -> Iterator[List[str]]:
    """Collect the root-logger warnings Comfy uses to report unmatched keys.

    ``comfy.lora.load_lora`` and ``comfy.sd.load_lora_for_models`` do not return
    the keys they failed to place -- they log them. A probe that only looked at
    the return value would call a LoRA built for another model "applied".
    """
    collected: List[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - handler
            try:
                message = record.getMessage()
            except Exception:  # noqa: BLE001 - a broken record is not a finding
                return
            collected.append(message)

    handler = _Collector(level=logging.WARNING)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    # A root logger raised *above* WARNING would drop the very records this is
    # here to see. It is lowered only in that case (never raised, which would
    # silence someone else's INFO output) and restored on the way out.
    if previous_level > logging.WARNING:
        root.setLevel(logging.WARNING)
    try:
        yield collected
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def unmatched_lora_messages(messages: Sequence[str]) -> List[str]:
    """The captured warnings that mean "this key was not applied"."""
    out: List[str] = []
    for message in messages:
        lowered = str(message).lower()
        if any(marker in lowered for marker in UNMATCHED_LORA_LOG_MARKERS):
            out.append(str(message))
    return out


def patch_key_counts(patcher: Any) -> Dict[str, int]:
    """``{model key: number of stacked patches}`` for a ``ModelPatcher``."""
    patches = getattr(patcher, "patches", None)
    if not isinstance(patches, dict):
        return {}
    return {str(key): len(value or []) for key, value in patches.items()}


def _patch_tensors(patch: Any) -> List[Any]:
    """The tensors a single patch is made of, whatever shape upstream gave it.

    A patch is either a ``WeightAdapterBase`` (``.weights``, which is what a
    LoRA file becomes) or one of the plain tuples ``("diff", (w,))`` /
    ``("set", (w,))``. Both are unwrapped here rather than assumed, because the
    only question being asked of them is "is any of this non-zero".
    """
    weights = getattr(patch, "weights", None)
    if weights is None:
        weights = patch
    stack = [weights]
    tensors: List[Any] = []
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return tensors


def describe_patch(entry: Sequence[Any]) -> Dict[str, Any]:
    """One entry of ``ModelPatcher.patches[key]``, as evidence.

    ``add_patches`` appends ``(strength_patch, patch, strength_model, offset,
    function)``. "Non-zero" means every factor of the delta is non-zero: a LoRA
    whose ``lora_up`` is all zeros (the state a freshly initialised adapter is
    in) contributes exactly nothing, and reporting that as a hit would make the
    gate pass on a LoRA that does not change a single pixel.
    """
    strength_patch = float(entry[0]) if len(entry) > 0 else 0.0
    patch = entry[1] if len(entry) > 1 else None
    strength_model = float(entry[2]) if len(entry) > 2 else 1.0
    tensors = _patch_tensors(patch)
    finite = all(bool(torch.isfinite(t).all()) for t in tensors)
    nonzero_tensors = [bool(torch.any(t != 0)) for t in tensors]
    nonzero = (
        bool(tensors)
        and strength_patch != 0.0
        and strength_model != 0.0
        and all(nonzero_tensors)
    )
    kind = type(patch).__name__
    if isinstance(patch, tuple) and patch and isinstance(patch[0], str):
        kind = str(patch[0])
    return {
        "kind": kind,
        "strength_patch": strength_patch,
        "strength_model": strength_model,
        "tensors": [list(t.shape) for t in tensors],
        "dtypes": sorted({str(t.dtype) for t in tensors}),
        "nonzero": nonzero,
        "finite": finite,
    }


def summarise_stacked_patches(
    before: Dict[str, int],
    patcher: Any,
    *,
    residual_prefixes: Sequence[str],
) -> Dict[str, Any]:
    """What the official node added to the patcher, key by key.

    Only the *difference* is attributed to the stacked LoRA: the RAVEN loader
    hands over a patcher whose patch dict is normally empty, but the summary is
    a diff rather than a total so that stays an observation instead of an
    assumption.
    """
    after = patch_key_counts(patcher)
    patches = getattr(patcher, "patches", {}) or {}
    added_keys = sorted(
        key for key, count in after.items() if count > int(before.get(key, 0))
    )
    state_keys = set()
    inner = getattr(patcher, "model", None)
    state_dict = getattr(inner, "state_dict", None)
    if callable(state_dict):
        with contextlib.suppress(Exception):
            state_keys = set(state_dict().keys())

    detail: List[Dict[str, Any]] = []
    nonzero_keys: List[str] = []
    zero_keys: List[str] = []
    residual_targets: List[str] = []
    non_base_targets: List[str] = []
    for key in added_keys:
        entries = list(patches.get(key, []))[int(before.get(key, 0)):]
        described = [describe_patch(entry) for entry in entries]
        if any(item["nonzero"] for item in described):
            nonzero_keys.append(key)
        else:
            zero_keys.append(key)
        if any(prefix in key for prefix in residual_prefixes):
            residual_targets.append(key)
        if state_keys and key not in state_keys:
            non_base_targets.append(key)
        if len(detail) < STACKED_PATCH_DETAIL_LIMIT:
            detail.append({"key": key, "patches": described})

    base_key_hits = [
        key
        for key in nonzero_keys
        if key.startswith("diffusion_model.")
        and key not in residual_targets
        and (not state_keys or key in state_keys)
    ]
    return {
        "patch_keys_before": len(before),
        "patch_keys_after": len(after),
        "target_keys": added_keys,
        "target_key_count": len(added_keys),
        "patch_count": sum(
            max(0, count - int(before.get(key, 0))) for key, count in after.items()
        ),
        "nonzero_key_count": len(nonzero_keys),
        "zero_keys": zero_keys,
        "nonzero_base_key_hits": base_key_hits,
        "residual_targets": residual_targets,
        "targets_outside_the_state_dict": non_base_targets,
        "patch_detail": detail,
    }


def check_stacked_lora(record: Dict[str, Any], checks: Checks) -> None:
    """Every gate the stacked-LoRA lane owns, read off the recorded evidence.

    Split from the call itself on purpose: the gates are then a pure function of
    what was observed, and can be exercised against a record that describes a
    failure nobody can produce on demand (a LoRA for another model, a file that
    targets the RAVEN residual).
    """
    checks.note(
        "stacked lora: the file",
        "{} -> {} ({} bytes)".format(
            record.get("lora_name"),
            record.get("path"),
            record.get("bytes"),
        ),
    )
    checks.expect(
        "stacked lora: the official node is the one upstream advertises",
        record.get("node_function") == OFFICIAL_LORA_FUNCTION
        and list(record.get("node_return_types") or []) == ["MODEL"],
        "{}.{} FUNCTION={!r} RETURN_TYPES={}".format(
            record.get("node_module"),
            record.get("node_class"),
            record.get("node_function"),
            record.get("node_return_types"),
        ),
    )
    checks.expect(
        "stacked lora: the official node ran exactly once",
        int(record.get("node_calls", 0)) == 1,
        "{}() called {} time(s); comfy.sd.load_lora_for_models reached {} "
        "time(s) underneath it".format(
            record.get("node_function"),
            record.get("node_calls"),
            record.get("load_lora_for_models_calls"),
        ),
    )
    checks.expect(
        "stacked lora: the node returned a fresh MODEL, not the one it was given",
        bool(record.get("returned_a_clone")),
        "returned {} (same object as the input: {})".format(
            record.get("patcher_class"), not record.get("returned_a_clone")
        ),
    )
    checks.expect(
        "stacked lora: the patcher downstream is still a stock ModelPatcher",
        bool(record.get("patcher_is_stock")) and bool(record.get("patcher_class_unchanged")),
        "{} (stock={}, unchanged from {}={})".format(
            record.get("patcher_class"),
            record.get("patcher_is_stock"),
            record.get("patcher_class_before"),
            record.get("patcher_class_unchanged"),
        ),
    )
    checks.expect(
        "stacked lora: the official load produced patches",
        int(record.get("patch_count", 0)) > 0 and int(record.get("target_key_count", 0)) > 0,
        "{} patch(es) over {} target key(s)".format(
            record.get("patch_count"), record.get("target_key_count")
        ),
    )
    hits = list(record.get("nonzero_base_key_hits") or [])
    checks.expect(
        "stacked lora: at least one non-zero patch landed on a base key",
        bool(hits),
        "{} non-zero base key(s){}; {} target(s) patched to zero".format(
            len(hits),
            " e.g. {}".format(hits[0]) if hits else "",
            len(record.get("zero_keys") or []),
        ),
    )
    checks.expect(
        "stacked lora: every key of the file was matched",
        not record.get("unmatched_warnings"),
        "{} unmatched-key warning(s) from comfy: {}".format(
            len(record.get("unmatched_warnings") or []),
            (record.get("unmatched_warnings") or [])[:3],
        ),
    )
    checks.expect(
        "stacked lora: every target is a real key of the model",
        not record.get("targets_outside_the_state_dict"),
        "{} target(s) are not in the model's state dict: {}".format(
            len(record.get("targets_outside_the_state_dict") or []),
            (record.get("targets_outside_the_state_dict") or [])[:3],
        ),
    )
    checks.expect(
        "stacked lora: nothing targeted the RAVEN residual's own parameters",
        not record.get("residual_targets"),
        "{} residual target(s): {}; the mandatory adapter is not a LoRA "
        "surface".format(
            len(record.get("residual_targets") or []),
            (record.get("residual_targets") or [])[:3],
        ),
    )
    expected_modules = record.get("expected_attachment_modules")
    checks.expect(
        "stacked lora: the mandatory RAVEN attachment came through untouched",
        bool(record.get("attachment_same_object"))
        and int(record.get("attachment_modules", -1))
        == int(record.get("attachment_modules_before", -2))
        and (
            expected_modules is None
            or int(record.get("attachment_modules", -1)) == int(expected_modules)
        ),
        "same object={} modules {} -> {} (published: {})".format(
            record.get("attachment_same_object"),
            record.get("attachment_modules_before"),
            record.get("attachment_modules"),
            expected_modules,
        ),
    )
    checks.expect(
        "stacked lora: the DiT object downstream is the same one the loader built",
        bool(record.get("diffusion_model_same_object")),
        "diffusion_model identity preserved: {}".format(
            record.get("diffusion_model_same_object")
        ),
    )


def stack_official_lora(
    *,
    env: ComfyEnv,
    model: Any,
    lora_name: str,
    strength: float,
    checks: Checks,
) -> Tuple[Any, Dict[str, Any]]:
    """Chain upstream's ``LoraLoaderModelOnly`` onto the RAVEN ``MODEL``.

    The node is **instantiated** and its own ``FUNCTION`` is invoked with the
    ``folder_paths``-relative name a workflow would carry.
    ``comfy.sd.load_lora_for_models`` is counted, not called: reaching it
    directly would test Comfy, while the claim under test is that the *node* a
    user wires up still works after this plugin's loader.
    """
    node_cls = getattr(env.upstream_nodes, OFFICIAL_LORA_NODE, None)
    if node_cls is None:
        raise ProbeError(
            "this ComfyUI has no nodes.{}; --stacked-lora-name cannot be honoured and "
            "the probe will not substitute its own LoRA loading".format(OFFICIAL_LORA_NODE)
        )
    function_name = str(getattr(node_cls, "FUNCTION", OFFICIAL_LORA_FUNCTION))
    if not callable(getattr(node_cls, function_name, None)):
        raise ProbeError(
            "nodes.{} advertises FUNCTION={!r}, which it does not have".format(
                OFFICIAL_LORA_NODE, function_name
            )
        )

    resolver = getattr(env.folder_paths, "get_full_path_or_raise", None)
    if not callable(resolver):
        resolver = getattr(env.folder_paths, "get_full_path", None)
    if not callable(resolver):
        raise ProbeError("this folder_paths cannot resolve a '{}' name".format(LORA_FOLDER))
    try:
        resolved = resolver(LORA_FOLDER, lora_name)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's own refusal
        raise ProbeError(
            "folder_paths cannot resolve '{}/{}': {}: {}. --stacked-lora-name is a "
            "name relative to ComfyUI's loras folder, not a path; nothing is "
            "downloaded".format(LORA_FOLDER, lora_name, type(exc).__name__, exc)
        ) from exc
    if not resolved or not Path(str(resolved)).is_file():
        raise ProbeError(
            "folder_paths resolves '{}/{}' to {!r}, which is not a file".format(
                LORA_FOLDER, lora_name, resolved
            )
        )

    file_facts = file_info(Path(str(resolved)))
    inner = getattr(model, "model", None)
    attachment = getattr(inner, "raven_lora_attachment", None)
    diffusion_model = getattr(inner, "diffusion_model", None)
    before_counts = patch_key_counts(model)

    expected_modules: Optional[int] = None
    with contextlib.suppress(Exception):
        from raven_streaming import lora as raven_lora

        expected_modules = int(raven_lora.EXPECTED_MODULE_COUNT)

    record: Dict[str, Any] = {
        "lora_name": lora_name,
        "strength": float(strength),
        "folder": LORA_FOLDER,
        "node_module": getattr(node_cls, "__module__", "?"),
        "node_class": getattr(node_cls, "__qualname__", OFFICIAL_LORA_NODE),
        "node_function": function_name,
        "node_return_types": list(getattr(node_cls, "RETURN_TYPES", ()) or ()),
        "patcher_class_before": type(model).__name__,
        "attachment_modules_before": len(getattr(attachment, "entries", []) or []),
        "expected_attachment_modules": expected_modules,
    }
    record.update(file_facts)

    node = node_cls()
    record["node_instance"] = type(node).__name__
    sd_module = getattr(env.comfy, "sd", None)
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        node_counter = stack.enter_context(count_calls(node_cls, function_name))
        sd_counter = CallCounter("comfy.sd.load_lora_for_models")
        if sd_module is not None and hasattr(sd_module, "load_lora_for_models"):
            stack.enter_context(count_calls(sd_module, "load_lora_for_models", sd_counter))
        warnings = stack.enter_context(capture_lora_warnings())
        result = getattr(node, function_name)(model, lora_name, float(strength))
    record["seconds"] = time.perf_counter() - started
    record["node_calls"] = node_counter.calls
    record["node_call_args"] = [list(call) for call in node_counter.scalar_args]
    record["load_lora_for_models_calls"] = sd_counter.calls
    record["unmatched_warnings"] = unmatched_lora_messages(warnings)
    record["warnings"] = [str(w) for w in warnings]

    # ``LoraLoader`` caches the whole file on the node instance
    # (``self.loaded_lora``) so a re-run does not read it twice. There is no
    # re-run here, and the 66 GB DiT is already resident: the cache is dropped
    # rather than left to the collector's timing.
    record["node_cached_the_file"] = getattr(node, "loaded_lora", None) is not None
    with contextlib.suppress(Exception):
        node.loaded_lora = None
    del node
    gc.collect()

    if not isinstance(result, (tuple, list)) or len(result) != 1:
        raise ProbeError(
            "nodes.{}.{}() returned {!r}, not the documented one-tuple of "
            "MODEL".format(OFFICIAL_LORA_NODE, function_name, type(result).__name__)
        )
    stacked = result[0]
    if not hasattr(stacked, "patches") or not hasattr(stacked, "model"):
        raise ProbeError(
            "nodes.{}.{}() returned a {}, which is not a ModelPatcher".format(
                OFFICIAL_LORA_NODE, function_name, type(stacked).__name__
            )
        )

    stacked_inner = getattr(stacked, "model", None)
    stacked_attachment = getattr(stacked_inner, "raven_lora_attachment", None)
    stock_cls = getattr(getattr(env.comfy, "model_patcher", None), "ModelPatcher", None)
    record.update(
        {
            "returned_a_clone": stacked is not model,
            "patcher_class": type(stacked).__name__,
            "patcher_class_unchanged": type(stacked) is type(model),
            "patcher_is_stock": stock_cls is not None and type(stacked) is stock_cls,
            "stock_patcher_class": None
            if stock_cls is None
            else "{}.{}".format(stock_cls.__module__, stock_cls.__qualname__),
            "model_same_object": stacked_inner is inner,
            "attachment_same_object": stacked_attachment is not None
            and stacked_attachment is attachment,
            "attachment_modules": len(getattr(stacked_attachment, "entries", []) or []),
            "diffusion_model_same_object": getattr(stacked_inner, "diffusion_model", None)
            is diffusion_model,
            "model_size_bytes": None,
        }
    )
    with contextlib.suppress(Exception):
        record["model_size_bytes"] = int(stacked.model_size())
    record.update(
        summarise_stacked_patches(
            before_counts, stacked, residual_prefixes=raven_residual_prefixes()
        )
    )

    check_stacked_lora(record, checks)
    return stacked, record


# --------------------------------------------------------------------------
# the optional real text lane (--text-encoder / --prompt)
# --------------------------------------------------------------------------


def dit_hidden_size() -> int:
    """The published H3 hidden width, from this package's own model config.

    ``prefill_text`` decides whether to run ``condition_proj`` + the token
    refiner by comparing the context width against exactly these two numbers,
    so they are what a context has to be one of.
    """
    from raven_streaming import lora as lora_mod

    return int(lora_mod.RavenBaseConfig().hidden_size)


def choose_clip_type(clip_loader_cls: Any) -> Tuple[str, str]:
    """Read ``CLIPLoader``'s **actual** schema and pick H3's type and device.

    Not hard-coded: the combo is read off ``INPUT_TYPES()`` at run time and the
    MiniMax entry is looked up in it. If a ComfyUI is used whose ``CLIPLoader``
    does not offer one, that is a compatibility fact the probe must state --
    guessing ``"minimax"`` would make ``comfy.sd.CLIPType`` fall back to
    ``STABLE_DIFFUSION`` inside the node (``getattr(..., type.upper(),
    CLIPType.STABLE_DIFFUSION)``) and the encoder would load as the wrong
    family, silently.

    ``device`` is taken the same way: ``"default"`` is what lets
    ``comfy.model_management.text_encoder_device()`` put the encode on the GPU,
    which is the behaviour ``docs/requirements.md`` claims and this probe is
    here to check. ``"cpu"`` is never chosen.
    """
    schema = clip_loader_cls.INPUT_TYPES()
    required = dict(schema.get("required", {}))
    optional = dict(schema.get("optional", {}))
    if "clip_name" not in required or "type" not in required:
        raise ProbeError(
            "this CLIPLoader's schema has no 'clip_name'/'type' input ({}); the probe "
            "refuses to guess how to drive it".format(sorted(required))
        )
    types = list(required["type"][0])
    match = [name for name in types if str(name).lower() in MINIMAX_CLIP_TYPE_NAMES]
    if not match:
        raise ProbeError(
            "this ComfyUI's CLIPLoader offers no MiniMax H3 text-encoder type "
            "(looked for {} in {}); it cannot load the H3 Qwen3-VL encoder".format(
                sorted(MINIMAX_CLIP_TYPE_NAMES), types
            )
        )
    device = "default"
    if "device" in optional:
        devices = list(optional["device"][0])
        if device not in devices:
            raise ProbeError(
                "CLIPLoader's device combo is {}, which has no 'default' entry; the "
                "probe will not pick a device policy upstream does not offer".format(devices)
            )
    return match[0], device


@contextlib.contextmanager
def watch_model_loads(model_management: Any) -> Iterator[List[Dict[str, Any]]]:
    """Record every ``load_models_gpu`` call for the duration of a block.

    ``comfy.sd.CLIP.load_model`` calls
    ``model_management.load_models_gpu([self.patcher], memory_required=...)``
    through the module object, so a temporary attribute swap sees the real
    load, with the real patcher, without changing what happens. It is the only
    way to observe *which device the encode ran on* from outside the official
    node -- the patcher's state after the fact can already have been changed by
    the next load.
    """
    original = model_management.load_models_gpu
    records: List[Dict[str, Any]] = []

    def wrapper(models: Any, *args: Any, **kwargs: Any) -> Any:
        given = list(models)
        entry: Dict[str, Any] = {
            "models": [
                {
                    "patcher": type(patcher).__name__,
                    "model": type(getattr(patcher, "model", None)).__name__,
                    "load_device": str(getattr(patcher, "load_device", None)),
                    "offload_device": str(getattr(patcher, "offload_device", None)),
                    "is_clip": bool(getattr(patcher, "is_clip", False)),
                    "id": id(patcher),
                }
                for patcher in given
            ],
            "memory_required": int(kwargs.get("memory_required", 0) or 0),
        }
        records.append(entry)
        result = original(models, *args, **kwargs)
        for patcher, described in zip(given, entry["models"]):
            current = getattr(patcher, "current_loaded_device", None)
            described["current_loaded_device"] = (
                str(current()) if callable(current) else None
            )
            loaded_size = getattr(patcher, "loaded_size", None)
            described["loaded_size"] = int(loaded_size()) if callable(loaded_size) else None
        return result

    model_management.load_models_gpu = wrapper
    try:
        yield records
    finally:
        model_management.load_models_gpu = original


def parameter_device_histogram(module: Any) -> Dict[str, Dict[str, int]]:
    """Bytes of parameters and buffers per device.

    Buffers are counted too: an NVFP4/AWQ checkpoint carries its packed weights
    and scales as buffers on some paths, and a histogram that ignored them
    would report an encoder as "off the GPU" while most of it still sat there.
    """
    out: Dict[str, Dict[str, int]] = {}

    def add(kind: str, tensor: Any) -> None:
        key = str(getattr(tensor, "device", "?"))
        bucket = out.setdefault(
            key, {"parameter_bytes": 0, "buffer_bytes": 0, "parameters": 0, "buffers": 0}
        )
        bucket["{}_bytes".format(kind)] += int(tensor.numel()) * int(tensor.element_size())
        bucket["{}s".format(kind)] += 1

    for _name, parameter in module.named_parameters():
        add("parameter", parameter)
    for _name, buffer in module.named_buffers():
        add("buffer", buffer)
    return {key: out[key] for key in sorted(out)}


def text_encoder_residency(clip: Any) -> Dict[str, Any]:
    """Where the text encoder's weights are, as Comfy itself sees it."""
    patcher = getattr(clip, "patcher", None)
    residency: Dict[str, Any] = {
        "patcher": type(patcher).__name__ if patcher is not None else None,
        "load_device": str(getattr(patcher, "load_device", None)),
        "offload_device": str(getattr(patcher, "offload_device", None)),
    }
    for name in ("loaded_size", "model_size", "current_loaded_device"):
        method = getattr(patcher, name, None)
        if callable(method):
            with contextlib.suppress(Exception):
                value = method()
                residency[name] = int(value) if name.endswith("size") else str(value)
    model = getattr(patcher, "model", None)
    if model is not None and hasattr(model, "named_parameters"):
        with contextlib.suppress(Exception):
            residency["devices"] = parameter_device_histogram(model)
    return residency


def offload_text_encoder(model_management: Any, clip: Any) -> Dict[str, Any]:
    """Get the encoder off the GPU through Comfy's own public API.

    ``unload_model_and_clones(patcher)`` is upstream's *targeted* unload -- it
    frees exactly this model (and its multigpu clones) and leaves everything
    else loaded, which is what the workflow wants: the DiT is about to be
    loaded and must not have to fight a 30 GB encoder for the card.

    Nothing here calls ``.to()`` on a module. Moving weights by hand would
    desynchronise Comfy's accounting from reality -- it would still believe the
    encoder is resident and would reserve room for it on the next load -- and
    the whole point of the check that follows is to see *Comfy's* view.

    When the API is absent (an older ComfyUI), that is reported, not worked
    around: the encoder then stays where it is and the DiT's own
    ``load_models_gpu`` evicts it, which is also a supported outcome.
    """
    patcher = getattr(clip, "patcher", None)
    unload = getattr(model_management, "unload_model_and_clones", None)
    if patcher is None:
        return {"method": None, "reason": "the CLIP has no patcher to unload"}
    if not callable(unload):
        return {
            "method": None,
            "reason": (
                "this ComfyUI has no model_management.unload_model_and_clones; the "
                "encoder is left to be evicted naturally by the DiT's own load"
            ),
        }
    started = time.perf_counter()
    try:
        unload(patcher)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {
            "method": "unload_model_and_clones",
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
    return {
        "method": "unload_model_and_clones",
        "seconds": time.perf_counter() - started,
    }


def encode_t2va(
    node_cls: Any,
    *,
    clip: Any,
    vae: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
) -> Tuple[Any, Any]:
    """Call the official ``MiniMaxH3ImageToVideo`` in its T2VA form.

    ``first_frame`` / ``last_frame`` are **not passed at all** -- that is what
    makes this T2VA rather than fl2va, and it is why the conditioning that
    comes back must carry no ``minimax_keyframes``.

    The node owns the tokenizer, the vision-block handling, the scheduled
    encode and the empty AV latent. None of it is reimplemented here: a probe
    that hand-rolled ``clip.tokenize`` / ``encode_from_tokens`` would be
    checking its own prompt handling, not ComfyUI's.
    """
    output = node_cls.execute(
        clip=clip,
        vae=vae,
        prompt=str(prompt),
        width=int(width),
        height=int(height),
        length=int(length),
    )
    result = getattr(output, "result", None)
    if result is None:
        result = tuple(output) if isinstance(output, (list, tuple)) else None
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ProbeError(
            "MiniMaxH3ImageToVideo returned {!r}; expected (CONDITIONING, LATENT)".format(
                type(output).__name__
            )
        )
    return result[0], result[1]


def clone_latent(latent: Any, nested_cls: Any) -> Dict[str, Any]:
    """A fresh copy of the official AV latent, same class, same values.

    Each run gets its own tensors so no run can be affected by what a previous
    one did with them, while the *contents* remain the ones the official node
    produced rather than something this probe made up.
    """
    streams = tuple(t.clone() for t in latent["samples"].unbind())
    return {"samples": nested_cls(streams)}


def check_text_lane_outputs(
    conditioning: Any, latent: Any, *, geometry: Dict[str, int]
) -> Tuple[Dict[str, Any], List[Check]]:
    """What the official T2VA node handed back, checked against the contract.

    Pure: given the node's two outputs it decides whether the streaming sampler
    would accept them, and describes them well enough that the report identifies
    the encode (tag histogram, hashes, extras, geometry).
    """
    from raven_streaming import contracts
    from raven_streaming import layout as layout_mod

    checks: List[Check] = []
    summary: Dict[str, Any] = {}

    def expect(name: str, condition: Any, detail: str = "", gate: bool = True) -> bool:
        ok = bool(condition)
        checks.append(Check(name=name, ok=ok, detail=detail, gate=gate))
        return ok

    entries = list(conditioning) if isinstance(conditioning, (list, tuple)) else []
    summary["entries"] = len(entries)
    extras = dict(entries[0][1]) if entries and isinstance(entries[0][1], dict) else {}
    summary["extras"] = sorted(extras)
    expect(
        "text: the encode produced exactly one conditioning entry",
        len(entries) == 1,
        "{} entr(ies)".format(len(entries)),
    )
    for key in ("minimax_keyframes", "minimax_refs"):
        expect(
            "text: no {} (this is T2VA, no frames were passed)".format(key),
            key not in extras,
            "extras: {}".format(sorted(extras)),
        )

    try:
        parsed = contracts.parse_conditioning(conditioning)
    except Exception as exc:  # noqa: BLE001
        expect(
            "text: the streaming sampler accepts this CONDITIONING",
            False,
            "{}: {}".format(type(exc).__name__, exc),
        )
        return summary, checks
    expect("text: the streaming sampler accepts this CONDITIONING", True,
           "{} token(s)".format(parsed.text_len))

    context = parsed.cross_attn
    summary["context"] = {
        "shape": list(context.shape),
        "dtype": str(context.dtype),
        "device": str(context.device),
        "finite": bool(torch.isfinite(context.float()).all()),
    }
    summary["text_len"] = int(parsed.text_len)
    expect("text: the context is finite", summary["context"]["finite"])
    expect(
        "text: the context is [1, L, dim] with L >= 1",
        len(context.shape) == 3 and context.shape[0] == 1 and context.shape[1] >= 1,
        "{}".format(list(context.shape)),
    )
    hidden = dit_hidden_size()
    expect(
        "text: the context width is a width the DiT knows",
        int(context.shape[2]) in (TEXT_EMBED_DIM, hidden),
        "dim={} (raw Qwen3-VL states are {}, already-refined states are {})".format(
            int(context.shape[2]), TEXT_EMBED_DIM, hidden
        ),
    )

    tags = parsed.token_tags
    if tags is None:
        summary["token_tags"] = None
        expect(
            "text: the encode carried minimax_token_tags",
            False,
            "no tags: every row would fall back to the text tag, which is not what a "
            "real H3 prompt produces",
            gate=False,
        )
    else:
        values = [int(v) for v in tags.reshape(-1).tolist()]
        histogram = {
            "video({})".format(layout_mod.VIDEO_TAG): values.count(layout_mod.VIDEO_TAG),
            "text({})".format(layout_mod.TEXT_TAG): values.count(layout_mod.TEXT_TAG),
            "audio({})".format(layout_mod.AUDIO_TAG): values.count(layout_mod.AUDIO_TAG),
        }
        summary["token_tags"] = {
            "shape": list(tags.shape),
            "dtype": str(tags.dtype),
            "device": str(tags.device),
            "histogram": histogram,
        }
        expect(
            "text: the tag vector covers every token",
            len(values) == int(parsed.text_len),
            "{} tag(s) for {} token(s)".format(len(values), parsed.text_len),
        )

    pooled = parsed.pooled_output
    summary["pooled_output"] = (
        None if pooled is None else {"shape": list(pooled.shape), "dtype": str(pooled.dtype)}
    )

    # -- the latent the same node built -------------------------------
    try:
        request = contracts.parse_latent(latent, warn_experimental=False)
    except Exception as exc:  # noqa: BLE001
        expect(
            "text: the streaming sampler accepts this LATENT",
            False,
            "{}: {}".format(type(exc).__name__, exc),
        )
        return summary, checks
    expect("text: the streaming sampler accepts this LATENT", True)
    summary["latent"] = {
        "frames": int(request.frames),
        "width": int(request.width),
        "height": int(request.height),
        "latent_t": int(request.latent_t),
        "latent_h": int(request.latent_h),
        "latent_w": int(request.latent_w),
        "audio_t": int(request.audio_t),
        "dtype": str(request.dtype),
        "device": str(request.device),
        "nested_class": request.nested_cls.__name__,
    }
    measured = (
        int(request.frames),
        int(request.width),
        int(request.height),
        int(request.latent_t),
        int(request.audio_t),
    )
    expected = (
        int(geometry["frames"]),
        int(geometry["width"]),
        int(geometry["height"]),
        int(geometry["latent_t"]),
        int(geometry["audio_t"]),
    )
    expect(
        "text: the official latent sits on the requested grid",
        measured == expected,
        "node built {} (frames, w, h, latent_t, audio_t), probe asked for {}".format(
            measured, expected
        ),
    )
    empty = not bool(request.video.any()) and not bool(request.audio.any())
    summary["latent"]["empty"] = empty
    expect(
        "text: the official latent is empty (T2VA starts from fresh noise)",
        empty,
        "a non-empty latent would be refused by the sampler",
    )
    return summary, checks


def run_text_lane(
    *,
    clip_loader_cls: Any,
    image_to_video_cls: Any,
    folder_paths: Any,
    model_management: Any,
    text_encoder_path: Path,
    prompt: str,
    geometry: Dict[str, int],
    video_vae: Any,
    device: torch.device,
    checks: Checks,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the real text encoder, encode the real prompt, then get it off the GPU.

    Returns ``(conditioning, latent, summary)``. Every collaborator is injected
    so the whole lane can be driven against fakes; the real call sites pass
    upstream's own classes.

    Order is the workflow's order and it is not an accident: the encoder is
    loaded, used and offloaded **before** the 66 GB DiT is built, which is the
    memory story ``docs/requirements.md`` describes ("GPU encode, then offloaded
    to CPU").
    """
    summary: Dict[str, Any] = {
        "prompt": str(prompt),
        "prompt_characters": len(str(prompt)),
        "path": str(text_encoder_path),
    }
    (name,) = register_model_files(folder_paths, TEXT_ENCODER_FOLDER, [text_encoder_path])
    summary["name"] = name

    clip_type, clip_device = choose_clip_type(clip_loader_cls)
    summary["clip_type"] = clip_type
    summary["clip_device_option"] = clip_device
    checks.note(
        "text: CLIPLoader's own schema chose the type",
        "type={!r} device={!r} (read from INPUT_TYPES, not hard-coded)".format(
            clip_type, clip_device
        ),
    )

    started = time.perf_counter()
    (clip,) = clip_loader_cls().load_clip(name, clip_type, clip_device)
    summary["load_seconds"] = time.perf_counter() - started
    summary["clip_class"] = type(clip).__name__
    summary["residency_after_load"] = text_encoder_residency(clip)

    with watch_model_loads(model_management) as loads:
        started = time.perf_counter()
        conditioning, latent = encode_t2va(
            image_to_video_cls,
            clip=clip,
            vae=video_vae,
            prompt=prompt,
            width=int(geometry["width"]),
            height=int(geometry["height"]),
            length=int(geometry["frames"]),
        )
        summary["encode_seconds"] = time.perf_counter() - started
    summary["loads_during_encode"] = loads

    patcher = getattr(clip, "patcher", None)
    encoder_loads = [
        entry
        for record in loads
        for entry in record["models"]
        if patcher is not None and entry["id"] == id(patcher)
    ]
    summary["encoder_loads"] = encoder_loads
    if encoder_loads:
        devices = {entry.get("current_loaded_device") or entry["load_device"] for entry in encoder_loads}
        on_compute = any(
            str(entry.get("current_loaded_device") or entry["load_device"]).startswith(device.type)
            for entry in encoder_loads
        )
        checks.expect(
            "text: the encode ran with the encoder loaded onto {}".format(device.type),
            on_compute,
            "load_models_gpu put it on {} ({} call(s) for this patcher)".format(
                sorted(devices), len(encoder_loads)
            ),
        )
    else:
        checks.note(
            "text: the encoder's load device was not observable",
            "no load_models_gpu call named this patcher; the encode may have run from "
            "weights that were already resident",
        )

    summary["residency_after_encode"] = text_encoder_residency(clip)
    summary["offload"] = offload_text_encoder(model_management, clip)
    summary["residency_after_offload"] = text_encoder_residency(clip)

    after = summary["residency_after_offload"]
    loaded_size = after.get("loaded_size")
    if summary["offload"].get("method") is None:
        checks.note(
            "text: the encoder was not offloaded explicitly",
            str(summary["offload"].get("reason")),
        )
    elif loaded_size is None:
        checks.note(
            "text: the encoder's residency is not reportable",
            "this patcher has no loaded_size()",
        )
    else:
        checks.expect(
            "text: the encoder holds no {} weights during sampling".format(device.type),
            int(loaded_size) == 0,
            "loaded_size={} devices={}".format(loaded_size, after.get("devices")),
        )

    lane_summary, lane_checks = check_text_lane_outputs(
        conditioning, latent, geometry=geometry
    )
    summary.update(lane_summary)
    checks.extend(lane_checks)
    return conditioning, latent, summary


# --------------------------------------------------------------------------
# in-process instrumentation of the node's own call graph
# --------------------------------------------------------------------------


@dataclass
class Instrumentation:
    """What the transparent wrappers observed during one ``sample`` call.

    Every wrapper here is installed on this process's module objects and
    removed again afterwards. None of them changes what the node computes:
    each one calls the original, records a duration or a value, and returns the
    original's result untouched. The runtime package is not modified -- the
    alternative would be to add probe hooks to shipping code, which is how a
    measurement ends up in a release.
    """

    timings: Dict[str, float] = field(default_factory=dict)
    memory_budget: Dict[str, Any] = field(default_factory=dict)
    #: the live ``RolloutMemoryBudget``; re-read after the run because the node
    #: splices the final-decode handover into ``detail`` on its way out
    budget_object: Any = None
    pipeline: Any = None
    pipeline_report: Dict[str, Any] = field(default_factory=dict)
    preview_disabled: bool = False
    preview_disabled_reason: str = ""
    load_calls: int = 0
    discard_pending_calls: int = 0
    #: The product path must never run a whole-clip decode of either stream.
    #: These count the times it did (see :func:`instrument`): the two kept
    #: diagnostic helpers, and the two ``comfy.sd.VAE.decode`` entry points
    #: themselves -- which is where a decode would land even if it did not go
    #: through a helper.
    decode_images_calls: int = 0
    decode_audio_calls: int = 0
    vae_decode_calls: Dict[str, int] = field(default_factory=dict)
    #: both collectors hand over exactly once per run
    finalize_image_calls: int = 0
    finalize_audio_calls: int = 0
    #: ``--cancel-after-chunk``: deliver this many chunks *through the
    #: collectors* and then raise the real ``SamplingCancelled`` from inside
    #: ``pipeline.on_chunk``. 0 disables it, and then the wrapper only counts.
    cancel_after_chunk: int = 0
    #: chunks that reached ``StreamingPipeline.on_chunk`` (i.e. were decoded)
    chunks_delivered: int = 0
    chunk_cancel_raised: bool = False
    #: the pipeline's own state at the instant the chunk-cancel was raised,
    #: read *before* the exception leaves ``on_chunk``
    state_before_cancel: Dict[str, Any] = field(default_factory=dict)
    #: ``decoder.abort()`` calls, per lane, and whether that method exists at
    #: all on this runtime. Feature-probed: the abort API is the runtime's, and
    #: a decoder without it is a note rather than a failure.
    decoder_aborts: Dict[str, int] = field(default_factory=dict)
    decoder_abort_available: Dict[str, bool] = field(default_factory=dict)

    def add(self, key: str, seconds: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + float(seconds)


@contextlib.contextmanager
def instrument(inst: Instrumentation) -> Iterator[Instrumentation]:
    """Wrap the node's collaborators for one execution, then put them back."""
    from raven_streaming import cache as cache_mod
    from raven_streaming import nodes as nodes_mod
    from raven_streaming import streaming_pipeline as pipeline_mod

    original_budget = nodes_mod.rollout_memory_budget
    original_make_load = nodes_mod.make_load_models
    original_decode_images = nodes_mod.decode_images
    original_decode_audio = nodes_mod.decode_audio
    original_build_pipeline = pipeline_mod.build_media_pipeline
    original_discard = cache_mod.ChunkKVCache.__dict__["discard_pending"]

    def budget_wrapper(**kwargs: Any) -> Any:
        started = time.perf_counter()
        budget = original_budget(**kwargs)
        inst.add("memory_budget_seconds", time.perf_counter() - started)
        inst.budget_object = budget
        with contextlib.suppress(Exception):
            inst.memory_budget = budget.to_dict()
        return budget

    def make_load_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _TimedLoader(original_make_load(*args, **kwargs), inst)

    def pipeline_wrapper(**kwargs: Any) -> Any:
        pipeline = original_build_pipeline(**kwargs)
        _capture_pipeline(inst, pipeline)
        return pipeline

    def decode_images_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Counted, not timed: the product path does not call this any more (the
        # IMAGE comes out of the collector), so any call during a sample is a
        # regression -- a second whole-clip decode at the peak that OOMed the
        # measured 39-frame run. The probe's own --compare-official-video
        # diagnostic calls it deliberately, and does so *outside* this window.
        inst.decode_images_calls += 1
        return original_decode_images(*args, **kwargs)

    def decode_audio_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Counted, not timed, for the same reason as ``decode_images``: the
        # product path no longer calls it (the AUDIO comes out of the collector
        # through ``finalize_audio``), and the whole-clip ``vae_decode_audio``
        # is the call that OOMed a 192-frame run on a 24 GiB card with the DiT
        # and the video VAE already evicted.
        inst.decode_audio_calls += 1
        return original_decode_audio(*args, **kwargs)

    def discard_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        inst.discard_pending_calls += 1
        return original_discard(self, *args, **kwargs)

    nodes_mod.rollout_memory_budget = budget_wrapper
    nodes_mod.make_load_models = make_load_wrapper
    nodes_mod.decode_images = decode_images_wrapper
    nodes_mod.decode_audio = decode_audio_wrapper
    pipeline_mod.build_media_pipeline = pipeline_wrapper
    cache_mod.ChunkKVCache.discard_pending = discard_wrapper
    try:
        yield inst
    finally:
        nodes_mod.rollout_memory_budget = original_budget
        nodes_mod.make_load_models = original_make_load
        nodes_mod.decode_images = original_decode_images
        nodes_mod.decode_audio = original_decode_audio
        pipeline_mod.build_media_pipeline = original_build_pipeline
        cache_mod.ChunkKVCache.discard_pending = original_discard


#: Public, integer-valued introspection a streaming decoder may expose. Every
#: one is optional: this is a feature probe, not a contract, so a runtime that
#: adds or drops one changes what the report contains and nothing else.
DECODER_STATE_FIELDS = (
    "pending_latents",
    "latents_seen",
    "frames_emitted",
    "chunks_done",
    "samples_emitted",
    "lookahead_latents",
    "finished",
)

#: Where a decoder parks the tensors a cancel has to release: the video
#: coordinator's pending latents and its 5-frame ``dec_overlap`` (on the decode
#: device), and the audio decoder's overlap-save history. Private on purpose --
#: they are the *evidence* that a cancel dropped something, and there is no
#: public name for "are you still holding it". Probed defensively.
DECODER_BUFFER_ATTRS = ("_pending", "_dec_overlap", "_buffer")


def decoder_lanes(pipeline: Any) -> Dict[str, Any]:
    """The two streaming decoders the pipeline drives, or ``None`` per lane."""
    return {
        "video": getattr(pipeline, "_video", None),
        "audio": getattr(pipeline, "_audio", None),
    }


def decoder_state(decoder: Any) -> Dict[str, Any]:
    """What one decoder is willing to say about the state it is holding.

    Everything here is optional and read defensively, because the abort/cancel
    surface is the runtime's to define. What the probe needs from it is only
    the shape of an answer to two questions -- "was anything buffered when the
    cancel landed" and "was it gone afterwards" -- and it asks them of whatever
    the decoder happens to expose.
    """
    if decoder is None:
        # Same shape as a present decoder, so "is anything still held" can be
        # asked of a released lane without a special case: a lane that is gone
        # is holding nothing, which is exactly the answer a cancel wants.
        return {"present": False, "buffers_held": [], "holds_buffers": False}
    state: Dict[str, Any] = {
        "present": True,
        "class": type(decoder).__name__,
        "has_abort": callable(getattr(decoder, "abort", None)),
    }
    for name in DECODER_STATE_FIELDS:
        if not hasattr(decoder, name):
            continue
        with contextlib.suppress(Exception):
            value = getattr(decoder, name)
            if isinstance(value, bool):
                state[name] = bool(value)
            elif isinstance(value, int):
                state[name] = int(value)
    held: List[str] = []
    for name in DECODER_BUFFER_ATTRS:
        if not hasattr(decoder, name):
            continue
        with contextlib.suppress(Exception):
            if getattr(decoder, name) is not None:
                held.append(name)
    state["buffers_held"] = held
    state["holds_buffers"] = bool(held)

    # How many latents this lane has already handed to the VAE, as opposed to
    # buffered: the video coordinator keeps what it cannot decide yet in
    # ``pending_latents``, so the difference is the part that has *been*
    # decoded. That is what makes "a decode was due" a derived expectation
    # rather than a guess about lookahead.
    if "latents_seen" in state:
        state["consumed_latents"] = max(
            0, int(state["latents_seen"]) - int(state.get("pending_latents", 0))
        )
    # The wrapped ``comfy.sd.VAE`` the coordinator drives counts its own calls;
    # that count *is* "went through the VAE", with nothing inferred.
    inner = getattr(decoder, "decoder", None)
    calls = getattr(inner, "decode_calls", None)
    if isinstance(calls, int):
        state["vae_decode_calls"] = int(calls)
    return state


def pipeline_state(pipeline: Any) -> Dict[str, Any]:
    """A snapshot of everything the pipeline is still holding, right now.

    Taken twice around a chunk cancellation -- once inside the callback that
    raises it, once after the node has returned -- because "the buffers were
    released" is only a claim if something was there to release.
    """
    state: Dict[str, Any] = {}
    for name in (
        "chunks",
        "frames_decoded",
        "samples_decoded",
        "collected_frames",
        "collected_samples",
        "pending_frames",
        "samples_available",
        "frames_muxed",
        "samples_muxed",
        "finished",
    ):
        if not hasattr(pipeline, name):
            continue
        with contextlib.suppress(Exception):
            value = getattr(pipeline, name)
            state[name] = bool(value) if isinstance(value, bool) else int(value)
    report = getattr(pipeline, "report", None)
    if callable(report):
        with contextlib.suppress(Exception):
            snapshot = report().to_dict()
            state["image_bytes"] = int(snapshot.get("image_bytes", 0))
            state["audio_bytes"] = int(snapshot.get("audio_bytes", 0))
            state["image_shape"] = list(snapshot.get("image_shape", ()) or ())
            state["audio_shape"] = list(snapshot.get("audio_shape", ()) or ())
    state["decoders"] = {
        lane: decoder_state(decoder) for lane, decoder in decoder_lanes(pipeline).items()
    }
    state["holds_decoder_buffers"] = any(
        entry.get("holds_buffers") for entry in state["decoders"].values()
    )
    return state


def _capture_pipeline(inst: Instrumentation, pipeline: Any) -> None:
    """Keep the pipeline and time its two flushes, without touching the class.

    ``finish()`` and ``finalize_image()`` are replaced on the **instance**, so
    nothing else in the process is affected and the object still behaves
    exactly as the node built it. The report is read after the fact:
    ``StreamingPipeline.report()`` is a pure snapshot and ``finish()`` is
    idempotent.

    The timings are separate because they answer different questions:
    ``finish`` is the tail flush plus the preview's last fragments, while
    ``finalize_image`` / ``finalize_audio`` are the two collectors handing over
    the buffers they have been filling all along -- the calls that *replaced*
    the whole-clip ``video_vae.decode`` and ``vae_decode_audio``, and therefore
    the numbers that have to be small. ``finalize_audio`` is not free (it is
    where the official whole-clip normalisation happens) but it is one pass
    over a few MB, not a decode.
    """
    if inst.pipeline is pipeline:
        return
    inst.pipeline = pipeline

    # -- chunk delivery, and the optional cancellation on top of it -------
    #
    # ``on_chunk`` is where a chunk is actually decoded: the incremental video
    # collector and the overlap-save audio collector run *inside* it, through
    # the real VAEs. Counting here is therefore counting chunks that reached
    # the decoders, and raising here -- after the original returned -- cancels
    # a run that has genuinely produced pixels and samples, which is the only
    # position from which "the buffers were dropped" means anything.
    #
    # It also lands before ``PhaseSwapCoordinator`` re-enters the DiT phase for
    # the next chunk (its own ``on_chunk`` is load -> deliver -> reload), so a
    # chunk cancel makes one DiT load fewer than a forward cancel at the same
    # chunk. :func:`dit_loads_after_chunk_cancel` is that arithmetic.
    original_on_chunk = getattr(pipeline, "on_chunk", None)
    if callable(original_on_chunk):
        def counted_on_chunk(*args: Any, **kwargs: Any) -> Any:
            from raven_streaming.consistency import SamplingCancelled

            result = original_on_chunk(*args, **kwargs)
            inst.chunks_delivered += 1
            limit = int(inst.cancel_after_chunk)
            if limit > 0 and inst.chunks_delivered >= limit and not inst.chunk_cancel_raised:
                inst.chunk_cancel_raised = True
                # Read *before* raising: once the exception is out, the node's
                # own cancel path has already emptied everything this is here
                # to prove was there.
                inst.state_before_cancel = pipeline_state(pipeline)
                raise SamplingCancelled(
                    "probe: cancelled after {} delivered chunk(s)".format(
                        inst.chunks_delivered
                    )
                )
            return result

        pipeline.on_chunk = counted_on_chunk

    # -- the decoders' own abort surface, feature-probed ------------------
    for lane, decoder in decoder_lanes(pipeline).items():
        abort = getattr(decoder, "abort", None)
        inst.decoder_abort_available[lane] = callable(abort)
        inst.decoder_aborts.setdefault(lane, 0)
        if not callable(abort):
            continue

        def make_counted_abort(label: str, call: Any) -> Any:
            def counted_abort(*args: Any, **kwargs: Any) -> Any:
                inst.decoder_aborts[label] = inst.decoder_aborts.get(label, 0) + 1
                return call(*args, **kwargs)

            return counted_abort

        with contextlib.suppress(Exception):
            decoder.abort = make_counted_abort(lane, abort)

    original_finish = pipeline.finish
    original_finalize = getattr(pipeline, "finalize_image", None)

    def timed_finish(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_finish(*args, **kwargs)
        finally:
            inst.add("preview_flush_seconds", time.perf_counter() - started)

    pipeline.finish = timed_finish

    if callable(original_finalize):
        def timed_finalize(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_finalize(*args, **kwargs)
            finally:
                inst.add("image_finalize_seconds", time.perf_counter() - started)
                inst.finalize_image_calls += 1

        pipeline.finalize_image = timed_finalize

    original_finalize_audio = getattr(pipeline, "finalize_audio", None)
    if callable(original_finalize_audio):
        def timed_finalize_audio(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_finalize_audio(*args, **kwargs)
            finally:
                inst.add("audio_finalize_seconds", time.perf_counter() - started)
                inst.finalize_audio_calls += 1

        pipeline.finalize_audio = timed_finalize_audio


class _TimedLoader:
    """Times the DiT-phase load closure and forwards everything else to it.

    A *delegating* wrapper rather than a copy: ``make_load_models`` rebinds
    ``residency`` and ``hard_cap`` on the closure **on every call**, and the
    node reads them after the run to write its memory record. A snapshot taken
    at wrap time would quietly hand back the values from before the rollout.

    The node reads those attributes with ``getattr(..., None)`` precisely
    because a harness may wrap this closure; delegating means the record stays
    complete anyway.
    """

    __slots__ = ("_closure", "_inst")

    def __init__(self, closure: Any, inst: "Instrumentation") -> None:
        self._closure = closure
        self._inst = inst

    def __call__(
        self, models: Any, memory_required: int = 0, force_full_load: bool = False
    ) -> Any:
        started = time.perf_counter()
        try:
            return self._closure(models, memory_required, force_full_load)
        finally:
            self._inst.add("model_load_seconds", time.perf_counter() - started)
            self._inst.load_calls += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._closure, name)


@dataclass
class ForwardCounter:
    """How many real ``forward_chunk`` calls ran, and whether we stopped one."""

    limit: int = 0
    calls: int = 0
    raised: bool = False


@contextlib.contextmanager
def cancel_after_forwards(
    target: Any, limit: int, counter: Optional[ForwardCounter] = None
) -> Iterator[ForwardCounter]:
    """Let ``limit`` real forwards through, then raise the real cancellation.

    The wrapper sits on the causal DiT **instance**, so the forward that runs
    is the genuine one -- the probe is cancelling a real rollout mid-flight,
    not simulating one. ``SamplingCancelled`` is imported from
    :mod:`raven_streaming.consistency` rather than re-declared, so the exception
    the node sees is identical to the one its own ``cancel_check`` raises (and
    the preview lane classifies by name as ``cancelled``, not ``error``).
    """
    from raven_streaming.consistency import SamplingCancelled

    counter = counter if counter is not None else ForwardCounter(limit=int(limit))
    counter.limit = int(limit)
    original = getattr(target, "forward_chunk", None)
    if not callable(original):
        raise ProbeError(
            "{} has no forward_chunk(); this is not the chunk-causal DiT the "
            "streaming sampler drives, so neither the forward count nor the "
            "cancellation point exists".format(type(target).__name__)
        )

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        counter.calls += 1
        if counter.limit > 0 and counter.calls >= counter.limit:
            counter.raised = True
            raise SamplingCancelled(
                "probe: cancelled after {} forward_chunk call(s)".format(counter.calls)
            )
        return result

    target.forward_chunk = wrapper
    try:
        yield counter
    finally:
        try:
            del target.forward_chunk
        except AttributeError:  # pragma: no cover - defensive
            target.forward_chunk = original


@contextlib.contextmanager
def count_forwards(target: Any, counter: Optional[ForwardCounter] = None) -> Iterator[ForwardCounter]:
    """``cancel_after_forwards`` with no limit: count, never interrupt."""
    with cancel_after_forwards(target, 0, counter) as counted:
        yield counted


@contextlib.contextmanager
def watch_vae_decodes(counts: Dict[str, int], **sockets: Any) -> Iterator[Dict[str, int]]:
    """Count ``comfy.sd.VAE.decode`` calls on the sockets, for the run's duration.

    ``VAE.decode`` is the *whole-clip* entry point, and it is the one the node
    no longer uses: the video collector drives the inner module's
    ``_adaptive_decode`` and the audio collector drives the inner module's
    ``decode``, neither of which passes through here. So a non-zero count is
    unambiguous -- something decoded a whole stream in one go, which is the
    allocation both lanes exist to avoid.

    Wrapped on the **instance**, so nothing else in the process is affected,
    and removed again afterwards. The probe's own ``--compare-official-video``
    diagnostic deliberately runs outside this window.
    """
    originals: Dict[str, Any] = {}
    for name, socket in sockets.items():
        counts.setdefault(name, 0)
        original = getattr(socket, "decode", None)
        if not callable(original):
            continue
        originals[name] = (socket, original)

        def make(label: str, call: Any) -> Any:
            def counted(*args: Any, **kwargs: Any) -> Any:
                counts[label] = counts.get(label, 0) + 1
                return call(*args, **kwargs)

            return counted

        socket.decode = make(name, original)
    try:
        yield counts
    finally:
        for name, (socket, original) in originals.items():
            try:
                del socket.decode
            except AttributeError:  # pragma: no cover - defensive
                socket.decode = original


# --------------------------------------------------------------------------
# preview envelope checks (pure: fed a list of recorded messages)
# --------------------------------------------------------------------------


def check_preview_messages(
    messages: Sequence[Tuple[str, Dict[str, Any], Optional[str]]],
    *,
    node_id: str,
    expect_end: str,
    expect_media: bool,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Check], bytes]:
    """Validate one session's envelopes against ``web/PROTOCOL.md`` v1.

    Pure and offline: it is handed the ``(message_type, body, client_id)``
    triples a :class:`~raven_streaming.preview.RecordingSender` collected, and
    returns ``(summary, checks, media_bytes)`` where ``media_bytes`` is the
    init segment followed by every fragment, in order -- i.e. exactly the byte
    stream an MSE ``SourceBuffer`` would have been appended.
    """
    from raven_streaming.media.mp4_boxes import box_types
    from raven_streaming.preview_session import (
        BACKEND_PHASES,
        EVENT_KINDS,
        MESSAGE_TYPE,
        PROTOCOL_VERSION,
    )

    checks: List[Check] = []

    def expect(name: str, condition: Any, detail: str = "", gate: bool = True) -> bool:
        ok = bool(condition)
        checks.append(Check(name=name, ok=ok, detail=detail, gate=gate))
        return ok

    bodies = [body for _type, body, _sid in messages]
    types = {message_type for message_type, _body, _sid in messages}
    events = [str(body.get("event")) for body in bodies]
    seqs = [body.get("seq") for body in bodies]

    summary: Dict[str, Any] = {
        "messages": len(bodies),
        "events": events,
        "seqs": seqs,
        "phases": [b.get("phase") for b in bodies if b.get("event") == "status"],
        "segments": events.count("segment"),
        "init_bytes": 0,
        "segment_bytes": 0,
        "segment_sizes": [],
        "end_reason": None,
        "session_ids": sorted({str(b.get("session_id")) for b in bodies}),
        "node_ids": sorted({str(b.get("node_id")) for b in bodies}),
    }

    if not expect(
        "preview: the session produced messages", bodies, "{} recorded".format(len(bodies))
    ):
        return summary, checks, b""

    expect(
        "preview: every message is a {} envelope".format(MESSAGE_TYPE),
        types == {MESSAGE_TYPE},
        "message types: {}".format(sorted(types)),
    )
    expect(
        "preview: protocol version is {}".format(PROTOCOL_VERSION),
        all(b.get("v") == PROTOCOL_VERSION for b in bodies),
        "versions: {}".format(sorted({b.get("v") for b in bodies})),
    )
    expect(
        "preview: one session for the whole run",
        len(summary["session_ids"]) == 1,
        "session ids: {}".format(summary["session_ids"]),
    )
    expect(
        "preview: every message carries the node id",
        summary["node_ids"] == [str(node_id)],
        "node ids: {} (expected {!r})".format(summary["node_ids"], str(node_id)),
    )
    expect(
        "preview: only known event kinds",
        set(events) <= set(EVENT_KINDS),
        "events seen: {}".format(sorted(set(events))),
    )

    # -- ordering -------------------------------------------------------
    expect(
        "preview: seq is contiguous and gap-free",
        seqs == list(range(seqs[0], seqs[0] + len(seqs))) if seqs and seqs[0] is not None else False,
        "seq {}..{} over {} message(s)".format(seqs[0], seqs[-1], len(seqs)),
    )
    expect("preview: the first event is open", events[0] == "open", "first event: {}".format(events[0]))
    expect(
        "preview: exactly one open, and it is first",
        events.count("open") == 1 and events.index("open") == 0,
        "open count: {}".format(events.count("open")),
    )
    expect(
        "preview: exactly one end, and it is last",
        events.count("end") == 1 and events[-1] == "end",
        "end count: {}, last event: {}".format(events.count("end"), events[-1]),
    )
    if "init" in events:
        first_segment = events.index("segment") if "segment" in events else len(events)
        expect(
            "preview: exactly one init, before any segment",
            events.count("init") == 1 and events.index("init") < first_segment,
            "init at {}, first segment at {}".format(events.index("init"), first_segment),
        )
    else:
        expect(
            "preview: no segment without an init",
            "segment" not in events,
            "segments were sent without an init segment",
        )

    phases = summary["phases"]
    expect(
        "preview: only backend status phases",
        set(phases) <= set(BACKEND_PHASES),
        "phases: {}".format(phases),
    )

    # -- open -----------------------------------------------------------
    open_body = bodies[0] if events[0] == "open" else None
    if open_body is not None:
        summary["open"] = {
            key: open_body.get(key)
            for key in ("mime", "width", "height", "fps", "audio", "duration_hint")
        }
        mime = str(open_body.get("mime", ""))
        expect(
            "preview: open carries an MSE-usable mime",
            mime.startswith("video/mp4") and "codecs" in mime,
            "mime: {!r}".format(mime),
        )
        if width is not None and height is not None:
            expect(
                "preview: open canvas matches the request",
                (open_body.get("width"), open_body.get("height")) == (int(width), int(height)),
                "open says {}x{}, request is {}x{}".format(
                    open_body.get("width"), open_body.get("height"), width, height
                ),
            )
        audio = open_body.get("audio") or {}
        expect(
            "preview: open declares 32 kHz stereo audio",
            (audio.get("sample_rate"), audio.get("channels"))
            == (AUDIO_SAMPLE_RATE, AUDIO_OUTPUT_CHANNELS),
            "audio: {}".format(audio),
        )

    # -- payloads -------------------------------------------------------
    init_blob = b""
    fragments: List[bytes] = []
    indices: List[int] = []
    decode_failures: List[str] = []
    for body in bodies:
        event = body.get("event")
        if event not in ("init", "segment"):
            continue
        try:
            raw = base64.b64decode(str(body.get("data", "")), validate=True)
        except Exception as exc:  # noqa: BLE001
            decode_failures.append("seq {}: {}: {}".format(body.get("seq"), type(exc).__name__, exc))
            continue
        if int(body.get("bytes", -1)) != len(raw):
            decode_failures.append(
                "seq {}: bytes={} but payload decodes to {}".format(
                    body.get("seq"), body.get("bytes"), len(raw)
                )
            )
            continue
        if event == "init":
            init_blob = raw
        else:
            fragments.append(raw)
            index = body.get("index")
            if isinstance(index, int):
                indices.append(index)

    expect(
        "preview: every payload is valid base64 of the declared length",
        not decode_failures,
        "; ".join(decode_failures[:5]),
    )
    expect(
        "preview: payload encoding is base64 everywhere",
        all(
            b.get("encoding") == "base64"
            for b in bodies
            if b.get("event") in ("init", "segment")
        ),
        "encodings: {}".format(
            sorted({b.get("encoding") for b in bodies if b.get("event") in ("init", "segment")})
        ),
    )

    summary["init_bytes"] = len(init_blob)
    summary["segment_sizes"] = [len(f) for f in fragments]
    summary["segment_bytes"] = sum(len(f) for f in fragments)

    if expect_media:
        expect("preview: an init segment was sent", bool(init_blob), "init bytes: {}".format(len(init_blob)))
        expect("preview: fragments were sent", bool(fragments), "{} fragment(s)".format(len(fragments)))
    if init_blob:
        try:
            types_seen = list(box_types(init_blob))
        except Exception as exc:  # noqa: BLE001
            types_seen = []
            expect("preview: init parses as MP4 boxes", False, "{}: {}".format(type(exc).__name__, exc))
        summary["init_boxes"] = types_seen
        expect(
            "preview: init is an MSE init segment (ftyp + moov)",
            types_seen[:1] == ["ftyp"] and "moov" in types_seen,
            "init boxes: {}".format(types_seen),
        )
    if fragments:
        bad = []
        first_boxes: List[str] = []
        for position, blob in enumerate(fragments):
            try:
                seen = list(box_types(blob))
            except Exception as exc:  # noqa: BLE001
                bad.append("fragment {}: {}: {}".format(position, type(exc).__name__, exc))
                continue
            if position == 0:
                first_boxes = seen
            if "moof" not in seen or "mdat" not in seen:
                bad.append("fragment {}: boxes {}".format(position, seen))
        summary["first_fragment_boxes"] = first_boxes
        expect(
            "preview: every fragment is moof + mdat",
            not bad,
            "; ".join(bad[:5]),
        )
        expect(
            "preview: fragment indices are contiguous from 0",
            indices == list(range(len(fragments))),
            "indices: {}".format(indices[:8] + (["..."] if len(indices) > 8 else [])),
        )

    # -- end ------------------------------------------------------------
    end_body = bodies[-1] if events[-1] == "end" else None
    if end_body is not None:
        summary["end_reason"] = end_body.get("reason")
        summary["end_segments"] = end_body.get("segments")
        expect(
            "preview: end reason is {!r}".format(expect_end),
            end_body.get("reason") == expect_end,
            "end: {}".format({k: v for k, v in end_body.items() if k in ("reason", "message", "segments")}),
        )
        expect(
            "preview: end counts the segments it sent",
            int(end_body.get("segments", -1)) == len(fragments),
            "end says {}, {} fragment(s) recorded".format(end_body.get("segments"), len(fragments)),
        )

    return summary, checks, init_blob + b"".join(fragments)


def decode_preview_media(blob: bytes, *, expect_frames: Optional[int] = None) -> Tuple[Dict[str, Any], List[Check]]:
    """Decode the streamed bytes with PyAV, in process. No ffmpeg binary.

    This is the only check that answers the question the whole preview lane
    exists for: *would a browser have been able to play what we sent?* Box
    types prove structure; a decoder proves the structure carries frames.
    """
    checks: List[Check] = []
    summary: Dict[str, Any] = {"bytes": len(blob)}

    def expect(name: str, condition: Any, detail: str = "", gate: bool = True) -> bool:
        ok = bool(condition)
        checks.append(Check(name=name, ok=ok, detail=detail, gate=gate))
        return ok

    if not blob:
        expect("media: there is a stream to decode", False, "no preview bytes were captured")
        return summary, checks
    try:
        import av  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        expect(
            "media: PyAV is available to decode the stream",
            False,
            "{}: {} (the preview lane needs PyAV to have produced these bytes at all)".format(
                type(exc).__name__, exc
            ),
        )
        return summary, checks

    summary["pyav_version"] = getattr(av, "__version__", "unknown")
    try:
        with av.open(io.BytesIO(blob)) as container:
            video_streams = [s for s in container.streams if s.type == "video"]
            audio_streams = [s for s in container.streams if s.type == "audio"]
            summary["video_streams"] = len(video_streams)
            summary["audio_streams"] = len(audio_streams)
            if video_streams:
                stream = video_streams[0]
                summary["codec"] = getattr(getattr(stream, "codec_context", None), "name", None)
                summary["width"] = getattr(stream, "width", None)
                summary["height"] = getattr(stream, "height", None)
            if audio_streams:
                stream = audio_streams[0]
                summary["audio_codec"] = getattr(getattr(stream, "codec_context", None), "name", None)
                summary["sample_rate"] = getattr(stream, "sample_rate", None)
        expect(
            "media: exactly one video stream",
            summary.get("video_streams") == 1,
            "video streams: {}".format(summary.get("video_streams")),
        )
        expect(
            "media: exactly one audio stream",
            summary.get("audio_streams") == 1,
            "audio streams: {}".format(summary.get("audio_streams")),
        )
        with av.open(io.BytesIO(blob)) as container:
            summary["video_frames"] = sum(1 for _ in container.decode(video=0))
        with av.open(io.BytesIO(blob)) as container:
            summary["audio_samples"] = sum(int(f.samples) for f in container.decode(audio=0))
    except Exception as exc:  # noqa: BLE001
        expect(
            "media: the concatenated stream decodes",
            False,
            "{}: {}".format(type(exc).__name__, exc),
        )
        summary["traceback"] = traceback.format_exc()
        return summary, checks

    expect(
        "media: the concatenated stream decodes",
        True,
        "{} frame(s), {} audio sample(s)".format(
            summary.get("video_frames"), summary.get("audio_samples")
        ),
    )
    if expect_frames is not None:
        expect(
            "media: every requested frame reached the stream",
            summary.get("video_frames") == int(expect_frames),
            "decoded {} frame(s), requested {}".format(summary.get("video_frames"), expect_frames),
        )
    expect(
        "media: audio is present",
        int(summary.get("audio_samples", 0)) > 0,
        "{} sample(s)".format(summary.get("audio_samples")),
    )
    return summary, checks


# --------------------------------------------------------------------------
# output checks
# --------------------------------------------------------------------------


#: Bytes per element for the dtypes an IMAGE buffer can plausibly be in.
_DTYPE_BYTES = {
    "torch.float32": 4,
    "torch.float": 4,
    "torch.float16": 2,
    "torch.half": 2,
    "torch.bfloat16": 2,
    "torch.float64": 8,
}


def check_streaming_cadence(
    report: Dict[str, Any], *, num_chunks: int, preview_disabled: bool
) -> Tuple[Dict[str, Any], List[Check]]:
    """Did the preview actually *stream*, chunk by chunk, or only at the end?

    "Streaming" is the one claim the whole preview lane exists to make, and it
    is not visible in a total: a run that muxed everything during ``finish()``
    and sent it in one burst produces the same fragment count as a run that
    emitted all the way through. What separates them is **when** bytes went
    out, which ``PipelineReport.chunk_emissions`` and ``fragments_before_finish``
    record per chunk.

    What is gated:

    * something reached the wire **before** the tail flush;
    * every *intermediate* chunk emitted at least one fragment -- "as soon as
      it can". The first chunk is exempt (the documented one-chunk startup
      delay: the decoder needs 2 latents of lookahead and the audio VAE adds
      its own margin) and so is the last, whose frames are the tail flush.

    What is **not** gated: any *proportion*. How much of the clip is out before
    the flush depends on the geometry, the encoder and the machine; the ratios
    are measured and reported, never asserted against a number made up here.
    """
    checks: List[Check] = []
    emissions = list(report.get("chunk_emissions", []) or [])
    fragments = int(report.get("fragments", 0))
    before_finish = int(report.get("fragments_before_finish", 0))
    frames_total = int(report.get("frames_muxed", 0))
    frames_before = sum(int(e.get("muxed_frames", 0)) for e in emissions)
    samples_before = sum(int(e.get("muxed_samples", 0)) for e in emissions)
    emitting = [int(e.get("chunk", -1)) for e in emissions if int(e.get("fragments", 0)) > 0]
    intermediates = set(range(1, max(0, int(num_chunks) - 1)))

    summary: Dict[str, Any] = {
        "chunks_recorded": len(emissions),
        "emitting_chunks": emitting,
        "intermediate_chunks": sorted(intermediates),
        "fragments": fragments,
        "fragments_before_finish": before_finish,
        "fragments_at_finish": max(0, fragments - before_finish),
        "frames_muxed_before_finish": frames_before,
        "frames_muxed_total": frames_total,
        "samples_muxed_before_finish": samples_before,
        "max_held_frames": max(
            [int(e.get("held_frames", 0)) for e in emissions] or [0]
        ),
        "first_fragment_chunk": report.get("first_fragment_chunk"),
        "first_fragment_latency": report.get("first_fragment_latency"),
        "emission_log": emissions,
    }
    if fragments:
        summary["fragments_before_finish_ratio"] = before_finish / float(fragments)
    if frames_total:
        summary["frames_before_finish_ratio"] = frames_before / float(frames_total)

    if preview_disabled:
        checks.append(
            Check(
                name="streaming: cadence not measurable",
                ok=True,
                detail="the preview lane was disabled, so nothing was emitted to time",
                gate=False,
            )
        )
        return summary, checks

    checks.append(
        Check(
            name="streaming: every chunk is accounted for in the emission log",
            ok=len(emissions) == int(num_chunks),
            detail="{} emission record(s) for {} chunk(s)".format(
                len(emissions), num_chunks
            ),
        )
    )
    # A clip with no intermediate chunk is *all* startup delay and tail: chunk 0
    # cannot emit (the video decoder needs 2 latents of lookahead and the audio
    # VAE adds its own margin) and chunk 1 is the flush. Measured on this
    # repository's own pipeline: 22 frames / 2 chunks emits nothing before
    # finish(), 39 / 3 emits from chunk 1, 192 / 12 emits from chunks 1-10. So
    # the cadence is only *gated* where a cadence exists.
    streams_early = int(num_chunks) >= 3
    checks.append(
        Check(
            name="streaming: fragments went out before the tail flush",
            ok=before_finish >= 1 if streams_early else True,
            detail="{} of {} fragment(s) were already sent when finish() was "
            "called{}".format(
                before_finish,
                fragments,
                ""
                if streams_early
                else " -- not gated: {} chunk(s) is entirely the one-chunk startup "
                "delay plus the tail flush, so there is no steady state to "
                "observe".format(num_chunks),
            ),
            gate=streams_early,
        )
    )
    if intermediates:
        missing = sorted(intermediates - set(emitting))
        checks.append(
            Check(
                name="streaming: every intermediate chunk emitted as soon as it could",
                ok=not missing,
                detail="chunk(s) {} produced no fragment; chunk 0 is exempt (the "
                "one-chunk startup delay) and so is chunk {} (the tail "
                "flush)".format(missing or "-", int(num_chunks) - 1),
            )
        )
    checks.append(
        Check(
            name="streaming: how much of the clip was out before the flush",
            ok=True,
            detail="{} of {} fragment(s) ({:.0%}), {} of {} muxed frame(s) ({:.0%}), "
            "{} audio sample(s); first fragment after chunk {} at {}. Measured, not "
            "expected: the proportion is a property of the geometry and the "
            "machine.".format(
                before_finish,
                fragments,
                summary.get("fragments_before_finish_ratio", 0.0),
                frames_before,
                frames_total,
                summary.get("frames_before_finish_ratio", 0.0),
                samples_before,
                report.get("first_fragment_chunk"),
                "n/a"
                if report.get("first_fragment_latency") is None
                else "{:.2f}s".format(report["first_fragment_latency"]),
            ),
            gate=False,
        )
    )
    return summary, checks


def check_collector_report(
    report: Dict[str, Any], *, geometry: Dict[str, int], cancelled: bool = False
) -> List[Check]:
    """Gate the collector half of ``PipelineReport``. It **is** the IMAGE.

    Since the node stopped calling ``video_vae.decode`` on the whole clip,
    these numbers are not preview diagnostics any more -- they describe the
    output the workflow gets. ``collected_frames``, ``expected_frames`` and the
    buffer's own shape/size are therefore gates, not notes: a collector that
    quietly wrote 34 of 39 frames would hand back a short clip, and the only
    place that shows up before the user sees it is here.

    A cancelled run is exempt: it releases the buffer on purpose and returns no
    partial IMAGE.
    """
    checks: List[Check] = []

    def expect(name: str, condition: Any, detail: str = "", gate: bool = True) -> None:
        checks.append(Check(name=name, ok=bool(condition), detail=detail, gate=gate))

    frames = int(geometry["frames"])
    collected = int(report.get("collected_frames", -1))
    expected = int(report.get("expected_frames", -1))
    shape = list(report.get("image_shape", []) or [])
    image_bytes = int(report.get("image_bytes", 0))
    dtype = str(report.get("image_dtype", ""))

    audio_collected = int(report.get("collected_samples", -1))
    audio_expected = int(report.get("expected_samples", -1))
    audio_shape = list(report.get("audio_shape", []) or [])
    audio_bytes = int(report.get("audio_bytes", 0))
    audio_dtype = str(report.get("audio_dtype", ""))
    samples = int(geometry["audio_t"]) * AUDIO_SAMPLES_PER_LATENT

    if cancelled:
        expect(
            "collector: a cancelled run kept no partial IMAGE",
            collected == 0 or not shape,
            "collected={} shape={}".format(collected, shape),
            gate=False,
        )
        expect(
            "collector: a cancelled run kept no partial AUDIO",
            audio_collected == 0 or not audio_shape,
            "collected={} shape={}".format(audio_collected, audio_shape),
            gate=False,
        )
        return checks

    expect(
        "collector: expected_frames is the requested frame count",
        expected == frames,
        "report says {}, the request is {}".format(expected, frames),
    )
    expect(
        "collector: every frame was collected",
        collected == frames and bool(report.get("image_complete")),
        "collected {}/{} (image_complete={})".format(
            collected, expected, report.get("image_complete")
        ),
    )
    expected_shape = [frames, int(geometry["height"]), int(geometry["width"]), 3]
    expect(
        "collector: the IMAGE buffer is [frames, H, W, 3]",
        shape == expected_shape,
        "{} (expected {})".format(shape, expected_shape),
    )
    itemsize = _DTYPE_BYTES.get(dtype)
    if shape and itemsize:
        size = itemsize
        for axis in shape:
            size *= int(axis)
        expect(
            "collector: image_bytes accounts for the whole buffer",
            image_bytes == size,
            "{} B reported, {} B for {} {} ({} B/element)".format(
                image_bytes, size, shape, dtype, itemsize
            ),
        )
    else:
        expect(
            "collector: image_bytes accounts for the whole buffer",
            image_bytes > 0,
            "{} B reported for shape {} dtype {!r}".format(image_bytes, shape, dtype),
        )
    checks.append(
        Check(
            name="collector: where the IMAGE was built",
            ok=True,
            detail="{} {} on {}".format(shape, dtype, report.get("image_device")),
            gate=False,
        )
    )

    # -- the audio collector, which is now the AUDIO output ---------------
    expect(
        "collector: expected_samples is the clip's audio length",
        audio_expected == samples,
        "report says {}, the grid says {} ({} audio latent(s) x {} sample(s))".format(
            audio_expected, samples, geometry["audio_t"], AUDIO_SAMPLES_PER_LATENT
        ),
    )
    expect(
        "collector: every audio sample was collected",
        audio_collected == samples and bool(report.get("audio_complete")),
        "collected {}/{} (audio_complete={})".format(
            audio_collected, audio_expected, report.get("audio_complete")
        ),
    )
    expected_audio_shape = [1, AUDIO_OUTPUT_CHANNELS, samples]
    expect(
        "collector: the waveform buffer is [1, 2, samples]",
        audio_shape == expected_audio_shape,
        "{} (expected {})".format(audio_shape, expected_audio_shape),
    )
    audio_itemsize = _DTYPE_BYTES.get(audio_dtype)
    if audio_shape and audio_itemsize:
        size = audio_itemsize
        for axis in audio_shape:
            size *= int(axis)
        expect(
            "collector: audio_bytes accounts for the whole buffer",
            audio_bytes == size,
            "{} B reported, {} B for {} {}".format(
                audio_bytes, size, audio_shape, audio_dtype
            ),
        )
    else:
        expect(
            "collector: audio_bytes accounts for the whole buffer",
            audio_bytes > 0,
            "{} B reported for shape {} dtype {!r}".format(
                audio_bytes, audio_shape, audio_dtype
            ),
        )
    checks.append(
        Check(
            name="collector: where the AUDIO was built",
            ok=True,
            detail="{} {} on {} ({:.1f} MiB host)".format(
                audio_shape,
                audio_dtype,
                report.get("audio_device"),
                audio_bytes / (1024.0 ** 2),
            ),
            gate=False,
        )
    )
    return checks


def _stream_stats(tensor: Any) -> Dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all())
    stats: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
    }
    if finite and tensor.numel():
        values = tensor.detach().float()
        stats.update(
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()) if values.numel() > 1 else 0.0,
            }
        )
    return stats


def check_outputs(
    latent: Any,
    images: Any,
    audio: Any,
    *,
    geometry: Dict[str, int],
) -> Tuple[Dict[str, Any], List[Check], Dict[str, Any]]:
    """Check the node's three outputs against the contract in ``README.md``.

    Returns ``(summary, checks, artifacts)``; ``artifacts`` holds CPU copies of
    the four tensors so a second run can be compared against them without
    keeping a GPU allocation alive between runs.
    """
    checks: List[Check] = []
    summary: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}

    def expect(name: str, condition: Any, detail: str = "", gate: bool = True) -> bool:
        ok = bool(condition)
        checks.append(Check(name=name, ok=ok, detail=detail, gate=gate))
        return ok

    # -- LATENT ---------------------------------------------------------
    if not expect(
        "output: LATENT is a dict with samples",
        isinstance(latent, dict) and "samples" in latent,
        "got {}".format(type(latent).__name__),
    ):
        return summary, checks, artifacts
    samples = latent["samples"]
    if not expect(
        "output: LATENT samples is a NestedTensor (video, audio) pair",
        bool(getattr(samples, "is_nested", False)),
        "got {}".format(type(samples).__name__),
    ):
        return summary, checks, artifacts
    streams = list(samples.unbind())
    if not expect(
        "output: LATENT carries exactly two streams",
        len(streams) == 2,
        "{} stream(s)".format(len(streams)),
    ):
        return summary, checks, artifacts
    video_latent, audio_latent = streams
    artifacts["video_latent"] = video_latent.detach().to("cpu").clone()
    artifacts["audio_latent"] = audio_latent.detach().to("cpu").clone()
    summary["video_latent"] = _stream_stats(video_latent)
    summary["audio_latent"] = _stream_stats(audio_latent)

    expected_video = [
        1,
        24,
        int(geometry["latent_t"]),
        int(geometry["latent_h"]),
        int(geometry["latent_w"]),
    ]
    expected_audio = [1, 32, AUDIO_OUTPUT_CHANNELS, int(geometry["audio_t"])]
    expect(
        "output: video latent shape",
        list(video_latent.shape) == expected_video,
        "{} (expected {})".format(list(video_latent.shape), expected_video),
    )
    expect(
        "output: audio latent shape",
        list(audio_latent.shape) == expected_audio,
        "{} (expected {})".format(list(audio_latent.shape), expected_audio),
    )
    expect("output: video latent is finite", summary["video_latent"]["finite"])
    expect("output: audio latent is finite", summary["audio_latent"]["finite"])

    # -- IMAGE ----------------------------------------------------------
    expected_images = [
        int(geometry["frames"]),
        int(geometry["height"]),
        int(geometry["width"]),
        3,
    ]
    if expect(
        "output: IMAGE is a tensor",
        isinstance(images, torch.Tensor),
        "got {}".format(type(images).__name__),
    ):
        artifacts["images"] = images.detach().to("cpu").clone()
        summary["images"] = _stream_stats(images)
        expect(
            "output: IMAGE shape is [frames, H, W, 3]",
            list(images.shape) == expected_images,
            "{} (expected {})".format(list(images.shape), expected_images),
        )
        expect("output: IMAGE is finite", summary["images"]["finite"])
        if summary["images"].get("finite"):
            low = summary["images"]["min"]
            high = summary["images"]["max"]
            expect(
                "output: IMAGE is in [0, 1]",
                low >= -PIXEL_RANGE_TOLERANCE and high <= 1.0 + PIXEL_RANGE_TOLERANCE,
                "min={:.6f} max={:.6f}".format(low, high),
            )

    # -- AUDIO ----------------------------------------------------------
    if expect(
        "output: AUDIO is a dict with waveform and sample_rate",
        isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio,
        "got {}".format(type(audio).__name__),
    ):
        waveform = audio["waveform"]
        artifacts["waveform"] = waveform.detach().to("cpu").clone()
        summary["audio"] = _stream_stats(waveform)
        summary["audio"]["sample_rate"] = int(audio["sample_rate"])
        expect(
            "output: AUDIO sample rate is {} Hz".format(AUDIO_SAMPLE_RATE),
            int(audio["sample_rate"]) == AUDIO_SAMPLE_RATE,
            "got {}".format(audio["sample_rate"]),
        )
        shape = list(waveform.shape)
        expect(
            "output: AUDIO waveform is [1, 2, N]",
            len(shape) == 3 and shape[0] == 1 and shape[1] == AUDIO_OUTPUT_CHANNELS and shape[2] > 0,
            "{}".format(shape),
        )
        expect("output: AUDIO is finite", summary["audio"]["finite"])
        expected_samples = int(geometry["audio_t"]) * AUDIO_SAMPLES_PER_LATENT
        if len(shape) == 3:
            summary["audio"]["expected_samples"] = expected_samples
            expect(
                "output: AUDIO length matches the audio latent",
                shape[2] == expected_samples,
                "{} sample(s), expected {} (audio_t {} x {})".format(
                    shape[2], expected_samples, geometry["audio_t"],
                    AUDIO_SAMPLES_PER_LATENT,
                ),
            )
        # -- the official whole-clip normalisation ------------------------
        #
        # ``finalize_audio`` reproduces ``vae_decode_audio``'s tail:
        #
        #     std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        #     std[std < 1.0] = 1.0
        #     audio /= std
        #
        # so a clip loud enough to be divided lands on exactly std == 0.2 and a
        # quiet one keeps its own, smaller std. Checking the *invariant* rather
        # than a fixed number is what makes this true for silence as well as
        # for music.
        if isinstance(waveform, torch.Tensor) and waveform.numel() > 1:
            observed = float(torch.std(waveform.float(), dim=[1, 2]).max())
            peak = float(waveform.float().abs().max())
            summary["audio"]["std"] = observed
            summary["audio"]["peak"] = peak
            summary["audio"]["normalised"] = (
                "vae_decode_audio tail: /= max(1, std*5), applied to the whole clip"
            )
            expect(
                "output: AUDIO carries the official whole-clip normalisation",
                observed <= AUDIO_NORMALISED_STD + AUDIO_NORMALISED_STD_TOLERANCE,
                "std={:.6f} (a scaled clip lands on {}, a quiet one stays below it); "
                "peak={:.6f}".format(observed, AUDIO_NORMALISED_STD, peak),
            )

    return summary, checks, artifacts


# --------------------------------------------------------------------------
# optional diagnostic: the collector against the whole-clip decode
# --------------------------------------------------------------------------


def official_compare_allowed(geometry: Dict[str, int]) -> Tuple[bool, str]:
    """May the whole-clip reference decode be attempted at this geometry?

    The product path stopped calling ``video_vae.decode`` on the whole clip for
    a measured reason: a 39-frame run died in it at 130.22 GiB allocated /
    139.12 GiB reserved (``nodes.prepare_final_decode``'s docstring). Running it
    again as a *diagnostic* is only defensible where it fits, so the probe
    refuses the sizes where it is known not to -- 192 and 362 frames above all --
    rather than turning a reference check into an OOM.

    The bound is on decoded pixels, not on the frame count alone: 39 frames at
    1376x768 is the same problem as 192 frames at a small canvas.
    """
    pixels = int(geometry["frames"]) * int(geometry["width"]) * int(geometry["height"])
    if pixels > OFFICIAL_COMPARE_MAX_PIXELS:
        return False, (
            "{}x{}x{} frame(s) = {} decoded pixels is over the {} the reference decode "
            "is allowed (the whole-clip decode is what OOMed the measured run; use "
            "--frames 39 at a small canvas for this comparison)".format(
                geometry["frames"],
                geometry["width"],
                geometry["height"],
                pixels,
                OFFICIAL_COMPARE_MAX_PIXELS,
            )
        )
    return True, "{} decoded pixels, within the {} bound".format(
        pixels, OFFICIAL_COMPARE_MAX_PIXELS
    )


def compare_official_video(
    *,
    video_vae: Any,
    latent: Any,
    images: Any,
    geometry: Dict[str, int],
    device: torch.device,
    decode_images: Any,
) -> Tuple[Dict[str, Any], List[Check]]:
    """Decode the finished latent the *old* way and compare it to the IMAGE.

    Strictly a diagnostic, and strictly after the node has returned: by then
    ``nodes.prepare_final_decode`` has already evicted the DiT (the node does it
    on its way out), so this runs in the memory state the whole-clip decode was
    always supposed to have -- which is the only state in which its cost means
    anything.

    What is gated and what is not:

    * **gated** -- that the reference decode completed at all. An OOM here is a
      finding, not a shrug: it is the same allocation the product path removed,
      measured at a size the probe already decided was small enough for it.
    * **gated** -- that it produced the same shape as the collector.
    * **not gated** -- bitwise equality. The collector decodes 7 latents at a
      time and the reference decodes the whole clip in one call; identical
      results are what the unit tests pin on CPU, but a different cuBLAS/cuDNN
      algorithm for a different tensor shape is a hardware fact, not a bug in
      this package. The numbers are reported either way, which is the point.

    Nothing here touches the product's outputs: the reference tensor is built,
    measured and dropped.
    """
    checks: List[Check] = []
    summary: Dict[str, Any] = {}

    allowed, reason = official_compare_allowed(geometry)
    summary["allowed"] = allowed
    summary["bound"] = reason
    if not allowed:
        checks.append(
            Check(
                name="official-decode: comparison skipped at this geometry",
                ok=True,
                detail=reason,
                gate=False,
            )
        )
        return summary, checks

    on_cuda = device.type == "cuda"
    if on_cuda:
        summary["allocated_before"] = int(torch.cuda.memory_allocated(device))
        summary["reserved_before"] = int(torch.cuda.memory_reserved(device))
        summary["peak_allocated_before"] = int(torch.cuda.max_memory_allocated(device))
        summary["peak_reserved_before"] = int(torch.cuda.max_memory_reserved(device))

    official = None
    started = time.perf_counter()
    try:
        official = decode_images(video_vae, latent)
        if on_cuda:
            torch.cuda.synchronize(device)
    except BaseException as exc:  # noqa: BLE001 - an OOM is the finding
        summary["seconds"] = time.perf_counter() - started
        summary["error"] = "{}: {}".format(type(exc).__name__, exc)
        oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        summary["out_of_memory"] = oom
        checks.append(
            Check(
                name="official-decode: the whole-clip reference decode completed",
                ok=False,
                detail=summary["error"]
                + (
                    " -- the DiT handover had already run, so this is the decode itself "
                    "not fitting"
                    if oom
                    else ""
                ),
            )
        )
        if on_cuda:
            summary["peak_allocated_after"] = int(torch.cuda.max_memory_allocated(device))
            summary["peak_reserved_after"] = int(torch.cuda.max_memory_reserved(device))
            torch.cuda.empty_cache()
        return summary, checks

    summary["seconds"] = time.perf_counter() - started
    checks.append(
        Check(
            name="official-decode: the whole-clip reference decode completed",
            ok=True,
            detail="{:.2f}s".format(summary["seconds"]),
        )
    )
    if on_cuda:
        summary["peak_allocated_after"] = int(torch.cuda.max_memory_allocated(device))
        summary["peak_reserved_after"] = int(torch.cuda.max_memory_reserved(device))
        summary["extra_peak_allocated"] = summary["peak_allocated_after"] - summary.get(
            "peak_allocated_before", 0
        )
        summary["extra_peak_reserved"] = summary["peak_reserved_after"] - summary.get(
            "peak_reserved_before", 0
        )

    summary["shape"] = list(getattr(official, "shape", []))
    collector_shape = list(getattr(images, "shape", []))
    checks.append(
        Check(
            name="official-decode: the reference has the collector's shape",
            ok=summary["shape"] == collector_shape,
            detail="reference {} vs collector {}".format(summary["shape"], collector_shape),
        )
    )
    metrics = tensor_metrics(images, official)
    summary["metrics"] = metrics
    checks.append(
        Check(
            name="official-decode: the collector matches the whole-clip decode",
            ok=bool(metrics.get("comparable")) and bool(metrics.get("bitwise")),
            detail=_metric_detail(metrics, with_bitwise=True),
            # A kernel that picks a different algorithm for a different tensor
            # shape is a hardware fact; the numbers are the finding, not a verdict.
            gate=False,
        )
    )

    del official
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    return summary, checks


# --------------------------------------------------------------------------
# cross-run comparison
# --------------------------------------------------------------------------


def tensor_metrics(a: Any, b: Any) -> Dict[str, Any]:
    """Bitwise equality first, then the numbers that describe a near-miss."""
    if a is None or b is None:
        return {"comparable": False, "reason": "one side is missing"}
    if list(a.shape) != list(b.shape):
        return {
            "comparable": False,
            "reason": "shape mismatch",
            "shapes": [list(a.shape), list(b.shape)],
        }
    left = a.detach().to("cpu")
    right = b.detach().to("cpu")
    bitwise = bool(torch.equal(left, right))
    lf = left.float()
    rf = right.float()
    diff = (lf - rf).abs()
    denominator = float(torch.linalg.vector_norm(lf)) if lf.numel() else 0.0
    return {
        "comparable": True,
        "bitwise": bitwise,
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mse": float((diff ** 2).mean()) if diff.numel() else 0.0,
        "rel_l2": (
            float(torch.linalg.vector_norm(lf - rf) / denominator) if denominator > 0 else 0.0
        ),
    }


def artifact_metrics(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    """Every cross-run comparison of one pair of runs' outputs."""
    a = first.get("artifacts", {})
    b = second.get("artifacts", {})
    return {
        "latent": {
            "video": tensor_metrics(a.get("video_latent"), b.get("video_latent")),
            "audio": tensor_metrics(a.get("audio_latent"), b.get("audio_latent")),
        },
        "images": tensor_metrics(a.get("images"), b.get("images")),
        "audio": tensor_metrics(a.get("waveform"), b.get("waveform")),
    }


def _metric_detail(metrics: Dict[str, Any], *, with_bitwise: bool = False) -> str:
    if not metrics.get("comparable"):
        return str(metrics.get("reason"))
    head = "bitwise={} ".format(metrics.get("bitwise")) if with_bitwise else ""
    return head + "max|d|={max_abs:.6g} rel_l2={rel_l2:.6g} mse={mse:.6g}".format(**metrics)


def bitwise_of(metrics: Dict[str, Any], key: str, stream: Optional[str] = None) -> bool:
    """``True`` only when that pair really was compared and matched bitwise."""
    entry = metrics.get(key, {})
    if stream is not None:
        entry = (entry or {}).get(stream, {})
    return bool(entry.get("comparable")) and bool(entry.get("bitwise"))


def image_audio_checks(
    metrics: Dict[str, Any], *, suffix: str, gate: bool
) -> List[Check]:
    """Gate (or merely report) IMAGE and AUDIO bitwise equality for one pair.

    Whether this can be a gate depends entirely on *which* pair it is. Run 1 is
    cold: it is where the allocator grows, where cuBLAS picks and caches a
    workspace, and where PyAV's encoder is constructed. A decode kernel that
    chooses a different algorithm the first time it sees a shape produces an
    IMAGE that differs from every later run's in the last bit, and failing the
    probe for that would be reporting the hardware. Two *warm* runs have no
    such excuse, so between them this is a gate.
    """
    checks: List[Check] = []
    for label, key in (("IMAGE", "images"), ("AUDIO", "audio")):
        checks.append(
            Check(
                name="determinism: {} is bitwise identical {}".format(label, suffix)
                if gate
                else "determinism: {} metrics {}".format(label, suffix),
                ok=bitwise_of(metrics, key) if gate else True,
                detail=_metric_detail(metrics[key], with_bitwise=True),
                gate=gate,
            )
        )
    return checks


def determinism_checks(metrics: Dict[str, Any], *, suffix: str = "across runs") -> List[Check]:
    """Gate the latents bitwise; record IMAGE/AUDIO without gating them.

    A non-deterministic kernel is not absorbed into a tolerance here: the node
    documents that a seed determines the clip, and a run that is only *nearly*
    reproducible has broken that, even when the difference is small. The size of
    the miss is still recorded, because "1 ulp of bf16 accumulation order" and
    "a different video" need telling apart.

    IMAGE / AUDIO are decoded *from* the latents, so the latent gate is the one
    that localises the fault; their numbers are reported for context.
    """
    checks: List[Check] = []
    for stream in ("video", "audio"):
        stream_metrics = metrics["latent"][stream]
        checks.append(
            Check(
                name="determinism: {} latent is bitwise identical {}".format(stream, suffix),
                ok=bool(stream_metrics.get("comparable")) and bool(stream_metrics.get("bitwise")),
                detail=_metric_detail(stream_metrics),
            )
        )
    checks.extend(image_audio_checks(metrics, suffix=suffix, gate=False))
    return checks


def memory_growth(first_memory: Dict[str, Any], second_memory: Dict[str, Any]) -> Dict[str, Any]:
    """Deltas of the *current* (not peak) resource use between two runs."""
    growth: Dict[str, Any] = {
        "rss_bytes": int(second_memory.get("rss_after", 0)) - int(first_memory.get("rss_after", 0))
    }
    if "cuda_allocated_after" in first_memory and "cuda_allocated_after" in second_memory:
        growth["cuda_allocated_bytes"] = int(second_memory["cuda_allocated_after"]) - int(
            first_memory["cuda_allocated_after"]
        )
        growth["cuda_reserved_bytes"] = int(second_memory.get("cuda_reserved_after", 0)) - int(
            first_memory.get("cuda_reserved_after", 0)
        )
    return growth


def growth_checks(
    growth: Dict[str, Any], *, label: str, window: str, gate: bool = True
) -> List[Check]:
    """Turn one growth measurement into checks (or diagnostics when ``gate``
    is False)."""
    checks = [
        Check(
            name="repeat: host RSS {}".format(label),
            ok=int(growth.get("rss_bytes", 0)) <= RSS_GROWTH_TOLERANCE_BYTES,
            detail="{} {} (tolerance {})".format(
                _gib(growth.get("rss_bytes", 0)), window, _gib(RSS_GROWTH_TOLERANCE_BYTES)
            ),
            gate=gate,
        )
    ]
    if "cuda_allocated_bytes" in growth:
        checks.append(
            Check(
                name="repeat: CUDA allocation {}".format(label),
                ok=int(growth["cuda_allocated_bytes"]) <= CUDA_GROWTH_TOLERANCE_BYTES,
                detail="allocated +{}, reserved +{} {} (tolerance {})".format(
                    _gib(growth["cuda_allocated_bytes"]),
                    _gib(growth.get("cuda_reserved_bytes", 0)),
                    window,
                    _gib(CUDA_GROWTH_TOLERANCE_BYTES),
                ),
                gate=gate,
            )
        )
    return checks


def compare_runs(first: Dict[str, Any], second: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Check]]:
    """The strict two-run gate: same seed, same process, same clip, same size.

    With exactly two sampled runs there is nowhere to put the warm-up: the
    first run is the one that made the DiT and the two VAEs resident and filled
    the allocator's caches, and the second is the only one that can be compared
    to it. So the growth between them is gated directly -- deliberately
    strictly, because that is all the evidence two runs offer.

    Three or more runs get :func:`compare_run_series` instead, which can tell
    warm-up from a leak.
    """
    summary = artifact_metrics(first, second)
    checks = determinism_checks(summary)

    growth = memory_growth(first.get("memory", {}), second.get("memory", {}))
    summary["rss_growth_bytes"] = growth["rss_bytes"]
    if "cuda_allocated_bytes" in growth:
        summary["cuda_allocated_growth_bytes"] = growth["cuda_allocated_bytes"]
        summary["cuda_reserved_growth_bytes"] = growth["cuda_reserved_bytes"]
    checks.extend(
        growth_checks(growth, label="did not grow materially", window="after run 2 vs run 1")
    )
    return summary, checks


def compare_run_series(runs: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Check]]:
    """Compare every sampled run against the first, and gate the *plateau*.

    Determinism, in three views of the same set of runs:

    * **every run against run 1**, which is the gate for the two *latent*
      streams. Comparing neighbours instead would let a slow drift pass one
      pair at a time, so this is the view that has to be exhaustive;
    * **each adjacent pair**, which is what localises *when* a series stopped
      agreeing -- "run 4 differs from run 1" and "run 4 differs from run 3" are
      different faults;
    * **the full pairwise matrix**, recorded as a per-artifact boolean grid.
      It costs a handful of CPU comparisons and turns "something drifted" into
      a picture of exactly which runs agree with which.

    IMAGE and AUDIO are gated **only between the last two runs**. Run 1 is cold
    -- allocator pools, cuBLAS workspace selection, the encoder's construction
    all happen there -- and a decode kernel that picks a different algorithm on
    its first sight of a shape is hardware, not a bug in this node. So the
    cold-vs-warm comparisons are diagnostics and the warm plateau (with
    ``--repeat 3``: run 2 against run 3) is the reproducibility gate. The
    latents stay gated everywhere, since they are produced before any of that
    matters.

    Memory, and why this is not just "did it grow":

    The first sampled run is a **warm-up** and its cost is not a leak. It is
    the run during which ``load_models_gpu`` makes the DiT's ModelPatcher and
    both VAE patchers resident, cuBLAS/cuDNN allocate their workspaces, the
    caching allocator grows its pools and PyAV's encoder is constructed. All of
    that is *supposed* to still be there for run 2, and a gate on run 1 -> run 2
    would fail a perfectly healthy process for doing what it was told.

    What a leak actually looks like is growth that keeps happening once the
    process is warm. So with **three or more** sampled runs the gate is the
    growth between the **last two** -- the plateau -- and the first-to-last
    total is recorded as a diagnostic, never as a gate. With only two runs there
    is no plateau to measure and the old strict gate stands.
    """
    runs = list(runs)
    checks: List[Check] = []
    summary: Dict[str, Any] = {
        "runs": len(runs),
        "comparisons": [],
        "adjacent": [],
        "pairwise": [],
        "bitwise_matrix": {},
        "memory": {},
    }

    if len(runs) < 2:
        summary["memory"] = {
            "mode": "single",
            "reason": "one sampled run: nothing to compare it against",
        }
        checks.append(
            Check(
                name="determinism: two or more sampled runs were compared",
                ok=True,
                detail="--repeat 1: determinism and memory-growth gates need at least 2 runs",
                gate=False,
            )
        )
        return summary, checks

    # -- every pair, computed once ---------------------------------------
    #
    # The three views below are slices of this one dict, so a pair can never be
    # described two different ways by two different gates.
    first = runs[0]
    pairs: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for left in range(1, len(runs) + 1):
        for right in range(left + 1, len(runs) + 1):
            pairs[(left, right)] = artifact_metrics(runs[left - 1], runs[right - 1])

    summary["pairwise"] = [
        dict(metrics, a=left, b=right) for (left, right), metrics in sorted(pairs.items())
    ]
    for name, key, stream in (
        ("video_latent", "latent", "video"),
        ("audio_latent", "latent", "audio"),
        ("images", "images", None),
        ("audio", "audio", None),
    ):
        grid: List[List[bool]] = []
        for left in range(1, len(runs) + 1):
            row: List[bool] = []
            for right in range(1, len(runs) + 1):
                if left == right:
                    row.append(True)
                    continue
                pair = pairs[(min(left, right), max(left, right))]
                row.append(bitwise_of(pair, key, stream))
            grid.append(row)
        summary["bitwise_matrix"][name] = grid

    # -- determinism: everything against run 1 ---------------------------
    for position in range(2, len(runs) + 1):
        metrics = pairs[(1, position)]
        summary["comparisons"].append(dict(metrics, run=position, against=1))
        checks.extend(
            determinism_checks(metrics, suffix="between run {} and run 1".format(position))
        )

    # -- determinism: each adjacent pair ---------------------------------
    #
    # Reported for every neighbour; gated for the **last** pair only, and there
    # gated for IMAGE and AUDIO too, because both of its runs are warm. The
    # latents of every adjacent pair are already covered by the vs-run-1 gate
    # above (bitwise equality is transitive), so repeating them here would only
    # duplicate failures.
    for position in range(2, len(runs) + 1):
        metrics = pairs[(position - 1, position)]
        summary["adjacent"].append(dict(metrics, run=position, against=position - 1))

    warm_pair = (len(runs) - 1, len(runs))
    warm_metrics = pairs[warm_pair]
    plateau_gated = len(runs) >= 3
    summary["image_reproducibility"] = {
        "gated_pair": list(warm_pair) if plateau_gated else None,
        "gated": plateau_gated,
        "images_bitwise": bitwise_of(warm_metrics, "images"),
        "audio_bitwise": bitwise_of(warm_metrics, "audio"),
        "cold_first_run_is_diagnostic_only": True,
        "note": (
            "IMAGE/AUDIO equality is gated between the last two runs, which are both "
            "warm. Comparisons involving run 1 are diagnostics: the first decode of a "
            "shape is where a kernel picks and caches its algorithm"
            if plateau_gated
            else "only two sampled runs, so the one available pair is cold-vs-warm and "
            "IMAGE/AUDIO equality is reported, not gated; --repeat 3 makes it a gate"
        ),
    }
    checks.extend(
        image_audio_checks(
            warm_metrics,
            suffix="between the last two warm runs ({} and {})".format(*warm_pair),
            gate=plateau_gated,
        )
    )

    # -- memory: plateau when there is one, strict when there is not ------
    if len(runs) >= 3:
        plateau = memory_growth(runs[-2].get("memory", {}), runs[-1].get("memory", {}))
        total = memory_growth(first.get("memory", {}), runs[-1].get("memory", {}))
        summary["memory"] = {
            "mode": "plateau",
            "gated": plateau,
            "diagnostic_total": total,
            "gate_window": "run {} -> run {}".format(len(runs) - 1, len(runs)),
            "note": (
                "run 1 is a warm-up: it is where load_models_gpu makes the DiT and both "
                "VAE patchers resident and where the allocator grows its pools. The gate "
                "is therefore the last-to-last growth; the first-to-last total is a "
                "diagnostic"
            ),
        }
        checks.extend(
            growth_checks(
                plateau,
                label="plateaued once the process was warm",
                window="between run {} and run {}".format(len(runs) - 1, len(runs)),
            )
        )
        checks.extend(
            growth_checks(
                total,
                label="total growth over the whole series (diagnostic)",
                window="between run 1 and run {}, warm-up included".format(len(runs)),
                gate=False,
            )
        )
    else:
        growth = memory_growth(first.get("memory", {}), runs[1].get("memory", {}))
        summary["memory"] = {
            "mode": "strict",
            "gated": growth,
            "gate_window": "run 1 -> run 2",
            "note": (
                "two sampled runs offer no plateau to measure, so the warm-up growth is "
                "gated directly; --repeat 3 or more separates warm-up from a leak"
            ),
        }
        checks.extend(
            growth_checks(growth, label="did not grow materially", window="after run 2 vs run 1")
        )
    return summary, checks


# --------------------------------------------------------------------------
# one execution of the node
# --------------------------------------------------------------------------


def run_plan(
    *, repeat: int, cancel_after_forward: int, cancel_after_chunk: int = 0
) -> List[str]:
    """The runs this invocation performs, in order.

    The cancelled run goes **first** when one is asked for: the question it
    answers is not "can the node be interrupted" but "is the process still
    usable afterwards", and that can only be answered by a normal run that
    follows it in the same interpreter, sharing the same loaded weights, the
    same preview manager and the same allocator.

    The two cancellation points are mutually exclusive on the command line, so
    a plan carries at most one cancelled run: ``cancel`` interrupts a *forward*
    (nothing has been decoded yet at the chunk it stops), ``cancel_chunk``
    interrupts *after* a chunk has been through both collectors' decoders.
    """
    plan: List[str] = []
    if int(cancel_after_forward) > 0:
        plan.append("cancel")
    elif int(cancel_after_chunk) > 0:
        plan.append("cancel_chunk")
    plan.extend(["sample"] * int(repeat))
    return plan


def loaded_model_snapshot(model_management: Any) -> List[Dict[str, Any]]:
    """What Comfy still has resident, as Comfy reports it.

    ``model_management.loaded_models()`` is the public list of every
    ``ModelPatcher`` the loader currently considers loaded. Recording it after
    every run is what turns "memory went up" into an explanation: a run that
    ends with the DiT and both VAEs resident *should* be holding several tens
    of GB, and that is warm-up, not a leak.
    """
    listing = getattr(model_management, "loaded_models", None)
    if not callable(listing):
        return []
    snapshot: List[Dict[str, Any]] = []
    try:
        patchers = list(listing())
    except Exception:  # noqa: BLE001 - diagnostics only
        return []
    for patcher in patchers:
        entry: Dict[str, Any] = {
            "patcher": type(patcher).__name__,
            "model": type(getattr(patcher, "model", None)).__name__,
            "is_clip": bool(getattr(patcher, "is_clip", False)),
        }
        for name, key in (
            ("loaded_size", "loaded_size"),
            ("model_size", "model_size"),
            ("current_loaded_device", "current_device"),
        ):
            method = getattr(patcher, name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    value = method()
                    entry[key] = str(value) if key == "current_device" else int(value)
        snapshot.append(entry)
    return snapshot


@dataclass
class ProbeContext:
    """Everything built once and reused by every run in the process."""

    args: argparse.Namespace
    env: ComfyEnv
    device: torch.device
    geometry: Dict[str, int]
    conditioning: Any
    model: Any
    video_vae: Any
    audio_vae: Any
    manager: Any
    recorder: Any
    sampler: Any
    diffusion_model: Any
    #: How each run gets its own LATENT. ``None`` builds the empty AV latent
    #: from the geometry; the text lane passes a factory that copies the one
    #: the official ``MiniMaxH3ImageToVideo`` node produced.
    latent_factory: Optional[Any] = None


def execute_run(context: ProbeContext, index: int, kind: str) -> Dict[str, Any]:
    """Run the sampler node once and check everything the run produced."""
    from raven_streaming.consistency import SamplingCancelled

    args = context.args
    checks = Checks()
    forward_cancelling = kind == "cancel"
    chunk_cancelling = kind == "cancel_chunk"
    cancelling = forward_cancelling or chunk_cancelling
    node_id = "{}-{}".format(PROBE_NODE_ID, index)

    message_offset = len(context.recorder.messages)
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(context.device)
    before = cuda_stats(context.device)
    rss_before = current_host_rss_bytes()

    if context.latent_factory is not None:
        latent = context.latent_factory()
    else:
        latent = build_empty_latent(
            context.geometry,
            context.env.comfy.nested_tensor.NestedTensor,
            device=context.env.comfy.model_management.intermediate_device(),
        )

    # The chunk count comes from the layout, not from the run: it is what the
    # phase-swap expectations below are measured against, so it has to be
    # derived independently of what the run says it did.
    text_len = 1
    with contextlib.suppress(Exception):
        text_len = int(context.conditioning[0][0].shape[1])
    num_chunks = layout_num_chunks(context.geometry, text_len)

    inst = Instrumentation()
    if chunk_cancelling:
        # Armed before the pipeline exists: ``_capture_pipeline`` installs the
        # wrapper when the node builds it, and reads the limit off here.
        inst.cancel_after_chunk = int(args.cancel_after_chunk)
    counter = ForwardCounter(limit=int(args.cancel_after_forward) if forward_cancelling else 0)
    outputs: Optional[Tuple[Any, Any, Any]] = None
    raised: Optional[BaseException] = None

    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        stack.enter_context(instrument(inst))
        if forward_cancelling:
            stack.enter_context(
                cancel_after_forwards(
                    context.diffusion_model, int(args.cancel_after_forward), counter
                )
            )
        else:
            # A chunk cancel counts forwards like a normal run: the interruption
            # comes from the collector side, and the forwards it did run are
            # part of the evidence that a real chunk was produced.
            stack.enter_context(count_forwards(context.diffusion_model, counter))
        stack.enter_context(
            watch_vae_decodes(
                inst.vae_decode_calls,
                video=context.video_vae,
                audio=context.audio_vae,
            )
        )
        try:
            outputs = context.sampler.sample(
                model=context.model,
                positive=context.conditioning,
                latent=latent,
                video_vae=context.video_vae,
                audio_vae=context.audio_vae,
                seed=int(args.seed),
                steps=int(args.steps),
                video_shift=float(args.video_shift),
                audio_shift=float(args.audio_shift),
                sink=int(args.sink),
                window=int(args.window),
                kv_cache_storage=str(args.kv_cache_storage),
                unique_id=node_id,
            )
        except BaseException as exc:  # noqa: BLE001 - the probe is the evidence
            raised = exc
    sample_seconds = time.perf_counter() - started

    # Re-read before anything is copied out of it: the node splices the
    # final-decode handover *and* the phase-swap record into the budget's
    # detail on its way out, so the budget is only complete once the call has
    # returned. Reading it at capture time would report the estimate and none
    # of what actually happened.
    if inst.budget_object is not None:
        with contextlib.suppress(Exception):
            inst.memory_budget = inst.budget_object.to_dict()

    payload: Dict[str, Any] = {
        "index": index,
        "kind": kind,
        "node_id": node_id,
        "timings": dict(inst.timings, sample_seconds=sample_seconds),
        "forward_chunk_calls": counter.calls,
        "dit_phase_loads": inst.load_calls,
        "num_chunks": num_chunks,
        "kv_cache_storage": str(args.kv_cache_storage),
        "memory_budget": dict(inst.memory_budget),
        "discard_pending_calls": inst.discard_pending_calls,
        "chunks_delivered": inst.chunks_delivered,
    }

    # -- what happened to the call itself --------------------------------
    if cancelling:
        if forward_cancelling:
            checks.expect(
                "cancel: the wrapper raised after {} forward(s)".format(
                    args.cancel_after_forward
                ),
                counter.raised,
                "{} forward_chunk call(s) ran".format(counter.calls),
            )
        else:
            checks.expect(
                "cancel: the wrapper raised after {} delivered chunk(s)".format(
                    args.cancel_after_chunk
                ),
                inst.chunk_cancel_raised,
                "{} chunk(s) reached pipeline.on_chunk, {} forward_chunk call(s) "
                "ran".format(inst.chunks_delivered, counter.calls),
            )
        checks.expect(
            "cancel: sample() propagated SamplingCancelled",
            isinstance(raised, SamplingCancelled),
            "raised {}".format(type(raised).__name__ if raised is not None else None),
        )
        checks.expect(
            "cancel: no partial output was returned",
            outputs is None,
            "outputs: {}".format(type(outputs).__name__),
        )
        checks.expect(
            "cancel: the staged KV rows were discarded",
            inst.discard_pending_calls >= 1,
            "discard_pending called {} time(s)".format(inst.discard_pending_calls),
        )
        checks.expect(
            "cancel: the DiT phase was entered at least once",
            inst.load_calls >= 1,
            "the rollout loads the DiT before its first forward; {} call(s) seen".format(
                inst.load_calls
            ),
        )
        payload["cancelled"] = True
        payload["exception"] = (
            "{}: {}".format(type(raised).__name__, raised) if raised is not None else None
        )
    elif raised is not None:
        payload["exception"] = "{}: {}".format(type(raised).__name__, raised)
        payload["traceback"] = "".join(
            traceback.format_exception(type(raised), raised, raised.__traceback__)
        )
        checks.fail("run: sample() completed", payload["exception"])
    else:
        checks.expect("run: sample() completed", True, "{:.2f}s".format(sample_seconds))
        checks.expect(
            "run: sample() returned LATENT/IMAGE/AUDIO",
            isinstance(outputs, tuple) and len(outputs) == 3,
            "returned {}".format(type(outputs).__name__),
        )
        expected_loads = expected_dit_loads(num_chunks)
        checks.expect(
            "run: the DiT phase was entered once per chunk",
            inst.load_calls == expected_loads,
            "the DiT closure was called {} time(s); the phase swap makes {} for {} "
            "chunk(s) (one before the first forward, one after each non-last "
            "chunk). On a card that holds everything at once the calls still "
            "happen and upstream simply moves nothing.".format(
                inst.load_calls, expected_loads, num_chunks
            ),
        )

    # Both outputs come out of the streaming collectors now. A whole-clip
    # decode of either stream during the run would be one of the two
    # allocations that OOMed on real hardware -- video at 39 frames on a 141
    # GiB card, audio at 192 frames on a 24 GiB one -- re-deriving data the
    # lanes already hold.
    checks.expect(
        "run: the node called no whole-clip decode helper",
        inst.decode_images_calls == 0 and inst.decode_audio_calls == 0,
        "decode_images x{}, decode_audio x{}; both outputs are supposed to come "
        "from finalize_image() / finalize_audio()".format(
            inst.decode_images_calls, inst.decode_audio_calls
        ),
    )
    checks.expect(
        "run: neither VAE's whole-clip decode() was called",
        not any(int(n) for n in inst.vae_decode_calls.values()),
        "VAE.decode calls: {} (the collectors drive the inner modules directly, so "
        "any call here is a whole-clip decode)".format(dict(inst.vae_decode_calls)),
    )
    if not cancelling and raised is None:
        checks.expect(
            "run: each collector handed over exactly once",
            inst.finalize_image_calls == 1 and inst.finalize_audio_calls == 1,
            "finalize_image x{}, finalize_audio x{}".format(
                inst.finalize_image_calls, inst.finalize_audio_calls
            ),
        )

    # -- the node's own accounting ---------------------------------------
    if inst.memory_budget:
        detail = inst.memory_budget.get("detail", {})
        checks.note(
            "budget: itemised rollout reserve",
            "total {} = KV {} + buffers {} + max(forward {}, decode {}) + safety {} "
            "[{} peak KV rows]".format(
                _gib(inst.memory_budget.get("total_bytes", 0)),
                _gib(inst.memory_budget.get("kv_cache_bytes", 0)),
                _gib(inst.memory_budget.get("rollout_buffer_bytes", 0)),
                _gib(inst.memory_budget.get("forward_workspace_bytes", 0)),
                _gib(inst.memory_budget.get("decode_workspace_bytes", 0)),
                _gib(inst.memory_budget.get("safety_bytes", 0)),
                detail.get("kv_peak_rows"),
            ),
        )
        checks.expect(
            "budget: the DiT's shape was measured, not assumed",
            "num_layers" in (detail.get("measured") or []),
            "measured: {}".format(detail.get("measured")),
            gate=False,
        )
    else:
        checks.fail("budget: the node priced the rollout", "no budget was captured", gate=False)

    if inst.pipeline is not None:
        with contextlib.suppress(Exception):
            inst.pipeline_report = inst.pipeline.report().to_dict()
        # Straight off the object, not inferred: the pipeline is the only thing
        # that knows whether its preview half survived the run.
        inst.preview_disabled = bool(getattr(inst.pipeline, "preview_disabled", False))
        inst.preview_disabled_reason = str(
            getattr(inst.pipeline, "preview_disabled_reason", "") or ""
        )
    payload["pipeline"] = dict(inst.pipeline_report)
    payload["preview_disabled"] = inst.preview_disabled
    payload["preview_disabled_reason"] = inst.preview_disabled_reason

    # -- the chunk cancellation: decoded, then dropped --------------------
    if chunk_cancelling:
        before_state = dict(inst.state_before_cancel)
        after_state = pipeline_state(inst.pipeline) if inst.pipeline is not None else {}
        payload["chunk_cancel"] = {
            "requested_after_chunks": int(args.cancel_after_chunk),
            "chunks_delivered": inst.chunks_delivered,
            "raised": inst.chunk_cancel_raised,
            "state_before": before_state,
            "state_after": after_state,
            "decoder_aborts": dict(inst.decoder_aborts),
            "decoder_abort_available": dict(inst.decoder_abort_available),
        }
        decoders_before = before_state.get("decoders") or {}
        decoded_before = int(before_state.get("frames_decoded", 0)) + int(
            before_state.get("samples_decoded", 0)
        )
        vae_calls_before = sum(
            int(entry.get("vae_decode_calls", 0) or 0) for entry in decoders_before.values()
        )
        checks.expect(
            "cancel: the delivered chunk(s) reached the decode lane",
            inst.chunks_delivered >= int(args.cancel_after_chunk)
            and (decoded_before > 0 or vae_calls_before > 0
                 or bool(before_state.get("holds_decoder_buffers"))),
            "{} chunk(s) delivered; {} frame(s) / {} sample(s) decoded, {} VAE decode "
            "call(s), decoder buffers held: {}".format(
                inst.chunks_delivered,
                before_state.get("frames_decoded"),
                before_state.get("samples_decoded"),
                vae_calls_before,
                before_state.get("holds_decoder_buffers"),
            ),
        )
        # "Went through the VAE" is derived, not assumed. The video coordinator
        # buffers what it cannot decide yet, so a chunk that is entirely still
        # in ``pending_latents`` has legitimately not been decoded -- that is
        # the lookahead the streaming lane is built on, and at
        # ``--cancel-after-chunk 1`` it is the expected state. Once anything has
        # been consumed *past* that buffer, a decode is due, and then it must
        # have happened.
        consumed = int((decoders_before.get("video") or {}).get("consumed_latents", 0))
        video_calls = int((decoders_before.get("video") or {}).get("vae_decode_calls", 0) or 0)
        checks.expect(
            "cancel: every latent the video decoder consumed had really been decoded",
            consumed == 0 or (video_calls >= 1 and int(before_state.get("frames_decoded", 0)) > 0),
            "{} latent(s) consumed past the lookahead buffer, {} VAE decode call(s), "
            "{} frame(s) decoded".format(
                consumed, video_calls, before_state.get("frames_decoded")
            )
            + (
                "; nothing was due yet at --cancel-after-chunk {} (the whole chunk is "
                "still in the decoder's lookahead buffer)".format(args.cancel_after_chunk)
                if consumed == 0
                else ""
            ),
        )
        checks.expect(
            "cancel: the decoders were holding state when it landed",
            bool(before_state.get("holds_decoder_buffers"))
            or int(before_state.get("pending_frames", 0)) > 0
            or int(before_state.get("samples_available", 0)) > 0,
            "pending_frames={} samples_available={} decoder buffers={}".format(
                before_state.get("pending_frames"),
                before_state.get("samples_available"),
                {
                    lane: entry.get("buffers_held")
                    for lane, entry in (before_state.get("decoders") or {}).items()
                },
            ),
        )
        checks.expect(
            "cancel: the pipeline held nothing afterwards",
            int(after_state.get("pending_frames", 0)) == 0
            and int(after_state.get("samples_available", 0)) == 0
            and int(after_state.get("image_bytes", 0)) == 0
            and int(after_state.get("audio_bytes", 0)) == 0
            and not after_state.get("holds_decoder_buffers"),
            "pending_frames={} samples_available={} image_bytes={} audio_bytes={} "
            "decoder buffers={}".format(
                after_state.get("pending_frames"),
                after_state.get("samples_available"),
                after_state.get("image_bytes"),
                after_state.get("audio_bytes"),
                {
                    lane: entry.get("buffers_held")
                    for lane, entry in (after_state.get("decoders") or {}).items()
                },
            ),
        )
        # The abort surface belongs to the runtime, so its absence is reported
        # rather than failed: what is gated above is the *effect* (nothing is
        # still held), which is true whether the release came from abort() or
        # from dropping the last reference.
        for lane in sorted(inst.decoder_abort_available):
            if inst.decoder_abort_available.get(lane):
                checks.expect(
                    "cancel: the {} decoder was aborted".format(lane),
                    int(inst.decoder_aborts.get(lane, 0)) >= 1,
                    "abort() called {} time(s)".format(inst.decoder_aborts.get(lane, 0)),
                )
            else:
                checks.note(
                    "cancel: the {} decoder has no abort()".format(lane),
                    "this runtime releases that lane by dereferencing it; the "
                    "'held nothing afterwards' gate still applies",
                )

    # -- the phase swap: what the node says it did, against what we counted --
    delivered = int(inst.pipeline_report.get("chunks", 0)) if inst.pipeline_report else 0
    phase_record = {
        key: value
        for key, value in (inst.memory_budget.get("detail", {}) or {}).items()
        if key.startswith("phase_swap")
    }
    payload["phase_swap"] = dict(
        phase_record,
        observed_dit_loads=inst.load_calls,
        delivered_chunks=delivered,
        layout_chunks=num_chunks,
    )
    if cancelling:
        derived = (
            dit_loads_after_chunk_cancel(delivered, num_chunks)
            if chunk_cancelling
            else dit_loads_after_cancel(delivered, num_chunks)
        )
        checks.expect(
            "cancel: the DiT phase count matches the chunks that finished",
            inst.load_calls == derived,
            "{} call(s) for {} delivered chunk(s) of {}; derived expectation is "
            "{} ({})".format(
                inst.load_calls,
                delivered,
                num_chunks,
                derived,
                "one before the first forward, one after each delivered chunk "
                "except the one the cancel interrupted"
                if chunk_cancelling
                else "one before the first forward, one after each delivered "
                "non-last chunk",
            ),
        )
    else:
        checks.expect(
            "run: the rollout delivered every chunk",
            delivered == num_chunks,
            "{} chunk(s) reached the pipeline, the layout says {}".format(
                delivered, num_chunks
            ),
        )
    if phase_record and not cancelling:
        # The node's own record of the swap, cross-checked against the layout
        # and against the calls this probe counted. Two independent accounts of
        # the same run disagreeing is a finding in itself.
        checks.expect(
            "phase swap: the node's record agrees with the layout and the count",
            int(phase_record.get("phase_swap_chunks", -1)) == num_chunks
            and int(phase_record.get("phase_swap_dit_loads", -1)) == num_chunks - 1
            and int(phase_record.get("phase_swap_dit_loads", -1)) + 1 == inst.load_calls,
            "{} vs {} layout chunk(s) and {} counted DiT call(s)".format(
                phase_record, num_chunks, inst.load_calls
            ),
        )
        checks.expect(
            "phase swap: the run ended in the VAE phase",
            str(phase_record.get("phase_swap_last_phase")) == "vae",
            "last phase {!r}; the last chunk leaves the VAEs loaded for the "
            "video/audio collector tail flush -- finalize_image/finalize_audio "
            "only normalise and return the host buffers it filled".format(
                phase_record.get("phase_swap_last_phase")
            ),
        )
    elif phase_record:
        # A cancelled run stops before the node writes its record, so one being
        # here at all is new behaviour. Only the invariant that holds wherever
        # a cancellation landed is checked: every reload the coordinator made,
        # plus the rollout's own first load, is a call this probe counted.
        checks.expect(
            "cancel: the phase record accounts for every DiT call counted",
            int(phase_record.get("phase_swap_dit_loads", -1)) + 1 == inst.load_calls,
            "{} vs {} counted DiT call(s)".format(phase_record, inst.load_calls),
        )
        checks.note(
            "phase swap: VAE phase loads",
            "{} VAE phase load(s), {} of them with the audio VAE".format(
                phase_record.get("phase_swap_vae_loads"),
                phase_record.get("phase_swap_audio_vae_loads"),
            ),
        )
    elif not cancelling:
        checks.note(
            "phase swap: the node recorded no phase-swap detail",
            "nothing under 'phase_swap_*' in the budget detail; only the calls this "
            "probe counted are available",
        )

    if inst.pipeline_report:
        report = inst.pipeline_report
        # -- the collectors: these *are* the IMAGE and AUDIO outputs -------
        collector_checks = check_collector_report(
            report, geometry=context.geometry, cancelled=cancelling
        )
        checks.extend(collector_checks)
        if not cancelling:
            cadence, cadence_checks = check_streaming_cadence(
                report,
                num_chunks=num_chunks,
                preview_disabled=inst.preview_disabled,
            )
            payload["streaming"] = cadence
            checks.extend(cadence_checks)

            # The node copies the emission log and the pre-flush fragment count
            # into its own memory record. Two accounts of the same cadence, so
            # they have to agree -- and if the node ever stops writing them,
            # this is where that shows up rather than in a silently thinner
            # report.
            detail = inst.memory_budget.get("detail", {}) or {}
            if "emission_log" in detail or "fragments_before_finish" in detail:
                logged = list(detail.get("emission_log", []) or [])
                cadence["node_emission_log_entries"] = len(logged)
                cadence["node_fragments_before_finish"] = detail.get(
                    "fragments_before_finish"
                )
                checks.expect(
                    "streaming: the node's own emission record matches the report",
                    len(logged) == len(cadence["emission_log"])
                    and int(detail.get("fragments_before_finish", -1))
                    == int(report.get("fragments_before_finish", -2)),
                    "node recorded {} emission(s) / {} fragment(s) before finish; the "
                    "pipeline report says {} / {}".format(
                        len(logged),
                        detail.get("fragments_before_finish"),
                        len(cadence["emission_log"]),
                        report.get("fragments_before_finish"),
                    ),
                )
        checks.expect(
            "preview: the media lane stayed enabled",
            (not inst.preview_disabled) if not cancelling else True,
            "preview_disabled={} reason={!r}".format(
                inst.preview_disabled, inst.preview_disabled_reason
            ),
        )
        checks.note(
            "pipeline: what was streamed",
            "{} chunk(s), {} frame(s) muxed, {} sample(s) muxed, {} fragment(s) / {} B, "
            "first fragment after chunk {}".format(
                report.get("chunks"),
                report.get("frames_muxed"),
                report.get("samples_muxed"),
                report.get("fragments"),
                report.get("fragment_bytes"),
                report.get("first_fragment_chunk"),
            ),
        )
        checks.expect(
            "pipeline: no preview send failed",
            int(report.get("send_failures", 0)) == 0 and int(report.get("errors", 0)) == 0,
            "send_failures={} errors={}".format(
                report.get("send_failures"), report.get("errors")
            ),
            gate=False,
        )

    # -- the preview envelopes -------------------------------------------
    messages = list(context.recorder.messages[message_offset:])
    preview_summary, preview_checks, media_blob = check_preview_messages(
        messages,
        node_id=node_id,
        expect_end="cancelled" if cancelling else "complete",
        expect_media=not cancelling,
        width=int(args.width),
        height=int(args.height),
    )
    payload["preview"] = preview_summary
    checks.extend(preview_checks)

    if not cancelling:
        media_summary, media_checks = decode_preview_media(
            media_blob, expect_frames=int(args.frames)
        )
        payload["media"] = media_summary
        checks.extend(media_checks)
    else:
        payload["media"] = {"skipped": "a cancelled run has no stream to play"}

    # -- session lifetime -------------------------------------------------
    active = context.manager.active_sessions
    checks.expect(
        "session: no preview session is left active",
        not active,
        "{} active session(s): {}".format(len(active), [s.session_id for s in active]),
    )
    payload["sessions"] = {
        "active": len(active),
        "terminal": len(context.manager.terminal_sessions),
        "cleanups": int(getattr(context.manager, "cleanups", 0)),
    }

    # -- outputs ----------------------------------------------------------
    artifacts: Dict[str, Any] = {}
    official_compare: Optional[Dict[str, Any]] = None
    official_checks: List[Check] = []
    if outputs is not None:
        latent_out, images, audio_out = outputs
        output_summary, output_checks, artifacts = check_outputs(
            latent_out, images, audio_out, geometry=context.geometry
        )
        payload["outputs"] = output_summary
        checks.extend(output_checks)

        # -- optional diagnostic, after the node has returned --------------
        if getattr(args, "compare_official_video", False) and not cancelling:
            from raven_streaming import nodes as nodes_mod

            official_compare, official_checks = compare_official_video(
                video_vae=context.video_vae,
                latent=latent_out,
                images=images,
                geometry=context.geometry,
                device=context.device,
                # The kept diagnostic helper, called explicitly and outside the
                # instrumented window, so the "the node never calls it" gate
                # above still means what it says.
                decode_images=nodes_mod.decode_images,
            )
        del latent_out, images, audio_out
    del outputs

    # -- memory -----------------------------------------------------------
    after = cuda_stats(context.device)
    memory: Dict[str, Any] = {
        "rss_before": rss_before,
        "rss_after": current_host_rss_bytes(),
        "rss_peak": peak_host_rss_bytes(),
    }
    if after:
        memory.update(
            {
                "cuda_allocated_before": before.get("allocated", 0),
                "cuda_allocated_after": after.get("allocated", 0),
                "cuda_reserved_before": before.get("reserved", 0),
                "cuda_reserved_after": after.get("reserved", 0),
                "cuda_peak_allocated": after.get("peak_allocated", 0),
                "cuda_peak_reserved": after.get("peak_reserved", 0),
            }
        )
    payload["memory"] = memory
    if official_compare is not None:
        # Recorded after the run's own memory, so the diagnostic's peak can
        # never be mistaken for the product path's.
        payload["official_compare"] = official_compare
        checks.extend(official_checks)

    # -- what Comfy still holds, so growth can be explained ---------------
    resident = loaded_model_snapshot(getattr(context.env.comfy, "model_management", None))
    payload["loaded_models"] = resident
    if resident:
        checks.note(
            "memory: models resident at the end of the run",
            "; ".join(
                "{}({}) {} on {}".format(
                    entry["patcher"],
                    entry["model"],
                    _gib(entry.get("loaded_size", 0)),
                    entry.get("current_device"),
                )
                for entry in resident
            ),
        )

    payload["checks"] = checks.to_list()
    payload["ok"] = checks.ok
    payload["artifacts"] = artifacts
    payload["_checks"] = checks.items
    return payload


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def run_probe(args: argparse.Namespace, report: Report) -> Report:
    """Fill ``report`` in place, so a crash still leaves everything measured.

    Every observation is attached to ``report`` as soon as it is made, and the
    check list is flushed in a ``finally``: a probe that dies loading the model
    must still be able to say what it had already established about the
    environment and the request.
    """
    checks = Checks()
    try:
        return _run_probe(args, report, checks)
    finally:
        report.checks = checks.to_list()


def text_lane_requested(args: argparse.Namespace) -> bool:
    """Is the optional real text lane switched on?

    Both flags or neither: ``--text-encoder`` without ``--prompt`` has nothing
    to encode, and ``--prompt`` without an encoder has nothing to encode it
    with. :func:`main` refuses the half-configured pair rather than quietly
    falling back to the synthetic lane, which would produce a report that looks
    like it verified text and did not.
    """
    return bool(getattr(args, "text_encoder", None)) and bool(getattr(args, "prompt", None))


def _run_probe(args: argparse.Namespace, report: Report, checks: Checks) -> Report:
    use_text_lane = text_lane_requested(args)
    use_stacked_lora = stacked_lora_requested(args)
    if use_stacked_lora:
        report.notes.append(
            "stacked lora lane: upstream's own nodes.{} is instantiated and its "
            "FUNCTION is called on the RAVEN MODEL with the folder_paths '{}' name "
            "'{}' at strength {}; comfy.sd.load_lora_for_models is counted, never "
            "called directly".format(
                OFFICIAL_LORA_NODE,
                LORA_FOLDER,
                getattr(args, "stacked_lora_name", None),
                stacked_lora_strength(args),
            )
        )
    if use_text_lane:
        report.notes.append(
            "real text lane: the official CLIPLoader loads the H3 text encoder and the "
            "official MiniMaxH3ImageToVideo (T2VA form) produces the CONDITIONING and "
            "the empty AV LATENT; nothing about the prompt is synthesised"
        )
    else:
        report.notes.append(
            "no text encoder is loaded: the CONDITIONING is synthetic and deterministic, "
            "so this probe verifies the DiT / VAE / media / node lane only"
        )
    report.notes.append("nothing is downloaded; every model path must already exist")
    report.notes.append(
        "audio is measured at two different points and they are not the same signal: "
        "the preview carries the collector's RAW chunk-wise PCM (it has to -- the "
        "normalisation is a whole-clip statistic that does not exist yet while the "
        "clip is still being made), while the AUDIO output is that same buffer after "
        "finalize_audio() applies vae_decode_audio's tail (/= max(1, std*5)) in "
        "place. A sample-for-sample comparison between what was streamed and what "
        "was returned is therefore expected to differ by exactly that divisor"
    )

    # -- environment ------------------------------------------------------
    comfy_root = resolve_comfy_root(args.comfy_root)
    env = import_comfy(comfy_root)
    device = torch.device(env.comfy.model_management.get_torch_device())
    report.environment = {
        "comfy_root": str(env.root),
        "comfy_commit": env.commit,
        "comfy_version": env.version,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "requested_device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            report.environment["cuda_device_name"] = torch.cuda.get_device_name(device)
    checks.expect(
        "environment: ComfyUI picked the requested device",
        device.type == torch.device(args.device).type,
        "comfy uses {}, --device asked for {}".format(device, args.device),
    )

    from raven_streaming import nodes as nodes_mod
    from raven_streaming import preview as preview_mod

    # -- the KV cache lane, against the node's own live schema -------------
    #
    # ``--kv-cache-storage`` is offered by a parser that must stay importable
    # without the package, so its choices are a copy. A copy that has drifted
    # would be rejected inside ``sample()`` after the 66 GB build, which is the
    # most expensive possible place to learn it.
    node_choices = tuple(getattr(nodes_mod, "KV_CACHE_STORAGE_CHOICES", ()) or ())
    node_default = getattr(nodes_mod, "DEFAULT_KV_CACHE_STORAGE", None)
    checks.expect(
        "kv cache: the probe offers exactly the node's storage modes",
        set(node_choices) == set(KV_CACHE_STORAGE_CHOICES)
        and str(node_default) == DEFAULT_KV_CACHE_STORAGE,
        "node offers {} (default {!r}); the probe offers {} (default {!r})".format(
            list(node_choices), node_default, list(KV_CACHE_STORAGE_CHOICES),
            DEFAULT_KV_CACHE_STORAGE,
        ),
    )
    report.environment["kv_cache_storage"] = str(args.kv_cache_storage)
    checks.note(
        "kv cache: the run's storage mode",
        "{!r} (node default {!r}); it is passed to sample() and recorded, so a "
        "GPU-vs-CPU pair of reports differs only here".format(
            str(args.kv_cache_storage), node_default
        ),
    )

    # -- inputs -----------------------------------------------------------
    geometry = latent_geometry(int(args.frames), int(args.width), int(args.height))
    report.inputs = {
        "lane": "text-encoder" if use_text_lane else "synthetic",
        "geometry": geometry,
        "rollout_rng": describe_rollout_rng(int(args.seed), device),
        "base": str(Path(args.base).expanduser()),
        "lora": str(Path(args.lora).expanduser()),
        "video_vae": str(Path(args.video_vae).expanduser()),
        "audio_vae": str(Path(args.audio_vae).expanduser()),
    }
    check_geometry_against_upstream(geometry, checks)

    # What the setup builds, filled in as it happens and attached to the report
    # **before** anything writes to it. Both lanes and both loaders share this
    # one dict, and a probe that dies half way through the setup still reports
    # the half it got.
    #
    # This is the bug a real vr run found: the dict used to be created inside
    # the DiT block, and reordering that block to run *after* the VAEs left the
    # VAE block writing into a name that did not exist yet -- a NameError
    # before a single weight was read.
    setup: Dict[str, Any] = {}
    report.setup = setup

    required_paths = [
        ("base", args.base),
        ("lora", args.lora),
        ("video_vae", args.video_vae),
        ("audio_vae", args.audio_vae),
    ]
    if use_text_lane:
        required_paths.append(("text_encoder", args.text_encoder))
        report.inputs["text_encoder"] = str(Path(args.text_encoder).expanduser())
    for label, value in required_paths:
        path = Path(os.path.expanduser(str(value)))
        if not path.is_file():
            raise ProbeError("--{} does not exist: {} (nothing is downloaded)".format(label.replace("_", "-"), path))

    # -- the two VAE sockets, through upstream's own VAELoader -------------
    #
    # Before the DiT on purpose: the text lane needs the *loaded* video VAE
    # (the official T2VA node takes a VAE socket), and the encoder has to be
    # loaded, used and offloaded before 66 GB of DiT arrives -- which is the
    # memory story docs/requirements.md describes.
    video_name, audio_name = register_model_files(
        env.folder_paths,
        VAE_FOLDER,
        [Path(os.path.expanduser(args.video_vae)), Path(os.path.expanduser(args.audio_vae))],
    )
    vae_loader = env.upstream_nodes.VAELoader()
    started = time.perf_counter()
    (video_vae,) = vae_loader.load_vae(video_name)
    setup["video_vae_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    (audio_vae,) = vae_loader.load_vae(audio_name)
    setup["audio_vae_seconds"] = time.perf_counter() - started
    setup["video_vae_name"] = video_name
    setup["audio_vae_name"] = audio_name
    setup["video_vae_class"] = type(video_vae.first_stage_model).__name__
    setup["audio_vae_class"] = type(audio_vae.first_stage_model).__name__

    comfy_vae_class = getattr(env.comfy.sd, "VAE", None)
    checks.expect(
        "vae: both sockets carry a real comfy.sd.VAE from VAELoader",
        comfy_vae_class is not None
        and isinstance(video_vae, comfy_vae_class)
        and isinstance(audio_vae, comfy_vae_class),
        "video={} audio={}".format(type(video_vae).__name__, type(audio_vae).__name__),
    )
    # The node's own feature probes, run here so a swapped or wrong VAE is
    # reported as *the node's* diagnosis ("the two VAE sockets are swapped")
    # rather than as a probe traceback. There is nothing to sample afterwards,
    # so this is where the probe stops.
    try:
        resolved_video = nodes_mod.resolve_video_vae(video_vae)
        resolved_audio = nodes_mod.resolve_audio_vae(audio_vae)
    except nodes_mod.NodeInputError as exc:
        checks.fail(
            "vae: the node accepted both sockets", "{}: {}".format(type(exc).__name__, exc)
        )
        return report
    checks.expect(
        "vae: the node accepted both sockets",
        resolved_video.kind == "video" and resolved_audio.kind == "audio",
        "video inner={} audio inner={}".format(
            setup["video_vae_class"], setup["audio_vae_class"]
        ),
    )

    # -- conditioning + latent: the real text lane, or the synthetic one ----
    latent_factory: Optional[Any] = None
    if use_text_lane:
        clip_loader_cls = getattr(env.upstream_nodes, "CLIPLoader", None)
        minimax_nodes = env.minimax_nodes
        image_to_video_cls = getattr(minimax_nodes, "MiniMaxH3ImageToVideo", None)
        if clip_loader_cls is None or image_to_video_cls is None:
            raise ProbeError(
                "this ComfyUI has no {}; the real text lane cannot run against it "
                "and the probe will not substitute its own encoder".format(
                    "CLIPLoader" if clip_loader_cls is None else "MiniMaxH3ImageToVideo"
                )
            )
        conditioning, official_latent, text_summary = run_text_lane(
            clip_loader_cls=clip_loader_cls,
            image_to_video_cls=image_to_video_cls,
            folder_paths=env.folder_paths,
            model_management=env.comfy.model_management,
            text_encoder_path=Path(os.path.expanduser(args.text_encoder)).resolve(),
            prompt=args.prompt,
            geometry=geometry,
            video_vae=video_vae,
            device=device,
            checks=checks,
        )
        report.inputs["text_lane"] = text_summary
        report.inputs["text_len"] = text_summary.get("text_len")
        nested_cls = type(official_latent["samples"])
        latent_factory = lambda: clone_latent(official_latent, nested_cls)  # noqa: E731
        report.notes.append(
            "--text-len is ignored in the text lane: the prompt's own token count "
            "decides L"
        )
        # ``run_text_lane`` kept no reference to the CLIP, so once the encode is
        # done the encoder's host memory is garbage. Collecting it *here* rather
        # than whenever the collector feels like it is what keeps the 66 GB DiT
        # build from starting on top of it.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report.inputs["text_lane"]["released_after_encode"] = True
        report.inputs["text_lane"]["host_rss_after_release"] = current_host_rss_bytes()
    else:
        conditioning, conditioning_describe = build_conditioning(
            int(args.text_len), int(args.seed)
        )
        report.inputs["text_len"] = int(args.text_len)
        report.inputs["conditioning"] = conditioning_describe

    # -- the loader node ---------------------------------------------------
    #
    # Last of the three loads, and that ordering is the whole memory story: the
    # two VAEs are small and the text encoder has already been used and
    # offloaded, so the 66 GB DiT arrives into a machine that is not still
    # holding an encoder it will never call again.
    started = time.perf_counter()
    (model,) = nodes_mod.RAVENModelLoader().load_model(
        str(Path(os.path.expanduser(args.base)).resolve()),
        str(Path(os.path.expanduser(args.lora)).resolve()),
        args.weight_dtype,
    )
    model_seconds = time.perf_counter() - started

    diffusion_model = model.model.diffusion_model
    manifest = getattr(model.model, "raven_lora_manifest", None)
    attachment = getattr(model.model, "raven_lora_attachment", None)
    setup.update(
        {
            "model_build_seconds": model_seconds,
            "patcher_class": type(model).__name__,
            "model_class": type(model.model).__name__,
            "diffusion_model_class": type(diffusion_model).__name__,
            "model_size_bytes": int(model.model_size()),
            "load_device": str(getattr(model, "load_device", "?")),
            "offload_device": str(getattr(model, "offload_device", "?")),
            "lora_modules": len(getattr(attachment, "entries", []) or []),
            "lora_manifest": getattr(manifest, "name", None),
            "weight_dtype": str(args.weight_dtype),
        }
    )
    checks.expect(
        "loader: the node returned a chunk-causal DiT",
        type(diffusion_model).__name__ == "RavenCausalMiniMaxH3Model",
        "diffusion_model is {}".format(type(diffusion_model).__name__),
    )
    checks.expect(
        "loader: the RAVEN residual is attached",
        setup["lora_modules"] > 0,
        "{} module(s) carry the residual".format(setup["lora_modules"]),
    )
    checks.note(
        "loader: built in {:.2f}s".format(model_seconds),
        "model_size={} patcher={}".format(
            _gib(setup["model_size_bytes"]), setup["patcher_class"]
        ),
    )

    # -- the optional stacked standard LoRA --------------------------------
    #
    # After the RAVEN loader and before anything samples, which is exactly
    # where a workflow puts it: MODEL -> LoraLoaderModelOnly -> sampler. From
    # here on ``model`` is the node's own clone, so every run below samples
    # through the stacked patches rather than around them.
    stacked_record: Optional[Dict[str, Any]] = None
    #: the mandatory attachment as it was *before* the runs, kept by identity so
    #: the post-run check compares two different observations rather than one
    #: object with itself
    attachment_before_runs = attachment
    if use_stacked_lora:
        model, stacked_record = stack_official_lora(
            env=env,
            model=model,
            lora_name=str(args.stacked_lora_name),
            strength=stacked_lora_strength(args),
            checks=checks,
        )
        setup["stacked_lora"] = stacked_record
        attachment_before_runs = getattr(
            getattr(model, "model", None), "raven_lora_attachment", None
        )

    # -- preview transport ------------------------------------------------
    manager = preview_mod.default_manager()
    recorder = preview_mod.RecordingSender()
    previous_sender = manager.sender
    manager.set_sender(recorder)
    report.notes.append(
        "preview transport: PreviewManager.set_sender(RecordingSender) on the default "
        "manager the node itself resolves through preview.install()"
    )

    sampler = nodes_mod.RAVENStreamingSampler()
    context = ProbeContext(
        args=args,
        env=env,
        device=device,
        geometry=geometry,
        conditioning=conditioning,
        model=model,
        video_vae=video_vae,
        audio_vae=audio_vae,
        manager=manager,
        recorder=recorder,
        sampler=sampler,
        diffusion_model=diffusion_model,
        latent_factory=latent_factory,
    )

    # -- the runs ---------------------------------------------------------
    plan = run_plan(
        repeat=int(args.repeat),
        cancel_after_forward=int(args.cancel_after_forward),
        cancel_after_chunk=int(getattr(args, "cancel_after_chunk", 0)),
    )
    report.args["plan"] = list(plan)

    completed: List[Dict[str, Any]] = []
    try:
        for index, kind in enumerate(plan, start=1):
            payload = execute_run(context, index, kind)
            checks.extend(prefixed(payload.pop("_checks"), "run {}".format(index)))
            artifacts = payload.pop("artifacts", {})
            completed.append({"payload": payload, "artifacts": artifacts, "kind": kind})
            report.runs.append(payload)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        sampled = [
            {
                "artifacts": entry["artifacts"],
                "memory": entry["payload"].get("memory", {}),
                "index": entry["payload"].get("index"),
            }
            for entry in completed
            if entry["kind"] == "sample"
        ]
        series, series_checks = compare_run_series(sampled)
        report.determinism = series
        checks.extend(series_checks)
        if len(sampled) < int(args.repeat):
            checks.fail(
                "determinism: every requested run finished",
                "{} of {} sampled run(s) completed".format(len(sampled), int(args.repeat)),
            )

        if "cancel" in plan or "cancel_chunk" in plan:
            after_cancel = [e for e in completed if e["kind"] == "sample"]
            checks.expect(
                "cancel: a normal run succeeded afterwards in the same process",
                bool(after_cancel) and bool(after_cancel[0]["payload"].get("ok")),
                "{} run(s) followed the {} one".format(
                    len(after_cancel),
                    "chunk-cancelled" if "cancel_chunk" in plan else "cancelled",
                ),
            )

        if stacked_record is not None:
            # The runs are over; the stacked patcher is the one they sampled
            # through. A residual that survived the *load* but not the rollout
            # would otherwise be invisible, and so would a patch dict that
            # sampling quietly emptied.
            after_inner = getattr(model, "model", None)
            after_attachment = getattr(after_inner, "raven_lora_attachment", None)
            after_counts = patch_key_counts(model)
            stacked_record["patch_keys_after_runs"] = len(after_counts)
            stacked_record["attachment_modules_after_runs"] = len(
                getattr(after_attachment, "entries", []) or []
            )
            stacked_record["attachment_same_object_after_runs"] = (
                after_attachment is not None and after_attachment is attachment_before_runs
            )
            checks.expect(
                "stacked lora: the patches and the RAVEN residual survived the run(s)",
                len(after_counts) == int(stacked_record.get("patch_keys_after", -1))
                and bool(stacked_record["attachment_same_object_after_runs"])
                and int(stacked_record["attachment_modules_after_runs"])
                == int(stacked_record.get("attachment_modules", -1)),
                "{} patched key(s) and {} residual module(s) after {} run(s)".format(
                    len(after_counts),
                    stacked_record["attachment_modules_after_runs"],
                    len(report.runs),
                ),
            )

        checks.expect(
            "session: the manager is empty when the probe finishes",
            not manager.active_sessions,
            "{} active session(s)".format(len(manager.active_sessions)),
        )
    finally:
        manager.set_sender(previous_sender)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def repeat_count(value: str) -> int:
    """``--repeat``: 1..10 sampled runs in one process.

    Bounded, not open-ended: every run is a full rollout plus a full decode, so
    an unbounded count is a way to queue hours of GPU time by typo. Ten is
    already far past the point where a leak would show up as a plateau.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--repeat must be an integer") from None
    if not 1 <= number <= MAX_REPEAT:
        raise argparse.ArgumentTypeError(
            "--repeat must be between 1 and {}, got {}".format(MAX_REPEAT, number)
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every path must be absolute and must already exist: this probe never "
            "downloads anything.\n\n"
            "By default no text encoder is loaded -- the CONDITIONING is synthetic and "
            "deterministic, so the result says nothing about the Qwen3-VL lane. "
            "--text-encoder + --prompt switch that on and make the run cost "
            "substantially more: the NVFP4 AWQ Qwen3-VL-32B file is another large "
            "checkpoint to read off disk and hold in host RAM, its encode runs on the "
            "GPU, and it is loaded, used and offloaded before the 66 GB DiT is built. "
            "Budget the extra load time and the extra host RAM on top of the >= 128 GB "
            "baseline.\n\n"
            "The report is written atomically to a path inside this repository, on "
            "success and on failure alike."
        ),
    )
    parser.add_argument("--comfy-root", default=None,
                        help="ComfyUI checkout (default: COMFYUI_PATH / COMFYUI_UPSTREAM_PATH "
                             "/ .cache/upstream/ComfyUI)")
    parser.add_argument("--base", required=True,
                        help="full non-pruned BF16 MiniMax H3 DiT (~66 GB)")
    parser.add_argument("--lora", required=True,
                        help="mandatory RAVEN PEFT LoRA (~5 GB FP32)")
    parser.add_argument("--video-vae", required=True, help="MiniMax H3 video VAE")
    parser.add_argument("--audio-vae", required=True, help="MiniMax H3 audio VAE")
    parser.add_argument("--text-encoder", default=None, metavar="PATH",
                        help="OPTIONAL: the H3 Qwen3-VL text encoder (NVFP4 AWQ). Given "
                             "together with --prompt it switches on the real text lane: "
                             "the official CLIPLoader + MiniMaxH3ImageToVideo produce the "
                             "CONDITIONING and the empty AV LATENT instead of the "
                             "synthetic stand-in. COSTS: a second large checkpoint read "
                             "and held in host RAM, a GPU encode, and minutes of extra "
                             "wall clock before sampling starts")
    parser.add_argument("--prompt", default=None, metavar="TEXT",
                        help="OPTIONAL: the prompt to encode. Requires --text-encoder; "
                             "the two are all-or-nothing. In the text lane --text-len is "
                             "ignored, because the prompt decides the token count")
    parser.add_argument("--stacked-lora-name", default=None, metavar="NAME",
                        help="OPTIONAL: stack a STANDARD Comfy LoRA on top of the RAVEN "
                             "MODEL by chaining upstream's own nodes.{} after the loader "
                             "node -- the node is instantiated and its own FUNCTION is "
                             "called, comfy.sd.load_lora_for_models is only counted. NAME "
                             "is a folder_paths '{}' RELATIVE name (what the combo in a "
                             "workflow holds, i.e. <ComfyUI>/models/loras/NAME), not a "
                             "path, and it must already exist. The run then samples "
                             "through the stacked MODEL and gates that a non-zero patch "
                             "landed on a base key, that no key went unmatched, that the "
                             "mandatory RAVEN residual is untouched and that the patcher "
                             "is still a stock ModelPatcher".format(
                                 OFFICIAL_LORA_NODE, LORA_FOLDER))
    parser.add_argument("--stacked-lora-strength", type=float, default=None, metavar="F",
                        help="strength_model for --stacked-lora-name (default: 1.0, the "
                             "node's own default). Requires --stacked-lora-name. A "
                             "strength of 0 is refused: the official node returns the "
                             "model untouched, so the report would claim a stacked LoRA "
                             "that patched nothing")
    parser.add_argument("--width", type=int, default=512,
                        help="canvas width, multiple of 32 (default: 512)")
    parser.add_argument("--height", type=int, default=288,
                        help="canvas height, multiple of 32 (default: 288)")
    parser.add_argument("--frames", type=int, default=39,
                        help="frame count on the 17k+5 grid (default: 39 = k 2)")
    parser.add_argument("--text-len", type=int, default=128,
                        help="synthetic conditioning length L in [1, L, 5120] (default: 128; "
                             "ignored when --text-encoder/--prompt are given)")
    parser.add_argument("--seed", type=int, default=0, help="sampler seed (default: 0)")
    parser.add_argument("--steps", type=int, default=4,
                        help="consistency NFEs per chunk (default: 4, RAVEN's published budget)")
    parser.add_argument("--video-shift", type=float, default=12.0, help="video sigma shift (default: 12)")
    parser.add_argument("--audio-shift", type=float, default=3.0, help="audio sigma shift (default: 3)")
    parser.add_argument("--sink", type=int, default=2, help="attention-sink chunks (default: 2)")
    parser.add_argument("--window", type=int, default=2, help="sliding-window chunks (default: 2)")
    parser.add_argument("--device", default=default_device(),
                        help="device the probe expects ComfyUI to use; env RAVEN_PROBE_DEVICE")
    parser.add_argument("--weight-dtype", default="default", choices=("default", "bf16", "fp32"),
                        help="loader weight dtype (default: let comfy decide)")
    parser.add_argument("--kv-cache-storage", default=DEFAULT_KV_CACHE_STORAGE,
                        choices=KV_CACHE_STORAGE_CHOICES,
                        help="where the committed chunk KV cache lives, passed straight "
                             "through to RAVENStreamingSampler.sample (default: {}). "
                             "'gpu' keeps it on the card -- fastest, and the mode whose "
                             "memory the budget has to cover; 'cpu'/'cpu_pinned' keep it "
                             "in host memory, pinned so the per-chunk copies can overlap. "
                             "The mode is recorded in the report".format(
                                 DEFAULT_KV_CACHE_STORAGE))
    parser.add_argument("--repeat", type=repeat_count, default=1, metavar="N",
                        help="sampled runs in one process, 1-{}. 2 or more runs the "
                             "determinism and session-leak gates; 3 or more also "
                             "separates warm-up from a leak by gating the growth between "
                             "the LAST TWO runs (the plateau) instead of the growth from "
                             "the first, which legitimately includes making the DiT and "
                             "both VAEs resident (default: 1)".format(MAX_REPEAT))
    parser.add_argument("--compare-official-video", action="store_true",
                        help="OPTIONAL diagnostic, default off: after a successful run, "
                             "decode the same finished latent the old whole-clip way "
                             "(nodes.decode_images, kept for exactly this) and compare it "
                             "to the collector's IMAGE bitwise. COSTS: a second full "
                             "decode at a whole-clip peak -- the allocation the product "
                             "path removed after it OOMed a measured 39-frame run. Only "
                             "attempted at small geometries (<= {} decoded pixels, i.e. "
                             "the recommended --frames 39 at 512x288); 192- and "
                             "362-frame requests are refused, not attempted. An OOM here "
                             "is reported as a FAILURE".format(OFFICIAL_COMPARE_MAX_PIXELS))
    cancel_group = parser.add_mutually_exclusive_group()
    cancel_group.add_argument("--cancel-after-forward", type=int, default=0, metavar="N",
                              help="let N real forward_chunk calls through, then raise the "
                                   "real SamplingCancelled; the cancelled run is followed "
                                   "by the normal run(s), which must still succeed. "
                                   "Mutually exclusive with --cancel-after-chunk: a run "
                                   "can only be stopped at one point, and two cancelled "
                                   "runs in one plan would double the wall clock for no "
                                   "extra evidence")
    cancel_group.add_argument("--cancel-after-chunk", type=int, default=0, metavar="N",
                              help="let N chunks be DELIVERED -- decoded by both "
                                   "collectors, through the real VAEs -- and then raise "
                                   "the real SamplingCancelled from inside "
                                   "pipeline.on_chunk. This is the later cancellation "
                                   "point: --cancel-after-forward stops before anything "
                                   "has been decoded, this one stops after pixels and "
                                   "samples exist, so it is what shows that a cancel "
                                   "drops the decoders' buffers instead of returning a "
                                   "partial clip")
    parser.add_argument("--json", default=DEFAULT_REPORT,
                        help="report path, must be inside this repository "
                             "(default: {})".format(DEFAULT_REPORT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.text_encoder) != bool(args.prompt):
        parser.error(
            "--text-encoder and --prompt are all-or-nothing: with only one of them "
            "there is nothing to encode (or nothing to encode it with), and falling "
            "back to the synthetic lane would produce a report that looks like it "
            "verified the text encoder when it did not"
        )
    if args.stacked_lora_strength is not None and not args.stacked_lora_name:
        parser.error(
            "--stacked-lora-strength needs --stacked-lora-name: a strength on its own "
            "stacks nothing, and a report that carried it would look like it had"
        )
    if args.stacked_lora_name and stacked_lora_strength(args) == 0.0:
        parser.error(
            "--stacked-lora-strength 0 is refused: LoraLoaderModelOnly returns the "
            "model unchanged at strength 0, so the run would verify nothing about "
            "stacking while the report said a LoRA was stacked"
        )
    try:
        report_path = resolve_report_path(args.json)
    except ProbeError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse exits

    report = Report()
    report.args = {
        key: value
        for key, value in vars(args).items()
        if key not in ("json",)
    }
    report.args["json"] = str(report_path)
    try:
        run_probe(args, report)
    except BaseException as exc:  # noqa: BLE001 - a crashed probe still reports
        report.errors.append("{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc()))

    atomic_write_json(report_path, report.to_dict())
    print(report.render())
    print("report: {}".format(report_path))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
