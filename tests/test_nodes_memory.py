"""The rollout VRAM budget the sampler node reserves from ComfyUI.

The estimate is a *pure function* of the request, the retention policy and the
DiT's own shape, so it is testable to the byte without a GPU. What is checked
here is not "does it equal 34.1 GiB" -- pinning the number would freeze the
model rather than the reasoning -- but that:

* the published request (1376x768, 192 frames, sink 2 / window 2, BF16) lands
  in a plausible band, and the KV cache is what dominates it;
* a small request costs far less, i.e. the estimate is a function of the shape
  and not a table of defaults;
* it is monotone in every knob that can only make the rollout bigger;
* the KV term follows :class:`~raven_streaming.cache.ChunkKVCache`'s real
  eviction policy, checked against an actually-driven cache rather than against
  a second copy of the same arithmetic.

Every number below is an *estimate*. None of it has been measured on a GPU; the
detail dict exists so that a real probe can be compared against it term by term.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raven_streaming import layout as layout_mod, lora as lora_mod, nodes  # noqa: E402
from raven_streaming.cache import ChunkKVCache  # noqa: E402

GIB = 1024 ** 3

#: The published full non-pruned H3 DiT, BF16 (comfy.ldm.minimax.model defaults).
PUBLISHED = nodes.DiTDimensions(
    num_layers=50,
    num_heads=56,
    head_dim=128,
    hidden_size=5376,
    ffn_hidden_size=14336,
    compute_dtype_size=2,
    measured=("published",),
)

#: What ``estimate_decode_memory`` returns for one 7-latent chunk at 1376x768.
TYPICAL_DECODE_WORKSPACE = 700_000_000


def budget(
    width: int = 1376,
    height: int = 768,
    frames: int = 192,
    *,
    text_len: int = 512,
    sink: int = 2,
    window=2,
    dims: nodes.DiTDimensions = PUBLISHED,
    decode: int = TYPICAL_DECODE_WORKSPACE,
    latent_dtype_size: int = 4,
    kv_cache_storage: str = "gpu",
) -> nodes.RolloutMemoryBudget:
    request = layout_mod.T2VALayout.from_request(
        text_len=text_len,
        frames=frames,
        width=width,
        height=height,
        warn_experimental=False,
    )
    return nodes.estimate_rollout_budget(
        layout=request,
        text_len=text_len,
        sink=sink,
        window=window,
        dims=dims,
        latent_dtype_size=latent_dtype_size,
        decode_workspace_bytes=decode,
        kv_cache_storage=kv_cache_storage,
    )


# --------------------------------------------------------------------------
# the published request
# --------------------------------------------------------------------------


def test_the_published_request_reserves_a_plausible_amount():
    estimate = budget()
    assert 25 * GIB <= estimate.total_bytes <= 40 * GIB, estimate.describe()

    # and it is the KV cache that makes it that size, not padding
    assert estimate.kv_cache_bytes > 0.6 * estimate.total_bytes
    # three retained media chunks plus the text, plus the one being staged
    assert estimate.detail["kv_peak_rows"] > 20_000


def test_the_total_is_exactly_the_documented_sum():
    import math

    estimate = budget()
    working = max(estimate.forward_workspace_bytes, estimate.decode_workspace_bytes)
    subtotal = estimate.kv_cache_bytes + estimate.rollout_buffer_bytes + working
    assert estimate.safety_bytes == (
        math.ceil(subtotal * nodes.SAFETY_FRACTION) + nodes.SAFETY_FLOOR_BYTES
    )
    assert estimate.total_bytes == subtotal + estimate.safety_bytes


def test_the_kv_row_price_is_two_tensors_per_layer():
    estimate = budget()
    detail = estimate.detail
    # K and V, every layer, heads x head_dim wide, in the compute dtype
    assert detail["kv_bytes_per_row"] == 2 * 50 * 56 * 128 * 2
    assert estimate.kv_cache_bytes == detail["kv_peak_rows"] * detail["kv_bytes_per_row"]
    # one retained chunk at this canvas is multiple GB, which is the whole
    # reason the reserve cannot be the VAE decode workspace
    chunk_rows = detail["chunk_rows"][1]
    assert chunk_rows * detail["kv_bytes_per_row"] > 6 * GIB


def test_a_small_request_costs_far_less():
    small = budget(512, 288, 39)
    big = budget()
    assert small.total_bytes < 8 * GIB
    assert small.total_bytes < big.total_bytes / 4


def test_the_estimate_is_a_function_of_the_shape_not_a_table():
    totals = {
        (w, h, f): budget(w, h, f).total_bytes
        for (w, h, f) in [(512, 288, 39), (768, 448, 90), (1024, 576, 124), (1376, 768, 192)]
    }
    assert len(set(totals.values())) == len(totals)
    assert sorted(totals.values()) == list(totals.values())

    # KV scales with packed rows, i.e. with area: 4x the pixels, ~4x the rows
    small = budget(512, 288, 192).detail
    large = budget(1024, 576, 192).detail
    ratio = large["chunk_rows"][1] / small["chunk_rows"][1]
    assert 3.5 < ratio < 4.5


# --------------------------------------------------------------------------
# monotonicity
# --------------------------------------------------------------------------


def test_more_sink_chunks_cost_more():
    totals = [budget(sink=s).total_bytes for s in (1, 2, 3, 4)]
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]


def test_a_longer_window_costs_more_and_no_eviction_costs_most():
    totals = [budget(window=w).total_bytes for w in (0, 1, 2, 4)]
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]
    assert budget(window=None).total_bytes > totals[-1]


def test_more_frames_never_costs_less():
    totals = [budget(frames=f).total_bytes for f in (22, 39, 90, 192, 362)]
    assert totals == sorted(totals)
    # with a sliding window the KV term saturates; the buffers keep growing,
    # which is exactly why the two are separate terms
    assert budget(frames=362).kv_cache_bytes == budget(frames=192).kv_cache_bytes
    assert budget(frames=362).rollout_buffer_bytes > budget(frames=192).rollout_buffer_bytes
    # without eviction it does not saturate
    assert budget(frames=362, window=None).kv_cache_bytes > budget(
        frames=192, window=None
    ).kv_cache_bytes


def test_a_wider_compute_dtype_doubles_the_cache():
    bf16 = budget()
    fp32 = budget(dims=nodes.DiTDimensions(50, 56, 128, 5376, 14336, 4))
    assert fp32.kv_cache_bytes == 2 * bf16.kv_cache_bytes
    assert fp32.total_bytes > bf16.total_bytes


def test_a_longer_prompt_costs_more():
    totals = [budget(text_len=n).total_bytes for n in (16, 128, 512, 4096)]
    assert totals == sorted(totals)
    # a very long prompt becomes the widest forward, not just more cached rows
    assert budget(text_len=40_000).detail["widest_chunk_rows"] == 40_000


def test_a_bigger_model_costs_more():
    smaller = budget(dims=nodes.DiTDimensions(25, 56, 128, 5376, 14336, 2))
    assert smaller.kv_cache_bytes * 2 == budget().kv_cache_bytes


# --------------------------------------------------------------------------
# the KV term against the real cache
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sink, window", [(1, 1), (2, 2), (2, 0), (3, 4), (2, None)])
def test_the_kv_peak_follows_the_real_eviction_policy(sink, window):
    """Drive an actual ``ChunkKVCache`` and compare, rather than re-deriving."""
    text_len = 37
    request = layout_mod.T2VALayout.from_request(
        text_len=text_len, frames=90, width=256, height=256, warn_experimental=False
    )
    estimate = nodes.estimate_rollout_budget(
        layout=request,
        text_len=text_len,
        sink=sink,
        window=window,
        dims=nodes.DiTDimensions(1, 1, 1, 8, 8, 4),
        decode_workspace_bytes=0,
    )

    cache = ChunkKVCache(1, sink=sink, window=window)
    rows_of = [text_len] + [chunk.rows for chunk in request.chunks[:-1]]
    peak_held = 0
    peak_gathered = 0
    for rows in rows_of:
        # what a cached forward sees: the retained history it reads, plus the
        # chunk it has staged and not yet committed
        peak_gathered = max(peak_gathered, cache.retained_rows)
        cache.stage(0, torch.zeros(rows, 1, 1), torch.zeros(rows, 1, 1))
        peak_held = max(peak_held, cache.retained_rows + rows)
        cache.commit()
    peak_gathered = max(peak_gathered, cache.retained_rows)
    peak_held = max(peak_held, cache.retained_rows)

    assert estimate.detail["kv_peak_rows"] == peak_held
    assert estimate.detail["kv_gathered_rows"] == peak_gathered


def test_the_last_chunk_is_not_priced_because_it_is_never_committed():
    request = layout_mod.T2VALayout.from_request(
        text_len=10, frames=39, width=256, height=256, warn_experimental=False
    )
    rows = nodes._committed_chunk_rows(request, 10)
    assert len(rows) == len(request.chunks)  # text + every chunk but the last
    assert rows[0] == 10
    assert rows[1:] == [chunk.rows for chunk in request.chunks[:-1]]


# --------------------------------------------------------------------------
# the working-set term
# --------------------------------------------------------------------------


def test_the_decode_workspace_only_matters_when_it_is_the_larger_one():
    lean = budget(decode=0)
    fat = budget(decode=64 * GIB)
    assert lean.total_bytes < fat.total_bytes
    # forward and decode never run at the same time, so it is a max, not a sum
    working_lean = max(lean.forward_workspace_bytes, lean.decode_workspace_bytes)
    assert lean.total_bytes == (
        lean.kv_cache_bytes
        + lean.rollout_buffer_bytes
        + working_lean
        + lean.safety_bytes
    )
    assert fat.total_bytes - fat.safety_bytes - (
        lean.total_bytes - lean.safety_bytes
    ) == 64 * GIB - lean.forward_workspace_bytes


def test_the_forward_workspace_counts_the_gather_and_the_fp32_lora_temporaries():
    from raven_streaming import runtime_linear

    estimate = budget()
    detail = estimate.detail
    assert estimate.forward_workspace_bytes == (
        detail["activation_bytes"]
        + detail["kv_gather_bytes"]
        + detail["lora_fp32_temp_bytes"]
    )
    # the LoRA residual is row-chunked against a fixed budget, so it does not
    # scale with the chunk; that is a property of runtime_linear, not a guess
    assert detail["lora_fp32_temp_bytes"] == runtime_linear.DEFAULT_TEMP_BUDGET_BYTES
    assert budget(512, 288, 39).detail["lora_fp32_temp_bytes"] == detail[
        "lora_fp32_temp_bytes"
    ]


def test_the_rollout_buffers_are_the_samplers_own_tensors():
    estimate = budget(512, 288, 39, decode=0)
    detail = estimate.detail
    request = layout_mod.T2VALayout.from_request(
        text_len=512, frames=39, width=512, height=288, warn_experimental=False
    )
    video_clip = 24 * request.latent_t * request.latent_h * request.latent_w
    audio_clip = 32 * 2 * request.audio_t
    video_chunk = 24 * 5 * request.latent_h * request.latent_w
    audio_chunk = 32 * 2 * max(c.audio_latents for c in request.chunks)
    assert estimate.rollout_buffer_bytes == 4 * (
        nodes.FULL_CLIP_TENSORS * (video_clip + audio_clip)
        + nodes.STEP_TENSORS * (video_chunk + audio_chunk)
    )
    assert detail["latent_dtype_size"] == 4
    # half-precision latents would halve them
    assert budget(512, 288, 39, decode=0, latent_dtype_size=2).rollout_buffer_bytes == (
        estimate.rollout_buffer_bytes // 2
    )


# --------------------------------------------------------------------------
# measuring the model
# --------------------------------------------------------------------------


def test_the_dimensions_are_measured_off_the_live_model():
    class Attention:
        heads = 7
        head_dim = 16

    class FC1:
        out_features = 512

    class MLP:
        fc1 = FC1()

    class Block:
        attn = Attention()
        mlp = MLP()

    class DiT:
        hidden_size = 128
        blocks = [Block(), Block(), Block()]
        dtype = torch.bfloat16

    class Patcher:
        model = type("BaseModel", (), {"diffusion_model": DiT()})()

    dims = nodes.dit_dimensions(Patcher())
    assert (dims.num_layers, dims.num_heads, dims.head_dim) == (3, 7, 16)
    assert dims.hidden_size == 128
    assert dims.ffn_hidden_size == 256  # fc1 emits gate + up
    assert dims.compute_dtype_size == 2
    assert dims.inner_dim == 112
    assert set(dims.measured) == {
        "num_layers",
        "num_heads",
        "head_dim",
        "hidden_size",
        "ffn_hidden_size",
        "compute_dtype_size",
    }


def test_an_unmeasurable_model_falls_back_to_the_published_numbers():
    dims = nodes.dit_dimensions(object())
    published = lora_mod.RavenBaseConfig()
    assert dims.num_layers == published.num_layers
    assert dims.num_heads == published.num_attention_heads
    assert dims.head_dim == published.attention_head_dim
    assert dims.hidden_size == published.hidden_size
    assert dims.ffn_hidden_size == published.ffn_hidden_size
    assert dims.compute_dtype_size == 2  # BF16, the published base dtype
    # and it says so, so a report can tell an estimate from a measurement
    assert dims.measured == ()
    assert dims.describe()["measured"] == []


@pytest.mark.parametrize(
    "dtype, size",
    [(torch.bfloat16, 2), (torch.float16, 2), (torch.float32, 4), (torch.float64, 8)],
)
def test_dtype_sizes_come_from_torch(dtype, size):
    assert nodes.dtype_size(dtype) == size


def test_an_unknown_dtype_falls_back_loudly_enough():
    assert nodes.dtype_size(None) == 2
    assert nodes.dtype_size(None, default=4) == 4
    assert nodes.dtype_size("not a dtype", default=4) == 4


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_the_budget_reports_every_term_it_used():
    estimate = budget()
    payload = estimate.to_dict()
    assert payload["total_bytes"] == estimate.total_bytes
    detail = payload["detail"]
    for key in (
        "kv_peak_rows",
        "kv_gathered_rows",
        "kv_bytes_per_row",
        "activation_bytes",
        "chunk_rows",
        "text_len",
        "sink",
        "window",
        "frames",
        "width",
        "height",
        "frame_rows",
        "num_layers",
        "num_heads",
        "head_dim",
        "compute_dtype_size",
        "safety_fraction",
    ):
        assert key in detail, key
    text = estimate.describe()
    assert "GiB" in text and "KV" in text and "safety" in text


# --------------------------------------------------------------------------
# where the KV cache lives
# --------------------------------------------------------------------------
#
# The same request, priced twice. Nothing about the *rollout* changes between
# these two: the same rows are committed, the same chunks are retained, the same
# tensors are read. What changes is which side of the PCIe bus the retained K/V
# sits on, and that is the difference between a reserve the published card can
# hold and one it cannot.


def test_the_gpu_cache_is_the_28_gib_one_the_estimate_started_from():
    estimate = budget(text_len=128, kv_cache_storage="gpu")
    assert estimate.kv_cache_storage == "gpu"
    # every retained row of every layer, on the card
    assert 27 * GIB < estimate.kv_cache_bytes < 29 * GIB
    assert estimate.cpu_kv_steady_bytes == estimate.cpu_kv_peak_bytes == 0
    assert estimate.kv_cache_bytes > 0.6 * estimate.total_bytes


@pytest.mark.parametrize("storage", ["cpu", "cpu_pinned"])
def test_a_host_backed_cache_costs_one_merged_slot_on_the_card(storage):
    estimate = budget(text_len=128, kv_cache_storage=storage)

    # nothing of the cache is persistently resident ...
    assert estimate.kv_cache_bytes == 0
    # ... the card carries one layer's merged retained + current K/V
    assert 0.55 * GIB < estimate.kv_slot_bytes < 0.58 * GIB
    detail = estimate.detail
    assert estimate.kv_slot_bytes == (
        2 * detail["kv_peak_rows"] * PUBLISHED.inner_dim * PUBLISHED.compute_dtype_size
    )
    # which is 50x cheaper than the same rows across all 50 layers
    gpu = budget(text_len=128, kv_cache_storage="gpu")
    assert gpu.kv_cache_bytes == estimate.kv_slot_bytes * PUBLISHED.num_layers

    # the host cost is real and reported, it is just not VRAM
    assert estimate.cpu_kv_peak_bytes == gpu.kv_cache_bytes
    assert 0 < estimate.cpu_kv_steady_bytes <= estimate.cpu_kv_peak_bytes
    assert detail["canonical_cpu_kv_steady_bytes"] == estimate.cpu_kv_steady_bytes
    assert detail["canonical_cpu_kv_peak_bytes"] == estimate.cpu_kv_peak_bytes
    assert detail["kv_host_pinned"] is (storage == "cpu_pinned")
    assert estimate.kv_host_pinned is (storage == "cpu_pinned")


def test_the_steady_state_is_the_settled_retention_not_the_staging_peak():
    """``peak`` includes the chunk being staged; ``steady`` is what is kept."""
    estimate = budget(text_len=128, kv_cache_storage="cpu_pinned")
    detail = estimate.detail
    assert detail["kv_steady_rows"] < detail["kv_peak_rows"]
    assert estimate.cpu_kv_steady_bytes == detail["kv_steady_rows"] * detail["kv_bytes_per_row"]


def test_moving_the_cache_to_the_host_is_what_makes_the_reserve_small():
    on_card = budget(kv_cache_storage="gpu")
    on_host = budget(kv_cache_storage="cpu_pinned")
    assert on_host.total_bytes < on_card.total_bytes / 5
    assert on_host.total_bytes < 6 * GIB
    # the slot replaces the gather inside the forward workspace: with the cache
    # on the host, the retained rows arrive with the current chunk's
    assert on_host.forward_workspace_bytes == (
        on_host.detail["activation_bytes"]
        + on_host.kv_slot_bytes
        + on_host.detail["lora_fp32_temp_bytes"]
    )
    assert on_card.forward_workspace_bytes == (
        on_card.detail["activation_bytes"]
        + on_card.detail["kv_gather_bytes"]
        + on_card.detail["lora_fp32_temp_bytes"]
    )


def test_the_storage_mode_changes_nothing_about_the_rollout_itself():
    """Same rows, same chunks, same retention -- only the address space differs."""
    shared = (
        "kv_peak_rows",
        "kv_gathered_rows",
        "kv_steady_rows",
        "kv_bytes_per_row",
        "chunk_rows",
        "activation_bytes",
        "widest_chunk_rows",
        "committed_chunks",
    )
    on_card = budget(kv_cache_storage="gpu").detail
    on_host = budget(kv_cache_storage="cpu_pinned").detail
    for key in shared:
        assert on_card[key] == on_host[key], key


def test_the_description_says_where_the_cache_is():
    assert "(gpu)" in budget(kv_cache_storage="gpu").describe()
    pinned = budget(kv_cache_storage="cpu_pinned").describe()
    assert "KV slot" in pinned and "pinned" in pinned
    assert "pageable" in budget(kv_cache_storage="cpu").describe()


# --------------------------------------------------------------------------
# the two phases
# --------------------------------------------------------------------------


def plan(
    *,
    storage: str = "cpu_pinned",
    num_streams: int = 2,
    total_gib: int = 141,
    free_gib: int = 137,
    dit_model_bytes: int = 66 * GIB,
    video_model_bytes: int = 4 * GIB,
    audio_model_bytes: int = GIB + GIB // 2,
) -> nodes.JointOffloadPlan:
    facts = nodes.DeviceMemoryFacts(
        device="cuda:0",
        total_bytes=total_gib * GIB,
        free_bytes=free_gib * GIB,
        extra_reserved_bytes=400 * 1024 ** 2,
        num_streams=num_streams,
        loaded_bytes={"dit": 0, "video": 0, "audio": 0},
        measured=True,
    )
    return nodes.plan_joint_offload(
        budget=budget(kv_cache_storage=storage),
        dims=PUBLISHED,
        facts=facts,
        dit_model_bytes=dit_model_bytes,
        video_model_bytes=video_model_bytes,
        audio_model_bytes=audio_model_bytes,
    )


def test_each_phase_asks_for_its_own_workspace_and_no_weights():
    joint = plan()
    reserve = budget(kv_cache_storage="cpu_pinned")

    assert joint.dit.memory_required == sum(joint.dit.items.values())
    assert joint.vae.memory_required == sum(joint.vae.items.values())

    # the DiT phase pays for the KV slot and the forward, never for the decode
    assert joint.dit.items["kv_slot_bytes"] == reserve.kv_slot_bytes
    assert joint.dit.items["activation_bytes"] == reserve.detail["activation_bytes"]
    assert "decode_workspace_bytes" not in joint.dit.items
    # the VAE phase pays for the decode and the latents that outlive the chunk
    assert joint.vae.items["decode_workspace_bytes"] == reserve.decode_workspace_bytes
    assert joint.vae.items["rollout_buffer_bytes"] == reserve.rollout_buffer_bytes
    # and neither includes a byte of weights
    for phase in (joint.dit, joint.vae):
        assert phase.memory_required < phase.model_bytes


def test_the_planned_peak_is_the_larger_phase_never_the_sum():
    joint = plan()
    assert joint.planned_peak_bytes == max(
        joint.dit.planned_peak_bytes, joint.vae.planned_peak_bytes
    )
    assert joint.planned_peak_bytes < (
        joint.dit.planned_peak_bytes + joint.vae.planned_peak_bytes
    )
    # the DiT is the big phase; the VAEs are 5.5 GiB plus a decode
    assert joint.dit.planned_peak_bytes > joint.vae.planned_peak_bytes
    assert joint.vae.planned_peak_bytes < 10 * GIB


def test_the_plan_does_not_depend_on_the_card():
    """The same request asks for the same thing on 24, 80 and 141 GiB."""
    plans = [plan(total_gib=t, free_gib=t - 4) for t in (24, 80, 141)]
    requests = {(p.dit.memory_required, p.vae.memory_required) for p in plans}
    assert len(requests) == 1
    peaks = {p.planned_peak_bytes for p in plans}
    assert len(peaks) == 1
    # the card is a fact in the report, and only that
    assert {p.facts.total_bytes for p in plans} == {24 * GIB, 80 * GIB, 141 * GIB}


def test_a_gpu_cache_moves_into_both_phases_because_it_never_leaves_the_card():
    joint = plan(storage="gpu")
    resident = budget(kv_cache_storage="gpu").kv_cache_bytes
    assert joint.dit.items["kv_resident_bytes"] == resident
    # it is still allocated while a chunk decodes, so the VAE phase carries it
    assert joint.vae.items["kv_resident_bytes"] == resident
    assert joint.vae.memory_required > plan(storage="cpu_pinned").vae.memory_required


def test_the_host_backed_plan_is_the_one_that_fits_the_reference_card():
    assert plan(storage="cpu_pinned").vae.within_planning is True
    # with 28 GiB of KV pinned to the card, no phase fits 22 GiB
    assert plan(storage="gpu").vae.within_planning is False


@pytest.mark.parametrize("num_streams", [0, 1, 2])
def test_the_offload_envelope_follows_the_measured_stream_count(num_streams):
    envelope = nodes.offload_envelope_bytes(PUBLISHED, num_streams)
    widest = nodes.largest_module_bytes(PUBLISHED)
    # fc1 is hidden -> 2 * ffn, the largest module in a block
    assert widest == 5376 * 2 * 14336 * 2
    assert envelope == (num_streams + 1) * widest
    # it is reported, never added to what this node asks upstream for
    joint = plan(num_streams=num_streams)
    assert joint.offload_envelope_bytes == envelope
    assert "offload_envelope_bytes" not in joint.dit.items
    assert joint.to_dict()["offload_envelope_bytes"] == envelope


def test_the_plan_reports_every_number_it_used():
    payload = plan().to_dict()
    assert set(payload) >= {
        "phases",
        "planned_peak_bytes",
        "within_planning",
        "planning_bytes",
        "hard_cap_bytes",
        "kv_cache_storage",
        "offload_envelope_bytes",
        "facts",
    }
    assert payload["planning_bytes"] == 22 * GIB
    assert payload["hard_cap_bytes"] == 24 * GIB
    assert set(payload["phases"]) == {"dit", "vae"}
    text = plan().describe()
    assert "phase swap on" in text and "reference budget" in text


def test_a_workspace_the_reference_card_cannot_hold_is_a_warning_not_an_error(caplog):
    """Live tensors above the budget: no offloading fixes that, so it is said."""
    import logging

    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        plan(storage="gpu").diagnose()  # no raise
    message = next(r.getMessage() for r in caplog.records)
    assert "workspace alone" in message
    assert "no amount" in message


def test_weights_bigger_than_the_reference_card_are_not_warned_about(caplog):
    """A 66 GiB DiT on a 24 GiB card is the normal case: upstream streams it."""
    import logging

    joint = plan(storage="cpu_pinned")
    assert joint.dit.model_bytes > 22 * GIB          # far above the reference
    assert joint.dit.memory_required < 22 * GIB      # ... but its workspace is not
    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        joint.diagnose()
    assert caplog.records == []


# --------------------------------------------------------------------------
# the facts, and what came back from a load
# --------------------------------------------------------------------------


class StubPatcher:
    def __init__(self, loaded=0, size=0, buffer=0):
        self._loaded = loaded
        self._size = size
        self.model = type("Model", (), {"model_offload_buffer_memory": buffer})()

    def loaded_size(self):
        return self._loaded

    def model_size(self):
        return self._size


class StubModelManagement:
    NUM_STREAMS = 2

    def __init__(self, total=141 * GIB, free=100 * GIB):
        self.total = total
        self.free = free

    def get_torch_device(self):
        return "cuda:0"

    def get_total_memory(self, device=None):
        return self.total

    def get_free_memory(self, device=None):
        return self.free

    def extra_reserved_memory(self):
        return 400 * 1024 ** 2


def test_other_processes_are_reported_on_their_own_line():
    """U_base is what is on the card and not ours. It is never charged to us."""
    facts = nodes.DeviceMemoryFacts.probe(
        {"dit": StubPatcher(loaded=10 * GIB), "video": StubPatcher(loaded=2 * GIB)},
        model_management=StubModelManagement(total=80 * GIB, free=50 * GIB),
    )
    assert facts.ours_loaded_bytes == 12 * GIB
    # 80 total - 50 free = 30 used, of which 12 is ours
    assert facts.baseline_used_bytes == 18 * GIB
    assert facts.num_streams == 2
    assert facts.measured is True
    assert "U_base (not ours) 18.00 GiB" in facts.describe()


def test_facts_that_cannot_be_measured_are_marked_as_such():
    class NoDevice(StubModelManagement):
        def get_total_memory(self, device=None):
            raise RuntimeError("no device")

    facts = nodes.DeviceMemoryFacts.probe({}, model_management=NoDevice())
    assert facts.measured is False
    assert facts.total_bytes == 0
    assert "unmeasured" in facts.describe()
    # and a plan made from them still asks for the same workspace
    joint = nodes.plan_joint_offload(
        budget=budget(kv_cache_storage="cpu_pinned"), dims=PUBLISHED, facts=facts
    )
    assert joint.dit.memory_required == plan().dit.memory_required


def test_the_residency_record_is_upstreams_own_accounting():
    residency = nodes.measure_residency(
        "dit",
        model=StubPatcher(loaded=12 * GIB, buffer=GIB // 2),
        video=StubPatcher(loaded=0),
        audio=StubPatcher(loaded=0),
        workspace_bytes=3 * GIB,
    )
    assert residency.model_bytes == 12 * GIB
    assert residency.offload_buffer_bytes == GIB // 2
    assert residency.predicted_peak_bytes == 12 * GIB + GIB // 2 + 3 * GIB
    assert residency.within_planning is True
    assert residency.over_bytes == 0
    assert "comfy offload buffer" in residency.describe()

    payload = residency.to_dict()
    assert payload["model_offload_buffer_memory"] == GIB // 2
    assert payload["planning_bytes"] == 22 * GIB


def test_a_residency_above_the_reference_is_recorded_not_refused(caplog):
    import logging

    residency = nodes.measure_residency(
        "dit", model=StubPatcher(loaded=60 * GIB), workspace_bytes=3 * GIB
    )
    assert residency.within_planning is False
    assert residency.over_bytes == 63 * GIB - 22 * GIB
    with caplog.at_level(logging.DEBUG, logger="raven_streaming.nodes"):
        assert nodes.report_residency(residency) is residency  # no raise
    message = next(r.getMessage() for r in caplog.records)
    # the verdict is in the record; it is not shouted, because upstream sized
    # this against memory that really was free
    assert "over by 41.00 GiB" in message
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# --------------------------------------------------------------------------
# the 24 GiB diagnostic
# --------------------------------------------------------------------------


def test_the_hard_cap_is_a_yardstick_and_never_stops_a_run(caplog):
    import logging

    reading = {
        "device": "cuda:0",
        "reserved_bytes": 30 * GIB,
        "driver_used_bytes": 70 * GIB,
        "total_bytes": 141 * GIB,
    }
    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        out = nodes.hard_cap_watch("a chunk", state=reading)
    assert out["over_reference"] is True
    assert out["where"] == "a chunk"
    message = next(r.getMessage() for r in caplog.records)
    # this process's reserve is the subject; the device total is context only
    assert "30.00 GiB reserved" in message
    assert "not this node's" in message


def test_a_run_inside_the_reference_says_nothing(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="raven_streaming.nodes"):
        out = nodes.hard_cap_watch(
            "the dit phase load",
            state={"device": "cuda:0", "reserved_bytes": 9 * GIB, "driver_used_bytes": None},
        )
    assert out["over_reference"] is False
    assert caplog.records == []


def test_an_unmeasurable_device_is_not_a_verdict():
    out = nodes.hard_cap_watch("nowhere", state={"reserved_bytes": None})
    assert out["over_reference"] is False
