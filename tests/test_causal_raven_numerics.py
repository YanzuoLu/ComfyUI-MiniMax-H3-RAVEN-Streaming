"""The causal lane reproduces RAVEN's operator order, the dense lane Comfy's.

Why these tests exist
---------------------
A real-CUDA BF16 audit of one DiT block (``tools/probe_causal_operator_parity``)
found the Comfy port bit-identical to RAVEN for the qkv GEMM, QK norm, RoPE, the
AdaLN GEMM, scale/shift and the SwiGLU MLP -- and *not* identical for three
things:

======================  ==========================================  ==========
stage                   difference                                  rel L2
======================  ==========================================  ==========
attention wrapper       4-D ``optimized_attention`` vs RAVEN's       0.0024
                        packed 3-D SDPA fallback                     -0.0028
AdaLN native input      BF16 ``silu(t_emb)`` vs fp32 SiLU -> BF16    0.0035
                                                                     -0.0052
gate                    ``addcmul_`` vs ``(x + gate*other).to()``    0.000886
======================  ==========================================  ==========

They accumulate to ~9% relative L2 on the 50-block ``video_x0``. The causal
lane now runs RAVEN's spelling of all three; the dense lane must keep running
Comfy's, because the whole M2 claim is that it only *adds* a lane.

Everything here is a **bitwise** claim at BF16, checked against a literal
transcription of the RAVEN source (``projects/minimax_h3/modeling/transformer/
model.py`` and ``utils/flash_attn.py``) written out in the test itself. The
stages the audit already cleared -- qkv/norm/RoPE and the MLP -- are replayed
through the *Comfy* modules on purpose: the point of a replay is to isolate
what this module owns, not to re-litigate what the audit settled.

Requires a local ComfyUI checkout (see ``tests/conftest.py``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conftest  # noqa: E402

_UPSTREAM = conftest.find_upstream_comfyui()
if _UPSTREAM is None:  # pragma: no cover - environment without a checkout
    pytest.skip("No local ComfyUI checkout found", allow_module_level=True)
conftest.add_to_sys_path(_UPSTREAM)

import comfy.ldm.minimax.model as comfy_minimax  # noqa: E402
from comfy.ldm.minimax.model import (  # noqa: E402
    pack_audio,
    patchify_video,
    rope_rotation_table,
    unpack_audio,
    unpatchify_video,
)

import raven_streaming.causal_model as cm  # noqa: E402
from raven_streaming import layout as layout_mod  # noqa: E402
from raven_streaming.cache import ChunkKVCache  # noqa: E402
from raven_streaming.causal_model import (  # noqa: E402
    CLEAN_TIMESTEP_TEXT,
    CausalModelError,
    raven_packed_attention,
    velocity_to_x0,
)
from test_causal_common import (  # noqa: E402
    TINY_CONFIG,
    build_models,
    random_inputs,
    tiny_bf16_models,  # noqa: F401  (fixture re-export)
    tiny_layout,
    tiny_models,  # noqa: F401  (fixture re-export)
)

NUM_LAYERS = TINY_CONFIG["num_layers"]
HIDDEN = TINY_CONFIG["hidden_size"]
HEADS = TINY_CONFIG["num_attention_heads"]
HEAD_DIM = TINY_CONFIG["attention_head_dim"]
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# literal RAVEN transcriptions (the reference every bitwise claim below uses)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ravens_math_sdp():
    """Run under the SDP reduction setting a RAVEN process has.

    RAVEN never touches ``allow_fp16_bf16_reduction_math_sdp``, so it runs at
    torch's default (reduction **off**, fp32 accumulation).
    ``comfy.model_management`` sets it **on** at import, process-wide. That one
    switch is the whole remaining SDPA difference the vr audit isolated, so the
    reference transcription below has to be evaluated the way RAVEN evaluates
    it, and :func:`raven_packed_attention` has to be invariant to whatever Comfy
    left set (pinned by ``test_sdpa_fallback_forces_the_reduction_off``).
    """
    previous = bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    try:
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)


def raven_sdpa_varlen(q, k, v, *, q_bounds, k_bounds, dropout_p, softmax_scale, causal):
    """``utils/flash_attn.py::_sdpa_varlen``, transcribed verbatim."""
    with ravens_math_sdp():
        out = torch.empty_like(q)
        pairs = zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
        for q_start, q_stop, k_start, k_stop in pairs:
            if q_start == q_stop:
                continue
            segment = torch.nn.functional.scaled_dot_product_attention(
                q[q_start:q_stop].transpose(0, 1),
                k[k_start:k_stop].transpose(0, 1),
                v[k_start:k_stop].transpose(0, 1),
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            ).transpose(0, 1)
            out[q_start:q_stop].copy_(segment)
    return out


def raven_modulate_scale_shift(x, shift, scale, indices, *, dtype):
    """``model.py::_modulate_scale_shift``, eager branch."""
    return (x * (1.0 + scale.index_select(0, indices))
            + shift.index_select(0, indices)).to(dtype)


def raven_modulate_gate(x, gate, other, indices, *, dtype):
    """``model.py::_modulate_gate``, eager branch."""
    return (x + gate.index_select(0, indices) * other).to(dtype)


def comfy_norm_rope(attn, q, k, rope_table, rows):
    """The fused QK-norm + RoPE call, exactly as ``Attention.forward`` makes it.

    Audited bit-identical to RAVEN's ``_apply_qk_norm`` + ``_apply_rope_qk``, so
    the replays below borrow it rather than transcribing the RAVEN pair.
    """
    import comfy.model_management
    import comfy.quant_ops

    q = q.reshape(1, rows, attn.heads, attn.head_dim).clone()
    k = k.reshape(1, rows, attn.heads, attn.head_dim).clone()
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=q.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=k.device)
    rot = rope_table.shape[-3] * 2
    comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_table, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0], k[0]


def raven_attention_replay(attn, x, rope_table, past):
    """RAVEN's ``CausalMiniMaxH3Attention.forward`` over the Comfy modules."""
    rows = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.reshape(rows, attn.heads, attn.head_dim).clone()
    q, k = comfy_norm_rope(attn, q, k, rope_table, rows)
    if past is not None:
        k = torch.cat((past[0], k), dim=0)
        v = torch.cat((past[1], v), dim=0)
    kv_rows = k.shape[0]
    out = raven_sdpa_varlen(
        q, k, v,
        q_bounds=[0, rows], k_bounds=[0, kv_rows],
        dropout_p=0.0, softmax_scale=attn.head_dim ** -0.5, causal=False,
    )
    return attn.out_proj(out.reshape(rows, attn.heads * attn.head_dim))


def raven_refiner_replay(model, text_states, dtype):
    """RAVEN's ``refine_prompt_embeds`` + ``MiniMaxH3TokenRefiner.forward``."""
    h = model.condition_proj(text_states.to(dtype))
    for block in model.token_refiner.blocks:
        x = block.norm1(h)
        rows = x.shape[0]
        attn = block.attn
        q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
        q = attn.q_norm(q.view(rows, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(rows, attn.heads, attn.head_dim))
        v = v.view(rows, attn.heads, attn.head_dim)
        out = raven_sdpa_varlen(
            q, k, v, q_bounds=[0, rows], k_bounds=[0, rows],
            dropout_p=0.0, softmax_scale=attn.head_dim ** -0.5, causal=False,
        )
        h = h + attn.out_proj(out.reshape(rows, attn.heads * attn.head_dim))
        h = h + block.mlp(block.norm2(h))
    return model.token_refiner.final_norm(h)


def raven_chunk_setup(model, layout, index, video_x, audio_x, t_v, t_a, dtype):
    """RAVEN's embed + shared AdaLN input + index tables for one chunk."""
    chunk = layout.chunks[index]
    audio_n, rows = chunk.audio_rows, chunk.rows
    device = video_x.device

    video_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    h = torch.cat((model.audio_patch_proj(audio_rows).to(dtype),
                   model.video_patch_proj(video_rows).to(dtype)), dim=0)

    unique_t = sorted({t_v, t_a})
    t_row = {value: i for i, value in enumerate(unique_t)}
    t_emb = model.time_embedder(torch.tensor(unique_t, dtype=torch.float32, device=device))
    # RAVEN: one fp32 SiLU per forward, cast once, shared by every block
    adaln_input = F.silu(t_emb).to(dtype)

    combined = torch.empty(rows, dtype=torch.long, device=device)
    combined[:audio_n] = t_row[t_a] * 3 + layout_mod.AUDIO_TAG
    combined[audio_n:] = t_row[t_v] * 3 + layout_mod.VIDEO_TAG
    inverse = torch.empty(rows, dtype=torch.long, device=device)
    inverse[:audio_n] = t_row[t_a]
    inverse[audio_n:] = t_row[t_v]

    rope_table = rope_rotation_table(
        model.rope_freqs(layout.chunk_position_ids(index), device), dtype)
    return h, adaln_input, combined, inverse, rope_table


def raven_block_stack_replay(model, h, adaln_input, combined, rope_table, cache, dtype):
    """RAVEN's block loop: AdaLN -> norm/mod -> attention -> gate -> MLP -> gate."""
    for layer, block in enumerate(model.blocks):
        projected = block.adaln_proj.linear(adaln_input)
        projected = projected.view(projected.shape[0] * 3, 6 * HIDDEN)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            projected.chunk(6, dim=-1)

        residual = h
        x = raven_modulate_scale_shift(block.norm1(h), shift_msa, scale_msa,
                                       combined, dtype=dtype)
        past = None if cache is None else cache.retained(layer)
        x = raven_attention_replay(block.attn, x, rope_table, past)
        h = raven_modulate_gate(residual, gate_msa, x, combined, dtype=dtype)

        residual = h
        x = raven_modulate_scale_shift(block.norm2(h), shift_mlp, scale_mlp,
                                       combined, dtype=dtype)
        x = block.mlp(x)
        h = raven_modulate_gate(residual, gate_mlp, x, combined, dtype=dtype)
    return h


def raven_chunk_replay(model, layout, index, video_x, audio_x, t_v, t_a, cache, dtype):
    """A whole causal chunk in RAVEN's order: embed -> blocks -> final layer.

    Cache reads come out of the live :class:`ChunkKVCache`, so this replays a
    chunk that attends a history without duplicating the cache implementation.
    """
    chunk = layout.chunks[index]
    audio_n = chunk.audio_rows
    h, adaln_input, combined, inverse, rope_table = raven_chunk_setup(
        model, layout, index, video_x, audio_x, t_v, t_a, dtype)
    h = raven_block_stack_replay(model, h, adaln_input, combined, rope_table,
                                 cache, dtype)

    final = model.final_layer
    projected = final.adaln_proj.linear(adaln_input)
    projected = projected.view(projected.shape[0], 2 * HIDDEN)
    shift, scale = projected.chunk(2, dim=-1)
    x = raven_modulate_scale_shift(final.norm(h), shift, scale, inverse, dtype=dtype)
    x = x.to(torch.float32)
    video_out = final.video_out(x[audio_n:])
    audio_out = final.audio_out(x[:audio_n])

    # fp32 out, as RAVEN's fp32 output heads produce it: the reference never
    # rounds the velocity back to the latent's dtype and neither does the lane
    video = unpatchify_video(video_out, chunk.video_latents, layout.latent_h // 2,
                             layout.latent_w // 2, model.latents_dim, model.patch_size)
    return [video, unpack_audio(audio_out)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def bf16_chunk_inputs(layout, index=0, seed=1):
    video, audio, context = random_inputs(layout, seed=seed)
    return (layout.video_chunk_latent(video, index).to(BF16),
            layout.audio_chunk_latent(audio, index).to(BF16),
            context.to(BF16))


def prefilled_cache(model, context, sink=8, window=None):
    cache = ChunkKVCache(NUM_LAYERS, sink=sink, window=window)
    model.prefill_text(context, cache=cache, compute_dtype=context.dtype)
    return cache


@pytest.fixture(autouse=True)
def isolated_attention_backend():
    """No test may inherit another's backend resolution or SDP setting.

    The resolution is process-wide by design (importing ``flash_attn`` once is
    the point), so every test that injects a fake kernel has to start and end
    from a clean cache.
    """
    cm._reset_attention_backends()
    previous = bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    try:
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
        cm._reset_attention_backends()


class ClaimsCuda(torch.Tensor):
    """A CPU tensor that reports ``is_cuda``, so the FA path is reachable here.

    The backend chain's preconditions are RAVEN's ("cuda, fp16/bf16,
    head_dim <= 256"). Faking the device is what lets the *real* precondition
    code and the *real* dispatch loop run on a CPU box against fake kernels;
    nothing downstream of the dispatch reads the device except the fake itself.
    """

    @property
    def is_cuda(self):  # noqa: D401 - property override
        return True


def cuda_like(*tensors):
    return tuple(t.as_subclass(ClaimsCuda) for t in tensors)


class FakeVarlen:
    """Stands in for ``flash_attn*_varlen_func``: records kwargs, returns a marker."""

    def __init__(self, marker: float, error: BaseException = None, result=None):
        self.marker = marker
        self.error = error
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return torch.full_like(kwargs["q"], self.marker)


def install_fake_flash(monkeypatch, *, fa3=None, fa2=None):
    """Put fake ``flash_attn_interface`` / ``flash_attn`` modules on sys.modules."""
    for module_name, func in (("flash_attn_interface", fa3), ("flash_attn", fa2)):
        if func is None:
            monkeypatch.delitem(sys.modules, module_name, raising=False)
            continue
        module = types.ModuleType(module_name)
        module.flash_attn_varlen_func = func
        monkeypatch.setitem(sys.modules, module_name, module)
    cm._reset_attention_backends()


def packed_triplet(rows=5, kv_rows=9, dtype=BF16, head_dim=HEAD_DIM, seed=21):
    torch.manual_seed(seed)
    return (torch.randn(rows, HEADS, head_dim).to(dtype),
            torch.randn(kv_rows, HEADS, head_dim).to(dtype),
            torch.randn(kv_rows, HEADS, head_dim).to(dtype))


def fused_qkv_triplet(rows=7, dtype=BF16, head_dim=HEAD_DIM, seed=23):
    """The no-history layout: contiguous q/k, ``v`` still the fused-QKV view."""
    torch.manual_seed(seed)
    inner = HEADS * head_dim
    qkv = torch.randn(rows, 3 * inner).to(dtype)
    q, k, v = qkv.split(inner, dim=-1)
    return (q.view(rows, HEADS, head_dim).contiguous(),
            k.view(rows, HEADS, head_dim).contiguous(),
            v.view(rows, HEADS, head_dim))


class CallCounter:
    """Counts calls to a module-level function, forwarding to the original."""

    def __init__(self, module, name):
        self.module, self.name = module, name
        self.original = getattr(module, name)
        self.calls = 0

    def __enter__(self):
        def traced(*args, **kwargs):
            self.calls += 1
            return self.original(*args, **kwargs)

        setattr(self.module, self.name, traced)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.original)
        return False


# ---------------------------------------------------------------------------
# 1. packed SDPA: layout, scale, and who calls it
# ---------------------------------------------------------------------------


def test_packed_attention_is_the_raven_fallback_bitwise():
    torch.manual_seed(0)
    q = torch.randn(11, HEADS, HEAD_DIM).to(BF16)
    k = torch.randn(37, HEADS, HEAD_DIM).to(BF16)   # merged [retained | current]
    v = torch.randn(37, HEADS, HEAD_DIM).to(BF16)
    scale = HEAD_DIM ** -0.5

    ours = raven_packed_attention(q, k, v, scale=scale)
    theirs = raven_sdpa_varlen(q, k, v, q_bounds=[0, 11], k_bounds=[0, 37],
                               dropout_p=0.0, softmax_scale=scale, causal=False)
    assert ours.shape == q.shape
    assert ours.dtype == q.dtype
    assert torch.equal(ours, theirs)


def test_packed_attention_runs_the_three_dim_layout():
    """[heads, rows, dim] per document, not a 4-D [1, heads, rows, dim] batch."""
    torch.manual_seed(1)
    q = torch.randn(5, HEADS, HEAD_DIM).to(BF16)
    k = torch.randn(9, HEADS, HEAD_DIM).to(BF16)
    v = torch.randn(9, HEADS, HEAD_DIM).to(BF16)

    with ravens_math_sdp():
        manual = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=HEAD_DIM ** -0.5,
        ).transpose(0, 1)
    assert torch.equal(raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5), manual)

    with pytest.raises(CausalModelError, match=r"\[rows, heads, dim\]"):
        raven_packed_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                               scale=HEAD_DIM ** -0.5)


def test_packed_attention_uses_the_head_dim_scale():
    torch.manual_seed(2)
    q = torch.randn(6, HEADS, HEAD_DIM).to(BF16)
    k = torch.randn(6, HEADS, HEAD_DIM).to(BF16)
    v = torch.randn(6, HEADS, HEAD_DIM).to(BF16)
    right = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    wrong = raven_packed_attention(q, k, v, scale=1.0)
    assert not torch.equal(right, wrong)


def test_the_causal_lane_calls_the_packed_seam_with_raven_arguments(tiny_bf16_models):
    """What every layer hands the backend: 3-D packed tensors, the head scale, one
    document of ``rows`` queries over ``rows + history`` keys, and a ``site``
    label saying which stack the call came from.

    Wrapping the module attribute is how a probe taps this lane (the old
    ``KVTap`` wrapped ``causal_model.optimized_attention``, which the causal path
    no longer calls). The label exists because the DiT blocks and the token
    refiner share this one seam: counting calls would renumber every DiT block
    by the refiner calls in front of it.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    calls = []
    original = cm.raven_packed_attention

    def spy(q, k, v, *, scale, site=None):
        calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), scale, site))
        return original(q, k, v, scale=scale, site=site)

    cm.raven_packed_attention = spy
    try:
        cache = prefilled_cache(causal, context)
        text_rows = layout.text_len
        causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16,
        )
    finally:
        cm.raven_packed_attention = original

    refiner_calls = TINY_CONFIG["token_refiner_num_layers"]
    assert len(calls) == refiner_calls + 2 * NUM_LAYERS
    for q_shape, k_shape, v_shape, scale, _ in calls:
        assert len(q_shape) == 3 and len(k_shape) == 3 and len(v_shape) == 3
        assert q_shape[1:] == (HEADS, HEAD_DIM)
        assert k_shape == v_shape
        assert scale == HEAD_DIM ** -0.5

    # the labels, in call order: refiner blocks, then the prefill's DiT stack,
    # then the chunk's DiT stack
    assert [site for *_, site in calls] == (
        [("text_refiner", i) for i in range(refiner_calls)]
        + [("dit", i) for i in range(NUM_LAYERS)] * 2
    )

    chunk_rows = layout.chunks[0].rows
    for q_shape, k_shape, *_ in calls[:refiner_calls]:
        assert q_shape[0] == k_shape[0] == text_rows    # refiner self-attention
    for q_shape, k_shape, *_ in calls[refiner_calls:refiner_calls + NUM_LAYERS]:
        assert q_shape[0] == text_rows and k_shape[0] == text_rows  # the prefill
    for q_shape, k_shape, *_ in calls[refiner_calls + NUM_LAYERS:]:
        assert q_shape[0] == chunk_rows                 # q is current-only
        assert k_shape[0] == chunk_rows + text_rows     # k/v carry the history


def test_the_site_label_is_diagnostic_only():
    """It must not be able to change a single number."""
    torch.manual_seed(12)
    q = torch.randn(7, HEADS, HEAD_DIM).to(BF16)
    k = torch.randn(13, HEADS, HEAD_DIM).to(BF16)
    v = torch.randn(13, HEADS, HEAD_DIM).to(BF16)
    plain = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    for site in (None, ("dit", 0), ("dit", 49), ("text_refiner", 1)):
        assert torch.equal(raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5,
                                                  site=site), plain)


def test_the_layer_index_label_is_the_attentions_own(tiny_bf16_models):
    _, causal = tiny_bf16_models
    for index, block in enumerate(causal.blocks):
        assert block.attn._attention_site == ("dit", index)
        assert block.attn._attention_site[1] == block.attn.layer_idx
    # a plain attribute, so it stays out of the checkpoint
    assert not any("_attention_site" in name for name in causal.state_dict())


def test_causal_attention_never_calls_optimized_attention(tiny_bf16_models):
    """The causal lane bypasses Comfy's dispatcher; the dense lane still uses it."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)

    with CallCounter(comfy_minimax, "optimized_attention") as counter:
        cache = prefilled_cache(causal, context)
        assert counter.calls == 0, "the text refiner still runs optimized_attention"
        causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16,
        )
        assert counter.calls == 0

        full_video, full_audio, _ = random_inputs(layout)
        causal(x=[full_video.to(BF16), full_audio.to(BF16)],
               timestep=torch.tensor([600.0]), context=context)
        # dense: 1 refiner block + 3 DiT blocks, all through the official path
        assert counter.calls == (TINY_CONFIG["token_refiner_num_layers"]
                                 + TINY_CONFIG["num_layers"])


def test_causal_attention_matches_a_raven_replay(tiny_bf16_models):
    """Module forward vs the transcription, on a chunk that reads the cache."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)

    block = causal.blocks[0]
    rows = layout.chunks[0].rows
    torch.manual_seed(3)
    x = torch.randn(rows, HIDDEN).to(BF16)
    rope_table = rope_rotation_table(
        causal.rope_freqs(layout.chunk_position_ids(0), x.device), BF16)

    ours = block.attn(x.clone(), rope_freqs=rope_table, cache=cache)
    theirs = raven_attention_replay(block.attn, x.clone(), rope_table,
                                    cache.retained(0))
    assert torch.equal(ours, theirs)


def test_attention_override_option_is_refused(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    with pytest.raises(CausalModelError, match="optimized_attention_override"):
        causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16,
            transformer_options={"optimized_attention_override": object()},
        )


# ---------------------------------------------------------------------------
# 1a. the seam *layout*: RAVEN's strides, not only RAVEN's values
# ---------------------------------------------------------------------------
#
# A full-model FA3 run put layer-0 q/k/v at 5e-5..9e-5 relative L2 with
# identical arithmetic on both sides, and the evidence was the memory layout:
#
#   RAVEN   q/k [7168, 128, 1] contiguous     v [21504, 128, 1] fused view
#   before  q/k [21504, 128, 1] fused view    v [7168, 128, 1] cloned
#
# -- exactly swapped, because RAVEN's norm/RoPE are eager (fresh q/k, v left
# alone) while upstream's fused kernel rewrites q/k inside the QKV buffer and
# then clones v. FA3 is stride-sensitive, so this decides which kernel path
# runs. These tests pin the layout at the seam, per stream position.

INNER = HEADS * HEAD_DIM
RAVEN_ROW_MAJOR = (INNER, HEAD_DIM, 1)          # a fresh [rows, heads, dim]
RAVEN_FUSED_VIEW = (3 * INNER, HEAD_DIM, 1)     # a view into the fused QKV


def capture_seam(monkeypatch):
    """Record ``(site, name) -> (stride, is_contiguous, rows)`` for every call."""
    calls = []
    original = cm.raven_packed_attention

    def spy(q, k, v, *, scale, site=None):
        calls.append({
            "site": site,
            "q": (tuple(q.stride()), q.is_contiguous(), int(q.shape[0])),
            "k": (tuple(k.stride()), k.is_contiguous(), int(k.shape[0])),
            "v": (tuple(v.stride()), v.is_contiguous(), int(v.shape[0])),
        })
        return original(q, k, v, scale=scale, site=site)

    monkeypatch.setattr(cm, "raven_packed_attention", spy)
    return calls


def test_the_text_prefill_hands_the_backend_ravens_no_history_layout(
        tiny_bf16_models, monkeypatch):
    """No history: q/k freshly materialised, v still the fused-QKV view."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)
    calls = capture_seam(monkeypatch)

    prefilled_cache(causal, context)

    refiner = [c for c in calls if c["site"][0] == "text_refiner"]
    blocks = [c for c in calls if c["site"][0] == "dit"]
    assert len(refiner) == TINY_CONFIG["token_refiner_num_layers"]
    assert len(blocks) == NUM_LAYERS

    for call in blocks:
        assert call["q"] == (RAVEN_ROW_MAJOR, True, layout.text_len)
        assert call["k"] == (RAVEN_ROW_MAJOR, True, layout.text_len)
        # v is *not* cloned on this lane: RAVEN hands the varlen kernel the
        # fused view, and upstream's ``v = v.clone()`` is what made it differ
        assert call["v"] == (RAVEN_FUSED_VIEW, False, layout.text_len)

    # the refiner already matched RAVEN and must not move
    for call in refiner:
        assert call["q"] == (RAVEN_ROW_MAJOR, True, layout.text_len)
        assert call["k"] == (RAVEN_ROW_MAJOR, True, layout.text_len)
        assert call["v"] == (RAVEN_FUSED_VIEW, False, layout.text_len)


def test_a_media_chunk_reading_history_hands_contiguous_kv(tiny_bf16_models, monkeypatch):
    """With history the merge allocates, exactly as RAVEN's scatter does."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    calls = capture_seam(monkeypatch)

    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
        compute_dtype=BF16)

    rows = layout.chunks[0].rows
    kv_rows = rows + layout.text_len
    assert len(calls) == NUM_LAYERS
    for call in calls:
        assert call["site"][0] == "dit"
        assert call["q"] == (RAVEN_ROW_MAJOR, True, rows)
        # ``torch.cat`` allocates: RAVEN's merged K/V are contiguous too
        assert call["k"] == (RAVEN_ROW_MAJOR, True, kv_rows)
        assert call["v"] == (RAVEN_ROW_MAJOR, True, kv_rows)


def test_a_media_forward_without_history_keeps_the_fused_v_view(
        tiny_bf16_models, monkeypatch):
    """The no-history branch of a *media* layer, reached through the module.

    ``forward_chunk`` always has the text prefill in front of it, so this branch
    is only reachable by driving the attention module directly -- which is
    exactly what a first-chunk-without-text rollout would do.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    rows = layout.chunks[0].rows
    empty = ChunkKVCache(NUM_LAYERS, sink=8, window=None)
    torch.manual_seed(31)
    x = torch.randn(rows, HIDDEN).to(BF16)
    rope_table = rope_rotation_table(
        causal.rope_freqs(layout.chunk_position_ids(0), x.device), BF16)
    calls = capture_seam(monkeypatch)

    causal.blocks[0].attn(x, rope_freqs=rope_table, cache=empty, update_cache=True)

    (call,) = calls
    assert call["q"] == (RAVEN_ROW_MAJOR, True, rows)
    assert call["k"] == (RAVEN_ROW_MAJOR, True, rows)
    assert call["v"] == (RAVEN_FUSED_VIEW, False, rows)


def test_the_staged_key_is_the_very_tensor_the_backend_saw(tiny_bf16_models, monkeypatch):
    """``copy_key=False``: k is already this forward's own contiguous buffer.

    One copy fewer than the old ``copy_key=True``, and safe only because
    nothing writes to k after the materialisation -- ``torch.cat`` allocates and
    no attention backend writes its inputs.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)

    pointers = []
    original = cm.raven_packed_attention

    def spy(q, k, v, *, scale, site=None):
        pointers.append((site, k.data_ptr(), v.data_ptr()))
        return original(q, k, v, scale=scale, site=site)

    monkeypatch.setattr(cm, "raven_packed_attention", spy)
    cache = prefilled_cache(causal, context)

    blocks = [p for p in pointers if p[0][0] == "dit"]
    for layer, (_, k_ptr, v_ptr) in enumerate(blocks):
        record_k, record_v = cache.retained(layer)
        assert record_k.data_ptr() == k_ptr        # handed over, not copied
        assert record_v.data_ptr() != v_ptr        # copied out of the fused view


def test_the_cache_record_is_contiguous_and_owns_exactly_its_rows(tiny_bf16_models):
    """A staged view would pin the fused QKV buffer (3x) for the record's life."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="clean", compute_dtype=BF16,
        video_eps=torch.zeros_like(video), audio_eps=torch.zeros_like(audio))

    for layer in range(NUM_LAYERS):
        for tensor in cache.retained(layer):
            assert tensor.is_contiguous()
            assert tuple(tensor.stride()) == RAVEN_ROW_MAJOR
            element = tensor.element_size()
            assert tensor.untyped_storage().size() == tensor.numel() * element


def test_a_single_row_chunk_is_still_staged_as_its_own_buffer(tiny_bf16_models):
    """The one case where ``contiguous()`` alone would hand the cache a view.

    With one row the leading stride is irrelevant, so the fused-QKV view reports
    itself contiguous and ``contiguous()`` returns it unchanged. Staging that
    would pin 3x the rows it holds for the life of the record, so
    ``_materialise_rows`` checks the storage as well as the layout.
    """
    _, causal = tiny_bf16_models
    torch.manual_seed(41)
    context = torch.randn(1, 1, TINY_CONFIG["text_dim"]).to(BF16)
    cache = ChunkKVCache(NUM_LAYERS, sink=8, window=None)
    assert causal.prefill_text(context, cache=cache, compute_dtype=BF16) == 1

    for layer in range(NUM_LAYERS):
        for tensor in cache.retained(layer):
            assert tensor.shape[0] == 1
            assert tensor.is_contiguous()
            assert tensor.untyped_storage().size() == tensor.numel() * tensor.element_size()


def test_materialise_rows_only_copies_when_it_has_to():
    inner = HEADS * HEAD_DIM
    qkv = torch.randn(6, 3 * inner).to(BF16)
    view = qkv.split(inner, dim=-1)[1].view(6, HEADS, HEAD_DIM)
    assert not view.is_contiguous()
    out = cm._materialise_rows(view)
    assert out is not view and out.is_contiguous()
    assert tuple(out.stride()) == RAVEN_ROW_MAJOR
    assert torch.equal(out, view)

    owned = torch.randn(6, HEADS, HEAD_DIM).to(BF16)
    assert cm._materialise_rows(owned) is owned          # nothing to do

    single = torch.randn(1, 3 * inner).to(BF16).split(inner, dim=-1)[1].view(
        1, HEADS, HEAD_DIM)
    assert single.is_contiguous()                        # size-1 leading dim
    copied = cm._materialise_rows(single)
    assert copied is not single
    assert copied.untyped_storage().size() == copied.numel() * copied.element_size()
    assert torch.equal(copied, single)


def test_the_record_survives_the_forwards_that_follow_it(tiny_bf16_models):
    """Lifetime: k is handed over uncopied, so nothing may write it afterwards."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    before = [(k.clone(), v.clone()) for k, v in
              (cache.retained(layer) for layer in range(NUM_LAYERS))]

    for _ in range(2):
        causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16)
    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="clean", compute_dtype=BF16,
        video_eps=torch.zeros_like(video), audio_eps=torch.zeros_like(audio))

    for layer, (keys, values) in enumerate(before):
        current_k, current_v = cache.retained(layer)
        assert torch.equal(current_k[:keys.shape[0]], keys)
        assert torch.equal(current_v[:values.shape[0]], values)


def test_the_dense_lane_keeps_upstreams_layout(tiny_bf16_models, monkeypatch):
    """None of this touches the dense path: it is still ``super().forward``."""
    import comfy.ldm.modules.attention as comfy_attention

    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    seen = []
    original = comfy_minimax.optimized_attention

    def spy(q, k, v, heads, *args, **kwargs):
        # upstream's containers: [1, heads, rows, dim]
        seen.append((tuple(q.peek().shape), tuple(v.peek().stride())))
        return original(q, k, v, heads, *args, **kwargs)

    monkeypatch.setattr(comfy_minimax, "optimized_attention", spy)
    causal(x=[video.to(BF16), audio.to(BF16)], timestep=torch.tensor([600.0]),
           context=context.to(BF16))

    assert len(seen) == TINY_CONFIG["token_refiner_num_layers"] + NUM_LAYERS
    for shape, v_stride in seen:
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == HEADS
        # upstream clones v, so the row stride inside the [1, h, rows, d]
        # container is the clone's -- never the fused view's 3x one
        assert v_stride[-2] == INNER
        assert v_stride[-2] != 3 * INNER
    del comfy_attention


# ---------------------------------------------------------------------------
# 1b. the backend chain: FA3 -> FA2 -> SDPA, RAVEN's own priority
# ---------------------------------------------------------------------------


def test_fa3_is_preferred_and_called_with_ravens_keyword_set(monkeypatch):
    fa3, fa2 = FakeVarlen(3.0), FakeVarlen(2.0)
    install_fake_flash(monkeypatch, fa3=fa3, fa2=fa2)
    q, k, v = cuda_like(*packed_triplet(rows=5, kv_rows=9))

    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5, site=("dit", 7))

    assert fa2.calls == []                      # FA3 wins, FA2 is never asked
    assert torch.equal(out, torch.full_like(q, 3.0))
    (kwargs,) = fa3.calls
    assert set(kwargs) == {"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
                           "seqused_q", "seqused_k", "max_seqlen_q", "max_seqlen_k",
                           "softmax_scale", "causal", "deterministic"}
    assert kwargs["seqused_q"] is None and kwargs["seqused_k"] is None
    assert kwargs["max_seqlen_q"] == 5 and kwargs["max_seqlen_k"] == 9
    assert kwargs["softmax_scale"] == HEAD_DIM ** -0.5
    assert kwargs["causal"] is False and kwargs["deterministic"] is False
    for name, expected in (("cu_seqlens_q", [0, 5]), ("cu_seqlens_k", [0, 9])):
        bounds = kwargs[name]
        assert bounds.dtype == torch.int32
        assert bounds.tolist() == expected
    assert kwargs["q"].shape == q.shape and kwargs["k"].shape == k.shape
    assert cm.raven_attention_backend()["last"]["backend"] == "fa3"


def test_fa2_is_called_with_ravens_keyword_set(monkeypatch):
    fa2 = FakeVarlen(2.0)
    install_fake_flash(monkeypatch, fa3=None, fa2=fa2)
    q, k, v = cuda_like(*packed_triplet(rows=4, kv_rows=11))

    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)

    assert torch.equal(out, torch.full_like(q, 2.0))
    (kwargs,) = fa2.calls
    assert set(kwargs) == {"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
                           "max_seqlen_q", "max_seqlen_k", "dropout_p",
                           "softmax_scale", "causal", "window_size", "deterministic"}
    assert kwargs["dropout_p"] == 0.0
    assert kwargs["window_size"] == (-1, -1)
    assert kwargs["causal"] is False and kwargs["deterministic"] is False
    assert kwargs["max_seqlen_q"] == 4 and kwargs["max_seqlen_k"] == 11
    backend = cm.raven_attention_backend()
    assert backend["last"]["backend"] == "fa2"
    assert backend["resolved"]["available"] == {"fa3": False, "fa2": True}


@pytest.mark.parametrize("disabled,expected", [
    ("FLASH_ATTN_3_AVAILABLE", "fa2"),
    ("FLASH_ATTN_2_AVAILABLE", "fa3"),
])
def test_ravens_env_switches_disable_a_backend(monkeypatch, disabled, expected):
    """The same export pins RAVEN and this lane to the same kernel."""
    install_fake_flash(monkeypatch, fa3=FakeVarlen(3.0), fa2=FakeVarlen(2.0))
    monkeypatch.setenv(disabled, "0")
    cm._reset_attention_backends()
    q, k, v = cuda_like(*packed_triplet())

    raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    backend = cm.raven_attention_backend()
    assert backend["last"]["backend"] == expected
    off = "fa3" if disabled.endswith("3_AVAILABLE") else "fa2"
    assert backend["resolved"]["available"][off] is False
    assert f"disabled by {disabled}=0" == backend["resolved"]["why"][off]


def test_both_env_switches_off_lands_on_sdpa(monkeypatch):
    install_fake_flash(monkeypatch, fa3=FakeVarlen(3.0), fa2=FakeVarlen(2.0))
    monkeypatch.setenv("FLASH_ATTN_3_AVAILABLE", "0")
    monkeypatch.setenv("FLASH_ATTN_2_AVAILABLE", "0")
    cm._reset_attention_backends()
    q, k, v = packed_triplet()

    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert torch.equal(out, raven_sdpa_varlen(q, k, v, q_bounds=[0, q.shape[0]],
                                              k_bounds=[0, k.shape[0]], dropout_p=0.0,
                                              softmax_scale=HEAD_DIM ** -0.5,
                                              causal=False))
    assert cm.raven_attention_backend()["last"]["backend"] == "sdpa"


def test_a_non_integer_env_switch_is_refused(monkeypatch):
    install_fake_flash(monkeypatch, fa3=FakeVarlen(3.0), fa2=FakeVarlen(2.0))
    monkeypatch.setenv("FLASH_ATTN_3_AVAILABLE", "yes")
    cm._reset_attention_backends()
    q, k, v = cuda_like(*packed_triplet())
    with pytest.raises(CausalModelError, match="FLASH_ATTN_3_AVAILABLE"):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)


def test_a_missing_package_falls_back_and_says_so(monkeypatch):
    install_fake_flash(monkeypatch, fa3=None, fa2=None)
    q, k, v = cuda_like(*packed_triplet())
    raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    backend = cm.raven_attention_backend()
    assert backend["last"]["backend"] == "sdpa"
    assert backend["last"]["reason"] == "no flash backend available"
    assert backend["resolved"]["available"] == {"fa3": False, "fa2": False}
    assert all("not importable" in why for why in backend["resolved"]["why"].values())

    # the snapshot is a copy: tampering with it cannot reroute the next call
    backend["resolved"]["available"]["fa3"] = True
    backend["resolved"]["why"]["fa3"] = "tampered"
    fresh = cm.raven_attention_backend()
    assert fresh["resolved"]["available"]["fa3"] is False
    assert "tampered" not in fresh["resolved"]["why"]["fa3"]


@pytest.mark.parametrize("error", [
    RuntimeError("FlashAttention only supports Hopper GPUs or newer."),
    RuntimeError("FlashAttention was not compiled with this head_dim"),
    NotImplementedError("varlen is not supported in this build"),
    TypeError("flash_attn_varlen_func() got an unexpected keyword argument 'seqused_q'"),
])
def test_an_unsupported_kernel_falls_back_to_the_next_backend(monkeypatch, error):
    fa3, fa2 = FakeVarlen(3.0, error=error), FakeVarlen(2.0)
    install_fake_flash(monkeypatch, fa3=fa3, fa2=fa2)
    q, k, v = cuda_like(*packed_triplet())

    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert torch.equal(out, torch.full_like(q, 2.0))
    assert len(fa3.calls) == 1 and len(fa2.calls) == 1

    # retired, so the rest of the rollout does not pay for it again
    raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert len(fa3.calls) == 1 and len(fa2.calls) == 2
    why = cm.raven_attention_backend()["resolved"]["why"]["fa3"]
    assert why.startswith("disabled after an unsupported error")
    assert type(error).__name__ in why


@pytest.mark.parametrize("error", [
    torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"),
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("boom"),
    KeyboardInterrupt(),
])
def test_a_real_kernel_failure_is_never_swallowed(monkeypatch, error):
    """Answering a crashed kernel with a different one would change the numbers."""
    fa3, fa2 = FakeVarlen(3.0, error=error), FakeVarlen(2.0)
    install_fake_flash(monkeypatch, fa3=fa3, fa2=fa2)
    q, k, v = cuda_like(*packed_triplet())

    with pytest.raises(type(error)):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert fa2.calls == []
    assert cm.raven_attention_backend()["resolved"]["available"]["fa3"] is True


def test_an_out_of_memory_message_that_mentions_support_is_still_fatal(monkeypatch):
    fa3 = FakeVarlen(3.0, error=RuntimeError(
        "CUDA out of memory; this shape is not supported at this size"))
    install_fake_flash(monkeypatch, fa3=fa3, fa2=FakeVarlen(2.0))
    q, k, v = cuda_like(*packed_triplet())
    with pytest.raises(RuntimeError, match="out of memory"):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)


def test_every_backend_receives_ravens_layout_untouched(monkeypatch):
    """No silent ``.contiguous()`` in front of the kernel: RAVEN does not either.

    RAVEN hands its varlen call a fused-QKV ``v`` view whenever there is no
    history to merge. Quietly repacking it here would put this lane back on a
    different FA3 kernel path -- the very thing the layout work removed.
    """
    q, k, v = fused_qkv_triplet()
    assert not v.is_contiguous()
    fa3 = FakeVarlen(3.0)
    install_fake_flash(monkeypatch, fa3=fa3, fa2=None)
    cq, ck, cv = cuda_like(q, k, v)

    raven_packed_attention(cq, ck, cv, scale=HEAD_DIM ** -0.5)

    (kwargs,) = fa3.calls
    assert kwargs["q"].is_contiguous() and kwargs["k"].is_contiguous()
    assert tuple(kwargs["q"].stride()) == (HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert not kwargs["v"].is_contiguous()
    assert tuple(kwargs["v"].stride()) == (3 * HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert kwargs["v"].data_ptr() == cv.data_ptr()


def test_fa2_also_receives_the_fused_v_view(monkeypatch):
    q, k, v = fused_qkv_triplet()
    fa2 = FakeVarlen(2.0)
    install_fake_flash(monkeypatch, fa3=None, fa2=fa2)
    raven_packed_attention(*cuda_like(q, k, v), scale=HEAD_DIM ** -0.5)
    (kwargs,) = fa2.calls
    assert not kwargs["v"].is_contiguous()
    assert tuple(kwargs["v"].stride()) == (3 * HEADS * HEAD_DIM, HEAD_DIM, 1)


def test_the_sdpa_fallback_takes_the_fused_v_view_and_returns_row_major():
    """The strided ``v`` is a layout choice, not a numerical one, on this backend."""
    q, k, v = fused_qkv_triplet()
    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert cm.raven_attention_backend()["last"]["backend"] == "sdpa"
    assert out.is_contiguous()
    assert tuple(out.stride()) == (HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert out.shape == q.shape
    # same values as the repacked reference (CPU SDPA is stride-insensitive);
    # the point of passing the view is which kernel FA3 picks, not this
    assert torch.equal(out, raven_packed_attention(q, k, v.contiguous(),
                                                   scale=HEAD_DIM ** -0.5))
    # ... and it is what RAVEN's own fallback produces for that same layout
    assert torch.equal(out, raven_sdpa_varlen(
        q, k, v, q_bounds=[0, q.shape[0]], k_bounds=[0, k.shape[0]],
        dropout_p=0.0, softmax_scale=HEAD_DIM ** -0.5, causal=False))


def test_a_tuple_return_is_normalised_to_the_output(monkeypatch):
    """Some flash_attn_interface builds return ``(out, softmax_lse)``."""
    q, k, v = cuda_like(*packed_triplet(rows=6, kv_rows=6))
    fa3 = FakeVarlen(3.0, result=(torch.full_like(q, 5.0), torch.zeros(6)))
    install_fake_flash(monkeypatch, fa3=fa3, fa2=None)
    out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert torch.equal(out, torch.full_like(q, 5.0))


def test_a_wrong_shaped_kernel_result_is_refused(monkeypatch):
    q, k, v = cuda_like(*packed_triplet(rows=6, kv_rows=6))
    fa3 = FakeVarlen(3.0, result=torch.zeros(3, HEADS, HEAD_DIM, dtype=BF16))
    install_fake_flash(monkeypatch, fa3=fa3, fa2=FakeVarlen(2.0))
    with pytest.raises(CausalModelError, match="expected"):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)


@pytest.mark.parametrize("kind,expected", [
    ("cpu", "not cuda"),
    ("fp32", "is not fp16/bf16"),
    ("wide", "> 256"),
])
def test_ravens_preconditions_route_to_sdpa(monkeypatch, kind, expected):
    """RAVEN asserts cuda / half / head_dim <= 256; this lane routes instead."""
    fa3 = FakeVarlen(3.0)
    install_fake_flash(monkeypatch, fa3=fa3, fa2=None)
    if kind == "cpu":
        q, k, v = packed_triplet()
    elif kind == "fp32":
        q, k, v = cuda_like(*packed_triplet(dtype=torch.float32))
    else:
        q, k, v = cuda_like(*packed_triplet(head_dim=320))

    raven_packed_attention(q, k, v, scale=0.1)
    assert fa3.calls == []
    last = cm.raven_attention_backend()["last"]
    assert last["backend"] == "sdpa"
    assert expected in last["reason"]


def test_sdpa_fallback_forces_the_reduction_off_and_restores_it():
    """The vr finding, as an invariant: the ambient switch cannot move us."""
    q, k, v = packed_triplet(rows=6, kv_rows=14)
    reference = raven_sdpa_varlen(q, k, v, q_bounds=[0, 6], k_bounds=[0, 14],
                                  dropout_p=0.0, softmax_scale=HEAD_DIM ** -0.5,
                                  causal=False)
    for ambient in (True, False):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(ambient)
        out = raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
        assert torch.equal(out, reference)
        # restored, not left at False for the rest of the process
        assert bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()) is ambient


def test_the_reduction_switch_is_restored_when_sdpa_raises(monkeypatch):
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

    def explode(*args, **kwargs):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(cm.F, "scaled_dot_product_attention", explode)
    q, k, v = packed_triplet()
    with pytest.raises(RuntimeError, match="exploded"):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()) is True


def test_the_backend_diagnostic_is_read_only_and_out_of_the_state_dict(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
        compute_dtype=BF16,
    )

    backend = cm.raven_attention_backend()
    assert backend["last"]["backend"] == "sdpa"          # no CUDA on this box
    assert backend["last"]["site"] == ("dit", NUM_LAYERS - 1)
    assert backend["last"]["rows"] == layout.chunks[0].rows
    assert backend["last"]["kv_rows"] == layout.chunks[0].rows + layout.text_len
    # nothing was imported: a CPU box cannot use FA, so resolution never ran
    assert backend["resolved"] is None

    # a snapshot, not the live state
    backend["last"]["backend"] = "tampered"
    assert cm.raven_attention_backend()["last"]["backend"] == "sdpa"

    # and none of it is a module attribute
    assert not any("backend" in name for name in causal.state_dict())
    assert not hasattr(causal.blocks[0].attn, "_attention_impl")


def test_the_resolution_is_cached_across_calls(monkeypatch):
    calls = []
    real_import = cm.importlib.import_module

    def counting_import(name, *args, **kwargs):
        calls.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cm.importlib, "import_module", counting_import)
    q, k, v = packed_triplet()
    for _ in range(3):
        raven_packed_attention(q, k, v, scale=HEAD_DIM ** -0.5)
    assert calls.count("flash_attn_interface") <= 1
    assert calls.count("flash_attn") <= 1


# ---------------------------------------------------------------------------
# 1d. timesteps: float32 arithmetic, because RAVEN's is
# ---------------------------------------------------------------------------
#
# RAVEN keeps timesteps in fp32 tensors from the rollout onwards:
# ``torch.tensor([sigma])`` -> ``repo_t = unique_timesteps.float()`` ->
# ``h3_t = where(repo_t == 0, clean_by_tag[tags], 1.0 - repo_t)``, an fp32
# subtraction whose result feeds both the time embedder and the clean mix
# ``t * x + (1 - t) * eps``. Spelling it ``1.0 - float(sigma)`` in Python keeps
# the schedule's double precision one step too long; the embedding probe
# measured ~1e-8 on ``time_embedder.out`` from exactly that.


def raven_h3_timestep(sigma):
    """``1.0 - repo_t`` as ``MiniMaxH3X0Model.forward`` evaluates it."""
    return 1.0 - torch.tensor([float(sigma)], dtype=torch.float32)


#: sigmas whose fp32 and double ``1 - sigma`` differ (most of them do)
SPLITTING_SIGMAS = [0.6, 0.3, 0.42, 0.001, 0.7331]


@pytest.mark.parametrize("sigma", SPLITTING_SIGMAS)
def test_the_h3_timestep_is_evaluated_in_float32(sigma):
    ours = cm._fp32_one_minus(sigma)
    assert ours == float(raven_h3_timestep(sigma)[0])
    # ... and that is a different number from the Python-double spelling
    assert ours != 1.0 - sigma
    # the carrier is exact: rebuilding the tensor cannot round it again
    assert torch.equal(torch.tensor([ours], dtype=torch.float32),
                       raven_h3_timestep(sigma))


def test_fp32_scalar_is_an_exact_carrier():
    assert cm._fp32_scalar(0.999) == float(torch.tensor(0.999, dtype=torch.float32))
    assert cm._fp32_scalar(0.999) != 0.999          # 0.999 is not an fp32 number
    assert cm._fp32_scalar(0.5) == 0.5              # ... but 0.5 is
    once = cm._fp32_scalar(0.999)
    assert cm._fp32_scalar(once) == once            # idempotent


@pytest.mark.parametrize("sigmas", [(0.6, 0.3), (0.42, 0.42), (0.001, 0.7331)])
def test_the_time_embedder_sees_ravens_timesteps(tiny_bf16_models, monkeypatch, sigmas):
    """The tensor the embedder is handed, bit for bit against RAVEN's."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)

    seen = []
    original = causal.time_embedder.forward
    monkeypatch.setattr(causal.time_embedder, "forward",
                        lambda t: (seen.append(t.clone()), original(t))[1])

    video_sigma, audio_sigma = sigmas
    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="noise", video_sigma=video_sigma, audio_sigma=audio_sigma,
        compute_dtype=BF16)

    (t_vals,) = seen
    expected = torch.unique(torch.cat([raven_h3_timestep(audio_sigma),
                                       raven_h3_timestep(video_sigma)]))
    assert t_vals.dtype == torch.float32
    assert torch.equal(t_vals, expected)

    # the old spelling would have produced a different input tensor
    doubles = torch.tensor(sorted({1.0 - video_sigma, 1.0 - audio_sigma}),
                           dtype=torch.float32)
    if float(expected[0]) != float(doubles[0]):
        assert not torch.equal(t_vals, doubles)


def test_the_clean_timestep_is_still_0999(tiny_bf16_models, monkeypatch):
    """fp32 arithmetic must not move the attested condition timestep."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)

    seen = []
    original = causal.time_embedder.forward
    monkeypatch.setattr(causal.time_embedder, "forward",
                        lambda t: (seen.append(t.clone()), original(t))[1])

    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="clean", compute_dtype=BF16,
        video_eps=torch.zeros_like(video), audio_eps=torch.zeros_like(audio))

    (t_vals,) = seen
    assert torch.equal(t_vals, torch.tensor([0.999], dtype=torch.float32))
    assert float(t_vals[0]) == cm._fp32_scalar(0.999)


def test_the_text_prefill_also_embeds_the_fp32_0999(tiny_bf16_models, monkeypatch):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)
    seen = []
    original = causal.time_embedder.forward
    monkeypatch.setattr(causal.time_embedder, "forward",
                        lambda t: (seen.append(t.clone()), original(t))[1])

    prefilled_cache(causal, context)

    (t_vals,) = seen
    assert torch.equal(t_vals, torch.tensor([0.999], dtype=torch.float32))


def test_the_clean_mix_uses_ravens_fp32_coefficients(tiny_bf16_models, monkeypatch):
    """``t * x0 + (1 - t) * eps`` with ``1 - t`` in fp32, not ``fp32(0.001)``.

    Caught at the patch projection, which is the first thing to see the mixed
    rows. The two spellings differ by 1.3e-5 of each other on the eps term.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    torch.manual_seed(77)
    eps_v = torch.randn(video.shape).to(BF16)
    eps_a = torch.randn(audio.shape).to(BF16)

    rows = []
    original = causal.video_patch_proj.forward
    monkeypatch.setattr(causal.video_patch_proj, "forward",
                        lambda x: (rows.append(x.clone()), original(x))[1])

    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="clean", compute_dtype=BF16,
        video_eps=eps_v, audio_eps=eps_a)

    (seen,) = rows
    t = cm._fp32_scalar(CLEAN_TIMESTEP_TEXT)
    raven = patchify_video(
        (t * video.to(torch.float32) + cm._fp32_one_minus(t) * eps_v.to(torch.float32)),
        causal.patch_size)
    assert torch.equal(seen, raven)

    doubled = patchify_video(
        (0.999 * video.to(torch.float32) + (1.0 - 0.999) * eps_v.to(torch.float32)),
        causal.patch_size)
    assert not torch.equal(seen, doubled)          # the spelling is observable
    assert torch.allclose(seen, doubled, rtol=1e-6, atol=1e-7)


def test_audio_sigma_from_video_already_runs_in_float32(tiny_bf16_models):
    """No change needed here: it was always an fp32 tensor computation.

    ``audio_sigma_from_video`` clamps and shifts on an fp32 tensor with the
    official ``time_shift_sigma``, exactly as the dense ``_forward`` does with
    ``(timestep / 1000).float().clamp(min=1e-6)``, and returns the fp32 result
    as an exact carrier.
    """
    from comfy.ldm.minimax.model import time_shift_sigma

    _, causal = tiny_bf16_models
    for sigma in SPLITTING_SIGMAS:
        ours = causal.audio_sigma_from_video(sigma)
        expected = float(time_shift_sigma(
            torch.as_tensor(sigma, dtype=torch.float32).clamp(min=1e-6),
            causal.sigma_shift_video, causal.sigma_shift_audio))
        assert ours == expected
        assert cm._fp32_scalar(ours) == ours       # already an fp32 number


def test_velocity_to_x0_matches_ravens_conversion():
    """RAVEN casts the timestep to ``x_t``'s dtype and subtracts there.

    With an fp32 ``x_t`` and an fp32-exact ``t`` -- which is what this module
    now produces everywhere -- the double subtraction is a single rounding onto
    the same fp32 number, so no change is warranted here.
    """
    torch.manual_seed(88)
    x_t = torch.randn(4, 6)
    velocity = torch.randn(4, 6)
    for sigma in SPLITTING_SIGMAS:
        t = cm._fp32_one_minus(sigma)
        ours = velocity_to_x0(x_t, velocity, t)
        cond_t = torch.tensor(t, dtype=x_t.dtype)      # RAVEN's cast
        raven = x_t + (1 - cond_t) * velocity
        assert torch.equal(ours, raven)


# ---------------------------------------------------------------------------
# 2. AdaLN: one fp32 SiLU per forward, shared, and no second one
# ---------------------------------------------------------------------------


def test_adaln_input_is_an_fp32_silu_cast_once(tiny_bf16_models):
    _, causal = tiny_bf16_models
    t_emb, adaln_input = causal._causal_time_embeddings([0.999, 0.4], "cpu", BF16)

    native = causal.time_embedder(torch.tensor([0.999, 0.4], dtype=torch.float32))
    assert native.dtype == torch.float32
    assert adaln_input.dtype == BF16
    assert torch.equal(adaln_input, F.silu(native).to(BF16))
    # ... and that is not what a SiLU at the module dtype produces
    assert not torch.equal(adaln_input, F.silu(native.to(BF16)))
    # the dense lane's t_emb is unchanged: still the compute-dtype cast
    assert torch.equal(t_emb, native.to(BF16))


def test_adaln_curve_form_keeps_the_official_fp32_input(comfyui_on_syspath):
    """The curve checkpoint folds the SiLU into its table; RAVEN has no such form."""
    _, causal = build_models(config=dict(TINY_CONFIG, adaln_curve_grid=8), dtype=BF16)
    t_emb, adaln_input = causal._causal_time_embeddings([0.5], "cpu", BF16)
    assert t_emb.dtype == torch.float32
    assert adaln_input is t_emb


def test_causal_forward_does_not_flip_apply_silu(tiny_bf16_models):
    """``apply_silu`` is read by the dense lane too, so it is never mutated."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
        compute_dtype=BF16,
    )
    for block in causal.blocks:
        assert block.adaln_proj.apply_silu is True
    assert causal.final_layer.adaln_proj.apply_silu is True


def test_block_adaln_params_come_from_the_shared_input(tiny_bf16_models):
    _, causal = tiny_bf16_models
    _, adaln_input = causal._causal_time_embeddings([0.999], "cpu", BF16)
    adaln = causal.blocks[1].adaln_proj

    projected = adaln.linear(adaln_input)
    projected = projected.view(projected.shape[0] * adaln.modalities,
                               adaln.expand * adaln.hidden)
    expected = projected.chunk(adaln.expand, dim=-1)
    ours = cm._raven_adaln_params(adaln, adaln_input)
    assert len(ours) == 6
    for a, b in zip(ours, expected):
        assert torch.equal(a, b)
    # the module's own forward would have SiLU'd again
    doubled = adaln(adaln_input)
    assert not torch.equal(ours[0], doubled[0])


def test_causal_block_needs_the_shared_adaln_input(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    rows = layout.chunks[0].rows
    x = torch.zeros(rows, HIDDEN, dtype=BF16)
    rope_table = rope_rotation_table(
        causal.rope_freqs(layout.chunk_position_ids(0), x.device), BF16)
    with pytest.raises(CausalModelError, match="pre-SiLU adaln_input"):
        causal.blocks[0](x, torch.zeros(1, TINY_CONFIG["time_embed_dim"], dtype=BF16),
                         [(0, rows, 0)], rope_table, cache=cache)


# ---------------------------------------------------------------------------
# 3. gate: RAVEN's expression, not addcmul_
# ---------------------------------------------------------------------------


def test_gate_matches_the_raven_expression_bitwise():
    torch.manual_seed(4)
    rows = 64
    x = torch.randn(rows, HIDDEN).to(BF16)
    other = torch.randn(rows, HIDDEN).to(BF16)
    gate = torch.randn(6, HIDDEN).to(BF16)
    segments = [(0, 20, 2), (20, rows, 5)]

    indices = torch.empty(rows, dtype=torch.long)
    for a, b, row in segments:
        indices[a:b] = row
    expected = raven_modulate_gate(x.clone(), gate, other, indices, dtype=BF16)

    ours = cm._raven_mod_gate(x.clone(), gate, other, segments)
    assert torch.equal(ours, expected)


def test_gate_is_not_the_fused_addcmul():
    """The two spellings genuinely disagree at BF16 -- this is the 0.000886."""
    from comfy.ldm.minimax.model import _mod_gate

    torch.manual_seed(5)
    rows = 64
    x = torch.randn(rows, HIDDEN).to(BF16)
    other = torch.randn(rows, HIDDEN).to(BF16)
    gate = torch.randn(3, HIDDEN).to(BF16)
    segments = [(0, rows, 1)]

    fused = _mod_gate(x.clone(), gate, other, segments)
    ours = cm._raven_mod_gate(x.clone(), gate, other, segments)
    assert not torch.equal(fused, ours)
    assert torch.allclose(fused.float(), ours.float(), atol=1e-1)


def test_gate_handles_a_per_row_index_tensor():
    torch.manual_seed(6)
    rows = 16
    x = torch.randn(rows, HIDDEN).to(BF16)
    other = torch.randn(rows, HIDDEN).to(BF16)
    gate = torch.randn(4, HIDDEN).to(BF16)
    indices = torch.randint(0, 4, (rows,))
    ours = cm._raven_mod_gate(x.clone(), gate, other, [(0, rows, indices)])
    assert torch.equal(ours, raven_modulate_gate(x.clone(), gate, other, indices,
                                                 dtype=BF16))


def test_scale_shift_is_still_the_official_one():
    """Audited identical, so the causal lane keeps calling upstream's."""
    from comfy.ldm.minimax.model import _mod_scale_shift

    torch.manual_seed(7)
    rows = 48
    x = torch.randn(rows, HIDDEN).to(BF16)
    shift = torch.randn(3, HIDDEN).to(BF16)
    scale = torch.randn(3, HIDDEN).to(BF16)
    indices = torch.full((rows,), 2, dtype=torch.long)
    assert torch.equal(
        _mod_scale_shift(x.clone(), shift, scale, [(0, rows, 2)]),
        raven_modulate_scale_shift(x.clone(), shift, scale, indices, dtype=BF16),
    )


# ---------------------------------------------------------------------------
# 4. final layer
# ---------------------------------------------------------------------------


def test_causal_final_layer_matches_raven(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    chunk = layout.chunks[0]
    audio_n, rows = chunk.audio_rows, chunk.rows
    torch.manual_seed(8)
    h = torch.randn(rows, HIDDEN).to(BF16)
    _, adaln_input = causal._causal_time_embeddings([0.4, 0.9], "cpu", BF16)

    video, audio = causal._causal_final_layer(
        h.clone(), adaln_input, (audio_n, rows, 1), (0, audio_n, 0))

    final = causal.final_layer
    projected = final.adaln_proj.linear(adaln_input)
    projected = projected.view(projected.shape[0], 2 * HIDDEN)
    shift, scale = projected.chunk(2, dim=-1)
    inverse = torch.empty(rows, dtype=torch.long)
    inverse[:audio_n] = 0
    inverse[audio_n:] = 1
    x = raven_modulate_scale_shift(final.norm(h.clone()), shift, scale, inverse,
                                   dtype=BF16).to(torch.float32)
    assert video.dtype == torch.float32 and audio.dtype == torch.float32
    assert torch.equal(video, final.video_out(x[audio_n:]))
    assert torch.equal(audio, final.audio_out(x[:audio_n]))


def test_causal_final_layer_differs_from_the_official_head(tiny_bf16_models):
    """Same modules, different AdaLN input: the BF16 SiLU is what moves."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    chunk = layout.chunks[0]
    audio_n, rows = chunk.audio_rows, chunk.rows
    torch.manual_seed(9)
    h = torch.randn(rows, HIDDEN).to(BF16)
    t_emb, adaln_input = causal._causal_time_embeddings([0.4], "cpu", BF16)

    ours = causal._causal_final_layer(h.clone(), adaln_input,
                                      (audio_n, rows, 0), (0, audio_n, 0))
    theirs = causal.final_layer(h.clone(), t_emb, (audio_n, rows, 0), (0, audio_n, 0))
    assert not torch.equal(ours[0], theirs[0])
    assert torch.allclose(ours[0], theirs[0], atol=5e-2)


# ---------------------------------------------------------------------------
# 5. text refiner
# ---------------------------------------------------------------------------


def test_text_refiner_matches_the_raven_replay(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)
    ours = causal._causal_refine_text(context[0], transformer_options={})
    assert torch.equal(ours, raven_refiner_replay(causal, context[0], BF16))


def test_text_refiner_is_close_to_the_official_one_and_ambient_independent(tiny_bf16_models):
    """Ours tracks RAVEN, not whatever backend or knob Comfy happens to have.

    Bitwise equality with RAVEN is pinned by
    ``test_text_refiner_matches_the_raven_replay``. Against *Comfy's* refiner
    only closeness can be claimed: ``optimized_attention`` picks its own backend
    (on this CPU box it binds ``attention_sub_quad``, on a CUDA box a fused
    kernel), and ``comfy.model_management`` flips
    ``allow_fp16_bf16_reduction_math_sdp`` on process-wide. Neither may move
    this lane's numbers.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)

    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    ours_reduced = causal._causal_refine_text(context[0], transformer_options={})
    official = causal.token_refiner(causal.condition_proj(context[0]))
    with ravens_math_sdp():
        ours_exact = causal._causal_refine_text(context[0], transformer_options={})

    assert torch.equal(ours_reduced, ours_exact)  # invariant to Comfy's switch
    assert ours_reduced.shape == official.shape
    assert torch.allclose(ours_reduced.float(), official.float(), atol=5e-2)


@pytest.mark.parametrize("prerefined", [False, True])
def test_prefill_text_never_writes_into_the_callers_context(tiny_bf16_models, prerefined):
    """The block stack accumulates in place; the row buffer must be ours.

    A pre-refined context already in the compute dtype survives ``.to(dtype)``
    as the same tensor, so without a copy the first block's gated residual would
    land in the caller's conditioning -- silently, and visibly only on the next
    rollout that reuses it.
    """
    official, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)
    if prerefined:
        context = official.preprocess_text_embeds(context)
        assert context.shape[-1] == HIDDEN
    before = context.clone()

    first = prefilled_cache(causal, context)
    assert torch.equal(context, before)

    # ... and a second prefill of the same tensor sees the same rows
    second = prefilled_cache(causal, context)
    for layer in range(NUM_LAYERS):
        assert torch.equal(first.retained(layer)[0], second.retained(layer)[0])


def test_prefill_text_uses_the_raven_refiner(tiny_bf16_models):
    """The K/V the cache holds are the ones RAVEN's refiner would have produced."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    _, _, context = bf16_chunk_inputs(layout)

    cache = prefilled_cache(causal, context)
    refined = raven_refiner_replay(causal, context[0], BF16)
    pre_cache = prefilled_cache(causal, refined.unsqueeze(0))
    for layer in range(NUM_LAYERS):
        assert torch.equal(cache.retained(layer)[0], pre_cache.retained(layer)[0])
        assert torch.equal(cache.retained(layer)[1], pre_cache.retained(layer)[1])


# ---------------------------------------------------------------------------
# 6. end to end
# ---------------------------------------------------------------------------


def test_block_stack_matches_a_raven_replay(tiny_bf16_models):
    """The hidden state after all blocks, before the head rounds anything away.

    This is where the gate rounding is visible: by the time the final layer and
    the BF16 velocity cast are done, a 3-block tiny model has often absorbed it.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)

    h, adaln_input, combined, _, rope_table = raven_chunk_setup(
        causal, layout, 0, video, audio, 0.4, 0.7, BF16)
    audio_n, rows = layout.chunks[0].audio_rows, layout.chunks[0].rows
    mod_segments = [(0, audio_n, int(combined[0])), (audio_n, rows, int(combined[-1]))]

    ours = causal._run_blocks(h.clone(), None, mod_segments, rope_table, {},
                              cache, False, adaln_input=adaln_input)
    theirs = raven_block_stack_replay(causal, h.clone(), adaln_input, combined,
                                      rope_table, cache, BF16)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("index", [0, 1])
def test_chunk_forward_matches_a_full_raven_replay(tiny_bf16_models, index):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    video, audio, context = video.to(BF16), audio.to(BF16), context.to(BF16)
    cache = prefilled_cache(causal, context)

    generator = torch.Generator().manual_seed(11)
    for previous in range(index):
        v = layout.video_chunk_latent(video, previous)
        a = layout.audio_chunk_latent(audio, previous)
        causal.forward_chunk(
            video_latent=v, audio_latent=a, layout=layout, chunk_index=previous,
            cache=cache, role="clean", compute_dtype=BF16,
            video_eps=torch.randn(v.shape, generator=generator).to(BF16),
            audio_eps=torch.randn(a.shape, generator=generator).to(BF16),
        )

    video_x = layout.video_chunk_latent(video, index)
    audio_x = layout.audio_chunk_latent(audio, index)
    sigma_v, sigma_a = 0.6, 0.3
    ours = causal.forward_chunk(
        video_latent=video_x, audio_latent=audio_x, layout=layout,
        chunk_index=index, cache=cache, role="noise",
        video_sigma=sigma_v, audio_sigma=sigma_a, compute_dtype=BF16,
    )
    theirs = raven_chunk_replay(causal, layout, index, video_x, audio_x,
                                1.0 - sigma_v, 1.0 - sigma_a, cache, BF16)
    for a, b in zip(ours, theirs):
        assert a.shape == b.shape
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 6b. the fp32 island: what leaves forward_chunk, and what stays BF16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["noise", "clean"])
@pytest.mark.parametrize("latent_dtype", [BF16, torch.float32])
def test_the_velocity_leaves_in_the_heads_fp32(tiny_bf16_models, role, latent_dtype):
    """``FinalLayer.video_out``/``audio_out`` are fp32 and stay fp32.

    Both roles, and independent of the latent's dtype: the old
    ``velocity.to(video_latent.dtype)`` was invisible for an fp32 ``LATENT`` and
    cost one bf16 ULP (~1.7e-3 relative) for a BF16 one.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    video, audio = video.to(latent_dtype), audio.to(latent_dtype)
    cache = prefilled_cache(causal, context)

    extra = {}
    if role == "noise":
        extra = dict(video_sigma=0.6, audio_sigma=0.3)
    else:
        extra = dict(video_eps=torch.zeros_like(video),
                     audio_eps=torch.zeros_like(audio))
    out = causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=cache, role=role, compute_dtype=BF16, **extra)

    assert [t.dtype for t in out] == [torch.float32, torch.float32]
    assert out[0].shape[2:] == (layout.chunks[0].video_latents,
                                layout.latent_h, layout.latent_w)
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
    # x0 stays in the fp32 the sampler now steps in
    assert velocity_to_x0(video.to(torch.float32), out[0], 0.4).dtype == torch.float32


def test_the_cacheless_lane_also_returns_fp32(tiny_bf16_models):
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, _ = bf16_chunk_inputs(layout)
    out = causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=None, role="noise", video_sigma=0.6, audio_sigma=0.3,
        compute_dtype=BF16)
    assert [t.dtype for t in out] == [torch.float32, torch.float32]


def test_the_cache_stays_in_the_compute_dtype(tiny_bf16_models):
    """Only the *output* side moved to fp32; the K/V the next chunk reads did not."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)
    cache = prefilled_cache(causal, context)
    for layer in range(NUM_LAYERS):
        keys, values = cache.retained(layer)
        assert keys.dtype == BF16 and values.dtype == BF16

    causal.forward_chunk(
        video_latent=video.to(torch.float32), audio_latent=audio.to(torch.float32),
        layout=layout, chunk_index=0, cache=cache, role="clean",
        compute_dtype=BF16,
        video_eps=torch.zeros_like(video, dtype=torch.float32),
        audio_eps=torch.zeros_like(audio, dtype=torch.float32))
    for layer in range(NUM_LAYERS):
        keys, values = cache.retained(layer)
        assert keys.dtype == BF16 and values.dtype == BF16


def test_an_fp32_latent_from_the_previous_step_re_enters_cleanly(tiny_bf16_models):
    """The next step hands back fp32; the patch projections are fp32 anyway.

    ``forward_chunk`` embeds through ``patchify_video(x.to(torch.float32))`` and
    ``pack_audio``, i.e. the same fp32 island the dense forward uses, and casts
    the projection output to the compute dtype before the blocks. So an fp32
    latent is what those projections already expect -- and a BF16 one embeds to
    exactly the same rows once upcast.
    """
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)

    first = causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=prefilled_cache(causal, context), role="noise",
        video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)
    next_video = velocity_to_x0(video.to(torch.float32), first[0], 0.4)
    next_audio = velocity_to_x0(audio.to(torch.float32), first[1], 0.7)
    assert next_video.dtype == torch.float32 and next_audio.dtype == torch.float32

    second = causal.forward_chunk(
        video_latent=next_video, audio_latent=next_audio, layout=layout,
        chunk_index=0, cache=prefilled_cache(causal, context), role="noise",
        video_sigma=0.4, audio_sigma=0.2, compute_dtype=BF16)
    assert [t.dtype for t in second] == [torch.float32, torch.float32]
    assert torch.isfinite(second[0]).all()

    # an exactly-representable fp32 latent embeds like its BF16 self
    same = causal.forward_chunk(
        video_latent=video.to(torch.float32), audio_latent=audio.to(torch.float32),
        layout=layout, chunk_index=0, cache=prefilled_cache(causal, context),
        role="noise", video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)
    for a, b in zip(first, same):
        assert torch.equal(a, b)


def test_dense_forward_is_still_bit_identical_at_bf16(tiny_bf16_models):
    official, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = random_inputs(layout)
    args = dict(x=[video.to(BF16), audio.to(BF16)], timestep=torch.tensor([700.0]),
                context=context.to(BF16))
    with torch.no_grad():
        theirs = official(**args)
        ours = causal(**args)
    for a, b in zip(ours, theirs):
        assert torch.equal(a, b)


def test_cacheless_chunk_forward_stays_on_the_official_operators(tiny_bf16_models):
    """No cache means no causal lane: the dense operators, restricted to a chunk."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, _ = bf16_chunk_inputs(layout)
    with CallCounter(comfy_minimax, "optimized_attention") as counter:
        causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=None, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16,
        )
    assert counter.calls == TINY_CONFIG["num_layers"]


# ---------------------------------------------------------------------------
# 7. every weight still goes through its module's __call__
# ---------------------------------------------------------------------------


def tiny_lora_inventory():
    """The RAVEN LoRA target inventory, evaluated at the tiny model's shapes."""
    from raven_streaming.lora import RavenBaseConfig

    return RavenBaseConfig(
        hidden_size=HIDDEN,
        num_layers=TINY_CONFIG["num_layers"],
        token_refiner_num_layers=TINY_CONFIG["token_refiner_num_layers"],
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_hidden_size=TINY_CONFIG["ffn_hidden_size"],
        latents_dim=TINY_CONFIG["latents_dim"],
        audio_latents_dim=TINY_CONFIG["audio_latents_dim"],
        text_dim=TINY_CONFIG["text_dim"],
        timestep_input_dim=TINY_CONFIG["timestep_input_dim"],
        time_embed_hidden_size=TINY_CONFIG["time_embed_hidden_size"],
        time_embed_dim=TINY_CONFIG["time_embed_dim"],
    ).modules()


def run_one_rollout_step(model, layout, hook_kind):
    """Prefill + one noise chunk with a counting hook on every LoRA target."""
    inventory = tiny_lora_inventory()
    named = dict(model.named_modules())
    missing = sorted(path for path in inventory if path not in named)
    assert missing == [], f"inventory paths absent from the model: {missing}"

    counts = {path: 0 for path in inventory}
    handles = []
    for path in inventory:
        def make(path):
            def hook(module, args, output=None):
                counts[path] += 1
            return hook
        module = named[path]
        if hook_kind == "forward":
            handles.append(module.register_forward_hook(make(path)))
        else:
            handles.append(module.register_forward_pre_hook(make(path)))
    try:
        video, audio, context = bf16_chunk_inputs(layout)
        cache = prefilled_cache(model, context)
        model.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=cache, role="noise", video_sigma=0.6, audio_sigma=0.3,
            compute_dtype=BF16,
        )
    finally:
        for handle in handles:
            handle.remove()
    return counts, inventory


def test_every_lora_target_module_is_called_on_the_causal_lane(tiny_bf16_models):
    """A LoRA residual is a forward hook: a bypassed ``__call__`` is a lost adapter."""
    from raven_streaming.lora import CATEGORY_ORDER, category_counts

    _, causal = tiny_bf16_models
    counts, inventory = run_one_rollout_step(causal, tiny_layout(), "forward")

    silent = sorted(path for path, n in counts.items() if n == 0)
    assert silent == [], f"modules the causal lane reached without __call__: {silent}"

    # every category of the published 266-module adapter is represented
    covered = category_counts(inventory[path] for path in counts if counts[path])
    assert set(CATEGORY_ORDER) == {c for c, n in covered.items() if n}
    expected = category_counts(inventory.values())
    assert covered == expected


def test_partial_offload_pre_hooks_fire_for_every_target(tiny_bf16_models):
    """Comfy's partial offload casts weights inside ``forward``; nothing may skip it."""
    _, causal = tiny_bf16_models
    counts, _ = run_one_rollout_step(causal, tiny_layout(), "pre")
    silent = sorted(path for path, n in counts.items() if n == 0)
    assert silent == []


def test_lora_style_hook_actually_changes_the_causal_output(tiny_bf16_models):
    """End to end: a residual on the AdaLN and attention Linears must be felt."""
    _, causal = tiny_bf16_models
    layout = tiny_layout()
    video, audio, context = bf16_chunk_inputs(layout)

    baseline = causal.forward_chunk(
        video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
        cache=prefilled_cache(causal, context), role="noise",
        video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)

    named = dict(causal.named_modules())
    handles = []
    for path in ("blocks.0.adaln_proj.linear", "blocks.0.attn.out_proj",
                 "final_layer.adaln_proj.linear", "token_refiner.blocks.0.attn.qkv_proj"):
        handles.append(named[path].register_forward_hook(
            lambda module, args, output: output * 1.01))
    try:
        patched = causal.forward_chunk(
            video_latent=video, audio_latent=audio, layout=layout, chunk_index=0,
            cache=prefilled_cache(causal, context), role="noise",
            video_sigma=0.6, audio_sigma=0.3, compute_dtype=BF16)
    finally:
        for handle in handles:
            handle.remove()

    assert not torch.equal(baseline[0], patched[0])
    assert not torch.equal(baseline[1], patched[1])
