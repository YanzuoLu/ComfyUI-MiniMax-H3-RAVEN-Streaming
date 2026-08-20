"""Pins RAVEN residual behaviour against the *real* comfy.ops / H3 modules.

Skipped unless a ComfyUI checkout is available (``COMFYUI_PATH`` /
``COMFYUI_UPSTREAM_PATH`` / ``.cache/upstream/ComfyUI``) with its dependencies
importable.

The load-bearing fact here: ``comfy.ldm.minimax.model.MLP`` calls
``comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`` instead of
``self.fc2(...)``. For a non-quantised weight that helper falls back to
``linear(act(x))`` - i.e. it still goes through ``Linear.__call__``, so the
forward hook fires. For an INT8 ``QuantizedTensor`` weight it dispatches to a
fused kernel and never calls the module, which is why attaching to a quantised
weight is refused at attach time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from raven_streaming import runtime_linear as rrl  # noqa: E402

try:  # the shared fixture helper, when running under pytest
    from conftest import find_upstream_comfyui
except Exception:  # noqa: BLE001 - direct execution fallback

    def find_upstream_comfyui():
        for var in ("COMFYUI_PATH", "COMFYUI_UPSTREAM_PATH"):
            value = os.environ.get(var)
            if value and (Path(value) / "comfy").is_dir():
                return Path(value)
        default = ROOT / ".cache" / "upstream" / "ComfyUI"
        return default if (default / "comfy").is_dir() else None


@pytest.fixture(scope="module")
def comfy():
    path = find_upstream_comfyui()
    if path is None:
        pytest.skip("no ComfyUI checkout (set COMFYUI_PATH)")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        import comfy.ops  # noqa: F401
        import comfy.ldm.minimax.model  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - missing optional deps
        pytest.skip("cannot import ComfyUI: {}: {}".format(type(exc).__name__, exc))
    return sys.modules["comfy"]


def _init_linear(lin, shape, gen, dtype):
    if getattr(lin, "weight", None) is None:  # aimdo lazy init
        lin.weight = torch.nn.Parameter(torch.empty(shape, dtype=dtype))
    with torch.no_grad():
        lin.weight.copy_((torch.randn(shape, generator=gen) * 0.05).to(dtype))
    lin.weight.requires_grad_(False)


def _build_mlp(comfy, hidden=32, ffn=24, dtype=torch.bfloat16, seed=0):
    import comfy.ldm.minimax.model as mm

    gen = torch.Generator().manual_seed(seed)
    mlp = mm.MLP(hidden, ffn, dtype=dtype, device=torch.device("cpu"),
                 operations=comfy.ops.manual_cast)
    _init_linear(mlp.fc1, (ffn * 2, hidden), gen, dtype)
    _init_linear(mlp.fc2, (hidden, ffn), gen, dtype)
    return mlp


def test_bf16_mlp_swiglu_path_triggers_both_hooks(comfy):
    """linear_input_act must not bypass the fc2 residual on a normal weight."""
    import comfy.ops as ops

    hidden, ffn, rank = 32, 24, 4
    mlp = _build_mlp(comfy, hidden, ffn)
    gen = torch.Generator().manual_seed(11)
    a1 = torch.randn(rank, hidden, generator=gen) * 0.1
    b1 = torch.randn(ffn * 2, rank, generator=gen) * 0.1
    a2 = torch.randn(rank, ffn, generator=gen) * 0.1
    b2 = torch.randn(hidden, rank, generator=gen) * 0.1

    rrl.attach_residual(mlp.fc1, a1, b1, path="mlp.fc1", alpha=128.0, rank=rank)
    rrl.attach_residual(mlp.fc2, a2, b2, path="mlp.fc2", alpha=128.0, rank=rank)
    scale = 128.0 / rank

    x = torch.randn(7, hidden, dtype=torch.bfloat16)
    with torch.no_grad():
        out = mlp(x)

    hook1 = getattr(mlp.fc1, rrl.HOOK_ATTR)
    hook2 = getattr(mlp.fc2, rrl.HOOK_ATTR)
    assert hook1.calls == 1, "fc1 hook did not run"
    assert hook2.calls == 1, "fc2 hook did not run: linear_input_act bypassed Linear.__call__"

    # reference: base output + PEFT residual at each stage, bf16 in/out
    def peft(base, xin, a, b):
        return (base.float() + torch.nn.functional.linear(
            torch.nn.functional.linear(xin.float(), a), b) * scale).to(base.dtype)

    with torch.no_grad():
        h = peft(torch.nn.functional.linear(x, mlp.fc1.weight), x, a1, b1)
        act = ops.INPUT_ACT_EAGER["swiglu"](h)
        expected = peft(torch.nn.functional.linear(act, mlp.fc2.weight), act, a2, b2)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)


def test_bf16_mlp_without_lora_is_unchanged(comfy):
    mlp = _build_mlp(comfy)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        before = mlp(x)
    rank = 4
    a = torch.zeros(rank, 24)
    b = torch.zeros(32, rank)
    spec, _ = rrl.attach_residual(mlp.fc2, a, b, path="mlp.fc2", alpha=128.0, rank=rank)
    spec.strength = 0.0  # disabled residual must be a no-op
    with torch.no_grad():
        after = mlp(x)
    assert torch.equal(before, after)


def test_quantized_weight_is_still_refused(comfy):
    """The fused INT8 path in linear_input_act would skip the hook -> refuse.\n"""
    import comfy.ops as ops

    lin = ops.manual_cast.Linear(8, 4, bias=False, dtype=torch.bfloat16,
                                 device=torch.device("cpu"))
    if getattr(lin, "weight", None) is None:
        lin.weight = torch.nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16))

    class FakeQuantized(torch.Tensor):
        _layout_cls = "TensorWiseINT8Layout"

    lin.weight = torch.nn.Parameter(
        torch.zeros(4, 8).as_subclass(FakeQuantized), requires_grad=False
    )
    with pytest.raises(rrl.RavenAttachError, match="quantized"):
        rrl.attach_residual(lin, torch.zeros(2, 8), torch.zeros(4, 2),
                            path="x", alpha=1.0, rank=2)


def test_comfy_linear_stays_a_leaf_after_attach(comfy):
    import comfy.ops as ops

    lin = ops.manual_cast.Linear(8, 4, bias=True, dtype=torch.float32,
                                 device=torch.device("cpu"))
    if getattr(lin, "weight", None) is None:
        lin.weight = torch.nn.Parameter(torch.zeros(4, 8))
        lin.bias = torch.nn.Parameter(torch.zeros(4))
    rrl.attach_residual(lin, torch.zeros(2, 8), torch.zeros(4, 2),
                        path="x", alpha=1.0, rank=2)
    direct = {n for n, _ in lin.named_parameters(recurse=False)}
    assert direct == {n for n, _ in lin.named_parameters(recurse=True)}
    assert direct == {"weight", "bias", "raven_lora_A_0", "raven_lora_B_0"}
    assert hasattr(lin, "comfy_cast_weights")
