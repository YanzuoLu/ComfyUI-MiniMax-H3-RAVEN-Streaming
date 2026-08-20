"""Activation-side LoRA residuals for official ComfyUI ``Linear`` modules.

Why this shape of attachment
----------------------------
The M0 LoRA lane must satisfy four constraints at the same time:

1. **Never materialise ``B @ A``.** The delta is applied to activations:
   ``out = (base_out + B(A(x.float())) * alpha / r * strength).to(base_out.dtype)``
   which is exactly ``peft.tuners.lora.layer.Linear.forward`` (accumulate in the
   adapter dtype, single cast back to the base output dtype at the end).

2. **Do not touch the base module's keys.** ``.weight`` / ``.bias`` and the
   module names stay byte-identical, so the official ``LoraLoaderModelOnly``
   (generic ``diffusion_model.<path>`` format) still resolves and patches the
   very same keys afterwards, and a RAVEN residual simply stacks on top of the
   patched base output.

3. **Be visible to ``CoreModelPatcher``'s dynamic offload accounting.**
   ``ModelPatcher._load_list`` builds its per-module plan like this::

       params = {n: p for n, p in m.named_parameters(recurse=False)}
       for name, param in m.named_parameters(recurse=True):
           if name not in params:
               default = True   # "default random weights in non leaf modules"
               break
       if not default and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
           ... module_size(m) ... load / offload / pin

   So attaching ``lora_A`` / ``lora_B`` as *child modules* of the base ``Linear``
   would turn that ``Linear`` into a non-leaf module and silently drop it from
   the loading plan entirely - its base weight would never be loaded, pinned or
   offloaded. Therefore A/B are registered as **direct parameters of the base
   leaf module** (``raven_lora_A_<slot>`` / ``raven_lora_B_<slot>``). They are
   then counted by ``comfy.model_management.module_size``, moved by
   ``load()``/``partially_load()``, and pinned by ``pin_weight_to_device`` along
   with the base weight, with no extra bookkeeping. The names deliberately do
   *not* end in ``.weight``, so ``comfy.lora.model_lora_keys_unet`` cannot
   accidentally expose them as LoRA-patchable keys.

4. **Touch no private Comfy machinery.** No ``weight_function`` / ``dynamic_vbar``
   / ``LowVramPatch`` poking: the residual is a plain ``register_forward_hook``
   on the module, pure ``torch``. Hook order == registration order, which is how
   several stacked adapters compose.

Known limitation, deliberately loud: ``comfy.ops.linear_input_act`` bypasses
``Linear.__call__`` for INT8-quantised weights (fused activation + INT8 GEMM),
so a forward hook would never run for such a module. Attaching to a quantised
weight therefore raises instead of silently producing a LoRA-less forward.
(For non-quantised weights ``linear_input_act`` *does* go through
``linear(act(x))``, i.e. ``Linear.__call__``, so the hook fires - this is pinned
by a test against the real ``comfy.ldm.minimax.model.MLP``.)

Memory: the residual is evaluated in **row chunks**. A streaming H3 forward can
carry ~60k packed rows; promoting a whole ``[60000, 28672]`` bf16 activation to
FP32 would cost 6.8 GB per LoRA'd MLP. Instead the leading dims are flattened to
rows and each chunk of ``row_chunk`` rows is promoted, accumulated over every
active adapter in FP32, and written back into the base output (in place when
inference-safe). Peak FP32 temporaries are therefore
``row_chunk * (in + rank + 2*out) * 4`` bytes, chosen from a temp budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "RavenAttachError",
    "RavenResidualError",
    "A_PARAM_TEMPLATE",
    "B_PARAM_TEMPLATE",
    "HOOK_ATTR",
    "DEFAULT_TEMP_BUDGET_BYTES",
    "MIN_ROW_CHUNK",
    "MAX_ROW_CHUNK",
    "ResidualSpec",
    "ResidualPlan",
    "RavenLoraForwardHook",
    "RavenLoraAttachment",
    "attach_residual",
    "attach_residuals",
    "lora_residual",
    "compute_row_chunk",
    "module_residual_specs",
]

A_PARAM_TEMPLATE = "raven_lora_A_{}"
B_PARAM_TEMPLATE = "raven_lora_B_{}"
HOOK_ATTR = "_raven_lora_hook"

#: FP32 temporary budget per residual evaluation (env: RAVEN_LORA_TEMP_BUDGET_MB).
DEFAULT_TEMP_BUDGET_BYTES = int(float(os.environ.get("RAVEN_LORA_TEMP_BUDGET_MB", "64")) * 1024 * 1024)
#: Row-chunk clamps: small enough to bound FP32 temporaries, large enough for a
#: healthy GEMM shape (env: RAVEN_LORA_ROW_CHUNK forces an exact value).
MIN_ROW_CHUNK = int(os.environ.get("RAVEN_LORA_MIN_ROW_CHUNK", "128"))
MAX_ROW_CHUNK = int(os.environ.get("RAVEN_LORA_MAX_ROW_CHUNK", "1024"))
_ENV_ROW_CHUNK = os.environ.get("RAVEN_LORA_ROW_CHUNK")
FORCED_ROW_CHUNK: Optional[int] = int(_ENV_ROW_CHUNK) if _ENV_ROW_CHUNK else None


def compute_row_chunk(
    in_features: int,
    out_features: int,
    rank_total: int,
    budget_bytes: int = DEFAULT_TEMP_BUDGET_BYTES,
    minimum: int = MIN_ROW_CHUNK,
    maximum: int = MAX_ROW_CHUNK,
) -> int:
    """Rows per residual chunk that fit ``budget_bytes`` of FP32 temporaries.

    Per row the chunk holds: the promoted input (``in``), the low-rank
    intermediates (``rank_total``), the residual (``out``) and the FP32
    accumulator (``out``).
    """
    if FORCED_ROW_CHUNK:
        return max(1, FORCED_ROW_CHUNK)
    per_row = 4 * (int(in_features) + int(rank_total) + 2 * int(out_features))
    rows = int(budget_bytes) // max(per_row, 1)
    return max(1, min(int(maximum), max(int(minimum), rows)))


class RavenAttachError(RuntimeError):
    """The module cannot carry a RAVEN activation-side residual."""


class RavenResidualError(RuntimeError):
    """The residual could not be evaluated at runtime."""


# --------------------------------------------------------------------------
# residual math
# --------------------------------------------------------------------------
def lora_residual(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float
) -> torch.Tensor:
    """``B(A(x.float())) * scale`` - two thin GEMMs, never ``B @ A``.

    Unchunked single-shot helper (tests, tools). The forward hook does not use
    it: it evaluates the same expression per row chunk to bound FP32 temporaries.
    """
    xf = x if x.dtype == torch.float32 else x.float()
    if a.device != xf.device:
        a = a.to(device=xf.device, non_blocking=True)
    if b.device != xf.device:
        b = b.to(device=xf.device, non_blocking=True)
    if a.dtype != torch.float32:
        a = a.float()
    if b.dtype != torch.float32:
        b = b.float()
    h = torch.nn.functional.linear(xf, a)
    out = torch.nn.functional.linear(h, b)
    if scale != 1.0:
        out = out * scale
    return out


@dataclass
class ResidualSpec:
    """One adapter's residual on one module."""

    name: str
    path: str
    a_param: str
    b_param: str
    alpha: float
    rank: int
    strength: float = 1.0
    enabled: bool = True

    @property
    def scale(self) -> float:
        return float(self.strength) * float(self.alpha) / float(self.rank)


@dataclass
class ResidualPlan:
    """Attach request: which module gets which A/B."""

    path: str
    module: nn.Module
    a: torch.Tensor
    b: torch.Tensor
    alpha: float
    rank: int
    strength: float = 1.0
    base_key: str = ""
    row_chunk: Optional[int] = None


class RavenLoraForwardHook:
    """Forward hook holding every RAVEN residual of one module, in order.

    PEFT semantics per row chunk: every active adapter accumulates against the
    promoted (FP32) base output and the accumulator is cast back to the base
    output dtype exactly once. Chunking only changes the GEMM tiling, so results
    can differ from an untiled reference by a few ULP on some backends (the
    per-row math is identical; the kernel/tile choice is not).
    """

    def __init__(self, row_chunk: Optional[int] = None) -> None:
        self.residuals: List[ResidualSpec] = []
        self.calls: int = 0
        self.chunks: int = 0
        # set by attach_residual; whoever empties the hook removes it
        self.handle: Optional[object] = None
        #: explicit rows per chunk; ``None`` derives it from the temp budget
        self.row_chunk: Optional[int] = row_chunk
        self._auto_chunk: Optional[int] = None
        self._auto_key: Optional[Tuple[int, int, int]] = None

    # -- chunk sizing ------------------------------------------------------
    def effective_row_chunk(
        self, in_features: int, out_features: int, active: Sequence[ResidualSpec]
    ) -> int:
        if self.row_chunk:
            return max(1, int(self.row_chunk))
        rank_total = sum(int(s.rank) for s in active)
        key = (int(in_features), int(out_features), rank_total)
        if key != self._auto_key or self._auto_chunk is None:
            self._auto_key = key
            self._auto_chunk = compute_row_chunk(in_features, out_features, rank_total)
        return self._auto_chunk

    # -- residual ----------------------------------------------------------
    def __call__(self, module: nn.Module, args, output):
        self.calls += 1
        active = [s for s in self.residuals if s.enabled and s.strength != 0.0]
        if not active:
            return output
        if not isinstance(output, torch.Tensor):
            raise RavenResidualError(
                "RAVEN LoRA expects a plain tensor output from {}, got {}".format(
                    type(module).__name__, type(output).__name__
                )
            )
        if not args:
            raise RavenResidualError(
                "RAVEN LoRA expects the activation as the first positional argument of "
                "{}".format(type(module).__name__)
            )
        x = args[0]
        if not isinstance(x, torch.Tensor):
            raise RavenResidualError(
                "RAVEN LoRA expects a tensor input, got {}".format(type(x).__name__)
            )
        if x.dim() < 2 or output.dim() < 2:
            raise RavenResidualError(
                "RAVEN LoRA expects at least 2D activations, got x{} out{}".format(
                    tuple(x.shape), tuple(output.shape)
                )
            )

        out_dtype = output.dtype
        in_features = x.shape[-1]
        out_features = output.shape[-1]
        # flatten leading dims to rows; both are contiguous in practice, so these
        # are views and the write-back below lands in `output` itself
        x2 = x if x.dim() == 2 else x.reshape(-1, in_features)
        o2 = output if output.dim() == 2 else output.reshape(-1, out_features)
        rows = o2.shape[0]
        if x2.shape[0] != rows:
            raise RavenResidualError(
                "RAVEN LoRA row mismatch: input {} vs output {}".format(
                    tuple(x.shape), tuple(output.shape)
                )
            )

        # In-place write-back is only safe when autograd cannot need the base
        # output and when `o2` really aliases `output`.
        inplace = output.is_contiguous() and (
            not torch.is_grad_enabled() or not (output.requires_grad or x.requires_grad)
        )
        chunk = self.effective_row_chunk(in_features, out_features, active)
        pieces: List[torch.Tensor] = []

        if rows == 0:
            return output

        params = [(getattr(module, s.a_param), getattr(module, s.b_param), s.scale) for s in active]
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            self.chunks += 1
            out_chunk = o2[start:stop]
            if inplace and out_dtype == torch.float32:
                acc = out_chunk  # accumulate straight into the base output
            else:
                acc = out_chunk.to(torch.float32, copy=True)
            xc = x2[start:stop]
            xf = xc if xc.dtype == torch.float32 else xc.float()
            for a, b, scale in params:
                if a.device != xf.device:
                    a = a.to(device=xf.device, non_blocking=True)
                if b.device != xf.device:
                    b = b.to(device=xf.device, non_blocking=True)
                if a.dtype != torch.float32:
                    a = a.float()
                if b.dtype != torch.float32:
                    b = b.float()
                resid = torch.nn.functional.linear(torch.nn.functional.linear(xf, a), b)
                if scale != 1.0:
                    resid.mul_(scale)  # same rounding as `residual * scaling`
                acc.add_(resid)  # same rounding as `result + residual`
            if acc is out_chunk:
                continue
            if inplace:
                out_chunk.copy_(acc)  # single cast back to the base dtype
            else:
                pieces.append(acc.to(out_dtype))

        if inplace:
            return output
        merged = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        return merged if output.dim() == 2 else merged.reshape(output.shape)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "RavenLoraForwardHook({} residual(s), {} calls, row_chunk={})".format(
            len(self.residuals), self.calls, self.row_chunk or "auto"
        )


def module_residual_specs(module: nn.Module) -> List[ResidualSpec]:
    hook = getattr(module, HOOK_ATTR, None)
    return list(hook.residuals) if hook is not None else []


# --------------------------------------------------------------------------
# attach / detach
# --------------------------------------------------------------------------
def _is_quantized(weight: torch.Tensor) -> bool:
    # comfy.quant_ops.QuantizedTensor duck-check, without importing comfy here.
    return hasattr(weight, "_layout_cls") or type(weight).__name__ == "QuantizedTensor"


def _validate_module(module: nn.Module, path: str) -> None:
    if not isinstance(module, nn.Module):
        raise RavenAttachError("{}: not an nn.Module".format(path))
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RavenAttachError("{}: module has no weight tensor".format(path))
    if weight.dim() != 2:
        raise RavenAttachError(
            "{}: expected a 2D Linear weight, got shape {}".format(path, tuple(weight.shape))
        )
    if _is_quantized(weight):
        raise RavenAttachError(
            "{}: weight is a quantized tensor. comfy.ops.linear_input_act bypasses "
            "Linear.__call__ for INT8 layouts, so a forward hook would silently not "
            "run. The M0 LoRA lane requires the non-quantized full model.".format(path)
        )
    children = [n for n, _ in module.named_children()]
    if children:
        raise RavenAttachError(
            "{}: target must stay a leaf module (children: {}); ModelPatcher._load_list "
            "skips modules whose parameters are not direct.".format(path, children)
        )


def _next_slot(module: nn.Module) -> int:
    slot = 0
    while hasattr(module, A_PARAM_TEMPLATE.format(slot)) or hasattr(
        module, B_PARAM_TEMPLATE.format(slot)
    ):
        slot += 1
    return slot


def attach_residual(
    module: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    path: str,
    alpha: float,
    rank: int,
    strength: float = 1.0,
    name: str = "raven_lora",
    row_chunk: Optional[int] = None,
) -> Tuple[ResidualSpec, Optional[object]]:
    """Register A/B as direct parameters of ``module`` and hook the residual.

    Returns ``(spec, handle)`` where ``handle`` is the newly created hook handle
    or ``None`` when the module already carried a RAVEN hook.
    """
    _validate_module(module, path)
    weight = module.weight
    if a.dim() != 2 or b.dim() != 2:
        raise RavenAttachError("{}: lora_A/lora_B must be 2D".format(path))
    if int(a.shape[0]) != int(b.shape[1]):
        raise RavenAttachError(
            "{}: rank mismatch between A{} and B{}".format(path, tuple(a.shape), tuple(b.shape))
        )
    if int(a.shape[0]) != int(rank):
        raise RavenAttachError(
            "{}: declared rank {} != A rank {}".format(path, rank, a.shape[0])
        )
    if int(a.shape[1]) != int(weight.shape[1]) or int(b.shape[0]) != int(weight.shape[0]):
        raise RavenAttachError(
            "{}: A{} / B{} do not fit base weight {}".format(
                path, tuple(a.shape), tuple(b.shape), tuple(weight.shape)
            )
        )

    slot = _next_slot(module)
    a_name = A_PARAM_TEMPLATE.format(slot)
    b_name = B_PARAM_TEMPLATE.format(slot)
    module.register_parameter(a_name, nn.Parameter(a.detach(), requires_grad=False))
    module.register_parameter(b_name, nn.Parameter(b.detach(), requires_grad=False))

    hook: Optional[RavenLoraForwardHook] = getattr(module, HOOK_ATTR, None)
    handle = None
    if hook is None:
        hook = RavenLoraForwardHook(row_chunk=row_chunk)
        setattr(module, HOOK_ATTR, hook)
        handle = module.register_forward_hook(hook)
        hook.handle = handle
    elif row_chunk is not None:
        hook.row_chunk = int(row_chunk)

    spec = ResidualSpec(
        name=name,
        path=path,
        a_param=a_name,
        b_param=b_name,
        alpha=float(alpha),
        rank=int(rank),
        strength=float(strength),
    )
    hook.residuals.append(spec)
    return spec, handle


@dataclass
class AttachedResidual:
    path: str
    module: nn.Module
    spec: ResidualSpec
    hook: RavenLoraForwardHook
    base_key: str = ""


class RavenLoraAttachment:
    """Handle for one attached adapter; supports strength updates and detach."""

    def __init__(self, name: str, entries: Sequence[AttachedResidual]):
        self.name = name
        self.entries: List[AttachedResidual] = list(entries)
        self._detached = False

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def detached(self) -> bool:
        return self._detached

    @property
    def strength(self) -> float:
        if not self.entries:
            return 0.0
        return self.entries[0].spec.strength

    @property
    def alpha(self) -> float:
        return self.entries[0].spec.alpha if self.entries else 0.0

    @property
    def rank(self) -> int:
        return self.entries[0].spec.rank if self.entries else 0

    def paths(self) -> List[str]:
        return [e.path for e in self.entries]

    def base_keys(self) -> List[str]:
        return [e.base_key for e in self.entries if e.base_key]

    def parameter_names(self) -> List[str]:
        out: List[str] = []
        for e in self.entries:
            out.append("{}.{}".format(e.path, e.spec.a_param))
            out.append("{}.{}".format(e.path, e.spec.b_param))
        return out

    def state_dict_keys(self, prefix: str = "diffusion_model.") -> List[str]:
        return ["{}{}".format(prefix, n) for n in self.parameter_names()]

    def parameter_numel(self) -> int:
        total = 0
        for e in self.entries:
            total += getattr(e.module, e.spec.a_param).numel()
            total += getattr(e.module, e.spec.b_param).numel()
        return total

    def parameter_bytes(self) -> int:
        total = 0
        for e in self.entries:
            for pname in (e.spec.a_param, e.spec.b_param):
                p = getattr(e.module, pname)
                total += p.numel() * p.element_size()
        return total

    def call_counts(self) -> Dict[str, int]:
        return {e.path: e.hook.calls for e in self.entries}

    def chunk_counts(self) -> Dict[str, int]:
        return {e.path: e.hook.chunks for e in self.entries}

    def row_chunks(self) -> Dict[str, int]:
        """Rows per residual chunk actually used per module."""
        out: Dict[str, int] = {}
        for e in self.entries:
            weight = getattr(e.module, "weight")
            out[e.path] = e.hook.effective_row_chunk(
                weight.shape[1], weight.shape[0], e.hook.residuals
            )
        return out

    def devices(self) -> Dict[str, str]:
        return {
            "{}.{}".format(e.path, pname): str(getattr(e.module, pname).device)
            for e in self.entries
            for pname in (e.spec.a_param, e.spec.b_param)
        }

    # -- mutation ----------------------------------------------------------
    def set_strength(self, strength: float) -> None:
        for e in self.entries:
            e.spec.strength = float(strength)

    def set_enabled(self, enabled: bool) -> None:
        for e in self.entries:
            e.spec.enabled = bool(enabled)

    def set_row_chunk(self, row_chunk: Optional[int]) -> None:
        """Force the rows per residual chunk (``None`` = derive from the budget)."""
        if row_chunk is not None and int(row_chunk) < 1:
            raise RavenAttachError("row_chunk must be >= 1, got {}".format(row_chunk))
        for e in self.entries:
            e.hook.row_chunk = None if row_chunk is None else int(row_chunk)

    def detach(self) -> None:
        """Remove this adapter's residuals and A/B parameters.

        Other adapters attached to the same modules keep working: the shared
        forward hook is only unregistered once its residual list is empty. Base
        keys are untouched throughout.
        """
        if self._detached:
            return
        for e in self.entries:
            hook = getattr(e.module, HOOK_ATTR, None)
            if hook is not None and e.spec in hook.residuals:
                hook.residuals.remove(e.spec)
            for pname in (e.spec.a_param, e.spec.b_param):
                if pname in e.module._parameters:
                    del e.module._parameters[pname]
            if hook is not None and not hook.residuals:
                if hook.handle is not None:
                    hook.handle.remove()
                    hook.handle = None
                try:
                    delattr(e.module, HOOK_ATTR)
                except AttributeError:  # pragma: no cover - defensive
                    pass
        self._detached = True

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "RavenLoraAttachment(name={!r}, modules={}, rank={}, alpha={}, strength={})".format(
            self.name, len(self.entries), self.rank, self.alpha, self.strength
        )


def attach_residuals(
    plan: Iterable[ResidualPlan], name: str = "raven_lora", row_chunk: Optional[int] = None
) -> RavenLoraAttachment:
    """Attach a whole adapter; rolls back completely if any module fails."""
    entries: List[AttachedResidual] = []
    try:
        for item in plan:
            spec, _handle = attach_residual(
                item.module,
                item.a,
                item.b,
                path=item.path,
                alpha=item.alpha,
                rank=item.rank,
                strength=item.strength,
                name=name,
                row_chunk=item.row_chunk if item.row_chunk is not None else row_chunk,
            )
            entries.append(
                AttachedResidual(
                    path=item.path,
                    module=item.module,
                    spec=spec,
                    hook=getattr(item.module, HOOK_ATTR),
                    base_key=item.base_key,
                )
            )
    except Exception:
        RavenLoraAttachment(name, entries).detach()
        raise
    return RavenLoraAttachment(name, entries)
