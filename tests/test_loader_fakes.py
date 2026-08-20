"""Tiny fake ComfyUI modules for the M1 loader tests.

Why fakes: the loader's contract is an *ordering* and *bookkeeping* contract
(load base weights -> attach the RAVEN residual -> build the patcher, with the
patcher's first ``model_size()`` already counting the residual). Pinning that
needs a structurally faithful module tree, not 66 GB of BF16 weights. These
fakes mirror the pinned upstream surface (``c67885b``) closely enough that
:mod:`raven_streaming.compat`'s real feature probe passes against them, and they
record an event log so the test can assert the call order.

The real upstream is exercised separately in ``tests/test_loader_official.py``
(skipped when no ComfyUI checkout is available), including the real
``comfy.lora.model_lora_keys_unet`` / ``comfy.sd.load_lora_for_models`` chain.

This module deliberately contains no tests; it is imported by the other
``test_loader_*`` modules (the name only follows the file naming convention).
"""

from __future__ import annotations

import enum
import json
import struct
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from raven_streaming import lora as rlora  # noqa: E402

# --------------------------------------------------------------------------
# a small but structurally identical H3 config
# --------------------------------------------------------------------------
TINY_CONFIG = rlora.RavenBaseConfig(
    hidden_size=64,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=4,
    attention_head_dim=16,
    ffn_hidden_size=48,
    latents_dim=4,
    audio_latents_dim=6,
    text_dim=32,
    timestep_input_dim=16,
    time_embed_hidden_size=64,
    time_embed_dim=48,
)
TINY_COUNTS = {"core": 12, "adaln": 3, "time": 2, "boundary": 5}
TINY_RANK = 4
TINY_ALPHA = 8.0
ROPE_INV_FREQ_LEN = 8

#: ordered (name, payload) log of the upstream calls the loader makes
EVENTS: List[Tuple[str, Any]] = []


def reset_events() -> None:
    EVENTS.clear()


def record(name: str, payload: Any = None) -> None:
    EVENTS.append((name, payload))


def event_names() -> List[str]:
    return [name for name, _ in EVENTS]


# --------------------------------------------------------------------------
# safetensors writer (tests must not depend on the safetensors package)
# --------------------------------------------------------------------------
_ST_NAMES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}


def write_safetensors(
    path: str, tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> str:
    header: Dict[str, Any] = {}
    blobs: List[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        contiguous = tensor.detach().contiguous()
        if contiguous.dtype not in _ST_NAMES:
            raise TypeError("unsupported test dtype {}".format(contiguous.dtype))
        # numpy has no bfloat16; the bit pattern is what goes on disk either way
        raw = (contiguous.view(torch.int16) if contiguous.dtype is torch.bfloat16
               else contiguous).numpy().tobytes()
        header[name] = {
            "dtype": _ST_NAMES[contiguous.dtype],
            "shape": list(contiguous.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for raw in blobs:
            fh.write(raw)
    return path


def write_tiny_lora(
    path: str,
    config: rlora.RavenBaseConfig = TINY_CONFIG,
    rank: int = TINY_RANK,
    alpha: float = TINY_ALPHA,
    seed: int = 7,
) -> str:
    gen = torch.Generator().manual_seed(seed)
    tensors: Dict[str, torch.Tensor] = {}
    for path_, entry in config.modules().items():
        tensors[rlora.PEFT_PREFIX + path_ + ".lora_A.weight"] = (
            torch.randn(rank, entry.in_features, generator=gen) * 0.02
        )
        tensors[rlora.PEFT_PREFIX + path_ + ".lora_B.weight"] = (
            torch.randn(entry.out_features, rank, generator=gen) * 0.02
        )
    return write_safetensors(path, tensors, {"lora_alpha": str(alpha)})


# --------------------------------------------------------------------------
# fake comfy.ldm.minimax.model
# --------------------------------------------------------------------------
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _ensure_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        child = getattr(node, part, None)
        if child is None:
            child = nn.Module()
            node.add_module(part, child)
        node = child
    return node, parts[-1]


class FakeMiniMaxH3Model(nn.Module):
    """Structural stand-in for ``comfy.ldm.minimax.model.MiniMaxH3Model``.

    Same module paths, same key names, same LoRA-eligible leaves - just small.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        token_refiner_num_layers: int = 1,
        num_attention_heads: int = 4,
        attention_head_dim: int = 16,
        ffn_hidden_size: int = 48,
        latents_dim: int = 4,
        audio_latents_dim: int = 6,
        patch_size: Sequence[int] = (1, 2, 2),
        text_dim: int = 32,
        timestep_input_dim: int = 16,
        time_embed_hidden_size: int = 64,
        time_embed_dim: int = 48,
        rope_inv_freq_len: int = ROPE_INV_FREQ_LEN,
        adaln_curve_grid: Optional[int] = None,
        image_model: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if adaln_curve_grid is not None:
            # mirrors the pruned/adaln-curve form: no time embedder, shared basis
            self.register_buffer(
                "adaln_t_table", torch.zeros(adaln_curve_grid, time_embed_dim), persistent=True
            )
        operations = operations or nn
        self.config = rlora.RavenBaseConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            token_refiner_num_layers=token_refiner_num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_hidden_size=ffn_hidden_size,
            latents_dim=latents_dim,
            audio_latents_dim=audio_latents_dim,
            patch_size=tuple(patch_size),
            text_dim=text_dim,
            timestep_input_dim=timestep_input_dim,
            time_embed_hidden_size=time_embed_hidden_size,
            time_embed_dim=time_embed_dim,
        )
        for path, entry in self.config.modules().items():
            if adaln_curve_grid is not None and path.startswith("time_embedder"):
                continue
            parent, leaf = _ensure_parent(self, path)
            parent.add_module(
                leaf,
                operations.Linear(
                    entry.in_features, entry.out_features, bias=True, dtype=dtype, device=device
                ),
            )
        for index in range(num_layers):
            block = self.get_submodule("blocks.{}".format(index))
            block.attn.q_norm = nn.Module()
            block.attn.q_norm.weight = nn.Parameter(
                torch.ones(attention_head_dim, dtype=dtype, device=device), requires_grad=False
            )
        self.rope = nn.Module()
        self.rope.register_buffer(
            "inv_freq", torch.ones(rope_inv_freq_len, dtype=torch.float32, device=device)
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - never sampled here
        raise NotImplementedError("the fake H3 DiT is structural only")


class FakeCausalMiniMaxH3Model(FakeMiniMaxH3Model):
    """Stand-in for the M2 causal/streaming DiT injected via ``unet_model_cls``."""


def build_tiny_state_dict(
    config: rlora.RavenBaseConfig = TINY_CONFIG,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 11,
    pruned: bool = False,
) -> Dict[str, torch.Tensor]:
    """The checkpoint the loader is asked to read (unprefixed keys)."""
    gen = torch.Generator().manual_seed(seed)
    sd: Dict[str, torch.Tensor] = {}
    for path, entry in config.modules().items():
        if pruned and path.startswith("time_embedder"):
            continue
        sd[path + ".weight"] = (
            torch.randn(entry.out_features, entry.in_features, generator=gen) * 0.05
        ).to(dtype)
        sd[path + ".bias"] = (torch.randn(entry.out_features, generator=gen) * 0.05).to(dtype)
    for index in range(config.num_layers):
        sd["blocks.{}.attn.q_norm.weight".format(index)] = torch.ones(
            config.attention_head_dim, dtype=dtype
        )
    sd["rope.inv_freq"] = torch.ones(ROPE_INV_FREQ_LEN, dtype=torch.float32)
    if pruned:
        sd["adaln_t_table"] = torch.zeros(16, config.time_embed_dim, dtype=torch.float32)
    return sd


# --------------------------------------------------------------------------
# fake comfy.utils
# --------------------------------------------------------------------------
#: path -> (state_dict, metadata) registry used by the fake ``load_torch_file``
FAKE_FILES: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, str]]] = {}


def register_file(path: str, state_dict: Dict[str, torch.Tensor], metadata=None) -> str:
    """Register a fake checkpoint; the file is also touched on disk.

    The loader resolves and existence-checks paths before reading them, so a
    registry entry alone would not be enough - the placeholder keeps that check
    honest without writing weights.
    """
    FAKE_FILES[str(path)] = (dict(state_dict), dict(metadata or {}))
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        file.write_bytes(b"fake checkpoint; contents live in FAKE_FILES")
    return str(path)


def fake_load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    record("load_torch_file", str(ckpt))
    try:
        sd, metadata = FAKE_FILES[str(ckpt)]
    except KeyError:
        raise FileNotFoundError("fake file registry has no {!r}".format(ckpt)) from None
    sd = dict(sd)
    return (sd, dict(metadata)) if return_metadata else sd


def fake_save_torch_file(sd, ckpt, metadata=None):  # pragma: no cover - unused
    register_file(ckpt, sd, metadata)


def fake_convert_old_quants(state_dict, model_prefix="", metadata=None):
    record("convert_old_quants", model_prefix)
    return state_dict, metadata


def fake_state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replaced = [(k, "{}{}".format(replace_prefix[rp], k[len(rp):])) for k in
                    filter(lambda a: a.startswith(rp), state_dict.keys())]
        for x in replaced:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def fake_calculate_parameters(sd, prefix=""):
    return sum(v.numel() for k, v in sd.items() if k.startswith(prefix))


def fake_weight_dtype(sd, prefix=""):
    dtypes: Dict[torch.dtype, int] = {}
    for k, v in sd.items():
        if k.startswith(prefix) and hasattr(v, "dtype"):
            dtypes[v.dtype] = dtypes.get(v.dtype, 0) + v.numel()
    return max(dtypes, key=dtypes.get) if dtypes else None


# --------------------------------------------------------------------------
# fake comfy.model_management
# --------------------------------------------------------------------------
def fake_get_torch_device():
    return torch.device("cpu")


def fake_unet_offload_device():
    return torch.device("cpu")


def fake_unet_dtype(model_params=0, supported_dtypes=(torch.bfloat16,), weight_dtype=None):
    if weight_dtype in supported_dtypes:
        return weight_dtype
    return supported_dtypes[0]


def fake_unet_manual_cast(weight_dtype, inference_device, supported_dtypes=()):
    return None


def fake_is_device_cpu(device):
    return torch.device(device).type == "cpu"


def fake_module_size(module):
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    for buf in module.buffers():
        total += buf.numel() * buf.element_size()
    return total


def fake_load_models_gpu(models, memory_required=0, force_patch_weights=False,
                         minimum_memory_required=None, force_full_load=False):
    record("load_models_gpu", len(models))
    for model in models:
        model.partially_load(model.load_device)


# --------------------------------------------------------------------------
# fake comfy.model_patcher
# --------------------------------------------------------------------------
class FakeModelPatcher:
    """The bits of ``ModelPatcher`` the loader relies on, with an event log."""

    def __init__(self, model, load_device=None, offload_device=None, size=0,
                 weight_inplace_update=False):
        record("patcher_init", type(model).__name__)
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.weight_inplace_update = weight_inplace_update
        self.cached_patcher_init = None
        self.patches: Dict[str, Any] = {}
        self.model_options: Dict[str, Any] = {"transformer_options": {}}
        self.attachments: Dict[str, Any] = {}
        self.parent = None

    # -- size / load ----------------------------------------------------
    def is_dynamic(self):
        return False

    def model_size(self):
        if self.size > 0:
            return self.size
        record("model_size", None)
        self.size = fake_module_size(self.model)
        return self.size

    def loaded_size(self):
        return int(getattr(self.model, "model_loaded_weight_memory", 0) or 0)

    def current_loaded_device(self):
        return self.load_device

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        record("partially_load", str(device_to))
        self.model.to(device_to)
        self.model.model_loaded_weight_memory = self.model_size()
        return self.model_size()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        self.model.to(device_to)
        self.model.model_loaded_weight_memory = 0
        return 0

    # -- patching / cloning ---------------------------------------------
    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        if filter_prefix is not None:
            sd = {k: v for k, v in sd.items() if k.startswith(filter_prefix)}
        return sd

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        applied = []
        model_sd = self.model_state_dict()
        for key in patches:
            if key in model_sd:
                self.patches.setdefault(key, []).append((strength_patch, patches[key]))
                applied.append(key)
        return applied

    def set_attachments(self, name, attachment):
        self.attachments[name] = attachment

    def clone(self, disable_dynamic=False, model_override=None):
        record("clone", disable_dynamic)
        cloned = type(self)(
            self.model if model_override is None else model_override,
            self.load_device,
            self.offload_device,
            self.size,
        )
        cloned.patches = {k: list(v) for k, v in self.patches.items()}
        cloned.cached_patcher_init = self.cached_patcher_init
        cloned.parent = self
        return cloned

    def detach(self, unpatch_all=True):
        return self.model


class FakeModelPatcherDynamic(FakeModelPatcher):
    """Mirrors ``ModelPatcherDynamic``'s CPU reroute and ``is_dynamic``."""

    def __new__(cls, model=None, load_device=None, offload_device=None, size=0,
                weight_inplace_update=False):
        if load_device is not None and fake_is_device_cpu(load_device):
            return FakeModelPatcher(model, load_device, offload_device, size,
                                    weight_inplace_update)
        return super().__new__(cls)

    def is_dynamic(self):
        return True


# --------------------------------------------------------------------------
# fake comfy.latent_formats / supported_models / model_base
# --------------------------------------------------------------------------
class FakeMiniMaxH3AV:
    latent_channels = 32
    latent_dimensions = 3


class FakeModelType(enum.Enum):
    EPS = 1
    FLOW_AV = 8


class FakeBaseModel(nn.Module):
    def __init__(self, model_config, model_type=FakeModelType.EPS, device=None, unet_model=None):
        super().__init__()
        self.model_config = model_config
        self.model_type = model_type
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher = None
        self.model_loaded_weight_memory = 0
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        unet_config = dict(model_config.unet_config)
        unet_config.pop("image_model", None)
        dtype = unet_config.pop("dtype", None)
        record("base_model_init", getattr(unet_model, "__name__", unet_model))
        self.diffusion_model = unet_model(
            **unet_config, dtype=dtype, device=device, operations=nn
        )
        self.diffusion_model.requires_grad_(False)
        self.diffusion_model.eval()

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        record("load_model_weights", assign)
        to_load = {}
        for k in list(sd.keys()):
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)
        to_load = self.model_config.process_unet_state_dict(to_load)
        self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
        return self

    def get_dtype(self):
        return self.diffusion_model.video_patch_proj.weight.dtype


class FakeMiniMaxH3BaseModel(FakeBaseModel):
    def __init__(self, model_config, model_type=FakeModelType.FLOW_AV, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=FakeMiniMaxH3Model)

    def audio_scale(self):
        return 1.0


class FakeMiniMaxH3Config:
    """Stand-in for ``comfy.supported_models.MiniMaxH3``."""

    unet_config = {"image_model": "minimax_h3"}
    sampling_settings = {"shift": 12.0, "audio_shift": 3.0}
    latent_format = FakeMiniMaxH3AV
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 0.114
    custom_operations = None
    quant_config = None
    optimizations = {"fp8": False}
    manual_cast_dtype = None

    def __init__(self, unet_config: Dict[str, Any]):
        self.unet_config = dict(unet_config)
        self.optimizations = dict(type(self).optimizations)
        self.manual_cast_dtype = None

    def set_inference_dtype(self, dtype, manual_cast_dtype, device=None):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def get_model(self, state_dict, prefix="", device=None):
        record("get_model", prefix)
        return FakeMiniMaxH3BaseModel(self, device=device)


# --------------------------------------------------------------------------
# fake comfy.model_detection
# --------------------------------------------------------------------------
def fake_unet_prefix_from_state_dict(state_dict):
    candidates = ["model.diffusion_model.", "model.model.", "net."]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    return top if counts[top] > 5 else "model."


def _count_blocks(keys, prefix_template):
    count = 0
    while True:
        prefix = prefix_template.format(count)
        if not any(k.startswith(prefix) for k in keys):
            break
        count += 1
    return count


def fake_model_config_from_unet(state_dict, unet_key_prefix="", use_base_if_no_match=False,
                                metadata=None):
    """Same shape-reading logic as upstream's MiniMax-H3 branch, on tiny tensors."""
    keys = set(state_dict.keys())
    pref = unet_key_prefix
    if "{}video_patch_proj.weight".format(pref) not in keys:
        return None
    if "{}audio_patch_proj.weight".format(pref) not in keys:
        return None
    cfg: Dict[str, Any] = {"image_model": "minimax_h3"}
    cfg["num_layers"] = _count_blocks(keys, "{}blocks.".format(pref) + "{}.")
    cfg["token_refiner_num_layers"] = _count_blocks(
        keys, "{}token_refiner.blocks.".format(pref) + "{}."
    )
    cfg["hidden_size"] = state_dict["{}video_patch_proj.weight".format(pref)].shape[0]
    cfg["latents_dim"] = state_dict["{}final_layer.video_out.weight".format(pref)].shape[0] // 4
    cfg["audio_latents_dim"] = state_dict["{}final_layer.audio_out.weight".format(pref)].shape[0]
    cfg["attention_head_dim"] = state_dict["{}blocks.0.attn.q_norm.weight".format(pref)].shape[0]
    qkv = state_dict["{}blocks.0.attn.qkv_proj.weight".format(pref)]
    cfg["num_attention_heads"] = qkv.shape[0] // (3 * cfg["attention_head_dim"])
    cfg["ffn_hidden_size"] = state_dict["{}blocks.0.mlp.fc1.weight".format(pref)].shape[0] // 2
    cfg["text_dim"] = state_dict["{}condition_proj.weight".format(pref)].shape[1]
    table_key = "{}adaln_t_table".format(pref)
    if table_key in keys:
        table = state_dict[table_key].shape
        cfg["adaln_curve_grid"] = table[0]
        cfg["time_embed_dim"] = table[1]
    else:
        te = state_dict["{}time_embedder.proj_in.weight".format(pref)]
        cfg["timestep_input_dim"] = te.shape[1]
        cfg["time_embed_hidden_size"] = te.shape[0]
        cfg["time_embed_dim"] = state_dict["{}time_embedder.proj_out.weight".format(pref)].shape[0]
    cfg["rope_inv_freq_len"] = state_dict["{}rope.inv_freq".format(pref)].shape[0]
    record("model_config_from_unet", cfg["num_layers"])
    return FakeMiniMaxH3Config(cfg)


# --------------------------------------------------------------------------
# fake comfy.lora / comfy.sd / folder_paths
# --------------------------------------------------------------------------
def fake_model_lora_keys_unet(model, key_map={}):
    """Upstream's generic-format rule for ``diffusion_model.*`` keys, verbatim.

    Including the branch that is easy to forget: upstream *also* maps every
    non-``.weight`` ``diffusion_model.*`` key onto itself ("generic lora format
    for not .weight without any weird key names"), which is why the RAVEN A/B
    parameters do appear in the key map under their own verbatim names. The
    ``lora_unet_*`` names - the ones a published LoRA actually uses - never
    point at them. The diffusers/lycoris part of upstream is a no-op for H3
    (``unet_to_diffusers`` returns ``{}`` without ``num_res_blocks``), so it is
    not modelled here.
    """
    sd = model.state_dict()
    for k in sd.keys():
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map[k[:-len(".weight")]] = k
            else:
                key_map[k] = k
    return key_map


def fake_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                              lora_metadata=None):
    key_map = fake_model_lora_keys_unet(model.model, {}) if model is not None else {}
    loaded = {}
    for name, key in key_map.items():
        up = lora.get(name + ".lora_up.weight", None)
        down = lora.get(name + ".lora_down.weight", None)
        if up is not None and down is not None:
            loaded[key] = ("lora", (up, down, None, None, None))
    new_patcher = model.clone() if model is not None else None
    if new_patcher is not None:
        new_patcher.add_patches(loaded, strength_model)
    return new_patcher, None


def fake_load_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
    raise NotImplementedError("fake")


def fake_load_diffusion_model_state_dict(sd, model_options={}, metadata=None,
                                         disable_dynamic=False):
    raise NotImplementedError("fake")


#: folder name -> {file name: absolute path}
FOLDERS: Dict[str, Dict[str, str]] = {}


def register_folder_file(folder: str, name: str, path: str) -> str:
    FOLDERS.setdefault(folder, {})[name] = str(path)
    return str(path)


def fake_get_full_path_or_raise(folder_name, filename):
    try:
        return FOLDERS[folder_name][filename]
    except KeyError:
        raise FileNotFoundError(
            "{} not in folder {}".format(filename, folder_name)
        ) from None


def fake_get_full_path(folder_name, filename):
    return FOLDERS.get(folder_name, {}).get(filename, None)


def fake_get_filename_list(folder_name):
    return sorted(FOLDERS.get(folder_name, {}))


def fake_get_folder_paths(folder_name):
    return []


# --------------------------------------------------------------------------
# module assembly
# --------------------------------------------------------------------------
def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def build_fake_modules() -> Dict[str, types.ModuleType]:
    """Every fake module, keyed by the name the loader imports it under."""
    utils = _module(
        "comfy.utils",
        load_torch_file=fake_load_torch_file,
        save_torch_file=fake_save_torch_file,
        convert_old_quants=fake_convert_old_quants,
        state_dict_prefix_replace=fake_state_dict_prefix_replace,
        calculate_parameters=fake_calculate_parameters,
        weight_dtype=fake_weight_dtype,
    )
    model_detection = _module(
        "comfy.model_detection",
        unet_prefix_from_state_dict=fake_unet_prefix_from_state_dict,
        model_config_from_unet=fake_model_config_from_unet,
    )
    model_management = _module(
        "comfy.model_management",
        get_torch_device=fake_get_torch_device,
        unet_offload_device=fake_unet_offload_device,
        unet_dtype=fake_unet_dtype,
        unet_manual_cast=fake_unet_manual_cast,
        is_device_cpu=fake_is_device_cpu,
        module_size=fake_module_size,
        load_models_gpu=fake_load_models_gpu,
    )
    model_patcher = _module(
        "comfy.model_patcher",
        ModelPatcher=FakeModelPatcher,
        CoreModelPatcher=FakeModelPatcher,
        ModelPatcherDynamic=FakeModelPatcherDynamic,
    )
    latent_formats = _module("comfy.latent_formats", MiniMaxH3AV=FakeMiniMaxH3AV)
    model_base = _module(
        "comfy.model_base",
        BaseModel=FakeBaseModel,
        MiniMaxH3=FakeMiniMaxH3BaseModel,
        ModelType=FakeModelType,
    )
    supported_models = _module("comfy.supported_models", MiniMaxH3=FakeMiniMaxH3Config)
    minimax_ldm = _module(
        "comfy.ldm.minimax.model",
        MiniMaxH3Model=FakeMiniMaxH3Model,
        FRAME_PER_TOKEN=FRAME_PER_TOKEN,
        FRAME_RESCALE=FRAME_RESCALE,
    )
    lora_module = _module("comfy.lora", model_lora_keys_unet=fake_model_lora_keys_unet)
    sd_module = _module(
        "comfy.sd",
        load_diffusion_model=fake_load_diffusion_model,
        load_diffusion_model_state_dict=fake_load_diffusion_model_state_dict,
        load_lora_for_models=fake_load_lora_for_models,
    )
    folder_paths = _module(
        "folder_paths",
        get_full_path_or_raise=fake_get_full_path_or_raise,
        get_full_path=fake_get_full_path,
        get_filename_list=fake_get_filename_list,
        get_folder_paths=fake_get_folder_paths,
    )
    return {
        "comfy": _module("comfy"),
        "comfy.ldm": _module("comfy.ldm"),
        "comfy.ldm.minimax": _module("comfy.ldm.minimax"),
        "comfy.utils": utils,
        "comfy.sd": sd_module,
        "comfy.lora": lora_module,
        "comfy.model_detection": model_detection,
        "comfy.model_management": model_management,
        "comfy.model_patcher": model_patcher,
        "comfy.model_base": model_base,
        "comfy.supported_models": supported_models,
        "comfy.latent_formats": latent_formats,
        "comfy.ldm.minimax.model": minimax_ldm,
        "folder_paths": folder_paths,
    }


def install_fake_modules(monkeypatch) -> Dict[str, types.ModuleType]:
    """Put the fakes in ``sys.modules`` for the duration of one test."""
    modules = build_fake_modules()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    reset_events()
    FAKE_FILES.clear()
    FOLDERS.clear()
    return modules
