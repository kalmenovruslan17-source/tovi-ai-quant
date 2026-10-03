"""SVDQuant (nunchaku W4A4) as a FastVideo transformer quantization method.

Importing this module registers two FastVideo quantization configs:

* ``svdq``       -- selected DiT linears run on nunchaku's W4A4 kernels.
* ``svdq_calib`` -- bf16 inference that records per-input-channel activation
                    maxima of the same linears (SmoothQuant calibration).

Both are selected through the typed API, e.g.
``EngineConfig(quantization=QuantizationConfig(transformer_quant="svdq"))``.
Settings come from environment variables (see ``SVDQSettings``) because
FastVideo instantiates the config class without arguments; the instance is
then pickled into the spawned GPU worker, so the values are fixed in the
parent process.

Conversion: FastVideo loads the bf16 checkpoint as usual, then calls
``fsdp_load._maybe_quantize_model``. We wrap that hook so the SVD split, 4-bit
quantization and kernel packing happen right after the weights land on the
GPU and *before* layerwise offload is attached. The packed tensors become
non-persistent buffers (they stay resident; offload only streams parameters)
and the bf16 weights are freed. A safetensors cache skips the conversion on
later starts.
"""
from __future__ import annotations

import atexit
import os
import re
import threading
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from fastvideo.layers.linear import LinearBase, UnquantizedLinearMethod
from fastvideo.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.logger import init_logger

from . import nunchaku_backend
from .quant_math import smoothing_factor, svdquant_decompose

logger = init_logger(__name__)

_BLOCK = r"(?:^|\.)transformer_blocks\.\d+\."
LAYER_PRESETS = {
    "ffn": _BLOCK + r"ff\.(?:fc_in|fc_out)$",
    "attn": _BLOCK + r"attn\.(?:to_q|to_k|to_v|to_out)$",
    "all": _BLOCK + r"(?:ff\.(?:fc_in|fc_out)|attn\.(?:to_q|to_k|to_v|to_out))$",
}
_BUFFER_NAMES = ("qweight", "wscales", "smooth", "proj_down", "proj_up", "wcscales")


@dataclass(frozen=True)
class SVDQSettings:
    rank: int = 32
    precision: str = "int4"  # auto | nvfp4 | int4 (int4 is the production default: nvfp4 has an unresolved ~6-8pct kernel_vs_model mismatch on this GPU)
    layers: str = "all"  # preset name from LAYER_PRESETS or a raw regex
    cache_path: str = ""  # packed-weights safetensors; "" disables caching
    calib_path: str = ""  # activation stats from QUANT_MODE=calib; "" = no smoothing
    smooth_alpha: float = 0.5

    @classmethod
    def from_env(cls) -> SVDQSettings:
        return cls(
            rank=int(os.environ.get("SVDQ_RANK", cls.rank)),
            precision=os.environ.get("SVDQ_PRECISION", cls.precision),
            layers=os.environ.get("SVDQ_LAYERS", cls.layers),
            cache_path=os.environ.get("SVDQ_CACHE_PATH", cls.cache_path),
            calib_path=os.environ.get("SVDQ_CALIB_PATH", cls.calib_path),
            smooth_alpha=float(os.environ.get("SVDQ_SMOOTH_ALPHA", cls.smooth_alpha)),
        )

    @property
    def layer_regex(self) -> re.Pattern[str]:
        return re.compile(LAYER_PRESETS.get(self.layers, self.layers))


class _SVDQConfigBase(QuantizationConfig):

    def __init__(self, settings: SVDQSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or SVDQSettings.from_env()

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> _SVDQConfigBase:
        return cls(SVDQSettings(**config))

    def _wants(self, layer: torch.nn.Module, prefix: str) -> bool:
        return isinstance(layer, LinearBase) and self.settings.layer_regex.search(prefix) is not None


class SVDQuantConfig(_SVDQConfigBase):

    def get_name(self) -> str:
        return "svdq"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return SVDQuantLinearMethod(prefix, self.settings) if self._wants(layer, prefix) else None


class SVDQCalibConfig(_SVDQConfigBase):

    def get_name(self) -> str:
        return "svdq_calib"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return SVDQCalibMethod(prefix, self.settings) if self._wants(layer, prefix) else None


class SVDQuantLinearMethod(UnquantizedLinearMethod):
    """Creates a bf16 weight for the loader; runs W4A4 once ``convert_model_to_svdq`` packed it."""

    def __init__(self, prefix: str, settings: SVDQSettings) -> None:
        super().__init__()
        self.prefix = prefix
        self.settings = settings
        self.precision: str | None = None  # resolved on the GPU during conversion
        self.alpha: float | None = None

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not getattr(layer, "_svdq_ready", False):
            if getattr(layer, "_svdq_dense", False):  # shape the kernels can't take
                return super().apply(layer, x, bias)
            raise RuntimeError(f"SVDQ layer {self.prefix!r} was never converted: the FastVideo loader hook did not "
                               "run (incompatible FastVideo version?). Run `python -m tovi_quant.doctor`.")
        orig_dtype = x.dtype
        if orig_dtype not in (torch.bfloat16, torch.float16):
            x = x.to(torch.bfloat16)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        bufs = {name: getattr(layer, f"_svdq_{name}") for name in _BUFFER_NAMES if hasattr(layer, f"_svdq_{name}")}
        out = nunchaku_backend.linear_forward(
            x2d,
            bufs,
            out_features=layer.output_size,
            precision=self.precision,
            alpha=self.alpha,
        )
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.view(*x.shape[:-1], layer.output_size).to(orig_dtype)


# --------------------------------------------------------------------------- calibration

_CALIB_LOCK = threading.Lock()
_CALIB_STATS: dict[str, torch.Tensor] = {}
_CALIB_STATE = {"calls": 0, "path": "", "atexit": False}
_CALIB_SAVE_EVERY = int(os.environ.get("SVDQ_CALIB_SAVE_EVERY", "300"))


def save_calibration(path: str | None = None) -> None:
    path = path or _CALIB_STATE["path"]
    if not path or not _CALIB_STATS:
        return
    with _CALIB_LOCK:
        stats = {k: v.detach().float().cpu() for k, v in _CALIB_STATS.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(stats, tmp)
    os.replace(tmp, path)


class SVDQCalibMethod(UnquantizedLinearMethod):
    """bf16 linear that tracks max |x| per input channel, keyed by layer prefix."""

    def __init__(self, prefix: str, settings: SVDQSettings) -> None:
        super().__init__()
        self.prefix = prefix
        if not settings.calib_path:
            raise ValueError("QUANT_MODE=calib needs SVDQ_CALIB_PATH (where to write activation stats)")
        _CALIB_STATE["path"] = settings.calib_path
        if not _CALIB_STATE["atexit"]:
            atexit.register(save_calibration)
            _CALIB_STATE["atexit"] = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        amax = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        with _CALIB_LOCK:
            prev = _CALIB_STATS.get(self.prefix)
            _CALIB_STATS[self.prefix] = amax if prev is None else torch.maximum(prev, amax)
            _CALIB_STATE["calls"] += 1
            flush = _CALIB_STATE["calls"] % _CALIB_SAVE_EVERY == 0
        if flush:
            save_calibration()
        return super().apply(layer, x, bias)


# --------------------------------------------------------------------------- conversion


def _cache_metadata(settings: SVDQSettings, precision: str) -> dict[str, str]:
    meta = {k: str(v) for k, v in asdict(settings).items() if k != "cache_path"}
    meta["precision"] = precision
    if settings.calib_path and os.path.exists(settings.calib_path):
        meta["calib_mtime"] = str(int(os.path.getmtime(settings.calib_path)))
    return meta


def _load_cache(settings: SVDQSettings, precision: str) -> dict[str, torch.Tensor]:
    path = settings.cache_path
    if not path or not os.path.exists(path):
        return {}
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework="pt") as f:
        found = f.metadata() or {}
    expected = _cache_metadata(settings, precision)
    if found != expected:
        logger.warning("SVDQ cache %s was built with %s, need %s; re-converting", path, found, expected)
        return {}
    return load_file(path)


def _save_cache(settings: SVDQSettings, precision: str, tensors: dict[str, torch.Tensor]) -> None:
    if not settings.cache_path:
        return
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(os.path.abspath(settings.cache_path)), exist_ok=True)
    tmp = f"{settings.cache_path}.tmp"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, tmp, metadata=_cache_metadata(settings, precision))
    os.replace(tmp, settings.cache_path)
    logger.info("SVDQ: wrote packed-weight cache %s", settings.cache_path)


def _pack_one(weight: torch.Tensor, settings: SVDQSettings, precision: str, act_amax: torch.Tensor | None,
              device: torch.device) -> tuple[dict[str, torch.Tensor], float]:
    w = weight.detach().to(device)
    smooth = smoothing_factor(act_amax, w, settings.smooth_alpha) if act_amax is not None else None
    decomposed = svdquant_decompose(w, settings.rank, precision, smooth=smooth)
    return nunchaku_backend.pack_linear(decomposed, dtype=w.dtype), decomposed.weight_rel_error


@torch.no_grad()
def convert_model_to_svdq(model: torch.nn.Module) -> None:
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(getattr(module, "quant_method", None), SVDQuantLinearMethod)]
    if not targets:
        return
    from torch.distributed.tensor import DTensor

    t0 = time.time()
    settings: SVDQSettings = targets[0][1].quant_method.settings
    first_weight = targets[0][1].weight
    device = first_weight.device if first_weight.is_cuda or not torch.cuda.is_available() else torch.device(
        "cuda", torch.cuda.current_device())
    precision = nunchaku_backend.resolve_precision(settings.precision, device)
    cache = _load_cache(settings, precision)
    calib: dict[str, torch.Tensor] = {}
    if settings.calib_path:
        if not os.path.exists(settings.calib_path):
            raise FileNotFoundError(f"SVDQ_CALIB_PATH={settings.calib_path} does not exist; run QUANT_MODE=calib first")
        calib = torch.load(settings.calib_path, map_location="cpu")

    needed = ["qweight", "wscales", "smooth", "proj_down", "proj_up"]
    if precision == "nvfp4":
        needed += ["wcscales", "wtscale"]
    fresh: dict[str, torch.Tensor] = {}
    errors: list[float] = []
    freed = converted = dense = 0
    for _, layer in targets:
        method: SVDQuantLinearMethod = layer.quant_method
        weight = layer.weight
        if isinstance(weight, DTensor):
            raise NotImplementedError("SVDQ does not support FSDP-sharded DiT weights (use_fsdp_inference=False)")
        reason = nunchaku_backend.packable_shape(layer.output_size, layer.input_size, settings.rank)
        if reason:
            logger.warning("SVDQ: keeping %s in bf16 (%s)", method.prefix, reason)
            layer._svdq_dense = True
            dense += 1
            continue
        cached = {n: cache.get(f"{method.prefix}.{n}") for n in needed}
        if all(v is not None for v in cached.values()):
            packed = {n: v.to(device) for n, v in cached.items()}
        else:
            packed, err = _pack_one(weight, settings, precision, calib.get(method.prefix), device)
            errors.append(err)
            fresh.update({f"{method.prefix}.{k}": v for k, v in packed.items()})
        wtscale = packed.pop("wtscale", None)
        method.precision = precision
        method.alpha = float(wtscale.item()) if wtscale is not None else None
        for name, tensor in packed.items():
            layer.register_buffer(f"_svdq_{name}", tensor.to(device), persistent=False)
        freed += weight.numel() * weight.element_size()
        layer.register_parameter("weight", None)
        layer._svdq_ready = True
        converted += 1

    if fresh:
        _save_cache(settings, precision, {**cache, **fresh})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "SVDQ: %d linears -> %s W4A4 (rank %d, smoothing=%s), %d kept bf16, %.2f GiB bf16 freed, "
        "%s, %.1fs", converted, precision, settings.rank, "on" if calib else "off", dense, freed / (1 << 30),
        (f"weight rel err mean {sum(errors) / len(errors):.4f} max {max(errors):.4f}" if errors else "from cache"),
        time.time() - t0)


def _install_loader_hook() -> None:
    from fastvideo.models.loader import fsdp_load

    original = getattr(fsdp_load, "_maybe_quantize_model", None)
    if original is None:
        raise ImportError("fastvideo.models.loader.fsdp_load._maybe_quantize_model not found; this FastVideo "
                          "version is not supported by tovi_quant (see README)")
    if getattr(original, "_tovi_svdq", False):
        return

    def _maybe_quantize_model(model, *args, **kwargs):
        original(model, *args, **kwargs)
        if kwargs.get("defer_weight_conversion_until_lora_merge"):
            if any(isinstance(getattr(m, "quant_method", None), SVDQuantLinearMethod) for m in model.modules()):
                raise NotImplementedError("SVDQ + LoRA merge is not supported")
        convert_model_to_svdq(model)

    _maybe_quantize_model._tovi_svdq = True
    fsdp_load._maybe_quantize_model = _maybe_quantize_model


def _register() -> None:
    for name, cls in (("svdq", SVDQuantConfig), ("svdq_calib", SVDQCalibConfig)):
        if name not in QUANTIZATION_METHODS:
            register_quantization_config(name)(cls)


# Runs in the API process and again in each spawned worker (unpickling the
# config instance imports this module there).
_register()
_install_loader_hook()
