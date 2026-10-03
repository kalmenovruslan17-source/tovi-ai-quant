"""Bridge to the nunchaku W4A4 kernels: weight packing and the linear forward.

We use nunchaku's own packer and op wrappers so the packed layout always
matches the compiled kernels. ``import nunchaku`` eagerly imports its
diffusers-based model zoo, which drags in diffusers/transformers versions that
may not match FastVideo's pins; we only need ``nunchaku._C``, ``nunchaku.ops``
and ``nunchaku.lora.flux.packer``, so the parent packages are stubbed and their
``__init__`` never runs.
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import math
import os
import sys
import types
from functools import lru_cache

import torch

from .quant_math import SVDQuantLinear

_STUB_PACKAGES = ("nunchaku", "nunchaku.lora", "nunchaku.lora.flux")


def _stub_package(name: str, path: list[str]) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = path
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = path
    sys.modules[name] = module


def _import_nunchaku(module: str) -> types.ModuleType:
    spec = importlib.util.find_spec("nunchaku")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("nunchaku is not installed; see requirements-quant.txt (QUANT_MODE=svdq needs it)")
    roots = list(spec.submodule_search_locations)
    for name in _STUB_PACKAGES:
        sub = name.split(".")[1:]
        _stub_package(name, [os.path.join(root, *sub) for root in roots])
    return importlib.import_module(module)


@lru_cache(maxsize=1)
def _packer_cls() -> type:
    """Pure-torch packer; importable without a GPU or the compiled extension."""
    return _import_nunchaku("nunchaku.lora.flux.packer").NunchakuWeightPacker


@lru_cache(maxsize=1)
def _ops() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        quantize_act=_import_nunchaku("nunchaku.ops.quantize").svdq_quantize_w4a4_act_fuse_lora_cuda,
        gemm=_import_nunchaku("nunchaku.ops.gemm").svdq_gemm_w4a4_cuda,
    )


def nunchaku_available() -> bool:
    try:
        _ops()
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return False
    return True


def resolve_precision(precision: str, device: torch.device | int | None = None) -> str:
    """``auto`` -> nvfp4 on Blackwell consumer/workstation parts (sm_120/121), else int4."""
    if precision in ("nvfp4", "int4"):
        return precision
    if precision != "auto":
        raise ValueError(f"SVDQ precision must be auto|nvfp4|int4, got {precision!r}")
    major, minor = torch.cuda.get_device_capability(device)
    return "nvfp4" if (major, minor) in ((12, 0), (12, 1)) else "int4"


def packable_shape(out_features: int, in_features: int, rank: int) -> str | None:
    """Return why the kernels can't take this shape, or None if they can."""
    if rank <= 0 or rank % 16:
        return f"rank={rank} must be a positive multiple of 16"
    # Packer tiling (bits=4, warp_n=128): mem_n=128 rows, mem_k*num_k_unrolls=128 cols.
    if out_features % 128:
        return f"out_features={out_features} is not a multiple of 128"
    if in_features % 128:
        return f"in_features={in_features} is not a multiple of 128"
    return None


def pack_linear(layer: SVDQuantLinear, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Pack an ``SVDQuantLinear`` into the tensors ``gemm_w4a4`` consumes.

    Mirrors deepcompressor's ``convert_to_nunchaku_w4x4y16_linear_weight`` for
    a per-tensor (nvfp4) or no (int4) outer scale. Bias is not packed: the
    FastVideo layer keeps its own bias and adds it after the GEMM.
    """
    qw = layer.qweight
    oc, ic = qw.codes.shape
    reason = packable_shape(oc, ic, layer.rank)
    if reason:
        raise ValueError(reason)
    packer = _packer_cls()(bits=4, warp_n=128)
    group = qw.group_size
    device = qw.codes.device

    scale = qw.group_scale.to(dtype).view(oc, 1, ic // group, 1)
    smooth = layer.smooth.to(dtype).view(-1, 1)
    packed = {
        "qweight": packer.pack_weight(qw.codes.to(torch.int32).contiguous()),
        # nvfp4 goes through pack_micro_scale (-> fp8 e4m3), int4 through pack_scale.
        "wscales": packer.pack_scale(scale, group_size=group),
        "smooth": packer.pack_scale(smooth, group_size=-1),
        "proj_down": packer.pack_lowrank_weight(layer.lora_down.to(dtype).contiguous(), down=True),
        "proj_up": packer.pack_lowrank_weight(layer.lora_up.to(dtype).contiguous(), down=False),
    }
    if qw.precision == "nvfp4":
        packed["wcscales"] = torch.ones(oc, dtype=dtype, device=device)
        packed["wtscale"] = torch.tensor([qw.tensor_scale], dtype=torch.float32)
    return packed


def linear_forward(
    x: torch.Tensor,
    bufs: dict[str, torch.Tensor],
    *,
    out_features: int,
    precision: str,
    alpha: float | None,
) -> torch.Tensor:
    """W4A4 linear on a 2D bf16/fp16 input (same calls as nunchaku's SVDQW4A4Linear)."""
    nk = _ops()
    fp4 = precision == "nvfp4"
    quantized_x, ascales, lora_act = nk.quantize_act(
        x,
        lora_down=bufs["proj_down"],
        smooth=bufs["smooth"],
        fp4=fp4,
        pad_size=256,
    )
    output = torch.empty(x.shape[0], out_features, dtype=x.dtype, device=x.device)
    rank = bufs["proj_up"].shape[1]
    nk.gemm(
        act=quantized_x,
        wgt=bufs["qweight"],
        out=output,
        ascales=ascales,
        wscales=bufs["wscales"],
        lora_act_in=lora_act,
        lora_up=bufs["proj_up"],
        bias=None,
        fp4=fp4,
        alpha=alpha,
        wcscales=bufs.get("wcscales"),
        act_unsigned=False,
        lora_scales=[1.0] * math.ceil(rank / 16),
    )
    return output
