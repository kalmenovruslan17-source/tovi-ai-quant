"""QUANT_MODE -> FastVideo ``EngineConfig.quantization``.

    bf16   production default, nothing changes
    fp4    FastVideo's built-in NVFP4 (flashinfer), MiniMax-H3 FFN linears only,
           round-to-nearest, no extra packages
    svdq   nunchaku SVDQuant W4A4 (NVFP4 on sm_120, INT4 on older GPUs) for the
           transformer-block linears picked by SVDQ_LAYERS
    calib  bf16 run that records activation maxima for SVDQuant smoothing
"""
from __future__ import annotations

import os

QUANT_MODES = ("bf16", "fp4", "svdq", "calib")
_MODEL_TAG = {"t2va": "fasth3", "ref2va": "minimax_h3"}


def get_quant_mode() -> str:
    mode = os.environ.get("QUANT_MODE", "bf16").strip().lower()
    if mode not in QUANT_MODES:
        raise ValueError(f"QUANT_MODE must be one of {QUANT_MODES}, got {mode!r}")
    return mode


def _default_paths(worker_mode: str) -> None:
    """Per-model cache/calibration files, unless set explicitly (an empty value disables)."""
    cache_dir = os.environ.get("QUANT_CACHE_DIR", "/workspace/quant_cache")
    tag = _MODEL_TAG[worker_mode]
    calib = os.path.join(cache_dir, f"{tag}-calib.pt")
    if "SVDQ_CALIB_PATH" not in os.environ:
        # svdq picks the stats up automatically once a calib run produced them.
        if get_quant_mode() == "calib" or os.path.exists(calib):
            os.environ["SVDQ_CALIB_PATH"] = calib
    if "SVDQ_CACHE_PATH" not in os.environ:
        smooth = "smooth" if os.environ.get("SVDQ_CALIB_PATH") else "nosmooth"
        name = "-".join([
            tag,
            os.environ.get("SVDQ_PRECISION", "auto"),
            f"r{os.environ.get('SVDQ_RANK', '32')}",
            os.environ.get("SVDQ_LAYERS", "all").replace("/", "_"),
            smooth,
        ])
        os.environ["SVDQ_CACHE_PATH"] = os.path.join(cache_dir, f"svdq-{name}.safetensors")


def build_quantization(mode: str, worker_mode: str):
    """Return a ``fastvideo.api.QuantizationConfig`` for ``mode`` (None for bf16)."""
    if mode == "bf16":
        return None
    from fastvideo.api import QuantizationConfig

    if mode == "fp4":
        return QuantizationConfig(transformer_quant="NVFP4")
    _default_paths(worker_mode)
    # Registers "svdq"/"svdq_calib" with FastVideo and hooks its loader.
    from . import fastvideo_svdq  # noqa: F401

    return QuantizationConfig(transformer_quant="svdq" if mode == "svdq" else "svdq_calib")


def describe(mode: str, worker_mode: str) -> str:
    if mode not in ("svdq", "calib"):
        return mode
    _default_paths(worker_mode)
    keys = ("SVDQ_PRECISION", "SVDQ_RANK", "SVDQ_LAYERS", "SVDQ_CALIB_PATH", "SVDQ_CACHE_PATH")
    return mode + " " + " ".join(f"{k}={os.environ.get(k, '<default>')}" for k in keys)
