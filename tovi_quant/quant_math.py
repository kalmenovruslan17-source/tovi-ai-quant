"""Pure-torch SVDQuant math: smoothing, low-rank split, W4 quantization, fake-quant reference.

Nothing here touches CUDA kernels, so it runs (and is unit-tested) on CPU. The
packed layout the nunchaku kernels consume is produced in ``nunchaku_backend``.

SVDQuant (W4A4) for one linear ``y = x @ W^T``:

    s          per-input-channel smoothing factor (1 when uncalibrated)
    W_s        = W * s                 activation outliers migrate into weights
    W_s        ~ U @ D + R             rank-r SVD branch kept in 16 bit
    y          = Q4(x / s) @ Q4(R)^T   4-bit GEMM on the residual
               + x @ (D / s)^T @ U^T   low-rank branch on the raw input

On sm_120 (Blackwell, RTX PRO 6000 / RTX 50xx) the 4-bit format is NVFP4:
e2m1 values, an fp8-e4m3 scale per group of 16 and one fp32 tensor scale.
INT4 (group 64, 16-bit scales) is what pre-Blackwell GPUs run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
INT4_MAX = 7.0
GROUP_SIZE = {"nvfp4": 16, "int4": 64}

# Magnitudes of the 8 non-negative e2m1 codes; code | 8 is the negative twin.
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def fp4_encode(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest e2m1 encoding of values already scaled into [-6, 6]."""
    midpoints = torch.tensor(_FP4_MIDPOINTS, dtype=x.dtype, device=x.device)
    magnitude = torch.bucketize(x.abs(), midpoints)
    # Values that round to zero get +0 (code 0), matching deepcompressor's codebook argmin.
    return (magnitude + 8 * ((x < 0) & (magnitude > 0))).to(torch.int32)


def fp4_decode(codes: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=codes.device)
    magnitude = values[(codes & 7).long()]
    return torch.where(codes >= 8, -magnitude, magnitude)


@dataclass
class QuantizedWeight:
    """4-bit weight before kernel packing.

    ``codes``: (oc, ic) int32 -- e2m1 codes in [0, 15] for nvfp4, signed ints
    in [-8, 7] for int4. ``group_scale``: (oc, ic // group) fp32, already
    representable in the storage dtype (fp8-e4m3 / bf16). ``tensor_scale``:
    the nvfp4 global scale (``alpha`` in the kernel), 1.0 for int4.
    """

    codes: torch.Tensor
    group_scale: torch.Tensor
    tensor_scale: float
    precision: str

    @property
    def group_size(self) -> int:
        return GROUP_SIZE[self.precision]

    def dequantize(self) -> torch.Tensor:
        oc, ic = self.codes.shape
        if self.precision == "nvfp4":
            values = fp4_decode(self.codes)
        else:
            values = self.codes.float()
        values = values.view(oc, ic // self.group_size, self.group_size)
        return (values * self.group_scale.unsqueeze(-1)).view(oc, ic) * self.tensor_scale


def _check_precision(precision: str) -> None:
    if precision not in GROUP_SIZE:
        raise ValueError(f"precision must be one of {sorted(GROUP_SIZE)}, got {precision!r}")


def quantize_weight(weight: torch.Tensor, precision: str) -> QuantizedWeight:
    """Round-to-nearest group quantization of a 2D weight."""
    _check_precision(precision)
    w = weight.float()
    oc, ic = w.shape
    group = GROUP_SIZE[precision]
    if ic % group:
        raise ValueError(f"in_features={ic} is not a multiple of the {precision} group size {group}")
    wg = w.view(oc, ic // group, group)
    group_amax = wg.abs().amax(dim=-1)

    if precision == "nvfp4":
        amax = float(w.abs().max())
        tensor_scale = amax / (FP4_MAX * FP8_E4M3_MAX) if amax > 0 else 1.0
        scale = (group_amax / (FP4_MAX * tensor_scale)).clamp(max=FP8_E4M3_MAX)
        scale = scale.to(torch.float8_e4m3fn).float()
        # All-zero (or fp8-underflowing) groups: any scale encodes them as 0.
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = fp4_encode((wg / (scale.unsqueeze(-1) * tensor_scale)).clamp(-FP4_MAX, FP4_MAX))
    else:
        tensor_scale = 1.0
        scale = (group_amax / INT4_MAX).to(torch.bfloat16).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)

    return QuantizedWeight(codes.view(oc, ic), scale, tensor_scale, precision)


def fake_quantize_activation(x: torch.Tensor, precision: str) -> torch.Tensor:
    """Simulate the kernel's per-token dynamic 4-bit activation quantization."""
    _check_precision(precision)
    group = GROUP_SIZE[precision]
    shape = x.shape
    xg = x.float().reshape(-1, shape[-1] // group, group)
    group_amax = xg.abs().amax(dim=-1, keepdim=True)
    if precision == "nvfp4":
        scale = (group_amax / FP4_MAX).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        deq = fp4_decode(fp4_encode((xg / scale).clamp(-FP4_MAX, FP4_MAX))) * scale
    else:
        scale = (group_amax / INT4_MAX).to(torch.bfloat16).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        deq = (xg / scale).round().clamp(-8, 7) * scale
    return deq.reshape(shape)


def smoothing_factor(act_amax: torch.Tensor, weight: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant migration strength: s_j = max|x_j|^a / max|W_:,j|^(1-a)."""
    w_amax = weight.float().abs().amax(dim=0).clamp(min=1e-5)
    a_amax = act_amax.float().to(w_amax.device).clamp(min=1e-5)
    s = a_amax.pow(alpha) / w_amax.pow(1.0 - alpha)
    # The kernel divides activations by s in 16-bit; keep it in a sane range.
    return s.clamp(min=1e-4, max=1e4)


@dataclass
class SVDQuantLinear:
    """All pieces of one SVDQuant linear, unpacked (fp32 except ``qweight``)."""

    qweight: QuantizedWeight
    lora_up: torch.Tensor  # (oc, r)
    lora_down: torch.Tensor  # (r, ic), already divided by ``smooth`` (acts on raw x)
    smooth: torch.Tensor  # (ic,)
    weight_rel_error: float  # ||W - W_hat||_F / ||W||_F in the original space

    @property
    def precision(self) -> str:
        return self.qweight.precision

    @property
    def rank(self) -> int:
        return self.lora_up.shape[1]


def svdquant_decompose(
    weight: torch.Tensor,
    rank: int,
    precision: str,
    smooth: torch.Tensor | None = None,
    niter: int = 4,
    seed: int = 0,
) -> SVDQuantLinear:
    """Split ``weight`` (oc, ic) into a 16-bit rank-``rank`` branch plus a 4-bit residual."""
    _check_precision(precision)
    w = weight.float()
    oc, ic = w.shape
    if smooth is None:
        smooth = torch.ones(ic, dtype=torch.float32, device=w.device)
    smooth = smooth.float().to(w.device)
    w_s = w * smooth.unsqueeze(0)

    if rank > 0:
        q = min(rank + 16, oc, ic)
        devices = [w.device] if w.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(w_s, q=q, niter=niter)
        up = u[:, :rank] * s[:rank].unsqueeze(0)
        down = v[:, :rank].T.contiguous()
        residual = w_s - up @ down
    else:
        up = w.new_zeros(oc, 0)
        down = w.new_zeros(0, ic)
        residual = w_s

    qweight = quantize_weight(residual, precision)
    w_hat = (qweight.dequantize() + up @ down) / smooth.unsqueeze(0)
    rel_error = float(torch.linalg.norm(w_hat - w) / torch.linalg.norm(w).clamp(min=1e-12))
    return SVDQuantLinear(
        qweight=qweight,
        lora_up=up,
        lora_down=down / smooth.unsqueeze(0),
        smooth=smooth,
        weight_rel_error=rel_error,
    )


def reference_forward(x: torch.Tensor, layer: SVDQuantLinear) -> torch.Tensor:
    """What the W4A4 kernel computes, in fp32 (bias excluded)."""
    xf = x.float()
    x_q = fake_quantize_activation(xf / layer.smooth, layer.precision)
    main = x_q @ layer.qweight.dequantize().T
    low_rank = (xf @ layer.lora_down.T) @ layer.lora_up.T
    return main + low_rank
