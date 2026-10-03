"""Pod preflight for quantization: environment + per-layer kernel check. No model weights needed.

    python -m tovi_quant.doctor            # ~1-2 min on the GPU
    python -m tovi_quant.doctor --json out.json

For each MiniMax-H3 linear shape it compares, on random weights with outlier
channels:
  * bf16 F.linear                  (reference output and timing)
  * SVDQuant via nunchaku kernels  vs. our fp32 fake-quant model of the same
    math -> "kernel_vs_model" must be ~1e-2 or below, otherwise the packing
    does not match the installed kernels
  * FastVideo native NVFP4 (flashinfer), the QUANT_MODE=fp4 path
and runs the FastVideo integration glue (config -> loader hook -> forward).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
import traceback

import torch
import torch.nn.functional as F

# (name, in_features, out_features) of the H3 block linears (hidden 5376, ffn 14336).
H3_SHAPES = [
    ("attn.to_q", 5376, 5376),
    ("ff.fc_in", 5376, 28672),
    ("ff.fc_out", 14336, 5376),
]


def _version(module: str) -> str:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__}: {exc})"


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _bench(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _make_problem(m: int, k: int, n: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * (k ** -0.5)
    x = torch.randn(m, k, device="cuda", generator=g)
    outliers = torch.randperm(k, device="cuda", generator=g)[:8]
    x[:, outliers] *= 25.0  # diffusion activations have a few huge channels
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def check_svdq(m: int, rank: int, report: dict) -> None:
    from tovi_quant import nunchaku_backend as nb
    from tovi_quant.quant_math import reference_forward, smoothing_factor, svdquant_decompose

    precision = nb.resolve_precision("auto")
    report["svdq_precision"] = precision
    rows = []
    for name, k, n in H3_SHAPES:
        x, w = _make_problem(m, k, n)
        y_bf16 = F.linear(x, w)
        row = {"layer": name, "m": m, "k": k, "n": n,
               "bf16_ms": _bench(lambda: F.linear(x, w))}
        for smooth_on in (False, True):
            smooth = smoothing_factor(x.float().abs().amax(0), w) if smooth_on else None
            layer = svdquant_decompose(w, rank, precision, smooth=smooth)
            bufs = nb.pack_linear(layer)
            alpha = float(bufs.pop("wtscale")) if "wtscale" in bufs else None
            run = lambda: nb.linear_forward(x, bufs, out_features=n, precision=precision, alpha=alpha)  # noqa: E731
            y_q = run()
            y_model = reference_forward(x, layer)
            tag = "smooth" if smooth_on else "plain"
            row[f"svdq_{tag}_err_vs_bf16"] = _rel(y_q, y_bf16)
            row[f"svdq_{tag}_kernel_vs_model"] = _rel(y_q, y_model)
            if not smooth_on:
                row["svdq_ms"] = _bench(run)
        rows.append(row)
    report["svdq"] = rows


def check_fastvideo_fp4(m: int, report: dict) -> None:
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config, convert_model_to_nvfp4

    rows = []
    for name, k, n in H3_SHAPES:
        x, w = _make_problem(m, k, n)
        prefix = f"transformer_blocks.0.{name}" if name.startswith("ff.") else None
        if prefix is None:  # FastVideo's NVFP4 set for H3 is FFN-only
            continue
        layer = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16, quant_config=NVFP4Config(),
                                 prefix=prefix).cuda()
        layer.weight.data.copy_(w)
        convert_model_to_nvfp4(layer)
        y = layer(x)[0]
        rows.append({"layer": name, "fp4_err_vs_bf16": _rel(y, F.linear(x, w)),
                     "fp4_ms": _bench(lambda: layer(x))})
    report["fastvideo_fp4"] = rows


def check_integration(report: dict) -> None:
    """Config -> quant_method -> loader hook -> forward, on one tiny FFN-shaped layer."""
    from tovi_quant.fastvideo_svdq import SVDQSettings, SVDQuantConfig, convert_model_to_svdq

    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.models.loader import fsdp_load

    cfg = SVDQuantConfig(SVDQSettings(layers="ffn"))
    model = torch.nn.Module()
    model.fc = ReplicatedLinear(1024, 2048, bias=True, params_dtype=torch.bfloat16, quant_config=cfg,
                                prefix="transformer_blocks.0.ff.fc_in")
    model = model.cuda()
    x, w = _make_problem(512, 1024, 2048)
    model.fc.weight.data.copy_(w)
    model.fc.bias.data.normal_()
    y_ref = F.linear(x, w, model.fc.bias)
    assert getattr(fsdp_load._maybe_quantize_model, "_tovi_svdq", False), "loader hook not installed"
    convert_model_to_svdq(model)
    assert model.fc.weight is None, "bf16 weight was not freed"
    y = model.fc(x.view(1, 512, 1024))[0]
    report["integration_err_vs_bf16"] = _rel(y.view(512, 2048), y_ref)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=8192, help="GEMM M (sequence tokens)")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args()

    report: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fastvideo": _version("fastvideo"),
        "flashinfer": _version("flashinfer"),
        "nunchaku_spec": str(importlib.util.find_spec("nunchaku") is not None),
    }
    if not torch.cuda.is_available():
        print(json.dumps(report, indent=2))
        print("no CUDA device", file=sys.stderr)
        return 1
    report["gpu"] = torch.cuda.get_device_name()
    report["capability"] = "sm_%d%d" % torch.cuda.get_device_capability()

    failures = []
    for name, fn in (("svdq", lambda: check_svdq(args.tokens, args.rank, report)),
                     ("fastvideo_fp4", lambda: check_fastvideo_fp4(args.tokens, report)),
                     ("integration", lambda: check_integration(report))):
        try:
            fn()
        except Exception:  # noqa: BLE001 - report every failing check, not just the first
            failures.append(name)
            report[f"{name}_error"] = traceback.format_exc()

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    bad_kernel = [r["layer"] for r in report.get("svdq", []) if r["svdq_plain_kernel_vs_model"] > 0.05]
    if bad_kernel:
        failures.append(f"svdq kernel/model mismatch on {bad_kernel} (packing vs installed nunchaku?)")
    print("FAILED: " + "; ".join(failures) if failures else "OK", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
