import pytest
import torch

from tovi_quant.quant_math import (
    fake_quantize_activation,
    fp4_decode,
    fp4_encode,
    quantize_weight,
    reference_forward,
    smoothing_factor,
    svdquant_decompose,
)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _problem(oc=256, ic=512, m=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(oc, ic, generator=g) * ic**-0.5
    x = torch.randn(m, ic, generator=g)
    x[:, [3, 77, 300]] *= 30.0  # activation outlier channels
    return x, w


def test_fp4_roundtrip_on_grid():
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = torch.cat([grid, -grid[1:]])
    assert torch.equal(fp4_decode(fp4_encode(values)), values)
    # round-to-nearest between grid points, and signed zero collapses to code 0
    assert fp4_decode(fp4_encode(torch.tensor([0.7, 2.6, -5.5, -0.1]))).tolist() == [0.5, 3.0, -6.0, 0.0]
    assert fp4_encode(torch.tensor([-0.1])).item() == 0


@pytest.mark.parametrize("precision,tol", [("nvfp4", 0.12), ("int4", 0.15)])
def test_quantize_weight_error_and_scales(precision, tol):
    _, w = _problem()
    q = quantize_weight(w, precision)
    assert _rel(q.dequantize(), w) < tol
    if precision == "nvfp4":
        assert q.group_scale.max() <= 448
        assert torch.equal(q.group_scale, q.group_scale.to(torch.float8_e4m3fn).float())
        assert int(q.codes.min()) >= 0 and int(q.codes.max()) <= 15
    else:
        assert torch.equal(q.group_scale, q.group_scale.to(torch.bfloat16).float())
        assert int(q.codes.min()) >= -8 and int(q.codes.max()) <= 7


def test_zero_weight_is_exact():
    q = quantize_weight(torch.zeros(128, 128), "nvfp4")
    assert torch.count_nonzero(q.dequantize()) == 0


@pytest.mark.parametrize("precision", ["nvfp4", "int4"])
def test_low_rank_branch_reduces_error(precision):
    _, w = _problem()
    w[:, 10] *= 40.0  # weight outlier column: exactly what the SVD branch should absorb
    plain = svdquant_decompose(w, 0, precision).weight_rel_error
    svd = svdquant_decompose(w, 32, precision).weight_rel_error
    assert svd < plain * 0.7


@pytest.mark.parametrize("precision", ["nvfp4", "int4"])
def test_smoothing_helps_activation_outliers(precision):
    x, w = _problem()
    y = x @ w.T
    plain = svdquant_decompose(w, 32, precision)
    smooth = svdquant_decompose(w, 32, precision, smooth=smoothing_factor(x.abs().amax(0), w))
    err_plain = _rel(reference_forward(x, plain), y)
    err_smooth = _rel(reference_forward(x, smooth), y)
    assert err_smooth < err_plain
    assert err_smooth < 0.1


def test_smoothing_preserves_full_precision_math():
    """With 4-bit rounding removed, smoothing + low-rank must reproduce W exactly."""
    x, w = _problem()
    s = smoothing_factor(x.abs().amax(0), w)
    layer = svdquant_decompose(w, 32, "nvfp4", smooth=s)
    w_hat = (layer.qweight.dequantize() + layer.lora_up @ (layer.lora_down * s)) / s
    assert _rel(w_hat, w) == pytest.approx(layer.weight_rel_error, rel=1e-4)
    # the low-rank branch acts on the raw input: x @ down_raw^T == (x / s) @ down_smoothed^T
    assert torch.allclose(x @ layer.lora_down.T, (x / s) @ (layer.lora_down * s).T, rtol=1e-4, atol=1e-4)


def test_fake_activation_quant_is_per_token():
    x = torch.randn(4, 64)
    x[0] *= 1000.0  # one huge token must not wreck the others
    xq = fake_quantize_activation(x, "nvfp4")
    assert _rel(xq[1:], x[1:]) < 0.2


def test_pack_layout_shapes():
    nb = pytest.importorskip("tovi_quant.nunchaku_backend")
    try:
        nb._packer_cls()
    except ImportError:
        pytest.skip("nunchaku packer not importable")
    _, w = _problem(oc=256, ic=384)
    for precision, scale_dtype in (("nvfp4", torch.float8_e4m3fn), ("int4", torch.float32)):
        packed = nb.pack_linear(svdquant_decompose(w, 32, precision), dtype=torch.bfloat16)
        assert packed["qweight"].shape == (256, 192) and packed["qweight"].dtype == torch.int8
        assert packed["proj_down"].shape == (384, 32) and packed["proj_up"].shape == (256, 32)
        group = 16 if precision == "nvfp4" else 64
        assert packed["wscales"].shape == (384 // group, 256)
        assert ("wtscale" in packed) == (precision == "nvfp4")
        if precision == "nvfp4":
            assert packed["wscales"].dtype == scale_dtype


def test_packable_shape():
    from tovi_quant.nunchaku_backend import packable_shape

    assert packable_shape(5376, 5376, 32) is None
    assert packable_shape(28672, 5376, 32) is None
    assert packable_shape(5376, 14336, 32) is None
    assert packable_shape(100, 5376, 32)
    assert packable_shape(5376, 5376, 24)
