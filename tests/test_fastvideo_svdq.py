"""FastVideo glue on CPU: config registration, loader hook, conversion, cache, calibration.

The nunchaku CUDA kernels are replaced by the fp32 fake-quant model of the same
math (``reference_forward``); kernel-vs-model agreement is checked on the GPU by
``python -m tovi_quant.doctor``.
"""
import pickle

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("fastvideo.layers.linear")

from fastvideo.layers.linear import ReplicatedLinear  # noqa: E402
from fastvideo.layers.quantization import get_quantization_config  # noqa: E402
from fastvideo.models.loader import fsdp_load  # noqa: E402

from tovi_quant import fastvideo_svdq as fs  # noqa: E402
from tovi_quant import nunchaku_backend as nb  # noqa: E402
from tovi_quant.quant_math import QuantizedWeight, SVDQuantLinear, reference_forward  # noqa: E402

PREFIXES = ["transformer_blocks.0.ff.fc_in", "transformer_blocks.0.ff.fc_out", "transformer_blocks.0.attn.to_q",
            "token_refiner.refiner_blocks.0.ff.fc_in", "proj_out"]


@pytest.fixture(autouse=True)
def fake_kernels(monkeypatch):

    def pack(layer, dtype=torch.bfloat16):
        q = layer.qweight
        out = {"qweight": q.codes, "wscales": q.group_scale, "smooth": layer.smooth, "proj_down": layer.lora_down,
               "proj_up": layer.lora_up}
        if q.precision == "nvfp4":
            out["wcscales"] = torch.ones(q.codes.shape[0])
            out["wtscale"] = torch.tensor([q.tensor_scale])
        return out

    def forward(x, bufs, *, out_features, precision, alpha):
        q = QuantizedWeight(bufs["qweight"], bufs["wscales"], alpha if alpha is not None else 1.0, precision)
        layer = SVDQuantLinear(q, bufs["proj_up"], bufs["proj_down"], bufs["smooth"], 0.0)
        return reference_forward(x, layer).to(x.dtype)

    monkeypatch.setattr(nb, "pack_linear", pack)
    monkeypatch.setattr(nb, "linear_forward", forward)
    monkeypatch.setattr(nb, "resolve_precision", lambda precision, device=None: "nvfp4")


def _model(config, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        ReplicatedLinear(256, 384, bias=(i == 0), params_dtype=torch.bfloat16, quant_config=config, prefix=p)
        for i, p in enumerate(PREFIXES))
    for layer in model.layers:
        layer.weight.data.normal_(std=256**-0.5)
        if layer.bias is not None:
            layer.bias.data.normal_()
    return model


def _inputs():
    x = torch.randn(2, 64, 256)
    x[..., 7] *= 20.0
    return x.to(torch.bfloat16)


def _dense_reference(model, x):
    return [F.linear(x.float(), l.weight.float(), None if l.bias is None else l.bias.float()) for l in model.layers]


def test_registered_and_picklable():
    assert get_quantization_config("svdq") is fs.SVDQuantConfig
    assert get_quantization_config("svdq_calib") is fs.SVDQCalibConfig
    cfg = fs.SVDQuantConfig(fs.SVDQSettings(rank=64, layers="ffn"))
    clone = pickle.loads(pickle.dumps(cfg))
    assert clone.settings == cfg.settings


def test_loader_hook_installed():
    assert getattr(fsdp_load._maybe_quantize_model, "_tovi_svdq", False)


@pytest.mark.parametrize("layers,expected", [
    ("all", PREFIXES[:3]),
    ("ffn", PREFIXES[:2]),
    ("attn", PREFIXES[2:3]),
])
def test_layer_selection(layers, expected):
    model = _model(fs.SVDQuantConfig(fs.SVDQSettings(layers=layers)))
    picked = [l.prefix for l in model.layers if isinstance(l.quant_method, fs.SVDQuantLinearMethod)]
    assert picked == expected


def test_convert_forward_and_cache(tmp_path, monkeypatch):
    settings = fs.SVDQSettings(rank=32, cache_path=str(tmp_path / "cache.safetensors"))
    model = _model(fs.SVDQuantConfig(settings))
    x = _inputs()
    ref = _dense_reference(model, x)

    fsdp_load._maybe_quantize_model(model)  # the wrapped FastVideo hook
    quantized = model.layers[:3]
    assert all(l.weight is None and l._svdq_ready for l in quantized)
    assert all(l.weight is not None for l in model.layers[3:])
    assert not any(n.startswith("layers.0._svdq") for n in model.state_dict())  # non-persistent
    outs = [l(x)[0] for l in model.layers]
    for out, r in zip(outs[:3], ref[:3]):
        assert out.dtype == torch.bfloat16 and out.shape == (2, 64, 384)
        assert (out.float() - r).norm() / r.norm() < 0.2  # no smoothing + a 20x outlier channel
    for out, r in zip(outs[3:], ref[3:]):  # untouched layers stay exact bf16
        assert torch.allclose(out.float(), r, atol=0.05, rtol=0.02)

    # second start: same weights come from the cache, no SVD
    assert (tmp_path / "cache.safetensors").exists()
    calls = []
    monkeypatch.setattr(fs, "_pack_one", lambda *a, **k: calls.append(1))
    model2 = _model(fs.SVDQuantConfig(settings))
    fs.convert_model_to_svdq(model2)
    assert not calls
    for a, b in zip(model.layers[:3], model2.layers[:3]):
        assert torch.equal(a(x)[0], b(x)[0])


def test_cache_invalidated_by_settings(tmp_path, monkeypatch):
    path = str(tmp_path / "cache.safetensors")
    fs.convert_model_to_svdq(_model(fs.SVDQuantConfig(fs.SVDQSettings(rank=32, cache_path=path))))
    calls = []
    real = fs._pack_one
    monkeypatch.setattr(fs, "_pack_one", lambda *a, **k: calls.append(1) or real(*a, **k))
    fs.convert_model_to_svdq(_model(fs.SVDQuantConfig(fs.SVDQSettings(rank=48, cache_path=path))))
    assert len(calls) == 3


def test_unconverted_layer_fails_loudly():
    model = _model(fs.SVDQuantConfig(fs.SVDQSettings()))
    with pytest.raises(RuntimeError, match="never converted"):
        model.layers[0](_inputs())


def test_calibration_then_smoothed_svdq(tmp_path):
    calib_path = str(tmp_path / "calib.pt")
    settings = fs.SVDQSettings(calib_path=calib_path)
    calib_model = _model(fs.SVDQCalibConfig(settings))
    x = _inputs()
    ref = _dense_reference(calib_model, x)
    for out, r in zip([l(x)[0] for l in calib_model.layers], ref):
        assert torch.allclose(out.float(), r, atol=0.05, rtol=0.02)  # calibration runs plain bf16
    fs.save_calibration()
    stats = torch.load(calib_path)
    assert sorted(stats) == sorted(PREFIXES[:3])
    assert torch.allclose(stats[PREFIXES[0]], x.float().abs().amax(dim=(0, 1)))

    plain = _model(fs.SVDQuantConfig(fs.SVDQSettings()))
    smooth = _model(fs.SVDQuantConfig(settings))
    fs.convert_model_to_svdq(plain)
    fs.convert_model_to_svdq(smooth)
    err = lambda m: float((m.layers[0](x)[0].float() - ref[0]).norm() / ref[0].norm())  # noqa: E731
    assert err(smooth) < err(plain)


def test_missing_calibration_file_is_an_error(tmp_path):
    model = _model(fs.SVDQuantConfig(fs.SVDQSettings(calib_path=str(tmp_path / "nope.pt"))))
    with pytest.raises(FileNotFoundError):
        fs.convert_model_to_svdq(model)
