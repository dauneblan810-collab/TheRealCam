import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from therealcam_trainer import reference as R
from therealcam_trainer.errors import TrcmError
from therealcam_trainer.trcm import encode_trcm


def t64(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


@pytest.mark.parametrize("cin, cout, k, stride, padding, groups, h, w, bias", [
    (3, 16, 3, 1, 1, 1, 64, 64, True),
    (6, 4, 3, 2, 1, 2, 11, 9, True),
    (4, 8, 5, 3, 2, 4, 13, 10, False),
    (2, 3, 1, 1, 0, 1, 5, 7, True),
    (6, 6, 3, 2, 0, 3, 8, 8, True),
])
def test_conv2d_matches_torch(cin, cout, k, stride, padding, groups, h, w, bias):
    rng = np.random.default_rng(cin * 100 + k)
    x = rng.normal(size=(cin, h, w))
    wt = rng.normal(size=(cout, cin // groups, k, k))
    b = rng.normal(size=cout) if bias else None
    got = R.conv2d(x, wt, b, stride, padding, groups)
    want = F.conv2d(t64(x)[None], t64(wt), None if b is None else t64(b), stride, padding, 1, groups)[0].numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("k, s, h, w", [(2, 2, 64, 64), (3, 2, 9, 11), (2, 3, 10, 7), (3, 3, 3, 5)])
def test_pools_match_torch(k, s, h, w):
    x = np.random.default_rng(k * 10 + s).normal(size=(5, h, w))
    np.testing.assert_allclose(R.pool2d(x, k, s, "max"), F.max_pool2d(t64(x)[None], k, s)[0].numpy())
    np.testing.assert_allclose(R.pool2d(x, k, s, "avg"), F.avg_pool2d(t64(x)[None], k, s)[0].numpy(), atol=1e-12)


def test_batchnorm_and_affine_match_torch():
    rng = np.random.default_rng(7)
    for shape in ((6, 5, 4), (6,)):
        x = rng.normal(size=shape)
        g, b, m, v = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6), rng.uniform(0.1, 2, 6)
        for eps in (1e-5, 1e-3):
            want = F.batch_norm(t64(x)[None], t64(m), t64(v), t64(g), t64(b), False, 0.0, eps)[0].numpy()
            np.testing.assert_allclose(R.batchnorm(x, g, b, m, v, eps), want, atol=1e-12)
        scale, bias = rng.normal(size=6), rng.normal(size=6)
        want = x * (scale.reshape(-1, 1, 1) if x.ndim == 3 else scale) + (bias.reshape(-1, 1, 1) if x.ndim == 3 else bias)
        np.testing.assert_allclose(R.channel_affine(x, scale, bias), want)


def test_activations_match_torch():
    x = np.array([-800.0, -50.0, -3.0, -1e-3, 0.0, 1e-3, 2.0, 50.0, 800.0])
    np.testing.assert_allclose(R.sigmoid(x), torch.sigmoid(t64(x)).numpy(), atol=1e-15)
    assert np.all(np.isfinite(R.sigmoid(x)))
    np.testing.assert_allclose(R.leaky_relu(x), F.leaky_relu(t64(x)).numpy())
    np.testing.assert_allclose(R.leaky_relu(x, 0.2), F.leaky_relu(t64(x), 0.2).numpy())
    np.testing.assert_allclose(R.relu(x), F.relu(t64(x)).numpy())


def _model(nodes, tensors, features=1, classes=2, params=1, norm=None):
    header = {
        "format": "trcm", "formatVersion": 1, "name": "ops", "featureSchema": 1,
        "inputs": {"image": {"width": 64, "height": 64, "channels": 3, "layout": "CHW",
                             "norm": norm or {"type": "per_image"}},
                   "features": [{"name": f"f{i}", "mean": 0.0, "std": 1.0} for i in range(features)]},
        "classes": [{"id": f"c{i}", "name": f"C{i}", "icon": "other"} for i in range(classes)],
        "params": [{"name": f"p{i}", "mean": 0.0, "std": 1.0} for i in range(params)],
        "outputs": {"class_logits": "logits", "params": "params"},
        "nodes": nodes,
    }
    return R.ReferenceModel(encode_trcm(header, tensors))


def test_graph_with_every_op_matches_torch():
    rng = np.random.default_rng(11)
    w = {k: rng.normal(size=s).astype(np.float32) * 0.3 for k, s in {
        "c1.w": (6, 3, 3, 3), "c1.b": (6,), "c2.w": (6, 2, 3, 3),
        "bn.g": (6,), "bn.b": (6,), "bn.m": (6,), "ca.s": (6,), "ca.b": (6,),
        "d1.w": (5, 6 * 7 * 7), "d1.b": (5,), "bn1.g": (5,), "bn1.b": (5,), "bn1.m": (5,),
        "lg.w": (2, 5 + 6 + 2), "pr.w": (2, 5 + 6 + 2), "pr.b": (2,)}.items()}
    w["bn.v"] = rng.uniform(0.5, 2.0, 6).astype(np.float32)
    w["bn1.v"] = rng.uniform(0.5, 2.0, 5).astype(np.float32)
    nodes = [
        {"op": "conv2d", "name": "c1", "inputs": ["image"],
         "attrs": {"inChannels": 3, "outChannels": 6, "kernel": 3, "stride": 2, "padding": 1}, "weights": ["c1.w", "c1.b"]},
        {"op": "batchnorm", "name": "bn", "inputs": ["c1"], "weights": ["bn.g", "bn.b", "bn.m", "bn.v"]},
        {"op": "leaky_relu", "name": "lr", "inputs": ["bn"], "attrs": {"alpha": 0.1}},
        {"op": "conv2d", "name": "c2", "inputs": ["lr"],
         "attrs": {"inChannels": 6, "outChannels": 6, "kernel": 3, "padding": 1, "groups": 3}, "weights": ["c2.w"]},
        {"op": "channel_affine", "name": "ca", "inputs": ["c2"], "weights": ["ca.s", "ca.b"]},
        {"op": "add", "name": "res", "inputs": ["ca", "lr"]},
        {"op": "tanh", "name": "th", "inputs": ["res"]},
        {"op": "maxpool2d", "name": "mp", "inputs": ["th"], "attrs": {"kernel": 3, "stride": 2}},
        {"op": "avgpool2d", "name": "ap", "inputs": ["mp"], "attrs": {"kernel": 2}},
        {"op": "dropout", "name": "do", "inputs": ["ap"]},
        {"op": "dense", "name": "d1", "inputs": ["do"], "attrs": {"in": 294, "out": 5}, "weights": ["d1.w", "d1.b"]},
        {"op": "batchnorm", "name": "bn1", "inputs": ["d1"], "attrs": {"eps": 1e-3},
         "weights": ["bn1.g", "bn1.b", "bn1.m", "bn1.v"]},
        {"op": "sigmoid", "name": "sg", "inputs": ["bn1"]},
        {"op": "global_avgpool", "name": "gap", "inputs": ["th"]},
        {"op": "flatten", "name": "fl", "inputs": ["gap"]},
        {"op": "identity", "name": "id", "inputs": ["features"]},
        {"op": "concat", "name": "cat", "inputs": ["sg", "fl", "id"]},
        {"op": "relu", "name": "rl", "inputs": ["cat"]},
        {"op": "dense", "name": "logits", "inputs": ["rl"], "attrs": {"in": 13, "out": 2}, "weights": ["lg.w"]},
        {"op": "dense", "name": "params", "inputs": ["cat"], "attrs": {"in": 13, "out": 2}, "weights": ["pr.w", "pr.b"]},
    ]
    m = _model(nodes, w, features=2, classes=2, params=1)
    thumb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    feats = {"f0": 0.5, "f1": None}
    logits, params = m.predict(thumb, feats)

    T = {k: torch.from_numpy(v.astype(np.float64)) for k, v in w.items()}
    x = torch.from_numpy(m.prepare_image(thumb))[None]
    c1 = F.conv2d(x, T["c1.w"], T["c1.b"], 2, 1)
    lr = F.leaky_relu(F.batch_norm(c1, T["bn.m"], T["bn.v"], T["bn.g"], T["bn.b"], False, 0.0, 1e-5), 0.1)
    c2 = F.conv2d(lr, T["c2.w"], None, 1, 1, 1, 3)
    th = torch.tanh(c2 * T["ca.s"].view(1, -1, 1, 1) + T["ca.b"].view(1, -1, 1, 1) + lr)
    ap = F.avg_pool2d(F.max_pool2d(th, 3, 2), 2)
    d1 = F.linear(ap.flatten(1), T["d1.w"], T["d1.b"])
    sg = torch.sigmoid(F.batch_norm(d1, T["bn1.m"], T["bn1.v"], T["bn1.g"], T["bn1.b"], False, 0.0, 1e-3))
    cat = torch.cat([sg, th.mean(dim=(2, 3)), torch.tensor([[0.5, 0.0]], dtype=torch.float64)], dim=1)
    np.testing.assert_allclose(logits, F.linear(F.relu(cat), T["lg.w"])[0].numpy(), atol=1e-9)
    np.testing.assert_allclose(params, F.linear(cat, T["pr.w"], T["pr.b"])[0].numpy(), atol=1e-9)
    assert m.shapes["c1"] == (6, 32, 32) and m.shapes["mp"] == (6, 15, 15) and m.shapes["ap"] == (6, 7, 7)


def _tiny(op_node_extra=None, **kw):
    nodes = [{"op": "global_avgpool", "name": "g", "inputs": ["image"]},
             {"op": "dense", "name": "logits", "inputs": ["g"], "attrs": {"in": 3, "out": 2}, "weights": ["w"]},
             {"op": "dense", "name": "params", "inputs": ["g"], "attrs": {"in": 3, "out": 2}, "weights": ["w"]}]
    if op_node_extra:
        nodes = op_node_extra(nodes)
    return _model(nodes, {"w": np.ones((2, 3), np.float32)}, features=0, **kw)


@pytest.mark.parametrize("edit, message", [
    (lambda n: [{"op": "softmax", "name": "s", "inputs": ["image"]}] + n, "неизвестная операция 'softmax'"),
    (lambda n: [{"op": "relu", "name": "r", "inputs": ["nope"]}] + n, "не определён"),
    (lambda n: n[:1] + [dict(n[1], attrs={"in": 4, "out": 2})] + n[2:], "in=4"),
    (lambda n: n + [{"op": "relu", "name": "g", "inputs": ["image"]}], "уже занято"),
    (lambda n: [{"op": "conv2d", "name": "c", "inputs": ["image"],
                 "attrs": {"inChannels": 3, "outChannels": 2, "kernel": 3}, "weights": ["w"]}] + n, "форма"),
    (lambda n: [{"op": "add", "name": "a", "inputs": ["image"]}] + n, "входов"),
])
def test_graph_validation_errors(edit, message):
    with pytest.raises(TrcmError, match=message):
        _tiny(edit)


def test_output_shape_must_match_classes():
    with pytest.raises(TrcmError, match="class_logits"):
        _tiny(classes=3)


def test_per_image_normalization():
    rng = np.random.default_rng(5)
    thumb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    x = thumb.transpose(2, 0, 1).astype(np.float64) / 255
    got = R.normalize_image(R.thumb_to_chw(thumb), {"type": "per_image"})
    np.testing.assert_allclose(got, (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean()))
    assert np.all(R.normalize_image(R.thumb_to_chw(np.zeros((64, 64, 3), np.uint8)), {"type": "per_image"}) == 0)
    flat = np.full((64, 64, 3), 100, np.uint8)
    flat[0, 0, 0] = 101  # почти константа: работает нижняя граница СКО 1/sqrt(12288)
    got = R.normalize_image(R.thumb_to_chw(flat), {"type": "per_image"})
    x = R.thumb_to_chw(flat)
    np.testing.assert_allclose(got, (x - x.mean()) * math.sqrt(12288))
    np.testing.assert_array_equal(R.thumb_to_chw(list(thumb.reshape(-1))), R.thumb_to_chw(thumb))


def test_channel_normalization():
    thumb = np.random.default_rng(6).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    norm = {"type": "channel", "mean": [0.5, 0.4, 0.3], "std": [0.2, 0.25, 0.3]}
    x = R.thumb_to_chw(thumb)
    want = (x - np.array(norm["mean"]).reshape(3, 1, 1)) / np.array(norm["std"]).reshape(3, 1, 1)
    np.testing.assert_allclose(R.normalize_image(x, norm), want)


def test_feature_normalization_rules():
    specs = [{"name": "a", "mean": 1.0, "std": 2.0}, {"name": "b", "mean": 0.0, "std": 0.0},
             {"name": "c", "mean": 0.0, "std": 1e-7}, {"name": "d", "mean": 0.0, "std": 1.0},
             {"name": "e", "mean": 5.0, "std": 0.5}, {"name": "f", "mean": 0.0, "std": 1.0},
             {"name": "g", "mean": 0.0, "std": 1.0}, {"name": "h", "mean": 0.0, "std": 1.0}]
    raw = {"a": 4.0, "b": 3.0, "c": -2.0, "d": float("nan"), "e": 1000.0, "f": True, "g": "x", "h": None}
    np.testing.assert_allclose(R.normalize_features(raw, specs), [1.5, 3.0, -2.0, 0.0, 6.0, 1.0, 0.0, 0.0])
    assert np.all(R.normalize_features({"e": -1e9, "a": float("inf")}, specs)[[0, 4]] == [0.0, -6.0])
    assert np.all(R.normalize_features(None, specs) == 0)
