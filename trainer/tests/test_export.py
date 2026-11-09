import re

import numpy as np
import pytest
import torch

from therealcam_trainer.errors import TrcmError
from therealcam_trainer.export import (
    export_trcnet,
    extend_classes,
    load_trcnet,
    refit_feature_norm,
    refit_param_norm,
)
from therealcam_trainer.model import ModelSpec, TrcNet, f32, per_image_normalize
from therealcam_trainer.reference import ReferenceModel
from therealcam_trainer.schema import FEATURE_NAMES_V1, PARAM_NAMES_V1
from therealcam_trainer.trcm import decode_trcm, encode_trcm

ATOL = 1e-4


def _randomize_bn(model: TrcNet, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, bn in model.conv_bn_pairs():
            n = bn.num_features
            bn.running_mean.copy_(torch.randn(n, generator=g) * 0.3)
            bn.running_var.copy_(torch.rand(n, generator=g) * 1.7 + 0.3)
            bn.weight.copy_(torch.rand(n, generator=g) + 0.5)
            bn.bias.copy_(torch.randn(n, generator=g) * 0.2)


def _spec(num_features: int, num_classes: int, seed: int = 0) -> ModelSpec:
    rng = np.random.default_rng(seed)
    names = list(FEATURE_NAMES_V1[:num_features]) if num_features <= 33 else [f"x{i}" for i in range(num_features)]
    pm = rng.normal(size=5)
    ps = rng.uniform(0.5, 3, 5)
    pm[3], ps[3] = 0.0, 1.0
    return ModelSpec(classes=[{"id": f"cat_{i:04x}", "name": f"Класс {i}", "icon": "stars"} for i in range(num_classes)],
                     feature_names=names, feature_mean=rng.normal(size=num_features).tolist(),
                     feature_std=rng.uniform(0.2, 3.0, num_features).tolist(), param_names=list(PARAM_NAMES_V1),
                     param_mean=pm.tolist(), param_std=ps.tolist())


def _inputs(spec: ModelSpec, seed: int, n: int = 4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        thumb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) if i else np.zeros((64, 64, 3), np.uint8)
        raw = {name: float(rng.normal(m, 2 * s)) for name, m, s in
               zip(spec.feature_names, spec.feature_mean, spec.feature_std)}
        for name in spec.feature_names[::3]:
            raw[name] = None
        out.append((thumb, raw))
    return out


def _torch_out(model: TrcNet, ref: ReferenceModel, thumb, raw):
    x = per_image_normalize(torch.from_numpy(thumb.copy()).permute(2, 0, 1)[None].float() / 255)
    f = torch.from_numpy(ref.prepare_features(raw)).float()[None]
    with torch.no_grad():
        logits, params = model.eval()(x, f)
    return logits[0].double().numpy(), params[0].double().numpy()


@pytest.mark.parametrize("features, classes, widths, hidden", [
    (33, 4, (16, 32, 48, 64), 96),
    (0, 1, (16, 32, 48, 64), 96),
    (3, 2, (5, 7, 9, 11), 13),
    (7, 5, (1, 3, 2, 4), 1),
])
def test_reference_matches_torch(features, classes, widths, hidden):
    torch.manual_seed(features + classes)
    model = TrcNet(features, classes, widths=widths, hidden=hidden)
    _randomize_bn(model, features)
    spec = _spec(features, classes)
    blob = export_trcnet(model, spec, name="t", created_utc="2026-01-01T00:00:00Z")
    ref = ReferenceModel(blob)
    assert ref.num_classes == classes and ref.num_params == 5 and len(ref.feature_specs) == features
    ops = [n.op for n in ref.nodes]
    assert "batchnorm" not in ops and "dropout" not in ops
    assert ("concat" in ops) == (features > 0)
    for thumb, raw in _inputs(spec, features):
        logits, params = ref.predict(thumb, raw)
        tl, tp = _torch_out(model, ref, thumb, raw)
        np.testing.assert_allclose(logits, tl, atol=ATOL)
        np.testing.assert_allclose(params, tp, atol=ATOL)


def test_header_contents():
    model = TrcNet(33, 3)
    spec = _spec(33, 3)
    h = decode_trcm(export_trcnet(model, spec, name="Моя модель", training={"samples": 5, "epochs": 2})).header
    assert h["format"] == "trcm" and h["formatVersion"] == 1 and h["featureSchema"] == 1
    assert h["trainer"] == {"name": "therealcam-trainer", "version": "0.1.0"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", h["createdUtc"])
    assert h["training"]["samples"] == 5 and h["training"]["baseModel"] is None and "metrics" in h["training"]
    assert h["inputs"]["image"] == {"width": 64, "height": 64, "channels": 3, "layout": "CHW",
                                    "norm": {"type": "per_image"}}
    assert [f["name"] for f in h["inputs"]["features"]] == FEATURE_NAMES_V1
    assert all(f32(f["mean"]) == f["mean"] and f32(f["std"]) == f["std"] for f in h["inputs"]["features"])
    assert h["classes"][0] == {"id": "cat_0000", "name": "Класс 0", "icon": "stars"}
    assert [p["name"] for p in h["params"]] == PARAM_NAMES_V1
    assert h["params"][3] == {"name": "focus_infinity", "mean": 0.0, "std": 1.0}
    assert h["outputs"] == {"class_logits": "logits", "params": "params"}


def test_export_import_export_identical():
    torch.manual_seed(3)
    model = TrcNet(33, 4)
    _randomize_bn(model, 3)
    spec = _spec(33, 4)
    blob = export_trcnet(model, spec, name="x", created_utc="2026-01-01T00:00:00Z", training={"dropout": 0.2})
    imported, spec2, header = load_trcnet(blob)
    assert imported.bn_frozen and spec2.class_ids == spec.class_ids and spec2.feature_mean == spec.feature_mean
    blob2 = export_trcnet(imported, spec2, name="x", created_utc="2026-01-01T00:00:00Z", training={"dropout": 0.2})
    assert blob2 == blob
    ref = ReferenceModel(blob)
    for thumb, raw in _inputs(spec, 9):
        a = _torch_out(model, ref, thumb, raw)
        b = _torch_out(imported, ref, thumb, raw)
        np.testing.assert_allclose(a[0], b[0], atol=1e-5)
        np.testing.assert_allclose(a[1], b[1], atol=1e-5)
    # в режиме train замороженный BN не меняет функцию (dropout выключен для сравнения)
    imported.drop.p = 0.0
    imported.train()
    x = torch.randn(2, 3, 64, 64)
    f = torch.randn(2, 33)
    with torch.no_grad():
        tr = imported(x, f)
        ev = imported.eval()(x, f)
    torch.testing.assert_close(tr[0], ev[0])


def test_extend_classes_keeps_old_outputs():
    torch.manual_seed(4)
    model = TrcNet(33, 3)
    _randomize_bn(model, 4)
    imported, spec, _ = load_trcnet(export_trcnet(model, _spec(33, 3), name="b"))
    extended = extend_classes(imported, 5, seed=1)
    assert extended.num_classes == 5 and extended.bn_frozen
    # веса старых строк скопированы бит в бит
    assert torch.equal(extended.cls_head.weight[:3], imported.cls_head.weight)
    assert torch.equal(extended.param_head.weight[:15], imported.param_head.weight)
    assert torch.equal(extended.param_head.bias[:15], imported.param_head.bias)
    x = per_image_normalize(torch.rand(3, 3, 64, 64))
    f = torch.randn(3, 33)
    with torch.no_grad():
        old_l, old_p = imported.eval()(x, f)
        new_l, new_p = extended.eval()(x, f)
    # GEMM другой ширины может суммировать в другом порядке: только ошибки округления float32
    torch.testing.assert_close(new_l[:, :3], old_l, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_p[:, :15], old_p, atol=1e-6, rtol=0)
    assert not torch.equal(new_l[:, 3:], torch.zeros_like(new_l[:, 3:]))
    # и в эталонном интерпретаторе (как на телефоне)
    spec5 = _spec(33, 5)
    spec5.feature_mean, spec5.feature_std = spec.feature_mean, spec.feature_std
    ref_old = ReferenceModel(export_trcnet(imported, spec, name="b"))
    ref_new = ReferenceModel(export_trcnet(extended, spec5, name="b5"))
    for thumb, raw in _inputs(spec, 2):
        lo, po = ref_old.predict(thumb, raw)
        ln, pn = ref_new.predict(thumb, raw)
        np.testing.assert_allclose(ln[:3], lo, atol=1e-12, rtol=0)
        np.testing.assert_allclose(pn[:15], po, atol=1e-12, rtol=0)


def test_refit_norm_preserves_outputs():
    torch.manual_seed(5)
    model = TrcNet(4, 2).eval()
    old_fm, old_fs = np.array([1.0, -2.0, 0.0, 3.0]), np.array([2.0, 0.5, 1.0, 1e-9])
    new_fm, new_fs = np.array([1.5, -1.0, 0.2, 3.0]), np.array([1.0, 0.7, 2.0, 4.0])
    raw = np.array([[2.0, -1.5, 0.3, 3.5]])
    z_old = (raw - old_fm) / np.where(old_fs > 1e-6, old_fs, 1.0)
    z_new = (raw - new_fm) / new_fs
    x = per_image_normalize(torch.rand(1, 3, 64, 64))
    pm0, ps0 = np.array([-3.0, 8.0, 1.0, 0.0, 0.1]), np.array([2.0, 1.5, 1.0, 1.0, 0.5])
    pm1, ps1 = np.array([-2.0, 7.0, 2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.5, 1.0, 0.25])
    with torch.no_grad():
        l0, p0 = model(x, torch.tensor(z_old, dtype=torch.float32))
        refit_feature_norm(model, old_fm, old_fs, new_fm, new_fs)
        refit_param_norm(model, PARAM_NAMES_V1, pm0, ps0, pm1, ps1)
        l1, p1 = model(x, torch.tensor(z_new, dtype=torch.float32))
    np.testing.assert_allclose(l1.numpy(), l0.numpy(), atol=1e-4)
    real0 = p0.numpy().reshape(2, 5) * ps0 + pm0
    real1 = p1.numpy().reshape(2, 5) * ps1 + pm1
    np.testing.assert_allclose(real1, real0, atol=1e-4)


def test_import_rejects_foreign_graph():
    header = {
        "format": "trcm", "formatVersion": 1, "name": "foreign", "featureSchema": 1,
        "inputs": {"image": {"width": 64, "height": 64, "channels": 3, "norm": {"type": "per_image"}}, "features": []},
        "classes": [{"id": "a", "name": "A", "icon": "stars"}],
        "params": [{"name": n, "mean": 0.0, "std": 1.0} for n in PARAM_NAMES_V1],
        "outputs": {"class_logits": "logits", "params": "params"},
        "nodes": [{"op": "global_avgpool", "name": "g", "inputs": ["image"]},
                  {"op": "dense", "name": "logits", "inputs": ["g"], "attrs": {"in": 3, "out": 1}, "weights": ["w1"]},
                  {"op": "dense", "name": "params", "inputs": ["g"], "attrs": {"in": 3, "out": 5}, "weights": ["w5"]}],
    }
    blob = encode_trcm(header, {"w1": np.ones((1, 3), np.float32), "w5": np.ones((5, 3), np.float32)})
    ReferenceModel(blob)  # корректная модель .trcm…
    with pytest.raises(TrcmError, match="TrcNet"):  # …но не TrcNet этого тренера
        load_trcnet(blob)
