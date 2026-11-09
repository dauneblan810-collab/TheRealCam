import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import torch

from conftest import REPO_ROOT
from therealcam_trainer import cli
from therealcam_trainer.export import load_trcnet
from therealcam_trainer.golden import make_golden
from therealcam_trainer.model import per_image_normalize
from therealcam_trainer.reference import ReferenceModel
from therealcam_trainer.schema import FEATURE_NAMES_V1
from therealcam_trainer.train import TrainConfig, train_model


def test_help_without_torch_or_display():
    cwd = REPO_ROOT / "trainer"
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    r = subprocess.run([sys.executable, "-m", "therealcam_trainer", "--help"], capture_output=True, text=True,
                       encoding="utf-8", cwd=cwd, env=env)
    assert r.returncode == 0 and "inspect-dataset" in r.stdout and "make-golden" in r.stdout
    r = subprocess.run([sys.executable, "-c", "import sys; import therealcam_trainer.cli as c; c.build_parser(); "
                        "print('torch' in sys.modules, 'tkinter' in sys.modules)"], capture_output=True, text=True,
                       cwd=cwd, env=env)
    assert r.stdout.split() == ["False", "False"], r.stderr


def test_cli_train_inspect_eval(synth_dir, tmp_path, capsys):
    out = tmp_path / "m.trcm"
    assert cli.main(["inspect-dataset", str(synth_dir)]) == 0
    assert "Пригодно для обучения: 200" in capsys.readouterr().out
    assert cli.main(["train", str(synth_dir), "-o", str(out), "--epochs", "2", "--allow-small", "--name", "e2e"]) == 0
    text = capsys.readouterr().out
    assert "эпоха   2/2" in text and "Модель сохранена" in text
    ref = ReferenceModel(out)
    assert ref.header["name"] == "e2e" and ref.num_classes == 4 and ref.header["training"]["samples"] == 200
    assert cli.main(["inspect-model", str(out)]) == 0
    assert "CRC-32 данных: OK" in capsys.readouterr().out
    assert cli.main(["eval", str(synth_dir), "--model", str(out), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["evaluated"] == 200 and 0.0 <= rep["accuracy"] <= 1.0 and set(rep["mae"]) == {
        "log2_exposure_s", "log2_iso", "log2_frames", "ev_bias"}


def test_refuses_small_dataset(small_synth, tmp_path, capsys):
    assert cli.main(["train", str(small_synth), "-o", str(tmp_path / "x.trcm"), "--epochs", "1"]) == 1
    assert "минимум 150" in capsys.readouterr().err
    assert not (tmp_path / "x.trcm").exists()


def test_finetune_adds_new_class(synth_ds, tmp_path):
    classes = synth_ds.classes()
    new_id = classes[-1].id
    base_ds = dataclasses.replace(synth_ds, samples=[s for s in synth_ds.samples if s.category_id != new_id])
    base = train_model(base_ds, TrainConfig(epochs=1, allow_small=True, seed=1))
    base_path = tmp_path / "base.trcm"
    base.export(base_path, name="база")
    assert [c["id"] for c in base.spec.classes] == [c.id for c in classes[:-1]]
    res = train_model(synth_ds, TrainConfig(epochs=1, allow_small=True, seed=1, base_model=str(base_path)))
    assert [c["id"] for c in res.spec.classes] == [c.id for c in classes]
    assert res.spec.feature_mean == base.spec.feature_mean  # нормализация базы сохранена
    header = ReferenceModel(res.export(None, name="дообученная")).header
    assert header["training"]["baseModel"] == "база" and len(header["training"]["baseModelSha256"]) == 64


def test_cancel_returns_best_so_far(synth_ds):
    import threading

    cancel = threading.Event()

    def progress(ev):
        if ev["type"] == "epoch":
            cancel.set()

    res = train_model(synth_ds, TrainConfig(epochs=50, allow_small=True), progress=progress, cancel=cancel)
    assert res.cancelled and res.epochs_run == 1 and res.best_epoch == 1


def test_golden_generation_is_deterministic(tmp_path):
    a = make_golden(tmp_path / "a")
    make_golden(tmp_path / "b")
    for name in ("model.trcm", "golden.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert a["max_abs_diff_torch"] < 1e-4 and a["classes"] >= 3 and a["features"] == 33


def test_committed_golden_matches_reference_and_torch():
    gdir = REPO_ROOT / "testdata" / "golden"
    golden = json.loads((gdir / "golden.json").read_text(encoding="utf-8"))
    assert set(golden) == {"model", "cases", "atol", "rtol"}
    assert golden["model"] == "model.trcm" and golden["atol"] == 1e-4 and golden["rtol"] == 1e-4
    ref = ReferenceModel(gdir / golden["model"])
    torch_model, _, _ = load_trcnet(gdir / golden["model"])
    assert ref.num_classes >= 3 and ref.feature_names == FEATURE_NAMES_V1
    assert all(not (s["mean"] == 0.0 and s["std"] == 1.0) for s in ref.feature_specs)
    assert len(golden["cases"]) >= 4
    nulls = 0
    for case in golden["cases"]:
        assert set(case) == {"thumb_rgb_hwc", "features", "expected_class_logits", "expected_params"}
        thumb = np.array(case["thumb_rgb_hwc"])
        assert thumb.shape == (12288,) and thumb.min() >= 0 and thumb.max() <= 255
        nulls += sum(v is None for v in case["features"].values())
        logits, params = ref.predict(thumb.astype(np.uint8), case["features"])
        np.testing.assert_allclose(logits, case["expected_class_logits"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(params, case["expected_params"], atol=1e-4, rtol=1e-4)
        x = per_image_normalize(torch.from_numpy(thumb.reshape(64, 64, 3)).permute(2, 0, 1)[None].float() / 255)
        f = torch.from_numpy(ref.prepare_features(case["features"])).float()[None]
        with torch.no_grad():
            tl, tp = torch_model(x, f)
        np.testing.assert_allclose(tl[0].numpy(), case["expected_class_logits"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(tp[0].numpy(), case["expected_params"], atol=1e-4, rtol=1e-4)
    assert nulls > 0
    assert any(np.all(np.array(c["thumb_rgb_hwc"]) == 0) for c in golden["cases"])
