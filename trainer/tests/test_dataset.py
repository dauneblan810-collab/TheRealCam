import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import edit_meta
from therealcam_trainer.dataset import (
    DatasetError,
    Sample,
    build_training_data,
    class_weights,
    dataset_report,
    derive_targets,
    format_report,
    load_dataset,
    split_train_val,
)
from therealcam_trainer.schema import FEATURE_NAMES_V1, ICON_PROFILES, PARAM_NAMES_V1, icon_profile


def _zip(src: Path, dest: Path, prefix: str) -> Path:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, prefix + p.relative_to(src).as_posix())
    return dest


def test_icon_profiles():
    assert len(ICON_PROFILES) == 40
    assert icon_profile("stars") == "astro" and icon_profile("moon") == "lunar" and icon_profile("clouds") == "sky"
    assert icon_profile("other") == "default" and icon_profile("неизвестный") == "default" and icon_profile(None) == "default"
    assert len(FEATURE_NAMES_V1) == 33 and len(set(FEATURE_NAMES_V1)) == 33


def test_load_folder_variants(synth_dir, synth_ds):
    ids = [s.sample_id for s in synth_ds.samples]
    assert len(ids) == 200 and synth_ds.kind == "folder"
    assert [s.sample_id for s in load_dataset(synth_dir / "dataset").samples] == ids
    assert [s.sample_id for s in load_dataset(synth_dir / "dataset" / "samples").samples] == ids
    # Папка выше: <копия телефона>/Android/data/TheRealCam/dataset
    holder = synth_dir.parent / (synth_dir.name + "_holder")
    if not holder.exists():
        (holder / "Android" / "data").mkdir(parents=True)
        try:
            (holder / "Android" / "data" / "TheRealCam").symlink_to(synth_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(synth_dir, holder / "Android" / "data" / "TheRealCam")
    assert [s.sample_id for s in load_dataset(holder).samples] == ids
    s = synth_ds.samples[0]
    assert s.thumb.shape == (64, 64, 3) and s.thumb.dtype == np.uint8
    assert set(s.features) == set(FEATURE_NAMES_V1)
    assert [c.icon for c in synth_ds.classes()] == ["stars", "moon", "clouds", "other"]


@pytest.mark.parametrize("prefix", ["", "dataset/", "TheRealCam/dataset/"])
def test_load_zip(synth_dir, synth_ds, tmp_path, prefix):
    z = _zip(synth_dir / "dataset", tmp_path / "export.zip", prefix)
    ds = load_dataset(z)
    assert ds.kind == "zip"
    assert [s.sample_id for s in ds.samples] == [s.sample_id for s in synth_ds.samples]
    np.testing.assert_array_equal(ds.samples[5].thumb, synth_ds.samples[5].thumb)
    assert [c.id for c in ds.classes()] == [c.id for c in synth_ds.classes()]


def test_not_a_dataset(tmp_path):
    with pytest.raises(DatasetError, match="не найдена"):
        load_dataset(tmp_path)
    with pytest.raises(DatasetError, match="не найден"):
        load_dataset(tmp_path / "missing")
    bad = tmp_path / "x.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(DatasetError, match="meta.json"):
        load_dataset(bad)


def test_complete_filter_and_problems(small_synth):
    dirs = sorted((small_synth / "samples").iterdir())
    edit_meta(dirs[0], lambda m: m["labels"].update(complete=False))
    edit_meta(dirs[1], lambda m: m["labels"].pop("complete"))
    edit_meta(dirs[2], lambda m: m["labels"].update(rating=None))  # complete=true, но поле пустое
    (dirs[3] / "thumb64.png").unlink()
    Image.new("RGB", (32, 32)).save(dirs[4] / "thumb64.png")
    Image.new("RGBA", (64, 64), (10, 20, 30, 128)).save(dirs[5] / "thumb64.png")  # другой режим -> RGB
    (dirs[6] / "meta.json").write_text("{broken", encoding="utf-8")
    ds = load_dataset(small_synth)
    ids = {s.sample_id for s in ds.samples}
    assert len(ds.samples) == 30 - 6
    assert dirs[5].name in ids and not ids & {d.name for d in dirs[:5]}
    rgba = next(s for s in ds.samples if s.sample_id == dirs[5].name)
    assert rgba.thumb.shape == (64, 64, 3) and tuple(rgba.thumb[0, 0]) == (10, 20, 30)
    c = ds.counts
    assert (c["incomplete"], c["bad_labels"], c["no_thumb"], c["bad_thumb"], c["bad_meta"]) == (2, 1, 1, 1, 1)
    text = " ".join(ds.problems)
    assert "32×32" in text and "нет thumb64.png" in text and "rating" in text
    report = format_report(dataset_report(ds))
    assert "НЕДОСТАТОЧНО" in report and "неполная оценка" in report


def test_categories_from_labels_when_json_missing(small_synth):
    cats = json.loads((small_synth / "categories.json").read_text(encoding="utf-8"))["categories"]
    (small_synth / "categories.json").unlink()
    ds = load_dataset(small_synth)
    by_id = {c["id"]: c for c in cats}
    for c in ds.classes():
        assert c.name == by_id[c.id]["name"] and c.icon == by_id[c.id]["icon"]


def test_features_null_and_selection(small_synth):
    dirs = sorted((small_synth / "samples").iterdir())
    for d in dirs[:28]:  # zoom заполнен только в dirs[28]: 1 из ≥20 обучающих (<10%)
        edit_meta(d, lambda m: m["features"].update(zoom=None))
    edit_meta(dirs[29], lambda m: m.update(features=None))
    ds = load_dataset(small_synth)
    assert all(v is None for v in next(s for s in ds.samples if s.sample_id == dirs[29].name).features.values())
    td = build_training_data(ds, seed=0)
    assert "zoom" not in td.feature_names and "scene_ev" in td.feature_names
    assert td.feats.shape == (30, len(td.feature_names)) and np.abs(td.feats).max() <= 6
    rep = dataset_report(ds)
    assert rep["missing_features"]["zoom"] > 0.9


def _sample(capture=None, user=None, preview=None, series=None, labels=None, iso_range=(100, 3200), rating=8):
    lab = {"rating": rating, "exposure": 0, "sharpness": 0, "noise": 0, "color": 0, "motion_blur": 0,
           "category_id": "c", "complete": True}
    lab.update(labels or {})
    return Sample(sample_id="20260101_000000_000_0000", thumb=np.zeros((64, 64, 3), np.uint8), features={},
                  labels=lab, category_id="c", rating=rating, mode="manual", series_id=None, series_role=None,
                  meta={"capture": capture or {}, "user_settings": user or {}, "preview": preview or {},
                        "series": series, "device": {"iso_range": list(iso_range) if iso_range else None}})


def test_derive_targets_base_and_fallbacks():
    v, m = derive_targets(_sample({"exposure_time_ns": 250_000_000, "iso": 800, "focus_distance_diopters": 0.05,
                                   "ev_compensation": 0.5}, series={"count": 12}))
    np.testing.assert_allclose(v, [-2.0, math.log2(800), math.log2(12), 1.0, 0.5])
    assert m.all()
    v, m = derive_targets(_sample({"exposure_time_ns": None, "iso": None, "ev_compensation": None},
                                  {"exposure_time_ns": 1_000_000_000, "frames": 4, "focus": "infinity", "ev": -1.0},
                                  {"iso": 3200}))
    np.testing.assert_allclose(v, [0.0, math.log2(3200), 2.0, 1.0, -1.0])
    v, m = derive_targets(_sample({}, {}, {"exposure_time_ns": 10_000_000, "focus_distance_diopters": 0.0}))
    assert m.tolist() == [True, False, True, True, True]
    assert v[2] == 0.0 and v[3] == 0.0 and v[4] == 0.0 and np.isnan(v[1])
    v, m = derive_targets(_sample({"focus_distance_diopters": 0.2}, {"focus": "auto"}))
    assert not m.any() and np.isnan(v).all()  # нет ни выдержки, ни ISO -> без потерь параметров


def _corr(labels, capture=None, iso_range=(100, 3200)):
    cap = {"exposure_time_ns": 125_000_000, "iso": 400}  # exp=-3, iso=log2(400)
    cap.update(capture or {})
    v, _ = derive_targets(_sample(cap, labels=labels, iso_range=iso_range), delta_ev=0.7)
    return v - np.array([-3.0, math.log2(cap["iso"]), 0.0, 0.0, 0.0])


def test_derive_targets_corrections():
    def check(got, want):
        np.testing.assert_allclose(got, want, atol=1e-9)

    check(_corr({"exposure": -1}), [0.35, 0.35, 0, 0, 0.7])
    check(_corr({"exposure": -1, "motion_blur": 1}), [-1.0, 1.7, 0, 0, 0.7])
    check(_corr({"exposure": 1}), [-0.35, -0.35, 0, 0, -0.7])
    # ISO упирается в минимум сенсора: дефицит уходит в выдержку
    check(_corr({"exposure": 1}, {"iso": 110}), [-0.35 - (0.35 - math.log2(110 / 100)), -math.log2(110 / 100), 0, 0, -0.7])
    check(_corr({"exposure": 1}, {"iso": 110}, iso_range=None), [-0.35, -0.35, 0, 0, -0.7])
    check(_corr({"motion_blur": 1}), [-1.0, 1.0, 0, 0, 0])
    check(_corr({"noise": -1}), [0.5, -0.5, 0, 0, 0])
    check(_corr({"noise": -1, "motion_blur": 1}), [-1.0, 1.0, 1.0, 0, 0])


def test_split_groups_and_classes(synth_ds):
    groups = [s.group_id for s in synth_ds.samples]
    classes = {c.id: i for i, c in enumerate(synth_ds.classes())}
    y = np.array([classes[s.category_id] for s in synth_ds.samples])
    tr, va = split_train_val(groups, y, len(classes), 0.15, seed=0)
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(len(y)))
    assert 0.08 <= len(va) / len(y) <= 0.25
    assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in va})
    assert set(y[tr]) == set(range(len(classes)))
    tr2, va2 = split_train_val(groups, y, len(classes), 0.15, seed=0)
    assert tr.tolist() == tr2.tolist() and va.tolist() == va2.tolist()
    # серии действительно есть и не разорваны
    assert any(groups.count(g) > 1 for g in set(groups))


def test_split_keeps_rare_class_in_train():
    y = np.array([0] * 20 + [1])
    tr, va = split_train_val([f"g{i}" for i in range(21)], y, 2, 0.5, seed=3)
    assert 20 in tr


def test_training_data_weights_and_norms(synth_ds):
    td = build_training_data(synth_ds, seed=0)
    k = len(td.classes)
    assert td.class_w.shape == (k,) and abs(float(td.class_w.mean()) - 1.0) < 1e-5
    counts = np.bincount(td.y[td.train_idx], minlength=k)
    assert np.argmax(td.class_w) == np.argmin(counts)
    ratings = np.array([s.rating for s in synth_ds.samples])
    frames = np.array([s.series_role == "frame" for s in synth_ds.samples])
    want = (0.05 + ((ratings - 1) / 9) ** 2) * np.where(frames, 0.5, 1.0)
    np.testing.assert_allclose(td.param_w, want, rtol=1e-6)
    fi = PARAM_NAMES_V1.index("focus_infinity")
    assert td.param_mean[fi] == 0.0 and td.param_std[fi] == 1.0
    assert np.all(td.param_std >= 0.25)
    assert np.all(td.feature_std > 0)
    # нормализация признаков — по обучающей части
    j = td.feature_names.index("scene_ev")
    vals = np.array([synth_ds.samples[i].features["scene_ev"] for i in td.train_idx])
    assert abs(td.feature_mean[j] - vals.mean()) < 1e-4
    sizes = {}
    for s in synth_ds.samples:
        sizes[s.group_id] = sizes.get(s.group_id, 0) + 1
    np.testing.assert_allclose(td.group_w, [1 / math.sqrt(sizes[s.group_id]) for s in synth_ds.samples], rtol=1e-6)


def test_class_weights_formula():
    w = class_weights(np.array([0] * 16 + [1] * 4 + [2]), 4)
    inv = np.array([1 / 4, 1 / 2, 1.0])
    np.testing.assert_allclose(w[:3], inv / inv.mean(), rtol=1e-6)
    assert w[3] == 1.0
