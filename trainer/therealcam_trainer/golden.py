"""Эталонные векторы `testdata/golden/` для проверки Kotlin-инференса (MODEL_FORMAT.md §6).

Модель — архитектура TrcNet по умолчанию, 2 эпохи на СИНТЕТИЧЕСКОМ датасете (только чтобы
статистики BN и веса были нетривиальными). Ожидаемые выходы считает эталонный NumPy-интерпретатор
и перепроверяет PyTorch.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from .schema import FEATURE_NAMES_V1

GOLDEN_SEED = 20261109
GOLDEN_SAMPLES = 160
GOLDEN_NAME = "golden-test-model"
GOLDEN_CREATED_UTC = "2026-11-09T00:00:00Z"
GOLDEN_DESCRIPTION = "Эталонная модель для TrcmGoldenTest (обучена 2 эпохи на синтетических данных)"
ATOL = 1e-4
RTOL = 1e-4


def golden_cases(seed: int = GOLDEN_SEED) -> list[tuple[str, np.ndarray, dict]]:
    """(название, миниатюра uint8 [64,64,3], сырые признаки) — разные сцены и крайние случаи."""
    from .synth import _scene, preview_to_thumb, render_scene, scene_features

    rng = np.random.default_rng(seed)
    day = datetime(2026, 8, 12)
    cases = []

    thumb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    feats = {name: round(float(rng.normal(0, 2)), 6) for name in FEATURE_NAMES_V1}
    for name in ("lux_log10", "sun_alt_deg", "moon_alt_deg"):
        feats[name] = None
    cases.append(("random", thumb, feats))

    sc = _scene("stars", rng, day)
    preview = render_scene("stars", rng, portrait=True)
    feats = scene_features(sc, preview)
    feats["sun_alt_deg"] = feats["moon_alt_deg"] = None
    cases.append(("stars", preview_to_thumb(preview), feats))

    sc = _scene("clouds", rng, day)
    preview = np.clip(render_scene("clouds", rng, portrait=False) * 1.6, 0, 1)
    feats = scene_features(sc, preview)
    cases.append(("bright", preview_to_thumb(preview), feats))

    cases.append(("zeros", np.zeros((64, 64, 3), dtype=np.uint8), {name: None for name in FEATURE_NAMES_V1}))

    sc = _scene("moon", rng, day)
    preview = render_scene("moon", rng, portrait=True)
    feats = scene_features(sc, preview)
    feats.update({"scene_ev": 40.0, "luma_mean": 5.0, "point_sources": 100000.0, "pitch_deg": -90.0,
                  "zoom": 100.0, "front": 1.0, "lux_log10": -50.0})
    cases.append(("moon_extreme_features", preview_to_thumb(preview), feats))

    sc = _scene("other", rng, day)
    preview = render_scene("other", rng, portrait=True)
    feats = scene_features(sc, preview)
    feats["lux_log10"] = None
    cases.append(("grey_texture", preview_to_thumb(preview), feats))
    return cases


def _dump(golden: dict) -> str:
    lines = ["{", f'  "model": {json.dumps(golden["model"])},', '  "cases": [']
    cases = golden["cases"]
    for i, case in enumerate(cases):
        sep = "," if i < len(cases) - 1 else ""
        lines.append("    " + json.dumps(case, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + sep)
    lines += ["  ],", f'  "atol": {json.dumps(golden["atol"])},', f'  "rtol": {json.dumps(golden["rtol"])}', "}"]
    return "\n".join(lines) + "\n"


def make_golden(out_dir: str | os.PathLike, seed: int = GOLDEN_SEED, progress=None) -> dict:
    """Пишет model.trcm и golden.json в out_dir; возвращает сводку."""
    import torch

    from .dataset import load_dataset
    from .export import load_trcnet
    from .model import per_image_normalize
    from .reference import ReferenceModel
    from .synth import generate_synthetic_dataset
    from .train import TrainConfig, quiet_torch, train_model

    quiet_torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trc-golden-") as tmp:
        generate_synthetic_dataset(tmp, n=GOLDEN_SAMPLES, seed=seed, write_images=False)
        ds = load_dataset(tmp)
        cfg = TrainConfig(epochs=2, batch_size=32, seed=seed, allow_small=True, device="cpu",
                          feature_names=list(FEATURE_NAMES_V1), patience=1000, num_threads=1)
        res = train_model(ds, cfg, progress=progress)
    trivial = [n for n, m, s in zip(res.spec.feature_names, res.spec.feature_mean, res.spec.feature_std)
               if m == 0.0 and s == 1.0]
    if trivial:
        raise RuntimeError(f"у признаков {trivial} тривиальная нормализация — смените seed")
    if len(res.spec.classes) < 3:
        raise RuntimeError("в эталонной модели меньше 3 классов")
    blob = res.export(None, name=GOLDEN_NAME, description=GOLDEN_DESCRIPTION, author="therealcam-trainer",
                      created_utc=GOLDEN_CREATED_UTC)
    ref = ReferenceModel(blob)
    imported, _, _ = load_trcnet(blob)

    cases = []
    max_diff = 0.0
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _, thumb, feats in golden_cases(seed):
            logits, params = ref.predict(thumb, feats)
            x = per_image_normalize(torch.from_numpy(thumb.copy()).permute(2, 0, 1)[None].float() / 255.0)
            f = torch.from_numpy(ref.prepare_features(feats)).float()[None]
            with torch.no_grad():
                for net in (res.model.eval(), imported.eval()):
                    tl, tp = net(x, f)
                    for got, want in ((tl[0].double().numpy(), logits), (tp[0].double().numpy(), params)):
                        diff = np.abs(got - want)
                        if np.any(diff > ATOL + RTOL * np.abs(want)):
                            raise RuntimeError(f"PyTorch и эталонный интерпретатор расходятся: {diff.max():.3g}")
                        max_diff = max(max_diff, float(diff.max()))
            cases.append({
                "thumb_rgb_hwc": [int(v) for v in thumb.reshape(-1)],
                "features": feats,
                "expected_class_logits": [float(v) for v in logits],
                "expected_params": [float(v) for v in params],
            })
    finally:
        torch.set_num_threads(old_threads)
    golden = {"model": "model.trcm", "cases": cases, "atol": ATOL, "rtol": RTOL}
    (out / "model.trcm").write_bytes(blob)
    (out / "golden.json").write_text(_dump(golden), encoding="utf-8")
    return {"model_path": str(out / "model.trcm"), "golden_path": str(out / "golden.json"),
            "model_bytes": len(blob), "cases": len(cases), "classes": len(res.spec.classes),
            "features": len(res.spec.feature_names), "max_abs_diff_torch": max_diff}
