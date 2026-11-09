"""Оценка готовой `.trcm` на датасете эталонным интерпретатором (как модель увидит телефон)."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Callable, Mapping

import numpy as np

from .dataset import DEFAULT_DELTA_EV, Dataset, derive_targets, load_dataset
from .reference import ReferenceModel
from .schema import FOCUS_PARAM, PARAM_NAMES_V1

_SHORT = {"log2_exposure_s": "выдержка", "log2_iso": "ISO", "log2_frames": "кадры", "ev_bias": "EV"}


def evaluate_model(dataset: Dataset | str | os.PathLike, model: ReferenceModel | str | os.PathLike,
                   delta_ev: float = DEFAULT_DELTA_EV, limit: int | None = None,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    ds = dataset if isinstance(dataset, Dataset) else load_dataset(dataset)
    ref = model if isinstance(model, ReferenceModel) else ReferenceModel(model)
    index = {cid: i for i, cid in enumerate(ref.class_ids)}
    k = ref.num_classes
    samples = ds.samples[:limit] if limit else ds.samples
    unknown: Counter = Counter()
    confusion = np.zeros((k, k), dtype=np.int64)
    top3 = 0
    abs_err: dict[str, list[float]] = {}
    focus_hits: list[float] = []
    for i, s in enumerate(samples):
        if progress is not None and (i % 100 == 0 or i == len(samples) - 1):
            progress(i + 1, len(samples))
        true = index.get(s.category_id)
        if true is None:
            unknown[s.category_id] += 1
            continue
        logits, params = ref.predict(s.thumb, s.features)
        pred = int(np.argmax(logits))
        confusion[true, pred] += 1
        top3 += int(true in np.argsort(-logits)[:3])
        real = ref.denormalize_params(params)[true]
        target, mask = derive_targets(s, delta_ev)
        for j, name in enumerate(ref.param_names):
            if name not in PARAM_NAMES_V1:
                continue
            t = PARAM_NAMES_V1.index(name)
            if not mask[t]:
                continue
            if name == FOCUS_PARAM:
                focus_hits.append(float((real[j] >= 0) == (target[t] >= 0.5)))
            else:
                abs_err.setdefault(name, []).append(abs(float(real[j]) - float(target[t])))
    n = int(confusion.sum())
    per_class = []
    for i, c in enumerate(ref.classes):
        total = int(confusion[i].sum())
        per_class.append({"id": c["id"], "name": c.get("name", c["id"]), "icon": c.get("icon"), "count": total,
                          "acc": float(confusion[i, i] / total) if total else None})
    return {
        "model": ref.header.get("name"),
        "dataset": ds.location,
        "samples": len(samples),
        "evaluated": n,
        "unknown_categories": dict(unknown),
        "accuracy": float(np.trace(confusion) / n) if n else None,
        "top3_accuracy": float(top3 / n) if n else None,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "mae": {name: float(np.mean(v)) for name, v in abs_err.items()},
        "focus_acc": float(np.mean(focus_hits)) if focus_hits else None,
        "delta_ev": delta_ev,
    }


def format_eval(rep: Mapping[str, Any]) -> str:
    lines = [f"Модель: {rep['model']}", f"Датасет: {rep['dataset']} — снимков {rep['samples']}, "
                                         f"оценено {rep['evaluated']}"]
    if rep["unknown_categories"]:
        lines.append("  пропущены категории, которых нет в модели: "
                     + ", ".join(f"{cid} ({cnt})" for cid, cnt in rep["unknown_categories"].items()))
    if rep["accuracy"] is None:
        lines.append("Нет снимков с категориями этой модели — оценивать нечего.")
        return "\n".join(lines)
    lines.append(f"Точность категории: {rep['accuracy']:.3f} (среди 3 лучших: {rep['top3_accuracy']:.3f})")
    lines.append("По классам:")
    for c in rep["per_class"]:
        acc = f"{c['acc']:.3f}" if c["acc"] is not None else "  —  "
        lines.append(f"  {c['name'][:28]:<28} {acc}  ({c['count']})")
    if rep["mae"]:
        lines.append("Средняя ошибка параметров для верного класса: "
                     + ", ".join(f"{_SHORT.get(k, k)} {v:.2f}" for k, v in rep["mae"].items())
                     + "  (log2: 1.0 = одна ступень EV)")
    if rep["focus_acc"] is not None:
        lines.append(f"Фокус на ∞: точность {rep['focus_acc']:.3f}")
    return "\n".join(lines)
