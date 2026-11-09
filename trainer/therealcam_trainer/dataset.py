"""Загрузка датасета приложения TheRealCam (docs/DATASET_FORMAT.md) и подготовка данных для обучения.

Источник истины — ``samples/*/meta.json``; манифест ``dataset.json`` только для справки.
В обучение идут лишь образцы с ``labels.complete == true`` и корректным ``thumb64.png``.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .errors import DatasetError
from .reference import effective_std, raw_feature_value
from .schema import (
    CATEGORIES_SCHEMA,
    DEFAULT_ICON,
    FEATURE_NAMES_V1,
    FOCUS_PARAM,
    LABEL_TERNARY_FIELDS,
    MIN_SAMPLES,
    PARAM_NAMES_V1,
    SAMPLE_SCHEMA,
    THUMB_SIZE,
    icon_profile,
)

DEFAULT_DELTA_EV = 0.7
DEFAULT_VAL_FRACTION = 0.15
MIN_FEATURE_PRESENCE = 0.10
FEATURE_STD_FLOOR = 1e-3
PARAM_STD_FLOOR = 0.25
FEATURE_CLIP = 6.0
# Кадры серий описывают одну сцену; для параметров важнее одиночные снимки и итоги серий.
FRAME_PARAM_FACTOR = 0.5
# Нормализация параметра, для которого в обучающей выборке нет ни одной цели.
PARAM_FALLBACK_MEAN = {
    "log2_exposure_s": math.log2(1 / 60),
    "log2_iso": math.log2(200),
    "log2_frames": 0.0,
    "focus_infinity": 0.0,
    "ev_bias": 0.0,
}
MAX_PROBLEMS_LISTED = 50
_SEARCH_MAX_DEPTH = 4
_SEARCH_MAX_DIRS = 5000
_ZIP_META_RE = re.compile(r"^(?P<prefix>(?:[^/]+/)*)samples/(?P<sid>[^/]+)/meta\.json$")

__all__ = [
    "DEFAULT_DELTA_EV",
    "DatasetError",
    "Category",
    "Sample",
    "Dataset",
    "TrainingData",
    "load_dataset",
    "derive_targets",
    "split_train_val",
    "select_features",
    "build_training_data",
    "dataset_report",
    "format_report",
]


# ---------------------------------------------------------------------------
# Модель данных


@dataclass
class Category:
    id: str
    name: str
    icon: str
    created_utc_ms: int | None = None

    @property
    def profile(self) -> str:
        return icon_profile(self.icon)

    def as_class(self) -> dict:
        return {"id": self.id, "name": self.name, "icon": self.icon}


@dataclass
class Sample:
    sample_id: str
    thumb: np.ndarray  # uint8 [64, 64, 3]
    features: dict[str, float | None]  # сырые значения признаков FEATURE_NAMES_V1
    labels: dict[str, Any]
    category_id: str
    rating: int
    mode: str | None
    series_id: str | None
    series_role: str | None  # "frame" | "result" | None
    meta: dict[str, Any]  # компактная выжимка meta.json для расчёта целей

    @property
    def group_id(self) -> str:
        return f"series:{self.series_id}" if self.series_id else f"sample:{self.sample_id}"


@dataclass
class Dataset:
    location: str  # что указал пользователь
    kind: str  # "folder" | "zip"
    root: str  # папка датасета или префикс внутри ZIP
    samples: list[Sample]
    categories: dict[str, Category]
    category_order: list[str]
    manifest: dict | None
    counts: Counter
    problems: list[str] = field(default_factory=list)

    def classes(self) -> list[Category]:
        """Категории, у которых есть хотя бы один пригодный образец, в порядке categories.json."""
        used = {s.category_id for s in self.samples}
        return [self.categories[cid] for cid in self.category_order if cid in used]


# ---------------------------------------------------------------------------
# Источники: папка или ZIP


class _FolderSource:
    kind = "folder"

    def __init__(self, root: Path):
        self.root = root

    def describe(self) -> str:
        return str(self.root)

    def sample_ids(self) -> list[str]:
        sdir = self.root / "samples"
        return sorted(p.name for p in sdir.iterdir() if p.is_dir() and (p / "meta.json").is_file())

    def read(self, rel: str) -> bytes | None:
        try:
            return self.root.joinpath(*rel.split("/")).read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            return None

    def close(self) -> None:
        pass


class _ZipSource:
    kind = "zip"

    def __init__(self, zf: zipfile.ZipFile, prefix: str, names: dict[str, str]):
        self.zf = zf
        self.prefix = prefix
        self.names = names  # нормализованное имя -> имя в архиве

    def describe(self) -> str:
        return self.prefix.rstrip("/") or "(корень архива)"

    def sample_ids(self) -> list[str]:
        ids = []
        for norm in self.names:
            m = _ZIP_META_RE.match(norm)
            if m and m.group("prefix") == self.prefix:
                ids.append(m.group("sid"))
        return sorted(set(ids))

    def read(self, rel: str) -> bytes | None:
        name = self.names.get(self.prefix + rel)
        if name is None:
            return None
        return self.zf.read(name)

    def close(self) -> None:
        self.zf.close()


def _has_samples(d: Path) -> bool:
    s = d / "samples"
    if not s.is_dir():
        return False
    try:
        return any((child / "meta.json").is_file() for child in s.iterdir())
    except OSError:
        return False


def _find_folder_root(path: Path) -> Path | None:
    """Ищет в ширину папку D с ``D/samples/*/meta.json`` (сама папка, TheRealCam/, любая выше)."""
    level = [path]
    visited = 0
    for _ in range(_SEARCH_MAX_DEPTH + 1):
        found = [d for d in level if _has_samples(d)]
        if found:
            named = [d for d in found if d.name == "dataset"]
            return (named or found)[0]
        nxt = []
        for d in level:
            try:
                children = sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")
                                  and c.name not in ("samples", "__MACOSX"))
            except OSError:
                continue
            nxt.extend(children)
            visited += len(children)
            if visited > _SEARCH_MAX_DIRS:
                return None
        level = nxt
    return None


def _open_zip(path: Path) -> _ZipSource:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetError(f"не удалось открыть ZIP {path}: {exc}") from None
    names: dict[str, str] = {}
    for info in zf.infolist():
        norm = info.filename.replace("\\", "/")
        if norm.startswith("__MACOSX/") or info.is_dir():
            continue
        names[norm] = info.filename
    prefixes = Counter()
    for norm in names:
        m = _ZIP_META_RE.match(norm)
        if m:
            prefixes[m.group("prefix")] += 1
    if not prefixes:
        zf.close()
        raise DatasetError(f"в архиве {path} нет файлов samples/<id>/meta.json — это не экспорт датасета TheRealCam")
    prefix = min(prefixes, key=lambda p: (p.count("/"), -prefixes[p], p))
    return _ZipSource(zf, prefix, names)


def _open_source(path: str | os.PathLike):
    p = Path(path).expanduser()
    if not p.exists():
        raise DatasetError(f"путь не найден: {p}")
    if p.is_file():
        if zipfile.is_zipfile(p):
            return _open_zip(p)
        raise DatasetError(f"{p} — не папка и не ZIP-архив")
    root = _find_folder_root(p)
    if root is None:
        raise DatasetError(
            f"в {p} не найдена папка датасета (ожидается dataset/samples/<id>/meta.json). "
            "Укажите папку TheRealCam/dataset, её родителя TheRealCam или ZIP-экспорт из приложения."
        )
    return _FolderSource(root)


# ---------------------------------------------------------------------------
# Разбор meta.json


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return raw_feature_value(value)


def _pos(value: Any) -> float | None:
    v = _num(value)
    return v if v is not None and v > 0 else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _validate_labels(labels: Mapping[str, Any]) -> str | None:
    rating = _int_value(labels.get("rating"))
    if rating is None or not 1 <= rating <= 10:
        return f"rating={labels.get('rating')!r} (нужно 1…10)"
    for key in LABEL_TERNARY_FIELDS:
        v = _int_value(labels.get(key))
        if v not in (-1, 0, 1):
            return f"{key}={labels.get(key)!r} (нужно -1/0/1)"
    mb = labels.get("motion_blur")
    if isinstance(mb, bool):
        mb = int(mb)
    if _int_value(mb) not in (0, 1):
        return f"motion_blur={labels.get('motion_blur')!r} (нужно 0/1)"
    cid = labels.get("category_id")
    if not isinstance(cid, str) or not cid:
        return "нет category_id"
    return None


def _clean_labels(labels: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(labels)
    out["rating"] = _int_value(labels["rating"])
    for key in LABEL_TERNARY_FIELDS:
        out[key] = _int_value(labels[key])
    mb = labels["motion_blur"]
    out["motion_blur"] = int(mb) if isinstance(mb, bool) else _int_value(mb)
    return out


def _obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _slim_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Только то, что нужно для целей обучения (без массивов гироскопа и т.п.)."""
    series = meta.get("series") if isinstance(meta.get("series"), dict) else None
    device = _obj(meta.get("device"))
    return {
        "capture": _obj(meta.get("capture")),
        "user_settings": _obj(meta.get("user_settings")),
        "preview": _obj(meta.get("preview")),
        "series": {k: series.get(k) for k in ("series_id", "role", "index", "count", "profile", "stack_mode")}
        if series is not None else None,
        "device": {"iso_range": device.get("iso_range"), "exposure_range_ns": device.get("exposure_range_ns")},
        "mode": meta.get("mode"),
        "photo_role": _obj(meta.get("photo")).get("role"),
    }


def _read_features(meta: Mapping[str, Any]) -> dict[str, float | None]:
    raw = _obj(meta.get("features"))
    return {name: raw_feature_value(raw.get(name)) for name in FEATURE_NAMES_V1}


def _load_thumb(data: bytes) -> tuple[np.ndarray | None, str | None]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.size != (THUMB_SIZE, THUMB_SIZE):
                return None, f"размер {img.size[0]}×{img.size[1]}, нужен 64×64"
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
    except Exception as exc:  # повреждённый PNG
        return None, f"не читается ({exc})"
    return arr, None


# ---------------------------------------------------------------------------
# Загрузка


def load_dataset(path: str | os.PathLike) -> Dataset:
    """Загружает датасет из папки (dataset/, TheRealCam/ или любой папки выше) или из ZIP."""
    src = _open_source(path)
    try:
        return _load_from_source(src, str(path))
    finally:
        src.close()


def _load_json(src, rel: str, problems: list[str]) -> dict | None:
    raw = src.read(rel)
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        problems.append(f"{rel}: некорректный JSON ({exc})")
        return None
    if not isinstance(data, dict):
        problems.append(f"{rel}: ожидается JSON-объект")
        return None
    return data


def _load_from_source(src, location: str) -> Dataset:
    problems: list[str] = []
    counts: Counter = Counter()

    categories: dict[str, Category] = {}
    order: list[str] = []
    cats_json = _load_json(src, "categories.json", problems)
    if cats_json is not None:
        if cats_json.get("schema") not in (None, CATEGORIES_SCHEMA):
            problems.append(f"categories.json: неизвестная схема {cats_json.get('schema')!r}")
        for item in cats_json.get("categories") or []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item.get("id"):
                problems.append("categories.json: категория без id пропущена")
                continue
            cid = item["id"]
            if cid in categories:
                continue
            name = item.get("name") if isinstance(item.get("name"), str) and item.get("name") else cid
            icon = item.get("icon") if isinstance(item.get("icon"), str) and item.get("icon") else DEFAULT_ICON
            categories[cid] = Category(cid, name, icon, _int_value(item.get("created_utc_ms")))
            order.append(cid)
    manifest = _load_json(src, "dataset.json", problems)

    samples: list[Sample] = []
    from_labels: dict[str, Category] = {}
    for sid in src.sample_ids():
        counts["found"] += 1
        rel = f"samples/{sid}"
        meta = _load_json(src, f"{rel}/meta.json", problems)
        if meta is None:
            counts["bad_meta"] += 1
            continue
        schema = meta.get("schema")
        if schema is not None and schema != SAMPLE_SCHEMA:
            counts["other_schema"] += 1
        labels = meta.get("labels")
        if not isinstance(labels, dict) or labels.get("complete") is not True:
            counts["incomplete"] += 1
            continue
        why = _validate_labels(labels)
        if why:
            counts["bad_labels"] += 1
            problems.append(f"{sid}: labels.complete=true, но {why} — снимок пропущен")
            continue
        labels = _clean_labels(labels)
        thumb_bytes = src.read(f"{rel}/thumb64.png")
        if thumb_bytes is None:
            counts["no_thumb"] += 1
            problems.append(f"{sid}: нет thumb64.png — снимок пропущен")
            continue
        thumb, why = _load_thumb(thumb_bytes)
        if thumb is None:
            counts["bad_thumb"] += 1
            problems.append(f"{sid}: thumb64.png {why} — снимок пропущен")
            continue
        series = meta.get("series") if isinstance(meta.get("series"), dict) else None
        series_id = series.get("series_id") if series else None
        if not isinstance(series_id, str) or not series_id:
            series_id = None
        role = series.get("role") if series and series.get("role") in ("frame", "result") else None
        cid = labels["category_id"]
        if cid not in categories:
            name = labels.get("category_name")
            icon = labels.get("category_icon")
            # Более поздний снимок (id сортируется по времени) несёт более свежее имя/значок.
            from_labels[cid] = Category(cid, name if isinstance(name, str) and name else cid,
                                        icon if isinstance(icon, str) and icon else DEFAULT_ICON)
        mode = meta.get("mode") if isinstance(meta.get("mode"), str) else None
        samples.append(Sample(
            sample_id=sid,
            thumb=thumb,
            features=_read_features(meta),
            labels=labels,
            category_id=cid,
            rating=labels["rating"],
            mode=mode,
            series_id=series_id,
            series_role=role,
            meta=_slim_meta(meta),
        ))
    if counts["other_schema"]:
        problems.append(f"{counts['other_schema']} снимков с неизвестной схемой meta.json (ожидается {SAMPLE_SCHEMA})")
    if from_labels:
        if cats_json is not None:
            problems.append("категории, которых нет в categories.json, взяты из labels: "
                            + ", ".join(sorted(from_labels)))
        for cid in sorted(from_labels):
            categories[cid] = from_labels[cid]
            order.append(cid)
    counts["used"] = len(samples)
    return Dataset(
        location=location,
        kind=src.kind,
        root=src.describe(),
        samples=samples,
        categories=categories,
        category_order=order,
        manifest=manifest,
        counts=counts,
        problems=problems,
    )


# ---------------------------------------------------------------------------
# Цели обучения


def derive_targets(sample: Sample, delta_ev: float = DEFAULT_DELTA_EV) -> tuple[np.ndarray, np.ndarray]:
    """Цели [P] в порядке PARAM_NAMES_V1 (NaN = нет цели) и маска доступности.

    База — фактические параметры снимка, затем поправки по оценкам пользователя
    (правила описаны в trainer/README.md).
    """
    meta = sample.meta
    cap, us, pv = meta["capture"], meta["user_settings"], meta["preview"]
    labels = sample.labels

    exp_ns = next((v for v in (_pos(cap.get("exposure_time_ns")), _pos(us.get("exposure_time_ns")),
                               _pos(pv.get("exposure_time_ns"))) if v is not None), None)
    iso_v = next((v for v in (_pos(cap.get("iso")), _pos(us.get("iso")), _pos(pv.get("iso"))) if v is not None), None)
    exp = math.log2(exp_ns / 1e9) if exp_ns is not None else math.nan
    iso = math.log2(iso_v) if iso_v is not None else math.nan

    series = meta.get("series")
    count = _pos(series.get("count")) if series else None
    if count is not None:
        frames = math.log2(max(1.0, count))
    else:
        f = _pos(us.get("frames"))
        frames = math.log2(max(1.0, f)) if f is not None else 0.0

    fd = _num(cap.get("focus_distance_diopters"))
    focus = 1.0 if ((fd is not None and fd <= 0.1) or us.get("focus") == "infinity") else 0.0
    ev = next((v for v in (_num(cap.get("ev_compensation")), _num(us.get("ev"))) if v is not None), 0.0)

    d = float(delta_ev)
    exposure = labels.get("exposure")
    blur = labels.get("motion_blur") == 1
    if exposure == -1:  # темно
        if blur:
            iso += d
        else:
            exp += d / 2
            iso += d / 2
        ev += d
    elif exposure == 1:  # пересвет
        exp -= d / 2
        iso -= d / 2
        ev -= d
        iso_range = meta["device"].get("iso_range")
        iso_min = _pos(iso_range[0]) if isinstance(iso_range, list) and iso_range else None
        if iso_min is not None and iso < math.log2(iso_min):
            deficit = math.log2(iso_min) - iso
            iso = math.log2(iso_min)
            exp -= deficit
    if blur:
        exp -= 1.0
        iso += 1.0
    if labels.get("noise") == -1:
        if blur:
            frames += 1.0
        else:
            iso -= 0.5
            exp += 0.5

    values = np.array([exp, iso, frames, focus, ev], dtype=np.float64)
    if exp_ns is None and iso_v is None:
        mask = np.zeros(len(PARAM_NAMES_V1), dtype=bool)
    else:
        mask = np.array([exp_ns is not None, iso_v is not None, True, True, True], dtype=bool)
    values[~mask] = np.nan
    return values, mask


# ---------------------------------------------------------------------------
# Разбиение train/val


def split_train_val(group_ids: Sequence[str], y: np.ndarray, num_classes: int,
                    val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Группы (серия или одиночный снимок) целиком уходят в train или val; ~val_fraction по каждому
    классу; в train остаётся хотя бы один образец каждого класса."""
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n < 2 or val_fraction <= 0:
        return np.arange(n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    gids = sorted(groups)
    rng = np.random.default_rng(seed)
    gids = [gids[i] for i in rng.permutation(len(gids))]
    total = np.bincount(y, minlength=num_classes)
    quota = np.floor(total * val_fraction + 0.5).astype(np.int64)
    val_cnt = np.zeros(num_classes, dtype=np.int64)
    train_cnt = total.copy()
    chosen: set[str] = set()
    target_total = int(round(n * val_fraction))
    for strict in (True, False):
        for gid in gids:
            if gid in chosen:
                continue
            cnt = np.bincount(y[groups[gid]], minlength=num_classes)
            cls = np.nonzero(cnt)[0]
            if np.any(train_cnt[cls] - cnt[cls] < 1):
                continue
            if np.any(val_cnt[cls] >= quota[cls]):
                continue
            if strict and np.any(val_cnt[cls] + cnt[cls] > quota[cls] + np.maximum(1, quota[cls] // 2)):
                continue
            chosen.add(gid)
            val_cnt += cnt
            train_cnt -= cnt
        if val_cnt.sum() * 2 >= target_total:
            break
    val = sorted(i for g in chosen for i in groups[g])
    val_set = set(val)
    train = [i for i in range(n) if i not in val_set]
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


# ---------------------------------------------------------------------------
# Признаки и нормализация


def feature_matrix(samples: Sequence[Sample], names: Sequence[str]) -> np.ndarray:
    """Сырые признаки [N, F], NaN — отсутствует."""
    out = np.full((len(samples), len(names)), np.nan, dtype=np.float64)
    for i, s in enumerate(samples):
        for j, name in enumerate(names):
            v = s.features.get(name)
            if v is not None:
                out[i, j] = v
    return out


def select_features(raw: np.ndarray, names: Sequence[str], min_presence: float = MIN_FEATURE_PRESENCE) -> list[str]:
    """Признаки, заполненные (не null) хотя бы в min_presence доли образцов."""
    if raw.shape[0] == 0:
        return []
    presence = np.isfinite(raw).mean(axis=0)
    return [n for n, p in zip(names, presence) if p >= min_presence]


def _f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def feature_norm(raw: np.ndarray, fallback_mean=None, fallback_std=None) -> tuple[np.ndarray, np.ndarray]:
    """mean/std по заполненным значениям; std < 1e-3 -> 1; признак без значений -> запасные mean/std."""
    f = raw.shape[1]
    mean = np.zeros(f)
    std = np.ones(f)
    for j in range(f):
        col = raw[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            if fallback_mean is not None:
                mean[j], std[j] = fallback_mean[j], fallback_std[j]
            continue
        mean[j] = col.mean()
        sd = col.std()
        std[j] = sd if sd >= FEATURE_STD_FLOOR else 1.0
    return _f32(mean), _f32(std)


def normalize_feature_matrix(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Та же формула, что в приложении: z=(x-mean)/std, отсутствующие -> 0, клип [-6, 6]."""
    present = np.isfinite(raw)
    eff = np.array([effective_std(s) for s in std], dtype=np.float64)
    z = np.where(present, (np.where(present, raw, 0.0) - mean) / eff, 0.0)
    return np.clip(z, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32), present


def param_norm(targets: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean/std целей; std >= 0.25; focus_infinity — логит: mean 0, std 1."""
    p = len(PARAM_NAMES_V1)
    mean = np.zeros(p)
    std = np.ones(p)
    for j, name in enumerate(PARAM_NAMES_V1):
        if name == FOCUS_PARAM:
            continue
        col = targets[mask[:, j], j]
        if col.size == 0:
            mean[j] = PARAM_FALLBACK_MEAN[name]
            continue
        mean[j] = col.mean()
        std[j] = max(float(col.std()), PARAM_STD_FLOOR)
    return _f32(mean), _f32(std)


# ---------------------------------------------------------------------------
# Данные для обучения


@dataclass
class TrainingData:
    classes: list[dict]
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    param_names: list[str]
    param_mean: np.ndarray
    param_std: np.ndarray
    sample_ids: list[str]
    images: np.ndarray  # uint8 [N, 64, 64, 3]
    feats: np.ndarray  # float32 [N, F], нормализованные
    feat_present: np.ndarray  # bool [N, F]
    y: np.ndarray  # int64 [N]
    targets: np.ndarray  # float32 [N, P], нормализованные (0 где нет цели)
    target_raw: np.ndarray  # float64 [N, P], реальные (NaN где нет цели)
    target_mask: np.ndarray  # bool [N, P]
    group_w: np.ndarray  # float32 [N]
    param_w: np.ndarray  # float32 [N]
    class_w: np.ndarray  # float32 [K]
    train_idx: np.ndarray
    val_idx: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """1/sqrt(частота класса), нормировано к среднему 1 по встречающимся классам."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = np.ones(num_classes)
    present = counts > 0
    if present.any():
        inv = 1.0 / np.sqrt(counts[present])
        w[present] = inv / inv.mean()
    return w.astype(np.float32)


def build_training_data(ds: Dataset, *, classes: Sequence[Mapping[str, Any]] | None = None,
                        feature_names: Sequence[str] | None = None,
                        fixed_feature_norm: tuple[Sequence[float], Sequence[float]] | None = None,
                        fixed_param_norm: tuple[Sequence[float], Sequence[float]] | None = None,
                        fallback_feature_norm: tuple[Sequence[float], Sequence[float]] | None = None,
                        delta_ev: float = DEFAULT_DELTA_EV, val_fraction: float = DEFAULT_VAL_FRACTION,
                        seed: int = 0, min_feature_presence: float = MIN_FEATURE_PRESENCE) -> TrainingData:
    samples = ds.samples
    if not samples:
        raise DatasetError("в датасете нет ни одного пригодного (полностью оценённого) снимка")
    notes: list[str] = []
    class_list = [dict(c) for c in classes] if classes is not None else [c.as_class() for c in ds.classes()]
    index = {c["id"]: i for i, c in enumerate(class_list)}
    missing = sorted({s.category_id for s in samples if s.category_id not in index})
    if missing:
        raise DatasetError(f"категории без класса: {', '.join(missing)}")
    n = len(samples)
    k = len(class_list)
    y = np.array([index[s.category_id] for s in samples], dtype=np.int64)

    targets = np.zeros((n, len(PARAM_NAMES_V1)))
    tmask = np.zeros((n, len(PARAM_NAMES_V1)), dtype=bool)
    for i, s in enumerate(samples):
        targets[i], tmask[i] = derive_targets(s, delta_ev)

    group_ids = [s.group_id for s in samples]
    train_idx, val_idx = split_train_val(group_ids, y, k, val_fraction, seed)

    if feature_names is None:
        all_raw = feature_matrix(samples, FEATURE_NAMES_V1)
        names = select_features(all_raw[train_idx], FEATURE_NAMES_V1, min_feature_presence)
        dropped = [f for f in FEATURE_NAMES_V1 if f not in names]
        if dropped:
            notes.append(f"признаки, заполненные меньше чем в {min_feature_presence:.0%} снимков, не используются: "
                         + ", ".join(dropped))
    else:
        names = list(feature_names)
    raw = feature_matrix(samples, names)
    if fixed_feature_norm is not None:
        fmean, fstd = _f32(fixed_feature_norm[0]), _f32(fixed_feature_norm[1])
    else:
        fb = fallback_feature_norm
        fmean, fstd = feature_norm(raw[train_idx], *(fb if fb is not None else (None, None)))
    feats, present = normalize_feature_matrix(raw, fmean, fstd)

    if fixed_param_norm is not None:
        pmean, pstd = _f32(fixed_param_norm[0]), _f32(fixed_param_norm[1])
    else:
        pmean, pstd = param_norm(targets[train_idx], tmask[train_idx])
        for j, name in enumerate(PARAM_NAMES_V1):
            if name != FOCUS_PARAM and not tmask[train_idx, j].any():
                notes.append(f"для параметра {name} нет ни одной цели — модель выдаст значение по умолчанию")
    tnorm = np.where(tmask, (np.nan_to_num(targets) - pmean) / pstd, 0.0).astype(np.float32)

    sizes = Counter(group_ids)
    group_w = np.array([1.0 / math.sqrt(sizes[g]) for g in group_ids], dtype=np.float32)
    param_w = np.array([(0.05 + ((s.rating - 1) / 9.0) ** 2) * (FRAME_PARAM_FACTOR if s.series_role == "frame" else 1.0)
                        for s in samples], dtype=np.float32)
    return TrainingData(
        classes=class_list,
        feature_names=list(names),
        feature_mean=fmean,
        feature_std=fstd,
        param_names=list(PARAM_NAMES_V1),
        param_mean=pmean,
        param_std=pstd,
        sample_ids=[s.sample_id for s in samples],
        images=np.stack([s.thumb for s in samples]).astype(np.uint8),
        feats=feats,
        feat_present=present,
        y=y,
        targets=tnorm,
        target_raw=targets,
        target_mask=tmask,
        group_w=group_w,
        param_w=param_w,
        class_w=class_weights(y[train_idx], k),
        train_idx=train_idx,
        val_idx=val_idx,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Отчёт для inspect-dataset


def dataset_report(ds: Dataset, delta_ev: float = DEFAULT_DELTA_EV) -> dict:
    n = len(ds.samples)
    per_cat = Counter(s.category_id for s in ds.samples)
    categories = []
    for cid in ds.category_order:
        c = ds.categories[cid]
        categories.append({"id": cid, "name": c.name, "icon": c.icon, "profile": c.profile, "count": per_cat.get(cid, 0)})
    modes = Counter(s.mode or "unknown" for s in ds.samples)
    ratings = Counter(s.rating for s in ds.samples)
    series_ids = {s.series_id for s in ds.samples if s.series_id}
    label_stats = {}
    for key in (*LABEL_TERNARY_FIELDS, "motion_blur"):
        label_stats[key] = dict(sorted(Counter(s.labels.get(key) for s in ds.samples).items()))
    raw = feature_matrix(ds.samples, FEATURE_NAMES_V1)
    missing = {name: (float(1.0 - np.isfinite(raw[:, j]).mean()) if n else 1.0)
               for j, name in enumerate(FEATURE_NAMES_V1)}
    avail = np.zeros(len(PARAM_NAMES_V1))
    for s in ds.samples:
        avail += derive_targets(s, delta_ev)[1]
    usable_features = select_features(raw, FEATURE_NAMES_V1) if n else []
    return {
        "location": ds.location,
        "kind": ds.kind,
        "root": ds.root,
        "manifest_complete_samples": (ds.manifest or {}).get("complete_samples"),
        "counts": {
            "found": ds.counts.get("found", 0),
            "incomplete": ds.counts.get("incomplete", 0),
            "bad_meta": ds.counts.get("bad_meta", 0),
            "bad_labels": ds.counts.get("bad_labels", 0),
            "no_thumb": ds.counts.get("no_thumb", 0),
            "bad_thumb": ds.counts.get("bad_thumb", 0),
            "used": n,
        },
        "min_samples": MIN_SAMPLES,
        "ready": n >= MIN_SAMPLES,
        "categories": categories,
        "modes": dict(sorted(modes.items())),
        "ratings": {r: ratings.get(r, 0) for r in range(1, 11)},
        "series": {
            "series": len(series_ids),
            "frames": sum(1 for s in ds.samples if s.series_role == "frame"),
            "results": sum(1 for s in ds.samples if s.series_role == "result"),
            "singles": sum(1 for s in ds.samples if not s.series_id),
        },
        "labels": label_stats,
        "missing_features": missing,
        "usable_features": usable_features,
        "targets_available": {name: (float(avail[j] / n) if n else 0.0) for j, name in enumerate(PARAM_NAMES_V1)},
        "problems": list(ds.problems),
    }


def format_report(rep: Mapping[str, Any]) -> str:
    c = rep["counts"]
    lines = [f"Датасет: {rep['location']}"]
    if rep["kind"] == "zip":
        lines.append(f"  ZIP-архив, папка внутри: {rep['root']}")
    else:
        lines.append(f"  папка датасета: {rep['root']}")
    if rep.get("manifest_complete_samples") is not None:
        lines.append(f"  dataset.json: complete_samples={rep['manifest_complete_samples']} "
                     "(для справки, тренер сканирует samples/ сам)")
    lines.append(f"Найдено снимков: {c['found']}")
    skipped = [("неполная оценка (labels.complete != true)", c["incomplete"]),
               ("повреждённый meta.json", c["bad_meta"]),
               ("некорректные labels", c["bad_labels"]),
               ("нет thumb64.png", c["no_thumb"]),
               ("неверный thumb64.png", c["bad_thumb"])]
    for what, count in skipped:
        if count:
            lines.append(f"  пропущено — {what}: {count}")
    n = c["used"]
    need = rep["min_samples"]
    status = "достаточно для обучения" if rep["ready"] else f"НЕДОСТАТОЧНО, нужно ещё {need - n}"
    lines.append(f"Пригодно для обучения: {n} (минимум {need}) — {status}")

    lines.append("")
    lines.append(f"Категории ({sum(1 for x in rep['categories'] if x['count'])} с данными):")
    for cat in rep["categories"]:
        pct = f"{100.0 * cat['count'] / n:5.1f}%" if n else "  —  "
        lines.append(f"  {cat['name'][:28]:<28} {cat['icon']:<13} профиль {cat['profile']:<9} {cat['count']:>6}  {pct}"
                     f"   [{cat['id']}]")
    modes = ", ".join(f"{k} {v}" for k, v in rep["modes"].items()) or "—"
    lines.append(f"Режимы съёмки: {modes}")
    s = rep["series"]
    lines.append(f"Серии: {s['series']} (кадров {s['frames']}, итогов {s['results']}); одиночных снимков {s['singles']}")
    lines.append("Оценки 1–10:")
    peak = max(rep["ratings"].values()) or 1
    for r, cnt in rep["ratings"].items():
        lines.append(f"  {r:>2} | {'#' * int(round(30 * cnt / peak)):<30} {cnt}")
    lab = rep["labels"]
    lines.append("Метки: " + "; ".join(f"{k} " + "/".join(f"{v}:{cnt}" for v, cnt in vals.items())
                                       for k, vals in lab.items()))
    lines.append("Доступность целей: " + ", ".join(f"{k} {v:.0%}" for k, v in rep["targets_available"].items()))
    lines.append(f"Пропуски признаков (доля null), используется {len(rep['usable_features'])} из "
                 f"{len(rep['missing_features'])} (нужно ≥10% заполненных):")
    items = [f"{name:<17}{frac:5.0%}" for name, frac in rep["missing_features"].items()]
    for i in range(0, len(items), 3):
        lines.append("  " + "   ".join(items[i:i + 3]))
    probs = rep["problems"]
    if probs:
        lines.append(f"Проблемы ({len(probs)}):")
        for p in probs[:MAX_PROBLEMS_LISTED]:
            lines.append(f"  - {p}")
        if len(probs) > MAX_PROBLEMS_LISTED:
            lines.append(f"  … и ещё {len(probs) - MAX_PROBLEMS_LISTED}")
    else:
        lines.append("Проблем не найдено.")
    return "\n".join(lines)
