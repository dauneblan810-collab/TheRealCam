"""СИНТЕТИЧЕСКИЙ датасет в точной структуре DATASET_FORMAT.md — только для тестов и демонстрации.

Это не данные пользователя и не «предобучение»: такие снимки нельзя использовать для настоящей
модели. Каждый файл помечен как синтетический (категории «(СИНТЕТИКА)», device.manufacturer
SYNTHETIC, labels.note, файл SYNTHETIC_DATASET.txt).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import CATEGORIES_SCHEMA, DATASET_SCHEMA, FEATURE_NAMES_V1, MIN_SAMPLES, SAMPLE_SCHEMA, icon_profile

SYNTH_NOTE = "SYNTHETIC — сгенерировано therealcam-trainer synth-dataset, только для тестов"
TZ_OFFSET_MIN = 180
PREVIEW_LONG, PREVIEW_SHORT = 256, 192

KINDS = {
    "stars": dict(name="Звёзды (СИНТЕТИКА)", icon="stars", share=0.30, exp=(1.0, 3.0), iso=(9.6, 11.6),
                  frames=(8, 16), focus_inf=True, hours=(21.0, 27.0), pitch=(20.0, 80.0), lux=(0.01, 2.0),
                  pv_exp=(1 / 15, 1 / 8), pv_iso=(1600, 3200), zoom=(1.0,), fd=(0.0, 0.05), tripod=0.8),
    "moon": dict(name="Луна (СИНТЕТИКА)", icon="moon", share=0.20, exp=(-9.0, -7.0), iso=(5.6, 7.6),
                 frames=(16, 32), focus_inf=True, hours=(19.0, 29.0), pitch=(10.0, 70.0), lux=(0.05, 5.0),
                 pv_exp=(1 / 60, 1 / 30), pv_iso=(400, 1600), zoom=(2.0, 3.0, 5.0), fd=(0.0, 0.05), tripod=0.8),
    "clouds": dict(name="Облака (СИНТЕТИКА)", icon="clouds", share=0.25, exp=(-12.0, -9.0), iso=(5.6, 6.6),
                   frames=(1, 4), focus_inf=True, hours=(8.0, 19.0), pitch=(10.0, 60.0), lux=(3000.0, 60000.0),
                   pv_exp=(1 / 4000, 1 / 500), pv_iso=(50, 100), zoom=(1.0, 0.6), fd=(0.0, 0.15), tripod=0.1),
    "other": dict(name="Другое (СИНТЕТИКА)", icon="other", share=0.25, exp=(-7.0, -5.0), iso=(6.6, 9.6),
                  frames=(1, 1), focus_inf=False, hours=(7.0, 23.0), pitch=(-30.0, 30.0), lux=(30.0, 1000.0),
                  pv_exp=(1 / 120, 1 / 30), pv_iso=(100, 800), zoom=(1.0, 2.0, 0.6), fd=(0.3, 5.0), tripod=0.05),
}
STACK_MODES = {"astro": "sigma_clip", "lunar": "lucky", "sky": "mean"}
ISO_RANGE = (50, 3200)


# ---------------------------------------------------------------------------
# Картинки (float RGB [H, W, 3] в [0, 1])


def _gradient(h: int, w: int, top, bottom) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.asarray(top, dtype=np.float32)[None, None, :]
    bottom = np.asarray(bottom, dtype=np.float32)[None, None, :]
    return np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()


def _blobs(rng, h: int, w: int, count: int, sigma: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        sy, sx = rng.uniform(*sigma), rng.uniform(*sigma)
        acc += rng.uniform(0.6, 1.6) * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)
    return acc


def render_stars(rng, h: int, w: int) -> np.ndarray:
    k = rng.uniform(0.6, 1.4)
    img = _gradient(h, w, np.array([0.010, 0.014, 0.030]) * k, np.array([0.04, 0.035, 0.05]) * rng.uniform(0.6, 2.0))
    img += rng.normal(0, 0.008, (h, w, 3)).astype(np.float32)
    n = int(rng.integers(60, 260))
    ys, xs = rng.integers(1, h - 1, n), rng.integers(1, w - 1, n)
    b = rng.uniform(0.25, 1.0, n) ** 2
    tint = 1 + rng.normal(0, 0.08, (n, 3))
    img[ys, xs] += (b[:, None] * tint).astype(np.float32)
    big = b > 0.45
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        img[ys[big] + dy, xs[big] + dx] += (0.6 * b[big, None] * tint[big]).astype(np.float32)
    if rng.random() < 0.5:  # силуэт земли
        horizon = h * rng.uniform(0.78, 0.94) + 4 * np.sin(np.linspace(0, rng.uniform(3, 9), w) + rng.uniform(0, 6))
        ground = np.arange(h)[:, None] > horizon[None, :]
        img[ground] = 0.012
    return np.clip(img, 0, 1)


def render_moon(rng, h: int, w: int) -> np.ndarray:
    img = _gradient(h, w, [0.010, 0.012, 0.022], [0.018, 0.018, 0.026]) * rng.uniform(0.7, 1.5)
    img += rng.normal(0, 0.006, (h, w, 3)).astype(np.float32)
    cy, cx = h * rng.uniform(0.3, 0.7), w * rng.uniform(0.3, 0.7)
    r = rng.uniform(12, 32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = np.clip(r - d + 0.5, 0, 1)
    limb = np.clip(1 - (d / r) ** 2, 0, 1) ** 0.25
    maria = np.zeros_like(d)
    for _ in range(int(rng.integers(3, 7))):
        my, mx = cy + rng.uniform(-0.6, 0.6) * r, cx + rng.uniform(-0.6, 0.6) * r
        ms = rng.uniform(0.12, 0.35) * r
        maria += np.exp(-(((yy - my) ** 2 + (xx - mx) ** 2) / (2 * ms * ms)))
    albedo = 1 - 0.35 * np.clip(maria, 0, 1)
    lit = np.ones_like(d)
    if rng.random() < 0.4:  # фаза: терминатор от смещённого круга
        shift = rng.uniform(0.4, 1.6) * r * rng.choice([-1, 1])
        d2 = np.sqrt((yy - cy) ** 2 + (xx - cx - shift) ** 2)
        lit = np.clip(d2 - r + 0.5, 0, 1)
    moon = disk * limb * albedo * lit * rng.uniform(0.75, 1.0)
    halo = 0.04 * np.exp(-np.maximum(d - r, 0) / (1.5 * r)) * (d > r)
    img += (moon + halo)[..., None] * np.array([1.0, 0.97, 0.9], dtype=np.float32)
    return np.clip(img, 0, 1)


def render_clouds(rng, h: int, w: int) -> np.ndarray:
    top = np.array([0.20, 0.40, 0.80]) * rng.uniform(0.8, 1.1)
    bottom = np.array([0.95, 0.62, 0.38]) if rng.random() < 0.2 else np.array([0.60, 0.75, 0.95])
    img = _gradient(h, w, top, bottom * rng.uniform(0.9, 1.05))
    cover = 1 - np.exp(-_blobs(rng, h, w, int(rng.integers(5, 16)), (8.0, 40.0)) * rng.uniform(0.6, 1.4))
    shade = np.linspace(1.0, 0.82, h, dtype=np.float32)[:, None]
    cloud = (np.array([0.96, 0.96, 0.98], dtype=np.float32) * rng.uniform(0.85, 1.05))[None, None, :] * shade[..., None]
    img = img * (1 - cover[..., None]) + cloud * cover[..., None]
    if rng.random() < 0.35:  # солнце в кадре: пересвеченное пятно
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx, r = rng.uniform(0, h * 0.6), rng.uniform(0, w), rng.uniform(6, 18)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img += (1.5 * np.exp(-np.maximum(d - r, 0) / (2 * r)))[..., None]
    img += rng.normal(0, 0.01, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 1)


def render_other(rng, h: int, w: int) -> np.ndarray:
    base = rng.uniform(0.3, 0.6) * (1 + rng.normal(0, 0.1, 3))
    img = np.broadcast_to(base.astype(np.float32), (h, w, 3)).copy()
    cell = int(rng.choice([8, 16, 32]))
    coarse = rng.random((h // cell + 1, w // cell + 1, 3)).astype(np.float32)
    texture = np.repeat(np.repeat(coarse, cell, axis=0), cell, axis=1)[:h, :w]
    img += (texture - 0.5) * rng.uniform(0.1, 0.3)
    for _ in range(int(rng.integers(3, 9))):
        y0, x0 = int(rng.integers(0, h - 8)), int(rng.integers(0, w - 8))
        y1, x1 = int(rng.integers(y0 + 4, h)), int(rng.integers(x0 + 4, w))
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * rng.uniform(0.6, 1.3) + rng.uniform(-0.1, 0.1)
    img += rng.normal(0, 0.03, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 1)


RENDERERS = {"stars": render_stars, "moon": render_moon, "clouds": render_clouds, "other": render_other}


def render_scene(kind: str, rng, portrait: bool = True) -> np.ndarray:
    h, w = (PREVIEW_LONG, PREVIEW_SHORT) if portrait else (PREVIEW_SHORT, PREVIEW_LONG)
    return RENDERERS[kind](rng, h, w).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


def preview_to_thumb(preview: np.ndarray) -> np.ndarray:
    """Как в приложении: центральный квадрат превью + box-фильтр 3×3 -> 64×64 RGB uint8."""
    h, w = preview.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = preview[y0:y0 + side, x0:x0 + side].astype(np.float64)
    f = side // 64
    sq = sq[: 64 * f, : 64 * f].reshape(64, f, 64, f, 3).mean(axis=(1, 3))
    return to_uint8(sq)


def compute_preview_features(rgb: np.ndarray, exposure_s: float, iso: float, aperture: float = 1.8) -> dict:
    """Признаки по превью — формулы MODEL_FORMAT.md §4 (в приложении их считает FeatureExtractor)."""
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    h, w = y.shape
    bins = np.clip(np.floor(y * 255), 0, 255).astype(np.int64)
    cdf = np.cumsum(np.bincount(bins.ravel(), minlength=256)) / y.size

    def pct(p: float) -> float:
        return float(np.argmax(cdf >= p)) / 255.0

    p50 = pct(0.50)
    lap = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    c = y[1:-1, 1:-1]
    peak = np.ones_like(c, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                peak &= c > y[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
    peak &= (c >= p50 + 0.12) & (c >= 0.2)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    t3h, t3w = h // 3, w // 3
    return {
        "scene_ev": math.log2(aperture ** 2 / exposure_s) - math.log2(iso / 100)
        + math.log2(max(float((y ** 2.2).mean()), 1e-4) / 0.18),
        "preview_exp_log2": math.log2(exposure_s),
        "preview_iso_log2": math.log2(iso),
        "luma_mean": float(y.mean()),
        "luma_std": float(y.std()),
        "luma_p05": pct(0.05),
        "luma_p50": p50,
        "luma_p95": pct(0.95),
        "clip_hi": float((y >= 250 / 255).mean()),
        "clip_lo": float((y <= 5 / 255).mean()),
        "bright_frac": float((y >= 0.7).mean()),
        "sharpness": math.log10(1e-6 + float(lap.var())),
        "point_sources": float(peak.sum()) * 10000.0 / y.size,
        "r_g": float(r.mean() / (g.mean() + 1e-4)),
        "b_g": float(b.mean() / (g.mean() + 1e-4)),
        "saturation": float(((mx - mn) / (mx + 1e-4)).mean()),
        "top_luma": float(y[:t3h].mean()),
        "bottom_luma": float(y[h - t3h:].mean()),
        "center_luma": float(y[t3h:2 * t3h, t3w:2 * t3w].mean()),
    }


# ---------------------------------------------------------------------------
# Сцены и снимки


def _r(v: float | None, nd: int = 4):
    return None if v is None else round(float(v), nd)


def _scene(kind: str, rng, day: datetime) -> dict:
    k = KINDS[kind]
    hour = rng.uniform(*k["hours"])
    local = day + timedelta(hours=hour)
    front = kind == "other" and rng.random() < 0.12
    tripod = rng.random() < k["tripod"]
    lux = math.exp(rng.uniform(math.log(k["lux"][0]), math.log(k["lux"][1])))
    has_location = rng.random() < 0.85
    if kind in ("stars", "moon"):
        sun_alt = rng.uniform(-50, -12)
    elif kind == "clouds":
        sun_alt = rng.uniform(8, 60)
    else:
        sun_alt = rng.uniform(5, 50) if 7 <= local.hour < 19 else rng.uniform(-40, -5)
    ideal_exp = rng.uniform(*k["exp"])
    ideal_iso = rng.uniform(*k["iso"])
    f_lo, f_hi = k["frames"]
    frames = int(rng.integers(f_lo, f_hi + 1))
    return {
        "kind": kind,
        "local": local,
        "portrait": rng.random() < 0.7,
        "front": front,
        "tripod": tripod,
        "lux": lux if rng.random() > 0.05 else None,
        "pitch": rng.uniform(*k["pitch"]) if not front else rng.uniform(-10, 10),
        "roll": rng.normal(0, 3),
        "azimuth": rng.uniform(0, 360),
        "motion_rms": 10 ** (rng.uniform(-3.5, -2.5) if tripod else rng.uniform(-1.8, -0.8)),
        "sun_alt": sun_alt if has_location else None,
        "moon_alt": (rng.uniform(10, 60) if kind == "moon" else rng.uniform(-40, 50)) if has_location else None,
        "moon_illum": rng.uniform(0.3, 1.0) if kind == "moon" else rng.uniform(0.0, 1.0),
        "location": {"lat": round(55.75 + rng.normal(0, 0.3), 5), "lon": round(37.61 + rng.normal(0, 0.5), 5),
                     "alt_m": round(rng.uniform(100, 250), 1), "accuracy_m": round(rng.uniform(3, 30), 1),
                     "provider": "gps", "age_ms": int(rng.integers(100, 5000)), "speed_mps": 0.0,
                     "bearing_deg": None} if has_location else None,
        "zoom": float(rng.choice(k["zoom"])) if not front else 1.0,
        "focus_preview": rng.uniform(*k["fd"]) if not front else rng.uniform(1.0, 3.0),
        "pv_exp": math.exp(rng.uniform(math.log(k["pv_exp"][0]), math.log(k["pv_exp"][1]))),
        "pv_iso": float(int(rng.uniform(*k["pv_iso"]))),
        "ideal_exp": ideal_exp,
        "ideal_iso": ideal_iso,
        "frames": frames,
    }


def _sensor_features(sc: dict) -> dict:
    local: datetime = sc["local"]
    h = local.hour + local.minute / 60 + local.second / 3600
    doy = local.timetuple().tm_yday
    return {
        "lux_log10": math.log10(1 + sc["lux"]) if sc["lux"] is not None else None,
        "pitch_deg": sc["pitch"],
        "roll_deg": sc["roll"],
        "motion_log10": math.log10(1e-4 + sc["motion_rms"]),
        "hour_sin": math.sin(2 * math.pi * h / 24),
        "hour_cos": math.cos(2 * math.pi * h / 24),
        "doy_sin": math.sin(2 * math.pi * (doy - 1) / 365.25),
        "doy_cos": math.cos(2 * math.pi * (doy - 1) / 365.25),
        "sun_alt_deg": sc["sun_alt"],
        "moon_alt_deg": sc["moon_alt"],
        "moon_illum": sc["moon_illum"],
        "focus_diopters": sc["focus_preview"],
        "zoom": sc["zoom"],
        "front": 1.0 if sc["front"] else 0.0,
    }


def scene_features(sc: dict, preview: np.ndarray) -> dict:
    feats = compute_preview_features(preview, sc["pv_exp"], sc["pv_iso"])
    feats.update(_sensor_features(sc))
    return {name: _r(feats.get(name), 6) for name in FEATURE_NAMES_V1}


def _user_choice(sc: dict, rng, mode: str) -> dict:
    err = rng.normal(0, 0.7) if mode != "ai" else rng.normal(0, 0.45)
    exp = sc["ideal_exp"] + err * 0.5 + rng.normal(0, 0.15)
    iso = float(np.clip(sc["ideal_iso"] + err * 0.5, math.log2(ISO_RANGE[0]), math.log2(ISO_RANGE[1])))
    exp_s = 2.0 ** exp
    frames = sc["frames"] if mode != "auto" else 1
    blur = int((not sc["tripod"]) and exp_s > 1 / 30 and rng.random() < 0.8)
    noisy = int((2 ** iso >= 800 and frames < 4 and rng.random() < 0.6) or rng.random() < 0.04)
    exposure = -1 if err < -0.75 else (1 if err > 0.75 else 0)
    rating = int(np.clip(round(8.5 - 1.5 * abs(err) - 2.5 * blur - 1.0 * noisy + rng.normal(0, 0.8)), 1, 10))
    return {
        "exp_s": exp_s,
        "iso": float(round(2.0 ** iso)),
        "frames": frames,
        "labels": {
            "rating": rating,
            "exposure": exposure,
            "sharpness": -1 if blur else (1 if rng.random() < 0.3 else 0),
            "noise": -1 if noisy else (1 if 2 ** iso < 200 and rng.random() < 0.5 else 0),
            "color": int(rng.choice([-1, 0, 0, 0, 0, 1])),
            "motion_blur": blur,
        },
    }


def _sample_id(dt: datetime, rng, used: set) -> str:
    while True:
        sid = f"{dt:%Y%m%d_%H%M%S}_{dt.microsecond // 1000:03d}_{int(rng.integers(0, 65536)):04x}"
        if sid not in used:
            used.add(sid)
            return sid


def _meta(sid: str, local: datetime, sc: dict, choice: dict, cat: dict, mode: str, features: dict,
          series: dict | None, rng, photo_role: str) -> dict:
    kind = sc["kind"]
    infinity = KINDS[kind]["focus_inf"] and not sc["front"]
    exp_ns = int(choice["exp_s"] * 1e9)
    utc_ms = int((local - timedelta(minutes=TZ_OFFSET_MIN)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    tz = timezone(timedelta(minutes=TZ_OFFSET_MIN))
    manual = mode in ("manual", "ai")
    focus_capture = 0.0 if infinity else round(float(rng.uniform(0.3, 5.0)), 3)
    if mode == "auto":
        user_settings = {"iso": None, "exposure_time_ns": None, "focus": "auto", "focus_distance_diopters": None,
                         "wb": "auto", "ev": 0.0, "frames": 1, "stack_mode": None, "timer_s": 0, "category_hint": None}
    else:
        user_settings = {"iso": int(choice["iso"]), "exposure_time_ns": exp_ns,
                         "focus": "infinity" if infinity else "auto",
                         "focus_distance_diopters": 0.0 if infinity else None, "wb": "auto", "ev": 0.0,
                         "frames": choice["frames"], "stack_mode": series["stack_mode"] if series else None,
                         "timer_s": 2 if sc["tripod"] else 0, "category_hint": cat["id"] if mode == "ai" else None}
    exp_ms = choice["exp_s"] * 1000
    gyro_rms = sc["motion_rms"]
    ai = None
    if mode == "ai":
        p = float(rng.uniform(0.5, 0.95))
        ai = {"model_name": "synthetic", "model_sha256": None, "probs": {cat["id"]: round(p, 3)},
              "chosen_class": cat["id"], "user_override": False,
              "params": {"log2_exposure_s": round(math.log2(choice["exp_s"]), 3),
                         "log2_iso": round(math.log2(choice["iso"]), 3),
                         "log2_frames": round(math.log2(max(1, choice["frames"])), 3),
                         "focus_infinity": 4.0 if infinity else -4.0, "ev_bias": 0.0}}
    labels = dict(choice["labels"])
    labels.update({"category_id": cat["id"], "category_name": cat["name"], "category_icon": cat["icon"],
                   "note": SYNTH_NOTE, "rated_utc_ms": utc_ms + 60000, "complete": True})
    pv_w, pv_h = (PREVIEW_SHORT, PREVIEW_LONG) if sc["portrait"] else (PREVIEW_LONG, PREVIEW_SHORT)
    return {
        "schema": SAMPLE_SCHEMA,
        "synthetic": True,
        "sample_id": sid,
        "created_utc_ms": utc_ms,
        "tz_offset_min": TZ_OFFSET_MIN,
        "local_time": local.replace(tzinfo=tz).isoformat(timespec="milliseconds"),
        "app": {"version": "synthetic", "version_code": 0},
        "device": {"manufacturer": "SYNTHETIC", "model": "therealcam-trainer synth", "sdk_int": 34,
                   "camera_id": "1" if sc["front"] else "0", "lens_facing": "front" if sc["front"] else "back",
                   "hardware_level": "full", "manual_sensor": True, "sensor_orientation": 90,
                   "physical_size_mm": [6.4, 4.8], "pixel_array": [4000, 3000], "focal_lengths_mm": [4.7],
                   "apertures": [1.8], "iso_range": list(ISO_RANGE), "max_analog_iso": 800,
                   "exposure_range_ns": [10000, 30000000000], "min_focus_distance_diopters": 10.0},
        "mode": mode,
        "photo": {"role": photo_role, "width": 4000, "height": 3000, "orientation_deg": 90 if sc["portrait"] else 0,
                  "jpeg_quality": 95, "file_name": f"TRC_{sid[:15]}.jpg"},
        "capture": {"exposure_time_ns": exp_ns, "iso": int(choice["iso"]), "aperture": 1.8, "focal_length_mm": 4.7,
                    "focus_distance_diopters": focus_capture, "ae_mode": "off" if manual else "on",
                    "af_mode": "off" if infinity else "continuous_picture", "awb_mode": "auto",
                    "wb_gains": [1.9, 1.0, 1.0, 1.6], "ev_compensation": 0.0, "flash": "off", "ois": "on",
                    "zoom_ratio": sc["zoom"], "frame_duration_ns": max(exp_ns, 33333333),
                    "noise_reduction": "high_quality", "sensor_timestamp_ns": int(rng.integers(1, 10 ** 12))},
        "user_settings": user_settings,
        "series": series,
        "location": sc["location"],
        "orientation": {"pitch_deg": _r(sc["pitch"], 2), "roll_deg": _r(sc["roll"], 2),
                        "azimuth_deg": _r(sc["azimuth"], 2), "device_rotation_deg": 0},
        "motion": {"window_ms": [-1500, 1000], "exposure_ms": [0, round(exp_ms, 3)],
                   "gyro": [[float(t), _r(rng.normal(0, gyro_rms), 6), _r(rng.normal(0, gyro_rms), 6),
                             _r(rng.normal(0, gyro_rms), 6)] for t in (-1500, -1000, -500, 0, 500)],
                   "accel": [[float(t), _r(rng.normal(0, 0.1), 4), _r(9.75 + rng.normal(0, 0.05), 4),
                              _r(rng.normal(0.4, 0.1), 4)] for t in (-1500, -1000, -500, 0, 500)],
                   "stats": {"pre_gyro_rms": _r(gyro_rms, 6), "exposure_gyro_rms": _r(gyro_rms, 6),
                             "exposure_gyro_max": _r(3 * gyro_rms, 6),
                             "exposure_rotation_rad": _r(gyro_rms * choice["exp_s"], 6),
                             "est_blur_px": _r(gyro_rms * choice["exp_s"] * 3000, 3)}},
        "environment": {"lux": _r(sc["lux"], 3), "pressure_hpa": 1012.3, "battery_temp_c": 30.0,
                        "sun_alt_deg": _r(sc["sun_alt"], 2), "sun_az_deg": 200.0 if sc["sun_alt"] is not None else None,
                        "moon_alt_deg": _r(sc["moon_alt"], 2),
                        "moon_az_deg": 120.0 if sc["moon_alt"] is not None else None,
                        "moon_illum": _r(sc["moon_illum"], 3)},
        "preview": {"exposure_time_ns": int(sc["pv_exp"] * 1e9), "iso": int(sc["pv_iso"]), "aperture": 1.8,
                    "ae_state": "converged", "af_state": "passive_focused",
                    "focus_distance_diopters": _r(sc["focus_preview"], 3), "width": pv_w, "height": pv_h},
        "features": features,
        "ai": ai,
        "labels": labels,
    }


def _write_jpeg(path: Path, img: np.ndarray, quality: int, scale: int = 1) -> None:
    im = Image.fromarray(to_uint8(img))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.BILINEAR)
    im.save(path, format="JPEG", quality=quality)


def _write_sample(sdir: Path, meta: dict, preview: np.ndarray, final: np.ndarray, write_images: bool) -> None:
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    Image.fromarray(preview_to_thumb(preview)).save(sdir / "thumb64.png", format="PNG")
    if write_images:
        _write_jpeg(sdir / "preview.jpg", preview, 90)
        _write_jpeg(sdir / "image.jpg", final, 85, scale=2)


def generate_synthetic_dataset(out_dir: str | os.PathLike, n: int = 200, seed: int = 1,
                               write_images: bool = True) -> Path:
    """Пишет OUT_DIR/dataset/{dataset.json, categories.json, samples/<id>/...}. Возвращает путь к dataset/."""
    if n < 1:
        raise ValueError("n должно быть ≥ 1")
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    root = out / "dataset"
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (out / "SYNTHETIC_DATASET.txt").write_text(
        "Это СИНТЕТИЧЕСКИЙ датасет, сгенерированный командой `therealcam-trainer synth-dataset`.\n"
        "Он нужен только для тестов и демонстрации тренера. Не используйте его для настоящей модели:\n"
        "модель TheRealCam обучается только на ваших собственных снимках и оценках.\n",
        encoding="utf-8",
    )
    base_ms = 1767225600000  # 2026-01-01T00:00:00Z
    cats = {}
    for i, (kind, k) in enumerate(KINDS.items()):
        cats[kind] = {"id": f"cat_{int(rng.integers(0, 65536)):04x}", "name": k["name"], "icon": k["icon"],
                      "created_utc_ms": base_ms + i * 1000}
    kinds = list(KINDS)
    shares = np.array([KINDS[k]["share"] for k in kinds])
    used_ids: set = set()
    written: list[str] = []
    start = datetime(2026, 3, 1)
    while len(written) < n:
        kind = kinds[int(rng.choice(len(kinds), p=shares / shares.sum()))]
        day = start + timedelta(days=int(rng.integers(0, 240)))
        sc = _scene(kind, rng, day)
        preview = render_scene(kind, rng, sc["portrait"])
        mode_p = {"stars": [0.1, 0.6, 0.3], "moon": [0.1, 0.6, 0.3], "clouds": [0.5, 0.3, 0.2],
                  "other": [0.8, 0.1, 0.1]}[kind]
        mode = ["auto", "manual", "ai"][int(rng.choice(3, p=mode_p))]
        choice = _user_choice(sc, rng, mode)
        feats = scene_features(sc, preview)
        cat = cats[kind]
        local = sc["local"].replace(microsecond=int(rng.integers(0, 1000)) * 1000)
        if choice["frames"] > 1:
            profile = icon_profile(cat["icon"])
            series_id = "S" + _sample_id(local, rng, used_ids)
            rated = min(choice["frames"], int(rng.integers(2, 5)), n - len(written) - 1)
            frame_infos = []
            for idx in range(max(rated, 0)):
                t = local + timedelta(seconds=idx * max(choice["exp_s"], 0.05))
                sid = _sample_id(t, rng, used_ids)
                shift = rng.integers(-2, 3, 2)
                frame = np.roll(preview, tuple(int(s) for s in shift), axis=(0, 1))
                frame = np.clip(frame + rng.normal(0, 0.02, frame.shape), 0, 1).astype(np.float32)
                f_choice = dict(choice, labels=dict(choice["labels"], rating=int(np.clip(
                    choice["labels"]["rating"] + rng.integers(-2, 2), 1, 10))))
                series = {"series_id": series_id, "role": "frame", "index": idx, "count": choice["frames"],
                          "profile": profile, "stack_mode": STACK_MODES.get(profile), "frames": None}
                meta = _meta(sid, t, sc, f_choice, cat, mode, feats, series, rng, "series_frame")
                _write_sample(root / "samples" / sid, meta, frame, frame, write_images)
                written.append(sid)
                frame_infos.append({"index": idx, "exposure_time_ns": int(choice["exp_s"] * 1e9),
                                    "iso": int(choice["iso"]), "accepted": True, "reject_reason": None,
                                    "offset_px": [int(shift[1]), int(shift[0])], "motion_rms": _r(sc["motion_rms"], 5),
                                    "sharpness": 3.0, "mean_luma": _r(float(frame.mean()), 4)})
            t = local + timedelta(seconds=choice["frames"] * max(choice["exp_s"], 0.05) + 1)
            sid = _sample_id(t, rng, used_ids)
            series = {"series_id": series_id, "role": "result", "index": -1, "count": choice["frames"],
                      "profile": profile, "stack_mode": STACK_MODES.get(profile), "frames": frame_infos}
            meta = _meta(sid, t, sc, choice, cat, mode, feats, series, rng, "series_result")
            _write_sample(root / "samples" / sid, meta, preview, preview, write_images)
            written.append(sid)
        else:
            sid = _sample_id(local, rng, used_ids)
            meta = _meta(sid, local, sc, choice, cat, mode, feats, None, rng, "single")
            _write_sample(root / "samples" / sid, meta, preview, preview, write_images)
            written.append(sid)
    (root / "categories.json").write_text(json.dumps(
        {"schema": CATEGORIES_SCHEMA, "synthetic": True, "categories": list(cats.values())},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "dataset.json").write_text(json.dumps(
        {"schema": DATASET_SCHEMA, "app_version": "synthetic", "updated_utc_ms": base_ms + 10 ** 9,
         "complete_samples": len(written), "min_samples_for_training": MIN_SAMPLES, "synthetic": True,
         "samples": sorted(written)}, ensure_ascii=False, indent=1), encoding="utf-8")
    return root
