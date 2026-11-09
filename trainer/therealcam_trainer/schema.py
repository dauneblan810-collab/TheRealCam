"""Константы форматов TheRealCam (docs/MODEL_FORMAT.md §2, §4, §5; docs/DATASET_FORMAT.md)."""

FEATURE_SCHEMA = 1

# Порядок — как в таблице MODEL_FORMAT.md §4.
FEATURE_NAMES_V1 = [
    "scene_ev",
    "preview_exp_log2",
    "preview_iso_log2",
    "luma_mean",
    "luma_std",
    "luma_p05",
    "luma_p50",
    "luma_p95",
    "clip_hi",
    "clip_lo",
    "bright_frac",
    "sharpness",
    "point_sources",
    "r_g",
    "b_g",
    "saturation",
    "top_luma",
    "bottom_luma",
    "center_luma",
    "lux_log10",
    "pitch_deg",
    "roll_deg",
    "motion_log10",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "sun_alt_deg",
    "moon_alt_deg",
    "moon_illum",
    "focus_diopters",
    "zoom",
    "front",
]

PARAM_NAMES_V1 = ["log2_exposure_s", "log2_iso", "log2_frames", "focus_infinity", "ev_bias"]
FOCUS_PARAM = "focus_infinity"
REGRESSION_PARAMS = [p for p in PARAM_NAMES_V1 if p != FOCUS_PARAM]

# Профиль съёмки -> ключи значков (MODEL_FORMAT.md §5).
PROFILE_ICONS = {
    "astro": ["stars", "milky_way", "night_sky", "aurora"],
    "lunar": ["moon"],
    "sky": ["clouds", "sky", "sunset", "sun"],
    "night": ["city_night", "fireworks", "concert", "fire", "lightning"],
    "trails": ["light_trails", "star_trails"],
    "long": ["waterfall", "water"],
    "action": ["sport", "pet", "bird", "car", "kids"],
    "landscape": ["landscape", "mountains", "forest", "snow", "city", "architecture"],
    "macro": ["macro", "flower", "insect"],
    "portrait": ["portrait", "people", "selfie"],
    "indoor": ["interior", "food"],
    "document": ["document", "screen"],
    "default": ["other"],
}
DEFAULT_PROFILE = "default"
DEFAULT_ICON = "other"

# Ключ значка -> профиль. Неизвестные ключи относятся к "default".
ICON_PROFILES = {icon: profile for profile, icons in PROFILE_ICONS.items() for icon in icons}


def icon_profile(icon) -> str:
    """Профиль съёмки для ключа значка; неизвестный ключ -> "default"."""
    if isinstance(icon, str):
        return ICON_PROFILES.get(icon, DEFAULT_PROFILE)
    return DEFAULT_PROFILE


MIN_SAMPLES = 150

SAMPLE_SCHEMA = "therealcam.sample/1"
CATEGORIES_SCHEMA = "therealcam.categories/1"
DATASET_SCHEMA = "therealcam.dataset/1"

THUMB_SIZE = 64
IMAGE_CHANNELS = 3

LABEL_TERNARY_FIELDS = ("exposure", "sharpness", "noise", "color")
