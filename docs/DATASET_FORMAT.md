# Формат датасета TheRealCam (schema `therealcam.sample/1`)

Корень на телефоне (папка приложения, разрешения не нужны):

```
Android/data/com.therealcam/files/TheRealCam/
├── dataset/                 ← ГОТОВЫЙ датасет: только полностью оценённые снимки
│   ├── dataset.json         ← манифест (см. ниже)
│   ├── categories.json      ← категории пользователя
│   └── samples/<sample_id>/
│       ├── meta.json        ← все данные снимка
│       ├── image.jpg        ← итоговый снимок, повернут «как надо», длинная сторона ≤ настройки (по умолч. 1600 px)
│       ├── preview.jpg      ← превью сцены в момент нажатия спуска, ровно 192×256 или 256×192, повернуто
│       └── thumb64.png      ← ТОЧНЫЙ вход модели: 64×64 RGB (центр-кроп квадрата превью + box-фильтр 3×3)
├── pending/<sample_id>/     ← то же самое, но оценка ещё неполная (в обучение НЕ идёт)
├── series/<series_id>/      ← полноразмерные кадры серий (для галереи и сведения)
├── models/                  ← импортированные модели *.trcm
└── exports/                 ← временные архивы экспорта
```

Из настроек можно сделать **экспорт в ZIP** (выбор места через системный диалог). Архив содержит
`dataset/` целиком. Тренер принимает и папку `dataset/`, и такой ZIP.

`sample_id` имеет вид `yyyyMMdd_HHmmss_SSS_xxxx` (локальное время + 4 hex). Он сортируется по времени и уникален.

## Правило полноты

Снимок попадает в `dataset/` (и в обучение) только если в `labels` заполнены **все** поля:

| поле | значения | смысл |
|---|---|---|
| `rating` | 1…10 | общая оценка |
| `exposure` | −1 / 0 / +1 | темно / норм / пересвет |
| `sharpness` | −1 / 0 / +1 | нерезко / норм / отлично |
| `noise` | −1 / 0 / +1 | шумно / терпимо / чисто |
| `color` | −1 / 0 / +1 | холодно / естественно / тепло |
| `motion_blur` | 0 / 1 | смаза нет / есть смаз |
| `category_id` | id категории | что снималось |

Тогда `labels.complete = true`. Тренер **обязан** игнорировать образцы с `complete != true`.

## `meta.json`

Неизвестные поля нужно игнорировать. Любое поле может быть `null`, если устройство его не дало.
Единицы указаны в именах полей. Оси датчиков такие же, как у Android `SensorManager`:
x вправо, y вверх, z из экрана, в естественной ориентации устройства.

```jsonc
{
  "schema": "therealcam.sample/1",
  "sample_id": "20261109_172233_512_a1b2",
  "created_utc_ms": 1762698153512,
  "tz_offset_min": 180,
  "local_time": "2026-11-09T17:22:33.512+03:00",
  "app": { "version": "0.1.0", "version_code": 1 },

  "device": {
    "manufacturer": "Xiaomi", "model": "Redmi 9", "sdk_int": 30,
    "camera_id": "0", "lens_facing": "back", "hardware_level": "limited",
    "manual_sensor": true, "sensor_orientation": 90,
    "physical_size_mm": [4.6, 3.4], "pixel_array": [4000, 3000],
    "focal_lengths_mm": [3.9], "apertures": [1.8],
    "iso_range": [100, 3200], "max_analog_iso": 800,
    "exposure_range_ns": [10000, 500000000], "min_focus_distance_diopters": 10.0
  },

  "mode": "manual",                       // auto | manual | ai

  "photo": {
    "role": "single",                     // single | series_frame | series_result
    "width": 4000, "height": 3000,        // как записано в JPEG (до поворота по EXIF)
    "orientation_deg": 90,                // EXIF-поворот
    "jpeg_quality": 95,
    "file_name": "TRC_20261109_172233.jpg"
  },

  "capture": {                            // фактические значения из CaptureResult
    "exposure_time_ns": 250000000, "iso": 800, "aperture": 1.8, "focal_length_mm": 3.9,
    "focus_distance_diopters": 0.0,       // 0 = бесконечность
    "ae_mode": "off", "af_mode": "off", "awb_mode": "auto",
    "wb_gains": [1.9, 1.0, 1.0, 1.6],     // R, Geven, Godd, B
    "ev_compensation": 0.0, "flash": "off", "ois": "off",
    "zoom_ratio": 1.0, "frame_duration_ns": 250000000,
    "noise_reduction": "high_quality", "sensor_timestamp_ns": 123456789
  },

  "user_settings": {                      // что выставил пользователь (ручной режим) или план AI
    "iso": 800, "exposure_time_ns": 250000000,
    "focus": "infinity",                  // auto | infinity | manual
    "focus_distance_diopters": 0.0,
    "wb": "auto", "ev": 0.0,
    "frames": 12, "stack_mode": "sigma_clip", "timer_s": 3,
    "category_hint": "cat_7f3a"           // категория, выбранная ДО снимка (может быть null)
  },

  "series": {                             // null для одиночного снимка
    "series_id": "S20261109_172233_512_a1b2",
    "role": "frame",                      // frame | result
    "index": 3,                           // номер кадра; у result = -1
    "count": 12, "profile": "astro", "stack_mode": "sigma_clip",
    "frames": [                           // только у result: что произошло с каждым кадром
      { "index": 0, "exposure_time_ns": 4000000000, "iso": 1600, "accepted": true,
        "reject_reason": null, "offset_px": [0, 0], "motion_rms": 0.002, "sharpness": 3.1, "mean_luma": 0.08 }
    ]
  },

  "location": { "lat": 55.75, "lon": 37.61, "alt_m": 150.0, "accuracy_m": 8.0,
                "provider": "gps", "age_ms": 1200, "speed_mps": 0.0, "bearing_deg": null },

  "orientation": { "pitch_deg": 35.2, "roll_deg": -1.5, "azimuth_deg": 180.3, "device_rotation_deg": 0 },

  "motion": {
    "window_ms": [-1500, 1000],           // окно записи относительно НАЧАЛА экспозиции
    "exposure_ms": [0, 250],              // экспозиция в том же времени
    "gyro":  [[-1500.0, 0.001, -0.002, 0.0005], ...],   // [t_ms, x, y, z] рад/с
    "accel": [[-1500.0, 0.12, 9.75, 0.40], ...],        // [t_ms, x, y, z] м/с²
    "stats": {
      "pre_gyro_rms": 0.010,              // рад/с за 1 с до экспозиции
      "exposure_gyro_rms": 0.005, "exposure_gyro_max": 0.020,
      "exposure_rotation_rad": 0.0012,    // интеграл |ω| за экспозицию
      "est_blur_px": 2.4                  // оценка смаза в пикселях полного кадра
    }
  },

  "environment": {
    "lux": 3.5, "pressure_hpa": 1012.3, "battery_temp_c": 31.2,
    "sun_alt_deg": -35.1, "sun_az_deg": 270.2,
    "moon_alt_deg": 20.1, "moon_az_deg": 120.0, "moon_illum": 0.83
  },

  "preview": {                            // состояние превью в момент нажатия спуска
    "exposure_time_ns": 66000000, "iso": 3200, "aperture": 1.8,
    "ae_state": "converged", "af_state": "passive_focused",
    "focus_distance_diopters": 0.5, "width": 192, "height": 256
  },

  "features": { "scene_ev": 2.1, "luma_mean": 0.07, "...": 0.0 },  // признаки модели, см. MODEL_FORMAT.md

  "ai": {                                 // null, если снимок не в режиме AI
    "model_name": "my-model-v3", "model_sha256": "…",
    "probs": { "cat_7f3a": 0.81, "cat_19bc": 0.12 },
    "chosen_class": "cat_7f3a", "user_override": false,
    "params": { "log2_exposure_s": 2.0, "log2_iso": 10.6, "log2_frames": 3.6, "focus_infinity": 4.2, "ev_bias": 0.0 }
  },

  "labels": {
    "rating": 8, "exposure": 0, "sharpness": 1, "noise": 0, "color": 0, "motion_blur": 0,
    "category_id": "cat_7f3a", "category_name": "Звёзды", "category_icon": "stars",
    "note": "", "rated_utc_ms": 1762698200000, "complete": true
  }
}
```

### Серии

Каждый кадр серии — **отдельный образец** (`series.role = "frame"`), его можно оценить отдельно.
Сведённый итог — тоже образец (`series.role = "result"`, `index = -1`). Сенсорные данные кадра
относятся к его собственной экспозиции. Для обучения параметров важнее всего одиночные снимки
и итоги серий. Кадры одной серии описывают одну сцену, поэтому тренер должен учитывать
их как группу (не давать серии перевешивать датасет и не разносить кадры одной серии по train/val).

## `categories.json`

```json
{ "schema": "therealcam.categories/1",
  "categories": [ { "id": "cat_7f3a", "name": "Звёзды", "icon": "stars", "created_utc_ms": 1762698000000 } ] }
```

`id` неизменен. Название и значок можно менять, связь с образцами при этом не теряется.
Список ключей значков и их профили съёмки перечислены в [`MODEL_FORMAT.md`](MODEL_FORMAT.md#значки-и-профили).

## `dataset.json`

```json
{ "schema": "therealcam.dataset/1", "app_version": "0.1.0", "updated_utc_ms": 1762698200000,
  "complete_samples": 152, "min_samples_for_training": 150,
  "samples": ["20261109_172233_512_a1b2", "..."] }
```

Манифест пересобирается при каждом изменении. Источник истины — папки `samples/*/meta.json`:
тренер сканирует их сам и на манифест не полагается.
