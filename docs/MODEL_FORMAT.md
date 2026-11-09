# Формат модели `.trcm` v1 и признаки сцены (feature schema 1)

Модель — маленький вычислительный граф с весами float32. Приложение выполняет его на чистом
Kotlin (`android/app/src/main/java/com/therealcam/ai/engine`), тренер — PyTorch. Эталонный
интерпретатор на NumPy лежит в `trainer/` и используется для проверки совпадения.

## 1. Бинарная раскладка

| смещение | размер | содержимое |
|---|---|---|
| 0 | 4 | ASCII `TRCM` |
| 4 | 4 | `uint32 LE` — версия формата = `1` |
| 8 | 4 | `uint32 LE` — длина заголовка `H` в байтах |
| 12 | H | заголовок, JSON в UTF-8 |
| 12+H | 0–3 | нулевые байты до кратности 4 от начала файла |
| … | 4·N | данные: `float32 LE`, N = `header.dataFloats` |

Тензор `t` занимает `data[t.offset : t.offset + prod(t.shape)]` (смещение в элементах float).
`header.dataCrc32` — CRC-32 (IEEE, как `zlib.crc32`) байтов секции данных. Если он не совпадает,
файл отклоняется. Ограничения загрузчика: заголовок ≤ 1 МиБ, данные ≤ 64 МиБ, узлов ≤ 256.

## 2. Заголовок

```jsonc
{
  "format": "trcm", "formatVersion": 1,
  "name": "my-model-v3", "description": "", "author": "",
  "createdUtc": "2026-11-09T12:00:00Z",
  "trainer": { "name": "therealcam-trainer", "version": "0.1.0" },
  "training": { "samples": 2134, "epochs": 60, "baseModel": null, "metrics": { "val_acc": 0.91 } },
  "featureSchema": 1,

  "inputs": {
    "image": { "width": 64, "height": 64, "channels": 3, "layout": "CHW",
               "norm": { "type": "per_image" } },        // или {"type":"channel","mean":[3],"std":[3]}
    "features": [ { "name": "scene_ev", "mean": 6.1, "std": 4.2 }, ... ]   // порядок = порядок во входе
  },

  "classes": [ { "id": "cat_7f3a", "name": "Звёзды", "icon": "stars" }, ... ],   // K классов
  "params":  [ { "name": "log2_exposure_s", "mean": -4.0, "std": 3.0 }, ... ],   // P параметров

  "outputs": { "class_logits": "logits", "params": "params" },    // имена узлов

  "nodes": [
    { "op": "conv2d", "name": "c1", "inputs": ["image"],
      "attrs": { "inChannels": 3, "outChannels": 16, "kernel": 3, "stride": 1, "padding": 1, "groups": 1 },
      "weights": ["c1.weight", "c1.bias"] },
    { "op": "relu", "name": "r1", "inputs": ["c1"] },
    ...
  ],
  "tensors": [ { "name": "c1.weight", "shape": [16, 3, 3, 3], "offset": 0 }, ... ],
  "dataFloats": 51234,
  "dataCrc32": 2882343476
}
```

### Входы

* `image` — тензор `[3, 64, 64]` (CHW, RGB) из `thumb64.png`: `x = v / 255`.
  * `norm.type = "per_image"`: `m = mean(x)`, `s = max(std(x), 1/sqrt(3·64·64))` по всем 12288
    значениям (std популяционное, деление на N); `x' = (x − m) / s`.
  * `norm.type = "channel"`: `x'[c] = (x[c] − mean[c]) / std[c]`.
* `features` — вектор длины F в порядке `inputs.features`. Для каждого признака
  `z = (x − mean) / std`, где `std ≤ 1e-6` заменяется на 1. **Если `x` отсутствует или не конечен
  (NaN/∞), `z = 0`.** Затем `z` ограничивается диапазоном `[−6, 6]`. Признаки, которых модель
  не перечислила, не используются.

### Выходы

* `class_logits` — `[K]`. Вероятности: softmax.
* `params` — `[K·P]`, построчно `[k][p]`: параметры для класса `k`. Реальное значение:
  `value = out · std + mean` (из `header.params`).

### Параметры v1 (P = 5, порядок задаётся заголовком, приложение ищет по имени)

| имя | смысл | как применяет приложение |
|---|---|---|
| `log2_exposure_s` | log2 выдержки одного кадра, сек | `2^v`, обрезается диапазоном сенсора |
| `log2_iso` | log2 ISO | `2^v`, обрезается диапазоном сенсора |
| `log2_frames` | log2 числа кадров серии | `round(2^v)`, 1…64 (лимит профиля) |
| `focus_infinity` | логит «фокус на ∞» | `sigmoid(v) ≥ 0.5` → фокус 0 диоптрий, иначе AF |
| `ev_bias` | компенсация экспозиции, EV | используется, если ручная экспозиция недоступна |

Если модель не содержит какого-то параметра, приложение использует авто-режим для этого параметра.
Если камера не поддерживает `MANUAL_SENSOR`, применяются только `ev_bias`, `focus_infinity` и `log2_frames`.

## 3. Операции графа

Тензоры бывают `[C,H,W]` (батч = 1) или вектор `[N]`. Узлы выполняются в порядке списка. Вход узла —
имя другого узла либо `image` / `features`.

| op | attrs | weights | семантика (как в PyTorch) |
|---|---|---|---|
| `conv2d` | `inChannels, outChannels, kernel, stride=1, padding=0, groups=1` | `W[out, in/groups, k, k]`, опц. `b[out]` | кросс-корреляция, нулевой паддинг |
| `dense` | `in, out` | `W[out, in]`, опц. `b[out]` | `y = W·x + b` (вход-тензор `[C,H,W]` сначала разворачивается) |
| `relu` | — | — | `max(0, x)` |
| `leaky_relu` | `alpha=0.01` | — | |
| `sigmoid`, `tanh` | — | — | |
| `maxpool2d`, `avgpool2d` | `kernel, stride=kernel` | — | без паддинга, `out = floor((H−k)/s)+1` |
| `global_avgpool` | — | — | `[C,H,W] → [C]` |
| `channel_affine` | — | `scale[C], bias[C]` | `y[c] = x[c]·scale[c] + bias[c]` (свёрнутый BatchNorm) |
| `batchnorm` | `eps=1e-5` | `gamma, beta, mean, var` | инференс-BN по каналам (для `[C,H,W]` и `[C]`) |
| `flatten` | — | — | `[C,H,W] → [C·H·W]` в порядке C, H, W |
| `concat` | — | — | склейка векторов (`[C,H,W]` разворачиваются) в порядке `inputs` |
| `add` | — | — | поэлементная сумма двух тензоров одной формы |
| `dropout`, `identity` | — | — | тождество |

Неизвестная операция — ошибка загрузки. Такую модель приложение отклоняет с понятным сообщением.

## 4. Признаки сцены (feature schema 1)

Признаки считает `FeatureExtractor` в приложении в момент нажатия спуска (в режиме AI — каждые ~0.5 с)
и сохраняет в `meta.json → features`. **Тренер берёт значения из `meta.json`, сам их не пересчитывает.**
Поэтому реализация признаков одна, и между приложением и тренером нет расхождений.

Превью `P` — повёрнутое кадр-превью 192×256 (портрет) или 256×192 (альбом), sRGB в `[0,1]`.
`Y = 0.2126R + 0.7152G + 0.0722B` (на sRGB-значениях).

| имя | определение |
|---|---|
| `scene_ev` | `log2(N²/t) − log2(S/100) + log2(max(mean(Y^2.2), 1e-4) / 0.18)`; t, S, N — выдержка (с), ISO, диафрагма превью (N=1.8, если неизвестна) |
| `preview_exp_log2` | `log2(t)` выдержки превью |
| `preview_iso_log2` | `log2(S)` ISO превью |
| `luma_mean`, `luma_std` | среднее и СКО `Y` |
| `luma_p05`, `luma_p50`, `luma_p95` | перцентили `Y` по гистограмме из 256 корзин (`floor(Y·255)`): наименьшая корзина `b` с накопленной долей ≥ p, значение `b/255` |
| `clip_hi`, `clip_lo` | доля `Y ≥ 250/255` и `Y ≤ 5/255` |
| `bright_frac` | доля `Y ≥ 0.7` |
| `sharpness` | `log10(1e-6 + var(L))`, `L = 4Y(x,y) − Y(x±1,y) − Y(x,y±1)` по внутренним пикселям |
| `point_sources` | число внутренних пикселей, строго ярче всех 8 соседей, с `Y ≥ p50 + 0.12` и `Y ≥ 0.2`, на 10 000 пикселей (звёзды, огни) |
| `r_g`, `b_g` | `mean(R)/(mean(G)+1e-4)`, `mean(B)/(mean(G)+1e-4)` |
| `saturation` | `mean((max−min)/(max+1e-4))` по RGB |
| `top_luma`, `bottom_luma` | среднее `Y` верхней и нижней трети строк |
| `center_luma` | среднее `Y` центрального блока 1/3 × 1/3 |
| `lux_log10` | `log10(1 + lux)` датчика освещённости |
| `pitch_deg` | угол оси камеры над горизонтом, −90…90 (вверх — положительный) |
| `roll_deg` | крен вокруг оси камеры, −180…180 (0 — телефон вертикально) |
| `motion_log10` | `log10(1e-4 + RMS|ω|)` гироскопа за последнюю 1 с, рад/с |
| `hour_sin`, `hour_cos` | `sin/cos(2π·h/24)`, h — местное время в часах с дробной частью |
| `doy_sin`, `doy_cos` | `sin/cos(2π·(день_года − 1)/365.25)` |
| `sun_alt_deg` | высота Солнца над горизонтом (нужна геопозиция, иначе отсутствует) |
| `moon_alt_deg` | высота Луны (нужна геопозиция) |
| `moon_illum` | освещённая доля диска Луны, 0…1 |
| `focus_diopters` | текущая дистанция фокуса превью, диоптрии (0 = ∞) |
| `zoom` | коэффициент зума |
| `front` | 1 — фронтальная камера, 0 — основная |

Отсутствующий признак записывается в `meta.json` как `null`.

## 5. Значки и профили

Ключ значка хранится в категории и в `classes[].icon`. Профиль определяет способ сведения серии
(см. `docs/CONCEPT.md`).

| профиль | ключи значков |
|---|---|
| `astro` | `stars`, `milky_way`, `night_sky`, `aurora` |
| `lunar` | `moon` |
| `sky` | `clouds`, `sky`, `sunset`, `sun` |
| `night` | `city_night`, `fireworks`, `concert`, `fire`, `lightning` |
| `trails` | `light_trails`, `star_trails` |
| `long` | `waterfall`, `water` |
| `action` | `sport`, `pet`, `bird`, `car`, `kids` |
| `landscape` | `landscape`, `mountains`, `forest`, `snow`, `city`, `architecture` |
| `macro` | `macro`, `flower`, `insect` |
| `portrait` | `portrait`, `people`, `selfie` |
| `indoor` | `interior`, `food` |
| `document` | `document`, `screen` |
| `default` | `other`, а также любой неизвестный ключ |

## 6. Эталонные векторы (`testdata/golden/`)

* `model.trcm` — небольшая модель, экспортированная тренером, с ≥ 3 классами и всеми операциями,
  которые использует архитектура по умолчанию;
* `golden.json`:

```jsonc
{
  "model": "model.trcm",
  "cases": [
    { "thumb_rgb_hwc": [/* 64·64·3 uint8, порядок H, W, C */],
      "features": { "scene_ev": 3.2, "lux_log10": null, ... },   // СЫРЫЕ значения, null = отсутствует
      "expected_class_logits": [ ... ], "expected_params": [ ... ] }
  ],
  "atol": 1e-4, "rtol": 1e-4
}
```

Тест приложения (`TrcmGoldenTest`) загружает их и сверяет свой Kotlin-инференс с ожидаемыми значениями.
