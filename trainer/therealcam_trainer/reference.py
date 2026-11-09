"""Эталонный NumPy-интерпретатор графов `.trcm` v1 (docs/MODEL_FORMAT.md §2–3).

Считает в float64 и выполняет узлы строго в порядке списка — так же, как Kotlin-движок
приложения. Используется для проверки экспорта, команды ``eval`` и golden-векторов.
"""

from __future__ import annotations

import math
import numbers
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .errors import TrcmError
from .schema import FEATURE_SCHEMA, IMAGE_CHANNELS, THUMB_SIZE
from .trcm import TrcmFile, decode_trcm, read_trcm

IMAGE_SHAPE = (IMAGE_CHANNELS, THUMB_SIZE, THUMB_SIZE)
IMAGE_VALUES = IMAGE_CHANNELS * THUMB_SIZE * THUMB_SIZE  # 12288
PER_IMAGE_STD_FLOOR = 1.0 / math.sqrt(IMAGE_VALUES)
FEATURE_STD_MIN = 1e-6
FEATURE_CLIP = 6.0
INPUT_NAMES = ("image", "features")

SUPPORTED_OPS = (
    "conv2d",
    "dense",
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "maxpool2d",
    "avgpool2d",
    "global_avgpool",
    "channel_affine",
    "batchnorm",
    "flatten",
    "concat",
    "add",
    "dropout",
    "identity",
)

_REQUIRED = object()


# ---------------------------------------------------------------------------
# Подготовка входов


def raw_feature_value(value: Any) -> float | None:
    """Сырое значение признака из meta.json -> float или None (отсутствует / не конечно)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, numbers.Real):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def effective_std(std: float) -> float:
    """``std <= 1e-6`` (и NaN) заменяется на 1."""
    return float(std) if float(std) > FEATURE_STD_MIN else 1.0


def normalize_features(raw: Mapping[str, Any] | None, specs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Вектор признаков длины F: ``z=(x-mean)/std``, отсутствующие -> 0, клип в [-6, 6]."""
    z = np.zeros(len(specs), dtype=np.float64)
    if not raw:
        return z
    for i, spec in enumerate(specs):
        v = raw_feature_value(raw.get(spec["name"]))
        if v is None:
            continue
        val = (v - float(spec["mean"])) / effective_std(spec["std"])
        z[i] = min(max(val, -FEATURE_CLIP), FEATURE_CLIP)
    return z


def thumb_to_chw(thumb: Any) -> np.ndarray:
    """uint8 HWC 64×64×3 (или плоский список длины 12288 в порядке H,W,C) -> CHW float64 ``v/255``."""
    a = np.asarray(thumb)
    if a.ndim == 1 and a.size == IMAGE_VALUES:
        a = a.reshape(THUMB_SIZE, THUMB_SIZE, IMAGE_CHANNELS)
    if a.shape != (THUMB_SIZE, THUMB_SIZE, IMAGE_CHANNELS):
        raise ValueError(f"миниатюра должна быть 64×64×3 (HWC), получено {a.shape}")
    if a.dtype != np.uint8:
        if np.any((a < 0) | (a > 255)) or np.any(np.asarray(a) != np.round(a)):
            raise ValueError("значения миниатюры должны быть целыми 0..255")
        a = a.astype(np.uint8)
    return a.transpose(2, 0, 1).astype(np.float64) / 255.0


def normalize_image(x: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    """Нормализация входа image: ``per_image`` или ``channel``."""
    kind = norm.get("type", "per_image")
    if kind == "per_image":
        m = float(x.mean())
        s = max(float(x.std()), PER_IMAGE_STD_FLOOR)  # популяционное СКО (деление на N)
        return (x - m) / s
    if kind == "channel":
        mean = np.asarray(norm["mean"], dtype=np.float64).reshape(-1, 1, 1)
        std = np.asarray(norm["std"], dtype=np.float64).reshape(-1, 1, 1)
        return (x - mean) / std
    raise TrcmError(f"неизвестный тип нормализации изображения: {kind!r}")


# ---------------------------------------------------------------------------
# Операции (семантика как в PyTorch, батч = 1)


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, padding: int = 0,
           groups: int = 1) -> np.ndarray:
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    c, h, wd = x.shape
    o, cg, k, _ = w.shape
    ho = (h - k) // stride + 1
    wo = (wd - k) // stride + 1
    win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::stride, ::stride][:, :ho, :wo]
    win = win.reshape(groups, cg, ho, wo, k, k)
    wg = w.reshape(groups, o // groups, cg, k, k)
    y = np.einsum("gchwij,gocij->gohw", win, wg, optimize=True).reshape(o, ho, wo)
    if b is not None:
        y = y + b[:, None, None]
    return y


def dense(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = w @ x.reshape(-1)
    return y + b if b is not None else y


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def pool2d(x: np.ndarray, kernel: int, stride: int, mode: str) -> np.ndarray:
    win = sliding_window_view(x, (kernel, kernel), axis=(1, 2))[:, ::stride, ::stride]
    return win.max(axis=(3, 4)) if mode == "max" else win.mean(axis=(3, 4))


def _per_channel(v: np.ndarray, ndim: int) -> np.ndarray:
    return v.reshape(-1, 1, 1) if ndim == 3 else v


def channel_affine(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x * _per_channel(scale, x.ndim) + _per_channel(bias, x.ndim)


def batchnorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    inv = gamma / np.sqrt(var + eps)
    return (x - _per_channel(mean, x.ndim)) * _per_channel(inv, x.ndim) + _per_channel(beta, x.ndim)


# ---------------------------------------------------------------------------
# Компиляция и проверка графа


@dataclass
class CompiledNode:
    index: int
    op: str
    name: str
    inputs: tuple[str, ...]
    shape: tuple[int, ...]
    run: Callable[[list[np.ndarray]], np.ndarray]


class _Ctx:
    def __init__(self, index: int, name: str, op: str, attrs: Mapping[str, Any]):
        self.index, self.name, self.op, self.attrs = index, name, op, attrs

    def error(self, msg: str) -> TrcmError:
        return TrcmError(f"узел #{self.index} '{self.name}' ({self.op}): {msg}")

    def int_attr(self, key: str, default: Any = _REQUIRED, minimum: int = 0) -> int:
        v = self.attrs.get(key, default)
        if v is _REQUIRED:
            raise self.error(f"нет обязательного атрибута '{key}'")
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
            raise self.error(f"атрибут '{key}' должен быть целым числом ≥ {minimum}, получено {v!r}")
        return v

    def float_attr(self, key: str, default: float) -> float:
        v = self.attrs.get(key, default)
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise self.error(f"атрибут '{key}' должен быть конечным числом, получено {v!r}")
        return float(v)


def _expect_inputs(ctx: _Ctx, shapes: list[tuple[int, ...]], count: int | None, minimum: int = 1) -> None:
    if count is not None and len(shapes) != count:
        raise ctx.error(f"ожидается входов: {count}, получено {len(shapes)}")
    if len(shapes) < minimum:
        raise ctx.error(f"нужен хотя бы {minimum} вход")


def _expect_3d(ctx: _Ctx, shape: tuple[int, ...]) -> None:
    if len(shape) != 3:
        raise ctx.error(f"вход должен быть тензором [C,H,W], получено {list(shape)}")


def _expect_weights(ctx: _Ctx, weights: list[np.ndarray], allowed: tuple[int, ...]) -> None:
    if len(weights) not in allowed:
        raise ctx.error(f"ожидается весов: {' или '.join(map(str, allowed))}, получено {len(weights)}")


def _expect_shape(ctx: _Ctx, arr: np.ndarray, shape: tuple[int, ...], what: str) -> None:
    if tuple(arr.shape) != tuple(shape):
        raise ctx.error(f"{what}: форма {list(arr.shape)}, ожидается {list(shape)}")


def _compile_node(ctx: _Ctx, in_shapes: list[tuple[int, ...]], weights: list[np.ndarray]):
    op = ctx.op
    if op == "conv2d":
        _expect_inputs(ctx, in_shapes, 1)
        s = in_shapes[0]
        _expect_3d(ctx, s)
        cin = ctx.int_attr("inChannels", minimum=1)
        cout = ctx.int_attr("outChannels", minimum=1)
        k = ctx.int_attr("kernel", minimum=1)
        stride = ctx.int_attr("stride", 1, minimum=1)
        pad = ctx.int_attr("padding", 0, minimum=0)
        groups = ctx.int_attr("groups", 1, minimum=1)
        if s[0] != cin:
            raise ctx.error(f"inChannels={cin}, а у входа {s[0]} каналов")
        if cin % groups or cout % groups:
            raise ctx.error(f"inChannels и outChannels должны делиться на groups={groups}")
        _expect_weights(ctx, weights, (1, 2))
        _expect_shape(ctx, weights[0], (cout, cin // groups, k, k), "веса W")
        bias = weights[1] if len(weights) == 2 else None
        if bias is not None:
            _expect_shape(ctx, bias, (cout,), "смещение b")
        if s[1] + 2 * pad < k or s[2] + 2 * pad < k:
            raise ctx.error(f"ядро {k} больше входа {list(s)} с паддингом {pad}")
        ho = (s[1] + 2 * pad - k) // stride + 1
        wo = (s[2] + 2 * pad - k) // stride + 1
        w = weights[0]
        return (cout, ho, wo), lambda xs: conv2d(xs[0], w, bias, stride, pad, groups)

    if op == "dense":
        _expect_inputs(ctx, in_shapes, 1)
        n_in = ctx.int_attr("in", minimum=1)
        n_out = ctx.int_attr("out", minimum=1)
        if math.prod(in_shapes[0]) != n_in:
            raise ctx.error(f"in={n_in}, а вход содержит {math.prod(in_shapes[0])} значений")
        _expect_weights(ctx, weights, (1, 2))
        _expect_shape(ctx, weights[0], (n_out, n_in), "веса W")
        bias = weights[1] if len(weights) == 2 else None
        if bias is not None:
            _expect_shape(ctx, bias, (n_out,), "смещение b")
        w = weights[0]
        return (n_out,), lambda xs: dense(xs[0], w, bias)

    if op in ("relu", "sigmoid", "tanh", "dropout", "identity", "leaky_relu"):
        _expect_inputs(ctx, in_shapes, 1)
        _expect_weights(ctx, weights, (0,))
        fn: Callable[[np.ndarray], np.ndarray]
        if op == "relu":
            fn = relu
        elif op == "sigmoid":
            fn = sigmoid
        elif op == "tanh":
            fn = np.tanh
        elif op == "leaky_relu":
            alpha = ctx.float_attr("alpha", 0.01)
            return in_shapes[0], lambda xs: leaky_relu(xs[0], alpha)
        else:
            return in_shapes[0], lambda xs: xs[0]
        return in_shapes[0], lambda xs: fn(xs[0])

    if op in ("maxpool2d", "avgpool2d"):
        _expect_inputs(ctx, in_shapes, 1)
        _expect_weights(ctx, weights, (0,))
        s = in_shapes[0]
        _expect_3d(ctx, s)
        k = ctx.int_attr("kernel", minimum=1)
        stride = ctx.int_attr("stride", k, minimum=1)
        if s[1] < k or s[2] < k:
            raise ctx.error(f"ядро {k} больше входа {list(s)}")
        mode = "max" if op == "maxpool2d" else "avg"
        shape = (s[0], (s[1] - k) // stride + 1, (s[2] - k) // stride + 1)
        return shape, lambda xs: pool2d(xs[0], k, stride, mode)

    if op == "global_avgpool":
        _expect_inputs(ctx, in_shapes, 1)
        _expect_weights(ctx, weights, (0,))
        _expect_3d(ctx, in_shapes[0])
        return (in_shapes[0][0],), lambda xs: xs[0].mean(axis=(1, 2))

    if op in ("channel_affine", "batchnorm"):
        _expect_inputs(ctx, in_shapes, 1)
        s = in_shapes[0]
        if len(s) not in (1, 3):
            raise ctx.error(f"вход должен быть [C,H,W] или [C], получено {list(s)}")
        c = s[0]
        if op == "channel_affine":
            _expect_weights(ctx, weights, (2,))
            for arr, what in zip(weights, ("scale", "bias")):
                _expect_shape(ctx, arr, (c,), what)
            scale, bias = weights
            return s, lambda xs: channel_affine(xs[0], scale, bias)
        eps = ctx.float_attr("eps", 1e-5)
        _expect_weights(ctx, weights, (4,))
        for arr, what in zip(weights, ("gamma", "beta", "mean", "var")):
            _expect_shape(ctx, arr, (c,), what)
        gamma, beta, mean, var = weights
        if np.any(var + eps <= 0):
            raise ctx.error("var + eps должно быть > 0")
        return s, lambda xs: batchnorm(xs[0], gamma, beta, mean, var, eps)

    if op == "flatten":
        _expect_inputs(ctx, in_shapes, 1)
        _expect_weights(ctx, weights, (0,))
        return (math.prod(in_shapes[0]),), lambda xs: xs[0].reshape(-1)

    if op == "concat":
        _expect_inputs(ctx, in_shapes, None, minimum=1)
        _expect_weights(ctx, weights, (0,))
        total = sum(math.prod(s) for s in in_shapes)
        return (total,), lambda xs: np.concatenate([a.reshape(-1) for a in xs])

    if op == "add":
        _expect_inputs(ctx, in_shapes, 2)
        _expect_weights(ctx, weights, (0,))
        if in_shapes[0] != in_shapes[1]:
            raise ctx.error(f"формы слагаемых различаются: {list(in_shapes[0])} и {list(in_shapes[1])}")
        return in_shapes[0], lambda xs: xs[0] + xs[1]

    raise ctx.error("неизвестная операция")  # pragma: no cover


def compile_graph(nodes: Any, tensors: Mapping[str, np.ndarray],
                  input_shapes: Mapping[str, tuple[int, ...]]) -> tuple[list[CompiledNode], dict[str, tuple[int, ...]]]:
    """Проверяет граф (операции, порядок, формы, веса) и готовит его к исполнению."""
    if not isinstance(nodes, list):
        raise TrcmError("nodes должен быть списком")
    shapes: dict[str, tuple[int, ...]] = dict(input_shapes)
    weights64 = {name: np.asarray(t, dtype=np.float64) for name, t in tensors.items()}
    compiled: list[CompiledNode] = []
    for i, raw in enumerate(nodes):
        if not isinstance(raw, dict):
            raise TrcmError(f"узел #{i} должен быть объектом")
        op = raw.get("op")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise TrcmError(f"узел #{i}: нет имени")
        if not isinstance(op, str) or op not in SUPPORTED_OPS:
            raise TrcmError(
                f"неизвестная операция {op!r} в узле #{i} '{name}'. Поддерживаются: {', '.join(SUPPORTED_OPS)}"
            )
        if name in shapes:
            raise TrcmError(f"узел #{i}: имя '{name}' уже занято")
        inputs = raw.get("inputs", [])
        attrs = raw.get("attrs") or {}
        wnames = raw.get("weights") or []
        if not isinstance(inputs, list) or not all(isinstance(n, str) for n in inputs):
            raise TrcmError(f"узел #{i} '{name}': inputs должен быть списком строк")
        if not isinstance(attrs, dict):
            raise TrcmError(f"узел #{i} '{name}': attrs должен быть объектом")
        if not isinstance(wnames, list) or not all(isinstance(n, str) for n in wnames):
            raise TrcmError(f"узел #{i} '{name}': weights должен быть списком строк")
        ctx = _Ctx(i, name, op, attrs)
        for inp in inputs:
            if inp not in shapes:
                raise ctx.error(f"вход '{inp}' не определён до этого узла")
        missing = [w for w in wnames if w not in weights64]
        if missing:
            raise ctx.error(f"нет тензоров: {', '.join(missing)}")
        shape, run = _compile_node(ctx, [shapes[n] for n in inputs], [weights64[w] for w in wnames])
        if any(d <= 0 for d in shape):
            raise ctx.error(f"пустой выход {list(shape)}")
        shapes[name] = tuple(int(d) for d in shape)
        compiled.append(CompiledNode(i, op, name, tuple(inputs), shapes[name], run))
    return compiled, shapes


# ---------------------------------------------------------------------------
# Модель


def _spec_list(items: Any, what: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise TrcmError(f"{what} должен быть списком")
    out = []
    seen = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TrcmError(f"{what}[{i}] должен быть объектом")
        name = item.get("name") if "name" in keys else item.get("id")
        if not isinstance(name, str) or not name:
            raise TrcmError(f"{what}[{i}]: нет имени")
        if name in seen:
            raise TrcmError(f"{what}: '{name}' повторяется")
        seen.add(name)
        for key in keys:
            if key in ("mean", "std"):
                v = item.get(key)
                if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
                    raise TrcmError(f"{what} '{name}': поле {key} должно быть конечным числом")
        out.append(dict(item))
    return out


class ReferenceModel:
    """Загруженная и проверенная модель .trcm с эталонным инференсом."""

    def __init__(self, source: TrcmFile | bytes | bytearray | str | os.PathLike):
        if isinstance(source, TrcmFile):
            trcm = source
        elif isinstance(source, (bytes, bytearray, memoryview)):
            trcm = decode_trcm(source)
        else:
            trcm = read_trcm(source)
        self.trcm = trcm
        h = trcm.header
        self.header = h

        schema = h.get("featureSchema", FEATURE_SCHEMA)
        if schema != FEATURE_SCHEMA:
            raise TrcmError(f"featureSchema={schema!r}, поддерживается {FEATURE_SCHEMA}")

        inputs = h.get("inputs")
        if not isinstance(inputs, dict) or not isinstance(inputs.get("image"), dict):
            raise TrcmError("в заголовке нет описания inputs.image")
        image = inputs["image"]
        dims = (image.get("channels"), image.get("height"), image.get("width"))
        if dims != IMAGE_SHAPE:
            raise TrcmError(f"вход image должен быть 3×64×64, в модели {dims}")
        if image.get("layout", "CHW") != "CHW":
            raise TrcmError(f"layout входа image должен быть CHW, в модели {image.get('layout')!r}")
        norm = image.get("norm") or {"type": "per_image"}
        if not isinstance(norm, dict):
            raise TrcmError("inputs.image.norm должен быть объектом")
        if norm.get("type", "per_image") == "channel":
            for key in ("mean", "std"):
                v = norm.get(key)
                if not isinstance(v, list) or len(v) != IMAGE_CHANNELS or not all(
                    isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x)) for x in v
                ):
                    raise TrcmError(f"norm.{key} должен быть списком из 3 чисел")
            if any(float(x) == 0.0 for x in norm["std"]):
                raise TrcmError("norm.std не может содержать 0")
        elif norm.get("type", "per_image") != "per_image":
            raise TrcmError(f"неизвестный тип нормализации изображения: {norm.get('type')!r}")
        self.image_norm = dict(norm)

        self.feature_specs = _spec_list(inputs.get("features", []), "inputs.features", ("name", "mean", "std"))
        self.classes = _spec_list(h.get("classes"), "classes", ("id",))
        self.param_specs = _spec_list(h.get("params"), "params", ("name", "mean", "std"))
        if not self.classes:
            raise TrcmError("в модели нет ни одного класса")
        if not self.param_specs:
            raise TrcmError("в модели нет ни одного параметра")

        outputs = h.get("outputs")
        if not isinstance(outputs, dict):
            raise TrcmError("в заголовке нет outputs")
        self.output_logits = outputs.get("class_logits")
        self.output_params = outputs.get("params")

        input_shapes = {"image": IMAGE_SHAPE, "features": (len(self.feature_specs),)}
        self.nodes, self.shapes = compile_graph(h.get("nodes"), trcm.tensors, input_shapes)
        k, p = self.num_classes, self.num_params
        for key, out, want in (("class_logits", self.output_logits, (k,)), ("params", self.output_params, (k * p,))):
            if not isinstance(out, str) or out not in self.shapes or out in INPUT_NAMES:
                raise TrcmError(f"outputs.{key} должен ссылаться на узел графа, получено {out!r}")
            if self.shapes[out] != want:
                raise TrcmError(f"выход {key} ('{out}') имеет форму {list(self.shapes[out])}, ожидается {list(want)}")

    # -- свойства

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_params(self) -> int:
        return len(self.param_specs)

    @property
    def feature_names(self) -> list[str]:
        return [s["name"] for s in self.feature_specs]

    @property
    def param_names(self) -> list[str]:
        return [s["name"] for s in self.param_specs]

    @property
    def class_ids(self) -> list[str]:
        return [c["id"] for c in self.classes]

    # -- инференс

    def prepare_image(self, thumb: Any) -> np.ndarray:
        return normalize_image(thumb_to_chw(thumb), self.image_norm)

    def prepare_features(self, raw: Mapping[str, Any] | None) -> np.ndarray:
        return normalize_features(raw, self.feature_specs)

    def run(self, image: np.ndarray, features: np.ndarray, keep_all: bool = False) -> dict[str, np.ndarray]:
        """Исполняет граф на уже подготовленных входах (image [3,64,64], features [F])."""
        image = np.asarray(image, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64).reshape(-1)
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"image должен иметь форму {IMAGE_SHAPE}, получено {image.shape}")
        if features.shape != (len(self.feature_specs),):
            raise ValueError(f"features должен иметь длину {len(self.feature_specs)}, получено {features.size}")
        env: dict[str, np.ndarray] = {"image": image, "features": features}
        for node in self.nodes:
            env[node.name] = node.run([env[n] for n in node.inputs])
        if keep_all:
            return env
        return {"class_logits": env[self.output_logits], "params": env[self.output_params]}

    def predict(self, thumb: Any, raw_features: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Миниатюра uint8 HWC + сырые признаки -> (class_logits [K], params [K·P])."""
        out = self.run(self.prepare_image(thumb), self.prepare_features(raw_features))
        return out["class_logits"], out["params"]

    def denormalize_params(self, params: np.ndarray) -> np.ndarray:
        """params [K·P] -> реальные значения [K, P]: ``out·std + mean``."""
        p = np.asarray(params, dtype=np.float64).reshape(self.num_classes, self.num_params)
        mean = np.array([float(s["mean"]) for s in self.param_specs])
        std = np.array([float(s["std"]) for s in self.param_specs])
        return p * std + mean


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
