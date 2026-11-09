"""TrcNet — маленькая свёрточная сеть по миниатюре 64×64 плюс MLP по признакам сцены (PyTorch).

Инициализация только случайная: никаких предобученных весов.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .schema import FOCUS_PARAM, PARAM_NAMES_V1

DEFAULT_WIDTHS = (16, 32, 48, 64)
DEFAULT_HIDDEN = 96
DEFAULT_DROPOUT = 0.2
PER_IMAGE_STD_FLOOR = 1.0 / math.sqrt(3 * 64 * 64)


def f32(value: float) -> float:
    """Значение, точно представимое во float32 (чтобы заголовок и Kotlin видели одно и то же число)."""
    return float(np.float32(value))


class TrcNet(nn.Module):
    """[conv3×3 → BN → ReLU → MaxPool2] ×3 → conv3×3 → BN → ReLU → GAP; concat(признаки) →
    Linear → ReLU → Dropout → головы: логиты [K] и параметры [K·P]."""

    def __init__(self, num_features: int, num_classes: int, num_params: int = len(PARAM_NAMES_V1),
                 widths=DEFAULT_WIDTHS, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if len(widths) != 4 or min(widths) < 1:
            raise ValueError("widths: нужно 4 положительных числа")
        if num_features < 0 or num_classes < 1 or num_params < 1 or hidden < 1:
            raise ValueError("некорректные размеры TrcNet")
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.num_params = int(num_params)
        self.widths = widths
        self.hidden = int(hidden)
        self.dropout_p = float(dropout)

        chans = (3,) + widths
        self.conv1 = nn.Conv2d(chans[0], chans[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(chans[1])
        self.conv2 = nn.Conv2d(chans[1], chans[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(chans[2])
        self.conv3 = nn.Conv2d(chans[2], chans[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(chans[3])
        self.conv4 = nn.Conv2d(chans[3], chans[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(chans[4])
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(widths[3] + self.num_features, self.hidden)
        self.drop = nn.Dropout(self.dropout_p)
        self.cls_head = nn.Linear(self.hidden, self.num_classes)
        self.param_head = nn.Linear(self.hidden, self.num_classes * self.num_params)
        # При дообучении импортированной модели BN уже свёрнут в свёртки: статистики не трогаем.
        self.bn_frozen = False

    def conv_bn_pairs(self) -> list[tuple[nn.Conv2d, nn.BatchNorm2d]]:
        return [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3), (self.conv4, self.bn4)]

    def forward(self, image: torch.Tensor, features: torch.Tensor | None = None):
        x = image
        for i, (conv, bn) in enumerate(self.conv_bn_pairs()):
            x = F.relu(bn(conv(x)))
            if i < 3:
                x = self.pool(x)
        x = x.mean(dim=(2, 3))
        if self.num_features:
            if features is None:
                raise ValueError("модели нужны признаки сцены")
            x = torch.cat([x, features], dim=1)
        h = self.drop(F.relu(self.fc(x)))
        return self.cls_head(h), self.param_head(h)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.bn_frozen:
            for _, bn in self.conv_bn_pairs():
                bn.eval()
        return self

    def arch(self) -> dict:
        return {"widths": list(self.widths), "hidden": self.hidden, "dropout": self.dropout_p}


def per_image_normalize(x: torch.Tensor) -> torch.Tensor:
    """x [B,3,64,64] в [0,1] -> (x - mean) / max(std, 1/sqrt(12288)) по каждой картинке."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.var(dim=(1, 2, 3), keepdim=True, correction=0).sqrt().clamp_min(PER_IMAGE_STD_FLOOR)
    return (x - m) / s


@dataclass
class ModelSpec:
    """Всё, что кроме весов описывает модель: классы, признаки, параметры и их нормализация."""

    classes: list[dict]
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    param_names: list[str] = field(default_factory=lambda: list(PARAM_NAMES_V1))
    param_mean: list[float] = field(default_factory=lambda: [0.0] * len(PARAM_NAMES_V1))
    param_std: list[float] = field(default_factory=lambda: [1.0] * len(PARAM_NAMES_V1))

    def __post_init__(self):
        self.feature_mean = [f32(v) for v in self.feature_mean]
        self.feature_std = [f32(v) for v in self.feature_std]
        self.param_mean = [f32(v) for v in self.param_mean]
        self.param_std = [f32(v) for v in self.param_std]
        if not (len(self.feature_names) == len(self.feature_mean) == len(self.feature_std)):
            raise ValueError("длины feature_names/mean/std не совпадают")
        if not (len(self.param_names) == len(self.param_mean) == len(self.param_std)):
            raise ValueError("длины param_names/mean/std не совпадают")
        if FOCUS_PARAM in self.param_names:
            i = self.param_names.index(FOCUS_PARAM)
            if self.param_mean[i] != 0.0 or self.param_std[i] != 1.0:
                raise ValueError("focus_infinity — логит: mean должен быть 0, std 1")

    @property
    def class_ids(self) -> list[str]:
        return [c["id"] for c in self.classes]

    def feature_specs(self) -> list[dict]:
        return [{"name": n, "mean": m, "std": s}
                for n, m, s in zip(self.feature_names, self.feature_mean, self.feature_std)]

    def param_specs(self) -> list[dict]:
        return [{"name": n, "mean": m, "std": s} for n, m, s in zip(self.param_names, self.param_mean, self.param_std)]
