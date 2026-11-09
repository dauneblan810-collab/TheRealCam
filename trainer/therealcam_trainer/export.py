"""Экспорт TrcNet в `.trcm` (только операции спецификации) и импорт обратно для дообучения.

BatchNorm сворачивается в веса свёрток, Dropout выбрасывается. При импорте свёрнутые веса
попадают в свёртки со смещением, а BN становится тождественным (gamma=1, beta=0, mean=0,
var=1, eps=0) и замораживается, так что повторный экспорт даёт те же самые числа.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import TRAINER_NAME, __version__
from .errors import TrcmError
from .model import DEFAULT_DROPOUT, ModelSpec, TrcNet
from .reference import ReferenceModel, effective_std
from .schema import FEATURE_SCHEMA, FOCUS_PARAM, IMAGE_CHANNELS, PARAM_NAMES_V1, THUMB_SIZE
from .trcm import TrcmFile, decode_trcm, encode_trcm, read_trcm

OUTPUT_LOGITS = "logits"
OUTPUT_PARAMS = "params"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trcnet_graph(widths, hidden: int, num_features: int, num_classes: int, num_params: int):
    """Узлы графа TrcNet и ожидаемые формы тензоров (порядок тензоров = порядок в данных)."""
    nodes: list[dict] = []
    shapes: "OrderedDict[str, list[int]]" = OrderedDict()

    def dense(name: str, inp: str, n_in: int, n_out: int) -> None:
        nodes.append({"op": "dense", "name": name, "inputs": [inp], "attrs": {"in": n_in, "out": n_out},
                      "weights": [f"{name}.weight", f"{name}.bias"]})
        shapes[f"{name}.weight"] = [n_out, n_in]
        shapes[f"{name}.bias"] = [n_out]

    prev, cin = "image", IMAGE_CHANNELS
    for i, cout in enumerate(widths, start=1):
        name = f"conv{i}"
        nodes.append({"op": "conv2d", "name": name, "inputs": [prev],
                      "attrs": {"inChannels": cin, "outChannels": int(cout), "kernel": 3, "stride": 1,
                                "padding": 1, "groups": 1},
                      "weights": [f"{name}.weight", f"{name}.bias"]})
        shapes[f"{name}.weight"] = [int(cout), cin, 3, 3]
        shapes[f"{name}.bias"] = [int(cout)]
        nodes.append({"op": "relu", "name": f"relu{i}", "inputs": [name]})
        prev = f"relu{i}"
        if i < len(widths):
            nodes.append({"op": "maxpool2d", "name": f"pool{i}", "inputs": [prev], "attrs": {"kernel": 2, "stride": 2}})
            prev = f"pool{i}"
        cin = int(cout)
    nodes.append({"op": "global_avgpool", "name": "gap", "inputs": [prev]})
    prev = "gap"
    if num_features > 0:
        nodes.append({"op": "concat", "name": "concat", "inputs": ["gap", "features"]})
        prev = "concat"
    dense("fc", prev, cin + num_features, hidden)
    nodes.append({"op": "relu", "name": "fc_relu", "inputs": ["fc"]})
    dense(OUTPUT_LOGITS, "fc_relu", hidden, num_classes)
    dense(OUTPUT_PARAMS, "fc_relu", hidden, num_classes * num_params)
    return nodes, shapes


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Сворачивает инференс-BN в свёртку (в float64), результат float32."""
    with torch.no_grad():
        w = conv.weight.detach().cpu().double()
        b = conv.bias.detach().cpu().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
        gamma = bn.weight.detach().cpu().double() if bn.affine else torch.ones_like(b)
        beta = bn.bias.detach().cpu().double() if bn.affine else torch.zeros_like(b)
        mean = bn.running_mean.detach().cpu().double()
        var = bn.running_var.detach().cpu().double()
        scale = gamma / torch.sqrt(var + bn.eps)
        wf = w * scale[:, None, None, None]
        bf = (b - mean) * scale + beta
    return wf.float().numpy(), bf.float().numpy()


def trcnet_tensors(model: TrcNet) -> "OrderedDict[str, np.ndarray]":
    tensors: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for i, (conv, bn) in enumerate(model.conv_bn_pairs(), start=1):
        w, b = fold_conv_bn(conv, bn)
        tensors[f"conv{i}.weight"] = w
        tensors[f"conv{i}.bias"] = b
    for name, lin in (("fc", model.fc), (OUTPUT_LOGITS, model.cls_head), (OUTPUT_PARAMS, model.param_head)):
        tensors[f"{name}.weight"] = lin.weight.detach().cpu().float().numpy()
        tensors[f"{name}.bias"] = lin.bias.detach().cpu().float().numpy()
    return tensors


def build_header(model: TrcNet, spec: ModelSpec, *, name: str, training: Mapping[str, Any] | None = None,
                 description: str = "", author: str = "", created_utc: str | None = None) -> dict:
    if len(spec.classes) != model.num_classes:
        raise ValueError(f"классов в описании {len(spec.classes)}, в сети {model.num_classes}")
    if len(spec.feature_names) != model.num_features:
        raise ValueError(f"признаков в описании {len(spec.feature_names)}, в сети {model.num_features}")
    if len(spec.param_names) != model.num_params:
        raise ValueError(f"параметров в описании {len(spec.param_names)}, в сети {model.num_params}")
    nodes, _ = trcnet_graph(model.widths, model.hidden, model.num_features, model.num_classes, model.num_params)
    train_meta = {"samples": 0, "epochs": 0, "baseModel": None, "metrics": {}}
    train_meta.update(dict(training or {}))
    return {
        "format": "trcm",
        "formatVersion": 1,
        "name": name,
        "description": description,
        "author": author,
        "createdUtc": created_utc or utc_now_iso(),
        "trainer": {"name": TRAINER_NAME, "version": __version__},
        "training": train_meta,
        "featureSchema": FEATURE_SCHEMA,
        "inputs": {
            "image": {"width": THUMB_SIZE, "height": THUMB_SIZE, "channels": IMAGE_CHANNELS, "layout": "CHW",
                      "norm": {"type": "per_image"}},
            "features": spec.feature_specs(),
        },
        "classes": [{"id": c["id"], "name": c.get("name") or c["id"], "icon": c.get("icon") or "other"}
                    for c in spec.classes],
        "params": spec.param_specs(),
        "outputs": {"class_logits": OUTPUT_LOGITS, "params": OUTPUT_PARAMS},
        "nodes": nodes,
    }


def export_trcnet(model: TrcNet, spec: ModelSpec, path: str | os.PathLike | None = None, *, name: str,
                  training: Mapping[str, Any] | None = None, description: str = "", author: str = "",
                  created_utc: str | None = None) -> bytes:
    """Экспортирует сеть в байты .trcm (и пишет файл, если задан path). Результат проверяется
    эталонным интерпретатором."""
    header = build_header(model, spec, name=name, training=training, description=description, author=author,
                          created_utc=created_utc)
    blob = encode_trcm(header, trcnet_tensors(model))
    ReferenceModel(decode_trcm(blob))  # проверка структуры того, что записали
    if path is not None:
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, p)
    return blob


def _strip(node: Mapping[str, Any]) -> dict:
    out = {"op": node.get("op"), "name": node.get("name"), "inputs": list(node.get("inputs") or [])}
    if node.get("attrs"):
        out["attrs"] = dict(node["attrs"])
    if node.get("weights"):
        out["weights"] = list(node["weights"])
    return out


def load_trcnet(source: TrcmFile | bytes | str | os.PathLike) -> tuple[TrcNet, ModelSpec, dict]:
    """Импорт .trcm, созданного этим тренером, обратно в TrcNet (для дообучения)."""
    if isinstance(source, TrcmFile):
        trcm = source
    elif isinstance(source, (bytes, bytearray)):
        trcm = decode_trcm(source)
    else:
        trcm = read_trcm(source)
    ref = ReferenceModel(trcm)
    h = trcm.header
    t = trcm.tensors

    def fail(why: str) -> TrcmError:
        return TrcmError(
            "дообучать можно только модели архитектуры TrcNet, экспортированные этим тренером: " + why
        )

    if ref.image_norm.get("type", "per_image") != "per_image":
        raise fail("нормализация изображения должна быть per_image")
    if ref.param_names != list(PARAM_NAMES_V1):
        raise fail(f"параметры {ref.param_names}, ожидаются {PARAM_NAMES_V1}")
    try:
        widths = tuple(int(t[f"conv{i}.weight"].shape[0]) for i in range(1, 5))
        hidden = int(t["fc.weight"].shape[0])
    except KeyError as exc:
        raise fail(f"нет тензора {exc}") from None
    k, p, nf = ref.num_classes, ref.num_params, len(ref.feature_specs)
    want_nodes, want_shapes = trcnet_graph(widths, hidden, nf, k, p)
    got_nodes = [_strip(n) for n in h["nodes"]]
    if got_nodes != want_nodes:
        for i, (a, b) in enumerate(zip(got_nodes, want_nodes)):
            if a != b:
                raise fail(f"узел #{i}: {a} вместо {b}")
        raise fail(f"узлов {len(got_nodes)}, ожидается {len(want_nodes)}")
    for name, shape in want_shapes.items():
        if name not in t or list(t[name].shape) != shape:
            raise fail(f"тензор {name}: ожидается форма {shape}")

    training = h.get("training") if isinstance(h.get("training"), dict) else {}
    dropout = training.get("dropout", DEFAULT_DROPOUT)
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        dropout = DEFAULT_DROPOUT
    model = TrcNet(nf, k, p, widths=widths, hidden=hidden, dropout=float(dropout))
    with torch.no_grad():
        for i, (conv, bn) in enumerate(model.conv_bn_pairs(), start=1):
            conv.weight.copy_(torch.from_numpy(np.array(t[f"conv{i}.weight"])))
            conv.bias.copy_(torch.from_numpy(np.array(t[f"conv{i}.bias"])))
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.eps = 0.0
        for name, lin in (("fc", model.fc), (OUTPUT_LOGITS, model.cls_head), (OUTPUT_PARAMS, model.param_head)):
            lin.weight.copy_(torch.from_numpy(np.array(t[f"{name}.weight"])))
            lin.bias.copy_(torch.from_numpy(np.array(t[f"{name}.bias"])))
    model.bn_frozen = True
    model.eval()
    spec = ModelSpec(
        classes=[{"id": c["id"], "name": c.get("name") or c["id"], "icon": c.get("icon") or "other"}
                 for c in ref.classes],
        feature_names=ref.feature_names,
        feature_mean=[float(s["mean"]) for s in ref.feature_specs],
        feature_std=[float(s["std"]) for s in ref.feature_specs],
        param_names=ref.param_names,
        param_mean=[float(s["mean"]) for s in ref.param_specs],
        param_std=[float(s["std"]) for s in ref.param_specs],
    )
    return model, spec, h


def extend_classes(model: TrcNet, num_classes: int, seed: int | None = None) -> TrcNet:
    """Добавляет классы в конец: строки старых классов (логит и P параметров) копируются,
    новые строки получают свежую случайную инициализацию."""
    k0, p = model.num_classes, model.num_params
    if num_classes < k0:
        raise ValueError("нельзя уменьшить число классов")
    if num_classes == k0:
        return model
    if seed is not None:
        torch.manual_seed(seed)
    new = TrcNet(model.num_features, num_classes, p, widths=model.widths, hidden=model.hidden,
                 dropout=model.dropout_p)
    old_sd = model.state_dict()
    new_sd = new.state_dict()
    for key, value in old_sd.items():
        if key.startswith("cls_head."):
            new_sd[key][:k0] = value
        elif key.startswith("param_head."):
            new_sd[key][: k0 * p] = value
        else:
            new_sd[key] = value.clone()
    new.load_state_dict(new_sd)
    for (_, bn_old), (_, bn_new) in zip(model.conv_bn_pairs(), new.conv_bn_pairs()):
        bn_new.eps = bn_old.eps  # eps не входит в state_dict
    new.bn_frozen = model.bn_frozen
    new.train(model.training)
    return new


def refit_feature_norm(model: TrcNet, old_mean, old_std, new_mean, new_std) -> None:
    """Меняет нормализацию признаков, сохраняя функцию сети (пересчёт весов первого слоя fc)."""
    if model.num_features == 0:
        return
    s0 = np.array([effective_std(s) for s in old_std])
    s1 = np.array([effective_std(s) for s in new_std])
    a = s1 / s0
    c = (np.asarray(new_mean, dtype=np.float64) - np.asarray(old_mean, dtype=np.float64)) / s0
    with torch.no_grad():
        w = model.fc.weight.detach().double()
        g = model.widths[3]
        wf = w[:, g:]
        bias = model.fc.bias.detach().double() + wf @ torch.from_numpy(c)
        w[:, g:] = wf * torch.from_numpy(a)[None, :]
        model.fc.weight.copy_(w.float())
        model.fc.bias.copy_(bias.float())


def refit_param_norm(model: TrcNet, param_names, old_mean, old_std, new_mean, new_std) -> None:
    """Меняет нормализацию параметров, сохраняя реальные значения выходов."""
    k, p = model.num_classes, model.num_params
    ratio = np.ones(p)
    shift = np.zeros(p)
    for j, name in enumerate(param_names):
        if name == FOCUS_PARAM:
            continue
        ratio[j] = float(old_std[j]) / float(new_std[j])
        shift[j] = (float(old_mean[j]) - float(new_mean[j])) / float(new_std[j])
    r = torch.from_numpy(np.tile(ratio, k))
    s = torch.from_numpy(np.tile(shift, k))
    with torch.no_grad():
        w = model.param_head.weight.detach().double() * r[:, None]
        b = model.param_head.bias.detach().double() * r + s
        model.param_head.weight.copy_(w.float())
        model.param_head.bias.copy_(b.float())
