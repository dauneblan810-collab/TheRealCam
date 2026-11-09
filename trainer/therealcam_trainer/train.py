"""Обучение TrcNet с нуля и дообучение существующей `.trcm` на датасете владельца."""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import (
    DEFAULT_DELTA_EV,
    DEFAULT_VAL_FRACTION,
    MIN_FEATURE_PRESENCE,
    Dataset,
    TrainingData,
    build_training_data,
    load_dataset,
)
from .errors import TrainError
from .export import export_trcnet, extend_classes, load_trcnet, refit_feature_norm, refit_param_norm
from .model import DEFAULT_DROPOUT, DEFAULT_HIDDEN, DEFAULT_WIDTHS, ModelSpec, TrcNet, per_image_normalize
from .schema import FOCUS_PARAM, MIN_SAMPLES

SCRATCH_EPOCHS = 60
SCRATCH_LR = 2e-3
FINETUNE_EPOCHS = 30
FINETUNE_LR = 3e-4
MIN_LR_RATIO = 0.02
SHIFT_PX = 4
FEATURE_NOISE = 0.05
FEATURE_DROP = 0.05

Progress = Callable[[dict], None]


@dataclass
class TrainConfig:
    epochs: int | None = None  # None: 60 с нуля, 30 при дообучении
    batch_size: int = 64
    lr: float | None = None  # None: 2e-3 с нуля, 3e-4 при дообучении
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 0
    patience: int = 12
    delta_ev: float = DEFAULT_DELTA_EV
    val_fraction: float = DEFAULT_VAL_FRACTION
    allow_small: bool = False
    label_smoothing: float = 0.05
    param_loss_weight: float = 1.0
    min_feature_presence: float = MIN_FEATURE_PRESENCE
    feature_names: list[str] | None = None  # принудительный список признаков (по умолчанию — отбор ≥10%)
    base_model: str | None = None  # путь к .trcm для дообучения
    refit_norm: bool = False
    augment: bool = True
    num_threads: int | None = None
    widths: tuple = DEFAULT_WIDTHS
    hidden: int = DEFAULT_HIDDEN
    dropout: float = DEFAULT_DROPOUT


@dataclass
class TrainResult:
    model: TrcNet
    spec: ModelSpec
    history: list[dict]
    best_epoch: int
    epochs_run: int
    cancelled: bool
    metrics: dict
    n_total: int
    n_train: int
    n_val: int
    device: str
    config: TrainConfig
    base_name: str | None = None
    base_sha256: str | None = None
    notes: list[str] = field(default_factory=list)

    def training_meta(self) -> dict:
        meta = {
            "samples": self.n_total,
            "epochs": self.epochs_run,
            "baseModel": self.base_name,
            "metrics": self.metrics,
            "trainSamples": self.n_train,
            "valSamples": self.n_val,
            "bestEpoch": self.best_epoch,
            "seed": self.config.seed,
            "deltaEv": self.config.delta_ev,
            "dropout": self.model.dropout_p,
            "device": self.device,
        }
        if self.base_sha256:
            meta["baseModelSha256"] = self.base_sha256
        if self.cancelled:
            meta["stoppedByUser"] = True
        return meta

    def export(self, path=None, *, name: str, description: str = "", author: str = "",
               created_utc: str | None = None) -> bytes:
        return export_trcnet(self.model, self.spec, path, name=name, training=self.training_meta(),
                             description=description, author=author, created_utc=created_utc)


def resolve_device(name: str | None = "auto") -> torch.device:
    """auto: cuda > mps > cpu. Явное имя проверяется на доступность."""
    name = (name or "auto").strip().lower()
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps is not None and mps.is_available())
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    try:
        dev = torch.device(name)
    except RuntimeError:
        raise TrainError(f"неизвестное устройство {name!r}: используйте auto, cpu, cuda или mps") from None
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise TrainError("CUDA недоступна: нужна сборка PyTorch с CUDA и драйвер NVIDIA (или выберите cpu)")
    if dev.type == "mps" and not mps_ok:
        raise TrainError("MPS (Apple GPU) недоступен на этом компьютере")
    if dev.type not in ("cpu", "cuda", "mps"):
        raise TrainError(f"устройство {name!r} не поддерживается")
    return dev


def quiet_torch() -> None:
    # NNPACK на части CPU печатает предупреждения; для такой маленькой сети он не нужен.
    try:
        torch.backends.nnpack.set_flags(False)
    except Exception:
        pass


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Аугментации и потери


def augment_images(x_u8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """uint8 [B,3,64,64] -> float [B,3,64,64] в [0,1]: флип, сдвиг ±4 px (reflect), яркость/контраст, шум."""
    b = x_u8.shape[0]
    dev = x_u8.device
    x = x_u8.float()
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    xp = F.pad(x, (SHIFT_PX,) * 4, mode="reflect")
    dy = torch.randint(0, 2 * SHIFT_PX + 1, (b,), generator=gen).to(dev)
    dx = torch.randint(0, 2 * SHIFT_PX + 1, (b,), generator=gen).to(dev)
    ar = torch.arange(x.shape[-1], device=dev)
    rows = (dy[:, None] + ar[None, :])[:, None, :, None]
    cols = (dx[:, None] + ar[None, :])[:, None, None, :]
    x = xp[torch.arange(b, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows, cols]
    contrast = (1 + (torch.rand(b, generator=gen) * 2 - 1) * 0.15).to(dev).view(b, 1, 1, 1)
    bright = ((torch.rand(b, generator=gen) * 2 - 1) * 12.0).to(dev).view(b, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * contrast + m + bright
    sigma = (torch.rand(b, generator=gen) * 4.0).to(dev).view(b, 1, 1, 1)
    x = x + torch.randn(x.shape, generator=gen).to(dev) * sigma
    return x.clamp(0, 255) / 255.0


def augment_features(f: torch.Tensor, present: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Шум σ=0.05 в нормализованном пространстве и выпадение признака в «отсутствует» с p=0.05."""
    if f.shape[1] == 0:
        return f
    dev = f.device
    noise = torch.randn(f.shape, generator=gen).to(dev) * FEATURE_NOISE
    keep = (torch.rand(f.shape, generator=gen) >= FEATURE_DROP).to(dev)
    out = torch.where(present, f + noise, f)
    out = torch.where(keep, out, torch.zeros_like(out))
    return out.clamp(-6.0, 6.0)


def loss_terms(logits, pout, y, tgt, tmask, gw, pw, class_w, label_smoothing: float, focus_idx: int | None):
    """Суммы для взвешенных потерь: (ce·w, w, param·w, w_param)."""
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    ce = (1 - label_smoothing) * nll + label_smoothing * (-logp.mean(dim=1))
    wc = class_w[y] * gw
    b, k = logits.shape
    p = pout.shape[1] // k
    rows = pout.view(b, k, p)[torch.arange(b, device=y.device), y]
    per = F.smooth_l1_loss(rows, tgt, reduction="none", beta=1.0)
    if focus_idx is not None:
        bce = F.binary_cross_entropy_with_logits(rows[:, focus_idx], tgt[:, focus_idx], reduction="none")
        per = torch.cat([per[:, :focus_idx], bce[:, None], per[:, focus_idx + 1:]], dim=1)
    per = (per * tmask).sum(dim=1)
    wp = pw * gw * (tmask.sum(dim=1) > 0).float()
    return (ce * wc).sum(), wc.sum(), (per * wp).sum(), wp.sum()


def _combine(terms, param_weight: float) -> torch.Tensor:
    ce_sum, ce_w, par_sum, par_w = terms
    loss = ce_sum / ce_w.clamp_min(1e-8)
    if float(par_w) > 0:
        loss = loss + param_weight * par_sum / par_w
    return loss


class _Tensors:
    def __init__(self, td: TrainingData, device: torch.device):
        self.images = torch.from_numpy(td.images).permute(0, 3, 1, 2).contiguous().to(device)
        self.feats = torch.from_numpy(td.feats).to(device)
        self.present = torch.from_numpy(td.feat_present).to(device)
        self.y = torch.from_numpy(td.y).to(device)
        self.targets = torch.from_numpy(td.targets).to(device)
        self.tmask = torch.from_numpy(td.target_mask.astype(np.float32)).to(device)
        self.group_w = torch.from_numpy(td.group_w).to(device)
        self.param_w = torch.from_numpy(td.param_w).to(device)
        self.class_w = torch.from_numpy(td.class_w).to(device)
        self.param_std = td.param_std.astype(np.float64)


@torch.no_grad()
def _evaluate(model: TrcNet, t: _Tensors, idx: np.ndarray, cfg: TrainConfig, focus_idx: int | None,
              param_names: list[str], batch: int = 256) -> dict:
    model.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    correct = 0
    p = len(param_names)
    abs_err = np.zeros(p)
    count = np.zeros(p)
    focus_ok = 0.0
    focus_n = 0.0
    for start in range(0, len(idx), batch):
        ci = torch.from_numpy(idx[start:start + batch]).to(t.y.device)
        x = per_image_normalize(t.images[ci].float() / 255.0)
        logits, pout = model(x, t.feats[ci])
        terms = loss_terms(logits, pout, t.y[ci], t.targets[ci], t.tmask[ci], t.group_w[ci], t.param_w[ci],
                           t.class_w, cfg.label_smoothing, focus_idx)
        for i, v in enumerate(terms):
            sums[i] += float(v)
        y = t.y[ci]
        correct += int((logits.argmax(dim=1) == y).sum())
        b, k = logits.shape
        rows = pout.view(b, k, p)[torch.arange(b, device=y.device), y]
        mask = t.tmask[ci]
        err = ((rows - t.targets[ci]).abs() * mask).sum(dim=0).double().cpu().numpy()
        abs_err += err * t.param_std
        count += mask.sum(dim=0).double().cpu().numpy()
        if focus_idx is not None:
            ok = ((rows[:, focus_idx] >= 0).float() == t.targets[ci][:, focus_idx]).float() * mask[:, focus_idx]
            focus_ok += float(ok.sum())
            focus_n += float(mask[:, focus_idx].sum())
    loss = sums[0] / max(sums[1], 1e-8) + (cfg.param_loss_weight * sums[2] / sums[3] if sums[3] > 0 else 0.0)
    mae = {name: (float(abs_err[j] / count[j]) if count[j] else None)
           for j, name in enumerate(param_names) if name != FOCUS_PARAM}
    return {"loss": loss, "acc": correct / max(len(idx), 1), "mae": mae,
            "focus_acc": (focus_ok / focus_n) if focus_n else None}


def format_epoch(ev: dict) -> str:
    """Строка прогресса эпохи для CLI и GUI."""
    parts = [f"эпоха {ev['epoch']:>3}/{ev['epochs']}", f"train {ev['train_loss']:.4f}"]
    if ev.get("val_loss") is not None:
        parts.append(f"val {ev['val_loss']:.4f}")
        parts.append(f"точность {ev['val_acc']:.3f}")
    mae = ev.get("mae") or {}
    short = {"log2_exposure_s": "выдержка", "log2_iso": "ISO", "log2_frames": "кадры", "ev_bias": "EV"}
    m = [f"{short.get(k, k)} {v:.2f}" for k, v in mae.items() if v is not None]
    if m:
        parts.append("MAE[" + ", ".join(m) + "]")
    if ev.get("focus_acc") is not None:
        parts.append(f"фокус∞ {ev['focus_acc']:.2f}")
    parts.append(f"lr {ev['lr']:.1e}")
    if ev.get("improved"):
        parts.append("*")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Обучение


def _merge_classes(base_classes: list[dict], ds: Dataset) -> list[dict]:
    """Классы базовой модели первыми (имя/значок обновляются из датасета), новые — в конец."""
    current = {c.id: c for c in ds.categories.values()}
    merged = []
    for c in base_classes:
        cur = current.get(c["id"])
        merged.append({"id": c["id"], "name": cur.name if cur else c["name"], "icon": cur.icon if cur else c["icon"]})
    known = {c["id"] for c in merged}
    merged += [c.as_class() for c in ds.classes() if c.id not in known]
    return merged


def train_model(dataset: Dataset | str, config: TrainConfig | None = None, progress: Progress | None = None,
                cancel: threading.Event | None = None) -> TrainResult:
    """Обучает модель (с нуля или от base_model) и возвращает лучшую по val loss."""
    cfg = config or TrainConfig()

    def emit(event: dict) -> None:
        if progress is not None:
            progress(event)

    def log(msg: str) -> None:
        emit({"type": "log", "message": msg})

    quiet_torch()
    old_threads = torch.get_num_threads()
    if cfg.num_threads:
        torch.set_num_threads(int(cfg.num_threads))
    try:
        ds = dataset if isinstance(dataset, Dataset) else load_dataset(dataset)
        n = len(ds.samples)
        if n == 0:
            raise TrainError("в датасете нет ни одного полностью оценённого снимка")
        if n < MIN_SAMPLES and not cfg.allow_small:
            raise TrainError(
                f"в датасете {n} полностью оценённых снимков, для обучения нужно минимум {MIN_SAMPLES}. "
                "Оцените больше снимков в «Real Галерее» (флаг --allow-small только для тестов)."
            )
        if cfg.batch_size < 1:
            raise TrainError("размер батча должен быть ≥ 1")
        finetune = cfg.base_model is not None
        epochs = int(cfg.epochs or (FINETUNE_EPOCHS if finetune else SCRATCH_EPOCHS))
        lr = float(cfg.lr or (FINETUNE_LR if finetune else SCRATCH_LR))
        if epochs < 1 or lr <= 0:
            raise TrainError("число эпох и скорость обучения должны быть положительными")
        device = resolve_device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed % (2 ** 32))
        base_name = base_sha = None
        if finetune:
            base_model, base_spec, base_header = load_trcnet(cfg.base_model)
            base_name = str(base_header.get("name") or Path(cfg.base_model).stem)
            base_sha = _file_sha256(cfg.base_model)
            classes = _merge_classes(base_spec.classes, ds)
            unknown = [c for c in base_spec.feature_names if c not in ds.samples[0].features]
            if unknown:
                log("признаки базовой модели, которых нет в схеме 1 (будут «отсутствует»): " + ", ".join(unknown))
            td = build_training_data(
                ds, classes=classes, feature_names=base_spec.feature_names,
                fixed_feature_norm=None if cfg.refit_norm else (base_spec.feature_mean, base_spec.feature_std),
                fixed_param_norm=None if cfg.refit_norm else (base_spec.param_mean, base_spec.param_std),
                fallback_feature_norm=(base_spec.feature_mean, base_spec.feature_std),
                delta_ev=cfg.delta_ev, val_fraction=cfg.val_fraction, seed=cfg.seed,
                min_feature_presence=cfg.min_feature_presence)
            model = extend_classes(base_model, len(classes), seed=cfg.seed)
            if cfg.refit_norm:
                refit_feature_norm(model, base_spec.feature_mean, base_spec.feature_std, td.feature_mean, td.feature_std)
                refit_param_norm(model, td.param_names, base_spec.param_mean, base_spec.param_std, td.param_mean,
                                 td.param_std)
            added = [c["name"] for c in classes[len(base_spec.classes):]]
            no_data = [c["name"] for i, c in enumerate(classes) if not np.any(td.y == i)]
            log(f"дообучение модели «{base_name}»: классов в базе {len(base_spec.classes)}, "
                f"добавлено {len(added)}" + (f" ({', '.join(added)})" if added else ""))
            if no_data:
                log("классы без снимков в этом датасете (сохраняются как есть): " + ", ".join(no_data))
        else:
            td = build_training_data(ds, feature_names=cfg.feature_names, delta_ev=cfg.delta_ev,
                                     val_fraction=cfg.val_fraction, seed=cfg.seed,
                                     min_feature_presence=cfg.min_feature_presence)
            model = TrcNet(len(td.feature_names), td.num_classes, len(td.param_names), widths=cfg.widths,
                           hidden=cfg.hidden, dropout=cfg.dropout)
        for note in td.notes:
            log(note)
        spec = ModelSpec(classes=td.classes, feature_names=td.feature_names, feature_mean=list(td.feature_mean),
                         feature_std=list(td.feature_std), param_names=td.param_names,
                         param_mean=list(td.param_mean), param_std=list(td.param_std))

        n_train, n_val = len(td.train_idx), len(td.val_idx)
        emit({"type": "start", "n_total": n, "n_train": n_train, "n_val": n_val, "epochs": epochs,
              "classes": [c["name"] for c in td.classes], "features": len(td.feature_names),
              "device": str(device), "finetune": finetune, "lr": lr})
        log(f"снимков {n}: train {n_train}, val {n_val}; классов {td.num_classes}; признаков "
            f"{len(td.feature_names)}; устройство {device}; эпох до {epochs}; lr {lr:g}")
        if n_val == 0:
            log("валидационная выборка пуста (слишком мало данных): ранняя остановка по train loss")

        model.to(device)
        t = _Tensors(td, device)
        focus_idx = td.param_names.index(FOCUS_PARAM) if FOCUS_PARAM in td.param_names else None
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        steps = math.ceil(n_train / cfg.batch_size)
        total_steps = max(1, epochs * steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        train_idx = td.train_idx

        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_epoch = 0
        best_metrics: dict = {}
        bad = 0
        history: list[dict] = []
        cancelled = False
        epochs_run = 0
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            model.train()
            perm = train_idx[torch.randperm(n_train, generator=gen).numpy()]
            run_loss = 0.0
            seen = 0
            for bi in range(steps):
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                ci = torch.from_numpy(perm[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]).to(device)
                if cfg.augment:
                    x = augment_images(t.images[ci], gen)
                    f = augment_features(t.feats[ci], t.present[ci], gen)
                else:
                    x = t.images[ci].float() / 255.0
                    f = t.feats[ci]
                logits, pout = model(per_image_normalize(x), f)
                terms = loss_terms(logits, pout, t.y[ci], t.targets[ci], t.tmask[ci], t.group_w[ci], t.param_w[ci],
                                   t.class_w, cfg.label_smoothing, focus_idx)
                loss = _combine(terms, cfg.param_loss_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += float(loss.detach()) * len(ci)
                seen += len(ci)
                emit({"type": "batch", "epoch": epoch, "epochs": epochs, "batch": bi + 1, "batches": steps})
            if cancelled and seen < n_train:
                break
            epochs_run = epoch
            train_loss = run_loss / max(seen, 1)
            if not math.isfinite(train_loss):
                raise TrainError(f"обучение разошлось на эпохе {epoch} (loss = {train_loss}); уменьшите lr")
            val = _evaluate(model, t, td.val_idx, cfg, focus_idx, td.param_names) if n_val else None
            monitor = val["loss"] if val else train_loss
            improved = monitor < best_loss - 1e-6
            if improved:
                best_loss = monitor
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
            ev = {"type": "epoch", "epoch": epoch, "epochs": epochs, "train_loss": train_loss,
                  "val_loss": val["loss"] if val else None, "val_acc": val["acc"] if val else None,
                  "mae": val["mae"] if val else {}, "focus_acc": val["focus_acc"] if val else None,
                  "lr": opt.param_groups[0]["lr"], "best_epoch": best_epoch, "improved": improved,
                  "elapsed_s": time.time() - t0}
            history.append({k: v for k, v in ev.items() if k != "type"})
            if improved:
                best_metrics = _metrics_from(ev)
            emit(ev)
            if bad >= cfg.patience:
                log(f"ранняя остановка: val loss не улучшался {cfg.patience} эпох (лучшая эпоха {best_epoch})")
                break
        if cancelled:
            log("обучение остановлено пользователем" + (f"; берём лучшую эпоху {best_epoch}" if best_epoch else ""))
        if best_epoch == 0 and epochs_run == 0:
            raise TrainError("обучение остановлено до завершения первой эпохи — модель не сохранена")
        model.load_state_dict(best_state)
        model.eval()
        model.to("cpu")
        result = TrainResult(model=model, spec=spec, history=history, best_epoch=best_epoch, epochs_run=epochs_run,
                             cancelled=cancelled, metrics=best_metrics, n_total=n, n_train=n_train, n_val=n_val,
                             device=str(device), config=cfg, base_name=base_name, base_sha256=base_sha,
                             notes=list(td.notes))
        emit({"type": "done", "best_epoch": best_epoch, "epochs_run": epochs_run, "cancelled": cancelled,
              "metrics": best_metrics})
        return result
    finally:
        if cfg.num_threads:
            torch.set_num_threads(old_threads)


def _metrics_from(ev: dict) -> dict:
    out: dict[str, Any] = {"train_loss": round(ev["train_loss"], 5)}
    if ev.get("val_loss") is not None:
        out["val_loss"] = round(ev["val_loss"], 5)
        out["val_acc"] = round(ev["val_acc"], 5)
    for k, v in (ev.get("mae") or {}).items():
        if v is not None:
            out[f"val_mae_{k}"] = round(v, 5)
    if ev.get("focus_acc") is not None:
        out["val_focus_acc"] = round(ev["focus_acc"], 5)
    return out
