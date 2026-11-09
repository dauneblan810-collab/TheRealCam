"""Командная строка: ``python -m therealcam_trainer <команда>`` или ``therealcam-trainer <команда>``.

torch и tkinter импортируются только командами, которым они нужны.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from collections import Counter
from pathlib import Path

from . import __version__
from .errors import TrainerError

EPILOG = """примеры:
  therealcam-trainer inspect-dataset ~/TheRealCam
  therealcam-trainer train ~/TheRealCam/dataset -o my-model.trcm --name "Моя модель"
  therealcam-trainer finetune dataset.zip --base friend.trcm -o my-model-v2.trcm
  therealcam-trainer inspect-model my-model.trcm
  therealcam-trainer eval ~/TheRealCam --model my-model.trcm
  therealcam-trainer gui
"""


def _add_train_args(p: argparse.ArgumentParser, finetune: bool) -> None:
    p.add_argument("path", help="папка датасета (dataset/, TheRealCam/ или выше) или ZIP-экспорт из приложения")
    if finetune:
        p.add_argument("--base", required=True, help="базовая модель .trcm, которую дообучаем")
    p.add_argument("-o", "--output", required=True, help="куда сохранить модель .trcm")
    p.add_argument("--name", help="имя модели (по умолчанию — имя файла)")
    p.add_argument("--description", default="", help="описание модели")
    p.add_argument("--author", default="", help="автор")
    p.add_argument("--epochs", type=int, help=f"число эпох (по умолчанию {30 if finetune else 60})")
    p.add_argument("--batch-size", type=int, default=64, help="размер батча (64)")
    p.add_argument("--lr", type=float, help=f"скорость обучения (по умолчанию {3e-4 if finetune else 2e-3:g})")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps (auto: cuda > mps > cpu)")
    p.add_argument("--seed", type=int, default=0, help="seed для воспроизводимости")
    p.add_argument("--allow-small", action="store_true", help="разрешить датасет меньше 150 снимков (тесты/демо)")
    p.add_argument("--delta-ev", type=float, default=0.7, help="шаг поправки экспозиции по оценкам, EV (0.7)")
    p.add_argument("--patience", type=int, default=12, help="ранняя остановка: эпох без улучшения val loss (12)")
    p.add_argument("--val-fraction", type=float, default=0.15, help="доля валидации (0.15)")
    p.add_argument("--threads", type=int, help="число потоков CPU для PyTorch")
    if finetune:
        p.add_argument("--refit-norm", action="store_true",
                       help="пересчитать нормализацию признаков/параметров по новому датасету")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="therealcam-trainer",
        description="Тренер моделей TheRealCam: обучение с нуля на вашем датасете и экспорт в .trcm",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"therealcam-trainer {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="команда")

    p = sub.add_parser("inspect-dataset", help="проверить датасет и показать статистику")
    p.add_argument("path", help="папка датасета или ZIP")
    p.add_argument("--delta-ev", type=float, default=0.7, help="шаг поправки экспозиции, EV")
    p.add_argument("--json", action="store_true", help="вывести отчёт в JSON")
    p.set_defaults(func=cmd_inspect_dataset)

    p = sub.add_parser("train", help="обучить новую модель с нуля")
    _add_train_args(p, finetune=False)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("finetune", help="дообучить существующую модель .trcm")
    _add_train_args(p, finetune=True)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("inspect-model", help="показать содержимое файла .trcm и проверить его")
    p.add_argument("file", help="файл .trcm")
    p.add_argument("--header", action="store_true", help="вывести весь JSON-заголовок")
    p.set_defaults(func=cmd_inspect_model)

    p = sub.add_parser("eval", help="оценить модель на датасете (эталонным интерпретатором, как на телефоне)")
    p.add_argument("path", help="папка датасета или ZIP")
    p.add_argument("--model", required=True, help="файл .trcm")
    p.add_argument("--delta-ev", type=float, default=0.7, help="шаг поправки экспозиции, EV")
    p.add_argument("--limit", type=int, help="оценить только первые N снимков")
    p.add_argument("--json", action="store_true", help="вывести результат в JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("make-golden", help="сгенерировать эталонные векторы для теста приложения")
    p.add_argument("-o", "--output", default="testdata/golden", help="папка (testdata/golden)")
    p.set_defaults(func=cmd_make_golden)

    p = sub.add_parser("synth-dataset", help="создать СИНТЕТИЧЕСКИЙ датасет (только для тестов/демо)")
    p.add_argument("out_dir", help="куда записать (будет создана папка dataset/)")
    p.add_argument("--n", type=int, default=200, help="число снимков (200)")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_synth_dataset)

    p = sub.add_parser("gui", help="открыть графический интерфейс")
    p.set_defaults(func=cmd_gui)
    return parser


# ---------------------------------------------------------------------------


def cmd_inspect_dataset(args) -> int:
    from .dataset import dataset_report, format_report, load_dataset

    rep = dataset_report(load_dataset(args.path), delta_ev=args.delta_ev)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(format_report(rep))
    return 0


def _run_training(args, finetune: bool) -> int:
    from .train import TrainConfig, format_epoch, train_model

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, seed=args.seed,
                      allow_small=args.allow_small, delta_ev=args.delta_ev, patience=args.patience,
                      val_fraction=args.val_fraction, num_threads=args.threads,
                      base_model=args.base if finetune else None, refit_norm=getattr(args, "refit_norm", False))
    out = Path(args.output)
    cancel = threading.Event()

    def on_sigint(signum, frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        print("\nОстанавливаю после текущего батча и сохраняю лучшую модель (ещё раз Ctrl+C — выход без сохранения)",
              file=sys.stderr, flush=True)

    def progress(ev: dict) -> None:
        if ev["type"] == "log":
            print(ev["message"], flush=True)
        elif ev["type"] == "epoch":
            print(format_epoch(ev), flush=True)

    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGINT, on_sigint)
    try:
        res = train_model(args.path, cfg, progress=progress, cancel=cancel)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    blob = res.export(out, name=args.name or out.stem, description=args.description, author=args.author)
    print(f"Модель сохранена: {out} ({len(blob) / 1024:.1f} КБ): классов {len(res.spec.classes)}, "
          f"признаков {len(res.spec.feature_names)}, эпох {res.epochs_run}, лучшая {res.best_epoch}")
    if res.metrics:
        print("Метрики лучшей эпохи: " + ", ".join(f"{k} {v}" for k, v in res.metrics.items()))
    print("Скопируйте файл на телефон и импортируйте его в настройках приложения TheRealCam.")
    return 0


def cmd_train(args) -> int:
    return _run_training(args, finetune=False)


def cmd_finetune(args) -> int:
    return _run_training(args, finetune=True)


def cmd_inspect_model(args) -> int:
    from .reference import ReferenceModel
    from .schema import icon_profile
    from .trcm import read_trcm

    path = Path(args.file)
    trcm = read_trcm(path)  # проверяет раскладку, лимиты и CRC
    h = trcm.header
    if args.header:
        print(json.dumps(h, ensure_ascii=False, indent=2))
        return 0
    print(f"Файл: {path} ({path.stat().st_size} байт), данных {trcm.data_floats} float, CRC-32 данных: OK")
    print(f"Имя: {h.get('name')!s}" + (f" — {h['description']}" if h.get("description") else ""))
    if h.get("author"):
        print(f"Автор: {h['author']}")
    tr = h.get("trainer") or {}
    print(f"Создана: {h.get('createdUtc')}  тренер: {tr.get('name')} {tr.get('version')}")
    training = h.get("training") or {}
    if training:
        print(f"Обучение: снимков {training.get('samples')}, эпох {training.get('epochs')}"
              + (f" (лучшая {training['bestEpoch']})" if training.get("bestEpoch") else "")
              + f", базовая модель: {training.get('baseModel') or '—'}")
        metrics = training.get("metrics") or {}
        if metrics:
            print("Метрики: " + ", ".join(f"{k} {v}" for k, v in metrics.items()))
    try:
        ref = ReferenceModel(trcm)
    except TrainerError as exc:
        print(f"Граф: ОШИБКА — {exc}")
        return 1
    norm = ref.image_norm.get("type", "per_image")
    print(f"Вход: изображение 3×64×64 CHW, нормализация {norm}; признаков {len(ref.feature_specs)} "
          f"(featureSchema {h.get('featureSchema', 1)})")
    for s in ref.feature_specs:
        print(f"  {s['name']:<18} mean {float(s['mean']):>10.4f}  std {float(s['std']):>10.4f}")
    print(f"Классы ({ref.num_classes}):")
    for i, c in enumerate(ref.classes):
        icon = c.get("icon")
        print(f"  {i:>2}  {c.get('name', c['id'])[:28]:<28} значок {icon!s:<13} профиль {icon_profile(icon):<9} [{c['id']}]")
    print(f"Параметры ({ref.num_params}):")
    for s in ref.param_specs:
        print(f"  {s['name']:<16} mean {float(s['mean']):>9.4f}  std {float(s['std']):>8.4f}")
    ops = Counter(n.op for n in ref.nodes)
    print(f"Граф: узлов {len(ref.nodes)} — " + ", ".join(f"{op}×{c}" if c > 1 else op for op, c in ops.items()))
    print(f"Выходы: class_logits='{ref.output_logits}' [{ref.num_classes}], params='{ref.output_params}' "
          f"[{ref.num_classes * ref.num_params}]")
    print("Проверка графа эталонным интерпретатором: OK")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate_model, format_eval

    def progress(done: int, total: int) -> None:
        if not args.json:
            print(f"\rоценено {done}/{total}", end="", file=sys.stderr, flush=True)

    rep = evaluate_model(args.path, args.model, delta_ev=args.delta_ev, limit=args.limit, progress=progress)
    if not args.json:
        print(file=sys.stderr)
        print(format_eval(rep))
    else:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_make_golden(args) -> int:
    from .golden import make_golden

    info = make_golden(args.output)
    print(f"Записано: {info['model_path']} ({info['model_bytes']} байт), {info['golden_path']} "
          f"({info['cases']} случаев); классов {info['classes']}, признаков {info['features']}; "
          f"макс. расхождение с PyTorch {info['max_abs_diff_torch']:.2e}")
    return 0


def cmd_synth_dataset(args) -> int:
    from .synth import generate_synthetic_dataset

    root = generate_synthetic_dataset(args.out_dir, n=args.n, seed=args.seed)
    print(f"СИНТЕТИЧЕСКИЙ датасет ({args.n} снимков) записан в {root}")
    print("Он только для тестов и демонстрации — настоящую модель обучайте на своих снимках.")
    return 0


def cmd_gui(args) -> int:
    from .gui import main as gui_main

    return gui_main([])


def _safe_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except TrainerError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
