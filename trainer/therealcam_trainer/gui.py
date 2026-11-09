"""Графический интерфейс тренера (tkinter).

Обучение идёт в фоновом потоке; поток только кладёт события в очередь, а окно забирает их
через ``after()`` — вызовы tkinter из других потоков небезопасны.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:  # Linux без пакета python3-tk
    tk = None

POLL_MS = 100
SCRATCH_DEFAULTS = ("60", "0.002")
FINETUNE_DEFAULTS = ("30", "0.0003")


class TrainerApp:
    def __init__(self, root: "tk.Tk"):
        self.root = root
        root.title("TheRealCam — тренер моделей")
        root.minsize(780, 600)
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()

        self.dataset_var = tk.StringVar()
        self.base_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.home() / "therealcam-model.trcm"))
        self.name_var = tk.StringVar(value="Моя модель")
        self.epochs_var = tk.StringVar(value=SCRATCH_DEFAULTS[0])
        self.batch_var = tk.StringVar(value="64")
        self.lr_var = tk.StringVar(value=SCRATCH_DEFAULTS[1])
        self.device_var = tk.StringVar(value="auto")
        self.allow_small_var = tk.BooleanVar(value=False)
        self.refit_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Выберите датасет: папку TheRealCam/dataset или ZIP-экспорт из приложения.")
        self.metrics_var = tk.StringVar(value="—")
        self.base_var.trace_add("write", lambda *_: self._on_base_changed())

        self._build()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(POLL_MS, self._poll)

    # -- интерфейс

    def _build(self) -> None:
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        def path_row(row: int, label: str, var, buttons) -> None:
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(frm, textvariable=var).grid(row=row, column=1, sticky="ew", **pad)
            box = ttk.Frame(frm)
            box.grid(row=row, column=2, sticky="e", **pad)
            for text, cmd in buttons:
                ttk.Button(box, text=text, command=cmd).pack(side="left", padx=(4, 0))

        path_row(0, "Датасет (папка или ZIP):", self.dataset_var,
                 [("Папка…", self._pick_dataset_dir), ("ZIP…", self._pick_dataset_zip)])
        path_row(1, "Базовая модель (дообучение):", self.base_var,
                 [("Выбрать…", self._pick_base), ("Очистить", lambda: self.base_var.set(""))])
        path_row(2, "Сохранить модель в:", self.output_var, [("Обзор…", self._pick_output)])
        ttk.Label(frm, text="Имя модели:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.name_var).grid(row=3, column=1, sticky="ew", **pad)

        params = ttk.Frame(frm)
        params.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(params, text="Эпох:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=1000, width=6, textvariable=self.epochs_var).pack(side="left", padx=(4, 14))
        ttk.Label(params, text="Размер батча:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=1024, width=6, textvariable=self.batch_var).pack(side="left", padx=(4, 14))
        ttk.Label(params, text="Скорость обучения:").pack(side="left")
        ttk.Entry(params, width=9, textvariable=self.lr_var).pack(side="left", padx=(4, 14))
        ttk.Label(params, text="Устройство:").pack(side="left")
        ttk.Combobox(params, width=7, state="readonly", values=("auto", "cpu", "cuda"),
                     textvariable=self.device_var).pack(side="left", padx=(4, 0))

        opts = ttk.Frame(frm)
        opts.grid(row=5, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(opts, text="Разрешить меньше 150 снимков (только для проверки)",
                        variable=self.allow_small_var).pack(side="top", anchor="w")
        ttk.Checkbutton(opts, text="Пересчитать нормализацию по новому датасету (при дообучении)",
                        variable=self.refit_var).pack(side="top", anchor="w")

        buttons = ttk.Frame(frm)
        buttons.grid(row=6, column=0, columnspan=3, sticky="w", **pad)
        self.inspect_btn = ttk.Button(buttons, text="Проверить датасет", command=self.start_inspect)
        self.inspect_btn.pack(side="left")
        self.train_btn = ttk.Button(buttons, text="Обучить", command=self.start_training)
        self.train_btn.pack(side="left", padx=(6, 0))
        self.stop_btn = ttk.Button(buttons, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100.0)
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.status_var).grid(row=8, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(frm, text="Последние метрики:").grid(row=9, column=0, sticky="nw", **pad)
        ttk.Label(frm, textvariable=self.metrics_var, wraplength=600, justify="left").grid(
            row=9, column=1, columnspan=2, sticky="w", **pad)

        self.log = scrolledtext.ScrolledText(frm, height=16, wrap="word", state="disabled")
        self.log.grid(row=10, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(10, weight=1)

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def log_text(self) -> str:
        return self.log.get("1.0", "end")

    # -- выбор файлов

    def _pick_dataset_dir(self) -> None:
        path = filedialog.askdirectory(title="Папка датасета (TheRealCam или TheRealCam/dataset)")
        if path:
            self.dataset_var.set(path)

    def _pick_dataset_zip(self) -> None:
        path = filedialog.askopenfilename(title="ZIP-экспорт датасета", filetypes=[("ZIP", "*.zip"), ("Все", "*.*")])
        if path:
            self.dataset_var.set(path)

    def _pick_base(self) -> None:
        path = filedialog.askopenfilename(title="Базовая модель", filetypes=[("Модель TheRealCam", "*.trcm"),
                                                                            ("Все", "*.*")])
        if path:
            self.base_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Сохранить модель", defaultextension=".trcm",
                                            filetypes=[("Модель TheRealCam", "*.trcm")])
        if path:
            self.output_var.set(path)
            self.name_var.set(Path(path).stem)

    def _on_base_changed(self) -> None:
        finetune = bool(self.base_var.get().strip())
        old, new = (SCRATCH_DEFAULTS, FINETUNE_DEFAULTS) if finetune else (FINETUNE_DEFAULTS, SCRATCH_DEFAULTS)
        if self.epochs_var.get().strip() == old[0]:
            self.epochs_var.set(new[0])
        if self.lr_var.get().strip() == old[1]:
            self.lr_var.set(new[1])

    # -- действия

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _set_busy(self, busy: bool, training: bool = False) -> None:
        state = "disabled" if busy else "normal"
        self.inspect_btn.configure(state=state)
        self.train_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if busy and training else "disabled")

    def _run_worker(self, target, training: bool) -> None:
        self.cancel.clear()
        self._set_busy(True, training)

        def wrapper() -> None:
            try:
                target()
            except Exception as exc:  # показываем пользователю, а не роняем поток молча
                from .errors import TrainerError

                msg = str(exc) if isinstance(exc, TrainerError) else f"{exc.__class__.__name__}: {exc}"
                if not isinstance(exc, TrainerError):
                    self.events.put({"type": "log", "message": traceback.format_exc()})
                self.events.put({"type": "error", "message": msg})
            finally:
                self.events.put({"type": "finished"})

        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def _dataset_path(self) -> str | None:
        path = self.dataset_var.get().strip()
        if not path:
            messagebox.showwarning("Датасет", "Укажите папку датасета или ZIP-архив.")
            return None
        return path

    def start_inspect(self) -> None:
        if self.busy:
            return
        path = self._dataset_path()
        if path is None:
            return
        self.status_var.set("Проверяю датасет…")

        def job() -> None:
            from .dataset import dataset_report, format_report, load_dataset

            rep = dataset_report(load_dataset(path))
            self.events.put({"type": "report", "text": format_report(rep), "ready": rep["ready"],
                             "count": rep["counts"]["used"]})

        self._run_worker(job, training=False)

    def _read_config(self):
        from .train import TrainConfig

        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            lr = float(self.lr_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Параметры", "Эпохи и размер батча — целые числа, скорость обучения — число (например 0.002).")
            return None
        if epochs < 1 or batch < 1 or lr <= 0:
            messagebox.showerror("Параметры", "Эпохи, размер батча и скорость обучения должны быть положительными.")
            return None
        base = self.base_var.get().strip() or None
        return TrainConfig(epochs=epochs, batch_size=batch, lr=lr, device=self.device_var.get() or "auto",
                           allow_small=bool(self.allow_small_var.get()), base_model=base,
                           refit_norm=bool(self.refit_var.get()))

    def start_training(self) -> None:
        if self.busy:
            return
        path = self._dataset_path()
        if path is None:
            return
        out = self.output_var.get().strip()
        if not out:
            messagebox.showwarning("Модель", "Укажите, куда сохранить модель (.trcm).")
            return
        if not out.lower().endswith(".trcm"):
            out += ".trcm"
            self.output_var.set(out)
        cfg = self._read_config()
        if cfg is None:
            return
        name = self.name_var.get().strip() or Path(out).stem
        self.progress.configure(value=0)
        self.metrics_var.set("—")
        self.status_var.set("Загружаю датасет…")
        self._append(("Дообучение" if cfg.base_model else "Обучение с нуля") + f": {path} → {out}")

        def job() -> None:
            from .train import train_model

            res = train_model(path, cfg, progress=self.events.put, cancel=self.cancel)
            blob = res.export(out, name=name)
            self.events.put({"type": "saved", "path": out, "bytes": len(blob), "classes": len(res.spec.classes),
                             "best_epoch": res.best_epoch, "cancelled": res.cancelled})

        self._run_worker(job, training=True)

    def stop(self) -> None:
        if self.busy:
            self.cancel.set()
            self.status_var.set("Останавливаю… лучшая модель будет сохранена после текущего батча.")

    def _on_close(self) -> None:
        if self.busy and not messagebox.askyesno("Выход", "Обучение ещё идёт. Прервать его и закрыть окно?"):
            return
        self.cancel.set()
        self.root.destroy()

    # -- события из фонового потока

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        try:
            self.root.after(POLL_MS, self._poll)
        except tk.TclError:  # окно уже закрыто
            pass

    def _handle(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "log":
            self._append(ev["message"])
        elif kind == "report":
            self._append(ev["text"])
            self.status_var.set(f"Датасет проверен: пригодно {ev['count']} снимков"
                                + ("" if ev["ready"] else " — меньше 150, обучение невозможно"))
        elif kind == "start":
            self.status_var.set(f"Обучение: train {ev['n_train']}, val {ev['n_val']}, устройство {ev['device']}")
        elif kind == "batch":
            frac = (ev["epoch"] - 1 + ev["batch"] / ev["batches"]) / ev["epochs"]
            self.progress.configure(value=100.0 * frac)
        elif kind == "epoch":
            from .train import format_epoch

            line = format_epoch(ev)
            self._append(line)
            self.metrics_var.set(line)
            self.status_var.set(f"Эпоха {ev['epoch']} из {ev['epochs']} (лучшая {ev['best_epoch']})")
        elif kind == "saved":
            self.progress.configure(value=100.0)
            msg = (f"Модель сохранена: {ev['path']} ({ev['bytes'] / 1024:.1f} КБ, классов {ev['classes']}, "
                   f"лучшая эпоха {ev['best_epoch']}). Скопируйте файл на телефон и импортируйте в настройках.")
            self._append(msg)
            self.status_var.set("Готово" + (" (остановлено пользователем)" if ev["cancelled"] else ""))
            messagebox.showinfo("Готово", msg)
        elif kind == "error":
            self._append("Ошибка: " + ev["message"])
            self.status_var.set("Ошибка")
            messagebox.showerror("Ошибка", ev["message"])
        elif kind == "finished":
            self._set_busy(False)


def _enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main(argv=None) -> int:
    if tk is None:
        print("Ошибка: не найден модуль tkinter. Linux: sudo apt install python3-tk; "
              "Windows/macOS: установите Python с python.org (tkinter входит в состав).", file=sys.stderr)
        return 1
    _enable_dpi_awareness()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Ошибка: не удалось открыть окно ({exc}). Нужен графический сеанс; без экрана используйте команды CLI.",
              file=sys.stderr)
        return 1
    TrainerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
