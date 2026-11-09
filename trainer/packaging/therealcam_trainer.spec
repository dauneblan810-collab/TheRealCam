# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: сборка «одной папкой» (one-folder).
#
#   cd trainer
#   pip install pyinstaller
#   pyinstaller packaging/therealcam_trainer.spec
#
# Результат: dist/TheRealCamTrainer/
#   therealcam-trainer-gui(.exe)  — окно тренера (без консоли)
#   therealcam-trainer(.exe)      — командная строка (те же команды, что и у python -m therealcam_trainer)
# Собирать нужно на той ОС, для которой делается сборка (Windows-сборка — на Windows, Linux — на Linux).
# В сборку попадает тот PyTorch, что установлен в окружении: CPU-сборка torch — маленький дистрибутив,
# CUDA-сборка — большой (несколько ГБ), зато обучает на видеокарте NVIDIA.

import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

common = dict(
    pathex=[ROOT],
    hiddenimports=collect_submodules("therealcam_trainer") + ["tkinter", "tkinter.ttk", "tkinter.scrolledtext"],
    excludes=["matplotlib", "IPython", "pytest", "torchvision", "torchaudio", "pandas", "scipy"],
    noarchive=False,
)

gui_a = Analysis([os.path.join(SPECPATH, "gui_entry.py")], **common)
cli_a = Analysis([os.path.join(SPECPATH, "cli_entry.py")], **common)

gui_exe = EXE(
    PYZ(gui_a.pure),
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="therealcam-trainer-gui",
    console=False,
)
cli_exe = EXE(
    PYZ(cli_a.pure),
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="therealcam-trainer",
    console=True,
)

coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    name="TheRealCamTrainer",
)
