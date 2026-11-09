"""Точка входа PyInstaller для командной строки тренера."""

import multiprocessing
import sys

from therealcam_trainer.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
