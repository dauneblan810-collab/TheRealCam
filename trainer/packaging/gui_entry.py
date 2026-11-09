"""Точка входа PyInstaller для окна тренера."""

import multiprocessing
import sys

from therealcam_trainer.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
