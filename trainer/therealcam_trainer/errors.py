"""Исключения тренера (без зависимостей, чтобы CLI не тянул torch ради --help)."""


class TrainerError(Exception):
    """Базовая ошибка с понятным пользователю сообщением."""


class TrcmError(TrainerError, ValueError):
    """Файл .trcm повреждён, не поддерживается или нарушает спецификацию."""


class DatasetError(TrainerError, ValueError):
    """Датасет не найден или не читается."""


class TrainError(TrainerError, RuntimeError):
    """Обучение невозможно с заданными данными или настройками."""
