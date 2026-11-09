import json
from pathlib import Path

import numpy as np
import pytest
import torch

from therealcam_trainer.dataset import load_dataset
from therealcam_trainer.synth import generate_synthetic_dataset
from therealcam_trainer.train import quiet_torch

torch.set_num_threads(2)
quiet_torch()

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    """СИНТЕТИЧЕСКИЙ датасет на 200 снимков (OUT/dataset/...), общий для всех тестов."""
    out = tmp_path_factory.mktemp("synth200")
    generate_synthetic_dataset(out, n=200, seed=1)
    return out


@pytest.fixture(scope="session")
def synth_ds(synth_dir):
    return load_dataset(synth_dir)


@pytest.fixture
def small_synth(tmp_path) -> Path:
    """Маленький датасет без JPEG — его можно портить в тестах."""
    generate_synthetic_dataset(tmp_path, n=30, seed=5, write_images=False)
    return tmp_path / "dataset"


def edit_meta(sample_dir: Path, fn) -> None:
    path = sample_dir / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    fn(meta)
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def random_thumb(rng) -> np.ndarray:
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
