"""Repository paths anchored on pyproject.toml so notebooks at any depth resolve correctly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyrootutils

REPO_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
MODELS_DIR = REPO_ROOT / "models"
SRC_DIR = REPO_ROOT / "src"
DATASET_ROOT = REPO_ROOT / "dataset"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPaths:
    out_tflite: Path
    out_audio_rs: Path


def get_paths(model_stem: str) -> ModelPaths:
    return ModelPaths(
        out_tflite=MODELS_DIR / f"{model_stem}.tflite",
        out_audio_rs=SRC_DIR / "audio_sample.rs",
    )
