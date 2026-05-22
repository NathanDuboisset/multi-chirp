"""Place-based dataset manifest (``dataset.json``).

Written by ``dataset_build.ipynb`` *before* any download so downstream
notebooks (analysis, training, results) read one small file instead of
re-deriving the species list from BirdNET + XC every time.
"""

from __future__ import annotations

from pathlib import Path

import pyrootutils
from pydantic import BaseModel, ConfigDict, Field

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

DATASETS_DIR = ROOT / "datasets"


class Place(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    lat: float
    lon: float
    radius_km: float


class SplitRatios(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training: float = 0.7
    validation: float = 0.15
    testing: float = 0.15


class Audio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_rate: int = 16_000
    clip_duration_s: float = 3.0


class Budgets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_clips_per_species: int
    non_target_clips_per_species: int
    non_target_per_species_cap: int
    audioset_clips_per_class: int
    audioset_max_total: int


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str
    place: Place
    n_targets: int
    target_species: list[str]
    other_species: list[str]
    audio: Audio = Field(default_factory=Audio)
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    split_seed: int = 42
    budgets: Budgets


def info_path(collection: str) -> Path:
    return DATASETS_DIR / collection / "dataset.json"


def write_dataset_info(info: DatasetInfo) -> Path:
    path = info_path(info.collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(info.model_dump_json(indent=2))
    return path


def load_dataset_info(collection: str) -> DatasetInfo:
    return DatasetInfo.model_validate_json(info_path(collection).read_text())
