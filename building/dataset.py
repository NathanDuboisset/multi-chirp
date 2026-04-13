"""
Symlink-based train/val/test split from ``raw_dataset/subsamples/<collection>/``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

RAW_DATASET_DIR = ROOT / "raw_dataset"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
DATASETS_DIR = ROOT / "datasets"
SPLIT_NAMES = ("training", "validation", "testing")
SPLIT_RATIOS = (0.7, 0.15, 0.15)


def build_dataset(
    collection_name: str,
    species_names: list[str],
    clips_per_species: int | None = None,
    max_class_size_training: int | None = None,
    split: tuple[float, float, float] = SPLIT_RATIOS,
    subsamples_dir: Path | None = None,
) -> Path:
    sdir = subsamples_dir or (SUBSAMPLES_DIR / collection_name)
    dataset_root = DATASETS_DIR / collection_name
    for s in SPLIT_NAMES:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    class_to_clips: dict[str, list[Path]] = {}
    for name in species_names:
        folder = name.replace(" ", "_")
        all_clips = sorted((sdir / folder).glob("*.wav"))
        if clips_per_species:
            all_clips = all_clips[:clips_per_species]
        class_to_clips[folder] = list(all_clips)

    non_target_clips = sorted((sdir / "non_target_other").glob("*.wav")) + sorted(
        (sdir / "non_target_empty").glob("*.wav")
    )
    if non_target_clips:
        class_to_clips["non_target"] = non_target_clips
    if clips_per_species:
        class_to_clips["non_target"] = class_to_clips["non_target"][:clips_per_species]

    min_class_size = min(len(clips) for clips in class_to_clips.values())
    for i_class, clips in class_to_clips.items():
        if len(clips) > min_class_size:
            clips = clips[:min_class_size]
            print(f"Truncated {i_class} to {min_class_size} clips")

    for folder, all_clips in class_to_clips.items():
        print(f"Processing {folder} with {len(all_clips)} clips")
        rng = random.Random(42)
        shuffled = list(all_clips)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        print(
            f"Training: {n_train}, Validation: {n_val}, Testing: {n - n_train - n_val}"
        )
        for split_name, files in zip(
            SPLIT_NAMES,
            [
                shuffled[:n_train],
                shuffled[n_train : n_train + n_val],
                shuffled[n_train + n_val :],
            ],
        ):
            if max_class_size_training is not None and split_name == "training":
                files = files[:max_class_size_training]
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            n_copy = 0
            for file_path in files:
                target = dest / f"{split_name}_{n_copy}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())
                n_copy += 1
            print(f"Copied {n_copy} clips from {folder} to {split_name}")

    return dataset_root
