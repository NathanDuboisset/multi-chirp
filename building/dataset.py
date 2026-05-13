"""Symlink target + non_target clips into a train/val/test collection."""

from __future__ import annotations

import random
import re
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
AUDIOSET_DIR = RAW_DATASET_DIR / "audioset"
NON_TARGET_OTHER_DIR = SUBSAMPLES_DIR / "non_target_other"
# Clips passed to BirdNET with no bird detected — kept on-disk under
# non_target_empty/ for historical reasons; semantically these are
# "birdnet-no-bird" clips, used as a quiet/ambient negative source.
BIRDNET_NO_BIRD_DIR = SUBSAMPLES_DIR / "non_target_empty"
DATASETS_DIR = ROOT / "datasets"

SPLIT_NAMES = ("training", "validation", "testing")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 42

# Clips from the same XC recording must stay in the same split.
_XC_GROUP_RE = re.compile(r"(?:^|__)(XC\d+)_")


def group_key(clip_path: Path) -> str:
    m = _XC_GROUP_RE.search(clip_path.name)
    return m.group(1) if m else clip_path.name


def audioset_round_robin() -> list[Path]:
    pools = [
        sorted(d.glob("*.wav"))
        for d in sorted(AUDIOSET_DIR.iterdir())
        if d.is_dir()
    ] if AUDIOSET_DIR.exists() else []
    pools = [p for p in pools if p]
    if not pools:
        return []
    cursors = [0] * len(pools)
    out: list[Path] = []
    while True:
        progressed = False
        for i, pool in enumerate(pools):
            if cursors[i] < len(pool):
                out.append(pool[cursors[i]])
                cursors[i] += 1
                progressed = True
        if not progressed:
            return out


def balanced_non_target(other_species_folders: list[Path]) -> list[Path]:
    """Build the non_target pool: 50/50 audioset + xc_other, plus all
    birdnet-no-bird on top. The audioset and xc_other buckets are each
    capped at the size of the smaller of the two so they contribute equally;
    birdnet-no-bird is taken in full since it's a small, distinct source.
    """
    audioset_pool = audioset_round_robin()
    other_pool: list[Path] = []
    for folder in other_species_folders:
        other_pool.extend(sorted(folder.glob("*.wav")))
    no_bird_pool = (
        sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
        if BIRDNET_NO_BIRD_DIR.exists()
        else []
    )

    n_each = min(len(audioset_pool), len(other_pool))
    out: list[Path] = []
    out.extend(audioset_pool[:n_each])
    out.extend(other_pool[:n_each])
    out.extend(no_bird_pool)
    print(
        f"Non-target pool: {len(out)} clips "
        f"({n_each} audioset + {n_each} xc_other + {len(no_bird_pool)} birdnet_no_bird)"
    )
    print(f"  audioset:         take {n_each} of {len(audioset_pool)} available")
    print(f"  xc_other:         take {n_each} of {len(other_pool)} available")
    print(f"  birdnet_no_bird:  take {len(no_bird_pool)} of {len(no_bird_pool)} available")
    return out


def split_class(clips: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for clip in clips:
        groups.setdefault(group_key(clip), []).append(clip)
    keys = sorted(groups.keys())
    rng.shuffle(keys)

    n = len(clips)
    targets = {
        "training": int(n * SPLIT_RATIOS[0]),
        "validation": int(n * SPLIT_RATIOS[1]),
        "testing": n - int(n * SPLIT_RATIOS[0]) - int(n * SPLIT_RATIOS[1]),
    }
    buckets: dict[str, list[Path]] = {s: [] for s in SPLIT_NAMES}
    for g in keys:
        pick = max(buckets, key=lambda b: targets[b] - len(buckets[b]))
        buckets[pick].extend(groups[g])
    return buckets


def build_task_dataset(
    collection_name: str,
    target_species: list[str],
    non_target_species: list[str],
) -> Path:
    dataset_root = DATASETS_DIR / collection_name
    for s in SPLIT_NAMES:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    class_to_clips: dict[str, list[Path]] = {}
    for name in target_species:
        folder = name.replace(" ", "_")
        clips = sorted((SUBSAMPLES_DIR / folder).glob("*.wav"))
        class_to_clips[folder] = clips

    non_target_folders = [
        SUBSAMPLES_DIR / n.replace(" ", "_") for n in non_target_species
    ]
    non_target = balanced_non_target(other_species_folders=non_target_folders)
    if non_target:
        class_to_clips["non_target"] = non_target

    rng = random.Random(SPLIT_SEED)
    for folder, all_clips in class_to_clips.items():
        print(f"Processing {folder} with {len(all_clips)} clips")
        buckets = split_class(all_clips, rng)
        print(
            f"Training: {len(buckets['training'])}, "
            f"Validation: {len(buckets['validation'])}, "
            f"Testing: {len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            files = buckets[split_name]
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(files):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())
            print(f"Copied {len(files)} clips from {folder} to {split_name}")

    return dataset_root


def build_dataset(
    collection_name: str,
    species_names: list[str],
) -> Path:
    target_keys = {n.replace(" ", "_") for n in species_names}
    dataset_root = DATASETS_DIR / collection_name
    for s in SPLIT_NAMES:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    class_to_clips: dict[str, list[Path]] = {}
    for name in species_names:
        folder = name.replace(" ", "_")
        clips = sorted((SUBSAMPLES_DIR / folder).glob("*.wav"))
        class_to_clips[folder] = clips

    non_target_clips = (
        audioset_round_robin()
        + [
            p
            for p in sorted(NON_TARGET_OTHER_DIR.glob("*.wav"))
            if not any(p.name.startswith(f"{k}__") for k in target_keys)
        ]
        + sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
    )
    if non_target_clips:
        class_to_clips["non_target"] = non_target_clips

    rng = random.Random(SPLIT_SEED)
    for folder, all_clips in class_to_clips.items():
        print(f"Processing {folder} with {len(all_clips)} clips")
        buckets = split_class(all_clips, rng)
        print(
            f"Training: {len(buckets['training'])}, "
            f"Validation: {len(buckets['validation'])}, "
            f"Testing: {len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            files = buckets[split_name]
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(files):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())
            print(f"Copied {len(files)} clips from {folder} to {split_name}")

    return dataset_root
