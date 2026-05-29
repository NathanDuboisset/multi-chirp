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
BIRDNET_NO_BIRD_DIR = SUBSAMPLES_DIR / "non_target_empty"
DATASETS_DIR = ROOT / "datasets"

SPLIT_NAMES = ("training", "validation", "testing")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 42

# Clips from the same recording must stay in the same split, across all sources:
#   XenoCanto:  [<species>__]XC<id>_<window>.wav        -> group on XC<id>
#   Macaulay:   [<species>__]ML<id>_<window>.wav        -> group on ML<id>
#   AudioSet:   <ytid>.wav  or  <ytid>_<offset>.wav     -> group on YT<ytid>
# YouTube IDs are exactly 11 chars from [A-Za-z0-9_-].
_XC_GROUP_RE = re.compile(r"(?:^|__)(XC\d+)_")
_ML_GROUP_RE = re.compile(r"(?:^|__)(ML\d+)_")
_AS_GROUP_RE = re.compile(r"^([A-Za-z0-9_-]{11})(?:_\d+)?$")


def group_key(clip_path: Path) -> str:
    name = clip_path.name
    m = _XC_GROUP_RE.search(name)
    if m:
        return m.group(1)
    m = _ML_GROUP_RE.search(name)
    if m:
        return m.group(1)
    m = _AS_GROUP_RE.match(clip_path.stem)
    if m:
        return f"YT{m.group(1)}"
    return name


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


def balanced_non_target(
    other_species_folders: list[Path],
    per_species_cap: int | None = None,
) -> list[Path]:
    audioset_pool = audioset_round_robin()
    other_pool: list[Path] = []
    for folder in other_species_folders:
        clips = sorted(folder.glob("*.wav"))
        if per_species_cap is not None:
            clips = clips[:per_species_cap]
        other_pool.extend(clips)
    no_bird_pool = (
        sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
        if BIRDNET_NO_BIRD_DIR.exists()
        else []
    )

    n_each = min(len(audioset_pool), len(other_pool))
    out = audioset_pool[:n_each] + other_pool[:n_each] + no_bird_pool
    cap_note = f" cap={per_species_cap}" if per_species_cap is not None else ""
    print(
        f"non_target: {len(out)} = {n_each}/{len(audioset_pool)} audioset "
        f"+ {n_each}/{len(other_pool)} xc_other + {len(no_bird_pool)} no_bird{cap_note}"
    )
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
    non_target_per_species_cap: int | None = None,
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
    non_target = balanced_non_target(
        other_species_folders=non_target_folders,
        per_species_cap=non_target_per_species_cap,
    )
    if non_target:
        class_to_clips["non_target"] = non_target

    rng = random.Random(SPLIT_SEED)
    for folder, all_clips in class_to_clips.items():
        buckets = split_class(all_clips, rng)
        print(
            f"{folder}: total={len(all_clips)} "
            f"train={len(buckets['training'])} "
            f"val={len(buckets['validation'])} "
            f"test={len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(buckets[split_name]):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())

    return dataset_root


def build_cascading_dataset(
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

    non_target_bird: list[Path] = []
    for sp in non_target_species:
        folder = SUBSAMPLES_DIR / sp.replace(" ", "_")
        if folder.exists():
            non_target_bird.extend(sorted(folder.glob("*.wav")))
    if non_target_bird:
        class_to_clips["non_target_bird"] = non_target_bird

    no_bird = audioset_round_robin() + (
        sorted(BIRDNET_NO_BIRD_DIR.glob("*.wav"))
        if BIRDNET_NO_BIRD_DIR.exists()
        else []
    )
    if no_bird:
        class_to_clips["no_bird"] = no_bird

    print(f"cascading pools: non_target_bird={len(non_target_bird)} no_bird={len(no_bird)}")

    rng = random.Random(SPLIT_SEED)
    for folder, all_clips in class_to_clips.items():
        buckets = split_class(all_clips, rng)
        print(
            f"{folder}: total={len(all_clips)} "
            f"train={len(buckets['training'])} "
            f"val={len(buckets['validation'])} "
            f"test={len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(buckets[split_name]):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())

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
        buckets = split_class(all_clips, rng)
        print(
            f"{folder}: total={len(all_clips)} "
            f"train={len(buckets['training'])} "
            f"val={len(buckets['validation'])} "
            f"test={len(buckets['testing'])}"
        )
        for split_name in SPLIT_NAMES:
            dest = dataset_root / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for i, file_path in enumerate(buckets[split_name]):
                target = dest / f"{split_name}_{i}.wav"
                if not target.exists():
                    target.symlink_to(file_path.resolve())

    return dataset_root
