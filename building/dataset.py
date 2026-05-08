"""
Symlink-based train/val/test split from ``raw_dataset/subsamples/<collection>/``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pyrootutils

# Clips named ``XC<id>_<idx>.wav`` come from a single Xeno-Canto recording and
# must stay together when splitting; everything else (audioset, legacy
# ``clip_<idx>.wav``) is treated as one-clip-per-group, i.e. clip-level split.
_XC_RE = re.compile(r"^XC(\d+)_")


def _group_key(clip_path: Path) -> str:
    m = _XC_RE.match(clip_path.name)
    if m:
        return f"XC{m.group(1)}"
    return clip_path.name

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


def _budget_non_target(
    sdir: Path,
    target_total: int,
    mix: tuple[float, float, float],
    audioset_dir_name: str,
) -> list[Path]:
    """Concatenate non_target clips across (audioset, xc_other, xc_empty) buckets
    according to ``mix``, falling back to higher-priority buckets if any one is short.
    Priority order matches the bucket list (audioset first)."""
    buckets = [
        (sdir / audioset_dir_name, mix[0]),
        (sdir / "non_target_other", mix[1]),
        (sdir / "non_target_empty", mix[2]),
    ]
    quotas = [int(target_total * frac) for _, frac in buckets]
    available = [sorted(d.glob("*.wav")) if d.exists() else [] for d, _ in buckets]

    chosen: list[Path] = []
    leftover = 0
    for (bucket_dir, _), quota, pool in zip(buckets, quotas, available):
        take = min(quota + leftover, len(pool))
        chosen.extend(pool[:take])
        deficit = (quota + leftover) - take
        if deficit > 0:
            print(
                f"  non_target bucket {bucket_dir.name}: short by {deficit} "
                f"(wanted {quota + leftover}, had {len(pool)})"
            )
        leftover = deficit
    return chosen


def build_dataset(
    collection_name: str,
    species_names: list[str],
    clips_per_species: int | None = None,
    max_class_size_training: int | None = None,
    split: tuple[float, float, float] = SPLIT_RATIOS,
    subsamples_dir: Path | None = None,
    audioset_dir_name: str = "non_target_audioset",
    non_target_mix: tuple[float, float, float] = (0.70, 0.20, 0.10),
    non_target_total: int | None = None,
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

    target_nt = non_target_total or clips_per_species
    if target_nt:
        non_target_clips = _budget_non_target(
            sdir, target_nt, non_target_mix, audioset_dir_name
        )
    else:
        non_target_clips = (
            sorted((sdir / audioset_dir_name).glob("*.wav"))
            + sorted((sdir / "non_target_other").glob("*.wav"))
            + sorted((sdir / "non_target_empty").glob("*.wav"))
        )
    if non_target_clips:
        class_to_clips["non_target"] = non_target_clips

    min_class_size = min(len(clips) for clips in class_to_clips.values())
    for i_class, clips in class_to_clips.items():
        if len(clips) > min_class_size:
            clips = clips[:min_class_size]
            print(f"Truncated {i_class} to {min_class_size} clips")

    for folder, all_clips in class_to_clips.items():
        print(f"Processing {folder} with {len(all_clips)} clips")
        groups: dict[str, list[Path]] = {}
        for clip in all_clips:
            groups.setdefault(_group_key(clip), []).append(clip)

        rng = random.Random(42)
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n = len(all_clips)
        targets = {
            "training": int(n * split[0]),
            "validation": int(n * split[1]),
            "testing": n - int(n * split[0]) - int(n * split[1]),
        }
        buckets: dict[str, list[Path]] = {s: [] for s in SPLIT_NAMES}
        # Greedy: assign each group to whichever bucket is most under quota,
        # which keeps split ratios close to ``split`` even with variable group
        # sizes.
        for g in group_keys:
            g_clips = groups[g]
            pick = max(buckets, key=lambda b: targets[b] - len(buckets[b]))
            buckets[pick].extend(g_clips)
        print(
            f"Training: {len(buckets['training'])}, "
            f"Validation: {len(buckets['validation'])}, "
            f"Testing: {len(buckets['testing'])} "
            f"(groups={len(group_keys)})"
        )
        for split_name, files in zip(
            SPLIT_NAMES,
            [buckets["training"], buckets["validation"], buckets["testing"]],
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
