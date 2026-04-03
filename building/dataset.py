"""
BirdNET on ``raw_dataset/species``, clips under ``raw_dataset/subsamples/<collection>/``,
symlink splits under ``datasets/<collection>/``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer as ModelAnalyzer
import pyrootutils
from tqdm import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

RAW_DATASET_DIR = ROOT / "raw_dataset"
SPECIES_DIR = RAW_DATASET_DIR / "species"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
PROCESSED_CSV_NAME = "processed_recordings.csv"
DATASETS_DIR = ROOT / "datasets"

BIRDNET_THRESHOLD = 0.92
CLIP_DURATION = 3.0
TARGET_SAMPLE_RATE = 16_000
SPLIT = (0.7, 0.15, 0.15)

_analyzer: ModelAnalyzer | None = None


def process_species(
    scientific_name: str,
    collection_name: str,
    threshold: float = BIRDNET_THRESHOLD,
) -> list[Path]:
    global _analyzer
    if _analyzer is None:
        _analyzer = ModelAnalyzer()

    folder_name = scientific_name.replace(" ", "_")
    raw_dir = SPECIES_DIR / folder_name
    species_out = SUBSAMPLES_DIR / collection_name / folder_name
    species_out.mkdir(parents=True, exist_ok=True)

    csv_path = SUBSAMPLES_DIR / collection_name / PROCESSED_CSV_NAME
    processed: set[tuple[str, str]] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s = (row.get("species") or "").strip()
                r = (row.get("recording") or "").strip()
                if s and r:
                    processed.add((s, r))

    existing = list(species_out.glob("clip_*.wav"))
    clip_idx = max(int(p.stem.split("_", 1)[1]) for p in existing) + 1 if existing else 0

    audio_files = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))
    clips: list[Path] = []

    for audio_path in tqdm(audio_files, desc=f"  BirdNET {collection_name}/{scientific_name}", leave=False):
        if (folder_name, audio_path.name) in processed:
            continue
        try:
            rec = Recording(_analyzer, str(audio_path), min_conf=threshold)
            rec.analyze()
        except Exception as e:
            print(f"    skip recording {audio_path.name}: {e}")
            continue
        name_lower = scientific_name.lower()
        detections = [
            d for d in rec.detections
            if name_lower in d.get("scientific_name", "").lower()
        ]
        for det in detections:
            start = float(det.get("start_time", 0.0))
            clip_path = species_out / f"clip_{clip_idx:05d}.wav"
            try:
                if not clip_path.exists():
                    sig, _ = librosa.load(
                        str(audio_path),
                        sr=TARGET_SAMPLE_RATE,
                        offset=start,
                        duration=CLIP_DURATION,
                        mono=True,
                        res_type="kaiser_fast",
                    )
                    target_len = int(CLIP_DURATION * TARGET_SAMPLE_RATE)
                    if len(sig) < target_len:
                        sig = np.pad(sig, (0, target_len - len(sig)))
                    else:
                        sig = sig[:target_len]
                    sf.write(str(clip_path), sig, TARGET_SAMPLE_RATE, subtype="PCM_16")
            except Exception as e:
                print(f"    skip clip {audio_path.name} @ {start:g}s: {e}")
                continue
            clips.append(clip_path)
            clip_idx += 1

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        append_new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["species", "recording"])
            if append_new:
                w.writeheader()
            w.writerow({"species": folder_name, "recording": audio_path.name})

    return clips


def build_dataset(
    collection_name: str,
    species_names: list[str],
    clips_per_species: int | None = None,
    split: tuple[float, float, float] = SPLIT,
    subsamples_dir: Path | None = None,
) -> Path:
    sdir = subsamples_dir or (SUBSAMPLES_DIR / collection_name)
    dataset_root = DATASETS_DIR / collection_name
    splits = ("training", "validation", "testing")
    for s in splits:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    for name in species_names:
        rng = random.Random(42)
        folder = name.replace(" ", "_")
        all_clips = sorted((sdir / folder).glob("*.wav"))
        if clips_per_species:
            all_clips = all_clips[:clips_per_species]
        shuffled = list(all_clips)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        tst = shuffled[n_train + n_val :]
        for subset, files in zip(splits, [train, val, tst]):
            dest = dataset_root / subset / folder
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = dest / f.name
                if not target.exists():
                    target.symlink_to(f.resolve())

    return dataset_root


def validate_and_build_all(
    collections: dict[str, list[str]],
    clips_per_species: int | None = None,
    threshold: float = BIRDNET_THRESHOLD,
) -> dict[str, Path]:
    SUBSAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for coll_name, names in collections.items():
        (SUBSAMPLES_DIR / coll_name).mkdir(parents=True, exist_ok=True)
        print(f"\nBirdNET subsamples: {coll_name} ({len(names)} species, threshold={threshold})...")
        for name in tqdm(names, desc=coll_name):
            process_species(name, coll_name, threshold=threshold)

    results: dict[str, Path] = {}
    for coll_name, names in collections.items():
        print(f"\nBuilding dataset: {coll_name}")
        path = build_dataset(
            coll_name,
            names,
            clips_per_species=clips_per_species,
            subsamples_dir=SUBSAMPLES_DIR / coll_name,
        )
        results[coll_name] = path
        print(f"  → {path}")

    return results
