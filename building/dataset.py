"""
dataset.py — validate recordings with BirdNET and build split datasets.

Pipeline per species:
  raw_dataset/<Species_name>/*.mp3   (original sample rate)
      → BirdNET inference at 48 kHz (internal resample)
      → validated 3s WAV clips at 16 kHz
      → datasets/<collection>/{training,validation,testing}/<Species_name>/

Four collections are created from 4 SpeciesInfo lists supplied by taxonomy.py:
  diff_species, diff_genus, diff_family, diff_order
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Sequence

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
DATASETS_DIR = ROOT / "datasets"

BIRDNET_THRESHOLD = 0.92
CLIP_DURATION = 3.0          # seconds — BirdNET operates on 3s windows
TARGET_SAMPLE_RATE = 16_000  # Hz — output rate for the ML datasets
BIRDNET_SAMPLE_RATE = 48_000 # Hz — BirdNET internal rate (librosa resamples)
SPLIT = (0.7, 0.15, 0.15)    # train / val / test

_analyzer: ModelAnalyzer | None = None


def _get_analyzer() -> ModelAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ModelAnalyzer()
    return _analyzer


def _detections_for_file(audio_path: Path, species_name: str) -> list[dict]:
    """Return BirdNET detections above threshold for the target species."""
    rec = Recording(
        _get_analyzer(),
        str(audio_path),
        min_conf=BIRDNET_THRESHOLD,
    )
    rec.analyze()
    name_lower = species_name.lower()
    return [
        d for d in rec.detections
        if name_lower in d.get("scientific_name", "").lower()
    ]


def _extract_clip(audio_path: Path, start: float, duration: float, out_path: Path) -> None:
    """Load a segment at 16 kHz and save as WAV."""
    sig, _ = librosa.load(
        str(audio_path),
        sr=TARGET_SAMPLE_RATE,
        offset=start,
        duration=duration,
        mono=True,
        res_type="kaiser_fast",
    )
    target_len = int(duration * TARGET_SAMPLE_RATE)
    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)))
    else:
        sig = sig[:target_len]
    sf.write(str(out_path), sig, TARGET_SAMPLE_RATE, subtype="PCM_16")


def process_species(
    scientific_name: str,
    out_dir: Path,
    threshold: float = BIRDNET_THRESHOLD,
) -> list[Path]:
    """Validate recordings for one species and write 3s clips at 16 kHz.

    Returns list of saved clip paths.
    """
    folder_name = scientific_name.replace(" ", "_")
    raw_dir = RAW_DATASET_DIR / folder_name
    species_out = out_dir / folder_name
    species_out.mkdir(parents=True, exist_ok=True)

    audio_files = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))
    clips: list[Path] = []
    clip_idx = 0

    for audio_path in tqdm(audio_files, desc=f"  BirdNET {scientific_name}", leave=False):
        detections = _detections_for_file(audio_path, scientific_name)
        for det in detections:
            start = float(det.get("start_time", 0.0))
            clip_path = species_out / f"clip_{clip_idx:05d}.wav"
            if not clip_path.exists():
                _extract_clip(audio_path, start, CLIP_DURATION, clip_path)
            clips.append(clip_path)
            clip_idx += 1

    return clips


def _split_files(files: list[Path], split: tuple[float, float, float], seed: int = 42) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def build_dataset(
    collection_name: str,
    species_names: list[str],
    clips_per_species: int | None = None,
    split: tuple[float, float, float] = SPLIT,
    validated_dir: Path | None = None,
) -> Path:
    """Assemble a TF-readable dataset from pre-validated clips.

    Structure:
        datasets/<collection_name>/
            training/<Species_name>/*.wav
            validation/<Species_name>/*.wav
            testing/<Species_name>/*.wav

    If `validated_dir` is None, defaults to raw_dataset/validated/.
    `clips_per_species` caps how many clips to use per class (balanced).
    Returns the dataset root path.
    """
    vdir = validated_dir or (RAW_DATASET_DIR / "validated")
    dataset_root = DATASETS_DIR / collection_name
    splits = ("training", "validation", "testing")
    for s in splits:
        (dataset_root / s).mkdir(parents=True, exist_ok=True)

    for name in species_names:
        folder = name.replace(" ", "_")
        src_dir = vdir / folder
        all_clips = sorted(src_dir.glob("*.wav"))
        if clips_per_species:
            all_clips = all_clips[:clips_per_species]
        train, val, test = _split_files(list(all_clips), split)
        for subset, files in zip(splits, [train, val, test]):
            dest = dataset_root / subset / folder
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = dest / f.name
                if not target.exists():
                    shutil.copy2(f, target)

    return dataset_root


def validate_and_build_all(
    collections: dict[str, list[str]],
    clips_per_species: int | None = None,
    threshold: float = BIRDNET_THRESHOLD,
) -> dict[str, Path]:
    """Full pipeline: validate all species then build all 4 datasets.

    `collections` maps collection_name → list of scientific names.
    Returns dict of collection_name → dataset root path.
    """
    validated_dir = RAW_DATASET_DIR / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)

    all_species = list({name for names in collections.values() for name in names})
    print(f"Validating {len(all_species)} species with BirdNET (threshold={threshold})...")
    for name in tqdm(all_species, desc="Species"):
        process_species(name, validated_dir, threshold=threshold)

    results: dict[str, Path] = {}
    for coll_name, names in collections.items():
        print(f"\nBuilding dataset: {coll_name}")
        path = build_dataset(coll_name, names, clips_per_species=clips_per_species, validated_dir=validated_dir)
        results[coll_name] = path
        print(f"  → {path}")

    return results
