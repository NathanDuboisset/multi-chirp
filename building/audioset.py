"""
audioset.py — fetch ambient `non_target` audio from Google's AudioSet.

Pipeline:
    1. download_audioset_focus_classes(cfg) → raw 10 s clips per class
    2. postprocess_audioset_clips(cfg)      → 16 kHz mono, 3.0 s WAV
    3. materialize_audioset_non_target(cfg, collection) → symlinks under
       raw_dataset/subsamples/<collection>/non_target_audioset/

Designed to plug into the existing XC pipeline in building/download.py and
the symlink-based assembly in building/dataset.py.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pyrootutils
import soundfile as sf
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from building.download import (
    CLIP_DURATION,
    RAW_DATASET_DIR,
    SUBSAMPLES_DIR,
    TARGET_SAMPLE_RATE,
    _require_ffmpeg,
    quiet,
)

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)


FOCUS_CLASSES: list[str] = [
    "Aircraft",
    "Bee, wasp, etc.",
    "Chainsaw",
    "Church bell",
    "Cricket",
    "Croak",
    "Fly, housefly",
    "Howl (wind)",
    "Insect",
    "Lawn mower",
    "Light engine (high frequency)",
    "Mosquito",
    "Outside, rural or natural",
    "Rain",
    "Rain on surface",
    "Raindrop",
    "Rustling leaves",
    "Stream",
    "Thunder",
    "Thunderstorm",
    "Traffic noise, roadway noise",
    "Truck",
    "Wind",
    "Wind noise (microphone)",
]


def _slugify(name: str) -> str:
    """display name → safe folder key, e.g. 'Bee, wasp, etc.' → 'bee_wasp_etc'."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


class AudioSetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    download_type: Literal["balanced_train", "unbalanced_train", "eval"] = (
        "unbalanced_train"
    )
    # YouTube starts rate-limiting / bot-flagging above ~4 parallel jobs.
    # Higher concurrency does not actually go faster end-to-end.
    n_jobs: int = 4
    clips_per_class: int = 400
    target_sample_rate: int = TARGET_SAMPLE_RATE
    target_clip_duration: float = CLIP_DURATION
    raw_dir: Path = RAW_DATASET_DIR / "audioset_raw"
    processed_dir: Path = RAW_DATASET_DIR / "audioset"
    download_format: str = "wav"


def _class_subdir(parent: Path, class_name: str) -> Path:
    """audioset-download writes one folder per label; the folder name varies
    slightly across versions (display name vs slug). Find the one that matches."""
    candidates = [class_name, _slugify(class_name), class_name.replace(" ", "_")]
    for cand in candidates:
        p = parent / cand
        if p.exists():
            return p
    return parent / class_name


def _count_raw(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(
        1
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".ogg", ".flac", ".mp3", ".m4a"}
    )


_AUDIOSET_ONTOLOGY_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)


def _valid_audioset_labels() -> set[str]:
    """Fetch AudioSet's official class_labels_indices.csv (cached locally)
    and return the set of valid display_name values."""
    import csv

    import requests

    cache = RAW_DATASET_DIR / "class_labels_indices.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_AUDIOSET_ONTOLOGY_URL, timeout=30)
        resp.raise_for_status()
        cache.write_bytes(resp.content)

    with cache.open("r", encoding="utf-8") as f:
        return {row["display_name"] for row in csv.DictReader(f)}


async def download_audioset_focus_classes(
    cfg: AudioSetConfig,
) -> dict[str, int]:
    """Download AudioSet clips for every class in FOCUS_CLASSES.

    Idempotent: classes that already have >= cfg.clips_per_class raw files are
    skipped. Returns {class_name: n_raw_files} after the run.
    """
    _require_ffmpeg()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    valid = _valid_audioset_labels()
    unknown = [c for c in FOCUS_CLASSES if c not in valid]
    if unknown:
        print(
            f"[audioset] WARNING: dropping {len(unknown)} class(es) not in "
            f"AudioSet ontology: {unknown}"
        )
    usable = [c for c in FOCUS_CLASSES if c in valid]

    pending: list[str] = []
    for cls in usable:
        n = _count_raw(_class_subdir(cfg.raw_dir, cls))
        if n >= cfg.clips_per_class:
            print(f"[audioset] {cls}: {n} clips, skipping.")
        else:
            pending.append(cls)

    if not pending:
        print("[audioset] all focus classes already populated.")
    else:
        print(f"[audioset] downloading {len(pending)} classes: {pending}")

        def _do_download() -> None:
            from audioset_download import Downloader

            d = Downloader(
                root_path=str(cfg.raw_dir),
                labels=pending,
                n_jobs=cfg.n_jobs,
                download_type=cfg.download_type,
            )
            with quiet():
                d.download(format=cfg.download_format)

        await asyncio.to_thread(_do_download)

    counts: dict[str, int] = {}
    for cls in usable:
        counts[cls] = _count_raw(_class_subdir(cfg.raw_dir, cls))
        if counts[cls] == 0:
            print(f"[audioset] WARNING: {cls!r} returned 0 clips.")
    return counts


def _resample_centre_clip(
    src: Path, target_sr: int, target_dur: float
) -> np.ndarray | None:
    """Load src, resample to target_sr mono, take centre target_dur seconds.
    Returns None if the audio is unreadable or shorter than ~1 s."""
    try:
        with quiet():
            audio, _ = librosa.load(
                str(src), sr=target_sr, mono=True, res_type="kaiser_fast"
            )
    except Exception as e:
        print(f"[audioset] skip unreadable {src.name}: {e}")
        return None

    target_len = int(target_dur * target_sr)
    if len(audio) < target_sr:  # < 1 s of audio is junk
        return None

    if len(audio) >= target_len:
        start = (len(audio) - target_len) // 2
        return audio[start : start + target_len].astype(np.float32)

    # shorter than target: pad with zeros centred
    pad = target_len - len(audio)
    left = pad // 2
    right = pad - left
    return np.pad(audio, (left, right)).astype(np.float32)


def postprocess_audioset_clips(cfg: AudioSetConfig) -> dict[str, int]:
    """Resample + centre-crop raw AudioSet clips into uniform 16 kHz / 3 s WAVs.

    Output layout:
        cfg.processed_dir / <class_key> / clip_<idx>.wav
    Idempotent: existing clip_<idx>.wav files are not regenerated.
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for cls in FOCUS_CLASSES:
        src_dir = _class_subdir(cfg.raw_dir, cls)
        key = _slugify(cls)
        dest_dir = cfg.processed_dir / key
        dest_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(dest_dir.glob("clip_*.wav"))
        next_idx = (
            max((int(p.stem.split("_")[1]) for p in existing), default=-1) + 1
        )
        already_processed_sources = {p.stem for p in existing}

        if not src_dir.exists():
            counts[key] = len(existing)
            continue

        raw_files = sorted(
            p
            for p in src_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".wav", ".ogg", ".flac", ".mp3", ".m4a"}
        )
        # cap at clips_per_class so postprocess can be run before download finishes
        raw_files = raw_files[: cfg.clips_per_class]

        written = 0
        for src in tqdm(raw_files, desc=f"[audioset] postproc {key}", leave=False):
            if src.stem in already_processed_sources:
                continue
            audio = _resample_centre_clip(
                src, cfg.target_sample_rate, cfg.target_clip_duration
            )
            if audio is None:
                continue
            dest = dest_dir / f"clip_{next_idx:05d}.wav"
            sf.write(str(dest), audio, cfg.target_sample_rate, subtype="PCM_16")
            next_idx += 1
            written += 1

        counts[key] = len(list(dest_dir.glob("clip_*.wav")))
        if written:
            print(f"[audioset] {key}: wrote {written} new clips ({counts[key]} total)")

    return counts


def materialize_audioset_non_target(
    cfg: AudioSetConfig,
    collection_name: str,
    total_clips: int,
) -> int:
    """Symlink a balanced subset of processed AudioSet clips into
    raw_dataset/subsamples/<collection_name>/non_target_audioset/.

    Round-robin across classes so all 24 categories are represented.
    """
    out_dir = SUBSAMPLES_DIR / collection_name / "non_target_audioset"
    out_dir.mkdir(parents=True, exist_ok=True)

    pools: list[list[Path]] = []
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        pool = sorted((cfg.processed_dir / key).glob("clip_*.wav"))
        if pool:
            pools.append(pool)

    if not pools:
        print("[audioset] WARNING: no processed clips found; skipping materialise.")
        return 0

    existing = sorted(out_dir.glob("clip_*.wav"))
    next_idx = max((int(p.stem.split("_")[1]) for p in existing), default=-1) + 1
    already_targets = {p.resolve() for p in existing}

    # round-robin pick until we hit total_clips or every pool is drained
    cursors = [0] * len(pools)
    written = 0
    target = max(0, total_clips - len(existing))
    while written < target:
        progressed = False
        for i, pool in enumerate(pools):
            if written >= target:
                break
            if cursors[i] >= len(pool):
                continue
            src = pool[cursors[i]].resolve()
            cursors[i] += 1
            progressed = True
            if src in already_targets:
                continue
            link = out_dir / f"clip_{next_idx:05d}.wav"
            link.symlink_to(src)
            next_idx += 1
            written += 1
        if not progressed:
            break

    total = len(list(out_dir.glob("clip_*.wav")))
    print(
        f"[audioset] {collection_name}/non_target_audioset: {written} new symlinks "
        f"({total} total, target={total_clips})"
    )
    return total
