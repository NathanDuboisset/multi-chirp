"""
audioset.py — fetch ambient `non_target` audio from Google's AudioSet.

Uses the Hugging Face `agkphysics/AudioSet` mirror in streaming mode so we
never download more than we need and never touch YouTube directly.

Pipeline:
    1. stream_download_audioset(cfg) → iterate the chosen HF split with
       streaming=True, keep only clips whose `human_labels` intersect
       FOCUS_CLASSES, pick the loudest CLIP_DURATION window inside the
       ~10 s example (by RMS), and write a 16 kHz mono WAV per clip.
       Stops once per-class cap or global cap is reached.
    2. materialize_audioset_non_target(cfg, collection) → symlinks under
       raw_dataset/subsamples/<collection>/non_target_audioset/.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pyrootutils
import soundfile as sf
from datasets import Audio, load_dataset
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from building.download import (
    CLIP_DURATION,
    RAW_DATASET_DIR,
    SUBSAMPLES_DIR,
    TARGET_SAMPLE_RATE,
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

    # HF dataset config + split. `unbalanced/train` is the only realistic
    # source if you want a few thousand clips spread across niche classes;
    # `balanced` only has ~50 per class.
    hf_config: Literal["balanced", "unbalanced", "full"] = "unbalanced"
    hf_split: Literal["train", "test"] = "train"

    clips_per_class: int = 250
    max_total_clips: int = 5000
    target_sample_rate: int = TARGET_SAMPLE_RATE
    target_clip_duration: float = CLIP_DURATION
    processed_dir: Path = RAW_DATASET_DIR / "audioset"

    # Sliding-window picker tuning.
    window_hop_s: float = 0.5
    # Reject clips whose loudest window is below this RMS — they're silent.
    min_rms: float = 5e-3


def _loudest_window(
    audio: np.ndarray, sr: int, dur: float, hop_s: float, min_rms: float
) -> np.ndarray | None:
    """Slide a `dur`-second window over `audio` and return the slice with
    the highest RMS. Pads (centred) if the input is shorter than `dur`.
    Returns None if even the best window is essentially silent.
    """
    n = int(dur * sr)
    if len(audio) < sr:
        return None

    if len(audio) <= n:
        pad = n - len(audio)
        left = pad // 2
        padded = np.pad(audio, (left, pad - left)).astype(np.float32)
        if float(np.sqrt(np.mean(padded**2))) < min_rms:
            return None
        return padded

    hop = max(1, int(hop_s * sr))
    best_start, best_rms = 0, -1.0
    for start in range(0, len(audio) - n + 1, hop):
        rms = float(np.sqrt(np.mean(audio[start : start + n] ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    if best_rms < min_rms:
        return None
    return audio[best_start : best_start + n].astype(np.float32)


def _existing_keys(class_dir: Path) -> set[str]:
    if not class_dir.exists():
        return set()
    return {p.stem for p in class_dir.glob("*.wav")}


def stream_download_audioset(cfg: AudioSetConfig) -> dict[str, int]:
    """Streaming download from the HF AudioSet mirror.

    Idempotent: filenames are `<video_id>.wav`, so re-running tops up to
    the configured caps without re-downloading existing clips.
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    focus_lookup = {c.lower(): c for c in FOCUS_CLASSES}

    counts: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        existing = _existing_keys(cfg.processed_dir / key)
        counts[key] = len(existing)
        seen[key] = existing
    total = sum(counts.values())

    if total >= cfg.max_total_clips:
        print(
            f"[audioset] global cap {cfg.max_total_clips} already reached "
            f"({total} clips on disk)."
        )
        return counts

    print(
        f"[audioset] streaming {cfg.hf_config}/{cfg.hf_split} via HF datasets "
        f"(per_class_cap={cfg.clips_per_class}, global_cap={cfg.max_total_clips}, "
        f"already on disk={total})."
    )

    ds = load_dataset(
        "agkphysics/AudioSet",
        cfg.hf_config,
        split=cfg.hf_split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.target_sample_rate))

    remaining = cfg.max_total_clips - total
    pbar = tqdm(total=remaining, desc="[audioset] stream")

    try:
        for example in ds:
            if total >= cfg.max_total_clips:
                break

            human_labels = example.get("human_labels") or []
            matched = next(
                (focus_lookup[lbl.lower()] for lbl in human_labels if lbl.lower() in focus_lookup),
                None,
            )
            if matched is None:
                continue

            key = _slugify(matched)
            if counts[key] >= cfg.clips_per_class:
                continue

            video_id = example.get("video_id") or f"sample{total:06d}"
            if video_id in seen[key]:
                continue

            audio = example["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            sr = int(audio["sampling_rate"])

            window = _loudest_window(
                arr,
                sr,
                cfg.target_clip_duration,
                cfg.window_hop_s,
                cfg.min_rms,
            )
            if window is None:
                continue

            class_dir = cfg.processed_dir / key
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / f"{video_id}.wav"
            sf.write(str(out_path), window, sr, subtype="PCM_16")
            seen[key].add(video_id)
            counts[key] += 1
            total += 1
            pbar.update(1)
    finally:
        pbar.close()

    print(f"[audioset] done — {total} clips on disk total.")
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        if counts.get(key, 0) == 0:
            print(f"[audioset] WARNING: {cls!r} has 0 clips.")
    return counts


async def stream_download_audioset_async(cfg: AudioSetConfig) -> dict[str, int]:
    """Awaitable wrapper for use inside the existing notebooks."""
    return await asyncio.to_thread(stream_download_audioset, cfg)


def materialize_audioset_non_target(
    cfg: AudioSetConfig,
    collection_name: str,
    total_clips: int,
) -> int:
    """Symlink a balanced subset of processed AudioSet clips into
    raw_dataset/subsamples/<collection_name>/non_target_audioset/.

    Round-robin across classes so all categories are represented.
    """
    out_dir = SUBSAMPLES_DIR / collection_name / "non_target_audioset"
    out_dir.mkdir(parents=True, exist_ok=True)

    pools: list[list[Path]] = []
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        pool = sorted((cfg.processed_dir / key).glob("*.wav"))
        if pool:
            pools.append(pool)

    if not pools:
        print("[audioset] WARNING: no processed clips found; skipping materialise.")
        return 0

    existing = sorted(out_dir.glob("clip_*.wav"))
    next_idx = max((int(p.stem.split("_")[1]) for p in existing), default=-1) + 1
    already_targets = {p.resolve() for p in existing}

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
