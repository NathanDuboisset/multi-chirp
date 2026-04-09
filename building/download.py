"""
download.py — async XC download + BirdNET extraction pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer as ModelAnalyzer
from dotenv import load_dotenv
import librosa
import numpy as np
import pyrootutils
import requests
import soundfile as sf
from tqdm import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
load_dotenv()

RAW_DATASET_DIR = ROOT / "raw_dataset"
SPECIES_DIR = RAW_DATASET_DIR / "species"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
LISTINGS_DIR = RAW_DATASET_DIR / "listings"
XC_API_BASE = "https://xeno-canto.org/api/3/recordings"
TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 3.0
POOL_SIZE = 10
BIRDNET_THRESHOLD = 0.92
NON_TARGET_CAP = 2000

_analyzers: dict[int, ModelAnalyzer] = {}
_analyzers_lock = threading.Lock()
_stderr_lock = threading.Lock()
_clip_indices: dict[str, int] = {}

XC_DL_INTERVAL = 0.2


@contextlib.contextmanager
def quiet():
    with _stderr_lock:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old, 2)
            os.close(old)


@dataclass(slots=True)
class RecordingTask:
    scientific_name: str
    species_key: str
    filename: str
    file_url: str
    excluded: frozenset[str]  # lowercase names of all target species


def _get_analyzer() -> ModelAnalyzer:
    tid = threading.get_ident()
    if tid not in _analyzers:
        with _analyzers_lock:
            if tid not in _analyzers:
                _analyzers[tid] = ModelAnalyzer()
    return _analyzers[tid]


def _write_clip(audio_file: Path, start: float, clip_path: Path) -> bool:
    try:
        if clip_path.exists():
            return True
        with quiet():
            sig, _ = librosa.load(
                str(audio_file),
                sr=TARGET_SAMPLE_RATE,
                offset=start,
                duration=CLIP_DURATION,
                mono=True,
                res_type="kaiser_fast",
            )
        target_len = int(CLIP_DURATION * TARGET_SAMPLE_RATE)
        sig = (
            np.pad(sig, (0, target_len - len(sig)))
            if len(sig) < target_len
            else sig[:target_len]
        )
        sf.write(str(clip_path), sig, TARGET_SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"    skip clip {audio_file.name} @ {start:g}s: {e}")
        return False


def _count_clips(collection_name: str, species_key: str) -> int:
    d = SUBSAMPLES_DIR / collection_name / species_key
    return sum(1 for _ in d.glob("clip_*.wav")) if d.exists() else 0


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_processed(
    csv_path: Path, filename: str, clips: int, lock: threading.Lock
) -> None:
    with lock:
        write_header = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            csv_path.touch()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["recording", "clips_extracted"])
            if write_header:
                w.writeheader()
            w.writerow({"recording": filename, "clips_extracted": str(clips)})


def _process_recording(
    task: RecordingTask,
    collection_name: str,
    clips_per_species: int,
    auto_delete: bool,
    lock: threading.Lock,
    nt_other_lock: threading.Lock,
    nt_empty_lock: threading.Lock,
) -> int:
    (SPECIES_DIR / task.species_key).mkdir(parents=True, exist_ok=True)
    audio_path = SPECIES_DIR / task.species_key / task.filename
    species_out = SUBSAMPLES_DIR / collection_name / task.species_key
    species_out.mkdir(parents=True, exist_ok=True)
    nt_other_out = SUBSAMPLES_DIR / collection_name / "non_target_other"
    nt_empty_out = SUBSAMPLES_DIR / collection_name / "non_target_empty"
    nt_other_out.mkdir(parents=True, exist_ok=True)
    nt_empty_out.mkdir(parents=True, exist_ok=True)
    processed_csv = LISTINGS_DIR / f"processed/{task.species_key}.csv"

    try:
        with requests.get(task.file_url, timeout=60, stream=True) as r:
            r.raise_for_status()
            with audio_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"[{task.scientific_name}] download failed {task.filename}: {e}")
        return 0

    clips_written = 0
    try:
        with quiet():
            rec = Recording(
                _get_analyzer(), str(audio_path), min_conf=BIRDNET_THRESHOLD
            )
            rec.analyze()

        target_lower = task.scientific_name.lower()
        target_dets = [
            d
            for d in rec.detections
            if target_lower in d.get("scientific_name", "").lower()
        ]
        other_dets = [
            d
            for d in rec.detections
            if d.get("scientific_name", "").strip().lower() not in task.excluded
            and target_lower not in d.get("scientific_name", "").lower()
        ]
        no_detections = not rec.detections
        del rec

        def _get_idx(key: str, directory: Path) -> int:
            if key not in _clip_indices:
                _clip_indices[key] = (
                    max(
                        (
                            int(p.stem.split("_", 1)[1])
                            for p in directory.glob("clip_*.wav")
                        ),
                        default=-1,
                    )
                    + 1
                )
            return _clip_indices[key]

        with lock:
            current = _clip_indices.get(f"{collection_name}/{task.species_key}", None)
            if current is None:
                current = _get_idx(f"{collection_name}/{task.species_key}", species_out)
            for det in target_dets:
                if current + clips_written >= clips_per_species:
                    break
                idx = _clip_indices[f"{collection_name}/{task.species_key}"]
                if _write_clip(
                    audio_path,
                    float(det.get("start_time", 0.0)),
                    species_out / f"clip_{idx:05d}.wav",
                ):
                    _clip_indices[f"{collection_name}/{task.species_key}"] += 1
                    clips_written += 1

        if other_dets:
            with nt_other_lock:
                nt_key = f"{collection_name}/non_target_other"
                _get_idx(nt_key, nt_other_out)
                for det in other_dets:
                    if _clip_indices[nt_key] >= NON_TARGET_CAP:
                        break
                    if _write_clip(
                        audio_path,
                        float(det.get("start_time", 0.0)),
                        nt_other_out / f"clip_{_clip_indices[nt_key]:05d}.wav",
                    ):
                        _clip_indices[nt_key] += 1

        if no_detections:
            with nt_empty_lock:
                nt_key = f"{collection_name}/non_target_empty"
                _get_idx(nt_key, nt_empty_out)
                if _clip_indices[nt_key] < NON_TARGET_CAP:
                    try:
                        info = sf.info(str(audio_path))
                        max_start = max(0.0, float(info.duration) - CLIP_DURATION)
                        start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
                    except Exception:
                        start = 0.0
                    if _write_clip(
                        audio_path,
                        start,
                        nt_empty_out / f"clip_{_clip_indices[nt_key]:05d}.wav",
                    ):
                        _clip_indices[nt_key] += 1

    except Exception as e:
        print(f"[{task.scientific_name}] BirdNET failed {task.filename}: {e}")
    finally:
        _append_processed(processed_csv, task.filename, clips_written, lock)
        if auto_delete:
            audio_path.unlink(missing_ok=True)
    return clips_written


async def _fetch_listing(
    session: aiohttp.ClientSession, scientific_name: str, max_recordings: int
) -> list[dict[str, str]]:
    species_key = scientific_name.replace(" ", "_")
    listing_path = LISTINGS_DIR / f"available/{species_key}.csv"
    if listing_path.exists():
        return _read_csv(listing_path)

    query = f'sp:"{scientific_name}" grp:birds q:A'
    rows: list[dict[str, str]] = []
    page = 1
    key = os.getenv("XC_API_KEY", "demo")
    while len(rows) < max_recordings:
        async with session.get(
            XC_API_BASE,
            params={"query": query, "key": key, "per_page": 100, "page": page},
            timeout=30,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        for rec in data.get("recordings", []):
            file_url = str(rec.get("file", "")).strip()
            xc_id = str(rec.get("id", "")).strip()
            if not file_url.startswith("http"):
                file_url = "https:" + file_url
            rows.append(
                {
                    "xc_id": xc_id,
                    "filename": f"XC{xc_id}.mp3",
                    "file_url": file_url,
                    "quality": str(rec.get("q", "")).strip(),
                }
            )
            if len(rows) >= max_recordings:
                break
        if page >= int(data.get("numPages", 1)) or not data.get("recordings"):
            break
        page += 1

    listing_path.parent.mkdir(parents=True, exist_ok=True)
    with listing_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["xc_id", "filename", "file_url", "quality"])
        w.writeheader()
        w.writerows(rows)
    return rows


async def download_and_process(
    species_names: list[str],
    collection_name: str,
    clips_per_species: int,
    max_recordings: int = 500,
    auto_delete: bool = False,
) -> None:
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    (SUBSAMPLES_DIR / collection_name).mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=POOL_SIZE * 2)
    ) as session:
        print("Fetching available listings...")
        listings = await asyncio.gather(
            *[_fetch_listing(session, name, max_recordings) for name in species_names]
        )

    excluded = frozenset(n.lower() for n in species_names)
    locks = {name.replace(" ", "_"): threading.Lock() for name in species_names}
    nt_other_lock = threading.Lock()
    nt_empty_lock = threading.Lock()

    jobs: list[RecordingTask] = []
    for scientific_name, available in zip(species_names, listings, strict=True):
        species_key = scientific_name.replace(" ", "_")
        processed = {
            r["recording"]
            for r in _read_csv(LISTINGS_DIR / f"processed/{species_key}.csv")
            if r.get("recording")
        }
        already = _count_clips(collection_name, species_key)
        if already >= clips_per_species:
            print(f"[{scientific_name}] {already} clips, skipping.")
            continue
        pending = [r for r in available if r["filename"] not in processed]
        random.shuffle(pending)
        pending = pending[: clips_per_species - already]
        print(
            f"[{scientific_name}] available={len(available)} processed={len(processed)} queued={len(pending)}"
        )
        for row in pending:
            jobs.append(
                RecordingTask(
                    scientific_name,
                    species_key,
                    row["filename"],
                    row["file_url"],
                    excluded,
                )
            )

    if not jobs:
        print("Nothing to process.")
        return

    sem = asyncio.Semaphore(POOL_SIZE)
    dl_lock = asyncio.Lock()
    dl_last: list[float] = [0.0]
    pbar = tqdm(total=len(jobs), desc="Download+BirdNET")

    async def _run(task: RecordingTask) -> None:
        async with dl_lock:
            now = asyncio.get_event_loop().time()
            wait = XC_DL_INTERVAL - (now - dl_last[0])
            if wait > 0:
                await asyncio.sleep(wait)
            dl_last[0] = asyncio.get_event_loop().time()
        async with sem:
            await asyncio.to_thread(
                _process_recording,
                task,
                collection_name,
                clips_per_species,
                auto_delete,
                locks[task.species_key],
                nt_other_lock,
                nt_empty_lock,
            )
        pbar.update(1)

    await asyncio.gather(*[_run(task) for task in jobs])
    pbar.close()
