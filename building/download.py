"""Per-species recording download + BirdNET extraction. Idempotent.

The download pipeline is source-agnostic: an audio `Source` (XenoCanto, eBird/
Macaulay Library, ...) supplies the listing and the per-recording fetch, and
this module drives the BirdNET extraction + non-target bucketing identically
for both.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import os
import random
import sys
import threading
from pathlib import Path

import librosa
import numpy as np
import pyrootutils
import soundfile as sf
from birdnetlib import Recording as BirdNETRecording
from birdnetlib.analyzer import Analyzer as ModelAnalyzer
from dotenv import load_dotenv
from tqdm.auto import tqdm

from building.sources import RecordingEntry, Source

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
NON_TARGET_OTHER_DIR = SUBSAMPLES_DIR / "non_target_other"
BIRDNET_NO_BIRD_DIR = SUBSAMPLES_DIR / "non_target_empty"

TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 3.0
BIRDNET_THRESHOLD = 0.92
NON_TARGET_CAP = 2000
POOL_SIZE = 10

_analyzers: dict[int, ModelAnalyzer] = {}
_analyzers_lock = threading.Lock()
_stderr_lock = threading.Lock()
_clip_indices: dict[str, int] = {}
_nt_lock = threading.Lock()


@contextlib.contextmanager
def quiet():
    with _stderr_lock:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_out = os.dup(1)
        old_err = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)


def get_analyzer() -> ModelAnalyzer:
    tid = threading.get_ident()
    if tid not in _analyzers:
        with _analyzers_lock:
            if tid not in _analyzers:
                _analyzers[tid] = ModelAnalyzer()
    return _analyzers[tid]


def next_idx(key: str, directory: Path) -> int:
    if key not in _clip_indices:
        _clip_indices[key] = (
            max(
                (
                    int(p.stem.rsplit("_", 1)[1])
                    for p in directory.glob("*.wav")
                    if p.stem.rsplit("_", 1)[-1].isdigit()
                ),
                default=-1,
            )
            + 1
        )
    return _clip_indices[key]


def write_clip(audio_path: Path, start: float, clip_path: Path) -> bool:
    if clip_path.exists():
        return True
    try:
        with quiet():
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
        return True
    except Exception as e:
        print(f"    skip clip {audio_path.name} @ {start:g}s: {e}")
        return False


async def fetch_listing(source: Source, scientific_name: str) -> list[RecordingEntry]:
    """Listings are cached per (source, species) so we hit the network once."""
    species_key = scientific_name.replace(" ", "_")
    listing_path = LISTINGS_DIR / "available" / source.name / f"{species_key}.csv"
    if listing_path.exists():
        with listing_path.open("r", newline="", encoding="utf-8") as f:
            return [
                RecordingEntry(
                    rec_id=r["rec_id"],
                    filename=r["filename"],
                    file_url=r["file_url"],
                    quality=r.get("quality", ""),
                )
                for r in csv.DictReader(f)
            ]

    rows = await source.fetch_listing(scientific_name)
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    with listing_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rec_id", "filename", "file_url", "quality"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "rec_id": r.rec_id,
                    "filename": r.filename,
                    "file_url": r.file_url,
                    "quality": r.quality,
                }
            )
    return rows


async def download_species_one_source(
    scientific_name: str,
    clips_per_species: int,
    source: Source,
) -> None:
    """Drive `source` until `species_dir` has `clips_per_species` total wavs
    (across all source prefixes). Idempotent: the listing CSV and the
    per-species processed.csv let reruns pick up where a previous run stopped.
    """
    species_key = scientific_name.replace(" ", "_")
    target_lower = scientific_name.lower()

    species_dir = SUBSAMPLES_DIR / species_key
    raw_dir = SPECIES_DIR / species_key
    for d in (species_dir, raw_dir, NON_TARGET_OTHER_DIR, BIRDNET_NO_BIRD_DIR):
        d.mkdir(parents=True, exist_ok=True)
    processed_csv = LISTINGS_DIR / "processed" / f"{species_key}.csv"
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    tag = f"[{scientific_name} / {source.name}]"

    clip_count = sum(1 for _ in species_dir.glob("*.wav"))
    if clip_count >= clips_per_species:
        print(f"{tag} {clip_count}/{clips_per_species} clips already on disk, skipping.")
        return

    available = await fetch_listing(source, scientific_name)

    if processed_csv.exists():
        with processed_csv.open("r", newline="", encoding="utf-8") as f:
            processed = {
                r["recording"] for r in csv.DictReader(f) if r.get("recording")
            }
    else:
        processed = set()
    pending = [r for r in available if r.filename not in processed]
    random.shuffle(pending)
    pending = pending[: clips_per_species - clip_count]
    if not pending:
        print(
            f"{tag} no pending recordings "
            f"(available={len(available)}, already processed={len(processed)}, "
            f"on disk={clip_count}/{clips_per_species})."
        )
        return

    print(
        f"{tag} {clip_count}/{clips_per_species} on disk; "
        f"queued {len(pending)} of {len(available)} available."
    )

    _clip_indices[species_key] = clip_count
    species_lock = threading.Lock()

    def handle(entry: RecordingEntry) -> int:
        nonlocal clip_count
        with species_lock:
            if clip_count >= clips_per_species:
                return 0

        rec_id = entry.rec_id
        audio_path = raw_dir / entry.filename

        try:
            source.download(entry, audio_path)
        except Exception as e:
            print(f"{tag} download failed {entry.filename}: {e}")
            return 0

        clips_written = 0
        try:
            with quiet():
                rec = BirdNETRecording(
                    get_analyzer(), str(audio_path), min_conf=BIRDNET_THRESHOLD
                )
                rec.analyze()

            target_dets, other_dets = [], []
            for d in rec.detections:
                sn = d.get("scientific_name", "").strip().lower()
                (target_dets if target_lower in sn else other_dets).append(d)
            no_detections = not rec.detections
            del rec

            with species_lock:
                for det in target_dets:
                    if clip_count >= clips_per_species:
                        break
                    idx = _clip_indices[species_key]
                    if write_clip(
                        audio_path,
                        float(det.get("start_time", 0.0)),
                        species_dir / f"{source.prefix}{rec_id}_{idx:05d}.wav",
                    ):
                        _clip_indices[species_key] += 1
                        clip_count += 1
                        clips_written += 1
                        if clip_count == clips_per_species:
                            print(f"{tag} reached {clips_per_species} clips.")

            with _nt_lock:
                if other_dets:
                    nt_key = "non_target_other"
                    next_idx(nt_key, NON_TARGET_OTHER_DIR)
                    for det in other_dets:
                        if _clip_indices[nt_key] >= NON_TARGET_CAP:
                            break
                        out = (
                            NON_TARGET_OTHER_DIR
                            / f"{species_key}__{source.prefix}{rec_id}_{_clip_indices[nt_key]:05d}.wav"
                        )
                        if write_clip(
                            audio_path, float(det.get("start_time", 0.0)), out
                        ):
                            _clip_indices[nt_key] += 1

                if no_detections:
                    nt_key = "non_target_empty"
                    next_idx(nt_key, BIRDNET_NO_BIRD_DIR)
                    if _clip_indices[nt_key] < NON_TARGET_CAP:
                        try:
                            info = sf.info(str(audio_path))
                            max_start = max(
                                0.0, float(info.duration) - CLIP_DURATION
                            )
                            start = (
                                random.uniform(0.0, max_start)
                                if max_start > 0
                                else 0.0
                            )
                        except Exception:
                            start = 0.0
                        out = (
                            BIRDNET_NO_BIRD_DIR
                            / f"{source.prefix}{rec_id}_{_clip_indices[nt_key]:05d}.wav"
                        )
                        if write_clip(audio_path, start, out):
                            _clip_indices[nt_key] += 1

        except Exception as e:
            print(f"{tag} BirdNET failed {entry.filename}: {e}")
        else:
            with species_lock:
                write_header = (
                    not processed_csv.exists() or processed_csv.stat().st_size == 0
                )
                with processed_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f, fieldnames=["recording", "clips_extracted"]
                    )
                    if write_header:
                        w.writeheader()
                    w.writerow(
                        {
                            "recording": entry.filename,
                            "clips_extracted": str(clips_written),
                        }
                    )
        finally:
            audio_path.unlink(missing_ok=True)
        return clips_written

    sem = asyncio.Semaphore(POOL_SIZE)
    dl_lock = asyncio.Lock()
    dl_last = [0.0]
    dl_interval = source.dl_interval
    pbar_recs = tqdm(
        total=len(pending), desc=f"[{scientific_name}] recs", position=0, leave=True
    )
    pbar_clips = tqdm(
        total=clips_per_species,
        initial=clip_count,
        desc=f"[{scientific_name}] clips",
        position=1,
        leave=True,
    )

    async def run(entry: RecordingEntry) -> None:
        if dl_interval > 0:
            async with dl_lock:
                now = asyncio.get_event_loop().time()
                wait = dl_interval - (now - dl_last[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                dl_last[0] = asyncio.get_event_loop().time()
        async with sem:
            written = await asyncio.to_thread(handle, entry)
        pbar_recs.update(1)
        if written:
            pbar_clips.update(written)

    try:
        await asyncio.gather(*[run(r) for r in pending])
    finally:
        pbar_clips.close()
        pbar_recs.close()


async def download_and_process(
    scientific_name: str,
    per_source_clips: int,
    sources: list[Source],
) -> None:
    """Build the species' clip pool from multiple sources.

    Each source has a per-source quota of `per_source_clips`. The global
    target is `per_source_clips * len(sources)`. The orchestration runs in
    two phases:

      A. For each source in order, raise the species' clip count by up to
         `per_source_clips` new clips from that source.
      B. If the global target still isn't met (some source ran out of
         recordings), iterate the sources again and let any of them top up
         to the global target. Per-source `processed.csv` filtering means
         no recording is attempted twice.
    """
    if not sources:
        raise ValueError("download_and_process: need at least one source")

    total_target = per_source_clips * len(sources)
    species_key = scientific_name.replace(" ", "_")
    species_dir = SUBSAMPLES_DIR / species_key
    species_dir.mkdir(parents=True, exist_ok=True)

    def current_total() -> int:
        return sum(1 for _ in species_dir.glob("*.wav"))

    def count_from(src: Source) -> int:
        # Clips are named "{prefix}{rec_id}_{idx:05d}.wav" — see write loop
        # in download_species_one_source. Empty prefix would match every
        # clip, so guard against that.
        if not src.prefix:
            return 0
        return sum(1 for _ in species_dir.glob(f"{src.prefix}*.wav"))

    source_list = ", ".join(
        f"{s.name}={count_from(s)}/{per_source_clips}" for s in sources
    )
    start_count = current_total()
    print(
        f"\n=== [{scientific_name}] target={total_target} clips "
        f"({per_source_clips}/source × {len(sources)} sources)  "
        f"on disk={start_count} [{source_list}] ==="
    )

    if start_count >= total_target:
        print(f"[{scientific_name}] already at target, nothing to do.")
        return

    async def _try_one_source(src: Source, cap: int, phase: str) -> None:
        # Per-source failures (missing taxonomy code, listing API hiccup,
        # auth error, ...) should never abort the outer species loop —
        # log and move on so the rest of the batch still runs.
        try:
            await download_species_one_source(scientific_name, cap, src)
        except Exception as e:
            print(
                f"[{scientific_name} / {src.name}] {phase} failed: "
                f"{type(e).__name__}: {e}"
            )

    print(f"[{scientific_name}] --- Phase A: per-source quota ---")
    for src in sources:
        already_from_src = count_from(src)
        if already_from_src >= per_source_clips:
            print(
                f"[{scientific_name} / {src.name}] "
                f"{already_from_src}/{per_source_clips} already on disk from this "
                f"source, skipping Phase A."
            )
            continue
        # Per-source cap: bring this source's contribution up to
        # per_source_clips. The function counts total clips for its stop
        # condition, so we pass current_total + remaining-needed-from-this-src.
        needed = per_source_clips - already_from_src
        cap = min(current_total() + needed, total_target)
        await _try_one_source(src, cap, "Phase A")

    if current_total() < total_target:
        print(
            f"[{scientific_name}] --- Phase B: top up "
            f"(have {current_total()}/{total_target}) ---"
        )
        for src in sources:
            if current_total() >= total_target:
                break
            await _try_one_source(src, total_target, "Phase B")

    per_source_summary = ", ".join(f"{s.name}={count_from(s)}" for s in sources)
    print(
        f"[{scientific_name}] done. "
        f"final {current_total()}/{total_target} clips on disk "
        f"[{per_source_summary}]."
    )
