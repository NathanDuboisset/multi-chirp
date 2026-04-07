"""
download.py — download recordings from Xeno-canto API v3.

Files are saved at their original audio rate; no resampling is done here.
Requires XC_API_KEY in .env.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pyrootutils
import requests
from tqdm import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
load_dotenv()

RAW_DATASET_DIR = ROOT / "raw_dataset"
# XC downloads: raw_dataset/species/<Genus_species>/*.mp3
SPECIES_DIR = RAW_DATASET_DIR / "species"
SUBSAMPLES_DIR = RAW_DATASET_DIR / "subsamples"
XC_API_BASE = "https://xeno-canto.org/api/3/recordings"
ACCEPTED_QUALITY = {"A", "B"}


def _safe_folder_name(scientific_name: str) -> str:
    return scientific_name.replace(" ", "_")


def _quality_query_from_set(quality: set[str]) -> str:
    # Xeno-canto supports q:">C" for A/B which is more efficient server-side.
    if quality == {"A", "B"}:
        return 'q:">C"'
    if quality == {"A"}:
        return "q:A"
    return ""


def _load_processed_recordings_for_species(
    scientific_name: str,
    subsamples_dir: Path = SUBSAMPLES_DIR,
) -> set[str]:
    species_key = _safe_folder_name(scientific_name)
    processed_names: set[str] = set()
    for csv_path in subsamples_dir.glob("*/processed_recordings.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("species") or "").strip() != species_key:
                        continue
                    rec = (row.get("recording") or "").strip()
                    if rec:
                        processed_names.add(rec)
        except Exception:
            continue
    return processed_names


def download_species(
    scientific_name: str,
    max_recordings: int = 100,
    quality: set[str] = ACCEPTED_QUALITY,
    dest_dir: Path = SPECIES_DIR,
    api_key: str | None = None,
    verbose: bool = True,
    skip_processed: bool = True,
) -> list[Path]:
    """Ensure up to `max_recordings` MP3s for `scientific_name` from XC.

    Existing files in `dest_dir` are counted first. Only missing files are
    downloaded, so reruns resume from current progress.
    """
    key = api_key or os.getenv("XC_API_KEY", "demo")
    species_dir = dest_dir / _safe_folder_name(scientific_name)
    species_dir.mkdir(parents=True, exist_ok=True)

    existing_audio = sorted([p for p in species_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav"}])
    existing_xc = sorted(species_dir.glob("XC*.mp3"))
    processed_names = _load_processed_recordings_for_species(scientific_name) if skip_processed else set()
    known_names = {p.name for p in existing_audio} | processed_names
    if verbose:
        print(
            f"[{scientific_name}] existing_total={len(existing_audio)} "
            f"processed={len(processed_names)} existing_xc={len(existing_xc)} target={max_recordings}"
        )
    if len(known_names) >= max_recordings:
        if verbose:
            print(f"[{scientific_name}] already complete, download=0")
        return existing_audio[:max_recordings]

    saved: list[Path] = list(existing_audio)
    saved_names = set(known_names)
    page = 1
    skipped_quality = 0
    skipped_duplicate = 0
    skipped_processed = 0
    failed = 0
    downloaded_new = 0

    quality_query = _quality_query_from_set(quality)
    base_query = f'sp:"{scientific_name}" grp:birds'
    if quality_query:
        base_query = f"{base_query} {quality_query}"

    while len(saved) < max_recordings:
        params = {
            "query": base_query,
            "key": key,
            "per_page": 100,
            "page": page,
        }
        resp = requests.get(XC_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recordings = data.get("recordings", [])
        if not recordings:
            break

        for rec in recordings:
            if rec.get("q", "") not in quality:
                skipped_quality += 1
                continue
            file_url = rec.get("file", "")
            if not file_url:
                continue
            if not file_url.startswith("http"):
                file_url = "https:" + file_url

            xc_id = rec.get("id", "unknown")
            filename = f"XC{xc_id}.mp3"
            dest = species_dir / filename

            if filename in saved_names:
                skipped_duplicate += 1
                if filename in processed_names and not dest.exists():
                    skipped_processed += 1
                continue

            try:
                audio_resp = requests.get(file_url, timeout=60, stream=True)
                audio_resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in audio_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                saved.append(dest)
                saved_names.add(filename)
                downloaded_new += 1
            except Exception as e:
                print(f"  Failed {filename}: {e}")
                failed += 1
                continue

            if len(saved) >= max_recordings:
                break

            time.sleep(0.3)

        if page >= int(data.get("numPages", 1)):
            break
        page += 1

    if verbose:
        print(
            f"[{scientific_name}] downloaded_new={downloaded_new} total_now={len(saved)} "
            f"missing_after={max(0, max_recordings - len(saved))} "
            f"skipped_duplicate={skipped_duplicate} skipped_processed={skipped_processed} "
            f"skipped_quality={skipped_quality} failed={failed}"
        )
    return saved[:max_recordings]


def download_species_list(
    species_names: list[str],
    max_recordings: int = 100,
    dest_dir: Path = SPECIES_DIR,
    verbose: bool = True,
    skip_processed: bool = True,
) -> dict[str, list[Path]]:
    results: dict[str, list[Path]] = {}
    for name in tqdm(species_names, desc="Downloading species"):
        paths = download_species(
            name,
            max_recordings=max_recordings,
            dest_dir=dest_dir,
            verbose=verbose,
            skip_processed=skip_processed,
        )
        results[name] = paths
        if verbose:
            print(f"  {name}: {len(paths)} files")
    return results
