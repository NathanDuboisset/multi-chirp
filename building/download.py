"""
download.py — download recordings from Xeno-canto API v3.

Files are saved at their original audio rate; no resampling is done here.
Requires XC_API_KEY in .env.
"""

from __future__ import annotations

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
XC_API_BASE = "https://xeno-canto.org/api/3/recordings"
ACCEPTED_QUALITY = {"A", "B"}


def _safe_folder_name(scientific_name: str) -> str:
    return scientific_name.replace(" ", "_")


def download_species(
    scientific_name: str,
    max_recordings: int = 100,
    quality: set[str] = ACCEPTED_QUALITY,
    dest_dir: Path = SPECIES_DIR,
    api_key: str | None = None,
) -> list[Path]:
    """Download up to `max_recordings` MP3s for `scientific_name` from XC.

    Returns a list of saved file paths.
    Skips files that already exist (idempotent).
    """
    key = api_key or os.getenv("XC_API_KEY", "demo")
    genus, epithet = scientific_name.split(" ", 1)
    species_dir = dest_dir / _safe_folder_name(scientific_name)
    species_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    page = 1
    fetched = 0

    while fetched < max_recordings:
        params = {
            "query": f'sp:"{scientific_name}" grp:birds',
            "key": key,
            "per_page": min(100, max_recordings - fetched),
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
                continue
            file_url = rec.get("file", "")
            if not file_url:
                continue
            if not file_url.startswith("http"):
                file_url = "https:" + file_url

            xc_id = rec.get("id", "unknown")
            filename = f"XC{xc_id}.mp3"
            dest = species_dir / filename

            if dest.exists():
                saved.append(dest)
                fetched += 1
                continue

            try:
                audio_resp = requests.get(file_url, timeout=60, stream=True)
                audio_resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in audio_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                saved.append(dest)
            except Exception as e:
                print(f"  Failed {filename}: {e}")

            fetched += 1
            if fetched >= max_recordings:
                break

            time.sleep(0.3)

        if page >= int(data.get("numPages", 1)):
            break
        page += 1

    return saved


def download_species_list(
    species_names: list[str],
    max_recordings: int = 100,
    dest_dir: Path = SPECIES_DIR,
) -> dict[str, list[Path]]:
    results: dict[str, list[Path]] = {}
    for name in tqdm(species_names, desc="Downloading species"):
        paths = download_species(name, max_recordings=max_recordings, dest_dir=dest_dir)
        results[name] = paths
        print(f"  {name}: {len(paths)} files")
    return results
