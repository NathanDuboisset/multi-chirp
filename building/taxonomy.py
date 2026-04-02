"""
taxonomy.py — download/cache bird taxonomy and select species for experiments.

Uses the eBird/Clements taxonomy CSV (Cornell Lab) for order/family/genus/species
hierarchy, and queries XC API v3 to filter species with enough recordings.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

EBIRD_TAXONOMY_URL = (
    "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2024/10/"
    "eBird_Taxonomy_v2024.csv"
)
RAW_DATASET_DIR = Path(__file__).parent.parent / "raw_dataset"
TAXONOMY_CACHE = RAW_DATASET_DIR / "taxonomy_cache.json"
XC_API_BASE = "https://xeno-canto.org/api/3/recordings"


class SpeciesInfo(BaseModel):
    scientific_name: str  # "Genus species"
    genus: str
    species_epithet: str
    family: str
    order: str
    num_recordings: int


def _fetch_ebird_taxonomy() -> pd.DataFrame:
    resp = requests.get(EBIRD_TAXONOMY_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Keep only species-level entries for birds
    df = df[df["CATEGORY"] == "species"].copy()
    df = df[["ORDER1", "FAMILY", "GENUS", "SPECIES_CODE", "SCI_NAME"]].copy()
    df.columns = ["order", "family", "genus", "code", "scientific_name"]
    df["species_epithet"] = df["scientific_name"].str.split().str[1]
    return df.dropna(subset=["genus", "species_epithet"])


def _query_xc_recording_count(genus: str, epithet: str, api_key: str) -> int:
    params = {
        "query": f'sp:"{genus} {epithet}" grp:birds',
        "key": api_key,
        "per_page": 1,
    }
    try:
        resp = requests.get(XC_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("numRecordings", 0))
    except Exception:
        return 0


def _load_or_build_cache(
    min_recordings: int = 100,
    max_species: Optional[int] = None,
    force_refresh: bool = False,
) -> list[SpeciesInfo]:
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if TAXONOMY_CACHE.exists() and not force_refresh:
        with open(TAXONOMY_CACHE) as f:
            raw = json.load(f)
        species = [SpeciesInfo(**s) for s in raw]
        return [s for s in species if s.num_recordings >= min_recordings]

    api_key = os.getenv("XC_API_KEY", "demo")
    df = _fetch_ebird_taxonomy()
    if max_species:
        df = df.head(max_species)

    results: list[SpeciesInfo] = []
    for _, row in df.iterrows():
        count = _query_xc_recording_count(row["genus"], row["species_epithet"], api_key)
        time.sleep(0.5)  # be polite to the API
        if count >= min_recordings:
            results.append(
                SpeciesInfo(
                    scientific_name=row["scientific_name"],
                    genus=row["genus"],
                    species_epithet=row["species_epithet"],
                    family=row["family"],
                    order=row["order"],
                    num_recordings=count,
                )
            )

    with open(TAXONOMY_CACHE, "w") as f:
        json.dump([s.model_dump() for s in results], f, indent=2)

    return results


def get_species_with_recordings(min_recordings: int = 100, force_refresh: bool = False) -> list[SpeciesInfo]:
    return _load_or_build_cache(min_recordings=min_recordings, force_refresh=force_refresh)


def select_same_genus(genus: str, n: int, species_pool: Optional[list[SpeciesInfo]] = None) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    candidates = [s for s in pool if s.genus.lower() == genus.lower()]
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} species found for genus {genus}, need {n}")
    candidates.sort(key=lambda s: -s.num_recordings)
    return candidates[:n]


def select_same_family(family: str, n: int, species_pool: Optional[list[SpeciesInfo]] = None) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    candidates = [s for s in pool if s.family.lower() == family.lower()]
    # Pick one species per genus (different genera)
    seen_genera: set[str] = set()
    selected: list[SpeciesInfo] = []
    for s in sorted(candidates, key=lambda s: -s.num_recordings):
        if s.genus not in seen_genera:
            seen_genera.add(s.genus)
            selected.append(s)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} distinct genera found in family {family}, need {n}")
    return selected


def select_same_order(order: str, n: int, species_pool: Optional[list[SpeciesInfo]] = None) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    candidates = [s for s in pool if s.order.lower() == order.lower()]
    seen_families: set[str] = set()
    selected: list[SpeciesInfo] = []
    for s in sorted(candidates, key=lambda s: -s.num_recordings):
        if s.family not in seen_families:
            seen_families.add(s.family)
            selected.append(s)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} distinct families found in order {order}, need {n}")
    return selected


def select_diff_order(n: int, species_pool: Optional[list[SpeciesInfo]] = None) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    seen_orders: set[str] = set()
    selected: list[SpeciesInfo] = []
    for s in sorted(pool, key=lambda s: -s.num_recordings):
        if s.order not in seen_orders:
            seen_orders.add(s.order)
            selected.append(s)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} distinct orders found, need {n}")
    return selected


if __name__ == "__main__":
    pool = get_species_with_recordings()
    print(f"Species with ≥100 XC recordings: {len(pool)}")
    for s in pool[:10]:
        print(f"  {s.scientific_name:40s}  order={s.order}  family={s.family}  n={s.num_recordings}")
