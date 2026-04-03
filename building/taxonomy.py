"""
taxonomy.py — download/cache bird taxonomy and select species for experiments.

Uses the eBird/Clements taxonomy CSV (Cornell Lab) for order/family/genus/species
hierarchy, and queries XC API v3 to filter species with enough recordings.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pyrootutils
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
load_dotenv()
# CITATE : https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/october-2025/2025-citation-checklist-downloads/
EBIRD_TAXONOMY_URL = (
    "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2025/10/"
    "eBird_taxonomy_v2025.csv"
)
RAW_DATASET_DIR = ROOT / "raw_dataset"
TAXONOMY_CACHE = RAW_DATASET_DIR / "taxonomy_cache.csv"
XC_API_BASE = "https://xeno-canto.org/api/3/recordings"


class SpeciesInfo(BaseModel):
    scientific_name: str  # "Genus species"
    genus: str
    species_epithet: str
    family: str
    order: str
    num_recordings: int


def _fetch_ebird_taxonomy() -> pd.DataFrame:
    taxonomy_file = RAW_DATASET_DIR / "taxonomy.csv"
    
    if not taxonomy_file.exists():
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(EBIRD_TAXONOMY_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        with open(taxonomy_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
            
    df = pd.read_csv(taxonomy_file)
    
    df = df[df["CATEGORY"] == "species"].copy()
    df = df[["ORDER", "FAMILY", "SCI_NAME"]].copy()
    df.columns = ["order", "family", "scientific_name"]
    df["genus"] = df["scientific_name"].str.split().str[0]
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
    dl_more: bool = True,
) -> list[SpeciesInfo]:
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    cached_data = {}
    if force_refresh and TAXONOMY_CACHE.exists():
        TAXONOMY_CACHE.unlink()
    if TAXONOMY_CACHE.exists() and not force_refresh:
        try:
            for _, row in pd.read_csv(TAXONOMY_CACHE).iterrows():
                cached_data[row["scientific_name"]] = row.to_dict()
        except Exception as e:
            print(f"Could not load existing cache: {e}")

    api_key = os.getenv("XC_API_KEY", "demo")
    df = _fetch_ebird_taxonomy()
    if max_species:
        df = df.head(max_species)
    if dl_more and not api_key:
        raise ValueError("XC_API_KEY is not set")

    results: list[SpeciesInfo] = []
    
    new_queries = 0
    total_species = len(df)

    pending_rows: list[dict] = []

    def flush_pending_rows() -> None:
        if not pending_rows:
            return
        # Append only newly queried rows to avoid rewriting the full CSV.
        pd.DataFrame(pending_rows).to_csv(
            TAXONOMY_CACHE,
            mode="a",
            index=False,
            header=not TAXONOMY_CACHE.exists(),
        )
        pending_rows.clear()
    
    pbar = tqdm(df.iterrows(), total=total_species, desc="Species")
    for _, row in pbar:
        sci_name = row["scientific_name"]
        
        if sci_name in cached_data and not force_refresh:
            s_info = SpeciesInfo(**cached_data[sci_name])
            pbar.set_description(f"{sci_name} (cached)")
        elif dl_more:
            pbar.set_description(f"{sci_name} (querying)")
            count = _query_xc_recording_count(row["genus"], row["species_epithet"], api_key)
            pbar.set_postfix(found=count)
            time.sleep(0.5)  # be polite to the API
            
            s_info = SpeciesInfo(
                scientific_name=sci_name,
                genus=row["genus"],
                species_epithet=row["species_epithet"],
                family=row["family"],
                order=row["order"],
                num_recordings=count,
            )
            cached_data[sci_name] = s_info.model_dump()
            new_queries += 1
            pending_rows.append(cached_data[sci_name])
            if len(pending_rows) >= 10:
                flush_pending_rows()
        else:
            pbar.set_description(f"{sci_name} (missing in cache)")
            continue

        results.append(s_info)

    # Flush any remaining newly queried rows.
    flush_pending_rows()

    return [s for s in results if s.num_recordings >= min_recordings]


def get_species_with_recordings(min_recordings: int = 100, force_refresh: bool = False, dl_more: bool = True) -> list[SpeciesInfo]:
    return _load_or_build_cache(min_recordings=min_recordings, force_refresh=force_refresh, dl_more=dl_more)


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


def get_potential_taxa(
    taxon_level: str,
    n: int,
    min_recordings: int = 100,
    species_pool: Optional[list[SpeciesInfo]] = None,
    force_refresh: bool = False,
    top_k: Optional[int] = 50,
) -> pd.DataFrame:
    """
    List candidate taxa that can support choosing `n` items with `min_recordings`.
    """
    taxon_level_lower = taxon_level.lower()
    if taxon_level_lower not in {"genus", "family", "order"}:
        raise ValueError("taxon_level must be one of: 'genus', 'family', 'order'")

    pool = species_pool or get_species_with_recordings(
        min_recordings=min_recordings,
        force_refresh=force_refresh,
    )

    groups: dict[str, list[SpeciesInfo]] = {}
    for s in pool:
        key = getattr(s, taxon_level_lower)
        groups.setdefault(key, []).append(s)

    rows: list[dict] = []
    for key, items in groups.items():
        n_species = len(items)
        n_distinct_genus = len({s.genus for s in items})
        n_distinct_family = len({s.family for s in items})
        n_distinct_order = len({s.order for s in items})

        if taxon_level_lower == "genus":
            required_metric = "n_species"
            required_value = n_species
        elif taxon_level_lower == "family":
            required_metric = "n_distinct_genus"
            required_value = n_distinct_genus
        else:  # order
            required_metric = "n_distinct_family"
            required_value = n_distinct_family

        if required_value >= n:
            rows.append(
                {
                    taxon_level_lower: key,
                    "n_species": n_species,
                    "n_distinct_genus": n_distinct_genus,
                    "n_distinct_family": n_distinct_family,
                    "n_distinct_order": n_distinct_order,
                    # used for sorting/debugging
                    "_required_metric": required_metric,
                    "_required_value": required_value,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        # Ensure stable columns even when empty.
        return pd.DataFrame(
            columns=[
                taxon_level_lower,
                "n_species",
                "n_distinct_genus",
                "n_distinct_family",
                "n_distinct_order",
            ]
        )

    required_col = {
        "genus": "n_species",
        "family": "n_distinct_genus",
        "order": "n_distinct_family",
    }[taxon_level_lower]

    df = df.sort_values([required_col, "n_species"], ascending=[False, False]).reset_index(drop=True)
    if top_k:
        df = df.head(top_k).reset_index(drop=True)

    return df.drop(columns=["_required_metric", "_required_value"])


if __name__ == "__main__":
    pool = get_species_with_recordings()
    print(f"Species with ≥100 XC recordings: {len(pool)}")
    for s in pool[:10]:
        print(f"  {s.scientific_name:40s}  order={s.order}  family={s.family}  n={s.num_recordings}")
