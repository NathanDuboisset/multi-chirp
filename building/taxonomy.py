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

MINIMUM_QUALITY: str = "B"

XC_QUALITY_FILTER: dict[str, str] = {
    "A": "q:A",
    "B": 'q:">C"',
    "C": 'q:">D"',
}


class SpeciesInfo(BaseModel):
    scientific_name: str
    genus: str
    species_epithet: str
    family: str
    order: str
    num_recordings_a: int = 0
    num_recordings_b: int = 0
    num_recordings_c: int = 0

    def recordings_at_least(self, min_quality: str = MINIMUM_QUALITY) -> int:
        """Cumulative count for quality >= min_quality (A ⊇ B ⊇ C)."""
        total = self.num_recordings_a
        if min_quality in ("B", "C"):
            total += self.num_recordings_b
        if min_quality == "C":
            total += self.num_recordings_c
        return total

    @property
    def num_recordings(self) -> int:
        return self.recordings_at_least(MINIMUM_QUALITY)


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


def _query_xc_counts(genus: str, epithet: str, api_key: str) -> dict[str, int]:
    """Query XC for quality-A, B, C recording counts. Returns {a, b, c}."""
    counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}
    for key_q, q_filter in (("a", "q:A"), ("b", "q:B"), ("c", "q:C")):
        try:
            resp = requests.get(
                XC_API_BASE,
                params={
                    "query": f'sp:"{genus} {epithet}" grp:birds {q_filter}',
                    "key": api_key,
                    "per_page": 1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            counts[key_q] = int(resp.json().get("numRecordings", 0))
        except Exception:
            pass
        time.sleep(0.2)
    return counts


def _load_cache() -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not TAXONOMY_CACHE.exists():
        return cached
    try:
        df = pd.read_csv(TAXONOMY_CACHE)
        for _, row in df.iterrows():
            d = row.to_dict()
            cached[d["scientific_name"]] = d
    except Exception as e:
        print(f"Could not load cache: {e}")
    return cached


def get_species_with_recordings(
    min_recordings: int = 100,
    force_refresh: bool = False,
    dl_more: bool = False,
) -> list[SpeciesInfo]:
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if force_refresh and TAXONOMY_CACHE.exists():
        TAXONOMY_CACHE.unlink()

    cached_data = _load_cache()

    def _to_species(d: dict) -> Optional[SpeciesInfo]:
        try:
            return SpeciesInfo(**d)
        except Exception:
            return None

    if not dl_more:
        return [
            s
            for d in cached_data.values()
            if (s := _to_species(d)) and s.num_recordings >= min_recordings
        ]

    api_key = os.getenv("XC_API_KEY", "demo")
    df = _fetch_ebird_taxonomy()
    uncached = df[~df["scientific_name"].isin(cached_data)]
    if uncached.empty:
        return [
            s
            for d in cached_data.values()
            if (s := _to_species(d)) and s.num_recordings >= min_recordings
        ]

    pending: list[dict] = []

    def _flush() -> None:
        if not pending:
            return
        pd.DataFrame(pending).to_csv(
            TAXONOMY_CACHE, mode="a", index=False, header=not TAXONOMY_CACHE.exists()
        )
        pending.clear()

    pbar = tqdm(uncached.iterrows(), total=len(uncached), desc="Querying XC")
    for _, row in pbar:
        sci_name = row["scientific_name"]
        pbar.set_description(sci_name)
        counts = _query_xc_counts(row["genus"], row["species_epithet"], api_key)
        pbar.set_postfix(A=counts["a"], B=counts["b"], C=counts["c"])
        entry = dict(
            scientific_name=sci_name,
            genus=row["genus"],
            species_epithet=row["species_epithet"],
            family=row["family"],
            order=row["order"],
            num_recordings_a=counts["a"],
            num_recordings_b=counts["b"],
            num_recordings_c=counts["c"],
        )
        cached_data[sci_name] = entry
        pending.append(entry)
        if len(pending) >= 10:
            _flush()
    _flush()

    return [
        s
        for d in cached_data.values()
        if (s := _to_species(d)) and s.num_recordings >= min_recordings
    ]


def select_same_genus(
    genus: str,
    n: int,
    species_pool: Optional[list[SpeciesInfo]] = None,
    avoid: Optional[list[str]] = None,
) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    avoid_set = set(avoid or [])
    candidates = [
        s
        for s in pool
        if s.genus.lower() == genus.lower() and s.scientific_name not in avoid_set
    ]
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} species found for genus {genus}, need {n}"
        )
    candidates.sort(key=lambda s: -s.num_recordings)
    return candidates[:n]


def select_same_family(
    family: str,
    n: int,
    species_pool: Optional[list[SpeciesInfo]] = None,
    avoid: Optional[list[str]] = None,
) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    avoid_set = set(avoid or [])
    candidates = [
        s
        for s in pool
        if s.family.lower() == family.lower() and s.scientific_name not in avoid_set
    ]
    seen_genera: set[str] = set()
    selected: list[SpeciesInfo] = []
    for s in sorted(candidates, key=lambda s: -s.num_recordings):
        if s.genus not in seen_genera:
            seen_genera.add(s.genus)
            selected.append(s)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError(
            f"Only {len(selected)} distinct genera found in family {family}, need {n}"
        )
    return selected


def select_same_order(
    order: str,
    n: int,
    species_pool: Optional[list[SpeciesInfo]] = None,
    avoid: Optional[list[str]] = None,
) -> list[SpeciesInfo]:
    pool = species_pool or get_species_with_recordings()
    avoid_set = set(avoid or [])
    candidates = [
        s
        for s in pool
        if s.order.lower() == order.lower() and s.scientific_name not in avoid_set
    ]
    seen_families: set[str] = set()
    selected: list[SpeciesInfo] = []
    for s in sorted(candidates, key=lambda s: -s.num_recordings):
        if s.family not in seen_families:
            seen_families.add(s.family)
            selected.append(s)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError(
            f"Only {len(selected)} distinct families found in order {order}, need {n}"
        )
    return selected


def _min_top_n_rec(items: list[SpeciesInfo], taxon_level: str, n: int) -> Optional[int]:
    """Simulate the greedy top-N selection and return the min recording count."""
    sorted_items = sorted(items, key=lambda s: -s.num_recordings)
    if taxon_level == "genus":
        selected = sorted_items[:n]
    elif taxon_level == "family":
        seen: set[str] = set()
        selected = []
        for s in sorted_items:
            if s.genus not in seen:
                seen.add(s.genus)
                selected.append(s)
            if len(selected) == n:
                break
    else:  # order
        seen_fam: set[str] = set()
        selected = []
        for s in sorted_items:
            if s.family not in seen_fam:
                seen_fam.add(s.family)
                selected.append(s)
            if len(selected) == n:
                break
    return min(s.num_recordings for s in selected) if len(selected) >= n else None


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
                    "min_top_n_rec": _min_top_n_rec(items, taxon_level_lower, n),
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
                "min_top_n_rec",
            ]
        )

    required_col = {
        "genus": "n_species",
        "family": "n_distinct_genus",
        "order": "n_distinct_family",
    }[taxon_level_lower]

    df = df.sort_values(
        [required_col, "n_species"], ascending=[False, False]
    ).reset_index(drop=True)
    if top_k:
        df = df.head(top_k).reset_index(drop=True)

    return df.drop(columns=["_required_metric", "_required_value"])


if __name__ == "__main__":
    pool = get_species_with_recordings()
    print(f"Species with ≥100 XC recordings: {len(pool)}")
    for s in pool[:10]:
        print(
            f"  {s.scientific_name:40s}  order={s.order}  family={s.family}  n={s.num_recordings}"
        )
