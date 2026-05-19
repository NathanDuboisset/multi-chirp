"""Area-scoped XC exploration: heatmaps, taxonomy, BirdNET pass."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import folium
import pandas as pd
import pyrootutils
from birdnetlib import Recording as BirdNETRecording
from dotenv import load_dotenv
from folium.plugins import HeatMap
from tqdm.auto import tqdm
from xenocanto3 import AsyncXenoCantoClient, Group, Quality, Recording

from building.data.download import (
    BIRDNET_THRESHOLD,
    POOL_SIZE,
    get_analyzer,
    quiet,
)
from building.data.taxonomy import MINIMUM_QUALITY

XC_DL_INTERVAL = 0.2  # seconds between successive XC downloads (rate limit)

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
load_dotenv()

RAW_DATASET_DIR = ROOT / "raw_dataset"
TAXONOMY_FILE = RAW_DATASET_DIR / "taxonomy.csv"
ANALYSIS_DIR = RAW_DATASET_DIR / "analysis"

BBox = tuple[float, float, float, float]  # (lat_min, lon_min, lat_max, lon_max)


def bbox_from_radius(lat: float, lon: float, radius_km: float) -> BBox:
    """Center + radius to (lat_min, lon_min, lat_max, lon_max)."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def _api_key() -> str:
    return os.getenv("XC_API_KEY", "demo")


async def _collect(builder) -> list[Recording]:
    return [rec async for rec in builder.all()]


async def fetch_species_global(
    species_names: list[str],
    min_quality: Quality = MINIMUM_QUALITY,
) -> dict[str, list[Recording]]:
    """One XC query per species, no area filter. Used for global heatmaps."""
    async with AsyncXenoCantoClient(api_key=_api_key()) as xc:
        builders = [
            xc.query().sp(name).grp(Group.BIRDS).q_min(min_quality)
            for name in species_names
        ]
        results = await asyncio.gather(*(_collect(b) for b in builders))
    return dict(zip(species_names, results, strict=True))


async def fetch_area_recordings(
    bbox: BBox,
    min_quality: Quality = MINIMUM_QUALITY,
) -> list[Recording]:
    """All bird recordings inside the bbox."""
    async with AsyncXenoCantoClient(api_key=_api_key()) as xc:
        builder = (
            xc.query()
            .grp(Group.BIRDS)
            .box(*bbox)
            .q_min(min_quality)
        )
        return await _collect(builder)


def recordings_to_df(recordings: Iterable[Recording]) -> pd.DataFrame:
    rows = [
        {
            "id": r.id,
            "scientific_name": r.scientific_name,
            "gen": r.gen,
            "sp": r.sp,
            "en": r.en,
            "lat": r.lat,
            "lon": r.lon,
            "q": r.q,
            "length_seconds": r.length_seconds,
            "cnt": r.cnt,
            "loc": r.loc,
            "date": r.date,
            "also": list(r.also),
            "annotation_set": r.annotation_set,
            "file": r.file,
        }
        for r in recordings
    ]
    return pd.DataFrame(rows)


def _layer_color(i: int) -> str:
    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "cadetblue", "darkgreen", "darkpurple", "pink",
    ]
    return palette[i % len(palette)]


def build_heatmap(
    per_species: dict[str, pd.DataFrame],
    center: tuple[float, float],
    bbox: BBox | None = None,
    zoom_start: int = 6,
) -> folium.Map:
    """Per-species toggleable heat layers, with optional bbox rectangle."""
    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="cartodbpositron")

    for i, (species, df) in enumerate(per_species.items()):
        pts = df.dropna(subset=["lat", "lon"])[["lat", "lon"]].values.tolist()
        fg = folium.FeatureGroup(name=f"{species} (n={len(pts)})", show=True)
        if pts:
            HeatMap(pts, radius=12, blur=18, min_opacity=0.4).add_to(fg)
        folium.CircleMarker(
            location=list(center),
            radius=4,
            color=_layer_color(i),
            fill=True,
        ).add_to(fg)
        fg.add_to(m)

    if bbox is not None:
        lat_min, lon_min, lat_max, lon_max = bbox
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="black",
            weight=2,
            fill=False,
            popup="Analysis area",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _load_taxonomy() -> pd.DataFrame:
    if not TAXONOMY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TAXONOMY_FILE}. Run anything from building.data.taxonomy first to cache it."
        )
    df = pd.read_csv(TAXONOMY_FILE)
    df = df[df["CATEGORY"] == "species"][["ORDER", "FAMILY", "SCI_NAME"]].copy()
    df.columns = ["order", "family", "scientific_name"]
    df["genus"] = df["scientific_name"].str.split().str[0]
    return df


def taxonomy_breakdown(
    area_df: pd.DataFrame, min_recordings: int = 1
) -> dict[str, pd.DataFrame]:
    """Per-level counts + totals. Species with fewer than ``min_recordings``
    recordings in the area are dropped before counting, so genus/family/order
    totals reflect only the species that survive the filter."""
    tax = _load_taxonomy()
    joined = area_df.merge(tax, on="scientific_name", how="left")
    if min_recordings > 1:
        per_species = joined.groupby("scientific_name").size()
        kept = per_species[per_species >= min_recordings].index
        joined = joined[joined["scientific_name"].isin(kept)]

    by_order = (
        joined.groupby("order", dropna=False)
        .agg(species=("scientific_name", "nunique"), recordings=("id", "count"))
        .sort_values("recordings", ascending=False)
        .reset_index()
    )
    by_family = (
        joined.groupby(["order", "family"], dropna=False)
        .agg(species=("scientific_name", "nunique"), recordings=("id", "count"))
        .sort_values("recordings", ascending=False)
        .reset_index()
    )
    by_genus = (
        joined.groupby(["family", "genus"], dropna=False)
        .agg(species=("scientific_name", "nunique"), recordings=("id", "count"))
        .sort_values("recordings", ascending=False)
        .reset_index()
    )
    totals = pd.DataFrame(
        [
            {
                "n_orders": joined["order"].nunique(dropna=True),
                "n_families": joined["family"].nunique(dropna=True),
                "n_genera": joined["genus"].nunique(dropna=True),
                "n_species": joined["scientific_name"].nunique(),
                "n_recordings": len(joined),
                "n_unmatched_in_taxonomy": int(joined["order"].isna().sum()),
            }
        ]
    )
    return {"totals": totals, "by_order": by_order, "by_family": by_family, "by_genus": by_genus}


def presence_in_area(species_names: list[str], area_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_name = area_df.groupby("scientific_name")
    for name in species_names:
        if name in by_name.groups:
            sub = by_name.get_group(name)
            rows.append(
                {
                    "scientific_name": name,
                    "recordings_in_area": len(sub),
                    "qualities": "".join(sorted(sub["q"].dropna().unique())),
                    "mean_length_sec": round(sub["length_seconds"].mean(), 1),
                }
            )
        else:
            rows.append(
                {
                    "scientific_name": name,
                    "recordings_in_area": 0,
                    "qualities": "",
                    "mean_length_sec": 0.0,
                }
            )
    return pd.DataFrame(rows)


def co_occurrence_counts(area_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for also in area_df["also"]:
        counter.update(s.strip() for s in also if s and s.strip())
    items = counter.most_common(top_n)
    return pd.DataFrame(items, columns=["species", "count"])


def annotation_summary(area_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows: list[dict] = []
    for _, r in area_df.iterrows():
        ann_set = r["annotation_set"]
        if ann_set is None:
            continue
        for a in ann_set.annotations:
            rows.append(
                {
                    "xc_id": r["id"],
                    "recording_species": r["scientific_name"],
                    "annotated_species": a.scientific_name,
                    "start": a.start_time,
                    "end": a.end_time,
                    "freq_low": a.frequency_low,
                    "freq_high": a.frequency_high,
                    "sound_type": a.sound_type,
                }
            )
    annotations = pd.DataFrame(rows)
    if annotations.empty:
        by_label = pd.DataFrame(columns=["annotated_species", "count"])
    else:
        by_label = (
            annotations.groupby("annotated_species")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    return {"annotations": annotations, "by_species": by_label}


AUDIO_CACHE_DIR = ANALYSIS_DIR / "audio"
BIRDNET_CACHE_DIR = ANALYSIS_DIR / "birdnet_cache"


def area_audio_cache_dir(
    lat: float,
    lon: float,
    radius_km: float,
    base: Path = AUDIO_CACHE_DIR,
) -> Path:
    """Per-area audio cache folder so multiple areas don't share mp3 files."""
    return base / f"r{radius_km:g}km_lat{lat:.4f}_lon{lon:.4f}"


def _cache_path(xc_id: int) -> Path:
    return BIRDNET_CACHE_DIR / f"XC{xc_id}.json"


def _load_cached_detections(xc_id: int, min_confidence: float) -> list[dict] | None:
    p = _cache_path(xc_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cached = float(data.get("min_confidence", 1.0))
    if cached > min_confidence:
        return None  # cached at a stricter threshold than asked for; would miss detections
    return [d for d in data.get("detections", []) if float(d.get("confidence", 0.0)) >= min_confidence]


def _save_cached_detections(xc_id: int, min_confidence: float, detections: list[dict]) -> None:
    BIRDNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slim = [
        {
            "scientific_name": d.get("scientific_name", ""),
            "common_name": d.get("common_name", ""),
            "confidence": float(d.get("confidence", 0.0)),
            "start_time": float(d.get("start_time", 0.0)),
            "end_time": float(d.get("end_time", 0.0)),
        }
        for d in detections
    ]
    _cache_path(xc_id).write_text(
        json.dumps({"min_confidence": min_confidence, "detections": slim})
    )


async def birdnet_area_analysis(
    area_df: pd.DataFrame,
    max_recordings: int | None = None,
    min_confidence: float = BIRDNET_THRESHOLD,
    cache_dir: Path = AUDIO_CACHE_DIR,
    pool_size: int = POOL_SIZE,
    delete_audio_after_cache: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download area recordings (cached) and BirdNET each. Returns
    {detections, by_species, hidden}."""
    work = area_df.dropna(subset=["file"]).reset_index(drop=True)
    if max_recordings is not None:
        work = work.head(max_recordings)

    needs_audio = any(
        _load_cached_detections(int(row.id), min_confidence) is None
        for row in work.itertuples()
    )
    if needs_audio:
        cache_dir.mkdir(parents=True, exist_ok=True)

    detections: list[dict] = []
    dl_lock = asyncio.Lock()
    dl_last = [0.0]
    sem = asyncio.Semaphore(pool_size)
    pbar = tqdm(total=len(work), desc="BirdNET analyse")

    async with AsyncXenoCantoClient(api_key=_api_key()) as xc:

        async def _process(row) -> None:
            xc_id = int(row.id)
            cached = _load_cached_detections(xc_id, min_confidence)
            if cached is not None:
                dets = cached
            else:
                audio_path = cache_dir / f"XC{xc_id}.mp3"
                try:
                    async with dl_lock:
                        now = asyncio.get_event_loop().time()
                        wait = XC_DL_INTERVAL - (now - dl_last[0])
                        if wait > 0:
                            await asyncio.sleep(wait)
                        dl_last[0] = asyncio.get_event_loop().time()
                    async with sem:
                        if not audio_path.exists():
                            await xc.download(
                                Recording(id=xc_id, file=row.file),
                                audio_path,
                                overwrite=False,
                            )
                        dets = await asyncio.to_thread(_run_birdnet, audio_path, min_confidence)
                except Exception as e:
                    print(f"[XC{xc_id}] skipped: {e}")
                    pbar.update(1)
                    return
                _save_cached_detections(xc_id, min_confidence, dets)
                if delete_audio_after_cache:
                    audio_path.unlink(missing_ok=True)

            primary = (row.scientific_name or "").strip().lower()
            declared_set = {primary} | {s.strip().lower() for s in (row.also or []) if s}
            for d in dets:
                sci = (d.get("scientific_name") or "").strip()
                detections.append(
                    {
                        "xc_id": xc_id,
                        "recording_species": row.scientific_name,
                        "detected_species": sci,
                        "common_name": d.get("common_name", ""),
                        "confidence": float(d.get("confidence", 0.0)),
                        "start_time": float(d.get("start_time", 0.0)),
                        "end_time": float(d.get("end_time", 0.0)),
                        "declared": sci.lower() in declared_set,
                    }
                )
            pbar.update(1)

        await asyncio.gather(*[_process(row) for row in work.itertuples()])

    pbar.close()
    det_df = pd.DataFrame(detections)
    if det_df.empty:
        empty = pd.DataFrame()
        return {"detections": det_df, "by_species": empty, "hidden": empty}

    by_species = (
        det_df.groupby("detected_species")
        .agg(
            n_detections=("xc_id", "count"),
            n_recordings=("xc_id", "nunique"),
            n_undeclared=("declared", lambda s: int((~s).sum())),
            mean_conf=("confidence", "mean"),
            max_conf=("confidence", "max"),
        )
        .reset_index()
        .sort_values("n_detections", ascending=False)
    )
    by_species["mean_conf"] = by_species["mean_conf"].round(3)
    by_species["max_conf"] = by_species["max_conf"].round(3)

    hidden = (
        by_species[by_species["n_undeclared"] > 0]
        .sort_values("n_undeclared", ascending=False)
        .reset_index(drop=True)
    )
    return {"detections": det_df, "by_species": by_species, "hidden": hidden}


def combined_species_table(
    area_df: pd.DataFrame,
    birdnet_detections: pd.DataFrame,
    min_recordings: int = 1,
) -> pd.DataFrame:
    """Per-species evidence: xc_primary, xc_also, birdnet_only (mutually
    exclusive per recording). Rows kept: total >= min_recordings."""
    xc_primary = (
        area_df.groupby("scientific_name").size().rename("xc_primary")
    )

    also = area_df[["id", "also"]].explode("also").dropna(subset=["also"])
    also["also"] = also["also"].astype(str).str.strip()
    also = also[also["also"] != ""]
    xc_also = (
        also.drop_duplicates(["id", "also"])
        .groupby("also")
        .size()
        .rename("xc_also")
    )
    xc_also.index.name = "scientific_name"

    if birdnet_detections.empty:
        birdnet_only = pd.Series(dtype=int, name="birdnet_only")
    else:
        undecl = birdnet_detections[~birdnet_detections["declared"]]
        birdnet_only = (
            undecl.drop_duplicates(["xc_id", "detected_species"])
            .groupby("detected_species")
            .size()
            .rename("birdnet_only")
        )
    birdnet_only.index.name = "scientific_name"

    merged = (
        pd.concat([xc_primary, xc_also, birdnet_only], axis=1)
        .fillna(0)
        .astype(int)
    )
    merged["total"] = merged.sum(axis=1)
    merged = merged[merged["total"] >= min_recordings]
    return (
        merged.reset_index()
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def longest_recordings(
    area_df: pd.DataFrame,
    center: tuple[float, float],
    top_n: int = 10,
) -> pd.DataFrame:
    """Top-N longest recordings in the area, with haversine distance (km) to ``center``."""
    df = area_df.dropna(subset=["length_seconds"]).copy()
    lat0, lon0 = center
    df["distance_km"] = [
        _haversine_km(lat0, lon0, lat, lon) if pd.notna(lat) and pd.notna(lon) else float("nan")
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    df = df.sort_values("length_seconds", ascending=False).head(top_n)
    out = df[["id", "scientific_name", "en", "length_seconds", "distance_km", "q", "loc", "date"]].copy()
    out["length_seconds"] = out["length_seconds"].round(1)
    out["distance_km"] = out["distance_km"].round(2)
    return out.reset_index(drop=True)


def _run_birdnet(audio_path: Path, min_confidence: float) -> list[dict]:
    with quiet():
        bn = BirdNETRecording(get_analyzer(), str(audio_path), min_conf=min_confidence)
        bn.analyze()
    return list(bn.detections)
