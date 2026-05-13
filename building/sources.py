"""Audio-recording sources for the per-species download pipeline.

A `Source` knows how to (a) list recordings for a scientific name and (b) fetch
the audio file for one entry. Both XenoCanto and eBird/Macaulay Library are
implemented here so [`download.py`](building/download.py) can drive either with
the same BirdNET-extraction loop.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from xenocanto3 import (
    AsyncXenoCantoClient,
    Group,
    Recording as XCRecording,
    XenoCantoClient,
)

from building.taxonomy import MINIMUM_QUALITY


@dataclass
class RecordingEntry:
    rec_id: str       # source-side identifier (XC: numeric id, ML: asset id)
    filename: str     # local filename, e.g. "XC739081.mp3" / "ML89108201.mp3"
    file_url: str     # direct audio URL
    quality: str = "" # XC quality grade, or ML rating; "" if unknown


class Source:
    """Base class. Subclasses must set `name` and `prefix` and implement
    `fetch_listing` + `download`."""

    name: str = ""
    prefix: str = ""
    # Seconds to wait between consecutive downloads (rate-limit politeness).
    dl_interval: float = 0.0

    async def fetch_listing(self, scientific_name: str) -> list[RecordingEntry]:
        raise NotImplementedError

    def download(self, entry: RecordingEntry, out_path: Path) -> None:
        """Blocking download to `out_path`. Called from a worker thread, so
        the implementation must be thread-safe."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- XenoCanto --------------------------------------------------------------

class XCSource(Source):
    name = "xc"
    prefix = "XC"
    dl_interval = 0.2

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("XC_API_KEY", "demo")
        self._sync = XenoCantoClient(api_key=self._api_key).__enter__()

    async def fetch_listing(self, scientific_name: str) -> list[RecordingEntry]:
        rows: list[RecordingEntry] = []
        async with AsyncXenoCantoClient(api_key=self._api_key) as ac:
            builder = (
                ac.query().sp(scientific_name).grp(Group.BIRDS).q_min(MINIMUM_QUALITY)
            )
            async for rec in builder.all():
                url = (rec.file or "").strip()
                if url and not url.startswith("http"):
                    url = "https:" + url
                rows.append(
                    RecordingEntry(
                        rec_id=str(rec.id),
                        filename=f"XC{rec.id}.mp3",
                        file_url=url,
                        quality=str(rec.q),
                    )
                )
        return rows

    def download(self, entry: RecordingEntry, out_path: Path) -> None:
        self._sync.download(
            XCRecording(id=int(entry.rec_id), file=entry.file_url),
            out_path,
            overwrite=False,
        )

    def close(self) -> None:
        self._sync.__exit__(None, None, None)


# -- eBird / Macaulay Library ----------------------------------------------

ML_SEARCH_URL = "https://media.ebird.org/api/v1/search"
ML_ASSET_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{}/audio"
_EBIRD_UA = "Mozilla/5.0 (compatible; multi-chirp/dataset-builder)"


class EBirdSource(Source):
    """Pulls audio from the Macaulay Library, looking species codes up via the
    official eBird API (uses `EBIRD_API_KEY` from env).

    The eBird API itself does not host media; the asset metadata + audio bytes
    come from `media.ebird.org` and the Cornell CDN. No auth is required for
    those two endpoints, but we keep a Session for connection reuse.
    """

    name = "ebird"
    prefix = "ML"
    # Cornell hasn't documented a hard rate limit on the public CDN; stay light.
    dl_interval = 0.1

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_per_species: int = 10_000,
        page_size: int = 100,
        sort: str = "rating_rank_desc",
    ) -> None:
        self._token = api_key or os.getenv("EBIRD_API_KEY")
        if not self._token:
            raise RuntimeError("EBIRD_API_KEY not set; cannot use EBirdSource.")
        self._max = max_per_species
        self._page_size = page_size
        self._sort = sort
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _EBIRD_UA
        self._tax_cache: dict[str, str] = {}  # sci_name (lower) -> speciesCode
        self._tax_loaded = False

    def _ensure_taxonomy(self) -> None:
        if self._tax_loaded:
            return
        from ebird.api.requests import get_taxonomy

        for row in get_taxonomy(self._token, locale="en"):
            self._tax_cache[row["sciName"].lower()] = row["speciesCode"]
        self._tax_loaded = True

    def species_code(self, scientific_name: str) -> str:
        self._ensure_taxonomy()
        code = self._tax_cache.get(scientific_name.lower())
        if not code:
            raise KeyError(f"No eBird species code for {scientific_name!r}")
        return code

    def _fetch_listing_sync(self, scientific_name: str) -> list[RecordingEntry]:
        code = self.species_code(scientific_name)
        rows: list[RecordingEntry] = []
        cursor: str | None = None
        while len(rows) < self._max:
            params: dict[str, object] = {
                "taxonCode": code,
                "mediaType": "audio",
                "count": self._page_size,
                "sort": self._sort,
            }
            if cursor:
                params["initialCursorMark"] = cursor
            r = self._session.get(ML_SEARCH_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("results", {})
            items = data.get("content") or []
            if not items:
                break
            for it in items:
                asset_id = str(it["assetId"])
                rows.append(
                    RecordingEntry(
                        rec_id=asset_id,
                        filename=f"ML{asset_id}.mp3",
                        file_url=ML_ASSET_URL.format(asset_id),
                        quality=str(it.get("rating", "")),
                    )
                )
                if len(rows) >= self._max:
                    break
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return rows

    async def fetch_listing(self, scientific_name: str) -> list[RecordingEntry]:
        return await asyncio.to_thread(self._fetch_listing_sync, scientific_name)

    def download(self, entry: RecordingEntry, out_path: Path) -> None:
        if out_path.exists() and out_path.stat().st_size > 0:
            return
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with self._session.get(entry.file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
        tmp.rename(out_path)

    def close(self) -> None:
        self._session.close()


def make_source(name: str, **kwargs: object) -> Source:
    if name == "xc":
        return XCSource(**kwargs)  # type: ignore[arg-type]
    if name == "ebird":
        return EBirdSource(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown source {name!r}")
