"""Geographic-scale package: slug helper for place-based scaling experiments.

The on-disk dataset / models / results folders are slug-named so the
(place, n_targets) context is visible at the filesystem level. Matches the
``geographic_task`` slug format with a ``scale`` prefix:
``scale_s{n_targets}_{round(lat)}_{round(lon)}_r{round(radius_km)}``.
"""

from __future__ import annotations

import re

from .dataset_info import DatasetInfo, info_path, load_dataset_info, write_dataset_info

__all__ = [
    "place_slug",
    "DatasetInfo",
    "info_path",
    "load_dataset_info",
    "write_dataset_info",
]


def place_slug(place_name: str, n_targets: int) -> str:
    """Build a dataset collection name from a place + target-count.

    ``place_slug("Paris", 10) -> "paris_10"``. Lowercased, non-alnum runs
    collapsed to ``_``. The slug is the single source of truth — the place
    coordinates / radius live in ``dataset.json`` so notebooks don't have to
    redefine them.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", place_name.lower()).strip("_")
    return f"{slug}_{n_targets}"
