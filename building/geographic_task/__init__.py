"""Geographic-task package: slug + path helpers shared by build/train/results notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

CASCADING_SUFFIX = "_cascading"


def task_slug(
    target_species: Sequence[str],
    lat: float,
    lon: float,
    radius_km: float,
    *,
    prefix: str = "task",
) -> str:
    """Canonical (species, area) slug used as the dataset+results folder name.

    Format: ``{prefix}_s{n_species}_{round(lat)}_{round(lon)}_r{round(radius_km)}``
    e.g. for 3 target species at Paris (48.86, 2.35) with 50 km radius:
    ``task_s3_49_2_r50``.
    """
    return (
        f"{prefix}_s{len(target_species)}"
        f"_{round(lat)}_{round(lon)}_r{round(radius_km)}"
    )


def cascading_slug(base: str) -> str:
    """Append the cascading variant suffix used for the on-disk collection."""
    return f"{base}{CASCADING_SUFFIX}"


def base_slug(collection: str) -> str:
    """Strip the cascading suffix; returns the shared results-folder slug."""
    if collection.endswith(CASCADING_SUFFIX):
        return collection[: -len(CASCADING_SUFFIX)]
    return collection


def results_dir(collection: str, root: Path) -> Path:
    """Per-experiment results folder: results/<base_slug>/."""
    return root / "results" / base_slug(collection)
