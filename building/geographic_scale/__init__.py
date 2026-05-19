"""Geographic-scale package: slug helper for place-based scaling experiments.

The on-disk dataset / models / results folders are slug-named so the
(place, n_targets) context is visible at the filesystem level. Matches the
``geographic_task`` slug format with a ``scale`` prefix:
``scale_s{n_targets}_{round(lat)}_{round(lon)}_r{round(radius_km)}``.
"""

from __future__ import annotations

from building.geographic_task import task_slug as _task_slug


def scale_slug(
    lat: float, lon: float, radius_km: float, n_targets: int = 10
) -> str:
    return _task_slug([None] * n_targets, lat, lon, radius_km, prefix="scale")
