# scripts/utils/geojson_utils.py
# -*- coding: utf-8 -*-
"""
Lightweight GeoJSON helpers used across ETL scripts.
Design goals:
- Minimal deps (stdlib + pandas)
- Read/flatten FeatureCollections
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import pandas as pd


def load_geojson(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_features(gj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    feats = gj.get("features") or []
    for feat in feats:
        # guard minimal structure
        if not isinstance(feat, dict):
            continue
        yield feat


def point_coordinates(feature: Dict[str, Any]) -> Optional[tuple[float, float]]:
    """Extract (lon, lat) for Point geometry.
    Returns None if not a Point or invalid coordinates.
    """
    geom = feature.get("geometry") or {}
    if not isinstance(geom, dict):
        return None
    if geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if (
        isinstance(coords, (list, tuple))
        and len(coords) == 2
        and all(isinstance(v, (int, float)) for v in coords)
    ):
        lon, lat = float(coords[0]), float(coords[1])
        return lon, lat
    return None


def flatten_point_features(
    gj: Dict[str, Any],
    *,
    lon_key: str = "経度",
    lat_key: str = "緯度",
    include_source: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Merge properties and point coordinates into flat dicts.

    - Copies all `properties` keys as-is
    - Adds `{lat_key, lon_key}` if geometry is a Point
    - Adds `source_file` if `include_source` is provided
    """
    out: List[Dict[str, Any]] = []
    for feat in iter_features(gj):
        props = feat.get("properties") or {}
        rec: Dict[str, Any] = dict(props) if isinstance(props, dict) else {}
        pt = point_coordinates(feat)
        if pt is not None:
            lon, lat = pt
            rec[lat_key] = lat
            rec[lon_key] = lon
        if include_source is not None:
            rec["source_file"] = str(include_source)
        out.append(rec)
    return out


def to_dataframe(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convenience DataFrame builder from records iterator/list."""
    return pd.DataFrame(list(records))

