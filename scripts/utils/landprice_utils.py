# scripts/utils/landprice_utils.py
# -*- coding: utf-8 -*-
"""
Utilities for KSJ L01 (地価公示) feature parsing from GeoJSON.
Handles schema differences between 2018 and 2023 exports by inferring
field positions from presence and types.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _as_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip().replace(",", "")
        return int(s) if s != "" else None
    except Exception:
        return None


def _as_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s != "" else None


def parse_l01_feature(feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a single KSJ L01 Feature to a normalized record.

    Returns dict with keys:
      - year, 市区町村コード, 市区町村名, 価格_円m2, 緯度, 経度, 利用現況
    or None if the record should be skipped.
    """
    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        return None

    # Year is consistently L01_005 as string or int
    year = _as_int(props.get("L01_005"))

    # 利用現況フィールドは年度で異なる:
    # - 2018: L01_025
    # - 2023: L01_027
    # 年が判別できる場合は確実に選択し、判別不能時は両方を安全に評価
    if year is not None and year <= 2019:
        current_use = _as_str(props.get("L01_025"))
    elif year is not None:
        current_use = _as_str(props.get("L01_027"))
    else:
        cu27 = _as_str(props.get("L01_027"))
        cu25 = _as_str(props.get("L01_025"))
        current_use = cu27 if cu27 == "住宅" else cu25
    if current_use != "住宅":
        return None

    # City code/name differ between years
    code_raw = props.get("L01_022")
    name_raw = props.get("L01_023")
    # In 2018, L01_022 is name, L01_021 is code
    if not (isinstance(code_raw, str) and code_raw.isdigit()) and not isinstance(code_raw, int):
        code_raw = props.get("L01_021")
        name_raw = props.get("L01_022")

    code = _as_str(code_raw)
    if code is not None:
        code = code.zfill(5)
    name = _as_str(name_raw)

    price = _as_int(props.get("L01_006"))

    # Geometry: Point [lon, lat]
    geom = feature.get("geometry") or {}
    lat = lon = None
    try:
        if isinstance(geom, dict) and geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                lon = float(coords[0])
                lat = float(coords[1])
    except Exception:
        pass

    rec = {
        "year": year,
        "市区町村コード": code,
        "市区町村名": name,
        "価格_円m2": price,
        "緯度": lat,
        "経度": lon,
        "利用現況": current_use,
    }

    # Basic required fields
    if rec["year"] is None or rec["市区町村コード"] is None or rec["価格_円m2"] is None:
        return None

    return rec
