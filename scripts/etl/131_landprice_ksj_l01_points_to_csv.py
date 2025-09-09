#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
131_landprice_ksj_l01_points_to_csv.py (GeoJSON)
- Reads KSJ L01 GeoJSON points (2018/2023)
- Normalizes schema differences and writes points CSV
- Region filter by 市区町村コード 先頭2桁
"""

import argparse, glob
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is on sys.path when run as a file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- config loader (project's util) ---
from scripts.utils.config_loader import load_yaml as _load_yaml
from scripts.utils.geojson_utils import load_geojson
from scripts.utils.landprice_utils import parse_l01_feature

def parse_geojson_file(gj_path: Path, pref_ok: set[str] | None):
    gj = load_geojson(gj_path)
    recs = []
    feats = gj.get("features") or []
    for feat in feats:
        rec = parse_l01_feature(feat)
        if not rec:
            continue
        # 県フィルタ
        code = rec.get("市区町村コード")
        if pref_ok and (not code or code[:2] not in pref_ok):
            continue
        rec["source_file"] = str(gj_path)
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--geojson_glob",
        nargs="+",
        required=True,
        help='e.g. "data/raw/landprice/2018/L01-*.geojson" "data/raw/landprice/2023/L01-*.geojson"',
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--region", default="all")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    enc_out = cfg.get("io", {}).get("encoding_out", "utf-8-sig")

    # prefecture filter
    region_cfg = cfg.get("study_regions", {}).get(args.region, {})
    pref_ok = set(region_cfg.get("prefecture_codes", []) or [])

    # collect files
    files = []
    for pat in args.geojson_glob:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    all_recs = []
    for f in files:
        try:
            all_recs.extend(parse_geojson_file(Path(f), pref_ok=pref_ok if pref_ok else None))
        except Exception as e:
            print(f"[WARN] parse failed: {f}: {e}")

    df = pd.DataFrame(
        all_recs,
        columns=[
            "year",
            "市区町村コード",
            "市区町村名",
            "価格_円m2",
            "緯度",
            "経度",
            "利用現況",
            "source_file",
        ],
    )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding=enc_out)

    # summary
    year_counts = (
        df["year"].value_counts(dropna=False).to_dict() if not df.empty else {}
    )
    lat_nonnull = int(df["緯度"].notnull().sum()) if not df.empty else 0
    lon_nonnull = int(df["経度"].notnull().sum()) if not df.empty else 0
    print(
        f"[OK] Stage1 wrote: {out} rows={len(df)} years={year_counts} lat_nonnull={lat_nonnull} lon_nonnull={lon_nonnull}"
    )


if __name__ == "__main__":
    main()
