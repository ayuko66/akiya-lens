# scripts/etl/132_landprice_ksj_l01_median_by_city.py
# -*- coding: utf-8 -*-
"""
Stage2 (fixed): point-level CSV -> city-year median aggregation
- Fix: ensure year is int (not 2023.0) before pivot
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
import pandas as pd

try:
    from config_loader import load_config
except Exception:
    import yaml

    def load_config(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_long_csv", required=True)
    ap.add_argument("--out_wide_csv", required=True)
    ap.add_argument("--base_year", type=int, default=2018)
    ap.add_argument("--target_year", type=int, default=2023)
    args = ap.parse_args()

    cfg = load_config(args.config)
    enc = cfg.get("io", {}).get("encoding_out", "utf-8-sig")

    df = pd.read_csv(args.in_csv, dtype={"市区町村コード": "string"}, encoding=enc)

    # yearをint化（NaNは除外）
    df = df[pd.notnull(df["year"])].copy()
    df["year"] = df["year"].astype(int)

    g = df.groupby(["year", "市区町村コード"], as_index=False).agg(
        市区町村名=(
            "市区町村名",
            lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0],
        ),
        標準地点数=("価格_円m2", "count"),
        住宅地価_中央値=("価格_円m2", "median"),
    )
    g["住宅地価_log中央値"] = g["住宅地価_中央値"].apply(
        lambda x: math.log(x) if pd.notnull(x) and x > 0 else None
    )
    g = g.sort_values(["year", "市区町村コード"])

    out_long = Path(args.out_long_csv)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_long, index=False, encoding=enc)

    w = g.pivot_table(
        index=["市区町村コード", "市区町村名"],
        columns="year",
        values=["住宅地価_中央値", "住宅地価_log中央値", "標準地点数"],
        aggfunc="first",
    )
    w.columns = [f"{c}_{y}" for c, y in w.columns]
    w = w.reset_index()

    # 2018/2023があれば差分作成
    if all(
        c in w.columns
        for c in [
            f"住宅地価_log中央値_{args.base_year}",
            f"住宅地価_log中央値_{args.target_year}",
        ]
    ):
        w["住宅地価_log差分"] = (
            w[f"住宅地価_log中央値_{args.target_year}"]
            - w[f"住宅地価_log中央値_{args.base_year}"]
        )
    else:
        w["住宅地価_log差分"] = None

    if all(
        c in w.columns
        for c in [
            f"住宅地価_中央値_{args.base_year}",
            f"住宅地価_中央値_{args.target_year}",
        ]
    ):
        w["住宅地価_増減率[%]"] = (
            w[f"住宅地価_中央値_{args.target_year}"]
            / w[f"住宅地価_中央値_{args.base_year}"]
            - 1.0
        ) * 100.0
    else:
        w["住宅地価_増減率[%]"] = None

    preferred = [
        "市区町村コード",
        "市区町村名",
        f"住宅地価_中央値_{args.base_year}",
        f"住宅地価_中央値_{args.target_year}",
        f"住宅地価_log中央値_{args.base_year}",
        f"住宅地価_log中央値_{args.target_year}",
        f"標準地点数_{args.base_year}",
        f"標準地点数_{args.target_year}",
        "住宅地価_log差分",
        "住宅地価_増減率[%]",
    ]
    for c in preferred:
        if c not in w.columns:
            w[c] = None
    w = w[preferred]

    out_wide = Path(args.out_wide_csv)
    out_wide.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(out_wide, index=False, encoding=enc)

    print(
        f"[OK] Stage2 wrote: {out_long} (long rows={len(g)}), {out_wide} (wide rows={len(w)})"
    )


if __name__ == "__main__":
    main()
