#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市区町村マスタCSV（5桁コード）を基に、Wikidata(QID)→OSM行政界を取得し、
1自治体=1フィーチャの GeoJSON を生成するリファレンス用スクリプト。

出力: data/geojson/municipalities.geojson（既定）

特長:
- config/ref_project.yaml の columns_map.city_master で列エイリアスを吸収
- config/etl_project.yaml の study_regions から都道府県コードで絞り込み可能
- ローカルキャッシュ（data/cache）で QID, 境界GeoJSON を再利用
- ネットワーク禁止モード（--no-network）ではキャッシュのみ利用
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path when run as a file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.osm.api import (
    get_session,
    wikidata_qid_from_code,
    fetch_boundary_geojson,
)
from scripts.utils.config_loader import load_yaml


def _load_geopandas():
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import shape  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "geopandas と shapely が必要です。pip install geopandas shapely を実行してください。"
        ) from e
    return gpd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _pick_first(s: pd.Series, aliases: List[str]) -> Optional[str]:
    for a in aliases:
        if a in s and pd.notna(s[a]) and str(s[a]).strip().lower() != "nan":
            return str(s[a]).strip()
    return None


def read_city_master(master_csv: Path, ref_cfg: dict, region_pref_codes: Optional[List[str]]) -> pd.DataFrame:
    col_map = (ref_cfg.get("columns_map") or {}).get("city_master") or {}
    df = pd.read_csv(master_csv)
    # 別名吸収
    alias_to_std: Dict[str, str] = {}
    for std, aliases in col_map.items():
        alias_to_std[std] = std
        for a in aliases or []:
            alias_to_std[a] = std
    df = df.rename(columns={c: alias_to_std.get(c, c) for c in df.columns})

    # 必須列
    for col in ["市区町村コード", "市区町村名", "都道府県コード", "都道府県名"]:
        if col not in df.columns:
            df[col] = ""

    # 型とゼロ埋め
    df["市区町村コード"] = df["市区町村コード"].astype(str).str.strip().str.zfill(5)
    df["都道府県コード"] = df["都道府県コード"].astype(str).str.strip().str.zfill(2)

    # 地域フィルタ
    if region_pref_codes:
        wanted = set([str(c).zfill(2) for c in region_pref_codes])
        df = df[df["都道府県コード"].isin(wanted)].copy()

    # 政令市の親コードなど、名称が欠損/"nan" の行は除外（区レベルを採用）
    df = df[~(df["市区町村名"].isna() | (df["市区町村名"].astype(str).str.lower() == "nan") | (df["市区町村名"].astype(str).str.strip() == ""))].copy()

    # 重複は後勝ちで1コード1行
    df = df.drop_duplicates(subset=["市区町村コード"], keep="last").reset_index(drop=True)
    return df[["市区町村コード", "市区町村名", "都道府県コード", "都道府県名"]]


def build_geojson(
    df: pd.DataFrame,
    *,
    cache_dir: Path,
    use_cache: bool,
    allow_network: bool,
    retries: int,
) -> "gpd.GeoDataFrame":
    gpd = _load_geopandas()

    session = get_session()
    boundary_dir = cache_dir / "boundary_geojson"
    _ensure_dir(boundary_dir)
    qid_cache_path = cache_dir / "qid_by_code.json"
    qid_cache: Dict[str, str] = _read_json(qid_cache_path) or {}

    recs: List[dict] = []

    for idx, row in df.iterrows():
        code = str(row["市区町村コード"]).zfill(5)
        name = str(row["市区町村名"]).strip()
        pref_code = str(row["都道府県コード"]).zfill(2)
        pref_name = str(row["都道府県名"]).strip()

        qid = qid_cache.get(code)
        if not qid and allow_network:
            last_err: Optional[Exception] = None
            for attempt in range(1, retries + 1):
                try:
                    qid = wikidata_qid_from_code(code, session)
                    if qid:
                        break
                except Exception as e:
                    last_err = e
                    time.sleep(0.5 * attempt)
            if not qid:
                print(f"⚠ QID解決失敗: code={code} name={name} err={last_err}", file=sys.stderr)
                continue
            # キャッシュ更新
            qid_cache[code] = qid
            if use_cache:
                _write_json(qid_cache_path, qid_cache)
        elif not qid and not allow_network:
            print(f"⚠ QIDキャッシュなしのためスキップ（no-network）: code={code} name={name}", file=sys.stderr)
            continue

        # 境界GeoJSONを取得
        gj_path = boundary_dir / f"{qid}.geojson"
        gj: Optional[dict] = None
        if use_cache and gj_path.exists():
            gj = _read_json(gj_path)

        if gj is None:
            if not allow_network:
                print(
                    f"⚠ 境界キャッシュなし（no-network）: skip code={code} qid={qid}",
                    file=sys.stderr,
                )
                continue
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    gj = fetch_boundary_geojson(qid, session)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.7 * attempt)
            if gj is None:
                print(f"⚠ 境界取得失敗: code={code} qid={qid} err={last_err}", file=sys.stderr)
                continue
            if use_cache:
                try:
                    with open(gj_path, "w", encoding="utf-8") as f:
                        json.dump(gj, f, ensure_ascii=False)
                except Exception:
                    pass

        # 1自治体=1フィーチャにまとめる（unary_union）
        try:
            # from_features が geometry の無いケースを返した場合に備えて安全に処理
            tmp = gpd.GeoDataFrame.from_features(gj.get("features", []))
            if "geometry" not in tmp.columns:
                raise ValueError("GeoJSON has no geometry column")
            tmp.set_crs(epsg=4326, inplace=True)
            tmp = tmp[~tmp.geometry.is_empty & tmp.geometry.notnull()].copy()
            # union_all 推奨（無ければunary_union）
            if hasattr(tmp.geometry, "union_all"):
                unified = tmp.geometry.union_all()
            else:
                unified = tmp.unary_union
        except Exception as e:
            print(f"⚠ 形状統合失敗: code={code} qid={qid} err={e}", file=sys.stderr)
            continue

        recs.append(
            {
                "市区町村コード": code,
                "市区町村名": name,
                "都道府県コード": pref_code,
                "都道府県名": pref_name,
                "wikidata": qid,
                "geometry": unified,
            }
        )

    if not recs:
        raise SystemExit("出力0件。入力やキャッシュ/ネットワーク設定を確認してください。")

    out_gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
    # コードで安定ソート
    out_gdf = out_gdf.sort_values(["都道府県コード", "市区町村コード"]).reset_index(drop=True)
    return out_gdf


def main() -> None:
    ap = argparse.ArgumentParser(description="市区町村境界GeoJSONを生成（Wikidata/OSM）")
    ap.add_argument("--config", default="config/etl_project.yaml", help="ETL設定YAML（study_regions, joinを参照）")
    ap.add_argument("--ref-config", default="config/ref_project.yaml", help="参照設定YAML（columns_map.city_master を参照）")
    ap.add_argument("--in", dest="in_csv", default="", help="市区町村マスタCSV（未指定なら config.join.city_master_path を参照）")
    ap.add_argument("--region", default="all", help="対象地域キー（config.study_regions のキー）")
    ap.add_argument("--out", dest="out_geojson", default="data/geojson/municipalities.geojson", help="出力GeoJSON パス")
    ap.add_argument("--cache-dir", default="data/cache", help="ローカルキャッシュディレクトリ（デフォルト: data/cache）")
    ap.add_argument("--no-cache", action="store_true", help="キャッシュを使わない")
    ap.add_argument("--no-network", action="store_true", help="ネットワークへアクセスしない（キャッシュのみ使用）")
    ap.add_argument("--retries", type=int, default=3, help="Wikidata/Overpass リトライ回数")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ref_cfg = load_yaml(args.ref_config)

    in_csv = args.in_csv
    if not in_csv:
        in_csv = (cfg.get("join") or {}).get("city_master_path", "data/master/city_master__all__v1__preview.csv")
    master_csv = Path(in_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"市区町村マスタCSVが見つかりません: {master_csv}")

    region_pref_codes = (cfg.get("study_regions") or {}).get(args.region, {}).get("prefecture_codes")

    df = read_city_master(master_csv, ref_cfg, region_pref_codes)

    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    allow_network = not args.no_network

    gdf = build_geojson(
        df,
        cache_dir=cache_dir,
        use_cache=use_cache,
        allow_network=allow_network,
        retries=args.retries,
    )

    out_path = Path(args.out_geojson)
    _ensure_dir(out_path.parent)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"✅ 出力: {out_path}  行数={len(gdf)}")


if __name__ == "__main__":
    main()
