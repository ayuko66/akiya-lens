#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
市区町村ごとに OSM の駅/スーパー/学校/病院(＋任意でクリニック) 件数を集計し、
面積(km^2)で割った密度をCSV出力するCLI。

前提:
- scripts/osm/api.py : overpass(), wikidata_qid_from_code(), fetch_boundary_geojson()
- scripts/osm/utils.py: geodesic_area_km2(), build_poi_query()

使い方例:
  python scripts/osm/fetch_poi_density.py \
    --in data/master/city_master__all__v1__preview.csv \
    --out data/processed/osm_density_nagano_yamanashi_gifu_shizuoka.csv \
    --include-public-transport-stations \
    --include-clinics
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path

# 同一プロジェクト内import（パスはプロジェクト構成に合わせて調整）
from api import (
    get_session,
    overpass,
    wikidata_qid_from_code,
    fetch_boundary_geojson,
)
from utils import geodesic_area_km2, build_poi_query


try:
    # プロジェクト共通のユーティリティを優先
    from scripts.utils.config_loader import load_yaml
except ImportError:
    # 見つからない場合のフォールバック
    import yaml

    def load_yaml(path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _get_col_value(row: dict, aliases: list[str]) -> str | None:
    for alias in aliases:
        v = row.get(alias)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":  # 文字列 "nan" を弾く
            return s
    return None


def read_municipalities(
    path: str, target_pref_codes: set[str], col_map: dict
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    # 標準名とエイリアスを結合したリストを作成
    code_aliases = ["市区町村コード"] + col_map.get("市区町村コード", [])
    pref_name_aliases = ["都道府県名"] + col_map.get("都道府県名", [])
    city_name_aliases = ["市区町村名"] + col_map.get("市区町村名", [])
    pref_code_aliases = ["都道府県コード"] + col_map.get("都道府県コード", [])

    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            code = _get_col_value(r, code_aliases)
            pref_name = _get_col_value(r, pref_name_aliases)
            city_name = _get_col_value(r, city_name_aliases)
            # city_nameがnanの場合、「政令市･郡･支庁･振興局等」列の値に置き換える
            if city_name is None or city_name.lower() == "nan":
                city_name = _get_col_value(r, ["政令市･郡･支庁･振興局等"])
            pref_code = _get_col_value(r, pref_code_aliases)

            # 都道府県コードでフィルタリング
            # pref_codeがなければ市区町村コードから導出
            current_pref_code = pref_code if pref_code else code[:2]
            if target_pref_codes and current_pref_code not in target_pref_codes:
                continue

            # 必須列がなければスキップ
            if not (code and pref_name and city_name):
                print(
                    f"  [WARN] 必須列が見つからないためスキップ: {r} code: {code}, pref_name: {pref_name}, city_name: {city_name}",
                    file=sys.stderr,
                )
                continue

            rows.append(
                {
                    "code": code,
                    "prefecture": pref_name,
                    "name": city_name,
                    # 任意列
                    "wikidata": r.get("wikidata", "").strip(),
                }
            )
    return rows


def count_unique_ids(elements: List[dict]) -> int:
    """
    Overpass JSONのelementsから type+id のユニーク件数を返す
    （node/way/relationの混在を重複なく数える）
    """
    seen: Set[Tuple[str, int]] = set()
    for el in elements:
        t = el.get("type")
        i = el.get("id")
        if t is not None and i is not None:
            seen.add((t, int(i)))
    return len(seen)


# 施設(駅、スーパーなど)の数を数える
def count_pois(
    qid: str,
    session,
    tag_exprs: Iterable[str],
    sleep_sec: float = 0.8,
    retries: int = 3,
) -> int:
    """
    指定タグ式の合算件数（ユニークIDベース）
    tag_exprs 例:
      '["railway"="station"]'
      '["public_transport"="station"]'
      '["shop"="supermarket"]'
      '["amenity"="school"]'
      '["amenity"="hospital"]'
      '["amenity"="clinic"]'
    """
    all_elems: List[dict] = []
    for tag in tag_exprs:
        # overpass APIにわたすクエリを生成
        q = build_poi_query(qid, tag)
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                data = overpass(q, session, sleep_sec=sleep_sec)
                all_elems.extend(data.get("elements", []))
                break
            except Exception as e:  # requests.HTTPError など
                last_err = e
                # polite backoff
                time.sleep(sleep_sec * attempt)
        else:
            raise RuntimeError(f"Overpass失敗: qid={qid}, tag={tag}, err={last_err}")
    return count_unique_ids(all_elems)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_csv",
        required=True,
        help="入力CSV (市区町村コード, 都道府県名, ...)",
    )
    ap.add_argument("--out", dest="out_csv", required=True, help="出力CSVパス")
    ap.add_argument("--config", required=True, help="設定YAMLファイルへのパス")
    ap.add_argument(
        "--region",
        default="yatsugatake_alps",
        help="対象地域 (configファイルの study_regions のキー)",
    )
    ap.add_argument(
        "--include-public-transport-stations",
        action="store_true",
        help="駅カウントに public_transport=station も含める",
    )
    ap.add_argument(
        "--include-clinics",
        action="store_true",
        help="病院カウントに amenity=clinic も合算する",
    )
    ap.add_argument(
        "--sleep-sec",
        type=float,
        default=1.0,
        help="Overpassアクセス間の待機秒（礼儀）",
    )
    ap.add_argument(
        "--retries", type=int, default=3, help="Overpass/Wikidata のリトライ回数"
    )
    args = ap.parse_args()

    # 設定ファイル読み込み
    cfg = load_yaml(args.config)

    # 対象地域の都道府県コードを取得
    region_cfg = cfg.get("study_regions", {}).get(args.region, {})
    target_pref_codes = set(region_cfg.get("prefecture_codes", []))
    if not target_pref_codes:
        print(f"対象地域: {args.region} (全域)")
    else:
        print(
            f"対象地域: {args.region} (都道府県コード: {', '.join(sorted(target_pref_codes))})"
        )

    # 参照設定ファイル(ref_project.yaml)から市区町村マスタの列名定義を取得
    config_path = Path(args.config)
    ref_config_path = config_path.parent / "ref_project.yaml"
    if not ref_config_path.exists():
        raise FileNotFoundError(f"参照設定ファイルが見つかりません: {ref_config_path}")
    ref_cfg = load_yaml(ref_config_path)

    col_map = ref_cfg.get("columns_map", {}).get("city_master", {})
    if not col_map:
        raise ValueError(
            f"設定ファイルに columns_map.city_master が見つかりません: {ref_config_path}"
        )

    rows = read_municipalities(args.in_csv, target_pref_codes, col_map)
    out_rows: List[Dict[str, object]] = []

    session = get_session()

    for idx, m in enumerate(rows, start=1):
        code = m["code"]
        name = m["name"]
        pref = m["prefecture"]
        qid = m.get("wikidata") or ""

        print(f"[{idx}/{len(rows)}] {pref} {name} ({code})")

        # 1) QID解決（CSVにwikidata列があればそれを優先）
        if not qid:
            qid = None
            last_err: Optional[Exception] = None
            # Wikidata QIDを取得
            for attempt in range(1, args.retries + 1):
                try:
                    qid = wikidata_qid_from_code(code, session)
                    if qid:
                        last_err = None  # 成功したらエラーをクリア
                        break
                except Exception as e:
                    last_err = e
                    print(
                        f"  QID取得試行 {attempt}/{args.retries} でエラー: {repr(e)}",
                        file=sys.stderr,
                    )
                    time.sleep(0.5 * attempt)  # エラー時にのみ待機
            if not qid:
                if last_err:
                    print(
                        f"  QID解決失敗（スキップ）: code={code}, err={last_err}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  QIDが見つかりませんでした（スキップ）: code={code}",
                        file=sys.stderr,
                    )
                continue
        print(f"  QID={qid}")

        # 2) 境界取得 → 面積計算
        gj = None
        last_err = None
        # 地図データを取得し、市区町村の面積を計算
        for attempt in range(1, args.retries + 1):
            try:
                # 地図データ取得
                gj = fetch_boundary_geojson(qid, session)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.7 * attempt)
        if gj is None:
            print(
                f"  ⚠ 境界取得失敗（スキップ）: qid={qid}, err={last_err}",
                file=sys.stderr,
            )
            continue
        # 面積を計算
        area_km2 = geodesic_area_km2(gj)
        if area_km2 <= 0:
            print(f"  ⚠ 面積が0（スキップ）: qid={qid}", file=sys.stderr)
            continue
        print(f"  面積 = {area_km2:.3f} km²")

        # 3) POIカウント
        # 駅
        station_tags = ['["railway"="station"]']
        if args.include_public_transport_stations:
            station_tags.append('["public_transport"="station"]')

        # スーパー・学校・医療
        supermarket_tags = ['["shop"="supermarket"]']
        school_tags = ['["amenity"="school"]']
        hospital_tags = ['["amenity"="hospital"]']
        clinic_tags = ['["amenity"="clinic"]'] if args.include_clinics else []

        print("  POIカウント中…")
        stations = count_pois(
            qid, session, station_tags, sleep_sec=args.sleep_sec, retries=args.retries
        )
        supermarkets = count_pois(
            qid,
            session,
            supermarket_tags,
            sleep_sec=args.sleep_sec,
            retries=args.retries,
        )
        schools = count_pois(
            qid, session, school_tags, sleep_sec=args.sleep_sec, retries=args.retries
        )
        hospitals = count_pois(
            qid,
            session,
            hospital_tags + clinic_tags,
            sleep_sec=args.sleep_sec,
            retries=args.retries,
        )

        # 4) 密度計算
        dens = {
            "stations_density_per_km2": stations / area_km2,
            "supermarkets_density_per_km2": supermarkets / area_km2,
            "schools_density_per_km2": schools / area_km2,
            "hospitals_density_per_km2": hospitals / area_km2,
        }

        out = {
            "code": code,
            "prefecture": pref,
            "name": name,
            "wikidata": qid,
            "area_km2": round(area_km2, 6),
            "stations": stations,
            "stations_density_per_km2": round(dens["stations_density_per_km2"], 6),
            "supermarkets": supermarkets,
            "supermarkets_density_per_km2": round(
                dens["supermarkets_density_per_km2"], 6
            ),
            "schools": schools,
            "schools_density_per_km2": round(dens["schools_density_per_km2"], 6),
            "hospitals": hospitals,
            "hospitals_density_per_km2": round(dens["hospitals_density_per_km2"], 6),
        }
        out_rows.append(out)

        # Overpassへの配慮
        time.sleep(max(0.2, args.sleep_sec))

    # 5) CSV 出力
    if not out_rows:
        print(
            "⚠ 出力0行。入力やネットワーク、タグを確認してください。", file=sys.stderr
        )
        sys.exit(2)

    fieldnames = list(out_rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"\n✅ Done: {args.out_csv}  ({len(out_rows)} rows)")
    sys.exit(0)


if __name__ == "__main__":
    main()
