#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
市区町村ごとに OSM の駅/スーパー/学校/病院（＋任意でクリニック）件数を集計し、
面積(km^2)で割った密度をCSV出力するCLI。

前提:
- scripts/osm/api.py : overpass(), wikidata_qid_from_code(), fetch_boundary_geojson()
- scripts/osm/utils.py: geodesic_area_km2(), build_poi_query()

実行例:
- 基本（キャッシュ有効, TTLデフォルト=168時間）
    python scripts/osm/fetch_poi_density.py \
      --in data/master/city_master__all__v1__preview.csv \
      --out data/processed/osm_density.csv \
      --config config/etl_project.yaml

- 駅の `public_transport=station` と クリニックを合算
    python scripts/osm/fetch_poi_density.py \
      --in data/master/city_master__all__v1__preview.csv \
      --out data/processed/osm_density_with_pt_clinic.csv \
      --config config/etl_project.yaml \
      --include-public-transport-stations \
      --include-clinics

- キャッシュディレクトリの指定と POI件数TTLを12時間に短縮
    python scripts/osm/fetch_poi_density.py \
      --in data/master/city_master__all__v1__preview.csv \
      --out data/processed/osm_density.csv \
      --config config/etl_project.yaml \
      --cache-dir ./mycache \
      --poi-cache-ttl-hours 12

- キャッシュを使わずに実行（最新取得）
    python scripts/osm/fetch_poi_density.py \
      --in data/master/city_master__all__v1__preview.csv \
      --out data/processed/osm_density_fresh.csv \
      --config config/etl_project.yaml \
      --no-cache

キャッシュ仕様:
- 目的: ネットワーク往復と重い計算（境界取得＋面積計算、POI件数取得）を再利用して高速化。
- 配置（デフォルト `data/cache`）:
  - code→QID: `qid_by_code.json`
  - QID→面積(km^2): `area_km2_by_qid.json`
  - 行政境界GeoJSON: `boundary_geojson/{QID}.geojson`
  - POI件数: `poi_counts.json`
- POI件数キャッシュ:
  - キー: 「QID | カテゴリ名 | タグ集合（空白除去＋ソートして連結）」
  - 値: `{ "count": <int>, "ts": <Epoch秒> }`
  - TTL: `--poi-cache-ttl-hours`（既定 168h）。`0`以下で無効＝常に再取得。
  - 一部カテゴリのみ未キャッシュ/期限切れの場合は、その分だけをOverpassへバッチ問い合わせ。
- 無効化オプション:
  - 全キャッシュ無効: `--no-cache`
  - POI件数のみ無効: `--no-poi-cache`
- 注意事項:
  - OSM/Wikidataの更新はTTL内には反映されません。最新化したい場合はTTLを短くする/`--no-cache`で再取得/`data/cache`を削除してください。
  - Overpassへの負荷を避けるため、クエリはカテゴリをまとめてバッチ実行し、礼儀的な待機を入れています。
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path
import sys

# Ensure project root is on sys.path when run as a file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 同一プロジェクト内import（パスはプロジェクト構成に合わせて調整）
from scripts.osm.api import (
    get_session,
    overpass,
    wikidata_qid_from_code,
    fetch_boundary_geojson,
)
from scripts.osm.utils import geodesic_area_km2, build_poi_query

from scripts.utils.config_loader import load_yaml


def _get_col_value(row: dict, aliases: list[str]) -> str | None:
    for alias in aliases:
        v = row.get(alias)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":  # 文字列 "nan" を弾く
            return s
    return None


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


def count_pois_batched(
    qid: str,
    session,
    categories: Dict[str, Iterable[str]],
    sleep_sec: float = 0.8,
    retries: int = 3,
) -> Dict[str, int]:
    """
    複数カテゴリ（キー: カテゴリ名 → 値: タグ式の配列）をまとめて1回のOverpassで件数取得。

    - 各カテゴリはサーバ側で集合和を取り（重複を排除）、`out count;` で総数を返す。
    - `categories` は順序を保持する（Python 3.7+ の dict は挿入順保持）。
    - 返り値は {カテゴリ名: 件数}。
    """
    # クエリ構築
    blocks: List[str] = [
        "[out:json][timeout:180];",
        f'rel["wikidata"="{qid}"]["boundary"="administrative"]->.r;',
        "(.r;);",
        "map_to_area -> .a;",
    ]

    # カテゴリごとに集合を作る
    for key, tag_exprs in categories.items():
        blocks.append("(")
        for tag in tag_exprs:
            blocks.append(f"  nwr{tag}(area.a);")
        blocks.append(f")->.{key};")

    # 各カテゴリのカウントを順に出力
    for key in categories.keys():
        blocks.append(f"(.{key};);")
        blocks.append("out count;")

    query = "\n".join(blocks)

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            data = overpass(query, session, sleep_sec=sleep_sec)
            break
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * attempt)
    else:
        raise RuntimeError(f"Overpass失敗(バッチ): qid={qid}, err={last_err}")

    # レスポンスから count を順に取り出し、カテゴリ順に対応付け
    counts: List[int] = []
    for el in data.get("elements", []):
        if el.get("type") != "count":
            continue
        tags = el.get("tags", {}) or {}
        # 代表的な2パターンに対応: total があればそれを使用、なければ nodes+ways+relations を合算
        total = tags.get("total")
        if total is not None:
            try:
                counts.append(int(total))
            except Exception:
                # 念のためフォールバック
                s = 0
                for k in ("nodes", "ways", "relations", "count"):
                    v = tags.get(k)
                    if v is not None:
                        try:
                            s += int(v)
                        except Exception:
                            pass
                counts.append(s)
        else:
            s = 0
            for k in ("nodes", "ways", "relations", "count"):
                v = tags.get(k)
                if v is not None:
                    try:
                        s += int(v)
                    except Exception:
                        pass
            counts.append(s)

    # カテゴリ数に満たない場合は0で埋める
    if len(counts) < len(categories):
        counts.extend([0] * (len(categories) - len(counts)))

    return {name: counts[i] for i, name in enumerate(categories.keys())}


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
    ap.add_argument(
        "--cache-dir",
        default="data/cache",
        help="ローカルキャッシュ格納ディレクトリ（デフォルト: data/cache）",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="ローカルキャッシュを使わない",
    )
    ap.add_argument(
        "--no-poi-cache",
        action="store_true",
        help="POI件数キャッシュを使わない",
    )
    ap.add_argument(
        "--poi-cache-ttl-hours",
        type=float,
        default=168.0,
        help="POI件数キャッシュのTTL（時間）。0以下で無効（常に再取得）",
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

    # ローカルキャッシュ（QID, 面積, POI件数）
    cache_enabled = not args.no_cache
    if cache_enabled:
        cache_dir = Path(args.cache_dir)
        _ensure_dir(cache_dir)
        boundary_dir = cache_dir / "boundary_geojson"
        _ensure_dir(boundary_dir)
        qid_cache_path = cache_dir / "qid_by_code.json"
        area_cache_path = cache_dir / "area_km2_by_qid.json"
        poi_cache_path = cache_dir / "poi_counts.json"
        qid_cache: Dict[str, str] = _read_json(qid_cache_path) or {}
        area_cache: Dict[str, float] = _read_json(area_cache_path) or {}
        poi_cache: Dict[str, dict] = _read_json(poi_cache_path) or {}
    else:
        boundary_dir = None  # type: ignore
        qid_cache_path = None  # type: ignore
        area_cache_path = None  # type: ignore
        poi_cache_path = None  # type: ignore
        qid_cache = {}
        area_cache = {}
        poi_cache = {}

    for idx, m in enumerate(rows, start=1):
        code = m["code"]
        name = m["name"]
        pref = m["prefecture"]
        qid = m.get("wikidata") or ""

        print(f"[{idx}/{len(rows)}] {pref} {name} ({code})")

        # 1) QID解決（CSVにwikidata列があればそれを優先）
        if not qid:
            # キャッシュに存在すれば使用
            if cache_enabled and code in qid_cache:
                qid = qid_cache.get(code) or ""
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
                # 成功時にキャッシュへ保存
                if cache_enabled and qid:
                    qid_cache[code] = qid
                    _write_json(qid_cache_path, qid_cache)
        print(f"  QID={qid}")

        # 2) 境界取得 → 面積計算（キャッシュ利用）
        area_km2: Optional[float] = None
        if cache_enabled and qid in area_cache:
            try:
                area_km2 = float(area_cache[qid])
            except Exception:
                area_km2 = None
        if area_km2 is None:
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
            # キャッシュへ保存
            if cache_enabled and area_km2 > 0:
                area_cache[qid] = area_km2
                _write_json(area_cache_path, area_cache)
                # GeoJSONも任意で保存（存在しない場合のみ）
                try:
                    if boundary_dir is not None:
                        out_gj = boundary_dir / f"{qid}.geojson"
                        if not out_gj.exists():
                            with open(out_gj, "w", encoding="utf-8") as f:
                                json.dump(gj, f, ensure_ascii=False)
                except Exception:
                    pass
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

        print("  POIカウント中…（バッチ処理）")
        cat_tags: Dict[str, Iterable[str]] = {
            "stations": station_tags,
            "supermarkets": supermarket_tags,
            "schools": school_tags,
            "hospitals": hospital_tags + clinic_tags,
        }
        cat_counts = count_pois_batched(
            qid,
            session,
            cat_tags,
            sleep_sec=args.sleep_sec,
            retries=args.retries,
        )
        stations = cat_counts.get("stations", 0)
        supermarkets = cat_counts.get("supermarkets", 0)
        schools = cat_counts.get("schools", 0)
        hospitals = cat_counts.get("hospitals", 0)

        print(
            f"stations: {stations}, supermarkets: {supermarkets}, schools: {schools}, hospitals: {hospitals}"
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

    # 5) CSV 出力（日本語カラム名に揃える）
    if not out_rows:
        print(
            "⚠ 出力0行。入力やネットワーク、タグを確認してください。", file=sys.stderr
        )
        sys.exit(2)

    # 出力カラムのマッピング（左: 内部キー → 右: 出力カラム名）
    header_map = {
        "code": "市区町村コード",
        "prefecture": "都道府県名",
        "name": "市区町村名",
        "wikidata": "Wikidata QID",
        "area_km2": "面積[km²]",
        "stations": "駅件数",
        "stations_density_per_km2": "駅密度[件/km²]",
        "supermarkets": "スーパー件数",
        "supermarkets_density_per_km2": "スーパー密度[件/km²]",
        "schools": "学校件数",
        "schools_density_per_km2": "学校密度[件/km²]",
        "hospitals": "病院件数",
        "hospitals_density_per_km2": "病院密度[件/km²]",
    }

    # 明示的な出力順（デフォルトのカラム順）
    ordered_internal_keys = [
        "code",
        "prefecture",
        "name",
        "wikidata",
        "area_km2",
        "stations",
        "stations_density_per_km2",
        "supermarkets",
        "supermarkets_density_per_km2",
        "schools",
        "schools_density_per_km2",
        "hospitals",
        "hospitals_density_per_km2",
    ]

    # 出力用の日本語ヘッダー
    fieldnames = [header_map[k] for k in ordered_internal_keys]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            # 内部キー → 日本語キーへ変換
            out_ja = {header_map[k]: r.get(k) for k in ordered_internal_keys}
            w.writerow(out_ja)

    print(f"\n✅ Done: {args.out_csv}  ({len(out_rows)} rows)")
    sys.exit(0)


if __name__ == "__main__":
    main()
