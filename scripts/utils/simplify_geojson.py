#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自治体境界 GeoJSON を Web 表示向けに堅牢に簡略化するスクリプト。
- Point/LineString など面でないジオメトリはバッファで擬似ポリゴン化して救済
- 簡略化後に無効/極小/面でない場合は元形状にフォールバック
- 使う属性だけ残して軽量化
"""

from pathlib import Path
import math
import geopandas as gpd

# Shapely 2.x / 1.8 互換の make_valid
try:
    from shapely import make_valid  # Shapely>=2.0
except Exception:
    from shapely.validation import make_valid  # Shapely<2.0

from shapely.ops import unary_union
from shapely.geometry import GeometryCollection

# ========= 設定（必要に応じて調整） =========
SRC = Path("data/geojson/municipalities.geojson")
DST = Path("data/geojson/municipalities_simplified.geojson")

TOLERANCE_M = 600  # 簡略化の許容誤差（メートル; Webメルカトル上）
POINT_BUFFER_M = 350  # Point を面にする半径（メートル）
LINE_BUFFER_M = 80  # LineString を面にする半径（メートル）
MIN_AREA_M2 = 5e4  # 簡略化後に許容する最小面積（㎡; 0.05 km²）
KEEP_COLS = ["市区町村コード", "市区町村名", "都道府県名", "geometry"]
# ==========================================


def _to_surface(geom, point_buf=POINT_BUFFER_M, line_buf=LINE_BUFFER_M):
    """面でないジオメトリを面（Polygon/MultiPolygon）に変換する"""
    if geom is None or geom.is_empty:
        return None

    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return geom

    if gt in ("Point", "MultiPoint"):
        g = unary_union(list(geom.geoms)) if gt == "MultiPoint" else geom
        return g.buffer(point_buf)

    if gt in ("LineString", "MultiLineString"):
        return geom.buffer(line_buf)

    if gt == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)
        # 何も面が無ければ全体を小さくバッファ
        return unary_union(list(geom.geoms)).buffer(point_buf)

    # 想定外は安全側で None
    return None


def _is_polygonish(geom):
    return geom is not None and (geom.geom_type in ("Polygon", "MultiPolygon"))


def _ensure_min_area(geom, min_area=MIN_AREA_M2):
    """面積が閾値より小さければ半径を増やして疑似面を再生成（Point 等）"""
    if geom is None or geom.is_empty:
        return None
    if geom.area >= min_area or geom.geom_type not in ("Polygon", "MultiPolygon"):
        return geom
    # 面積不足時は、同等面積になる円の半径を逆算して少し上乗せ
    radius = math.sqrt(min_area / math.pi)
    return geom.buffer(radius)


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"not found: {SRC}")

    # 読み込み → メルカトル化（メートル単位で簡略化・バッファしたいので）
    gdf = gpd.read_file(SRC)
    gdf = gdf.to_crs(3857)

    # 元ジオメトリ退避 & 妥当化
    gdf["geom_src"] = gdf.geometry.map(
        lambda g: make_valid(g).buffer(0) if g is not None else None
    )

    # まず面に救済（Point/Line 等をポリゴン化）
    gdf["geometry"] = gdf["geom_src"].map(_to_surface)

    # 救済できなかった（None）ものは元に戻す（最後の砦）
    gdf.loc[gdf["geometry"].isna(), "geometry"] = gdf.loc[
        gdf["geometry"].isna(), "geom_src"
    ]

    # 簡略化
    gdf["geom_simplified"] = gdf.geometry.simplify(TOLERANCE_M, preserve_topology=True)

    # 妥当性チェック：面でない/無効/極小は元形状へフォールバック
    def _rescue(row):
        g = row["geom_simplified"]
        if g is None or g.is_empty or not _is_polygonish(g) or g.area < MIN_AREA_M2:
            base = row["geometry"]
            if base is None or base.is_empty:
                return None
            # 面積不足なら補強
            base2 = _ensure_min_area(base, MIN_AREA_M2)
            # 再簡略化（軽めに）
            return base2.simplify(TOLERANCE_M / 2.0, preserve_topology=True)
        return g

    gdf["geometry"] = gdf.apply(_rescue, axis=1)

    # まだ面でない/None を最終除外（※基本的に起きない想定）
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    # WGS84 に戻す & 列を絞る
    gdf = gdf.to_crs(4326)

    # 市区町村コードの0埋め（念のため）
    if "市区町村コード" in gdf.columns:
        gdf["市区町村コード"] = (
            gdf["市区町村コード"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )

    keep = [c for c in KEEP_COLS if c in gdf.columns]
    gdf = gdf[keep]

    # 進捗ログ
    before_counts = gpd.read_file(SRC)
    print("== BEFORE ==")
    print(before_counts.geometry.type.value_counts(dropna=False))
    print("== AFTER  ==")
    print(gdf.geometry.type.value_counts(dropna=False))

    DST.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(DST, driver="GeoJSON")
    print(f"Wrote: {DST}  (from: {SRC})")


if __name__ == "__main__":
    main()
