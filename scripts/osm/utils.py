# OpenStreetMap (OSM) のデータを扱うためのutil
from pyproj import Geod

geod = Geod(ellps="WGS84")


def build_poi_query(qid: str, tag_expr: str) -> str:
    """特定の行政区画内にある施設（POI: Point of Interest）を検索するためのOverpass APIクエリ生成
    Parameters
    ----------
    qid: str
        行政区画のWikidata ID ("Q1766" など)
    tag_expr: str
        探したい施設を表すOSMのタグ式。例: '["amenity"="school"]'

    Notes
    -----
    - ``tag_expr`` には Overpass API のタグ指定式全体を渡す。
      ``build_poi_query`` 内では括弧を追加しないため、呼び出し側で
      先頭と末尾の ``[]`` を含めた形で指定する。

    処理の流れ:
      1. ``qid`` を使って行政区画のエリアを特定
      2. そのエリア内にある ``tag_expr`` を満たす OSM の要素
         (node/way/relation) を取得
      3. 詳細データではなく ID のみを返す ``out ids qt;`` を指定
    """

    return f"""
    [out:json][timeout:180];
    rel["wikidata"="{qid}"]["boundary"="administrative"]->.rel;
    area.rel->.a;
    nwr{tag_expr}(area.a);
    out ids qt;
    """


def geodesic_area_km2(geojson_obj: dict) -> float:
    """GeoJSON形式で与えられたポリゴンの面積を計算
    geojson_obj: GeoJSONオブジェクト
    """

    total_m2 = 0.0
    for feat in geojson_obj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            # 図形がポリゴンの場合の面積を計算
            total_m2 += _polygon_area(geom["coordinates"])
            # 図形が多角形の場合の面積を計算
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                total_m2 += _polygon_area(poly)
    # total_m2をk㎡に変換して返却
    return total_m2 / 1_000_000.0


def _polygon_area(coords):
    """ポリゴンの面積を計算"""
    outer = coords[0]  # 頂点座標(緯度経度リスト)を受け取る
    lons, lats = zip(*outer)
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area)
