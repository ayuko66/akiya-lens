import time
import requests
from osm2geojson import json2geojson
from requests import Session

# TODO: configまたはenvで管理
UA = "akiya-lens/0.1 (contact: ayuko.iwata@gmail.com)"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def get_session(user_agent: str = UA) -> Session:
    """Create a requests.Session with a common User-Agent."""
    s = Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def overpass(query: str, session: Session, sleep_sec: float = 1.0) -> dict:
    """Overpass APIに問い合わせ、JSONを返す"""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    res = session.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=180)
    time.sleep(sleep_sec)  # polite wait
    res.raise_for_status()
    return res.json()


# 市区町村コードに対応するWikidata QIDを取得する
def wikidata_qid_from_code(code: str, session: Session) -> str | None:
    """
    市区町村コード(5桁) or 6桁(チェックディジット付) → QID
    Wikidata側は P429 (dantai code) を使用
    """
    code = code.strip()
    # 6桁で来たら完全一致、5桁で来たら前方一致
    if len(code) >= 6:
        filter_clause = f'FILTER(STR(?code_value) = "{code}")'
    else:
        filter_clause = f'FILTER(STRSTARTS(STR(?code_value), "{code}"))'

    query = f"""
    SELECT ?item ?code_value WHERE {{
      ?item wdt:P429 ?code_value .
      {filter_clause}
    }} LIMIT 1
    """
    headers = {"Accept": "application/sparql-results+json"}
    res = session.get(
        WIKIDATA_SPARQL, params={"query": query}, headers=headers, timeout=60
    )
    res.raise_for_status()
    data = res.json()
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    return bindings[0]["item"]["value"].rsplit("/", 1)[-1]


# WikidataのQIDを元に、OpenStreetMapから対応する行政境界のデータを取得し、GeoJSON形式で返却
def fetch_boundary_geojson(qid: str, session: Session) -> dict:
    """行政界boundaryをGeoJSONで返す"""
    q = f"""
    [out:json][timeout:180];
    rel["wikidata"="{qid}"]["boundary"="administrative"];
    out body;
    >;
    out skel qt;
    """
    return json2geojson(overpass(q, session))
