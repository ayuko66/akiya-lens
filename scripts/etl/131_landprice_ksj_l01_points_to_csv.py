#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
131_landprice_ksj_l01_points_to_csv.py (HARDENED)
- Robust for KSJ L01 (2018/2023) across namespace differences
- Resolves xlink:href="#ptX" -> gml:Point[@gml:id="ptX"]/gml:pos
- Also supports inline app:position/gml:Point/gml:pos
- Year fallback from filename: L01-18_*.xml -> 2018, L01-23_*.xml -> 2023
- Region filter by 市区町村コード 先頭2桁
"""

import argparse, re, glob
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

# --- config loader (project's util or fallback) ---
try:
    from config_loader import load_config as _load_yaml
except Exception:
    import yaml

    def _load_yaml(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def localname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def get_attr_any_ns(elem, name_tail: str):
    for k, v in elem.attrib.items():
        if k.endswith("}" + name_tail) or k == name_tail:
            return v
    return None


def build_point_map(root: ET.Element):
    """
    Build map: gml:Point @gml:id -> 'lat lon' from gml:pos
    Works regardless of the exact GML namespace (3.2 vs 3.2.1, etc.).
    """
    pt_map = {}
    for el in root.iter():
        if localname(el.tag) == "Point":
            pid = get_attr_any_ns(el, "id")
            if not pid:
                continue
            pos_text = None
            for child in el.iter():
                if localname(child.tag) == "pos" and child.text and child.text.strip():
                    pos_text = child.text.strip()
                    break
            if pos_text:
                pt_map[pid] = pos_text
    return pt_map


def infer_year_from_filename(path: Path):
    m = re.search(r"L01-(\d{2})_", path.name)
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy  # 18->2018, 23->2023


def parse_landprice_file(xml_path: Path, pref_ok: set[str] | None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pt_map = build_point_map(root)
    recs = []

    for e in root.iter():
        if localname(e.tag) != "LandPrice":
            continue

        # 利用現況
        cur = None
        for child in e:
            if localname(child.tag) == "currentUse":
                cur = (child.text or "").strip()
                break
        if cur != "住宅":
            continue

        # 年
        y = None
        # search for gml:timePosition int
        for child in e.iter():
            if localname(child.tag) == "timePosition":
                txt = (child.text or "").strip()
                if txt.isdigit():
                    y = int(txt)
                    break
        if y is None:
            # try direct year text
            for child in e:
                if localname(child.tag) == "year":
                    txt = (child.text or "").strip()
                    if txt.isdigit():
                        y = int(txt)
                        break
        if y is None:
            y = infer_year_from_filename(xml_path)

        # 市区町村コード・名
        code = name = None
        for child in e:
            ln = localname(child.tag)
            if ln == "administrativeAreaCode":
                code = (child.text or "").strip()
            elif ln == "cityName":
                name = (child.text or "").strip()

        # 県フィルタ
        if pref_ok and (not code or code[:2] not in pref_ok):
            continue

        # 価格
        price = None
        for child in e:
            if localname(child.tag) == "postedLandPrice":
                txt = (child.text or "").strip().replace(",", "")
                if re.fullmatch(r"\d+", txt):
                    price = int(txt)
                break
        if y is None or not code or price is None:
            continue

        # 座標
        lat = lon = None
        # inline pos
        pos_text = None
        for sub in e.iter():
            if localname(sub.tag) == "pos" and sub.text and sub.text.strip():
                pos_text = sub.text.strip()
                break
        # xlink href
        if not pos_text:
            for sub in e:
                if localname(sub.tag) == "position":
                    href = get_attr_any_ns(sub, "href")
                    if href and href.startswith("#"):
                        ref = href[1:]
                        pos_text = pt_map.get(ref)
                    break
        if pos_text and " " in pos_text:
            try:
                lat_s, lon_s = pos_text.split()
                lat, lon = float(lat_s), float(lon_s)
            except Exception:
                pass

        recs.append(
            {
                "year": int(y),
                "市区町村コード": str(code).zfill(5),
                "市区町村名": name,
                "価格_円m2": price,
                "緯度": lat,
                "経度": lon,
                "利用現況": "住宅",
                "source_file": str(xml_path),
            }
        )
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--xml_glob",
        nargs="+",
        required=True,
        help='e.g. "data/raw/landprice/2018/L01-*.xml" "data/raw/landprice/2023/L01-*.xml"',
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
    for pat in args.xml_glob:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    all_recs = []
    for f in files:
        try:
            all_recs.extend(
                parse_landprice_file(Path(f), pref_ok=pref_ok if pref_ok else None)
            )
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
