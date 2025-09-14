import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _load_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:
        raise SystemExit(
            "geopandas is required for this script. Please install geopandas and shapely."
        ) from e
    return gpd


def _read_municipalities(gpd, muni_path: Path, target_crs: str) -> "gpd.GeoDataFrame":
    if not muni_path.exists():
        raise FileNotFoundError(f"Municipality GeoJSON not found: {muni_path}")
    gdf = gpd.read_file(muni_path)

    # Normalize municipality code column
    cand_cols = [
        "市区町村コード",
        "標準地域コード",
        "全国地方公共団体コード",
        "全国地方公共団体番号",
        "muni_code",
        "city_code",
    ]
    code_col = None
    for c in cand_cols:
        if c in gdf.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError(
            "Municipality GeoJSON must contain a municipality code column (e.g., 市区町村コード)."
        )
    if code_col != "市区町村コード":
        gdf = gdf.rename(columns={code_col: "市区町村コード"})

    # Ensure string, 5-digit, zero-padded
    gdf["市区町村コード"] = gdf["市区町村コード"].astype(str).str.zfill(5)

    # Drop non-polygonal geometries if any
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

    if gdf.crs is None:
        logging.warning("Municipality GeoJSON has no CRS. Assuming EPSG:4326 (lon/lat).")
        gdf.set_crs(epsg=4326, inplace=True)

    gdf = gdf.to_crs(target_crs)

    # Precompute municipal area for weights
    gdf["_muni_area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["_muni_area_m2"] > 0].copy()
    return gdf


def _read_climate_mesh(gpd, climate_files: List[Path], target_crs: str) -> "gpd.GeoDataFrame":
    if not climate_files:
        raise FileNotFoundError("No climate GeoJSON files found under data/raw/climate")

    gdfs = []
    for f in climate_files:
        logging.info(f"Reading climate mesh: {f}")
        gdf = gpd.read_file(f)
        if gdf.crs is None:
            # KSJ G02 (jgd) is typically EPSG:4612 (Tokyo datum), which aligns with EPSG:4612 in file
            # Most files embed CRS; if missing, assume EPSG:4612 then convert.
            logging.warning(f"{f} has no CRS; assuming EPSG:4612")
            gdf.set_crs(epsg=4612, inplace=True)
        gdfs.append(gdf)
    mesh = pd.concat(gdfs, ignore_index=True)
    mesh = gpd.GeoDataFrame(mesh, geometry="geometry", crs=gdfs[0].crs)
    mesh = mesh.to_crs(target_crs)

    # Keep only needed columns to reduce memory
    needed = [
        "G02_001",  # mesh code
        # Precip monthly + annual
        *[f"G02_{i:03d}" for i in range(2, 14)],  # 002-013 monthly precip (0.1 mm)
        "G02_014",  # annual precip (0.1 mm)
        # Temperatures (monthly)
        *[f"G02_{i:03d}" for i in range(27, 39)],  # 027-038 monthly mean temp (0.1 C)
        *[f"G02_{i:03d}" for i in range(39, 51)],  # 039-050 monthly mean min temp (0.1 C)
        # Snow depth monthly maxima (focus months: Dec-Apr; 5 fields assumed)
        *[f"G02_{i:03d}" for i in range(54, 59)],  # 054-058 monthly max snow depth (cm)
    ]
    keep_cols = [c for c in needed if c in mesh.columns]
    missing = sorted(set(needed) - set(keep_cols))
    if missing:
        logging.warning(f"Missing expected climate columns: {missing}")
    mesh = mesh[[*keep_cols, "geometry"]].copy()

    # Replace KSJ missing sentinels with NA
    MISSING_VALUES = {999999, 99999, 9999}
    for c in [col for col in mesh.columns if col.startswith("G02_")]:
        mesh[c] = pd.to_numeric(mesh[c], errors="coerce")
        mesh[c] = mesh[c].where(~mesh[c].isin(MISSING_VALUES))

    # Derived metrics at mesh level
    # Temperature scale factor (0.1 C)
    tavg_cols = [c for c in [f"G02_{i:03d}" for i in range(27, 39)] if c in mesh.columns]
    tmin_cols = [c for c in [f"G02_{i:03d}" for i in range(39, 51)] if c in mesh.columns]

    if not tavg_cols or not tmin_cols:
        logging.error("Temperature monthly columns not found as expected. Check G02 schema.")

    def _row_mean_safe(row, cols, scale: float = 1.0):
        vals = pd.to_numeric(row[cols], errors="coerce")
        return (vals.mean() / scale) if len(cols) else pd.NA

    # Annual means (skip NA after sentinel replacement)
    mesh["tavg_annual_c"] = mesh.apply(lambda r: _row_mean_safe(r, tavg_cols, 10.0), axis=1)
    mesh["tmin_annual_c"] = mesh.apply(lambda r: _row_mean_safe(r, tmin_cols, 10.0), axis=1)
    # Approximate Tmax from Tavg and Tmin: Tmax ≈ 2*Tavg - Tmin
    mesh["tmax_annual_c"] = mesh.apply(
        lambda r: (2 * r["tavg_annual_c"] - r["tmin_annual_c"]) if pd.notnull(r["tavg_annual_c"]) and pd.notnull(r["tmin_annual_c"]) else pd.NA,
        axis=1,
    )

    # Annual precipitation (0.1 mm -> mm)
    p_cols = [c for c in [f"G02_{i:03d}" for i in range(2, 14)] if c in mesh.columns]
    if "G02_014" in mesh.columns:
        pa = pd.to_numeric(mesh["G02_014"], errors="coerce")
        pa = pa.where(~pa.isin(MISSING_VALUES))
        mesh["precip_annual_mm"] = pa / 10.0
        # Fill NA using monthly sum if available
        if p_cols:
            monthly_sum = mesh.apply(
                lambda r: pd.to_numeric(r[p_cols], errors="coerce").where(~pd.to_numeric(r[p_cols], errors="coerce").isin(list(MISSING_VALUES))).sum(min_count=1),
                axis=1,
            )
            mesh.loc[mesh["precip_annual_mm"].isna(), "precip_annual_mm"] = monthly_sum[mesh["precip_annual_mm"].isna()] / 10.0
    else:
        # Fallback only: sum monthly 002-013
        mesh["precip_annual_mm"] = mesh.apply(
            lambda r: pd.to_numeric(r[p_cols], errors="coerce").where(~pd.to_numeric(r[p_cols], errors="coerce").isin(list(MISSING_VALUES))).sum(min_count=1) / 10.0,
            axis=1,
        )

    # Annual deepest snow depth (cm): max of available winter months 054-058
    snow_cols = [c for c in [f"G02_{i:03d}" for i in range(54, 59)] if c in mesh.columns]
    if snow_cols:
        def _max_ignore_missing(row):
            vals = pd.to_numeric(row[snow_cols], errors="coerce")
            vals = vals.where(~vals.isin(MISSING_VALUES))
            return vals.max()
        mesh["snow_annual_max_cm"] = mesh.apply(_max_ignore_missing, axis=1)
    else:
        mesh["snow_annual_max_cm"] = pd.NA

    return mesh


def _area_weighted_aggregate(gpd, muni_gdf, mesh_gdf, chunk_size: int = 0) -> pd.DataFrame:
    # Intersection overlay
    logging.info("Computing intersection overlay (this may take a while)...")
    inter = gpd.overlay(muni_gdf[["市区町村コード", "geometry", "_muni_area_m2"]], mesh_gdf[[
        "geometry",
        "tavg_annual_c",
        "tmin_annual_c",
        "tmax_annual_c",
        "precip_annual_mm",
        "snow_annual_max_cm",
    ]], how="intersection")

    inter["_inter_area_m2"] = inter.geometry.area
    inter = inter[inter["_inter_area_m2"] > 0].copy()

    # Weight per municipality
    inter["_w"] = inter["_inter_area_m2"] / inter["_muni_area_m2"]

    # Variable-specific normalization (exclude missing cells per variable)
    vars_ = [
        ("平均気温", "tavg_annual_c"),
        ("最低気温", "tmin_annual_c"),
        ("最高気温", "tmax_annual_c"),
        ("年降水量", "precip_annual_mm"),
        ("年最深積雪", "snow_annual_max_cm"),
    ]

    results = []
    for city, grp in inter.groupby("市区町村コード"):
        rec = {"市区町村コード": city}
        base_area = grp["_muni_area_m2"].iloc[0]
        for out_col, src_col in vars_:
            g = grp[["_w", src_col]].copy()
            g = g[g[src_col].notna()].copy()
            if g.empty:
                rec[out_col] = pd.NA
                rec.setdefault("_weight_sums", {})[out_col] = 0.0
                continue
            wsum = g["_w"].sum()
            g["_wn_var"] = g["_w"] / wsum if wsum > 0 else 0.0
            rec[out_col] = (g["_wn_var"] * g[src_col]).sum()
            rec.setdefault("_weight_sums", {})[out_col] = float(wsum)
        results.append(rec)

    out_df = pd.DataFrame(results)
    return out_df


def run(
    muni_geojson: Path,
    climate_glob: str,
    output_csv: Path,
    target_crs: str = "EPSG:32654",
    prefecture_codes: Optional[List[str]] = None,
):
    _setup_logging()
    gpd = _load_geopandas()

    muni_gdf = _read_municipalities(gpd, muni_geojson, target_crs)

    # Optionally filter municipalities by prefecture codes (prefix of 市区町村コード)
    if prefecture_codes:
        prefecture_codes = [str(c).zfill(2) for c in prefecture_codes]
        muni_gdf = muni_gdf[muni_gdf["市区町村コード"].str[:2].isin(prefecture_codes)].copy()

    climate_files = sorted([Path(p) for p in glob.glob(climate_glob)])
    mesh_gdf = _read_climate_mesh(gpd, climate_files, target_crs)

    agg = _area_weighted_aggregate(gpd, muni_gdf, mesh_gdf)

    # Validate and export
    # Ensure numeric dtype then round; keep pandas NA
    out = agg.copy()
    for col, ndigits in [("平均気温", 2), ("最低気温", 2), ("最高気温", 2)]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(ndigits)
    out["年降水量"] = pd.to_numeric(out["年降水量"], errors="coerce").round(1)
    # Snow depth integer cm (nullable integer)
    out["年最深積雪"] = pd.to_numeric(out["年最深積雪"], errors="coerce")
    out["年最深積雪"] = out["年最深積雪"].round(0).astype("Int64")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out[[
        "市区町村コード",
        "最高気温",
        "最低気温",
        "平均気温",
        "年最深積雪",
        "年降水量",
    ]].to_csv(output_csv, index=False, encoding="utf-8")

    logging.info(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate KSJ G02 (2022) climate to municipalities (area-weighted)")
    parser.add_argument(
        "--muni-geo",
        type=Path,
        default=Path("data/geojson/municipalities.geojson"),
        help="Municipality boundary GeoJSON path (must include 市区町村コード)",
    )
    parser.add_argument(
        "--climate-glob",
        type=str,
        default="data/raw/climate/G02-22_*jgd.geojson",
        help="Glob for KSJ G02 2022 GeoJSON tiles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/climate_city__v1.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--pref-codes",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of prefecture codes to filter municipalities (e.g., 19 20 21 22)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/etl_project.yaml"),
        help="Config YAML to read default region filter",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="yatsugatake_alps",
        help="study_regions key to use when --pref-codes is not provided",
    )
    args = parser.parse_args()
    pref_codes = args.pref_codes
    # If pref codes not explicitly provided, load from config/region (study_regions)
    if not pref_codes:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            pref_codes = (
                cfg.get("study_regions", {})
                .get(args.region, {})
                .get("prefecture_codes", [])
            )
        except Exception as e:
            logging.warning(f"Failed to load region defaults from {args.config}: {e}")
            pref_codes = None

    run(
        muni_geojson=args.muni_geo,
        climate_glob=args.climate_glob,
        output_csv=args.output,
        prefecture_codes=pref_codes,
    )


if __name__ == "__main__":
    main()
