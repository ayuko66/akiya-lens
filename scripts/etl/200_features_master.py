import argparse
import sys
import pandas as pd
from pathlib import Path

# Ensure project root on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config_loader import load_yaml


def load_csv(path: Path, dtype_code="Int64") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize key
    if "市区町村コード" in df.columns:
        # Use pandas nullable integer to handle joins with NaNs safely
        df["市区町村コード"] = df["市区町村コード"].astype(dtype_code)
    return df


def pivot_population_wide(pop_long: pd.DataFrame) -> pd.DataFrame:
    # Keep only years we care about
    pop = pop_long[pop_long["year"].isin([2018, 2023])].copy()

    # Ensure code is key type
    pop["市区町村コード"] = pop["市区町村コード"].astype("Int64")

    # Compute 転入超過率[‰] if missing
    if "転入超過率[‰]" not in pop.columns:
        if {"転入率[‰]", "転出率[‰]"}.issubset(pop.columns):
            pop["転入超過率[‰]"] = pop["転入率[‰]"] - pop["転出率[‰]"]
        else:
            pop["転入超過率[‰]"] = pd.NA

    # Columns to include from population stats
    cols_keep = [
        "総人口",
        "15歳未満人口",
        "15〜64歳人口",
        "65歳以上人口",
        "出生数",
        "死亡数",
        "転入者数",
        "転出者数",
        "高齢化率[%]",
        "年少人口率[%]",
        "生産年齢人口率[%]",
        "出生率[‰]",
        "死亡率[‰]",
        "転入率[‰]",
        "転出率[‰]",
        "転入超過率[‰]",
    ]

    # Narrow to usable columns
    base_cols = ["市区町村コード", "市区町村名", "year"]
    use_cols = [c for c in base_cols + cols_keep if c in pop.columns]
    pop_narrow = pop[use_cols].copy()

    # Pivot to wide with year prefix
    pop_wide = pop_narrow.set_index(["市区町村コード", "市区町村名", "year"]).unstack("year")
    # Flatten multiindex columns and rename to desired pattern: {year}_{col}
    pop_wide.columns = [f"{int(year)}_{col}" for col, year in pop_wide.columns]
    pop_wide = pop_wide.reset_index()


    return pop_wide


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Args: read region filter from config to unify behavior
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(repo_root / "config/etl_project.yaml"))
    ap.add_argument("--region", default="yatsugatake_alps")
    args = ap.parse_args()

    # Paths
    p_feh2023_raw = repo_root / "data/raw/FEH_00200522_2023_2508.csv"
    p_pop_long = repo_root / f"data/processed/population_stats__v1__{args.region}.csv"
    p_osm = repo_root / "data/processed/osm_density_with_pt_clinic.csv"
    p_land = repo_root / "data/processed/132_landprice_residential_median__wide__v1.csv"
    p_clim = repo_root / "data/processed/climate_city__v1.csv"
    p_housing = repo_root / "data/processed/housing_vacancy__wide__v1_preview.csv"
    p_citym = repo_root / "data/master/city_master__all__v1_preview.csv"

    # Load
    pop_long = load_csv(p_pop_long)
    osm = load_csv(p_osm)
    land = load_csv(p_land)
    clim = load_csv(p_clim)
    housing = load_csv(p_housing)
    citym = load_csv(p_citym)

    # Load target prefecture codes from config (yatsugatake_alps by default)
    cfg = load_yaml(args.config)
    pref_codes = cfg.get("study_regions", {}).get(args.region, {}).get("prefecture_codes", [])
    pref_codes = [str(c).zfill(2) for c in pref_codes]

    # Build base municipalities from FEH 2023 raw (exclude national/pref summaries)
    feh = pd.read_csv(p_feh2023_raw)
    code_col = "全国、都道府県、市区町村 コード"
    name_col = "全国、都道府県、市区町村"
    # Keep rows with 5-digit municipal codes; drop rows ending with '000' (prefecture-level)
    feh[code_col] = feh[code_col].astype(str).str.zfill(5)
    base_codes = (
        feh[feh[code_col].str.match(r"^[0-9]{5}$") & ~feh[code_col].str.endswith("000")][[code_col, name_col]]
        .drop_duplicates()
        .rename(columns={code_col: "市区町村コード", name_col: "市区町村名"})
    )
    # Restrict to study region (config-driven). If empty, keep all.
    if pref_codes:
        base_codes = base_codes[base_codes["市区町村コード"].str[:2].isin(pref_codes)].copy()
    base_codes["市区町村コード"] = base_codes["市区町村コード"].astype("Int64")

    # Population wide
    pop_wide = pivot_population_wide(pop_long)

    # Base frame: municipalities from FEH 2023 raw
    df = base_codes.copy()

    # Attach prefecture info from city master
    citym_all = citym[["市区町村コード", "都道府県コード", "都道府県名"]].drop_duplicates()
    df = df.merge(citym_all, on="市区町村コード", how="left")

    # Merge processed housing vacancy (wide) to bring 2018/2023 metrics where available
    hv_cols = [
        "市区町村コード",
        "住宅総数_2018",
        "住宅総数_2023",
        "空き家_2018",
        "空き家_2023",
        "空き家率_2018",
        "空き家率_2023",
        "空き家_増加率_5年_%",
        "空き家率_差分_5年_pt",
    ]
    df = df.merge(housing[[c for c in hv_cols if c in housing.columns]], on="市区町村コード", how="left")

    # Compute 2023 vacancy metrics directly from raw FEH to fill gaps
    feh_sel = feh[[code_col, name_col, "総数", "空き家"]].copy()
    feh_sel[code_col] = feh_sel[code_col].astype(str).str.zfill(5)
    feh_sel = feh_sel[feh_sel[code_col].isin(df["市区町村コード"].astype(str))]
    # to numeric
    for c in ["総数", "空き家"]:
        feh_sel[c] = pd.to_numeric(feh_sel[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    feh_sel = (
        feh_sel.groupby([code_col], as_index=False)[["総数", "空き家"]].sum(min_count=1)
        .rename(columns={code_col: "市区町村コード", "総数": "住宅総数_2023_raw", "空き家": "空き家_2023_raw"})
    )
    feh_sel["市区町村コード"] = feh_sel["市区町村コード"].astype("Int64")
    feh_sel["空き家率_2023_raw"] = (feh_sel["空き家_2023_raw"] / feh_sel["住宅総数_2023_raw"]) * 100
    df = df.merge(feh_sel, on="市区町村コード", how="left")
    # Fill missing processed metrics with raw-derived ones
    for tgt, raw in [
        ("住宅総数_2023", "住宅総数_2023_raw"),
        ("空き家_2023", "空き家_2023_raw"),
        ("空き家率_2023", "空き家率_2023_raw"),
    ]:
        if tgt in df.columns and raw in df.columns:
            df[tgt] = df[tgt].fillna(df[raw])

    # Merge population (wide) by code only to avoid name mismatches
    df = df.merge(pop_wide.drop(columns=["市区町村名"], errors="ignore"), on="市区町村コード", how="left")

    # Merge OSM (counts and area densities)
    df = df.merge(
        osm[[
            "市区町村コード",
            "駅件数", "駅密度[件/km²]",
            "スーパー件数", "スーパー密度[件/km²]",
            "学校件数", "学校密度[件/km²]",
            "病院件数", "病院密度[件/km²]",
        ]],
        on="市区町村コード", how="left",
    )

    # Merge land prices
    df = df.merge(
        land[[
            "市区町村コード",
            "住宅地価_log中央値_2018",
            "住宅地価_log中央値_2023",
            "住宅地価_log差分",
            "住宅地価_中央値_2018",
            "住宅地価_中央値_2023",
            "住宅地価_増減率[%]",
            "標準地点数_2018",
            "標準地点数_2023",
        ]],
        on="市区町村コード", how="left",
    )

    # Merge climate
    df = df.merge(
        clim[[
            "市区町村コード",
            "平均気温",
            "年最深積雪",
            "年降水量",
            "最低気温",
            "最高気温",
        ]],
        on="市区町村コード", how="left",
    )

    # Merge city master (categorical flags)
    df = df.merge(
        citym[["市区町村コード", "過疎地域市町村", "都市種別"]],
        on="市区町村コード", how="left",
    )

    # Compute population-adjusted densities for 2023
    pop_col = "2023_総人口"
    if pop_col in df.columns:
        # Avoid division by zero
        denom = pd.to_numeric(df[pop_col], errors="coerce").replace({0: pd.NA})
        df["2023年総人口あたりのスーパー密度"] = df["スーパー件数"] / denom
        df["2023年総人口あたりの学校密度"] = df["学校件数"] / denom
        df["2023年総人口あたりの病院密度"] = df["病院件数"] / denom
        df["2023年総人口あたりの駅密度"] = df["駅件数"] / denom
    else:
        # Create empty columns if population missing (should not happen for included areas)
        for cname in [
            "2023年総人口あたりのスーパー密度",
            "2023年総人口あたりの学校密度",
            "2023年総人口あたりの病院密度",
            "2023年総人口あたりの駅密度",
        ]:
            df[cname] = pd.NA

    # Reorder columns to match the requested list as much as possible
    desired_order = [
        # population 2018
        "2018_15〜64歳人口",
        "2018_15歳未満人口",
        "2018_65歳以上人口",
        "2018_出生数",
        "2018_出生率[‰]",
        "2018_年少人口率[%]",
        "2018_死亡数",
        "2018_死亡率[‰]",
        "2018_生産年齢人口率[%]",
        "2018_総人口",
        "2018_転入率[‰]",
        "2018_転入者数",
        "2018_転入超過率[‰]",
        "2018_転出率[‰]",
        "2018_転出者数",
        "2018_高齢化率[%]",
        # OSM + adjusted densities
        "スーパー件数",
        "スーパー密度[件/km²]",
        "2023年総人口あたりのスーパー密度",
        # land price
        "住宅地価_log中央値_2018",
        "住宅地価_log中央値_2023",
        "住宅地価_log差分",
        "住宅地価_中央値_2018",
        "住宅地価_中央値_2023",
        "住宅地価_増減率[%]",
        # OSM school
        "学校件数",
        "学校密度[件/km²]",
        "2023年総人口あたりの学校密度",
        # climate
        "平均気温",
        "年最深積雪",
        "年降水量",
        "最低気温",
        "最高気温",
        # land standard points
        "標準地点数_2018",
        "標準地点数_2023",
        # hospitals
        "病院件数",
        "病院密度[件/km²]",
        "2023年総人口あたりの病院密度",
        # vacancy
        "空き家_2018",
        "空き家_2023",
        "空き家_増加率_5年_%",
        "空き家率_2018",
        "空き家率_2023",
        "空き家率_差分_5年_pt",
        # households
        "住宅総数_2018",
        "住宅総数_2023",
        # city master flags
        "過疎地域市町村",
        "都市種別",
        # prefecture code
        "都道府県コード",
        # stations
        "駅件数",
        "駅密度[件/km²]",
        "2023年総人口あたりの駅密度",
    ]

    # Always include identifying keys at the front
    front = ["市区町村コード", "市区町村名", "都道府県名"]
    cols = front + [c for c in desired_order if c in df.columns]
    # Include any other columns that were not explicitly ordered (append at end)
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]

    # Output
    out_path = repo_root / "data/processed/features_master__wide__v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path.relative_to(repo_root)} with {len(df)} rows, cols={len(df.columns)} region={args.region}"
    )


if __name__ == "__main__":
    main()
