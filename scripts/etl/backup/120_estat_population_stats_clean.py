# -*- coding: utf-8 -*-
"""
FEI_CITY_2509.csv（社会・人口統計体系）から人口動態（2018, 2023）を整形。
- 正規化指標を付与
- etl_project.yaml の study_regions で市区町村をフィルタ
- 列名の表記ゆれ（15〜64/15-64/15～64 等）をロバストに解決
- 年別CSVと縦持ちCSVを出力
"""

from pathlib import Path
import argparse
import sys
import re
import pandas as pd

# Ensure project root is on sys.path when run as a file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config_loader import load_yaml

DEFAULT_NA = ["***", "-", "X"]
DEFAULT_YEARS = (2018, 2023)

# デフォルト候補（「最有力」→「別表記」→「コード名正規表現」）を用意
DEFAULT_COLS_CANDIDATES = {
    "市区町村コード": ["地域 コード", "標準地域コード", "全国地方公共団体コード"],
    "市区町村名": ["地域", "市区町村", "団体名"],
    "year_raw": ["調査年", "年次"],
    "総人口": ["A1101_総人口【人】", r"^A1101_.*総人口.*"],
    "15歳未満人口": ["A1301_15歳未満人口【人】", r"^A1301_.*15.*未満.*人口.*"],
    # ←ここが揺れやすい
    "15〜64歳人口": [
        "A1302_15〜64歳人口【人】",
        "A1302_15-64歳人口【人】",
        "A1302_15～64歳人口【人】",
        "A1302_15~64歳人口【人】",
        r"^A1302_.*15.?64歳.*人口.*",
    ],
    "65歳以上人口": ["A1303_65歳以上人口【人】", r"^A1303_.*65歳以上.*人口.*"],
    "出生数": ["A4101_出生数【人】", r"^A4101_.*出生数.*"],
    "死亡数": ["A4200_死亡数【人】", r"^A4200_.*死亡数.*"],
    "転入者数": ["A5103_転入者数【人】", r"^A5103_.*転入者数.*"],
    "転出者数": ["A5104_転出者数【人】", r"^A5104_.*転出者数.*"],
    "世帯数": ["A7101_世帯数【世帯】", "A7101_世帯数", r"^A7101_.*世帯数.*"],
}

DERIVED_COLS = [
    "高齢化率[%]",
    "年少人口率[%]",
    "生産年齢人口率[%]",
    "出生率[‰]",
    "死亡率[‰]",
    "転入率[‰]",
    "転出率[‰]",
    "転入超過率[‰]",
    "1世帯当たり人員",
]


def _to_int_year(s: str):
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None


def _read_csv_with_enc_candidates(
    path: Path, enc_candidates: list, na_values=DEFAULT_NA
) -> pd.DataFrame:
    last_err = None
    for enc in enc_candidates:
        try:
            return pd.read_csv(path, encoding=enc, na_values=na_values, dtype=str)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Failed to read {path} with encodings {enc_candidates}: {last_err}"
    )


# 記号の表記ゆれを正規化（全角/半角/波ダッシュ類/ダッシュ類）
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s2 = str(s)
    # 波ダッシュ・全角チルダ・全角ハイフンなどを半角ハイフンに寄せる
    s2 = (
        s2.replace("〜", "-")
        .replace("～", "-")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("~", "-")
    )
    # フルブラケットなどの統一（必要なら拡張）
    return s2


def _resolve_column(df: pd.DataFrame, candidates: list) -> str | None:
    cols = list(df.columns)
    cols_norm = {_normalize(c): c for c in cols}  # 正規化→元名

    for cand in candidates:
        # 1) まずは完全一致（そのまま）
        if cand in cols:
            return cand
        # 2) 正規化して一致
        candn = _normalize(cand)
        if candn in cols_norm:
            return cols_norm[candn]
        # 3) 正規表現（^A1302_... など）
        if cand.startswith("^") or ("(" in cand or ".*" in cand or "?" in cand):
            pat = re.compile(cand)
            for c in cols:
                if re.search(pat, c):
                    return c
    return None


def _build_cols_map_from_cfg_or_default(cfg: dict, df: pd.DataFrame) -> dict:
    """
    etl_project.yaml に datasets.population_stats.columns_map があればそれも候補に含め、
    実在する列名へ解決する。
    """
    user_map = (
        cfg.get("datasets", {}).get("population_stats", {}).get("columns_map", {})
    )
    # ユーザー指定があれば先頭に差し込んで優先（配列なら順に優先）
    candidates = {
        k: list(v) if isinstance(v, list) else [v]
        for k, v in DEFAULT_COLS_CANDIDATES.items()
    }
    for k, v in user_map.items():
        if isinstance(v, list):
            candidates.setdefault(k, [])
            candidates[k] = list(v) + candidates[k]
        else:
            candidates.setdefault(k, [])
            candidates[k] = [v] + candidates[k]

    resolved = {}
    missing = {}
    for k, cand_list in candidates.items():
        col = _resolve_column(df, cand_list)
        if col is None:
            missing[k] = cand_list
        else:
            resolved[k] = col

    if missing:
        # デバッグを助けるため、各キーで正規表現に合いそうな列候補も例示する
        tips = []
        for k, cand_list in missing.items():
            hints = []
            for cand in cand_list:
                try:
                    if cand.startswith("^") or ".*" in cand:
                        pat = re.compile(cand)
                        match_cols = [c for c in df.columns if re.search(pat, c)]
                        if match_cols:
                            hints.extend(match_cols)
                except re.error:
                    pass
            hint_txt = (
                f"  - {k}: 候補={cand_list[:3]}… 近い列例={hints[:3]}"
                if hints
                else f"  - {k}: 候補={cand_list[:3]}…"
            )
            tips.append(hint_txt)
        raise KeyError("入力CSVに必要列が見つかりません。\n" + "\n".join(tips))

    return resolved


def _filter_by_region(df: pd.DataFrame, region: str, cfg: dict) -> pd.DataFrame:
    if not region or region == "all":
        return df
    try:
        pref_codes = cfg["study_regions"][region]["prefecture_codes"]
    except KeyError:
        raise ValueError(f"Region '{region}' not found in config")
    if not pref_codes:
        return df
    return df[df["市区町村コード"].str[:2].isin(pref_codes)].copy()


def main(
    config_path: str, in_file: str, out_dir: str, region: str, years=DEFAULT_YEARS
):
    cfg = load_yaml(config_path)
    enc_in = cfg["io"]["encoding_in_candidates"]
    enc_out = cfg["io"]["encoding_out"]

    df_raw = _read_csv_with_enc_candidates(Path(in_file), enc_in)

    # 列名解決（表記ゆれに強い）
    cols_map = _build_cols_map_from_cfg_or_default(cfg, df_raw)

    # 必要列抽出 → 日本語標準名に
    df = df_raw[[cols_map[k] for k in cols_map]].copy()
    df.rename(columns={v: k for k, v in cols_map.items()}, inplace=True)

    # 整形
    df["市区町村コード"] = df["市区町村コード"].astype(str).str.zfill(5)
    df["year"] = df["year_raw"].apply(_to_int_year)
    df.drop(columns=["year_raw"], inplace=True)

    # 数値化
    num_cols = [
        "総人口",
        "15歳未満人口",
        "15〜64歳人口",
        "65歳以上人口",
        "出生数",
        "死亡数",
        "転入者数",
        "転出者数",
        "世帯数",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )

    # 正規化・派生
    pop = df["総人口"]
    hh = df["世帯数"]
    df["高齢化率[%]"] = (df["65歳以上人口"] / pop * 100).where(pop > 0)
    df["年少人口率[%]"] = (df["15歳未満人口"] / pop * 100).where(pop > 0)
    df["生産年齢人口率[%]"] = (df["15〜64歳人口"] / pop * 100).where(pop > 0)
    df["出生率[‰]"] = (df["出生数"] / pop * 1000).where(pop > 0)
    df["死亡率[‰]"] = (df["死亡数"] / pop * 1000).where(pop > 0)
    df["転入率[‰]"] = (df["転入者数"] / pop * 1000).where(pop > 0)
    df["転出率[‰]"] = (df["転出者数"] / pop * 1000).where(pop > 0)
    df["転入超過率[‰]"] = ((df["転入者数"] - df["転出者数"]) / pop * 1000).where(
        pop > 0
    )
    df["1世帯当たり人員"] = (pop / hh).where(hh > 0)

    # 地域フィルタ
    df = _filter_by_region(df, region, cfg)

    # 出力
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "市区町村コード",
        "市区町村名",
        "year",
        "総人口",
        "15歳未満人口",
        "15〜64歳人口",
        "65歳以上人口",
        "出生数",
        "死亡数",
        "転入者数",
        "転出者数",
        "世帯数",
    ] + DERIVED_COLS

    for y in years:
        dfy = df.loc[df["year"] == y, base_cols].copy()
        dfy.to_csv(
            out_dir / f"population_stats_{region}_{y}.csv",
            index=False,
            encoding=enc_out,
        )

    long_path = out_dir / f"population_stats_{region}__long__v1_preview.csv"
    df[base_cols].sort_values(["市区町村コード", "year"]).to_csv(
        long_path, index=False, encoding=enc_out
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_file", default="data/raw/FEI_CITY_2509.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument(
        "--region", default="all", help="study_regions のキー名 (例: yatsugatake_alps)"
    )
    args = ap.parse_args()
    main(args.config, args.in_file, args.out_dir, args.region)
