#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分割ソース（人口 / 出生 / 死亡 / 転入 / 転出）＋（任意）年齢別人口を統合し、
人口動態（2018, 2023=2022代用）＋正規化指標を作成。
- 転入・転出は 2018 専用CSV と 2020–2023 CSV をマージして利用
- 欠損はNaNのまま保持
- study_regions でフィルタ
- 年別CSVと縦持ちCSVを出力（render_filename対応）

★ 仕様変更:
  - 「世帯数」「転入超過率[‰]」「1世帯当たり人員」を削除

仕様： docs/人口動態ETL仕様.md
"""

import argparse, sys, re, unicodedata
from pathlib import Path
import os
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config_loader import load_yaml, render_filename

DEFAULT_YEARS = (2018, 2023)
DEFAULT_NA = ["***", "-", "X"]

# 共通キー候補
CANDS_COMMON = {
    "市区町村コード": [
        # 優先してJIS市区町村コード（補助コード）を採用
        "都道府県・市部－郡部－市区町村別 補助コード",
        "地域 補助コード",
        "地域 コード",
        "標準地域コード",
        "全国地方公共団体コード",
        "全国、都道府県、市区町村コード",
        "都道府県・市部－郡部－市区町村別 コード",
        "都道府県（特別区－指定都市再掲） 補助コード",
    ],
    "市区町村名": [
        "地域",
        "市区町村",
        "団体名",
        "全国、都道府県、市区町村",
        "都道府県・市部－郡部－市区町村別",
    ],
    # e-Statの表は「時間軸(年次)」の表記ゆれが多い
    "year_raw": [
        "調査年",
        "年次",
        "時間軸(年次)",
        "時間軸（年次）",
        r"時間軸.*年次.*",
    ],
}

# ★人口のみ（世帯数の候補は削除）
CANDS_POP = {
    "総人口": ["A1101_総人口【人】", r"^A1101_.*総人口.*"],
}

# 転入・転出（ファイル分割対応）
CANDS_MOVE_IN = {"転入者数": ["A5103_転入者数【人】", r"^A5103_.*転入者数.*"]}
CANDS_MOVE_OUT = {"転出者数": ["A5104_転出者数【人】", r"^A5104_.*転出者数.*"]}

# 年齢3区分（年齢別ファイル用）
CANDS_AGE = {
    "総人口": ["総人口", r".*総人口.*"],
    "15歳未満人口": ["15歳未満人口", r".*15.?歳.?未満.*"],
    "15〜64歳人口": [
        "15〜64歳人口",
        "15-64歳人口",
        "15～64歳人口",
        "15~64歳人口",
        r"15.?64歳.*",
    ],
    "65歳以上人口": ["65歳以上人口", r".*65.?歳.?以上.*"],
}


def _normalize_dash(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("〜", "-")
        .replace("～", "-")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("~", "-")
    )


def _detect_header_row(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if "表章項目 コード" in line or "表章項目コード" in line:
                    return i
    except Exception:
        pass
    return 0


def _read_csv_auto(
    path: Path, encs, na=DEFAULT_NA, skip_header_detect=True
) -> pd.DataFrame:
    header_row = _detect_header_row(path) if skip_header_detect else 0
    last = None
    for enc in encs:
        try:
            return pd.read_csv(
                path, encoding=enc, na_values=na, dtype=str, skiprows=header_row
            )
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to read {path} with encodings {encs}: {last}")


def _read_excel_auto(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str)


def _resolve_raw_path(raw_dir: Path, rel: str | None) -> Path | None:
    """Resolve a raw input relative path against data/raw, with fallback to population/.

    Examples:
      rel='population/xxx.csv' -> data/raw/population/xxx.csv
      rel='xxx.csv' -> try data/raw/xxx.csv, then data/raw/population/xxx.csv
    """
    if not rel:
        return None
    p = raw_dir / rel
    if p.exists():
        return p
    if not rel.startswith("population/"):
        p2 = raw_dir / "population" / rel
        if p2.exists():
            return p2
    return p


def _resolve_col(df: pd.DataFrame, candidates) -> str | None:
    cols = list(df.columns)
    norm_map = {_normalize_dash(c): c for c in cols}
    for cand in candidates if isinstance(candidates, list) else [candidates]:
        if cand in cols:
            return cand
        cn = _normalize_dash(cand)
        if cn in norm_map:
            return norm_map[cn]
        if isinstance(cand, str) and (cand.startswith("^") or ".*" in cand):
            try:
                pat = re.compile(cand)
                for c in cols:
                    if re.search(pat, c):
                        return c
            except re.error:
                pass
    return None


def _select_and_rename(df: pd.DataFrame, cand_dict: dict, strict=True) -> pd.DataFrame:
    resolved, missing = {}, {}
    for k, v in cand_dict.items():
        col = _resolve_col(df, v)
        if col is None:
            missing[k] = v
        else:
            resolved[k] = col
    if missing and strict:
        lines = ["必要列が見つかりません:"]
        for k, v in missing.items():
            lines.append(f"- {k}: 候補={v if isinstance(v, list) else [v]}")
        # よくある原因: e-Statのダウンロードで『表章項目を列に配置』ではなく縦持ち(『表章項目』『総数』列)になっている
        cols_preview = ", ".join(list(df.columns)[:8])
        lines.append(f"先頭の列: {cols_preview}")
        raise KeyError("\n".join(lines))
    out = df[[resolved[k] for k in resolved]].copy() if resolved else pd.DataFrame()
    out.rename(columns={resolved[k]: k for k in resolved}, inplace=True)
    return out


def _to_year(s: str):
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )


def _filter_by_region(df: pd.DataFrame, cfg: dict, region: str) -> pd.DataFrame:
    if not region or region == "all":
        return df
    try:
        pref_codes = cfg["study_regions"][region]["prefecture_codes"]
    except KeyError:
        raise ValueError(f"Region '{region}' not found in config")
    if not pref_codes:
        return df
    return df[df["市区町村コード"].str[:2].isin(pref_codes)].copy()


def _attach_year(
    df: pd.DataFrame, year_col: str | None, fallback_year: int
) -> pd.DataFrame:
    if df.empty:
        return df
    if year_col and year_col in df.columns:
        df["year"] = df[year_col].apply(_to_year)
        df = df.drop(columns=[year_col])
    else:
        df["year"] = fallback_year
    return df


def _load_moving_pair(
    raw_dir: Path, enc_in, file_all: str, file_2018: str, col_cands: dict
) -> pd.DataFrame:
    """2018専用CSVと2020–2023用CSVを読み、縦結合して返す。片方のみでもOK。

    - 2018: 『全国・都道府県・市区町村2018～ コード』, 『総数』, year=2018
    - 2020–2023: 『地域 コード』, 『時間軸（年次）』, 『総数』
    """
    frames = []
    out_col = list(col_cands.keys())[0] if col_cands else "値"
    if file_all:
        p_all = raw_dir / file_all
        if p_all.exists():
            df_all = _read_csv_auto(p_all, enc_in)
            sel_all = _extract_moving_all_simple(df_all, out_col=out_col)
            frames.append(sel_all)
    if file_2018:
        p_18 = raw_dir / file_2018
        if p_18.exists():
            df18 = _read_csv_auto(p_18, enc_in)
            sel18 = _extract_moving_2018_simple(df18, out_col=out_col)
            frames.append(sel18)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _norm_age_col(c: str) -> str:
    """列名正規化: 全角→半角, 空白削除, 「歳/才」除去, 波ダッシュ→'-'"""
    s = unicodedata.normalize("NFKC", str(c))
    s = s.replace("歳", "").replace("才", "")
    s = (
        s.replace("～", "-")
        .replace("~", "-")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    s = re.sub(r"\s+", "", s)
    return s


def _aggregate_age_csv(path: Path, encodings, year_out: int) -> pd.DataFrame:
    """年齢詳細CSV(0-4,5-9,...,100歳以上) → 性別=計 だけ抽出して3区分に集計"""
    df = _read_csv_auto(path, encodings, skip_header_detect=False)
    if "性別" in df.columns:
        df = df[df["性別"] == "計"].copy()

    colmap = {c: _norm_age_col(c) for c in df.columns}
    dfn = df.rename(columns=colmap)

    # 数値化（団体/名称系以外）
    keep_str = {"団体コード", "都道府県名", "市区町村名", "性別", "総数"}
    for c in dfn.columns:
        if c not in keep_str and not c.endswith("名") and not c.endswith("コード"):
            dfn[c] = pd.to_numeric(
                dfn[c].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    code_col = (
        "団体コード"
        if "団体コード" in df.columns
        else ("市区町村コード" if "市区町村コード" in df.columns else None)
    )
    name_col = (
        "市区町村名"
        if "市区町村名" in df.columns
        else ("地域" if "地域" in df.columns else None)
    )
    total_col = (
        "総数"
        if "総数" in dfn.columns
        else ("総人口" if "総人口" in dfn.columns else None)
    )
    if code_col is None or name_col is None or total_col is None:
        raise KeyError(
            "年齢別CSVに必要列（団体コード/市区町村名/総数）が見つかりません"
        )

    # 正規化後の年齢区分
    child_bins = ["0-4", "5-9", "10-14"]
    work_bins = [
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
    ]
    elder_bins = [
        "65-69",
        "70-74",
        "75-79",
        "80-84",
        "85-89",
        "90-94",
        "95-99",
    ]  # + 100以上

    def _strict_sum_bins(cols):
        # すべての必要列が存在しない場合は全行NA
        if not set(cols).issubset(dfn.columns):
            return pd.Series([pd.NA] * len(dfn), index=dfn.index)
        # どれか1つでもNAが含まれたらNA（部分和を許容しない）
        # min_count=len(cols) により、非NAの数が列数に満たないと結果はNA
        return dfn[cols].sum(axis=1, min_count=len(cols))

    hundred_col = (
        "100以上"
        if "100以上" in dfn.columns
        else ("100-" if "100-" in dfn.columns else None)
    )
    hundred_series = (
        dfn[hundred_col] if hundred_col and hundred_col in dfn.columns else 0
    )

    # 市区町村コードは『団体コード』等が長い場合があるため、先頭5桁を採用
    code_series = df[code_col].astype(str).str.slice(0, 5).str.zfill(5)
    out = pd.DataFrame(
        {
            "市区町村コード": code_series,
            "市区町村名": df[name_col].astype(str),
            # 総数はカンマ付きなので数値化ヘルパを使用
            "総人口": _to_num(dfn[total_col]),
            "15歳未満人口": _strict_sum_bins(child_bins),
            "15〜64歳人口": _strict_sum_bins(work_bins),
            # 65歳以上は『100以上』が無い場合もあるため、基本ビンは厳密合計、100以上は存在すれば加算
            "65歳以上人口": _strict_sum_bins(elder_bins)
            + (hundred_series if isinstance(hundred_series, (pd.Series, int)) else 0),
        }
    )
    out["year"] = int(year_out)
    return out


def _extract_simple_total(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """出生/死亡CSVから、表章項目や性別を参照せず『総数』のみで集約して抽出。

    - 市区町村コード: 原則『都道府県・市部－郡部－市区町村別 コード』を使用（なければ候補から解決）
    - 市区町村名: 候補から解決（『都道府県・市部－郡部－市区町村別』 など）
    - 年次: 原則『時間軸(年次) コード』を使用（なければ候補から解決）
    - 値: 『総数』を数値化し、(コード, 名称, 年次)で合計
    - 返り値: [市区町村コード, 市区町村名, year_raw, out_col]
    """
    preferred_code = [
        "都道府県・市部－郡部－市区町村別 コード",
        "地域 コード",
        "標準地域コード",
        "全国地方公共団体コード",
        "全国、都道府県、市区町村コード",
        "都道府県・市部－郡部－市区町村別 補助コード",
    ]
    preferred_year = [
        "時間軸(年次) コード",
        "時間軸（年次） コード",
    ] + CANDS_COMMON["year_raw"]

    code_col = _resolve_col(df, preferred_code)
    if code_col is None:
        code_col = _resolve_col(df, CANDS_COMMON["市区町村コード"]) or "市区町村コード"
    name_col = _resolve_col(df, CANDS_COMMON["市区町村名"]) or "市区町村名"
    year_raw_col = _resolve_col(df, preferred_year)
    if year_raw_col is None:
        year_raw_col = _resolve_col(df, CANDS_COMMON["year_raw"]) or "年次"

    if code_col is None or name_col is None or year_raw_col is None:
        raise KeyError("出生/死亡CSVの地域/年次の列が解決できませんでした")
    if "総数" not in df.columns:
        raise KeyError("出生/死亡CSVに『総数』列が見つかりません")

    dff = df.copy()
    dff["__val__"] = _to_num(dff["総数"])  # 数値化
    grp = (
        dff.groupby([code_col, name_col, year_raw_col], dropna=False)["__val__"]
        .sum()
        .reset_index()
    )
    out = grp.rename(
        columns={
            code_col: "市区町村コード",
            name_col: "市区町村名",
            year_raw_col: "year_raw",
            "__val__": out_col,
        }
    )
    return out


def _extract_moving_2018_simple(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """2018年専用の転入/転出CSVから、市区町村コードと総数のみで抽出。year=2018固定。

    - 市区町村コード: 『全国・都道府県・市区町村2018～ コード』を使用
    - 値: 『総数』を数値化
    - year: 2018 固定
    - 返り値: [市区町村コード, year, out_col]
    """
    code_col = _resolve_col(
        df,
        [
            "全国・都道府県・市区町村2018～ コード",
            "全国・都道府県・市区町村2018- コード",
            "全国・都道府県・市区町村2018~ コード",
        ],
    )
    if code_col is None:
        raise KeyError(
            "2018転入/転出CSVに『全国・都道府県・市区町村2018～ コード』が見つかりません"
        )
    if "総数" not in df.columns:
        raise KeyError("2018転入/転出CSVに『総数』列が見つかりません")

    out = pd.DataFrame(
        {
            "市区町村コード": df[code_col].astype(str).str.zfill(5),
            out_col: _to_num(df["総数"]),
            "year": 2018,
        }
    )
    return out


def _extract_moving_all_simple(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """2020–2023等の転入/転出CSVから、地域コード/年次/総数で抽出。

    - 市区町村コード: 『地域 コード』
    - year: 『時間軸（年次）』から数値抽出
    - 値: 『総数』を数値化
    - 返り値: [市区町村コード, year, out_col]
    """
    code_col = _resolve_col(df, ["地域 コード"]) or "地域 コード"
    year_col = _resolve_col(df, ["時間軸（年次）"]) or "時間軸（年次）"
    if code_col is None or year_col is None:
        raise KeyError(
            "転入/転出CSVに『地域 コード』または『時間軸（年次）』が見つかりません"
        )
    if "総数" not in df.columns:
        raise KeyError("転入/転出CSVに『総数』列が見つかりません")

    out = pd.DataFrame(
        {
            "市区町村コード": df[code_col].astype(str).str.zfill(5),
            "year": df[year_col].apply(_to_year),
            out_col: _to_num(df["総数"]),
        }
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", default="all")  # study_regions のキー

    # raw_dir 直下のファイル名（実パスは cfg["io"]["raw_dir"] を前置）
    # 生データは data/raw/population/ 配下を既定とする
    ap.add_argument("--births_file", default="population/FEH_00450011_2509_births.csv")
    ap.add_argument("--deaths_file", default="population/FEH_00450011_2509_deaths.csv")

    # 転入・転出（2020–2023と2018を分けて指定）
    ap.add_argument(
        "--moving_in_file", default="population/FEH_00200523_2509_moving_in.csv"
    )
    ap.add_argument(
        "--moving_in_2018_file",
        default="population/FEH_00200523_2509_2018_moving_in.csv",
    )
    ap.add_argument(
        "--moving_out_file", default="population/FEH_00200523_2509_moving_out.csv"
    )
    ap.add_argument(
        "--moving_out_2018_file",
        default="population/FEH_00200523_2509_2018_moving_out.csv",
    )

    # 年齢別（任意） 2018 / 2022(→2023扱い)
    ap.add_argument("--age2018_file", default="population/1804ssnen.csv")
    ap.add_argument("--age2022_file", default="population/2204ssnen.csv")

    # 出力テンプレ
    ap.add_argument(
        "--out_long_template",
        default="population_stats__${project.version}.csv",
    )
    ap.add_argument(
        "--out_year_template",
        default="population_stats_${year}__${project.version}.csv",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    enc_in = cfg["io"]["encoding_in_candidates"]
    enc_out = cfg["io"]["encoding_out"]
    raw_dir = Path(cfg["io"]["raw_dir"])
    processed_dir = Path(cfg["io"]["processed_dir"])
    project = cfg.get("project") or {}

    # 1) 基本ソース読込（人口は年齢別CSVから算出）
    births_sel = pd.DataFrame()
    deaths_sel = pd.DataFrame()
    # 出生
    if args.births_file:
        p_births = _resolve_raw_path(raw_dir, args.births_file)
        if p_births and p_births.exists():
            births_df = _read_csv_auto(p_births, enc_in)
            births_sel = _extract_simple_total(births_df, out_col="出生数")
            births_sel["市区町村コード"] = (
                births_sel["市区町村コード"].astype(str).str.zfill(5)
            )
            births_sel["year"] = births_sel["year_raw"].apply(_to_year)
            births_sel.drop(columns=["year_raw"], inplace=True)
    # 死亡
    if args.deaths_file:
        p_deaths = _resolve_raw_path(raw_dir, args.deaths_file)
        if p_deaths and p_deaths.exists():
            deaths_df = _read_csv_auto(p_deaths, enc_in)
            deaths_sel = _extract_simple_total(deaths_df, out_col="死亡数")
            deaths_sel["市区町村コード"] = (
                deaths_sel["市区町村コード"].astype(str).str.zfill(5)
            )
            deaths_sel["year"] = deaths_sel["year_raw"].apply(_to_year)
            deaths_sel.drop(columns=["year_raw"], inplace=True)

    # 出生・死亡: 『表章項目』『性別』は参照せず、
    #  市区町村コード=『都道府県・市部－郡部－市区町村別 コード』/候補、
    #  年次=『時間軸(年次) コード』/候補、
    #  値=『総数』を数値化して集約
    # 2) 転入・転出（分割ファイルの結合）

    move_in = _load_moving_pair(
        raw_dir, enc_in, args.moving_in_file, args.moving_in_2018_file, CANDS_MOVE_IN
    )
    move_out = _load_moving_pair(
        raw_dir, enc_in, args.moving_out_file, args.moving_out_2018_file, CANDS_MOVE_OUT
    )

    # 3) 年齢別人口（総人口もここから採用）
    #    引数が未指定なら、プロジェクト標準の生データパスを自動採用
    if not args.age2018_file:
        auto18 = raw_dir / "population/1804ssnen.csv"
        if auto18.exists():
            args.age2018_file = str(auto18.relative_to(raw_dir))
    if not args.age2022_file:
        auto22 = raw_dir / "population/2204ssnen.csv"
        if auto22.exists():
            args.age2022_file = str(auto22.relative_to(raw_dir))
    age18 = pd.DataFrame()
    age22 = pd.DataFrame()
    if args.age2018_file:
        age18 = _aggregate_age_csv(raw_dir / args.age2018_file, enc_in, year_out=2018)
    if args.age2022_file:
        age22 = _aggregate_age_csv(
            raw_dir / args.age2022_file, enc_in, year_out=2023
        )  # 2022→2023扱い

    # 4) マージ（市区町村コード + year）
    #    人口の基礎は年齢別CSV（性別=計）由来の総人口を使用
    frames = []
    if not age18.empty:
        frames.append(
            age18[
                [
                    "市区町村コード",
                    "市区町村名",
                    "year",
                    "総人口",
                    "15歳未満人口",
                    "15〜64歳人口",
                    "65歳以上人口",
                ]
            ]
        )
    if not age22.empty:
        frames.append(
            age22[
                [
                    "市区町村コード",
                    "市区町村名",
                    "year",
                    "総人口",
                    "15歳未満人口",
                    "15〜64歳人口",
                    "65歳以上人口",
                ]
            ]
        )

    if frames:
        base = pd.concat(frames, ignore_index=True)
    else:
        raise RuntimeError(
            "年齢別人口CSVが見つかりません。--age2018_file および/または --age2022_file を指定してください。"
        )
    if not births_sel.empty:
        base = base.merge(
            births_sel[["市区町村コード", "year", "出生数"]],
            on=["市区町村コード", "year"],
            how="left",
        )
    if not deaths_sel.empty:
        base = base.merge(
            deaths_sel[["市区町村コード", "year", "死亡数"]],
            on=["市区町村コード", "year"],
            how="left",
        )
    if not move_in.empty:
        base = base.merge(
            move_in[["市区町村コード", "year", "転入者数"]],
            on=["市区町村コード", "year"],
            how="left",
        )
    if not move_out.empty:
        base = base.merge(
            move_out[["市区町村コード", "year", "転出者数"]],
            on=["市区町村コード", "year"],
            how="left",
        )
    # 3.5) 年齢3区分は base に既に含まれているため、重複マージはしない
    #      ただし、過去の実行で重複列（*_x, *_y）が混入したケースに備え、後で合成する

    # 5) 数値化・派生（★ 世帯数・転入超過率・1世帯当たり人員は廃止）
    for c in [
        "総人口",
        "出生数",
        "死亡数",
        "転入者数",
        "転出者数",
        "15歳未満人口",
        "15〜64歳人口",
        "65歳以上人口",
    ]:
        if c in base.columns:
            base[c] = _to_num(base[c])

    # 5.5) 年齢3区分の重複列がある場合は統合（*_x を優先、無い場合 *_y）
    def _coalesce(dst: str):
        x, y = f"{dst}_x", f"{dst}_y"
        if x in base.columns or y in base.columns:
            base[dst] = base[x] if x in base.columns else pd.NA
            if y in base.columns:
                base[dst] = base[dst].fillna(base[y])

    for col in ["15歳未満人口", "15〜64歳人口", "65歳以上人口"]:
        _coalesce(col)

    # 存在しない列は空列を用意（派生率計算を素直にNaNで通すため）
    for cc in ["出生数", "死亡数", "転入者数", "転出者数"]:
        if cc not in base.columns:
            base[cc] = pd.NA

    pop = base.get("総人口")
    if pop is not None:
        if "65歳以上人口" in base.columns:
            base["高齢化率[%]"] = base["65歳以上人口"] / pop * 100
        if "15歳未満人口" in base.columns:
            base["年少人口率[%]"] = base["15歳未満人口"] / pop * 100
        if "15〜64歳人口" in base.columns:
            base["生産年齢人口率[%]"] = base["15〜64歳人口"] / pop * 100
        if "出生数" in base.columns:
            base["出生率[‰]"] = base["出生数"] / pop * 1000
        if "死亡数" in base.columns:
            base["死亡率[‰]"] = base["死亡数"] / pop * 1000
        if "転入者数" in base.columns:
            base["転入率[‰]"] = base["転入者数"] / pop * 1000
        if "転出者数" in base.columns:
            base["転出率[‰]"] = base["転出者数"] / pop * 1000
        # ★ 転入超過率[‰] は削除

    # 6) 地域フィルタ
    base = _filter_by_region(base, cfg, args.region)

    # 6.5) 集計単位のノイズ除去（都道府県レベルなど）: コードが末尾"000"の行は除外
    if "市区町村コード" in base.columns:
        base = base[~base["市区町村コード"].astype(str).str.endswith("000")].copy()

    # 7) 出力
    processed_dir.mkdir(parents=True, exist_ok=True)
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
        "高齢化率[%]",
        "年少人口率[%]",
        "生産年齢人口率[%]",
        "出生率[‰]",
        "死亡率[‰]",
        "転入率[‰]",
        "転出率[‰]",
        # ★ 「世帯数」「転入超過率[‰]」「1世帯当たり人員」は出力から削除
    ]
    base_cols = [c for c in base_cols if c in base.columns]

    # 年別
    for y in DEFAULT_YEARS:
        dfy = base.loc[base["year"] == y, base_cols].copy()
        out_name = render_filename(
            args.out_year_template.replace("${year}", str(y)), project
        )
        if args.region != "all":
            out_name = out_name.replace(".csv", f"__{args.region}.csv")
        out_path = processed_dir / out_name
        dfy.to_csv(out_path, index=False, encoding=enc_out)
        print(f"✅ 出力: {out_path}  shape={dfy.shape}")

    # 縦持ち
    long_name = render_filename(args.out_long_template, project)
    if args.region != "all":
        long_name = long_name.replace(".csv", f"__{args.region}.csv")
    long_path = processed_dir / long_name
    base[base_cols].sort_values(["市区町村コード", "year"]).to_csv(
        long_path, index=False, encoding=enc_out
    )
    print(f"✅ 出力: {long_path}  shape={base[base_cols].shape}")


if __name__ == "__main__":
    main()
