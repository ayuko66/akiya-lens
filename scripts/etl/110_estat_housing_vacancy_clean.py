\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, io, sys
from pathlib import Path
import pandas as pd
#sys.path.append("/mnt/data")
from scripts.utils.config_loader import load_yaml, render_filename
import re
import unicodedata
from glob import glob


def detect_header_row(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "表章項目 コード" in line:
                return i
    return 0

def _normalize_colname(col: str) -> str:
    # 全角→半角, NFKCで正規化
    s = unicodedata.normalize("NFKC", str(col))
    # ゼロ幅/不可視系を除去
    s = s.replace("\u200b", "").replace("\ufeff", "")
    # スペース類（半角/全角/タブ等）をすべて単一の空文字に
    s = re.sub(r"[\s\u3000]+", "", s)
    # よくある表記ゆれを寄せる
    # 例: "地域コード" と "地域 コード" は同一化済み（空白除去で同じになる）
    return s

def _build_alias_map(colmap: dict) -> dict:
    alias_to_std = {}
    for std, aliases in (colmap or {}).items():
        for cand in [std, *(aliases or [])]:
            alias_to_std[_normalize_colname(cand)] = std

    # e-Stat 揺れ対策の追加エイリアス
    extra = {
        # 2018系
        "地域コード": "市区町村コード",
        "地域ｺｰﾄﾞ": "市区町村コード",
        "地域": "市区町村名",
        "市区町村": "市区町村名",
        "団体名": "市区町村名",
        # 2023系（今回エラーのやつ）
        "全国、都道府県、市区町村コード": "市区町村コード",
        "全国、都道府県、市区町村": "市区町村名",
        # 念のためカンマ無し・全角/半角揺れも吸収
        "全国都道府県市区町村コード": "市区町村コード",
        "全国都道府県市区町村": "市区町村名",
    }
    for k, v in extra.items():
        alias_to_std[_normalize_colname(k)] = v
    return alias_to_std

def _smart_rename(df: pd.DataFrame, alias_to_std: dict) -> pd.DataFrame:
    # 正規化キーでの rename
    ren = {}
    norm_cols = {c: _normalize_colname(c) for c in df.columns}
    for orig, norm in norm_cols.items():
        if norm in alias_to_std:
            ren[orig] = alias_to_std[norm]
    df2 = df.rename(columns=ren)

    need_code = "市区町村コード" not in df2.columns
    need_name = "市区町村名" not in df2.columns

    # 追加フォールバック：正規化名で部分一致
    if need_code:
        cand_code = [
            c for c in df.columns
            if ("コード" in c and any(k in c for k in ["地域", "全国", "市区町村"]))
        ]
        if not cand_code:
            # 正規化で再探索
            cands = [c for c in df.columns if "コード" in c]
            cand_code = [c for c in cands if any(s in _normalize_colname(c) for s in ["地域","全国都道府県市区町村","市区町村"])]
        if cand_code:
            df2 = df2.rename(columns={cand_code[0]: "市区町村コード"})

    if need_name:
        cand_name = [
            c for c in df.columns
            if any(k in c for k in ["地域", "市区町村", "団体名", "全国、都道府県、市区町村"])
        ]
        cand_name = [c for c in cand_name if c != df2.get("市区町村コード", "___")]
        if cand_name:
            df2 = df2.rename(columns={cand_name[0]: "市区町村名"})

    return df2

def load_estat_table(path: Path, year: int, colmap: dict) -> pd.DataFrame:
    header_row = detect_header_row(path)
    df = pd.read_csv(path, encoding="utf-8", skiprows=header_row)

    # 1) 列名正規化 + エイリアス適用
    alias_to_std = _build_alias_map(colmap)
    df = _smart_rename(df, alias_to_std)

    # 2) 必須列チェック（ここで見つからなければ詳細を表示して落とす）
    need = ["市区町村コード","市区町村名","総数","空き家"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print("ファイル内の列名:", list(df.columns))
        raise ValueError(f"必要列が見つかりません: {miss} in {path.name}")

    # 3) 数値整形
    for c in ["総数","空き家"]:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .replace({"-": None, "": None})
                 .astype(float)
        )
    # 4) コードは5桁ゼロ埋め
    df["市区町村コード"] = df["市区町村コード"].astype(str).str.zfill(5)

    # 5) 市区町村レベルに限定（全国/都道府県などの上位集計は除外）
    df = df[df["市区町村コード"].str.match(r"^\d{5}$")]
    df = df[df["市区町村コード"].str[-3:] != "000"]

    df["year"] = int(year)
    return df[["市区町村コード","市区町村名","総数","空き家","year"]]


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", default="all")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ds = cfg["datasets"]["housing_vacancy"]
    colmap = ds.get("columns_map", {})
    raw_dir = Path(cfg["io"]["raw_dir"])
    processed_dir = Path(cfg["io"]["processed_dir"])
    enc_out = cfg["io"].get("encoding_out","utf-8-sig")

    frames = []
    for y, fname in ds["sources"].items():
        frames.append(load_estat_table(raw_dir/fname, int(y), colmap))
    tall = pd.concat(frames, ignore_index=True)


    master_spec = cfg["join"]["city_master_path"]   # パターン or 具体ファイル
    paths = sorted(glob(master_spec))
    if not paths:
        raise FileNotFoundError(f"市区町村マスタが見つかりません: {master_spec}")
    master_path = Path(paths[-1])  # ソート末尾=最新っぽいのを採用
    print(f"Using city master: {master_path}")

    master = pd.read_csv(master_path, encoding="utf-8-sig")
    master["市区町村コード"] = master["市区町村コード"].astype(str).str.zfill(5)
    master["都道府県コード"] = master["都道府県コード"].astype(str).str.zfill(2)

    tall = tall.merge(master, on=["市区町村コード","市区町村名"], how="left")

    tall["空き家率"] = (tall["空き家"] / tall["総数"] * 100).round(4)

    wide = tall.pivot_table(
        index=["市区町村コード","市区町村名","都道府県コード","都道府県名"],
        columns="year",
        values=["総数","空き家","空き家率"],
        aggfunc="sum"
    )
    wide.columns = [f"{a}_{b}" for a,b in wide.columns]
    wide = wide.reset_index()

    wide["空き家率_2023"] = wide["空き家率_2023"].round(2)
    wide["空き家_増加率_5年_%"] = ((wide["空き家_2023"] - wide["空き家_2018"]) / wide["空き家_2018"]) * 100
    wide["空き家率_差分_5年_pt"] = wide["空き家率_2023"] - wide["空き家率_2018"]

    region_cfg = cfg.get("study_regions", {}).get(args.region, {})
    pref_codes = region_cfg.get("prefecture_codes", [])
    if pref_codes:
        wide = wide[wide["都道府県コード"].isin(pref_codes)].copy()

    if args.out:
        out_path = Path(args.out)
    else:
        fname = ds["output"]["filename_template"]
        fname = render_filename(fname, cfg.get("project") or {})
        out_path = processed_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False, encoding=enc_out)
    print(f"✅ 出力: {out_path}  形状={wide.shape}")
    print(wide.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
