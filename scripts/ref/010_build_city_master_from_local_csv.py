\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, io, sys
from pathlib import Path
import pandas as pd
from typing import Dict, List
sys.path.append('/mnt/data')
from scripts.utils.config_loader import load_yaml, render_filename

ENCODING_CANDIDATES = ["cp932", "shift_jis", "utf-8-sig", "utf-8"]
TARGET_COLS = [
    "市区町村コード","市区町村名","都道府県コード","都道府県名","市区町村ふりがな",
    "政令市･郡･支庁･振興局等","政令市･郡･支庁･振興局等（ふりがな）","過疎地域市町村","都市種別"
]

def read_csv_auto(path: Path) -> pd.DataFrame:
    b = path.read_bytes()
    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV読み込み失敗（encoding候補全滅）: {last_err}")

def to_str_strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\\u3000", " ", regex=False).str.strip()

def fix_numeric_codes(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            try:
                if df[c].dropna().astype(int).astype(float).equals(df[c].dropna()):
                    df[c] = df[c].apply(lambda x: str(int(x)) if pd.notna(x) else "")
            except Exception:
                pass
    return df

def normalize_columns_with_config(df: pd.DataFrame, colmap: Dict[str, List[str]]) -> pd.DataFrame:
    alias_to_std = {}
    for std, aliases in colmap.items():
        alias_to_std[std] = std
        for a in aliases or []:
            alias_to_std[a] = std
    rename = {c: alias_to_std.get(c, c) for c in df.columns}
    return df.rename(columns=rename)

def derive_pref_code(df: pd.DataFrame):
    # 市区町村コードを5桁ゼロ埋め→先頭2桁を都道府県コード（2桁ゼロ埋め）
    if "都道府県コード" not in df.columns and "市区町村コード" in df.columns:
        codes5 = to_str_strip(df["市区町村コード"]).str.zfill(5)
        df["都道府県コード"] = codes5.str[:2]
    df["都道府県コード"] = to_str_strip(df.get("都道府県コード", "")).str.zfill(2)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = ""
    return df

def shape_output(df: pd.DataFrame) -> pd.DataFrame:
    # コード列をゼロ埋め固定
    if "市区町村コード" in df.columns:
        df["市区町村コード"] = to_str_strip(df["市区町村コード"]).str.zfill(5)
    for c in TARGET_COLS:
        if c in df.columns:
            df[c] = to_str_strip(df[c])
    derive_pref_code(df)
    df = ensure_columns(df)
    out = df[TARGET_COLS].drop_duplicates(subset=["市区町村コード"], keep="last")
    out = out.sort_values(["都道府県コード", "市区町村コード"]).reset_index(drop=True)
    return out

def validate(out: pd.DataFrame, cfg: dict):
    vcfg = (cfg.get("validation") or {}).get("city_master") or {}
    required = vcfg.get("required") or []
    for col in required:
        if col not in out.columns:
            raise ValueError(f"必須列が不足: {col}")
    import re
    rules = vcfg.get("rules") or {}
    for col, rule in rules.items():
        if "pattern" in rule and col in out.columns:
            pat = re.compile(rule["pattern"])
            bad = ~out[col].astype(str).fillna("").str.match(pat)
            if bad.any():
                print(f"⚠ ルール違反: {col} pattern={rule['pattern']} count={bad.sum()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    colmap = (cfg.get("columns_map") or {}).get("city_master") or {}
    naming = (cfg.get("naming") or {}).get("city_master") or {}
    io_cfg = cfg.get("io") or {}
    out_dir = io_cfg.get("output_dir", "/mnt/data/data/master")
    enc_out = io_cfg.get("encoding_out", "utf-8-sig")

    df = read_csv_auto(Path(args.src))
    df = fix_numeric_codes(df)
    df = normalize_columns_with_config(df, colmap)
    out = shape_output(df)
    validate(out, cfg)

    if args.out:
        out_path = Path(args.out)
    else:
        fname = naming.get("filename_template", "city_master__all__v1.csv")
        fname = render_filename(fname, cfg.get("project") or {})
        out_path = Path(out_dir) / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding=enc_out)
    print(f"✅ 出力: {out_path}  行数={len(out)}")

if __name__ == "__main__":
    main()
