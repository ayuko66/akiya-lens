#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weighted residual CatBoost for vacancy-rate change (2018→2023)
- target: Δp = p23 - p18  （--target dlogit で Δlogit に切替可）
- sample_weight: inverse-variance using housing stock N18/N23（無ければ均一重み）
- features: 2018_* 水準（空き家率_2018は除外） + 自動Δ(23−18)生成（空き家率は除外）
- evaluations:
    ① Global 5-fold OOF
    ② Regime-CV（2018空き家率の分位で low/mid/high）
    ③ Leave-Regime-Out（他レジームで学習→当該レジームで評価）
- OOF-SHAP: foldごとに検証集合のSHAPを集め、自治体別TopKをCSVに保存
- model save: 学習済みGlobalモデルを .cbm へ保存（アプリから読み込み可）

Usage:
  python scripts/train_residual_catboost.py \
      --data data/processed/features_master__wide__v1.csv \
      --out_model models/catboost_residual_model.cbm \
      --out_metrics data/processed/model_metrics.json \
      --out_shap data/processed/shap_topk.csv \
      --loss Huber:delta=1.0 \
      --target delta        # or dlogit
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import re


# ---------- utils ----------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("‰", "", regex=False),
        errors="coerce",
    )


def _prep_code(x):
    """市区町村コードを5桁ゼロ埋め。SeriesでもスカラでもOK。"""
    if isinstance(x, pd.Series):
        s = x.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        return s  # Seriesを返す
    # スカラ（numpy.str_ / int / float など）
    s = re.sub(r"\.0$", "", str(x))
    return s.zfill(5)  # 文字列を返す


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.clip(p, eps, 1 - eps)
    return np.log(x / (1 - x))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# 置き換え版（スクリプト内の _weighted_scores を丸ごと置換）
def _weighted_scores(y_true, y_pred, w):
    # scikit-learn のバージョンによっては mean_squared_error に squared 引数が無い
    try:
        rmse = mean_squared_error(y_true, y_pred, sample_weight=w, squared=False)
    except TypeError:
        # 古い版：squared引数なし → MSEを計算して自前で平方根をとる
        mse = mean_squared_error(y_true, y_pred, sample_weight=w)
        rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred, sample_weight=w)
    return float(rmse), float(r2)


def _pairwise_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """2018_XXX と 2023_XXX の**双方が数値**の列について Δ を自動生成。
    空き家率はターゲットなので除外。新列名は `<ベース名>`。"""
    out = df.copy()
    c18 = [c for c in df.columns if c.startswith("2018_")]
    for c in c18:
        base = c.replace("2018_", "", 1)
        c23 = "2023_" + base
        if c23 in df.columns and base not in ("空き家率",):
            a = _to_num(df[c])
            b = _to_num(df[c23])
            if a.notna().sum() >= 10 and b.notna().sum() >= 10:
                out[f"{base}△"] = b - a
    return out


# ---------- feature builder ----------
def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """2018水準（空き家率_2018は除外） + 自動Δ を使う。"""
    df2 = _pairwise_deltas(df)

    # 2018_* のうち空き家率_2018以外を採用
    cols_2018 = [
        c for c in df2.columns if c.startswith("2018_") and c != "2018_空き家率"
    ]
    # 名前が日本語の列も混じるため、数値化できるもののみ採用
    cols_num_2018 = []
    for c in cols_2018:
        s = _to_num(df2[c])
        if s.notna().sum() >= len(s) * 0.4:  # 4割以上が数値ならOK
            df2[c] = s
            cols_num_2018.append(c)

    # 追加：静的な項目として扱う列（存在していれば）
    static_cands = [
        "平均気温",
        "年最深積雪",
        "年降水量",
        "最低気温",
        "スーパー密度[件/km²]",
        "学校密度[件/km²]",
        "病院密度[件/km²]",
        "駅密度[件/km²]",
    ]
    static_cols = []
    for c in static_cands:
        if c in df2.columns:
            df2[c] = _to_num(df2[c])
            static_cols.append(c)

    delta_cols = [c for c in df2.columns if c.endswith("△")]

    use_cols = cols_num_2018 + static_cols + delta_cols
    X = df2[use_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, use_cols


# ---------- weights ----------
def make_inverse_variance_weights(df: pd.DataFrame) -> pd.Series:
    """Var(Δp) ≈ p23(1-p23)/N23 + p18(1-p18)/N18 の逆数。なければ均一。"""
    have_n = all(c in df.columns for c in ["住宅総数_2018", "住宅総数_2023"])
    p18 = _to_num(df["空き家率_2018"]) / 100.0
    p23 = _to_num(df["空き家率_2023"]) / 100.0
    if have_n:
        N18 = _to_num(df["住宅総数_2018"])
        N23 = _to_num(df["住宅総数_2023"])
        var = p23 * (1 - p23) / N23 + p18 * (1 - p18) / N18
        # クリップで極端値を抑制
        lo, hi = np.nanpercentile(var, [5, 95])
        var = var.clip(lower=lo, upper=hi)
        w = 1.0 / var
    else:
        w = pd.Series(1.0, index=df.index, dtype=float)
    w = w.fillna(w.median())
    return w


# ---------- regimes ----------
def make_regime(p18_percent: pd.Series, q=(1 / 3, 2 / 3)) -> pd.Series:
    p = _to_num(p18_percent) / 100.0
    q1, q2 = np.quantile(p.dropna(), q)
    lab = pd.cut(p, bins=[-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
    return lab


# ---------- OOF + SHAP ----------
def oof_train_eval(
    X: pd.DataFrame,
    y: np.ndarray,
    w: pd.Series,
    params: Dict,
    seed: int = 42,
    cat_features: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[str, float], List[CatBoostRegressor], pd.DataFrame]:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    models, shap_rows = [], []

    for tr, te in kf.split(X):
        m = CatBoostRegressor(**params)
        m.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=w.iloc[tr].values,
            eval_set=(X.iloc[te], y[te]),
            verbose=False,
        )
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        models.append(m)

        # OOF-SHAP: 検証集合に対してのみ
        try:
            pool = Pool(X.iloc[te])
            sv = m.get_feature_importance(pool, type="ShapValues")
            sv = sv[:, :-1]  # 最後の列は期待値
            shap_rows.append(pd.DataFrame(sv, index=X.index[te], columns=X.columns))
        except Exception as e:
            print(f"[WARN] SHAP計算に失敗: {e}")

    rmse, r2 = _weighted_scores(y, oof, w)
    metrics = {"rmse": rmse, "r2": r2}

    shap_oof = (
        pd.concat(shap_rows, axis=0).reindex(X.index)
        if shap_rows
        else pd.DataFrame(index=X.index, columns=X.columns)
    )
    return oof, metrics, models, shap_oof


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/features_master__wide__v1.csv"),
    )
    ap.add_argument(
        "--out_model", type=Path, default=Path("models/catboost_residual_model.cbm")
    )
    ap.add_argument(
        "--out_metrics", type=Path, default=Path("data/processed/model_metrics.json")
    )
    ap.add_argument(
        "--out_shap", type=Path, default=Path("data/processed/shap_topk.csv")
    )
    ap.add_argument(
        "--target", choices=["delta", "dlogit"], default="delta"
    )  # Δp or Δlogit
    ap.add_argument(
        "--loss", default="Huber:delta=1.0"
    )  # e.g., RMSE / MAE / Quantile:alpha=0.5
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "市区町村コード" not in df.columns:
        raise SystemExit("列『市区町村コード』が見つかりません。")

    # 必須列チェック
    need = ["空き家率_2018", "空き家率_2023"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"列『{c}』が見つかりません。")

    # 基本ベクトル
    code = _prep_code(df["市区町村コード"])
    p18 = _to_num(df["空き家率_2018"]) / 100.0
    p23 = _to_num(df["空き家率_2023"]) / 100.0
    mask = p18.notna() & p23.notna()
    dfm = df.loc[mask].copy()
    code = code.loc[mask]
    p18 = p18.loc[mask].values
    p23 = p23.loc[mask].values

    # ターゲット
    if args.target == "delta":
        y = (p23 - p18).astype(float)
        to_display_pred = lambda p18_, yhat_: (p18_ + yhat_)  # 予測後のp23
    else:
        y = (_logit(p23) - _logit(p18)).astype(float)
        to_display_pred = lambda p18_, dl_: _sigmoid(_logit(p18_) + dl_)

    # 重み
    w = make_inverse_variance_weights(dfm)

    # 特徴量
    X, feature_names = build_feature_matrix(dfm)

    # レジーム
    regime = make_regime(dfm["空き家率_2018"])

    # モデル設定
    params = dict(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        loss_function=args.loss,
        eval_metric="RMSE",
        random_seed=args.seed,
        verbose=False,
    )

    # ① Global OOF
    oof, metrics_global, models, shap_oof = oof_train_eval(
        X, y, w, params, seed=args.seed
    )
    print(f"[Global] RMSE={metrics_global['rmse']:.3f}  R²={metrics_global['r2']:.3f}")

    # ② Regime-CV
    scores_reg: Dict[str, Dict[str, float]] = {}
    for r in ["low", "mid", "high"]:
        idx = regime.values == r
        if idx.sum() < 40:
            continue
        oof_r, m_r, _, _ = oof_train_eval(
            X.iloc[idx], y[idx], w.iloc[idx], params, seed=args.seed + 1
        )
        scores_reg[r] = m_r
    for r, m in scores_reg.items():
        print(f"[Regime {r}] RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}")

    # ③ Leave-Regime-Out
    scores_lro: Dict[str, Dict[str, float]] = {}
    for hold in ["low", "mid", "high"]:
        te = regime.values == hold
        if te.sum() < 20:  # 小さすぎる場合スキップ
            continue
        tr = ~te
        m = CatBoostRegressor(**params)
        m.fit(X.iloc[tr], y[tr], sample_weight=w.iloc[tr].values, verbose=False)
        pred = m.predict(X.iloc[te])
        rmse, r2 = _weighted_scores(y[te], pred, w.iloc[te])
        scores_lro[hold] = {"rmse": rmse, "r2": r2}
    for r, m in scores_lro.items():
        print(f"[Leave-Regime-Out {r}] RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}")

    # 学習済みGlobalモデル（全データ）を保存
    final_model = CatBoostRegressor(**params)
    final_model.fit(X, y, sample_weight=w.values, verbose=False)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(args.out_model))
    print(f"[Model] Saved to {args.out_model}")

    # メトリクス保存（アプリで読む用）
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_metrics.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "catboost": {
                    "rmse": metrics_global["rmse"],
                    "r2": metrics_global["r2"],
                    "loss": args.loss,
                    "iterations": args.iterations,
                    "depth": args.depth,
                    "learning_rate": args.lr,
                    "target": args.target,
                },
                "regime_cv": scores_reg,
                "leave_regime_out": scores_lro,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Metrics] Saved to {args.out_metrics}")

    # OOF-SHAP TopK（自治体コードごと）
    topk = args.topk
    top_rows = []
    if not shap_oof.empty:
        for idx in shap_oof.index:
            row = shap_oof.loc[idx]
            top = row.abs().sort_values(ascending=False).head(topk)
            items = [f"{feat}:{row[feat]:+.3f}" for feat in top.index]
            top_rows.append(
                {
                    "市区町村コード": _prep_code(dfm.loc[idx, "市区町村コード"]),
                    "TopK_SHAP": " | ".join(items),
                }
            )
        shap_df = pd.DataFrame(top_rows).drop_duplicates("市区町村コード")
        args.out_shap.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(args.out_shap, index=False)
        print(f"[SHAP] Top{topk} saved to {args.out_shap}")
    else:
        print("[SHAP] OOF shap matrix is empty (skipped).")


if __name__ == "__main__":
    main()
