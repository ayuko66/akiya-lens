"""Shared utilities for the vacancy rate diff CatBoost model."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover - graceful degradation
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore


def get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return the column coerced to numeric, preserving index."""

    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("‰", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_model_feature_matrix(
    df: pd.DataFrame, feature_names: Iterable[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract features for CatBoost predictions, returning missing columns."""

    feature_list = list(feature_names)
    matrix = pd.DataFrame(index=df.index)
    missing: List[str] = []

    for name in feature_list:
        if name in df.columns:
            series = df[name]
            if pd.api.types.is_numeric_dtype(series):
                matrix[name] = series
            else:
                numeric = (
                    series.astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("‰", "", regex=False)
                )
                numeric = pd.to_numeric(numeric, errors="coerce")
                if numeric.notna().sum() >= series.notna().sum() * 0.5:
                    matrix[name] = numeric
                else:
                    matrix[name] = series
        else:
            missing.append(name)
            matrix[name] = pd.Series(np.nan, index=df.index)

    return matrix[feature_list], missing


def enrich_diff_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the derived features used during model training."""

    augmented = df.copy()

    diff_specs = [
        ("住宅地価_log中央値_変化量", "住宅地価_log中央値_2023", "住宅地価_log中央値_2018"),
        ("Δ出生率", "2023_出生率[‰]", "2018_出生率[‰]"),
        ("Δ死亡率", "2023_死亡率[‰]", "2018_死亡率[‰]"),
        ("Δ年少人口率", "2023_年少人口率[%]", "2018_年少人口率[%]"),
        ("Δ高齢化率", "2023_高齢化率[%]", "2018_高齢化率[%]"),
        ("Δ生産年齢人口率", "2023_生産年齢人口率[%]", "2018_生産年齢人口率[%]"),
        ("Δ転入超過率", "2023_転入超過率[‰]", "2018_転入超過率[‰]"),
    ]

    for new_col, col_recent, col_base in diff_specs:
        if col_recent in augmented.columns and col_base in augmented.columns:
            augmented[new_col] = get_numeric_series(augmented, col_recent) - get_numeric_series(
                augmented, col_base
            )

    if "過疎地域市町村" in augmented.columns:
        dummies = pd.get_dummies(
            augmented["過疎地域市町村"], prefix="過疎地域市町村"
        )
        for col in dummies.columns:
            augmented[col] = dummies[col]

    # Align with training: add missing-value indicator columns and fill NA with 0
    for col in list(augmented.columns):
        if augmented[col].isna().any():
            augmented[f"{col}_is_na"] = augmented[col].isna().astype(int)

    augmented = augmented.fillna(0)

    return augmented


def fit_baseline(baseline: pd.Series, target: pd.Series) -> tuple[float, float]:
    """Fit a simple linear baseline between baseline and target deltas."""

    valid_mask = baseline.notna() & target.notna()
    if valid_mask.sum() < 2:
        return 0.0, 0.0
    x = baseline.loc[valid_mask].to_numpy(dtype=float)
    y = target.loc[valid_mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_predictions(
    df: pd.DataFrame,
    model: Optional[CatBoostRegressor],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """Return enriched dataframe with model predictions and residual diagnostics."""

    messages: List[str] = []

    vac18 = get_numeric_series(df, "空き家率_2018")
    vac23 = get_numeric_series(df, "空き家率_2023")
    delta_observed = vac23 - vac18

    slope, intercept = fit_baseline(vac18, delta_observed)
    base_pred = vac18 * slope + intercept

    feature_matrix: Optional[pd.DataFrame] = None
    resid_pred = pd.Series(0.0, index=df.index, dtype=float)

    if model is not None and hasattr(model, "feature_names_"):
        feature_names = list(model.feature_names_)
        if feature_names:
            augmented_df = enrich_diff_model_features(df)
            feature_matrix, missing = build_model_feature_matrix(
                augmented_df, feature_names
            )
            if missing:
                messages.append(
                    "学習済みモデルで使用した列が一部見つかりませんでした: "
                    + ", ".join(missing[:5])
                    + ("..." if len(missing) > 5 else "")
                )
            try:
                resid_pred = pd.Series(
                    model.predict(feature_matrix),
                    index=feature_matrix.index,
                    dtype=float,
                )
            except Exception as exc:
                messages.append(f"CatBoost予測に失敗しました: {exc}")
                resid_pred = pd.Series(0.0, index=df.index, dtype=float)
                feature_matrix = None
        else:
            messages.append("CatBoostモデルに特徴量名が含まれていません。")
    else:
        messages.append("CatBoostモデルが読み込めなかったため、残差補正なしで表示します。")

    pred_delta = base_pred + resid_pred
    pred_vacancy_2023 = vac18 + pred_delta
    residual_gap = delta_observed - pred_delta

    enriched = df.copy()
    enriched["Δ(23-18)"] = delta_observed
    enriched["baseline_pred_delta"] = base_pred
    enriched["residual_model_pred"] = resid_pred
    enriched["pred_delta"] = pred_delta
    enriched["pred_空き家率_2023"] = pred_vacancy_2023
    enriched["残差(実測-予測)"] = residual_gap

    return enriched, feature_matrix, messages


def compute_shap_topk(
    model: Optional[CatBoostRegressor],
    features: Optional[pd.DataFrame],
    k: int = 3,
    codes: Optional[pd.Series] = None,
    exclude_centroid: bool = True,
) -> Tuple[dict[str, List[str]], Optional[str]]:
    """Compute SHAP top-K contributions per sample."""

    if model is None or features is None or features.empty or Pool is None:
        return {}, None

    try:
        pool = Pool(features)
        shap_values = model.get_feature_importance(pool, type="ShapValues")
    except Exception as exc:
        return {}, f"SHAP値の計算に失敗しました: {exc}"

    shap_matrix = pd.DataFrame(
        shap_values[:, :-1], index=features.index, columns=features.columns
    )

    top_lookup: dict[str, List[str]] = {}
    codes_prepared: Optional[pd.Series] = None
    if codes is not None:
        codes_prepared = codes.astype(str).str.zfill(5)

    centroid_cols = {"centroid_lat_std", "centroid_lon_std"} if exclude_centroid else set()
    exclude_suffixes = ("_is_na",)

    for idx, row in shap_matrix.iterrows():
        if codes_prepared is not None and idx in codes_prepared.index:
            key = codes_prepared.loc[idx]
        else:
            key = str(idx)
        sorted_feats = row.abs().sort_values(ascending=False)
        top_candidates = sorted_feats.head(max(k, 5))
        filtered = [
            feat
            for feat in top_candidates.index
            if feat not in centroid_cols
            and not any(feat.endswith(suffix) for suffix in exclude_suffixes)
        ]
        if len(filtered) >= k:
            selected = filtered[:k]
        else:
            selected = filtered + [
                feat
                for feat in top_candidates.index
                if feat not in filtered
            ][: k - len(filtered)]

        top_lookup[key] = [f"{feat}: {row[feat]:+.3f}" for feat in selected]

    return top_lookup, None
