# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

from typing import Any, Optional

import japanize_matplotlib as jmp
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv  # ← ここが重要！
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.diff_model_utils import compute_predictions, compute_shap_topk

# データ読み込み
df = pd.read_csv("data/processed/features_master__wide__v1.csv")

pd.set_option("display.max_rows", None)
# print(df.columns.to_list())  # 必要なら列名確認

# 派生特徴
df["住宅地価_log中央値_変化量"] = (
    df["住宅地価_log中央値_2023"] - df["住宅地価_log中央値_2018"]
)
df["Δ出生率"] = df["2023_出生率[‰]"] - df["2018_出生率[‰]"]
df["Δ死亡率"] = df["2023_死亡率[‰]"] - df["2018_死亡率[‰]"]
df["Δ年少人口率"] = df["2023_年少人口率[%]"] - df["2018_年少人口率[%]"]
df["Δ高齢化率"] = df["2023_高齢化率[%]"] - df["2018_高齢化率[%]"]
df["Δ生産年齢人口率"] = df["2023_生産年齢人口率[%]"] - df["2018_生産年齢人口率[%]"]
df["Δ転入超過率"] = df["2023_転入超過率[‰]"] - df["2018_転入超過率[‰]"]

# 学習用テーブル
df_feature_diff = df.copy()
df_feature_diff.set_index("市区町村コード", inplace=True)  # ← in-placeに

use_cols = [
    "空き家率_2018",
    "空き家率_2023",
    "2018_出生率[‰]",
    "2018_年少人口率[%]",
    "2018_死亡率[‰]",
    "2018_生産年齢人口率[%]",
    "2018_転入超過率[‰]",
    "2018_高齢化率[%]",
    "Δ出生率",
    "Δ年少人口率",
    "Δ死亡率",
    "Δ生産年齢人口率",
    "Δ転入超過率",
    "Δ高齢化率",
    "2018年総人口あたりのスーパー密度",
    "2018年総人口あたりの学校密度",
    "2018年総人口あたりの病院密度",
    "2018年総人口あたりの駅密度",
    "住宅地価_log中央値_2018",
    "住宅地価_log中央値_変化量",
    "平均気温",
    "年最深積雪",
    "年降水量",
    "最低気温",
    "過疎地域市町村",
    "centroid_lat_std",
    "centroid_lon_std",
]
df_feature_diff = df_feature_diff[use_cols]

# カテゴリ→ダミー
df_feature_diff = pd.get_dummies(df_feature_diff, columns=["過疎地域市町村"])

# 欠損フラグ列を追加
for col in df_feature_diff.columns:
    if df_feature_diff[col].isna().any():
        flag_col = f"{col}_is_na"
        df_feature_diff[flag_col] = df_feature_diff[col].isna().astype(int)

# 欠損をゼロで補完（CatBoostは欠損も扱えるがフラグで補助）
df_feature_diff = df_feature_diff.fillna(0)

# 目的変数と特徴量
X_full_diff = df_feature_diff.drop(["空き家率_2018", "空き家率_2023"], axis=1)
y_full_diff = df_feature_diff["空き家率_2023"]

# チューニング共通設定
base_params = dict(
    loss_function="RMSE",  # 最適化
    eval_metric="RMSE",  # 主表示
    custom_metric=["RMSE", "R2"],  # これでR2もログに出る
    random_seed=42,
    verbose=False,
)

train_pool = Pool(X_full_diff, y_full_diff)


def objective(trial: optuna.Trial) -> float:
    params = base_params.copy()
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli"]
    )
    params.update(
        {
            "iterations": trial.suggest_int("iterations", 400, 1600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.6, 1.0),
            "bootstrap_type": bootstrap_type,
        }
    )

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 5.0
        )
    else:  # Bernoulli
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    cv_result = cv(
        train_pool,
        params,
        fold_count=5,
        shuffle=True,
        partition_random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )

    best_idx = cv_result["test-RMSE-mean"].idxmin()
    best_rmse = float(cv_result["test-RMSE-mean"].iloc[best_idx])
    trial.set_user_attr("best_iteration", int(best_idx) + 1)
    return best_rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = base_params.copy()
best_params.update(study.best_params)
best_iteration = study.best_trial.user_attrs.get(
    "best_iteration", best_params.get("iterations")
)
if best_iteration is not None:
    best_params["iterations"] = int(best_iteration)

print("Best RMSE:", study.best_value)
print("Best Params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# 最終学習
final_diff_model = CatBoostRegressor(**best_params)
final_diff_model.fit(train_pool, verbose=False)

# ベストパラメータでのCV結果も確認
cv_result_best = cv(
    train_pool,
    best_params,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    verbose=False,
    early_stopping_rounds=50,
)
rmse_full_cv_diff = float(cv_result_best["test-RMSE-mean"].min())
r2_full_cv_diff = float(
    cv_result_best["test-R2-mean"].iloc[cv_result_best["test-RMSE-mean"].idxmin()]
)

print(f"Tuned Model Cross-Validation RMSE (5-fold min mean): {rmse_full_cv_diff:.3f}")
print(f"Tuned Model Cross-Validation R² (corresponding mean): {r2_full_cv_diff:.3f}")

# SHAP（CatBoostはTreeベースなのでTreeExplainerが安定）
explainer_full_diff = shap.TreeExplainer(final_diff_model)
shap_values_full_diff = explainer_full_diff(X_full_diff)
shap.summary_plot(shap_values_full_diff.values, X_full_diff)  # .valuesを明示すると安全

final_diff_model.save_model("models/final_diff_model.cbm")

enriched_df, feature_matrix_cached, predict_messages = compute_predictions(
    df, final_diff_model
)

code_series = None
if feature_matrix_cached is not None and "市区町村コード" in df.columns:
    code_series = df.loc[feature_matrix_cached.index, "市区町村コード"]

shap_lookup, shap_message = compute_shap_topk(
    final_diff_model, feature_matrix_cached, codes=code_series
)

for msg in predict_messages:
    print(f"[predict] {msg}")
if shap_message:
    print(f"[shap] {shap_message}")


def _ensure_native(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _format_code(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    code_str = str(value)
    if code_str.endswith(".0"):
        code_str = code_str[:-2]
    code_str = code_str.strip()
    if not code_str:
        return None
    return code_str.zfill(5)


inspector_payload: dict[str, dict[str, Any]] = {}
if "市区町村コード" in enriched_df.columns:
    for _, row in enriched_df.iterrows():
        code = _format_code(row.get("市区町村コード"))
        if not code:
            continue

        entry = {
            "市区町村コード": code,
            "市区町村名": row.get("市区町村名"),
            "都道府県名": row.get("都道府県名"),
            "空き家率_2018": _ensure_native(row.get("空き家率_2018")),
            "空き家率_2023": _ensure_native(row.get("空き家率_2023")),
            "Δ(23-18)": _ensure_native(row.get("Δ(23-18)")),
            "pred_空き家率_2023": _ensure_native(row.get("pred_空き家率_2023")),
            "残差(実測-予測)": _ensure_native(row.get("残差(実測-予測)")),
            "baseline_pred_delta": _ensure_native(row.get("baseline_pred_delta")),
            "residual_model_pred": _ensure_native(row.get("residual_model_pred")),
            "pred_delta": _ensure_native(row.get("pred_delta")),
            "tooltip": {
                "akiya_name": row.get("市区町村名"),
                "akiya_vac18": _ensure_native(row.get("空き家率_2018")),
                "akiya_vac23": _ensure_native(row.get("空き家率_2023")),
                "akiya_delta": _ensure_native(row.get("Δ(23-18)")),
            },
            "shap_top3": shap_lookup.get(code, []),
        }

        inspector_payload[code] = entry

inspector_path = Path("data/processed/diff_model_inspector.json")
inspector_path.parent.mkdir(parents=True, exist_ok=True)
with inspector_path.open("w", encoding="utf-8") as handle:
    json.dump(inspector_payload, handle, ensure_ascii=False, indent=2)


metrics_payload = {
    "rmse": float(rmse_full_cv_diff),
    "mse": float(rmse_full_cv_diff**2),
    "r2": float(r2_full_cv_diff),
    "iterations": int(best_params.get("iterations", 0)),
    "depth": int(best_params.get("depth", 0)),
    "learning_rate": float(best_params.get("learning_rate", 0.0)),
    "bootstrap_type": best_params.get("bootstrap_type"),
    "l2_leaf_reg": _ensure_native(best_params.get("l2_leaf_reg")),
    "rsm": _ensure_native(best_params.get("rsm")),
}

if "bagging_temperature" in best_params:
    metrics_payload["bagging_temperature"] = float(best_params["bagging_temperature"])
if "subsample" in best_params:
    metrics_payload["subsample"] = float(best_params["subsample"])

metrics_path = Path("data/processed/model_metrics.json")
existing_metrics = {}
if metrics_path.exists():
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing_metrics = json.load(handle)
    except json.JSONDecodeError:
        existing_metrics = {}

existing_metrics["catboost"] = metrics_payload

with metrics_path.open("w", encoding="utf-8") as handle:
    json.dump(existing_metrics, handle, ensure_ascii=False, indent=2)
