"""空き家率回帰モデリングスクリプト。

このスクリプトは、``data/processed/features_master__wide__v1.csv`` から生成された
特徴量を使用して、2018年から2023年の空き家率の変化を予測するためにCatBoostと
XGBoostモデルを訓練します。このワークフローは解釈可能性を重視して構築されています。
具体的には、多重共線性の高い特徴量を削除し、欠損値をネイティブに扱えるツリーベースの
モデルを使用し、クロスバリデーションの各フォールドでCatBoostから得られるSHAP値を
集計します。

典型的な使用法
-------------
.. code-block:: bash

    python notebook/vacancy_rate_regression.py \\
        --data-path data/processed/features_master__wide__v1.csv \\
        --shap-output data/processed/catboost_mean_abs_shap.csv

これにより、フォールドごとのメトリクス、クロスバリデーションの平均メトリクス、および
グローバルなSHAP重要度の上位が表示されます。すべての特徴量のSHAP重要度は、下流の分析の
ために追加で保存され、平均メトリクスはJSONファイルに書き込まれるため、ノートブックや
ダッシュボードで追跡できます。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

DATA_PATH = Path("data/processed/features_master__wide__v1.csv")
SHAP_OUTPUT_PATH = Path("data/processed/catboost_mean_abs_shap.csv")
METRICS_OUTPUT_PATH = Path("data/processed/model_metrics.json")
RANDOM_STATE = 42

DEFAULT_DROP_COLS = {
    "市区町村コード",
    "市区町村名",
    "都道府県名",
    "空き家率_差分_5年_pt",
    "空き家率_2023",
    "空き家率_2018",
    "空き家_増加率_5年_%",
    "空き家率_2023_raw",
}

DEFAULT_MAX_MISSING_RATIO = 0.4
DEFAULT_MIN_UNIQUE_VALUES = 5
DEFAULT_CORR_THRESHOLD = 0.9
DEFAULT_VIF_THRESHOLD = 8.0


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"特徴量ファイルが {path} に見つかりません。実行前に生成してください。"
        )

    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame,
    *,
    drop_cols: set[str] | None = None,
    max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO,
    min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
):
    df = df.copy()

    target = df["空き家率_2023"] - df["空き家率_2018"]

    mask = target.notna()
    df = df.loc[mask]
    target = target.loc[mask]

    drop_cols = drop_cols or DEFAULT_DROP_COLS

    numeric_cols = [
        col
        for col in df.columns
        if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    feat_df = df[numeric_cols].copy()

    valid_cols = [
        col
        for col in feat_df.columns
        if feat_df[col].isna().mean() <= max_missing_ratio
        and feat_df[col].nunique(dropna=True) > min_unique_values
    ]
    feat_df = feat_df[valid_cols]

    feat_df = remove_high_correlation_features(feat_df, threshold=corr_threshold)
    feat_df = remove_high_vif_features(feat_df, threshold=vif_threshold)

    if feat_df.empty:
        raise ValueError(
            "No usable features remain after preprocessing. Adjust the thresholds."
        )

    return feat_df, target


def remove_high_correlation_features(
    feat_df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """絶対ペアワイズ相関が ``threshold`` を超える列を削除します。

    引数
    ----------
    feat_df : pd.DataFrame
        入力特徴量行列。この関数はインプレースで変更しません。
    threshold : float
        特徴量のペア間で許容される最大の絶対相関。

    戻り値
    -------
    pd.DataFrame
        相関フィルターを通過した特徴量のサブセットを含むデータフレーム。
    """

    if feat_df.empty:
        return feat_df

    corr_matrix = feat_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return feat_df.drop(columns=to_drop)


def _variance_inflation_factor(feat_df: pd.DataFrame) -> pd.Series:
    """各列の分散拡大係数（VIF）を計算します。

    欠損値はVIF計算の目的でのみ平均値で補完され、元の ``feat_df`` は変更されません。
    """
    if feat_df.empty:
        return pd.Series(dtype=float)

    filled = feat_df.fillna(feat_df.mean(numeric_only=True))
    corr = filled.corr().to_numpy()

    # ``corr`` can be singular, therefore we use the pseudo inverse.
    try:  # `corr` は特異行列になる可能性があるため、疑似逆行列を使用します。
        inv_corr = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr + np.eye(corr.shape[0]) * 1e-8)

    vif = pd.Series(np.diag(inv_corr), index=feat_df.columns)
    return vif


def remove_high_vif_features(
    feat_df: pd.DataFrame, threshold: float = 8.0
) -> pd.DataFrame:
    """分散拡大係数（VIF）が ``threshold`` を超える列を繰り返し削除します。

    引数
    ----------
    feat_df : pd.DataFrame
        入力特徴量行列。この関数はインプレースで変更しません。
    threshold : float
        VIFのしきい値。値が高いほど、より強い多重共線性を示します。

    戻り値
    -------
    pd.DataFrame
        VIFフィルターを通過した特徴量のサブセットを含むデータフレーム。
    """
    columns = list(feat_df.columns)
    reduced = feat_df.copy()

    while len(columns) > 1:
        vif = _variance_inflation_factor(reduced[columns])
        if vif.empty or vif.max() <= threshold:
            break

        worst_col = vif.idxmax()
        columns.remove(worst_col)

    return reduced[columns]


def train_catboost(
    X_train,
    X_valid,
    y_train,
    y_valid,
    *,
    iterations: int,
    depth: int,
    learning_rate: float,
    od_wait: int,
):
    from catboost import CatBoostRegressor, Pool

    cat_model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        loss_function="RMSE",
        od_type="Iter",
        od_wait=od_wait,
        verbose=False,
        allow_writing_files=False,
    )

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    cat_model.fit(train_pool, eval_set=valid_pool, verbose=False)

    preds = cat_model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)

    return cat_model, {"mse": mse, "r2": r2}


def train_xgboost(
    X_train,
    X_valid,
    y_train,
    y_valid,
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    min_child_weight: float,
    early_stopping_rounds: int,
):
    from xgboost import XGBRegressor

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    xgb_model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
    )

    xgb_model.fit(
        X_train_imp,
        y_train,
        eval_set=[(X_valid_imp, y_valid)],
        verbose=False,
    )

    preds = xgb_model.predict(X_valid_imp)
    mse = mean_squared_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)

    return xgb_model, imputer, {"mse": mse, "r2": r2}


def compute_catboost_shap(
    model, X_valid: pd.DataFrame, y_valid: pd.Series
) -> pd.Series:
    from catboost import Pool

    pool = Pool(X_valid, y_valid)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_values = shap_values[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X_valid.columns).sort_values(ascending=False)


def _summarise_metrics(metrics: list[dict[str, float]]) -> tuple[float, float]:
    mse = np.mean([m["mse"] for m in metrics]) if metrics else float("nan")
    r2 = np.mean([m["r2"] for m in metrics]) if metrics else float("nan")
    return mse, r2


def _report_goal_attainment(
    mse: float,
    r2: float,
    *,
    mse_threshold: float,
    r2_threshold: float,
    mse_direction: str,
    r2_direction: str,
) -> str:
    if mse_direction == "<=":
        meets_mse = mse <= mse_threshold
    else:
        meets_mse = mse >= mse_threshold

    if r2_direction == ">=":
        meets_r2 = r2 >= r2_threshold
    else:
        meets_r2 = r2 <= r2_threshold

    return (
        f"Goals -> MSE {'OK' if meets_mse else 'NOT OK'} ({mse_direction}{mse_threshold}), "
        f"R^2 {'OK' if meets_r2 else 'NOT OK'} ({r2_direction}{r2_threshold})"
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解釈可能な空き家率回帰モデルを訓練します。"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="ワイド形式の特徴量CSVへのパス。",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=SHAP_OUTPUT_PATH,
        help="平均化されたCatBoostのSHAP重要度を保存する場所。",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_OUTPUT_PATH,
        help="平均化されたクロスバリデーションのメトリクスをJSONとして保存する場所。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="クロスバリデーションのフォールド数。",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=DEFAULT_MAX_MISSING_RATIO,
        help="特徴量ごとに許容される欠損値の最大比率。",
    )
    parser.add_argument(
        "--min-unique-values",
        type=int,
        default=DEFAULT_MIN_UNIQUE_VALUES,
        help="特徴量ごとに必要なユニーク値の最小数。",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=DEFAULT_CORR_THRESHOLD,
        help="特徴量を削除する前の、許容される最大絶対ペアワイズ相関。",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=DEFAULT_VIF_THRESHOLD,
        help="特徴量を削除する前の、許容される最大分散拡大係数（VIF）。",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.5,
        help="平均二乗誤差（MSE）メトリクスの目標しきい値。",
    )
    parser.add_argument(
        "--mse-direction",
        choices=["<=", ">="],
        default="<=",
        help="メトリクスがしきい値以下（<=）または以上（>=）の場合にMSEの目標が達成されるかどうか。",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.5,
        help="R^2メトリクスの目標しきい値。",
    )
    parser.add_argument(
        "--r2-direction",
        choices=[">=", "<="],
        default=">=",
        help="メトリクスがしきい値以上（>=）または以下（<=）の場合にR^2の目標が達成されるかどうか。",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=2000,
        help="CatBoostのブースティングイテレーション数。",
    )
    parser.add_argument(
        "--catboost-depth",
        type=int,
        default=6,
        help="CatBoostの木の深さ。",
    )
    parser.add_argument(
        "--catboost-learning-rate",
        type=float,
        default=0.05,
        help="CatBoostの学習率。",
    )
    parser.add_argument(
        "--catboost-od-wait",
        type=int,
        default=50,
        help="CatBoostの早期停止前に改善が見られないイテレーション数。",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=2000,
        help="XGBoostのブースティングラウンド数。",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.03,
        help="XGBoostの学習率。",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=4,
        help="XGBoostの木の最大深度。",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="XGBoostの訓練インスタンスのサブサンプル比率。",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.8,
        help="XGBoostで各木を構築する際の列のサブサンプル比率。",
    )
    parser.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=1.0,
        help="XGBoostのL2正則化項。",
    )
    parser.add_argument(
        "--xgb-min-child-weight",
        type=float,
        default=1.0,
        help="XGBoostの子ノードに必要なインスタンスの重みの最小合計。",
    )
    parser.add_argument(
        "--xgb-early-stopping",
        type=int,
        default=50,
        help="XGBoostの早期停止前に改善が見られないラウンド数。",
    )
    return parser.parse_args(args=args)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args)

    df = load_data(args.data_path)
    X, y = prepare_features(
        df,
        max_missing_ratio=args.max_missing_ratio,
        min_unique_values=args.min_unique_values,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )

    print("--- データセット概要 ---")
    print(f"総レコード数: {len(df)}")
    print(f"目的変数の平均: {y.mean():.3f} | 目的変数の標準偏差: {y.std():.3f}")
    print(f"選択された特徴量数: {X.shape[1]} / {df.shape[1]}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)

    cat_metrics_list: list[dict[str, float]] = []
    xgb_metrics_list: list[dict[str, float]] = []
    shap_importances: list[pd.Series] = []

    print(f"\n--- {args.folds}分割交差検証を実行 ---")

    for fold, (train_index, valid_index) in enumerate(kf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{args.folds} ---")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        cat_model, cat_metrics = train_catboost(
            X_train,
            X_valid,
            y_train,
            y_valid,
            iterations=args.catboost_iterations,
            depth=args.catboost_depth,
            learning_rate=args.catboost_learning_rate,
            od_wait=args.catboost_od_wait,
        )
        _, _, xgb_metrics = train_xgboost(
            X_train,
            X_valid,
            y_train,
            y_valid,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            early_stopping_rounds=args.xgb_early_stopping,
        )

        cat_metrics_list.append(cat_metrics)
        xgb_metrics_list.append(xgb_metrics)

        shap_importance = compute_catboost_shap(cat_model, X_valid, y_valid)
        shap_importances.append(shap_importance)

        print(
            f"CatBoost - MSE: {cat_metrics['mse']:.3f} | R^2: {cat_metrics['r2']:.3f}"
        )
        print(
            f"XGBoost  - MSE: {xgb_metrics['mse']:.3f} | R^2: {xgb_metrics['r2']:.3f}"
        )

    cat_mse, cat_r2 = _summarise_metrics(cat_metrics_list)
    xgb_mse, xgb_r2 = _summarise_metrics(xgb_metrics_list)

    shap_df = pd.concat(shap_importances, axis=1)
    shap_df.columns = [f"fold_{i}" for i in range(1, len(shap_importances) + 1)]
    avg_shap_importance = shap_df.mean(axis=1).sort_values(ascending=False)

    args.shap_output.parent.mkdir(parents=True, exist_ok=True)
    avg_shap_importance.to_csv(
        args.shap_output, encoding="utf-8-sig", header=["mean_abs_shap"]
    )

    print("\n--- 交差検証の平均結果 ---")
    print("--- CatBoost ---")
    print(f"Average MSE: {cat_mse:.3f} | Average R^2: {cat_r2:.3f}")
    goal_kwargs = {
        "mse_threshold": args.mse_threshold,
        "r2_threshold": args.r2_threshold,
        "mse_direction": args.mse_direction,
        "r2_direction": args.r2_direction,
    }
    print(_report_goal_attainment(cat_mse, cat_r2, **goal_kwargs))
    print("--- XGBoost ---")
    print(f"Average MSE: {xgb_mse:.3f} | Average R^2: {xgb_r2:.3f}")
    print(_report_goal_attainment(xgb_mse, xgb_r2, **goal_kwargs))
    print("\n--- SHAP重要度の平均 (上位, CatBoost) ---")
    print(avg_shap_importance.head(10))

    metrics_payload = {
        "catboost": {"mse": cat_mse, "r2": cat_r2},
        "xgboost": {"mse": xgb_mse, "r2": xgb_r2},
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
