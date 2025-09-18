"""空き家率の回帰モデリングスクリプト。

`data/processed/features_master__wide__v1.csv` から生成された特徴量を用いて、
2018年から2023年にかけての空き家率の変化を説明・予測するために、CatBoostとXGBoostの残差モデルを学習します。
ドメイン特徴量（人口動態、アクセシビリティ、地価、気候）とそれらの政策に関連する交互作用を設計し、
高い共線性を持つ特徴を除去、ベースラインモデルからの残差改善に対して決定木モデルを適合させ、クロスバリデーションのフォールド全体で
CatBoostのSHAP値を集計します。

usage
-------------
    python notebook/vacancy_rate_regression.py \
        --data-path data/processed/features_master__wide__v1.csv \
        --shap-output data/processed/catboost_mean_abs_shap.csv

これにより、フォールドごとのメトリクス、クロスバリデーションの平均メトリクス、および上位のグローバルSHAP重要度が出力されます。
すべての特徴量のSHAP重要度は、下流の分析のために追加で保存され、平均メトリクスはJSONファイルに書き込みます。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

DATA_PATH = Path("data/processed/features_master__wide__v1.csv")
SHAP_OUTPUT_PATH = Path("data/processed/catboost_mean_abs_shap.csv")
METRICS_OUTPUT_PATH = Path("data/processed/model_metrics.json")
RANDOM_STATE = 42

DEFAULT_MAX_MISSING_RATIO = 0.4
DEFAULT_MIN_UNIQUE_VALUES = 5
DEFAULT_CORR_THRESHOLD = 0.9
DEFAULT_VIF_THRESHOLD = 8.0


def load_data(path: Path) -> pd.DataFrame:
    """特徴量特徴CSVの読み込み"""
    if not path.exists():
        raise FileNotFoundError(
            f"特徴量ファイルが {path} に見つかりません。実行前に生成してください。"
        )
    return pd.read_csv(path)


def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """``column`` の数値Seriesを返却。見つからない場合はNaNにフォールバックします。"""

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


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """率算を行う割り算関数"""
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def engineer_domain_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """解釈可能な特徴量を導出し、ターゲットとベースラインとともに返却。"""

    df = df.copy()

    vacancy_2018 = _get_numeric_series(df, "空き家率_2018")
    vacancy_2023 = _get_numeric_series(df, "空き家率_2023")
    # 目的変数： 空き家変化量
    target = vacancy_2023 - vacancy_2018

    features = pd.DataFrame(index=df.index)

    pop_2018 = _get_numeric_series(df, "2018_総人口")
    pop_2023 = _get_numeric_series(df, "2023_総人口")
    features["総人口_増減率"] = _safe_divide(pop_2023 - pop_2018, pop_2018)

    deaths_2018 = _get_numeric_series(df, "2018_死亡数")
    deaths_2023 = _get_numeric_series(df, "2023_死亡数")
    death_rate_2018 = _safe_divide(deaths_2018, pop_2018)
    death_rate_2023 = _safe_divide(deaths_2023, pop_2023)
    features["死亡数_per_pop_2018"] = death_rate_2018
    features["死亡数_per_pop_2023"] = death_rate_2023
    features["死亡数_per_pop_Δ"] = death_rate_2023 - death_rate_2018

    inm_2018 = _get_numeric_series(df, "2018_転入者数")
    inm_2023 = _get_numeric_series(df, "2023_転入者数")
    outm_2018 = _get_numeric_series(df, "2018_転出者数")
    outm_2023 = _get_numeric_series(df, "2023_転出者数")

    inflow_rate_2018 = _safe_divide(inm_2018, pop_2018)
    inflow_rate_2023 = _safe_divide(inm_2023, pop_2023)
    outflow_rate_2018 = _safe_divide(outm_2018, pop_2018)
    outflow_rate_2023 = _safe_divide(outm_2023, pop_2023)

    features["転入者数_per_pop_2018"] = inflow_rate_2018
    features["転入者数_per_pop_2023"] = inflow_rate_2023
    features["転入者数_per_pop_Δ"] = inflow_rate_2023 - inflow_rate_2018

    features["純移動率_2018"] = inflow_rate_2018 - outflow_rate_2018
    features["純移動率_2023"] = inflow_rate_2023 - outflow_rate_2023
    features["人口ターンオーバー_2018"] = inflow_rate_2018 + outflow_rate_2018
    features["人口ターンオーバー_2023"] = inflow_rate_2023 + outflow_rate_2023

    working_age_2018 = _get_numeric_series(df, "2018_生産年齢人口率[%]")
    working_age_2023 = _get_numeric_series(df, "2023_生産年齢人口率[%]")
    senior_rate_2018 = _get_numeric_series(df, "2018_高齢化率[%]")
    senior_rate_2023 = _get_numeric_series(df, "2023_高齢化率[%]")
    youth_rate_2018 = _get_numeric_series(df, "2018_年少人口率[%]")

    features["生産年齢人口率_Δ"] = working_age_2023 - working_age_2018
    features["高齢化率_Δ"] = senior_rate_2023 - senior_rate_2018

    school_density = _get_numeric_series(df, "学校密度[件/km²]")
    hospital_density = _get_numeric_series(df, "病院密度[件/km²]")
    station_density = _get_numeric_series(df, "駅密度[件/km²]")
    area_col = None
    for candidate in ("面積[km²]", "area_km2"):
        if candidate in df.columns:
            area_col = _get_numeric_series(df, candidate)
            break
    if area_col is None:
        area_col = pd.Series(np.nan, index=df.index, dtype="float64")

    log_school = np.log1p(school_density.clip(lower=0))
    log_hospital = np.log1p(hospital_density.clip(lower=0))
    log_station = np.log1p(station_density.clip(lower=0))
    log_area = np.log1p(area_col.clip(lower=0))

    features["log1p_学校密度"] = log_school
    features["log1p_病院密度"] = log_hospital
    features["log1p_駅密度"] = log_station
    features["log1p_面積"] = log_area

    land_log_diff = _get_numeric_series(df, "住宅地価_log差分")
    land_pct_change = _get_numeric_series(df, "住宅地価_増減率[%]")
    std_points_2018 = _get_numeric_series(df, "標準地点数_2018")
    std_points_2023 = _get_numeric_series(df, "標準地点数_2023")

    features["住宅地価_log差分"] = land_log_diff
    features["住宅地価_増減率"] = land_pct_change
    features["標準地点数_2018"] = std_points_2018
    features["標準地点数_2023"] = std_points_2023

    snow_depth = _get_numeric_series(df, "年最深積雪")
    precipitation = _get_numeric_series(df, "年降水量")

    features["高齢化率2023_x_log駅密度"] = senior_rate_2023 * log_station
    features["高齢化率2023_x_log病院密度"] = senior_rate_2023 * log_hospital
    features["人口増減率_x_地価log差分"] = features["総人口_増減率"] * land_log_diff
    features["最深積雪_x_log面積"] = snow_depth * log_area
    features["年降水量_x_log駅密度"] = precipitation * log_station
    features["log学校密度_x_年少人口率2018"] = log_school * youth_rate_2018

    # さらなるフィルタリングの前に、すべてNaNの特徴量を削除
    features = features.dropna(axis=1, how="all")

    mask = target.notna() & vacancy_2018.notna()
    features = features.loc[mask]
    target = target.loc[mask]
    baseline = vacancy_2018.loc[mask]

    return features, target, baseline


def prepare_features(
    df: pd.DataFrame,
    *,
    max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO,
    min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
):
    features, target, baseline = engineer_domain_features(df)

    valid_cols = [
        col
        for col in features.columns
        if features[col].isna().mean() <= max_missing_ratio
        and features[col].nunique(dropna=True) > min_unique_values
    ]
    features = features[valid_cols]

    features = remove_high_correlation_features(features, threshold=corr_threshold)
    features = remove_high_vif_features(features, threshold=vif_threshold)

    if features.empty:
        raise ValueError(
            "前処理後に使用可能な特徴量が残っていません。しきい値を調整してください。"
        )
    # 残った特徴量を出力
    print("\n--- 最終的に使用される特徴量 ---")
    for feature in features.columns:
        print(f"- {feature}")
    print("-" * 30)

    return features, target, baseline


def remove_high_correlation_features(
    feat_df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """絶対ペアワイズ相関が ``threshold`` を超える列を削除します。

    Parameters
    ----------
    feat_df:
        Input feature matrix. It is *not* modified in place.
    threshold:
        特徴量の任意のペア間で許容される最大の絶対相関。

    Returns
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

    欠損値はVIF計算の目的でのみ平均値で補完され、
    元の ``feat_df`` は変更されません。
    """

    if feat_df.empty:
        return pd.Series(dtype=float)

    filled = feat_df.fillna(feat_df.mean(numeric_only=True))
    corr = filled.corr().to_numpy()

    # ``corr`` は特異行列になる可能性があるため、疑似逆行列を使用します。
    try:
        inv_corr = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr + np.eye(corr.shape[0]) * 1e-8)

    vif = pd.Series(np.diag(inv_corr), index=feat_df.columns)
    return vif


def remove_high_vif_features(
    feat_df: pd.DataFrame, threshold: float = 8.0
) -> pd.DataFrame:
    """分散拡大係数が ``threshold`` を超える列を繰り返し削除します。

    Parameters
    ----------
    feat_df:
        入力特徴量行列。この場で変更はされません。
    threshold:
        The VIF threshold; higher values indicate stronger multicollinearity.

    Returns
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
    y_train,
    X_valid=None,
    y_valid=None,
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

    eval_set = (
        Pool(X_valid, y_valid) if X_valid is not None and y_valid is not None else None
    )
    cat_model.fit(train_pool, eval_set=eval_set, verbose=False)

    preds = cat_model.predict(X_valid) if X_valid is not None else None

    return cat_model, preds


def train_xgboost(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
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
    X_valid_imp = imputer.transform(X_valid) if X_valid is not None else None

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

    eval_set = (
        [(X_valid_imp, y_valid)]
        if X_valid_imp is not None and y_valid is not None
        else None
    )
    xgb_model.fit(
        X_train_imp,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    preds = xgb_model.predict(X_valid_imp) if X_valid_imp is not None else None

    return xgb_model, imputer, preds


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
    mse = np.mean([m["mse"] for m in metrics]) if metrics else float("nan")  # type: ignore
    r2 = np.mean([m["r2"] for m in metrics]) if metrics else float("nan")  # type: ignore
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
    if mse_direction == "<=":  # noqa: PLR2004
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
        description="解釈可能な空き家率回帰モデルを学習します。"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the wide feature CSV.",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=SHAP_OUTPUT_PATH,
        help="平均化されたCatBoost SHAP重要度を保存する場所。",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_OUTPUT_PATH,
        help="平均化されたクロスバリデーションメトリクスをJSONとして保存する場所。",
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
        help="特徴量ごとの許容される最大欠損値率。",
    )
    parser.add_argument(
        "--min-unique-values",
        type=int,
        default=DEFAULT_MIN_UNIQUE_VALUES,
        help="特徴量ごとに必要な最小ユニーク値数。",
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
        help="特徴量を削除する前の、許容される最大分散拡大係数。",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.5,
        help="平均二乗誤差メトリクスの目標しきい値。",
    )
    parser.add_argument(
        "--mse-direction",
        choices=["<=", ">="],
        default="<=",
        help="メトリクスがしきい値以下か以上かでMSE目標が満たされるかどうか。",
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
        help="メトリクスがしきい値以上か以下かでR^2目標が満たされるかどうか。",
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
        help="CatBoostの早期停止前に改善がないイテレーション数。",
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
        help="XGBoostの最大木深度。",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="XGBoostの学習インスタンスのサブサンプル比率。",
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
        help="XGBoostで子に必要なインスタンス重みの最小合計。",
    )
    parser.add_argument(
        "--xgb-early-stopping",
        type=int,
        default=50,
        help="XGBoostの早期停止前に改善がないラウンド数。",
    )
    return parser.parse_args(args=args)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args)

    df = load_data(args.data_path)
    X, y, baseline = prepare_features(
        df,
        max_missing_ratio=args.max_missing_ratio,
        min_unique_values=args.min_unique_values,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )

    print("--- Dataset Overview ---")
    print(f"Total records: {len(df)} | Usable records: {len(y)}")
    print(f"Target mean: {y.mean():.3f} | Target std: {y.std():.3f}")
    print(f"Selected engineered features: {X.shape[1]}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)

    cat_metrics_list: list[dict[str, float]] = []
    xgb_metrics_list: list[dict[str, float]] = []
    shap_importances: list[pd.Series] = []

    print(f"\n--- Running {args.folds}-fold Cross-Validation ---")

    for fold, (train_index, valid_index) in enumerate(kf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{args.folds} ---")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        baseline_train = baseline.iloc[train_index]
        baseline_valid = baseline.iloc[valid_index]

        baseline_model = LinearRegression()
        baseline_model.fit(baseline_train.to_frame(), y_train)
        base_train_pred = baseline_model.predict(baseline_train.to_frame())
        base_valid_pred = baseline_model.predict(baseline_valid.to_frame())

        resid_train = y_train - base_train_pred
        resid_valid = y_valid - base_valid_pred

        cat_model, cat_resid_pred = train_catboost(
            X_train,
            resid_train,
            X_valid=X_valid,
            y_valid=resid_valid,
            iterations=args.catboost_iterations,
            depth=args.catboost_depth,
            learning_rate=args.catboost_learning_rate,
            od_wait=args.catboost_od_wait,
        )
        _, _, xgb_resid_pred = train_xgboost(
            X_train,
            resid_train,
            X_valid=X_valid,
            y_valid=resid_valid,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            early_stopping_rounds=args.xgb_early_stopping,
        )

        cat_pred = base_valid_pred + cat_resid_pred
        xgb_pred = base_valid_pred + xgb_resid_pred

        cat_metrics = {
            "mse": mean_squared_error(y_valid, cat_pred),
            "r2": r2_score(y_valid, cat_pred),
        }
        xgb_metrics = {
            "mse": mean_squared_error(y_valid, xgb_pred),
            "r2": r2_score(y_valid, xgb_pred),
        }

        cat_metrics_list.append(cat_metrics)
        xgb_metrics_list.append(xgb_metrics)

        shap_importance = compute_catboost_shap(cat_model, X_valid, resid_valid)
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
    print("\n--- Average Top SHAP (CatBoost) ---")
    print(avg_shap_importance)

    metrics_payload = {
        "catboost": {"mse": cat_mse, "r2": cat_r2},
        "xgboost": {"mse": xgb_mse, "r2": xgb_r2},
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
