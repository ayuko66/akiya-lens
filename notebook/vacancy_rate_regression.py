import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

DATA_PATH = Path("data/processed/features_master__wide__v1.csv")
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    if "空き家率_差分_5年_pt" in df.columns:
        target = df["空き家率_差分_5年_pt"]
    else:
        target = df["空き家率_2023"] - df["空き家率_2018"]

    mask = target.notna()
    df = df.loc[mask]
    target = target.loc[mask]

    drop_cols = {
        "市区町村コード",
        "市区町村名",
        "都道府県名",
        "空き家率_差分_5年_pt",
        "空き家率_2023",
        "空き家率_2018",
        "空き家_増加率_5年_%",
        "空き家率_2023_raw",
    }

    numeric_cols = [
        col
        for col in df.columns
        if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    feat_df = df[numeric_cols].copy()

    valid_cols = [
        col
        for col in feat_df.columns
        if feat_df[col].isna().mean() <= 0.4 and feat_df[col].nunique(dropna=True) > 5
    ]
    feat_df = feat_df[valid_cols]

    corr_matrix = feat_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    feat_df = feat_df.drop(columns=to_drop)

    return feat_df, target


def train_catboost(X_train, X_valid, y_train, y_valid):
    cat_model = CatBoostRegressor(
        iterations=2000,
        depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        loss_function="RMSE",
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    cat_model.fit(train_pool, eval_set=valid_pool, verbose=False)

    preds = cat_model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)

    return cat_model, {"mse": mse, "r2": r2}


def train_xgboost(X_train, X_valid, y_train, y_valid):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
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
    model: CatBoostRegressor, X_valid: pd.DataFrame, y_valid: pd.Series
) -> pd.Series:
    pool = Pool(X_valid, y_valid)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_values = shap_values[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X_valid.columns).sort_values(ascending=False)


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cat_metrics_list = []
    xgb_metrics_list = []
    shap_importances = []

    print(f"--- Running 5-fold Cross-Validation ---")
    print(f"Features: {X.shape[1]} | Total samples: {len(X)}")

    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        cat_model, cat_metrics = train_catboost(X_train, X_valid, y_train, y_valid)
        _, _, xgb_metrics = train_xgboost(X_train, X_valid, y_train, y_valid)

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

    avg_cat_mse = np.mean([m["mse"] for m in cat_metrics_list])
    avg_cat_r2 = np.mean([m["r2"] for m in cat_metrics_list])
    avg_xgb_mse = np.mean([m["mse"] for m in xgb_metrics_list])
    avg_xgb_r2 = np.mean([m["r2"] for m in xgb_metrics_list])
    avg_shap_importance = (
        pd.concat(shap_importances).groupby(level=0).mean().sort_values(ascending=False)
    )

    print("\n--- Average Cross-Validation Results ---")
    print("--- CatBoost ---")
    print(f"Average MSE: {avg_cat_mse:.3f} | Average R^2: {avg_cat_r2:.3f}")
    print("--- XGBoost ---")
    print(f"Average MSE: {avg_xgb_mse:.3f} | Average R^2: {avg_xgb_r2:.3f}")
    print("\n--- Average Top SHAP (CatBoost) ---")
    print(avg_shap_importance.head(10))


if __name__ == "__main__":
    main()
