import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor, Pool, cv
import numpy as np

df = pd.read_csv("data/processed/features_master__wide__v1.csv")

df_feature = df.copy()
df_feature.set_index("市区町村コード")
df_feature = df_feature[
    [
        "空き家率_2018",
        "空き家率_2023",
        "2018_出生率[‰]",
        "2018_年少人口率[%]",
        "2018_死亡率[‰]",
        "2018_生産年齢人口率[%]",
        "2018_転入超過率[‰]",
        "2018_高齢化率[%]",
        "2023_出生率[‰]",
        "2023_年少人口率[%]",
        "2023_死亡率[‰]",
        "2023_生産年齢人口率[%]",
        "2023_転入超過率[‰]",
        "2023_高齢化率[%]",
        "スーパー密度[件/km²]",
        "2023年総人口あたりのスーパー密度",
        "学校密度[件/km²]",
        "2023年総人口あたりの学校密度",
        "病院密度[件/km²]",
        "2023年総人口あたりの病院密度",
        "駅密度[件/km²]",
        "2023年総人口あたりの駅密度",
        "住宅地価_log中央値_2018",
        "住宅地価_log中央値_2023",
        "平均気温",
        "年最深積雪",
        "年降水量",
        "最低気温",
        "過疎地域市町村",
    ]
]

df_feature = pd.get_dummies(df_feature, columns=["過疎地域市町村"])

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Define features (X_full) and target (y_full)
# Exclude '空き家率_2023' from features
X_full = df_feature.drop("空き家率_2023", axis=1)
y_full = df_feature["空き家率_2023"]

params_full = dict(
    iterations=1000,  # 学習イテレーション数
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",  # 最適化はRMSE
    eval_metric="R2",  # 主表示をR²にする
    custom_metric=["RMSE"],  # RMSEもログに出す
    random_seed=42,
    verbose=False,
)
params_full["eval_metric"] = (
    "RMSE"  # Use RMSE as the primary evaluation metric for this model
)
params_full["custom_metric"] = ["RMSE", "R2"]  # Add R2 to custom_metric


# Perform cross-validation
cv_full = cv(
    Pool(X_full, y_full),  # Features and target
    params_full,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    verbose=False,  # Suppress verbose output during CV
)

# Extract and print the mean RMSE and R2 from the cross-validation results
rmse_full_cv = cv_full["test-RMSE-mean"].iloc[-1]
r2_full_cv = cv_full["test-R2-mean"].iloc[-1]

print(f"Full Model Cross-Validation RMSE (5-fold mean): {rmse_full_cv:.3f}")
print(f"Full Model Cross-Validation R² (5-fold mean): {r2_full_cv:.3f}")

# cvで出したパラメータを使ってモデルを作成
from catboost import CatBoostRegressor, Pool

final_inference_model = CatBoostRegressor(**params_full)
final_inference_model.fit(X_full, y_full, verbose=False)  # Train without verbose output

# 残差をCatboost解析
from sklearn.linear_model import LinearRegression


# 1) 2018/2023 の空き家率が入ってる行だけに絞る（欠損落とし）
cols_needed = ["空き家率_2018", "空き家率_2023"]
df_bl = df.dropna(subset=cols_needed).copy()

# 2) 単回帰ベースライン: 2023 ~ 2018
X_base = df_bl[["空き家率_2018"]].values
y_base = df_bl["空き家率_2023"].values

lr = LinearRegression()
lr.fit(X_base, y_base)

print(f"回帰式: 2023空き家率 ≈ {lr.coef_[0]:.4f} * 2018空き家率 + {lr.intercept_:.4f}")
print(f"R²(単回帰): {lr.score(X_base, y_base):.3f}")

# 3) 残差の算出 (residual: 単回帰モデルでは説明できない空き家率の変化分)
df_bl["baseline_pred"] = lr.predict(X_base)  # 予測値
df_bl["residual"] = df_bl["空き家率_2023"] - df_bl["baseline_pred"]  # 残差

X_total = df_feature.loc[
    df_bl.index,
    [
        "2018_出生率[‰]",
        "2018_年少人口率[%]",
        "2018_死亡率[‰]",
        "2018_生産年齢人口率[%]",
        "2018_転入超過率[‰]",
        "2018_高齢化率[%]",
        "2023_出生率[‰]",
        "2023_年少人口率[%]",
        "2023_死亡率[‰]",
        "2023_生産年齢人口率[%]",
        "2023_転入超過率[‰]",
        "2023_高齢化率[%]",
        "スーパー密度[件/km²]",
        "2023年総人口あたりのスーパー密度",
        "学校密度[件/km²]",
        "2023年総人口あたりの学校密度",
        "病院密度[件/km²]",
        "2023年総人口あたりの病院密度",
        "駅密度[件/km²]",
        "2023年総人口あたりの駅密度",
        "住宅地価_log中央値_2018",
        "住宅地価_log中央値_2023",
        "平均気温",
        "年最深積雪",
        "年降水量",
        "最低気温",
        "過疎地域市町村_A",
        "過疎地域市町村_B",
        "過疎地域市町村_C",  # Use one-hot encoded columns
    ],
]

# 残差を目的変数としてCatBoost
y_resid = df_bl["residual"]


params = dict(
    iterations=1000,  # 学習イテレーション数
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",  # 最適化はRMSE
    eval_metric="R2",  # 主表示をR²にする
    custom_metric=["RMSE"],  # RMSEもログに出す
    random_seed=42,
    verbose=False,
)

# 交差検証
cv_resid = cv(
    Pool(X_total, y_resid),  # 特徴量とターゲット
    params,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
)

print(cv_resid.columns)  # 出力されるカラムを確認

r2 = cv_resid["test-R2-mean"].iloc[-1]
rmse = cv_resid["test-RMSE-mean"].iloc[-1]  # すでに√済みなのでそのままRMSE

print(f"残差モデル R² (5-fold mean): {r2:.3f}")
print(f"残差モデル RMSE (5-fold mean): {rmse:.3f}")

residual_model = CatBoostRegressor(**params)
residual_model.fit(X_total, y_resid, verbose=False)  # Train without verbose output

# Define the path to save the model
model_path = "models/vacancy_catboost_residual_model.cbm"

# Save the model
residual_model.save_model(model_path)

print(f"Residual model saved to {model_path}")
