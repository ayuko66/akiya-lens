import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

df = pd.read_csv("data/processed/features_master__wide__v1.csv")

# --- 必要列の存在チェック（例: 住宅総数は列名合わせてね） ---
# df['住宅総数_2018'], df['住宅総数_2023'] が無ければ重みは後でスキップ

# ========== 1) ベースライン OOF ==========
mask = df[["空き家率_2018", "空き家率_2023"]].notna().all(axis=1)
d = df.loc[mask].reset_index(drop=True).copy()
X_base = d[["空き家率_2018"]].values
y23 = d["空き家率_2023"].values

oof_pred = np.zeros(len(d))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in kf.split(X_base):
    lr = LinearRegression().fit(X_base[tr], y23[tr])
    oof_pred[va] = lr.predict(X_base[va])

d["residual_oof"] = y23 - oof_pred  # OOF残差（RMSE評価ならこれを使う）

# ========== 2) 分散安定化（Δlogit）+ 重み ==========
eps = 1e-6
p18 = (d["空き家率_2018"] / 100).clip(eps, 1 - eps)
p23 = (d["空き家率_2023"] / 100).clip(eps, 1 - eps)
logit = lambda p: np.log(p / (1 - p))
d["d_logit"] = logit(p23) - logit(p18)  # 目的変数候補①

# 二項近似によるVar(Δp) ≈ p23(1-p23)/N23 + p18(1-p18)/N18
if {"住宅総数_2018", "住宅総数_2023"}.issubset(d.columns):
    N18 = d["住宅総数_2018"].clip(lower=10)  # クリップで発散防止
    N23 = d["住宅総数_2023"].clip(lower=10)
    var_dp = p23 * (1 - p23) / N23 + p18 * (1 - p18) / N18
    w = 1 / np.clip(var_dp, np.quantile(var_dp, 0.05), np.quantile(var_dp, 0.95))
else:
    w = np.ones(len(d))

# ========== 3) 説明変数のセットアップ ==========
# ※ 2023年の説明変数は「将来予測」にはリーキーなので、
#    解析用(関連要因の把握)と予測用(2018＆地理/気候/OSM等)は分ける
feature_cols = [
    # 2018/固定の説明変数群
    "2018_出生率[‰]",
    "2018_年少人口率[%]",
    "2018_死亡率[‰]",
    "2018_生産年齢人口率[%]",
    "2018_転入超過率[‰]",
    "2018_高齢化率[%]",
    "スーパー密度[件/km²]",
    "学校密度[件/km²]",
    "病院密度[件/km²]",
    "駅密度[件/km²]",
    "住宅地価_log中央値_2018",
    "平均気温",
    "年最深積雪",
    "年降水量",
    "最低気温",
]
# 解析モードでは↓を追加（ただし将来予測用途では外す）
feature_cols_explan = feature_cols + [
    "2023年総人口あたりのスーパー密度",
    "2023年総人口あたりの学校密度",
    "2023年総人口あたりの病院密度",
    "2023年総人口あたりの駅密度",
    "住宅地価_log中央値_2023",
    "2023_出生率[‰]",
    "2023_年少人口率[%]",
    "2023_死亡率[‰]",
    "2023_生産年齢人口率[%]",
    "2023_転入超過率[‰]",
    "2023_高齢化率[%]",
]

X = d[[c for c in feature_cols_explan if c in d.columns]].copy()

# 欠損フラグを追加（OSM_missingなど）
for c in X.columns:
    if X[c].isna().any():
        X[f"{c}__isnan"] = X[c].isna().astype(int)
X = X.fillna(X.median(numeric_only=True))

# ========== 4) レジーム分割 ==========
# --- レジーム作成（重複境界を自動で落とすと安全） ---
d["regime"] = pd.qcut(
    d["空き家率_2018"], q=3, labels=["low", "mid", "high"], duplicates="drop"
)

cat_params = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
)

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def cv_fit_predict(X_, y_, w_, group=None, n_splits_k=5, n_splits_g=3):
    """部分集合ごとに安全にOOFを返す。位置インデックスで一貫させる。"""
    # resetで 0..n-1 の RangeIndex にそろえる
    X_ = X_.reset_index(drop=True)
    y_ = y_.reset_index(drop=True)
    w_ = pd.Series(w_).reset_index(drop=True)

    assert (
        len(X_) == len(y_) == len(w_)
    ), f"len mismatch: X={len(X_)}, y={len(y_)}, w={len(w_)}"

    n = len(y_)
    if group is None:
        k = min(max(2, n_splits_k), n)  # サンプルが少ないときに落ちないように
        splitter = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = splitter.split(X_)
    else:
        g = pd.Series(group).reset_index(drop=True)
        k = min(max(2, n_splits_g), g.nunique(), n)  # GroupKFoldは分割数<=群数
        splitter = GroupKFold(n_splits=k)
        splits = splitter.split(X_, y_, groups=g)

    oof = np.zeros(n)
    for tr, va in splits:
        model = CatBoostRegressor(**cat_params)
        model.fit(
            X_.iloc[tr],
            y_.iloc[tr],
            sample_weight=w_.iloc[tr],
            eval_set=(X_.iloc[va], y_.iloc[va]),
            verbose=False,
        )
        oof[va] = model.predict(X_.iloc[va])
    return oof


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  # ここはMSE（squared引数は使わない）
    return dict(
        R2=r2_score(y_true, y_pred),
        RMSE=np.sqrt(mse),  # 自前でRMSEに変換
    )


# ====== A) グローバル1本（Δlogitを目的） ======
y = d["d_logit"]
w_series = pd.Series(w, index=d.index)  # 既に作ってある重み

oof_global = cv_fit_predict(X, y, w_series)
print("Global:", metrics(y, oof_global))

# ====== B) レジーム別（3本） ======
oof_regime = np.zeros(len(d))
for r in d["regime"].unique():
    ridx = np.where(d["regime"].to_numpy() == r)[0]  # 0..n-1 の位置配列
    Xr = X.loc[ridx]
    yr = y.loc[ridx]
    wr = w_series.loc[ridx]
    oof_regime[ridx] = cv_fit_predict(Xr, yr, wr)  # 部分集合を渡すのがミソ

print("Regime:", metrics(y, oof_regime))

# ====== C) Leave-Regime-Out（外挿耐性チェック） ======
oof_lro = cv_fit_predict(X, y, w_series, group=d["regime"])
print("Leave-Regime-Out:", metrics(y, oof_lro))

print("regime counts:\n", d["regime"].value_counts(dropna=False))
print("Index max:", d.index.max(), "len(d)-1:", len(d) - 1)

print("Shapes:", X.shape, len(d), len(y))
print("Any NaN in y?", pd.isna(y).any())
print("Weight summary:", pd.Series(w).describe())
print("Regimes:", d["regime"].unique())
