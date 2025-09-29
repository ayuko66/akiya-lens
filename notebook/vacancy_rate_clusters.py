import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/features_master__wide__v1.csv")
df["空き家率_変化量"] = df["空き家率_2023"] - df["空き家率_2018"]


# 特徴量の選択
features_for_clustering = df[["空き家率_2018", "空き家率_2023", "空き家率_変化量"]]

# データのスケーリング
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled_array = scaler.fit_transform(features_for_clustering)
features_scaled = pd.DataFrame(
    features_scaled_array, columns=features_for_clustering.columns
)

# NaN値の処理と対応する行の削除
features_scaled_cleaned = features_scaled.dropna()
cleaned_indices = features_scaled_cleaned.index
df_cleaned = df.loc[cleaned_indices].copy()

# クラスタリングの実行 (最適なクラスター数は前回のエルボー法の結果から4と仮定)
from sklearn.cluster import KMeans

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(features_scaled_cleaned)

# クラスターラベルをdf_cleanedに追加
df_cleaned["cluster_label"] = kmeans.labels_

# クラスタリング結果を含むDataFrameをCSVに出力
df_cleaned.to_csv("data/processed/clustered_akiya_data.csv", index=False)

print(
    "クラスタリング結果を含むデータが data/processed/clustered_akiya_data.csv に出力されました。"
)
