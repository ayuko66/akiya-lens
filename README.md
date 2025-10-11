# Akiya-Lens

空き家率の現況と増勢、予測残差をワンビューで確認できる自治体向けダッシュボードです。  
CatBoost を用いた残差学習モデルで 2023 年の空き家率を推定し、2018→2023 年の変化と組み合わせて 4 段階のリスク区分を提示します。

> **免責**: 本プロダクトは助言的ツールです。投資・行政・取引判断を保証するものではありません。

## テーマ
市町村や移住希望者向けに空き家の要因を可視化することにより、利活用施策の検討をサポートします。

## 背景
日本の空き家は令和5年時点で全国で約900万戸以上（住宅総数の13%以上）。特に人口減少や高齢化が進む地方で深刻化。
参考：[総務省統計局 令和5年住宅・土地統計調査](https://www.stat.go.jp/data/jyutaku/2023/bunseki.html)

### 主な背景要因
1.	少子高齢化・人口流出（都市部への移住）
2.	相続後の管理放棄（権利関係が複雑）
3.	住宅価格上昇（リフォーム・解体費用）
4.	利活用の需要不足(交通など生活インフラ、雇用の不足)
---
 > メモ：
私自身、将来的に山麓移住をして終の棲家を見つけたいと思っており、AI活用による地方地域の活性化と移住希望者の支援を目指しています。
本プロダクトは市区町村の空き家の要因を予測をすることで、利活用推進の手がかりに。
また、移住希望者のIターンにおける不安要素の解消と生産年齢人口の増加促進の一助となるツール。
---

## 主な機能
- **Streamlit ダッシュボード** (`app.py`): Folium マップ上で各自治体のリスク区分、空き家率、予測値、残差を可視化。自治体インスペクタで SHAP トップ要因も確認可能。
- **4 段階リスク分類**: 空き家率の水準 (P75 以上を高水準) と 2018→2023 年の増勢 (増・横ばい・減) を組み合わせた `(最優先)/(注意)/(警戒)/(低)` の 4 区分。
- **CatBoost モデル**: 機械学習により2023年空き家率を予測、`diff_model_inspector.json` を通じて事前計算済み SHAP/予測をキャッシュ可能。
- **再現性のあるデータパイプライン**: `data/processed/` 以下に前処理済みの特徴量・予測・メタ情報を管理。`scripts/etl/` や `scripts/models/` が再計算ロジックを保持。

---

## リポジトリ構成 (抜粋)
```
.
├── app.py                         # Streamlit UI 本体
├── data/
│   ├── processed/
│   │   ├── features_master__wide__v1.csv  # メイン特徴量テーブル
│   │   ├── diff_model_inspector.json      # 事前計算の SHAP / 予測キャッシュ (任意)
│   │   └── model_metrics.json             # モデル評価値
│   └── geojson/
│       └── municipalities_simplified.geojson
├── notebook/
│   ├── train_model_vacancy_rate_diff.py   # CatBoost 残差モデル学習スクリプト
│   ├── vacancy_rate_residual.ipynb        # モデリング検証
│   └── ...                                # EDA / 可視化ノート
├── scripts/
│   ├── models/
│   │   └── diff_model_utils.py            # 予測・SHAP 計算ユーティリティ
│   └── utils/                             # GeoJSON/OSM 等の補助ツール
├── models/
│   └── final_diff_model.cbm               # 学習済み CatBoost モデル (任意)
├── pyproject.toml
└── uv.lock
```

---

## セットアップ

### 1. Python 環境
Python 3.10 以上

> `geopandas` / `shapely` / `catboost` などネイティブ依存のあるライブラリを使用しています。必要に応じて `gdal` 等のシステム依存パッケージを事前にインストールしてください。

### 2. データ配置
`data/processed/features_master__wide__v1.csv` などの前処理済みデータを配置してください。  
この CSV には市区町村コード、空き家率、人口動態、POI 密度などの特徴量がまとめられています。

- 推論のみを行う場合は上記 CSV と `models/final_diff_model.cbm` があれば実行可能です。
- `diff_model_inspector.json` が存在する場合、Streamlit アプリはそのキャッシュを利用し、初回ロードを高速化します。

---

## ダッシュボードの実行

```bash
streamlit run app.py
```

主な UI 機能:
- **地図セレクタ**: 空き家率 (実測/予測)、残差、4 段階リスクで塗り分け。
- **自治体インスペクタ**: 選択した自治体の指標、予測値、残差、SHAP 上位要因 (最大 3 件) を表示。
- **テーブル表示**: 全自治体の最新指標をテーブル (エクスポート可) で確認。

---

## モデルの再学習

### モデル (CatBoost)
1. データ更新: `data/processed/features_master__wide__v1.csv` を最新化。
2. 学習: `notebook/train_model_vacancy_rate_diff.py` を実行 (スクリプトとしても利用可)。
   - Optuna でハイパーパラメータ探索を実施。
   - 学習後 `models/final_diff_model.cbm` を上書き保存。
3. 予測・SHAP 計算: `scripts/models/diff_model_utils.py` の `compute_predictions` / `compute_shap_topk` を利用。
   - `diff_model_inspector.json` を更新すると、Streamlit 側でキャッシュとして活用可能。

### 4 段階リスク分類
- `app.py` 内の `classify_risk` が P75 をしきい値に増勢 (Δ) と組み合わせてラベルを算出しています。
- `DEFAULT_TOLERANCE` (横ばい判定) は UI から変更可能。

---

## データセットと出典
- **空き家率 (目的変数)**: 総務省統計局 令和 5 年住宅・土地統計調査 (e-Stat)
- **POI 密度 (スーパー/学校/病院/駅など)**: OpenStreetMap Overpass API
- **地価**: 国土交通省 国土数値情報
- **気象**: 平均メッシュ気象データ
- **人口動態**: e-Stat, RESAS など
- 追加データは `scripts/etl/` 配下で加工しています。詳細仕様は `docs/` を参照してください。

### メモ：
- コンビニなどの生活インフラ(駅、スーパー、学校、病院)密度
  - OpenStreetMap Overpass API
    - [overpass API](https://overpass-api.de/)
    - [OpenStreetMap Overpass API](https://overpass-turbo.eu/)
    - [日本語 wiki](https://wiki.openstreetmap.org/wiki/JA:Overpass_API)
- 地価公示中央値
  - [国土交通省 国土数値情報](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-v3_1.html)
- 降水量、気温、最深積雪の気象要素
  - [平均メッシュデータ](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G02-2022.html)
- 人口動態
  - 人口数、転入数、転出数、出生数、死亡数、人口増加率、生産年齢人口割合、老年人口割合、など
  - [RESAS API](https://opendata.resas-portal.go.jp/docs/api/v1/index.html) ※2025年API終了
  - https://www.e-stat.go.jp/stat-search/files?page=1&stat_infid=000040207587
- 市区町村財務情報
  - [総務省 地方財政状況調査関係資料](https://www.soumu.go.jp/iken/kessan_jokyo_2.html)



---

## よくある質問
- **残差が NaN になる**  
  `diff_model_inspector.json` のキー綴りが誤っている場合があります (例: `"△実測-予測)"`)。正しいキー `"△(実測-予測)"` または `compute_predictions` が出力する `"残差(実測-予測)"` を使用してください。

---

## ライセンス・クレジット
- 本プロジェクトはオープンデータを活用し、オープンソースとして公開しています。利用時は各データ提供元のライセンスに従ってください。
- 作者: Ayuko Iwata  
  ターゲットは山麓移住希望者・自治体職員を想定しており、AI による地域課題解決の実例として開発しています。

---

## 今後の展望
- 対象地域を全国へ拡大し、空き家バンク / 施策効果との連動を検討。
- メッシュ粒度での推定や地形情報の追加。
- 生成 AI との連携による利活用提案の自動化。

---
### 類似サービス(参考)

- 自治体向け空き家予測
  - [将来住居・不動産流通予測AI　MiraiE.ai（ミラーエ）](https://www.microgeo.biz/jp/service/1165)
- 学術研究
  - [空き家マップ](https://webmap.sakura.ne.jp/wp/commentary/)
- 空き家バンク(LIFULL) ※予測はしないが、自治体と連携して空き家情報を集約
  - [空き家バンク](https://www.homes.co.jp/akiyabank/)

---

バグ報告や機能要望は Issue でお気軽にお知らせください。
