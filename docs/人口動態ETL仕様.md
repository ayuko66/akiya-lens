# ETL人口動態データ作成 仕様書

## 概要

* 入力: **社会・人口統計体系（FEI\_CITY\_2509.csv）**
* 出力: **人口動態（2018年 / 2023年）＋正規化項目付きCSV**
* 目的: 空き家率分析に利用するための人口動態特徴量セットを作成する

---

## スクリプト

* `scripts/etl/120_estat_population_stats_clean.py`

---

## 入力

* `data/raw/FEI_CITY_2509.csv`
* 設定: `config/etl_project.yaml`

  * `io.encoding_in_candidates`: 入力エンコーディング候補
  * `io.encoding_out`: 出力エンコーディング
  * `study_regions`: 地域フィルタ（例: `yatsugatake_alps` → 山梨/長野/岐阜/静岡）
  * （任意）`datasets.population_stats.columns_map`: 列名マッピング（表記ゆれ対応）

---

## 出力

指定したリージョンごとに以下3ファイルを生成。

```
data/processed/
├ population_stats_<region>_2018.csv
├ population_stats_<region>_2023.csv
└ population_stats_<region>__long__v1_preview.csv
```

* `<region>` はコマンド引数 `--region` に指定した study\_regions のキー
  例: `yatsugatake_alps`, `all`

---

## 出力カラム仕様

### 基本項目

| 列名       | 説明              |
| -------- | --------------- |
| 市区町村コード  | JIS5桁コード        |
| 市区町村名    | 市区町村名           |
| year     | 調査年（2018, 2023） |
| 総人口      | 総人口（人）          |
| 15歳未満人口  | 15歳未満人口（人）      |
| 15〜64歳人口 | 生産年齢人口（人）       |
| 65歳以上人口  | 高齢人口（人）         |
| 出生数      | 出生数（人）          |
| 死亡数      | 死亡数（人）          |
| 転入者数     | 転入者数（人）         |
| 転出者数     | 転出者数（人）         |
| 世帯数      | 世帯数（世帯）         |

### 派生・正規化項目

| 列名          | 説明                  |
| ----------- | ------------------- |
| 高齢化率\[%]    | 65歳以上人口 ÷ 総人口 ×100  |
| 年少人口率\[%]   | 15歳未満人口 ÷ 総人口 ×100  |
| 生産年齢人口率\[%] | 15〜64歳人口 ÷ 総人口 ×100 |
| 出生率\[‰]     | 出生数 ÷ 総人口 ×1000     |
| 死亡率\[‰]     | 死亡数 ÷ 総人口 ×1000     |
| 転入率\[‰]     | 転入者数 ÷ 総人口 ×1000    |
| 転出率\[‰]     | 転出者数 ÷ 総人口 ×1000    |
| 転入超過率\[‰]   | (転入−転出) ÷ 総人口 ×1000 |
| 1世帯当たり人員    | 総人口 ÷ 世帯数           |

---

## 実行方法

```bash
# 八ヶ岳アルプス地域のみ
PYTHONPATH=. python scripts/etl/120_estat_population_stats_clean.py \
  --config config/etl_project.yaml \
  --in_file data/raw/FEI_CITY_2509.csv \
  --out_dir data/processed \
  --region yatsugatake_alps

# 全国対象
PYTHONPATH=. python scripts/etl/120_estat_population_stats_clean.py \
  --config config/etl_project.yaml \
  --in_file data/raw/FEI_CITY_2509.csv \
  --out_dir data/processed \
  --region all
```

---

## 注意点

* 列名の表記ゆれ（例: `15〜64歳人口`, `15-64歳人口`）は正規化処理で吸収
* 政令市は区単位データが含まれることがある → 必要に応じて市単位集約処理を追加可能
* `etl_project.yaml` に列マッピングを追加すると確実に安定（例:）

  ```yaml
  datasets:
    population_stats:
      columns_map:
        15〜64歳人口: ["A1302_15-64歳人口【人】"]
  ```

---
