## 地価公示（KSJ L01）ETL・特徴量化 仕様書

> 本ドキュメントは Issue に貼り付ける or `docs/landprice_spec.md` として保存して運用ください。
> 目的：**2018/2023 の地価公示（住宅）を市区町村レベルに集計し、回帰モデル（LightGBM）と SHAP 可視化に利用可能な CSV を安定生成**する。

---

## TL;DR

* 入力：国交省 **KSJ L01**（地価公示）XML（2018・2023、都道府県別ファイル）。
* 出力：

  1. **points**（住宅のみ・座標つき）
  2. **long**（年×市区町村の中央値）
  3. **wide**（2018/2023 の横持ち＋差分・増減率）
* 特徴量の主役：`住宅地価_log中央値_2018`（水準）と `住宅地価_log差分`（勢い）。
* 2018 の座標は **`xlink:href` → `gml:Point` 解決**で確実に取得。
* 欠損（片年しかない自治体）は **NA のまま**（LightGBM は NA を許容）。
* CLI で **再現可能**、YAML で **差し替え可能**、日本語列名は **プロジェクト統一**。

---

## 1. スコープ

* 空き家率（目的変数：2023年）に対する説明変数として、\*\*住宅地価の水準（2018）**と**変化（2018→2023）\*\*を整形・提供。
* 将来予測（2028 など）を見据え、**過去時点の特徴だけで次期を予測**できるデータ設計を意識。
* 座標は現時点の学習で未使用だが、**空間集計や地図可視化**の拡張に備え、欠損なく保持。

---

## 2. ディレクトリ & ファイル

```
config/
  etl_project.yaml              # グローバル設定・データセット定義
data/
  raw/
    landprice/
      2018/L01-*.xml
      2023/L01-*.xml
  processed/
    131_landprice_points__v1.csv
    132_landprice_residential_median__long__v1.csv
    132_landprice_residential_median__wide__v1.csv
scripts/
  etl/
    131_landprice_ksj_l01_points_to_csv.py      # Stage1: XML → points
    132_landprice_ksj_l01_median_by_city.py     # Stage2: points → long / wide
```

---

## 3. YAML（抜粋：`config/etl_project.yaml`）

```yaml
io:
  encoding_in_candidates: ["utf-8-sig","utf-8","cp932","shift_jis"]
  encoding_out: utf-8-sig

datasets:
  landprice_ksj_l01:
    sources:
      "2018": "data/raw/landprice/2018/L01-*.xml"
      "2023": "data/raw/landprice/2023/L01-*.xml"
    namespaces:
      app: "http://nlftp.mlit.go.jp/ksj/schemas/ksj-app"
      gml: "http://schemas.opengis.net/gml/3.2.1/gml.xsd"
      xlink: "http://www.w3.org/1999/xlink"
    item_xpath: ".//app:LandPrice"
    fields:
      year: "app:year/gml:TimeInstant/gml:timePosition"
      市区町村コード: "app:administrativeAreaCode"
      市区町村名: "app:cityName"
      価格_円m2: "app:postedLandPrice"
      利用現況: "app:currentUse"
      座標_pos: "app:position/gml:Point/gml:pos"

study_regions:
  yatsugatake_alps:
    prefecture_codes: ["19","20","21","22"]
```

> ※ Stage1 は YAML の名前空間を尊重しつつ、\**実装側で `xlink:href` 解決と year のフォールバック（ファイル名 L01-18\_* → 2018 など）を内包\*\*しています。

---

## 4. Stage1：XML → points（住宅のみ・座標つき）

**スクリプト**：`scripts/etl/131_landprice_ksj_l01_points_to_csv.py`
**入力**：`data/raw/landprice/{2018,2023}/L01-*.xml`
**出力**：`data/processed/131_landprice_points__v1.csv`

### 4.1 実行例

```bash
PYTHONPATH=. python scripts/etl/131_landprice_ksj_l01_points_to_csv.py \
  --config config/etl_project.yaml \
  --xml_glob "data/raw/landprice/2018/L01-*.xml" "data/raw/landprice/2023/L01-*.xml" \
  --out_csv data/processed/131_landprice_points__v1.csv \
  --region yatsugatake_alps
```

### 4.2 抽出・処理仕様

* **フィルタ**：`利用現況 = 住宅`
* **年（year）**：

  1. `app:year/gml:TimeInstant/gml:timePosition`
  2. `app:year/gml:timePosition`
  3. `app:year`（数値）
  4. **ファイル名から推定**：`L01-18_*.xml → 2018`, `L01-23_*.xml → 2023`
* **座標**：

  * まず `app:position/gml:Point/gml:pos`（インライン）
  * だめなら `app:position/@xlink:href="#ptX"` を **`gml:Point[@gml:id="ptX"]/gml:pos` に解決**
  * GML 名前空間差は**ローカル名ベース**で解釈（3.2/3.2.1 両対応）
* **県コードフィルタ**：`--region` で指定。`市区町村コード` 先頭 2 桁と突合。
* **出力カラム（日本語統一）**：

  * `year`, `市区町村コード`（5桁ゼロ埋め）, `市区町村名`, `価格_円m2`, `緯度`, `経度`, `利用現況`, `source_file`

---

## 5. Stage2：points → long / wide（集計・特徴量）

**スクリプト**：`scripts/etl/132_landprice_ksj_l01_median_by_city.py`
**入力**：`data/processed/131_landprice_points__v1.csv`
**出力**：

* `data/processed/132_landprice_residential_median__long__v1.csv`
* `data/processed/132_landprice_residential_median__wide__v1.csv`

### 5.1 実行例

```bash
python scripts/etl/132_landprice_ksj_l01_median_by_city.py \
  --config config/etl_project.yaml \
  --in_csv data/processed/131_landprice_points__v1.csv \
  --out_long_csv data/processed/132_landprice_residential_median__long__v1.csv \
  --out_wide_csv data/processed/132_landprice_residential_median__wide__v1.csv \
  --base_year 2018 --target_year 2023
```

### 5.2 集計仕様

* **long 出力**（年×市区町村）

  * `year, 市区町村コード, 市区町村名, 標準地点数, 住宅地価_中央値, 住宅地価_log中央値`
  * 中央値は `価格_円m2` の中央値、`log` は自然対数（`x>0` のみ）
* **wide 出力**（横持ち）

  * `住宅地価_中央値_{2018,2023}`
  * `住宅地価_log中央値_{2018,2023}`
  * `標準地点数_{2018,2023}`
  * `住宅地価_log差分`（= `log中央値_2023 - log中央値_2018`）
  * `住宅地価_増減率[%]`（= `(中央値_2023 / 中央値_2018 - 1) * 100`）
* **注意**：両年が揃わない自治体は差分・増減率が **NA**。

---

## 6. バリデーション（品質ゲート）

> すべて **実データで検証済み**。下記を満たさない場合は ETL を確認。

* **points**：

  * `year ∈ {2018, 2023}`, `利用現況 = 住宅`
  * `市区町村コード` は 5 桁
  * `価格_円m2 > 0`
  * `緯度/経度` が **2018/2023 とも非欠損**（参照解決が効いていること）
* **long**：

  * `標準地点数`, `住宅地価_中央値` が **points から再計算と一致**
  * `住宅地価_log中央値 = log(住宅地価_中央値)`（誤差 ≦ 1e-6）
* **wide**：

  * `long` 再ピボット結果と **全カラム一致**（誤差 ≦ 1e-6）
  * `住宅地価_log差分` と `住宅地価_増減率[%]` が数式と一致

---

## 7. 特徴量の利用指針（LightGBM × SHAP）

* **主特徴**

  * **水準**：`住宅地価_log中央値_2018`
  * **勢い**：`住宅地価_log差分`（= ほぼ割合変化に相当、安定）
* **他特徴（別ETL）と組み合わせ**

  * 人口動態：2018水準＋2018→2023 変化
  * 空き家率：2018水準や増加率（参考）
* **共線性**は LightGBM 的には問題になりにくいが、**SHAP の安定性**のため

  * 水準＋変化の “重なり過ぎ” を感じたら、どちらかに寄せる or 相関の強い片方をドロップ

---

## 8. 既知の注意点・限界

* **標準地の空間分布は不均一**（都市部に多く、農村部に少ない）→ `標準地点数` を参照して信頼度を把握。
* 市区町村により **片年のみ**のケースがあり、`log差分`・`増減率[%]` が NA になる。
* KSJ L01 のスキーマ差（年版・名前空間差）を実装で吸収しているが、将来版で XPath が変わった場合は **YAML** 側で `fields` を差し替え可能。

---

## 9. 将来拡張（オプション）

* **GeoJSON/WKT 出力**：points からジオメトリを保存（地図UI直結）。
* **メッシュ/駅圏/学区**などの空間集計：座標を用いたゾーニング集計。
* **学習用マスタ結合スクリプト**：空き家率・人口動態のワイド表と `市区町村コード` で結合。
* **CI で品質ゲート**：上記バリデーションを pytest に落とし込み、PR 時に検証。

---

## 10. 実行ログの例（正常系）

```
[OK] Stage1 wrote: data/processed/131_landprice_points__v1.csv \
rows=2503 years={2018: 1256, 2023: 1247} lat_nonnull=2503 lon_nonnull=2503

[OK] Stage2 wrote: ... (long rows=...) , ... (wide rows=...)
# long と points の中央値・件数が一致、wide は long 再ピボットと一致
```

---

## 11. ライセンス・出典

* データ出典：国土数値情報（地価公示）（KSJ L01）
* 本プロジェクト内での利用は、出典表記・利用規約に従います（リポジトリの `NOTICE`/`README` に明記）。

---

## 12. 変更履歴（抜粋）

* **v1**

  * Stage1 を堅牢化：`xlink:href → gml:Point/gml:pos` 参照解決、year のファイル名フォールバック、ローカル名ベースの GML 対応。
  * Stage2 を厳密化：`year` の int 化、`long → wide` の差分・増減率生成。
  * バリデーション手順整備・日本語列名の統一。

---

### 付録 A：データ辞書（主な列）

| ファイル   | 列名            | 意味                                |
| ------ | ------------- | --------------------------------- |
| points | year          | 公示年（2018 / 2023）                  |
|        | 市区町村コード       | JIS5桁                             |
|        | 市区町村名         | 自治体名                              |
|        | 価格\_円m2       | 住宅の標準地 1㎡あたり価格                    |
|        | 緯度 / 経度       | 標準地の座標（WGS84、XMLの gml\:pos をパース）  |
| long   | 標準地点数         | 該当市区町村×年の住宅標準地件数                  |
|        | 住宅地価\_中央値     | 上記の中央値                            |
|        | 住宅地価\_log中央値  | 住宅地価\_中央値の自然対数                    |
| wide   | 住宅地価\_log差分   | `log中央値_2023 - log中央値_2018`       |
|        | 住宅地価\_増減率\[%] | `(中央値_2023 / 中央値_2018 - 1) * 100` |

