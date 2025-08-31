# 山麓 空き家マップ(Akiya‑Lens) プロジェクト構成ガイド v1

本ドキュメントは、**モデル作成に必要なデータ準備を最優先**としつつ、将来的な**自動更新パイプライン**や**Streamlitアプリ**まで見据えた、実行構成とディレクトリ構成の標準をまとめたものです。

---

## 0. ゴールと方針

* まずは **手動ダウンロードしたCSVを起点**に、市区町村マスタ（参照データ）を安定生成。
* 参照（ref）とETL（domain/source別前処理）を分離し、**設定はconfigに外出し**。
* 早期に **Streamlitアプリ**の素振りができる形にして、後からデータやUIを差し替え可能に。

---

## 1. ディレクトリ構成（最小運用 → 将来拡張可能）

```
akiya-lens/
├─ app/                         # Streamlit アプリ（UI層）
│  ├─ Home.py                   # トップページ（メトリクス、説明）
│  ├─ pages/
│  │  ├─ 01_地図_市区町村マスタ.py   # 地図とテーブルの参照
│  │  └─ 02_分析_ベースライン.py     # 係数やSHAPの可視化（将来）
│  └─ utils/
│     ├─ io.py                  # データ読み取り（キャッシュ含む）
│     └─ viz.py                 # 地図・図表の描画ヘルパ
│
├─ config/
│  ├─ ref_project.yaml          # 参照（マスタ）用の列マップ・命名など
│  └─ etl_project.yaml          # ETL用（将来作成・更新が多い）
│
├─ data/
│  ├─ raw/                      # 手動DLファイルの置き場（上書きOK）
│  ├─ interim/                  # 中間生成物（parquet推奨）
│  ├─ processed/                # ETL後（結合前のクリーン表）
│  ├─ master/                   # 参照マスタ（市区町村マスタ等）
│  └─ geojson/                  # 市区町村境界データ（GeoJSON, shp変換済みなど）
|
├─ docs/
│  └─ projects.md               # プロジェクト構成ガイド
│
├─ scripts/
│  ├─ ref/
│  │  └─ 010_build_city_master_from_local_csv.py
│  ├─ etl/                      # 空き家率/人口などのソース別前処理（将来）
│  └─ utils/                    # 共通関数（将来: io_utils, validate_utils など）
│
├─ notebooks/                   # 確認・探索ノート
│  └─ 01_ref_audit_city_master.ipynb
│
├─ pyproject.toml               # 依存とツール設定（任意: uv/poetry対応）
├─ README.md
└─ LICENSE (任意)
```

> **ポイント**
>
> * **ref**と**etl**を分けることで、更新頻度や責務の違いを明確化。
> * アプリは\*\*app/**配下に固定**（Streamlitの`pages/`規約に従う）。
> * `data/master` には**バージョン付き・日付付き**の成果物を将来置ける設計（命名テンプレで対応）。
> * `data/geojson` に市区町村境界ファイル（行政区域GeoJSONなど）を格納。

---

## 2. 実行構成（現状）

* **設定**: `config/ref_project.yaml`
* **スクリプト**: `scripts/ref/010_build_city_master_from_local_csv.py`
* **入力**: `data/raw/…` に置いた手動DL CSV（例: `FEA_hyoujun.csv`）
* **出力**: `data/master/city_master__all__v1__YYYYMMDD.csv`（`--out`未指定時）

### 2.1 使い方

```bash
# 例：明示出力する場合
python scripts/ref/010_build_city_master_from_local_csv.py \
  --config config/ref_project.yaml \
  --src data/raw/FEA_hyoujun.csv \
  --out data/master/city_master__all__v1__preview.csv

# 例：--out を省略し、configの命名テンプレを使う場合
python scripts/ref/010_build_city_master_from_local_csv.py \
  --config config/ref_project.yaml \
  --src data/raw/FEA_hyoujun.csv
```

### 2.2 スクリプトの主な仕様

* **文字コード自動判定**: `cp932`, `shift_jis`, `utf-8-sig`, `utf-8`
* **列名ゆらぎを吸収**: `config/ref_project.yaml` の `columns_map.city_master` に従い正規化
* **都道府県コードの補完**: 無い場合、**市区町村コードの先頭2桁**から導出
* **出力列（順序固定）**:
  `市区町村コード, 市区町村名, 都道府県コード, 都道府県名, 市区町村ふりがな, 過疎地域市町村, 都市種別`
* **重複処理**: `市区町村コード` で後勝ち、コード順に整列
* **保存形式**: UTF-8-SIG（Excelとの互換性重視）

---

## 3. 設定ファイル（config/ref\_project.yaml）

参照マスタ（city\_master）用の列マップと命名テンプレを保持。

```yaml
project:
  name: akiya-lens
  role: "reference"
  version: v1

io:
  output_dir: data/master
  encoding_out: utf-8-sig

columns_map:
  city_master:
    市区町村コード: ["標準地域コード", "全国地方公共団体コード", "全国地方公共団体番号", "市区町村コード"]
    市区町村名: ["市区町村", "団体名"]
    都道府県名: ["都道府県", "都道府県名"]
    都道府県コード: ["都道府県コード"]
    市区町村ふりがな: ["市区町村（ふりがな）", "市区町村名（よみ）"]
    過疎地域市町村: ["過疎地域市町村"]
    都市種別: ["都市種別"]

naming:
  city_master:
    filename_template: "city_master__all__${project.version}__${date:%Y%m%d}.csv"
```

> 列が追加・名称が変わっても、**YAMLに追記するだけ**でスクリプトはそのまま再利用可能。

---

## 4. Streamlit アプリ構成（最小スケルトン）

```
app/
├─ Home.py
├─ pages/
│  ├─ 01_地図_市区町村マスタ.py
│  └─ 02_分析_ベースライン.py
└─ utils/
   ├─ io.py
   └─ viz.py
```

### 4.1 データ読み込み（app/utils/io.py）

```python
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_city_master(path: str | Path):
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_geojson(path: str | Path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

### 4.2 トップページ（app/Home.py）

```python
import streamlit as st
from pathlib import Path
from utils.io import load_city_master

st.set_page_config(page_title="Akiya-Lens", layout="wide")
st.title("Akiya-Lens: 市区町村マスタの確認")

DEFAULT_MASTER = Path("data/master").glob("city_master__all__v1__*.csv")
DEFAULT_MASTER = sorted(DEFAULT_MASTER)[-1] if DEFAULT_MASTER else None

path = st.text_input("市区町村マスタCSVパス", value=str(DEFAULT_MASTER) if DEFAULT_MASTER else "")
if path:
    df = load_city_master(path)
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"行数: {len(df):,}")
```

### 4.3 地図ページ（app/pages/01\_地図\_市区町村マスタ.py）

> GeoJSONを読み込み choropleth に拡張予定。

```python
import streamlit as st
import pydeck as pdk
from utils.io import load_city_master, load_geojson

st.title("市区町村マスタ × 地図")

csv_path = "data/master/city_master__all__v1__YYYYMMDD.csv"
geo_path = "data/geojson/municipalities.geojson"

try:
    df = load_city_master(csv_path)
    gj = load_geojson(geo_path)
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=36.2, longitude=138.0, zoom=6),
        layers=[
            pdk.Layer("GeoJsonLayer", gj, get_fill_color="[180, 180, 200, 100]", pickable=True)
        ]
    ))
    st.dataframe(df.head(50), use_container_width=True)
except Exception as e:
    st.error(f"地図描画エラー: {e}")
```

### 4.4 実行

```bash
streamlit run app/Home.py
```

---

## 5. 命名規約とバージョニング

* **スクリプト**: `<順序>_<対象>_<動詞>.py`（例: `010_build_city_master_from_local_csv.py`）
* **参照データ出力**: `city_master__all__v<schema>__YYYYMMDD.csv`

  * スキーマ変更時は `v1 → v2` に上げて併存可能。
* **GeoJSON**: `data/geojson/municipalities.geojson` のように固定ファイル名で置き換えやすくする。

---

## 6. 将来の拡張（ロードマップ）

* **ETL構成**（`config/etl_project.yaml`／`scripts/etl/`）を段階的に追加

  * 空き家率ETL → 人口ETL → ハザードETL → JOINテーブル
* **バリデーション**（`pandera` or `Great Expectations`）を導入して品質担保
* **自動更新**（Makefile/CI→将来はAirflow/Prefect）
* **地図**（GeoJSON境界を choropleth 表示、ズーム単位切替など）

---

## 7. よくある質問（運用Tips）

* **列名が合わない** → `config/ref_project.yaml` の `columns_map` にエイリアスを追記
* **県を絞りたい** → 現状は全国対象。将来 `targets.prefectures` を使ってフィルタをオンに
* **GeoJSONはどこに置く？** → `data/geojson/` にまとめる。行政界は e-Stat や国土地理院から取得。
* **ファイルが巨大** → `interim` は parquet に、`processed` では必要列にスリム化

---

## 8. 参考コマンド（再掲）

```bash
# 参照マスタの生成（config利用）
python scripts/ref/010_build_city_master_from_local_csv.py \
  --config config/ref_project.yaml \
  --src data/raw/FEA_hyoujun.csv

# Streamlit の起動
streamlit run app/Home.py
```

---
