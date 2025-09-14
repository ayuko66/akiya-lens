# OSM POI密度 ETL 仕様書

本仕様書は、市区町村ごとに OpenStreetMap (OSM) の POI（駅・スーパー・学校・病院＋任意でクリニック）件数を集計し、地表面積(km²)で割った密度を算出・CSV出力する処理の仕様を記載する。

## 対象スクリプトと役割
- `scripts/osm/fetch_poi_density.py`: CLI本体。入出力/設定/キャッシュ/集計フローを実装。
- `scripts/osm/api.py`: Overpass/Wikidata APIの呼び出し、境界のGeoJSON取得。
- `scripts/osm/utils.py`: 面積計算（測地線; WGS84）とクエリユーティリティ。

## 入力
- 市区町村マスターCSV
  - 必須列: `市区町村コード`, `都道府県名`, `市区町村名`
  - 任意列: `wikidata`（既知のQIDがあれば優先利用）
  - 列名は `config/ref_project.yaml` の `columns_map.city_master` でエイリアス指定可能。
- 設定YAML: `config/etl_project.yaml`
  - `study_regions.<region>.prefecture_codes` で対象都道府県コード（先頭2桁）を指定。

## 出力
- CSV（指定パス `--out`）
  - カラム（日本語表記）
    - 市区町村コード, 都道府県名, 市区町村名, Wikidata QID, 面積[km²]
    - 駅件数, 駅密度[件/km²]
    - スーパー件数, スーパー密度[件/km²]
    - 学校件数, 学校密度[件/km²]
    - 病院件数, 病院密度[件/km²]

## 実行例
- 基本（キャッシュ有効, TTL既定=168時間）
  ```bash
  python scripts/osm/fetch_poi_density.py \
    --in data/master/city_master__all__v1__preview.csv \
    --out data/processed/osm_density.csv \
    --config config/etl_project.yaml
  ```
- 駅の `public_transport=station` と クリニックを合算
  ```bash
  python scripts/osm/fetch_poi_density.py \
    --in data/master/city_master__all__v1_preview.csv \
    --out data/processed/osm_density_with_pt_clinic.csv \
    --config config/etl_project.yaml \
    --include-public-transport-stations \
    --include-clinics
  ```
- キャッシュディレクトリ指定＋POI件数TTLを12時間に短縮
  ```bash
  python scripts/osm/fetch_poi_density.py \
    --in data/master/city_master__all__v1__preview.csv \
    --out data/processed/osm_density.csv \
    --config config/etl_project.yaml \
    --cache-dir ./mycache \
    --poi-cache-ttl-hours 12
  ```
- キャッシュを無効化（常に最新取得）
  ```bash
  python scripts/osm/fetch_poi_density.py \
    --in data/master/city_master__all__v1__preview.csv \
    --out data/processed/osm_density_fresh.csv \
    --config config/etl_project.yaml \
    --no-cache
  ```

## 主なオプション
- `--region`: `etl_project.yaml` の `study_regions` キー（デフォルト: `yatsugatake_alps`）
- `--include-public-transport-stations`: 駅に `public_transport=station` を含める
- `--include-clinics`: 病院に `amenity=clinic` を合算
- `--sleep-sec`: Overpassアクセス間の待機秒（礼儀）
- `--retries`: APIリトライ回数
- キャッシュ関連
  - `--cache-dir`: キャッシュ格納ディレクトリ（既定: `data/cache`）
  - `--no-cache`: 全てのローカルキャッシュ（QID/面積/POI件数）を無効化
  - `--no-poi-cache`: POI件数キャッシュのみ無効化
  - `--poi-cache-ttl-hours`: POI件数キャッシュのTTL（時間）。0以下で常に再取得

## 処理フロー
1. 入力CSVを読み込み、対象都道府県コードでフィルタ。
2. QID解決（優先順位）
   - CSVの `wikidata` 列 → QIDキャッシュ（code→QID） → Wikidata SPARQL（P429: 団体コード）
   - 成功時はキャッシュに保存。
3. 行政境界取得と面積計算
   - Overpassで `rel["wikidata"=QID]["boundary"="administrative"]` を取得 → GeoJSON へ変換。
   - 測地線（WGS84）で面積(km²)を計算し、面積キャッシュに保存。GeoJSONも保存（任意再利用）。
4. POI件数集計（カテゴリ：駅/スーパー/学校/病院）
   - カテゴリごとにタグ集合を定義。
   - まず POI件数キャッシュ（キー: `QID|カテゴリ|タグ集合`）を確認し、未キャッシュ/期限切れのみを Overpass にバッチ問い合わせ（`out count;`）。
   - 取得結果をキャッシュに保存。
5. 密度計算（件数/面積）し、CSV出力。

## キャッシュ仕様
- 目的: ネットワーク往復と重い計算（境界取得・面積計算・POI集計）を再利用して高速化。
- 既定配置: `data/cache/`
  - `qid_by_code.json`: `code` → `QID`
  - `area_km2_by_qid.json`: `QID` → `area_km2`
  - `boundary_geojson/{QID}.geojson`: 境界GeoJSON（任意再利用）
  - `poi_counts.json`: `"QID|カテゴリ|tag1;tag2;..."` → `{ "count": <int>, "ts": <epoch秒> }`
- TTL
  - POI件数キャッシュのみ `--poi-cache-ttl-hours` で制御（既定 168h）。
  - `0`以下で常に再取得。QID/面積は基本的に長期間安定のためTTLなし。
- 無効化
  - 全無効: `--no-cache`
  - POI件数のみ無効: `--no-poi-cache`
- 反映タイミング
  - OSM/Wikidataの更新はTTL内には反映されない。最新化したいときはTTL短縮/`--no-cache`/キャッシュ削除を選択。

## バッチ問い合わせ仕様（Overpass）
- 行政界を `.a` として解決後、カテゴリごとの集合を `.stations` などへ格納。
- 各カテゴリについて `out count;` を順に出力し、`nodes/ways/relations/total` から総数を解釈。
- これにより、カテゴリ数分のリクエストを1回の往復で完了。

## レート制御・リトライ
- `--sleep-sec` による礼儀的待機。エラー時は指数バックオフ。
- Overpass ならびに Wikidata の利用規約・レート制限を遵守。

## 既知の制約
- Overpassのミラー可用性やクォータに依存。
- 面積は GeoJSON のポリゴン/マルチポリゴンを WGS84測地線で計算。座標系の前提に注意。
- QIDはWikidataの `P429`（団体コード）に依存。CSVのコードと対応がない場合は解決不可。
- 面積が0または境界取得失敗の自治体はスキップ。

## 変更履歴
- v0.2: バッチ集計・ローカルキャッシュ（QID/面積/POI件数/TTL）を追加。
- v0.1: 初版（単純集計）。
