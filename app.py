"""Streamlit interface for the Akiya-Lens residual model.

このアプリケーションは ``docs/残差モデルとアプリ設計仕様.md`` の仕様に従います。
マスター特徴量テーブルをロードし、ベースライン + 残差パイプラインを再構築し、
リスクカテゴリ、予測、SHAP由来の説明を含むインタラクティブな市町村マップをレンダリングします。

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import folium
    from folium import features
    from branca.colormap import linear as branca_linear
    from branca.element import MacroElement, Template
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover - handled gracefully in UI
    folium = None
    features = None
    branca_linear = None
    MacroElement = None
    Template = None
    st_folium = None

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover - handled gracefully in UI
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore

from scripts.models.diff_model_utils import (
    compute_predictions,
    compute_shap_topk,
    get_numeric_series,
)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    REPO_ROOT / "data/processed/features_master__wide__v1.csv"
)  # 市区町村データ(特徴量)
GEOJSON_PATH = REPO_ROOT / "data/geojson/municipalities.geojson"  # 地図データ(geojson)
MODEL_PATH = REPO_ROOT / "models/final_diff_model.cbm"  # 学習済みモデル(CatBoost)
METRICS_PATH = REPO_ROOT / "data/processed/model_metrics.json"  # 評価メトリクス
INSPECTOR_PATH = REPO_ROOT / "data/processed/diff_model_inspector.json"

DEFAULT_TOLERANCE = 0.1  # "横ばい"とする閾値 (スライダーUIで調整可)
# 市区町村塗りつぶしセレクト
MAP_OPTIONS = (
    "4段階リスク",
    "2023年空き家率（実測）",
    "2023年空き家率（予測）",
    "残差（実測−予測）",
)
# リスクラベル定義（色・凡例テキストを一元管理）
RISK_LEVELS = {
    "(最優先)": {"color": "#d73027", "legend": "赤(最優先) 高・➚"},
    "(注意)": {"color": "#fc8d59", "legend": "オレンジ(注意) 高・横/↓"},
    "(警戒)": {"color": "#fee08b", "legend": "黄(警戒) 低・➚"},
    "(低)": {"color": "#1a9850", "legend": "緑(低) 低・横/↓"},
}

# 高低×トレンドの組合せごとのラベル割り当て
RISK_RULES = {
    (True, "増"): "(最優先)",
    (True, "横ばい"): "(注意)",
    (True, "減"): "(注意)",
    (False, "増"): "(警戒)",
    (False, "横ばい"): "(低)",
    (False, "減"): "(低)",
}

RISK_LEGEND_ORDER = ["(最優先)", "(注意)", "(警戒)", "(低)"]
DEFAULT_RISK_LABEL = "(低)"

NAN_COLOR = "#d9d9d9"


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # BOM除去
    return df


@st.cache_data(show_spinner=False)  # "実行中…ださない"
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {path}")
    df = pd.read_csv(path)
    df = _normalise_columns(df)
    df["市区町村コード"] = (
        df["市区町村コード"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)  # 都道府県コード1〜9を0埋め
    )
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_model_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_inspector_payload(path: Path, _signature: float) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError:
        pass
    return {}


@st.cache_resource(show_spinner=False)
def load_catboost_model(path: Path) -> Optional[CatBoostRegressor]:
    if CatBoostRegressor is None:
        return None
    if not path.exists():
        return None
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


# ---------------------------------------------------------------------------
# Risk classification & geo utilities
# ---------------------------------------------------------------------------


# 4レベル空き家リスクの判定
def classify_risk(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    df = df.copy()
    vac18 = get_numeric_series(df, "空き家率_2018")
    vac23 = get_numeric_series(df, "空き家率_2023")
    delta = vac23 - vac18

    if vac23.dropna().empty:
        threshold = np.nan
    else:
        threshold = float(np.nanpercentile(vac23.dropna(), 75))

    trend = np.where(
        delta > tolerance, "増", np.where(delta < -tolerance, "減", "横ばい")
    )
    high_now = (
        vac23 >= threshold if not np.isnan(threshold) else np.full(len(vac23), False)
    )

    labels = []
    for is_high, trend_value in zip(high_now, trend):
        label = RISK_RULES.get((bool(is_high), trend_value), DEFAULT_RISK_LABEL)
        labels.append(label)

    df["P75_threshold"] = threshold
    df["Δ(23-18)"] = delta
    df["リスク区分"] = labels
    df["トレンド"] = trend

    return df


def _format_shap_text(items: Iterable[str]) -> str:
    values = [item for item in items if item]
    if not values:
        return "SHAP情報なし"
    return "<br/>".join(values)


def build_static_geojson(
    geojson: Optional[dict[str, Any]],
    df: pd.DataFrame,
    shap_lookup: dict[str, list[str]],
    tooltip_lookup: Optional[dict[str, Dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    if geojson is None:
        return None

    static_geojson = json.loads(json.dumps(geojson))
    dedup_df = (
        df.dropna(subset=["市区町村コード"])
        .drop_duplicates(subset=["市区町村コード"], keep="last")
        .set_index("市区町村コード")
    )
    lookup = dedup_df.to_dict("index")

    for feature in static_geojson.get("features", []):
        props = feature.setdefault("properties", {})
        code = str(props.get("市区町村コード", "")).zfill(5)
        record = lookup.get(code)
        props["akiya_code"] = code
        if record is None:
            props["akiya_has_data"] = False
            # 動的フィールドも初期化
            props.setdefault("akiya_name", "不明")
            props.setdefault("akiya_vac18", None)
            props.setdefault("akiya_vac23", None)
            props.setdefault("akiya_pred", None)
            props.setdefault("akiya_residual", None)
            props.setdefault("akiya_delta", None)
            props.setdefault("akiya_shap", "SHAP情報なし")
            props.setdefault("akiya_risk_dynamic", DEFAULT_RISK_LABEL)
            props.setdefault("akiya_trend_dynamic", "")
            props.setdefault("akiya_map_dynamic", None)
            continue

        props["akiya_has_data"] = True
        tooltip_info = tooltip_lookup.get(code) if tooltip_lookup else None
        props["akiya_name"] = tooltip_info.get("akiya_name") if isinstance(
            tooltip_info, dict
        ) and tooltip_info.get("akiya_name") else record.get("市区町村名", "不明")
        props["akiya_vac18"] = _safe_value(
            tooltip_info.get("akiya_vac18")
            if isinstance(tooltip_info, dict) and "akiya_vac18" in tooltip_info
            else record.get("空き家率_2018")
        )
        props["akiya_vac23"] = _safe_value(
            tooltip_info.get("akiya_vac23")
            if isinstance(tooltip_info, dict) and "akiya_vac23" in tooltip_info
            else record.get("空き家率_2023")
        )
        props["akiya_pred"] = _safe_value(record.get("pred_空き家率_2023"))
        props["akiya_residual"] = _safe_value(record.get("残差(実測-予測)"))
        props["akiya_delta"] = _safe_value(
            tooltip_info.get("akiya_delta")
            if isinstance(tooltip_info, dict) and "akiya_delta" in tooltip_info
            else record.get("Δ(23-18)")
        )
        internal_key = str(record.get("_akiya_internal_index", code))
        shap_items = shap_lookup.get(internal_key, [])
        props["akiya_shap"] = _format_shap_text(shap_items)

        # Dynamic placeholders
        props["akiya_risk_dynamic"] = DEFAULT_RISK_LABEL
        props["akiya_trend_dynamic"] = ""
        props["akiya_map_dynamic"] = None

    return static_geojson


def update_geojson_dynamic_properties(
    geojson: Optional[dict[str, Any]],
    risk_lookup: dict[str, str],
    trend_lookup: dict[str, str],
    map_lookup: dict[str, Optional[float]],
) -> None:
    if geojson is None:
        return

    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        code = props.get("akiya_code") or str(props.get("市区町村コード", "")).zfill(5)
        risk = risk_lookup.get(code, DEFAULT_RISK_LABEL)
        trend = trend_lookup.get(code, "")
        value = None
        if map_lookup:
            value = map_lookup.get(code)
            if isinstance(value, float) and np.isnan(value):
                value = None
        props["akiya_risk_dynamic"] = risk
        props["akiya_trend_dynamic"] = trend
        props["akiya_map_dynamic"] = value


def _safe_value(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_map(
    geojson: Optional[dict[str, Any]],
    map_option: str,
    legend_name: str,
    risk_lookup: dict[str, str],
    trend_lookup: dict[str, str],
    map_lookup: dict[str, Optional[float]],
) -> Optional[folium.Map]:
    if geojson is None or folium is None or st_folium is None:
        return None
    # foliumライブラリでマップを準備 (初期表示は日本国列島)
    m = folium.Map(location=[35.6, 137.8], zoom_start=8, tiles="cartodbpositron")

    # ツールチップ設定（静的GeoJSONに含まれるプロパティ）
    def tooltip_fields() -> list[str]:
        return [
            "akiya_name",
            "akiya_vac18",
            "akiya_vac23",
            "akiya_delta",
            "akiya_risk_dynamic",
            "akiya_trend_dynamic",
        ]

    # ツールチップ設定 (日本語表示名)
    tooltip = features.GeoJsonTooltip(
        fields=tooltip_fields(),
        aliases=[
            "自治体",
            "空き家率(2018)%",
            "空き家率(2023)%",
            "Δ(23-18)pt",
            "リスク区分",
            "トレンド",
        ],
        localize=True,
        sticky=False,
    )

    # 地図の色分けごとに設定
    if map_option == "4段階リスク":
        color_map = {label: info["color"] for label, info in RISK_LEVELS.items()}

        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature.get("properties", {})
            if not props.get("akiya_has_data"):
                return {
                    "fillColor": NAN_COLOR,
                    "color": "#666666",
                    "weight": 0.2,
                    "fillOpacity": 0.5,
                }
            code = props.get("akiya_code") or str(
                props.get("市区町村コード", "")
            ).zfill(5)
            risk = risk_lookup.get(code, DEFAULT_RISK_LABEL)
            color = color_map.get(risk, NAN_COLOR)
            return {
                "fillColor": color,
                "color": "#4d4d4d",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }

        # GeoJSONレイヤーを地図に追加。上記のstyle_functionを適用
        folium.GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=lambda _: {"weight": 1.5, "color": "#000000"},
            tooltip=tooltip,
        ).add_to(m)

        # HTMLで自作した凡例を地図の左下に追加
        if Template is not None and MacroElement is not None:
            legend_items_html = "".join(
                [
                    (
                        '<div style="display:flex; align-items:center; margin-bottom:4px;">'
                        f"<span style=\"display:inline-block;width:14px;height:14px;background:{RISK_LEVELS[label]['color']};border:1px solid #555;\"></span>"
                        f"<span style=\"margin-left:8px;\">{RISK_LEVELS[label]['legend']}</span>"
                        "</div>"
                    )
                    for label in RISK_LEGEND_ORDER
                ]
            )
            legend_template = """
            {% macro html(this, kwargs) %}
            <div style="position: fixed; bottom: 30px; left: 30px; width: 230px;
                        background-color: #ffffff; border: 1px solid #999999;
                        z-index: 9999; font-size: 13px; padding: 10px;
                        color: #333333; box-shadow: 2px 2px 4px rgba(0,0,0,0.25);">
              <b style="display:block; margin-bottom:6px;">4段階リスク</b>
              __ITEMS__
            </div>
            {% endmacro %}
            """
            legend_template = legend_template.replace("__ITEMS__", legend_items_html)
            legend = MacroElement()
            legend._template = Template(legend_template)
            m.get_root().add_child(legend)
    # --- ケース2: それ以外のオプション（空き家率など）が選択された場合 ---
    else:
        values = [
            v
            for v in map_lookup.values()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not values or branca_linear is None:
            colormap = None
        else:
            vmin, vmax = float(np.min(values)), float(np.max(values))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            colormap = branca_linear.YlOrRd_09.scale(vmin, vmax)  # グラデ表示
            colormap.caption = legend_name
            colormap.add_to(m)  # 地図に凡例（カラースケール）を追加

        # 各市区町村をどのように色付けするかの設定
        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature.get("properties", {})
            if not props.get("akiya_has_data"):
                return {
                    "fillColor": NAN_COLOR,
                    "color": "#666666",
                    "weight": 0.2,
                    "fillOpacity": 0.5,
                }
            code = props.get("akiya_code") or str(
                props.get("市区町村コード", "")
            ).zfill(5)
            value = map_lookup.get(code)
            if (
                value is None
                or (isinstance(value, float) and np.isnan(value))
                or colormap is None
            ):
                color = NAN_COLOR
            else:
                color = colormap(value)
            return {
                "fillColor": color,
                "color": "#4d4d4d",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }

        # GeoJSONレイヤーを地図に追加
        folium.GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=lambda _: {"weight": 1.5, "color": "#000000"},
            tooltip=tooltip,
        ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Akiya-Lens", layout="wide")
    st.title("🏠 Akiya-Lens 残差モデルビューア")
    st.caption("2018→2023 空き家率のベースライン + 残差モデル解析")

    try:
        features_df = load_features(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    geojson = load_geojson(GEOJSON_PATH)
    metrics = load_model_metrics(METRICS_PATH)
    model = load_catboost_model(MODEL_PATH)
    if INSPECTOR_PATH.exists():
        inspector_signature: Optional[float] = INSPECTOR_PATH.stat().st_mtime
        inspector_payload = load_inspector_payload(
            INSPECTOR_PATH, inspector_signature
        )
    else:
        inspector_signature = None
        inspector_payload = {}

    with st.sidebar:
        st.header("表示設定")
        tol = st.slider("横ばい許容幅 (％ポイント)", 0.0, 1.0, DEFAULT_TOLERANCE, 0.05)
        map_option = st.selectbox("地図の色分け", MAP_OPTIONS)
        if model is None:
            st.warning(
                "CatBoostモデルが読み込めませんでした。ベースラインのみを表示します。"
            )
        if metrics:
            st.subheader("モデル指標")
            cat_metrics = metrics.get("catboost")
            if cat_metrics:
                st.metric("CatBoost R²", f"{cat_metrics.get('r2', np.nan):.3f}")
                st.metric("CatBoost MSE", f"{cat_metrics.get('mse', np.nan):.3f}")

    cache_stale = (
        st.session_state.get("inspector_signature") != inspector_signature
    )

    if "model_cache" not in st.session_state or cache_stale:
        if inspector_payload:
            enriched_once = features_df.copy()
            pred_records: List[Dict[str, Any]] = []
            shap_lookup_once: Dict[str, List[str]] = {}
            tooltip_lookup_once: Dict[str, Dict[str, Any]] = {}

            for code, payload in inspector_payload.items():
                code_str = str(code).zfill(5)
                shap_lookup_once[code_str] = payload.get("shap_top3", [])
                tooltip_info = payload.get("tooltip")
                if isinstance(tooltip_info, dict):
                    tooltip_lookup_once[code_str] = tooltip_info

                record = {
                    "市区町村コード": str(payload.get("市区町村コード", code_str)).zfill(5),
                    "baseline_pred_delta": payload.get("baseline_pred_delta"),
                    "residual_model_pred": payload.get("residual_model_pred"),
                    "pred_delta": payload.get("pred_delta"),
                    "pred_空き家率_2023": payload.get("pred_空き家率_2023"),
                    "残差(実測-予測)": payload.get("残差(実測-予測)"),
                    "Δ(23-18)": payload.get("Δ(23-18)"),
                }
                pred_records.append(record)

            if pred_records:
                pred_df = pd.DataFrame(pred_records).drop_duplicates(
                    subset=["市区町村コード"], keep="last"
                )
                pred_df["市区町村コード"] = (
                    pred_df["市区町村コード"].astype(str).str.zfill(5)
                )
                pred_cols = [
                    "baseline_pred_delta",
                    "residual_model_pred",
                    "pred_delta",
                    "pred_空き家率_2023",
                    "残差(実測-予測)",
                    "Δ(23-18)",
                ]
                for col in pred_cols:
                    if col in pred_df.columns:
                        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

                enriched_once = enriched_once.merge(
                    pred_df, on="市区町村コード", how="left"
                )
            cache_messages: List[str] = []
            static_geojson = build_static_geojson(
                geojson, enriched_once, shap_lookup_once, tooltip_lookup_once
            )
        else:
            enriched_once, feature_matrix, base_messages = compute_predictions(
                features_df, model
            )
            if feature_matrix is not None and "市区町村コード" in features_df.columns:
                code_series = features_df.loc[feature_matrix.index, "市区町村コード"]
            else:
                code_series = None
            shap_lookup_once, shap_message_once = compute_shap_topk(
                model, feature_matrix, codes=code_series
            )
            cache_messages = base_messages.copy()
            cache_messages.append(
                "diff_model_inspector.json が見つからなかったため、予測とSHAPを再計算しました。"
            )
            if shap_message_once:
                cache_messages.append(shap_message_once)
            tooltip_lookup_once = {}
            static_geojson = build_static_geojson(
                geojson, enriched_once, shap_lookup_once
            )

        st.session_state["model_cache"] = {
            "enriched_df": enriched_once,
            "shap_lookup": shap_lookup_once,
            "tooltip_lookup": tooltip_lookup_once,
            "geojson_static": static_geojson,
            "messages": cache_messages,
        }
        st.session_state["inspector_signature"] = inspector_signature

    cache = st.session_state["model_cache"]
    enriched_df = cache["enriched_df"].copy()
    shap_lookup = cache["shap_lookup"]
    tooltip_lookup = cache.get("tooltip_lookup", {})
    static_geojson = cache.get("geojson_static")
    if static_geojson is None and geojson is not None:
        static_geojson = build_static_geojson(
            geojson, enriched_df, shap_lookup, tooltip_lookup
        )
        cache["geojson_static"] = static_geojson
    info_messages = list(cache.get("messages", []))

    classified_df = classify_risk(enriched_df, tol)

    risk_lookup = (
        classified_df.set_index("市区町村コード")["リスク区分"].to_dict()
        if "市区町村コード" in classified_df.columns
        else {}
    )
    trend_lookup = (
        classified_df.set_index("市区町村コード")["トレンド"].to_dict()
        if "市区町村コード" in classified_df.columns
        else {}
    )

    if map_option == "2023年空き家率（実測）":
        map_lookup = (
            classified_df.set_index("市区町村コード")["空き家率_2023"].to_dict()
            if "市区町村コード" in classified_df.columns
            else {}
        )
        legend = "2023年空き家率 (%)"
    elif map_option == "2023年空き家率（予測）":
        map_lookup = (
            classified_df.set_index("市区町村コード")["pred_空き家率_2023"].to_dict()
            if "市区町村コード" in classified_df.columns
            else {}
        )
        legend = "予測空き家率 (%)"
    elif map_option == "残差（実測−予測）":
        map_lookup = (
            classified_df.set_index("市区町村コード")["残差(実測-予測)"].to_dict()
            if "市区町村コード" in classified_df.columns
            else {}
        )
        legend = "残差 (pt)"
    else:
        map_lookup = {}
        legend = ""

    update_geojson_dynamic_properties(
        static_geojson, risk_lookup, trend_lookup, map_lookup
    )
    folium_map = build_map(
        static_geojson, map_option, legend, risk_lookup, trend_lookup, map_lookup
    )

    map_col, inspector_col = st.columns((2.2, 1.0))

    with map_col:
        if folium_map is None:
            st.warning(
                "Folium または GeoJSON が利用できないため、地図は表示できません。"
            )
            map_state = {}
        else:
            map_state = st_folium(
                folium_map,
                height=650,
                use_container_width=True,
                key="akiya_map",
            )

    selected_code = st.session_state.get("selected_code")
    props = None
    if map_state:
        drawing = map_state.get("last_active_drawing") or map_state.get(
            "last_object_clicked"
        )
        if drawing and isinstance(drawing, dict):
            props = drawing.get("properties", {})
    if props:
        code = props.get("akiya_code") or props.get("市区町村コード")
        if code:
            selected_code = str(code).zfill(5)
            st.session_state["selected_code"] = selected_code

    with inspector_col:
        st.subheader("自治体インスペクタ")
        if not selected_code:
            st.info("地図上の自治体をクリックすると詳細を表示します。")
        else:
            row = classified_df[
                classified_df["市区町村コード"].astype(str).str.zfill(5)
                == selected_code
            ]
            if row.empty:
                st.warning("選択した自治体のデータが見つかりません。")
            else:
                record = row.iloc[0]
                st.markdown(f"### {record.get('市区町村名', '不明')} ({selected_code})")
                st.markdown(
                    f"- **空き家率** 2018: {record.get('空き家率_2018', np.nan):.2f}% / 2023: {record.get('空き家率_2023', np.nan):.2f}%"
                )
                st.markdown(
                    f"- **Δ(23-18):** {record.get('Δ(23-18)', np.nan):.2f} pt | **リスク:** {risk_lookup.get(selected_code, DEFAULT_RISK_LABEL)} | **トレンド:** {trend_lookup.get(selected_code, '')}"
                )
                st.markdown(
                    f"- **予測空き家率(2023):** {record.get('pred_空き家率_2023', np.nan):.2f}% | **残差:** {record.get('残差(実測-予測)', np.nan):.2f} pt"
                )
                top_factors = shap_lookup.get(selected_code)
                st.markdown("- **要因Top3**")
                if top_factors:
                    for item in top_factors:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("  - SHAP情報なし")

    st.subheader("自治体別テーブル")
    table_cols = [
        "市区町村コード",
        "市区町村名",
        "都道府県名",
        "空き家率_2018",
        "空き家率_2023",
        "Δ(23-18)",
        "リスク区分",
        "トレンド",
        "pred_空き家率_2023",
        "残差(実測-予測)",
    ]
    existing_cols = [col for col in table_cols if col in classified_df.columns]
    table_data = classified_df[existing_cols].copy()
    if "リスク区分" in table_data.columns:
        table_data = table_data.sort_values("リスク区分")
    st.dataframe(table_data, use_container_width=True)

    if info_messages:
        st.subheader("ログ / 注意事項")
        for msg in info_messages:
            if not msg:
                continue
            st.markdown(f"- {msg}")

    st.caption("データソース: data/processed/features_master__wide__v1.csv 他")


if __name__ == "__main__":
    main()
