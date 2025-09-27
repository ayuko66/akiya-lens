"""Streamlit interface for the Akiya-Lens residual model.

This application follows the specification in ``docs/残差モデルとアプリ設計仕様.md``.
このアプリケーションは ``docs/残差モデルとアプリ設計仕様.md`` の仕様に従います。
マスター特徴量テーブルをロードし、ベースライン + 残差パイプラインを再構築し、
リスクカテゴリ、予測、SHAP由来の説明を含むインタラクティブな市町村マップをレンダリングします。

このアプリは、オプションの依存関係や
成果物（CatBoostモデル、GeoJSON、SHAP値）が欠落している場合でも、適切に機能するように設計されています。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:  # Optional imports – app remains usable without them
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


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data/processed/features_master__wide__v1.csv"
GEOJSON_PATH = REPO_ROOT / "data/geojson/municipalities.geojson"
MODEL_PATH = REPO_ROOT / "models/catboost_residual_model.cbm"
METRICS_PATH = REPO_ROOT / "data/processed/model_metrics.json"

DEFAULT_TOLERANCE = 0.1  # percentage point threshold for "横ばい"
MAP_OPTIONS = (
    "4段階リスク",
    "2023年空き家率（実測）",
    "2023年空き家率（予測）",
    "残差（実測−予測）",
)

RISK_COLORS = {
    "赤(最優先)": "#d73027",
    "オレンジ(注意)": "#fc8d59",
    "黄(警戒)": "#fee08b",
    "緑(低)": "#1a9850",
}

NAN_COLOR = "#d9d9d9"


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {path}")
    df = pd.read_csv(path)
    df = _normalise_columns(df)
    df["市区町村コード"] = (
        df["市区町村コード"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
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
# Feature utilities
# ---------------------------------------------------------------------------


def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("‰", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_model_feature_matrix(
    df: pd.DataFrame, feature_names: Iterable[str]
) -> Tuple[pd.DataFrame, List[str]]:
    feature_list = list(feature_names)
    matrix = pd.DataFrame(index=df.index)
    missing: List[str] = []

    for name in feature_list:
        if name in df.columns:
            series = df[name]
            if pd.api.types.is_numeric_dtype(series):
                matrix[name] = series
            else:
                numeric = (
                    series.astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("‰", "", regex=False)
                )
                numeric = pd.to_numeric(numeric, errors="coerce")
                if numeric.notna().sum() >= series.notna().sum() * 0.5:
                    matrix[name] = numeric
                else:
                    matrix[name] = series
        else:
            missing.append(name)
            matrix[name] = pd.Series(np.nan, index=df.index)

    return matrix[feature_list], missing


# ---------------------------------------------------------------------------
# Prediction & SHAP utilities
# ---------------------------------------------------------------------------


def _fit_baseline(baseline: pd.Series, target: pd.Series) -> tuple[float, float]:
    valid_mask = baseline.notna() & target.notna()
    if valid_mask.sum() < 2:
        return 0.0, 0.0
    x = baseline.loc[valid_mask].to_numpy(dtype=float)
    y = target.loc[valid_mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_predictions(
    df: pd.DataFrame,
    model: Optional[CatBoostRegressor],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    messages: List[str] = []

    vac18 = _get_numeric_series(df, "空き家率_2018")
    vac23 = _get_numeric_series(df, "空き家率_2023")
    delta_observed = vac23 - vac18

    slope, intercept = _fit_baseline(vac18, delta_observed)
    base_pred = vac18 * slope + intercept

    feature_matrix: Optional[pd.DataFrame] = None
    resid_pred = pd.Series(0.0, index=df.index, dtype=float)

    if model is not None and hasattr(model, "feature_names_"):
        feature_names = list(model.feature_names_)
        if feature_names:
            feature_matrix, missing = build_model_feature_matrix(df, feature_names)
            if missing:
                messages.append(
                    "学習済みモデルで使用した列が一部見つかりませんでした: "
                    + ", ".join(missing[:5])
                    + ("..." if len(missing) > 5 else "")
                )
            try:
                resid_pred = pd.Series(
                    model.predict(feature_matrix),
                    index=feature_matrix.index,
                    dtype=float,
                )
            except Exception as exc:  # pragma: no cover - defensive
                messages.append(f"CatBoost予測に失敗しました: {exc}")
                resid_pred = pd.Series(0.0, index=df.index, dtype=float)
                feature_matrix = None
        else:
            messages.append("CatBoostモデルに特徴量名が含まれていません。")
    else:
        messages.append(
            "CatBoostモデルが読み込めなかったため、残差補正なしで表示します。"
        )

    pred_delta = base_pred + resid_pred
    pred_vacancy_2023 = vac18 + pred_delta
    residual_gap = delta_observed - pred_delta

    enriched = df.copy()
    enriched["Δ(23-18)"] = delta_observed
    enriched["baseline_pred_delta"] = base_pred
    enriched["residual_model_pred"] = resid_pred
    enriched["pred_delta"] = pred_delta
    enriched["pred_空き家率_2023"] = pred_vacancy_2023
    enriched["残差(実測-予測)"] = residual_gap

    return enriched, feature_matrix, messages


def compute_shap_topk(
    model: Optional[CatBoostRegressor],
    features: Optional[pd.DataFrame],
    k: int = 3,
) -> Tuple[dict[str, list[str]], Optional[str]]:
    if model is None or features is None or features.empty or Pool is None:
        return {}, None

    try:
        pool = Pool(features)
        shap_values = model.get_feature_importance(pool, type="ShapValues")
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"SHAP値の計算に失敗しました: {exc}"

    shap_matrix = pd.DataFrame(
        shap_values[:, :-1], index=features.index, columns=features.columns
    )

    top_lookup: dict[str, list[str]] = {}
    for idx, row in shap_matrix.iterrows():
        top = row.abs().sort_values(ascending=False).head(k)
        formatted = [f"{feat}: {row[feat]:+.3f}" for feat in top.index]
        top_lookup[str(idx)] = formatted
    return top_lookup, None


# ---------------------------------------------------------------------------
# Risk classification & geo utilities
# ---------------------------------------------------------------------------


def classify_risk(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    df = df.copy()
    vac18 = _get_numeric_series(df, "空き家率_2018")
    vac23 = _get_numeric_series(df, "空き家率_2023")
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
        if is_high and trend_value == "増":
            labels.append("赤(最優先)")
        elif is_high:
            labels.append("オレンジ(注意)")
        elif trend_value == "増":
            labels.append("黄(警戒)")
        else:
            labels.append("緑(低)")

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


def augment_geojson(
    geojson: Optional[dict[str, Any]],
    df: pd.DataFrame,
    shap_lookup: dict[str, list[str]],
) -> Optional[dict[str, Any]]:
    if geojson is None:
        return None

    geojson_copy = json.loads(json.dumps(geojson))
    dedup_df = (
        df.dropna(subset=["市区町村コード"])
        .drop_duplicates(subset=["市区町村コード"], keep="last")
        .set_index("市区町村コード")
    )
    data_lookup = dedup_df.to_dict("index")

    for feature in geojson_copy.get("features", []):
        props = feature.setdefault("properties", {})
        code = str(props.get("市区町村コード", "")).zfill(5)
        record = data_lookup.get(code)
        props["akiya_code"] = code
        if record is None:
            props["akiya_has_data"] = False
            continue

        props["akiya_has_data"] = True
        props["akiya_name"] = record.get("市区町村名", "不明")
        props["akiya_vac18"] = _safe_value(record.get("空き家率_2018"))
        props["akiya_vac23"] = _safe_value(record.get("空き家率_2023"))
        props["akiya_delta"] = _safe_value(record.get("Δ(23-18)"))
        props["akiya_risk"] = record.get("リスク区分")
        props["akiya_pred"] = _safe_value(record.get("pred_空き家率_2023"))
        props["akiya_residual"] = _safe_value(record.get("残差(実測-予測)"))
        props["akiya_trend"] = record.get("トレンド")
        internal_key = str(record.get("_akiya_internal_index", code))
        shap_items = shap_lookup.get(internal_key, [])
        props["akiya_shap"] = _format_shap_text(shap_items)
        props["akiya_map"] = _safe_value(
            record.get("map_value") if "map_value" in record else np.nan
        )

    return geojson_copy


def _safe_value(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_map(
    geojson: Optional[dict[str, Any]],
    df: pd.DataFrame,
    map_option: str,
    legend_name: str,
) -> Optional[folium.Map]:
    if geojson is None or folium is None or st_folium is None:
        return None

    m = folium.Map(location=[35.6, 137.8], zoom_start=6, tiles="cartodbpositron")

    def tooltip_fields() -> list[str]:
        return [
            "akiya_name",
            "akiya_vac18",
            "akiya_vac23",
            "akiya_delta",
            "akiya_risk",
            "akiya_trend",
        ]

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

    popup = features.GeoJsonPopup(
        fields=["akiya_name", "akiya_shap", "akiya_pred", "akiya_residual"],
        aliases=[
            "自治体",
            "SHAP Top3",
            "予測空き家率(2023)%",
            "残差(実測-予測)pt",
        ],
        localize=True,
    )

    if map_option == "4段階リスク":
        color_map = RISK_COLORS

        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature.get("properties", {})
            if not props.get("akiya_has_data"):
                return {
                    "fillColor": NAN_COLOR,
                    "color": "#666666",
                    "weight": 0.2,
                    "fillOpacity": 0.5,
                }
            risk = props.get("akiya_risk")
            color = color_map.get(risk, NAN_COLOR)
            return {
                "fillColor": color,
                "color": "#4d4d4d",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }

        folium.GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=lambda _: {"weight": 1.5, "color": "#000000"},
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

        if Template is not None and MacroElement is not None:
            legend_template = """
            {% macro html(this, kwargs) %}
            <div style="position: fixed; bottom: 30px; left: 30px; width: 220px;
                        background-color: #ffffff; border: 1px solid #999999;
                        z-index: 9999; font-size: 13px; padding: 10px;
                        color: #333333; box-shadow: 2px 2px 4px rgba(0,0,0,0.25);">
              <b style="display:block; margin-bottom:6px;">4段階リスク</b>
              <div style="display:flex; align-items:center; margin-bottom:4px;">
                <span style="display:inline-block;width:14px;height:14px;background:#d73027;border:1px solid #555;"></span>
                <span style="margin-left:8px;">赤(最優先)</span>
              </div>
              <div style="display:flex; align-items:center; margin-bottom:4px;">
                <span style="display:inline-block;width:14px;height:14px;background:#fc8d59;border:1px solid #555;"></span>
                <span style="margin-left:8px;">オレンジ(注意)</span>
              </div>
              <div style="display:flex; align-items:center; margin-bottom:4px;">
                <span style="display:inline-block;width:14px;height:14px;background:#fee08b;border:1px solid #555;"></span>
                <span style="margin-left:8px;">黄(警戒)</span>
              </div>
              <div style="display:flex; align-items:center;">
                <span style="display:inline-block;width:14px;height:14px;background:#1a9850;border:1px solid #555;"></span>
                <span style="margin-left:8px;">緑(低)</span>
              </div>
            </div>
            {% endmacro %}
            """
            legend = MacroElement()
            legend._template = Template(legend_template)
            m.get_root().add_child(legend)
    else:
        values = df["map_value"].dropna().to_numpy(dtype=float)
        if values.size == 0 or branca_linear is None:
            colormap = None
        else:
            vmin, vmax = float(np.min(values)), float(np.max(values))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            colormap = branca_linear.YlOrRd_09.scale(vmin, vmax)
            colormap.caption = legend_name
            colormap.add_to(m)

        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature.get("properties", {})
            if not props.get("akiya_has_data"):
                return {
                    "fillColor": NAN_COLOR,
                    "color": "#666666",
                    "weight": 0.2,
                    "fillOpacity": 0.5,
                }
            value = props.get("akiya_map")
            if value is None or colormap is None:
                color = NAN_COLOR
            else:
                color = colormap(value)
            return {
                "fillColor": color,
                "color": "#4d4d4d",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }

        folium.GeoJson(
            geojson,
            style_function=style_function,
            highlight_function=lambda _: {"weight": 1.5, "color": "#000000"},
            tooltip=tooltip,
            popup=popup,
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

    enriched_df, feature_matrix, info_messages = compute_predictions(features_df, model)
    enriched_df = classify_risk(enriched_df, tol)

    shap_lookup, shap_message = compute_shap_topk(model, feature_matrix)
    if shap_message:
        info_messages.append(shap_message)

    display_df = enriched_df.copy()
    display_df["_akiya_internal_index"] = display_df.index.astype(str)

    if map_option == "2023年空き家率（実測）":
        display_df["map_value"] = _get_numeric_series(display_df, "空き家率_2023")
        legend = "2023年空き家率 (%)"
    elif map_option == "2023年空き家率（予測）":
        display_df["map_value"] = display_df.get("pred_空き家率_2023")
        legend = "予測空き家率 (%)"
    elif map_option == "残差（実測−予測）":
        display_df["map_value"] = display_df.get("残差(実測-予測)")
        legend = "残差 (pt)"
    else:
        display_df["map_value"] = np.nan
        legend = ""

    geojson_aug = augment_geojson(geojson, display_df, shap_lookup)
    folium_map = build_map(geojson_aug, display_df, map_option, legend)

    if folium_map is None:
        st.warning("Folium または GeoJSON が利用できないため、地図は表示できません。")
    else:
        st_folium(folium_map, height=650, use_container_width=True)

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
    existing_cols = [col for col in table_cols if col in display_df.columns]
    table_data = display_df[existing_cols].copy()
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
