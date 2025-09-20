#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create wide fiscal feature comparison between 2018 and 2023 with change metrics."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "fiscal_features.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "etl_project.yaml"

TARGET_COLUMNS: Tuple[str, ...] = (
    "地方税",
    "地方交付税",
    "民生費",
    "教育費",
    "土木費",
    "人件費",
    "公債費",
    "扶助費",
    "地方債現在高",
)

BASE_COLUMNS: Tuple[str, ...] = (
    "年度",
    "市区町村コード",
    "市区町村名",
    "住民基本台帳登載人口",
)


def _detect_project_version(path: Path) -> Optional[str]:
    try:
        from scripts.utils.config_loader import load_yaml  # type: ignore

        config = load_yaml(path)
        version = config.get("project", {}).get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        pass

    try:
        with path.open("r", encoding="utf-8") as handle:
            inside_project = False
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("project:"):
                    inside_project = True
                    continue
                if inside_project and not line.startswith(" "):
                    inside_project = False
                if inside_project and stripped.startswith("version:"):
                    _, _, value = stripped.partition(":")
                    value = value.strip().strip('"').strip("'")
                    if value:
                        return value
    except FileNotFoundError:
        return None
    except Exception:
        pass
    return None


def _safe_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_fiscal_rows(
    path: Path,
) -> Tuple[Dict[str, Dict[int, Dict[str, Optional[float]]]], List[int]]:
    result: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    year_set: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in [*BASE_COLUMNS, *TARGET_COLUMNS]
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"必要な列が存在しません: {missing}")
        for row in reader:
            year_raw = row.get("年度")
            if not year_raw:
                continue
            try:
                year = int(float(year_raw))
            except ValueError:
                continue
            code = row["市区町村コード"].strip()
            if not code:
                continue
            store = result.setdefault(code, {})
            year_set.add(year)
            year_data = store.setdefault(year, {})
            for column in BASE_COLUMNS:
                year_data[column] = row.get(column)
            for column in TARGET_COLUMNS:
                year_data[column] = _safe_float(row.get(column))
    years_sorted = sorted(year_set)
    return result, years_sorted


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return format(value, ".15g")


def compute_changes(
    data: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    base_year: int,
    target_year: int,
) -> List[List[str]]:
    header: List[str] = [
        "市区町村コード",
        f"市区町村名_{base_year}",
        f"市区町村名_{target_year}",
        f"住民基本台帳登載人口_{base_year}",
        f"住民基本台帳登載人口_{target_year}",
    ]
    for col in TARGET_COLUMNS:
        header.extend(
            [
                f"{col}_{base_year}",
                f"{col}_{target_year}",
                f"{col}_欠損",
                f"{col}_変化量",
                f"{col}_変化率",
                f"{col}_ログ差分",
            ]
        )

    rows: List[List[str]] = [header]

    for code in sorted(data.keys()):
        year_info = data[code]
        row_base = year_info.get(base_year, {})
        row_target = year_info.get(target_year, {})

        output: List[str] = [
            code,
            (row_base.get("市区町村名") or row_target.get("市区町村名") or ""),
            (row_target.get("市区町村名") or row_base.get("市区町村名") or ""),
            (row_base.get("住民基本台帳登載人口") or ""),
            (row_target.get("住民基本台帳登載人口") or ""),
        ]

        for col in TARGET_COLUMNS:
            val_base = row_base.get(col)
            val_target = row_target.get(col)
            diff = None
            ratio = None
            logdiff = None
            missing_flag = "1"
            if isinstance(val_base, float) and isinstance(val_target, float):
                missing_flag = "0"
                diff = val_target - val_base
                if val_base != 0:
                    ratio = diff / val_base
                if val_base > 0 and val_target > 0:
                    logdiff = math.log(val_target) - math.log(val_base)
            output.extend(
                [
                    _fmt(val_base),
                    _fmt(val_target),
                    missing_flag,
                    _fmt(diff),
                    _fmt(ratio),
                    _fmt(logdiff),
                ]
            )
        rows.append(output)

    return rows


def write_csv(rows: List[List[str]], path: Path, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fiscal comparison dataset for 2018 vs 2023."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Input fiscal features CSV (default: data/processed/fiscal_features.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, a filename containing the project version and years is used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Project config YAML to detect version (default: config/etl_project.yaml).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for output CSV (default: utf-8-sig).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, years_available = load_fiscal_rows(args.source)
    if not years_available:
        raise ValueError("入力データに年度が含まれていません。")
    target_year = years_available[-1]
    base_year = target_year - 5
    if base_year not in years_available:
        raise ValueError(
            f"比較対象の年度が不足しています: 最新年度 {target_year} と 5 年前 {base_year} を抽出できません。"
        )

    filtered: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for code, per_year in data.items():
        keep = {
            year: payload
            for year, payload in per_year.items()
            if year in {base_year, target_year}
        }
        if keep:
            filtered[code] = keep

    rows = compute_changes(filtered, base_year, target_year)
    output_path = args.output
    if output_path is None:
        version = _detect_project_version(args.config) or "unknown"
        years_label = f"{base_year}_{target_year}"
        filename = f"fiscal_features__wide__delta__{version}__{years_label}.csv"
        output_path = PROJECT_ROOT / "data" / "processed" / filename
    write_csv(rows, output_path, args.encoding)
    print(f"Saved fiscal comparison to {output_path}")


if __name__ == "__main__":
    main()
