#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build fiscal feature dataset from Ministry of Internal Affairs fiscal tables."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_COLUMNS = [
    ("year", "年度"),
    ("municipality_code", "市区町村コード"),
    ("municipality_name", "市区町村名"),
    ("population", "住民基本台帳登載人口"),
    ("fiscal_index", "財政力指数"),
    ("fiscal_index_winsorized", "財政力指数_ロバスト"),
    ("current_balance_ratio", "経常収支比率"),
    ("current_balance_ratio_winsorized", "経常収支比率_ロバスト"),
    ("real_debt_service_ratio", "実質公債費比率"),
    ("future_burden_ratio", "将来負担比率"),
    ("standard_fiscal_needs", "基準財政需要額"),
    ("standard_fiscal_revenue", "基準財政収入額"),
    ("standard_fiscal_size", "標準財政規模"),
    ("local_tax", "地方税"),
    ("local_tax_per_capita", "人口あたり地方税"),
    ("local_allocation_tax", "地方交付税"),
    ("welfare_expenses", "民生費"),
    ("welfare_share", "民生費割合"),
    ("education_expenses", "教育費"),
    ("education_share", "教育費割合"),
    ("public_works_expenses", "土木費"),
    ("public_works_share", "土木費割合"),
    ("personnel_expenses", "人件費"),
    ("debt_service_expenses", "公債費"),
    ("assistance_expenses", "扶助費"),
    ("outstanding_local_bonds", "地方債現在高"),
    ("bonds_per_capita", "人口あたり地方債現在高"),
    ("expenditure_total", "歳出総額"),
    ("grant_dependency_ratio", "交付税依存度"),
]

YEAR_FILES = {
    2018: {
        "overview": "2018_市町村別決算状況_概況_000676025.csv",
        "revenue": "2018_市町村別決算概況_歳入内訳_000676026.csv",
        "purpose": "2018_市町村別決算概況_目的別歳出内訳_000676027.csv",
        "nature": "2018_市町村別決算状況_性質別歳出内訳_000676028.csv",
        "bonds": "2018_市町村別決算状況_地方債現在高等_000676029.csv",
    },
    2023: {
        "overview": "2023_市町村別決算状況_概況_000999905.csv",
        "revenue": "2023_市町村別決算状況_歳入内訳_000999906.csv",
        "purpose": "2023_市町村別決算状況_目的別歳出内訳_000999908.csv",
        "nature": "2023_市町村別決算状況_性質別歳出内訳_000999910.csv",
        "bonds": "2023_市町村別決算状況_地方債現在高等_000999911.csv",
    },
}


def _normalize_code(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    code = str(value).strip().replace("\u3000", "").replace("-", "")
    if not code:
        return None
    if not code.isdigit():
        return None
    if len(code) == 6:
        core = code[:-1]
        if core.endswith("000"):
            return None
        return core
    if len(code) == 5:
        # Files for Hokkaido lose the leading zero and retain the check digit.
        core = "0" + code[:-1]
        if core.endswith("000"):
            return None
        return core
    return None


def _clean_numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "nan", "NaN", "None"}:
        return None
    text = text.replace("\u3000", "").replace(",", "")
    text = text.replace("△", "-").replace("▲", "-")
    try:
        return float(text)
    except ValueError:
        return None


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() if fn else "" for fn in reader.fieldnames]
        for raw in reader:
            row: Dict[str, str] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                key = key.strip() if isinstance(key, str) else key
                if not key:
                    continue
                if isinstance(value, list):
                    value = value[0] if value else ""
                if value is None:
                    value = ""
                if isinstance(value, str):
                    value = value.strip()
                row[key] = value
            yield row


def _load_table(
    path: Path,
    rename_map: Dict[str, str],
    numeric_cols: Iterable[str],
    keep_name: bool = False,
) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    for row in _iter_rows(path):
        mapped = {new: row.get(old, "") for old, new in rename_map.items()}
        code = _normalize_code(mapped.get("municipality_code"))
        if not code:
            continue
        record: Dict[str, object] = {"municipality_code": code}
        name = mapped.get("municipality_name", "")
        if name and (keep_name or "municipality_name" not in record):
            record["municipality_name"] = str(name).replace("\u3000", "").strip()
        for col in numeric_cols:
            record[col] = _clean_numeric(mapped.get(col))
        records[code] = record
    return records


def _merge(
    base: Dict[str, Dict[str, object]], extra: Dict[str, Dict[str, object]]
) -> None:
    for code, payload in extra.items():
        if code not in base:
            continue
        target = base[code]
        name = payload.get("municipality_name")
        if name and not target.get("municipality_name"):
            target["municipality_name"] = name
        for key, value in payload.items():
            if key in {"municipality_code", "municipality_name"}:
                continue
            if value is not None:
                target[key] = value


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def _add_winsorized(
    records: Dict[str, Dict[str, object]],
    key: str,
    lower_q: float,
    upper_q: float,
    suffix: str,
) -> None:
    values = [
        float(value)
        for value in (record.get(key) for record in records.values())
        if isinstance(value, float)
    ]
    if len(values) < 2:
        for record in records.values():
            record[suffix] = record.get(key)
        return
    values.sort()
    lower = _quantile(values, lower_q)
    upper = _quantile(values, upper_q)
    if lower is None or upper is None:
        for record in records.values():
            record[suffix] = record.get(key)
        return
    for record in records.values():
        value = record.get(key)
        if isinstance(value, float):
            clamped = value
            if value < lower:
                clamped = lower
            elif value > upper:
                clamped = upper
            record[suffix] = clamped
        else:
            record[suffix] = None


def build_fiscal_features_for_year(
    input_dir: Path, year: int
) -> List[Dict[str, object]]:
    spec = YEAR_FILES.get(year)
    if not spec:
        raise ValueError(f"Unsupported year: {year}")

    overview = _load_table(
        input_dir / spec["overview"],
        rename_map={
            "市区町村コード": "municipality_code",
            "市区町村名": "municipality_name",
            "住民基本台帳登載人口": "population",
            "財政力指数": "fiscal_index",
            "経常収支比率": "current_balance_ratio",
            "実質公債費比率": "real_debt_service_ratio",
            "将来負担比率": "future_burden_ratio",
            "基準財政需要額": "standard_fiscal_needs",
            "基準財政収入額": "standard_fiscal_revenue",
            "標準財政規模": "standard_fiscal_size",
            "歳出総額": "expenditure_total",
            "歳入総額": "revenue_total",
        },
        numeric_cols=[
            "population",
            "fiscal_index",
            "current_balance_ratio",
            "real_debt_service_ratio",
            "future_burden_ratio",
            "standard_fiscal_needs",
            "standard_fiscal_revenue",
            "standard_fiscal_size",
            "expenditure_total",
            "revenue_total",
        ],
        keep_name=True,
    )

    revenue = _load_table(
        input_dir / spec["revenue"],
        rename_map={
            "市区町村コード": "municipality_code",
            "市区町村名": "municipality_name",
            "地方税": "local_tax",
            "地方交付税": "local_allocation_tax",
        },
        numeric_cols=["local_tax", "local_allocation_tax"],
    )

    purpose_exp = _load_table(
        input_dir / spec["purpose"],
        rename_map={
            "市区町村コード": "municipality_code",
            "市区町村名": "municipality_name",
            "民生費": "welfare_expenses",
            "教育費": "education_expenses",
            "土木費": "public_works_expenses",
            "公債費": "debt_service_expenses",
        },
        numeric_cols=[
            "welfare_expenses",
            "education_expenses",
            "public_works_expenses",
            "debt_service_expenses",
        ],
    )

    nature_exp = _load_table(
        input_dir / spec["nature"],
        rename_map={
            "市区町村コード": "municipality_code",
            "市区町村名": "municipality_name",
            "人件費": "personnel_expenses",
            "扶助費": "assistance_expenses",
        },
        numeric_cols=[
            "personnel_expenses",
            "assistance_expenses",
        ],
    )

    bonds = _load_table(
        input_dir / spec["bonds"],
        rename_map={
            "市区町村コード": "municipality_code",
            "市区町村名": "municipality_name",
            "地方債現在高": "outstanding_local_bonds",
        },
        numeric_cols=["outstanding_local_bonds"],
    )

    _merge(overview, revenue)
    _merge(overview, purpose_exp)
    _merge(overview, nature_exp)
    _merge(overview, bonds)

    for record in overview.values():
        record["year"] = int(year)
        needs = record.get("standard_fiscal_needs")
        revenue_std = record.get("standard_fiscal_revenue")
        size = record.get("standard_fiscal_size")
        grant_ratio: Optional[float] = None
        if (
            isinstance(needs, float)
            and isinstance(revenue_std, float)
            and isinstance(size, float)
            and size > 0
        ):
            grant_ratio = (needs - revenue_std) / size * 100
        record["grant_dependency_ratio"] = grant_ratio

        population = record.get("population")
        local_tax = record.get("local_tax")
        local_bonds = record.get("outstanding_local_bonds")
        expenditure_total = record.get("expenditure_total")

        per_capita_tax = None
        per_capita_bonds = None
        if isinstance(population, float) and population > 0:
            if isinstance(local_tax, float):
                per_capita_tax = local_tax / population
            if isinstance(local_bonds, float):
                per_capita_bonds = local_bonds / population
        record["local_tax_per_capita"] = per_capita_tax
        record["bonds_per_capita"] = per_capita_bonds

        for source_key, share_key in [
            ("welfare_expenses", "welfare_share"),
            ("education_expenses", "education_share"),
            ("public_works_expenses", "public_works_share"),
        ]:
            value = record.get(source_key)
            if (
                isinstance(value, float)
                and isinstance(expenditure_total, float)
                and expenditure_total > 0
            ):
                record[share_key] = value / expenditure_total
            else:
                record[share_key] = None

    _add_winsorized(
        overview,
        "current_balance_ratio",
        0.01,
        0.99,
        "current_balance_ratio_winsorized",
    )
    _add_winsorized(overview, "fiscal_index", 0.01, 0.99, "fiscal_index_winsorized")

    field_order = [column for column, _ in OUTPUT_COLUMNS]
    rows: List[Dict[str, object]] = []
    for code in sorted(overview.keys()):
        record = overview[code]
        rows.append({col: record.get(col) for col in field_order})
    return rows


def build_fiscal_features(
    input_dir: Path, years: Iterable[int]
) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for year in years:
        all_rows.extend(build_fiscal_features_for_year(input_dir, int(year)))
    all_rows.sort(key=lambda row: (row.get("year"), row.get("municipality_code")))
    return all_rows


def _format_for_csv(value: object) -> str:
    """出力CSVフォーマット設定"""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return format(value, ".15g")
    return str(value)


def write_csv(rows: List[Dict[str, object]], output_path: Path, encoding: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [label for _, label in OUTPUT_COLUMNS]
    with output_path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [_format_for_csv(row.get(column)) for column, _ in OUTPUT_COLUMNS]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fiscal feature dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "fiscal",
        help="Directory containing raw fiscal CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "fiscal_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding for output CSV (default: utf-8, Excel friendly).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2023],
        help="Fiscal years to process (default: 2018 2023).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_fiscal_features(args.input_dir, args.years)
    write_csv(rows, args.output, args.encoding)
    print(f"Saved fiscal features to {args.output}")


if __name__ == "__main__":
    main()
