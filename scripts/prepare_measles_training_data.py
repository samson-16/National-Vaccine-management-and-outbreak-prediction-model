"""Build a woreda-period measles training panel from an APHI line list.

The default workflow treats each row in the input file as one measles case,
aggregates cases to woreda-month or woreda-week, creates lag features, and
labels whether the next period crosses the outbreak threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

WOREDA_CANDIDATES = [
    "woreda",
    "woreda_hospital",
    "woreda_name",
    "district",
    "admin3",
    "admin_3",
    "admin3_name",
]

DATE_CANDIDATES = [
    "date_onset_of_disease",
    "date_of_onset",
    "onset_date",
    "date_onset",
    "disease_onset_date",
    "date_seen_at_health_facilities",
    "date_seen_at_health_facility",
    "date_seen",
    "report_date",
]

CASE_COUNT_CANDIDATES = [
    "measles_case",
    "measles_cases",
    "cases",
    "case_count",
    "total_cases",
]


def normalize_column(name: object) -> str:
    value = str(name).strip().lower()
    value = value.replace("%", " percent ")
    value = value.replace("&", " and ")
    value = re.sub(r"[/\\]+", "_", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in df.columns:
        normalized = normalize_column(column)
        count = seen.get(normalized, 0)
        seen[normalized] = count + 1
        columns.append(normalized if count == 0 else f"{normalized}_{count + 1}")
    df = df.copy()
    df.columns = columns
    return df


def read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def resolve_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    label: str,
) -> str:
    if explicit:
        normalized = normalize_column(explicit)
        if normalized in df.columns:
            return normalized
        raise ValueError(f"{label} column '{explicit}' was not found after normalization.")

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for candidate in candidates:
        matches = [column for column in df.columns if candidate in column]
        if matches:
            return matches[0]

    preview = ", ".join(df.columns[:40])
    raise ValueError(
        f"Could not infer {label} column. Use --{label.replace(' ', '-')}-col. "
        f"Available columns include: {preview}"
    )


def parse_periods(series: pd.Series, time_unit: str) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if time_unit == "month":
        return dates.dt.to_period("M").dt.to_timestamp()
    if time_unit == "week":
        return dates.dt.to_period("W-SUN").apply(lambda period: period.start_time)
    raise ValueError(f"Unsupported time unit: {time_unit}")


def aggregate_cases(
    df: pd.DataFrame,
    woreda_col: str,
    date_col: str,
    time_unit: str,
    case_count_col: str | None,
) -> pd.DataFrame:
    working = df[[woreda_col, date_col] + ([case_count_col] if case_count_col else [])].copy()
    working[woreda_col] = working[woreda_col].astype(str).str.strip()
    working["period_start"] = parse_periods(working[date_col], time_unit)
    working = working[
        working[woreda_col].notna()
        & (working[woreda_col] != "")
        & (working[woreda_col].str.lower() != "nan")
        & working["period_start"].notna()
    ]

    if case_count_col:
        working[case_count_col] = pd.to_numeric(working[case_count_col], errors="coerce").fillna(0)
        cases = (
            working.groupby([woreda_col, "period_start"], as_index=False)[case_count_col]
            .sum()
            .rename(columns={case_count_col: "cases"})
        )
    else:
        cases = (
            working.groupby([woreda_col, "period_start"], as_index=False)
            .size()
            .rename(columns={"size": "cases"})
        )

    return cases.rename(columns={woreda_col: "woreda"})


def complete_panel(cases: pd.DataFrame, time_unit: str) -> pd.DataFrame:
    frequency = "MS" if time_unit == "month" else "W-MON"
    start = cases["period_start"].min()
    end = cases["period_start"].max()
    periods = pd.date_range(start=start, end=end, freq=frequency)
    woredas = sorted(cases["woreda"].dropna().unique())
    panel_index = pd.MultiIndex.from_product(
        [woredas, periods], names=["woreda", "period_start"]
    )
    panel = (
        cases.set_index(["woreda", "period_start"])
        .reindex(panel_index, fill_value=0)
        .reset_index()
    )
    panel["cases"] = panel["cases"].astype(int)
    return panel


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["year"] = panel["period_start"].dt.year
    panel["month"] = panel["period_start"].dt.month
    panel["iso_week"] = panel["period_start"].dt.isocalendar().week.astype(int)
    return panel


def add_lag_features(panel: pd.DataFrame, lags: list[int], horizon: int, threshold: int) -> pd.DataFrame:
    panel = panel.sort_values(["woreda", "period_start"]).copy()
    grouped = panel.groupby("woreda", group_keys=False)

    for lag in lags:
        panel[f"cases_lag_{lag}"] = grouped["cases"].shift(lag)

    panel["cases_rolling_3_prev"] = grouped["cases"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    panel["cases_rolling_6_prev"] = grouped["cases"].transform(
        lambda series: series.shift(1).rolling(6, min_periods=1).mean()
    )
    panel["cases_cumulative_prev"] = grouped["cases"].transform(lambda series: series.cumsum().shift(1))

    future_columns = []
    for step in range(1, horizon + 1):
        column = f"cases_future_{step}"
        panel[column] = grouped["cases"].shift(-step)
        future_columns.append(column)

    target_col = f"future_cases_h{horizon}"
    panel[target_col] = panel[future_columns].sum(axis=1, min_count=horizon)
    panel[f"outbreak_next_h{horizon}"] = (panel[target_col] >= threshold).astype("Int64")
    panel.loc[panel[target_col].isna(), f"outbreak_next_h{horizon}"] = pd.NA

    panel = panel.drop(columns=future_columns)
    return panel


def load_covariate(path: Path, time_unit: str) -> pd.DataFrame:
    cov = normalize_columns(read_table(path))
    woreda_col = resolve_column(cov, None, WOREDA_CANDIDATES, "woreda")
    cov = cov.rename(columns={woreda_col: "woreda"})
    cov["woreda"] = cov["woreda"].astype(str).str.strip()

    date_col = next((column for column in DATE_CANDIDATES + ["period_start", "period"] if column in cov.columns), None)
    if date_col:
        cov["period_start"] = parse_periods(cov[date_col], time_unit)
        cov = cov.drop(columns=[date_col])
    elif {"year", "month"}.issubset(cov.columns):
        cov["period_start"] = pd.to_datetime(
            cov["year"].astype(str) + "-" + cov["month"].astype(str) + "-01",
            errors="coerce",
        )

    prefix = normalize_column(path.stem)
    key_cols = ["woreda"] + (["period_start"] if "period_start" in cov.columns else [])
    rename_map = {
        column: f"{prefix}_{column}"
        for column in cov.columns
        if column not in key_cols
    }
    cov = cov.rename(columns=rename_map)
    return cov


def merge_covariates(panel: pd.DataFrame, covariate_paths: list[Path], time_unit: str) -> pd.DataFrame:
    merged = panel.copy()
    for path in covariate_paths:
        cov = load_covariate(path, time_unit)
        keys = ["woreda", "period_start"] if "period_start" in cov.columns else ["woreda"]
        merged = merged.merge(cov, on=keys, how="left")
    return merged


def write_data_dictionary(path: Path, columns: list[str], horizon: int) -> None:
    descriptions = {
        "woreda": "Woreda name as provided in the source file.",
        "period_start": "Start date of the aggregation period.",
        "cases": "Observed measles case count in this woreda-period.",
        "year": "Calendar year of period_start.",
        "month": "Calendar month of period_start.",
        "iso_week": "ISO week of period_start.",
        "cases_rolling_3_prev": "Mean observed cases in the previous 3 periods.",
        "cases_rolling_6_prev": "Mean observed cases in the previous 6 periods.",
        "cases_cumulative_prev": "Cumulative observed cases before this period.",
        f"future_cases_h{horizon}": f"Total cases over the next {horizon} period(s).",
        f"outbreak_next_h{horizon}": "Binary target: 1 if future cases meet/exceed the outbreak threshold.",
    }
    rows = []
    for column in columns:
        if column.startswith("cases_lag_"):
            description = f"Observed cases {column.removeprefix('cases_lag_')} period(s) before this period."
        else:
            description = descriptions.get(column, "Source or covariate feature.")
        rows.append({"column": column, "description": description})
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_lags(value: str) -> list[int]:
    lags = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lags or any(lag < 1 for lag in lags):
        raise argparse.ArgumentTypeError("Lags must be a comma-separated list of positive integers.")
    return lags


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a woreda-period measles training panel.")
    parser.add_argument("--input", required=True, type=Path, help="APHI line-list CSV/XLSX/parquet file.")
    parser.add_argument("--sheet", help="Excel sheet name or index. Defaults to the first sheet.")
    parser.add_argument("--time-unit", choices=["month", "week"], default="month")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "amhara_woreda_month_training.csv")
    parser.add_argument("--woreda-col", help="Source woreda column name if inference fails.")
    parser.add_argument("--date-col", help="Source onset/reporting date column name if inference fails.")
    parser.add_argument("--case-count-col", help="Use this numeric case-count column instead of row counts.")
    parser.add_argument("--horizon", type=int, default=1, help="Number of future periods in the target window.")
    parser.add_argument("--outbreak-threshold", type=int, default=5, help="Case threshold for outbreak_next_h.")
    parser.add_argument("--lags", type=parse_lags, default=parse_lags("1,2,3"))
    parser.add_argument(
        "--covariate",
        action="append",
        type=Path,
        default=[],
        help="Optional woreda-level or woreda-period CSV/XLSX/parquet covariate file. Can be repeated.",
    )
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be at least 1")

    try:
        sheet: str | int | None = args.sheet
        if isinstance(sheet, str) and sheet.isdigit():
            sheet = int(sheet)
        raw = normalize_columns(read_table(args.input, sheet=sheet))
        woreda_col = resolve_column(raw, args.woreda_col, WOREDA_CANDIDATES, "woreda")
        date_col = resolve_column(raw, args.date_col, DATE_CANDIDATES, "date")
        case_count_col = (
            resolve_column(raw, args.case_count_col, CASE_COUNT_CANDIDATES, "case count")
            if args.case_count_col
            else None
        )

        cases = aggregate_cases(raw, woreda_col, date_col, args.time_unit, case_count_col)
        if cases.empty:
            raise ValueError("No valid rows remained after woreda/date cleaning.")

        panel = complete_panel(cases, args.time_unit)
        panel = add_time_features(panel)
        panel = add_lag_features(panel, args.lags, args.horizon, args.outbreak_threshold)
        if args.covariate:
            panel = merge_covariates(panel, args.covariate, args.time_unit)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.out, index=False)

        dictionary_path = args.out.with_name(args.out.stem + "_data_dictionary.csv")
        write_data_dictionary(dictionary_path, list(panel.columns), args.horizon)

        summary = {
            "input": str(args.input),
            "output": str(args.out),
            "data_dictionary": str(dictionary_path),
            "rows": int(len(panel)),
            "woredas": int(panel["woreda"].nunique()),
            "periods": int(panel["period_start"].nunique()),
            "time_unit": args.time_unit,
            "target": f"outbreak_next_h{args.horizon}",
            "outbreak_threshold": args.outbreak_threshold,
            "date_column": date_col,
            "woreda_column": woreda_col,
        }
        summary_path = args.out.with_name(args.out.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Wrote {args.out}")
        print(f"Wrote {dictionary_path}")
        print(f"Wrote {summary_path}")
        print(f"Rows: {summary['rows']} | Woredas: {summary['woredas']} | Periods: {summary['periods']}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
