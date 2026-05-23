"""Prepare Ethiopia vaccine coverage features from public downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_VACCINE_DIR = ROOT / "data" / "raw" / "vaccine"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DEFAULT_MODEL_MATRIX = PROCESSED_DIR / "measles_training_demo_model_matrix.csv"

MCV1_PDF_FALLBACK = {
    2012: 62,
    2013: 55,
    2014: 54,
    2015: 55,
    2016: 57,
    2017: 58,
    2018: 54,
    2019: 57,
    2020: 59,
    2021: 53,
    2022: 55,
    2023: 61,
}

MCV2_PDF_FALLBACK = {
    2019: 41,
    2020: 46,
    2021: 46,
    2022: 48,
    2023: 53,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-vaccine-dir", type=Path, default=RAW_VACCINE_DIR)
    parser.add_argument("--model-matrix", type=Path, default=DEFAULT_MODEL_MATRIX)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def read_csv_best_effort(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def find_column(columns: list[str], candidates: list[str], contains: list[str] | None = None) -> str | None:
    lower_map = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if contains:
        for col in columns:
            lowered = col.lower()
            if all(token.lower() in lowered for token in contains):
                return col
    return None


def normalize_who_csv(path: Path, antigen: str) -> pd.DataFrame:
    df = read_csv_best_effort(path)
    columns = list(df.columns)
    year_col = find_column(columns, ["YEAR", "Year", "TimeDim", "DIM_TIME", "TIME_PERIOD"], ["year"])
    value_col = find_column(columns, ["Numeric", "NumericValue", "Value", "VALUE", "OBS_VALUE", "FactValueNumeric"])
    country_col = find_column(columns, ["COUNTRY", "Country", "SpatialDim", "SpatialDimValueCode"])
    country_display_col = find_column(
        columns,
        ["COUNTRY_DISPLAY", "Country display", "Location", "SpatialDim", "ParentLocation"],
        ["country", "display"],
    )
    gho_col = find_column(columns, ["GHO", "IndicatorCode", "Indicator"])
    gho_display_col = find_column(columns, ["GHO_DISPLAY", "IndicatorTitle", "Indicator", "Indicator display"], ["gho", "display"])

    if year_col is None or value_col is None:
        raise ValueError(f"Could not identify year/value columns in {path}. Columns: {columns}")

    out = pd.DataFrame(
        {
            "country": "Ethiopia",
            "country_code": "ETH",
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            f"{antigen.lower()}_coverage_real_national": pd.to_numeric(df[value_col], errors="coerce"),
            f"{antigen.lower()}_source_indicator": df[gho_col].astype(str) if gho_col else antigen,
            f"{antigen.lower()}_source_indicator_name": df[gho_display_col].astype(str) if gho_display_col else antigen,
            f"{antigen.lower()}_source_country_code": df[country_col].astype(str) if country_col else "ETH",
            f"{antigen.lower()}_source_country_name": df[country_display_col].astype(str) if country_display_col else "Ethiopia",
        }
    )
    out = out.dropna(subset=["year", f"{antigen.lower()}_coverage_real_national"]).copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values("year")
    out = out.drop_duplicates(subset=["country_code", "year"], keep="last")
    out[f"{antigen.lower()}_coverage_source"] = f"WHO_GHO_Athena_{antigen}"
    out[f"{antigen.lower()}_coverage_level"] = "national_annual_real"
    return out.reset_index(drop=True)


def fallback_from_pdf_values(antigen: str, values: dict[int, int]) -> pd.DataFrame:
    col = f"{antigen.lower()}_coverage_real_national"
    rows = []
    for year, value in sorted(values.items()):
        rows.append(
            {
                "country": "Ethiopia",
                "country_code": "ETH",
                "year": year,
                col: value,
                f"{antigen.lower()}_source_indicator": antigen,
                f"{antigen.lower()}_source_indicator_name": f"{antigen} WUENIC estimate from WHO Ethiopia profile PDF",
                f"{antigen.lower()}_source_country_code": "ETH",
                f"{antigen.lower()}_source_country_name": "Ethiopia",
                f"{antigen.lower()}_coverage_source": "WHO_WUENIC_country_profile_pdf_fallback",
                f"{antigen.lower()}_coverage_level": "national_annual_real_pdf",
            }
        )
    return pd.DataFrame(rows)


def load_antigen(raw_dir: Path, antigen: str, file_name: str, fallback: dict[int, int]) -> tuple[pd.DataFrame, str]:
    path = raw_dir / file_name
    if path.exists() and path.stat().st_size > 0:
        try:
            return normalize_who_csv(path, antigen), "csv"
        except Exception as exc:  # noqa: BLE001
            return fallback_from_pdf_values(antigen, fallback), f"pdf_fallback_after_csv_parse_error: {exc}"
    return fallback_from_pdf_values(antigen, fallback), "pdf_fallback_missing_csv"


def build_annual_table(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    mcv1, mcv1_mode = load_antigen(raw_dir, "MCV1", "who_ethiopia_mcv1_coverage.csv", MCV1_PDF_FALLBACK)
    mcv2, mcv2_mode = load_antigen(raw_dir, "MCV2", "who_ethiopia_mcv2_coverage.csv", MCV2_PDF_FALLBACK)
    annual = mcv1.merge(mcv2, on=["country", "country_code", "year"], how="outer").sort_values("year")
    annual["mcv1_coverage_real_national"] = pd.to_numeric(annual["mcv1_coverage_real_national"], errors="coerce")
    annual["mcv2_coverage_real_national"] = pd.to_numeric(annual["mcv2_coverage_real_national"], errors="coerce")
    annual["mcv1_gap_to_95"] = (95.0 - annual["mcv1_coverage_real_national"]).clip(lower=0)
    annual["mcv2_gap_to_95"] = (95.0 - annual["mcv2_coverage_real_national"]).clip(lower=0)
    annual["mcv1_mcv2_dropout_real_national"] = (
        annual["mcv1_coverage_real_national"] - annual["mcv2_coverage_real_national"]
    )
    annual["effective_immunity_gap_real_national"] = annual[["mcv1_gap_to_95", "mcv2_gap_to_95"]].mean(axis=1)
    annual["vaccine_data_level"] = "national_annual_real"
    annual["vaccine_data_source"] = "WHO_GHO_Athena_or_WHO_WUENIC_profile"
    annual["coverage_real_flag"] = 1
    annual["coverage_imputed_flag"] = 0
    metadata = {
        "mcv1_mode": mcv1_mode,
        "mcv2_mode": mcv2_mode,
        "annual_rows": int(len(annual)),
        "year_min": int(annual["year"].min()) if len(annual) else None,
        "year_max": int(annual["year"].max()) if len(annual) else None,
    }
    return annual.reset_index(drop=True), metadata


def build_model_join(model_path: Path, annual: pd.DataFrame) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model matrix: {model_path}")
    model = pd.read_csv(model_path)
    model["year"] = pd.to_numeric(model["year"], errors="coerce").astype(int)
    numeric_cols = [
        "mcv1_coverage_real_national",
        "mcv2_coverage_real_national",
        "mcv1_gap_to_95",
        "mcv2_gap_to_95",
        "mcv1_mcv2_dropout_real_national",
        "effective_immunity_gap_real_national",
    ]
    feature_cols = [
        "year",
        *numeric_cols,
        "vaccine_data_level",
        "vaccine_data_source",
        "coverage_real_flag",
        "coverage_imputed_flag",
    ]

    model_years = sorted(model["year"].dropna().unique().tolist())
    annual_features = annual[feature_cols].copy().sort_values("year")
    annual_features = annual_features.drop_duplicates(subset=["year"], keep="last")
    real_years = set(annual_features["year"].dropna().astype(int).tolist())
    annual_features = annual_features.set_index("year").reindex(model_years)
    for col in numeric_cols:
        annual_features[col] = pd.to_numeric(annual_features[col], errors="coerce").ffill().bfill()
    annual_features["vaccine_data_source"] = annual_features["vaccine_data_source"].ffill().bfill().fillna(
        "WHO_GHO_Athena_or_WHO_WUENIC_profile"
    )
    annual_features["vaccine_data_level"] = [
        "national_annual_real" if int(year) in real_years else "national_annual_real_year_gap_filled"
        for year in annual_features.index
    ]
    annual_features["coverage_real_flag"] = [1 if int(year) in real_years else 0 for year in annual_features.index]
    annual_features["coverage_imputed_flag"] = [0 if int(year) in real_years else 1 for year in annual_features.index]
    annual_features = annual_features.reset_index().rename(columns={"index": "year"})
    out = model.merge(annual_features[feature_cols], on="year", how="left")
    return out


def write_report(report_path: Path, annual: pd.DataFrame, metadata: dict[str, Any], raw_dir: Path) -> None:
    missing_request_only = raw_dir / "vaccine_sources_unavailable_request_only.json"
    unavailable_note = "Not found"
    if missing_request_only.exists():
        unavailable_note = missing_request_only.read_text(encoding="utf-8")
    lines = [
        "# Vaccine Data Readiness Report",
        "",
        "## What was added",
        "",
        "The project now has real public Ethiopia national annual measles vaccine coverage features.",
        "These are joined to the woreda-month model by year and kept separate from woreda-level proxy coverage features.",
        "",
        f"- Annual vaccine rows: `{metadata.get('annual_rows')}`",
        f"- Year range: `{metadata.get('year_min')}` to `{metadata.get('year_max')}`",
        f"- MCV1 load mode: `{metadata.get('mcv1_mode')}`",
        f"- MCV2 load mode: `{metadata.get('mcv2_mode')}`",
        "",
        "## Latest Available Coverage Rows",
        "",
    ]
    latest = annual.sort_values("year").tail(8)
    for _, row in latest.iterrows():
        lines.append(
            f"- {int(row['year'])}: MCV1 `{row.get('mcv1_coverage_real_national')}`, "
            f"MCV2 `{row.get('mcv2_coverage_real_national')}`"
        )
    lines.extend(
        [
            "",
            "## Request-Only Sources Not Used",
            "",
            "The APHI EPI workbook and DHS microdata were not downloaded because they require request/access workflows.",
            "This keeps the project consistent with the no-request constraint.",
            "",
            "```json",
            unavailable_note,
            "```",
            "",
            "## Modeling Use",
            "",
            "Use `data/processed/measles_training_demo_model_matrix_with_vaccine.csv` for retraining.",
            "The real national vaccine features should be used alongside existing woreda-level proxy fields.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    annual, metadata = build_annual_table(args.raw_vaccine_dir)
    annual_path = args.processed_dir / "ethiopia_vaccine_coverage_annual.csv"
    model_features_path = args.processed_dir / "ethiopia_vaccine_coverage_for_model.csv"
    model_matrix_path = args.processed_dir / "measles_training_demo_model_matrix_with_vaccine.csv"
    annual.to_csv(annual_path, index=False)
    model_features = annual[
        [
            "year",
            "mcv1_coverage_real_national",
            "mcv2_coverage_real_national",
            "mcv1_gap_to_95",
            "mcv2_gap_to_95",
            "mcv1_mcv2_dropout_real_national",
            "effective_immunity_gap_real_national",
            "vaccine_data_level",
            "vaccine_data_source",
            "coverage_real_flag",
            "coverage_imputed_flag",
        ]
    ].copy()
    model_features.to_csv(model_features_path, index=False)
    joined = build_model_join(args.model_matrix, annual)
    joined.to_csv(model_matrix_path, index=False)
    write_report(args.reports_dir / "vaccine_data_readiness_report.md", annual, metadata, args.raw_vaccine_dir)
    print(
        json.dumps(
            {
                "annual_path": str(annual_path),
                "model_features_path": str(model_features_path),
                "model_matrix_with_vaccine_path": str(model_matrix_path),
                **metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
