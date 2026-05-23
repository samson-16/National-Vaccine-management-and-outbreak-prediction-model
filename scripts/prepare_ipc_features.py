"""Prepare FEWS NET IPC food insecurity features for the measles model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_IPC_DIR = ROOT / "data" / "raw" / "ipc"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DEFAULT_LOCAL_IPC_METADATA = ROOT / "IPCPopulation.csv"
DEFAULT_MODEL_MATRIX = PROCESSED_DIR / "measles_training_demo_model_matrix_with_vaccine.csv"
FALLBACK_MODEL_MATRIX = PROCESSED_DIR / "measles_training_demo_model_matrix.csv"


SCENARIO_PRIORITY = {
    "CS": 1,
    "ML": 2,
    "ML1": 2,
    "ML2": 3,
    "FIPE6": 3,
    "PN": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ipc-dir", type=Path, default=RAW_IPC_DIR)
    parser.add_argument("--local-ipc-metadata", type=Path, default=DEFAULT_LOCAL_IPC_METADATA)
    parser.add_argument("--model-matrix", type=Path, default=DEFAULT_MODEL_MATRIX)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).to_period("M").to_timestamp()
    end = pd.Timestamp(end).to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS")


def load_ipc_timeseries(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "fewsnet_ethiopia_ipc_phase3plus_population.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FEWS NET IPC time-series file: {path}. Run scripts/download_ipc_population_data.py first."
        )
    df = read_csv(path)
    for column in ["reporting_date", "projection_start", "projection_end"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in [
        "low_value",
        "high_value",
        "pct_phase3",
        "pct_phase4",
        "pct_phase5",
        "pct_population_min",
        "pct_population_max",
        "phase3_low_value",
        "phase3_high_value",
        "phase4_low_value",
        "phase4_high_value",
        "phase5_low_value",
        "phase5_high_value",
    ]:
        if column in df.columns:
            df[column] = safe_number(df[column])
    df = df[df["country_code"].astype(str).str.upper().eq("ET")].copy()
    df = df[df["phase"].astype(str).eq("3+")].copy()
    # FEWS NET uses very large sentinel upper bounds for open-ended ranges such
    # as ">= 15 million"; keep the conservative lower estimate instead of
    # letting the sentinel distort model features.
    sentinel_high = df["high_value"] > 200_000_000
    df.loc[sentinel_high, "high_value"] = df.loc[sentinel_high, "low_value"]
    sentinel_low = df["low_value"] > 200_000_000
    df.loc[sentinel_low, "low_value"] = np.nan
    return df


def load_local_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_csv(path)
    eth = df[df["country"].astype(str).str.contains("Ethiopia", case=False, na=False)].copy()
    return eth


def build_monthly_ipc_features(ipc: pd.DataFrame, model_months: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ipc.empty:
        raise ValueError("No Ethiopia IPC Phase 3+ rows available.")
    model_periods = sorted(pd.to_datetime(model_months).dt.to_period("M").dt.to_timestamp().unique())
    for period_start in model_periods:
        active = ipc[
            (ipc["projection_start"].notna())
            & (ipc["projection_end"].notna())
            & (ipc["projection_start"] <= period_start)
            & (ipc["projection_end"] >= period_start)
        ].copy()
        imputed = 0
        if active.empty:
            fallback = ipc[ipc["projection_start"].notna()].copy()
            fallback["distance_days"] = (fallback["projection_start"] - period_start).abs().dt.days
            active = fallback.sort_values("distance_days").head(1).copy()
            imputed = 1
        active["scenario_priority"] = active["scenario"].astype(str).map(SCENARIO_PRIORITY).fillna(9)
        active = active.sort_values(["scenario_priority", "reporting_date"], ascending=[True, False])
        row = active.iloc[0]
        low = float(row.get("low_value")) if pd.notna(row.get("low_value")) else np.nan
        high = float(row.get("high_value")) if pd.notna(row.get("high_value")) else np.nan
        midpoint = np.nanmean([low, high])
        pct_min = row.get("pct_population_min")
        pct_max = row.get("pct_population_max")
        pct_mid = np.nanmean([pct_min, pct_max]) if pd.notna(pct_min) or pd.notna(pct_max) else np.nan
        rows.append(
            {
                "period_start": period_start,
                "year": int(pd.Timestamp(period_start).year),
                "month": int(pd.Timestamp(period_start).month),
                "ipc_phase3plus_population_low": low,
                "ipc_phase3plus_population_high": high,
                "ipc_phase3plus_population_mid": midpoint,
                "ipc_phase3plus_population_mid_millions": midpoint / 1_000_000 if pd.notna(midpoint) else np.nan,
                "ipc_phase3plus_pct_population_mid": pct_mid,
                "ipc_scenario_code": str(row.get("scenario")),
                "ipc_scenario_name": str(row.get("scenario_name")),
                "ipc_scenario_priority": float(row.get("scenario_priority")),
                "ipc_reporting_date": row.get("reporting_date"),
                "ipc_projection_start": row.get("projection_start"),
                "ipc_projection_end": row.get("projection_end"),
                "ipc_phase3plus_real_flag": 1 if not imputed else 0,
                "ipc_phase3plus_imputed_flag": imputed,
                "ipc_data_source": "FEWS_NET_ipcpopulationsize_public_api",
            }
        )
    monthly = pd.DataFrame(rows)
    numeric_columns = [
        "ipc_phase3plus_population_low",
        "ipc_phase3plus_population_high",
        "ipc_phase3plus_population_mid",
        "ipc_phase3plus_population_mid_millions",
        "ipc_phase3plus_pct_population_mid",
        "ipc_scenario_priority",
    ]
    for column in numeric_columns:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
        monthly[column] = monthly[column].ffill().bfill()
    pct_available = 0 if monthly["ipc_phase3plus_pct_population_mid"].isna().all() else 1
    monthly["ipc_phase3plus_pct_population_available_flag"] = pct_available
    monthly["ipc_phase3plus_pct_population_mid"] = monthly["ipc_phase3plus_pct_population_mid"].fillna(0.0)
    return monthly


def join_ipc_to_model(model_path: Path, monthly: pd.DataFrame) -> pd.DataFrame:
    if not model_path.exists() and model_path == DEFAULT_MODEL_MATRIX and FALLBACK_MODEL_MATRIX.exists():
        model_path = FALLBACK_MODEL_MATRIX
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model matrix: {model_path}")
    model = pd.read_csv(model_path)
    model["period_start"] = pd.to_datetime(model["period_start"]).dt.to_period("M").dt.to_timestamp()
    join_cols = [
        "period_start",
        "ipc_phase3plus_population_low",
        "ipc_phase3plus_population_high",
        "ipc_phase3plus_population_mid",
        "ipc_phase3plus_population_mid_millions",
        "ipc_phase3plus_pct_population_mid",
        "ipc_phase3plus_pct_population_available_flag",
        "ipc_scenario_priority",
        "ipc_phase3plus_real_flag",
        "ipc_phase3plus_imputed_flag",
        "ipc_data_source",
    ]
    out = model.merge(monthly[join_cols], on="period_start", how="left")
    return out


def write_report(
    report_path: Path,
    ipc_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    monthly: pd.DataFrame,
    joined: pd.DataFrame,
) -> None:
    scenario_counts = monthly["ipc_scenario_name"].value_counts(dropna=False).to_dict()
    metadata_level = "none"
    if not metadata.empty:
        has_subnational = metadata[["admin_1", "admin_2", "admin_3"]].notna().any().any()
        metadata_level = "subnational" if has_subnational else "national_only"
    latest = monthly.sort_values("period_start").tail(8)
    lines = [
        "# IPC Food Stress Data Readiness Report",
        "",
        "## What was added",
        "",
        "The project now uses real public FEWS NET Ethiopia IPC Phase 3+ population estimate time-series data.",
        "The manually downloaded `IPCPopulation.csv` is metadata/catalog-level data; the actual monthly population estimates come from the public FEWS NET `ipcpopulationsize` endpoint.",
        "",
        f"- Raw IPC rows: `{len(ipc_raw)}`",
        f"- Monthly feature rows: `{len(monthly)}`",
        f"- Joined model rows: `{len(joined)}`",
        f"- Local metadata Ethiopia level: `{metadata_level}`",
        f"- Scenario counts: `{scenario_counts}`",
        "",
        "## Latest Monthly Features",
        "",
    ]
    for _, row in latest.iterrows():
        lines.append(
            f"- {pd.Timestamp(row['period_start']).date()}: "
            f"{row['ipc_phase3plus_population_mid_millions']:.2f} million people Phase 3+ "
            f"({row['ipc_scenario_name']})"
        )
    lines.extend(
        [
            "",
            "## Modeling Use",
            "",
            "Use `data/processed/measles_training_demo_model_matrix_with_vaccine_ipc.csv` for the next retraining run.",
            "These IPC features are national monthly food-stress covariates. They do not provide woreda-level IPC variation in the current public file.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_matrix if args.model_matrix.exists() else FALLBACK_MODEL_MATRIX
    model = pd.read_csv(model_path)
    ipc_raw = load_ipc_timeseries(args.raw_ipc_dir)
    metadata = load_local_metadata(args.local_ipc_metadata)
    monthly = build_monthly_ipc_features(ipc_raw, model["period_start"])
    joined = join_ipc_to_model(model_path, monthly)
    monthly_path = args.processed_dir / "ethiopia_ipc_phase3plus_monthly.csv"
    joined_path = args.processed_dir / "measles_training_demo_model_matrix_with_vaccine_ipc.csv"
    monthly.to_csv(monthly_path, index=False)
    joined.to_csv(joined_path, index=False)
    write_report(args.reports_dir / "ipc_data_readiness_report.md", ipc_raw, metadata, monthly, joined)
    print(
        json.dumps(
            {
                "monthly_ipc_features": str(monthly_path),
                "model_matrix_with_vaccine_ipc": str(joined_path),
                "raw_ipc_rows": int(len(ipc_raw)),
                "monthly_rows": int(len(monthly)),
                "joined_rows": int(len(joined)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
