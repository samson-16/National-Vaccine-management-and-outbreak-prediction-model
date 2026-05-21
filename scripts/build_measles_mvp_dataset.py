"""Build the Ethiopia measles MVP datasets with removable synthetic labels.

Outputs:
  data/processed/location_reference_woredas.csv
  data/processed/measles_woreda_month_real_weak_only.csv
  data/processed/measles_woreda_month_synthetic_overlay.csv
  data/processed/measles_woreda_month_ml_ready_mixed.csv
  data/processed/external_measles_normalized.csv          (when available)
  data/processed/measles_training_selected.csv
  data/processed/source_inventory.csv
  data/processed/measles_woreda_month_data_dictionary.csv
  reports/modeling_readiness_report.md

The synthetic overlay is intentionally easy to remove:
  - filter synthetic_flag == 0, or
  - train from measles_woreda_month_real_weak_only.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPHI_MEASLES = Path(r"C:\Users\ASUS\Downloads\Measels_2025_agg_from_LL.xlsx")
RAW_EXTERNAL_DIR = ROOT / "data" / "raw" / "external"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DEFAULT_ADMIN_BOUNDARIES = ROOT / "eth_admin_boundaries.xlsx"
DEFAULT_IPC_POPULATION = ROOT / "IPCPopulation.csv"
DEFAULT_WHO_REPORTED_CASES = ROOT / "reported-cases-data.xlsx"

JHU_TOP_STATES_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/measles_data/main/"
    "Top_states_time_series.csv"
)
JHU_COUNTY_UPDATES_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/measles_data/main/"
    "measles_county_all_updates.csv"
)
OWID_GLOBAL_MEASLES_URL = (
    "https://ourworldindata.org/grapher/reported-cases-of-measles.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)
WHO_REPORTED_CASES_XLSX_URL = (
    "https://srhdpeuwpubsa-geecgzbpd5h0fueu.z01.azurefd.net/whdh/WIISE/export/"
    "reported-cases-data.xlsx"
)
TIDYTUESDAY_WHO_MONTHLY_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
    "data/2025/2025-06-24/cases_month.csv"
)

FLAG_COLUMNS = [
    "real_flag",
    "weak_label_flag",
    "synthetic_flag",
    "imputed_flag",
    "guessed_flag",
    "external_country_flag",
    "ethiopia_flag",
]


@dataclass(frozen=True)
class OutputPaths:
    location_reference: Path
    real_weak: Path
    synthetic_overlay: Path
    mixed: Path
    external_normalized: Path
    selected_training: Path
    demo_model_matrix: Path
    demo_provenance: Path
    source_inventory: Path
    data_dictionary: Path
    summary: Path
    readiness_report: Path


def paths() -> OutputPaths:
    return OutputPaths(
        location_reference=PROCESSED_DIR / "location_reference_woredas.csv",
        real_weak=PROCESSED_DIR / "measles_woreda_month_real_weak_only.csv",
        synthetic_overlay=PROCESSED_DIR / "measles_woreda_month_synthetic_overlay.csv",
        mixed=PROCESSED_DIR / "measles_woreda_month_ml_ready_mixed.csv",
        external_normalized=PROCESSED_DIR / "external_measles_normalized.csv",
        selected_training=PROCESSED_DIR / "measles_training_selected.csv",
        demo_model_matrix=PROCESSED_DIR / "measles_training_demo_model_matrix.csv",
        demo_provenance=PROCESSED_DIR / "measles_training_demo_provenance.csv",
        source_inventory=PROCESSED_DIR / "source_inventory.csv",
        data_dictionary=PROCESSED_DIR / "measles_woreda_month_data_dictionary.csv",
        summary=PROCESSED_DIR / "measles_mvp_build_summary.json",
        readiness_report=REPORTS_DIR / "modeling_readiness_report.md",
    )


def normalize_column(name: object) -> str:
    value = str(name).strip().lower()
    value = value.replace("%", " percent ")
    value = value.replace("&", " and ")
    value = re.sub(r"[/\\]+", "_", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def name_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_float(*parts: object) -> float:
    key = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def stable_int(low: int, high: int, *parts: object) -> int:
    if high < low:
        raise ValueError("high must be >= low")
    return low + int(stable_float(*parts) * (high - low + 1))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


def ensure_dirs() -> None:
    RAW_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, **kwargs)
        return path
    except PermissionError:
        fallback = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(fallback, index=False, **kwargs)
        print(f"Warning: {path} is locked. Wrote {fallback} instead.", file=sys.stderr)
        return fallback


def safe_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(path.stem + "_new" + path.suffix)
        fallback.write_text(text, encoding="utf-8")
        print(f"Warning: {path} is locked. Wrote {fallback} instead.", file=sys.stderr)
        return fallback


def read_ephi_measles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"EPHI measles workbook not found: {path}")

    df = pd.read_excel(path)
    df = df.rename(columns={column: normalize_column(column) for column in df.columns})
    required = {"region", "zone", "woreda", "year", "epi_week", "msls", "msls_dths"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"EPHI measles workbook is missing columns: {sorted(missing)}")

    df = df[list(required)].copy()
    for column in ["region", "zone", "woreda"]:
        df[column] = df[column].map(normalize_text)
    df["region"] = df["region"].replace({"Southern Ethiopi": "Southern Ethiopia"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["epi_week"] = pd.to_numeric(df["epi_week"], errors="coerce").astype("Int64")
    df["msls"] = pd.to_numeric(df["msls"], errors="coerce").fillna(0).astype(int)
    df["msls_dths"] = pd.to_numeric(df["msls_dths"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["year", "epi_week"])
    df = df[(df["woreda"] != "") & (df["region"] != "")]
    return df


def epi_week_to_month(year: int, epi_week: int) -> pd.Timestamp:
    week = max(1, min(52, int(epi_week)))
    # Use the Thursday inside the ISO week so week 1 of a year lands in January.
    return pd.Timestamp(date.fromisocalendar(int(year), week, 4)).to_period("M").to_timestamp()


def aggregate_ephi_to_month(df: pd.DataFrame, outbreak_threshold: int) -> pd.DataFrame:
    working = df.copy()
    working["period_start"] = [
        epi_week_to_month(int(year), int(week))
        for year, week in zip(working["year"], working["epi_week"])
    ]
    grouped = (
        working.groupby(["region", "zone", "woreda", "period_start"], as_index=False)
        .agg(real_cases=("msls", "sum"), real_deaths=("msls_dths", "sum"))
    )
    grouped["real_outbreak_label"] = (grouped["real_cases"] >= outbreak_threshold).astype(int)
    return grouped


def load_admin3_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        admin = pd.read_excel(path, sheet_name="eth_admin3")
    except Exception:
        return pd.DataFrame()
    admin = admin.rename(columns={column: normalize_column(column) for column in admin.columns})
    required = {"adm1_name", "adm2_name", "adm3_name", "adm3_pcode"}
    if not required.issubset(admin.columns):
        return pd.DataFrame()
    keep = [
        "adm1_name",
        "adm2_name",
        "adm3_name",
        "adm1_pcode",
        "adm2_pcode",
        "adm3_pcode",
        "area_sqkm",
        "version",
        "center_lat",
        "center_lon",
    ]
    keep = [column for column in keep if column in admin.columns]
    admin = admin[keep].copy()
    for column in ["adm1_name", "adm2_name", "adm3_name"]:
        admin[column] = admin[column].map(normalize_text)
    admin["region_key"] = admin["adm1_name"].map(name_key)
    admin["zone_key"] = admin["adm2_name"].map(name_key)
    admin["woreda_key"] = admin["adm3_name"].map(name_key)
    return admin.drop_duplicates(["region_key", "zone_key", "woreda_key"])


def build_location_reference(ephi_monthly: pd.DataFrame, admin_boundaries: Path | None = None) -> pd.DataFrame:
    location = (
        ephi_monthly[["region", "zone", "woreda"]]
        .drop_duplicates()
        .sort_values(["region", "zone", "woreda"])
        .reset_index(drop=True)
    )
    location["country"] = "Ethiopia"
    location["admin_level"] = "woreda"
    location["admin1_region"] = location["region"]
    location["admin2_zone"] = location["zone"]
    location["admin3_woreda"] = location["woreda"]
    location["admin3_pcode"] = ""
    location["woreda_name_clean"] = location["woreda"].map(name_key)
    location["region_key"] = location["admin1_region"].map(name_key)
    location["zone_key"] = location["admin2_zone"].map(name_key)
    location["woreda_key"] = location["admin3_woreda"].map(name_key)
    admin = load_admin3_reference(admin_boundaries) if admin_boundaries else pd.DataFrame()
    if not admin.empty:
        location = location.merge(
            admin,
            on=["region_key", "zone_key", "woreda_key"],
            how="left",
            suffixes=("", "_admin"),
        )
        location["admin3_pcode"] = location["adm3_pcode"].fillna("")
        location["area_km2"] = location["area_sqkm"]
        location["centroid_lat"] = location["center_lat"]
        location["centroid_lon"] = location["center_lon"]
        location["geometry_source"] = "eth_admin_boundaries.xlsx"
        location["geometry_version"] = location["version"].fillna("unknown")
        location["source_notes"] = location["admin3_pcode"].apply(
            lambda value: (
                "Matched to Ethiopia COD-AB admin3 p-code from eth_admin_boundaries.xlsx."
                if str(value).strip()
                else "No exact COD-AB admin3 p-code match; kept EPHI workbook location name."
            )
        )
    else:
        location["area_km2"] = pd.NA
        location["centroid_lat"] = pd.NA
        location["centroid_lon"] = pd.NA
        location["geometry_source"] = "ephi_workbook_only"
        location["geometry_version"] = "no_boundary_file_loaded"
        location["source_notes"] = (
            "Reference built from EPHI measles workbook names. Add COD-AB p-codes later "
            "for production geospatial joins."
        )
    location["location_id"] = [
        "ETH:" + (row.admin3_pcode if str(row.admin3_pcode).strip() else hashlib.sha1(
            f"{row.admin1_region}|{row.admin2_zone}|{row.admin3_woreda}".encode("utf-8")
        ).hexdigest()[:12])
        for row in location.itertuples(index=False)
    ]
    location["location_name"] = (
        location["admin3_woreda"] + ", " + location["admin2_zone"] + ", " + location["admin1_region"]
    )
    return location[
        [
            "country",
            "admin_level",
            "admin1_region",
            "admin2_zone",
            "admin3_woreda",
            "admin3_pcode",
            "woreda_name_clean",
            "location_id",
            "location_name",
            "geometry_source",
            "geometry_version",
            "area_km2",
            "centroid_lat",
            "centroid_lon",
            "source_notes",
        ]
    ]


def month_range(start_month: str, end_month: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start_month), pd.Timestamp(end_month), freq="MS")


def build_base_panel(location: pd.DataFrame, start_month: str, end_month: str) -> pd.DataFrame:
    months = month_range(start_month, end_month)
    panel = location.merge(pd.DataFrame({"period_start": months}), how="cross")
    panel["year"] = panel["period_start"].dt.year
    panel["month"] = panel["period_start"].dt.month
    panel["country"] = "Ethiopia"
    panel["external_country_flag"] = 0
    panel["ethiopia_flag"] = 1
    return panel


def add_proxy_features(panel: pd.DataFrame, real_monthly: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    totals = (
        real_monthly.groupby(["region", "zone", "woreda"], as_index=False)
        .agg(location_real_cases_total=("real_cases", "sum"))
    )
    totals["woreda_name_clean"] = totals["woreda"].map(name_key)
    region_totals = (
        real_monthly.groupby("region", as_index=False)
        .agg(region_real_cases_total=("real_cases", "sum"))
        .rename(columns={"region": "admin1_region"})
    )
    panel = panel.merge(
        totals[["woreda_name_clean", "location_real_cases_total"]],
        on="woreda_name_clean",
        how="left",
    )
    panel = panel.merge(region_totals, on="admin1_region", how="left")
    panel["location_real_cases_total"] = panel["location_real_cases_total"].fillna(0)
    panel["region_real_cases_total"] = panel["region_real_cases_total"].fillna(0)

    max_location_cases = max(1.0, float(panel["location_real_cases_total"].max()))
    max_region_cases = max(1.0, float(panel["region_real_cases_total"].max()))

    rows: list[dict[str, float | int | str]] = []
    for row in panel.itertuples(index=False):
        seed = row.location_id
        region_key = name_key(row.admin1_region)
        population_base = {
            "oromia": 260_000,
            "amhara": 210_000,
            "somali": 170_000,
            "tigray": 160_000,
            "sidama": 155_000,
            "benishangul gumz": 115_000,
        }.get(region_key, 145_000)
        population_total = int(population_base * (0.65 + stable_float(seed, "pop") * 1.7))
        under5_population = int(population_total * (0.125 + stable_float(seed, "under5") * 0.055))
        school_age_population = int(population_total * (0.22 + stable_float(seed, "school") * 0.08))
        if pd.notna(getattr(row, "area_km2", pd.NA)) and float(row.area_km2) > 0:
            density = population_total / float(row.area_km2)
        else:
            density = 45 + stable_float(seed, "density") * 650
        mcv1 = 48 + stable_float(seed, "mcv1") * 42
        mcv2 = max(15, mcv1 - (10 + stable_float(seed, "mcv2_gap") * 30))
        dropout = max(0, min(100, mcv1 - mcv2))
        susceptible_proxy = under5_population * max(0.02, (100 - mcv1) / 100)
        travel_time = 20 + stable_float(seed, "travel") * 220

        month_angle = 2 * math.pi * (int(row.month) - 1) / 12
        rainfall_anomaly = (
            0.85 * math.sin(month_angle - 1.2)
            + (stable_float(seed, row.period_start, "rain") - 0.5) * 1.1
        )
        dry_season_flag = 1 if int(row.month) in {1, 2, 3, 4, 11, 12} else 0
        food_phase = 1 + int(stable_float(seed, row.period_start, "food") * 4)
        conflict_events = int(stable_float(seed, row.period_start, "conflict") > 0.78)
        if stable_float(seed, row.period_start, "conflict-high") > 0.95:
            conflict_events += stable_int(1, 4, seed, row.period_start, "conflict-count")

        location_intensity = float(row.location_real_cases_total) / max_location_cases
        region_intensity = float(row.region_real_cases_total) / max_region_cases
        access_stress = min(1.0, travel_time / 240)
        immunity_gap = max(0.0, (95 - mcv1) / 95)
        rainfall_stress = min(1.0, abs(rainfall_anomaly) / 1.5)
        risk_score = (
            0.32 * immunity_gap
            + 0.12 * min(1.0, density / 700)
            + 0.10 * access_stress
            + 0.10 * ((food_phase - 1) / 4)
            + 0.08 * min(1.0, conflict_events / 3)
            + 0.18 * location_intensity
            + 0.10 * region_intensity
            + 0.06 * dry_season_flag
            + 0.04 * rainfall_stress
        )
        rows.append(
            {
                "population_total_est": population_total,
                "under5_population_est": under5_population,
                "school_age_population_est": school_age_population,
                "population_density_est": round(density, 3),
                "mcv1_coverage_est": round(mcv1, 3),
                "mcv2_coverage_est": round(mcv2, 3),
                "measles_dropout_proxy": round(dropout, 3),
                "susceptible_children_proxy": round(susceptible_proxy, 3),
                "travel_time_healthcare_min_est": round(travel_time, 3),
                "rainfall_anomaly_proxy": round(rainfall_anomaly, 3),
                "dry_season_flag": dry_season_flag,
                "food_insecurity_phase_proxy": food_phase,
                "conflict_events_proxy": conflict_events,
                "location_real_cases_total": int(row.location_real_cases_total),
                "region_real_cases_total": int(row.region_real_cases_total),
                "mvp_risk_score_proxy": round(risk_score, 6),
                "feature_source_notes": (
                    "Proxy covariates generated deterministically from location/month because "
                    "public rasters/APIs are not yet materialized into this workspace."
                ),
            }
        )
    features = pd.DataFrame(rows)
    return pd.concat([panel.reset_index(drop=True), features], axis=1)


def build_real_weak_panel(
    panel: pd.DataFrame,
    ephi_monthly: pd.DataFrame,
    outbreak_threshold: int,
) -> pd.DataFrame:
    labels = ephi_monthly.copy()
    labels["woreda_name_clean"] = labels["woreda"].map(name_key)
    labels = labels.rename(
        columns={
            "real_cases": "target_cases",
            "real_deaths": "target_deaths",
            "real_outbreak_label": "target_outbreak",
            "region": "admin1_region",
            "zone": "admin2_zone",
            "woreda": "admin3_woreda",
        }
    )
    merge_cols = ["admin1_region", "admin2_zone", "woreda_name_clean", "period_start"]
    panel = panel.merge(
        labels[merge_cols + ["target_cases", "target_deaths", "target_outbreak"]],
        on=merge_cols,
        how="left",
    )
    real_mask = panel["target_cases"].notna()
    panel["label_type"] = "unknown"
    panel.loc[real_mask, "label_type"] = "real_ephi"
    panel["label_confidence"] = 0.0
    panel.loc[real_mask, "label_confidence"] = 1.0
    panel["label_source"] = ""
    panel.loc[real_mask, "label_source"] = "EPHI provided measles aggregate line-list workbook"
    panel["real_flag"] = real_mask.astype(int)
    panel["weak_label_flag"] = 0
    panel["synthetic_flag"] = 0
    panel["imputed_flag"] = 1
    panel["guessed_flag"] = 1
    panel["outbreak_threshold_cases"] = outbreak_threshold
    panel["source_notes"] = (
        "Rows with label_type=real_ephi use provided EPHI cases/deaths. "
        "Unknown rows are not assumed to be zero."
    )
    return panel


def download_file(url: str, destination: Path, timeout: int = 30) -> tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VMOPS-measles-mvp/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            destination.write_bytes(response.read())
        return True, "downloaded"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - message is surfaced in source inventory.
        return False, str(exc)


def download_external_sources(force: bool, include_who_export: bool) -> list[dict[str, Any]]:
    sources = [
        {
            "id": "jhu_us_top_states_weekly",
            "url": JHU_TOP_STATES_URL,
            "path": RAW_EXTERNAL_DIR / "jhu_top_states_time_series.csv",
            "role": "external_measles_outbreak_shape",
        },
        {
            "id": "jhu_us_county_daily",
            "url": JHU_COUNTY_UPDATES_URL,
            "path": RAW_EXTERNAL_DIR / "jhu_measles_county_all_updates.csv",
            "role": "external_measles_county_reporting",
        },
        {
            "id": "owid_who_global_measles_reported_cases",
            "url": OWID_GLOBAL_MEASLES_URL,
            "path": RAW_EXTERNAL_DIR / "owid_reported_cases_of_measles.csv",
            "role": "external_global_country_year_measles_cases",
        },
        {
            "id": "tidytuesday_who_monthly_measles_cases",
            "url": TIDYTUESDAY_WHO_MONTHLY_URL,
            "path": RAW_EXTERNAL_DIR / "tidytuesday_who_cases_month.csv",
            "role": "external_global_country_month_measles_cases",
        }
    ]
    if include_who_export:
        sources.append(
            {
                "id": "who_immunization_reported_cases_export",
                "url": WHO_REPORTED_CASES_XLSX_URL,
                "path": RAW_EXTERNAL_DIR / "who_reported_cases_data.xlsx",
                "role": "external_who_country_year_reported_cases",
                "timeout": 180,
            }
        )
    results = []
    for source in sources:
        destination = Path(source["path"])
        if destination.exists() and not force:
            ok, message = True, "already_exists"
        else:
            ok, message = download_file(source["url"], destination, timeout=int(source.get("timeout", 30)))
        results.append({**source, "available": ok and destination.exists(), "status": message})
    return results


def normalize_jhu_top_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "week_start" not in df.columns:
        return pd.DataFrame()
    case_cols = [column for column in df.columns if column.endswith("_cases")]
    if not case_cols:
        return pd.DataFrame()

    long = df.melt(
        id_vars=["week_start", "week_end"],
        value_vars=case_cols,
        var_name="state_code",
        value_name="cases",
    )
    long["state_code"] = long["state_code"].str.replace("_cases", "", regex=False)
    long["date"] = pd.to_datetime(long["week_start"], errors="coerce")
    long["period_start"] = long["date"].dt.to_period("M").dt.to_timestamp()
    long["cases"] = pd.to_numeric(long["cases"], errors="coerce").fillna(0).astype(int)
    monthly = (
        long.groupby(["state_code", "period_start"], as_index=False)
        .agg(cases=("cases", "sum"))
    )
    monthly["country"] = "United States"
    monthly["admin_level"] = "state"
    monthly["location_id"] = "US:" + monthly["state_code"]
    monthly["location_name"] = monthly["state_code"]
    monthly["date"] = monthly["period_start"]
    monthly["period_frequency"] = "monthly"
    monthly["deaths"] = pd.NA
    monthly["source_dataset"] = "JHU Measles Tracking Team Top_states_time_series.csv"
    monthly["external_country_flag"] = 1
    monthly["ethiopia_flag"] = 0
    monthly["label_type"] = "external_public"
    monthly["real_flag"] = 1
    monthly["weak_label_flag"] = 0
    monthly["synthetic_flag"] = 0
    monthly["imputed_flag"] = 0
    monthly["guessed_flag"] = 0
    monthly["source_notes"] = (
        "Public external measles time series. Use for outbreak-shape support, "
        "not as Ethiopian ground truth."
    )
    return monthly[
        [
            "country",
            "admin_level",
            "location_id",
            "location_name",
            "date",
            "period_start",
            "period_frequency",
            "cases",
            "deaths",
            "source_dataset",
            "external_country_flag",
            "ethiopia_flag",
            "label_type",
            "real_flag",
            "weak_label_flag",
            "synthetic_flag",
            "imputed_flag",
            "guessed_flag",
            "source_notes",
        ]
    ]


def normalize_jhu_county_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=standard_external_columns())
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        # Some raw snapshots are emitted as whitespace-separated quoted records.
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=" ", quotechar='"', skipinitialspace=True)
            for parsed in reader:
                rows.extend([item.split(",") for item in parsed if item.strip()])
        df = pd.DataFrame(
            rows,
            columns=["location_name", "location_id", "location_type", "date", "outcome_type", "value"],
        )
    df = df.rename(columns={column: normalize_column(column) for column in df.columns})
    if "outcome_type" not in df.columns:
        outcome_cols = [column for column in df.columns if "outcome" in column]
        if outcome_cols:
            df = df.rename(columns={outcome_cols[0]: "outcome_type"})
    required = {"location_name", "location_id", "location_type", "date", "value"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=standard_external_columns())

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df[df["date"].notna()].copy()
    monthly = (
        df.groupby(["location_id", "location_name", "location_type", "period_start"], as_index=False)
        .agg(cases=("value", "sum"))
    )
    monthly["cases"] = monthly["cases"].clip(lower=0).round().astype(int)
    monthly["country"] = "United States"
    monthly["admin_level"] = monthly["location_type"].map(normalize_text).replace("", "county")
    monthly["location_id"] = "US:" + monthly["location_id"].astype(str)
    monthly["date"] = monthly["period_start"]
    monthly["period_frequency"] = "monthly"
    monthly["deaths"] = pd.NA
    monthly["source_dataset"] = "JHU Measles Tracking Team measles_county_all_updates.csv"
    monthly["external_country_flag"] = 1
    monthly["ethiopia_flag"] = 0
    monthly["label_type"] = "external_public"
    monthly["real_flag"] = 1
    monthly["weak_label_flag"] = 0
    monthly["synthetic_flag"] = 0
    monthly["imputed_flag"] = 0
    monthly["guessed_flag"] = 0
    monthly["source_notes"] = (
        "Public external US county/month measles cases. Use for outbreak-shape "
        "support, not as Ethiopian ground truth."
    )
    return monthly[standard_external_columns()]


def standard_external_columns() -> list[str]:
    return [
        "country",
        "admin_level",
        "location_id",
        "location_name",
        "date",
        "period_start",
        "period_frequency",
        "cases",
        "deaths",
        "source_dataset",
        "external_country_flag",
        "ethiopia_flag",
        "label_type",
        "real_flag",
        "weak_label_flag",
        "synthetic_flag",
        "imputed_flag",
        "guessed_flag",
        "source_notes",
    ]


def finalize_external_frame(
    df: pd.DataFrame,
    *,
    country_col: str,
    year_col: str,
    cases_col: str,
    source_dataset: str,
    code_col: str | None = None,
    notes: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=standard_external_columns())
    working = df.copy()
    working[country_col] = working[country_col].map(normalize_text)
    working = working[working[country_col].str.lower() != "ethiopia"].copy()
    working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
    working[cases_col] = pd.to_numeric(working[cases_col], errors="coerce")
    working = working[
        (working[country_col] != "")
        & working[year_col].notna()
        & working[cases_col].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=standard_external_columns())

    working["period_start"] = pd.to_datetime(
        working[year_col].astype(int).astype(str) + "-01-01",
        errors="coerce",
    )
    working["date"] = working["period_start"]
    working["cases"] = working[cases_col].clip(lower=0).round().astype(int)
    working["deaths"] = pd.NA
    working["country"] = working[country_col]
    working["admin_level"] = "country"
    if code_col and code_col in working.columns:
        code_values = working[code_col].fillna("").astype(str).str.strip()
        working["location_id"] = code_values.where(
            code_values != "",
            working["country"].map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]),
        )
    else:
        working["location_id"] = working["country"].map(
            lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        )
    working["location_id"] = "GLOBAL:" + working["location_id"].astype(str)
    working["location_name"] = working["country"]
    working["period_frequency"] = "annual"
    working["source_dataset"] = source_dataset
    working["external_country_flag"] = 1
    working["ethiopia_flag"] = 0
    working["label_type"] = "external_public"
    working["real_flag"] = 1
    working["weak_label_flag"] = 0
    working["synthetic_flag"] = 0
    working["imputed_flag"] = 0
    working["guessed_flag"] = 0
    working["source_notes"] = notes
    return working[standard_external_columns()]


def normalize_owid_global_measles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=standard_external_columns())
    df = pd.read_csv(path)
    normalized = {column: normalize_column(column) for column in df.columns}
    df = df.rename(columns=normalized)
    country_col = "entity" if "entity" in df.columns else None
    year_col = "year" if "year" in df.columns else None
    code_col = "code" if "code" in df.columns else None
    candidate_case_cols = [
        column
        for column in df.columns
        if column not in {"entity", "code", "year"}
        and ("measles" in column or "reported_cases" in column or "cases" in column)
    ]
    if not country_col or not year_col or not candidate_case_cols:
        return pd.DataFrame(columns=standard_external_columns())
    return finalize_external_frame(
        df,
        country_col=country_col,
        year_col=year_col,
        cases_col=candidate_case_cols[0],
        code_col=code_col,
        source_dataset="Our World in Data / WHO GHO reported cases of measles",
        notes=(
            "Annual country-level public measles cases. Use for global outbreak "
            "context and external support, not as Ethiopian woreda ground truth."
        ),
    )


def normalize_who_reported_cases_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=standard_external_columns())
    try:
        df = pd.read_excel(path, sheet_name="Data")
    except Exception:
        return pd.DataFrame(columns=standard_external_columns())
    df = df.rename(columns={column: normalize_column(column) for column in df.columns})
    required = {"name", "code", "year", "disease", "cases"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=standard_external_columns())
    df = df[df["disease"].astype(str).str.lower().isin(["measles", "msl"])].copy()
    df = df[df["name"].astype(str).str.lower() != "ethiopia"].copy()

    return finalize_external_frame(
        df,
        country_col="name",
        year_col="year",
        cases_col="cases",
        code_col="code",
        source_dataset="WHO Immunization Data Portal reported cases export",
        notes=(
            "Annual country-level WHO immunization reported-cases export filtered "
            "to measles. Non-Ethiopia rows only."
        ),
    )


def normalize_tidytuesday_who_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=standard_external_columns())
    df = pd.read_csv(path)
    df = df.rename(columns={column: normalize_column(column) for column in df.columns})
    required = {"region", "country", "iso3", "year", "measles_total"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=standard_external_columns())
    month_col = "month_num" if "month_num" in df.columns else "month"
    if month_col not in df.columns:
        return pd.DataFrame(columns=standard_external_columns())

    working = df.copy()
    working["country"] = working["country"].map(normalize_text)
    working = working[working["country"].str.lower() != "ethiopia"].copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working[month_col] = pd.to_numeric(working[month_col], errors="coerce")
    working["measles_total"] = pd.to_numeric(working["measles_total"], errors="coerce")
    working = working[
        (working["country"] != "")
        & working["year"].notna()
        & working[month_col].between(1, 12)
        & working["measles_total"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=standard_external_columns())

    working["period_start"] = pd.to_datetime(
        working["year"].astype(int).astype(str)
        + "-"
        + working[month_col].astype(int).astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    working["date"] = working["period_start"]
    working["cases"] = working["measles_total"].clip(lower=0).round().astype(int)
    working["deaths"] = pd.NA
    working["admin_level"] = "country"
    working["location_id"] = "GLOBAL:" + working["iso3"].fillna("").astype(str).str.strip()
    missing_id = working["location_id"].eq("GLOBAL:")
    working.loc[missing_id, "location_id"] = (
        "GLOBAL:"
        + working.loc[missing_id, "country"].map(
            lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        )
    )
    working["location_name"] = working["country"]
    working["period_frequency"] = "monthly"
    working["source_dataset"] = "TidyTuesday / WHO provisional monthly measles cases"
    working["external_country_flag"] = 1
    working["ethiopia_flag"] = 0
    working["label_type"] = "external_public"
    working["real_flag"] = 1
    working["weak_label_flag"] = 0
    working["synthetic_flag"] = 0
    working["imputed_flag"] = 0
    working["guessed_flag"] = 0
    working["source_notes"] = (
        "Public global monthly country-level WHO provisional measles cases, "
        "republished by TidyTuesday. Non-Ethiopia rows only."
    )
    return working[standard_external_columns()]


def normalize_external_sources() -> pd.DataFrame:
    who_export_path = RAW_EXTERNAL_DIR / "who_reported_cases_data.xlsx"
    if not who_export_path.exists() and DEFAULT_WHO_REPORTED_CASES.exists():
        who_export_path = DEFAULT_WHO_REPORTED_CASES
    frames = [
        normalize_jhu_top_states(RAW_EXTERNAL_DIR / "jhu_top_states_time_series.csv"),
        normalize_jhu_county_daily(RAW_EXTERNAL_DIR / "jhu_measles_county_all_updates.csv"),
        normalize_owid_global_measles(RAW_EXTERNAL_DIR / "owid_reported_cases_of_measles.csv"),
        normalize_tidytuesday_who_monthly(RAW_EXTERNAL_DIR / "tidytuesday_who_cases_month.csv"),
        normalize_who_reported_cases_export(who_export_path),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=standard_external_columns())
    normalized = pd.concat(frames, ignore_index=True)
    return normalized.drop_duplicates(
        subset=["country", "admin_level", "location_id", "period_start", "source_dataset"],
        keep="first",
    )


def external_calibration(external: pd.DataFrame) -> dict[str, float]:
    if external.empty or "cases" not in external:
        return {"p75_nonzero_cases": 18.0, "outbreak_month_rate": 0.14}
    calibration_data = external.copy()
    if "period_frequency" in calibration_data.columns:
        monthly = calibration_data[calibration_data["period_frequency"] == "monthly"]
        if not monthly.empty:
            calibration_data = monthly
    cases = pd.to_numeric(calibration_data["cases"], errors="coerce").fillna(0)
    nonzero = cases[cases > 0]
    if nonzero.empty:
        return {"p75_nonzero_cases": 18.0, "outbreak_month_rate": 0.14}
    # The bundled public US source is a "top states" series, so its outbreak
    # frequency is intentionally down-weighted before calibrating Ethiopia MVP
    # synthetic labels.
    raw_rate = float((cases >= 5).mean())
    return {
        "p75_nonzero_cases": float(nonzero.quantile(0.75)),
        "outbreak_month_rate": max(0.08, min(0.20, raw_rate * 0.35)),
    }


def generate_synthetic_overlay(
    real_weak: pd.DataFrame,
    external: pd.DataFrame,
    outbreak_threshold: int,
) -> pd.DataFrame:
    calibration = external_calibration(external)
    unknown = real_weak[real_weak["label_type"] == "unknown"].copy()
    unknown["period_start"] = pd.to_datetime(unknown["period_start"])
    unknown = unknown.sort_values(["location_id", "period_start"])
    real_rows = real_weak[real_weak["label_type"] == "real_ephi"].copy()
    real_rows["period_start"] = pd.to_datetime(real_rows["period_start"])
    real_case_values = pd.to_numeric(real_rows["target_cases"], errors="coerce").dropna()
    real_case_scale = float(real_case_values[real_case_values > 0].quantile(0.75)) if (real_case_values > 0).any() else 18.0
    case_scale = max(outbreak_threshold, min(80.0, (0.65 * real_case_scale) + (0.35 * calibration["p75_nonzero_cases"])))
    observed_by_location = {
        location_id: list(group["period_start"])
        for location_id, group in real_rows.groupby("location_id")
    }
    observed_by_region = {
        region: list(group["period_start"])
        for region, group in real_rows.groupby("admin1_region")
    }
    rows: list[dict[str, Any]] = []

    def month_gap(current: pd.Timestamp, candidates: list[pd.Timestamp]) -> int | None:
        if not candidates:
            return None
        current_month = current.year * 12 + current.month
        gaps = [abs(current_month - (candidate.year * 12 + candidate.month)) for candidate in candidates]
        return min(gaps) if gaps else None

    for location_id, group in unknown.groupby("location_id", sort=False):
        location_seed = int(hashlib.sha256(str(location_id).encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(location_seed)
        wave_remaining = 0
        wave_age = 0
        wave_peak = 0
        prior_cases = 0

        for row in group.itertuples(index=False):
            row_seed = f"{row.location_id}|{row.period_start:%Y-%m-%d}"
            row_rng = random.Random(int(hashlib.sha256(row_seed.encode("utf-8")).hexdigest()[:16], 16))
            risk = float(row.mvp_risk_score_proxy)
            month = int(row.month)
            dry_boost = 0.10 if int(row.dry_season_flag) else 0.0
            conflict_boost = min(0.12, float(row.conflict_events_proxy) * 0.035)
            food_boost = max(0.0, (float(row.food_insecurity_phase_proxy) - 2.0) * 0.025)
            immunity_boost = max(0.0, (95.0 - float(row.mcv1_coverage_est)) / 95.0) * 0.11
            rainfall_boost = min(0.08, abs(float(row.rainfall_anomaly_proxy)) * 0.035)
            observed_gap = month_gap(row.period_start, observed_by_location.get(row.location_id, []))
            region_gap = month_gap(row.period_start, observed_by_region.get(row.admin1_region, []))
            near_real_boost = 0.0
            if observed_gap is not None and observed_gap <= 3:
                near_real_boost += 0.10
            elif region_gap is not None and region_gap <= 2:
                near_real_boost += 0.05

            situation_risk = min(
                0.95,
                risk
                + dry_boost
                + conflict_boost
                + food_boost
                + immunity_boost
                + rainfall_boost
                + near_real_boost,
            )
            start_z = -5.10 + 4.35 * situation_risk + (calibration["outbreak_month_rate"] - 0.14) * 0.45
            start_probability = max(0.004, min(0.22, sigmoid(start_z)))

            if wave_remaining <= 0 and row_rng.random() < start_probability:
                wave_remaining = 2 + int(row_rng.random() * (1 + 2 * situation_risk))
                wave_age = 0
                severity = 0.65 + row_rng.random() * 1.65
                population_factor = min(1.6, max(0.65, float(row.under5_population_est) / 25000.0))
                wave_peak = max(
                    outbreak_threshold,
                    int(round(case_scale * (0.65 + 1.75 * situation_risk) * severity * population_factor)),
                )

            if wave_remaining > 0:
                shape = [0.42, 1.0, 0.68, 0.34, 0.16][min(wave_age, 4)]
                jitter = 0.82 + row_rng.random() * 0.36
                cases = int(round(wave_peak * shape * jitter))
                if wave_age == 0:
                    cases = max(outbreak_threshold, cases)
                cases = max(0, min(cases, 220))
                label_type = "synthetic_model"
                label_confidence = round(0.34 + 0.28 * situation_risk, 3)
                wave_age += 1
                wave_remaining -= 1
            else:
                sporadic_probability = min(0.42, 0.08 + 0.35 * situation_risk + (0.08 if prior_cases > 0 else 0))
                if row_rng.random() < sporadic_probability:
                    cases = row_rng.randint(1, max(1, outbreak_threshold - 1))
                else:
                    cases = 0
                label_type = "synthetic_negative"
                label_confidence = round(0.22 + 0.18 * (1.0 - situation_risk), 3)

            cfr = 0.002 + 0.009 * min(1.0, float(row.food_insecurity_phase_proxy) / 5)
            deaths = 0
            if cases >= 20 and row_rng.random() < min(0.45, cases * cfr):
                deaths = max(1, int(round(cases * cfr)))

            rows.append(
                {
                    "country": "Ethiopia",
                    "admin1_region": row.admin1_region,
                    "admin2_zone": row.admin2_zone,
                    "admin3_woreda": row.admin3_woreda,
                    "admin3_pcode": row.admin3_pcode,
                    "woreda_name_clean": row.woreda_name_clean,
                    "location_id": row.location_id,
                    "location_name": row.location_name,
                    "period_start": row.period_start,
                    "target_cases": cases,
                    "target_deaths": deaths,
                    "target_outbreak": int(cases >= outbreak_threshold),
                    "label_type": label_type,
                    "label_confidence": label_confidence,
                    "label_source": "district_time_risk_overlay",
                    "real_flag": 0,
                    "weak_label_flag": 0,
                    "synthetic_flag": 1,
                    "imputed_flag": 1,
                    "guessed_flag": 1,
                    "external_country_flag": 0,
                    "ethiopia_flag": 1,
                    "outbreak_threshold_cases": outbreak_threshold,
                    "source_notes": (
                        "Demo-augmented label generated from district-month risk context. "
                        "Remove with synthetic_flag == 0 for real-only analysis."
                    ),
                    "generation_context": (
                        f"risk={situation_risk:.3f}; dry={int(row.dry_season_flag)}; "
                        f"mcv1={float(row.mcv1_coverage_est):.1f}; food_phase={row.food_insecurity_phase_proxy}; "
                        f"conflict={row.conflict_events_proxy}; rain_anom={float(row.rainfall_anomaly_proxy):.2f}; "
                        f"near_real_gap={observed_gap}; region_gap={region_gap}; month={month}"
                    ),
                }
            )
            prior_cases = cases
    return pd.DataFrame(rows)


def merge_synthetic(real_weak: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    mixed = real_weak.copy()
    if synthetic.empty:
        return mixed
    keys = ["location_id", "period_start"]
    overlay_cols = [
        "target_cases",
        "target_deaths",
        "target_outbreak",
        "label_type",
        "label_confidence",
        "label_source",
        "real_flag",
        "weak_label_flag",
        "synthetic_flag",
        "imputed_flag",
        "guessed_flag",
        "source_notes",
        "generation_context",
    ]
    overlay = synthetic[keys + overlay_cols].copy()
    mixed = mixed.merge(overlay, on=keys, how="left", suffixes=("", "_synthetic"))
    unknown_mask = mixed["label_type"] == "unknown"
    for column in overlay_cols:
        synthetic_col = f"{column}_synthetic"
        if synthetic_col in mixed.columns:
            mixed.loc[unknown_mask, column] = mixed.loc[unknown_mask, synthetic_col]
            mixed = mixed.drop(columns=[synthetic_col])
    return mixed


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_id", "period_start"]).copy()
    grouped = df.groupby("location_id", group_keys=False)
    for lag in [1, 2, 3]:
        df[f"target_cases_lag_{lag}"] = grouped["target_cases"].shift(lag)
    df["target_cases_rolling_3_prev"] = grouped["target_cases"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    df["target_outbreak_lag_1"] = grouped["target_outbreak"].shift(1)
    return df


def write_demo_training_files(mixed: pd.DataFrame, out: OutputPaths) -> dict[str, str]:
    """Write a cleaner demo matrix plus a separate provenance sidecar.

    The matrix is meant for model demos and does not include the obvious label
    provenance columns. The sidecar keeps those fields auditable by key.
    """
    key_cols = ["location_id", "period_start"]
    provenance_cols = [
        "country",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "location_name",
        "target_cases",
        "target_deaths",
        "target_outbreak",
        "label_type",
        "label_confidence",
        "label_source",
        "real_flag",
        "weak_label_flag",
        "synthetic_flag",
        "imputed_flag",
        "guessed_flag",
        "source_notes",
        "generation_context",
    ]
    provenance_cols = [column for column in provenance_cols if column in mixed.columns]
    provenance = mixed[key_cols + [column for column in provenance_cols if column not in key_cols]].copy()

    model_cols = [
        "location_id",
        "period_start",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "admin3_pcode",
        "year",
        "month",
        "target_cases",
        "target_deaths",
        "target_outbreak",
        "population_total_est",
        "under5_population_est",
        "school_age_population_est",
        "population_density_est",
        "mcv1_coverage_est",
        "mcv2_coverage_est",
        "measles_dropout_proxy",
        "susceptible_children_proxy",
        "travel_time_healthcare_min_est",
        "rainfall_anomaly_proxy",
        "dry_season_flag",
        "food_insecurity_phase_proxy",
        "conflict_events_proxy",
        "mvp_risk_score_proxy",
        "target_cases_lag_1",
        "target_cases_lag_2",
        "target_cases_lag_3",
        "target_cases_rolling_3_prev",
        "target_outbreak_lag_1",
    ]
    model_cols = [column for column in model_cols if column in mixed.columns]
    matrix = mixed[model_cols].copy()
    return {
        "demo_model_matrix": str(safe_to_csv(matrix, out.demo_model_matrix)),
        "demo_provenance": str(safe_to_csv(provenance, out.demo_provenance)),
    }


def select_training_export(
    mixed: pd.DataFrame,
    external: pd.DataFrame,
    include_external: bool,
    include_synthetic: bool,
    exclude_synthetic: bool,
    ethiopia_only: bool,
    real_only: bool,
) -> pd.DataFrame:
    selected = mixed.copy()
    if real_only:
        selected = selected[selected["label_type"] == "real_ephi"]
    if exclude_synthetic or not include_synthetic:
        selected = selected[selected["synthetic_flag"] == 0]
    if include_external and not ethiopia_only and not external.empty:
        external_as_training = external.rename(
            columns={"cases": "target_cases", "deaths": "target_deaths"}
        ).copy()
        external_as_training["target_outbreak"] = (
            pd.to_numeric(external_as_training["target_cases"], errors="coerce").fillna(0) >= 5
        ).astype(int)
        for column in selected.columns:
            if column not in external_as_training.columns:
                external_as_training[column] = pd.NA
        external_as_training = external_as_training[selected.columns]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            selected = pd.concat([selected, external_as_training], ignore_index=True)
    if ethiopia_only:
        selected = selected[selected["ethiopia_flag"] == 1]
    return selected


def write_source_inventory(
    out: Path,
    ephi_path: Path,
    admin_boundaries: Path,
    ipc_population: Path,
    who_reported_cases: Path,
    external_downloads: list[dict[str, Any]],
    external: pd.DataFrame,
) -> None:
    rows = [
        {
            "source_id": "ephi_measles_2025_agg_from_line_list",
            "source_name": "EPHI provided Measels_2025_agg_from_LL.xlsx",
            "source_type": "provided_project_file",
            "path_or_url": str(ephi_path),
            "used": ephi_path.exists(),
            "role": "primary_real_ethiopia_labels",
            "notes": "Aggregated weekly measles cases and deaths by region/zone/woreda.",
        },
        {
            "source_id": "synthetic_risk_overlay",
            "source_name": "Deterministic synthetic MVP label overlay",
            "source_type": "generated",
            "path_or_url": "data/processed/measles_woreda_month_synthetic_overlay.csv",
            "used": True,
            "role": "removable_fake_labels_for_mvp_completeness",
            "notes": "Remove with synthetic_flag == 0. Not official surveillance truth.",
        },
        {
            "source_id": "who_ethiopia_don_2023",
            "source_name": "WHO Disease Outbreak News - Measles Ethiopia",
            "source_type": "public_web_reference",
            "path_or_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2023-DON460",
            "used": False,
            "role": "planned_weak_label_context",
            "notes": "Listed for future weak-label extraction; not automatically parsed in this build.",
        },
        {
            "source_id": "hdx_cod_ab_eth",
            "source_name": "HDX/OCHA Ethiopia COD-AB administrative boundaries",
            "source_type": "provided_project_file" if admin_boundaries.exists() else "public_download_reference",
            "path_or_url": str(admin_boundaries) if admin_boundaries.exists() else "https://data.humdata.org/dataset/cod-ab-eth",
            "used": admin_boundaries.exists(),
            "role": "planned_location_pcode_geometry",
            "notes": "Used for exact admin3 p-code, area, and centroid matches where EPHI names align." if admin_boundaries.exists() else "Not materialized in this build; location reference is EPHI-name based.",
        },
        {
            "source_id": "worldpop_age_sex",
            "source_name": "WorldPop age/sex population",
            "source_type": "public_download_reference",
            "path_or_url": "https://hub.worldpop.org/project/categories?id=8",
            "used": False,
            "role": "planned_population_covariates",
            "notes": "Proxy covariates used until rasters are downloaded and aggregated.",
        },
        {
            "source_id": "fewsnet_api",
            "source_name": "FEWS NET Data Warehouse API",
            "source_type": "provided_project_file" if ipc_population.exists() else "public_api_reference",
            "path_or_url": str(ipc_population) if ipc_population.exists() else "https://help.fews.net/fdw/fews-net-api",
            "used": ipc_population.exists(),
            "role": "planned_food_security_covariates",
            "notes": "File is present but current build still uses deterministic food-stress proxies; detailed IPC join is a next step." if ipc_population.exists() else "Proxy food stress used until API extraction is added.",
        },
    ]
    for item in external_downloads:
        rows.append(
            {
                "source_id": item["id"],
                "source_name": item["id"].replace("_", " ").title(),
                "source_type": "public_download",
                "path_or_url": item["url"],
                "used": bool(item.get("available")) and not external.empty,
                "role": item["role"],
                "notes": item.get("status", ""),
            }
        )
    if not any(item["id"] == "who_immunization_reported_cases_export" for item in external_downloads):
        rows.append(
            {
                "source_id": "who_immunization_reported_cases_export",
                "source_name": "WHO Immunization Data Portal reported cases export",
                "source_type": "provided_project_file" if who_reported_cases.exists() else "public_download_optional_large",
                "path_or_url": str(who_reported_cases) if who_reported_cases.exists() else WHO_REPORTED_CASES_XLSX_URL,
                "used": who_reported_cases.exists() and not external.empty,
                "role": "external_who_country_year_reported_cases",
                "notes": "Used from manually downloaded project-root workbook." if who_reported_cases.exists() else "Optional large workbook. Run with --include-who-export to attempt download.",
            }
        )
    safe_to_csv(pd.DataFrame(rows), out)


def write_data_dictionary(out: Path) -> None:
    descriptions = {
        "country": "Country for the row. Ethiopia rows are the target MVP; external rows are support data.",
        "admin1_region": "Ethiopia region from EPHI or boundary source.",
        "admin2_zone": "Ethiopia zone from EPHI or boundary source.",
        "admin3_woreda": "Ethiopia woreda from EPHI or boundary source.",
        "admin3_pcode": "Official admin-3 p-code when available. Blank in EPHI-name-only build.",
        "location_id": "Stable generated location identifier.",
        "location_name": "Human-readable location name.",
        "period_start": "First day of the month represented by this row.",
        "target_cases": "Measles cases for the month. Real for label_type=real_ephi; fake for synthetic labels.",
        "target_deaths": "Measles deaths for the month when available or synthetically estimated.",
        "target_outbreak": "1 when target_cases meets/exceeds the outbreak threshold, else 0.",
        "label_type": "Label provenance: real_ephi, weak_public_report, synthetic_model, synthetic_negative, unknown, or external_public.",
        "label_confidence": "Heuristic confidence from 0 to 1. Real EPHI rows are 1.",
        "label_source": "Short source identifier for the label.",
        "generation_context": "District-month factors used by the demo augmentation model to create a plausible label.",
        "real_flag": "1 only for directly observed real labels.",
        "weak_label_flag": "1 only for public-report weak labels.",
        "synthetic_flag": "1 for fake/simulated labels. Filter synthetic_flag == 0 to remove fake labels.",
        "imputed_flag": "1 when covariates or labels include imputed/proxy values.",
        "guessed_flag": "1 when heuristic estimates are present.",
        "external_country_flag": "1 for non-Ethiopian support rows.",
        "ethiopia_flag": "1 for Ethiopia rows.",
        "source_notes": "Human-readable provenance and caution notes.",
    }
    feature_notes = {
        "population_total_est": "Estimated/proxy total population. Replace with WorldPop aggregation for production.",
        "under5_population_est": "Estimated/proxy under-5 population.",
        "school_age_population_est": "Estimated/proxy school-age population.",
        "population_density_est": "Estimated/proxy population density.",
        "mcv1_coverage_est": "Estimated/proxy MCV1 coverage.",
        "mcv2_coverage_est": "Estimated/proxy MCV2 coverage.",
        "measles_dropout_proxy": "Proxy difference between MCV1 and MCV2 coverage.",
        "susceptible_children_proxy": "Proxy susceptible under-5 count from coverage gap.",
        "travel_time_healthcare_min_est": "Estimated/proxy travel time to healthcare.",
        "rainfall_anomaly_proxy": "Synthetic/proxy rainfall anomaly until CHIRPS is joined.",
        "dry_season_flag": "Seasonality flag used for synthetic risk generation.",
        "food_insecurity_phase_proxy": "Synthetic/proxy food insecurity phase.",
        "conflict_events_proxy": "Synthetic/proxy conflict stress count.",
        "mvp_risk_score_proxy": "Transparent synthetic risk score used to generate removable fake labels.",
        "target_cases_lag_1": "Previous month target_cases after final label selection.",
        "target_cases_lag_2": "Two-month lag target_cases after final label selection.",
        "target_cases_lag_3": "Three-month lag target_cases after final label selection.",
        "target_cases_rolling_3_prev": "Mean target_cases over prior three months.",
        "target_outbreak_lag_1": "Previous month outbreak label.",
    }
    rows = [{"column": key, "description": value} for key, value in descriptions.items()]
    rows.extend({"column": key, "description": value} for key, value in feature_notes.items())
    safe_to_csv(pd.DataFrame(rows), out)


def write_readiness_report(
    out: Path,
    summary: dict[str, Any],
    include_external: bool,
    external_available: bool,
) -> None:
    text = f"""# Modeling Readiness Report

## What this dataset can support

This build creates a practical MVP dataset for Ethiopia woreda-month measles risk modeling. The strongest real outcome source is the provided EPHI workbook, which contributes {summary["real_rows"]} real labeled woreda-month rows across {summary["real_woredas"]} woredas. The mixed MVP file fills unlabeled months with removable synthetic labels so the model and dashboard can be tested end-to-end.

Location reference quality:

- EPHI locations matched to COD-AB p-codes: {summary.get("locations_with_pcode", 0)} of {summary.get("location_reference_rows", 0)}.
- External support rows: {summary.get("external_rows", 0)}.

Recommended modeling modes:

- Real-only academic check: filter `label_type == "real_ephi"`.
- Real plus weak-label mode: filter `synthetic_flag == 0`.
- MVP/demo mode: use `measles_woreda_month_ml_ready_mixed.csv`.

## What it cannot honestly prove yet

The mixed file cannot prove real national Ethiopia forecasting performance because many rows are synthetic. Synthetic rows are useful for UI demos, pipeline testing, and exploring model behavior, but they are not official surveillance truth. Public covariates are currently represented by deterministic proxy features until WorldPop, CHIRPS, COD-AB, FEWS NET, and healthcare-access layers are downloaded and spatially joined.

## External-country data

External-country support data was {"included" if include_external else "not requested"} and was {"available" if external_available else "not available in this run"}. External rows are marked with `external_country_flag = 1` and should be used only for outbreak-shape support, pretraining experiments, or synthetic calibration. They are not Ethiopian ground truth.

## Removing fake labels

Fake labels are intentionally easy to remove:

```text
synthetic_flag == 0
```

For a clean file, run:

```powershell
python scripts/filter_measles_dataset.py --remove-synthetic
```

To also delete the provenance/flag columns after filtering:

```powershell
python scripts/filter_measles_dataset.py --remove-synthetic --drop-flag-columns
```
"""
    safe_write_text(out, text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the measles MVP datasets.")
    parser.add_argument("--ephi-measles", type=Path, default=DEFAULT_EPHI_MEASLES)
    parser.add_argument("--admin-boundaries", type=Path, default=DEFAULT_ADMIN_BOUNDARIES)
    parser.add_argument("--ipc-population", type=Path, default=DEFAULT_IPC_POPULATION)
    parser.add_argument("--who-reported-cases", type=Path, default=DEFAULT_WHO_REPORTED_CASES)
    parser.add_argument("--start-month", default="2021-01-01")
    parser.add_argument("--end-month", default="2025-12-01")
    parser.add_argument("--outbreak-threshold", type=int, default=5)
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--include-who-export", action="store_true")
    parser.add_argument("--include-synthetic", action="store_true", default=True)
    parser.add_argument("--exclude-synthetic", action="store_true")
    parser.add_argument("--ethiopia-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out = paths()

    try:
        ephi_raw = read_ephi_measles(args.ephi_measles)
        ephi_monthly = aggregate_ephi_to_month(ephi_raw, args.outbreak_threshold)
        location = build_location_reference(ephi_monthly, args.admin_boundaries)
        base = build_base_panel(location, args.start_month, args.end_month)
        base = add_proxy_features(base, ephi_monthly)
        real_weak = build_real_weak_panel(base, ephi_monthly, args.outbreak_threshold)

        external_downloads: list[dict[str, Any]] = []
        if args.include_external:
            external_downloads = download_external_sources(args.force_download, args.include_who_export)
        external = normalize_external_sources()

        synthetic = generate_synthetic_overlay(real_weak, external, args.outbreak_threshold)
        mixed = merge_synthetic(real_weak, synthetic)
        mixed = add_lag_features(mixed)
        real_weak_with_lags = add_lag_features(real_weak)
        selected = select_training_export(
            mixed=mixed,
            external=external,
            include_external=args.include_external,
            include_synthetic=args.include_synthetic,
            exclude_synthetic=args.exclude_synthetic,
            ethiopia_only=args.ethiopia_only,
            real_only=args.real_only,
        )

        actual_outputs: dict[str, str] = {}
        actual_outputs["location_reference"] = str(safe_to_csv(location, out.location_reference))
        actual_outputs["real_weak"] = str(safe_to_csv(real_weak_with_lags, out.real_weak))
        actual_outputs["synthetic_overlay"] = str(safe_to_csv(synthetic, out.synthetic_overlay))
        actual_outputs["mixed"] = str(safe_to_csv(mixed, out.mixed))
        actual_outputs["external_normalized"] = str(safe_to_csv(external, out.external_normalized))
        actual_outputs["selected_training"] = str(safe_to_csv(selected, out.selected_training))
        actual_outputs.update(write_demo_training_files(mixed, out))
        write_source_inventory(
            out.source_inventory,
            args.ephi_measles,
            args.admin_boundaries,
            args.ipc_population,
            args.who_reported_cases,
            external_downloads,
            external,
        )
        write_data_dictionary(out.data_dictionary)

        summary = {
            "ephi_input": str(args.ephi_measles),
            "admin_boundaries_input": str(args.admin_boundaries),
            "ipc_population_input": str(args.ipc_population),
            "who_reported_cases_input": str(args.who_reported_cases),
            "real_weekly_rows": int(len(ephi_raw)),
            "real_monthly_rows": int(len(ephi_monthly)),
            "real_rows": int((mixed["label_type"] == "real_ephi").sum()),
            "real_woredas": int(ephi_monthly["woreda"].nunique()),
            "location_reference_rows": int(len(location)),
            "locations_with_pcode": int((location["admin3_pcode"].astype(str).str.strip() != "").sum()),
            "real_weak_rows": int(len(real_weak_with_lags)),
            "synthetic_overlay_rows": int(len(synthetic)),
            "mixed_rows": int(len(mixed)),
            "external_rows": int(len(external)),
            "selected_training_rows": int(len(selected)),
            "start_month": args.start_month,
            "end_month": args.end_month,
            "outbreak_threshold": args.outbreak_threshold,
            "outputs": {
                field: actual_outputs.get(field, str(getattr(out, field)))
                for field in out.__dataclass_fields__
            },
        }
        safe_write_text(out.summary, json.dumps(summary, indent=2))
        write_readiness_report(out.readiness_report, summary, args.include_external, not external.empty)

        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
