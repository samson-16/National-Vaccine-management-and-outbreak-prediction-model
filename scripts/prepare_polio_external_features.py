"""Prepare downloaded public polio/AFP support datasets into model-ready features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
POLIO_VACCINE_DIR = RAW_DIR / "polio_vaccine"
POLIO_SURVEILLANCE_DIR = RAW_DIR / "polio_surveillance"
WASH_DIR = RAW_DIR / "wash"
LOG_PATH = RAW_DIR / "polio_external_download_log.csv"


def read_gho_csv(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["year", value_name])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["year", value_name])
    if "Dim1" in df.columns and df["Dim1"].notna().any():
        total = df[df["Dim1"].astype(str).eq("RESIDENCEAREATYPE_TOTL")].copy()
        if not total.empty:
            df = total
    year_col = "TimeDim" if "TimeDim" in df.columns else "time"
    value_col = "NumericValue" if "NumericValue" in df.columns else "Value" if "Value" in df.columns else "value"
    if year_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["year", value_name])
    out = df[[year_col, value_col]].copy()
    out.columns = ["year", value_name]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    if out[value_name].isna().all() and "Value" in df.columns and value_col != "Value":
        out[value_name] = pd.to_numeric(df["Value"], errors="coerce")
    out = out.dropna(subset=["year"]).drop_duplicates("year", keep="last")
    out["year"] = out["year"].astype(int)
    return out.sort_values("year")


def prepare_vaccine_features() -> pd.DataFrame:
    sources = {
        "pol3_coverage_real_national": POLIO_VACCINE_DIR / "who_ethiopia_pol3_coverage.csv",
        "ipv1_coverage_real_national": POLIO_VACCINE_DIR / "who_ethiopia_ipv1_coverage.csv",
        "ipv2_coverage_real_national": POLIO_VACCINE_DIR / "who_ethiopia_ipv2_coverage.csv",
    }
    frames = [read_gho_csv(path, column) for column, path in sources.items()]
    years = sorted(set().union(*[set(frame["year"].dropna().astype(int)) for frame in frames if not frame.empty]))
    if not years:
        out = pd.DataFrame(columns=["year", *sources.keys()])
    else:
        out = pd.DataFrame({"year": years})
        for frame in frames:
            out = out.merge(frame, on="year", how="left")
    for column in sources:
        out[column + "_available_flag"] = out[column].notna().astype(int) if column in out.columns else 0
    out["country"] = "Ethiopia"
    out["data_level"] = "national_annual"
    out["source_notes"] = "WHO GHO OData where available; missing IPV endpoints are retained as empty/manual-needed fields."
    return out


def prepare_surveillance_benchmarks() -> pd.DataFrame:
    path = POLIO_SURVEILLANCE_DIR / "owid_polio_screening_testing.csv"
    if not path.exists():
        return pd.DataFrame(columns=["country", "year", "source_dataset", "source_notes"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["country", "year", "source_dataset", "source_notes"])
    renamed = df.rename(columns={"Entity": "country", "Year": "year", "Code": "country_code"})
    if "country" not in renamed.columns or "year" not in renamed.columns:
        return pd.DataFrame(columns=["country", "year", "source_dataset", "source_notes"])
    renamed["source_dataset"] = "Our World in Data polio screening and testing"
    renamed["source_notes"] = "External benchmark only; not used as Ethiopia woreda labels."
    return renamed


def prepare_wash_features() -> pd.DataFrame:
    sources = {
        "open_defecation_pct": WASH_DIR / "who_ethiopia_open_defecation.csv",
        "basic_sanitation_pct": WASH_DIR / "who_ethiopia_basic_sanitation.csv",
        "basic_drinking_water_pct": WASH_DIR / "who_ethiopia_basic_water.csv",
        "safely_managed_water_pct": WASH_DIR / "who_ethiopia_safely_managed_water.csv",
    }
    frames = [read_gho_csv(path, column) for column, path in sources.items()]
    years = sorted(set().union(*[set(frame["year"].dropna().astype(int)) for frame in frames if not frame.empty]))
    if not years:
        out = pd.DataFrame(columns=["year", *sources.keys()])
    else:
        out = pd.DataFrame({"year": years})
        for frame in frames:
            out = out.merge(frame, on="year", how="left")
    for column in sources:
        out[column + "_available_flag"] = out[column].notna().astype(int) if column in out.columns else 0
    out["country"] = "Ethiopia"
    out["data_level"] = "national_annual"
    out["source_notes"] = "WHO GHO WASH indicators; used as sanitation/water vulnerability context."
    return out


def write_report(vaccine: pd.DataFrame, surveillance: pd.DataFrame, wash: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_summary = ""
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        counts = log["status"].value_counts().to_dict() if "status" in log.columns else {}
        log_summary = json.dumps(counts, indent=2)
    text = f"""# Polio External Data Readiness Report

## What was prepared

- Vaccine coverage rows: `{len(vaccine)}`
- External surveillance benchmark rows: `{len(surveillance)}`
- WASH vulnerability rows: `{len(wash)}`

## Download status summary

```json
{log_summary}
```

## Notes

The AFP workbooks remain the primary Ethiopia surveillance source. These external files are supporting covariates or benchmarks only; they are not Ethiopia woreda-level polio labels.

Missing IPV, WorldPop, Healthsites, IOM/HDX, or ACLED resources should be listed as manual-needed in `data/raw/polio_external_download_log.csv` when direct public downloads are unavailable.
"""
    (REPORTS_DIR / "polio_external_data_readiness_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    vaccine = prepare_vaccine_features()
    surveillance = prepare_surveillance_benchmarks()
    wash = prepare_wash_features()
    vaccine.to_csv(PROCESSED_DIR / "polio_vaccine_coverage_annual.csv", index=False)
    surveillance.to_csv(PROCESSED_DIR / "polio_external_surveillance_benchmarks.csv", index=False)
    wash.to_csv(PROCESSED_DIR / "polio_wash_vulnerability_features.csv", index=False)
    write_report(vaccine, surveillance, wash)
    print(
        json.dumps(
            {
                "vaccine_rows": len(vaccine),
                "surveillance_rows": len(surveillance),
                "wash_rows": len(wash),
                "outputs": {
                    "vaccine": str(PROCESSED_DIR / "polio_vaccine_coverage_annual.csv"),
                    "surveillance": str(PROCESSED_DIR / "polio_external_surveillance_benchmarks.csv"),
                    "wash": str(PROCESSED_DIR / "polio_wash_vulnerability_features.csv"),
                    "report": str(REPORTS_DIR / "polio_external_data_readiness_report.md"),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
