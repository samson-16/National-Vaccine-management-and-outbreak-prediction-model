"""CSV-backed services for API inputs and model-output alerts."""

from __future__ import annotations

import csv
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_INPUT_DIR = ROOT / "data" / "api_inputs"
MEASLES_ALERTS = ROOT / "model_outputs_xgboost" / "next_month_alert_woredas.csv"
MEASLES_TOP_RISK = ROOT / "model_outputs_xgboost" / "top_risk_woredas_latest.csv"
POLIO_ALERTS = ROOT / "model_outputs_polio_afp" / "polio_afp_next_month_surveillance_alerts.csv"
POLIO_RECOMMENDATIONS = ROOT / "model_outputs_polio_afp" / "polio_afp_preparedness_recommendations.csv"


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def records_from_csv(path: Path, limit: int = 100, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    filters = filters or {}
    for column, value in filters.items():
        if value and column in df.columns:
            df = df[df[column].astype(str).str.casefold().eq(value.casefold())]
    if "alert_level" in df.columns:
        order = {"critical": 0, "high": 1, "watch": 2, "monitor": 3, "none": 4}
        df["_alert_order"] = df["alert_level"].map(order).fillna(9)
        sort_columns = ["_alert_order"]
        ascending = [True]
        if "risk_probability" in df.columns:
            sort_columns.append("risk_probability")
            ascending.append(False)
        if "high_surveillance_risk_probability" in df.columns:
            sort_columns.append("high_surveillance_risk_probability")
            ascending.append(False)
        df = df.sort_values(sort_columns, ascending=ascending)
        df = df.drop(columns=["_alert_order"])
    return [
        {str(key): clean_value(value) for key, value in row.items()}
        for row in df.head(limit).to_dict(orient="records")
    ]


def split_text_field(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def normalize_measles_alert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "disease": "measles",
        "location_id": row.get("location_id"),
        "prediction_period_start": row.get("prediction_period_start"),
        "region": row.get("admin1_region"),
        "zone": row.get("admin2_zone"),
        "woreda": row.get("admin3_woreda"),
        "alert_level": row.get("alert_level"),
        "alert_role": row.get("alert_role"),
        "risk_probability": row.get("risk_probability"),
        "risk_bucket": row.get("risk_bucket"),
        "alert_reasons": [row.get("alert_reason")] if row.get("alert_reason") else [],
        "recommended_action": ["prioritize measles vaccination outreach and monitoring"],
    }


def normalize_polio_alert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "disease": "polio_afp",
        "location_id": row.get("location_id"),
        "prediction_period_start": row.get("prediction_period_start"),
        "region": row.get("admin1_region"),
        "zone": row.get("admin2_zone"),
        "woreda": row.get("admin3_woreda"),
        "alert_level": row.get("alert_level"),
        "alert_role": row.get("alert_role"),
        "probabilities": {
            "high_surveillance_risk": row.get("high_surveillance_risk_probability"),
            "poor_stool_adequacy": row.get("poor_stool_adequacy_probability"),
            "delayed_reporting": row.get("delayed_reporting_probability"),
            "under_vaccinated_afp": row.get("under_vaccinated_afp_probability"),
            "suspected_poliovirus": row.get("suspected_poliovirus_probability"),
        },
        "alert_reasons": split_text_field(row.get("alert_reasons")),
        "recommended_action": split_text_field(row.get("recommended_action")),
    }


def latest_measles_alerts(limit: int = 100, region: str | None = None, zone: str | None = None) -> list[dict[str, Any]]:
    rows = records_from_csv(MEASLES_ALERTS, limit=limit, filters={"admin1_region": region or "", "admin2_zone": zone or ""})
    return [normalize_measles_alert(row) for row in rows]


def latest_polio_alerts(limit: int = 100, region: str | None = None, zone: str | None = None) -> list[dict[str, Any]]:
    rows = records_from_csv(POLIO_ALERTS, limit=limit, filters={"admin1_region": region or "", "admin2_zone": zone or ""})
    return [normalize_polio_alert(row) for row in rows]


def combined_alerts(limit: int = 100, region: str | None = None, zone: str | None = None) -> list[dict[str, Any]]:
    records = latest_measles_alerts(limit=limit, region=region, zone=zone) + latest_polio_alerts(limit=limit, region=region, zone=zone)
    order = {"critical": 0, "high": 1, "watch": 2, "monitor": 3, None: 9}
    records.sort(key=lambda item: (order.get(item.get("alert_level"), 9), item.get("disease", ""), item.get("woreda") or ""))
    return records[:limit]


def woreda_risk_summary(woreda: str) -> dict[str, Any]:
    key = woreda.casefold()
    measles_rows = [
        row
        for row in latest_measles_alerts(limit=5000)
        if str(row.get("woreda", "")).casefold() == key
    ]
    polio_rows = [
        row
        for row in latest_polio_alerts(limit=5000)
        if str(row.get("woreda", "")).casefold() == key
    ]
    return {
        "woreda": woreda,
        "measles_alerts": measles_rows,
        "polio_afp_alerts": polio_rows,
        "has_any_alert": bool(measles_rows or polio_rows),
    }


def append_input_record(kind: str, payload: dict[str, Any]) -> tuple[str, Path]:
    API_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    record = {
        "id": record_id,
        "received_at": datetime.now(timezone.utc).isoformat(),
        **{key: clean_value(value) for key, value in payload.items()},
    }
    path = API_INPUT_DIR / f"{kind}.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return record_id, path
