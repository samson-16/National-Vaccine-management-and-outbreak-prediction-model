"""FastAPI entrypoint for VMOPS model outputs and incoming reports."""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AfpReportIn, CaseReportIn, StoredRecordOut, VaccineStockIn
from api.services import (
    append_input_record,
    combined_alerts,
    latest_measles_alerts,
    latest_polio_alerts,
    woreda_risk_summary,
)


app = FastAPI(
    title="VMOPS Risk API",
    description="API for measles outbreak-risk outputs, polio AFP surveillance-risk alerts, and operational inputs.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/alerts/measles/latest")
def get_measles_alerts(
    limit: int = Query(default=100, ge=1, le=5000),
    region: str | None = None,
    zone: str | None = None,
) -> list[dict]:
    return latest_measles_alerts(limit=limit, region=region, zone=zone)


@app.get("/api/alerts/polio/latest")
def get_polio_alerts(
    limit: int = Query(default=100, ge=1, le=5000),
    region: str | None = None,
    zone: str | None = None,
) -> list[dict]:
    return latest_polio_alerts(limit=limit, region=region, zone=zone)


@app.get("/api/alerts/combined/latest")
def get_combined_alerts(
    limit: int = Query(default=100, ge=1, le=5000),
    region: str | None = None,
    zone: str | None = None,
) -> list[dict]:
    return combined_alerts(limit=limit, region=region, zone=zone)


@app.get("/api/woredas/{woreda}/risk-summary")
def get_woreda_risk_summary(woreda: str) -> dict:
    return woreda_risk_summary(woreda)


@app.post("/api/case-reports", response_model=StoredRecordOut)
def create_case_report(report: CaseReportIn) -> StoredRecordOut:
    record_id, path = append_input_record("case_reports", report.model_dump(mode="json"))
    return StoredRecordOut(status="stored", id=record_id, path=str(path))


@app.post("/api/afp-reports", response_model=StoredRecordOut)
def create_afp_report(report: AfpReportIn) -> StoredRecordOut:
    record_id, path = append_input_record("afp_reports", report.model_dump(mode="json"))
    return StoredRecordOut(status="stored", id=record_id, path=str(path))


@app.post("/api/vaccine-stock", response_model=StoredRecordOut)
def create_vaccine_stock(report: VaccineStockIn) -> StoredRecordOut:
    record_id, path = append_input_record("vaccine_stock", report.model_dump(mode="json"))
    return StoredRecordOut(status="stored", id=record_id, path=str(path))
