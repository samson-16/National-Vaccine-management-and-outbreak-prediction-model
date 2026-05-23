"""Request and response schemas for the VMOPS API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class CaseReportIn(BaseModel):
    disease: Literal["measles", "polio_afp", "other"]
    region: str
    zone: str
    woreda: str
    onset_date: date | None = None
    notification_date: date | None = None
    age_years: float | None = Field(default=None, ge=0)
    sex: Literal["M", "F", "unknown"] = "unknown"
    vaccination_status: str | None = None
    lab_result: str | None = None
    final_classification: str | None = None
    outcome: str | None = None


class AfpReportIn(BaseModel):
    region: str
    zone: str
    woreda: str
    onset_date: date | None = None
    notification_date: date | None = None
    investigation_date: date | None = None
    age_years: float | None = Field(default=None, ge=0)
    sex: Literal["M", "F", "unknown"] = "unknown"
    vx_status_investigation: int | None = None
    vx_status_validation: int | None = None
    date_first_collected: date | None = None
    date_second_collected: date | None = None
    date_received_at_lab: date | None = None
    date_result_received_from_lab: date | None = None
    lab_result: str | None = None
    final_classification: str | None = None


class VaccineStockIn(BaseModel):
    region: str
    zone: str
    woreda: str
    facility: str | None = None
    vaccine: str
    stock_on_hand: int = Field(ge=0)
    doses_issued: int | None = Field(default=None, ge=0)
    doses_used: int | None = Field(default=None, ge=0)
    wastage: int | None = Field(default=None, ge=0)
    reporting_date: date


class StoredRecordOut(BaseModel):
    status: Literal["stored"]
    id: str
    path: str
