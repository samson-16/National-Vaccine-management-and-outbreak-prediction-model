"""Run live next-month measles risk prediction from a saved model artifact.

The script is intentionally small for the defense demo: it loads the saved
XGBoost artifact produced by train_measles_mvp_model.py, rebuilds the latest
feature matrix, and writes a CSV that can be consumed by the dashboard or API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_measles_mvp_model import (
    DEFAULT_ETHIOPIA_INPUT,
    DEFAULT_OUTPUT_DIR,
    FeaturePreprocessor,
    add_external_priors,
    json_safe,
    load_ethiopia_matrix,
    make_next_month_alert_woredas,
)


DEFAULT_ARTIFACT = DEFAULT_OUTPUT_DIR / "xgboost_model_artifact.joblib"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "live_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ETHIOPIA_INPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon-months", type=int, default=None)
    parser.add_argument("--all-periods", action="store_true")
    return parser.parse_args()


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities.astype(float)


def build_prediction_output(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    preprocessor = FeaturePreprocessor.from_state(artifact["preprocessor_state"])
    X = preprocessor.transform(df)
    probabilities = predict_probabilities(artifact["model"], X)
    threshold = float(artifact["threshold"])
    bins = artifact.get("risk_bucket_bins", [-0.001, 0.25, 0.5, 0.75, 1.001])
    labels = artifact.get("risk_bucket_labels", ["low", "moderate", "high", "very_high"])
    output = df[
        [
            "location_id",
            "period_start",
            "prediction_period_start",
            "admin1_region",
            "admin2_zone",
            "admin3_woreda",
            "year",
            "month",
            "target_outbreak_next_month",
            "target_cases_next_month",
        ]
    ].copy()
    output["model_name"] = artifact.get("model_name", "xgboost")
    output["risk_probability"] = probabilities
    output["risk_prediction"] = (probabilities >= threshold).astype(int)
    output["actual_label_available"] = output["target_outbreak_next_month"].notna().astype(int)
    output["risk_bucket"] = pd.cut(output["risk_probability"], bins=bins, labels=labels).astype(str)
    return output


def add_alert_fields(predictions: pd.DataFrame) -> pd.DataFrame:
    latest_period = predictions["prediction_period_start"].max()
    latest = (
        predictions[predictions["prediction_period_start"].eq(latest_period)]
        .sort_values("risk_probability", ascending=False)
        .reset_index(drop=True)
    )
    alerts = make_next_month_alert_woredas(latest)
    alert_columns = [
        "location_id",
        "alert_role",
        "alert_level",
        "source_zone_high_risk_count",
        "source_zone_very_high_count",
        "source_zone_max_risk_probability",
        "source_zone_top_source_woreda",
        "alert_reason",
    ]
    out = latest.merge(alerts[alert_columns], on="location_id", how="left")
    out["alert_role"] = out["alert_role"].fillna("none")
    out["alert_level"] = out["alert_level"].fillna("none")
    out["alert_reason"] = out["alert_reason"].fillna("No high-risk same-zone alert generated.")
    for column in [
        "source_zone_high_risk_count",
        "source_zone_very_high_count",
        "source_zone_max_risk_probability",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    out["source_zone_top_source_woreda"] = out["source_zone_top_source_woreda"].fillna("")
    return out


def main() -> int:
    args = parse_args()
    if not args.artifact.exists():
        raise FileNotFoundError(f"Missing model artifact: {args.artifact}")
    artifact = joblib.load(args.artifact)
    horizon_months = args.horizon_months or int(artifact.get("metadata", {}).get("horizon_months") or 1)
    df = load_ethiopia_matrix(args.input, horizon_months)
    month_features = pd.DataFrame(artifact["month_features"])
    df = add_external_priors(df, month_features)
    predictions = build_prediction_output(df, artifact)
    output = predictions if args.all_periods else add_alert_fields(predictions)
    output = output.sort_values("risk_probability", ascending=False).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "artifact": str(args.artifact),
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(output)),
        "prediction_period_start": str(output["prediction_period_start"].iloc[0]) if len(output) else None,
        "high_or_very_high_rows": int(output["risk_bucket"].isin(["high", "very_high"]).sum()) if len(output) else 0,
    }
    print(json.dumps(json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
