# Vaccine-Preventable Disease Risk Decision-Support System

This project is a public-health decision-support MVP for Ethiopia vaccine-preventable disease preparedness. It uses historical, de-identified, aggregated measles surveillance data to estimate woreda-level next-month outbreak risk, and AFP surveillance data to estimate polio surveillance-risk and preparedness alerts.

The system does not replace official surveillance, clinical judgment, or public-health authority decisions.

## What The Demo Shows

- End-to-end preprocessing pipeline from real Ethiopia measles workbooks to woreda-month training data.
- XGBoost as the primary next-month outbreak-risk model.
- Random Forest, KNN, and Logistic Regression as comparison models.
- Confusion matrix, model comparison, feature importance, and top-risk outputs.
- Live prediction from a saved model artifact without retraining.
- Polio AFP surveillance-risk modeling with 2021-2023 calibration, 2024 validation, and 2025 testing/demo evaluation.
- Streamlit dashboard for defense presentation and decision-support visualization.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main Workflow

Prepare the real Ethiopia measles model matrix:

```powershell
python scripts\prepare_real_measles_updates.py
```

Train XGBoost, compare models, save the model artifact, and generate dashboard outputs:

```powershell
python scripts\train_measles_mvp_model.py --primary-model xgboost --output-dir model_outputs_xgboost
```

Run live next-month prediction from the saved artifact:

```powershell
python scripts\predict_measles_risk.py --input data\processed\measles_training_real_model_matrix.csv --output model_outputs_xgboost\live_predictions.csv
```

Train the Polio AFP Surveillance Risk Module:

```powershell
python scripts\train_polio_surveillance_risk_model.py
```

Open the dashboard:

```powershell
streamlit run dashboard\app.py
```

Run the API for frontend integration:

```powershell
uvicorn api.main:app --reload --port 8000
```

## Key Outputs

```text
model_outputs_xgboost/model_comparison.csv
model_outputs_xgboost/confusion_matrix.csv
model_outputs_xgboost/feature_importance.csv
model_outputs_xgboost/top_risk_woredas_latest.csv
model_outputs_xgboost/new_2025_outbreak_locations.csv
model_outputs_xgboost/next_month_alert_woredas.csv
model_outputs_xgboost/measles_next_month_predictions.csv
model_outputs_xgboost/live_predictions.csv
model_outputs_xgboost/xgboost_model_artifact.joblib
data/processed/polio_surveillance_training_matrix.csv
model_outputs_polio_afp/polio_afp_next_month_predictions.csv
model_outputs_polio_afp/polio_afp_next_month_surveillance_alerts.csv
model_outputs_polio_afp/polio_afp_zone_watch_alerts.csv
model_outputs_polio_afp/polio_afp_signal_trends.csv
model_outputs_polio_afp/polio_afp_preparedness_recommendations.csv
model_outputs_polio_afp/polio_surveillance_model_comparison.csv
model_outputs_polio_afp/polio_surveillance_feature_importance.csv
model_outputs_polio_afp/polio_surveillance_confusion_matrix.csv
model_outputs_polio_afp/polio_surveillance_evaluation_metrics.json
model_outputs_polio_afp/polio_surveillance_model_artifact.joblib
reports/training_readiness_report.md
reports/polio_surveillance_training_report.md
```

## API Endpoints

The frontend should read model outputs through the API rather than directly reading CSV files.

```text
GET  /api/health
GET  /api/alerts/measles/latest
GET  /api/alerts/polio/latest
GET  /api/alerts/combined/latest
GET  /api/woredas/{woreda}/risk-summary
POST /api/case-reports
POST /api/afp-reports
POST /api/vaccine-stock
```

For the MVP, POST inputs are stored as CSV files under `data/api_inputs/`. In production, these endpoints should write to the vaccine-management database.

## Output Meaning

`target_outbreak_next_month` and `target_cases_next_month` are actual future labels when available. For the latest future prediction month, these are blank because the real future outcome is unknown.

Model outputs are:

```text
risk_probability  Model-estimated next-month outbreak risk.
risk_bucket       low, moderate, high, or very_high.
risk_prediction   Binary alert using the tuned XGBoost threshold.
```

Risk buckets:

```text
low        0.00-0.25
moderate   0.25-0.50
high       0.50-0.75
very_high  0.75-1.00
```

## Polio AFP Module Meaning

The polio module is an AFP surveillance-risk and preparedness alerting system, not a confirmed polio outbreak prediction system. The current AFP data has suspected poliovirus surveillance signals, but no explicit confirmed WPV/cVDPV-positive rows in the processed summary.

The derived polio targets are:

```text
target_high_surveillance_risk_next_month
target_poor_stool_adequacy_next_month
target_delayed_reporting_next_month
target_under_vaccinated_afp_signal_next_month
target_suspected_poliovirus_signal_next_month
```

Polio split:

```text
training/calibration: prediction years 2021-2023
validation:           prediction year 2024
testing/demo:         prediction year 2025
```

Polio alert outputs include:

- `critical`: suspected poliovirus surveillance signal or very high predicted surveillance risk.
- `high`: high surveillance-risk or under-vaccinated AFP signal.
- `watch`: same zone as a high-risk source woreda.
- `monitor`: moderate surveillance weakness or component signal.

## Alert Outputs

`new_2025_outbreak_locations.csv` lists woredas whose first recorded outbreak occurred in 2025.

`next_month_alert_woredas.csv` includes:

- `high_risk_source`: the woreda itself is high or very-high risk.
- `nearby_same_zone`: the woreda is in the same zone as a high-risk source woreda.
- `critical`: own bucket is very-high.
- `high`: own bucket is high.
- `watch`: nearby same-zone woreda near a very-high source.
- `monitor`: nearby same-zone woreda near a high source.

## Data Privacy

Direct identifiers are excluded from processed training data. Raw case IDs are replaced by one-way record hashes in the de-identified line-list output, and the model is trained on aggregated woreda-month features.

## Defense Story

The best way to present the project is:

> A vaccine-management decision-support system with an outbreak-risk prediction module. The ML layer identifies woredas that may need vaccination outreach, monitoring, or logistics preparedness, while the backend vaccine-management repo handles stock and operational workflows.

For polio:

> Because confirmed polio labels are not available in the current AFP files, the system uses AFP surveillance quality, vaccination status, reporting delays, stool adequacy, and suspected poliovirus signals to predict surveillance-risk alerts and preparedness priorities.

