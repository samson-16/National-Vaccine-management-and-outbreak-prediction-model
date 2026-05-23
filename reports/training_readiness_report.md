# Training Readiness Report

## What was trained

The MVP model predicts next-month measles outbreak risk for Ethiopia woreda-month rows.
The current-month target columns are excluded from the feature matrix to avoid direct leakage.
XGBoost is the primary model for the final project; Random Forest and KNN are comparison models.

- Primary model: `xgboost`
- Training backend: `scikit-learn`
- Decision threshold: `0.80`
- Test rows: `20185`
- Test positive rows: `146`

## Time-Based 2025 Test Metrics

- Precision: `0.340`
- Recall: `0.658`
- F1: `0.449`
- F2: `0.554`
- ROC-AUC: `0.914`
- PR-AUC: `0.397`

## Model Comparison

- `xgboost` (primary): test F2 `0.554`, recall `0.658`, precision `0.340`
- `random_forest` (comparison): test F2 `0.559`, recall `0.664`, precision `0.342`
- `logistic_regression` (comparison): test F2 `0.441`, recall `0.479`, precision `0.335`
- `knn` (comparison): test F2 `0.179`, recall `0.295`, precision `0.069`

## Defense Outputs

- `top_risk_woredas_latest.csv`: `1835` latest prediction rows sorted by risk.
- `new_2025_outbreak_locations.csv`: `51` woredas whose first recorded outbreak occurred in 2025.
- `next_month_alert_woredas.csv`: `1553` high-risk or same-zone nearby alert rows.

## External Resource Use

External-country measles rows were not appended to Ethiopia evaluation.
They were summarized into monthly seasonality and outbreak-shape priors only.

- External source file: `C:\Projects\VMOPS\data\processed\external_measles_normalized_new.csv`
- External rows available: `20345`
- External monthly rows used: `2328`
- External locations used: `316`

## Top Model Drivers

- `under15_confirmed_cases`: 0.168 (other)
- `suspected_records`: 0.052 (other)
- `line_list_records`: 0.042 (other)
- `ipc_phase3plus_population_high`: 0.038 (population)
- `mcv2_coverage_real_national`: 0.038 (immunity_proxy)
- `epi_linked_cases`: 0.023 (other)
- `year`: 0.019 (other)
- `admin1_region_Somali`: 0.017 (region)
- `mcv1_coverage_real_national`: 0.017 (immunity_proxy)
- `unknown_vaccine_share_confirmed`: 0.015 (other)

## Latest High-Risk Woredas

- Goro (Bale) (Oromia): 0.984 risk for 2026-01-01
- Harena Buluk (Oromia): 0.977 risk for 2026-01-01
- Gelana (West Guji) (Oromia): 0.975 risk for 2026-01-01
- Dasenech /Kuraz (South Ethiopia): 0.886 risk for 2026-01-01
- Adola Rede (Oromia): 0.884 risk for 2026-01-01
- Karat Zuria (South Ethiopia): 0.879 risk for 2026-01-01
- Bena Tsemay (South Ethiopia): 0.869 risk for 2026-01-01
- Saba Boru (Oromia): 0.866 risk for 2026-01-01
- Salamago (South Ethiopia): 0.866 risk for 2026-01-01
- Hamer (South Ethiopia): 0.862 risk for 2026-01-01

## What this can support

This is suitable for a defense demo, feature-importance explanation, and dashboard risk ranking.
It is strongest as a prototype decision-support model, not as an official national surveillance result.

## Important limitations

This run used the real Ethiopia measles update line-list aggregation. Zero-filled panel months represent no line-list record for that woreda-month, not synthetic generated labels.
Public external-country rows support outbreak behavior assumptions, but they are not Ethiopian ground truth.
Final academic claims should clearly state the label source and any zero-fill/imputation assumptions.
