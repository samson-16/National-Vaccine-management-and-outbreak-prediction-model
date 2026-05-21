# Modeling Readiness Report

## What this dataset can support

This build creates a practical MVP dataset for Ethiopia woreda-month measles risk modeling. The strongest real outcome source is the provided EPHI workbook, which contributes 70 real labeled woreda-month rows across 44 woredas. The mixed MVP file fills unlabeled months with removable synthetic labels so the model and dashboard can be tested end-to-end.

Location reference quality:

- EPHI locations matched to COD-AB p-codes: 28 of 44.
- External support rows: 20345.

Recommended modeling modes:

- Real-only academic check: filter `label_type == "real_ephi"`.
- Real plus weak-label mode: filter `synthetic_flag == 0`.
- MVP/demo mode: use `measles_woreda_month_ml_ready_mixed.csv`.

## What it cannot honestly prove yet

The mixed file cannot prove real national Ethiopia forecasting performance because many rows are synthetic. Synthetic rows are useful for UI demos, pipeline testing, and exploring model behavior, but they are not official surveillance truth. Public covariates are currently represented by deterministic proxy features until WorldPop, CHIRPS, COD-AB, FEWS NET, and healthcare-access layers are downloaded and spatially joined.

## External-country data

External-country support data was included and was available. External rows are marked with `external_country_flag = 1` and should be used only for outbreak-shape support, pretraining experiments, or synthetic calibration. They are not Ethiopian ground truth.

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
