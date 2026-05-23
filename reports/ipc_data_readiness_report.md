# IPC Food Stress Data Readiness Report

## What was added

The project now uses real public FEWS NET Ethiopia IPC Phase 3+ population estimate time-series data.
The manually downloaded `IPCPopulation.csv` is metadata/catalog-level data; the actual monthly population estimates come from the public FEWS NET `ipcpopulationsize` endpoint.

- Raw IPC rows: `157`
- Monthly feature rows: `60`
- Joined model rows: `2640`
- Local metadata Ethiopia level: `national_only`
- Scenario counts: `{'Current Situation': 51, 'Most Likely': 9}`

## Latest Monthly Features

- 2025-05-01: 14.50 million people Phase 3+ (Most Likely)
- 2025-06-01: 14.50 million people Phase 3+ (Most Likely)
- 2025-07-01: 14.50 million people Phase 3+ (Most Likely)
- 2025-08-01: 14.50 million people Phase 3+ (Most Likely)
- 2025-09-01: 11.50 million people Phase 3+ (Current Situation)
- 2025-10-01: 11.50 million people Phase 3+ (Current Situation)
- 2025-11-01: 11.50 million people Phase 3+ (Current Situation)
- 2025-12-01: 9.50 million people Phase 3+ (Current Situation)

## Modeling Use

Use `data/processed/measles_training_demo_model_matrix_with_vaccine_ipc.csv` for the next retraining run.
These IPC features are national monthly food-stress covariates. They do not provide woreda-level IPC variation in the current public file.
