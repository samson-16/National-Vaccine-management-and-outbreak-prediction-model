# Real Ethiopia Measles Updates Readiness Report

## What was built

The five Ethiopia measles update workbooks from 2021-2025 were parsed as real line-list/case-based surveillance records, de-identified, and aggregated to woreda-month labels. The resulting model matrix does not use the previous synthetic label overlay.

- Raw line-list records loaded: `87394`
- Clean records retained after date/location filtering: `87394`
- Observed monthly positive location rows: `9996`
- Complete real-only panel rows: `111660`
- Unique woredas/locations: `1861`
- COD-AB p-code matches: `465` of `1861`
- Total confirmed/compatible cases: `69700`
- Total deaths among confirmed/compatible cases: `506`
- Outbreak threshold: `5` confirmed/compatible cases per woreda-month

## Label definition

`target_cases` counts final-classification codes 1, 2, or 3: lab-confirmed, epidemiologically linked, or clinically compatible measles records. Discarded records are retained as QA/context counts but are not counted as target measles cases.

Months with no line-list record for a woreda are filled as zero-case panel months. They are marked in provenance as `real_line_list_zero_filled`, not synthetic model labels.

## Covariates

The matrix includes real public WHO national MCV1/MCV2 coverage, real public FEWS NET national IPC Phase 3+ monthly population estimates, woreda identity/location fields, lagged surveillance features, and existing deterministic proxy covariates for population/access/climate/conflict where direct woreda covariates are still unavailable.

## Modeling use

Use:

```powershell
python scripts\train_measles_mvp_model.py --input data\processed\measles_training_real_model_matrix.csv
```

This is now the preferred real-only model input for the project.
