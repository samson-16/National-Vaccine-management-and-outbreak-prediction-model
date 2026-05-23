# Vaccine Data Readiness Report

## What was added

The project now has real public Ethiopia national annual measles vaccine coverage features.
These are joined to the woreda-month model by year and kept separate from woreda-level proxy coverage features.

- Annual vaccine rows: `25`
- Year range: `2000` to `2024`
- MCV1 load mode: `csv`
- MCV2 load mode: `csv`

## Latest Available Coverage Rows

- 2017: MCV1 `59.0`, MCV2 `0.0`
- 2018: MCV1 `54.0`, MCV2 `0.0`
- 2019: MCV1 `56.0`, MCV2 `35.0`
- 2020: MCV1 `57.0`, MCV2 `41.0`
- 2021: MCV1 `60.0`, MCV2 `41.0`
- 2022: MCV1 `62.0`, MCV2 `52.0`
- 2023: MCV1 `68.0`, MCV2 `53.0`
- 2024: MCV1 `72.0`, MCV2 `59.0`

## Request-Only Sources Not Used

The APHI EPI workbook and DHS microdata were not downloaded because they require request/access workflows.
This keeps the project consistent with the no-request constraint.

```json
[
  {
    "source_id": "aphi_epi_2018_2024",
    "source_name": "APHI Routine Immunization Coverage and Vaccine Utilization Dataset (EPI) from 2018-2024",
    "url": "https://rdmc.aphi.gov.et/items/a5807fcc-abd2-45dd-b5e7-e73e6667a802/full",
    "reason": "Request-copy/restricted access; not downloaded because project constraint excludes request-only sources."
  },
  {
    "source_id": "ethiopia_mini_dhs_2019_microdata",
    "source_name": "2019 Ethiopia Mini DHS microdata",
    "url": "https://microdata.worldbank.org/catalog/3946",
    "reason": "Requires account/access request; not downloaded because project constraint excludes request-only sources."
  }
]
```

## Modeling Use

Use `data/processed/measles_training_demo_model_matrix_with_vaccine.csv` for retraining.
The real national vaccine features should be used alongside existing woreda-level proxy fields.
