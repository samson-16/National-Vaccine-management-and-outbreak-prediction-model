# Raw Data Folder

Place source files here without editing their contents. Rename files to the expected local names in `config/data_catalog.json` so the inventory checker and prep scripts can find them.

Recommended first files:

```text
data/raw/aphi/measles_line_list_amhara_2022_2025.xlsx
data/raw/aphi/measles_cases_deaths_amhara_2024_2025.xlsx
data/raw/boundaries/eth_admin3_woredas_cod_ab.zip
data/raw/aphi/routine_immunization_epi_amhara_2018_2024.xlsx
```

Do not store fabricated labels here. If a file is manually extracted from WHO or EPHI notices, save it under `data/raw/weak_labels/` and document exactly how it was extracted.
