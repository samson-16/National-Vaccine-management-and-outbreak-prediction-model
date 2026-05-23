# Polio AFP Surveillance Risk Readiness Report

## What was built

The Ethiopia AFP weekly analysis workbooks from 2021-2025 were parsed from the `Linelist` sheet, de-identified, and aggregated to woreda-month surveillance features.

- Raw AFP records loaded: `8510`
- Clean AFP records retained after date/location filtering: `8510`
- Observed woreda-month rows: `7045`
- Unique AFP woredas/locations: `1639`
- COD-AB p-code matches: `342` of `1639`
- Under-15 AFP records: `8371`
- Under-vaccinated AFP records: `1631`
- Adequate stool records: `7833`
- Suspected poliovirus lab-result records: `445`
- Explicit positive/confirmed polio lab-result records: `0`

## Lab result distribution

- `2-Negative`: `7124`
- `3-NPENT`: `724`
- `1-Suspected Poliovirus`: `445`
- `missing`: `217`

## Final classification distribution

- `3`: `8038`
- `missing`: `360`
- `7`: `96`
- `2`: `15`
- `8`: `1`

## How to use this module

Use the output as a `Polio AFP Surveillance Risk Module`, not as a confirmed outbreak prediction model unless official positive labels are added.

```powershell
python scripts\prepare_polio_afp_surveillance.py
```

Key outputs:

- `data/processed/polio_afp_line_list_deidentified.csv`
- `data/processed/polio_afp_woreda_month_features.csv`
- `model_outputs_polio_afp/polio_afp_latest_surveillance_risk.csv`
- `model_outputs_polio_afp/polio_afp_top_risk_woredas.csv`

## Defense wording

For polio, because paralytic cases are rare and confirmed outbreak labels are limited, the system uses AFP surveillance indicators and vaccination status to identify areas that may need strengthened surveillance, immunization outreach, or preparedness.

## Important limitation

Rows labelled `1-Suspected Poliovirus` are retained as a surveillance signal, not as confirmed polio outbreak labels. Numeric final-classification codes are counted transparently, but the script does not reinterpret them as confirmed polio without an official codebook or explicit WPV/cVDPV/positive lab confirmation.
