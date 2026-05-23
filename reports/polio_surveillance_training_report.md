# Polio AFP Surveillance-Risk Training Report

## What was trained

This run trained derived-label AFP surveillance-risk alerting models, not a confirmed polio outbreak model.

- Input rows: `7045`
- Training/calibration rows: `3862`
- Validation rows: `1419`
- Test/demo rows: `1566`
- Latest alert rows: `198`

## Split Definition

- Training/calibration: prediction years `2021-2023`
- Validation: prediction year `2024`
- Testing/demo: prediction year `2025`

## Target Positive Rates

- `high_surveillance_risk`: train `0.007052186177715092`, validation `0.01048951048951049`, test `0.005780346820809248`
- `poor_stool_adequacy`: train `0.08321579689703808`, validation `0.08391608391608392`, test `0.06069364161849711`
- `delayed_reporting`: train `0.6459802538787024`, validation `0.6678321678321678`, test `0.6184971098265896`
- `under_vaccinated_afp`: train `0.1622002820874471`, validation `0.22727272727272727`, test `0.3352601156069364`
- `suspected_poliovirus`: train `0.03526093088857546`, validation `0.11538461538461539`, test `0.1416184971098266`

## 2025 Test Metrics

- `high_surveillance_risk`: F2 `0.000`, recall `0.000`, precision `0.000`, PR-AUC `0.008`
- `poor_stool_adequacy`: F2 `0.255`, recall `1.000`, precision `0.064`, PR-AUC `0.066`
- `delayed_reporting`: F2 `0.890`, recall `1.000`, precision `0.618`, PR-AUC `0.726`
- `under_vaccinated_afp`: F2 `0.716`, recall `1.000`, precision `0.335`, PR-AUC `0.322`
- `suspected_poliovirus`: F2 `0.418`, recall `0.694`, precision `0.161`, PR-AUC `0.178`

## Top Features

- `high_surveillance_risk` / `timely_lab_result_rate`: `0.206`
- `high_surveillance_risk` / `afp_surveillance_quality_score`: `0.125`
- `high_surveillance_risk` / `median_investigation_delay_days`: `0.121`
- `high_surveillance_risk` / `median_lab_result_delay_days`: `0.116`
- `high_surveillance_risk` / `median_lab_receipt_delay_days`: `0.063`
- `poor_stool_adequacy` / `timely_notification_rate`: `0.067`
- `poor_stool_adequacy` / `timely_lab_result_count`: `0.054`
- `poor_stool_adequacy` / `adequate_stool_rate`: `0.043`
- `poor_stool_adequacy` / `under_vaccinated_afp_cases`: `0.041`
- `poor_stool_adequacy` / `fup_done_rate`: `0.037`
- `delayed_reporting` / `final_classification_missing_count`: `0.045`
- `delayed_reporting` / `timely_lab_result_rate`: `0.042`
- `delayed_reporting` / `afp_cases`: `0.041`
- `delayed_reporting` / `median_lab_result_delay_days`: `0.039`
- `delayed_reporting` / `timely_investigation_rate`: `0.037`
- `under_vaccinated_afp` / `under_vaccinated_afp_share`: `0.077`
- `under_vaccinated_afp` / `under_vaccinated_afp_cases`: `0.077`
- `under_vaccinated_afp` / `under15_afp_cases`: `0.042`
- `under_vaccinated_afp` / `suspected_poliovirus_lab_result_count`: `0.035`
- `under_vaccinated_afp` / `timely_investigation_rate`: `0.035`
- `suspected_poliovirus` / `zero_dose_afp_cases`: `0.078`
- `suspected_poliovirus` / `timely_investigation_count`: `0.075`
- `suspected_poliovirus` / `afp_surveillance_quality_score`: `0.055`
- `suspected_poliovirus` / `month`: `0.049`
- `suspected_poliovirus` / `year`: `0.045`

## Latest Alerts

- `critical`: Aleltu, North Shewa (OR), Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: DEKSIS, Arsi, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Siraro, West Arsi, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Dire Dawa Operational Woreda, Dire Dawa, Dire Dawa for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Jeldu, West Shewa, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Moyalee, Borena, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Ejersa Lafo, West Shewa, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal; high next-month surveillance-risk probability
- `critical`: Adami Tulu Jido Kombolcha, East Shewa, Oromia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal
- `critical`: Boloso Bombe, Wolayita, South Ethiopia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal
- `critical`: Chorso, Gedeo, South Ethiopia for 2026-01-01 - suspected poliovirus surveillance signal; poor stool adequacy risk; delayed AFP notification or investigation risk; under-vaccinated AFP signal

## Important Limitation

These are AFP surveillance-risk and preparedness alerts. They should not be presented as confirmed polio outbreak probabilities unless official WPV/cVDPV positive labels are added.
