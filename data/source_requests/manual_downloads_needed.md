# Manual Downloads Needed

The pipeline successfully downloaded and used:

- JHU top-state weekly measles data: `data/raw/external/jhu_top_states_time_series.csv`
- JHU county update measles data: `data/raw/external/jhu_measles_county_all_updates.csv`
- OWID / WHO GHO global reported measles cases: `data/raw/external/owid_reported_cases_of_measles.csv`

The sources below were not downloaded automatically. You can download them manually and place them at the suggested local paths.

## 1. WHO Immunization Data Portal Reported Cases Export

Link:

https://srhdpeuwpubsa-geecgzbpd5h0fueu.z01.azurefd.net/whdh/WIISE/export/reported-cases-data.xlsx

Suggested local path:

```text
data/raw/external/who_reported_cases_data.xlsx
```

Status:

The automatic download timed out because this is a large Excel export. The script already knows how to try parsing this file if it exists locally.

After downloading, run:

```powershell
python scripts/build_measles_mvp_dataset.py --include-external --include-who-export
```

## 2. HDX/OCHA Ethiopia COD-AB Administrative Boundaries

Link:

https://data.humdata.org/dataset/cod-ab-eth

Suggested local path:

```text
data/raw/boundaries/eth_admin3_woredas_cod_ab.zip
```

Status:

Not downloaded in this build. This is needed for official woreda p-codes, geometry, centroids, areas, and neighbor/spatial-lag features. The current `location_reference_woredas.csv` is built from EPHI workbook names only.

## 3. WorldPop Ethiopia Population / Age-Sex Rasters

Link:

https://hub.worldpop.org/project/categories?id=8

Suggested local folder:

```text
data/raw/worldpop/
```

Status:

Not downloaded in this build. The current dataset uses deterministic population and age-structure proxy fields. Downloading WorldPop rasters would allow real population, under-5, under-1, school-age, and density features.

## 4. FEWS NET Data Warehouse / Food Security Data

Link:

https://help.fews.net/fdw/fews-net-api

Suggested local folder:

```text
data/raw/conflict/fewsnet/
```

Status:

Not downloaded in this build. The current dataset uses a food-insecurity proxy. FEWS NET data would improve drought, food insecurity, market stress, and malnutrition-risk proxies.

## 5. WHO Ethiopia Disease Outbreak News Context

Link:

https://www.who.int/emergencies/disease-outbreak-news/item/2023-DON460

Suggested local path if manually extracted:

```text
data/raw/weak_labels/who_ethiopia_measles_outbreak_woredas_2023.csv
```

Status:

Not automatically parsed. This is useful for public weak labels and narrative validation, but it is not a clean national line-list.
