# Data Dictionary — `master_immunisation_ghana_261_final.csv` (261 × 44)

Unit: **district** (261). Immunisation/health-system fields are **region-level** (16 regions) assigned to
districts (ecological design — see `docs/phase3_study_design.md`). Built by `scripts/01` → `02` → `03`.

| Column | Type | Source | Description |
|---|---|---|---|
| `district_id` | str | GSS 2021 | District (MMDA) name |
| `Region` | str | GSS 2021 | One of 16 post-2019 regions |
| `Class` | str | GSS 2021 | Metropolitan / Municipal / District |
| `Latitude`,`Longitude` | float | GSS 2021 | District centroid |
| `total_pop` | int | GSS 2021 | Total population |
| `pop_0_14`,`pop_15_64`,`pop_65plus` | int | GSS 2021 | Age-band population |
| `child_pop_pct` | float | derived | Children 0–14 as % of total |
| `poverty_incidence`,`poverty_intensity` | float | GSS 2021 | Poverty incidence / intensity (%) |
| `census_uninsured_pct` | float | GSS 2021 | Population uninsured (%) |
| `illiterate_pct`,`employed_pct` | float | GSS 2021 | % of **total** population (M5 denominator caveat) |
| `imm_bcg_pct` … `imm_measles_pct` | float | DHS 2022 (region) | Antigen coverage (BCG, DPT1–3, Polio1–3, Measles) |
| `imm_fully_vaccinated_pct` | float | DHS 2022 (region) | **Continuous outcome** — fully vaccinated (8 antigens) |
| `imm_no_vaccination_pct` | float | DHS 2022 (region) | Zero-dose proxy |
| `dpt_dropout_rate` | float | derived | `imm_dpt1_pct − imm_dpt3_pct` (equity signal) |
| `imm_coverage_composite` | float | derived | Mean 4-antigen coverage 0–1 (**coverage, not inequality**) |
| `cm_u5mr`,`cm_imr`,`cm_nmr` | float | DHS 2022 (region) | Under-5 / infant / neonatal mortality (/1000) |
| `diarrhea_prev_pct` | float | DHS 2022 (region) | Diarrhoea prevalence |
| `dhs_no_insurance_women_pct` | float | DHS 2022 (region) | Women without health insurance |
| `facility_delivery_pct` | float | DHS 2022 (region) | Births at a health facility |
| `women_no_education_pct`,`women_literate_pct` | float | DHS 2022 (region) | Women's education |
| `risk_index_binary` | 0/1 | derived (M3) | **PRIMARY outcome** — 1 if fully-vax < 80% (absolute) |
| `risk_index_binary_rel` | 0/1 | derived (M3) | Sensitivity — 1 if < median OR no-vax > P75 |
| `idri` | float | derived (Phase 4) | Immunisation District Risk Index 0–1 (district-varying vulnerability) |
| `lisa_cluster` | str | Phase 4 | LISA cluster on IDRI: HH/LL/HL/LH/NS; `NA (no polygon)` for Guan |
| `gi_z` | float | Phase 4 | Getis-Ord Gi* z-score (NaN for Guan — no polygon) |
| `hotspot_flag` | int | Phase 4 | 1 hotspot / −1 coldspot / 0 not significant |
| `rf_risk_score`,`xgb_risk_score` | float | Phase 5 | Out-of-fold (LORO-CV) predicted risk probability |
| `shap_top_feature` | str | Phase 5 | Top SHAP determinant for the district (XGBoost, descriptive) |

**District crosswalk:** `data/geospatial/crosswalk_260.csv` links 261 Census districts ↔ 260 GeoJSON
polygons (Guan is the sole structural gap). See `data/geospatial/crosswalk_notes.md`.
