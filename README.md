# Subnational immunisation coverage gaps and structural determinants in Ghana

**Ecological equity mapping and an interpretable district risk index across 261 districts**

[![CI](https://github.com/valentineghanem-bit/ghana-immunisation-coverage-261-districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/ghana-immunisation-coverage-261-districts/actions/workflows/ci.yml)
&nbsp;Reporting: STROBE + RECORD &nbsp;·&nbsp; License: MIT

---

## 1. Overview

This repository contains the full, reproducible analysis behind an ecological study of childhood
immunisation coverage across **all 261 districts of Ghana**. Using the 2022 Ghana Demographic and Health
Survey (DHS; region-level immunisation, 16 regions) linked to the 2021 Ghana Population and Housing Census
(district-level socioeconomics, 261 districts), it maps coverage, builds an **Immunisation District Risk
Index (IDRI)**, characterises its spatial structure, and tests whether machine learning improves district
risk classification over logistic regression. The design is explicitly **ecological** and methodologically
candid: immunisation coverage is region-resolved (effective N = 16), so it is mapped at region level while
spatial and predictive analyses use a district-varying socioeconomic surface.

**Author:** Valentine Golden Ghanem · Ghana COCOBOD Cocoa Clinic, Accra; University of Cape Coast (PhD
candidate) · ORCID 0009-0002-8332-0220.

## 2. Key findings

- Full-immunisation coverage averaged **75.2%** (median 77.8%); **180 / 261 districts (69.0%)** fell below the 80% target.
- A marked **Northern–Southern gradient**: Northern-belt districts (n=55) averaged 67.5% full coverage
  versus 77.2% in the Southern belt (n=206), with higher zero-dose prevalence (3.8% vs 1.1%) and DPT1–DPT3
  dropout (10.2 vs 7.7 points). Coverage ranged from 55.8% (Northern) to 87.7% (Western North); Western
  Region (62.5%) was a southern low-coverage exception.
- District vulnerability (IDRI) was strongly spatially clustered (**Global Moran's I = 0.72, z = 17.5,
  p < 0.001**), with 51 high–high LISA districts and 55 Getis–Ord hotspots forming a contiguous northern block.
- Under leave-one-region-out validation, **logistic regression (AUC 0.82, Brier 0.16) outperformed random
  forest (AUC 0.54) and XGBoost (AUC 0.46)**, which failed to generalise — machine learning added no benefit
  over logistic regression in this ecological setting.
- Facility-delivery coverage and women's lack of health insurance were the leading correlates of district risk.

## 3. Study design

Cross-sectional, ecological (small-area) study; **261 districts of Ghana**. Reporting follows STROBE
(observational studies) and RECORD (routinely-collected health data). The primary binary outcome classes a
district high-risk when full immunisation coverage (eight basic antigens) falls below the 80% national/GVAP
target; a relative definition is a sensitivity analysis. Ecological fallacy and the modifiable areal unit
problem (MAUP) are declared limitations.

## 4. Data sources

| Source | Unit | Used for |
|---|---|---|
| 2022 Ghana DHS (StatCompiler) | Region (16) | Immunisation coverage, child mortality, diarrhoea, facility delivery, women's education/insurance |
| 2021 Ghana Population & Housing Census (GSS) | District (261) | Poverty, uninsured %, illiteracy, employment, age structure |
| WHO Global Health Observatory | Country | National EPI/VPD/mortality/workforce trends (Introduction context only) |
| 2017 district boundaries | 260 polygons | Choropleth + spatial weights (Guan District: sole structural gap) |

Raw datasets are not redistributed here; the analysis reads them from a sibling `Research Datasets/` folder.
The de-identified linked analytical dataset is included under `data/processed/`.

## 5. Repository structure

```
.
├── data/
│   ├── processed/   master_immunisation_ghana_261_final.csv (261×44) + intermediate CSVs
│   ├── geospatial/  crosswalk_260.csv (built by 00b) + crosswalk_notes.md
│   └── data_dictionary.md
├── scripts/         00 profiling · 00b crosswalk · 01 master · 02 spatial · 03 ML · 04 table1 · 05 national-context
├── outputs/         figures/ · maps/ · tables/
├── dashboard/       index.html (HI-EI interactive dashboard) + ghana_districts_compact.geojson
├── poster/          poster.html (A0 conference poster)
├── docs/            phase0_phase1 · phase3_study_design · phase4_spatial · phase5_ml
├── evidence/        phase2_evidence_bank.md (24 sources, stratified + contrasting audit)
├── tests/           test_master_csv.py (11 tests)
├── qa/              QA_PASSED_2026-06-26.txt
├── .github/workflows/ci.yml
├── CITATION.cff · LICENSE · README.md · requirements.txt · .gitignore
```

## 6. Methods / pipeline

1. **Crosswalk** (`00b_build_crosswalk.py`) — 261 Census districts ↔ 260 polygons (232 exact + 28 vetted;
   Guan the sole gap), hard-validated.
2. **Master dataset** (`01_build_master_dataset.py`) — region-level DHS assigned to districts (ecological);
   derived indices; primary (<80%) + sensitivity outcomes.
3. **Spatial** (`02_spatial_analysis.py`) — IDRI from six district-varying determinants; queen contiguity;
   Global Moran's I, LISA, Getis–Ord Gi* (9,999 permutations) on the IDRI surface.
4. **Machine learning** (`03_ml_pipeline.py`) — logistic regression, random forest, XGBoost on
   **determinant-only features** (leakage-guarded); **leave-one-region-out** cross-validation; AUC,
   sensitivity, specificity, Brier; SHAP (exploratory).
5. **Tables + national context** (`04_make_table1.py`, `05_national_context.py`).

## 7. Reproducibility

```bash
pip install -r requirements.txt          # CI-installable (Python 3.11–3.13)
python scripts/00b_build_crosswalk.py
python scripts/01_build_master_dataset.py
python scripts/02_spatial_analysis.py    # needs geopandas/esda/libpysal
python scripts/03_ml_pipeline.py
pytest tests/ -v                          # 11 tests
```

Random seed fixed at `SEED = 42`. Software: Python 3.13, pandas, GeoPandas 1.1, libpysal 4.14, esda 2.8,
scikit-learn 1.8, XGBoost 3.3, SHAP 0.52. Continuous integration (`.github/workflows/ci.yml`) installs
dependencies, compiles all scripts, and runs the test suite on Python 3.11 and 3.12.

## 8. Outputs

- `data/processed/master_immunisation_ghana_261_final.csv` — analytical dataset (261 × 44).
- `outputs/maps/` — region coverage choropleth, LISA cluster map, IDRI choropleth, Getis–Ord hotspots.
- `outputs/figures/` — leave-one-region-out ROC, SHAP summary, Moran scatter, national-context trends.
- `outputs/tables/` — Table 1 (district characteristics), ML performance, spatial autocorrelation, national context.

## 9. Dashboard & poster — view or download

- **Interactive dashboard:** [`dashboard/index.html`](dashboard/index.html) — HI-EI dashboard (region
  coverage choropleth, ranking, coverage–poverty scatter, LISA, KPIs). Self-contained; open in a browser.
- **Conference poster:** [`poster/poster.html`](poster/poster.html) — A0 (841 × 1189 mm).

> Built with the bespoke HI-EI vanilla-JS + inline-SVG/ECharts pipeline (supersedes the legacy 60 KB ceiling).

## 10. Data dictionary

See [`data/data_dictionary.md`](data/data_dictionary.md) — definitions, source and type for all 44 columns
of the final dataset, including the ecological-unit note (region-level vs district-level fields).

## 11. Analytical verification

- `pytest tests/` — 11 checks on the committed dataset (261 rows, expected missingness only in `gi_z` for
  Guan, binary outcomes, dropout formula, IDRI ∈ [0,1], risk scores ∈ [0,1], 180/261 high-risk).
- QA badge: [`qa/QA_PASSED_2026-06-26.txt`](qa/QA_PASSED_2026-06-26.txt).
- All headline numbers (180/261, AUC 0.82, Moran's I 0.72) are reconciled across dataset, dashboard and poster.

## 12. Citation

See [`CITATION.cff`](CITATION.cff). Ghanem VG. *Subnational immunisation coverage gaps and structural
determinants in Ghana: ecological equity mapping and an interpretable district risk index across 261
districts.* 2026. ORCID 0009-0002-8332-0220. **Target journal:** *Vaccine* (Q1).

## 13. License & ethics

MIT License (see [`LICENSE`](LICENSE)); source datasets retain their original terms. Secondary analysis of
de-identified, publicly available aggregate data; a consent waiver applies (Ghana Health Service Ethics
Review Committee / University of Cape Coast IRB notification).

## 14. Acknowledgements & contact

Ghana Statistical Service and the DHS Program for open data; WHO Global Health Observatory.
Contact: Valentine Golden Ghanem — valentineghanem@gmail.com · ORCID 0009-0002-8332-0220.
