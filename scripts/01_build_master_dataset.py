"""
Article 17 — Ghana Immunisation Equity
Script: 01_build_master_dataset.py
Phase: 1 — Build and save the canonical Master CSV

Author: Valentine Golden Ghanem
Date: 2026-06-26
AIPOCH: v6.5

Output:
  ../data/processed/master_immunisation_ghana_261.csv   (261 × 37 at Phase 1, zero missing)
    37 = 35 base/derived + risk_index_binary (primary) + risk_index_binary_rel (sensitivity).
    5 spatial/ML columns (lisa_cluster, hotspot_flag, rf_risk_score, xgb_risk_score,
    shap_top_feature) are appended in Phase 4-5  ->  42 columns final.

Fail-Fast Gate (4-step):
  1. Syntax   — python -m py_compile scripts/01_build_master_dataset.py
  2. Logic    — verified denominators, dropout formula, outcome thresholds (primary + sensitivity)
  3. Epi      — district N = 261 confirmed; Guan present; regions == 16
  4. Linting  — PEP8 compliant

Epid Council rulings applied (2026-06-26):
  M3 — primary outcome = absolute (< 80% fully vaccinated; GVAP / national target).
       risk_index_binary_rel = relative (median / P75) sensitivity definition.
  M4 — imm_equity_index renamed imm_coverage_composite (mean 4-antigen coverage,
       NOT an inequality measure).
  M5 — illiterate_pct / employed_pct use a TOTAL-population denominator (declared; GSS
       literacy is conventionally defined for ages 11+/15+) — see README limitation.
  NOTE — DHS immunisation + health-system covariates are REGION-level (16 unique value-sets)
       assigned to districts for mapping only. Inference respects the ecological hierarchy
       (region-grouped CV; LISA on the IDRI surface, not on raw region-constant coverage).

Usage:
  python scripts/01_build_master_dataset.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# Windows-safe UTF-8 console (avoids cp1252 UnicodeEncodeError on arrows / check marks)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCE  = os.path.join(os.path.dirname(PROJECT_ROOT), "Research Datasets")
DATA_OUT     = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(DATA_OUT, exist_ok=True)

SEED = 42
ABS_TARGET_FV = 80.0   # GVAP / national fully-vaccinated coverage target (%) — primary threshold

# ─── 2022 DHS region mapping ──────────────────────────────────────────────────
KEEP_2022 = {
    "Ahafo", "Ashanti", "Bono", "Bono East", "Central", "Eastern",
    "Greater Accra", "Oti", "Upper East", "Upper West", "Western North",
    "Western (post 2022)", "Volta (post 2022)",
    "..Northern(post 2022)", "..Savannah", "..Northeast",
}

REGION_MAP = {
    "Western (post 2022)": "Western",
    "Volta (post 2022)":   "Volta",
    "..Northern(post 2022)": "Northern",
    "..Savannah":            "Savannah",
    "..Northeast":           "North East",
}

# ─── DHS indicator lists ──────────────────────────────────────────────────────
IMM_INDICATORS = [
    "BCG vaccination received",
    "DPT 1 vaccination received",
    "DPT 2 vaccination received",
    "DPT 3 vaccination received",
    "Polio 1 vaccination received",
    "Polio 2 vaccination received",
    "Polio 3 vaccination received",
    "Measles vaccination received",
    "Fully vaccinated (8 basic antigens)",
    "Received no vaccinations",
]
CM_INDICATORS  = ["Under-five mortality rate", "Infant mortality rate", "Neonatal mortality rate"]
DIA_INDICATORS = ["Children with diarrhea"]
INS_INDICATORS = ["No health insurance [Women]"]
ACC_INDICATORS = ["Place of delivery: Health facility"]
EDU_INDICATORS = ["Women with no education", "Women who are literate"]

RENAME_MAP = {
    "imm_BCG vaccination received":            "imm_bcg_pct",
    "imm_DPT 1 vaccination received":          "imm_dpt1_pct",
    "imm_DPT 2 vaccination received":          "imm_dpt2_pct",
    "imm_DPT 3 vaccination received":          "imm_dpt3_pct",
    "imm_Polio 1 vaccination received":        "imm_polio1_pct",
    "imm_Polio 2 vaccination received":        "imm_polio2_pct",
    "imm_Polio 3 vaccination received":        "imm_polio3_pct",
    "imm_Measles vaccination received":        "imm_measles_pct",
    "imm_Fully vaccinated (8 basic antigens)": "imm_fully_vaccinated_pct",
    "imm_Received no vaccinations":            "imm_no_vaccination_pct",
    "cm_Under-five mortality rate":            "cm_u5mr",
    "cm_Infant mortality rate":                "cm_imr",
    "cm_Neonatal mortality rate":              "cm_nmr",
    "dia_Children with diarrhea":              "diarrhea_prev_pct",
    "ins_No health insurance [Women]":         "dhs_no_insurance_women_pct",
    "acc_Place of delivery: Health facility":  "facility_delivery_pct",
    "edu_Women with no education":             "women_no_education_pct",
    "edu_Women who are literate":              "women_literate_pct",
}

# Phase-1 deterministic columns (37). Spatial/ML columns are appended in Phase 4-5 (-> 42).
FINAL_COLS_PHASE1 = [
    # Identifiers
    "district_id", "Region", "Class", "Latitude", "Longitude",
    # Population
    "total_pop", "pop_0_14", "pop_15_64", "pop_65plus", "child_pop_pct",
    # Census covariates (district-varying)
    "poverty_incidence", "poverty_intensity",
    "census_uninsured_pct", "illiterate_pct", "employed_pct",
    # DHS immunisation outcomes (region-level)
    "imm_bcg_pct", "imm_dpt1_pct", "imm_dpt2_pct", "imm_dpt3_pct",
    "imm_polio1_pct", "imm_polio2_pct", "imm_polio3_pct",
    "imm_measles_pct", "imm_fully_vaccinated_pct", "imm_no_vaccination_pct",
    # Derived immunisation
    "dpt_dropout_rate", "imm_coverage_composite",
    # Child health + structural determinants (region-level)
    "cm_u5mr", "cm_imr", "cm_nmr",
    "diarrhea_prev_pct", "dhs_no_insurance_women_pct",
    "facility_delivery_pct", "women_no_education_pct", "women_literate_pct",
    # Outcomes (binary)
    "risk_index_binary", "risk_index_binary_rel",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_dhs(path: str) -> pd.DataFrame:
    """Load DHS CSV, strip HXL metadata row, cast Value to float."""
    df = pd.read_csv(path, low_memory=False)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def load_dhs_pivot(path: str, indicators: list, prefix: str) -> pd.DataFrame:
    """Filter to 2022 survey, target indicators; return Region × Indicator pivot."""
    df = load_dhs(path)
    df = df[df["SurveyYear"] == "2022"].copy()
    df = df[df["Location"].isin(KEEP_2022)].copy()
    df["Region"] = df["Location"].map(REGION_MAP).fillna(df["Location"])
    df = df[df["Indicator"].isin(indicators)].copy()
    pivot = df.groupby(["Region", "Indicator"])["Value"].mean().unstack()
    pivot.columns = [f"{prefix}_{c}" for c in pivot.columns]
    return pivot


def load_master_sheet(path: str) -> pd.DataFrame:
    """Load GSS Census 2021 Master Sheet; derive per-capita columns.

    M5: illiterate_pct / employed_pct use TOTAL population as denominator (transparent;
    GSS literacy is conventionally defined for ages 11+/15+) — declared in README limitations.
    """
    ms = pd.read_excel(path)
    ms.rename(
        columns={"Metropolitan, Municipal, and District Assemblies (MMDA's)": "District"},
        inplace=True,
    )
    ms["child_pop_pct"]        = (ms["Male Population 0-14"] + ms["Female Population 0-14"]) / ms["Total Population"] * 100
    ms["census_uninsured_pct"] = ms["Uninsured Population"] / ms["Total Population"] * 100
    ms["illiterate_pct"]       = ms["Illiterate Population"] / ms["Total Population"] * 100   # M5: denom = total pop
    ms["employed_pct"]         = ms["Employed Population"] / ms["Total Population"] * 100      # M5: denom = total pop
    ms["pop_0_14"]             = ms["Male Population 0-14"] + ms["Female Population 0-14"]
    ms["pop_15_64"]            = ms["Male Population 15-64"] + ms["Female Population 15-64"]
    ms["pop_65plus"]           = ms["Male Population 65+"] + ms["Female Population 65+"]
    ms.rename(columns={
        "Total Population":   "total_pop",
        "Incidence of Poverty": "poverty_incidence",
        "Intensity of Poverty": "poverty_intensity",
    }, inplace=True)
    return ms


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Fail-fast validation of Master CSV before writing."""
    assert len(df) == 261,           f"FAIL: Expected 261 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, f"FAIL: {df.isnull().sum().sum()} missing values"
    assert "imm_fully_vaccinated_pct" in df.columns, "FAIL: primary outcome column missing"
    assert df["risk_index_binary"].isin([0, 1]).all(), "FAIL: risk_index_binary not binary"
    assert df["risk_index_binary_rel"].isin([0, 1]).all(), "FAIL: risk_index_binary_rel not binary"
    assert np.allclose(df["dpt_dropout_rate"], df["imm_dpt1_pct"] - df["imm_dpt3_pct"]), "FAIL: dropout formula"
    assert df["imm_coverage_composite"].between(0, 1).all(), "FAIL: coverage composite out of [0,1]"
    assert df["Region"].nunique() == 16, f"FAIL: Expected 16 regions, got {df['Region'].nunique()}"
    assert df["district_id"].str.contains("Guan", case=False).any(), "FAIL: Guan District missing"
    assert list(df.columns) == FINAL_COLS_PHASE1, "FAIL: column order/set does not match schema"
    print("✓ All validation checks passed.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ARTICLE 17 - BUILDING MASTER DATASET (Phase 1)")
    print("=" * 70)

    # 1. Load Master Sheet
    ms = load_master_sheet(os.path.join(DATA_SOURCE, "Master Sheet.xlsx"))

    # 2. Load DHS pivots (region-level)
    df_imm = load_dhs_pivot(os.path.join(DATA_SOURCE, "immunization_subnational_gha.csv"), IMM_INDICATORS, "imm")
    df_cm  = load_dhs_pivot(os.path.join(DATA_SOURCE, "child-mortality-rates_subnational_gha.csv"), CM_INDICATORS, "cm")
    df_dia = load_dhs_pivot(os.path.join(DATA_SOURCE, "diarrhea_subnational_gha.csv"), DIA_INDICATORS, "dia")
    df_ins = load_dhs_pivot(os.path.join(DATA_SOURCE, "health-insurance_subnational_gha.csv"), INS_INDICATORS, "ins")
    df_acc = load_dhs_pivot(os.path.join(DATA_SOURCE, "access-to-health-care_subnational_gha.csv"), ACC_INDICATORS, "acc")
    df_edu = load_dhs_pivot(os.path.join(DATA_SOURCE, "select-education-indicators_subnational_gha.csv"), EDU_INDICATORS, "edu")

    region_df = df_imm.join([df_cm, df_dia, df_ins, df_acc, df_edu], how="outer").reset_index()

    # 3. Merge (region-level DHS assigned to districts for mapping; ecological — see docstring)
    df = ms.merge(region_df, on="Region", how="left")
    df.rename(columns=RENAME_MAP, inplace=True)

    # 4. Derived columns
    df["dpt_dropout_rate"]       = df["imm_dpt1_pct"] - df["imm_dpt3_pct"]
    df["imm_coverage_composite"] = (
        df["imm_bcg_pct"] + df["imm_dpt3_pct"] +
        df["imm_polio3_pct"] + df["imm_measles_pct"]
    ) / 400   # M4: mean 4-antigen coverage (0-1), not an inequality measure

    # 5. Binary outcomes
    #    PRIMARY (M3): absolute — district fully-vaccinated below national/GVAP target (< 80%).
    df["risk_index_binary"] = (df["imm_fully_vaccinated_pct"] < ABS_TARGET_FV).astype(int)
    #    SENSITIVITY: relative — below sample median fully-vax OR above P75 no-vaccination.
    MEDIAN_FV = df["imm_fully_vaccinated_pct"].median()
    P75_NV    = df["imm_no_vaccination_pct"].quantile(0.75)
    df["risk_index_binary_rel"] = (
        (df["imm_fully_vaccinated_pct"] < MEDIAN_FV) |
        (df["imm_no_vaccination_pct"]   > P75_NV)
    ).astype(int)

    # 6. Rename District -> district_id for schema consistency
    df.rename(columns={"District": "district_id"}, inplace=True)

    # 7. Select and order final columns (Phase 1; spatial/ML cols appended Phase 4-5)
    missing = [c for c in FINAL_COLS_PHASE1 if c not in df.columns]
    assert not missing, f"FAIL: expected columns absent after build: {missing}"
    df = df[FINAL_COLS_PHASE1].copy()

    # 8. Validate
    print(f"\nDataset shape before validation: {df.shape}")
    validate(df)

    # 9. Save
    out_path = os.path.join(DATA_OUT, "master_immunisation_ghana_261.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✓ Master CSV saved: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  PRIMARY outcome (absolute, fully_vax < {ABS_TARGET_FV:.0f}%): "
          f"{df['risk_index_binary'].sum()} / 261 high-risk "
          f"({df['risk_index_binary'].mean()*100:.1f}%)")
    print(f"  SENSITIVITY outcome (relative, fully_vax < {MEDIAN_FV:.1f}% OR no_vax > {P75_NV:.2f}%): "
          f"{df['risk_index_binary_rel'].sum()} / 261 high-risk "
          f"({df['risk_index_binary_rel'].mean()*100:.1f}%)")
    print("\n  NOTE: 5 spatial/ML columns (lisa_cluster, hotspot_flag, rf_risk_score,")
    print("        xgb_risk_score, shap_top_feature) are appended in Phase 4-5 -> 42 cols final.")

    # 10. Syntax self-check
    import py_compile
    py_compile.compile(__file__, doraise=True)
    print("\n✓ Script syntax verified (py_compile passed).")


if __name__ == "__main__":
    main()
