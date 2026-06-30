"""
Article 17 — Ghana Immunisation Equity
Script: 00_data_profiling.py
Phase: 1 — Data profiling, QC, Table 1, dataset integration strategy

Author: Valentine Golden Ghanem
Date: 2026-06-26
AIPOCH: v6.5

Inputs (from ../Research Datasets/):
  - Master Sheet.xlsx                                            (GSS Census 2021; 261 districts)
  - immunization_subnational_gha.csv                            (DHS StatCompiler; 2022; region-level)
  - child-mortality-rates_subnational_gha.csv                   (DHS; 2022; region-level)
  - diarrhea_subnational_gha.csv                                (DHS; 2022; region-level)
  - health-insurance_subnational_gha.csv                        (DHS; 2022; region-level)
  - access-to-health-care_subnational_gha.csv                   (DHS; 2022; region-level)
  - select-education-indicators_subnational_gha.csv             (DHS; 2022; region-level)
  - immunization_coverage_and_vaccine_preventable_diseases...   (WHO GHO; national; 1974-2024)
  - health_workforce_indicators_gha.csv                         (WHO GHO; national)
  - Ghana_New_260_District.geojson                              (District boundaries; 260 polygons)

Outputs:
  - Console: profiling report, Table 1, equity gap table
  - The 261<->260 district crosswalk is built separately by scripts/00b_build_crosswalk.py
    (-> ../data/geospatial/crosswalk_260.csv). This script does NOT write the crosswalk.

Usage:
  python scripts/00_data_profiling.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import json

# Windows-safe UTF-8 console (avoids cp1252 UnicodeEncodeError on arrows / check marks)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCE = os.path.join(os.path.dirname(PROJECT_ROOT), "Research Datasets")
GEO_OUT = os.path.join(PROJECT_ROOT, "data", "geospatial")
os.makedirs(GEO_OUT, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SEED = 42

# 2022 DHS 16-region names (post-2019 reorganisation)
KEEP_2022 = {
    "Ahafo", "Ashanti", "Bono", "Bono East", "Central", "Eastern",
    "Greater Accra", "Oti", "Upper East", "Upper West", "Western North",
    "Western (post 2022)", "Volta (post 2022)",
    "..Northern(post 2022)", "..Savannah", "..Northeast",
}

# Standardise to clean region names for merge with Master Sheet
REGION_MAP = {
    "Western (post 2022)": "Western",
    "Volta (post 2022)": "Volta",
    "..Northern(post 2022)": "Northern",
    "..Savannah": "Savannah",
    "..Northeast": "North East",
}

# Northern belt regions (for equity gap analysis)
NORTHERN_BELT = {"Northern", "North East", "Savannah", "Upper East", "Upper West"}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_dhs(path: str) -> pd.DataFrame:
    """Load a DHS StatCompiler CSV, stripping the HXL metadata row (row with # prefix)."""
    df = pd.read_csv(path, low_memory=False)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def filter_2022(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DHS data to 2022 survey wave and 16 post-2019 regions, then standardise names."""
    df2 = df[df["SurveyYear"] == "2022"].copy()
    df2 = df2[df2["Location"].isin(KEEP_2022)].copy()
    df2["Region"] = df2["Location"].map(REGION_MAP).fillna(df2["Location"])
    return df2


def load_dhs_pivot(path: str, indicators: list, prefix: str) -> pd.DataFrame:
    """Load DHS file, filter to 2022 + target indicators, pivot to Region × Indicator."""
    df = load_dhs(path)
    df = filter_2022(df)
    df = df[df["Indicator"].isin(indicators)].copy()
    pivot = df.groupby(["Region", "Indicator"])["Value"].mean().unstack()
    pivot.columns = [f"{prefix}_{c}" for c in pivot.columns]
    return pivot


def load_master_sheet(path: str) -> pd.DataFrame:
    """Load GSS Census 2021 Master Sheet and compute derived columns."""
    ms = pd.read_excel(path)
    ms.rename(
        columns={"Metropolitan, Municipal, and District Assemblies (MMDA's)": "District"},
        inplace=True,
    )
    ms["child_pop_pct"] = (
        (ms["Male Population 0-14"] + ms["Female Population 0-14"])
        / ms["Total Population"] * 100
    )
    ms["census_uninsured_pct"] = ms["Uninsured Population"] / ms["Total Population"] * 100
    ms["illiterate_pct"] = ms["Illiterate Population"] / ms["Total Population"] * 100
    ms["employed_pct"] = ms["Employed Population"] / ms["Total Population"] * 100
    ms["pop_0_14"] = ms["Male Population 0-14"] + ms["Female Population 0-14"]
    ms["pop_15_64"] = ms["Male Population 15-64"] + ms["Female Population 15-64"]
    ms["pop_65plus"] = ms["Male Population 65+"] + ms["Female Population 65+"]
    ms.rename(
        columns={
            "Total Population": "total_pop",
            "Incidence of Poverty": "poverty_incidence",
            "Intensity of Poverty": "poverty_intensity",
        },
        inplace=True,
    )
    return ms


# ─── DHS Indicator Lists ──────────────────────────────────────────────────────

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

CM_INDICATORS = [
    "Under-five mortality rate",
    "Infant mortality rate",
    "Neonatal mortality rate",
]

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ARTICLE 17 — DATA PROFILING (Phase 1)")
    print("=" * 70)

    # --- Load Master Sheet ---
    ms_path = os.path.join(DATA_SOURCE, "Master Sheet.xlsx")
    ms = load_master_sheet(ms_path)
    print(f"\n[1] Master Sheet (GSS Census 2021): {ms.shape} | Missing: {ms.isnull().sum().sum()}")
    print(f"    Districts: {len(ms)} | Regions: {ms['Region'].nunique()}")
    guan = ms[ms["District"].str.contains("Guan", case=False, na=False)]
    print(f"    Guan District present: {'YES' if len(guan) > 0 else 'NO'} (261st district, Oti, pop {guan['total_pop'].values[0] if len(guan) > 0 else 'N/A':,})")

    # --- Load DHS files ---
    df_imm = load_dhs_pivot(os.path.join(DATA_SOURCE, "immunization_subnational_gha.csv"), IMM_INDICATORS, "imm")
    df_cm  = load_dhs_pivot(os.path.join(DATA_SOURCE, "child-mortality-rates_subnational_gha.csv"), CM_INDICATORS, "cm")
    df_dia = load_dhs_pivot(os.path.join(DATA_SOURCE, "diarrhea_subnational_gha.csv"), DIA_INDICATORS, "dia")
    df_ins = load_dhs_pivot(os.path.join(DATA_SOURCE, "health-insurance_subnational_gha.csv"), INS_INDICATORS, "ins")
    df_acc = load_dhs_pivot(os.path.join(DATA_SOURCE, "access-to-health-care_subnational_gha.csv"), ACC_INDICATORS, "acc")
    df_edu = load_dhs_pivot(os.path.join(DATA_SOURCE, "select-education-indicators_subnational_gha.csv"), EDU_INDICATORS, "edu")

    region_df = df_imm.join([df_cm, df_dia, df_ins, df_acc, df_edu], how="outer").reset_index()
    print(f"\n[2] DHS 2022 region pivot: {region_df.shape} | Regions: {len(region_df)}")

    # --- Merge ---
    full_df = ms.merge(region_df, on="Region", how="left")
    full_df.rename(columns=RENAME_MAP, inplace=True)

    # --- Derived indicators ---
    full_df["dpt_dropout_rate"] = full_df["imm_dpt1_pct"] - full_df["imm_dpt3_pct"]
    full_df["imm_coverage_composite"] = (   # M4: mean 4-antigen coverage (0-1), not an inequality measure
        full_df["imm_bcg_pct"] + full_df["imm_dpt3_pct"] +
        full_df["imm_polio3_pct"] + full_df["imm_measles_pct"]
    ) / 400

    # Binary outcomes (M3): PRIMARY absolute (< 80% fully vaccinated); SENSITIVITY relative.
    ABS_TARGET_FV = 80.0
    full_df["risk_index_binary"] = (full_df["imm_fully_vaccinated_pct"] < ABS_TARGET_FV).astype(int)
    median_fv = full_df["imm_fully_vaccinated_pct"].median()
    p75_nv    = full_df["imm_no_vaccination_pct"].quantile(0.75)
    full_df["risk_index_binary_rel"] = (
        (full_df["imm_fully_vaccinated_pct"] < median_fv) |
        (full_df["imm_no_vaccination_pct"] > p75_nv)
    ).astype(int)

    print(f"\n[3] Merged dataset: {full_df.shape} | Missing: {full_df.isnull().sum().sum()}")
    print(f"    PRIMARY (absolute, fully_vax < {ABS_TARGET_FV:.0f}%): "
          f"{full_df['risk_index_binary'].sum()} / 261 ({full_df['risk_index_binary'].mean()*100:.1f}%) high-risk")
    print(f"    SENSITIVITY (relative, fully_vax < {median_fv:.1f}% OR no_vax > {p75_nv:.2f}%): "
          f"{full_df['risk_index_binary_rel'].sum()} / 261 ({full_df['risk_index_binary_rel'].mean()*100:.1f}%) high-risk")

    # --- Table 1 summary ---
    print("\n" + "=" * 70)
    print("TABLE 1 — DESCRIPTIVE STATISTICS (261 districts)")
    print("=" * 70)
    numeric_cols = [
        "total_pop", "child_pop_pct", "poverty_incidence", "poverty_intensity",
        "census_uninsured_pct", "illiterate_pct", "employed_pct",
        "imm_bcg_pct", "imm_dpt1_pct", "imm_dpt3_pct", "imm_measles_pct",
        "imm_fully_vaccinated_pct", "imm_no_vaccination_pct",
        "dpt_dropout_rate", "imm_coverage_composite",
        "cm_u5mr", "cm_imr", "cm_nmr",
        "diarrhea_prev_pct", "dhs_no_insurance_women_pct",
        "facility_delivery_pct", "women_no_education_pct",
    ]
    desc = full_df[numeric_cols].describe(percentiles=[0.25, 0.50, 0.75]).T[
        ["mean", "std", "50%", "25%", "75%", "min", "max"]
    ].round(2)
    desc.columns = ["Mean", "SD", "Median", "Q1", "Q3", "Min", "Max"]
    print(desc.to_string())

    # --- Equity gap ---
    full_df["zone"] = full_df["Region"].apply(
        lambda r: "Northern belt" if r in NORTHERN_BELT else "Southern belt"
    )
    print("\n" + "=" * 70)
    print("EQUITY GAP — NORTHERN vs SOUTHERN BELT")
    print("=" * 70)
    gap = full_df.groupby("zone").agg(
        districts=("District", "count"),
        poverty=("poverty_incidence", "mean"),
        fully_vax=("imm_fully_vaccinated_pct", "mean"),
        no_vax=("imm_no_vaccination_pct", "mean"),
        dpt3=("imm_dpt3_pct", "mean"),
        dpt_dropout=("dpt_dropout_rate", "mean"),
        U5MR=("cm_u5mr", "mean"),
        facility_del=("facility_delivery_pct", "mean"),
        no_edu=("women_no_education_pct", "mean"),
        high_risk_pct=("risk_index_binary", lambda x: x.mean() * 100),
    ).round(2)
    print(gap.T.to_string())

    # --- DPT dropout by region ---
    print("\n" + "=" * 70)
    print("DPT1→DPT3 DROPOUT BY REGION (equity signal)")
    print("=" * 70)
    dropout_reg = (
        full_df.groupby("Region")["dpt_dropout_rate"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    print(dropout_reg.to_string())

    # --- Missing data ---
    print("\n" + "=" * 70)
    print("MISSING DATA CHECK")
    print("=" * 70)
    missing = full_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("✓ Zero missing values in merged dataset.")
    else:
        print(missing.to_string())

    print("\n✓ Phase 1 profiling complete.")
    print(f"  Full dataset shape: {full_df.shape}")
    print(f"  Run scripts/01_build_master_dataset.py to save Master CSV.")


if __name__ == "__main__":
    main()
