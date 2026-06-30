"""
Article 17 — Ghana Immunisation Equity
Script: 04_make_table1.py
Phase: 6 — Generate Table 1 (district characteristics, overall and by belt)

Produces a real artefact for the manuscript: outputs/tables/table1_district_characteristics.csv
Values trace to the verified final dataset (261 x 44).

Usage:
  python scripts/04_make_table1.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(PROJECT_ROOT, "data", "processed")
TABS = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(TABS, exist_ok=True)

NORTHERN_BELT = {"Northern", "North East", "Savannah", "Upper East", "Upper West"}

# (label, column, format) — mean(SD) for continuous; coverage shown as mean
ROWS = [
    ("Total population, median",            "total_pop",                 "median0"),
    ("Poverty incidence, % mean (SD)",      "poverty_incidence",         "msd1"),
    ("Population uninsured, % mean (SD)",    "census_uninsured_pct",      "msd1"),
    ("Illiterate, % mean (SD)",             "illiterate_pct",            "msd1"),
    ("Full immunisation coverage, % mean (SD)", "imm_fully_vaccinated_pct", "msd1"),
    ("Zero-dose (no vaccination), % mean (SD)", "imm_no_vaccination_pct",   "msd1"),
    ("DPT3 coverage, % mean (SD)",          "imm_dpt3_pct",              "msd1"),
    ("DPT1-DPT3 dropout, points mean (SD)", "dpt_dropout_rate",          "msd1"),
    ("Under-5 mortality, /1000 mean (SD)",  "cm_u5mr",                   "msd1"),
    ("Facility delivery, % mean (SD)",      "facility_delivery_pct",     "msd1"),
    ("Women, no education, % mean (SD)",    "women_no_education_pct",    "msd1"),
    ("IDRI (0-1), mean (SD)",               "idri",                      "msd2"),
]


def fmt(series, kind):
    if kind == "median0":
        return f"{series.median():,.0f}"
    if kind == "msd1":
        return f"{series.mean():.1f} ({series.std(ddof=1):.1f})"
    if kind == "msd2":
        return f"{series.mean():.2f} ({series.std(ddof=1):.2f})"
    return ""


def main():
    df = pd.read_csv(os.path.join(PROC, "master_immunisation_ghana_261_final.csv"))
    df["belt"] = np.where(df["Region"].isin(NORTHERN_BELT), "Northern", "Southern")

    groups = [("Overall (n=%d)" % len(df), df),
              ("Northern belt (n=%d)" % (df.belt == "Northern").sum(), df[df.belt == "Northern"]),
              ("Southern belt (n=%d)" % (df.belt == "Southern").sum(), df[df.belt == "Southern"])]

    out = {"Characteristic": [r[0] for r in ROWS]}
    for gname, gdf in groups:
        out[gname] = [fmt(gdf[col], kind) for _, col, kind in ROWS]

    # High-risk row (primary outcome, absolute <80%)
    out["Characteristic"].append("High-risk district (<80% fully vax), n (%)")
    for gname, gdf in groups:
        n = int(gdf["risk_index_binary"].sum())
        out[gname].append(f"{n} ({100*n/len(gdf):.0f}%)")

    t1 = pd.DataFrame(out)
    path = os.path.join(TABS, "table1_district_characteristics.csv")
    t1.to_csv(path, index=False, encoding="utf-8")
    print("✓ Table 1 written:", path)
    print(t1.to_string(index=False))


if __name__ == "__main__":
    main()
