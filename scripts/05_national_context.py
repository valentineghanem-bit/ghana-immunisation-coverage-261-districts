"""
Article 17 — Ghana Immunisation Equity
Script: 05_national_context.py
Phase: 6 — National-context framing from WHO GHO national files (Introduction/Discussion only)

These WHO GHO files are COUNTRY-level (Ghana; no district resolution) and therefore CANNOT enter the
261-district model. They are used solely for national-context framing of VPD burden / EPI trajectory.

INPUTS (../../Research Datasets/)
  - vaccine_preventable_communicable_diseases_indicators_gha.csv   (national EPI coverage + VPD cases, 1974-2024)
  - child_mortality_indicators_gha.csv                             (national U5MR trend)
  - health_workforce_indicators_gha.csv                            (national workforce density, 2004-2023)

OUTPUTS
  - outputs/tables/national_context.csv
  - outputs/figures/national_context_trends.png|.svg  (Panel A: EPI coverage; Panel B: U5MR decline)
  - console: key national numbers for the manuscript

Usage:
  python scripts/05_national_context.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCE  = os.path.join(os.path.dirname(PROJECT_ROOT), "Research Datasets")
FIGS = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABS = os.path.join(PROJECT_ROOT, "outputs", "tables")
for d in (FIGS, TABS):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito colourblind-safe
CB = {"dpt3": "#0072B2", "mcv1": "#D55E00", "pol3": "#009E73", "u5mr": "#CC79A7"}


def load(f):
    df = pd.read_csv(os.path.join(DATA_SOURCE, f), low_memory=False)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].copy()
    df["year"] = pd.to_numeric(df["STARTYEAR"], errors="coerce")
    df["val"] = pd.to_numeric(df["Numeric"], errors="coerce")
    return df


def series(df, keyword, exclude=None):
    m = df["GHO (DISPLAY)"].str.contains(keyword, case=False, na=False)
    if exclude:
        m &= ~df["GHO (DISPLAY)"].str.contains(exclude, case=False, na=False)
    s = df[m].dropna(subset=["year", "val"]).groupby("year")["val"].mean().sort_index()
    return s


def main():
    print("=" * 70)
    print("ARTICLE 17 - NATIONAL CONTEXT (WHO GHO; national framing only)")
    print("=" * 70)

    vpd = load("vaccine_preventable_communicable_diseases_indicators_gha.csv")
    cm  = load("child_mortality_indicators_gha.csv")
    hw  = load("health_workforce_indicators_gha.csv")

    # DTP3 coverage is absent from this WHO file; use the coverage indicators that exist.
    mcv1 = series(vpd, "Measles.*immunization coverage")
    pol3 = series(vpd, r"Polio \(Pol3\)")
    u5mr = series(cm, "Under-five mortality rate")
    u5mr = u5mr[u5mr.index >= 1990]   # modern, relevant window (avoids 1932 historical outlier)
    measles_cases = series(vpd, "Measles - number of reported cases")

    # workforce density: match the explicit per-10,000 indicators only (not raw counts)
    docs = series(hw, r"Medical doctors \(per 10,000\)")
    nurses = series(hw, r"Nursing and midwifery personnel \(per 10,000\)")

    def span(s, label, unit=""):
        if s.empty:
            print(f"  {label}: (not found)")
            return None
        y0, y1 = s.index.min(), s.index.max()
        print(f"  {label}: {s.iloc[0]:.1f}{unit} ({int(y0)}) -> {s.iloc[-1]:.1f}{unit} ({int(y1)})")
        return s

    print("\nKey national series:")
    span(mcv1, "MCV1/measles coverage", "%")
    span(pol3, "Pol3 coverage", "%")
    span(u5mr, "Under-5 mortality (/1000, 1990+)")
    span(measles_cases, "Measles reported cases")
    if not docs.empty:
        print(f"  Medical doctors per 10,000 (latest {int(docs.index[-1])}): {docs.iloc[-1]:.2f}")
    if not nurses.empty:
        print(f"  Nurses/midwives per 10,000 (latest {int(nurses.index[-1])}): {nurses.iloc[-1]:.2f}")

    # ── Save tidy national-context table ──────────────────────────────────────
    parts = []
    for name, s in [("MCV1_coverage_pct", mcv1), ("Pol3_coverage_pct", pol3),
                    ("U5MR_per1000", u5mr), ("Measles_cases", measles_cases)]:
        if not s.empty:
            parts.append(s.rename(name))
    natl = pd.concat(parts, axis=1).reset_index().rename(columns={"year": "Year"})
    natl.to_csv(os.path.join(TABS, "national_context.csv"), index=False, encoding="utf-8")

    # ── 2-panel figure ────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    for s, key, lab in [(mcv1, "mcv1", "Measles (MCV1)"), (pol3, "pol3", "Polio (Pol3)")]:
        if not s.empty:
            axA.plot(s.index, s.values, color=CB[key], lw=2, label=lab)
            axA.annotate(lab, (s.index[-1], s.values[-1]), color=CB[key], fontsize=8,
                         xytext=(4, 0), textcoords="offset points", va="center")
    axA.set_ylim(0, 100); axA.set_xlabel("Year"); axA.set_ylabel("Coverage among 1-year-olds (%)")
    axA.set_title("A. Ghana national EPI coverage rose then plateaued", fontsize=9, loc="left")
    axA.axhline(80, ls="--", color="grey", lw=0.8); axA.grid(alpha=0.25)

    if not measles_cases.empty:
        axB.plot(measles_cases.index, measles_cases.values, color=CB["u5mr"], lw=2)
        axB.fill_between(measles_cases.index, measles_cases.values, alpha=0.12, color=CB["u5mr"])
        axB.set_yscale("log")
    axB.set_xlabel("Year"); axB.set_ylabel("Measles reported cases (log scale)")
    axB.set_title("B. National measles burden collapsed as coverage rose", fontsize=9, loc="left")
    axB.grid(alpha=0.25, which="both")

    fig.suptitle("Ghana national context (WHO GHO) — country-level framing; not used in the district model",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGS, "national_context_trends.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "national_context_trends.svg"), bbox_inches="tight")
    plt.close(fig)
    print("\n✓ outputs/figures/national_context_trends.png + national_context.csv written")

    import py_compile
    py_compile.compile(__file__, doraise=True)
    print("✓ Script syntax verified.")


if __name__ == "__main__":
    main()
