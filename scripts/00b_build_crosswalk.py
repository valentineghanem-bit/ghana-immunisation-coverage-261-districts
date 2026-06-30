"""
Article 17 — Ghana Immunisation Equity
Script: 00b_build_crosswalk.py
Phase: 1 — Build & hard-validate the 261 (Master Sheet) <-> 260 (GeoJSON) district crosswalk

Author: Valentine Golden Ghanem
Date: 2026-06-26
AIPOCH: v6.5

WHY THIS SCRIPT EXISTS
  The Cowork phase-1 log asserted a "260/260" crosswalk but no code produced it (audit finding E4).
  The vetted Project-15 crosswalk was checked against THIS project's raw GeoJSON and found to
  declare two FALSE structural gaps (Awutu Senya West, Sagnarigu Municipal) — the raw file
  `Ghana_New_260_District.geojson` actually contains the matching `AWUTU SENYA` and `SAGNERIGU`
  polygons. This script regenerates the crosswalk DIRECTLY from the raw datasets, with every
  non-exact match explicitly vetted, so the only true 261->260 gap is Guan (no 2017 polygon).

INPUTS  (../../Research Datasets/)
  - Master Sheet.xlsx                 (GSS Census 2021; 261 districts)
  - Ghana_New_260_District.geojson    (2017 boundaries; 260 polygons; DISTRICT/REGION props)

OUTPUT
  - ../data/geospatial/crosswalk_260.csv
    columns: master_sheet_region, master_sheet_district, geojson_district, geojson_region,
             match_method, note

HARD VALIDATION (asserts; script fails loudly if any breaks)
  - 261 master rows; set-equal to raw Master Sheet districts
  - every one of the 260 GeoJSON polygons assigned to exactly one master district
  - no master district receives two polygons
  - exactly ONE structural gap (master with no polygon) = Guan

Usage:
  python scripts/00b_build_crosswalk.py
"""

import os
import re
import sys
import json
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCE  = os.path.join(os.path.dirname(PROJECT_ROOT), "Research Datasets")
GEO_OUT      = os.path.join(PROJECT_ROOT, "data", "geospatial")
os.makedirs(GEO_OUT, exist_ok=True)

# Explicitly vetted matches for every GeoJSON DISTRICT that does NOT exact-normalise to a
# Master Sheet district. Key = raw GeoJSON DISTRICT string; value = exact Master Sheet district.
# Each was checked against the raw Master Sheet (see audit 2026-06-26). Dangerous fuzzy
# look-alikes that were REJECTED are noted inline.
OVERRIDE = {
    "ADENTA MUNICIPAL":              "Adentan Municipal",
    "KASENA NANKANA EAST":           "Kasena Nankana Municipal",   # NOT 'Kasena Nankana West' (the East unit = Municipal)
    "MFANTSEMAN MUNICIPAL":          "Mfantsiman Municipal",
    "TWIFO HEMANG LOWER DENKYIRA":   "Twifo Heman Lower Denkyira",
    "TWIFO ATTI-MORKWA":             "Twifo Ati Morkwa",
    "AWUTU SENYA":                   "Awutu Senya West",           # residual polygon (East = 'AWUTU SENYA EAST')
    "AGOTIME ZIOPE":                 "Agortime-Ziope",
    "BOSOMTWE":                      "Bosomtwi",
    "DORMAA MUNICIPAL":              "Dormaa Central Municipal",   # NOT Dormaa West/East
    "CAPE COAST METROPOLITAN":       "Cape Cape Metropolitan Area (CCMA)-Cape Coast South & Cape Coast North",
    "SEKYERE AFRAM PLAINS NORTH":    "Sekyere Afram Plains",       # master has single Sekyere Afram Plains
    "TAMALE METROPOLITAN":           "Tamale Metropolitan Area (TMA)-Tamale Central & Tamale South",
    "SAGNERIGU":                     "Sagnarigu Municipal",        # spelling variant; NOT absorbed in Tamale
    "BOLGA  EAST":                   "Bolgatanga East",            # NOT 'Ga East'
    "ASSIN FOSU":                    "Assin Central Municipal",    # Assin Fosu = capital of Assin Central; NOT Assin South
    "ADANSI AKROFUOM":               "Akrofuom",
    "KUMASI METROPOLITAN":           "Kumasi Metropolitan Area (KMA)-Bantama, Manhyia North, Manhyia South, Nhyiaeso, & Subin",
    "OKAIKWEI NORTH MUNICIPAL":      "Okaikoi North Municipal",
    "TEMA METROPOLITAN":             "Tema Metropolitan Area (TMA)-Tema Central & Tema East",
    "ACCRA METROPOLIS":              "Accra Metropolitan Area (AMA)-Ablekuma South, Ashiedu Keteke & Okaikoi South",
    "SEKONDI TAKORADI METROPOLIS":   "Sekondi Takoradi Metropolitan Area (STMA)- Takoradi, Sekondi & Essikado-Ketan",
    "ASENE AKROSO MANSO":            "Asene Manso Akroso",
    "UPPER MANYA":                   "Upper Manya Krobo",
    "LOWER MANYA":                   "Lower Manya Krobo Municipal",
    "AKWAPEM SOUTH":                 "Akwapim South Municipal",
    "DENKYEMBOUR":                   "Denkyembuor",
    "AKYEM MANSA":                   "Akyemansa",
    "AKWAPEM NORTH":                 "Akwapim North Municipal",
}

# Master district with no 2017 polygon (created after the boundary file) — the sole 261->260 gap.
STRUCTURAL_GAPS = {
    "Guan": "Created 2019 (LI 2363), Oti Region; absent from 2017 GeoJSON. Tabular/ML only; excluded from maps.",
}

SUFFIX = re.compile(r"\b(METROPOLITAN|METROPOLIS|MUNICIPAL(ITY)?|DISTRICT|ASSEMBLY|AREA|TMA)\b")


def norm(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("-", " ").replace("/", " ").replace(".", " ").replace(",", " ").replace("'", "")
    s = SUFFIX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    print("=" * 70)
    print("ARTICLE 17 - BUILD 261<->260 DISTRICT CROSSWALK (Phase 1)")
    print("=" * 70)

    ms = pd.read_excel(os.path.join(DATA_SOURCE, "Master Sheet.xlsx"))
    ms.rename(columns={"Metropolitan, Municipal, and District Assemblies (MMDA's)": "District"}, inplace=True)
    ms["District"] = ms["District"].astype(str).str.strip()
    master = ms["District"].tolist()
    master_region = dict(zip(ms["District"], ms["Region"].astype(str).str.strip()))
    norm_to_master = {}
    for m in master:
        norm_to_master.setdefault(norm(m), m)

    gj = json.load(open(os.path.join(DATA_SOURCE, "Ghana_New_260_District.geojson"), encoding="utf-8"))
    geo = [(f["properties"]["DISTRICT"].strip(), f["properties"]["REGION"].strip()) for f in gj["features"]]

    rows = []
    used = {}
    for gd, gr in geo:
        if gd in OVERRIDE:
            md, method = OVERRIDE[gd], "manual_vetted"
        elif norm(gd) in norm_to_master:
            md, method = norm_to_master[norm(gd)], "exact_normalized"
        else:
            raise AssertionError(f"FAIL: GeoJSON district {gd!r} has no exact match and no override")
        assert md not in used, f"FAIL: master {md!r} assigned twice ({used.get(md)!r} and {gd!r})"
        used[md] = gd
        rows.append({
            "master_sheet_region": master_region[md], "master_sheet_district": md,
            "geojson_district": gd, "geojson_region": gr, "match_method": method, "note": "",
        })

    # structural gaps (master with no polygon)
    for gap, note in STRUCTURAL_GAPS.items():
        assert gap in master, f"FAIL: declared gap {gap!r} not in Master Sheet"
        assert gap not in used, f"FAIL: {gap!r} declared a gap but was matched to {used.get(gap)!r}"
        rows.append({
            "master_sheet_region": master_region[gap], "master_sheet_district": gap,
            "geojson_district": "", "geojson_region": "", "match_method": "structural_gap", "note": note,
        })

    xw = pd.DataFrame(rows)

    # ── HARD VALIDATION ──────────────────────────────────────────────────────
    assert len(xw) == 261, f"FAIL: expected 261 rows, got {len(xw)}"
    assert set(xw["master_sheet_district"]) == set(master), "FAIL: master side != raw Master Sheet"
    matched = xw[xw["match_method"] != "structural_gap"]
    assert len(matched) == 260, f"FAIL: expected 260 matched, got {len(matched)}"
    assert matched["geojson_district"].nunique() == 260, "FAIL: not all 260 polygons used uniquely"
    assert set(matched["geojson_district"]) == {g for g, _ in geo}, "FAIL: matched polygons != raw GeoJSON set"
    gaps = xw[xw["match_method"] == "structural_gap"]["master_sheet_district"].tolist()
    assert gaps == ["Guan"], f"FAIL: expected sole gap Guan, got {gaps}"

    out = os.path.join(GEO_OUT, "crosswalk_260.csv")
    xw.to_csv(out, index=False, encoding="utf-8")
    print(f"\n✓ Crosswalk written: {out}")
    print(f"  261 master districts | 260 polygons matched 1:1 | sole structural gap: Guan")
    print(f"  match methods: {xw['match_method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
