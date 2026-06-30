"""
Article 17 — Ghana Immunisation Equity
Script: 02_spatial_analysis.py
Phase: 4 — Spatial pipeline (ecological design, Epid Council M1)

Author: Valentine Golden Ghanem
Date: 2026-06-26
AIPOCH: v6.5

DESIGN (binding, M1):
  - Immunisation COVERAGE is region-level (16 units) -> mapped as a REGION choropleth only.
    NO district LISA/Moran's I on raw coverage (region-constant -> would re-draw region borders, MAUP).
  - The Immunisation District Risk Index (IDRI) is a DISTRICT-VARYING socioeconomic vulnerability
    surface built from GSS Census determinants. Spatial autocorrelation (Global Moran's I, LISA,
    Getis-Ord Gi*) is computed on the IDRI surface — a legitimate district-varying target.

INPUTS
  - data/processed/master_immunisation_ghana_261.csv          (261 x 37; from 01)
  - data/geospatial/crosswalk_260.csv                         (from 00b)
  - ../../Research Datasets/Ghana_New_260_District.geojson    (260 polygons)

OUTPUTS
  - data/processed/master_immunisation_ghana_261_spatial.csv  (261 rows + idri, lisa_cluster, hotspot_flag)
  - outputs/maps/region_coverage_choropleth.png|.svg
  - outputs/maps/idri_choropleth.png|.svg
  - outputs/maps/lisa_cluster_map.png|.svg
  - outputs/maps/getisord_hotspot_map.png|.svg
  - outputs/figures/moran_scatter_idri.png|.svg
  - outputs/tables/spatial_autocorrelation_summary.csv

Usage:
  python scripts/02_spatial_analysis.py
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

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from libpysal.weights import Queen, KNN
from libpysal.weights.util import attach_islands
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local


def _seed():
    """esda uses numpy's global RNG for conditional permutations — reseed for reproducibility."""
    np.random.seed(SEED)

SEED = 42
np.random.seed(SEED)
PERMUTATIONS = 9999
SIG = 0.05

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCE  = os.path.join(os.path.dirname(PROJECT_ROOT), "Research Datasets")
PROC         = os.path.join(PROJECT_ROOT, "data", "processed")
MAPS         = os.path.join(PROJECT_ROOT, "outputs", "maps")
FIGS         = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABS         = os.path.join(PROJECT_ROOT, "outputs", "tables")
for d in (MAPS, FIGS, TABS):
    os.makedirs(d, exist_ok=True)

# District-varying determinants for the IDRI (GSS Census; genuinely vary within region).
# Sign +1 = higher value -> higher vulnerability; -1 = protective (reverse-coded).
IDRI_COMPONENTS = {
    "poverty_incidence":    +1,
    "poverty_intensity":    +1,
    "census_uninsured_pct": +1,
    "illiterate_pct":       +1,
    "child_pop_pct":        +1,   # higher child dependency = more demand pressure
    "employed_pct":         -1,   # employment protective
}

LISA_LABELS = {0: "NS", 1: "HH", 2: "LH", 3: "LL", 4: "HL"}
LISA_COLORS = {"NS": "#d9d9d9", "HH": "#d7191c", "LH": "#abd9e9", "LL": "#2c7bb6", "HL": "#fdae61"}


def build_idri(df: pd.DataFrame) -> pd.Series:
    """Deterministic district vulnerability index: mean of signed z-scores, min-max scaled to 0-1."""
    z = pd.DataFrame(index=df.index)
    for col, sign in IDRI_COMPONENTS.items():
        s = df[col].astype(float)
        z[col] = sign * (s - s.mean()) / s.std(ddof=0)
    comp = z.mean(axis=1)
    return (comp - comp.min()) / (comp.max() - comp.min())


def main():
    print("=" * 70)
    print("ARTICLE 17 - SPATIAL ANALYSIS (Phase 4)")
    print("=" * 70)

    master = pd.read_csv(os.path.join(PROC, "master_immunisation_ghana_261.csv"))
    master["idri"] = build_idri(master)
    print(f"[1] Master: {master.shape} | IDRI range {master['idri'].min():.3f}-{master['idri'].max():.3f}")

    xw = pd.read_csv(os.path.join(PROC.replace('processed', 'geospatial'), "crosswalk_260.csv"))
    gdf = gpd.read_file(os.path.join(DATA_SOURCE, "Ghana_New_260_District.geojson"))
    gdf = gdf.merge(xw[["geojson_district", "master_sheet_district"]],
                    left_on="DISTRICT", right_on="geojson_district", how="left")
    gdf = gdf.merge(master, left_on="master_sheet_district", right_on="district_id", how="left")
    gdf = gdf[gdf["idri"].notna()].copy().reset_index(drop=True)   # 260 mapped (Guan excluded)
    print(f"[2] GeoDataFrame for mapping: {len(gdf)} polygons (Guan excluded — no geometry)")

    # ── Region coverage choropleth (raw coverage = region-level only) ─────────────
    reg = gdf.dissolve(by="Region", aggfunc="first")
    fig, ax = plt.subplots(figsize=(7, 8))
    reg.plot(column="imm_fully_vaccinated_pct", cmap="viridis", legend=True, ax=ax,
             edgecolor="white", linewidth=0.6,
             legend_kwds={"label": "Fully vaccinated (%)", "shrink": 0.6})
    ax.set_title("Childhood full immunisation coverage by region, Ghana (DHS 2022)\n"
                 "Region-level — coverage is not district-resolved (ecological design)",
                 fontsize=10)
    ax.axis("off")
    _save(fig, MAPS, "region_coverage_choropleth")

    # ── Spatial weights (queen contiguity; islands attached to nearest neighbour) ──
    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    n_islands = len(w.islands)
    if n_islands:
        w = attach_islands(w, KNN.from_dataframe(gdf, k=1, silence_warnings=True))
    w.transform = "r"
    print(f"[3] Queen weights: n={w.n} | islands attached={n_islands} | avg neighbours={w.mean_neighbors:.2f}")

    y = gdf["idri"].values

    # ── Global Moran's I (IDRI + poverty check) ───────────────────────────────────
    rows = []
    for name, vals in [("IDRI", y),
                       ("poverty_incidence", gdf["poverty_incidence"].values),
                       ("illiterate_pct", gdf["illiterate_pct"].values)]:
        _seed()
        mi = Moran(vals, w, permutations=PERMUTATIONS)
        rows.append({"variable": name, "morans_I": round(mi.I, 4),
                     "expected_I": round(mi.EI, 4), "z_sim": round(mi.z_sim, 3),
                     "p_sim": mi.p_sim})
        print(f"[4] Global Moran's I [{name}]: I={mi.I:.3f}, z={mi.z_sim:.2f}, p={mi.p_sim:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(TABS, "spatial_autocorrelation_summary.csv"), index=False)

    # ── Moran scatter (IDRI) ──────────────────────────────────────────────────────
    _seed()
    mi_idri = Moran(y, w, permutations=PERMUTATIONS)
    lag = (w.sparse @ ((y - y.mean()) / y.std(ddof=0)))
    zy = (y - y.mean()) / y.std(ddof=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(zy, lag, s=18, color="#1a5276", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    b = np.polyfit(zy, lag, 1)[0]
    xs = np.linspace(zy.min(), zy.max(), 50)
    ax.plot(xs, b * xs, color="#d7191c", lw=1.6)
    ax.set_xlabel("IDRI (standardised)"); ax.set_ylabel("Spatial lag of IDRI")
    ax.set_title(f"Moran scatterplot — IDRI clusters spatially\nMoran's I = {mi_idri.I:.3f} (p = {mi_idri.p_sim:.4f})",
                 fontsize=10)
    _save(fig, FIGS, "moran_scatter_idri")

    # ── LISA (Local Moran) on IDRI ────────────────────────────────────────────────
    _seed()
    lisa = Moran_Local(y, w, permutations=PERMUTATIONS)
    quad = np.where(lisa.p_sim < SIG, lisa.q, 0)
    gdf["lisa_cluster"] = [LISA_LABELS[q] for q in quad]
    counts = gdf["lisa_cluster"].value_counts().to_dict()
    print(f"[5] LISA clusters (p<{SIG}): {counts}")
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7, 8))
    for lab, col in LISA_COLORS.items():
        sub = gdf[gdf["lisa_cluster"] == lab]
        if len(sub):
            sub.plot(ax=ax, color=col, edgecolor="white", linewidth=0.4)
    handles = [Patch(facecolor=LISA_COLORS[l], edgecolor="grey",
                     label=f"{l} ({int((gdf['lisa_cluster'] == l).sum())})")
               for l in ["HH", "LL", "LH", "HL", "NS"] if (gdf["lisa_cluster"] == l).any()]
    ax.set_title("LISA clusters of district vulnerability (IDRI), Ghana\n"
                 "HH = high-vulnerability cluster (priority targeting)", fontsize=10)
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, title="Cluster (n)")
    ax.axis("off")
    _save(fig, MAPS, "lisa_cluster_map")

    # ── Getis-Ord Gi* hotspots on IDRI ────────────────────────────────────────────
    _seed()
    go = G_Local(y, w, star=True, permutations=PERMUTATIONS)
    gdf["gi_z"] = go.Zs
    gdf["gi_p"] = go.p_sim
    gdf["hotspot_flag"] = np.where((go.p_sim < SIG) & (go.Zs > 0), 1,
                            np.where((go.p_sim < SIG) & (go.Zs < 0), -1, 0))
    nhot = int((gdf["hotspot_flag"] == 1).sum()); ncold = int((gdf["hotspot_flag"] == -1).sum())
    print(f"[6] Getis-Ord Gi*: {nhot} hotspots (high-vulnerability), {ncold} coldspots")
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = ListedColormap(["#2c7bb6", "#d9d9d9", "#d7191c"])
    gdf.assign(_h=gdf["hotspot_flag"] + 1).plot(column="_h", cmap=cmap, categorical=True,
        ax=ax, edgecolor="white", linewidth=0.4, legend=True,
        legend_kwds={"labels": ["Coldspot (low)", "Not sig.", "Hotspot (high)"], "loc": "lower left"})
    ax.set_title("Getis-Ord Gi* hotspots of district vulnerability (IDRI), Ghana", fontsize=10)
    ax.axis("off")
    _save(fig, MAPS, "getisord_hotspot_map")

    # ── IDRI choropleth ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 8))
    gdf.plot(column="idri", cmap="YlOrRd", scheme="quantiles", k=5, legend=True, ax=ax,
             edgecolor="white", linewidth=0.4,
             legend_kwds={"title": "IDRI quintile", "loc": "lower left", "fontsize": 8})
    ax.set_title("Immunisation District Risk Index (IDRI) — structural vulnerability, Ghana",
                 fontsize=10)
    ax.axis("off")
    _save(fig, MAPS, "idri_choropleth")

    # ── Persist enriched master (261 rows; Guan -> NA cluster, 0 hotspot) ──────────
    spatial_cols = gdf[["district_id", "idri", "lisa_cluster", "gi_z", "hotspot_flag"]]
    out = master.merge(spatial_cols.drop(columns="idri"), on="district_id", how="left")
    out["lisa_cluster"] = out["lisa_cluster"].fillna("NA (no polygon)")
    out["hotspot_flag"] = out["hotspot_flag"].fillna(0).astype(int)
    out_path = os.path.join(PROC, "master_immunisation_ghana_261_spatial.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✓ Spatial-enriched master saved: {out.shape} -> {out_path}")
    print(f"  Figures/maps in outputs/maps and outputs/figures.")

    import py_compile
    py_compile.compile(__file__, doraise=True)
    print("✓ Script syntax verified.")


def _save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(folder, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
