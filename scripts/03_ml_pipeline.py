"""
Article 17 — Ghana Immunisation Equity
Script: 03_ml_pipeline.py
Phase: 5 — ML risk-classification pipeline (ecological, Epid Council M1/M2/M3)

Author: Valentine Golden Ghanem
Date: 2026-06-26
AIPOCH: v6.5

BINDING COUNCIL RULINGS ENCODED:
  M2 (leakage)  — features are STRUCTURAL DETERMINANTS only. ALL immunisation-coverage variables,
                  derived coverage indices, and downstream consequences (U5MR/IMR/NMR/diarrhoea) are
                  EXCLUDED. The model never sees the outcome's constituents.
  M1 (ecology)  — the binary outcome is region-level (constant within region). Validation uses
                  LEAVE-ONE-REGION-OUT cross-validation (16 folds): train on 15 regions, predict the
                  held-out region. Out-of-fold (OOF) probabilities are the honest, reported scores.
                  True N for the outcome = 16; district features add within-region texture only.
  M3 (outcome)  — PRIMARY outcome `risk_index_binary` (absolute <80%, imbalanced 180/261) modelled with
                  class weighting; `risk_index_binary_rel` reported as sensitivity.
  [20][21]      — a LOGISTIC REGRESSION baseline is reported alongside RF/XGBoost; calibration (Brier)
                  is reported (per Christodoulou 2019 / Boakye 2025: ML rarely beats LR; report calibration).

INPUT
  - data/processed/master_immunisation_ghana_261_spatial.csv   (261 x 41; from 02)
OUTPUT
  - data/processed/master_immunisation_ghana_261_final.csv     (+ rf_risk_score, xgb_risk_score, shap_top_feature)
  - outputs/tables/ml_performance.csv
  - outputs/figures/roc_curves.png|.svg
  - outputs/figures/shap_summary.png|.svg

Usage:
  python scripts/03_ml_pipeline.py
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, confusion_matrix
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(PROJECT_ROOT, "data", "processed")
FIGS = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABS = os.path.join(PROJECT_ROOT, "outputs", "tables")
for d in (FIGS, TABS):
    os.makedirs(d, exist_ok=True)

# Determinant-only features (M2). District-varying GSS + coordinates, then region-level structural context.
FEATURES = [
    # district-varying (GSS Census)
    "poverty_incidence", "poverty_intensity", "census_uninsured_pct", "illiterate_pct",
    "employed_pct", "child_pop_pct", "log_total_pop", "Latitude", "Longitude",
    # region-level structural determinants (context; flagged region-level)
    "facility_delivery_pct", "women_no_education_pct", "dhs_no_insurance_women_pct", "women_literate_pct",
]
# Hard exclusion guard — these must NEVER enter the feature matrix.
FORBIDDEN = {
    "imm_bcg_pct", "imm_dpt1_pct", "imm_dpt2_pct", "imm_dpt3_pct", "imm_polio1_pct", "imm_polio2_pct",
    "imm_polio3_pct", "imm_measles_pct", "imm_fully_vaccinated_pct", "imm_no_vaccination_pct",
    "dpt_dropout_rate", "imm_coverage_composite", "cm_u5mr", "cm_imr", "cm_nmr", "diarrhea_prev_pct",
    "risk_index_binary", "risk_index_binary_rel", "idri", "lisa_cluster", "gi_z", "hotspot_flag",
}


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {"AUC": round(roc_auc_score(y, p), 3), "Sensitivity": round(sens, 3),
            "Specificity": round(spec, 3), "Accuracy": round((tp + tn) / len(y), 3),
            "Brier": round(brier_score_loss(y, p), 3)}


def oof_predict(model_fn, X, y, groups):
    """Leave-one-region-out out-of-fold probabilities (M1)."""
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    print("=" * 70)
    print("ARTICLE 17 - ML PIPELINE (Phase 5)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(PROC, "master_immunisation_ghana_261_spatial.csv"))
    df["log_total_pop"] = np.log10(df["total_pop"])

    assert not (set(FEATURES) & FORBIDDEN), "FAIL: a forbidden (leakage) column is in FEATURES"
    print(f"[1] Features ({len(FEATURES)}): determinant-only (M2 leakage guard passed)")
    X = df[FEATURES].astype(float).values
    groups = df["Region"].values

    pos = int(df["risk_index_binary"].sum()); neg = len(df) - pos
    spw = neg / pos
    print(f"[2] Primary outcome risk_index_binary: {pos}/{len(df)} positive (imbalanced) | scale_pos_weight={spw:.3f}")

    model_factories = {
        "LogisticRegression": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))]),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, random_state=SEED,
            eval_metric="logloss", verbosity=0),
    }

    rows = []
    oof_store = {}
    for outcome in ["risk_index_binary", "risk_index_binary_rel"]:
        y = df[outcome].astype(int).values
        for name, fac in model_factories.items():
            oof = oof_predict(fac, X, y, groups)
            m = metrics(y, oof)
            m.update({"model": name, "outcome": outcome})
            rows.append(m)
            if outcome == "risk_index_binary":
                oof_store[name] = oof
            tag = "PRIMARY" if outcome == "risk_index_binary" else "sensitivity"
            print(f"[3] {tag:11} {name:18} AUC={m['AUC']} Sens={m['Sensitivity']} "
                  f"Spec={m['Specificity']} Brier={m['Brier']}")

    perf = pd.DataFrame(rows)[["outcome", "model", "AUC", "Sensitivity", "Specificity", "Accuracy", "Brier"]]
    perf.to_csv(os.path.join(TABS, "ml_performance.csv"), index=False)

    # ── ROC curves (primary outcome, OOF) ─────────────────────────────────────────
    y = df["risk_index_binary"].astype(int).values
    fig, ax = plt.subplots(figsize=(6, 6))
    palette = {"LogisticRegression": "#1a5276", "RandomForest": "#1e8449", "XGBoost": "#b9770e"}
    for name, oof in oof_store.items():
        fpr, tpr, _ = roc_curve(y, oof)
        ax.plot(fpr, tpr, color=palette[name], lw=1.8,
                label=f"{name} (AUC={roc_auc_score(y, oof):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Leave-one-region-out ROC — district immunisation-risk classification\n"
                 "(honest ecological validation; determinant-only features)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "roc_curves.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "roc_curves.svg"), bbox_inches="tight")
    plt.close(fig)

    # ── Final risk scores (OOF probabilities) ─────────────────────────────────────
    df["rf_risk_score"] = np.round(oof_store["RandomForest"], 4)
    df["xgb_risk_score"] = np.round(oof_store["XGBoost"], 4)

    # ── SHAP (XGBoost fit on all data; top feature per district) ───────────────────
    shap_top = pd.Series(["NA"] * len(df))
    try:
        import shap
        xgb_full = model_factories["XGBoost"]()
        xgb_full.fit(X, y)
        explainer = shap.TreeExplainer(xgb_full)
        sv = explainer.shap_values(X)
        shap_top = pd.Series([FEATURES[i] for i in np.argmax(np.abs(sv), axis=1)])
        plt.figure()
        shap.summary_plot(sv, pd.DataFrame(X, columns=FEATURES), show=False, plot_size=(8, 6))
        plt.title("SHAP — drivers of predicted district immunisation risk (XGBoost)", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "shap_summary.png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(FIGS, "shap_summary.svg"), bbox_inches="tight")
        plt.close()
        print(f"[4] SHAP top-feature distribution: {shap_top.value_counts().head(5).to_dict()}")
    except Exception as e:
        print(f"[4] SHAP skipped: {e}")
    df["shap_top_feature"] = shap_top.values

    out_path = os.path.join(PROC, "master_immunisation_ghana_261_final.csv")
    df.drop(columns=["log_total_pop"]).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✓ Final master saved: {df.drop(columns=['log_total_pop']).shape} -> {out_path}")

    # Headline interpretation (per [20][21])
    best = perf[perf.outcome == "risk_index_binary"].sort_values("AUC", ascending=False).iloc[0]
    lr = perf[(perf.outcome == "risk_index_binary") & (perf.model == "LogisticRegression")].iloc[0]
    print(f"\n  Best model: {best['model']} AUC={best['AUC']} vs LogisticRegression AUC={lr['AUC']} "
          f"(ΔAUC={best['AUC']-lr['AUC']:+.3f})")
    print("  Interpretation: report whether ML meaningfully beats LR (Christodoulou 2019 [20]: often not).")

    import py_compile
    py_compile.compile(__file__, doraise=True)
    print("✓ Script syntax verified.")


if __name__ == "__main__":
    main()
