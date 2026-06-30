"""
Validation tests for the canonical Master CSV (Article 17 — Ghana Immunisation).
Runs in CI without the raw source data: it validates the committed final dataset.

    pytest tests/test_master_csv.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "data", "processed", "master_immunisation_ghana_261_final.csv")

CORE_COMPLETE = [  # columns that must have ZERO missing values
    "district_id", "Region", "imm_fully_vaccinated_pct", "imm_no_vaccination_pct",
    "dpt_dropout_rate", "imm_coverage_composite", "poverty_incidence", "illiterate_pct",
    "risk_index_binary", "risk_index_binary_rel", "idri", "rf_risk_score", "xgb_risk_score",
]


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(CSV), f"Master CSV missing: {CSV}"
    return pd.read_csv(CSV)


def test_row_count(df):
    assert len(df) == 261, f"expected 261 districts, got {len(df)}"


def test_sixteen_regions(df):
    assert df["Region"].nunique() == 16


def test_guan_present(df):
    assert df["district_id"].str.contains("Guan", case=False).any()


def test_core_columns_complete(df):
    miss = df[CORE_COMPLETE].isnull().sum().sum()
    assert miss == 0, f"{miss} missing values in core columns"


def test_only_expected_missing(df):
    # The single legitimate NaN: gi_z for Guan (no 2017 polygon -> no Getis-Ord z-score).
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    assert list(miss.index) == ["gi_z"] and int(miss.iloc[0]) == 1, f"unexpected missing: {dict(miss)}"


def test_primary_outcome_binary(df):
    assert set(df["risk_index_binary"].dropna().unique()) <= {0, 1}
    assert set(df["risk_index_binary_rel"].dropna().unique()) <= {0, 1}


def test_primary_outcome_count(df):
    # absolute <80% target -> 180/261 high-risk (M3)
    assert int(df["risk_index_binary"].sum()) == 180


def test_dropout_formula(df):
    assert np.allclose(df["dpt_dropout_rate"], df["imm_dpt1_pct"] - df["imm_dpt3_pct"])


def test_idri_unit_interval(df):
    assert df["idri"].between(0, 1).all()


def test_coverage_composite_unit_interval(df):
    assert df["imm_coverage_composite"].between(0, 1).all()


def test_risk_scores_are_probabilities(df):
    for col in ["rf_risk_score", "xgb_risk_score"]:
        assert df[col].between(0, 1).all(), f"{col} outside [0,1]"
