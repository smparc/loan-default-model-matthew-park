import numpy as np
import pandas as pd

from loan_risk.data import ID_COLUMN, OUTCOME_COLUMNS, RULE_COLUMNS
from loan_risk.features import (
    CANDIDATE_FEATURES,
    CORE_FEATURES,
    EMPLOYMENT_VARIANTS,
    MISMATCH,
    MISMATCH_THRESHOLD,
    MONOTONIC,
    NOT_PROVIDED,
    VERIFIED,
    build_features,
    income_verification,
    linear_matrix,
    loan_to_income_alternatives,
    tree_matrix,
)


def test_income_verification_groups(applications):
    counts = income_verification(applications).value_counts().to_dict()
    assert counts == {VERIFIED: 1632, MISMATCH: 85, NOT_PROVIDED: 283}


def test_mismatch_cutoff_sits_in_the_empty_gap(applications):
    ratio = (applications["documented_monthly_income"] / applications["stated_monthly_income"]).dropna()
    assert ((ratio < 0.40) | (ratio > 0.89)).all()
    assert 0.40 < MISMATCH_THRESHOLD < 0.89


def test_features_ignore_outcome_and_rule_columns(applications):
    # Leakage guard: scrambling everything that isn't an applicant input must not change any feature.
    scrambled = applications.copy()
    rng = np.random.default_rng(0)
    for column in [ID_COLUMN, *RULE_COLUMNS, *OUTCOME_COLUMNS]:
        scrambled[column] = rng.permutation(scrambled[column].to_numpy())
    pd.testing.assert_frame_equal(build_features(applications), build_features(scrambled))


def test_model_inputs_exclude_non_applicant_columns(features):
    forbidden = {ID_COLUMN, *RULE_COLUMNS, *OUTCOME_COLUMNS}
    for variant in EMPLOYMENT_VARIANTS:
        assert forbidden.isdisjoint(tree_matrix(features, variant).columns)
        assert forbidden.isdisjoint(linear_matrix(features, variant).columns)


def test_employment_variants(features):
    v1, v2, v3 = (tree_matrix(features, v) for v in ("V1", "V2", "V3"))
    assert "employment" in v1 and "unemployed" not in v1
    assert "unemployed" in v2 and "employment" not in v2
    assert list(v3.columns) == CORE_FEATURES
    assert {"self_employed", "unemployed"} <= set(linear_matrix(features, "V1").columns)
    assert "self_employed" not in linear_matrix(features, "V2")


def test_every_model_column_has_a_direction_rule(features):
    columns = set(tree_matrix(features, "V1", tuple(CANDIDATE_FEATURES)).columns) | {"unemployed"}
    assert columns <= set(MONOTONIC)


def test_model_inputs_are_finite(features):
    for variant in EMPLOYMENT_VARIANTS:
        assert np.isfinite(tree_matrix(features, variant).to_numpy(dtype=float)).all()
        assert np.isfinite(linear_matrix(features, variant).to_numpy(dtype=float)).all()


def test_ratios_match_their_definitions(applications, features):
    row, f = applications.iloc[0], features.iloc[0]
    assert f["loan_to_income"] == row["loan_amount"] / row["stated_monthly_income"]
    assert f["withdrawal_to_deposit"] == row["monthly_withdrawals"] / row["monthly_deposits"]
    alternatives = loan_to_income_alternatives(applications)
    assert alternatives["stated income"].equals(features["loan_to_income"])
    assert np.isfinite(alternatives.to_numpy()).all()
