import numpy as np

from loan_risk.baseline import (
    adjusted_rule_score,
    oracle_default_probability,
    rebuild_rule_score,
    rule_components,
    rule_tiers,
    tiers_from_score,
)
from loan_risk.data import ID_COLUMN, OUTCOME_COLUMNS, RULE_COLUMNS


def test_rebuilt_rubric_reproduces_every_decision(applications):
    rebuilt = rebuild_rule_score(applications)
    assert (tiers_from_score(rebuilt) == rule_tiers(applications)).all()
    gap = (rebuilt - applications["rule_based_score"]).abs().round(2)
    # The CSV rounds withdrawals, deposits and documented income, so a few scores differ slightly.
    # One applicant's rounded documented income lands just outside the 10% match tolerance.
    assert (gap > 0.1).sum() == 1


def test_partial_income_match_branch_fires_only_once(applications):
    assert (rule_components(applications)["income_verification"] == 40).sum() == 1


def test_applicants_without_documents_are_never_auto_approved(applications):
    no_docs = applications["documented_monthly_income"].isna()
    assert applications.loc[no_docs, "rule_based_score"].max() < 75


def test_adjusted_score_changes_only_the_targeted_applicants(applications):
    original = applications["rule_based_score"]
    self_employed = applications["employment_status"].eq("self_employed")
    change = (adjusted_rule_score(applications, self_employed_points=100) - original).round(1)
    assert (change[self_employed] == 6.0).all() and (change[~self_employed] == 0.0).all()
    consistent = applications["bank_has_consistent_deposits"]
    change = (adjusted_rule_score(applications, ignore_deposit_consistency=True) - original).round(1)
    assert (change[~consistent] == 6.0).all() and (change[consistent] == 0.0).all()


def test_oracle_probability_uses_only_applicant_columns(applications):
    p = oracle_default_probability(applications)
    assert p.min() >= 1 / (1 + np.exp(2.25)) - 1e-12
    assert p.max() <= 1 / (1 + np.exp(-2.75)) + 1e-12
    scrambled = applications.copy()
    rng = np.random.default_rng(0)
    for column in [ID_COLUMN, *RULE_COLUMNS, *OUTCOME_COLUMNS]:
        scrambled[column] = rng.permutation(scrambled[column].to_numpy())
    assert np.array_equal(p, oracle_default_probability(scrambled))


def test_thresholds():
    assert list(tiers_from_score([49.9, 50, 74.9, 75])) == ["deny", "review", "review", "approve"]
