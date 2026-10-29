import numpy as np

from loan_risk.fairness import score_conditional_odds

GROUPS = ["employed", "self_employed"]


def _simulate(group_effect: float, seed: int, n: int = 6000):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    group = rng.choice(GROUPS, size=n)
    p = 1 / (1 + np.exp(-(score + group_effect * (group == "self_employed"))))
    return (rng.uniform(size=n) < p).astype(int), score, group


def test_detects_a_group_the_score_over_penalizes():
    y, score, group = _simulate(group_effect=-1.0, seed=0)  # defaults less than the score implies
    result = score_conditional_odds(y, score, group, order=GROUPS)
    assert result.loc["self_employed", "odds ratio"] < 0.5
    assert result.loc["self_employed", "p-value"] < 0.001


def test_finds_nothing_when_the_score_is_fair():
    y, score, group = _simulate(group_effect=0.0, seed=1)
    result = score_conditional_odds(y, score, group, order=GROUPS)
    assert result.loc["self_employed", "ci_low"] < 1 < result.loc["self_employed", "ci_high"]


def test_group_outcomes_by_hand():
    from loan_risk.fairness import group_outcomes

    y = np.array([0, 1, 0, 0, 1, 0])
    tiers = np.array(["approve", "approve", "deny", "approve", "review", "review"], dtype=object)
    groups = np.array(["employed", "employed", "employed", "self_employed", "self_employed", "self_employed"])
    table = group_outcomes(y, tiers, groups, order=GROUPS, p=[0.1, 0.5, 0.2, 0.1, 0.6, 0.3])
    employed, self_employed = table.loc["employed"], table.loc["self_employed"]
    assert employed["auto-approved"] == 2 / 3 and self_employed["auto-approved"] == 1 / 3
    assert self_employed["approval ratio"] == 0.5
    assert employed["good applicants denied"] == 0.5  # 1 of 2 repaid employed applicants denied
    assert self_employed["good applicants not auto-approved"] == 0.5
    assert employed["defaults auto-approved"] == 1.0 and self_employed["defaults auto-approved"] == 0.0
    assert employed["mean predicted risk"] == np.mean([0.1, 0.5, 0.2])
