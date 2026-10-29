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
