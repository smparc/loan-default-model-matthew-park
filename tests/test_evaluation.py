import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from loan_risk.evaluation import (
    binary_metrics,
    confusion_table,
    convention_metrics,
    matched_tiers,
    paired_bootstrap,
    tier_table,
)


def test_binary_metrics_by_hand():
    m = binary_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["F1"] == pytest.approx(2 / 3)
    assert m["FPR"] == pytest.approx(1 / 2)
    assert m["FNR"] == pytest.approx(1 / 3)


def test_conventions_treat_the_review_tier_differently():
    y = np.array([1, 1, 0, 0])
    tiers = np.array(["deny", "review", "approve", "deny"], dtype=object)
    a, b = convention_metrics(y, tiers).iloc[0], convention_metrics(y, tiers).iloc[1]
    assert (a["TP"], a["FP"], a["FN"]) == (1, 1, 1)  # review counts as "predicted repay"
    assert (b["TP"], b["FP"], b["FN"]) == (2, 1, 0)  # review counts as "predicted default"


def test_confusion_table():
    table = confusion_table([1, 0, 1, 0, 0], [1, 1, 0, 0, 0])
    assert table.loc["defaulted", "predicted default"] == 1
    assert table.loc["repaid", "predicted default"] == 1
    assert table.to_numpy().sum() == 5


def test_matched_tiers_hit_the_volumes_in_risk_order():
    tiers = matched_tiers(np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2]), n_approve=2, n_deny=2)
    assert list(tiers) == ["deny", "approve", "review", "review", "deny", "approve"]


def test_tier_table_counts():
    table = tier_table([1, 0, 0, 1], np.array(["approve", "approve", "review", "deny"], dtype=object))
    assert table.loc["approve", "applicants"] == 2
    assert table.loc["deny", "default rate"] == 1.0
    assert table["share of applicants"].sum() == pytest.approx(1.0)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    noise = rng.normal(size=300)
    same = paired_bootstrap(y, noise, noise, roc_auc_score, n_boot=200)
    assert same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = paired_bootstrap(y, noise, y + 0.5 * noise, roc_auc_score, n_boot=200)
    assert better["difference"] > 0 and better["ci_low"] > 0
