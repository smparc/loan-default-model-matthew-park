import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from loan_risk.eda import adversarial_auc, binned_rate, format_rates, rate_table


def test_rate_table_counts_and_interval():
    y = [1, 0, 0, 1, 1, 0, 0, 0]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    table = rate_table(y, groups, order=["b", "a"])
    assert list(table.index) == ["b", "a"]
    assert (table.loc["a", "n"], table.loc["a", "defaults"], table.loc["a", "rate"]) == (4, 2, 0.5)
    low, high = proportion_confint(1, 4, method="wilson")
    assert np.isclose(table.loc["b", "ci_low"], low) and np.isclose(table.loc["b", "ci_high"], high)


def test_binned_rate_covers_every_row():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < x).astype(int)
    table = binned_rate(x, y, n_bins=10)
    assert len(table) == 10 and table["n"].sum() == 1000
    assert table["rate"].iloc[-1] > table["rate"].iloc[0]


def test_adversarial_auc_is_near_half_for_a_random_split():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=list("abc"))
    target = rng.uniform(size=600) < 0.2
    for auc in adversarial_auc(X, target).values():
        assert 0.35 < auc < 0.65


def test_format_rates_shows_percentages():
    table = rate_table([1, 0, 0, 0], ["a", "a", "a", "a"])
    shown = format_rates(table)
    assert shown.loc["a", "rate (%)"] == 25.0
    assert "ci_low (%)" in shown.columns
