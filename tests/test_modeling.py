import numpy as np
import pandas as pd
import pytest

from loan_risk.data import default_label
from loan_risk.features import CATEGORICAL, MONOTONIC, tree_matrix
from loan_risk.modeling import (
    ADDITIVE_GBM_PARAMS,
    PRODUCTION_VARIANT,
    cross_validate,
    cv_splits,
    make_additive_gbm,
    make_logistic,
    make_production_model,
    one_se_choice,
)


@pytest.fixture(scope="module")
def fitted(resolved, resolved_features):
    X = tree_matrix(resolved_features, PRODUCTION_VARIANT)
    return make_production_model(X.columns).fit(X, default_label(resolved)), X


def test_additive_model_settings(features):
    columns = tree_matrix(features, "V2").columns
    model = make_additive_gbm(columns)
    assert model.interaction_cst == "no_interactions"
    assert list(model.monotonic_cst) == [MONOTONIC[c] for c in columns]
    assert list(model.categorical_features) == [c in CATEGORICAL for c in columns]


def test_production_model_uses_the_tuned_settings(features):
    params = make_production_model(tree_matrix(features, PRODUCTION_VARIANT).columns).get_params()
    assert all(params[name] == value for name, value in ADDITIVE_GBM_PARAMS.items())


def test_model_has_no_interactions(fitted):
    # Changing one feature must move the log-odds by the same amount whatever the other features are.
    model, X = fitted
    rows = X.iloc[:5].copy()
    for column, (low, high) in {"loan_to_income": (0.2, 0.6), "bank_ending_balance": (100.0, 3000.0)}.items():
        lo, hi = rows.copy(), rows.copy()
        lo[column], hi[column] = low, high
        shift = model.decision_function(hi) - model.decision_function(lo)
        assert np.allclose(shift, shift[0], atol=1e-9)


def test_direction_constraints_hold(fitted):
    model, X = fitted
    for column, grid, direction in [
        ("loan_to_income", np.linspace(0.02, 3.0, 60), 1),
        ("withdrawal_to_deposit", np.linspace(0.3, 0.95, 60), 1),
        ("bank_ending_balance", np.linspace(0, 8000, 60), -1),
    ]:
        probe = pd.concat([X.iloc[[0]]] * len(grid), ignore_index=True)
        probe[column] = grid
        assert np.all(direction * np.diff(model.decision_function(probe)) >= -1e-12), column


def test_cross_validate_scores_every_row_once_per_repeat():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (X["a"] + rng.normal(size=200) > 0).astype(int).to_numpy()
    result = cross_validate(make_logistic, X, y, cv_splits(y, n_splits=4, n_repeats=2))
    assert result.oof.shape == (2, 200) and not np.isnan(result.oof).any()
    assert len(result.fold_metrics) == 8


def test_one_se_rule_prefers_the_simpler_candidate():
    summary = pd.DataFrame(
        {"log-loss": [0.530, 0.533, 0.560], "log-loss se": [0.004, 0.004, 0.004]}, index=["complex", "simple", "poor"]
    )
    assert one_se_choice(summary, simplest_first=["poor", "simple", "complex"]) == "simple"
    summary.loc["simple", "log-loss"] = 0.540
    assert one_se_choice(summary, simplest_first=["poor", "simple", "complex"]) == "complex"
