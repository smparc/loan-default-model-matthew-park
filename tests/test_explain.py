import numpy as np
import pytest

from loan_risk.baseline import oracle_default_probability, oracle_log_odds_terms
from loan_risk.data import default_label
from loan_risk.explain import (
    AdditiveExplainer,
    MATERIALITY,
    decision_for,
    describe,
    largest_approvable_loan,
    probability_with,
    report,
    sigmoid,
)
from loan_risk.features import MISMATCH, NOT_PROVIDED, VERIFIED, tree_matrix
from loan_risk.modeling import PRODUCTION_VARIANT, make_production_model


@pytest.fixture(scope="module")
def setup(resolved, resolved_features):
    X = tree_matrix(resolved_features, PRODUCTION_VARIANT)
    model = make_production_model(X.columns).fit(X, default_label(resolved))
    return model, X, AdditiveExplainer(model, X)


def test_contributions_add_up_to_the_prediction_exactly(setup):
    model, X, explainer = setup
    total = explainer.base_value + explainer.contributions(X).sum(axis=1)
    assert np.allclose(total, model.decision_function(X), atol=1e-9)


def test_contributions_average_zero_over_the_background(setup):
    _, X, explainer = setup
    assert np.allclose(explainer.contributions(X).mean(), 0, atol=1e-9)


def test_explanation_is_consistent_and_sorted(setup):
    model, X, explainer = setup
    row = X.iloc[[3]]
    probability, reasons = explainer.explain(row)
    assert probability == pytest.approx(float(sigmoid(model.decision_function(row))[0]))
    assert list(reasons["contribution"].abs()) == sorted(reasons["contribution"].abs(), reverse=True)
    assert (reasons["material"] == (reasons["contribution"].abs() >= MATERIALITY)).all()
    # Removing a factor that raises risk must lower the predicted risk, and vice versa.
    without = reasons["risk if this factor were typical"]
    assert ((reasons["contribution"] > 0) == (without < probability)).all()


def test_plain_english_descriptions():
    assert describe("income_verification", NOT_PROVIDED) == "No income documents provided"
    assert describe("income_verification", MISMATCH) == "Documented income is far below stated income"
    assert describe("income_verification", VERIFIED).startswith("Income verified")
    assert describe("loan_to_income", 0.456) == "Loan is 46% of stated monthly income"
    assert describe("bank_ending_balance", 1234.0) == "Bank balance of $1,234"
    assert describe("bank_has_overdrafts", 1) == "Overdrafts on the account"


def test_decisions_and_report():
    assert decision_for(0.10, 0.2, 0.5) == "auto-approve"
    assert decision_for(0.30, 0.2, 0.5) == "send to review"
    assert decision_for(0.50, 0.2, 0.5) == "deny"


def test_report_lists_material_reasons(setup):
    _, X, explainer = setup
    probability, reasons = explainer.explain(X.iloc[[0]])
    text = report("APP-TEST", probability, reasons, 0.2, 0.5, 0.25)
    assert text.startswith("APP-TEST: predicted default risk")
    for factor in reasons.loc[reasons["material"], "factor"]:
        assert factor in text


def test_largest_approvable_loan_respects_the_cutoff(setup, resolved):
    _, X, explainer = setup
    row, income = X.iloc[[0]], resolved["stated_monthly_income"].iloc[0]
    assert largest_approvable_loan(explainer, row, income, approve_below=1.0) == 5000
    assert largest_approvable_loan(explainer, row, income, approve_below=0.0) is None
    loan = largest_approvable_loan(explainer, row, income, approve_below=0.25)
    if loan is not None:
        assert probability_with(explainer, row, loan_to_income=loan / income) < 0.25


def test_oracle_terms_rebuild_the_oracle_probability(applications):
    terms = oracle_log_odds_terms(applications)
    assert np.allclose(sigmoid(terms.sum(axis=1) - 2.25), oracle_default_probability(applications), atol=1e-12)
