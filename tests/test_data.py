import numpy as np
import pytest

from loan_risk.data import default_label, split_by_outcome, train_test_indices


def test_shape(applications):
    assert applications.shape == (2000, 15)


def test_nulls_only_where_expected(applications):
    nulls = applications.isna().sum()
    assert nulls[nulls > 0].to_dict() == {"documented_monthly_income": 283, "days_to_default": 1455}


def test_documented_income_missing_exactly_when_no_documents(applications):
    missing = applications["documented_monthly_income"].isna()
    assert (missing == applications["num_documents_submitted"].eq(0)).all()


def test_days_to_default_only_known_for_defaults(applications):
    # Filled in only after a default happens, so using it as a feature would leak the answer.
    known = applications["days_to_default"].notna()
    assert (known == applications["actual_outcome"].eq("defaulted")).all()


def test_split_by_outcome(applications):
    resolved, ongoing = split_by_outcome(applications)
    assert (len(resolved), len(ongoing)) == (1836, 164)
    assert not resolved["actual_outcome"].eq("ongoing").any()


def test_default_label(resolved):
    y = default_label(resolved)
    assert y.sum() == 545
    assert set(np.unique(y)) == {0, 1}


def test_default_label_rejects_ongoing_loans(applications):
    with pytest.raises(ValueError):
        default_label(applications)


def test_train_test_split_is_stratified_and_disjoint(resolved):
    y = default_label(resolved)
    train, test = train_test_indices(y)
    assert (len(train), len(test)) == (1377, 459)
    assert set(train).isdisjoint(test)
    assert len(set(train) | set(test)) == len(y)
    assert abs(y[train].mean() - y[test].mean()) < 0.005
