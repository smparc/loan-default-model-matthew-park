import pytest

from loan_risk.data import load_applications, split_by_outcome
from loan_risk.features import build_features


@pytest.fixture(scope="session")
def applications():
    return load_applications()


@pytest.fixture(scope="session")
def resolved(applications):
    return split_by_outcome(applications)[0]


@pytest.fixture(scope="session")
def features(applications):
    return build_features(applications)
