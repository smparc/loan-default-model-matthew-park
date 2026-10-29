"""Load the loan applications and define the prediction target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "loan_applications.csv"
SEED = 42

ID_COLUMN = "applicant_id"
# Known when the application is made: the only columns features may use.
APPLICANT_COLUMNS = [
    "stated_monthly_income",
    "documented_monthly_income",
    "loan_amount",
    "employment_status",
    "bank_ending_balance",
    "bank_has_overdrafts",
    "bank_has_consistent_deposits",
    "monthly_withdrawals",
    "monthly_deposits",
    "num_documents_submitted",
]
# Outputs of the current rule-based system: the baseline, never a model input.
RULE_COLUMNS = ["rule_based_score", "rule_based_decision"]
# Known only after the loan plays out: never a model input.
OUTCOME_COLUMNS = ["actual_outcome", "days_to_default"]

EMPLOYMENT_GROUPS = ["employed", "self_employed", "unemployed"]
RULE_DECISIONS = ["approved", "flagged_for_review", "denied"]
OUTCOMES = ["repaid", "defaulted", "ongoing"]


def load_applications(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and check it has the expected columns and values."""
    df = pd.read_csv(path)
    expected = [ID_COLUMN, *APPLICANT_COLUMNS, *RULE_COLUMNS, *OUTCOME_COLUMNS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for column, allowed in [
        ("employment_status", EMPLOYMENT_GROUPS),
        ("rule_based_decision", RULE_DECISIONS),
        ("actual_outcome", OUTCOMES),
    ]:
        unexpected = set(df[column].unique()) - set(allowed)
        if unexpected:
            raise ValueError(f"unexpected {column} values: {sorted(unexpected)}")
    for column in ("bank_has_overdrafts", "bank_has_consistent_deposits"):
        if df[column].dtype != bool:
            raise ValueError(f"{column} should be True/False")
    return df[expected]


def split_by_outcome(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (resolved, ongoing). Ongoing loans have no outcome yet, so they can't be labeled."""
    ongoing = df["actual_outcome"].eq("ongoing")
    return df.loc[~ongoing].reset_index(drop=True), df.loc[ongoing].reset_index(drop=True)


def default_label(df: pd.DataFrame) -> np.ndarray:
    """1 if the loan defaulted, 0 if it was repaid. Only defined for resolved loans."""
    if df["actual_outcome"].eq("ongoing").any():
        raise ValueError("ongoing loans have no label; drop them first with split_by_outcome")
    return df["actual_outcome"].eq("defaulted").astype(int).to_numpy()


def train_test_indices(y: np.ndarray, test_size: float = 0.25, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split, so the training and test sets keep the same default rate."""
    rows = np.arange(len(y))
    train, test = train_test_split(rows, test_size=test_size, stratify=y, random_state=seed)
    return np.sort(train), np.sort(test)
