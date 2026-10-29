"""Feature engineering: turn raw application fields into model inputs.

Every feature uses only information available when the application is made, and each one
maps to a reason a loan reviewer can read (for example "the loan is 45% of monthly income").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Documented / stated income. Verified applicants sit at 0.90-1.05 and mismatches at
# 0.21-0.40, with nothing in between (one verified applicant rounds to 0.8999), so any cutoff
# inside that gap gives the same groups. The brief's "3x+" rule of thumb would miss the 29
# mismatches whose stated income is 2.5-3x their documented income.
MISMATCH_THRESHOLD = 0.65

VERIFIED, MISMATCH, NOT_PROVIDED = 0, 1, 2
INCOME_VERIFICATION_LABELS = {VERIFIED: "verified", MISMATCH: "mismatch", NOT_PROVIDED: "not provided"}
EMPLOYMENT_CODES = {"employed": 0, "self_employed": 1, "unemployed": 2}

# Features in every model.
CORE_FEATURES = [
    "income_verification",
    "loan_to_income",
    "withdrawal_to_deposit",
    "bank_ending_balance",
    "bank_has_overdrafts",
]
# How employment enters the model; the fairness analysis compares all three.
EMPLOYMENT_VARIANTS = {"V1": ["employment"], "V2": ["unemployed"], "V3": []}
VARIANT_DESCRIPTIONS = {
    "V1": "employment status (3 groups)",
    "V2": "unemployed flag only",
    "V3": "no employment information",
}
# Extra columns tested one at a time in the feature ablation.
CANDIDATE_FEATURES = [
    "bank_has_consistent_deposits",
    "num_documents_submitted",
    "stated_monthly_income",
    "loan_amount",
    "doc_to_stated_ratio",
]
# Direction of effect enforced in the gradient-boosting model:
# +1 = can only raise risk, -1 = can only lower it, 0 = learned freely.
MONOTONIC = {
    "income_verification": 0,
    "loan_to_income": 1,
    "withdrawal_to_deposit": 1,
    "bank_ending_balance": -1,
    "bank_has_overdrafts": 1,
    "employment": 0,
    "unemployed": 0,
    "bank_has_consistent_deposits": 0,
    "num_documents_submitted": 0,
    "stated_monthly_income": 0,
    "loan_amount": 0,
    "doc_to_stated_ratio": -1,
}
CATEGORICAL = {"income_verification", "employment"}

FEATURE_LABELS = {
    "income_verification": "Income verification",
    "loan_to_income": "Loan / monthly income",
    "withdrawal_to_deposit": "Withdrawals / deposits",
    "bank_ending_balance": "Bank balance ($)",
    "bank_has_overdrafts": "Has overdrafts",
    "employment": "Employment status",
    "unemployed": "Unemployed",
    "bank_has_consistent_deposits": "Consistent deposits",
    "num_documents_submitted": "Documents submitted",
    "stated_monthly_income": "Stated monthly income ($)",
    "loan_amount": "Loan amount ($)",
    "doc_to_stated_ratio": "Documented / stated income",
}


def income_verification(df: pd.DataFrame) -> pd.Series:
    """0 = documented income matches stated, 1 = far below stated, 2 = no documents."""
    has_docs = df["documented_monthly_income"].notna()
    ratio = df["documented_monthly_income"] / df["stated_monthly_income"]
    codes = np.where(~has_docs, NOT_PROVIDED, np.where(ratio < MISMATCH_THRESHOLD, MISMATCH, VERIFIED))
    return pd.Series(codes, index=df.index, name="income_verification")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All engineered features (core, employment and ablation candidates), one row per applicant."""
    documented_ratio = df["documented_monthly_income"] / df["stated_monthly_income"]
    return pd.DataFrame(
        {
            "income_verification": income_verification(df),
            "loan_to_income": df["loan_amount"] / df["stated_monthly_income"],
            "withdrawal_to_deposit": df["monthly_withdrawals"] / df["monthly_deposits"],
            "bank_ending_balance": df["bank_ending_balance"].astype(float),
            "bank_has_overdrafts": df["bank_has_overdrafts"].astype(int),
            "employment": df["employment_status"].map(EMPLOYMENT_CODES).astype(int),
            "unemployed": df["employment_status"].eq("unemployed").astype(int),
            "bank_has_consistent_deposits": df["bank_has_consistent_deposits"].astype(int),
            "num_documents_submitted": df["num_documents_submitted"].astype(int),
            "stated_monthly_income": df["stated_monthly_income"].astype(float),
            "loan_amount": df["loan_amount"].astype(float),
            # Neutral 1.0 when there are no documents; income_verification already flags those.
            "doc_to_stated_ratio": documented_ratio.fillna(1.0),
        },
        index=df.index,
    )


def loan_to_income_alternatives(df: pd.DataFrame) -> pd.DataFrame:
    """Three ways to measure affordability; the modeling step compares them."""
    documented_or_stated = df["documented_monthly_income"].fillna(df["stated_monthly_income"])
    return pd.DataFrame(
        {
            "stated income": df["loan_amount"] / df["stated_monthly_income"],
            "documented income (stated if missing)": df["loan_amount"] / documented_or_stated,
            "bank deposits": df["loan_amount"] / df["monthly_deposits"],
        },
        index=df.index,
    )


def tree_matrix(features: pd.DataFrame, variant: str = "V3", extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Inputs for the gradient-boosting models: raw scales, categoricals as integer codes."""
    columns = CORE_FEATURES + EMPLOYMENT_VARIANTS[variant] + list(extra)
    return features[columns].copy()


def linear_matrix(features: pd.DataFrame, variant: str = "V3", loan_to_income: pd.Series | None = None) -> pd.DataFrame:
    """Inputs for logistic regression: one-hot categoricals and log loan-to-income."""
    lti = features["loan_to_income"] if loan_to_income is None else loan_to_income
    X = pd.DataFrame(
        {
            "income_mismatch": features["income_verification"].eq(MISMATCH).astype(int),
            "income_not_provided": features["income_verification"].eq(NOT_PROVIDED).astype(int),
            "log_loan_to_income": np.log(lti),
            "withdrawal_to_deposit": features["withdrawal_to_deposit"],
            "bank_ending_balance": features["bank_ending_balance"],
            "bank_has_overdrafts": features["bank_has_overdrafts"],
        },
        index=features.index,
    )
    if variant == "V1":
        X["self_employed"] = features["employment"].eq(EMPLOYMENT_CODES["self_employed"]).astype(int)
    if variant in ("V1", "V2"):
        X["unemployed"] = features["unemployed"]
    return X
