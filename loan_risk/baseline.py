"""The current rule-based system: its score and decisions, simple patches, and a reference ceiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import VERIFIED, income_verification

APPROVE_AT, REVIEW_AT = 75, 50
WEIGHTS = {
    "income_verification": 0.30,
    "income_level": 0.25,
    "account_stability": 0.20,
    "employment": 0.15,
    "debt_to_income": 0.10,
}
EMPLOYMENT_POINTS = {"employed": 100, "self_employed": 60, "unemployed": 20}
CONSISTENT_DEPOSIT_POINTS = 30
DECISION_TO_TIER = {"approved": "approve", "flagged_for_review": "review", "denied": "deny"}


def rule_tiers(df: pd.DataFrame) -> np.ndarray:
    """The rules' actual decision for each applicant, as approve / review / deny."""
    return df["rule_based_decision"].map(DECISION_TO_TIER).to_numpy()


def tiers_from_score(score) -> np.ndarray:
    """Apply the rules' thresholds: approve at 75+, review at 50-74.9, deny below 50."""
    score = np.asarray(score, dtype=float)
    return np.where(score >= APPROVE_AT, "approve", np.where(score >= REVIEW_AT, "review", "deny")).astype(object)


def rule_components(df: pd.DataFrame) -> pd.DataFrame:
    """The rubric's five factors (0-100 points each), rebuilt from the CSV columns."""
    stated, documented, loan = df["stated_monthly_income"], df["documented_monthly_income"], df["loan_amount"]
    within_10pct = ((stated - documented).abs() / stated) <= 0.10
    return pd.DataFrame(
        {
            "income_verification": np.where(income_verification(df) != VERIFIED, 0, np.where(within_10pct, 95, 40)),
            "income_level": np.where(stated >= 3 * loan, 90, np.where(stated >= 2 * loan, 45, 0)),
            "account_stability": np.clip(
                np.where(df["bank_ending_balance"] > 500, 40, 10)
                + np.where(df["bank_has_overdrafts"], 0, 30)
                + np.where(df["bank_has_consistent_deposits"], CONSISTENT_DEPOSIT_POINTS, 0),
                0,
                100,
            ),
            "employment": df["employment_status"].map(EMPLOYMENT_POINTS),
            "debt_to_income": np.clip(100 - df["monthly_withdrawals"] / df["monthly_deposits"] * 100, 0, 100),
        },
        index=df.index,
    )


def rebuild_rule_score(df: pd.DataFrame) -> pd.Series:
    """The weighted rubric score, rounded to 0.1 like the original."""
    components = rule_components(df)
    return sum(components[factor] * weight for factor, weight in WEIGHTS.items()).round(1)


def adjusted_rule_score(df: pd.DataFrame, self_employed_points: float = 60,
                        ignore_deposit_consistency: bool = False) -> pd.Series:
    """The original score with one or both employment-linked rubric factors patched.

    - `self_employed_points`: employment points for self-employed applicants (the rubric gives 60,
      employed applicants get 100).
    - `ignore_deposit_consistency`: give everyone the 30 "consistent deposits" points, so that
      factor no longer affects the score (nobody's score goes down).

    Applied as an exact adjustment to `rule_based_score`, so the only difference from the real rules
    is the change being tested.
    """
    score = df["rule_based_score"].astype(float)
    score = score + WEIGHTS["employment"] * (self_employed_points - EMPLOYMENT_POINTS["self_employed"]) * (
        df["employment_status"].eq("self_employed")
    )
    if ignore_deposit_consistency:
        score = score + WEIGHTS["account_stability"] * CONSISTENT_DEPOSIT_POINTS * ~df["bank_has_consistent_deposits"]
    return score.round(1)


def oracle_log_odds_terms(df: pd.DataFrame) -> pd.DataFrame:
    """The generator's true effect of each factor on the log-odds of default. Diagnostic only.

    Summed and shifted by -2.25, these give the oracle's log-odds. Used only to check what the model
    learned against the truth.
    """
    return pd.DataFrame(
        {
            "income_verification": 1.25 * (income_verification(df) != VERIFIED),
            "loan_to_income": 1.25 * (df["loan_amount"] > df["stated_monthly_income"] * 0.3),
            "bank_has_overdrafts": 1.0 * df["bank_has_overdrafts"],
            "unemployed": 0.5 * df["employment_status"].eq("unemployed"),
            "bank_ending_balance": 0.5 * (df["bank_ending_balance"] < 500),
            "withdrawal_to_deposit": 0.5 * (df["monthly_withdrawals"] / np.maximum(df["monthly_deposits"], 1) > 0.8),
        },
        index=df.index,
    ).astype(float)


def oracle_default_probability(df: pd.DataFrame) -> np.ndarray:
    """The generator's true default probability, recomputed from the CSV. Diagnostic only.

    It relies on knowing how the synthetic data was made, so it is never used to train, tune or
    select anything. It only shows the best ranking any model could reach on this data.
    """
    risk = (
        0.25 * (income_verification(df) != VERIFIED)
        + 0.25 * (df["loan_amount"] > df["stated_monthly_income"] * 0.3)
        + 0.20 * df["bank_has_overdrafts"]
        + 0.10 * df["employment_status"].eq("unemployed")
        + 0.10 * (df["bank_ending_balance"] < 500)
        + 0.10 * (df["monthly_withdrawals"] / np.maximum(df["monthly_deposits"], 1) > 0.8)
    )
    return (1 / (1 + np.exp(-5 * (risk.astype(float) - 0.45)))).to_numpy()
