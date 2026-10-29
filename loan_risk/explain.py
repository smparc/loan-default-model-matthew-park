"""Exact per-applicant explanations for the additive model, in plain English."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MISMATCH, NOT_PROVIDED, VERIFIED

# Contributions smaller than this (in log-odds; about 2 percentage points of risk for a typical
# applicant) are listed as minor rather than given as reasons.
MATERIALITY = 0.10
LOAN_MENU = [300, 500, 1000, 1500, 2000, 3000, 5000]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def describe(column: str, value) -> str:
    """Plain-English description of one model input."""
    if column == "income_verification":
        return {
            VERIFIED: "Income verified: documents match stated income",
            MISMATCH: "Documented income is far below stated income",
            NOT_PROVIDED: "No income documents provided",
        }[int(value)]
    if column == "loan_to_income":
        return f"Loan is {value:.0%} of stated monthly income"
    if column == "withdrawal_to_deposit":
        return f"Withdrawals are {value:.0%} of deposits"
    if column == "bank_ending_balance":
        return f"Bank balance of ${value:,.0f}"
    if column == "bank_has_overdrafts":
        return "Overdrafts on the account" if value else "No overdrafts"
    if column == "unemployed":
        return "Unemployed" if value else "Has an income source"
    if column == "employment":
        return {0: "Employed", 1: "Self-employed", 2: "Unemployed"}[int(value)]
    return f"{column} = {value}"


class AdditiveExplainer:
    """Splits each prediction of an additive model exactly into one contribution per feature.

    For a model whose log-odds are constant + f_1(x_1) + ... + f_k(x_k), the contribution of feature
    j is f_j(x_j) minus its average over a background set of applicants. The contributions plus the
    base value (the background's average log-odds) add up to the prediction exactly, and for an
    additive model they are also the exact SHAP values.
    """

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.columns = list(background.columns)
        background = background.reset_index(drop=True)
        self._reference = background.iloc[[0]]
        self._offset = {c: self._partial(c, background[c].to_numpy()).mean() for c in self.columns}
        self.base_value = float(model.decision_function(background).mean())

    def _partial(self, column: str, values) -> np.ndarray:
        # Vary one feature on a fixed reference row; because the model is additive, the result
        # differs from that feature's own contribution only by a constant.
        frame = self._reference.loc[self._reference.index.repeat(len(values))].reset_index(drop=True)
        frame[column] = np.asarray(values)
        return self.model.decision_function(frame)

    def effect(self, column: str, values) -> np.ndarray:
        """The model's learned contribution of one feature across a range of values."""
        return self._partial(column, values) - self._offset[column]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """One column per feature: its contribution to each applicant's log-odds of default."""
        X = X[self.columns]
        return pd.DataFrame({c: self.effect(c, X[c].to_numpy()) for c in self.columns}, index=X.index)

    def explain(self, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Predicted default probability for one applicant, and the reasons behind it, largest first."""
        contributions = self.contributions(row).iloc[0].to_numpy()
        log_odds = self.base_value + contributions.sum()
        reasons = pd.DataFrame(
            {
                "factor": [describe(c, row[c].iloc[0]) for c in self.columns],
                "contribution": contributions,
                "risk if this factor were typical": sigmoid(log_odds - contributions),
            },
            index=self.columns,
        )
        reasons["effect"] = np.where(reasons["contribution"] > 0, "raises risk", "lowers risk")
        reasons["material"] = reasons["contribution"].abs() >= MATERIALITY
        order = reasons["contribution"].abs().sort_values(ascending=False).index
        return float(sigmoid(log_odds)), reasons.loc[order]


def decision_for(probability: float, approve_below: float, deny_at: float) -> str:
    """Three-tier decision from a predicted default probability."""
    if probability < approve_below:
        return "auto-approve"
    return "deny" if probability >= deny_at else "send to review"


def report(applicant_id: str, probability: float, reasons: pd.DataFrame, approve_below: float, deny_at: float,
           typical_probability: float) -> str:
    """A reviewer-facing explanation of one decision."""
    lines = [
        f"{applicant_id}: predicted default risk {probability:.0%} -> {decision_for(probability, approve_below, deny_at)}",
        f"  (applicant with typical values on every factor: {typical_probability:.0%}; "
        f"auto-approve below {approve_below:.0%}, deny at {deny_at:.0%} or more)",
    ]
    for label, side in [("Raises risk", reasons["contribution"] > 0), ("Lowers risk", reasons["contribution"] < 0)]:
        material = reasons[side & reasons["material"]]
        if len(material):
            lines.append(f"  {label}:")
            lines += [f"    - {r.factor} (if typical instead: {r['risk if this factor were typical']:.0%})"
                      for _, r in material.iterrows()]
    minor = reasons[~reasons["material"]]
    if len(minor):
        lines.append("  Minor factors: " + "; ".join(minor["factor"]))
    return "\n".join(lines)


def probability_with(explainer: AdditiveExplainer, row: pd.DataFrame, **changes) -> float:
    """Predicted default probability for this applicant with some inputs changed."""
    candidate = row[explainer.columns].copy()
    for column, value in changes.items():
        candidate[column] = value
    return float(sigmoid(explainer.model.decision_function(candidate))[0])


def largest_approvable_loan(explainer: AdditiveExplainer, row: pd.DataFrame, stated_income: float,
                            approve_below: float, menu=LOAN_MENU) -> int | None:
    """The largest loan on the product menu this applicant would be auto-approved for, if any.

    Risk can only rise with loan size relative to income (a direction constraint), so the approvable
    amounts are always the smallest ones on the menu.
    """
    approvable = [amount for amount in sorted(menu)
                  if probability_with(explainer, row, loan_to_income=amount / stated_income) < approve_below]
    return max(approvable) if approvable else None
