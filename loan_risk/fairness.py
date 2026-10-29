"""Fairness checks across employment groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def group_outcomes(y, tiers, groups, order, reference: str = "employed", p=None) -> pd.DataFrame:
    """Decisions and outcomes for each group.

    Covers what the brief asks for (approval and default rates per group) plus the error rates
    that show who bears the mistakes, and, if predicted probabilities are given, calibration.
    """
    y, tiers, groups = np.asarray(y), np.asarray(tiers), np.asarray(groups)
    rows = {}
    for group in order:
        member = groups == group
        good, bad = member & (y == 0), member & (y == 1)
        approved = member & (tiers == "approve")
        rows[group] = {
            "applicants": int(member.sum()),
            "default rate": y[member].mean(),
            "auto-approved": (tiers[member] == "approve").mean(),
            "sent to review": (tiers[member] == "review").mean(),
            "denied": (tiers[member] == "deny").mean(),
            "default rate among approved": y[approved].mean() if approved.any() else np.nan,
            "good applicants denied": (tiers[good] == "deny").mean(),
            "good applicants not auto-approved": (tiers[good] != "approve").mean(),
            "defaults auto-approved": (tiers[bad] == "approve").mean(),
        }
        if p is not None:
            rows[group]["mean predicted risk"] = np.asarray(p, dtype=float)[member].mean()
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.insert(3, "approval ratio", table["auto-approved"] / table.loc[reference, "auto-approved"])
    return table


def score_conditional_odds(y, score, groups, order, reference: str = "employed") -> pd.DataFrame:
    """Does group membership still predict default once the score is known?

    Fits logit(default) ~ score + one indicator per non-reference group. An odds ratio below 1 means
    the group defaults less often than its score implies (the score over-states its risk); above 1,
    the reverse. A fair score leaves nothing for group membership to explain.
    """
    groups = np.asarray(groups)
    levels = [g for g in order if g != reference]
    X = pd.DataFrame({"const": 1.0, "score": np.asarray(score, dtype=float)})
    for level in levels:
        X[level] = (groups == level).astype(float)
    fit = sm.Logit(np.asarray(y), X).fit(disp=0)
    ci = fit.conf_int()
    return pd.DataFrame(
        {
            "odds ratio": np.exp(fit.params[levels]),
            "ci_low": np.exp(ci.loc[levels, 0]),
            "ci_high": np.exp(ci.loc[levels, 1]),
            "p-value": fit.pvalues[levels],
        }
    )
