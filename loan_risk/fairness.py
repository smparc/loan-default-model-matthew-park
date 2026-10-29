"""Fairness checks across employment groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
