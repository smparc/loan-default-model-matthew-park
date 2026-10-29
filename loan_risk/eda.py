"""Helpers for the data audit and exploratory analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from .data import SEED


def _with_rates(table: pd.DataFrame) -> pd.DataFrame:
    low, high = proportion_confint(table["defaults"], table["n"], method="wilson")
    table["rate"] = table["defaults"] / table["n"]
    table["ci_low"], table["ci_high"] = low, high
    return table


def rate_table(y, groups, order=None) -> pd.DataFrame:
    """Default rate per group with a 95% Wilson confidence interval."""
    frame = pd.DataFrame({"group": np.asarray(groups), "y": np.asarray(y)})
    table = frame.groupby("group", sort=False)["y"].agg(n="size", defaults="sum")
    if order is not None:
        table = table.reindex(order)
    table.index.name = None
    return _with_rates(table)


def binned_rate(x, y, n_bins: int = 10) -> pd.DataFrame:
    """Default rate within equal-sized (quantile) bins of x, so no cutoffs are hand-picked."""
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y)})
    frame["bin"] = pd.qcut(frame["x"], q=n_bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        x_low=("x", "min"), x_high=("x", "max"), x_median=("x", "median"), n=("y", "size"), defaults=("y", "sum")
    )
    return _with_rates(table.reset_index(drop=True))


def adversarial_auc(X: pd.DataFrame, is_target, seed: int = SEED) -> dict[str, float]:
    """How well can a classifier tell two sets of rows apart? 0.5 means not at all.

    Used to check whether ongoing loans look different from resolved ones.
    """
    target = np.asarray(is_target).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "gradient boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    return {
        name: roc_auc_score(target, cross_val_predict(model, X, target, cv=cv, method="predict_proba")[:, 1])
        for name, model in models.items()
    }


def format_rates(table: pd.DataFrame, columns=("rate", "ci_low", "ci_high")) -> pd.DataFrame:
    """Copy of a rate table with rates shown as percentages, for display."""
    out = table.copy()
    present = [c for c in columns if c in out.columns]
    for column in present:
        out[column] = (100 * out[column]).round(1)
    return out.rename(columns={c: f"{c} (%)" for c in present})
