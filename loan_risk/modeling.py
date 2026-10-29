"""Candidate models, repeated cross-validation and the model-selection rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import SEED
from .features import CATEGORICAL, MONOTONIC

# Sections 4-6 of the notebook use the variant without employment status (V3). The fairness analysis
# (section 7) recommends adding back one "unemployed" flag (V2): equally accurate overall, and the
# only variant that stays calibrated for unemployed applicants.
WORKING_VARIANT = "V3"
RECOMMENDED_VARIANT = "V2"
# The additive model's settings, picked by the cross-validated grid search with the one-standard-error rule.
ADDITIVE_GBM_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 4, "min_samples_leaf": 50}
METRICS = ["ROC-AUC", "PR-AUC", "log-loss", "Brier"]


def make_logistic(columns=None) -> Pipeline:
    """M1: logistic regression on standardized inputs (straight-line effects on the log-odds)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def make_additive_gbm(columns, learning_rate: float = 0.05, max_leaf_nodes: int = 8, min_samples_leaf: int = 30,
                      l2_regularization: float = 1.0) -> HistGradientBoostingClassifier:
    """M2: gradient boosting where every tree uses a single feature, with fixed directions of effect.

    Because each tree splits on one feature only, the model is a sum of per-feature step functions
    (a boosted GAM), so every prediction splits exactly into per-feature contributions. Direction
    constraints (from features.MONOTONIC) stop it learning effects that contradict common sense.
    """
    columns = list(columns)
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=500,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        interaction_cst="no_interactions",
        monotonic_cst=[MONOTONIC[c] for c in columns],
        categorical_features=[c in CATEGORICAL for c in columns],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=SEED,
    )


def make_unconstrained_gbm(columns, max_leaf_nodes: int = 15, max_depth: int | None = None,
                           learning_rate: float = 0.05, min_samples_leaf: int = 30) -> HistGradientBoostingClassifier:
    """M3: ordinary gradient boosting, with interactions allowed and no direction constraints."""
    columns = list(columns)
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=500,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        categorical_features=[c in CATEGORICAL for c in columns],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=SEED,
    )


def make_production_model(columns) -> HistGradientBoostingClassifier:
    """The selected model: M2 with the tuned settings."""
    return make_additive_gbm(columns, **ADDITIVE_GBM_PARAMS)


def make_platt_calibrated(columns) -> CalibratedClassifierCV:
    """The selected model plus Platt scaling, fitted on internal cross-validated predictions.

    A single sigmoid on top keeps predictions additive in log-odds, so explanations stay exact.
    """
    return CalibratedClassifierCV(make_production_model(columns), method="sigmoid", cv=5, ensemble=False)


@dataclass
class CVResult:
    """Cross-validated performance of one candidate."""

    fold_metrics: pd.DataFrame  # one row per fold
    oof: np.ndarray  # (n_repeats, n_rows) out-of-fold predicted probabilities

    @property
    def mean_oof(self) -> np.ndarray:
        return self.oof.mean(axis=0)


def cv_splits(y, n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Repeated stratified folds as (repeat, train rows, test rows). Every candidate uses the same folds."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(k // n_splits, train, test) for k, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y))]


def _metrics(y, p) -> dict[str, float]:
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "log-loss": log_loss(y, p),
        "Brier": brier_score_loss(y, p),
    }


def cross_validate(factory, X: pd.DataFrame, y, splits) -> CVResult:
    """Fit `factory(X.columns)` on each training fold and score it on the matching held-out fold."""
    y = np.asarray(y)
    n_repeats = max(repeat for repeat, _, _ in splits) + 1
    oof = np.full((n_repeats, len(y)), np.nan)
    rows = []
    for repeat, train, test in splits:
        model = factory(X.columns).fit(X.iloc[train], y[train])
        p = model.predict_proba(X.iloc[test])[:, 1]
        oof[repeat, test] = p
        rows.append(_metrics(y[test], p))
    return CVResult(pd.DataFrame(rows), oof)


def reference_folds(score, y, splits, is_probability: bool = False) -> pd.DataFrame:
    """Fold metrics for a fixed score (the rules, or the oracle) on the same folds as the models.

    Log-loss and Brier only make sense for probabilities, so they are left empty otherwise.
    """
    score, y = np.asarray(score, dtype=float), np.asarray(y)
    rows = []
    for _, _, test in splits:
        row = {"ROC-AUC": roc_auc_score(y[test], score[test]), "PR-AUC": average_precision_score(y[test], score[test])}
        if is_probability:
            row |= {"log-loss": log_loss(y[test], score[test]), "Brier": brier_score_loss(y[test], score[test])}
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=METRICS)


def summarize(fold_metrics: dict[str, pd.DataFrame], reference_auc=None) -> pd.DataFrame:
    """Mean of each metric across folds, the standard error of log-loss, and the paired AUC gain."""
    rows = {}
    for name, folds in fold_metrics.items():
        row = {metric: folds[metric].mean() for metric in METRICS}
        row["ROC-AUC sd"] = folds["ROC-AUC"].std(ddof=1)
        row["log-loss se"] = folds["log-loss"].std(ddof=1) / np.sqrt(len(folds))
        if reference_auc is not None:
            gain = folds["ROC-AUC"].to_numpy() - np.asarray(reference_auc)
            row["AUC gain vs rules"] = gain.mean()
            row["folds better than rules"] = f"{int((gain > 0).sum())}/{len(gain)}"
        rows[name] = row
    columns = ["ROC-AUC", "ROC-AUC sd", "PR-AUC", "log-loss", "log-loss se", "Brier"]
    if reference_auc is not None:
        columns += ["AUC gain vs rules", "folds better than rules"]
    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def one_se_choice(summary: pd.DataFrame, simplest_first: list[str]) -> str:
    """Lowest mean log-loss, unless a simpler candidate is within one standard error of it."""
    best = summary["log-loss"].idxmin()
    threshold = summary.loc[best, "log-loss"] + summary.loc[best, "log-loss se"]
    for name in simplest_first:
        if summary.loc[name, "log-loss"] <= threshold:
            return name
    return best
