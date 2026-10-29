"""Metrics for comparing the rule-based baseline with the model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import REPO_ROOT, SEED

RESULTS_DIR = REPO_ROOT / "results"
TIERS = ["approve", "review", "deny"]
# Flagged applications go to reviewers whose decisions aren't in the data, so a yes/no prediction
# needs an assumption about them. These two conventions bracket the truth.
CONVENTIONS = {
    "A: only deny = predicted default": ("deny",),
    "B: not auto-approved = predicted default": ("deny", "review"),
}


def ranking_metrics(y, risk) -> dict[str, float]:
    """Threshold-free ranking quality. `risk` must be higher for riskier applicants."""
    return {"ROC-AUC": roc_auc_score(y, risk), "PR-AUC": average_precision_score(y, risk)}


def probability_metrics(y, p) -> dict[str, float]:
    """Quality of predicted probabilities (lower is better for both)."""
    return {"log-loss": log_loss(y, p), "Brier": brier_score_loss(y, p)}


def binary_metrics(y, flagged) -> dict[str, float]:
    """Counts and rates for a yes/no prediction, where `flagged` means "predicted to default"."""
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "FPR": fp / (fp + tn),  # good applicants flagged
        "FNR": fn / (fn + tp),  # defaults not flagged
    }


def convention_metrics(y, tiers) -> pd.DataFrame:
    """Binary metrics for a three-tier decision under both conventions."""
    tiers = np.asarray(tiers)
    rows = {name: binary_metrics(y, np.isin(tiers, flagged)) for name, flagged in CONVENTIONS.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def confusion_table(y, flagged) -> pd.DataFrame:
    """2x2 confusion matrix: actual outcome by predicted outcome."""
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    return pd.DataFrame(
        [[int((~y & ~flagged).sum()), int((~y & flagged).sum())], [int((y & ~flagged).sum()), int((y & flagged).sum())]],
        index=pd.Index(["repaid", "defaulted"], name="actual"),
        columns=["predicted repay", "predicted default"],
    )


def tier_table(y, tiers) -> pd.DataFrame:
    """How many applicants land in each tier, and how many of them defaulted."""
    frame = pd.DataFrame({"tier": np.asarray(tiers), "defaulted": np.asarray(y)})
    table = frame.groupby("tier")["defaulted"].agg(applicants="size", defaulted="sum").reindex(TIERS, fill_value=0)
    table["repaid"] = table["applicants"] - table["defaulted"]
    table["share of applicants"] = table["applicants"] / table["applicants"].sum()
    table["default rate"] = table["defaulted"] / table["applicants"]
    table.index.name = None
    return table[["applicants", "share of applicants", "repaid", "defaulted", "default rate"]]


def matched_tiers(risk, n_approve: int, n_deny: int) -> np.ndarray:
    """Approve the n_approve lowest-risk applicants, deny the n_deny riskiest, review the rest.

    Matching another system's tier sizes uses no outcome labels, so it can't flatter the model.
    Ties are broken by row order.
    """
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="mergesort")
    tiers = np.full(len(risk), "review", dtype=object)
    tiers[order[:n_approve]] = "approve"
    tiers[order[len(risk) - n_deny:]] = "deny"
    return tiers


def paired_bootstrap(y, score_a, score_b, metric, n_boot: int = 2000, seed: int = SEED) -> dict[str, float]:
    """95% CI for metric(y, score_b) - metric(y, score_a), resampling applicants with replacement."""
    y, score_a, score_b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        if y[rows].min() == y[rows].max():
            continue
        diffs.append(metric(y[rows], score_b[rows]) - metric(y[rows], score_a[rows]))
    diffs = np.asarray(diffs)
    return {
        "difference": metric(y, score_b) - metric(y, score_a),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "share of resamples where B is better": float((diffs > 0).mean()),
    }


def strategy_curve(y, risk, shares) -> np.ndarray:
    """Default rate among the lowest-risk `share` of applicants, for each share in `shares`."""
    y = np.asarray(y)
    order = np.argsort(np.asarray(risk, dtype=float), kind="mergesort")
    cumulative_rate = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    counts = np.clip(np.round(np.asarray(shares) * len(y)).astype(int), 1, len(y))
    return cumulative_rate[counts - 1]


def bootstrap_strategy_band(y, risk, shares, n_boot: int = 500, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """95% band for the strategy curve, resampling applicants with replacement."""
    y, risk = np.asarray(y), np.asarray(risk, dtype=float)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        curves.append(strategy_curve(y[rows], risk[rows], shares))
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def approvals_at_predicted_rate(p, target_rate: float) -> int:
    """How many applicants can be approved, lowest predicted risk first, while their average
    predicted default probability stays at or below `target_rate`.

    The cutoff is set from predictions only, as it would be in production, so checking the actual
    default rate of those applicants afterwards is a fair test.
    """
    expected = np.cumsum(np.sort(np.asarray(p, dtype=float))) / np.arange(1, len(p) + 1)
    return int(np.searchsorted(expected, target_rate, side="right"))


def swap_sets(y, old_tiers, new_tiers, groups, order) -> pd.DataFrame:
    """Who a new system approves that the old one didn't, and the reverse, with their default rates."""
    y, groups = np.asarray(y), np.asarray(groups)
    old_tiers, new_tiers = np.asarray(old_tiers), np.asarray(new_tiers)
    masks = {
        "newly approved": (new_tiers == "approve") & (old_tiers != "approve"),
        "no longer approved": (new_tiers != "approve") & (old_tiers == "approve"),
    }
    rows = {}
    for name, mask in masks.items():
        rows[name] = {"applicants": int(mask.sum()), "default rate": float(y[mask].mean()) if mask.any() else float("nan")}
        rows[name] |= {group: int((groups[mask] == group).sum()) for group in order}
    return pd.DataFrame.from_dict(rows, orient="index")


def dollars_approved(y, tiers, amounts) -> dict[str, float]:
    """Total principal auto-approved, split by whether the loan defaulted."""
    y, amounts = np.asarray(y), np.asarray(amounts, dtype=float)
    approved = np.asarray(tiers) == "approve"
    return {
        "$ lent to defaulters": float(amounts[approved & (y == 1)].sum()),
        "$ lent to repayers": float(amounts[approved & (y == 0)].sum()),
    }


def save_table(table: pd.DataFrame, name: str) -> None:
    """Save a results table to results/<name>.csv so the README's numbers can be checked."""
    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(4).to_csv(RESULTS_DIR / f"{name}.csv")
