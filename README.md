# Loan Default Model: Bree ML Intern Take-Home

This repo builds a model that predicts whether a loan applicant will default, compares it with Bree's current rule-based scorer, and checks fairness across employment groups. The written summary will be added once the analysis is complete.

## Setup

Requires Python 3.12.

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

## Data

`loan_applications.csv` was produced by running the brief's script once, unchanged:

```bash
python generate_data.py
```

Expected output:

```
Generated 2000 rows
Outcomes: {'repaid': 1291, 'defaulted': 545, 'ongoing': 164}
Rule decisions: {'approved': 922, 'flagged_for_review': 800, 'denied': 278}
```

The script seeds NumPy's legacy global random generator (seed 42), which produces the same numbers on every NumPy version. Re-running it therefore reproduces the committed file exactly (SHA-256 `f776877803cdc02d2b7b1d912b37c6d73fd05f4e1989472f5459d4f7ddce94e0`).

**Note:** the brief describes the outcomes as ~78% repaid / ~14% defaulted / ~8% ongoing. The script actually produces 64.5% / 27.3% / 8.2%. All analysis uses the data as generated.

## Code layout

| Path | Purpose |
|---|---|
| `analysis.ipynb` | The analysis, executed with outputs (start here) |
| `analysis.py` | The same notebook as plain text (jupytext), for readable diffs |
| `loan_risk/data.py` | Load and validate the CSV; define the label (defaulted = 1, repaid = 0; ongoing loans have no label); stratified train/test split |
| `loan_risk/features.py` | Engineered features, the model inputs for each employment variant, and the direction constraints |
| `loan_risk/eda.py` | Default rates with confidence intervals, decile-binned rates, and the ongoing-vs-resolved check |
| `loan_risk/baseline.py` | The rule-based system: its score and decisions, a rebuild of the rubric, patched versions, and the oracle ceiling (diagnostic only) |
| `loan_risk/evaluation.py` | Ranking and probability metrics, confusion matrices under two conventions, matched-volume tiers, paired bootstrap, strategy curves, swap sets and the dollar view |
| `loan_risk/fairness.py` | Test of whether a score mis-prices an employment group |
| `loan_risk/modeling.py` | The candidate models (logistic regression, additive direction-constrained boosting, unconstrained boosting), repeated cross-validation, and the model-selection rule |
| `loan_risk/explain.py` | Exact per-applicant contributions (equal to SHAP values for this additive model), plain-English reasons, and "what would change the decision" |
| `loan_risk/plots.py` | Shared chart style (colorblind-safe palette) and figure saving |
| `figures/` | Charts referenced in this README |
| `results/` | CSV tables behind the numbers quoted in this README |
| `tests/` | Data-integrity, leakage, feature, metric and baseline tests |

To regenerate the executed notebook from its source:

```bash
jupytext --to ipynb analysis.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

## Tests

```bash
pytest
```
