# Loan Default Model: Bree ML Intern Take-Home

This repo predicts whether a loan applicant will default, compares the model with Bree's rule-based scorer, and checks fairness across employment groups. The written summary is below. Every number in it comes from `analysis.ipynb`; the tables are in `results/` and the figures in `figures/`.

**In short**
- **Model:** an additive gradient-boosted model with direction constraints. It is interpretable by design: every prediction is a typical applicant's risk plus five exact per-feature contributions.
- **Accuracy:** AUC 0.738 against 0.706 for the rules, scoring all 1,836 resolved loans out-of-fold (0.761 vs 0.725 on the held-out test set). That is 91% of the gain any model could achieve on this data.
- **At today's approval volumes:** fewer defaults approved, more defaults caught, fewer good applicants wrongly denied, and no penalty on self-employed applicants.
- **Recommended configuration:** add one "unemployed" flag (V2), pending compliance sign-off.

## Setup and reproduction

Requires Python 3.12.

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
pytest                               # 53 tests, a few seconds
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # re-runs everything, about 3 minutes
```

`analysis.py` is the same notebook as plain text (jupytext), for readable diffs. To rebuild the notebook from it, run `jupytext --to ipynb analysis.py` before the `nbconvert` command.

`loan_applications.csv` was produced by running the brief's `generate_data.py` once, unchanged. Its expected output:

```
Generated 2000 rows
Outcomes: {'repaid': 1291, 'defaulted': 545, 'ongoing': 164}
Rule decisions: {'approved': 922, 'flagged_for_review': 800, 'denied': 278}
```

The script seeds NumPy's legacy random generator, which produces the same numbers on every NumPy version, so re-running it reproduces the committed file exactly (SHA-256 `f776877803cdc02d2b7b1d912b37c6d73fd05f4e1989472f5459d4f7ddce94e0`).

## Written summary

### 1. Data and handling

- **The brief's class balance is wrong:** 27% of applications default (29.7% of the 1,836 resolved loans), not ~14%. That's only mild imbalance, so no resampling, which would distort the probabilities. Costs go into decision thresholds; I report PR-AUC, never accuracy.
- **Missing income documents are a signal** ([fig. 1](figures/fig1_income_verification.png)): the 283 applicants without documents default at 49%, against 26% when income is verified. Documented ÷ stated income is either 90–105% or 21–40%, so misrepresentation separates cleanly; the brief's "3x+" rule misses 29 of the 85 cases. Both become one feature, `income_verification`, with no imputed income.
- **Risk jumps in steps** ([fig. 2](figures/fig2_risk_steps.png)): 14–19% default below a 30% loan-to-income ratio, 39–46% above. The model needs to learn thresholds.
- **The 164 ongoing loans are excluded from training.** Here they're indistinguishable from resolved loans (every p ≥ 0.096; classifier AUC 0.52 and 0.43). In production, use a 180-day performance window, since every default occurs by day 179. The model expects 50 of the 164 to default (truth: 51).
- **Dropped:**
  - `days_to_default` leaks the outcome.
  - The rule score and decision are the old system's outputs.
  - Deposit consistency has no signal but is a proxy for employment.
  - The document count adds nothing beyond "none".

### 2. Model and features

- **Gradient boosting where every tree uses one feature, with fixed directions of effect.** Each prediction splits exactly into five per-feature contributions (the exact SHAP values; [fig. 10](figures/fig10_contributions_beeswarm.png)), and a larger loan relative to income can never lower risk. Built with scikit-learn, since XGBoost needs an extra system library on macOS.
- **Selection rule, fixed in advance** ([fig. 5](figures/fig5_model_comparison.png)): with 25% held out, lowest cross-validated log-loss wins, unless a simpler model is within one standard error.
  - AUC: rules 0.701, logistic regression 0.721, unconstrained boosting 0.702–0.718 (it overfits), the additive model 0.731, and the oracle ceiling 0.733 (the generator's true probabilities, a reference only).
  - The constraints make the model both more interpretable and more accurate.
- **What matters most** ([fig. 9](figures/fig9_learned_effects.png)): loan ÷ income, then income verification, overdrafts, withdrawals and balance. The model found the true 30% threshold on its own. One weak-feature artifact (low withdrawals) is disclosed rather than tuned away (notebook section 6.2).
- **Explaining one applicant** ([fig. 11](figures/fig11_applicant_explanations.png)): `explain_applicant(id)` gives the ranked reasons and what would change the outcome, for example "a $300 loan instead of $1,000 would be auto-approved".

### 3. Results and trade-offs

At today's volumes (851 auto-approved, 720 reviewed, 265 denied), all 1,836 loans scored out-of-fold (notebook section 5.4):

| | Rules | Patched rules\* | Model (V3) | Recommended (V2) | Oracle |
|---|---|---|---|---|---|
| ROC-AUC | 0.706 | 0.718 | 0.738 | 0.739 | 0.741 |
| Defaults auto-approved | 141 | 124 | 128 | 127 | 123 |
| Defaults denied | 143 | 155 | 167 | 169 | 170 |
| Good applicants wrongly denied | 122 | 110 | 98 | 96 | 95 |
| Self-employed approval ratio | 0.60 | 0.88 | 0.89 | 0.90 | 0.93 |

\*Self-employed scored like employed; deposit consistency ignored. V3 (no employment input) is the working model in notebook sections 4–6; V2 adds an "unemployed" flag (section 7).

- **Held-out test set** ([fig. 7](figures/fig7_roc_test.png)): AUC 0.761 vs 0.725 (gain 95% CI −0.002 to +0.073; 97% of bootstrap resamples favour the model).
- **Robustness:** the model beats the rules in 24 of 25 folds and captures 91% of the achievable gain.
- **Calibration** ([fig. 6](figures/fig6_calibration.png)): 29.8% default predicted vs 29.6% observed on the test set.
- **The brief's metrics** need a yes/no prediction, but reviewers' decisions on flagged cases are unknown, so two conventions bracket the truth:
  - **A** (only "deny" = predicted default): precision 0.54 → 0.63, recall 0.26 → 0.31, F1 0.35 → 0.41, good applicants wrongly denied 9.5% → 7.6%.
  - **B** (not auto-approved = predicted default): F1 0.528 → 0.545, defaults slipping through 25.9% → 23.5%.

  Confusion matrices: notebook sections 3.3, 5.2 and 5.4.
- **Deployed tomorrow,** per 1,000 applications: 7 fewer defaults approved, 13 more caught, 13 fewer good applicants denied, and 24% less lent to defaulters. Both error types fall, so there's no trade-off at today's volumes.
- **Or grow instead** ([fig. 8](figures/fig8_strategy_curve.png)): approve 54% instead of 46% at the same 16.6% default rate; that group's actual rate was 16.6%.
- **Caveats:**
  - Any model could add at most about 0.035 AUC here.
  - A two-line rules patch gets a third of that and matches the model at the approval cutoff; the model's edge is the deny tier and calibrated probabilities.
  - In dollars, the model beats the patch only if repaid loans earn less than about 27% of principal.
- **Recommendation:** replace the rules with the model (V2), starting in shadow mode.

### 4. Fairness

- **The rules are biased against self-employed applicants** ([fig. 3](figures/fig3_employment.png), [fig. 4](figures/fig4_rule_score_bias.png)). They default like employed applicants (29.0% vs 27.2%, p = 0.45) but are auto-approved 36% vs 60% of the time. At the same score they're a third less likely to default (odds ratio 0.67, p = 0.002).
- **The model corrects it** ([fig. 12](figures/fig12_fairness_by_group.png)).
  - Approval ratio 0.60 → 0.89 (0.93 at true risk); odds ratio 0.99.
  - The model's inputs can't infer self-employment (AUC 0.51–0.54).
  - Creditworthy unemployed applicants wrongly denied: 45% → 12% (13% at true risk).
- **Unemployment is the hard case.** Unemployed applicants really are riskier (47% default), mostly through affordability: 91% ask for loans above 30% of their income.
  - Dropping employment (V3) doesn't remove the information, because the inputs predict unemployment at AUC 0.82–0.86. It makes the model understate their risk by about 6 points.
  - V3 approves 27 unemployed applicants where true risk justifies 10, and the 14 extra approvals default at 50%.
  - Aggregate accuracy (AUC 0.738–0.739 for every variant) can't see this.
- **Recommendation: an "unemployed" flag, with self-employed treated exactly like employed (V2).** Equally accurate, calibrated for every group, and 13 unemployed approvals against 10 at true risk.
  - It needs compliance sign-off (source of income is protected in some places, for example Quebec's "social condition"). Otherwise, drop employment and monitor.
  - Avoid per-group thresholds (deliberate disparate treatment) and reweighting (it repeats V3's under-pricing).
  - Keep monitoring by group. Compliance and the credit-risk committee decide.

### 5. Production risk

- **First likely failure: the income-verification data.** A biweekly pay stub read as monthly income (about 46% of the true figure) makes honest applicants look like misrepresentation, which would mean silent mass denials, likely among hourly and gig workers. Alert on any applicant in the empty 40–90% documented ÷ stated gap, and on drift in the 14% no-documents rate.
- **Next:**
  - Shadow mode, then a head-to-head with the rules as a fallback.
  - Monitor drift and early delinquency (defaults start at day 15).
  - Use deposit-based loan ÷ income against inflated stated income (+0.0035 log-loss).
  - Log every decision.
- **Open questions:**
  - *20% more defaults caught vs 10% more good applicants denied* means 109 vs 129 applicants here. It's worth it if the loss per default × 109 exceeds the margin per good loan × 129 (with equal loan sizes, margins below about 85% of principal). The business decides, with compliance.
  - *"Why was this person denied?":* exact ranked reasons, a decision log, and a model that fits on one page (fig. 9).
  - *Recession:* the model's probabilities would run low. Recalibrate the intercept from early delinquency and retrain; the rules can't adapt.
- **Synthetic-data caveat:** outcomes exist even for denied applicants. Real data only covers approved loans, so keep a small randomized exploration slice.

**With more time:** out-of-time validation, reject inference, Explainable Boosting Machines for smoother weak features, and income-stability features from bank transactions.

**Use of AI tools:** I used Claude Code (Anthropic) as a pair programmer for code, analysis and drafting. I reviewed every result, and the decisions and the reasoning behind them are mine.

## Code layout

| Path | Purpose |
|---|---|
| `analysis.ipynb` | The analysis, executed with outputs (start here) |
| `analysis.py` | The same notebook as plain text (jupytext), for readable diffs |
| `loan_risk/data.py` | Load and validate the CSV; the label (defaulted = 1, repaid = 0; ongoing loans have no label); stratified train/test split |
| `loan_risk/features.py` | Engineered features, the model inputs for each employment variant, and the direction constraints |
| `loan_risk/eda.py` | Default rates with confidence intervals, decile-binned rates, and the classifier check of whether two groups of rows can be told apart |
| `loan_risk/baseline.py` | The rule-based system: its score and decisions, a rebuild of the rubric, patched versions, and the oracle ceiling (diagnostic only) |
| `loan_risk/evaluation.py` | Ranking and probability metrics, confusion matrices under two conventions, matched-volume tiers, paired bootstrap, strategy curves, swap sets and the dollar view |
| `loan_risk/modeling.py` | Candidate models, repeated cross-validation, the model-selection rule, and the working and recommended variants |
| `loan_risk/explain.py` | Exact per-applicant contributions, plain-English reasons, and "what would change the decision" |
| `loan_risk/fairness.py` | Per-group approval, error and calibration tables, and the test of whether a score mis-prices a group |
| `loan_risk/plots.py` | Shared chart style (colorblind-safe palette) and figure saving |
| `figures/`, `results/` | Charts and CSV tables behind the numbers above |
| `tests/` | Data-integrity, leakage, feature, metric, model, explanation and fairness tests |
