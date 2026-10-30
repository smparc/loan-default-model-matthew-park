# Loan Default Model: Bree ML Intern Take-Home

This repo predicts whether a loan applicant will default, compares the model with Bree's rule-based scorer, and checks fairness across employment groups. The written summary is below; every number in it comes from `analysis.ipynb` (tables in `results/`, figures in `figures/`).

**In short:**
- **Model:** an additive gradient-boosted model with direction constraints. It is interpretable by design: every prediction is a typical applicant's risk plus five exact per-feature contributions.
- **Accuracy:** AUC 0.738, against 0.706 for the rules (5×5 cross-validation over all 1,836 resolved loans; 0.761 vs 0.725 on the held-out test set). That captures about 95% of the gain any model could achieve on this data.
- **At today's approval volumes:** fewer defaults approved, more defaults caught, fewer good applicants wrongly denied, and the rules' penalty on self-employed applicants removed.

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

- **The brief's class balance is wrong.** It says ~14% of loans defaulted. The script produces 27% of all applications, which is 29.7% of the 1,836 resolved loans. The imbalance is therefore mild (70/30), so I didn't resample: SMOTE or class weights would distort the predicted probabilities, which the decisions rely on. Business costs are handled through decision thresholds instead, and I report PR-AUC alongside ROC-AUC, never accuracy.
- **Missing documented income is a signal, not noise** ([fig. 1](figures/fig1_income_verification.png)).
  - It is missing for exactly the 283 applicants who submitted no documents, and they default at 49%, against 26% when income is verified.
  - Documented ÷ stated income is either 90–105% or 21–40%, with an empty gap in between. So misrepresentation separates cleanly, and the brief's "3x+" rule of thumb would miss 29 of the 85 cases.
  - Both become one feature, `income_verification` (verified / mismatch / not provided). I didn't impute an income, because that would invent a verification that never happened.
- **Risk jumps in steps** ([fig. 2](figures/fig2_risk_steps.png)). The default rate goes from 14–19% to 39–46% once the loan passes 30% of monthly income, and also rises with overdrafts, low balances and high withdrawals. That shape is why I chose a model that learns thresholds.
- **Ongoing loans (164) are excluded from training, and that is safe here.**
  - No feature differs significantly between ongoing and resolved loans (every p ≥ 0.096), and classifiers can't tell them apart (AUC 0.52 and 0.43).
  - In production they would not be random: the newest loans are ongoing, and early defaulters are labeled first. The fix there is a 180-day performance window, since every default here happens by day 179.
  - The model expects 50 of the 164 to default; the generator's truth implies 51.
- **Excluded columns:**
  - `days_to_default` exists only for defaults, so it would leak the answer.
  - The rule score and decision are outputs of the system being replaced, and they carry its bias.
  - `bank_has_consistent_deposits` predicts nothing (30% vs 29%) but tracks employment (82% of employed vs 42% of self-employed applicants), so it is a proxy.
  - `num_documents` adds nothing once "no documents" is captured.

### 2. Model and features

- **Model: gradient boosting where every tree uses a single feature, with fixed directions of effect.**
  - Because each tree uses one feature, the model is a sum of five learned step functions, and every prediction splits exactly into per-feature contributions. For this kind of model those contributions are the exact SHAP values.
  - The constraints guarantee, for example, that a larger loan relative to income never lowers risk.
  - I used scikit-learn's `HistGradientBoostingClassifier` rather than XGBoost: it is the same family of algorithms, and XGBoost needs an extra system library on macOS.
- **How it was chosen** ([fig. 5](figures/fig5_model_comparison.png)):
  - A 25% test set was held out. On the other 1,377 loans, every candidate was scored with repeated 5-fold cross-validation, using a rule fixed in advance: lowest log-loss, unless a simpler model is within one standard error.
  - Results: rules 0.701 AUC, logistic regression 0.721, unconstrained boosting 0.702–0.718 (it overfits), the additive model 0.731, and the ceiling 0.733. The ceiling is the "oracle": the generator's true probabilities, used only as a reference.
  - So the constraints that make the model interpretable also make it more accurate. All 12 tuning settings perform the same, and no extra column helps.
- **What matters most** ([fig. 9](figures/fig9_learned_effects.png)): loan ÷ income, then income verification, overdrafts, withdrawals ÷ deposits and bank balance.
  - The model found the generator's 30% affordability threshold on its own, and close to the true effect sizes for income verification and overdrafts.
  - Weak effects are learned only roughly. In particular, an exaggerated benefit for withdrawals under 34% of deposits comes from about 80 lucky training loans. I disclose it rather than tune it away, because the fix would stop the model learning small real groups (section 6.2 of the notebook).
- **Explaining individual decisions** ([fig. 11](figures/fig11_applicant_explanations.png)). `explain_applicant(id)` gives the decision, the reasons ranked by size, the risk the applicant would have if each factor were typical, and what would change the outcome. Example: "a $300 loan instead of $1,000 would be auto-approved".

### 3. Results and trade-offs

Main comparison, at today's volumes (851 auto-approved, 720 reviewed, 265 denied), scoring all 1,836 loans out-of-fold:

| | Rules | Patched rules* | **Model** | Oracle |
|---|---|---|---|---|
| ROC-AUC | 0.706 | 0.718 | **0.738** | 0.741 |
| Defaults auto-approved | 141 | 124 | **128** | 123 |
| Defaults denied | 143 | 155 | **167** | 170 |
| Good applicants wrongly denied | 122 | 110 | **98** | 95 |
| Self-employed approval ratio | 0.60 | 0.88 | **0.89** | 0.93 |

\*The rules with two lines changed: self-employed applicants scored like employed ones, and deposit consistency ignored.

- **On the untouched test set:** AUC 0.761 vs 0.725 for the rules. The 95% CI for the gain is −0.002 to +0.073, and 97% of bootstrap resamples favour the model.
- **Robustness:** the model beats the rules in 24 of 25 cross-validation folds, and its probabilities are calibrated (29.8% predicted vs 29.6% observed on the test set; [fig. 6](figures/fig6_calibration.png)).
- **Precision, recall and error rates.** The brief's metrics need a yes/no prediction, but reviewers' decisions on flagged applications aren't in the data. So I report two conventions that bracket the truth:
  - **A** (only "deny" counts as predicting default): precision 0.54 → 0.63, recall 0.26 → 0.31, F1 0.35 → 0.41, good applicants wrongly denied 9.5% → 7.6%.
  - **B** (anything not auto-approved counts): F1 0.53 → 0.55, defaults slipping through auto-approval 25.9% → 23.5%.

  Confusion matrices are in notebook sections 3.3 and 5.2.
- **If we deployed tomorrow at today's volumes,** per 1,000 applications the model would:
  - auto-approve 7 fewer defaults;
  - catch 13 more defaults in the deny tier;
  - wrongly deny 13 fewer good applicants;
  - lend 24% less money to applicants who default.

  Both kinds of error fall at once, so there is no trade-off at today's volumes.
- **Or grow instead** ([fig. 8](figures/fig8_strategy_curve.png)). A cutoff set from predicted probabilities alone approves 54% of applicants instead of 46% at the same 16.6% default rate, and the actual default rate of that group came out at exactly 16.6%.
- **Honest caveats:**
  - The most any model could add here was about 0.035 AUC, because outcomes are noisy by design.
  - A two-line patch to the rules already captures a third of that, and matches the model at the approval cutoff. The model's edge is in the deny tier, calibrated probabilities, and fixing every weight at once.
  - In dollars, the model beats the patched rules only if repaid loans earn less than about 27% of principal, because it favours loans that are small relative to income.
- **Recommendation:** replace the rules with the model, starting in shadow mode. Ship the rules patch first if anything must change before the model is approved.

### 4. Fairness

- **The rules are biased against self-employed applicants** ([fig. 4](figures/fig4_rule_score_bias.png)).
  - Self-employed applicants default at the same rate as employed ones (29.0% vs 27.2%, p = 0.45), yet the rules auto-approve 36% vs 60%.
  - At the same rule score, self-employed applicants are a third less likely to default (odds ratio 0.67, p = 0.002). That is the 6-point employment penalty, plus a hidden second one through the deposit-consistency factor.
- **The model corrects it** ([fig. 12](figures/fig12_fairness_by_group.png)).
  - The approval ratio rises from 0.60 to 0.89, close to the 0.93 that pricing everyone at their true risk would give.
  - At the same score, employment no longer predicts default (odds ratio 0.99).
  - The model's inputs can't infer self-employment (AUC 0.51–0.54), so it can't penalize it indirectly either.
  - It also stops routinely denying creditworthy unemployed applicants: 45% wrongly denied under the rules, 12% under the model, 13% under true risk.
- **The hard case is unemployment.** Unemployed applicants really are riskier (47% default), mostly because 91% of them ask for loans above 30% of their income.
  - Dropping employment entirely (the version used in notebook sections 4–6) does not remove the information, because the inputs predict unemployment at AUC 0.82–0.86. Instead the model understates their risk by about 6 points.
  - It then approves 27 unemployed applicants where true risk justifies 10, and the 14 extra approvals default at 50%.
  - Aggregate accuracy can't see this: all three ways of handling employment score AUC 0.738–0.739. Only the group-level comparison shows it.
- **Recommendation: one narrow "unemployed" flag, with self-employed treated exactly like employed (V2).** It is equally accurate, calibrated for every group, and approves 13 unemployed applicants against 10 at true risk. It needs compliance sign-off, because source of income is a protected or sensitive ground in some jurisdictions (for example, "social condition" in Quebec). If the flag can't be used, drop employment entirely and monitor approved unemployed applicants' default rate.
- **What not to do:** separate thresholds per group (deliberate disparate treatment), or reweighting to equalize approvals, which repeats the under-pricing problem.
- **Keep measuring:** collect employment status even if it isn't scored, and track per-group approval, wrongly-denied and calibration rates monthly.
- **Who decides:** compliance and the credit-risk committee. My role is to put the trade-off in front of them in numbers.

### 5. Production risk

- **First thing likely to go wrong: the upstream income-verification data.**
  - The model's top features depend on extracting income from documents. If a biweekly pay stub is read as monthly income (about 46% of the true monthly figure), honest applicants land in the "mismatch" zone. The result would be silent mass denials, likely concentrated among hourly and gig workers.
  - Guardrails: the training data has essentially nobody between 40% and 90% documented ÷ stated income, so alert on any applicant landing there; and alert if the share with no documents drifts from 14%.
- **What to check next:**
  - Run in shadow mode alongside the rules first, then a live head-to-head with the rules as a fallback.
  - Monitor drift in inputs and scores, and early delinquency (defaults start at day 15).
  - Watch for gaming: stated income can be inflated. Loan ÷ income based on bank deposits is harder to game and costs little accuracy (log-loss +0.0035).
  - Log every decision (inputs, model version, risk, cutoffs, ranked reasons), so any decision can be re-derived for an audit.
- **The brief's open questions:**
  - *Excluding ongoing loans:* see section 1.
  - *20% more defaults caught but 10% more good applicants wrongly denied:* here that means 109 more defaults caught against 129 more good applicants denied. It is worth it if the loss on a default times 109 exceeds the margin on a good loan times 129; with equal loan sizes, roughly whenever margins are below 85% of principal. The business sets the margin and the risk appetite; compliance checks who bears the denials.
  - *"Why was this person denied?":* exact ranked reasons, directions that can never contradict common sense, a decision log, and a model that fits on one page as five curves plus a constant (fig. 9).
  - *A recession that doubles default rates:* the rules keep approving the same profiles. The model's rankings may hold, but its probabilities would be about half the real level. Shift its intercept to the new default rate using early-delinquency data, retrain as outcomes arrive, and re-run the group checks, because self-employed income may become riskier in a downturn.
- **Synthetic-data caveats:** outcomes exist even for denied applicants, which real lending never sees. Future training data will only include approved loans, so I'd keep a small randomized exploration slice. "Ongoing" was also assigned at random here.

### With more time

- Out-of-time validation on real, dated applications.
- Reject inference for applicants who were denied.
- Explainable Boosting Machines or per-feature smoothing for the weak features.
- Direct income-stability features from bank transactions, to replace the employment label.
- Cost-based thresholds using real unit economics.

### Use of AI tools

I used Claude Code (Anthropic) as a pair programmer for code, analysis and drafting. I reviewed every result, and the modeling, evaluation and fairness decisions, and the reasoning behind them, are mine.

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
