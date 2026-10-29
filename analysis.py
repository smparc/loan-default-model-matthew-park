# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loan default model: analysis
#
# This notebook predicts whether a loan applicant will default, compares the model with Bree's
# rule-based scorer, and checks fairness across employment groups. Reusable code lives in the
# `loan_risk/` package; this notebook runs it and explains the results.
#
# **Contents**
# 1. Data audit: what's in the data, what's missing, and where it differs from the brief
# 2. Exploratory analysis: which signals predict default, and how
# 3. The rule-based baseline: how well it ranks risk, what its decisions cost, and whether it is fair
# 4. Choosing the model: three candidates compared by cross-validation, tuning, features, calibration
# 5. The model against the baseline: the untouched test set, robustness, and what deployment would change
# 6. Explaining decisions: exact per-applicant reasons, what the model learned, what would change a decision
# 7. Fairness across employment groups: who gets approved, who bears the errors, and what to do about it
#
# To re-run: `jupytext --to ipynb analysis.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`

# %%
import warnings

# shap imports tqdm, which warns when Jupyter progress widgets aren't installed; the warning is noise here.
warnings.filterwarnings("ignore", message="IProgress not found")

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from loan_risk import plots
from loan_risk.baseline import (
    adjusted_rule_score,
    oracle_default_probability,
    oracle_log_odds_terms,
    rebuild_rule_score,
    rule_components,
    rule_tiers,
    tiers_from_score,
)
from loan_risk.data import EMPLOYMENT_GROUPS, default_label, load_applications, split_by_outcome, train_test_indices
from loan_risk.eda import adversarial_auc, binned_rate, format_rates, rate_table
from loan_risk.evaluation import (
    CONVENTIONS,
    approvals_at_predicted_rate,
    binary_metrics,
    bootstrap_strategy_band,
    confusion_table,
    convention_metrics,
    dollars_approved,
    matched_tiers,
    paired_bootstrap,
    probability_metrics,
    ranking_metrics,
    save_table,
    strategy_curve,
    swap_sets,
    tier_table,
)
from loan_risk.explain import (
    AdditiveExplainer,
    decision_for,
    largest_approvable_loan,
    probability_with,
    report,
    sigmoid,
)
from loan_risk.fairness import group_outcomes, score_conditional_odds
from loan_risk.features import (
    CANDIDATE_FEATURES,
    EMPLOYMENT_VARIANTS,
    FEATURE_LABELS,
    INCOME_VERIFICATION_LABELS,
    MISMATCH,
    MISMATCH_THRESHOLD,
    NOT_PROVIDED,
    VARIANT_DESCRIPTIONS,
    VERIFIED,
    build_features,
    linear_matrix,
    loan_to_income_alternatives,
    tree_matrix,
)
from loan_risk.modeling import (
    ADDITIVE_GBM_PARAMS,
    WORKING_VARIANT,
    cross_validate,
    cv_splits,
    make_additive_gbm,
    make_logistic,
    make_platt_calibrated,
    make_production_model,
    make_unconstrained_gbm,
    one_se_choice,
    reference_folds,
    summarize,
)

plots.apply_style()
pd.set_option("display.precision", 4)
GROUP_LABELS = {"employed": "Employed", "self_employed": "Self-employed", "unemployed": "Unemployed"}

applications = load_applications()
resolved, ongoing = split_by_outcome(applications)
y = default_label(resolved)
features = build_features(resolved)
print(f"{len(applications):,} applications: {len(resolved):,} resolved "
      f"({y.sum()} defaults = {y.mean():.1%}), {len(ongoing)} ongoing")

# %% [markdown]
# ## 1. Data audit
#
# ### 1.1 Columns and missing values
#
# Fifteen columns. Only two have gaps, and both gaps have a known cause (sections 1.3 and 1.5).

# %%
audit = pd.DataFrame(
    {
        "type": applications.dtypes.astype(str),
        "missing": applications.isna().sum(),
        "first row": applications.iloc[0].astype(str),
    }
)
display(audit)

# %% [markdown]
# ### 1.2 What the brief says vs what the data shows
#
# I checked each data characteristic the brief lists against the generated file.

# %%
outcome_share = applications["actual_outcome"].value_counts(normalize=True)
doc_ratio = applications["documented_monthly_income"] / applications["stated_monthly_income"]
stated_multiple = 1 / doc_ratio
mismatch = doc_ratio < MISMATCH_THRESHOLD
three_x = int((stated_multiple >= 3).sum())
employment_share = applications["employment_status"].value_counts(normalize=True)
employment_rates = rate_table(y, resolved["employment_status"], EMPLOYMENT_GROUPS)
_, p_self = proportions_ztest(
    employment_rates.loc[["self_employed", "employed"], "defaults"],
    employment_rates.loc[["self_employed", "employed"], "n"],
)

brief_check = pd.DataFrame(
    [
        ("Outcomes: repaid / defaulted / ongoing", "~78% / ~14% / ~8%",
         " / ".join(f"{outcome_share[o]:.1%}" for o in ["repaid", "defaulted", "ongoing"]),
         "No: about twice as many defaults"),
        ("Documented income missing", "~15%",
         f"{applications['documented_monthly_income'].isna().mean():.1%}", "Yes"),
        ("Misrepresentation (stated income 3x+ documented)", "~5% of rows",
         f"{mismatch.sum()} clear mismatches ({mismatch.mean():.1%}); only {three_x} are 3x+",
         f"Partly: a 3x cutoff misses {mismatch.sum() - three_x} of {mismatch.sum()}"),
        ("Employment: employed / self-employed / unemployed", "three values",
         " / ".join(f"{employment_share[g]:.0%}" for g in EMPLOYMENT_GROUPS), "Yes"),
        ("Self-employed default about as often as employed", "roughly the same",
         f"{employment_rates.loc['self_employed', 'rate']:.1%} vs "
         f"{employment_rates.loc['employed', 'rate']:.1%} (p = {p_self:.2f})", "Yes"),
    ],
    columns=["Claim", "Brief", "Data", "Holds?"],
).set_index("Claim")
display(brief_check)

# %% [markdown]
# **Two discrepancies matter for modeling.**
# - **The default rate is about double what the brief says**: 27% of all applications, 30% of
#   resolved ones. The classes are only mildly imbalanced, so there's no case for resampling.
# - **The brief's "3x+" description of misrepresentation is too narrow.** Mismatches are better
#   found by the empty gap in documented ÷ stated income (section 2.1); a 3x cutoff would miss a
#   third of them.

# %% [markdown]
# ### 1.3 Missing documented income is structural, not random

# %%
display(pd.crosstab(
    applications["num_documents_submitted"],
    applications["documented_monthly_income"].isna().map({True: "missing", False: "present"}),
    margins=True,
))

# %% [markdown]
# Documented income is missing **exactly** when an applicant submitted no documents (283 people).
# So the gap is a signal in its own right: these applicants default far more often (section 2.1).
# Filling in a typical income (mean or median imputation) would invent a verified income and erase
# that signal, so the model gets an explicit "not provided" category instead.

# %% [markdown]
# ### 1.4 Floors and value sets

# %%
at_floor = applications["stated_monthly_income"] == 500
print(f"Stated income at exactly $500 (the generator's floor): {at_floor.sum()} applicants, "
      f"{applications.loc[at_floor, 'employment_status'].eq('unemployed').mean():.0%} of them unemployed")
print(f"Bank balance at exactly $0 (floor): {(applications['bank_ending_balance'] == 0).sum()} applicants")
print(f"Loan amounts take {applications['loan_amount'].nunique()} values: "
      f"{sorted(applications['loan_amount'].unique().tolist())}")

# %% [markdown]
# Both floors are where the generator clipped values, so the true figure below them is unknown (a
# $0 balance may hide an overdrawn account). Tree-based models handle this without special care.
# Loan amounts come from a menu of seven sizes, so loan ÷ income varies mostly through income.

# %% [markdown]
# ### 1.5 `days_to_default` leaks the outcome

# %%
defaulted = applications[applications["actual_outcome"] == "defaulted"]
defaulted_features = build_features(defaulted)
print(f"days_to_default is filled in for {applications['days_to_default'].notna().sum()} rows, all of them "
      f"defaults; range {defaulted['days_to_default'].min():.0f}-{defaulted['days_to_default'].max():.0f} days")
candidates = {
    "loan / income": defaulted_features["loan_to_income"],
    "withdrawals / deposits": defaulted_features["withdrawal_to_deposit"],
    "bank balance": defaulted["bank_ending_balance"],
    "rule-based score": defaulted["rule_based_score"],
}
display(pd.DataFrame({
    "Spearman correlation with days_to_default": {
        name: stats.spearmanr(values, defaulted["days_to_default"]).statistic for name, values in candidates.items()
    }
}).round(3))

# %% [markdown]
# It only exists after a loan has defaulted, so as a feature it would hand the model the answer: it is
# excluded. It also shows no pattern (spread evenly over 15-179 days and unrelated to any feature), so
# there's nothing to learn about *when* defaults happen. One fact is still useful: **every default
# happens by day 179**, which matters for the ongoing loans below.

# %% [markdown]
# ### 1.6 Ongoing loans look like everyone else

# %%
all_features = build_features(applications)
is_ongoing = applications["actual_outcome"].eq("ongoing")
profile = pd.DataFrame(
    {
        "stated monthly income ($)": applications["stated_monthly_income"],
        "loan amount ($)": applications["loan_amount"],
        "loan / income": all_features["loan_to_income"],
        "bank balance ($)": applications["bank_ending_balance"],
        "has overdrafts (share)": applications["bank_has_overdrafts"].astype(float),
        "no income documents (share)": applications["documented_monthly_income"].isna().astype(float),
        "unemployed (share)": all_features["unemployed"].astype(float),
        "rule-based score": applications["rule_based_score"],
    }
)
ongoing_vs_resolved = profile.groupby(is_ongoing.map({False: "resolved", True: "ongoing"})).mean().T[["resolved", "ongoing"]]
ongoing_vs_resolved["p-value (Mann-Whitney)"] = [
    stats.mannwhitneyu(profile.loc[is_ongoing, c], profile.loc[~is_ongoing, c]).pvalue for c in profile.columns
]
display(ongoing_vs_resolved.round(3))
for name, auc in adversarial_auc(all_features, is_ongoing).items():
    print(f"Can {name} tell ongoing loans from resolved ones? Cross-validated AUC = {auc:.2f} (0.5 = chance)")

# %% [markdown]
# The averages differ a little (for example, 21% vs 16% have overdrafts), but no difference is
# statistically significant for a group of 164, and neither classifier can tell the groups apart.
# With no real signal, cross-validated AUC scatters around 0.5 and can land below it, as it does for
# gradient boosting here. In this dataset "ongoing" was assigned at random, so **leaving these 164
# loans out of training introduces no bias here.**
#
# In a real lending book it would. Ongoing loans are the newest ones, and among new loans the fast
# defaulters are already labeled while slow outcomes are not. Dropping them over-weights early
# defaults and older loans. The production fix is a fixed performance window: label a loan only once
# it is 180 days old, since no default here happens after day 179. The alternatives are worse:
# calling ongoing loans "repaid" would mislabel the roughly 30% that will default, and a survival
# model needs each loan's time on book, which the data doesn't record. The final model scores these
# 164 loans at the end.

# %% [markdown]
# ## 2. Exploratory analysis
#
# ### 2.1 Income verification: a clean gap, and the strongest single signal

# %%
ratio = doc_ratio.dropna()
status = features["income_verification"].map(INCOME_VERIFICATION_LABELS)
verification_rates = rate_table(y, status, ["verified", "mismatch", "not provided"])
n_mismatch = int((ratio < MISMATCH_THRESHOLD).sum())
missed_by_3x = int(((ratio > 1 / 3) & (ratio < MISMATCH_THRESHOLD)).sum())

fig, (ax_hist, ax_rate) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.8, 1]})
counts, _, _ = ax_hist.hist(ratio, bins=np.arange(0.15, 1.1251, 0.025), color=plots.BLUE,
                            edgecolor=plots.SURFACE, linewidth=1, zorder=2)
top = counts.max()
ax_hist.axvline(MISMATCH_THRESHOLD, color=plots.INK_SECONDARY, linewidth=1, zorder=3)
ax_hist.axvline(1 / 3, color=plots.MUTED, linewidth=1, zorder=3)
ax_hist.text(MISMATCH_THRESHOLD + 0.012, top * 0.97, "cutoff used: 65%", color=plots.INK_SECONDARY,
             fontsize=8.5, va="top")
ax_hist.text(1 / 3 - 0.012, top * 0.97, "brief's 3x rule: 33%", color=plots.MUTED, fontsize=8.5,
             va="top", ha="right")
ax_hist.annotate(f"{n_mismatch} mismatches\n({missed_by_3x} missed by\nthe 3x rule)", xy=(0.38, top * 0.05),
                 xytext=(0.42, top * 0.24), fontsize=8.5, color=plots.INK, ha="left",
                 arrowprops={"arrowstyle": "-", "color": plots.INK_SECONDARY, "linewidth": 0.8})
ax_hist.text(0.87, top * 0.55, f"{len(ratio) - n_mismatch:,} verified", ha="right", fontsize=8.5, color=plots.INK)
ax_hist.set_title("Documented ÷ stated income (applicants with documents)")
ax_hist.set_xlabel("documented income as a share of stated income")
ax_hist.set_ylabel("applicants")
plots.percent_axis(ax_hist, "x")
plots.rate_bars(ax_rate, verification_rates, labels=["Verified", "Mismatch", "Not provided"])
ax_rate.set_title("Default rate by verification status")
ax_rate.set_ylim(0, 0.7)
fig.tight_layout()
plots.save(fig, "fig1_income_verification")
plt.show()
display(format_rates(verification_rates))

# %% [markdown]
# Documented income either matches stated income (90-105%) or falls far below it (21-40%), with
# nothing in between. So any cutoff inside the gap gives the same split; I use 65%. The brief's "3x"
# description would miss the 29 mismatches between 33% and 40%.
#
# Applicants whose income can't be verified default far more often: 44% with a mismatch and 49% with
# no documents, against 26% when income is verified. The two failure modes behave alike, which is why
# they share one feature with three levels rather than an imputed income.

# %% [markdown]
# ### 2.2 Risk jumps in steps
#
# Each panel splits applicants into ten equal-sized groups by one feature (no hand-picked cutoffs) and
# shows each group's default rate. The shaded band is the 95% confidence interval; the gray line is
# the overall default rate.

# %%
drivers = {
    "loan_to_income": "Loan ÷ monthly income",
    "bank_ending_balance": "Bank balance",
    "withdrawal_to_deposit": "Withdrawals ÷ deposits",
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
binned = {}
for ax, (column, title) in zip(axes, drivers.items()):
    table = binned_rate(features[column], y, n_bins=10)
    binned[title] = table
    ax.axhline(y.mean(), color=plots.MUTED, linewidth=1, zorder=1)
    plots.rate_line(ax, table)
    ax.set_title(title)
axes[0].set_xscale("log")
axes[0].set_xticks([0.1, 0.2, 0.3, 0.5, 1, 2], ["10%", "20%", "30%", "50%", "100%", "200%"])
axes[0].minorticks_off()
plots.dollar_axis(axes[1])
plots.percent_axis(axes[2], "x")
axes[0].set_ylabel("default rate")
axes[0].set_ylim(0, 0.75)
axes[0].text(axes[0].get_xlim()[0], y.mean() + 0.012, f" overall {y.mean():.0%}", color=plots.MUTED, fontsize=8.5)
fig.tight_layout()
plots.save(fig, "fig2_risk_steps")
plt.show()
display(pd.concat({
    title: format_rates(table[["x_low", "x_high", "n", "rate", "ci_low", "ci_high"]]) for title, table in binned.items()
}).round(3))

# %% [markdown]
# - **Loan ÷ income** has a sharp step: 14-19% default below 30% of monthly income, 39-46% above it,
#   and little change after that.
# - **Bank balance** matters mostly at the low end: 44% default in the lowest tenth (under about $150)
#   and 33-39% up to about $1,200, against roughly 25% above. This pattern is noisier.
# - **Withdrawals ÷ deposits** is flat until about 80%, then rises to 36-40%.
#
# A straight-line (linear) term gets step shapes like these wrong. Tree-based models learn them
# directly from the data, without anyone choosing the cutoffs by hand.

# %% [markdown]
# ### 2.3 Yes/no signals, and two columns that carry no signal

# %%
def yes_no(series: pd.Series) -> pd.Series:
    return series.map({True: "yes", False: "no"})


binary_rates = pd.concat({
    "Has overdrafts": rate_table(y, yes_no(resolved["bank_has_overdrafts"]), ["no", "yes"]),
    "Documents submitted": rate_table(y, resolved["num_documents_submitted"].astype(str), ["0", "1", "2"]),
    "Consistent deposits": rate_table(y, yes_no(resolved["bank_has_consistent_deposits"]), ["no", "yes"]),
})
display(format_rates(binary_rates))

with_label = resolved.assign(defaulted=y)
consistent = with_label["bank_has_consistent_deposits"]
proxy = pd.DataFrame({
    "share with consistent deposits": with_label.groupby("employment_status")["bank_has_consistent_deposits"].mean(),
    "default rate if consistent": with_label[consistent].groupby("employment_status")["defaulted"].mean(),
    "default rate if not": with_label[~consistent].groupby("employment_status")["defaulted"].mean(),
}).reindex(EMPLOYMENT_GROUPS)
display((100 * proxy).round(1).rename(index=GROUP_LABELS).add_suffix(" (%)"))

# %% [markdown]
# **Overdrafts** nearly double the default rate (26% to 49%). Two columns carry no signal:
# - **Number of documents** matters only as zero versus any: one and two documents default at 25%
#   and 28%.
# - **Deposit consistency** makes no meaningful difference, overall (30% vs 29%) or within any
#   employment group. Yet it tracks employment closely: 82% of employed applicants have consistent
#   deposits, against 42% of self-employed and 36% of unemployed. A model using it would learn
#   nothing about risk but could penalize self-employment indirectly, so it is left out and
#   revisited in the fairness analysis.

# %% [markdown]
# ### 2.4 Employment: similar risk, very different treatment

# %%
decision_mix = pd.crosstab(
    applications["employment_status"], applications["rule_based_decision"], normalize="index"
).reindex(EMPLOYMENT_GROUPS)
labels = [GROUP_LABELS[g] for g in EMPLOYMENT_GROUPS]
fig, (ax_default, ax_approve) = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
plots.rate_bars(ax_default, employment_rates, labels)
ax_default.set_title("Default rate (resolved loans)")
plots.rate_bars(ax_approve, decision_mix[["approved"]].rename(columns={"approved": "rate"}), labels,
                color=plots.RULES_COLOR)
ax_approve.set_title("Auto-approved by the rules (all applicants)")
ax_default.set_ylim(0, 0.7)
fig.tight_layout()
plots.save(fig, "fig3_employment")
plt.show()

summary = format_rates(employment_rates).join((100 * decision_mix).round(1).add_prefix("rules: ").add_suffix(" (%)"))
display(summary.rename(index=GROUP_LABELS))
ratio_self = decision_mix.loc["self_employed", "approved"] / decision_mix.loc["employed", "approved"]
print(f"Approval ratio, self-employed vs employed: {ratio_self:.2f} (below 0.8 is the usual warning line)")

# %% [markdown]
# Self-employed applicants default about as often as employed ones (29% vs 27%; not a significant
# difference, p = 0.45). Yet the rules auto-approve them far less often: 36% vs 59%, an approval
# ratio of 0.60, well below the 0.8 warning line used in fair-lending checks. Unemployed applicants
# really are riskier (47%) and are almost never auto-approved (1.5%). Whether the rule score actually
# mis-prices these groups, once the other risk factors are accounted for, is tested when the
# baseline is evaluated.

# %% [markdown]
# ### 2.5 What this means for the model
#
# - **Label and exclusions.** Predict defaulted (1) vs repaid (0) on the 1,836 resolved loans. Leave out
#   `days_to_default` (it leaks the outcome), the rule-based score and decision (outputs of the system
#   being replaced) and `applicant_id`.
# - **Missing documents and mismatches become one feature**, `income_verification` (verified /
#   mismatch / not provided), instead of an imputed income.
# - **Use a model that can learn steps.** Risk jumps at thresholds (section 2.2), so a straight-line
#   effect would mis-fit; trees find the thresholds in the data.
# - **Drop columns with no signal**: deposit consistency (also an employment proxy) and the document
#   count beyond zero.
# - **No resampling.** A 30% default rate is only mildly imbalanced, and resampling would distort the
#   predicted probabilities. Decision thresholds handle the business costs instead.

# %% [markdown]
# ## 3. The rule-based baseline
#
# The rules combine five factors into a 0-100 score, then approve at 75 or above, flag for human
# review at 50-74 and deny below 50. This section measures how well that works and where it goes
# wrong. The rules need no training, so they are evaluated on all 1,836 resolved loans.
#
# ### 3.1 How the score really behaves

# %%
rebuilt = rebuild_rule_score(applications)
gap = (rebuilt - applications["rule_based_score"]).abs().round(2)
print(f"Rebuilt rubric: same decision for {(tiers_from_score(rebuilt) == rule_tiers(applications)).sum():,} of "
      f"{len(applications):,} applicants. Score identical for {(gap == 0).sum():,}, 0.1 apart for {(gap == 0.1).sum()}, "
      f"further apart for {(gap > 0.1).sum()} (all from rounding in the CSV)")
components = rule_components(applications)
print("Income-verification points awarded:", components["income_verification"].value_counts().to_dict())
no_docs = applications["documented_monthly_income"].isna()
print(f"Highest score of any applicant without documents: {applications.loc[no_docs, 'rule_based_score'].max():.1f} "
      "(auto-approval needs 75)")

# %% [markdown]
# Rebuilding the rubric from the CSV reproduces all 2,000 decisions, so the baseline is understood
# exactly. Two quirks stand out:
# - **Income verification is effectively all-or-nothing**: 95 points when documents match, 0
#   otherwise. The 40-point "partial match" branch fires for one applicant, and only because the CSV
#   rounds documented income.
# - **Applicants without documents can never be auto-approved**: missing those 28.5 weighted points
#   caps their score at 64.5. That fits the data (they default at 49%), but it comes from the weights
#   rather than from an explicit policy.

# %% [markdown]
# ### 3.2 How well the rules rank risk

# %%
rule_risk = -resolved["rule_based_score"].to_numpy()
oracle_p = oracle_default_probability(resolved)
ranking = pd.DataFrame.from_dict(
    {
        "Rule-based score": ranking_metrics(y, rule_risk),
        "Oracle: true default probability (diagnostic only)": ranking_metrics(y, oracle_p),
    },
    orient="index",
)
display(ranking.round(3))
save_table(ranking, "baseline_ranking")
no_risk_factors = oracle_p == oracle_p.min()
print(f"Most any model could add over the rules: {ranking['ROC-AUC'].iloc[1] - ranking['ROC-AUC'].iloc[0]:.3f} AUC")
print(f"Applicants with none of the generator's risk factors: {no_risk_factors.sum()}, "
      f"and {y[no_risk_factors].mean():.0%} of them still default")

# %% [markdown]
# **AUC** is the probability that a randomly chosen defaulter gets a riskier score than a randomly
# chosen repayer: 0.5 is a coin flip, 1.0 is perfect. **PR-AUC** (average precision) measures how
# cleanly the riskiest scores pick out defaulters; a random score gets about 0.30 here, the default
# rate. The rules reach **0.706 AUC**.
#
# Because the brief includes the generator, I can compute each applicant's true default probability:
# the "oracle". It is never used to train, tune or select anything; it only shows the ceiling, **0.741
# AUC**. Outcomes are noisy by design: 459 applicants with none of the risk factors still default 9% of
# the time. So **the most any model can add over the rules is about 0.035 AUC**, and models should be
# judged against that ceiling rather than a perfect 1.0.

# %% [markdown]
# ### 3.3 What the decisions cost

# %%
rule_decisions = rule_tiers(resolved)
decision_table = tier_table(y, rule_decisions)
display(decision_table.round(3))
rule_metrics = convention_metrics(y, rule_decisions)
display(rule_metrics.round(3))
for name, flagged in CONVENTIONS.items():
    print(f"Confusion matrix, convention {name}")
    display(confusion_table(y, np.isin(rule_decisions, flagged)))
save_table(decision_table, "baseline_tiers")
save_table(rule_metrics, "baseline_conventions")

# %% [markdown]
# - Auto-approved applicants default at 17% and denied ones at 54%. The **review tier holds 39% of
#   applicants with a 36% default rate, close to the 30% average**: the rules hand their hardest cases
#   to people.
# - Precision, recall, F1 and error rates need a yes/no prediction, but reviewers' decisions on
#   flagged applications aren't in the data. So I report two conventions that bracket the truth:
#   - **A** counts only "deny" as predicting default, as if reviewers approved every flagged case. Its
#     false-positive rate is the share of good applicants wrongly denied: **9.5%**. But 74% of
#     defaults are not denied.
#   - **B** counts anything not auto-approved as predicting default, as if reviewers denied every
#     flagged case. Its false-negative rate is the share of defaults that slip through
#     auto-approval: **26%**. But 45% of good applicants are held back.
# - The model is compared under both conventions later.

# %% [markdown]
# ### 3.4 Does the score mis-price employment groups?
#
# A fair score means the same risk for everyone with the same score. The test fits
# logit(default) ~ score + employment group. If group still predicts default once the score is known,
# the score mis-prices that group. An odds ratio below 1 means the group defaults less often than its
# score implies, so the score over-states its risk.

# %%
score_bias = score_conditional_odds(
    y, resolved["rule_based_score"] / 10, resolved["employment_status"], order=EMPLOYMENT_GROUPS
)
display(score_bias.rename(index=GROUP_LABELS).round(3))
save_table(score_bias, "baseline_score_bias")

band_labels = ["<50", "50-59", "60-69", "70-74", "75-79", "80+"]
bands = pd.cut(resolved["rule_based_score"], [0, 50, 60, 70, 75, 80, 101], right=False,
               labels=band_labels).astype(str).to_numpy()
fig, ax = plt.subplots(figsize=(7.8, 3.7))
x = np.arange(len(band_labels))
band_rates = {}
for group, color in [("employed", plots.BLUE), ("self_employed", plots.ORANGE)]:
    in_group = resolved["employment_status"].eq(group).to_numpy()
    table = rate_table(y[in_group], bands[in_group], band_labels)
    band_rates[GROUP_LABELS[group]] = table
    ax.fill_between(x, table["ci_low"], table["ci_high"], color=color, alpha=0.10, linewidth=0)
    ax.plot(x, table["rate"], color=color, marker="o", markersize=6, markeredgecolor=plots.SURFACE,
            markeredgewidth=1.5, label=GROUP_LABELS[group])
for boundary in (0.5, 3.5):
    ax.axvline(boundary, color=plots.AXIS, linewidth=1)
for center, zone in [(0, "denied"), (2, "flagged for review"), (4.5, "approved")]:
    ax.text(center, 0.80, zone, ha="center", color=plots.MUTED, fontsize=8.5)
ax.set_xticks(x, band_labels)
ax.set_xlabel("rule-based score")
ax.set_ylabel("default rate")
ax.set_ylim(0, 0.85)
plots.percent_axis(ax)
ax.legend(loc="lower left")
ax.set_title("At the same rule score, self-employed applicants default less")
fig.tight_layout()
plots.save(fig, "fig4_rule_score_bias")
plt.show()
display(pd.concat({g: format_rates(t[["n", "rate", "ci_low", "ci_high"]]) for g, t in band_rates.items()}, axis=1))

# %%
score = resolved["rule_based_score"]
self_marginal = (resolved["employment_status"].eq("self_employed") & score.between(69, 74.95)).to_numpy()
employed_marginal = (resolved["employment_status"].eq("employed") & score.between(75, 80.95)).to_numpy()
counts_2x2 = [[y[m].sum(), (1 - y[m]).sum()] for m in (self_marginal, employed_marginal)]
print(f"Self-employed scoring 69-74.9 (approved if their 6-point penalty were removed): "
      f"{y[self_marginal].mean():.0%} default (n = {self_marginal.sum()})")
print(f"Employed scoring 75-80.9 (approved today): {y[employed_marginal].mean():.0%} default "
      f"(n = {employed_marginal.sum()}); Fisher exact test p = {stats.fisher_exact(counts_2x2).pvalue:.2f}")
partial_credit = rule_components(resolved)["income_level"].eq(45).to_numpy()
employed_75 = (resolved["employment_status"].eq("employed") & score.between(75, 79.95)).to_numpy()
print(f"Employed applicants scoring 75-79.9: {employed_75.sum()}. {partial_credit[employed_75].mean():.0%} of them "
      f"got partial credit for income only 2-3x the loan; those default at {y[employed_75 & partial_credit].mean():.0%}")

# %% [markdown]
# **The score over-states self-employed applicants' risk.** At the same score they are about a third
# less likely to default (odds ratio 0.67, 95% CI 0.52-0.86, p = 0.002). The chart shows it band by
# band, most starkly at 75-79, where employed applicants default at 33% and self-employed ones at 10%.
# This is the 6-point penalty at work (weight 0.15 times the 40-point gap): self-employed applicants
# need better finances than employed ones to reach the same score. Unemployed applicants also do a
# little better than their score implies (odds ratio 0.75), but that isn't statistically significant
# (p = 0.13).
#
# Two supporting observations:
# - **Near the cutoff:** self-employed applicants who would be approved without the penalty (scores
#   69-74.9) default at 23%, while employed applicants approved today with 75-80.9 default at 34%.
#   That points the same way but isn't significant on its own (p = 0.15), so the score-conditional
#   test above is the real evidence.
# - **A second rubric flaw:** employed applicants scoring 75-79 default at 33%, no less than those at
#   70-74. Most of them (71%) only reach the approval band through partial credit for income just 2-3x
#   the loan, and section 2.2 showed loans that large relative to income are already high-risk.

# %% [markdown]
# ### 3.5 Can a simple patch to the rules fix this?
#
# Before building a model, it's worth asking how far a small change to the rubric gets. Each patch is
# applied exactly to the original score:
# - **Patch 1:** score self-employed applicants like employed ones (+6 points).
# - **Patch 2:** also stop rewarding deposit consistency (everyone gets those points), since it tracks
#   employment but doesn't predict default (section 2.3).
#
# For a fair comparison of ranking quality, every version approves as many applicants as today's rules
# (851) and denies as many (265). The oracle row shows the best possible result at those volumes.

# %%
n_approve, n_deny = int((rule_decisions == "approve").sum()), int((rule_decisions == "deny").sum())
group_mask = {g: resolved["employment_status"].eq(g).to_numpy() for g in EMPLOYMENT_GROUPS}


def approval_ratio(tiers: np.ndarray, group: str) -> float:
    return (tiers[group_mask[group]] == "approve").mean() / (tiers[group_mask["employed"]] == "approve").mean()


rule_versions = {
    "Rules as they are": rule_risk,
    "Patch 1: self-employed scored like employed": -adjusted_rule_score(resolved, self_employed_points=100).to_numpy(),
    "Patch 2: ...and deposit consistency ignored": -adjusted_rule_score(
        resolved, self_employed_points=100, ignore_deposit_consistency=True).to_numpy(),
    "Oracle (ceiling, diagnostic only)": oracle_p,
}
rows = {}
for name, risk in rule_versions.items():
    same_volume = matched_tiers(risk, n_approve, n_deny)
    gain = paired_bootstrap(y, rule_risk, risk, roc_auc_score)
    rows[name] = {
        "ROC-AUC": roc_auc_score(y, risk),
        "AUC gain [95% CI]": f"{gain['difference']:+.3f} [{gain['ci_low']:+.3f}, {gain['ci_high']:+.3f}]",
        "defaults auto-approved": int(y[same_volume == "approve"].sum()),
        "defaults denied": int(y[same_volume == "deny"].sum()),
        "good applicants denied": int((1 - y)[same_volume == "deny"].sum()),
        "self-employed approval ratio": approval_ratio(same_volume, "self_employed"),
        "unemployed approval ratio": approval_ratio(same_volume, "unemployed"),
    }
rule_patches = pd.DataFrame.from_dict(rows, orient="index")
display(rule_patches.round(3))
save_table(rule_patches, "baseline_patches")

# %% [markdown]
# - **Patch 1** lifts the self-employed approval ratio from 0.60 to 0.79 and improves ranking slightly
#   (AUC 0.711; the confidence interval just touches zero). At the same volume it lets through fewer
#   defaults (129 vs 141), because the self-employed applicants it promotes are safer than their old
#   score said.
# - **Patch 2** goes further: AUC 0.718 (a clear gain of +0.011, 95% CI +0.003 to +0.019), 124
#   defaults auto-approved, 155 caught in the deny tier, 110 good applicants denied instead of 122,
#   and a self-employed approval ratio of 0.88. Removing a factor that only measured employment made
#   the rules both fairer and more accurate.
# - **Patch 2 is the real baseline to beat.** It is a two-line change a rules team could ship
#   tomorrow, and it captures about a third of the achievable AUC gain (0.706 to 0.718, against a
#   0.741 ceiling). At the approval cutoff it is already close to the ceiling (124 defaults vs 123).
#   What patches don't fix is the rest of the rubric: partial credit for income only 2-3x the loan,
#   overdrafts carrying 6 of 100 points despite nearly doubling the default rate, and a withdrawals
#   factor that rises in a straight line when the data shows a step near 80%. Re-tuning those by hand
#   is what a model does systematically, and the next sections test whether that is worth it.

# %% [markdown]
# ## 4. Choosing the model
#
# **Held-out test set.** A stratified 25% of resolved loans (459) is set aside here and not touched
# until section 5. Every choice in this section is made by repeated cross-validation on the other
# 1,377 loans: 5 folds, repeated 3 times with different shuffles. Each candidate is scored on 15
# held-out folds, and all candidates use the same folds, so they can be compared fold by fold.
#
# **Metrics.** The main criterion is log-loss. It rewards confident, correct probabilities and heavily
# punishes confident mistakes, so it checks ranking and calibration together. That matters because
# lending decisions use the probabilities themselves. AUC, PR-AUC and the Brier score are reported
# alongside it.
#
# **Selection rule, fixed before seeing results:** choose the candidate with the lowest
# cross-validated log-loss, unless a simpler candidate is within one standard error of it, in which
# case choose the simpler one.
#
# **Features.** The candidates use the five core features and leave employment status out (variant
# V3). Section 4.4 shows that adding employment back doesn't change accuracy, which makes using it
# purely a fairness question, answered in section 7.

# %%
train_rows, test_rows = train_test_indices(y)
y_dev = y[train_rows]


def dev(frame: pd.DataFrame) -> pd.DataFrame:
    """The development-set rows of a per-applicant table."""
    return frame.iloc[train_rows].reset_index(drop=True)


splits = cv_splits(y_dev)
rules_folds = reference_folds(rule_risk[train_rows], y_dev, splits)
oracle_folds = reference_folds(oracle_p[train_rows], y_dev, splits, is_probability=True)
print(f"Development set: {len(train_rows):,} loans ({y_dev.mean():.1%} defaulted). "
      f"Test set: {len(test_rows)} loans ({y[test_rows].mean():.1%} defaulted), untouched until section 5.")

# %% [markdown]
# ### 4.1 Three candidate models
#
# - **M1: logistic regression.** The standard transparent credit model: each feature moves the
#   log-odds in a straight line.
# - **M2: additive gradient boosting with direction constraints.** Boosted trees where every tree may
#   split on only one feature, so the model is a sum of per-feature step functions (a boosted GAM).
#   It can learn the steps seen in section 2.2, each prediction splits exactly into per-feature
#   contributions, and each feature's effect is forced to point the sensible way: a larger loan
#   relative to income can never lower risk, and a larger balance can never raise it.
# - **M3: ordinary gradient boosting**, in two sizes, with interactions allowed and no constraints. It
#   tests whether interactions or unconstrained shapes add anything.

# %%
core_tree, core_linear = tree_matrix(features, WORKING_VARIANT), linear_matrix(features, WORKING_VARIANT)
candidates = {
    "M1: logistic regression": (make_logistic, core_linear),
    "M2: additive GBM, direction-constrained": (make_additive_gbm, core_tree),
    "M3: unconstrained GBM, 15 leaves": (make_unconstrained_gbm, core_tree),
    "M3: unconstrained GBM, depth 3": (partial(make_unconstrained_gbm, max_depth=3), core_tree),
}
candidate_cv = {name: cross_validate(factory, dev(X), y_dev, splits) for name, (factory, X) in candidates.items()}
model_comparison = summarize(
    {
        "Rules (reference)": rules_folds,
        **{name: result.fold_metrics for name, result in candidate_cv.items()},
        "Oracle (ceiling, diagnostic only)": oracle_folds,
    },
    reference_auc=rules_folds["ROC-AUC"],
)
display(model_comparison.round(4))
save_table(model_comparison, "models_cv")

# %%
short_names = {
    "M1: logistic regression": "M1 logistic regression",
    "M3: unconstrained GBM, 15 leaves": "M3 unconstrained, 15 leaves",
    "M3: unconstrained GBM, depth 3": "M3 unconstrained, depth 3",
    "M2: additive GBM, direction-constrained": "M2 additive, constrained",
}
fold_gains = {
    name: candidate_cv[name].fold_metrics["ROC-AUC"].to_numpy() - rules_folds["ROC-AUC"].to_numpy() for name in short_names
}
ceiling_gain = float((oracle_folds["ROC-AUC"] - rules_folds["ROC-AUC"]).mean())
label_x = max(max(gain.max() for gain in fold_gains.values()), ceiling_gain) + 0.008
jitter = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.4, 3.3))
for row, (name, gain) in enumerate(fold_gains.items()):
    color = plots.MODEL_COLOR if name.startswith("M2") else plots.RULES_COLOR
    ax.scatter(gain, row + jitter.uniform(-0.13, 0.13, len(gain)), s=24, color=color, edgecolor=plots.SURFACE,
               linewidth=0.8, zorder=3)
    ax.plot([gain.mean()] * 2, [row - 0.28, row + 0.28], color=plots.INK, linewidth=2, zorder=4)
    ax.text(label_x, row, f"mean {gain.mean():+.3f}", va="center", fontsize=8.5, color=plots.INK_SECONDARY)
ax.axvline(0, color=plots.AXIS, linewidth=1, zorder=1)
ax.axvline(ceiling_gain, color=plots.INK_SECONDARY, linewidth=1, zorder=1)
ax.text(ceiling_gain, len(fold_gains) - 0.45, f"ceiling (oracle) {ceiling_gain:+.3f}", ha="center", fontsize=8.5,
        color=plots.INK_SECONDARY)
ax.set_yticks(range(len(fold_gains)), list(short_names.values()))
ax.tick_params(axis="y", length=0)
ax.set_xlim(right=label_x + 0.03)
ax.set_ylim(-0.6, len(fold_gains) - 0.15)
ax.grid(axis="x", visible=True)
ax.grid(axis="y", visible=False)
ax.set_xlabel("AUC gain over the rules in each of the 15 cross-validation folds (bar = mean)")
ax.set_title("The additive model beats the rules in every fold and nearly reaches the ceiling")
fig.tight_layout()
plots.save(fig, "fig5_model_comparison")
plt.show()

# %% [markdown]
# - **M2 is best.** AUC 0.731 against 0.701 for the rules on these folds, better in all 15 folds, and
#   the lowest log-loss (0.535). It captures about 95% of the gain the oracle says is possible.
# - **M1 (logistic regression)** also beats the rules (0.721), but its log-loss is more than one
#   standard error worse than M2's: straight-line effects can't represent the steps in section 2.2.
# - **M3 (unconstrained boosting) does worse than M2**, and the larger version barely beats the rules
#   (better in only 6 of 15 folds). With 1,377 loans, the freedom to fit interactions mostly fits
#   noise, and nothing suggests the risk drivers interact.
# - So the constraints that make M2 interpretable also make it more accurate. There is no
#   accuracy-versus-explainability trade-off to make here.

# %% [markdown]
# ### 4.2 Tuning the additive model
#
# A small grid over the learning rate, the number of leaves per tree and the minimum loans per leaf
# (12 combinations), scored on the same folds. Under the selection rule, "simpler" means fewer leaves,
# then more loans per leaf.

# %%
grid = [
    {"learning_rate": rate, "max_leaf_nodes": leaves, "min_samples_leaf": per_leaf}
    for rate in (0.05, 0.1) for leaves in (4, 8, 16) for per_leaf in (20, 50)
]


def setting_name(setting: dict) -> str:
    return (f"learning rate {setting['learning_rate']}, {setting['max_leaf_nodes']} leaves, "
            f"min {setting['min_samples_leaf']} per leaf")


grid_cv = {setting_name(s): cross_validate(partial(make_additive_gbm, **s), dev(core_tree), y_dev, splits) for s in grid}
grid_summary = summarize({name: result.fold_metrics for name, result in grid_cv.items()})
simplest_first = [setting_name(s) for s in sorted(
    grid, key=lambda s: (s["max_leaf_nodes"], -s["min_samples_leaf"], grid_summary.loc[setting_name(s), "log-loss"]))]
chosen_setting = one_se_choice(grid_summary, simplest_first)
display(grid_summary.sort_values("log-loss").round(4))
save_table(grid_summary, "models_grid")
print("Chosen:", chosen_setting)
assert chosen_setting == setting_name(ADDITIVE_GBM_PARAMS), "update ADDITIVE_GBM_PARAMS in loan_risk/modeling.py"

# %% [markdown]
# Every setting lands within 0.0007 log-loss of the others. The model is insensitive to tuning, which
# is reassuring: its accuracy comes from its structure, not from a lucky setting. The rule picks the
# simplest one, which also ties for the best log-loss: learning rate 0.05, 4 leaves, at least 50 loans
# per leaf. That is the production model from here on.

# %% [markdown]
# ### 4.3 Which features earn their place?
#
# Each candidate column is added to the core five, one at a time. Loan ÷ income is also recomputed
# using documented income or bank deposits instead of stated income.

# %%
feature_sets = {"Core five (selected)": core_tree}
for extra in CANDIDATE_FEATURES:
    feature_sets[f"+ {FEATURE_LABELS[extra].lower()}"] = tree_matrix(features, WORKING_VARIANT, (extra,))
for name, alternative in loan_to_income_alternatives(resolved).items():
    if name != "stated income":
        feature_sets[f"loan / income using {name}"] = core_tree.assign(loan_to_income=alternative.to_numpy())
feature_summary = summarize(
    {name: cross_validate(make_production_model, dev(X), y_dev, splits).fold_metrics for name, X in feature_sets.items()}
)
display(feature_summary.round(4))
save_table(feature_summary, "models_feature_ablation")

# %% [markdown]
# - **No extra column improves log-loss by more than its standard error**, so under the selection rule
#   none is added. The document count comes closest, but its only real signal is "zero documents",
#   which income verification already captures (one and two documents default at 25% and 28%).
#   Adding raw stated income makes the model worse.
# - **Stated income gives the best loan ÷ income.** The bank-deposit version costs a little (log-loss
#   +0.0035, within one standard error) but is harder for applicants to game, which is worth weighing
#   in production (section 8).

# %% [markdown]
# ### 4.4 Does employment status add accuracy?

# %%
variant_cv = {v: cross_validate(make_production_model, dev(tree_matrix(features, v)), y_dev, splits) for v in EMPLOYMENT_VARIANTS}
variant_summary = summarize(
    {f"{v}: {VARIANT_DESCRIPTIONS[v]}": result.fold_metrics for v, result in variant_cv.items()},
    reference_auc=rules_folds["ROC-AUC"],
)
display(variant_summary.round(4))
save_table(variant_summary, "models_employment_variants")

# %% [markdown]
# All three variants are equally accurate: log-loss within 0.0007 of each other, with the version
# without employment nominally the best. Once income verification, affordability and bank behaviour
# are known, employment status adds no predictive value. So whether to use it is entirely a fairness
# question (section 7). The working model leaves it out.

# %% [markdown]
# ### 4.5 Are the probabilities calibrated?
#
# A calibrated model's 20% means 20%: among applicants given 20%, about one in five defaults. That
# matters when probabilities drive thresholds, pricing or expected-loss estimates.

# %%
selected_cv = variant_cv[WORKING_VARIANT]
platt_cv = cross_validate(make_platt_calibrated, dev(core_tree), y_dev, splits)
display(summarize({"Selected model": selected_cv.fold_metrics,
                   "Selected model + Platt scaling": platt_cv.fold_metrics}).round(4))

oof = selected_cv.mean_oof
decile = pd.qcut(oof, 10, labels=False)
calibration = rate_table(y_dev, decile, list(range(10)))
calibration["predicted"] = pd.Series(oof).groupby(decile).mean()
calibration["predicted inside 95% CI"] = calibration["predicted"].between(calibration["ci_low"], calibration["ci_high"])

fig, ax = plt.subplots(figsize=(4.8, 4.3))
ax.plot([0, 0.8], [0, 0.8], color=plots.AXIS, linewidth=1, zorder=1)
ax.text(0.52, 0.70, "perfect calibration", fontsize=8, color=plots.MUTED, ha="center")
ax.errorbar(calibration["predicted"], calibration["rate"],
            yerr=[calibration["rate"] - calibration["ci_low"], calibration["ci_high"] - calibration["rate"]],
            fmt="none", ecolor=plots.INK_SECONDARY, elinewidth=1, zorder=2)
ax.plot(calibration["predicted"], calibration["rate"], color=plots.BLUE, marker="o", markersize=6,
        markeredgecolor=plots.SURFACE, markeredgewidth=1.5, zorder=3)
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.8)
plots.percent_axis(ax)
plots.percent_axis(ax, "x")
ax.grid(axis="x", visible=True)
ax.set_xlabel("predicted default probability (decile average)")
ax.set_ylabel("observed default rate")
ax.set_title("Predicted vs observed default rate")
fig.tight_layout()
plots.save(fig, "fig6_calibration")
plt.show()
display(format_rates(calibration[["n", "predicted", "rate", "ci_low", "ci_high", "predicted inside 95% CI"]],
                     columns=("predicted", "rate", "ci_low", "ci_high")))
print(f"Average predicted probability {oof.mean():.1%} vs observed default rate {y_dev.mean():.1%}")

# %% [markdown]
# Calibration is good: 29.7% predicted on average against 29.7% observed, and in every decile the
# predicted rate falls inside the 95% confidence interval of the observed rate. Platt scaling
# (refitting a sigmoid on top) doesn't improve log-loss, so the model is used as is. That also keeps
# its explanations simple, because every contribution is in the model's own log-odds.

# %% [markdown]
# ### 4.6 The choice
#
# The production model is **M2: additive gradient boosting with direction constraints** (learning
# rate 0.05, 4 leaves, at least 50 loans per leaf) on the five core features, without recalibration.
# Sections 5 and 6 use the working version without employment status; section 7 decides whether to
# add an employment input back.
# - **It wins on the rule set in advance:** lowest cross-validated log-loss, better than the rules in
#   all 15 folds, about 95% of the possible AUC gain.
# - **It is interpretable by design:** every prediction is a sum of five per-feature contributions,
#   and each feature's effect points the sensible way (section 6 shows them).
# - **The choice isn't fragile:** every tuning setting performs the same, no extra column helps, and
#   employment status adds nothing.
#
# Section 5 trains it on all 1,377 development loans and scores the untouched test set for the first
# time.

# %% [markdown]
# ## 5. The model against the baseline
#
# The production model is trained once on all 1,377 development loans and then scores the 459 test
# loans it has never seen. It is compared with the rules as they are, the patched rules from section
# 3.5 (the stronger baseline) and the oracle ceiling.
#
# ### 5.1 Ranking quality on the test set

# %%
final_model = make_production_model(core_tree.columns).fit(dev(core_tree), y_dev)
p_test = final_model.predict_proba(core_tree.iloc[test_rows])[:, 1]
y_test = y[test_rows]
patched_risk = -adjusted_rule_score(resolved, self_employed_points=100, ignore_deposit_consistency=True).to_numpy()
test_scores = {
    "Rules as they are": rule_risk[test_rows],
    "Patched rules (patch 2)": patched_risk[test_rows],
    "Model": p_test,
    "Oracle (ceiling, diagnostic only)": oracle_p[test_rows],
}
rows = {}
for name, score in test_scores.items():
    rows[name] = ranking_metrics(y_test, score)
    if name.startswith(("Model", "Oracle")):
        rows[name] |= probability_metrics(y_test, score)
test_ranking = pd.DataFrame.from_dict(rows, orient="index")
display(test_ranking.round(4))
save_table(test_ranking, "test_ranking")

test_gains = pd.DataFrame.from_dict(
    {f"Model vs {name.lower()}": paired_bootstrap(y_test, test_scores[name], p_test, roc_auc_score)
     for name in ["Rules as they are", "Patched rules (patch 2)"]},
    orient="index",
)
display(test_gains.round(4))
save_table(test_gains, "test_auc_gains")
print(f"Test set: the model predicts {p_test.mean():.1%} default on average; {y_test.mean():.1%} actually defaulted")

# %%
test_rule_tiers = rule_tiers(resolved)[test_rows]
roc_styles = {
    "Rules as they are": (plots.RULES_COLOR, 2.0),
    "Patched rules (patch 2)": (plots.ORANGE, 2.0),
    "Model": (plots.MODEL_COLOR, 2.4),
    "Oracle (ceiling, diagnostic only)": (plots.INK_SECONDARY, 1.0),
}
fig, ax = plt.subplots(figsize=(5.4, 4.8))
ax.plot([0, 1], [0, 1], color=plots.AXIS, linewidth=1)
for name, score in test_scores.items():
    fpr, tpr, _ = roc_curve(y_test, score)
    color, width = roc_styles[name]
    ax.plot(fpr, tpr, color=color, linewidth=width, label=f"{name}: AUC {roc_auc_score(y_test, score):.3f}")
for flagged, label in [(("deny",), "rules: deny"), (("deny", "review"), "rules: not auto-approved")]:
    point = binary_metrics(y_test, np.isin(test_rule_tiers, flagged))
    ax.scatter(point["FPR"], point["recall"], s=42, color=plots.RULES_COLOR, edgecolor=plots.SURFACE, linewidth=1.5,
               zorder=5)
    ax.annotate(label, (point["FPR"], point["recall"]), xytext=(8, -12), textcoords="offset points", fontsize=8,
                color=plots.INK_SECONDARY)
ax.set_xlabel("false-positive rate (good applicants flagged)")
ax.set_ylabel("true-positive rate (defaults flagged)")
ax.grid(axis="x", visible=True)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("ROC curves on the untouched test set")
fig.tight_layout()
plots.save(fig, "fig7_roc_test")
plt.show()

# %% [markdown]
# - On loans it has never seen, the model reaches **AUC 0.761**, against 0.725 for the rules and 0.742
#   for the patched rules, close to the oracle's 0.766.
# - The gain over the rules is +0.036 (95% CI -0.002 to +0.073; 97% of bootstrap resamples favour the
#   model), and over the patched rules +0.020 (CI -0.013 to +0.053). A 459-loan test set can't pin
#   down differences this small, which is why section 5.3 adds a larger check.
# - Every system scores higher here than in cross-validation, the oracle included (0.766 vs 0.733),
#   so this test set is simply an easier sample. The gaps between systems are what matter.
# - The probabilities stay calibrated on new data: 29.8% predicted on average, 29.6% observed.
# - PR-AUC is about the same for every system here (0.54-0.55), the oracle included. With only 136
#   defaults it is too noisy to separate them.

# %% [markdown]
# ### 5.2 Decisions on the test set, at today's volumes
#
# Each system auto-approves as many test applicants as the rules did and denies as many, so the
# comparison is purely about who lands where. The confusion matrices use the two conventions from
# section 3.3.

# %%
n_test_approve, n_test_deny = int((test_rule_tiers == "approve").sum()), int((test_rule_tiers == "deny").sum())
test_tiers = {"Rules as they are": test_rule_tiers}
for name in ["Patched rules (patch 2)", "Model", "Oracle (ceiling, diagnostic only)"]:
    test_tiers[name] = matched_tiers(test_scores[name], n_test_approve, n_test_deny)
display(pd.concat({name: tier_table(y_test, tiers)[["applicants", "repaid", "defaulted", "default rate"]]
                   for name, tiers in test_tiers.items()}).round(3))
test_conventions = pd.concat({name: convention_metrics(y_test, tiers) for name, tiers in test_tiers.items()})
display(test_conventions.round(3))
save_table(test_conventions, "test_conventions")
display(pd.concat(
    {(name, convention.split(":")[0]): confusion_table(y_test, np.isin(test_tiers[name], flagged))
     for name in ["Rules as they are", "Model"] for convention, flagged in CONVENTIONS.items()},
    axis=1,
))

# %% [markdown]
# On the test set the model's gain shows up at the approval cutoff: 25 defaults slip through
# auto-approval against the rules' 32 (the oracle also lets through 25), so under convention B its
# false-negative rate is 18.4% against 23.5%. Its deny tier performs like the rules' (41 vs 42 defaults
# caught). With only 67 denials per system those counts move by several loans from sample to sample,
# so section 5.4 uses all 1,836 loans for stable estimates.

# %% [markdown]
# ### 5.3 Robustness: every loan scored out-of-fold
#
# 459 test loans give wide confidence intervals, and tier counts that small are noisy. So the
# production pipeline also runs through 5-fold cross-validation repeated 5 times on all 1,836
# resolved loans. Every loan is scored by models that never saw it; those scores, averaged over the 5
# repeats, drive the portfolio and fairness analyses that follow. The pipeline is fixed, so nothing
# is being selected here.

# %%
portfolio_splits = cv_splits(y, n_splits=5, n_repeats=5)
portfolio_cv = cross_validate(make_production_model, core_tree, y, portfolio_splits)
oof_all = portfolio_cv.mean_oof
rows = {}
for name, score in {"Model vs rules as they are": rule_risk, "Model vs patched rules": patched_risk}.items():
    gain = np.array([roc_auc_score(y[test], portfolio_cv.oof[repeat, test]) - roc_auc_score(y[test], score[test])
                     for repeat, _, test in portfolio_splits])
    rows[name] = {"mean AUC gain": gain.mean(), "smallest": gain.min(), "largest": gain.max(),
                  "folds where the model is better": f"{(gain > 0).sum()}/{len(gain)}"}
robustness = pd.DataFrame.from_dict(rows, orient="index")
display(robustness.round(4))
save_table(robustness, "robustness_auc_gains")
print("Out-of-fold AUC over all 1,836 loans: " + ", ".join(
    f"{name} {roc_auc_score(y, score):.3f}"
    for name, score in {"model": oof_all, "rules": rule_risk, "patched rules": patched_risk, "oracle": oracle_p}.items()))

# %% [markdown]
# Across 25 folds the model beats the rules in 24 (mean +0.032) and the patched rules in 22 (mean
# +0.021). Over all 1,836 loans its out-of-fold AUC is 0.738, against 0.706 for the rules, 0.718 for
# the patched rules and 0.741 for the oracle. The test-set result is not a fluke of one split.

# %% [markdown]
# ### 5.4 What would change if we deployed tomorrow?
#
# All 1,836 resolved loans, scored out-of-fold, at today's volumes: 851 auto-approved, 720 sent to
# review, 265 denied.

# %%
portfolio_tiers = {
    "Rules as they are": rule_decisions,
    "Patched rules (patch 2)": matched_tiers(patched_risk, n_approve, n_deny),
    "Model": matched_tiers(oof_all, n_approve, n_deny),
    "Oracle (ceiling, diagnostic only)": matched_tiers(oracle_p, n_approve, n_deny),
}
loan_amounts = resolved["loan_amount"].to_numpy()
rows = {}
for name, tiers in portfolio_tiers.items():
    approved, denied = tiers == "approve", tiers == "deny"
    rows[name] = {
        "defaults auto-approved": int(y[approved].sum()),
        "defaults denied": int(y[denied].sum()),
        "good applicants denied": int((1 - y)[denied].sum()),
        "default rate among approved": y[approved].mean(),
        **dollars_approved(y, tiers, loan_amounts),
        "self-employed approval ratio": approval_ratio(tiers, "self_employed"),
    }
portfolio = pd.DataFrame.from_dict(rows, orient="index")
display(portfolio.round(3))
save_table(portfolio, "portfolio_same_volume")
counts = ["defaults auto-approved", "defaults denied", "good applicants denied"]
display((portfolio[counts] * 1000 / len(y)).round(1).add_suffix(" per 1,000 applications"))
display(swap_sets(y, rule_decisions, portfolio_tiers["Model"], resolved["employment_status"],
                  order=EMPLOYMENT_GROUPS).rename(columns=GROUP_LABELS).round(3))

# %%
for name in ["Rules as they are", "Patched rules (patch 2)"]:
    saved = portfolio.loc[name, "$ lent to defaulters"] - portfolio.loc["Model", "$ lent to defaulters"]
    forgone = portfolio.loc[name, "$ lent to repayers"] - portfolio.loc["Model", "$ lent to repayers"]
    if forgone <= 0 or saved / forgone >= 1:
        verdict = "at any realistic margin" + (f" (break-even: {saved / forgone:.0%} of principal)" if forgone > 0 else "")
    else:
        verdict = f"if repaid loans earn less than {saved / forgone:.0%} of principal"
    print(f"Model vs {name.lower()}: ${saved:,.0f} less lent to defaulters, ${forgone:,.0f} less lent to repayers. "
          f"Assuming a default loses the whole loan, the model comes out ahead {verdict}.")

rule_default_rate = y[rule_decisions == "approve"].mean()
k = approvals_at_predicted_rate(oof_all, rule_default_rate)
growth = np.argsort(oof_all, kind="mergesort")[:k]
print(f"\nToday the rules approve {n_approve} applicants ({n_approve / len(y):.0%}) with a {rule_default_rate:.1%} default rate.")
print(f"Setting the model's cutoff so its predicted default rate among approved is {rule_default_rate:.1%} approves "
      f"{k} ({k / len(y):.0%}): {k - n_approve} more. Their actual default rate: {y[growth].mean():.1%}.")

# %%
shares = np.arange(0.10, 0.951, 0.01)
band_low, band_high = bootstrap_strategy_band(y, oof_all, shares)
fig, ax = plt.subplots(figsize=(7.8, 4.1))
ax.fill_between(shares, band_low, band_high, color=plots.MODEL_COLOR, alpha=0.10, linewidth=0)
for name, score, color in [("Rules as they are", rule_risk, plots.RULES_COLOR),
                           ("Patched rules (patch 2)", patched_risk, plots.ORANGE),
                           ("Model (shaded: 95% band)", oof_all, plots.MODEL_COLOR)]:
    ax.plot(shares, strategy_curve(y, score, shares), color=color, linewidth=2, label=name)
today = n_approve / len(y)
ax.scatter([today], [rule_default_rate], s=60, color=plots.INK, edgecolor=plots.SURFACE, linewidth=1.5, zorder=5)
ax.annotate(f"today: {today:.0%} approved,\n{rule_default_rate:.1%} of them default", (today, rule_default_rate),
            xytext=(-150, 22), textcoords="offset points", fontsize=8.5, color=plots.INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": plots.INK_SECONDARY, "linewidth": 0.8})
ax.scatter([k / len(y)], [y[growth].mean()], s=60, color=plots.MODEL_COLOR, edgecolor=plots.SURFACE, linewidth=1.5,
           zorder=5)
ax.annotate(f"model at the same default rate:\n{k / len(y):.0%} approved", (k / len(y), y[growth].mean()),
            xytext=(18, -34), textcoords="offset points", fontsize=8.5, color=plots.INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": plots.INK_SECONDARY, "linewidth": 0.8})
ax.set_xlabel("share of applicants auto-approved, lowest risk first")
ax.set_ylabel("default rate among approved")
plots.percent_axis(ax)
plots.percent_axis(ax, "x")
ax.legend(loc="upper left")
ax.set_title("Default rate among approved applicants, by approval volume")
fig.tight_layout()
plots.save(fig, "fig8_strategy_curve")
plt.show()


# %% [markdown]
# **At today's volumes**, compared with the rules, the model:
# - lets 13 fewer defaults through auto-approval (141 to 128: 7 fewer per 1,000 applications);
# - catches 24 more defaults in the deny tier (143 to 167: 13 more per 1,000);
# - wrongly denies 24 fewer good applicants (122 to 98: 13 fewer per 1,000).
#
# Both kinds of error fall at once, so at today's volumes there is no trade-off to weigh.
#
# **Who changes.** The 162 applicants the model newly approves default at 28%, and include 69
# self-employed and 25 unemployed applicants. The 162 it stops approving default at 36% and are almost
# all employed (153). The self-employed approval ratio rises from 0.60 to 0.89.
#
# **Money.** $47,400 less is lent to applicants who default (a 24% cut) and $34,000 less to applicants
# who repay, because the model favours loans that are small relative to income. Assuming a default
# loses the whole loan, the model beats today's rules at any realistic margin, and beats the patched
# rules if repaid loans earn less than about 27% of principal.
#
# **Or grow instead.** Setting the model's cutoff so that its *predicted* default rate among approved
# matches today's 16.6% would approve 991 applicants (54%) instead of 851 (46%), 140 more, and their
# *actual* default rate is 16.6%, exactly as predicted. The curve shows the full menu. The model keeps
# the default rate below today's rules at almost every approval volume. The patched rules match it
# around today's volume (within the model's uncertainty band), but fall behind once more than about
# 55% of applicants are approved.

# %% [markdown]
# ### 5.5 Verdict
#
# - **Against today's rules, the model is better on every measure:** AUC +0.032 (better in 24 of 25
#   folds; +0.036 on the test set), fewer defaults approved, more caught, fewer good applicants denied,
#   a quarter less money lent to defaulters, and self-employed applicants treated almost like employed
#   ones. It reaches about 95% of the gain any model could achieve on this data.
# - **Against the patched rules, the gain is smaller but consistent:** AUC +0.021 (better in 22 of 25
#   folds). The two are level at today's approval cutoff (128 vs 124 defaults let through), and the
#   model catches 12 more defaults while wrongly denying 12 fewer good applicants in the deny tier. It
#   also produces calibrated probabilities, which the rules can't. Those are what make "approve 140
#   more at the same default rate" possible, and they can be re-calibrated when conditions change
#   (section 8).
# - **Recommendation:** replace the rules with the model, starting in shadow mode. If something has to
#   ship before the model is approved, the two-line rules patch delivers most of the fairness gain and
#   part of the accuracy gain immediately.

# %% [markdown]
# ## 6. Explaining decisions
#
# The model is additive: its log-odds of default are a constant plus one learned contribution per
# feature. So every prediction splits exactly into a typical applicant's risk plus five
# contributions, with nothing left over. For an additive model these contributions are also exactly
# the SHAP values, so no approximation is involved. Explanations use the production model trained in
# section 5, with the development set as the reference population.
#
# ### 6.1 Exact contributions and decision cutoffs

# %%
explainer = AdditiveExplainer(final_model, dev(core_tree))
all_contributions = explainer.contributions(core_tree)
gap = np.abs(explainer.base_value + all_contributions.sum(axis=1) - final_model.decision_function(core_tree)).max()
print(f"Largest gap between (base value + contributions) and the model's own log-odds: {gap:.1e}")
approve_below = float(np.quantile(oof_all, n_approve / len(y)))
deny_at = float(np.quantile(oof_all, 1 - n_deny / len(y)))
typical_risk = float(sigmoid(explainer.base_value))
print(f"Cutoffs that reproduce today's volumes: auto-approve below {approve_below:.1%} predicted risk, "
      f"deny at {deny_at:.1%} or more. Applicant with typical values on every factor: {typical_risk:.1%}.")

# %% [markdown]
# ### 6.2 What the model learned, compared with the truth
#
# Each panel shows a feature's learned contribution to the log-odds of default (blue) next to the
# generator's true effect (gray), both relative to a typical applicant. The true effects come from the
# data generator and are used only as a check, never for training.

# %%
truth_offset = oracle_log_odds_terms(dev(resolved)).mean()
true_effect = {
    "loan_to_income": lambda v: 1.25 * (np.asarray(v) > 0.3),
    "bank_ending_balance": lambda v: 0.5 * (np.asarray(v) < 500),
    "withdrawal_to_deposit": lambda v: 0.5 * (np.asarray(v) > 0.8),
    "income_verification": lambda v: 1.25 * (np.asarray(v) != VERIFIED),
    "bank_has_overdrafts": lambda v: 1.0 * np.asarray(v),
}
grids = {
    "loan_to_income": np.geomspace(0.03, 5, 400),
    "bank_ending_balance": np.linspace(0, 8000, 400),
    "withdrawal_to_deposit": np.linspace(0.30, 0.95, 400),
}
levels = {
    "income_verification": ([VERIFIED, MISMATCH, NOT_PROVIDED], ["Verified", "Mismatch", "Not provided"]),
    "bank_has_overdrafts": ([0, 1], ["No overdrafts", "Overdrafts"]),
}
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
effect_axes = list(axes[0]) + list(axes[1][:2])
for ax, column in zip(axes[0], grids):
    grid = grids[column]
    ax.plot(grid, true_effect[column](grid) - truth_offset[column], color=plots.INK_SECONDARY, linewidth=1,
            label="generator's true effect")
    ax.plot(grid, explainer.effect(column, grid), color=plots.MODEL_COLOR, linewidth=2, label="learned by the model")
    ax.set_title(FEATURE_LABELS[column])
axes[0][0].set_xscale("log")
axes[0][0].set_xticks([0.05, 0.1, 0.3, 1, 3], ["5%", "10%", "30%", "100%", "300%"])
axes[0][0].minorticks_off()
plots.dollar_axis(axes[0][1])
plots.percent_axis(axes[0][2], "x")
axes[0][0].legend(loc="upper left", fontsize=8)
for ax, (column, (values, labels)) in zip(axes[1], levels.items()):
    x = np.arange(len(values))
    ax.scatter(x - 0.08, true_effect[column](values) - truth_offset[column], s=50, color=plots.INK_SECONDARY, zorder=3)
    ax.scatter(x + 0.08, explainer.effect(column, values), s=60, color=plots.MODEL_COLOR, edgecolor=plots.SURFACE,
               linewidth=1.5, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_title(FEATURE_LABELS[column])
low = min(ax.get_ylim()[0] for ax in effect_axes)
high = max(ax.get_ylim()[1] for ax in effect_axes)
for ax in effect_axes:
    ax.set_ylim(low, high)
    ax.axhline(0, color=plots.AXIS, linewidth=1, zorder=0)
axes[0][0].set_ylabel("contribution to log-odds")
axes[1][0].set_ylabel("contribution to log-odds")
importance = all_contributions.abs().mean().sort_values()
axes[1][2].barh([FEATURE_LABELS[c] for c in importance.index], importance.to_numpy(), height=0.45, color=plots.MODEL_COLOR)
axes[1][2].set_title("Average size of each contribution")
axes[1][2].set_xlabel("mean |contribution| (log-odds)")
axes[1][2].grid(axis="x", visible=True)
axes[1][2].grid(axis="y", visible=False)
axes[1][2].tick_params(axis="y", length=0)
fig.tight_layout()
plots.save(fig, "fig9_learned_effects")
plt.show()
display(importance.sort_values(ascending=False).rename(index=FEATURE_LABELS).rename("mean |contribution| (log-odds)")
        .round(3).to_frame())
for column, grid in grids.items():
    effect = explainer.effect(column, grid)
    step = int(np.argmax(np.abs(np.diff(effect))))
    print(f"{FEATURE_LABELS[column]}: largest learned step between {grid[step]:,.3g} and {grid[step + 1]:,.3g} "
          f"({effect[step + 1] - effect[step]:+.2f}); total range {np.ptp(effect):.2f} log-odds")
for column, (values, labels) in levels.items():
    effect = explainer.effect(column, values)
    truth = true_effect[column](values) - truth_offset[column]
    print(f"{FEATURE_LABELS[column]}: " + ", ".join(
        f"{label} {learned:+.2f} (true {true:+.2f})" for label, learned, true in zip(labels, effect, truth)))

# %% [markdown]
# **The strong effects are recovered precisely, the weak ones only roughly.**
# - **Loan ÷ income:** the model found the generator's threshold (the jump happens at 29.5% of
#   income; the truth is 30%) and roughly its size. Beyond about 70% of income it adds more risk,
#   which the truth doesn't. Very large loans relative to income are mostly taken by low-income,
#   often unemployed applicants, so this tail partly stands in for employment status, which the model
#   doesn't see (section 7 returns to this).
# - **Income verification and overdrafts:** close to the true effects (for example, "no documents"
#   +1.05 vs +1.03 true; overdrafts +0.96 vs +0.83).
# - **Bank balance:** right direction and roughly right size, but the step is blurred (around $1,300
#   instead of $500) and there is an extra one near $4,400.
# - **Withdrawals ÷ deposits:** rises above about 80% as it should. It also has one artifact:
#   applicants withdrawing under 34% of their deposits get a large risk reduction (-0.66) that the true
#   process doesn't have. About 80 training loans in that range happened to default less.
#
# I tested two fixes. Requiring at least 100 loans per leaf removes the pocket at no cost in accuracy
# (cross-validated log-loss 0.5338 vs 0.5341). But then a cross-validation training fold can no
# longer single out a group as small as the unemployed (about 90 applicants), which the employment
# comparisons need. Bagging (averaging 10 models fitted to resampled data) didn't remove it. A pocket
# of 80 noisy loans and a real group of 90 look the same to any regularizer; only more data separates
# them. So the artifact is disclosed rather than tuned away, and it's the first thing I'd revisit with
# more data. The weak features are also why only contributions of at least 0.10 log-odds are given as
# reasons.

# %% [markdown]
# ### 6.3 All applicants at once
#
# Each dot is one applicant's contribution from one feature (the exact SHAP value), coloured by the
# feature's value: darker means higher.

# %%
value_cmap = LinearSegmentedColormap.from_list("feature value", ["#86b6ef", "#0d366b"])
explanation = shap.Explanation(
    values=all_contributions.to_numpy(),
    base_values=np.full(len(core_tree), explainer.base_value),
    data=core_tree.to_numpy(dtype=float),
    feature_names=[FEATURE_LABELS[c] for c in core_tree.columns],
)
np.random.seed(0)  # shap shuffles the dots randomly; a fixed seed keeps the figure reproducible
shap.plots.beeswarm(explanation, show=False, color=value_cmap)
fig = plt.gcf()
fig.set_size_inches(8, 3.8)
fig.axes[0].set_xlabel("contribution to log-odds of default")
fig.axes[0].set_title("Contribution of each feature, for every applicant", loc="left")
plots.save(fig, "fig10_contributions_beeswarm")
plt.show()

# %% [markdown]
# Loan ÷ income spreads applicants furthest apart, from about -0.7 to +1.0 log-odds, followed by
# income verification and overdrafts. Balance and withdrawals mostly nudge risk, apart from the
# low-withdrawal pocket discussed above.

# %% [markdown]
# ### 6.4 Explaining individual applicants
#
# `explain_applicant` gives a reviewer:
# - the decision and the predicted risk;
# - the reasons, ranked by size, each with the risk the applicant would have if that factor were
#   typical;
# - what would change the decision.
#
# The four examples below come from the test set, so none was used to train the model: a typical
# auto-approval, a typical review case, a typical denial, and a self-employed applicant the rules held
# back but the model approves.

# %%
def explain_applicant(applicant_id: str) -> str:
    """Reviewer-facing explanation of the model's decision for one applicant."""
    position = int(np.flatnonzero(resolved["applicant_id"].to_numpy() == applicant_id)[0])
    row, applicant = core_tree.iloc[[position]], resolved.iloc[position]
    probability, reasons = explainer.explain(row)
    lines = [report(applicant_id, probability, reasons, approve_below, deny_at, typical_risk)]
    if probability >= approve_below:
        changes = []
        loan = largest_approvable_loan(explainer, row, applicant["stated_monthly_income"], approve_below)
        if loan is not None and loan < applicant["loan_amount"]:
            changes.append(f"a loan of ${loan:,} instead of ${applicant['loan_amount']:,} would be auto-approved")
        if row["income_verification"].iloc[0] != VERIFIED:
            verified = probability_with(explainer, row, income_verification=VERIFIED)
            changes.append(f"with verified income documents, risk would be {verified:.0%} "
                           f"({decision_for(verified, approve_below, deny_at)})")
        lines.append("  What would change the decision: "
                     + ("; ".join(changes) if changes else "no smaller loan or document alone is enough"))
    lines.append(f"  For comparison, the rules: {applicant['rule_based_decision'].replace('_', ' ')} "
                 f"(score {applicant['rule_based_score']:.1f}; applicant is {GROUP_LABELS[applicant['employment_status']].lower()})")
    return "\n".join(lines)


test_ids = resolved["applicant_id"].to_numpy()[test_rows]
test_decision = np.where(p_test < approve_below, "approve", np.where(p_test >= deny_at, "deny", "review"))
examples = {}
for tier, label in [("approve", "typical auto-approval"), ("review", "typical review case"), ("deny", "typical denial")]:
    members = np.flatnonzero(test_decision == tier)
    examples[label] = test_ids[members[np.argmin(np.abs(p_test[members] - np.median(p_test[members])))]]
flipped = np.flatnonzero((resolved["employment_status"].to_numpy()[test_rows] == "self_employed")
                         & (test_rule_tiers != "approve") & (test_decision == "approve"))
examples["self-employed: held back by the rules, approved by the model"] = test_ids[
    flipped[np.argmax(resolved["rule_based_score"].to_numpy()[test_rows][flipped])]]
for label, applicant_id in examples.items():
    print(f"[{label}]\n{explain_applicant(applicant_id)}\n")

# %%
def short_label(column: str, value) -> str:
    return {
        "income_verification": lambda v: "Income: " + INCOME_VERIFICATION_LABELS[int(v)],
        "loan_to_income": lambda v: f"Loan / income: {v:.0%}",
        "withdrawal_to_deposit": lambda v: f"Withdrawals / deposits: {v:.0%}",
        "bank_ending_balance": lambda v: f"Balance: ${v:,.0f}",
        "bank_has_overdrafts": lambda v: "Overdrafts: " + ("yes" if v else "no"),
    }[column](value)


fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.8), sharex=True)
for ax, (label, applicant_id) in zip(axes.ravel(), examples.items()):
    position = int(np.flatnonzero(resolved["applicant_id"].to_numpy() == applicant_id)[0])
    row = core_tree.iloc[[position]]
    probability, reasons = explainer.explain(row)
    reasons = reasons.iloc[::-1]
    ax.barh([short_label(c, row[c].iloc[0]) for c in reasons.index], reasons["contribution"], height=0.5,
            color=[plots.RAISES_RISK if c > 0 else plots.LOWERS_RISK for c in reasons["contribution"]])
    ax.axvline(0, color=plots.AXIS, linewidth=1)
    ax.set_title(f"{applicant_id}: {probability:.0%} risk, {decision_for(probability, approve_below, deny_at)}\n{label}",
                 fontsize=9.5)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("contribution to log-odds (red raises risk, blue lowers it)")
fig.tight_layout()
plots.save(fig, "fig11_applicant_explanations")
plt.show()

# %% [markdown]
# - **The reasons match intuition and the data.** The denial is driven by missing income documents,
#   overdrafts and a low balance; the review case by a loan twice the applicant's monthly income.
# - **Each explanation says what would change the outcome,** which is what an applicant needs from an
#   adverse-action notice. A $300 loan instead of $1,000 would be auto-approved for APP-0472, and
#   verified income documents would move APP-0369 from denial to review.
# - **APP-0841 shows the self-employment penalty in a single case.** The rules scored this applicant
#   74.9, 0.1 below auto-approval; the 6-point self-employment penalty is what kept them out. The model
#   puts their risk at 14%, well inside the approval range. One caveat: part of that low risk comes
#   from the low-withdrawal pocket described in section 6.2 (withdrawals at 31% of deposits). Even
#   with that factor set to typical, their risk would be 25%, still below the approval cutoff.

# %% [markdown]
# ### 6.5 Answering "why was this person denied?"
#
# - **Exact reasons, in order.** Every decision splits exactly into five contributions, so the reasons
#   a reviewer or regulator sees are the model's actual computation, not an approximation fitted
#   afterwards. Ranked by size, they serve as the principal reasons an adverse-action notice has to
#   give.
# - **Reasons always point the sensible way.** The direction constraints guarantee, for example, that
#   a larger loan relative to income is never a reason for lower risk, and the tests check it.
# - **Actionable alternatives.** Where possible the explanation says what would change the outcome: a
#   smaller loan from the product menu, or verified income documents.
# - **Reproducible decisions.** For each decision, log the inputs, the model version, the predicted
#   risk, the cutoffs in force and the ranked contributions. Any past decision can then be re-derived
#   exactly.
# - **The whole model fits on one page.** Five learned curves plus a constant (figure 9) are the entire
#   model. That makes it as inspectable as the rule rubric, and it was learned from outcomes rather than
#   set by hand.
# - **Employment in the reasons.** The working model has no employment input, so employment never
#   appears as a reason. Section 7 recommends adding one flag for unemployed applicants; it would then
#   appear as a reason for them, as employment-related reasons already do in standard adverse-action
#   reason lists (for example, "temporary or irregular employment").

# %% [markdown]
# ## 7. Fairness across employment groups
#
# The brief asks three things:
# - does the model replicate the rules' bias against self-employed applicants, correct it, or
#   introduce new biases;
# - what the approval and default rates are for each group, under the baseline and the model;
# - what to do about any bias found.
#
# Everything here uses the out-of-fold scores for all 1,836 resolved loans at today's volumes
# (section 5.4), because the test set holds only 46 unemployed applicants. The comparison includes the
# three ways of using employment (V1: all three groups, V2: an "unemployed" flag only, V3: nothing),
# the patched rules, the straight-line logistic regression, and the oracle, which shows what pricing
# every applicant at their true risk would look like.

# %%
variant_oof = {"V3": oof_all}
for variant in ("V1", "V2"):
    variant_oof[variant] = cross_validate(make_production_model, tree_matrix(features, variant), y,
                                          portfolio_splits).mean_oof
linear_oof = cross_validate(make_logistic, linear_matrix(features, "V1"), y, portfolio_splits).mean_oof
employment = resolved["employment_status"].to_numpy()
fair_systems = {
    "Rules as they are": (rule_decisions, None),
    "Patched rules (patch 2)": (portfolio_tiers["Patched rules (patch 2)"], None),
    "Model, no employment (V3)": (portfolio_tiers["Model"], oof_all),
    "Model + unemployed flag (V2)": (matched_tiers(variant_oof["V2"], n_approve, n_deny), variant_oof["V2"]),
    "Model, all three groups (V1)": (matched_tiers(variant_oof["V1"], n_approve, n_deny), variant_oof["V1"]),
    "Logistic regression (M1, V1)": (matched_tiers(linear_oof, n_approve, n_deny), linear_oof),
    "Oracle (ceiling, diagnostic only)": (portfolio_tiers["Oracle (ceiling, diagnostic only)"], oracle_p),
}
group_tables = {name: group_outcomes(y, tiers, employment, EMPLOYMENT_GROUPS, p=p)
                for name, (tiers, p) in fair_systems.items()}
save_table(pd.concat(group_tables), "fairness_group_outcomes")

# %% [markdown]
# ### 7.1 The ground truth

# %%
truth = rate_table(y, employment, EMPLOYMENT_GROUPS)
p_vs_employed = {group: proportions_ztest(truth.loc[[group, "employed"], "defaults"], truth.loc[[group, "employed"], "n"])[1]
                 for group in ["self_employed", "unemployed"]}
display(format_rates(truth).assign(**{"p-value vs employed": pd.Series(p_vs_employed)}).rename(index=GROUP_LABELS))

# %% [markdown]
# Self-employed applicants default about as often as employed ones (29.0% vs 27.2%, p = 0.45).
# Unemployed applicants default far more often (46.7%, p < 0.001). That difference is real, and a fair
# system should reflect it, as long as it prices it correctly.

# %% [markdown]
# ### 7.2 Approval and default rates by group, baseline vs model

# %%
brief_columns = ["applicants", "default rate", "auto-approved", "approval ratio", "sent to review", "denied",
                 "default rate among approved"]
display(pd.concat({name: group_tables[name][brief_columns]
                   for name in ["Rules as they are", "Model, no employment (V3)"]}).rename(index=GROUP_LABELS, level=1).round(3))
error_columns = ["good applicants denied", "good applicants not auto-approved", "defaults auto-approved"]
display(pd.concat({name: group_tables[name][error_columns]
                   for name in ["Rules as they are", "Model, no employment (V3)", "Oracle (ceiling, diagnostic only)"]})
        .rename(index=GROUP_LABELS, level=1).round(3))

# %%
markers = {
    "Rules as they are": (plots.RULES_COLOR, "o"),
    "Model, no employment (V3)": (plots.MODEL_COLOR, "o"),
    "Model + unemployed flag (V2)": (plots.ORANGE, "o"),
    "Oracle (ceiling, diagnostic only)": (plots.INK, "D"),
}
measures = {
    "auto-approved": "Auto-approved",
    "good applicants denied": "Good applicants wrongly denied",
    "default rate among approved": "Default rate among approved",
}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
rows = np.arange(len(EMPLOYMENT_GROUPS))[::-1]
for ax, (column, title) in zip(axes, measures.items()):
    for offset, (name, (color, marker)) in zip(np.linspace(0.24, -0.24, len(markers)), markers.items()):
        values = group_tables[name].loc[EMPLOYMENT_GROUPS, column].to_numpy()
        hollow = marker == "D"
        ax.scatter(values, rows + offset, s=46, marker=marker, color="none" if hollow else color,
                   edgecolor=color if hollow else plots.SURFACE, linewidth=1.5 if hollow else 1, label=name, zorder=3)
    ax.set_title(title)
    ax.set_xlim(left=0)
    plots.percent_axis(ax, "x")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(rows, [GROUP_LABELS[g] for g in EMPLOYMENT_GROUPS])
axes[0].tick_params(axis="y", length=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
plots.save(fig, "fig12_fairness_by_group")
plt.show()

# %% [markdown]
# - **The model corrects the bias against self-employed applicants.** The rules auto-approve 36% of
#   them against 60% of employed applicants (approval ratio 0.60). The model approves 46% against 52%
#   (0.89), close to the 0.93 that true risk would justify. Good self-employed applicants wrongly denied
#   fall from 12% to 9%.
# - **The model is far less harsh on creditworthy unemployed applicants.** The rules wrongly deny 45%
#   of unemployed applicants who would have repaid; the model denies 12%, about what true risk would
#   (13%).
# - **But without employment information the model over-approves unemployed applicants.** It
#   auto-approves 15% of them, where true risk would justify 5.5%, and 37% of those it approves
#   default, against 14-15% for the other groups. That is a new problem, examined next.

# %% [markdown]
# ### 7.3 Does any score mis-price a group?
#
# The same test as in section 3.4: once a system's score is known, does employment group still
# predict default? An odds ratio above 1 means the score understates that group's risk; below 1, it
# overstates it.

# %%
conditional = {}
for name, (_, p) in fair_systems.items():
    if name == "Rules as they are":
        score = resolved["rule_based_score"].to_numpy() / 10
    elif name == "Patched rules (patch 2)":
        score = -patched_risk / 10
    else:
        score = np.log(p / (1 - p))
    result = score_conditional_odds(y, score, employment, order=EMPLOYMENT_GROUPS)
    conditional[name] = {f"{GROUP_LABELS[g]}: {column}": result.loc[g, column]
                         for g in result.index for column in ["odds ratio", "p-value"]}
conditional = pd.DataFrame.from_dict(conditional, orient="index")
display(conditional.round(3))
save_table(conditional, "fairness_score_conditional")
understated = pd.DataFrame({name: 100 * (group_tables[name]["default rate"] - group_tables[name]["mean predicted risk"])
                            for name, (_, p) in fair_systems.items() if p is not None}).T
display(understated.rename(columns=GROUP_LABELS).round(1).add_suffix(": risk understated by (pts)"))

# %% [markdown]
# - **Every model variant, and the patched rules, prices self-employed applicants correctly** (odds
#   ratios 0.99-1.01, against 0.67 for the original rules). The penalty is gone.
# - **Without employment (V3), the model understates unemployed applicants' risk** by about 6 points
#   (40.9% predicted, 46.7% actual; odds ratio 1.34, p = 0.10). That is why it over-approves them. With
#   the unemployed flag (V2) the gap halves, to about 3 points (odds ratio 1.14, p = 0.46).

# %% [markdown]
# ### 7.4 Keep employment, drop it, or narrow it?

# %%
unemployed = employment == "unemployed"
variant_rows = {}
for name in ["Model, all three groups (V1)", "Model + unemployed flag (V2)", "Model, no employment (V3)",
             "Oracle (ceiling, diagnostic only)"]:
    tiers, p = fair_systems[name]
    approved_unemployed = unemployed & (tiers == "approve")
    variant_rows[name] = {
        "ROC-AUC": roc_auc_score(y, p),
        "log-loss": probability_metrics(y, p)["log-loss"],
        "defaults auto-approved": int(y[tiers == "approve"].sum()),
        "good applicants denied": int((1 - y)[tiers == "deny"].sum()),
        "self-employed approval ratio": group_tables[name].loc["self_employed", "approval ratio"],
        "unemployed approval ratio": group_tables[name].loc["unemployed", "approval ratio"],
        "unemployed approved": int(approved_unemployed.sum()),
        "...of whom defaulted": int(y[approved_unemployed].sum()),
        "unemployed risk understated by (pts)": understated.loc[name, "unemployed"],
    }
variant_comparison = pd.DataFrame.from_dict(variant_rows, orient="index")
display(variant_comparison.round(3))
save_table(variant_comparison, "fairness_variants")
extra = unemployed & (fair_systems["Model, no employment (V3)"][0] == "approve") & (
    fair_systems["Model + unemployed flag (V2)"][0] != "approve")
low, high = proportion_confint(y[extra].sum(), extra.sum(), method="wilson")
print(f"Unemployed applicants approved without the flag (V3) but not with it (V2): {extra.sum()}; "
      f"{y[extra].sum()} of them defaulted ({y[extra].mean():.0%}, 95% CI {low:.0%}-{high:.0%})")
print(f"Unemployed applicants asking for a loan above 30% of stated income: "
      f"{(features.loc[unemployed, 'loan_to_income'] > 0.3).mean():.0%} "
      f"(employed: {(features.loc[employment == 'employed', 'loan_to_income'] > 0.3).mean():.0%})")

# %% [markdown]
# - **Overall, the three variants are equally accurate** (AUC 0.738-0.739). V2 is marginally ahead:
#   127 defaults auto-approved and 96 good applicants denied, against 128 and 98 without employment
#   (V3).
# - **They treat self-employed applicants the same way** (approval ratio 0.89-0.92).
# - **They differ only for unemployed applicants.** Without the flag (V3), the model approves 27 of
#   them where true risk justifies 10. The 14 it approves that the flagged model (V2) doesn't default
#   at 50% (95% CI 27-73%). With the flag, 13 are approved, close to the oracle's 10.
# - **Most of the unemployed gap is affordability, not the label.** 91% of unemployed applicants ask
#   for a loan above 30% of their stated income, against 43% of employed applicants. That is a
#   legitimate, measured reason to approve fewer of them.
#
# Aggregate accuracy couldn't tell these variants apart (section 4.4). Only the group-level
# comparison against true risk shows what dropping employment actually does.

# %% [markdown]
# ### 7.5 Removing a column doesn't remove the information

# %%
proxy_rows = {}
for positive, negative in [("self_employed", "employed"), ("unemployed", "employed")]:
    keep = np.isin(employment, [positive, negative])
    proxy_rows[f"{GROUP_LABELS[positive]} vs {GROUP_LABELS[negative].lower()}, from the model's five inputs"] = \
        adversarial_auc(core_tree[keep], employment[keep] == positive)
keep = np.isin(employment, ["self_employed", "employed"])
proxy_rows["Self-employed vs employed, from deposit consistency alone (not a model input)"] = \
    adversarial_auc(features.loc[keep, ["bank_has_consistent_deposits"]], employment[keep] == "self_employed")
display(pd.DataFrame.from_dict(proxy_rows, orient="index").round(3))
display(all_contributions.groupby(employment).mean().reindex(EMPLOYMENT_GROUPS)
        .rename(index=GROUP_LABELS, columns=FEATURE_LABELS).round(2))

# %% [markdown]
# - **Unemployment is easy to infer from the model's inputs** (AUC 0.82-0.86), mainly through income
#   and loan ÷ income. Dropping the employment column doesn't stop the model reacting to unemployment.
#   It reacts through proxies, and less accurately, which is the miscalibration in section 7.3. On
#   average, unemployed applicants' loan ÷ income contribution is far higher than other groups', so
#   affordability already carries most of their extra risk.
# - **Self-employment is not inferable from the inputs** (AUC 0.51-0.54, close to chance). So the model
#   can't penalize self-employed applicants indirectly either.
# - **Deposit consistency would have been a proxy** (AUC 0.69 on its own) while carrying no risk
#   signal (section 2.3). Leaving it out was right, and the rubric's use of it is a second, hidden
#   self-employment penalty.

# %% [markdown]
# ### 7.6 Does the model introduce new biases?

# %%
denied_good = pd.DataFrame({name: group_tables[name]["good applicants denied"] for name in [
    "Rules as they are", "Patched rules (patch 2)", "Logistic regression (M1, V1)", "Model, no employment (V3)",
    "Model + unemployed flag (V2)", "Oracle (ceiling, diagnostic only)"]}).T
display((100 * denied_good).rename(columns=GROUP_LABELS).round(1).add_suffix(": good applicants denied (%)"))
print("Share without income documents: " + ", ".join(
    f"{GROUP_LABELS[g]} {(features.loc[employment == g, 'income_verification'] == NOT_PROVIDED).mean():.1%}"
    for g in EMPLOYMENT_GROUPS))

# %% [markdown]
# - **Dropping employment over-approves unemployed applicants** (V3, section 7.4). That is the main new
#   bias, in the opposite direction from the rules.
# - **Model form matters.** The straight-line logistic regression wrongly denies 27% of good
#   unemployed applicants, twice what true risk implies (13%), because it extends the loan-to-income
#   effect in a straight line where the true effect levels off. The additive model doesn't (12-16%).
# - **The patched rules fix the self-employment penalty but keep the unemployment one.** They wrongly
#   deny 48% of good unemployed applicants.
# - **The heavy weight on missing documents doesn't fall unevenly:** 14-15% of employed and
#   self-employed applicants have none, and 9% of unemployed.

# %% [markdown]
# ### 7.7 Fairness criteria conflict
#
# Three common fairness criteria pull in different directions:
# - **equal approval rates** across groups;
# - **equal error rates**, for example the same share of good applicants wrongly denied in every group;
# - **calibration**: the same predicted risk means the same actual risk in every group.
#
# When groups truly default at different rates, no score can satisfy all three at once (Kleinberg et
# al. 2016; Chouldechova 2017). For employed and self-employed applicants, whose default rates match,
# the model comes close on all three. For unemployed applicants it can't: equal approval rates would
# mean approving many loans that won't be repaid. I put calibration first, so that a risk score means
# the same thing for everyone, and judge approval gaps and error rates against what pricing everyone at
# their true risk would produce (the oracle).

# %% [markdown]
# ### 7.8 Recommendation
#
# 1. **Remove the self-employment penalty.** The rules overstate self-employed risk (odds ratio 0.67).
#    Every model variant prices it correctly (0.99-1.01) and lifts the self-employed approval ratio
#    from 0.60 to about 0.9, close to the 0.93 true risk justifies. There is no trade-off here.
# 2. **For unemployment, use one narrow flag (V2) rather than dropping it.** Unemployed applicants
#    really are riskier, and most of the gap is affordability, which the model already measures.
#    Dropping the flag (V3) doesn't remove the information. It makes the model understate their risk
#    by about 6 points and approve 27 where true risk justifies 10, and the extra approvals default at
#    about 50%. That is lending people money they can't repay, not fairness. The flag keeps every group
#    calibrated, matches true-risk approvals closely (13 vs 10), treats self-employed applicants exactly
#    like employed ones, and costs nothing in accuracy.
# 3. **Conditions.** Employment status, or source of income, is a protected or sensitive ground in
#    some places. Examples: "social condition" in Quebec, "source of income" in several provinces, and
#    receipt of public assistance under the US Equal Credit Opportunity Act. So compliance has to sign
#    off. If the flag can't be used, fall back to V3 and watch the default rate of approved unemployed
#    applicants closely. Either way, the longer-term fix is to replace the label with direct evidence of
#    repayment capacity, such as verified benefit or other income.
# 4. **What not to do.** Separate approval thresholds per group means treating groups differently on
#    purpose, which fair-lending rules generally prohibit. Reweighting the training data to equalize
#    approval rates would push the model further from true risk for unemployed applicants: the same
#    failure as V3.
# 5. **Keep measuring.** Collect employment status even when the model doesn't use it. Report
#    approval rates, wrongly-denied rates and calibration by group every month, and alert when a
#    group's approval ratio drops below 0.8 without a matching difference in risk.
# 6. **Who decides:** compliance and the credit-risk committee. My job is to put the trade-off in front
#    of them in numbers.
#
# Sections 4-6 used V3 as the working model. V2 performs the same overall (section 7.4), so those
# results carry over; the differences are the group-level ones shown here.
