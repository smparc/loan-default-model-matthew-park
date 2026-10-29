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
#
# To re-run: `jupytext --to ipynb analysis.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from loan_risk import plots
from loan_risk.data import EMPLOYMENT_GROUPS, default_label, load_applications, split_by_outcome
from loan_risk.eda import adversarial_auc, binned_rate, format_rates, rate_table
from loan_risk.features import INCOME_VERIFICATION_LABELS, MISMATCH_THRESHOLD, build_features

plots.apply_style()
pd.set_option("display.precision", 3)
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
