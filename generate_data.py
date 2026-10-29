import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

employment = np.random.choice(['employed', 'self_employed', 'unemployed'], n, p=[0.6, 0.3, 0.1])
stated_income = np.where(employment == 'employed',
    np.random.normal(5000, 1500, n),
    np.where(employment == 'self_employed',
        np.random.normal(4500, 2000, n),
        np.random.normal(1500, 800, n)))
stated_income = np.clip(stated_income, 500, 25000).round(0)

# 15% missing documented income, 5% misrepresentation
has_docs = np.random.random(n) > 0.15
misrepresents = np.random.random(n) < 0.05
doc_income = np.where(has_docs,
    np.where(misrepresents,
        stated_income * np.random.uniform(0.2, 0.4, n),
        stated_income * np.random.uniform(0.9, 1.05, n)),
    np.nan)

loan_amount = np.random.choice([300, 500, 1000, 1500, 2000, 3000, 5000], n,
    p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05])
bank_balance = np.random.normal(2000, 1500, n).clip(0, 15000).round(0)
has_overdrafts = np.random.random(n) < np.where(bank_balance < 500, 0.6, 0.1)
consistent_deposits = np.random.random(n) < np.where(employment == 'employed', 0.8, 0.4)
monthly_deposits = np.where(has_docs, doc_income, stated_income) * np.random.uniform(0.85, 1.0, n)
monthly_withdrawals = monthly_deposits * np.random.uniform(0.3, 0.95, n)
num_docs = np.where(has_docs, np.random.choice([1, 2], n, p=[0.4, 0.6]), 0)

# Rule-based scoring
inc_ver_score = np.where(~has_docs, 0,
    np.where(misrepresents, 0,
        np.where(np.abs(stated_income - doc_income) / stated_income <= 0.1, 95, 40)))
inc_level = np.where(stated_income >= 3 * loan_amount, 90,
    np.where(stated_income >= 2 * loan_amount, 45, 0))
acct_stab = (np.where(bank_balance > 500, 40, 10) +
    np.where(~has_overdrafts, 30, 0) +
    np.where(consistent_deposits, 30, 0)).clip(0, 100)
emp_score = np.where(employment == 'employed', 100,
    np.where(employment == 'self_employed', 60, 20))
dti = np.where(monthly_deposits > 0,
    np.clip(100 - (monthly_withdrawals / monthly_deposits * 100), 0, 100), 50)

rule_score = (inc_ver_score * 0.30 + inc_level * 0.25 + acct_stab * 0.20 +
    emp_score * 0.15 + dti * 0.10).round(1)
rule_decision = np.where(rule_score >= 75, 'approved',
    np.where(rule_score >= 50, 'flagged_for_review', 'denied'))

# Actual outcomes (correlated with real risk, not rule score)
real_risk = (
    0.25 * np.where(has_docs & ~misrepresents, 0, 1) +
    0.25 * np.where(loan_amount <= stated_income * 0.3, 0, 1) +
    0.20 * np.where(has_overdrafts, 1, 0) +
    0.10 * np.where(employment == 'unemployed', 1, 0) +
    0.10 * np.where(bank_balance < 500, 1, 0) +
    0.10 * np.where(monthly_withdrawals / np.maximum(monthly_deposits, 1) > 0.8, 1, 0)
)
default_prob = 1 / (1 + np.exp(-5 * (real_risk - 0.45)))
defaults = np.random.random(n) < default_prob
ongoing = np.random.random(n) < 0.08

actual_outcome = np.where(ongoing, 'ongoing', np.where(defaults, 'defaulted', 'repaid'))
days_to_default = np.where(actual_outcome == 'defaulted',
    np.random.randint(15, 180, n), np.nan)

df = pd.DataFrame({
    'applicant_id': [f'APP-{i:04d}' for i in range(n)],
    'stated_monthly_income': stated_income,
    'documented_monthly_income': doc_income.round(0),
    'loan_amount': loan_amount,
    'employment_status': employment,
    'bank_ending_balance': bank_balance,
    'bank_has_overdrafts': has_overdrafts,
    'bank_has_consistent_deposits': consistent_deposits,
    'monthly_withdrawals': monthly_withdrawals.round(0),
    'monthly_deposits': monthly_deposits.round(0),
    'num_documents_submitted': num_docs,
    'rule_based_score': rule_score,
    'rule_based_decision': rule_decision,
    'actual_outcome': actual_outcome,
    'days_to_default': days_to_default
})

df.to_csv('loan_applications.csv', index=False)
print(f"Generated {len(df)} rows")
print(f"Outcomes: {df.actual_outcome.value_counts().to_dict()}")
print(f"Rule decisions: {df.rule_based_decision.value_counts().to_dict()}")
