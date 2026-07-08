"""
Financial utility functions for the IDBI AI module.

All domain-specific financial mathematics live here.
These are pure functions (no side effects, no ML dependencies)
that can be unit-tested independently.

Key functions:
  - calculate_emi()              Monthly EMI via reducing-balance formula
  - calculate_dti()              Debt-to-Income ratio
  - calculate_repayment_capacity()
  - calculate_cash_flow_health()
  - calculate_expected_loss()    EL = PD × LGD × EAD
  - calculate_loan_to_income_ratio()
  - classify_financial_health()  Maps ratios to STRONG/MODERATE/WEAK/CRITICAL
"""

from __future__ import annotations

import math

from ai.config import DEFAULT_LGD, MONTHS_PER_YEAR


def calculate_emi(
    principal: float,
    annual_rate_pct: float,
    tenure_months: int,
) -> float:
    """
    Calculate monthly EMI using the standard reducing-balance formula.

    Formula:
        EMI = P × r × (1 + r)^n / ((1 + r)^n − 1)

    where r = monthly interest rate = annual_rate_pct / 12 / 100.

    Args:
        principal: Loan principal amount in INR.
        annual_rate_pct: Annual interest rate as a percentage (e.g., 12.5).
        tenure_months: Loan repayment period in months.

    Returns:
        Monthly EMI in INR. Returns 0.0 if principal or tenure is zero.
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0

    monthly_rate = annual_rate_pct / 100 / MONTHS_PER_YEAR

    if monthly_rate == 0:
        # Zero-interest loan
        return round(principal / tenure_months, 2)

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** tenure_months
        / ((1 + monthly_rate) ** tenure_months - 1)
    )
    return round(emi, 2)


def calculate_dti(
    monthly_obligations: float,
    monthly_income: float,
) -> float:
    """
    Calculate Debt-to-Income (DTI) ratio.

    DTI = total monthly debt obligations / gross monthly income.

    A DTI > 0.65 is considered high risk per IDBI policy.

    Args:
        monthly_obligations: Total monthly debt payments (EMI + existing loans).
        monthly_income: Gross monthly income.

    Returns:
        DTI ratio (0 to 1+). Returns 1.0 if income is zero (extreme risk).
    """
    if monthly_income <= 0:
        return 1.0
    return round(monthly_obligations / monthly_income, 4)


def calculate_repayment_capacity(
    monthly_income: float,
    monthly_emi: float,
    other_obligations: float = 0.0,
) -> float:
    """
    Calculate monthly repayment capacity (disposable income after all obligations).

    Args:
        monthly_income: Gross monthly income in INR.
        monthly_emi: Proposed loan EMI in INR.
        other_obligations: Total of other monthly debt obligations in INR.

    Returns:
        Repayment capacity in INR. Negative means the borrower is over-leveraged.
    """
    return round(monthly_income - monthly_emi - other_obligations, 2)


def calculate_cash_flow_health(
    monthly_income: float,
    total_monthly_obligations: float,
) -> float:
    """
    Calculate cash flow health as income-to-obligation ratio.

    A ratio > 2.0 = healthy; 1.5–2.0 = moderate; < 1.5 = stressed.

    Args:
        monthly_income: Gross monthly income.
        total_monthly_obligations: Sum of all monthly debt payments.

    Returns:
        Cash flow health ratio. Returns 0.0 if income is zero.
    """
    if monthly_income <= 0:
        return 0.0
    if total_monthly_obligations <= 0:
        return float("inf")   # No obligations — perfect cash flow
    return round(monthly_income / total_monthly_obligations, 4)


def calculate_expected_loss(
    default_probability: float,
    loan_amount: float,
    lgd: float = DEFAULT_LGD,
) -> float:
    """
    Calculate Expected Loss (EL) using the Basel II formula.

    EL = PD × LGD × EAD

    where:
        PD  = Probability of Default (from ML model)
        LGD = Loss Given Default (fraction; default 45% for unsecured)
        EAD = Exposure at Default = loan_amount (simplified: outstanding balance)

    Args:
        default_probability: Model-predicted probability of default (0–1).
        loan_amount: Outstanding loan balance (EAD) in INR.
        lgd: Loss Given Default fraction. Defaults to config.DEFAULT_LGD.

    Returns:
        Expected Loss in INR, rounded to 2 decimal places.
    """
    if not (0.0 <= default_probability <= 1.0):
        raise ValueError(f"default_probability must be in [0, 1]. Got: {default_probability}")
    if loan_amount < 0:
        raise ValueError(f"loan_amount must be non-negative. Got: {loan_amount}")
    if not (0.0 <= lgd <= 1.0):
        raise ValueError(f"lgd must be in [0, 1]. Got: {lgd}")

    return round(default_probability * lgd * loan_amount, 2)


def calculate_loan_to_income_ratio(
    loan_amount: float,
    annual_income: float,
) -> float:
    """
    Calculate loan-to-income (LTI) ratio.

    A ratio > 20 is considered very high risk per IDBI policy.

    Args:
        loan_amount: Requested loan amount in INR.
        annual_income: Annual income in INR.

    Returns:
        LTI ratio. Returns float('inf') if income is zero.
    """
    if annual_income <= 0:
        return float("inf")
    return round(loan_amount / annual_income, 4)


def classify_financial_health(
    dti_ratio: float,
    repayment_capacity: float,
    cash_flow_health: float,
) -> str:
    """
    Classify overall financial health into four buckets.

    Rules (in priority order):
        CRITICAL : DTI > 0.75 OR repayment_capacity < 0
        WEAK     : DTI > 0.55 OR cash_flow_health < 1.3
        MODERATE : DTI > 0.40 OR cash_flow_health < 1.7
        STRONG   : otherwise

    Args:
        dti_ratio: Debt-to-Income ratio.
        repayment_capacity: Monthly disposable income after EMI.
        cash_flow_health: Income / obligations ratio.

    Returns:
        One of "STRONG", "MODERATE", "WEAK", "CRITICAL".
    """
    if dti_ratio > 0.75 or repayment_capacity < 0:
        return "CRITICAL"
    if dti_ratio > 0.55 or cash_flow_health < 1.3:
        return "WEAK"
    if dti_ratio > 0.40 or cash_flow_health < 1.7:
        return "MODERATE"
    return "STRONG"


def compute_borrower_score(
    credit_score_proxy_scaled: float,   # 0–100, higher = better
    dti_ratio: float,                    # lower = better
    repayment_capacity: float,           # higher = better
    monthly_income: float,
    employment_stability: float,         # 0–1
    cash_flow_health: float,             # higher = better
    weights: dict | None = None,
) -> float:
    """
    Compute a composite borrower score (0–100).

    Components (defaults from config):
        - credit_score_proxy    35%
        - dti_ratio             25%  (inverted: low DTI = high score)
        - repayment_capacity    20%  (normalised against monthly income)
        - employment_stability  10%
        - cash_flow_health      10%  (capped at ratio of 3.0)

    Returns:
        Borrower score 0–100, rounded to 1 decimal place.
    """
    from ai.config import BORROWER_SCORE_WEIGHTS
    w = weights or BORROWER_SCORE_WEIGHTS

    # Component scores normalised to 0–100
    cs = min(credit_score_proxy_scaled, 100)

    dti_score = max(0, (1 - dti_ratio)) * 100       # DTI 0 → 100, DTI 1 → 0
    dti_score = min(dti_score, 100)

    # Repayment capacity: positive capacity relative to monthly income
    if monthly_income > 0:
        rc_score = min(max(repayment_capacity / monthly_income, 0), 1) * 100
    else:
        rc_score = 0.0

    es_score = min(employment_stability, 1) * 100

    # Cash flow health: 1.0 → 0, 3.0+ → 100
    cf_score = min(max((cash_flow_health - 1.0) / 2.0, 0), 1) * 100

    score = (
        w.get("credit_score_proxy", 0.35) * cs
        + w.get("dti_ratio", 0.25) * dti_score
        + w.get("repayment_capacity", 0.20) * rc_score
        + w.get("employment_stability", 0.10) * es_score
        + w.get("cash_flow_health", 0.10) * cf_score
    )
    return round(min(max(score, 0), 100), 1)


def compute_trust_score(
    credit_history_length_months: int,
    default_on_file: bool,
    employment_stability: float,
    credit_score_proxy_scaled: float,
) -> float:
    """
    Compute a borrower trust score (0–100) based on credit history signals.

    Penalises past defaults heavily. Rewards long credit history and
    stable employment.

    Returns:
        Trust score 0–100.
    """
    base = credit_score_proxy_scaled                          # 0–100

    # Credit history bonus: up to +20 for 10+ years
    history_bonus = min(credit_history_length_months / 120, 1.0) * 20

    # Employment stability bonus: up to +10
    employment_bonus = employment_stability * 10

    # Past default penalty: -30
    default_penalty = 30 if default_on_file else 0

    score = base + history_bonus + employment_bonus - default_penalty
    return round(min(max(score, 0), 100), 1)
