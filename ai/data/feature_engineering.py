"""
Feature engineering for the IDBI AI module.

The IDBI dataset already contains many pre-computed financial features
(debt_to_income, emi_amount, credit_score, area_default_rate, etc.).
This module derives additional interaction and ratio features.

Derived features:
  - repayment_capacity      monthly_income - emi_amount - existing_emi
  - cash_flow_health        monthly_income / (emi_amount + existing_emi)
  - loan_to_income_ratio    loan_amount / income
  - employment_stability    scaled 0-1 from employment_length (60 months = max)
  - credit_score_normalized credit_score / 900
  - near_retirement_flag    age > 58
  - high_dti_flag           dti_ratio > 0.60
  - delinquency_flag        delinquency_count > 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.config import TARGET_COLUMN
from ai.logger import get_logger
from ai.utils.financial_utils import (
    calculate_cash_flow_health,
    calculate_loan_to_income_ratio,
    calculate_repayment_capacity,
)

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame with canonical column names.

    Returns:
        DataFrame with additional derived feature columns appended.
    """
    logger.info("Starting feature engineering. Input shape: %s", df.shape)

    df = _add_repayment_capacity(df)
    df = _add_cash_flow_health(df)
    df = _add_loan_to_income_ratio(df)
    df = _add_employment_stability(df)
    df = _add_credit_score_normalized(df)
    df = _add_near_retirement_flag(df)
    df = _add_high_dti_flag(df)
    df = _add_delinquency_flag(df)

    new_cols = [
        "repayment_capacity", "cash_flow_health", "loan_to_income_ratio",
        "employment_stability", "credit_score_normalized",
        "near_retirement_flag", "high_dti_flag", "delinquency_flag",
    ]
    existing = [c for c in new_cols if c in df.columns]
    logger.info("Feature engineering complete. New features: %s", existing)
    return df


# ─── Individual Feature Functions ────────────────────────────────────────────

def _add_repayment_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly surplus after EMI + existing obligations."""
    if "monthly_income" in df.columns and "emi_amount" in df.columns:
        existing = df.get("existing_emi", pd.Series(0, index=df.index))
        df["repayment_capacity"] = (
            df["monthly_income"] - df["emi_amount"] - existing
        ).round(2)
    else:
        logger.warning("Cannot compute repayment_capacity — columns missing.")
        df["repayment_capacity"] = np.nan
    return df


def _add_cash_flow_health(df: pd.DataFrame) -> pd.DataFrame:
    """Income-to-obligation ratio (>2 = healthy, <1.3 = stressed)."""
    if "monthly_income" in df.columns and "emi_amount" in df.columns:
        existing = df.get("existing_emi", pd.Series(0, index=df.index))
        total_obligations = df["emi_amount"] + existing
        df["cash_flow_health"] = df.apply(
            lambda row: calculate_cash_flow_health(
                row["monthly_income"],
                total_obligations.loc[row.name],
            ),
            axis=1,
        ).round(4)
    else:
        df["cash_flow_health"] = np.nan
    return df


def _add_loan_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """loan_amount / annual_income."""
    if "loan_amount" in df.columns and "income" in df.columns:
        df["loan_to_income_ratio"] = (
            df["loan_amount"] / df["income"].replace(0, np.nan)
        ).round(4)
    else:
        df["loan_to_income_ratio"] = np.nan
    return df


def _add_employment_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Employment duration scaled 0-1 (60 months = fully stable)."""
    if "employment_length" in df.columns:
        df["employment_stability"] = (
            df["employment_length"].clip(0, 60) / 60
        ).round(4)
    else:
        df["employment_stability"] = 0.5
    return df


def _add_credit_score_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CIBIL credit score to 0-1 range (300-900 scale)."""
    if "credit_score" in df.columns:
        df["credit_score_normalized"] = (
            (df["credit_score"].clip(300, 900) - 300) / 600
        ).round(4)
    else:
        df["credit_score_normalized"] = 0.5
    return df


def _add_near_retirement_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if borrower is older than 58."""
    if "age" in df.columns:
        df["near_retirement_flag"] = (df["age"] > 58).astype(int)
    else:
        df["near_retirement_flag"] = 0
    return df


def _add_high_dti_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if DTI ratio > 0.60."""
    if "dti_ratio" in df.columns:
        df["high_dti_flag"] = (df["dti_ratio"] > 0.60).astype(int)
    else:
        df["high_dti_flag"] = 0
    return df


def _add_delinquency_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if borrower has any past delinquencies."""
    if "delinquency_count" in df.columns:
        df["delinquency_flag"] = (df["delinquency_count"] > 0).astype(int)
    else:
        df["delinquency_flag"] = 0
    return df
