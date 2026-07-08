"""
Data cleaning pipeline for the IDBI AI module.

Handles:
  - Dropping leakage/ID columns (defined in config.DROP_COLUMNS)
  - Data type coercion
  - Missing value imputation (median for numeric, mode for categorical)
  - Outlier capping via IQR (preserves dataset size)
  - Negative / impossible value correction
  - Yes/No binary column conversion
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ai.config import DROP_COLUMNS, TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)

# Numeric columns — impute with median
NUMERIC_IMPUTE_COLUMNS = [
    "age", "income", "monthly_income", "employment_length",
    "credit_score", "credit_history_length", "delinquency_count",
    "loan_amount", "loan_tenure", "interest_rate", "emi_amount",
    "existing_emi", "dti_ratio", "area_default_rate",
    "district_risk_score", "state_risk_score", "distance_to_branch_km",
]

# Categorical columns — impute with mode
CATEGORICAL_COLUMNS = [
    "gender", "education", "marital_status", "employment_type",
    "loan_purpose", "loan_product", "loan_source_type",
    "urban_rural_flag", "branch_region", "population_density_band",
    "economic_activity_type", "channel",
]

# Columns that must never be negative
NON_NEGATIVE_COLUMNS = [
    "age", "income", "monthly_income", "employment_length",
    "credit_score", "loan_amount", "loan_tenure",
    "emi_amount", "existing_emi",
]

# Yes/No string columns to convert to binary int (0/1)
YES_NO_COLUMNS = ["has_mortgage", "has_dependents", "has_cosigner"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Steps run in order.

    Args:
        df: Raw DataFrame (canonical column names applied by loader).

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Starting data cleaning. Input shape: %s", df.shape)

    df = _drop_leakage_columns(df)
    df = _remove_duplicates(df)
    df = _fix_data_types(df)
    df = _enforce_non_negative(df)
    df = _encode_yes_no(df)
    df = _impute_missing_values(df)
    df = _cap_outliers(df)
    df = _fix_age_bounds(df)

    logger.info("Cleaning complete. Output shape: %s", df.shape)
    return df


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID, leakage, and high-cardinality columns defined in config."""
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info("Dropped %d leakage/ID columns: %s", len(to_drop), to_drop)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


def _fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their expected types."""
    int_cols = ["age", "employment_length", "credit_history_length",
                "delinquency_count", "loan_tenure", "service_area_cluster"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = ["income", "monthly_income", "loan_amount", "interest_rate",
                  "emi_amount", "existing_emi", "dti_ratio",
                  "area_default_rate", "district_risk_score",
                  "state_risk_score", "distance_to_branch_km"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_cols = CATEGORICAL_COLUMNS
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan})

    return df


def _enforce_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Replace negative values in non-negative columns with NaN."""
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                logger.warning("Column '%s': %d negative values → NaN.", col, neg)
                df.loc[df[col] < 0, col] = np.nan
    return df


def _encode_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No string columns to binary int (Yes=1, No=0)."""
    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "YES": 1, "NO": 0}).fillna(0).astype(int)
            logger.debug("Encoded '%s' as binary (Yes=1, No=0).", col)
    return df


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    total_before = df.isnull().sum().sum()

    for col in NUMERIC_IMPUTE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            val = df[col].median()
            df[col] = df[col].fillna(val)
            logger.debug("Imputed '%s' with median=%.4f", col, val)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
            logger.debug("Imputed '%s' with mode='%s'", col, val)

    # Any remaining numeric NaNs
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any() and col != TARGET_COLUMN:
            df[col] = df[col].fillna(df[col].median())

    logger.info("Imputation: %d → %d missing values.", total_before, df.isnull().sum().sum())
    return df


def _cap_outliers(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Cap outliers using IQR Winsorization (preserves dataset size)."""
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c != TARGET_COLUMN]
    for col in columns:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        capped = ((df[col] < lower) | (df[col] > upper)).sum()
        if capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.debug("Capped %d outliers in '%s'.", capped, col)
    return df


def _fix_age_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip age to [18, 100]."""
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=18, upper=100)
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing value summary."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).sort_values("missing_pct", ascending=False)
