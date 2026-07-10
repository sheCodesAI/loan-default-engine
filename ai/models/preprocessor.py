"""
Preprocessing pipeline builder for the IDBI AI module.

CRITICAL DESIGN PRINCIPLE:
  Only include features that CAN be faithfully reproduced at inference time
  from the UI inputs. Any feature listed here that gets hardcoded to a single
  constant at inference time KILLS the model's discriminative power.

Feature groups:
  NUMERIC_FEATURES  → StandardScaler
  NOMINAL_FEATURES  → OneHotEncoder
  BINARY_FEATURES   → PassThrough
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ai.config import TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)

# ── Features that are AVAILABLE at inference time ─────────────────────────────
# These come directly from the UI form / CSV upload.

NUMERIC_FEATURES: List[str] = [
    # Demographics
    "age",
    # Financial — directly captured in UI
    "income",
    "monthly_income",
    "employment_length",
    "credit_score",
    "credit_history_length",
    "delinquency_count",
    # Loan — directly captured in UI
    "loan_amount",
    "loan_tenure",
    "interest_rate",
    "emi_amount",
    "existing_emi",
    "dti_ratio",
    # Geographic risk scores — computed from state lookup
    "area_default_rate",
    "district_risk_score",
    "state_risk_score",
    # Engineered (computed from UI inputs)
    "repayment_capacity",
    "cash_flow_health",
    "loan_to_income_ratio",
    "employment_stability",
    "credit_score_normalized",
]

NOMINAL_FEATURES: List[str] = [
    # Directly captured in UI
    "employment_type",
    "loan_purpose",
    "state",          # Geographic state — high signal for geo-risk
    # From dataset — can be derived from loan product type
    "loan_product",
    "loan_source_type",
    "branch_region",
]

BINARY_FEATURES: List[str] = [
    "has_mortgage",         # derived from home_ownership
    "has_dependents",       # from dependents > 0
    "has_cosigner",         # from co_applicant checkbox
    "past_default_flag",    # from dataset
    "near_retirement_flag", # age > 58
    "high_dti_flag",        # dti_ratio > 0.60
    "delinquency_flag",     # delinquency_count > 0
]


def build_preprocessor(
    df: Optional[pd.DataFrame] = None,
    numeric_features: Optional[List[str]] = None,
    nominal_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build an unfitted sklearn ColumnTransformer.

    If a DataFrame is provided, column lists are automatically filtered
    to only include columns present in the DataFrame.

    Args:
        df: Optional DataFrame used to filter column lists.
        numeric_features: Override numeric feature list.
        nominal_features: Override nominal feature list.

    Returns:
        Unfitted ColumnTransformer.
    """
    num_cols = numeric_features or NUMERIC_FEATURES
    nom_cols = nominal_features or NOMINAL_FEATURES
    bin_cols = list(BINARY_FEATURES)

    if df is not None:
        available = set(df.columns) - {TARGET_COLUMN}
        num_cols = [c for c in num_cols if c in available]
        nom_cols = [c for c in nom_cols if c in available]
        bin_cols = [c for c in bin_cols if c in available]

    logger.info(
        "Preprocessor columns — numeric: %d, nominal: %d, binary: %d",
        len(num_cols), len(nom_cols), len(bin_cols),
    )

    transformers = []

    if num_cols:
        transformers.append(("numeric", StandardScaler(), num_cols))

    if nom_cols:
        transformers.append((
            "nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            nom_cols,
        ))

    if bin_cols:
        transformers.append((
            "binary",
            FunctionTransformer(validate=False),
            bin_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names
