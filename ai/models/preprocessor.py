"""
Preprocessing pipeline builder for the IDBI AI module.

Builds a sklearn ColumnTransformer that handles:
  - Numeric features     → StandardScaler
  - Nominal categoricals → OneHotEncoder
  - Binary (0/1) int     → PassThrough (no scaling needed)

Updated for the IDBI dataset which has:
  - Actual credit_score (no loan_grade proxy needed)
  - Pre-computed dti_ratio, emi_amount, area_default_rate
  - Geographic risk scores from the dataset

Design: Separate preprocessor (not embedded inside model Pipeline) so it
can be inspected, saved, and loaded independently from the model.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ai.config import TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)

# ─── Feature Column Groups ────────────────────────────────────────────────────

NUMERIC_FEATURES: List[str] = [
    # Demographics
    "age",
    # Financial
    "income",
    "monthly_income",
    "employment_length",
    "credit_score",
    "credit_history_length",
    "delinquency_count",
    # Loan
    "loan_amount",
    "loan_tenure",
    "interest_rate",
    "emi_amount",
    "existing_emi",
    "dti_ratio",
    # Geographic / area-level
    "area_default_rate",
    "district_risk_score",
    "state_risk_score",
    "distance_to_branch_km",
    "service_area_cluster",
    # Engineered
    "repayment_capacity",
    "cash_flow_health",
    "loan_to_income_ratio",
    "employment_stability",
    "credit_score_normalized",
]

NOMINAL_FEATURES: List[str] = [
    "employment_type",
    "gender",
    "education",
    "marital_status",
    "loan_purpose",
    "loan_product",
    "loan_source_type",
    "urban_rural_flag",
    "branch_region",
    "population_density_band",
    "economic_activity_type",
    "channel",
    "state",               # High-value geographic signal: captures state-level economic patterns
]

BINARY_FEATURES: List[str] = [
    "has_mortgage",         # already encoded 0/1 by cleaner
    "has_dependents",       # already encoded 0/1
    "has_cosigner",         # already encoded 0/1
    "past_default_flag",
    "near_retirement_flag",
    "high_dti_flag",
    "delinquency_flag",
]


def build_preprocessor(
    df: Optional[pd.DataFrame] = None,
    numeric_features: Optional[List[str]] = None,
    nominal_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build an unfitted sklearn ColumnTransformer.

    If a DataFrame is provided, column lists are automatically filtered
    to only include columns present in the DataFrame — making it robust
    to optional/missing columns.

    Args:
        df: Optional DataFrame used to filter column lists.
        numeric_features: Override numeric feature list.
        nominal_features: Override nominal feature list.

    Returns:
        Unfitted :class:`sklearn.compose.ColumnTransformer`.
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
