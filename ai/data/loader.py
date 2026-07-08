"""
Dataset loader for the IDBI AI module.

Responsible for:
  - Loading the raw CSV from data/raw/
  - Validating the presence of required columns (via COLUMN_MAPPING)
  - Renaming columns to canonical internal names
  - Logging dataset shape and target distribution
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ai.config import COLUMN_MAPPING, DATASET_PATH, TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw loan dataset from a CSV file.

    Args:
        path: Path to CSV file. Defaults to ``config.DATASET_PATH``.

    Returns:
        DataFrame with canonical column names.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    csv_path = path or DATASET_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            f"Please place your loan CSV file at: {csv_path}\n"
            f"Recommended dataset: Kaggle 'Credit Risk Dataset' (laotse/credit-risk-dataset)"
        )

    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    df = _rename_to_canonical(df)
    _validate_required_columns(df)
    _log_target_distribution(df)

    return df


def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename dataset columns to canonical internal names using COLUMN_MAPPING.

    Only renames columns that are present in the DataFrame; ignores any
    mapping entries whose source column doesn't exist in the data.
    """
    reverse_map = {v: k for k, v in COLUMN_MAPPING.items()}
    rename_dict = {col: canonical for col, canonical in reverse_map.items() if col in df.columns}

    if rename_dict:
        logger.debug("Renaming columns: %s", rename_dict)
        df = df.rename(columns=rename_dict)

    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure the target column and at minimum the income and loan amount
    columns are present after renaming.

    Raises:
        ValueError: If critical columns are missing.
    """
    # After rename, "income" = annual_income, "loan_amount" = loan_amount
    required = {TARGET_COLUMN, "income", "loan_amount"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Critical columns missing from dataset after renaming: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Verify COLUMN_MAPPING in config.py matches your dataset.\n"
            f"TARGET_COLUMN={TARGET_COLUMN}"
        )

    logger.info("Column validation passed. Columns present: %s", list(df.columns))


def _log_target_distribution(df: pd.DataFrame) -> None:
    """Log class balance information for the target column."""
    if TARGET_COLUMN not in df.columns:
        return

    counts = df[TARGET_COLUMN].value_counts()
    total = len(df)

    for label, count in counts.items():
        pct = count / total * 100
        logger.info(
            "Target class %s → %d rows (%.1f%%)", label, count, pct
        )

    if len(counts) == 2:
        minority_pct = counts.min() / total * 100
        if minority_pct < 20:
            logger.warning(
                "Class imbalance detected: minority class is %.1f%%. "
                "scale_pos_weight will be set in XGBoost to compensate.",
                minority_pct,
            )


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excluding the target)."""
    return [col for col in df.columns if col != TARGET_COLUMN]
