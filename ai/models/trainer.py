"""
Model trainer for the IDBI AI module.

Trains an XGBoost classifier with:
  - Automatic class imbalance handling via scale_pos_weight
  - Stratified K-Fold cross-validation
  - Early stopping on eval set to prevent overfitting

Design decision: XGBoost chosen over alternatives because:
  - Consistently top AUC on tabular credit datasets
  - Native SHAP TreeExplainer support (faster than KernelExplainer)
  - Built-in handling for missing values
  - Well-supported, stable, production-proven library
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from ai.config import CV_FOLDS, RANDOM_STATE, TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight for XGBoost to handle class imbalance.

    scale_pos_weight = count(negative class) / count(positive class)

    Args:
        y: Target series with binary labels (0=no default, 1=default).

    Returns:
        Positive float weight.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    if pos == 0:
        raise ValueError("No positive class (defaults) found in target column.")
    weight = neg / pos
    logger.info(
        "Class balance — neg: %d, pos: %d → scale_pos_weight = %.2f",
        neg, pos, weight,
    )
    return round(weight, 4)


def get_default_xgb_params(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Return sensible default XGBoost parameters for credit risk modelling.

    These are used for initial training before hyperparameter tuning.

    Args:
        scale_pos_weight: Class imbalance weight.

    Returns:
        Dict of XGBoost hyperparameters.
    """
    return {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "base_score": 0.5,          # CRITICAL: explicit float prevents XGBoost 3.x SHAP bug
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 30,
) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    If X_val and y_val are provided, early stopping is enabled to prevent
    overfitting. Otherwise trains for the full n_estimators rounds.

    Args:
        X_train: Training features (preprocessed numpy array).
        y_train: Training labels.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation labels.
        params: XGBoost parameters. Uses defaults if None.
        early_stopping_rounds: Patience for early stopping.

    Returns:
        Fitted :class:`xgboost.XGBClassifier`.
    """
    spw = compute_scale_pos_weight(y_train)
    model_params = params or get_default_xgb_params(scale_pos_weight=spw)

    logger.info("Training XGBoost model with params: %s", model_params)

    model = XGBClassifier(**model_params)

    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(
            "Training complete. Best iteration: %d",
            model.best_iteration if hasattr(model, "best_iteration") else model_params["n_estimators"],
        )
    else:
        model.fit(X_train, y_train)
        logger.info("Training complete (no early stopping).")

    return model


def cross_validate_model(
    model: XGBClassifier,
    X: np.ndarray,
    y: pd.Series,
    n_folds: int = CV_FOLDS,
    scoring: tuple = ("roc_auc", "average_precision", "f1"),
) -> Dict[str, Any]:
    """
    Run stratified K-fold cross-validation and return aggregated metrics.

    Stratified folds ensure each fold maintains the same class ratio,
    which is critical for imbalanced datasets.

    Args:
        model: Unfitted XGBClassifier instance.
        X: Full preprocessed feature matrix.
        y: Full target series.
        n_folds: Number of stratified folds.
        scoring: Tuple of sklearn scorer names.

    Returns:
        Dict mapping metric names to (mean, std) tuples.
    """
    logger.info("Running %d-fold stratified cross-validation...", n_folds)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=list(scoring),
        return_train_score=False,
        n_jobs=-1,
    )

    summary: Dict[str, Any] = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        summary[metric] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "scores": [round(float(s), 4) for s in scores],
        }
        logger.info(
            "CV %s: %.4f ± %.4f",
            metric, summary[metric]["mean"], summary[metric]["std"],
        )

    return summary
