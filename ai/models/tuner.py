"""
Hyperparameter tuner for the IDBI AI module.

Uses RandomizedSearchCV as the default tuning strategy.

Design decision: RandomizedSearchCV over GridSearchCV because:
  - Samples n_iter random combinations instead of exhaustive search
  - Finds near-optimal parameters in a fraction of the time
  - Easily scaled by increasing n_iter when more compute is available
  - No additional dependencies (part of scikit-learn)

FUTURE_INTEGRATION: Optuna-based tuning is designed as an optional
enhancement. The interface (tune_hyperparameters) remains the same.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from ai.config import (
    RANDOM_STATE,
    RANDOMIZED_SEARCH_CV,
    RANDOMIZED_SEARCH_N_ITER,
    RANDOMIZED_SEARCH_SCORING,
    XGBOOST_PARAM_DIST,
)
from ai.logger import get_logger
from ai.models.trainer import compute_scale_pos_weight, get_default_xgb_params

logger = get_logger(__name__)


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.Series,
    param_dist: Optional[Dict[str, Any]] = None,
    n_iter: int = RANDOMIZED_SEARCH_N_ITER,
    cv_folds: int = RANDOMIZED_SEARCH_CV,
    scoring: str = RANDOMIZED_SEARCH_SCORING,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training labels.
        param_dist: Hyperparameter search space. Defaults to config.XGBOOST_PARAM_DIST.
        n_iter: Number of random configurations to evaluate.
        cv_folds: Number of stratified CV folds for each configuration.
        scoring: Sklearn scorer name (default: "roc_auc").
        verbose: Verbosity level (0 = silent, 1 = progress, 2 = per-fold).

    Returns:
        Dict containing:
            "best_params": best hyperparameter dict
            "best_score": best CV score (mean)
            "cv_results": full RandomizedSearchCV cv_results_ dict
    """
    search_space = param_dist or XGBOOST_PARAM_DIST

    # Include class imbalance weight in the base model
    spw = compute_scale_pos_weight(y_train)
    base_params = get_default_xgb_params(scale_pos_weight=spw)
    # Remove params that are in the search space to avoid conflicts
    for key in search_space:
        base_params.pop(key, None)

    base_model = XGBClassifier(**base_params)

    cv_strategy = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )

    logger.info(
        "Starting RandomizedSearchCV: n_iter=%d, cv=%d, scoring=%s",
        n_iter, cv_folds, scoring,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        refit=True,                  # Refit on full training data with best params
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    logger.info(
        "RandomizedSearchCV complete. Best %s = %.4f",
        scoring, search.best_score_,
    )
    logger.info("Best params: %s", search.best_params_)

    return {
        "best_params": search.best_params_,
        "best_score": round(search.best_score_, 4),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
    }
