"""
Training Pipeline for the IDBI AI Credit Risk Intelligence Platform.

Orchestrates the complete training flow in order:
  Data Loading → Data Cleaning → Feature Engineering →
  Model Training → Evaluation → SHAP → Model Saving

Usage:
    python -m ai.pipeline.train_pipeline

Place your dataset at: ai/data/raw/loan_data.csv
Recommended: Kaggle "Credit Risk Dataset" (laotse/credit-risk-dataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ai.config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    DEFAULT_PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from ai.data.cleaner import clean
from ai.data.eda import run_eda
from ai.data.feature_engineering import engineer_features
from ai.data.loader import load_dataset
from ai.explainability.shap_engine import SHAPEngine
from ai.logger import get_logger
from ai.models.evaluator import (
    evaluate_model,
    find_optimal_threshold,
    generate_metrics_report,
)
from ai.models.persistence import (
    save_model,
    save_model_metadata,
    save_preprocessor,
    save_training_metrics,
)
from ai.models.preprocessor import build_preprocessor, get_feature_names_out
from ai.models.trainer import train_model
from ai.models.tuner import tune_hyperparameters

logger = get_logger(__name__)


class TrainPipeline:
    """
    End-to-end training pipeline orchestrator.

    Steps:
      1. Load data
      2. Clean data
      3. Run EDA (saves plots)
      4. Feature engineering
      5. Preprocess (fit encoder + scaler)
      6. Train/test split
      7. Train XGBoost
      8. Evaluate
      9. Hyperparameter tuning (optional)
      10. SHAP plots
      11. Save model artifacts
    """

    def __init__(self, run_tuning: bool = False) -> None:
        """
        Args:
            run_tuning: If True, run RandomizedSearchCV before final training.
                        Set to False for fast iteration during development.
        """
        self.run_tuning = run_tuning

    def run(self, dataset_path: Optional[Path] = None) -> dict:
        """
        Execute the full training pipeline.

        Args:
            dataset_path: Override dataset path. Defaults to config.DATASET_PATH.

        Returns:
            Dict with final metrics and artifact paths.
        """
        logger.info("=" * 60)
        logger.info("  IDBI AI Training Pipeline — Starting")
        logger.info("=" * 60)

        # ── Step 1: Load ─────────────────────────────────────────────────
        df = load_dataset(dataset_path)

        # ── Step 2: Clean ────────────────────────────────────────────────
        df = clean(df)

        # ── Step 3: EDA ──────────────────────────────────────────────────
        logger.info("Step 3: Running EDA...")
        run_eda(df)

        # ── Step 4: Feature Engineering ──────────────────────────────────
        logger.info("Step 4: Feature engineering...")
        df = engineer_features(df)

        # ── Step 5: Prepare X, y ─────────────────────────────────────────
        logger.info("Step 5: Preparing features and target...")
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN].astype(int)

        # ── Step 6: Train / Validation / Test split ───────────────────────
        # Strategy: 80% train → further split into 85% inner-train + 15% inner-val
        # Inner-val drives early stopping so the holdout test set stays pure.
        logger.info("Step 6: Splitting data (train / val / test)...")

        X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Inner split: 85% for actual training, 15% for early-stopping validation
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_trainval_raw, y_trainval,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y_trainval,
        )

        # Fit preprocessor ONLY on training data to prevent leakage
        preprocessor = build_preprocessor(df=X_train_raw)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val   = preprocessor.transform(X_val_raw)
        X_test  = preprocessor.transform(X_test_raw)
        feature_names = get_feature_names_out(preprocessor)

        logger.info(
            "Inner-train: %d | Val: %d | Test: %d | Features: %d",
            len(X_train), len(X_val), len(X_test), X_train.shape[1],
        )

        # ── Step 7: Hyperparameter Tuning (optional) ──────────────────────
        best_params = None
        if self.run_tuning:
            logger.info("Step 7: Running hyperparameter tuning (this may take a while)...")
            tuning_result = tune_hyperparameters(X_train, y_train)
            best_params = tuning_result["best_params"]
            logger.info("Tuning complete. Best AUC: %.4f", tuning_result["best_score"])

        # ── Step 8: Train ─────────────────────────────────────────────────
        logger.info("Step 8: Training XGBoost model...")
        model = train_model(
            X_train, y_train,
            X_val=X_val, y_val=y_val,   # Use inner-val for early stopping, not test!
            params=best_params,
            early_stopping_rounds=50,   # Give model more patience
        )

        # ── Step 9: Evaluate ──────────────────────────────────────────────
        logger.info("Step 9: Evaluating model...")
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_prob_train = model.predict_proba(X_train)[:, 1]

        optimal_threshold, _ = find_optimal_threshold(y_test.values, y_prob_test, metric="f1")

        test_metrics = evaluate_model(y_test.values, y_prob_test, threshold=optimal_threshold, dataset_label="test")
        train_metrics = evaluate_model(y_train.values, y_prob_train, threshold=optimal_threshold, dataset_label="train")

        print("\n" + generate_metrics_report(test_metrics))

        # ── Step 10: SHAP plots ───────────────────────────────────────────
        logger.info("Step 10: Generating SHAP plots...")
        try:
            shap_engine = SHAPEngine(model=model, feature_names=feature_names)
            # Use a representative sample (max 500 rows for speed)
            sample_size = min(500, len(X_test))
            rng = np.random.RandomState(RANDOM_STATE)
            sample_idx = rng.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[sample_idx]

            shap_engine.plot_summary(X_sample)
            shap_engine.plot_bar_importance(X_sample)
            logger.info("SHAP plots saved.")
        except Exception as e:
            logger.warning("SHAP plot generation failed (non-critical): %s", e)

        # ── Step 11: Save artifacts ───────────────────────────────────────
        logger.info("Step 11: Saving model artifacts...")
        model_path = save_model(model)
        prep_path = save_preprocessor(preprocessor)

        all_metrics = {
            "test": test_metrics,
            "train": train_metrics,
            "tuning_performed": self.run_tuning,
        }
        save_training_metrics(all_metrics)
        save_model_metadata(
            feature_names=feature_names,
            metrics=test_metrics,
            threshold=optimal_threshold,
            extra={"n_training_samples": len(X_train), "n_test_samples": len(X_test)},
        )

        logger.info("=" * 60)
        logger.info("  Training Pipeline Complete!")
        logger.info("  Model AUC-ROC : %.4f", test_metrics["auc_roc"])
        logger.info("  KS Statistic  : %.4f", test_metrics["ks_statistic"])
        logger.info("  Gini          : %.4f", test_metrics["gini_coefficient"])
        logger.info("  Threshold     : %.4f", optimal_threshold)
        logger.info("  Artifacts     : %s", ARTIFACTS_DIR)
        logger.info("=" * 60)

        return {
            "model_path": str(model_path),
            "preprocessor_path": str(prep_path),
            "feature_names": feature_names,
            "metrics": test_metrics,
            "threshold": optimal_threshold,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDBI AI Training Pipeline")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning before final training."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset CSV. Defaults to ai/data/raw/loan_data.csv"
    )
    args = parser.parse_args()

    pipeline = TrainPipeline(run_tuning=args.tune)
    result = pipeline.run(dataset_path=Path(args.dataset) if args.dataset else None)
    print("\nTraining complete. Artifacts saved.")
