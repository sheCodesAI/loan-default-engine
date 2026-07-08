"""
Model persistence — versioned save and load for models and preprocessors.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from ai.config import (
    ARTIFACTS_DIR,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODEL_VERSION,
    PREPROCESSOR_FILENAME,
    TRAINING_METRICS_FILENAME,
)
from ai.logger import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: Optional[Path] = None) -> Path:
    """Save trained model to disk using joblib."""
    save_path = path or (ARTIFACTS_DIR / MODEL_FILENAME)
    joblib.dump(model, save_path)
    logger.info("Model saved: %s", save_path)
    return save_path


def load_model(path: Optional[Path] = None) -> Any:
    """Load trained model from disk."""
    load_path = path or (ARTIFACTS_DIR / MODEL_FILENAME)
    if not load_path.exists():
        raise FileNotFoundError(f"Model not found at: {load_path}. Run training first.")
    model = joblib.load(load_path)
    logger.info("Model loaded: %s", load_path)
    return model


def save_preprocessor(preprocessor: Any, path: Optional[Path] = None) -> Path:
    """Save fitted preprocessor to disk."""
    save_path = path or (ARTIFACTS_DIR / PREPROCESSOR_FILENAME)
    joblib.dump(preprocessor, save_path)
    logger.info("Preprocessor saved: %s", save_path)
    return save_path


def load_preprocessor(path: Optional[Path] = None) -> Any:
    """Load fitted preprocessor from disk."""
    load_path = path or (ARTIFACTS_DIR / PREPROCESSOR_FILENAME)
    if not load_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {load_path}. Run training first.")
    preprocessor = joblib.load(load_path)
    logger.info("Preprocessor loaded: %s", load_path)
    return preprocessor


def save_model_metadata(
    feature_names: list,
    metrics: Dict[str, Any],
    threshold: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save model metadata as JSON for audit and reproducibility.

    Includes version, training date, feature list, evaluation metrics,
    and decision threshold.
    """
    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_file": MODEL_FILENAME,
        "preprocessor_file": PREPROCESSOR_FILENAME,
        "feature_names": feature_names,
        "decision_threshold": threshold,
        "evaluation": {
            "auc_roc": metrics.get("auc_roc"),
            "ks_statistic": metrics.get("ks_statistic"),
            "gini_coefficient": metrics.get("gini_coefficient"),
            "f1_score": metrics.get("f1_score"),
        },
        **(extra or {}),
    }
    path = ARTIFACTS_DIR / METADATA_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Model metadata saved: %s", path)
    return path


def save_training_metrics(metrics: Dict[str, Any]) -> Path:
    """Save full training metrics report as JSON."""
    path = ARTIFACTS_DIR / TRAINING_METRICS_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Training metrics saved: %s", path)
    return path


def load_model_metadata() -> Dict[str, Any]:
    """Load model metadata JSON."""
    path = ARTIFACTS_DIR / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
