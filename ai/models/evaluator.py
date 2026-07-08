"""
Model evaluation utilities for the IDBI AI Credit Risk Intelligence Platform.

Computes banking-specific credit risk metrics:
  - AUC-ROC          — overall discrimination power
  - Gini Coefficient — standard risk ranking metric (2 * AUC - 1)
  - KS Statistic     — maximum separation between defaults and non-defaults
  - Average Precision— precision-recall area (good for imbalanced data)
  - Precision, Recall, F1
  - Confusion Matrix
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ai.config import MIN_ACCEPTABLE_AUC
from ai.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    dataset_label: str = "test",
) -> Dict[str, any]:
    """
    Compute comprehensive credit risk classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted default probabilities.
        threshold: Decision threshold for class prediction.
        dataset_label: Label name for logging (e.g., 'train', 'test').

    Returns:
        Dict containing all evaluation metrics.
    """
    logger.info("Evaluating model on '%s' dataset...", dataset_label)

    # Class predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Standard classification metrics
    auc_roc = float(roc_auc_score(y_true, y_prob))
    gini = 2.0 * auc_roc - 1.0
    avg_precision = float(average_precision_score(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # KS Statistic calculation
    # Method 1: max difference between TPR and FPR across all thresholds
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = float(np.max(tpr - fpr))

    # Log warning if AUC falls below policy threshold
    if auc_roc < MIN_ACCEPTABLE_AUC:
        logger.warning(
            "AUC-ROC (%.4f) on '%s' is below the minimum policy threshold of %.2f!",
            auc_roc, dataset_label, MIN_ACCEPTABLE_AUC,
        )

    metrics = {
        "dataset": dataset_label,
        "auc_roc": auc_roc,
        "gini_coefficient": gini,
        "ks_statistic": ks_stat,
        "average_precision": avg_precision,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximizes a specific metric.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted default probabilities.
        metric: Metric to maximize ('f1' or 'ks').

    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_score = -1.0
    best_threshold = 0.50

    if metric == "f1":
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = t
    elif metric == "ks":
        # KS is maximized where TPR - FPR is maximum
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
        idx = np.argmax(tpr - fpr)
        best_threshold = float(thresholds_roc[idx])
        # clip threshold to reasonable ranges
        best_threshold = max(min(best_threshold, 0.95), 0.05)
        best_score = float(tpr[idx] - fpr[idx])
    else:
        logger.warning("Unknown metric '%s' for threshold tuning. Defaulting to 0.50", metric)

    logger.info(
        "Optimal threshold search complete. Metric: %s | Threshold: %.4f | Score: %.4f",
        metric, best_threshold, best_score,
    )
    return best_threshold, best_score


def generate_metrics_report(metrics: Dict[str, any]) -> str:
    """
    Generate a formatted CLI report summarizing the model metrics.

    Args:
        metrics: Dictionary returned by evaluate_model().

    Returns:
        Formatted multi-line report string.
    """
    cm = metrics["confusion_matrix"]
    total = sum(cm.values())

    report = [
        f"============================================================",
        f"  IDBI Credit Risk Model Metrics — {metrics['dataset'].upper()}",
        f"============================================================",
        f"  AUC-ROC            : {metrics['auc_roc']:.4f}",
        f"  Gini Coefficient   : {metrics['gini_coefficient']:.4f}",
        f"  KS Statistic       : {metrics['ks_statistic']:.4f}",
        f"  Average Precision  : {metrics['average_precision']:.4f}",
        f"  Precision          : {metrics['precision']:.4f}",
        f"  Recall             : {metrics['recall']:.4f}",
        f"  F1 Score           : {metrics['f1_score']:.4f}",
        f"------------------------------------------------------------",
        f"  Confusion Matrix (Total Samples: {total}):",
        f"    Actual Defaults (1)  | TP: {cm['true_positive']:<6} | FN: {cm['false_negative']:<6}",
        f"    Actual Non-Defs (0)  | FP: {cm['false_positive']:<6} | TN: {cm['true_negative']:<6}",
        f"============================================================",
    ]
    return "\n".join(report)
