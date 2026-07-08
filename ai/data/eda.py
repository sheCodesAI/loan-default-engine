"""
Exploratory Data Analysis (EDA) module for the IDBI AI module.

Generates statistical summaries and plots saved to artifacts/eda_plots/.
Run once after data cleaning, before feature engineering.

All plots are saved as PNG files; no interactive display is required
(supports headless server environments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe on headless servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai.config import EDA_PLOTS_DIR, TARGET_COLUMN
from ai.logger import get_logger

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Run the full EDA suite and save all plots + stats.

    Args:
        df: Cleaned DataFrame with canonical column names.
        output_dir: Directory to save plots. Defaults to config.EDA_PLOTS_DIR.
    """
    out = output_dir or EDA_PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Running EDA. Output directory: %s", out)

    generate_summary_stats(df, out)
    plot_class_distribution(df, out)
    plot_correlation_heatmap(df, out)
    plot_numeric_distributions(df, out)
    plot_categorical_distributions(df, out)
    plot_loan_amount_by_target(df, out)

    logger.info("EDA complete. All plots saved to: %s", out)


def generate_summary_stats(df: pd.DataFrame, out: Path) -> None:
    """Save descriptive statistics to a CSV file."""
    stats = df.describe(include="all").T
    stats_path = out / "summary_stats.csv"
    stats.to_csv(stats_path)
    logger.info("Summary stats saved: %s", stats_path)


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of target class distribution (default vs non-default)."""
    if TARGET_COLUMN not in df.columns:
        logger.warning("Target column '%s' not found. Skipping class distribution plot.", TARGET_COLUMN)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Class Distribution", fontsize=14, fontweight="bold")

    counts = df[TARGET_COLUMN].value_counts()
    labels = ["No Default (0)", "Default (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Absolute Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Percentage Split")

    plt.tight_layout()
    _save(fig, out / "class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Correlation heatmap for all numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    corr = numeric_df.corr()
    n = len(corr.columns)
    figsize = (max(10, n), max(8, n - 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="black" if abs(corr.values[i, j]) < 0.7 else "white",
            )

    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, out / "correlation_heatmap.png")


def plot_numeric_distributions(df: pd.DataFrame, out: Path) -> None:
    """Histogram + KDE for each numeric feature split by target class."""
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != TARGET_COLUMN
    ]
    if not numeric_cols:
        return

    ncols = 3
    nrows = (len(numeric_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()

    colors = {0: "#2ecc71", 1: "#e74c3c"}
    target_labels = {0: "No Default", 1: "Default"}

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label in df[TARGET_COLUMN].unique() if TARGET_COLUMN in df.columns else [None]:
            subset = df[df[TARGET_COLUMN] == label] if label is not None else df
            ax.hist(
                subset[col].dropna(),
                bins=40,
                alpha=0.6,
                color=colors.get(label, "#3498db"),
                label=target_labels.get(label, str(label)),
                density=True,
            )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Target Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "numeric_distributions.png")


def plot_categorical_distributions(df: pd.DataFrame, out: Path) -> None:
    """Stacked bar charts for categorical features vs target."""
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != TARGET_COLUMN
    ]
    if not cat_cols or TARGET_COLUMN not in df.columns:
        return

    ncols = 2
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ct = pd.crosstab(df[col], df[TARGET_COLUMN], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
        ax.set_title(f"{col} vs Default Rate", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Percentage (%)")
        ax.legend(["No Default", "Default"])
        ax.tick_params(axis="x", rotation=45)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Features vs Default Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "categorical_distributions.png")


def plot_loan_amount_by_target(df: pd.DataFrame, out: Path) -> None:
    """Box plot of loan_amount split by default status."""
    if "loan_amount" not in df.columns or TARGET_COLUMN not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        df[df[TARGET_COLUMN] == 0]["loan_amount"].dropna(),
        df[df[TARGET_COLUMN] == 1]["loan_amount"].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, labels=["No Default", "Default"])
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_title("Loan Amount Distribution by Default Status", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loan Amount (INR)")
    plt.tight_layout()
    _save(fig, out / "loan_amount_by_target.png")


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close to free memory."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Plot saved: %s", path)
