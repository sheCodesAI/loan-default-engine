"""
Central configuration for the IDBI AI module.

All paths, constants, hyperparameter search spaces, scoring weights,
and business rule thresholds are defined here. Import from this module
instead of hard-coding values across files.

Dataset: Custom IDBI loan dataset (loan_data.csv)
  - 15,001 rows, 45 columns
  - Target: loan_default (0 = no default, 1 = default)
"""

from pathlib import Path
from typing import Any, Dict, List

# ─── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent          # ai/
PROJECT_DIR: Path = BASE_DIR.parent                        # IDBI/
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
EDA_PLOTS_DIR: Path = ARTIFACTS_DIR / "eda_plots"
LOG_DIR: Path = ARTIFACTS_DIR / "logs"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, ARTIFACTS_DIR, EDA_PLOTS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_FILENAME: str = "loan_data.csv"
DATASET_PATH: Path = DATA_RAW_DIR / DATASET_FILENAME
TARGET_COLUMN: str = "loan_default"

# Column mapping: internal_canonical → csv_column_name
# The loader builds the reverse map to rename CSV columns → canonical names.
COLUMN_MAPPING: Dict[str, str] = {
    "income":                   "annual_income",
    "employment_length":        "employment_length_months",
    "loan_tenure":              "loan_term_months",
    "dti_ratio":                "debt_to_income",
    "credit_history_length":    "credit_history_length_months",
    # Identical names — listed for documentation only
    "age":                      "age",
    "gender":                   "gender",
    "education":                "education",
    "marital_status":           "marital_status",
    "employment_type":          "employment_type",
    "monthly_income":           "monthly_income",
    "existing_emi":             "existing_emi",
    "credit_score":             "credit_score",
    "delinquency_count":        "delinquency_count",
    "past_default_flag":        "past_default_flag",
    "loan_amount":              "loan_amount",
    "interest_rate":            "interest_rate",
    "emi_amount":               "emi_amount",
    "loan_purpose":             "loan_purpose",
    "loan_product":             "loan_product",
    "loan_source_type":         "loan_source_type",
    "state":                    "state",
    "district":                 "district",
    "urban_rural_flag":         "urban_rural_flag",
    "branch_region":            "branch_region",
    "service_area_cluster":     "service_area_cluster",
    "distance_to_branch_km":    "distance_to_branch_km",
    "area_default_rate":        "area_default_rate",
    "district_risk_score":      "district_risk_score",
    "state_risk_score":         "state_risk_score",
    "population_density_band":  "population_density_band",
    "economic_activity_type":   "economic_activity_type",
    "channel":                  "channel",
    "has_mortgage":             "has_mortgage",
    "has_dependents":           "has_dependents",
    "has_cosigner":             "has_cosigner",
}

# Columns to DROP before training to prevent data leakage and remove IDs
DROP_COLUMNS: List[str] = [
    "loan_id",               # Identifier
    "customer_id",           # Identifier
    "disbursal_date",        # Temporal — not useful for cross-sectional model
    "first_emi_default_flag",  # Near-target leakage (directly signals default)
    "repayment_status",      # Can reveal outcome post-disbursement
    "pincode",               # Too-high cardinality (thousands of unique values)
    "branch_code",           # Too-high cardinality
    "city",                  # High cardinality; state/district capture location
    "district",              # High cardinality; state + district_risk_score sufficient
]

# ─── Model Artifact Versioning ───────────────────────────────────────────────
MODEL_VERSION: str = "v1"
MODEL_FILENAME: str = f"model_{MODEL_VERSION}.joblib"
PREPROCESSOR_FILENAME: str = f"preprocessor_{MODEL_VERSION}.joblib"
METADATA_FILENAME: str = "model_metadata.json"
TRAINING_METRICS_FILENAME: str = "training_metrics.json"

# ─── Train / Test Split ──────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
CV_FOLDS: int = 5

# ─── Decision Threshold ──────────────────────────────────────────────────────
DEFAULT_PROBABILITY_THRESHOLD: float = 0.50
MIN_ACCEPTABLE_AUC: float = 0.70

# ─── Risk Level Bands ────────────────────────────────────────────────────────
RISK_LEVELS: Dict[str, tuple] = {
    "LOW":       (0.00, 0.30),
    "MEDIUM":    (0.30, 0.55),
    "HIGH":      (0.55, 0.75),
    "VERY_HIGH": (0.75, 1.01),
}

# ─── Borrower Scoring Weights (must sum to 1.0) ───────────────────────────────
BORROWER_SCORE_WEIGHTS: Dict[str, float] = {
    "credit_score":         0.35,
    "dti_ratio":            0.25,
    "repayment_capacity":   0.20,
    "employment_stability": 0.10,
    "cash_flow_health":     0.10,
}

# ─── Business Rules Configuration ────────────────────────────────────────────
BUSINESS_RULES_CONFIG: Dict[str, Any] = {
    # Hard reject — immediate reject
    "hard_reject_credit_score": 500,
    "hard_reject_dti": 0.85,
    # Soft rule thresholds — manual review
    "min_credit_score": 600,
    "max_dti_ratio": 0.65,
    "min_age": 21,
    "max_age": 70,
    "max_loan_to_income_ratio": 20.0,
    "min_employment_months": 6,
    "high_risk_purposes": ["Vehicle"],
}

# ─── Financial Defaults ──────────────────────────────────────────────────────
DEFAULT_LGD: float = 0.45
MONTHS_PER_YEAR: int = 12

# ─── Legacy — kept for inference-time use when loan_grade is provided ─────────
LOAN_GRADE_ORDER: List[str] = ["A", "B", "C", "D", "E", "F", "G"]
LOAN_GRADE_TO_PROXY: Dict[str, int] = {g: i for i, g in enumerate(LOAN_GRADE_ORDER)}

# ─── Hyperparameter Search Space ─────────────────────────────────────────────
RANDOMIZED_SEARCH_N_ITER: int = 50
RANDOMIZED_SEARCH_CV: int = 5
RANDOMIZED_SEARCH_SCORING: str = "roc_auc"

XGBOOST_PARAM_DIST: Dict[str, Any] = {
    "n_estimators":      [100, 200, 300, 400, 500],
    "max_depth":         [3, 4, 5, 6, 7, 8],
    "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample":         [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             [0, 0.1, 0.2, 0.3],
    "reg_alpha":         [0, 0.01, 0.1, 0.5],
    "reg_lambda":        [0.5, 1.0, 1.5, 2.0],
    "scale_pos_weight":  [1, 2, 3, 5],
}

# ─── SHAP ────────────────────────────────────────────────────────────────────
SHAP_TOP_N_FEATURES: int = 10
SHAP_PLOT_MAX_DISPLAY: int = 20

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
