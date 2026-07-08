# IDBI Bank AI Credit Risk Intelligence Platform

> AI-powered decision support system for IDBI Bank loan officers.  
> Predicts default risk, explains predictions, incorporates geographic risk, and generates actionable recommendations.

---

## Module Scope

This repository contains **only the AI module**. Frontend, backend, authentication, and database are out of scope.

---

## Architecture

```
ai/
├── config.py                    # Central config — all constants, paths, weights
├── logger.py                    # Structured logging
├── schemas/
│   ├── input_schema.py          # Pydantic v2 input models (InferenceRequest)
│   └── output_schema.py         # Pydantic v2 output models (RiskIntelligenceOutput)
├── data/
│   ├── loader.py                # CSV loading + column validation
│   ├── cleaner.py               # Missing values, outliers, types
│   ├── eda.py                   # EDA plots → artifacts/eda_plots/
│   └── feature_engineering.py  # DTI, EMI, repayment capacity, credit proxy
├── utils/
│   ├── financial_utils.py       # Pure financial math (EMI, DTI, EL, scores)
│   └── geo_data.py              # Static offline state-level climate risk
├── models/
│   ├── preprocessor.py          # ColumnTransformer (scale + encode)
│   ├── trainer.py               # XGBoost training + CV
│   ├── tuner.py                 # RandomizedSearchCV tuning
│   ├── evaluator.py             # AUC, KS, Gini, F1, confusion matrix
│   └── persistence.py           # Versioned joblib save/load + JSON metadata
├── explainability/
│   ├── shap_engine.py           # SHAP TreeExplainer + plots
│   └── risk_factors.py          # Factor extraction + officer narrative
├── engines/
│   ├── borrower_360.py          # Engine 1: Financial strength scoring
│   ├── geo_resilience.py        # Engine 2: Geo + climate risk
│   ├── business_rules.py        # Engine 3: Configurable banking policy rules
│   ├── recommendation.py        # Engine 4: Decision engine
│   └── risk_intelligence.py     # Engine 5: Final output composer
├── simulator/
│   └── what_if.py               # What-If scenario simulator
├── pipeline/
│   ├── train_pipeline.py        # End-to-end training orchestrator
│   └── inference_pipeline.py    # End-to-end inference orchestrator
└── tests/
    ├── test_financial_utils.py
    ├── test_feature_engineering.py
    ├── test_borrower_360.py
    ├── test_geo_resilience.py
    ├── test_business_rules.py
    ├── test_recommendation.py
    └── test_inference_pipeline.py
```

---

## Inference Pipeline Flow

```
InferenceRequest
    │
    ├─► Borrower 360° Intelligence      → DTI, EMI, borrower score, trust score
    ├─► Geo & Resilience Intelligence   → climate risk, resilience score, risk adjustment
    ├─► ML Prediction (XGBoost)         → raw default probability
    ├─► SHAP Explanation                → top risk factors + officer narrative
    ├─► Business Rules Engine           → hard/soft rule violations
    ├─► Recommendation Engine           → APPROVE / REJECT / MANUAL_REVIEW / ...
    └─► AI Risk Intelligence            → final composed RiskIntelligenceOutput
```

---

## Training Pipeline Flow

```
loan_data.csv → Load → Clean → EDA → Feature Engineering →
Preprocess → Train → Evaluate → SHAP Plots → Save Artifacts
```

---

## Dataset Setup

1. Download the **Credit Risk Dataset** from Kaggle:  
   `kaggle.com/datasets/laotse/credit-risk-dataset`

2. Place it at:
   ```
   ai/data/raw/loan_data.csv
   ```

3. The dataset must contain these columns:
   ```
   person_age, person_income, person_home_ownership, person_emp_length,
   loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status,
   loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
   ```

4. If using a different dataset, update `COLUMN_MAPPING` in `ai/config.py`.

---

## Installation

```bash
pip install -r requirements.txt
```

**Python >= 3.10 required.**

---

## Running the Training Pipeline

```bash
# Standard training (no tuning — fast)
python -m ai.pipeline.train_pipeline

# With RandomizedSearchCV hyperparameter tuning (slower, better model)
python -m ai.pipeline.train_pipeline --tune

# Custom dataset path
python -m ai.pipeline.train_pipeline --dataset /path/to/data.csv
```

**Artifacts saved to `ai/artifacts/`:**
- `model_v1.joblib`
- `preprocessor_v1.joblib`
- `model_metadata.json`
- `training_metrics.json`
- `shap_summary.png`, `shap_bar_importance.png`
- `eda_plots/` — EDA visualisations

---

## Running the Inference Pipeline

```python
from ai.pipeline.inference_pipeline import InferencePipeline
from ai.schemas.input_schema import (
    InferenceRequest, BorrowerInput, LoanInput, GeoInput,
    EmploymentType, HomeOwnership, LoanPurpose, LoanGrade
)

pipeline = InferencePipeline()   # Load once, reuse many times

request = InferenceRequest(
    request_id="LOAN-2024-001",
    borrower=BorrowerInput(
        age=38,
        annual_income=1_500_000,
        employment_type=EmploymentType.SALARIED,
        employment_length_months=84,
        home_ownership=HomeOwnership.OWN,
        credit_history_length_months=120,
    ),
    loan=LoanInput(
        loan_amount=800_000,
        loan_tenure_months=48,
        loan_purpose=LoanPurpose.PERSONAL,
        interest_rate=11.5,
        loan_grade=LoanGrade.B,
    ),
    geo=GeoInput(state="Maharashtra"),
)

result = pipeline.run(request)

print(f"Default Probability : {result.default_probability:.1%}")
print(f"Risk Level          : {result.risk_level.value}")
print(f"Expected Loss       : ₹{result.expected_loss:,.0f}")
print(f"Recommendation      : {result.recommendation.action.value}")
print(f"Summary             : {result.summary}")
```

---

## What-If Simulator

```python
from ai.simulator.what_if import WhatIfSimulator
from ai.schemas.input_schema import WhatIfOverrides

simulator = WhatIfSimulator(inference_pipeline=pipeline)

overrides = WhatIfOverrides(
    loan_amount=600_000,     # Reduce loan amount
    co_applicant=True,       # Add co-applicant
)

whatif = simulator.simulate(request, overrides, scenario_name="Reduced Loan + Co-applicant")

print(f"Base probability    : {whatif.base_scenario.default_probability:.1%}")
print(f"Modified probability: {whatif.modified_scenario.default_probability:.1%}")
print(f"Delta               : {whatif.delta_probability:+.1%}")
print(f"Insight             : {whatif.insight}")
```

---

## Running Tests

```bash
# Run all tests
python -m pytest ai/tests/ -v

# With coverage report
python -m pytest ai/tests/ -v --cov=ai --cov-report=term-missing
```

**Tests do NOT require a trained model** — the inference pipeline tests use mocks.

---

## Geo & Climate Risk

The Geo & Resilience Engine uses a **fully offline static dataset** of 35+ Indian states with flood, drought, cyclone, and heatwave risk scores derived from public NDMA/IMD classifications.

**No paid API is used anywhere in this module.**

`FUTURE_INTEGRATION` hooks are clearly marked in `geo_data.py` for when live NDMA/IMD API access becomes available.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| XGBoost | Best AUC on tabular credit data; native SHAP TreeExplainer |
| RandomizedSearchCV | No extra dependencies; efficient; scalable by increasing n_iter |
| Pydantic v2 schemas | Type-safe I/O; FastAPI-ready for future backend integration |
| Separate preprocessor | Prevents data leakage; independently serialisable and loadable |
| Offline geo risk | Zero paid APIs; FUTURE_INTEGRATION stubs for live API replacement |
| Expected Loss in `financial_utils.py` | Financial math belongs in utilities, not intelligence logic |
| Business Rules alongside ML | Rules handle policy absolutes; ML handles probabilistic risk |

---

## Product Statement

> We are building an AI-powered Credit Risk Intelligence Platform that assists IDBI Bank officers in evaluating loan applications by predicting default probability, explaining the reasons behind predictions, incorporating geographic and environmental risk factors, estimating financial exposure, and generating actionable recommendations to support faster, more transparent, and more consistent lending decisions.

---

## FUTURE_INTEGRATION Items

The following features are designed with clean interfaces but not yet implemented due to data unavailability:

| Feature | Location | Notes |
|---|---|---|
| Live CIBIL credit score | `engines/borrower_360.py` | Currently proxied via loan_grade |
| Existing loan EMI obligations | `data/feature_engineering.py` | Not in base dataset |
| Live NDMA/IMD geo API | `utils/geo_data.py` | Static lookup used |
| Marital status / dependents | `schemas/input_schema.py` | Schema present, not used in scoring |
| Credit bureau default history | `engines/borrower_360.py` | Flagged as FUTURE_INTEGRATION |
