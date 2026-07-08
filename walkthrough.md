# IDBI AI Credit Risk Intelligence Platform — Walkthrough

> **For IDBI Hackathon Judges**  
> This document explains the architecture, AI workflow, and how to run and demonstrate the project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [AI Workflow](#3-ai-workflow)
4. [Intelligence Engines](#4-intelligence-engines)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training Pipeline](#6-model-training-pipeline)
7. [SHAP Explainability](#7-shap-explainability)
8. [Business Rules Engine](#8-business-rules-engine)
9. [Recommendation Logic](#9-recommendation-logic)
10. [What-If Simulator](#10-what-if-simulator)
11. [How to Run](#11-how-to-run)
12. [How to Demo During Judging](#12-how-to-demo-during-judging)

---

## 1. Project Overview

**IDBI AI Credit Risk Intelligence Platform** is an AI-powered decision support system for IDBI Bank loan officers. It helps evaluate loan applications by:

- 🎯 **Predicting default probability** via an XGBoost ML model
- 🔍 **Explaining predictions** via SHAP feature importance
- 🌏 **Incorporating geographic risk** via Geo & Resilience Intelligence
- 💡 **Generating actionable recommendations** via a multi-signal recommendation engine
- 🔮 **Enabling scenario analysis** via the What-If Simulator

The AI **assists** human loan officers — it never replaces human judgment.

**Product Statement:**  
> *We are building an AI-powered Credit Risk Intelligence Platform that assists IDBI Bank officers in evaluating loan applications by predicting default probability, explaining the reasons behind predictions, incorporating geographic and environmental risk factors, estimating financial exposure, and generating actionable recommendations to support faster, more transparent, and more consistent lending decisions.*

---

## 2. Folder Structure

```
IDBI/
├── demo.py                        ← Hackathon demo entry point
├── walkthrough.md                 ← This document
├── PROJECT_HEALTH_REPORT.md       ← Technical assessment
├── requirements.txt               ← All dependencies
├── idbi_plann.md                  ← Project specification
├── README.md                      ← Developer README
│
└── ai/                            ← Complete AI module
    ├── config.py                  ← Centralised configuration constants
    ├── logger.py                  ← Structured logging setup
    │
    ├── schemas/
    │   ├── input_schema.py        ← Pydantic models: BorrowerInput, LoanInput, GeoInput
    │   └── output_schema.py       ← Pydantic models: RiskIntelligenceOutput, WhatIfOutput
    │
    ├── data/
    │   ├── loader.py              ← CSV loading with column aliasing and validation
    │   ├── cleaner.py             ← Missing value imputation, leakage removal, type normalisation
    │   ├── eda.py                 ← Automated EDA: distribution plots, correlation heatmaps
    │   └── feature_engineering.py ← Derived features: repayment_capacity, credit_score_normalized, etc.
    │
    ├── utils/
    │   ├── financial_utils.py     ← EMI, DTI, expected loss, borrower score calculations
    │   └── geo_data.py            ← Offline geographic risk profiles for all Indian states
    │
    ├── models/
    │   ├── preprocessor.py        ← ColumnTransformer: StandardScaler + OHE + passthrough
    │   ├── trainer.py             ← XGBoost training with early stopping + cross-validation
    │   ├── tuner.py               ← RandomizedSearchCV hyperparameter tuning
    │   ├── evaluator.py           ← AUC, Gini, KS, F1, confusion matrix metrics
    │   └── persistence.py         ← Versioned joblib save/load + JSON metadata
    │
    ├── engines/
    │   ├── borrower_360.py        ← Borrower 360° financial profile engine
    │   ├── geo_resilience.py      ← Geographic & climate risk engine
    │   ├── business_rules.py      ← Configurable banking policy rules
    │   ├── recommendation.py      ← Final decision engine (Approve/Reject/etc.)
    │   └── risk_intelligence.py   ← Orchestrates all engines into one output
    │
    ├── explainability/
    │   ├── shap_engine.py         ← SHAP TreeExplainer wrapper
    │   └── risk_factors.py        ← Human-readable SHAP output formatter
    │
    ├── pipeline/
    │   ├── train_pipeline.py      ← End-to-end training orchestrator
    │   └── inference_pipeline.py  ← End-to-end inference orchestrator
    │
    ├── simulator/
    │   └── what_if.py             ← What-If Simulator (scenario comparison)
    │
    ├── artifacts/                 ← Generated during training (not committed to Git)
    │   ├── model_v1.joblib        ← Trained XGBoost model
    │   ├── preprocessor_v1.joblib ← Fitted ColumnTransformer
    │   ├── model_metadata.json    ← Feature names, threshold, metrics
    │   ├── training_metrics.json  ← Full train/test evaluation metrics
    │   ├── shap_summary.png       ← Global SHAP beeswarm plot
    │   ├── shap_bar_importance.png← SHAP mean |value| bar chart
    │   └── eda_plots/             ← EDA visualisation outputs
    │
    ├── data/raw/
    │   └── loan_data.csv          ← Source training dataset (15,000 rows, 45 columns)
    │
    └── tests/
        ├── test_financial_utils.py    ← 20 unit tests
        ├── test_feature_engineering.py← 12 unit tests
        ├── test_borrower_360.py       ← 9 unit tests
        ├── test_geo_resilience.py     ← 14 unit tests
        ├── test_business_rules.py     ← 8 unit tests
        ├── test_inference_pipeline.py ← 11 integration tests
        └── test_recommendation.py     ← 10 unit tests
```

---

## 3. AI Workflow

### Training Pipeline

```
loan_data.csv
     │
     ▼
[1] Data Loading      → Validate schema, alias columns, log class balance
     │
     ▼
[2] Data Cleaning     → Drop leakage/ID columns, impute missing values,
                         encode binary flags (has_mortgage, etc.)
     │
     ▼
[3] EDA               → Distribution plots, correlation heatmaps (saved to artifacts/)
     │
     ▼
[4] Feature Engg.     → repayment_capacity, cash_flow_health, loan_to_income_ratio,
                         employment_stability, credit_score_normalized, near_retirement_flag,
                         high_dti_flag, delinquency_flag
     │
     ▼
[5] Train/Val/Test    → 80% trainval → 85% inner-train + 15% val
  Split                  20% held-out test set (stays pure)
     │
     ▼
[6] Preprocessing     → StandardScaler (numeric) + OHE (categorical) + passthrough (binary)
     │
     ▼
[7] XGBoost Training  → scale_pos_weight for imbalance, early stopping on inner-val
     │
     ▼
[8] Evaluation        → AUC-ROC, Gini, KS statistic, optimal F1 threshold search
     │
     ▼
[9] SHAP              → TreeExplainer for global feature importance plots
     │
     ▼
[10] Save Artifacts   → model_v1.joblib, preprocessor_v1.joblib, metadata JSON
```

### Inference Pipeline

```
InferenceRequest (BorrowerInput + LoanInput + GeoInput)
     │
     ├─► [A] Borrower 360° Engine  → Score, EMI, DTI, repayment capacity, trust score
     │
     ├─► [B] Geo Resilience Engine → Resilience score, climate impact, risk adjustment
     │
     ├─► [C] ML Preprocessing     → Build feature vector, apply ColumnTransformer.transform()
     │
     ├─► [D] XGBoost Predict      → Default probability P(default)
     │
     ├─► [E] SHAP Engine          → Per-instance SHAP values → top risk factors
     │
     ├─► [F] Business Rules Engine → Policy checks (age, DTI, employment, purpose)
     │
     ├─► [G] Recommendation Engine → Approve / Reject / Manual Review / Adjust
     │
     └─► [H] Risk Intelligence    → Compose final RiskIntelligenceOutput
```

---

## 4. Intelligence Engines

### Borrower 360° Intelligence (`engines/borrower_360.py`)

Builds a comprehensive financial portrait of the borrower.

| Output | Description |
|--------|-------------|
| `borrower_score` | Weighted composite: credit (35%) + income stability (30%) + history (20%) + age (15%) |
| `financial_health` | STRONG / MODERATE / WEAK / CRITICAL based on DTI + repayment capacity |
| `estimated_emi` | Calculated via standard EMI formula: P·r·(1+r)^n / ((1+r)^n-1) |
| `dti_ratio` | (EMI + existing obligations) / monthly income |
| `repayment_capacity` | Monthly surplus after EMI and obligations |
| `cash_flow_health` | 1 - (obligations / income), capped at [0, 1] |
| `trust_score` | Derived from credit history length, delinquency count, past defaults |

### Geo & Resilience Intelligence (`engines/geo_resilience.py`)

**This is the platform's USP (Unique Selling Point).**

- Uses an **offline state-level risk database** (`utils/geo_data.py`) covering all 36 Indian states/UTs
- Each state profile includes: flood risk, drought risk, cyclone risk, heatwave risk, economic activity type
- Computes a **Resilience Score** (0–100) based on composite weighted risk
- Assigns **Climate Impact** category: LOW / MEDIUM / HIGH / EXTREME
- Applies a **Risk Adjustment** (basis points) that influences the final recommendation

### Business Rules Engine (`engines/business_rules.py`)

Configurable rule engine that enforces banking policy constraints.

| Rule | Type | Condition |
|------|------|-----------|
| `HARD_REJECT_MIN_AGE` | HARD | Age < 21 → Auto-reject |
| `HARD_REJECT_HIGH_DTI` | HARD | DTI > 70% → Auto-reject |
| `SOFT_SHORT_EMPLOYMENT` | SOFT | Employment < 6 months → Flag |
| `SOFT_LOW_CREDIT_PROXY` | SOFT | Credit proxy score < 40 → Flag |
| `SOFT_HIGH_RISK_PURPOSE` | SOFT | Loan purpose = Vehicle → Flag |

Hard violations trigger immediate rejection. Soft violations add caution to the recommendation.

### Recommendation Engine (`engines/recommendation.py`)

Multi-signal decision engine that synthesises all engine outputs:

```
Input signals: default_probability + risk_level + geo_adjustment + business_rules + borrower_score

Decision tree:
  → Hard reject?                           → REJECT
  → P(default) < 0.35 + no soft violations → APPROVE
  → P(default) < 0.35 + co-applicant helps → REQUEST_CO_APPLICANT
  → P(default) < 0.35 + insurance helps    → REQUEST_INSURANCE  
  → P(default) 0.35–0.60                  → MANUAL_REVIEW
  → P(default) > 0.60                     → REJECT
  → Loan amount high relative to income   → REDUCE_LOAN_AMOUNT
  → Tenure extension would help           → INCREASE_TENURE
```

### AI Risk Intelligence (`engines/risk_intelligence.py`)

Orchestrates all engines and composes the final `RiskIntelligenceOutput`:
- Combines ML prediction with all engine outputs
- Calculates Expected Loss = P(default) × LGD × Loan Amount
- Generates a human-readable AI summary text

---

## 5. Feature Engineering

Eight derived features added to each training sample and each inference request:

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| `repayment_capacity` | `monthly_income − emi_amount − existing_emi` | Monthly surplus cash |
| `cash_flow_health` | `1 − (emi_amount + existing_emi) / monthly_income`, clipped [0,1] | Cash flow quality (0=stressed, 1=healthy) |
| `loan_to_income_ratio` | `loan_amount / annual_income` | Leverage relative to income |
| `employment_stability` | `employment_length / 240`, clipped [0,1] | Job tenure normalised to 20-year scale |
| `credit_score_normalized` | `(credit_score − 300) / 600`, clipped [0,1] | CIBIL score on 0–1 scale |
| `near_retirement_flag` | `1 if age > 58` | Binary flag for near-retirement risk |
| `high_dti_flag` | `1 if dti_ratio > 0.50` | Binary flag for high debt burden |
| `delinquency_flag` | `1 if delinquency_count > 0` | Binary flag for any past delinquency |

---

## 6. Model Training

### Algorithm: XGBoost (`XGBClassifier`)

**Why XGBoost?**
- Consistently top AUC on tabular credit datasets
- Native SHAP `TreeExplainer` support (10–100× faster than KernelExplainer)
- Built-in handling for missing values
- Proven in production banking systems worldwide

### Preprocessing

A three-group `ColumnTransformer` is applied:

| Group | Transformer | Columns |
|-------|-------------|---------|
| Numeric (23 cols) | `StandardScaler` | age, income, credit_score, dti_ratio, loan_amount, ... |
| Nominal (13 cols) | `OneHotEncoder(drop='first')` | employment_type, loan_purpose, state, branch_region, ... |
| Binary (7 cols) | Passthrough | has_mortgage, past_default_flag, near_retirement_flag, ... |

### Training Configuration

- **Scale pos weight**: auto-computed as `n_negative / n_positive`
- **Early stopping**: 50 rounds patience on inner validation set
- **Data splits**: 68% inner-train / 12% inner-val / 20% test (stratified)
- **Artifacts**: versioned as `model_v1.joblib`

---

## 7. SHAP Explainability

SHAP (SHapley Additive exPlanations) provides **per-prediction, per-feature** explanations:

- Each feature receives a SHAP value indicating how much it shifted the default probability up or down
- `shap_value > 0` → Feature **increases** default probability (raises risk)
- `shap_value < 0` → Feature **decreases** default probability (lowers risk)
- Top-N features are surfaced to the loan officer as human-readable risk factors

Generated outputs:
- `ai/artifacts/shap_summary.png` — global beeswarm importance plot
- `ai/artifacts/shap_bar_importance.png` — mean |SHAP value| bar chart
- Per-request SHAP values embedded in `RiskIntelligenceOutput.shap_explanation`

---

## 8. Business Rules Engine

The `BusinessRulesEngine` is separate from the ML model and provides:

- **Hard rules**: Absolute constraints that override the ML model (regulatory compliance)
- **Soft rules**: Advisory flags that inform the recommendation but don't auto-reject
- **Configurable**: Rules and thresholds are passed as a dict at instantiation time
- **Transparent**: Every violated rule produces a named `RuleViolation` with a human reason

This separation allows the bank's compliance team to update rules without retraining the model.

---

## 9. Recommendation Logic

The `RecommendationEngine` generates a final lending action:

| Action | Trigger |
|--------|---------|
| `APPROVE` | P(default) < threshold, no violations, clean profile |
| `REQUEST_CO_APPLICANT` | Borderline risk, co-applicant present could help |
| `REQUEST_INSURANCE` | Moderate risk, insurance coverage recommended |
| `REDUCE_LOAN_AMOUNT` | Loan-to-income ratio too high |
| `INCREASE_TENURE` | EMI too high for income; longer tenure reduces EMI |
| `MANUAL_REVIEW` | Ambiguous risk; human officer required |
| `REJECT` | Hard rule violated or P(default) > 0.60 |

A `confidence` score (0–1) and `reasons` list are generated with every recommendation.

---

## 10. What-If Simulator

The `WhatIfSimulator` allows loan officers to model "what happens if I change X?":

Supported overrides:
- `loan_amount` — reduce/increase the requested loan
- `loan_tenure_months` — extend or shorten the repayment period
- `co_applicant` — add/remove a co-borrower
- `insurance` — add/remove loan insurance
- `interest_rate` — test at a different rate

The simulator runs the **full inference pipeline twice** (base + modified) and outputs:
- Side-by-side probability, expected loss, risk level, and recommendation
- A delta analysis: `ΔP(default)` and `ΔExpected Loss`
- A plain-English insight for the officer

---

## 11. How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place the dataset
```
ai/data/raw/loan_data.csv    (15,000 rows, 45 columns)
```

### Step 3: Train the model
```bash
python -m ai.pipeline.train_pipeline
```
This will:
- Load and clean the dataset
- Run EDA and save plots to `ai/artifacts/eda_plots/`
- Engineer features
- Train XGBoost with proper train/val/test splits
- Evaluate and log metrics (AUC-ROC, Gini, KS)
- Save `model_v1.joblib`, `preprocessor_v1.joblib`, `model_metadata.json`

### Step 4: Run the demo
```bash
python demo.py
```

### Step 5: Run all tests
```bash
python -m pytest ai/tests/ -v
```

### Optional: Run hyperparameter tuning
```bash
python -m ai.pipeline.train_pipeline --tune
# (or edit TrainingPipeline(run_tuning=True) in train_pipeline.py)
```

---

## 12. How to Demo During Judging

### Recommended script for judges:

**1. Start with the architecture** (30 seconds)
> "We built a modular AI system with three intelligence layers: Borrower 360°, Geo & Resilience, and ML Risk Prediction. Each layer is independent, testable, and pluggable."

**2. Run the demo live**
```bash
python demo.py
```
Walk through each section:
- **Borrower 360°**: "The AI has computed a borrower score of X, financial health is STRONG, and the repayment capacity is ₹28,500/month."
- **Geo & Resilience**: "Maharashtra has a resilience score of 62. The system automatically enriches the application with climate risk data — flood, drought, and cyclone risk — without the officer needing to look it up."
- **SHAP Explanation**: "The AI doesn't just give a number — it explains *why*. These are the top factors driving the risk score, ranked by their impact."
- **Business Rules**: "The compliance team's rules are enforced separately from the model. This is important for regulatory auditing."
- **What-If Simulation**: "Watch what happens when we add a co-applicant and reduce the loan amount. The expected loss drops by ₹1.2L."

**3. Highlight key differentiators**
- Geo & Resilience is the USP — no other system incorporates state-level climate risk
- SHAP explainability ensures regulatory transparency (RBI AI guidelines)
- What-If Simulator supports pre-approval negotiation
- Batch Dataset Evaluation UI allows running inferences on multiple applications via CSV upload
- Business Rules Engine is configurable without retraining

**4. Discuss production readiness**
- 84 unit + integration tests
- Pydantic v2 schema validation on every request
- Structured logging with timestamps
- Clean modular architecture ready for FastAPI wrapping
