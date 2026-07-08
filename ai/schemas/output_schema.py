"""
Output schemas for the IDBI AI inference pipeline.

All engine outputs are typed Pydantic v2 models. The final
RiskIntelligenceOutput is the single object returned to the API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Shared Enumerations ──────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class FinancialHealth(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    CRITICAL = "CRITICAL"


class ClimateImpact(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class RecoveryCapacity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RuleSeverity(str, Enum):
    SOFT = "SOFT"    # Flags for manual review
    HARD = "HARD"    # Triggers immediate reject


class RecommendationAction(str, Enum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"
    REDUCE_LOAN_AMOUNT = "REDUCE_LOAN_AMOUNT"
    INCREASE_TENURE = "INCREASE_TENURE"
    REQUEST_CO_APPLICANT = "REQUEST_CO_APPLICANT"
    REQUEST_INSURANCE = "REQUEST_INSURANCE"


# ─── Engine 1 Output ─────────────────────────────────────────────────────────

class Borrower360Output(BaseModel):
    """Output of the Borrower 360° Intelligence Engine."""

    borrower_score: float = Field(..., ge=0, le=100, description="Composite score 0–100")
    dti_ratio: float = Field(..., description="Debt-to-Income ratio")
    repayment_capacity: float = Field(..., description="Monthly disposable income after EMI (INR)")
    cash_flow_health: float = Field(..., ge=0, description="Income-to-obligation ratio")
    trust_score: float = Field(..., ge=0, le=100, description="Creditworthiness score 0–100")
    financial_health: FinancialHealth
    estimated_emi: float = Field(..., description="Estimated monthly EMI for this loan (INR)")
    disposable_income: float = Field(..., description="Monthly income after all obligations (INR)")
    loan_to_income_ratio: float = Field(..., description="loan_amount / annual_income")


# ─── Engine 2 Output ─────────────────────────────────────────────────────────

class GeoResilienceOutput(BaseModel):
    """Output of the Geo & Resilience Intelligence Engine."""

    state: str
    geo_resilience_score: float = Field(..., ge=0, le=100, description="Higher = more resilient")
    flood_risk: float = Field(..., ge=0, le=1)
    drought_risk: float = Field(..., ge=0, le=1)
    cyclone_risk: float = Field(..., ge=0, le=1)
    heatwave_risk: float = Field(..., ge=0, le=1)
    composite_climate_risk: float = Field(..., ge=0, le=1, description="Weighted composite of all risks")
    climate_impact: ClimateImpact
    recovery_capacity: RecoveryCapacity
    risk_adjustment: float = Field(..., description="Multiplier applied to final risk score (1.0 = neutral)")
    data_source: str = Field(default="STATIC_OFFLINE", description="STATIC_OFFLINE | FUTURE_INTEGRATION")


# ─── Business Rules Output ───────────────────────────────────────────────────

class RuleViolation(BaseModel):
    """A single business rule that was violated."""

    rule_name: str
    severity: RuleSeverity
    message: str
    actual_value: Any
    threshold_value: Any


class BusinessRulesOutput(BaseModel):
    """Output of the Business Rules Engine."""

    hard_reject: bool = Field(..., description="True if any HARD rule is violated → immediate reject")
    rule_violations: List[RuleViolation] = Field(default_factory=list)
    soft_violations_count: int = 0
    hard_violations_count: int = 0
    override_flags: List[str] = Field(
        default_factory=list,
        description="Labels for conditions that may warrant officer override",
    )


# ─── SHAP / Explainability Output ─────────────────────────────────────────────

class RiskFactor(BaseModel):
    """A single SHAP-derived risk factor."""

    feature: str
    shap_value: float
    direction: str = Field(..., description="INCREASES_RISK | DECREASES_RISK")
    magnitude: str = Field(..., description="HIGH | MEDIUM | LOW")
    human_readable_name: str
    description: str


class SHAPOutput(BaseModel):
    """Output of the SHAP Explainability Engine."""

    top_risk_factors: List[RiskFactor]
    base_value: float = Field(..., description="Model's expected output (log-odds)")
    predicted_probability: float
    narrative: str = Field(..., description="Plain-English explanation for the officer")


# ─── Recommendation Output ───────────────────────────────────────────────────

class RecommendationOutput(BaseModel):
    """Output of the Recommendation Engine."""

    action: RecommendationAction
    confidence: float = Field(..., ge=0, le=1)
    reasons: List[str]
    suggested_adjustments: Optional[Dict[str, Any]] = Field(
        None, description="e.g., {'reduce_loan_by': 200000} or {'increase_tenure_by': 12}"
    )


# ─── Final Composite Output ───────────────────────────────────────────────────

class RiskIntelligenceOutput(BaseModel):
    """
    Final output of the AI Risk Intelligence Engine.

    This is the single object returned to the API / backend layer.
    """

    request_id: Optional[str] = None
    timestamp: str

    # Core prediction
    default_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    expected_loss: float = Field(..., description="EL = PD × LGD × EAD in INR")
    early_warning_indicator: bool = Field(
        ..., description="True when multiple risk signals co-occur"
    )

    # Engine outputs
    borrower_360: Borrower360Output
    geo_resilience: GeoResilienceOutput
    business_rules: BusinessRulesOutput
    shap_explanation: SHAPOutput
    recommendation: RecommendationOutput

    # Officer-facing summary
    summary: str = Field(..., description="One-paragraph plain-English decision summary")


# ─── What-If Simulator Output ────────────────────────────────────────────────

class WhatIfScenario(BaseModel):
    """A single scenario result inside a What-If simulation."""

    scenario_name: str
    overrides_applied: Dict[str, Any]
    default_probability: float
    risk_level: RiskLevel
    expected_loss: float
    recommendation: RecommendationAction
    borrower_score: float


class WhatIfOutput(BaseModel):
    """Output of the What-If Simulator."""

    base_scenario: WhatIfScenario
    modified_scenario: WhatIfScenario
    delta_probability: float = Field(..., description="modified_prob - base_prob")
    delta_expected_loss: float
    risk_level_changed: bool
    recommendation_changed: bool
    insight: str = Field(..., description="Plain-English insight for the officer")
