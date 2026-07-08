"""
Recommendation Engine for the IDBI AI module.

Consolidates signals from all upstream engines and the ML model to produce
a concrete, actionable recommendation for the loan officer.

Recommendation actions:
  APPROVE             — low risk, no rule violations, strong borrower profile
  MANUAL_REVIEW       — medium risk or soft rule violations
  REJECT              — very high risk or hard rule violations
  REDUCE_LOAN_AMOUNT  — risk reduces significantly with a smaller loan
  INCREASE_TENURE     — EMI burden is too high; longer tenure makes it manageable
  REQUEST_CO_APPLICANT— risk profile improves substantially with a co-applicant
  REQUEST_INSURANCE   — loan can be approved with mandatory insurance

The engine applies a decision matrix across risk level, business rules,
and borrower score to derive the most appropriate action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.config import DEFAULT_LGD
from ai.logger import get_logger
from ai.schemas.input_schema import BorrowerInput, LoanInput
from ai.schemas.output_schema import (
    Borrower360Output,
    BusinessRulesOutput,
    GeoResilienceOutput,
    RecommendationAction,
    RecommendationOutput,
    RiskLevel,
)
from ai.utils.financial_utils import calculate_emi, calculate_loan_to_income_ratio

logger = get_logger(__name__)


class RecommendationEngine:
    """
    Engine 4: Recommendation Engine.

    Applies a transparent rule-based decision matrix on top of ML output.
    All decision logic is documented and auditable.
    """

    def generate(
        self,
        risk_level: RiskLevel,
        default_probability: float,
        borrower_360: Borrower360Output,
        geo: GeoResilienceOutput,
        rules: BusinessRulesOutput,
        borrower: BorrowerInput,
        loan: LoanInput,
    ) -> RecommendationOutput:
        """
        Generate a recommendation based on all engine outputs.

        Args:
            risk_level: ML-derived risk level.
            default_probability: Predicted probability of default.
            borrower_360: Borrower financial profile.
            geo: Geo resilience output.
            rules: Business rules evaluation result.
            borrower: Original borrower input.
            loan: Original loan input.

        Returns:
            :class:`RecommendationOutput` with action, confidence, and reasons.
        """
        logger.info("Generating recommendation for risk_level=%s, PD=%.4f", risk_level, default_probability)

        reasons: List[str] = []
        adjustments: Optional[Dict[str, Any]] = None

        # ── Step 1: Hard reject check ──────────────────────────────────
        if rules.hard_reject:
            action = RecommendationAction.REJECT
            confidence = 0.95
            reasons.append("Hard business rule violation(s) triggered.")
            for v in rules.rule_violations:
                if v.severity.value == "HARD":
                    reasons.append(f"  • {v.message}")
            return self._build(action, confidence, reasons, adjustments)

        # ── Step 2: Very high risk → Reject ───────────────────────────
        if risk_level == RiskLevel.VERY_HIGH:
            action = RecommendationAction.REJECT
            confidence = 0.90
            reasons.append(f"Default probability is very high ({default_probability:.1%}).")
            reasons.append("ML model confidence and risk level strongly indicate default.")
            return self._build(action, confidence, reasons, adjustments)

        # ── Step 3: Low risk → Approve ────────────────────────────────
        if risk_level == RiskLevel.LOW and rules.soft_violations_count == 0:
            action = RecommendationAction.APPROVE
            confidence = min(0.90, 1.0 - default_probability)
            reasons.append(f"Default probability is low ({default_probability:.1%}).")
            reasons.append(f"Borrower score is {borrower_360.borrower_score:.1f}/100 — strong profile.")
            if geo.geo_resilience_score >= 60:
                reasons.append(f"Geographic resilience score is good ({geo.geo_resilience_score:.1f}/100).")
            return self._build(action, confidence, reasons, adjustments)

        # ── Step 4: Medium risk — suggest improvements ─────────────────
        if risk_level == RiskLevel.MEDIUM:
            action, adjustments, extra_reasons = self._suggest_medium_risk_action(
                borrower, loan, borrower_360, rules
            )
            reasons.append(f"Default probability is moderate ({default_probability:.1%}).")
            reasons.extend(extra_reasons)
            confidence = 0.70
            return self._build(action, confidence, reasons, adjustments)

        # ── Step 5: High risk — evaluate improvement options ───────────
        if risk_level == RiskLevel.HIGH:
            action, adjustments, extra_reasons = self._suggest_high_risk_action(
                borrower, loan, borrower_360, rules
            )
            reasons.append(f"Default probability is elevated ({default_probability:.1%}).")
            reasons.extend(extra_reasons)
            confidence = 0.75
            return self._build(action, confidence, reasons, adjustments)

        # ── Fallback ──────────────────────────────────────────────────
        return self._build(
            RecommendationAction.MANUAL_REVIEW,
            0.60,
            ["Risk signals are mixed. Officer review recommended."],
            None,
        )

    def _suggest_medium_risk_action(
        self,
        borrower: BorrowerInput,
        loan: LoanInput,
        b360: Borrower360Output,
        rules: BusinessRulesOutput,
    ):
        """Determine best action for medium-risk loans."""
        reasons = []

        # Check if tenure increase helps DTI
        if b360.dti_ratio > 0.50:
            extended_tenure = loan.loan_tenure_months + 24
            new_emi = calculate_emi(loan.loan_amount, loan.interest_rate, extended_tenure)
            monthly_inc = borrower.monthly_income or borrower.annual_income / 12
            new_dti = (new_emi + loan.existing_monthly_obligations) / monthly_inc
            if new_dti < 0.45:
                reasons.append(f"Increasing tenure by 24 months would reduce DTI to {new_dti:.1%}.")
                return (
                    RecommendationAction.INCREASE_TENURE,
                    {"increase_tenure_by_months": 24, "new_tenure": extended_tenure, "new_dti": round(new_dti, 4)},
                    reasons,
                )

        # Co-applicant check
        if not loan.co_applicant and b360.borrower_score < 60:
            reasons.append("A co-applicant would significantly strengthen the application.")
            return RecommendationAction.REQUEST_CO_APPLICANT, None, reasons

        # Insurance can offset risk for moderate cases
        if not loan.insurance and b360.borrower_score >= 55:
            reasons.append("Loan insurance would mitigate residual risk.")
            return RecommendationAction.REQUEST_INSURANCE, None, reasons

        if rules.soft_violations_count >= 2:
            reasons.append(f"{rules.soft_violations_count} soft rule violations require officer review.")
            return RecommendationAction.MANUAL_REVIEW, None, reasons

        return RecommendationAction.MANUAL_REVIEW, None, ["Medium risk profile requires officer review."]

    def _suggest_high_risk_action(
        self,
        borrower: BorrowerInput,
        loan: LoanInput,
        b360: Borrower360Output,
        rules: BusinessRulesOutput,
    ):
        """Determine best action for high-risk loans."""
        reasons = []
        monthly_inc = borrower.monthly_income or borrower.annual_income / 12

        # Check if reduced loan amount brings risk down
        if b360.loan_to_income_ratio > 10:
            reduced_amount = min(loan.loan_amount * 0.70, monthly_inc * 12 * 8)
            new_emi = calculate_emi(reduced_amount, loan.interest_rate, loan.loan_tenure_months)
            new_dti = (new_emi + loan.existing_monthly_obligations) / monthly_inc
            if new_dti < 0.55:
                reasons.append(f"Reducing loan by 30% would lower DTI to {new_dti:.1%}.")
                return (
                    RecommendationAction.REDUCE_LOAN_AMOUNT,
                    {"reduce_by": round(loan.loan_amount - reduced_amount, 2), "suggested_amount": round(reduced_amount, 2)},
                    reasons,
                )

        if not loan.co_applicant:
            reasons.append("Co-applicant is strongly recommended for high-risk profiles.")
            return RecommendationAction.REQUEST_CO_APPLICANT, None, reasons

        reasons.append("High default probability. Detailed officer review is mandatory.")
        return RecommendationAction.MANUAL_REVIEW, None, reasons

    @staticmethod
    def _build(
        action: RecommendationAction,
        confidence: float,
        reasons: List[str],
        adjustments: Optional[Dict[str, Any]],
    ) -> RecommendationOutput:
        return RecommendationOutput(
            action=action,
            confidence=round(confidence, 4),
            reasons=reasons,
            suggested_adjustments=adjustments,
        )
