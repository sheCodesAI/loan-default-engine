"""
AI Risk Intelligence Engine — final output composer.

This is the last step in the inference pipeline. It assembles outputs
from all upstream engines into a single RiskIntelligenceOutput.

Responsibilities:
  1. Determine risk level from default probability
  2. Apply geo risk adjustment to the probability
  3. Compute Expected Loss (via financial_utils)
  4. Set early warning indicator
  5. Compose the final structured output
  6. Generate a human-readable summary paragraph
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ai.config import RISK_LEVELS
from ai.logger import get_logger
from ai.schemas.input_schema import BorrowerInput, LoanInput
from ai.schemas.output_schema import (
    Borrower360Output,
    BusinessRulesOutput,
    GeoResilienceOutput,
    RecommendationOutput,
    RiskIntelligenceOutput,
    RiskLevel,
    SHAPOutput,
)
from ai.utils.financial_utils import calculate_expected_loss

logger = get_logger(__name__)

# Early warning: co-occurring risk factors that together signal elevated concern
EWI_THRESHOLD_SOFT_VIOLATIONS = 2
EWI_THRESHOLD_DTI = 0.55
EWI_THRESHOLD_GEO_RISK = 0.50


class RiskIntelligenceEngine:
    """
    Engine 5: AI Risk Intelligence — final output composer.

    Stateless. Called once per inference request as the final step.
    """

    def compose(
        self,
        default_probability: float,
        borrower_360: Borrower360Output,
        geo: GeoResilienceOutput,
        rules: BusinessRulesOutput,
        shap_output: SHAPOutput,
        recommendation: RecommendationOutput,
        loan: LoanInput,
        request_id: Optional[str] = None,
    ) -> RiskIntelligenceOutput:
        """
        Compose the final RiskIntelligenceOutput.

        Args:
            default_probability: Raw ML predicted probability (pre-geo-adjustment).
            borrower_360: Borrower financial profile.
            geo: Geo resilience output.
            rules: Business rules output.
            shap_output: SHAP explanation output.
            recommendation: Recommendation engine output.
            loan: Loan input (for EL calculation).
            request_id: Optional audit trail ID.

        Returns:
            :class:`RiskIntelligenceOutput` — the single final API response object.
        """
        # ── Geo-adjusted default probability ───────────────────────────
        adjusted_probability = min(
            default_probability * geo.risk_adjustment, 1.0
        )
        adjusted_probability = round(adjusted_probability, 4)

        logger.info(
            "Risk Intelligence — raw_prob=%.4f, geo_adj=%.4f, adjusted_prob=%.4f",
            default_probability, geo.risk_adjustment, adjusted_probability,
        )

        # ── Risk level classification ───────────────────────────────────
        risk_level = classify_risk_level(adjusted_probability)

        # ── Expected Loss (PD × LGD × EAD) ─────────────────────────────
        expected_loss = calculate_expected_loss(
            default_probability=adjusted_probability,
            loan_amount=loan.loan_amount,
        )

        # ── Early Warning Indicator ─────────────────────────────────────
        ewi = _compute_ewi(borrower_360, geo, rules, adjusted_probability)

        # ── Summary paragraph ───────────────────────────────────────────
        summary = _generate_summary(
            risk_level=risk_level,
            adjusted_probability=adjusted_probability,
            expected_loss=expected_loss,
            borrower_360=borrower_360,
            geo=geo,
            recommendation=recommendation,
            ewi=ewi,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        return RiskIntelligenceOutput(
            request_id=request_id,
            timestamp=timestamp,
            default_probability=adjusted_probability,
            risk_level=risk_level,
            expected_loss=expected_loss,
            early_warning_indicator=ewi,
            borrower_360=borrower_360,
            geo_resilience=geo,
            business_rules=rules,
            shap_explanation=shap_output,
            recommendation=recommendation,
            summary=summary,
        )


def classify_risk_level(probability: float) -> RiskLevel:
    """
    Classify default probability into a RiskLevel enum.

    Args:
        probability: Adjusted default probability (0–1).

    Returns:
        :class:`RiskLevel` enum value.
    """
    for level, (low, high) in RISK_LEVELS.items():
        if low <= probability < high:
            return RiskLevel(level)
    return RiskLevel.VERY_HIGH


def _compute_ewi(
    b360: Borrower360Output,
    geo: GeoResilienceOutput,
    rules: BusinessRulesOutput,
    probability: float,
) -> bool:
    """
    Compute the Early Warning Indicator.

    Returns True when multiple risk signals co-occur:
      - High DTI AND elevated geo risk
      - Multiple soft violations AND poor borrower score
      - High probability AND high geo climate risk
    """
    signals = 0
    if b360.dti_ratio > EWI_THRESHOLD_DTI:
        signals += 1
    if geo.composite_climate_risk > EWI_THRESHOLD_GEO_RISK:
        signals += 1
    if rules.soft_violations_count >= EWI_THRESHOLD_SOFT_VIOLATIONS:
        signals += 1
    if b360.borrower_score < 40:
        signals += 1
    if probability > 0.45:
        signals += 1

    ewi = signals >= 3
    if ewi:
        logger.warning("Early Warning Indicator TRIGGERED — %d risk signals co-occurring.", signals)
    return ewi


def _generate_summary(
    risk_level: RiskLevel,
    adjusted_probability: float,
    expected_loss: float,
    borrower_360: Borrower360Output,
    geo: GeoResilienceOutput,
    recommendation: RecommendationOutput,
    ewi: bool,
) -> str:
    """Generate a concise officer-facing summary paragraph."""
    pct = round(adjusted_probability * 100, 1)
    el_lakh = round(expected_loss / 100_000, 2)

    lines = [
        f"The AI model estimates a {pct}% probability of default "
        f"(Risk Level: {risk_level.value}), with an expected loss of ₹{el_lakh}L.",
        f"Borrower financial score is {borrower_360.borrower_score:.1f}/100 "
        f"(Health: {borrower_360.financial_health.value}) "
        f"with a DTI ratio of {borrower_360.dti_ratio:.1%}.",
        f"Geographic resilience for {geo.state} is {geo.geo_resilience_score:.1f}/100 "
        f"(Climate Impact: {geo.climate_impact.value}).",
        f"Recommendation: {recommendation.action.value}.",
    ]

    if ewi:
        lines.append(
            "⚠ EARLY WARNING: Multiple risk signals are co-occurring. "
            "Senior credit officer review is strongly advised."
        )

    return " ".join(lines)
