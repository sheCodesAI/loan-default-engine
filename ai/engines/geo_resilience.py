"""
Geo & Resilience Intelligence Engine.

Enriches the borrower's location with climate and disaster risk data
using a fully offline static lookup (no paid APIs required).

Outputs GeoResilienceOutput containing:
  - Flood / drought / cyclone / heatwave risk scores (0–1)
  - Composite climate risk (0–1)
  - Geo resilience score (0–100, higher = more resilient)
  - Climate impact classification (LOW/MODERATE/HIGH/SEVERE)
  - Recovery capacity (LOW/MEDIUM/HIGH)
  - Risk adjustment multiplier for the final probability

FUTURE_INTEGRATION: Replace static lookup with live NDMA/IMD API calls.
The interface (compute method) is stable and will not change.
"""

from __future__ import annotations

from ai.logger import get_logger
from ai.schemas.input_schema import BorrowerInput, GeoInput
from ai.schemas.output_schema import ClimateImpact, GeoResilienceOutput, RecoveryCapacity
from ai.utils.geo_data import (
    get_composite_climate_risk,
    get_geo_resilience_score,
    get_risk_adjustment_multiplier,
    get_state_risk_profile,
)

logger = get_logger(__name__)

# Climate impact classification thresholds
_CLIMATE_IMPACT_BANDS = {
    ClimateImpact.LOW: (0.00, 0.25),
    ClimateImpact.MODERATE: (0.25, 0.45),
    ClimateImpact.HIGH: (0.45, 0.65),
    ClimateImpact.SEVERE: (0.65, 1.01),
}


class GeoResilienceEngine:
    """
    Engine 2: Geo & Resilience Intelligence.

    Stateless — call compute() for each inference request.
    """

    def compute(
        self,
        geo: GeoInput,
        borrower: BorrowerInput,
        loan_purpose: str = "",
    ) -> GeoResilienceOutput:
        """
        Compute the Geo & Resilience profile for the borrower's location.

        Args:
            geo: Validated geographic input (state, district, pin_code).
            borrower: Borrower input (used for occupation-based risk amplification).
            loan_purpose: Loan purpose string (e.g., "AGRICULTURE").

        Returns:
            :class:`GeoResilienceOutput` with all risk scores.
        """
        logger.info("Computing Geo Resilience for state: %s", geo.state)

        profile = get_state_risk_profile(geo.state)

        # Use employment_type as a proxy for occupation when explicit
        # occupation is not in the schema (FUTURE_INTEGRATION)
        occupation = borrower.employment_type.value

        composite_risk = get_composite_climate_risk(profile, occupation=occupation)
        resilience_score = get_geo_resilience_score(composite_risk)
        risk_adj = get_risk_adjustment_multiplier(composite_risk, profile.recovery_capacity)

        climate_impact = _classify_climate_impact(composite_risk)
        recovery_capacity = RecoveryCapacity(profile.recovery_capacity)

        output = GeoResilienceOutput(
            state=geo.state,
            geo_resilience_score=resilience_score,
            flood_risk=profile.flood_risk,
            drought_risk=profile.drought_risk,
            cyclone_risk=profile.cyclone_risk,
            heatwave_risk=profile.heatwave_risk,
            composite_climate_risk=composite_risk,
            climate_impact=climate_impact,
            recovery_capacity=recovery_capacity,
            risk_adjustment=risk_adj,
            data_source="STATIC_OFFLINE",
        )

        logger.info(
            "Geo Resilience — state=%s, composite_risk=%.3f, resilience=%.1f, adj=%.3f",
            geo.state, composite_risk, resilience_score, risk_adj,
        )
        return output


def _classify_climate_impact(composite_risk: float) -> ClimateImpact:
    """Map composite risk score to ClimateImpact enum."""
    for impact, (low, high) in _CLIMATE_IMPACT_BANDS.items():
        if low <= composite_risk < high:
            return impact
    return ClimateImpact.SEVERE
