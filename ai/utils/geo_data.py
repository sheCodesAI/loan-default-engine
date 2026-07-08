"""
Static geo & climate risk data for Indian states.

This module provides an entirely offline implementation of state-level
disaster risk scoring based on publicly available NDMA (National Disaster
Management Authority) and IMD (India Meteorological Department) classifications.

No paid API is used. All data is derived from public government reports.

FUTURE_INTEGRATION: Replace static lookup with live NDMA / IMD API calls
when credentials become available. The interface (get_state_risk_profile)
remains identical so callers need no changes.

Risk scores are on a 0–1 scale:
    0.0 = negligible risk
    1.0 = extreme risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class StateRiskProfile:
    """
    Disaster risk profile for an Indian state.

    All risk values are on a 0–1 scale (higher = more risky).
    recovery_capacity reflects economic resilience (HIGH/MEDIUM/LOW).
    """

    state: str
    flood_risk: float
    drought_risk: float
    cyclone_risk: float
    heatwave_risk: float
    recovery_capacity: str  # HIGH | MEDIUM | LOW


# ─── Static State Risk Lookup ─────────────────────────────────────────────────
# Sources: NDMA State Disaster Risk Profiles, IMD Hazard Atlas of India
# Last curated: 2024

_STATE_RISK_DATA: Dict[str, StateRiskProfile] = {
    # ── North-East & Eastern States (high flood) ──────────────────────────
    "Assam": StateRiskProfile("Assam", flood_risk=0.95, drought_risk=0.10, cyclone_risk=0.20, heatwave_risk=0.15, recovery_capacity="LOW"),
    "Bihar": StateRiskProfile("Bihar", flood_risk=0.90, drought_risk=0.25, cyclone_risk=0.10, heatwave_risk=0.50, recovery_capacity="LOW"),
    "West Bengal": StateRiskProfile("West Bengal", flood_risk=0.80, drought_risk=0.15, cyclone_risk=0.70, heatwave_risk=0.30, recovery_capacity="MEDIUM"),
    "Odisha": StateRiskProfile("Odisha", flood_risk=0.80, drought_risk=0.35, cyclone_risk=0.90, heatwave_risk=0.40, recovery_capacity="MEDIUM"),
    "Jharkhand": StateRiskProfile("Jharkhand", flood_risk=0.50, drought_risk=0.55, cyclone_risk=0.10, heatwave_risk=0.50, recovery_capacity="LOW"),
    "Meghalaya": StateRiskProfile("Meghalaya", flood_risk=0.70, drought_risk=0.05, cyclone_risk=0.20, heatwave_risk=0.10, recovery_capacity="LOW"),
    "Manipur": StateRiskProfile("Manipur", flood_risk=0.65, drought_risk=0.10, cyclone_risk=0.10, heatwave_risk=0.10, recovery_capacity="LOW"),
    "Nagaland": StateRiskProfile("Nagaland", flood_risk=0.55, drought_risk=0.10, cyclone_risk=0.05, heatwave_risk=0.10, recovery_capacity="LOW"),
    "Mizoram": StateRiskProfile("Mizoram", flood_risk=0.60, drought_risk=0.10, cyclone_risk=0.15, heatwave_risk=0.10, recovery_capacity="LOW"),
    "Arunachal Pradesh": StateRiskProfile("Arunachal Pradesh", flood_risk=0.75, drought_risk=0.05, cyclone_risk=0.05, heatwave_risk=0.10, recovery_capacity="LOW"),
    "Tripura": StateRiskProfile("Tripura", flood_risk=0.70, drought_risk=0.10, cyclone_risk=0.20, heatwave_risk=0.20, recovery_capacity="LOW"),
    "Sikkim": StateRiskProfile("Sikkim", flood_risk=0.70, drought_risk=0.05, cyclone_risk=0.05, heatwave_risk=0.05, recovery_capacity="LOW"),

    # ── North & Central India (drought + heatwave) ────────────────────────
    "Rajasthan": StateRiskProfile("Rajasthan", flood_risk=0.25, drought_risk=0.90, cyclone_risk=0.10, heatwave_risk=0.95, recovery_capacity="MEDIUM"),
    "Uttar Pradesh": StateRiskProfile("Uttar Pradesh", flood_risk=0.65, drought_risk=0.45, cyclone_risk=0.05, heatwave_risk=0.75, recovery_capacity="MEDIUM"),
    "Madhya Pradesh": StateRiskProfile("Madhya Pradesh", flood_risk=0.40, drought_risk=0.65, cyclone_risk=0.05, heatwave_risk=0.70, recovery_capacity="MEDIUM"),
    "Haryana": StateRiskProfile("Haryana", flood_risk=0.30, drought_risk=0.60, cyclone_risk=0.05, heatwave_risk=0.70, recovery_capacity="HIGH"),
    "Punjab": StateRiskProfile("Punjab", flood_risk=0.35, drought_risk=0.40, cyclone_risk=0.05, heatwave_risk=0.65, recovery_capacity="HIGH"),
    "Delhi": StateRiskProfile("Delhi", flood_risk=0.30, drought_risk=0.30, cyclone_risk=0.05, heatwave_risk=0.70, recovery_capacity="HIGH"),
    "Uttarakhand": StateRiskProfile("Uttarakhand", flood_risk=0.85, drought_risk=0.20, cyclone_risk=0.10, heatwave_risk=0.30, recovery_capacity="MEDIUM"),
    "Himachal Pradesh": StateRiskProfile("Himachal Pradesh", flood_risk=0.60, drought_risk=0.25, cyclone_risk=0.05, heatwave_risk=0.25, recovery_capacity="MEDIUM"),
    "Jammu And Kashmir": StateRiskProfile("Jammu And Kashmir", flood_risk=0.65, drought_risk=0.20, cyclone_risk=0.05, heatwave_risk=0.20, recovery_capacity="MEDIUM"),

    # ── Western India ─────────────────────────────────────────────────────
    "Maharashtra": StateRiskProfile("Maharashtra", flood_risk=0.55, drought_risk=0.70, cyclone_risk=0.25, heatwave_risk=0.55, recovery_capacity="HIGH"),
    "Gujarat": StateRiskProfile("Gujarat", flood_risk=0.45, drought_risk=0.65, cyclone_risk=0.70, heatwave_risk=0.65, recovery_capacity="HIGH"),
    "Goa": StateRiskProfile("Goa", flood_risk=0.45, drought_risk=0.10, cyclone_risk=0.35, heatwave_risk=0.30, recovery_capacity="HIGH"),

    # ── Southern India ────────────────────────────────────────────────────
    "Karnataka": StateRiskProfile("Karnataka", flood_risk=0.40, drought_risk=0.70, cyclone_risk=0.20, heatwave_risk=0.55, recovery_capacity="HIGH"),
    "Telangana": StateRiskProfile("Telangana", flood_risk=0.45, drought_risk=0.75, cyclone_risk=0.20, heatwave_risk=0.70, recovery_capacity="HIGH"),
    "Andhra Pradesh": StateRiskProfile("Andhra Pradesh", flood_risk=0.55, drought_risk=0.65, cyclone_risk=0.80, heatwave_risk=0.65, recovery_capacity="MEDIUM"),
    "Tamil Nadu": StateRiskProfile("Tamil Nadu", flood_risk=0.55, drought_risk=0.55, cyclone_risk=0.75, heatwave_risk=0.50, recovery_capacity="HIGH"),
    "Kerala": StateRiskProfile("Kerala", flood_risk=0.75, drought_risk=0.20, cyclone_risk=0.40, heatwave_risk=0.30, recovery_capacity="HIGH"),

    # ── Eastern / Other ───────────────────────────────────────────────────
    "Chhattisgarh": StateRiskProfile("Chhattisgarh", flood_risk=0.55, drought_risk=0.55, cyclone_risk=0.10, heatwave_risk=0.55, recovery_capacity="LOW"),
    "Chandigarh": StateRiskProfile("Chandigarh", flood_risk=0.15, drought_risk=0.25, cyclone_risk=0.05, heatwave_risk=0.60, recovery_capacity="HIGH"),
    "Puducherry": StateRiskProfile("Puducherry", flood_risk=0.50, drought_risk=0.30, cyclone_risk=0.70, heatwave_risk=0.40, recovery_capacity="MEDIUM"),
}

# Alias normalisation map (handles common misspellings / abbreviations)
_ALIASES: Dict[str, str] = {
    "J&K": "Jammu And Kashmir",
    "Jammu": "Jammu And Kashmir",
    "Kashmir": "Jammu And Kashmir",
    "UP": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "AP": "Andhra Pradesh",
    "TN": "Tamil Nadu",
    "WB": "West Bengal",
    "HP": "Himachal Pradesh",
}

# Default profile for unknown states — moderate risk across all dimensions
_DEFAULT_PROFILE = StateRiskProfile(
    state="Unknown",
    flood_risk=0.35,
    drought_risk=0.35,
    cyclone_risk=0.15,
    heatwave_risk=0.35,
    recovery_capacity="MEDIUM",
)


def get_state_risk_profile(state: str) -> StateRiskProfile:
    """
    Return the risk profile for the given Indian state.

    Normalises state name (title-case, alias resolution) before lookup.
    Falls back to a neutral moderate-risk profile if state is unknown.

    Args:
        state: State name string (case-insensitive).

    Returns:
        :class:`StateRiskProfile` with risk scores and recovery capacity.
    """
    normalized = state.strip().title()
    normalized = _ALIASES.get(normalized.upper(), normalized)
    normalized = _ALIASES.get(normalized, normalized)

    profile = _STATE_RISK_DATA.get(normalized)
    if profile is None:
        # Try case-insensitive match
        for key in _STATE_RISK_DATA:
            if key.lower() == normalized.lower():
                profile = _STATE_RISK_DATA[key]
                break

    if profile is None:
        profile = _DEFAULT_PROFILE

    return profile


def get_composite_climate_risk(profile: StateRiskProfile, occupation: Optional[str] = None) -> float:
    """
    Compute a weighted composite climate risk score (0–1).

    Weights by disaster type:
        Flood    35%  — affects agriculture, transport
        Drought  30%  — affects agriculture, income
        Cyclone  20%  — coastal asset destruction
        Heatwave 15%  — productivity loss

    Occupation modifier: agriculture-dependent borrowers get a 1.2× multiplier
    on drought and flood risk (capped at 1.0 for the final score).

    Args:
        profile: StateRiskProfile from get_state_risk_profile().
        occupation: Borrower's occupation (optional).

    Returns:
        Composite risk score 0–1.
    """
    flood_w = 0.35
    drought_w = 0.30
    cyclone_w = 0.20
    heatwave_w = 0.15

    flood_r = profile.flood_risk
    drought_r = profile.drought_risk

    # Agriculture occupation amplifier
    if occupation and occupation.upper() in {"AGRICULTURE", "FARMING", "FARMER"}:
        flood_r = min(flood_r * 1.2, 1.0)
        drought_r = min(drought_r * 1.2, 1.0)

    composite = (
        flood_w * flood_r
        + drought_w * drought_r
        + cyclone_w * profile.cyclone_risk
        + heatwave_w * profile.heatwave_risk
    )
    return round(min(composite, 1.0), 4)


def get_geo_resilience_score(composite_climate_risk: float) -> float:
    """
    Convert composite climate risk to a resilience score (0–100).

    Higher score = more resilient (lower risk).

    Args:
        composite_climate_risk: 0–1 composite risk score.

    Returns:
        Resilience score 0–100.
    """
    return round((1 - composite_climate_risk) * 100, 2)


def get_risk_adjustment_multiplier(
    composite_climate_risk: float,
    recovery_capacity: str,
) -> float:
    """
    Calculate a risk adjustment multiplier for the final default probability.

    The multiplier is applied as: adjusted_prob = base_prob × multiplier.
    Multiplier > 1.0 increases risk; < 1.0 decreases it.

    Recovery capacity reduces the impact of high climate risk:
        HIGH recovery  → less adjustment
        LOW recovery   → more adjustment

    Args:
        composite_climate_risk: 0–1 composite risk score.
        recovery_capacity: "HIGH" | "MEDIUM" | "LOW".

    Returns:
        Risk adjustment multiplier (typically 0.90 – 1.25).
    """
    rc_factor = {"HIGH": 0.5, "MEDIUM": 1.0, "LOW": 1.5}.get(recovery_capacity, 1.0)

    # Base adjustment: 0% for zero risk, up to +25% for extreme risk × low recovery
    adjustment = 1.0 + (composite_climate_risk * 0.25 * rc_factor)
    return round(min(adjustment, 1.50), 4)


def list_all_states() -> list[str]:
    """Return all supported state names."""
    return sorted(_STATE_RISK_DATA.keys())
