"""
Risk factor extraction and narrative generation for the IDBI AI module.

Transforms raw SHAP values into human-readable risk factors and
plain-English narratives for loan officers.

The goal: an officer should understand WHY the model made a prediction
without needing to understand SHAP or machine learning.
"""

from __future__ import annotations

from typing import List, Tuple

from ai.schemas.output_schema import RiskFactor, SHAPOutput


# ─── Human-readable feature name mapping ─────────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "income": "Annual Income",
    "loan_amount": "Loan Amount",
    "interest_rate": "Interest Rate",
    "age": "Borrower Age",
    "employment_length": "Employment Duration",
    "credit_history_length": "Credit History Length",
    "loan_percent_income": "Loan-to-Income %",
    "dti_ratio": "Debt-to-Income Ratio",
    "repayment_capacity": "Monthly Repayment Capacity",
    "cash_flow_health": "Cash Flow Health",
    "credit_score_proxy": "Credit Grade",
    "credit_score_proxy_scaled": "Credit Score (Proxy)",
    "estimated_emi": "Estimated Monthly EMI",
    "loan_to_income_ratio": "Loan-to-Income Ratio",
    "employment_stability": "Employment Stability",
    "income_per_year_employed": "Income Per Year Employed",
    "near_retirement_flag": "Near Retirement Age",
    "high_risk_purpose_flag": "High-Risk Loan Purpose",
    "default_on_file": "Past Default on File",
    "home_ownership": "Home Ownership",
    "loan_purpose": "Loan Purpose",
    "loan_grade": "Loan Grade",
}

# SHAP magnitude thresholds
MAGNITUDE_HIGH_THRESHOLD = 0.10
MAGNITUDE_MEDIUM_THRESHOLD = 0.04


def extract_top_risk_factors(
    top_features: List[Tuple[str, float]],
    base_value: float,
    predicted_probability: float,
) -> List[RiskFactor]:
    """
    Convert raw SHAP (feature, value) pairs into structured RiskFactor objects.

    Args:
        top_features: List of (feature_name, shap_value) sorted by |shap_value|.
        base_value: SHAP base value (log-odds scale).
        predicted_probability: Model's predicted default probability.

    Returns:
        List of :class:`RiskFactor` objects.
    """
    risk_factors = []

    for feature_name, shap_val in top_features:
        direction = "INCREASES_RISK" if shap_val > 0 else "DECREASES_RISK"
        abs_val = abs(shap_val)

        if abs_val >= MAGNITUDE_HIGH_THRESHOLD:
            magnitude = "HIGH"
        elif abs_val >= MAGNITUDE_MEDIUM_THRESHOLD:
            magnitude = "MEDIUM"
        else:
            magnitude = "LOW"

        human_name = _get_human_name(feature_name)
        description = _generate_factor_description(feature_name, shap_val, direction)

        risk_factors.append(RiskFactor(
            feature=feature_name,
            shap_value=round(shap_val, 5),
            direction=direction,
            magnitude=magnitude,
            human_readable_name=human_name,
            description=description,
        ))

    return risk_factors


def generate_shap_output(
    top_features: List[Tuple[str, float]],
    base_value: float,
    predicted_probability: float,
    risk_level: str,
) -> SHAPOutput:
    """
    Build a complete SHAPOutput object for the inference response.

    Args:
        top_features: SHAP feature-value pairs.
        base_value: SHAP explainer base value.
        predicted_probability: Predicted default probability.
        risk_level: Risk level string (LOW/MEDIUM/HIGH/VERY_HIGH).

    Returns:
        :class:`SHAPOutput` ready for inclusion in RiskIntelligenceOutput.
    """
    risk_factors = extract_top_risk_factors(top_features, base_value, predicted_probability)
    narrative = generate_narrative(risk_factors, predicted_probability, risk_level)

    return SHAPOutput(
        top_risk_factors=risk_factors,
        base_value=round(base_value, 4),
        predicted_probability=round(predicted_probability, 4),
        narrative=narrative,
    )


def generate_narrative(
    risk_factors: List[RiskFactor],
    predicted_probability: float,
    risk_level: str,
) -> str:
    """
    Generate a plain-English explanation narrative for loan officers.

    Args:
        risk_factors: Extracted risk factors.
        predicted_probability: Default probability (0–1).
        risk_level: Risk classification string.

    Returns:
        Narrative string (2–4 sentences).
    """
    pct = round(predicted_probability * 100, 1)

    # Opening sentence
    risk_descriptions = {
        "LOW": "low likelihood of default",
        "MEDIUM": "moderate default risk",
        "HIGH": "elevated default risk",
        "VERY_HIGH": "very high default risk",
    }
    risk_desc = risk_descriptions.get(risk_level, "uncertain risk level")
    narrative = f"The AI model assigns a {pct}% probability of default, indicating {risk_desc}. "

    # Top risk drivers
    increasing = [f for f in risk_factors if f.direction == "INCREASES_RISK"]
    decreasing = [f for f in risk_factors if f.direction == "DECREASES_RISK"]

    if increasing:
        top_drivers = [f.human_readable_name for f in increasing[:3]]
        narrative += (
            f"The primary factors increasing default risk are: {', '.join(top_drivers)}. "
        )

    if decreasing:
        top_protectors = [f.human_readable_name for f in decreasing[:2]]
        narrative += (
            f"Factors reducing risk include: {', '.join(top_protectors)}. "
        )

    narrative += "Please review the detailed factor breakdown below before making a final decision."
    return narrative.strip()


def _get_human_name(feature_name: str) -> str:
    """Map internal feature name to display name."""
    # Handle OHE column names like "nominal__home_ownership_RENT"
    if "__" in feature_name:
        parts = feature_name.split("__", 1)[-1]
        base = parts.split("_")[0] if "_" in parts else parts
        value = parts.replace(base + "_", "")
        base_human = FEATURE_DISPLAY_NAMES.get(base, base.replace("_", " ").title())
        return f"{base_human}: {value.title()}"
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").title())


def _generate_factor_description(
    feature_name: str,
    shap_value: float,
    direction: str,
) -> str:
    """Generate a short one-line description for a risk factor."""
    human_name = _get_human_name(feature_name)
    impact = "significantly" if abs(shap_value) >= MAGNITUDE_HIGH_THRESHOLD else "moderately"
    action = "increases" if direction == "INCREASES_RISK" else "reduces"
    return f"{human_name} {impact} {action} the predicted default probability."
