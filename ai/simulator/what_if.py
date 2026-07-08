"""
What-If Simulator — Signature Feature of the IDBI AI Platform.

Allows loan officers to interactively modify key loan parameters
and immediately observe how the risk profile changes.

Supported overrides (from WhatIfOverrides schema):
  - loan_amount
  - loan_tenure_months
  - co_applicant (bool)
  - insurance (bool)
  - interest_rate

The simulator runs the complete inference pipeline twice:
  1. Base scenario (original request)
  2. Modified scenario (with overrides applied)

Then computes deltas and generates an insight for the officer.
"""

from __future__ import annotations

import copy
from typing import Any

from ai.logger import get_logger
from ai.schemas.input_schema import InferenceRequest, WhatIfOverrides
from ai.schemas.output_schema import (
    RecommendationAction,
    RiskLevel,
    WhatIfOutput,
    WhatIfScenario,
)

logger = get_logger(__name__)


class WhatIfSimulator:
    """
    What-If Simulator.

    Requires an initialized InferencePipeline instance.
    Runs the full inference pipeline for base and modified scenarios.
    """

    def __init__(self, inference_pipeline: Any = None) -> None:
        """
        Initialize with an inference pipeline.

        Args:
            inference_pipeline: An initialized InferencePipeline instance.
                If None, automatically loads the trained model from disk.
        """
        if inference_pipeline is None:
            from ai.pipeline.inference_pipeline import InferencePipeline
            inference_pipeline = InferencePipeline()
        self.pipeline = inference_pipeline

    def run(
        self,
        request: InferenceRequest,
        overrides: WhatIfOverrides,
        scenario_name: str = "What-If Scenario",
    ) -> WhatIfOutput:
        """Alias for simulate() — convenience method."""
        return self.simulate(request, overrides, scenario_name)

    def simulate(
        self,
        request: InferenceRequest,
        overrides: WhatIfOverrides,
        scenario_name: str = "What-If Scenario",
    ) -> WhatIfOutput:
        """
        Run base and modified scenarios and return delta analysis.

        Args:
            request: Original inference request.
            overrides: Parameter overrides to apply.
            scenario_name: Label for the modified scenario.

        Returns:
            :class:`WhatIfOutput` with side-by-side scenario comparison.
        """
        logger.info("Running What-If simulation: %s", scenario_name)

        # ── Base scenario ────────────────────────────────────────────────
        base_result = self.pipeline.run(request)
        base_scenario = self._extract_scenario("Base Scenario", {}, base_result)

        # ── Modified scenario ────────────────────────────────────────────
        modified_request = self._apply_overrides(request, overrides)
        modified_result = self.pipeline.run(modified_request)
        overrides_dict = overrides.model_dump(exclude_none=True)
        modified_scenario = self._extract_scenario(scenario_name, overrides_dict, modified_result)

        # ── Delta analysis ───────────────────────────────────────────────
        delta_prob = modified_scenario.default_probability - base_scenario.default_probability
        delta_el = modified_scenario.expected_loss - base_scenario.expected_loss
        risk_changed = base_scenario.risk_level != modified_scenario.risk_level
        rec_changed = base_scenario.recommendation != modified_scenario.recommendation

        insight = self._generate_insight(
            base_scenario, modified_scenario,
            delta_prob, overrides_dict, risk_changed, rec_changed,
        )

        logger.info(
            "What-If complete — ΔP=%.4f, ΔEL=%.2f, risk_changed=%s, rec_changed=%s",
            delta_prob, delta_el, risk_changed, rec_changed,
        )

        return WhatIfOutput(
            base_scenario=base_scenario,
            modified_scenario=modified_scenario,
            delta_probability=round(delta_prob, 4),
            delta_expected_loss=round(delta_el, 2),
            risk_level_changed=risk_changed,
            recommendation_changed=rec_changed,
            insight=insight,
        )

    @staticmethod
    def _apply_overrides(
        request: InferenceRequest,
        overrides: WhatIfOverrides,
    ) -> InferenceRequest:
        """
        Create a modified copy of the request with overrides applied.

        Uses deep copy to avoid mutating the original request.
        """
        # Serialize to dict, apply overrides, reconstruct
        req_dict = request.model_dump()

        if overrides.loan_amount is not None:
            req_dict["loan"]["loan_amount"] = overrides.loan_amount
        if overrides.loan_tenure_months is not None:
            req_dict["loan"]["loan_tenure_months"] = overrides.loan_tenure_months
        if overrides.co_applicant is not None:
            req_dict["loan"]["co_applicant"] = overrides.co_applicant
        if overrides.insurance is not None:
            req_dict["loan"]["insurance"] = overrides.insurance
        if overrides.interest_rate is not None:
            req_dict["loan"]["interest_rate"] = overrides.interest_rate

        return InferenceRequest(**req_dict)

    @staticmethod
    def _extract_scenario(
        name: str,
        overrides: dict,
        result: Any,
    ) -> WhatIfScenario:
        """Extract a WhatIfScenario from a full RiskIntelligenceOutput."""
        return WhatIfScenario(
            scenario_name=name,
            overrides_applied=overrides,
            default_probability=result.default_probability,
            risk_level=result.risk_level,
            expected_loss=result.expected_loss,
            recommendation=result.recommendation.action,
            borrower_score=result.borrower_360.borrower_score,
        )

    @staticmethod
    def _generate_insight(
        base: WhatIfScenario,
        modified: WhatIfScenario,
        delta_prob: float,
        overrides: dict,
        risk_changed: bool,
        rec_changed: bool,
    ) -> str:
        """Generate a plain-English insight about the scenario delta."""
        direction = "decreased" if delta_prob < 0 else "increased"
        pct_change = abs(round(delta_prob * 100, 1))

        lines = [
            f"Applying the changes {list(overrides.keys())} "
            f"{direction} default probability by {pct_change}% "
            f"(from {base.default_probability:.1%} to {modified.default_probability:.1%})."
        ]

        if risk_changed:
            lines.append(
                f"Risk level changed from {base.risk_level.value} to {modified.risk_level.value}."
            )

        if rec_changed:
            lines.append(
                f"Recommendation changed from {base.recommendation.value} "
                f"to {modified.recommendation.value}."
            )

        if delta_prob < -0.05:
            lines.append("This scenario shows a meaningful improvement. Consider advising the applicant accordingly.")
        elif delta_prob > 0.05:
            lines.append("This scenario worsens the risk profile. The original parameters are preferable.")
        else:
            lines.append("The impact of this change on risk is minimal.")

        return " ".join(lines)
