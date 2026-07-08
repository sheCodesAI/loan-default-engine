"""
Inference Pipeline for the IDBI AI Credit Risk Intelligence Platform.

Orchestrates the complete inference flow in order:
  Borrower 360° Intelligence →
  Geo & Resilience Intelligence →
  ML Prediction →
  SHAP Explanation →
  Business Rules Engine →
  Recommendation Engine →
  AI Risk Intelligence →
  Final Output

Usage:
    from ai.pipeline.inference_pipeline import InferencePipeline
    from ai.schemas.input_schema import InferenceRequest

    pipeline = InferencePipeline()
    result = pipeline.run(request)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ai.config import DEFAULT_PROBABILITY_THRESHOLD, SHAP_TOP_N_FEATURES
from ai.engines.borrower_360 import Borrower360Engine
from ai.engines.business_rules import BusinessRulesEngine
from ai.engines.geo_resilience import GeoResilienceEngine
from ai.engines.recommendation import RecommendationEngine
from ai.engines.risk_intelligence import RiskIntelligenceEngine
from ai.explainability.risk_factors import generate_shap_output
from ai.explainability.shap_engine import SHAPEngine
from ai.logger import get_logger
from ai.models.persistence import load_model, load_model_metadata, load_preprocessor
from ai.schemas.input_schema import InferenceRequest
from ai.schemas.output_schema import RiskIntelligenceOutput

logger = get_logger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline.

    Loads model and preprocessor once at initialization (lazy loading).
    Subsequent calls to run() are stateless and thread-safe
    (each call creates its own local state).

    Initialize once; reuse for multiple inference calls.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        """
        Initialize the inference pipeline.

        Loads model, preprocessor, and metadata from artifacts/.
        Initializes all five engines.

        Args:
            threshold: Decision threshold. If None, loads from model metadata.
        """
        logger.info("Initializing InferencePipeline...")

        # ── Load artifacts ───────────────────────────────────────────────
        self.model = load_model()
        self.preprocessor = load_preprocessor()

        try:
            metadata = load_model_metadata()
            self.feature_names: list = metadata.get("feature_names", [])
            self.threshold: float = threshold or metadata.get("decision_threshold", DEFAULT_PROBABILITY_THRESHOLD)
        except FileNotFoundError:
            logger.warning("Model metadata not found. Using default threshold.")
            self.feature_names = []
            self.threshold = threshold or DEFAULT_PROBABILITY_THRESHOLD

        # ── SHAP Engine ──────────────────────────────────────────────────
        self.shap_engine = SHAPEngine(
            model=self.model,
            feature_names=self.feature_names,
        )

        # ── Intelligence Engines (stateless — instantiate once) ──────────
        self.borrower_360_engine = Borrower360Engine()
        self.geo_engine = GeoResilienceEngine()
        self.rules_engine = BusinessRulesEngine()
        self.recommendation_engine = RecommendationEngine()
        self.risk_engine = RiskIntelligenceEngine()

        logger.info(
            "InferencePipeline ready — threshold=%.4f, features=%d",
            self.threshold, len(self.feature_names),
        )

    def run(self, request: InferenceRequest) -> RiskIntelligenceOutput:
        """
        Run the complete inference pipeline for a single loan application.

        Args:
            request: Validated :class:`InferenceRequest`.

        Returns:
            :class:`RiskIntelligenceOutput` — final structured AI decision.
        """
        logger.info(
            "Inference started — request_id=%s, borrower_age=%d, loan=%.0f",
            request.request_id, request.borrower.age, request.loan.loan_amount,
        )

        # ── Step 1: Borrower 360° Intelligence ───────────────────────────
        borrower_360 = self.borrower_360_engine.compute(
            borrower=request.borrower,
            loan=request.loan,
        )

        # ── Step 2: Geo & Resilience Intelligence ─────────────────────────
        geo = self.geo_engine.compute(
            geo=request.geo,
            borrower=request.borrower,
            loan_purpose=request.loan.loan_purpose.value,
        )

        # ── Step 3: ML Prediction ─────────────────────────────────────────
        X_input = self._build_feature_vector(request, borrower_360)
        X_processed = self.preprocessor.transform(X_input)
        raw_probability = float(self.model.predict_proba(X_processed)[0, 1])
        logger.info("ML raw default probability: %.4f", raw_probability)

        # ── Step 4: SHAP Explanation ──────────────────────────────────────
        shap_vals, base_val = self.shap_engine.get_instance_shap(X_processed)
        top_features = self.shap_engine.get_top_features(
            shap_vals, top_n=SHAP_TOP_N_FEATURES
        )

        # Determine risk level for narrative (pre-geo-adjustment)
        from ai.engines.risk_intelligence import classify_risk_level
        preliminary_risk = classify_risk_level(raw_probability)

        shap_output = generate_shap_output(
            top_features=top_features,
            base_value=base_val,
            predicted_probability=raw_probability,
            risk_level=preliminary_risk.value,
        )

        # ── Step 5: Business Rules Engine ────────────────────────────────
        rules = self.rules_engine.evaluate(
            borrower=request.borrower,
            loan=request.loan,
            credit_score_proxy=borrower_360.borrower_score,
        )

        # ── Step 6: Recommendation Engine ────────────────────────────────
        # Use geo-adjusted probability for recommendation
        geo_adjusted_prob = min(raw_probability * geo.risk_adjustment, 1.0)
        adjusted_risk_level = classify_risk_level(geo_adjusted_prob)

        recommendation = self.recommendation_engine.generate(
            risk_level=adjusted_risk_level,
            default_probability=geo_adjusted_prob,
            borrower_360=borrower_360,
            geo=geo,
            rules=rules,
            borrower=request.borrower,
            loan=request.loan,
        )

        # ── Step 7: AI Risk Intelligence (Final Composer) ─────────────────
        result = self.risk_engine.compose(
            default_probability=raw_probability,
            borrower_360=borrower_360,
            geo=geo,
            rules=rules,
            shap_output=shap_output,
            recommendation=recommendation,
            loan=request.loan,
            request_id=request.request_id,
        )

        logger.info(
            "Inference complete — risk=%s, action=%s, adj_prob=%.4f",
            result.risk_level.value,
            result.recommendation.action.value,
            result.default_probability,
        )

        return result

    def _build_feature_vector(
        self,
        request: InferenceRequest,
        borrower_360: "Borrower360Output",
    ) -> pd.DataFrame:
        """
        Build a single-row DataFrame for the preprocessor from the request.

        Maps InferenceRequest fields → canonical training feature names.
        FUTURE_INTEGRATION fields are filled with sensible defaults.
        """
        from ai.utils.financial_utils import calculate_loan_to_income_ratio
        from ai.utils.geo_data import get_state_risk_profile

        b = request.borrower
        l = request.loan
        g = request.geo

        monthly_income = b.monthly_income or (b.annual_income / 12)
        emp_stability = min(b.employment_length_months / 60, 1.0)
        cs = b.credit_score if b.credit_score is not None else 650
        cs_normalized = max(min(cs, 900) - 300, 0) / 600

        geo_profile = get_state_risk_profile(g.state)
        state_risk = round(
            geo_profile.flood_risk * 0.3 + geo_profile.drought_risk * 0.4
            + geo_profile.cyclone_risk * 0.2 + geo_profile.heatwave_risk * 0.1, 4
        )
        district_risk = state_risk   # FUTURE_INTEGRATION: district-level lookup

        loan_to_income = calculate_loan_to_income_ratio(l.loan_amount, b.annual_income)

        row = {
            # Numeric
            "age": b.age,
            "income": b.annual_income,
            "monthly_income": monthly_income,
            "employment_length": b.employment_length_months,
            "credit_score": cs,
            "credit_history_length": b.credit_history_length_months or 0,
            "delinquency_count": 0,
            "loan_amount": l.loan_amount,
            "loan_tenure": l.loan_tenure_months,
            "interest_rate": l.interest_rate,
            "emi_amount": borrower_360.estimated_emi,
            "existing_emi": l.existing_monthly_obligations,
            "dti_ratio": borrower_360.dti_ratio,
            "area_default_rate": 9.0,       # FUTURE_INTEGRATION: area mean
            "district_risk_score": district_risk,
            "state_risk_score": state_risk,
            "distance_to_branch_km": 10.0,  # FUTURE_INTEGRATION
            "service_area_cluster": 0,       # FUTURE_INTEGRATION
            "repayment_capacity": borrower_360.repayment_capacity,
            "cash_flow_health": borrower_360.cash_flow_health,
            "loan_to_income_ratio": loan_to_income,
            "employment_stability": emp_stability,
            "credit_score_normalized": cs_normalized,
            # Categorical
            "employment_type": b.employment_type.value,
            "gender": "Male",           # FUTURE_INTEGRATION
            "education": "Graduate",    # FUTURE_INTEGRATION
            "marital_status": b.marital_status or "Married",
            "loan_purpose": l.loan_purpose.value,
            "loan_product": "Unsecured",   # FUTURE_INTEGRATION
            "loan_source_type": "New",     # FUTURE_INTEGRATION
            "urban_rural_flag": "Urban",   # FUTURE_INTEGRATION
            "branch_region": "WestZone",   # FUTURE_INTEGRATION: default to most common
            "population_density_band": "Medium",   # FUTURE_INTEGRATION
            "economic_activity_type": "Salaried",  # FUTURE_INTEGRATION
            "channel": "Branch",       # FUTURE_INTEGRATION
            "state": g.state,          # Geographic state — OHE-encoded in preprocessor
            # Binary
            "has_mortgage": int(b.home_ownership.value.upper() == "MORTGAGE"),
            "has_dependents": int(b.dependents > 0),
            "has_cosigner": int(l.co_applicant),
            "past_default_flag": 0,     # FUTURE_INTEGRATION
            "near_retirement_flag": int(b.age > 58),
            "high_dti_flag": int(borrower_360.dti_ratio > 0.60),
            "delinquency_flag": 0,      # FUTURE_INTEGRATION
        }

        return pd.DataFrame([row])
