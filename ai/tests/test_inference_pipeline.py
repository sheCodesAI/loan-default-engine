"""
Integration test for the inference pipeline.

Tests that all engines chain correctly without a trained model by
mocking the ML components. The unit-level engine tests cover
individual engine correctness; this test covers pipeline wiring.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from ai.schemas.input_schema import (
    BorrowerInput, EmploymentType, GeoInput, HomeOwnership,
    InferenceRequest, LoanInput, LoanGrade, LoanPurpose
)


def make_request() -> InferenceRequest:
    return InferenceRequest(
        request_id="test-001",
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
            co_applicant=False,
            insurance=False,
        ),
        geo=GeoInput(state="Maharashtra"),
    )


class TestInferencePipelineIntegration:
    """
    Integration tests using mocked ML model and preprocessor.

    These tests verify that the pipeline wiring is correct and that
    all engines are called and their outputs flow correctly.
    A trained model is NOT required to run these tests.
    """

    @pytest.fixture
    def mock_pipeline(self):
        """Create an InferencePipeline with mocked ML components."""
        with patch("ai.pipeline.inference_pipeline.load_model") as mock_model_fn, \
             patch("ai.pipeline.inference_pipeline.load_preprocessor") as mock_prep_fn, \
             patch("ai.pipeline.inference_pipeline.load_model_metadata") as mock_meta_fn:

            # Mock model that returns 30% default probability
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.70, 0.30]])
            mock_model_fn.return_value = mock_model

            # Mock preprocessor that passes through a dummy array
            mock_prep = MagicMock()
            mock_prep.transform.return_value = np.zeros((1, 20))
            mock_prep_fn.return_value = mock_prep

            # Mock metadata
            mock_meta_fn.return_value = {
                "feature_names": [f"feat_{i}" for i in range(20)],
                "decision_threshold": 0.50,
            }

            from ai.pipeline.inference_pipeline import InferencePipeline
            pipeline = InferencePipeline()
            pipeline.shap_engine = MagicMock()
            pipeline.shap_engine.get_instance_shap.return_value = (
                np.random.randn(20) * 0.05,   # shap values
                -0.50,                          # base value
            )
            pipeline.shap_engine.get_top_features.return_value = [
                (f"feat_{i}", float(np.random.randn() * 0.05)) for i in range(10)
            ]

            return pipeline

    def test_pipeline_runs_without_error(self, mock_pipeline):
        request = make_request()
        result = mock_pipeline.run(request)
        assert result is not None

    def test_request_id_preserved(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.request_id == "test-001"

    def test_probability_in_range(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert 0 <= result.default_probability <= 1

    def test_risk_level_valid(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.risk_level.value in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}

    def test_expected_loss_non_negative(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.expected_loss >= 0

    def test_recommendation_action_valid(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        valid_actions = {
            "APPROVE", "MANUAL_REVIEW", "REJECT",
            "REDUCE_LOAN_AMOUNT", "INCREASE_TENURE",
            "REQUEST_CO_APPLICANT", "REQUEST_INSURANCE",
        }
        assert result.recommendation.action.value in valid_actions

    def test_borrower_360_attached(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.borrower_360 is not None
        assert 0 <= result.borrower_360.borrower_score <= 100

    def test_geo_resilience_attached(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.geo_resilience is not None
        assert result.geo_resilience.state == "Maharashtra"

    def test_shap_output_attached(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.shap_explanation is not None
        assert len(result.shap_explanation.top_risk_factors) > 0

    def test_summary_non_empty(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert isinstance(result.summary, str)
        assert len(result.summary) > 50

    def test_timestamp_present(self, mock_pipeline):
        result = mock_pipeline.run(make_request())
        assert result.timestamp is not None
        assert "T" in result.timestamp   # ISO 8601 format
