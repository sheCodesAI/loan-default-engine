"""Unit tests for data/feature_engineering.py."""

import pandas as pd
import numpy as np
import pytest
from ai.data.feature_engineering import (
    engineer_features,
    _add_repayment_capacity,
    _add_cash_flow_health,
    _add_loan_to_income_ratio,
    _add_employment_stability,
    _add_credit_score_normalized,
    _add_near_retirement_flag,
    _add_high_dti_flag,
    _add_delinquency_flag,
)


def make_sample_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal sample DataFrame for testing feature engineering."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.randint(22, 70, n),
        "income": rng.uniform(200_000, 2_000_000, n),
        "monthly_income": rng.uniform(20_000, 150_000, n),
        "employment_length": rng.randint(0, 120, n),
        "credit_score": rng.randint(300, 900, n),
        "credit_history_length": rng.randint(6, 120, n),
        "delinquency_count": rng.randint(0, 5, n),
        "loan_amount": rng.uniform(50_000, 1_500_000, n),
        "loan_tenure": rng.randint(12, 60, n),
        "interest_rate": rng.uniform(8.0, 24.0, n),
        "emi_amount": rng.uniform(5_000, 40_000, n),
        "existing_emi": rng.uniform(0, 20_000, n),
        "dti_ratio": rng.uniform(0.1, 0.9, n),
    })


class TestEngineerFeatures:
    def test_returns_dataframe(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_rows_dropped(self):
        df = make_sample_df(100)
        result = engineer_features(df)
        assert len(result) == 100

    def test_credit_score_normalized_added(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert "credit_score_normalized" in result.columns
        assert result["credit_score_normalized"].between(0, 1).all()

    def test_employment_stability_in_range(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert "employment_stability" in result.columns
        assert result["employment_stability"].between(0, 1).all()

    def test_near_retirement_flag_binary(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert set(result["near_retirement_flag"].unique()).issubset({0, 1})

    def test_loan_to_income_positive(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert "loan_to_income_ratio" in result.columns
        assert (result["loan_to_income_ratio"] >= 0).all()

    def test_high_dti_flag(self):
        df = make_sample_df()
        df["dti_ratio"] = 0.75
        result = engineer_features(df)
        assert (result["high_dti_flag"] == 1).all()


class TestCreditScoreNormalization:
    def test_max_score_normalized(self):
        df = pd.DataFrame({"credit_score": [900]})
        result = _add_credit_score_normalized(df)
        assert result["credit_score_normalized"].iloc[0] == 1.0

    def test_min_score_normalized(self):
        df = pd.DataFrame({"credit_score": [300]})
        result = _add_credit_score_normalized(df)
        assert result["credit_score_normalized"].iloc[0] == 0.0

    def test_out_of_bounds_capped(self):
        df = pd.DataFrame({"credit_score": [950, 250]})
        result = _add_credit_score_normalized(df)
        assert result["credit_score_normalized"].iloc[0] == 1.0
        assert result["credit_score_normalized"].iloc[1] == 0.0


class TestAgeRiskFlag:
    def test_above_58_flagged(self):
        df = pd.DataFrame({"age": [59, 65, 70]})
        result = _add_near_retirement_flag(df)
        assert result["near_retirement_flag"].sum() == 3

    def test_below_58_not_flagged(self):
        df = pd.DataFrame({"age": [25, 40, 58]})
        result = _add_near_retirement_flag(df)
        assert result["near_retirement_flag"].sum() == 0
