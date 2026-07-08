"""Unit tests for financial_utils.py — pure financial math functions."""

import pytest
from ai.utils.financial_utils import (
    calculate_dti,
    calculate_emi,
    calculate_expected_loss,
    calculate_cash_flow_health,
    calculate_loan_to_income_ratio,
    calculate_repayment_capacity,
    classify_financial_health,
    compute_borrower_score,
    compute_trust_score,
)


class TestCalculateEMI:
    def test_standard_emi(self):
        """Standard 5L loan, 12% pa, 36 months."""
        emi = calculate_emi(500_000, 12.0, 36)
        assert 16_000 < emi < 17_000, f"EMI {emi} out of expected range"

    def test_zero_principal_returns_zero(self):
        assert calculate_emi(0, 12.0, 36) == 0.0

    def test_zero_tenure_returns_zero(self):
        assert calculate_emi(500_000, 12.0, 0) == 0.0

    def test_zero_interest_rate(self):
        """Zero interest = principal / tenure."""
        emi = calculate_emi(120_000, 0.0, 12)
        assert emi == pytest.approx(10_000.0, rel=1e-4)

    def test_emi_increases_with_rate(self):
        emi_low = calculate_emi(500_000, 8.0, 36)
        emi_high = calculate_emi(500_000, 18.0, 36)
        assert emi_high > emi_low


class TestCalculateDTI:
    def test_normal_dti(self):
        dti = calculate_dti(monthly_obligations=20_000, monthly_income=60_000)
        assert dti == pytest.approx(0.3333, rel=1e-2)

    def test_zero_income_returns_one(self):
        assert calculate_dti(10_000, 0) == 1.0

    def test_dti_above_one_possible(self):
        """DTI can exceed 1 for over-leveraged borrowers."""
        dti = calculate_dti(80_000, 60_000)
        assert dti > 1.0


class TestCalculateExpectedLoss:
    def test_standard_el(self):
        el = calculate_expected_loss(0.20, 500_000, lgd=0.45)
        assert el == pytest.approx(45_000.0, rel=1e-4)

    def test_zero_probability_zero_loss(self):
        assert calculate_expected_loss(0.0, 500_000) == 0.0

    def test_full_probability_max_loss(self):
        el = calculate_expected_loss(1.0, 500_000, lgd=0.45)
        assert el == pytest.approx(225_000.0, rel=1e-4)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            calculate_expected_loss(1.5, 500_000)

    def test_negative_loan_raises(self):
        with pytest.raises(ValueError):
            calculate_expected_loss(0.3, -100_000)


class TestClassifyFinancialHealth:
    def test_strong(self):
        assert classify_financial_health(0.20, 30_000, 3.5) == "STRONG"

    def test_moderate(self):
        assert classify_financial_health(0.45, 5_000, 1.6) == "MODERATE"

    def test_weak(self):
        assert classify_financial_health(0.60, 5_000, 1.2) == "WEAK"

    def test_critical_negative_capacity(self):
        assert classify_financial_health(0.40, -5_000, 0.9) == "CRITICAL"

    def test_critical_high_dti(self):
        assert classify_financial_health(0.80, 1_000, 2.0) == "CRITICAL"


class TestComputeBorrowerScore:
    def test_score_in_range(self):
        score = compute_borrower_score(
            credit_score_proxy_scaled=80.0,
            dti_ratio=0.30,
            repayment_capacity=20_000,
            monthly_income=60_000,
            employment_stability=0.80,
            cash_flow_health=2.5,
        )
        assert 0 <= score <= 100

    def test_high_score_for_strong_profile(self):
        score = compute_borrower_score(
            credit_score_proxy_scaled=95.0,
            dti_ratio=0.10,
            repayment_capacity=50_000,
            monthly_income=80_000,
            employment_stability=1.0,
            cash_flow_health=4.0,
        )
        assert score >= 70

    def test_low_score_for_weak_profile(self):
        score = compute_borrower_score(
            credit_score_proxy_scaled=10.0,
            dti_ratio=0.90,
            repayment_capacity=-5_000,
            monthly_income=20_000,
            employment_stability=0.05,
            cash_flow_health=0.5,
        )
        assert score <= 30


class TestCashFlowHealth:
    def test_healthy(self):
        ratio = calculate_cash_flow_health(60_000, 20_000)
        assert ratio == pytest.approx(3.0, rel=1e-4)

    def test_zero_obligations(self):
        ratio = calculate_cash_flow_health(60_000, 0)
        assert ratio == float("inf")

    def test_zero_income(self):
        assert calculate_cash_flow_health(0, 20_000) == 0.0
