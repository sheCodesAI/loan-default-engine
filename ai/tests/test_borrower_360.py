"""Unit tests for the Borrower 360° Intelligence Engine."""

import pytest
from ai.engines.borrower_360 import Borrower360Engine
from ai.schemas.input_schema import BorrowerInput, EmploymentType, HomeOwnership, LoanInput, LoanGrade, LoanPurpose


def make_borrower(**kwargs) -> BorrowerInput:
    defaults = dict(
        age=35,
        annual_income=1_200_000,
        employment_type=EmploymentType.SALARIED,
        employment_length_months=60,
        home_ownership=HomeOwnership.RENT,
        credit_history_length_months=84,
    )
    defaults.update(kwargs)
    return BorrowerInput(**defaults)


def make_loan(**kwargs) -> LoanInput:
    defaults = dict(
        loan_amount=500_000,
        loan_tenure_months=36,
        loan_purpose=LoanPurpose.PERSONAL,
        interest_rate=12.0,
        loan_grade=LoanGrade.B,
    )
    defaults.update(kwargs)
    return LoanInput(**defaults)


class TestBorrower360Engine:
    def setup_method(self):
        self.engine = Borrower360Engine()

    def test_compute_returns_output(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert result is not None

    def test_borrower_score_in_range(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert 0 <= result.borrower_score <= 100

    def test_trust_score_in_range(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert 0 <= result.trust_score <= 100

    def test_dti_positive(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert result.dti_ratio >= 0

    def test_emi_positive(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert result.estimated_emi > 0

    def test_high_income_high_score(self):
        rich = make_borrower(annual_income=5_000_000)
        result = self.engine.compute(rich, make_loan())
        poor = make_borrower(annual_income=200_000)
        result_poor = self.engine.compute(poor, make_loan())
        assert result.borrower_score > result_poor.borrower_score

    def test_financial_health_classification(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert result.financial_health.value in {"STRONG", "MODERATE", "WEAK", "CRITICAL"}

    def test_loan_to_income_ratio_positive(self):
        result = self.engine.compute(make_borrower(), make_loan())
        assert result.loan_to_income_ratio > 0

    def test_large_loan_increases_dti(self):
        small = self.engine.compute(make_borrower(), make_loan(loan_amount=200_000))
        large = self.engine.compute(make_borrower(), make_loan(loan_amount=2_000_000))
        assert large.dti_ratio > small.dti_ratio
