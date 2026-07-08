"""Unit tests for the Business Rules Engine."""

import pytest
from ai.engines.business_rules import BusinessRulesEngine
from ai.schemas.input_schema import (
    BorrowerInput, EmploymentType, HomeOwnership, LoanInput, LoanGrade, LoanPurpose
)


def make_borrower(**kwargs) -> BorrowerInput:
    defaults = dict(
        age=35, annual_income=1_200_000,
        employment_type=EmploymentType.SALARIED,
        employment_length_months=60,
        home_ownership=HomeOwnership.RENT,
    )
    defaults.update(kwargs)
    return BorrowerInput(**defaults)


def make_loan(**kwargs) -> LoanInput:
    defaults = dict(
        loan_amount=500_000, loan_tenure_months=36,
        loan_purpose=LoanPurpose.PERSONAL,
        interest_rate=12.0,
        loan_grade=LoanGrade.B,
    )
    defaults.update(kwargs)
    return LoanInput(**defaults)


class TestBusinessRulesEngine:
    def setup_method(self):
        self.engine = BusinessRulesEngine()

    def test_clean_borrower_no_violations(self):
        result = self.engine.evaluate(make_borrower(), make_loan(), credit_score_proxy=80.0)
        assert not result.hard_reject
        assert result.hard_violations_count == 0
        assert result.soft_violations_count == 0

    def test_very_young_borrower_hard_reject(self):
        """Age below minimum should trigger hard reject.
        Use model_construct to bypass Pydantic schema validation so we
        can test the business rules engine's age check directly.
        """
        # model_construct skips Pydantic validators — lets us test edge cases
        young = BorrowerInput.model_construct(
            age=19, annual_income=800_000,
            employment_type=EmploymentType.SALARIED,
            employment_length_months=60,
            home_ownership=HomeOwnership.RENT,
            monthly_income=66_666,
            existing_loans_count=0, dependents=0,
        )
        result = self.engine.evaluate(young, make_loan())
        assert result.hard_reject
        assert result.hard_violations_count >= 1
        rule_names = [v.rule_name for v in result.rule_violations]
        assert "HARD_REJECT_MIN_AGE" in rule_names

    def test_high_dti_soft_violation(self):
        """Very high EMI relative to income should trigger soft DTI violation."""
        # Force high DTI: small income, large loan
        poor_borrower = make_borrower(annual_income=200_000)
        big_loan = make_loan(loan_amount=2_000_000, loan_tenure_months=12, interest_rate=18.0)
        result = self.engine.evaluate(poor_borrower, big_loan, credit_score_proxy=70.0)
        rule_names = [v.rule_name for v in result.rule_violations]
        # Should have soft OR hard DTI violation
        assert any("DTI" in r for r in rule_names)

    def test_short_employment_soft_violation(self):
        new_employee = make_borrower(employment_length_months=2)
        result = self.engine.evaluate(new_employee, make_loan(), credit_score_proxy=70.0)
        rule_names = [v.rule_name for v in result.rule_violations]
        assert "SOFT_SHORT_EMPLOYMENT" in rule_names

    def test_vehicle_loan_soft_violation(self):
        vehicle = make_loan(loan_purpose=LoanPurpose.VEHICLE)
        result = self.engine.evaluate(make_borrower(), vehicle, credit_score_proxy=70.0)
        rule_names = [v.rule_name for v in result.rule_violations]
        assert "SOFT_HIGH_RISK_PURPOSE" in rule_names

    def test_hard_reject_flag_when_hard_violation(self):
        young = BorrowerInput.model_construct(
            age=19, annual_income=800_000,
            employment_type=EmploymentType.SALARIED,
            employment_length_months=60,
            home_ownership=HomeOwnership.RENT,
            monthly_income=66_666,
            existing_loans_count=0, dependents=0,
        )
        result = self.engine.evaluate(young, make_loan())
        assert result.hard_reject is True

    def test_no_hard_reject_for_soft_violations_only(self):
        new_emp = make_borrower(employment_length_months=3)
        result = self.engine.evaluate(new_emp, make_loan(), credit_score_proxy=70.0)
        assert not result.hard_reject

    def test_custom_config_override(self):
        custom_config = {**self.engine.config, "min_employment_months": 24}
        engine = BusinessRulesEngine(config=custom_config)
        borrower = make_borrower(employment_length_months=18)
        result = engine.evaluate(borrower, make_loan(), credit_score_proxy=70.0)
        rule_names = [v.rule_name for v in result.rule_violations]
        assert "SOFT_SHORT_EMPLOYMENT" in rule_names
