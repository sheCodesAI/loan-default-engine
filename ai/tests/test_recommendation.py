"""Unit tests for the Recommendation Engine."""

import pytest
from ai.engines.recommendation import RecommendationEngine
from ai.schemas.input_schema import (
    BorrowerInput, EmploymentType, HomeOwnership,
    LoanInput, LoanGrade, LoanPurpose
)
from ai.schemas.output_schema import (
    Borrower360Output, BusinessRulesOutput, FinancialHealth,
    GeoResilienceOutput, ClimateImpact, RecoveryCapacity,
    RecommendationAction, RiskLevel,
)


def make_borrower(**kwargs) -> BorrowerInput:
    defaults = dict(
        age=35, annual_income=1_500_000,
        employment_type=EmploymentType.SALARIED,
        employment_length_months=72,
        home_ownership=HomeOwnership.OWN,
        credit_history_length_months=96,
    )
    defaults.update(kwargs)
    return BorrowerInput(**defaults)


def make_loan(**kwargs) -> LoanInput:
    defaults = dict(
        loan_amount=500_000, loan_tenure_months=36,
        loan_purpose=LoanPurpose.PERSONAL,
        interest_rate=10.5,
        loan_grade=LoanGrade.A,
    )
    defaults.update(kwargs)
    return LoanInput(**defaults)


def make_b360(score: float = 75.0, dti: float = 0.25) -> Borrower360Output:
    return Borrower360Output(
        borrower_score=score, dti_ratio=dti,
        repayment_capacity=40_000, cash_flow_health=3.0,
        trust_score=80.0, financial_health=FinancialHealth.STRONG,
        estimated_emi=16_000, disposable_income=40_000,
        loan_to_income_ratio=0.33,
    )


def make_geo(score: float = 75.0) -> GeoResilienceOutput:
    return GeoResilienceOutput(
        state="Maharashtra", geo_resilience_score=score,
        flood_risk=0.3, drought_risk=0.5, cyclone_risk=0.1, heatwave_risk=0.3,
        composite_climate_risk=0.32, climate_impact=ClimateImpact.MODERATE,
        recovery_capacity=RecoveryCapacity.HIGH, risk_adjustment=1.05,
        data_source="STATIC_OFFLINE",
    )


def make_rules(hard_reject: bool = False, soft: int = 0) -> BusinessRulesOutput:
    return BusinessRulesOutput(
        hard_reject=hard_reject,
        rule_violations=[],
        soft_violations_count=soft,
        hard_violations_count=1 if hard_reject else 0,
    )


class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine()

    def test_approve_low_risk_clean(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.LOW, default_probability=0.15,
            borrower_360=make_b360(75, 0.25), geo=make_geo(),
            rules=make_rules(), borrower=make_borrower(), loan=make_loan(),
        )
        assert rec.action == RecommendationAction.APPROVE

    def test_reject_hard_rule(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.MEDIUM, default_probability=0.40,
            borrower_360=make_b360(), geo=make_geo(),
            rules=make_rules(hard_reject=True),
            borrower=make_borrower(), loan=make_loan(),
        )
        assert rec.action == RecommendationAction.REJECT

    def test_reject_very_high_risk(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.VERY_HIGH, default_probability=0.82,
            borrower_360=make_b360(), geo=make_geo(),
            rules=make_rules(), borrower=make_borrower(), loan=make_loan(),
        )
        assert rec.action == RecommendationAction.REJECT

    def test_confidence_in_range(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.LOW, default_probability=0.15,
            borrower_360=make_b360(), geo=make_geo(),
            rules=make_rules(), borrower=make_borrower(), loan=make_loan(),
        )
        assert 0 <= rec.confidence <= 1

    def test_reasons_non_empty(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.MEDIUM, default_probability=0.40,
            borrower_360=make_b360(score=50, dti=0.55), geo=make_geo(),
            rules=make_rules(), borrower=make_borrower(), loan=make_loan(),
        )
        assert len(rec.reasons) >= 1

    def test_medium_risk_manual_review(self):
        rec = self.engine.generate(
            risk_level=RiskLevel.MEDIUM, default_probability=0.42,
            borrower_360=make_b360(score=62, dti=0.42), geo=make_geo(),
            rules=make_rules(soft=2), borrower=make_borrower(), loan=make_loan(),
        )
        assert rec.action in {
            RecommendationAction.MANUAL_REVIEW,
            RecommendationAction.INCREASE_TENURE,
            RecommendationAction.REQUEST_CO_APPLICANT,
            RecommendationAction.REQUEST_INSURANCE,
        }
