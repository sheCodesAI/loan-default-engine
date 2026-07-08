"""
Business Rules Engine for the IDBI AI module.

Applies configurable banking policy rules to the borrower and loan inputs.
Rules are classified as HARD (immediate reject) or SOFT (flag for review).

Design decisions:
  - Rules are configured in config.py — no code change needed to update thresholds.
  - The engine NEVER replaces the ML model. It operates in parallel and
    adds override signals that the Recommendation Engine uses.
  - Hard-reject rules represent regulatory or policy absolutes
    (e.g., minimum age, catastrophic DTI).
  - Soft rules represent risk signals requiring officer judgement.
"""

from __future__ import annotations

from typing import List

from ai.config import BUSINESS_RULES_CONFIG
from ai.logger import get_logger
from ai.schemas.input_schema import BorrowerInput, LoanInput
from ai.schemas.output_schema import BusinessRulesOutput, RuleSeverity, RuleViolation
from ai.utils.financial_utils import calculate_dti, calculate_emi, calculate_loan_to_income_ratio

logger = get_logger(__name__)


class BusinessRulesEngine:
    """
    Engine 3: Business Rules Engine.

    Evaluates a fixed set of banking policy rules against borrower and
    loan inputs. Rules are driven by BUSINESS_RULES_CONFIG in config.py.
    """

    def __init__(self, config: dict | None = None) -> None:
        """
        Initialize with rule configuration.

        Args:
            config: Override rule configuration dict.
                    Defaults to config.BUSINESS_RULES_CONFIG.
        """
        self.config = config or BUSINESS_RULES_CONFIG

    def evaluate(
        self,
        borrower: BorrowerInput,
        loan: LoanInput,
        credit_score_proxy: float = 50.0,
    ) -> BusinessRulesOutput:
        """
        Evaluate all business rules and return results.

        Args:
            borrower: Validated borrower input.
            loan: Validated loan input.
            credit_score_proxy: Credit score proxy 0–100 (from Borrower 360°).

        Returns:
            :class:`BusinessRulesOutput` with violations and flags.
        """
        logger.info("Evaluating business rules...")
        violations: List[RuleViolation] = []
        override_flags: List[str] = []

        monthly_income = borrower.monthly_income or (borrower.annual_income / 12)
        estimated_emi = calculate_emi(
            loan.loan_amount, loan.interest_rate, loan.loan_tenure_months
        )
        total_obligations = estimated_emi + loan.existing_monthly_obligations
        dti = calculate_dti(total_obligations, monthly_income)
        lti = calculate_loan_to_income_ratio(loan.loan_amount, borrower.annual_income)

        # Grade proxy: lower value = worse grade (0=best A, 100=worst G inverted)
        # For rules, use raw grade proxy (higher = worse credit)
        # credit_score_proxy_scaled: 100=A, 0=G
        grade_raw = (100 - credit_score_proxy) / (100 / 6)  # back to 0–6 scale

        # ── HARD rules ────────────────────────────────────────────────────
        # 1. Catastrophic credit grade (≥ G equivalent)
        hard_grade_thresh = self.config.get("hard_reject_credit_score_proxy", 1)
        if grade_raw >= (6 - hard_grade_thresh + 1):
            violations.append(RuleViolation(
                rule_name="HARD_REJECT_CREDIT_GRADE",
                severity=RuleSeverity.HARD,
                message="Borrower has the lowest credit grade (G) — hard reject per policy.",
                actual_value=f"Grade proxy: {grade_raw:.1f}",
                threshold_value=f"Max allowed: {hard_grade_thresh}",
            ))

        # 2. Catastrophic DTI
        hard_dti = self.config.get("hard_reject_dti", 0.85)
        if dti >= hard_dti:
            violations.append(RuleViolation(
                rule_name="HARD_REJECT_DTI",
                severity=RuleSeverity.HARD,
                message=f"DTI ratio {dti:.2%} exceeds hard-reject threshold of {hard_dti:.0%}.",
                actual_value=round(dti, 4),
                threshold_value=hard_dti,
            ))

        # 3. Age below minimum
        if borrower.age < self.config.get("min_age", 21):
            violations.append(RuleViolation(
                rule_name="HARD_REJECT_MIN_AGE",
                severity=RuleSeverity.HARD,
                message=f"Borrower age {borrower.age} is below minimum {self.config['min_age']}.",
                actual_value=borrower.age,
                threshold_value=self.config["min_age"],
            ))

        # ── SOFT rules ────────────────────────────────────────────────────
        # 4. DTI above soft threshold
        soft_dti = self.config.get("max_dti_ratio", 0.65)
        if dti >= soft_dti and dti < hard_dti:
            violations.append(RuleViolation(
                rule_name="SOFT_HIGH_DTI",
                severity=RuleSeverity.SOFT,
                message=f"DTI ratio {dti:.2%} exceeds recommended maximum of {soft_dti:.0%}.",
                actual_value=round(dti, 4),
                threshold_value=soft_dti,
            ))

        # 5. High loan-to-income ratio
        max_lti = self.config.get("max_loan_to_income_ratio", 20.0)
        if lti > max_lti:
            violations.append(RuleViolation(
                rule_name="SOFT_HIGH_LOAN_TO_INCOME",
                severity=RuleSeverity.SOFT,
                message=f"Loan-to-income ratio {lti:.1f}x exceeds maximum {max_lti:.0f}x.",
                actual_value=round(lti, 2),
                threshold_value=max_lti,
            ))

        # 6. Short employment duration
        min_emp = self.config.get("min_employment_months", 6)
        if borrower.employment_length_months < min_emp:
            violations.append(RuleViolation(
                rule_name="SOFT_SHORT_EMPLOYMENT",
                severity=RuleSeverity.SOFT,
                message=(
                    f"Employment length {borrower.employment_length_months} months "
                    f"is below recommended minimum of {min_emp} months."
                ),
                actual_value=borrower.employment_length_months,
                threshold_value=min_emp,
            ))

        # 7. Maximum age (near retirement risk)
        max_age = self.config.get("max_age", 70)
        if borrower.age > max_age:
            violations.append(RuleViolation(
                rule_name="SOFT_AGE_ABOVE_MAX",
                severity=RuleSeverity.SOFT,
                message=f"Borrower age {borrower.age} exceeds maximum recommended age of {max_age}.",
                actual_value=borrower.age,
                threshold_value=max_age,
            ))

        # 8. High-risk loan purpose
        high_risk_purposes = self.config.get("high_risk_purposes", ["VENTURE"])
        if loan.loan_purpose.value in high_risk_purposes:
            violations.append(RuleViolation(
                rule_name="SOFT_HIGH_RISK_PURPOSE",
                severity=RuleSeverity.SOFT,
                message=f"Loan purpose '{loan.loan_purpose.value}' is flagged as high risk.",
                actual_value=loan.loan_purpose.value,
                threshold_value="Low-risk purpose required for auto-approval",
            ))
            override_flags.append("HIGH_RISK_PURPOSE_REQUIRES_REVIEW")

        # 9. No co-applicant for high loan amount
        if loan.loan_amount > 1_000_000 and not loan.co_applicant:
            override_flags.append("LARGE_LOAN_NO_CO_APPLICANT")

        hard_count = sum(1 for v in violations if v.severity == RuleSeverity.HARD)
        soft_count = sum(1 for v in violations if v.severity == RuleSeverity.SOFT)
        hard_reject = hard_count > 0

        logger.info(
            "Business rules — hard: %d, soft: %d, hard_reject=%s",
            hard_count, soft_count, hard_reject,
        )

        return BusinessRulesOutput(
            hard_reject=hard_reject,
            rule_violations=violations,
            soft_violations_count=soft_count,
            hard_violations_count=hard_count,
            override_flags=override_flags,
        )
