"""
Borrower 360° Intelligence Engine.

Computes a comprehensive financial profile of the loan applicant
using only domain financial logic — no ML model involved.

Outputs a Borrower360Output containing:
  - DTI ratio
  - Repayment capacity
  - Cash flow health
  - Borrower score (0–100)
  - Trust score (0–100)
  - Financial health classification
  - Estimated EMI
  - Loan-to-income ratio
"""

from __future__ import annotations

from ai.logger import get_logger
from ai.schemas.input_schema import BorrowerInput, LoanInput
from ai.schemas.output_schema import Borrower360Output, FinancialHealth
from ai.utils.financial_utils import (
    calculate_cash_flow_health,
    calculate_dti,
    calculate_emi,
    calculate_loan_to_income_ratio,
    calculate_repayment_capacity,
    classify_financial_health,
    compute_borrower_score,
    compute_trust_score,
)

logger = get_logger(__name__)


class Borrower360Engine:
    """
    Engine 1: Borrower 360° Intelligence.

    Stateless — call compute() for each inference request.
    All financial logic is delegated to financial_utils.py.
    """

    def compute(
        self,
        borrower: BorrowerInput,
        loan: LoanInput,
    ) -> Borrower360Output:
        """
        Compute the full Borrower 360° profile.

        Args:
            borrower: Validated borrower input.
            loan: Validated loan input.

        Returns:
            :class:`Borrower360Output` with all computed scores.
        """
        logger.info("Computing Borrower 360° profile...")

        # ── EMI calculation ──────────────────────────────────────────────
        estimated_emi = calculate_emi(
            principal=loan.loan_amount,
            annual_rate_pct=loan.interest_rate,
            tenure_months=loan.loan_tenure_months,
        )

        monthly_income = borrower.monthly_income or (borrower.annual_income / 12)
        total_obligations = estimated_emi + loan.existing_monthly_obligations

        # ── Core ratios ──────────────────────────────────────────────────
        dti = calculate_dti(
            monthly_obligations=total_obligations,
            monthly_income=monthly_income,
        )

        repayment_capacity = calculate_repayment_capacity(
            monthly_income=monthly_income,
            monthly_emi=estimated_emi,
            other_obligations=loan.existing_monthly_obligations,
        )

        cash_flow_health = calculate_cash_flow_health(
            monthly_income=monthly_income,
            total_monthly_obligations=total_obligations,
        )

        lti = calculate_loan_to_income_ratio(
            loan_amount=loan.loan_amount,
            annual_income=borrower.annual_income,
        )

        # ── Credit score proxy ───────────────────────────────────────────
        if loan.loan_grade is not None:
            from ai.config import LOAN_GRADE_ORDER, LOAN_GRADE_TO_PROXY
            grade_idx = LOAN_GRADE_TO_PROXY.get(loan.loan_grade.value, 3)
            max_g = len(LOAN_GRADE_ORDER) - 1
            cs_scaled = (max_g - grade_idx) / max_g * 100
        elif borrower.credit_score is not None:
            # Map CIBIL 300–900 → 0–100
            cs_scaled = (borrower.credit_score - 300) / 600 * 100
        else:
            cs_scaled = 50.0   # Neutral default; FUTURE_INTEGRATION

        # ── Employment stability ─────────────────────────────────────────
        emp_stability = min(borrower.employment_length_months / 60, 1.0)

        # ── Composite scores ─────────────────────────────────────────────
        borrower_score = compute_borrower_score(
            credit_score_proxy_scaled=cs_scaled,
            dti_ratio=dti,
            repayment_capacity=repayment_capacity,
            monthly_income=monthly_income,
            employment_stability=emp_stability,
            cash_flow_health=cash_flow_health,
        )

        trust_score = compute_trust_score(
            credit_history_length_months=borrower.credit_history_length_months or 0,
            default_on_file=False,    # FUTURE_INTEGRATION: integrate credit bureau data
            employment_stability=emp_stability,
            credit_score_proxy_scaled=cs_scaled,
        )

        # ── Financial health classification ───────────────────────────────
        health_str = classify_financial_health(dti, repayment_capacity, cash_flow_health)
        financial_health = FinancialHealth(health_str)

        disposable_income = repayment_capacity  # After EMI + obligations

        output = Borrower360Output(
            borrower_score=borrower_score,
            dti_ratio=dti,
            repayment_capacity=repayment_capacity,
            cash_flow_health=cash_flow_health,
            trust_score=trust_score,
            financial_health=financial_health,
            estimated_emi=estimated_emi,
            disposable_income=disposable_income,
            loan_to_income_ratio=lti,
        )

        logger.info(
            "Borrower 360° complete — score=%.1f, DTI=%.2f, health=%s",
            borrower_score, dti, health_str,
        )
        return output
