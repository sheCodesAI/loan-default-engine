"""
Input schemas for the IDBI AI inference pipeline.

All inference requests must be validated against these Pydantic v2 models
before entering any engine. This enforces type safety and provides clear
error messages to the API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class EmploymentType(str, Enum):
    SALARIED = "Salaried"
    SELF_EMPLOYED = "SelfEmployed"
    BUSINESS = "Business"
    FARMER = "Farmer"
    UNEMPLOYED = "Unemployed"
    FREELANCER = "Freelancer"
    RETIRED = "Retired"


class HomeOwnership(str, Enum):
    OWN = "Own"
    RENT = "Rent"
    MORTGAGE = "Mortgage"
    OTHER = "Other"


class LoanPurpose(str, Enum):
    PERSONAL = "Personal"
    EDUCATION = "Education"
    VEHICLE = "Vehicle"
    HOME = "Home"
    AGRI = "Agri"
    BUSINESS = "Business"
    MEDICAL = "Medical"
    VENTURE = "Venture"
    HOME_IMPROVEMENT = "HomeImprovement"
    DEBT_CONSOLIDATION = "DebtConsolidation"


class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"



# ─── Sub-models ───────────────────────────────────────────────────────────────

class BorrowerInput(BaseModel):
    """Financial and demographic profile of the loan applicant."""

    age: int = Field(..., ge=18, le=100, description="Borrower age in years")
    annual_income: float = Field(..., gt=0, description="Annual gross income in INR")
    monthly_income: Optional[float] = Field(
        None, description="Monthly income — derived from annual_income if not provided"
    )
    employment_type: EmploymentType = Field(..., description="Nature of employment")
    employment_length_months: int = Field(
        ..., ge=0, description="Total months in current employment"
    )
    home_ownership: HomeOwnership
    credit_score: Optional[int] = Field(
        None,
        ge=300,
        le=900,
        description="CIBIL score. FUTURE_INTEGRATION — use loan_grade as proxy if absent.",
    )
    credit_history_length_months: Optional[int] = Field(
        None, ge=0, description="Length of credit history in months"
    )
    existing_loans_count: int = Field(
        0, ge=0, description="Number of existing active loans. FUTURE_INTEGRATION."
    )
    dependents: int = Field(0, ge=0, description="Number of financial dependents")
    marital_status: Optional[str] = Field(
        None, description="FUTURE_INTEGRATION — marital status"
    )

    @model_validator(mode="after")
    def derive_monthly_income(self) -> "BorrowerInput":
        """Auto-derive monthly income from annual income when not provided."""
        if self.monthly_income is None:
            self.monthly_income = round(self.annual_income / 12, 2)
        return self

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 21:
            raise ValueError("Borrower must be at least 21 years old per bank policy.")
        return v


class LoanInput(BaseModel):
    """Details of the loan being applied for."""

    loan_amount: float = Field(..., gt=0, description="Requested loan amount in INR")
    loan_tenure_months: int = Field(
        ..., gt=0, le=360, description="Loan repayment period in months"
    )
    loan_purpose: LoanPurpose
    interest_rate: float = Field(
        ..., gt=0, le=100, description="Annual interest rate as percentage"
    )
    loan_grade: Optional[LoanGrade] = Field(
        None, description="Bank-assigned loan grade A–G"
    )
    co_applicant: bool = Field(False, description="Whether a co-applicant is present")
    insurance: bool = Field(
        False, description="Whether loan insurance is taken"
    )
    existing_monthly_obligations: float = Field(
        0.0, ge=0, description="Total existing monthly loan EMIs in INR"
    )

    @field_validator("interest_rate")
    @classmethod
    def validate_interest_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Interest rate must be positive.")
        return v


class GeoInput(BaseModel):
    """Geographic location of the borrower — used by Geo & Resilience Engine."""

    state: str = Field(..., description="Indian state (e.g., Maharashtra, Bihar)")
    district: Optional[str] = Field(None, description="District name")
    pin_code: Optional[str] = Field(None, description="6-digit PIN code")

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("pin_code")
    @classmethod
    def validate_pin_code(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.isdigit():
            raise ValueError("PIN code must contain only digits.")
        if v is not None and len(v) != 6:
            raise ValueError("PIN code must be exactly 6 digits.")
        return v


# ─── Top-level Request ────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """
    Complete inference request combining borrower, loan, and geo information.

    This is the single entry point into the inference pipeline.
    """

    request_id: Optional[str] = Field(
        None, description="Optional unique ID for tracking and audit."
    )
    borrower: BorrowerInput
    loan: LoanInput
    geo: GeoInput

    model_config = {"str_strip_whitespace": True}


class WhatIfOverrides(BaseModel):
    """
    Scenario overrides for the What-If Simulator.

    Only provide the fields you want to change; all others retain
    their values from the base InferenceRequest.
    """

    loan_amount: Optional[float] = Field(None, gt=0)
    loan_tenure_months: Optional[int] = Field(None, gt=0, le=360)
    co_applicant: Optional[bool] = None
    insurance: Optional[bool] = None
    interest_rate: Optional[float] = Field(None, gt=0, le=100)
