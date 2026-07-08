#!/usr/bin/env python3
"""
IDBI Bank — AI Credit Risk Intelligence Platform
Demo Script

Demonstrates the complete end-to-end AI inference pipeline for a hackathon
presentation. Loads the trained model, runs inference on a sample borrower,
displays all intelligence outputs, and runs a What-If simulation.

Usage:
    python demo.py

Requirements:
    pip install -r requirements.txt
    python -m ai.pipeline.train_pipeline   (run first to generate model)
"""

from __future__ import annotations

import sys
import os
from typing import Any

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── ANSI Colour Helpers ─────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"

def b(s: str) -> str:    return f"{C.BOLD}{s}{C.RESET}"
def cyan(s: str) -> str: return f"{C.CYAN}{s}{C.RESET}"
def grn(s: str) -> str:  return f"{C.GREEN}{s}{C.RESET}"
def yel(s: str) -> str:  return f"{C.YELLOW}{s}{C.RESET}"
def red(s: str) -> str:  return f"{C.RED}{s}{C.RESET}"
def dim(s: str) -> str:  return f"{C.DIM}{s}{C.RESET}"
def mag(s: str) -> str:  return f"{C.MAGENTA}{s}{C.RESET}"

def section(title: str, width: int = 62) -> None:
    pad = width - len(title) - 4
    print(f"\n{C.CYAN}{'━' * 2} {C.BOLD}{title}{C.RESET}{C.CYAN} {'━' * pad}{C.RESET}")

def header(title: str, width: int = 64) -> None:
    print(f"\n{C.BLUE}{'═' * width}{C.RESET}")
    pad = (width - len(title)) // 2
    print(f"{C.BLUE}║{C.RESET}{' ' * pad}{C.BOLD}{C.WHITE}{title}{C.RESET}{' ' * (width - pad - len(title))}{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}{'═' * width}{C.RESET}")

def row(label: str, value: str, width: int = 30) -> None:
    label_fmt = f"  {label:<{width}}"
    print(f"{dim(label_fmt)}: {b(value)}")

def risk_colour(level: str) -> str:
    colours = {"LOW": grn, "MEDIUM": yel, "HIGH": yel, "VERY_HIGH": red}
    fn = colours.get(level, str)
    return fn(b(level))

def inr(amount: float) -> str:
    if amount >= 100_000:
        return f"₹{amount/100_000:.2f}L"
    return f"₹{amount:,.0f}"

def pct(p: float) -> str:
    return f"{p*100:.1f}%"


def main() -> None:
    header("  IDBI BANK  ·  AI Credit Risk Intelligence Platform  ")

    # ── Load pipeline ───────────────────────────────────────────────────────
    print(f"\n{dim('Loading trained model and initializing engines...')}")
    try:
        from ai.pipeline.inference_pipeline import InferencePipeline
        from ai.simulator.what_if import WhatIfSimulator
        from ai.schemas.input_schema import (
            BorrowerInput, EmploymentType, GeoInput, HomeOwnership,
            InferenceRequest, LoanInput, LoanGrade, LoanPurpose, WhatIfOverrides,
        )
        pipeline = InferencePipeline()
        simulator = WhatIfSimulator(pipeline)
        print(grn("  ✔  Model loaded successfully"))
    except FileNotFoundError as e:
        print(red(f"\n  ✘  {e}"))
        print(yel("\n  Run the training pipeline first:"))
        print(b("      python -m ai.pipeline.train_pipeline"))
        sys.exit(1)

    # ── Sample Borrower Profile ─────────────────────────────────────────────
    section("BORROWER PROFILE")
    borrower = BorrowerInput(
        age=42,
        annual_income=1_800_000,          # ₹18 L / year
        employment_type=EmploymentType.SALARIED,
        employment_length_months=96,       # 8 years
        home_ownership=HomeOwnership.MORTGAGE,
        credit_score=720,
        credit_history_length_months=144,  # 12 years
        existing_loans_count=1,
        dependents=2,
    )
    loan = LoanInput(
        loan_amount=1_200_000,             # ₹12 L
        loan_tenure_months=60,             # 5 years
        loan_purpose=LoanPurpose.HOME,
        interest_rate=10.5,
        loan_grade=LoanGrade.B,
        co_applicant=False,
        insurance=False,
        existing_monthly_obligations=12_000,
    )
    geo = GeoInput(state="Maharashtra")

    request = InferenceRequest(
        request_id="DEMO-2026-001",
        borrower=borrower,
        loan=loan,
        geo=geo,
    )

    row("Borrower ID",        "DEMO-2026-001")
    row("Age",                f"{borrower.age} years")
    row("Employment",         f"{borrower.employment_type.value}  ({borrower.employment_length_months // 12} yrs experience)")
    row("Annual Income",      inr(borrower.annual_income))
    row("Credit Score",       str(borrower.credit_score))
    row("State",              geo.state)
    row("Loan Amount",        inr(loan.loan_amount))
    row("Loan Purpose",       loan.loan_purpose.value)
    row("Tenure",             f"{loan.loan_tenure_months} months")
    row("Interest Rate",      f"{loan.interest_rate}%")

    # ── Run Inference ───────────────────────────────────────────────────────
    print(f"\n{dim('  Running AI inference pipeline...')}", end="", flush=True)
    result = pipeline.run(request)
    print(f"\r{grn('  ✔  Inference complete')}{' ' * 20}")

    # ── Borrower 360° Intelligence ──────────────────────────────────────────
    section("BORROWER 360° INTELLIGENCE")
    b360 = result.borrower_360
    score_str = f"{b360.borrower_score:.1f} / 100"
    row("Borrower Score",      score_str)
    row("Financial Health",    b(b360.financial_health.value if hasattr(b360.financial_health, 'value') else str(b360.financial_health)))
    row("Cash Flow Health",    f"{min(float(b360.cash_flow_health), 1.0):.3f}")
    row("Repayment Capacity",  inr(b360.repayment_capacity) + "/month")
    row("Trust Score",         f"{b360.trust_score:.1f} / 100")
    row("DTI Ratio",           pct(b360.dti_ratio))
    row("Estimated EMI",       inr(b360.estimated_emi) + "/month")
    row("Loan-to-Income",      f"{b360.loan_to_income_ratio:.2f}×")

    # ── Geo & Resilience Intelligence ───────────────────────────────────────
    section("GEO & RESILIENCE INTELLIGENCE")
    geo_r = result.geo_resilience
    row("State",                geo_r.state)
    row("Resilience Score",     f"{geo_r.geo_resilience_score:.1f} / 100")
    row("Climate Impact",       b(geo_r.climate_impact.value))
    row("Flood Risk",           f"{geo_r.flood_risk:.2f}")
    row("Drought Risk",         f"{geo_r.drought_risk:.2f}")
    row("Cyclone Risk",         f"{geo_r.cyclone_risk:.2f}")
    row("Risk Adjustment",      f"+{geo_r.risk_adjustment:.3f}\u00d7")

    # ── AI Risk Prediction ──────────────────────────────────────────────────
    section("AI RISK PREDICTION")
    prob = result.default_probability
    prob_str = f"{prob:.1%}"
    if prob < 0.35:
        prob_display = grn(b(prob_str))
    elif prob < 0.60:
        prob_display = yel(b(prob_str))
    else:
        prob_display = red(b(prob_str))

    print(f"  {'Default Probability':<30}: {prob_display}")
    row("Risk Level",           risk_colour(result.risk_level.value))
    row("Expected Loss",        inr(result.expected_loss))

    # ── SHAP Top Risk Factors ───────────────────────────────────────────────
    section("TOP RISK FACTORS  (SHAP Explainability)")
    shap_out = result.shap_explanation
    for i, factor in enumerate(shap_out.top_risk_factors[:8], 1):
        direction = factor.direction.value if hasattr(factor.direction, 'value') else str(factor.direction)
        direction_icon = "↑" if direction == "INCREASES_RISK" else "↓"
        direction_color = red if direction == "INCREASES_RISK" else grn
        print(
            f"  {i:2}. {factor.human_readable_name:<26} "
            f"{direction_color(direction_icon + ' ' + direction.replace('_', ' '))}  "
            f"{dim('shap=' + f'{factor.shap_value:+.4f}')}"
        )

    # ── Business Rules Engine ───────────────────────────────────────────────
    section("BUSINESS RULES ENGINE")
    br = result.business_rules
    hard_str = red(str(br.hard_violations_count)) if br.hard_violations_count > 0 else grn("0")
    soft_str = yel(str(br.soft_violations_count)) if br.soft_violations_count > 0 else grn("0")
    row("Hard Reject",          red("YES ✘") if br.hard_reject else grn("NO  ✔"))
    row("Hard Violations",      hard_str)
    row("Soft Violations",      soft_str)
    if br.rule_violations:
        print(f"  {dim('Violations:')}")
        for v in br.rule_violations:
            severity_col = red if v.severity == "HARD" else yel
            print(f"    {severity_col('•')} [{v.severity}] {v.rule_name}: {dim(v.message)}")


    # ── Final Recommendation ────────────────────────────────────────────────
    section("FINAL AI RECOMMENDATION")
    rec = result.recommendation
    action_colours = {
        "APPROVE": grn, "MANUAL_REVIEW": yel, "REJECT": red,
        "REDUCE_LOAN_AMOUNT": yel, "INCREASE_TENURE": yel,
        "REQUEST_CO_APPLICANT": yel, "REQUEST_INSURANCE": yel,
    }
    action_fn = action_colours.get(rec.action.value, str)
    print(f"  {'Decision':<30}: {action_fn(b(rec.action.value))}")
    row("Confidence",           pct(rec.confidence))
    if rec.reasons:
        print(f"  {dim('Reasons:')}")
        for r in rec.reasons[:4]:
            print(f"    • {r}")

    # ── What-If Simulation ──────────────────────────────────────────────────
    print(f"\n{C.BLUE}{'═' * 64}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  🔮  WHAT-IF SIMULATOR{C.RESET}")
    print(f"{C.BLUE}{'═' * 64}{C.RESET}")
    print(f"\n  {dim('Scenario: Add co-applicant + request insurance')}")
    print(f"  {dim('Goal    : Can this change the recommendation?')}\n")

    whatif_overrides = WhatIfOverrides(
        co_applicant=True,
        insurance=True,
        loan_amount=900_000,    # Reduce loan from ₹12L to ₹9L
    )

    wi = simulator.simulate(request, whatif_overrides, scenario_name="Add Co-applicant + Insurance + Reduce Loan")

    # Table header
    print(f"  {'Metric':<28} {'BEFORE':>12} {'AFTER':>12} {'DELTA':>10}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*10}")

    def delta_str(d: float, is_pct: bool = True) -> str:
        sign = "+" if d > 0 else ""
        if is_pct:
            s = f"{sign}{d:.1%}"
        else:
            s = f"{sign}{d/100_000:.2f}L"
        return (red if d > 0 else grn)(s) if d != 0 else dim("—")

    base = wi.base_scenario
    mod  = wi.modified_scenario
    print(f"  {'Default Probability':<28} {pct(base.default_probability):>12} {pct(mod.default_probability):>12} {delta_str(wi.delta_probability):>22}")
    print(f"  {'Expected Loss':<28} {inr(base.expected_loss):>12} {inr(mod.expected_loss):>12} {delta_str(wi.delta_expected_loss, is_pct=False):>22}")
    print(f"  {'Risk Level':<28} {base.risk_level.value:>12} {mod.risk_level.value:>12} {'changed' if wi.risk_level_changed else dim('unchanged'):>22}")
    print(f"  {'Recommendation':<28} {base.recommendation.value:>12} {mod.recommendation.value:>12} {'changed' if wi.recommendation_changed else dim('unchanged'):>22}")

    print(f"\n  {C.CYAN}Insight:{C.RESET}")
    # Word-wrap the insight
    words = wi.insight.split()
    line, lines = "  ", []
    for word in words:
        if len(line) + len(word) > 68:
            lines.append(line)
            line = "    " + word + " "
        else:
            line += word + " "
    lines.append(line)
    for l in lines:
        print(l)

    # ── Summary ─────────────────────────────────────────────────────────────
    section("AI SUMMARY")
    print(f"\n  {result.summary}")
    print()
    print(f"  {dim('Timestamp:')} {result.timestamp}")
    print(f"  {dim('Request ID:')} {result.request_id}")
    print()
    print(f"{C.BLUE}{'═' * 64}{C.RESET}")
    print(f"  {grn('✔')} IDBI AI Credit Risk Intelligence Platform  ·  Demo Complete")
    print(f"{C.BLUE}{'═' * 64}{C.RESET}\n")


if __name__ == "__main__":
    main()
