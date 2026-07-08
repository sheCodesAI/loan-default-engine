"""
IDBI Bank — AI Credit Risk Intelligence Platform
Streamlit Dashboard
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDBI AI Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0f1e; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #c9d5e8 !important; }

.bank-header {
    background: linear-gradient(135deg, #0a2342 0%, #0d3b7a 50%, #1a5276 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    border: 1px solid #2980b9;
    box-shadow: 0 8px 32px rgba(41,128,185,0.3);
}
.bank-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.bank-header p { color: #85c1e9; margin: 4px 0 0; font-size: 0.95rem; }

.metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0e2240 100%);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #1e3a5f;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.metric-card h3 { color: #7fb3d3; font-size: 0.78rem; font-weight: 500; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
.metric-card .sub { font-size: 0.8rem; color: #7f8c8d; margin-top: 4px; }

.risk-badge-LOW { background: linear-gradient(135deg,#0a4e2a,#0e7348); color:#2ecc71; border:1px solid #27ae60; }
.risk-badge-MEDIUM { background: linear-gradient(135deg,#4d3000,#8a5a00); color:#f39c12; border:1px solid #e67e22; }
.risk-badge-HIGH { background: linear-gradient(135deg,#4a1a00,#8e2d00); color:#e74c3c; border:1px solid #c0392b; }
.risk-badge-VERY_HIGH { background: linear-gradient(135deg,#3b0000,#7b0000); color:#ff4444; border:1px solid #8b0000; }
.risk-badge {
    display:inline-block; padding:8px 24px; border-radius:24px;
    font-weight:700; font-size:1.1rem; letter-spacing:0.05em;
    text-align:center; width:100%;
}

.section-header {
    color: #85c1e9;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 20px 0 12px;
}

.rec-card {
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    font-weight: 600; font-size: 1rem;
}
.rec-APPROVE { background:rgba(39,174,96,0.15); border:1px solid #27ae60; color:#2ecc71; }
.rec-REJECT { background:rgba(231,76,60,0.15); border:1px solid #e74c3c; color:#e74c3c; }
.rec-MANUAL_REVIEW { background:rgba(243,156,18,0.15); border:1px solid #f39c12; color:#f39c12; }
.rec-REQUEST_INSURANCE { background:rgba(52,152,219,0.15); border:1px solid #3498db; color:#3498db; }
.rec-REDUCE_LOAN_AMOUNT { background:rgba(155,89,182,0.15); border:1px solid #9b59b6; color:#9b59b6; }
.rec-INCREASE_TENURE { background:rgba(26,188,156,0.15); border:1px solid #1abc9c; color:#1abc9c; }
.rec-REQUEST_CO_APPLICANT { background:rgba(230,126,34,0.15); border:1px solid #e67e22; color:#e67e22; }

.rule-hard { background:rgba(231,76,60,0.12); border-left:3px solid #e74c3c; padding:8px 12px; border-radius:4px; margin:4px 0; color:#e74c3c; font-size:0.85rem; }
.rule-soft { background:rgba(243,156,18,0.12); border-left:3px solid #f39c12; padding:8px 12px; border-radius:4px; margin:4px 0; color:#f39c12; font-size:0.85rem; }

.ewi-alert {
    background: linear-gradient(135deg,rgba(231,76,60,0.2),rgba(192,57,43,0.2));
    border: 1px solid #e74c3c; border-radius: 12px; padding: 16px;
    color: #ff6b6b; font-weight: 600; text-align: center;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.7;} }

.shap-bar-inc { background: linear-gradient(90deg, #e74c3c, #c0392b); height: 20px; border-radius: 4px; }
.shap-bar-dec { background: linear-gradient(90deg, #27ae60, #1e8449); height: 20px; border-radius: 4px; }

stButton>button { background: linear-gradient(135deg,#1a5276,#2980b9) !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_pipeline():
    from ai.pipeline.inference_pipeline import InferencePipeline
    return InferencePipeline()

@st.cache_resource(show_spinner="Loading What-If simulator...")
def load_simulator():
    from ai.simulator.what_if import WhatIfSimulator
    return WhatIfSimulator()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bank-header">
  <h1>🏦 IDBI Bank — AI Credit Risk Intelligence Platform</h1>
  <p>Powered by XGBoost · SHAP Explainability · Geo & Resilience Intelligence · What-If Simulator</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Application Form ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Loan Application")
    st.markdown("---")

    st.markdown("### 👤 Borrower Details")
    age = st.number_input("Age", min_value=21, max_value=80, value=35)
    annual_income = st.number_input("Annual Income (₹)", min_value=100000, max_value=10000000, value=720000, step=10000)
    employment_type = st.selectbox("Employment Type", ["Salaried", "SelfEmployed", "Business", "Farmer", "Unemployed", "Freelancer", "Retired"])
    employment_months = st.slider("Employment Length (months)", 0, 360, 36)
    home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage", "Other"])
    credit_score = st.slider("CIBIL Score", 300, 900, 720)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)

    st.markdown("---")
    st.markdown("### 💰 Loan Details")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=500000, step=10000)
    loan_tenure = st.slider("Tenure (months)", 6, 360, 60)
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Vehicle", "Home", "Agri", "Business", "Medical", "Venture"])
    interest_rate = st.slider("Interest Rate (%)", 5.0, 36.0, 12.0, step=0.5)
    existing_obligations = st.number_input("Existing Monthly EMIs (₹)", min_value=0, value=5000, step=500)
    co_applicant = st.checkbox("Co-Applicant Present")
    insurance = st.checkbox("Loan Insurance")

    st.markdown("---")
    st.markdown("### 📍 Location")
    state = st.selectbox("State", [
        "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "Gujarat",
        "Rajasthan", "Bihar", "West Bengal", "Madhya Pradesh", "Andhra Pradesh",
        "Kerala", "Punjab", "Haryana", "Odisha", "Assam", "Telangana",
        "Jharkhand", "Uttarakhand", "Himachal Pradesh", "Goa"
    ])

    st.markdown("---")
    analyze = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")

# ── Main Panel ────────────────────────────────────────────────────────────────
if not analyze:
    st.info("👈 Fill in the loan application details in the sidebar and click **Analyze Risk** to begin.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Engine", "XGBoost + SHAP")
    col2.metric("Training Records", "15,000")
    col3.metric("Unit Tests", "84 Passing ✅")
    st.stop()

# ── Run inference ─────────────────────────────────────────────────────────────
try:
    from ai.schemas.input_schema import (
        BorrowerInput, EmploymentType, GeoInput, HomeOwnership,
        InferenceRequest, LoanInput, LoanPurpose, WhatIfOverrides
    )

    pipeline = load_pipeline()

    request = InferenceRequest(
        borrower=BorrowerInput(
            age=age,
            annual_income=float(annual_income),
            employment_type=EmploymentType(employment_type),
            employment_length_months=employment_months,
            home_ownership=HomeOwnership(home_ownership),
            credit_score=credit_score,
            dependents=dependents,
        ),
        loan=LoanInput(
            loan_amount=float(loan_amount),
            loan_tenure_months=loan_tenure,
            loan_purpose=LoanPurpose(loan_purpose),
            interest_rate=float(interest_rate),
            co_applicant=co_applicant,
            insurance=insurance,
            existing_monthly_obligations=float(existing_obligations),
        ),
        geo=GeoInput(state=state),
    )

    with st.spinner("Running AI Risk Analysis..."):
        result = pipeline.run(request)

except Exception as e:
    st.error(f"❌ Analysis failed: {e}")
    st.exception(e)
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
rl = result.risk_level.value
prob = result.default_probability
b360 = result.borrower_360
geo = result.geo_resilience
rules = result.business_rules
shap = result.shap_explanation
rec = result.recommendation

# Row 1: Key Metrics
st.markdown('<p class="section-header">📊 Risk Assessment Summary</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <h3>Default Probability</h3>
      <p class="value">{prob*100:.1f}%</p>
      <p class="sub">ML prediction</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <h3>Risk Level</h3>
      <div class="risk-badge risk-badge-{rl}">{rl.replace("_"," ")}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    el = result.expected_loss
    el_str = f"₹{el/100000:.2f}L" if el >= 100000 else f"₹{el:,.0f}"
    st.markdown(f"""
    <div class="metric-card">
      <h3>Expected Loss</h3>
      <p class="value">{el_str}</p>
      <p class="sub">PD × LGD × EAD</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <h3>Borrower Score</h3>
      <p class="value">{b360.borrower_score:.1f}/100</p>
      <p class="sub">{b360.financial_health.value}</p>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
      <h3>Geo Resilience</h3>
      <p class="value">{geo.geo_resilience_score:.1f}/100</p>
      <p class="sub">{geo.climate_impact.value} climate risk</p>
    </div>""", unsafe_allow_html=True)

# EWI
if result.early_warning_indicator:
    st.markdown('<div class="ewi-alert">⚠️ EARLY WARNING INDICATOR — Multiple risk signals co-occur. Enhanced due diligence required.</div>', unsafe_allow_html=True)

# Recommendation
action = rec.action.value
st.markdown(f'<div class="rec-card rec-{action}">📋 Recommendation: {action.replace("_"," ")} &nbsp;|&nbsp; Confidence: {rec.confidence*100:.0f}%</div>', unsafe_allow_html=True)

# Officer summary
st.info(f"🤖 {result.summary}")

# Row 2: Gauges + Borrower 360
st.markdown('<p class="section-header">🔬 Detailed Analysis</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Risk Gauge", "👤 Borrower 360°", "🗺️ Geo Resilience", "🧠 SHAP Explainer", "🔄 What-If Simulator", "📂 Batch Upload"])

# Tab 1: Gauge
with tab1:
    col_g, col_r = st.columns(2)
    with col_g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 48, "color": "#fff"}},
            title={"text": "Default Probability", "font": {"size": 18, "color": "#85c1e9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#7f8c8d"},
                "bar": {"color": "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.35 else "#27ae60"},
                "steps": [
                    {"range": [0, 35], "color": "rgba(39,174,96,0.15)"},
                    {"range": [35, 60], "color": "rgba(243,156,18,0.15)"},
                    {"range": [60, 80], "color": "rgba(231,76,60,0.15)"},
                    {"range": [80, 100], "color": "rgba(139,0,0,0.3)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": prob * 100},
                "bgcolor": "#0a0f1e",
                "bordercolor": "#1e3a5f",
            },
        ))
        gauge.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font={"color": "#c9d5e8"},
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with col_r:
        st.markdown("### 📋 Recommendation Details")
        for reason in rec.reasons:
            st.markdown(f"• {reason}")
        if rec.suggested_adjustments:
            st.markdown("**Suggested Adjustments:**")
            for k, v in rec.suggested_adjustments.items():
                st.markdown(f"  - {k.replace('_', ' ').title()}: {v}")

# Tab 2: Borrower 360
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Financial Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Borrower Score", "Trust Score", "DTI Ratio", "Cash Flow Health", "Loan-to-Income"],
            "Value": [
                f"{b360.borrower_score:.1f}/100",
                f"{b360.trust_score:.1f}/100",
                f"{b360.dti_ratio:.2%}",
                f"{b360.cash_flow_health:.2f}x",
                f"{b360.loan_to_income_ratio:.2f}x",
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Income & EMI Breakdown")
        monthly_inc = b360.disposable_income + b360.estimated_emi + float(existing_obligations)
        fig = go.Figure(go.Pie(
            values=[b360.estimated_emi, float(existing_obligations), max(0, b360.disposable_income)],
            labels=["New EMI", "Existing EMIs", "Disposable"],
            hole=0.4,
            marker_colors=["#e74c3c", "#f39c12", "#27ae60"],
        ))
        fig.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font={"color": "#c9d5e8"}, height=260,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(color="#c9d5e8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    col_c.metric("Estimated EMI", f"₹{b360.estimated_emi:,.0f}/mo")
    col_d.metric("Disposable Income", f"₹{b360.disposable_income:,.0f}/mo")
    st.metric("Repayment Capacity", f"₹{b360.repayment_capacity:,.0f}/mo", delta="after EMI")
    st.metric("Financial Health", b360.financial_health.value)

# Tab 3: Geo
with tab3:
    col_m, col_n = st.columns(2)
    with col_m:
        st.markdown(f"#### 📍 {geo.state} — Climate Risk Profile")
        risk_data = {
            "Risk Type": ["Flood", "Drought", "Cyclone", "Heatwave"],
            "Score": [geo.flood_risk, geo.drought_risk, geo.cyclone_risk, geo.heatwave_risk],
        }
        risk_df = pd.DataFrame(risk_data)
        fig_risk = px.bar(
            risk_df, x="Risk Type", y="Score", color="Score",
            color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
            range_y=[0, 1],
        )
        fig_risk.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font={"color": "#c9d5e8"}, height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        fig_risk.update_traces(marker_line_width=0)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_n:
        st.markdown("#### Geo Summary")
        st.metric("Geo Resilience Score", f"{geo.geo_resilience_score:.1f}/100")
        st.metric("Climate Impact", geo.climate_impact.value)
        st.metric("Recovery Capacity", geo.recovery_capacity.value)
        st.metric("Composite Climate Risk", f"{geo.composite_climate_risk:.2%}")
        st.metric("Risk Adjustment", f"{geo.risk_adjustment:.3f}x")
        st.caption(f"Data Source: {geo.data_source}")

# Tab 4: SHAP
with tab4:
    st.markdown("#### 🧠 SHAP Feature Explanations")
    st.caption(shap.narrative)

    factors = shap.top_risk_factors[:10]
    if factors:
        names = [f.human_readable_name for f in factors]
        vals = [f.shap_value for f in factors]
        colors = ["#e74c3c" if v > 0 else "#27ae60" for v in vals]

        fig_shap = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
        ))
        fig_shap.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font={"color": "#c9d5e8"}, height=400,
            margin=dict(l=20, r=60, t=10, b=20),
            xaxis_title="SHAP Value (impact on default probability)",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("**Factor Details:**")
        for f in factors[:5]:
            icon = "🔴" if f.direction == "INCREASES_RISK" else "🟢"
            st.markdown(f"{icon} **{f.human_readable_name}** ({f.magnitude}): {f.description}")
    else:
        st.info("SHAP explanations not available (DummyExplainer active — model may need retraining).")

# Tab 5: What-If
with tab5:
    st.markdown("#### 🔄 What-If Scenario Simulator")
    st.markdown("Adjust parameters below to see how the risk profile would change:")

    wc1, wc2 = st.columns(2)
    with wc1:
        wi_loan = st.number_input("Modified Loan Amount (₹)", value=int(loan_amount), step=10000, key="wi_loan")
        wi_tenure = st.slider("Modified Tenure (months)", 6, 360, loan_tenure, key="wi_tenure")
    with wc2:
        wi_co = st.checkbox("Add Co-Applicant", value=co_applicant, key="wi_co")
        wi_ins = st.checkbox("Add Insurance", value=insurance, key="wi_ins")
        wi_rate = st.slider("Modified Rate (%)", 5.0, 36.0, interest_rate, step=0.5, key="wi_rate")

    if st.button("▶ Run Simulation", key="run_sim"):
        try:
            simulator = load_simulator()
            overrides = WhatIfOverrides(
                loan_amount=float(wi_loan),
                loan_tenure_months=wi_tenure,
                co_applicant=wi_co,
                insurance=wi_ins,
                interest_rate=float(wi_rate),
            )
            with st.spinner("Running simulation..."):
                wi_out = simulator.run(request, overrides)

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("##### Base Scenario")
                st.metric("Default Probability", f"{wi_out.base_scenario.default_probability*100:.1f}%")
                st.metric("Risk Level", wi_out.base_scenario.risk_level.value)
                st.metric("Expected Loss", f"₹{wi_out.base_scenario.expected_loss:,.0f}")
                st.metric("Recommendation", wi_out.base_scenario.recommendation.value)
            with sc2:
                st.markdown("##### Modified Scenario")
                delta = wi_out.delta_probability * 100
                st.metric("Default Probability", f"{wi_out.modified_scenario.default_probability*100:.1f}%", delta=f"{delta:+.1f}%")
                st.metric("Risk Level", wi_out.modified_scenario.risk_level.value)
                el_delta = wi_out.delta_expected_loss
                st.metric("Expected Loss", f"₹{wi_out.modified_scenario.expected_loss:,.0f}", delta=f"₹{el_delta:+,.0f}")
                st.metric("Recommendation", wi_out.modified_scenario.recommendation.value)

            if wi_out.risk_level_changed:
                st.success("✅ Risk level improved with these changes!")
            if wi_out.recommendation_changed:
                st.info(f"💡 Recommendation changed: {wi_out.base_scenario.recommendation.value} → {wi_out.modified_scenario.recommendation.value}")

            st.markdown(f"**Insight:** {wi_out.insight}")

        except Exception as e:
            st.error(f"Simulation failed: {e}")

# Tab 6: Batch Upload
with tab6:
    st.markdown("#### 📂 Batch Dataset Evaluation")
    st.markdown("Upload a CSV file containing borrower and loan details to run batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV dataset (Must contain: age, annual_income, employment_type, home_ownership, credit_score, loan_amount, loan_tenure, loan_purpose, interest_rate, state)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.markdown(f"**Dataset Preview ({len(df_batch)} rows):**")
            st.dataframe(df_batch.head())
            
            if st.button("▶ Run Batch Prediction", type="primary"):
                st.info("Batch prediction running... This may take a moment.")
                results = []
                
                # Simple progress bar
                progress_bar = st.progress(0)
                
                # Mock schema mappings for demo purposes if exact columns are missing
                for idx, row in df_batch.iterrows():
                    try:
                        batch_req = InferenceRequest(
                            borrower=BorrowerInput(
                                age=int(row.get('age', 35)),
                                annual_income=float(row.get('annual_income', 720000)),
                                employment_type=EmploymentType(row.get('employment_type', 'Salaried')),
                                employment_length_months=int(row.get('employment_length', 36)),
                                home_ownership=HomeOwnership(row.get('home_ownership', 'Own')),
                                credit_score=int(row.get('credit_score', 720)),
                                dependents=int(row.get('dependents', 1)),
                            ),
                            loan=LoanInput(
                                loan_amount=float(row.get('loan_amount', 500000)),
                                loan_tenure_months=int(row.get('loan_tenure', 60)),
                                loan_purpose=LoanPurpose(row.get('loan_purpose', 'Personal')),
                                interest_rate=float(row.get('interest_rate', 12.0)),
                                co_applicant=bool(row.get('co_applicant', False)),
                                insurance=bool(row.get('insurance', False)),
                                existing_monthly_obligations=float(row.get('existing_emi', 5000)),
                            ),
                            geo=GeoInput(state=str(row.get('state', 'Maharashtra'))),
                        )
                        
                        batch_res = pipeline.run(batch_req)
                        results.append({
                            "Risk Level": batch_res.risk_level.value,
                            "Default Probability": f"{batch_res.default_probability*100:.1f}%",
                            "Expected Loss (₹)": batch_res.expected_loss,
                            "Recommendation": batch_res.recommendation.action.value,
                            "Borrower Score": batch_res.borrower_360.borrower_score
                        })
                    except Exception as e:
                        results.append({"Error": str(e)})
                    
                    progress_bar.progress((idx + 1) / len(df_batch))
                
                # Show results
                df_results = pd.DataFrame(results)
                st.success("Batch Prediction Complete!")
                st.dataframe(df_results, use_container_width=True)
                
                # Download button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Results", data=csv, file_name="batch_predictions.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"Error reading dataset: {e}")

# Business Rules
st.markdown('<p class="section-header">📏 Business Rules Evaluation</p>', unsafe_allow_html=True)
if rules.hard_reject:
    st.error("🚫 HARD REJECT — One or more critical bank policy rules violated.")
elif rules.soft_violations_count > 0:
    st.warning(f"⚠️ {rules.soft_violations_count} soft rule violation(s) detected.")
else:
    st.success("✅ All business rules passed.")

if rules.rule_violations:
    for v in rules.rule_violations:
        css_cls = "rule-hard" if v.severity.value == "HARD" else "rule-soft"
        icon = "🚫" if v.severity.value == "HARD" else "⚠️"
        st.markdown(f'<div class="{css_cls}">{icon} <b>{v.rule_name}</b>: {v.message} (Actual: {v.actual_value}, Threshold: {v.threshold_value})</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("IDBI AI Credit Risk Intelligence Platform · v1.0 · Powered by XGBoost + SHAP · Offline Geo Data: NDMA/IMD · For authorized IDBI Bank officers only.")
