"""
Generate a realistic synthetic loan dataset with proper statistical relationships
between features and the default target variable.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 15000

# ── Core borrower attributes ──────────────────────────────────────────────────
age = np.random.normal(38, 12, N).clip(21, 70).astype(int)
gender = np.random.choice(["Male", "Female", "Other"], N, p=[0.60, 0.38, 0.02])
education = np.random.choice(["HighSchool", "Graduate", "PostGraduate", "Other"], N, p=[0.25, 0.45, 0.25, 0.05])
marital_status = np.random.choice(["Married", "Single", "Divorced"], N, p=[0.60, 0.30, 0.10])
employment_type = np.random.choice(["Salaried", "SelfEmployed", "Business", "Farmer", "Unemployed"], N, p=[0.45, 0.20, 0.15, 0.12, 0.08])

# Employment stability: Unemployed and Farmer get less stable
emp_stability_base = {
    "Salaried": 0.75, "Business": 0.65, "SelfEmployed": 0.55,
    "Farmer": 0.45, "Unemployed": 0.20
}
emp_base = np.array([emp_stability_base[e] for e in employment_type])
employment_length_months = (np.random.exponential(48, N) * emp_base).clip(0, 360).astype(int)

# Annual income — strongly correlated with education and employment type
income_base = {
    "HighSchool": 30000, "Graduate": 60000, "PostGraduate": 100000, "Other": 35000
}
emp_income_mult = {
    "Salaried": 1.0, "Business": 1.4, "SelfEmployed": 1.2, "Farmer": 0.7, "Unemployed": 0.4
}
annual_income = np.array([
    max(12000, np.random.lognormal(
        np.log(income_base[education[i]] * emp_income_mult[employment_type[i]]), 0.5
    ))
    for i in range(N)
]).round(-2)

monthly_income = (annual_income / 12).round(0)

# Credit score — depends on income and employment
credit_score_base = 600 + (annual_income / 20000).clip(0, 150) + np.random.normal(0, 80, N)
# Employed/stable people get higher credit scores
credit_score_base += np.where(employment_type == "Salaried", 30, 0)
credit_score_base += np.where(employment_type == "Unemployed", -80, 0)
credit_score = credit_score_base.clip(300, 900).round(0).astype(int)

credit_history_length_months = (employment_length_months * 1.5 + np.random.normal(0, 24, N)).clip(0, 360).astype(int)

# Delinquency — negatively correlated with credit score
delinquency_prob = 0.8 - (credit_score - 300) / 750
delinquency_count = np.random.binomial(5, np.clip(delinquency_prob * 0.15, 0, 0.5), N)

# Past default — strong predictor
past_default_prob = np.where(credit_score < 550, 0.40, np.where(credit_score < 650, 0.20, 0.05))
past_default_flag = np.random.binomial(1, past_default_prob, N)

# ── Loan attributes ───────────────────────────────────────────────────────────
loan_amount = np.random.lognormal(np.log(annual_income * 5), 0.6, N).clip(50000, 10000000).round(-3)
loan_term_months = np.random.choice([12, 18, 24, 36, 48, 60, 72, 84, 120], N, p=[0.05, 0.05, 0.10, 0.20, 0.20, 0.15, 0.10, 0.08, 0.07])
loan_purpose = np.random.choice(["Home", "Personal", "Vehicle", "Education", "Business", "Agri"], N, p=[0.25, 0.25, 0.15, 0.15, 0.12, 0.08])
loan_product = np.where(np.isin(loan_purpose, ["Home", "Vehicle", "Agri"]), "Secured", "Unsecured")

# Interest rate — higher for risky borrowers
base_rate = 10.0
interest_rate = (base_rate + 
                 (800 - credit_score) / 50 + 
                 np.random.normal(0, 2, N) +
                 np.where(employment_type == "Unemployed", 4, 0) +
                 np.where(loan_product == "Unsecured", 2, 0)
                ).clip(6.0, 36.0).round(2)

# EMI calculation: P * r * (1+r)^n / ((1+r)^n - 1)
monthly_rate = interest_rate / 12 / 100
n = loan_term_months
emi_amount = (loan_amount * monthly_rate * (1 + monthly_rate)**n / ((1 + monthly_rate)**n - 1)).round(0)

existing_emi = np.random.exponential(monthly_income * 0.1, N).clip(0, monthly_income * 0.5).round(0)
debt_to_income = ((emi_amount + existing_emi) / monthly_income).clip(0.01, 2.0).round(3)

# ── Geographic attributes ──────────────────────────────────────────────────────
states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "Gujarat",
          "Rajasthan", "Bihar", "Madhya Pradesh", "Andhra Pradesh", "Kerala",
          "West Bengal", "Telangana"]
state_weights = [0.13, 0.09, 0.09, 0.12, 0.08, 0.07, 0.08, 0.07, 0.06, 0.05, 0.09, 0.07]
state = np.random.choice(states, N, p=state_weights)

# State risk scores — calibrated from real economic data
state_risk_map = {
    "Maharashtra": 0.42, "Karnataka": 0.38, "Tamil Nadu": 0.40, "Uttar Pradesh": 0.65,
    "Gujarat": 0.35, "Rajasthan": 0.55, "Bihar": 0.72, "Madhya Pradesh": 0.58,
    "Andhra Pradesh": 0.48, "Kerala": 0.30, "West Bengal": 0.52, "Telangana": 0.44,
}
state_risk_score = np.array([state_risk_map.get(s, 0.50) for s in state]) + np.random.normal(0, 0.05, N)
state_risk_score = state_risk_score.clip(0.1, 0.95).round(3)

district_risk_score = (state_risk_score + np.random.normal(0, 0.08, N)).clip(0.05, 0.95).round(3)
urban_rural_flag = np.random.choice(["Urban", "SemiUrban", "Rural"], N, p=[0.45, 0.30, 0.25])
branch_region_map = {
    "Maharashtra": "WestZone", "Gujarat": "WestZone", "Rajasthan": "WestZone",
    "Karnataka": "SouthZone", "Tamil Nadu": "SouthZone", "Kerala": "SouthZone",
    "Andhra Pradesh": "SouthZone", "Telangana": "SouthZone",
    "Uttar Pradesh": "NorthZone", "West Bengal": "EastZone",
    "Bihar": "EastZone", "Madhya Pradesh": "CentralZone",
}
branch_region = np.array([branch_region_map.get(s, "CentralZone") for s in state])
area_default_rate = (5.0 + state_risk_score * 25 + np.random.normal(0, 2, N)).clip(0.5, 35.0).round(1)
population_density_band = np.where(np.isin(urban_rural_flag, ["Urban"]), 
                                    np.random.choice(["High", "Medium"], N, p=[0.6, 0.4]),
                                    np.random.choice(["Low", "Medium"], N, p=[0.6, 0.4]))
economic_activity_type = np.random.choice(["Salaried", "MSME", "Agri", "Mixed"], N, p=[0.40, 0.25, 0.20, 0.15])
channel = np.random.choice(["Branch", "Digital", "Field", "Partner"], N, p=[0.40, 0.30, 0.20, 0.10])
loan_source_type = np.random.choice(["New", "Repeat", "TopUp"], N, p=[0.60, 0.30, 0.10])
service_area_cluster = np.random.randint(0, 5, N)
distance_to_branch_km = np.random.exponential(15, N).clip(0, 80).round(1)

# ── Additional flags ─────────────────────────────────────────────────────────
has_mortgage = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
has_dependents = np.random.choice(["Yes", "No"], N, p=[0.55, 0.45])
has_cosigner = np.random.choice(["Yes", "No"], N, p=[0.25, 0.75])

# ── Default probability (core target generation) ──────────────────────────────
# The probability is a function of ACTUAL risk factors with realistic weights
logit = (
    -3.0                                                              # base intercept
    + 1.8  * (credit_score < 550).astype(float)                      # very low credit
    + 1.0  * ((credit_score >= 550) & (credit_score < 650)).astype(float)  # low credit
    - 0.8  * ((credit_score >= 750) & (credit_score < 850)).astype(float)  # good credit
    - 1.5  * (credit_score >= 850).astype(float)                     # excellent credit
    + 2.5  * past_default_flag                                        # STRONGEST predictor
    + 1.5  * (debt_to_income > 0.70).astype(float)                   # high DTI
    + 1.0  * ((debt_to_income > 0.50) & (debt_to_income <= 0.70)).astype(float)
    - 0.8  * (debt_to_income < 0.30).astype(float)                   # healthy DTI
    + 0.8  * (loan_to_income := (loan_amount / annual_income.clip(1)) > 10).astype(float)
    + 0.6  * (employment_type == "Unemployed").astype(float)
    - 0.5  * (employment_type == "Salaried").astype(float)
    + 0.4  * (employment_length_months < 6).astype(float)
    - 0.4  * (employment_length_months > 60).astype(float)
    + 0.6  * (delinquency_count > 2).astype(float)
    + 0.4  * (delinquency_count > 0).astype(float)
    + 0.5  * state_risk_score                                         # high-risk states
    + 0.3  * (interest_rate > 20).astype(float)                      # very high rates = risky
    + np.random.normal(0, 0.5, N)                                    # noise
)

default_prob = 1 / (1 + np.exp(-logit))
loan_default = np.random.binomial(1, default_prob, N)

print(f"Default rate: {loan_default.mean():.2%}")
print(f"Credit score corr: {np.corrcoef(credit_score, loan_default)[0,1]:.3f}")
print(f"DTI corr: {np.corrcoef(debt_to_income, loan_default)[0,1]:.3f}")
print(f"Past default corr: {np.corrcoef(past_default_flag, loan_default)[0,1]:.3f}")

# ── Assemble DataFrame ────────────────────────────────────────────────────────
loan_id = [f"LN{100000+i}" for i in range(N)]
customer_id = [f"C{200000+i}" for i in range(N)]
district = ["Unknown"] * N
city = ["Unknown"] * N
pincode = np.random.randint(100000, 999999, N)
branch_code = ["AUTO"] * N
disbursal_date = ["2024-01-01"] * N
repayment_status = ["Active"] * N
first_emi_default_flag = np.random.binomial(1, default_prob * 0.3, N)

df = pd.DataFrame({
    "loan_id": loan_id,
    "customer_id": customer_id,
    "age": age,
    "gender": gender,
    "education": education,
    "marital_status": marital_status,
    "employment_type": employment_type,
    "employment_length_months": employment_length_months,
    "annual_income": annual_income.astype(int),
    "monthly_income": monthly_income.astype(int),
    "existing_emi": existing_emi.astype(int),
    "debt_to_income": debt_to_income,
    "credit_score": credit_score,
    "credit_history_length_months": credit_history_length_months,
    "delinquency_count": delinquency_count,
    "past_default_flag": past_default_flag,
    "loan_amount": loan_amount.astype(int),
    "loan_term_months": loan_term_months,
    "interest_rate": interest_rate,
    "emi_amount": emi_amount.astype(int),
    "loan_purpose": loan_purpose,
    "loan_product": loan_product,
    "disbursal_date": disbursal_date,
    "repayment_status": repayment_status,
    "first_emi_default_flag": first_emi_default_flag,
    "loan_source_type": loan_source_type,
    "state": state,
    "district": district,
    "city": city,
    "pincode": pincode,
    "urban_rural_flag": urban_rural_flag,
    "branch_code": branch_code,
    "branch_region": branch_region,
    "service_area_cluster": service_area_cluster,
    "distance_to_branch_km": distance_to_branch_km,
    "area_default_rate": area_default_rate,
    "district_risk_score": district_risk_score,
    "state_risk_score": state_risk_score,
    "population_density_band": population_density_band,
    "economic_activity_type": economic_activity_type,
    "channel": channel,
    "has_mortgage": has_mortgage,
    "has_dependents": has_dependents,
    "has_cosigner": has_cosigner,
    "loan_default": loan_default,
})

out_path = Path("ai/data/raw/loan_data.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path}")
print(f"Columns: {list(df.columns)}")
