from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict churn risk and return a simple retention recommendation.",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    contract_month_to_month: int
    internet_service_fiber_optic: int
    payment_method_electronic_check: int

@app.get("/")
def root():
    return {
        "message": "Telecom Churn Prediction API is running",
        "endpoints": ["/predict", "/docs"]
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    score = 0.0

    if customer.tenure < 12:
        score += 0.30
    elif customer.tenure < 24:
        score += 0.15

    if customer.contract_month_to_month == 1:
        score += 0.25

    if customer.internet_service_fiber_optic == 1:
        score += 0.15

    if customer.payment_method_electronic_check == 1:
        score += 0.15

    if customer.monthly_charges > 80:
        score += 0.10

    if customer.total_charges < 1000:
        score += 0.05

    churn_probability = min(round(score, 3), 0.95)

    if churn_probability >= 0.60:
        risk_level = "high"
        action = "Prioritize retention outreach"
    elif churn_probability >= 0.35:
        risk_level = "medium"
        action = "Monitor and target with light-touch retention"
    else:
        risk_level = "low"
        action = "No immediate intervention needed"

    return {
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "recommended_action": action
    }
