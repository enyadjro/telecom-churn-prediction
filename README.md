# Cost-Sensitive Customer Churn Prediction for a Telecom Provider

This project develops an end-to-end, interpretable machine learning pipeline to predict customer churn for a telecommunications provider.  
Rather than optimizing raw classification accuracy, the emphasis is on **cost-sensitive decision-making**, combining probability calibration, class imbalance handling, and business-aware threshold optimization.

---

## Problem Overview

Customer churn is a major source of revenue loss in subscription-based businesses.  
The objective is to estimate **each customer’s probability of churn** and use these probabilities to support **data-driven retention strategies**.

Instead of relying on a fixed 0.5 classification cutoff, the model selects an operating threshold that **minimizes expected business cost**, explicitly balancing:
- missed churners (lost revenue), and
- unnecessary retention outreach (intervention cost).

---

## Key Features

- End-to-end ML workflow: EDA → feature engineering → modeling → evaluation
- Class imbalance handled using **SMOTE** (training data only)
- **Calibrated churn probabilities** using isotonic regression
- Business-cost–aware threshold optimization
- Strong emphasis on **interpretability and explainability**

---

## Repository Structure

```text
data/        Raw and processed datasets
notebooks/   Exploratory analysis and experiments
src/         Reusable modeling and evaluation code
figures/     Generated plots and diagnostics
models/      Trained models and calibration artifacts
