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
telecom-churn-prediction/
├── **data/**        # Raw and processed datasets
├── **notebooks/**   # Exploratory analysis and experiments
├── **src/**         # Reusable modeling and evaluation code
├── **figures/**     # Generated plots and diagnostics
├── **models/**      # Trained models and calibration artifacts
└── README.md
```

## Modeling Approach (Interpretable + Cost-Sensitive)

We evaluated multiple classification models (e.g., linear and tree-based) during experimentation.  
The final pipeline uses **Logistic Regression** as the primary model because it provides **well-behaved probabilities** and strong **interpretability**, which aligns with the project goal: **cost-sensitive decision-making** (not just accuracy).

**Pipeline highlights**
- **Imbalance handling:** SMOTE applied **only to training folds** (prevents leakage)
- **Probability calibration:** Isotonic regression on a calibration split
- **Decision rule:** Business-aware **threshold optimization** to minimize expected cost (vs. default 0.5 cutoff)
- **Explainability:** Permutation importance + partial dependence to interpret drivers of churn risk

---

## Results & Diagnostics

### Discrimination performance
ROC and Precision–Recall curves summarize ranking performance on the test set.

![ROC Curve](figures/roc_curve_test.png)

![Precision-Recall Curve](figures/pr_curve_test.png)

### Cost-sensitive thresholding (decision-focused)
Instead of a fixed 0.5 threshold, we select an operating point that minimizes expected cost by balancing:
- **False negatives** (missed churners → lost revenue)
- **False positives** (unnecessary retention offers → intervention cost)

![Cost Curve / Threshold](figures/cost_curve_threshold.png)

![Confusion Matrix at Optimal Threshold](figures/confusion_matrix_optthr.png)

### Interpretability (what drives churn risk)
Top drivers are consistent with telecom churn behavior: tenure, contract type, and internet service features.

![Top Feature Importance](figures/feature_importance.png)

![Partial Dependence (Top Features)](figures/partial_dependence_top_features.png)

