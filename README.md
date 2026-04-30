# Which Policyholders Will Cost the Most?
**Predictive Modeling · Insurance Analytics · LightGBM · XGBoost · SHAP**

---

## Overview

Insurance companies price policies at onboarding without a reliable way to predict which customers will generate the highest claims. This project built and compared models to predict claim likelihood and expected loss cost for new policyholders — translating outputs into pricing, underwriting, and retention strategy.

> Full write-up: https://jasminebahremand.my.canva.site/

---

## Key Findings
- **LightGBM achieved the lowest RMSE (564.08)** on loss cost prediction — strongest across all regression models tested
- **XGBoost achieved ROC-AUC of 0.7921** on claim classification — correctly identifying high-risk policyholders 79% of the time
- **Emerging policyholders (3–5 years) carried the highest average loss cost ($713)** — nearly double that of Loyal customers (11+ years, $408) — risk does not grow linearly with tenure
- **Low-premium customers show disproportionately high claims costs** relative to what they paid, suggesting systematic underpricing at the low end of the portfolio

---

## Key Visuals

### Model Performance Comparison (RMSE)
![RMSE Comparison](plots/model_comparison_halc.png)
LightGBM achieved the lowest HALC RMSE (564.08) across all regression models tested — outperforming Gradient Boosting, GLM, Neural Network, Random Forest, and XGBoost.

### Classification Model Performance (ROC-AUC)
![ROC AUC](plots/roc_curve_comparison.png)
XGBoost achieved the highest ROC-AUC (0.7921) across all classification models — correctly identifying high-risk policyholders 79% of the time.

### Loss Cost by Customer Tenure
![Tenure](plots/segment_avg_halc.png)
Emerging policyholders (3–5 years) carry the highest average loss cost. Loyal customers (11+ years) show the lowest and most predictable risk.

| Segment | Mean Loss Cost |
|---------|---------------|
| New (0–2 yrs) | $598 |
| Emerging (3–5 yrs) | $713 |
| Established (6–10 yrs) | $664 |
| Loyal (11+ yrs) | $408 |

---

## Methods

- Feature engineering from raw date fields (age, vehicle age, driving experience, policy duration)
- Regression: GLM (Tweedie), Random Forest, Gradient Boosting, XGBoost, LightGBM, Neural Network
- Classification: XGBoost, LightGBM, Random Forest, Logistic Regression (L1), Neural Network
- 5-fold cross-validation for model selection
- SHAP for model interpretation and feature importance
- Customer segmentation by insurance tenure and premium tier

---

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP · TensorFlow · Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook insurance_risk_modeling.ipynb
```

Upload `insurance_train.csv` to `/content/` before running.

---

## Data

Dataset provided as part of USC Marshall coursework and is not publicly available. Key features include policy dates, vehicle registration year, net premium, insurance tenure, and demographic fields. Engineered features: age, vehicle age, driving experience, policy duration, and time since last renewal.

---

## Files

- `insurance_risk_modeling.ipynb` — full modeling notebook
- `requirements.txt` — dependencies
- `plots/` — generated visualizations
