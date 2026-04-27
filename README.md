# When Does a Customer Become a Liability?
**Predictive Modeling · Insurance Analytics · LightGBM · SHAP**

---

## Overview
Built predictive models to estimate insurance claim likelihood and expected loss cost for new policyholders, translating model outputs into actionable pricing, underwriting, and retention strategies.

---

## Key Findings
- **LightGBM achieved R²=0.77 and MAE=14.14** on loss cost prediction — strongest across all models tested
- **Gradient Boosting achieved ROC-AUC=0.7889** on claim classification — highest across seven models
- **Sample weighting nearly tripled recall for actual claims** (0.06 → 0.18) with only marginal ROC-AUC reduction — a meaningfully better model for real-world use
- **Insurance tenure and net premium were the top SHAP drivers** — Emerging policyholders (3–5 years) showed the highest average loss cost and volatility; Loyal customers (11+ years) showed the lowest

---

## Key Visuals

### Model Performance Comparison (R²)
![R2 Comparison](plots/model_performance_r2_comparison.jpg)

LightGBM achieved R²=0.77 on loss cost prediction, outperforming all other models across both LC and HALC targets — making it the clear choice for deployment.

### Classification Model Performance (ROC-AUC)
![ROC AUC](plots/classification_model_roc_auc_comparison.jpg)

Gradient Boosting achieved the highest ROC-AUC (0.7889) across all seven classification models tested — model choice meaningfully impacts claim detection performance.

### Loss Cost by Customer Tenure
![Tenure](plots/loss_cost_by_tenure_segment.jpg)

Emerging policyholders (3–5 years) carry the highest average loss cost and greatest volatility. Loss cost decreases as tenure increases — long-term customers represent lower and more predictable underwriting risk.

| Segment | Mean LC |
|---------|---------|
| New (0–2 yrs) | 598 |
| Emerging (3–5 yrs) | 713 |
| Established (6–10 yrs) | 664 |
| Loyal (11+ yrs) | 386 |

### Loss Cost by Premium Tier
![Premium](plots/loss_cost_by_premium_tier.jpg)

Low-tier policyholders show comparable average loss costs to high-tier — suggesting low-premium customers may be systematically underpriced relative to their actual risk.

---

## Methods
- Feature engineering from raw date fields (age, vehicle age, driving experience, policy duration)
- Regression: GLM (Tweedie), Random Forest, Gradient Boosting, XGBoost, LightGBM, Neural Network
- Classification: Gradient Boosting, Random Forest, Logistic Regression (Lasso/Ridge), MLP
- 5-fold cross-validation and hyperparameter tuning via grid search
- SHAP for model interpretation and feature importance
- Customer segmentation by insurance tenure and premium tier

---

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP · Statsmodels · Matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook insurance_risk_modeling.ipynb
```

---

## Data
Dataset provided as part of USC Marshall coursework and is not publicly available.

Key features include policy dates, vehicle registration year, net premium, insurance tenure, and demographic fields. Engineered features: age, vehicle age, driving experience, policy duration, and time since last renewal.

---

## Files
- `insurance_risk_modeling.ipynb` — full modeling notebook
- `requirements.txt` — dependencies
- `plots/` — generated visualizations
