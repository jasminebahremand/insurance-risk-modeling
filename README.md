# When Does a Customer Become a Liability?
**Predictive Modeling · Insurance Analytics**

## Overview
Built predictive models to estimate insurance claim likelihood and expected loss cost for new policyholders, translating model outputs into actionable pricing, underwriting, and retention strategies.

> Full write-up available at [portfolio URL]

## Methods
- Feature engineering and preprocessing
- Regression: GLM (Tweedie), MLP, XGBoost, LightGBM
- Classification: Gradient Boosting, Random Forest, Logistic Regression, MLP
- 5-fold cross-validation and model comparison
- SHAP for model interpretation and feature importance
- Customer segmentation by insurance tenure and premium tier

## Key Findings
- **LightGBM achieved R²=0.77 and MAE=14.14** on loss cost prediction — strongest across all models tested
- **Gradient Boosting achieved ROC-AUC=0.7889** on claim classification
- **Sample weighting nearly tripled recall for actual claims** (0.06 → 0.18) with only marginal ROC-AUC reduction (0.7889 → 0.7888) — a meaningfully better model for real-world use
- **Insurance tenure and net premium were the top SHAP drivers** — Emerging policyholders (3–5 years) showed the highest average loss cost and volatility

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP · Statsmodels

## Files
- `insurance_risk_modeling.py` — full modeling workflow
- `requirements.txt` — project dependencies

## Key Visual Insights

### Model Performance (R²)
![R2 Comparison](plots/model_performance_r2_comparison.jpg)
LightGBM outperformed XGBoost across both targets; XGBoost showed poor fit for HALC (negative R²).

### Classification Model Performance (ROC-AUC)
![ROC AUC](plots/classification_model_roc_auc_comparison.jpg)
Gradient Boosting achieved the highest ROC-AUC, indicating strongest performance in distinguishing claim vs. no-claim cases.

### Loss Cost by Customer Tenure
![Tenure](plots/loss_cost_by_tenure_segment.jpg)
Loss cost peaks among Emerging customers (3–5 years), indicating elevated risk early in the policy lifecycle.

### Loss Cost by Premium Tier
![Premium](plots/loss_cost_by_premium_tier.jpg)
Lower premium tiers exhibit higher average loss cost, suggesting potential underpricing risk.


## How to Run
```bash
pip install -r requirements.txt
python insurance_risk_modeling.py
```

> Dataset provided as part of USC Marshall coursework and is not available publicly.
