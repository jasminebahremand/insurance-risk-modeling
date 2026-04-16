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

## Plots
- `loss_cost_distribution.png`
- `model_performance_comparison.png`
- `roc_curve.png`
- `shap_feature_importance.png`
- `segment_risk_comparison.png`

## How to Run
```bash
pip install -r requirements.txt
python insurance_risk_modeling.py
```

> Dataset provided as part of USC Marshall coursework and is not available publicly.
