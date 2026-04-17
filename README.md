# When Does a Customer Become a Liability?
**Predictive Modeling · Insurance Analytics**

## Overview
Built predictive models to estimate insurance claim likelihood and expected loss cost for new policyholders, translating model outputs into actionable pricing, underwriting, and retention strategies.

> Full write-up coming soon

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
- **Insurance tenure and net premium were the top SHAP drivers** — Emerging policyholders (1–2 years) showed the highest average loss cost and volatility

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP · Statsmodels

## Files
- `insurance_risk_modeling.py` — full modeling workflow
- `requirements.txt` — project dependencies

## Key Visual Insights

### Model Performance (R²)
![R2 Comparison](plots/model_performance_r2_comparison.jpg)
LightGBM achieved R²=0.77 on loss cost prediction, outperforming XGBoost across both targets. XGBoost showed severely poor fit for HALC (R²=−2.00), making LightGBM the clear choice for deployment.

### Classification Model Performance (ROC-AUC)
![ROC AUC](plots/classification_model_roc_auc_comparison.jpg)
Gradient Boosting achieved the highest ROC-AUC (0.7889) across all seven classification models tested, with a spread of ~0.04 between best and worst — indicating model choice meaningfully impacts claim detection performance.

### Loss Cost by Customer Tenure
![Tenure](plots/loss_cost_by_tenure_segment.jpg)
New policyholders (0 years) carry the highest average loss cost (345.80), with Emerging (1–2 years) close behind at 327.77. Loss cost decreases as tenure increases, suggesting long-term customers represent lower underwriting risk.

### Loss Cost by Premium Tier
![Premium](plots/loss_cost_by_premium_tier.jpg)
Low-tier policyholders average $328.20 in loss cost vs $316.18 for high-tier — a gap that suggests low-premium customers may be systematically underpriced relative to their actual risk.

## How to Run
```bash
pip install -r requirements.txt
python insurance_risk_modeling.py
```

> Dataset provided as part of USC Marshall coursework and is not available publicly.
