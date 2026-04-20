# When Does a Customer Become a Liability?
**Predictive Modeling · Insurance Analytics · LightGBM · SHAP**

---

## Overview
Built predictive models to estimate insurance claim likelihood and expected loss cost for new policyholders, translating model outputs into actionable pricing, underwriting, and retention strategies.

> Full write-up available at [portfolio URL]

---

## Key Findings
- **LightGBM achieved R²=0.77 and MAE=14.14** on loss cost prediction — strongest across all models tested
- **Gradient Boosting achieved ROC-AUC=0.7889** on claim classification — highest across seven models
- **Sample weighting nearly tripled recall for actual claims** (0.06 → 0.18) with only marginal ROC-AUC reduction — a meaningfully better model for real-world use
- **Insurance tenure and net premium were the top SHAP drivers** — Emerging policyholders (3–5 years) showed the highest average loss cost and volatility; Loyal customers (11+ years) showed the lowest

---

## Key Visuals

### Model Performance Comparison (R²)
![R2 Comparison](plots/model_performance_r2_comparison.png)

LightGBM achieved R²=0.77 on loss cost prediction, outperforming XGBoost across both targets. XGBoost showed severely poor fit for HALC (R²=−2.00), making LightGBM the clear choice for deployment.

### Classification Model Performance (ROC-AUC)
![ROC AUC](plots/classification_model_roc_auc_comparison.png)

Gradient Boosting achieved the highest ROC-AUC (0.7889) across all seven classification models tested — model choice meaningfully impacts claim detection performance.

### Loss Cost by Customer Tenure
![Tenure](plots/loss_cost_by_tenure_segment.png)

Emerging policyholders (3–5 years) carry the highest average loss cost and greatest volatility. Loss cost decreases as tenure increases — long-term customers represent lower and more predictable underwriting risk.

### Loss Cost by Premium Tier
![Premium](plots/loss_cost_by_premium_tier.png)

Low-tier policyholders average higher loss costs than high-tier — suggesting low-premium customers may be systematically underpriced relative to their actual risk.

---

## Methods
- Feature engineering and preprocessing
- Regression: GLM (Tweedie), MLP, XGBoost, LightGBM
- Classification: Gradient Boosting, Random Forest, Logistic Regression, MLP
- 5-fold cross-validation and model comparison
- SHAP for model interpretation and feature importance
- Customer segmentation by insurance tenure and premium tier

---

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP · Statsmodels

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook insurance_risk_modeling.ipynb
```

---

## Data
Dataset provided as part of USC Marshall coursework and is not publicly available.

---

## Files
- `insurance_risk_modeling.ipynb` — full modeling notebook
- `requirements.txt` — dependencies
- `plots/` — generated visualizations
