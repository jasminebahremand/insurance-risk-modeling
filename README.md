# Insurance Risk Modeling
Predictive Modeling · Insurance Analytics

## Overview
Built predictive models to estimate insurance claim likelihood and loss cost, translating model outputs into pricing and risk management insights.

## Methods
- Feature engineering and preprocessing
- Regression (GLM, XGBoost, LightGBM)
- Classification (Boosting, Random Forest, Logistic Regression)
- Cross-validation and model comparison
- SHAP for model interpretation

## Key Findings
- LightGBM achieved the strongest performance for loss prediction (R² ≈ 0.77)
- Boosting achieved the highest ROC-AUC for claim classification (~0.79)
- Insurance tenure and net premium were key drivers of risk
- Higher-risk segments showed greater loss variability

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · LightGBM · SHAP

## Files
- insurance_risk_modeling.py — modeling workflow
- requirements.txt — project dependencies

## Plots
- loss_cost_distribution.png
- model_performance_comparison.png
- roc_curve.png
- shap_feature_importance.png
- segment_risk_comparison.png

## How to Run
pip install -r requirements.txt  
python insurance_risk_modeling.py
