# Insurance Risk Modeling

## Overview
Built predictive models to estimate insurance claim likelihood and loss cost for smarter auto insurance pricing. This project focused on three outputs: claim status (CS), loss cost per exposure unit (LC), and historically adjusted loss cost (HALC). 

## Business Problem
Insurers need to price policies accurately enough to avoid underpricing high-risk customers and losing lower-risk ones. This project addressed that problem by predicting expected losses and claim likelihood from policyholder, contract, and vehicle data. 

## Targets
- **LC** = `X.15 / X.16`
- **HALC** = `(X.15 / X.16) * X.18`
- **CS** = whether a policyholder makes a claim (`1`) or not (`0`) 

## Project Workflow
- Engineered temporal features such as age, license years, and contract start date components
- Applied label encoding for categorical variables and median imputation for missing values
- Tested MLP, GLM (Tweedie), XGBoost, and LightGBM for loss prediction
- Tested Boosting, weighted Boosting, Random Forest, Logistic Regression, and MLP for claim classification
- Used 5-fold cross-validation for model evaluation
- Applied SHAP and segment analysis to interpret key drivers of risk and support pricing strategy 

## Model Strategy
For regression, the project compared multiple models and selected **LightGBM** as the final approach for predicting LC and HALC based on overall performance. For classification, **Boosting** and **weighted Boosting** performed best, with weighting used to better handle class imbalance. 

## Key Findings
- LightGBM achieved the strongest overall performance for loss prediction and was selected as the final regression model
- Boosting achieved the highest ROC-AUC for claim classification, with weighted Boosting improving classification behavior under imbalance
- Insurance tenure (`X.8`) and net premium (`X.14`) emerged as important risk drivers
- Newer and lower-premium customers showed higher-risk patterns, while loyal customers appeared more retention-worthy 

## Tools
Python, Pandas, Scikit-learn, XGBoost, LightGBM, SHAP

## Files
- `insurance_risk_modeling.py` — reconstructed modeling workflow
- `insurance_risk_modeling_report.pdf` — technical report
- `insurance_risk_modeling_presentation.pdf` — presentation deck

## Note
This repository contains a reconstructed version of the original project workflow for portfolio purposes. It is designed to closely reflect the methods, modeling choices, and interpretation described in the report and presentation, though exact implementation details may differ from the original team submission. 
