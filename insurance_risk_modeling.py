"""
insurance_risk_model.py

Reconstructed workflow closely aligned to the original insurance project report.

Key features of this reconstruction:
- Target creation:
    LC   = X.15 / X.16
    HALC = (X.15 / X.16) * X.18
    CS   = 1 if claim occurs, else 0
- Label encoding for categorical variables
- Median imputation for missing values
- Temporal feature engineering
- 5-fold cross-validation
- Model-specific feature selection
- Regression models: MLP, GLM(Tweedie), XGBoost, LightGBM
- Classification models: Boosting, Boosting(weighted), Random Forest,
  Logistic Regression, Logistic Lasso, Logistic Ridge, MLP
- SHAP interpretation for final LightGBM model
- Segment analysis by tenure and premium

This is a portfolio-quality reconstruction based on the project report/presentation.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import TweedieRegressor, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

TRAIN_PATH = Path("insurance_train.csv")
TEST_PATH = Path("insurance_test.csv")
PREDICTION_OUTPUT = Path("insurance_risk_predictions.csv")

RANDOM_STATE = 42
N_SPLITS = 5

DATE_COLS = ["X.2", "X.3", "X.4", "X.5", "X.6"]
CATEGORICAL_COLS = ["X.7", "X.13", "X.19", "X.20", "X.21", "X.27"]

# Project brief variable meanings used in comments
# X.8  = years with insurer
# X.14 = net premium
# X.22 = vehicle registration year


# =========================================================
# HELPERS
# =========================================================

def check_files() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    return df_train, df_test


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build targets from raw insurance fields.

    LC   = X.15 / X.16
    HALC = LC * X.18
    CS   = claim indicator
    """
    df = df.copy()

    x15 = pd.to_numeric(df["X.15"], errors="coerce")
    x16 = pd.to_numeric(df["X.16"], errors="coerce")
    x18 = pd.to_numeric(df["X.18"], errors="coerce")

    df["CS"] = (x16 > 0).astype(int)
    df["LC"] = 0.0
    df["HALC"] = 0.0

    mask = x16 > 0
    df.loc[mask, "LC"] = (x15[mask] / x16[mask]).fillna(0.0)
    df.loc[mask, "HALC"] = (df.loc[mask, "LC"] * x18[mask]).fillna(0.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal feature engineering aligned to the report:
    - age
    - license years
    - contract date parts
    - a few sensible derived features from the same date fields
    """
    df = df.copy()

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    reference_year = 2020

    # X.5 = date of birth
    if "X.5" in df.columns:
        df["age"] = reference_year - df["X.5"].dt.year

    # X.6 = driver license issuance date
    if "X.6" in df.columns:
        df["license_years"] = reference_year - df["X.6"].dt.year

    # X.22 = vehicle registration year
    if "X.22" in df.columns:
        df["vehicle_age"] = reference_year - pd.to_numeric(df["X.22"], errors="coerce")

    # X.2 = contract start date
    if "X.2" in df.columns:
        df["contract_start_year"] = df["X.2"].dt.year
        df["contract_start_month"] = df["X.2"].dt.month
        df["contract_start_weekday"] = df["X.2"].dt.weekday

    # Useful derived durations
    if {"X.2", "X.4"}.issubset(df.columns):
        df["policy_duration_days"] = (df["X.4"] - df["X.2"]).dt.days

    if {"X.2", "X.3"}.issubset(df.columns):
        df["days_since_last_renewal"] = (df["X.3"] - df["X.2"]).dt.days

    # Drop raw date fields after extracting information
    df = df.drop(columns=[c for c in DATE_COLS if c in df.columns], errors="ignore")

    return df


def preprocess_like_project(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fit_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder, SimpleImputer]:
    """
    Project-consistent preprocessing:
    - label encoding for categorical variables
    - median imputation for missing values

    Returns processed train/test with same columns.
    """
    X_train = df_train[fit_cols].copy()
    X_test = df_test[fit_cols].copy()

    # Label encoding via OrdinalEncoder on known categorical columns present in data
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    non_cat_cols = [c for c in X_train.columns if c not in cat_cols]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if cat_cols:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))

    # Median imputation across all columns
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # Preserve numeric type
    X_train_imputed = X_train_imputed.apply(pd.to_numeric, errors="coerce")
    X_test_imputed = X_test_imputed.apply(pd.to_numeric, errors="coerce")

    return X_train_imputed, X_test_imputed, encoder, imputer


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def evaluate_regression_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    use_log_target: bool = True,
    n_splits: int = N_SPLITS,
) -> Dict[str, float]:
    """
    5-fold CV regression evaluation.
    If use_log_target=True, fit on log1p(y) and convert back via expm1.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_all = []
    y_pred_all = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if use_log_target:
            model.fit(X_train, np.log1p(y_train))
            pred = np.expm1(model.predict(X_valid))
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)

        pred = np.clip(pred, 0, None)

        y_true_all.extend(y_valid.tolist())
        y_pred_all.extend(pred.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    return {
        "rmse": rmse(y_true_all, y_pred_all),
        "mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "r2": float(r2_score(y_true_all, y_pred_all)),
    }


def evaluate_classification_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    5-fold CV classification evaluation using ROC-AUC.
    Also returns confusion matrix / report on pooled out-of-fold predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_all = []
    y_proba_all = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if sample_weight is not None:
            fold_weights = sample_weight[train_idx]
            model.fit(X_train, y_train, sample_weight=fold_weights)
        else:
            model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_valid)[:, 1]
        else:
            raise ValueError(f"Model {type(model).__name__} does not support predict_proba.")

        y_true_all.extend(y_valid.tolist())
        y_proba_all.extend(proba.tolist())

    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    y_pred_all = (y_proba_all >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true_all, y_proba_all)),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
        "classification_report": classification_report(y_true_all, y_pred_all, digits=4),
    }


def get_permutation_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 8,
) -> List[str]:
    """
    Match report logic:
    GLM used top 8 features from permutation importance.
    """
    base_model = TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)
    base_model.fit(X, y)

    perm = permutation_importance(
        base_model,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    return importance_df.head(n_top)["feature"].tolist()


def get_tree_top_features(
    model,
    X: pd.DataFrame,
    y_log: pd.Series,
    n_top: int = 15,
) -> List[str]:
    """
    Match report logic:
    XGBoost and LightGBM used top-ranked features from internal feature importance.
    """
    model.fit(X, y_log)

    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} has no feature_importances_")

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance_df.head(n_top)["feature"].tolist()


def create_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment analysis aligned to report:
    - insurance tenure (X.8)
    - premium (X.14)
    """
    df = df.copy()

    if "X.8" in df.columns:
        df["tenure_segment"] = pd.cut(
            pd.to_numeric(df["X.8"], errors="coerce"),
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=["New (0-2)", "Emerging (3-5)", "Established (6-10)", "Loyal (11+)"],
        )

    if "X.14" in df.columns:
        df["premium_segment"] = pd.cut(
            pd.to_numeric(df["X.14"], errors="coerce"),
            bins=[-np.inf, 300, 800, 2000, np.inf],
            labels=["Low (0-300)", "Medium (300-800)", "High (800-2000)", "Very High (2000+)"],
        )

    return df


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    check_files()

    df_train_raw, df_test_raw = load_data()
    print("Loaded data.")
    print("Train shape:", df_train_raw.shape)
    print("Test shape:", df_test_raw.shape)

    # Build targets on train
    df_train = create_targets(df_train_raw)

    # Feature engineering
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test_raw)

    # Drop raw target-generating columns and IDs from predictors
    cols_to_drop = ["X.1", "X.15", "X.16", "X.17", "X.18"]
    df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors="ignore")
    df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], errors="ignore")

    # Targets
    y_lc = df_train["LC"].copy()
    y_halc = df_train["HALC"].copy()
    y_cs = df_train["CS"].copy()

    # Features
    feature_cols = [c for c in df_train.columns if c not in ["LC", "HALC", "CS"]]
    X_raw = df_train[feature_cols].copy()
    X_test_raw = df_test[feature_cols].copy()

    # Project-like preprocessing
    X, X_test, encoder, imputer = preprocess_like_project(X_raw, X_test_raw, feature_cols)

    # =========================
    # TASK 1: REGRESSION
    # =========================

    print("\n" + "=" * 60)
    print("TASK 1: LOSS COST PREDICTION")
    print("=" * 60)

    # MLP uses all features
    mlp_lc = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE)
    mlp_halc = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE)

    # GLM uses top 8 permutation importance features
    glm_top8_lc = get_permutation_top_features(X, y_lc, n_top=8)
    glm_top8_halc = get_permutation_top_features(X, y_halc, n_top=8)

    glm_lc = TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)
    glm_halc = TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)

    # XGBoost / LightGBM use tree importance-based top features
    temp_xgb = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    temp_lgb = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

    xgb_top_lc = get_tree_top_features(temp_xgb, X, np.log1p(y_lc), n_top=15)
    xgb_top_halc = get_tree_top_features(temp_xgb, X, np.log1p(y_halc), n_top=15)
    lgb_top_lc = get_tree_top_features(temp_lgb, X, np.log1p(y_lc), n_top=15)
    lgb_top_halc = get_tree_top_features(temp_lgb, X, np.log1p(y_halc), n_top=15)

    xgb_lc = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    xgb_halc = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    lgb_lc = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    lgb_halc = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

    regression_results = []

    # LC
    regression_results.append({
        "model": "MLP_LC",
        **evaluate_regression_cv(mlp_lc, X, y_lc, use_log_target=True),
    })
    regression_results.append({
        "model": "GLM_Tweedie_LC",
        **evaluate_regression_cv(glm_lc, X[glm_top8_lc], y_lc, use_log_target=False),
    })
    regression_results.append({
        "model": "XGBoost_LC",
        **evaluate_regression_cv(xgb_lc, X[xgb_top_lc], y_lc, use_log_target=True),
    })
    regression_results.append({
        "model": "LightGBM_LC",
        **evaluate_regression_cv(lgb_lc, X[lgb_top_lc], y_lc, use_log_target=True),
    })

    # HALC
    regression_results.append({
        "model": "MLP_HALC",
        **evaluate_regression_cv(mlp_halc, X, y_halc, use_log_target=True),
    })
    regression_results.append({
        "model": "GLM_Tweedie_HALC",
        **evaluate_regression_cv(glm_halc, X[glm_top8_halc], y_halc, use_log_target=False),
    })
    regression_results.append({
        "model": "XGBoost_HALC",
        **evaluate_regression_cv(xgb_halc, X[xgb_top_halc], y_halc, use_log_target=True),
    })
    regression_results.append({
        "model": "LightGBM_HALC",
        **evaluate_regression_cv(lgb_halc, X[lgb_top_halc], y_halc, use_log_target=True),
    })

    regression_results_df = pd.DataFrame(regression_results)
    print("\nRegression Results")
    print(regression_results_df.to_string(index=False))

    # Final regression model choice from report: LightGBM
    final_lgb_lc = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    final_lgb_halc = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

    final_lgb_lc.fit(X[lgb_top_lc], np.log1p(y_lc))
    final_lgb_halc.fit(X[lgb_top_halc], np.log1p(y_halc))

    # =========================
    # TASK 2: CLASSIFICATION
    # =========================

    print("\n" + "=" * 60)
    print("TASK 2: CLAIM CLASSIFICATION")
    print("=" * 60)

    # Weighted sample vector for weighted boosting
    class_counts = y_cs.value_counts()
    class_weight_map = {
        0: 1.0,
        1: class_counts[0] / class_counts[1]
    }
    sample_weights = y_cs.map(class_weight_map).values

    clf_results = []

    # Boosting
    boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf_results.append({
        "model": "Boosting",
        **evaluate_classification_cv(boosting, X, y_cs),
    })

    # Weighted boosting
    weighted_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf_results.append({
        "model": "Boosting_Weighted",
        **evaluate_classification_cv(weighted_boosting, X, y_cs, sample_weight=sample_weights),
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf_results.append({
        "model": "RandomForest",
        **evaluate_classification_cv(rf, X, y_cs),
    })

    # Logistic
    logit = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf_results.append({
        "model": "LogisticRegression",
        **evaluate_classification_cv(logit, X, y_cs),
    })

    # Logistic Lasso
    logit_lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf_results.append({
        "model": "LogisticRegression_Lasso",
        **evaluate_classification_cv(logit_lasso, X, y_cs),
    })

    # Logistic Ridge
    logit_ridge = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        C=0.5,
        random_state=RANDOM_STATE,
    )
    clf_results.append({
        "model": "LogisticRegression_Ridge",
        **evaluate_classification_cv(logit_ridge, X, y_cs),
    })

    # MLP
    mlp_cls = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    clf_results.append({
        "model": "MLP",
        **evaluate_classification_cv(mlp_cls, X, y_cs),
    })

    clf_results_df = pd.DataFrame([
        {"model": r["model"], "roc_auc": r["roc_auc"]}
        for r in clf_results
    ]).sort_values("roc_auc", ascending=False)

    print("\nClassification Results")
    print(clf_results_df.to_string(index=False))

    # Final classification model choice from report:
    # weighted boosting selected due to better classification behavior under imbalance
    final_cls = GradientBoostingClassifier(random_state=RANDOM_STATE)
    final_cls.fit(X, y_cs, sample_weight=sample_weights)

    # Show weighted boosting details
    weighted_detail = next(r for r in clf_results if r["model"] == "Boosting_Weighted")
    print("\nWeighted Boosting Confusion Matrix")
    print(weighted_detail["confusion_matrix"])
    print("\nWeighted Boosting Classification Report")
    print(weighted_detail["classification_report"])

    # =========================
    # SHAP INTERPRETATION
    # =========================

    print("\n" + "=" * 60)
    print("MODEL INTERPRETATION")
    print("=" * 60)

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(final_lgb_lc)
            shap_values = explainer.shap_values(X[lgb_top_lc])

            # LightGBM regression shap values shape: (n_samples, n_features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                "feature": lgb_top_lc,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)

            print("\nTop SHAP Features (final LightGBM LC)")
            print(shap_df.head(10).to_string(index=False))
        except Exception as e:
            print("SHAP step skipped:", e)
    else:
        print("SHAP not installed; skipped.")

    # =========================
    # SEGMENT ANALYSIS
    # =========================

    print("\n" + "=" * 60)
    print("SEGMENT ANALYSIS")
    print("=" * 60)

    seg_df = create_targets(df_train_raw)
    seg_df = create_segments(seg_df)

    if "tenure_segment" in seg_df.columns:
        tenure_summary = seg_df.groupby("tenure_segment")["LC"].agg(["mean", "std", "count"])
        print("\nLC by tenure segment")
        print(tenure_summary.to_string())

    if "premium_segment" in seg_df.columns:
        premium_summary = seg_df.groupby("premium_segment")["LC"].agg(["mean", "std", "count"])
        print("\nLC by premium segment")
        print(premium_summary.to_string())

    # =========================
    # FINAL TEST PREDICTIONS
    # =========================

    print("\n" + "=" * 60)
    print("FINAL TEST PREDICTIONS")
    print("=" * 60)

    lc_pred = np.clip(np.expm1(final_lgb_lc.predict(X_test[lgb_top_lc])), 0, None)
    halc_pred = np.clip(np.expm1(final_lgb_halc.predict(X_test[lgb_top_halc])), 0, None)
    cs_pred = final_cls.predict_proba(X_test)[:, 1]

    final_result = pd.DataFrame({
        "LC": lc_pred,
        "HALC": halc_pred,
        "CS": cs_pred,
    })

    final_result.to_csv(PREDICTION_OUTPUT, index=False)
    print(f"Saved predictions to {PREDICTION_OUTPUT.resolve()}")
    print(final_result.head().to_string(index=False))


if __name__ == "__main__":
    main()
