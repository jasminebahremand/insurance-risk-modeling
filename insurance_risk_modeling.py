"""
Insurance Risk Modeling
Predictive Modeling · Insurance Analytics
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy.stats import pointbiserialr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
TRAIN_PATH = Path("insurance_train.csv")
TEST_PATH = Path("insurance_test.csv")
PREDICTION_OUTPUT = Path("insurance_risk_predictions.csv")
PLOTS_DIR = "plots"

RANDOM_STATE = 42
N_SPLITS = 5

DATE_COLS = ["X.2", "X.3", "X.4", "X.5", "X.6"]
CATEGORICAL_COLS = ["X.7", "X.13", "X.19", "X.20", "X.21", "X.27"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Helpers
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def check_files() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Load data
# -----------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


# -----------------------------
# Targets and feature engineering
# -----------------------------
def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x15 = pd.to_numeric(df["X.15"], errors="coerce")
    x16 = pd.to_numeric(df["X.16"], errors="coerce")
    x18 = pd.to_numeric(df["X.18"], errors="coerce")

    df["CS"] = (x16 > 0).astype(int)
    df["LC"] = 0.0
    df["HALC"] = 0.0

    valid = x16 > 0
    df.loc[valid, "LC"] = (x15[valid] / x16[valid]).fillna(0.0)
    df.loc[valid, "HALC"] = (df.loc[valid, "LC"] * x18[valid]).fillna(0.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    reference_year = 2020

    if "X.5" in df.columns:
        df["age"] = reference_year - df["X.5"].dt.year

    if "X.6" in df.columns:
        df["license_years"] = reference_year - df["X.6"].dt.year

    if "X.22" in df.columns:
        df["vehicle_age"] = reference_year - pd.to_numeric(df["X.22"], errors="coerce")

    if "X.2" in df.columns:
        df["contract_start_year"] = df["X.2"].dt.year
        df["contract_start_month"] = df["X.2"].dt.month
        df["contract_start_weekday"] = df["X.2"].dt.weekday

    if {"X.2", "X.4"}.issubset(df.columns):
        df["policy_duration_days"] = (df["X.4"] - df["X.2"]).dt.days

    if {"X.2", "X.3"}.issubset(df.columns):
        df["days_since_last_renewal"] = (df["X.3"] - df["X.2"]).dt.days

    df = df.drop(columns=[c for c in DATE_COLS if c in df.columns], errors="ignore")
    return df


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))

    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    return X_train, X_test


# -----------------------------
# Plot 1: Loss cost distribution
# -----------------------------
def plot_loss_cost_distribution(y_lc: pd.Series) -> None:
    plt.figure()
    sns.histplot(y_lc, bins=40)
    plt.title("Loss Cost Distribution")
    plt.xlabel("Loss Cost")
    plt.ylabel("Frequency")
    save_plot("loss_cost_distribution.png")


# -----------------------------
# Feature selection
# -----------------------------
def get_glm_top_features(X: pd.DataFrame, y: pd.Series, n_top: int = 8) -> list[str]:
    glm = TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)
    glm.fit(X, y)

    perm = permutation_importance(
        glm,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )

    feature_df = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    return feature_df.head(n_top)["feature"].tolist()


def get_tree_top_features(model, X: pd.DataFrame, y_log: pd.Series, n_top: int = 15) -> list[str]:
    model.fit(X, y_log)

    feature_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return feature_df.head(n_top)["feature"].tolist()


# -----------------------------
# Regression evaluation
# -----------------------------
def evaluate_regression_cv(model, X: pd.DataFrame, y: pd.Series, use_log_target: bool = True) -> dict:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

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


# -----------------------------
# Classification evaluation
# -----------------------------
def evaluate_classification_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_true_all = []
    y_proba_all = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
        else:
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_valid)[:, 1]

        y_true_all.extend(y_valid.tolist())
        y_proba_all.extend(proba.tolist())

    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)

    return {
        "roc_auc": float(roc_auc_score(y_true_all, y_proba_all)),
        "y_true": y_true_all,
        "y_proba": y_proba_all,
    }


# -----------------------------
# Plot 2: Model comparison
# -----------------------------
def plot_model_performance(regression_results: pd.DataFrame, classification_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    reg_plot = regression_results.copy()
    sns.barplot(data=reg_plot, x="model", y="r2", ax=axes[0])
    axes[0].set_title("Regression Model Performance (R²)")
    axes[0].tick_params(axis="x", rotation=45)

    cls_plot = classification_results.copy()
    sns.barplot(data=cls_plot, x="model", y="roc_auc", ax=axes[1])
    axes[1].set_title("Classification Model Performance (ROC-AUC)")
    axes[1].tick_params(axis="x", rotation=45)

    save_plot("model_performance_comparison.png")


# -----------------------------
# Plot 3: ROC curve
# -----------------------------
def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    save_plot("roc_curve.png")


# -----------------------------
# Plot 4: SHAP / feature importance
# -----------------------------
def plot_shap_feature_importance(model, X: pd.DataFrame, feature_names: list[str]) -> None:
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[feature_names])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "importance": mean_abs_shap,
            }).sort_values("importance", ascending=False).head(10)

            plt.figure()
            sns.barplot(data=shap_df, x="importance", y="feature")
            plt.title("SHAP Feature Importance")
            save_plot("shap_feature_importance.png")
            return
        except Exception:
            pass

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(10)

    plt.figure()
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Feature Importance")
    save_plot("shap_feature_importance.png")


# -----------------------------
# Plot 5: Segment comparison
# -----------------------------
def create_segments(df: pd.DataFrame) -> pd.DataFrame:
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


def plot_segment_risk_comparison(seg_df: pd.DataFrame) -> None:
    if "tenure_segment" not in seg_df.columns:
        return

    summary = (
        seg_df.groupby("tenure_segment", observed=False)["LC"]
        .mean()
        .reset_index()
    )

    plt.figure()
    sns.barplot(data=summary, x="tenure_segment", y="LC")
    plt.title("Average Loss Cost by Tenure Segment")
    plt.xlabel("Tenure Segment")
    plt.ylabel("Average Loss Cost")
    plt.xticks(rotation=20)
    save_plot("segment_risk_comparison.png")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    check_files()

    train_raw, test_raw = load_data()
    print("Train shape:", train_raw.shape)
    print("Test shape:", test_raw.shape)

    train_df = create_targets(train_raw)
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_raw)

    cols_to_drop = ["X.1", "X.15", "X.16", "X.17", "X.18"]
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors="ignore")

    y_lc = train_df["LC"].copy()
    y_halc = train_df["HALC"].copy()
    y_cs = train_df["CS"].copy()

    feature_cols = [c for c in train_df.columns if c not in ["LC", "HALC", "CS"]]
    X, X_test = preprocess_data(train_df, test_df, feature_cols)

    plot_loss_cost_distribution(y_lc)

    # -------------------------
    # Regression models
    # -------------------------
    glm_top_lc = get_glm_top_features(X, y_lc, n_top=8)
    glm_top_halc = get_glm_top_features(X, y_halc, n_top=8)

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

    regression_models = {
        "MLP_LC": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE), X, y_lc, True),
        "GLM_Tweedie_LC": (TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000), X[glm_top_lc], y_lc, False),
        "XGBoost_LC": (XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=300,
                                   learning_rate=0.05, max_depth=5, subsample=0.8,
                                   colsample_bytree=0.8, random_state=RANDOM_STATE), X[xgb_top_lc], y_lc, True),
        "LightGBM_LC": (LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, n_estimators=300,
                                     learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE), X[lgb_top_lc], y_lc, True),
        "MLP_HALC": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE), X, y_halc, True),
        "GLM_Tweedie_HALC": (TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000), X[glm_top_halc], y_halc, False),
        "XGBoost_HALC": (XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=300,
                                     learning_rate=0.05, max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, random_state=RANDOM_STATE), X[xgb_top_halc], y_halc, True),
        "LightGBM_HALC": (LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, n_estimators=300,
                                       learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE), X[lgb_top_halc], y_halc, True),
    }

    regression_results = []
    for name, (model, X_model, y_model, use_log) in regression_models.items():
        result = evaluate_regression_cv(model, X_model, y_model, use_log_target=use_log)
        result["model"] = name
        regression_results.append(result)

    regression_results_df = pd.DataFrame(regression_results)
    print("\nRegression Results")
    print(regression_results_df.to_string(index=False))

    # final regression models
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

    # -------------------------
    # Classification models
    # -------------------------
    class_counts = y_cs.value_counts()
    sample_weights = y_cs.map({0: 1.0, 1: class_counts[0] / class_counts[1]}).values

    classification_models = {
        "Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), None),
        "Boosting_Weighted": (GradientBoostingClassifier(random_state=RANDOM_STATE), sample_weights),
        "RandomForest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), None),
        "LogisticRegression": (LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE), None),
        "LogisticRegression_Lasso": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE), None),
        "LogisticRegression_Ridge": (LogisticRegression(penalty="l2", solver="liblinear", C=0.5, max_iter=1000, random_state=RANDOM_STATE), None),
        "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE), None),
    }

    classification_results = []
    weighted_boosting_detail = None

    for name, (model, weights) in classification_models.items():
        result = evaluate_classification_cv(model, X, y_cs, sample_weight=weights)
        result["model"] = name
        classification_results.append(result)

        if name == "Boosting_Weighted":
            weighted_boosting_detail = result

    classification_results_df = pd.DataFrame(
        [{"model": r["model"], "roc_auc": r["roc_auc"]} for r in classification_results]
    ).sort_values("roc_auc", ascending=False)

    print("\nClassification Results")
    print(classification_results_df.to_string(index=False))

    plot_model_performance(regression_results_df, classification_results_df)

    if weighted_boosting_detail is not None:
        plot_roc_curve(weighted_boosting_detail["y_true"], weighted_boosting_detail["y_proba"])

    final_cls = GradientBoostingClassifier(random_state=RANDOM_STATE)
    final_cls.fit(X, y_cs, sample_weight=sample_weights)

    plot_shap_feature_importance(final_lgb_lc, X, lgb_top_lc)

    seg_df = create_targets(train_raw)
    seg_df = create_segments(seg_df)
    plot_segment_risk_comparison(seg_df)

    # -------------------------
    # Final predictions
    # -------------------------
    lc_pred = np.clip(np.expm1(final_lgb_lc.predict(X_test[lgb_top_lc])), 0, None)
    halc_pred = np.clip(np.expm1(final_lgb_halc.predict(X_test[lgb_top_halc])), 0, None)
    cs_pred = final_cls.predict_proba(X_test)[:, 1]

    final_predictions = pd.DataFrame({
        "LC": lc_pred,
        "HALC": halc_pred,
        "CS": cs_pred,
    })

    final_predictions.to_csv(PREDICTION_OUTPUT, index=False)

    print(f"\nPredictions saved to: {PREDICTION_OUTPUT}")
    print("Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
