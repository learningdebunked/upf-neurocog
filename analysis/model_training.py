"""
model_training.py  —  UPF-NeuroCog v2.0
=========================================
Trains and evaluates the cognitive-fatigue risk prediction model.

Architecture
------------
  1. LLM NOVA classifier  (nova_llm_classifier.py)  →  NOVA 1-4 score per food item
  2. XGBoost classifier on tabular cohort features   →  binary risk (0/1)
  3. SHAP explainer                                   →  feature importance
  4. Optional: secondary LLM narrative report on top SHAP features

Inputs
------
  data/integrated_cohort.parquet   — output of integration_pipeline.py

Outputs
-------
  models/xgb_model.pkl             — trained XGBoost model
  models/label_encoder.pkl         — LabelEncoder for categorical columns
  results/metrics.json             — AUROC, F1, precision, recall, accuracy
  results/shap_summary.csv         — SHAP mean |value| per feature
  figures/shap_beeswarm.png        — SHAP beeswarm plot
  figures/roc_curve.png            — ROC curve

Usage
-----
    python analysis/model_training.py \
        --cohort   data/integrated_cohort.parquet \
        --output   results/ \
        --models   models/ \
        --figures  figures/
"""

import os
import json
import pickle
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Optional heavy imports ──────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — run `pip install xgboost`")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — run `pip install shap`")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1.  Feature catalogue
# ---------------------------------------------------------------------------

# Core tabular features used by the model.
# Pipeline outputs may not include all columns; missing ones are imputed.

FEATURE_GROUPS = {
    "dietary": [
        "upf_energy_pct",          # % of daily energy from NOVA 3/4 foods
        "nova4_energy_kcal",        # absolute NOVA 4 energy (kcal)
        "total_energy_kcal",        # total daily energy intake
        "sugar_g",                  # total sugars (g)
        "sodium_mg",                # sodium (mg)
        "fiber_g",                  # dietary fiber (g)
        "saturated_fat_g",          # saturated fat (g)
    ],
    "behavioural": [
        "brfss_poor_mh_days",       # poor mental health days (past 30)
        "brfss_sleep_insufficient", # binary sleep flag
        "brfss_inactive",           # physical inactivity flag
        "brfss_stress_score",       # composite stress-behaviour score
    ],
    "clinical": [
        "bmi",
        "hba1c_pct",
        "systolic_bp",
        "diastolic_bp",
    ],
    "demographic": [
        "age_years_norm",           # normalised age
        "sex_male",                 # binary (1=male)
        "race_ethnicity_code",      # encoded
        "pir_ratio",                # poverty-income ratio
    ],
    "sdoh": [
        "svi_percentile",
        "food_desert_flag",
        "places_mental_health_poor_pct",
    ],
    "eeg": [
        "eeg_pac_mi",               # DEAP-derived PAC modulation index
        "eeg_frontal_alpha_asym",   # DEAP-derived FAA
    ],
}

ALL_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]
TARGET_COL   = "cognitive_risk_label"


# ---------------------------------------------------------------------------
# 2.  Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_cohort(parquet_path: str) -> pd.DataFrame:
    logger.info(f"Loading cohort: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for XGBoost."""

    # Keep only features that are present; add zeros for missing
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"  {len(missing)} feature(s) not in cohort; zero-imputing: {missing[:8]} …")
        for col in missing:
            df[col] = 0.0

    X = df[ALL_FEATURES].copy()

    # Encode any remaining categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Impute numeric NaNs with median
    X = X.fillna(X.median(numeric_only=True))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in cohort.")
    y = df[TARGET_COL].astype(int)

    logger.info(f"  Features: {X.shape[1]}  |  positive rate: {y.mean():.3f}")
    return X, y, encoders


# ---------------------------------------------------------------------------
# 3.  Model training
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,
    "scale_pos_weight":  1,       # adjust for class imbalance below
    "eval_metric":       "auc",
    "random_state":      42,
    "tree_method":       "hist",  # GPU-ready; falls back to CPU hist
    "n_jobs":            -1,
}


def train(X: pd.DataFrame, y: pd.Series,
          cv_folds: int = 5) -> tuple:
    """
    Trains XGBoost with 5-fold StratifiedKFold cross-validation.
    Returns (final_model, cv_results).
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError("xgboost is required for model training.")

    # Handle class imbalance
    pos_rate = y.mean()
    if pos_rate < 0.35 or pos_rate > 0.65:
        scale = (1 - pos_rate) / (pos_rate + 1e-9)
        logger.info(f"  Imbalanced classes (pos_rate={pos_rate:.3f}). "
                    f"Setting scale_pos_weight={scale:.2f}")
        XGB_PARAMS["scale_pos_weight"] = scale

    model = xgb.XGBClassifier(**XGB_PARAMS)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
        return_train_score=True,
        n_jobs=-1,
    )

    logger.info(f"  CV AUROC: {cv_results['test_roc_auc'].mean():.4f} "
                f"± {cv_results['test_roc_auc'].std():.4f}")
    logger.info(f"  CV F1:    {cv_results['test_f1'].mean():.4f} "
                f"± {cv_results['test_f1'].std():.4f}")

    # Final model on full data
    model.fit(X, y)
    return model, cv_results


# ---------------------------------------------------------------------------
# 4.  Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(model, X: pd.DataFrame, y: pd.Series,
             cv_results: dict) -> dict:
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred       = model.predict(X)

    metrics = {
        "auroc":                round(roc_auc_score(y, y_pred_proba), 4),
        "f1":                   round(f1_score(y, y_pred), 4),
        "precision":            round(precision_score(y, y_pred), 4),
        "recall":               round(recall_score(y, y_pred), 4),
        "accuracy":             round(accuracy_score(y, y_pred), 4),
        # Cross-validation summary
        "cv_auroc_mean":        round(float(cv_results["test_roc_auc"].mean()), 4),
        "cv_auroc_std":         round(float(cv_results["test_roc_auc"].std()), 4),
        "cv_f1_mean":           round(float(cv_results["test_f1"].mean()), 4),
        "cv_f1_std":            round(float(cv_results["test_f1"].std()), 4),
        "n_samples":            int(len(y)),
        "n_features":           int(X.shape[1]),
        "positive_rate":        round(float(y.mean()), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# 5.  SHAP explainability
# ---------------------------------------------------------------------------

def shap_explain(model, X: pd.DataFrame,
                 output_dir: str, figures_dir: str):
    """
    Computes SHAP values and saves:
      - results/shap_summary.csv       (mean |SHAP| per feature)
      - figures/shap_beeswarm.png
    """
    if not _SHAP_AVAILABLE:
        logger.warning("shap not installed — skipping SHAP step.")
        return

    logger.info("Computing SHAP values …")

    # Use a sample for speed on large cohorts (max 5000 rows)
    X_sample = X.sample(min(5000, len(X)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    shap_df = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    summary_path = os.path.join(output_dir, "shap_summary.csv")
    shap_df.to_csv(summary_path, index=False)
    logger.info(f"  Saved SHAP summary → {summary_path}")
    logger.info(f"  Top 5 features:\n{shap_df.head(5).to_string(index=False)}")

    # Beeswarm plot
    if _MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=X.columns.tolist(),
                          show=False, max_display=20)
        fig_path = os.path.join(figures_dir, "shap_beeswarm.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved SHAP beeswarm → {fig_path}")


# ---------------------------------------------------------------------------
# 6.  ROC curve
# ---------------------------------------------------------------------------

def plot_roc(model, X: pd.DataFrame, y: pd.Series,
             figures_dir: str, auroc: float):
    if not _MPL_AVAILABLE:
        return
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Cognitive Fatigue Risk (UPF-NeuroCog v2.0)")
    ax.legend(loc="lower right")
    roc_path = os.path.join(figures_dir, "roc_curve.png")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved ROC curve → {roc_path}")


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def run(cohort_path: str  = "data/integrated_cohort.parquet",
        output_dir: str   = "results/",
        models_dir: str   = "models/",
        figures_dir: str  = "figures/") -> dict:

    for d in [output_dir, models_dir, figures_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    df = load_cohort(cohort_path)
    X, y, encoders = preprocess(df)

    logger.info("Training XGBoost model …")
    model, cv_results = train(X, y)

    logger.info("Evaluating model …")
    metrics = evaluate(model, X, y, cv_results)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Saved metrics → {metrics_path}")
    logger.info(json.dumps({k: v for k, v in metrics.items()
                             if "cv" in k or k in ["auroc", "f1"]}, indent=2))

    # Save model & encoders
    model_path = os.path.join(models_dir, "xgb_model.pkl")
    enc_path   = os.path.join(models_dir, "label_encoder.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    with open(enc_path, "wb") as fh:
        pickle.dump(encoders, fh)
    logger.info(f"Saved model → {model_path}")

    # SHAP
    shap_explain(model, X, output_dir, figures_dir)

    # ROC plot
    plot_roc(model, X, y, figures_dir, metrics["auroc"])

    logger.info("Model training pipeline complete.")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPF-NeuroCog v2.0 — Model Training")
    parser.add_argument("--cohort",   default="data/integrated_cohort.parquet",
                        help="Path to integrated cohort parquet")
    parser.add_argument("--output",   default="results/",
                        help="Directory for metrics and SHAP CSVs")
    parser.add_argument("--models",   default="models/",
                        help="Directory for saved model artefacts")
    parser.add_argument("--figures",  default="figures/",
                        help="Directory for figures (SHAP, ROC)")
    args = parser.parse_args()

    run(
        cohort_path=args.cohort,
        output_dir=args.output,
        models_dir=args.models,
        figures_dir=args.figures,
    )
