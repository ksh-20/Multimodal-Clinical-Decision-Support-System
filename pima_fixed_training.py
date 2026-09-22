"""
pima_fixed_training.py — Retrains the Pima Indians diabetes model with the
correct methodology (split-first, no target leakage) and saves the pipeline
artifact to models/diabetes_model.joblib.

Run from inside knowledge_layer/:
  python pima_fixed_training.py

Prerequisites:
  pip install -r requirements.txt
  Place pima-indians-diabetes.csv in data/

The saved artifact has the same structure as the one produced by the notebook
(pima_diabetes_ml_benchmark.ipynb), so adapters.py works with both.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pima_training")

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

try:
    from imblearn.combine import SMOTETomek
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    log.warning("imbalanced-learn not installed; training without SMOTETomek.")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config import DIABETES_MODEL_PATH, DIABETES_PHYSIO_COLS, PIMA_CSV_PATH

SEED = 42
COL_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["HOMA_IR"] = (d["Glucose"] * d["Insulin"]) / 405.0
    d["BMI_Cat"] = pd.cut(d["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    d["Glucose_Cat"] = pd.cut(d["Glucose"], bins=[0, 99, 139, 199, 300], labels=[0, 1, 2, 3]).astype(float)
    d["Age_Cat"] = pd.cut(d["Age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    d["Metabolic_Score"] = (
        (d["Glucose"] >= 140).astype(int)
        + (d["BMI"] >= 30).astype(int)
        + (d["BloodPressure"] >= 80).astype(int)
        + (d["Age"] >= 35).astype(int)
    )
    d["Insulin_Glucose_Ratio"] = d["Insulin"] / (d["Glucose"] + 1e-5)
    d["Glucose_BMI"] = d["Glucose"] * d["BMI"]
    d["Age_BMI"] = d["Age"] * d["BMI"]
    d["DPF_Glucose"] = d["DiabetesPedigreeFunction"] * d["Glucose"]
    d["Preg_Age_Ratio"] = d["Pregnancies"] / (d["Age"] + 1.0)
    d["Insulin_Log"] = np.log1p(d["Insulin"])
    d["Glucose_Log"] = np.log1p(d["Glucose"])
    return d


def load_data() -> pd.DataFrame:
    if not PIMA_CSV_PATH.exists():
        log.error(
            "CSV not found at %s. Copy pima-indians-diabetes.csv into data/.",
            PIMA_CSV_PATH,
        )
        sys.exit(1)
    df = pd.read_csv(PIMA_CSV_PATH, names=COL_NAMES)
    log.info("Loaded %d rows from %s.", len(df), PIMA_CSV_PATH)
    return df


def train() -> None:
    df = load_data()

    df_clean = df.copy()
    df_clean[DIABETES_PHYSIO_COLS] = df_clean[DIABETES_PHYSIO_COLS].replace(0, np.nan)
    X_raw = df_clean.drop(columns=["Outcome"])
    y = df_clean["Outcome"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.15, stratify=y, random_state=SEED
    )

    train_medians = X_train_raw[DIABETES_PHYSIO_COLS].median()
    log.info("Training-set imputation medians:\n%s", train_medians.to_string())

    X_train_raw = X_train_raw.copy()
    X_test_raw = X_test_raw.copy()
    X_train_raw[DIABETES_PHYSIO_COLS] = X_train_raw[DIABETES_PHYSIO_COLS].fillna(train_medians)
    X_test_raw[DIABETES_PHYSIO_COLS] = X_test_raw[DIABETES_PHYSIO_COLS].fillna(train_medians)

    X_train_eng = engineer_features(X_train_raw)
    X_test_eng = engineer_features(X_test_raw)
    feature_list = list(X_train_eng.columns)
    log.info("Feature set: %d features.", len(feature_list))

    if HAS_SMOTE:
        smt = SMOTETomek(random_state=SEED)
        X_train_res, y_train_res = smt.fit_resample(X_train_eng, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=feature_list)
        log.info("Post-SMOTETomek distribution: %s", Counter(y_train_res))
    else:
        X_train_res, y_train_res = X_train_eng, y_train
        log.info("Skipping SMOTETomek (not installed).")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=feature_list)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_eng), columns=feature_list)
    X_train_orig_scaled = pd.DataFrame(scaler.transform(X_train_eng), columns=feature_list)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=SEED),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="logloss", n_jobs=-1
        )
    if HAS_LGB:
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=-1
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_name, best_score, best_model = "", -1.0, None

    log.info("Running 5-fold stratified CV on pre-SMOTE training data (metric: ROC-AUC)...")
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_train_orig_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        log.info("  %s: %.4f ± %.4f", name, scores.mean(), scores.std())
        if scores.mean() > best_score:
            best_score, best_name, best_model = scores.mean(), name, clf

    log.info("Champion (by CV ROC-AUC): %s (%.4f)", best_name, best_score)

    best_model.fit(X_train_scaled, y_train_res)

    y_prob_test = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = best_model.predict(X_test_scaled)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_sens = recall_score(y_test, y_pred_test)
    test_spec = accuracy_score(y_test[y_test == 0], y_pred_test[y_test == 0])
    test_f1 = f1_score(y_test, y_pred_test)

    log.info("\n=== HOLD-OUT TEST SET RESULTS ===")
    log.info("  Test ROC-AUC:   %.4f", test_auc)
    log.info("  Test Accuracy:  %.2f%%", test_acc * 100)
    log.info("  Sensitivity:    %.2f%%", test_sens * 100)
    log.info("  Specificity:    %.2f%%", test_spec * 100)
    log.info("  F1-Score:       %.4f", test_f1)
    log.info("  Test n=%d  (positive=%d, negative=%d)", len(y_test), y_test.sum(), (y_test == 0).sum())
    acc_ci = 1.96 * (test_acc * (1 - test_acc) / len(y_test)) ** 0.5
    log.info("  Accuracy 95%% CI: ±%.1f%%", acc_ci * 100)

    DIABETES_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "scaler": scaler,
        "champion_model": best_model,
        "champion_name": best_name,
        "cv_roc_auc": best_score,
        "feature_list": feature_list,
        "physio_cols": DIABETES_PHYSIO_COLS,
        "train_medians": train_medians.to_dict(),
        "seed": SEED,
        "test_metrics": {
            "roc_auc": round(test_auc, 4),
            "accuracy": round(test_acc, 4),
            "sensitivity": round(test_sens, 4),
            "f1": round(test_f1, 4),
        },
    }
    joblib.dump(artifact, DIABETES_MODEL_PATH)
    log.info("Artifact saved to: %s", DIABETES_MODEL_PATH)


if __name__ == "__main__":
    train()
