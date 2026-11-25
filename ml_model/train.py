"""
Part 1 - ML Tool: training script (Logistic Regression + feature engineering) (Telco Customer Churn)
Run (from the repo root):  python -m ml_model.train
Output: ml_model/models/churn_model.joblib + model_meta.json  (full sklearn Pipeline: preprocessing + classifier)
"""
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from config import MODEL_PATH, TELCO_DATA_PATH
from ml_model.features import NUMERIC, CATEGORICAL, FEATURES, add_features
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

# All paths come from the shared config.py (team rule: no paths inside the code).
DATA_PATH = TELCO_DATA_PATH
META_PATH = MODEL_PATH.with_name("model_meta.json")   # saved next to the model



def load_and_clean(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has blank strings for new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return add_features(df.drop(columns=["customerID"]))


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler()),
                           ("poly", PolynomialFeatures(2, include_bias=False))]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    # Logistic Regression: best CV accuracy/AUC here, fast, explainable, well-calibrated probabilities
    clf = LogisticRegression(C=0.1, max_iter=5000)
    return Pipeline([("prep", pre), ("model", clf)])


MIN_ACCURACY = 0.80   # business constraint: accuracy must not drop below this


def find_threshold(X_tr, y_tr, min_acc=MIN_ACCURACY, margin=0.005) -> float:
    """Highest Churn-F1 threshold such that accuracy stays >= min_acc.
    Uses out-of-fold probabilities on the TRAIN set only (test set untouched).
    `margin` is a safety buffer because test accuracy fluctuates a bit vs CV."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(build_pipeline(), X_tr, y_tr, cv=skf,
                            method="predict_proba")[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.70, 0.01):
        pred = oof >= t
        if accuracy_score(y_tr, pred) >= min_acc + margin:
            f = f1_score(y_tr, pred)
            if f > best_f1:
                best_t, best_f1 = float(round(t, 2)), f
    return best_t


def main():
    df = load_and_clean()
    X, y = df[FEATURES], df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    thr = find_threshold(X_tr, y_tr)
    pipe = build_pipeline().fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f} | PR-AUC: {average_precision_score(y_te, proba):.3f}")
    for name, t in [("default threshold 0.50", 0.5), (f"tuned threshold {thr:.2f}", thr)]:
        preds = (proba >= t).astype(int)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_te, preds):.3f}")
        print(classification_report(y_te, preds, target_names=["Stay", "Churn"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps({"features": FEATURES, "numeric": NUMERIC,
                                     "categorical": CATEGORICAL,
                                     "threshold": thr}, indent=2))
    print(f"Saved model -> {MODEL_PATH}  (threshold={thr})")


if __name__ == "__main__":
    main()
