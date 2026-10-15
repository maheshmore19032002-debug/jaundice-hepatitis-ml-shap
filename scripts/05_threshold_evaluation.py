
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_DIR = ROOT / "outputs" / "models"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

target = "jaundice_hepatitis"
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "random_forest": MODEL_DIR / "random_forest.joblib",
    "xgboost": MODEL_DIR / "xgboost.joblib",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, path in models.items():
    model = pd.read_pickle(path) if False else None

    import joblib
    model = joblib.load(path)

    # Out-of-fold probabilities: training data only
    oof_prob = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Select threshold maximizing F1 on training OOF predictions
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_values = [
        f1_score(y_train, (oof_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(f1_values))])

    # Evaluate selected threshold on untouched test set
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    default_pred = (test_prob >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "selected_threshold": best_threshold,
        "training_oof_f1_at_selected_threshold": max(f1_values),
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_average_precision": average_precision_score(y_test, test_prob),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "test_precision_at_0.50": precision_score(
            y_test, default_pred, zero_division=0
        ),
        "test_recall_at_0.50": recall_score(
            y_test, default_pred, zero_division=0
        ),
        "test_f1_at_0.50": f1_score(
            y_test, default_pred, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)
results_path = OUT_DIR / "threshold_evaluation_comparison.csv"
results_df.to_csv(results_path, index=False)

summary = {
    "threshold_selection": "Maximize F1 using 5-fold out-of-fold training predictions",
    "threshold_grid": [0.01, 0.50, 0.01],
    "test_set_used_for_threshold_selection": False,
    "results_file": str(results_path),
    "results": results,
}

summary_path = ROOT / "outputs" / "reports" / "threshold_evaluation_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nTHRESHOLD EVALUATION RESULTS")
print(results_df.round(4).to_string(index=False))
print(f"\nResults saved: {results_path}")
print(f"Summary saved: {summary_path}")
