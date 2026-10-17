
"""
Script 05: Threshold evaluation for baseline models.

LASI Wave 1 (2017-18)
Outcome: Self-reported jaundice/hepatitis among adults aged 45+.

Thresholds are selected using training-set out-of-fold predictions.
The held-out test set is used only for final evaluation.
State is included as a categorical predictor through the saved pipeline.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_DIR = ROOT / "outputs" / "models"
OUT_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [MODEL_DIR, OUT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

if not DATA.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA}\n"
        "Run scripts/02_frequency_and_preprocessing.py first."
    )

df = pd.read_csv(DATA)

TARGET = "jaundice_hepatitis"

FEATURES = [
    "dm005",
    "dm003",
    "dm013",
    "dm010",
    "living_arrangements",
    "mpce_quintile",
    "hc102",
    "residence",
    "state",
]

EXPECTED_COLUMNS = [TARGET] + FEATURES

missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

if missing_columns or extra_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}; "
        f"Unexpected columns: {extra_columns}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    raise ValueError("Missing values found in processed dataset.")

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("Both outcome classes must be present.")

if not (df["dm005"] >= 45).all():
    raise ValueError("Participants younger than 45 were found.")

if not df["state"].isin(range(1, 37)).all():
    raise ValueError("Unexpected state codes found.")


# --------------------------------------------------
# 3. TRAIN/TEST SPLIT
# --------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Ensure categorical inputs have the same representation
# used by Script 04.
CATEGORICAL = [
    "dm003",
    "dm013",
    "dm010",
    "living_arrangements",
    "mpce_quintile",
    "hc102",
    "residence",
    "state",
]

X_train[CATEGORICAL] = X_train[CATEGORICAL].astype(str)
X_test[CATEGORICAL] = X_test[CATEGORICAL].astype(str)


# --------------------------------------------------
# 4. MODEL FILES
# --------------------------------------------------

model_files = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "random_forest": MODEL_DIR / "random_forest.joblib",
    "xgboost": MODEL_DIR / "xgboost.joblib",
}

for name, path in model_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {path}\n"
            "Run scripts/04_train_models.py first."
        )


# --------------------------------------------------
# 5. THRESHOLD EVALUATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

thresholds = np.arange(0.01, 1.00, 0.01)
results = []

for name, path in model_files.items():

    print(f"\nEvaluating: {name}")

    model = joblib.load(path)

    # Confirm saved pipeline expects the same features.
    if hasattr(model, "feature_names_in_"):
        saved_features = list(model.feature_names_in_)

        if saved_features != FEATURES:
            raise ValueError(
                f"Feature mismatch for {name}.\n"
                f"Saved model features: {saved_features}\n"
                f"Current features: {FEATURES}\n"
                "Retrain the model using Script 04."
            )

    # Training-only out-of-fold probabilities.
    oof_prob = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Select the threshold that maximizes training OOF F1.
    f1_values = [
        f1_score(
            y_train,
            (oof_prob >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    # Evaluate on untouched test set.
    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    default_pred = (
        test_prob >= 0.50
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1],
    ).ravel()

    results.append({
        "model": name,
        "selected_threshold": best_threshold,
        "training_oof_f1_at_selected_threshold": float(
            f1_values[best_index]
        ),
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_average_precision": average_precision_score(
            y_test, test_prob
        ),
        "test_precision": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "test_f1": f1_score(
            y_test, test_pred, zero_division=0
        ),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_test, test_pred
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
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

    print(f"Selected threshold: {best_threshold:.2f}")
    print("Evaluation completed.")


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = OUT_DIR / "threshold_evaluation_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "threshold_selection": (
        "Maximize F1 using 5-fold out-of-fold training predictions"
    ),
    "threshold_grid": {
        "start": 0.01,
        "stop_exclusive": 1.00,
        "step": 0.01,
    },
    "test_set_used_for_threshold_selection": False,
    "state_included_as_categorical_predictor": True,
    "predictors": FEATURES,
    "total_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "results_file": str(results_path.relative_to(ROOT)),
    "results": results,
}

summary_path = REPORT_DIR / "threshold_evaluation_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("THRESHOLD EVALUATION RESULTS")
print("=" * 65)

print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {results_path}")
print(f"Summary saved: {summary_path}")
print("\nScript 05 completed successfully.")
