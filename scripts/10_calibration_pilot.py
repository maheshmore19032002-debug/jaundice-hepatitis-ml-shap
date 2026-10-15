
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)

# ==================================================
# 1. LOAD EXISTING PROCESSED DATASET
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "jaundice_hepatitis_model_data.csv"
)

TARGET = "jaundice_hepatitis"

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

print("=" * 70)
print("CALIBRATED RANDOM FOREST - THRESHOLD PILOT")
print("=" * 70)

print(f"\nTotal participants: {len(df):,}")
print(f"Positive cases: {(y == 1).sum():,}")
print(f"Negative cases: {(y == 0).sum():,}")
print(f"Overall prevalence: {y.mean():.4%}")

# ==================================================
# 2. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDataset prevalence")
print(f"Training prevalence: {y_train.mean():.4f}")
print(f"Test prevalence:     {y_test.mean():.4f}")

# ==================================================
# 3. DEFINE RANDOM FOREST
# ==================================================

base_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# ==================================================
# 4. CALIBRATE USING TRAINING DATA ONLY
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

calibrated_rf = CalibratedClassifierCV(
    estimator=base_rf,
    method="sigmoid",
    cv=cv,
    ensemble=True,
)

print("\nTraining calibrated Random Forest...")
calibrated_rf.fit(X_train, y_train)

# ==================================================
# 5. PREDICT TEST PROBABILITIES
# ==================================================

# Test data remains naturally imbalanced.
# Do not undersample or calibrate using the test set.

prob = calibrated_rf.predict_proba(X_test)[:, 1]

print("\nCALIBRATION SUMMARY")
print("-" * 70)

print(f"ROC-AUC:                 {roc_auc_score(y_test, prob):.4f}")
print(f"Average Precision:       {average_precision_score(y_test, prob):.4f}")
print(f"Brier Score:             {brier_score_loss(y_test, prob):.4f}")
print(f"Mean predicted probability: {prob.mean():.4f}")
print(f"Actual test prevalence:    {y_test.mean():.4f}")

print("\nPredicted probability distribution")
print(pd.Series(prob).describe().round(4).to_string())

# ==================================================
# 6. COMPARE DIFFERENT THRESHOLDS
# ==================================================

thresholds = [0.02, 0.03, 0.05, 0.10, 0.15]

print("\n" + "=" * 100)
print("THRESHOLD COMPARISON")
print("=" * 100)

header = (
    f"{'Threshold':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'Balanced Acc.':>14} "
    f"{'FP':>8} "
    f"{'FN':>8} "
    f"{'TP':>8} "
    f"{'TN':>8}"
)

print(header)
print("-" * 100)

for threshold in thresholds:

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_test, pred, zero_division=0
    )

    recall = recall_score(
        y_test, pred, zero_division=0
    )

    f1 = f1_score(
        y_test, pred, zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test, pred
    )

    print(
        f"{threshold:>10.2f} "
        f"{precision:>10.4f} "
        f"{recall:>10.4f} "
        f"{f1:>10.4f} "
        f"{balanced_acc:>14.4f} "
        f"{fp:>8} "
        f"{fn:>8} "
        f"{tp:>8} "
        f"{tn:>8}"
    )

# ==================================================
# 7. NO FILES SAVED
# ==================================================

print("\n" + "=" * 70)
print("Experiment completed.")
print("No CSV, model, JSON, graph, or report was saved.")
print("=" * 70)
