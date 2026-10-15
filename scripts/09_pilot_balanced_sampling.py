
"""
Script 09: Pilot test using random undersampling.

- Retain all positive cases in the training split.
- Randomly sample equal numbers of negative cases.
- Train Logistic Regression and Random Forest.
- Evaluate on the original imbalanced test split.
- SAVE NOTHING: no files are created or modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

# =====================================================
# 1. SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "jaundice_hepatitis_model_data.csv"
)

TARGET = "jaundice_hepatitis"

NUMERIC = ["dm005"]

CATEGORICAL = [
    "dm003",
    "dm013",
    "dm010",
    "living_arrangements",
    "mpce_quintile",
    "hc102",
    "residence",
]

FEATURES = NUMERIC + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.20

# =====================================================
# 2. LOAD EXISTING DATA (READ ONLY)
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required = FEATURES + [TARGET]
missing = sorted(set(required) - set(df.columns))

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].copy()

df[TARGET] = pd.to_numeric(
    df[TARGET], errors="raise"
).astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1.")

if df[required].isna().any().any():
    raise ValueError(
        "Missing values detected. Check the processed dataset first."
    )

X = df[FEATURES]
y = df[TARGET]

print("=" * 65)
print("PILOT EXPERIMENT: RANDOM UNDERSAMPLING")
print("No output files will be saved.")
print("=" * 65)

print(f"\nTotal participants: {len(df):,}")
print(f"Positive cases: {(y == 1).sum():,}")
print(f"Negative cases: {(y == 0).sum():,}")
print(f"Original prevalence: {y.mean() * 100:.2f}%")

# =====================================================
# 3. SPLIT FIRST — TEST SET REMAINS IMBALANCED
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining split before balancing:")
print(y_train.value_counts().sort_index().to_string())

print("\nUntouched test split:")
print(y_test.value_counts().sort_index().to_string())

# =====================================================
# 4. UNDERSAMPLE NEGATIVE TRAINING CASES
# =====================================================

train_df = X_train.copy()
train_df[TARGET] = y_train.to_numpy()

positive_df = train_df[train_df[TARGET] == 1]
negative_df = train_df[train_df[TARGET] == 0]

if len(positive_df) == 0:
    raise ValueError("No positive cases in the training split.")

# Keep all positive training cases.
# Randomly select the same number of negative cases.
negative_sample = negative_df.sample(
    n=len(positive_df),
    random_state=RANDOM_STATE,
    replace=False,
)

# Temporary in-memory balanced dataset only.
balanced_train = pd.concat(
    [positive_df, negative_sample],
    axis=0,
).sample(frac=1, random_state=RANDOM_STATE)

X_train_balanced = balanced_train[FEATURES]
y_train_balanced = balanced_train[TARGET].astype(int)

print("\nBalanced training split:")
print(y_train_balanced.value_counts().sort_index().to_string())
print(f"Balanced training size: {len(balanced_train):,}")

# =====================================================
# 5. PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC),
    ("categorical", categorical_pipeline, CATEGORICAL),
])

# =====================================================
# 6. MODELS
# =====================================================

models = {
    "logistic_regression_undersampled": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "random_forest_undersampled": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# =====================================================
# 7. TRAIN AND EVALUATE
# =====================================================

results = []

for name, classifier in models.items():
    print(f"\nTraining: {name}", flush=True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train_balanced, y_train_balanced)

    # Evaluate only on the original imbalanced test split.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "precision_at_050": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_at_050": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_at_050": f1_score(
            y_test, predictions, zero_division=0
        ),
        "balanced_accuracy_at_050": balanced_accuracy_score(
            y_test, predictions
        ),
        "brier_score": brier_score_loss(y_test, probabilities),
        "mean_predicted_probability": probabilities.mean(),
        "actual_test_prevalence": y_test.mean(),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    results.append(result)

    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"Average Precision: {result['average_precision']:.4f}")
    print(f"Precision: {result['precision_at_050']:.4f}")
    print(f"Recall: {result['recall_at_050']:.4f}")
    print(f"F1: {result['f1_at_050']:.4f}")
    print(
        "Balanced accuracy: "
        f"{result['balanced_accuracy_at_050']:.4f}"
    )
    print(f"Brier score: {result['brier_score']:.4f}")
    print(
        "Mean predicted probability: "
        f"{result['mean_predicted_probability']:.4f}"
    )
    print(
        "Actual test prevalence: "
        f"{result['actual_test_prevalence']:.4f}"
    )
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# =====================================================
# 8. DISPLAY COMPARISON — TERMINAL ONLY
# =====================================================

results_df = pd.DataFrame(results).sort_values(
    "average_precision",
    ascending=False,
)

print("\n" + "=" * 65)
print("PILOT RESULTS: IMBALANCED TEST SET")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

print("\nExperiment completed.")
print("No CSV, model, JSON, graph or report was saved.")
