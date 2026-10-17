
"""
Script 04: Train baseline machine learning models.

LASI Wave 1 (2017-18)
Outcome: Self-reported jaundice/hepatitis among adults aged 45+.

Models:
- Logistic Regression
- Random Forest
- XGBoost

Includes state as a categorical predictor.
"""

from pathlib import Path
import json

import joblib
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
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODELS = ROOT / "outputs" / "models"
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"

for folder in [MODELS, TABLES, REPORTS]:
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

NUMERIC = [
    "dm005",  # Age
]

CATEGORICAL = [
    "dm003",                 # Sex
    "dm013",                 # Caste
    "dm010",                 # Religion
    "living_arrangements",   # Living arrangements
    "mpce_quintile",         # MPCE quintile
    "hc102",                 # Health insurance
    "residence",             # Rural/urban residence
    "state",                 # State/UT
]

FEATURES = NUMERIC + CATEGORICAL
EXPECTED = [TARGET] + FEATURES

missing_columns = sorted(set(EXPECTED) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(EXPECTED))

if missing_columns or extra_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}; "
        f"Unexpected columns: {extra_columns}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    raise ValueError(
        "Missing values found. Check the processed dataset first."
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

if not (df["dm005"] >= 45).all():
    raise ValueError("Participants younger than 45 were found.")

if not df["state"].isin(range(1, 37)).all():
    raise ValueError("Unexpected state codes found.")

if df[TARGET].nunique() != 2:
    raise ValueError("Both outcome classes must be present.")

print("=" * 65)
print("BASELINE MODEL TRAINING")
print("=" * 65)
print(f"Dataset rows: {len(df):,}")
print(f"Predictors: {len(FEATURES)}")
print(f"State categories observed: {df['state'].nunique()}")


# --------------------------------------------------
# 3. TRAIN/TEST SPLIT
# --------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Training positive cases: {int(y_train.sum()):,}")
print(f"Test positive cases: {int(y_test.sum()):,}")


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

# Convert categorical codes to strings so that they are treated
# explicitly as categories, not continuous numeric measurements.
X_train[CATEGORICAL] = X_train[CATEGORICAL].astype(str)
X_test[CATEGORICAL] = X_test[CATEGORICAL].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            CATEGORICAL,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

# Calculate imbalance ratio using training data only.
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    ),
}


# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []

for name, estimator in models.items():

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1],
    ).ravel()

    results.append({
        "model": name,
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(
            y_test, probability
        ),
        "precision": precision_score(
            y_test, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_test, prediction, zero_division=0
        ),
        "f1": f1_score(
            y_test, prediction, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, prediction
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    # Save the complete pipeline, including preprocessing.
    joblib.dump(model, MODELS / f"{name}.joblib")

    print(f"Completed: {name}")


# --------------------------------------------------
# 7. SAVE RESULTS AND TRAINING SUMMARY
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = TABLES / "baseline_model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "total_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "total_predictors": int(len(FEATURES)),
    "numeric_predictors": NUMERIC,
    "categorical_predictors": CATEGORICAL,
    "state_variable_included": True,
    "state_categories_in_full_dataset": int(df["state"].nunique()),
    "positive_cases": int(y.sum()),
    "negative_cases": int((y == 0).sum()),
    "positive_prevalence_percent": round(float(y.mean() * 100), 2),
    "scale_pos_weight_xgboost": float(scale_pos_weight),
    "random_state": 42,
    "classification_threshold": 0.5,
}

with open(
    REPORTS / "training_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("BASELINE MODEL RESULTS")
print("=" * 65)

print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {results_path}")
print(f"Training summary: {REPORTS / 'training_summary.json'}")
print(f"Models saved in: {MODELS}")
print("\nScript 04 completed successfully.")
