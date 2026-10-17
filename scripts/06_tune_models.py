
"""
Script 06: Hyperparameter tuning for baseline machine learning models.

LASI Wave 1 (2017-18)
Outcome: Self-reported jaundice/hepatitis among adults aged 45+.

Models:
- Logistic Regression
- Random Forest
- XGBoost

State is included as a categorical predictor.
Hyperparameter selection uses training data only.
The held-out test set is reserved for final evaluation.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
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
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT / "data" / "processed"
    / "jaundice_hepatitis_model_data.csv"
)

MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [MODEL_DIR, TABLE_DIR, REPORT_DIR]:
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

NUMERIC = ["dm005"]

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

FEATURES = NUMERIC + CATEGORICAL
REQUIRED = FEATURES + [TARGET]

missing = sorted(set(REQUIRED) - set(df.columns))
extra = sorted(set(df.columns) - set(REQUIRED))

if missing or extra:
    raise ValueError(
        f"Missing columns: {missing}; "
        f"Unexpected columns: {extra}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    raise ValueError(
        "Missing values found. Check the processed dataset first."
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("Both outcome classes must be present.")

if not (df["dm005"] >= 45).all():
    raise ValueError("Participants younger than 45 were found.")

if not df["state"].isin(range(1, 37)).all():
    raise ValueError("Unexpected state codes found.")

print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)
print(f"Dataset rows: {len(df):,}")
print(f"Predictors: {len(FEATURES)}")
print(f"State categories: {df['state'].nunique()}")


# --------------------------------------------------
# 3. FEATURES AND TRAIN/TEST SPLIT
# --------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# State and other coded predictors are categorical,
# not continuous numeric measurements.
X[CATEGORICAL] = X[CATEGORICAL].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())

if positive == 0:
    raise ValueError("No positive cases in the training data.")

class_ratio = negative / positive

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Training positive cases: {positive:,}")
print(f"Training negative cases: {negative:,}")


# --------------------------------------------------
# 4. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, NUMERIC),
    ("categorical", categorical_pipe, CATEGORICAL),
])


# --------------------------------------------------
# 5. DEFINE MODELS AND SEARCH SPACES
# --------------------------------------------------

models = {
    "logistic_regression": (
        LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),
        {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
    ),

    "random_forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_leaf": [5, 10, 20],
            "classifier__max_features": ["sqrt", 0.5],
            "classifier__class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        },
    ),

    "xgboost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        ),
        {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "classifier__scale_pos_weight": [
                1.0,
                10.0,
                20.0,
                class_ratio,
            ],
        },
    ),
}


# --------------------------------------------------
# 6. CROSS-VALIDATION CONFIGURATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []
best_parameters = {}

# 12 parameter combinations per model.
N_ITER = 12


# --------------------------------------------------
# 7. TUNE, EVALUATE AND SAVE MODELS
# --------------------------------------------------

for name, (estimator, parameters) in models.items():

    print(f"\n{'-' * 60}")
    print(f"Tuning model: {name}")
    print(f"{'-' * 60}", flush=True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )

    # Hyperparameter selection uses training data only.
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Save tuned model separately from baseline models.
    model_path = MODEL_DIR / f"{name}_tuned.joblib"
    joblib.dump(best_model, model_path)

    best_parameters[name] = {
        "best_cv_average_precision": float(search.best_score_),
        "best_parameters": search.best_params_,
        "saved_model": str(model_path.relative_to(ROOT)),
        "predictors": FEATURES,
        "state_included": True,
    }

    # Evaluate the selected model on the untouched test set.
    probability = best_model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1],
    ).ravel()

    results.append({
        "model": name,
        "best_cv_average_precision": float(search.best_score_),
        "test_roc_auc": roc_auc_score(y_test, probability),
        "test_average_precision": average_precision_score(
            y_test, probability
        ),
        "test_precision_at_0.50": precision_score(
            y_test, prediction, zero_division=0
        ),
        "test_recall_at_0.50": recall_score(
            y_test, prediction, zero_division=0
        ),
        "test_f1_at_0.50": f1_score(
            y_test, prediction, zero_division=0
        ),
        "test_balanced_accuracy_at_0.50": balanced_accuracy_score(
            y_test, prediction
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    print(f"Best CV Average Precision: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Saved model: {model_path}", flush=True)


# --------------------------------------------------
# 8. SAVE RESULTS AND REPORTS
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "best_cv_average_precision",
    ascending=False,
)

results_path = TABLE_DIR / "tuned_model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)

report_path = REPORT_DIR / "tuning_summary.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "target": TARGET,
            "predictors": FEATURES,
            "numeric_predictors": NUMERIC,
            "categorical_predictors": CATEGORICAL,
            "state_included": True,
            "total_rows": int(len(df)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "cv_folds": 5,
            "random_search_iterations_per_model": N_ITER,
            "selection_metric": "average_precision",
            "classification_threshold": 0.50,
            "test_set_used_for_tuning": False,
            "models": best_parameters,
        },
        file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("TUNED MODEL RESULTS")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {results_path}")
print(f"Best parameters saved: {report_path}")
print(f"Tuned models saved in: {MODEL_DIR}")
print("\nScript 06 completed successfully.")
