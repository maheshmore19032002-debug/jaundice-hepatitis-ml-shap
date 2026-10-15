
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


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "jaundice_hepatitis"
NUMERIC = ["dm005"]
CATEGORICAL = [
    "dm003", "dm013", "dm010",
    "living_arrangements", "mpce_quintile",
    "hc102", "residence",
]
FEATURES = NUMERIC + CATEGORICAL

df = pd.read_csv(DATA)

required = FEATURES + [TARGET]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Required columns missing: {missing}")

df = df[required].copy()
X = df[FEATURES]
y = df[TARGET].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Outcome must contain both 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

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

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())
class_ratio = negative / positive

models = {
    "logistic_regression": (
        LogisticRegression(max_iter=3000, random_state=42),
        {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_leaf": [5, 10, 20],
            "classifier__max_features": ["sqrt", 0.5],
            "classifier__class_weight": [
                "balanced", "balanced_subsample"
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
                1.0, 10.0, 20.0, class_ratio
            ],
        },
    ),
}

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

results = []
best_parameters = {}

for name, (estimator, parameters) in models.items():
    print(f"\nTuning: {name}", flush=True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Save tuned model separately from baseline models
    model_path = MODEL_DIR / f"{name}_tuned.joblib"
    joblib.dump(best_model, model_path)

    best_parameters[name] = {
        "best_cv_average_precision": float(search.best_score_),
        "best_parameters": search.best_params_,
        "saved_model": str(model_path),
    }

    # Final held-out test evaluation at threshold 0.50
    probability = best_model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, prediction, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "best_cv_average_precision": search.best_score_,
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

    print(f"Best CV AP: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Saved: {model_path}", flush=True)

results_df = pd.DataFrame(results).sort_values(
    "best_cv_average_precision", ascending=False
)

results_path = TABLE_DIR / "tuned_model_comparison.csv"
results_df.to_csv(results_path, index=False)

report_path = REPORT_DIR / "tuning_summary.json"
report_path.write_text(
    json.dumps(best_parameters, indent=2, default=str),
    encoding="utf-8",
)

print("\nTUNED MODEL RESULTS")
print(results_df.round(4).to_string(index=False))
print(f"\nResults saved: {results_path}")
print(f"Best parameters saved: {report_path}")
