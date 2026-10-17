
"""
Script 07: Class imbalance experiments using class weights and SMOTENC.

LASI Wave 1 (2017-18)
Outcome: Self-reported jaundice/hepatitis among adults aged 45+.

Experiments:
- Logistic Regression with class weighting
- Logistic Regression with SMOTENC
- Random Forest with class weighting
- Random Forest with SMOTENC

All preprocessing and resampling are fitted within training folds.
The held-out test set is never resampled.
State is treated as a categorical predictor.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
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

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC


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
REQUIRED = FEATURES + [TARGET]

missing_columns = sorted(set(REQUIRED) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(REQUIRED))

if missing_columns or extra_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}; "
        f"Unexpected columns: {extra_columns}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    raise ValueError(
        "Missing values found in processed dataset."
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError(
        "Outcome must contain only 0 and 1."
    )

if df[TARGET].nunique() != 2:
    raise ValueError(
        "Both outcome classes must be present."
    )

if not (df["dm005"] >= 45).all():
    raise ValueError(
        "Participants younger than 45 were found."
    )

if not df["state"].isin(range(1, 37)).all():
    raise ValueError(
        "Unexpected state codes found."
    )

print("=" * 70)
print("CLASS IMBALANCE EXPERIMENTS")
print("=" * 70)
print(f"Dataset rows: {len(df):,}")
print(f"Positive cases: {(df[TARGET] == 1).sum():,}")
print(f"Negative cases: {(df[TARGET] == 0).sum():,}")
print(f"State categories: {df['state'].nunique()}")


# --------------------------------------------------
# 3. TRAIN/TEST SPLIT
# --------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Convert coded predictors to categorical strings.
# Age remains numeric.
X[CATEGORICAL] = X[CATEGORICAL].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")


# --------------------------------------------------
# 4. IMPUTATION BEFORE SMOTENC
# --------------------------------------------------

# The transformer preserves the feature order:
# age first, followed by the eight categorical predictors.
impute_for_smote = ColumnTransformer(
    transformers=[
        (
            "age",
            SimpleImputer(strategy="median"),
            NUMERIC,
        ),
        (
            "categories",
            SimpleImputer(strategy="most_frequent"),
            CATEGORICAL,
        ),
    ],
    verbose_feature_names_out=False,
)

# After the above transformation:
# index 0 = age
# indices 1-8 = categorical predictors, including state
categorical_indices = list(range(1, len(FEATURES)))

smote = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=0.5,
    random_state=42,
    k_neighbors=5,
)


# --------------------------------------------------
# 5. PREPROCESSING AFTER RESAMPLING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SkPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            [0],
        ),
        (
            "categorical",
            SkPipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_indices,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 6. MODEL CONFIGURATIONS
# --------------------------------------------------

models = {
    "logistic_regression": {
        "estimator": LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        ),
        "smote_estimator": LogisticRegression(
            C=0.01,
            class_weight=None,
            max_iter=3000,
            random_state=42,
        ),
    },

    "random_forest": {
        "estimator": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            max_features=0.5,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "smote_estimator": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            max_features=0.5,
            max_depth=10,
            class_weight=None,
            random_state=42,
            n_jobs=1,
        ),
    },
}


# --------------------------------------------------
# 7. CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []
saved_models = {}

for model_name, configs in models.items():

    for method in ["class_weight", "smotenc"]:

        print(
            f"\nExperiment: {model_name} | {method}",
            flush=True,
        )

        steps = [
            (
                "impute_for_smote",
                clone(impute_for_smote),
            ),
        ]

        if method == "smotenc":
            steps.append(
                ("sampler", clone(smote))
            )
        else:
            steps.append(
                ("sampler", "passthrough")
            )

        classifier = (
            configs["estimator"]
            if method == "class_weight"
            else configs["smote_estimator"]
        )

        steps.extend([
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                clone(classifier),
            ),
        ])

        pipeline = ImbPipeline(steps)

        # Resampling occurs only when the pipeline is fitted.
        # During cross-validation, each training fold is resampled
        # independently; validation folds remain untouched.
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            },
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        # Fit on the full training split only.
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        key = f"{model_name}_{method}"
        model_path = MODEL_DIR / f"{key}.joblib"

        joblib.dump(pipeline, model_path)
        saved_models[key] = str(model_path.relative_to(ROOT))

        results.append({
            "model": model_name,
            "method": method,
            "cv_average_precision_mean": float(
                cv_results["test_average_precision"].mean()
            ),
            "cv_average_precision_sd": float(
                cv_results["test_average_precision"].std()
            ),
            "cv_roc_auc_mean": float(
                cv_results["test_roc_auc"].mean()
            ),
            "cv_roc_auc_sd": float(
                cv_results["test_roc_auc"].std()
            ),
            "test_average_precision": float(
                average_precision_score(y_test, probabilities)
            ),
            "test_roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "test_precision_at_050": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "test_recall_at_050": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "test_f1_at_050": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "test_balanced_accuracy_at_050": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

        print(
            "CV Average Precision: "
            f"{cv_results['test_average_precision'].mean():.4f} "
            "+/- "
            f"{cv_results['test_average_precision'].std():.4f}",
            flush=True,
        )

        print(
            "Test Average Precision: "
            f"{average_precision_score(y_test, probabilities):.4f}",
            flush=True,
        )

        print(f"Saved model: {model_path}", flush=True)


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "cv_average_precision_mean",
    ascending=False,
)

results_path = TABLE_DIR / "imbalance_experiment_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "primary_metric": "5-fold CV Average Precision",
    "target": TARGET,
    "predictors": FEATURES,
    "numeric_predictors": NUMERIC,
    "categorical_predictors": CATEGORICAL,
    "state_included": True,
    "smote_method": "SMOTENC",
    "smote_sampling_strategy": 0.5,
    "smote_k_neighbors": 5,
    "test_set_resampled": False,
    "cross_validation_folds": 5,
    "random_state": 42,
    "classification_threshold": 0.50,
    "saved_models": saved_models,
    "results": results,
}

summary_path = REPORT_DIR / "imbalance_experiment_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("IMBALANCE EXPERIMENT RESULTS")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {results_path}")
print(f"Summary saved: {summary_path}")
print(f"Models saved in: {MODEL_DIR}")
print("\nScript 07 completed successfully.")
