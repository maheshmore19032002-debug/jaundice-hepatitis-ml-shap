
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
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, confusion_matrix
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [MODEL_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "jaundice_hepatitis"
NUMERIC = ["dm005"]
CATEGORICAL = [
    "dm003", "dm013", "dm010",
    "living_arrangements", "mpce_quintile",
    "hc102", "residence"
]
FEATURES = NUMERIC + CATEGORICAL

df = pd.read_csv(DATA)
required = FEATURES + [TARGET]

if not set(required).issubset(df.columns):
    raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

df = df[required].copy()
X = df[FEATURES]
y = df[TARGET].astype(int)

if y.isna().any() or not set(y.unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1 without missing values.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Impute before SMOTENC, keeping categorical values as categories.
impute_for_smote = ColumnTransformer([
    ("age", SimpleImputer(strategy="median"), NUMERIC),
    ("categories", SimpleImputer(strategy="most_frequent"), CATEGORICAL),
], verbose_feature_names_out=False)

# Column order after imputation: age first, then seven categorical variables.
categorical_indices = list(range(1, len(FEATURES)))

smote = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=0.5,
    random_state=42,
    k_neighbors=5
)

# Encode categorical variables only after resampling.
preprocessor = ColumnTransformer([
    ("numeric", SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), [0]),
    ("categorical", SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_indices)
])

# Use parameters found in Script 06.
models = {
    "logistic_regression": {
        "estimator": LogisticRegression(
            C=0.01, class_weight="balanced",
            max_iter=3000, random_state=42
        ),
        "smote_estimator": LogisticRegression(
            C=0.01, class_weight=None,
            max_iter=3000, random_state=42
        )
    },
    "random_forest": {
        "estimator": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20,
            max_features=0.5, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=1
        ),
        "smote_estimator": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20,
            max_features=0.5, max_depth=10,
            class_weight=None, random_state=42, n_jobs=1
        )
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
saved_models = {}

for model_name, configs in models.items():
    for method in ["class_weight", "smotenc"]:
        print(f"\nExperiment: {model_name} | {method}", flush=True)

        steps = [
            ("impute_for_smote", clone(impute_for_smote))
        ]

        if method == "smotenc":
            steps.append(("sampler", clone(smote)))
        else:
            steps.append(("sampler", "passthrough"))

        steps.extend([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(
                configs["estimator"] if method == "class_weight"
                else configs["smote_estimator"]
            ))
        ])

        pipeline = ImbPipeline(steps)

        # All preprocessing and resampling occur within CV training folds.
        cv_results = cross_validate(
            pipeline,
            X_train, y_train,
            cv=cv,
            scoring={
                "average_precision": "average_precision",
                "roc_auc": "roc_auc"
            },
            n_jobs=-1,
            error_score="raise"
        )

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        key = f"{model_name}_{method}"
        model_path = MODEL_DIR / f"{key}.joblib"
        joblib.dump(pipeline, model_path)
        saved_models[key] = str(model_path)

        results.append({
            "model": model_name,
            "method": method,
            "cv_average_precision_mean":
                float(cv_results["test_average_precision"].mean()),
            "cv_average_precision_sd":
                float(cv_results["test_average_precision"].std()),
            "cv_roc_auc_mean":
                float(cv_results["test_roc_auc"].mean()),
            "test_average_precision":
                float(average_precision_score(y_test, probabilities)),
            "test_roc_auc":
                float(roc_auc_score(y_test, probabilities)),
            "test_precision_at_050":
                float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall_at_050":
                float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1_at_050":
                float(f1_score(y_test, predictions, zero_division=0)),
            "test_balanced_accuracy_at_050":
                float(balanced_accuracy_score(y_test, predictions)),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        })

        print(
            f"CV AP: {cv_results['test_average_precision'].mean():.4f} "
            f"+/- {cv_results['test_average_precision'].std():.4f}",
            flush=True
        )
        print(
            f"Test AP: {average_precision_score(y_test, probabilities):.4f}",
            flush=True
        )

results_df = pd.DataFrame(results).sort_values(
    "cv_average_precision_mean", ascending=False
)

results_path = TABLE_DIR / "imbalance_experiment_comparison.csv"
results_df.to_csv(results_path, index=False)

summary_path = REPORT_DIR / "imbalance_experiment_summary.json"
summary_path.write_text(
    json.dumps({
        "primary_metric": "5-fold CV Average Precision",
        "smote_method": "SMOTENC",
        "smote_sampling_strategy": 0.5,
        "test_set_resampled": False,
        "saved_models": saved_models,
        "results": results
    }, indent=2),
    encoding="utf-8"
)

print("\nIMBALANCE EXPERIMENT RESULTS")
print(results_df.round(4).to_string(index=False))
print(f"\nResults saved: {results_path}")
print(f"Summary saved: {summary_path}")
