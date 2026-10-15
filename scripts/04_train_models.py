
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
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, confusion_matrix
)

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/jaundice_hepatitis_model_data.csv"
MODELS = ROOT / "outputs/models"
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

for folder in [MODELS, TABLES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

TARGET = "jaundice_hepatitis"
NUMERIC = ["dm005"]
CATEGORICAL = [
    "dm003", "dm013", "dm010", "living_arrangements",
    "mpce_quintile", "hc102", "residence"
]
FEATURES = NUMERIC + CATEGORICAL

if df.isna().any().any():
    raise ValueError("Missing values found. Check processed dataset first.")

X, y = df[FEATURES], df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), NUMERIC),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL)
])

# Class imbalance ratio calculated using training data only
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
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
        n_jobs=-1
    )
}

results = []

for name, estimator in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, prediction, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    })

    joblib.dump(model, MODELS / f"{name}.joblib")
    print(f"Completed: {name}")

results_df = pd.DataFrame(results)
results_df.to_csv(TABLES / "baseline_model_comparison.csv", index=False)

with open(REPORTS / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_rows": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_cases": int(y.sum()),
        "scale_pos_weight_xgboost": float(scale_pos_weight),
        "random_state": 42,
        "classification_threshold": 0.5
    }, f, indent=2)

print("\nBASELINE MODEL RESULTS")
print(results_df.round(4).to_string(index=False))
print(f"\nResults saved: {TABLES / 'baseline_model_comparison.csv'}")
print(f"Models saved in: {MODELS}")
