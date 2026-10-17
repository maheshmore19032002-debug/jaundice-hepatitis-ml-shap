"""
Script 09: Variable-specific SHAP analysis.
Separate from Script 08 (overall/global SHAP).

Focus variables: State/UT, Caste, Residence, MPCE quintile, and Age.
Uses the same processed data, tuned Random Forest pipeline, and stratified
80:20 split (random_state=42). SHAP explains model predictions, not causation.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- PATHS / SETTINGS --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_tuned.joblib"
METADATA_PATH = ROOT / "outputs" / "reports" / "variable_metadata.csv"
VALUE_LABELS_PATH = ROOT / "outputs" / "reports" / "variable_value_labels.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "shap" / "variable_specific"
TABLE_DIR = ROOT / "outputs" / "tables" / "shap_variable_specific"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in (FIG_DIR, TABLE_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "jaundice_hepatitis"
NUMERIC = ["dm005"]
CATEGORICAL = [
    "dm003", "dm013", "dm010", "living_arrangements",
    "mpce_quintile", "hc102", "residence", "state",
]
FEATURES = NUMERIC + CATEGORICAL
FOCUS_CATEGORICAL = ["state", "dm013", "residence", "mpce_quintile"]
AGE_VARIABLE = "dm005"

TEAL = "#159D91"
CORAL = "#E76F51"
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})


def norm(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    except (ValueError, TypeError):
        return text


def text(value):
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none"} else value


# -------------------- LOAD LABEL METADATA --------------------
for path in (DATA_PATH, MODEL_PATH, METADATA_PATH, VALUE_LABELS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

metadata_df = pd.read_csv(METADATA_PATH).fillna("")
labels_df = pd.read_csv(VALUE_LABELS_PATH).fillna("")
metadata_labels = {
    text(r["variable_name"]): text(r["variable_label"])
    for _, r in metadata_df.iterrows()
    if text(r["variable_name"])
}
VALUE_LABELS = {}
for _, r in labels_df.iterrows():
    variable, code, label = text(r["variable_name"]).lower(), norm(r["category_code"]), text(r["category_label"])
    if variable and code and label:
        VALUE_LABELS.setdefault(variable, {})[code] = label

# LASI source metadata stores state labels under stateid; model uses state.
if "stateid" in VALUE_LABELS:
    VALUE_LABELS["state"] = VALUE_LABELS["stateid"].copy()

FALLBACK = {
    "dm005": "Age (years)", "dm013": "Caste / social group",
    "residence": "Place of residence",
    "mpce_quintile": "Household consumption expenditure quintile",
    "state": "State / union territory",
}


def variable_label(variable):
    return metadata_labels.get(variable, "") or FALLBACK.get(variable, variable)


def category_label(variable, value):
    key = norm(value)
    return VALUE_LABELS.get(variable, {}).get(key, f"Code {key} (label unavailable)")


# -------------------- DATA / MODEL / FIXED TEST SPLIT --------------------
df = pd.read_csv(DATA_PATH)
missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns in processed dataset: {missing}")

df = df[FEATURES + [TARGET]].copy()
for col in CATEGORICAL:
    df[col] = df[col].astype(str).str.strip()

X = df[FEATURES].copy()
y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = joblib.load(MODEL_PATH)
if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
    raise TypeError("Expected a saved sklearn pipeline with a 'preprocessor' step.")
classifier_step = "classifier" if "classifier" in model.named_steps else (
    "model" if "model" in model.named_steps else None
)
if classifier_step is None:
    raise ValueError(f"Cannot identify classifier step: {list(model.named_steps)}")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps[classifier_step]
X_sample, y_sample = X_test.copy(), y_test.copy()
probabilities = model.predict_proba(X_sample)[:, 1]

performance = {
    "model_path": str(MODEL_PATH),
    "classifier_step": classifier_step,
    "test_observations": int(len(X_sample)),
    "test_positive_count": int(y_sample.sum()),
    "test_positive_prevalence": float(y_sample.mean()),
    "test_average_precision": float(average_precision_score(y_sample, probabilities)),
    "test_roc_auc": float(roc_auc_score(y_sample, probabilities)),
    "mean_predicted_probability": float(probabilities.mean()),
}
print("=" * 68)
print("SCRIPT 09: VARIABLE-SPECIFIC SHAP")
print(json.dumps(performance, indent=2))
print(f"State labels available: {len(VALUE_LABELS.get('state', {}))}")

# -------------------- SHAP VALUES --------------------
print("\nTransforming data and calculating SHAP values...", flush=True)
Xt = preprocessor.transform(X_sample)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
Xt = np.asarray(Xt)
feature_names = list(preprocessor.get_feature_names_out())
if Xt.shape[1] != len(feature_names):
    raise ValueError("Transformed data and feature names do not match.")

explainer = shap.TreeExplainer(classifier)
raw = explainer.shap_values(Xt)
sv = np.asarray(raw[1] if isinstance(raw, list) and len(raw) > 1 else
                raw[0] if isinstance(raw, list) else raw)
if sv.ndim == 3:
    if sv.shape[-1] == 2:
        sv = sv[:, :, 1]
    elif sv.shape[1] == 2:
        sv = sv[:, 1, :]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")
if sv.shape != Xt.shape:
    raise ValueError(f"SHAP shape {sv.shape} != transformed data shape {Xt.shape}")


# -------------------- MAP ENCODED FEATURES TO SOURCE VARIABLES --------------------
def source_variable(encoded_name):
    name = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    for var in sorted(CATEGORICAL, key=len, reverse=True):
        if name == var or name.startswith(var + "_"):
            return var
    if name == "dm005" or name.startswith("dm005_"):
        return "dm005"
    return name


feature_source = [source_variable(n) for n in feature_names]
grouped = {}
for var in FEATURES:
    idx = [i for i, source in enumerate(feature_source) if source == var]
    if idx:
        grouped[var] = sv[:, idx].sum(axis=1)

missing_groups = sorted(set(FEATURES) - set(grouped))
if missing_groups:
    raise ValueError(
        f"Could not map encoded SHAP features to: {missing_groups}. "
        f"Inspect preprocessor feature names: {feature_names[:15]}"
    )

# -------------------- CATEGORY-SPECIFIC TABLES AND PLOTS --------------------
all_summaries = []
for var in FOCUS_CATEGORICAL:
    category_values = X_sample[var].astype(str).map(norm).to_numpy()
    contrib = grouped[var]
    rows = []
    unique_codes = sorted(set(category_values), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    for code in unique_codes:
        mask = category_values == code
        n = int(mask.sum())
        positive_n = int(y_sample.to_numpy()[mask].sum())
        rows.append({
            "variable": var,
            "variable_label": variable_label(var),
            "category_code": code,
            "category_label": category_label(var, code),
            "n_test": n,
            "positive_cases": positive_n,
            "observed_positive_prevalence": positive_n / n if n else np.nan,
            "mean_predicted_probability": float(probabilities[mask].mean()) if n else np.nan,
            "mean_grouped_shap": float(contrib[mask].mean()) if n else np.nan,
            "mean_abs_grouped_shap": float(np.abs(contrib[mask]).mean()) if n else np.nan,
        })

    out = pd.DataFrame(rows)
    safe = {"dm013": "caste", "mpce_quintile": "mpce_quintile"}.get(var, var)
    out.to_csv(TABLE_DIR / f"shap_category_summary_{safe}.csv",
               index=False, encoding="utf-8-sig")
    all_summaries.append(out)

    # SHAP magnitude by category; state figure is limited to top 20 for readability.
    plot_df = out.sort_values("mean_abs_grouped_shap", ascending=False)
    suffix = ""
    if var == "state" and len(plot_df) > 20:
        plot_df = plot_df.head(20)
        suffix = " (top 20)"
    plot_df = plot_df.sort_values("mean_abs_grouped_shap")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(plot_df) + 1.2)))
    bars = ax.barh(plot_df["category_label"], plot_df["mean_abs_grouped_shap"], color=TEAL)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean absolute grouped SHAP value")
    ax.set_ylabel(variable_label(var))
    ax.set_title(f"Category-Specific SHAP: {variable_label(var)}{suffix}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"shap_{safe}_category_importance.png", bbox_inches="tight")
    plt.close(fig)

    # Descriptive observed prevalence by category.
    prev = out.sort_values("observed_positive_prevalence")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(prev) + 1.2)))
    bars = ax.barh(prev["category_label"], prev["observed_positive_prevalence"] * 100, color=CORAL)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xlabel("Observed positive prevalence in held-out test set (%)")
    ax.set_ylabel(variable_label(var))
    ax.set_title(f"Observed Prevalence by {variable_label(var)}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{safe}_observed_prevalence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved category summary: {safe}")


pd.concat(all_summaries, ignore_index=True).to_csv(
    TABLE_DIR / "shap_category_summaries_all_focus_variables.csv",
    index=False, encoding="utf-8-sig"
)

# -------------------- AGE-SPECIFIC ANALYSIS --------------------
age_df = pd.DataFrame({
    "age": pd.to_numeric(X_sample[AGE_VARIABLE], errors="coerce").to_numpy(),
    "age_grouped_shap": grouped[AGE_VARIABLE],
    "observed_outcome": y_sample.to_numpy(),
    "predicted_probability": probabilities,
}).dropna(subset=["age", "age_grouped_shap"])
age_df.to_csv(TABLE_DIR / "shap_age_observation_level.csv",
              index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(age_df["age"], age_df["age_grouped_shap"],
           s=14, alpha=0.22, color=TEAL, edgecolors="none")
ax.axhline(0, color=CORAL, linestyle="--", linewidth=1.2)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Grouped SHAP value for age")
ax.set_title("Age Dependence: Age vs Grouped SHAP Contribution")
fig.tight_layout()
fig.savefig(FIG_DIR / "shap_age_dependence.png", bbox_inches="tight")
plt.close(fig)

try:
    age_df["age_group"] = pd.qcut(age_df["age"], q=8, duplicates="drop")
    age_summary = age_df.groupby("age_group", observed=False).agg(
        n=("age", "size"),
        mean_age=("age", "mean"),
        mean_grouped_shap=("age_grouped_shap", "mean"),
        mean_abs_grouped_shap=("age_grouped_shap", lambda s: float(np.abs(s).mean())),
        positive_cases=("observed_outcome", "sum"),
        mean_predicted_probability=("predicted_probability", "mean"),
    ).reset_index()
    age_summary["observed_positive_prevalence"] = age_summary["positive_cases"] / age_summary["n"]
    age_summary["age_group"] = age_summary["age_group"].astype(str)
    age_summary.to_csv(TABLE_DIR / "shap_age_group_summary.csv",
                       index=False, encoding="utf-8-sig")
except ValueError as exc:
    print(f"Age-group summary skipped: {exc}")

# -------------------- SAVE SUMMARY --------------------
summary = {
    **performance,
    "analysis_type": "variable_specific_shap",
    "variables_analyzed": FOCUS_CATEGORICAL + [AGE_VARIABLE],
    "shap_method": "TreeExplainer",
    "shap_shape": list(sv.shape),
    "tables_directory": str(TABLE_DIR),
    "figures_directory": str(FIG_DIR),
    "interpretation_note": (
        "SHAP explains model predictions, not causal effects. Category prevalence is "
        "descriptive. Mean predicted probability is not calibrated risk unless calibration "
        "has been assessed. Grouped SHAP sums encoded contributions for each source variable."
    ),
}
summary_path = REPORT_DIR / "shap_variable_specific_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print("\n" + "=" * 68)
print("VARIABLE-SPECIFIC SHAP ANALYSIS COMPLETED")
print(f"Figures: {FIG_DIR}")
print(f"Tables: {TABLE_DIR}")
print(f"Summary: {summary_path}")
print("\nNote: SHAP values are model explanations, not causal effects.")
