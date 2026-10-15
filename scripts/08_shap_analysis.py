
"""
Script 08: LASI SHAP analysis with metadata-based category labels.

Requires:
    data/processed/jaundice_hepatitis_model_data.csv
    outputs/models/random_forest_tuned.joblib
    outputs/reports/variable_metadata.csv
    outputs/reports/variable_value_labels.csv

Saves:
    Global and encoded feature importance
    SHAP beeswarm and bar plots
    Age dependence plot
    Caste category analysis
    Caste prevalence and prediction plots
    Local waterfall plot
    SHAP heatmap
    CSV tables and JSON report
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_tuned.joblib"
METADATA_PATH = ROOT / "outputs" / "reports" / "variable_metadata.csv"
VALUE_LABELS_PATH = ROOT / "outputs" / "reports" / "variable_value_labels.csv"

FIG_DIR = ROOT / "outputs" / "figures" / "shap"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [FIG_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

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

# Consistent publication-friendly palette
NAVY = "#23395B"
TEAL = "#159D91"
CORAL = "#E76F51"
PURPLE = "#8172B3"
GREY = "#687385"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
# 2. LOAD LASI METADATA AND VALUE LABELS
# =========================================================

for path in [DATA_PATH, MODEL_PATH, METADATA_PATH, VALUE_LABELS_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Run Script 01 first to create the metadata and value-label files."
        )

metadata_df = pd.read_csv(METADATA_PATH).fillna("")
value_labels_df = pd.read_csv(VALUE_LABELS_PATH).fillna("")

required_meta_cols = {"variable_name", "variable_label"}
if not required_meta_cols.issubset(metadata_df.columns):
    raise ValueError(
        "variable_metadata.csv must contain variable_name and variable_label."
    )

required_value_cols = {
    "variable_name", "category_code", "category_label"
}
if not required_value_cols.issubset(value_labels_df.columns):
    raise ValueError(
        "variable_value_labels.csv must contain variable_name, "
        "category_code and category_label."
    )

metadata_labels = {
    str(row["variable_name"]).strip(): str(row["variable_label"]).strip()
    for _, row in metadata_df.iterrows()
}


def normalize_code(value):
    """Normalize values such as 2, 2.0 and '2.0' to a common key."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    except (ValueError, TypeError):
        return text


# variable -> normalized category code -> actual metadata label
VALUE_LABELS = {}

for _, row in value_labels_df.iterrows():
    variable = str(row["variable_name"]).strip()
    code = normalize_code(row["category_code"])
    label = str(row["category_label"]).strip()

    if variable and code and label:
        VALUE_LABELS.setdefault(variable, {})[code] = label


FALLBACK_LABELS = {
    "dm005": "Age (years)",
    "dm003": "Sex",
    "dm013": "Caste / social group",
    "dm010": "Religion",
    "living_arrangements": "Living arrangement",
    "mpce_quintile": "Household consumption expenditure quintile",
    "hc102": "Health insurance coverage",
    "residence": "Place of residence",
    TARGET: "Self-reported jaundice / hepatitis in the past two years",
}


def display_label(variable):
    """Use original LASI variable label, with fallback only if blank."""
    label = metadata_labels.get(variable, "").strip()

    if label and label.lower() not in {"nan", "none"}:
        return label

    return FALLBACK_LABELS.get(variable, variable)


def category_label(variable, code):
    """
    Use the actual category label exported from LASI metadata.
    Never silently invent a category label.
    """
    key = normalize_code(code)
    label = VALUE_LABELS.get(variable, {}).get(key)

    if label:
        return label

    return f"Code {key} (label unavailable)"


print("\nVARIABLE LABELS")
for variable in FEATURES + [TARGET]:
    print(f"{variable}: {display_label(variable)}")

print("\nCATEGORY LABEL AVAILABILITY")
for variable in CATEGORICAL:
    labels = VALUE_LABELS.get(variable, {})
    print(f"{variable}: {len(labels)} category labels found")


# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv(DATA_PATH)

missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
if missing:
    raise ValueError(f"Required columns missing from processed dataset: {missing}")

df = df[FEATURES + [TARGET]].copy()
X = df[FEATURES]
y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1.")

# Reproduce the split used in the earlier modeling scripts.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = joblib.load(MODEL_PATH)

if not hasattr(model, "named_steps"):
    raise TypeError("Expected a saved sklearn Pipeline.")

if "preprocessor" not in model.named_steps:
    raise ValueError("Pipeline has no 'preprocessor' step.")

if "classifier" not in model.named_steps:
    raise ValueError("Pipeline has no 'classifier' step.")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Reproducible sample of at most 1,000 test participants.
rng = np.random.default_rng(42)
sample_size = min(1000, len(X_test))
sample_idx = np.sort(
    rng.choice(len(X_test), size=sample_size, replace=False)
)

X_sample = X_test.iloc[sample_idx].copy()
y_sample = y_test.iloc[sample_idx].copy()

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)
feature_names = list(preprocessor.get_feature_names_out())

if X_transformed.shape[1] != len(feature_names):
    raise ValueError("Transformed feature names do not match the data matrix.")

probabilities = model.predict_proba(X_sample)[:, 1]

performance = {
    "model": "random_forest_tuned",
    "sample_size": int(sample_size),
    "sample_positive_count": int(y_sample.sum()),
    "sample_positive_prevalence": float(y_sample.mean()),
    "sample_average_precision": float(
        average_precision_score(y_sample, probabilities)
    ),
    "sample_roc_auc": float(roc_auc_score(y_sample, probabilities)),
    "mean_predicted_probability": float(probabilities.mean()),
}

print("\nSHAP SAMPLE PERFORMANCE")
print(json.dumps(performance, indent=2))


# =========================================================
# 4. SHAP VALUES
# =========================================================

print("\nCalculating SHAP values...", flush=True)

explainer = shap.TreeExplainer(classifier)
raw_shap = explainer.shap_values(X_transformed)

if isinstance(raw_shap, list):
    shap_values = np.asarray(
        raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    )
else:
    shap_values = np.asarray(raw_shap)

# Compatibility with SHAP versions returning 3D arrays.
if shap_values.ndim == 3:
    if shap_values.shape[-1] == 2:
        shap_values = shap_values[:, :, 1]
    elif shap_values.shape[1] == 2:
        shap_values = shap_values[:, 1, :]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

if shap_values.shape != X_transformed.shape:
    raise ValueError(
        f"SHAP shape {shap_values.shape} does not match "
        f"transformed data {X_transformed.shape}."
    )

expected = np.asarray(explainer.expected_value).reshape(-1)
base_value = float(expected[1] if len(expected) > 1 else expected[0])

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.repeat(base_value, len(X_sample)),
    data=X_transformed,
    feature_names=feature_names,
)


# =========================================================
# 5. MAP ENCODED FEATURES TO SOURCE VARIABLES
# =========================================================

def original_variable(transformed_name):
    name = transformed_name

    for prefix in ["numeric__", "categorical__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Longest names first to avoid accidental partial matches.
    for variable in sorted(CATEGORICAL, key=len, reverse=True):
        if name == variable or name.startswith(variable + "_"):
            return variable

    if name == "dm005" or name.startswith("dm005_"):
        return "dm005"

    return name


feature_map = {
    feature: original_variable(feature)
    for feature in feature_names
}


def category_suffix(feature, variable):
    """Extract one-hot category code from a transformed column name."""
    name = feature

    for prefix in ["numeric__", "categorical__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    prefix = variable + "_"
    if name.startswith(prefix):
        return name[len(prefix):]

    return None



def readable_encoded_feature(feature):
    """Show category names without numerical codes."""
    variable = feature_map[feature]
    variable_name = display_label(variable)
    suffix = category_suffix(feature, variable)

    if variable in CATEGORICAL and suffix is not None:
        actual_category = category_label(variable, suffix)

        # Remove leading numerical code, e.g. "2 Hindu" -> "Hindu"
        actual_category = re.sub(
            r"^\s*\d+(?:\.0)?\s*[:.\-]?\s*",
            "",
            actual_category
        ).strip()

        return f"{variable_name}: {actual_category}"

    return variable_name



readable_names = [
    readable_encoded_feature(feature)
    for feature in feature_names
]


# =========================================================
# 6. ENCODED FEATURE IMPORTANCE TABLE
# =========================================================

encoded_importance = pd.DataFrame({
    "encoded_feature": feature_names,
    "display_label": readable_names,
    "original_variable": [feature_map[f] for f in feature_names],
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    "mean_shap": shap_values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

encoded_importance.to_csv(
    TABLE_DIR / "shap_encoded_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 7. GROUPED IMPORTANCE BY ORIGINAL PREDICTOR
# =========================================================

grouped_values = {}

for variable in FEATURES:
    indices = [
        i for i, feature in enumerate(feature_names)
        if feature_map[feature] == variable
    ]

    if indices:
        grouped_values[variable] = shap_values[:, indices].sum(axis=1)

group_importance = pd.DataFrame([
    {
        "variable": variable,
        "display_label": display_label(variable),
        "mean_abs_grouped_shap": float(np.abs(values).mean()),
        "mean_grouped_shap": float(values.mean()),
    }
    for variable, values in grouped_values.items()
]).sort_values("mean_abs_grouped_shap", ascending=False)

group_importance.to_csv(
    TABLE_DIR / "shap_global_variable_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(11, 7))
plot_df = group_importance.sort_values("mean_abs_grouped_shap")

bar_colors = [
    CORAL if variable == "dm013" else TEAL
    for variable in plot_df["variable"]
]

bars = ax.barh(
    plot_df["display_label"],
    plot_df["mean_abs_grouped_shap"],
    color=bar_colors,
)

ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax.set_xlabel("Mean absolute grouped SHAP value")
ax.set_ylabel("Predictor")
ax.set_title("Global Feature Importance — Tuned Random Forest")
ax.invert_yaxis()
fig.tight_layout()
output_file = (
    ROOT / "outputs" / "figures" / "shap"
    / "01_global_feature_importance.png"
).resolve()

output_file.parent.mkdir(parents=True, exist_ok=True)

print("ROOT:", repr(str(ROOT)))
print("Output file:", repr(str(output_file)))

fig.canvas.draw()
fig.savefig(
    str(output_file),
    format="png",
    dpi=300,
    bbox_inches="tight",
)

print("Graph saved:", output_file)
plt.close(fig)

output_file = FIG_DIR / "01_global_feature_importance.png"

print("Saving figure to:", repr(str(output_file)))

fig.savefig(
    str(output_file),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

print("Figure saved successfully.")
# =========================================================
# 8. SHAP BEESWARM
# =========================================================

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=readable_names,
    max_display=20,
    show=False,
    plot_size=None,
    color=plt.get_cmap("coolwarm"),
)

plt.title("SHAP Beeswarm — Encoded Predictors", pad=14)
plt.tight_layout()
fig = plt.gcf()

output_file = (
    ROOT / "outputs" / "figures" / "shap"
    / "02_shap_beeswarm.png"
).resolve()

fig.savefig(
    str(output_file),
    format="png",
    dpi=300,
    bbox_inches="tight",
)

print("Saved:", output_file)
plt.close(fig)


# =========================================================
# 9. SHAP ENCODED BAR PLOT
# =========================================================

plt.figure(figsize=(11, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=readable_names,
    plot_type="bar",
    max_display=20,
    show=False,
    plot_size=None,
    color=TEAL,
)

plt.title("Mean Absolute SHAP — Encoded Predictors", pad=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_shap_encoded_bar.png", bbox_inches="tight", dpi=300)
plt.close("all")


# =========================================================
# 10. AGE DEPENDENCE PLOT
# =========================================================

age_indices = [
    i for i, feature in enumerate(feature_names)
    if feature_map[feature] == "dm005"
]

if age_indices:
    age_idx = age_indices[0]

    age_df = pd.DataFrame({
        "age": X_sample["dm005"].to_numpy(),
        "age_shap_value": shap_values[:, age_idx],
    })

    age_df.to_csv(
        TABLE_DIR / "shap_age_dependence_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    points = ax.scatter(
        age_df["age"],
        age_df["age_shap_value"],
        c=age_df["age_shap_value"],
        cmap="coolwarm",
        alpha=0.75,
        s=38,
        edgecolors="white",
        linewidths=0.3,
    )

    ax.axhline(0, color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel(display_label("dm005"))
    ax.set_ylabel("SHAP value for age")
    ax.set_title("Age Dependence Plot")
    fig.colorbar(points, ax=ax, label="SHAP value")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_age_dependence.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 11. DYNAMIC CASTE CATEGORY ANALYSIS
# =========================================================

caste_variable = "dm013"
caste_value_labels = VALUE_LABELS.get(caste_variable, {})

# Derive categories from observed data and actual metadata value labels.
observed_caste_codes = {
    normalize_code(v)
    for v in pd.to_numeric(X_sample[caste_variable], errors="coerce").dropna()
}

all_caste_codes = sorted(
    observed_caste_codes | set(caste_value_labels.keys()),
    key=lambda x: (not x.replace(".", "", 1).isdigit(),
                   float(x) if x.replace(".", "", 1).isdigit() else x),
)

caste_rows = []

for code in all_caste_codes:
    category_name = category_label(caste_variable, code)
    numeric_code = float(code) if code.replace(".", "", 1).isdigit() else np.nan

    mask = (
        pd.to_numeric(X_sample[caste_variable], errors="coerce")
        .map(normalize_code)
        .eq(code)
        .to_numpy()
    )

    # Find the one-hot column for this caste code.
    caste_feature_indices = []

    for i, feature in enumerate(feature_names):
        if feature_map[feature] != caste_variable:
            continue

        suffix = category_suffix(feature, caste_variable)
        if suffix is not None and normalize_code(suffix) == code:
            caste_feature_indices.append(i)

    if not mask.any():
        caste_rows.append({
            "caste_code": code,
            "caste": category_name,
            "sample_n": 0,
            "observed_positive_n": 0,
            "observed_positive_prevalence": np.nan,
            "mean_predicted_probability": np.nan,
            "mean_signed_category_shap": np.nan,
            "mean_abs_category_shap": np.nan,
            "note": "Category absent from SHAP sample",
        })
        continue

    if not caste_feature_indices:
        caste_shap_mean = np.nan
        caste_shap_abs = np.nan
        note = "No matching one-hot column; check encoder and source codes"
    else:
        # SHAP contribution of this caste indicator among members of that category.
        category_shap = shap_values[mask][:, caste_feature_indices].sum(axis=1)
        caste_shap_mean = float(category_shap.mean())
        caste_shap_abs = float(np.abs(category_shap).mean())
        note = "Descriptive model explanation; not a causal effect"

    caste_rows.append({
        "caste_code": code,
        "caste": category_name,
        "sample_n": int(mask.sum()),
        "observed_positive_n": int(y_sample.to_numpy()[mask].sum()),
        "observed_positive_prevalence": float(y_sample.to_numpy()[mask].mean()),
        "mean_predicted_probability": float(probabilities[mask].mean()),
        "mean_signed_category_shap": caste_shap_mean,
        "mean_abs_category_shap": caste_shap_abs,
        "note": note,
    })

caste_results = pd.DataFrame(caste_rows)

caste_results.to_csv(
    TABLE_DIR / "shap_caste_category_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 12. CASTE OBSERVED VS PREDICTED PLOT
# =========================================================

valid_caste = caste_results[caste_results["sample_n"] > 0].copy()

if not valid_caste.empty:
    x = np.arange(len(valid_caste))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(
        x - width / 2,
        valid_caste["observed_positive_prevalence"] * 100,
        width,
        label="Observed positive prevalence",
        color=TEAL,
    )

    b2 = ax.bar(
        x + width / 2,
        valid_caste["mean_predicted_probability"] * 100,
        width,
        label="Mean predicted probability",
        color=CORAL,
    )

    ax.bar_label(b1, fmt="%.2f%%", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.2f%%", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_caste["caste"], rotation=20, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_title("Caste Categories: Observed vs Predicted")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "05_caste_observed_vs_predicted.png",
        bbox_inches="tight",
    )
    plt.close(fig)


# =========================================================
# 13. CASTE-SPECIFIC SHAP CONTRIBUTION PLOT
# =========================================================

valid_caste_shap = caste_results.dropna(
    subset=["mean_signed_category_shap"]
).copy()

if not valid_caste_shap.empty:
    colors = [
        TEAL if value >= 0 else CORAL
        for value in valid_caste_shap["mean_signed_category_shap"]
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        valid_caste_shap["caste"],
        valid_caste_shap["mean_signed_category_shap"],
        color=colors,
    )

    ax.axhline(0, color=GREY, linewidth=1)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_ylabel("Mean signed category SHAP value")
    ax.set_xlabel(display_label(caste_variable))
    ax.set_title("Caste-Specific SHAP Contributions")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "06_caste_category_shap.png",
        bbox_inches="tight",
    )
    plt.close(fig)


# =========================================================
# 14. FULL TEST-SET CASTE SUMMARY
# =========================================================

caste_test = pd.DataFrame({
    "caste_code": pd.to_numeric(X_test[caste_variable], errors="coerce"),
    "actual_outcome": y_test.to_numpy(),
})

caste_summary = (
    caste_test.groupby("caste_code", dropna=False)
    .agg(
        n=("actual_outcome", "size"),
        positive_n=("actual_outcome", "sum"),
        positive_prevalence=("actual_outcome", "mean"),
    )
    .reset_index()
)

caste_summary["caste"] = caste_summary["caste_code"].map(
    lambda v: category_label(caste_variable, v)
)

caste_summary.to_csv(
    TABLE_DIR / "caste_test_set_outcome_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 15. LOCAL WATERFALL WITH READABLE LABELS
# =========================================================

top_idx = int(np.argmax(probabilities))

local_explanation = shap.Explanation(
    values=shap_values[top_idx],
    base_values=base_value,
    data=X_transformed[top_idx],
    feature_names=readable_names,
)

plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    local_explanation,
    max_display=15,
    show=False,
)

plt.title(
    "Local SHAP Explanation — Highest Predicted Probability "
    f"({probabilities[top_idx]:.3f})",
    pad=12,
)

plt.tight_layout()
plt.savefig(
    FIG_DIR / "07_local_waterfall_highest_risk.png",
    bbox_inches="tight",
)
plt.close("all")

local_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "display_label": readable_names,
    "transformed_value": X_transformed[top_idx],
    "shap_value": shap_values[top_idx],
    "absolute_shap_value": np.abs(shap_values[top_idx]),
}).sort_values("absolute_shap_value", ascending=False)

local_df.to_csv(
    TABLE_DIR / "shap_local_highest_probability.csv",
    index=False,
    encoding="utf-8-sig",
)

local_record = X_sample.iloc[top_idx].to_dict()
local_record["predicted_probability"] = float(probabilities[top_idx])
local_record["actual_outcome"] = int(y_sample.iloc[top_idx])

with open(
    REPORT_DIR / "shap_local_highest_probability.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(local_record, f, indent=2, default=str)


# =========================================================
# 16. SHAP HEATMAP WITH READABLE LABELS
# =========================================================

top_feature_idx = np.argsort(
    np.abs(shap_values).mean(axis=0)
)[-15:]

n_heat = min(100, len(shap_values))
heat = shap_values[:n_heat, :][:, top_feature_idx]
heat_names = [readable_names[i] for i in top_feature_idx]

fig, ax = plt.subplots(figsize=(13, 8))

sns.heatmap(
    heat.T,
    cmap="vlag",
    center=0,
    xticklabels=False,
    yticklabels=heat_names,
    cbar_kws={"label": "SHAP value"},
    ax=ax,
)

ax.set_xlabel(f"Participants in SHAP sample (first {n_heat})")
ax.set_ylabel("Predictor / category")
ax.set_title("SHAP Contribution Heatmap")

fig.tight_layout()
fig.savefig(FIG_DIR / "08_shap_heatmap.png", bbox_inches="tight")
plt.close(fig)


# =========================================================
# 17. SAVE DISPLAY LABEL AUDIT
# =========================================================

label_audit = pd.DataFrame([
    {
        "variable_name": variable,
        "metadata_label": metadata_labels.get(variable, ""),
        "display_label_used": display_label(variable),
        "category_labels_available": len(VALUE_LABELS.get(variable, {})),
    }
    for variable in FEATURES + [TARGET]
])

label_audit.to_csv(
    TABLE_DIR / "shap_display_label_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

# Export category labels used for all study categorical variables.
category_audit_rows = []

for variable in CATEGORICAL:
    for code, label in VALUE_LABELS.get(variable, {}).items():
        category_audit_rows.append({
            "variable_name": variable,
            "variable_label": display_label(variable),
            "category_code": code,
            "category_label": label,
        })

pd.DataFrame(
    category_audit_rows,
    columns=[
        "variable_name",
        "variable_label",
        "category_code",
        "category_label",
    ],
).to_csv(
    TABLE_DIR / "shap_category_label_audit.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 18. SAVE SUMMARY
# =========================================================

summary = {
    **performance,
    "model_path": str(MODEL_PATH),
    "metadata_path": str(METADATA_PATH),
    "value_labels_path": str(VALUE_LABELS_PATH),
    "shap_method": "TreeExplainer",
    "shap_output_shape": list(shap_values.shape),
    "base_value": base_value,
    "global_importance_file": str(
        TABLE_DIR / "shap_global_variable_importance.csv"
    ),
    "encoded_importance_file": str(
        TABLE_DIR / "shap_encoded_feature_importance.csv"
    ),
    "caste_analysis_file": str(
        TABLE_DIR / "shap_caste_category_analysis.csv"
    ),
    "display_label_audit_file": str(
        TABLE_DIR / "shap_display_label_audit.csv"
    ),
    "figures_directory": str(FIG_DIR),
    "interpretation_note": (
        "SHAP explains model predictions, not causal effects. "
        "Subgroup estimates are descriptive and may be unstable when "
        "sample sizes are small. Category labels are sourced from the "
        "saved LASI value-label metadata where available."
    ),
}

with open(
    REPORT_DIR / "shap_analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


# =========================================================
# 19. PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)

print("\nGlobal variable importance:")
print(group_importance.round(5).to_string(index=False))

print("\nCaste category analysis:")
print(caste_results.round(5).to_string(index=False))

print("\nCategory labels used:")
print(
    pd.DataFrame(category_audit_rows).to_string(index=False)
    if category_audit_rows
    else "No categorical value labels were found in saved metadata."
)

print(f"\nFigures saved: {FIG_DIR}")
print(f"Tables saved: {TABLE_DIR}")
print(f"Summary saved: {REPORT_DIR / 'shap_analysis_summary.json'}")
