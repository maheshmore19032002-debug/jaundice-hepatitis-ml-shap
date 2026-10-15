
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("data/raw/data_working.dta")
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"
PROCESSED = ROOT / "data" / "processed"

for folder in [TABLES, REPORTS, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)

OUTCOME = "ht201"
PREDICTORS = [
    "dm005", "dm003", "dm013", "dm010",
    "living_arrangements", "mpce_quintile", "hc102", "residence"
]
VARIABLES = [OUTCOME] + PREDICTORS

# Read only required variables
_, meta = pyreadstat.read_dta(str(DATA), metadataonly=True)
missing_vars = sorted(set(VARIABLES) - set(meta.column_names))
if missing_vars:
    raise ValueError(f"Variables not found: {missing_vars}")

df, meta = pyreadstat.read_dta(str(DATA), usecols=VARIABLES)

# Convert study variables to numeric
for col in VARIABLES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Variable labels from the source metadata
labels = dict(zip(meta.column_names, meta.column_labels or [""] * len(meta.column_names)))

# 1. Overall missingness and raw frequency tables
audit = []
for col in VARIABLES:
    s = df[col]
    audit.append({
        "variable": col,
        "variable_label": labels.get(col, ""),
        "non_missing": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "missing_percent": round(s.isna().mean() * 100, 2),
        "unique_non_missing_values": int(s.nunique())
    })

    freq = s.value_counts(dropna=False).sort_index().rename_axis("value").reset_index(name="count")
    freq["percentage_of_all_rows"] = (freq["count"] / len(df) * 100).round(2)
    freq.insert(0, "variable", col)
    freq.insert(1, "variable_label", labels.get(col, ""))
    freq.to_csv(TABLES / f"frequency_{col}.csv", index=False)

pd.DataFrame(audit).to_csv(TABLES / "variable_missingness_audit.csv", index=False)

# 2. Basic validation of known coding
valid_codes = {
    "ht201": [1, 2],
    "dm003": [1, 2],
    "dm013": [1, 2, 3, 4, 5],
    "dm010": list(range(1, 11)),
    "living_arrangements": [1, 2, 3, 4, 5],
    "mpce_quintile": [1, 2, 3, 4, 5],
    "hc102": [1, 2],
    "residence": [1, 2]
}

code_audit = []
for col, allowed in valid_codes.items():
    observed = sorted(df[col].dropna().unique().tolist())
    unexpected = [v for v in observed if v not in allowed]
    code_audit.append({
        "variable": col,
        "observed_values": str(observed),
        "unexpected_values": str(unexpected),
        "unexpected_count": int(df[col].isin(unexpected).sum())
    })

pd.DataFrame(code_audit).to_csv(TABLES / "category_code_audit.csv", index=False)

# Age summary and age frequency table
df["dm005"].describe().to_frame("age").to_csv(TABLES / "age_summary.csv")
age_freq = df["dm005"].value_counts(dropna=False).sort_index().rename_axis("age").reset_index(name="count")
age_freq["percentage"] = (age_freq["count"] / len(df) * 100).round(2)
age_freq.to_csv(TABLES / "age_frequency.csv", index=False)

# 3. Prepare the modelling cohort: age 45+
cohort = df.loc[df["dm005"] >= 45].copy()
invalid_outcome = ~cohort[OUTCOME].isin([1, 2]) & cohort[OUTCOME].notna()
invalid_outcome_count = int(invalid_outcome.sum())

# Keep valid outcome codes only, then recode
cohort = cohort.loc[cohort[OUTCOME].isin([1, 2])].copy()
cohort["jaundice_hepatitis"] = cohort[OUTCOME].map({1: 1, 2: 0})

# Do not include original outcome code as a predictor
model_df = cohort[["jaundice_hepatitis"] + PREDICTORS].copy()

# Missingness in eligible cohort before complete-case deletion
model_df.isna().sum().rename("missing_count").to_csv(
    TABLES / "model_cohort_missingness.csv"
)

before_drop = len(model_df)
model_df = model_df.dropna().copy()
model_df["jaundice_hepatitis"] = model_df["jaundice_hepatitis"].astype(int)

model_df.to_csv(PROCESSED / "jaundice_hepatitis_model_data.csv", index=False)

# Final outcome frequency table
outcome_freq = model_df["jaundice_hepatitis"].value_counts().sort_index()
pd.DataFrame({
    "outcome": ["No", "Yes"],
    "value": [0, 1],
    "count": [int(outcome_freq.get(0, 0)), int(outcome_freq.get(1, 0))],
    "percentage": [
        round(outcome_freq.get(0, 0) / len(model_df) * 100, 2) if len(model_df) else 0,
        round(outcome_freq.get(1, 0) / len(model_df) * 100, 2) if len(model_df) else 0
    ]
}).to_csv(TABLES / "final_outcome_frequency.csv", index=False)

summary = {
    "original_rows": len(df),
    "original_columns_in_dataset": getattr(meta, "number_columns", None),
    "age_45_plus_rows": int((df["dm005"] >= 45).sum()),
    "valid_outcome_rows_after_age_filter": len(cohort),
    "invalid_outcome_codes_excluded": invalid_outcome_count,
    "rows_before_missing_value_removal": before_drop,
    "rows_removed_due_to_missing_values": before_drop - len(model_df),
    "final_model_rows": len(model_df),
    "final_model_columns": list(model_df.columns),
    "positive_cases": int((model_df["jaundice_hepatitis"] == 1).sum()),
    "negative_cases": int((model_df["jaundice_hepatitis"] == 0).sum()),
    "positive_prevalence_percent": round(model_df["jaundice_hepatitis"].mean() * 100, 2) if len(model_df) else None
}

with open(REPORTS / "data_preparation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDATA AUDIT AND PREPROCESSING COMPLETE")
print(json.dumps(summary, indent=2))
print(f"\nTables saved to: {TABLES}")
print(f"Cleaned dataset saved to: {PROCESSED}")
