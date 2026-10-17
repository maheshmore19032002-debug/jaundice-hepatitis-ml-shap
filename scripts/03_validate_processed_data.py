
"""
Script 03: Validate the processed LASI modelling dataset.

Checks:
- Expected columns and dataset structure
- Missing values
- Binary outcome coding
- Age eligibility (45+)
- Categorical predictor codes
- State/UT codes
- Saves a reproducibility validation report
"""

from pathlib import Path
import json

import pandas as pd


# --------------------------------------------------
# 1. PATHS AND VARIABLES
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
OUT = ROOT / "outputs" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "jaundice_hepatitis"

PREDICTORS = [
    "dm005",                 # Age
    "dm003",                 # Sex
    "dm013",                 # Caste
    "dm010",                 # Religion
    "living_arrangements",   # Living arrangements
    "mpce_quintile",         # MPCE quintile
    "hc102",                 # Health insurance
    "residence",             # Rural/urban
    "state",                 # State/UT
]

EXPECTED = [TARGET] + PREDICTORS


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not DATA.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA}\n"
        "Run scripts/02_frequency_and_preprocessing.py first."
    )

df = pd.read_csv(DATA)


# --------------------------------------------------
# 3. CHECK DATASET STRUCTURE
# --------------------------------------------------

missing_cols = sorted(set(EXPECTED) - set(df.columns))
extra_cols = sorted(set(df.columns) - set(EXPECTED))

if missing_cols or extra_cols:
    raise ValueError(
        f"Missing columns: {missing_cols}; "
        f"Unexpected columns: {extra_cols}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].to_dict()
    raise ValueError(
        f"Missing values remain in the processed dataset: {missing_counts}"
    )


# --------------------------------------------------
# 4. CHECK OUTCOME AND AGE
# --------------------------------------------------

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

if not (df["dm005"] >= 45).all():
    raise ValueError("Participants younger than 45 were found.")


# --------------------------------------------------
# 5. CHECK CATEGORICAL CODES
# --------------------------------------------------

valid_codes = {
    "dm003": [1, 2],
    "dm013": [1, 2, 3, 4, 5],
    "dm010": list(range(1, 11)),
    "living_arrangements": [1, 2, 3, 4, 5],
    "mpce_quintile": [1, 2, 3, 4, 5],
    "hc102": [1, 2],
    "residence": [1, 2],
    "state": list(range(1, 37)),
}

issues = {}

for col, allowed in valid_codes.items():
    observed = sorted(df[col].unique().tolist())
    unexpected = [
        value for value in observed
        if value not in allowed
    ]

    if unexpected:
        issues[col] = unexpected

if issues:
    raise ValueError(
        f"Unexpected category codes: {issues}"
    )


# --------------------------------------------------
# 6. STATE VARIABLE AUDIT
# --------------------------------------------------

state_counts = (
    df["state"]
    .value_counts()
    .sort_index()
)

state_summary = [
    {
        "state_code": int(code),
        "participant_count": int(count),
    }
    for code, count in state_counts.items()
]

# Save state distribution for review
pd.DataFrame(state_summary).to_csv(
    OUT / "validated_state_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# --------------------------------------------------
# 7. SAVE VALIDATION SUMMARY
# --------------------------------------------------

summary = {
    "validation_status": "PASSED",
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_values": int(df.isna().sum().sum()),
    "positive_cases": int((df[TARGET] == 1).sum()),
    "negative_cases": int((df[TARGET] == 0).sum()),
    "positive_prevalence_percent": round(
        df[TARGET].mean() * 100, 2
    ),
    "predictors": PREDICTORS,
    "state_variable": "state",
    "observed_state_categories": int(df["state"].nunique()),
    "state_codes_observed": sorted(
        int(code) for code in df["state"].unique()
    ),
    "state_distribution_file": "validated_state_distribution.csv",
}

report_path = OUT / "processed_data_validation.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET VALIDATION PASSED")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Positive cases: {(df[TARGET] == 1).sum():,}")
print(f"Negative cases: {(df[TARGET] == 0).sum():,}")
print(f"Positive prevalence: {df[TARGET].mean() * 100:.2f}%")
print(f"Predictors: {len(PREDICTORS)}")
print(f"Observed state categories: {df['state'].nunique()}")

print(f"\nValidation report: {report_path}")
print(f"State distribution: {OUT / 'validated_state_distribution.csv'}")
