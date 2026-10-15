
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "jaundice_hepatitis_model_data.csv"
OUT = ROOT / "outputs" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "jaundice_hepatitis"
PREDICTORS = [
    "dm005", "dm003", "dm013", "dm010",
    "living_arrangements", "mpce_quintile", "hc102", "residence"
]
EXPECTED = [TARGET] + PREDICTORS

df = pd.read_csv(DATA)

# Check dataset structure
missing_cols = sorted(set(EXPECTED) - set(df.columns))
extra_cols = sorted(set(df.columns) - set(EXPECTED))

if missing_cols or extra_cols:
    raise ValueError(
        f"Missing columns: {missing_cols}; Unexpected columns: {extra_cols}"
    )

if df.empty:
    raise ValueError("Processed dataset is empty.")

if df.isna().any().any():
    raise ValueError("Missing values remain in the processed dataset.")

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

if not (df["dm005"] >= 45).all():
    raise ValueError("Participants younger than 45 were found.")

# Check category values against expected coding
valid_codes = {
    "dm003": [1, 2],
    "dm013": [1, 2, 3, 4, 5],
    "dm010": list(range(1, 11)),
    "living_arrangements": [1, 2, 3, 4, 5],
    "mpce_quintile": [1, 2, 3, 4, 5],
    "hc102": [1, 2],
    "residence": [1, 2],
}

issues = {}
for col, allowed in valid_codes.items():
    observed = sorted(df[col].unique().tolist())
    unexpected = [value for value in observed if value not in allowed]
    if unexpected:
        issues[col] = unexpected

if issues:
    raise ValueError(f"Unexpected category codes: {issues}")

# Save validation summary
summary = {
    "validation_status": "PASSED",
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_values": int(df.isna().sum().sum()),
    "positive_cases": int((df[TARGET] == 1).sum()),
    "negative_cases": int((df[TARGET] == 0).sum()),
    "positive_prevalence_percent": round(df[TARGET].mean() * 100, 2),
    "predictors": PREDICTORS,
}

pd.Series(summary).to_json(
    OUT / "processed_data_validation.json",
    indent=2,
    force_ascii=False
)

print("\nDATASET VALIDATION PASSED")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Positive cases: {(df[TARGET] == 1).sum():,}")
print(f"Negative cases: {(df[TARGET] == 0).sum():,}")
print(f"Positive prevalence: {df[TARGET].mean() * 100:.2f}%")
print(f"Validation report: {OUT / 'processed_data_validation.json'}")
