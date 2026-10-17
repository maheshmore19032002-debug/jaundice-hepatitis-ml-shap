
"""
Script 02: Frequency Analysis and Data Preprocessing.

LASI Wave 1 (2017-18)
Study: Self-reported jaundice/hepatitis among adults aged 45+.

Purpose:
- Audit missing values and category frequencies.
- Validate known variable codes.
- Restrict the cohort to age >= 45.
- Recode the outcome: ht201 1=Yes, 2=No.
- Include state as a categorical predictor.
- Create a complete-case modelling dataset.
- Export state-wise outcome summaries.
"""

from pathlib import Path
import json

import pandas as pd
import pyreadstat


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "raw" / "data_working.dta"

TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"
PROCESSED = ROOT / "data" / "processed"

for folder in [TABLES, REPORTS, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. STUDY VARIABLES
# --------------------------------------------------

OUTCOME = "ht201"
STATE_SOURCE = "stateid"

PREDICTORS = [
    "dm005",                 # Age
    "dm003",                 # Sex
    "dm013",                 # Caste
    "dm010",                 # Religion
    "living_arrangements",   # Living arrangements
    "mpce_quintile",         # MPCE quintile
    "hc102",                 # Health insurance
    "residence",             # Rural/urban residence
    STATE_SOURCE,            # State/UT identifier
]

VARIABLES = [OUTCOME] + PREDICTORS


# --------------------------------------------------
# 3. CHECK INPUT DATASET
# --------------------------------------------------

def main():

    print("=" * 70)
    print("LASI WAVE 1: FREQUENCY AUDIT AND PREPROCESSING")
    print("=" * 70)

    if not DATA.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA}\n\n"
            "Please place data_working.dta inside data/raw/."
        )

    print(f"\nInput dataset: {DATA}")
    print("Reading metadata...")

    _, metadata = pyreadstat.read_dta(
        str(DATA),
        metadataonly=True,
    )

    available_variables = set(metadata.column_names)
    missing_vars = sorted(set(VARIABLES) - available_variables)

    if missing_vars:
        raise ValueError(
            f"Required variables not found in the source dataset: {missing_vars}"
        )

    # --------------------------------------------------
    # 4. READ REQUIRED VARIABLES ONLY
    # --------------------------------------------------

    print("Reading selected study variables...")

    df, metadata = pyreadstat.read_dta(
        str(DATA),
        usecols=VARIABLES,
    )

    original_rows = len(df)

    # Convert study variables to numeric.
    # Unexpected/non-numeric values become missing and are audited.
    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Variable labels from source metadata
    source_labels = metadata.column_labels or []
    labels = dict(zip(metadata.column_names, source_labels))

    print(f"Rows loaded: {original_rows}")
    print(f"Study variables loaded: {len(VARIABLES)}")

    # --------------------------------------------------
    # 5. MISSINGNESS AUDIT AND FREQUENCY TABLES
    # --------------------------------------------------

    audit = []

    for col in VARIABLES:
        s = df[col]

        audit.append({
            "variable": col,
            "variable_label": labels.get(col, ""),
            "non_missing": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_percent": round(s.isna().mean() * 100, 2),
            "unique_non_missing_values": int(s.nunique()),
        })

        freq = (
            s.value_counts(dropna=False)
            .sort_index()
            .rename_axis("value")
            .reset_index(name="count")
        )

        freq["percentage_of_all_rows"] = (
            freq["count"] / len(df) * 100
        ).round(2)

        freq.insert(0, "variable", col)
        freq.insert(1, "variable_label", labels.get(col, ""))

        freq.to_csv(
            TABLES / f"frequency_{col}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(audit).to_csv(
        TABLES / "variable_missingness_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 6. VALIDATE CATEGORY CODES
    # --------------------------------------------------

    valid_codes = {
        "ht201": [1, 2],
        "dm003": [1, 2],
        "dm013": [1, 2, 3, 4, 5],
        "dm010": list(range(1, 11)),
        "living_arrangements": [1, 2, 3, 4, 5],
        "mpce_quintile": [1, 2, 3, 4, 5],
        "hc102": [1, 2],
        "residence": [1, 2],
        "stateid": list(range(1, 37)),
    }

    code_audit = []

    for col, allowed in valid_codes.items():
        observed = sorted(df[col].dropna().unique().tolist())
        unexpected = [v for v in observed if v not in allowed]

        code_audit.append({
            "variable": col,
            "observed_values": str(observed),
            "unexpected_values": str(unexpected),
            "unexpected_count": int(df[col].isin(unexpected).sum()),
        })

    pd.DataFrame(code_audit).to_csv(
        TABLES / "category_code_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 7. STATE CODE AND LABEL REPORT
    # --------------------------------------------------

    state_labels = metadata.variable_value_labels.get("stateid", {})

    state_label_rows = [
        {
            "state_code": str(code),
            "state_name": str(label),
        }
        for code, label in state_labels.items()
    ]

    state_labels_df = pd.DataFrame(
        state_label_rows,
        columns=["state_code", "state_name"],
    )

    state_labels_df.to_csv(
        TABLES / "state_code_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 8. AGE SUMMARY AND FREQUENCY TABLE
    # --------------------------------------------------

    df["dm005"].describe().to_frame("age").to_csv(
        TABLES / "age_summary.csv",
        encoding="utf-8-sig",
    )

    age_freq = (
        df["dm005"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("age")
        .reset_index(name="count")
    )

    age_freq["percentage"] = (
        age_freq["count"] / len(df) * 100
    ).round(2)

    age_freq.to_csv(
        TABLES / "age_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 9. PREPARE COHORT: AGE >= 45
    # --------------------------------------------------

    cohort = df.loc[df["dm005"] >= 45].copy()

    age_45_plus_rows = len(cohort)

    # Count invalid non-missing outcome codes before excluding them.
    invalid_outcome = (
        ~cohort[OUTCOME].isin([1, 2])
        & cohort[OUTCOME].notna()
    )

    invalid_outcome_count = int(invalid_outcome.sum())

    # Retain valid outcome codes only.
    cohort = cohort.loc[
        cohort[OUTCOME].isin([1, 2])
    ].copy()

    # Outcome recoding: 1 = Yes; 2 = No
    cohort["jaundice_hepatitis"] = cohort[OUTCOME].map({
        1: 1,
        2: 0,
    })

    # --------------------------------------------------
    # 10. CREATE MODELLING DATASET
    # --------------------------------------------------

    # Retain all predictors, including stateid.
    model_df = cohort[
        ["jaundice_hepatitis"] + PREDICTORS
    ].copy()

    # Rename stateid to a readable predictor name.
    model_df = model_df.rename(
        columns={"stateid": "state"}
    )

    # Export missingness before complete-case deletion.
    model_df.isna().sum().rename(
        "missing_count"
    ).to_csv(
        TABLES / "model_cohort_missingness.csv",
        encoding="utf-8-sig",
    )

    rows_before_drop = len(model_df)

    # Complete-case analysis, consistent with the existing workflow.
    model_df = model_df.dropna().copy()

    model_df["jaundice_hepatitis"] = (
        model_df["jaundice_hepatitis"].astype(int)
    )

    # State is a categorical code, not a continuous measurement.
    # Retain its numeric codes here; model preprocessing must
    # explicitly encode state as categorical.
    model_df["state"] = model_df["state"].astype(int)

    rows_removed_missing = rows_before_drop - len(model_df)

    # --------------------------------------------------
    # 11. SAVE PROCESSED DATASET
    # --------------------------------------------------

    output_csv = (
        PROCESSED / "jaundice_hepatitis_model_data.csv"
    )

    model_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 12. FINAL OUTCOME FREQUENCY TABLE
    # --------------------------------------------------

    outcome_freq = (
        model_df["jaundice_hepatitis"]
        .value_counts()
        .sort_index()
    )

    total_final = len(model_df)

    pd.DataFrame({
        "outcome": ["No", "Yes"],
        "value": [0, 1],
        "count": [
            int(outcome_freq.get(0, 0)),
            int(outcome_freq.get(1, 0)),
        ],
        "percentage": [
            round(outcome_freq.get(0, 0) / total_final * 100, 2)
            if total_final else 0,
            round(outcome_freq.get(1, 0) / total_final * 100, 2)
            if total_final else 0,
        ],
    }).to_csv(
        TABLES / "final_outcome_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 13. STATE-WISE OUTCOME SUMMARY
    # --------------------------------------------------

    state_summary = (
        model_df.groupby("state", observed=True)
        ["jaundice_hepatitis"]
        .agg(
            total_participants="count",
            positive_cases="sum",
            observed_prevalence="mean",
        )
        .reset_index()
    )

    state_summary["state_name"] = (
        state_summary["state"]
        .map({
            int(code): str(name)
            for code, name in state_labels.items()
        })
    )

    state_summary["observed_prevalence_percent"] = (
        state_summary["observed_prevalence"] * 100
    ).round(2)

    state_summary = state_summary[
        [
            "state",
            "state_name",
            "total_participants",
            "positive_cases",
            "observed_prevalence_percent",
        ]
    ]

    state_summary.to_csv(
        TABLES / "state_wise_outcome_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 14. SAVE PREPROCESSING SUMMARY
    # --------------------------------------------------

    summary = {
        "input_dataset": DATA.name,
        "original_rows": int(original_rows),
        "original_columns_in_dataset": getattr(
            metadata, "number_columns", None
        ),
        "study_variables_loaded": VARIABLES,
        "age_45_plus_rows": int(age_45_plus_rows),
        "valid_outcome_rows_after_age_filter": int(len(cohort)),
        "invalid_outcome_codes_excluded": invalid_outcome_count,
        "rows_before_missing_value_removal": int(rows_before_drop),
        "rows_removed_due_to_missing_values": int(rows_removed_missing),
        "final_model_rows": int(len(model_df)),
        "final_model_columns": list(model_df.columns),
        "final_model_column_count": int(len(model_df.columns)),
        "predictor_count": int(len(PREDICTORS)),
        "state_variable_source": STATE_SOURCE,
        "state_variable_final_name": "state",
        "state_categories_observed": int(model_df["state"].nunique()),
        "positive_cases": int(
            (model_df["jaundice_hepatitis"] == 1).sum()
        ),
        "negative_cases": int(
            (model_df["jaundice_hepatitis"] == 0).sum()
        ),
        "positive_prevalence_percent": round(
            model_df["jaundice_hepatitis"].mean() * 100, 2
        ) if len(model_df) else None,
    }

    with open(
        REPORTS / "data_preparation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # 15. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA AUDIT AND PREPROCESSING COMPLETE")
    print("=" * 70)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nTables saved to: {TABLES}")
    print(f"Reports saved to: {REPORTS}")
    print(f"Cleaned dataset saved to: {output_csv}")

    print("\nAdditional state outputs:")
    print(f"- {TABLES / 'state_code_labels.csv'}")
    print(f"- {TABLES / 'state_wise_outcome_summary.csv'}")

    print("\nScript 02 completed successfully.")


if __name__ == "__main__":
    main()
