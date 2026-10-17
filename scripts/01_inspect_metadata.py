
"""
Script 01: Inspect LASI Wave 1 dataset metadata.

Purpose:
- Read Stata metadata without loading the full dataset into memory.
- Export variable names and labels.
- Check whether the study variables exist.
- Export category/value labels, including state names.
- Save a metadata summary for reproducibility.
"""

from pathlib import Path
import json
import pandas as pd
import pyreadstat


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data_working.dta"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. VARIABLES REQUIRED FOR THIS STUDY
# --------------------------------------------------

OUTCOME = "ht201"

PREDICTORS = [
    "dm005",                 # Age
    "dm003",                 # Sex
    "dm013",                 # Caste
    "dm010",                 # Religion
    "living_arrangements",   # Living arrangements
    "mpce_quintile",         # MPCE quintile
    "hc102",                 # Health insurance
    "residence",             # Rural/urban residence
    "stateid",               # State/UT identifier
]

REQUIRED_VARIABLES = [OUTCOME] + PREDICTORS


# --------------------------------------------------
# 3. READ METADATA ONLY
# --------------------------------------------------

def main():
    print("=" * 65)
    print("LASI WAVE 1: METADATA INSPECTION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n"
            "Place data_working.dta inside the project's data/raw folder."
        )

    print(f"\nDataset: {DATA_PATH}")
    print("Reading metadata only; the full dataset will not be loaded...")

    _, metadata = pyreadstat.read_dta(
        str(DATA_PATH),
        metadataonly=True
    )

    variable_names = metadata.column_names
    variable_labels = list(metadata.column_labels or [])

    if len(variable_labels) < len(variable_names):
        variable_labels += [""] * (
            len(variable_names) - len(variable_labels)
        )

    # --------------------------------------------------
    # 4. BUILD VARIABLE METADATA TABLE
    # --------------------------------------------------

    metadata_df = pd.DataFrame({
        "variable_name": variable_names,
        "variable_label": variable_labels,
    })

    metadata_df["required_for_study"] = (
        metadata_df["variable_name"].isin(REQUIRED_VARIABLES)
    )

    metadata_file = REPORT_DIR / "variable_metadata.csv"
    metadata_df.to_csv(
        metadata_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # 5. CHECK REQUIRED VARIABLES
    # --------------------------------------------------

    available_variables = set(variable_names)
    availability = []

    for variable in REQUIRED_VARIABLES:
        row = metadata_df.loc[
            metadata_df["variable_name"] == variable
        ]

        label = (
            row["variable_label"].iloc[0]
            if not row.empty
            else ""
        )

        availability.append({
            "variable_name": variable,
            "variable_label": label,
            "available": variable in available_variables,
            "role": "Outcome" if variable == OUTCOME else "Predictor",
        })

    availability_df = pd.DataFrame(availability)

    availability_file = REPORT_DIR / "required_variables_check.csv"
    availability_df.to_csv(
        availability_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # 6. EXPORT CATEGORY / VALUE LABELS
    # --------------------------------------------------

    value_label_rows = []

    for variable, labels in metadata.variable_value_labels.items():
        for code, label in labels.items():
            value_label_rows.append({
                "variable_name": variable,
                "category_code": str(code),
                "category_label": str(label),
            })

    value_labels_df = pd.DataFrame(
        value_label_rows,
        columns=[
            "variable_name",
            "category_code",
            "category_label",
        ],
    )

    value_labels_file = REPORT_DIR / "variable_value_labels.csv"
    value_labels_df.to_csv(
        value_labels_file,
        index=False,
        encoding="utf-8-sig",
    )

    # State-specific label report
    state_labels_df = value_labels_df.loc[
        value_labels_df["variable_name"] == "stateid"
    ].copy()

    state_labels_file = REPORT_DIR / "state_value_labels.csv"
    state_labels_df.to_csv(
        state_labels_file,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 7. SAVE DATASET SUMMARY
    # --------------------------------------------------

    summary = {
        "dataset_file": DATA_PATH.name,
        "dataset_label": getattr(metadata, "file_label", None),
        "number_of_rows": getattr(metadata, "number_rows", None),
        "number_of_columns": len(variable_names),
        "required_variable_count": len(REQUIRED_VARIABLES),
        "required_variables_found": int(
            availability_df["available"].sum()
        ),
        "required_variables_missing": availability_df.loc[
            ~availability_df["available"], "variable_name"
        ].tolist(),
        "state_variable": "stateid",
        "state_labels_exported": len(state_labels_df),
        "metadata_only": True,
    }

    summary_file = REPORT_DIR / "dataset_metadata_summary.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # --------------------------------------------------
    # 8. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n--- DATASET SUMMARY ---")
    print(f"Rows: {summary['number_of_rows']}")
    print(f"Columns: {summary['number_of_columns']}")
    print(f"Required variables: {summary['required_variable_count']}")
    print(f"State labels found: {summary['state_labels_exported']}")

    print("\n--- REQUIRED VARIABLE CHECK ---")
    print(availability_df.to_string(index=False))

    print("\n--- STATE VALUE LABELS ---")
    if state_labels_df.empty:
        print("WARNING: No value labels found for stateid.")
    else:
        print(state_labels_df.to_string(index=False))

    print("\n--- OUTPUT FILES ---")
    print(f"Variable metadata: {metadata_file}")
    print(f"Required variable check: {availability_file}")
    print(f"All value labels: {value_labels_file}")
    print(f"State value labels: {state_labels_file}")
    print(f"Dataset summary: {summary_file}")

    if summary["required_variables_missing"]:
        print("\nWARNING: Some required variables were not found.")
        print("Review variable names before proceeding.")
    else:
        print("\nSUCCESS: All required variables were found.")

    print("\nScript 01 completed.")


if __name__ == "__main__":
    main()
