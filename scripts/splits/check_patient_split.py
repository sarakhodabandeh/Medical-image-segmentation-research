from pathlib import Path
import pandas as pd


def find_project_root():
    current = Path(__file__).resolve().parent

    for directory in (current, *current.parents):
        if (
            (directory / "configs").is_dir()
            and (directory / "src").is_dir()
            and (directory / "requirements.txt").is_file()
        ):
            return directory

    raise RuntimeError("Could not locate project root.")


PROJECT_ROOT = find_project_root()

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bus_uclm_patient_split.csv"
)


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Split CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"filename", "patient_id", "label", "split"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Unique filenames: {df['filename'].nunique()}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Duplicate filenames: {df['filename'].duplicated().sum()}")

    print("\n" + "=" * 60)
    print("2. IMAGES BY SPLIT")
    print("=" * 60)
    print(pd.crosstab(df["split"], df["label"], margins=True))

    print("\n" + "=" * 60)
    print("3. UNIQUE PATIENTS BY SPLIT")
    print("=" * 60)

    for split, group in df.groupby("split"):
        print(
            f"{split}: {group['patient_id'].nunique()} patients, "
            f"{len(group)} images"
        )

    print("\n" + "=" * 60)
    print("4. PATIENT OVERLAP")
    print("=" * 60)

    patient_sets = {
        split: set(group["patient_id"].dropna())
        for split, group in df.groupby("split")
    }

    split_names = ["train", "val", "test"]

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = patient_sets.get(a, set()) & patient_sets.get(b, set())

            print(f"{a} vs {b}: {len(overlap)} overlapping patients")

            if overlap:
                print("  IDs:", sorted(overlap))

    print("\n" + "=" * 60)
    print("5. DATA QUALITY CHECKS")
    print("=" * 60)
    print("Missing values:")
    print(df[list(required)].isna().sum())

    print("\nUnexpected split names:")
    print(sorted(set(df["split"].dropna()) - {"train", "val", "test"}))

    print("\nPatient IDs assigned to multiple splits:")
    patient_split_counts = df.groupby("patient_id")["split"].nunique()
    bad_patients = patient_split_counts[patient_split_counts > 1]
    print(bad_patients if len(bad_patients) else "None")

    print("\nCheck completed.")


if __name__ == "__main__":
    main()