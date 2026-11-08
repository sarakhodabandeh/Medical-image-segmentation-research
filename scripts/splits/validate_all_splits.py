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

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()

REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "bus_uclm_patient_split.csv"

FOLDERS = {
    "image": PROJECT_ROOT / "data" / "processed" / "matched_splits",
    "patient": PROJECT_ROOT / "data" / "processed" / "patient_level_splits",
}


def main():
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"Reference split file not found: {REFERENCE_PATH}")

    reference = pd.read_csv(REFERENCE_PATH)
    if "filename" not in reference.columns:
        raise ValueError(
            f"Reference file does not contain a 'filename' column: {REFERENCE_PATH}"
        )

    reference_files = set(reference["filename"].dropna().astype(str))

    print("=" * 65)
    print("VALIDATING ALL SPLITS")
    print("=" * 65)

    for split_type, folder in FOLDERS.items():
        if not folder.is_dir():
            print(f"\\n{split_type.upper()}: folder not found: {folder}")
            continue

        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            print(f"\\n{split_type.upper()}: no CSV files found in {folder}")
            continue

        for csv_path in csv_files:
            if csv_path.name.endswith("summary.csv"):
                continue

            df = pd.read_csv(csv_path)

            required_columns = {"filename", "split", "patient_id"}
            missing_columns = required_columns - set(df.columns)

            print(f"\\n{split_type.upper()}: {csv_path.name}")

            if missing_columns:
                print(f"Missing required columns: {sorted(missing_columns)}")
                print("STATUS: ERROR - invalid split schema")
                continue

            files = set(df["filename"].dropna().astype(str))

            missing = reference_files - files
            extra = files - reference_files
            duplicate_count = int(df["filename"].duplicated().sum())

            patient_overlap = {}
            for a, b in (
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            ):
                patients_a = set(
                    df.loc[df["split"] == a, "patient_id"].dropna().astype(str)
                )
                patients_b = set(
                    df.loc[df["split"] == b, "patient_id"].dropna().astype(str)
                )
                patient_overlap[f"{a}-{b}"] = len(patients_a & patients_b)

            print(f"Rows: {len(df)}")
            print(f"Unique images: {len(files)}")
            print(f"Missing images: {len(missing)}")
            print(f"Extra images: {len(extra)}")
            print(f"Duplicate images: {duplicate_count}")
            print(f"Patient overlap: {patient_overlap}")

            if missing or extra or duplicate_count:
                print("STATUS: ERROR")
            elif split_type == "patient" and any(patient_overlap.values()):
                print("STATUS: ERROR - patient leakage")
            elif split_type == "image" and not all(patient_overlap.values()):
                print("STATUS: WARNING - unexpected zero overlap")
            else:
                print("STATUS: OK")


if __name__ == "__main__":
    main()
