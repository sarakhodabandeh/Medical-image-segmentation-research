from itertools import combinations
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
INPUT = PROJECT_ROOT / "results" / "test_evaluation" / "test_per_image_metrics.csv"
OUTPUT = PROJECT_ROOT / "results" / "patient_overlap_audit"


def main():
    df = pd.read_csv(INPUT)
    required = {"seed", "split_type", "filename", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    test = df[
        ["seed", "split_type", "filename", "patient_id", "label"]
    ].drop_duplicates()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in ("image", "patient"):
        subset = test[test["split_type"] == split]
        patient_sets = {
            seed: set(group["patient_id"].dropna().astype(str).unique())
            for seed, group in subset.groupby("seed")
        }

        print(f"\\n--- {split.upper()} SPLIT ---")
        for seed, patients in patient_sets.items():
            print(f"Seed {seed}: {len(patients)} patients")

        rows = []
        for seed_a, seed_b in combinations(sorted(patient_sets), 2):
            shared = patient_sets[seed_a] & patient_sets[seed_b]
            rows.append({
                "split_type": split,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "patients_a": len(patient_sets[seed_a]),
                "patients_b": len(patient_sets[seed_b]),
                "shared_patients": len(shared),
                "shared_patient_ids": ";".join(sorted(shared)),
            })

        overlap = pd.DataFrame(
            rows,
            columns=[
                "split_type", "seed_a", "seed_b", "patients_a",
                "patients_b", "shared_patients", "shared_patient_ids",
            ],
        )
        print("\\nPairwise overlap:")
        print(overlap.to_string(index=False))
        overlap.to_csv(
            OUTPUT / f"patient_overlap_{split}.csv",
            index=False,
        )

    print("\\nSaved overlap audit files.")


if __name__ == "__main__":
    main()
