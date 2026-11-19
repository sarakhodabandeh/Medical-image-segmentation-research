import argparse
from itertools import combinations

import pandas as pd

from src.utils.paths import PROJECT_ROOT, project_path

INPUT = PROJECT_ROOT / "results" / "test_evaluation" / "test_per_image_metrics.csv"
OUTPUT = PROJECT_ROOT / "results" / "patient_overlap_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit patient overlap between the image-level and patient-level splits."
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(INPUT),
        help="Per-image metrics CSV from evaluate_paired_test.py.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT), help="Folder for the audit CSVs.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = project_path(args.metrics_csv)
    output_dir = project_path(args.output_dir)
    df = pd.read_csv(input_path)
    required = {"seed", "split_type", "filename", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    test = df[["seed", "split_type", "filename", "patient_id", "label"]].drop_duplicates()

    output_dir.mkdir(parents=True, exist_ok=True)

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
            rows.append(
                {
                    "split_type": split,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "patients_a": len(patient_sets[seed_a]),
                    "patients_b": len(patient_sets[seed_b]),
                    "shared_patients": len(shared),
                    "shared_patient_ids": ";".join(sorted(shared)),
                }
            )

        overlap = pd.DataFrame(
            rows,
            columns=[
                "split_type",
                "seed_a",
                "seed_b",
                "patients_a",
                "patients_b",
                "shared_patients",
                "shared_patient_ids",
            ],
        )
        print("\\nPairwise overlap:")
        print(overlap.to_string(index=False))
        overlap.to_csv(
            output_dir / f"patient_overlap_{split}.csv",
            index=False,
        )

    print("\\nSaved overlap audit files.")


if __name__ == "__main__":
    main()
