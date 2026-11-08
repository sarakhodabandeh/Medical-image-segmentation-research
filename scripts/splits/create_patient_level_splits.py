import numpy as np
import pandas as pd
from pathlib import Path


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

INPUT = PROJECT_ROOT / "data" / "processed" / "bus_uclm_patient_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "patient_level_splits"

SEEDS = [42, 123, 2025, 7, 99]
N_CANDIDATES = 10000

SPLITS = ["train", "val", "test"]

TARGET_IMAGES = {
    "train": 464,
    "val": 100,
    "test": 106,
}

TARGET_CLASS_COUNTS = {
    "Benign": {"train": 113, "val": 26, "test": 31},
    "Malignant": {"train": 64, "val": 12, "test": 14},
    "Normal": {"train": 287, "val": 62, "test": 61},
}


def evaluate_assignment(assignment, patients, patient_table):
    """
    assignment: dictionary patient_id -> train/val/test

    Lower score means closer to target image/class counts.
    """

    score = 0.0
    report = {}

    for split in SPLITS:
        selected = [
            patient
            for patient in patients
            if assignment[patient] == split
        ]

        counts = patient_table.loc[selected].sum()
        total_images = int(counts.sum())

        report[split] = {
            "patients": len(selected),
            "images": total_images,
        }

        target_total = TARGET_IMAGES[split]

        score += (
            (total_images - target_total) / target_total
        ) ** 2

        for label in TARGET_CLASS_COUNTS:
            target = TARGET_CLASS_COUNTS[label][split]
            actual = int(counts[label])

            score += (
                (actual - target) / max(target, 1)
            ) ** 2

    return score, report


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {"filename", "patient_id", "label", "split"}

    if not required.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found.")

    patients = sorted(df["patient_id"].unique())

    if len(patients) != 37:
        raise ValueError(
            f"Expected 37 patients, found {len(patients)}"
        )

    patient_table = pd.crosstab(
        df["patient_id"],
        df["label"],
    )

    for label in TARGET_CLASS_COUNTS:
        if label not in patient_table.columns:
            patient_table[label] = 0

    patient_table = patient_table[
        ["Benign", "Malignant", "Normal"]
    ]

    patient_table = patient_table.reindex(patients)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        best_score = float("inf")
        best_assignment = None
        best_report = None

        # Keep exactly 25 train, 6 val, 6 test patients.
        for _ in range(N_CANDIDATES):
            shuffled = rng.permutation(patients)

            assignment = {}

            for patient in shuffled[:25]:
                assignment[patient] = "train"

            for patient in shuffled[25:31]:
                assignment[patient] = "val"

            for patient in shuffled[31:37]:
                assignment[patient] = "test"

            score, report = evaluate_assignment(
                assignment,
                patients,
                patient_table,
            )

            if score < best_score:
                best_score = score
                best_assignment = assignment.copy()
                best_report = report

        result = df.copy()

        result["split"] = result["patient_id"].map(
            best_assignment
        )

        if result["split"].isna().any():
            raise ValueError("Some patients were not assigned.")

        patient_sets = {
            split: set(
                result.loc[result["split"] == split, "patient_id"]
            )
            for split in SPLITS
        }

        for i in range(len(SPLITS)):
            for j in range(i + 1, len(SPLITS)):
                overlap = (
                    patient_sets[SPLITS[i]]
                    & patient_sets[SPLITS[j]]
                )

                if overlap:
                    raise ValueError(
                        f"Patient overlap found: {overlap}"
                    )

        output_path = (
            OUTPUT_DIR / f"patient_level_seed_{seed}.csv"
        )

        result[
            ["filename", "patient_id", "label", "split"]
        ].sort_values(
            ["split", "filename"]
        ).to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print(f"PATIENT-LEVEL SEED {seed}")
        print("=" * 60)

        print(f"Objective score: {best_score:.6f}")

        for split in SPLITS:
            selected = result[result["split"] == split]
            counts = selected["label"].value_counts()

            print(
                f"\n{split.upper()}: "
                f"{selected['patient_id'].nunique()} patients, "
                f"{len(selected)} images"
            )

            for label in ["Benign", "Malignant", "Normal"]:
                print(f"  {label}: {counts.get(label, 0)}")

        print(f"\nSaved: {output_path}")

        summary = {
            "seed": seed,
            "objective_score": best_score,
        }

        for split in SPLITS:
            selected = result[result["split"] == split]

            summary[f"{split}_patients"] = (
                selected["patient_id"].nunique()
            )
            summary[f"{split}_images"] = len(selected)

            for label in ["Benign", "Malignant", "Normal"]:
                summary[f"{split}_{label}"] = int(
                    (selected["label"] == label).sum()
                )

        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    summary_path = OUTPUT_DIR / "patient_split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()