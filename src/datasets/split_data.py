

import csv
from pathlib import Path
from collections import defaultdict, Counter

from bus_uclm import BUSUCLMDataset


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bus_uclm_patient_split.csv"
)

# Keep the patient-level split we selected.
TRAIN_PATIENTS = [
    "ANFO", "ASSC", "CHCO", "CHSP", "CHVI",
    "COST", "COVA", "CRCI", "DAPA", "ELCO",
    "FLBA", "FUHI", "HUBL", "KIFO", "MENE",
    "NIRO", "OSCU", "PAGY", "POFR", "SECH",
    "SHST", "SIBA", "STSP", "TOCI", "VITR",
]

VAL_PATIENTS = [
    "COPE", "ORPE", "PLBA", "RARE", "UNCU", "WAQU",
]

TEST_PATIENTS = [
    "ALWI", "ANAT", "CAWI", "CODE", "FLKA", "LOTI",
]


def main():
    dataset = BUSUCLMDataset()

    patient_splits = {
        "train": set(TRAIN_PATIENTS),
        "val": set(VAL_PATIENTS),
        "test": set(TEST_PATIENTS),
    }

    # Verify that no patient appears in multiple splits.
    assert patient_splits["train"].isdisjoint(patient_splits["val"])
    assert patient_splits["train"].isdisjoint(patient_splits["test"])
    assert patient_splits["val"].isdisjoint(patient_splits["test"])

    assigned_patients = set().union(*patient_splits.values())
    dataset_patients = {sample["patient_id"] for sample in dataset.samples}

    assert assigned_patients == dataset_patients, (
        f"Patient mismatch!\n"
        f"Missing: {dataset_patients - assigned_patients}\n"
        f"Unknown: {assigned_patients - dataset_patients}"
    )

    rows = []
    split_counts = defaultdict(Counter)
    split_patients = defaultdict(set)

    for sample in dataset.samples:
        patient_id = sample["patient_id"]

        split = next(
            name
            for name, patients in patient_splits.items()
            if patient_id in patients
        )

        rows.append({
            "filename": sample["filename"],
            "patient_id": patient_id,
            "label": sample["label"],
            "split": split,
        })

        split_counts[split][sample["label"]] += 1
        split_patients[split].add(patient_id)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "patient_id", "label", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total images: {len(rows)}")
    print(f"Total patients: {len(dataset_patients)}")
    print(f"\nSaved split file to:\n{OUTPUT_PATH}")

    print("\nSplit summary:")
    for split in ["train", "val", "test"]:
        print(
            f"{split}: {len(split_patients[split])} patients, "
            f"{sum(split_counts[split].values())} images, "
            f"labels={dict(split_counts[split])}"
        )

    # Verify every image appears exactly once.
    filenames = [row["filename"] for row in rows]
    assert len(filenames) == len(set(filenames))
    assert len(rows) == len(dataset)

    print("\nPatient overlap check: PASSED")
    print("Image coverage check: PASSED")
    print("Split CSV creation: PASSED")


if __name__ == "__main__":
    main()
