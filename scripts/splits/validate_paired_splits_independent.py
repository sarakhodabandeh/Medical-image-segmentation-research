from itertools import combinations
from pathlib import Path

import pandas as pd


EXPECTED_SEEDS = [42, 123, 2025, 7, 99]
LABELS = ["Benign", "Malignant", "Normal"]
TARGET_VAL = {"Benign": 26, "Malignant": 12, "Normal": 62}
TARGET_TEST = {"Benign": 31, "Malignant": 14, "Normal": 61}


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
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "paired_splits"
REPORT_PATH = SPLIT_DIR / "independent_paired_split_audit.csv"
MANIFEST_PATH = SPLIT_DIR / "paired_split_manifest.csv"


def patient_set(frame, split):
    return set(frame.loc[frame["split"] == split, "patient_id"].dropna())


def validate_one_seed(seed):
    patient_path = SPLIT_DIR / f"patient_level_seed_{seed}.csv"
    image_path = SPLIT_DIR / f"image_level_seed_{seed}.csv"

    if not patient_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(
            f"Missing paired split CSV for seed {seed}: "
            f"{patient_path.name} / {image_path.name}"
        )

    pl = pd.read_csv(patient_path)
    il = pd.read_csv(image_path)
    required = {"filename", "patient_id", "label", "split"}

    for name, frame in [("patient-level", pl), ("image-level", il)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Seed {seed} {name} file missing columns: {sorted(missing)}"
            )
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Seed {seed} {name} file contains missing required values")
        if frame["filename"].duplicated().any():
            raise ValueError(f"Seed {seed} {name} file has duplicate filenames")
        unexpected = set(frame["split"].unique()) - {"train", "val", "test"}
        if unexpected:
            raise ValueError(
                f"Seed {seed} {name} file has unexpected split labels: {unexpected}"
            )

    paired_rows = ["filename", "patient_id", "label"]
    for split in ("val", "test"):
        left = (
            pl.loc[pl["split"] == split, paired_rows]
            .sort_values("filename")
            .reset_index(drop=True)
        )
        right = (
            il.loc[il["split"] == split, paired_rows]
            .sort_values("filename")
            .reset_index(drop=True)
        )
        if not left.equals(right):
            raise AssertionError(
                f"Seed {seed}: {split} rows differ between paired conditions"
            )

    # No filename may occur in more than one split within a condition.
    for name, frame in [("patient-level", pl), ("image-level", il)]:
        split_files = {
            split: set(frame.loc[frame["split"] == split, "filename"])
            for split in ("train", "val", "test")
        }
        for a, b in combinations(("train", "val", "test"), 2):
            if split_files[a] & split_files[b]:
                raise AssertionError(
                    f"Seed {seed} {name}: filename overlap between {a} and {b}"
                )

    pl_train = pl[pl["split"] == "train"]
    il_train = il[il["split"] == "train"]
    if len(pl_train) != len(il_train):
        raise AssertionError(f"Seed {seed}: paired train image counts differ")
    if pl_train["label"].value_counts().to_dict() != il_train["label"].value_counts().to_dict():
        raise AssertionError(f"Seed {seed}: paired train class distributions differ")

    val_patients = patient_set(pl, "val")
    test_patients = patient_set(pl, "test")
    heldout = val_patients | test_patients

    pl_train_patients = patient_set(pl, "train")
    il_train_patients = patient_set(il, "train")
    if pl_train_patients & heldout:
        raise AssertionError(f"Seed {seed}: patient-level train overlaps held-out patients")
    if not heldout.issubset(il_train_patients):
        raise AssertionError(f"Seed {seed}: image-level train misses held-out patients")

    patient_overlap = {
        "train_val": len(pl_train_patients & val_patients),
        "train_test": len(pl_train_patients & test_patients),
        "val_test": len(val_patients & test_patients),
    }
    if any(patient_overlap.values()):
        raise AssertionError(f"Seed {seed}: patient-level patient leakage: {patient_overlap}")

    counts = {}
    for split, target in (("val", TARGET_VAL), ("test", TARGET_TEST)):
        for condition, frame in (("patient", pl), ("image", il)):
            actual = frame.loc[frame["split"] == split, "label"].value_counts().to_dict()
            if any(actual.get(label, 0) != target[label] for label in LABELS):
                raise AssertionError(
                    f"Seed {seed} {condition} {split}: expected {target}, got {actual}"
                )
        counts[f"{split}_images"] = int((pl["split"] == split).sum())

    # Independent confirmation that held-out patients are represented in
    # image-level train, while patient-level train remains patient-disjoint.
    leakage_patients = il_train_patients & heldout

    return {
        "seed": seed,
        "patient_train_images": len(pl_train),
        "image_train_images": len(il_train),
        "train_class_counts_patient": str(pl_train["label"].value_counts().reindex(LABELS, fill_value=0).to_dict()),
        "train_class_counts_image": str(il_train["label"].value_counts().reindex(LABELS, fill_value=0).to_dict()),
        "val_images": counts["val_images"],
        "test_images": counts["test_images"],
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "image_train_heldout_patients": len(leakage_patients),
        "patient_train_heldout_overlap": 0,
        "paired_val_rows_identical": True,
        "paired_test_rows_identical": True,
        "status": "PASS",
    }


def main():
    if not SPLIT_DIR.is_dir():
        raise FileNotFoundError(
            f"Paired split directory not found: {SPLIT_DIR}. "
            "Generate the paired splits first."
        )

    if MANIFEST_PATH.is_file():
        manifest = pd.read_csv(MANIFEST_PATH)
        if "seed" not in manifest.columns:
            raise ValueError("Paired split manifest is missing the seed column")
        manifest_seeds = sorted(manifest["seed"].astype(int).tolist())
        if manifest_seeds != sorted(EXPECTED_SEEDS):
            raise AssertionError(
                f"Manifest seeds {manifest_seeds} do not match {EXPECTED_SEEDS}"
            )

    reports = [validate_one_seed(seed) for seed in EXPECTED_SEEDS]
    report = pd.DataFrame(reports)
    report.to_csv(REPORT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nAll {len(EXPECTED_SEEDS)} paired split configurations passed.")
    print("Independent audit report:", REPORT_PATH)


if __name__ == "__main__":
    main()
