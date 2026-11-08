import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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
OUTPUT = PROJECT_ROOT / "data" / "processed" / "bus_uclm_image_split.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    df = pd.read_csv(INPUT)

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
    assert df["filename"].is_unique
    assert df["patient_id"].notna().all()
    assert df["label"].notna().all()

    # Split images independently, stratifying by label.
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=df["label"],
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    result = result[
        ["filename", "patient_id", "label", "split"]
    ].sort_values(["split", "filename"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print(f"Saved: {OUTPUT}")
    print(f"Total images: {len(result)}")
    print("\nImages by split and label:")
    print(pd.crosstab(result["split"], result["label"], margins=True))

    print("\nUnique patients by split:")
    for split, group in result.groupby("split"):
        print(
            f"{split}: {len(group)} images, "
            f"{group['patient_id'].nunique()} patients"
        )

    patient_sets = {
        split: set(group["patient_id"])
        for split, group in result.groupby("split")
    }

    print("\nPatient overlap (expected for image-level splitting):")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[a] & patient_sets[b]
        print(f"{a} vs {b}: {len(overlap)} patients")


if __name__ == "__main__":
    main()