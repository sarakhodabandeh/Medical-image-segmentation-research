from pathlib import Path
import numpy as np
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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bus_uclm_patient_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matched_splits"
)

SEEDS = [42, 123, 2025, 7, 99]

SPLITS = ["train", "val", "test"]
# --------------------------------------------------
# Load patient-level reference split
# --------------------------------------------------
def main():
    df = pd.read_csv(INPUT)

    required = {"filename", "patient_id", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV is missing required columns.")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found.")

    # Exact class counts from the existing patient-level split
    target_counts = (
        pd.crosstab(df["label"], df["split"])
        .reindex(columns=SPLITS, fill_value=0)
    )

    print("Target counts from patient-level reference:")
    print(target_counts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    # --------------------------------------------------
    # Generate image-level splits
    # --------------------------------------------------

    for seed in SEEDS:

        rng = np.random.default_rng(seed)

        parts = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Split independently within each label.
        # This guarantees exact class counts in each split.
        for label in sorted(df["label"].unique()):

            label_df = (
                df[df["label"] == label]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )

            counts = target_counts.loc[label]

            n_train = int(counts["train"])
            n_val = int(counts["val"])
            n_test = int(counts["test"])

            if n_train + n_val + n_test != len(label_df):
                raise ValueError(
                    f"Count mismatch for label {label}"
                )

            # Independent reproducible shuffle for this label
            indices = rng.permutation(len(label_df))

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            parts["train"].append(label_df.iloc[train_idx])
            parts["val"].append(label_df.iloc[val_idx])
            parts["test"].append(label_df.iloc[test_idx])

        # Combine splits
        split_frames = []

        for split in SPLITS:
            part = pd.concat(parts[split], ignore_index=True).copy()
            part["split"] = split
            split_frames.append(part)

        result = pd.concat(split_frames, ignore_index=True)

        result = result[
            ["filename", "patient_id", "label", "split"]
        ].sort_values(["split", "filename"])

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if len(result) != len(df):
            raise ValueError("Total image count changed.")

        if result["filename"].duplicated().any():
            raise ValueError("Duplicate filename in generated split.")

        actual_counts = pd.crosstab(
            result["label"], result["split"]
        ).reindex(
            index=target_counts.index,
            columns=SPLITS,
            fill_value=0
        )

        if not actual_counts.equals(target_counts):
            raise ValueError(
                f"Class distribution mismatch for seed {seed}"
            )

        # Save CSV
        output_path = OUTPUT_DIR / f"image_level_seed_{seed}.csv"
        result.to_csv(output_path, index=False)

        # --------------------------------------------------
        # Patient overlap analysis
        # --------------------------------------------------

        patient_sets = {
            split: set(result.loc[result["split"] == split, "patient_id"])
            for split in SPLITS
        }

        overlap_tv = len(patient_sets["train"] & patient_sets["val"])
        overlap_tt = len(patient_sets["train"] & patient_sets["test"])
        overlap_vt = len(patient_sets["val"] & patient_sets["test"])

        summary = {
            "seed": seed,
            "train_images": len(parts["train"][0]) + sum(
                len(x) for x in parts["train"][1:]
            ) if len(parts["train"]) > 1 else len(parts["train"][0]),
            "val_images": int((result["split"] == "val").sum()),
            "test_images": int((result["split"] == "test").sum()),
            "train_val_patient_overlap": overlap_tv,
            "train_test_patient_overlap": overlap_tt,
            "val_test_patient_overlap": overlap_vt,
            "train_patients": len(patient_sets["train"]),
            "val_patients": len(patient_sets["val"]),
            "test_patients": len(patient_sets["test"]),
        }

        all_summaries.append(summary)

        print(f"\nSeed {seed}")
        print(f"Saved: {output_path}")
        print(
            f"Images: train={summary['train_images']}, "
            f"val={summary['val_images']}, "
            f"test={summary['test_images']}"
        )
        print(
            f"Patient overlap: train-val={overlap_tv}, "
            f"train-test={overlap_tt}, "
            f"val-test={overlap_vt}"
        )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "image_split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("ALL SPLITS GENERATED")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()