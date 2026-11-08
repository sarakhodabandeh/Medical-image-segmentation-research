import argparse
import itertools
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

DEFAULT_INPUT = "data/processed/bus_uclm_patient_split.csv"
OUTPUT_DIR = "../../data/processed/paired_split_candidates"

# Reference class counts from the previously established matched image split.
TARGET_VAL = {"Benign": 26, "Malignant": 12, "Normal": 62}
TARGET_TEST = {"Benign": 31, "Malignant": 14, "Normal": 61}
LABELS = ["Benign", "Malignant", "Normal"]


def choose_rows(pool, n, target_counts, rng, required_patients):
    """Choose exactly n rows, ensuring every assigned patient is represented."""
    if len(pool) < n:
        return None

    # Guarantee at least one validation/test image from every assigned patient.
    seed_rows = []
    for patient_id in required_patients:
        patient_rows = pool[pool["patient_id"] == patient_id]
        if patient_rows.empty:
            return None
        seed_rows.append(
            patient_rows.sample(
                n=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        )
    chosen = pd.concat(seed_rows)
    remaining = pool[~pool["filename"].isin(chosen["filename"])]

    chosen_counts = chosen["label"].value_counts().to_dict()
    desired = {
        label: max(0, int(target_counts[label]) - chosen_counts.get(label, 0))
        for label in LABELS
    }

    # Fill toward the requested class counts first.
    additions = []
    for label in LABELS:
        count = min(
            desired[label],
            int((remaining["label"] == label).sum()),
        )
        if count:
            additions.append(
                remaining[remaining["label"] == label].sample(
                    n=count,
                    replace=False,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            )

    if additions:
        extra = pd.concat(additions)
        chosen = pd.concat([chosen, extra])
        remaining = remaining[~remaining["filename"].isin(extra["filename"])]

    # If requested class counts did not fill the split, fill remaining slots
    # by the largest normalized class deficit, respecting available images.
    while len(chosen) < n:
        current = chosen["label"].value_counts().to_dict()
        available = remaining["label"].value_counts().to_dict()
        candidates = [
            label for label in LABELS if available.get(label, 0) > 0
        ]
        if not candidates:
            return None
        label = max(
            candidates,
            key=lambda x: (
                (target_counts[x] - current.get(x, 0))
                / max(target_counts[x], 1),
                available.get(x, 0),
            ),
        )
        one = remaining[remaining["label"] == label].sample(
            n=1,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        chosen = pd.concat([chosen, one])
        remaining = remaining[remaining["filename"] != one.iloc[0]["filename"]]

    return chosen.sample(
        frac=1,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )


def class_deviation(actual, target):
    # Normalized absolute error; avoids the Normal class dominating by size.
    return sum(
        abs(actual.get(label, 0) - target[label]) / max(target[label], 1)
        for label in LABELS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Search candidate paired patient-held-out validation/test cohorts."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    df = pd.read_csv(input_path)
    required = {"filename", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    df = df[list(required)].copy()
    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found in the input CSV.")
    if not set(df["label"].unique()).issubset(set(LABELS)):
        raise ValueError(f"Unexpected labels: {sorted(set(df['label']) - set(LABELS))}")

    patient_ids = sorted(df["patient_id"].unique())
    if len(patient_ids) != 37:
        print(f"WARNING: expected 37 patients, found {len(patient_ids)}")

    by_patient = {
        pid: df[df["patient_id"] == pid].copy()
        for pid in patient_ids
    }

    rng = np.random.default_rng(args.seed)
    candidates = []
    seen_pairs = set()

    for trial in range(args.trials):
        selected = rng.choice(patient_ids, size=12, replace=False).tolist()
        rng.shuffle(selected)
        val_patients = tuple(sorted(selected[:6]))
        test_patients = tuple(sorted(selected[6:]))

        # Keep validation/test roles distinct; reverse assignment is a
        # different candidate because the target sizes differ.
        signature = (val_patients, test_patients)
        if signature in seen_pairs:
            continue
        seen_pairs.add(signature)

        val_pool = pd.concat([by_patient[p] for p in val_patients])
        test_pool = pd.concat([by_patient[p] for p in test_patients])

        if len(val_pool) < 100 or len(test_pool) < 106:
            continue

        val_rows = choose_rows(val_pool, 100, TARGET_VAL, rng, val_patients)
        if val_rows is None:
            continue
        test_rows = choose_rows(test_pool, 106, TARGET_TEST, rng, test_patients)
        if test_rows is None:
            continue

        val_counts = val_rows["label"].value_counts().to_dict()
        test_counts = test_rows["label"].value_counts().to_dict()

        # Remaining images of validation/test patients are the leakage pool
        # available to image-level training.
        heldout_patients = set(val_patients) | set(test_patients)
        eval_filenames = set(val_rows["filename"]) | set(test_rows["filename"])
        heldout_remainder = df[
            df["patient_id"].isin(heldout_patients)
            & ~df["filename"].isin(eval_filenames)
        ]

        patient_train = df[~df["patient_id"].isin(heldout_patients)]
        patient_train_counts = patient_train["label"].value_counts().to_dict()

        # Need at least one unused image from every held-out patient for a
        # meaningful image-level leakage contrast.
        leftover_by_patient = heldout_remainder.groupby("patient_id").size()
        if len(leftover_by_patient) != 12 or (leftover_by_patient < 1).any():
            continue

        # Image-level can draw from all non-evaluation images, including
        # leftover images from the 12 validation/test patients.
        image_train_pool = df[~df["filename"].isin(eval_filenames)]
        train_n = len(patient_train)
        if len(image_train_pool) < train_n:
            continue

        # The patient-level training set is the full set of images from the
        # other 25 patients. For the paired image-level arm, we will later
        # sample exactly train_n images while forcing at least one leakage
        # image from each of the 12 held-out patients.
        leakage_images = heldout_remainder.copy()
        leakage_n = len(leakage_images)
        if leakage_n < 12:
            continue

        # Candidate quality: class composition in validation/test, plus
        # reasonable patient-level training class composition.
        val_err = class_deviation(val_counts, TARGET_VAL)
        test_err = class_deviation(test_counts, TARGET_TEST)

        # Avoid candidates with an extremely small training set or very
        # little leakage opportunity.
        train_size_penalty = abs(train_n - 464) / 464
        leakage_penalty = 0.0 if leakage_n >= 60 else (60 - leakage_n) / 60

        # Penalize class imbalance in the patient-level training cohort
        # relative to the original full dataset proportions.
        full_counts = df["label"].value_counts().to_dict()
        full_n = len(df)
        train_n_actual = len(patient_train)
        train_prop_error = 0.0
        for label in LABELS:
            full_prop = full_counts.get(label, 0) / full_n
            train_prop = patient_train_counts.get(label, 0) / max(train_n_actual, 1)
            train_prop_error += abs(train_prop - full_prop)

        score = (
            2.0 * val_err
            + 2.0 * test_err
            + 0.75 * train_prop_error
            + 0.5 * train_size_penalty
            + 0.5 * leakage_penalty
        )

        candidates.append({
            "score": score,
            "trial": trial,
            "val_patients": ";".join(val_patients),
            "test_patients": ";".join(test_patients),
            "val_pool_images": len(val_pool),
            "test_pool_images": len(test_pool),
            "val_images": len(val_rows),
            "test_images": len(test_rows),
            "val_Benign": val_counts.get("Benign", 0),
            "val_Malignant": val_counts.get("Malignant", 0),
            "val_Normal": val_counts.get("Normal", 0),
            "test_Benign": test_counts.get("Benign", 0),
            "test_Malignant": test_counts.get("Malignant", 0),
            "test_Normal": test_counts.get("Normal", 0),
            "patient_train_images": train_n,
            "patient_train_Benign": patient_train_counts.get("Benign", 0),
            "patient_train_Malignant": patient_train_counts.get("Malignant", 0),
            "patient_train_Normal": patient_train_counts.get("Normal", 0),
            "image_level_leakage_images_available": leakage_n,
            "image_level_leakage_patients": len(leftover_by_patient),
            "val_class_error": val_err,
            "test_class_error": test_err,
            "train_class_prop_error": train_prop_error,
        })

    if not candidates:
        raise RuntimeError(
            "No valid candidates found. Increase --trials or relax target counts."
        )

    result = pd.DataFrame(candidates).sort_values("score").head(args.top)
    out_dir = PROJECT_ROOT / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top_paired_split_candidates.csv"
    result.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(f"\nValid candidates: {len(candidates)}")
    print(f"Saved top {len(result)} candidates to: {out_path}")
    print("\nTop candidates:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
