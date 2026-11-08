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

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()

DEFAULT_MASTER = "data/processed/bus_uclm_patient_split.csv"
DEFAULT_CANDIDATES = (
    "data/processed/paired_split_candidates/top_paired_split_candidates.csv"
)

OUTPUT_DIR = "../../data/processed/paired_splits"
LABELS = ["Benign", "Malignant", "Normal"]
TARGET_VAL = {"Benign": 26, "Malignant": 12, "Normal": 62}
TARGET_TEST = {"Benign": 31, "Malignant": 14, "Normal": 61}
SEEDS = [42, 123, 2025, 7, 99]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_patient_list(value):
    return [item for item in str(value).split(";") if item]


def exact_sample_with_patient_coverage(pool, n, target_counts, required_patients,
                                       rng, max_attempts=20000):
    """
    Select exact class counts while ensuring:
      - every assigned patient has >=1 selected image;
      - every assigned patient has >=1 unselected image left for leakage.
    """
    required_patients = list(required_patients)
    grouped = {
        pid: pool[pool["patient_id"] == pid].reset_index(drop=True)
        for pid in required_patients
    }

    if any(len(grouped[pid]) < 2 for pid in required_patients):
        return None

    for _ in range(max_attempts):
        selected_seed_rows = []
        reserved_rows = []

        for pid in required_patients:
            patient_rows = grouped[pid]
            indices = rng.choice(len(patient_rows), size=2, replace=False)
            selected_seed_rows.append(patient_rows.iloc[[indices[0]]])
            reserved_rows.append(patient_rows.iloc[[indices[1]]])

        seed_df = pd.concat(selected_seed_rows, ignore_index=True)
        reserved_df = pd.concat(reserved_rows, ignore_index=True)

        seed_counts = seed_df["label"].value_counts().to_dict()
        if any(seed_counts.get(label, 0) > target_counts[label] for label in LABELS):
            continue

        excluded = set(seed_df["filename"]) | set(reserved_df["filename"])
        available = pool[~pool["filename"].isin(excluded)]

        additions = []
        feasible = True
        for label in LABELS:
            needed = target_counts[label] - seed_counts.get(label, 0)
            label_pool = available[available["label"] == label]
            if len(label_pool) < needed:
                feasible = False
                break
            if needed:
                chosen_indices = rng.choice(len(label_pool), size=needed, replace=False)
                additions.append(label_pool.iloc[chosen_indices])

        if not feasible:
            continue

        parts = [seed_df] + additions
        selected = pd.concat(parts, ignore_index=True)

        if len(selected) != n:
            continue
        if selected["filename"].duplicated().any():
            continue

        selected_patients = set(selected["patient_id"])
        if selected_patients != set(required_patients):
            continue

        # The reserved row plus any other unused images guarantee that every
        # held-out patient contributes at least one image to image-level train.
        return selected.sample(
            frac=1,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).reset_index(drop=True)

    return None


def build_candidate(master, candidate, rng):
    val_patients = parse_patient_list(candidate["val_patients"])
    test_patients = parse_patient_list(candidate["test_patients"])

    if len(val_patients) != 6 or len(test_patients) != 6:
        return None, "expected exactly six validation and six test patients"
    if set(val_patients) & set(test_patients):
        return None, "validation and test patient groups overlap"

    val_pool = master[master["patient_id"].isin(val_patients)]
    test_pool = master[master["patient_id"].isin(test_patients)]

    val_rows = exact_sample_with_patient_coverage(
        val_pool, 100, TARGET_VAL, val_patients, rng
    )
    if val_rows is None:
        return None, "could not construct exact validation sample"

    test_rows = exact_sample_with_patient_coverage(
        test_pool, 106, TARGET_TEST, test_patients, rng
    )
    if test_rows is None:
        return None, "could not construct exact test sample"

    heldout_patients = set(val_patients) | set(test_patients)
    eval_rows = pd.concat([val_rows, test_rows], ignore_index=True)
    eval_filenames = set(eval_rows["filename"])

    patient_train = master[~master["patient_id"].isin(heldout_patients)].copy()
    leakage_pool = master[
        master["patient_id"].isin(heldout_patients)
        & ~master["filename"].isin(eval_filenames)
    ].copy()

    leakage_by_patient = leakage_pool.groupby("patient_id").size()
    if len(leakage_by_patient) != 12 or (leakage_by_patient < 1).any():
        return None, "not every held-out patient has a leftover leakage image"

    patient_train_counts = patient_train["label"].value_counts().to_dict()
    leakage_counts = leakage_pool["label"].value_counts().to_dict()

    # Include every leftover image from held-out patients in image-level train.
    # Fill the remaining per-class quota from non-held-out patients, so the
    # two training sets have exactly the same size and class distribution.
    image_train_parts = [leakage_pool]
    for label in LABELS:
        needed = (
            patient_train_counts.get(label, 0)
            - leakage_counts.get(label, 0)
        )
        if needed < 0:
            return None, (
                f"leakage pool has more {label} images than patient-level "
                "training can match"
            )

        available = patient_train[patient_train["label"] == label]
        if len(available) < needed:
            return None, f"not enough non-held-out {label} images to match"
        if needed:
            selected_indices = rng.choice(len(available), size=needed, replace=False)
            image_train_parts.append(available.iloc[selected_indices])

    image_train = pd.concat(image_train_parts, ignore_index=True)

    # Construct two CSVs with identical validation/test rows.
    patient_level = pd.concat(
        [
            patient_train.assign(split="train"),
            val_rows.assign(split="val"),
            test_rows.assign(split="test"),
        ],
        ignore_index=True,
    )
    image_level = pd.concat(
        [
            image_train.assign(split="train"),
            val_rows.assign(split="val"),
            test_rows.assign(split="test"),
        ],
        ignore_index=True,
    )

    columns = ["filename", "patient_id", "label", "split"]
    patient_level = patient_level[columns].sort_values(
        ["split", "filename"]
    ).reset_index(drop=True)
    image_level = image_level[columns].sort_values(
        ["split", "filename"]
    ).reset_index(drop=True)

    return {
        "patient_level": patient_level,
        "image_level": image_level,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "patient_train": patient_train,
        "leakage_pool": leakage_pool,
        "patient_train_counts": patient_train_counts,
        "leakage_counts": leakage_counts,
        "candidate_score": float(candidate["score"]),
        "candidate_trial": int(candidate["trial"]),
    }, None


def validate_pair(pair):
    pl = pair["patient_level"]
    il = pair["image_level"]

    pl_val = set(pl.loc[pl["split"] == "val", "filename"])
    il_val = set(il.loc[il["split"] == "val", "filename"])
    pl_test = set(pl.loc[pl["split"] == "test", "filename"])
    il_test = set(il.loc[il["split"] == "test", "filename"])

    assert pl_val == il_val, "Validation filenames differ between paired splits"
    assert pl_test == il_test, "Test filenames differ between paired splits"

    for frame_name, frame in [("patient-level", pl), ("image-level", il)]:
        if frame["filename"].duplicated().any():
            raise AssertionError(f"Duplicate filename in {frame_name}")
        for split in ["train", "val", "test"]:
            subset = frame[frame["split"] == split]
            if subset.empty:
                raise AssertionError(f"Empty {split} split in {frame_name}")

    pl_train = pl[pl["split"] == "train"]
    il_train = il[il["split"] == "train"]
    if len(pl_train) != len(il_train):
        raise AssertionError("Train image counts differ")
    if pl_train["label"].value_counts().to_dict() != il_train["label"].value_counts().to_dict():
        raise AssertionError("Train class distributions differ")

    heldout = set(pair["val_patients"]) | set(pair["test_patients"])
    if set(pl_train["patient_id"]) & heldout:
        raise AssertionError("Patient-level training contains held-out patients")
    if not heldout.issubset(set(il_train["patient_id"])):
        raise AssertionError("Image-level training does not include all held-out patients")

    # Check the paired evaluation class counts.
    for split, target in [("val", TARGET_VAL), ("test", TARGET_TEST)]:
        for frame in [pl, il]:
            actual = frame[frame["split"] == split]["label"].value_counts().to_dict()
            if any(actual.get(label, 0) != target[label] for label in LABELS):
                raise AssertionError(f"Unexpected {split} class counts: {actual}")

    return True


def select_diverse_candidates(valid_candidates, count=5, overlap_penalty=0.01):
    """
    Exhaustively select the five-candidate combination minimizing:
      sum(candidate scores) + overlap_penalty * total pairwise held-out-patient overlap.
    """
    best = None
    best_objective = float("inf")

    for combo in itertools.combinations(valid_candidates, count):
        total_score = sum(item["candidate_score"] for item in combo)
        overlap = 0
        for left, right in itertools.combinations(combo, 2):
            left_patients = set(left["val_patients"]) | set(left["test_patients"])
            right_patients = set(right["val_patients"]) | set(right["test_patients"])
            overlap += len(left_patients & right_patients)

        objective = total_score + overlap_penalty * overlap
        if objective < best_objective:
            best_objective = objective
            best = combo

    return best, best_objective


def main():
    parser = argparse.ArgumentParser(
        description="Create and validate five paired image-level/patient-level splits."
    )
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--overlap-penalty", type=float, default=0.01)
    args = parser.parse_args()

    master_path = project_path(args.master)
    candidates_path = project_path(args.candidates)
    output_dir = project_path(OUTPUT_DIR)

    if not master_path.exists():
        raise FileNotFoundError(master_path)
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)

    master = pd.read_csv(master_path)
    required = {"filename", "patient_id", "label"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master CSV missing columns: {sorted(missing)}")
    master = master[list(required)].copy()

    if master["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in master CSV")
    if set(master["label"].unique()) != set(LABELS):
        raise ValueError("Master CSV does not contain exactly the expected labels")

    candidates_df = pd.read_csv(candidates_path).head(args.candidate_pool)
    rng = np.random.default_rng(args.seed)

    valid = []
    rejected = []
    for _, candidate in candidates_df.iterrows():
        pair, reason = build_candidate(master, candidate, rng)
        if pair is None:
            rejected.append((int(candidate["trial"]), reason))
            continue
        validate_pair(pair)
        valid.append(pair)

    print(f"Candidate rows examined: {len(candidates_df)}")
    print(f"Candidates passing construction and validation: {len(valid)}")
    if rejected:
        print("Rejected candidates:")
        for trial, reason in rejected:
            print(f"  trial={trial}: {reason}")

    if len(valid) < len(SEEDS):
        raise RuntimeError(
            f"Only {len(valid)} candidates were feasible; need at least {len(SEEDS)}. "
            "Increase --candidate-pool or run the candidate search with more trials."
        )

    selected, objective = select_diverse_candidates(
        valid, count=len(SEEDS), overlap_penalty=args.overlap_penalty
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    manifest = []

    for seed, pair in zip(SEEDS, selected):
        pl_path = output_dir / f"patient_level_seed_{seed}.csv"
        il_path = output_dir / f"image_level_seed_{seed}.csv"

        pair["patient_level"].to_csv(pl_path, index=False)
        pair["image_level"].to_csv(il_path, index=False)

        pl_train = pair["patient_level"][pair["patient_level"]["split"] == "train"]
        il_train = pair["image_level"][pair["image_level"]["split"] == "train"]
        val = pair["patient_level"][pair["patient_level"]["split"] == "val"]
        test = pair["patient_level"][pair["patient_level"]["split"] == "test"]

        heldout = set(pair["val_patients"]) | set(pair["test_patients"])
        leakage_patients = set(il_train["patient_id"]) & heldout

        reports.append({
            "seed": seed,
            "candidate_trial": pair["candidate_trial"],
            "candidate_score": pair["candidate_score"],
            "train_images_both": len(pl_train),
            "train_Benign_both": int((pl_train["label"] == "Benign").sum()),
            "train_Malignant_both": int((pl_train["label"] == "Malignant").sum()),
            "train_Normal_both": int((pl_train["label"] == "Normal").sum()),
            "val_images_both": len(val),
            "val_Benign": int((val["label"] == "Benign").sum()),
            "val_Malignant": int((val["label"] == "Malignant").sum()),
            "val_Normal": int((val["label"] == "Normal").sum()),
            "test_images_both": len(test),
            "test_Benign": int((test["label"] == "Benign").sum()),
            "test_Malignant": int((test["label"] == "Malignant").sum()),
            "test_Normal": int((test["label"] == "Normal").sum()),
            "patient_level_train_patients": pl_train["patient_id"].nunique(),
            "image_level_train_patients": il_train["patient_id"].nunique(),
            "image_level_leakage_patients": len(leakage_patients),
            "image_level_leakage_images": len(pair["leakage_pool"]),
            "val_patients": ";".join(pair["val_patients"]),
            "test_patients": ";".join(pair["test_patients"]),
            "patient_level_csv": str(pl_path.relative_to(PROJECT_ROOT)),
            "image_level_csv": str(il_path.relative_to(PROJECT_ROOT)),
        })

        manifest.append({
            "seed": seed,
            "candidate_trial": pair["candidate_trial"],
            "candidate_score": pair["candidate_score"],
            "val_patients": ";".join(pair["val_patients"]),
            "test_patients": ";".join(pair["test_patients"]),
            "paired_selection_objective": objective,
        })

    report_path = output_dir / "paired_split_validation_report.csv"
    manifest_path = output_dir / "paired_split_manifest.csv"
    pd.DataFrame(reports).to_csv(report_path, index=False)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print("\nSelected five paired splits:")
    print(pd.DataFrame(reports).to_string(index=False))
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved validation report: {report_path}")
    print("\nAll five pairs passed structural validation.")
    print("No training runs were started.")


if __name__ == "__main__":
    main()
