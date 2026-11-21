"""Create the v2 patient-grouped 5-fold paired splits.

Design (one fold k = 0..4):
- Patients are assigned to 5 test folds with StratifiedGroupKFold (groups = patient, strata =
  image label), so every patient is a test patient in exactly one fold.
- Each test patient's images are divided, per label, into a test half (rounded up) and a leak
  half. The test half is the test set of that fold for both arms.
- The remaining (non-test) patients are divided into validation patients (one inner
  StratifiedGroupKFold fold, about 20%) and training patients. The validation set is all images
  of the validation patients and is identical for both arms.
- Patient-level arm: training = all images of the training patients.
- Image-level arm: training = the leak half of the test patients plus the training-patient
  images minus a random subset with the same label counts as the leak half, so both arms have
  training sets of the same size and class composition.

Input: the master metadata CSV (--master). Outputs, written to --output-dir:
fold{k}_patient.csv and fold{k}_image.csv (columns filename, patient_id, label, split) and
kfold_manifest.csv. The default seed fixes the assignment; rerunning with the same seed
reproduces the committed files exactly.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.paths import PROJECT_ROOT

DEFAULT_MASTER = "data/processed/bus_uclm_patient_split.csv"
DEFAULT_OUTPUT = "data/processed/kfold_splits"
LABELS = ["Benign", "Malignant", "Normal"]
N_FOLDS = 5


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def split_test_patient(rows, rng):
    """Split one patient's images into (test half, leak half), per label, test half rounded up."""
    test_parts, leak_parts = [], []
    for _, group in rows.groupby("label", sort=True):
        order = rng.permutation(len(group))
        n_test = int(np.ceil(len(group) / 2))
        test_parts.append(group.iloc[order[:n_test]])
        leak_parts.append(group.iloc[order[n_test:]])
    return pd.concat(test_parts), pd.concat(leak_parts)


def remove_matching(train, counts, rng):
    """Drop a random subset of `train` with the given label counts."""
    drop = []
    for label, n in counts.items():
        candidates = train.index[train["label"] == label].to_numpy()
        if n > len(candidates):
            raise ValueError(f"Not enough {label} training images to match the leak half.")
        drop.extend(rng.choice(candidates, size=n, replace=False))
    return train.drop(index=drop)


def make_fold(master, fold, test_patients, rng, seed):
    test_rows = master[master["patient_id"].isin(test_patients)]
    rest = master[~master["patient_id"].isin(test_patients)]

    test_half, leak_half = [], []
    for _, rows in test_rows.groupby("patient_id", sort=True):
        t, leak = split_test_patient(rows, rng)
        test_half.append(t)
        leak_half.append(leak)
    test_half, leak_half = pd.concat(test_half), pd.concat(leak_half)

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
    _, val_index = next(inner.split(rest, rest["label"], rest["patient_id"]))
    val_patients = set(rest.iloc[val_index]["patient_id"])
    val = rest[rest["patient_id"].isin(val_patients)]
    train_patient_level = rest[~rest["patient_id"].isin(val_patients)]

    leak_counts = leak_half["label"].value_counts().to_dict()
    train_image_level = pd.concat(
        [remove_matching(train_patient_level, leak_counts, rng), leak_half]
    )

    def assemble(train):
        parts = [
            train.assign(split="train"),
            val.assign(split="val"),
            test_half.assign(split="test"),
        ]
        frame = pd.concat(parts)[["filename", "patient_id", "label", "split"]]
        return frame.sort_values(["split", "filename"]).reset_index(drop=True)

    return assemble(train_patient_level), assemble(train_image_level)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    master = pd.read_csv(project_path(args.master), encoding="utf-8-sig")
    master = (
        master[["filename", "patient_id", "label"]].sort_values("filename").reset_index(drop=True)
    )
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    manifest = []
    for fold, (_, test_index) in enumerate(
        outer.split(master, master["label"], master["patient_id"])
    ):
        test_patients = sorted(set(master.iloc[test_index]["patient_id"]))
        patient_level, image_level = make_fold(master, fold, test_patients, rng, args.seed)
        patient_level.to_csv(output_dir / f"fold{fold}_patient.csv", index=False)
        image_level.to_csv(output_dir / f"fold{fold}_image.csv", index=False)

        for arm, frame in (("patient", patient_level), ("image", image_level)):
            row = {"fold": fold, "arm": arm}
            for part in ("train", "val", "test"):
                subset = frame[frame["split"] == part]
                row[f"{part}_images"] = len(subset)
                row[f"{part}_patients"] = subset["patient_id"].nunique()
                for label in LABELS:
                    row[f"{part}_{label.lower()}"] = int((subset["label"] == label).sum())
            manifest.append(row)
        print(f"fold {fold}: test patients {', '.join(test_patients)}")

    manifest = pd.DataFrame(manifest)
    manifest.to_csv(output_dir / "kfold_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
