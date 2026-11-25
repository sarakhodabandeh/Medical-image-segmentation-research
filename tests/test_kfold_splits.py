"""Checks of the committed v2 patient-grouped 5-fold paired splits (no dataset zip needed)."""

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "kfold_splits"
MASTER = PROJECT_ROOT / "data" / "processed" / "bus_uclm_patient_split.csv"
FOLDS = range(5)


def load(fold, arm):
    return pd.read_csv(SPLIT_DIR / f"fold{fold}_{arm}.csv")


def part(frame, name):
    return frame[frame["split"] == name]


def patients(frame, name):
    return set(part(frame, name)["patient_id"])


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("arm", ["patient", "image"])
def test_split_file_is_well_formed(fold, arm):
    frame = load(fold, arm)
    assert list(frame.columns) == ["filename", "patient_id", "label", "split"]
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame["filename"].is_unique


@pytest.mark.parametrize("fold", FOLDS)
def test_val_and_test_identical_in_both_arms(fold):
    p, i = load(fold, "patient"), load(fold, "image")
    for name in ("val", "test"):
        assert sorted(part(p, name)["filename"]) == sorted(part(i, name)["filename"])


@pytest.mark.parametrize("fold", FOLDS)
def test_training_sets_matched_in_size_and_labels(fold):
    p, i = part(load(fold, "patient"), "train"), part(load(fold, "image"), "train")
    assert len(p) == len(i)
    assert p["label"].value_counts().to_dict() == i["label"].value_counts().to_dict()


@pytest.mark.parametrize("fold", FOLDS)
def test_patient_overlap_properties(fold):
    p, i = load(fold, "patient"), load(fold, "image")
    test_patients = patients(p, "test")
    val_patients = patients(p, "val")
    # Patient-level arm: no held-out patient in training.
    assert not (patients(p, "train") & (test_patients | val_patients))
    # Image-level arm: every test patient also contributes training images; validation stays
    # patient-disjoint in both arms.
    assert test_patients <= patients(i, "train")
    assert not (patients(i, "train") & val_patients)
    assert not (val_patients & test_patients)


def test_every_patient_is_tested_exactly_once():
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    seen = []
    for fold in FOLDS:
        seen.extend(patients(load(fold, "patient"), "test"))
    assert sorted(seen) == sorted(master["patient_id"].unique())


def test_each_image_appears_once_per_fold_and_test_images_once_overall():
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    test_files = []
    for fold in FOLDS:
        p, i = load(fold, "patient"), load(fold, "image")
        assert set(p["filename"]) <= set(master["filename"])
        assert set(i["filename"]) <= set(master["filename"])
        test_files.extend(part(p, "test")["filename"])
    assert len(test_files) == len(set(test_files))
