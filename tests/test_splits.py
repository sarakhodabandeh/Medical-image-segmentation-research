"""Checks of the committed paired split CSVs (no dataset zip needed).

For each seed there is a patient-level and a matched image-level split. They must share the
same validation and test images, have training sets of equal size and class composition, and
differ only in whether the held-out patients also appear in training.
"""

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "paired_splits"
SEEDS = [42, 123, 2025, 7, 99]
LABELS = {"Benign", "Malignant", "Normal"}


def load(kind, seed):
    return pd.read_csv(SPLIT_DIR / f"{kind}_level_seed_{seed}.csv")


def part(frame, name):
    return frame[frame["split"] == name]


def patients(frame, name):
    return set(part(frame, name)["patient_id"])


def rows(frame, name):
    subset = part(frame, name)
    return sorted(zip(subset["filename"], subset["patient_id"], subset["label"]))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["patient", "image"])
def test_split_file_is_well_formed(kind, seed):
    frame = load(kind, seed)

    assert list(frame.columns) == ["filename", "patient_id", "label", "split"]
    assert set(frame["split"]) == {"train", "val", "test"}
    assert set(frame["label"]) <= LABELS
    assert frame["filename"].is_unique  # no image appears in two partitions
    assert len(part(frame, "val")) == 100
    assert len(part(frame, "test")) == 106


@pytest.mark.parametrize("seed", SEEDS)
def test_validation_and_test_images_are_identical_in_both_strategies(seed):
    patient_level, image_level = load("patient", seed), load("image", seed)

    assert rows(patient_level, "val") == rows(image_level, "val")
    assert rows(patient_level, "test") == rows(image_level, "test")


@pytest.mark.parametrize("seed", SEEDS)
def test_training_sets_are_matched(seed):
    patient_level, image_level = load("patient", seed), load("image", seed)
    train_p, train_i = part(patient_level, "train"), part(image_level, "train")

    assert len(train_p) == len(train_i)
    assert train_p["label"].value_counts().to_dict() == train_i["label"].value_counts().to_dict()


@pytest.mark.parametrize("seed", SEEDS)
def test_patient_level_split_has_no_patient_overlap(seed):
    frame = load("patient", seed)
    train, val, test = (patients(frame, name) for name in ("train", "val", "test"))

    assert len(val) == 6 and len(test) == 6
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@pytest.mark.parametrize("seed", SEEDS)
def test_image_level_split_trains_on_the_held_out_patients(seed):
    frame = load("image", seed)
    held_out = patients(frame, "val") | patients(frame, "test")

    assert len(held_out) == 12
    assert held_out <= patients(frame, "train")


def test_manifest_lists_all_seeds():
    manifest = pd.read_csv(SPLIT_DIR / "paired_split_manifest.csv")

    assert sorted(manifest["seed"]) == sorted(SEEDS)
