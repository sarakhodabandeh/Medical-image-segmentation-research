"""Checks for the BUS-UCLM dataset class. Skipped when the dataset zip is not in data/raw/."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.bus_uclm import DEFAULT_ZIP_PATH, BUSUCLMDataset

pytestmark = pytest.mark.skipif(
    not DEFAULT_ZIP_PATH.exists(), reason="BUS-UCLM zip not found in data/raw/"
)


@pytest.fixture(scope="module")
def dataset():
    return BUSUCLMDataset()


def test_dataset_size_and_patient_count(dataset):
    assert len(dataset) == 670
    assert len({sample["patient_id"] for sample in dataset.samples}) == 37


def test_sample_contents(dataset):
    sample = dataset[0]

    assert sample["label"] in {"Benign", "Malignant", "Normal"}
    assert sample["image"].shape[0] == 3
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[1:] == sample["mask"].shape[1:]
    assert sample["image"].min() >= 0.0
    assert sample["image"].max() <= 1.0
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}


def test_batch_loading(dataset):
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    assert batch["image"].shape[0] == 4
    assert batch["mask"].shape[0] == 4
    assert len(batch["filename"]) == 4


def test_master_split_is_patient_disjoint():
    patients = {}
    filenames = []
    for split in ("train", "val", "test"):
        split_dataset = BUSUCLMDataset(split=split)
        patients[split] = {sample["patient_id"] for sample in split_dataset.samples}
        filenames.extend(sample["filename"] for sample in split_dataset.samples)

    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])
    assert len(filenames) == 670
    assert len(set(filenames)) == 670
