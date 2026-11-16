import sys
from pathlib import Path
from collections import Counter


from src.utils.paths import PROJECT_ROOT

from src.datasets.bus_uclm import BUSUCLMDataset

def test_splits():
    datasets = {
        "train": BUSUCLMDataset(split="train"),
        "val": BUSUCLMDataset(split="val"),
        "test": BUSUCLMDataset(split="test"),
    }

    all_filenames = []
    all_patients = {}

    for split, dataset in datasets.items():
        labels = Counter(
            sample["label"] for sample in dataset.samples
        )
        patients = {
            sample["patient_id"] for sample in dataset.samples
        }

        print(
            f"{split}: {len(dataset)} images, "
            f"{len(patients)} patients, labels={dict(labels)}"
        )

        all_filenames.extend(
            sample["filename"] for sample in dataset.samples
        )

        all_patients[split] = patients

    assert all_patients["train"].isdisjoint(all_patients["val"])
    assert all_patients["train"].isdisjoint(all_patients["test"])
    assert all_patients["val"].isdisjoint(all_patients["test"])

    assert len(all_filenames) == 670
    assert len(set(all_filenames)) == 670

    # Check that an image and mask can be loaded.
    sample = datasets["train"][0]

    print("\nSample filename:", sample["filename"])
    print("Image shape:", tuple(sample["image"].shape))
    print("Mask shape:", tuple(sample["mask"].shape))
    print("Image range:", sample["image"].min().item(),
          sample["image"].max().item())
    print("Mask values:", sample["mask"].unique().tolist())

    print("\nAll split tests PASSED!")
