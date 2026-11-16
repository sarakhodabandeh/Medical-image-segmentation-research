import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


from src.utils.paths import PROJECT_ROOT

from src.datasets.bus_uclm import BUSUCLMDataset


def test_dataset():
    dataset = BUSUCLMDataset()

    print("Dataset size:", len(dataset))
    print(
        "Unique patients:",
        len({sample["patient_id"] for sample in dataset.samples})
    )

    sample = dataset[0]

    print("\nFirst sample:")
    print("Filename:", sample["filename"])
    print("Patient ID:", sample["patient_id"])
    print("Label:", sample["label"])
    print("Image shape:", tuple(sample["image"].shape))
    print("Mask shape:", tuple(sample["mask"].shape))
    print(
        "Image range:",
        sample["image"].min().item(),
        sample["image"].max().item()
    )
    print("Mask values:", torch.unique(sample["mask"]))

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    print("\nFirst batch:")
    print("Image batch shape:", tuple(batch["image"].shape))
    print("Mask batch shape:", tuple(batch["mask"].shape))
    print("Filenames:", batch["filename"])
