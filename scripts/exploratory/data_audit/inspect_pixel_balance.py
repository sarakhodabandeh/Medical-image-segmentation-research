from collections import Counter

import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.bus_uclm import BUSUCLMDataset


def main():
    dataset = BUSUCLMDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    total_pixels = 0
    foreground_pixels = 0
    empty_masks = 0
    image_count = 0
    class_counts = Counter()

    for batch in loader:
        masks = batch["mask"]
        masks = F.interpolate(
            masks,
            size=(256, 256),
            mode="nearest",
        )

        foreground_pixels += int(masks.sum().item())
        total_pixels += masks.numel()
        image_count += masks.size(0)

        for label, mask in zip(batch["label"], masks):
            class_counts[label] += 1
            if mask.sum().item() == 0:
                empty_masks += 1

    background_pixels = total_pixels - foreground_pixels

    positive_fraction = foreground_pixels / total_pixels
    negative_fraction = background_pixels / total_pixels

    print("Training images:", image_count)
    print("Class counts:", dict(class_counts))
    print("Images with empty masks:", empty_masks)
    print("Total pixels:", total_pixels)
    print("Foreground pixels:", foreground_pixels)
    print("Background pixels:", background_pixels)
    print(f"Foreground fraction: {positive_fraction:.6f}")
    print(f"Background fraction: {negative_fraction:.6f}")

    if foreground_pixels > 0:
        print(
            "Raw positive weight estimate:",
            background_pixels / foreground_pixels,
        )


if __name__ == "__main__":
    main()
