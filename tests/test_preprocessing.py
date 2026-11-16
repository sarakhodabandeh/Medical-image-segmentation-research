import sys
from pathlib import Path

import torch


from src.utils.paths import PROJECT_ROOT

from src.datasets.preprocessing import resize_with_padding


def test_preprocessing():
    images = torch.rand(2, 3, 606, 856)
    masks = torch.zeros(2, 1, 606, 856)

    masks[0, 0, 200:400, 300:500] = 1

    resized_images, resized_masks = resize_with_padding(
        images,
        masks,
        size=256,
    )

    print("Image shape:", tuple(resized_images.shape))
    print("Mask shape:", tuple(resized_masks.shape))
    print(
        "Image min/max:",
        resized_images.min().item(),
        resized_images.max().item()
    )
    print("Mask values:", torch.unique(resized_masks))
    print("Foreground fraction:", resized_masks.mean().item())

    assert resized_images.shape == (2, 3, 256, 256)
    assert resized_masks.shape == (2, 1, 256, 256)
    assert set(torch.unique(resized_masks).tolist()).issubset({0.0, 1.0})

    print("\nPreprocessing tests passed.")
