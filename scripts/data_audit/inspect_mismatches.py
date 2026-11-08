
import zipfile
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

ROOT = (
    "BUS-UCLM Breast ultrasound lesion segmentation dataset/"
    "BUS-UCLM"
)

samples = [
    f"HESN_{i:03d}.png"
    for i in range(13)
]

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    fig, axes = plt.subplots(
        len(samples), 3, figsize=(15, 12)
    )

    for i, filename in enumerate(samples):
        image_path = f"{ROOT}/images/{filename}"
        mask_path = f"{ROOT}/masks/{filename}"

        with Image.open(BytesIO(archive.read(image_path))) as im:
            image = im.convert("RGB").copy()

        # Preserve original RGB colors for inspection.
        with Image.open(BytesIO(archive.read(mask_path))) as im:
            original_mask = im.convert("RGB").copy()

        image_array = np.array(image)
        mask_rgb = np.array(original_mask)

        print(f"\n--- {filename} ---")
        print("Image dimensions (width, height):", image.size)
        print("Mask dimensions (width, height):", original_mask.size)

        colors, counts = np.unique(
            mask_rgb.reshape(-1, 3),
            axis=0,
            return_counts=True
        )

        print("Number of unique RGB colors:", len(colors))
        print("Most common mask colors (RGB, pixel count):")

        order = np.argsort(counts)[::-1]
        for idx in order[:10]:
            print(
                tuple(int(v) for v in colors[idx]),
                int(counts[idx])
            )

        # Diagnostic only; do not use this resized mask for training.
        resized_mask = original_mask.resize(
            image.size,
            resample=Image.Resampling.NEAREST
        )
        resized_rgb = np.array(resized_mask)

        # Treat any non-black pixel as foreground for visualization.
        foreground = np.any(resized_rgb != 0, axis=2)

        overlay = image_array.copy()
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay[foreground] = (
            0.55 * overlay[foreground].astype(np.float32)
            + 0.45 * red
        ).astype(np.uint8)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(
            f"{filename}\nImage: {image.size}"
        )

        axes[i, 1].imshow(original_mask)
        axes[i, 1].set_title(
            f"Original RGB mask: {original_mask.size}"
        )

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            "Temporary resize overlay\nAlignment not yet verified"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
