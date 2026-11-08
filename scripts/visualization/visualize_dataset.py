import csv
import random
import zipfile
from io import BytesIO
from pathlib import Path

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

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

DATASET_ROOT = (
    "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"
)


def read_image(archive, path):
    with Image.open(BytesIO(archive.read(path))) as image:
        return np.array(image.convert("RGB"))


def read_mask(archive, path):
    with Image.open(BytesIO(archive.read(path))) as image:
        # Convert the RGB mask to a single-channel binary mask.
        gray = np.array(image.convert("L"))
        return (gray > 0).astype(np.uint8)


def main():
    random.seed(42)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        csv_path = f"{DATASET_ROOT}/INFO.csv"

        with archive.open(csv_path) as file:
            rows = list(
                csv.DictReader(
                    (line.decode("utf-8-sig") for line in file),
                    delimiter=";"
                )
            )

        # Select one example from each class.
        selected = []

        for label in ["Benign", "Malignant", "Normal"]:
            candidates = [
                row for row in rows if row["Label"] == label
            ]
            selected.append(random.choice(candidates))

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        for row_index, row in enumerate(selected):
            filename = row["Image"]
            image_path = f"{DATASET_ROOT}/images/{filename}"
            mask_path = f"{DATASET_ROOT}/masks/{filename}"

            image = read_image(archive, image_path)
            mask = read_mask(archive, mask_path)

            # Create a red overlay where the mask marks the lesion.
            overlay = image.copy()
            overlay[mask == 1] = (
                0.55 * overlay[mask == 1]
                + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)

            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(
                f"{row['Label']} — Original"
            )

            axes[row_index, 1].imshow(
                mask, cmap="gray", vmin=0, vmax=1
            )
            axes[row_index, 1].set_title(
                f"Binary mask — {int(mask.sum())} foreground pixels"
            )

            axes[row_index, 2].imshow(overlay)
            axes[row_index, 2].set_title("Mask overlay")

            for ax in axes[row_index]:
                ax.axis("off")

            print(
                f"{filename} | Label={row['Label']} "
                f"| Mask foreground pixels={int(mask.sum())}"
            )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()