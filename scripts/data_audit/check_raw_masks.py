
import csv
import zipfile
from io import BytesIO
from pathlib import Path

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


ROOT = find_project_root()

ZIP_PATH = (
    ROOT
    / "data"
    / "raw"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

ZIP_ROOT = (
    "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"
)

SAMPLES = ["ANFO_007.png", "FUHI_003.png", "CHSP_010.png"]


def main():
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        info_path = f"{ZIP_ROOT}/INFO.csv"

        with archive.open(info_path) as file:
            reader = csv.DictReader(
                (line.decode("utf-8-sig") for line in file),
                delimiter=";",
            )
            metadata = {
                row["Image"]: row
                for row in reader
            }

        for filename in SAMPLES:
            if filename not in metadata:
                print(f"Not found in INFO.csv: {filename}")
                continue

            image_path = f"{ZIP_ROOT}/images/{filename}"
            mask_path = f"{ZIP_ROOT}/masks/{filename}"

            with Image.open(
                BytesIO(archive.read(image_path))
            ) as im:
                image = np.array(im.convert("RGB"))

            with Image.open(
                BytesIO(archive.read(mask_path))
            ) as im:
                mask = np.array(im.convert("RGB"))

            print(f"\n{filename}")
            print("Label:", metadata[filename]["Label"])
            print("Image dimensions (W,H):", Image.fromarray(image).size)
            print("Mask dimensions (W,H):", Image.fromarray(mask).size)
            print("Mask colors:", np.unique(
                mask.reshape(-1, 3), axis=0
            )[:10])

            if image.shape[:2] != mask.shape[:2]:
                print("DIMENSION MISMATCH — not overlaying this sample.")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title(f"Original: {filename}")

            axes[1].imshow(mask)
            axes[1].set_title("Raw dataset mask")

            axes[2].imshow(image)
            axes[2].imshow(
                np.any(mask != 0, axis=2),
                cmap="Reds",
                alpha=0.45,
            )
            axes[2].set_title("Mask overlay")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
