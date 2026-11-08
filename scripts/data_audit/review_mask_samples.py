
import zipfile
import csv
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


# Keep the same imports, ZIP_PATH, ROOT, and labels-loading code above.

selected = [
    "ANFO_007.png",
    "ANFO_006.png",
    "FUHI_003.png",
    "CHSP_010.png",
]

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    with archive.open(f"{ROOT}/INFO.csv", "r") as file:
        reader = csv.DictReader(
            (line.decode("utf-8-sig") for line in file),
            delimiter=";"
        )
        labels = {row["Image"]: row["Label"] for row in reader}

    fig, axes = plt.subplots(
        len(selected), 3, figsize=(15, 12), constrained_layout=True
    )

    for i, filename in enumerate(selected):
        image_path = f"{ROOT}/images/{filename}"
        mask_path = f"{ROOT}/masks/{filename}"

        with Image.open(BytesIO(archive.read(image_path))) as im:
            image = im.convert("RGB").copy()

        with Image.open(BytesIO(archive.read(mask_path))) as im:
            mask = im.convert("RGB").copy()

        image_array = np.array(image)
        mask_array = np.array(mask)

        if image.size != mask.size:
            print("Skipping dimension mismatch:", filename)
            for ax in axes[i]:
                ax.axis("off")
            continue

        foreground = np.any(mask_array != 0, axis=2)

        overlay = image_array.copy()
        overlay[foreground] = (
            0.55 * overlay[foreground].astype(np.float32)
            + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{filename} — Original", fontsize=10)

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title("Original mask", fontsize=10)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Overlay — {labels.get(filename, 'Unknown')}",
            fontsize=10
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.show()

