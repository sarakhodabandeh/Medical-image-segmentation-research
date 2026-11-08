
import zipfile
import csv
from pathlib import Path
from io import BytesIO
from collections import Counter

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


def read_image(archive, path):
    with Image.open(BytesIO(archive.read(path))) as im:
        return im.copy()


with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    names = set(archive.namelist())

    with archive.open(f"{ROOT}/INFO.csv", "r") as f:
        rows = csv.DictReader(
            (line.decode("utf-8-sig") for line in f),
            delimiter=";"
        )
        metadata = list(rows)

    summary = Counter()
    label_counts = Counter()
    mismatches = []
    missing_files = []
    mask_color_counts = Counter()

    for row in metadata:
        filename = row["Image"]
        label = row["Label"]

        image_path = f"{ROOT}/images/{filename}"
        mask_path = f"{ROOT}/masks/{filename}"

        label_counts[label] += 1

        if image_path not in names or mask_path not in names:
            missing_files.append(filename)
            summary["missing_files"] += 1
            continue

        image = read_image(archive, image_path).convert("RGB")
        mask = read_image(archive, mask_path).convert("RGB")

        image_size = image.size
        mask_size = mask.size

        if image_size != mask_size:
            mismatches.append(
                (filename, label, image_size, mask_size)
            )
            summary["dimension_mismatch"] += 1
        else:
            summary["matching_dimensions"] += 1

        mask_array = np.array(mask)
        foreground = np.any(mask_array != 0, axis=2)
        foreground_pixels = int(foreground.sum())

        if foreground_pixels == 0:
            summary["empty_masks"] += 1
        else:
            summary["nonempty_masks"] += 1

        colors = {
            tuple(int(v) for v in color)
            for color in np.unique(
                mask_array.reshape(-1, 3), axis=0
            )
        }

        for color in colors:
            mask_color_counts[color] += 1

    print("\n=== DATASET VALIDATION REPORT ===")
    print("Metadata records:", len(metadata))
    print("Labels:", dict(label_counts))
    print("\nFile checks:")
    print("Missing image/mask pairs:", len(missing_files))
    print("Matching dimensions:", summary["matching_dimensions"])
    print("Dimension mismatches:", summary["dimension_mismatch"])
    print("Empty masks:", summary["empty_masks"])
    print("Non-empty masks:", summary["nonempty_masks"])

    print("\nDistinct RGB colors found across masks:")
    for color, count in mask_color_counts.most_common():
        print(f"  {color}: present in {count} masks")

    print("\n=== DIMENSION MISMATCHES ===")
    for filename, label, image_size, mask_size in mismatches:
        print(
            f"{filename} | label={label} | "
            f"image={image_size} | mask={mask_size}"
        )

    print("\n=== MISSING FILES ===")
    for filename in missing_files:
        print(filename)

    print("\nValidation completed. No files were modified.")
