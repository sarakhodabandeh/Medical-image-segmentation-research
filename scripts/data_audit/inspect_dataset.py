
import zipfile
import csv
from pathlib import Path
from io import BytesIO
from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage

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

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH.resolve()}")

def read_mask(archive, path):
    with Image.open(BytesIO(archive.read(path))) as image:
        return np.array(image.convert("L")) > 0

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    with archive.open(f"{ROOT}/INFO.csv") as file:
        rows = list(csv.DictReader(
            (line.decode("utf-8-sig") for line in file),
            delimiter=";"
        ))

    summary = Counter()
    empty_by_label = Counter()
    nonempty_by_label = Counter()
    anomalies = []
    mask_areas = []

    for row in rows:
        filename = row["Image"]
        label = row["Label"].strip()

        image_path = f"{ROOT}/images/{filename}"
        mask_path = f"{ROOT}/masks/{filename}"

        if image_path not in archive.namelist():
            anomalies.append((filename, "Missing image"))
            continue

        if mask_path not in archive.namelist():
            anomalies.append((filename, "Missing mask"))
            continue

        with Image.open(BytesIO(archive.read(image_path))) as im:
            image_size = im.size

        mask = read_mask(archive, mask_path)

        if mask.shape[::-1] != image_size:
            anomalies.append((
                filename,
                f"Size mismatch: image={image_size}, mask={mask.shape[::-1]}"
            ))
            continue

        foreground = int(mask.sum())
        total_pixels = mask.size
        area_fraction = foreground / total_pixels

        summary[label] += 1
        mask_areas.append((filename, label, area_fraction))

        if foreground == 0:
            empty_by_label[label] += 1
        else:
            nonempty_by_label[label] += 1

        if label == "Normal" and foreground > 0:
            anomalies.append((
                filename,
                "Normal image has non-empty mask"
            ))

        if label in ("Benign", "Malignant") and foreground == 0:
            anomalies.append((
                filename,
                f"{label} image has empty mask"
            ))

    print("=== DATASET SUMMARY ===")
    print("CSV rows processed:", sum(summary.values()))
    print("\nLabel counts:")
    for label, count in sorted(summary.items()):
        print(f"  {label}: {count}")

    print("\nEmpty masks by label:")
    for label, count in sorted(empty_by_label.items()):
        print(f"  {label}: {count}")

    print("\nNon-empty masks by label:")
    for label, count in sorted(nonempty_by_label.items()):
        print(f"  {label}: {count}")

    print("\n=== MASK CONNECTED COMPONENTS ===")
    component_counts = Counter()

    for filename, label, _ in mask_areas:
        mask_path = f"{ROOT}/masks/{filename}"
        mask = read_mask(archive, mask_path)
        components, number = ndimage.label(mask)
        component_counts[number] += 1

    print("Number of connected components per mask:")
    for count, number_of_masks in sorted(component_counts.items()):
        print(f"  {count} components: {number_of_masks} masks")

    print("\n=== MASK AREA FRACTIONS ===")
    fractions = [area for _, _, area in mask_areas]

    if fractions:
        print(f"Minimum: {min(fractions):.6f}")
        print(f"Median:  {np.median(fractions):.6f}")
        print(f"Maximum: {max(fractions):.6f}")

        print("\nLargest mask fractions:")
        for filename, label, fraction in sorted(
            mask_areas, key=lambda item: item[2], reverse=True
        )[:10]:
            print(f"  {filename} | {label} | {fraction:.2%}")

    print("\n=== ANOMALIES ===")
    print("Total anomalies:", len(anomalies))
    for filename, issue in anomalies[:50]:
        print(f"  {filename}: {issue}")

    if len(anomalies) > 50:
        print(f"  ... and {len(anomalies) - 50} more")

        print("\n=== INFO.CSV STRUCTURE ===")

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(f"{ROOT}/INFO.csv") as file:
                reader = csv.DictReader(
                    (line.decode("utf-8-sig") for line in file),
                    delimiter=";"
                )

                print("Columns:", reader.fieldnames)

                for row in list(reader)[:15]:
                    print(row)


