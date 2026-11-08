
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

records = []
skipped = []

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    with archive.open(f"{ROOT}/INFO.csv", "r") as file:
        reader = csv.DictReader(
            (line.decode("utf-8-sig") for line in file),
            delimiter=";"
        )

        for row in reader:
            filename = row["Image"]

            image_path = f"{ROOT}/images/{filename}"
            mask_path = f"{ROOT}/masks/{filename}"

            with Image.open(BytesIO(archive.read(image_path))) as im:
                image_size = im.size

            with Image.open(BytesIO(archive.read(mask_path))) as im:
                mask = np.array(im.convert("RGB"))

            if image_size != (mask.shape[1], mask.shape[0]):
                skipped.append(filename)
                continue

            # A pixel belongs to the lesion if it is not black.
            foreground = np.any(mask != 0, axis=2)
            lesion_pixels = int(foreground.sum())
            total_pixels = foreground.size

            fraction = lesion_pixels / total_pixels

            # Count connected components in the binary mask.
            from scipy.ndimage import label
            _, component_count = label(foreground)

            records.append({
                "filename": filename,
                "class": row["Label"],
                "lesion_pixels": lesion_pixels,
                "lesion_fraction": fraction,
                "components": int(component_count),
            })

print("Analyzed images:", len(records))
print("Skipped for dimension mismatch:", len(skipped))

print("\nMask summary by class:")
for class_name in ("Normal", "Benign", "Malignant"):
    subset = [r for r in records if r["class"] == class_name]
    fractions = [r["lesion_fraction"] for r in subset]

    print(f"\n{class_name}: {len(subset)} images")

    if fractions:
        print("  Empty masks:", sum(f == 0 for f in fractions))
        print("  Minimum lesion fraction:", round(min(fractions), 5))
        print("  Median lesion fraction:", round(float(np.median(fractions)), 5))
        print("  Maximum lesion fraction:", round(max(fractions), 5))

print("\nConnected-component counts:")
print(dict(Counter(r["components"] for r in records)))

print("\n10 largest masks:")
for r in sorted(
    records,
    key=lambda x: x["lesion_fraction"],
    reverse=True
)[:10]:
    print(
        f"{r['filename']} | {r['class']} | "
        f"fraction={r['lesion_fraction']:.4f} | "
        f"components={r['components']}"
    )

print("\n10 smallest non-empty masks:")
nonempty = [r for r in records if r["lesion_fraction"] > 0]

for r in sorted(
    nonempty,
    key=lambda x: x["lesion_fraction"]
)[:10]:
    print(
        f"{r['filename']} | {r['class']} | "
        f"fraction={r['lesion_fraction']:.6f} | "
        f"components={r['components']}"
    )
