import csv
import zipfile
from collections import Counter, defaultdict

from src.utils.paths import PROJECT_ROOT

ZIP_PATH = (
    PROJECT_ROOT / "data" / "raw" / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

ROOT = "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"

patients = defaultdict(list)

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    with archive.open(f"{ROOT}/INFO.csv") as file:
        rows = csv.DictReader((line.decode("utf-8-sig") for line in file), delimiter=";")

        for row in rows:
            filename = row["Image"]
            patient_id = filename.split("_")[0]
            patients[patient_id].append(row)

print("Number of patients:", len(patients))
print("Total images:", sum(len(images) for images in patients.values()))

print("\nImages and labels per patient:")
for patient_id, images in sorted(patients.items()):
    labels = Counter(row["Label"] for row in images)
    print(
        f"{patient_id}: {len(images)} images | "
        f"Benign={labels['Benign']}, "
        f"Malignant={labels['Malignant']}, "
        f"Normal={labels['Normal']}"
    )

print("\nImages per patient summary:")
counts = [len(images) for images in patients.values()]
print("Minimum:", min(counts))
print("Maximum:", max(counts))
print("Average:", round(sum(counts) / len(counts), 2))
