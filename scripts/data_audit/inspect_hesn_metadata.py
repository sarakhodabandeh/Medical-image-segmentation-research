
import zipfile
import csv
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

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

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    csv_path = f"{ROOT}/INFO.csv"

    with archive.open(csv_path, "r") as file:
        text_file = (
            line.decode("utf-8-sig")
            for line in file
        )

        reader = csv.DictReader(text_file, delimiter=";")

        print("INFO.csv columns:")
        print(reader.fieldnames)
        print("\nHESN records:\n")

        for row in reader:
            filename = row.get("Image", "")

            if filename.startswith("HESN_"):
                print(f"\n{filename}")
                for key, value in row.items():
                    print(f"  {key}: {value}")
