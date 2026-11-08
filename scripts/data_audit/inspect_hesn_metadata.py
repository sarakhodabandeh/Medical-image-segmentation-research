
import zipfile
import csv
from pathlib import Path

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
