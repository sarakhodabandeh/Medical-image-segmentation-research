import pandas as pd
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

path = PROJECT_ROOT / "data" / "processed" / "bus_uclm_patient_split.csv"
df = pd.read_csv(path)

print("Images per patient:")
patient_summary = (
    df.groupby("patient_id")
    .agg(
        n_images=("filename", "count"),
        n_labels=("label", "nunique"),
        labels=("label", lambda x: ", ".join(sorted(set(x)))),
    )
    .sort_values("n_images", ascending=False)
)

print(patient_summary.to_string())

print("\nSummary statistics:")
print(patient_summary["n_images"].describe())

print("\nNumber of patients by image count:")
print(
    patient_summary["n_images"]
    .value_counts()
    .sort_index()
    .to_string()
)
