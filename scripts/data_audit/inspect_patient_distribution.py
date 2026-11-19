"""Summarise how many images each patient contributes.

Input: data/processed/bus_uclm_patient_split.csv (the master metadata).
Output, printed to the screen: images per patient, summary statistics, and the number of patients
for each image count.
"""

import pandas as pd

from src.utils.paths import PROJECT_ROOT

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
print(patient_summary["n_images"].value_counts().sort_index().to_string())
