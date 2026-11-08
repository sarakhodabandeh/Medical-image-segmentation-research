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

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()

INPUT = PROJECT_ROOT / "results" / "test_evaluation" / "test_per_image_metrics.csv"
OUTPUT = PROJECT_ROOT / "results" / "sensitivity_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

# Pair image-level and patient-level predictions for the same test images
keys = ["seed", "filename", "patient_id", "label"]

image = df[df["split_type"] == "image"].copy()
patient = df[df["split_type"] == "patient"].copy()

paired = image.merge(
    patient,
    on=keys,
    suffixes=("_image", "_patient"),
    validate="one_to_one"
)

metrics = {
    "lesion_dice": ("dice", "lesion"),
    "lesion_iou": ("iou", "lesion"),
    "normal_fp_rate": ("normal_false_positive_image", "normal"),
    "normal_fp_area": ("false_positive_area_fraction", "normal"),
}

all_results = []

for metric_name, (column, group_type) in metrics.items():
    if group_type == "lesion":
        data = paired[paired["label"].isin(["Benign", "Malignant"])].copy()
        data = data.dropna(subset=[f"{column}_image", f"{column}_patient"])
    else:
        data = paired[paired["label"] == "Normal"].copy()

    data["difference"] = (
        data[f"{column}_patient"] - data[f"{column}_image"]
    )

    # 1. Equal weight per seed
    seed_means = data.groupby("seed")["difference"].mean()
    seed_weighted = seed_means.mean()

    # 2. Equal weight per patient
    patient_seed = (
        data.groupby(["patient_id", "seed"])["difference"]
        .mean()
        .reset_index()
    )
    patient_means = patient_seed.groupby("patient_id")["difference"].mean()
    patient_weighted = patient_means.mean()

    # 3. Equal weight per unique image
    image_means = data.groupby("filename")["difference"].mean()
    unique_image_weighted = image_means.mean()

    all_results.append({
        "metric": metric_name,
        "n_seeds": data["seed"].nunique(),
        "n_patient_seed_pairs": len(patient_seed),
        "n_unique_patients": patient_means.size,
        "n_unique_images": image_means.size,
        "equal_weight_per_seed": seed_weighted,
        "equal_weight_per_patient": patient_weighted,
        "equal_weight_per_unique_image": unique_image_weighted,
    })

    print(f"\n--- {metric_name} ---")
    print(f"Equal weight per seed:        {seed_weighted:.6f}")
    print(f"Equal weight per patient:     {patient_weighted:.6f}")
    print(f"Equal weight per unique image:{unique_image_weighted:.6f}")

results = pd.DataFrame(all_results)
results.to_csv(OUTPUT / "aggregation_sensitivity.csv", index=False)

print("\nSaved:")
print(OUTPUT / "aggregation_sensitivity.csv")
print("\nFull results:")
print(results.to_string(index=False))
