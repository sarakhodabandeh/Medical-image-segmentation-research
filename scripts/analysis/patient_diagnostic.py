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

INPUT = (
    PROJECT_ROOT
    / "results"
    / "test_evaluation"
    / "test_per_image_metrics.csv"
)
OUTPUT = PROJECT_ROOT / "results" / "patient_diagnostic"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

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
        "dice": ("dice", ["Benign", "Malignant"]),
        "iou": ("iou", ["Benign", "Malignant"]),
        "normal_fp_rate": ("normal_false_positive_image", ["Normal"]),
        "normal_fp_area": ("false_positive_area_fraction", ["Normal"]),
    }

    # Unique image counts and seed appearances
    counts = (
        paired.groupby(["patient_id", "label"])
        .agg(
            unique_images=("filename", "nunique"),
            seed_image_observations=("filename", "size")
        )
        .reset_index()
    )

    count_pivot = counts.pivot(
        index="patient_id",
        columns="label",
        values="unique_images"
    ).fillna(0)

    for label in ["Benign", "Malignant", "Normal"]:
        if label not in count_pivot.columns:
            count_pivot[label] = 0

    seed_counts = (
        paired.groupby("patient_id")["seed"]
        .nunique()
        .rename("n_seeds")
    )

    total_observations = (
        paired.groupby("patient_id")
        .size()
        .rename("total_seed_image_observations")
    )

    summary = count_pivot.join(seed_counts).join(total_observations)

    # Calculate per-patient, per-seed paired metrics
    for metric_name, (column, labels) in metrics.items():

        subset = paired[paired["label"].isin(labels)].copy()

        subset["image_value"] = subset[f"{column}_image"]
        subset["patient_value"] = subset[f"{column}_patient"]

        if metric_name in ["dice", "iou"]:
            subset = subset.dropna(
                subset=["image_value", "patient_value"]
            )

        subset["difference"] = (
            subset["patient_value"] - subset["image_value"]
        )

        # Average over images within each patient and seed
        per_seed = (
            subset.groupby(["patient_id", "seed"])
            .agg(
                image_split_mean=("image_value", "mean"),
                patient_split_mean=("patient_value", "mean"),
                difference=("difference", "mean")
            )
            .reset_index()
        )

        # Aggregate across seeds, giving each patient equal weight
        patient_stats = (
            per_seed.groupby("patient_id")
            .agg(
                image_mean=( "image_split_mean", "mean"),
                patient_mean=("patient_split_mean", "mean"),
                mean_difference=("difference", "mean"),
                positive_seed_count=("difference", lambda x: (x > 0).sum()),
                negative_seed_count=("difference", lambda x: (x < 0).sum()),
                zero_seed_count=("difference", lambda x: (x == 0).sum()),
                seed_observations=("seed", "nunique")
            )
        )

        patient_stats = patient_stats.rename(columns={
            "image_mean": f"{metric_name}_image_mean",
            "patient_mean": f"{metric_name}_patient_mean",
            "mean_difference": f"{metric_name}_difference",
            "positive_seed_count": f"{metric_name}_positive_seeds",
            "negative_seed_count": f"{metric_name}_negative_seeds",
            "zero_seed_count": f"{metric_name}_zero_seeds",
            "seed_observations": f"{metric_name}_n_seeds"
        })

        summary = summary.join(patient_stats)

    summary = summary.reset_index()
    summary = summary.rename(columns={"index": "patient_id"})

    summary.to_csv(
        OUTPUT / "patient_diagnostic_summary.csv",
        index=False
    )

    # Detailed patient-seed differences
    details = []

    for metric_name, (column, labels) in metrics.items():
        subset = paired[paired["label"].isin(labels)].copy()

        subset["image_value"] = subset[f"{column}_image"]
        subset["patient_value"] = subset[f"{column}_patient"]

        if metric_name in ["dice", "iou"]:
            subset = subset.dropna(
                subset=["image_value", "patient_value"]
            )

        per_seed = (
            subset.groupby(["patient_id", "seed"])
            .agg(
                image_mean=("image_value", "mean"),
                patient_mean=("patient_value", "mean")
            )
            .reset_index()
        )

        per_seed["metric"] = metric_name
        per_seed["difference"] = (
            per_seed["patient_mean"] - per_seed["image_mean"]
        )

        details.append(per_seed)

    detail_df = pd.concat(details, ignore_index=True)

    detail_df.to_csv(
        OUTPUT / "patient_diagnostic_by_seed.csv",
        index=False
    )

    print("\nPatient diagnostic summary:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT / "patient_diagnostic_summary.csv")
    print(OUTPUT / "patient_diagnostic_by_seed.csv")


if __name__ == "__main__":
    main()
