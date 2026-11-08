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

OUTPUT = PROJECT_ROOT / "results" / "leave_one_patient_out"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
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
        "lesion_dice": ("dice", "lesion"),
        "lesion_iou": ("iou", "lesion"),
        "normal_fp_rate": ("normal_false_positive_image", "normal"),
        "normal_fp_area": ("false_positive_area_fraction", "normal"),
    }

    def calculate(data):
        seed_means = data.groupby("seed")["difference"].mean()
        seed_weighted = seed_means.mean()

        patient_seed = (
            data.groupby(["patient_id", "seed"])["difference"]
            .mean()
            .reset_index()
        )
        patient_means = patient_seed.groupby("patient_id")["difference"].mean()
        patient_weighted = patient_means.mean()

        image_means = data.groupby("filename")["difference"].mean()
        unique_image_weighted = image_means.mean()

        return {
            "equal_weight_per_seed": seed_weighted,
            "equal_weight_per_patient": patient_weighted,
            "equal_weight_per_unique_image": unique_image_weighted,
        }

    all_lopo = []
    all_summary = []

    for metric_name, (column, group_type) in metrics.items():

        if group_type == "lesion":
            data = paired[
                paired["label"].isin(["Benign", "Malignant"])
            ].copy()
            data = data.dropna(
                subset=[f"{column}_image", f"{column}_patient"]
            )
        else:
            data = paired[paired["label"] == "Normal"].copy()

        data["difference"] = (
            data[f"{column}_patient"] - data[f"{column}_image"]
        )

        baseline = calculate(data)

        patients = sorted(data["patient_id"].unique())

        for excluded_patient in patients:
            reduced = data[
                data["patient_id"] != excluded_patient
            ].copy()

            result = calculate(reduced)

            row = {
                "metric": metric_name,
                "excluded_patient": excluded_patient,
                **result,
            }
            all_lopo.append(row)

        metric_lopo = pd.DataFrame([
            row for row in all_lopo if row["metric"] == metric_name
        ])

        summary = {
            "metric": metric_name,
            "n_patients": len(patients),
        }

        for method in [
            "equal_weight_per_seed",
            "equal_weight_per_patient",
            "equal_weight_per_unique_image",
        ]:
            values = metric_lopo[method]

            summary[f"{method}_baseline"] = baseline[method]
            summary[f"{method}_min"] = values.min()
            summary[f"{method}_max"] = values.max()
            summary[f"{method}_max_abs_change"] = (
                values - baseline[method]
            ).abs().max()

        all_summary.append(summary)

        print(f"\n--- {metric_name} ---")
        print("Baseline:")
        for method, value in baseline.items():
            print(f"  {method}: {value:.6f}")

        print("\nLOPO ranges:")
        for method in [
            "equal_weight_per_seed",
            "equal_weight_per_patient",
            "equal_weight_per_unique_image",
        ]:
            print(
                f"  {method}: "
                f"{metric_lopo[method].min():.6f} "
                f"to {metric_lopo[method].max():.6f}"
            )

    lopo_df = pd.DataFrame(all_lopo)
    summary_df = pd.DataFrame(all_summary)

    lopo_df.to_csv(
        OUTPUT / "leave_one_patient_out_all.csv",
        index=False
    )
    summary_df.to_csv(
        OUTPUT / "leave_one_patient_out_summary.csv",
        index=False
    )

    print("\nSaved:")
    print(OUTPUT / "leave_one_patient_out_all.csv")
    print(OUTPUT / "leave_one_patient_out_summary.csv")

    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
