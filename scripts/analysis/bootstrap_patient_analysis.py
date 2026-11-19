import argparse

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
from src.utils.paths import PROJECT_ROOT, project_path

INPUT = PROJECT_ROOT / "results" / "test_evaluation" / "test_per_image_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "patient_bootstrap"


N_BOOTSTRAP = 10000
RANDOM_SEED = 20261102
CONFIDENCE = 0.95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patient-level cluster bootstrap of paired split-strategy differences."
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(INPUT),
        help="Per-image metrics CSV from evaluate_paired_test.py.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Folder for the output CSVs.")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed of the bootstrap random generator.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = project_path(args.metrics_csv)
    output_dir = project_path(args.output_dir)
    n_bootstrap = args.n_bootstrap
    random_seed = args.random_seed
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(input_path)

    required = {
        "seed",
        "split_type",
        "patient_id",
        "label",
        "dice",
        "iou",
        "normal_false_positive_image",
        "false_positive_area_fraction",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # --------------------------------------------------
    # Calculate paired metrics for each patient and seed
    # --------------------------------------------------

    rows = []

    for (seed, patient_id), group in df.groupby(["seed", "patient_id"]):
        result = {
            "seed": seed,
            "patient_id": patient_id,
        }

        for split_type in ["image", "patient"]:
            g = group[group["split_type"] == split_type]

            lesions = g[g["label"].isin(["Benign", "Malignant"])]
            normals = g[g["label"] == "Normal"]

            result[f"{split_type}_lesion_images"] = len(lesions)
            result[f"{split_type}_normal_images"] = len(normals)

            result[f"{split_type}_lesion_dice"] = lesions["dice"].mean() if len(lesions) else np.nan

            result[f"{split_type}_lesion_iou"] = lesions["iou"].mean() if len(lesions) else np.nan

            result[f"{split_type}_normal_fp_rate"] = (
                normals["normal_false_positive_image"].mean() if len(normals) else np.nan
            )

            result[f"{split_type}_normal_fp_area"] = (
                normals["false_positive_area_fraction"].mean() if len(normals) else np.nan
            )

        for metric in [
            "lesion_dice",
            "lesion_iou",
            "normal_fp_rate",
            "normal_fp_area",
        ]:
            result[f"{metric}_difference_patient_minus_image"] = (
                result[f"patient_{metric}"] - result[f"image_{metric}"]
            )

        rows.append(result)

    patient_seed = pd.DataFrame(rows)

    # --------------------------------------------------
    # Aggregate repeated observations within each patient
    # --------------------------------------------------

    metrics = [
        "lesion_dice",
        "lesion_iou",
        "normal_fp_rate",
        "normal_fp_area",
    ]

    patient_rows = []

    for patient_id, group in patient_seed.groupby("patient_id"):
        row = {
            "patient_id": patient_id,
            "n_seeds": group["seed"].nunique(),
        }

        for metric in metrics:
            for split_type in ["image", "patient"]:
                col = f"{split_type}_{metric}"
                row[f"{col}_mean"] = group[col].mean()

            diff_col = f"{metric}_difference_patient_minus_image"
            row[f"{diff_col}_mean"] = group[diff_col].mean()

        patient_rows.append(row)

    patient_summary = pd.DataFrame(patient_rows)

    # --------------------------------------------------
    # Cluster bootstrap: resample unique patients
    # --------------------------------------------------

    rng = np.random.default_rng(random_seed)

    n_patients = len(patient_summary)

    if n_patients < 2:
        raise ValueError("At least two unique patients are required.")

    bootstrap_results = {}

    for metric in metrics:
        col = f"{metric}_difference_patient_minus_image_mean"

        values = patient_summary[col].dropna().to_numpy()

        if len(values) < 2:
            print(f"Skipping {metric}: insufficient patient observations.")
            continue

        observed = values.mean()

        sampled_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))

        bootstrap_means = values[sampled_indices].mean(axis=1)

        alpha = 1 - CONFIDENCE
        lower = np.quantile(bootstrap_means, alpha / 2)
        upper = np.quantile(bootstrap_means, 1 - alpha / 2)

        bootstrap_results[metric] = {
            "n_unique_patients": len(values),
            "mean_difference_patient_minus_image": observed,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "bootstrap_probability_difference_positive": (bootstrap_means > 0).mean(),
        }

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    patient_seed.to_csv(output_dir / "patient_seed_metrics.csv", index=False)

    patient_summary.to_csv(output_dir / "patient_aggregated_metrics.csv", index=False)

    bootstrap_df = pd.DataFrame(bootstrap_results).T
    bootstrap_df.to_csv(output_dir / "cluster_bootstrap_summary.csv")

    # --------------------------------------------------
    # Print report
    # --------------------------------------------------

    print("\nUnique patients:", n_patients)
    print("Patient-seed observations:", len(patient_seed))

    print("\nPatient observations per patient:")
    print(patient_summary[["patient_id", "n_seeds"]].to_string(index=False))

    print("\nCLUSTER BOOTSTRAP RESULTS")
    print(bootstrap_df.round(5).to_string())

    print("\nSaved files:")
    for path in sorted(output_dir.glob("*.csv")):
        print(path)


if __name__ == "__main__":
    main()
