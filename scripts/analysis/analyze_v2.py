"""Statistical analysis of the v2 runs (pre-specified in docs/analysis_plan_v2.md).

Input: results/v2/evaluation/test_per_image_v2.csv and run_summary_v2.csv (evaluate_v2.py).
For every endpoint, the paired difference d = patient-level minus image-level is formed for each
test image, fold and training seed (same image, same fold, same training seed in both arms).
Each endpoint is then estimated two ways:
- linear mixed model d ~ 1 with a random intercept per patient and variance components for
  training seed and image (statsmodels MixedLM); the intercept is the mean paired difference;
- hierarchical bootstrap (10,000 replicates): resample patients, then training seeds within each
  resampled patient; percentile 95% CI of the patient-weighted mean difference.
The primary endpoint (normal-image false-positive area fraction, valid region, threshold 0.5) is
reported without correction; secondary endpoints get Holm-adjusted p-values.
Outputs, written to --output-dir: paired_effects_v2.csv, auroc_v2.csv, seed_variability_v2.csv
and arm_means_v2.csv.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.utils.paths import PROJECT_ROOT

LESION = ("Benign", "Malignant")
PRIMARY = ("fp_area_fraction_valid", "Normal")
KEY_SECONDARY = ("dice_valid", "Lesion")
SECONDARY = [
    ("iou_valid", "Lesion"),
    ("recall_valid", "Lesion"),
    ("lesion_sensitivity_valid", "Lesion"),
    ("fp_image_valid", "Normal"),
    ("fp_image_valid_min10", "Normal"),
    ("fp_image_valid_min50", "Normal"),
    ("fp_image_valid_min100", "Normal"),
    ("fp_components_valid", "Normal"),
]
SENSITIVITY = [
    ("dice_full", "Lesion"),
    ("iou_full", "Lesion"),
    ("fp_area_fraction_full", "Normal"),
    ("fp_image_full", "Normal"),
]
N_BOOT = 10_000


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def paired_differences(frame, metric, subset):
    rows = (
        frame[frame["label"].isin(LESION)]
        if subset == "Lesion"
        else frame[frame["label"] == "Normal"]
    )
    keys = ["fold", "train_seed", "filename", "patient_id", "threshold_name"]
    wide = rows.pivot_table(index=keys, columns="arm", values=metric, aggfunc="first").reset_index()
    wide = wide.dropna(subset=["patient", "image"])
    wide["d"] = wide["patient"] - wide["image"]
    return wide


def mixed_model(diffs):
    """Mean paired difference with 95% CI and p-value from a linear mixed model."""
    if diffs["d"].std() == 0:
        value = diffs["d"].iloc[0]
        return value, value, value, np.nan, "no variation (identical in both arms)"
    specs = [
        {"seed": "0 + C(train_seed)", "image": "0 + C(filename)"},
        {"seed": "0 + C(train_seed)"},
        None,
    ]
    for vc in specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm("d ~ 1", diffs, groups=diffs["patient_id"], vc_formula=vc)
                fit = model.fit(reml=True, method=["lbfgs", "powell"])
            if not np.isfinite(fit.bse["Intercept"]):
                continue
            low, high = fit.conf_int().loc["Intercept"]
            label = (
                "patient + seed + image"
                if vc and "image" in vc
                else ("patient + seed" if vc else "patient")
            )
            return fit.params["Intercept"], low, high, fit.pvalues["Intercept"], label
        except (ValueError, np.linalg.LinAlgError):
            continue
    return (np.nan,) * 4 + ("failed",)


def hierarchical_bootstrap(diffs, rng):
    table = diffs.groupby(["patient_id", "train_seed"])["d"].mean().unstack()
    values = table.to_numpy()  # patients x seeds, NaN where missing
    n_patients, n_seeds = values.shape
    patients = rng.integers(0, n_patients, size=(N_BOOT, n_patients))
    seeds = rng.integers(0, n_seeds, size=(N_BOOT, n_patients, n_seeds))
    sampled = values[patients[..., None], seeds]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimates = np.nanmean(np.nanmean(sampled, axis=2), axis=1)
    estimates = estimates[np.isfinite(estimates)]
    point = np.nanmean(np.nanmean(values, axis=1))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, low, high, float((estimates > 0).mean())


def analyse(frame, rng):
    endpoints = (
        [(PRIMARY, "primary")]
        + [(KEY_SECONDARY, "key secondary")]
        + [(e, "secondary") for e in SECONDARY]
        + [(e, "sensitivity (full canvas)") for e in SENSITIVITY]
    )
    rows = []
    for threshold_name in sorted(frame["threshold_name"].unique()):
        current = frame[frame["threshold_name"] == threshold_name]
        for (metric, subset), role in endpoints:
            diffs = paired_differences(current, metric, subset)
            if diffs.empty:
                continue
            estimate, low, high, p, structure = mixed_model(diffs)
            boot_point, boot_low, boot_high, share = hierarchical_bootstrap(diffs, rng)
            rows.append(
                {
                    "threshold_name": threshold_name,
                    "role": role,
                    "metric": metric,
                    "images": subset,
                    "n_patients": diffs["patient_id"].nunique(),
                    "n_test_images": diffs["filename"].nunique(),
                    "n_pairs": len(diffs),
                    "mean_patient_level": diffs["patient"].mean(),
                    "mean_image_level": diffs["image"].mean(),
                    "mean_difference": diffs["d"].mean(),
                    "mixed_estimate": estimate,
                    "mixed_ci_low": low,
                    "mixed_ci_high": high,
                    "mixed_p": p,
                    "mixed_random_effects": structure,
                    "bootstrap_patient_weighted_mean": boot_point,
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                    "bootstrap_share_above_zero": share,
                }
            )
    effects = pd.DataFrame(rows)
    effects["p_holm"] = np.nan
    for threshold_name, group in effects.groupby("threshold_name"):
        mask = group["role"].isin(["key secondary", "secondary"]) & group["mixed_p"].notna()
        if mask.any():
            adjusted = multipletests(group.loc[mask, "mixed_p"], method="holm")[1]
            effects.loc[group.index[mask], "p_holm"] = adjusted
    return effects


def auroc_differences(summary, rng):
    rows = []
    for threshold_name, group in summary.groupby("threshold_name"):
        wide = group.pivot_table(
            index=["fold", "train_seed"], columns="arm", values="image_auroc_max_probability"
        ).dropna()
        d = (wide["patient"] - wide["image"]).to_numpy()
        boot = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
        rows.append(
            {
                "threshold_name": threshold_name,
                "n_run_pairs": len(d),
                "mean_auroc_patient_level": wide["patient"].mean(),
                "mean_auroc_image_level": wide["image"].mean(),
                "mean_difference": d.mean(),
                "bootstrap_ci_low": np.percentile(boot, 2.5),
                "bootstrap_ci_high": np.percentile(boot, 97.5),
            }
        )
    # AUROC uses max probability, so it does not depend on the threshold: keep one row.
    return pd.DataFrame(rows).head(1).drop(columns="threshold_name")


def seed_variability(summary):
    metrics = [
        "lesion_dice_valid",
        "normal_fp_area_fraction_valid",
        "normal_fp_image_valid",
        "image_auroc_max_probability",
    ]
    rows = []
    for (threshold_name, arm, fold), group in summary.groupby(["threshold_name", "arm", "fold"]):
        row = {"threshold_name": threshold_name, "arm": arm, "fold": fold, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd_across_seeds"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def arm_means(summary):
    columns = [
        c
        for c in summary.columns
        if c.startswith(("lesion_", "normal_", "image_auroc", "fp_components"))
    ]
    grouped = summary.groupby(["threshold_name", "arm"])[columns]
    means = grouped.mean().add_suffix("_mean")
    sds = grouped.std(ddof=1).add_suffix("_sd_across_runs")
    return pd.concat([means, sds], axis=1).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", default="results/v2/evaluation")
    parser.add_argument("--output-dir", default="results/v2/analysis")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    frame = pd.read_csv(input_dir / "test_per_image_v2.csv")
    summary = pd.read_csv(input_dir / "run_summary_v2.csv")
    rng = np.random.default_rng(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects = analyse(frame, rng)
    effects.to_csv(output_dir / "paired_effects_v2.csv", index=False)
    auroc_differences(summary, rng).to_csv(output_dir / "auroc_v2.csv", index=False)
    seed_variability(summary).to_csv(output_dir / "seed_variability_v2.csv", index=False)
    arm_means(summary).to_csv(output_dir / "arm_means_v2.csv", index=False)

    shown = effects[effects["threshold_name"] == "fixed_0.5"][
        ["role", "metric", "mean_difference", "mixed_ci_low", "mixed_ci_high", "mixed_p", "p_holm"]
    ]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
