"""Figure: paired differences (patient-level minus image-level) for the v2 runs.

One panel per endpoint (lesion Dice, normal-image false-positive area fraction, share of normal
images with a false positive), valid region, threshold 0.5. Each dot is one fold and training
seed (mean over that run's test images); the line and band are the mixed-model estimate and its
95% CI from results/v2/analysis/paired_effects_v2.csv.
Inputs: results/v2/evaluation/run_summary_v2.csv, results/v2/analysis/paired_effects_v2.csv.
Output: results/v2/figures/paired_differences_v2.pdf and .png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

PANELS = [
    ("lesion_dice_valid", "dice_valid", "Lesion Dice"),
    ("normal_fp_area_fraction_valid", "fp_area_fraction_valid", "Normal-image FP area fraction"),
    ("normal_fp_image_valid", "fp_image_valid", "Normal images with any FP"),
]
SEED_MARKERS = {0: "o", 1: "s", 2: "^"}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="results/v2/evaluation/run_summary_v2.csv")
    parser.add_argument("--effects", default="results/v2/analysis/paired_effects_v2.csv")
    parser.add_argument("--output-dir", default="results/v2/figures")
    args = parser.parse_args()

    summary = pd.read_csv(resolve_path(args.summary))
    summary = summary[summary["threshold_name"] == "fixed_0.5"]
    effects = pd.read_csv(resolve_path(args.effects))
    effects = effects[effects["threshold_name"] == "fixed_0.5"].set_index("metric")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
    for ax, (run_metric, effect_metric, title) in zip(axes, PANELS):
        wide = summary.pivot_table(index=["fold", "train_seed"], columns="arm", values=run_metric)
        diff = (wide["patient"] - wide["image"]).reset_index(name="d")
        effect = effects.loc[effect_metric]
        ax.axhspan(effect["mixed_ci_low"], effect["mixed_ci_high"], color="0.85", lw=0)
        ax.axhline(effect["mixed_estimate"], color="0.3", lw=1)
        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        for seed, marker in SEED_MARKERS.items():
            part = diff[diff["train_seed"] == seed]
            ax.scatter(
                part["fold"] + (seed - 1) * 0.12,
                part["d"],
                s=14,
                marker=marker,
                facecolor="white",
                edgecolor="black",
                lw=0.7,
                label=f"seed {seed}",
                zorder=3,
            )
        ax.set_xticks(range(5), [str(f + 1) for f in range(5)])
        ax.set_xlabel("Fold")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Patient-level minus image-level")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", handletextpad=0.2)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"paired_differences_v2.{suffix}", dpi=300)
    print(f"Saved {output_dir / 'paired_differences_v2.pdf'}")


if __name__ == "__main__":
    main()
