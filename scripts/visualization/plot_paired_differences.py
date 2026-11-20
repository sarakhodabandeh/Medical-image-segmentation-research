"""Plot the paired differences (patient-level minus image-level) for each split seed.

Input: test_per_image_metrics.csv (--metrics-csv) written by evaluate_paired_test.py.
Outputs, written to --output-dir: paired_differences.csv (the plotted values) and
fig1_paired_differences.png (Figure 1 of the manuscript: lesion macro-Dice and mean normal-image
false-positive area fraction).
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import PROJECT_ROOT, project_path

DEFAULT_INPUT = PROJECT_ROOT / "results" / "test_evaluation" / "test_per_image_metrics.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figures"
SEED_ORDER = [42, 123, 2025, 7, 99]
LESION_LABELS = ["Benign", "Malignant"]
MINUS = "\u2212"


def paired_differences(df):
    """Return one row per seed with patient-minus-image differences."""
    lesion = df[df["label"].isin(LESION_LABELS)]
    normal = df[df["label"] == "Normal"]
    dice = lesion.groupby(["seed", "split_type"])["dice"].mean().unstack("split_type")
    fp_area = (
        normal.groupby(["seed", "split_type"])["false_positive_area_fraction"]
        .mean()
        .unstack("split_type")
    )
    result = pd.DataFrame(
        {
            "delta_lesion_dice": dice["patient"] - dice["image"],
            "delta_normal_fp_area_fraction": fp_area["patient"] - fp_area["image"],
        }
    )
    return result.reindex(SEED_ORDER).rename_axis("seed").reset_index()


def draw(differences, path):
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.6), dpi=300)
    panels = [
        (axes[0], "delta_lesion_dice", "(a) Lesion macro-Dice", "\u0394 Dice"),
        (
            axes[1],
            "delta_normal_fp_area_fraction",
            "(b) Normal-image FP area fraction",
            "\u0394 FP area fraction",
        ),
    ]
    seeds = [str(s) for s in differences["seed"]]
    for ax, column, title, ylabel in panels:
        values = differences[column]
        colours = ["#444444" if v >= 0 else "#bbbbbb" for v in values]
        ax.bar(seeds, values, color=colours, edgecolor="black", linewidth=0.6, width=0.6)
        mean = values.mean()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(mean, color="black", linestyle="--", linewidth=0.8)
        label = f"mean {mean:+.3f}".replace("-", MINUS)
        ax.text(len(seeds) - 0.5, mean, label, va="bottom", fontsize=6.5)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlabel("Split seed")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(seeds) + 0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paired patient-minus-image differences.")
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_INPUT),
        help="Per-image metrics CSV from evaluate_paired_test.py.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT), help="Folder for the figure and the CSV."
    )
    args = parser.parse_args()

    metrics_path = project_path(args.metrics_csv)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    differences = paired_differences(pd.read_csv(metrics_path))
    differences.to_csv(output_dir / "paired_differences.csv", index=False)
    draw(differences, output_dir / "fig1_paired_differences.png")

    print(differences.round(4).to_string(index=False))
    print("\nSaved:")
    print(output_dir / "paired_differences.csv")
    print(output_dir / "fig1_paired_differences.png")


if __name__ == "__main__":
    main()
