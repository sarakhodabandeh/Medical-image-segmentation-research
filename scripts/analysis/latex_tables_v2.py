"""Write the LaTeX tables of the v2 manuscript from the v2 result files.

Inputs: data/processed/kfold_splits/kfold_manifest.csv, results/v2/evaluation/run_summary_v2.csv,
results/v2/analysis/paired_effects_v2.csv and auroc_v2.csv.
Output: results/v2/paper/tables_v2.tex with four table environments (fold composition, main
paired effects, sensitivity analyses, run-level ranges), included by the manuscript.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import PROJECT_ROOT

MAIN_ROWS = [
    ("dice_valid", "Lesion Dice"),
    ("iou_valid", "Lesion IoU"),
    ("recall_valid", "Lesion pixel recall"),
    ("lesion_sensitivity_valid", "Lesion detection sensitivity"),
    ("fp_area_fraction_valid", "Normal FP area fraction$^{\\dagger}$"),
    ("fp_image_valid", "Normal images with FP (any pixel)"),
    ("fp_image_valid_min10", "\\quad components $\\geq$ 10 px"),
    ("fp_image_valid_min50", "\\quad components $\\geq$ 50 px"),
    ("fp_image_valid_min100", "\\quad components $\\geq$ 100 px"),
    ("fp_components_valid", "FP components per normal image"),
]
SENSITIVITY_ROWS = [
    ("dice", "Lesion Dice"),
    ("iou", "Lesion IoU"),
    ("fp_area_fraction", "Normal FP area fraction"),
    ("fp_image", "Normal images with FP"),
]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def signed(value, digits=3):
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"
    return f"${text}$"


def ci(low, high, digits=3):
    return f"[{signed(low, digits)}, {signed(high, digits)}]"


def digits_for(metric):
    return 4 if "fp_area" in metric else 3


def p_text(value):
    if pd.isna(value):
        return "--"
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def table_folds(manifest):
    rows = []
    for fold, group in manifest.groupby("fold"):
        p = group[group["arm"] == "patient"].iloc[0]
        i = group[group["arm"] == "image"].iloc[0]
        rows.append(
            f"{fold + 1} & {p.test_patients} & {p.test_images} ({p.test_benign}/{p.test_malignant}/{p.test_normal}) "
            f"& {p.val_patients} & {p.val_images} & {p.train_images} ({p.train_benign}/{p.train_malignant}/{p.train_normal}) "
            f"& {p.train_patients} / {i.train_patients} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Patient-grouped 5-fold paired splits. Test and validation sets are identical in both arms; "
            "training sets have the same size and class composition (benign/malignant/normal) in both arms. "
            "The last column gives the number of training patients in the patient-level / image-level arm; in the "
            "image-level arm the test patients also contribute their leak-half images to training.}",
            "\\label{tab:folds}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{ccccccc}",
            "\\toprule",
            "Fold & Test patients & Test images (B/M/N) & Val.\\ patients & Val.\\ images & Training images (B/M/N) & Training patients \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
        ]
    )


def table_main(effects, auroc):
    current = effects[effects["threshold_name"] == "fixed_0.5"].set_index("metric")
    rows = []
    for metric, label in MAIN_ROWS:
        e = current.loc[metric]
        k = digits_for(metric)
        rows.append(
            f"{label} & {e.mean_patient_level:.{k}f} & {e.mean_image_level:.{k}f} & {signed(e.mixed_estimate, k)} "
            f"& {ci(e.mixed_ci_low, e.mixed_ci_high, k)} & {ci(e.bootstrap_ci_low, e.bootstrap_ci_high, k)} "
            f"& {p_text(e.mixed_p if metric == 'fp_area_fraction_valid' else e.p_holm)} \\\\"
        )
        if metric == "lesion_sensitivity_valid":
            rows.append("\\midrule")
    a = auroc.iloc[0]
    rows.append("\\midrule")
    rows.append(
        f"Image-level AUROC (normal vs lesion)$^{{\\ddagger}}$ & {a.mean_auroc_patient_level:.3f} & {a.mean_auroc_image_level:.3f} "
        f"& {signed(a.mean_difference)} & -- & {ci(a.bootstrap_ci_low, a.bootstrap_ci_high)} & -- \\\\"
    )
    lesion = current.loc["dice_valid"]
    normal = current.loc["fp_area_fraction_valid"]
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Main results (valid image region, threshold 0.5). Means are over all test-image "
            f"evaluations ({int(lesion.n_pairs)} lesion and {int(normal.n_pairs)} normal per arm: "
            f"{int(lesion.n_test_images)} lesion and {int(normal.n_test_images)} normal test images $\\times$ 3 training seeds). "
            "$\\Delta$ is patient-level minus image-level, estimated with the linear mixed model (random effects "
            "for patient, training seed and image); the bootstrap CI resamples patients and then training seeds. "
            "$p$ values are Holm-adjusted across the secondary endpoints. "
            "$^{\\dagger}$Pre-specified primary endpoint (unadjusted $p$). "
            "$^{\\ddagger}$Threshold-free; mean over the 15 runs per arm, bootstrap over the 15 paired runs.}",
            "\\label{tab:main}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Endpoint & Patient-level & Image-level & $\\Delta$ & 95\\% CI (mixed model) & 95\\% CI (bootstrap) & $p$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
        ]
    )


def table_sensitivity(effects):
    rows = []
    for stem, label in SENSITIVITY_ROWS:
        cells = []
        for threshold_name, region in (
            ("fixed_0.5", "valid"),
            ("val_tuned", "valid"),
            ("fixed_0.5", "full"),
        ):
            e = effects[
                (effects["threshold_name"] == threshold_name)
                & (effects["metric"] == f"{stem}_{region}")
            ].iloc[0]
            k = digits_for(stem)
            cells.append(f"{signed(e.mixed_estimate, k)} {ci(e.mixed_ci_low, e.mixed_ci_high, k)}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Sensitivity of the paired differences (patient minus image, mixed-model estimate and 95\\% CI) "
            "to the threshold and to the evaluation region. The validation-chosen threshold maximises validation "
            "lesion Dice for each run (range 0.20 to 0.85).}",
            "\\label{tab:sensitivity}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Endpoint & Threshold 0.5, valid region & Validation-chosen threshold, valid region & Threshold 0.5, full canvas \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/v2/paper/tables_v2.tex")
    args = parser.parse_args()

    manifest = pd.read_csv(resolve_path("data/processed/kfold_splits/kfold_manifest.csv"))
    effects = pd.read_csv(resolve_path("results/v2/analysis/paired_effects_v2.csv"))
    auroc = pd.read_csv(resolve_path("results/v2/analysis/auroc_v2.csv"))

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "% Generated by scripts/analysis/latex_tables_v2.py; do not edit by hand.\n\n"
        + "\n\n".join(
            [table_folds(manifest), table_main(effects, auroc), table_sensitivity(effects)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
