"""Run the paper pipeline from split validation to figures, in order.

Steps: (1) validate the committed paired splits, (2) train the ten U-Net models
(5 seeds x image-level / patient-level), (3) evaluate them (full canvas and valid region),
(4) run the statistical analyses, (5) draw Figure 1.

Training is skipped for every run whose best_model.pt already exists, so an interrupted run can be
resumed. The split files are never regenerated. Training takes several hours on one GPU; use
--skip-training to reuse existing checkpoints. Use --dry-run to only print the commands.

Usage: python run_paper.py [--skip-training] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 2025, 7, 99]
SPLIT_TYPES = ["image", "patient"]


def run(command, dry_run):
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def script(path, *args):
    return [sys.executable, path, *args]


def main():
    parser = argparse.ArgumentParser(description="Reproduce the paper pipeline.")
    parser.add_argument("--skip-training", action="store_true", help="reuse existing checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()

    print("== 1. Validate the paired splits")
    run(script("scripts/splits/validate_paired_splits_independent.py"), args.dry_run)

    print("\n== 2. Train the paired models")
    for seed in SEEDS:
        for split_type in SPLIT_TYPES:
            output_dir = f"results/paired_full_{split_type}_{seed}"
            if args.skip_training or (ROOT / output_dir / "best_model.pt").exists():
                print(f"skip {output_dir} (checkpoint exists or training skipped)")
                continue
            split_csv = f"data/processed/paired_splits/{split_type}_level_seed_{seed}.csv"
            run(
                script(
                    "scripts/training/train_unet.py",
                    "--split-csv",
                    split_csv,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    output_dir,
                ),
                args.dry_run,
            )

    print("\n== 3. Evaluate on the held-out test sets")
    run(script("scripts/evaluation/evaluate_paired_test.py"), args.dry_run)
    run(script("scripts/evaluation/evaluate_paired_test_valid_region.py"), args.dry_run)

    print("\n== 4. Statistical analyses")
    for name in (
        "scripts/analysis/sensitivity_analysis.py",
        "scripts/analysis/bootstrap_patient_analysis.py",
        "scripts/analysis/leave_one_patient_out.py",
        "scripts/splits/audit_patient_overlap.py",
    ):
        run(script(name), args.dry_run)

    print("\n== 5. Figure 1")
    run(script("scripts/visualization/plot_paired_differences.py"), args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run finished; nothing was executed.")


if __name__ == "__main__":
    main()
