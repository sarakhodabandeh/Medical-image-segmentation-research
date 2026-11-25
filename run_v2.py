"""Run the v2 pipeline: train the 30 U-Net runs, evaluate them, run the statistics.

Design: patient-grouped 5-fold paired splits (data/processed/kfold_splits/), two arms
(patient-level, image-level) and three training seeds per fold and arm, all with
configs/v2_unet.yaml. Run folders: results/v2/unet/{arm}/fold{k}/train_seed{s}/.

Finished runs (training_complete.txt present) are skipped, so the whole pipeline can be stopped
and restarted at any time. A run folder without that marker was interrupted: it is renamed to
<folder>_interrupted_<time> and the run is trained again from the start.
Runs are ordered by training seed, then fold, so partial results are balanced across folds.

Steps after training only read the result files (the example figure also needs the
checkpoints), so `python run_v2.py --skip-training` regenerates every table and figure.

Usage: python run_v2.py [--skip-training] [--dry-run] [--train-seeds 0 1 2] [--folds 0 1 2 3 4]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ["patient", "image"]
MODEL = "unet"
CONFIG = "configs/v2_unet.yaml"


def run(command, dry_run):
    print("+", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run the v2 pipeline.")
    parser.add_argument("--skip-training", action="store_true", help="reuse existing checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    print("== 1. Check the k-fold split files")
    run([sys.executable, "-m", "pytest", "-q", "tests/test_kfold_splits.py"], args.dry_run)

    print("\n== 2. Train")
    total = len(args.train_seeds) * len(args.folds) * len(ARMS)
    done = 0
    for train_seed in args.train_seeds:
        for fold in args.folds:
            for arm in ARMS:
                done += 1
                output_dir = f"results/v2/{MODEL}/{arm}/fold{fold}/train_seed{train_seed}"
                folder = ROOT / output_dir
                if args.skip_training or (folder / "training_complete.txt").exists():
                    print(f"[{done}/{total}] skip {output_dir}")
                    continue
                if folder.exists() and not args.dry_run:
                    moved = folder.with_name(
                        f"{folder.name}_interrupted_{time.strftime('%Y%m%d_%H%M%S')}"
                    )
                    folder.rename(moved)
                    print(f"[{done}/{total}] interrupted run moved to {moved.relative_to(ROOT)}")
                print(f"[{done}/{total}] {output_dir}  ({time.strftime('%H:%M:%S')})")
                run(
                    [
                        sys.executable,
                        "scripts/training/train_unet.py",
                        "--config",
                        CONFIG,
                        "--model",
                        MODEL,
                        "--split-csv",
                        f"data/processed/kfold_splits/fold{fold}_{arm}.csv",
                        "--train-seed",
                        str(train_seed),
                        "--output-dir",
                        output_dir,
                    ],
                    args.dry_run,
                )

    print("\n== 3. Evaluate (validation-chosen threshold, valid region and full canvas)")
    run([sys.executable, "scripts/evaluation/evaluate_v2.py", "--model", MODEL], args.dry_run)

    print("\n== 4. Statistics")
    run([sys.executable, "scripts/analysis/analyze_v2.py"], args.dry_run)

    print("\n== 5. Figures and LaTeX tables")
    run([sys.executable, "scripts/visualization/figure_v2_paired_differences.py"], args.dry_run)
    run([sys.executable, "scripts/visualization/figure_v2_examples.py"], args.dry_run)
    run([sys.executable, "scripts/analysis/latex_tables_v2.py"], args.dry_run)

    print(
        "\nDone. Results are in results/v2/evaluation/, results/v2/analysis/, results/v2/figures/"
    )
    print("and results/v2/paper/.")


if __name__ == "__main__":
    main()
