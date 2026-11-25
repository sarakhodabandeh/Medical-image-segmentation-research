# Changelog

All notable changes to this repository are listed here. The experiments and results of the
manuscript were not changed by any of these edits: the evaluation and statistics scripts were rerun
after the refactoring and reproduced the committed result files exactly.

## [Unreleased] - v2 experiments

Nothing in this section changes the v1 paper runs: every new training option is off in
`configs/paper_unet.yaml`.

### Added
- `docs/analysis_plan_v2.md`: pre-specified design, endpoints and statistical model.
- Patient-grouped 5-fold paired splits (`scripts/splits/create_kfold_paired_splits.py`,
  `data/processed/kfold_splits/`) with tests (`tests/test_kfold_splits.py`).
- `configs/v2_unet.yaml` and new optional training settings in `train_unet.py`: learning-rate
  scheduler, gradient scaling for fp16, train-time augmentation (`src/datasets/augment.py`),
  in-memory image cache, `--train-seed`, validation false-positive area fraction and learning
  rate in `training_log.csv`, and a `training_complete.txt` marker at the end of each run.
- `scripts/evaluation/evaluate_v2.py` and `scripts/analysis/analyze_v2.py` (mixed model and
  hierarchical bootstrap; adds `statsmodels` to the requirements), `run_v2.py`.
- Results of the 30 v2 runs (`results/v2/`): training logs, per-image test metrics, run
  summaries and analysis tables.
- Manuscript outputs: `scripts/visualization/figure_v2_paired_differences.py`,
  `scripts/visualization/figure_v2_examples.py` and `scripts/analysis/latex_tables_v2.py`, which
  write `results/v2/figures/` and `results/v2/paper/tables_v2.tex`; step 5 of `run_v2.py`.

## [1.0.1] - 2026-11-21

No change to the paper experiments, split files or the result tables of the manuscript.

### Changed
- Exploratory scripts write to `results/exploratory/` and `data/processed/exploratory/`
  instead of the paper's result folders; their configuration moved to
  `configs/exploratory/unet.yaml`.
- Side-experiment result files that were mixed into the paper folders moved to
  `results/exploratory/` (`test_per_image.csv`, `preprocessing_comparison/`,
  `patient_diagnostic_*.csv`, `influential_patients.csv`).
- README: the DOI badge and the archive link use the all-versions (concept) DOI
  10.5281/zenodo.24394780; the manuscript version stays 10.5281/zenodo.24394781.
- `pyproject.toml` version now matches the release.

### Removed
- `results/sensitivity_analysis/patient_overlap_*.csv`, stale copies of
  `results/patient_overlap_audit/` that the sensitivity script no longer writes.

## [1.0.0] - 2026-11-20

### Added
- `pyproject.toml` so the project installs with `pip install -e .`; one shared `PROJECT_ROOT` in
  `src/utils/paths.py`.
- `configs/paper_unet.yaml` with the settings used for the paper.
- Command-line options (`--metrics-csv`, `--output-dir`, `--split-dir`) for the statistics, audit
  and validation scripts.
- `scripts/visualization/plot_paired_differences.py` (Figure 1) and `run_paper.py` (full pipeline).
- Real pytest tests for the models, preprocessing, dataset, configuration and the paired splits.
- GitHub Actions workflow (ruff and pytest), ruff configuration, pre-commit hook, pinned
  `requirements.txt`, `requirements-dev.txt`, `.gitattributes`, `data/README.md`.
- Module docstrings for all paper scripts.
- MIT `LICENSE` and `CITATION.cff`.

### Changed
- Split generators now write inside the repository (they previously wrote two folders above it).
- Side experiments moved to `scripts/exploratory/` and `data/processed/exploratory/`; their result
  folders are no longer tracked by git.
- Code formatted with ruff; em dashes and banner comments removed.
- README rewritten to describe the repository as it is.
- `train_unet_save_epochs.py` merged into `train_unet.py` as the `--save-every-epoch` option.

### Removed
- Superseded scripts (`create_paired_splits_hardened.py`, `evaluate_test.py`,
  `validate_all_splits.py`) and two scripts that could not run (`diagnose_unet.py`,
  `evaluate_checkpoint.py`). All remain in the git history.
- The unused `tqdm` requirement and the empty `docs/` folder.
