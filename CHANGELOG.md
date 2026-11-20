# Changelog

All notable changes to this repository are listed here. The experiments and results of the
manuscript were not changed by any of these edits: the evaluation and statistics scripts were rerun
after the refactoring and reproduced the committed result files exactly.

## [Unreleased]

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

### Removed
- Superseded scripts (`create_paired_splits_hardened.py`, `evaluate_test.py`,
  `validate_all_splits.py`) and two scripts that could not run (`diagnose_unet.py`,
  `evaluate_checkpoint.py`). All remain in the git history.
- The unused `tqdm` requirement and the empty `docs/` folder.
