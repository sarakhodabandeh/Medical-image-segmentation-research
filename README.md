# Patient-level vs image-level data splitting in breast ultrasound segmentation

![tests](https://github.com/sarakhodabandeh/Medical-image-segmentation-research/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24394780.svg)](https://doi.org/10.5281/zenodo.24394780)

Code for the paired study *Effect of Patient-Level versus Image-Level Data Splitting on Breast
Ultrasound Lesion Segmentation and Normal-Image False Positives: A Paired Evaluation on the BUS-UCLM
Dataset* (manuscript in preparation).

The study trains a U-Net on the BUS-UCLM breast ultrasound dataset with two ways of splitting the
data, by image and by patient, using five paired splits whose validation and test images are
identical. It compares lesion overlap (Dice, IoU, pixel recall) and false-positive predictions on
normal images. In these experiments the splitting strategy had no consistent effect on lesion
overlap, but patient-level splitting was associated with more false positives on normal images;
the numbers, uncertainty analyses and limitations are in the manuscript.

## Study design

- **Data:** BUS-UCLM after quality control: 670 image-mask pairs from 37 patients (170 benign,
  90 malignant, 410 normal). 13 pairs with mismatched image and mask sizes are excluded.
- **Splits:** five seeds (42, 123, 2025, 7, 99). For each seed, the patient-level and the
  image-level split share the same 100 validation and 106 test images (6 validation and 6 test
  patients). The training sets have the same size and class counts; the image-level training set also
  contains images of the held-out patients. The split files are in `data/processed/paired_splits/`.
- **Model:** a U-Net (16-32-64-128 encoder, 256-channel bottleneck) on 256 x 256 inputs, trained for
  10 epochs with Adam and a loss of 0.5 x BCE (positive weight 5) + 0.5 x Dice. The checkpoint with the
  highest validation lesion macro-Dice at threshold 0.5 is kept. Settings: `configs/paper_unet.yaml`.
- **Evaluation:** fixed threshold 0.5. Lesion macro-Dice, IoU and pixel recall on benign and malignant
  test images; false-positive image rate and false-positive area fraction on normal test images. The
  analyses are repeated inside the valid image region (padding excluded), with three aggregation
  schemes, a patient-level bootstrap (10,000 replicates) and leave-one-patient-out.

## Installation

Python 3.12 is required. Create and activate an environment, install PyTorch for your system, then
the project:

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# install PyTorch first (see https://pytorch.org/get-started/locally/ for your CUDA version),
# then the project and the development tools
pip install torch torchvision
pip install -e . -r requirements-dev.txt
```

The paper experiments used Python 3.12, torch 2.11.0+cu128 and torchvision 0.26.0+cu128 on one
NVIDIA GeForce RTX 2070 GPU. Other package versions are pinned in `requirements.txt`. The commands
below work in both shells once the environment is active.

## Data

The dataset is not redistributed. Download BUS-UCLM from its publication
([Vallez et al., *Scientific Data* 2025](https://doi.org/10.1038/s41597-025-04562-3)), follow its
license, and put the zip file, unextracted, at
`data/raw/BUS-UCLM Breast ultrasound lesion segmentation dataset.zip`. The layout of `data/` and the
meaning of each CSV are described in [`data/README.md`](data/README.md).

## Reproducing the paper

Run everything with one command (training takes several hours on a single GPU; runs whose
`best_model.pt` already exists are skipped):

```bash
python run_paper.py
```

`python run_paper.py --dry-run` prints the commands without running them, and `--skip-training`
reuses existing checkpoints. The steps can also be run one by one:

```bash
python scripts/splits/validate_paired_splits_independent.py
python scripts/training/train_unet.py --split-csv data/processed/paired_splits/patient_level_seed_42.csv --seed 42 --output-dir results/paired_full_patient_42
python scripts/evaluation/evaluate_paired_test.py
python scripts/evaluation/evaluate_paired_test_valid_region.py
python scripts/analysis/sensitivity_analysis.py
python scripts/analysis/bootstrap_patient_analysis.py
python scripts/analysis/leave_one_patient_out.py
python scripts/splits/audit_patient_overlap.py
python scripts/visualization/plot_paired_differences.py
```

The training command is shown for one run; `run_paper.py` trains all ten
(`image` and `patient` for each of the five seeds). Checkpoints are written to
`results/paired_full_{image|patient}_{seed}/` and are not committed.

| Manuscript item | Script | Output |
|---|---|---|
| Tables 1 and 2: test performance and paired differences | `scripts/evaluation/evaluate_paired_test.py` | `results/test_evaluation/test_summary_by_run.csv`, `results/test_evaluation/test_paired_comparison.csv` |
| Table 3: aggregation sensitivity | `scripts/analysis/sensitivity_analysis.py` | `results/sensitivity_analysis/aggregation_sensitivity.csv` |
| Table 4: patient-level bootstrap | `scripts/analysis/bootstrap_patient_analysis.py` | `results/patient_bootstrap/cluster_bootstrap_summary.csv` |
| Table 5: leave-one-patient-out | `scripts/analysis/leave_one_patient_out.py` | `results/leave_one_patient_out/leave_one_patient_out_summary.csv` |
| Table 6: valid-region analysis | `scripts/evaluation/evaluate_paired_test_valid_region.py` | `results/test_evaluation_valid_region/test_paired_comparison_valid_region.csv` |
| Figure 1: paired differences by seed | `scripts/visualization/plot_paired_differences.py` | `results/figures/fig1_paired_differences.png` |
| Split properties (Methods) | `scripts/splits/validate_paired_splits_independent.py` | `data/processed/paired_splits/independent_paired_split_audit.csv` |
| Patients shared between seeds (Methods) | `scripts/splits/audit_patient_overlap.py` | `results/patient_overlap_audit/` |

Warning: `scripts/splits/create_paired_splits.py` and `scripts/splits/create_master_split.py`
overwrite the committed split files that the results depend on. They are not part of the pipeline;
do not rerun them unless you intend to replace the splits.

## v2 experiments (in progress)

A revised design that addresses the limitations of the v1 runs. It is pre-specified in
[`docs/analysis_plan_v2.md`](docs/analysis_plan_v2.md) and does not change any v1 file.

- **Splits:** patient-grouped 5-fold paired design in `data/processed/kfold_splits/`
  (`scripts/splits/create_kfold_paired_splits.py`). Every patient is tested in exactly one fold;
  each test patient's images are split into a test half (identical in both arms) and a leak half
  (used for training in the image-level arm only).
- **Training:** `configs/v2_unet.yaml`: up to 200 epochs with early stopping (patience 25), learning
  rate halved on plateau, augmentation, fp16 with gradient scaling; three training seeds per fold
  and arm (30 runs, `--train-seed`).
- **Evaluation:** `scripts/evaluation/evaluate_v2.py`: valid region as the primary analysis, fixed
  and validation-chosen thresholds, false-positive measures with minimum component sizes,
  lesion-level detection and image-level AUROC.
- **Statistics:** `scripts/analysis/analyze_v2.py`: mixed model on paired differences (patient,
  training seed and image as random effects) and a hierarchical bootstrap.

```bash
python run_v2.py --dry-run     # list the 30 training runs and the later steps
python run_v2.py               # train, evaluate and analyse (can be stopped and restarted)
```

Outputs: `results/v2/unet/{arm}/fold{k}/train_seed{s}/` (training logs; checkpoints are not
committed), `results/v2/evaluation/`, `results/v2/analysis/`, `results/v2/figures/` and
`results/v2/paper/tables_v2.tex` (the tables of the manuscript).
`python run_v2.py --skip-training` regenerates all of them from the committed results, except the
example figure, which needs the checkpoints.

## Repository structure

```text
Medical-image-segmentation-research/
├── .github/workflows/tests.yml     continuous integration (ruff and pytest)
├── configs/
│   ├── paper_unet.yaml             settings used for the paper
│   ├── v2_unet.yaml                settings of the v2 experiments
│   └── exploratory/unet.yaml       configuration of the earlier exploratory runs
├── data/
│   ├── README.md                   layout of data/ and meaning of the CSV files
│   ├── raw/                        dataset zip (not committed)
│   └── processed/
│       ├── bus_uclm_patient_split.csv   master list of images, patients and labels
│       ├── paired_split_candidates/     candidate validation/test cohorts
│       ├── paired_splits/               the ten split files, manifest and audit reports
│       ├── kfold_splits/                v2 patient-grouped 5-fold paired splits
│       └── exploratory/                 earlier splits, not used in the paper
├── results/                        committed result tables of the paper (checkpoints are not committed)
│   ├── paired_full_{image|patient}_{seed}/   training logs
│   ├── test_evaluation/ and test_evaluation_valid_region/
│   ├── sensitivity_analysis/, patient_bootstrap/, leave_one_patient_out/
│   ├── patient_overlap_audit/, figures/
│   └── exploratory/                outputs of the side experiments (new outputs are not committed)
├── scripts/
│   ├── data_audit/                 dataset inspection and validation
│   ├── splits/                     split search, creation, validation and audit
│   ├── training/                   train_unet.py (--save-every-epoch keeps a checkpoint per epoch)
│   ├── evaluation/                 full-canvas and valid-region evaluation
│   ├── analysis/                   sensitivity, bootstrap and leave-one-patient-out
│   ├── visualization/              dataset samples, predictions and Figure 1
│   └── exploratory/                side experiments and diagnostics (see its README)
├── src/
│   ├── datasets/                   BUS-UCLM dataset class and preprocessing
│   ├── models/                     U-Net and Attention U-Net
│   └── utils/                      project paths
├── tests/                          pytest tests
├── run_paper.py                    runs the whole pipeline
├── run_v2.py                       runs the v2 pipeline
├── docs/analysis_plan_v2.md        pre-specified v2 analysis plan
├── pyproject.toml, requirements.txt, requirements-dev.txt, .pre-commit-config.yaml
├── CHANGELOG.md
├── CITATION.cff                    how to cite this repository
├── LICENSE                         MIT License
└── README.md
```

## Tests and code quality

```bash
pytest -q          # the dataset test is skipped when the dataset zip is not present
ruff check .
ruff format --check .
```

The same checks run on every push through GitHub Actions.

## Limitations of the experiments

- Each strategy was trained once per seed for at most 10 epochs without data augmentation. The
  early-stopping patience (10 epochs) equals the epoch limit, so it never triggers, and the best
  checkpoint was the last epoch for three of the ten models. Mixed precision is used without gradient
  scaling. Absolute performance is therefore modest and should be read as a comparison between
  splitting strategies, not as a tuned segmentation result.
- The five test sets share patients (30 patient-seed observations from 21 unique patients), so they
  are not independent cohorts.
- One dataset and one architecture were studied. See the manuscript for the full discussion.

## Data and citation

Archived code, all versions: https://doi.org/10.5281/zenodo.24394780. The version used for the
manuscript is v1.0.0: https://doi.org/10.5281/zenodo.24394781

Raw data are not redistributed. If you use this code, please cite the manuscript once available (GitHub also shows a "Cite this
repository" button, generated from `CITATION.cff`) and the dataset paper:

> Vallez N, Bueno G, Deniz O, et al. BUS-UCLM: Breast ultrasound lesion segmentation dataset.
> *Scientific Data*. 2025;12:242. https://doi.org/10.1038/s41597-025-04562-3

The U-Net follows Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical
image segmentation. MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The BUS-UCLM dataset is not
included in this repository and has its own license and terms of use; the MIT License does not
cover it.
