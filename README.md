# The Impact of Patient-Level and Image-Level Data Splitting on Breast Ultrasound Lesion Segmentation Performance

This repository contains the code and experimental workflow for a study
investigating how **image-level** and **patient-level** data splitting
affect breast ultrasound lesion segmentation and false-positive
predictions on normal images.

The study uses the BUS-UCLM Breast Ultrasound Lesion Segmentation
Dataset and a U-Net segmentation model. Its central design is a paired
comparison: for each of five split configurations, the image-level and
patient-level experiments use exactly the same validation and test
images, with matched training-set sizes and class compositions. The key
difference is whether other images from patients represented in the
held-out sets are allowed to remain in the training set.

------------------------------------------------------------------------

## Table of contents

-   [Study overview](#study-overview)
-   [Dataset](#dataset)
-   [Experimental design](#experimental-design)
-   [Model and training
    configuration](#model-and-training-configuration)
-   [Evaluation](#evaluation)
-   [Main findings reported in the
    manuscript](#main-findings-reported-in-the-manuscript)
-   [Repository structure](#repository-structure)
-   [Reproduction workflow](#reproduction-workflow)
-   [Environment setup](#environment-setup)
-   [Data and output policy](#data-and-output-policy)
-   [Reproducibility notes and
    interpretation](#reproducibility-notes-and-interpretation)
-   [Citation](#citation)
-   [License](#license)

## Study overview

### Research question

How does the unit of data partitioning (image-level versus
patient-level) affect:

-   lesion segmentation performance on benign and malignant images; and
-   false-positive predictions on normal images?

### Dataset

The experiments use the **BUS-UCLM Breast Ultrasound Lesion Segmentation
Dataset**.

The original dataset contains 683 images from 38 patients. Following
quality control, 13 image-mask pairs with unresolved spatial-dimension
mismatches were excluded. The final experimental dataset contains:

| Category | Number of images |
|---|---:|
| Benign | 170 |
| Malignant | 90 |
| Normal | 410 |
| **Total** | **670** |

Normal images have empty lesion masks. Patient identifiers are retained
for patient-level splitting and patient-aware analyses.

**Dataset access:** The dataset is not included in this repository.
Obtain it from the official BUS-UCLM publication/repository and follow
its license and citation requirements. Do not commit the raw dataset or
patient data to Git.

### Experimental design

Five paired split configurations are used, with seeds:

`42, 123, 2025, 7, 99`

For each seed, both splitting strategies share exactly the same
validation and test images:

| Partition | Benign | Malignant | Normal | Total |
|---|---:|---:|---:|---:|
| Validation | 26 | 12 | 62 | 100 |
| Test | 31 | 14 | 61 | 106 |

-   **Patient-level split:** all images from patients assigned to
    validation or test are excluded from training.
-   **Image-level split:** the same validation and test images are used,
    but other images from those held-out patients may remain in
    training. Additional training images are selected to match the
    patient-level training set's size and class composition.

The paired split validation workflow checks paired validation/test
filenames, training-set size and class balance, and the intended
patient-overlap properties.

The five test configurations are **not independent cohorts**: some
patients occur in more than one seed. The analyses therefore treat
seed-level comparisons descriptively and include patient-aware
aggregation and sensitivity analyses.

### Model and training configuration

The primary model is a standard U-Net for binary lesion segmentation:

-   RGB input and one-channel output logits
-   Encoder channels: 16, 32, 64, 128
-   Bottleneck channels: 256
-   Double-convolution blocks with batch normalization and ReLU
-   Max pooling in the encoder, transposed convolutions and skip
    connections in the decoder

Main training settings:

| Setting | Value |
|---|---|
| Input size | 256 × 256 |
| Image interpolation | Bilinear |
| Mask interpolation | Nearest-neighbor |
| Image padding | Replicate |
| Mask padding | Constant zero |
| Batch size | 4 |
| Maximum epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | 0.5 BCE-with-logits + 0.5 Dice loss |
| Positive-class weight in BCE | 5 |
| Mixed precision | Enabled for CUDA training |
| Checkpoint selection | Highest validation lesion macro-Dice at threshold 0.5 |
| Early-stopping patience | 10 epochs |

Images are converted to RGB and scaled to `[0, 1]`. Masks are converted
to binary single-channel masks. Aspect ratio is preserved during
resizing; the resized image is center-padded to 256 × 256 using
replicated border values, while masks are padded with zeros.

The fixed probability threshold for the reported test evaluation is
**0.5**. The test set is not used to tune this threshold or select
checkpoints.

## Evaluation

### Primary evaluation metrics

For lesion-containing (benign and malignant) images:

-   **Lesion macro-Dice**
-   **Lesion macro-IoU**
-   **Pixel recall**

For normal images:

-   **Normal false-positive image rate:** proportion of normal images
    with at least one predicted positive pixel.
-   **Mean normal false-positive area fraction:** mean proportion of
    pixels predicted positive across normal images.

The primary metrics are calculated on the full 256 × 256 processed
canvas, including padding.

### Sensitivity and patient-aware analyses

The study additionally includes:

1.  **Valid-region evaluation:** recomputes metrics only within the
    resized source-image region, excluding artificial padding, and
    separately quantifies predictions in padded regions.
2.  **Aggregation sensitivity:** compares equal weighting per seed, per
    unique patient, and per unique test image.
3.  **Patient-level bootstrap:** resamples unique patients with
    replacement for 10,000 replicates and reports percentile-based 95%
    confidence intervals.
4.  **Leave-one-patient-out (LOPO):** repeats the paired comparison
    after excluding each patient in turn.

The bootstrap proportion of positive differences is descriptive and is
**not a p-value**. No conventional independent-samples test is applied
to the five seed-level comparisons.

## Main findings reported in the manuscript

The manuscript reports that:

-   Neither splitting strategy consistently outperformed the other for
    lesion Dice or IoU across the five paired configurations.
-   The direction of the overall lesion Dice/IoU difference depended on
    the aggregation strategy.
-   Patient-level splitting was associated with higher normal-image
    false-positive measures in four of five configurations.
-   The patient-level bootstrap confidence interval for mean normal
    false-positive area fraction excluded zero, while the intervals for
    lesion Dice, lesion IoU, and normal false-positive image rate
    included zero.
-   The valid-region sensitivity analysis showed a broadly similar
    pattern to the full-canvas evaluation.

These findings are specific to this dataset, model, and experimental
setup; they should not be interpreted as a universal ranking of
splitting strategies.

## Repository structure

The project is organized by function. The following is the intended
high-level layout; local data, checkpoints, generated results, and the
virtual environment remain on the development machine and are not
intended to be committed.

``` text
medical-segmentation-paper/
├── configs/
│   └── unet.yaml
├── data/
│   ├── raw/                 # Local dataset; do not commit
│   └── processed/           # Local metadata and split files
├── docs/
│   ├── dependency_report.txt
│   └── project_structure.txt
├── experiments/
├── notebooks/
├── paper/
├── results/                 # Local checkpoints and generated outputs
├── scripts/
│   ├── analysis/            # Statistical, patient-aware and model diagnostics
│   ├── data_audit/          # Dataset and mask quality-control scripts
│   ├── diagnostics/         # Environment and path diagnostics
│   ├── evaluation/          # Full-canvas and valid-region evaluation
│   ├── splits/              # Split generation and validation
│   ├── training/            # Model training scripts
│   └── visualization/       # Dataset, prediction and error visualizations
├── src/
│   ├── datasets/
│   └── models/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore               # Add before publishing to GitHub
```

### Script categories

| Directory | Purpose |
|---|---|
| `scripts/data_audit/` | Inspect dataset metadata, image-mask pairs, dimensions, masks, labels and patient distribution. |
| `scripts/splits/` | Create, search and validate image-level, patient-level and paired split definitions. |
| `scripts/training/` | Train U-Net models and run training comparisons. |
| `scripts/evaluation/` | Evaluate checkpoints and paired test sets, including valid-region sensitivity evaluation. |
| `scripts/analysis/` | Run statistical, patient-aware, probability, false-positive and BatchNorm analyses. |
| `scripts/visualization/` | Visualize dataset samples, predictions, errors and paired comparisons. |
| `scripts/diagnostics/` | Check project paths, GPU availability and batch-size behavior. |
| `tests/` | Focused tests for configuration, dataset, preprocessing, splits and model definitions. |
| `src/` | Reusable dataset, preprocessing and model implementation. |
| `configs/` | Experiment configuration files. |
| `docs/` | Project structure and dependency reports. |

## Reproduction workflow

Run commands from the repository root. On Windows PowerShell, use the
virtual environment's Python executable as shown below. The commands
reference the current organized script locations.

**Important:** Review the input paths and configuration of each script
before running it. Some scripts create or overwrite CSV manifests,
reports, checkpoints, or generated outputs. Do not rerun
split-generation or training commands over existing experiment artifacts
unless the outputs have been backed up and replacement is intended.

### 1. Dataset preparation and quality control

Relevant scripts are in `scripts/data_audit/`, including:

-   `inspect_dataset.py`
-   `validate_dataset.py`
-   `analyze_masks.py`
-   `inspect_mismatches.py`
-   `review_mask_samples.py`

These scripts inspect image-mask consistency, class distribution, empty
masks, mask geometry, and visual examples. The final experimental
dataset excludes the 13 image-mask pairs with unresolved dimension
mismatches.

Run an individual audit script from the repository root, for example:

``` powershell
.\.venv\Scripts\python.exe scripts\data_audit\inspect_dataset.py
```

Check each script's configured dataset path before execution.

### 2. Construct paired splits

The split workflow scripts are in `scripts/splits/`.

Run the candidate search before generating paired splits:

``` powershell
.\.venv\Scripts\python.exe scripts\splits\search_paired_split_candidates.py
.\.venv\Scripts\python.exe scripts\splits\create_paired_splits.py
```

The split-generation scripts expect the patient metadata/split input and
candidate CSV at the paths configured in the scripts. Review those paths
and the dataset metadata requirements before running.

The paired split files are written under `data/processed/paired_splits/`
and include patient-level and image-level CSVs for each seed, together
with validation and manifest reports.

**Do not run the split generator casually:** it can overwrite existing
split CSVs, manifests and reports. Existing split artifacts should be
backed up before regeneration.

The independent paired-split validation script is:

``` powershell
.\.venv\Scripts\python.exe scripts\splits\validate_paired_splits_independent.py
```

This validator checks the existing paired split artifacts. Review its
output before proceeding.

### 3. Train U-Net models

The standard training script is located at
`scripts/training/train_unet.py`.

To inspect its available command-line options:

``` powershell
.\.venv\Scripts\python.exe scripts\training\train_unet.py --help
```

Use the arguments supported by the script to select the split CSV, seed,
configuration and output directory. The paired experiment checkpoints
are expected by the evaluation scripts under:

``` text
results/paired_full_{image|patient}_{seed}/best_model.pt
```

Here, `image` and `patient` denote the two split conditions, and
`{seed}` is one of `42`, `123`, `2025`, `7`, or `99`.

`train_unet_save_epochs.py` is an additional U-Net training variant that
also saves per-epoch checkpoints for validation/epoch diagnostics. It is
not interchangeable with the standard trainer's output layout unless the
corresponding output paths are configured consistently.

Training commands can consume substantial GPU time and write
checkpoints. Confirm configuration, input splits and output paths before
starting a training run.

### 4. Evaluate the paired test sets

Primary full-canvas evaluation:

``` powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_paired_test.py
```

Valid-region sensitivity evaluation:

``` powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_paired_test_valid_region.py
```

The evaluation scripts use a fixed threshold of 0.5 and expect the
paired checkpoints and split CSVs to exist at the paths specified in the
scripts.

The full-canvas evaluator writes the per-image metrics file:

``` text
results/test_evaluation/test_per_image_metrics.csv
```

The valid-region evaluator writes a separate per-image file:

``` text
results/test_evaluation_valid_region/test_per_image_metrics_valid_region.csv
```

The evaluation scripts also write summary and paired-comparison CSV
files in their configured output directories.

### 5. Run statistical and sensitivity analyses

The following analyses consume the full-canvas per-image test metrics
unless otherwise configured:

``` powershell
.\.venv\Scripts\python.exe scripts\analysis\sensitivity_analysis.py
.\.venv\Scripts\python.exe scripts\analysis\bootstrap_patient_analysis.py
.\.venv\Scripts\python.exe scripts\analysis\leave_one_patient_out.py
```

The expected full-canvas input is:

``` text
results/test_evaluation/test_per_image_metrics.csv
```

The valid-region evaluator produces a separate set of CSV outputs. The
statistical scripts above should not be assumed to analyze valid-region
metrics unless their input paths are explicitly changed.

Other diagnostic scripts examine validation thresholds, prediction
probabilities, preprocessing, BatchNorm behavior, influential patients
and individual errors. These are supplementary analyses and are not
required to reproduce the primary paired comparison.

### 6. Visualize test errors

The `scripts/visualization/visualize_test_errors.py` script reads:

``` text
results/test_evaluation/test_per_image.csv
```

This is a separate input from the primary paired evaluator's
`test_per_image_metrics.csv`. The visualization script also expects its
configured checkpoint to exist. Confirm that both files are present
before running it.

Other visualization scripts may require specific checkpoints, datasets
or intermediate results. Check their configured paths before execution.

## Environment setup

Python 3.12 was used during development. The project uses PyTorch and
CUDA for GPU training.

Create and activate a virtual environment (Windows PowerShell):

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment's Python executable
directly:

``` powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your operating system, NVIDIA
driver, and CUDA runtime by following the official PyTorch installation
selector. The development environment used a CUDA-enabled PyTorch build
(`torch 2.11.0+cu128`), but this exact build may not be appropriate for
every machine.

Check the installation:

``` powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data and output policy

The repository should contain code, configuration, documentation, and
(where appropriate) small split-definition/manifest files. It should
**not** contain:

-   the raw BUS-UCLM images or masks;
-   local virtual environments such as `.venv/`;
-   model checkpoints (`.pt`, `.pth`);
-   large generated result directories or temporary logs;
-   Python cache files such as `__pycache__/` and `*.pyc`;
-   IDE-specific local settings.

Keep the dataset in the local location expected by the dataset loader,
or update the relevant configuration/path arguments. Do not publish
patient-identifying information or files that are not permitted by the
dataset license.

Before publishing, review `git status` and confirm that no raw data,
checkpoints, local environment files, private metadata or other
restricted artifacts are staged.

## Reproducibility notes and interpretation

-   Use the same paired validation and test images for both split
    strategies.
-   Preserve the matched training-set size and class composition.
-   Keep the threshold fixed at 0.5 for test evaluation.
-   Do not treat the five seeds as five independent patient cohorts.
-   Distinguish full-canvas primary metrics from valid-region
    sensitivity metrics.
-   The statistical scripts described above consume the full-canvas
    per-image metrics file; valid-region statistical summaries require a
    separate, explicitly configured analysis.
-   Record the Python, PyTorch, CUDA, and dependency versions used for
    each reproduction.
-   Preserve the original split manifests, reports and checkpoints for
    each completed experiment.

## Citation

If you use this repository or reproduce the experiments, cite the
associated manuscript when it is published. Please also cite the
BUS-UCLM dataset paper:

Vallez N, Bueno G, Deniz O, et al. **BUS-UCLM: Breast ultrasound lesion
segmentation dataset.** *Scientific Data*. 2025;12:242.\
https://doi.org/10.1038/s41597-025-04562-3

The manuscript also references the original U-Net paper:

Ronneberger O, Fischer P, Brox T. **U-Net: Convolutional Networks for
Biomedical Image Segmentation.** MICCAI 2015.\
https://doi.org/10.1007/978-3-319-24574-4_28

## License

No repository license is specified here. Add a license only after
deciding which license is appropriate for the code and confirming
compatibility with the dataset's terms. The dataset's license and the
code license are separate.
