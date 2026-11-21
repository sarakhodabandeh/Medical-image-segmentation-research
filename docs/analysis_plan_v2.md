# Analysis plan for the v2 experiments

This plan was committed before any v2 model was trained. Any later change to it is recorded in
the git history and reported in the manuscript.

## Question

Does training on other images of the test patients (image-level splitting, a controlled leakage
setting) change lesion segmentation and false-positive predictions on normal images, compared with
patient-level splitting, when the models are trained to convergence and training randomness is
taken into account?

## Data and splits

- BUS-UCLM after quality control: 670 images, 37 patients (`data/processed/bus_uclm_patient_split.csv`).
- Patient-grouped 5-fold paired design (`scripts/splits/create_kfold_paired_splits.py`, seed 2027,
  files in `data/processed/kfold_splits/`). Every patient is a test patient in exactly one fold.
  Each test patient's images are split per label into a test half (rounded up) and a leak half.
- Patient-level arm: training on the training patients only. Image-level arm: the leak half of the
  test patients replaces training-patient images with the same label counts. Validation and test
  images are identical in both arms; training sets have equal size and class composition.

## Training

U-Net, `configs/v2_unet.yaml`: up to 200 epochs, early stopping with patience 25 on validation
lesion macro-Dice at threshold 0.5, ReduceLROnPlateau (factor 0.5, patience 10), Adam (lr 0.001,
weight decay 0.0001), batch size 4, loss 0.5 x BCE (positive weight 5) + 0.5 x soft Dice, fp16
mixed precision with gradient scaling, train-time augmentation (horizontal flip, rotation ±10°,
scale 0.9 to 1.1, brightness and contrast ±0.2, gamma 0.8 to 1.2). Checkpoint selection: highest
validation lesion macro-Dice at threshold 0.5. Three training seeds (0, 1, 2) per fold and arm:
30 runs.

## Endpoints

All endpoints are computed per test image inside the valid (unpadded) image region and analysed as
paired differences d = patient-level minus image-level for the same image, fold and training seed.

- **Primary:** mean false-positive area fraction on normal test images, threshold 0.5.
- **Key secondary:** lesion Dice on benign and malignant test images, threshold 0.5.
- **Secondary (Holm-adjusted together with the key secondary):** lesion IoU, pixel recall and
  lesion-level sensitivity (IoU ≥ 0.1); normal-image false-positive image indicator (any pixel, and
  after removing predicted components smaller than 10, 50 and 100 pixels); false-positive
  components per normal image.
- **Threshold-free:** normal-vs-lesion image-level AUROC of the maximum predicted probability,
  compared per fold and training seed.
- **Sensitivity analyses:** all endpoints at the validation-chosen threshold (maximising validation
  lesion Dice, grid 0.05 to 0.95); full 256 x 256 canvas instead of the valid region.

## Statistical model

For each endpoint: linear mixed model `d ~ 1` with a random intercept per patient and variance
components for training seed and image (`scripts/analysis/analyze_v2.py`); the intercept is the
mean paired difference, reported with its 95% CI and p-value. Confirmation with a hierarchical
bootstrap (10,000 replicates: patients, then training seeds within patients; percentile 95% CI).
Run-to-run variability (SD across the three training seeds) is reported next to the effects.
The direction and size of the primary endpoint are interpreted from the estimate and its CI, not
from the p-value alone.

## Not part of v2

A second architecture, an unconstrained random image-split arm, external validation and changes
to the loss weighting are not included; they are listed as limitations.
