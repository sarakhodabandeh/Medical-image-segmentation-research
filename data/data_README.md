# Data layout

This folder holds the dataset inputs and the split files used in the paper. Raw images are
**not** stored in the repository.

```text
data/
├── raw/
│   └── BUS-UCLM Breast ultrasound lesion segmentation dataset.zip   (not committed)
└── processed/
    ├── bus_uclm_patient_split.csv     master list: one row per image
    ├── paired_split_candidates/       output of the candidate search
    ├── paired_splits/                 the splits used in the paper
    └── exploratory/                   earlier splits, not used in the paper
```

## raw/

Download BUS-UCLM from its official publication (Vallez et al., *Scientific Data* 2025,
https://doi.org/10.1038/s41597-025-04562-3) and follow its license. Put the **zip file** in
`data/raw/` with exactly the file name above. The code reads images and masks directly from
the zip, so it must **not** be extracted.

## processed/bus_uclm_patient_split.csv

Master list of every image kept after quality control (37 patients). Columns: `filename`,
`patient_id`, `label` (Benign, Malignant or Normal) and `split` (the original fixed
patient-level train/validation/test assignment). It is created by
`scripts/splits/create_master_split.py`.

## processed/paired_split_candidates/

`top_paired_split_candidates.csv`, written by `scripts/splits/search_paired_split_candidates.py`:
the candidate patient-held-out validation/test cohorts that were ranked by the search.

## processed/paired_splits/

Written by `scripts/splits/create_paired_splits.py` and checked by
`scripts/splits/validate_paired_splits_independent.py`.

| File | Content |
|---|---|
| `patient_level_seed_{seed}.csv` | Patient-level split for one seed |
| `image_level_seed_{seed}.csv` | Matched image-level split for the same seed |
| `paired_split_manifest.csv` | Overview of the ten split files |
| `paired_split_validation_report.csv` | Checks made while generating the splits |
| `independent_paired_split_audit.csv` | Result of the independent audit |

The seeds are 42, 123, 2025, 7 and 99. For each seed, both CSVs share identical validation
and test images (100 and 106 images, from 6 validation and 6 test patients).

> **Do not regenerate these files.** The paper results depend on them, and the generator can
> overwrite them.

## processed/exploratory/

Splits from earlier experiments (`matched_splits/`, `patient_level_splits/`,
`bus_uclm_image_split.csv`). Not needed to reproduce the paper.
