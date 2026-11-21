"""Evaluate the v2 runs (patient-grouped 5-fold paired design).

For every run folder results/v2/{model}/{arm}/fold{k}/train_seed{s}/ with a best_model.pt:
1. Predict the validation set and choose the threshold (grid 0.05 to 0.95) that maximises the
   validation lesion macro-Dice inside the valid (unpadded) image region.
2. Predict the test set and compute, at the fixed threshold 0.5 and at the validation-chosen
   threshold, per-image metrics in the valid region (primary) and on the full 256 x 256 canvas:
   - lesion images: Dice, IoU, pixel recall; lesion-level detection (a reference lesion counts
     as detected when a predicted component overlaps it with IoU >= 0.1) and false-positive
     components per image;
   - normal images: false-positive image indicator (any predicted pixel, and after removing
     predicted components smaller than 10, 50 and 100 pixels), false-positive area fraction and
     number of false-positive components.
   The maximum predicted probability in the valid region is also stored for every image
   (threshold-free; used for the normal-vs-lesion image-level AUROC).
Outputs, written to --output-dir: test_per_image_v2.csv (one row per run, image and threshold)
and run_summary_v2.csv (one row per run and threshold, including the image-level AUROC).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import resize_with_padding
from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet
from src.utils.paths import PROJECT_ROOT

MIN_COMPONENT_SIZES = [10, 50, 100]
DETECTION_IOU = 0.1
DETECTION_MIN_SIZE = 10
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
LESION_LABELS = ("Benign", "Malignant")
STRUCTURE = np.ones((3, 3), dtype=bool)  # 8-connectivity


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def valid_region(original_h, original_w, size):
    """Boolean [size, size] mask of pixels that come from the resized image (not padding)."""
    scale = min(size / original_h, size / original_w)
    new_h, new_w = max(1, round(original_h * scale)), max(1, round(original_w * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    region = np.zeros((size, size), dtype=bool)
    region[top : top + new_h, left : left + new_w] = True
    return region


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_class = AttentionUNet if checkpoint.get("model_name") == "attention_unet" else UNet
    model = model_class(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def predict(model, dataset, config, device):
    """Yield (sample info, probability map, target mask, valid region) per image."""
    size = config["data"]["image_size"]
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            original_h, original_w = images.shape[-2:]
            images, masks = resize_with_padding(images, masks, size)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
            target = (masks[0, 0] >= 0.5).cpu().numpy()
            info = {key: batch[key][0] for key in ("filename", "patient_id", "label")}
            yield info, probability, target, valid_region(original_h, original_w, size)


def remove_small(prediction, min_size):
    labels, count = ndimage.label(prediction, structure=STRUCTURE)
    if count == 0:
        return prediction
    sizes = ndimage.sum(prediction, labels, index=np.arange(1, count + 1))
    keep = np.isin(labels, np.flatnonzero(sizes >= min_size) + 1)
    return keep


def detection(prediction, target):
    """Return (reference lesions, detected lesions, false-positive components)."""
    prediction = remove_small(prediction, DETECTION_MIN_SIZE)
    pred_labels, n_pred = ndimage.label(prediction, structure=STRUCTURE)
    gt_labels, n_gt = ndimage.label(target, structure=STRUCTURE)
    matched_pred = set()
    detected = 0
    for g in range(1, n_gt + 1):
        gt = gt_labels == g
        hit = False
        for p in np.unique(pred_labels[gt]):
            if p == 0:
                continue
            component = pred_labels == p
            iou = (component & gt).sum() / (component | gt).sum()
            if iou >= DETECTION_IOU:
                hit = True
                matched_pred.add(int(p))
        detected += int(hit)
    false_components = sum(
        1
        for p in range(1, n_pred + 1)
        if p not in matched_pred and not ((pred_labels == p) & target).any()
    )
    return n_gt, detected, false_components


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def image_metrics(probability, target, region, threshold, label):
    """Metrics for one image at one threshold, valid region (primary) and full canvas."""
    prediction_full = probability >= threshold
    row = {}
    for name, mask in (("valid", region), ("full", np.ones_like(region))):
        pred = prediction_full & mask
        tgt = target & mask
        tp, fp, fn = int((pred & tgt).sum()), int((pred & ~tgt).sum()), int((~pred & tgt).sum())
        pixels = int(mask.sum())
        if label in LESION_LABELS:
            row[f"dice_{name}"] = ratio(2 * tp, 2 * tp + fp + fn)
            row[f"iou_{name}"] = ratio(tp, tp + fp + fn)
            row[f"recall_{name}"] = ratio(tp, tp + fn)
        else:
            row[f"fp_area_fraction_{name}"] = ratio(fp, pixels)
            row[f"fp_image_{name}"] = int(fp > 0)
    pred_valid = prediction_full & region
    if label not in LESION_LABELS:
        for min_size in MIN_COMPONENT_SIZES:
            row[f"fp_image_valid_min{min_size}"] = int(remove_small(pred_valid, min_size).any())
    n_gt, detected, false_components = detection(pred_valid, target & region)
    row["fp_components_valid"] = false_components
    if label in LESION_LABELS:
        row["n_reference_lesions"] = n_gt
        row["lesion_sensitivity_valid"] = ratio(detected, n_gt)
    return row


def choose_threshold(model, val_dataset, config, device):
    stored = list(predict(model, val_dataset, config, device))
    best_threshold, best_dice = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        dices = []
        for info, probability, target, region in stored:
            if info["label"] in LESION_LABELS:
                pred, tgt = (probability >= threshold) & region, target & region
                tp = (pred & tgt).sum()
                dices.append(ratio(2 * tp, pred.sum() + tgt.sum()))
        mean_dice = float(np.nanmean(dices))
        if mean_dice > best_dice:
            best_threshold, best_dice = float(threshold), mean_dice
    return best_threshold, best_dice


def find_runs(results_dir, model_name):
    root = resolve_path(results_dir) / "v2" / model_name
    runs = []
    for checkpoint in sorted(root.glob("*/fold*/train_seed*/best_model.pt")):
        if not re.fullmatch(r"train_seed\d+", checkpoint.parent.name):
            continue
        if not (checkpoint.parent / "training_complete.txt").exists():
            print(f"skip unfinished run {checkpoint.parent}")
            continue
        train_seed_dir, fold_dir, arm_dir = (
            checkpoint.parents[0],
            checkpoint.parents[1],
            checkpoint.parents[2],
        )
        runs.append(
            {
                "arm": arm_dir.name,
                "fold": int(fold_dir.name.removeprefix("fold")),
                "train_seed": int(train_seed_dir.name.removeprefix("train_seed")),
                "checkpoint": checkpoint,
            }
        )
    return runs


def summarize(frame):
    rows = []
    keys = ["model", "arm", "fold", "train_seed", "threshold_name"]
    for key, group in frame.groupby(keys, sort=True):
        lesions = group[group["label"].isin(LESION_LABELS)]
        normals = group[group["label"] == "Normal"]
        row = dict(zip(keys, key))
        row["threshold"] = group["threshold"].iloc[0]
        for column in [
            "dice_valid",
            "iou_valid",
            "recall_valid",
            "dice_full",
            "iou_full",
            "recall_full",
        ]:
            row[f"lesion_{column}"] = lesions[column].mean()
        row["lesion_sensitivity_valid"] = lesions["lesion_sensitivity_valid"].mean()
        for column in [
            "fp_area_fraction_valid",
            "fp_image_valid",
            "fp_image_valid_min10",
            "fp_image_valid_min50",
            "fp_image_valid_min100",
            "fp_area_fraction_full",
            "fp_image_full",
        ]:
            row[f"normal_{column}"] = normals[column].mean()
        row["fp_components_per_image_valid"] = group["fp_components_valid"].mean()
        is_lesion = group["label"].isin(LESION_LABELS).astype(int)
        row["image_auroc_max_probability"] = (
            roc_auc_score(is_lesion, group["max_probability_valid"])
            if is_lesion.nunique() == 2
            else float("nan")
        )
        row["n_lesion_images"], row["n_normal_images"] = len(lesions), len(normals)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--splits-dir", default="data/processed/kfold_splits")
    parser.add_argument("--model", default="unet", choices=["unet", "attention_unet"])
    parser.add_argument("--output-dir", default="results/v2/evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs(args.results_dir, args.model)
    if not runs:
        raise FileNotFoundError(f"No v2 checkpoints found under {args.results_dir}/v2/{args.model}")

    rows = []
    for run in runs:
        split_csv = resolve_path(args.splits_dir) / f"fold{run['fold']}_{run['arm']}.csv"
        model, config, checkpoint = load_model(run["checkpoint"], device)
        val_dataset = BUSUCLMDataset(split="val", split_path=split_csv)
        test_dataset = BUSUCLMDataset(split="test", split_path=split_csv)
        tuned_threshold, val_dice = choose_threshold(model, val_dataset, config, device)
        name = f"{run['arm']}_fold{run['fold']}_seed{run['train_seed']}"
        print(
            f"{name}: epoch {checkpoint['epoch']}, validation threshold {tuned_threshold:.2f} "
            f"(validation lesion Dice {val_dice:.4f})"
        )

        for info, probability, target, region in predict(model, test_dataset, config, device):
            base = {
                "model": args.model,
                "arm": run["arm"],
                "fold": run["fold"],
                "train_seed": run["train_seed"],
                "selected_epoch": checkpoint["epoch"],
                **info,
                "max_probability_valid": float(probability[region].max()),
            }
            for threshold_name, threshold in (("fixed_0.5", 0.5), ("val_tuned", tuned_threshold)):
                rows.append(
                    {
                        **base,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        **image_metrics(probability, target, region, threshold, info["label"]),
                    }
                )

    frame = pd.DataFrame(rows)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "test_per_image_v2.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(output_dir / "run_summary_v2.csv", index=False)
    print(f"Wrote {len(frame)} rows to {output_dir / 'test_per_image_v2.csv'}")
    print(f"Wrote {len(summary)} rows to {output_dir / 'run_summary_v2.csv'}")


if __name__ == "__main__":
    main()
