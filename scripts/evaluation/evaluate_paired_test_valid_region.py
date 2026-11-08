import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)
from src.models.unet import UNet



def find_project_root():
    current = Path(__file__).resolve().parent

    for directory in (current, *current.parents):
        if (
            (directory / "configs").is_dir()
            and (directory / "src").is_dir()
            and (directory / "requirements.txt").is_file()
        ):
            return directory

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()
DEFAULT_SEEDS = [42, 123, 2025, 7, 99]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path



def preprocess(images, masks, size, padding_mode):
    if padding_mode == "replicate":
        return resize_with_padding(images, masks, size)
    if padding_mode == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding mode: {padding_mode}")


def make_valid_region_mask(batch_size, original_h, original_w, size, device):
    """
    True only for pixels originating from the resized ultrasound image.
    Newly introduced square-canvas padding is False.
    The geometry exactly mirrors resize_with_padding / resize_with_constant_padding.
    """
    scale = min(size / original_h, size / original_w)
    new_h = max(1, round(original_h * scale))
    new_w = max(1, round(original_w * scale))

    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    left = pad_w // 2

    valid = torch.zeros((batch_size, 1, size, size), dtype=torch.bool, device=device)
    valid[:, :, top:top + new_h, left:left + new_w] = True
    return valid


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_model(model, loader, device, image_size, padding_mode, use_amp, run_name):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            original_images = batch["image"].to(device, non_blocking=True)
            original_masks = batch["mask"].to(device, non_blocking=True)

            # DataLoader batches must have a common H,W to stack tensors.
            original_h, original_w = original_images.shape[-2:]
            images, masks = preprocess(
                original_images, original_masks, image_size, padding_mode
            )

            valid_region = make_valid_region_mask(
                images.shape[0], original_h, original_w, image_size, device
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(images)

            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5

            for i in range(images.shape[0]):
                pred = predictions[i, 0]
                target = targets[i, 0]
                valid = valid_region[i, 0]
                padding = ~valid

                # Metrics inside the actual resized image area only.
                pred_valid = pred & valid
                target_valid = target & valid
                tp = int((pred_valid & target_valid).sum().item())
                fp = int((pred_valid & ~target_valid).sum().item())
                fn = int((~pred_valid & target_valid).sum().item())
                tn = int((~pred_valid & ~target_valid & valid).sum().item())
                valid_pixels = int(valid.sum().item())

                # Separately quantify predictions in the newly added padding.
                padding_pixels = int(padding.sum().item())
                padding_fp = int((pred & padding).sum().item())
                padding_prediction_fraction = safe_ratio(padding_fp, padding_pixels)

                label = batch["label"][i]
                is_lesion = label in ("Benign", "Malignant")
                dice = safe_ratio(2 * tp, 2 * tp + fp + fn) if is_lesion else np.nan
                iou = safe_ratio(tp, tp + fp + fn) if is_lesion else np.nan
                recall = safe_ratio(tp, tp + fn) if is_lesion else np.nan
                precision = safe_ratio(tp, tp + fp) if is_lesion else np.nan

                rows.append({
                    "run": run_name,
                    "filename": batch["filename"][i],
                    "patient_id": batch["patient_id"][i],
                    "label": label,
                    "valid_image_pixels": valid_pixels,
                    "padding_pixels": padding_pixels,
                    "tp_valid_pixels": tp,
                    "fp_valid_pixels": fp,
                    "fn_valid_pixels": fn,
                    "tn_valid_pixels": tn,
                    "dice_valid_region": dice,
                    "iou_valid_region": iou,
                    "pixel_recall_valid_region": recall,
                    "pixel_precision_valid_region": precision,
                    "normal_false_positive_image_valid_region": (
                        int(fp > 0) if label == "Normal" else np.nan
                    ),
                    "false_positive_area_fraction_valid_region": (
                        safe_ratio(fp, valid_pixels)
                    ),
                    "padding_predicted_pixels": padding_fp,
                    "padding_prediction_fraction": padding_prediction_fraction,
                    "padding_share_of_all_predictions": safe_ratio(
                        padding_fp, int(pred.sum().item())
                    ),
                })

    return pd.DataFrame(rows)


def summarize(frame, seed, split_type):
    lesions = frame[frame.label.isin(["Benign", "Malignant"])]
    normals = frame[frame.label == "Normal"]

    result = {
        "seed": seed,
        "split_type": split_type,
        "n_test_images": len(frame),
        "n_lesion_images": len(lesions),
        "n_normal_images": len(normals),
        "lesion_macro_dice_valid_region": lesions.dice_valid_region.mean(),
        "lesion_macro_iou_valid_region": lesions.iou_valid_region.mean(),
        "lesion_macro_pixel_recall_valid_region": (
            lesions.pixel_recall_valid_region.mean()
        ),
        "lesion_macro_pixel_precision_valid_region": (
            lesions.pixel_precision_valid_region.mean()
        ),
        "normal_false_positive_image_rate_valid_region": (
            normals.normal_false_positive_image_valid_region.mean()
        ),
        "normal_mean_false_positive_area_fraction_valid_region": (
            normals.false_positive_area_fraction_valid_region.mean()
        ),
        "normal_pixel_false_positive_rate_valid_region": safe_ratio(
            normals.fp_valid_pixels.sum(),
            normals.fp_valid_pixels.sum() + normals.tn_valid_pixels.sum(),
        ),
        "mean_padding_prediction_fraction": frame.padding_prediction_fraction.mean(),
        "mean_padding_share_of_all_predictions": (
            frame.padding_share_of_all_predictions.mean()
        ),
    }

    for label in ("Benign", "Malignant"):
        subset = lesions[lesions.label == label]
        key = label.lower()
        result[f"{key}_n"] = len(subset)
        result[f"{key}_dice_valid_region"] = subset.dice_valid_region.mean()
        result[f"{key}_iou_valid_region"] = subset.iou_valid_region.mean()
        result[f"{key}_pixel_recall_valid_region"] = (
            subset.pixel_recall_valid_region.mean()
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate paired BUS-UCLM checkpoints while excluding only "
            "the artificial square-canvas padding from metrics."
        )
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--splits-dir", default="data/processed/paired_splits")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir", default="results/test_evaluation_valid_region"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    image_size = int(base_config["data"]["image_size"])
    batch_size = args.batch_size or int(base_config["data"]["batch_size"])
    num_workers = int(base_config["data"]["num_workers"])
    use_amp = bool(base_config["training"]["use_amp"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = resolve_path(args.splits_dir)
    results_dir = resolve_path(args.results_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Threshold: 0.5 (fixed; no test-set tuning)")
    print("Metric region: resized source image only; artificial square padding excluded")

    all_frames = []
    summaries = []

    for seed in args.seeds:
        for split_type in ("image", "patient"):
            split_csv = splits_dir / f"{split_type}_level_seed_{seed}.csv"
            checkpoint_path = (
                results_dir / f"paired_full_{split_type}_{seed}" / "best_model.pt"
            )

            if not split_csv.exists():
                raise FileNotFoundError(f"Missing split CSV: {split_csv}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            cfg = checkpoint.get("config", base_config)
            checkpoint_seed = int(checkpoint.get("seed", seed))
            if checkpoint_seed != seed:
                raise ValueError(
                    f"Seed mismatch in {checkpoint_path}: {checkpoint_seed}"
                )

            padding_mode = checkpoint.get("padding_mode", "replicate")
            dataset = BUSUCLMDataset(split="test", split_path=split_csv)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            model = UNet(
                in_channels=cfg["model"]["in_channels"],
                out_channels=cfg["model"]["out_channels"],
                base_channels=cfg["model"]["base_channels"],
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

            run_name = f"{split_type}_level_seed_{seed}"
            print(
                f"Evaluating {run_name}: {len(dataset)} test images; "
                f"selected epoch={checkpoint.get('epoch')}; padding={padding_mode}"
            )

            frame = evaluate_model(
                model,
                loader,
                device,
                image_size,
                padding_mode,
                use_amp,
                run_name,
            )
            frame["seed"] = seed
            frame["split_type"] = split_type
            all_frames.append(frame)
            summaries.append(summarize(frame, seed, split_type))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_image = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    per_image_path = output_dir / "test_per_image_metrics_valid_region.csv"
    summary_path = output_dir / "test_summary_valid_region.csv"
    paired_path = output_dir / "test_paired_comparison_valid_region.csv"

    per_image.to_csv(per_image_path, index=False)
    summary.to_csv(summary_path, index=False)

    metrics = [
        "lesion_macro_dice_valid_region",
        "lesion_macro_iou_valid_region",
        "lesion_macro_pixel_recall_valid_region",
        "lesion_macro_pixel_precision_valid_region",
        "normal_false_positive_image_rate_valid_region",
        "normal_mean_false_positive_area_fraction_valid_region",
        "normal_pixel_false_positive_rate_valid_region",
        "mean_padding_prediction_fraction",
        "mean_padding_share_of_all_predictions",
    ]
    wide = summary.pivot(index="seed", columns="split_type", values=metrics)
    paired = pd.DataFrame(index=wide.index)

    for metric in metrics:
        paired[f"{metric}_image_level"] = wide[(metric, "image")]
        paired[f"{metric}_patient_level"] = wide[(metric, "patient")]
        paired[f"{metric}_difference_patient_minus_image"] = (
            wide[(metric, "patient")] - wide[(metric, "image")]
        )

    paired.reset_index().to_csv(paired_path, index=False)

    print("\nValid-region evaluation completed.")
    print("Output directory:", output_dir)
    print("\nSummary by run:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(per_image_path)
    print(summary_path)
    print(paired_path)


if __name__ == "__main__":
    main()
