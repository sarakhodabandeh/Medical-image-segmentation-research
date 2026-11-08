import argparse
import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.models.unet import UNet
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def preprocess_batch(images, masks, size, padding_mode):
    if padding_mode == "replicate":
        return resize_with_padding(images, masks, size)
    if padding_mode == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unsupported padding mode: {padding_mode}")


def make_model(checkpoint, device):
    cfg = checkpoint["config"]
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def bn_snapshot(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            rows.append({
                "layer": name,
                "num_features": module.num_features,
                "running_mean_abs_mean": module.running_mean.detach().abs().mean().item(),
                "running_var_mean": module.running_var.detach().mean().item(),
                "running_var_min": module.running_var.detach().min().item(),
                "running_var_max": module.running_var.detach().max().item(),
                "num_batches_tracked": int(module.num_batches_tracked.item()),
            })
    return rows


def evaluate(model, loader, device, image_size, padding_mode, epoch, variant):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = preprocess_batch(
                images, masks, image_size, padding_mode
            )
            probs = torch.sigmoid(model(images).float())

            for i, label in enumerate(batch["label"]):
                prob = probs[i, 0]
                target = masks[i, 0] >= 0.5
                pred = prob >= 0.5
                positive = int(pred.any().item())
                pred_count = int(pred.sum().item())
                target_count = int(target.sum().item())
                intersection = int((pred & target).sum().item())

                if label in ("Benign", "Malignant"):
                    denom = pred_count + target_count
                    dice = (2.0 * intersection / denom) if denom else 1.0
                    recall = intersection / target_count if target_count else float("nan")
                else:
                    dice = float("nan")
                    recall = float("nan")

                rows.append({
                    "epoch": epoch,
                    "variant": variant,
                    "filename": batch["filename"][i],
                    "patient_id": batch["patient_id"][i],
                    "label": label,
                    "dice_at_0_5": dice,
                    "pixel_recall_at_0_5": recall,
                    "normal_any_fp_at_0_5": positive if label == "Normal" else "",
                    "predicted_area_fraction": pred.float().mean().item(),
                    "prob_mean": prob.mean().item(),
                    "prob_q95": torch.quantile(prob.flatten(), 0.95).item(),
                    "prob_max": prob.max().item(),
                })
    return rows


def summarize(rows):
    lesion = [r for r in rows if r["label"] in ("Benign", "Malignant")]
    normal = [r for r in rows if r["label"] == "Normal"]
    dice_values = [r["dice_at_0_5"] for r in lesion]
    recall_values = [r["pixel_recall_at_0_5"] for r in lesion]
    return {
        "lesion_images": len(lesion),
        "normal_images": len(normal),
        "lesion_macro_dice": sum(dice_values) / len(dice_values) if dice_values else float("nan"),
        "lesion_macro_pixel_recall": sum(recall_values) / len(recall_values) if recall_values else float("nan"),
        "normal_fp_image_rate": (
            sum(int(r["normal_any_fp_at_0_5"]) for r in normal) / len(normal)
            if normal else float("nan")
        ),
        "normal_mean_predicted_area_fraction": (
            sum(r["predicted_area_fraction"] for r in normal) / len(normal)
            if normal else float("nan")
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate BatchNorm statistics on training images and compare validation metrics."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/diagnostic_patient_99/epoch_checkpoints",
    )
    parser.add_argument(
        "--split-csv",
        default="data/processed/paired_splits/patient_level_seed_99.csv",
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--output-dir", default="results/bn_recalibration")
    parser.add_argument("--epochs", type=int, nargs="*", default=[8, 10])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    checkpoint_dir = resolve_path(args.checkpoint_dir)
    split_path = resolve_path(args.split_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = BUSUCLMDataset(split="train", split_path=split_path)
    val_ds = BUSUCLMDataset(split="val", split_path=split_path)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    all_rows = []
    summary_rows = []
    bn_rows = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training images for BN calibration: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print("Calibration uses training images only; no gradients or optimizer steps.")

    for epoch in args.epochs:
        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if not ckpt_path.exists():
            print(f"WARNING: missing checkpoint: {ckpt_path}")
            continue

        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        cfg = checkpoint.get("config", default_config)
        image_size = cfg["data"]["image_size"]
        padding_mode = checkpoint.get("padding_mode", "replicate")

        original_model = make_model(checkpoint, device)
        original_model.eval()

        # Baseline evaluation with the checkpoint's original BN buffers.
        baseline_rows = evaluate(
            original_model, val_loader, device, image_size,
            padding_mode, epoch, "original_bn",
        )
        all_rows.extend(baseline_rows)
        baseline_summary = summarize(baseline_rows)
        summary_rows.append({"epoch": epoch, "variant": "original_bn", **baseline_summary})

        # Recalibrate a separate copy: reset only BN running buffers.
        calibrated_model = copy.deepcopy(original_model)
        bn_modules = [
            module for module in calibrated_model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        for module in bn_modules:
            module.reset_running_stats()
            module.momentum = None  # cumulative average over calibration batches

        # U-Net has no dropout. train() is used so BN uses batch statistics
        # and updates running buffers; inference_mode prevents gradients.
        calibrated_model.train()
        with torch.inference_mode():
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                images, _ = preprocess_batch(
                    images, masks, image_size, padding_mode
                )
                _ = calibrated_model(images)

        calibrated_model.eval()
        calibrated_rows = evaluate(
            calibrated_model, val_loader, device, image_size,
            padding_mode, epoch, "recalibrated_bn",
        )
        all_rows.extend(calibrated_rows)
        calibrated_summary = summarize(calibrated_rows)
        summary_rows.append({
            "epoch": epoch, "variant": "recalibrated_bn", **calibrated_summary
        })

        for variant, model in [
            ("original_bn", original_model),
            ("recalibrated_bn", calibrated_model),
        ]:
            for row in bn_snapshot(model):
                bn_rows.append({"epoch": epoch, "variant": variant, **row})

        print(
            f"Epoch {epoch}: "
            f"Dice {baseline_summary['lesion_macro_dice']:.4f} -> "
            f"{calibrated_summary['lesion_macro_dice']:.4f}; "
            f"Normal FP {baseline_summary['normal_fp_image_rate']:.3f} -> "
            f"{calibrated_summary['normal_fp_image_rate']:.3f}"
        )

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "bn_recalibration_per_image.csv", all_rows)
    write_csv(output_dir / "bn_recalibration_summary.csv", summary_rows)
    write_csv(output_dir / "bn_recalibration_layer_stats.csv", bn_rows)

    print("\nFinished.")
    print(f"Results folder: {output_dir}")
    print("Files: bn_recalibration_per_image.csv, "
          "bn_recalibration_summary.csv, bn_recalibration_layer_stats.csv")


if __name__ == "__main__":
    main()
