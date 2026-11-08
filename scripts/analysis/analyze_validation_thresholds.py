import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
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
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT / "results" / "unet_replicate_pw5"
)

DEFAULT_OUTPUT = (
    DEFAULT_CHECKPOINT_DIR / "validation_threshold_analysis.csv"
)
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved epoch checkpoint on the original "
            "patient-level validation split across probability thresholds."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Folder containing epoch_001.pt ... epoch_010.pt",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path",
    )
    parser.add_argument(
        "--threshold-start", type=float, default=0.10
    )
    parser.add_argument(
        "--threshold-stop", type=float, default=0.90
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.05
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No epoch_*.pt checkpoints found in {checkpoint_dir}"
        )

    # Load the first checkpoint to recover the exact training configuration.
    first_ckpt = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    config = first_ckpt["config"]
    seed = int(first_ckpt.get("seed", config["experiment"]["seed"]))
    padding_mode = first_ckpt.get("padding_mode", "replicate")
    if padding_mode != "replicate":
        raise ValueError(
            f"This script currently expects replicate padding, got {padding_mode!r}"
        )

    # The saved config points to the split CSV used by this experiment.
    split_csv = Path(config["data"]["split_csv"])
    if not split_csv.is_absolute():
        split_csv = PROJECT_ROOT / split_csv
    if not split_csv.exists():
        raise FileNotFoundError(f"Training split CSV not found: {split_csv}")

    image_size = int(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    use_amp = bool(config["training"].get("use_amp", False))

    dataset = BUSUCLMDataset(split="val", split_path=split_csv)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=int(config["model"]["in_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)

    thresholds = np.round(
        np.arange(
            args.threshold_start,
            args.threshold_stop + args.threshold_step / 2,
            args.threshold_step,
        ),
        4,
    ).tolist()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Split CSV: {split_csv}")
    print(f"Seed: {seed}")
    print(f"Validation images: {len(dataset)}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Padding: {padding_mode}")
    print(f"Thresholds: {thresholds}")
    print("Test data will not be loaded or evaluated.")

    rows = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        ckpt_seed = int(checkpoint.get("seed", seed))
        ckpt_padding = checkpoint.get("padding_mode", padding_mode)
        if ckpt_seed != seed or ckpt_padding != padding_mode:
            raise ValueError(
                f"Inconsistent checkpoint metadata in {checkpoint_path.name}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Collect predictions once; calculate all threshold metrics from
        # the same probability maps for a fair within-checkpoint comparison.
        records = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                images, masks = resize_with_padding(
                    images, masks, image_size
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    logits = model(images)

                probabilities = torch.sigmoid(logits.float()).cpu()
                masks = masks.cpu()

                for i, label in enumerate(batch["label"]):
                    records.append(
                        (
                            label,
                            probabilities[i, 0].numpy(),
                            (masks[i, 0].numpy() >= 0.5),
                        )
                    )

        lesion_records = [
            r for r in records if r[0] in ("Benign", "Malignant")
        ]
        normal_records = [r for r in records if r[0] == "Normal"]

        if not lesion_records or not normal_records:
            raise RuntimeError(
                "Validation set must contain both lesion and Normal images."
            )

        for threshold in thresholds:
            lesion_dices = []
            lesion_recalls = []

            for _, prob, target in lesion_records:
                pred = prob >= threshold
                intersection = np.logical_and(pred, target).sum()
                pred_area = pred.sum()
                target_area = target.sum()

                dice_denominator = pred_area + target_area
                dice = (
                    2.0 * intersection / dice_denominator
                    if dice_denominator > 0 else 1.0
                )
                recall = (
                    intersection / target_area
                    if target_area > 0 else float("nan")
                )
                lesion_dices.append(dice)
                lesion_recalls.append(recall)

            normal_fp_flags = []
            normal_fp_areas = []
            for _, prob, _ in normal_records:
                pred = prob >= threshold
                normal_fp_flags.append(bool(pred.any()))
                normal_fp_areas.append(float(pred.mean()))

            row = {
                "checkpoint": checkpoint_path.name,
                "epoch": int(checkpoint.get("epoch", -1)),
                "threshold": threshold,
                "lesion_macro_dice": float(np.mean(lesion_dices)),
                "lesion_macro_pixel_recall": float(np.nanmean(lesion_recalls)),
                "normal_fp_image_rate": float(np.mean(normal_fp_flags)),
                "mean_normal_fp_area_fraction": float(np.mean(normal_fp_areas)),
                "lesion_images": len(lesion_records),
                "normal_images": len(normal_records),
            }
            rows.append(row)

        print(
            f"Evaluated {checkpoint_path.name} "
            f"({len(lesion_records)} lesion, {len(normal_records)} Normal images)"
        )

    fieldnames = [
        "checkpoint",
        "epoch",
        "threshold",
        "lesion_macro_dice",
        "lesion_macro_pixel_recall",
        "normal_fp_image_rate",
        "mean_normal_fp_area_fraction",
        "lesion_images",
        "normal_images",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinished.")
    print(f"Rows written: {len(rows)}")
    print(f"Output: {output_path}")
    print(
        "\nReminder: use this Validation analysis to predefine a selection "
        "rule; do not choose a threshold using Test results."
    )


if __name__ == "__main__":
    main()
