import sys
import csv
from pathlib import Path

import torch
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


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)
from src.models.unet import UNet
# Add checkpoints saved after each training epoch.
epoch_dir = ROOT / "results" / "unet_replicate_pw5"

for epoch_path in sorted(epoch_dir.glob("epoch_*.pt")):
    model_name = f"replicate_pw5_{epoch_path.stem}"
    CHECKPOINTS[model_name] = epoch_path

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60, 0.65,
    0.70, 0.75, 0.80,
]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = BUSUCLMDataset(split="val")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    results = []

    for model_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"\nSkipping missing checkpoint: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        config = checkpoint["config"]
        size = config["data"]["image_size"]

        # Use the padding mode recorded in the checkpoint.
        # The fallback matches the two experiments in this project.
        default_padding = (
            "constant"
            if model_name.startswith("constant")
            else "replicate"
        )

        padding_mode = checkpoint.get("padding_mode", default_padding)

        if padding_mode == "constant":
            resize_fn = resize_with_constant_padding
        elif padding_mode == "replicate":
            resize_fn = resize_with_padding
        else:
            raise ValueError(
                f"Unknown padding mode: {padding_mode}"
            )

        model = UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_channels=config["model"]["base_channels"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        records = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                images, masks = resize_fn(images, masks, size)

                probabilities = torch.sigmoid(model(images))

                for i, label in enumerate(batch["label"]):
                    records.append({
                        "label": label,
                        "prob": probabilities[i, 0].cpu(),
                        "target": masks[i, 0].cpu() >= 0.5,
                    })

        print("\n" + "=" * 60)
        print("Model:", model_name)
        print("Checkpoint:", checkpoint_path)
        print("Checkpoint epoch:", checkpoint["epoch"])
        print("Padding:", padding_mode)
        print(
            "Saved best validation Dice:",
            checkpoint.get("best_val_lesion_dice", "not recorded"),
        )
        print("Validation samples:", len(records))

        for threshold in THRESHOLDS:
            lesion_dices = []
            tp = fp = fn = 0
            normal_images = 0
            normal_fp_images = 0
            normal_positive_fractions = []

            for record in records:
                pred = record["prob"] >= threshold
                target = record["target"]
                label = record["label"]

                tp += (pred & target).sum().item()
                fp += (pred & ~target).sum().item()
                fn += (~pred & target).sum().item()

                if label in ("Benign", "Malignant"):
                    intersection = (pred & target).sum().item()
                    denominator = pred.sum().item() + target.sum().item()

                    dice = (
                        2 * intersection / denominator
                        if denominator > 0
                        else 1.0
                    )
                    lesion_dices.append(dice)

                elif label == "Normal":
                    normal_images += 1
                    normal_positive_fractions.append(
                        pred.float().mean().item()
                    )

                    if pred.any().item():
                        normal_fp_images += 1

            mean_dice = (
                sum(lesion_dices) / len(lesion_dices)
                if lesion_dices else 0.0
            )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            pixel_dice = 2 * tp / max(2 * tp + fp + fn, 1)
            normal_fp_rate = (
                normal_fp_images / max(normal_images, 1)
            )
            mean_normal_fraction = (
                sum(normal_positive_fractions)
                / max(len(normal_positive_fractions), 1)
            )

            row = {
                "model": model_name,
                "padding": padding_mode,
                "epoch": checkpoint["epoch"],
                "threshold": threshold,
                "mean_lesion_dice": mean_dice,
                "pixel_dice": pixel_dice,
                "pixel_precision": precision,
                "pixel_recall": recall,
                "normal_fp_rate": normal_fp_rate,
                "mean_normal_positive_fraction": mean_normal_fraction,
            }
            results.append(row)

            print(f"\nThreshold: {threshold:.2f}")
            print(f"Mean lesion Dice: {mean_dice:.4f}")
            print(f"Pixel Dice: {pixel_dice:.4f}")
            print(f"Pixel precision: {precision:.4f}")
            print(f"Pixel recall: {recall:.4f}")
            print(f"Normal image FP rate: {normal_fp_rate:.4f}")
            print(
                "Mean normal positive fraction:",
                f"{mean_normal_fraction:.4f}",
            )

    if results:
        output_path = (
                ROOT / "results" / "checkpoint_threshold_comparison.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

        print("\nComparison CSV saved to:", output_path)
    else:
        print("\nNo checkpoints were evaluated.")


if __name__ == "__main__":
    main()
