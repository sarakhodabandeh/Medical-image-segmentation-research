import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
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
from src.models.attention_unet import AttentionUNet
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)




def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_batch(images, masks, size, padding_mode):
    if padding_mode == "replicate":
        return resize_with_padding(images, masks, size)
    if padding_mode == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding_mode: {padding_mode}")


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits).flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def calculate_metrics(
    model, loader, device, image_size, use_amp,
    bce_loss_fn, padding_mode, config
):
    model.eval()
    total_loss = 0.0
    total_images = 0
    lesion_dice_sum = 0.0
    lesion_count = 0
    normal_count = 0
    normal_false_positive_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"]
            images, masks = preprocess_batch(
                images, masks, image_size, padding_mode
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = (
                    config["loss"]["bce_weight"] * bce_loss_fn(logits, masks)
                    + config["loss"]["dice_weight"] * dice_loss(logits, masks)
                )

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size
            predictions = torch.sigmoid(logits) >= 0.5

            for i, label in enumerate(labels):
                pred = predictions[i, 0]
                target = masks[i, 0] >= 0.5
                if label in ("Benign", "Malignant"):
                    intersection = (pred & target).sum().item()
                    denominator = pred.sum().item() + target.sum().item()
                    dice = (
                        2.0 * intersection / denominator
                        if denominator > 0 else 1.0
                    )
                    lesion_dice_sum += dice
                    lesion_count += 1
                elif label == "Normal":
                    normal_count += 1
                    if pred.any().item():
                        normal_false_positive_count += 1

    return {
        "loss": total_loss / max(total_images, 1),
        "lesion_dice": lesion_dice_sum / max(lesion_count, 1),
        "normal_false_positive_rate": (
            normal_false_positive_count / normal_count
            if normal_count else 0.0
        ),
        "lesion_images": lesion_count,
        "normal_images": normal_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net using a specified BUS-UCLM split CSV."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["unet", "attention_unet"],
        default="unet",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--padding", choices=["replicate", "constant"], default="replicate"
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = args.seed if args.seed is not None else config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    num_workers = config["data"]["num_workers"]
    use_amp = config["training"]["use_amp"]
    padding_mode = args.padding

    split_csv_path = resolve_project_path(args.split_csv)
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    train_dataset = BUSUCLMDataset(
        split="train", split_path=split_csv_path
    )
    val_dataset = BUSUCLMDataset(
        split="val", split_path=split_csv_path
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_classes = {
        "unet": UNet,
        "attention_unet": AttentionUNet,
    }

    model_class = model_classes[args.model]

    model = model_class(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    print("Model architecture:", args.model)
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    pos_weight_value = config["loss"].get("pos_weight", 1.0)
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device)
    )

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists. Choose a new --output-dir."
        )

    best_val_dice = -1.0
    best_val_normal_fp = None
    epochs_without_improvement = 0
    patience = config["training"]["early_stopping_patience"]

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Split CSV:", split_csv_path)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Epochs:", epochs)
    print("Image size:", image_size)
    print("Batch size:", batch_size)
    print("Padding:", padding_mode)
    print("BCE positive weight:", pos_weight_value)
    print("Checkpoint selection: highest validation lesion macro-Dice at threshold 0.5")
    print("Normal-image FP rate is reported separately, not used as a selection constraint.")

    with log_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch", "train_loss", "val_loss",
            "val_lesion_dice_at_0_5",
            "val_normal_false_positive_rate_at_0_5",
            "val_lesion_images", "val_normal_images", "epoch_seconds",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            seen_images = 0

            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                images, masks = preprocess_batch(
                    images, masks, image_size, padding_mode
                )
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    logits = model(images)
                    bce = bce_loss_fn(logits, masks)
                    dice = dice_loss(logits, masks)
                    loss = (
                        config["loss"]["bce_weight"] * bce
                        + config["loss"]["dice_weight"] * dice
                    )

                loss.backward()
                optimizer.step()
                current_batch_size = images.size(0)
                running_loss += loss.item() * current_batch_size
                seen_images += current_batch_size

            train_loss = running_loss / max(seen_images, 1)
            metrics = calculate_metrics(
                model, val_loader, device, image_size, use_amp,
                bce_loss_fn, padding_mode, config
            )
            epoch_seconds = time.time() - start_time

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"val_lesion_dice@0.5={metrics['lesion_dice']:.4f} | "
                f"normal_FP_rate@0.5={metrics['normal_false_positive_rate']:.4f} | "
                f"time={epoch_seconds:.1f}s"
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "val_lesion_dice_at_0_5": metrics["lesion_dice"],
                "val_normal_false_positive_rate_at_0_5":
                    metrics["normal_false_positive_rate"],
                "val_lesion_images": metrics["lesion_images"],
                "val_normal_images": metrics["normal_images"],
                "epoch_seconds": round(epoch_seconds, 2),
            })
            file.flush()

            # Predefined primary selection rule: maximize validation lesion
            # macro-Dice at fixed threshold 0.5. FP rate is a secondary outcome.
            if metrics["lesion_dice"] > best_val_dice:
                best_val_dice = metrics["lesion_dice"]
                best_val_normal_fp = metrics["normal_false_positive_rate"]
                epochs_without_improvement = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_lesion_dice_at_0_5": best_val_dice,
                    "val_normal_false_positive_rate_at_0_5":
                        best_val_normal_fp,
                    "val_loss": metrics["loss"],
                    "config": config,
                    "seed": seed,
                    "split_csv": str(split_csv_path),
                    "padding_mode": padding_mode,
                    "selection_rule":
                        "maximum validation lesion macro-Dice at threshold 0.5",
                }, checkpoint_path)
                print(
                    f"  Saved best_model.pt | "
                    f"lesion Dice@0.5={best_val_dice:.4f} | "
                    f"normal FP={best_val_normal_fp:.2%}"
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    "Early stopping: validation lesion Dice did not improve "
                    f"for {patience} consecutive epoch(s)."
                )
                break

    print("\nTraining finished.")
    print(f"Best validation lesion Dice@0.5: {best_val_dice:.4f}")
    print(f"Normal false-positive rate@0.5 at selected epoch: {best_val_normal_fp:.2%}")
    print(f"Best model: {checkpoint_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
