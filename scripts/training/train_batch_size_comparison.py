import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
def resolve_path(value):
    path = Path(value)
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
    raise ValueError(f"Unknown padding mode: {padding_mode}")


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits).flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def evaluate(model, loader, device, image_size, use_amp,
             bce_loss_fn, padding_mode, config):
    model.eval()
    total_loss = 0.0
    total_images = 0
    lesion_dice_sum = 0.0
    lesion_recall_sum = 0.0
    lesion_count = 0
    normal_count = 0
    normal_fp_count = 0
    normal_fp_area_sum = 0.0

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

            batch_n = images.size(0)
            total_loss += loss.item() * batch_n
            total_images += batch_n
            predictions = torch.sigmoid(logits) >= 0.5

            for i, label in enumerate(labels):
                pred = predictions[i, 0]
                target = masks[i, 0] >= 0.5
                if label in ("Benign", "Malignant"):
                    intersection = (pred & target).sum().item()
                    pred_count = pred.sum().item()
                    target_count = target.sum().item()
                    denom = pred_count + target_count
                    dice = (2.0 * intersection / denom) if denom else 1.0
                    recall = intersection / target_count if target_count else 0.0
                    lesion_dice_sum += dice
                    lesion_recall_sum += recall
                    lesion_count += 1
                elif label == "Normal":
                    normal_count += 1
                    normal_fp_area_sum += pred.float().mean().item()
                    if pred.any().item():
                        normal_fp_count += 1

    return {
        "val_loss": total_loss / max(total_images, 1),
        "val_lesion_dice_at_0_5": lesion_dice_sum / max(lesion_count, 1),
        "val_lesion_pixel_recall_at_0_5": lesion_recall_sum / max(lesion_count, 1),
        "val_normal_fp_image_rate_at_0_5": (
            normal_fp_count / normal_count if normal_count else 0.0
        ),
        "val_normal_mean_fp_area_fraction_at_0_5": (
            normal_fp_area_sum / normal_count if normal_count else 0.0
        ),
        "val_lesion_images": lesion_count,
        "val_normal_images": normal_count,
    }


def collect_bn_stats(model, batch_size, epoch):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            rows.append({
                "batch_size": batch_size,
                "epoch": epoch,
                "layer": name,
                "num_features": module.num_features,
                "running_mean_abs_mean": module.running_mean.detach().abs().mean().item(),
                "running_var_mean": module.running_var.detach().mean().item(),
                "running_var_min": module.running_var.detach().min().item(),
                "running_var_max": module.running_var.detach().max().item(),
                "num_batches_tracked": int(module.num_batches_tracked.item()),
            })
    return rows


def run_experiment(batch_size, args, config, train_dataset, val_dataset, device):
    seed = args.seed
    set_seed(seed)

    image_size = config["data"]["image_size"]
    use_amp = bool(config["training"]["use_amp"])
    padding_mode = args.padding

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    pos_weight = config["loss"].get("pos_weight", 1.0)
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    out_dir = resolve_path(args.output_dir) / f"batch_{batch_size}"
    ckpt_dir = out_dir / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    best_path = out_dir / "best_model.pt"
    if log_path.exists() or best_path.exists() or any(ckpt_dir.iterdir()):
        raise FileExistsError(
            f"Output already contains results: {out_dir}. Choose a new --output-dir."
        )

    log_rows = []
    bn_rows = []
    best_dice = -1.0
    optimizer_steps_total = 0

    print(f"\n=== Batch size {batch_size} ===")
    print(f"Training images: {len(train_dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Epochs: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        seen_images = 0
        epoch_steps = 0

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
                dloss = dice_loss(logits, masks)
                loss = (
                    config["loss"]["bce_weight"] * bce
                    + config["loss"]["dice_weight"] * dloss
                )

            loss.backward()
            optimizer.step()
            n = images.size(0)
            running_loss += loss.item() * n
            seen_images += n
            epoch_steps += 1
            optimizer_steps_total += 1

        train_loss = running_loss / max(seen_images, 1)
        metrics = evaluate(
            model, val_loader, device, image_size, use_amp,
            bce_loss_fn, padding_mode, config
        )
        seconds = time.time() - start

        row = {
            "batch_size": batch_size,
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "steps_this_epoch": epoch_steps,
            "optimizer_steps_cumulative": optimizer_steps_total,
            "training_images_seen_cumulative": len(train_dataset) * epoch,
            "epoch_seconds": round(seconds, 2),
        }
        log_rows.append(row)
        bn_rows.extend(collect_bn_stats(model, batch_size, epoch))

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "batch_size": batch_size,
            "train_loss": train_loss,
            **metrics,
            "config": config,
            "seed": seed,
            "split_csv": str(resolve_path(args.split_csv)),
            "padding_mode": padding_mode,
            "optimizer_steps_cumulative": optimizer_steps_total,
            "training_images_seen_cumulative": len(train_dataset) * epoch,
            "checkpoint_type": "batch_size_comparison",
        }, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if metrics["val_lesion_dice_at_0_5"] > best_dice:
            best_dice = metrics["val_lesion_dice_at_0_5"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "batch_size": batch_size,
                **metrics,
                "config": config,
                "seed": seed,
                "split_csv": str(resolve_path(args.split_csv)),
                "padding_mode": padding_mode,
                "selection_rule": "maximum validation lesion macro-Dice at threshold 0.5",
            }, best_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"lesion_Dice={metrics['val_lesion_dice_at_0_5']:.4f} | "
            f"lesion_recall={metrics['val_lesion_pixel_recall_at_0_5']:.4f} | "
            f"Normal_FP={metrics['val_normal_fp_image_rate_at_0_5']:.3f} | "
            f"steps={epoch_steps} | {seconds:.1f}s"
        )

    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    bn_path = out_dir / "batchnorm_stats_by_epoch.csv"
    with bn_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(bn_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bn_rows)

    print(f"Saved: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Controlled patient-level U-Net comparison across batch sizes."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument(
        "--split-csv",
        default="data/processed/paired_splits/patient_level_seed_99.csv",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[4, 16, 32])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--padding", choices=["replicate", "constant"], default="replicate")
    parser.add_argument("--output-dir", default="results/batch_size_comparison_seed99")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = resolve_path(args.split_csv)
    train_dataset = BUSUCLMDataset(split="train", split_path=split_path)
    val_dataset = BUSUCLMDataset(split="val", split_path=split_path)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Split:", split_path)
    print("Train/Validation:", len(train_dataset), "/", len(val_dataset))
    print("Fixed epochs:", args.epochs)
    print("Batch sizes:", args.batch_sizes)
    print("Learning rate:", config["training"]["learning_rate"])
    print("Loss weights:", config["loss"]["bce_weight"], config["loss"]["dice_weight"])
    print("AMP:", config["training"]["use_amp"])
    print("Padding:", args.padding)
    print("\nDesign: same image exposure (same epochs), report optimizer-step counts.")
    print("No test set is loaded or evaluated.")

    for batch_size in args.batch_sizes:
        run_experiment(
            batch_size, args, config, train_dataset, val_dataset, device
        )

    print("\nAll batch-size experiments finished.")
    print("Upload each batch_*/training_log.csv and batchnorm_stats_by_epoch.csv")


if __name__ == "__main__":
    main()
