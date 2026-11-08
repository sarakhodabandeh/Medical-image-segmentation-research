import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch", groups=8):
        super().__init__()
        if norm_type == "batch":
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            if out_channels % groups != 0:
                raise ValueError(
                    f"out_channels={out_channels} is not divisible by groups={groups}"
                )
            norm1 = nn.GroupNorm(groups, out_channels)
            norm2 = nn.GroupNorm(groups, out_channels)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetNorm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16,
                 norm_type="batch", groups=8):
        super().__init__()
        c = base_channels
        block = lambda in_c, out_c: DoubleConv(in_c, out_c, norm_type, groups)

        self.enc1 = block(in_channels, c)
        self.enc2 = block(c, c * 2)
        self.enc3 = block(c * 2, c * 4)
        self.enc4 = block(c * 4, c * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = block(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = block(c * 2, c)
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


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

            n = images.size(0)
            total_loss += loss.item() * n
            total_images += n
            predictions = torch.sigmoid(logits) >= 0.5

            for i, label in enumerate(labels):
                pred = predictions[i, 0]
                target = masks[i, 0] >= 0.5
                if label in ("Benign", "Malignant"):
                    intersection = (pred & target).sum().item()
                    pred_n = pred.sum().item()
                    target_n = target.sum().item()
                    denom = pred_n + target_n
                    dice = 2.0 * intersection / denom if denom else 1.0
                    recall = intersection / target_n if target_n else 0.0
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


def norm_stats(model, norm_type, epoch):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            rows.append({
                "norm_type": norm_type,
                "epoch": epoch,
                "layer": name,
                "running_mean_abs_mean": module.running_mean.detach().abs().mean().item(),
                "running_var_mean": module.running_var.detach().mean().item(),
                "running_var_min": module.running_var.detach().min().item(),
                "running_var_max": module.running_var.detach().max().item(),
                "num_batches_tracked": int(module.num_batches_tracked.item()),
            })
        elif isinstance(module, nn.GroupNorm):
            rows.append({
                "norm_type": norm_type,
                "epoch": epoch,
                "layer": name,
                "running_mean_abs_mean": "",
                "running_var_mean": "",
                "running_var_min": "",
                "running_var_max": "",
                "num_batches_tracked": "",
            })
    return rows


def run_one(norm_type, args, config, train_ds, val_ds, device):
    set_seed(args.seed)
    batch_size = args.batch_size
    image_size = config["data"]["image_size"]
    use_amp = bool(config["training"]["use_amp"])

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model = UNetNorm(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        norm_type=norm_type,
        groups=args.groups,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    pos_weight = config["loss"].get("pos_weight", 1.0)
    bce_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    out_dir = resolve_path(args.output_dir) / norm_type
    ckpt_dir = out_dir / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    best_path = out_dir / "best_model.pt"
    if log_path.exists() or best_path.exists() or any(ckpt_dir.iterdir()):
        raise FileExistsError(
            f"Output already contains results: {out_dir}. Use a new --output-dir."
        )

    logs = []
    stats = []
    best_dice = -1.0
    cumulative_steps = 0

    print(f"\n=== Normalization: {norm_type} | batch size: {batch_size} ===")
    print(f"Steps per epoch: {len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = preprocess_batch(
                images, masks, image_size, args.padding
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = (
                    config["loss"]["bce_weight"] * bce_fn(logits, masks)
                    + config["loss"]["dice_weight"] * dice_loss(logits, masks)
                )
            loss.backward()
            optimizer.step()
            n = images.size(0)
            running_loss += loss.item() * n
            seen += n
            epoch_steps += 1
            cumulative_steps += 1

        train_loss = running_loss / max(seen, 1)
        metrics = evaluate(
            model, val_loader, device, image_size, use_amp,
            bce_fn, args.padding, config,
        )
        seconds = time.time() - start
        row = {
            "norm_type": norm_type,
            "batch_size": batch_size,
            "groups": args.groups if norm_type == "group" else "",
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "steps_this_epoch": epoch_steps,
            "optimizer_steps_cumulative": cumulative_steps,
            "training_images_seen_cumulative": len(train_ds) * epoch,
            "epoch_seconds": round(seconds, 2),
        }
        logs.append(row)
        stats.extend(norm_stats(model, norm_type, epoch))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "norm_type": norm_type,
            "groups": args.groups if norm_type == "group" else None,
            "batch_size": batch_size,
            "train_loss": train_loss,
            **metrics,
            "config": config,
            "seed": args.seed,
            "split_csv": str(resolve_path(args.split_csv)),
            "padding_mode": args.padding,
            "optimizer_steps_cumulative": cumulative_steps,
            "training_images_seen_cumulative": len(train_ds) * epoch,
            "checkpoint_type": "normalization_comparison",
        }
        torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if metrics["val_lesion_dice_at_0_5"] > best_dice:
            best_dice = metrics["val_lesion_dice_at_0_5"]
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"Dice={metrics['val_lesion_dice_at_0_5']:.4f} | "
            f"Recall={metrics['val_lesion_pixel_recall_at_0_5']:.4f} | "
            f"Normal_FP={metrics['val_normal_fp_image_rate_at_0_5']:.3f} | "
            f"{seconds:.1f}s"
        )

    def write_csv(path, rows):
        if rows:
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    write_csv(log_path, logs)
    write_csv(out_dir / "normalization_stats_by_epoch.csv", stats)
    print(f"Saved: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Controlled BatchNorm vs GroupNorm U-Net comparison."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument(
        "--split-csv",
        default="data/processed/paired_splits/patient_level_seed_99.csv",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--padding", choices=["replicate", "constant"], default="replicate")
    parser.add_argument("--output-dir", default="results/norm_comparison_seed99")
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = resolve_path(args.split_csv)
    train_ds = BUSUCLMDataset(split="train", split_path=split_path)
    val_ds = BUSUCLMDataset(split="val", split_path=split_path)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Split:", split_path)
    print("Train/Validation:", len(train_ds), "/", len(val_ds))
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("GroupNorm groups:", args.groups)
    print("No test data is loaded. Validation is used only for model selection.")

    run_one("batch", args, config, train_ds, val_ds, device)
    run_one("group", args, config, train_ds, val_ds, device)
    print("\nNormalization comparison completed.")
    print("Upload batch/training_log.csv and group/training_log.csv, plus stats CSVs.")


if __name__ == "__main__":
    main()
