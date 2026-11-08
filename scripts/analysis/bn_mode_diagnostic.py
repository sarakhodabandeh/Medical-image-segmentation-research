import argparse
import copy
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml


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


def load_checkpoint(path, device):
    # Checkpoints are produced by this project's own training script.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Compare U-Net BatchNorm eval-mode and train-mode predictions."
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
    parser.add_argument("--output-dir", default="results/bn_mode_diagnostic")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, nargs="*", default=list(range(1, 11)))
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    split_path = resolve_path(args.split_csv)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image_size = config["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BUSUCLMDataset(split="val", split_path=split_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    per_image_rows = []
    bn_rows = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation images: {len(dataset)}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    for epoch in args.epochs:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if not checkpoint_path.exists():
            print(f"WARNING: missing checkpoint, skipping: {checkpoint_path}")
            continue

        checkpoint = load_checkpoint(checkpoint_path, device)
        ckpt_config = checkpoint.get("config", config)
        model = UNet(
            in_channels=ckpt_config["model"]["in_channels"],
            out_channels=ckpt_config["model"]["out_channels"],
            base_channels=ckpt_config["model"]["base_channels"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Two identical copies ensure eval-mode inference cannot be affected
        # by BatchNorm running-stat updates during train-mode inference.
        eval_model = copy.deepcopy(model)
        train_model = copy.deepcopy(model)
        eval_model.eval()
        train_model.train()

        padding_mode = checkpoint.get("padding_mode", "replicate")
        bn_before = {}
        for name, module in train_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_before[name] = {
                    "mean": module.running_mean.detach().clone(),
                    "var": module.running_var.detach().clone(),
                    "batches": int(module.num_batches_tracked.item()),
                }

        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                images, masks = preprocess_batch(
                    images, masks, image_size, padding_mode
                )

                # Deliberately disable AMP so the comparison isolates mode.
                eval_logits = eval_model(images)
                train_logits = train_model(images)

                eval_prob = torch.sigmoid(eval_logits).float()
                train_prob = torch.sigmoid(train_logits).float()

                for i, label in enumerate(batch["label"]):
                    pe = eval_prob[i, 0]
                    pt = train_prob[i, 0]
                    per_image_rows.append({
                        "epoch": epoch,
                        "filename": batch["filename"][i],
                        "patient_id": batch["patient_id"][i],
                        "label": label,
                        "eval_prob_mean": pe.mean().item(),
                        "train_prob_mean": pt.mean().item(),
                        "eval_prob_q95": torch.quantile(pe.flatten(), 0.95).item(),
                        "train_prob_q95": torch.quantile(pt.flatten(), 0.95).item(),
                        "eval_prob_max": pe.max().item(),
                        "train_prob_max": pt.max().item(),
                        "eval_area_fraction_at_0_5": (pe >= 0.5).float().mean().item(),
                        "train_area_fraction_at_0_5": (pt >= 0.5).float().mean().item(),
                        "eval_any_positive_at_0_5": int((pe >= 0.5).any().item()),
                        "train_any_positive_at_0_5": int((pt >= 0.5).any().item()),
                        "probability_map_mae": (pe - pt).abs().mean().item(),
                        "threshold_disagreement_fraction": (
                            ((pe >= 0.5) != (pt >= 0.5)).float().mean().item()
                        ),
                    })

        # Record per-layer BN running-stat changes caused by train-mode passes.
        for name, module in train_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                before = bn_before[name]
                after_mean = module.running_mean.detach()
                after_var = module.running_var.detach()
                bn_rows.append({
                    "epoch": epoch,
                    "layer": name,
                    "num_features": module.num_features,
                    "num_batches_tracked_before": before["batches"],
                    "num_batches_tracked_after": int(module.num_batches_tracked.item()),
                    "running_mean_abs_before": before["mean"].abs().mean().item(),
                    "running_mean_abs_after": after_mean.abs().mean().item(),
                    "running_mean_change_mean_abs": (
                        after_mean - before["mean"]
                    ).abs().mean().item(),
                    "running_var_mean_before": before["var"].mean().item(),
                    "running_var_mean_after": after_var.mean().item(),
                    "running_var_change_mean_abs": (
                        after_var - before["var"]
                    ).abs().mean().item(),
                })

        print(f"Completed epoch {epoch:03d}")

    per_image_path = output_dir / "bn_mode_per_image.csv"
    bn_path = output_dir / "bn_running_stats_by_layer.csv"

    if per_image_rows:
        with per_image_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    if bn_rows:
        with bn_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(bn_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bn_rows)

    print("\nDiagnostic complete.")
    print(f"Per-image results: {per_image_path}")
    print(f"BatchNorm layer results: {bn_path}")


if __name__ == "__main__":
    main()
