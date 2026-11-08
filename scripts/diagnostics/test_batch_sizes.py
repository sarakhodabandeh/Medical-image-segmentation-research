import argparse
import gc
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

    raise RuntimeError("Could not locate project root.")


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


def preprocess(images, masks, size, padding):
    if padding == "replicate":
        return resize_with_padding(images, masks, size)
    if padding == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding mode: {padding}")


def main():
    parser = argparse.ArgumentParser(
        description="Check GPU memory for candidate U-Net batch sizes without training."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument(
        "--split-csv",
        default="data/processed/paired_splits/patient_level_seed_99.csv",
    )
    parser.add_argument("--padding", choices=["replicate", "constant"], default="replicate")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[4, 6, 8, 10])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test is intended for the RTX GPU.")

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    dataset = BUSUCLMDataset(split="train", split_path=resolve_path(args.split_csv))
    image_size = config["data"]["image_size"]
    use_amp = bool(config["training"]["use_amp"])

    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", len(dataset), "training images")
    print("Image size:", image_size)
    print("AMP:", use_amp)
    print("This is a one-step memory test only; no optimizer step or checkpoint is saved.\n")

    for batch_size in args.batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        model = UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_channels=config["model"]["base_channels"],
        ).to(device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        pos_weight = config["loss"].get("pos_weight", 1.0)
        bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        try:
            batch = next(iter(loader))
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = preprocess(images, masks, image_size, args.padding)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):
                logits = model(images)
                probs = torch.sigmoid(logits).flatten(start_dim=1)
                targets = masks.flatten(start_dim=1)
                intersection = (probs * targets).sum(dim=1)
                denominator = probs.sum(dim=1) + targets.sum(dim=1)
                dice_loss = 1.0 - (
                    (2.0 * intersection + 1.0) / (denominator + 1.0)
                ).mean()
                loss = (
                    config["loss"]["bce_weight"] * bce_fn(logits, masks)
                    + config["loss"]["dice_weight"] * dice_loss
                )

            loss.backward()
            torch.cuda.synchronize()

            allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)

            print(
                f"Batch {batch_size:>2}: PASS | "
                f"peak allocated={allocated:.2f} GB | "
                f"peak reserved={reserved:.2f} GB | "
                f"free after step={free_gb:.2f} GB"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size:>2}: OUT OF MEMORY")
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"Batch {batch_size:>2}: OUT OF MEMORY")
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            del model
            del optimizer
            if "images" in locals():
                del images
            if "masks" in locals():
                del masks
            if "logits" in locals():
                del logits
            if "loss" in locals():
                del loss
            gc.collect()
            torch.cuda.empty_cache()

    print("\nMemory test finished. No training weights or project files were changed.")


if __name__ == "__main__":
    main()
